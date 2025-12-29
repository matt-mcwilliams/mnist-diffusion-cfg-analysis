import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalTimeEmbedding(nn.Module): # AI Generated - learn implementation later
    def __init__(self, dim, max_period=10000):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.max_period = max_period

    def forward(self, t):
        # t: (B,) floats (can pass t.float()/T)
        t = t.to(dtype=torch.float32)
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, device=device) / half
        )
        args = t[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
        return emb


class ResBlock(nn.Module):
  def __init__(self, in_channels, out_channels, time_dim, label_dim):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding='same')
    self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding='same')

    self.time_map = nn.Linear(time_dim, out_channels)
    self.label_map = nn.Linear(label_dim, out_channels)
    self.act = nn.ReLU()

    self.batchnorm = nn.BatchNorm2d(out_channels)
    self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    self.in_channels = in_channels
    self.out_channels = out_channels

  def forward(self, x, time_emb, label_emb):
    '''
    x : batch x in_channels x width x height
    time_emb : batch x time_dim
    y_emb : batch x label_dim

    out : batch x out_channels x width x height
    '''
    h1 = self.act(self.conv1(x))
    h2 = self.batchnorm(h1) + self.time_map(time_emb)[...,None,None] + self.label_map(label_emb)[...,None,None]
    h3 = self.act(self.conv2(h2))
    x_skip = self.skip(x)
    return h3 + x_skip


class MNISTUNet(nn.Module):
  def __init__(self, betas, base_channels=16, time_dim=32, label_dim=11, T=1000):
    super().__init__()

    self.timeEmbed = SinusoidalTimeEmbedding(time_dim)
    self.T = T
    self.register_buffer('betas', betas)

    self.down1 = ResBlock(1, base_channels, time_dim, label_dim)
    self.down2 = ResBlock(base_channels, base_channels*2, time_dim, label_dim)
    self.downscale = nn.MaxPool2d(2)

    self.connector = ResBlock(base_channels*2, base_channels*4, time_dim, label_dim)

    self.upscale1 = nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=2, stride=2)
    self.up1 = ResBlock(base_channels*4, base_channels*2, time_dim, label_dim)

    self.upscale2 = nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=2, stride=2)
    self.up2 = ResBlock(base_channels*2, base_channels, time_dim, label_dim)

    self.output = nn.Conv2d(base_channels, 1, 1)

  def forward(self, x, t, y):
    # y=-1 indicates no class
    time_emb = self.timeEmbed(t)
    label_emb = F.one_hot(y+1, 11).to(dtype=torch.float32)
    d1 = self.down1(x, time_emb, label_emb)

    d2 = self.downscale(d1)
    d2 = self.down2(d2, time_emb, label_emb)

    connection = self.downscale(d2)
    connection = self.connector(connection, time_emb, label_emb)

    u2 = self.upscale1(connection)
    u2 = self.up1(torch.cat((u2, d2), dim=1), time_emb, label_emb)

    u1 = self.upscale2(u2)
    u1 = self.up2(torch.cat((u1, d1), dim=1), time_emb, label_emb)

    return self.output(u1)
