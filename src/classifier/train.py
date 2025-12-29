# classifier/train.py
import torch
import torch.nn.functional as F


def train(model, x_train, y_train, batch_size, iterations, learning_rate, scheduler_step_size=-1, scheduler_gamma=0.1, device=None):
  optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
  lossi = []

  model.train()
  for i in range(iterations):
    optimizer.zero_grad()

    idx = torch.randint(0, len(x_train), (batch_size,), device=device)
    x = x_train[idx].view(-1, 1, 28, 28)
    x += torch.randn_like(x) * 0.1 # add small amount of random noise to account for noisy backgrounds in generated samples
    y = F.one_hot(y_train[idx].view(-1), 10)
    
    pred = model(x)
    loss = ((y - pred) ** 2).mean()

    loss.backward()
    optimizer.step()
    if scheduler_step_size > 0:
      scheduler.step()

    lossi.append(loss.item())

    if i%10==0:
      print(f"{i=} {loss.item()}")

  return lossi
