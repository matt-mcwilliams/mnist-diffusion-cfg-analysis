import argparse
import torch
from diffusion.model import MNISTUNet
from diffusion.sample import generate_samples
import matplotlib.pyplot as plt
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

T = 1000

def cosine_beta_schedule(T, s=0.008, device=device):
    t = torch.linspace(0, T, T + 1)
    f = torch.cos(((t / T) + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return betas.clamp(1e-5, 0.999).to(device=device)

betas = cosine_beta_schedule(T)

guidance_variables = [ 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 
    2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0 ]

if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        
        parser.add_argument('model_path')
        parser.add_argument('digit')
        parser.add_argument('sample_path')
        
        args = parser.parse_args()
        
        model = MNISTUNet(betas, T=T)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        
        for s in guidance_variables:
          labels, samples = generate_samples(model, num_generations=3, digit=int(args.digit), guidance_scale=s, random_seed=1, device=device)
          for i, sample in enumerate(samples):
                  img = sample.squeeze().cpu().detach().numpy()
                  plt.imshow(img, cmap='gray')
                  plt.axis('off')
                  plt.savefig(f'{args.sample_path}/s-{s}-{i+1}.png', bbox_inches='tight')
                  plt.close()
        