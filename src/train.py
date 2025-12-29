import argparse
from diffusion.model import MNISTUNet
from diffusion.train import train
from keras.datasets import mnist
import torch
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

if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        
        parser.add_argument('model_path')
        parser.add_argument('--load', '-l')
        
        args = parser.parse_args()
        
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = torch.tensor(x_train / 127.5, dtype=torch.float32, device=device) - 1
        y_train = torch.tensor(y_train, dtype=torch.int64, device=device)
        
        model = MNISTUNet(betas, T=T)
        if args.load:
                model.load_state_dict(torch.load(args.load))
        model = model.to(device=device)

        lossi = train(model, x_train, y_train, batch_size=128, 
                      iterations=4000, learning_rate=1e-3, device=device)
        
        torch.save(model.state_dict(), args.model_path)