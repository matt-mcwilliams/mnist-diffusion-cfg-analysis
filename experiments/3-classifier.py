import argparse
import torch
from diffusion.model import MNISTUNet
from diffusion.sample import generate_samples
from classifier.model import Classifier
import matplotlib.pyplot as plt
import math
import numpy as np
from datetime import datetime
current_datetime = datetime.now()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

T = 1000

num_samples = 200

def cosine_beta_schedule(T, device, s=0.008):
    t = torch.linspace(0, T, T + 1)
    f = torch.cos(((t / T) + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return betas.clamp(1e-5, 0.999).to(device=device)

betas = cosine_beta_schedule(T, device)

guidance_variables = [ 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 
    2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0 ]


if __name__ == '__main__':

    torch.manual_seed(67)

    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('classifier_path')
    parser.add_argument('sample_path')
    args = parser.parse_args()

    model = MNISTUNet(betas, T=T).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    classifier = Classifier().to(device)
    classifier.load_state_dict(torch.load(args.classifier_path, map_location=device))
    classifier.eval()

    text_log_path = f'{args.sample_path}/_log.txt'

    def write(*text):
        with open(text_log_path, 'a', encoding='utf-8') as f:
            f.write(' '.join(str(t) for t in text) + '\n')

    write('\n\n', current_datetime, 'Classifier probability statistics by guidance scale\n')

    correct_probs_over_s = []

    for s in guidance_variables:

        labels, samples = generate_samples(
            model,
            num_generations=num_samples,
            guidance_scale=s,
            random_seed=1,
            device=device
        )

        with torch.no_grad():
            probs = classifier(samples) # classifier ends with softmax
            batch_idx = torch.arange(probs.size(0))
            correct_probs = probs[batch_idx, labels]
        
        labels = labels.detach().cpu()
        samples = samples.detach().cpu()
        probs = probs.detach().cpu()
        correct_probs = correct_probs.detach().cpu()

        correct_probs_over_s.append(correct_probs)

        for i, (label, sample) in enumerate(zip(labels[:10], samples[:10])):
          img = sample.squeeze().cpu().numpy()
          plt.imshow(img, cmap='gray')
          plt.axis('off')

          fig = plt.gcf()
          fig.text(0.5, -0.05, f'digit={label}; p={correct_probs[i]:.4f}', ha='center', fontsize=12)

          plt.savefig(f'{args.sample_path}/sample-s-{s}-{i+1}.png', bbox_inches='tight')
          plt.close()

        # general statistics

        mean = correct_probs.mean().item()
        std = correct_probs.std().item()
        median = correct_probs.median().item()
        pmin = correct_probs.min().item()
        pmax = correct_probs.max().item()

        write(f'\nGuidance: {s}')
        write(f'Probability assigned to correct label:')
        write(f"mean={mean:.4f} | "
              f"std={std:.4f} | "
              f"median={median:.4f} | "
              f"min={pmin:.4f} | "
              f"max={pmax:.4f}")
        

        # extreme probs

        p_01 = (correct_probs < 0.01).sum().item() / num_samples
        p_01_1 = ((correct_probs >= 0.01) &( correct_probs < 0.1)).sum().item() / num_samples
        p_1_9 = ((correct_probs >= 0.1) & (correct_probs < 0.9)).sum().item() / num_samples
        p_9 = (correct_probs >= 0.9).sum().item() / num_samples

        write(f'< 0.01: {p_01:.4f} | '
              f'0.01-0.1: {p_01_1:.4f} | '
              f'0.1-0.9: {p_1_9:.4f} | '
              f'> 0.9: {p_9:.4f} ')

        # top-1 accuracy

        predicted_digits = probs.argmax(dim=1)
        top_1_accuracy = (predicted_digits == labels).sum().item() / num_samples
        write(f'top-1 accuracy: {top_1_accuracy}')

        # intra-class pixel variance

        samples_cpu = samples.detach().cpu()  # (N, 1, 28, 28)
        labels_cpu = labels.detach().cpu()

        class_variances = {}

        for c in labels_cpu.unique():
            idx = (labels_cpu == c)
            class_samples = samples_cpu[idx]

            if class_samples.size(0) < 2:
                continue  # variance undefined for single sample

            # pixel-wise variance, then mean over pixels
            pixel_var = class_samples.var(dim=0, unbiased=False)
            mean_pixel_var = pixel_var.mean().item()

            class_variances[c.item()] = mean_pixel_var

        write(class_variances)
        write(class_variances.values())


        # histogram

        plt.hist(
            correct_probs,
            range=(0.0, 1.0),
            density=True,
            bins=20,
            histtype="step",
            color="black"
        )
        plt.savefig(f"{args.sample_path}/hist-s-{s}.png", bbox_inches="tight")
        plt.close()
    
    # scatter plot

    xs = []
    ys = []

    for s, probs in zip(guidance_variables, correct_probs_over_s):
        probs_np = probs.detach().cpu().numpy()
        xs.extend([s] * len(probs_np))
        ys.extend(probs_np)

    plt.figure()
    plt.scatter(xs, ys, alpha=0.6)
    plt.xlabel("Guidance strength")
    plt.ylabel("Classifier probability (correct label)")
    plt.ylim(0.0, 1.0)

    plt.savefig(f"{args.sample_path}/scatter.png",
                bbox_inches="tight")
    plt.close()
