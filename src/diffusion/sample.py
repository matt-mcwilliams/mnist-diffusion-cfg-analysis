import torch

def generate_samples(model, num_generations=1, guidance_scale=2, digit=-1, random_seed=None, device=None):
        
  assert digit % 1 == 0
  assert digit >= -1 and digit <= 9
  
  alphas = 1 - model.betas
  alpha_bars = torch.cumprod(alphas, dim=0).to(device=device)

  generator = (torch.Generator(device=device).manual_seed(random_seed)) if random_seed else (torch.Generator(device=device))


  with torch.no_grad():
    x_t = torch.randn(num_generations,1,28,28, device=device, generator=generator)
    y = torch.randint(0,10,(num_generations,),dtype=torch.int64,generator=generator, device=device) \
      if digit==-1 else torch.full((num_generations,),digit,dtype=torch.int64,device=device)
    for t in reversed(range(1,model.T)):
      sigma = torch.sqrt(model.betas[t])
      z = torch.randn(x_t.size(), device=device, generator=generator) if t > 1 else torch.zeros_like(x_t, device=device)

      eps_y = model(x_t, torch.tensor([t], dtype=torch.long, device=device), y)
      eps_null = model(x_t, torch.tensor([t], dtype=torch.long, device=device), -torch.ones_like(y, dtype=torch.int64, device=device))

      eps = (1 + guidance_scale) * eps_y - guidance_scale * eps_null

      x_t = (1 / torch.sqrt(alphas[t])) * (x_t - (((1 - alphas[t]) / torch.sqrt(1 - alpha_bars[t])) * eps)) + sigma * z
  
  return y, x_t