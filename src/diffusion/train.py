# diffusion/train.py
import torch


def train(model, x_train, y_train, batch_size, iterations, learning_rate, scheduler_step_size=-1, scheduler_gamma=0.1, device=None):
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
        lossi = []
        
        alphas = 1 - model.betas
        alpha_bars = torch.cumprod(alphas, dim=0).to(device=device)

        model.train()
        for i in range(iterations):
                optimizer.zero_grad()

                idx = torch.randint(0, len(x_train), (batch_size,), device=device)
                x_0 = x_train[idx].view(-1, 1, 28, 28)
                y = y_train[idx].view(-1)

                y_null_mask = torch.ceil(torch.randint_like(y,0,10,dtype=torch.float32)/10).to(dtype=torch.int64)
                y = y_null_mask * (y+1) - 1 # make 10% of labels -1 (null).

                t = torch.randint(1, model.T, (batch_size,), device=device, dtype=torch.long)
                alpha_bar_t = alpha_bars[t.to(dtype=torch.int)].view(-1, 1, 1, 1)

                eps = torch.randn_like(x_0)

                x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * eps

                eps_pred = model(x_t, t, y)

                loss = ((eps - eps_pred)**2).mean()
                loss.backward()
                optimizer.step()
                if scheduler_step_size > 0:
                        scheduler.step()

                lossi.append(loss.item())

                if i%10==0:
                        print(f"{i=} {loss.item()}")

        return lossi
