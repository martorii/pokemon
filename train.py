from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from IPython.display import HTML
from model import ContextUnet


if __name__ == '__main__':
    # diffusion hyperparameters
    timesteps = 500
    beta1 = 1e-4
    beta2 = 0.02

    # network hyperparameters
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu:0")
    n_feat = 64 # 64 hidden dimension feature
    n_cfeat = 18 # context vector is of size 18
    height = 16 # 96x96 image
    save_dir = './weights/'

    # training hyperparameters
    batch_size = 16
    n_epoch = 1
    lrate = 1e-3


    # construct DDPM noise schedule
    b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
    a_t = 1 - b_t
    ab_t = torch.cumsum(a_t.log(), dim=0).exp()
    ab_t[0] = 1

    # construct model
    nn_model = ContextUnet(in_channels=3, n_feat=n_feat, n_cfeat=n_cfeat, height=height).to(device)


    def denoise_add_noise(x, t, pred_noise, z=None):
        if z is None:
            z = torch.randn_like(x)
        noise = b_t.sqrt()[t] * z
        mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
        return mean + noise


    # define sampling function for DDIM
    # removes the noise using ddim
    def denoise_ddim(x, t, t_prev, pred_noise):
        ab = ab_t[t]
        ab_prev = ab_t[t_prev]

        x0_pred = ab_prev.sqrt() / ab.sqrt() * (x - (1 - ab).sqrt() * pred_noise)
        dir_xt = (1 - ab_prev).sqrt() * pred_noise

        return x0_pred + dir_xt


    from dataloader import CustomDataset
    dataset = CustomDataset(
        annotations_file='data/annotation_file.pkl',
        img_dir='data',
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,))
            ]
        )
    )
    # load dataset and construct optimizer
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    optim = torch.optim.Adam(nn_model.parameters(), lr=lrate)

    # helper function: perturbs an image to a specified noise level
    def perturb_input(x, t, noise):
        return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise


    # training without context code
    import os

    # set into train mode
    nn_model.train()

    # Start loop for each epoch
    for ep in range(n_epoch):
        print(f'epoch {ep}')
        losses = []

        # linearly decay learning rate
        optim.param_groups[0]['lr'] = lrate * (1 - ep / n_epoch)

        # Load pbar for batch
        pbar = tqdm(dataloader, mininterval=2)
        for x, _ in pbar:  # x: images, _: context vector
            # Clean gradients
            optim.zero_grad()
            # Attach to device
            x = x.to(device)

            # perturb data
            noise = torch.randn_like(x)
            t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device)
            x_pert = perturb_input(x, t, noise)

            # use network to recover noise
            pred_noise = nn_model(x_pert, t / timesteps)

            # loss is mean squared error between the predicted and true noise
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()

            optim.step()
            losses.append(loss.item())
            del loss

        print(f"Loss at epoch {ep} = {np.mean(losses)}")

        # save model periodically
        if ep % 4 == 0 or ep == int(n_epoch - 1):
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            torch.save(nn_model.state_dict(), save_dir + f"model_{ep}.pth")
            print('saved model at ' + save_dir + f"model_{ep}.pth")
