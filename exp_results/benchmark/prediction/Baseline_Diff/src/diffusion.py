"""
This file provides a simple, self-contained implementation of DDIM (with DDPM as a special case).
"""
from dataclasses import dataclass
import math

import torch
import torch.nn as nn

from src.blocks import unsqueeze_to
from src.utils import Hilburn_Loss, avg_fss

from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class DiffusionModelConfig:

    num_timesteps: int
    target_type: str = "pred_eps"
    noise_schedule_type: str = "cosine"
    loss_type: str = "l2"
    gamma_type: str = "ddim"

    def __post_init__(self):
        assert self.num_timesteps > 0
        assert self.target_type in ("pred_x_0", "pred_eps", "pred_v")
        assert self.noise_schedule_type in ("linear", "cosine")
        assert self.loss_type in ("l1", "l2", "Hilburn_Loss")
        assert self.gamma_type in ("ddim", "ddpm")


class DiffusionModel(nn.Module):

    def __init__(
        self,
        input_shape: tuple[int, ...],
        nn_module: nn.Module,
        config: DiffusionModelConfig,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.nn_module = nn_module
        self.num_timesteps = config.num_timesteps
        self.target_type = config.target_type
        self.gamma_type = config.gamma_type
        self.noise_schedule_type = config.noise_schedule_type
        self.loss_type = config.loss_type

        # Input shape must be either (c,) or (c, h, w) or (c, t, h, w)
        assert len(input_shape) in (1, 3, 4)

        # Construct the noise schedule  # Control noise degree
        if self.noise_schedule_type == "linear":
            beta_t = torch.linspace(1e-4, 2e-2, self.num_timesteps + 1)
            alpha_t = torch.cumprod(1 - beta_t, dim=0) ** 0.5
        elif self.noise_schedule_type == "cosine":
            linspace = torch.linspace(0, 1, self.num_timesteps + 1)
            f_t = torch.cos((linspace + 0.008) / (1 + 0.008) * math.pi / 2) ** 2
            bar_alpha_t = f_t / f_t[0]
            beta_t = torch.zeros_like(bar_alpha_t)
            beta_t[1:] = (1 - (bar_alpha_t[1:] / bar_alpha_t[:-1])).clamp(min=0, max=0.999)
            alpha_t = torch.cumprod(1 - beta_t, dim=0) ** 0.5
        else:
            raise AssertionError(f"Invalid {self.noise_schedule_type=}.")

        # These tensors are shape (num_timesteps + 1, *self.input_shape)
        # For example, 2D: (num_timesteps + 1, 1, 1, 1)
        #              1D: (num_timesteps + 1, 1)
        alpha_t = unsqueeze_to(alpha_t, len(self.input_shape) + 1)
        sigma_t = (1 - alpha_t ** 2).clamp(min=0) ** 0.5
        self.register_buffer("alpha_t", alpha_t)
        self.register_buffer("sigma_t", sigma_t)

    def loss(self, x: torch.Tensor, cond, lead_time, gf_sigmat=0):  # 
        """
        Returns
        -------
        loss: (bsz, *input_shape)
        """
        bsz, *_ = x.shape  # bsz = batch size
        t_sample = torch.randint(1, self.num_timesteps + 1, size=(bsz,), device=x.device)  # Random time step
        lead_time = torch.full((bsz,), lead_time, device=x.device, dtype=torch.int64)
        #eps = torch.zeros_like(x)  #? random noise size like x
        eps = torch.randn_like(x)  # random noise size B, 1, H, W
        eps = torch.tensor(gaussian_filter(eps.cpu().numpy(), sigma=gf_sigmat), device=x.device) if gf_sigmat > 0 else eps
        #eps[:, 0, :, :] = torch.randn((x.size(0), x.size(2), x.size(3)))  # Shrimp: only add noise to radar img (others all 0)
        x_t = self.alpha_t[t_sample] * x + self.sigma_t[t_sample] * eps  # Add noise B, 1, H, W
        pred_target = self.nn_module(torch.cat((x_t, cond), dim=1), t_sample, lead_time)  # cat(), B, C, H, W
        
        if self.target_type == "pred_x_0":  # Predict original x
            gt_target = x
        elif self.target_type == "pred_eps":  # Predict random noise
            gt_target = eps
        elif self.target_type == "pred_v":  # Predict noise based on formulas
            gt_target = self.alpha_t[t_sample] * eps - self.sigma_t[t_sample] * x
        else:
            raise AssertionError(f"Invalid {self.target_type=}.")

        #gt_target = gt_target[:, :1, :, :]  # No broadcasting

        if self.loss_type == "l2":
            loss = 0.5 * (gt_target - pred_target) ** 2
        elif self.loss_type == "l1":
            loss = torch.abs(gt_target - pred_target)
        elif self.loss_type == "Hilburn_Loss":
            loss = Hilburn_Loss.loss(pred_target, gt_target)
        else:
            raise AssertionError(f"Invalid {self.loss_type=}.")

        return loss

    @torch.no_grad()
    def sample(self, cond, bsz:int, device:str, num_sampling_timesteps:int, lead_time, gf_sigma1=0, gf_sigma2=0):
        """
        Parameters
        ----------
        num_sampling_timesteps: int. If unspecified, defaults to self.num_timesteps.

        Returns
        -------
        samples: (num_sampling_timesteps + 1, bsz, *self.input_shape)
            index 0 corresponds to x_0
            index t corresponds to x_t
            last index corresponds to random noise
        """
        num_sampling_timesteps = num_sampling_timesteps or self.num_timesteps
        assert 1 <= num_sampling_timesteps <= self.num_timesteps

        #x = torch.randn((bsz, *self.input_shape), device=device)  # Inital random noise b, c, H, W
        #x = torch.randn((bsz, 1, 128, 128), device=device)  #? Inital random noise b, 1, 128, 128 (radar img)
        x = torch.randn((bsz, 1, *cond.shape[2:]), device=device)  #? Inital random noise b, 1, H, W (radar img)
        x = torch.tensor(gaussian_filter(x.cpu().numpy(), sigma=gf_sigma1), device=device) if gf_sigma1 > 0 else x
        
        # ++img
        t_start = torch.empty((bsz,), dtype=torch.int64, device=device) # shape: b
        t_end = torch.empty((bsz,), dtype=torch.int64, device=device) # shape: b
        lead_time = torch.full((bsz,), lead_time, device=device, dtype=torch.int64)  # Lead time

        subseq = torch.linspace(self.num_timesteps, 0, num_sampling_timesteps + 1).round()  # sequence from num_sampling_timesteps to 0
        samples = torch.empty((num_sampling_timesteps + 1, bsz, *x.shape[1:]), device=device)  # to save the sample
        samples[-1] = x  # x = the first step

        # Note that t_start > t_end we're traversing pairwise down subseq.
        # For example, subseq here could be [500, 400, 300, 200, 100, 0]
        for idx, (scalar_t_start, scalar_t_end) in enumerate(zip(subseq[:-1], subseq[1:])):  #Extract pair step

            t_start.fill_(scalar_t_start)
            t_end.fill_(scalar_t_end)
            noise = torch.zeros_like(x) if scalar_t_end == 0 else torch.randn_like(x)  # Noise B, 1, H, W
            noise = torch.tensor(gaussian_filter(noise.cpu().numpy(), sigma=gf_sigma2), device=device) if gf_sigma2 > 0 else noise
            #noise_cond = torch.cat((noise, cond), dim=1)  # Only add noise on radar img dim

            if self.gamma_type == "ddim":
                gamma_t = 0.0
            elif self.gamma_type == "ddpm":
                gamma_t = (
                    self.sigma_t[t_end] / self.sigma_t[t_start] *
                    (1 - self.alpha_t[t_start] ** 2 / self.alpha_t[t_end] ** 2) ** 0.5
                )
            else:
                raise AssertionError(f"Invalid {self.gamma_type=}.")

            x_cond = torch.cat((x, cond), dim=1)  # Shrimp: Initial random noise b, 1, 128, 128
            nn_out = self.nn_module(x_cond, t_start, lead_time)  # nn.out B, C, H, W
            if self.target_type == "pred_x_0":
                pred_x_0 = nn_out
                pred_eps = (x - self.alpha_t[t_start] * nn_out) / self.sigma_t[t_start]
            elif self.target_type == "pred_eps":
                pred_x_0 = (x - self.sigma_t[t_start] * nn_out) / self.alpha_t[t_start]
                pred_eps = nn_out
            elif self.target_type == "pred_v":
                pred_x_0 = self.alpha_t[t_start] * x - self.sigma_t[t_start] * nn_out
                pred_eps = self.sigma_t[t_start] * x + self.alpha_t[t_start] * nn_out
            else:
                raise AssertionError(f"Invalid {self.target_type=}.")

            x = (
                (self.alpha_t[t_end] * pred_x_0) +
                (self.sigma_t[t_end] ** 2 - gamma_t ** 2).clamp(min=0) ** 0.5 * pred_eps +
                (gamma_t * noise)
            )  # sample of step t
            samples[-1 - idx - 1] = x

        return samples
    
    @torch.no_grad()
    def fss(self, x: torch.Tensor, cond, lead_time, gf_sigmat=0):  # 
        """
        Returns
        -------
        fss: (bsz, *input_shape)
        """
        bsz, *_ = x.shape  # bsz = batch size
        t_sample = torch.randint(1, self.num_timesteps + 1, size=(bsz,), device=x.device)  # Random time step
        lead_time = torch.full((bsz,), lead_time, device=x.device, dtype=torch.int64)
        eps = torch.randn_like(x)  # random noise size B, 1, H, W
        eps = torch.tensor(gaussian_filter(eps.cpu().numpy(), sigma=gf_sigmat), device=x.device) if gf_sigmat > 0 else eps
        x_t = self.alpha_t[t_sample] * x + self.sigma_t[t_sample] * eps  # Add noise B, 1, H, W
        pred_target = self.nn_module(torch.cat((x_t, cond), dim=1), t_sample, lead_time)  # cat(), B, C, H, W
        
        if self.target_type == "pred_x_0":  # Predict original x
            gt_target = x
        elif self.target_type == "pred_eps":  # Predict random noise
            gt_target = eps
        elif self.target_type == "pred_v":  # Predict noise based on formulas
            gt_target = self.alpha_t[t_sample] * eps - self.sigma_t[t_sample] * x
        else:
            raise AssertionError(f"Invalid {self.target_type=}.")

        fss = avg_fss(pred_target/2+0.5, gt_target/2+0.5)

        return fss

