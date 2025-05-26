"""
This file provides a Conditional Flow Matching (CFM) implementation to replace DDIM/DDPM.
Based on the paper "Flow Matching for Generative Modeling" (Lipman et al. 2023)
"""
from dataclasses import dataclass
import math

import torch
import torch.nn as nn

from src.blocks import unsqueeze_to
from src.utils import Hilburn_Loss, avg_fss

from scipy.ndimage import gaussian_filter
import numpy as np


@dataclass(frozen=True)
class FlowMatchingConfig:
    """Configuration for Flow Matching model"""
    num_timesteps: int
    path_type: str = "optimal_transport"  # "optimal_transport" or "diffusion_like"
    loss_type: str = "l2"
    sigma_min: float = 0.001  # minimum noise level at t=1

    def __post_init__(self):
        assert self.num_timesteps > 0
        assert self.path_type in ("optimal_transport", "diffusion_like")
        assert self.loss_type in ("l1", "l2", "Hilburn_Loss")


class FlowMatchingModel(nn.Module):
    """
    Conditional Flow Matching model that replaces DiffusionModel
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        nn_module: nn.Module,
        config: FlowMatchingConfig,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.nn_module = nn_module
        self.num_timesteps = config.num_timesteps
        self.path_type = config.path_type
        self.loss_type = config.loss_type
        self.sigma_min = config.sigma_min

        # Input shape must be either (c,) or (c, h, w) or (c, t, h, w)
        assert len(input_shape) in (1, 3, 4)

    def get_conditional_flow_params(self, t, x1):
        """
        Get conditional flow parameters for time t and target x1
        
        For Optimal Transport path:
        - mu_t(x1) = t * x1
        - sigma_t = (1 - (1 - sigma_min) * t)
        
        For diffusion-like path:
        - Similar to original diffusion but with flow matching formulation
        """
        if self.path_type == "optimal_transport":
            # Optimal Transport conditional path
            mu_t = t * x1  # Linear interpolation in mean
            sigma_t = 1 - (1 - self.sigma_min) * t  # Linear interpolation in std
            
            # Vector field for OT path: u_t(x|x1) = (x1 - (1-sigma_min)*x) / (1-(1-sigma_min)*t)
            # But we'll compute this in the loss function
            
        else:  # diffusion_like
            # Variance preserving path similar to diffusion
            alpha_t = torch.cos(t * math.pi / 2)
            sigma_t = torch.sin(t * math.pi / 2)
            mu_t = alpha_t * x1
            
        return mu_t, sigma_t

    def sample_conditional_path(self, t, x0, x1):
        """
        Sample from conditional probability path p_t(x|x1)
        
        Args:
            t: time tensor (batch_size,)
            x0: noise sample (batch_size, *input_shape)
            x1: data sample (batch_size, *input_shape)
        
        Returns:
            x_t: sample at time t
            target: target vector field
        """
        # Ensure t has the right shape for broadcasting
        t = t.view(-1, *([1] * len(self.input_shape)))
        
        if self.path_type == "optimal_transport":
            # Optimal Transport path: x_t = (1-(1-sigma_min)*t)*x0 + t*x1
            sigma_t = 1 - (1 - self.sigma_min) * t
            x_t = sigma_t * x0 + t * x1
            
            # Target vector field: u_t = x1 - (1-sigma_min)*x0
            target = x1 - (1 - self.sigma_min) * x0
            
        else:  # diffusion_like
            # Variance preserving interpolation
            alpha_t = torch.cos(t * math.pi / 2)
            sigma_t = torch.sin(t * math.pi / 2)
            
            x_t = alpha_t * x1 + sigma_t * x0
            
            # Target vector field for VP path
            target = -math.pi/2 * (torch.sin(t * math.pi / 2) * x1 - torch.cos(t * math.pi / 2) * x0)
            
        return x_t, target

    def loss(self, x: torch.Tensor, cond, lead_time, gf_sigmat=0):
        """
        Conditional Flow Matching loss
        
        Args:
            x: data samples (batch_size, *input_shape)
            cond: conditioning information
            lead_time: lead time for prediction
            gf_sigmat: gaussian filter sigma for noise (for compatibility)
        
        Returns:
            loss: CFM loss
        """
        bsz, *_ = x.shape
        
        # Sample random time
        t = torch.rand(bsz, device=x.device)  # t ~ U[0,1]
        lead_time = torch.full((bsz,), lead_time, device=x.device, dtype=torch.int64)
        
        # Sample noise
        x0 = torch.randn_like(x)
        if gf_sigmat > 0:
            x0 = torch.tensor(gaussian_filter(x0.cpu().numpy(), sigma=gf_sigmat), device=x.device)
        
        # Sample from conditional path and get target vector field
        x_t, target_vf = self.sample_conditional_path(t, x0, x)
        
        # Predict vector field
        # Convert t to timestep format expected by the network (similar to diffusion)
        t_discrete = (t * self.num_timesteps).long().clamp(0, self.num_timesteps - 1)
        pred_vf = self.nn_module(torch.cat((x_t, cond), dim=1), t_discrete, lead_time)
        
        # Compute loss
        if self.loss_type == "l2":
            loss = 0.5 * (target_vf - pred_vf) ** 2
        elif self.loss_type == "l1":
            loss = torch.abs(target_vf - pred_vf)
        elif self.loss_type == "Hilburn_Loss":
            loss = Hilburn_Loss.loss(pred_vf, target_vf)
        else:
            raise AssertionError(f"Invalid {self.loss_type=}.")

        return loss

    @torch.no_grad()
    def sample(self, cond, bsz: int, device: str, num_sampling_timesteps: int, lead_time, gf_sigma1=0, gf_sigma2=0):
        """
        Sample from the flow model using ODE integration
        
        Args:
            cond: conditioning information
            bsz: batch size
            device: device
            num_sampling_timesteps: number of integration steps
            lead_time: lead time
            gf_sigma1: gaussian filter for initial noise
            gf_sigma2: gaussian filter for intermediate noise (not used in deterministic sampling)
        
        Returns:
            samples: trajectory of samples from t=1 to t=0
        """
        num_sampling_timesteps = num_sampling_timesteps or self.num_timesteps
        
        # Start from noise at t=1
        x = torch.randn((bsz, 1, *cond.shape[2:]), device=device)
        if gf_sigma1 > 0:
            x = torch.tensor(gaussian_filter(x.cpu().numpy(), sigma=gf_sigma1), device=device)
        
        lead_time = torch.full((bsz,), lead_time, device=device, dtype=torch.int64)
        
        # Time steps from 1 to 0
        dt = 1.0 / num_sampling_timesteps
        times = torch.linspace(1.0, 0.0, num_sampling_timesteps + 1)
        
        samples = torch.empty((num_sampling_timesteps + 1, bsz, *x.shape[1:]), device=device)
        samples[0] = x  # t=1 (noise)
        
        # Integrate ODE: dx/dt = -v_t(x)
        for i in range(num_sampling_timesteps):
            t_curr = times[i]
            t_discrete = torch.full((bsz,), int(t_curr * self.num_timesteps), device=device, dtype=torch.long)
            t_discrete = t_discrete.clamp(0, self.num_timesteps - 1)
            
            # Predict vector field
            x_cond = torch.cat((x, cond), dim=1)
            v_pred = self.nn_module(x_cond, t_discrete, lead_time)
            
            # Euler step: x_{t-dt} = x_t - dt * v_t(x_t)
            x = x - dt * v_pred
            samples[i + 1] = x
        
        return samples

    @torch.no_grad()
    def fss(self, x: torch.Tensor, cond, lead_time, gf_sigmat=0):
        """
        Compute FSS in a way that's comparable to diffusion model
        Compare network prediction vs ground truth target (not final samples)
        
        For CFM, we need to compare the final prediction (x_0) vs ground truth,
        not the vector field predictions.
        """
        bsz, *_ = x.shape
        
        try:
            # Sample random time (similar to diffusion)
            t = torch.rand(bsz, device=x.device)  # t ~ U[0,1]
            lead_time = torch.full((bsz,), lead_time, device=x.device, dtype=torch.int64)
            
            # Sample noise
            x0 = torch.randn_like(x)
            if gf_sigmat > 0:
                x0 = torch.tensor(gaussian_filter(x0.cpu().numpy(), sigma=gf_sigmat), device=x.device)
            
            # Sample from conditional path
            x_t, target_vf = self.sample_conditional_path(t, x0, x)
            
            # Predict vector field
            t_discrete = (t * self.num_timesteps).long().clamp(0, self.num_timesteps - 1)
            pred_vf = self.nn_module(torch.cat((x_t, cond), dim=1), t_discrete, lead_time)
            
            # Convert vector field prediction to data prediction
            # For optimal transport: x_t = (1-(1-σ_min)*t)*x_0 + t*x_1
            # So: x_1 = (x_t - (1-(1-σ_min)*t)*x_0) / t
            # But we want to predict x_1 from the vector field
            
            # Method 1: Use the vector field to predict x_1 directly
            # target_vf = x_1 - (1-σ_min)*x_0, so x_1 = target_vf + (1-σ_min)*x_0
            t_expanded = t.view(-1, *([1] * len(self.input_shape)))
            
            # Ground truth x_1 (the actual data)
            gt_x1 = x
            
            # Predicted x_1 from vector field
            pred_x1 = pred_vf + (1 - self.sigma_min) * x0
            
            # Now compare predicted vs actual precipitation data (like diffusion)
            fss_result = avg_fss(pred_x1/2+0.5, gt_x1/2+0.5)
            
            # Validate FSS result
            if np.isnan(fss_result) or np.isinf(fss_result):
                return torch.tensor(0.0, device=x.device)
            
            return torch.tensor(fss_result, device=x.device)
            
        except Exception as e:
            # If anything fails, return 0 instead of crashing
            return torch.tensor(0.0, device=x.device) 