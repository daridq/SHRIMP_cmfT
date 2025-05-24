"""
Flow Matching implementation inspired by diffusion.py structure.
Supports DiT (Diffusion Transformer) as backbone architecture.
"""
from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from src.blocks import unsqueeze_to
from src.utils import Hilburn_Loss, avg_fss


@dataclass(frozen=True)
class FlowMatchingConfig:
    """Configuration for Flow Matching model."""
    
    # Flow matching specific parameters
    sigma_min: float = 1e-4          # Minimum noise level
    sigma_max: float = 1.0           # Maximum noise level  
    rho: float = 7.0                 # Time step distribution parameter
    
    # Training parameters
    target_type: str = "velocity"     # "velocity" or "x_0"
    loss_type: str = "l2"            # "l1", "l2", "Hilburn_Loss"
    
    # Sampling parameters
    num_sampling_steps: int = 50     # Number of steps for ODE solver
    solver_type: str = "euler"       # "euler", "heun", "dopri5"
    
    def __post_init__(self):
        assert self.sigma_min > 0 and self.sigma_max > self.sigma_min
        assert self.rho > 0
        assert self.target_type in ("velocity", "x_0")
        assert self.loss_type in ("l1", "l2", "Hilburn_Loss")
        assert self.solver_type in ("euler", "heun", "dopri5")
        assert self.num_sampling_steps > 0


class FlowMatchingModel(nn.Module):
    """
    Flow Matching model with DiT backbone support.
    
    Flow Matching learns a continuous vector field that transforms noise to data,
    enabling high-quality generation through ODE solving.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        nn_module: nn.Module,
        config: FlowMatchingConfig,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.nn_module = nn_module  # DiT backbone
        self.config = config
        
        # Validate input shape (c,) or (c, h, w) or (c, t, h, w)
        assert len(input_shape) in (1, 3, 4), f"Invalid input shape: {input_shape}"
        
        # Flow matching uses continuous time in [0, 1]
        self.sigma_min = config.sigma_min
        self.sigma_max = config.sigma_max
        self.rho = config.rho
        
    def _get_noise_schedule(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get noise schedule for flow matching.
        Returns alpha_t and sigma_t for time t ∈ [0, 1].
        """
        # Linear interpolation schedule #TODO: change to sigmoid of normal distribution
        alpha_t = 1.0 - t
        sigma_t = self.sigma_min + t * (self.sigma_max - self.sigma_min)
        
        # Reshape to match input dimensions
        alpha_t = unsqueeze_to(alpha_t, len(self.input_shape) + 1)
        sigma_t = unsqueeze_to(sigma_t, len(self.input_shape) + 1)
        
        return alpha_t, sigma_t
    
    def _sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample time steps according to the rho distribution."""
        # Sample from uniform distribution and apply power transformation
        u = torch.rand(batch_size, device=device)
        t = u ** (1.0 / self.rho)
        return t
    
    def _get_velocity_target(
        self, 
        x_0: torch.Tensor, 
        x_1: torch.Tensor, 
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the target velocity field for flow matching.
        
        For linear interpolation path: x_t = (1-t)*x_0 + t*x_1
        The velocity is: v_t = dx_t/dt = x_1 - x_0
        """
        return x_1 - x_0
    
    def _get_flow_path(
        self, 
        x_0: torch.Tensor, 
        x_1: torch.Tensor, 
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Get point on the flow path at time t.
        Linear interpolation: x_t = (1-t)*x_0 + t*x_1
        """
        alpha_t, _ = self._get_noise_schedule(t)
        return alpha_t * x_0 + (1 - alpha_t) * x_1
    
    def loss(
        self, 
        x: torch.Tensor, 
        cond: torch.Tensor, 
        lead_time: int, 
        gf_sigmat: float = 0
    ) -> torch.Tensor:
        """
        Compute Flow Matching training loss.
        
        Args:
            x: Target data (B, C, H, W)
            cond: Conditioning information (B, C_cond, H, W) 
            lead_time: Lead time for prediction
            gf_sigmat: Gaussian filter sigma for noise
            
        Returns:
            loss: Training loss tensor
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Sample time steps
        t = self._sample_time(batch_size, device)
        lead_time_tensor = torch.full((batch_size,), lead_time, device=device, dtype=torch.int64)
        
        # Sample noise (x_0 in flow matching terminology)
        noise = torch.randn_like(x)
        if gf_sigmat > 0:
            noise = torch.tensor(
                gaussian_filter(noise.cpu().numpy(), sigma=gf_sigmat), 
                device=device
            )
        
        # Get point on flow path
        x_t = self._get_flow_path(noise, x, t)
        
        # Prepare input for DiT backbone
        model_input = torch.cat([x_t, cond], dim=1)
        
        # Get model prediction
        if hasattr(self.nn_module, 'forward_with_cfg'):
            # DiT with classifier-free guidance
            pred_output = self.nn_module.forward_with_cfg(
                model_input, t, lead_time_tensor, cfg_scale=1.0
            )
        else:
            # Standard forward pass
            pred_output = self.nn_module(model_input, t, lead_time_tensor)
        
        # Compute target based on target_type
        if self.config.target_type == "velocity":
            target = self._get_velocity_target(noise, x, t)
        elif self.config.target_type == "x_0":
            target = x
        else:
            raise ValueError(f"Invalid target_type: {self.config.target_type}")
        
        # Compute loss
        if self.config.loss_type == "l2":
            loss = 0.5 * (target - pred_output) ** 2
        elif self.config.loss_type == "l1":
            loss = torch.abs(target - pred_output)
        elif self.config.loss_type == "Hilburn_Loss":
            loss = Hilburn_Loss.loss(pred_output, target)
        else:
            raise ValueError(f"Invalid loss_type: {self.config.loss_type}")
        
        return loss
    
    def _euler_step(
        self, 
        x: torch.Tensor, 
        cond: torch.Tensor, 
        t: torch.Tensor, 
        dt: float, 
        lead_time_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Single Euler integration step."""
        model_input = torch.cat([x, cond], dim=1)
        
        if hasattr(self.nn_module, 'forward_with_cfg'):
            velocity = self.nn_module.forward_with_cfg(
                model_input, t, lead_time_tensor, cfg_scale=1.0
            )
        else:
            velocity = self.nn_module(model_input, t, lead_time_tensor)
        
        return x + dt * velocity
    
    def _heun_step(
        self, 
        x: torch.Tensor, 
        cond: torch.Tensor, 
        t: torch.Tensor, 
        dt: float, 
        lead_time_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Single Heun integration step (2nd order)."""
        # First step (Euler)
        model_input = torch.cat([x, cond], dim=1)
        
        if hasattr(self.nn_module, 'forward_with_cfg'):
            v1 = self.nn_module.forward_with_cfg(
                model_input, t, lead_time_tensor, cfg_scale=1.0
            )
        else:
            v1 = self.nn_module(model_input, t, lead_time_tensor)
        
        x_temp = x + dt * v1
        
        # Second step
        model_input_temp = torch.cat([x_temp, cond], dim=1)
        t_next = t + dt
        
        if hasattr(self.nn_module, 'forward_with_cfg'):
            v2 = self.nn_module.forward_with_cfg(
                model_input_temp, t_next, lead_time_tensor, cfg_scale=1.0
            )
        else:
            v2 = self.nn_module(model_input_temp, t_next, lead_time_tensor)
        
        # Heun correction
        return x + 0.5 * dt * (v1 + v2)
    
    @torch.no_grad()
    def sample(
        self, 
        cond: torch.Tensor, 
        batch_size: int, 
        device: str, 
        num_sampling_steps: Optional[int] = None,
        lead_time: int = 0,
        gf_sigma1: float = 0,
        gf_sigma2: float = 0,
        cfg_scale: float = 1.0
    ) -> torch.Tensor:
        """
        Generate samples using ODE solver.
        
        Args:
            cond: Conditioning information
            batch_size: Number of samples to generate
            device: Device to run on
            num_sampling_steps: Number of integration steps
            lead_time: Lead time for prediction
            gf_sigma1: Gaussian filter for initial noise
            gf_sigma2: Gaussian filter for intermediate steps
            cfg_scale: Classifier-free guidance scale
            
        Returns:
            samples: Generated samples (num_steps + 1, B, C, H, W)
        """
        num_steps = num_sampling_steps or self.config.num_sampling_steps
        
        # Initialize with noise
        x = torch.randn((batch_size, 1, *cond.shape[2:]), device=device)
        if gf_sigma1 > 0:
            x = torch.tensor(
                gaussian_filter(x.cpu().numpy(), sigma=gf_sigma1), 
                device=device
            )
        
        # Time schedule
        t_schedule = torch.linspace(0, 1, num_steps + 1, device=device)
        dt = 1.0 / num_steps
        
        lead_time_tensor = torch.full((batch_size,), lead_time, device=device, dtype=torch.int64)
        
        # Store samples
        samples = torch.empty((num_steps + 1, batch_size, *x.shape[1:]), device=device)
        samples[0] = x
        
        # ODE integration
        for i in range(num_steps):
            t = t_schedule[i].expand(batch_size)
            
            if self.config.solver_type == "euler":
                x = self._euler_step(x, cond, t, dt, lead_time_tensor)
            elif self.config.solver_type == "heun":
                x = self._heun_step(x, cond, t, dt, lead_time_tensor)
            else:
                raise ValueError(f"Unsupported solver: {self.config.solver_type}")
            
            # Optional noise injection for intermediate steps
            if gf_sigma2 > 0 and i < num_steps - 1:
                noise = torch.randn_like(x) * 0.1  # Small noise injection
                noise = torch.tensor(
                    gaussian_filter(noise.cpu().numpy(), sigma=gf_sigma2), 
                    device=device
                )
                x = x + noise
            
            samples[i + 1] = x
        
        return samples
    
    @torch.no_grad()
    def fss(
        self, 
        x: torch.Tensor, 
        cond: torch.Tensor, 
        lead_time: int, 
        gf_sigmat: float = 0
    ) -> torch.Tensor:
        """
        Compute Fractions Skill Score for evaluation.
        
        Args:
            x: Ground truth data
            cond: Conditioning information  
            lead_time: Lead time for prediction
            gf_sigmat: Gaussian filter sigma
            
        Returns:
            fss: Fractions Skill Score
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Sample random time step
        t = self._sample_time(batch_size, device)
        lead_time_tensor = torch.full((batch_size,), lead_time, device=device, dtype=torch.int64)
        
        # Add noise
        noise = torch.randn_like(x)
        if gf_sigmat > 0:
            noise = torch.tensor(
                gaussian_filter(noise.cpu().numpy(), sigma=gf_sigmat), 
                device=device
            )
        
        # Get noisy version
        x_t = self._get_flow_path(noise, x, t)
        
        # Get model prediction
        model_input = torch.cat([x_t, cond], dim=1)
        
        if hasattr(self.nn_module, 'forward_with_cfg'):
            pred_output = self.nn_module.forward_with_cfg(
                model_input, t, lead_time_tensor, cfg_scale=1.0
            )
        else:
            pred_output = self.nn_module(model_input, t, lead_time_tensor)
        
        # Determine ground truth based on target type
        if self.config.target_type == "velocity":
            gt_target = self._get_velocity_target(noise, x, t)
        elif self.config.target_type == "x_0":
            gt_target = x
        else:
            raise ValueError(f"Invalid target_type: {self.config.target_type}")
        
        # Compute FSS
        fss = avg_fss(pred_output / 2 + 0.5, gt_target / 2 + 0.5)
        
        return fss


# Utility function to create Flow Matching model with DiT
def create_flow_matching_dit(
    input_shape: Tuple[int, ...],
    dit_config: dict,
    fm_config: FlowMatchingConfig,
) -> FlowMatchingModel:
    """
    Create Flow Matching model with DiT backbone.
    
    Args:
        input_shape: Shape of input data
        dit_config: Configuration for DiT model
        fm_config: Flow Matching configuration
        
    Returns:
        FlowMatchingModel with DiT backbone
    """
    from src.DiTModels import DiT_models  # Import your DiT implementation
    
    # Create DiT backbone
    dit_backbone = DiT_models[dit_config['model_name']](
        input_size=dit_config['input_size'],
        in_channels=dit_config['in_channels'],
        **dit_config.get('model_kwargs', {})
    )
    
    # Create Flow Matching model
    flow_model = FlowMatchingModel(
        input_shape=input_shape,
        nn_module=dit_backbone,
        config=fm_config
    )
    
    return flow_model

def get_default_config() -> FlowMatchingConfig:
    """Get default Flow Matching configuration."""
    return FlowMatchingConfig(
        sigma_min=1e-4,
        sigma_max=1.0,
        rho=7.0,
        target_type="velocity",
        loss_type="l2",
        num_sampling_steps=50,
        solver_type="heun"
    )
