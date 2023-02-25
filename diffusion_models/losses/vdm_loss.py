"""Implements loss functions for a VDM."""

from typing import Optional, Union

import torch
from torch import nn
import torch.nn.functional as F


# Loss function for a VDM that learns to model the noise.
def p_losses(
    denoise_model: nn.Module,
    x_start: torch.Tensor, 
    t: torch.Tensor,
    noise: Optional[torch.Tensor]=None,
    loss_type: str="l1",
    device: Union[torch.device, str] = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> torch.Tensor:
    """Implements the simplified loss function from the DDPM paper.
    
    This implements equation (14) from the paper:

    \[L(\theta) = \mathbb{E}_{t, x_0, \epsilon}[||\epsilon - \epsilon_\theta(x_t, t)||_2^2\]

    via a single-sample estimate for a batch of x_t and t.
    """
    if noise is None:
        noise = torch.randn_like(x_start, device=device)

    # Get x_t samples from q(x_t | x_0).
    x_noisy = denoise_model.q_sample(x_start=x_start, t=t, noise=noise)

    # Get noise predictions $\epsilon_\theta(x_t, t)$.
    predicted_noise = denoise_model(x_noisy, t)

    # Technically using the L2 loss is "correct", but we implement the option
    # to use other types of reconstruction losses.
    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == 'huber':
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError(f"Loss type {loss_type} is not implemented.")
    
    return loss
    
    
