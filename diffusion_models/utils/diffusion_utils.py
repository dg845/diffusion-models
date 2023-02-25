"""Contains helpful functions for running the diffusion process."""

import numpy as np
import torch


def extract(a: torch.Tensor, t: torch.Tensor, x_shape) -> torch.Tensor:
    """Extract the appropriate t index for a batch of indices.
    
    a should be a tensor of shape (timesteps) and t should be a tensor of shape
    (batch_size), where each element is in [0, timesteps - 1]. We then grab the
    values of a at the index given by t and reshape this into a tensor of shape
    (batch_size, 1, ..., 1) where we have len(x_shape) - 1 1s.

    Reshaping is important because it allows us to broadcast the value at
    out[i][0]...[0] to perform elementwise computations with instance i of a
    batched tensor x with shape x_shape = (batch_size,...).
    """
    # a should have shape (timesteps) [e.g. a beta schedule]
    # t should have shape (batch_size), w/ elements of t in [0, timesteps - 1]
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    
def normal_kl(mean1, logvar1, mean2, logvar2):
    """Calculates the KL divergence between two normal distributions
    parameterized by their mean and log variance.
    """
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                    + torch.cdist(mean1, mean2) * torch.exp(-logvar2))

def meanflat(x: torch.Tensor):
    return torch.mean(x, dim=list(range(1, len(x.shape))))


def approx_standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    # Assumes data is integers [0, 255] rescaled to [-1, 1].
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(torch.maximum(cdf_plus, torch.full_like(cdf_plus, 1e-12)))
    log_one_minus_cdf_min = torch.log(torch.maximum(1. - cdf_min, torch.full_like(cdf_min, 1e-12)))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(
            x > 0.999,
            log_one_minus_cdf_min,
            torch.log(torch.maximum(cdf_delta, torch.full_like(cdf_delta, 1e-12)))
        )
    )
    return log_probs