"""Variance schedules for the forward diffusion process (encoders q)"""

import torch

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def get_var_sched_fn(var_sched_str):
    if var_sched_str == "cosine":
        var_sched_fn = cosine_beta_schedule
    elif var_sched_str == "linear":
        var_sched_fn = linear_beta_schedule
    elif var_sched_str == "quadratic":
        var_sched_fn = quadratic_beta_schedule
    elif var_sched_str == "sigmoid":
        var_sched_fn = sigmoid_beta_schedule
    else:
        raise NotImplementedError(f"Variance schedule {var_sched_str} is not implemented.")
    return var_sched_fn