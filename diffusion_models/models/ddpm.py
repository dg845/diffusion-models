"""Implements classes for variational diffusion models."""

from typing import Callable, Optional, Union

from tqdm.auto import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_models.utils.diffusion_utils import discretized_gaussian_log_likelihood, extract, meanflat, normal_kl

class DDPM(nn.Module):
    """Implements a diffusion model that learns to predict the source noise."""

    def __init__(self, model: nn.Module, timesteps: int, var_sched_fn: Callable) -> None:
        super().__init__()
        self.denoise_model = model
        self.timesteps = timesteps

        # Get variance schedule
        # self.betas has shape (self.timesteps)
        self.betas = var_sched_fn(timesteps=self.timesteps)

        # Calculate quantities related to alphas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # Quantities for forward diffusion process q(x_t | x_{t - 1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1. - self.alphas_cumprod)

        # Quantities to relate x_t, x_0, and \epsilon_0
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)

        # Quantities for ground truth posterior q(x_{t - 1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        # Clip posterior_log_variance since the posterior variance starts out at 0
        self.posterior_log_variance = torch.log(F.pad(self.posterior_variance[1:], (1, 0), value=self.posterior_variance[1]))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

    def forward(self, x: torch.Tensor, device) -> torch.Tensor:
        batch_size = x.shape[0]

        # Sample timesteps uniformly at random from [0,..., T - 1] each sample
        # in the batch.
        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()

        return self.denoise_model(x, t)
    
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor]=None,
        device: Union[torch.device, str]="cuda" if torch.cuda.is_available() else "cpu",
    ) -> torch.Tensor:
        """Run the forward diffusion process to get x_start at noise level t.
        
        In particular, this samples from q(x_t | x_0) in batch mode via

        \[ x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon_0 \]

        where $\epsilon_0$ is a noise sample sampled from a standard Gaussian.
        """
        if noise is None:
            noise = torch.randn_like(x_start, device=device)
        
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def q_mean_variance(self, x_start: torch.Tensor, t: torch.Tensor):
        """Get the mean, variance, and log variance of the forward process
        posterior q(x_t | x_0).
        """
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        mean = sqrt_alphas_cumprod_t * x_start
        
        var = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_var = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        
        return mean, var, log_var
    
    def q_posterior_mean_variance(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        """Get the mean, variance, and log variance of the reverse diffusion
        process posterior q(x_{t - 1} | x_t, x_0).
        """
        mean_x_start_coef_t = extract(self.posterior_mean_coef1, t, x_t.shape)
        mean_x_t_coef_t = extract(self.posterior_mean_coef2, t, x_t.shape)
        posterior_mean = mean_x_start_coef_t * x_start + mean_x_t_coef_t * x_t

        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = extract(self.posterior_log_variance, t, x_t.shape)

        return posterior_mean, posterior_variance, posterior_log_variance
    
    def p_mean_variance(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        device: Union[torch.device, str]="cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Calculate the mean, variance, and log variance of the denoising transition
        p_theta(x_{t - 1} | x_t).
        """
        # Estimate source noise using our denoising model
        model_output = self.denoise_model(x, t)
        pred_xstart = self.predict_xstart_from_eps(x_t=x, t=t, eps=model_output)

        # We choose the mean parameterization to match the posterior mean
        # (except we use the predicted x_start instead of true x_start)
        # and choose the variance equal to the posterior variance.
        model_mean, model_var, model_log_var = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        # Reshape the var and log_var to x.shape (from [batch_size, 1, ...,1]).
        model_var = model_var * torch.ones(x.shape, device=device)
        model_log_var = model_log_var * torch.ones(x.shape, device=device)

        return model_mean, model_var, model_log_var
    
    def predict_xstart_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor):
        """Calculate x_0 in terms of x_t and eps ($\epsilon_0$)."""
        sqrt_recip_alphas_cumprod_t = extract(
            self.sqrt_recip_alphas_cumprod, t, x_t.shape
        )
        sqrt_recipm1_alphas_cumprod_t = extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )

        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * eps
    
    def p_losses(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor]=None,
        loss_type: str="l2",
        device: Union[torch.device, str] = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> torch.Tensor:
        """Implements the simplified loss function from the DDPM paper.
        
        This implements equation (14) from the paper:

        \[L(\theta) = \mathbb{E}_{t, x_0, \epsilon}[||\epsilon - \epsilon_\theta(x_t, t)||_2^2\]

        via a single-sample estimate for a batch of x_t and t.
        """
        if noise is None:
            noise = torch.randn_like(x_start, device=device)
        
        # Get x_t samples from q(x_t | x_0)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise, device=device)

        # Get noise predictions from our denoising model $\epsilon_\theta(x_t, t)$.
        predicted_noise = self.denoise_model(x_noisy, t)

        # L_simple uses the L2 loss, but we implement the ability to experiment
        # with other reconstruction losses.
        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == 'huber':
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError(f"Loss type {loss_type} is not implemented.")
        
        return loss
    
    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        t_index: int,
    ) -> torch.Tensor:
        """Run one step of the reverse diffusion process.
        
        This implements one step in the loop in Algorithm 2 of the DDPM paper
        in batch modee. Given the previous sample x_t at noise level t, we
        get a sample x_{t-1} at noise level t - 1 using our formula for the
        denoising transition $p_\theta(x_{t-1} | x_t)$ via the
        reparameterization trick.
        """
        betas_t = extract(self.betas, t, x.shape)

        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Use our noise prediction model to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.denoise_model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def p_sample_loop(
        self,
        shape,
        device: Union[torch.device, str]="cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Implements the sampling loop for the reverse diffusion process.

        This runs the reverse diffusion process from timestep T - 1 down to 0,
        getting samples at timestep t using the p_sample() method.
        """
        b = shape[0]
        # Start from pure noise for each sample in the batch.
        img = torch.randn(shape, device=device)
        imgs = []

        for i in tqdm(
            reversed(range(0, self.timesteps)),
            desc='sampling loop time step',
            total=self.timesteps
            ):
            img = self.p_sample(
                img,
                torch.full((b,), i, device=device, dtype=torch.long),
                i
            )
            # imgs.append(img.cpu().numpy())
            imgs.append(img.cpu())
        
        return imgs
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int=16,
        channels: int=3,
        image_size: int=28,
        device: Union[torch.device, str]="cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Sample from the diffusion model by running the reverse diffusion process."""
        return self.p_sample_loop(
            shape=(batch_size, channels, image_size, image_size), device=device
        )
    
    def vb_terms_bpd(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        device: Union[torch.device, str]="cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Get the terms of the negative VLB to approximate the NLL for the
        reconstruction term L_0 and denoising matching terms
        L_1, ..., L_{T - 1}, then calculate the bits per dimension of those
        terms.
        """
        true_mean, _, true_log_var = self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)
        model_mean, _, model_log_var = self.p_mean_variance(x=x_t, t=t, device=device)
        kl = normal_kl(true_mean, true_log_var, model_mean, model_log_var)
        # Calculate the bits per dim. from the KL term (reduce along all but the batch dim).
        kl = meanflat(kl) / np.log(2.)

        # Handle the decoder term L_0.
        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=model_mean, log_scales=0.5 * model_log_var
        )
        # Calculate the bits per dim. for the decoder
        decoder_nll = meanflat(decoder_nll) / np.log(2.)

        output = torch.where(torch.eq(t, torch.zeros_like(t)), decoder_nll, kl)
        return output
    
    def prior_bpd(
        self,
        x_start: torch.Tensor,
        device: Union[torch.device, str]="cuda" if torch.cuda.is_available() else "cpu",
    ):
        B, T = x_start.shape[0], self.timesteps
        qt_mean, _, qt_log_var = self.q_mean_variance(x_start, t=torch.full([B], T - 1, device=device))
        kl_prior = normal_kl(qt_mean, qt_log_var,
            torch.zeros_like(qt_mean, device=device), torch.zeros_like(qt_log_var, device=device)
        )
        # Calculate the bits per dim.
        return meanflat(kl_prior) / np.log(2)
    
    @torch.no_grad()
    def calc_bpd_loop(
        self,
        x_start: torch.Tensor,
        device: Union[torch.device, str]="cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Calculates the bits per dimension via approximating the NLL using
        the diffusion VLB.
        """
        batch_size = x_start.shape[0]
        terms_bpd_bt = torch.zeros(batch_size, self.timesteps, device=device)
        for i in tqdm(
            reversed(range(0, self.timesteps)),
            desc='BPD loop time step',
            total=self.timesteps,
            ):
            t = torch.full([batch_size], i, device=device)
            x_t = self.q_sample(x_start, t, device=device)
            # This should have shape (batch_size)
            new_vals_b = self.vb_terms_bpd(x_start, x_t, t, device=device)
            
            # Insert new_vals_b as a column tensor into terms at the
            # appropriate column of terms_bpd_bt.
            mask_bt = torch.eq(t[:, None], torch.arange(self.timesteps, device=device)[None, :])
            terms_bpd_bt = terms_bpd_bt * (~mask_bt) + new_vals_b[:, None] * mask_bt
        
        prior_bpd_b = self.prior_bpd(x_start, device=device)
        total_bpd_b = torch.sum(terms_bpd_bt, dim=1) + prior_bpd_b

        return total_bpd_b
