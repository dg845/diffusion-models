"""Implements a U-Net network based on lucidrains/denoising-diffusion-pytorch.

https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
"""

from functools import partial
from typing import Optional, Tuple

import torch
from torch import nn

from diffusion_models.modules.attention import Attention, LinearAttention
from diffusion_models.modules.model_blocks import Downsample, Upsample
from diffusion_models.modules.model_blocks import ConvNextBlock, Residual, ResnetBlock, PreNorm
from diffusion_models.modules.position_embeddings import SinusoidalPositionEmbeddings
from diffusion_models.utils.utils import default, exists


class Unet(nn.Module):
    def __init__(
        self,
        dim: int,
        init_dim: Optional[int]=None,
        out_dim: Optional[int]=None,
        dim_mults: Tuple[int, ...]=(1, 2, 4, 8),
        channels: int=3,
        self_condition=False,
        resnet_block_groups: int=4,
    ) -> None:
        super().__init__()

        # ----Determine dimensions----

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)

        dims = [init_dim, *map(lambda m: init_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.out_dim = default(out_dim, channels)

        block_cls = partial(ResnetBlock, groups=resnet_block_groups)
        
        # ----Handle time embeddings----

        time_dim = dim * 4
        # Computes position embeddings for noise levels t
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # ----Define model layers----

        # Initial conv layer on noised images
        # Preserves the spatial size
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0)

        self.downsample_layers = nn.ModuleList([])
        self.upsample_layers = nn.ModuleList([])
        num_resolutions = len(in_out)

        # Create downsample layers
        for idx, (dim_in, dim_out) in enumerate(in_out):
            is_last = idx >= (num_resolutions - 1)

            # Each downsample stage consists of:
            #   - 2 ResNet blocks
            #   - 1 pre-GroupNorm linear attention block
            #   - 1 2x downsample operation
            self.downsample_layers.append(
                nn.ModuleList(
                    [
                        block_cls(dim_in, dim_in, time_emb_dim=time_dim),
                        block_cls(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )
        
        # Bottleneck Layer
        #   - ResNet/ConvNeXT block
        #   - pre-GroupNorm multi-head attention block
        #   - ResNet/ConvNeXT block
        mid_dim = dims[-1]
        self.mid_block1 = block_cls(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_cls(mid_dim, mid_dim, time_emb_dim=time_dim)

        # Create upsample layers
        for idx, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = idx >= (num_resolutions - 1)

            # Each upsample stage consists of:
            #   - 2 ResNet blocks
            #   - 1 pre-GroupNorm linear attention block
            #   - 1 2x upsample operation
            self.upsample_layers.append(
                nn.ModuleList(
                    [
                        # Input is dim_in * dim_out to accomodate shortcut
                        # connections from downsample layers
                        block_cls(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_cls(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )
        
        # Final ResNet block
        self.final_res_block = block_cls(dim * 2, dim, time_emb_dim=time_dim)
        # Final conv blocks to scale the image back up to original size
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)
    
    def forward(self, x: torch.Tensor, time: torch.Tensor, x_self_cond=None) -> torch.Tensor:
        # x should have initial shape (batch_size, num_channels, height, width)
        # t should have initial shape (batch_size, 1)
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()
        
        t = self.time_mlp(time)

        # Stack of intermediate downsample x values for shortcut connections
        # between the downsample and upsample layers at the same resolution.
        h = []

        # Downsample
        for block1, block2, attn, downsample in self.downsample_layers:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)
        
        # Bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # Upsample
        for block1, block2, attn, upsample in self.upsample_layers:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)
        
        # Final residual block and conv
        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)
