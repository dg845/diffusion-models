"""PyTorch implementation of the U-Net architecture used in the DDPM paper
https://arxiv.org/abs/2006.11239."""

from functools import partial
from typing import Optional, Tuple

from einops import rearrange
from einops.layers.torch import Rearrange

import torch
from torch import nn

from diffusion_models.modules.attention import Attention, LinearAttention
from diffusion_models.modules.model_blocks import Residual
from diffusion_models.modules.position_embeddings import SinusoidalPositionEmbeddings
from diffusion_models.utils.utils import default, exists


# ----Upsample/Downsample----


def Upsample(dim: int, dim_out: Optional[int]=None) -> nn.Module:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim: int, dim_out: Optional[int]=None) -> nn.Module:
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )


# ----Normalization Blocks----


class PreNorm(nn.Module):
    """Applies an activation and then group norm before a submodel fn."""
    def __init__(self, dim, fn, act_fn=nn.Identity(), groups=1) -> None:
        super().__init__()
        self.fn = fn
        self.act_fn = act_fn
        self.norm = nn.GroupNorm(groups, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.act_fn(x)
        return self.fn(x)


# ----Residual Block----


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8, dropout=0.1) -> None:
        super().__init__()

        self.act_fn = nn.SiLU()
        self.dropout = nn.Dropout(p=dropout)

        self.conv1 = PreNorm(dim, nn.Conv2d(dim, dim_out, 3, padding=1), self.act_fn, groups=groups)
        self.conv2 = PreNorm(dim_out, nn.Conv2d(dim_out, dim_out, 3, padding=1), self.act_fn, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

        # For the time embedding
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
            if exists(time_emb_dim)
            else None
        )
    
    def forward(self, x, time_emb=None):
        h = self.conv1(x)

        # Add the time step embedding, if available.
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            h += time_emb
        
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.res_conv(x)


# ----U-Net----


class DDPMUnet(nn.Module):
    def __init__(
        self,
        dim: int,
        init_dim: Optional[int]=None,
        out_dim: Optional[int]=None,
        dim_mults: Tuple[int, ...]=(1, 2, 4, 8),
        attn_resolutions: Tuple[int, ...]=(16,),
        image_size: int=32,
        channels: int=3,
        resnet_block_groups: int=4,
        dropout: float=0.1,
        use_linear_attn: bool=True,
    ) -> None:
        super().__init__()

        # ----Setup----

        self.channels = channels

        init_dim = default(init_dim, dim)

        dims = [init_dim, *map(lambda m: init_dim * m, dim_mults)]
        resolutions = [image_size // (2 ** n) for n in range(len(dim_mults))]
        in_out = list(zip(dims[:-1], dims[1:], resolutions))

        out_dim = default(out_dim, channels)

        attn_cls = Attention
        if use_linear_attn:
            attn_cls = LinearAttention
        
        res_block = partial(ResnetBlock, groups=resnet_block_groups, dropout=dropout)

        # ----Handle time embeddings----
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # ----Downsample Layers----

        # Initial convolutional layer
        self.init_conv = nn.Conv2d(self.channels, init_dim, 3, padding=1)

        self.downsample_layers = nn.ModuleList([])
        num_resolutions = len(in_out)

        for idx, (dim_in, dim_out, image_size) in enumerate(in_out):
            is_last = idx >= (num_resolutions - 1)
            use_attn = image_size in attn_resolutions

            # Each downsample block consists of:
            #   - (ResnetBlock + pre-GroupNorm attention block (if using)) x 2
            #   - 2x Downsample operation x 1
            self.downsample_layers.append(
                nn.ModuleList(
                    [
                        res_block(dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, attn_cls(dim_out))) if use_attn else nn.Identity(),
                        res_block(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, attn_cls(dim_out))) if use_attn else nn.Identity(),
                        Downsample(dim_out, dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        # ----Bottleneck Layers----
        mid_dim = dims[-1]
        self.mid_block1 = res_block(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = res_block(mid_dim, mid_dim, time_emb_dim=time_dim)

        # ----Upsample Layers----

        self.upsample_layers = nn.ModuleList([])

        for idx, (dim_in, dim_out, image_size) in enumerate(reversed(in_out)):
            is_last = idx >= (num_resolutions) - 1
            use_attn = image_size in attn_resolutions

            # Each upsample block consists of:
            #   - (ResnetBlock + pre-GroupNorm attention block (if using)) x 2
            #   - 2x Upsample operation x 1
            self.upsample_layers.append(
                nn.ModuleList(
                    [
                        res_block(dim_out + dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, attn_cls(dim_out))) if use_attn else nn.Identity(),
                        res_block(dim_out + dim_out, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, attn_cls(dim_in))) if use_attn else nn.Identity(),
                        Upsample(dim_in, dim_in) if not is_last else nn.Identity()
                    ]
                )
            )
        
        # Final convolutional layer
        self.final_conv = PreNorm(
            dim,
            nn.Conv2d(dim, out_dim, 3, padding=1),
            act_fn=nn.SiLU(),
            groups=resnet_block_groups,
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # x should have initial shape (batch_size, num_channels, height, width)
        # t should have initial shape (batch_size, 1)
        x = self.init_conv(x)

        t = self.time_mlp(time)

        # Stack of intermediate downsample x values for shortcut connections
        # between the downsample and upsample layers at the same resolution.
        h = []

        # Downsample
        for block1, attn1, block2, attn2, downsample in self.downsample_layers:
            x = block1(x, t)
            x = attn1(x)
            # print(x.shape)
            h.append(x)

            x = block2(x, t)
            x = attn2(x)
            # print(x.shape)
            h.append(x)

            x = downsample(x)
        
        # Bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # Upsample
        for block1, attn1, block2, attn2, upsample in self.upsample_layers:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = attn1(x)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn2(x)

            x = upsample(x)
        
        # Final conv layer
        x = self.final_conv(x)

        return x








