import math
from dataclasses import dataclass

import torch
from torch import nn



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.dim = dim

    def forward(self,
                pos: int | torch.Tensor,
                dim: int = 256,
                base: float = 10000.0,
        ) -> torch.Tensor:
        """
        Compute a sinusoidal positional embedding for a single position.

        Args:
        pos   : int or 0-D tensor – position index.
        dim   : int – embedding dimension (must be even).
        base  : float – base for the exponential frequency scaling.

        Returns:
        Tensor of shape (dim,) containing the positional embedding.
        """
        pos = torch.as_tensor(pos, dtype=torch.float)        # if pos was int
        if pos.dim() == 0:                                   # scalar → (1,)
            pos = pos.unsqueeze(0)

        # Half of the dimensions will be sine, half cosine
        half_dim = dim // 2

        # Exponent term:  base^{2i/dim}  for i = 0 .. half_dim-1
        exponent = torch.arange(half_dim, dtype=torch.float)
        div_term = base ** (2 * exponent / dim).to("cuda")  # (half_dim,)

        # Compute the value: pos / base^{2i/dim}
        value = pos.unsqueeze(-1) / div_term                    # (num_pos, half_dim)

        emb = torch.empty(pos.size(0), dim, dtype=torch.float, device="cuda")
        emb[:, 0::2] = torch.sin(value)              # even indices  -> sin
        emb[:, 1::2] = torch.cos(value)              # odd  indices -> cos
        return emb

from typing import Union


class Downsample1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1DBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, ker_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, ker_size, padding=ker_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, ker_size):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1DBlock(in_channels, out_channels, ker_size),
            Conv1DBlock(out_channels, out_channels, ker_size),
        ])

        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, condition):
        first_out = self.blocks[0](x)

        embedded_condition = self.cond_encoder(condition) # [B, 2 * C_out, 1]

        embedded_condition = embedded_condition.reshape(embedded_condition.shape[0],
                                                      2,
                                                      self.out_channels,
                                                      1)
        scale = embedded_condition[:, 0, ...]
        bias = embedded_condition[:, 1, ...]

        # apply FILM

        out = first_out * scale + bias

        second_out = self.blocks[1](out)

        out = second_out + self.residual_conv(x)

        return out

class ConditionalUnet1D(nn.Module):
    def __init__(self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=5,
        ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPositionalEmbedding(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ResidualBlock(
                mid_dim, mid_dim, cond_dim=cond_dim,
                ker_size=kernel_size
            ),
            ResidualBlock(
                mid_dim, mid_dim, cond_dim=cond_dim,
                ker_size=kernel_size
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ResidualBlock(
                    dim_in, dim_out, cond_dim=cond_dim,
                    ker_size=kernel_size),
                ResidualBlock(
                    dim_out, dim_out, cond_dim=cond_dim,
                    ker_size=kernel_size),
                Downsample1D(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ResidualBlock(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    ker_size=kernel_size),
                ResidualBlock(
                    dim_in, dim_in, cond_dim=cond_dim,
                    ker_size=kernel_size),
                Upsample1D(dim_in) if not is_last else nn.Identity()
            ]))

        final_conv = nn.Sequential(
            Conv1DBlock(start_dim, start_dim, ker_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            global_cond=None):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1,-2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1,-2)
        # (B,T,C)
        return x


class ResidualBlockTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, ker_size):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1DBlock(in_channels, out_channels, ker_size),
            Conv1DBlock(out_channels, out_channels, ker_size),
        ])

        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        self.attention_layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=out_channels,
                                       nhead=4),
            num_layers=1,
        )

        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, condition):
        first_out = self.blocks[0](x) # [B, c_out, T]

        embedded_condition = self.cond_encoder(condition) # [B, 2 * C_out, 1]

        embedded_condition = embedded_condition.reshape(embedded_condition.shape[0],
                                                      2,
                                                      self.out_channels,
                                                      1)
        scale = embedded_condition[:, 0, ...]
        bias = embedded_condition[:, 1, ...]

        # apply FILM

        out = first_out * scale + bias

        # attention

        attn = self.attention_layer(out.permute(2, 0, 1)).permute(1, 2, 0) # receive [T, B, c_out] -> return [B, c_out, T]

        out = out + attn

        second_out = self.blocks[1](out)

        # skip connection

        out = second_out + self.residual_conv(x)

        return out

class ConditionalUnet1DTransformer(nn.Module):
    def __init__(self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[192, 384, 512, 640],
        kernel_size=5,
        ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPositionalEmbedding(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ResidualBlockTransformer(
                mid_dim, mid_dim, cond_dim=cond_dim,
                ker_size=kernel_size
            ),
            ResidualBlockTransformer(
                mid_dim, mid_dim, cond_dim=cond_dim,
                ker_size=kernel_size
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ResidualBlockTransformer(
                    dim_in, dim_out, cond_dim=cond_dim,
                    ker_size=kernel_size),
                ResidualBlockTransformer(
                    dim_out, dim_out, cond_dim=cond_dim,
                    ker_size=kernel_size),
                Downsample1D(dim_out) if not is_last else nn.Identity()
            ]))


        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ResidualBlockTransformer(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    ker_size=kernel_size),
                ResidualBlockTransformer(
                    dim_in, dim_in, cond_dim=cond_dim,
                    ker_size=kernel_size),
                Upsample1D(dim_in) if not is_last else nn.Identity()
            ]))

        final_conv = nn.Sequential(
            Conv1DBlock(start_dim, start_dim, ker_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            global_cond=None):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1,-2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1,-2)
        # (B,T,C)
        return x


def cosine_betas(T: int, max_beta: float = 0.999, device="cpu") -> torch.Tensor:
    """Cosine schedule from Nichol & Dhariwal (2021)."""
    s = 0.008
    steps = torch.arange(T + 1, dtype=torch.float32, device=device)
    alphas_bar = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
    betas = 1 - alphas_bar[1:] / alphas_bar[:-1]
    return betas.clamp(max=max_beta)


@dataclass
class Config:
    T: int = 1000  # number of training timesteps
    device: str = "cpu"

class CosineDDPMScheduler:
    """Minimal DDPM scheduler (cosine β schedule, ε-prediction only)."""

    def __init__(self, cfg: Config):
        self.config = cfg

        self.device = torch.device(cfg.device)
        betas = cosine_betas(cfg.T, device=self.device)

        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        # Pre-compute frequently used terms
        self.betas = betas
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)

        # Posterior variance for p(x_{t-1}|x_t,x_0) (original DDPM formula)
        alpha_bar_prev = torch.cat([torch.ones(1, device=self.device), alpha_bar[:-1]])
        self.posterior_var = (
                betas * (1 - alpha_bar_prev) / (1 - alpha_bar)
        )

        # default inference schedule = all steps reversed
        self.timesteps = torch.arange(cfg.T - 1, -1, -1, device=self.device)

    # ---------- public API --------------------------------------------------

    @torch.no_grad()
    def add_noise(self, x0: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """q(x_t | x_0) = sqrt(ᾱ_t)·x0 + sqrt(1-ᾱ_t)·eps"""
        sqrt_ab = self.sqrt_alpha_bar[t].view(-1, 1, 1)
        sqrt_om = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
        return sqrt_ab * x0 + sqrt_om * eps

    @torch.no_grad()
    def step(self, eps_pred: torch.Tensor, t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        """
        Reverse-step pθ(x_{t-1}|x_t) для косинусного DDPM.
        t: (B,) long      x_t: (B, C, H, W)
        """
        beta_t = self.betas[t].view(-1, 1, 1, 1)  # β_t
        sqrt_ab_t = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)  # √ᾱ_t
        sqrt_omab_t = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)

        # t  shape (B,)  — все t ≥ 1
        sqrt_ab_prev = self.sqrt_alpha_bar[t - 1].view(-1, 1, 1, 1)  # (B,1,1,1)
        alpha_bar_prev = sqrt_ab_prev ** 2

        x0_pred = (x_t - sqrt_omab_t * eps_pred) / sqrt_ab_t  # Eq.(15)


        alpha_t = 1.0 - beta_t  # α_t

        alpha_bar_t = sqrt_ab_t ** 2  # ᾱ_t
        coef1 = (sqrt_ab_prev * beta_t) / (1.0 - alpha_bar_prev + 1e-8)
        coef2 = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t + 1e-8)

        # Eq.(7)
        mean = coef1 * x0_pred + coef2 * x_t

        var = self.posterior_var[t].view(-1, 1, 1, 1)
        noise = torch.randn_like(x_t) * (t > 0).float().view(-1, 1, 1, 1)

        return mean + torch.sqrt(var) * noise  # x_{t-1}


