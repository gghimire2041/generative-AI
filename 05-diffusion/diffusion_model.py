"""
Diffusion Model Implementation
Optimized for M4 Max MacBook Pro with Metal Performance Shaders

This module implements a comprehensive diffusion model with both DDPM and DDIM sampling,
featuring UNet architecture with attention mechanisms and optimized for Apple Silicon.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DiffusionConfig:
    """Configuration class for diffusion model parameters"""
    # Model architecture
    image_size: int = 64
    in_channels: int = 3
    model_channels: int = 128
    out_channels: int = 3
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (16, 8)
    dropout: float = 0.0
    channel_mult: Tuple[int, ...] = (1, 2, 2, 2)
    conv_resample: bool = True
    use_scale_shift_norm: bool = True
    
    # Diffusion process
    timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"  # "linear", "cosine", "quad"
    
    # Training
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
    gradient_accumulation_steps: int = 1
    
    # Optimization for M4 Max
    use_mps: bool = True
    mixed_precision: bool = True
    compile_model: bool = True


class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding layer for diffusion timesteps.
    Projects timestep to higher dimensional space for better conditioning.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.dim // 2
        
        # Create sinusoidal embeddings
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -embeddings)
        embeddings = timesteps[:, None].float() * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        
        if self.dim % 2 == 1:  # Zero pad if odd dimension
            embeddings = torch.nn.functional.pad(embeddings, (0, 1, 0, 0))
            
        return embeddings


class ResBlock(nn.Module):
    """
    Residual block with time conditioning and optional attention.
    Core building block of the UNet architecture.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        dropout: float = 0.0,
        use_scale_shift_norm: bool = True,
        use_attention: bool = False
    ):
        super().__init__()
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_attention = use_attention
        
        # First conv block
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels * (2 if use_scale_shift_norm else 1))
        )
        
        # Second conv block
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()
            
        # Attention layer
        if use_attention:
            self.attention = AttentionBlock(out_channels)
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        residual = self.residual_conv(x)
        
        # First conv block
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add time conditioning
        time_out = self.time_mlp(time_emb)[:, :, None, None]
        
        if self.use_scale_shift_norm:
            scale, shift = time_out.chunk(2, dim=1)
            h = h * (scale + 1) + shift
        else:
            h = h + time_out
        
        # Second conv block
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        # Residual connection
        h = h + residual
        
        # Apply attention if specified
        if self.use_attention:
            h = self.attention(h)
            
        return h


class AttentionBlock(nn.Module):
    """Multi-head self-attention block for spatial attention in diffusion models."""
    
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "Channels must be divisible by num_heads"
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x
        
        h = self.norm(x)
        qkv = self.qkv(h)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, num_heads, head_dim, H*W)
        
        # Compute attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.einsum('bhdi,bhdj->bhij', q, k) * scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        h = torch.einsum('bhij,bhdj->bhdi', attn, v)
        h = h.reshape(B, C, H, W)
        h = self.proj_out(h)
        
        return h + residual


class UNet(nn.Module):
    """
    UNet architecture for diffusion models with time conditioning.
    Optimized for Apple Silicon with efficient memory usage.
    """
    
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        
        # Time embedding
        time_embed_dim = config.model_channels * 4
        self.time_embed = nn.Sequential(
            TimeEmbedding(config.model_channels),
            nn.Linear(config.model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # Input projection
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(config.in_channels, config.model_channels, 3, padding=1)
        ])
        
        # Encoder blocks
        input_block_channels = [config.model_channels]
        ch = config.model_channels
        ds = 1
        
        for level, mult in enumerate(config.channel_mult):
            for _ in range(config.num_res_blocks):
                layers = [ResBlock(
                    ch, 
                    mult * config.model_channels,
                    time_embed_dim,
                    config.dropout,
                    config.use_scale_shift_norm,
                    use_attention=ds in config.attention_resolutions
                )]
                ch = mult * config.model_channels
                self.input_blocks.append(nn.Sequential(*layers))
                input_block_channels.append(ch)
                
            if level != len(config.channel_mult) - 1:
                # Downsample
                self.input_blocks.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                input_block_channels.append(ch)
                ds *= 2
        
        # Middle block
        self.middle_block = nn.Sequential(
            ResBlock(ch, ch, time_embed_dim, config.dropout, config.use_scale_shift_norm, True),
            ResBlock(ch, ch, time_embed_dim, config.dropout, config.use_scale_shift_norm, False)
        )
        
        # Decoder blocks
        self.output_blocks = nn.ModuleList([])
        
        for level, mult in list(enumerate(config.channel_mult))[::-1]:
            for i in range(config.num_res_blocks + 1):
                ich = input_block_channels.pop()
                layers = [ResBlock(
                    ch + ich,
                    mult * config.model_channels,
                    time_embed_dim,
                    config.dropout,
                    config.use_scale_shift_norm,
                    use_attention=ds in config.attention_resolutions
                )]
                ch = mult * config.model_channels
                
                if level and i == config.num_res_blocks:
                    # Upsample
                    layers.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
                    ds //= 2
                    
                self.output_blocks.append(nn.Sequential(*layers))
        
        # Output projection
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, config.out_channels, 3, padding=1)
        )
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Forward pass through UNet"""
        # Time embedding
        t_emb = self.time_embed(timesteps)
        
        # Encoder
        hs = []
        h = x
        for module in self.input_blocks:
            if isinstance(module, nn.Sequential) and len(module) > 0 and isinstance(module[0], ResBlock):
                h = module[0](h, t_emb)
            else:
                h = module(h)
            hs.append(h)
        
        # Middle
        if isinstance(self.middle_block[0], ResBlock):
            h = self.middle_block[0](h, t_emb)
            h = self.middle_block[1](h, t_emb)
        else:
            h = self.middle_block(h)
        
        # Decoder
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            if isinstance(module, nn.Sequential) and len(module) > 0 and isinstance(module[0], ResBlock):
                h = module[0](h, t_emb)
                if len(module) > 1:  # Has upsample
                    h = module[1](h)
            else:
                h = module(h)
        
        return self.out(h)


class GaussianDiffusion:
    """
    Gaussian diffusion process with DDPM and DDIM sampling.
    Implements the forward and reverse diffusion processes.
    """
    
    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.timesteps = config.timesteps
        
        # Create noise schedule
        self.betas = self._get_beta_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_variance = posterior_variance
        self.posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def _get_beta_schedule(self) -> torch.Tensor:
        """Create noise schedule"""
        if self.config.beta_schedule == "linear":
            return torch.linspace(self.config.beta_start, self.config.beta_end, self.timesteps)
        elif self.config.beta_schedule == "cosine":
            return self._cosine_beta_schedule()
        elif self.config.beta_schedule == "quad":
            return torch.linspace(self.config.beta_start**0.5, self.config.beta_end**0.5, self.timesteps)**2
        else:
            raise ValueError(f"Unknown beta schedule: {self.config.beta_schedule}")
    
    def _cosine_beta_schedule(self) -> torch.Tensor:
        """Cosine noise schedule as proposed in improved DDPM"""
        def alpha_bar(time_step):
            return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2
        
        betas = []
        for i in range(self.timesteps):
            t1 = i / self.timesteps
            t2 = (i + 1) / self.timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
        return torch.tensor(betas)
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample from q(x_t | x_0) - forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def q_posterior_mean_variance(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        """Compute mean and variance of q(x_{t-1} | x_t, x_0)"""
        posterior_mean_coef1_t = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        posterior_mean_coef2_t = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        posterior_mean = posterior_mean_coef1_t * x_start + posterior_mean_coef2_t * x_t
        
        posterior_variance_t = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_t = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        return posterior_mean, posterior_variance_t, posterior_log_variance_t
    
    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Predict x_0 from x_t and predicted noise"""
        sqrt_recip_alphas_cumprod_t = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise
    
    def p_mean_variance(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor):
        """Apply the model to get p(x_{t-1} | x_t)"""
        model_output = model(x, t)
        
        # Predict x_0
        x_recon = self.predict_start_from_noise(x, t, model_output)
        x_recon = torch.clamp(x_recon, -1.0, 1.0)  # Clip to valid range
        
        # Get posterior mean and variance
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(x_recon, x, t)
        
        return model_mean, posterior_variance, posterior_log_variance, x_recon
    
    def p_sample(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample x_{t-1} from the model"""
        if noise is None:
            noise = torch.randn_like(x)
        
        model_mean, _, model_log_variance, _ = self.p_mean_variance(model, x, t)
        
        # No noise when t == 0
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x.shape) - 1))))
        
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
    
    def ddim_sample(
        self, 
        model: nn.Module, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        t_next: torch.Tensor,
        eta: float = 0.0
    ) -> torch.Tensor:
        """DDIM sampling step"""
        alpha_prod_t = self._extract(self.alphas_cumprod, t, x.shape)
        alpha_prod_t_next = self._extract(self.alphas_cumprod, t_next, x.shape)
        
        # Predict noise
        pred_noise = model(x, t)
        
        # Predict x_0
        x_0_pred = (x - torch.sqrt(1 - alpha_prod_t) * pred_noise) / torch.sqrt(alpha_prod_t)
        x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)
        
        # DDIM step
        variance = eta * torch.sqrt((1 - alpha_prod_t_next) / (1 - alpha_prod_t)) * torch.sqrt(1 - alpha_prod_t / alpha_prod_t_next)
        
        mean = torch.sqrt(alpha_prod_t_next) * x_0_pred + torch.sqrt(1 - alpha_prod_t_next - variance**2) * pred_noise
        
        if eta > 0:
            noise = torch.randn_like(x)
            return mean + variance * noise
        else:
            return mean
    
    def training_losses(self, model: nn.Module, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """Compute training losses"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_t = self.q_sample(x_start, t, noise)
        model_output = model(x_t, t)
        
        # Simple MSE loss on predicted noise
        loss = F.mse_loss(model_output, noise)
        
        return loss
    
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
        """Extract values from tensor a at indices t and reshape for broadcasting"""
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class DiffusionModel:
    """
    Main diffusion model class that combines UNet and Gaussian diffusion.
    Optimized for Apple Silicon M4 Max.
    """
    
    def __init__(self, config: DiffusionConfig):
        self.config = config
        
        # Set up device - prioritize MPS for M4 Max
        if config.use_mps and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using MPS (Metal Performance Shaders) for Apple Silicon")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Using CUDA")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU")
        
        # Initialize model and diffusion
        self.model = UNet(config).to(self.device)
        self.diffusion = GaussianDiffusion(config)
        
        # Move diffusion parameters to device
        self._move_diffusion_to_device()
        
        # Compile model for better performance on M4 Max
        if config.compile_model and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model)
                logger.info("Model compiled for optimized performance")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
            eps=1e-8
        )
        
        # Mixed precision training for M4 Max
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
    
    def _move_diffusion_to_device(self):
        """Move diffusion parameters to the correct device"""
        for attr in ['betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev',
                     'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod',
                     'log_one_minus_alphas_cumprod', 'sqrt_recip_alphas_cumprod',
                     'sqrt_recipm1_alphas_cumprod', 'posterior_variance',
                     'posterior_log_variance_clipped', 'posterior_mean_coef1',
                     'posterior_mean_coef2']:
            if hasattr(self.diffusion, attr):
                setattr(self.diffusion, attr, getattr(self.diffusion, attr).to(self.device))
    
    def train_step(self, batch: torch.Tensor) -> float:
        """Single training step"""
        self.model.train()
        
        batch = batch.to(self.device)
        batch_size = batch.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.config.timesteps, (batch_size,), device=self.device)
        
        # Training step with mixed precision
        if self.scaler:
            with torch.autocast(device_type=str(self.device).split(':')[0], dtype=torch.float16):
                loss = self.diffusion.training_losses(self.model, batch, t)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss = self.diffusion.training_losses(self.model, batch, t)
            loss.backward()
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        
        return loss.item()
    
    @torch.no_grad()
    def sample(
        self, 
        batch_size: int = 1,
        shape: Optional[Tuple[int, ...]] = None,
        num_steps: Optional[int] = None,
        method: str = "ddpm",
        eta: float = 0.0
    ) -> torch.Tensor:
        """Sample from the model"""
        self.model.eval()
        
        if shape is None:
            shape = (self.config.in_channels, self.config.image_size, self.config.image_size)
        
        if num_steps is None:
            num_steps = self.config.timesteps
        
        # Start from random noise
        x = torch.randn(batch_size, *shape, device=self.device)
        
        if method == "ddpm":
            return self._ddpm_sample(x, num_steps)
        elif method == "ddim":
            return self._ddim_sample(x, num_steps, eta)
        else:
            raise ValueError(f"Unknown sampling method: {method}")
    
    def _ddpm_sample(self, x: torch.Tensor, num_steps: int) -> torch.Tensor:
        """DDPM sampling loop"""
        timesteps = torch.arange(num_steps - 1, -1, -1, device=self.device)
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)
            x = self.diffusion.p_sample(self.model, x, t_batch)
            
            if i % 100 == 0:
                logger.info(f"Sampling step {i}/{num_steps}")
        
        return x
    
    def _ddim_sample(self, x: torch.Tensor, num_steps: int, eta: float) -> torch.Tensor:
        """DDIM sampling loop"""
        # Create sampling schedule
        c = self.config.timesteps // num_steps
        timesteps = torch.arange(0, self.config.timesteps, c, device=self.device)
        timesteps = torch.flip(timesteps, [0])
        
        for i in range(len(timesteps)):
            t = timesteps[i]
            t_next = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(-1, device=self.device)
            
            t_batch = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)
            t_next_batch = torch.full((x.shape[0],), max(t_next, 0), device=self.device, dtype=torch.long)
            
            x = self.diffusion.ddim_sample(self.model, x, t_batch, t_next_batch, eta)
            
            if i % 10 == 0:
                logger.info(f"DDIM sampling step {i}/{len(timesteps)}")
        
        return x
    
    def save_checkpoint(self, filepath: str, epoch: int, loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config,
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"Checkpoint loaded from {filepath}")
        return checkpoint['epoch'], checkpoint['loss']
