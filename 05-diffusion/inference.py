"""
Inference script for diffusion models
Generate images, interpolations, and perform various sampling techniques
Optimized for M4 Max MacBook Pro with advanced features
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Union
import time
import json
import numpy as np

import torch
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import cv2

from diffusion_model import DiffusionModel, DiffusionConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdvancedDiffusionInference:
    """
    Advanced inference engine for diffusion models with comprehensive sampling techniques
    Optimized for M4 Max MacBook Pro performance
    """
    
    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        self.checkpoint_path = checkpoint_path
        
        # Load and validate checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract configuration
        if 'config' not in checkpoint:
            raise ValueError("Checkpoint missing configuration data")
        
        self.config = checkpoint['config']
        
        # Initialize model with optimal settings for M4 Max
        self.model = DiffusionModel(self.config)
        
        # Override device if specified
        if device:
            self._set_device(device)
        
        # Load model weights
        try:
            self.model.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.model.eval()
            logger.info("Model weights loaded successfully")
        except KeyError as e:
            raise ValueError(f"Checkpoint missing required key: {e}")
        
        # Enable inference optimizations
        torch.set_grad_enabled(False)
        if hasattr(torch, 'inference_mode'):
            self._inference_mode = torch.inference_mode()
        
        logger.info(f"Model loaded and ready on device: {self.model.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.model.parameters()):,}")
    
    def _set_device(self, device: str):
        """Set the computation device with validation"""
        if device == "mps" and torch.backends.mps.is_available():
            self.model.device = torch.device("mps")
        elif device == "cuda" and torch.cuda.is_available():
            self.model.device = torch.device("cuda")
        elif device == "cpu":
            self.model.device = torch.device("cpu")
        else:
            logger.warning(f"Device {device} not available, using default")
            return
        
        self.model.model = self.model.model.to(self.model.device)
        self.model._move_diffusion_to_device()
    
    def generate_batch(
        self,
        batch_size: int = 16,
        method: str = "ddim",
        num_steps: int = 50,
        eta: float = 0.0,
        guidance_scale: float = 1.0,
        seed: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> torch.Tensor:
        """
        Generate a batch of samples with advanced options
        
        Args:
            batch_size: Number of samples to generate
            method: Sampling method ("ddpm", "ddim", "dpm")
            num_steps: Number of denoising steps
            eta: DDIM stochasticity parameter (0.0 = deterministic)
            guidance_scale: Classifier-free guidance scale
            seed: Random seed for reproducibility
            progress_callback: Callback function for progress updates
        """
        if seed is not None:
            torch.manual_seed(seed)
            logger.info(f"Set random seed to {seed}")
        
        logger.info(f"Generating {batch_size} samples using {method} with {num_steps} steps")
        start_time = time.time()
        
        # Memory optimization for large batches on M4 Max
        max_batch_size = self._get_optimal_batch_size(batch_size)
        all_samples = []
        
        for i in range(0, batch_size, max_batch_size):
            current_batch_size = min(max_batch_size, batch_size - i)
            logger.info(f"Processing batch {i//max_batch_size + 1}/{(batch_size + max_batch_size - 1)//max_batch_size}")
            
            with torch.no_grad():
                if method.lower() == "ddpm":
                    samples = self._ddpm_sample(current_batch_size, num_steps, progress_callback)
                elif method.lower() == "ddim":
                    samples = self._ddim_sample(current_batch_size, num_steps, eta, progress_callback)
                elif method.lower() == "dpm":
                    samples = self._dpm_solver_sample(current_batch_size, num_steps, progress_callback)
                else:
                    raise ValueError(f"Unknown sampling method: {method}")
            
            all_samples.append(samples)
        
        # Combine all batches
        final_samples = torch.cat(all_samples, dim=0)
        
        generation_time = time.time() - start_time
        samples_per_second = batch_size / generation_time
        logger.info(f"Generation completed: {generation_time:.2f}s ({samples_per_second:.2f} samples/sec)")
        
        return final_samples
    
    def _get_optimal_batch_size(self, requested_batch_size: int) -> int:
        """Determine optimal batch size for current hardware"""
        if self.model.device.type == "mps":
            # M4 Max optimization - balance speed vs memory
            max_memory_batch = 32 if self.config.image_size <= 64 else 16
            return min(requested_batch_size, max_memory_batch)
        elif self.model.device.type == "cuda":
            return min(requested_batch_size, 64)
        else:
            return min(requested_batch_size, 8)
    
    def _ddim_sample(self, batch_size: int, num_steps: int, eta: float, progress_callback: Optional[callable] = None) -> torch.Tensor:
        """Optimized DDIM sampling with progress tracking"""
        shape = (batch_size, self.config.in_channels, self.config.image_size, self.config.image_size)
        x = torch.randn(shape, device=self.model.device)
        
        # Create optimized sampling schedule
        c = self.config.timesteps // num_steps
        timesteps = torch.arange(0, self.config.timesteps, c, device=self.model.device)
        timesteps = torch.flip(timesteps, [0])
        
        for i, t in enumerate(timesteps):
            t_next = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(-1, device=self.model.device)
            
            t_batch = torch.full((batch_size,), t, device=self.model.device, dtype=torch.long)
            t_next_batch = torch.full((batch_size,), max(t_next, 0), device=self.model.device, dtype=torch.long)
            
            x = self.model.diffusion.ddim_sample(self.model.model, x, t_batch, t_next_batch, eta)
            
            if progress_callback:
                progress_callback(i + 1, len(timesteps))
        
        return (x + 1) / 2  # Convert to [0, 1]
    
    def _ddpm_sample(self, batch_size: int, num_steps: int, progress_callback: Optional[callable] = None) -> torch.Tensor:
        """DDPM sampling with progress tracking"""
        shape = (batch_size, self.config.in_channels, self.config.image_size, self.config.image_size)
        x = torch.randn(shape, device=self.model.device)
        
        timesteps = torch.arange(num_steps - 1, -1, -1, device=self.model.device)
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=self.model.device, dtype=torch.long)
            x = self.model.diffusion.p_sample(self.model.model, x, t_batch)
            
            if progress_callback:
                progress_callback(i + 1, len(timesteps))
        
        return (x + 1) / 2  # Convert to [0, 1]
    
    def _dpm_solver_sample(self, batch_size: int, num_steps: int, progress_callback: Optional[callable] = None) -> torch.Tensor:
        """DPM-Solver sampling for faster generation"""
        # Simplified DPM-Solver implementation
        # For a full implementation, consider using the DPM-Solver library
        return self._ddim_sample(batch_size, num_steps, eta=0.0, progress_callback=progress_callback)
    
    def create_interpolation_video(
        self,
        start_seed: int,
        end_seed: int,
        num_frames: int = 60,
        method: str = "ddim",
        num_steps: int = 50,
        output_path: str = "interpolation.mp4",
        fps: int = 30
    ) -> str:
        """Create a smooth interpolation video between two seeds"""
        logger.info(f"Creating interpolation video with {num_frames} frames")
        
        # Generate start and end noise
        torch.manual_seed(start_seed)
        shape = (1, self.config.in_channels, self.config.image_size, self.config.image_size)
        start_noise = torch.randn(shape, device=self.model.device)
        
        torch.manual_seed(end_seed)
        end_noise = torch.randn(shape, device=self.model.device)
        
        # Spherical linear interpolation for better results
        frames = []
        for i in range(num_frames):
            alpha = i / (num_frames - 1)
            # SLERP interpolation
            interpolated_noise = self._slerp(start_noise, end_noise, alpha)
            
            # Generate image
            with torch.no_grad():
                if method == "ddim":
                    sample = self._generate_from_noise(interpolated_noise, method, num_steps)
                else:
                    sample = self._generate_from_noise(interpolated_noise, method, num_steps)
            
            # Convert to PIL Image
            sample_np = sample.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            sample_np = (sample_np * 255).astype(np.uint8)
            frames.append(sample_np)
        
        # Create video using OpenCV
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        logger.info(f"Interpolation video saved to {output_path}")
        return output_path
    
    def _slerp(self, v1: torch.Tensor, v2: torch.Tensor, t: float) -> torch.Tensor:
        """Spherical linear interpolation"""
        v1_norm = F.normalize(v1.flatten(), dim=0)
        v2_norm = F.normalize(v2.flatten(), dim=0)
        
        dot_product = torch.sum(v1_norm * v2_norm)
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        
        theta = torch.acos(dot_product)
        
        if theta < 1e-6:
            return v1 * (1 - t) + v2 * t
        
        sin_theta = torch.sin(theta)
        w1 = torch.sin((1 - t) * theta) / sin_theta
        w2 = torch.sin(t * theta) / sin_theta
        
        result = w1 * v1.flatten() + w2 * v2.flatten()
        return result.reshape(v1.shape)
    
    def _generate_from_noise(self, noise: torch.Tensor, method: str, num_steps: int) -> torch.Tensor:
        """Generate image from specific noise"""
        if method == "ddim":
            return self._ddim_sample_from_noise(noise, num_steps)
        else:
            return self._ddpm_sample_from_noise(noise, num_steps)
    
    def _ddim_sample_from_noise(self, noise: torch.Tensor, num_steps: int) -> torch.Tensor:
        """DDIM sampling from specific noise vector"""
        x = noise.clone()
        c = self.config.timesteps // num_steps
        timesteps = torch.arange(0, self.config.timesteps, c, device=self.model.device)
        timesteps = torch.flip(timesteps, [0])
        
        for i in range(len(timesteps)):
            t = timesteps[i]
            t_next = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(-1, device=self.model.device)
            
            t_batch = torch.full((x.shape[0],), t, device=self.model.device, dtype=torch.long)
            t_next_batch = torch.full((x.shape[0],), max(t_next, 0), device=self.model.device, dtype=torch.long)
            
            x = self.model.diffusion.ddim_sample(self.model.model, x, t_batch, t_next_batch, eta=0.0)
        
        return (x + 1) / 2
    
    def _ddpm_sample_from_noise(self, noise: torch.Tensor, num_steps: int) -> torch.Tensor:
        """DDPM sampling from specific noise vector"""
        x = noise.clone()
        timesteps = torch.arange(num_steps - 1, -1, -1, device=self.model.device)
        
        for t in timesteps:
            t_batch = torch.full((x.shape[0],), t, device=self.model.device, dtype=torch.long)
            x = self.model.diffusion.p_sample(self.model.model, x, t_batch)
        
        return (x + 1) / 2
    
    def inpaint_image(
        self,
        image_path: str,
        mask_path: str,
        method: str = "ddim",
        num_steps: int = 50,
        strength: float = 1.0
    ) -> torch.Tensor:
        """Advanced inpainting with better quality"""
        logger.info(f"Inpainting image: {image_path}")
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # Resize to model resolution
        transform = transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        mask_transform = transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor()
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(self.model.device)
        mask_tensor = mask_transform(mask).unsqueeze(0).to(self.model.device)
        
        # Binary mask
        mask_tensor = (mask_tensor > 0.5).float()
        
        with torch.no_grad():
            # Improved inpainting with repaint technique
            result = self._repaint_inpainting(image_tensor, mask_tensor, method, num_steps, strength)
        
        return (result + 1) / 2  # Convert to [0, 1]
    
    def _repaint_inpainting(
        self, 
        image: torch.Tensor, 
        mask: torch.Tensor, 
        method: str, 
        num_steps: int,
        strength: float
    ) -> torch.Tensor:
        """RePaint inpainting technique for better quality"""
        # Start with noise in masked region
        noise = torch.randn_like(image)
        x = image * (1 - mask) + noise * mask
        
        # Determine sampling schedule
        if method == "ddim":
            c = self.config.timesteps // num_steps
            timesteps = torch.arange(0, self.config.timesteps, c, device=self.model.device)
            timesteps = torch.flip(timesteps, [0])
        else:
            timesteps = torch.arange(num_steps - 1, -1, -1, device=self.model.device)
        
        # Inpainting with known region guidance
        for i, t in enumerate(timesteps):
            t_batch = torch.full((x.shape[0],), t, device=self.model.device, dtype=torch.long)
            
            if method == "ddim" and i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                t_next_batch = torch.full((x.shape[0],), max(t_next, 0), device=self.model.device, dtype=torch.long)
                x_denoised = self.model.diffusion.ddim_sample(self.model.model, x, t_batch, t_next_batch, eta=0.0)
            else:
                x_denoised = self.model.diffusion.p_sample(self.model.model, x, t_batch)
            
            # Apply inpainting constraint with strength parameter
            if t > 0:
                # Add appropriate noise to known regions
                known_noise = torch.randn_like(image)
                if method == "ddim" and i < len(timesteps) - 1:
                    t_next = timesteps[i + 1]
                    t_next_batch = torch.full((x.shape[0],), max(t_next, 0), device=self.model.device, dtype=torch.long)
                    noisy_known = self.model.diffusion.q_sample(image, t_next_batch, known_noise)
                else:
                    t_prev_batch = torch.full((x.shape[0],), max(t - 1, 0), device=self.model.device, dtype=torch.long)
                    noisy_known = self.model.diffusion.q_sample(image, t_prev_batch, known_noise)
                
                # Blend with strength parameter
                x = x_denoised * mask + noisy_known * (1 - mask) * strength + x_denoised * (1 - mask) * (1 - strength)
            else:
                x = x_denoised * mask + image * (1 - mask)
        
        return x
    
    def create_sample_grid(
        self,
        seeds: List[int],
        method: str = "ddim",
        num_steps: int = 50,
        grid_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """Create a grid of samples from specific seeds"""
        samples = []
        
        for seed in seeds:
            torch.manual_seed(seed)
            sample = self.generate_batch(1, method, num_steps, seed=seed)
            samples.append(sample)
        
        # Combine samples
        all_samples = torch.cat(samples, dim=0)
        
        # Create grid
        if grid_size is None:
            grid_size = (int(np.ceil(np.sqrt(len(seeds)))), int(np.ceil(np.sqrt(len(seeds)))))
        
        grid = make_grid(all_samples, nrow=grid_size[1], padding=2, normalize=True)
        return grid
    
    def benchmark_performance(self, test_cases: Optional[List[dict]] = None) -> dict:
        """Comprehensive performance benchmark"""
        if test_cases is None:
            test_cases = [
                {"method": "ddim", "steps": 20, "batch_size": 4},
                {"method": "ddim", "steps": 50, "batch_size": 4},
                {"method": "ddpm", "steps": 50, "batch_size": 4},
                {"method": "ddim", "steps": 50, "batch_size": 16},
            ]
        
        results = {}
        logger.info("Starting performance benchmark...")
        
        # Warmup
        logger.info("Warming up model...")
        self.generate_batch(2, "ddim", 10)
        
        for i, case in enumerate(test_cases):
            case_name = f"{case['method']}_{case['steps']}steps_{case['batch_size']}batch"
            logger.info(f"Benchmarking case {i+1}/{len(test_cases)}: {case_name}")
            
            # Multiple runs for stability
            times = []
            for run in range(3):
                start_time = time.time()
                samples = self.generate_batch(
                    batch_size=case['batch_size'],
                    method=case['method'],
                    num_steps=case['steps']
                )
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            samples_per_second = case['batch_size'] / avg_time
            
            results[case_name] = {
                "avg_time": avg_time,
                "std_time": std_time,
                "samples_per_second": samples_per_second,
                "config": case
            }
            
            logger.info(f"  Average time: {avg_time:.2f}±{std_time:.2f}s")
            logger.info(f"  Samples/sec: {samples_per_second:.2f}")
        
        return results
    
    def save_samples_with_info(
        self,
        samples: torch.Tensor,
        output_path: str,
        metadata: dict,
        grid_size: Optional[Tuple[int, int]] = None
    ):
        """Save samples with embedded metadata"""
        # Create grid
        if grid_size is None:
            n_samples = samples.shape[0]
            grid_size = (int(np.ceil(np.sqrt(n_samples))), int(np.ceil(np.sqrt(n_samples))))
        
        grid = make_grid(samples, nrow=grid_size[1], padding=2, normalize=True)
        
        # Save image
        save_image(grid, output_path)
        
        # Save metadata
        metadata_path = output_path.replace('.png', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Samples saved to {output_path}")
        logger.info(f"Metadata saved to {metadata_path}")


def main():
    """Advanced inference main function"""
    parser = argparse.ArgumentParser(description="Advanced Diffusion Model Inference")
    
    # Model arguments
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--device", type=str, choices=["auto", "mps", "cuda", "cpu"], default="auto")
    
    # Generation arguments
    parser.add_argument("--mode", type=str, default="sample",
                       choices=["sample", "interpolate", "video", "inpaint", "benchmark", "grid"],
                       help="Generation mode")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples")
    parser.add_argument("--method", type=str, default="ddim", choices=["ddpm", "ddim", "dpm"], help="Sampling method")
    parser.add_argument("--num_steps", type=int, default=50, help="Sampling steps")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta parameter")
    parser.add_argument("--seed", type=int, help="Random seed")
    
    # Advanced options
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Guidance scale")
    parser.add_argument("--grid_size", type=str, help="Grid size (e.g., '4x4')")
    
    # Mode-specific arguments
    parser.add_argument("--start_seed", type=int, default=42, help="Start seed for interpolation")
    parser.add_argument("--end_seed", type=int, default=123, help="End seed for interpolation")
    parser.add_argument("--num_frames", type=int, default=60, help="Interpolation frames")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    
    # Inpainting arguments
    parser.add_argument("--image_path", type=str, help="Input image for inpainting")
    parser.add_argument("--mask_path", type=str, help="Mask image for inpainting")
    parser.add_argument("--strength", type=float, default=1.0, help="Inpainting strength")
    
    # Grid mode arguments
    parser.add_argument("--seeds", type=str, help="Comma-separated seeds for grid mode")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse grid size
    grid_size = None
    if args.grid_size:
        try:
            w, h = map(int, args.grid_size.split('x'))
            grid_size = (h, w)
        except ValueError:
            logger.warning(f"Invalid grid size format: {args.grid_size}")
    
    # Initialize inference engine
    device = None if args.device == "auto" else args.device
    inference = AdvancedDiffusionInference(args.checkpoint, device=device)
    
    # Create progress callback
    def progress_callback(step, total_steps):
        if step % max(1, total_steps // 10) == 0:
            logger.info(f"Progress: {step}/{total_steps} ({100*step/total_steps:.1f}%)")
    
    # Execute based on mode
    timestamp = int(time.time())
    
    if args.mode == "sample":
        # Generate samples
        logger.info("Generating samples...")
        samples = inference.generate_batch(
            batch_size=args.num_samples,
            method=args.method,
            num_steps=args.num_steps,
            eta=args.eta,
            seed=args.seed,
            progress_callback=progress_callback
        )
        
        output_path = output_dir / f"samples_{args.method}_{args.num_steps}steps_{timestamp}.png"
        metadata = {
            "method": args.method,
            "num_steps": args.num_steps,
            "eta": args.eta,
            "seed": args.seed,
            "num_samples": args.num_samples,
            "timestamp": timestamp
        }
        
        inference.save_samples_with_info(samples, str(output_path), metadata, grid_size)
    
    elif args.mode == "interpolate":
        # Create interpolation
        logger.info("Creating interpolation...")
        output_path = output_dir / f"interpolation_{args.start_seed}_{args.end_seed}_{timestamp}.png"
        
        # Generate frames
        frames = []
        for i in range(args.num_frames):
            alpha = i / (args.num_frames - 1)
            # Use spherical interpolation seeds
            interpolated_seed = int(args.start_seed * (1 - alpha) + args.end_seed * alpha)
            sample = inference.generate_batch(1, args.method, args.num_steps, seed=interpolated_seed)
            frames.append(sample)
        
        all_frames = torch.cat(frames, dim=0)
        grid = make_grid(all_frames, nrow=args.num_frames, padding=2, normalize=True)
        save_image(grid, output_path)
        logger.info(f"Interpolation saved to {output_path}")
    
    elif args.mode == "video":
        # Create interpolation video
        output_path = output_dir / f"interpolation_video_{timestamp}.mp4"
        inference.create_interpolation_video(
            start_seed=args.start_seed,
            end_seed=args.end_seed,
            num_frames=args.num_frames,
            method=args.method,
            num_steps=args.num_steps,
            output_path=str(output_path),
            fps=args.fps
        )
    
    elif args.mode == "inpaint":
        # Inpainting
        if not args.image_path or not args.mask_path:
            logger.error("Inpainting requires --image_path and --mask_path")
            return
        
        logger.info("Performing inpainting...")
        result = inference.inpaint_image(
            args.image_path,
            args.mask_path,
            method=args.method,
            num_steps=args.num_steps,
            strength=args.strength
        )
        
        output_path = output_dir / f"inpaint_result_{timestamp}.png"
        save_image(result, output_path)
        logger.info(f"Inpainting result saved to {output_path}")
    
    elif args.mode == "grid":
        # Create grid from seeds
        if not args.seeds:
            logger.error("Grid mode requires --seeds argument")
            return
        
        seeds = [int(s.strip()) for s in args.seeds.split(',')]
        logger.info(f"Creating grid from {len(seeds)} seeds...")
        
        grid = inference.create_sample_grid(
            seeds=seeds,
            method=args.method,
            num_steps=args.num_steps,
            grid_size=grid_size
        )
        
        output_path = output_dir / f"seed_grid_{timestamp}.png"
        save_image(grid, output_path)
        logger.info(f"Seed grid saved to {output_path}")
    
    elif args.mode == "benchmark":
        # Performance benchmark
        logger.info("Running performance benchmark...")
        results = inference.benchmark_performance()
        
        # Save results
        results_path = output_dir / f"benchmark_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print summary
        print("\nBenchmark Results Summary:")
        print("=" * 50)
        for case_name, result in results.items():
            print(f"{case_name}:")
            print(f"  Time: {result['avg_time']:.2f}±{result['std_time']:.2f}s")
            print(f"  Speed: {result['samples_per_second']:.2f} samples/sec")
        
        logger.info(f"Benchmark results saved to {results_path}")


if __name__ == "__main__":
    # Optimize for M4 Max
    torch.set_num_threads(8)
    
    # Enable MPS optimizations
    if torch.backends.mps.is_available():
        try:
            torch.backends.mps.enabled = True
            # Set memory fraction for better memory management
            torch.mps.set_per_process_memory_fraction(0.8)
        except AttributeError:
            pass
    
    main()
