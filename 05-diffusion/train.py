"""
Training script for diffusion models
Optimized for M4 Max MacBook Pro with comprehensive logging and checkpointing
"""

import os
import argparse
import logging
from typing import Optional
from tqdm import tqdm
import wandb
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt

from diffusion_model import DiffusionModel, DiffusionConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ImageDataset(Dataset):
    """Custom dataset wrapper for images with preprocessing"""
    
    def __init__(self, dataset_path: str, image_size: int = 64, normalize: bool = True):
        self.image_size = image_size
        
        # Define transforms optimized for M4 Max
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
        
        if normalize:
            # Normalize to [-1, 1] range for diffusion models
            transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        
        self.transform = transforms.Compose(transform_list)
        
        # Load dataset based on path
        if os.path.isdir(dataset_path):
            self.dataset = datasets.ImageFolder(dataset_path, transform=self.transform)
        else:
            # Try built-in datasets
            if dataset_path.lower() == 'cifar10':
                self.dataset = datasets.CIFAR10(
                    root='./data', 
                    train=True, 
                    download=True, 
                    transform=self.transform
                )
            elif dataset_path.lower() == 'celeba':
                self.dataset = datasets.CelebA(
                    root='./data',
                    split='train',
                    download=True,
                    transform=self.transform
                )
            else:
                raise ValueError(f"Unknown dataset: {dataset_path}")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if isinstance(self.dataset[idx], tuple):
            return self.dataset[idx][0]  # Return only image, not label
        return self.dataset[idx]


class DiffusionTrainer:
    """Main training class for diffusion models"""
    
    def __init__(self, config: DiffusionConfig, dataset_path: str, experiment_name: str = "diffusion_experiment"):
        self.config = config
        self.experiment_name = experiment_name
        
        # Create directories
        self.checkpoint_dir = Path("checkpoints") / experiment_name
        self.sample_dir = Path("samples") / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = DiffusionModel(config)
        logger.info(f"Model initialized on device: {self.model.device}")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Initialize dataset and dataloader
        self.dataset = ImageDataset(dataset_path, config.image_size)
        
        # Optimize DataLoader for M4 Max
        num_workers = min(8, os.cpu_count() or 4)  # M4 Max has 12 cores, use 8 for data loading
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None
        )
        
        logger.info(f"Dataset loaded: {len(self.dataset)} images")
        logger.info(f"DataLoader: {len(self.dataloader)} batches, {num_workers} workers")
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.model.optimizer, 
            T_max=config.num_epochs,
            eta_min=config.learning_rate * 0.01
        )
        
        # Initialize wandb for experiment tracking
        self.use_wandb = False
        
    def init_wandb(self, project_name: str = "diffusion-models"):
        """Initialize Weights & Biases logging"""
        try:
            wandb.init(
                project=project_name,
                name=self.experiment_name,
                config=self.config.__dict__
            )
            self.use_wandb = True
            logger.info("Wandb initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
    
    def train(self, resume_from: Optional[str] = None):
        """Main training loop"""
        start_epoch = 0
        best_loss = float('inf')
        
        # Resume from checkpoint if specified
        if resume_from and os.path.exists(resume_from):
            start_epoch, last_loss = self.model.load_checkpoint(resume_from)
            logger.info(f"Resumed training from epoch {start_epoch}")
        
        logger.info("Starting training...")
        
        for epoch in range(start_epoch, self.config.num_epochs):
            epoch_loss = self._train_epoch(epoch)
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Log metrics
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} - Loss: {epoch_loss:.6f} - LR: {current_lr:.2e}")
            
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": epoch_loss,
                    "learning_rate": current_lr
                })
            
            # Save checkpoint
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                checkpoint_path = self.checkpoint_dir / "best_model.pt"
                self.model.save_checkpoint(str(checkpoint_path), epoch, epoch_loss)
                logger.info(f"New best model saved with loss: {best_loss:.6f}")
            
            # Regular checkpoint saving
            if (epoch + 1) % 10 == 0:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
                self.model.save_checkpoint(str(checkpoint_path), epoch, epoch_loss)
            
            # Generate samples
            if (epoch + 1) % 5 == 0:
                self._generate_samples(epoch)
        
        logger.info("Training completed!")
        
        if self.use_wandb:
            wandb.finish()
    
    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.model.train()
        total_loss = 0.0
        num_batches = len(self.dataloader)
        
        progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Ensure batch is in correct format
            if isinstance(batch, (list, tuple)):
                batch = batch[0]  # Take only images, ignore labels
            
            loss = self.model.train_step(batch)
            total_loss += loss
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({'loss': f'{loss:.6f}', 'avg_loss': f'{avg_loss:.6f}'})
            
            # Log batch-level metrics to wandb
            if self.use_wandb and batch_idx % 50 == 0:
                wandb.log({
                    "batch_loss": loss,
                    "batch": epoch * num_batches + batch_idx
                })
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def _generate_samples(self, epoch: int, num_samples: int = 16):
        """Generate and save sample images"""
        logger.info(f"Generating samples for epoch {epoch+1}...")
        
        # Generate samples using DDPM
        samples = self.model.sample(
            batch_size=num_samples,
            method="ddpm",
            num_steps=min(250, self.config.timesteps)  # Faster sampling for training
        )
        
        # Convert from [-1, 1] to [0, 1]
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)
        
        # Create grid and save
        grid = make_grid(samples, nrow=4, padding=2, normalize=False)
        
        # Save as image
        sample_path = self.sample_dir / f"samples_epoch_{epoch+1:03d}.png"
        save_image(grid, sample_path)
        
        # Log to wandb if available
        if self.use_wandb:
            wandb.log({
                "samples": wandb.Image(str(sample_path)),
                "epoch": epoch
            })
        
        logger.info(f"Samples saved to {sample_path}")
    
    def evaluate_fid(self, real_images_path: str, num_samples: int = 5000):
        """Evaluate FID score (requires additional dependencies)"""
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
            
            fid = FrechetInceptionDistance(feature=2048)
            fid = fid.to(self.model.device)
            
            # Load real images
            real_dataset = ImageDataset(real_images_path, self.config.image_size, normalize=False)
            real_loader = DataLoader(real_dataset, batch_size=32, shuffle=False)
            
            # Process real images
            for batch in tqdm(real_loader, desc="Processing real images"):
                batch = batch.to(self.model.device)
                if batch.shape[1] == 3:  # RGB
                    batch = (batch * 255).byte()
                fid.update(batch, real=True)
            
            # Generate fake images
            num_batches = (num_samples + 31) // 32
            for _ in tqdm(range(num_batches), desc="Generating fake images"):
                fake_batch = self.model.sample(batch_size=32, method="ddim", num_steps=50)
                fake_batch = (fake_batch + 1) / 2  # Convert to [0, 1]
                fake_batch = (fake_batch * 255).byte()
                fid.update(fake_batch, real=False)
            
            fid_score = fid.compute()
            logger.info(f"FID Score: {fid_score:.4f}")
            
            if self.use_wandb:
                wandb.log({"fid_score": fid_score})
            
            return fid_score.item()
            
        except ImportError:
            logger.warning("torchmetrics not installed. Cannot compute FID score.")
            return None


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train Diffusion Model")
    
    # Model arguments
    parser.add_argument("--image_size", type=int, default=64, help="Image size")
    parser.add_argument("--model_channels", type=int, default=128, help="Base model channels")
    parser.add_argument("--timesteps", type=int, default=1000, help="Number of diffusion timesteps")
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "cosine", "quad"])
    
    # Training arguments
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset path or name")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation")
    
    # System arguments
    parser.add_argument("--use_mps", action="store_true", default=True, help="Use MPS acceleration")
    parser.add_argument("--mixed_precision", action="store_true", default=True, help="Use mixed precision")
    parser.add_argument("--compile_model", action="store_true", default=True, help="Compile model")
    
    # Experiment arguments
    parser.add_argument("--experiment_name", type=str, default="diffusion_experiment", help="Experiment name")
    parser.add_argument("--resume_from", type=str, help="Resume from checkpoint")
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb logging")
    parser.add_argument("--wandb_project", type=str, default="diffusion-models", help="Wandb project name")
    
    args = parser.parse_args()
    
    # Create configuration
    config = DiffusionConfig(
        image_size=args.image_size,
        model_channels=args.model_channels,
        timesteps=args.timesteps,
        beta_schedule=args.beta_schedule,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_mps=args.use_mps,
        mixed_precision=args.mixed_precision,
        compile_model=args.compile_model
    )
    
    # Initialize trainer
    trainer = DiffusionTrainer(config, args.dataset, args.experiment_name)
    
    # Initialize wandb if requested
    if args.use_wandb:
        trainer.init_wandb(args.wandb_project)
    
    # Start training
    try:
        trainer.train(resume_from=args.resume_from)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # Save emergency checkpoint
        emergency_path = trainer.checkpoint_dir / "emergency_checkpoint.pt"
        trainer.model.save_checkpoint(str(emergency_path), 0, 0.0)
        logger.info(f"Emergency checkpoint saved to {emergency_path}")


if __name__ == "__main__":
    # Set optimal threading for M4 Max
    torch.set_num_threads(8)  # Use 8 threads for optimal performance
    
    # Enable optimized attention for better memory usage
    try:
        torch.backends.cuda.enable_flash_sdp(False)  # Disable for MPS compatibility
        torch.backends.mps.enabled = True
    except AttributeError:
        pass
    
    main()
