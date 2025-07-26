import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.0002
NUM_EPOCHS = 50
NOISE_DIM = 100
IMAGE_DIM = 28 * 28  # MNIST image dimensions flattened

# Create directories for saving results
os.makedirs('generated_images', exist_ok=True)
os.makedirs('models', exist_ok=True)

class Generator(nn.Module):
    """
    Generator network that creates fake images from random noise
    """
    def __init__(self, noise_dim=100, img_dim=784):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
            # Input layer
            nn.Linear(noise_dim, 256),
            nn.LeakyReLU(0.2),
            
            # Hidden layers
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            
            # Output layer
            nn.Linear(1024, img_dim),
            nn.Tanh()  # Output values between -1 and 1
        )
    
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    """
    Discriminator network that distinguishes real from fake images
    """
    def __init__(self, img_dim=784):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            # Input layer
            nn.Linear(img_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            # Hidden layers
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            
            # Output layer
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output probability between 0 and 1
        )
    
    def forward(self, x):
        return self.model(x)

def load_data():
    """
    Load and preprocess MNIST dataset
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])
    
    dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        transform=transform, 
        download=True
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True
    )
    
    return dataloader

def weights_init(m):
    """
    Initialize network weights
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

def save_sample_images(generator, epoch, noise_dim=100, num_samples=16):
    """
    Generate and save sample images
    """
    generator.eval()
    with torch.no_grad():
        # Generate random noise
        noise = torch.randn(num_samples, noise_dim).to(device)
        
        # Generate fake images
        fake_images = generator(noise)
        
        # Reshape and denormalize
        fake_images = fake_images.view(-1, 1, 28, 28)
        fake_images = fake_images * 0.5 + 0.5  # Denormalize from [-1,1] to [0,1]
        
        # Save images
        save_image(fake_images, f'generated_images/epoch_{epoch}.png', nrow=4)
    
    generator.train()

def plot_losses(g_losses, d_losses):
    """
    Plot training losses
    """
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('GAN Training Losses')
    plt.savefig('training_losses.png')
    plt.show()

def train_gan():
    """
    Main training function
    """
    # Load data
    dataloader = load_data()
    
    # Initialize networks
    generator = Generator(NOISE_DIM, IMAGE_DIM).to(device)
    discriminator = Discriminator(IMAGE_DIM).to(device)
    
    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    # Loss function and optimizers
    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    
    # Training loop
    g_losses = []
    d_losses = []
    
    print("Starting training...")
    
    for epoch in range(NUM_EPOCHS):
        epoch_g_loss = 0
        epoch_d_loss = 0
        
        for i, (real_images, _) in enumerate(dataloader):
            batch_size = real_images.size(0)
            
            # Flatten images and move to device
            real_images = real_images.view(batch_size, -1).to(device)
            
            # Create labels
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # ===============================
            # Train Discriminator
            # ===============================
            d_optimizer.zero_grad()
            
            # Train with real images
            real_outputs = discriminator(real_images)
            d_loss_real = criterion(real_outputs, real_labels)
            
            # Train with fake images
            noise = torch.randn(batch_size, NOISE_DIM).to(device)
            fake_images = generator(noise)
            fake_outputs = discriminator(fake_images.detach())
            d_loss_fake = criterion(fake_outputs, fake_labels)
            
            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # ===============================
            # Train Generator
            # ===============================
            g_optimizer.zero_grad()
            
            # Generate fake images and get discriminator's opinion
            fake_outputs = discriminator(fake_images)
            g_loss = criterion(fake_outputs, real_labels)  # We want discriminator to think these are real
            
            g_loss.backward()
            g_optimizer.step()
            
            # Accumulate losses
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
        
        # Calculate average losses for the epoch
        avg_g_loss = epoch_g_loss / len(dataloader)
        avg_d_loss = epoch_d_loss / len(dataloader)
        
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        
        # Print progress
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] | '
              f'D Loss: {avg_d_loss:.4f} | '
              f'G Loss: {avg_g_loss:.4f}')
        
        # Save sample images every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_sample_images(generator, epoch + 1)
        
        # Save models every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), f'models/generator_epoch_{epoch+1}.pth')
            torch.save(discriminator.state_dict(), f'models/discriminator_epoch_{epoch+1}.pth')
    
    print("Training completed!")
    
    # Plot final losses
    plot_losses(g_losses, d_losses)
    
    # Save final models
    torch.save(generator.state_dict(), 'models/generator_final.pth')
    torch.save(discriminator.state_dict(), 'models/discriminator_final.pth')
    
    return generator, discriminator

def generate_samples(generator_path='models/generator_final.pth', num_samples=25):
    """
    Generate samples using a trained generator
    """
    # Load generator
    generator = Generator(NOISE_DIM, IMAGE_DIM).to(device)
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.eval()
    
    # Generate samples
    with torch.no_grad():
        noise = torch.randn(num_samples, NOISE_DIM).to(device)
        fake_images = generator(noise)
        
        # Reshape and denormalize
        fake_images = fake_images.view(-1, 1, 28, 28)
        fake_images = fake_images * 0.5 + 0.5
        
        # Convert to numpy and plot
        fake_images = fake_images.cpu().numpy()
        
        fig, axes = plt.subplots(5, 5, figsize=(10, 10))
        for i, ax in enumerate(axes.flat):
            ax.imshow(fake_images[i, 0], cmap='gray')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('final_generated_samples.png')
        plt.show()

if __name__ == "__main__":
    print("Simple GAN Implementation for MNIST")
    print("=" * 50)
    
    # Train the GAN
    generator, discriminator = train_gan()
    
    # Generate final samples
    print("\nGenerating final samples...")
    generate_samples()
    
    print("\nTraining complete! Check the following files:")
    print("- generated_images/ : Sample images during training")
    print("- models/ : Saved model weights")
    print("- training_losses.png : Loss curves")
    print("- final_generated_samples.png : Final generated samples")
