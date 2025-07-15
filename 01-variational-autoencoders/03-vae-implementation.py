#!/usr/bin/env python3
"""
Variational Autoencoder (VAE) Implementation from Scratch
=======================================================

This implementation creates a VAE from scratch using only NumPy to demonstrate
the core concepts of probabilistic modeling, variational inference, and 
generative modeling.

Architecture:
Input (784) → Encoder → μ(20), σ²(20) → Sampling → z(20) → Decoder → Output (784)

Key Concepts Demonstrated:
- Probabilistic encoder/decoder
- Reparameterization trick
- ELBO loss (reconstruction + KL divergence)
- Latent space interpolation
- Generation from prior

Requirements:
- numpy
- matplotlib (for visualization)
- scipy (for statistics)

Installation:
pip install numpy matplotlib scipy

Author: VAE Tutorial Implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pickle
import time
from matplotlib.patches import Ellipse

class VAE:
    """
    Variational Autoencoder Implementation
    
    This class implements a VAE with:
    - Probabilistic encoder that outputs mean and variance
    - Reparameterization trick for differentiable sampling
    - Decoder that reconstructs from latent codes
    - ELBO loss function with reconstruction and KL terms
    """
    
    def __init__(self, input_dim=784, latent_dim=20, hidden_dim=400, learning_rate=0.001):
        """
        Initialize VAE architecture.
        
        Args:
            input_dim (int): Dimension of input data (e.g., 784 for MNIST)
            latent_dim (int): Dimension of latent space
            hidden_dim (int): Dimension of hidden layers
            learning_rate (float): Learning rate for optimization
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        
        # Initialize encoder parameters
        self.init_encoder()
        
        # Initialize decoder parameters
        self.init_decoder()
        
        # Training history
        self.history = {
            'total_loss': [],
            'reconstruction_loss': [],
            'kl_loss': [],
            'epoch_times': []
        }
        
        # Cache for backward pass
        self.cache = {}
        
    def init_encoder(self):
        """
        Initialize encoder network parameters.
        
        Encoder Architecture:
        Input → Hidden → Mean & Log Variance
        """
        # First layer: input → hidden
        self.W_enc1 = np.random.randn(self.input_dim, self.hidden_dim) * np.sqrt(2.0 / self.input_dim)
        self.b_enc1 = np.zeros((1, self.hidden_dim))
        
        # Second layer: hidden → hidden
        self.W_enc2 = np.random.randn(self.hidden_dim, self.hidden_dim) * np.sqrt(2.0 / self.hidden_dim)
        self.b_enc2 = np.zeros((1, self.hidden_dim))
        
        # Mean layer: hidden → latent_dim
        self.W_mu = np.random.randn(self.hidden_dim, self.latent_dim) * np.sqrt(2.0 / self.hidden_dim)
        self.b_mu = np.zeros((1, self.latent_dim))
        
        # Log variance layer: hidden → latent_dim
        self.W_log_var = np.random.randn(self.hidden_dim, self.latent_dim) * np.sqrt(2.0 / self.hidden_dim)
        self.b_log_var = np.zeros((1, self.latent_dim))
        
    def init_decoder(self):
        """
        Initialize decoder network parameters.
        
        Decoder Architecture:
        Latent → Hidden → Output
        """
        # First layer: latent → hidden
        self.W_dec1 = np.random.randn(self.latent_dim, self.hidden_dim) * np.sqrt(2.0 / self.latent_dim)
        self.b_dec1 = np.zeros((1, self.hidden_dim))
        
        # Second layer: hidden → hidden
        self.W_dec2 = np.random.randn(self.hidden_dim, self.hidden_dim) * np.sqrt(2.0 / self.hidden_dim)
        self.b_dec2 = np.zeros((1, self.hidden_dim))
        
        # Output layer: hidden → input_dim
        self.W_dec3 = np.random.randn(self.hidden_dim, self.input_dim) * np.sqrt(2.0 / self.hidden_dim)
        self.b_dec3 = np.zeros((1, self.input_dim))
        
    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU activation."""
        return (x > 0).astype(float)
    
    def sigmoid(self, x):
        """Sigmoid activation function."""
        # Clip to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid activation."""
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def encode(self, x):
        """
        Encode input data to latent distribution parameters.
        
        Args:
            x (numpy.ndarray): Input data of shape (batch_size, input_dim)
            
        Returns:
            tuple: (mu, log_var) - mean and log variance of latent distribution
        """
        # First hidden layer
        h1 = self.relu(np.dot(x, self.W_enc1) + self.b_enc1)
        
        # Second hidden layer
        h2 = self.relu(np.dot(h1, self.W_enc2) + self.b_enc2)
        
        # Mean and log variance
        mu = np.dot(h2, self.W_mu) + self.b_mu
        log_var = np.dot(h2, self.W_log_var) + self.b_log_var
        
        # Cache for backward pass
        self.cache['encode'] = {
            'x': x,
            'h1': h1,
            'h2': h2,
            'mu': mu,
            'log_var': log_var
        }
        
        return mu, log_var
    
    def sample_latent(self, mu, log_var):
        """
        Sample latent variables using reparameterization trick.
        
        Args:
            mu (numpy.ndarray): Mean of latent distribution
            log_var (numpy.ndarray): Log variance of latent distribution
            
        Returns:
            tuple: (z, epsilon) - sampled latent code and noise
        """
        # Sample noise from standard normal
        epsilon = np.random.normal(0, 1, size=mu.shape)
        
        # Reparameterization: z = μ + σ * ε
        std = np.exp(0.5 * log_var)
        z = mu + std * epsilon
        
        return z, epsilon
    
    def decode(self, z):
        """
        Decode latent code to reconstruction.
        
        Args:
            z (numpy.ndarray): Latent code of shape (batch_size, latent_dim)
            
        Returns:
            numpy.ndarray: Reconstructed data
        """
        # First hidden layer
        h1 = self.relu(np.dot(z, self.W_dec1) + self.b_dec1)
        
        # Second hidden layer
        h2 = self.relu(np.dot(h1, self.W_dec2) + self.b_dec2)
        
        # Output layer with sigmoid activation
        x_reconstructed = self.sigmoid(np.dot(h2, self.W_dec3) + self.b_dec3)
        
        # Cache for backward pass
        self.cache['decode'] = {
            'z': z,
            'h1': h1,
            'h2': h2,
            'x_reconstructed': x_reconstructed
        }
        
        return x_reconstructed
    
    def forward(self, x):
        """
        Complete forward pass through VAE.
        
        Args:
            x (numpy.ndarray): Input data
            
        Returns:
            tuple: (x_reconstructed, mu, log_var, z, epsilon)
        """
        # Encode
        mu, log_var = self.encode(x)
        
        # Sample
        z, epsilon = self.sample_latent(mu, log_var)
        
        # Decode
        x_reconstructed = self.decode(z)
        
        return x_reconstructed, mu, log_var, z, epsilon
    
    def reconstruction_loss(self, x, x_reconstructed):
        """
        Compute reconstruction loss (negative log likelihood).
        
        For binary data, this is binary cross-entropy.
        For continuous data, this could be MSE.
        
        Args:
            x (numpy.ndarray): Original data
            x_reconstructed (numpy.ndarray): Reconstructed data
            
        Returns:
            float: Reconstruction loss
        """
        # Binary cross-entropy loss
        # Clip to prevent log(0)
        x_reconstructed = np.clip(x_reconstructed, 1e-8, 1 - 1e-8)
        
        # Compute BCE
        bce = -np.sum(x * np.log(x_reconstructed) + (1 - x) * np.log(1 - x_reconstructed))
        
        # Return mean loss
        return bce / x.shape[0]
    
    def kl_divergence(self, mu, log_var):
        """
        Compute KL divergence between q(z|x) and p(z).
        
        For q(z|x) = N(μ, σ²I) and p(z) = N(0, I):
        KL(q||p) = 0.5 * Σ[σ² + μ² - 1 - log(σ²)]
        
        Args:
            mu (numpy.ndarray): Mean of latent distribution
            log_var (numpy.ndarray): Log variance of latent distribution
            
        Returns:
            float: KL divergence
        """
        kl = 0.5 * np.sum(np.exp(log_var) + mu**2 - 1 - log_var)
        return kl / mu.shape[0]
    
    def compute_loss(self, x, x_reconstructed, mu, log_var, beta=1.0):
        """
        Compute total VAE loss (ELBO).
        
        Loss = Reconstruction Loss + β * KL Loss
        
        Args:
            x (numpy.ndarray): Original data
            x_reconstructed (numpy.ndarray): Reconstructed data
            mu (numpy.ndarray): Mean of latent distribution
            log_var (numpy.ndarray): Log variance of latent distribution
            beta (float): Weight for KL term (β-VAE)
            
        Returns:
            tuple: (total_loss, reconstruction_loss, kl_loss)
        """
        recon_loss = self.reconstruction_loss(x, x_reconstructed)
        kl_loss = self.kl_divergence(mu, log_var)
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def backward(self, x, x_reconstructed, mu, log_var, epsilon, beta=1.0):
        """
        Compute gradients for all parameters.
        
        Args:
            x (numpy.ndarray): Original data
            x_reconstructed (numpy.ndarray): Reconstructed data
            mu (numpy.ndarray): Mean of latent distribution
            log_var (numpy.ndarray): Log variance of latent distribution
            epsilon (numpy.ndarray): Noise used in reparameterization
            beta (float): Weight for KL term
        """
        batch_size = x.shape[0]
        
        # Get cached values
        encode_cache = self.cache['encode']
        decode_cache = self.cache['decode']
        
        # Gradient of reconstruction loss w.r.t. output
        x_reconstructed_clipped = np.clip(x_reconstructed, 1e-8, 1 - 1e-8)
        grad_recon = (x_reconstructed_clipped - x) / (x_reconstructed_clipped * (1 - x_reconstructed_clipped))
        grad_recon = grad_recon / batch_size
        
        # Backward through decoder
        self.backward_decoder(grad_recon, decode_cache)
        
        # Gradient of KL loss w.r.t. mu and log_var
        grad_mu_kl = mu / batch_size
        grad_log_var_kl = 0.5 * (np.exp(log_var) - 1) / batch_size
        
        # Gradient through sampling (reparameterization trick)
        grad_z = self.cache['grad_z']  # From decoder backward pass
        
        std = np.exp(0.5 * log_var)
        grad_mu_recon = grad_z
        grad_log_var_recon = grad_z * 0.5 * std * epsilon
        
        # Combine gradients
        grad_mu = grad_mu_recon + beta * grad_mu_kl
        grad_log_var = grad_log_var_recon + beta * grad_log_var_kl
        
        # Backward through encoder
        self.backward_encoder(grad_mu, grad_log_var, encode_cache)
        
    def backward_decoder(self, grad_output, cache):
        """
        Backward pass through decoder.
        
        Args:
            grad_output (numpy.ndarray): Gradient from loss
            cache (dict): Cached values from forward pass
        """
        z = cache['z']
        h1 = cache['h1']
        h2 = cache['h2']
        x_reconstructed = cache['x_reconstructed']
        
        # Gradient through output layer
        grad_pre_sigmoid = grad_output * self.sigmoid_derivative(
            np.dot(h2, self.W_dec3) + self.b_dec3
        )
        
        grad_W_dec3 = np.dot(h2.T, grad_pre_sigmoid)
        grad_b_dec3 = np.sum(grad_pre_sigmoid, axis=0, keepdims=True)
        grad_h2 = np.dot(grad_pre_sigmoid, self.W_dec3.T)
        
        # Gradient through second hidden layer
        grad_pre_relu2 = grad_h2 * self.relu_derivative(
            np.dot(h1, self.W_dec2) + self.b_dec2
        )
        
        grad_W_dec2 = np.dot(h1.T, grad_pre_relu2)
        grad_b_dec2 = np.sum(grad_pre_relu2, axis=0, keepdims=True)
        grad_h1 = np.dot(grad_pre_relu2, self.W_dec2.T)
        
        # Gradient through first hidden layer
        grad_pre_relu1 = grad_h1 * self.relu_derivative(
            np.dot(z, self.W_dec1) + self.b_dec1
        )
        
        grad_W_dec1 = np.dot(z.T, grad_pre_relu1)
        grad_b_dec1 = np.sum(grad_pre_relu1, axis=0, keepdims=True)
        grad_z = np.dot(grad_pre_relu1, self.W_dec1.T)
        
        # Update parameters
        self.W_dec3 -= self.learning_rate * grad_W_dec3
        self.b_dec3 -= self.learning_rate * grad_b_dec3
        self.W_dec2 -= self.learning_rate * grad_W_dec2
        self.b_dec2 -= self.learning_rate * grad_b_dec2
        self.W_dec1 -= self.learning_rate * grad_W_dec1
        self.b_dec1 -= self.learning_rate * grad_b_dec1
        
        # Cache gradient for sampling backward pass
        self.cache['grad_z'] = grad_z
        
    def backward_encoder(self, grad_mu, grad_log_var, cache):
        """
        Backward pass through encoder.
        
        Args:
            grad_mu (numpy.ndarray): Gradient w.r.t. mu
            grad_log_var (numpy.ndarray): Gradient w.r.t. log_var
            cache (dict): Cached values from forward pass
        """
        x = cache['x']
        h1 = cache['h1']
        h2 = cache['h2']
        
        # Gradient through mu and log_var layers
        grad_W_mu = np.dot(h2.T, grad_mu)
        grad_b_mu = np.sum(grad_mu, axis=0, keepdims=True)
        grad_h2_mu = np.dot(grad_mu, self.W_mu.T)
        
        grad_W_log_var = np.dot(h2.T, grad_log_var)
        grad_b_log_var = np.sum(grad_log_var, axis=0, keepdims=True)
        grad_h2_log_var = np.dot(grad_log_var, self.W_log_var.T)
        
        # Combine gradients for h2
        grad_h2 = grad_h2_mu + grad_h2_log_var
        
        # Gradient through second hidden layer
        grad_pre_relu2 = grad_h2 * self.relu_derivative(
            np.dot(h1, self.W_enc2) + self.b_enc2
        )
        
        grad_W_enc2 = np.dot(h1.T, grad_pre_relu2)
        grad_b_enc2 = np.sum(grad_pre_relu2, axis=0, keepdims=True)
        grad_h1 = np.dot(grad_pre_relu2, self.W_enc2.T)
        
        # Gradient through first hidden layer
        grad_pre_relu1 = grad_h1 * self.relu_derivative(
            np.dot(x, self.W_enc1) + self.b_enc1
        )
        
        grad_W_enc1 = np.dot(x.T, grad_pre_relu1)
        grad_b_enc1 = np.sum(grad_pre_relu1, axis=0, keepdims=True)
        
        # Update parameters
        self.W_mu -= self.learning_rate * grad_W_mu
        self.b_mu -= self.learning_rate * grad_b_mu
        self.W_log_var -= self.learning_rate * grad_W_log_var
        self.b_log_var -= self.learning_rate * grad_b_log_var
        self.W_enc2 -= self.learning_rate * grad_W_enc2
        self.b_enc2 -= self.learning_rate * grad_b_enc2
        self.W_enc1 -= self.learning_rate * grad_W_enc1
        self.b_enc1 -= self.learning_rate * grad_b_enc1
        
    def train_batch(self, x_batch, beta=1.0):
        """
        Train on a single batch.
        
        Args:
            x_batch (numpy.ndarray): Batch of input data
            beta (float): Weight for KL term
            
        Returns:
            tuple: (total_loss, reconstruction_loss, kl_loss)
        """
        # Forward pass
        x_reconstructed, mu, log_var, z, epsilon = self.forward(x_batch)
        
        # Compute loss
        total_loss, recon_loss, kl_loss = self.compute_loss(
            x_batch, x_reconstructed, mu, log_var, beta
        )
        
        # Backward pass
        self.backward(x_batch, x_reconstructed, mu, log_var, epsilon, beta)
        
        return total_loss, recon_loss, kl_loss
    
    def train(self, x_train, x_val=None, epochs=100, batch_size=128, beta=1.0, 
              beta_annealing=False, print_interval=10):
        """
        Train the VAE.
        
        Args:
            x_train (numpy.ndarray): Training data
            x_val (numpy.ndarray, optional): Validation data
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            beta (float): Weight for KL term
            beta_annealing (bool): Whether to use beta annealing
            print_interval (int): Print progress every N epochs
        """
        print(f"Training VAE for {epochs} epochs...")
        print(f"Training samples: {x_train.shape[0]}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Latent dimension: {self.latent_dim}")
        print(f"Beta: {beta}")
        print("-" * 60)
        
        num_batches = len(x_train) // batch_size
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Beta annealing (gradually increase KL weight)
            if beta_annealing:
                current_beta = min(beta, beta * epoch / (epochs * 0.5))
            else:
                current_beta = beta
            
            # Shuffle training data
            indices = np.random.permutation(len(x_train))
            x_train_shuffled = x_train[indices]
            
            epoch_total_loss = 0
            epoch_recon_loss = 0
            epoch_kl_loss = 0
            
            # Process batches
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                x_batch = x_train_shuffled[start_idx:end_idx]
                
                # Train on batch
                total_loss, recon_loss, kl_loss = self.train_batch(x_batch, current_beta)
                
                epoch_total_loss += total_loss
                epoch_recon_loss += recon_loss
                epoch_kl_loss += kl_loss
            
            # Average losses
            epoch_total_loss /= num_batches
            epoch_recon_loss /= num_batches
            epoch_kl_loss /= num_batches
            
            # Store history
            self.history['total_loss'].append(epoch_total_loss)
            self.history['reconstruction_loss'].append(epoch_recon_loss)
            self.history['kl_loss'].append(epoch_kl_loss)
            
            epoch_time = time.time() - start_time
            self.history['epoch_times'].append(epoch_time)
            
            # Print progress
            if epoch % print_interval == 0:
                print(f"Epoch {epoch+1:3d}/{epochs}: "
                      f"Total: {epoch_total_loss:.4f}, "
                      f"Recon: {epoch_recon_loss:.4f}, "
                      f"KL: {epoch_kl_loss:.4f}, "
                      f"β: {current_beta:.4f}, "
                      f"Time: {epoch_time:.2f}s")
                
                # Validation
                if x_val is not None:
                    val_loss, val_recon, val_kl = self.evaluate(x_val, current_beta)
                    print(f"         Validation: "
                          f"Total: {val_loss:.4f}, "
                          f"Recon: {val_recon:.4f}, "
                          f"KL: {val_kl:.4f}")
        
        print("-" * 60)
        print("Training completed!")
        
        # Final evaluation
        final_loss, final_recon, final_kl = self.evaluate(x_train, beta)
        print(f"Final Training Loss: {final_loss:.4f} "
              f"(Recon: {final_recon:.4f}, KL: {final_kl:.4f})")
        
        if x_val is not None:
            val_loss, val_recon, val_kl = self.evaluate(x_val, beta)
            print(f"Final Validation Loss: {val_loss:.4f} "
                  f"(Recon: {val_recon:.4f}, KL: {val_kl:.4f})")
    
    def evaluate(self, x_test, beta=1.0):
        """
        Evaluate the VAE on test data.
        
        Args:
            x_test (numpy.ndarray): Test data
            beta (float): Weight for KL term
            
        Returns:
            tuple: (total_loss, reconstruction_loss, kl_loss)
        """
        x_reconstructed, mu, log_var, z, epsilon = self.forward(x_test)
        return self.compute_loss(x_test, x_reconstructed, mu, log_var, beta)
    
    def generate(self, num_samples=16):
        """
        Generate new samples from the prior.
        
        Args:
            num_samples (int): Number of samples to generate
            
        Returns:
            numpy.ndarray: Generated samples
        """
        # Sample from prior p(z) = N(0, I)
        z = np.random.normal(0, 1, (num_samples, self.latent_dim))
        
        # Decode to get samples
        generated_samples = self.decode(z)
        
        return generated_samples
    
    def reconstruct(self, x):
        """
        Reconstruct input data.
        
        Args:
            x (numpy.ndarray): Input data
            
        Returns:
            numpy.ndarray: Reconstructed data
        """
        mu, log_var = self.encode(x)
        # Use mean (no sampling) for reconstruction
        reconstructed = self.decode(mu)
        return reconstructed
    
    def interpolate(self, x1, x2, num_steps=10):
        """
        Interpolate between two inputs in latent space.
        
        Args:
            x1 (numpy.ndarray): First input
            x2 (numpy.ndarray): Second input
            num_steps (int): Number of interpolation steps
            
        Returns:
            numpy.ndarray: Interpolated samples
        """
        # Encode both inputs
        mu1, _ = self.encode(x1.reshape(1, -1))
        mu2, _ = self.encode(x2.reshape(1, -1))
        
        # Linear interpolation in latent space
        alphas = np.linspace(0, 1, num_steps).reshape(-1, 1)
        z_interp = (1 - alphas) * mu1 + alphas * mu2
        
        # Decode interpolated latent codes
        interpolated = self.decode(z_interp)
        
        return interpolated
    
    def plot_training_history(self):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Total loss
        axes[0, 0].plot(self.history['total_loss'])
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # Reconstruction loss
        axes[0, 1].plot(self.history['reconstruction_loss'])
        axes[0, 1].set_title('Reconstruction Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
        
        # KL loss
        axes[1, 0].plot(self.history['kl_loss'])
        axes[1, 0].set_title('KL Divergence Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)
        
        # Training time
        axes[1, 1].plot(self.history['epoch_times'])
        axes[1, 1].set_title('Training Time per Epoch')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_latent_space(self, x_test, y_test=None, num_samples=1000):
        """
        Plot 2D latent space (only works for 2D latent space).
        
        Args:
            x_test (numpy.ndarray): Test data
            y_test (numpy.ndarray, optional): Test labels for coloring
            num_samples (int): Number of samples to plot
        """
        if self.latent_dim != 2:
            print("Latent space visualization only works for 2D latent space!")
            return
        
        # Encode test data
        mu, log_var = self.encode(x_test[:num_samples])
        
        # Plot latent space
        plt.figure(figsize=(10, 8))
        
        if y_test is not None:
            scatter = plt.scatter(mu[:, 0], mu[:, 1], c=y_test[:num_samples], 
                                cmap='tab10', alpha=0.7)
            plt.colorbar(scatter)
        else:
            plt.scatter(mu[:, 0], mu[:, 1], alpha=0.7)
        
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.title('Latent Space Representation')
        plt.grid(True)
        plt.show()
    
    def plot_generated_samples(self, num_samples=16, image_shape=(28, 28)):
        """
        Plot generated samples.
        
        Args:
            num_samples (int): Number of samples to generate
            image_shape (tuple): Shape to reshape samples for visualization
        """
        # Generate samples
        generated = self.generate(num_samples)
        
        # Plot samples
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        axes = axes.ravel()
        
        for i in range(min(num_samples, 16)):
            img = generated[i].reshape(image_shape)
            axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f'Sample {i+1}')
        
        plt.tight_layout()
        plt.suptitle('Generated Samples from Prior', y=1.02)
        plt.show()
    
    def plot_reconstructions(self, x_test, num_samples=8, image_shape=(28, 28)):
        """
        Plot original images and their reconstructions.
        
        Args:
            x_test (numpy.ndarray): Test data
            num_samples (int): Number of samples to show
            image_shape (tuple): Shape to reshape samples for visualization
        """
        # Get reconstructions
        reconstructions = self.reconstruct(x_test[:num_samples])
        
        # Plot originals and reconstructions
        fig, axes = plt.subplots(2, num_samples, figsize=(12, 4))
        
        for i in range(num_samples):
            # Original
            axes[0, i].imshow(x_test[i].reshape(image_shape), cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original', loc='left')
            
            # Reconstruction
            axes[1, i].imshow(reconstructions[i].reshape(image_shape), cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstruction', loc='left')
        
        plt.tight_layout()
        plt.show()
    
    def plot_interpolation(self, x1, x2, num_steps=10, image_shape=(28, 28)):
        """
        Plot interpolation between two images.
        
        Args:
            x1 (numpy.ndarray): First image
            x2 (numpy.ndarray): Second image
            num_steps (int): Number of interpolation steps
            image_shape (tuple): Shape for visualization
        """
        # Get interpolation
        interpolated = self.interpolate(x1, x2, num_steps)
        
        # Plot interpolation
        fig, axes = plt.subplots(1, num_steps, figsize=(15, 3))
        
        for i in range(num_steps):
            axes[i].imshow(interpolated[i].reshape(image_shape), cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f'α = {i/(num_steps-1):.1f}')
        
        plt.tight_layout()
        plt.suptitle('Latent Space Interpolation', y=1.05)
        plt.show()
    
    def save_model(self, filepath):
        """Save the trained model."""
        model_data = {
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'hidden_dim': self.hidden_dim,
            'learning_rate': self.learning_rate,
            'encoder_params': {
                'W_enc1': self.W_enc1, 'b_enc1': self.b_enc1,
                'W_enc2': self.W_enc2, 'b_enc2': self.b_enc2,
                'W_mu': self.W_mu, 'b_mu': self.b_mu,
                'W_log_var': self.W_log_var, 'b_log_var': self.b_log_var
            },
            'decoder_params': {
                'W_dec1': self.W_dec1, 'b_dec1': self.b_dec1,
                'W_dec2': self.W_dec2, 'b_dec2': self.b_dec2,
                'W_dec3': self.W_dec3, 'b_dec3': self.b_dec3
            },
            'history': self.history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")


def create_synthetic_data(num_samples=5000, data_type='mnist_like'):
    """
    Create synthetic data for VAE training.
    
    Args:
        num_samples (int): Number of samples to generate
        data_type (str): Type of data to generate
        
    Returns:
        tuple: (x_train, x_val, y_train, y_val)
    """
    if data_type == 'mnist_like':
        print(f"Creating MNIST-like synthetic data: {num_samples} samples")
        
        # Create different patterns
        x_data = []
        y_data = []
        
        for i in range(num_samples):
            # Create 28x28 image
            img = np.zeros((28, 28))
            
            # Different patterns for different classes
            pattern_type = i % 10
            
            if pattern_type == 0:  # Circle
                center = (14, 14)
                radius = 8
                for r in range(28):
                    for c in range(28):
                        if (r - center[0])**2 + (c - center[1])**2 <= radius**2:
                            img[r, c] = 1
            
            elif pattern_type == 1:  # Vertical line
                img[:, 13:15] = 1
            
            elif pattern_type == 2:  # Horizontal line
                img[13:15, :] = 1
            
            elif pattern_type == 3:  # Diagonal line
                for i in range(28):
                    if i < 28:
                        img[i, i] = 1
            
            elif pattern_type == 4:  # Square
                img[8:20, 8:20] = 1
                img[10:18, 10:18] = 0
            
            elif pattern_type == 5:  # Cross
                img[13:15, :] = 1
                img[:, 13:15] = 1
            
            elif pattern_type == 6:  # Corners
                img[0:8, 0:8] = 1
                img[0:8, 20:28] = 1
                img[20:28, 0:8] = 1
                img[20:28, 20:28] = 1
            
            elif pattern_type == 7:  # Border
                img[0:2, :] = 1
                img[26:28, :] = 1
                img[:, 0:2] = 1
                img[:, 26:28] = 1
            
            elif pattern_type == 8:  # Checkerboard
                for r in range(28):
                    for c in range(28):
                        if (r // 4 + c // 4) % 2 == 0:
                            img[r, c] = 1
            
            else:  # Random noise
                img = np.random.rand(28, 28) > 0.7
            
            # Add some noise
            noise = np.random.rand(28, 28) * 0.1
            img = np.clip(img + noise, 0, 1)
            
            x_data.append(img.flatten())
            y_data.append(pattern_type)
        
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        
        # Split into train and validation
        split_idx = int(0.8 * num_samples)
        x_train = x_data[:split_idx]
        x_val = x_data[split_idx:]
        y_train = y_data[:split_idx]
        y_val = y_data[split_idx:]
        
        print(f"Training set: {x_train.shape[0]} samples")
        print(f"Validation set: {x_val.shape[0]} samples")
        print(f"Data shape: {x_train.shape[1]} features")
        
        return x_train, x_val, y_train, y_val
    
    else:
        raise ValueError(f"Unknown data type: {data_type}")


def demonstrate_vae():
    """
    Demonstrate the VAE with a complete example.
    """
    print("=" * 60)
    print("VARIATIONAL AUTOENCODER DEMONSTRATION")
    print("=" * 60)
    
    # Create synthetic data
    x_train, x_val, y_train, y_val = create_synthetic_data(2000, 'mnist_like')
    
    # Create VAE
    vae = VAE(
        input_dim=784,      # 28x28 images
        latent_dim=10,      # 10D latent space
        hidden_dim=256,     # Hidden layer size
        learning_rate=0.001
    )
    
    print("\nVAE Architecture:")
    print("Input (784) → Encoder → μ(10), σ²(10) → Sampling → z(10) → Decoder → Output (784)")
    print()
    
    # Train the VAE
    vae.train(
        x_train, x_val,
        epochs=50,
        batch_size=64,
        beta=1.0,
        beta_annealing=False,
        print_interval=10
    )
    
    # Plot training history
    vae.plot_training_history()
    
    # Show reconstructions
    print("\n" + "=" * 60)
    print("RECONSTRUCTIONS")
    print("=" * 60)
    vae.plot_reconstructions(x_val, num_samples=8)
    
    # Show generated samples
    print("\n" + "=" * 60)
    print("GENERATED SAMPLES")
    print("=" * 60)
    vae.plot_generated_samples(num_samples=16)
    
    # Show interpolation
    print("\n" + "=" * 60)
    print("LATENT SPACE INTERPOLATION")
    print("=" * 60)
    vae.plot_interpolation(x_val[0], x_val[1], num_steps=10)
    
    # Save model
    vae.save_model('vae_model.pkl')
    
    return vae


def demonstrate_latent_space_exploration():
    """
    Demonstrate latent space exploration with 2D VAE.
    """
    print("\n" + "=" * 60)
    print("LATENT SPACE EXPLORATION (2D)")
    print("=" * 60)
    
    # Create data
    x_train, x_val, y_train, y_val = create_synthetic_data(1000, 'mnist_like')
    
    # Create VAE with 2D latent space for visualization
    vae_2d = VAE(
        input_dim=784,
        latent_dim=2,       # 2D for visualization
        hidden_dim=256,
        learning_rate=0.001
    )
    
    print("Training VAE with 2D latent space for visualization...")
    vae_2d.train(
        x_train, x_val,
        epochs=30,
        batch_size=64,
        beta=1.0,
        print_interval=10
    )
    
    # Plot latent space
    vae_2d.plot_latent_space(x_val, y_val, num_samples=500)
    
    # Generate samples by sampling from latent space grid
    print("\nGenerating samples from latent space grid...")
    
    # Create grid in latent space
    n_grid = 10
    z_grid = np.zeros((n_grid * n_grid, 2))
    
    for i in range(n_grid):
        for j in range(n_grid):
            z_grid[i * n_grid + j] = [
                -3 + 6 * i / (n_grid - 1),  # z1 from -3 to 3
                -3 + 6 * j / (n_grid - 1)   # z2 from -3 to 3
            ]
    
    # Decode grid points
    generated_grid = vae_2d.decode(z_grid)
    
    # Plot grid
    fig, axes = plt.subplots(n_grid, n_grid, figsize=(10, 10))
    
    for i in range(n_grid):
        for j in range(n_grid):
            idx = i * n_grid + j
            img = generated_grid[idx].reshape(28, 28)
            axes[i, j].imshow(img, cmap='gray')
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Generated Samples from 2D Latent Space Grid', y=1.02)
    plt.show()
    
    return vae_2d


if __name__ == "__main__":
    """
    Main execution with comprehensive VAE demonstration.
    """
    
    print("Starting VAE implementation demonstration...")
    print("This implementation demonstrates all key VAE concepts:")
    print("- Probabilistic encoder/decoder")
    print("- Reparameterization trick")
    print("- ELBO loss function")
    print("- Latent space properties")
    print("- Generation and interpolation")
    print()
    
    # Main demonstration
    trained_vae = demonstrate_vae()
    
    # 2D latent space exploration
    vae_2d = demonstrate_latent_space_exploration()
    
    print("\n" + "=" * 60)
    print("IMPLEMENTATION HIGHLIGHTS")
    print("=" * 60)
    print("""
Key Features of this VAE Implementation:

1. **Probabilistic Framework**:
   - Encoder outputs mean (μ) and log-variance (log σ²)
   - Proper sampling using reparameterization trick
   - KL divergence computed analytically

2. **Mathematical Rigor**:
   - ELBO loss function derived from variational principles
   - Correct gradient computation through stochastic layers
   - Proper handling of log-variance for numerical stability

3. **Training Features**:
   - β-VAE support for controlling disentanglement
   - Beta annealing for stable training
   - Comprehensive monitoring and visualization

4. **Visualization Tools**:
   - Latent space exploration
   - Reconstruction quality assessment
   - Interpolation demonstrations
   - Generated sample quality

5. **Educational Value**:
   - Clear separation of encoder, sampling, and decoder
   - Step-by-step gradient computation
   - Extensive comments explaining each operation

Core Concepts Demonstrated:
- Variational inference approximation
- Reparameterization trick for differentiable sampling
- Balance between reconstruction and regularization
- Latent space structure and properties
- Generative modeling capabilities
    """)
    
    print("\n" + "=" * 60)
    print("MATHEMATICAL INSIGHTS")
    print("=" * 60)
    print("""
Key Mathematical Concepts:

1. **ELBO Decomposition**:
   ELBO = E[log p(x|z)] - KL(q(z|x)||p(z))
   
   - First term: Reconstruction likelihood
   - Second term: Regularization (keep latent space structured)

2. **Reparameterization Trick**:
   z = μ + σ * ε, where ε ~ N(0,1)
   
   - Enables backpropagation through stochastic layers
   - Maintains distributional properties
   - Critical for end-to-end training

3. **KL Divergence (Analytical)**:
   KL(N(μ,σ²)||N(0,1)) = ½[σ² + μ² - 1 - log σ²]
   
   - Closed-form solution for Gaussian distributions
   - Regularizes latent space to be close to standard normal
   - Enables sampling from learned distribution

4. **Loss Function Balance**:
   - Reconstruction loss: ensures faithful reconstruction
   - KL loss: ensures structured, continuous latent space
   - β parameter: controls trade-off between the two

5. **Generative Process**:
   z ~ N(0,1) → decode(z) → x
   
   - Sample from simple prior
   - Transform through learned decoder
   - Generate realistic data samples
    """)
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("""
To continue exploring VAEs:

1. **Experiment with Architecture**:
   - Try different latent dimensions
   - Vary hidden layer sizes
   - Test different activation functions

2. **Advanced VAE Variants**:
   - Conditional VAE (CVAE)
   - β-VAE for disentanglement
   - Vector Quantized VAE (VQ-VAE)
   - Hierarchical VAE

3. **Real Data Applications**:
   - MNIST digit generation
   - CIFAR-10 image generation
   - Custom image datasets

4. **Modern Frameworks**:
   - PyTorch VAE implementation
   - TensorFlow Probability
   - JAX for research

5. **Advanced Topics**:
   - Normalizing flows
   - Diffusion models
   - Generative adversarial networks (GANs)
   - Autoregressive models

This implementation provides the mathematical foundation for understanding
all modern generative models!
    """)
    
    print("\n" + "=" * 60)
    print("COMPARISON WITH PREVIOUS MODELS")
    print("=" * 60)
    print("""
Model Evolution:

1. **Neural Networks (Classification)**:
   - Deterministic: x → y
   - Discriminative: learn p(y|x)
   - Supervised learning

2. **CNNs (Feature Learning)**:
   - Spatial structure: maintain image relationships
   - Hierarchical features: simple → complex
   - Still discriminative

3. **VAEs (Generative Modeling)**:
   - Probabilistic: x → p(z|x) → p(x|z)
   - Generative: learn p(x)
   - Unsupervised learning
   - Latent variable models

Key Advancement: VAEs bridge the gap between discriminative and generative
models, enabling both understanding and creation of data.
    """)
