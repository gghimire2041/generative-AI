#!/usr/bin/env python3
"""
Convolutional Neural Network Implementation from Scratch
======================================================

This implementation creates a CNN from scratch using only NumPy to demonstrate
the core concepts of convolutional layers, pooling, and backpropagation.

Architecture:
Input (28×28×1) → Conv2D(32@3×3) → ReLU → MaxPool(2×2) → 
Conv2D(64@3×3) → ReLU → MaxPool(2×2) → Flatten → Dense(128) → ReLU → Dense(10) → Softmax

Requirements:
- numpy
- matplotlib (for visualization)
- PIL (for image processing)

Installation:
pip install numpy matplotlib pillow

Author: CNN Tutorial Implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import time

class Conv2D:
    """
    2D Convolutional Layer Implementation
    
    This layer applies multiple filters to the input to extract features.
    Each filter slides across the input and computes dot products to create feature maps.
    """
    
    def __init__(self, num_filters, filter_size, stride=1, padding=0):
        """
        Initialize convolutional layer.
        
        Args:
            num_filters (int): Number of filters (output channels)
            filter_size (int): Size of square filters (e.g., 3 for 3×3)
            stride (int): Stride for convolution
            padding (int): Padding size
        """
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        
        # Initialize filters using He initialization (good for ReLU)
        self.filters = None  # Will be initialized when input shape is known
        self.biases = np.zeros((num_filters, 1))
        
        # Cache for backward pass
        self.cache = {}
        
    def initialize_filters(self, input_channels):
        """Initialize filters once we know the input shape."""
        # He initialization: std = sqrt(2 / (filter_size^2 * input_channels))
        std = np.sqrt(2.0 / (self.filter_size * self.filter_size * input_channels))
        self.filters = np.random.randn(
            self.num_filters, input_channels, self.filter_size, self.filter_size
        ) * std
        
    def add_padding(self, input_data, padding):
        """Add zero padding to input data."""
        if padding == 0:
            return input_data
        
        # input_data shape: (batch_size, height, width, channels)
        padded = np.pad(
            input_data, 
            ((0, 0), (padding, padding), (padding, padding), (0, 0)), 
            mode='constant'
        )
        return padded
    
    def forward(self, input_data):
        """
        Forward pass through convolutional layer.
        
        Args:
            input_data (numpy.ndarray): Input of shape (batch_size, height, width, channels)
            
        Returns:
            numpy.ndarray: Output feature maps of shape (batch_size, out_height, out_width, num_filters)
        """
        batch_size, input_height, input_width, input_channels = input_data.shape
        
        # Initialize filters if not done yet
        if self.filters is None:
            self.initialize_filters(input_channels)
        
        # Add padding
        padded_input = self.add_padding(input_data, self.padding)
        
        # Calculate output dimensions
        out_height = (input_height + 2 * self.padding - self.filter_size) // self.stride + 1
        out_width = (input_width + 2 * self.padding - self.filter_size) // self.stride + 1
        
        # Initialize output
        output = np.zeros((batch_size, out_height, out_width, self.num_filters))
        
        # Perform convolution
        for batch in range(batch_size):
            for f in range(self.num_filters):
                for i in range(out_height):
                    for j in range(out_width):
                        # Calculate the region to convolve
                        start_i = i * self.stride
                        start_j = j * self.stride
                        end_i = start_i + self.filter_size
                        end_j = start_j + self.filter_size
                        
                        # Extract region
                        region = padded_input[batch, start_i:end_i, start_j:end_j, :]
                        
                        # Compute convolution (dot product)
                        output[batch, i, j, f] = np.sum(region * self.filters[f]) + self.biases[f]
        
        # Cache for backward pass
        self.cache = {
            'input': input_data,
            'padded_input': padded_input,
            'output': output
        }
        
        return output
    
    def backward(self, grad_output, learning_rate=0.001):
        """
        Backward pass through convolutional layer.
        
        Args:
            grad_output (numpy.ndarray): Gradient from next layer
            learning_rate (float): Learning rate for parameter updates
            
        Returns:
            numpy.ndarray: Gradient w.r.t. input
        """
        input_data = self.cache['input']
        padded_input = self.cache['padded_input']
        
        batch_size, input_height, input_width, input_channels = input_data.shape
        batch_size, out_height, out_width, num_filters = grad_output.shape
        
        # Initialize gradients
        grad_filters = np.zeros_like(self.filters)
        grad_biases = np.zeros_like(self.biases)
        grad_input = np.zeros_like(padded_input)
        
        # Compute gradients
        for batch in range(batch_size):
            for f in range(num_filters):
                for i in range(out_height):
                    for j in range(out_width):
                        start_i = i * self.stride
                        start_j = j * self.stride
                        end_i = start_i + self.filter_size
                        end_j = start_j + self.filter_size
                        
                        # Gradient w.r.t. filter
                        region = padded_input[batch, start_i:end_i, start_j:end_j, :]
                        grad_filters[f] += region * grad_output[batch, i, j, f]
                        
                        # Gradient w.r.t. input
                        grad_input[batch, start_i:end_i, start_j:end_j, :] += \
                            self.filters[f] * grad_output[batch, i, j, f]
        
        # Gradient w.r.t. biases
        grad_biases = np.sum(grad_output, axis=(0, 1, 2)).reshape(-1, 1)
        
        # Update parameters
        self.filters -= learning_rate * grad_filters
        self.biases -= learning_rate * grad_biases
        
        # Remove padding from input gradient
        if self.padding > 0:
            grad_input = grad_input[:, self.padding:-self.padding, self.padding:-self.padding, :]
        
        return grad_input


class MaxPool2D:
    """
    2D Max Pooling Layer Implementation
    
    This layer reduces spatial dimensions by taking the maximum value
    in each pooling window.
    """
    
    def __init__(self, pool_size=2, stride=2):
        """
        Initialize max pooling layer.
        
        Args:
            pool_size (int): Size of pooling window
            stride (int): Stride for pooling
        """
        self.pool_size = pool_size
        self.stride = stride
        self.cache = {}
        
    def forward(self, input_data):
        """
        Forward pass through max pooling layer.
        
        Args:
            input_data (numpy.ndarray): Input of shape (batch_size, height, width, channels)
            
        Returns:
            numpy.ndarray: Pooled output
        """
        batch_size, input_height, input_width, channels = input_data.shape
        
        # Calculate output dimensions
        out_height = (input_height - self.pool_size) // self.stride + 1
        out_width = (input_width - self.pool_size) // self.stride + 1
        
        # Initialize output
        output = np.zeros((batch_size, out_height, out_width, channels))
        
        # Keep track of max positions for backward pass
        max_mask = np.zeros_like(input_data)
        
        # Perform max pooling
        for batch in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        start_i = i * self.stride
                        start_j = j * self.stride
                        end_i = start_i + self.pool_size
                        end_j = start_j + self.pool_size
                        
                        # Extract pooling region
                        region = input_data[batch, start_i:end_i, start_j:end_j, c]
                        
                        # Find max value and its position
                        max_val = np.max(region)
                        output[batch, i, j, c] = max_val
                        
                        # Create mask for backward pass
                        max_pos = np.unravel_index(np.argmax(region), region.shape)
                        max_mask[batch, start_i + max_pos[0], start_j + max_pos[1], c] = 1
        
        # Cache for backward pass
        self.cache = {
            'input': input_data,
            'max_mask': max_mask,
            'output': output
        }
        
        return output
        
    def backward(self, grad_output):
        """
        Backward pass through max pooling layer.
        
        Args:
            grad_output (numpy.ndarray): Gradient from next layer
            
        Returns:
            numpy.ndarray: Gradient w.r.t. input
        """
        input_data = self.cache['input']
        max_mask = self.cache['max_mask']
        
        batch_size, out_height, out_width, channels = grad_output.shape
        
        # Initialize input gradient
        grad_input = np.zeros_like(input_data)
        
        # Distribute gradients to max positions
        for batch in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        start_i = i * self.stride
                        start_j = j * self.stride
                        end_i = start_i + self.pool_size
                        end_j = start_j + self.pool_size
                        
                        # Find where the max was in the original window
                        mask_region = max_mask[batch, start_i:end_i, start_j:end_j, c]
                        grad_input[batch, start_i:end_i, start_j:end_j, c] += \
                            mask_region * grad_output[batch, i, j, c]
        
        return grad_input


class Dense:
    """
    Fully Connected (Dense) Layer Implementation
    """
    
    def __init__(self, units, activation='relu'):
        """
        Initialize dense layer.
        
        Args:
            units (int): Number of output units
            activation (str): Activation function ('relu', 'softmax', 'none')
        """
        self.units = units
        self.activation = activation
        self.weights = None
        self.biases = None
        self.cache = {}
        
    def initialize_weights(self, input_size):
        """Initialize weights using He initialization."""
        self.weights = np.random.randn(input_size, self.units) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros((1, self.units))
    
    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU function."""
        return (x > 0).astype(float)
    
    def softmax(self, x):
        """Softmax activation function."""
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, input_data):
        """
        Forward pass through dense layer.
        
        Args:
            input_data (numpy.ndarray): Input data
            
        Returns:
            numpy.ndarray: Output after dense layer and activation
        """
        # Flatten input if needed
        if input_data.ndim > 2:
            batch_size = input_data.shape[0]
            input_data = input_data.reshape(batch_size, -1)
        
        # Initialize weights if not done yet
        if self.weights is None:
            self.initialize_weights(input_data.shape[1])
        
        # Linear transformation
        z = np.dot(input_data, self.weights) + self.biases
        
        # Apply activation
        if self.activation == 'relu':
            output = self.relu(z)
        elif self.activation == 'softmax':
            output = self.softmax(z)
        else:
            output = z
        
        # Cache for backward pass
        self.cache = {
            'input': input_data,
            'z': z,
            'output': output
        }
        
        return output
    
    def backward(self, grad_output, learning_rate=0.001):
        """
        Backward pass through dense layer.
        
        Args:
            grad_output (numpy.ndarray): Gradient from next layer
            learning_rate (float): Learning rate for parameter updates
            
        Returns:
            numpy.ndarray: Gradient w.r.t. input
        """
        input_data = self.cache['input']
        z = self.cache['z']
        
        # Gradient w.r.t. pre-activation
        if self.activation == 'relu':
            grad_z = grad_output * self.relu_derivative(z)
        elif self.activation == 'softmax':
            # For softmax with cross-entropy, gradient is simplified
            grad_z = grad_output  # Assuming grad_output is already computed correctly
        else:
            grad_z = grad_output
        
        # Gradients w.r.t. weights and biases
        grad_weights = np.dot(input_data.T, grad_z)
        grad_biases = np.sum(grad_z, axis=0, keepdims=True)
        
        # Gradient w.r.t. input
        grad_input = np.dot(grad_z, self.weights.T)
        
        # Update parameters
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases
        
        return grad_input


class SimpleCNN:
    """
    Simple CNN Implementation for Educational Purposes
    
    Architecture:
    Input (28×28×1) → Conv2D(32@3×3) → ReLU → MaxPool(2×2) → 
    Conv2D(64@3×3) → ReLU → MaxPool(2×2) → Flatten → Dense(128) → ReLU → Dense(10) → Softmax
    """
    
    def __init__(self, num_classes=10, learning_rate=0.001):
        """
        Initialize CNN.
        
        Args:
            num_classes (int): Number of output classes
            learning_rate (float): Learning rate for training
        """
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        
        # Define layers
        self.conv1 = Conv2D(num_filters=32, filter_size=3, stride=1, padding=1)
        self.pool1 = MaxPool2D(pool_size=2, stride=2)
        self.conv2 = Conv2D(num_filters=64, filter_size=3, stride=1, padding=1)
        self.pool2 = MaxPool2D(pool_size=2, stride=2)
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=num_classes, activation='softmax')
        
        # Training history
        self.history = {'loss': [], 'accuracy': []}
        
    def forward(self, x):
        """
        Forward pass through the entire CNN.
        
        Args:
            x (numpy.ndarray): Input batch of shape (batch_size, 28, 28, 1)
            
        Returns:
            numpy.ndarray: Output predictions
        """
        # First convolutional block
        x = self.conv1.forward(x)
        x = self.relu(x)
        x = self.pool1.forward(x)
        
        # Second convolutional block
        x = self.conv2.forward(x)
        x = self.relu(x)
        x = self.pool2.forward(x)
        
        # Fully connected layers
        x = self.dense1.forward(x)
        x = self.dense2.forward(x)
        
        return x
    
    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def cross_entropy_loss(self, predictions, targets):
        """
        Compute cross-entropy loss.
        
        Args:
            predictions (numpy.ndarray): Predicted probabilities
            targets (numpy.ndarray): One-hot encoded targets
            
        Returns:
            float: Cross-entropy loss
        """
        # Clip predictions to prevent log(0)
        predictions = np.clip(predictions, 1e-8, 1 - 1e-8)
        
        # Compute cross-entropy
        loss = -np.sum(targets * np.log(predictions)) / predictions.shape[0]
        
        return loss
    
    def backward(self, predictions, targets):
        """
        Backward pass through the entire CNN.
        
        Args:
            predictions (numpy.ndarray): Predicted probabilities
            targets (numpy.ndarray): One-hot encoded targets
        """
        # Compute gradient of loss w.r.t. predictions
        grad_predictions = (predictions - targets) / predictions.shape[0]
        
        # Backpropagate through dense layers
        grad = self.dense2.backward(grad_predictions, self.learning_rate)
        grad = self.dense1.backward(grad, self.learning_rate)
        
        # Reshape gradient for conv layers
        batch_size = grad.shape[0]
        grad = grad.reshape(batch_size, 7, 7, 64)  # Shape after pool2
        
        # Backpropagate through conv layers
        grad = self.pool2.backward(grad)
        grad = self.conv2.backward(grad, self.learning_rate)
        grad = self.pool1.backward(grad)
        grad = self.conv1.backward(grad, self.learning_rate)
    
    def train_batch(self, x_batch, y_batch):
        """
        Train on a single batch.
        
        Args:
            x_batch (numpy.ndarray): Input batch
            y_batch (numpy.ndarray): Target batch (one-hot encoded)
            
        Returns:
            tuple: (loss, accuracy)
        """
        # Forward pass
        predictions = self.forward(x_batch)
        
        # Compute loss
        loss = self.cross_entropy_loss(predictions, y_batch)
        
        # Compute accuracy
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_batch, axis=1)
        accuracy = np.mean(predicted_classes == true_classes)
        
        # Backward pass
        self.backward(predictions, y_batch)
        
        return loss, accuracy
    
    def train(self, x_train, y_train, x_val=None, y_val=None, epochs=10, batch_size=32):
        """
        Train the CNN.
        
        Args:
            x_train (numpy.ndarray): Training data
            y_train (numpy.ndarray): Training labels (one-hot encoded)
            x_val (numpy.ndarray, optional): Validation data
            y_val (numpy.ndarray, optional): Validation labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
        """
        print(f"Training CNN for {epochs} epochs...")
        print(f"Training samples: {x_train.shape[0]}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        print("-" * 50)
        
        num_batches = len(x_train) // batch_size
        
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_accuracy = 0
            
            # Shuffle training data
            indices = np.random.permutation(len(x_train))
            x_train_shuffled = x_train[indices]
            y_train_shuffled = y_train[indices]
            
            start_time = time.time()
            
            # Process batches
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                x_batch = x_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                
                # Train on batch
                loss, accuracy = self.train_batch(x_batch, y_batch)
                
                epoch_loss += loss
                epoch_accuracy += accuracy
                
                # Print progress
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{num_batches}, "
                          f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            
            # Average metrics
            epoch_loss /= num_batches
            epoch_accuracy /= num_batches
            
            # Store history
            self.history['loss'].append(epoch_loss)
            self.history['accuracy'].append(epoch_accuracy)
            
            epoch_time = time.time() - start_time
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s")
            print(f"Average Loss: {epoch_loss:.4f}, Average Accuracy: {epoch_accuracy:.4f}")
            
            # Validation
            if x_val is not None and y_val is not None:
                val_predictions = self.forward(x_val)
                val_loss = self.cross_entropy_loss(val_predictions, y_val)
                val_accuracy = np.mean(np.argmax(val_predictions, axis=1) == np.argmax(y_val, axis=1))
                print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
            
            print("-" * 50)
    
    def predict(self, x):
        """
        Make predictions on input data.
        
        Args:
            x (numpy.ndarray): Input data
            
        Returns:
            numpy.ndarray: Predicted class probabilities
        """
        return self.forward(x)
    
    def predict_classes(self, x):
        """
        Predict classes for input data.
        
        Args:
            x (numpy.ndarray): Input data
            
        Returns:
            numpy.ndarray: Predicted class labels
        """
        probabilities = self.predict(x)
        return np.argmax(probabilities, axis=1)
    
    def evaluate(self, x_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            x_test (numpy.ndarray): Test data
            y_test (numpy.ndarray): Test labels (one-hot encoded)
            
        Returns:
            tuple: (loss, accuracy)
        """
        predictions = self.predict(x_test)
        loss = self.cross_entropy_loss(predictions, y_test)
        accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1))
        return loss, accuracy
    
    def plot_training_history(self):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.plot(self.history['loss'])
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        ax2.plot(self.history['accuracy'])
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """Save the trained model."""
        model_data = {
            'conv1_filters': self.conv1.filters,
            'conv1_biases': self.conv1.biases,
            'conv2_filters': self.conv2.filters,
            'conv2_biases': self.conv2.biases,
            'dense1_weights': self.dense1.weights,
            'dense1_biases': self.dense1.biases,
            'dense2_weights': self.dense2.weights,
            'dense2_biases': self.dense2.biases,
            'history': self.history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")


def create_sample_data(num_samples=1000, image_size=28):
    """
    Create sample data for demonstration.
    
    Args:
        num_samples (int): Number of samples to generate
        image_size (int): Size of square images
        
    Returns:
        tuple: (x_data, y_data) where x_data is images and y_data is one-hot labels
    """
    print(f"Creating sample data: {num_samples} samples of size {image_size}×{image_size}")
    
    # Generate random images
    x_data = np.random.rand(num_samples, image_size, image_size, 1)
    
    # Create simple patterns for different classes
    y_data = np.zeros((num_samples, 10))
    
    for i in range(num_samples):
        # Create simple pattern based on image characteristics
        img = x_data[i, :, :, 0]
        
        # Different patterns for different classes
        if np.mean(img[:14, :14]) > 0.6:  # Bright top-left
            class_label = 0
        elif np.mean(img[:14, 14:]) > 0.6:  # Bright top-right
            class_label = 1
        elif np.mean(img[14:, :14]) > 0.6:  # Bright bottom-left
            class_label = 2
        elif np.mean(img[14:, 14:]) > 0.6:  # Bright bottom-right
            class_label = 3
        elif np.mean(img[7:21, 7:21]) > 0.6:  # Bright center
            class_label = 4
        elif np.sum(img > 0.8) > 50:  # Many bright pixels
            class_label = 5
        elif np.sum(img < 0.2) > 200:  # Many dark pixels
            class_label = 6
        elif np.var(img) > 0.1:  # High variance
            class_label = 7
        elif np.mean(img) > 0.7:  # Overall bright
            class_label = 8
        else:  # Default class
            class_label = 9
        
        y_data[i, class_label] = 1
    
    print(f"Data created with class distribution:")
    for i in range(10):
        count = np.sum(y_data[:, i])
        print(f"Class {i}: {count} samples")
    
    return x_data, y_data


def demonstrate_cnn():
    """
    Demonstrate the CNN with a complete example.
    """
    print("=" * 60)
    print("CONVOLUTIONAL NEURAL NETWORK DEMONSTRATION")
    print("=" * 60)
    
    # Create sample data
    x_train, y_train = create_sample_data(800, 28)
    x_val, y_val = create_sample_data(200, 28)
    
    # Create and train CNN
    cnn = SimpleCNN(num_classes=10, learning_rate=0.01)
    
    print("\nCNN Architecture:")
    print("Input (28×28×1) → Conv2D(32@3×3) → ReLU → MaxPool(2×2) →")
    print("Conv2D(64@3×3) → ReLU → MaxPool(2×2) → Flatten →")
    print("Dense(128) → ReLU → Dense(10) → Softmax")
    print()
    
    # Train the model
    cnn.train(
        x_train, y_train,
        x_val, y_val,
        epochs=5,
        batch_size=32
    )
    
    # Evaluate on validation set
    val_loss, val_accuracy = cnn.evaluate(x_val, y_val)
    print(f"Final Validation Loss: {val_loss:.4f}")
    print(f"Final Validation Accuracy: {val_accuracy:.4f}")
    
    # Plot training history
    cnn.plot_training_history()
    
    # Test individual predictions
    print("\n" + "=" * 60)
    print("TESTING INDIVIDUAL PREDICTIONS")
    print("=" * 60)
    
    test_samples = x_val[:5]
    test_labels = y_val[:5]
    
    predictions = cnn.predict(test_samples)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(test_labels, axis=1)
    
    for i in range(5):
        print(f"Sample {i+1}:")
        print(f"  True class: {true_classes[i]}")
        print(f"  Predicted class: {predicted_classes[i]}")
        print(f"  Confidence: {predictions[i, predicted_classes[i]]:.4f}")
        print()
    
    # Save model
    cnn.save_model('cnn_model.pkl')
    
    return cnn


def visualize_filters(cnn, layer_name='conv1'):
    """
    Visualize learned filters.
    
    Args:
        cnn (SimpleCNN): Trained CNN model
        layer_name (str): Name of layer to visualize
    """
    if layer_name == 'conv1':
        filters = cnn.conv1.filters
        title = "First Conv Layer Filters (32 filters)"
    elif layer_name == 'conv2':
        filters = cnn.conv2.filters
        title = "Second Conv Layer Filters (64 filters)"
    else:
        print(f"Layer {layer_name} not found")
        return
    
    num_filters = filters.shape[0]
    num_cols = 8
    num_rows = (num_filters + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 2 * num_rows))
    fig.suptitle(title, fontsize=16)
    
    for i in range(num_filters):
        row = i // num_cols
        col = i % num_cols
        
        if num_rows == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]
        
        # For first layer, show filter directly
        if layer_name == 'conv1':
            filter_img = filters[i, 0, :, :]  # First channel
        else:
            # For later layers, show average across channels
            filter_img = np.mean(filters[i], axis=0)
        
        ax.imshow(filter_img, cmap='gray')
        ax.set_title(f'Filter {i+1}')
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(num_filters, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        if num_rows == 1:
            axes[col].axis('off')
        else:
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    """
    Main execution with comprehensive CNN demonstration.
    """
    
    # Run the demonstration
    print("Starting CNN implementation demonstration...")
    print("This will create a CNN from scratch and train it on synthetic data.")
    print("Note: This is educational code - real implementations use optimized libraries!")
    print()
    
    trained_cnn = demonstrate_cnn()
    
    # Visualize learned filters
    print("\n" + "=" * 60)
    print("VISUALIZING LEARNED FILTERS")
    print("=" * 60)
    
    visualize_filters(trained_cnn, 'conv1')
    
    print("\n" + "=" * 60)
    print("IMPLEMENTATION NOTES")
    print("=" * 60)
    print("""
Key Features of This CNN Implementation:

1. **Pure NumPy Implementation**: 
   - No external ML libraries (TensorFlow, PyTorch)
   - Educational focus on understanding core concepts
   - All operations implemented from scratch

2. **Core CNN Components**:
   - Conv2D: Convolutional layers with multiple filters
   - MaxPool2D: Max pooling for dimensionality reduction
   - Dense: Fully connected layers
   - Proper forward and backward propagation

3. **Mathematical Operations**:
   - Convolution: Element-wise multiplication and summation
   - Pooling: Maximum or average within windows
   - Backpropagation: Gradient computation through all layers

4. **Training Features**:
   - Batch processing
   - Learning rate optimization
   - Training/validation monitoring
   - Model persistence

5. **Limitations** (for educational purposes):
   - Not optimized for speed (uses loops instead of vectorization)
   - Limited to basic architectures
   - No advanced features (batch norm, dropout, etc.)
   - Synthetic data only

6. **Real-World Considerations**:
   - Production CNNs use optimized libraries (cuDNN, MKL)
   - Advanced architectures (ResNet, Inception, etc.)
   - Sophisticated training techniques
   - Real datasets and preprocessing

This implementation demonstrates the fundamental concepts that underlie
all modern CNN frameworks. Understanding these basics is crucial for
effective deep learning practice.
    """)
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("""
To continue learning:

1. **Experiment with Architecture**:
   - Add more convolutional layers
   - Try different filter sizes
   - Experiment with different pooling strategies

2. **Implement Advanced Features**:
   - Batch normalization
   - Dropout layers
   - Different activation functions
   - Advanced optimizers (Adam, RMSprop)

3. **Try Real Data**:
   - MNIST digit recognition
   - CIFAR-10 image classification
   - Custom image datasets

4. **Modern Frameworks**:
   - TensorFlow/Keras
   - PyTorch
   - JAX

5. **Advanced Architectures**:
   - ResNet (residual connections)
   - Inception (multi-scale features)
   - DenseNet (dense
