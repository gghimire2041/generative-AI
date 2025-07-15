#!/usr/bin/env python3
"""
Neural Network Implementation for 28x28 Grayscale Image Classification
=====================================================================

This implementation creates a neural network from scratch using only NumPy
to classify 28x28 grayscale images (like cats vs non-cats).

Requirements:
- numpy
- matplotlib (for visualization)
- PIL (for image processing)

Installation:
pip install numpy matplotlib pillow

Author: Neural Network Tutorial
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import pickle

class NeuralNetwork:
    """
    A simple feedforward neural network implementation from scratch.
    
    Architecture:
    Input Layer (784 neurons) → Hidden Layer (128 neurons) → Output Layer (1 neuron)
    
    Activation Functions:
    - Hidden Layer: ReLU
    - Output Layer: Sigmoid
    """
    
    def __init__(self, input_size=784, hidden_size=128, output_size=1, learning_rate=0.001):
        """
        Initialize the neural network with random weights and zero biases.
        
        Args:
            input_size (int): Number of input features (784 for 28x28 images)
            hidden_size (int): Number of neurons in hidden layer
            output_size (int): Number of output neurons (1 for binary classification)
            learning_rate (float): Learning rate for gradient descent
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights using He initialization (good for ReLU)
        self.W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((hidden_size, 1))
        
        self.W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((output_size, 1))
        
        # Store training history
        self.training_history = {'loss': [], 'accuracy': []}
        
    def relu(self, z):
        """
        ReLU activation function: f(z) = max(0, z)
        
        Args:
            z (numpy.ndarray): Input array
            
        Returns:
            numpy.ndarray: Output after ReLU activation
        """
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        """
        Derivative of ReLU function: f'(z) = 1 if z > 0, else 0
        
        Args:
            z (numpy.ndarray): Input array
            
        Returns:
            numpy.ndarray: Derivative values
        """
        return (z > 0).astype(float)
    
    def sigmoid(self, z):
        """
        Sigmoid activation function: f(z) = 1 / (1 + e^(-z))
        
        Args:
            z (numpy.ndarray): Input array
            
        Returns:
            numpy.ndarray: Output after sigmoid activation
        """
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        """
        Derivative of sigmoid function: f'(z) = f(z) * (1 - f(z))
        
        Args:
            z (numpy.ndarray): Input array (pre-activation)
            
        Returns:
            numpy.ndarray: Derivative values
        """
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def forward_propagation(self, X):
        """
        Perform forward propagation through the network.
        
        Args:
            X (numpy.ndarray): Input data of shape (input_size, m)
                              where m is the number of examples
        
        Returns:
            dict: Dictionary containing all intermediate values
        """
        # Ensure X is in correct shape
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Layer 1: Input → Hidden
        Z1 = np.dot(self.W1, X) + self.b1  # Linear transformation
        A1 = self.relu(Z1)                  # ReLU activation
        
        # Layer 2: Hidden → Output
        Z2 = np.dot(self.W2, A1) + self.b2  # Linear transformation
        A2 = self.sigmoid(Z2)               # Sigmoid activation
        
        # Store intermediate values for backpropagation
        cache = {
            'X': X,
            'Z1': Z1,
            'A1': A1,
            'Z2': Z2,
            'A2': A2
        }
        
        return A2, cache
    
    def compute_loss(self, A2, Y):
        """
        Compute binary cross-entropy loss.
        
        Loss = -1/m * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]
        
        Args:
            A2 (numpy.ndarray): Predicted outputs of shape (1, m)
            Y (numpy.ndarray): True labels of shape (1, m)
        
        Returns:
            float: Cross-entropy loss
        """
        m = Y.shape[1]  # Number of examples
        
        # Clip predictions to prevent log(0)
        A2 = np.clip(A2, 1e-8, 1 - 1e-8)
        
        # Compute cross-entropy loss
        loss = -1/m * np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))
        
        return loss
    
    def backward_propagation(self, cache, Y):
        """
        Perform backward propagation to compute gradients.
        
        Args:
            cache (dict): Intermediate values from forward propagation
            Y (numpy.ndarray): True labels of shape (1, m)
        
        Returns:
            dict: Dictionary containing gradients
        """
        # Extract values from cache
        X = cache['X']
        A1 = cache['A1']
        A2 = cache['A2']
        Z1 = cache['Z1']
        Z2 = cache['Z2']
        
        m = X.shape[1]  # Number of examples
        
        # Backward propagation for output layer
        dZ2 = A2 - Y  # Derivative of loss w.r.t. Z2
        dW2 = 1/m * np.dot(dZ2, A1.T)  # Gradient w.r.t. W2
        db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)  # Gradient w.r.t. b2
        
        # Backward propagation for hidden layer
        dA1 = np.dot(self.W2.T, dZ2)  # Derivative w.r.t. A1
        dZ1 = dA1 * self.relu_derivative(Z1)  # Derivative w.r.t. Z1
        dW1 = 1/m * np.dot(dZ1, X.T)  # Gradient w.r.t. W1
        db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)  # Gradient w.r.t. b1
        
        gradients = {
            'dW1': dW1,
            'db1': db1,
            'dW2': dW2,
            'db2': db2
        }
        
        return gradients
    
    def update_parameters(self, gradients):
        """
        Update network parameters using gradient descent.
        
        Args:
            gradients (dict): Dictionary containing gradients
        """
        # Extract gradients
        dW1 = gradients['dW1']
        db1 = gradients['db1']
        dW2 = gradients['dW2']
        db2 = gradients['db2']
        
        # Update parameters
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
    
    def predict(self, X):
        """
        Make predictions on input data.
        
        Args:
            X (numpy.ndarray): Input data of shape (input_size, m)
        
        Returns:
            numpy.ndarray: Binary predictions (0 or 1)
        """
        A2, _ = self.forward_propagation(X)
        predictions = (A2 > 0.5).astype(int)
        return predictions
    
    def compute_accuracy(self, X, Y):
        """
        Compute prediction accuracy.
        
        Args:
            X (numpy.ndarray): Input data
            Y (numpy.ndarray): True labels
        
        Returns:
            float: Accuracy as a percentage
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == Y) * 100
        return accuracy
    
    def train(self, X_train, Y_train, X_val=None, Y_val=None, epochs=1000, print_interval=100):
        """
        Train the neural network.
        
        Args:
            X_train (numpy.ndarray): Training data of shape (input_size, m)
            Y_train (numpy.ndarray): Training labels of shape (1, m)
            X_val (numpy.ndarray, optional): Validation data
            Y_val (numpy.ndarray, optional): Validation labels
            epochs (int): Number of training epochs
            print_interval (int): Interval for printing progress
        """
        print(f"Training neural network for {epochs} epochs...")
        print(f"Architecture: {self.input_size} → {self.hidden_size} → {self.output_size}")
        print(f"Learning rate: {self.learning_rate}")
        print("-" * 50)
        
        for epoch in range(epochs):
            # Forward propagation
            A2, cache = self.forward_propagation(X_train)
            
            # Compute loss
            loss = self.compute_loss(A2, Y_train)
            
            # Backward propagation
            gradients = self.backward_propagation(cache, Y_train)
            
            # Update parameters
            self.update_parameters(gradients)
            
            # Store training history
            self.training_history['loss'].append(loss)
            
            # Compute accuracy
            train_accuracy = self.compute_accuracy(X_train, Y_train)
            self.training_history['accuracy'].append(train_accuracy)
            
            # Print progress
            if epoch % print_interval == 0:
                print(f"Epoch {epoch:4d}: Loss = {loss:.6f}, Accuracy = {train_accuracy:.2f}%")
                
                if X_val is not None and Y_val is not None:
                    val_accuracy = self.compute_accuracy(X_val, Y_val)
                    print(f"           Validation Accuracy = {val_accuracy:.2f}%")
        
        print("-" * 50)
        print("Training completed!")
        
        # Final results
        final_train_accuracy = self.compute_accuracy(X_train, Y_train)
        print(f"Final Training Accuracy: {final_train_accuracy:.2f}%")
        
        if X_val is not None and Y_val is not None:
            final_val_accuracy = self.compute_accuracy(X_val, Y_val)
            print(f"Final Validation Accuracy: {final_val_accuracy:.2f}%")
    
    def plot_training_history(self):
        """
        Plot training loss and accuracy over epochs.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot loss
        ax1.plot(self.training_history['loss'])
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.training_history['accuracy'])
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """
        Save the trained model to a file.
        
        Args:
            filepath (str): Path to save the model
        """
        model_data = {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'learning_rate': self.learning_rate,
            'training_history': self.training_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model from a file.
        
        Args:
            filepath (str): Path to the saved model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.W1 = model_data['W1']
        self.b1 = model_data['b1']
        self.W2 = model_data['W2']
        self.b2 = model_data['b2']
        self.input_size = model_data['input_size']
        self.hidden_size = model_data['hidden_size']
        self.output_size = model_data['output_size']
        self.learning_rate = model_data['learning_rate']
        self.training_history = model_data['training_history']
        
        print(f"Model loaded from {filepath}")


def preprocess_image(image_path, target_size=(28, 28)):
    """
    Preprocess an image for neural network input.
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target size for resizing (height, width)
    
    Returns:
        numpy.ndarray: Preprocessed image as a flattened array
    """
    # Load image
    img = Image.open(image_path)
    
    # Convert to grayscale
    img = img.convert('L')
    
    # Resize to target size
    img = img.resize(target_size)
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Normalize pixel values to [0, 1]
    img_array = img_array.astype(np.float32) / 255.0
    
    # Flatten to 1D array
    img_flattened = img_array.flatten()
    
    return img_flattened


def create_sample_dataset(num_samples=1000):
    """
    Create a synthetic dataset for demonstration.
    This simulates 28x28 grayscale images with binary labels.
    
    Args:
        num_samples (int): Number of samples to generate
    
    Returns:
        tuple: (X, Y) where X is data and Y is labels
    """
    print("Creating synthetic dataset...")
    
    # Generate random 28x28 images
    X = np.random.rand(784, num_samples)  # Random pixel values [0, 1]
    
    # Create labels based on a simple rule (e.g., bright images = cats)
    # This is just for demonstration - real labels would come from actual data
    brightness = np.mean(X, axis=0)  # Average brightness of each image
    Y = (brightness > 0.5).astype(int).reshape(1, -1)  # Binary labels
    
    print(f"Created dataset with {num_samples} samples")
    print(f"Positive samples (label=1): {np.sum(Y)}")
    print(f"Negative samples (label=0): {num_samples - np.sum(Y)}")
    
    return X, Y


def demonstrate_neural_network():
    """
    Demonstrate the neural network with a complete example.
    """
    print("=" * 60)
    print("NEURAL NETWORK DEMONSTRATION")
    print("=" * 60)
    
    # Create synthetic dataset
    X_train, Y_train = create_sample_dataset(800)  # Training set
    X_val, Y_val = create_sample_dataset(200)      # Validation set
    
    # Create and train neural network
    nn = NeuralNetwork(
        input_size=784,    # 28x28 pixels
        hidden_size=128,   # Hidden layer size
        output_size=1,     # Binary classification
        learning_rate=0.01
    )
    
    # Train the network
    nn.train(
        X_train, Y_train,
        X_val, Y_val,
        epochs=500,
        print_interval=100
    )
    
    # Plot training history
    nn.plot_training_history()
    
    # Test individual predictions
    print("\n" + "=" * 60)
    print("TESTING INDIVIDUAL PREDICTIONS")
    print("=" * 60)
    
    # Test a few samples
    for i in range(5):
        sample = X_val[:, i:i+1]  # Get single sample
        true_label = Y_val[0, i]
        prediction = nn.predict(sample)[0, 0]
        confidence = nn.forward_propagation(sample)[0][0, 0]
        
        print(f"Sample {i+1}:")
        print(f"  True label: {true_label}")
        print(f"  Prediction: {prediction}")
        print(f"  Confidence: {confidence:.4f}")
        print()
    
    # Save the model
    nn.save_model('trained_model.pkl')
    
    return nn


def classify_image_example(model, image_path):
    """
    Example of how to classify a single image.
    
    Args:
        model (NeuralNetwork): Trained neural network
        image_path (str): Path to the image file
    """
    print(f"\nClassifying image: {image_path}")
    
    # Preprocess the image
    img_data = preprocess_image(image_path)
    img_data = img_data.reshape(-1, 1)  # Reshape for model input
    
    # Make prediction
    prediction = model.predict(img_data)[0, 0]
    confidence = model.forward_propagation(img_data)[0][0, 0]
    
    # Interpret results
    label = "Cat" if prediction == 1 else "Not Cat"
    
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.4f}")
    
    # Display the image
    img = Image.open(image_path).convert('L').resize((28, 28))
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap='gray')
    plt.title(f"Prediction: {label} (Confidence: {confidence:.4f})")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    """
    Main execution block with complete example.
    """
    
    # Run the demonstration
    trained_model = demonstrate_neural_network()
    
    print("\n" + "=" * 60)
    print("USAGE INSTRUCTIONS")
    print("=" * 60)
    print("""
To use this neural network with your own images:

1. Prepare your images:
   - Images should be in common formats (JPEG, PNG, etc.)
   - They will be automatically converted to 28x28 grayscale
   
2. Prepare your dataset:
   - Organize images into folders (e.g., 'cats/', 'not_cats/')
   - Create labels (1 for cats, 0 for not cats)
   
3. Load and preprocess your data:
   ```python
   # Example for a single image
   img_data = preprocess_image('path/to/your/image.jpg')
   img_data = img_data.reshape(-1, 1)
   
   # Make prediction
   prediction = trained_model.predict(img_data)
   ```

4. For batch processing:
   ```python
   # Load multiple images
   images = []
   labels = []
   
   for image_path in your_image_paths:
       img_data = preprocess_image(image_path)
       images.append(img_data)
       labels.append(your_label)  # 1 for cat, 0 for not cat
   
   # Convert to numpy arrays
   X = np.array(images).T  # Shape: (784, num_images)
   Y = np.array(labels).reshape(1, -1)  # Shape: (1, num_images)
   
   # Train or evaluate
   accuracy = trained_model.compute_accuracy(X, Y)
   ```

5. Save and load models:
   ```python
   # Save trained model
   trained_model.save_model('my_cat_classifier.pkl')
   
   # Load model later
   new_model = NeuralNetwork()
   new_model.load_model('my_cat_classifier.pkl')
   ```

Key Features of this Implementation:
- Built from scratch using only NumPy
- Clear mathematical operations
- Comprehensive documentation
- Training visualization
- Model persistence
- Image preprocessing utilities
- Easy to extend and modify

Next Steps:
- Try with real cat/dog datasets
- Experiment with different architectures
- Add regularization techniques
- Implement other activation functions
- Explore convolutional layers for better image processing
    """)
    
    # Example usage with a hypothetical image
    print("\n" + "=" * 60)
    print("EXAMPLE USAGE")
    print("=" * 60)
    print("""
# Example of classifying a new image:
# classify_image_example(trained_model, 'path/to/cat_image.jpg')

# Note: Since this is a demonstration with synthetic data,
# the model won't perform well on real images. To train on
# real cat images, you would need a proper dataset with
# actual cat and non-cat images.
    """)
