import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import string
import random
import os
import pickle
from collections import Counter

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Hyperparameters
SEQUENCE_LENGTH = 100      # Length of input sequences
HIDDEN_SIZE = 128         # Size of hidden layer
NUM_LAYERS = 2            # Number of RNN layers
BATCH_SIZE = 64           # Batch size for training
LEARNING_RATE = 0.002     # Learning rate
NUM_EPOCHS = 50           # Number of training epochs
DROPOUT = 0.3             # Dropout rate
TEMPERATURE = 0.8         # Sampling temperature for generation

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('generated_text', exist_ok=True)

class TextDataset:
    """
    Dataset class for handling text data and creating sequences
    """
    def __init__(self, text_file=None, text_string=None, seq_length=100):
        self.seq_length = seq_length
        
        # Load text data
        if text_file and os.path.exists(text_file):
            with open(text_file, 'r', encoding='utf-8') as f:
                self.text = f.read()
        elif text_string:
            self.text = text_string
        else:
            # Use a sample text if no file provided
            self.text = self._get_sample_text()
        
        # Clean and prepare text
        self.text = self._clean_text(self.text)
        print(f"Text length: {len(self.text)} characters")
        
        # Create character mappings
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Characters: {''.join(self.chars[:50])}{'...' if len(self.chars) > 50 else ''}")
        
        # Create sequences
        self.sequences, self.targets = self._create_sequences()
        
    def _get_sample_text(self):
        """Get sample text for demonstration"""
        return """
        To be or not to be, that is the question:
        Whether 'tis nobler in the mind to suffer
        The slings and arrows of outrageous fortune,
        Or to take arms against a sea of troubles
        And by opposing end them. To die—to sleep,
        No more; and by a sleep to say we end
        The heart-ache and the thousand natural shocks
        That flesh is heir to: 'tis a consummation
        Devoutly to be wish'd. To die, to sleep;
        To sleep, perchance to dream—ay, there's the rub:
        For in that sleep of death what dreams may come,
        When we have shuffled off this mortal coil,
        Must give us pause—there's the respect
        That makes calamity of so long life.
        """ * 10  # Repeat for more training data
    
    def _clean_text(self, text):
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Keep only printable ASCII characters
        printable = set(string.printable)
        text = ''.join(filter(lambda x: x in printable, text))
        
        # Replace multiple whitespaces with single space
        import re
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _create_sequences(self):
        """Create input-output sequences for training"""
        sequences = []
        targets = []
        
        # Create overlapping sequences
        for i in range(0, len(self.text) - self.seq_length):
            # Input sequence
            seq = self.text[i:i + self.seq_length]
            seq_indices = [self.char_to_idx[ch] for ch in seq]
            sequences.append(seq_indices)
            
            # Target sequence (shifted by 1)
            target = self.text[i + 1:i + self.seq_length + 1]
            target_indices = [self.char_to_idx[ch] for ch in target]
            targets.append(target_indices)
        
        print(f"Created {len(sequences)} sequences")
        return np.array(sequences), np.array(targets)
    
    def get_dataloader(self, batch_size=64, shuffle=True):
        """Create PyTorch DataLoader"""
        # Convert to tensors
        sequences_tensor = torch.LongTensor(self.sequences)
        targets_tensor = torch.LongTensor(self.targets)
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(sequences_tensor, targets_tensor)
        
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            drop_last=True
        )
        
        return dataloader

class CharRNN(nn.Module):
    """
    Character-level RNN for text generation
    """
    def __init__(self, vocab_size, hidden_size, num_layers=2, dropout=0.3, rnn_type='LSTM'):
        super(CharRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.rnn_type = rnn_type
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # RNN layer
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                hidden_size, 
                hidden_size, 
                num_layers, 
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                hidden_size, 
                hidden_size, 
                num_layers, 
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:  # Vanilla RNN
            self.rnn = nn.RNN(
                hidden_size, 
                hidden_size, 
                num_layers, 
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        # Embedding
        embedded = self.embedding(x)
        
        # RNN forward pass
        rnn_out, hidden = self.rnn(embedded, hidden)
        
        # Apply dropout
        rnn_out = self.dropout(rnn_out)
        
        # Reshape for linear layer
        rnn_out = rnn_out.reshape(-1, self.hidden_size)
        
        # Output layer
        output = self.fc(rnn_out)
        
        return output, hidden
    
    def init_hidden(self, batch_size):
        """Initialize hidden state"""
        if self.rnn_type == 'LSTM':
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            return (h0, c0)
        else:
            return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

def train_model(model, dataloader, dataset, num_epochs=50, learning_rate=0.002):
    """
    Train the RNN model
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    train_losses = []
    
    print("Starting training...")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        hidden = model.init_hidden(BATCH_SIZE)
        
        for batch_idx, (sequences, targets) in enumerate(dataloader):
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            # Detach hidden state to prevent backprop through entire history
            if model.rnn_type == 'LSTM':
                hidden = (hidden[0].detach(), hidden[1].detach())
            else:
                hidden = hidden.detach()
            
            # Forward pass
            output, hidden = model(sequences, hidden)
            loss = criterion(output, targets.reshape(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            total_loss += loss.item()
        
        # Calculate average loss
        avg_loss = total_loss / len(dataloader)
        train_losses.append(avg_loss)
        
        # Update learning rate
        scheduler.step(avg_loss)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
            
            # Generate sample text
            sample_text = generate_text(model, dataset, length=100, temperature=0.8)
            print(f'Sample: "{sample_text[:50]}..."')
            print("-" * 50)
        
        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            save_model(model, dataset, f'models/char_rnn_epoch_{epoch+1}.pth')
    
    print("Training completed!")
    return train_losses

def generate_text(model, dataset, seed_text="", length=200, temperature=1.0):
    """
    Generate text using trained model
    """
    model.eval()
    
    # Use random seed if none provided
    if not seed_text:
        seed_text = random.choice(dataset.chars)
    
    # Ensure seed text is valid
    seed_text = ''.join([ch for ch in seed_text if ch in dataset.char_to_idx])
    if not seed_text:
        seed_text = random.choice(dataset.chars)
    
    # Convert seed to indices
    input_seq = [dataset.char_to_idx[ch] for ch in seed_text]
    
    # Initialize hidden state
    hidden = model.init_hidden(1)
    generated_text = seed_text
    
    with torch.no_grad():
        for _ in range(length):
            # Prepare input
            if len(input_seq) > SEQUENCE_LENGTH:
                input_seq = input_seq[-SEQUENCE_LENGTH:]
            
            input_tensor = torch.LongTensor([input_seq]).to(device)
            
            # Forward pass
            output, hidden = model(input_tensor, hidden)
            
            # Get probabilities for next character
            output = output[-1]  # Take last output
            probabilities = torch.softmax(output / temperature, dim=0)
            
            # Sample next character
            next_char_idx = torch.multinomial(probabilities, 1).item()
            next_char = dataset.idx_to_char[next_char_idx]
            
            # Append to generated text
            generated_text += next_char
            input_seq.append(next_char_idx)
    
    model.train()
    return generated_text

def save_model(model, dataset, filepath):
    """
    Save model and dataset information
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'vocab_size': dataset.vocab_size,
        'hidden_size': model.hidden_size,
        'num_layers': model.num_layers,
        'rnn_type': model.rnn_type,
        'char_to_idx': dataset.char_to_idx,
        'idx_to_char': dataset.idx_to_char,
        'chars': dataset.chars
    }
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """
    Load saved model
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    # Create model
    model = CharRNN(
        vocab_size=checkpoint['vocab_size'],
        hidden_size=checkpoint['hidden_size'],
        num_layers=checkpoint['num_layers'],
        rnn_type=checkpoint['rnn_type']
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create dummy dataset object for character mappings
    class DummyDataset:
        def __init__(self):
            self.char_to_idx = checkpoint['char_to_idx']
            self.idx_to_char = checkpoint['idx_to_char']
            self.chars = checkpoint['chars']
            self.vocab_size = checkpoint['vocab_size']
    
    dataset = DummyDataset()
    
    return model, dataset

def plot_training_loss(losses):
    """
    Plot training loss curve
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.show()

def interactive_generation(model, dataset):
    """
    Interactive text generation
    """
    print("\n" + "="*50)
    print("Interactive Text Generation")
    print("Type 'quit' to exit")
    print("="*50)
    
    while True:
        seed = input("\nEnter seed text (or press Enter for random): ").strip()
        
        if seed.lower() == 'quit':
            break
        
        try:
            length = int(input("Enter length (default 200): ") or "200")
            temperature = float(input("Enter temperature 0.1-2.0 (default 0.8): ") or "0.8")
        except ValueError:
            length = 200
            temperature = 0.8
        
        print("\nGenerating text...")
        generated = generate_text(model, dataset, seed, length, temperature)
        print(f"\nGenerated text:\n{'-'*30}")
        print(generated)
        print("-"*30)

def main():
    """
    Main training and demonstration function
    """
    print("Character-Level RNN Text Generation")
    print("=" * 50)
    
    # Create dataset
    print("Loading and preparing dataset...")
    
    # You can specify a text file here
    # dataset = TextDataset(text_file='your_text_file.txt', seq_length=SEQUENCE_LENGTH)
    
    # Or use the built-in sample text
    dataset = TextDataset(seq_length=SEQUENCE_LENGTH)
    
    # Create dataloader
    dataloader = dataset.get_dataloader(batch_size=BATCH_SIZE, shuffle=True)
    
    # Create model
    print(f"\nCreating {model_type} model...")
    model_type = 'LSTM'  # Change to 'RNN', 'LSTM', or 'GRU'
    model = CharRNN(
        vocab_size=dataset.vocab_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        rnn_type=model_type
    ).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Train model
    train_losses = train_model(
        model, 
        dataloader, 
        dataset, 
        num_epochs=NUM_EPOCHS, 
        learning_rate=LEARNING_RATE
    )
    
    # Plot training loss
    plot_training_loss(train_losses)
    
    # Save final model
    save_model(model, dataset, 'models/char_rnn_final.pth')
    
    # Generate sample texts with different temperatures
    print("\nGenerating sample texts with different temperatures:")
    print("=" * 60)
    
    for temp in [0.5, 0.8, 1.0, 1.2]:
        print(f"\nTemperature {temp}:")
        print("-" * 30)
        sample = generate_text(model, dataset, seed_text="to be", length=150, temperature=temp)
        print(sample)
    
    # Save generated samples
    with open('generated_text/samples.txt', 'w') as f:
        for temp in [0.5, 0.8, 1.0, 1.2]:
            f.write(f"Temperature {temp}:\n")
            f.write("-" * 30 + "\n")
            sample = generate_text(model, dataset, seed_text="to be", length=200, temperature=temp)
            f.write(sample + "\n\n")
    
    # Interactive generation
    try:
        interactive_generation(model, dataset)
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    main()
    
    print("\nTraining complete! Check the following files:")
    print("- models/ : Saved model checkpoints")
    print("- generated_text/ : Generated text samples")
    print("- training_loss.png : Training loss curve")
    
    print("\nTo use a saved model:")
    print("model, dataset = load_model('models/char_rnn_final.pth')")
    print("text = generate_text(model, dataset, 'your seed', length=200)")
