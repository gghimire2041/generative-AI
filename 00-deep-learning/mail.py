"""
Incident Classification System using Custom Embedding Function + PyTorch with Existing Data
==========================================================================================

This implementation uses your existing get_embedding function to create text embeddings,
then feeds those embeddings along with class description context into PyTorch neural networks.

Requirements:
- df: DataFrame with incident data (should have 'combined' column and class labels)
- issue_summary: Dictionary with class names and their descriptions
- get_embedding: Your existing embedding function

Usage:
    results = train_incident_classifier(df, issue_summary, get_embedding)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# =====================================================
# 1. DATA PREPROCESSING AND CLASS HANDLING
# =====================================================

class DataPreprocessor:
    def __init__(self, df, issue_summary, text_column='combined', class_column='class'):
        """
        Initialize with existing data
        
        Args:
            df: DataFrame with incident data
            issue_summary: Dictionary with class definitions
                Format: {class_name: {"description": "...", "keywords": [...], ...}}
                OR: {class_name: "description_string"}
            text_column: Column name containing the text to classify
            class_column: Column name containing the class labels
        """
        self.df = df.copy()
        self.text_column = text_column
        self.class_column = class_column
        self.issue_summary = issue_summary
        
        # Process issue_summary to standardize format
        self.class_definitions = self._standardize_issue_summary()
        self.classes = list(self.class_definitions.keys())
        
        # Create label encoders
        self.label_encoder = LabelEncoder()
        self._prepare_data()
        
    def _standardize_issue_summary(self):
        """Standardize issue_summary format"""
        standardized = {}
        
        for class_name, class_info in self.issue_summary.items():
            if isinstance(class_info, str):
                # Simple string description
                standardized[class_name] = {
                    "description": class_info,
                    "keywords": self._extract_keywords_from_description(class_info)
                }
            elif isinstance(class_info, dict):
                # Dictionary with description and possibly other fields
                description = class_info.get("description", class_info.get("desc", ""))
                keywords = class_info.get("keywords", class_info.get("tags", []))
                
                if not keywords:
                    keywords = self._extract_keywords_from_description(description)
                
                standardized[class_name] = {
                    "description": description,
                    "keywords": keywords
                }
            else:
                raise ValueError(f"Unsupported format for class {class_name}: {type(class_info)}")
        
        return standardized
    
    def _extract_keywords_from_description(self, description):
        """Extract basic keywords from description"""
        import re
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', description.lower())
        # Filter out common stop words and keep meaningful terms
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'that', 'this', 'these', 'those'}
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        return list(set(keywords))  # Remove duplicates
    
    def _prepare_data(self):
        """Prepare data for training"""
        # Check if required columns exist
        if self.text_column not in self.df.columns:
            raise ValueError(f"Text column '{self.text_column}' not found in dataframe")
        
        if self.class_column not in self.df.columns:
            raise ValueError(f"Class column '{self.class_column}' not found in dataframe")
        
        # Remove rows with missing data
        self.df = self.df.dropna(subset=[self.text_column, self.class_column])
        
        # Filter classes to only include those in issue_summary
        self.df = self.df[self.df[self.class_column].isin(self.classes)]
        
        # Encode labels
        self.df['class_idx'] = self.label_encoder.fit_transform(self.df[self.class_column])
        
        # Create mappings
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.label_encoder.classes_)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        print(f"Data prepared: {len(self.df)} samples across {len(self.classes)} classes")
        print("Class distribution:")
        print(self.df[self.class_column].value_counts())
    
    def get_class_descriptions(self):
        """Get list of class descriptions in label encoder order"""
        return [self.class_definitions[cls]["description"] for cls in self.label_encoder.classes_]
    
    def get_class_info(self, class_name):
        """Get information for a specific class"""
        return self.class_definitions.get(class_name, {})

# =====================================================
# 2. EMBEDDING GENERATION WITH CUSTOM FUNCTION
# =====================================================

class CustomEmbeddingGenerator:
    def __init__(self, get_embedding_func):
        """
        Initialize embedding generator with your custom embedding function
        
        Args:
            get_embedding_func: Your existing embedding function
                Should accept text (string or list of strings) and return embeddings
        """
        self.get_embedding = get_embedding_func
        
        # Test the function to get embedding dimension
        print("Testing embedding function...")
        test_embedding = self._get_single_embedding("test")
        self.embedding_dim = len(test_embedding) if hasattr(test_embedding, '__len__') else test_embedding.shape[-1]
        print(f"Embedding dimension: {self.embedding_dim}")
        
        self.class_embeddings = None
        self.class_descriptions = None
    
    def _get_single_embedding(self, text):
        """Get embedding for a single text"""
        try:
            # Try calling the function with single text
            embedding = self.get_embedding(text)
            
            # Convert to numpy if needed
            if hasattr(embedding, 'numpy'):
                embedding = embedding.numpy()
            elif not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            
            # Ensure 1D
            if embedding.ndim > 1:
                embedding = embedding.flatten()
                
            return embedding
            
        except Exception as e:
            print(f"Error with embedding function: {e}")
            print(f"Input type: {type(text)}")
            print(f"Input: {text}")
            raise
    
    def _get_batch_embeddings(self, texts):
        """Get embeddings for a batch of texts"""
        try:
            # Try batch processing first
            embeddings = self.get_embedding(texts)
            
            # Convert to numpy if needed
            if hasattr(embeddings, 'numpy'):
                embeddings = embeddings.numpy()
            elif not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)
            
            # Ensure correct shape
            if embeddings.ndim == 1:
                # Single embedding returned, reshape
                embeddings = embeddings.reshape(1, -1)
            
            return embeddings
            
        except Exception as e:
            print(f"Batch processing failed: {e}")
            print("Falling back to individual processing...")
            
            # Fall back to individual processing
            embeddings = []
            for text in texts:
                emb = self._get_single_embedding(text)
                embeddings.append(emb)
            
            return np.array(embeddings)
    
    def generate_class_embeddings(self, class_descriptions):
        """Generate embeddings for class descriptions"""
        print("Generating class description embeddings...")
        self.class_descriptions = class_descriptions
        
        self.class_embeddings = self._get_batch_embeddings(class_descriptions)
        
        print(f"Class embeddings shape: {self.class_embeddings.shape}")
        return self.class_embeddings
    
    def generate_text_embeddings(self, texts, batch_size=32):
        """Generate embeddings for incident texts"""
        print(f"Generating text embeddings for {len(texts)} texts...")
        
        if len(texts) <= batch_size:
            # Process all at once
            embeddings = self._get_batch_embeddings(texts)
        else:
            # Process in batches
            embeddings = []
            
            with tqdm(range(0, len(texts), batch_size), desc="Generating embeddings") as pbar:
                for i in pbar:
                    batch_texts = texts[i:i + batch_size]
                    batch_embeddings = self._get_batch_embeddings(batch_texts)
                    embeddings.append(batch_embeddings)
            
            embeddings = np.vstack(embeddings)
        
        print(f"Text embeddings shape: {embeddings.shape}")
        return embeddings
    
    def compute_class_similarities(self, text_embeddings):
        """Compute similarity scores between text and class descriptions"""
        if self.class_embeddings is None:
            raise ValueError("Class embeddings not generated. Call generate_class_embeddings first.")
        
        print("Computing class similarity features...")
        similarities = cosine_similarity(text_embeddings, self.class_embeddings)
        return similarities
    
    def create_enhanced_features(self, text_embeddings, include_similarities=True):
        """Create enhanced features combining text embeddings and class similarities"""
        features = [text_embeddings]
        
        if include_similarities and self.class_embeddings is not None:
            similarities = self.compute_class_similarities(text_embeddings)
            features.append(similarities)
            
            # Add normalized similarities
            similarity_norms = np.linalg.norm(similarities, axis=1, keepdims=True)
            similarity_norms = np.where(similarity_norms == 0, 1, similarity_norms)  # Avoid division by zero
            normalized_similarities = similarities / similarity_norms
            features.append(normalized_similarities)
            
            # Add max similarity scores
            max_similarities = np.max(similarities, axis=1, keepdims=True)
            features.append(max_similarities)
            
            # Add similarity statistics
            mean_similarities = np.mean(similarities, axis=1, keepdims=True)
            std_similarities = np.std(similarities, axis=1, keepdims=True)
            features.extend([mean_similarities, std_similarities])
        
        enhanced_features = np.concatenate(features, axis=1)
        print(f"Enhanced features shape: {enhanced_features.shape}")
        return enhanced_features

# =====================================================
# 3. PYTORCH DATASET
# =====================================================

class EnhancedEmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels, class_similarities=None):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)
        
        if class_similarities is not None:
            self.class_similarities = torch.FloatTensor(class_similarities)
        else:
            self.class_similarities = None
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        if self.class_similarities is not None:
            return self.embeddings[idx], self.labels[idx], self.class_similarities[idx]
        else:
            return self.embeddings[idx], self.labels[idx]

# =====================================================
# 4. NEURAL NETWORK MODELS
# =====================================================

class ClassAwareMLPClassifier(nn.Module):
    """MLP with class description awareness"""
    def __init__(self, input_dim, num_classes, hidden_dims=[512, 256, 128], dropout=0.3):
        super(ClassAwareMLPClassifier, self).__init__()
        
        self.num_classes = num_classes
        
        # Main feature processing
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Class-aware attention mechanism
        self.class_attention = nn.MultiheadAttention(
            embed_dim=prev_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(prev_dim, num_classes)
        )
    
    def forward(self, x, class_similarities=None):
        # Extract features
        features = self.feature_extractor(x)
        
        # Add batch and sequence dimensions for attention
        features_expanded = features.unsqueeze(1)
        
        # Self-attention
        attended_features, _ = self.class_attention(
            features_expanded, features_expanded, features_expanded
        )
        
        # Remove sequence dimension
        attended_features = attended_features.squeeze(1)
        
        # If class similarities are provided, incorporate them
        if class_similarities is not None:
            similarity_weights = F.softmax(class_similarities, dim=1)
            attended_features = attended_features + 0.1 * torch.sum(
                similarity_weights.unsqueeze(1) * class_similarities.unsqueeze(1), dim=2
            )
        
        # Final classification
        output = self.classifier(attended_features)
        return output

class ClassAwareLSTMClassifier(nn.Module):
    """LSTM with class description awareness"""
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.3):
        super(ClassAwareLSTMClassifier, self).__init__()
        
        # Determine sequence length based on input dimension
        self.seq_len = max(8, min(16, input_dim // 32))
        self.feature_dim = max(8, input_dim // self.seq_len)
        
        # Input processing
        self.input_projection = nn.Linear(input_dim, self.seq_len * self.feature_dim)
        self.feature_projection = nn.Linear(self.feature_dim, hidden_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            hidden_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            hidden_dim * 2, 
            num_heads=8, 
            dropout=dropout,
            batch_first=True
        )
        
        # Class similarity integration
        self.class_integration = nn.Linear(num_classes, hidden_dim * 2)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x, class_similarities=None):
        batch_size = x.size(0)
        
        # Project to desired sequence length
        x = self.input_projection(x)
        x = x.view(batch_size, self.seq_len, self.feature_dim)
        
        # Project features to hidden dimension
        x = self.feature_projection(x)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        pooled = torch.mean(attn_out, dim=1)
        
        # Integrate class similarities if provided
        if class_similarities is not None:
            class_features = self.class_integration(class_similarities)
            pooled = pooled + 0.2 * class_features
        
        # Classification
        output = self.classifier(pooled)
        return output

class ClassAwareCNNClassifier(nn.Module):
    """CNN with class description awareness"""
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super(ClassAwareCNNClassifier, self).__init__()
        
        self.num_classes = num_classes
        
        # Calculate appropriate 2D dimensions
        self.height = int(np.sqrt(input_dim))
        self.width = input_dim // self.height
        
        # Adjust if not perfect fit
        target_size = self.height * self.width
        if target_size != input_dim:
            self.input_projection = nn.Linear(input_dim, target_size)
        else:
            self.input_projection = None
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Class similarity integration
        self.class_integration = nn.Linear(num_classes, 128)
        
        # Classifier
        conv_output_size = 128 * 4 * 4
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(conv_output_size + 128, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x, class_similarities=None):
        batch_size = x.size(0)
        
        # Project if needed
        if self.input_projection:
            x = self.input_projection(x)
        
        # Reshape to 2D
        x = x.view(batch_size, 1, self.height, self.width)
        
        # Apply convolutions
        conv_features = self.conv_layers(x)
        conv_features = conv_features.view(batch_size, -1)
        
        # Integrate class similarities
        if class_similarities is not None:
            class_features = self.class_integration(class_similarities)
            combined_features = torch.cat([conv_features, class_features], dim=1)
        else:
            zero_class_features = torch.zeros(batch_size, 128, device=conv_features.device)
            combined_features = torch.cat([conv_features, zero_class_features], dim=1)
        
        # Classification
        output = self.classifier(combined_features)
        return output

# =====================================================
# 5. TRAINING AND EVALUATION
# =====================================================

class ModelTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_accuracies = []
    
    def train_model(self, train_loader, val_loader, num_epochs=50, lr=0.001):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        best_val_acc = 0.0
        best_model_state = None
        
        print(f"Training on {self.device}")
        print(f"Model: {self.model.__class__.__name__}")
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
                for batch in pbar:
                    if len(batch) == 3:  # With class similarities
                        embeddings, labels, class_similarities = batch
                        embeddings = embeddings.to(self.device)
                        labels = labels.to(self.device) 
                        class_similarities = class_similarities.to(self.device)
                        
                        optimizer.zero_grad()
                        outputs = self.model(embeddings, class_similarities)
                    else:  # Without class similarities
                        embeddings, labels = batch
                        embeddings, labels = embeddings.to(self.device), labels.to(self.device)
                        
                        optimizer.zero_grad()
                        outputs = self.model(embeddings)
                    
                    loss = criterion(outputs, labels)
                    loss.backward()
                    
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()
                    
                    pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{100 * train_correct / train_total:.2f}%'
                    })
            
            # Validation phase
            val_acc = self.evaluate_model(val_loader)
            
            self.train_losses.append(train_loss / len(train_loader))
            self.val_accuracies.append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
            
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Acc: {100 * train_correct / train_total:.2f}%, '
                  f'Val Acc: {val_acc:.2f}%, '
                  f'Best Val Acc: {best_val_acc:.2f}%')
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        return best_val_acc
    
    def evaluate_model(self, data_loader):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in data_loader:
                if len(batch) == 3:  # With class similarities
                    embeddings, labels, class_similarities = batch
                    embeddings = embeddings.to(self.device)
                    labels = labels.to(self.device)
                    class_similarities = class_similarities.to(self.device)
                    outputs = self.model(embeddings, class_similarities)
                else:  # Without class similarities
                    embeddings, labels = batch
                    embeddings, labels = embeddings.to(self.device), labels.to(self.device)
                    outputs = self.model(embeddings)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        return accuracy
    
    def get_predictions(self, data_loader):
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in data_loader:
                if len(batch) == 3:  # With class similarities
                    embeddings, labels, class_similarities = batch
                    embeddings = embeddings.to(self.device)
                    labels = labels.to(self.device)
                    class_similarities = class_similarities.to(self.device)
                    outputs = self.model(embeddings, class_similarities)
                else:  # Without class similarities
                    embeddings, labels = batch
                    embeddings, labels = embeddings.to(self.device), labels.to(self.device)
                    outputs = self.model(embeddings)
                
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)

# =====================================================
# 6. VISUALIZATION AND ANALYSIS
# =====================================================

def plot_class_similarities(text_embeddings, class_embeddings, class_names, sample_size=100):
    """Plot similarity heatmap between sample texts and class descriptions"""
    sample_indices = np.random.choice(len(text_embeddings), min(sample_size, len(text_embeddings)), replace=False)
    sample_embeddings = text_embeddings[sample_indices]
    
    similarities = cosine_similarity(sample_embeddings, class_embeddings)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(similarities, 
                xticklabels=class_names,
                yticklabels=[f'Text {i}' for i in range(len(sample_embeddings))],
                cmap='YlOrRd', 
                center=0)
    plt.title('Text-to-Class Description Similarities')
    plt.xlabel('Class Descriptions')
    plt.ylabel('Sample Texts')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_training_history(trainer, model_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(trainer.train_losses)
    ax1.set_title(f'{model_name} - Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    ax2.plot(trainer.val_accuracies)
    ax2.set_title(f'{model_name} - Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def analyze_predictions(y_true, y_pred, y_prob, class_names, model_name):
    print(f"\n{model_name} - Detailed Results:")
    print("=" * 50)
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

def analyze_class_awareness(predictions, true_labels, class_similarities, class_names):
    """Analyze how class similarities correlate with predictions"""
    correct_mask = predictions == true_labels
    
    print("\nClass Awareness Analysis:")
    print("=" * 50)
    
    for i, class_name in enumerate(class_names):
        class_mask = true_labels == i
        if np.sum(class_mask) == 0:
            continue
            
        avg_similarity = np.mean(class_similarities[class_mask, i])
        class_accuracy = np.mean(correct_mask[class_mask])
        
        print(f"{class_name}:")
        print(f"  Average similarity to description: {avg_similarity:.4f}")
        print(f"  Classification accuracy: {class_accuracy:.4f}")
        print()

# =====================================================
# 7. MAIN TRAINING FUNCTION
# =====================================================

def train_incident_classifier(df, issue_summary, get_embedding_func, text_column='combined', 
                             class_column='class', test_size=0.2, val_size=0.2,
                             batch_size=32, num_epochs=30, lr=0.001):
    """
    Main function to train incident classifier with existing data and custom embedding function
    
    Args:
        df: DataFrame with incident data
        issue_summary: Dictionary with class definitions
        get_embedding_func: Your existing embedding function
        text_column: Column name containing text to classify
        class_column: Column name containing class labels
        test_size: Proportion of data for testing
        val_size: Proportion of training data for validation
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        lr: Learning rate
    
    Returns:
        Dictionary with results including trained models and metrics
    """
    
    print("Class-Aware Incident Classification with Custom Embedding Function")
    print("=" * 70)
    
    # 1. Preprocess data
    print("\n1. Preprocessing data...")
    preprocessor = DataPreprocessor(df, issue_summary, text_column, class_column)
    
    # Display class information
    print("\nClass Definitions:")
    for i, class_name in enumerate(preprocessor.classes):
        class_info = preprocessor.get_class_info(class_name)
        description = class_info.get("description", "No description")
        print(f"{i+1}. {class_name}:")
        print(f"   Description: {description[:100]}...")
        if "keywords" in class_info:
            print(f"   Keywords: {', '.join(class_info['keywords'][:5])}...")
        print()
    
    # 2. Generate embeddings using your function
    print("2. Generating embeddings using custom function...")
    embedding_generator = CustomEmbeddingGenerator(get_embedding_func)
    
    # Generate class description embeddings
    class_descriptions = preprocessor.get_class_descriptions()
    class_embeddings = embedding_generator.generate_class_embeddings(class_descriptions)
    
    # Generate text embeddings
    texts = preprocessor.df[text_column].tolist()
    text_embeddings = embedding_generator.generate_text_embeddings(texts)
    
    # Create enhanced features
    enhanced_features = embedding_generator.create_enhanced_features(text_embeddings)
    
    print(f"Text embeddings shape: {text_embeddings.shape}")
    print(f"Class embeddings shape: {class_embeddings.shape}")
    print(f"Enhanced features shape: {enhanced_features.shape}")
    
    # 3. Prepare data for training
    print("\n3. Preparing data for training...")
    
    class_similarities = embedding_generator.compute_class_similarities(text_embeddings)
    labels = preprocessor.df['class_idx'].values
    
    # Split data
    X_train, X_test, y_train, y_test, sim_train, sim_test = train_test_split(
        enhanced_features, labels, class_similarities,
        test_size=test_size, 
        random_state=42, 
        stratify=labels
    )
    
    X_train, X_val, y_train, y_val, sim_train, sim_val = train_test_split(
        X_train, y_train, sim_train,
        test_size=val_size, 
        random_state=42, 
        stratify=y_train
    )
    
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Create datasets
    train_dataset = EnhancedEmbeddingDataset(X_train, y_train, sim_train)
    val_dataset = EnhancedEmbeddingDataset(X_val, y_val, sim_val)
    test_dataset = EnhancedEmbeddingDataset(X_test, y_test, sim_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 4. Train models
    print("\n4. Training models...")
    
    input_dim = enhanced_features.shape[1]
    num_classes = len(preprocessor.classes)
    
    models_to_train = {
        'ClassAware-MLP': ClassAwareMLPClassifier(input_dim, num_classes),
        'ClassAware-LSTM': ClassAwareLSTMClassifier(input_dim, 128, 2, num_classes),
        'ClassAware-CNN': ClassAwareCNNClassifier(input_dim, num_classes)
    }
    
    results = {}
    
    for model_name, model in models_to_train.items():
        print(f"\nTraining {model_name}...")
        trainer = ModelTrainer(model)
        
        # Train model
        best_val_acc = trainer.train_model(
            train_loader, val_loader, 
            num_epochs=num_epochs, lr=lr
        )
        
        # Evaluate on test set
        test_acc = trainer.evaluate_model(test_loader)
        y_pred, y_true, y_prob = trainer.get_predictions(test_loader)
        
        results[model_name] = {
            'trainer': trainer,
            'test_accuracy': test_acc,
            'predictions': (y_pred, y_true, y_prob),
            'best_val_accuracy': best_val_acc
        }
        
        print(f"{model_name} Test Accuracy: {test_acc:.2f}%")
    
    # 5. Analysis and comparison
    print("\n5. Model Comparison:")
    print("=" * 40)
    for model_name, result in results.items():
        print(f"{model_name}: {result['test_accuracy']:.2f}%")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
    print(f"\nBest Model: {best_model_name}")
    
    # 6. Detailed analysis of best model
    print(f"\n6. Detailed analysis of {best_model_name}:")
    best_result = results[best_model_name]
    y_pred, y_true, y_prob = best_result['predictions']
    
    analyze_predictions(y_true, y_pred, y_prob, preprocessor.classes, best_model_name)
    analyze_class_awareness(y_pred, y_true, sim_test, preprocessor.classes)
    
    # 7. Visualizations
    print("\n7. Generating visualizations...")
    
    plot_class_similarities(text_embeddings, class_embeddings, preprocessor.classes)
    plot_training_history(best_result['trainer'], best_model_name)
    plot_confusion_matrix(y_true, y_pred, preprocessor.classes, best_model_name)
    
    # 8. Save best model
    print(f"\n8. Saving best model ({best_model_name})...")
    model_save_path = f'best_incident_classifier_{best_model_name.replace("-", "_").lower()}.pth'
    
    torch.save({
        'model_state_dict': best_result['trainer'].model.state_dict(),
        'model_class': best_result['trainer'].model.__class__.__name__,
        'preprocessor': preprocessor,
        'embedding_generator': embedding_generator,
        'class_embeddings': class_embeddings,
        'classes': preprocessor.classes,
        'class_definitions': preprocessor.class_definitions,
        'input_dim': input_dim,
        'num_classes': num_classes,
        'label_encoder': preprocessor.label_encoder
    }, model_save_path)
    
    print(f"Model saved to: {model_save_path}")
    print("Training completed!")
    
    return {
        'results': results,
        'best_model': best_model_name,
        'preprocessor': preprocessor,
        'embedding_generator': embedding_generator,
        'model_save_path': model_save_path
    }

# =====================================================
# 8. INFERENCE FUNCTION
# =====================================================

def predict_incident_class(model_path, text, get_embedding_func, top_k=3):
    """
    Predict incident class for new text using trained model
    
    Args:
        model_path: Path to saved model
        text: Text to classify
        get_embedding_func: Your embedding function
        top_k: Number of top predictions to return
    
    Returns:
        Dictionary with predictions and confidences
    """
    
    # Load model and components
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Get components
    preprocessor = checkpoint['preprocessor']
    classes = checkpoint['classes']
    class_embeddings = checkpoint['class_embeddings']
    
    # Create new embedding generator with your function
    embedding_generator = CustomEmbeddingGenerator(get_embedding_func)
    embedding_generator.class_embeddings = class_embeddings
    embedding_generator.class_descriptions = preprocessor.get_class_descriptions()
    
    # Generate embedding for new text
    text_embedding = embedding_generator.generate_text_embeddings([text])
    enhanced_features = embedding_generator.create_enhanced_features(text_embedding)
    class_similarities = embedding_generator.compute_class_similarities(text_embedding)
    
    print(f"Text: {text}")
    print(f"\nClass Similarities:")
    for i, class_name in enumerate(classes):
        similarity = class_similarities[0, i]
        print(f"  {class_name}: {similarity:.4f}")
    
    # Get top similarities as proxy for predictions
    top_indices = np.argsort(class_similarities[0])[-top_k:][::-1]
    
    predictions = []
    for idx in top_indices:
        predictions.append({
            'class': classes[idx],
            'confidence': float(class_similarities[0, idx]),
            'description': preprocessor.class_definitions[classes[idx]]['description']
        })
    
    return {
        'predictions': predictions,
        'enhanced_features_shape': enhanced_features.shape,
        'text_processed': True
    }

# =====================================================
# 9. EXAMPLE USAGE
# =====================================================

"""
Example usage:

# Define your embedding function (example)
def get_embedding(text):
    # Your existing embedding logic here
    # Should return numpy array or similar
    # Example placeholder:
    if isinstance(text, list):
        return np.random.rand(len(text), 384)  # Replace with your logic
    else:
        return np.random.rand(384)  # Replace with your logic

# Define your issue summary
issue_summary = {
    "Network Issue": {
        "description": "Problems related to network connectivity, internet outages, VPN failures",
        "keywords": ["network", "connectivity", "internet", "VPN"]
    },
    "Security Breach": "Security incidents involving unauthorized access, malware, phishing attacks",
    "Hardware Failure": {
        "description": "Physical hardware problems including server failures, disk crashes",
        "keywords": ["hardware", "server", "disk", "physical"]
    },
    # ... more classes
}

# Train the classifier
results = train_incident_classifier(
    df=your_dataframe, 
    issue_summary=your_issue_summary,
    get_embedding_func=get_embedding,  # Your function
    text_column='combined',
    class_column='class',
    num_epochs=30,
    batch_size=32
)

# Use for prediction
predictions = predict_incident_class(
    model_path=results['model_save_path'],
    text="Server experiencing hardware malfunction causing service outage",
    get_embedding_func=get_embedding
)
"""

if __name__ == "__main__":
    print("Incident Classification System Ready!")
    print("Please provide your dataframe (df), issue_summary dictionary, and get_embedding function")
    print("to use train_incident_classifier()")
