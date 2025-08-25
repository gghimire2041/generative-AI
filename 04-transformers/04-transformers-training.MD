# Transformers: Full Architecture & Training Implementation

*Part 2 of 4 - Advanced Implementation Guide for Senior Data Scientists*

## Table of Contents
1. [Complete Transformer Architecture](#complete-architecture)
2. [Training Dynamics & Optimization](#training-optimization)
3. [Loss Functions & Training Pipeline](#loss-training)

---

## Complete Transformer Architecture {#complete-architecture}

### Full Transformer Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, norm_first=True):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm_first = norm_first
    
    def forward(self, x, mask=None):
        if self.norm_first:
            # Pre-normalization
            x = x + self.dropout1(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask))
            x = x + self.dropout2(self.feed_forward(self.norm2(x)))
        else:
            # Post-normalization
            x = self.norm1(x + self.dropout1(self.self_attn(x, x, x, mask)))
            x = self.norm2(x + self.dropout2(self.feed_forward(x)))
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, norm_first=True):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm_first = norm_first
    
    def forward(self, x, encoder_output=None, src_mask=None, tgt_mask=None):
        if self.norm_first:
            # Self-attention
            x = x + self.dropout1(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), tgt_mask))
            # Cross-attention
            if encoder_output is not None:
                x = x + self.dropout2(self.cross_attn(self.norm2(x), encoder_output, encoder_output, src_mask))
            # Feed-forward
            x = x + self.dropout3(self.feed_forward(self.norm3(x)))
        else:
            # Post-normalization
            x = self.norm1(x + self.dropout1(self.self_attn(x, x, x, tgt_mask)))
            if encoder_output is not None:
                x = self.norm2(x + self.dropout2(self.cross_attn(x, encoder_output, encoder_output, src_mask)))
            x = self.norm3(x + self.dropout3(self.feed_forward(x)))
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048,
                 max_len=5000, dropout=0.1, norm_first=True):
        super().__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Encoder stack
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, norm_first)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder stack
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout, norm_first)
            for _ in range(num_decoder_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        self.init_weights()
    
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(self, src, src_mask=None):
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x
    
    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        return self.output_projection(decoder_output)

# Utility functions
def create_padding_mask(seq, pad_token=0):
    return (seq != pad_token).unsqueeze(1).unsqueeze(2)

def create_causal_mask(size, device=None):
    mask = torch.tril(torch.ones(size, size, device=device))
    return mask.unsqueeze(0).unsqueeze(0)

def create_combined_mask(tgt, pad_token=0):
    tgt_len = tgt.size(1)
    device = tgt.device
    causal_mask = create_causal_mask(tgt_len, device)
    padding_mask = create_padding_mask(tgt, pad_token)
    return causal_mask & padding_mask
```

---

## Training Dynamics & Optimization {#training-optimization}

### Learning Rate Scheduling

```python
class TransformerLRScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
    
    def step(self):
        self.step_num += 1
        lr = self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5)
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

class CosineAnnealingWarmup:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.step_num = 0
    
    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            lr = self.base_lr * self.step_num / self.warmup_steps
        else:
            progress = (self.step_num - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress)) / 2
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

# Optimizer factory
def create_optimizer(model, optimizer_type='adamw', lr=1e-4, weight_decay=0.01):
    decay = []
    no_decay = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or "norm" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    
    params = [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0}
    ]
    
    if optimizer_type.lower() == 'adamw':
        return torch.optim.AdamW(params, lr=lr)
    elif optimizer_type.lower() == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")
```

### Mixed Precision Training

```python
class MixedPrecisionTrainer:
    def __init__(self, model, optimizer, scheduler=None, max_grad_norm=1.0):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = torch.cuda.amp.GradScaler()
        self.max_grad_norm = max_grad_norm
    
    def training_step(self, batch, criterion):
        src, tgt = batch
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        src_mask = create_padding_mask(src)
        tgt_mask = create_combined_mask(tgt_input)
        
        self.optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            logits = self.model(src, tgt_input, src_mask, tgt_mask)
            loss = criterion(
                logits.contiguous().view(-1, logits.size(-1)),
                tgt_output.contiguous().view(-1)
            )
        
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        if self.scheduler:
            self.scheduler.step()
        
        return loss.item()

# Gradient accumulation
class GradientAccumulator:
    def __init__(self, model, optimizer, accumulation_steps=4, max_grad_norm=1.0):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.scaler = torch.cuda.amp.GradScaler()
        self.max_grad_norm = max_grad_norm
        self.accumulated_steps = 0
    
    def accumulate_step(self, batch, criterion):
        src, tgt = batch
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        src_mask = create_padding_mask(src)
        tgt_mask = create_combined_mask(tgt_input)
        
        with torch.cuda.amp.autocast():
            logits = self.model(src, tgt_input, src_mask, tgt_mask)
            loss = criterion(
                logits.contiguous().view(-1, logits.size(-1)),
                tgt_output.contiguous().view(-1)
            ) / self.accumulation_steps
        
        self.scaler.scale(loss).backward()
        self.accumulated_steps += 1
        
        return loss.item() * self.accumulation_steps
    
    def step(self):
        if self.accumulated_steps != self.accumulation_steps:
            return False
        
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        
        self.accumulated_steps = 0
        return True
```

---

## Loss Functions & Training Pipeline {#loss-training}

### Advanced Loss Functions

```python
class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, smoothing=0.1, ignore_index=-100):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target):
        batch_size = pred.size(0)
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.vocab_size - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        if self.ignore_index >= 0:
            mask = (target != self.ignore_index).float()
            true_dist = true_dist * mask.unsqueeze(1)
            mask_sum = mask.sum()
        else:
            mask_sum = batch_size
        
        loss = F.kl_div(F.log_softmax(pred, dim=1), true_dist, reduction='none').sum(dim=1)
        
        if self.ignore_index >= 0:
            loss = loss * mask
        
        return loss.sum() / mask_sum if mask_sum > 0 else loss.sum()

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
    
    def forward(self, pred, target):
        ce_loss = self.ce_loss(pred, target)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
```

### Training Metrics and Evaluation

```python
class TransformerMetrics:
    def __init__(self, vocab_size, ignore_index=-100):
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        self.total_loss = 0
        self.total_tokens = 0
        self.correct_predictions = 0
        self.total_predictions = 0
    
    def update(self, logits, targets, loss):
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        
        if self.ignore_index >= 0:
            mask = (targets_flat != self.ignore_index)
        else:
            mask = torch.ones_like(targets_flat, dtype=torch.bool)
        
        valid_tokens = mask.sum().item()
        
        if valid_tokens > 0:
            predictions = logits_flat.argmax(dim=-1)
            correct = (predictions == targets_flat) & mask
            
            self.total_loss += loss * valid_tokens
            self.total_tokens += valid_tokens
            self.correct_predictions += correct.sum().item()
            self.total_predictions += valid_tokens
    
    def compute(self):
        if self.total_tokens == 0:
            return {'loss': 0, 'accuracy': 0, 'perplexity': float('inf')}
        
        avg_loss = self.total_loss / self.total_tokens
        accuracy = self.correct_predictions / self.total_predictions
        perplexity = math.exp(min(avg_loss, 100))
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'perplexity': perplexity
        }

def compute_bleu_score(predictions, references, max_n=4):
    """Compute BLEU score for sequence evaluation"""
    from collections import Counter
    
    def get_ngrams(sequence, n):
        return [tuple(sequence[i:i+n]) for i in range(len(sequence) - n + 1)]
    
    precisions = []
    for n in range(1, max_n + 1):
        total_overlap = 0
        total_pred_count = 0
        
        for pred, ref in zip(predictions, references):
            pred_ngrams = Counter(get_ngrams(pred, n))
            ref_ngrams = Counter(get_ngrams(ref, n))
            
            overlap = sum(min(pred_ngrams[ng], ref_ngrams[ng]) for ng in pred_ngrams)
            total_overlap += overlap
            total_pred_count += len(get_ngrams(pred, n))
        
        precisions.append(total_overlap / total_pred_count if total_pred_count > 0 else 0)
    
    if min(precisions) > 0:
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    else:
        geo_mean = 0
    
    # Brevity penalty
    pred_len = sum(len(pred) for pred in predictions)
    ref_len = sum(len(ref) for ref in references)
    
    if pred_len > ref_len:
        bp = 1
    else:
        bp = math.exp(1 - ref_len / pred_len) if pred_len > 0 else 0
    
    return bp * geo_mean
```

### Complete Training Pipeline

```python
class TransformerTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, device, 
                 scheduler=None, criterion=None, max_grad_norm=1.0):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        
        if criterion is None:
            self.criterion = LabelSmoothingLoss(
                vocab_size=model.output_projection.out_features,
                smoothing=0.1,
                ignore_index=-100
            )
        else:
            self.criterion = criterion
        
        # Mixed precision setup
        self.scaler = torch.cuda.amp.GradScaler()
        
        # Metrics
        self.train_metrics = TransformerMetrics(model.output_projection.out_features)
        self.val_metrics = TransformerMetrics(model.output_projection.out_features)
    
    def train_epoch(self):
        self.model.train()
        self.train_metrics.reset()
        
        for batch_idx, (src, tgt) in enumerate(self.train_loader):
            src, tgt = src.to(self.device), tgt.to(self.device)
            
            # Prepare input and target
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # Create masks
            src_mask = create_padding_mask(src)
            tgt_mask = create_combined_mask(tgt_input)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                logits = self.model(src, tgt_input, src_mask, tgt_mask)
                loss = self.criterion(
                    logits.contiguous().view(-1, logits.size(-1)),
                    tgt_output.contiguous().view(-1)
                )
            
            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.scheduler:
                self.scheduler.step()
            
            # Update metrics
            self.train_metrics.update(logits, tgt_output, loss.item())
            
            if batch_idx % 100 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}')
        
        return self.train_metrics.compute()
    
    def validate(self):
        self.model.eval()
        self.val_metrics.reset()
        
        with torch.no_grad():
            for src, tgt in self.val_loader:
                src, tgt = src.to(self.device), tgt.to(self.device)
                
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                src_mask = create_padding_mask(src)
                tgt_mask = create_combined_mask(tgt_input)
                
                with torch.cuda.amp.autocast():
                    logits = self.model(src, tgt_input, src_mask, tgt_mask)
                    loss = self.criterion(
                        logits.contiguous().view(-1, logits.size(-1)),
                        tgt_output.contiguous().view(-1)
                    )
                
                self.val_metrics.update(logits, tgt_output, loss.item())
        
        return self.val_metrics.compute()
    
    def train(self, num_epochs, save_path=None, eval_every=1):
        """Complete training loop"""
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch + 1}/{num_epochs}')
            
            # Train
            train_metrics = self.train_epoch()
            print(f'Train - Loss: {train_metrics["loss"]:.4f}, '
                  f'Accuracy: {train_metrics["accuracy"]:.4f}, '
                  f'Perplexity: {train_metrics["perplexity"]:.2f}')
            
            # Validate
            if (epoch + 1) % eval_every == 0:
                val_metrics = self.validate()
                print(f'Val - Loss: {val_metrics["loss"]:.4f}, '
                      f'Accuracy: {val_metrics["accuracy"]:.4f}, '
                      f'Perplexity: {val_metrics["perplexity"]:.2f}')
                
                # Save best model
                if save_path and val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                        'epoch': epoch,
                        'val_loss': val_metrics["loss"],
                        'val_accuracy': val_metrics["accuracy"]
                    }, save_path)
                    print(f'Best model saved to {save_path}')

# Usage example
def create_training_setup(model, train_loader, val_loader, device):
    """Factory function to create complete training setup"""
    
    # Optimizer
    optimizer = create_optimizer(model, 'adamw', lr=1e-4, weight_decay=0.01)
    
    # Scheduler
    scheduler = TransformerLRScheduler(optimizer, model.d_model, warmup_steps=4000)
    
    # Loss function
    criterion = LabelSmoothingLoss(
        vocab_size=model.output_projection.out_features,
        smoothing=0.1
    )
    
    # Trainer
    trainer = TransformerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        criterion=criterion,
        max_grad_norm=1.0
    )
    
    return trainer
```

---

## Summary

This document provides the complete transformer architecture and training implementation:

**Key Architecture Components:**
1. **Encoder-Decoder Structure**: Full implementation with proper masking
2. **Layer Normalization**: Pre/post-norm variants for training stability
3. **Residual Connections**: Essential for deep network training
4. **Mask Utilities**: Padding, causal, and combined mask creation

**Training Essentials:**
1. **Learning Rate Scheduling**: Warmup + decay strategies
2. **Mixed Precision**: Memory efficiency and speed optimization
3. **Gradient Accumulation**: Large effective batch sizes
4. **Advanced Loss Functions**: Label smoothing and focal loss

**Production Features:**
1. **Comprehensive Metrics**: Loss, accuracy, perplexity tracking
2. **Model Checkpointing**: Best model saving and restoration
3. **Evaluation Tools**: BLEU score computation
4. **Complete Training Loop**: Ready-to-use training pipeline

**Next Documents:**
- Part 3: Advanced Techniques (Beam Search, Attention Variants, Optimization)
- Part 4: Production Deployment (Serving, Monitoring, Scaling)
