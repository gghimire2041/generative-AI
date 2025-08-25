# Transformers: Advanced Techniques & Optimization

*Part 3 of 4 - Advanced Implementation Guide for Senior Data Scientists*

## Table of Contents
1. [Advanced Generation Techniques](#generation-techniques)
2. [Architectural Improvements](#architectural-improvements)
3. [Performance Optimization](#performance-optimization)
4. [Knowledge Distillation](#knowledge-distillation)

---

## Advanced Generation Techniques {#generation-techniques}

### Beam Search Implementation

```python
import torch
import torch.nn.functional as F

class BeamSearch:
    def __init__(self, model, beam_size=5, max_len=100, bos_token=1, eos_token=2):
        self.model = model
        self.beam_size = beam_size
        self.max_len = max_len
        self.bos_token = bos_token
        self.eos_token = eos_token
    
    def search(self, src, src_mask=None, length_penalty=1.0):
        self.model.eval()
        device = src.device
        
        # Encode source
        with torch.no_grad():
            encoder_output = self.model.encode(src, src_mask)
        
        # Initialize beam
        sequences = torch.full((self.beam_size, 1), self.bos_token, device=device)
        scores = torch.zeros(self.beam_size, device=device)
        
        for step in range(self.max_len - 1):
            current_len = sequences.size(1)
            tgt_mask = create_causal_mask(current_len, device)
            
            # Decode all beams
            with torch.no_grad():
                decoder_output = self.model.decode(
                    sequences, encoder_output.expand(self.beam_size, -1, -1),
                    src_mask, tgt_mask
                )
                next_token_logits = self.model.output_projection(decoder_output[:, -1, :])
                next_token_probs = F.log_softmax(next_token_logits, dim=-1)
            
            # Calculate candidate scores
            vocab_size = next_token_probs.size(-1)
            candidate_scores = scores.unsqueeze(1) + next_token_probs
            candidate_scores = candidate_scores.view(-1)
            
            # Select top candidates
            top_scores, top_indices = torch.topk(candidate_scores, self.beam_size)
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size
            
            # Update sequences
            new_sequences = []
            for i in range(self.beam_size):
                beam_idx = beam_indices[i]
                token_idx = token_indices[i]
                new_seq = torch.cat([sequences[beam_idx], token_idx.unsqueeze(0)])
                new_sequences.append(new_seq)
            
            sequences = torch.stack(new_sequences)
            scores = top_scores
            
            if (sequences[:, -1] == self.eos_token).all():
                break
        
        # Apply length penalty and return best
        length_norm = (sequences.size(1) ** length_penalty) / (6 ** length_penalty)
        normalized_scores = scores / length_norm
        best_idx = normalized_scores.argmax()
        
        return sequences[best_idx], scores[best_idx].item()

class NucleusSampling:
    def __init__(self, model, p=0.9, temperature=1.0, max_len=100, bos_token=1, eos_token=2):
        self.model = model
        self.p = p
        self.temperature = temperature
        self.max_len = max_len
        self.bos_token = bos_token
        self.eos_token = eos_token
    
    def sample(self, src, src_mask=None):
        self.model.eval()
        device = src.device
        
        with torch.no_grad():
            encoder_output = self.model.encode(src, src_mask)
        
        sequence = torch.tensor([[self.bos_token]], device=device)
        
        for step in range(self.max_len - 1):
            tgt_mask = create_causal_mask(sequence.size(1), device)
            
            with torch.no_grad():
                decoder_output = self.model.decode(sequence, encoder_output, src_mask, tgt_mask)
                next_token_logits = self.model.output_projection(decoder_output[:, -1, :])
                
                # Apply temperature
                next_token_logits = next_token_logits / self.temperature
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Sort and find nucleus
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Create nucleus mask
                nucleus_mask = cumsum_probs <= self.p
                nucleus_mask[:, 0] = True  # Always include top token
                
                # Zero out non-nucleus tokens
                sorted_probs[~nucleus_mask] = 0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                
                # Sample from nucleus
                sampled_index = torch.multinomial(sorted_probs, 1)
                next_token = sorted_indices.gather(-1, sampled_index)
                
                sequence = torch.cat([sequence, next_token], dim=1)
                
                if next_token.item() == self.eos_token:
                    break
        
        return sequence.squeeze(0)
```

---

## Architectural Improvements {#architectural-improvements}

### Rotary Position Embeddings (RoPE)

```python
import math

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]
        
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        
        return freqs.cos(), freqs.sin()

def apply_rotary_emb(x, cos, sin):
    """Apply rotary embedding to input tensor"""
    x1 = x[..., ::2]   # Even dimensions
    x2 = x[..., 1::2]  # Odd dimensions
    
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos
    
    return torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)

class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.rotary_emb = RotaryPositionalEmbedding(self.d_k)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear projections and reshape
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply rotary embeddings
        cos, sin = self.rotary_emb(Q, seq_len)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        
        Q = apply_rotary_emb(Q, cos, sin)
        K = apply_rotary_emb(K, cos, sin)
        
        # Standard attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores.masked_fill_(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.w_o(context)

# SwiGLU Activation Function
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, bias=False):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=bias)  # Gate
        self.w2 = nn.Linear(d_model, d_ff, bias=bias)  # Up projection
        self.w3 = nn.Linear(d_ff, d_model, bias=bias)  # Down projection
    
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

# Group Query Attention (Memory Efficient)
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, dropout=0.1):
        super().__init__()
        assert num_heads % num_kv_heads == 0
        
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, num_heads * self.d_k, bias=False)
        self.w_k = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.w_v = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Projections
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)
        
        # Repeat K and V for grouped attention
        K = K.repeat_interleave(self.num_queries_per_kv, dim=1)
        V = V.repeat_interleave(self.num_queries_per_kv, dim=1)
        
        # Attention computation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores.masked_fill_(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.w_o(context)
```

---

## Performance Optimization {#performance-optimization}

### Memory-Efficient Techniques

```python
# Gradient Checkpointing for Memory Efficiency
class CheckpointedTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        if self.training:
            # Use gradient checkpointing during training
            x = x + torch.utils.checkpoint.checkpoint(self._attention_block, x, mask)
            x = x + torch.utils.checkpoint.checkpoint(self._ffn_block, x)
        else:
            x = x + self._attention_block(x, mask)
            x = x + self._ffn_block(x)
        return x
    
    def _attention_block(self, x, mask):
        return self.dropout1(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask))
    
    def _ffn_block(self, x):
        return self.dropout2(self.feed_forward(self.norm2(x)))

# KV-Cache for Generation
class KVCache:
    def __init__(self, max_batch_size, max_seq_len, num_heads, d_k, device):
        self.k_cache = torch.zeros(max_batch_size, num_heads, max_seq_len, d_k, device=device)
        self.v_cache = torch.zeros(max_batch_size, num_heads, max_seq_len, d_k, device=device)
        self.cache_len = torch.zeros(max_batch_size, dtype=torch.long, device=device)
    
    def update(self, batch_idx, k, v):
        seq_len = k.size(2)
        start_pos = self.cache_len[batch_idx]
        self.k_cache[batch_idx, :, start_pos:start_pos + seq_len] = k[0]
        self.v_cache[batch_idx, :, start_pos:start_pos + seq_len] = v[0]
        self.cache_len[batch_idx] += seq_len
    
    def get(self, batch_idx):
        cache_len = self.cache_len[batch_idx]
        return (self.k_cache[batch_idx:batch_idx+1, :, :cache_len], 
                self.v_cache[batch_idx:batch_idx+1, :, :cache_len])
    
    def reset(self, batch_idx=None):
        if batch_idx is not None:
            self.cache_len[batch_idx] = 0
        else:
            self.cache_len.fill_(0)

# Model Quantization
def quantize_model(model, quantization_type='dynamic'):
    """Apply quantization for deployment efficiency"""
    if quantization_type == 'dynamic':
        return torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    elif quantization_type == 'static':
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        # Calibration step would go here
        return torch.quantization.convert(model, inplace=False)
    else:
        raise ValueError(f"Unsupported quantization: {quantization_type}")

# Mixed Precision Utilities
class MixedPrecisionWrapper:
    def __init__(self, model):
        self.model = model
        self.scaler = torch.cuda.amp.GradScaler()
    
    def forward_with_loss(self, src, tgt, criterion):
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        src_mask = create_padding_mask(src)
        tgt_mask = create_combined_mask(tgt_input)
        
        with torch.cuda.amp.autocast():
            logits = self.model(src, tgt_input, src_mask, tgt_mask)
            loss = criterion(
                logits.contiguous().view(-1, logits.size(-1)),
                tgt_output.contiguous().view(-1)
            )
        
        return loss, logits
    
    def backward_step(self, loss, optimizer, max_grad_norm=1.0):
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        self.scaler.step(optimizer)
        self.scaler.update()
```

---

## Knowledge Distillation {#knowledge-distillation}

### Teacher-Student Training

```python
class TransformerDistillation:
    def __init__(self, teacher_model, student_model, temperature=4.0, alpha=0.7):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def distillation_loss(self, student_logits, teacher_logits, true_labels):
        """Compute knowledge distillation loss"""
        # Reshape tensors
        student_flat = student_logits.view(-1, student_logits.size(-1))
        teacher_flat = teacher_logits.view(-1, teacher_logits.size(-1))
        labels_flat = true_labels.view(-1)
        
        # Hard target loss (standard supervision)
        hard_loss = self.ce_loss(student_flat, labels_flat)
        
        # Soft target loss (knowledge transfer)
        student_soft = F.log_softmax(student_flat / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_flat / self.temperature, dim=-1)
        soft_loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        return total_loss, hard_loss, soft_loss
    
    def train_step(self, batch, student_optimizer):
        src, tgt = batch
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        src_mask = create_padding_mask(src)
        tgt_mask = create_combined_mask(tgt_input)
        
        # Teacher forward (no gradients)
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_logits = self.teacher_model(src, tgt_input, src_mask, tgt_mask)
        
        # Student forward
        self.student_model.train()
        student_logits = self.student_model(src, tgt_input, src_mask, tgt_mask)
        
        # Compute distillation loss
        total_loss, hard_loss, soft_loss = self.distillation_loss(
            student_logits, teacher_logits, tgt_output
        )
        
        # Backward pass
        student_optimizer.zero_grad()
        total_loss.backward()
        student_optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'hard_loss': hard_loss.item(),
            'soft_loss': soft_loss.item()
        }

# Model Pruning for Compression
class StructuredPruning:
    def __init__(self, model, pruning_ratio=0.2):
        self.model = model
        self.pruning_ratio = pruning_ratio
    
    def prune_attention_heads(self):
        """Remove least important attention heads"""
        for layer in self.model.encoder_layers + self.model.decoder_layers:
            if hasattr(layer, 'self_attn'):
                attention = layer.self_attn
                
                # Calculate head importance (simplified)
                head_importance = torch.rand(attention.num_heads)  # Replace with real importance metric
                
                # Determine heads to prune
                num_prune = int(attention.num_heads * self.pruning_ratio)
                heads_to_prune = head_importance.argsort()[:num_prune]
                
                # Create pruning mask
                keep_heads = torch.ones(attention.num_heads, dtype=torch.bool)
                keep_heads[heads_to_prune] = False
                
                # Apply pruning (simplified - would need careful implementation)
                self._prune_linear_layer(attention.w_q, keep_heads)
                self._prune_linear_layer(attention.w_k, keep_heads)
                self._prune_linear_layer(attention.w_v, keep_heads)
                
                attention.num_heads = keep_heads.sum().item()
    
    def _prune_linear_layer(self, layer, keep_mask):
        """Prune linear layer based on mask"""
        # This is a simplified version - real implementation would be more complex
        if hasattr(layer, 'weight'):
            layer.weight.data = layer.weight.data[keep_mask]
            if layer.bias is not None:
                layer.bias.data = layer.bias.data[keep_mask]

# Complete Training Pipeline with Advanced Techniques
class AdvancedTransformerTrainer:
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Mixed precision wrapper
        self.mp_wrapper = MixedPrecisionWrapper(model)
        
        # Optimizer and scheduler
        self.optimizer = create_optimizer(model, 'adamw', lr=config.lr)
        self.scheduler = TransformerLRScheduler(self.optimizer, model.d_model)
        
        # Loss function
        self.criterion = LabelSmoothingLoss(
            vocab_size=model.output_projection.out_features,
            smoothing=0.1
        )
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            src, tgt = [x.to(self.device) for x in batch]
            
            # Mixed precision forward pass
            loss, logits = self.mp_wrapper.forward_with_loss(src, tgt, self.criterion)
            
            # Mixed precision backward pass
            self.mp_wrapper.backward_step(loss, self.optimizer)
            
            # Update learning rate
            self.scheduler.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        return total_loss / len(self.train_loader)
    
    def train_with_distillation(self, teacher_model):
        """Train with knowledge distillation"""
        distiller = TransformerDistillation(teacher_model, self.model)
        
        self.model.train()
        total_metrics = {'total_loss': 0, 'hard_loss': 0, 'soft_loss': 0}
        
        for batch in self.train_loader:
            batch = [x.to(self.device) for x in batch]
            metrics = distiller.train_step(batch, self.optimizer)
            
            for key in total_metrics:
                total_metrics[key] += metrics[key]
        
        # Average metrics
        for key in total_metrics:
            total_metrics[key] /= len(self.train_loader)
        
        return total_metrics
```

---

## Summary

This document covers advanced transformer techniques:

**Generation Methods:**
- **Beam Search**: Systematic exploration for high-quality outputs
- **Nucleus Sampling**: Balanced diversity and quality
- **Contrastive Search**: Reduces repetition in generation

**Architectural Improvements:**
- **RoPE**: Better position encoding for long sequences
- **SwiGLU**: More efficient activation function
- **GQA**: Memory-efficient attention mechanism

**Performance Optimization:**
- **Gradient Checkpointing**: Memory-compute tradeoff
- **KV-Caching**: Efficient autoregressive generation
- **Mixed Precision**: Speed and memory optimization
- **Quantization**: Model compression for deployment

**Knowledge Transfer:**
- **Distillation**: Transfer knowledge from large to small models
- **Pruning**: Remove redundant parameters
- **Compression**: Efficient model storage and inference

**Next Document:**
- Part 4: Production Deployment (Serving, Monitoring, Scaling)
