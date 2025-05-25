# CFM-DiT Parameter Optimization Strategy

## 📊 Parameter Comparison Overview

| Parameter           | Original DiT | CFM-DiT       | Improvement  | Reason                             |
| ------------------- | ------------ | ------------- | ------------ | ---------------------------------- |
| **Training Epochs** | 400          | 300           | 25% ↓        | CFM converges faster               |
| **Batch Size**      | 4            | 8             | 100% ↑       | CFM more stable with large batches |
| **Timesteps**       | 1000         | 500           | 50% ↓        | CFM needs fewer timesteps          |
| **Sampling Steps**  | 500          | 100           | 80% ↓        | ODE integration more efficient     |
| **Learning Rate**   | 0.0001       | 0.0001-0.0002 | Can increase | CFM training more stable           |

## 🎯 Core Optimization Principles

### 1. Timestep Optimization (1000 → 500)

**Original Diffusion**:

- Needs many timesteps to model complex noise scheduling
- Each timestep requires learning different denoising tasks
- More timesteps require larger model capacity

**CFM Advantages**:

- Uses continuous time t ∈ [0,1], no need for discretization
- Optimal transport paths are more direct, need less time resolution
- Vector field prediction more stable than noise prediction

```python
# Diffusion: Complex noise scheduling
α_t = cos(πt/2), σ_t = sin(πt/2)

# CFM: Simple linear interpolation
x_t = (1-(1-σ_min)*t) * x_0 + t * x_1
```

### 2. Sampling Step Optimization (500 → 100)

**Original Diffusion**:

- DDIM/DDPM requires multi-step denoising process
- Each step has cumulative errors
- Needs fine timestep scheduling

**CFM Advantages**:

- ODE integration directly from noise to data
- Deterministic paths, no randomness
- Euler method can achieve good results

```python
# CFM sampling: Simple ODE integration
for i in range(num_steps):
    v_pred = model(x, t)
    x = x - dt * v_pred  # Euler step
```

### 3. Batch Size Optimization (4 → 8)

**CFM Stability Advantages**:

- Vector field prediction smoother than noise prediction
- Smaller gradient variance, supports larger batches
- Higher memory efficiency (fewer timesteps)

**Practical Effects**:

- Faster training speed
- More stable gradient updates
- Better batch normalization effects

### 4. Training Epoch Optimization (400 → 300)

**CFM Convergence Advantages**:

- More direct training objective (vector field vs noise)
- More stable loss function
- Less mode collapse risk

## 🔬 Technical Deep Analysis

### CFM vs Diffusion Training Objectives

**Diffusion Training**:

```python
# Multiple possible targets
if target_type == "pred_eps":
    target = noise
elif target_type == "pred_x_0":
    target = original_data
elif target_type == "pred_v":
    target = α_t * noise - σ_t * data
```

**CFM Training**:

```python
# Unified vector field target
target = x_1 - (1 - σ_min) * x_0  # Always predict vector field
```

### Numerical Stability Comparison

**Diffusion Challenges**:

- Noise scheduling parameter sensitivity
- Uneven learning difficulty across timesteps
- Cumulative error problems

**CFM Advantages**:

- Linear interpolation paths numerically stable
- Similar learning difficulty at all time points
- Controllable ODE integration errors

## 📈 Experimental Validation Strategy

### 1. Progressive Parameter Tuning

**Stage 1: Basic Validation**

```bash
# Use conservative parameters to validate CFM basic functionality
--epochs 100 --batch-size 4 --timesteps 500 --sampling-timesteps 100
```

**Stage 2: Performance Optimization**

```bash
# Increase batch size and learning rate
--epochs 200 --batch-size 8 --learning-rate 0.0002
```

**Stage 3: Full Optimization**

```bash
# Use all optimization parameters
--epochs 300 --batch-size 8 --timesteps 500 --sampling-timesteps 100
```

### 2. Comparative Experiment Design

**Control Variables**:

- Same dataset and preprocessing
- Same network architecture (DiT-XL/2)
- Same evaluation metrics (FSS)

**Variable Parameters**:

- Training algorithm: Diffusion vs CFM
- Parameter configuration: Original vs Optimized

### 3. Performance Monitoring Metrics

**Training Efficiency**:

- Training time per epoch
- Memory usage
- GPU utilization

**Model Quality**:

- Training loss convergence speed
- Validation FSS scores
- Final sampling quality

## ⚙️ Parameter Tuning Recommendations

### Learning Rate Tuning

```python
# CFM can use higher learning rates
learning_rates = [0.0001, 0.0002, 0.0005]

# Reason: Vector field prediction more stable
# Recommendation: Start with 0.0002, observe convergence
```

### Batch Size Tuning

```python
# CFM supports larger batches
batch_sizes = [4, 8, 16, 32]

# Limiting factor: GPU memory
# Recommendation: Use largest batch size memory allows
```

### Timestep Tuning

```python
# CFM needs fewer timesteps
timesteps = [250, 500, 1000]

# Experience: 500 steps usually sufficient
# Recommendation: Start with 500, increase if results poor
```

### Model Size Considerations

```python
# DiT model size options
dit_models = ["DiT-S/4", "DiT-B/4", "DiT-L/4", "DiT-XL/2"]

# CFM may benefit from larger models due to vector field complexity
# Recommendation: Start with DiT-L/4 or DiT-XL/2
```

## 🔍 Architecture-Specific Optimizations

### DiT-Specific Parameters

**Attention Mechanisms**:

- CFM's continuous time may benefit from temporal attention
- Consider increasing attention heads for better time modeling

**Positional Encoding**:

- Time embedding crucial for CFM
- May need larger embedding dimensions

**Layer Depth**:

- Vector field prediction may require deeper networks
- Consider increasing transformer layers

### Memory Optimization

**Gradient Checkpointing**:

```python
# Enable for larger models
--gradient-checkpointing
```

**Mixed Precision**:

```python
# Faster training with minimal quality loss
--mixed-precision
```

## 📊 Expected Results

### Training Metrics

- **Convergence**: 25-40% faster than diffusion
- **Stability**: Lower loss variance
- **Memory**: 10-20% more efficient

### Sampling Metrics

- **Speed**: 80% reduction in sampling time
- **Quality**: Comparable or better FSS scores
- **Consistency**: More deterministic outputs

### Resource Usage

- **GPU Memory**: More efficient due to fewer timesteps
- **Training Time**: Shorter overall training duration
- **Inference Speed**: Significantly faster sampling

## 🚀 Quick Start Configuration

### Recommended Starting Configuration

```bash
python shrimp_cfmT.py \
    --epochs 300 \
    --batch-size 8 \
    --timesteps 500 \
    --sampling-timesteps 100 \
    --learning-rate 0.0002 \
    --path-type optimal_transport \
    --sigma-min 0.001 \
    --dit-model DiT-L/4 \
    --train-model
```

### Advanced Configuration

```bash
python shrimp_cfmT.py \
    --epochs 250 \
    --batch-size 16 \
    --timesteps 250 \
    --sampling-timesteps 50 \
    --learning-rate 0.0003 \
    --path-type optimal_transport \
    --sigma-min 0.0005 \
    --dit-model DiT-XL/2 \
    --gradient-checkpointing \
    --mixed-precision \
    --train-model
```

This optimization strategy leverages CFM's theoretical advantages while maintaining practical compatibility with existing DiT architectures and workflows.
