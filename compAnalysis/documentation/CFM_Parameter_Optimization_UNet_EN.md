# Flow Matching Parameter Optimization Guide for UNet

## Principles of Parameter Adjustment from Diffusion to Flow Matching

### 🚀 Core Optimization Principles

Flow Matching has the following key advantages over Diffusion, which guide our parameter optimization:

1. **More Direct Paths**: OT paths are more direct than diffusion noise scheduling
2. **Faster Convergence**: More stable training process, requires fewer epochs
3. **More Efficient Sampling**: Requires fewer function evaluations
4. **Better Batch Stability**: More friendly to large batch sizes

---

## 📊 Parameter Comparison Analysis

### 1. **Training Efficiency Parameters**

| Parameter       | Diffusion | Flow Matching    | Reason for Change                           |
| --------------- | --------- | ---------------- | ------------------------------------------- |
| `epochs`        | 400       | 300              | CFM converges faster, 25% reduction         |
| `batch_size`    | 4         | 8                | CFM more stable with large batches          |
| `learning_rate` | 0.0001    | [0.0001, 0.0002] | Test higher learning rates, CFM more stable |

**Principle**: Flow Matching's loss function is more direct (directly predicting vector fields), avoiding complex noise scheduling and reparameterization in diffusion, making training more stable and fast.

### 2. **Timestep Optimization**

| Parameter            | Diffusion | Flow Matching | Reason for Change                      |
| -------------------- | --------- | ------------- | -------------------------------------- |
| `timesteps`          | 1000      | 500           | CFM can use fewer training steps       |
| `sampling_timesteps` | 500       | 100           | CFM sampling efficiency, 80% reduction |

**Principle**:

- **During Training**: CFM doesn't need complex noise scheduling, 500 steps sufficient to cover noise-to-data path
- **During Sampling**: ODE integration more efficient than DDIM/DDPM, 100 steps can achieve high-quality samples

### 3. **Flow Matching Specific Parameters**

| Parameter   | Values                                  | Description                         |
| ----------- | --------------------------------------- | ----------------------------------- |
| `path_type` | ["optimal_transport", "diffusion_like"] | Test two path types                 |
| `sigma_min` | [0.001, 0.01]                           | Test different minimum noise levels |

**Principle**:

- **Optimal Transport**: Provides most direct paths, usually better performance
- **Diffusion-like**: Maintains behavior similar to original diffusion, convenient for comparison
- **sigma_min**: Controls noise level at path endpoints, affects generation quality

### 4. **Model Capacity Adjustment**

| Parameter     | Diffusion | Flow Matching | Reason for Change                                                  |
| ------------- | --------- | ------------- | ------------------------------------------------------------------ |
| `embed_dim`   | 64        | [64, 128]     | Test larger capacity, CFM may benefit from stronger expressiveness |
| `num_workers` | 2         | 4             | Improve data loading efficiency                                    |

**Principle**: CFM's vector field prediction may be more complex than diffusion's noise prediction, larger model capacity may bring better performance.

---

## 🎯 Experimental Design Strategy

### Current configuration will produce experiment combinations:

```
Total experiments = 2(path_type) × 2(sigma_min) × 2(embed_dim) × 2(learning_rate) × 2(history_frames)
                  = 32 experiment combinations
```

### Recommended Experiment Priority:

#### **First Priority** (Most promising configuration):

```python
{
    "path_type": "optimal_transport",
    "sigma_min": 0.001,
    "embed_dim": 128,
    "learning_rate": 0.0002,
    "batch_size": 8,
    "timesteps": 500,
    "sampling_timesteps": 100
}
```

#### **Second Priority** (Comparison baseline):

```python
{
    "path_type": "diffusion_like",
    "sigma_min": 0.001,
    "embed_dim": 64,
    "learning_rate": 0.0001,
    # Other parameters same as above
}
```

---

## 📈 Expected Performance Improvements

Based on papers and theoretical analysis, expected improvements:

| Metric                | Expected Improvement | Reason                       |
| --------------------- | -------------------- | ---------------------------- |
| **Training Time**     | 30-50% reduction     | Fewer epochs + larger batch  |
| **Sampling Speed**    | 80% reduction        | 100 vs 500 sampling steps    |
| **Memory Efficiency** | 10-20% improvement   | Larger batch size feasible   |
| **Sample Quality**    | 5-15% improvement    | More direct generation paths |

---

## ⚠️ Important Notes

### 1. **Resource Requirement Changes**

- **Memory**: batch_size doubled, requires more GPU memory
- **Computation**: embed_dim increased, computation increases
- **Time**: Overall training time should decrease

### 2. **Monitoring Metrics**

- **Convergence Speed**: Observe validation loss descent curve
- **Sampling Quality**: Compare generated samples' FSS scores
- **Training Stability**: Observe loss fluctuation degree

### 3. **Debugging Suggestions**

If encountering problems:

1. First test with small parameters (embed_dim=64, batch_size=4)
2. Gradually increase parameter scale
3. Monitor GPU memory usage

---

## 🔧 Quick Test Command

Test optimal configuration:

```bash
python3 shrimp_cfmU.py \
    --epochs 50 \
    --batch-size 8 \
    --timesteps 500 \
    --sampling-timesteps 100 \
    --path-type optimal_transport \
    --sigma-min 0.001 \
    --embed-dim 128 \
    --learning-rate 0.0002 \
    --train-model
```

This configuration should achieve better performance in shorter time!

## 🔬 Technical Deep Dive

### Why These Parameters Work Better for CFM

#### **Batch Size Increase (4→8)**

- **CFM Advantage**: Vector field prediction is smoother than noise prediction
- **Gradient Stability**: Lower gradient variance supports larger batches
- **Memory Efficiency**: Fewer timesteps allow larger batches within same memory

#### **Timestep Reduction (1000→500)**

- **Training**: CFM uses continuous time t∈[0,1], doesn't need fine discretization
- **Sampling**: ODE integration converges faster than iterative denoising
- **Quality**: Optimal transport paths are inherently more efficient

#### **Learning Rate Flexibility**

- **Stability**: Vector field regression is more stable than noise prediction
- **Convergence**: Direct optimization target reduces training complexity
- **Robustness**: Less sensitive to hyperparameter choices

### Mathematical Foundation

**Diffusion Training Objective**:

```
L = E[||ε_θ(x_t, t) - ε||²]  # Predict noise
```

**CFM Training Objective**:

```
L = E[||v_θ(x_t, t) - u_t||²]  # Predict vector field
```

The CFM objective is more direct and stable, enabling the parameter optimizations described above.

## 📊 Validation Strategy

### Progressive Testing

1. **Baseline Validation**: Start with conservative parameters
2. **Incremental Optimization**: Gradually apply optimizations
3. **Performance Comparison**: Compare against original diffusion baseline

### Key Metrics to Monitor

- **Training Loss**: Should converge faster than diffusion
- **FSS Scores**: Should achieve comparable or better values
- **Sampling Quality**: Visual inspection of generated samples
- **Resource Usage**: Memory and time efficiency gains

This optimization guide provides a systematic approach to leveraging CFM's advantages while maintaining compatibility with existing workflows.
