# CFM-DiT: Conditional Flow Matching with Diffusion Transformer

This is an implementation that converts DiT (Diffusion Transformer) models from diffusion to Conditional Flow Matching (CFM).

## 🚀 Key Features

- **Faster Training**: CFM requires fewer training steps than diffusion
- **Faster Sampling**: Sampling steps reduced from 500 to 100 (80% improvement)
- **Better Stability**: Optimal transport paths are more direct than noise scheduling
- **Full Compatibility**: Maintains complete compatibility with original DiT architecture

## 📁 File Structure

```
cfmT/
├── src/
│   ├── flow_matching.py      # CFM core implementation (identical to cfmU)
│   ├── DiTModels.py          # DiT model architecture
│   ├── dataset.py            # Dataset processing
│   ├── blocks.py             # Network components
│   ├── utils.py              # Utility functions
│   └── DatasetBuilder.py     # Dataset builder
├── shrimp_cfmT.py           # Main training script
├── CfmT_cmd_generate.py     # Experiment command generator
└── README_CFM.md            # This file
```

## 🔧 Installation and Usage

### 1. Environment Requirements

```bash
# Same environment as original DiT
torch >= 1.9.0
numpy
matplotlib
tqdm
scipy
tensorboard
```

### 2. Train Model

```bash
# Basic training command
python shrimp_cfmT.py \
    --epochs 300 \
    --batch-size 8 \
    --timesteps 500 \
    --sampling-timesteps 100 \
    --learning-rate 0.0001 \
    --path-type optimal_transport \
    --sigma-min 0.001 \
    --dit-model DiT-XL/2 \
    --train-model \
    --sat-files-path /path/to/satellite/data \
    --rainfall-files-path /path/to/rainfall/data \
    --model-path ./models \
    --results ./results
```

### 3. Generate Experiment Commands

```bash
# Generate optimized CFM experiment configurations
python CfmT_cmd_generate.py

# This generates:
# - cfmT_experiments_summary.txt: All experiment commands
# - jobs/*.pbs: GADI cluster job scripts
```

### 4. Run on GADI

```bash
# Submit all jobs
cd jobs
for script in *.pbs; do qsub $script; done
```

## ⚙️ CFM-Specific Parameters

### New Parameters

- `--path-type`: Path type

  - `optimal_transport` (recommended): Optimal transport path
  - `diffusion_like`: Diffusion-like path

- `--sigma-min`: Minimum noise level (default: 0.001)
  - Controls noise amount at t=1
  - Smaller values → more deterministic paths

### Optimized Parameters

Compared to original DiT, CFM uses the following optimized parameters:

- **epochs**: 300 (reduced from 400)
- **batch_size**: 8 (increased from 4)
- **timesteps**: 500 (reduced from 1000)
- **sampling_timesteps**: 100 (reduced from 500)
- **learning_rate**: 0.0001-0.0002 (can use higher learning rates)

## 📊 Performance Comparison

| Metric          | Diffusion DiT | CFM DiT | Improvement |
| --------------- | ------------- | ------- | ----------- |
| Training Steps  | 1000          | 500     | 50% ↓       |
| Sampling Steps  | 500           | 100     | 80% ↓       |
| Training Epochs | 400           | 300     | 25% ↓       |
| Batch Size      | 4             | 8       | 100% ↑      |

## 🔬 Technical Details

### CFM vs Diffusion

**Diffusion Process**:

```
x_t = α_t * x_0 + σ_t * ε
Target: predict ε, x_0, or v
```

**CFM Process**:

```
x_t = (1-(1-σ_min)*t) * x_0 + t * x_1
Target: predict vector field u_t = x_1 - (1-σ_min)*x_0
```

### Optimal Transport Path

CFM uses optimal transport paths to connect noise and data:

- **More Direct**: Linear interpolation paths
- **More Stable**: Avoids complex noise scheduling
- **Faster**: Requires fewer integration steps

### FSS Compatibility

CFM's FSS calculation is fully compatible with diffusion:

- Compare network prediction vs ground truth target
- Use same normalization (`/2+0.5`)
- Measure single-step prediction accuracy

## 🐛 Troubleshooting

### Common Issues

1. **FSS becomes nan**

   - Check numerical stability
   - Ensure input data range is correct
   - Verify network output doesn't contain inf/nan

2. **Training instability**

   - Lower learning rate
   - Increase sigma_min value
   - Check data preprocessing

3. **Poor sampling quality**
   - Increase sampling steps
   - Check if model is sufficiently trained
   - Verify conditional inputs

### Debugging Tools

```bash
# Check model parameters
python -c "
from src.flow_matching import FlowMatchingModel, FlowMatchingConfig
print('CFM model loaded successfully')
"

# Verify data loading
python -c "
from src.dataset import SatelliteDataset
print('Dataset loading works')
"
```

## 📈 Experiment Recommendations

### Parameter Tuning

1. **Learning Rate**: Start with 0.0001, CFM can use higher learning rates
2. **Batch Size**: CFM is more stable with large batches, recommend 8 or larger
3. **Timesteps**: 500 steps usually sufficient, can try fewer
4. **Sampling Steps**: 100 steps is a good starting point

### Comparative Experiments

Recommend running both diffusion and CFM experiments for comparison:

- Use same dataset and preprocessing
- Compare training time and final performance
- Analyze FSS convergence curves

## 🔍 Key Differences from UNet Version

While both cfmU and cfmT use the same CFM core (`flow_matching.py`), they differ in:

### Architecture

- **cfmU**: Uses UNet with skip connections
- **cfmT**: Uses Transformer (DiT) with attention mechanisms

### Model Parameters

- **cfmU**: `embed_dim`, `dim_scales` for UNet layers
- **cfmT**: `dit_model` for Transformer size (DiT-S/4, DiT-XL/2, etc.)

### Computational Requirements

- **cfmU**: More memory efficient, faster training
- **cfmT**: Higher memory usage, potentially better quality

### Use Cases

- **cfmU**: Good for resource-constrained environments
- **cfmT**: Better for high-quality generation tasks

Both implementations maintain the same CFM advantages: faster training, more efficient sampling, and better stability compared to their respective diffusion counterparts.
