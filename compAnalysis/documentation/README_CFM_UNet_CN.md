# Conditional Flow Matching (CFM) Implementation

This directory contains a modified version of the original diffusion+UNet model, now using **Conditional Flow Matching** (CFM) instead of traditional diffusion models.

## What is Conditional Flow Matching?

Conditional Flow Matching is a simulation-free approach for training Continuous Normalizing Flows (CNFs) based on the paper ["Flow Matching for Generative Modeling" (Lipman et al. 2023)](https://arxiv.org/abs/2210.02747).

### Key Advantages over Diffusion Models:

1. **Straighter Paths**: Uses Optimal Transport (OT) paths that are more direct from noise to data
2. **Faster Training**: More stable and efficient training process
3. **Faster Sampling**: Requires fewer function evaluations (NFE) during inference
4. **Better Performance**: Often achieves better sample quality and likelihood

## Changes Made

### 1. New Flow Matching Module (`src/flow_matching.py`)

- **`FlowMatchingModel`**: Replaces `DiffusionModel` with equivalent interface
- **`FlowMatchingConfig`**: Configuration for flow matching parameters
- **Two Path Types**:
  - `"optimal_transport"`: Linear interpolation paths (recommended)
  - `"diffusion_like"`: Variance-preserving paths similar to diffusion

### 2. Updated Training Script (`shrimp.py`)

- Modified to use `FlowMatchingModel` instead of `DiffusionModel`
- Added new parameters:
  - `--path-type`: Choose between "optimal_transport" or "diffusion_like"
  - `--sigma-min`: Minimum noise level (default: 0.001)
- Maintains backward compatibility with existing parameters

### 3. Updated Command Generation (`CfmU_cmd_generate.py`)

- Optimized parameters for CFM efficiency
- Reduced epochs (400→300), timesteps (1000→500), sampling steps (500→100)
- Increased batch size (4→8) for better CFM stability
- Changed job prefix from "DiffExp" to "CfmU"

## Usage

### Basic Training Command

```bash
python3 shrimp.py \
    --epochs 400 \
    --batch-size 4 \
    --timesteps 1000 \
    --path-type optimal_transport \
    --sigma-min 0.001 \
    --loss-type l2 \
    --learning-rate 0.0001 \
    --sat-files-path "/path/to/satellite/data" \
    --rainfall-files-path "/path/to/radar/data" \
    --train-model
```

### Generate GADI Job Scripts

```bash
python3 CfmU_cmd_generate.py
```

This will generate PBS scripts for running experiments on GADI.

### Test the Implementation

```bash
python3 test_cfm.py
```

## Key Parameters

| Parameter     | Description              | Default             | Options                               |
| ------------- | ------------------------ | ------------------- | ------------------------------------- |
| `--path-type` | Type of probability path | `optimal_transport` | `optimal_transport`, `diffusion_like` |
| `--sigma-min` | Minimum noise level      | `0.001`             | Float > 0                             |
| `--timesteps` | Number of time steps     | `1000`              | Integer > 0                           |
| `--loss-type` | Loss function            | `l2`                | `l1`, `l2`, `Hilburn_Loss`            |

## Technical Details

### Optimal Transport Path

For the optimal transport path, the conditional probability path is:

- `x_t = (1 - (1 - σ_min) * t) * x_0 + t * x_1`
- Target vector field: `u_t = x_1 - (1 - σ_min) * x_0`

Where:

- `x_0`: Noise sample
- `x_1`: Data sample
- `t`: Time ∈ [0, 1]
- `σ_min`: Minimum noise level

### Diffusion-like Path

For backward compatibility, a variance-preserving path similar to diffusion:

- `x_t = cos(πt/2) * x_1 + sin(πt/2) * x_0`
- Target vector field: `u_t = -π/2 * (sin(πt/2) * x_1 - cos(πt/2) * x_0)`

## Expected Benefits

1. **Training Speed**: ~2-3x faster convergence compared to diffusion
2. **Sampling Speed**: ~2-5x fewer function evaluations needed
3. **Sample Quality**: Better FID scores and perceptual quality
4. **Stability**: More robust training with fewer hyperparameter tuning

## Compatibility

The implementation maintains full backward compatibility with the original interface:

- Same UNet architecture
- Same data loading and preprocessing
- Same evaluation metrics (FSS)
- Same model saving/loading format

## References

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [TorchCFM Library](https://github.com/atong01/conditional-flow-matching)
- [Diffusion Meets Flow Matching](https://diffusionflow.github.io/)
