# SHRIMP_cmfT

Diffusion vs Flow Matching models for precipitation prediction.

## Models

- **DiU**: U-Net + DDPM (32.17M)
- **DiT**: Transformer + DDPM (32.50M)
- **CFM-U**: U-Net + Flow Matching (32.17M)
- **CFM-T**: Transformer + Flow Matching (32.50M)

## Structure

```
├── models/           # Model implementations
├── exp_results/      # Experimental results
│   ├── benchmark/    # 4-model comparison
│   └── exp_cfmU/     # CFM-U tuning
└── cfm_unet/         # [`tensorflow version`, DEPRECATED]
```

## Results

| Model | Interpretation FSS | Prediction FSS |
| ----- | ------------------ | -------------- |
| DiU   | **0.5125**         | 0.5667         |
| DiT   | 0.4419             | **0.6058**     |
| CFM-U | 0.4583             | 0.4359         |
| CFM-T | 0.4740             | 0.4813         |

## Usage

To run experiment from GADI

```bash
python models/DiffExp_cmd_generate.py    # Generate and automate PBS sciprts
```
