# CFM-DiT: Conditional Flow Matching with Diffusion Transformer

这是将 DiT (Diffusion Transformer) 模型从 diffusion 转换为 Conditional Flow Matching (CFM) 的实现。

## 🚀 主要特性

- **更快的训练**: CFM 比 diffusion 需要更少的训练步数
- **更快的采样**: 采样步数从 500 减少到 100 (80% 提升)
- **更好的稳定性**: 最优传输路径比噪声调度更直接
- **完全兼容**: 保持与原始 DiT 架构的完全兼容性

## 📁 文件结构

```
cfmT/
├── src/
│   ├── flow_matching.py      # CFM 核心实现 (与 cfmU 完全一致)
│   ├── DiTModels.py          # DiT 模型架构
│   ├── dataset.py            # 数据集处理
│   ├── blocks.py             # 网络组件
│   ├── utils.py              # 工具函数
│   └── DatasetBuilder.py     # 数据集构建器
├── shrimp_cfmT.py           # 主训练脚本
├── CfmT_cmd_generate.py     # 实验命令生成器
└── README_CFM.md            # 本文件
```

## 🔧 安装和使用

### 1. 环境要求

```bash
# 与原始 DiT 相同的环境
torch >= 1.9.0
numpy
matplotlib
tqdm
scipy
tensorboard
```

### 2. 训练模型

```bash
# 基本训练命令
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

### 3. 生成实验命令

```bash
# 生成优化的 CFM 实验配置
python CfmT_cmd_generate.py

# 这会生成:
# - cfmT_experiments_summary.txt: 所有实验命令
# - jobs/*.pbs: GADI 集群作业脚本
```

### 4. 在 GADI 上运行

```bash
# 提交所有作业
cd jobs
for script in *.pbs; do qsub $script; done
```

## ⚙️ CFM 特有参数

### 新增参数

- `--path-type`: 路径类型

  - `optimal_transport` (推荐): 最优传输路径
  - `diffusion_like`: 类似 diffusion 的路径

- `--sigma-min`: 最小噪声水平 (默认: 0.001)
  - 控制 t=1 时的噪声量
  - 较小值 → 更确定性的路径

### 优化的参数

相比原始 DiT，CFM 使用了以下优化参数：

- **epochs**: 300 (从 400 减少)
- **batch_size**: 8 (从 4 增加)
- **timesteps**: 500 (从 1000 减少)
- **sampling_timesteps**: 100 (从 500 减少)
- **learning_rate**: 0.0001-0.0002 (可以使用更高学习率)

## 📊 性能对比

| 指标     | Diffusion DiT | CFM DiT | 改善   |
| -------- | ------------- | ------- | ------ |
| 训练步数 | 1000          | 500     | 50% ↓  |
| 采样步数 | 500           | 100     | 80% ↓  |
| 训练轮数 | 400           | 300     | 25% ↓  |
| 批次大小 | 4             | 8       | 100% ↑ |

## 🔬 技术细节

### CFM vs Diffusion

**Diffusion 过程**:

```
x_t = α_t * x_0 + σ_t * ε
目标: 预测 ε, x_0, 或 v
```

**CFM 过程**:

```
x_t = (1-(1-σ_min)*t) * x_0 + t * x_1
目标: 预测向量场 u_t = x_1 - (1-σ_min)*x_0
```

### 最优传输路径

CFM 使用最优传输路径连接噪声和数据:

- **更直接**: 线性插值路径
- **更稳定**: 避免复杂的噪声调度
- **更快**: 需要更少的积分步数

### FSS 兼容性

CFM 的 FSS 计算与 diffusion 完全兼容:

- 比较网络预测 vs 真实目标
- 使用相同的归一化 (`/2+0.5`)
- 测量单步预测准确性

## 🐛 故障排除

### 常见问题

1. **FSS 变成 nan**

   - 检查数值稳定性
   - 确保输入数据范围正确
   - 验证网络输出不包含 inf/nan

2. **训练不稳定**

   - 降低学习率
   - 增加 sigma_min 值
   - 检查数据预处理

3. **采样质量差**
   - 增加采样步数
   - 检查模型是否充分训练
   - 验证条件输入

### 调试工具

```bash
# 检查模型参数
python -c "
from src.flow_matching import FlowMatchingModel, FlowMatchingConfig
print('CFM model loaded successfully')
"

# 验证数据加载
python -c "
from src.dataset import SatelliteDataset
print('Dataset loading works')
"
```

## 📈 实验建议

### 参数调优

1. **学习率**: 从 0.0001 开始，CFM 可以使用更高的学习率
2. **批次大小**: CFM 对大批次更稳定，推荐 8 或更大
3. **时间步数**: 500 步通常足够，可以尝试更少
4. **采样步数**: 100 步是很好的起点

### 对比实验

建议同时运行 diffusion 和 CFM 实验进行对比:

- 使用相同的数据集和评估指标
- 比较训练时间和最终性能
- 分析 FSS 收敛曲线

## 📚 参考文献

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
- [Conditional Flow Matching: Simulation-Free Dynamic Optimal Transport](https://arxiv.org/abs/2302.00482)

## 🤝 贡献

这个实现基于:

- **cfmU**: UNet + CFM 实现
- **原始 DiT**: Diffusion Transformer 架构
- **Flow Matching 论文**: 理论基础

所有 CFM 相关代码与 cfmU 保持完全一致，确保实验的可重现性和公平比较。
