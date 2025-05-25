# Flow Matching 参数优化指南

## 从 Diffusion 到 Flow Matching 的参数调整原理

### 🚀 核心优化原理

Flow Matching 相比 Diffusion 有以下关键优势，这些优势指导了我们的参数优化：

1. **更直接的路径**: OT 路径比 diffusion 的噪声调度更直接
2. **更快的收敛**: 训练过程更稳定，需要更少的 epoch
3. **更高效的采样**: 需要更少的函数评估次数
4. **更好的批次稳定性**: 对大 batch size 更友好

---

## 📊 参数对比分析

### 1. **训练效率参数**

| 参数            | Diffusion | Flow Matching    | 变化原因                            |
| --------------- | --------- | ---------------- | ----------------------------------- |
| `epochs`        | 400       | 300              | CFM 收敛更快，减少 25%              |
| `batch_size`    | 4         | 8                | CFM 对大 batch 更稳定，提升训练效率 |
| `learning_rate` | 0.0001    | [0.0001, 0.0002] | 测试稍高学习率，CFM 训练更稳定      |

**原理**: Flow Matching 的损失函数更直接（直接预测向量场），避免了 diffusion 中复杂的噪声调度和重参数化，因此训练更稳定快速。

### 2. **时间步数优化**

| 参数                 | Diffusion | Flow Matching | 变化原因                     |
| -------------------- | --------- | ------------- | ---------------------------- |
| `timesteps`          | 1000      | 500           | CFM 训练时间步可以更少       |
| `sampling_timesteps` | 500       | 100           | CFM 采样效率高，大幅减少 80% |

**原理**:

- **训练时**: CFM 不需要复杂的噪声调度，500 步足够覆盖从噪声到数据的路径
- **采样时**: ODE 积分比 DDIM/DDPM 更高效，100 步就能获得高质量样本

### 3. **Flow Matching 特有参数**

| 参数        | 取值                                    | 说明                   |
| ----------- | --------------------------------------- | ---------------------- |
| `path_type` | ["optimal_transport", "diffusion_like"] | 测试两种路径类型       |
| `sigma_min` | [0.001, 0.01]                           | 测试不同的最小噪声水平 |

**原理**:

- **Optimal Transport**: 提供最直接的路径，通常性能更好
- **Diffusion-like**: 保持与原 diffusion 相似的行为，便于对比
- **sigma_min**: 控制路径端点的噪声水平，影响生成质量

### 4. **模型容量调整**

| 参数          | Diffusion | Flow Matching | 变化原因                                 |
| ------------- | --------- | ------------- | ---------------------------------------- |
| `embed_dim`   | 64        | [64, 128]     | 测试更大容量，CFM 可能受益于更强表达能力 |
| `num_workers` | 2         | 4             | 提升数据加载效率                         |

**原理**: CFM 的向量场预测可能比 diffusion 的噪声预测更复杂，更大的模型容量可能带来更好的性能。

---

## 🎯 实验设计策略

### 当前配置将产生的实验组合：

```
总实验数 = 2(path_type) × 2(sigma_min) × 2(embed_dim) × 2(learning_rate) × 2(history_frames)
         = 32 个实验组合
```

### 推荐的实验优先级：

#### **第一优先级** (最有希望的配置):

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

#### **第二优先级** (对比基线):

```python
{
    "path_type": "diffusion_like",
    "sigma_min": 0.001,
    "embed_dim": 64,
    "learning_rate": 0.0001,
    # 其他参数同上
}
```

---

## 📈 预期性能提升

基于论文和理论分析，预期改进：

| 指标         | 预期提升    | 原因                    |
| ------------ | ----------- | ----------------------- |
| **训练时间** | 30-50% 减少 | 更少 epoch + 更大 batch |
| **采样速度** | 80% 减少    | 100 vs 500 采样步数     |
| **内存效率** | 10-20% 提升 | 更大 batch size 可行    |
| **样本质量** | 5-15% 提升  | 更直接的生成路径        |

---

## ⚠️ 注意事项

### 1. **资源需求变化**

- **内存**: batch_size 翻倍，需要更多 GPU 内存
- **计算**: embed_dim 增大，计算量增加
- **时间**: 总体训练时间应该减少

### 2. **监控指标**

- **收敛速度**: 观察 validation loss 下降曲线
- **采样质量**: 比较生成样本的 FSS 分数
- **训练稳定性**: 观察 loss 的波动程度

### 3. **调试建议**

如果遇到问题：

1. 先用小参数测试 (embed_dim=64, batch_size=4)
2. 逐步增加参数规模
3. 监控 GPU 内存使用情况

---

## 🔧 快速测试命令

测试最优配置：

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

这个配置应该能在更短时间内达到更好的性能！
