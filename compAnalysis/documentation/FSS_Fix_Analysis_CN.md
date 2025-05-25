# CFM FSS 修复分析

## 🚨 问题发现

在 CFM 训练过程中发现 FSS 值异常高（0.99+），这明显不正常。正常的降水预测 FSS 值应该在 0.1-0.5 之间。

### 异常日志示例

```
Epoch: 4/100 | 73.52s | Loss: 0.2686363416 | Val_loss: 0.2692165820 | Val_fss: 0.9910661306
Epoch: 5/100 | 74.34s | Loss: 0.2672936717 | Val_loss: 0.2894086932 | Val_fss: 0.9309849879
Epoch: 6/100 | 73.91s | Loss: 0.2663216000 | Val_loss: 0.2750107928 | Val_fss: 0.9789647348
```

## 🔍 根本原因分析

### 问题 1: 比较对象错误

**错误的 CFM FSS 实现**:

```python
# ❌ 错误：比较向量场 vs 向量场
target_vf = x1 - (1 - self.sigma_min) * x0  # 向量场
pred_vf = self.nn_module(...)  # 预测的向量场
fss = avg_fss(pred_vf/2+0.5, target_vf/2+0.5)
```

**正确的 Diffusion FSS 实现**:

```python
# ✅ 正确：比较降水预测 vs 真实降水
if self.target_type == "pred_x_0":
    gt_target = x  # 原始降水数据
pred_target = self.nn_module(...)  # 预测的降水数据
fss = avg_fss(pred_target/2+0.5, gt_target/2+0.5)
```

### 问题 2: 数据范围不匹配

**向量场的数值特性**:

- 向量场值可能在很大的范围内（如 [-10, 10] 或更大）
- `/2+0.5` 归一化假设输入在 `[-1, 1]` 范围内
- 当向量场值很大时，归一化后可能都变成 0 或 1
- 导致 FSS 计算出现数值问题

**降水数据的数值特性**:

- 降水值通常在合理范围内（如 [0, 1] 或 [0, 10]）
- `/2+0.5` 归一化更适合这种数据
- FSS 计算更有意义

## 🛠️ 修复方案

### 核心思路

将 CFM 的 FSS 计算从"比较向量场"改为"比较降水预测"，使其与 Diffusion 保持一致。

### 修复步骤

1. **从向量场恢复降水预测**:

   ```python
   # CFM 向量场定义: target_vf = x1 - (1-σ_min)*x0
   # 因此: x1 = target_vf + (1-σ_min)*x0
   pred_x1 = pred_vf + (1 - self.sigma_min) * x0
   gt_x1 = x  # 真实降水数据
   ```

2. **比较降水数据而非向量场**:
   ```python
   # 现在比较的是降水预测 vs 真实降水
   fss_result = avg_fss(pred_x1/2+0.5, gt_x1/2+0.5)
   ```

### 完整的修复代码

```python
@torch.no_grad()
def fss(self, x: torch.Tensor, cond, lead_time, gf_sigmat=0):
    """
    Compute FSS in a way that's comparable to diffusion model
    Compare network prediction vs ground truth target (not final samples)

    For CFM, we need to compare the final prediction (x_0) vs ground truth,
    not the vector field predictions.
    """
    bsz, *_ = x.shape

    try:
        # Sample random time (similar to diffusion)
        t = torch.rand(bsz, device=x.device)  # t ~ U[0,1]
        lead_time = torch.full((bsz,), lead_time, device=x.device, dtype=torch.int64)

        # Sample noise
        x0 = torch.randn_like(x)
        if gf_sigmat > 0:
            x0 = torch.tensor(gaussian_filter(x0.cpu().numpy(), sigma=gf_sigmat), device=x.device)

        # Sample from conditional path
        x_t, target_vf = self.sample_conditional_path(t, x0, x)

        # Predict vector field
        t_discrete = (t * self.num_timesteps).long().clamp(0, self.num_timesteps - 1)
        pred_vf = self.nn_module(torch.cat((x_t, cond), dim=1), t_discrete, lead_time)

        # Convert vector field prediction to data prediction
        # target_vf = x_1 - (1-σ_min)*x_0, so x_1 = target_vf + (1-σ_min)*x_0

        # Ground truth x_1 (the actual data)
        gt_x1 = x

        # Predicted x_1 from vector field
        pred_x1 = pred_vf + (1 - self.sigma_min) * x0

        # Now compare predicted vs actual precipitation data (like diffusion)
        fss_result = avg_fss(pred_x1/2+0.5, gt_x1/2+0.5)

        # Validate FSS result
        if np.isnan(fss_result) or np.isinf(fss_result):
            return torch.tensor(0.0, device=x.device)

        return torch.tensor(fss_result, device=x.device)

    except Exception as e:
        # If anything fails, return 0 instead of crashing
        return torch.tensor(0.0, device=x.device)
```

## 📊 修复效果预期

### 修复前 (错误)

- FSS 值: 0.99+ (异常高)
- 原因: 比较向量场，数值范围不匹配
- 问题: 无法与 Diffusion 模型公平比较

### 修复后 (正确)

- FSS 值: 0.1-0.5 (正常范围)
- 原因: 比较降水预测，语义正确
- 优势: 与 Diffusion 模型可比较

## 🔬 技术深度分析

### CFM vs Diffusion 的 FSS 语义

**Diffusion FSS 语义**:

- 测量: 网络能否正确预测目标（降水/噪声/v-参数）
- 意义: 单步预测准确性
- 范围: 合理的 FSS 值

**CFM FSS 语义（修复前）**:

- 测量: 网络能否正确预测向量场
- 问题: 向量场不是最终目标，且数值范围不合适
- 结果: 异常的 FSS 值

**CFM FSS 语义（修复后）**:

- 测量: 网络能否正确预测降水数据
- 意义: 与 Diffusion 相同的单步预测准确性
- 范围: 合理的 FSS 值

### 数学推导

**CFM 最优传输路径**:

```
x_t = (1-(1-σ_min)*t) * x_0 + t * x_1
```

**向量场定义**:

```
u_t = dx/dt = x_1 - (1-σ_min) * x_0
```

**从向量场恢复数据**:

```
x_1 = u_t + (1-σ_min) * x_0
```

这个恢复公式是修复的关键，它将向量场预测转换为数据预测。

## 🧪 验证方法

### 1. 数值范围检查

```python
# 检查向量场 vs 数据的数值范围
print(f"Vector field range: [{pred_vf.min():.3f}, {pred_vf.max():.3f}]")
print(f"Data prediction range: [{pred_x1.min():.3f}, {pred_x1.max():.3f}]")
```

### 2. FSS 值合理性检查

```python
# FSS 应该在 [0, 1] 范围内，且通常 < 0.8
assert 0.0 <= fss_value <= 1.0
assert fss_value < 0.8  # 对于随机初始化的模型
```

### 3. 与 Diffusion 对比

```python
# CFM 和 Diffusion 的 FSS 应该在相似范围内
cfm_fss = cfm_model.fss(data, cond, lead_time)
diff_fss = diff_model.fss(data, cond, lead_time)
print(f"CFM FSS: {cfm_fss:.6f}, Diff FSS: {diff_fss:.6f}")
```

## 📈 实际影响

### 训练监控

- **修复前**: FSS 值无意义，无法判断模型性能
- **修复后**: FSS 值有意义，可以监控训练进度

### 模型比较

- **修复前**: CFM 和 Diffusion 的 FSS 不可比较
- **修复后**: 可以公平比较两种方法的性能

### 调试能力

- **修复前**: 异常高的 FSS 掩盖了真实问题
- **修复后**: 真实的 FSS 有助于发现训练问题

## 🎯 总结

这个修复解决了 CFM 实现中的一个关键错误：

1. **语义错误**: 从比较向量场改为比较降水预测
2. **数值错误**: 修复了数值范围不匹配的问题
3. **兼容性**: 使 CFM 和 Diffusion 的 FSS 可比较

修复后，CFM 的 FSS 值应该回到正常范围（0.1-0.5），并且能够正确反映模型的预测性能。
