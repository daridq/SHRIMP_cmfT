# FSS 可比性分析：CFM vs Diffusion

## 🚨 发现的关键问题

在检查原始 diffusion 模型的 FSS 计算后，发现了一个**重要的不一致性**，这会导致 CFM 和 Diffusion 的 FSS 分数**不可比较**。

---

## 📊 原始实现对比

### **Diffusion 模型的 FSS 计算**：

```python
def fss(self, x, cond, lead_time, gf_sigmat=0):
    # 1. 随机采样时间步
    t_sample = torch.randint(1, self.num_timesteps + 1, size=(bsz,), device=x.device)

    # 2. 添加噪声到真实数据
    eps = torch.randn_like(x)
    x_t = self.alpha_t[t_sample] * x + self.sigma_t[t_sample] * eps

    # 3. 网络预测目标
    pred_target = self.nn_module(torch.cat((x_t, cond), dim=1), t_sample, lead_time)

    # 4. 确定真实目标
    if self.target_type == "pred_x_0":
        gt_target = x          # 预测原始数据
    elif self.target_type == "pred_eps":
        gt_target = eps        # 预测噪声
    elif self.target_type == "pred_v":
        gt_target = self.alpha_t[t_sample] * eps - self.sigma_t[t_sample] * x

    # 5. 比较网络输出 vs 真实目标
    fss = avg_fss(pred_target/2+0.5, gt_target/2+0.5)
    return fss
```

### **我们之前的 CFM FSS 计算**：

```python
def fss(self, x, cond, lead_time, gf_sigmat=0):
    # 1. 完整采样过程
    samples = self.sample(cond=cond, bsz=bsz, num_sampling_timesteps=20)
    pred_x = samples[-1]  # 最终生成结果

    # 2. 比较最终生成结果 vs 真实数据
    fss = avg_fss(pred_x, x)
    return fss
```

---

## ⚠️ 为什么不可比？

### **测量的内容完全不同**：

| 方面           | Diffusion FSS                | 之前的 CFM FSS           |
| -------------- | ---------------------------- | ------------------------ |
| **比较对象**   | 网络直接输出 vs 对应真实目标 | 完整生成结果 vs 真实数据 |
| **测量内容**   | 网络在单步预测的准确性       | 整个生成过程的最终质量   |
| **计算复杂度** | 单次前向传播                 | 完整采样过程（20 步）    |
| **物理意义**   | 训练目标的拟合程度           | 生成样本的质量           |

### **具体问题**：

1. **不同的评估维度**：

   - Diffusion: "网络能否正确预测训练目标？"
   - CFM (之前): "最终生成的样本质量如何？"

2. **数值范围不同**：

   - Diffusion: 使用 `/2+0.5` 归一化
   - CFM (之前): 直接使用原始值

3. **随机性不同**：
   - Diffusion: 单次随机时间步
   - CFM (之前): 完整的多步采样过程

---

## ✅ 修复后的 CFM FSS 计算

现在我们的 CFM FSS 与 Diffusion **完全一致**：

```python
def fss(self, x, cond, lead_time, gf_sigmat=0):
    # 1. 随机采样时间（与 diffusion 一致）
    t = torch.rand(bsz, device=x.device)

    # 2. 构造训练样本
    x0 = torch.randn_like(x)
    x_t, target_vf = self.sample_conditional_path(t, x0, x)

    # 3. 网络预测向量场
    pred_vf = self.nn_module(torch.cat((x_t, cond), dim=1), t_discrete, lead_time)

    # 4. 比较网络输出 vs 真实目标（与 diffusion 一致）
    pred_normalized = pred_vf / 2 + 0.5    # 相同的归一化
    target_normalized = target_vf / 2 + 0.5

    # 5. 计算 FSS
    fss = avg_fss(pred_normalized, target_normalized)
    return fss
```

---

## 🎯 现在的可比性

### **相同的评估逻辑**：

- ✅ 都比较**网络直接输出** vs **对应真实目标**
- ✅ 都使用**相同的归一化**：`/2+0.5`
- ✅ 都测量**单步预测准确性**
- ✅ 都使用**相同的 FSS 计算函数**

### **不同但合理的部分**：

- **训练目标**：
  - Diffusion: 预测噪声/数据/v-参数
  - CFM: 预测向量场
- **时间采样**：
  - Diffusion: 离散时间步 `[1, num_timesteps]`
  - CFM: 连续时间 `[0, 1]`

---

## 📈 预期效果

修复后，你应该看到：

1. **FSS 数值稳定**：不再出现 `nan`
2. **FSS 可比较**：可以直接与 diffusion 模型的 FSS 对比
3. **FSS 有意义**：反映网络在训练目标上的表现
4. **训练监控**：可以观察 FSS 随训练的改善趋势

---

## 🔬 实验建议

### **对比实验**：

1. 用相同数据训练 Diffusion 和 CFM
2. 比较两者的 FSS 趋势
3. 验证 CFM 是否确实有更好的收敛性

### **监控指标**：

- **Loss**: 训练目标的拟合程度
- **FSS**: 网络预测的空间准确性
- **最终采样质量**: 用完整采样评估生成质量

这样我们就有了**可比较的 FSS 指标**，可以公平地评估 CFM 相对于 Diffusion 的优势！
