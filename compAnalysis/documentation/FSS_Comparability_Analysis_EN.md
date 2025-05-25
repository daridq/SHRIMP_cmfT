# FSS Comparability Analysis: CFM vs Diffusion

## 🚨 Key Issue Discovered

After examining the FSS calculation in the original diffusion model, we discovered an **important inconsistency** that would make FSS scores between CFM and Diffusion **incomparable**.

---

## 📊 Original Implementation Comparison

### **Diffusion Model FSS Calculation**:

```python
def fss(self, x, cond, lead_time, gf_sigmat=0):
    # 1. Randomly sample timestep
    t_sample = torch.randint(1, self.num_timesteps + 1, size=(bsz,), device=x.device)

    # 2. Add noise to real data
    eps = torch.randn_like(x)
    x_t = self.alpha_t[t_sample] * x + self.sigma_t[t_sample] * eps

    # 3. Network predicts target
    pred_target = self.nn_module(torch.cat((x_t, cond), dim=1), t_sample, lead_time)

    # 4. Determine ground truth target
    if self.target_type == "pred_x_0":
        gt_target = x          # Predict original data
    elif self.target_type == "pred_eps":
        gt_target = eps        # Predict noise
    elif self.target_type == "pred_v":
        gt_target = self.alpha_t[t_sample] * eps - self.sigma_t[t_sample] * x

    # 5. Compare network output vs ground truth target
    fss = avg_fss(pred_target/2+0.5, gt_target/2+0.5)
    return fss
```

### **Our Previous CFM FSS Calculation**:

```python
def fss(self, x, cond, lead_time, gf_sigmat=0):
    # 1. Complete sampling process
    samples = self.sample(cond=cond, bsz=bsz, num_sampling_timesteps=20)
    pred_x = samples[-1]  # Final generated result

    # 2. Compare final generated result vs real data
    fss = avg_fss(pred_x, x)
    return fss
```

---

## ⚠️ Why Are They Incomparable?

### **Completely Different Measurements**:

| Aspect                       | Diffusion FSS                                       | Previous CFM FSS                         |
| ---------------------------- | --------------------------------------------------- | ---------------------------------------- |
| **Comparison Object**        | Network direct output vs corresponding ground truth | Complete generation result vs real data  |
| **Measurement Content**      | Network single-step prediction accuracy             | Overall generation process final quality |
| **Computational Complexity** | Single forward pass                                 | Complete sampling process (20 steps)     |
| **Physical Meaning**         | Training target fitting degree                      | Generated sample quality                 |

### **Specific Problems**:

1. **Different Evaluation Dimensions**:

   - Diffusion: "Can the network correctly predict training targets?"
   - CFM (previous): "How is the final generated sample quality?"

2. **Different Numerical Ranges**:

   - Diffusion: Uses `/2+0.5` normalization
   - CFM (previous): Uses original values directly

3. **Different Randomness**:
   - Diffusion: Single random timestep
   - CFM (previous): Complete multi-step sampling process

---

## ✅ Fixed CFM FSS Calculation

Now our CFM FSS is **completely consistent** with Diffusion:

```python
def fss(self, x, cond, lead_time, gf_sigmat=0):
    # 1. Randomly sample time (consistent with diffusion)
    t = torch.rand(bsz, device=x.device)

    # 2. Construct training sample
    x0 = torch.randn_like(x)
    x_t, target_vf = self.sample_conditional_path(t, x0, x)

    # 3. Network predicts vector field
    pred_vf = self.nn_module(torch.cat((x_t, cond), dim=1), t_discrete, lead_time)

    # 4. Compare network output vs ground truth target (consistent with diffusion)
    pred_normalized = pred_vf / 2 + 0.5    # Same normalization
    target_normalized = target_vf / 2 + 0.5

    # 5. Calculate FSS
    fss = avg_fss(pred_normalized, target_normalized)
    return fss
```

---

## 🎯 Current Comparability

### **Same Evaluation Logic**:

- ✅ Both compare **network direct output** vs **corresponding ground truth target**
- ✅ Both use **same normalization**: `/2+0.5`
- ✅ Both measure **single-step prediction accuracy**
- ✅ Both use **same FSS calculation function**

### **Different but Reasonable Parts**:

- **Training Targets**:
  - Diffusion: Predict noise/data/v-parameter
  - CFM: Predict vector field
- **Time Sampling**:
  - Diffusion: Discrete timesteps `[1, num_timesteps]`
  - CFM: Continuous time `[0, 1]`

---

## 📈 Expected Effects

After the fix, you should see:

1. **Stable FSS Values**: No more `nan` occurrences
2. **Comparable FSS**: Can directly compare with diffusion model FSS
3. **Meaningful FSS**: Reflects network performance on training targets
4. **Training Monitoring**: Can observe FSS improvement trends during training

---

## 🔬 Experimental Recommendations

### **Comparative Experiments**:

1. Train Diffusion and CFM with same data
2. Compare FSS trends of both
3. Verify if CFM indeed has better convergence

### **Monitoring Metrics**:

- **Loss**: Training target fitting degree
- **FSS**: Spatial accuracy of network predictions
- **Final Sampling Quality**: Evaluate generation quality with complete sampling

This way we have **comparable FSS metrics** that can fairly evaluate CFM's advantages over Diffusion!

## 🔍 Technical Deep Dive

### Why This Comparison is Fair

**Semantic Equivalence**:

- Both measure the network's ability to predict its respective training targets
- Both use the same spatial accuracy metric (FSS)
- Both provide meaningful training feedback

**Methodological Consistency**:

- Same evaluation frequency (per batch/epoch)
- Same numerical preprocessing
- Same statistical aggregation

**Practical Value**:

- Enables direct performance comparison
- Provides consistent training monitoring
- Supports hyperparameter optimization

### Mathematical Foundation

**Diffusion Target Prediction**:

```
Network learns: f_θ(x_t, t) → target_type
FSS measures: spatial_accuracy(f_θ(x_t, t), ground_truth)
```

**CFM Target Prediction**:

```
Network learns: f_θ(x_t, t) → vector_field
FSS measures: spatial_accuracy(f_θ(x_t, t), ground_truth_vf)
```

Both measure the same fundamental capability: how well the network predicts its training target.

## 📊 Validation Protocol

### Sanity Checks

1. **Range Validation**: FSS values should be in [0, 1]
2. **Trend Validation**: FSS should improve during training
3. **Comparison Validation**: CFM and Diffusion FSS should be in similar ranges

### Cross-Validation

1. **Same Data**: Use identical datasets for both models
2. **Same Architecture**: Use equivalent network architectures
3. **Same Evaluation**: Apply identical FSS calculation procedures

This ensures that any performance differences reflect the fundamental algorithmic advantages of CFM over Diffusion, rather than evaluation inconsistencies.
