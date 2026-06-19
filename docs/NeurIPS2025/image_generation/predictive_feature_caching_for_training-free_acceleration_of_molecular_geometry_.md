---
title: >-
  [论文解读] Predictive Feature Caching for Training-free Acceleration of Molecular Geometry Generation
description: >-
  [NeurIPS 2025][图像生成][分子几何生成] 将图像领域的预测式特征缓存（predictive feature caching）策略迁移到分子几何生成领域，利用采样轨迹中隐藏状态的时间平滑性，实现免训练的2-3倍推理加速，且与其他优化手段组合可达7倍加速。 流匹配模型（flow matching）是当前分子几何生…
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "分子几何生成"
  - "流匹配"
  - "特征缓存"
  - "免训练加速"
  - "SE(3)等变"
---

# Predictive Feature Caching for Training-free Acceleration of Molecular Geometry Generation

**会议**: NeurIPS 2025  
**arXiv**: [2510.04646](https://arxiv.org/abs/2510.04646)  
**代码**: 无  
**领域**: 分子生成 / 推理加速  
**关键词**: 分子几何生成, 流匹配, 特征缓存, 免训练加速, SE(3)等变

## 一句话总结

将图像领域的预测式特征缓存（predictive feature caching）策略迁移到分子几何生成领域，利用采样轨迹中隐藏状态的时间平滑性，实现免训练的2-3倍推理加速，且与其他优化手段组合可达7倍加速。

## 研究背景与动机

流匹配模型（flow matching）是当前分子几何生成的SOTA方法，但推理时需要数百次神经网络前向传播，计算代价高昂。在药物发现管线中，需生成50万甚至超过100万个分子候选，生成器的推理时间成为主要瓶颈。

现有的加速方法（轨迹重参数化、渐进蒸馏、潜空间方法）都需要额外训练，增加数据和算力开销。本文追求互补方向——**免训练**的加速方案，灵感来自图像生成领域的特征缓存技术：

- 在ODE求解过程中，相邻时间步的中间激活值变化平滑
- 可以缓存并预测这些中间特征，避免完整前向传播
- 此方向在分子领域完全未被探索

## 方法详解

### 整体框架

分子几何生成基于条件流匹配（CFM），通过学习时间相关的向量场 $v_\theta(x_t, t)$，将噪声分布传输到数据分布。分子参数化为 $x = (c, a, b)$，分别代表坐标、原子类型、键级。采样时用欧拉离散化：

$$x_{k+1} = x_k + \Delta t_k \, v_\theta(x_k, t_k)$$

网络骨干由L个块组成：$g_L \circ \cdots \circ g_1$。由于ODE右端连续且网络对$(x,t)$连续，中间激活值随时间平滑变化。本文利用此平滑性，仅对最后一个块L应用预测缓存。

### 关键设计

**TaylorSeer缓存**：在预设的检查点时间步（每D步）执行完整前向传播并缓存：

$$C(x_t) = \{F(x_t), \Delta F(x_t), \dots, \Delta^m F(x_t)\}$$

对窗口内的中间时间步，使用m阶Taylor预测器进行特征预测：

$$F_{\text{pred},m}(x_{t+k}) = F(x_t) + \sum_{i=1}^{m} \frac{\Delta^i F(x_t)}{i! D^i} (-k)^i$$

当 $m=0$ 退化为朴素缓存（直接复用），$m=1$ 为线性预测, $m=2$ 为二次预测。

**Adams-Bashforth (AB) 缓存**：利用j步AB线性多步递推预测：

$$F_{\text{AB}(j)}(x_{t+k}) = \sum_{i=1}^{j} (-1)^{i+1} \binom{j}{i} F(x_{t+k+i})$$

使用最近j个缓存输出预测当前输出。

**等变性保持**：缓存操作是时间标量的线性组合和有限差分，与群作用$G = E(3) \times S_N$可交换。如果基密度$G$-不变且向量场$G$-等变，则预测的评估保持等变性，终端密度保持不变性。

### 损失函数 / 训练策略

本方法**完全免训练**，直接作用于预训练的SemlaFlow模型。无需修改模型权重，仅在推理时引入缓存逻辑。实现上保证无论缓存间隔D如何，最后一步总是完整计算，确保最终输出质量。

## 实验关键数据

### 主实验 — GEOM-Drugs数据集

| 步数 | 方法 | 分子稳定性↑ | 有效性(PRC)↑ | 能量/原子↓ | 应变/原子↓ | 吞吐量↑ |
|------|------|-----------|-------------|-----------|-----------|--------|
| 100 | Base | 0.98 | 0.88 | 2.38 | 1.50 | 11.4 |
| 51 | Base | 0.98 | 0.86 | 2.51 | 1.63 | 21.9 |
| 51 | Taylor m=2 | 0.98 | **0.86** | **2.25** | **1.46** | 22.1 |
| 51 | AB j=3 | 0.98 | **0.87** | **2.15** | **1.40** | 22.1 |
| 34 | Base | 0.97 | 0.85 | 2.62 | 1.78 | 32.2 |
| 34 | Taylor m=2 | 0.97 | 0.83 | **2.25** | **1.53** | 32.4 |
| 34 | AB j=3 | 0.97 | **0.85** | **2.25** | **1.51** | 32.1 |
| 26 | Base | 0.97 | 0.82 | 2.69 | 1.85 | 41.0 |
| 26 | AB j=3 | 0.96 | **0.82** | **2.30** | **1.60** | 41.2 |

### 组合加速实验 — 与正交优化叠加

| 加速方法组合 | 推理时间(生成1万分子) | 加速比 |
|-------------|---------------------|--------|
| Base (100步) | ~14 min | 1× |
| 仅缓存 (AB, D=2) | ~4.7 min | ~3× |
| 缓存 + torch.compile | ~3 min | ~4.5× |
| 缓存 + compile + TF32 | **~2 min** | **~7×** |

### 关键发现

1. **质量等价的2倍加速**：在51步+缓存(D=2)配置下，所有质量指标与100步基线匹配甚至超越，吞吐量翻倍
2. **缓存优于朴素步数削减**：直接减步到51步会显著降低质量，但缓存在同等步数下保持甚至改善质量（能量和应变下降）
3. **AB缓存优于Taylor缓存**：在所有配置下AB j=3一致优于Taylor m=2
4. **与编译优化正交叠加**：缓存的加速与torch.compile和TF32核互不冲突，组合后达7倍加速
5. **等变性完整保持**：理论证明和实验验证缓存不破坏SE(3)等变性

## 亮点与洞察

- **跨领域迁移的成功案例**：将图像/视频扩散模型的缓存加速技术成功适配到分子SE(3)等变架构
- **免训练的即插即用**：无需微调或重训练，直接提升已有预训练模型的推理效率
- **反直觉的质量提升**：缓存不仅不降低质量，在能量和应变指标上反而有所改善，可能是因为缓存的平滑效果类似某种正则化
- **实际影响力大**：将生成1万个分子的时间从14分钟降至2分钟，对药物发现的大规模采样有直接意义

## 局限与展望

- 主要在SemlaFlow一个模型上验证，扩展到其他分子生成模型（如GeoDiff、MDM）待确认
- 缓存间隔D和预测器阶数的超参数选择缺乏自适应机制（类似TeaCache的思路可引入）
- 仅在GEOM-Drugs和QM9两个数据集评估，更大规模和更复杂分子的评估有待进一步验证
- Peak memory略有增加，但增幅较小

## 相关工作与启发

- 缓存技术在图像领域已有丰富研究（DeepCache、FORA、TaylorSeer、AB-Cache、TeaCache），本文是首次向分子领域的迁移
- 与训练型加速方法（轨迹重参数化、蒸馏、潜空间方法）正交互补
- 启发：任何基于多步迭代生成的模型，只要中间特征具备平滑性，都可以考虑预测缓存加速

## 评分

- 新颖性：⭐⭐⭐⭐（首次将特征缓存迁移到分子生成）
- 技术深度：⭐⭐⭐⭐（等变性保持的理论分析扎实）
- 实验充分度：⭐⭐⭐⭐（系统的消融和组合实验）
- 实用性：⭐⭐⭐⭐⭐（免训练即插即用，实际加速效果显著）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] DreamCache: Finetuning-Free Lightweight Personalized Image Generation via Feature Caching](../../CVPR2025/image_generation/dreamcache_finetuning-free_lightweight_personalized_image_generation_via_feature.md)
- [\[AAAI 2026\] ProCache: Constraint-Aware Feature Caching with Selective Computation for Diffusion Transformer Acceleration](../../AAAI2026/image_generation/procache_constraint-aware_feature_caching_with_selective_computation_for_diffusi.md)
- [\[ICCV 2025\] LazyMAR: Accelerating Masked Autoregressive Models via Feature Caching](../../ICCV2025/image_generation/lazymar_accelerating_masked_autoregressive_models_via_feature_caching.md)
- [\[CVPR 2026\] ResCa: Residual Caching for Diffusion Transformers Acceleration](../../CVPR2026/image_generation/resca_residual_caching_for_diffusion_transformers_acceleration.md)
- [\[CVPR 2026\] Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration](../../CVPR2026/image_generation/adaptive_spectral_feature_forecasting_for_diffusion_sampling_acceleration.md)

</div>

<!-- RELATED:END -->
