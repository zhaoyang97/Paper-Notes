---
title: >-
  [论文解读] Tortoise and Hare Guidance: Accelerating Diffusion Model Inference with Multirate Integration
description: >-
  [NeurIPS 2025][图像生成][扩散模型加速] 提出 Tortoise and Hare Guidance (THG)，一种免训练的扩散采样加速策略，将 classifier-free guidance (CFG) ODE 重构为多速率 ODE 系统，噪声估计使用细粒度步长（乌龟方程），附加引导项使用粗粒度步长（兔子方程），减少最多 30% 的函数评估次数 (NFE) 而几乎不损失生成质量。
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "扩散模型加速"
  - "Classifier-Free Guidance"
  - "多速率积分"
  - "NFE压缩"
  - "免训练"
---

# Tortoise and Hare Guidance: Accelerating Diffusion Model Inference with Multirate Integration

**会议**: NeurIPS 2025  
**arXiv**: [2511.04117](https://arxiv.org/abs/2511.04117)  
**代码**: 有 ([https://github.com/yhlee-add/THG](https://github.com/yhlee-add/THG))  
**领域**: 图像生成 / 扩散模型  
**关键词**: 扩散模型加速, Classifier-Free Guidance, 多速率积分, NFE压缩, 免训练  

## 一句话总结

提出 Tortoise and Hare Guidance (THG)，一种免训练的扩散采样加速策略，将 classifier-free guidance (CFG) ODE 重构为多速率 ODE 系统，噪声估计使用细粒度步长（乌龟方程），附加引导项使用粗粒度步长（兔子方程），减少最多 30% 的函数评估次数 (NFE) 而几乎不损失生成质量。

## 研究背景与动机

### 扩散模型的推理瓶颈
扩散模型在图像生成领域取得了巨大成功，但推理速度慢是其主要瓶颈。每次生成需要多步去噪，每步需要一次或多次网络前向传播 (Function Evaluation, NFE)。

### Classifier-Free Guidance 的计算冗余
CFG 是当前主流的条件生成方法，其公式为：
$$\hat{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t) + s \cdot [\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t)]$$
- 第一项 $\epsilon_\theta(x_t)$：无条件噪声估计
- 第二项 $s \cdot [\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t)]$：附加引导项

每步 CFG 需要**两次网络前向传播**（有条件和无条件），是主要计算瓶颈。

### 关键观察

**附加引导项对数值误差的敏感度远低于噪声估计项**。传统的均匀步长求解器未能利用这一不对称性，造成了大量冗余计算。

## 方法详解

### 整体框架

将 CFG ODE 分解为两个具有不同时间尺度的子系统：

```
CFG ODE: dx/dt = f(x,t) + g(x,t)
  ├── 乌龟方程 (Tortoise): dx/dt = f(x,t)  — 噪声估计，细粒度步长
  └── 兔子方程 (Hare):     dx/dt = g(x,t)  — 附加引导，粗粒度步长
```

### 关键设计

#### 1. 多速率 ODE 分解
将 CFG 的求解从单一 ODE 分解为多速率系统：

- **乌龟方程** (慢速, 精细)：以原始时间步长计算噪声估计 $\epsilon_\theta(x_t)$ 和 $\epsilon_\theta(x_t, c)$
- **兔子方程** (快速, 粗糙)：附加引导项 $g(x_t) = s \cdot [\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t)]$ 仅在粗网格上计算

关键点：兔子方程跳跃多个细粒度步长，在粗网格点之间使用插值或外推。

#### 2. 误差界分析
通过严格的误差界分析证明：
- 噪声估计项的 Lipschitz 常数较大，需要细步长控制误差
- 引导项的 Lipschitz 常数较小，可以容忍较大步长
- 量化关系：引导项的误差界比噪声估计项小 $O(s)$ 倍

#### 3. 误差界感知的时间步采样器 (Error-bound-aware Timestep Sampler)
自适应选择步长大小：
- 在噪声发生快速变化的区域（如时间步中间阶段），使用更细的步长
- 在变化平缓的区域（如接近终止时间），允许更大的步长
- 基于局部误差估计动态调整乌龟/兔子方程的步长比

#### 4. 引导尺度调度器 (Guidance-scale Scheduler)
当兔子方程跨越较大的时间区间时，简单外推可能不稳定。引入调度器：
- 在大跨度区间适当降低引导尺度 $s$
- 确保外推的稳定性
- 不影响最终生成质量

### 损失函数 / 训练策略

THG 是**完全免训练**的方法：
- 不需要修改或重新训练扩散模型
- 仅改变推理时的 ODE 求解策略
- 与任何 CFG 兼容的扩散模型即插即用

## 实验关键数据

### 主实验

在 Stable Diffusion 和 SDXL 上的表现：

| 方法 | NFE ↓ | FID ↓ | CLIP Score ↑ | ImageReward ↑ | ΔImageReward |
|------|-------|-------|-------------|--------------|-------------|
| DDIM (50步) | 100 | 15.2 | 0.312 | 0.876 | 基准 |
| DPM-Solver++ (25步) | 50 | 15.8 | 0.310 | 0.871 | -0.005 |
| PNDM (25步) | 50 | 16.1 | 0.308 | 0.865 | -0.011 |
| PAB | 70 | 15.5 | 0.311 | 0.872 | -0.004 |
| DeepCache | 60 | 16.4 | 0.307 | 0.858 | -0.018 |
| **THG (ours)** | **70** | **15.3** | **0.311** | **0.873** | **-0.003** |
| **THG (ours, aggressive)** | **50** | 15.9 | 0.309 | 0.844 | -0.032 |

在相同 NFE 预算下的对比：

| 方法 | NFE=50 FID ↓ | NFE=50 ImageReward ↑ | NFE=70 FID ↓ | NFE=70 ImageReward ↑ |
|------|------------|-------------------|------------|-------------------|
| DPM-Solver++ | 15.8 | 0.871 | 15.4 | 0.874 |
| DeepCache | 17.2 | 0.845 | 16.4 | 0.858 |
| PAB | 16.5 | 0.860 | 15.5 | 0.872 |
| **THG** | **15.5** | **0.878** | **15.3** | **0.873** |

### 消融实验

| 配置 | NFE | FID | ImageReward | 关键 |
|------|-----|-----|------------|------|
| Full THG | 70 | **15.3** | **0.873** | 完整方法 |
| 无自适应步长 | 70 | 15.8 | 0.865 | 自适应步长重要 |
| 无引导尺度调度 | 70 | 16.2 | 0.852 | 调度器稳定大跨度外推 |
| 粗网格比=2:1 | 85 | 15.4 | 0.872 | 保守设置 |
| 粗网格比=4:1 | 55 | 16.5 | 0.838 | 过于激进 |
| 粗网格比=3:1 (默认) | 70 | 15.3 | 0.873 | 最佳平衡 |

### 关键发现

1. **30% NFE 减少几乎无损**：THG 在减少 30% 计算的情况下，$\Delta$ImageReward $\leq 0.032$
2. **优于同等预算的替代方法**：在相同 NFE 下，THG 的 FID 和 ImageReward 均优于 DeepCache、PAB 等
3. **自适应步长贡献显著**：相比固定步长比，自适应步长采样器提供了 +0.008 ImageReward 提升
4. **引导尺度调度是稳定性保障**：移除调度器导致 FID 上升 0.9，主要影响大引导尺度场景
5. **3:1 步长比最优**：乌龟:兔子 = 3:1 在效率和质量间取得最佳平衡

## 亮点与洞察

1. **数学动机清晰**：从 ODE 误差分析出发，发现引导项的鲁棒性，是严格推导而非经验观察
2. **免训练设计**：无需额外训练成本，即插即用
3. **命名生动**：乌龟和兔子的比喻直观传达了多速率的思想
4. **实用性强**：可与现有扩散模型直接集成，加速推理
5. **开源代码**：促进社区复现和扩展

## 局限与展望

1. **仅适用于 CFG 模式**：对不使用 CFG 的方法（如 flow matching）不直接适用
2. **外推精度有限**：当引导尺度 $s$ 很大时，粗网格外推可能引入可察觉的伪影
3. **步长比需要调优**：最优步长比可能依赖于具体模型和任务
4. **与其他加速技术的兼容性**：与蒸馏等方法的联合使用有待探索
5. **视频生成扩展**：对视频扩散模型的适用性有待验证

## 相关工作与启发

- **DPM-Solver++**：高阶 ODE 求解器加速扩散采样
- **DeepCache**：缓存中间特征减少冗余计算
- **PAB (Pyramid Attention Broadcast)**：渐进式注意力广播加速
- **多速率积分**：经典数值分析中的技术，本文首次应用于扩散模型
- **启发方向**：探索更细粒度的组件级多速率分解（如不同层使用不同步长）

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首次将多速率积分引入扩散采样加速
- **理论深度**: ⭐⭐⭐⭐ — 有严格的误差界分析
- **实验充分性**: ⭐⭐⭐⭐ — 多模型、多指标、充分消融
- **实际影响**: ⭐⭐⭐⭐⭐ — 免训练、即插即用，直接降低推理成本
- **写作质量**: ⭐⭐⭐⭐ — 清晰生动，命名巧妙

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] SpecDiff: Accelerating Diffusion Model Inference with Self-Speculation](../../AAAI2026/image_generation/specdiff_accelerating_diffusion_model_inference_with_self-speculation.md)
- [\[NeurIPS 2025\] Accelerating Parallel Diffusion Model Serving with Residual Compression](accelerating_parallel_diffusion_model_serving_with_residual_compression.md)
- [\[CVPR 2026\] SenCache: Accelerating Diffusion Model Inference via Sensitivity-Aware Caching](../../CVPR2026/image_generation/sencache_accelerating_diffusion_model_inference_via_sensitivity-aware_caching.md)
- [\[NeurIPS 2025\] Entropy Rectifying Guidance for Diffusion and Flow Models](entropy_rectifying_guidance_for_diffusion_and_flow_models.md)
- [\[CVPR 2026\] Accelerating Diffusion via Hybrid Data-Pipeline Parallelism Based on Conditional Guidance Scheduling](../../CVPR2026/image_generation/accelerating_diffusion_via_hybrid_data-pipeline_parallelism_based_on_conditional.md)

</div>

<!-- RELATED:END -->
