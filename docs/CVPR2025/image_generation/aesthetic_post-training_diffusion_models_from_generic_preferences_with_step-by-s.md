---
title: >-
  [论文解读] Aesthetic Post-Training Diffusion Models from Generic Preferences with Step-by-step Preference Optimization
description: >-
  [CVPR 2025][图像生成][逐步偏好优化] 本文提出 Step-by-step Preference Optimization（SPO），在每个去噪步中从同一噪声潜变量采样多个候选，用 step-aware 偏好模型选择 win/lose 对来指导扩散模型微调，从通用偏好数据中隐式蒸馏美学信息，在 SD-1.5 和 SDXL 上显著提升美学质量且收敛速度远快于 DPO。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "逐步偏好优化"
  - "美学对齐"
  - "扩散模型后训练"
  - "step-aware偏好模型"
  - "在线学习"
---

# Aesthetic Post-Training Diffusion Models from Generic Preferences with Step-by-step Preference Optimization

**会议**: CVPR 2025  
**arXiv**: [2406.04314](https://arxiv.org/abs/2406.04314)  
**代码**: [https://github.com/RockeyCoss/SPO](https://github.com/RockeyCoss/SPO)  
**领域**: 对齐RLHF / 扩散模型  
**关键词**: 逐步偏好优化, 美学对齐, 扩散模型后训练, step-aware偏好模型, 在线学习

## 一句话总结

本文提出 Step-by-step Preference Optimization（SPO），在每个去噪步中从同一噪声潜变量采样多个候选，用 step-aware 偏好模型选择 win/lose 对来指导扩散模型微调，从通用偏好数据中隐式蒸馏美学信息，在 SD-1.5 和 SDXL 上显著提升美学质量且收敛速度远快于 DPO。

## 研究背景与动机

**领域现状**：将文本到图像扩散模型与人类偏好对齐是近年来的研究热点。DPO（Direct Preference Optimization）已被成功应用于扩散模型微调（如 Diffusion-DPO），通过偏好数据对来鼓励模型生成更受人类偏好的图像。

**现有痛点**：现有 DPO 方法存在两个核心问题。第一，**通用偏好与美学偏好不一致**：公开数据集（如 Pick-a-Pic）中的偏好标签混合了布局/构图意见和美学意见。一张图可能因为 prompt 对齐更好而被标为"preferred"，但在美学细节上其实更差（例如有伪影或细节模糊）。这种噪声标签会损害模型的美学改进。第二，**两轨迹方法难以捕捉细微差异**：Diffusion-DPO 比较的是两条完全不同轨迹生成的最终图像，将偏好标签传播到所有中间步。由于两条轨迹的中间态差异巨大（主要在布局上），模型难以聚焦于细微的美学差异。

**核心矛盾**：要提升美学质量，需要美学专属的偏好数据，但收集成本极高。通用偏好数据便宜但包含与美学矛盾的标签。如何从通用偏好数据中经济地提取美学信号？

**本文目标** 1）如何在不收集专门美学偏好数据的情况下提升图像美学？2）如何让偏好比较聚焦于细微的视觉细节而非大尺度布局差异？

**切入角度**：作者观察到，如果 win/lose 对来自同一个噪声潜变量、且只经过一步或很少几步去噪，那么两者之间的差异就只存在于图像细节（颜色、纹理、清晰度等美学层面），而非布局。这样即使使用通用偏好模型评判，也会自然聚焦于美学维度。

**核心 idea**：在每个去噪步从同一噪声采样多个候选、用 step-aware 偏好模型选 win/lose 对来微调扩散模型，使偏好信号自然聚焦于美学细节。

## 方法详解

### 整体框架

SPO 是一种在线强化学习方法。在训练的每个迭代中：给定一个中间噪声状态 $x_t$，从条件分布 $p_\theta(x_{t-1}|x_t, c, t)$ 中采样 $k$ 个去噪候选 $\{x_{t-1}^1, ..., x_{t-1}^k\}$。然后使用 Step-Aware Preference Model（SPM）对这 $k$ 个候选评分，选出质量最高的作为 win 样本、最低的作为 lose 样本，构成一个偏好对用于 DPO 损失更新。之后，从候选池中**随机**选一个样本作为下一步去噪的起点。整个过程在所有时间步上重复，形成逐步优化。

### 关键设计

1. **Step-Aware Preference Model (SPM)**:

    - 功能：在任意去噪时间步评估含噪中间态的图像质量
    - 核心思路：标准偏好模型（如 PickScore）只能处理干净图像 $x_0$，无法评估含噪的中间态 $x_t$。SPM 以 PickScore 为初始化，在 CLIP 视觉编码器中加入时间步条件的自适应 LayerNorm（借鉴 DiT 的设计），使模型能根据时间步 $t$ 调整特征提取行为。训练时，对每对偏好图像添加相同噪声到时间步 $t$，假设原始偏好顺序保持不变。通过以下概率公式预测偏好：$\hat{p}_w = \frac{\exp(\tau \cdot f_{\text{CLIP-V}}(x_t^w, t) \cdot f_{\text{CLIP-T}}(c))}{\exp(\tau \cdot f_{\text{CLIP-V}}(x_t^w, t) \cdot f_{\text{CLIP-T}}(c)) + \exp(\tau \cdot f_{\text{CLIP-V}}(x_t^l, t) \cdot f_{\text{CLIP-T}}(c))}$。为缓解噪声图像与预训练 CLIP 的域差距，使用 DDIM 直接从 $x_t$ 估计 $\hat{x}_0$
    - 设计动机：SPO 的核心需求是在中间去噪步评估图像质量。没有时间步条件的偏好模型面对高噪声图像会完全失效。SPM 是实现逐步偏好优化的基础设施

2. **随机选择初始化下一步**:

    - 功能：确保训练轨迹的多样性，防止偏向特定模式
    - 核心思路：每步选完 win/lose 对后，**不是**用 win 样本初始化下一步，而是从 $k$ 个候选中随机选一个。如果只用 win 样本，训练轨迹会偏向高偏好区域，降低泛化能力；如果只用 lose 样本，模型会持续从低质量区域学习。随机选择保证了轨迹分布的多样性
    - 设计动机：消融实验证实，使用 win 样本或 lose 样本初始化都导致显著的性能下降。随机选择是简单但关键的设计

3. **多步扩展（MSPO）用于强模型**:

    - 功能：为 SDXL 等强模型增加候选间的差异性
    - 核心思路：对于 SDXL 这样的强扩散模型，单步去噪产生的候选差异太小，导致 SPM 难以区分。MSPO 将单步扩展为多步：从 $x_t$ 采样 $k$ 个 $x_{t-1}$，然后每个继续去噪 $j$ 步得到 $x_{t-j}$，在 $x_{t-j}$ 层面选 win/lose 对。$j=4$ 时效果最佳。当 $j \to \infty$ 时，MSPO 退化为标准 Diffusion-DPO
    - 设计动机：平衡候选差异的大小——太小则 SPM 判断困难，太大则回到了布局差异主导的问题

### 损失函数

SPO 的损失函数是标准 DPO 损失在步级别的应用：

$\mathcal{L}(\theta) = -\mathbb{E}_{t, c, x_{t-1}^w, x_{t-1}^l} \left[ \log\sigma\left(\beta \log\frac{p_\theta(x_{t-1}^w|c,t,x_t)}{p_{\text{ref}}(x_{t-1}^w|c,t,x_t)} - \beta \log\frac{p_\theta(x_{t-1}^l|c,t,x_t)}{p_{\text{ref}}(x_{t-1}^l|c,t,x_t)}\right) \right]$

其中 $\beta=10$ 为正则化强度。训练使用 LoRA 微调，SD-1.5 用 rank=4，SDXL 用 rank=64。

## 实验关键数据

### 主实验（SDXL）

| 方法 | PickScore ↑ | HPSV2 ↑ | ImageReward ↑ | Aesthetic ↑ |
|------|------------|---------|---------------|-------------|
| SDXL | 21.95 | 26.95 | 0.538 | 5.950 |
| Diffusion-DPO | 22.64 | 29.31 | 0.944 | 6.015 |
| MAPO | 22.11 | 28.22 | 0.717 | 6.096 |
| **SPO** | **23.06** | **31.80** | **1.080** | **6.364** |

### 主实验（SD-1.5）

| 方法 | PickScore ↑ | HPSV2 ↑ | ImageReward ↑ | Aesthetic ↑ |
|------|------------|---------|---------------|-------------|
| SD-1.5 | 20.53 | 23.79 | -0.163 | 5.365 |
| DDPO | 21.06 | 24.91 | 0.082 | 5.591 |
| Diffusion-DPO | 20.98 | 25.05 | 0.112 | 5.505 |
| **SPO** | **21.43** | **26.45** | **0.171** | **5.887** |

### 消融实验

| 消融项 | PickScore | HPSV2 | ImageReward | Aesthetic |
|--------|-----------|-------|-------------|-----------|
| 用 lose 样本初始化 | 17.87 | 11.31 | -2.269 | 3.963 |
| 用 win 样本初始化 | 19.36 | 18.63 | -1.374 | 5.338 |
| **随机初始化 (SPO)** | **21.43** | **26.45** | **0.171** | **5.887** |
| SPM 无时间步条件 | 21.19 | 25.84 | 0.137 | 5.678 |
| 用 PickScore 替代 SPM | 20.28 | 23.09 | -0.298 | 5.410 |

### 关键发现

- SPO 在四个自动评估指标上全面超越 Diffusion-DPO，尤其 Aesthetic 分数提升 +0.349（SDXL）
- 用户研究显示 SPO 在视觉吸引力上以 58.27% 的胜率显著优于 Diffusion-DPO
- SPO 的训练计算量仅为 Diffusion-DPO 的 **4.9%**（SDXL）和 **20.8%**（SD-1.5），收敛速度大幅领先
- 时间步范围 [0-750] 最佳；包含太大时间步（[750-1000]）反而有害，因为高噪声区域几乎无图像细节可比较
- GenEval 上 SPO 与 SDXL 基线相比有轻微改善（+1.77%），但不如 Diffusion-DPO 在 prompt 对齐上的提升大

## 亮点与洞察

- **"从通用数据蒸馏美学"的思路非常elegent**：不需要昂贵的美学专属数据集，通过确保 win/lose 对只在细节上有差异，巧妙地让通用偏好模型的判断自然聚焦美学层面
- **训练效率碾压 DPO**：GPU 小时数仅为 DPO 的 5%（SDXL），这得益于更准确的偏好信号减少了无效更新
- **SPM 是可复用的工具**：step-aware 偏好模型作为独立组件，可用于其他需要评估中间去噪态的场景
- **随机选择初始化的简单性**：最简单的随机策略优于精心设计的选择策略，提醒我们不要过度工程化

## 局限与展望

- **不适用于 Flow Matching 模型**：SPO 要求中间步是随机的（DDIM with η=1.0），而 SD3、Flux 等 flow matching 模型的中间步是确定性的，无法从同一 $x_t$ 采样多个候选
- **对 prompt 对齐帮助有限**：SPO 专注美学细节，对布局/构图层面的 prompt 对齐改进不大
- GenEval 上 SPO (55.20) 低于 Diffusion-DPO (59.58)，说明美学与 prompt 对齐存在一定 trade-off
- 未探索与 RL-based 方法（如 DDPO）的结合
- SPM 的训练需要额外 8-29h GPU 时间，虽然是一次性成本但不可忽略

## 相关工作与启发

- **Diffusion-DPO (Wallace et al., 2023)**：将 DPO 应用于扩散模型的开创性工作，但使用两轨迹偏好传播策略
- **D3PO (Yang et al., 2023)**：类似于 DPO 但在线生成偏好对，同样受限于布局差异主导
- **DenseReward (Yang et al., 2024)**：用时间折扣改进 DPO，但仍基于两轨迹框架
- **DDPO / DPOK**：用策略梯度微调扩散模型，计算代价更高
- SPO 的逐步比较思路对未来扩散模型的精细化控制有启发——将优化粒度从轨迹级别细化到步级别，可能是一个通用的改进范式

## 评分

- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 实用性: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Finite Difference Flow Optimization for RL Post-Training of Text-to-Image Models](finite_difference_flow_optimization_for_rl_post-training_of_text-to-image_models.md)
- [\[CVPR 2025\] Curriculum Direct Preference Optimization for Diffusion and Consistency Models](curriculum_direct_preference_optimization_for_diffusion_and_consistency_models.md)
- [\[CVPR 2025\] Calibrated Multi-Preference Optimization for Aligning Diffusion Models](calibrated_multi-preference_optimization_for_aligning_diffusion_models.md)
- [\[CVPR 2025\] ShowHowTo: Generating Scene-Conditioned Step-by-Step Visual Instructions](showhowto_generating_scene-conditioned_step-by-step_visual_instructions.md)
- [\[ICML 2025\] Smoothed Preference Optimization via ReNoise Inversion for Aligning Diffusion Models with Varied Human Preferences](../../ICML2025/image_generation/smoothed_preference_optimization_via_renoise_inversion_for_aligning_diffusion_mo.md)

</div>

<!-- RELATED:END -->
