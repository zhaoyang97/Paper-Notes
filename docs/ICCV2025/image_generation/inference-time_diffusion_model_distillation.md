---
title: >-
  [论文解读] Inference-Time Diffusion Model Distillation
description: >-
  [ICCV 2025][图像生成][扩散蒸馏] 提出 Distillation++，一种推理时扩散蒸馏框架，在采样过程中利用预训练教师模型的引导来修正学生蒸馏模型的去噪路径，无需额外训练数据或微调即可显著缩小师生模型间的性能差距。 扩散模型通过迭代去噪过程生成高质量图像，但采样速度慢（通常需要几十到上百步 NFE）…
tags:
  - "ICCV 2025"
  - "图像生成"
  - "扩散蒸馏"
  - "推理时蒸馏"
  - "Score Distillation Sampling"
  - "教师引导采样"
  - "少步生成"
---

# Inference-Time Diffusion Model Distillation

**会议**: ICCV 2025  
**arXiv**: [2412.08871](https://arxiv.org/abs/2412.08871)  
**代码**: [GitHub](https://github.com/geon-yeong/Distillation-pp)  
**领域**: 扩散模型/图像生成  
**关键词**: 扩散蒸馏, 推理时蒸馏, Score Distillation Sampling, 教师引导采样, 少步生成

## 一句话总结

提出 Distillation++，一种推理时扩散蒸馏框架，在采样过程中利用预训练教师模型的引导来修正学生蒸馏模型的去噪路径，无需额外训练数据或微调即可显著缩小师生模型间的性能差距。

## 研究背景与动机

扩散模型通过迭代去噪过程生成高质量图像，但采样速度慢（通常需要几十到上百步 NFE）。蒸馏模型（学生模型）通过对预训练扩散模型（教师模型）进行知识压缩，将采样过程压缩到少数几步（如 4 步），大幅加速了生成过程。

然而，现有蒸馏模型仍面临两个核心挑战：

**师生性能差距**：蒸馏模型在多步采样时会累积误差，导致生成质量不如教师模型。例如 Consistency Models 在增加 NFE 时质量并不一定提升，因为一致性误差在时间区间间累积。

**分布偏移问题**：一些方法引入真实训练数据来弥补差距，但教师和学生的数据分布不一致可能导致在分布外（OOD）提示词上的性能下降。

**后训练选项缺乏**：很多蒸馏模型直接预测 PF-ODE 端点而非轨迹切线方向，导致它们与传统 ODE 求解器不兼容，限制了后训练改进的空间。

现有工作仅在训练阶段进行蒸馏，而本文提出了一个关键洞察：**能否在推理阶段继续利用教师模型来引导学生模型的采样？** 这开辟了推理时蒸馏这一全新方向。

## 方法详解

### 整体框架

Distillation++ 的核心思想是在学生模型的采样过程中（特别是早期 1-2 步），引入教师模型的引导来修正去噪路径。具体来说，将学生模型的采样重新建模为一个近端优化问题，并以 Score Distillation Sampling（SDS）损失作为正则化项。

### 关键设计

1. **SDS 蒸馏损失（$\ell_{\text{distill}}$）**:

    - 功能：定义学生去噪估计与教师模型之间的对齐损失
    - 核心思路：将学生的去噪估计 $\hat{x}_0^\theta(t)$ 重新加噪到时刻 $s$，然后用教师模型去噪得到 $\hat{x}_0^\psi(s)$，SDS 损失简化为：
    $\ell_{\text{distill}}(x; \psi, s) = \frac{\bar{\alpha}_s}{1-\bar{\alpha}_s} \|x - \hat{x}_0^\psi(s)\|_2^2$
    - 设计动机：高质量的去噪估计应当在被随机扰动后，仍能通过教师模型良好地重建。这沿用了 SDS 框架在蒸馏训练中的成功经验，将其扩展到推理阶段。

2. **推理时教师引导更新规则**:

    - 功能：在每个采样步中，将学生估计与教师估计融合
    - 核心思路：通过 Decomposed Diffusion Sampling（DDS）框架绕开不可行的 Jacobian 计算，得到简洁的更新公式：
    $\hat{x}_{\text{new}}^\theta(t) = (1-\lambda)\hat{x}_0^\theta(t) + \lambda \hat{x}_0^\psi(s)$
      其中 $\lambda$ 为引导强度。这等价于学生和教师去噪估计之间的插值，然后代入 DDIM 采样公式继续前进。
    - 设计动机：这种插值形式类似于 CFG 中的条件引导机制，但引导方向来自教师模型而非文本条件，因此称为"教师引导"（Teacher Guidance）。

3. **减噪时间调度策略（Renoising Schedule）**:

    - 功能：设计教师模型评估的时间步 $s$ 相对于当前步 $t$ 的关系
    - 核心思路：采用递减时间步调度 $s = t - \Delta t$，而非传统 SDS 中的随机时间步。这模拟反向扩散的渐进改进过程。
    - 设计动机：学生模型通常学习跳跃到每个子区间的端点，在端点处利用教师模型修正方向效果最好。实验表明递减调度优于随机调度和固定调度。

### 损失函数 / 训练策略

Distillation++ 是**无训练**的框架，不需要任何梯度更新或微调。它仅在推理采样中修改去噪过程：

- 仅在前 1-2 步施加教师引导，最小化额外计算开销
- 使用简单的常数 $\lambda$ 作为引导强度
- 兼容多种学生模型（LCM、DMD2、SDXL-Lightning 等）和多种求解器（Euler、DPM++ 2S Ancestral）

## 实验关键数据

### 主实验

在 MS-COCO 10K 上进行定量评估，4 步基线采样 + 1 步推理时蒸馏：

| 模型 | FID↓ | ImageReward↑ | PickScore↑ |
|------|------|-------------|------------|
| LCM | 20.674 | 0.561 | 0.494 |
| **LCM++** | **20.149** | **0.597** | **0.505** |
| LCM-LoRA | 20.300 | 0.494 | 0.490 |
| **LCM-LoRA++** | **19.815** | **0.522** | **0.510** |
| SDXL-Lightning | 24.506 | 0.787 | 0.496 |
| **Light++** | **23.876** | **0.820** | **0.503** |
| DMD2 | 21.238 | 0.777 | 0.490 |
| **DMD2++** | **20.937** | **0.797** | **0.510** |
| SDXL-Turbo | 18.612 | 0.296 | 0.499 |
| **Turbo++** | **18.481** | **0.310** | **0.501** |

### 消融实验

| 配置 | FID↓ | ImageReward↑ | 说明 |
|------|------|-------------|------|
| DMD2 基线 | 21.238 | 0.777 | 无教师引导 |
| s=random t | 21.105 | 0.771 | 随机时间步 |
| s=t | 21.342 | 0.777 | 同步时间步 |
| **s=t−Δt** | **20.937** | **0.797** | 递减时间步（最优） |

计算开销对比（LCM, 4+1 步 vs 5 步 vs 6 步）：

| 指标 | 4+1步 | 5步 | 6步 |
|------|-------|-----|-----|
| FID↓ | **20.149** | 20.732 | 21.540 |
| ImageReward↑ | **0.597** | 0.593 | 0.585 |
| 时间(秒) | 1.987 | 1.996 | 2.250 |

### 关键发现

- Distillation++ 在所有蒸馏基线上**一致性**地提升 FID、ImageReward 和 PickScore
- 仅增加 1 步教师评估的时间开销与增加 1 步学生采样相当甚至更短（得益于并行计算）
- 增加学生模型的采样步数并不能保证语义对齐和物理可行性的改善，但教师引导可以

## 亮点与洞察

- **首个推理时蒸馏框架**：不同于所有现有方法在训练时蒸馏，Distillation++ 在采样时持续利用教师引导，建立了师生终身协作的范式
- **无数据无微调**：完全免于额外训练数据或参数更新，是一种即插即用的后训练改进方案
- **通用兼容性**：适用于不同类型的蒸馏模型和求解器
- **早期引导效果显著**：空间布局在采样初期已基本确定，因此仅需在前 1-2 步引导即可获得大幅改善

## 局限与展望

- 推理时需要加载教师模型（如 SDXL），增加了显存需求
- 目前仅在图像生成上验证，视频扩散蒸馏是有前景的扩展方向
- 恒定 $\lambda$ 可能不是最优，时间依赖的引导强度有进一步探索空间
- 与 Flow Matching 模型的协同采样值得探索

## 相关工作与启发

- 与 DreamSampler、CFG++ 等条件采样工作密切相关，将其思想从条件引导扩展到师生蒸馏引导
- SDS 损失在 3D 生成中广泛使用，本文将其从训练目标重新定位为推理时优化目标
- 启示：推理时计算（inference-time compute）不仅适用于 LLM，在扩散模型中同样有价值

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次提出推理时蒸馏概念，开辟全新方向
- 实验充分度: ⭐⭐⭐⭐ 多基线验证，消融和计算开销分析详尽
- 写作质量: ⭐⭐⭐⭐ 数学推导清晰，从 SDS 到插值形式的简化优雅
- 价值: ⭐⭐⭐⭐ 即插即用的实用方案，但额外的教师模型开销限制了部署场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation](sana-sprint_one-step_diffusion_with_continuous-time_consistency_distillation.md)
- [\[ICCV 2025\] Fewer Denoising Steps or Cheaper Per-Step Inference: Towards Compute-Optimal Diffusion Model Deployment](fewer_denoising_steps_or_cheaper_per-step_inference_towards_compute-optimal_diff.md)
- [\[CVPR 2025\] Random Conditioning for Diffusion Model Compression with Distillation](../../CVPR2025/image_generation/random_conditioning_for_diffusion_model_compression_with_distillation.md)
- [\[NeurIPS 2025\] Progressive Inference-Time Annealing of Diffusion Models for Sampling from Boltzmann Densities](../../NeurIPS2025/image_generation/progressive_inference-time_annealing_of_diffusion_models_for_sampling_from_boltz.md)
- [\[ICLR 2026\] Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency](../../ICLR2026/image_generation/large_scale_diffusion_distillation_via_score-regularized_continuous-time_consist.md)

</div>

<!-- RELATED:END -->
