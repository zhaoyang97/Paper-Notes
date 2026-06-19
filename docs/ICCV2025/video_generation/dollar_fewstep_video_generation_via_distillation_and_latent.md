---
title: >-
  [论文解读] DOLLAR: Few-Step Video Generation via Distillation and Latent Reward Optimization
description: >-
  [ICCV 2025][视频生成][video generation] 结合变分分数蒸馏（VSD）和一致性蒸馏实现少步视频生成，同时提出潜空间奖励模型微调方法进一步优化特定质量维度，4步student模型在VBench上达82.57分超越teacher模型和Gen-3/Kling等商业基线，1步蒸馏实现278.6倍采样加速。
tags:
  - "ICCV 2025"
  - "视频生成"
  - "video generation"
  - "distillation"
  - "consistency distillation"
  - "latent reward"
  - "few-step generation"
---

# DOLLAR: Few-Step Video Generation via Distillation and Latent Reward Optimization

**会议**: ICCV 2025  
**arXiv**: [2412.15689](https://arxiv.org/abs/2412.15689)  
**代码**: 无  
**领域**: 视频生成 / 扩散模型加速  
**关键词**: video generation, distillation, consistency distillation, latent reward, few-step generation

## 一句话总结

结合变分分数蒸馏（VSD）和一致性蒸馏实现少步视频生成，同时提出潜空间奖励模型微调方法进一步优化特定质量维度，4步student模型在VBench上达82.57分超越teacher模型和Gen-3/Kling等商业基线，1步蒸馏实现278.6倍采样加速。

## 研究背景与动机

**领域现状**：扩散概率模型在视频生成领域取得了显著进展，能够生成高质量的文本到视频内容。然而，这些模型通常需要50步以上的迭代采样才能生成满意的视频，计算开销极大。

**现有痛点**：直接减少采样步数往往会导致视频质量退化或生成多样性下降。现有的图像领域蒸馏方法（如SANA-Sprint可实现1-4步图像生成）在视频领域面临额外挑战——视频需要在时间维度上维持一致性，简单减少步数极易产生时间闪烁和质量下降。此外，现有视频蒸馏方法通常仅支持特定步数（如仅能4步采样），缺乏步数灵活性。

**核心矛盾**：如何在大幅减少采样步数的同时保持视频的质量和多样性？如何通过奖励信号进一步优化蒸馏后模型的特定质量指标？

**本文目标**：提出DOLLAR框架，通过混合蒸馏策略实现灵活步数的高质量视频生成，并引入潜空间奖励微调进一步提升特定质量维度。

**切入角度**：将VSD（保多样性）和一致性蒸馏（保质量与步数灵活性）互补结合，同时在潜空间而非像素空间进行奖励优化以降低显存开销。

**核心 idea**：混合蒸馏+潜空间奖励微调=少步高质量视频生成。

## 方法详解

### 整体框架

DOLLAR采用两阶段训练策略：(1) **混合蒸馏阶段**——结合变分分数蒸馏(VSD)和一致性蒸馏(CD)，将teacher模型的50步采样能力压缩至student模型的1-4步；(2) **潜空间奖励优化阶段**——使用latent reward model对蒸馏后的student模型进行微调，针对性地提升特定质量指标。

### 关键设计

1. **变分分数蒸馏（VSD）**：
    - 功能：通过分布匹配将teacher模型的多步采样分布对齐到student模型的少步分布
    - 核心思路：VSD最小化teacher和student输出分布之间的KL散度。设teacher模型为 $\epsilon_\phi$，student模型为 $\epsilon_\theta$，VSD优化目标为 $\mathcal{L}_{\text{VSD}} = \mathbb{E}_{t,\epsilon}\left[\|\epsilon_\theta(x_t, t) - \epsilon_\phi(x_t, t)\|^2\right]$，其中 $x_t$ 为加噪后的视频latent
    - 设计动机：纯一致性蒸馏可能导致模式坍缩（mode collapse），VSD通过分布匹配保证生成多样性

2. **一致性蒸馏（CD）**：
    - 功能：确保student模型在不同步数下输出一致的结果
    - 核心思路：对同一噪声输入，要求student在任意中间时间步的预测结果一致，即 $f_\theta(x_t, t) \approx f_\theta(x_{t'}, t')$，其中 $x_t$ 和 $x_{t'}$ 位于同一PF-ODE轨迹上
    - 设计动机：使student不受限于固定步数，可在1-4步之间灵活选择，避免step-specific训练

3. **混合蒸馏策略**：
    - 功能：将VSD和CD以加权方式联合训练
    - 核心思路：总损失为 $\mathcal{L} = \lambda_{\text{VSD}} \mathcal{L}_{\text{VSD}} + \lambda_{\text{CD}} \mathcal{L}_{\text{CD}}$
    - 设计动机：VSD保多样性，CD保质量和步数灵活性，两者互补

4. **潜空间奖励模型微调（Latent Reward Fine-tuning）**：
    - 功能：利用潜空间奖励模型对蒸馏后的student进一步微调，提升指定质量维度
    - 核心思路：不要求reward model可微，而是在latent空间中操作。通过在潜空间直接计算奖励信号，避免将视频解码到像素空间，大幅降低GPU显存需求。可针对任意奖励指标（美学质量、文本对齐、时间一致性等）进行优化
    - 设计动机：传统奖励微调需要在像素空间计算梯度，对于128帧视频显存开销不可接受；潜空间操作同时解决显存和可微性两个问题

### 损失函数 / 训练策略

- 第一阶段：VSD loss + Consistency loss 联合训练，权重在训练过程中动态调整
- 第二阶段：Latent reward fine-tuning，使用策略梯度方法优化reward指标
- 训练数据：使用teacher模型的50步DDIM采样结果作为目标分布
- 视频规格：128帧@12FPS（约10秒），远超之前大多数方法（2-4秒）

## 实验关键数据

### 主实验

| 方法 | 步数 | VBench Score | 加速比 |
|------|------|-------------|--------|
| Teacher (50-step DDIM) | 50 | ~81 | 1× |
| DOLLAR (4-step) | 4 | **82.57** | 12.5× |
| DOLLAR (1-step) | 1 | ~79 | **278.6×** |
| Gen-3 | - | <82.57 | - |
| T2V-Turbo | 4 | <82.57 | - |
| Kling | - | <82.57 | - |

### 消融实验

| 配置 | VBench Score | 说明 |
|------|-------------|------|
| 仅VSD | ~80 | 多样性好但质量不足 |
| 仅CD | ~79 | 质量好但多样性差 |
| VSD + CD | ~81.5 | 互补提升 |
| VSD + CD + Latent Reward | **82.57** | 奖励微调进一步突破 |

### 关键发现

- 4步student模型在VBench上超越50步teacher模型，蒸馏+奖励微调可超越原始模型
- 1步蒸馏实现278.6倍加速，接近实时生成
- 人类评估进一步验证4步student优于50步teacher
- 潜空间奖励微调使蒸馏后模型在特定维度（如时间一致性）上显著提升

## 亮点与洞察

- **蒸馏+奖励=超越teacher**：蒸馏后的student不仅更快，还能通过奖励微调在质量上超越teacher，打破"蒸馏必然损失质量"的常见认知
- **潜空间而非像素空间**：在潜空间进行奖励计算大幅降低显存需求，使长视频（10秒128帧）的奖励微调变得可行
- **步数灵活性**：一致性蒸馏赋予的步数灵活性（1-4步均可用）为不同应用场景提供了质量-速度的灵活权衡
- **10秒长视频验证**：在128帧@12FPS的10秒视频上验证，远超之前方法仅在短视频上的实验

## 局限与展望

- 论文未公开代码，可复现性有待验证
- 潜空间奖励模型的具体设计细节和训练过程描述不够充分
- 仅在VBench上验证，缺少更多元化的视频质量评估基准
- 未讨论更长视频（如30秒以上）或更高分辨率场景的适用性
- 奖励微调依赖奖励模型质量，存在over-optimization风险

## 相关工作与启发

- **Consistency Models**：Song et al. 提出的一致性模型是本文CD组件的理论基础
- **ProlificDreamer/VSD**：Wang et al. 提出的VSD被本文拓展到视频蒸馏
- **SANA-Sprint**：图像领域的1-4步生成先行工作
- **T2V-Turbo**：之前的视频蒸馏工作，本文性能超越
- **启发**：潜空间奖励微调的思路可推广到其他生成模型（3D生成、音频合成等）

## 评分

- 新颖性: ⭐⭐⭐⭐ VSD+CD混合蒸馏和潜空间奖励微调的结合在视频生成领域较新
- 实验充分度: ⭐⭐⭐ VBench结果有说服力，但缺消融细节和多基准对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机明确
- 价值: ⭐⭐⭐⭐ 接近实时的视频生成具有重要应用价值
---
title: >-
  [论文解读] DOLLAR: Few-Step Video Generation via Distillation and Latent Reward Optimization
description: >-
  [ICCV 2025][视频理解][video generation] 结合变分分数蒸馏（VSD）和一致性蒸馏实现few-step视频生成，同时提出潜空间奖励模型微调方法进一步优化生成质量，4步生成的10秒视频（128帧@12FPS）在VBench上达82.57分超越teacher模型和Gen-3/Kling等基线，1步蒸馏实现278.6倍加速。
tags:
  - ICCV 2025
  - 视频理解
  - video generation
  - distillation
  - consistency distillation
  - latent reward
  - few-step
  - VBench
---

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] SGMD: Score Gradient Matching Distillation for Few-Step Video Diffusion](../../ICML2026/video_generation/sgmd_score_gradient_matching_distillation_for_few-step_video_diffusion_distillat.md)
- [\[CVPR 2025\] FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance](../../CVPR2025/video_generation/flashmotion_few-step_controllable_video_generation_with_trajectory_guidance.md)
- [\[ICCV 2025\] VPO: Aligning Text-to-Video Generation Models with Prompt Optimization](vpo_aligning_text-to-video_generation_models_with_prompt_optimization.md)
- [\[ICCV 2025\] Adversarial Distribution Matching for Diffusion Distillation Towards Efficient Image and Video Synthesis](adversarial_distribution_matching_for_diffusion_distillation_towards_efficient_i.md)
- [\[ICCV 2025\] FVGen: Accelerating Novel-View Synthesis with Adversarial Video Diffusion Distillation](fvgen_accelerating_novel-view_synthesis_with_adversarial_video_diffusion_distill.md)

</div>

<!-- RELATED:END -->
