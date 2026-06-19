---
title: >-
  [论文解读] DenseDPO: Fine-Grained Temporal Preference Optimization for Video Diffusion Models
description: >-
  [NeurIPS 2025 (Spotlight)][视频生成][扩散模型] 识别并解决视频 DPO 的运动偏差问题——通过从 GT 视频加噪去噪构造结构对齐的视频对来固定运动维度、在时间片段级标注密集偏好来获取更精准的学习信号、用现成 VLM 自动标注来降低成本，仅用 1/3 标注数据即大幅提升运动生成质量同时匹配视觉质量和文本对齐。
tags:
  - "NeurIPS 2025 (Spotlight)"
  - "视频生成"
  - "扩散模型"
  - "DPO"
  - "运动偏差"
  - "片段级偏好"
  - "引导式生成"
---

# DenseDPO: Fine-Grained Temporal Preference Optimization for Video Diffusion Models

**会议**: NeurIPS 2025 (Spotlight)  
**arXiv**: [2506.03517](https://arxiv.org/abs/2506.03517)  
**代码**: [https://snap-research.github.io/DenseDPO/](https://snap-research.github.io/DenseDPO/)  
**领域**: 视频生成 / 偏好优化  
**关键词**: video diffusion, DPO, 运动偏差, 片段级偏好, 引导式生成

## 一句话总结

识别并解决视频 DPO 的运动偏差问题——通过从 GT 视频加噪去噪构造结构对齐的视频对来固定运动维度、在时间片段级标注密集偏好来获取更精准的学习信号、用现成 VLM 自动标注来降低成本，仅用 1/3 标注数据即大幅提升运动生成质量同时匹配视觉质量和文本对齐。

## 研究背景与动机

**领域现状**：DPO（Direct Preference Optimization）已被成功应用于图像扩散模型的后训练对齐，并开始迁移到视频扩散模型。标准流程是从独立噪声生成两个视频，让人类标注偏好，再用 DPO 损失优化模型。

**现有痛点**：直接从图像 DPO 迁移到视频时暴露了一个被忽视的根本缺陷——**运动偏差（motion bias）**。由于当前视频模型擅长生成高质量的慢动作视频但在动态场景中容易产生瑕疵，从独立噪声生成的视频对中，标注者系统性地偏好静止但干净的视频而非动态但有瑕疵的。DPO 训练后模型进一步学会了"少动=好"，运动强度大幅下降。这一问题在 Snap 和多篇文献中反复出现。

**核心矛盾**：视频质量是多维度的——视觉质量和动态程度往往负相关。当标注者被要求对整个视频给一个二元偏好时，这些维度被不可避免地纠缠在一起。同时，人类对长视频（如 5 秒）的偏好评判不够精确——瑕疵可能只出现在部分时间段，但整体标签无法表达这种细粒度差异。

**本文目标** （1）消除视频 DPO 中的运动偏差；（2）获得更精确的时间细粒度偏好信号；（3）降低视频偏好标注成本。

**切入角度**：借鉴 Pareto 优化思想——要在多个目标间优化，应该固定某些属性只变化其他属性。具体地，受 SDEdit 启发，从 GT 视频加噪再去噪来生成结构相似（运动对齐）但细节不同的视频对。

**核心 idea**：用引导式采样控制运动变量不变，在时间片段级标注密集偏好，让 DPO 只优化视觉质量而不损害运动。

## 方法详解

### 整体框架

DenseDPO 修改了标准视频 DPO pipeline 的两个核心环节：**数据构造**（用引导式采样取代独立噪声采样）和**标注粒度**（用片段级密集偏好取代整视频二元偏好）。整体流程：（1）从真实视频数据集中选取高质量视频；（2）对每个视频加部分噪声后用两个不同随机种子去噪，得到运动对齐但细节不同的视频对；（3）将视频切成短片段（如 1 秒），在片段级标注偏好；（4）用修改后的 DPO 损失训练模型。

### 关键设计

1. **StructuralDPO：引导式视频对构造（消除运动偏差）**:

    - 功能：生成运动轨迹相似但局部视觉细节不同的视频对
    - 核心思路：给定 GT 视频 $\mathbf{x}$ 和引导级别 $\eta \in [0.65, 0.8]$，构造部分加噪视频 $\mathbf{x}_n^0 = (1-\eta)\mathbf{x} + \eta \boldsymbol{\epsilon}^0$ 和 $\mathbf{x}_n^1 = (1-\eta)\mathbf{x} + \eta \boldsymbol{\epsilon}^1$，然后从步 $n = \text{round}(\eta \cdot N)$ 去噪到步 1。$\eta$ 控制结构相似度——低 $\eta$ 意味着更强引导，视频更相似
    - 设计动机：扩散模型的早期步控制全局运动和布局，后期步控制局部细节。从部分加噪出发保留了 GT 的运动轨迹，使两个视频仅在局部细节上竞争——标注者无法基于运动量做偏好选择，只能比较视觉质量。这从根本上消除了运动偏差

2. **DenseDPO：片段级密集偏好标注**:

    - 功能：将一对视频的单个偏好标签扩展为多个片段的独立偏好标签
    - 核心思路：将 T 帧视频拆分为 $F = \lceil T/s \rceil$ 个片段（每段 $s$ 帧，默认 1 秒），对每个片段独立标注偏好 $\mathbf{l} \in \{-1, +1\}^F$。因为引导式采样产生的视频在时间上有一一对应关系，跨片段比较是可行的。DPO 损失变为 $\mathcal{L} = -\mathbb{E}\left[\log\sigma\left(-\beta \sum_{f=1}^{F} l_f \cdot (s_f^0 - s_f^1)\right)\right]$
    - 设计动机：实验发现超过 60% 的视频对中，不同时间片段的偏好方向不一致——某些片段视频 A 更好，某些片段视频 B 更好。整视频标注要么给 tie 要么选"瑕疵偏少的那个"，无法利用这种细粒度信息。片段级标注一举三得：更精确的信号、更少的 tie（>80% 对含至少一个非 tie 片段）、更多有效训练样本

3. **VLM 自动标注片段级偏好**:

    - 功能：用现成视觉语言模型（如 GPT-o3）替代人工标注
    - 核心思路：将两个视频的对应短片段（~1 秒）输入 VLM，让其判断哪个更好。关键洞察是虽然 VLM 在评判长视频（5 秒）时表现差，但在短片段上能给出与人类一致的判断
    - 设计动机：训练专用视频奖励模型需要大量数据和标注，而 DenseDPO 恰好将问题简化到 VLM 能处理的粒度——短片段比较。这降低了整个方法的使用门槛

### 损失函数 / 训练策略

基于 Flow-DPO 的 rectified flow 扩展。$\beta = 500$，LoRA rank=128 微调视频模型。AdamW 优化器，global batch size=256，训练 1000 步。从 55K 高质量视频中选取 30K prompt，DenseDPO 只用其中 10K 视频对标注密集偏好。

## 实验关键数据

### 主实验

在 VideoJAM-bench（128 个强运动 prompt）和 MotionBench（419 个多样化动态 prompt）上评估，使用 VBench 和 VisionReward 两套指标体系。

| 方法 | Dynamic Degree↑ | Visual Quality↑ | Text Alignment↑ | Motion Smoothness↑ | 标注量 |
|------|-----------------|------------------|------------------|---------------------|--------|
| Pre-trained | 84.16 | 0.192 | 0.770 | 92.40 | — |
| SFT | 83.25 | 0.205 | 0.773 | 92.72 | — |
| VanillaDPO | **80.25** ↓ | 0.371 | 0.867 | 93.43 | 10K 对 |
| StructuralDPO | 84.69 | 0.341 | 0.843 | 92.94 | 10K 对 |
| **DenseDPO** | **85.38** | **0.376** | **0.863** | **93.56** | **~3K 对** |

VanillaDPO 的 Dynamic Degree 从 84.16 暴跌到 80.25（运动偏差的直接证据），而 DenseDPO 恢复到 85.38 同时在视觉质量和文本对齐上匹配 VanillaDPO。

### 消融实验

| 配置 | Dynamic Degree | Visual Quality | 说明 |
|------|---------------|----------------|------|
| VanillaDPO | 80.25 | 0.371 | 严重运动偏差 |
| StructuralDPO | 84.69 | 0.341 | 恢复运动但质量略降 |
| DenseDPO (人工标注) | 85.38 | 0.376 | 全面恢复 |
| DenseDPO (GPT-o3 标注) | ~85 | ~0.37 | 接近人工标注 |
| 片段长度 s=1s | 最优 | 最优 | 默认选择 |
| 片段长度 s=全视频 | 退化为 Structural | 退化为 Structural | 验证密集标注必要性 |

### 关键发现

- **运动偏差是视频 DPO 的核心缺陷**：VanillaDPO 在所有实验中都显著降低运动强度，这是从图像 DPO 直接迁移的代价。控制变量（引导式采样）是解决的关键。
- **60%+ 视频对包含混合偏好**：不同时间段偏好方向不同。整视频标注对这些样本的处理要么丢弃（tie）要么引入噪声——片段级标注本质上解决了数据利用率问题。
- **VLM 在短片段上可靠**：GPT-o3 在 1 秒片段的偏好判断上与人类高度一致，而在 5 秒整视频上失败。DenseDPO 将偏好粒度缩小到 VLM 的能力范围内。
- **数据效率提升显著**：DenseDPO 仅用约 3K 有效视频对（10K 中 80% 含非 tie 片段但每对产生多个训练信号）即超越 10K 对的 VanillaDPO。

## 亮点与洞察

- **精准诊断运动偏差**：这是一个被广泛观察但未被系统解决的问题。论文不仅定义了问题，还通过定量实验证明 VanillaDPO 在所有测试中都出现运动下降。识别正确问题本身就是重要贡献。
- **Pareto 控制变量的思想极具迁移价值**：借鉴 SDEdit 的引导式采样来"固定运动、只比视觉"，这种控制变量设计可推广到任何多维偏好学习——例如在图像中固定构图只比色彩，在文本中固定逻辑只比表达。
- **片段级标注是视频偏好的正确粒度**：视频的时间维度是图像没有的，需要时间维度上的密集信号。这个观察类似于 NLP 中从句子级到 token 级偏好学习的演进。
- **VLM 自动标注的实用突破**：不需要训练专用视频奖励模型，仅利用现成 VLM 在短片段上的可靠性即可完成标注，大幅降低了方法的使用门槛。

## 局限与展望

- **依赖 GT 视频**：引导式采样需要真实视频作为起点，无法做纯生成的无参考 DPO。对于缺乏高质量视频数据的场景是限制。
- **η 选择与内容相关**：引导级别 η 的最优值可能因基础模型能力和内容类型（动态程度）而异，当前使用统一的 [0.65, 0.8] 范围可能不够精细。
- **长视频 (>10s) 未验证**：随着视频长度增加，片段数量增多，标注和训练成本也线性增长，且片段间独立性假设可能不成立。
- **VLM 标注在复杂场景下可能退化**：遮挡、快速运动、细微差异等情况下 VLM 准确率可能不足。
- **LoRA 微调的天花板**：当前仅用 LoRA 微调，全量微调能否进一步释放潜力未知。

## 相关工作与启发

- **vs Diffusion-DPO (Wallace et al., 2023)**: 图像 DPO 的开创工作。直接迁移到视频时出现运动偏差——DenseDPO 是专门为视频设计的修正。核心区别在于数据构造和标注粒度。
- **vs VisionReward / VideoAlign (2024)**: 这些方法训练专用视频奖励模型来做 DPO，需要大规模标注。DenseDPO 用现成 VLM 免训练标注片段级偏好，更轻量。但专用奖励模型在复杂场景下可能更准确。
- **vs Sentence-level DPO (语言模型)**: 语言模型中已有在句子而非全文级别标注偏好的工作。DenseDPO 将这种细粒度偏好思想推广到视频的时间维度，是跨模态的自然对应。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 运动偏差诊断 + 引导式控制变量 + 片段级密集标注，三重创新相互配合
- 实验充分度: ⭐⭐⭐⭐⭐ 两套 benchmark、多维指标、完整消融、人工 vs VLM 标注对比
- 写作质量: ⭐⭐⭐⭐⭐ 从问题诊断到解决方案的逻辑链极为清晰，Spotlight 实至名归
- 价值: ⭐⭐⭐⭐⭐ 为视频 DPO 建立了正确范式，引导式控制变量思想可广泛迁移

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] LocalDPO: Direct Localized Detail Preference Optimization for Video Diffusion Models](../../CVPR2026/video_generation/mind_the_generative_details_direct_localized_detail_preference_optimization_for_.md)
- [\[ICCV 2025\] MotionAgent: Fine-grained Controllable Video Generation via Motion Field Agent](../../ICCV2025/video_generation/motionagent_fine-grained_controllable_video_generation_via_motion_field_agent.md)
- [\[ICCV 2025\] V.I.P.: Iterative Online Preference Distillation for Efficient Video Diffusion Models](../../ICCV2025/video_generation/vip_iterative_online_preference_distillation_for_efficient_video_diffusion_model.md)
- [\[CVPR 2026\] AlcheMinT: Fine-grained Temporal Control for Multi-Reference Consistent Video Generation](../../CVPR2026/video_generation/alchemint_fine-grained_temporal_control_for_multi-reference_consistent_video_gen.md)
- [\[AAAI 2026\] MotionCharacter: Fine-Grained Motion Controllable Human Video Generation](../../AAAI2026/video_generation/motioncharacter_fine-grained_motion_controllable_human_video_generation.md)

</div>

<!-- RELATED:END -->
