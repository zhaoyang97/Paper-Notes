---
title: >-
  [论文解读] VideoDPO: Omni-Preference Alignment for Video Diffusion Generation
description: >-
  [CVPR 2025][视频生成][视频扩散模型] VideoDPO 首次将 DPO（Direct Preference Optimization）适配到视频扩散模型，提出 OmniScore 综合评分体系同时衡量视觉质量和语义对齐，结合自动偏好数据生成 pipeline 和基于分数差异的数据重加权策略，在 VideoCrafter2、T2V-Turbo 和 CogVideoX 上均取得了显著的偏好对齐提升。
tags:
  - "CVPR 2025"
  - "视频生成"
  - "视频扩散模型"
  - "DPO偏好对齐"
  - "视频质量评估"
  - "数据重加权"
  - "文本到视频"
---

# VideoDPO: Omni-Preference Alignment for Video Diffusion Generation

**会议**: CVPR 2025  
**arXiv**: [2412.14167](https://arxiv.org/abs/2412.14167)  
**代码**: [https://videodpo.github.io/](https://videodpo.github.io/)  
**领域**: 扩散模型 / 视频生成  
**关键词**: 视频扩散模型, DPO偏好对齐, 视频质量评估, 数据重加权, 文本到视频

## 一句话总结

VideoDPO 首次将 DPO（Direct Preference Optimization）适配到视频扩散模型，提出 OmniScore 综合评分体系同时衡量视觉质量和语义对齐，结合自动偏好数据生成 pipeline 和基于分数差异的数据重加权策略，在 VideoCrafter2、T2V-Turbo 和 CogVideoX 上均取得了显著的偏好对齐提升。

## 研究背景与动机

**领域现状**：文本到视频（T2V）扩散模型发展迅速，VideoCrafter、Open-Sora、CogVideoX 等模型能够从文本prompt生成多样化的视频内容。DPO 作为偏好对齐方法已在 LLM 和图像生成领域取得显著成功（如 DiffusionDPO 用于文本到图像模型），但尚未被系统性地应用到视频扩散模型。

**现有痛点**：当前视频扩散模型生成的视频在两个维度上存在不足：（1）视觉质量—帧内清晰度不够、帧间运动不连贯、时域闪烁等；（2）语义对齐—生成内容与文本描述不匹配。这些问题部分源于大规模预训练数据中的低质量样本（低分辨率、模糊、文本-视频不匹配）。已有的视觉奖励模型通常只关注单一维度（质量或语义），实验表明这些维度之间相关性很低，只优化一个维度无法自动改善其他维度。

**核心矛盾**：视频质量是多维度的——帧内视觉质量、帧间时序一致性、文本语义对齐三者之间相关性低，需要综合考量。现有方法要么只用奖励模型做基于梯度的微调（如 VADER），要么只关注单一质量维度，无法实现全面的偏好对齐。

**本文目标**：（1）设计一个综合评分系统全面覆盖视频生成质量的各个维度；（2）建立自动化的偏好数据构建pipeline，避免昂贵的人工标注；（3）通过 DPO 训练有效提升视频模型的整体用户偏好。

**切入角度**：作者分析了视频质量各子维度之间的 Pearson 相关系数，发现帧内质量、帧间一致性和语义对齐之间相关性很低（如图3(d)），因此必须同时考虑所有维度。

**核心 idea**：提出 OmniScore 综合评分 + 自动 best-vs-worst 偏好对构建 + 基于频率分布的数据重加权，三者协同构成完整的视频 DPO 对齐方案。

## 方法详解

### 整体框架

VideoDPO pipeline 分三步：（1）对每个 prompt 生成 N 个视频，用 OmniScore 综合评分；（2）选取最高分和最低分视频作为偏好对（winning vs losing），构建偏好数据集；（3）计算所有视频的 OmniScore 频率直方图，对偏好对进行重加权后用 DPO 损失训练。

### 关键设计

1. **OmniScore 综合偏好评分**:

    - 功能：从多维度全面评估生成视频的质量
    - 核心思路：OmniScore 包含三大维度的子评分：（a）**帧内质量**——图像质量和美学吸引力，评估单帧的保真度和视觉美感；（b）**帧间质量**——主体一致性、背景一致性、时域闪烁、运动平滑度和运动动态程度，评估帧间的视觉连贯性；（c）**文本-视频语义对齐**——使用视觉-语言基础模型度量视频内容与文本 prompt 的匹配程度。各维度使用预训练的评分模型独立打分，然后综合为最终的 OmniScore。
    - 设计动机：实验分析（Pearson 相关热力图）表明不同质量维度间的相关性很低，意味着只优化单个维度（如仅优化美学分数）不仅不能自动提升其他维度，甚至可能造成退化。综合评分确保 DPO 训练不会"偏科"。

2. **基于分数排序的偏好数据自动生成**:

    - 功能：无需人工标注，自动构建高质量的偏好对数据集
    - 核心思路：对数据集中的 K=10,000 个人类书写的 prompt（来自 VidProm），每个 prompt 用待对齐的模型生成 N=4 个视频。对每个视频用 OmniScore 打分 $s_i = S(v_i, p)$，选择最高分视频为 winning sample $v^W$、最低分为 losing sample $v^L$，形成偏好对 $(v^W, v^L)$。
    - 设计动机：选择分数差距最大的 best vs worst 对能提供最清晰的偏好信号。使用模型自身生成的视频构建偏好对（on-policy），确保偏好数据的分布与模型当前能力相匹配。使用 VidProm 的人类书写 prompt 使模型更好地适配真实用户输入。

3. **OmniScore 驱动的数据重加权**:

    - 功能：让模型更多关注"区分度高"的偏好对，提升训练效率和效果
    - 核心思路：统计所有生成视频的 OmniScore 频率直方图 $p(\cdot)$，定义偏好对的采样概率为 $\text{prob}(s^W, s^L) = \sqrt{p(s^W) \cdot p(s^L)}$，然后计算重加权因子 $w_{\text{pair}} = (\beta / \text{prob}(s^W, s^L))^\alpha$，其中 $\beta$ 设为最高频样本的概率，$\alpha$ 控制加权强度（实验中 $\alpha=0.72$）。最终 DPO 损失乘以 $w_{\text{pair}}$。
    - 设计动机：直接使用 DPO 时，很多偏好对的分数差异极小，模型难以有效区分。低频出现的"典型偏好对"（如一个特别好 + 一个特别差）往往包含更多对齐信息，通过频率反加权增大这些样本的训练权重。当 $\alpha=0$ 时退化为标准 DPO。

### 损失函数 / 训练策略

最终训练损失为 $L_{\text{video}} = L_{\text{DPO}}(p, v^W, v^L) \cdot w_{\text{pair}}$，其中 DPO 损失采用 DiffusionDPO 的扩散模型适配形式。训练 3000 步，全局 batch size 8，AdamW 优化器 lr=6e-6，4 张 A100 GPU。频率直方图 bin 宽度为 0.01。

## 实验关键数据

### 主实验

VBench 综合评测（Total = Quality + Semantics 加权）：

| 模型 | 方法 | VBench Total↑ | Quality↑ | Semantics↑ | HPS(V)↑ | PickScore↑ |
|------|------|---------------|----------|-----------|---------|-----------|
| VC2 | Baseline | 80.44 | 82.20 | 73.42 | 0.258 | 20.65 |
| VC2 | VADER | 80.59 | 82.46 | 73.09 | 0.259 | 20.62 |
| VC2 | **VideoDPO** | **81.93** | **83.07** | **77.38** | **0.261** | 20.65 |
| Turbo | Baseline | 80.95 | 82.71 | 73.93 | 0.262 | 21.15 |
| Turbo | **VideoDPO** | **81.80** | **83.80** | **73.81** | 0.260 | **21.18** |
| CogVid | Baseline | 79.30 | 82.35 | 67.10 | - | 19.81 |
| CogVid | **VideoDPO** | **79.80** | **83.00** | 66.99 | - | 19.79 |

### 消融实验

数据重加权策略消融（以 VC2 为基线，VBench Total）：

| 配置 | VBench Total | 说明 |
|------|-------------|------|
| Baseline（无 DPO）| 80.44 | 原始模型 |
| DPO w/o 重加权 ($\alpha=0$) | 81.11 | 标准 DPO 已有提升 |
| DPO + 重加权 ($\alpha=0.72$) | **81.93** | 重加权进一步提升 0.82 |
| SFT（监督微调）| 78.78 | 纯 SFT 反而下降 |

偏好对选择策略消融显示 best-vs-worst 策略优于随机配对和相邻排名配对。

### 关键发现

- VideoDPO 在三个不同架构的模型（UNet-based VC2、蒸馏模型 T2V-Turbo、DiT-based CogVideoX）上均有效，证明了方法的通用性
- 语义对齐维度的提升尤为显著（VC2 上从 73.42 到 77.38，提升近 4 个百分点），说明 DPO 对语义理解的校准效果突出
- VBench 16 个子维度的详细分析显示，Multiple Objects（+11.63）、Spatial Relationship（+12.85）和 Scene（+15.78）提升最大，这些恰好是 T2V 模型最薄弱的方面
- SFT 直接用 winning 样本微调反而导致 VBench Total 下降（80.44→78.78），证明偏好对比学习比直接拟合更有效
- 数据重加权的收益显著且稳定（$\alpha=0.72$ 为最优）

## 亮点与洞察

- **多维度偏好评分（OmniScore）的设计思路**：不同质量维度之间的低相关性分析非常有说服力，证明了"一个分数管所有"的必要性。这种思路可以推广到任何多属性优化问题中，如图像编辑、3D 生成等
- **频率反加权策略简单有效**：利用 OmniScore 分布直方图来识别"信息量大"的偏好对，本质上是一种难例挖掘（hard example mining），但从统计分布角度出发更加优雅
- **首次系统性地将 DPO 适配到视频扩散**：虽然概念上不复杂，但需要解决视频偏好数据构建、多维度评分、扩散模型 DPO 损失适配等多个工程挑战

## 局限与展望

- OmniScore 的各维度权重目前是固定的，不同用户和应用场景对各维度的重视程度可能不同，未来可探索自适应权重调整
- 偏好数据完全由自动评分生成（RLAIF），没有真实的人类偏好标注验证，评分模型本身的偏差会传递到对齐结果中
- 实验中每个 prompt 仅生成 4 个视频，偏好对的多样性有限，更大的采样数可能带来更好的效果但会增加计算成本
- CogVideoX 上的提升相对较小（79.30→79.80），DiT 架构可能需要不同的 DPO 超参数调优
- Motion dynamics 和 dynamic degree 等指标在 DPO 后有所下降（VC2 上 motion smoothness 从 97.73 降到 92.18），说明对齐可能牺牲了部分运动丰富度来换取整体质量，这是一个值得关注的 trade-off

## 相关工作与启发

- **vs DiffusionDPO**: DiffusionDPO 是图像领域的 DPO 适配，只关注图像质量或单一语义维度；VideoDPO 引入了 OmniScore 综合评分和数据重加权，更适合视频的多维度质量要求
- **vs VADER**: VADER 使用可微奖励模型直接对扩散模型最后几步做梯度优化；VideoDPO 采用 DPO 的偏好对比学习范式，不需要可微奖励模型，训练更稳定
- **vs T2V-Turbo v2**: T2V-Turbo v2 也探索了奖励梯度用于一致性蒸馏模型的优化，但本质上是优化单一奖励；VideoDPO 的偏好对比学习更适合多维度偏好的综合优化

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统性地将 DPO 适配到视频扩散模型，OmniScore 和重加权策略有一定创新但不算突破性
- 实验充分度: ⭐⭐⭐⭐⭐ 三个不同模型、多个评测指标、详细的子维度分析、充分的消融实验
- 写作质量: ⭐⭐⭐⭐ 整体结构清晰，OmniScore 维度间相关性分析有说服力，但部分技术细节描述较分散
- 价值: ⭐⭐⭐⭐ 为视频生成模型的后训练提供了实用的偏好对齐方案，OmniScore 评分体系本身也有独立价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] OmniVDiff: Omni Controllable Video Diffusion for Generation and Understanding](../../AAAI2026/video_generation/omnivdiff_omni_controllable_video_diffusion_for_generation_and_understanding.md)
- [\[CVPR 2025\] Can Text-to-Video Generation Help Video-Language Alignment?](can_text-to-video_generation_help_video-language_alignment.md)
- [\[ICCV 2025\] V.I.P.: Iterative Online Preference Distillation for Efficient Video Diffusion Models](../../ICCV2025/video_generation/vip_iterative_online_preference_distillation_for_efficient_video_diffusion_model.md)
- [\[ICCV 2025\] Prompt-A-Video: Prompt Your Video Diffusion Model via Preference-Aligned LLM](../../ICCV2025/video_generation/prompt-a-video_prompt_your_video_diffusion_model_via_preference-aligned_llm.md)
- [\[CVPR 2025\] Articulated Kinematics Distillation from Video Diffusion Models](articulated_kinematics_distillation_from_video_diffusion_models.md)

</div>

<!-- RELATED:END -->
