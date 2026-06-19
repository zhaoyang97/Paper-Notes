---
title: >-
  [论文解读] Multi-Group Proportional Representation for Text-to-Image Models
description: >-
  [CVPR 2025][图像生成][文本到图像] 本文提出Multi-Group Proportional Representation (MPR)指标，用于系统化度量文本到图像模型中交叉人口群体的代表性偏差，并开发了基于该指标的优化算法，在保持生成质量的前提下引导T2I模型向更均衡的群体代表性方向调整。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "文本到图像"
  - "公平性"
  - "比例代表性"
  - "交叉群体"
  - "偏见缓解"
---

# Multi-Group Proportional Representation for Text-to-Image Models

**会议**: CVPR 2025  
**arXiv**: [2505.24023](https://arxiv.org/abs/2505.24023)  
**代码**: 无  
**领域**: 扩散模型 / AI安全  
**关键词**: 文本到图像, 公平性, 比例代表性, 交叉群体, 偏见缓解

## 一句话总结
本文提出Multi-Group Proportional Representation (MPR)指标，用于系统化度量文本到图像模型中交叉人口群体的代表性偏差，并开发了基于该指标的优化算法，在保持生成质量的前提下引导T2I模型向更均衡的群体代表性方向调整。

## 研究背景与动机

**领域现状**：文本到图像(T2I)生成模型（如Stable Diffusion、DALL-E、Midjourney等）已能从文本描述生成逼真图像，被广泛应用于内容创作、广告设计等场景。随着这些模型的大规模部署，其社会影响日益受到关注。

**现有痛点**：T2I模型在生成涉及人物的图像时，会系统性地放大训练数据中的人口统计偏见——过度代表某些群体（如年轻白人男性）而忽视少数群体（如老年女性、特定种族交叉群体）。这种代表性偏差可能传播刻板印象、边缘化少数人群。尽管"负责任AI"受到广泛关注，但目前缺乏**系统化方法论**来(1)量化T2I模型中交叉群体的代表性偏差，(2)在训练中优化公平性。

**核心矛盾**：现有公平性指标通常只关注单一人口属性（如仅性别或仅种族），忽略了**交叉性**问题。例如，一个模型可能在性别分布上均衡，在种族分布上也均衡，但在"非裔女性"等交叉群体上严重不足。单属性公平性指标无法捕获这种交叉偏差。

**本文目标**：(1) 定义一个考虑交叉群体的、灵活可配置的代表性度量指标；(2) 开发可优化该指标的训练算法；(3) 验证该方法能在保持图像质量的前提下改善代表性。

**切入角度**：借鉴计算社会选择理论中的"比例代表性"概念，将其适配到T2I模型评估场景。不需要强制所有群体等比例出现（这可能不符合现实），而是允许用户根据具体场景指定目标分布。

**核心idea**：用"最差情况群体偏差"（worst-case group deviation）作为代表性度量的核心，确保没有任何交叉群体被严重欠代表。

## 方法详解

### 整体框架
MPR框架包含两个组件：(1) MPR度量指标——给定T2I模型生成的图像集合，计算所有预定义交叉群体的代表性与目标分布之间的最大偏差；(2) MPR优化算法——将MPR指标作为训练目标，通过微调T2I模型的文本条件模块来最小化最差情况偏差。工作流程为：生成一批图像→用属性分类器标注群体属性→计算MPR→反传梯度更新模型。

### 关键设计

1. **MPR度量指标 (Multi-Group Proportional Representation Metric)**:

    - 功能：量化T2I模型对交叉人口群体的代表性偏差
    - 核心思路：定义人口群体空间为多个属性（如性别、种族、年龄）的笛卡尔积，形成交叉群体集合$\mathcal{G}$（如"年轻黑人女性"、"老年亚裔男性"等）。对于每个群体$g \in \mathcal{G}$，计算其在生成图像中的实际比例$\hat{p}(g)$与目标比例$p^*(g)$的偏差。MPR定义为所有群体中最大的相对偏差：$\text{MPR} = \max_{g \in \mathcal{G}} |\hat{p}(g) - p^*(g)| / p^*(g)$。目标比例$p^*(g)$可以由用户根据应用场景指定（如人口普查数据、均匀分布等）
    - 设计动机：取最差情况(worst-case)设计确保不会有任何群体被遗忘——即使平均代表性看起来不错，只要有一个交叉群体被严重欠代表，MPR值就会很高。这比平均偏差更能反映公平性问题

2. **属性分类与群体标注 (Attribute Classification Pipeline)**:

    - 功能：自动标注生成图像中人物的人口统计属性
    - 核心思路：使用预训练的人脸检测和属性分类模型，从生成的图像中检测人脸并预测其性别、感知种族、年龄段等属性。这些属性预测被组合成交叉群体标签。为处理分类器的不确定性，采用软标签（概率分布）而非硬标签，使后续的MPR计算更鲁棒
    - 设计动机：手动标注大量生成图像不可行，需要自动化pipeline。使用软标签可以缓解属性分类器本身偏差对MPR计算的影响

3. **MPR优化算法 (MPR Optimization)**:

    - 功能：微调T2I模型使其生成更均衡的群体代表性
    - 核心思路：将MPR作为可微分的训练损失函数。在每个训练步中：(a)用当前模型生成一批图像；(b)通过属性分类器获取群体分布估计；(c)计算MPR损失；(d)反传梯度更新模型的文本编码器或cross-attention层。为保持生成质量，同时保留原始扩散损失并加权。关键技巧是对max操作进行soft relaxation（使用log-sum-exp近似），使梯度可以流向多个偏差较大的群体
    - 设计动机：直接优化最差情况偏差比优化平均偏差更能确保公平性改善。soft relaxation避免了max操作的梯度消失问题，使训练过程更稳定

### 损失函数 / 训练策略
总训练损失为：$\mathcal{L} = \mathcal{L}_{diff} + \alpha \cdot \mathcal{L}_{MPR}$，其中$\mathcal{L}_{diff}$为标准扩散去噪损失保持生成质量，$\mathcal{L}_{MPR}$为MPR损失推动代表性均衡。仅微调text encoder和cross-attention参数，冻结UNet主干以保持视觉生成能力。

## 实验关键数据

### 主实验

| 模型 | 优化前MPR↓ | 优化后MPR↓ | FID变化 | CLIP Score变化 |
|------|----------|----------|--------|--------------|
| Stable Diffusion v1.5 | 0.72 | 0.31 | +1.2 | -0.3 |
| Stable Diffusion v2.1 | 0.68 | 0.28 | +0.9 | -0.2 |
| SDXL | 0.61 | 0.24 | +1.5 | -0.4 |

### 消融/分析实验

| 分析维度 | 配置 | MPR↓ | 说明 |
|---------|------|------|------|
| 目标分布选择 | 均匀分布 | 0.28 | 所有群体等比例 |
| 目标分布选择 | 人口普查分布 | 0.22 | 按真实人口比例 |
| 群体粒度 | 单属性(仅性别) | 0.15 | 粗粒度更容易优化 |
| 群体粒度 | 二属性(性别+种族) | 0.28 | 交叉群体更具挑战 |
| 群体粒度 | 三属性(性别+种族+年龄) | 0.41 | 群体越多、优化越难 |
| soft vs hard max | Soft relaxation | 0.28 | 更稳定的训练 |
| soft vs hard max | Hard max | 0.35 | 梯度不稳定 |

### 关键发现
- MPR优化可以将代表性偏差降低55-60%，同时FID仅增加1-2点、CLIP Score几乎不变，说明公平性改善与生成质量可以基本兼顾
- 交叉群体的偏差远大于单属性偏差：即使性别和种族各自接近均衡，交叉群体（如"老年非裔女性"）的代表性仍可能非常低
- 群体粒度越细（考虑更多属性的交叉），优化难度越大，但MPR框架仍能带来显著改善
- 人口普查分布比均匀分布更容易优化（MPR从0.22 vs 0.28），因为前者更接近训练数据的自然分布
- Soft max relaxation对训练稳定性至关重要：hard max版本的MPR损失导致梯度集中在单一群体上，训练振荡严重

## 亮点与洞察
- **将社会选择理论引入T2I公平性**是一个有洞见的跨领域迁移：比例代表性的概念天然适合量化"谁被代表了、谁被忽视了"这一问题，比简单的统计距离更有解释性
- **最差情况设计**是核心亮点：现实中最严重的伤害往往发生在最被忽视的群体，MPR的max设计确保了对这些群体的关注，比平均指标更能反映公平性
- **框架的灵活性**值得强调：MPR允许用户自定义目标分布和群体划分，使其能适应不同文化和法律背景的公平性需求，不是一刀切的方案

## 局限与展望
- 属性分类器本身存在偏差（尤其对交叉群体的识别准确率较低），这会影响MPR计算的准确性，形成系统性误差
- 仅关注视觉层面的代表性（面部属性），不涉及文本描述中的刻板印象（如"doctor" vs "nurse"的性别联想）
- 实验聚焦于面部属性（性别、种族、年龄），未涉及其他受保护属性（残障、体型等）
- 优化后的模型在特定prompts上可能出现不自然的"强制多样性"效果，用户体验需要进一步评估
- MPR优化需要多次生成+分类的循环，计算开销较大，不适合实时部署

## 相关工作与启发
- **vs Fair Diffusion / Inclusive T2I**: 之前的公平性工作通常只考虑单一属性或使用简单的重采样策略。MPR首次系统化地处理了交叉群体公平性
- **vs DALL-E 3 system card**: 商业模型通过系统提示词注入来增加多样性（如自动添加种族描述词），这是一种hack而非根本解决方案。MPR提供了可量化、可优化的框架
- **vs Demographic Parity / Equal Opportunity**: 这些是分类模型的经典公平性指标，MPR将类似理念适配到了生成模型场景

## 评分
- 新颖性: ⭐⭐⭐⭐ 将比例代表性概念系统化地引入T2I公平性度量，交叉群体关注度很好
- 实验充分度: ⭐⭐⭐⭐ 多个SD模型、多种配置的实验，分析维度全面
- 写作质量: ⭐⭐⭐⭐ 形式化定义清晰，motivation阐述充分
- 价值: ⭐⭐⭐⭐ 为T2I模型公平性评估和优化提供了标准化框架，有实际应用意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Generative Modeling of Class Probability for Multi-Modal Representation Learning](generative_modeling_of_class_probability_for_multi_modal_representation_learning.md)
- [\[CVPR 2025\] MCCD: Multi-Agent Collaboration-based Compositional Diffusion for Complex Text-to-Image Generation](mccd_multi-agent_collaboration-based_compositional_diffusion_for_complex_text-to.md)
- [\[CVPR 2025\] CLIP Under the Microscope: A Fine-Grained Analysis of Multi-Object Representation](clip_under_the_microscope_a_fine-grained_analysis_of_multi-object_representation.md)
- [\[NeurIPS 2025\] FairImagen: Post-Processing for Bias Mitigation in Text-to-Image Models](../../NeurIPS2025/image_generation/fairimagen_post-processing_for_bias_mitigation_in_text-to-image_models.md)
- [\[CVPR 2025\] Compass Control: Multi Object Orientation Control for Text-to-Image Generation](compass_control_multi_object_orientation_control_for_text-to-image_generation.md)

</div>

<!-- RELATED:END -->
