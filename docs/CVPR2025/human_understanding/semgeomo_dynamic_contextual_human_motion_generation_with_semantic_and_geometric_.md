---
title: >-
  [论文解读] SemGeoMo: Dynamic Contextual Human Motion Generation with Semantic and Geometric Guidance
description: >-
  [CVPR 2025][人体理解][动态上下文运动生成] 提出SemGeoMo，通过LLM自动标注器提供语义引导并结合affordance-level和joint-level的层级几何引导，在两阶段框架中实现动态上下文环境下的高质量人体交互运动生成，同时输出对应文本描述。 动态上下文运动生成旨在生成适应真实动态环境的人类交互…
tags:
  - "CVPR 2025"
  - "人体理解"
  - "动态上下文运动生成"
  - "语义几何引导"
  - "Affordance Map"
  - "LLM标注器"
  - "人物交互"
---

# SemGeoMo: Dynamic Contextual Human Motion Generation with Semantic and Geometric Guidance

**会议**: CVPR 2025  
**arXiv**: [2503.01291](https://arxiv.org/abs/2503.01291)  
**代码**: [项目主页](https://4dvlab.github.io/project_page/semgeomo/)  
**领域**: Human Understanding  
**关键词**: 动态上下文运动生成, 语义几何引导, Affordance Map, LLM标注器, 人物交互

## 一句话总结

提出SemGeoMo，通过LLM自动标注器提供语义引导并结合affordance-level和joint-level的层级几何引导，在两阶段框架中实现动态上下文环境下的高质量人体交互运动生成，同时输出对应文本描述。

## 研究背景与动机

动态上下文运动生成旨在生成适应真实动态环境的人类交互运动，对机器人交互、VR/AR等应用至关重要。现有方法存在两类局限：

1. **文本驱动的联合生成方法** — 同时生成人和物体的运动导致搜索空间过大，生成质量次优，且缺乏细粒度控制
2. **上下文运动生成方法** — 大多仅处理静态环境（固定家具），少数处理动态目标的方法（如OMOMO）缺乏文本语义引导，且未充分利用细粒度几何表示

核心挑战在于如何构建有效的语义引导（理解"应该如何交互"）和几何引导（确保接触准确、避免穿透），并将二者融合。

## 方法详解

### 整体框架

两阶段条件扩散框架：(1) SemGeo层级引导生成——在文本和点云条件下，用双分支Transformer联合生成affordance-level和joint-level交互线索；(2) SemGeo引导的运动生成——利用第一阶段的几何线索和语义信息引导详细人体运动生成（基于Motion ControlNet + MDM）。

### 关键设计1：LLM自动交互文本标注器

- **功能**: 从4D点云自动生成粗到细的交互文本描述，消除人工标注需求
- **核心思路**: 两步流程——(a) 从点云提取包围盒和运动轨迹，结合预定义动作/类别列表，用LoRA微调的LLaMA生成粗粒度描述；(b) 结合预测的手部关节位置计算接触信息，让LLM将交互分为三个阶段生成细粒度描述（如"左手接触箱子左下方"）
- **设计动机**: LLM具有交互过程的常识知识，可以推理出合理的交互方式。粗到细的标注策略逐步增加描述细粒度，使语义引导在不同层级都发挥作用

### 关键设计2：双分支Transformer层级几何引导生成

- **功能**: 联合生成affordance map和关节位置，捕获粗到细的交互几何线索
- **核心思路**: 用条件扩散模型在CLIP文本特征$F_{text}$和BPS点云特征$F_{pc}$引导下，通过JointTransformer和AffordanceTransformer两个分支并行生成。AffordanceTransformer用cross-attention建模affordance与点云几何的紧密关系，最后通过互交叉注意力将affordance信息反馈到关节位置分支进行refinement
- **设计动机**: 将接触几何生成与运动生成解耦，降低了单一模型的学习难度。affordance提供粗粒度"哪里接触"信息，关节位置提供精确空间定位，二者互补

### 关键设计3：SemGeo条件模块与Motion ControlNet

- **功能**: 有效融合多层级语义和几何条件引导全身运动生成
- **核心思路**: 用LongCLIP（支持长文本）提取细粒度文本特征$F'_{text}$，将点云特征与affordance map拼接经MLP+Temporal Transformer提取时空特征$F$，再用互交叉注意力融合关节和affordance特征。条件输入Motion ControlNet（冻结MDM权重），采样时用classifier guidance的关节损失$L_{joint}$和脚部稳定性损失$L_{foot}$进行refinement
- **设计动机**: ControlNet架构允许利用预训练MDM的运动先验，而多层级条件确保语义合理性和几何准确性的dual保证

### 损失函数

- 阶段一: $\mathcal{L} = \mathbb{E}_{x^0,t}\|\hat{x}_\theta(x^t,t,c) - x^0\|_1$（$L_1$重建损失）
- 阶段二采样引导: $L_{joint} = \frac{1}{J}\sum|J_{pred} - J'_h|_2 \cdot \text{Mask}$（接触关节约束）；$L_{foot}$惩罚脚部离地、滑动和加速度

## 实验关键数据

### 主实验：FullBodyManipulation数据集

| 方法 | HandJPE↓ | MPJPE↓ | $C_{prec}$↑ | $C_{rec}$↑ | FID↓ | R-score↑ |
|------|----------|--------|------------|----------|------|----------|
| SceneDiff | 95.38 | 19.84 | 0.64 | 0.19 | 1.64 | 0.59 |
| OMOMO | 33.18 | 18.06 | 0.77 | 0.71 | 1.98 | 0.38 |
| CHOIS | 31.68 | 17.12 | 0.76 | 0.58 | 2.27 | 0.49 |
| **SemGeoMo (GT text)** | **27.84** | **16.62** | **0.84** | **0.74** | **1.17** | **0.66** |

### 消融实验亮点

- 语义引导（文本描述）对接触准确性和运动质量都有显著贡献
- 细粒度LLM标注比粗粒度标注进一步提升性能
- 双分支联合生成优于分别生成affordance和关节位置

### 关键发现

- SemGeoMo在三个人物交互数据集上均达到SOTA
- 方法展示了对未见物体、人-人交互、可变形物体的泛化能力
- 同时生成运动和文本描述增强了交互的可解释性
- LLM标注器接近甚至匹配人工标注的效果

## 亮点与洞察

1. **语义+几何双引导的完整性**: 文本提供"应该做什么"的常识，affordance和关节位置提供"应该在哪做"的精确约束
2. **LLM作为交互常识的来源**: 利用LLM推理能力自动生成标注，既减轻人工成本又提供丰富语义
3. **粗到细的层级设计**: 从affordance到关节到全身运动的逐步细化，降低了生成难度

## 局限与展望

- 依赖多个预训练模型（LLaMA、CLIP、LongCLIP、MDM），系统复杂度高
- 当前主要在tabletop manipulation场景验证，大规模全身交互场景需更多数据
- LLM标注器的质量受限于微调数据的覆盖范围

## 相关工作与启发

- LLM作为自动标注器的思路可推广到其他缺乏文本标注的运动数据集
- 双分支Transformer联合生成多种中间表示的架构可用于其他多层级任务

## 评分

⭐⭐⭐⭐ — 框架设计清晰，多层级引导思路有条理。LLM标注器是实用创新。在三个数据集上一致SOTA证明了方法的有效性。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] MotionReFit: Dynamic Motion Blending for Versatile Motion Editing](motionrefit_motion_editing.md)
- [\[CVPR 2025\] D3-Human: Dynamic Disentangled Digital Human from Monocular Video](d3-human_dynamic_disentangled_digital_human_from_monocular_video.md)
- [\[CVPR 2025\] X-Dyna: Expressive Dynamic Human Image Animation](x-dyna_expressive_dynamic_human_image_animation.md)
- [\[ICCV 2025\] SemTalk: Holistic Co-speech Motion Generation with Frame-level Semantic Emphasis](../../ICCV2025/human_understanding/semtalk_holistic_co-speech_motion_generation_with_frame-level_semantic_emphasis.md)
- [\[CVPR 2025\] Lost in Translation, Found in Context: Sign Language Translation with Contextual Cues](lost_in_translation_found_in_context_sign_language_translation_with_contextual_c.md)

</div>

<!-- RELATED:END -->
