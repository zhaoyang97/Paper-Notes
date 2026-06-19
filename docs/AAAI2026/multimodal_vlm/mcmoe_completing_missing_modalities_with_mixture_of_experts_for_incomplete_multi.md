---
title: >-
  [论文解读] MCMoE: Completing Missing Modalities with Mixture of Experts for Incomplete Multimodal Action Quality Assessment
description: >-
  [AAAI 2026][多模态VLM][不完整多模态学习] 本文首次探索不完整多模态动作质量评估问题，提出 MCMoE 框架，利用自适应门控模态生成器（AGMG）补全缺失模态，并通过混合专家（MoE）动态融合单模态和跨模态联合表示，在单阶段训练中统一学习，在三个公开 AQA 基准上的完整和不完整场景中均达到 SOTA，且参数量仅 4.90M。
tags:
  - "AAAI 2026"
  - "多模态VLM"
  - "不完整多模态学习"
  - "动作质量评估"
  - "混合专家"
  - "模态补全"
  - "跨模态融合"
---

# MCMoE: Completing Missing Modalities with Mixture of Experts for Incomplete Multimodal Action Quality Assessment

**会议**: AAAI 2026  
**arXiv**: [2511.17397](https://arxiv.org/abs/2511.17397)  
**代码**: [https://github.com/XuHuangbiao/MCMoE](https://github.com/XuHuangbiao/MCMoE)  
**领域**: 多模态VLM  
**关键词**: 不完整多模态学习, 动作质量评估, 混合专家, 模态补全, 跨模态融合

## 一句话总结

本文首次探索不完整多模态动作质量评估问题，提出 MCMoE 框架，利用自适应门控模态生成器（AGMG）补全缺失模态，并通过混合专家（MoE）动态融合单模态和跨模态联合表示，在单阶段训练中统一学习，在三个公开 AQA 基准上的完整和不完整场景中均达到 SOTA，且参数量仅 4.90M。

## 研究背景与动机

多模态动作质量评估（AQA）通过融合 RGB、光流、音频等互补信息，能够对高度相似的动作序列进行细粒度判别。然而，现实推理场景中常因传感器故障、环境限制或隐私问题导致部分模态不可用。现有多模态模型假设训练和推理时所有模态均可用，一旦缺失任何模态就无法正常工作，甚至产生灾难性性能退化。

现有解决方案分为两类：（1）模态补全方法——用扩散模型、VAE、GAN 等重型生成器重建缺失数据，计算代价高；（2）跨模态联合表示学习——提取联合特征但通常采用两阶段训练流水线，增加训练复杂度。

核心矛盾在于：如何在不依赖重型生成器的前提下，高效地在单阶段训练中同时学习单模态特有知识和跨模态共享知识，从而补偿缺失模态带来的干扰。

本文的核心 idea：利用模态补全（MMC）和混合专家（MoE）之间的互补性——MMC 为 MoE 提供高置信度特征以支撑单模态和联合学习，MoE 反过来动态精炼和修补 MMC 生成的特征，减少对高保真重型生成器的依赖。

## 方法详解

### 整体框架

输入：RGB（v）、光流（f）、音频（a）三种模态的视频特征（由冻结的预训练提取器 VST/I3D/AST 提取）。训练时所有模态可见，通过随机掩码模拟不完整场景；推理时缺失模态以零向量初始化。

Pipeline：特征提取 → 共享时序增强模块（STEM）→ 自适应门控模态生成器（AGMG）补全缺失模态 → 混合专家（MoE）+ 软路由器动态融合 → 跨模态融合模块（CFM）→ 基于等级的回归网络 → 分数预测。

### 关键设计

1. **共享时序增强模块 (STEM)**:

    - 功能：将不同模态特征映射到共享潜在空间，弥合跨模态语义鸿沟
    - 核心思路：使用可堆叠的 3 层 Transformer 编码器（2 头自注意力）处理各模态特征，参数共享以捕获跨模态共性
    - 设计动机：避免直接多模态特征交互，为后续单模态学习铺路，同时增强时序上下文建模

2. **自适应门控模态生成器 (AGMG)**:

    - 功能：从可用模态动态生成缺失模态的特征表示
    - 核心思路：采用 2 层 4 头交叉注意力机制——可用模态拼接作为 Key/Value，零初始化的缺失模态作为 Query，经 L 层迭代精炼。然后通过门控层（gating layer）动态加权融合，根据当前输入的完整程度调节融合权重
    - 设计动机：（1）交叉注意力利用可用模态信息逐步补全缺失模态；（2）门控层抑制不完美生成带来的误差传播；（3）相比重型 GAN/扩散模型，该设计轻量高效

3. **混合专家 (MoE) + 软路由器**:

    - 功能：为每种模态设计专用的单模态专家（轻量 2 层 MLP），动态混合所有专家知识提取跨模态联合表示
    - 核心思路：对于模态 v 的特征，分别由视觉专家、光流专家、音频专家处理；软路由器（2 层 MLP + Softmax）根据输入动态估计各专家的重要性权重，加权求和得到最终特征
    - 设计动机：（1）每个专家挖掘特定模态知识，共同协作捕获跨模态模式；（2）软路由器适应多种缺失模态组合；（3）对齐损失迫使联合表示与单模态表示对齐，实现单阶段统一学习

4. **跨模态融合模块 (CFM)**:

    - 功能：捕获模态间相关性，映射到任务特定潜在空间
    - 核心思路：简单的两层卷积块即可有效融合，因为 MoE 处理后的特征已包含丰富的跨模态语义
    - 设计动机：利用 MoE 已有的跨模态信息，避免复杂的融合架构

### 损失函数 / 训练策略

四个损失函数的加权和：
- **重建损失 $\mathcal{L}_{recon}$**：MSE，监督 AGMG 生成高保真特征
- **对齐损失 $\mathcal{L}_{align}$**：KL 散度，对齐联合表示和单模态表示，驱动单阶段统一学习
- **多样性损失 $\mathcal{L}_{div}$**：三元组损失，确保不同等级模式关注不同动作表现
- **任务损失 $\mathcal{L}_{task}$**：MSE，回归预测分数

训练时随机掩码模拟不完整场景，使用 Adam 优化器 + 余弦退火学习率调度。

## 实验关键数据

### 主实验（不完整模态场景平均性能）

| 数据集 | 指标 | MCMoE | 之前SOTA | 提升 |
|--------|------|-------|----------|------|
| FS1000 | SP.Corr./MSE | 0.782/15.37 | 0.668/26.08 (MoMKE) | +17.1%/↓38.0% |
| Fis-V | SP.Corr./MSE | 0.734/17.02 | 0.660/21.63 (MoMKE) | +11.2%/↓21.3% |
| RG | SP.Corr./MSE | 0.697/7.89 | 0.623/9.13 (MoMKE) | +11.9%/↓11.5% |

### 消融实验

| 配置 | RG 平均 | RG 完整 | 说明 |
|------|---------|---------|------|
| Baseline | 0.532/25.79 | 0.718/7.05 | 直接求和 |
| + STEM | 0.573/19.92 | 0.744/6.83 | 时序增强有效 |
| + AGMG | 0.647/9.45 | 0.779/6.04 | 模态补全关键 |
| + MoE | 0.675/7.91 | 0.817/5.30 | 专家融合进一步提升 |
| + CFM (完整) | 0.697/7.89 | 0.842/4.85 | 跨模态融合锦上添花 |
| w/o AGMG | 0.635/16.54 | 0.724/6.44 | 移除AGMG性能大降 |
| w/o MoE | 0.658/7.97 | 0.777/5.48 | 移除MoE性能降低 |

### 关键发现
- AGMG 是性能提升的最大贡献者，说明动态模态补全对弥合缺失数据语义鸿沟至关重要
- MMC 和 MoE 存在显著互补性：MMC 为 MoE 提供高置信度输入，MoE 为 MMC 提供动态精炼
- 三个损失函数各有作用，移除任何一个都导致性能下降，重建损失影响最大
- 等级数 N=4 时最优；MCMoE 仅 4.90M 参数 / 1.34 GFLOPs，远低于同期方法
- t-SNE 可视化清晰说明 MCMoE 能有效区分不同动作质量等级

## 亮点与洞察

- **MMC-MoE 互补设计**：核心 insight，两组件互相增强。MMC 降低 MoE 处理零矩阵的难度，MoE 减轻对高保真生成的要求，形成良性循环
- **单阶段统一学习**：通过对齐损失巧妙绕过两阶段训练的复杂性
- **极致轻量**：仅 4.90M 参数和 1.34 GFLOPs，远低于同期竞争方法（GCNet 8.78M/1191G）
- **AGMG 的门控设计**：门控层抑制不完美生成的误差传播，比直接使用生成特征更鲁棒
- **可扩展性**：框架可灵活扩展到任意数量的模态

## 局限与展望

- 仅在三种模态（RGB、光流、音频）上验证，未探索文本、骨架等更多模态
- AQA 任务特定，不确定能否直接泛化到其他不完整多模态任务
- AGMG 生成质量在极端稀少模态（仅 audio）时仍有差距
- 未探索训练时也存在模态缺失的场景

## 相关工作与启发

- **vs MoMKE**: MoMKE 是两阶段方法，MCMoE 单阶段统一学习且参数更少
- **vs ActionMAE**: ActionMAE 用 MAE 重建缺失模态，单阶段但无法充分利用跨模态融合
- **vs PAMFN**: PAMFN 在完整模态下最强，但无法处理不完整场景

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次探索不完整多模态 AQA，MMC-MoE 互补设计新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集、全面的不完整组合、丰富消融和可视化
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，图表丰富，公式推导完整
- 价值: ⭐⭐⭐⭐ 解决真实场景中普遍存在的模态缺失问题，轻量高效具有实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] BriMA: Bridged Modality Adaptation for Multi-Modal Continual Action Quality Assessment](../../CVPR2026/multimodal_vlm/brima_bridged_modality_adaptation_for_multi-modal_continual_action_quality_asses.md)
- [\[ICCV 2025\] A Quality-Guided Mixture of Score-Fusion Experts Framework for Human Recognition](../../ICCV2025/multimodal_vlm/a_qualityguided_mixture_of_scorefusion_experts_framework_for.md)
- [\[ICML 2026\] Calibrated Multimodal Representation Learning with Missing Modalities](../../ICML2026/multimodal_vlm/calibrated_multimodal_representation_learning_with_missing_modalities.md)
- [\[ICML 2026\] Toward Structural Multimodal Representations: Specialization, Selection, and Sparsification via Mixture-of-Experts](../../ICML2026/multimodal_vlm/toward_structural_multimodal_representations_specialization_selection_and_sparsi.md)
- [\[ICLR 2026\] VisJudge-Bench: Aesthetics and Quality Assessment of Visualizations](../../ICLR2026/multimodal_vlm/visjudge-bench_aesthetics_and_quality_assessment_of_visualizations.md)

</div>

<!-- RELATED:END -->
