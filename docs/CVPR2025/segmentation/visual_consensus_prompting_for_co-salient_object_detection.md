---
title: >-
  [论文解读] Visual Consensus Prompting for Co-Salient Object Detection
description: >-
  [CVPR 2025][语义分割][共显著物体检测] 本文首次将参数高效的提示学习范式引入共显著物体检测（CoSOD）任务，提出视觉共识提示（VCP），通过将共识提取与分散过程嵌入可学习的提示中，在冻结基础模型的条件下以极少可训练参数超越 13 个全参数微调方法。 领域现状：共显著物体检测（CoSOD）旨在从一组相关图像中检…
tags:
  - "CVPR 2025"
  - "语义分割"
  - "共显著物体检测"
  - "视觉提示学习"
  - "参数高效微调"
  - "共识提示"
  - "Transformer"
---

# Visual Consensus Prompting for Co-Salient Object Detection

**会议**: CVPR 2025  
**arXiv**: [2504.14254](https://arxiv.org/abs/2504.14254)  
**代码**: [https://github.com/WJ-CV/VCP](https://github.com/WJ-CV/VCP)  
**领域**: 分割 / 显著性检测  
**关键词**: 共显著物体检测, 视觉提示学习, 参数高效微调, 共识提示, Transformer

## 一句话总结

本文首次将参数高效的提示学习范式引入共显著物体检测（CoSOD）任务，提出视觉共识提示（VCP），通过将共识提取与分散过程嵌入可学习的提示中，在冻结基础模型的条件下以极少可训练参数超越 13 个全参数微调方法。

## 研究背景与动机

**领域现状**：共显著物体检测（CoSOD）旨在从一组相关图像中检测共同出现的显著物体。现有方法普遍采用三阶段架构：(1) 编码多尺度特征；(2) 共识提取与分散；(3) 预测输出。训练方式均为全参数微调。

**现有痛点**：(1) 架构层面——编码阶段和共识提取是分离的。编码器的特征被用来提取共识，但精心提取的共识无法及时反馈指导编码过程。编码器只在整个训练结束时才通过梯度回传间接调整，缺乏编码与共识之间的有效交互。(2) 训练范式层面——全参数微调所有参数（包括大型预训练模型）参数效率低下、计算和存储开销大。而且 CoSOD 数据集规模有限，全量微调可能损害预训练模型中的通用知识表示。

**核心矛盾**：CoSOD 任务的核心在于"共识"，但现有架构将共识提取和特征编码分离处理，无法实现两者的有效交互。同时，全微调范式随着基础模型规模的增大变得越来越不实际。

**本文目标**：设计一种交互有效且参数高效的 CoSOD 架构，将共识信息融入提示中实现编码与共识的深度交互，同时仅需极少可训练参数。

**切入角度**：作者发现，如果将共识提取和分散过程嵌入可学习的视觉提示中，则 CoSOD 需要的"编码与共识高效交互"天然对应提示学习范式中的"提示引导冻结模型"。提示在每一层 Transformer 中与特征交互，恰好实现了共识对编码的实时指导。

**核心 idea**：将共识表示构建为任务特定的视觉提示，通过共识提示生成器（CPG）从冻结嵌入中挖掘共识、共识提示分散器（CPD）将共识注入各层 Transformer block，使冻结模型被"唤醒"执行 CoSOD 任务。

## 方法详解

### 整体框架

基础模型采用 SegFormer（ImageNet 预训练），冻结全部参数。输入一组相关图像经 SegFormer 得到四个尺度的嵌入特征。CPG 从这些冻结嵌入中挖掘共显著物体表示，生成共识提示 $P_{Co}$。CPD 利用 $P_{Co}$ 结合嵌入提示 $P_{Em}$ 和手工提示 $P_{Hand}$，形成视觉共识提示 $P_{Visual}^{Co}$，注入冻结 Transformer 各层。最后通过简化的预测头生成共显著预测图。

### 关键设计

1. **共识提示生成器（CPG）**:

    - 功能：从冻结嵌入特征中挖掘组内共显著物体表示，生成共识提示
    - 核心思路：分三步走。(a) **显著性估计**：预定义 $j$ 个可学习显著性种子，通过与嵌入特征的残差计算和软分配概率进行聚类，得到更新后的显著性种子表示。利用更新后的种子与嵌入特征交互生成显著性估计图 $M^s$：$M^s = conv[MLP(L_2(S_{seed}^{update})), P_{em}]$。(b) **共识种子选择**：利用显著性估计图过滤非显著区域，将剩余的像素嵌入作为共识种子 $Co_{seed}$。通过计算每个种子与组内平均显著特征的相关性得分，选取 top-k 个最具代表性的种子 $Co_{seed}^{rep}$。(c) **共识提示生成**：将代表性种子作为动态卷积核映射回原始嵌入空间，再通过空间注意力增强，得到共识提示 $P_{Co}$
    - 设计动机：与之前利用额外显著性检测数据集进行辅助训练不同，CPG 通过原型学习的方式在嵌入空间中"发现"显著性种子。先过滤非显著区域再提取共识的两步策略，有效排除了背景干扰

2. **共识提示分散器（CPD）**:

    - 功能：将共识提示融入各层 Transformer block，生成任务特定的视觉共识提示以引导冻结模型
    - 核心思路：CPD 整合三类提示——共识提示 $P_{Co}$（来自 CPG 的组内共识信息）、嵌入提示 $P_{Em}$（冻结嵌入特征的降维版本）、手工提示 $P_{Hand}$（通过快速傅里叶变换生成的传统视觉特征）。三者融合形成视觉共识提示 $P_{Visual}^{Co}$，以自适应方式注入冻结 Transformer 的不同深度层。通过降维因子 $r$ 控制可训练参数量（$C_r = C_s / r$），实现参数效率
    - 设计动机：仅用简单可训练参数作为视觉提示（如 EVP）无法建模组内共识关系，在 CoSOD 任务上表现很差。将共识信息注入提示后，提示不再是通用的前景线索，而是任务特定的共显著性线索

3. **多尺度显著性监督与简化预测头**:

    - 功能：通过多阶段监督确保显著性估计的准确性，并简化解码过程
    - 核心思路：对 SegFormer 四个阶段的显著性估计图 $\{M^s\}_{s=1}^4$ 都施加 CoSOD 标签的监督，保持显著性估计与共识注意力目标的一致性。简化 SegFormer 原始解码器为轻量预测头并集成分类器
    - 设计动机：多尺度监督为 CPG 的显著性种子学习提供了更直接的梯度信号，避免了仅靠最终输出回传导致的梯度弥散

### 损失函数 / 训练策略

使用 CoSOD 标准标签对多尺度显著性估计图和最终预测图进行监督。冻结 SegFormer 全部参数，仅训练 CPG、CPD 及预测头中的可调参数。训练数据集包括 COCO-9k 和 DUT-class 的各种组合。

## 实验关键数据

### 主实验

三个基准数据集上与 13 个 SOTA 方法的对比（训练集：S+D，即 COCO-SEG + DUT-class）：

| 数据集 | 指标 | SCED (ACMM23) | CONDA (ECCV24) | **VCP (Ours)** | 提升 |
|--------|------|--------------|----------------|----------------|------|
| CoCA | $S_m$↑ | 0.741 | 0.763 | **0.819** | +5.6% |
| CoCA | $F_m$↑ | 0.610 | 0.640 | **0.708** | +6.8% |
| CoCA | MAE↓ | 0.084 | 0.089 | **0.054** | -2.7% |
| CoSOD3k | $S_m$↑ | 0.865 | 0.862 | **0.895** | +3.0% |
| CoSal2015 | $S_m$↑ | 0.894 | 0.900 | **0.927** | +2.7% |

### 消融实验

| 配置 | CoCA $S_m$↑ | CoCA $F_m$↑ | 说明 |
|------|-----------|-----------|------|
| EVP (仅简单提示) | 0.686 | 0.510 | 简单可训练参数无法建模共识 |
| VCP (C+D训练) | 0.774 | 0.660 | 使用较小训练集 |
| VCP (S+D训练) | **0.819** | **0.708** | 完整模型，SOTA |

### 关键发现

- 在最能反映模型鲁棒性的 CoCA 数据集上，VCP 以 +6.8% 的 $F_m$ 改进大幅超越所有全微调方法，示性参数效率和性能可以兼得
- EVP（简单提示学习用于 SOD）在 CoSOD 上表现极差（CoCA $S_m$ 仅 0.686），证明共识建模是 CoSOD 的核心难点，简单提示无法解决
- VCP 使用两种训练集配置（C+D 和 S+D）都能超越全微调方法，说明方法对训练数据不敏感
- 冻结基础模型 + 少量可训练参数的范式，反而比全量微调获得了更好的结果，这可能因为避免了有限 CoSOD 数据对预训练知识的破坏

## 亮点与洞察

- **共识嵌入提示的架构创新**：将 CoSOD 特有的共识概念与通用的提示学习范式完美结合。共识作为提示注入每一层，实现了编码与共识的深度交互——这是传统三阶段架构做不到的
- **参数效率的"反常"优势**：冻结大模型、仅调少量参数反而超越全微调，这并非偶然。CoSOD 数据集规模有限（几万张），全微调可能过拟合并破坏预训练表示，而提示学习保留了预训练知识
- **原型学习代替外部模型**：CPG 通过可学习显著性种子的聚类实现显著性估计，无需额外的 SOD 数据集或预训练 SOD 模型，设计简洁
- **top-k 共识种子选择**：从全组所有像素嵌入中选择最具代表性的 k 个作为共识表示，这一操作巧妙地实现了跨图像的共识发现

## 局限与展望

- 基础模型限定为 SegFormer，未验证在更大规模模型（如 ViT-Large、SAM）上的效果
- CPG 中 saliency seed 数量 $j$ 和 consensus seed 数量 $k$ 的选择可能需要针对不同数据集调整
- 推理时需要整组图像一起处理，组大小的变化可能影响性能
- 未探讨在开放词汇或自然语言引导的 CoSOD 场景下的扩展

## 相关工作与启发

- **vs SCED/GCoNet+**: 这些方法采用传统三阶段架构和全微调，在 CoCA 上 $S_m$ 不超过 0.741。VCP 用提示学习达到 0.819，证明了新范式的优越性
- **vs EVP/VSCode**: EVP 为 SOD 设计简单提示，VSCode 引入任务/域特定提示。但两者都未建模组间共识，直接用于 CoSOD 效果很差。VCP 的核心创新在于"共识即提示"
- **vs VPT**: VPT 开创了视觉提示学习，但其提示是通用的可学习嵌入。VCP 的提示是从数据中动态生成的共识表示，更具任务针对性

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 将共识嵌入提示的思路非常巧妙，完美匹配 CoSOD 任务特性与提示学习范式
- 实验充分度: ⭐⭐⭐⭐⭐ 对比13个方法，三个数据集，六个指标，对比非常全面
- 写作质量: ⭐⭐⭐⭐ 动机论述清晰，但方法部分符号较多，CPG流程略显复杂
- 价值: ⭐⭐⭐⭐⭐ 在参数效率和性能间取得极佳平衡，为 CoSOD 社区指明了提示学习的新方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Self-supervised Co-salient Object Detection via Feature Correspondences at Multiple Scales](../../ECCV2024/segmentation/self-supervised_co-salient_object_detection_via_feature_correspondences_at_multi.md)
- [\[CVPR 2025\] RSONet: Region-guided Selective Optimization Network for RGB-T Salient Object Detection](rsonet_region-guided_selective_optimization_network_for_rgb-t_salient_object_det.md)
- [\[CVPR 2025\] DA-VPT: Semantic-Guided Visual Prompt Tuning for Vision Transformers](da-vpt_semantic-guided_visual_prompt_tuning_for_vision_transformers.md)
- [\[CVPR 2026\] Generalizable Co-Salient Object Detection via Mixed Content-Style Modulation](../../CVPR2026/segmentation/generalizable_co-salient_object_detection_via_mixed_content-style_modulation.md)
- [\[CVPR 2025\] G2HFNet: GeoGran-Aware Hierarchical Feature Fusion Network for Salient Object Detection in Optical Remote Sensing Images](binwang2hfnet_geogran-aware_hierarchical_feature_fusion_network_for_salient_obje.md)

</div>

<!-- RELATED:END -->
