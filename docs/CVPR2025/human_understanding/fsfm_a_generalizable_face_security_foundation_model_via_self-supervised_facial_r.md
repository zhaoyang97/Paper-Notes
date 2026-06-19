---
title: >-
  [论文解读] FSFM: A Generalizable Face Security Foundation Model via Self-Supervised Facial Representation Learning
description: >-
  [CVPR 2025][人体理解][人脸安全] FSFM 提出首个面向人脸安全任务的自监督预训练框架，通过 CRFR-P 面部掩码策略 + MIM/ID 双任务协同学习真实人脸的 3C 表示（区域内一致性、区域间连贯性、局部到全局对应性），在深伪检测、活体检测和扩散伪造检测三大任务上超越任务专用 SOTA。
tags:
  - "CVPR 2025"
  - "人体理解"
  - "人脸安全"
  - "自监督预训练"
  - "掩码图像建模"
  - "实例判别"
  - "基础模型"
---

# FSFM: A Generalizable Face Security Foundation Model via Self-Supervised Facial Representation Learning

**会议**: CVPR 2025  
**arXiv**: [2412.12032](https://arxiv.org/abs/2412.12032)  
**代码**: [https://fsfm-3c.github.io](https://fsfm-3c.github.io)  
**领域**: 人体理解  
**关键词**: 人脸安全, 自监督预训练, 掩码图像建模, 实例判别, 基础模型

## 一句话总结
FSFM 提出首个面向人脸安全任务的自监督预训练框架，通过 CRFR-P 面部掩码策略 + MIM/ID 双任务协同学习真实人脸的 3C 表示（区域内一致性、区域间连贯性、局部到全局对应性），在深伪检测、活体检测和扩散伪造检测三大任务上超越任务专用 SOTA。

## 研究背景与动机

**领域现状**：人脸安全领域面临三大威胁任务：深度伪造检测（DfD）、人脸活体检测（FAS）和扩散伪造检测（DiFF）。现有方法绝大多数采用全监督学习，使用各种骨干网络（Xception、EfficientNet、ViT），权重从 ImageNet 预训练初始化，且每个任务都有独立的专用方法。

**现有痛点**：(1) 全监督需要大量标注或生成式增强，成本高且可扩展性差；(2) ImageNet 预训练提供的是自然图像表征，缺乏人脸特有的"真实性"表征，限制了模型在人脸安全任务上的泛化能力；(3) DfD 关注数字伪造痕迹、FAS 关注物理欺骗线索，两者被视为不兼容的独立任务，缺乏统一的基础表征。

**核心矛盾**：现有的面部自监督方法（FaRL、MARLIN 等）主要学习面部分析任务（表情识别、属性分析）所需的显著面部特征，但伪造和欺骗的人脸在这些显著特征上与真实人脸相似——它们忽略了区分真伪所需的"真实性"表征。

**本文目标** 如何从大量无标签真实人脸图像中学习一种鲁棒且可迁移的基础表征，使其能同时提升跨数据集的深伪检测、跨域活体检测和未见过的扩散伪造检测？

**切入角度**：融合 MIM 和 ID 两种自监督范式的互补优势——MIM 擅长局部像素级上下文感知，ID 擅长全局语义对齐——并设计面部结构感知的掩码策略来注入面部先验。

**核心 idea**：用 CRFR-P 面部掩码驱动 MIM 捕获区域内/区域间关系，同时耦合 ID 自蒸馏建立局部到全局对应，三个学习目标（3C）共同预训练出通用人脸安全基础模型。

## 方法详解

### 整体框架
输入为无标签真实人脸图像。CRFR-P 掩码策略生成两个掩码（完整掩码 M 和面部区域掩码 $M_{fr}$），MIM 网络（在线编码器 + 在线解码器）从可见 patch 重建被遮蔽区域，ID 网络（在线分支 + 目标分支通过 EMA 更新）将掩码后的局部特征与完整图像的全局特征对齐。预训练后，在线编码器（vanilla ViT）作为通用基础模型，只需简单微调即可应用于各下游人脸安全任务。

### 关键设计

1. **CRFR-P 面部掩码策略（Covering Random Facial Region + Proportional）**:

    - 功能：生成同时促进区域内一致性和区域间连贯性的面部掩码
    - 核心思路：先用人脸解析器将面部分为 8 个语义区域（眼睛、鼻子、嘴巴、眉毛、面部边界、头发、皮肤、背景）。然后 (1) 完全遮蔽一个随机选择的面部区域（如整个鼻子），迫使模型从其他区域推断被遮蔽区域——学习区域间连贯性；(2) 对剩余区域按比例均匀遮蔽——每个区域都保留部分可见 patch，确保模型能学习每个区域的内部纹理一致性
    - 设计动机：随机掩码容易完全遮蔽小但信息量大的区域（如眼睛），且可能通过同区域残留 patch 走"捷径"。CRFR-P 强制遮蔽整个区域消除捷径，同时比例采样保证所有区域参与学习

2. **双任务 MIM 重建目标**:

    - 功能：从两个层面监督掩码重建——全局像素重建 + 特定面部区域重建
    - 核心思路：在线编码器只处理可见 patch，在线解码器补全全部 patch。损失为 $\mathcal{L}_{rec} = \mathcal{L}_{rec}^m + \lambda_{fr}\mathcal{L}_{rec}^{fr}$，其中 $\mathcal{L}_{rec}^m$ 是所有被遮蔽 patch 的 MSE 重建损失，$\mathcal{L}_{rec}^{fr}$ 专门针对被完全覆盖的面部区域的 MSE 损失
    - 设计动机：额外的面部区域损失 $\mathcal{L}_{rec}^{fr}$ 显式强化了区域间连贯性学习——被完全遮蔽的区域只能从其他区域的信息推断，不存在"从同区域残留patch复制"的捷径

3. **ID 自蒸馏网络（Local-to-Global Correspondence）**:

    - 功能：建立掩码后局部特征与完整图像全局语义之间的对齐
    - 核心思路：在线分支接收 CRFR-P 掩码处理后的 patch，目标分支接收完整 patch（无掩码），两个分支共享编码器架构。对称的表示解码器 $D_o^r$ 和 $D_t^r$ 将编码器输出映射到解齐空间，再通过投影头+预测头进行负余弦相似度 $\mathcal{L}_{sim}$ 的最小化。目标分支参数通过 EMA 更新，采用 stop-gradient 仅对在线分支反传梯度
    - 设计动机：选择完整 patch 而非其他掩码视角作为目标分支输入，保证了语义完整性。表示解码器的引入解耦了像素重建和语义对齐的特征空间，避免 MIM 的低级任务干扰 ID 的高级语义学习

### 损失函数 / 训练策略
- 总损失：$\mathcal{L} = \mathcal{L}_{rec}^m + \lambda_{fr}\mathcal{L}_{rec}^{fr} + \lambda_{cl}\mathcal{L}_{sim}$
- 在 VGGFace2（310万真脸图像）上预训练 vanilla ViT-B
- 预训练后仅取在线编码器 $E_o$，在各下游任务数据集上简单微调

## 实验关键数据

### 主实验

**跨数据集深伪检测（在 FF++ 上训练，跨域测试 Video-level AUC %）**:

| 方法 | 预训练 | CDFV2 | DFDC | DFDCP | WDF | 平均 |
|------|--------|-------|------|-------|-----|------|
| **FSFM ViT-B** | SSL(VF2) | **91.44** | **83.47** | **89.71** | **86.96** | **87.90** |
| ViT-B | Sup(IN) | 86.24 | 74.48 | 82.11 | 81.20 | 81.01 |
| MAE ViT-B | SSL(IN) | 79.51 | 75.93 | 87.10 | 80.96 | 80.88 |
| DINO ViT-B | SSL(IN) | 80.47 | 76.90 | 84.64 | 82.06 | 81.02 |
| SBIs (CVPR'22) | Init(IN) | 93.18 | 72.42 | 86.15 | - | - |

FSFM 平均 AUC 比 ImageNet 监督预训练高 6.9 个点，比 MAE/DINO 等视觉自监督方法高 ~7 个点，且在 DFDC 上超专用 SOTA 方法 SBIs 约 11 个点。

### 消融实验

| 配置 | 平均 AUC | 说明 |
|------|---------|------|
| CRFR-P + MIM + ID (Full) | 最高 | 3C 完整 |
| Random mask + MIM + ID | 降低 | 缺失面部结构先验 |
| CRFR-P + MIM only | 降低 | 缺失全局语义对齐 |
| CRFR-P + ID only | 降低 | 缺失像素级上下文感知 |
| Fasking-I + MIM + ID | 降低 | 掩码策略不如 CRFR-P |
| FRP + MIM + ID | 中等 | 有一致性但缺连贯性 |
| CRFR-R + MIM + ID | 中等 | 有连贯性但随机部分可能遮蔽小区域 |

### 关键发现
- CRFR-P 掩码策略是最大贡献者——它比随机掩码和 Fasking-I 都好，因为它同时保证了区域内一致性和区域间连贯性
- MIM 和 ID 确实互补——单独使用任一都不如合用，MIM 提供局部细节感知、ID 提供全局语义
- 预训练数据量越多越好——增加无标签真脸数据可以进一步提升性能（一个"免费午餐"效应）
- 一个 vanilla ViT 就能统一服务三大人脸安全任务，无需任务专用架构

## 亮点与洞察
- **CRFR-P 掩码设计**：用面部语义分区巧妙地将面部先验注入通用 MIM 框架，"遮一个区域 + 比例遮其他"的组合既防止捷径又保证覆盖，设计简洁但效果显著
- **3C 协同设计**：MIM 的两级重建损失（全局 + 区域）与 ID 的自蒸馏形成三个层次的学习目标（像素→区域→实例），层层递进，覆盖了从低级到高级的完整表征谱
- **通用性验证**：用同一个预训练模型+简单微调就能在跨数据集 DfD、跨域 FAS、未见扩散伪造三个任务上全部超越各自的专用 SOTA，这是强有力的泛化证据

## 局限与展望
- CRFR-P 依赖现成的人脸解析器，如果解析失败（如极端姿态、遮挡），掩码策略可能退化为随机掩码
- 仅使用 ViT-B (86M)，更大规模的 ViT-L/H 或更多预训练数据的潜力未被充分探索
- 预训练只用图像，未利用视频的时序信息，对视频级伪造检测的提升空间可能更大
- 10 个评测数据集主要是面部，跨模态（如声纹+人脸）的场景未涉及

## 相关工作与启发
- **vs MAE**: MAE 使用随机掩码在 ImageNet 上预训练，缺乏面部结构先验。FSFM 通过 CRFR-P 注入面部语义，在人脸任务上显著优于 MAE
- **vs DINO**: DINO 只做实例判别没有像素级重建。FSFM 证明了 MIM+ID 的组合在面部安全任务上比单一范式更有效
- **vs FaRL**: FaRL 依赖图文对的对比学习，需要额外的文本数据。FSFM 仅需无标签人脸图像，数据需求更低且更专注于"真实性"表征
- **vs MCF**: MCF 也结合 MIM+ID 但在 LFW 上预训练且缺乏有效的面部掩码策略，FSFM 在所有指标上优于 MCF

## 评分
- 新颖性: ⭐⭐⭐⭐ CRFR-P 掩码策略新颖有效，但 MIM+ID 组合的大框架已有先例
- 实验充分度: ⭐⭐⭐⭐⭐ 10个数据集、3大任务、多个预训练基线对比、充分消融
- 写作质量: ⭐⭐⭐⭐ 动机推导清晰，3C 概念统一性好
- 价值: ⭐⭐⭐⭐⭐ 人脸安全基础模型方向的重要工作，实践价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Bi-Level Optimization for Self-Supervised AI-Generated Face Detection](../../ICCV2025/human_understanding/bi-level_optimization_for_self-supervised_ai-generated_face_detection.md)
- [\[CVPR 2026\] Action Motifs: Self-Supervised Hierarchical Representation of Human Body Movements](../../CVPR2026/human_understanding/action_motifs_self-supervised_hierarchical_representation_of_human_body_movement.md)
- [\[ECCV 2024\] VideoClusterNet: Self-Supervised and Adaptive Face Clustering for Videos](../../ECCV2024/human_understanding/videoclusternet_self-supervised_and_adaptive_face_clustering_for_videos.md)
- [\[CVPR 2025\] ControlFace: Harnessing Facial Parametric Control for Face Rigging](controlface_harnessing_facial_parametric_control_for_face_rigging.md)
- [\[ECCV 2024\] Pose-Aware Self-Supervised Learning with Viewpoint Trajectory Regularization](../../ECCV2024/human_understanding/pose-aware_self-supervised_learning_with_viewpoint_trajectory_regularization.md)

</div>

<!-- RELATED:END -->
