---
title: >-
  [论文解读] NegRefine: Refining Negative Label-Based Zero-Shot OOD Detection
description: >-
  [ICCV 2025][多模态VLM][OOD检测] 本文提出 NegRefine，通过 LLM 过滤负标签集中的专有名词和子类别标签，并设计多标签匹配评分函数来处理图像同时匹配分布内和负标签的情况，在 ImageNet-1K 基准上平均 AUROC 提升 1.82%、FPR95 降低 4.35%，刷新了零样本 OOD 检测 SOTA。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "OOD检测"
  - "零样本"
  - "CLIP"
  - "负标签"
  - "多标签匹配"
---

# NegRefine: Refining Negative Label-Based Zero-Shot OOD Detection

**会议**: ICCV 2025  
**arXiv**: [2507.09795](https://arxiv.org/abs/2507.09795)  
**代码**: [https://github.com/ah-ansari/NegRefine](https://github.com/ah-ansari/NegRefine)  
**领域**: 多模态VLM  
**关键词**: OOD检测, 零样本, CLIP, 负标签, 多标签匹配

## 一句话总结
本文提出 NegRefine，通过 LLM 过滤负标签集中的专有名词和子类别标签，并设计多标签匹配评分函数来处理图像同时匹配分布内和负标签的情况，在 ImageNet-1K 基准上平均 AUROC 提升 1.82%、FPR95 降低 4.35%，刷新了零样本 OOD 检测 SOTA。

## 研究背景与动机

**领域现状**：基于 CLIP 的零样本 OOD 检测近年来取得显著进展。负标签方法（如 NegLabel、CSP）通过从 WordNet 中选取语义上远离分布内类别的词作为"负标签"，利用 CLIP 的图文相似度区分分布内和 OOD 样本，是当前最具前景的方向。

**现有痛点**：负标签方法存在三个关键问题：(a) **子类别重叠**——负标签中可能包含分布内标签的子类别（如 "african daisy" 是 "daisy" 的子类），而 CLIP 倾向给更细粒度的标签更高分，导致分布内样本被误判为 OOD；(b) **专有名词干扰**——WordNet 中大量专有名词（如 "costa rica"）对某些分布内图像有极高匹配度；(c) **多标签匹配**——真实图像常包含多个物体或匹配多个描述，分布内图像可能同时高度匹配某些负标签。

**核心矛盾**：现有方法仅依赖文本语义相似度的阈值来选择负标签，缺乏对词汇层级关系的显式建模；评分函数独立考虑每个标签的匹配度，未考虑一张图像可能同时匹配多个标签的现实情况。

**本文目标** (a) 如何清理负标签集，去除会导致误判的子类别和专有名词；(b) 如何设计评分函数，使其对同时匹配分布内和负标签的图像鲁棒。

**切入角度**：利用 LLM 的语义理解能力判断词汇间的层级关系和专有名词属性；利用 CLIP 训练时学到的"多物体描述"能力，通过拼接标签构造类似 caption 的文本来检测多标签匹配。

**核心 idea**：用 LLM 过滤有害负标签 + 用标签拼接构造多匹配评分，两手抓提升零样本 OOD 检测的可靠性。

## 方法详解

### 整体框架
NegRefine 在 NegLabel/CSP 的框架上进行改进。输入是一张测试图像和分布内标签集 $Y_{in}$。方法首先通过 NegFilter 机制从负标签集 $Y_{neg}$ 中移除专有名词和子类别标签得到精炼集 $Y'_{neg}$，然后在推理时对每张图像计算原始 NegLabel 评分 $S_{NegLabel}$ 和多标签匹配评分 $S_{MM}$，加权求和得到最终的分布内评分 $S(x) = S_{NegLabel}(x) + \alpha \times S_{MM}(x)$。

### 关键设计

1. **NegFilter 负标签过滤机制**:

    - 功能：从初始负标签集中移除专有名词和分布内标签的子类别
    - 核心思路：对每个负标签 $w$，先用 LLM 判断是否为专有名词（"Is $w$ a proper noun, like the name of an entity?"）；若非专有名词，再找 $Y_{in}$ 中与 $w$ 最相似的 $n=10$ 个标签，逐一询问 LLM 是否为子类别关系。确认为专有名词或子类别的标签从 $Y'_{neg}$ 中移除
    - 设计动机：WordNet 层级结构不可靠（如 "african daisy" 和 "daisy" 在同一级），LLM 在判断语义关系方面更准确灵活。实验中专有名词过滤去除了 1749 个标签（20.6%），子类别过滤去除了 307 个（3.6%）

2. **Multi-Matching Score ($S_{MM}$)**:

    - 功能：为同时匹配分布内和负标签的图像提供额外的分布内评分补偿
    - 核心思路：取图像与 $Y_{in}$ 和 $Y'_{neg}$ 各 top-$k$ 最匹配的标签，构造拼接文本 $t_{i,j}$ = "$y_i$ and $\tilde{y}_j$"。计算 $S_{MM}(x) = \max_{i,j} \frac{e^{\text{sim}(x, t_{i,j})/\tau}}{e^{\text{sim}(x, t_{i,j})/\tau} + e^{\text{sim}(x, \tilde{y}_j)/\tau}}$。如果图像确实同时包含分布内物体和负标签物体，拼接文本的相似度会显著高于单独负标签的相似度，$S_{MM}$ 偏大；若分布内标签不相关（OOD 样本），拼接文本相似度不会增加，$S_{MM}$ 偏小
    - 设计动机：CLIP 训练于 (image, caption) 对，caption 常描述图像中的多个物体。利用 CLIP 对组合描述的理解能力，来判别多标签匹配是否真实

3. **最终评分函数**:

    - $S(x) = S_{NegLabel}(x) + \alpha \times S_{MM}(x)$，其中 $\alpha = 2$
    - $S_{NegLabel}$ 捕获整体分布内/OOD 倾向，$S_{MM}$ 对多标签匹配的分布内样本提供补偿

### 损失函数 / 训练策略
本方法为零样本方法，无需训练。NegFilter 中使用 Qwen2.5-14B-Instruct 作为 LLM，过滤只需做一次（离线预处理）。

## 实验关键数据

### 主实验
ImageNet-1K 作为分布内数据，4 个可靠 OOD 数据集（排除了被证实有高分布内污染的 SUN、Places、Textures）：

| 方法 | iNaturalist AUROC↑ | OpenImage-O AUROC↑ | Clean AUROC↑ | NINCO AUROC↑ | 平均 AUROC↑ | 平均 FPR95↓ |
|------|-------|-------|-------|-------|-------|-------|
| MCM | 94.59 | 92.00 | 83.24 | 74.34 | 86.04 | 52.59 |
| GL-MCM | 96.44 | 92.91 | 84.78 | 76.03 | 87.54 | 44.46 |
| CLIPN | 96.20 | 92.22 | 87.31 | 78.72 | 88.61 | 39.39 |
| NegLabel | 99.49 | 93.74 | 86.79 | 77.30 | 89.33 | 35.11 |
| CSP | 99.60 | 94.09 | 88.32 | 77.88 | 89.97 | 34.47 |
| **NegRefine** | **99.57** | **95.00** | **90.65** | **81.92** | **91.79** | **30.12** |

### 消融实验

| NegFilter | $S_{MM}$ | 平均 AUROC↑ | 平均 FPR95↓ | 说明 |
|-----------|----------|-----------|-----------|------|
| ✗ | ✗ | 89.97 | 34.47 | CSP 基线 |
| ✓ | ✗ | 90.95 | 31.88 | 仅过滤 FPR95 降2.59% |
| ✗ | ✓ | 91.35 | 31.70 | 仅多匹配 FPR95 降2.77% |
| ✓ | ✓ | **91.79** | **30.12** | 完整方法 FPR95 降4.35% |

### 关键发现
- **两个组件互补增益**：NegFilter 和 $S_{MM}$ 各自独立贡献约 2.6-2.8% 的 FPR95 改进，合并后达到 4.35%
- **子类别过滤比专有名词过滤影响更大**：子类别虽仅移除 307 个标签（vs 专有名词 1749 个），但对 FPR95 的单独改善更大（0.88% vs 0.61%），说明子类别重叠问题更严重
- **$S_{MM}$ 优于局部特征方法**：GL-MCM 用 CLIP 的 patch token 处理多物体问题，但本文的标签拼接策略更有效，因为即使单物体图像也可能匹配多标签（如 Fig 1(d) 的项链与叶形），patch 特征无法处理这种情况
- **OOD 数据集选择很重要**：作者遵循 [Bitterwolf 2023] 的发现，排除了 SUN（26.2% 实际为分布内）和 Places（59.5%）等被污染的 OOD 数据集

## 亮点与洞察
- **LLM 作为语义过滤器**：用 LLM 判断词汇间的层级关系和专有名词属性，比依赖 WordNet 本身的层级结构更可靠。这一思路可扩展到需要精细语义判断的其他任务
- **标签拼接的巧妙设计**：将 "$y_i$ and $\tilde{y}_j$" 构造成类似 caption 的文本，利用 CLIP 对多物体描述的理解力来检测多标签匹配。简单直觉但效果显著
- **重新审视 OOD benchmark 的可靠性**：文章指出常用 OOD 数据集（SUN、Places、Textures）存在严重的分布内污染问题，采用更可靠的评估方案，这对整个 OOD 检测社区有参考价值

## 局限与展望
- **LLM 依赖**：NegFilter 依赖 LLM 的判断准确性，不同 LLM 可能给出不同结果
- **超参数敏感性**：$\alpha=2$、$k=5$ 均为固定值，论文虽有消融但未自动调参
- **计算开销**：推理时需额外计算 $k^2=25$ 个拼接文本的相似度，增加了延迟
- **仅适用于负标签范式**：$S_{MM}$ 的设计绑定于负标签-分布内标签对比的框架
- **改进方向**：可探索用 VLM 直接对图像内容做多标签检测替代拼接策略；可研究自适应 $\alpha$ 而非固定加权

## 相关工作与启发
- **vs NegLabel [ICLR'24]**：NegLabel 首创负标签思路但忽略了标签集质量问题和多标签匹配，NegRefine 直接在其框架上补强
- **vs CSP [AAAI'24]**：CSP 用形容词构造超类标签来扩展负标签的 OOD 覆盖范围，但同样受困于子类别和多标签问题
- **vs GL-MCM [NeurIPS'23]**：GL-MCM 用 CLIP 局部特征处理多物体图像，但仅对空间上可分离的多物体有效，本文方法更具一般性

## 评分
- 新颖性: ⭐⭐⭐⭐ 两个组件虽各自简单但组合有效，标签拼接思路新颖
- 实验充分度: ⭐⭐⭐⭐ 消融详尽，对比充分，采用更可靠的评估数据集
- 写作质量: ⭐⭐⭐⭐ 用清晰的案例图解说明三个痛点，逻辑链完整
- 价值: ⭐⭐⭐⭐ 在零样本 OOD 检测上实现了实质性改进，方法可直接迁移

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Mind the Way You Select Negative Texts: Pursuing the Distance Consistency in OOD Detection with VLMs](../../CVPR2026/multimodal_vlm/mind_the_way_you_select_negative_texts_pursuing_the_distance_consistency_in_ood_.md)
- [\[AAAI 2026\] Cross-modal Proxy Evolving for OOD Detection with Vision-Language Models](../../AAAI2026/multimodal_vlm/cross-modal_proxy_evolving_for_ood_detection_with_vision-lan.md)
- [\[CVPR 2026\] Activation Matters: Test-time Activated Negative Labels for OOD Detection with Vision-Language Models](../../CVPR2026/multimodal_vlm/activation_matters_test-time_activated_negative_labels_for_ood_detection_with_vi.md)
- [\[ICCV 2025\] Exploiting Vision Language Model for Training-Free 3D Point Cloud OOD Detection](exploiting_vision_language_model_for_training-free_3d_point_cloud_ood_detection_.md)
- [\[ICCV 2025\] Interpretable Zero-Shot Learning with Locally-Aligned Vision-Language Model](interpretable_zero-shot_learning_with_locally-aligned_vision-language_model.md)

</div>

<!-- RELATED:END -->
