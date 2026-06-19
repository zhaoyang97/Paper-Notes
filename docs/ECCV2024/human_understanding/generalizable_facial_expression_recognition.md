---
title: >-
  [论文解读] Generalizable Facial Expression Recognition
description: >-
  [ECCV 2024][人体理解][表情识别] 提出 CAFE 方法，通过在固定 CLIP 人脸特征上学习 Sigmoid Mask 选取表情相关特征，配合通道分离和通道多样性损失，实现仅使用单个训练集就能在多个未见数据集上大幅超越 SOTA 表情识别方法的零样本泛化能力。 问题定义：当前 SOTA 表情识别（FER）方法在…
tags:
  - "ECCV 2024"
  - "人体理解"
  - "表情识别"
  - "零样本泛化"
  - "Sigmoid Mask"
  - "CLIP"
  - "领域泛化"
---

# Generalizable Facial Expression Recognition

**会议**: ECCV 2024  
**arXiv**: [2408.10614](https://arxiv.org/abs/2408.10614)  
**代码**: [有](https://github.com/zyh-uaiaaaa/Generalizable-FER)  
**领域**: 人体理解  
**关键词**: 表情识别, 零样本泛化, Sigmoid Mask, CLIP, 领域泛化

## 一句话总结

提出 CAFE 方法，通过在固定 CLIP 人脸特征上学习 Sigmoid Mask 选取表情相关特征，配合通道分离和通道多样性损失，实现仅使用单个训练集就能在多个未见数据集上大幅超越 SOTA 表情识别方法的零样本泛化能力。

## 研究背景与动机

**问题定义**：当前 SOTA 表情识别（FER）方法在训练集对应的测试集上表现优异，但在有领域差异的测试集上性能急剧下降。例如在 RAF-DB 上训练的 EAC 模型在 RAF-DB 测试集上达到 89.54% 准确率，但在 AffectNet 上仅有 43.91%。

**现有方法的不足**：

**SOTA FER 方法泛化差**：SCN、RUL、EAC、OFER 等方法过度拟合训练集的域特定信息（domain-specific），在有域差异的测试集上表现不稳定。越强的方法（如 EAC）在源域越好，但在跨域上甚至比弱方法更差——说明是"过拟合式"的性能提升。

**领域自适应 FER 方法需目标域数据**：现有域自适应方法需要获取目标域的标注或未标注样本进行微调，在真实部署中往往不可行，因为我们事先不知道测试样本的分布。

**人类启发的核心洞察**：人类识别表情时遵循"先定位人脸 → 再提取表情特征"的两步过程。面对有域差异的图像，人类先排除域相关特征（如背景、光照），再聚焦于表情相关特征做判断。CAFE 方法模仿了这一认知过程：

- 第一步：用预训练大模型（CLIP）提取泛化的人脸特征
- 第二步：训练 FER 模型学习 Mask，从人脸特征中选取表情相关特征

**为什么 CLIP 特征直接用不行？** CLIP 提取的是通用人脸特征，包含身份、年龄、光照等非表情信息。将通用特征直接用于 FER 任务是 non-trivial 的——需要一种机制来精确选取与表情相关的特征维度，同时保持泛化能力不退化。

## 方法详解

### 整体框架

CAFE 包含三个核心组件：

1. **固定 CLIP 特征提取**：用冻结的 CLIP ViT-B/32 提取人脸特征 $\mathbf{F} \in \mathbb{R}^{N \times C}$（训练全程固定）
2. **Sigmoid Mask 学习**：ResNet-18 骨干网络学习 Mask，通过 Sigmoid 正则化后应用于 CLIP 特征
3. **通道分离 + 通道多样性**：将 mask 后的特征按通道分成 7 组对应 7 种基本表情，直接生成 logits，避免 FC 层过拟合

### 关键设计

1. **Sigmoid Mask 学习（Mask on Fixed Face Features）**

   **功能**：学习一个概率掩码，从固定的 CLIP 人脸特征中筛选出表情相关的特征通道。

   **核心思路**：FER 模型（ResNet-18）提取特征 $\mathbf{f}$，经 reshape 后生成 Mask $\mathbf{M}$，再通过 Sigmoid 函数正则化：

    $\mathbf{M}_s = \text{Sigmoid}(\mathbf{M})$
    $\widetilde{\mathbf{F}} = \mathbf{M}_s \mathbf{F}$

   选取后的特征 $\widetilde{\mathbf{F}}$ 送入标准分类损失：

    $l_{cls} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{e^{\mathbf{W}_{y_i} \widetilde{\mathbf{F}}_i}}{\sum_j^L e^{\mathbf{W}_j \widetilde{\mathbf{F}}_i}}$

   **设计动机**：（a）固定 CLIP 特征可防止 FER 模型优化人脸特征导致过拟合训练集，保持泛化性；（b）Sigmoid 函数将 Mask 值限制在 $[0, 1]$，提供概率化的特征选择语义——每个通道的值代表该特征被选中的概率，类似人类从人脸特征中选择表情特征的过程；（c）Sigmoid 的非线性引入有助于捕获非线性模式，同时归一化效果降低了 Mask 的过拟合能力。

2. **通道分离模块（Channel-Separation）**

   **功能**：将 Mask 后的 512 维特征按通道均分为 7 组，每组对应一种基本表情，通过 MaxPool 直接生成 logits，绕过 FC 层。

   **核心思路**：将 $\widetilde{\mathbf{F}}$ 按通道分为 $\{\widetilde{\mathbf{F}}_1, ..., \widetilde{\mathbf{F}}_7\}$（各约 73 通道），对每组应用随机通道丢弃后 MaxPool：

    $\overline{\mathbf{F}}^d = \{\max(\widetilde{\mathbf{F}}_1 \mathbf{M}_1), \max(\widetilde{\mathbf{F}}_2 \mathbf{M}_2), ..., \max(\widetilde{\mathbf{F}}_L \mathbf{M}_L)\}$

   基于 $\overline{\mathbf{F}}^d$ 计算分离损失 $l_{sep}$。

   **设计动机**：三层考虑——（a）FC 层学习能力过强，容易过拟合训练集标签，直接从特征到 logits 的映射更简单；（b）512 维特征可能过大，每组仅 73 维的 Mask 更可能只关注有用信息而非冗余；（c）类似标签分布学习，一张图可能包含多种表情特征（如复合表情），7 组特征分别关注 7 种表情有理论依据。

3. **通道多样性损失（Channel-Diverse Loss）**

   **功能**：强制不同表情类别对应的 Mask 尽可能多样化，避免不同表情通道学习到相似的特征模式。

   **核心思路**：

    $l_{div} = 1 - \frac{1}{Nc} \sum_{i=1}^{N} \sum_{j=1}^{L} \widetilde{\mathbf{F}}_{\max_{ij}}$

   其中 $\widetilde{\mathbf{F}}_{\max}$ 是对每组特征取 MaxPool 后的结果，$c=73$ 为归一化常数。

   **设计动机**：如果不同表情对应的 Mask 高度相似，则模型无法有效区分不同表情的特征子空间。通过最大化每组特征的最大值，迫使各组 Mask 选择不同的特征通道，提升 Mask 的多样性和判别性。

### 损失函数 / 训练策略

总训练损失：

$$l_{train} = l_{cls} + \lambda \cdot l_{sep} + \beta \cdot l_{div}$$

- $\lambda = 1.5$（分离损失权重），$\beta = 5$（多样性损失权重）
- 推理时只需 $l_{cls}$ 对应的模块，$l_{sep}$ 和 $l_{div}$ 分支可丢弃
- 优化器 Adam，学习率 0.0002，ExponentialLR 调度（gamma=0.9）

## 实验关键数据

### 主实验

在 RAF-DB 训练、5 个数据集上测试的结果（准确率%）：

| 方法 | RAF-DB (源域) | FERPlus | AffectNet | SFEW2.0 | MMA | 均值 |
|------|-------------|---------|-----------|---------|-----|------|
| SCN | 87.32 | 58.37 | 42.85 | 44.89 | 36.52 | 53.99 |
| EAC | 89.54 | 54.38 | 43.91 | 43.39 | 37.27 | 53.70 |
| OFER | 89.07 | 53.90 | 42.73 | 43.88 | 36.43 | 53.20 |
| **CAFE** | **88.72** | **73.16** | **45.86** | **52.86** | **56.80** | **63.48** |

CAFE 的跨域均值达到 63.48%，相比最佳基线 EAC（53.70%）提升 **+9.78%**，且在每个未见测试集上均大幅领先。

### 消融实验

在 RAF-DB 训练的消融结果：

| Mask | Separation | Diverse | FERPlus | AffectNet | SFEW2.0 | MMA | 均值 |
|------|------------|---------|---------|-----------|---------|-----|------|
| ✗ | ✗ | ✗ | 58.05 | 43.25 | 42.76 | 42.61 | 46.67 |
| ✓ | ✗ | ✗ | 70.90 | 43.77 | 51.63 | 55.65 | 55.49 |
| ✓ | ✓ | ✗ | 72.01 | 45.17 | 53.31 | 56.69 | 56.80 |
| ✓ | ✓ | ✓ | **73.16** | **45.86** | **52.86** | **56.80** | **57.17** |

- Sigmoid Mask 是最关键组件：+8.82% 均值提升
- 通道分离进一步提升 +1.31%
- 通道多样性损失再提升 +0.37%

### 关键发现

- **SOTA FER 方法的"过拟合陷阱"**：EAC 在源域超过 SCN，但在跨域上反而更差，说明更强的拟合能力带来更差的泛化
- **Sigmoid 函数至关重要**：无 Sigmoid 的 Mask 均值仅 58.05%，有 Sigmoid 则达到 63.48%——归一化效果是防止 Mask 过拟合的关键
- **CLIP+Finetune 远不如 CAFE**：同样参数量下，CLIP+Finetune 仅 55.73%，CAFE 达到 63.48%（+7.75%），证明是 Mask 设计而非单纯 CLIP 特征的功劳
- **不同骨干网络均有效**：MobileNet（+1.64%）、ResNet-18（+8.47%）、ResNet-50（+3.09%）均有提升

## 亮点与洞察

1. **认知启发的设计理念**：模仿人类"先看脸→再看表情"的两步认知过程，将问题分解为特征提取和特征选择两个独立步骤
2. **"固定+学习"的范式**：固定大模型特征保泛化，学习轻量 Mask 保精度，二者解耦是成功关键
3. **避免 FC 层的巧妙设计**：通道分离后直接 MaxPool 生成 logits，减少了一个重要的过拟合来源
4. **推理成本低**：训练时的辅助模块可在推理时丢弃，实际部署仅需 CLIP + ResNet-18 + Mask

## 局限与展望

- 仅考虑 7 种基本表情，未扩展到细粒度表情或连续情感空间（VA 值预测）
- CLIP 特征维度（512）与 ResNet-18 输出维度相同时效果最佳，维度不匹配时需要简单的 mean 操作可能限制性能
- 未探索更强的大模型（如 DINOv2、EVA-CLIP）作为特征提取器
- 通道分离为均匀分割，未考虑不同表情可能需要不同数量的特征通道

## 相关工作与启发

- **CLIP for downstream tasks**：固定 CLIP 特征 + 轻量任务头的范式在多个领域验证有效，CAFE 为 FER 提供了一个优雅的适配方案
- 思路可推广到其他细粒度识别任务（如微表情识别、动作单元检测），只要源域和目标域存在域差异
- 通道分离 + MaxPool 直接生成 logits 的策略可能对其他需要防过拟合的分类任务有启发

## 评分

- **新颖性**: ⭐⭐⭐⭐ — Sigmoid Mask 在固定大模型特征上学习表情特征选择的思路新颖且直觉合理，通道分离避免 FC 层的设计有巧思
- **实验充分度**: ⭐⭐⭐⭐⭐ — 5 个数据集交叉验证、5 种训练设置、详细消融、多骨干、CLIP+Finetune 对比、Sigmoid 影响、超参研究，非常全面
- **写作质量**: ⭐⭐⭐⭐ — 问题定义清晰，与人类认知的类比生动，图表丰富
- **价值**: ⭐⭐⭐⭐ — 首次系统性研究 FER 零样本泛化问题，方法简洁高效，对实际部署有重要参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] SynFER: Towards Boosting Facial Expression Recognition with Synthetic Data](../../ICCV2025/human_understanding/synfer_towards_boosting_facial_expression_recognition_with_synthetic_data.md)
- [\[ECCV 2024\] TF-FAS: Twofold-Element Fine-Grained Semantic Guidance for Generalizable Face Anti-Spoofing](tf-fas_twofold-element_fine-grained_semantic_guidance_for_generalizable_face_ant.md)
- [\[CVPR 2026\] CLEX: Complementary Label Exchange Learning for Noisy Facial Expression Recognition](../../CVPR2026/human_understanding/clex_complementary_label_exchange_learning_for_noisy_facial_expression_recogniti.md)
- [\[ECCV 2024\] How Video Meetings Change Your Expression](how_video_meetings_change_your_expression.md)
- [\[CVPR 2026\] Dynamic Label Noise Suppression with Optimal Teacher Pool for Facial Expression Recognition](../../CVPR2026/human_understanding/dynamic_label_noise_suppression_with_optimal_teacher_pool_for_facial_expression_.md)

</div>

<!-- RELATED:END -->
