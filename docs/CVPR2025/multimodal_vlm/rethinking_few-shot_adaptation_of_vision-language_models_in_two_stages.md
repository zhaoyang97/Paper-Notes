---
title: >-
  [论文解读] Rethinking Few-Shot Adaptation of Vision-Language Models in Two Stages
description: >-
  [CVPR 2025][多模态VLM][少样本适配] 通过分析 PEFT 在少样本适配中的学习动态，发现训练过程天然分为"任务级特征提取"和"可用类别特化"两个阶段，据此提出 2SFS：先调 LayerNorm 学通用特征，再训练线性分类器提升已知类判别，在 base-to-novel 和 all-to-all 两种设定下均达到或超越 SOTA。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "少样本适配"
  - "CLIP"
  - "参数高效微调"
  - "两阶段学习"
  - "视觉语言模型"
---

# Rethinking Few-Shot Adaptation of Vision-Language Models in Two Stages

**会议**: CVPR 2025  
**arXiv**: [2503.11609](https://arxiv.org/abs/2503.11609)  
**代码**: [https://github.com/FarinaMatteo/rethinking_fewshot_vlms](https://github.com/FarinaMatteo/rethinking_fewshot_vlms)  
**领域**: 多模态VLM  
**关键词**: 少样本适配, CLIP, 参数高效微调, 两阶段学习, 视觉语言模型

## 一句话总结

通过分析 PEFT 在少样本适配中的学习动态，发现训练过程天然分为"任务级特征提取"和"可用类别特化"两个阶段，据此提出 2SFS：先调 LayerNorm 学通用特征，再训练线性分类器提升已知类判别，在 base-to-novel 和 all-to-all 两种设定下均达到或超越 SOTA。

## 研究背景与动机

Few-Shot Adaptation (FSA) 的核心矛盾是：VLM 参数量巨大（数亿级），而每类仅有少量样本（如 16-shot），导致全参数微调必然过拟合。现有方法主要分为两类：

1. **Prompt Tuning**（CoOp、CoCoOp、MaPLe 等）：在文本/视觉编码器中插入可训练上下文向量
2. **Adapter-based**（CLIP-Adapter、TaskRes、CLIP-LoRA 等）：在冻结模型外围添加参数化适配模块

这些方法都在某个特定设定（base-to-novel 或 all-to-all）下表现良好，但**跨设定泛化性差**。例如，MMA 在 base-to-novel 上是 SOTA，但 all-to-all 下大幅退化；CLIP-LoRA 反之亦然。

更关键的是，社区对 FSA 训练过程中到底发生了什么缺乏深入理解。作者通过实验首次揭示了一个关键现象：**PEFT 的学习动态存在一个天然的"断点"**，在此之前 base 和 novel 类性能共同上升，之后 base 类继续提升但 novel 类开始退化。

## 方法详解

### 整体框架

2SFS（Two-Stage Few-Shot Adaptation）将固定的计算预算 $m$ 次迭代拆分为两个阶段：
- **第一阶段**（$\alpha \times m$ 步）：微调 LayerNorm 参数学习任务级特征
- **第二阶段**（$(1-\alpha) \times m$ 步）：冻结特征提取器，训练线性分类器

推理时对 base 类直接查表（$O(1)$），novel 类才需通过文本编码器计算嵌入。

### 关键设计

1. **学习动态分析与断点发现**:
    - 功能：揭示 PEFT 训练过程的内在规律，为两阶段设计提供理论依据
    - 核心思路：在 CLIP ViT-B/16 上用三种 PEFT 方法（LayerNorm tuning、LoRA、BitFit）以 16-shot 训练 8000 步，持续监测 base/novel 类在 held-out 数据上的准确率。发现所有 PEFT 方法都呈现一致模式——存在一个断点，断点前 base 和 novel 类性能**同时上升**（学到了好的任务级特征），断点后 base 类继续提升但 novel 类开始退化（特化到可用数据）
    - 设计动机：这不是传统意义上的过拟合（因为是在 held-out 数据上评估的），而是 PEFT 在特化已有类别时覆盖了对整个任务有用的知识。LayerNorm tuning 在断点后最鲁棒（退化最慢），因此被选为第一阶段的 PEFT 策略

2. **第一阶段：LayerNorm 微调**:
    - 功能：学习可迁移的任务级特征表示
    - 核心思路：微调视觉和文本编码器中所有 LayerNorm 实例的 scale $\gamma$ 和 shift $\beta$ 参数。对 $d$ 维激活向量 $\mathbf{a}$，$\text{LayerNorm}(\mathbf{a}) = \gamma \odot \frac{\mathbf{a} - \mu(\mathbf{a})}{\sigma(\mathbf{a})} + \beta$。使用标准 softmax 交叉熵损失在 base 类上优化 $\alpha \times m$ 步
    - 设计动机：LayerNorm 参数量少但影响全局特征分布，且实验表明其在断点后退化最缓，最适合学任务级特征。关键点在于只训 $\alpha \times m$ 步就停下来，避免进入特化阶段

3. **第二阶段：线性分类器 + 选择性推理**:
    - 功能：提升 base 类判别能力，同时保护 novel 类泛化性
    - 核心思路：冻结第一阶段学到的 LayerNorm 参数，用第一阶段文本编码器的 base 类嵌入 $\phi_b = f^t_{\omega^{*t}_{LN}}(b)$ 初始化分类器权重矩阵 $\Phi_{\mathcal{B}}$，然后在剩余 $(1-\alpha) \times m$ 步中优化 $\Phi_{\mathcal{B}}$。推理时，base 类直接用分类器行向量 $\phi_b^*$（$O(1)$ 查表），novel 类才需通过文本编码器计算 $f^t_{\omega^{*t}_{LN}}(c)$
    - 设计动机：切换到不同参数集合（从 LayerNorm → 分类器权重），避免进一步微调破坏第一阶段学到的任务级特征。线性分类器是最简单的形式，但因为用了第一阶段的好特征做初始化和优化目标，效果出奇地好。选择性推理是两阶段设计的独特副产品

### 损失函数 / 训练策略

- 两个阶段都使用标准 softmax 交叉熵损失，分别对 LayerNorm 参数和分类器权重优化
- 超参数固定：$\alpha = 0.5$（训练预算平分），$m = 8000$ 步，SGD 优化器
- 所有实验跨 11 个数据集、2 种设定、3 种骨干使用**完全相同的超参数**

## 实验关键数据

### 主实验（Base-to-Novel, ViT-B/16, 16-shot 平均）

| 方法 | Base | Novel | HM |
|------|------|-------|-----|
| CLIP zero-shot | 69.34 | 74.22 | 71.70 |
| CoOp | 82.69 | 63.22 | 71.66 |
| MaPLe | 82.28 | 75.14 | 78.55 |
| CLIP-LoRA | 85.32 | 70.63 | 77.28 |
| MMA | 83.20 | 76.80 | 79.87 |
| **2SFS** | **85.55** | **75.48** | **80.20** |

单数据集亮点：

| 数据集 | 指标 | 2SFS | MMA | 说明 |
|--------|------|------|-----|------|
| Stanford Cars | HM | 78.46 | 75.70 | 提升2.76% |
| Oxford Flowers | HM | 85.83 | 85.48 | 略优 |
| Caltech101 | HM | 96.52 | 96.15 | 略优 |
| ImageNet | HM | 74.20 | 74.02 | 略优 |

### 消融实验

| 配置 | 说明 |
|------|------|
| 第一阶段用 LoRA 替代 LN | HM 下降约 1-2%，验证 LN 更鲁棒 |
| $\alpha = 0$ (仅分类器) | 退化为 linear probing，HM 显著下降 |
| $\alpha = 1$ (仅 LN tuning) | 缺少分类器特化阶段，base 类偏弱 |
| 不同骨干 (ViT-B/32, ViT-L/14) | 趋势一致，2SFS 均表现最优或最优之一 |

### 关键发现

- **2SFS 是唯一在 base-to-novel 和 all-to-all 两种设定下都保持竞争力的方法**：MMA 在 base-to-novel 强但 all-to-all 弱，CLIP-LoRA 反之
- 断点现象在所有数据集、所有 PEFT 方法中**一致出现**，是 PEFT 学习动态的固有特性
- LayerNorm tuning 参数量极少但效果出人意料地好，说明 LayerNorm 的 scale/shift 控制着特征分布的关键方面
- 选择性推理是两阶段设计的独有优势：base 类推理零成本

## 亮点与洞察

- **思路极简但洞察深刻**：核心贡献不是复杂架构，而是对 PEFT 学习动态的深入分析。"先学通用特征，再学分类器"这一 old-school recipes 的回归，在有理论分析支撑后变得令人信服
- **固定超参数跨设定/骨干/数据集**：这在 FSA 领域非常罕见，说明方法的鲁棒性极强
- **断点现象本身就是有价值的发现**：可以指导其他 PEFT 方法的早停策略
- 选择性推理提供了推理效率的实际收益

## 局限与展望

- 断点位置因数据集和 PEFT 方法而异，$\alpha = 0.5$ 是经验选择，自适应确定断点位置可能更优
- 仅验证了分类任务，未扩展到检测/分割等下游任务
- 第二阶段用最简单的线性分类器，更复杂的分类头（如 non-linear adapter）可能进一步提升
- 仅在 CLIP 系列模型上验证，能否迁移到 SigLIP、EVA-CLIP 等其他 VLM 待验证

## 相关工作与启发

- 与 CoOp/CoCoOp 等 prompt tuning 方法正交，2SFS 可以理解为一种更通用的"何时停止 PEFT + 何时切换到简单分类器"的框架
- 断点现象与 continual learning 中的灾难性遗忘有相似之处，但本质不同（base 类性能不降，是 novel 类退化）
- 启发：在其他 PEFT 场景（如 instruction tuning）中是否也存在类似的两阶段动态？

## 评分

- 新颖性: ⭐⭐⭐⭐ 核心洞察新颖，方法简洁但有效
- 实验充分度: ⭐⭐⭐⭐⭐ 11个数据集×2设定×3骨干，超参数固定，对比全面
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，从观察到方法到实验一气呵成
- 价值: ⭐⭐⭐⭐ 对 PEFT 学习动态的分析具有广泛启发性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Sparsity Outperforms Low-Rank Projections in Few-Shot Adaptation](../../ICCV2025/multimodal_vlm/sparsity_outperforms_low-rank_projections_in_few-shot_adaptation.md)
- [\[CVPR 2025\] Generalized Few-Shot 3D Point Cloud Segmentation with Vision-Language Model](generalized_few-shot_3d_point_cloud_segmentation_with_vision-language_model.md)
- [\[CVPR 2025\] Realistic Test-Time Adaptation of Vision-Language Models](realistic_test-time_adaptation_of_vision-language_models.md)
- [\[CVPR 2025\] UNEM: UNrolled Generalized EM for Transductive Few-Shot Learning](unem_unrolled_generalized_em_for_transductive_few-shot_learning.md)
- [\[CVPR 2025\] Vision-Language Models Do Not Understand Negation](vision-language_models_do_not_understand_negation.md)

</div>

<!-- RELATED:END -->
