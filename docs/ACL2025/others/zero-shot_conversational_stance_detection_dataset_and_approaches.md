---
title: >-
  [论文解读] Zero-Shot Conversational Stance Detection: Dataset and Approaches
description: >-
  [ACL 2025][stance detection] 构建了首个零样本多轮多方对话立场检测数据集 ZS-CSD（280 个目标、17,063 条对话样本），并提出 SITPCL 模型，结合说话者交互编码器与目标感知原型对比学习，在零样本对话立场检测中取得 SOTA（F1-macro 43.81%）。
tags:
  - "ACL 2025"
  - "stance detection"
  - "zero-shot"
  - "conversational"
  - "speaker interaction"
  - "对比学习"
---

# Zero-Shot Conversational Stance Detection: Dataset and Approaches

**会议**: ACL 2025  
**arXiv**: [2506.17693](https://arxiv.org/abs/2506.17693)  
**代码**: [GitHub](https://github.com/whu-yzding/ZS-CSD)  
**领域**: 其他  
**关键词**: stance detection, zero-shot, conversational, speaker interaction, prototypical contrastive learning

## 一句话总结

构建了首个零样本多轮多方对话立场检测数据集 ZS-CSD（280 个目标、17,063 条对话样本），并提出 SITPCL 模型，结合说话者交互编码器与目标感知原型对比学习，在零样本对话立场检测中取得 SOTA（F1-macro 43.81%）。

## 研究背景与动机

立场检测旨在从文本中识别用户对特定目标的观点倾向（支持/反对/中立），在情感识别、论点挖掘和谣言检测等领域有广泛应用。随着社交媒体上在线辩论日益增多，对话场景下的立场检测成为重要研究方向。

现有对话立场检测存在三个关键局限：

1. **目标数量极其有限**：最大的对话立场检测数据集 MT-CSD 仅包含 5 个目标，且现有数据集仅覆盖名词短语型或帖子型目标中的一种，无法应对真实世界中大量未见目标的场景。
2. **忽略说话者信息**：此前的标注和建模过程仅考虑回复关系和历史上下文，未充分利用说话者上下文信息（同一用户的历史发言）和潜在的说话者间交互信息（不同用户间的互动关系）。
3. **缺乏零样本设置**：对话立场检测研究局限于目标内（in-target）和跨目标（cross-target）两种任务，尚未探索更贴近实际应用的零样本场景——即对完全未见过的目标进行立场判断。

本文针对以上问题，构建了首个零样本对话立场检测数据集 ZS-CSD，并提出 SITPCL 模型作为基准方法。

## 方法详解

### 整体框架

SITPCL（Speaker Interaction and Target-aware Prototypical Contrastive Learning）是一个四阶段流水线：
1. **话语编码器**：使用预训练语言模型（Chinese-RoBERTa）分别编码每条话语-目标对，添加立场模板后经 GRU 层获取上下文表示序列
2. **说话者交互编码器**：通过注意力机制分别建模说话者内部和说话者之间的依赖关系
3. **目标感知原型对比学习**：以目标原型为锚点的对比学习，增强不同目标间的表示区分
4. **分类器**：softmax 分类器输出立场预测

### 关键设计

1. **ZS-CSD 数据集构建**:

    - 功能：提供首个大规模、高质量的零样本对话立场检测评测资源
    - 核心思路：从微博收集 200 万帖子及评论，经关键词筛选（6 个争议性领域）、讨论树构建、多深度采样后，由 8 名标注者完成两阶段标注——先由 3 名专家确定目标（每段对话 1-2 个），再由 5 名标注者标注立场
    - 设计亮点：(a) 包含名词短语目标（113 个）和声明型目标（167 个）两种类型，共 280 个目标；(b) 标注时同时考虑对话历史、说话者上下文和说话者交互信息；(c) Cohen's Kappa = 0.83，标注质量高；(d) 训练/验证/测试集的目标完全不重叠，确保真正的零样本评估

2. **说话者交互编码器**:

    - 功能：捕捉对话中同一说话者内部的观点连贯性以及不同说话者之间的互动关系
    - 核心思路（说话者内依赖）：将当前说话者的上一次增强表示与当前话语表示拼接后作为查询向量，通过注意力机制在该说话者的历史上下文上聚合信息，得到说话者内状态向量 $v_i^{intra}$
    - 核心思路（说话者间依赖）：以当前话语的上下文表示 $h_i$ 为查询，以其他说话者的先前增强状态作为键，通过注意力机制捕捉跨说话者的交互信号，得到说话者间状态向量 $v_i^{inter}$
    - 融合方式：$v_i = W_3[v_i^{intra} \oplus v_i^{inter}] + b_3$，线性变换后得到最终的说话者增强表示
    - 设计动机：社交媒体对话中，用户立场往往具有连贯性（同一用户在不同回复中倾向保持一致），且不同用户间存在相互影响和对抗关系

3. **目标感知原型对比学习**:

    - 功能：增强模型对不同目标的区分能力，提升零样本泛化
    - 核心思路：为每个目标 $t$ 计算原型表示 $p_t$（该目标下所有话语表示的平均值），使用 InfoNCE 风格的对比损失将话语表示拉近对应目标原型、推远其他目标原型
    - 损失公式：$\mathcal{L}_{TPC} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(\text{sim}(v_i, p_{y_i})/\tau)}{\sum_{k=1}^{K}\exp(\text{sim}(x_i, p_k)/\tau)}$
    - 设计动机：零样本场景下模型需理解"目标"的概念，而非记忆具体目标；通过使不同目标的表示空间逐步分离，模型可以将这种区分能力迁移到未见目标上

### 损失函数 / 训练策略

总损失：$\mathcal{L} = \mathcal{L}_{CE} + \gamma \mathcal{L}_{TPC}$

- $\mathcal{L}_{CE}$：标准交叉熵损失，用于三分类立场预测
- $\mathcal{L}_{TPC}$：目标感知原型对比损失，温度参数 $\tau$ 控制分布锐度
- $\gamma$：平衡系数

训练配置：
- PLM：Chinese-RoBERTa-wwm-ext，GRU 隐藏维度 768
- 优化器：AdamW，学习率 1e-5，权重衰减 1e-6
- 训练 20 epochs，batch size 16
- 5 个随机种子取平均
- 硬件：2× NVIDIA RTX 3090，训练时间 < 3 小时

## 实验关键数据

### 主实验

零样本对话立场检测 F1-macro 得分：

| 方法 | 混合目标 | 名词短语目标 | 声明目标 |
|------|---------|------------|---------|
| Llama3-8B (zero-shot) | 35.91 | 39.37 | 33.79 |
| GPT-4o-mini (zero-shot) | 38.16 | 40.21 | 36.57 |
| GPT-3.5 (zero-shot) | 40.25 | 47.35 | 35.53 |
| Llama3-70B (zero-shot) | 41.07 | 48.72 | 36.56 |
| Qwen2.5-14B (zero-shot) | 42.78 | 44.60 | 41.34 |
| RoBERTa (fine-tuned) | 40.27 | 42.79 | 38.71 |
| Branch-BERT | 43.11 | 44.48 | 38.96 |
| GLAN | 41.79 | 45.04 | 39.83 |
| **SITPCL (ours)** | **43.81** | **47.54** | **41.47** |

### 消融实验

| 配置 | 混合目标 F1-macro | 说明 |
|------|-----------------|------|
| SITPCL (完整) | **43.81** | 全部组件 |
| W/o 说话者交互编码器 (SIE) | 42.91 (-0.90) | 移除说话者交互建模 |
| W/o 目标原型对比学习 (TPCL) | 42.59 (-1.22) | 移除对比学习 |
| W/o 两者 (BOTH) | 41.40 (-2.41) | 仅保留基础编码器 |

### 关键发现

1. **SITPCL 在所有目标类型上均取得最优**：混合目标 43.81%，名词短语目标 47.54%，声明目标 41.47%，全面超越 LLM 零样本方法和微调基线
2. **即使最强 LLM（Llama3-70B）也仅达 41.07%**：凸显零样本对话立场检测的极高难度——最优方法 F1-macro 也不到 44%
3. **两个组件协同互补**：SIE 对 Favor 类提升明显，TPCL 对 Neutral 类提升显著（+3.95%），移除两者导致最大性能下降（-2.41%）
4. **对话深度影响模型表现**：SITPCL 在浅层对话（depth=1）中表现突出（40.19% vs 基线 ~25-33%），在深层对话（depth≥6）中也保持稳定（45.25%）
5. **名词短语目标 vs 声明目标**：声明型目标普遍更难（所有方法均低 3-8 个百分点），因为声明本身表达观点，用户的立场表达更隐晦间接

## 亮点与洞察

- **首个零样本对话立场检测数据集**：280 个目标远超此前最大的 5 个目标（MT-CSD），涵盖名词短语和声明两种类型，填补了该领域的重要空白
- **说话者交互建模思路新颖**：不仅利用回复关系，还通过注意力机制捕捉同一说话者的观点连贯性和不同说话者间的观点对抗，为对话理解提供了更细粒度的信号
- **原型对比学习的零样本适配**：将原型网络的思想与对比学习结合，使模型学会基于"目标概念"而非"具体目标"做判断，有效提升零样本泛化
- **LLM vs 小模型的有趣对比**：70B LLM 在名词短语目标上逼近小模型，但在声明目标上落后明显，说明对话结构的利用比语言理解能力更关键

## 局限与展望

- **性能上限偏低**：最优 F1-macro 仅 43.81%，距离实际应用仍有较大差距
- **仅支持中文**：数据集来源于微博，标注语言为中文，跨语言泛化性未验证
- **目标类型划分较粗**：仅分名词短语和声明两类，未考虑更细粒度的目标语义分类
- **缺乏多模态信息**：社交媒体对话常包含图片、表情包等多模态信号，本文仅使用文本
- **说话者交互建模可强化**：当前使用简单注意力机制，可探索图神经网络或更复杂的对话结构建模方法

## 相关工作与启发

- **vs VAST/C-STANCE/EZ-STANCE**（句子级零样本）：本文首次将零样本设定拓展到对话级别，增加了说话者交互的建模维度
- **vs MT-CSD/MmMtCSD**（对话级立场检测）：这些工作仅有 3-5 个目标且限于 in-target/cross-target 设定，本文的 280 目标零样本设置更贴近真实场景
- **vs LLM 零样本方法**：即使 70B 参数的 LLM 也无法超越精心设计的小模型方法，表明对话立场检测的困难不仅在于语言理解，更在于需要捕捉对话结构和交互动态
- **启发**：对话场景下的 NLP 任务应更多关注说话者身份和交互模式，这类结构化信息是 LLM 通过简单 prompt 难以充分利用的

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个零样本对话立场检测数据集+任务定义，填补重要空白
- 实验充分度: ⭐⭐⭐⭐ 涵盖 LLM 和微调基线，消融、深度分析、可视化完整
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，数据集构建过程详尽透明
- 价值: ⭐⭐⭐ 数据集贡献大于方法贡献，SITPCL 模型设计相对常规，43.81% 的绝对性能也暗示该任务仍需更强的方法

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] USDC: A Dataset of User Stance and Dogmatism in Long Conversations](usdc_a_dataset_of_underlineuser_underlinestance_and_underlinedogmatism_in_long_u.md)
- [\[CVPR 2026\] Zero-shot Detection of AI-Generated Image via RAW-RGB Alignment](../../CVPR2026/others/zero-shot_detection_of_ai-generated_image_via_raw-rgb_alignment.md)
- [\[CVPR 2025\] Zero-Shot Head Swapping in Real-World Scenarios](../../CVPR2025/others/zero-shot_head_swapping_in_real-world_scenarios.md)
- [\[ACL 2025\] A Practical Approach for Building Production-Grade Conversational Agents with Workflow Graphs](a_practical_approach_for_building_production-grade_conversational_agents_with_wo.md)
- [\[ACL 2025\] CONFETTI: Conversational Function-Calling Evaluation Through Turn-Level Interactions](confetti_conversational_function-calling_evaluation_through_turn-level_interacti.md)

</div>

<!-- RELATED:END -->
