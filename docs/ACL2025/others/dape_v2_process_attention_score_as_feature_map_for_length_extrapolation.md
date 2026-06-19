---
title: >-
  [论文解读] DAPE V2: Process Attention Score as Feature Map for Length Extrapolation
description: >-
  [ACL 2025][长度外推] 本文将 Transformer 的注意力分数视为特征图，通过在注意力分数上施加卷积操作（而非简单的 key-query 点积）来显著提升 Transformer 在长序列上的长度外推能力，将外推问题转化为经典的图像特征处理问题。 领域现状：Transformer 模型在语言处理、计算机视觉等…
tags:
  - "ACL 2025"
  - "长度外推"
  - "注意力机制"
  - "位置编码"
  - "卷积处理"
  - "特征图"
---

# DAPE V2: Process Attention Score as Feature Map for Length Extrapolation

**会议**: ACL 2025  
**arXiv**: [2410.04798](https://arxiv.org/abs/2410.04798)  
**代码**: [https://github.com/chuanyang-Zheng/DAPE](https://github.com/chuanyang-Zheng/DAPE)  
**领域**: 其他  
**关键词**: 长度外推、注意力机制、位置编码、卷积处理、特征图

## 一句话总结

本文将 Transformer 的注意力分数视为特征图，通过在注意力分数上施加卷积操作（而非简单的 key-query 点积）来显著提升 Transformer 在长序列上的长度外推能力，将外推问题转化为经典的图像特征处理问题。

## 研究背景与动机

**领域现状**：Transformer 模型在语言处理、计算机视觉等领域取得了巨大成功，但其核心的 key-query 点积注意力机制在处理超出训练长度的序列时性能会急剧下降。现有的位置编码方法如 RoPE、ALiBi、Kerple 等试图通过不同方式嵌入位置信息来缓解这一问题。

**现有痛点**：传统位置编码方法（如 RoPE）在输入长度超过训练长度两倍时就会完全失效。即便是像 FIRE、CoPE 这样的改进方法，在大幅外推时性能下降仍然严重。此外，现有方法大多是静态的、预定义的，无法根据不同输入动态调整。

**核心矛盾**：问题的根源不仅仅在于位置编码的设计，而在于注意力分数计算本身——朴素的 key-query 点积表达能力有限，限制了 Transformer 在长序列上的泛化能力。

**本文目标**：(1) 揭示长度外推问题的本质原因是注意力分数的表达能力不足；(2) 通过更精细地处理注意力分数来提升外推性能。

**切入角度**：作者在尝试将 DAPE（数据自适应位置编码）与 NoPE（无位置编码）结合时，发现仅在注意力分数上加 MLP（不含任何位置信息）也能显著提升性能。这一"意外发现"表明，关键不在于位置编码，而在于对注意力分数本身的进一步处理。

**核心 idea**：将注意力分数张量 $[B,H,T,T]$ 类比为图像特征图 $[B,C,H,W]$，在注意力头维度上施加卷积操作来增强注意力的表达能力，从而将长度外推问题转化为成熟的计算机视觉特征处理问题。

## 方法详解

### 整体框架

DAPE V2 在标准 Transformer 的 key-query 乘法之后、softmax 之前，插入一个卷积模块来处理注意力分数。输入是 key-query 点积得到的原始注意力矩阵，经过下三角 mask 后，通过 $1 \times k$ 的卷积核在 key 维度和 head 维度上进行交互，输出是精炼后的注意力分数，再送入 softmax 继续标准注意力流程。

### 关键设计

1. **注意力分数的特征图视角**:

    - 功能：将注意力分数张量重新解释为可处理的特征图
    - 核心思路：注意力分数的形状为 $[B,H,T,T]$，与图像特征图 $[B,C,H,W]$ 完全对应。原版 DAPE 使用 MLP 处理注意力分数，等价于 $1 \times 1$ 卷积。本文提出使用 $1 \times k$（如 $1 \times 3$）的更大卷积核，在 key 维度上跨头进行卷积，使相邻 token 和不同 head 之间能够交换信息
    - 设计动机：$1 \times 1$ 卷积（即 MLP）在图像处理中被认为表达能力受限，扩大感受野到 $1 \times 3$ 可以捕获 key 维度上的局部模式关系，这是对经典 CV 知识的直接迁移

2. **下三角 mask 保证因果性**:

    - 功能：防止信息泄露，保证自回归（causal）性质
    - 核心思路：在卷积之前对注意力分数执行 `torch.tril` 操作，确保只有过去位置的信息能参与当前位置的注意力计算。卷积使用 stride=1、padding=$k-1$ 来保持序列长度不变
    - 设计动机：作者通过信息泄露实验验证了这一设计的必要性——去掉下三角 mask 后模型可以达到近零 loss（perplexity=1），证明卷积确实能利用注意力数据中的信息

3. **理论证明：卷积实现联想回忆**:

    - 功能：从理论上证明带卷积的 Transformer 能无需位置编码即可完成联想回忆（associative recall）任务
    - 核心思路：以"Hakuna→Matata"为例，使用 $1 \times 2$ 卷积核 $[-1, 1]$，可以使注意力集中到目标 token 的下一个位置。具体地，卷积将 key 向量变换为相邻 token 的差值 $W_K(x_2 - x_1)$，当查询向量 $x_N = x_1$ 时，注意力自然集中到 $x_2$（"Matata"）
    - 设计动机：联想回忆任务是 Transformer perplexity 的主要来源。标准 Transformer 通过隐式的位置编码机制实现这一功能，而卷积提供了一种显式的、无需位置编码的替代方案

### 损失函数 / 训练策略

使用标准的语言建模交叉熵损失。DAPE V2 的额外参数量很小（仅为卷积核和 MLP 参数），采用与基线相同的 AdamW 优化器训练。训练在 8 GPU 上进行，125M 模型用 50k 步，350M/2.7B 模型也用 50k 步。

## 实验关键数据

### 主实验

在 Arxiv 和 Books3 数据集上，训练长度 512，评估不同外推长度（ppl 越低越好）：

| 方法 | 512 | 1024 | 2048 | 4096 | 8192 |
|------|-----|------|------|------|------|
| NoPE | 5.10 | 42.27 | 极大 | 极大 | 极大 |
| RoPE | 4.57 | 86.20 | 237.67 | 256.12 | — |
| Kerple | 4.57 | 4.37 | 5.09 | 6.80 | 9.08 |
| DAPE-Kerple (1×1) | 4.49 | 4.20 | 4.17 | 3.95 | 3.70 |
| **DAPE₁ₓ₃-Kerple** | **4.44** | **4.14** | **4.09** | **3.87** | **3.58** |

大模型 2.7B 在 Books3 上的表现：

| 方法 | 512 | 1024 | 2048 | 4096 |
|------|-----|------|------|------|
| RoPE | 21.01 | 25.00 | 48.13 | 160.59 |
| Kerple | 21.14 | 22.08 | 23.38 | 27.21 |
| DAPE-Kerple | 20.52 | 21.01 | 20.23 | 19.67 |
| **DAPE₁ₓ₃-Kerple** | **20.16** | **20.54** | **19.80** | **19.02** |

### 消融实验

不同卷积核大小的影响（Arxiv，训练长度 128）：

| 配置 | 128 | 8192 | 说明 |
|------|-----|------|------|
| Kerple（无卷积） | 8.30 | 12.59 | 基线 |
| DAPE-Kerple (1×1) | 8.21 | 4.97 | MLP 等价于 1×1 卷积 |
| DAPE₁ₓ₃-Kerple (1×3) | 8.15 | 4.60 | 最佳性价比 |
| DAPE₁ₓ₅-Kerple (1×5) | 8.13 | 4.57 | 更大核略有提升 |
| DAPE₁ₓ₇-Kerple (1×7) | 8.12 | 4.57 | 进一步增大收益递减 |

### 关键发现

- **卷积核大小 1×3 是最佳性价比选择**：从 1×1 到 1×3 带来显著提升，继续增大到 1×5/1×7 收益递减
- **DAPE₁ₓ₃ 在更低计算成本下性能更优**：$D_{DAPE}=10$ 的 DAPE₁ₓ₃ 超过 $D_{DAPE}=64$ 的原版 DAPE
- **训练短序列可媲美长序列**：DAPE₁ₓ₃ 训练长度 128 的表现可媲美 RoPE 训练长度 4096，节省大量训练时间
- **大模型上额外开销比例递减**：350M 模型开销比为 0.75，6.7B 模型仅 0.89，说明随模型增大，DAPE₁ₓ₃ 的额外成本可忽略
- **在 CHE benchmark 上也有效**：在 11 个任务中，DAPE₁ₓ₃-Kerple 在 8 个任务上超过基线 Kerple

## 亮点与洞察

- **视角转换的巧妙性**：将序列模型中的注意力机制类比为 CV 中的特征图，这一跨领域的联想非常精彩。这种思维方式使得 CV 领域积累了几十年的特征处理技术可以直接迁移到 NLP
- **从"意外发现"到系统性理论**：DAPE+NoPE 的偶然实验揭示了一个深层次问题——注意力计算本身才是瓶颈，而不是位置编码。这种从实验现象出发、逐步建立理论解释的研究路径值得学习
- **可迁移到其他注意力变体**：这一思路可以推广到任何使用注意力机制的模型（视觉 Transformer、跨注意力等），通过在注意力分数上施加卷积来增强表达能力

## 局限与展望

- **仅在语言建模任务上验证**：未涉及其他长上下文任务如长文档 QA、摘要等
- **核大小选择缺乏自动化**：不同任务和数据集的最优核大小可能不同，目前需要手动调参
- **未与最新的长上下文方法对比**：如 LongRoPE、YaRN 等针对已有大模型的后训练外推方法
- **卷积沿 query 维度的拓展**：当前仅使用 $1 \times k$ 核（沿 key 维度），使用完整的 $k \times k$ 核可能带来进一步提升但计算开销更大
- **与线性注意力/稀疏注意力的结合**：能否在高效注意力方法上也应用此思路值得探索

## 相关工作与启发

- **vs DAPE**：原版 DAPE 使用 MLP（等价于 1×1 卷积）自适应调整位置编码偏置，本文发现更大的卷积核可以在无位置编码时也带来提升，证明核心贡献来自注意力分数的精炼而非位置编码自适应
- **vs RoPE/ALiBi/Kerple**：这些方法都是静态的位置编码方案，与本文提出的动态注意力分数处理机制互补——DAPE V2 可以叠加在这些方法之上进一步提升性能
- **vs FlashConv/Hyena 等混合模型**：传统混合模型在 token 值上施加卷积，本文在注意力分数上施加卷积，保留了原始 token 值不变，是一种更轻量的增强方式

## 评分

- 新颖性: ⭐⭐⭐⭐ 注意力特征图视角很新颖，但核心操作（在注意力上加卷积）相对直观
- 实验充分度: ⭐⭐⭐⭐⭐ 实验非常全面，多数据集、多模型规模、消融充分
- 写作质量: ⭐⭐⭐⭐ 论述清晰，但与 DAPE 有文本重叠，部分内容冗余
- 价值: ⭐⭐⭐⭐ 为 Transformer 注意力机制提供了新的改进思路，但实际大模型中的应用还需验证

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] EpiCoDe: Boosting Model Performance Beyond Training with Extrapolation and Contrastive Decoding](epicode_boosting_model_performance_beyond_training_with_extrapolation_and_contra.md)
- [\[ACL 2025\] Map&Make: Schema Guided Text to Table Generation](mapmake_schema_guided_text_to_table_generation.md)
- [\[ACL 2025\] Unique Hard Attention: A Tale of Two Sides](unique_hard_attention_a_tale_of_two_sides.md)
- [\[ACL 2025\] MapQaTor: An Extensible Framework for Efficient Annotation of Map-Based QA Datasets](mapqator_an_extensible_framework_for_efficient_annotation_of_map-based_qa_datase.md)
- [\[ACL 2025\] What Matters in Evaluating Book-Length Stories? A Systematic Study of Long Story Evaluation](what_matters_in_evaluating_book-length_stories_a_systematic_study_of_long_story_.md)

</div>

<!-- RELATED:END -->
