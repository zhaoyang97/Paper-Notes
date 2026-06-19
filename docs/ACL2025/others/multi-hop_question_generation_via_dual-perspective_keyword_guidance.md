---
title: >-
  [论文解读] Multi-Hop Question Generation via Dual-Perspective Keyword Guidance
description: >-
  [ACL 2025][多跳问题生成] 定义了双视角关键词——问题关键词（捕捉提问者意图）和文档关键词（反映 QA 对相关内容），并提出 DPKG 框架，通过扩展 Transformer 编码器和两个答案感知解码器，将关键词无缝集成到多跳问题生成过程中。 多跳问题生成（MQG）要求综合文档中多处信息片段来生成需要多步推理才能回…
tags:
  - "ACL 2025"
  - "多跳问题生成"
  - "双视角关键词"
  - "答案感知注意力"
  - "关键词引导"
  - "HotpotQA"
---

# Multi-Hop Question Generation via Dual-Perspective Keyword Guidance

**会议**: ACL 2025  
**arXiv**: [2505.15299](https://arxiv.org/abs/2505.15299)  
**代码**: [GitHub](https://github.com/imaodong/DPKG)  
**领域**: 其他  
**关键词**: 多跳问题生成, 双视角关键词, 答案感知注意力, 关键词引导, HotpotQA

## 一句话总结

定义了双视角关键词——问题关键词（捕捉提问者意图）和文档关键词（反映 QA 对相关内容），并提出 DPKG 框架，通过扩展 Transformer 编码器和两个答案感知解码器，将关键词无缝集成到多跳问题生成过程中。

## 研究背景与动机

多跳问题生成（MQG）要求综合文档中多处信息片段来生成需要多步推理才能回答的问题。其核心挑战在于：**如何有效定位文档中与问答对相关的关键信息片段**。

现有方法的问题：

**关键词利用不充分**：许多研究仅关注文档特定关键词（如 MulQG、SGCM），或仅在解码时约束问题特定关键词（如 CQG），未能全面利用关键词的引导潜力。

**未区分关键词角色**：现有工作忽视了两类关键词的本质区别：
   - **问题关键词**：来自问题本身，反映提问者的意图，必须出现在生成的问题中
   - **文档关键词**：来自文档，反映与 QA 对相关的内容，用于定位关键信息片段

**两类关键词协同**：问题关键词和文档关键词共同定位文档中的关键信息片段——这是人类自然的提问过程（先看答案和文档→找相关信息片段→选择必须出现在问题中的关键词→生成问题）。

核心思路：区分并显式利用双视角关键词，更好地引导多跳问题生成。

## 方法详解

### 整体框架

DPKG 框架包含三个主要组件：
1. **扩展 Transformer 编码器**：同时编码文档、答案和文档-答案拼接，产生三组隐藏状态
2. **关键词生成解码器**：基于文档和答案状态生成双视角关键词序列
3. **问题生成解码器**：利用关键词定位的信息片段生成多跳问题

两个解码器结构相似但独立，共享编码器输出但各有不同的任务目标。

### 关键设计

1. **扩展 Transformer 编码器**：

    - 功能：同时编码三个输入——文档 $D^i$、答案 $A^i$、文档-答案拼接 $D^i;A^i$
    - 核心思路：修改内部结构使三部分共享自注意力但保持独立编码
    - 输出：$H_{doc}^i$（文档状态）、$H_{ans}^i$（答案状态）、$H_{da}^i$（文档-答案状态）
    - 设计动机：分别编码文档和答案便于后续计算答案感知状态

2. **答案感知注意力机制 (Answer-aware Attention)**：

    - 功能：在解码器中引入答案信息来加权文档注意力
    - 核心公式：$H_a = \text{softmax}(\frac{H_k^{t-1} H_{doc}^T}{\sqrt{d}} \odot K_{weight}) H_{doc}$
    - 其中 $K_{weight} = \text{MeanPooling}(\frac{H_{doc} H_{ans}^T}{\sqrt{d}})$ 捕捉文档-答案关系
    - 设计动机：$K_{weight}$ 编码了文档中哪些部分与答案相关，通过逐元素乘法引导注意力聚焦到答案相关区域

3. **融合模块 (Fusion Module)**：

    - 功能：结合答案感知状态 $H_a$ 和标准交叉注意力状态 $H_h$
    - 门控机制：$H_k^t = gate \odot H_a + (1-gate) \odot H_h$，$gate = \text{sigmoid}([H_a; H_h])$
    - 设计动机：自适应地平衡答案感知信息和原始上下文信息

4. **关键词引导的两种模式**：

    - **Hard 模式**：在关键词前添加特殊前缀 `<qes>` 或 `<doc>` 标识类型，在关键词生成阶段识别角色
    - **Soft 模式**：不添加前缀，在问题生成阶段动态识别每个关键词的角色
    - Hard 模式在 SF 设置中更优，Soft 模式在 Full 设置中更优

### 损失函数 / 训练策略

联合训练损失：$\mathcal{L} = \beta_1 \mathcal{L}_1 + \beta_2 \mathcal{L}_2 + \beta_3 \mathcal{L}_3$

- $\mathcal{L}_1$：关键词生成交叉熵损失
- $\mathcal{L}_2$：问题生成交叉熵损失
- $\mathcal{L}_3$：缩小生成关键词表示与真值关键词表示的差距
    - $\mathcal{L}_3 = \|F_k - F_g\|_2$，其中 $F_k$ 和 $F_g$ 分别是关键词解码器输出和 BART 编码的真值关键词的均值池化表示
    - 设计动机：训练时使用真值关键词，推理时使用生成关键词，$\mathcal{L}_3$ 桥接这一差距

关键词标注：使用 SpaCy（en_core_web_sm 模型）自动标注，聚焦于与答案和问题相关的句子，提取实体或短语作为关键词。疑问词（what、how 等）作为问题关键词。

## 实验关键数据

### 主实验—HotpotQA（表格）

**SF (Supporting Fact) 设置：**

| 模型 | BLEU-4 | METEOR | ROUGE-L |
|------|--------|--------|---------|
| BART | 20.39 | 23.46 | 37.23 |
| CQG | 25.09 | 27.45 | 41.83 |
| QA4QG-large | 25.70 | 27.44 | 46.48 |
| SGCM | 26.16 | 28.51 | 44.06 |
| **DPKG_hard** | **26.80** | 27.87 | **46.50** |
| DPKG_soft | 26.19 | **28.51** | 46.36 |

**Full 设置：**

| 模型 | BLEU-4 | METEOR | ROUGE-L |
|------|--------|--------|---------|
| BART | 16.77 | 20.07 | 33.69 |
| CQG | 21.46 | 24.97 | 39.61 |
| SGCM | 22.61 | 26.04 | 40.61 |
| DPKG_hard | 22.74 | 24.90 | **43.29** |
| **DPKG_soft** | **23.33** | 25.21 | 43.18 |

### 消融/关键词类型实验（表格）

| 模型 | 关键词得分(KP) | BLEU-4 (SF) | ROUGE-L (SF) |
|------|---------------|-------------|--------------|
| DPKG_hard | 88.13 | 26.80 | 46.50 |
| DPKG_soft | 88.86 | 26.19 | 46.36 |
| DPKG_D (仅文档关键词) | 87.05 | 23.95 | 45.62 |
| DPKG_Q (仅问题关键词) | 79.36 | 25.77 | 45.16 |

消融（模块消融，SF 设置）：

| 配置 | BLEU-4 | ROUGE-L |
|------|--------|---------|
| DPKG_hard | 26.80 | 46.50 |
| w/o $\mathcal{L}_3$ | 24.74 | 45.08 |
| w/o Answer-aware | 24.83 | 45.09 |

### 关键发现

1. **双视角优于单视角**：即使使用真值关键词，DPKG_hard/soft 也始终优于仅用问题关键词 (DPKG_Q) 或仅用文档关键词 (DPKG_D)
2. **问题关键词比文档关键词重要**：DPKG_Q 的问题生成质量显著优于 DPKG_D，尽管其关键词生成得分更低（79.36 vs 87.05）
3. **$\mathcal{L}_3$ 和答案感知注意力同等重要**：移除任一模块都导致 BLEU-4 下降约 2 个点
4. **Hard vs Soft 模式互补**：Hard 在 SF 中更好（处理短文档），Soft 在 Full 中更好（处理长文档噪声）
5. **TS-BART 验证了双视角关键词的普适价值**：即使是简单的两阶段 BART 也能从双视角关键词中获益
6. DPKG 在 ROUGE-L 上提升特别显著（SF: 46.50, Full: 43.29），说明生成的问题在长程语义上与参考问题更匹配

## 亮点与洞察

1. **关键词的角色分化**：将关键词从单一概念细分为"问题关键词"和"文档关键词"，是一个简单但有效的概念贡献。这与人类提问的认知过程高度吻合
2. **训练-推理差距的处理**：$\mathcal{L}_3$ 通过拉近生成关键词和真值关键词的表示来缩小训练时用真值、推理时用生成的差距，方法简洁有效
3. **框架可扩展性**：DPKG 的编码器-双解码器架构可以容纳更多类型的中间生成任务

## 局限与展望

1. **关键词标注依赖 SpaCy**：自动标注质量受限于 NER 工具，在复杂文档中可能遗漏重要关键词
2. **仅在 HotpotQA 上评估**：缺乏在其他多跳 QA 数据集（如 2WikiMultiHopQA）上的验证
3. **基于 BART 的架构**：未探索更大的 PLM（如 T5-large、LLaMA）是否能进一步提升
4. **Hard 和 Soft 模式未统一**：理想情况下应有一种自动选择模式的机制，而非人工选择
5. **关键词生成是累积误差源**：关键词质量直接影响问题质量，错误会传播放大

## 相关工作与启发

- CQG 仅在解码时约束问题关键词，DPKG 将这一思路扩展到双视角并引入独立的关键词生成阶段
- QA4QG 用 QA 模块增强 BART，DPKG 用关键词引导模块增强——思路类似但关键词引导更显式
- 答案感知注意力机制借鉴了 Wang et al. (2024) 的工作，但在多跳场景中的应用是新的

## 评分

- **新颖性**: ⭐⭐⭐⭐ — "双视角关键词"定义清晰、直觉自然、效果显著，是一个精致的概念创新
- **实验充分度**: ⭐⭐⭐ — 在 HotpotQA 上实验全面（主实验、关键词类型分析、消融），但缺少其他数据集验证
- **写作质量**: ⭐⭐⭐⭐ — 关键词定义和标注过程用图示直观说明，框架描述清晰
- **价值**: ⭐⭐⭐⭐ — 为 MQG 任务提供了新的视角和有效方法，代码已开源，关键词标注方法可复用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] AIDE: Attribute-Guided Multi-Hop Data Expansion for Data Scarcity in Task-Specific Fine-tuning](aide_attribute-guided_multi-hop_data_expansion_for_data_scarcity_in_task-specifi.md)
- [\[ACL 2025\] DRS: Deep Question Reformulation With Structured Output](drs_deep_question_reformulation_with_structured_output.md)
- [\[ACL 2025\] Counterspeech the Ultimate Shield! Multi-Conditioned Counterspeech Generation through Attributed Prefix Learning](hippro_counterspeech_gen.md)
- [\[AAAI 2026\] Local Guidance for Configuration-Based Multi-Agent Pathfinding](../../AAAI2026/others/local_guidance_for_configuration-based_multi-agent_pathfinding.md)
- [\[ACL 2025\] Evaluating Design Decisions for Dual Encoder-based Entity Disambiguation](evaluating_design_decisions_for_dual_encoder-based_entity_disambiguation.md)

</div>

<!-- RELATED:END -->
