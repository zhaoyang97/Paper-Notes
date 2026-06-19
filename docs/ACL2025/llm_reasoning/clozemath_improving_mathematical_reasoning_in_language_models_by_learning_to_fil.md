---
title: >-
  [论文解读] ClozeMath: Improving Mathematical Reasoning in Language Models by Learning to Fill Equations
description: >-
  [ACL 2025][LLM推理][数学推理] ClozeMath 提出了一种受人类完形填空学习启发的微调策略，通过掩码数学解答中的方程式并训练模型预测它们（text-infilling目标），与标准语言模型目标联合训练，在GSM8K和MATH上显著超越了强基线Masked Thought，并在推理时间扩展和鲁棒性测试中表现出更好的泛化能力。
tags:
  - "ACL 2025"
  - "LLM推理"
  - "数学推理"
  - "Text Infilling"
  - "方程掩码"
  - "PrefixLM"
  - "Chain-of-Thought"
---

# ClozeMath: Improving Mathematical Reasoning in Language Models by Learning to Fill Equations

**会议**: ACL 2025  
**arXiv**: [2506.03763](https://arxiv.org/abs/2506.03763)  
**代码**: 无（Qualcomm AI Research）  
**领域**: LLM Reasoning  
**关键词**: 数学推理, Text Infilling, 方程掩码, PrefixLM, Chain-of-Thought

## 一句话总结

ClozeMath 提出了一种受人类完形填空学习启发的微调策略，通过掩码数学解答中的方程式并训练模型预测它们（text-infilling目标），与标准语言模型目标联合训练，在GSM8K和MATH上显著超越了强基线Masked Thought，并在推理时间扩展和鲁棒性测试中表现出更好的泛化能力。

## 研究背景与动机

当前LLM的数学推理能力主要通过Chain-of-Thought（CoT）格式的数据训练来增强——模型学习在给出答案前生成中间推理步骤。然而，这种训练范式存在一个根本性问题：

**预测范式的局限**：标准的下一个token预测目标可能并不符合人类的学习方式。人类学习数学解法时，倾向于先理解通用方法，再处理具体细节（先把握"解题思路"，再算具体数字），而非简单记忆"哪个步骤接哪个步骤"。

**Masked Thought的缺陷**：最近的强基线方法Masked Thought Fine-tuning (MFT) 通过随机掩码解答中的token来迫使模型关注更远的问题定义信息。但其随机掩码策略存在**虚假关联**问题：当连续的数学变换步骤紧密互联时，掩码可能导致模型在缺少前置步骤定义（如变量定义被掩码）的情况下被迫预测后续步骤，学到错误的依赖关系。

**核心洞察**：在数学解答中，文本部分（rationales）描述的是通用解题方法，而方程式是问题特定的计算。作者受到人类完形填空（cloze）练习的启发：给定解题的文字推理过程，让模型**填充缺失的方程**，从而强化模型对数学关系的推断能力。

## 方法详解

### 整体框架

ClozeMath 在标准语言模型训练之上增加了一个 text-infilling（文本填充）目标。训练时，模型同时优化两个损失：
- $\mathcal{L}_{\text{lm}}$：标准的语言建模目标（给定问题预测解答）
- $\mathcal{L}_{\text{tf}}$：方程填充目标（给定问题和被掩码了方程的解答，预测被掩码的方程）

最终目标：$\mathcal{L}_{\text{ClozeMath}} = \mathcal{L}_{\text{lm}} + \mathcal{L}_{\text{tf}}$

推理时使用常规的逐token生成，不需要任何特殊处理。

### 关键设计

1. **方程掩码策略（Equation Masking）**:

    - **功能**：识别数学解答中的所有方程式，用特殊掩码token（\<X\>, \<Y\>等）替换它们，保留文本推理部分完整。
    - **渐进式反掩码**：给定解答中有 $|F^i|$ 个方程，生成 $|F^i|$ 个text-infilling样本：第一个掩码所有方程，第二个保留第一个方程但掩码剩余的，依此类推。这遵循了数学解答中方程的因果依赖关系——每个方程只依赖之前的方程。
    - **为什么不随机掩码**：消融实验证明，随机掩码（类似T5的span corruption）会破坏文本推理部分的逻辑连贯性，显著降低性能（74.22% → 71.19%）。

2. **PrefixLM架构**:

    - **做法**：在预训练的decoder-only模型上实现PrefixLM——prompt部分（问题+被掩码的解答）使用双向注意力，目标序列（被掩码的方程）使用因果注意力，用 \<SEP\> token分隔。
    - **核心动机**：双向注意力允许模型更好地理解prefix中的信息（问题定义+文字推理），从而更有效地推断缺失的方程。
    - **关键发现**：PrefixLM单独使用时改善微弱（71.57% → 71.79%），但与方程填充目标结合后效果显著（71.57% → 74.22%）。这说明充分利用双向上下文的前提是有合适的训练目标。

3. **样本平衡**:

    - 由于text-infilling目标的样本数取决于方程数量，实践中通过复制语言建模样本来维持两个目标大约50:50的样本比例。

### 损失函数 / 训练策略

- 联合训练损失：$\mathcal{L}_{\text{ClozeMath}} = \mathcal{L}_{\text{lm}} + \mathcal{L}_{\text{tf}}$
- 使用 LoRA（rank=32）微调基础语言模型（DeepSeek-Math-7B-base、Llama-3.1-8B、Llama-3.2-3B、Llama-3.2-1B）
- 扩展词表以支持 \<SEP\> 和掩码token

## 实验关键数据

### 主实验

| 数据集 | 模型 | 基线(Base) | MFT | ClozeMath | ClozeMath vs MFT |
|--------|------|-----------|-----|-----------|-----------------|
| GSM8K | DeepSeek-Math-7B | 59.21 | 70.20 | **74.22** | +4.02 |
| GSM8K | Llama-3.1-8B | 49.58 | 64.82 | **70.00** | +5.18 |
| GSM8K | Llama-3.2-3B | 17.66 | 45.03 | **53.15** | +8.12 |
| GSM8K | Llama-3.2-1B | 4.62 | 21.15 | **27.89** | +6.74 |
| MATH | DeepSeek-Math-7B | 31.68 | 33.42 | **36.90** | +3.48 |
| MATH | Llama-3.1-8B | 18.06 | 20.94 | **22.88** | +1.94 |

### 消融实验

| 配置 | GSM8K准确率 | 说明 |
|------|------------|------|
| ClozeMath完整版 | **74.22%** | 方程掩码 + PrefixLM |
| W/o Text-infilling | 71.79% | 仅PrefixLM，无填充目标 |
| W/o PrefixLM | 72.71% | 方程填充 + CausalLM |
| W/o 两者（标准IT）| 71.57% | 传统instruction tuning |
| 随机掩码（非方程）| 71.19% | 破坏逻辑连贯性，性能最低 |

### 关键发现

- ClozeMath 在**所有模型规模**上都一致超越 MFT，且在训练过程中更加样本高效（每个checkpoint都更好）
- **推理时间扩展**：使用CoT decoding（k=9）时，ClozeMath同样优于MFT（如DeepSeek-Math: 77.10% vs 76.50%），证明其在增加推理计算量时的扩展性
- **鲁棒性测试（GSM-Symbolic）**：在添加新约束的变体问题上，ClozeMath的优势更加明显（DeepSeek-Math GSM-P1: 49.25% vs 44.25%，提升5%）
- 模型越小，ClozeMath相对MFT的改进越大（Llama-3.2-3B在GSM8K上+8.12）

## 亮点与洞察

- **人类学习的类比非常精准**：完形填空是语言学习的经典方法，将其迁移到数学推理中是一个优雅的设计。模型先理解解题思路（保留的文本），再推导具体方程，符合"先掌握方法论，再处理细节"的学习范式
- **对Masked Thought虚假关联的分析很有洞见**：具体指出了MFT在变换步骤紧密关联时可能学到错误依赖的问题（如缺少变量b的定义却要预测4*b=24）
- **PrefixLM + 方程填充的协同效应**：单独使用PrefixLM几乎无效，但与方程填充结合后效果显著，说明架构选择和训练目标需要协同设计

## 局限与展望

- 仅在10B以下模型上验证，更大模型的效果未知
- 目前仅针对数学推理，是否能推广到其他需要结构化推理的领域（如代码生成、逻辑推理）有待验证
- 方程识别依赖启发式规则，对于更复杂的数学表达（如LaTeX格式的复杂公式）可能需要更鲁棒的解析
- CoT decoding评估因MATH数据集格式问题仅限于GSM8K

## 相关工作与启发

- 与 Masked Thought (Chen et al., 2024) 形成直接对比，展示了有针对性的掩码策略（方程 vs 随机）的差异
- Text infilling 思路源自 T5 (Raffel et al., 2020) 和 UL2 (Tay et al., 2023)，但创新在于只掩码方程而非随机span
- PrefixLM 的应用受 Liu et al. (2018) 启发，验证了其在推理任务中配合特定目标的有效性
- 对研究社区的启发：**训练目标的设计应考虑任务的结构特性**，而非简单套用通用方案

## 评分

- 新颖性: ⭐⭐⭐⭐ 完形填空的类比新颖且优雅，方程掩码策略简单有效
- 实验充分度: ⭐⭐⭐⭐⭐ 多模型、多数据集、消融、鲁棒性、推理时间扩展，非常全面
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰，对MFT缺陷的分析深入，图示直观
- 价值: ⭐⭐⭐⭐ 提供了一种即插即用的微调策略，可广泛应用于数学推理LLM的训练

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Chain-of-Reasoning: Towards Unified Mathematical Reasoning in Large Language Models via a Multi-Paradigm Perspective](chain_of_reasoning_unified_math.md)
- [\[ACL 2025\] Safe: Enhancing Mathematical Reasoning in Large Language Models via Retrospective Step-aware Formal Verification](safe_math_reasoning.md)
- [\[ACL 2025\] LogicPro: Improving Complex Logical Reasoning via Program-Guided Learning](logicpro_program_guided_reasoning.md)
- [\[ACL 2025\] Improving Chain-of-Thought Reasoning via Quasi-Symbolic Abstractions](improving_chain-of-thought_reasoning_via_quasi-symbolic_abstractions.md)
- [\[ICLR 2026\] MathFimer: Enhancing Mathematical Reasoning by Expanding Reasoning Steps through Fill-in-the-Middle Task](../../ICLR2026/llm_reasoning/mathfimer_enhancing_mathematical_reasoning_by_expanding_reasoning_steps_through_.md)

</div>

<!-- RELATED:END -->
