---
title: >-
  [论文解读] How Do Transformers Learn to Associate Tokens: Gradient Leading Terms Bring Mechanistic Understanding
description: >-
  [ICLR 2026][可解释性][Transformer] 通过对训练梯度的前导项近似分析，推导出Transformer在训练早期阶段各权重矩阵的闭式表达——均可分解为三种基函数（bigram、token-interchangeability、context mapping）的简单组合——从而揭示Transformer如何从自然语言数据中学习"bird"↔"flew"这类语义关联，且理论预测与真实LLM的学到权重高度吻合。
tags:
  - "ICLR 2026"
  - "可解释性"
  - "Transformer"
  - "训练动态"
  - "梯度前导项"
  - "语义关联"
  - "闭式权重表达"
---

# How Do Transformers Learn to Associate Tokens: Gradient Leading Terms Bring Mechanistic Understanding

**会议**: ICLR 2026  
**arXiv**: [2601.19208](https://arxiv.org/abs/2601.19208)  
**代码**: 无  
**领域**: LLM / NLP (Mechanistic Interpretability)  
**关键词**: Transformer可解释性, 训练动态, 梯度前导项, 语义关联, 闭式权重表达

## 一句话总结
通过对训练梯度的前导项近似分析，推导出Transformer在训练早期阶段各权重矩阵的闭式表达——均可分解为三种基函数（bigram、token-interchangeability、context mapping）的简单组合——从而揭示Transformer如何从自然语言数据中学习"bird"↔"flew"这类语义关联，且理论预测与真实LLM的学到权重高度吻合。

## 研究背景与动机

**领域现状**："bird"和"flew"之间的语义关联（semantic association）是语言建模的基础——模型靠它超越死记硬背、实现泛化和连贯文本生成。理解这些关联如何在语言模型中被学习和表示，是连接深度学习与语言学理论、为大型语言模型建立机制性基础的关键。

**现有痛点**：现有 Transformer 可解释性研究大体分两条路线——(1) 分析已训练好模型的内部表示（如 attention 可视化、probing）；(2) 在简化模型/合成任务上做理论分析（如单头 attention、合成结构语言、去掉位置编码或残差连接）。两条路线要么只看训练后的静态结果、不解释"怎么学到的"，要么用了和真实训练相距甚远的假设，结论难以推广到实际 LLM。

**核心矛盾**：我们缺乏一套机制性理解，去说明 Transformer 是如何在**真实自然语言数据上、用标准训练流程**逐步学到语义关联的——而这恰恰发生在训练过程当中，被静态分析和过度简化的理论都绕开了。

**本文切入点**：从**训练动态（training dynamics）**视角切入，在标准架构（含相对位置编码、因果掩码、残差流）上利用梯度的**前导项近似（leading-term approximation）**，推导出可解析、可验证的权重闭式表达，刻画训练早期权重如何成形。

**核心洞察**：Transformer 的每组权重在训练早期都能写成三种反映语料统计特性的基函数的简单组合，每种基函数对应一类语义关联的学习机制。

## 方法详解

### 整体框架

本文不训练新模型、也不去剖析已训练好的网络内部，而是把分析对象放在**训练过程本身**：对一个标准 attention 语言模型（Transformer）在自然语言语料上做自回归训练，逐步追踪每组权重矩阵（$W_Q$、$W_K$、$W_V$、embedding 等）是怎样被梯度一点点塑造出来的。核心工具是对梯度更新做前导项（leading-term）近似，最终把每组权重在训练早期阶段写成三种语料统计基函数的闭式组合，从而把"模型学到了什么语义关联"翻译成"语料里有什么统计规律"。

### 关键设计

**1. 梯度前导项近似：把不可解的训练动态变成可解析的统计量**

直接追踪 Transformer 完整的梯度更新几乎得不到闭式结果，因为梯度里混杂着权重之间的高阶耦合项。本文的做法是对梯度做展开后只保留对权重更新贡献最大的主导项（leading term），把高阶项丢掉。这一近似之所以成立，关键在于训练早期权重尚小，高阶项（权重的高次幂）相对前导项可以忽略；而一旦只看前导项，梯度就不再依赖随机初始化的细节，可以直接用语料库的共现统计量来表达。换句话说，训练早期权重往哪个方向走，主要由数据的统计结构决定，而非优化轨迹的偶然性——这正是后面能写出闭式解的前提。把分析锚在训练早期也有实证支撑：诱导头、线性语义关系等核心能力本就在早期成形并延续到收敛，所以早期既重要又恰好可解。

**2. 三种基函数分解：把语义关联拆成三条统计通路**

前导项分析的结论是，每组权重都能写成三种基函数的组合，每种对应一类语义关联的来源。**Bigram 映射**捕捉相邻 token 的依赖——当 token $A$ 频繁出现在 $B$ 之前时，权重朝着加强 $A \rightarrow B$ 关联的方向增长，这是最直接的下一 token 统计（如"the"→"cat"）。**可互换性映射（token-interchangeability mapping）**捕捉功能相似：像"car"和"truck"并不直接相邻共现，但它们出现在相似的上下文（如"The ___ sat on the mat"）里、扮演相同的语法角色，于是被赋予相近的表示——这恰好是分布式语义假说（distributional semantics）在训练动态里的数学化身。**Context 映射**则编码更长程的前缀-后缀共现，即"给定一段上下文，什么 token 最可能跟随"的条件统计。三者各管一条通路，"bird"↔"flew"这类关联正是这三条通路叠加涌现的结果，而非被某处直接记下。

**3. 闭式权重表达与功能性分工：让理论可被真实权重检验**

把三种基函数代回，每个权重矩阵（输出矩阵、Value 矩阵、Query–Key 矩阵）在训练早期就能写成它们的线性组合，组合系数由架构细节（层数、头数）和训练超参数决定。这个闭式表达不只是形式整洁，它还暴露出各组件的分工：Query–Key 权重主要由 bigram 与可互换性映射主导，决定"该 attend 到哪里"；Value 权重则主要由 context 映射主导，决定"attend 到之后传递什么信息"。这种分工此前多停留在直觉层面，这里第一次有了可定量核对的闭式来源。也正因为权重被表达成纯由语料统计算出的量，理论预测才能与真实训练 LLM 的权重直接做定量对比——并构成从描述（写出表达式）、到解释（每项对应哪种统计）、再到预测（与实测权重吻合）三个层次的递进。

## 实验关键数据

### 理论 vs 真实权重对比

| 验证维度 | 结果 | 说明 |
|---------|------|------|
| 权重近似精度 | 高度吻合 | 理论闭式表达与实际训练权重的模式高度一致 |
| 在真实LLM上验证 | 成功 | 不仅在小模型上，在实际规模的LLM上也验证了理论预测 |
| 定性分析 | 可解释 | 闭式表达能够解释模型学到的具体语义关联模式 |

### 消融实验

| 分析维度 | 关键发现 | 说明 |
|---------|---------|------|
| 训练早期 vs 后期 | 早期近似更精确 | 符合前导项近似的理论预期 |
| 不同基函数贡献 | 三者均有显著贡献 | 移除任一基函数都会显著降低近似质量 |
| 不同权重矩阵 | 功能性分化 | Q/K权重更依赖bigram，V权重更依赖context |
| 不同层的行为 | 层间差异 | 浅层更偏向bigram，深层更偏向context |

### 关键发现
- **Transformer权重的"可分解性"**：复杂的权重矩阵可以分解为仅3种基于语料库统计的简单基函数的组合，这大幅简化了对Transformer学习机制的理解
- **语义关联的涌现机制**：语义关联不是被直接编码的，而是通过bigram共现统计、分布式可互换性和上下文模式的交互作用涌现出来的
- **功能性分化**：Q/K矩阵和V矩阵在语义关联编码中扮演本质不同的角色，这与直觉一致但首次有了理论支撑
- **理论预测的实用性**：闭式表达在实际LLM上得到验证，说明这不仅是理论上的优美结果，而是具有实际解释力的工具

## 亮点与洞察
- **从训练动态视角理解Transformer**：不同于多数可解释性工作分析"训练好的模型做了什么"，本文分析"模型是如何学到这些的"——这提供了更根本的理解
- **三基函数分解的简洁性**：将高维复杂的权重矩阵归结为3种有清晰统计语义的基函数，既优美又有解释力，是连接深度学习与计算语言学的桥梁
- **分布式语义假说的理论证据**：token-interchangeability基函数直接对应"在相似上下文中出现的词具有相似语义"的语言学假说，本文从训练动态的纯数学分析中自然导出了这一结论
- **定量验证而非仅定性描述**：不仅仅是理论推导，还在真实LLM上做了系统性的定量验证

## 局限与展望
- 前导项近似在训练后期（权重增大后）精度下降，对完全训练后的模型解释力有限
- 分析主要集中在训练早期阶段，训练中后期的非线性效应和特征复杂化未被充分捕捉
- 三种基函数的分解是否适用于更大规模的模型（数十亿参数级别）需要进一步验证
- 仅分析了语义关联这一特定能力，Transformer的其他能力（如推理、计划）是否也可以用类似框架分析？
- 闭式表达的实际应用场景（如指导模型初始化、架构设计或知识编辑）尚未被充分探索
- 对多层Transformer中层间交互和残差连接的影响分析有待深入

## 相关工作与启发
- **Mechanistic Interpretability**（Olah et al., 2020）：电路级别的Transformer可解释性分析
- **Induction Heads**（Olsson et al., 2022）：attention head级别的特定计算模式
- **Transformer training dynamics**（Li et al., 2023等）：分析Transformer训练过程的动态特性
- **Distributional Semantics**（Harris, 1954; Firth, 1957）：词的语义由其分布上下文决定——本文为这一经典假说提供了从神经网络训练动态角度的理论支撑
- **Feature Learning Theory**：近年来关于神经网络如何学习特征的理论工作（lazy training vs feature learning regime）
- 启发：梯度前导项分析是一种强大但被低估的分析工具，可能广泛应用于理解其他神经网络架构和任务。将训练动态与语料库统计直接联系起来的方法论，为"数据如何塑造模型"提供了新的分析框架

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ （梯度前导项分析 + 三基函数分解 = 全新理论框架）
- 实验充分度: ⭐⭐⭐⭐ （在真实LLM上验证，但可扩展到更大模型）
- 写作质量: ⭐⭐⭐⭐ （理论深度高，数学推导严谨）
- 价值: ⭐⭐⭐⭐⭐ （对Transformer机制性理解的重要理论贡献，连接深度学习与语言学）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] How Do Transformers Learn Implicit Reasoning?](../../NeurIPS2025/interpretability/how_do_transformers_learn_implicit_reasoning.md)
- [\[ICLR 2026\] How Transformers Learn Causal Structures In-Context: Explainable Mechanism Meets Theoretical Guarantee](how_transformers_learn_causal_structures_in-context_explainable_mechanism_meets_.md)
- [\[ICLR 2026\] From Tokens to Thoughts: How LLMs and Humans Trade Compression for Meaning](from_tokens_to_thoughts_how_llms_and_humans_trade_compression_for_meaning.md)
- [\[ICLR 2026\] Concept-TRAK: Understanding how diffusion models learn concepts through concept attribution](concept-trak_understanding_how_diffusion_models_learn_concepts_through_concept_a.md)
- [\[ICLR 2026\] Towards Understanding Subliminal Learning: When and How Hidden Biases Transfer](towards_understanding_subliminal_learning_when_and_how_hidden_biases_transfer.md)

</div>

<!-- RELATED:END -->
