---
title: >-
  [论文解读] Towards Intrinsic Interpretability of Large Language Models: A Survey of Design Principles and Architectures
description: >-
  [ACL 2026][可解释性][内在可解释性] 系统综述了 LLM 内在可解释性的最新进展，将现有方法分为五大设计范式（功能透明性、概念对齐、表征可分解性、显式模块化、潜在稀疏归纳），并讨论了开放挑战和未来方向。 领域现状：大语言模型在各类 NLP 任务上取得了显著成功，但其内部机制的不透明性（黑盒特性）阻碍了可信部署…
tags:
  - "ACL 2026"
  - "可解释性"
  - "内在可解释性"
  - "大语言模型"
  - "设计范式分类"
  - "模块化架构"
  - "稀疏归纳"
---

# Towards Intrinsic Interpretability of Large Language Models: A Survey of Design Principles and Architectures

**会议**: ACL 2026  
**arXiv**: [2604.16042](https://arxiv.org/abs/2604.16042)  
**代码**: 无  
**领域**: 可解释性  
**关键词**: 内在可解释性, 大语言模型, 设计范式分类, 模块化架构, 稀疏归纳

## 一句话总结

系统综述了 LLM 内在可解释性的最新进展，将现有方法分为五大设计范式（功能透明性、概念对齐、表征可分解性、显式模块化、潜在稀疏归纳），并讨论了开放挑战和未来方向。

## 研究背景与动机

**领域现状**：大语言模型在各类 NLP 任务上取得了显著成功，但其内部机制的不透明性（黑盒特性）阻碍了可信部署，尤其在医疗、法律等高风险领域。现有可解释 AI 综述主要聚焦于事后解释方法（post-hoc），如 LIME、SHAP、稀疏自编码器、因果干预等。

**现有痛点**：事后解释方法通过外部近似来解释已训练好的模型，存在"保真度鸿沟"——解释和模型的真实计算之间存在根本性偏差。即使是因果干预方法（如 ROME），虽然局部保真度更强，但其解释粒度过细，难以聚合为对模型整体行为的连贯理解。

**核心矛盾**：历史上，内在可解释的模型（如线性模型、决策树）在表达能力上远不及黑盒大模型，导致"可解释性 vs 性能"被视为不可调和的 trade-off。但近期研究表明，通过将模块化、稀疏性、解纠缠等归纳偏置嵌入现代架构，这一 trade-off 正在被打破。

**本文目标**：为内在可解释性方法提供统一的分类框架，系统梳理设计原则，明确各方法的优劣和适用场景，并指出未来研究方向。

**切入角度**：不同于事后解释综述从"工具"出发，本文从"设计原则"出发，关注如何从架构和训练过程中构建透明性。

**核心 idea**：将内在可解释性方法组织为五大设计范式，每个范式代表一种不同的"透明性来源"。

## 方法详解

### 整体框架

本文不从"事后解释工具"出发，而是以"透明性的来源"为主轴，把内在可解释方法组织成一套五范式的分类体系：功能透明性、概念对齐、表征可分解性、显式模块化、潜在稀疏归纳。这五个范式回答的是同一个问题的不同侧面——可解释性究竟应该嵌进计算过程本身、表征空间、还是网络结构里。下文以前三个范式为代表展开其设计原则（后两个范式归入实验对比表统一讨论），并在最后梳理各范式的训练成本谱系。

### 关键设计

**1. 功能透明性（Functional Transparency）：让每一步计算本身就可读**

这是最直接的透明性来源——如果计算过程本身就是透明的，就不再需要任何外部近似工具去事后解释。代表方法包括广义加性模型（GAMs）及其扩展（GA2M、EBMs、GAMI-Net），它们用可加性约束把每个特征的贡献单独拆出来可视化；自解释神经网络（SENN）把预测分解为基础概念与对应的相关性分数；B-cos 网络通过权重-输入对齐变换让前向计算等价于一个线性解释；Kolmogorov-Arnold Networks（KANs）则用可学习的样条函数替换固定激活，使每条边上的形状函数都可读。代价是可加性约束限制了建模能力，而 KANs 能否扩展到大规模 LLM 也尚未得到验证。

**2. 概念对齐（Concept Alignment）：把内部表征绑定到人类概念**

概念是人类思维的基本单位，因此把模型的中间表征对齐到人类可理解的概念，往往能给出最自然的解释。概念瓶颈模型（CBMs）在中间层强制预测一组人类定义的概念，再仅基于这些概念做最终预测；CB-LLM 把这一思路搬到 LLM 上，靠混合瓶颈加对抗训练在引入瓶颈的同时保住性能；Label-free CBM 借助 CLIP 自动发现概念，绕开人工标注；Codebook Features 则用向量量化得到离散化的概念编码。主要隐患是概念定义通常需要领域专家，而混合瓶颈中的残差通道可能泄露信息、绕过瓶颈，从而削弱解释的保真度。

**3. 表征可分解性（Representational Decomposability）：在表征层面拆出独立可读的分量**

这一范式不改动整体架构，只在表征空间里引入分解结构。Backpack 语言模型为每个词学习多个"含义向量"（sense vectors），再用上下文权重加权组合，于是可以追踪某个词在当前语境下究竟激活了哪一个含义；CoCoMix 则在训练中预测连续概念并把它们混合进表征，使概念级别的信息在整个前向过程中保持可追溯。它的好处是改动局部、兼容现有架构，代价是 Backpack 这类含义向量机制会带来额外的推理开销。

### 损失函数 / 训练策略

本文为综述，不涉及具体训练。但它总结了各范式的训练成本特征：功能透明性和概念对齐方法训练成本低-中，显式模块化（MoE）方法成本中-高，潜在稀疏归纳（如 $L_0$ 正则化）成本极高。

## 实验关键数据

### 主实验

综合对比表（节选自 Table 1）：

| 方法类别 | 代表方法 | 可解释性来源 | 训练成本 | 推理成本 | 性能影响 |
|---------|---------|------------|---------|---------|---------|
| 功能透明性 | KANs, B-cos LMs | 形状函数/线性解释 | 中-高 | 中-高 | ≈ 基线 |
| 概念对齐 | CB-LLM, CBMs | 概念分数 | 高 | 低 | ↓ 或 ≈ |
| 表征可分解 | Backpack, CoCoMix | 含义向量/连续概念 | 中 | 高 | ↓ 或 ≈ |
| 显式模块化 | MoE-X, MONET | 稀疏专家/单义专家 | 低-高 | 低-中 | ≈ 或 ↑ |
| 稀疏归纳 | Weight-Sparse, GLU | 稀疏电路/激活路径 | 极高/低 | 低 | ↓ 或 ≈ |

### 消融实验

各范式可解释性-性能 trade-off 对比：

| 范式 | 保真度 | 粒度 | 可扩展性 | 性能保持 |
|------|-------|------|---------|---------|
| 功能透明性 | 最高 | 特征级 | 差 | 中 |
| 概念对齐 | 高 | 概念级 | 中 | 中 |
| 表征可分解 | 中 | 词/概念级 | 中 | 中 |
| 显式模块化 | 中 | 专家/路由级 | 好 | 好 |
| 稀疏归纳 | 中-高 | 电路/神经元级 | 好 | 中 |

### 关键发现

- 显式模块化（MoE 类方法）在可扩展性和性能保持方面最有优势，是目前最有前景的范式
- 功能透明性方法保真度最高但可扩展性最差，难以直接应用于数十亿参数的 LLM
- 概念对齐方法依赖人工概念定义，CB-LLM 开始探索自动概念发现但仍处于早期
- $L_0$ 正则化产生的权重稀疏模型虽然电路可解释，但训练成本极高（约 3x 标准训练）
- GLU/SwiGLU 是"免费"的稀疏归纳——几乎所有现代 LLM 已在使用，但其可解释性潜力尚未被充分挖掘

## 亮点与洞察

- **五范式分类框架**非常清晰实用——将分散的文献统一在共同的设计原则下，便于研究者定位自己的工作和发现研究空白
- **"可解释性不一定牺牲性能"的论证**有力——MoE-X、B-cos LMs 等方法表明，精心设计的归纳偏置可以在保持性能的同时提供可解释性
- **跨范式组合的潜力**被明确指出——例如将概念对齐与显式模块化结合（概念瓶颈 + MoE），或表征可分解与稀疏归纳结合，开辟了广阔的研究空间

## 局限与展望

- 大部分内在可解释方法仅在中小规模模型上验证，是否能扩展到百亿/千亿参数 LLM 尚不确定
- 缺乏统一的可解释性评估指标——不同方法的"可解释性"定义和衡量标准不一致
- 对多模态大模型的内在可解释性研究几乎空白
- 未来方向包括：可解释性与安全对齐的结合、可解释的推理链路追踪、动态稀疏激活的可解释性分析

## 相关工作与启发

- **vs 事后解释综述（Madsen et al., 2022; Zhao et al., 2024）**: 这些综述聚焦于分析已训练模型的工具（如探针、注意力可视化），本文关注从设计层面构建透明性
- **vs 机制可解释性（Sharkey et al., 2025）**: 机制可解释性是事后方法中最接近内在可解释的方向，但仍是"逆向工程"而非"正向设计"

## 评分

- 新颖性: ⭐⭐⭐⭐ 五范式分类框架是新贡献，但综述本身不提出新方法
- 实验充分度: ⭐⭐⭐ 综述论文，无原创实验，但 Table 1 的对比整理很有参考价值
- 写作质量: ⭐⭐⭐⭐⭐ 分类清晰，覆盖全面，适合作为该领域的入门指南
- 价值: ⭐⭐⭐⭐ 为快速增长的内在可解释性领域提供了急需的结构化框架

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Prototype Transformer: Towards Language Model Architectures Interpretable by Design](../../ICML2026/interpretability/prototype_transformer_towards_language_model_architectures_interpretable_by_desi.md)
- [\[ACL 2026\] Knowledge Vector of Logical Reasoning in Large Language Models](knowledge_vector_of_logical_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Compositional Steering of Large Language Models with Steering Tokens](compositional_steering_of_large_language_models_with_steering_tokens.md)
- [\[ACL 2026\] Experiments or Outcomes? Probing Scientific Feasibility in Large Language Models](experiments_or_outcomes_probing_scientific_feasibility_in_large_language_models.md)
- [\[ICML 2026\] Dual Mechanisms of Value Expression: Intrinsic vs. Prompted Values in Large Language Models](../../ICML2026/interpretability/dual_mechanisms_of_value_expression_intrinsic_vs_prompted_values_in_large_langua.md)

</div>

<!-- RELATED:END -->
