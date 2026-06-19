---
title: >-
  [论文解读] SHAP Meets Tensor Networks: Provably Tractable Explanations with Parallelism
description: >-
  [NeurIPS 2025][音频/语音][SHAP] 本文首次为张量网络（Tensor Networks）提供可证明精确的 SHAP 解释计算框架，证明张量列车（Tensor Train）结构下 SHAP 可在多对数时间内并行计算（NC² 复杂度），并通过归约揭示二值化神经网络中**宽度而非深度**才是 SHAP 计算的核心瓶颈。
tags:
  - "NeurIPS 2025"
  - "音频/语音"
  - "SHAP"
  - "Tensor Networks"
  - "Tensor Train"
  - "可解释性"
  - "并行计算"
  - "二值化神经网络"
  - "参数化复杂度"
---

# SHAP Meets Tensor Networks: Provably Tractable Explanations with Parallelism

**会议**: NeurIPS 2025  
**arXiv**: [2510.21599](https://arxiv.org/abs/2510.21599)  
**代码**: 待确认  
**领域**: 音频语音  
**关键词**: SHAP, Tensor Networks, Tensor Train, 可解释性, 并行计算, 二值化神经网络, 参数化复杂度  

## 一句话总结

本文首次为张量网络（Tensor Networks）提供可证明精确的 SHAP 解释计算框架，证明张量列车（Tensor Train）结构下 SHAP 可在多对数时间内并行计算（NC² 复杂度），并通过归约揭示二值化神经网络中**宽度而非深度**才是 SHAP 计算的核心瓶颈。

## 背景与动机

1. **SHAP 计算的不可解性**：Shapley 值作为最主流的事后解释方法，对黑箱模型（如神经网络）的精确计算是 NP-Hard 的，现有精确算法仅适用于决策树等简单模型，表达力远不够。
2. **近似方法的局限**：KernelSHAP、DeepSHAP、FastSHAP 等近似方法通过采样或启发式降低计算成本，但无法提供精确性保证，在高风险应用场景中缺乏可信度。
3. **张量网络的独特优势**：张量网络源自量子物理，兼具高表达力和结构化透明性——其函数逼近能力可媲美某些神经网络族，而物理启发的结构又便于推导高效精确解。
4. **模型压缩与解释性的双重需求**：张量网络广泛用于神经网络的抽象与压缩，若能为其提供精确 SHAP 算法，则同时解决了模型压缩后的可解释性问题。
5. **并行计算的理论空白**：此前文献从未从计算复杂度理论角度分析 SHAP 的**可并行性**——即何时 SHAP 可在多对数时间内完成，这对大规模部署至关重要。
6. **神经网络 SHAP 困难的细粒度分析缺失**：已知神经网络 SHAP 是 NP-Hard，但哪些结构参数（深度 vs 宽度 vs 稀疏度）是计算瓶颈，此前缺乏细粒度研究。

## 方法详解

### 整体框架：一般张量网络的 SHAP 计算

- **功能**：为任意结构的张量网络（TN）提供可证明精确的 SHAP 计算框架。
- **为什么**：SHAP 公式涉及对所有特征子集求和（指数级），直接计算不可行；需要利用 TN 的结构来避免指数级枚举。
- **怎么做**：
  1. **张量化表示**（Proposition 1）：将 SHAP 公式重写为**修正加权联盟张量** $\tilde{\mathcal{W}}$ 和**边际值张量** $\mathcal{V}^{(M,P)}$ 的收缩运算，即 $\mathcal{T}^{(M,P)} = \tilde{\mathcal{W}} \times_S \mathcal{V}^{(M,P)}$。
  2. **构造加权联盟张量**（Lemma 1）：$\tilde{\mathcal{W}}$ 可表示为稀疏张量列车，核心维度为 $n_{in}^2$，可在 $O(\log n_{in})$ 时间用 $O(n_{in}^3)$ 并行处理器构造。
  3. **构造边际值张量**（Lemma 2）：引入"路由张量" $\mathcal{M}^{(i)}$ 模拟介入机制——根据子集 $S$ 中特征是否包含，决定使用输入实例值或分布期望值，再与模型 TN 和分布 TN 收缩。
  4. **最终 SHAP 矩阵**：对输入 $x$ 的 SHAP 值通过边际 SHAP 张量与输入的秩-1 张量收缩得到（Equation 3）。

### 关键设计 1：张量列车（TT）的高效并行 SHAP

- **功能**：证明当模型和分布均为 TT 结构时，SHAP 计算属于 NC² 复杂度类。
- **为什么**：一般 TN 的 SHAP 是 #P-Hard（Proposition 2），需要限制结构才能获得高效算法；TT 是最常用的 TN 子族，具有一维链式拓扑。
- **怎么做**：
  1. **核心定理**（Theorem 1）：证明当 $\mathcal{T}^M$ 和 $\mathcal{T}^P$ 均为 TT 时，边际 SHAP 张量本身也是 TT，其核心由 $\mathcal{M}^{(i)} \times \mathcal{I}^{(i)} \times \mathcal{P}^{(i)} \times \mathcal{G}^{(i)}$ 的收缩构成。
  2. **并行扫描**（Proposition 3）：TT 收缩可通过经典的并行前缀扫描（parallel scan）以 $O(\log^2 n_{in})$ 深度完成，结合矩阵乘法属于 NC¹ 的结论，整体属于 NC²。
  3. **归约其他模型**（Theorem 2）：决策树、树集成、线性模型、线性 RNN 均可在 NC 内归约为等价 TT，从而继承 NC² 的 SHAP 复杂度——这收紧了此前所有已知的多项式时间结果。

### 关键设计 2：二值化神经网络（BNN）的细粒度复杂度分析

- **功能**：通过参数化复杂度理论，分析 BNN 的深度、宽度、稀疏度对 SHAP 计算难度的影响。
- **为什么**：此前仅知"神经网络 SHAP 是 NP-Hard"，但未区分哪些结构参数是瓶颈，无法指导网络设计以获得可解释性。
- **怎么做**（Theorem 3）：
  1. **固定深度仍困难**（para-NP-Hard）：通过从 3SAT 归约——任何 CNF 公式可编码为深度-2 BNN，而 CNF 的 SHAP 是 #P-Hard。
  2. **固定宽度可解**（XP）：将 BNN 逐层转换为 2D 张量网络，再向后收缩得到等价 TT，编译时间 $O(R^W \cdot \text{poly}(D, n_{in}))$，对固定 $W$ 是多项式。
  3. **固定宽度+稀疏度高效**（FPT）：进一步约束 reified cardinality $R$ 后，组合爆炸被限制在 $R^W$ 项中，运行时间为 $g(W,R) \cdot n^{O(1)}$，属于 FPT。

## 实验关键数据

本文为**理论工作**，核心贡献是复杂度证明而非实验评估。但给出了以下关键的理论结果总结：

| 模型族 | 此前已知复杂度 | 本文结果 | 分布类 |
|:---|:---|:---|:---|
| 决策树 | P（TreeSHAP） | **NC²** | TT 分布（含独立/经验/马尔可夫/HMM/Born Machine） |
| 树集成 | P | **NC²** | TT 分布 |
| 线性模型 | P | **NC²** | TT 分布 |
| 线性 RNN | P | **NC²** | TT 分布 |
| 张量列车 | 首次研究 | **NC²** | TT 分布 |
| BNN（固定深度） | — | **para-NP-Hard** | 经验/独立/TT |
| BNN（固定宽度） | — | **XP** | 经验/独立/TT |
| BNN（固定宽度+稀疏度） | — | **FPT** | 经验/独立/TT |

**关键发现**：

- 对所有经典 ML 模型，SHAP 不仅多项式可解，还可在多对数时间内并行完成。
- BNN 中，宽度（而非深度）是 SHAP 可解性的关键分界线：固定深度仍 NP-Hard，固定宽度则可解。
- TT 所支持的分布类远比此前算法使用的独立/经验分布更具表达力，覆盖了复杂的特征依赖关系。

## 亮点

1. **首个张量网络 SHAP 精确算法**，弥合了高表达力模型与精确解释之间的鸿沟。
2. **首次分析 SHAP 的并行可计算性**，将 TreeSHAP 等经典结果从 P 收紧至 NC²，为大规模并行部署奠定理论基础。
3. **BNN 的 "宽度 > 深度" 洞察**极具启发性——表明设计窄且稀疏的二值化网络可同时获得压缩和可解释性。
4. **统一框架**：通过 TT 归约统一处理决策树、线性模型、RNN 等多种模型族的 SHAP 复杂度分析。

## 局限与展望

1. **纯理论贡献**：缺乏实际 SHAP 计算的实验验证（运行时间、GPU 加速效果等），实际部署的可行性待考察。
2. **仅限二值化神经网络**：BNN 的结论无法直接推广到全精度网络——实际场景中全精度模型更常见。
3. **TT 结构的限制**：一般 TN 的 SHAP 仍是 #P-Hard，TT 要求一维链式拓扑，某些模型族可能无法高效归约到 TT。
4. **未涉及其他解释方法**：如 Banzhaf 值、LIME 等替代解释方法的复杂度比较未探讨。
5. **分布假设**：TT 分布虽表达力强，但实际数据分布是否能被 TT 良好近似需要进一步研究。

## 与相关工作的对比

### vs TreeSHAP (Lundberg et al., 2020)
TreeSHAP 为决策树和树集成提供多项式时间精确 SHAP 算法，但仅适用于独立特征分布或树模型内部结构。本文通过 TT 归约将复杂度从 P 收紧至 NC²，且支持更广泛的 TT 分布类（含马尔可夫、HMM 等依赖结构），在分布表达力和计算效率上均有理论提升。

### vs Van den Broeck et al. (2022) / Arenas et al. (2024)
这些工作探索了布尔电路/可判定分解图（d-DNNF）等知识编译形式下的 SHAP 复杂度。本文的 TN 框架在表达力上超越了这些形式（TT 可编码决策树和线性模型等多种模型），且首次引入并行复杂度分析，提供了更细粒度的计算理论视角。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （首次将张量网络理论与SHAP计算深度结合，开创性地分析并行复杂度）
- 实验充分度: ⭐⭐ （纯理论工作，无实验验证，但理论证明严谨完整）
- 写作质量: ⭐⭐⭐⭐ （形式化严谨，图示清晰；部分证明过于简略需看附录）
- 价值: ⭐⭐⭐⭐ （对XAI理论研究有重要推动作用，BNN宽度洞察具有实践指导意义）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Toward Complex-Valued Neural Networks for Waveform Generation](../../ICLR2026/audio_speech/toward_complex-valued_neural_networks_for_waveform_generation.md)
- [\[ACL 2026\] Indic-CodecFake meets SATYAM: Towards Detecting Neural Audio Codec Synthesized Speech Deepfakes in Indic Languages](../../ACL2026/audio_speech/indic-codecfake_meets_satyam_towards_detecting_neural_audio_codec_synthesized_sp.md)
- [\[NeurIPS 2025\] Node-Based Editing for Multimodal Generation of Text, Audio, Image, and Video](node-based_editing_for_multimodal_generation_of_text_audio_image_and_video.md)
- [\[NeurIPS 2025\] Target Speaker Extraction Through Comparing Noisy Positive and Negative Audio Enrollments](target_speaker_extraction_through_comparing_noisy_positive_and_negative_audio_en.md)
- [\[NeurIPS 2025\] Accelerate Creation of Product Claims Using Generative AI](accelerate_creation_of_product_claims_using_generative_ai.md)

</div>

<!-- RELATED:END -->
