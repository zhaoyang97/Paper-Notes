---
title: >-
  [论文解读] ClaimPKG: Enhancing Claim Verification via Pseudo-Subgraph Generation with Lightweight Specialized LLM
description: >-
  [ACL 2025][模型压缩][知识图谱] 提出 ClaimPKG 框架，通过轻量级专用 LLM 将文本声明转换为伪子图表示，再从知识图谱中检索相关子图作为证据，最终由通用 LLM 进行推理验证，在 FactKG 数据集上比 SOTA 高出 9%-12% 准确率。 领域现状：声明验证（Claim Verification）…
tags:
  - "ACL 2025"
  - "模型压缩"
  - "知识图谱"
  - "声明验证"
  - "伪子图生成"
  - "Trie约束解码"
  - "LLM推理"
---

# ClaimPKG: Enhancing Claim Verification via Pseudo-Subgraph Generation with Lightweight Specialized LLM

**会议**: ACL 2025  
**arXiv**: [2505.22552](https://arxiv.org/abs/2505.22552)  
**代码**: [https://github.com/HoangHoang1408/ClaimPKG](https://github.com/HoangHoang1408/ClaimPKG)  
**领域**: 模型压缩  
**关键词**: 知识图谱, 声明验证, 伪子图生成, Trie约束解码, LLM推理

## 一句话总结

提出 ClaimPKG 框架，通过轻量级专用 LLM 将文本声明转换为伪子图表示，再从知识图谱中检索相关子图作为证据，最终由通用 LLM 进行推理验证，在 FactKG 数据集上比 SOTA 高出 9%-12% 准确率。

## 研究背景与动机

**领域现状**：声明验证（Claim Verification）是对抗虚假信息传播的关键技术，需要系统具备从外部知识源中检索证据并进行推理的能力。现有方法主要基于非结构化文本语料库。

**现有痛点**：
   - 大多数验证方法依赖非结构化文本，通过 CoT 推理分解声明，但文本的固有局限性导致实体歧义和多跳关系难以有效处理
   - 知识图谱（KG）虽然提供了结构化关系，但现有 KG 方法缺乏端到端解决方案，通常需要预先提取实体
   - LLM 虽有强大推理能力，但在实体消歧和多跳推理等 KG 特有任务上表现不佳

**核心矛盾**：如何在一个统一的框架中同时利用 KG 的结构化表示优势和 LLM 的推理能力？现有方法要么只用文本（缺乏结构化推理），要么 KG 方法过于模块化（缺乏端到端整合）。

**本文目标**：三大限制——(1) 实体歧义：系统必须准确消歧声明中的实体；(2) 多跳推理：复杂声明需要跨多个证据源推理；(3) KG 与 LLM 整合有限。

**切入角度**：引入"伪子图"作为桥梁，用轻量级专用 LLM 将文本声明转化为图结构表示，再通过检索算法在 KG 中找到真实证据子图。

**核心 idea**：用专用小模型生成伪子图 + Trie 约束确保实体正确性 + 通用 LLM 做最终推理，实现声明到 KG 子图的无缝对接。

## 方法详解

### 整体框架

ClaimPKG 包含三个阶段：
1. **伪子图生成（Pseudo-Subgraph Generation）**：KG 专用的轻量级 LLM 在 Trie 约束下生成伪子图
2. **子图检索（Subgraph Retrieval）**：检索算法以伪子图为查询，在 KG 中找到相关的真实子图作为证据
3. **通用推理（General Reasoning）**：通用 LLM 对声明和检索到的子图进行推理，生成判定和解释

数学框架上，将 $p_\theta(v,j|c,\mathcal{G})$ 分解为：

$$p_\theta(v,j|c,\mathcal{G}) = \sum_{\mathcal{S}_c} p_\theta(v,j|c,\mathcal{S}_c) \cdot p_\theta(\mathcal{S}_c|c,\mathcal{G})$$

进一步将子图选择分解为通过伪子图 $\mathcal{P}_c$ 的两步过程：

$$p_\theta(\mathcal{S}_c|c,\mathcal{G}) = \sum_{\mathcal{P}_c} p_\theta(\mathcal{S}_c|\mathcal{P}_c,\mathcal{G}) \cdot p_\theta(\mathcal{P}_c|c,\mathcal{G})$$

### 关键设计

#### 1. 专用 LLM + Trie 约束解码

- **功能**：将文本声明转换为由三元组 $(e, r, e')$ 组成的伪子图
- **核心思路**：微调一个轻量 LLM（如 Llama-3.2-3B）进行联合实体-关系抽取。对于间接引用的实体（未显式命名），用 $\text{unknown}_i$ 标记，信号化后续消歧需求
- **Trie 约束**：构建 KG 实体集的 Trie 树 $\mathcal{T}$，在生成实体时（`<e>` 到 `</e>` 之间）限制 token 选择只能沿 Trie 路径，确保生成的实体 100% 存在于 KG 中
- **多表示**：使用 beam search（beam size=5）生成多个伪子图 $\mathbb{P}_c = \{\mathcal{P}_c^{(i)}\}_{i=1}^N$，提高三元组覆盖率
- **设计动机**：通用 LLM 在 KG 实体抽取上表现不佳（实验证明 70B few-shot 的实体正确率仅 86.52%），而微调 3B 模型 + Trie 约束可达到 100% 正确率

#### 2. 子图检索算法

- **功能**：将伪子图中的三元组匹配到 KG 中的真实三元组
- **核心思路**：
    - 将伪三元组分为**不完整三元组**（含 unknown 实体）和**完整三元组**（两端实体均已知）
    - 不完整三元组：对每个 unknown 实体 $u$，收集与其相关的显式实体 $\mathcal{E}_u$ 的候选集，通过实体评分机制（公式5）选择最佳候选
    - 完整三元组：用关系相似度函数 $\text{Sim}(r_1, r_2)$ 在 KG 中找到两实体间最相似的 $k_2$ 个关系
- **关系评分函数**：使用 BGE-Large-EN-v1.5 编码计算点积相似度
- **参数设置**：$k_1=3$, $k_2=1$
- **设计动机**：伪子图作为文本到图结构的桥梁，解决了模态不匹配问题

#### 3. 通用推理模块

- **功能**：基于声明 $c$ 和检索到的证据子图 $\mathcal{S}_c^*$，生成判定 $v$ 和解释 $j$
- **核心思路**：使用通用 LLM（如 Llama-70B、Qwen-72B）进行 CoT 推理
- **公式**：$p_\theta(v,j|c,\mathcal{S}_c^*) = p_\theta(v|c,j,\mathcal{S}_c^*) \cdot p_\theta(j|c,\mathcal{S}_c^*)$
- **设计动机**：模型无关设计，可灵活集成不同 SOTA LLM

### 损失函数/训练策略

- 专用 LLM 训练：在 FactKG 训练集上微调，使用标准语言模型损失
- 训练数据量分析：仅需 100 个样本即可达到满意准确率（Llama-3.2-3B: 79.35%），5K 样本后趋于饱和
- General LLM 无需训练，零样本推理

## 实验关键数据

### 主实验

在 FactKG 数据集上的准确率对比（%）：

| 方法 | Negation | Existence | Conjunction | Multi-hop | One-hop | 平均 |
|------|----------|-----------|-------------|-----------|---------|------|
| Zero-shot CoT (Llama-70B) | 64.34 | 64.62 | 72.47 | 65.58 | 78.32 | 69.07 |
| GEAR (Finetuned BERT) | 79.72 | 79.19 | 78.63 | 68.39 | 77.34 | 76.65 |
| KG-GPT (Llama-70B) | 70.91 | 65.06 | 86.64 | 58.87 | 92.02 | 74.70 |
| **ClaimPKG (3B* + Qwen-72B)** | **85.27** | **86.90** | 84.02 | **78.71** | 91.20 | **85.22** |
| **ClaimPKG (3B* + Llama-70B)** | 84.58 | 84.20 | **85.68** | 78.49 | 90.26 | 84.64 |

### 消融实验

| 配置 | 实体正确率 | 平均准确率 |
|------|-----------|-----------|
| 完整 ClaimPKG | 100.0% | 84.64% |
| 去掉 Trie 约束 | 87.50% | 82.74% (-1.90) |
| Few-shot 替代专用 LLM | 86.52% | 77.63% (-7.01) |
| 去掉不完整三元组检索 | 100.0% | 65.08% (-19.56) |

### 关键发现

1. **证据检索至关重要**：纯 LLM CoT 最高仅 69.07%，远低于使用证据的方法
2. **专用小模型 > 通用大模型**：微调 1B 专用 LLM 优于 70B 通用 LLM 的 few-shot（83.91% vs 77.63%）
3. **伪子图提升 12 个点**：ClaimPKG 比 KG-GPT 高 12%，比 GEAR 高 9%
4. **零样本迁移**：在 HoVer 和 FEVEROUS 上比 Llama-70B CoT 高约 4%
5. **错误分析**：200 个错误中 0% 是结构错误、28.5% 是检索错误、71.5% 是推理错误

## 亮点与洞察

1. **伪子图是关键创新**：通过中间表示解决了文本-图结构的模态不匹配问题，比直接让 LLM 处理 KG 有效得多
2. **Trie 约束的优雅设计**：在保证实体 100% 正确的同时允许关系自由生成，兼顾了精确性和灵活性
3. **可扩展性好**：KG 更新时只需更新 Entity-Trie，无需重新训练
4. **样本效率高**：100 个训练样本即可达到满意效果，训练成本极低

## 局限与展望

1. 推理错误占 71.5%，通用 LLM 在复杂推理场景仍有不足，需要增强推理模块
2. 训练样本过多（>5K）会导致过拟合，需要正则化策略
3. 检索错误（28.5%）说明直接子图检索无法提供完整证据，需要隐式推理能力
4. 当前仅在 DBpedia 上验证，对其他 KG 的泛化能力有待验证
5. LLM 本身的偏差可能影响事实核查系统的可靠性

## 相关工作与启发

- **ProgramFC & FOLK**：基于文本的模块化验证管道，ClaimPKG 统一了这些步骤
- **KG-GPT**：先前 KG+LLM 方法的代表，但管道式设计导致性能受限
- **StructGPT & RoG**：在 KBQA 等相关任务上的 KG-LLM 结合工作，启发了 ClaimPKG 的设计
- 启发：伪子图的思想可以推广到其他需要文本-图结构对齐的任务

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 伪子图+Trie约束的组合设计新颖且有效
- **实验充分度**: ⭐⭐⭐⭐⭐ — 多基线、消融、泛化、错误分析、backbone对比非常全面
- **写作质量**: ⭐⭐⭐⭐ — 数学框架清晰，但部分描述较冗长
- **价值**: ⭐⭐⭐⭐ — 对 KG 增强的事实核查领域有较好的推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Traversal Verification for Speculative Tree Decoding](../../NeurIPS2025/model_compression/traversal_verification_for_speculative_tree_decoding.md)
- [\[ICLR 2026\] LightMem: Lightweight and Efficient Memory-Augmented Generation](../../ICLR2026/model_compression/lightmem_lightweight_and_efficient_memory-augmented_generation.md)
- [\[ACL 2025\] Predicting Through Generation: Why Generation Is Better for Prediction](predicting_through_generation_why_generation_is_better_for_prediction.md)
- [\[ACL 2025\] Direct Behavior Optimization: Unlocking the Potential of Lightweight LLMs](direct_behavior_optimization_unlocking_the_potential_of_lightweight_llms.md)
- [\[ACL 2025\] Bone Soups: A Seek-and-Soup Model Merging Approach for Controllable Multi-Objective Generation](bone_soups_multi_objective_gen.md)

</div>

<!-- RELATED:END -->
