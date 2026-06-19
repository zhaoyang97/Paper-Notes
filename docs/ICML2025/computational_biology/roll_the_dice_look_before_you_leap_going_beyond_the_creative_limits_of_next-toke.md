---
title: >-
  [论文解读] Roll the Dice & Look Before You Leap: Going Beyond the Creative Limits of Next-Token Prediction
description: >-
  [ICML2025 Oral][计算生物][next-token prediction] 本文设计了一套最小化算法任务来量化语言模型的"创造力极限"，证明 next-token 学习在需要"思维跳跃"的开放式任务中是近视的，而多 token 方法（teacherless 训练、离散扩散模型）以及输入层噪声注入（seed-conditioning）能显著提升生成的多样性与原创性。
tags:
  - "ICML2025 Oral"
  - "计算生物"
  - "next-token prediction"
  - "multi-token prediction"
  - "creativity"
  - "teacherless training"
  - "扩散模型"
  - "seed-conditioning"
  - "algorithmic creativity"
---

# Roll the Dice & Look Before You Leap: Going Beyond the Creative Limits of Next-Token Prediction

**会议**: ICML2025 Oral  
**arXiv**: [2504.15266](https://arxiv.org/abs/2504.15266)  
**代码**: [chenwu98/algorithmic-creativity](https://github.com/chenwu98/algorithmic-creativity)  
**领域**: LLM/NLP  
**关键词**: next-token prediction, multi-token prediction, creativity, teacherless training, diffusion, seed-conditioning, algorithmic creativity

## 一句话总结

本文设计了一套最小化算法任务来量化语言模型的"创造力极限"，证明 next-token 学习在需要"思维跳跃"的开放式任务中是近视的，而多 token 方法（teacherless 训练、离散扩散模型）以及输入层噪声注入（seed-conditioning）能显著提升生成的多样性与原创性。

## 研究背景与动机

当前语言模型在开放式创造性任务（如设计数学题、生成研究 idea、构造双关语）中表现受限。这类任务的核心特征是需要一个**隐式的、随机的"思维跳跃"（leap of thought）**——在生成输出前，模型需要预先规划多个相互关联的随机决策。

现有评估方法面临两大困难：(1) 真实任务的创造力评估高度主观；(2) 模型训练数据覆盖整个互联网，原创性难以界定。因此，作者转向**可控的算法任务**，设计了 loose abstraction 来客观量化模型的创造力。

借鉴认知科学文献 (Boden, 2003)，作者识别出两类基本创造模式：

- **组合式创造力 (Combinational Creativity)**：在已知关系图中发现新颖的多跳连接（如双关语需要搜索语义图找到连接两个不相关概念的"桥梁词"）
- **探索式创造力 (Exploratory Creativity)**：在规则约束下构建全新模式（如设计可解的数学题需要构造满足逻辑约束的新结构）

## 方法详解

### 算法创造力度量

给定模型生成的样本集 $T$，定义算法创造力指标：

$$\hat{\text{cr}}_N(T) = \frac{\text{uniq}(\{s \in T \mid \neg \text{mem}_S(s) \wedge \text{coh}(s)\})}{|T|}$$

其中 $\text{coh}(s)$ 判断连贯性，$\text{mem}_S(s)$ 判断是否为训练集记忆，$\text{uniq}(\cdot)$ 计算唯一样本数。该指标同时衡量**连贯性 + 原创性 + 多样性**。

### 四类算法任务

| 任务 | 创造力类型 | 核心要求 | 思维跳跃的本质 |
|------|-----------|---------|--------------|
| Sibling Discovery | 组合式 | 在二部图中生成新颖的兄弟-父节点三元组 $(γ, γ', Γ)$ | 需要先规划父节点（双关语的 punchline）再生成子节点 |
| Triangle Discovery | 组合式 | 在知识图谱中生成新颖三角形 $(v_1, v_2, v_3)$ | 需要同时协调三条边，更复杂的高阶规划 |
| Circle Construction | 探索式 | 生成可重排为环图的邻接表 | 需要构造新颖的排列 $\pi$ 作为隐式规划 |
| Line Construction | 探索式 | 生成可重排为线图的邻接表 | 类似环构造但终止条件不同 |

### NTP 为何在这些任务中受限

核心论证：NTP 的自回归分解 $p(s_i | s_{<i}, z)$ 会被 **Clever Hans 作弊**所误导。以 Sibling Discovery 为例：

1. 理想生成过程：先规划父节点 $z := \Gamma$，再按 $p(\gamma | z)$ 独立采样子节点，仅需 $O(m \cdot n)$ 数据
2. NTP 实际学习：模型在预测 $\Gamma$ 时已看到 $(\gamma, \gamma')$，直接利用输入中的线索（Clever Hans cheat），导致缺乏学习隐式规划 $z$ 的梯度信号
3. 后果：模型通过 $p(\gamma' | \gamma)$ 学习第二个子节点，需要 $O(m \cdot n^2)$ 数据，效率低 $n$ 倍

### 多 token 方法

- **Teacherless 训练**：用 dummy token 遮盖输入中的真实序列 $s$，模型仅从 prompt $p$ 同时预测所有位置的 token，避免 Clever Hans 作弊
- **离散扩散模型 (SEDD)**：从全遮盖/损坏序列出发，迭代去噪恢复所有 token，天然具备全局视野

### Seed-Conditioning

传统多样性来源于输出层的温度采样，但这可能导致"认知过载"——模型需同时处理多条思维线索来计算边缘化的 token 分布。

Seed-conditioning 方案：训练时为每个样本关联一个随机无意义前缀字符串（seed），推理时使用新 seed 生成新样本。直觉上，固定一个随机种子让模型专注于单一思维线，而非维护多条运行中的想法。

## 实验关键数据

### 主实验：Gemma v1 (2B) 上多 token 训练的效果

| 任务 | NTP 创造力 | 多 token 创造力 | 提升倍数 | NTP 记忆率 | 多 token 记忆率 |
|------|-----------|---------------|---------|-----------|---------------|
| Sibling Discovery | ~0.05 | ~0.25 | **~5×** | 高 | 显著降低 |
| Triangle Discovery | ~0.04 | ~0.20 | **~5×** | 高 | 显著降低 |
| Circle Construction | 低 | 中等 | 提升 | 高 | 降低 |
| Line Construction | 低 | 中等 | 提升 | 高 | 降低 |

### 小模型对比：GPT-2 (86M) vs SEDD (90M)

扩散模型 SEDD 在 4 个任务中有 3 个显著优于 NTP 训练的 GPT-2（Sibling Discovery 上略差）。Teacherless 训练对小模型增益有限（优化困难），但 top-K 样本的创造力仍有提升。

### Seed-Conditioning 效果

- 即使使用**贪心解码**（temperature=0），seed-conditioning 也能产生非平凡的创造力得分
- Seed 长度增加一致地提升创造力
- Seed-conditioning 的创造力与温度采样**可比甚至更优**（尤其在探索式创造力任务上）
- 在固定温度下，添加 seed 前缀总是优于无前缀

### 真实任务验证：摘要生成

在 XSUM 和 CNN/DailyMail 上，Teacherless 训练的大模型在相同质量（Rouge）下实现了更高的多样性（Self-BLEU），且一致提升了摘要质量。

## 亮点与洞察

1. **最小化可控测试平台**：通过极简算法任务隔离创造力的核心计算需求，绕过真实任务评估的主观性和不可控性
2. **NTP 近视性的新论证**：不同于 B&N'24 的推理正确性差距，本文展示的是**多样性差距**，且出现在仅需 2-token 前瞻的简单任务中
3. **排列不变性**：Triangle Discovery、Circle/Line Construction 是排列不变的——没有任何 token 重排序对 NTP 友好，这对基于 permutation 的改进方案（如 XL-Net 式方法）提出了挑战
4. **Seed-conditioning 的惊喜发现**：任意无意义的随机前缀竟能被模型转化为有意义的随机性，即使确定性解码也能产生多样输出
5. **记忆化分析**：NTP 的创造力不足主要源于**过度记忆**训练数据，而非生成不连贯——多 token 方法大幅减少记忆化

## 局限与展望

1. **最小化任务的局限**：算法任务是真实创造力的极端简化，在此成功不能保证在复杂任务上成功（但失败可以保证在复杂任务上也失败）
2. **Teacherless 训练的优化困难**：对小模型效果有限，与 Gloeckle et al. (2024) 的发现一致
3. **Seed-conditioning 需要专门训练**：不如温度采样即插即用，且在扩散模型上无增益
4. **未充分探索**：模型规模效应、预训练影响、上下文学习 vs. 微调的对比等均未深入分析
5. **创造力定义狭窄**：仅覆盖 Boden 分类法中的两类，未涉及变革式创造力（Transformative Creativity）和大 C 创造力
6. **分布内创造力**：本文衡量的原创性是训练分布内的新颖性，未涉及分布外泛化

## 相关工作与启发

- **Bachmann & Nagarajan (2024)**：path-star 任务展示 NTP 在推理正确性上的局限，本文将讨论扩展到创造力/多样性维度
- **Khona et al. (2024)**：路径连通性任务中的多样性-准确性权衡，本文展示多 token 方法可改善此权衡
- **Allen-Zhu & Li (2023)**：NTP 在大数据量下可学习复杂 CFG，本文的负面结果不矛盾（展示的是小数据下的次优性）
- **Lou et al. (2023) SEDD**：离散扩散模型，本文借用作为多 token 基线
- **DeSalvo et al. (2024)**：通过可变 soft-prompt 诱导多样性，与 seed-conditioning 思路相似

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 从算法任务角度系统量化 NTP 的创造力极限，视角独特且启发性强
- 实验充分度: ⭐⭐⭐⭐ — 四类任务 + 两种模型规模 + 消融实验 + 真实任务验证，较全面；但模型规模受限
- 写作质量: ⭐⭐⭐⭐⭐ — 动机清晰、论证严密、与认知科学文献衔接自然
- 价值: ⭐⭐⭐⭐ — 为多 token 预测和 seed-conditioning 提供了新的理论支撑，但最小化任务到真实应用的 gap 仍需验证

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Latent Imputation before Prediction: A New Computational Paradigm for De Novo Peptide Sequencing](latent_imputation_before_prediction_a_new_computational_paradigm_for_de_novo_pep.md)
- [\[NeurIPS 2025\] Is Sequence Information All You Need for Bayesian Optimization of Antibodies?](../../NeurIPS2025/computational_biology/is_sequence_information_all_you_need_for_bayesian_optimization_of_antibodies.md)
- [\[ICLR 2026\] Extending Sequence Length is Not All You Need: Effective Integration of Multimodal Signals for Gene Expression Prediction](../../ICLR2026/computational_biology/extending_sequence_length_is_not_all_you_need_effective_integration_of_multimoda.md)
- [\[NeurIPS 2025\] One Small Step with Fingerprints, One Giant Leap for De Novo Molecule Generation from Mass Spectra](../../NeurIPS2025/computational_biology/one_small_step_with_fingerprints_one_giant_leap_for_de_novo_molecule_generation_.md)
- [\[AAAI 2026\] MergeDNA: Context-aware Genome Modeling with Dynamic Tokenization through Token Merging](../../AAAI2026/computational_biology/mergedna_context-aware_genome_modeling_with_dynamic_tokenization_through_token_m.md)

</div>

<!-- RELATED:END -->
