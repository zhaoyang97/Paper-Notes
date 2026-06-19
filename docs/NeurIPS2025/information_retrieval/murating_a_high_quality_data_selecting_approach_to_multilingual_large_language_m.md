---
title: >-
  [论文解读] MuRating: A High Quality Data Selecting Approach to Multilingual Large Language Model Pretraining
description: >-
  [NeurIPS 2025][信息检索/RAG][多语言数据选择] 提出 MuRating，一个可扩展的多语言数据选择框架：先通过配对比较聚合多个英文数据质量评分器，再借助翻译将质量信号迁移到 17 种语言，训练出语言无关的多语言质量评估模型，在 1.2B 和 7B 规模 LLM 预训练中取得了持续的性能提升。
tags:
  - "NeurIPS 2025"
  - "信息检索/RAG"
  - "多语言数据选择"
  - "预训练数据质量"
  - "Bradley-Terry模型"
  - "配对比较"
  - "跨语言对齐"
---

# MuRating: A High Quality Data Selecting Approach to Multilingual Large Language Model Pretraining

**会议**: NeurIPS 2025  
**arXiv**: [2507.01785](https://arxiv.org/abs/2507.01785)  
**代码**: [https://github.com/aialt/MuRater](https://github.com/aialt/MuRater)  
**领域**: 信息检索  
**关键词**: 多语言数据选择, 预训练数据质量, Bradley-Terry模型, 配对比较, 跨语言对齐

## 一句话总结

提出 MuRating，一个可扩展的多语言数据选择框架：先通过配对比较聚合多个英文数据质量评分器，再借助翻译将质量信号迁移到 17 种语言，训练出语言无关的多语言质量评估模型，在 1.2B 和 7B 规模 LLM 预训练中取得了持续的性能提升。

## 研究背景与动机

高质量预训练数据是 LLM 性能的关键驱动力。现有的模型化数据选择方法（如 QuRater、DCLM、AskLLM、FineWeb-Edu）已在英文数据选择上取得成功，但几乎**完全面向英文**，在多语言 LLM 预训练中留下了关键空白。

现有方法的局限：

**英文中心化**：主流数据选择方法未针对非英文语言设计或验证

**评分不一致**：不同评分器在不同任务上各有优劣（如 DCLM 在 HellaSwag 强但 ARC-Challenge 弱；QuRater 反之），缺乏统一框架

**多语言尝试的风险**：FineWeb2-HQ 使用基准数据集作为正样本训练语言特定分类器，存在测试集污染风险

**绝对评分的跨语言不一致性**：翻译引入的微妙偏差会影响绝对分数（pointwise scoring），导致跨语言评分不可靠

## 方法详解

### 整体框架

MuRating 是一个两阶段框架：

**阶段 1：英文评分器聚合** → 将多个英文评分器的判断统一为单一质量分数

**阶段 2：多语言迁移** → 通过翻译将英文质量信号投射到 17 种目标语言

### 关键设计

**1. 英文 AutoRater 聚合（阶段 1）**

给定文本对 $(t_A, t_B)$ 和 $N$ 个评分器，每个评分器 $n$ 给出分数 $S_A^n$ 和 $S_B^n$。计算多数投票置信度：

$$P_{A>B} = \frac{1}{|N|} \sum_{n \in N} \mathbb{I}[S_A^n > S_B^n]$$

使用 Bradley-Terry 模型和 BCE 损失训练统一评分器：

$$\mathcal{L}_\theta = \mathbb{E}_{(t_A, t_B, p_{B \succ A}) \in \mathcal{J}} \left[ -p_{B \succ A} \log \sigma(s_\theta(t_B) - s_\theta(t_A)) - (1 - p_{B \succ A}) \log \sigma(s_\theta(t_A) - s_\theta(t_B)) \right]$$

聚合的四个评分器：GPT-4o（多次正反方向评分取平均）、AskLLM（Flan-T5-XXL）、FineWeb-Edu Classifier、DCLM（fastText）。

**2. 基于翻译的多语言偏好迁移（阶段 2）**

核心假设：翻译保留语义内容和文本对之间的相对质量关系，即 $P_{A^m > B^m} \approx P_{A^{en} > B^{en}}$。

构建三类训练数据：
- **单语对 (monolingual)**：$(t_A^m, t_B^m)$，同种语言内比较，150K 对
- **跨语言对 (cross-lingual)**：$(t_A^m, t_B^{m'})$，不同语言间比较，促进跨语言一致性，150K 对
- **平行对 (parallel)**：$(t_A^m, t_A^{m'})$，相同内容的不同语言翻译，赋予中性偏好 $P \approx 0.5$，75K 对

平行对的正则化损失：

$$\mathcal{L}_{\text{parallel}} = \mathbb{E}_{(t_A^m, t_A^{m'}) \in \mathcal{J}'} \left[ -\log\sigma(s_\theta(t_A^m) - s_\theta(t_A^{m'})) - \log\sigma(s_\theta(t_A^{m'}) - s_\theta(t_A^m)) \right]$$

最终目标：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{pairwise}} + \lambda \cdot \mathcal{L}_{\text{parallel}}$$

**3. 配对评分 vs 绝对评分**

论文通过实验证明配对（pairwise）比绝对（pointwise）评分更适合跨语言迁移：
- 绝对评分在中等质量范围（3-6 分）表现出显著的跨语言波动
- 配对评分在翻译引入偏差时仍保持跨语言一致性（只要相对排序不变即可）

**4. 模型架构**

基于 BGE-M3 编码器微调，添加线性头预测质量评分。选择 BGE-M3 因其强多语言表示能力和轻量设计。验证集准确率 93%，训练集 97%。

### 损失函数 / 训练策略

- 英文评分器聚合：300K 英文文本对，GPT-4o 正反方向多次评分消除顺序偏差
- 多语言训练集：150K 单语对 + 150K 跨语言对 + 75K 平行对
- 翻译使用 GPT-4o，17 种目标语言均衡分布
- 预训练使用 LLaMA 架构，1.2B 参数单 epoch 训练，英文 200B token + 多语言 300B token

## 实验关键数据

### 主实验

**多语言结果（18 种语言，1.2B 模型）：**

| 选择方法 | 阅读理解 (5任务) | 常识推理 (2任务) | 世界知识 (4任务) | 平均 (11任务) |
|---------|-----------------|-----------------|-----------------|-------------|
| Uniform (+50%) | 53.16 | 54.58 | 38.25 | 48.66 |
| HPLT-2 | 50.38 | 49.77 | 36.96 | 45.70 |
| FineWeb-2 | 50.83 | 52.48 | 35.53 | 46.28 |
| QuRater-M | 54.58 | 54.87 | 38.12 | 49.19 |
| MuRater(M) | 54.91 | 55.48 | 39.68 | 50.02 |
| **MuRater(E)** | **56.05** | **56.42** | **40.40** | **50.96** |

**英文结果（12 个基准，1.2B 模型）：**

| 选择方法 | 阅读理解 (6任务) | 常识推理 (4任务) | 世界知识 (2任务) | 平均 (12任务) |
|---------|-----------------|-----------------|-----------------|-------------|
| Uniform (+50%) | 43.93 | 59.06 | 20.36 | 48.70 |
| AskLLM | 42.83 | 58.40 | 20.21 | 47.82 |
| DCLM | 46.00 | 58.99 | 22.37 | 50.23 |
| QuRater | 43.54 | 58.58 | 20.47 | 48.33 |
| **MuRater** | **47.13** | **59.95** | **22.53** | **51.23** |

**7B 模型结果（1T token 训练）：**

| 选择方法 | 阅读理解 | 常识推理 | 世界知识 | 平均 |
|---------|---------|---------|---------|------|
| QuRater-M | 61.96 | 63.28 | 43.31 | 56.18 |
| **MuRater** | **62.78** | **64.40** | **44.50** | **57.23** |

### 消融实验

**跨语言和平行对的有效性**：
- 加入对齐训练（跨语言对+平行对）后，平行文本评分的 MSE 更低，斜率更接近 1
- 表明语言间评分一致性显著提升

**Pairwise vs Pointwise 评分迁移**：
- GPT-4o 对 200 对阿拉伯语和西班牙语平行文本各评分 20 次
- Pointwise 评分在中等质量范围跨语言变异大
- Pairwise 评分的跨语言一致性显著更强（点符合 y=x 线）

**MuRater(E) vs MuRater(M)**：
- MuRater(E)：从英文评分出发翻译到多语言（英文锚定训练）
- MuRater(M)：将多语言对翻译成英文评分再投射回去
- MuRater(E) 始终优于 MuRater(M)，因英文语料更多样化，提供更稳定的监督信号

### 关键发现

1. **评分器聚合的优势**：MuRater 综合了各评分器的长处，在所有基准上均匀表现强劲，避免了单一评分器的偏好偏差
2. **英文锚定训练更有效**：MuRater(E) > MuRater(M)，英文语料的多样性为跨语言迁移提供了更丰富的信号
3. **配对比较的鲁棒性**：在翻译引入微妙偏差时，配对评分保持一致性，绝对评分显著不稳定
4. **规模一致性**：从 1.2B 到 7B（×5.8 倍参数，×2 倍数据），MuRater 的优势持续存在
5. 在 13 语言子集上 MuRater(E) 比 FineWeb2-HQ 高约 3 个百分点

## 亮点与洞察

1. **配对比较的深层洞察**：利用了"翻译保留相对质量但不保留绝对质量"这一关键观察，使得配对范式天然适合跨语言场景
2. **平行对作为正则化**：赋予平行翻译中性偏好 ($P \approx 0.5$) 的设计巧妙——它强制模型学习语言无关的质量度量，而非语言特征
3. **统一多源判断**：通过 Bradley-Terry 模型将不同评分器的异质输出统一为一致的质量分数，避免了手动选择"最佳"评分器的问题
4. **实用的翻译策略**：使用 GPT-4o 进行翻译虽有成本，但只需在训练集构建时执行一次，推理时 MuRater 可以直接评估任何语言

## 局限与展望

1. **语言覆盖有限**：仅 17 种目标语言，排除了大量低资源语言
2. **依赖 GPT-4o 偏差**：翻译和部分评分使用 GPT-4o 可能引入模型特有偏差
3. **领域偏好**：英文评分器侧重事实和信息性内容，对叙事和创意文本的评估能力有限
4. **缺乏语言特定调优**：统一的多语言评分器可能无法捕捉各语言的独特文化和语言特征
5. 翻译质量对评分迁移效果的灵敏度需要更深入的分析
6. 仅在 LLaMA 架构上验证，对其他架构的泛化性未知

## 相关工作与启发

- **QuRater** (Wettig et al., 2024)：使用 LLM 评估数据教育价值，MuRating 将其英文框架扩展到多语言
- **DCLM** (Li et al., 2024)：fastText 分类器方案，快但偏好偏差大。MuRating 聚合后更鲁棒
- **FineWeb2-HQ** (Messmer et al., 2025)：训练语言特定分类器，有测试集污染风险。MuRating 从英文迁移更安全
- **AskLLM** (Sachdeva et al., 2024)：基于 prompt 的 LLM 评分，被 MuRating 作为输入评分器之一聚合
- Bradley-Terry 配对比较模型在此场景下的应用与 RLHF 中的奖励模型训练高度类似
- 跨语言一致性的平行对正则化思想可迁移到其他多语言评估任务

## 评分

- 新颖性: ⭐⭐⭐⭐ 配对比较 + 翻译迁移的两阶段框架设计巧妙，解决了多语言数据质量评估的真实痛点
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 1.2B 和 7B 两个规模、18 种语言、多个基线对比、充分的消融实验
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，实验组织有条理，但相关工作与方法部分略显冗余
- 价值: ⭐⭐⭐⭐⭐ 填补了多语言预训练数据选择的重要空白，框架可扩展、可复用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] SafeRAG: Benchmarking Security in Retrieval-Augmented Generation of Large Language Model](../../ACL2025/information_retrieval/saferag_benchmarking_security_in_retrieval-augmented_generation_of_large_languag.md)
- [\[ECCV 2024\] Towards Open-Ended Visual Recognition with Large Language Model](../../ECCV2024/information_retrieval/towards_open-ended_visual_recognition_with_large_language_models.md)
- [\[ICML 2026\] Linguistic Nepotism: Trading-off Quality for Language Preference in Multilingual RAG](../../ICML2026/information_retrieval/linguistic_nepotism_trading-off_quality_for_language_preference_in_multilingual_.md)
- [\[ICML 2026\] Understand and Accelerate Memory Processing Pipeline for Large Language Model Inference](../../ICML2026/information_retrieval/understand_and_accelerate_memory_processing_pipeline_for_disaggregated_llm_infer.md)
- [\[ACL 2025\] Investigating Language Preference of Multilingual RAG Systems](../../ACL2025/information_retrieval/investigating_language_preference_of_multilingual_rag_systems.md)

</div>

<!-- RELATED:END -->
