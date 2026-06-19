---
title: >-
  [论文解读] Chain-of-Retrieval Augmented Generation (CoRAG)
description: >-
  [NeurIPS 2025][信息检索/RAG][RAG] 提出 CoRAG 框架，通过拒绝采样自动生成中间检索链（子查询→子答案），微调 LLM 学习迭代检索和推理，并支持多种测试时解码策略（贪心 / Best-of-N / 树搜索）灵活扩展计算量，在多跳 QA 上 EM 提升 26+ 点，KILT 基准 9/10 任务达到 SOTA。
tags:
  - "NeurIPS 2025"
  - "信息检索/RAG"
  - "RAG"
  - "Chain-of-Retrieval"
  - "多跳推理"
  - "拒绝采样"
  - "检索链"
  - "测试时计算扩展"
---

# Chain-of-Retrieval Augmented Generation (CoRAG)

**会议**: NeurIPS 2025  
**arXiv**: [2501.14342](https://arxiv.org/abs/2501.14342)  
**代码**: [microsoft/LMOps/corag](https://github.com/microsoft/LMOps/tree/main/corag)  
**领域**: 信息检索  
**关键词**: RAG, Chain-of-Retrieval, 多跳推理, 拒绝采样, 检索链, 测试时计算扩展

## 一句话总结

提出 CoRAG 框架，通过拒绝采样自动生成中间检索链（子查询→子答案），微调 LLM 学习迭代检索和推理，并支持多种测试时解码策略（贪心 / Best-of-N / 树搜索）灵活扩展计算量，在多跳 QA 上 EM 提升 26+ 点，KILT 基准 9/10 任务达到 SOTA。

## 研究背景与动机

**领域现状**：传统 RAG 采用"单步检索 + 生成"流水线，检索器（如 bi-encoder）为效率牺牲了表达能力，面对复杂多跳查询时检索质量成为瓶颈。

**现有痛点**：(a) 单次检索常无法获取回答复杂问题所需的全部信息；(b) 多跳推理中"该查什么"取决于推理的实时状态，事先无法确定；(c) 已有多步检索方法（FLARE、IRCoT、Self-RAG）主要依赖少样本提示或闭源模型蒸馏，缺乏端到端训练。

**核心矛盾**：如何让开源 LLM 从仅含最终答案的 QA 数据集中自动学得迭代检索策略，并在推理时灵活控制计算-性能权衡？

**切入角度**：借鉴链式思维（Chain-of-Thought）的成功经验，将检索过程展开为子查询-子答案组成的"检索链"，每一步基于前序推理状态动态决定下一步查什么。

**核心 idea**：用拒绝采样为 QA 数据集自动生成检索链作为训练信号，微调 LLM 学习迭代检索，测试时通过 Best-of-N 和树搜索灵活扩展计算。

## 方法详解

### 整体框架

三步流程：**(1)** 拒绝采样生成检索链训练数据 → **(2)** 多任务统一微调（子查询预测 + 子答案预测 + 最终答案预测） → **(3)** 多样化测试时解码策略。模型接收"当前状态"（原始查询 + 已有子查询/子答案链）作为输入，预测"下一步动作"（新的子查询或最终答案）。

### 关键设计

1. **拒绝采样生成检索链**

    - **功能**：从仅含 $(Q, A)$ 的 QA 数据集自动构造包含中间推理步骤的检索链
    - **核心机制**：对每个样本，LLM 基于当前状态 $(Q, Q_{<i}, A_{<i})$ 采样子查询 $Q_i$，再用 E5-large 检索 top-5 文档后生成子答案 $A_i$。至多采样 16 条候选链，按正确答案的条件 log-likelihood $\log P(A|Q, Q_{1:L}, A_{1:L})$ 打分，选最优链
    - **设计动机**：摆脱对人工标注和闭源模型的依赖，利用 LLM 自身能力自举地产出高质量训练数据

2. **多任务统一训练**

    - **功能**：在同一框架下联合优化三个相关预测任务
    - **三个损失**：$L_\text{sub\_query} = -\log P(Q_i | Q, Q_{<i}, A_{<i})$（学"接下来该查什么"）；$L_\text{sub\_answer} = -\log P(A_i | Q_i, D_{1:k}^{(i)})$（学"从检索结果中提取什么"）；$L_\text{final\_answer} = -\log P(A | Q, Q_{1:L}, A_{1:L}, D_{1:k})$（学"最终如何综合回答"）
    - **设计动机**：三个任务互相增强——学会生成好的子查询帮助最终推理，预测最终答案的监督又反向优化子查询生成

3. **测试时计算扩展**

    - **三种解码策略**：(a) **贪心解码**——固定链长 $L$，顺序生成子查询和子答案；(b) **Best-of-N 采样**——温度 0.7 采样 $N$ 条链，以"No relevant information found"的 log-likelihood 作为惩罚分，选惩罚最低的链；(c) **树搜索**——BFS 扩展并通过 rollout 平均惩罚评估各状态
    - **缩放律**：计算量（总 token 消耗）与性能之间的 Pareto 前沿近似遵循 $y = a \log(x+b) + c$ 的 log-linear 关系

### 训练细节

- 基座模型：Llama-3.1-8B-Instruct，全参数微调
- 多跳 QA 数据集 125k 样本训练 1 epoch；KILT 数据集 660k 样本训练 1 epoch
- 最大序列长度 3k tokens；检索语料库为 KILT 提供的英文 Wikipedia（约 36M 段落）
- KILT 评测额外微调 E5-Mistral 检索器 + RankLLaMA 重排器以提升排序质量

## 实验关键数据

### 多跳 QA 主实验

| 数据集 | CoRAG-8B (最佳配置) | 最强 Baseline | EM 提升 |
|--------|-------------------|-------------|---------|
| 2WikiMultihopQA | **72.5** (L=10, N=8) | Search-o1-32B 58.0 | +14.5 |
| HotpotQA | **56.3** (L=10, N=8) | ITER-RETGEN 45.1 | +11.2 |
| MuSiQue | **30.9** (L=10, N=8) | ITER-RETGEN 26.1 | +4.8 |
| Bamboogle | 54.4 (L=10, N=8) | Search-o1-32B **56.0** | -1.6 |

> Bamboogle 仅 125 个样本方差大，且部分问题需要比检索语料库更新的知识，使用商业搜索引擎的系统天然占优。

### KILT 基准（隐藏测试集）

CoRAG-8B 在 9/10 任务上取得 SOTA：Entity Linking (AIDA 93.9)、Slot Filling (T-REx 88.0, zsRE 87.2)、Open QA (NQ 63.1, HotpotQA 60.6, TriviaQA 88.3)、Fact Verification (FEVER 93.1)。唯一未超越的 FEVER 任务仅以微弱差距落后于 Atlas-11B（93.1 vs 93.5），而 CoRAG 参数量仅为其 73%。

### 消融实验

| 配置 | 2Wiki EM | HotpotQA EM | MuSiQue EM |
|------|----------|-------------|------------|
| CoRAG (L=6, greedy) | 70.6 | 54.4 | 27.7 |
| + 迭代拒绝采样（第 2 轮） | 72.2 (+1.6) | 53.4 (-1.0) | 26.6 (-1.1) |
| + GPT-4o 蒸馏 | **75.1** (+4.5) | **56.6** (+2.2) | **28.2** (+0.5) |
| 弱到强：1B 生成 → 8B 训练 | 59.3 | 50.3 | 22.3 |
| 弱到强：3B 生成 → 8B 训练 | 69.9 | 53.9 | 25.2 |
| 测试时换 BM25 + best-of-4 | 62.6 | 51.6 | 23.5 |
| 测试时换 E5-base + best-of-4 | 70.8 | 53.0 | 26.3 |

### 检索召回率提升

| 数据集 | E5-large R@10 | CoRAG R@10 | 提升 |
|--------|--------------|-----------|------|
| HotpotQA | 59.1 | 72.1 | +13.0 |
| 2WikiMultihopQA | 54.9 | 81.4 | **+26.5** |
| Bamboogle | 31.2 | 59.2 | +28.0 |
| MuSiQue | 29.0 | 47.1 | +18.1 |

### 关键发现

- **多跳任务收益巨大，单跳收益有限**：MuSiQue 等多跳数据集通过扩展链长和采样数可持续提升，而 NQ、TriviaQA 等单跳数据集边际收益很小
- **弱到强泛化有效**：3B 模型生成的检索链训练 8B 模型，性能接近 8B 自生成（2Wiki EM 69.9 vs 70.6），大幅降低数据生成成本
- **检索器鲁棒性强**：即使测试时替换为弱检索器（BM25），链式检索机制仍能通过多步查询弥补检索质量不足
- **自适应停止尚不成熟**：学习何时停止虽能节省 token，但以性能下降为代价，最优配置依赖数据集特征

## 亮点与洞察

- **数据生成范式的突破**：拒绝采样优雅解决了"中间检索步骤无标注"的核心困难，无需人工标注或闭源模型即可从 QA-only 数据集构造完整训练信号
- **测试时计算的量化指导**：log-linear 缩放律为实际部署时的延时-精度权衡提供定量决策依据
- **"未找到信息"作为自评估信号**：将模型对"No relevant information found"的输出概率作为检索质量的内部度量，用于 Best-of-N 选择，巧妙避免了测试时无法获取正确答案的问题
- **检索链显著提升检索召回**：通过迭代查询改写，CoRAG 在所有多跳数据集上将 R@10 提升 13-28 个百分点，证明方法不仅改善最终答案还真正改善了检索质量

## 局限性与改进方向

- 主要在短答案 QA 上评估，长文本生成场景（如摘要、报告）未覆盖
- 自适应链长仍是开放问题——固定 $L$ 不够灵活，但学习何时停止的效果不理想
- 拒绝采样的计算成本较高（每个样本最多 16 条候选链），弱到强泛化可部分缓解
- 计算-性能缩放律在不同任务上系数不同，缺乏统一的自适应计算分配机制

## 相关工作对比

- **vs 标准 RAG**：单步检索 → 多步迭代，多跳任务 EM 提升 10-26 点
- **vs FLARE / IRCoT / Self-RAG**：从依赖提示工程或蒸馏 → 端到端微调，系统性能更优
- **vs Search-o1**：CoRAG-8B 在 2Wiki 上以 72.5 大幅超越 Search-o1-32B 的 58.0，参数量仅为 1/4
- **vs IterDRAG**：同样研究测试时缩放，但 CoRAG 通过微调获得更强的基础能力而非仅靠 few-shot

## 评分

- 新颖性: ⭐⭐⭐⭐ 拒绝采样生成检索链 + 测试时计算扩展的组合新颖，但各组件均有已知先验工作
- 实验充分度: ⭐⭐⭐⭐⭐ 多跳 QA + KILT 全任务 + 消融 + 缩放分析 + 弱到强 + 检索器鲁棒性，非常全面
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，实验分析深入，图表信息量大
- 价值: ⭐⭐⭐⭐⭐ KILT 9/10 SOTA，提供可复现的开源代码，对 RAG 系统设计有强实践指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Retrieval-Augmented Generation for Reliable Interpretation of Radio Regulations](retrieval-augmented_generation_for_reliable_interpretation_of_radio_regulations.md)
- [\[NeurIPS 2025\] Reliable Decision Making via Calibration Oriented Retrieval Augmented Generation](reliable_decision_making_via_calibration_oriented_retrieval_augmented_generation.md)
- [\[NeurIPS 2025\] Cooperative Retrieval-Augmented Generation for Question Answering: Mutual Information Exchange and Ranking by Contrasting Layers](cooperative_retrieval-augmented_generation_for_question_answering_mutual_informa.md)
- [\[NeurIPS 2025\] HyperGraphRAG: Retrieval-Augmented Generation via Hypergraph-Structured Knowledge Representation](hypergraphrag_retrieval-augmented_generation_via_hypergraph-structured_knowledge.md)
- [\[ACL 2025\] GeAR: Generation Augmented Retrieval](../../ACL2025/information_retrieval/gear_generation_augmented_retrieval.md)

</div>

<!-- RELATED:END -->
