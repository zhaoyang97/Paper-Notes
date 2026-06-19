---
title: >-
  [论文解读] PhantomWiki: On-Demand Datasets for Reasoning and Retrieval Evaluation
description: >-
  [ICML 2025][LLM评测][benchmark] 提出 PhantomWiki——一个按需生成虚构世界语料库和 QA 对的评测框架，通过上下文无关文法（CFG）控制推理难度、调节宇宙规模控制检索难度，实现对 LLM 推理与检索能力的解耦评估，同时天然抵抗数据泄漏。 现有 QA benchmark（如 SQuAD、H…
tags:
  - "ICML 2025"
  - "LLM评测"
  - "benchmark"
  - "reasoning evaluation"
  - "retrieval evaluation"
  - "synthetic data"
  - "data contamination"
---

# PhantomWiki: On-Demand Datasets for Reasoning and Retrieval Evaluation

**会议**: ICML 2025  
**arXiv**: [2502.20377](https://arxiv.org/abs/2502.20377)  
**代码**: [github.com/kilian-group/phantom-wiki](https://github.com/kilian-group/phantom-wiki)  
**领域**: LLM评测  
**关键词**: benchmark, reasoning evaluation, retrieval evaluation, synthetic data, data contamination

## 一句话总结

提出 PhantomWiki——一个按需生成虚构世界语料库和 QA 对的评测框架，通过上下文无关文法（CFG）控制推理难度、调节宇宙规模控制检索难度，实现对 LLM 推理与检索能力的解耦评估，同时天然抵抗数据泄漏。

## 研究背景与动机

现有 QA benchmark（如 SQuAD、HotpotQA、DROP、MuSiQue 等）存在两个根本性问题：

**数据泄漏与记忆化**：静态数据集不可避免地被爬取进训练数据，导致模型性能虚高。GSM8K 上的研究已表明，前沿模型在题目微调后性能大幅下降，说明模型可能只是在"背答案"。

**推理与检索难以解耦**：当问题涉及真实世界知识（如莫扎特的出生日期），无法分辨模型究竟是在推理、检索还是回忆预训练中的记忆。修改维基百科事实来构造新 QA 对又面临跨文章事实一致性的挑战（改了莫扎特的生日，贝多芬文章中关于两人会面的记载也需要同步修改）。

因此，作者的核心思路是：**不基于任何现有数据，每次评估时按需生成全新的虚构世界**，从根本上消除数据污染，并通过独立控制推理步数和语料规模来分别评估推理和检索能力。

## 方法详解

### 整体框架

PhantomWiki 的流水线包含四个阶段：

1. **生成角色（Characters）**：构建包含 n 个角色的虚构世界，先生成家族树（迭代生成父子关系），再用 Erdős–Rényi 模型生成友谊图。
2. **生成事实（Facts）**：为每个角色分配姓名（基于性别和家族姓氏，总计 1500 万种全名组合）、出生日期（与家族关系一致）、职业（300+ 种）和爱好（600+ 种）。
3. **生成文章（Articles）**：用预定义模板将事实转换为类 Wiki 传记文章（每篇约 160 tokens），这些文章是评估时提供给模型的唯一信息。
4. **生成 QA 对（Questions & Answers）**：通过上下文无关文法（CFG）生成问题模板，再通过 Prolog 逻辑编程求解所有正确答案。

### 关键设计

#### 上下文无关文法（CFG）生成问题

CFG 是 PhantomWiki 最核心的设计之一。通过递归组合问题模板，CFG 可以系统地生成不同复杂度的问题：

- **递归深度 d=1**：简单问题如 "Who is the friend of David?"
- **递归深度 d 增大**：生成多跳问题如 "Who is the nephew of the friend of the brother of David?"
- **问题类型多样化**：
    - 关系查询："Who is the <<<relation>>> of <<<name>>>?"
    - 属性查询："Who is the person whose hobby is birdwatching?"
    - 聚合查询："How many brothers does David have?"
    - 组合查询："How many friends does the brother of the person whose hobby is birdwatching have?"

关系类型的选择也影响推理步数：`nephew` 需要 2 步推理（sibling + son），而 `second cousin` 需要 5 步。CFG 在生成问题的同时并行生成对应的 Prolog 查询模板。

#### Prolog 逻辑编程求解答案

将虚构世界表示为 Prolog 事实和规则：

- 事实：`hobby("David", "birdwatching")`
- 规则：`nephew(X, Y) :- sibling(X, A), son(A, Y)`

Prolog 程序可以穷举所有满足约束的答案，确保答案的**完备性和可验证性**。例如问题 "Who is the nephew of the friend of the person whose hobby is birdwatching?" 对应三条 Prolog 查询语句的联合求解。

#### 难度控制机制

- **推理难度**：通过 CFG 递归深度 d 和关系类型控制推理步数（1~15+ 步）
- **检索难度**：通过宇宙规模 n 控制语料量，当语料超过模型上下文窗口时迫使模型使用检索

#### 抗数据泄漏分析

可能的宇宙数量为 $\Theta(2^{n^2} \cdot c^n)$，其中 n 为角色数、c 为属性选项总数。即使 n=100，宇宙空间也是天文数字，记忆化几乎不可能。

### 损失函数 / 训练策略

PhantomWiki 本身是评测工具而非训练方法。但论文探索了在 PhantomWiki 上微调 LLM：

- **SFT（监督微调）**：在 10 个 PhantomWiki 实例上微调 Qwen2.5-0.5B 和 3B，效果与基线持平
- **GRPO（Group Relative Policy Optimization）**：在相同数据上训练，F1 有明显提升（Qwen2.5-3B 从 16.82 提升至 31.38），但仍远非最优
- 微调后模型在新数据集上的表现仍随推理步数增加而下降，证明 PhantomWiki 评测对记忆化具有鲁棒性

评估指标为 **answer-level F1 score**：模型需预测所有答案（逗号分隔列表），按答案级别计算精确率/召回率/F1。

## 实验关键数据

### 主实验

在三种宇宙规模（n=50/500/5000）下，评测了 4 个前沿 LLM × 5 种 prompting 策略：

| 模型 | 方法 | n=50 F1(%) | n=500 F1(%) | n=5000 F1(%) |
|------|------|-----------|------------|-------------|
| DeepSeek-R1-32B | CoT (In-Context) | **52.42** | 19.65 | — |
| GPT-4o | CoT (In-Context) | 50.66 | 41.02 | — |
| Llama-3.3-70B | CoT (In-Context) | 48.37 | 25.99 | — |
| GPT-4o | ReAct (Agentic) | 38.70 | 37.39 | **36.85** |
| Llama-3.3-70B | ReAct (Agentic) | 35.83 | **35.56** | 30.89 |
| Gemini-1.5-Flash | ReAct (Agentic) | 30.92 | 26.99 | 23.47 |
| GPT-4o | ZeroShot-RAG | 28.05 | 22.32 | 18.13 |
| DeepSeek-R1-32B | ReAct (Agentic) | 5.47 | 3.57 | 4.74 |

### 消融实验

| 方法 | Qwen2.5-0.5B F1(%) | Qwen2.5-3B (LoRA) F1(%) | 说明 |
|------|--------------------|-----------------------|------|
| ZeroShot | 11.78 ± 0.94 | 16.82 ± 2.37 | 基线 |
| CoT | 2.68 ± 0.22 | 13.71 ± 0.81 | 小模型难以遵循 CoT 格式 |
| SFT | 11.71 ± 1.10 | 16.89 ± 2.22 | 监督微调无显著提升 |
| GRPO | **13.25 ± 0.93** | **31.38 ± 0.86** | RL 微调显著提升推理能力 |

### 生成效率

| 宇宙规模 n | 总耗时 | 事实生成 | 文章生成 | 问题生成 |
|-----------|--------|---------|---------|---------|
| 100 | 0.97s | 0.46s | 0.07s | 0.44s |
| 1,000 | 2.86s | 0.90s | 0.59s | 1.37s |
| 10,000 | 20.91s | 5.38s | 5.87s | 9.66s |
| 100,000 | 5.57min | 0.81min | 0.97min | 3.79min |
| 1,000,000 | 3.86h | 9.47min | 11.77min | 3.51h |

### 关键发现

1. **推理能力随步数急剧下降**：所有模型和 prompting 方法的 F1 都随推理步数增加而显著下降。CoT 比 ZeroShot 衰减更慢，但每增加一步推理仍然更具挑战。
2. **RAG 在多跳推理上几乎失效**：ZeroShot-RAG 和 CoT-RAG 在 5 步以上推理的 F1 接近零，因为单次检索无法获取多跳推理所需的所有文档。
3. **Agentic prompting (ReAct) 在大规模检索中表现最优**：动态交互式检索显著优于一次性 RAG，在 n=5000 时 ReAct 是唯一可用的范式。
4. **DeepSeek-R1 推理强但工具使用弱**：CoT 下表现最佳，但 ReAct 下 F1 极低（~5%），暴露了其 tool-calling 能力的不足。
5. **多分支推理是额外的失败模式**：模型在寻找所有可能答案时经常遗漏分支（如找 great-grandchild 时漏掉某些 grandchildren 的后代）。

## 亮点与洞察

- **极其优雅的评测哲学**：发布的是"数据集生成流水线"而非固定数据集，从根本上解决了 benchmark 过时和数据泄漏问题。
- **CFG + Prolog 的组合非常巧妙**：CFG 保证问题的系统性覆盖和难度可控，Prolog 保证答案的穷举正确性，两者结合实现了"全自动、可验证、可控难度"的评测。
- **解耦推理与检索**：通过独立调节问题复杂度和语料规模这两个正交维度，首次实现了对 LLM 推理和检索能力的系统解耦测量。
- **模板文章看似简单实则聪明**：约 160 tokens 的模板文章避免了 LLM 改写带来的幻觉问题，同时保持了生成速度和成本优势。
- **可扩展到百万级**：n=1M 的宇宙（接近 Wikipedia 传记条目规模）仅需约 4 小时即可在普通 CPU 上生成。

## 局限与展望

1. **文本过于模板化**：当前文章风格单一（"The job of David is a farmer"），与真实 Wikipedia 差距较大。虽然 LLM 改写引入了幻觉风险，但一致性保持的改写方法值得探索。
2. **关系类型有限**：目前仅包含家庭关系和友谊关系，缺少职业关系、师生关系等现实世界中的复杂社会关系。
3. **问题类型仍较基础**：缺少对比型（"谁更年长？"）、多约束型等更复杂的问题模式。论文提到 CFG 可扩展，但当前版本尚未实现。
4. **仅限文本模态**：未涉及视觉、音频等多模态评测。
5. **BM25 用于 RAG 可能不公平**：合成文本的特殊性使得神经检索器不适用，但仅用 BM25 可能低估了 RAG 的真实潜力。

## 相关工作与启发

- **CLUTRR (Sinha et al., 2019)**：通过众包家庭故事测试关系推理，PhantomWiki 可视为其在大规模、全自动、多跳设置下的扩展。
- **RepLiQA (Monteiro et al., 2024)**：众包语料以抵抗数据泄漏，但仍是静态数据集，不如 PhantomWiki 的按需生成彻底。
- **ToolQA (Zhuang et al., 2023)**：多领域工具增强 QA 评测，但使用真实数据、需要人工验证。PhantomWiki 聚焦于逻辑推理的系统压力测试。
- **RULER (Hsieh et al., 2024)**：长上下文评测框架，但范围较窄。PhantomWiki 在检索维度上提供了更丰富的评测。

**启发**：这种"发布生成器而非数据集"的 benchmark 范式值得在更多研究方向推广，尤其是对于重视公平评估的领域。GRPO 微调显著优于 SFT 也暗示了强化学习在复杂推理任务上的潜力。

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 新颖性 | ⭐⭐⭐⭐ | "生成器即Benchmark"的理念在 QA 评测领域较为新颖 |
| 技术深度 | ⭐⭐⭐⭐ | CFG + Prolog 的设计精巧且有理论分析 |
| 实验质量 | ⭐⭐⭐⭐⭐ | 多模型×多策略×多规模的全面评测，结论清晰 |
| 实用性 | ⭐⭐⭐⭐ | pip install 即可使用，HuggingFace 有示例数据 |
| 写作质量 | ⭐⭐⭐⭐ | 结构清晰，图表信息量大 |
| 综合评分 | ⭐⭐⭐⭐ | 优秀的评测基础设施工作，解耦推理与检索的思路有价值 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models](../../ACL2025/llm_evaluation/retrieval_models_arent_tool-savvy_benchmarking_tool_retrieval_for_large_language.md)
- [\[ECCV 2024\] Sync from the Sea: Retrieving Alignable Videos from Large-Scale Datasets](../../ECCV2024/llm_evaluation/sync_from_the_sea_retrieving_alignable_videos_from_large-scale_datasets.md)
- [\[ACL 2025\] MARS: Benchmarking the Metaphysical Reasoning Abilities of Language Models with a Multi-task Evaluation Dataset](../../ACL2025/llm_evaluation/mars_benchmarking_the_metaphysical_reasoning_abilities_of_language_models_with_a.md)
- [\[ACL 2025\] PhysReason: A Comprehensive Benchmark towards Physics-Based Reasoning](../../ACL2025/llm_evaluation/physreason_a_comprehensive_benchmark_towards_physics-based_reasoning.md)
- [\[ICCV 2025\] 3DSRBench: A Comprehensive 3D Spatial Reasoning Benchmark](../../ICCV2025/llm_evaluation/3dsrbench_a_comprehensive_3d_spatial_reasoning_benchmark.md)

</div>

<!-- RELATED:END -->
