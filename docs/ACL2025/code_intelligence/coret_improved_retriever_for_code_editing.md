---
title: >-
  [论文解读] CoRet: Improved Retriever for Code Editing
description: >-
  [ACL 2025][代码智能][代码检索] 提出 CoRet，一个面向代码编辑任务的稠密检索模型，通过整合代码语义、仓库文件层级结构和调用图依赖关系，并使用针对仓库级检索设计的对数似然损失函数，在 SWE-bench 和 Long Code Arena 上比现有模型的 Recall 至少提升 15 个百分点。
tags:
  - "ACL 2025"
  - "代码智能"
  - "代码检索"
  - "Code Editing"
  - "调用图"
  - "Repository-level"
  - "SWE-bench"
---

# CoRet: Improved Retriever for Code Editing

**会议**: ACL 2025  
**arXiv**: [2505.24715](https://arxiv.org/abs/2505.24715)  
**代码**: 无  
**领域**: Code Retrieval / 代码编辑检索  
**关键词**: 代码检索, Code Editing, 调用图, Repository-level, SWE-bench

## 一句话总结

提出 CoRet，一个面向代码编辑任务的稠密检索模型，通过整合代码语义、仓库文件层级结构和调用图依赖关系，并使用针对仓库级检索设计的对数似然损失函数，在 SWE-bench 和 Long Code Arena 上比现有模型的 Recall 至少提升 15 个百分点。

## 研究背景与动机

代码编辑是软件开发中的核心任务：开发者根据自然语言描述（如 GitHub Issue/PR）对代码仓库进行修改。成功的代码编辑首先需要准确地**检索到需要修改的代码片段**——这在大型真实仓库中尤其困难。

**现有模型的三个根本问题**：

**语义不对齐**：现有预训练编码器（如 CodeBERT, GraphCodeBERT, UniXcoder）学习的是 docstring-code 的语义匹配，但问题描述（issue）和需要修改的代码片段之间的语义关系与此完全不同。CodeSage 在 code search 上表现优秀，但这种能力不能直接迁移到代码编辑检索。

**缺失仓库结构信息**：现有模型把每个代码片段当作独立处理，丢失了文件路径等仓库层级信息。但文件路径是极强的检索信号——issue 描述中经常隐含对文件位置的暗示。

**忽略运行时依赖**：函数间的调用关系（call graph）对理解代码语义至关重要——一个函数调用了什么、被什么调用，决定了它的功能角色。现有模型不编码这种依赖关系。

## 方法详解

### 整体框架

CoRet 是一个双塔稠密检索模型：
- Query Encoder $Q(\cdot; \theta_q)$：编码自然语言 issue 描述
- Code Encoder $C(\cdot; \theta_c)$：编码代码片段（含上下文信息）
- 检索打分：$f(q, c_i) = \text{sim}(Q(q), C(c_i))$（余弦相似度）
- 两个编码器权重共享（$\theta_c = \theta_q$），基于 CodeSage Small 初始化

### 关键设计

1. **代码分块策略（Chunking）**：做什么→将仓库拆分为语义上有意义的原子单元；核心思路→以函数、类、类方法为基本单元（chunk），每个 chunk 前缀加上文件路径；设计动机→文件粒度太粗且缺乏语义凝聚性，函数级别更适合检索定位。一个仓库可能有上万个 chunk。

2. **仓库级对数似然损失函数**：做什么→用对数似然损失替代标准对比损失来训练检索模型；核心思路→对每个 query $q_i$，最大化检索到所有正确代码块 $c^* \in \mathcal{C}_i^*$ 的平均对数似然：

$$\mathcal{L}(\theta) = \frac{1}{N} \sum_i^N \frac{1}{|\mathcal{C}_i^*|} \sum_{c^* \in \mathcal{C}_i^*} \log \frac{\exp(\mathbf{q}_i \cdot \mathbf{c}^* / \tau)}{\Gamma(\mathbf{q}, \mathcal{C}_i)}$$

由于归一化项涉及整个仓库（可达万级 chunk），用随机采样实例内负样本（≤1024）近似；设计动机→标准对比损失使用跨 batch 的负样本，但代码编辑检索的核心是在**同一仓库内**区分相关和不相关代码，实例内负样本更有针对性。温度 $\tau = 0.05$。

3. **调用图上下文 (Call Graph Context)**：做什么→将调用图邻居的代码文本拼接到每个代码块中；核心思路→对代码块 $c_i$，找到其下游调用邻居 $\mathcal{N}(c_i)$，拼接为 $[c_i; \text{[DOWN]}; c_{out}]$，加上 segment type embedding 区分主代码和上下文；设计动机→调用图反映了代码实体间的运行时依赖关系，被调用函数的功能有助于理解调用者的语义。仅使用下游邻居（函数调用出去的），引入特殊 [DOWN] token 标识。

4. **文件路径前缀**：做什么→在每个代码块前添加其文件路径；核心思路→路径作为字符串直接拼接到 chunk 开头；设计动机→文件路径包含重要的语义和位置信息（如 `tests/test_api.py` 暗示这是 API 测试相关代码），模型通过训练学会利用这一信号。

5. **Mean Pooling 替代 [CLS]**：做什么→使用所有 token 的均值池化替代标准 [CLS] token 表示；设计动机→在早期实验中发现 mean pooling 带来适度性能提升。

### 损失函数 / 训练策略

- 损失函数：仓库级对数似然损失（见上），等价于多类分类交叉熵
- 负样本策略：实例内随机采样最多 1024 个负样本（同仓库内不相关的 chunk）
- 温度参数 $\tau = 0.05$
- 权重共享：query 和 code 编码器共享参数
- 训练数据：SWE-bench 的仓库级代码编辑问题，ground truth 来自解析 pull request

## 实验关键数据

### 主实验（表格）

**Perfect Recall@k（chunk 级别，完美检索率）**

| 模型 | SWE-bench Verified @5 | @20 | MRR | LCA @5 | @20 | MRR |
|------|----------------------|-----|-----|--------|-----|-----|
| CodeSage S | 0.34 | 0.51 | 0.35 | 0.26 | 0.34 | 0.28 |
| CoRet −CG | 0.52 | 0.69 | 0.52 | 0.32 | 0.41 | 0.45 |
| CoRet −CG +file | 0.54 | 0.69 | 0.52 | 0.29 | 0.38 | 0.44 |
| **CoRet** | **0.54** | **0.71** | **0.53** | **0.32** | **0.47** | **0.47** |

在 SWE-bench Verified 上，recall@5 相比 CodeSage S 提升 **52.9%**，recall@20 提升约 **35%**。

### 消融实验（表格）

**文件路径信息的影响（SWE-bench Verified，chunk Accuracy）**

| 模型 | 无文件路径 @5 | @20 | MRR | 有文件路径 @5 | @20 | MRR |
|------|-------------|-----|-----|-------------|-----|-----|
| BM25 | 0.15 | 0.21 | 0.14 | 0.16 | 0.22 | 0.16 |
| CodeSage S | 0.40 | 0.58 | 0.33 | 0.40 | 0.57 | 0.35 |
| **CoRet** | 0.42 | 0.58 | 0.42 | **0.53** | **0.70** | **0.53** |

**负样本数量的影响**：
- 从 8 增到 1024 个负样本，recall@20 提升近 10 个百分点
- 证明实例内负样本确实更有效

**调用图上下文消融**：
- CoRet −CG（无调用图）到 CoRet：LCA @20 从 0.41 → 0.47（+6pt）
- 用同文件随机 chunk 替代调用图邻居（CoRet −CG +file）反而降低了 LCA 性能，验证了调用图的针对性

### 关键发现

1. **现有模型严重不足**：即使是最好的 baseline CodeSage S，在 SWE-bench 上 recall@5 仅 34%，说明代码编辑检索与传统 code search 有本质区别。
2. **损失函数改进是最大贡献**：从 CodeSage S 到 CoRet −CG（仅改损失函数），recall@5 即从 0.34 跳至 0.52。
3. **文件路径是强信号**：CoRet 经训练学会了利用文件路径信息，去除路径后 recall@5 从 0.53 降至 0.42（BM25 和 CodeSage 不受影响因为它们没学到这个信号）。
4. **调用图对多文件编辑更有用**：在 LCA（多文件编辑更常见）上，调用图带来的提升（+6pt @20）比 SWE-bench（+2pt @20）更明显。
5. **实例内负样本优于跨实例负样本**：因为检索的核心是在同一仓库内区分，来自其他仓库的负样本提供的信号较弱。

## 亮点与洞察

- **问题定义清晰且实用**：明确指出代码编辑检索 ≠ 代码搜索，两者的语义对齐目标不同（issue→edit location vs query→code function）。
- **损失函数设计简洁有力**：从对比学习切换到对数似然，本质是从"学好的表征"到"学好的检索"的范式转变，改动小但效果显著。
- **消融实验设计出色**：每个设计选择都有对应的消融——调用图 vs 同文件随机 chunk、文件路径有无、负样本类型和数量——说服力强。
- **模型轻量但高效**：基于 CodeSage Small（小模型），通过训练策略而非模型规模取胜。

## 局限与展望

1. **仅支持 Python**：chunking 和调用图提取策略针对 Python，扩展到其他语言需要额外工程（但 SWE-PolyBench 已发布）。
2. **SWE-bench 多为单文件编辑**：文件级别 recall 可以很高，LCA 多文件编辑场景才是更好的 benchmark。
3. **仅使用编码器模型**：现代 LLM 经过修改也可输出嵌入（如 LLM2Vec），可能提供更好的 baseline 但训练成本更高。
4. **调用图邻居选择策略**：仅用下游邻居，利用图的拓扑属性（如中心性、社区结构）可能带来进一步提升。
5. **未端到端评估对代码编辑的影响**：只评估了检索性能，未评估检索结果如何影响下游代码编辑（如 pass@k on SWE-bench）。

## 相关工作与启发

- **CodeSage**（Zhang et al., 2024）：最强 baseline 和 CoRet 的 backbone 来源
- **RepoFusion**（Shrivastava et al., 2023）：仓库级代码理解的先驱
- **SWE-bench**（Jimenez et al., 2024）：定义了仓库级代码编辑的评测标准
- **调用图在代码理解中的作用**（Bansal et al., 2023）：为 CoRet 引入调用图提供了理论依据
- **Agentless**（Xia et al., 2024）和 **SWE-agent**（Yang et al., 2024）：代码编辑 agent，其性能受限于检索质量

## 评分

- **新颖性**: ★★★☆☆ — 各个组件（对数似然损失、调用图上下文、文件路径）都不算全新，但组合应用到代码编辑检索是有价值的
- **实验充分度**: ★★★★☆ — 消融实验全面，但数据集仅两个且都以 Python 为主
- **写作质量**: ★★★★☆ — 问题定义和动机阐述清晰，方法描述严谨
- **价值**: ★★★★☆ — 为 SWE-bench 等代码编辑任务提供了实用的检索增强基础设施

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] OASIS: Order-Augmented Strategy for Improved Code Search](oasis_order-augmented_strategy_for_improved_code_search.md)
- [\[ACL 2025\] GALLa: Graph Aligned Large Language Models for Improved Source Code Understanding](galla_graph_aligned_large_language_models.md)
- [\[ICML 2026\] Pull Requests as a Training Signal for Repo-Level Code Editing](../../ICML2026/code_intelligence/pull_requests_as_a_training_signal_for_repo-level_code_editing.md)
- [\[ACL 2025\] DynaCode: A Dynamic Complexity-Aware Code Benchmark for Evaluating Large Language Models in Code Generation](dynacode_a_dynamic_complexity-aware_code_benchmark_for_evaluating_large_language.md)
- [\[ACL 2026\] To Diff or Not to Diff? Structure-Aware and Adaptive Output Formats for Efficient LLM-based Code Editing](../../ACL2026/code_intelligence/to_diff_or_not_to_diff_structure-aware_and_adaptive_output_formats_for_efficient.md)

</div>

<!-- RELATED:END -->
