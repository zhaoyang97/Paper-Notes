---
title: >-
  [论文解读] Alpha-SQL: Zero-Shot Text-to-SQL using Monte Carlo Tree Search
description: >-
  [ICML 2025][自监督学习][Text-to-SQL] Alpha-SQL 将零样本 Text-to-SQL 建模为树搜索问题，通过蒙特卡洛树搜索 (MCTS) 框架结合 LLM-as-Action-Model 和自监督奖励函数，无需微调即可在 BIRD 数据集上以 32B 开源模型达到 69.7% 执行精度，超越基于 GPT-4o 的零样本 SOTA 2.5 个百分点。
tags:
  - "ICML 2025"
  - "自监督学习"
  - "Text-to-SQL"
  - "蒙特卡洛树搜索 (MCTS)"
  - "大语言模型"
  - "零样本推理"
  - "测试时计算"
---

# Alpha-SQL: Zero-Shot Text-to-SQL using Monte Carlo Tree Search

**会议**: ICML 2025  
**arXiv**: [2502.17248](https://arxiv.org/abs/2502.17248)  
**代码**: [HKUSTDial/Alpha-SQL](https://github.com/HKUSTDial/Alpha-SQL)  
**领域**: 自监督学习  
**关键词**: Text-to-SQL, 蒙特卡洛树搜索 (MCTS), 大语言模型, 零样本推理, 测试时计算

## 一句话总结

Alpha-SQL 将零样本 Text-to-SQL 建模为树搜索问题，通过蒙特卡洛树搜索 (MCTS) 框架结合 LLM-as-Action-Model 和自监督奖励函数，无需微调即可在 BIRD 数据集上以 32B 开源模型达到 69.7% 执行精度，超越基于 GPT-4o 的零样本 SOTA 2.5 个百分点。

## 研究背景与动机

Text-to-SQL 任务旨在将自然语言查询转换为 SQL 语句，在数据分析和数据库交互中有重要应用价值。当前 LLM-based 方法分为两类：

**微调方法**：在标注数据上微调 LLM（如 CodeS、CHESS-SQL），效果好但成本高，新模型出现后需重复训练流程

**零样本方法**：利用 LLM 预训练知识直接生成 SQL（如 DAIL-SQL、C3），免去标注依赖但性能有限

**核心挑战**：零样本场景下，模型难以将通用预训练知识迁移到特定的 SQL 生成任务。LLM 在处理自然语言与数据库 schema 之间的复杂映射时，难以准确理解 schema 关系、构造复杂 SQL、并在不同上下文中保持鲁棒性。

**本文动机**：将 SQL 生成任务分解为更小的可管理子任务，通过渐进式构造（progressive construction）逐步构建 SQL，在每一步提供上下文指导，降低单步复杂度。这一过程自然可以建模为树搜索问题。

## 方法详解

### 整体框架

Alpha-SQL 将 Text-to-SQL 建模为树结构搜索空间上的搜索问题，核心思路是将 SQL 查询的构造过程表示为从根节点到叶节点的路径：

- **节点 (V)**：表示 SQL 构造过程中的部分推理状态。根节点 $v_0$ 为空查询状态（含问题 $q$ 和数据库 schema $\mathcal{D}$），中间节点存储增量推理步骤，叶节点为完整 SQL 或终止状态
- **边 (E)**：表示 SQL 构造动作（如选择表、添加条件、应用聚合函数等）
- **路径**：从根到叶的路径对应一个完整的推理轨迹，产生一个候选 SQL：$y = v_0 \oplus v_1 \oplus \cdots \oplus v_t$

搜索空间 $|\mathcal{S}|$ 随数据库 schema 和查询复杂度呈指数增长（表数量、列数量、JOIN 条件、嵌套子查询等），穷举搜索不可行。Alpha-SQL 利用 MCTS 框架高效探索这一空间。

### 关键设计

#### 1. LLM-as-Action-Model

在 MCTS 的每一步，Alpha-SQL 调用 LLM 作为动作模型，基于当前上下文（问题、schema、已有推理轨迹）生成推理动作。每步推理以 Chain-of-Thought 形式存储在节点中，保持上下文连贯性：

$$v_{i+1} = LLM(q, \mathcal{D}, Actions(v_0, \ldots, v_i), Prompt(a_i))$$

**动作空间定义**：共 7 种 SQL 构造推理动作，每个动作有专门的 prompt 模板：

| 动作 | 名称 | 功能描述 |
|------|------|----------|
| $A_1$ | Question Rephrasing | 将模糊问题分解为结构化的"条件列表+问题"格式，消除歧义 |
| $A_2$ | Schema Selection | 通过 CoT 推理从完整 schema 中识别出与问题相关的表和列子集 |
| $A_3$ | Column Value Identification | 识别 WHERE 子句所需的列值过滤条件（如 "Bob" → name='Bob'） |
| $A_4$ | Column Function Identification | 识别需要的聚合/标量函数（如 COUNT、STRFTIME 等） |
| $A_5$ | SQL Generation | 采用 Divide-and-Conquer CoT 策略，将复杂查询分解为子任务并组合 |
| $A_6$ | SQL Revision | 基于执行结果反馈的多轮纠错，最多 $N_{revision}$ 轮 |
| $A_7$ | Termination | 终止推理轨迹，必须跟在 $A_5$ 或 $A_6$ 之后 |

**动作顺序约束**：动作间存在严格的先后顺序关系（通过转移矩阵定义有效转移），确保逻辑一致性。每个动作在单条推理轨迹中最多出现一次，防止无限循环。经计算，有效推理路径超过 3000 条。

#### 2. MCTS 候选 SQL 生成

每次 MCTS rollout 包含四个经典阶段：

**(1) Selection（选择）**：从根节点开始，使用 UCT（Upper Confidence Bound for Trees）公式选择节点：

$$UCT(v, a) = \frac{Q(v, a)}{N(v, a)} + c \sqrt{\frac{\ln N(v)}{N(v, a)}}$$

其中 $N(v, a)$ 为动作访问次数，$Q(v, a)$ 为累计奖励。若存在未访问子节点（$N(v,a)=0$），优先选择这些节点。

**(2) Expansion（扩展）**：根据当前节点类型确定有效后续动作，每个动作采样 $N_{expansion}$ 次（温度 $T_{expansion}$），生成 $N_{expansion} \times |E_{valid}|$ 个子节点。**关键剪枝**：若多次采样的最终结果相同（如 Schema Selection 选出相同子集），仅保留一个节点，大幅减少分支因子。

**(3) Simulation（模拟）**：迭代选择和扩展直到到达终止节点，所有新扩展节点持久保留在树中。

**(4) Backpropagation（回传）**：在终止节点处评估预测 SQL，从终止节点回溯到根节点，更新路径上所有节点的 $Q$ 和 $N$ 值：

$$Q(v, a) = Q(v, a) + r, \quad N(v) = N(v) + 1$$

**最终 SQL 选择**：完成 $N_{rollout}$ 次 rollout 后，收集所有到达终止节点的完整轨迹，执行所有候选 SQL，选择执行结果一致性最高的作为最终输出。

#### 3. 自监督奖励函数

奖励函数是 MCTS 的核心评估机制。传统方法（Outcome Reward Model、Progress Reward Model）需要标注数据训练，难以泛化。

**核心直觉**：类似于人类推理中的置信度 —— 一个有信心的人在多次尝试中会给出一致的答案（高一致性=高置信=高质量）。

**实现方式**：对每条推理路径的预测 SQL $y$，用高温采样生成 $N_{reward}$ 个候选 SQL $\{y_i\}$，过滤掉无效查询后计算自一致性得分：

$$R(y, q, \mathcal{D}) = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\text{Execute}(y, \mathcal{D}) = \text{Execute}(y_i, \mathcal{D})]$$

该函数完全自监督，不需要任何标注数据。

#### 4. 数据库值检索

为解决用户表述与数据库存储值之间的语义差异（如 "America" vs "United States"），采用两阶段方法：

- **离线预处理**：对 TEXT 类型列值用 MinHash 生成签名并本地存储
- **在线检索**：从用户问题中提取关键词，用 LSH（Locality Sensitive Hashing）匹配 MinHash 签名，经编辑距离阈值 $\epsilon_{edit}$ 和语义相似度阈值 $\epsilon_{semantic}$ 过滤后注入 schema prompt

### 损失函数 / 训练策略

Alpha-SQL 不涉及任何模型训练或微调，全部在推理时完成。核心超参数设定：

| 参数 | 值 | 说明 |
|------|-----|------|
| $N_{rollout}$ | 24 | MCTS 搜索轮次 |
| $N_{expansion}$ | 3 | 每个动作的采样次数 |
| $T_{expansion}$ | 0.8 | 扩展阶段采样温度 |
| $N_{reward}$ | 5 | 奖励计算采样次数 |
| $T_{reward}$ | 1.0 | 奖励计算采样温度 |
| $N_{revision}$ | 10 | SQL Revision 最大迭代次数 |
| $\epsilon_{edit}$ | 0.3 | 编辑距离相似度阈值 |
| $\epsilon_{semantic}$ | 0.6 | 语义相似度阈值 |

## 实验关键数据

### 主实验

评估数据集：BIRD（1534 样本，更复杂）和 Spider（1034 样本），指标为执行精度 (EX)。

**BIRD Development Set 结果**：

| 方法 | 推理模型 | 零样本 | 总体 EX (%) |
|------|----------|--------|-------------|
| CodeS (SFT) | CodeS-7B | ✗ | 57.0 |
| CHESS-SQL | DS-Coder-33B | ✗ | 65.0 |
| Distillery | GPT-4o | ✗ | 67.2 |
| RSL-SQL | GPT-4o | ✓ | 67.2 |
| **Alpha-SQL** | **Qwen2.5-7B** | **✓** | **66.8** |
| **Alpha-SQL** | **Qwen2.5-14B** | **✓** | **68.4** |
| **Alpha-SQL** | **Qwen2.5-32B** | **✓** | **69.7** |

**Spider Development Set 结果**：

| 方法 | 推理模型 | 零样本 | All EX (%) |
|------|----------|--------|------------|
| CodeS (SFT) | CodeS-15B | ✗ | 84.9 |
| DIN-SQL | GPT-4 | ✓ | 82.8 |
| SuperSQL | GPT-4 | ✓ | 87.0 |
| **Alpha-SQL** | **Qwen2.5-14B** | **✓** | **87.0** |

### 消融实验

在 SDS（Subsampled Development Set, 147 样本）上以 Qwen2.5-Coder-7B 进行：

**动作空间消融**：

| 配置 | EX (%) | 变化 | 说明 |
|------|--------|------|------|
| Alpha-SQL (完整) | 64.6 | — | 全部 7 个推理动作 |
| w/o $A_1$ (Question Rephrasing) | 63.9 | −0.7 | 问题改写有助消除歧义 |
| w/o $A_2$ (Schema Selection) | 63.1 | −1.5 | Schema 筛选重要性较高 |
| w/o $A_3$ (Column Value Identification) | 64.2 | −0.4 | 列值识别有一定贡献 |
| w/o $A_4$ (Column Function Identification) | 64.0 | −0.6 | 函数识别有中等贡献 |
| w/o $A_6$ (SQL Revision) | 62.8 | −1.8 | **影响最大**，执行反馈纠错至关重要 |

**与 Baseline LLM 对比（SDS 数据集，直接 prompting）**：

| 模型 | EX (%) | 备注 |
|------|--------|------|
| QwQ-32B-Preview | 38.8 | 推理模型 |
| Phi-4 | 43.5 | — |
| Qwen2.5-Coder-7B | 47.6 | — |
| DeepSeek-R1 | 50.3 | 推理模型 |
| Deepseek-V3 | 51.2 | — |
| GPT-4o | 53.7 | — |
| Gemini-1.5-Pro | 56.2 | — |
| Gemini-2.0-Flash-Thinking | 60.8 | 推理模型 |
| **Phi-4 + Alpha-SQL** | **60.0** | **+16.5** |
| **Qwen-7B + Alpha-SQL** | **64.6** | **+17.0，超越所有 baseline** |

### 关键发现

1. **7B 模型 + Alpha-SQL 超越 GPT-4o 直接推理**：Qwen2.5-Coder-7B（47.6%）加 Alpha-SQL 后达到 64.6%，远超 GPT-4o 的 53.7%
2. **性能随 rollout 数增加而提升**：MCTS rollout 数与上界精度和最终精度正相关，仅 24 次 rollout 即可在 3000+ 可能路径中高效找到高质量 SQL
3. **SQL Revision 是最关键动作**：去掉后下降 1.8%，说明数据库执行反馈在 Text-to-SQL 中不可或缺
4. **plug-and-play 优势**：Alpha-SQL 对不同底座模型（Qwen、Phi-4）均有 15-17% 的提升，验证了框架通用性

## 亮点与洞察

1. **问题建模创新**：将 Text-to-SQL 从"一步生成"重新建模为"树搜索问题"，自然融入 MCTS 框架，是 test-time compute scaling 在 structured data 领域的优秀应用
2. **自监督奖励函数**：基于执行结果自一致性的奖励设计巧妙地绕开了对标注数据的依赖，启发了 zero-shot 场景下的通用评估思路
3. **小模型大能力**：用搜索框架补偿模型能力不足，7B 模型媲美甚至超越 GPT-4o，展示了 inference-time scaling 的巨大潜力
4. **动作空间设计合理**：7 种动作覆盖了 Text-to-SQL 的关键难点（schema 理解、值匹配、函数选择、纠错），动作排序约束保证了推理轨迹的逻辑一致性
5. **剪枝策略实用**：对结果相同的扩展节点去重，在不损失信息的前提下大幅减少搜索成本

## 局限与展望

1. **推理成本高**：24 次 MCTS rollout 中每步都需要调用 LLM，每个问题的推理成本远高于单次生成，限制了实时交互场景的适用性
2. **依赖外部 embedding 模型**：数据库值检索的语义匹配依赖 OpenAI text-embedding-3-large，增加了对外部服务的依赖
3. **动作空间固定**：7 种动作为人工设计，是否为最优集合、是否可自动发现更好的动作组合值得探索
4. **评估局限**：仅在 BIRD 和 Spider 上评估，未测试更多样化的数据库场景（如多数据库、流式 SQL 等）
5. **可扩展方向**：可探索将 MCTS 与强化学习结合进行在线自我改进，或引入 process reward model 替代自一致性奖励

## 相关工作与启发

- **CHASE-SQL**：多步管线生成+验证候选 SQL，但依赖微调 Gemini；Alpha-SQL 用 MCTS 动态替代了静态管线
- **rStar**：MCTS + LLM 用于通用推理任务的先驱工作；Alpha-SQL 将其适配到 Text-to-SQL 的特定挑战
- **Tree of Thoughts**：树搜索增强 LLM 推理的方向；Alpha-SQL 将树搜索+SQL 执行反馈结合得更深入
- **Test-time computation scaling**：本文是 inference-time scaling 在数据库任务上的成功案例，进一步验证了"搜索 > 更大模型"的趋势

## 评分

| 维度 | 评分 (1-5) | 说明 |
|------|-----------|------|
| 新颖性 | ⭐⭐⭐⭐ | MCTS 用于 Text-to-SQL 是新颖组合，自监督奖励设计巧妙 |
| 技术深度 | ⭐⭐⭐⭐ | 框架设计完整，动作空间、剪枝、奖励函数环环相扣 |
| 实验充分性 | ⭐⭐⭐⭐ | 多模型、多 benchmark、消融实验全面 |
| 实用价值 | ⭐⭐⭐⭐ | plug-and-play 且无需微调，部署门槛低 |
| 写作质量 | ⭐⭐⭐⭐ | 结构清晰，图表丰富，公式规范 |
| **综合** | **⭐⭐⭐⭐** | **零样本 Text-to-SQL 的标杆工作，MCTS + LLM 的优秀实践** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] CObL: Toward Zero-Shot Ordinal Layering without User Prompting](../../ICCV2025/self_supervised/cobl_toward_zero-shot_ordinal_layering_without_user_prompting.md)
- [\[ICML 2026\] From Zero to Hero: Advancing Zero-Shot Foundation Models for Tabular Outlier Detection](../../ICML2026/self_supervised/from_zero_to_hero_advancing_zero-shot_foundation_models_for_tabular_outlier_dete.md)
- [\[ICML 2025\] Towards Benchmarking Foundation Models for Tabular Data With Text](towards_benchmarking_foundation_models_for_tabular_data_with_text.md)
- [\[ICML 2026\] InfoAtlas: A Foundation Model for Zero-Shot Statistical Dependence Estimation](../../ICML2026/self_supervised/infoatlas_a_foundation_model_for_zero-shot_statistical_dependence_estimate.md)
- [\[NeurIPS 2025\] T-REGS: Minimum Spanning Tree Regularization for Self-Supervised Learning](../../NeurIPS2025/self_supervised/t-regs_minimum_spanning_tree_regularization_for_self-supervised_learning.md)

</div>

<!-- RELATED:END -->
