---
title: >-
  [论文解读] LearNAT: Learning NL2SQL with AST-guided Task Decomposition for Large Language Models
description: >-
  [ICLR 2026][代码智能][NL2SQL] LearNAT 用 AST 引导的 MCTS 搜索自动合成"可验证"的 NL2SQL 任务分解数据，再用感知 margin 的 DPO 做细粒度多步偏好优化，让一个 7B 小模型在 NL2SQL 上达到接近 GPT-4 的水平。 领域现状：把自然语言查询翻译成可执行 SQL…
tags:
  - "ICLR 2026"
  - "代码智能"
  - "NL2SQL"
  - "任务分解"
  - "抽象语法树(AST)"
  - "蒙特卡洛树搜索(MCTS)"
  - "偏好优化(DPO)"
---

# LearNAT: Learning NL2SQL with AST-guided Task Decomposition for Large Language Models

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=q6kXd8Gpfj](https://openreview.net/forum?id=q6kXd8Gpfj)  
**代码**: [https://github.com/MrBlankness/LearNAT](https://github.com/MrBlankness/LearNAT)  
**领域**: 代码智能 / NL2SQL / LLM 后训练  
**关键词**: NL2SQL, 任务分解, 抽象语法树(AST), 蒙特卡洛树搜索(MCTS), 偏好优化(DPO)  

## 一句话总结
LearNAT 用 AST 引导的 MCTS 搜索自动合成"可验证"的 NL2SQL 任务分解数据，再用感知 margin 的 DPO 做细粒度多步偏好优化，让一个 7B 小模型在 NL2SQL 上达到接近 GPT-4 的水平。

## 研究背景与动机
**领域现状**：把自然语言查询翻译成可执行 SQL（NL2SQL）能让不懂 SQL 的人直接查数据库。当前 SOTA（C3-SQL、DIN-SQL、SuperSQL 等）几乎都靠 GPT-4 这类大型私有模型，配合复杂 prompt 工程和 test-time scaling 来堆性能。

**现有痛点**：依赖私有大模型带来两个硬伤——一是封闭、不可复现、有数据隐私顾虑；二是 test-time scaling 让推理成本高得离谱（论文里 Alpha-SQL 风格的推理搜索会把单条查询的 token 从 1.8K 暴涨到 204.5K）。因此本文转向一个更务实的方向：在资源受限场景下，把**小规模开源 LLM** 的"模型级"性能本身做强。

**核心矛盾**：作者的探索实验发现，复杂 NL2SQL 查询可以拆成"高层任务分解 + 低层 NL2SQL 翻译"两件事。LLM 因为预训练充分，**翻译**这件事干得不错；但**分解**这件事干得很差——当人工把子任务喂给模型时性能猛涨 30.4%，可一旦让模型自己分解，只涨 3.4%。更糟的是，让模型把 sub-SQL 翻译回自然语言子任务时会因幻觉引入错误，而用 Sentence-Transformer / GLM-4 去判断子任务对不对，准确率只有 46.8% / 36.0%，根本不可靠（比如金标准查"用户名"、生成的子任务查"用户ID"，GLM-4 还判它对）。

**本文目标**：构建一个**可验证**的任务分解框架，再用强化学习把分解能力刻进小模型里。

**核心 idea**：**用程序化的 AST（抽象语法树）替代不可靠的 LLM 自评**——既用 AST 引导搜索、剪枝、给节点打分来合成高质量分解数据，又用 AST 相似度算出的 margin 来做细粒度偏好学习。

## 方法详解

### 整体框架
LearNAT 分两阶段离线流程：先用 **Decomposition Synthesis Procedure（分解合成过程）** 跑 AST 引导的 MCTS，从训练集里挖出"成功分解轨迹"（给 SFT 用）和"对比节点对"（给偏好学习用）；再用 **Margin-Aware Reinforcement Learning（感知 margin 的强化学习）** 先 SFT 暖身、再用带 AST margin 的 DPO 微调目标小模型。默认用 GLM-4-Plus 合成数据，微调 Qwen2.5-Coder。

```mermaid
graph LR
    A[复杂NL查询Q + Gold SQL Y] --> B[Gold SQL 的 AST]
    A --> C[MCTS 逐步预测子任务]
    B -.AST引导剪枝/打分.-> C
    C --> D[节点分类: 进展/冗余/无效]
    D --> E[成功分解轨迹 → SFT]
    D --> F[对比节点对 → DPO]
    E --> G[Warm-up SFT]
    F --> H[DPO with AST Margin]
    G --> H
    H --> I[强化分解能力的小模型]
```

### 关键设计

**1. AST 引导的 MCTS 搜索：用语法树把"无限文本动作空间"压成可验证的搜索。** 分解被建模成一棵蒙特卡洛树：根节点是原查询 $Q$，每条根到叶的路径就是一个分解序列，每个状态 $s_i=\{q_i, y_i, AT(y_i), AT^{sum}(y_i), R(s_i)\}$ 同时记录该步子任务 $q_i$、对应 sub-SQL $y_i$、它的 AST，以及从根累加到当前的合并 AST $AT^{sum}(y_i)=(\bigcup_j N(AT(y_j)), \bigcup_j E(AT(y_j)))$。关键在于：传统 MCTS 动作集有限（"拿起""放下"），但 LLM 生成文本动作是无限的——同一条 sub-SQL 能写成无数种自然语言说法，搜索空间爆炸。LearNAT 用 Gold SQL 的 AST 当"地图"来约束这个空间，让搜索有据可依。

**2. 基于子树关系的节点分类与剪枝：把无意义的搜索分支提前砍掉。** 每个节点按它的 AST 与 Gold SQL AST 的关系分三类——**进展节点**（$AT(y_i)$ 是 $AT(Y)$ 的子树、但不是父节点累加 AST 的子树，说明带来了新信息）、**冗余节点**（是 $AT(Y)$ 子树但也已被父节点覆盖，没新贡献）、**无效节点**（根本不是 $AT(Y)$ 的子树，即错误分解）。子树关系用集合包含严格定义：$isSubtree(AT_1,AT_2)=1$ 当且仅当 $N_1\subseteq N_2 \wedge E_1\subseteq E_2$。因为有效子任务序列只对应路径上的进展动作，遇到冗余/无效节点就直接终止该分支的扩展——这正是把 naive MCTS 的 33 万 token 成本压到 13 万（降 56.38%）的关键。

**3. 基于 AST 相似度的规则化奖励估计：彻底丢掉不靠谱的 LLM 自评。** 别的 LLM-MCTS 方法靠模型自己估节点奖励，但 GPT-4 在 BIRD 上自评也才 46.35% 准确率。LearNAT 改用纯规则：只对进展节点算奖励 $R(s_i)=sim(AT^{sum}(y_i), AT(Y))$，且把相似度拆成节点级与结构级两部分加权——$R(s_i)=\alpha\cdot sim_{node}+(1-\alpha)\cdot sim_{struct}$，其中节点级看节点重叠度、结构级用树编辑距离衡量。这样奖励既可解释又可验证。配合"自改进示范池"（用前几轮成功分解的样本，按 AST 相似度选 top-3 当 few-shot），分解成功率从 CoT 的 59.07% 提到 80.00%。

**4. 带 AST Margin 的 DPO：让模型不仅知道"哪步更好"，还知道"好多少"。** SFT 暖身先用成功轨迹做监督微调，最小化标准对数似然 $L_{SFT}=-\mathbb{E}_{(x,t)}[\sum_i \log p_\theta(t_i|t_{1:i-1},x)]$，把分解格式和基础能力刻进去。但纯 SFT 有"悲观"问题：正反馈会连带把错误推理路径的概率也抬高，所以再用 DPO 压制错误子任务。普通 DPO 把所有正负步同等对待，对多步推理太粗。LearNAT 的做法是把合成阶段算出的 AST 奖励差直接当 margin 塞进损失：$margin=R(s_i^w)-R(s_i^l)$，损失变成 $L_{MDPO}=-\mathbb{E}[\log\sigma(\hat{r}_\theta(x,y^w)-\hat{r}_\theta(x,y^l)-\triangle r)]$。这个 offset 让模型学到偏好的"幅度"而非只学"方向"，且无需额外训练奖励模型——margin 完全复用搜索阶段的 AST 相似度。

## 实验关键数据

### 主实验表格
BIRD-dev（域内）与 Spider-dev（域外）执行准确率（EX），节选：

| 方法 | 类别 | 骨干 | BIRD Total | Spider Total |
|------|------|------|-----------|-------------|
| SuperSQL (VLDB'24) | System | GPT-4 | 58.5 | 87.0 |
| MAC-SQL (COLING'25) | System | GPT-4 | 59.4 | 86.7 |
| CodeS (SIGMOD'24) | Model | CodeS-7B | 57.0 | 85.4 |
| CodeS | Model | CodeS-15B | 58.5 | 84.9 |
| **LearNAT** | Model | Qwen2.5-Coder-7B | **58.1** | **86.4** |
| **LearNAT** | Model | Qwen2.5-Coder-14B | 61.2 | 86.9 |
| **LearNAT** | Model | Qwen2.5-Coder-32B | **65.0** | **88.4** |

LearNAT(7B) 单次前向就超过所有 model-level 基线，并优于多数 system-level 方法；32B 版在 Spider 上 88.4%，超第二名 SuperSQL 1.4%。

### 消融实验表格
Qwen2.5-Coder-7B 为骨干，绿色为移除后的性能损失：

| 消融 | BIRD Total | Spider Total |
|------|-----------|-------------|
| LearNAT（完整） | 58.1 | 86.4 |
| w/o LearNAT（整体去掉） | 47.5 (↓10.6) | 77.0 (↓9.4) |
| → 仅 DPO | 52.8 (↓5.3) | 79.0 (↓7.4) |
| → 仅 CoT 分解 | 49.3 (↓8.7) | 79.4 (↓7.0) |
| w/o AST Guide | 53.1 (↓5.0) | 79.9 (↓6.5) |
| w/o SFT | 54.5 (↓3.6) | 81.0 (↓5.3) |
| w/o MDPO | 53.7 (↓4.4) | 80.9 (↓5.4) |
| MDPO → 普通 DPO | 56.6 (↓1.4) | 85.1 (↓1.3) |

每个组件去掉都掉点；AST 引导和 MDPO 的 margin 设计都被验证有效。

### 关键发现
- **分解合成过程**：成功率 80.00%，比 CoT 高 20.93%、比 naive MCTS 高 8.45%，且 token 成本比 naive MCTS 降 56.38%。
- **自改进示范**：首轮提升 1.99%，但第 2/3 轮仅 +0.4%/+0.27%，边际递减，故 3 轮后停止。
- **同协议公平对比**：在 SynCoT / SQL-o1 / Alpha-SQL / OmniSQL 各自协议下，LearNAT 综合表现普遍更优（如套用 Alpha-SQL 协议 BIRD 达 68.4%）。
- **数据效率**：仅用 BIRD-train 合成 7.2K 条带可验证中间子任务的数据，就胜过 OmniSQL 的 250 万条合成数据——质量胜过数量。

## 亮点与洞察
- **用"程序真值"替代"模型自评"是核心洞察**：NL2SQL 有 Gold SQL 这个可解析的程序结构，AST 让"对不对""差多少"都变成可计算的规则，绕开了 LLM 自评和语义相似度判别都不可靠的死结。
- **把"搜索得到的奖励"直接喂回偏好优化**：合成阶段算的 AST margin 一物两用，既剪枝又当 DPO 的 offset，省掉单独训练奖励模型，pipeline 很干净。
- **模型级 vs 系统级的成本论述很到位**：明确指出 7B 单次前向就能逼近靠多候选+精化+一致性检查堆 token 的系统级方法，性能/成本权衡说服力强。

## 局限与展望
- **依赖 Gold SQL**：整套可验证分解建立在训练集有标准 SQL 的前提上，AST 引导本质是有监督的，迁移到没有 Gold SQL 的开放场景会受限。
- **错误分析暴露短板**：50 个失败案例里 schema linking（25）、浮点计算、未知规则、错误答案仍占大头，说明 AST 验证管不住语义层面的 schema 对齐错误。
- **自改进示范边际递减**：超过 1 轮收益就很小，分解能力的上限可能受合成模型 GLM-4-Plus 本身限制。
- **AST margin 的权重 $\alpha$、相似度定义偏启发式**，对不同 SQL 方言/复杂嵌套查询的鲁棒性还需更多验证。

## 相关工作与启发
- **NL2SQL 路线**：系统级（DIN-SQL、SuperSQL、MAC-SQL）靠 GPT-4 + 复杂 pipeline；模型级（CodeS、SENSE、OmniSQL）靠微调小模型。LearNAT 属后者，但用"可验证分解"切入。
- **LLM + 任务分解 / 推理**：与 CoT、SQL-o1、Alpha-SQL 的推理时搜索互补——LearNAT 把搜索成本前置到训练阶段，推理时单次前向，对部署友好。
- **偏好优化**：在 DPO 基础上引入步级 margin，思路可推广到其他有程序化真值的多步推理任务（代码生成、定理证明、agent 规划），凡是能定义"中间步可验证奖励"的场景都能借鉴这套"搜索打分→margin DPO"范式。

## 评分
- **新颖性**: ⭐⭐⭐⭐ — 用 AST 把 NL2SQL 分解变成可验证搜索 + 把搜索奖励复用为 DPO margin，组合很巧，是干净的"程序真值替代模型自评"思路。
- **实验充分度**: ⭐⭐⭐⭐ — 两数据集、三模型规模、系统级/模型级双对比、同协议公平复测、消融与错误分析齐全。
- **写作质量**: ⭐⭐⭐⭐ — 探索实验→两条 Observation→方法的动机链条清晰，公式与图配合到位。
- **价值**: ⭐⭐⭐⭐ — 让 7B 开源模型逼近 GPT-4 且推理零额外开销，对资源受限的可复现 NL2SQL 部署有实用价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Personality-Guided Code Generation Using Large Language Models](../../ACL2025/code_intelligence/personality_guided_code_gen.md)
- [\[ICLR 2026\] Evolving Graph Structured Programs for Circuit Generation with Large Language Models](evolving_graph_structured_programs_for_circuit_generation_with_large_language_mo.md)
- [\[ICLR 2026\] CrossPL: Systematic Evaluation of Large Language Models for Cross Programming Language Interoperating Code Generation](crosspl_systematic_evaluation_of_large_language_models_for_cross_programming_lan.md)
- [\[ICLR 2026\] Local Success Does Not Compose: Benchmarking Large Language Models for Compositional Formal Verification](local_success_does_not_compose_benchmarking_large_language_models_for_compositio.md)
- [\[ICLR 2026\] Training Large Language Models To Reason In Parallel With Global Forking Tokens](training_large_language_models_to_reason_in_parallel_with_global_forking_tokens.md)

</div>

<!-- RELATED:END -->
