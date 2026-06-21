---
title: >-
  [论文解读] MemGen: Weaving Generative Latent Memory for Self-Evolving Agents
description: >-
  [ICLR 2026][LLM Agent][生成式记忆] MemGen 让 LLM 智能体在推理过程中由一个「记忆触发器」实时判断何时需要回忆，再由一个「记忆编织器」生成一段机器原生的潜在 token 序列注入推理流，从而把记忆与思考交织成动态循环，在冻结主干、不改一个参数的前提下显著超越参数式与检索式记忆。
tags:
  - "ICLR 2026"
  - "LLM Agent"
  - "生成式记忆"
  - "潜在记忆"
  - "自进化智能体"
  - "元认知触发"
  - "LoRA"
  - "强化学习"
---

# MemGen: Weaving Generative Latent Memory for Self-Evolving Agents

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=vI56m4Iu4e](https://openreview.net/forum?id=vI56m4Iu4e)  
**代码**: [https://github.com/bingreeky/MemGen](https://github.com/bingreeky/MemGen)  
**领域**: LLM Agent / 智能体记忆 / 自进化  
**关键词**: 生成式记忆, 潜在记忆, 自进化智能体, 元认知触发, LoRA, 强化学习  

## 一句话总结
MemGen 让 LLM 智能体在推理过程中由一个「记忆触发器」实时判断何时需要回忆，再由一个「记忆编织器」生成一段机器原生的潜在 token 序列注入推理流，从而把记忆与思考交织成动态循环，在冻结主干、不改一个参数的前提下显著超越参数式与检索式记忆。

## 研究背景与动机
**领域现状**：让 LLM 智能体从环境交互中自我进化，目前主要靠两类记忆。一是**参数式记忆**（SFT / GRPO / DPO 等），把经验直接烧进模型参数；二是**检索式记忆**（Mem0、AWM、ExpeL 等），把经验外化成结构化数据库，用时再检索拼进上下文。

**现有痛点**：参数式记忆改动主干参数，不可避免地带来灾难性遗忘，侵蚀通用知识；检索式记忆虽不侵入参数，却高度依赖「上下文工程」，走的是一次性检索、拼接、再执行的僵硬流水线，记忆只是被「贴」在 query 前，无法与推理真正融合。即便是已有的潜在记忆方法（KV-cache 类只解决长上下文，潜在 embedding 类仍要改 LLM 参数），也都停留在「按相似度检索」而非「生成式重构」，缺少人脑那种思考与回忆相互塑形的连续交互。

**核心矛盾**：智能体记忆要么改参数（遗忘）、要么靠拼接（僵硬），始终无法做到「在推理的任意关键节点，按需生成贴合当下状态的记忆并无缝融入思考」。

**本文目标**：把智能体记忆设计成一种**动态认知能力**——能在推理途中流式、重构式地生成记忆，并与推理交织演化。

**核心 idea**：**「触发 + 编织」双组件**——用 RL 训练一个元认知监控器在 token 级别决定「何时回忆」，再用一个生成器把当前认知状态重构成定长潜在记忆序列注入推理，主干 LLM 全程冻结，记忆知识只内化进编织器。

## 方法详解

### 整体框架
MemGen 在冻结的核心推理器 $\pi_\theta$ 自回归生成的同时，逐 token 监控其内部隐藏状态：触发器 $\mathcal{T}_{trigger}$ 判断当前是否需要「回忆」，一旦决定 INVOKE 就暂停生成，由编织器 $\mathcal{W}_{weaver}$ 以当前认知状态为刺激、生成一段潜在记忆 token 序列 $M_t$，前置注入后推理器再恢复生成。整个过程是「生成→监控→触发→编织→重整合」的递归循环，主干参数一动不动。

```mermaid
flowchart LR
    A[状态 s_t] --> B[冻结推理器 π_θ<br/>逐 token 生成]
    B --> C{触发器 T_trigger<br/>读隐藏状态 H_t,<j}
    C -->|SKIP| B
    C -->|INVOKE 暂停| D[编织器 W_weaver<br/>重构潜在记忆 M_t]
    E[(外部记忆库<br/>可选)] -.-> D
    D --> F[M_t 前置注入<br/>z_t,j ~ π_θ·|s_t,z,M_t]
    F --> B
```

### 关键设计

**1. 交织式记忆-推理循环：把记忆从「开场白」变成「随叫随到的伴随者」**。传统框架只在任务起点检索一次记忆，MemGen 则让记忆在 token 级别动态参与。推理器对状态 $s_t$ 生成动作 $a_t=(z_{t,1},\dots,z_{t,L_t})$ 时，每生成一个 token 都产生对应隐藏状态序列 $H_{t,<j}=(h_{t,1},\dots,h_{t,j-1})$；触发器据此算出调用概率，若需回忆则编织器产出定长 $M_t\in\mathbb{R}^{K\times d_{model}}$，其隐藏状态被前置到 $H_{t,<j}$，推理器在富化上下文上继续：$z_{t,j}\sim\pi_\theta(\cdot\mid s_t,z_{t,<j},M_t)$。关键在于 $M_t$ 不是对历史内容的逐字复述，而是经编织器过滤、整合的**选择性重构**，类比人脑的记忆巩固——这让推理从线性展开升级为「与记忆的递归对话」，且因 $\pi_\theta$ 冻结而天然保住通用能力、规避遗忘。

**2. 记忆触发器：用元认知信号学会「何时该回忆」**。触发器实例化为挂在 $\pi_\theta$ 上的轻量 LoRA 适配器，输入是隐藏状态而非文本（潜在 embedding 保留了 softmax 解码后会丢失的上下文敏感信息），输出调用概率 $p_j=\sigma(\mathcal{T}_{trigger}(h_{t,1},\dots,h_{t,j-1}))$，再采样二元决策 $d_j\sim\mathrm{Bernoulli}(p_j)$。为控开销，它采用**句子粒度激活**：只在分隔符集合 $D$（逗号、句号等）处才计算，$z_j\notin D$ 时 $p_j=0$，把干预限制在语义边界以保解码效率。训练用 RL，目标是「在关键处稀疏调用」，因此引入**奖励自适应惩罚**：$\max_\phi\ \mathbb{E}[R(\tau_i)-\lambda\sum_{i,j}\max(0,\tilde d_{i,j}-\bar p)]$，其中 $\bar p$ 是高奖励轨迹（奖励超过批次中位数）上的平均激活概率，从而压制无谓激活又保留必要回忆。

**3. 记忆编织器：把经验内化进自己、生成式合成潜在记忆**。编织器同样是一个挂在 $\pi_\theta$ 上的 LoRA 适配器，接收 hook $H_{t,<j}$ 后输出潜在记忆矩阵 $M_t=\mathcal{W}^{\theta'}_{weaver}(H_{t,<j})$，经线性层投影对齐到 token 嵌入空间再前置注入。**最关键的解耦**：智能体吸收新经验时，知识只写进 $\mathcal{W}_{weaver}$ 的参数，$\pi_\theta$ 完全不动——这使 MemGen 与优化算法无关、与各种主干兼容，无论 SFT 还是 GRPO/DAPO，都在统一目标下更新：$\max_{\theta_{lora}}\mathbb{E}\,R(x_i,\tau)$，梯度只回传到 $\theta'$。训练上采用「先用随机插入器作触发器的轻量替身训练编织器，再冻结编织器训练触发器」的两阶段流程。此外编织器可与任意检索系统（MemoryBank、ExpeL 等）组合：检索到的文本记忆与 hook 合并喂入编织器，让它同时整合内部参数知识与外部信息。

## 实验关键数据

### 主实验表格（SmolLM3-3B / Qwen3-8B，准确率 %，节选）

| 方法 | ALFWorld | TriviaQA | PopQA | KodCode | GPQA | GSM8K |
|---|---|---|---|---|---|---|
| Vanilla (3B) | 18.96 | 10.47 | 8.23 | 37.05 | 9.35 | 47.63 |
| GRPO (3B) | 55.35 | 65.88 | 45.16 | 68.48 | 22.73 | 80.03 |
| ExpeL (3B) | 36.18 | 46.20 | 28.16 | 51.14 | 15.15 | 56.23 |
| **MemGen GRPO (3B)** | **63.60** | **79.30** | **58.60** | **72.85** | 25.20 | **83.47** |
| GRPO (8B) | 85.60 | 76.15 | 58.90 | 73.35 | 39.54 | 92.30 |
| AWM (8B) | 80.33 | 69.30 | 43.69 | - | - | - |
| **MemGen GRPO (8B)** | **90.60** | **80.65** | **62.30** | **76.16** | **40.24** | **93.20** |

覆盖 9 数据集 / 5 领域、对比 12 个基线（提示式 / 参数式 / 检索式 / 潜在计算四类）。MemGen 在所有领域全面领先：Qwen3-8B 上 KodCode +27.06%、PopQA +28.17%（相对 vanilla），并超过强 GRPO 最多 3.4%。

### 消融实验（潜在记忆长度 K 敏感性）
潜在记忆序列长度 $K\in\{2,4,8\}$，过短信息量不足、过长引入噪声，存在性能最优区间（详见原文 Figure 6 左）。集成检索式记忆的变体结果置于附录 F。

### 关键发现
- **跨域泛化（RQ2）**：在 KodCode 上训练，MATH 从 36.6%→54.2%；触发器会按任务自适应调节调用频率——GSM8K 提升最大（+19.64%）调用最多，KodCode 提升最小调用最少，说明它能判断「在陌生域少插记忆」以避免域冲突。
- **持续学习（RQ3）**：顺序训练四数据集，MemGen 在 KodCode 训练后 AQuA 仍保 40.34%（ExpeL 27.14%、SFT 28.61%），有效缓解灾难性遗忘。
- **涌现人类式记忆层级（RQ4）**：无显式监督下，潜在记忆经 t-SNE 自发按域聚类（机器原生、不可读但有结构规律）；后验干预（删除特定 cluster）揭示出功能分化的**规划记忆、程序记忆、工作记忆**三类——例如删掉 Cluster 2 会激增规划/组合推理失败。

## 亮点与洞察
- **「生成式重构」而非「检索复述」**：把记忆当作可被重新合成的认知产物，跳出了检索式记忆「按相似度取回」的范式局限。
- **冻结主干 + 外挂双 LoRA**：知识只进编织器，从架构上彻底隔离了「学新经验」与「保通用能力」，天然免疫灾难性遗忘，且与优化算法 / 主干解耦。
- **token 级元认知触发**：句子粒度激活 + 奖励自适应惩罚，让「何时回忆」本身成为可学习、可解释的稀疏决策。
- **可解释性意外收获**：功能分化的记忆层级是自发涌现而非设计强加，为「机器认知向自然化形态演进」提供了实证线索。

## 局限与展望
- 潜在记忆机器原生、人类不可读，强制解码只见结构规律不见语义，可解释性仍有限，调试与可信度存疑。
- 触发器 + 编织器双组件 + 两阶段训练，工程复杂度高于纯检索方案；句子粒度激活虽控了开销，但插入仍带来一定推理延迟（原文附录 E.3.3 验证可接受）。
- 实验主干集中在 1.5B–8B，更大模型上潜在记忆的容量与收益尚待验证。
- 「人类式记忆层级」基于后验干预的归因解读，因果性证据仍偏弱。

## 相关工作与启发
- **参数式记忆**（FireAct、AgentLumos、Agent-FLAN）：经验进参数，强但遗忘。
- **检索式记忆**（Mem0、AWM、ExpeL、MemoryBank、G-Memory、AgentKB）：经验外化进数据库，不遗忘但僵硬。
- **潜在 / 潜在计算**（Coconut、CODI、CoLaR、SoftCoT、LatentSeek、Co-processor）：用潜在状态做原生推理或调制生成，MemGen 归属此脉络但强调推理-记忆交织与生成式重构。
- **启发**：把「记忆」从静态数据/参数升级为「按需生成的动态认知模块」，并用元认知信号控制其调用时机，为自进化智能体提供了一条「不改主干也能持续学习」的新路径。

## 评分
- **新颖性**: ⭐⭐⭐⭐⭐ 「生成式潜在记忆 + token 级元认知触发 + 冻结主干外挂双 LoRA」是对智能体记忆范式的实质性重构，并意外揭示涌现的功能分化记忆层级。
- **实验充分度**: ⭐⭐⭐⭐ 9 数据集 5 领域、12 基线、3 种主干，覆盖主实验 / 泛化 / 持续学习 / 机制分析四问，较全面；更大模型与延迟代价的权衡可再补。
- **写作质量**: ⭐⭐⭐⭐⭐ 「记忆-推理交织」的认知隐喻贯穿全文，图 1/2 对比清晰，RQ 驱动的结构读来顺畅。
- **价值**: ⭐⭐⭐⭐⭐ 同时解决遗忘与僵硬两大痛点，方法与优化算法/主干解耦、可即插即用，对自进化智能体与记忆系统社区有较强借鉴意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] SEARL: Joint Optimization of Policy and Tool Graph Memory for Self-Evolving Agents](../../ACL2026/llm_agent/searl_joint_optimization_of_policy_and_tool_graph_memory_for_self-evolving_agent.md)
- [\[ICLR 2026\] AlphaAgentEvo: Evolution-Oriented Alpha Mining via Self-Evolving Agentic Reinforcement Learning](alphaagentevo_evolution-oriented_alpha_mining_via_self-evolving_agentic_reinforc.md)
- [\[ICML 2026\] Self-evolving LLM agents with in-distribution Optimization](../../ICML2026/llm_agent/self-evolving_llm_agents_with_in-distribution_optimization.md)
- [\[ICLR 2026\] Your Agent May Misevolve: Emergent Risks in Self-evolving LLM Agents](your_agent_may_misevolve_emergent_risks_in_self-evolving_llm_agents.md)
- [\[ICLR 2026\] Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](agentic_context_engineering_evolving_contexts_for_self-improving_language_models.md)

</div>

<!-- RELATED:END -->
