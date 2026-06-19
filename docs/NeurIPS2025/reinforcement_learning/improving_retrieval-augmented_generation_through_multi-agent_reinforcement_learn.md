---
title: >-
  [论文解读] Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning
description: >-
  [NeurIPS 2025][强化学习][RAG] 将复杂 RAG 流水线中的多个组件（Query Rewriter、Selector、Generator）建模为协作多智能体系统，使用 MAPPO 算法进行联合优化，以最终答案的 F1 分数作为共享奖励，在多个 QA 基准上超越现有单模块优化方法。 领域现状： RAG 系统通…
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "RAG"
  - "多智能体强化学习"
  - "MAPPO"
  - "联合优化"
  - "问答系统"
---

# Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning

**会议**: NeurIPS 2025  
**arXiv**: [2501.15228](https://arxiv.org/abs/2501.15228)  
**代码**: [GitHub](https://github.com/chenyiqun/MMOA-RAG)  
**领域**: 强化学习 / NLP  
**关键词**: RAG, 多智能体强化学习, MAPPO, 联合优化, 问答系统

## 一句话总结

将复杂 RAG 流水线中的多个组件（Query Rewriter、Selector、Generator）建模为协作多智能体系统，使用 MAPPO 算法进行联合优化，以最终答案的 F1 分数作为共享奖励，在多个 QA 基准上超越现有单模块优化方法。

## 研究背景与动机

**领域现状**: RAG 系统通过检索外部知识增强 LLM 的生成能力，现代 RAG 系统由多个组件组成——查询重写、文档检索、文档筛选和答案生成。

**现有痛点**: 各组件通常通过 SFT 独立优化，导致模块目标与最终生成准确答案的全局目标不一致。例如，检索器按照 nDCG 优化的"高相关性"文档不一定有助于生成正确答案。

**核心矛盾**: 现有端到端优化方法（如用 PPO 或 DPO 优化单个 RAG 组件）仅关注两组件的简单 pipeline 或孤立优化单个模块，无法充分建模多组件间的协作关系。

**本文目标**: 如何联合优化 RAG 系统中多个组件的参数，使各模块的优化目标与最终答案质量对齐。

**切入角度**: 将 RAG 建模为协作多智能体强化学习（Co-MARL）问题，用 MAPPO 实现多智能体联合优化。

**核心 idea**: 把 RAG 看成合作博弈——各组件是 agent，共享"最终答案F1"作为全局奖励，用 MAPPO 同步优化所有 agent。

## 方法详解

### 整体框架

MMOA-RAG 将 RAG 建模为 $\langle \mathcal{G}, \mathcal{O}, \mathcal{A}, \mathcal{R} \rangle$ 的多智能体系统。四模块流水线：Query Rewriter → Retriever（固定不训练）→ Selector → Generator。三个可训练 agent 共享同一个 LLM，通过参数共享降低训练开销。

### 关键设计

1. **多智能体建模（Co-MARL）**:

    - **功能**: 将 RAG 的各组件视为 RL agent，共享全局奖励
    - **为什么**: 单独优化各模块会导致目标不一致；多智能体框架自然建模了组件间的协作关系
    - **怎么做**: 定义三个 agent：Query Rewriter（QR）接收问题 $q$，输出子问题 $subq$；Selector（S）接收 $q$ 和候选文档 $D$，输出选中的文档 ID 子集 $D_{\text{selected}}$；Generator（G）接收 $q$ 和 $D_{\text{selected}}$，生成最终答案
    - **区别**: 相比 Rewrite-Retrieve-Read（仅优化 rewriter）或 BGM（仅优化 bridge 模块），MMOA-RAG 同时优化三个组件

2. **智能体的观测/动作/奖励设计**:

    - **功能**: 为每个 agent 定义精确的 MDP 元素
    - **为什么**: 不同 agent 的角色差异要求差异化的动作空间和惩罚项设计
    - **怎么做**: 
        - QR 的动作空间为完整词表 $\mathcal{V}$，奖励 $R_{QR} = R_{\text{shared}} + P_{QR}$，当子问题数 > 4 时惩罚 -0.5
        - Selector 的动作空间限制为 $\{$"0", "1", ..., "K-1", "Document", ","$\}$，大幅缩小探索空间；格式错误或重复 ID 时惩罚 -1
        - Generator 的动作空间为 $\mathcal{V}$，生成答案过长时惩罚 -0.5
        - 共享奖励 $R_{\text{shared}}$ 为预测答案的 F1 分数
    - **区别**: Selector 的受限动作空间设计巧妙——将自由文本生成约束为结构化 ID 选择，显著提升训练稳定性

3. **MAPPO 联合优化**:

    - **功能**: 使用 Multi-Agent PPO 算法联合更新所有 agent
    - **为什么**: MAPPO 在全合作环境中使用共享全局奖励促进agent间协作，比独立 PPO 更适合此场景
    - **怎么做**: Actor 损失采用标准 PPO 的 clip objective，扩展到多 agent：
    $\mathcal{L}_{\text{Actor}}(\theta) = \sum_i \sum_t \min(r_t^i \hat{A}_{\pi_\theta}^{i,t},\ \text{clip}(r_t^i, 1-\epsilon, 1+\epsilon) \hat{A}_{\pi_\theta}^{i,t})$
   最终奖励包含 KL 惩罚以防止偏离 SFT 基线：
    $R(s_t^i, a_t^i) = R_i - \beta \log \frac{\pi_\theta(Answer_i | O_i)}{\pi_{\theta_{\text{SFT}}}(Answer_i | O_i)}$
    - **区别**: 三个 agent 共享同一 LLM 参数（参数共享机制），训练效率高

### 训练策略

- **Warm Start (SFT)**: 先对每个 agent 的任务进行 SFT 预热，使模型具备基本指令跟随能力
- **MAPPO 联合训练**: 在 SFT 基础上执行 Collect Rollout（依次经过 QR→Retriever→S→G），计算共享奖励和各 agent 惩罚，用 GAE 估计优势函数后更新 Actor 和 Critic
- mini-batch 并行加速训练

## 实验关键数据

### 主实验（Contriever 检索器 + Llama-3-8B-Instruct）

| 方法 | HotpotQA F1 | 2Wiki F1 | AmbigQA F1 |
|------|:---:|:---:|:---:|
| LLM w/o RAG | 31.18 | 29.47 | 33.42 |
| Vanilla RAG w/o train | 30.67 | 22.84 | 33.56 |
| Vanilla RAG w SFT | 44.49 | 43.36 | 44.36 |
| SELF-RAG | 38.93 | 38.86 | 39.04 |
| RetRobust | 46.49 | 44.51 | 44.78 |
| Rewrite-Retrieve-Read | 46.32 | 44.17 | 45.92 |
| BGM | 44.54 | 43.29 | 45.76 |
| RAG-DDR | 44.26 | 44.18 | 45.83 |
| **MMOA-RAG** | **48.29** | **46.40** | **48.59** |
| Δ vs best baseline | +1.80 | +1.89 | +2.67 |

### 消融实验（移除不同 agent 的联合优化）

| 配置 | HotpotQA F1 | 2Wiki F1 | AmbigQA F1 |
|------|:---:|:---:|:---:|
| MMOA-RAG (QR+S+G) | **48.29** | **46.40** | **48.59** |
| MMOA-RAG w/o QR | 47.07 | 45.25 | 47.19 |
| MMOA-RAG w/o S | 47.94 | 46.19 | 47.53 |
| MMOA-RAG w/o G | 最差（图中） | 最差 | 最差 |

### 泛化性实验（不同 RAG 配置的 SFT→MAPPO 提升）

| 配置 | HotpotQA F1 (SFT→MAPPO) | 2Wiki F1 (SFT→MAPPO) | AmbigQA F1 (SFT→MAPPO) |
|------|:---:|:---:|:---:|
| QR+S+G | 44.69→48.29 (+3.60) | 42.97→46.40 (+3.43) | 46.71→48.59 (+1.88) |
| S+G | 43.14→47.07 (+3.93) | 42.40→45.25 (+2.85) | 45.82→47.19 (+1.37) |
| QR+G | 45.00→47.94 (+2.94) | 42.91→46.19 (+3.28) | 45.31→47.53 (+2.22) |

### 关键发现

- **联合优化 > 单独优化**: 完整的三 agent 联合优化（QR+S+G）在所有数据集上最优
- **Generator 是最关键 agent**: 移除 G 的联合优化导致最大性能下降，尤其在单跳 AmbigQA 上
- **Selector 可被 Generator 部分替代**: w/o S 的下降最小，因为联合训练后 Generator 学会了一定的降噪能力
- **多跳数据集收益更大**: MAPPO 在 HotpotQA/2Wiki（多跳）上的提升（~3.5 F1）大于 AmbigQA（单跳，~1.9 F1），表明多模块协作在复杂推理中更重要
- **MAPPO 一致有效**: 在 QR+S+G、S+G、QR+G 三种配置下，MAPPO 均比 SFT 带来显著提升，证明框架的通用性

## 亮点与洞察

- **建模视角新颖**: 首次将 RAG 系统建模为协作多智能体任务，为复杂 AI pipeline 的端到端优化提供了新范式
- **Selector 动作空间设计**: 将文档筛选从自由文本生成约束为结构化 ID 输出，是工程上的巧妙决策——大幅降低了探索空间和训练不稳定性
- **参数共享**: 三个 agent 共享同一 LLM（仅通过不同 prompt 区分角色），使训练计算开销接近单 agent PPO
- **惩罚项设计**: 各 agent 的轻量惩罚（子问题数量、格式合规、答案长度）在不影响主要优化目标的同时约束了输出质量

## 局限与展望

- Retriever 被固定不参与优化，理论上检索模块的联合优化可能带来更大收益
- 仅在 Llama-3-8B 上验证，未测试更大规模模型或闭源 LLM
- 共享奖励仅基于 F1 分数，未考虑效率（延迟/成本）等多目标优化
- MAPPO 的训练开销未与其他方法定量对比
- 未探索 DAG 形式的更复杂 RAG 工作流或循环调用场景

## 相关工作与启发

- **MAPPO (Yu et al., 2022)**: 在 StarCraft II 中验证的多智能体 PPO 算法，本文将其迁移到 NLP pipeline
- **Rewrite-Retrieve-Read / BGM**: 用 PPO 优化单个 RAG 模块的先驱工作，本文将其扩展到多模块联合
- **Search-R1 / R1-Searcher**: 同期用 RL 优化 RAG 推理的工作，但聚焦单 agent
- **InstructGPT**: KL 惩罚的灵感来源
- **启发**: Co-MARL 建模为优化任何多模块 AI 系统提供了通用做法——可推广到 multi-agent coding、tool use pipeline 等场景

## 评分

- 新颖性: ⭐⭐⭐⭐ RAG 建模为 Co-MARL 视角新颖，Selector 动作空间设计巧妙
- 实验充分度: ⭐⭐⭐⭐ 三数据集 + 多 baseline + agent 消融 + 配置泛化实验 + 不同检索器验证
- 写作质量: ⭐⭐⭐⭐ 公式推导完整，框架图清晰，尤其 agent 元素定义严谨
- 价值: ⭐⭐⭐⭐ 为复杂 RAG 系统优化提供了可复制的通用框架，开放源码

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] TAdaRAG: Task Adaptive Retrieval-Augmented Generation via On-the-Fly Knowledge Graph Construction](../../AAAI2026/reinforcement_learning/tadarag_task_adaptive_retrieval-augmented_generation_via_on-the-fly_knowledge_gr.md)
- [\[NeurIPS 2025\] Robust and Diverse Multi-Agent Learning via Rational Policy Gradient](robust_and_diverse_multi-agent_learning_via_rational_policy_gradient.md)
- [\[NeurIPS 2025\] Sequential Multi-Agent Dynamic Algorithm Configuration](sequential_multi-agent_dynamic_algorithm_configuration.md)
- [\[NeurIPS 2025\] Knowledge-based Visual Question Answer with Multimodal Processing, Retrieval and Filtering](knowledge-based_visual_question_answer_with_multimodal_processing_retrieval_and_.md)
- [\[NeurIPS 2025\] Extending NGU to Multi-Agent RL: A Preliminary Study](extending_ngu_to_multi-agent_rl_a_preliminary_study.md)

</div>

<!-- RELATED:END -->
