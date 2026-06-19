---
title: >-
  [论文解读] Federated In-Context Learning: Iterative Refinement for Improved Answer Quality
description: >-
  [ICML 2025][LLM安全][联邦学习] 本文提出 Fed-ICL，一种联邦 In-Context Learning 框架，通过客户端与服务端之间的多轮迭代协作，在不传输模型参数的情况下利用分散在各客户端的高质量示例逐步改善回答质量，并建立了收敛保证。 领域现状： In-Context Learning（ICL）使语…
tags:
  - "ICML 2025"
  - "LLM安全"
  - "联邦学习"
  - "上下文学习"
  - "问答"
  - "迭代优化"
  - "通信效率"
---

# Federated In-Context Learning: Iterative Refinement for Improved Answer Quality

**会议**: ICML 2025  
**arXiv**: [2506.07440](https://arxiv.org/abs/2506.07440)  
**代码**: 无  
**领域**: AI Safety  
**关键词**: 联邦学习, 上下文学习, 问答, 迭代优化, 通信效率

## 一句话总结
本文提出 Fed-ICL，一种联邦 In-Context Learning 框架，通过客户端与服务端之间的多轮迭代协作，在不传输模型参数的情况下利用分散在各客户端的高质量示例逐步改善回答质量，并建立了收敛保证。

## 研究背景与动机

**领域现状**: In-Context Learning（ICL）使语言模型能够通过输入中的示例来完成任务，无需修改参数。ICL 的效果高度依赖示例的质量和多样性。在实际场景中，高质量示例可能分布在不同的客户端设备上。

**现有痛点**: 现有方法要么需要传输模型参数（如 FedAvg），通信开销巨大且不适合超大语言模型；要么仅使用本地数据做 ICL，无法利用跨客户端的示例多样性。将分散的示例集中到服务端又违反数据隐私原则。

**核心矛盾**: 如何在不传输模型参数和原始数据的前提下，让语言模型享受到分布式高质量示例带来的 ICL 提升？

**本文目标**: 设计一种通信高效的联邦 ICL 方法，利用分散在各客户端的示例数据改善中心服务端的回答质量。

**切入角度**: 传输的不是模型参数也不是原始数据，而是 ICL 生成的"回答"——通过多轮迭代，客户端用本地示例优化服务端的回答。

**核心 idea**: 服务端发送当前最佳回答给客户端，客户端用本地示例作为 ICL context 对回答进行改进，改进后的回答再返回服务端聚合，迭代执行。

## 方法详解

### 整体框架
输入：问题 $q$，$K$ 个客户端各持有本地示例集 $\{(q_i^k, a_i^k)\}$
输出：服务端的最终回答 $a^*$

每轮迭代：
1. **服务端广播**: 发送当前回答 $a^t$ 给所有客户端
2. **客户端改进**: 每个客户端 $k$ 利用本地示例 + 当前回答构造 ICL prompt，生成改进回答 $a_k^{t+1}$
3. **服务端聚合**: 收集所有客户端的改进回答，通过聚合策略得到 $a^{t+1}$

### 关键设计

1. **迭代改进机制（Iterative Refinement Mechanism）**:

    - 功能：多轮交互逐步提升回答质量
    - 核心思路：prompt 构造为：$[\text{Question}: q, \text{Current Answer}: a^t, \text{Examples}: \{(q_i, a_i)\}, \text{Instruction}: \text{Improve the answer}]$
    - 设计动机：单轮 ICL 受限于单个客户端的示例质量，迭代机制允许多个客户端的知识逐步融合。每轮迭代相当于在"回答空间"中做一步优化

2. **回答聚合策略（Answer Aggregation Strategy）**:

    - 功能：在服务端将多个客户端的改进回答合并为一个更好的回答
    - 核心思路：支持多种聚合方式——(i) 投票聚合：多数投票选最佳回答，(ii) LLM 聚合：用 LLM 综合多个回答生成最终答案，(iii) 评分聚合：用 LLM 对每个回答评分后加权平均
    - 设计动机：不同聚合策略适用于不同任务——分类任务适合投票，开放式 QA 适合 LLM 聚合

3. **收敛性理论保证（Convergence Guarantee）**:

    - 功能：证明 Fed-ICL 的迭代改进过程在一定条件下收敛
    - 核心思路：将回答质量建模为一个潜在函数，证明每轮迭代的改进量满足递减条件：$V(a^{t+1}) - V(a^t) \geq -c \cdot \|a^t - a^*\|^2$
    - 设计动机：理论保证使得 Fed-ICL 不仅是经验有效的 heuristic，而是有理论支撑的算法

### 损失函数 / 训练策略
不涉及模型训练。核心度量为回答质量（如 BLEU、ROUGE、EM score）。通信内容仅为文本回答，通信开销极低。

## 实验关键数据

### 主实验

| 数据集 | 指标 | Fed-ICL | 单客户端 ICL | FedAvg | 集中式 ICL |
|--------|------|---------|-------------|--------|-----------|
| NQ (Natural Questions) | EM | **47.3** | 38.2 | 43.8 | 49.1 |
| TriviaQA | EM | **62.8** | 51.4 | 58.3 | 65.2 |
| SQuAD 2.0 | F1 | **78.5** | 68.9 | 74.2 | 80.1 |
| WebQuestions | EM | **43.7** | 35.1 | 40.6 | 45.9 |

### 消融实验

| 配置 | NQ EM | 通信量 (相对) | 说明 |
|------|-------|-------------|------|
| Fed-ICL (5 轮) | **47.3** | 1x | 最佳性能 |
| Fed-ICL (1 轮) | 41.5 | 0.2x | 单轮不足以融合多客户端知识 |
| Fed-ICL (10 轮) | 47.6 | 2x | 边际收益递减 |
| 投票聚合 | 44.8 | 1x | 简单但丢失信息 |
| LLM 聚合 | **47.3** | 1x | 最佳聚合策略 |
| FedAvg (LLaMA-7B) | 43.8 | 1000x+ | 传输模型参数，通信量远大 |

### 关键发现
- Fed-ICL 在 QA 任务上接近集中式 ICL 的性能（差距 2-4 EM），但通信量小 3 个数量级
- 迭代改进是关键——5 轮交互带来约 6-10 EM 的提升
- LLM 聚合优于简单投票聚合，能更好地综合多个改进方向
- Fed-ICL 对客户端数量的扩展性好，10-50 个客户端均表现稳定

## 亮点与洞察
- **范式创新**: "传输回答而非参数"的联邦 ICL 思路新颖且实用
- **极低通信开销**: 通信内容仅为短文本，比传输模型参数高效数千倍
- **隐私友好**: 不传输原始数据和模型参数，自然保护隐私
- **模型无关**: 适用于任何支持 ICL 的 LLM，无需修改模型

## 局限与展望
- 回答的迭代改进可能泄露间接信息（如回答的分布反映客户端数据特征）
- 对于非 QA 任务（如生成、翻译）的适用性有待验证
- 聚合策略的选择需要针对任务手工调整
- 理论保证依赖的假设（如 LLM 改进回答的概率模型）可能在实际中不完全满足

## 相关工作与启发
- FedAvg (McMahan et al., 2017): 经典联邦学习
- Self-Refine (Madaan et al., 2023): 单模型自改进
- 本文将 self-refine 的思想推广到联邦设定，是有意义的交叉创新

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "联邦 ICL"是全新的范式
- 实验充分度: ⭐⭐⭐⭐ 4 个 QA 数据集，多种聚合策略消融
- 写作质量: ⭐⭐⭐⭐ 动机清晰，实验设计合理
- 价值: ⭐⭐⭐⭐ 对隐私保护 LLM 应用有实际意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] ICLShield: Exploring and Mitigating In-Context Learning Backdoor Attacks](iclshield_exploring_and_mitigating_in-context_learning_backdoor_attacks.md)
- [\[ICML 2025\] POPri: Private Federated Learning using Preference-Optimized Synthetic Data](popri_private_federated_learning_using_preference-optimized_synthetic_data.md)
- [\[NeurIPS 2025\] FedSVD: Adaptive Orthogonalization for Private Federated Learning with LoRA](../../NeurIPS2025/llm_safety/fedsvd_adaptive_orthogonalization_for_private_federated_learning_with_lora.md)
- [\[ACL 2025\] Answer When Needed, Forget When Not: Language Models Pretend to Forget via In-Context Knowledge Unlearning](../../ACL2025/llm_safety/answer_when_needed_forget_when_not_language_models_pretend_to_forget_via_in-cont.md)
- [\[ICCV 2025\] Geminio: Language-Guided Gradient Inversion Attacks in Federated Learning](../../ICCV2025/llm_safety/geminio_language-guided_gradient_inversion_attacks_in_federated_learning.md)

</div>

<!-- RELATED:END -->
