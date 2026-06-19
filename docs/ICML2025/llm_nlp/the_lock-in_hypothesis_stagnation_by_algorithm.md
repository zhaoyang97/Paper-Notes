---
title: >-
  [论文解读] The Lock-in Hypothesis: Stagnation by Algorithm
description: >-
  [ICML 2025][LLM 其他][反馈循环] 本文提出并形式化了"锁定假说"（Lock-in Hypothesis）：LLM 训练与部署过程中形成的人类-AI 反馈循环会固化用户的现有信念，导致群体观点多样性不可逆地丧失，甚至锁定在错误信念上。 大语言模型（LLM）正日益影响人类的信念和价值观。这产生了一个自我强化的反…
tags:
  - "ICML 2025"
  - "LLM 其他"
  - "反馈循环"
  - "信念锁定"
  - "多样性损失"
  - "人机交互"
  - "贝叶斯建模"
---

# The Lock-in Hypothesis: Stagnation by Algorithm

**会议**: ICML 2025  
**arXiv**: [2506.06166](https://arxiv.org/abs/2506.06166)  
**代码**: 无（网站: thelockinhypothesis.com）  
**领域**: LLM/NLP  
**关键词**: 反馈循环, 信念锁定, 多样性损失, 人机交互, 贝叶斯建模

## 一句话总结

本文提出并形式化了"锁定假说"（Lock-in Hypothesis）：LLM 训练与部署过程中形成的人类-AI 反馈循环会固化用户的现有信念，导致群体观点多样性不可逆地丧失，甚至锁定在错误信念上。

## 研究背景与动机

大语言模型（LLM）正日益影响人类的信念和价值观。这产生了一个自我强化的反馈循环：

1. AI 从人类数据中学习价值观（预训练 + 后训练阶段）
2. AI 在交互中影响人类观点
3. 被影响的观点又被 AI 重新吸收
4. 如此循环往复

这一动态类似于推荐系统中的"回音室效应"（Echo Chamber），但存在本质区别：推荐系统是**个性化**优化（针对每个用户），而 LLM 主要是**集体优化**（通过 RLHF 等对全体用户统一对齐）。因此 LLM 可能在群体层面造成锁定效应。

现有研究的三个主要缺口：
- 大多关注 AI 对人类的**单向**影响，忽视了双向反馈循环
- 缺乏反馈循环的**机制性解释**
- 大多数证据来自**实验室**环境，非真实使用场景

## 方法详解

### 整体框架

本文从三个层面渐进式地论证锁定假说：

1. **形式化贝叶斯模型**（§3）：构建多智能体信念更新的数学框架，推导锁定发生的充要条件
2. **基于 Agent 的 LLM 仿真**（§4）：用 GPT-4.1-Nano 模拟 100 个 agent 的信念演化过程
3. **真实数据因果推断**（§5）：在 WildChat-1M 数据集上检验概念多样性随 GPT 版本迭代的变化

### 关键设计

#### 1. 贝叶斯信念更新模型

考虑 $N$ 个 agent 估计未知量 $\mu \in \mathbb{R}$。每个 agent $i$ 在时间步 $t$ 获得带噪观测 $o_{i,t} \sim \mathcal{N}(\mu, \sigma_i^2)$，并维护：

- **私有信念**：基于自身观测的后验 $\mathcal{N}(\hat{\mu}_{i,t}, p_{i,t}^{-1})$
- **聚合信念**：综合自身与他人信息的后验 $\mathcal{N}(\hat{\nu}_{i,t}, q_{i,t}^{-1})$

关键在于：agent $i$ 只能看到 agent $j$ 的**聚合信念**，而非私有信念。这导致信息被重复计算（double-counting），形成反馈循环。

#### 2. 信任矩阵（Trust Matrix）

定义信任矩阵 $\mathbf{W} \in \mathbb{R}_{\geq 0}^{N \times N}$，其中 $w_{i,j}$ 表示 agent $i$ 对 agent $j$ 信念的信任程度：

- $w_{i,j} = 0$：完全忽略
- $0 < w_{i,j} < 1$：部分信任（折扣精度）
- $w_{i,j} = 1$：完全信任
- $w_{i,j} > 1$：超额信任

对于人类-LLM 交互场景，构造特殊信任矩阵：1 个 LLM + $(N-1)$ 个人类用户，LLM 以 $\lambda_1$ 信任每个人类（偏好学习强度），人类以 $\lambda_2$ 信任 LLM。人类之间无直接通信。

#### 3. 锁定的相变条件（核心定理）

**定理 3.2（反馈循环导致集体锁定）**：当信任矩阵 $\mathbf{W}$ 的谱半径 $\rho(\mathbf{W}) > 1$ 时，至少存在一个 agent 的信念几乎必然**不会**收敛到真值：

$$\Pr\left[\lim_{t \to \infty} \hat{\mu}_{i,t} = \mu\right] = 0$$

反之，当 $\rho(\mathbf{W}) < 1$ 时，所有 agent 几乎必然收敛到真值。

**推论 3.3（人类-LLM 锁定条件）**：锁定发生的临界条件为：

$$(N-1)\lambda_1 \lambda_2 > 1$$

这是一个**非常弱**的条件。例如 $N = 101$ 时，只需 $\lambda_1, \lambda_2 > 0.1$ 即可触发锁定——即人类和 AI 对彼此报告信念的折扣因子小于 10 即可。

#### 4. 概念多样性度量（Lineage Diversity）

为衡量概念多样性的变化，作者提出**谱系多样性**指标：

从 WildChat 语料中提取约 544 万条自然语言概念，通过层次聚类构建概念层次树 $\mathcal{T}$。对概念集合 $\mathcal{C}$，定义：

$$D_{\text{lineage}}(\mathcal{C}; \mathcal{T}) = \frac{\log|\mathcal{T}| - \log \mathbb{E}_{u,v \sim \text{Unif}(\mathcal{C})} [|\mathcal{T}| / |\mathcal{T}_{l(u,v)}|]}{\log |\mathcal{T}|}$$

其中 $l(u,v)$ 是概念 $u, v$ 的最低公共祖先。该指标归一化到 $[0, 1]$，1 表示完全多样，0 表示完全同质。

### 损失函数 / 训练策略

本文不涉及传统意义上的模型训练，但核心的"训练动态"体现在多 agent 贝叶斯更新的递推方程：

- **精度更新**：$\mathbf{q}_{t+1} = \mathbf{p}_{t+1} + \mathbf{W} \cdot \mathbf{q}_t$
- **信念更新**：$\hat{\boldsymbol{\nu}}_{t+1} \odot \mathbf{q}_{t+1} = \hat{\boldsymbol{\mu}}_{t+1} \odot \mathbf{p}_{t+1} + \mathbf{W}(\hat{\boldsymbol{\nu}}_t \odot \mathbf{q}_t)$

当 $\rho(\mathbf{W}) > 1$ 时，精度 $\mathbf{q}_t$ 以指数速度增长，但其增长由反馈循环中的信息重复计算驱动，而非真实新证据。这导致群体对**错误信念**变得极端自信。

## 实验关键数据

### 主实验

#### Agent 仿真实验（§4）

100 个 GPT-4.1-Nano 模拟的 agent，在 4 个 r/ChangeMyView 话题上交互 200 轮：

| 话题 | 初始信念分布 | 最终信念分布 | 多样性变化 |
|------|------------|------------|----------|
| Trump & Discourse | 双峰分布 | 集中于极端值 ~0.1 | 熵大幅下降 |
| Population Decline | 双峰分布 | 集中于极端值 ~0.9 | 熵大幅下降 |
| Citizens United | 双峰分布 | 集中于极端值 ~0.9 | 熵大幅下降 |
| RBG Legacy | 双峰分布（均值~0.5） | 集中于 ~0.9 | 熵大幅下降 |

#### WildChat 真实数据分析（§5）

| 多样性指标 | GPT-4 趋势 | GPT-3.5t 趋势 | GPT-4-0125 切换 | GPT-3.5t-0613 切换 | GPT-3.5t-0125 切换 |
|-----------|-----------|-------------|----------------|-------------------|-------------------|
| Lineage (value-laden) | ↓ (p<.05) | ↑ (p<.05) | ↓ (p<.05) | ↓ (p<.05) | ↓ (p<.05) |
| Lineage (all) | ↓ (p<.05) | ↓ (p<.05) | ↓ (p<.05) | ↓ (p<.05) | ↓ (p<.05) |
| Depth (value-laden) | ↓ (p<.05) | ↓ (p<.05) | ↓ (p<.05) | ↓ (p<.05) | ↓ (p=.41) |
| Topic Entropy | ↓ (p=.07) | ↓ (p=.09) | ↓ (p<.05) | ↓ (p<.05) | ↑ (p=.06) |
| Jaccard Distance | ↓ (p<.05) | ↑ (p<.05) | ↓ (p<.05) | ↓ (p<.05) | ↓ (p=.63) |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| $(N-1)\lambda_1\lambda_2 = 0.9$ | 收敛到真值 | 谱半径 < 1，反馈循环自衰减，不发生锁定 |
| $(N-1)\lambda_1\lambda_2 = 1.0$ | 趋向真值但过度自信 | 临界点，开始出现精度爆炸 |
| $(N-1)\lambda_1\lambda_2 = 1.1$ | 收敛到错误值 | 谱半径 > 1，每次仿真均锁定在不同的错误信念 |
| 移除模板消息 | Lineage ↓ (p<.05) | 排除 API 用途后仍观测到多样性下降 |
| 仅价值相关概念 | Lineage ↓ (p<.05) | 道德/政治/宗教领域的锁定效应更显著 |
| Per-user 回归 | ↓ (p=.07) | 控制用户身份等混杂变量后仍呈负向，中等支持 |

### 关键发现

1. **相变现象**：存在一个清晰的临界阈值 $(N-1)\lambda_1\lambda_2 = 1$，超过该阈值后群体必然锁定到错误信念
2. **信念趋同模式**：仿真中观察到三种趋同模式——观点翻转（flip）、模糊化（hedging）、极端收敛
3. **版本更新加速多样性损失**：GPT 新版本上线后，WildChat 中人类消息的概念多样性出现**不连续的突然下降**
4. **GPT-4 与 GPT-3.5 行为分化**：GPT-4 用户的价值类概念多样性持续下降，GPT-3.5 用户的趋势则较模糊

## 亮点与洞察

- **理论-仿真-实证三层论证**：从贝叶斯数学模型到 LLM agent 仿真再到真实数据因果推断，层层递进，论证严密
- **相变条件极弱**：$(N-1)\lambda_1\lambda_2 > 1$ 的条件很容易在实际中满足，意味着锁定可能已经在发生
- **新度量指标**：提出的 Lineage Diversity 利用概念层次结构，比传统 Shannon 熵更能捕捉语义层面的多样性变化
- **回归拐点设计（RKD）**：巧妙利用 GPT 版本切换这一外生事件作为自然实验，部分解决因果推断难题
- **连接推荐系统与 LLM alignment 文献**：指出推荐系统的回音室研究是个性化层面的，而 LLM 的锁定效应是集体层面的，这是一个重要的概念区分

## 局限与展望

1. **无随机对照实验（RCT）**：WildChat 分析未能排除所有时间序列混杂因素，需要与 AI 实验室合作做真正的 RCT
2. **仿真过于简化**：100 个 agent 由同一个 LLM 模拟，与真实人类信念更新机制有差距；未考虑新经验证据的引入
3. **GPT-3.5 结果模糊**：假说 1 在 GPT-3.5 上未获得一致支持，可能与高低端模型分工有关但缺乏深入分析
4. **概念提取依赖 prompting**：概念提取和价值类标注均依赖 GPT-4o-mini 的 prompting pipeline，可能引入系统偏差
5. **仅关注多样性损失**：多样性损失是锁定的必要非充分条件，完全锁定的实证证据仍缺乏
6. **缺乏缓解策略**：仅指出问题但未提出具体的算法或政策干预方案

## 相关工作与启发

- **推荐系统回音室**：RecSys 的回音室效应研究（Cinelli et al., 2021）是最接近的类比，但 LLM 的集体对齐机制使得锁定效应在群体层面更突出
- **Model Collapse**（Shumailov et al., 2024）：模型在自身生成数据上训练导致性能退化，与锁定假说共享"反馈循环"的核心结构
- **迭代学习理论**（Griffiths & Kalish, 2007）：个体从同样学习方式的他人那里学习，本文在此基础上加入了拓扑结构和相互信任的考量
- **RLHF 的盲区**：当前对齐方法将人类反馈视为不可影响的 oracle（Bai et al., 2022），忽视了 LLM 对人类偏好的反向影响
- **对未来研究的启发**：如何在偏好学习中打破反馈循环？是否可以通过控制 $\lambda_1$（降低偏好学习强度）或 $\lambda_2$（提醒用户独立思考）来使 $(N-1)\lambda_1\lambda_2 < 1$？

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|----------|------|
| 新颖性 | 5 | 首次形式化并实证检验人类-LLM 反馈循环的锁定效应 |
| 理论深度 | 5 | 贝叶斯模型严谨，相变条件推导优雅 |
| 实验充分性 | 4 | 三层论证但 WildChat 因果推断仍有局限 |
| 写作质量 | 4 | 结构清晰，但 LaTeX 公式密集，部分读者可能需要较强数学背景 |
| 影响力 | 5 | 对 AI 安全、对齐、治理均有深远意义 |
| **总分** | **4.6** | 非常优秀的跨学科工作，理论与实证兼备 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Binary Hypothesis Testing for Softmax Models and Leverage Score Models](binary_hypothesis_testing_for_softmax_models_and_leverage_score_models.md)
- [\[ACL 2025\] Literature Meets Data: A Synergistic Approach to Hypothesis Generation](../../ACL2025/llm_nlp/literature_meets_data_hypothesis.md)
- [\[ACL 2025\] A Survey of Automatic Prompt Optimization with Instruction-focused Heuristic-based Search Algorithm](../../ACL2025/llm_nlp/a_survey_of_automatic_prompt_optimization_with_instruction-focused_heuristic-bas.md)
- [\[ACL 2025\] HyGenar: An LLM-Driven Hybrid Genetic Algorithm for Few-Shot Grammar Generation](../../ACL2025/llm_nlp/hygenar_an_llm-driven_hybrid_genetic_algorithm_for_few-shot_grammar_generation.md)
- [\[ICLR 2026\] Rethinking Code Similarity for Automated Algorithm Design with LLMs](../../ICLR2026/llm_nlp/rethinking_code_similarity_for_automated_algorithm_design_with_llms.md)

</div>

<!-- RELATED:END -->
