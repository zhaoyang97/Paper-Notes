---
title: >-
  [论文解读] Strategyproof Reinforcement Learning from Human Feedback
description: >-
  [NeurIPS 2025][LLM对齐][RLHF] 首次从机制设计角度研究 RLHF 中多标注者策略性操纵问题，证明了策略防操纵（strategyproofness）与政策对齐之间存在根本性权衡，并提出 Pessimistic Median of MLEs 算法实现近似策略防操纵。 RLHF 已成为 LLM 对齐的核心方…
tags:
  - "NeurIPS 2025"
  - "LLM对齐"
  - "RLHF"
  - "策略防操纵"
  - "多元偏好"
  - "机制设计"
  - "社会福利最大化"
---

# Strategyproof Reinforcement Learning from Human Feedback

**会议**: NeurIPS 2025  
**arXiv**: [2503.09561](https://arxiv.org/abs/2503.09561)  
**代码**: 未公开  
**领域**: LLM对齐  
**关键词**: RLHF, 策略防操纵, 多元偏好, 机制设计, 社会福利最大化  

## 一句话总结

首次从机制设计角度研究 RLHF 中多标注者策略性操纵问题，证明了策略防操纵（strategyproofness）与政策对齐之间存在根本性权衡，并提出 Pessimistic Median of MLEs 算法实现近似策略防操纵。

## 研究背景与动机

RLHF 已成为 LLM 对齐的核心方法，但在多标注者场景下存在一个被忽视的关键问题：**标注者可能策略性地歪曲其偏好反馈**，以将学到的政策向自身偏好倾斜。

具体例子：在 LLM 微调中，标注者可能系统性地误报偏好来放大特定偏见，强化对其有利的叙事。

现有 RLHF 方法的问题：
- Zhong et al. (2024) 的 Pessimistic Social Welfare 方法不防操纵
- Chakraborty et al. (2024) 的 MaxMin-RLHF 也不防操纵
- 对抗鲁棒的 RLHF 工作（Mandal et al., 2024 等）关注的是对抗性损坏而非策略性操纵——二者在视角和技术上有本质区别
- 基于支付机制的方法（VCG 类机制）在实际应用中不现实

核心研究问题：**如何在不依赖支付机制的情况下，使 RLHF 对策略性标注者具有鲁棒性？**

## 方法详解

### 整体框架

本文在离线 RLHF 设置下，考虑 $k$ 个标注者各自拥有线性奖励函数 $r_{\theta_i^*}(s,a) = \langle \theta_i^*, \phi(s,a) \rangle$，使用 Bradley-Terry 偏好模型。目标是最大化社会福利 $\mathcal{W}(\rho, \pi) = \frac{1}{k}\sum_{i=1}^k J_i(\rho, \pi)$。

策略性操纵的形式化：标注者 $i$ 可能使用被操纵的奖励参数 $\tilde{\theta}_i$ 来采样偏好标签，而非使用真实参数 $\theta_i^*$。

### 关键理论结果一：现有方法的脆弱性

**命题 3.3**：Pessimistic Social Welfare 和 MaxMin-RLHF **均不防操纵**。

**命题 3.4**：即使只有**一个**标注者策略性操纵，Pessimistic Social Welfare 的社会福利可以被降至任意低：

$$\mathcal{W}(\hat{\pi}) \leq \varepsilon, \quad \text{而} \quad \mathcal{W}(\pi^*) \geq BL - 2\varepsilon$$

### 关键理论结果二：根本性不可能定理

**定理 3.5**：任何策略防操纵的 RLHF 算法，其最坏情况次优性至少为：

$$\text{SubOpt}(\hat{\pi}) \geq \frac{k-1}{k}$$

**推论 3.6**：任何策略防操纵 RLHF 方法的近似比至多为 $\alpha(\rho, \hat{\pi}) \leq \frac{1}{k}$。

证明利用了 **Gibbard-Satterthwaite 定理**：将 RLHF 映射为投票问题，策略防操纵的规则必须是独裁规则或限制在两个选项之间——两种情况都导致低社会福利。

### 关键设计：Pessimistic Median of MLEs 算法

为绕过不可能定理，引入额外假设（Assumption 2）：政策特征 $\{\mathbb{E}_{s \sim \rho}[\phi(s,\pi(s))]: \pi \in \Pi\}$ 张成 $\mathbb{R}^d$ 中的超矩形。

算法步骤：
1. 对每个标注者 $i$，从数据 $\mathcal{D}_i$ 计算 MLE $\hat{\theta}_i^{MLE}$
2. 构造置信椭球 $C_i = \{\theta: \|\hat{\theta}_i^{MLE} - \theta\|_{\Sigma_{\mathcal{D}_i}} \leq f(d,n,\delta)\}$
3. 取中位数置信集 $\mathscr{C} = \{\text{med}(\theta_1,\ldots,\theta_k): \theta_i \in C_i\}$
4. 计算悲观中位数回报 $\underline{\mathcal{W}}(\pi) = \min_{\theta \in \mathscr{C}} \mathbb{E}[\langle \theta, \phi(s,\pi(s)) \rangle]$
5. 输出 $\hat{\pi} = \arg\max_{\pi} \underline{\mathcal{W}}(\pi)$

### 损失函数 / 训练策略

悲观优化策略——在所有可能的中位数参数中取最差情况，然后选择在这个最差情况下表现最好的政策。这结合了不确定性量化（置信集）和稳健聚合（中位数规则）。

## 实验关键数据

### 主实验——近似策略防操纵性

**定理 4.1**：Pessimistic Median of MLEs 对标注者 $i$ 是 $\tilde{\mathcal{O}}(\kappa_i\sqrt{d/n})$-策略防操纵的：

$$J_i(\hat{\pi}(\tilde{\mathcal{D}}_i, \mathcal{D}_{-i})) - J_i(\hat{\pi}(\mathcal{D}_i^*, \mathcal{D}_{-i})) \leq const \cdot \kappa_i\sqrt{\frac{d + \log(k/\delta)}{n}}$$

其中 $\kappa_i = \max_{\pi} \|\mathbb{E}[\phi(s,\pi(s))]\|_{\Sigma_{\mathcal{D}_i}^{-1}}$ 是均匀政策覆盖系数。

关键洞察：需要**均匀政策覆盖**（而非仅最优政策覆盖），因为操纵者可能诱导任意远离最优的政策。

### 社会福利保证

**定理 4.2**：在诚实报告或弱占优策略下，输出政策的次优性满足：

$$\text{SubOpt}(\hat{\pi}) \leq const \cdot \left(\sqrt{\frac{d\log(k/\delta)}{k}} + \max_i \kappa_i^* \cdot k\sqrt{\frac{d + \log(k/\delta)}{n}}\right)$$

| 来源 | 次优性贡献 | 消失条件 |
|------|-----------|---------|
| 中位数近似误差 | $O(\sqrt{d\log k / k})$ | $k \to \infty$ |
| 奖励估计误差 | $O(\kappa^* k \sqrt{d/n})$ | $n \to \infty$ |

### 特殊情况分析

| 场景 | 次优性上界 |
|------|-----------|
| 单个标注者 ($k=1$) | $O(\kappa_1^* \sqrt{d/n})$，与经典 RLHF 一致 |
| $k$ 个相同偏好标注者 | $O(\kappa^* k \sqrt{d/n})$，避免了中位数误差项 |
| MDP 扩展 | 类似结构，$\kappa$ 替换为基于状态占用的 $\nu$ |

### 关键发现

1. 存在**策略防操纵与政策对齐的根本权衡**：不可能同时完美实现两者
2. 中位数规则在高维度下提供近似防操纵性，且防操纵强度随样本量增加而增强
3. 需要均匀政策覆盖这一点出人意料——标准 RLHF 理论仅需最优政策覆盖

## 亮点与洞察

1. **跨学科贡献**：首次将社会选择理论（Gibbard-Satterthwaite 定理）引入 RLHF 理论分析，建立了不可能结果
2. **不依赖支付的机制设计**：与基于 VCG 的方法不同，本方法不需要给标注者支付，更贴近实际
3. **覆盖系数的新角色**：揭示了均匀政策覆盖在策略性环境下的必要性，这是标准离线 RL 分析中没有的
4. **优雅的降级**：当样本量和标注者数量增加时，算法自然收敛到最优政策
5. **实际意义**：随着 LLM 被更多使用，标注者的策略性操纵是真实威胁，本文为此提供了理论基础

## 局限与展望

1. **线性奖励函数假设**：实际偏好可能高度非线性，推广到非线性奖励是重要方向
2. **Bradley-Terry 模型假设**：更一般的偏好模型（如 Plackett-Luce）下的分析
3. **缺少实验验证**：全文为理论分析，缺少在实际 LLM 微调场景下的实证评估
4. **超矩形假设（Assumption 2）**：这是不可能定理的突破口，但该假设的实际满足程度不明确
5. **计算复杂度**：中位数置信集上的悲观优化可能在高维度下计算困难
6. **$k$ 因子损失**：多标注者时次优性中的额外 $k$ 因子似乎是算法设计的代价

## 相关工作与启发

- **与 Zhong et al. (2024) 的关系**：以其 Pessimistic Social Welfare 为出发点，展示了其不防操纵性
- **与 Siththaranjan et al. (2023) 的关系**：指出标准 RLHF 隐式使用 Borda 计数投票规则，可能产生操纵激励
- **与对抗鲁棒 RLHF 的区别**：对抗模型假设固定比例的损坏样本，策略模型则假设理性标注者最大化自身效用
- **启发**：悲观 + 中位数的设计范式可推广到其他存在策略性参与者的学习场景

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首次将策略防操纵性引入 RLHF 并建立不可能定理
- **理论深度**: ⭐⭐⭐⭐⭐ — 不可能定理和算法设计都有扎实的理论支撑
- **实验充分度**: ⭐⭐ — 纯理论分析，完全没有实验
- **写作质量**: ⭐⭐⭐⭐ — 逻辑清晰，定理陈述精确，但部分符号较重
- **实用价值**: ⭐⭐⭐⭐ — 问题动机强烈，但算法的实际可计算性和可扩展性有待验证
- **综合**: ⭐⭐⭐⭐ (8.5/10) — 极具开创性的理论工作，开辟了 RLHF 与机制设计交叉的新方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Curiosity-Driven Reinforcement Learning from Human Feedback](../../ACL2025/llm_alignment/curiosity_driven_rlhf.md)
- [\[ICML 2025\] M³HF: Multi-agent Reinforcement Learning from Multi-phase Human Feedback of Mixed Quality](../../ICML2025/llm_alignment/m3hf_multi-agent_reinforcement_learning_from_multi-phase_human_feedback_of_mixed.md)
- [\[NeurIPS 2025\] Reinforcement Learning Finetunes Small Subnetworks in Large Language Models](reinforcement_learning_finetunes_small_subnetworks_in_large_language_models.md)
- [\[ACL 2026\] PERSA: Reinforcement Learning for Professor-Style Personalized Feedback with LLMs](../../ACL2026/llm_alignment/persa_reinforcement_learning_for_professor-style_personalized_feedback_with_llms.md)
- [\[ACL 2025\] Understanding Impact of Human Feedback via Influence Functions](../../ACL2025/llm_alignment/influence_functions_rlhf.md)

</div>

<!-- RELATED:END -->
