---
title: >-
  [论文解读] On the Sample Complexity of Differentially Private Policy Optimization
description: >-
  [NeurIPS 2025][LLM安全][差分隐私] 首次系统性研究差分隐私（DP）约束下策略优化的样本复杂度，提出统一的元算法框架，分析DP-PG、DP-NPG和DP-REBEL三种隐私策略优化算法，证明隐私代价通常仅作为样本复杂度的低阶项出现。 问题背景 策略优化（PO）如REINFORCE、PPO、GRPO在机器人控…
tags:
  - "NeurIPS 2025"
  - "LLM安全"
  - "差分隐私"
  - "策略优化"
  - "样本复杂度"
  - "强化学习"
  - "隐私保护"
---

# On the Sample Complexity of Differentially Private Policy Optimization

**会议**: NeurIPS 2025  
**arXiv**: [2510.21060](https://arxiv.org/abs/2510.21060)  
**作者**: Yi He (Wayne State University), Xingyu Zhou (Wayne State University)  
**代码**: 未公开  
**领域**: AI安全  
**关键词**: 差分隐私, 策略优化, 样本复杂度, 强化学习, 隐私保护  

## 一句话总结

首次系统性研究差分隐私（DP）约束下策略优化的样本复杂度，提出统一的元算法框架，分析DP-PG、DP-NPG和DP-REBEL三种隐私策略优化算法，证明隐私代价通常仅作为样本复杂度的低阶项出现。

## 研究背景与动机

### 问题背景
策略优化（PO）如REINFORCE、PPO、GRPO在机器人控制、医疗保健和LLM训练中广泛应用。然而，当PO部署在敏感领域时，隐私问题日益突出：
- **医疗场景**：患者交互数据（医疗状态→处方→疗效）构成敏感信息
- **LLM训练**：用户提示词（prompt）可能包含私人信息
- 实证研究已表明标准GRPO存在隐私泄露问题

### 已有工作的不足
- 现有隐私策略梯度工作主要从经验角度出发，缺乏样本复杂度理论保证
- 在线遗憾角度的工作（如隐私PPO）仅限于表格型或线性设定
- 标准差分隐私定义无法直接应用于策略优化，原因有二：(1) PO中不存在固定数据集（on-policy采样）；(2) 改变一个样本会因策略变化导致后续所有样本改变
- 隐私策略评估（policy evaluation）仅评估给定策略，而非寻找最优策略

### 核心动机
回答核心问题：**差分隐私在策略优化中引入了多少样本复杂度代价？** 为此需要：定义适合PO的DP概念、设计统一算法框架、建立严格的样本复杂度界。

## 方法详解

### 问题建模
采用bandit公式化（可标准推广至MDP）：给定初始状态 $x \sim \rho$，动作 $y \sim \pi_\theta(\cdot|x)$，奖励 $r(x,y) \in [-R_{\max}, R_{\max}]$，目标最大化：
$$J(\theta) = \mathbb{E}_{x \sim \rho, y \sim \pi_\theta(\cdot|x)}[r(x,y)]$$

### PO中差分隐私的定义（Definition 2）
关键创新：将"用户"作为隐私单元。数据集 $D$ 由 $N$ 个用户组成（如 $N$ 个患者或 $N$ 个prompt），隐私保护要求更换一个用户不会显著改变最终输出策略：
$$\mathbb{P}[\mathcal{M}(D) \in S] \leq e^\varepsilon \cdot \mathbb{P}[\mathcal{M}(D') \in S] + \delta$$
其中 $D, D'$ 仅在一个用户上不同。

### 元算法（Algorithm 1）
设计批量单程（batched one-pass）元算法：
1. 每轮 $t$ 从分布中采集 $m$ 个新样本 $\bar{D}_t = \{(x_i, y_i, y_i')\}_{i=1}^m$
2. 计算优势估计 $\hat{A}_t(x_i, y_i) = r(x_i, y_i) - r(x_i, y_i')$
3. 调用 PrivUpdate 预言机更新策略 $\theta_{t+1}$
4. 总样本量 $N = m \cdot T$，关键在于平衡每轮精度（$m$）与迭代次数（$T$）

由单程设计和DP并行组合，只要PrivUpdate满足 $(\varepsilon, \delta)$-DP，整体算法即满足PO中的 $(\varepsilon, \delta)$-DP。

### DP-PG：隐私策略梯度
- 计算REINFORCE风格梯度估计后加入高斯噪声，噪声方差 $\sigma^2 = \frac{16\log(1.25/\delta) R_{\max}^2 G^2}{m^2 \varepsilon^2}$
- **FOSP收敛**（Theorem 2）：$\mathbb{E}[\|\nabla J(\theta_U)\|^2] \leq O_\delta\left(\frac{1}{\sqrt{N}} + \left(\frac{\sqrt{d}}{N\varepsilon}\right)^{2/3}\right)$
- **全局最优收敛**（Theorem 3）：在Fisher非退化和兼容函数近似假设下，样本复杂度为 $N \geq O_\delta\left(\frac{1}{\alpha^4 \gamma^4} + \frac{\sqrt{d}}{\alpha^3 \gamma^3 \varepsilon}\right)$
- **Softmax表格设定**（Theorem 5）：通过log-barrier正则化消除对 $\gamma$ 的依赖，$N \geq O\left(\frac{1}{\alpha^6} + \frac{\sqrt{d}}{\alpha^{9/2} \varepsilon}\right)$

### DP-NPG：隐私自然策略梯度
- 将PO归约为一系列隐私回归问题，利用隐私最小二乘预言机
- **主定理**（Theorem 4）：对任意比较策略 $\pi^*$，$J(\pi^*) - \frac{1}{T}\sum_{t=1}^T J(\pi_t) \leq \sqrt{\frac{\beta W^2 \log|\mathcal{Y}|}{2T}} + \frac{\sqrt{C_{\mu \to \pi^*}}}{T}\sum_{t=1}^T \text{err}_t$
- **一般函数类**（Corollary 1，指数机制）：纯DP（$\delta=0$），$N = \widetilde{O}\left(\left(\frac{1}{\alpha^4} + \frac{1}{\alpha^4\varepsilon}\right) \cdot \log|\mathcal{W}|\right)$
- **Log-linear低维**（Corollary 2）：$N = \widetilde{O}_\delta\left(\frac{d}{\alpha^4} + \frac{d}{\alpha^3\varepsilon} + \frac{d}{\alpha^2\varepsilon^2}\right)$
- **Log-linear高维**（Corollary 3）：$N = \widetilde{O}_\delta\left(\frac{1}{\alpha^6} + \frac{1}{\alpha^5\varepsilon}\right)$

### DP-REBEL：隐私REBEL
结构与DP-NPG类似，将PO归约为相对奖励差异上的回归问题（Theorem 6），达到与DP-NPG几乎相同的样本复杂度界。

## 实验关键数据

### 实验1：CartPole-v1 隐私预算对比

在CartPole-v1环境上，使用两层全连接网络（64隐藏单元），训练100个epoch，batch size=10，$\delta=10^{-5}$，3个随机种子取平均。

| 算法 | $\varepsilon$ | 平均最终奖励 | 标准差 | 最佳epoch均值 |
|------|:---:|:---:|:---:|:---:|
| PG | N/A | 334.37 | 25.25 | 361.70 |
| DP-PG | 5 | 190.34 | 52.91 | 199.17 |
| DP-PG | 3 | 143.87 | 22.88 | 187.17 |
| NPG | N/A | 492.90 | 10.04 | 500.00 |
| DP-NPG | 5 | 478.73 | 28.05 | 494.70 |
| DP-NPG | 3 | 400.87 | 65.37 | 410.43 |

关键发现：(1) NPG在隐私和非隐私设定中均显著优于PG；(2) DP-NPG在$\varepsilon=5$时几乎达到最优性能（约500），验证了理论预测中隐私代价为低阶项；(3) 隐私预算降低时性能如理论预期般下降。

### 实验2：各算法样本复杂度理论结果汇总

| 算法 | 设定 | 非隐私项 | 隐私代价项 |
|------|------|:---:|:---:|
| DP-PG | FOSP | $O(1/\alpha^4)$ | $O_\delta(\sqrt{d}/(\alpha^3\varepsilon))$ |
| DP-PG | 全局（Fisher非退化） | $O(1/(\alpha^4\gamma^4))$ | $O_\delta(\sqrt{d}/(\alpha^3\gamma^3\varepsilon))$ |
| DP-PG | 全局（Softmax表格） | $O(1/\alpha^6)$ | $O(\sqrt{d}/(\alpha^{9/2}\varepsilon))$ |
| DP-NPG | 一般函数类（纯DP） | $O(1/\alpha^4)$ | $O(1/(\alpha^4\varepsilon))$ |
| DP-NPG | Log-linear低维 | $O(d/\alpha^4)$ | $O_\delta(d/(\alpha^3\varepsilon))$ |
| DP-NPG | Log-linear高维 | $O(1/\alpha^6)$ | $O_\delta(1/(\alpha^5\varepsilon))$ |
| DP-REBEL | 各设定 | 与DP-NPG相同 | 与DP-NPG相同 |

核心结论：在所有设定中，隐私代价均表现为样本复杂度的**加性低阶项**（当 $\varepsilon, d$ 为常数时）。

## 亮点

- **PO中DP的准确定义**：创造性地借鉴在线bandit文献中"用户"概念，解决了on-policy学习动态下隐私单元定义的微妙问题，区分了SFT和RL微调中隐私保护的本质差异
- **统一元算法框架**：单一batched one-pass框架统一涵盖DP-PG、DP-NPG、DP-REBEL三类算法，由并行组合自然获得隐私保证
- **隐私代价的量化**：严格证明在多种设定下隐私代价仅为低阶项，为隐私保护PO的实际部署提供了理论支撑
- **NPG的归约视角**：将DP-NPG/DP-REBEL归约为隐私回归问题，可直接复用隐私估计/监督学习中的成熟结果
- **新的隐私最小二乘引理**（Lemma 2）：基于指数机制的隐私LS结果，适用于序列化自适应数据，可能具有独立研究价值

## 局限与展望

- **仅限单程采样**：当前结果聚焦于one-pass设定，多次遍历场景下的样本复杂度可能更优
- **缺乏大规模实验**：仅在CartPole-v1上做了概念验证，未在LLM-RLHF、医疗等实际隐私敏感场景中评估
- **$W$的维度依赖**：DP-NPG中由于私有化噪声，权重 $W$ 可能依赖 $\sqrt{d}$，影响最终bound质量
- **Bandit建模简化**：主体聚焦于bandit公式而非完整MDP，虽然推广是标准的但长时序隐私分析更复杂
- **Fisher非退化假设强**：全局收敛结果依赖Fisher矩阵正定（$\gamma > 0$），当$\gamma$很小时bound较松
- **未涉及用户级组合隐私**：每个用户仅交互一次，未覆盖同一用户多次参与的更现实场景

## 与相关工作的对比

- **Rio et al. (2025)**：实证研究隐私策略梯度（PPO），无样本复杂度理论保证；本文提供首个系统理论分析
- **Chowdhury & Zhou (2022)**：表格型MDP的隐私PPO在线遗憾分析；本文考虑一般函数类的优化视角
- **Zhou & Tan (2024)**：线性MDP的隐私OPT-PPO；本文框架更通用，涵盖NPG和REBEL
- **Balle et al. (2016)**：隐私策略评估而非策略优化
- **非隐私PO文献**（Yuan et al. 2023, Agarwal et al. 2021）：本文的非隐私项精确恢复已有最优bound，隐私项为额外加性代价
- **隐私随机凸优化**（Bassily et al. 2019）：本文通过归约将PO与经典隐私优化连接，但on-policy特性引入新挑战

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次系统建立隐私PO的样本复杂度理论，PO中DP定义的创新处理有价值
- 实验充分度: ⭐⭐ — 仅CartPole一个环境，实验为附录中的概念验证
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，理论体系完整，动机和技术路线阐述流畅
- 价值: ⭐⭐⭐⭐ — 填补PO隐私理论空白，为RLHF等隐私敏感应用的算法设计提供理论指导

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Differentially Private Federated Low Rank Adaptation Beyond Fixed-Matrix](differentially_private_federated_low_rank_adaptation_beyond_fixed-matrix.md)
- [\[ICLR 2026\] wd1: Weighted Policy Optimization for Reasoning in Diffusion Language Models](../../ICLR2026/llm_safety/wd1_weighted_policy_optimization_for_reasoning_in_diffusion_language_models.md)
- [\[ACL 2026\] Differentially Private Synthetic Text Generation for Retrieval-Augmented Generation (RAG)](../../ACL2026/llm_safety/differentially_private_synthetic_text_generation_for_retrieval-augmented_generat.md)
- [\[ICLR 2026\] PURGE: Reinforcement Unlearning via Group Relative Policy Optimization](../../ICLR2026/llm_safety/reinforcement_unlearning_via_group_relative_policy_optimization.md)
- [\[NeurIPS 2025\] FedSVD: Adaptive Orthogonalization for Private Federated Learning with LoRA](fedsvd_adaptive_orthogonalization_for_private_federated_learning_with_lora.md)

</div>

<!-- RELATED:END -->
