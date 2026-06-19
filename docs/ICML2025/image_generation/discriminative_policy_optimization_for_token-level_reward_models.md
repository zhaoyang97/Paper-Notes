---
title: >-
  [论文解读] Discriminative Policy Optimization for Token-Level Reward Models
description: >-
  [ICML2025][图像生成][token-level reward model] 提出 Q-function Reward Model (Q-RM)，通过将奖励建模与语言生成解耦，定义判别式策略来学习 token 级 Q 函数，从偏好数据中无需细粒度标注即可获得精确的 token 级奖励信号，显著提升 PPO/REINFORCE 的推理性能与训练效率。
tags:
  - "ICML2025"
  - "图像生成"
  - "token-level reward model"
  - "process reward model"
  - "Q-function"
  - "discriminative policy"
  - "PPO"
  - "REINFORCE"
  - "LLM alignment"
---

# Discriminative Policy Optimization for Token-Level Reward Models

**会议**: ICML2025  
**arXiv**: [2505.23363](https://arxiv.org/abs/2505.23363)  
**代码**: [homzer/Q-RM](https://github.com/homzer/Q-RM)  
**领域**: 图像生成  
**关键词**: token-level reward model, process reward model, Q-function, discriminative policy, PPO, REINFORCE, LLM alignment

## 一句话总结

提出 Q-function Reward Model (Q-RM)，通过将奖励建模与语言生成解耦，定义判别式策略来学习 token 级 Q 函数，从偏好数据中无需细粒度标注即可获得精确的 token 级奖励信号，显著提升 PPO/REINFORCE 的推理性能与训练效率。

## 研究背景与动机

- **过程奖励模型 (PRM)** 相比结果奖励模型 (ORM) 能提供更细粒度的逐步反馈，但存在粒度不匹配问题：PPO 在 token 级操作，而 ORM/PRM 在序列或步骤级给奖励
- **DPO-RM** 方案将奖励定义为 $r^{\text{DPO}}(s_t, a_t) = \beta \log \frac{\pi(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)}$ ，存在两个核心缺陷：
    1. 生成式语言建模与奖励建模耦合：高生成概率 ≠ 高奖励（如模型对错误答案也很自信）
    2. 依赖参考模型 $\pi_{\text{ref}}$ 引入额外不确定性，导致异常奖励分配
- **可视化证据**：DPO-RM 倾向于给换行符等非关键 token 分配高奖励，却忽略关键数值 token（如 "$7"、"$133"），而 Q-RM 能正确给正确 token 高分、错误 token 低分

## 方法详解

### 1. 判别式策略定义

与生成式策略 $\pi(a_t|s_t)$ 不同，定义判别式策略 $\phi(s_t, a_t)$ 同时接受状态和动作作为输入，通过 logit 值 $Z(s_t, a_t)$ 经 softmax 转换为概率：

$$\phi(s_t, a_t) = \frac{\exp Z(s_t, a_t)}{\sum_{a'_t \in \mathcal{A}} \exp Z(s_t, a'_t)}$$

核心区别：生成式策略输出所有动作的概率分布，判别式策略评估特定动作的奖励。

### 2. 奖励推导

在最大熵 RL 框架下，最优判别式策略满足：

$$\beta \log \phi^*(s_t, a_t) = Q^*(s_t, a_t) - V^*(s_t)$$

结合 Bellman 方程得到 token 级奖励：$r(s_t, a_t) = \beta \log \phi^*(s_t, a_t) + V^*(s_t) - V^*(s_{t+1})$

### 3. 轨迹奖励分解与化简

轨迹奖励 $\mathcal{R}(\tau)$ 被分解为 $\beta(\mathcal{Q}(\tau) - \mathcal{V}(\tau))$，其中：

- $\mathcal{Q}(\tau) = \frac{1}{T}\sum_{t=0}^{T-1}(Z^*(s_t, a_t) - z_t)$，$z_t = \max_{a_t} Z^*(s_t, a_t)$
- $\mathcal{V}(\tau)$ 为调整后的配分函数对数均值

**关键理论贡献**：证明 $\mathcal{V}(\tau)$ 的上界受最优策略熵约束（$0 \leq \mathcal{V}(\tau) \leq \mathcal{H}^*(\tau)$），当最优策略近似确定性时 $|\mathcal{V}(\tau^w) - \mathcal{V}(\tau^l)| \to 0$，可安全忽略。

### 4. 训练目标

最终损失函数基于 Bradley-Terry 模型：

$$p(\tau^w \succeq \tau^l) = \sigma\left[\beta\left(\frac{1}{N}\sum_{t=0}^{N-1}Z^*(s_t^w, a_t^w) - \frac{1}{M}\sum_{t=0}^{M-1}Z^*(s_t^l, a_t^l)\right) - \gamma\right]$$

其中 $\gamma$ 为全局偏置超参数（固定为 2），$\beta = 0.2$。仅需偏好数据训练，无需细粒度标注。

### 5. Q-RM 与 PPO/REINFORCE 集成

- **PPO**：直接计算优势函数 $A(s_t, a_t) = Z^*_{\text{std}}(s_t, a_t) - V_\psi(s_t)$，无需 GAE
- **REINFORCE**：用 $Z^*_{\text{std}}(s_t, a_t)$ 作为累积奖励
- 对所有 token 奖励做标准化（均值 0，方差 1）确保训练稳定

### 6. 理论保证

**Proposition 3.4**：最优 Q 函数 $Q^*(s_t, a_t)$ 与判别式策略 logit $Z^*(s_t, a_t)$ 偏差期望一致，即用 $Z^*$ 计算优势函数等价于用 $Q^*$。

## 实验关键数据

**设置**：策略模型 Llama-3.2-3B-Instruct，奖励模型 Llama-3-70B-Instruct，LoRA rank 128，学习率 1e-5。

### 数学推理 (GSM8K & MATH)

| 方法 | GSM8K Pass@1 | GSM8K Pass@16 | MATH Pass@1 | MATH Pass@16 | Avg Pass@1 |
|------|-------------|---------------|-------------|--------------|------------|
| SFT | 63.08 | 87.95 | 27.57 | 55.48 | 45.33 |
| DPO | 68.16 | 91.13 | 29.80 | 58.44 | 48.98 |
| PPO+ORM | 66.26 | 88.02 | 27.22 | 56.59 | 46.74 |
| PPO+DPO-RM | 68.67 | 88.02 | 27.39 | 55.72 | 48.03 |
| **PPO+Q-RM** | **72.23** | 92.49 | **32.95** | **64.19** | **52.59** |
| REINFORCE+ORM | 67.55 | 89.69 | 29.60 | 57.86 | 48.58 |
| **REINFORCE+Q-RM** | **72.10** | **93.48** | **34.45** | 62.87 | **53.28** |

- PPO+Q-RM 比 ORM 提升 **+5.85** 平均 Pass@1，比 DPO-RM 提升 **+4.56**
- REINFORCE+Q-RM 比 ORM 提升 **+4.70**，比 DPO-RM 提升 **+5.73**

### QA-Feedback

| 方法 | Relevance | Factuality | Completeness | Avg |
|------|-----------|------------|--------------|-----|
| PPO+Q-RM | **0.5510** | 0.6814 | **0.5545** | **0.5956** |
| REINFORCE+Q-RM | 0.5454 | 0.6808 | 0.5490 | 0.5917 |
| PPO+DPO-RM | 0.4769 | 0.6802 | 0.5323 | 0.5631 |

### 训练效率

- Q-RM 在 GSM8K 上收敛速度比 ORM **快 12×**
- Q-RM 在 MATH 上收敛速度比 step-level PRM **快 11×**

## 亮点与洞察

1. **解耦思路精妙**：将奖励建模从语言生成中解耦，用判别式策略替代生成式策略建模奖励，从根本上避免了"生成概率高≠奖励高"的冲突
2. **理论完备**：证明 logit $Z^*$ 与最优 Q 函数偏差结构一致，可直接替代 Q 值计算优势函数，无需 GAE
3. **实用性强**：无需细粒度标注，仅用偏好数据即可训练；$\gamma$ 固定为常数即可跨任务工作
4. **训练效率跃升**：收敛速度提升 11-12×，大幅降低 RL 训练成本
5. **奖励可视化**直觉清晰：Q-RM 精准定位关键 token（正确数值高分、错误数值低分），DPO-RM 则对换行符等噪声 token 敏感

## 局限与展望

1. **Assumption 3.3 的适用范围**：假设最优策略熵趋近于零在创意生成等多样性场景可能不成立
2. **$\gamma$ 固定为常数**：实际上 $\gamma$ 随样本变化，固定常数是近似，对极端长度差异的偏好对可能不理想
3. **奖励模型规模依赖**：实验使用 70B 奖励模型配 3B 策略模型，在资源受限场景下的表现未充分验证
4. **评估任务偏数学推理**：对代码生成、开放域对话等任务的泛化性需进一步验证
5. **判别式策略的 softmax 计算**：仍需遍历整个词表进行归一化，理论化简依赖假设

## 相关工作与启发

- **DPO/SimPO/ORPO**：离线 RL 对齐方法，Q-RM 作为在线 RL 的奖励模型与之互补
- **Implicit-PRM (CE)**：同为 token 级 PRM 但基于生成式策略，Q-RM 通过判别式解耦获得更好性能
- **PGG-RM**：另一种 token 级奖励方法，Q-RM 在几乎所有指标上超越
- **启发**：判别式与生成式解耦的思路可推广到其他需要细粒度信用分配的场景（如代码调试、长文本生成）

## 评分

- 新颖性: ⭐⭐⭐⭐ （判别式策略解耦奖励建模是新颖视角）
- 实验充分度: ⭐⭐⭐⭐⭐ （4 个任务，多种 baseline，含 PPO 和 REINFORCE 两种 RL 框架）
- 写作质量: ⭐⭐⭐⭐ （理论推导清晰，实验组织系统）
- 价值: ⭐⭐⭐⭐⭐ （对 LLM 对齐中 token 级奖励建模有实质推动）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] PPO-MI: Efficient Black-Box Model Inversion via Proximal Policy Optimization](ppo-mi_efficient_black-box_model_inversion_via_proximal_policy_optimization.md)
- [\[ICML 2025\] ToMA: Token Merge with Attention for Diffusion Models](toma_token_merge_with_attention_for_diffusion_models.md)
- [\[NeurIPS 2025\] Diffusion Model as a Noise-Aware Latent Reward Model for Step-Level Preference Optimization](../../NeurIPS2025/image_generation/diffusion_model_as_a_noiseaware_latent_reward_model_for_step.md)
- [\[ICML 2026\] Principled RL for Flow Matching Emerges from the Chunk-level Policy Optimization](../../ICML2026/image_generation/principled_rl_for_flow_matching_emerges_from_the_chunk-level_policy_optimization.md)
- [\[ICML 2025\] Direct Discriminative Optimization: Your Likelihood-Based Visual Generative Model is also a GAN Discriminator](direct_discriminative_optimization_your_likelihood-based_visual_generative_model.md)

</div>

<!-- RELATED:END -->
