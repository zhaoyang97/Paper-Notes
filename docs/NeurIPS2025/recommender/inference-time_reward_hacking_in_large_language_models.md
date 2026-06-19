---
title: >-
  [论文解读] Inference-Time Reward Hacking in Large Language Models
description: >-
  [NeurIPS 2025 Spotlight][推荐系统][reward hacking] 本文从数学上证明了推理时对齐方法（如 BoN）在优化代理奖励时不可避免地会出现 reward hacking（真实奖励先升后降），提出了 Best-of-Poisson (BoP) 采样方法近似最优 KL-奖励折中分布，并设计了 HedgeTune 算法通过一维寻根找到最优推理时参数，在数学推理和人类偏好场景中有效缓解 reward hacking。
tags:
  - "NeurIPS 2025 Spotlight"
  - "推荐系统"
  - "reward hacking"
  - "inference-time alignment"
  - "Best-of-N"
  - "winner's curse"
  - "hedging"
---

# Inference-Time Reward Hacking in Large Language Models

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2506.19248](https://arxiv.org/abs/2506.19248)  
**代码**: 无  
**领域**: 推荐系统  
**关键词**: reward hacking, inference-time alignment, Best-of-N, winner's curse, hedging

## 一句话总结
本文从数学上证明了推理时对齐方法（如 BoN）在优化代理奖励时不可避免地会出现 reward hacking（真实奖励先升后降），提出了 Best-of-Poisson (BoP) 采样方法近似最优 KL-奖励折中分布，并设计了 HedgeTune 算法通过一维寻根找到最优推理时参数，在数学推理和人类偏好场景中有效缓解 reward hacking。

## 研究背景与动机
**领域现状**: 当前 LLM 对齐方法（RLHF、DPO、BoN 等）的核心范式都是最大化奖励函数同时最小化与参考模型的 KL 散度。其中 Best-of-N (BoN) 因简单高效而被广泛使用——生成 N 个候选回复，选奖励最高的输出。

**现有痛点**: 所有代理奖励模型都是不完美的，它们无法精确捕获正确性、有用性、安全性等复杂目标。优化一个有偏的代理奖励会导致 reward hacking：真实性能先提升再下降。

**核心矛盾**: BoN 等方法天然受"赢家诅咒"（winner's curse）影响——当候选数 N 增大时，被选中的回复倾向于代理奖励高估了真实质量的那个，导致过度优化。

**本文目标**: 刻画推理时 reward hacking 的不可避免性，并提供实用的缓解机制。

**切入角度**: 从信息论和拍卖理论中的赢家诅咒切入，将推理时对齐的参数调优问题转化为一维寻根问题。

**核心 idea**: 通过 Poisson 随机化采样数近似最优带温分布，再用 HedgeTune 找到 hacking 阈值，实现对代理奖励的最优"对冲"。

## 方法详解

### 整体框架
本文研究的核心优化目标是标准的 KL 约束奖励最大化：
$$\pi^{\star} = \arg\max_{\pi} \mathbb{E}_{\pi}[r_p(X)] - \frac{1}{\lambda} D_{\text{KL}}(\pi \| \pi_{\text{ref}})$$

理论最优解是参考分布的指数倾斜（exponential tilt），但在实践中无法直接采样（需要遍历所有可能的续写）。因此需要推理时近似方法。

Pipeline：
1. 从参考模型 $\pi_{\text{ref}}$ 采样 N 个候选回复
2. 用代理奖励模型对候选打分
3. 通过选择机制（BoN/SBoN/BoP）选出一个输出
4. 用 HedgeTune 校准选择机制的参数以避免过度优化

### 关键设计
1. **Reward Hacking 的形式化定义 (Definition 1)**: 定义 hacking 阈值 $\theta^{\dagger}$——当推理时参数超过此阈值后，真实奖励开始下降。通过 Theorem 1 证明在 TP2（全正性）和单调似然比条件下，真实奖励函数关于参数要么单调、要么恰好有一个极值点（unimodal），从而证明 reward hacking 是不可避免的。

2. **Best-of-Poisson (BoP) 采样 (Algorithm 3)**: 核心创新——将 BoN 的固定采样数 N 替换为 Poisson 分布随机变量 $n' \sim \text{Poisson}(\mu)$，取 $n = n' + 1$ 保证至少一个样本。BoP 的密度为：
    $q_{\mu}(x) = (\mu x + 1) e^{\mu(x-1)}, \quad x \in [0,1]$
   关键优势：BoP 用单一参数 $\mu$ 即可近似最优倾斜分布，KL 差距仅为 $O(10^{-4})$（在均匀代理奖励假设下）。这意味着 BoP 可以作为 RLHF 最优策略的推理时近似，无需为每个 $\lambda$ 重新微调模型。

3. **HedgeTune 算法 (Algorithm 4)**: 目标是找到 hacking 阈值 $\theta^{\dagger}$，使真实奖励的边际收益为零。

    - 对每个 prompt，将代理奖励分数映射到经验分位数 $u \in [0,1]$
    - 构造残差函数 $R(\theta) = \mathbb{E}_{u \sim p_\theta}[r_t(u) \cdot \psi(u, \theta)]$
    - 通过二分法或牛顿法求解 $\bar{R}(\theta^{\star}) = 0$
    - 对 BoN：找最优 N；对 SBoN：找最优 $\lambda$；对 BoP：找最优 $\mu$

### 损失函数 / 训练策略
- HedgeTune 不需要访问 LLM 分布本身，只需代理奖励和真实奖励的评分数据
- 需要一次性校准（one-time calibration），适用于可验证奖励场景（数学推理、程序合成）或使用 LLM-as-a-judge
- 代理奖励模型使用标准 binary cross-entropy 损失在偏好对上训练

## 实验关键数据

### 主实验一：可验证奖励场景
使用 PPE 数据集（GPT-4o-mini / Claude Haiku 3 生成的回复），三个奖励模型打分：

| 数据集 | 奖励模型 | BoN 最优 N | BoP 最优 μ | HedgeTune 恢复峰值 |
|---------|----------|-----------|-----------|-------------------|
| MMLU Pro | InternLM-2 1.8B | ~8 | ~7 | ✓ 成功 |
| MATH | Llama-3-Offset-Bias 8B | ~16 | ~14 | ✓ 成功 |
| GPQA | Skywork-Llama-3.1 8B | ~32 | ~30 | ✓ 成功 |

关键发现：即使使用 RewardBench 排名第 12 的 Skywork 8B 奖励模型，在 GPQA 上 BoN 仍出现 hacking（N 过大后准确率下降）。HedgeTune 在所有设置中成功恢复最佳操作点。

### 主实验二：人类偏好场景
使用 Pythia 1.4B 参考模型 + AlpacaFarm + AlpacaRM 金标准奖励：

| 代理 RM 训练数据大小 | 标签噪声 | BoN hacking 阈值 N† | SBoN 最优 λ† | BoP hacking 阈值 μ† |
|--------------------|---------|--------------------|-------------|-------------------|
| 10k | 0% | ~16 | ~2.5 | ~14 |
| 20k | 0% | ~64 | ~4.0 | ~60 |
| 46k | 25% | ~8 | ~1.5 | ~7 |
| 80k | 25% | ~32 | ~3.0 | ~28 |

关键发现：代理 RM 训练数据越少或噪声越大，hacking 阈值越低（更早开始退化）。SBoN 通过温度 $\lambda$ 可以实现峰值真实奖励而不发生 hacking。

### 消融实验

| 方法 | 参数数量 | 能否近似最优分布 | KL gap | hacking 缓解能力 |
|------|---------|----------------|--------|-----------------|
| BoN | 1 (N) | 否 | N/A | 需要 HedgeTune |
| SBoN | 2 (N, λ) | 否（但更灵活） | N/A | λ=0 可回退到参考 |
| BoP | 1 (μ) | 是（gap < 8×10⁻⁴） | O(10⁻⁴) | 需要 HedgeTune |
| 最优倾斜分布 | 1 (λ) | 是（理论最优） | 0 | 不可采样 |

### 关键发现
- BoP 用单参数即可达到与最优倾斜分布近乎相同的 KL-奖励折中，KL 差距始终 < 8×10⁻⁴
- reward hacking 的"先升后降"模式是 MLR 密度族（包括 BoN、BoP）的固有属性
- HedgeTune 的计算开销极小（仅一维寻根），可直接复用已有采样数据
- SBoN 在某些设置下能以某个固定 λ 完全避免 hacking（当阈值不可达时返回最佳可达奖励）

## 亮点与洞察
- 将拍卖理论的赢家诅咒与 LLM 对齐优雅地联系起来，理论新颖且有实践指导意义
- BoP 的设计极其精巧：Poisson 随机化引入指数结构，自然逼近最优倾斜分布
- Theorem 1 的通用性很强——适用于任何满足 TP2 的推理时方法，不限于 BoN
- HedgeTune 实用性好：无需访问 LLM 内部参数，只需黑盒评分数据

## 局限与展望
- HedgeTune 需要访问真实奖励（或强 judge），在无法验证的开放式任务中适用性受限
- 理论分析依赖均匀代理奖励假设（虽然通过 CDF 变换损失小，但离散情况需要附录额外处理）
- 未讨论多轮对话或序列决策场景中的 reward hacking
- BoP 的 Poisson 随机化带来采样数的方差，可能影响延迟的可预测性

## 相关工作与启发
- **vs Gao et al. (2023) 的 scaling law**: Gao 等经验性地观察了 BoN 的 reward hacking，本文首次给出了严格的数学证明（Theorem 1 的不可避免性）
- **vs SBoN (Mayrink Verdun et al. 2025)**: SBoN 引入温度参数提供灵活性，但需要调两个参数；BoP 用单参数达到相近效果
- **vs RLHF/DPO**: 本文方法完全在推理时操作，不需要微调模型，可视为 RLHF 的免训练替代
- **vs Huang et al. (2025) 的 coverage 分析**: 他们证明 BoN 在 N 大时必然 hack，本文提供了具体的缓解方案

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 赢家诅咒视角+BoP+HedgeTune 组合创新度很高，理论贡献扎实
- 实验充分度: ⭐⭐⭐⭐ 覆盖可验证奖励和人类偏好两类场景，多个奖励模型和数据集，但缺少更大规模 LLM 实验
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，理论推导严谨，图表信息量大且美观
- 价值: ⭐⭐⭐⭐⭐ 对推理时对齐方法的安全部署有直接指导意义，BoP 和 HedgeTune 可即插即用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Inference-Aware Prompt Optimization for Aligning Black-Box Large Language Models](../../AAAI2026/recommender/inference-aware_prompt_optimization_for_aligning_black-box_large_language_models.md)
- [\[NeurIPS 2025\] R²ec: Towards Large Recommender Models with Reasoning](r2ec_towards_large_recommender_models_with_reasoning.md)
- [\[ACL 2025\] KERL: Knowledge-Enhanced Personalized Recipe Recommendation using Large Language Models](../../ACL2025/recommender/kerl_knowledge-enhanced_personalized_recipe_recommendation_using_large_language_.md)
- [\[ICLR 2026\] From Evaluation to Defense: Advancing Safety in Video Large Language Models](../../ICLR2026/recommender/from_evaluation_to_defense_advancing_safety_in_video_large_language_models.md)
- [\[ICML 2025\] PARM: Multi-Objective Test-Time Alignment via Preference-Aware Autoregressive Reward Model](../../ICML2025/recommender/parm_multi-objective_test-time_alignment_via_preference-aware_autoregressive_rew.md)

</div>

<!-- RELATED:END -->
