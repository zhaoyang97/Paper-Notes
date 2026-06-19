---
title: >-
  [论文解读] Inference-time Alignment in Continuous Space
description: >-
  [NeurIPS 2025][LLM对齐][inference-time alignment] 提出 Simple Energy Adaptation (SEA)，将推理时对齐从"离散空间搜索"范式转变为"连续空间优化"范式，通过在连续 logit 空间上进行基于梯度的 Langevin 采样来逼近 RLHF 最优策略，在 AdvBench 上相对最优基线提升 77.51%，在 MATH 上提升 16.36%。
tags:
  - "NeurIPS 2025"
  - "LLM对齐"
  - "inference-time alignment"
  - "energy-based model"
  - "Langevin dynamics"
  - "RLHF"
  - "continuous optimization"
  - "reward model"
---

# Inference-time Alignment in Continuous Space

**会议**: NeurIPS 2025  
**arXiv**: [2505.20081](https://arxiv.org/abs/2505.20081)  
**代码**: [GitHub](https://github.com/YigeYuan/SEA)  
**领域**: LLM对齐  
**关键词**: inference-time alignment, energy-based model, Langevin dynamics, RLHF, continuous optimization, reward model

## 一句话总结

提出 Simple Energy Adaptation (SEA)，将推理时对齐从"离散空间搜索"范式转变为"连续空间优化"范式，通过在连续 logit 空间上进行基于梯度的 Langevin 采样来逼近 RLHF 最优策略，在 AdvBench 上相对最优基线提升 77.51%，在 MATH 上提升 16.36%。

## 研究背景与动机

推理时对齐（Inference-time Alignment）因无需额外训练、可即插即用而受到越来越多关注。现有方法主要采用"离散空间搜索"范式：

**Best-of-N (BoN)**：从基础策略生成 N 个候选响应，选择奖励最高的

**ARGS**：逐 token 根据奖励信号选择输出

**CBS**：在 chunk 级别进行束搜索

这些方法的核心问题：

- **依赖基础策略质量**：当基础策略较弱时，生成的候选集中很可能没有高质量响应
- **指数级采样需求**：若基础策略生成最优响应的概率为 $\sigma$，BoN 至少包含一个最优响应的概率为 $1-(1-\sigma)^N$，当 $\sigma$ 很小时需要指数级增长的 N
- **受限于离散搜索空间**：无法利用奖励模型的梯度信息来主动"移动"到高奖励区域

## 方法详解

### 整体框架

SEA 的核心思路：不在离散响应空间中搜索，而是在连续 logit 空间中，沿着奖励梯度方向迭代优化初始响应。

**三步流程**：
1. 从基础策略 $\pi_{\text{ref}}$ 生成初始响应，获取其连续 logit 表示
2. 定义基于最优 RLHF 策略的能量函数
3. 通过 Langevin 动力学在连续空间中迭代优化 logit，最后解码为离散文本

### 关键设计

**能量函数定义**：

RLHF 的最优策略具有如下闭式解：

$$\pi^*(y|x) = \frac{1}{Z(x)} \exp(E(x, y))$$

其中能量函数为：

$$E(x, y) = \log \pi_{\text{ref}}(y|x) + \alpha \cdot r(x, y)$$

$\alpha$ 控制奖励与 KL 惩罚之间的权衡，$Z(x)$ 为配分函数。

**Langevin MCMC 采样**：

由于直接从最优策略采样需要计算不可行的配分函数，SEA 使用梯度信息进行 Langevin 采样：

$$y^{(n+1)} \leftarrow y^{(n)} - \eta \nabla_y E(x, y^{(n)}) + \epsilon^{(n)}$$

其中 $\epsilon^{(n)} \sim \mathcal{N}(0, I)$ 为高斯噪声，$\eta$ 为步长。

关键在于：$\nabla_y \log \pi^*(y|x) = -\nabla_y E(x, y)$（配分函数对 y 的梯度为零），因此无需计算 $Z(x)$。

**连续化处理**：

- 离散 token 序列不可微分，SEA 使用 LLM 的连续 logit 作为 $y$ 的表示
- 采用 Straight-Through Estimator：前向传播用 argmax（离散），反向传播用 softmax（连续）
- 直接将连续 logit 作为输入 token 送入参考模型和奖励模型

**多初始化策略**：

同时运行多个 Langevin chain（默认 4 个），每个从基础策略的不同采样初始化，最终选择奖励最高的响应。

### 损失函数

SEA 是推理时方法，不涉及训练。其优化目标为最小化能量函数：

$$\min_y E(x, y) = -[\log \pi_{\text{ref}}(y|x) + \alpha \cdot r(x, y)]$$

等价于在保持与参考策略接近的同时最大化奖励。

## 实验关键数据

### 主实验

**AdvBench 安全性评估**（Harmful Rate ↓）：

| 方法 | LLaMA-3.2-1B | LLaMA-3.2-3B | LLaMA-3-8B | LLaMA-3.2-1B-Instruct |
|------|-------------|-------------|-----------|----------------------|
| SFT | 65.96% | 50.77% | 14.42% | 0.77% |
| BoN-64 | 43.85% | 28.27% | 8.85% | 0.77% |
| ARGS | 25.96% | 22.50% | 8.27% | 0.19% |
| CBS | 24.81% | 23.65% | 6.35% | 0.96% |
| **SEA** | **5.58%** | **6.92%** | **3.85%** | **0.19%** |

SEA 在 LLaMA-3.2-1B-Base 上相对 SFT 降低 91.54% 的有害率，远超 BoN-64。

**TruthfulQA 真实性评估**（TR ↑）：

| 方法 | LLaMA-3.2-1B | LLaMA-3.2-3B | LLaMA-3-8B | LLaMA-3.2-1B-Instruct |
|------|-------------|-------------|-----------|----------------------|
| SFT | 59.0% | 64.0% | 62.0% | 72.0% |
| BoN-64 | 78.0% | 74.0% | 72.0% | 77.0% |
| **SEA** | **78.0%** | **80.0%** | **76.0%** | **89.0%** |

SEA 在提升真实性的同时保持了信息量和多样性（Diversity 也是最高）。

**数学推理评估**（LLaMA-3.2-1B-Instruct）：

| 方法 | GSM8K Acc | MATH Acc |
|------|----------|---------|
| SFT | 32.00% | 27.50% |
| BoN-64 | 57.00% | 16.00% |
| **SEA** | **58.00%** | **32.00%** |

搜索方法在 MATH 上甚至不如 SFT（BoN-64 仅 16%），而 SEA 将准确率提升至 32%。

### 消融实验

**LLaMA-3.2-1B-Base 上的消融**（AdvBench HR ↓）：

| 变体 | HR (%) |
|------|--------|
| SEA (4 chains) | 5.58 |
| SEA (1 chain) | 13.65 |
| w/ Random Init | 4.04 |
| w/o Reward | 19.62 |
| w/o Reference | 12.69 |
| w/o Noise | 6.73 |

- 即使单链也比 SFT（65.96%）大幅提升
- 随机初始化在安全任务上甚至更好（因为原始响应本身很有害，从中出发更难优化）
- 去掉奖励模型后仍有提升，因为 Langevin 动力学的随机游走本身扩展了搜索空间

### 关键发现

1. **Deep Alignment**：SEA 的 KL 预算均匀分布在所有 token 位置上，不像传统方法集中在前几个 token（浅层对齐）
2. **抗 Prefilling Attack**：在 LLaMA-3.2-1B-Instruct 上，即使注入 7 个有害 prefix token，SEA 的攻击成功率始终为 0%，而 BoN-32 被完全攻破
3. **奖励快速收敛**：约 30 步迭代后奖励趋于稳定，响应质量达到高水平

## 亮点与洞察

1. **范式创新**：从"离散搜索"到"连续优化"的转变简洁有力，天然利用了奖励模型的梯度信息
2. **理论优美**：将 RLHF 最优策略建模为 EBM，利用 Langevin 动力学采样，理论基础扎实
3. **深度对齐**：SEA 在所有 token 位置同时进行对齐，天然避免了浅层对齐的脆弱性
4. **即插即用**：不需要修改模型参数，可应用于任何基础 LLM + 奖励模型的组合
5. **弱模型也能用**：不受基础策略能力限制——即使基础模型很弱，梯度优化仍能找到高奖励区域

## 局限性

1. **计算开销**：需要对奖励模型和参考模型进行反向传播计算梯度，比 BoN 更消耗 GPU 显存
2. **Straight-Through Estimator 的近似误差**：连续 logit 近似离散 token 可能引入偏差
3. **奖励模型依赖**：性能上限受奖励模型质量限制，奖励黑客风险仍然存在
4. **仅在 LLaMA-3 系列上验证**，GPT/Claude 等闭源模型无法直接使用（需要梯度访问）
5. **多初始化增加成本**：默认 4 chains 意味着 4 倍的计算资源

## 相关工作与启发

- **Best-of-N / ARGS / CBS**：离散搜索方法，性能受基础策略和候选集限制
- **COLD (Qin et al., 2022)**：在词汇空间中用梯度采样进行可控生成，SEA 将此思路引入对齐场景
- **MuCoCO / MuCoLa**：在中间表示上优化实现可控推理的先驱工作
- **COLD-Attack**：利用能量函数进行越狱攻击，SEA 方向相反——用于防御和对齐
- **启发**：连续优化方法在推理时对齐中严重未被探索，SEA 打开了这一方向；未来可探索更高效的采样器（如 HMC）或自适应步长策略

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 离散→连续的范式转换是推理时对齐的全新方向
- **技术深度**: ⭐⭐⭐⭐ — EBM + Langevin dynamics 理论清晰，但连续化近似有局限
- **实验充分性**: ⭐⭐⭐⭐⭐ — 4 种模型、3 大任务、7 个基线、丰富的消融和深度分析
- **实用价值**: ⭐⭐⭐⭐ — 即插即用形式有实用潜力，但计算开销限制了大规模部署
- **写作质量**: ⭐⭐⭐⭐ — 动机清晰，方法直观，可视化丰富
- **综合评分**: ⭐⭐⭐⭐⭐ (9/10)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Debiasing Reward Models via Causally Motivated Inference-Time Intervention](../../ACL2026/llm_alignment/debiasing_reward_models_via_causally_motivated_inference-time_intervention.md)
- [\[NeurIPS 2025\] Limited Preference Data? Learning Better Reward Model with Latent Space Synthesis](limited_preference_data_learning_better_reward_model_with_latent_space_synthesis.md)
- [\[ICML 2026\] Reward Shaping for (Inference-Time) Alignment: A Stackelberg Game Perspective](../../ICML2026/llm_alignment/reward_shaping_for_inference-time_alignment_a_stackelberg_game_perspective.md)
- [\[ACL 2026\] To Intervene or Not: Guiding Inference-time Alignment with Probabilistic Model Blending](../../ACL2026/llm_alignment/to_intervene_or_not_guiding_inference-time_alignment_with_probabilistic_model_bl.md)
- [\[AAAI 2026\] W2S-AlignTree: Weak-to-Strong Inference-Time Alignment for Large Language Models via Monte Carlo Tree Search](../../AAAI2026/llm_alignment/w2s-aligntree_weak-to-strong_inference-time_alignment_for_large_language_models_.md)

</div>

<!-- RELATED:END -->
