---
title: >-
  [论文解读] Constrained Discrete Diffusion
description: >-
  [NeurIPS 2025][计算生物][离散扩散模型] 提出 CDD（Constrained Discrete Diffusion），将可微约束优化投影算子嵌入离散扩散模型的去噪过程中，无需重训练即可在采样时强制满足序列级约束，在毒性文本生成、分子设计和指令遵循三类任务上实现零约束违反。 离散扩散模型（如 MDLM、UDL…
tags:
  - "NeurIPS 2025"
  - "计算生物"
  - "离散扩散模型"
  - "约束优化"
  - "增广拉格朗日"
  - "可控文本生成"
  - "分子生成"
  - "毒性缓解"
---

# Constrained Discrete Diffusion

**会议**: NeurIPS 2025  
**arXiv**: [2503.09790](https://arxiv.org/abs/2503.09790)  
**代码**: 待确认  
**领域**: 计算生物  
**关键词**: 离散扩散模型, 约束优化, 增广拉格朗日, 可控文本生成, 分子生成, 毒性缓解

## 一句话总结

提出 CDD（Constrained Discrete Diffusion），将可微约束优化投影算子嵌入离散扩散模型的去噪过程中，无需重训练即可在采样时强制满足序列级约束，在毒性文本生成、分子设计和指令遵循三类任务上实现零约束违反。

## 研究背景与动机

离散扩散模型（如 MDLM、UDLM）已展现出强大的文本和分子序列生成能力，但在实际应用中需要满足各种约束——毒性阈值、化学合成可行性规则、指令遵循要求等。

**自回归模型的根本困难**：逐 token 生成使得序列级约束难以强制执行。现有方案（RLHF、拒绝采样、后处理）均为**软约束**，无法提供可证明的合规保证。

**离散扩散的独特机会**：每步去噪暴露**完整序列的全局视图**，天然适合施加序列级结构约束。然而，直接应用连续扩散中的欧几里得投影到离散概率单纯形上是不合适的。

CDD 的核心创新：设计一个基于 **KL 散度**（而非欧几里得距离）的投影算子，通过**增广拉格朗日方法**在每个去噪步骤中求解约束优化子问题，将生成分布投影到可行域上。

## 方法详解

### 整体框架

CDD 的流程：
1. 标准离散扩散的初始化（全 [MASK] 或均匀分布）
2. 每个去噪步骤后，执行**投影操作**：
    - 输入：去噪器输出的概率分布 $\bm{x}_t'$
    - 输出：满足约束的投影分布 $\bm{x}_s$
3. 使用投影后的分布继续下一步去噪

关键特性：**无需训练**——投影仅在采样时执行，不修改模型权重。

### 关键设计

**KL 投影算子**：

$$\bm{x}_s = \mathcal{P}_{\mathbf{C}}(\bm{x}_t') = \arg\min_{\bm{y}} D_{\text{KL}}(\bm{x}_t' \| \bm{y}) \quad \text{s.t.} \quad \arg\max(\bm{y}) \in \mathbf{C}$$

在概率分布空间中，KL 散度比欧几里得距离更加自然——它保证了投影后的分布是约束可行域中与原分布"最近"的分布。

**Gumbel-Softmax 可微化**：$\arg\max$ 不可微，使用 Gumbel-Softmax 松弛近似：

$$\tilde{\phi}(\bm{x}_t^i)(v) = \frac{\exp\left(\frac{\log \bm{x}_t^i(v) + \xi_v}{T_{\text{sample}}}\right)}{\sum_{v'=1}^{N} \exp\left(\frac{\log \bm{x}_t^i(v') + \xi_{v'}}{T_{\text{sample}}}\right)}$$

其中 $\xi_v$ 为 Gumbel(0,1) 噪声，$T_{\text{sample}}$ 控制逼近 $\arg\max$ 的程度。

**增广拉格朗日投影**：约束违反量定义为 $\Delta g_i(\tilde{\phi}(\bm{x}_t)) = \max(0, g_i(\tilde{\phi}(\bm{x}_t)) - \tau_i)$。增广拉格朗日目标：

$$\mathcal{L}_{\text{ALM}}(\bm{y}, \bm{x}_t'; \lambda, \mu) = D_{\text{KL}}(\bm{x}_t' \| \bm{y}) + \sum_{i=1}^{m} \lambda_i \Delta g_i(\tilde{\phi}(\bm{y})) + \frac{\mu_i}{2} \Delta g_i(\tilde{\phi}(\bm{y}))^2$$

迭代求解：
- 梯度更新：$\bm{y} \leftarrow \bm{y} - \eta \nabla_{\bm{y}} \mathcal{L}_{\text{ALM}}$
- 乘子更新：$\lambda \leftarrow \lambda + \mu \Delta g(\bm{y}^*)$
- 罚项递增：$\mu \leftarrow \min(\alpha \mu, \mu_{\max})$

### 损失函数

CDD 本身**不引入训练损失**——它是纯采样时技术。底层扩散模型使用标准的 MDLM/UDLM 去噪目标训练。投影使用的增广拉格朗日是一个采样时的在线优化目标。

**收敛性保证**（Theorem 4.1）：在约束集 $\bm{C}$ 满足 $\beta$-prox-regularity 条件下，投影后与可行域的 KL 距离以 $(1 - \bm{\alpha}_t)$ 的速率衰减：

$$D_{\text{KL}}(\bm{x}_s', \bm{C}) \leq (1 - \bm{\alpha}_t) D_{\text{KL}}(\bm{x}_t', \bm{C}) + \bm{\alpha}_{t+1}^2 G^2$$

在 $\mathcal{O}(\bm{\alpha}_{\min}^{-1})$ 步后达到 $\epsilon$-可行性。

## 实验关键数据

### 主实验

**毒性文本生成**（RealToxicityPrompts，1000 样本）：

| 模型 | 参数量 | PPL↓ | Coherence↑ | 违反率 ($\tau$=0.25)↓ | 违反率 ($\tau$=0.50)↓ | 违反率 ($\tau$=0.75)↓ |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| GPT-2 | 124M | 18.78 | 42.68 | 33.2% | 21.6% | 13.1% |
| GPT-2+PPLM | 124M | 46.40 | 18.88 | 16.1% | 8.4% | 4.0% |
| Llama 3.2 | 1B | 15.66 | 57.10 | 34.9% | 27.8% | 23.1% |
| MDLM | 110M | 46.72 | 20.02 | 32.1% | 23.2% | 17.2% |
| **CDD ($\tau$=0.25)** | **110M** | **61.55** | **20.16** | **0.0%** | **0.0%** | **0.0%** |
| **CDD ($\tau$=0.50)** | **110M** | **59.44** | **20.30** | — | **0.0%** | **0.0%** |
| **CDD ($\tau$=0.75)** | **110M** | **54.87** | **20.88** | — | — | **0.0%** |

所有 CDD 配置在对应阈值下实现 **0% 违反率**，PPL 与 MDLM 基线相当，coherence 有所提升。

**分子生成（合成可行性约束）**：

| 模型 | 有效分子↑ | 新颖分子↑ | QED↑ | 违反率 ($\tau$=3.0)↓ |
|:--|:--:|:--:|:--:|:--:|
| AR | 1023 | 0 | 0.46 | 91.6% |
| UDLM | 895 | 21 | 0.47 | 89.4% |
| UDLM+D-CFG | 850 | 18 | 0.47 | 80.6% |
| **CDD ($\tau$=3.0)** | **353** | **36** | **0.63** | **0.0%** |
| **CDD ($\tau$=4.5)** | **938** | **33** | **0.58** | **0.0%** |

### 消融实验

**分子新颖性约束**：

| 模型 | Valid & Novel↑ | QED↑ | 违反率↓ |
|:--|:--:|:--:|:--:|
| MDLM | 271 | 0.45 | 54.53% |
| UDLM | 345 | 0.46 | 61.45% |
| **CDD** | **511** | **0.45** | **0.0%** |

CDD 将有效且新颖的分子数从 345 提升至 511（+48%），同时违反率从 61.45% 降至 0%。

增广拉格朗日超参数敏感性分析：在全参数网格搜索范围内，拉格朗日松弛均收敛到可行解——方法对超参数高度鲁棒。

### 关键发现

1. CDD 在所有三类任务的所有配置中均实现**零约束违反**，这是所有基线（包括 10× 更大的 Llama 3.2）无法达到的
2. 约束满足不需要牺牲生成质量：PPL 基本保持，coherence 甚至略有提升
3. 投影算子通过 KL 散度保持与原始去噪分布的接近，保留了生成多样性
4. 方法对约束函数的形式高度灵活——可以是分类器（毒性检测）、黑盒函数（合成可行性）或符号规则（指令遵循）

## 亮点与洞察

- ⭐ 首次在离散扩散框架中实现可证明的硬约束满足，填补了重要空白
- ⭐ training-free 的设计使方法可以即插即用于任何预训练的离散扩散模型
- 用 KL 散度替代欧几里得距离做投影，是对连续扩散约束方法的正确离散化推广
- 增广拉格朗日方法避免了拒绝采样的低效和后处理的不可靠

## 局限性

- CDD 在每个去噪步骤中运行拉格朗日内循环，推理速度较慢（约 2-3× 额外开销）
- Gumbel-Softmax 只是 $\arg\max$ 的近似，理论收敛保证依赖于 prox-regularity 假设
- 约束函数 $g_i$ 需要可微或有可微代理模型，纯符号约束需要额外建模
- 实验中的扩散模型规模较小（110M / 92M），在更大模型上的行为尚未验证

## 相关工作与启发

CDD 连接了约束优化和生成模型两个领域。与连续扩散的约束方法（如 MPGD）相比，CDD 的 KL 投影更适合离散概率单纯形。与 RLHF 等对齐方法相比，CDD 提供**硬保证**而非软引导。未来方向包括将投影算子扩展到自回归解码（通过 speculative decoding 类似的机制），以及支持更复杂的组合约束。

## 评分

⭐⭐⭐⭐⭐ (5/5)

| 维度 | 评分 |
|:--|:--:|
| 新颖性 | ⭐⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐⭐ |

问题定义精准，技术方案优雅，实验覆盖三类不同领域且均达到零违反。是离散扩散模型可控生成方向的重要里程碑。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Split Gibbs Discrete Diffusion Posterior Sampling](split_gibbs_discrete_diffusion_posterior_sampling.md)
- [\[NeurIPS 2025\] Why Masking Diffusion Works: Condition on the Jump Schedule for Improved Discrete Diffusion](why_masking_diffusion_works_condition_on_the_jump_schedule_for_improved_discrete.md)
- [\[NeurIPS 2025\] Remasking Discrete Diffusion Models with Inference-Time Scaling](remasking_discrete_diffusion_models_with_inference-time_scaling.md)
- [\[ICML 2025\] GenMol: A Drug Discovery Generalist with Discrete Diffusion](../../ICML2025/computational_biology/genmol_a_drug_discovery_generalist_with_discrete_diffusion.md)
- [\[ICLR 2026\] Discrete Diffusion Trajectory Alignment via Stepwise Decomposition](../../ICLR2026/computational_biology/discrete_diffusion_trajectory_alignment_via_stepwise_decomposition.md)

</div>

<!-- RELATED:END -->
