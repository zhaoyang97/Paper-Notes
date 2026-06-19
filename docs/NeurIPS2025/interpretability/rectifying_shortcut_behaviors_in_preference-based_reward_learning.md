---
title: >-
  [论文解读] Rectifying Shortcut Behaviors in Preference-based Reward Learning
description: >-
  [NeurIPS 2025][可解释性][reward hacking] 提出 PRISM（Preference-based Reward Invariance for Shortcut Mitigation），将 reward hacking 统一建模为 shortcut learning 问题，通过群不变核（group-invariant kernels）和随机特征映射近似来同时缓解多种 spurious correlation（冗长性、谄媚、语气等），在 out-of-distribution 偏好数据和下游策略模型上一致提升表现。
tags:
  - "NeurIPS 2025"
  - "可解释性"
  - "reward hacking"
  - "shortcut learning"
  - "group-invariant kernel"
  - "RLHF"
  - "preference alignment"
---

# Rectifying Shortcut Behaviors in Preference-based Reward Learning

**会议**: NeurIPS 2025  
**arXiv**: [2510.19050](https://arxiv.org/abs/2510.19050)  
**代码**: 待确认  
**领域**: 可解释性  
**关键词**: reward hacking, shortcut learning, group-invariant kernel, RLHF, preference alignment

## 一句话总结

提出 PRISM（Preference-based Reward Invariance for Shortcut Mitigation），将 reward hacking 统一建模为 shortcut learning 问题，通过群不变核（group-invariant kernels）和随机特征映射近似来同时缓解多种 spurious correlation（冗长性、谄媚、语气等），在 out-of-distribution 偏好数据和下游策略模型上一致提升表现。

## 研究背景与动机

RLHF 中 reward model 面临的核心问题是 **reward hacking** — 模型利用训练数据中的虚假相关特征（如更长的回答、讨好用户的语气、谄媚态度）而非真正与人类意图对齐的特征来获取高分。具体表现：

- **冗长性（Verbosity）**：reward model 偏好更长回答，不管内容质量
- **谄媚（Sycophancy）**：偏好顺从用户观点的回答，即使用户错误
- **概念关联（Concept correlation）**：将无关文本概念（如"food"）与目标属性（如正面情感）错误关联

prior work 的不足：

**逐个处理**：如 ODIN 只处理 verbosity，RRM 只处理长度偏差 — 无法同时处理多种 shortcuts

**需要属性标注**：multi-objective 方法需要细粒度标注（如 HelpSteer），实际中难以获取

**缺乏统一理论**：不同 bias 被视为独立问题，没有统一框架

本文核心洞察：**reward hacking 本质上就是 shortcut learning** — 与分类任务中模型利用背景、纹理等虚假特征的问题本质相同，可以借鉴不变性理论来统一处理。

## 方法详解

### 整体框架

PRISM 的核心思路（如 Figure 2 所示）：

1. **建模 shortcuts 为群作用**：将冗长性、谄媚等视为群 $\mathcal{G}$ 对回答 $y$ 的变换（如增减长度、改变语气）
2. **学习群不变核**：确保 reward 对这些变换不变
3. **用随机特征映射近似核**：使得 reward margin 不依赖 spurious 特征
4. **修正 BT loss**：从 reward margin 中减去 shortcut 的核距离，迫使模型关注 generalizable 特征

### 关键设计

**群不变核（Definition 1）**：

对 response space $\mathcal{Y}$ 上的核 $\kappa$，定义群不变核：

$$\mathcal{K}(y_w, y_l | x) = \int_{g \in \mathcal{G}} \int_{g' \in \mathcal{G}} \kappa(gy_w, g'y_l | x) \, d\mu(g) \, d\mu(g')$$

满足 $\mathcal{K}(gy_w, g'y_l | x) = \mathcal{K}(y_w, y_l | x)$，即对任意群变换不变。

**随机特征映射近似（Proposition 1）**：

由于直接计算 Haar 积分不可行，用随机特征映射 $\Phi$ 近似：

$$\Phi(y) = \left[\phi\left(y, t_j, \frac{sk}{n}\right)\right]_{j=1\dots m, k=-n\dots n} \in \mathbb{R}^{(2n+1) \times m}$$

其中 $\phi$ 是归一化的经验 CDF。当 bin 数 $n \to \infty$ 时，$\langle \Phi(y_w), \Phi(y_l) \rangle \to \mathcal{K}_s(y_w, y_l | x)$。

**轨道间距离（Theorem 1）**：特征映射的内积可准确反映两个回答在 shortcut 空间中的轨道距离 $d_{\mathcal{G}}$。

**实际核函数**：对 $m$ 种 shortcut，用 RBF 核的凸组合：

$$\mathcal{K}_{\text{inv}} = \sum_{j=1}^m \alpha_j \exp\left(-\frac{\|\Phi_j(y_w, x) - \Phi_j(y_l, x)\|^2}{\omega_j^2}\right)$$

**PRISM 完整损失函数**：

$$\mathcal{L}_{\text{PRISM}}(\theta) = -\frac{1}{N}\sum_{i=1}^N \log\sigma\left(\Delta_{r_\theta}(y_w, y_l | x) - \lambda_1 \mathcal{K}_{\text{inv}}(y_w, y_l | x)\right) + \lambda_2 \mathcal{R}_{\text{global}}(\theta)$$

- 第一项：从 reward margin 中**减去** shortcut 核值，迫使模型不依赖 spurious 差异
- 第二项：全局去相关正则，惩罚 reward 与 shortcut 特征的 batch-level 相关:

$$\mathcal{R}_{\text{global}}(\theta) = \sum_{j=1}^m \left(\frac{\text{Cov}_{\mathcal{B}}(r_\theta, \Phi_j)}{\sigma_{\mathcal{B},r_\theta} \cdot \sigma_{\mathcal{B},\Phi_j}}\right)^2$$

### Shortcut 特征提取

- **Rule-based**：长度（字符数）、词汇多样性（TTR = unique tokens / total tokens）
- **LLM-as-Judge**：通过 GPT-4o + LangChain API 提取谄媚度、创造性、帮助性（0-10 分）
- LRU 缓存(10K entries) + 批量并行处理 + 启发式 fallback

### 损失函数 / 训练策略

- $\lambda_1, \lambda_2$ 采用课程学习：线性从 0.01 增至 0.1（前半程），再降至 0.06（后半程）
- 学习率 $2 \times 10^{-6}$，cosine annealing + 3% warmup
- 权重 $\alpha_j$ 通过 learnable softmax 层学习
- 硬件：8 × NVIDIA A6000
- 实现基于 HuggingFace + DeepSpeed

## 实验关键数据

### 主实验

**RewardBench**：

| 方法 | Base Model | Chat | Chat Hard | Safety | Reasoning | Score |
|------|-----------|------|-----------|--------|-----------|-------|
| Bradley-Terry | Llama-3 8B | 99.4 | 65.1 | 87.8 | 86.4 | 83.6 |
| RLHFlow | Llama-3 8B | 99.4 | 65.1 | 87.8 | 86.4 | 84.7 |
| GRM | Llama-3 8B | 98.6 | 67.8 | 89.4 | 92.3 | 87.0 |
| **PRISM** | Llama-3 8B | **98.7** | **68.3** | **91.1** | **93.1** | **87.8** |

PRISM 在三个困难类别（Chat Hard、Safety、Reasoning）上都有提升，总分 87.8 为最佳。

**RM-Bench（更难的 benchmark）**：

| 方法 | Chat | Math | Code | Safety | Easy | Normal | Hard | Avg |
|------|------|------|------|--------|------|--------|------|-----|
| Skywork-8B | 69.5 | 60.6 | 54.5 | 95.7 | 89.0 | 74.7 | 46.6 | 70.1 |
| URM-8B | 71.2 | 61.8 | 54.1 | 93.1 | 84.0 | 73.2 | 53.0 | 70.0 |
| **PRISM (8B)** | 70.6 | **70.8** | **57.0** | 94.1 | **90.6** | **76.3** | 46.9 | **71.0** |

在 Math (+9.0) 和 Code (+2.9) 上的提升尤为显著，因为这些类别中 shortcut 更难利用。

### 消融实验

**下游策略模型评估（AlpacaEval-2）**：

使用不同 reward model 训练 Gemma-9B 策略模型：
- PRISM 诱导的策略模型 win rate 更高，response length 适中
- BT、RRM、ODIN baseline 的策略模型或 win rate 低，或 response 过长

**Shortcut 相关性分析（Figure 4）**：

| Shortcut | BT (PCC) | PRISM (PCC) |
|----------|----------|-------------|
| Response Length | 强正相关 | ≈ 0 |
| Tone | 非平凡相关 | ≈ 0 |
| Sycophancy | 非平凡相关 | ≈ 0 |

PRISM 在所有三个 shortcut 维度上都实现了近零相关系数，直接证明了 shortcut 缓解的有效性。

### 关键发现

1. 多 shortcut 联合处理优于逐个处理 — RewardBench 上的一致提升来自多维度的同时正则化
2. PRISM 学到的 $\alpha_j$ 权重揭示了不同 shortcut 的相对重要性
3. 课程学习的 $\lambda$ 调度避免了过早惩罚导致的 under-fitting
4. Rule-based 和 LLM-Judge 特征互补 — 前者快速覆盖简单 shortcut，后者处理语义层面的 bias

## 亮点与洞察

1. **统一框架的优雅性**：将 reward hacking 的多种表现（冗长性、谄媚、概念关联等）统一为一个 shortcut learning 框架，借鉴不变性理论解决
2. **理论保证（Theorem 2）**：给出了 generalization bound，随着 shortcut 特征数 $m$、bin 数 $n$、群元素数 $|\mathcal{G}|$、训练样本 $N$ 增加，经验风险趋近最优风险
3. **灵活的特征接口**：支持从简单的长度计数到复杂的 LLM-as-Judge，易于扩展新 shortcut
4. **近零相关的直观验证**：Figure 4 的 PCC 分析一目了然，说服力极强
5. **不需要属性标注**：与 HelpSteer2 RM 等需要细粒度标注的方法不同，PRISM 无需任何人工属性标注

## 局限与展望

1. **GPT-4o API 成本**：LLM-as-Judge 特征提取需要大量 API 调用，增加了训练成本
2. **特征选择依赖先验**：需要预先指定哪些 shortcut 需要缓解，未覆盖的 shortcut 仍可能被利用
3. **核函数选择**：仅使用 RBF 核，未探讨其他核函数（如多项式核、Matern 核）的效果
4. **评估偏差**：RewardBench 和 RM-Bench 本身可能有未知 bias，影响评估的公正性
5. **可扩展性**：随着 shortcut 类型增多，$\alpha_j$ 的学习和特征计算复杂度可能成为瓶颈
6. **理论-实践 gap**：Theorem 2 基于 RKHS 的假设在 LLM-based reward model 上的适用性有待验证

## 相关工作与启发

- **ODIN (Chen et al.)**: 专门针对 verbosity bias 添加长度惩罚，是单 shortcut 方法的代表
- **RRM (Liu et al.)**: 通过正则化减少 reward model 对长度的依赖，但仅处理长度
- **GRM (Yang et al.)**: 通过正则化改进 reward model 泛化，PRISM 在其基础上进一步提升
- **Invariant Risk Minimization (Arjovsky et al.)**: 不变性理论的经典工作，PRISM 将其核心思想引入 RLHF
- **Shortcut Learning (Geirhos et al.)**: 分类任务中 shortcut 问题的系统综述，本文将其推广到 preference learning

**启发**：群不变核的框架具有广泛适用性，可推广到 DPO 等直接偏好学习算法、多模态 reward model、以及 code generation 中的逻辑正确性 vs 表面格式等 shortcut 问题。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 将 reward hacking 统一为 shortcut learning 的视角极具开创性，群不变核理论的引入令人耳目一新
- 实验充分度: ⭐⭐⭐⭐ 多 benchmark 验证 + 下游策略评估 + 相关性分析，较为全面
- 写作质量: ⭐⭐⭐⭐ 理论推导完整，方法动机清晰，但部分核数学细节可能对读者门槛较高
- 价值: ⭐⭐⭐⭐⭐ 解决了 RLHF 中最核心的 reward hacking 问题，框架通用性强，对实际对齐工作直接有用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Probing Preference Representations: A Multi-Dimensional Evaluation and Analysis Method for Reward Models](../../AAAI2026/interpretability/probing_preference_representations_a_multi-dimensional_evaluation_and_analysis_m.md)
- [\[NeurIPS 2025\] Understanding Prompt Tuning and In-Context Learning via Meta-Learning](understanding_prompt_tuning_and_in-context_learning_via_meta-learning.md)
- [\[ACL 2025\] Establishing Trustworthy LLM Evaluation via Shortcut Neuron Analysis](../../ACL2025/interpretability/shortcut_neuron_eval.md)
- [\[ICML 2025\] Configurable Preference Tuning with Rubric-Guided Synthetic Data](../../ICML2025/interpretability/configurable_preference_tuning_with_rubric-guided_synthetic_data.md)
- [\[ACL 2026\] Learning What Matters: Dynamic Dimension Selection and Aggregation for Interpretable Vision-Language Reward Modeling](../../ACL2026/interpretability/learning_what_matters_dynamic_dimension_selection_and_aggregation_for_interpreta.md)

</div>

<!-- RELATED:END -->
