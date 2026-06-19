---
title: >-
  [论文解读] Spend Wisely: Maximizing Post-Training Gains in Iterative Synthetic Data Bootstrapping
description: >-
  [NeurIPS 2025 Spotlight][自监督学习][合成数据] 首次从理论上分析了迭代合成数据自举训练中的预算分配问题，证明恒定策略无法高概率收敛，而指数增长策略在最坏情况下优于多项式策略，并在图像去噪（DPM）和数学推理（LLM）实验中验证了该结论。 问题背景 现代基础模型在后训练阶段常采用迭代"自举"范式：模…
tags:
  - "NeurIPS 2025 Spotlight"
  - "自监督学习"
  - "合成数据"
  - "迭代自举"
  - "预算分配策略"
  - "指数增长策略"
  - "扩散模型"
  - "大语言模型"
  - "后训练优化"
---

# Spend Wisely: Maximizing Post-Training Gains in Iterative Synthetic Data Bootstrapping

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2501.18962](https://arxiv.org/abs/2501.18962)  
**作者**: Pu Yang (Peking University), Yunzhen Feng (NYU), Ziyuan Chen (Peking University), Yuhang Wu (UC Berkeley), Zhuoyuan Li (NUS)  
**代码**: 未公开  
**领域**: 图像复原  
**关键词**: 合成数据, 迭代自举, 预算分配策略, 指数增长策略, 扩散模型, 大语言模型, 后训练优化  

## 一句话总结

首次从理论上分析了迭代合成数据自举训练中的预算分配问题，证明恒定策略无法高概率收敛，而指数增长策略在最坏情况下优于多项式策略，并在图像去噪（DPM）和数学推理（LLM）实验中验证了该结论。

## 研究背景与动机

### 问题背景
现代基础模型在后训练阶段常采用迭代"自举"范式：模型生成合成数据→外部验证器过滤低质量样本→高质量子集用于进一步微调。这个过程重复多轮后模型性能持续提升。该范式广泛应用于LLM的推理能力提升（STaR等）、指令跟随、对齐训练，以及视觉领域的图像分割和去噪等任务。

### 已有工作的不足
- 现有实践中，每轮迭代生成的合成数据量通常保持**固定不变**（恒定策略），缺乏理论指导
- Ferbach et al. (2024) 提供了唯一的理论分析，但仅在**无限样本假设**下证明收敛，无法给出有限样本下的预算分配指导
- 模型坍塌（model collapse）研究关注合成数据的风险，但未涉及如何在迭代中最优地分配生成预算
- 对于实践者而言，如何在有限计算预算下决定每轮生成和选择多少合成数据仍是一个开放问题

### 核心动机
回答一个关键问题：在固定总计算预算下，如何跨迭代分配合成数据的生成量和训练量，以最大化最终模型性能？

## 方法详解

### 问题建模
- **生成模型**：参数化模型 $f(\cdot;\theta)$ 生成数据 $x \sim \mathcal{P}_\theta$
- **奖励模型**：$\mathcal{R}(x) \in [0,1]$ 评估生成数据质量
- **目标**：最大化期望奖励 $r(\theta) = \mathbb{E}_{x \sim \mathcal{P}_\theta}[\mathcal{R}(x)]$
- **迭代过程**：每轮 $t$ 生成 $N_t$ 个样本，按概率 $\mathcal{R}(x)$ 选择 $n_t$ 个高质量样本，用于梯度下降更新模型
- **策略定义**：$\pi: \{n_t\}_{t=0,1,\cdots}$ 控制每轮选择的数据量
- **总计算成本**：$C(\pi, T) = \sum_{t=0}^{T-1} c_g N_t + c_t n_t$，其中 $c_g$ 和 $c_t$ 分别为生成和训练的单样本成本

### 高斯案例分析（热身）
在高斯生成器 $\mathcal{P}_\theta = \mathcal{N}(\theta, \sigma^2)$ 和指数奖励 $\mathcal{R}(x) = \exp(-x^2/(2\kappa^2))$ 设定下，使用MLE更新，可解析求解最优策略：

$$n_t \propto \left(1 + \frac{\sigma^2}{\kappa^2}\right)^t$$

**Theorem 3.1** 证明最优策略是指数增长——直觉上，早期更新对最终误差的贡献随总迭代次数 $T$ 指数衰减，因此后期迭代需要指数级更多的样本来保证精度。

### 一般设定下的主要理论结果

**Theorem 4.1（恒定策略不收敛）**：对任意恒定策略 $\pi_{\text{const}}$（$n_t = n_0$），以至少 $1/4$ 的概率，
$$r^* - r(\theta_{\pi_{\text{const}}}^{(T)}) \geq c \cdot n_0^{-1/2}$$
恒定策略下梯度中的随机噪声项正比于 $n_t^{-1}$，保持不衰减，导致性能存在不可消除的下界。

**Theorem 4.2（递增策略收敛）**：对任意递增策略 $\pi$（$n_t$ 单调递增），对任意 $\varepsilon > 0$，存在足够大的 $T$，使得以高概率 $r^* - r(\theta_\pi^{(T)}) \leq \varepsilon$。

**Theorem 4.3（指数策略指数收敛）**：存在指数增长策略 $\pi_{\text{exp}}^*$，以高概率满足：
$$r^* - r(\theta_{\pi_{\text{exp}}^*}^{(T)}) = \mathcal{O}((1+\zeta)^{-2T})$$

**Theorem 4.4（指数策略最坏情况最优）**：对任意恒定或多项式增长策略 $\pi$，在最坏情况下，指数策略达到同等性能所需的总计算成本不超过 $\pi$ 的成本。

### 核心结论层级
1. 恒定策略 → 无法收敛到最优奖励
2. 多项式增长策略 → 可收敛但速率较慢
3. **指数增长策略 → 指数收敛且最坏情况最优**

## 实验关键数据

### 实验1：图像去噪（扩散模型）

使用预训练DDPM在MNIST数据集上微调进行去噪，PSNR作为奖励信号，训练批大小 $B=640$。

| 策略 | $n_t$ 设定 | s=10 PSNR | s=20 PSNR |
|------|-----------|-----------|-----------|
| Pre-trained | N/A | 40.92 | 36.96 |
| constant | $1 \cdot B$ | 42.75 | 38.91 |
| constant | $10 \cdot B$ | 43.15 | 39.45 |
| constant | $100 \cdot B$ | 44.18 | 39.91 |
| linear | $t \cdot B$ | 43.90 | 39.90 |
| linear | $3t \cdot B$ | 44.23 | 40.08 |
| linear | $10t \cdot B$ | 44.64 | 40.26 |
| **exponential** | $1.1^t \cdot B$ / $1.05^t \cdot B$ | **44.84** | **40.14** |

- 指数策略在 $s=10$ 下取得最佳PSNR（44.84），在 $s=20$ 下与最优线性策略相当
- 恒定策略初期与指数策略匹配，但很快达到性能平台并因过拟合而衰退
- 线性策略整体排名第二

### 实验2：数学推理（LLM）

使用Llama-3-8B-Base进行全参数微调，在GSM-Symbolic三个难度的数据集上评估，训练批大小 $B=256$。

| 策略 | $n_t$ 设定 | GSM_symbolic | GSM_p1 | GSM_p2 |
|------|-----------|-------------|--------|--------|
| Pre-trained | N/A | 55.60 | 34.98 | 15.72 |
| constant | $10 \cdot B$ | 60.28 | 38.50 | 16.93 |
| constant | $30 \cdot B$ | 60.16 | 39.68 | 17.73 |
| constant | $100 \cdot B$ | 64.04 | 41.66 | 18.81 |
| linear | $3t \cdot B$ | 62.54 | 38.88 | 17.01 |
| linear | $10t \cdot B$ | 61.96 | 39.94 | 17.29 |
| linear | $30t \cdot B$ | 64.24 | 40.84 | 19.17 |
| **exponential** | $10 \cdot 2^t \cdot B$ | **65.66** | **47.26** | **20.65** |

- 指数策略在所有三个难度下均取得最佳结果，尤其在GSM_p1上从35%提升至47%（+12.28pp）
- 线性策略整体第二但在大预算下无法匹配指数策略
- 指数策略提升最为稳定，没有恒定策略的过拟合问题

## 亮点

- **首创性理论框架**：首次形式化分析迭代合成数据训练的预算分配问题，建立了完整的收敛性理论层级（恒定→多项式→指数）
- **实用指导价值**：直接给出实践指导——后期迭代应使用指数级更多数据，打破了当前普遍使用恒定策略的范式
- **跨模态验证**：在图像（DPM去噪）和文本（LLM数学推理）两个领域验证了理论预测的一致性
- **严格理论保证**：指数策略不仅收敛速度最快（指数收敛率），在最坏情况分析下的计算成本也不超过任何多项式策略
- **优雅的理论直觉**：早期更新的误差对最终性能的影响随迭代指数衰减，因此后期需要指数级更多样本的结论具有直观说服力

## 局限与展望

- **理论假设较强**：需要损失函数的正则性假设（Assumptions B.1-B.3）和奖励与损失的关联假设，实际场景可能不完全满足
- **仅考虑SFT框架**：未涉及RLHF/DPO等强化学习后训练范式，作者在结论中提到这是未来方向
- **Oracle奖励模型**：图像去噪实验使用真实干净图像计算PSNR作为奖励，实际中通常无法获得
- **计算规模有限**：LLM实验仅在8B模型上验证，未在更大规模模型（70B+）上测试
- **超参数选择**：指数策略需要选择底数 $u$（如1.05、1.1、2），论文虽进行了消融但未给出自适应选择方法
- **未考虑数据多样性**：仅关注数据量的分配，未分析数据多样性随迭代的变化对策略的影响
- **梯度下降假设**：理论分析基于单步梯度下降更新，与实际中多步SGD训练存在差距

## 与相关工作的对比

- **Ferbach et al. (2024)**：唯一已有的理论分析，但仅在无限样本假设下证明收敛，本文首次给出有限样本下的最优预算分配策略
- **STaR (Zelikman et al., 2022)**：经典的迭代合成数据训练方法，使用恒定策略，本文理论和实验均表明恒定策略不是最优选择
- **Model Collapse (Shumailov et al., 2024)**：关注合成数据训练的退化风险，本文通过验证器过滤机制避免退化并研究最优分配
- **Singh et al. (2025), Guan et al. (2025)**：利用搜索算法生成多样推理路径的迭代方法，但未研究预算分配问题
- **El Firdoussi et al. (2025)**：从随机矩阵理论角度分析合成数据潜力，关注点不同但互补

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次形式化并解决迭代合成数据训练的预算分配问题，理论贡献开创性
- 实验充分度: ⭐⭐⭐⭐ — 覆盖图像和文本两个领域，包含充分的消融和对比，但计算规模偏小
- 写作质量: ⭐⭐⭐⭐⭐ — 从高斯热身到一般理论再到实验的递进结构清晰优雅，定理陈述精准
- 价值: ⭐⭐⭐⭐ — 为实践者提供了明确的策略指导，但假设条件限制了直接推广的范围

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Mitra: Mixed Synthetic Priors for Enhancing Tabular Foundation Models](mitra_mixed_synthetic_priors_for_enhancing_tabular_foundation_models.md)
- [\[NeurIPS 2025\] TabArena: A Living Benchmark for Machine Learning on Tabular Data](tabarena_a_living_benchmark_for_machine_learning_on_tabular_data.md)
- [\[ICLR 2026\] Maximizing Asynchronicity in Event-based Neural Networks](../../ICLR2026/self_supervised/maximizing_asynchronicity_in_event-based_neural_networks.md)
- [\[ICLR 2026\] Maximizing Incremental Information Entropy for Contrastive Learning](../../ICLR2026/self_supervised/maximizing_incremental_information_entropy_for_contrastive_learning.md)
- [\[ICML 2025\] Beyond Sensor Data: Foundation Models of Behavioral Data from Wearables Improve Health Predictions](../../ICML2025/self_supervised/beyond_sensor_data_foundation_models_of_behavioral_data_from_wearables_improve_h.md)

</div>

<!-- RELATED:END -->
