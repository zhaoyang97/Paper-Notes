---
title: >-
  [论文解读] LT-Soups: Bridging Head and Tail Classes via Subsampled Model Soups
description: >-
  [NeurIPS 2025][模型压缩][长尾分布] 提出 LT-Soups，一个两阶段模型融合框架，通过在不同不平衡比例的子采样数据上训练多个模型并进行权重平均，在长尾分布的全频谱上实现头部类和尾部类的均衡性能。 领域现状：真实数据集通常呈长尾分布，少数头部类占据主导。CLIP 等视觉语言基础模型通过 PEFT（如 LoR…
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "长尾分布"
  - "模型融合"
  - "CLIP微调"
  - "类别不平衡"
  - "参数高效微调"
---

# LT-Soups: Bridging Head and Tail Classes via Subsampled Model Soups

**会议**: NeurIPS 2025  
**arXiv**: [2511.10683](https://arxiv.org/abs/2511.10683)  
**代码**: [GitHub](https://github.com/Masseeh/LT-Soups)  
**领域**: 模型压缩与高效学习 (Model Compression)  
**关键词**: 长尾分布, 模型融合, CLIP微调, 类别不平衡, 参数高效微调

## 一句话总结

提出 LT-Soups，一个两阶段模型融合框架，通过在不同不平衡比例的子采样数据上训练多个模型并进行权重平均，在长尾分布的全频谱上实现头部类和尾部类的均衡性能。

## 研究背景与动机

**领域现状**：真实数据集通常呈长尾分布，少数头部类占据主导。CLIP 等视觉语言基础模型通过 PEFT（如 LoRA、AdaptFormer）+逻辑调整（LA）损失可达到 SOTA，但 PEFT 在保持尾部类性能的同时牺牲了头部类准确率。

**现有痛点**：(1) PEFT 在尾部密集场景表现好，但在平衡或头部密集分布下退化；(2) 全量微调虽适应性强但易遗忘预训练知识；(3) 传统 Model Soups 在同一不平衡数据上训练，过度偏向头部类。

**核心矛盾**：头部类和尾部类性能之间存在根本性的 trade-off，没有单一方法能在所有不平衡配置下一致表现良好。

**本文目标**：设计一种在不同不平衡比率 $\rho$ 和头尾比率 $\eta$ 下都表现鲁棒的方法。

**切入角度**：引入头尾比率 $\eta = H/T$ 作为不平衡分布的额外刻画维度，在此双轴框架下分析现有方法的局限。

**核心idea**：通过在不同不平衡程度的子集上分别微调模型，然后权重平均，让不同模型"专注"于不平衡频谱的不同区域，聚合后实现头尾类的平衡表示。

## 方法详解

### 整体框架

LT-Soups 分两个阶段：(1) 在多个具有渐进不平衡比率的子采样数据集上分别微调模型，然后进行递归权重平均；(2) 在完整数据集上仅微调分类头，用类平衡损失恢复头部类信息。

### 关键设计

1. **双轴不平衡刻画**：除传统的不平衡比率 $\rho = n_K / n_1$（最少类/最多类样本数之比）外，引入头尾比率：
    $\eta = \frac{H}{T} = \frac{|\{c \mid n_c > \tau\}|}{|\{c \mid n_c \leq \tau\}|}$
   通过在 CIFAR100 上系统变化 $\rho$ 和 $\eta$，揭示了 PEFT 在尾部密集时优越、全量微调在头部密集时更优的现象。

2. **渐进子采样策略**：构建不平衡比率指数增长的子集序列：
    $\{D_{\rho_i} \mid \rho_i = 2^i, \; i \in \{0, 1, 2, \dots, \lceil\log_2(\rho)\rceil\}\}$
   保留前 $N$ 个子集。每个子集训练 $M$ 个 bootstrap 副本以减小方差，共产生 $NM$ 个模型。

3. **递归权重插值融合**：按不平衡度升序排列，递归合并：
    $\theta_n = (1 - \lambda)\theta_n + \lambda\theta_{n-1}$
   其中 $\lambda$ 控制对前序模型（更平衡的子集）的保留程度。相比均匀平均，递归策略在需要大幅适应的数据集上表现更优。

4. **分类器重训练（CR）**：冻结骨干网络的合并表示，在完整数据集上用 LA 损失微调分类头，调整决策边界。PEFT 和传统 Model Soups 不从 CR 受益，而 LT-Soups 因子采样导致头部信息不完整，CR 可有效恢复。

### 损失函数 / 训练策略

- 第一阶段使用 Logit Adjustment（LA）损失：
  $$\ell_{LA}(y, g(\bm{z})) = -\log \frac{\exp(g_y(\bm{z}) + \log\pi_y)}{\sum_{y'} \exp(g_{y'}(\bm{z}) + \log\pi_{y'})}$$
- 每个模型训练使用 EMA（$\mu=0.99$）作为正则化
- 所有第一阶段模型可完全并行训练

## 实验关键数据

### 主实验

在合成和真实长尾数据集上的对比（CLIP 骨干）：

| 方法 | CIFAR100-LT All | CIFAR100-LT Few | Places-LT All | ImageNet-LT All | NIH-CXR-LT All | iNat2018 All |
|------|----------------|----------------|---------------|-----------------|----------------|-------------|
| Linear Probing | 70.0 | 60.4 | 48.8 | 74.2 | 17.5 | 60.4 |
| Full-FT | 79.6 | 69.3 | 46.6 | 73.9 | 38.0 | 76.1 |
| PEFT | 81.3 | 77.1 | 51.5 | 77.0 | 38.5 | 79.1 |
| Model Soups | 82.1 | 73.0 | 49.4 | 76.0 | 38.0 | 76.4 |
| **LT-Soups** | **83.5** | **78.0** | **51.7** | **77.4** | **39.3** | 78.2 |

### 消融实验

TinyImageNet-LT 上子采样策略的消融（不同固定 $\rho$ 的 Soups 对比）：

| 方法 | All | Head | Tail |
|------|-----|------|------|
| Full-FT | 73.2 | 83.4 | 67.7 |
| PEFT | 77.1 | 83.0 | 73.9 |
| Soups-1（最平衡子集） | 71.7 | 74.6 | 70.1 |
| Soups-100（全数据=传统Soups） | 77.6 | 85.9 | 73.0 |
| **LT-Soups（跨频谱融合）** | **78.6** | 85.0 | **75.2** |

分类器重训练（CR）的效果：

| 方法 | All | Head | Tail |
|------|-----|------|------|
| LT-Soups Stage 1 | 78.1 | 84.9 | 74.5 |
| LT-Soups (+ CR) | **78.6** | **85.0** | **75.2** |

### 关键发现

- 不同 $\rho$ 的 Soups 在头尾类上有不同偏好（Soups-8 尾部准确率最高 75.0，Soups-100 头部最高 85.9），LT-Soups 跨频谱融合实现了最佳综合平衡
- LT-Soups 对损失函数选择部分不敏感（CE/CB/LA 均有效），因子采样和模型平均提供了结构性正则化
- 递归融合策略在需要大幅适应的数据集（如 iNaturalist）上显著优于均匀平均（78.2 vs 74.7）
- 5 个基准的平均性能显示，LT-Soups 在 many/medium/few 三个分组上均最优

## 亮点与洞察

- 引入头尾比率 $\eta$ 作为不平衡分布的第二维度，揭示了 PEFT 和全量微调各自的最优工作区间
- 子采样+模型融合的思路简洁有效，可完全并行训练，计算开销可控
- 将 Model Soups 首次应用于长尾分类问题，通过渐进不平衡的子集训练解决了传统 Soups 的头部偏向
- 实验覆盖 5 个基准、多种不平衡结构，结论具有较强说服力

## 局限与展望

- 头尾类划分使用固定阈值（$n_c > 100$），可能过于简化。广义 Pareto 分布等参数化框架可能更精确
- 仅在 CLIP ViT 骨干上验证，其他基础模型（如 DINOv2）的泛化性未知
- 需要训练 $NM+1$ 个模型，虽然可并行化但仍增加了计算资源需求
- $\lambda$ 仅使用 0.3 和 0.7 两个值，更精细的自适应调度可能带来进一步提升

## 相关工作与启发

- Model Soups：通过权重平均提升鲁棒性的开创性工作，本文将其推广到长尾场景
- LIFT (Shi et al.)：PEFT + LA 达到 SOTA，本文揭示了其在非尾部密集场景的局限
- BBN、RIDE、SADE 等集成方法：推理时需多专家，LT-Soups 融合为单一网络更高效
- 对基础模型时代的长尾学习提出了新的方法论视角

## 评分

- 新颖性: ⭐⭐⭐⭐ 双轴不平衡分析框架新颖，子采样 Model Soups 的思路简洁且有效
- 实验充分度: ⭐⭐⭐⭐⭐ 5个基准+可控合成实验+详尽消融，实验设计严谨
- 写作质量: ⭐⭐⭐⭐ 动机推导清晰，从 toy experiment 到方法设计逻辑流畅
- 价值: ⭐⭐⭐⭐ 对基础模型时代长尾学习提供了实用且高效的解决方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Bone Soups: A Seek-and-Soup Model Merging Approach for Controllable Multi-Objective Generation](../../ACL2025/model_compression/bone_soups_multi_objective_gen.md)
- [\[NeurIPS 2025\] RAT: Bridging RNN Efficiency and Attention Accuracy via Chunk-based Sequence Modeling](rat_bridging_rnn_efficiency_and_attention_accuracy_via_chunk-based_sequence_mode.md)
- [\[NeurIPS 2025\] Accurate and Efficient Low-Rank Model Merging in Core Space](accurate_and_efficient_low-rank_model_merging_in_core_space.md)
- [\[NeurIPS 2025\] A Granular Study of Safety Pretraining under Model Abliteration](a_granular_study_of_safety_pretraining_under_model_abliteration.md)
- [\[NeurIPS 2025\] Toward Efficient Inference Attacks: Shadow Model Sharing via Mixture-of-Experts](toward_efficient_inference_attacks_shadow_model_sharing_via_mixture-of-experts.md)

</div>

<!-- RELATED:END -->
