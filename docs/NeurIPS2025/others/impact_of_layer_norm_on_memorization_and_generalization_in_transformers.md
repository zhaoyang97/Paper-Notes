---
title: >-
  [论文解读] Impact of Layer Norm on Memorization and Generalization in Transformers
description: >-
  [NeurIPS 2025][Layer Normalization] 系统揭示了LayerNorm在Pre-LN和Post-LN Transformer中的截然不同：角色：Pre-LN中LN对学习至关重要，移除会破坏泛化；Post-LN中LN驱动记忆化，移除可抑制记忆化并恢复真实标签。 Layer Normalizatio…
tags:
  - "NeurIPS 2025"
  - "Layer Normalization"
  - "记忆化"
  - "generalization"
  - "Pre-LN"
  - "Post-LN"
---

# Impact of Layer Norm on Memorization and Generalization in Transformers

**会议**: NeurIPS 2025  
**arXiv**: [2511.10566](https://arxiv.org/abs/2511.10566)  
**代码**: [GitHub](https://github.com/JEKimLab/NeurIPS2025_LayernormMemorization)  
**领域**: Transformer架构分析 / 深度学习理论  
**关键词**: Layer Normalization, 记忆化, generalization, Pre-LN, Post-LN  
**作者**: Rishi Singhal, Jung-Eun Kim  
**机构**: North Carolina State University

## 一句话总结

系统揭示了LayerNorm在Pre-LN和Post-LN Transformer中的**截然不同**角色：Pre-LN中LN对学习至关重要，移除会破坏泛化；Post-LN中LN驱动记忆化，移除可抑制记忆化并恢复真实标签。

## 研究背景与动机

Layer Normalization (LN) 是Transformer的基本组件，负责稳定训练和改善优化。当前架构主要有两种LN放置方式：Post-LN（Vaswani 2017，LN在残差连接之后）和Pre-LN（Xiong 2020，LN在子层之前），后者因梯度流更稳定而成为GPT、LLaMA、ViT等现代架构的首选。

尽管已有工作研究了注意力头和FFN在记忆化中的角色，但**LN对记忆化和学习的影响**几乎未被探索。先前Xu et al. (2019)仅模糊暗示LN可能导致Pre-LN模型过拟合。本文发现，LN在两种架构中的作用是**定性不同的**：Pre-LN中LN是学习的关键，Post-LN中LN是记忆化的关键。这一发现通过梯度分析获得了理论支持。

## 方法详解

### 整体框架

方法包含三个层次的分析：(1) 移除LN可学习参数(保留归一化操作N(x))，比较完整模型与无LN模型的学习/记忆化行为；(2) 分层移除(早期/中期/后期)定位最关键的LN层；(3) 梯度范数分析解释观察到的现象。

### 关键设计

1. **LN参数移除实验设计**: 保留标准化操作 $N(x) = (x-\mu)/\sigma$ 但移除可学习的缩放 $w$ 和偏置 $b$ 参数。引入1%的噪声标签训练至100%训练准确率以确保记忆化发生。定义四个指标：学习准确率(测试集)、记忆化得分(噪声标签被记忆比例)、恢复得分(LN移除后恢复真实标签比例)、随机预测得分。设计动机：通过remove而非add来定位LN的因果作用。

2. **分层分析 (Section 5)**: 将 $N$ 层Transformer分为早期($1..N/3$)、中期($N/3+1..2N/3$)和后期($2N/3+1..N$)三组，逐组移除LN参数。发现早期LN最关键——Pre-LN中移除早期LN对学习破坏最大($\Delta_{\text{overfit}}^{\text{Pre, early}} > \Delta_{\text{overfit}}^{\text{Pre, middle}} > \Delta_{\text{overfit}}^{\text{Pre, later}}$)，Post-LN中移除早期LN对记忆化抑制最有效($\Delta_{\text{overfit}}^{\text{Post, early}} < \Delta_{\text{overfit}}^{\text{Post, middle}} < \Delta_{\text{overfit}}^{\text{Post, later}}$)。

3. **梯度范数分析 (Section 6)**: 计算损失对LN输入的梯度 $g_x = \partial\mathcal{L}/\partial x$，分别对测试样本和噪声标签样本取均值得到 $\|g_x^{\text{learn}}\|_2$ 和 $\|g_x^{\text{mem}}\|_2$。**Theorem 1** 证明 $\|g_x^{\text{learn}}\|_2 \geq \|g_x^{\text{mem}}\|_2$ across all layers。关键观察：Pre-LN中学习/记忆化梯度比值远大于Post-LN($\frac{\|g_x^{\text{learn}}\|}{\|g_x^{\text{mem}}\|}|_{\text{Pre-LN}} \gg \frac{\|g_x^{\text{learn}}\|}{\|g_x^{\text{mem}}\|}|_{\text{Post-LN}}$)。这解释了为何Pre-LN中移除LN主要破坏学习，Post-LN中移除LN主要抑制记忆化。

4. **早期层梯度上界分析 (Theorems 2&3)**: 推导了梯度范数的上界。Pre-LN：$\|g_{x_i}\|_2 \leq s_{\max}(P_2) \cdot \prod_{j=i}^N (1 + s_{\max}(J_{\text{FFN}}^{\text{LN}_2(x_j')} J_{\text{LN}_2}^{x_j'})) \cdot \prod_{j=i}^N (1 + s_{\max}(J_{\text{MHSA}}^{\text{LN}_1(x_j)} J_{\text{LN}_1}^{x_j}))$。由于乘积项随层数减少而递减，证明了 $\text{UB}(\|g_{x_1}\|_2) \geq \text{UB}(\|g_{x_2}\|_2) \geq \cdots$。

### 损失函数 / 训练策略

使用交叉熵损失 $\mathcal{L} = -\sum_k y_k \log(\hat{y}_k)$，学习率2e-5，batch size 16，Post-LN训练40 epochs，Pre-LN训练70 epochs（因LN移除后学习受损需更多epoch观察恢复）。所有实验3个随机种子。

## 实验关键数据

### 主实验

| 模型类型 | 代表模型 | LN移除后学习准确率 | 记忆化分数变化 | 恢复分数 | 说明 |
|---------|---------|------------------|--------------|---------|------|
| Post-LN | ELECTRA (News) | 几乎不变 (~稳定) | 大幅下降 | 高 (绿条) | LN移除抑制记忆化 |
| Post-LN | BERT (Emotions) | 保持稳定 | 大幅下降 | 高 | 跨模型一致 |
| Pre-LN | Qwen2 (News) | 大幅下降 (~30%↓) | 仍高 | 极低 | LN移除破坏学习 |
| Pre-LN | ViT-B (CIFAR10) | 大幅下降 | 仍高 | 极低 | 视觉模型也一致 |
| Post-LN | DistilBERT | 不受影响 | 未显著下降 | 低 | 唯一例外 |

### 消融实验（分层移除）

| 配置 | Pre-LN学习影响 | Post-LN记忆化影响 | $\Delta_{\text{overfit}}$ |
|------|---------------|------------------|--------------------------|
| 移除早期LN | 学习破坏最严重 | 记忆化抑制最有效 | Pre最大 / Post最小 |
| 移除中期LN | 中等影响 | 中等效果 | 中等 |
| 移除后期LN | 影响最小 | 效果最小 | Pre最小 / Post最大 |

### 关键发现

- **Pre-LN vs Post-LN的根本区别**：Pre-LN中梯度范数在第一层极高而后续层几乎为零(集中)，故早期LN移除后无法恢复；Post-LN中梯度范数跨层逐渐衰减(分散)，后续层可补偿早期层缺失。
- **13个模型×6个数据集的一致性**：BERT/RoBERTa/DeBERTa/ELECTRA/Longformer (Post-LN) + GPT2/GPTNeo/Qwen2/ViT-B/ViT-S/DeiT/RoBERTa-PreLN (Pre-LN) 均验证核心发现。
- **DistilBERT例外**：可能因蒸馏训练使其他组件对记忆化有更大影响。

## 亮点与洞察

- 发现了LN在Pre/Post两种架构中的"对偶"角色——一个对学习关键、一个对记忆化关键——这是全新的认知
- 早期层LN最关键的结论与近期"深层无用"(layer pruning)文献契合，但提供了更精确的归因
- 梯度范数ratio的分析提供了简洁的理论解释框架
- 13模型6数据集的大规模验证使结论有很强的可信度

## 局限与展望

- 仅研究了1%噪声标签这一种记忆化诱导方式，自然记忆化(长尾分布等)可能有不同表现
- 理论证明依赖于Theorem 1中"同类样本特征相似"的假设，实际中可能不完全成立
- Post-LN只在语言模型中有实际架构(视觉Transformer几乎都是Pre-LN)，限制了Post-LN结论的直接应用范围
- 未探讨RMSNorm (Qwen2使用)与标准LayerNorm的差异影响

## 相关工作与启发

- 扩展了Xu et al. (2019)对LN过拟合的初步观察，给出了更精确的Pre/Post区分
- 与近期layer pruning工作(Men et al. 2024, Lad et al. 2024)互补：它们发现深层"无用"，本文精确定位到LN是关键因素
- 对实际应用的启示：Post-LN模型可通过移除LN参数来进行记忆化缓解和隐私保护

## 关键结论速查

- Pre-LN: LN移除 → 学习崩溃 + 记忆化持续 + 过拟合加剧
- Post-LN: LN移除 → 学习不变 + 记忆化抑制 + 真实标签恢复
- 早期层LN最关键，因为梯度范数上界随层数递减
- 梯度比 $\|g_x^{\text{learn}}\|/\|g_x^{\text{mem}}\|$ 在Pre-LN中远大于Post-LN
- 实验覆盖: BERT/RoBERTa/DeBERTa/ELECTRA/Longformer/DistilBERT (Post-LN) + GPT2/GPTNeo/Qwen2/ViT-B/ViT-S/DeiT/RoBERTa-PreLN (Pre-LN)

## 评分

- 新颖性: ⭐⭐⭐⭐ LN在Pre/Post中作用的"对偶性"是重要且意外的发现
- 实验充分度: ⭐⭐⭐⭐⭐ 13模型×6数据集，覆盖NLP和CV，结论一致性极强
- 写作质量: ⭐⭐⭐⭐ 结构清晰，但正文略长，部分可精简
- 价值: ⭐⭐⭐⭐ 对Transformer架构理解有实质性推进，对记忆化缓解有直接实用意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Do ImageNet-trained Models Learn Shortcuts? The Impact of Frequency Shortcuts on Generalization](../../CVPR2025/others/do_imagenet-trained_models_learn_shortcuts_the_impact_of_frequency_shortcuts_on_.md)
- [\[NeurIPS 2025\] The Cost of Robustness: Tighter Bounds on Parameter Complexity for Robust Memorization in ReLU Nets](the_cost_of_robustness_tighter_bounds_on_parameter_complexity_for_robust_memoriz.md)
- [\[NeurIPS 2025\] Fostering the Ecosystem of AI for Social Impact Requires Expanding and Strengthening Evaluation Standards](fostering_the_ecosystem_of_ai_for_social_impact_requires_expanding_and_strengthe.md)
- [\[NeurIPS 2025\] A Theoretical Framework for Grokking: Interpolation followed by Riemannian Norm Minimisation](a_theoretical_framework_for_grokking_interpolation_followed_by_riemannian_norm_m.md)
- [\[NeurIPS 2025\] Generalized Linear Mode Connectivity for Transformers](generalized_linear_mode_connectivity_for_transformers.md)

</div>

<!-- RELATED:END -->
