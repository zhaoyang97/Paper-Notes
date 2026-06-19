---
title: >-
  [论文解读] Emergence of Linear Truth Encodings in Language Models
description: >-
  [NeurIPS 2025][可解释性][真值编码] 提出 **Truth Co-occurrence Hypothesis (TCH)**——真实陈述倾向于与其他真实陈述共现——并通过一个最简单的单层 Transformer 玩具模型，端到端地展示了线性真值子空间如何通过两阶段训练动态（先记忆 → 后编码真值）自然涌现，为理解 LLM 中广泛报告的线性真值表示提供了首个机制性解释。
tags:
  - "NeurIPS 2025"
  - "可解释性"
  - "真值编码"
  - "线性表示"
  - "联想记忆"
  - "训练动态"
  - "LayerNorm"
---

# Emergence of Linear Truth Encodings in Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2510.15804](https://arxiv.org/abs/2510.15804)  
**代码**: [https://github.com/shauli-ravfogel/truth-encoding-neurips](https://github.com/shauli-ravfogel/truth-encoding-neurips)  
**领域**: 可解释性  
**关键词**: 真值编码, 线性表示, 联想记忆, 训练动态, LayerNorm

## 一句话总结
提出 **Truth Co-occurrence Hypothesis (TCH)**——真实陈述倾向于与其他真实陈述共现——并通过一个最简单的单层 Transformer 玩具模型，端到端地展示了线性真值子空间如何通过两阶段训练动态（先记忆 → 后编码真值）自然涌现，为理解 LLM 中广泛报告的线性真值表示提供了首个机制性解释。

## 研究背景与动机

**关键观测**：大量 probing 研究发现，LLM 的残差流中存在**低秩线性子空间**，能跨领域地分离真假陈述（如 "2+2=4" vs "法国首都是罗马"），这一发现引发了对 LLM 幻觉缓解的关注。

**两个未解之谜**：
   - **Why**：为什么训练过程中会涌现这样的子空间？为什么模型需要编码"真值"这个潜变量？
   - **How**：在推理时，这种线性分离是如何被计算出来的？

**已有解释的不足**：
   - Persona 假说（Joshi et al., 2024）：将真假与词汇风格（如维基百科 vs 社交媒体）关联，但这是表面线索而非本质机制
   - 缺乏一个不依赖词汇线索的、从训练动态角度的机制解释

**本文切入点**：如果同一段文本中的真假陈述是相关的（TCH），那么推断"真值"可以降低语言模型损失——这为模型学习真值表示提供了优化层面的动机。

## 方法详解

### 整体框架

**三步走**：
1. 用真实语料验证 TCH 假说（MAVEN-FACT 数据集）
2. 构建最简玩具模型，分析真值编码的涌现机制
3. 在合成数据和自然语言数据上验证理论预测，并在预训练 LLM 中检验

### 关键设计

#### 1. Truth Co-occurrence Hypothesis (TCH) 的验证

在 MAVEN-FACT 语料中统计事件级事实性标注：
- 总体虚假率 $p = 0.0209$
- 同一文档中两个事件同时虚假的概率 $= 0.0009$，是独立基线 $p^2 = 0.00044$ 的 **≈2 倍**
- 聚类比率（方差超额）= 1.23，说明存在 23% 的文档间额外异质性
- $\chi^2$ 检验 $= 4.17 \times 10^3$，$p \approx 9 \times 10^{-49}$（极度显著）
- **结论**：虚假陈述在文档中呈聚类分布——追踪"真值"对 LM 是有益的

#### 2. 数据生成过程

每个训练样本是四 token 序列 $x \; y \; x' \; y'$：
- $x, x'$ 为主语（如 "法国首都"），$y, y'$ 为属性（如 "巴黎"）
- 每个 $x$ 有唯一正确属性 $g(x)$
- 以概率 $\rho$ 生成真实序列（$y = g(x), y' = g(x')$）
- 以概率 $1-\rho$ 生成虚假序列（$y, y'$ 均从属性集均匀随机采样）
- **关键**：同一样本中 $y$ 和 $y'$ 的真假**完全相关**——这是 TCH 的核心体现

**推断真值的收益量化**：当 $|\mathcal{A}| \to \infty$ 时，知道 $T$ vs 不知道 $T$ 的损失差 = $H_2(\rho)$（$\rho$ 的二元熵）。在 $\rho = 0.5$ 时收益最大。

#### 3. 玩具模型分析

**模型**：单层 Transformer，均匀因果注意力 + LayerNorm，正交 one-hot 嵌入，维度 $d = 4N + 3$。
- Token 嵌入 $e_z$、位置嵌入 $p_t$、反嵌入（unembedding）$u_z$ 均为正交向量
- 前向传播：$F_W(z_{1:t})_t = U \cdot \mathsf{N}\left(e_{z_t} + p_t + \frac{1}{t}\sum_{s=1}^t W(e_{z_s} + p_s)\right)$
- 其中 $\mathsf{N}(v) = v / \|v\|$ 为 LayerNorm

**Value 矩阵 $W$ 的结构**：训练后 $W$ 呈现清晰的块结构：
- $W e_x = -\alpha_1 e_x + \beta_1 u_{g(x)}$：主语 → 正确属性 + 自身抑制
- $W e_y = \alpha_2 e_{g^{-1}(y)} - \beta_2 u_y$：属性 → 对应主语 + 自身抑制
- $W p_t$：位置嵌入 → 均匀的属性/主语分布

#### 4. 线性分离与锐化机制

**核心量**：对于同时注意到 $x$ 和 $y$ 的 token，其残差流中有：
$$\zeta(x,y) = W(e_x + e_y) = -\alpha_1 e_x + \alpha_2 e_{g^{-1}(y)} + \beta_1 u_{g(x)} - \beta_2 u_y$$

**关键不等式**：$\|\zeta(x, g(x))\|^2 = \|\zeta(x, y)\|^2 - 2\alpha_1\alpha_2 - 2\beta_1\beta_2$ （当 $y \neq g(x)$ 时）

- 真实序列的 $\zeta$ **范数更小**（因为向量中的分量会互相抵消）
- 虚假序列的 $\zeta$ 范数更大（分量不抵消）
- **LayerNorm 将范数差异转化为置信度差异**：真实序列归一化后振幅更大 → softmax 输出更尖锐 → 对正确属性更自信

**定理 1（锐化）**：对于满足上述结构的 $W$，真实上下文下模型对 $g(x')$ 的 logit 间距 > 0，虚假上下文下间距 = 0。

**定理 2（线性真值方向）**：
- **无 LayerNorm**：$y$ 位置的输出**不存在**真假的线性分离器
- **有 LayerNorm**：存在线性分离器，margin 至少为 $\delta = \frac{1}{2\sqrt{2}}\left(1 - \frac{1}{\sqrt{1 + \alpha^2 + \beta^2}}\right)$
- **关键洞察**：LayerNorm 是线性真值编码涌现的**必要条件**

**定理 3（训练动态）**：在简化设置下，两步梯度下降 on $L_1$ + 一步 on $L_3$ 即可产生所需的块结构，LayerNorm 在梯度结构中起关键作用。

### 损失函数 / 训练策略

标准的自回归语言模型损失（交叉熵）：
$$L(W) = \sum_{t=1}^3 \mathbb{E}_{z_{1:t+1}}\left[-\log \mathcal{S}_\beta(F_W(z_{1:t}))_{z_{t+1}}\right]$$

使用 Adam 优化器，逆温度 $\beta = \sqrt{d}$（对应 RMSNorm）。

## 实验关键数据

### 主实验

**合成数据（1 层模型，$\rho = 0.99$，$|\mathcal{S}| = |\mathcal{A}| = 512$）**：

两阶段训练动态：
1. **记忆阶段**（前 ~1000 批次）：模型快速学会 $g(x)$ 映射，在真实和虚假序列上都预测正确属性的概率→1
2. **真值编码阶段**（~7500 批次后突然出现）：线性分类 AUC 急剧上升，同时模型开始降低在虚假序列上预测正确属性的概率

**PCA 可视化**：
- LayerNorm 前：真假表示围绕原点聚类，无分离
- LayerNorm 后：出现清晰的线性分离
- 输入嵌入：$e_x \approx -e_{g(x)}$（主语和其正确属性的嵌入近似反向）

### 消融实验

**$\rho$ 的影响**：
- $\rho = 0.999$：延迟出现但仍会涌现
- $\rho = 1.0$：不涌现（无虚假样本则无需编码真值）——但玩具模型预测在 $\rho = 1$ 时也能涌现（OOD 泛化）

**层数影响**：
- 1 层：真值编码在 $x'$ 位置出现
- 多层：可能先在 $y$ 位置编码，然后复制到 $x'$；或直接在 $x'$ 编码

**自然语言实验（CounterFact 数据集）**：
- 小 Transformer（2/5/9 层，d=256）训练在实例化 TCH 的配对数据上
- **结果与合成数据一致**：快速记忆 → 延迟的线性编码涌现 → 虚假序列上概率下降
- 1 层模型出现 epoch-wise double descent

### 关键发现

**在预训练 LLaMA-3-8B 上的验证**：
- 前缀中的虚假句子**降低**模型对后续正确属性的预测概率（2 个虚假前缀导致概率下降 4.55×），**支持 TCH**
- LLaMA-3-8B 在所有中间层和最后层都能线性分离真假实例（分类准确率 >95%）
- 在真值子空间上进行**干预**（添加 $\alpha(\mu_T - \mu_F)$ 到表示中）可以**逆转虚假上下文的影响**，提升正确属性的概率

## 亮点与洞察

- **TCH 假说的优雅性**：不依赖词汇风格线索，仅靠真假的共现统计就能解释线性真值编码的涌现——这比 "persona" 解释更深层
- **LayerNorm 的意外关键性**：是线性分离的**必要组件**——无 LayerNorm 则无线性分离。这一发现与 Stolfo et al. (2024) 关于 "confidence neurons" 的工作一致
- **两阶段动态的普遍性**：在玩具模型、合成 NLP 数据、预训练 LLM 中均观察到，暗示这可能是一个普遍的训练机制
- **理论-实验闭环**：从假说（TCH）→ 数据验证 → 玩具模型分析 → 合成实验 → 预训练模型验证，形成完整的论证链
- **实用价值**：理解真值编码的机制可能有助于开发更好的幻觉检测和缓解策略

## 局限与展望
- **玩具模型过于简化**：单层、正交嵌入、均匀注意力——与真实 Transformer 差距巨大
- **仅一种关系**：合成数据只包含一个潜在关系，真实语料有多种关系的复杂交互
- **不考虑逻辑约束**：真实世界中的真假有传递性、互斥性等逻辑关系，模型中未体现
- **虚假分布过于简化**：均匀随机替换 vs 真实世界中虚假信息的条件分布更复杂
- **TCH 验证在小规模语料上**：MAVEN-FACT 的虚假率仅 2%，更大规模的验证尚缺
- **未能完全解释多层模型的行为**：不同随机种子可能采取不同策略

## 相关工作与启发
- **vs Li et al. (2024b) / Marks & Tegmark (2024)**：这些工作发现了线性真值编码的存在，本文解释了其**涌现机制**
- **vs Joshi et al. (2024, Persona 假说)**：Persona 依赖词汇线索，TCH 提供了更底层的统计解释——两者可能共同作用
- **vs Geva et al. (2021, key-value memory)**：本文建立在"MLP 是关联记忆"的理解之上，展示了真值编码如何利用已记忆的事实关联
- **vs Stolfo et al. (2024, confidence neurons)**：两者发现了一致的机制——LayerNorm 通过范数调控置信度，同时使真假表示线性可分

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ TCH 假说 + 玩具模型的机制分析是全新的理论贡献，对理解 LLM 内部表示有根本性意义
- 实验充分度: ⭐⭐⭐⭐ 合成实验充分，预训练模型验证有说服力，但自然语言实验规模较小
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，从假说到验证的逻辑链清晰优美
- 价值: ⭐⭐⭐⭐⭐ 对 LLM 可解释性和幻觉理解有深远影响，可能启发新的幻觉缓解技术

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Minimizing False-Positive Attributions in Explanations of Non-Linear Models](minimizing_false-positive_attributions_in_explanations_of_non-linear_models.md)
- [\[ACL 2025\] Probing the Geometry of Truth: Consistency and Generalization of Truth Directions](../../ACL2025/interpretability/probing_the_geometry_of_truth_consistency_and_generalization_of_truth_directions.md)
- [\[NeurIPS 2025\] Table as a Modality for Large Language Models](table_as_a_modality_for_large_language_models.md)
- [\[ICLR 2026\] Beyond Linear Probes: Dynamic Safety Monitoring for Language Models](../../ICLR2026/interpretability/beyond_linear_probes_dynamic_safety_monitoring_for_language_models.md)
- [\[NeurIPS 2025\] Are Greedy Task Orderings Better Than Random in Continual Linear Regression?](are_greedy_task_orderings_better_than_random_in_continual_linear_regression.md)

</div>

<!-- RELATED:END -->
