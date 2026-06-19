---
title: >-
  [论文解读] What Happens During the Loss Plateau? Understanding Abrupt Learning in Transformers
description: >-
  [NeurIPS 2025][可解释性][abrupt learning] 系统研究 Transformer 训练中的"突变学习"现象，揭示 loss 平台期内模型已学会部分解、同时表现出输出重复偏差和表示坍缩，并证明注意力图的缓慢学习是关键瓶颈，相关发现在 Pythia/OLMo 等 LLM 预训练早期也得到验证。
tags:
  - "NeurIPS 2025"
  - "可解释性"
  - "abrupt learning"
  - "loss plateau"
  - "Transformer"
  - "representation collapse"
  - "repetition bias"
  - "注意力机制"
---

# What Happens During the Loss Plateau? Understanding Abrupt Learning in Transformers

**会议**: NeurIPS 2025  
**arXiv**: [2506.13688](https://arxiv.org/abs/2506.13688)  
**代码**: [github.com/pulkitgopalani/tf-loss-plateau](https://github.com/pulkitgopalani/tf-loss-plateau)  
**领域**: 可解释性  
**关键词**: abrupt learning, loss plateau, Transformer training dynamics, representation collapse, repetition bias, attention map

## 一句话总结

系统研究 Transformer 训练中的"突变学习"现象，揭示 loss 平台期内模型已学会部分解、同时表现出输出重复偏差和表示坍缩，并证明注意力图的缓慢学习是关键瓶颈，相关发现在 Pythia/OLMo 等 LLM 预训练早期也得到验证。

## 研究背景与动机

**突变学习（Abrupt Learning）** 是 Transformer 训练中常见的现象：模型性能在长时间的平台期（loss plateau）中停滞，随后突然急剧提升。这一现象被认为是"涌现"（emergence）的一种表现形式。

尽管已有工作针对特定任务（如 ICL、稀疏奇偶校验、语法学习等）研究了此现象，但缺乏：

**统一的理解**：不同任务间的共通模式和机制尚不明确

**通用性**：多数工作依赖于特定假设，限制了结论的泛化

**与 LLM 的联系**：小规模实验发现是否适用于大模型预训练尚未验证

本文三个核心问题：
- 平台期内模型的输入-输出行为和内部表示有什么共同模式？
- 突然提升前积累了哪些隐性进展？
- 这些发现是否具有普遍性？

## 方法详解

### 整体框架

以浅层 Transformer（1-2 层）在算法任务上的训练为主要研究平台，辅以 LLM 预训练验证。核心算法任务为 **移动窗口求和（MWS）**：

输入序列 $x_1, x_2, \ldots, x_n$，输出 $y_i$：

$$y_i = \begin{cases} x_1 & i=1 \\ (x_{i-1} + x_i) \bmod p & i \geq 2 \end{cases}$$

实验设置：$n=16, p=17$，使用 1 层 1 头线性注意力 Transformer。

### 关键发现一：平台期内的部分解（Partial Solution）

模型在 loss 平台期已学会预测**较简单的 token 子集**。例如在 MWS 任务中，$y_1 = x_1$ 只需复制，模型很早就能正确预测第一个输出 token，而后续 token 仍然错误。这一模式在多种算法任务中一致出现。

### 关键发现二：输出重复偏差（Repetition Bias）

平台期内模型倾向于生成重复 token，如 $x, x, x, \ldots$。定义**重复频率**：

$$\rho = \frac{1}{n-1} \sum_{i=1}^{n-1} \mathbf{1}[y_i = y_{i+1}]$$

在训练早期（约前 50 步），$\rho$ 从接近 0 快速上升到约 0.8，显示这是梯度训练引入的隐式偏差。

对于无连续重复的任务（如前缀和），重复偏差以另一种形式出现——输出中只出现少数几个 token。用**序列熵**量化：

$$\text{SeqEnt}(y_1, \ldots, y_n) = \sum_{i=1}^{|V|} p_i \log(1/p_i)$$

### 关键发现三：表示坍缩（Representation Collapse）

不同输出位置的隐藏表示变得几乎平行。度量方式为位置间的余弦相似度：

$$\text{COS}_{i,j} = \frac{\langle \mathbf{h}_i, \mathbf{h}_j \rangle}{\|\mathbf{h}_i\| \|\mathbf{h}_j\|}$$

训练早期，输出位置间的平均余弦相似度急速上升到约 0.95（排除已学会的第一个位置），随性能提升后显著下降。

重要区分：这与深层 softmax Transformer 的初始化 rank collapse **不同** — 本文的表示坍缩发生在训练几步之后而非初始化时，且使用浅层线性注意力。

### 关键发现四：注意力图学习是关键瓶颈

**注意力进展度量（APM）**：

$$\text{APM} = \frac{\sum_{(i,j) \in \Omega} |A_{ij}|}{\sum_{(i,j)} |A_{ij}|}$$

其中 $\Omega$ 是最优注意力图中的位置集合。APM 在平台期内**单调递增**，从接近 0 到约 0.8，且增长比 loss/accuracy 变化更平滑 — 说明注意力图在 loss 看似停滞时已在缓慢进步。

### 干预实验：偏置注意力图

通过在训练时用掩码 $M$ 对注意力图做乘性缩放：对最优位置 $(i,j) \in \Omega$ 乘以 $c$：

- **$c > 1$（偏向最优）**：表示坍缩程度降低（峰值余弦从 0.95 降到约 0.6），重复频率更低，收敛更快
- **$0 < c < 1$（偏离最优）**：平台期延长，表示坍缩和重复偏差加剧

### 最优初始化实验

- 固定注意力层权重为最优值 → 几乎无平台期，快速收敛，无表示坍缩
- 固定 MLP 或 Embedding 为最优值 → 训练动态几乎不变，仍出现平台期和坍缩

结论：注意力图是造成表示坍缩和 loss 平台的主要瓶颈。

### 训练策略

- 使用 Adam 优化器，学习率 $10^{-4}$，无 weight decay
- 在线训练（每步新采样 256 个样本），无固定训练集
- 验证了 SGD、Muon 等不同优化器以及不同超参下现象的一致性

## 实验关键数据

### 主实验：多任务验证

| 任务 | 部分解内容 | 重复偏差 | 表示坍缩 |
|------|-----------|----------|----------|
| MWS（移动窗口求和） | $y_1 = x_1$ 复制 | $\rho \approx 0.8$ | COS $\approx 0.95$ |
| 前缀和 | 首个 token | 序列熵低 | COS $\approx 0.80$ |
| 多位加法 | 最终进位 | 存在 | 存在 |
| 排列 | 部分位置 | 存在 | 存在 |
| 直方图 | 部分 bin | 存在 | 存在 |

所有任务均一致观察到三个现象，证实了发现的普遍性。

### 注意力干预实验

| 干预条件 | 峰值 COS | 收敛时间 | 峰值重复频率 |
|----------|----------|----------|-------------|
| $c = 1$（基线） | $\approx 0.95$ | 正常 | $\approx 0.8$ |
| $c = 10$（强偏向最优） | $\approx 0.6$ | 显著加速 | 低 |
| $c = 0.2$（偏离最优） | $> 0.95$ | 显著延迟 | 高且持续 |
| 固定 KQ 为最优 | $\approx 0.45$ | 非常快 | 低 |
| 固定 KQOV 为最优 | $\approx 0.15$ | 最快 | 极低 |
| 固定 MLP 为最优 | $\approx 0.95$ | 基本不变 | 基本不变 |

### LLM 验证

| 模型 | 早期峰值 COS | 重复偏差 |
|------|-------------|---------|
| Pythia-14M | $> 0.9$ | 存在 |
| Pythia-1B | $> 0.9$ | 存在 |
| Pythia-1.4B | $> 0.9$ | 存在 |
| Pythia-2.8B | $> 0.9$ | 存在 |
| OLMo-2 7B (step 150) | $\approx 0.93$ | 存在 |
| OLMo-2 7B (step 600) | $\approx 0.43$ | 消退 |

### 关键发现

1. **重复序列更容易学习**：REPEAT1 任务（$y_i = x_1$）几乎无 loss 平台期，仅 1 步梯度就使 COS 达到约 0.5
2. **$\alpha_1$ 指标**（$\frac{1}{n}\sum \mathbf{1}[y_i = y_1]$）在早期快速接近 1.0 — 模型倾向复制首个 token 到所有位置
3. **注意力层而非 MLP 导致坍缩**：前后注意力层的残差流对比显示，坍缩发生在注意力层**之后**
4. **APM 在 loss 下降前已有进展**：隐性进步在平台期内累积

## 亮点与洞察

1. **统一视角**：首次系统性地将部分解、重复偏差、表示坍缩和注意力学习作为一个整体来理解 Transformer 的训练动态
2. **因果关系验证**：通过注意力图干预实验，不仅观察到相关性，更建立了注意力学习 → 坍缩/重复 → 平台期的因果链
3. **"搜索"假说的提出**：将平台期类比为注意力层在 token 空间中的搜索过程，类似于两层神经网络中第一层对齐目标函数支撑的过程
4. **小模型到大模型的桥接**：在 Pythia 和 OLMo 的预训练早期验证了相同现象，极大增强了结论的说服力
5. **方法论清晰**：APM、重复频率、序列熵等多种量化指标，使抽象的训练动态变得可测量可比较

## 局限与展望

1. **理论解释不足**：为何梯度训练引入表示坍缩偏差？仅提出"搜索"假说但缺乏严格理论验证
2. **小模型主导**：主要结论基于 1-2 层 Transformer，多层深层模型中各层如何交互仍不清楚
3. **算法任务特殊性**：所用任务有明确最优解，自然语言等无明确最优解的场景表现如何需要进一步研究
4. **线性注意力限制**：虽也在 softmax 注意力上验证了部分结论，但线性注意力的特殊性可能影响某些发现的泛化
5. **LLM 验证有限**：仅检查了早期 checkpoint 的表示坍缩，未深入分析 LLM 中的部分解和注意力进展

## 相关工作与启发

- **与 grokking 的区别**：grokking 是固定数据集上的记忆后泛化现象，本文研究的是在线训练中的平台期突破，两者机制不同
- **与 rank collapse 的区别**：后者发生在深层 softmax Transformer 的初始化时，而本文的表示坍缩是训练引入的、发生在浅层线性注意力模型中
- **对 LLM 训练的启发**：理解早期训练动态有助于设计更好的学习率调度、注意力初始化策略和课程学习方案
- **注意力转移**：发现与视觉 Transformer 中预训练注意力转移的有效性一致，暗示注意力模式是跨任务可迁移的关键知识

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次统一揭示 loss 平台期的多维特征及注意力学习的核心作用
- 实验充分度: ⭐⭐⭐⭐⭐ — 多任务验证、干预实验、多种度量、LLM 验证，极为充分
- 写作质量: ⭐⭐⭐⭐⭐ — 逻辑清晰，图表精美，量化指标设计巧妙
- 价值: ⭐⭐⭐⭐⭐ — 对理解 Transformer 训练动态具有基础性贡献，对 LLM 训练实践有指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Understanding Prompt Tuning and In-Context Learning via Meta-Learning](understanding_prompt_tuning_and_in-context_learning_via_meta-learning.md)
- [\[NeurIPS 2025\] Causal Head Gating: A Framework for Interpreting Roles of Attention Heads in Transformers](causal_head_gating_a_framework_for_interpreting_roles_of_attention_heads_in_tran.md)
- [\[NeurIPS 2025\] How Intrinsic Motivation Shapes Learned Representations in Decision Transformers: A Cognitive Interpretability Analysis](toward_explainable_offline_rl_analyzing_representations_in_intrinsically_motivat.md)
- [\[NeurIPS 2025\] Improving Perturbation-based Explanations by Understanding the Role of Uncertainty Calibration](improving_perturbation-based_explanations_by_understanding_the_role_of_uncertain.md)
- [\[ICML 2025\] What Makes an Ensemble (Un)interpretable?](../../ICML2025/interpretability/what_makes_an_ensemble_un_interpretable.md)

</div>

<!-- RELATED:END -->
