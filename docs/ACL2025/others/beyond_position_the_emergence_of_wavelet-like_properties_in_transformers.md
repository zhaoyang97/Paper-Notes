---
title: >-
  [论文解读] Beyond Position: the emergence of wavelet-like properties in Transformers
description: >-
  [ACL2025][RoPE] 通过频率分析和小波分解，揭示了使用 RoPE 位置编码的 Transformer 模型中注意力头自发涌现出类小波（wavelet-like）的多分辨率处理特性，以弥补 RoPE 在位置精度和频率分辨率之间的固有权衡。 位置编码是 Transformer 架构的基础组件，使天然对排列不变的模型能…
tags:
  - "ACL2025"
  - "RoPE"
  - "wavelet transform"
  - "positional encoding"
  - "multi-resolution"
  - "uncertainty principle"
  - "注意力机制"
---

# Beyond Position: the emergence of wavelet-like properties in Transformers

**会议**: ACL2025  
**arXiv**: [2410.18067](https://arxiv.org/abs/2410.18067)  
**代码**: -  
**领域**: Others (Transformer 分析)  
**关键词**: RoPE, wavelet transform, positional encoding, multi-resolution, uncertainty principle, attention head  

## 一句话总结

通过频率分析和小波分解，揭示了使用 RoPE 位置编码的 Transformer 模型中注意力头自发涌现出类小波（wavelet-like）的多分辨率处理特性，以弥补 RoPE 在位置精度和频率分辨率之间的固有权衡。

## 研究背景与动机

位置编码是 Transformer 架构的基础组件，使天然对排列不变的模型能捕获序列信息。RoPE（Rotary Position Embeddings）通过旋转变换将相对位置信息嵌入 embedding 中，在实践中取得了巨大成功，被 LLaMA、Mistral、Qwen 等主流模型广泛采用。

然而，理论分析（Barbero et al., 2024）揭示了 RoPE 的固有局限：它基于固定频率的正弦函数，在位置精度和频率分辨率之间存在类似不确定性原理（Heisenberg uncertainty principle）的权衡。具体来说：
- 较大的旋转角 θ 提供精确的位置信息，但快速旋转周期会混淆远距离关系
- 较小的 θ 更好地捕获长程模式，但会模糊局部位置

这产生了一个悖论：**为什么具有这些理论缺陷的模型在实践中表现如此出色？**

作者假设答案在于：配备 RoPE 的模型学会了通过发展涌现的、类小波的处理策略来补偿这些限制。

## 方法详解

### 整体分析框架

本文构建了一套完整的频域-小波-信息论分析框架来刻画注意力头的涌现特性：

### 1. 频率分析

对每个注意力头 h 的注意力分布 a_h(t)，计算功率谱密度（PSD）：

$$P_h(\omega) = |\mathcal{F}\{a_h(t)\}|^2$$

将频域划分为三个频带：
- **低频**（0-0.25 ω_N）：捕获全局上下文
- **中频**（0.25-0.75 ω_N）：中间尺度信息
- **高频**（0.75-ω_N）：细粒度局部信息

定义**频率选择性** S(h) 衡量每个头对特定频率的专注程度。

### 2. 小波分析

使用 Daubechies-2（db2）小波对注意力模式进行多尺度分解：

$$W_h(s, \tau) = \int a_h(t) \psi_{s,\tau}(t) dt$$

小波变换提供了时间-频率（位置-频率）的联合表示，弥补了纯频率分析缺乏位置局部化的缺点。在每个尺度上计算**小波熵**来衡量注意力能量在不同尺度和位置上的分布。

### 3. 不确定性分析

计算两种熵来验证不确定性原理：
- **位置熵** H_p(h)：注意力分布在 token 位置上的均匀程度
- **频谱熵** H_s(h)：归一化功率谱的熵

通过 position-spectrum 相关性 ρ(h) 来量化二者的权衡关系：

$$\rho(h) = \frac{\text{Cov}(H_p(h), H_s(h))}{\sigma_{H_p} \sigma_{H_s}}$$

ρ 接近 −1 表示位置精度和频谱精度之间存在强权衡，接近 +1 表示成功整合两者。

### 4. 尺度不变性测试

通过序列缩放（α ∈ {0.5, 0.25}）生成变体，计算小波系数的**尺度敏感度**：

$$S_h(\alpha) = 1 - \cos(W_h(x), W_h(x_\alpha))$$

低 S_h(α) 说明位置表示在序列长度变化时保持稳定。

### 5. 帧完备性验证

通过逆小波变换计算重建误差 ε，验证学习到的表示是否形成稳定的帧（frame）结构。

## 实验

### 实验设置

- **模型**：Gemma 2 (2B), Pythia (2.8B/12B), LLaMA-3.2 (1B), Mistral (7B), Qwen 2.5 (0.5B/5B)
- **数据**：500 个 Wikipedia 序列，长度各异
- **硬件**：A100/L4/T4 GPU

### 主要发现

**发现1：多分辨率处理的涌现**

注意力头自发分化为局部和全局处理器。频率分析显示一致的分层频率响应：
- 低频成分占 60-80% 的频谱能量（上下文骨干）
- 中频贡献稳定的 15-25%
- 高频处理细粒度细节，占 5-15%

随着信息在层间传播，低频的主导地位逐渐减弱，中高频成分的影响力增大，类似小波分析的自适应精细化过程。

**发现2：尺度不变性**

| 模型 | 尺度敏感度 (0.5x) | 尺度敏感度 (0.25x) | 位置-频谱相关 ρ | 重建误差 |
|------|----------|----------|------|------|
| Qwen 2.5 (0.5B) | 0.058 | 0.100 | -0.438 | 1.26e-07 |
| LLaMA 3.2 (1B) | 0.038 | 0.089 | -0.510 | 1.28e-07 |
| Mistral (7B) | 0.030 | 0.074 | -0.421 | 1.41e-07 |
| Pythia (2.8B) | 0.082 | 0.121 | -0.737 | 1.16e-07 |

RoPE 模型的尺度敏感度极低（Mistral 仅 0.030），且误差从 0.5x 到 0.25x 大约翻倍，符合小波理论预测的幂律缩放行为。

**发现3：不确定性原理的统计确认**

所有分析的模型都展现出**一致性的负位置-频谱相关 ρ**，直接证实了模型隐式学习并遵循了 Heisenberg-Gabor 不确定性原理。两种策略类型：
- **Mistral 7B**：高方差专业化策略（μ=0.804, σ=0.414），注意力头高度分化
- **Pythia 2.8B**：低方差均匀策略（μ=0.174），处于"过渡探索阶段"

**发现4：训练演化轨迹**

通过分析 Pythia 在不同训练阶段的检查点（步骤 0 到 143000），发现三个阶段：
1. **初始阶段**（step 0-128）：高频率选择性（~0.76），低频谱熵（~2.29），头未分化
2. **探索阶段**（step 512-1000）：频率选择性降至最低（0.230），频谱熵升至最高（3.522），模型积极探索
3. **专业化阶段**（step 5000+）：频谱熵逐渐降低，频率选择性恢复，模型精细化并巩固表示

### 消融实验（位置编码对比）

| 模型 | PE 类型 | 尺度敏感度 (0.5x) | 频率选择性 | 频谱熵 | ρ |
|------|---------|----------|---------|------|-----|
| LLaMA-3.2 | RoPE | **0.038** | 0.728 | 2.425 | -0.502 |
| flan-t5 | T5 (相对偏置) | 0.627 | 0.704 | 2.696 | -0.790 |
| BERT | 绝对 PE | 0.507 | 0.743 | 2.449 | -0.606 |
| GPT-2 | 无显式 PE | 0.141 | 0.514 | 2.868 | -0.672 |

**RoPE 独有的优势**：尺度敏感度远低于其他方案。T5 和 BERT 的显式编码过于刚性，GPT-2 虽在无 PE 下达到中等尺度不变性，但通过频谱扩散（高熵低选择性）的方式实现，缺乏 RoPE 的精确性。

## 亮点与洞察

1. **信号处理视角的深刻类比**：将注意力头类比为小波基函数，multi-head attention 类比为小波帧，提供了理解 Transformer 内部机制的全新视角
2. **涌现性质而非设计**：类小波特性并非人为设计，而是训练过程中自发涌现的——模型为克服 RoPE 的理论限制而发展出的最优策略
3. **RoPE 的独特地位**：首次用实验证明了 RoPE 在催化多分辨率策略方面的独特性
4. **训练动态的相变现象**：发现了明确的"探索→专业化"训练阶段转变，类似于物理系统的相变
5. **重建误差极低**（~10⁻⁷），验证了小波分析框架在注意力模式上的适用性

## 局限性

1. **推理时分析**：主要在训练完成后的推理时进行分析，缺乏训练过程中的连续细粒度追踪
2. **模型范围有限**：仅分析了开源英文文本模型，是否适用于视觉、音频等其他模态或多语言场景尚不确定
3. **因果关系待明确**：虽然观察到类小波特性与 RoPE 的关联，但具体的因果机制（为什么 RoPE 催化了这种涌现）缺乏严格证明
4. **实际应用指导有限**：虽然提出了预初始化头部为小波基、基于频率的剪枝等方向，但缺乏实验验证

## 相关工作

- **位置编码系列**：Sinusoidal PE (Vaswani, 2017), ALiBi (Press et al., 2021), T5 相对 PE (Raffel et al., 2020), RoPE (Su et al., 2024)
- **信号处理视角的网络分析**：激活函数的高阶谐波 (Selesnick and Burrus, 1998), 信息瓶颈理论 (Tishby and Zaslavsky, 2015)
- **RoPE 理论分析**：Barbero et al. (2024) 关于 RoPE 旋转编码的理论限制

## 评分

⭐⭐⭐⭐ (4/5)

高度原创的分析性工作，将信号处理理论与 Transformer 机制分析巧妙结合。实验覆盖了多种模型规模和架构，消融实验（PE 方案对比）设计精到。但工作偏分析性，对模型设计和训练的实际指导意义有待进一步开发。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Online Sparsification of Bipartite-Like Clusters in Graphs](../../ICML2025/others/online_sparsification_of_bipartite-like_clusters_in_graphs.md)
- [\[ICLR 2026\] Dynamical properties of dense associative memory](../../ICLR2026/others/dynamical_properties_of_dense_associative_memory.md)
- [\[ACL 2025\] Implicit Reasoning in Transformers is Reasoning through Shortcuts](implicit_reasoning_in_transformers_is_reasoning_through_shortcuts.md)
- [\[ACL 2025\] Enhancing Transformers for Generalizable First-Order Logical Entailment](enhancing_fol_entailment.md)
- [\[NeurIPS 2025\] Out-of-distribution Generalisation is Hard: Evidence from ARC-like Tasks](../../NeurIPS2025/others/out-of-distribution_generalisation_is_hard_evidence_from_arc-like_tasks.md)

</div>

<!-- RELATED:END -->
