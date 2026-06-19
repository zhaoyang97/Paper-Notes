---
title: >-
  [论文解读] MixANT: Observation-dependent Memory Propagation for Stochastic Dense Action Anticipation
description: >-
  [ICCV 2025][LLM效率][动作预测] 提出 MixANT，通过混合专家方法为 Mamba 的遗忘门（A 矩阵）引入输入依赖性，动态选择上下文相关的 A 矩阵控制时序记忆传播，在 50Salads、Breakfast 和 Assembly101 三个密集动作预测数据集上全面超越 SOTA。
tags:
  - "ICCV 2025"
  - "LLM效率"
  - "动作预测"
  - "Mamba"
  - "混合专家"
  - "状态空间模型"
  - "密集预测"
---

# MixANT: Observation-dependent Memory Propagation for Stochastic Dense Action Anticipation

**会议**: ICCV 2025  
**arXiv**: [2509.11394](https://arxiv.org/abs/2509.11394)  
**代码**: [talalwasim.github.io/MixANT](https://talalwasim.github.io/MixANT/)  
**领域**: LLM效率（序列建模/Mamba架构改进）  
**关键词**: 动作预测, Mamba, 混合专家, 状态空间模型, 密集预测

## 一句话总结

提出 MixANT，通过混合专家方法为 Mamba 的遗忘门（A 矩阵）引入输入依赖性，动态选择上下文相关的 A 矩阵控制时序记忆传播，在 50Salads、Breakfast 和 Assembly101 三个密集动作预测数据集上全面超越 SOTA。

## 研究背景与动机

### 问题定义

随机长期密集动作预测 (Stochastic Long-term Dense Action Anticipation)：给定一段已观察的视频帧序列（占视频总长度 $\alpha$ 比例），逐帧预测未来 $\beta$ 比例的动作标签，且需要生成多个可能的未来预测样本（25 个），以应对未来行为的不确定性。预测时间通常长达数分钟。

### 已有方法的不足

**Transformer 二次复杂度限制**：DiffAnt 使用 Transformer 进行密集预测，但序列长度可达数千帧，二次复杂度成为瓶颈

**Mamba 的 A 矩阵输入无关**：MANTA 使用标准 Mamba 处理长序列取得了较好效果，但 Mamba 仅对三个参数（B、C、Δ）实现了输入依赖，而**控制时序记忆传播的核心参数 A 矩阵仍然是静态的**

$$h_t = \bar{\mathbf{A}} h_{t-1} + \bar{\mathbf{B}} x_t$$

A 矩阵决定了多少过去信息被记忆或遗忘。在动作预测中，不同上下文（准备沙拉 vs 煮咖啡）需要不同的记忆策略，静态 A 矩阵无法适应

**输入序列含零填充**：预测任务中未来帧用零填充，A 矩阵理应能选择性忽略这些零序列，但静态 A 无法做到

**直接使 A 输入依赖的技术难题**：qk 乘法破坏 Mamba 次二次复杂度优势；大 MLP 引入过多参数

### 核心动机

A 矩阵控制隐状态的演化，相当于 RNN 的遗忘门，对序列建模至关重要。不同语义上下文需要不同的遗忘策略。如何在不损失计算效率的前提下让 A 矩阵依赖输入？

**核心 idea**：用混合专家方法维护多个 A 矩阵，通过轻量级路由器根据输入特征选择最相关的 A 矩阵，实现输入依赖且保持计算效率。

## 方法详解

### 整体框架

MixANT 由 $K=15$ 个序列处理块组成。前 $K_0=3$ 个块使用标准双向 Mamba，后 $K_E=12$ 个块使用提出的 MixMamba 块。整体嵌入在扩散模型框架中：从高斯噪声 $\hat{Y}_T$ 出发，经 $T$ 步迭代去噪生成密集预测 $\hat{Y}_0$，推理时使用 DDIM 采样 25 个样本。

### 关键设计

#### 1. **MixMamba 层的 S6+ 算法**

- **功能**：维护 $E=5$ 个专家 A 矩阵 $\{\mathbf{A}_1, \mathbf{A}_2, ..., \mathbf{A}_E\} \in \mathbb{R}^{E \times D \times N}$，根据输入动态选择

- **路由机制**：
$$\gamma(x) = \text{softmax}(W_g \cdot \text{mean}(x))$$

$$\mathbf{A}(x) = \mathbf{A}_{\hat{e}}, \quad \hat{e} = \arg\max_e \gamma_e(x)$$

其中 $W_g \in \mathbb{R}^{D \times E}$ 是可学习的投影矩阵，$\gamma(x) \in \mathbb{R}^{B \times E}$ 是路由向量。注意路由仅基于已观察帧的特征 $F_{t,1:P}^{k-1}$ 计算。

- **设计动机**：计算开销极低（仅一次 mean pooling + 矩阵乘法 + softmax），不影响 Mamba 的次二次复杂度。通过 argmax 硬选择确保每次只使用一个 A 矩阵，避免混合多个 A 的计算开销。

#### 2. **混合架构设计（前静态后混合）**

- **功能**：前 3 个块使用标准 Mamba，后 12 个块使用 MixMamba
- **核心思路**：早期层提取通用低级特征，适合统一处理；后期层需要根据语义上下文做不同决策，适合专家路由
- **设计动机**：过早引入路由会在模型尚未提取有意义特征时就强迫专门化，导致性能下降。消融实验证实 $K_0 = 3$ 为最优，过多或过少静态块都不好

#### 3. **统一路由器配置**

- **功能**：MixMamba 层中的前向和后向 MixSSM 单元共享同一个路由向量 $\gamma$
- **核心思路**：计算一个路由向量后，前向选择 $\mathbf{A}_{\hat{e}}$，后向自动获得对应的 $\overleftarrow{\mathbf{A}}_{\hat{e}}$
- **设计动机**：独立路由会破坏 SSM 的双向性——前向和后向应学习同一个 A 矩阵的两个方向，而非两个不同的 A 矩阵

### 损失函数 / 训练策略

$$\mathcal{L}_{total} = (1 - \lambda_{lb}) \mathcal{L}_{rec} + \lambda_{lb} \cdot \mathcal{L}_{lb}$$

- **重建损失**：$\mathcal{L}_{rec} = \|Y - \hat{Y}_0\|^2$（预测与 one-hot 真实标签的 L2 损失）
- **负载均衡损失**：

$$\mathcal{L}_{lb} = \sum_{k=K_0+1}^{K} \text{KL}\left(\frac{C^k}{\sum_e C^k_e} \Big\| \mathcal{U}(E)\right)$$

鼓励所有专家被均匀使用，其中 $C^k_e = \sum_{b=1}^B \gamma^k_e(F_{t,1:P}^{k-1}(b))$ 记录批次内每个专家的使用权重。训练时使用扩散过程采样噪声步 $t$，推理时使用 DDIM 采样 25 个预测样本。

## 实验关键数据

### 主实验

Breakfast 数据集（$\alpha = 0.2$）：

| 方法 | Mean MoC ($\beta$=0.1) | Mean MoC ($\beta$=0.5) | Top-1 ($\beta$=0.1) | Top-1 ($\beta$=0.5) |
|------|:---:|:---:|:---:|:---:|
| UAAA | 15.7 | 13.0 | 28.9 | 28.0 |
| DiffAnt | 24.7 | 22.3 | 31.3 | 30.1 |
| GTDA | 24.0 | 20.6 | 51.2 | 45.0 |
| MANTA | 27.7 | 23.8 | 55.5 | 46.9 |
| **MixANT** | **29.6** | **25.0** | **57.1** | **48.4** |

Assembly101 数据集（$\alpha = 0.2$, 202 类动作）：

| 方法 | Mean MoC ($\beta$=0.1) | Top-1 ($\beta$=0.1) |
|------|:---:|:---:|
| GTDA | 6.4 | 18.0 |
| MANTA | 6.7 | 16.9 |
| **MixANT** | **8.0** | **20.3** |

50Salads 数据集（$\alpha = 0.2$）：

| 方法 | Mean MoC ($\beta$=0.1) | Top-1 ($\beta$=0.1) |
|------|:---:|:---:|
| MANTA | 28.6 | 68.3 |
| **MixANT** | **30.3** | **71.5** |

MixANT 在三个数据集、所有观察比例和预测时长设定下几乎全面超越所有已有方法。

### 消融实验

| 配置 | Mean MoC | Top-1 MoC | 说明 |
|------|:---:|:---:|------|
| E=1 (=标准 Mamba) | 27.7 | 55.5 | 基线 |
| E=3 | 28.8 | 56.4 | 增加专家提升明显 |
| **E=5** | **29.6** | **57.1** | **最优** |
| E=8 | 28.9 | 55.8 | 过多专家性能下降 |
| $K_0=0$ (全部 MixMamba) | 28.4 | 55.9 | 过早路由有害 |
| **$K_0=3$** | **29.6** | **57.1** | **最优** |
| $K_0=6$ | 28.7 | 56.2 | 过多静态块限制容量 |
| 独立路由 | 28.5 | 55.7 | 破坏双向性 |
| **统一路由** | **29.6** | **57.1** | **保持双向一致性** |
| 无负载均衡 | 28.6 | 56.2 | 部分专家欠训练 |
| **有负载均衡** | **29.6** | **57.1** | **均匀使用所有专家** |

### 关键发现

- **A 矩阵输入依赖性对预测任务至关重要**：从 E=1 到 E=5，Mean MoC 提升 1.9%，Top-1 提升 1.6%，仅通过改变 A 矩阵的选择方式即获得显著提升
- **专家数量存在最优点**：E=5 为最佳，过多专家（E=8）导致训练信号稀疏（每个专家训练不充分）
- **负载均衡损失的重要性**：无此约束时第 2 个专家被选择近 50%，E1/E4 几乎不被选择，造成容量浪费
- **专家选择模式揭示语义结构**：t-SNE 可视化显示，尽管训练时仅用原子动作监督，专家选择模式自发地按高级活动类别聚类（如"做沙拉"和"泡茶"选择不同的专家组合），说明 A 矩阵确实学到了语义感知的记忆策略
- **Assembly101 上相对提升最大**：在 202 类的复杂数据集上 Mean MoC 相对提升约 20-32%，说明更复杂的任务更受益于输入依赖的记忆控制

## 亮点与洞察

1. **精准定位 Mamba 的关键弱点**：通过理论分析和实验证明，A 矩阵的输入独立性是 Mamba 在长期预测任务中的瓶颈，这一发现对 Mamba 架构的改进有普遍启示意义
2. **MoE 用于 SSM 参数而非 MLP**：不同于已有工作将 MoE 应用于 Mamba 块外部的 MLP，本文首次将混合专家引入 Mamba 块内部的核心参数 A 矩阵
3. **专家选择的可解释性**：t-SNE 的聚类结果提供了 A 矩阵确实学到语义感知记忆策略的强证据，增强了方法的可信度
4. **零额外推理开销**：路由器计算量极小（mean + matmul + softmax），且 argmax 硬选择意味着每层仍只用一个 A 矩阵，与标准 Mamba 推理复杂度相同

## 局限与展望

1. **仅在动作预测任务验证**：A 矩阵的 MoE 方法是否在视频理解、语言建模等其他 Mamba 应用中也有效尚未验证
2. **硬选择路由**：argmax 导致梯度不可微，训练依赖路由向量的 softmax 概率传梯度，soft MoE 可能效果不同
3. **固定专家数量**：所有 MixMamba 层使用相同数量的专家，自适应专家数可能更优
4. **路由器设计简单**：仅用 mean pooling + 线性投影，更复杂的路由策略（如基于注意力的路由）可能进一步提升
5. **扩散模型框架的开销**：主要贡献在于 MixMamba 层，但整体方法依赖 25 步 DDIM 采样和扩散训练，计算开销仍然较大

## 相关工作与启发

- MANTA 证明了 Mamba 优于 Transformer 和卷积架构用于密集动作预测，本文在此基础上指出并解决 A 矩阵的局限
- BlackMamba、MoE-Mamba 等工作将 MoE 用于 Mamba 的 MLP 层，本文首次将 MoE 引入 A 矩阵本身
- Mamba-2 的 SSD 框架通过结构化矩阵约束提供了另一种思路，值得与本文方法对比

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首次提出 A 矩阵的混合专家方法，问题定位精准，解决方案简洁优雅
- **实验充分度**: ⭐⭐⭐⭐ — 三个数据集全面评测 + 多维度消融（专家数/静态块数/路由配置/负载均衡）+ 专家选择可视化分析
- **写作质量**: ⭐⭐⭐⭐ — 问题动机清晰，从 Mamba 的参数输入依赖性切入自然流畅
- **价值**: ⭐⭐⭐⭐ — 对 Mamba 架构的改进有普遍意义，不仅限于动作预测任务

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Stochastic Sparse Attention for Memory-Bound Inference](../../ICML2026/llm_efficiency/stochastic_sparse_attention_for_memory-bound_inference.md)
- [\[ACL 2025\] DIVE into MoE: Diversity-Enhanced Reconstruction of Large Language Models from Dense into Mixture-of-Experts](../../ACL2025/llm_efficiency/dive_moe_reconstruction.md)
- [\[NeurIPS 2025\] Hardware-aligned Hierarchical Sparse Attention for Efficient Long-term Memory Access](../../NeurIPS2025/llm_efficiency/hardware-aligned_hierarchical_sparse_attention_for_efficient_long-term_memory_ac.md)
- [\[ICML 2026\] DOT-MoE: 用可微 optimal transport 把 dense LLM 转成 MoE](../../ICML2026/llm_efficiency/dot-moe_differentiable_optimal_transport_for_moefication.md)
- [\[ACL 2025\] EpMAN: Episodic Memory AttentioN for Generalizing to Longer Contexts](../../ACL2025/llm_efficiency/epman_episodic_memory_attention_for_generalizing_to_longer_contexts.md)

</div>

<!-- RELATED:END -->
