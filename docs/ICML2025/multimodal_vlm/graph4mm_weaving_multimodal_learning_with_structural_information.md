---
title: >-
  [论文解读] Graph4MM: Weaving Multimodal Learning with Structural Information
description: >-
  [ICML2025][多模态VLM][多模态图] 提出 Graph4MM 框架，通过 Hop-Diffused Attention 将多跳图结构信息注入自注意力机制，并设计 MM-QFormer 实现跨模态融合，在生成和判别任务上平均提升 6.93%。 现实世界的多模态数据通常具有超越简单一对一映射（如图文对）的复杂结构关系…
tags:
  - "ICML2025"
  - "多模态VLM"
  - "多模态图"
  - "结构信息融合"
  - "注意力机制"
  - "QFormer"
  - "零样本分类"
---

# Graph4MM: Weaving Multimodal Learning with Structural Information

**会议**: ICML2025  
**arXiv**: [2510.16990](https://arxiv.org/abs/2510.16990)  
**代码**: [GitHub](https://github.com/YennNing/Graph4MM)  
**领域**: 多模态VLM  
**关键词**: 多模态图, 结构信息融合, Hop-Diffused Attention, QFormer, 零样本分类

## 一句话总结

提出 Graph4MM 框架，通过 Hop-Diffused Attention 将多跳图结构信息注入自注意力机制，并设计 MM-QFormer 实现跨模态融合，在生成和判别任务上平均提升 6.93%。

## 研究背景与动机

现实世界的多模态数据通常具有超越简单一对一映射（如图文对）的复杂结构关系。例如在学术论文中，图像与其标题是直接配对关系，但图像与后续章节内容、页面摘要之间的关系是非线性的、多层次的。现有 VLM（如 BLIP2、Qwen2-VL）仍局限于建模一对一的图文关系，无法捕捉复杂的多模态交互。

先驱工作 MMGL 虽然将模态数据建模为图，但存在两个关键缺陷：

**邻居无差别对待**：简单拼接邻居的多模态数据，不区分不同距离（hop）的节点重要性

**图作为独立模态**：将图拓扑结构作为独立模态与文本/视觉并行注入，但由于预训练语言/视觉模型的特征空间已高度对齐，图嵌入反而引入语义鸿沟，导致性能下降

作者的核心洞察是：**图结构不应作为独立模态，而应作为指导模态内/模态间交互的结构先验**。

## 方法详解

### 多模态图建模

定义多模态图 $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{T}, \mathcal{P})$，其中每个节点 $v_i$ 包含可选的文本属性 $t_{v_i}$ 和视觉属性 $p_{v_i}$。边分为三类：文本-文本、图像-图像、文本-图像。对于目标节点，根据 $\tau$-hop 邻居抽取文本子图 $\mathcal{G}_t$ 和视觉子图 $\mathcal{G}_p$。

### Hop-Diffused Attention

这是本文核心创新，将多跳图结构信息融入自注意力机制，分三步：

**Step 1: 自注意力计算**。对视觉嵌入 $\mathbf{H}_P \in \mathbb{R}^{|\mathcal{V}_p| \times d}$，计算标准注意力矩阵：

$$\mathbf{A}'_{i,j} = \text{Softmax}_j\left(\frac{\mathbf{q}_{v_i}^\top \mathbf{k}_{v_j}}{\sqrt{d}}\right)$$

**Step 2: 因果掩码**。根据图的边集 $\mathcal{E}_p$ 定义掩码 $\mathbf{M}_{i,j}$，仅允许相连节点之间的注意力，使注意力与图拓扑对齐：

$$\mathbf{A}_{i,j} = \text{Softmax}(\mathbf{M}_{i,j} \cdot \mathbf{A}'_{i,j})$$

**Step 3: 扩散机制**。通过迭代传播注意力捕获多跳结构信息：

$$\boldsymbol{\mathcal{A}} = \sum_{i=0}^{\infty} \theta_i \mathbf{A}^i, \quad \theta_i = \alpha(1-\alpha)^i, \quad \alpha \in (0,1)$$

其中 $\theta_i$ 为指数衰减系数，$\alpha$ 控制远距离邻居的影响力。最终通过残差连接更新嵌入：

$$\mathbf{H}_P \leftarrow \mathbf{H}_P + \boldsymbol{\mathcal{A}} \mathbf{H}_P$$

**理论保证**：作者证明 Hop-Diffused Attention 比堆叠 $k$ 层 GAT 保留更高的 Dirichlet Energy，即 $\mathcal{E}_{\text{Hop-Diffused}}(\mathbf{X}^{(1)}) > \mathcal{E}_{\text{GAT}}(\mathbf{X}^{(k)})$，有效缓解过平滑问题。

**轻量替代：Hop-Aware Attention**。为降低计算复杂度（从 $O(|\mathcal{V}_p| \cdot d^2)$ 降至 $O(|\mathcal{V}_p| \cdot d)$），引入可学习的 hop 嵌入 $\mathbf{h}_{\text{hop}}^{(h)}$，直接加到节点嵌入上，让下游模型自适应学习不同 hop 信息的重要性。

### MM-QFormer（多映射查询变换器）

受 BLIP2 的 Q-Former 启发，设计用于跨模态融合的模块：

1. **共享自注意力**：将可学习查询 token $\mathbf{Q}_v^{(0)}$ 与文本嵌入 $\mathbf{H}_T$ 拼接，通过共享自注意力让查询 token 感知文本上下文
2. **跨模态交叉注意力**：更新后的查询 token 作为 query，视觉嵌入 $\mathbf{H}_P$ 作为 key/value，提取与文本相关的视觉特征
3. **前馈网络**：两层全连接网络进一步加工查询 token

经过 $L$ 层后，最终查询 token 作为多模态 token 插入文本属性 token 之后，送入冻结的预训练语言模型生成输出。

### 训练损失

模型采用标准的自回归语言建模损失，冻结视觉编码器和 LLM，仅训练 Hop-Diffused Attention 模块和 MM-QFormer 中的参数。

## 实验关键数据

### 数据集

- **WikiWeb2M**（生成任务）：文档章节摘要生成，包含页面描述、章节文本、图像、标题等多模态网页内容
- **Ele-Fashion**（判别任务）：产品零样本分类，节点表示产品，边表示共购关系

### 主实验结果（OPT-125M backbone）

| 方法 | BLEU-4 | ROUGE-L | CIDEr | Acc(%) |
|------|--------|---------|-------|--------|
| BLIP2 (Subgraph Text) | 0.0000 | 0.0530 | 0.0063 | 31.37 |
| Qwen2-VL (Subgraph Text) | 0.0000 | 0.1192 | 0.0084 | 12.33 |
| MMGL (Subgraph T&I) | 0.0778 | 0.4041 | 0.7712 | 99.85 |
| MMGL (Subgraph T&I+GNN) | 0.0633 | 0.3814 | 0.6326 | 70.89 |
| **Graph4MM Hop-Diffused** | **0.0800** | **0.4076** | **0.7831** | **100.00** |

### LLaMA-1B backbone 结果

| 方法 | BLEU-4 | ROUGE-L | CIDEr | Acc(%) |
|------|--------|---------|-------|--------|
| MMGL (Subgraph T&I) | 0.1157 | 0.4685 | 1.1072 | 98.07 |
| **Graph4MM Hop-Diffused** | **0.1177** | **0.4713** | **1.1221** | **100.00** |

### 消融实验（OPT-125M，生成任务）

| 变体 | BLEU-4 | ROUGE-L | CIDEr |
|------|--------|---------|-------|
| Hop-Diffused MM-QFormer（完整） | 0.0800 | 0.4076 | 0.7831 |
| 移除文本子图结构 | 0.0786 | 0.4065 | 0.7765 |
| 移除图像子图结构 | 0.0769 | 0.4044 | 0.7684 |

关键发现：移除图像模态的结构信息导致更显著的性能下降，因为文本可通过提示词（如"来自 1-hop 邻居的上下文"）保留部分结构信息，而图像则无此途径。

## 亮点与洞察

1. **重新审视图在多模态学习中的角色**：理论+实证证明图结构不应作为独立模态注入（如 MMGL 的 GNN 方式导致性能下降），而应作为引导模态交互的结构先验
2. **Hop-Diffused Attention 的理论保证**：通过 Dirichlet Energy 分析证明其避免过平滑，优于堆叠多层 GNN，且仅用单层即可捕获多跳信息
3. **小模型打败大模型**：Graph4MM 使用 OPT-125M/LLaMA-1B 等小模型，在引入结构信息后超越了 BLIP2-OPT-2.7B 和 Qwen2-VL-7B 等大模型
4. **Hop-Aware 作为轻量替代**：提供了计算复杂度从 $O(d^2)$ 降至 $O(d)$ 的替代方案，性能接近甚至部分超越 Hop-Diffused

## 局限与展望

1. **数据集规模有限**：仅在 WikiWeb2M 和 Ele-Fashion 两个数据集上验证，缺少更大规模/更多样化场景的测试
2. **图构建依赖人工定义**：边的建立依赖预定义规则（如章节层级、共购关系），未探索自动图构建方法
3. **backbone 规模较小**：仅使用 OPT-125M 和 LLaMA-1B，未验证在更大规模 LLM（如 7B+）上的效果和扩展性
4. **扩散步数 $K$ 的选择**：无穷级数截断为有限步，但论文未充分讨论 $K$ 选择对不同图结构的敏感性
5. **缺少与最新多模态图方法的比较**：主要对比 MMGL，未与其他近期多模态图学习方法（如 GraphAdapter 等）比较

## 相关工作与启发

- **MMGL (Yoon et al., 2023)**：首个将多模态数据建模为图的工作，但简单拼接邻居且将图作为独立模态
- **BLIP2 (Li et al., 2023)**：Q-Former 的设计启发了 MM-QFormer，但 BLIP2 仅处理单一图文对
- **Personalized PageRank / APPNP**：扩散机制的理论基础来自 PPR，将其推广到注意力矩阵上
- **启发**：在多模态学习中，结构信息的价值在于"如何引导注意力分配"，而非作为额外特征注入

## 评分

- 新颖性: ⭐⭐⭐⭐ — Hop-Diffused Attention 将 PPR 扩散与注意力掩码结合的设计新颖，重新定义了图在多模态学习中的角色
- 实验充分度: ⭐⭐⭐ — 消融实验充分但数据集仅两个，backbone 规模较小
- 写作质量: ⭐⭐⭐⭐ — 动机清晰，理论分析扎实，符号体系一致
- 价值: ⭐⭐⭐⭐ — 为多模态学习中引入结构信息提供了理论与实践指导

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Learning Optimal Multimodal Information Bottleneck Representations](learning_optimal_multimodal_information_bottleneck_representations.md)
- [\[CVPR 2026\] Information-Theoretic Decomposition for Multimodal Interaction Learning](../../CVPR2026/multimodal_vlm/information-theoretic_decomposition_for_multimodal_interaction_learning.md)
- [\[CVPR 2026\] SO-Bench: A Structural Output Evaluation of Multimodal LLM](../../CVPR2026/multimodal_vlm/so-bench_a_structural_output_evaluation_of_multimodal_llm.md)
- [\[AAAI 2026\] Conditional Information Bottleneck for Multimodal Fusion: Overcoming Shortcut Learning in Sarcasm Detection](../../AAAI2026/multimodal_vlm/conditional_information_bottleneck_for_multimodal_fusion_overcoming_shortcut_lea.md)
- [\[CVPR 2026\] StructXLIP: Enhancing Vision-Language Models with Multimodal Structural Cues](../../CVPR2026/multimodal_vlm/structxlip_enhancing_vision-language_models_with_multimodal_structural_cues.md)

</div>

<!-- RELATED:END -->
