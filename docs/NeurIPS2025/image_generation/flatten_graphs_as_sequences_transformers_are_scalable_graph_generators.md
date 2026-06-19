---
title: >-
  [论文解读] Flatten Graphs as Sequences: Transformers are Scalable Graph Generators
description: >-
  [NeurIPS 2025][图像生成][graph generation] 提出 AutoGraph，通过分段欧拉邻域路径（SENT）将图无损展平为 token 序列，直接用 decoder-only Transformer 建模，实现比扩散模型快 100× 的图生成速度，同时在合成和分子基准上达到 SOTA。
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "graph generation"
  - "autoregressive model"
  - "Eulerian trail"
  - "language model"
  - "scalability"
---

# Flatten Graphs as Sequences: Transformers are Scalable Graph Generators

**会议**: NeurIPS 2025  
**arXiv**: [2502.02216](https://arxiv.org/abs/2502.02216)  
**代码**: [AutoGraph](https://github.com/BorgwardtLab/AutoGraph)  
**领域**: 图像生成  
**关键词**: graph generation, autoregressive model, Eulerian trail, language model, scalability

## 一句话总结

提出 AutoGraph，通过分段欧拉邻域路径（SENT）将图无损展平为 token 序列，直接用 decoder-only Transformer 建模，实现比扩散模型快 100× 的图生成速度，同时在合成和分子基准上达到 SOTA。

## 研究动机

图生成是药物发现、蛋白质设计、程序合成等的基础任务。现有方法的核心瓶颈：

- **扩散模型**：需要全邻接矩阵操作（$O(n^2)$ 内存），每步计算谱特征（$O(n^3)$），严重限制可扩展性
- **传统自回归模型**（GraphRNN 等）：序列不由 token 组成，需要特殊架构（RNN），无法利用 LLM 的强大能力；且序列前缀不保证是诱导子图，丢失结构信息
- 图生成领域未能像文本/图像那样受益于 Transformer 和大规模预训练的进步

### 解决思路

**本文目标**：### 核心概念：分段欧拉邻域路径（SENT）

**分段欧拉路径（SET）**：将图分解为多段 trail，每条边恰好被访问一次。


## 方法详解

### 核心概念：分段欧拉邻域路径（SENT）

**分段欧拉路径（SET）**：将图分解为多段 trail，每条边恰好被访问一次。但 SET 的前缀生成的图仅是子图而非诱导子图，丢失结构信息。

**分段欧拉邻域路径（SENT）**：在 SET 基础上为每个节点附加邻域信息。每个元组 $w_i = (v_i, A_i)$ 包含节点 $v_i$ 和邻域集 $A_i \subseteq V$。

**关键定理**：若 SENT $s$ 是因果（causal）且半哈密顿（semi-hamiltonian）的，则其任意前缀的生成图都是 $G_s$ 的诱导子图。这建立了图前缀与语言建模中子句的直接对应。

### 充要条件（Theorem 2.15）

SENT $s$ 在图 $G$ 中是因果且哈密顿的，当且仅当每个元组 $w = (v, A_v)$ 满足：

$$A_v = \mathcal{N}_G(v) \cap V_s(w)$$

即邻域集恰好是 $v$ 的图邻居中已被访问的节点。

### 采样算法

基于随机路径采样（Algorithm 1）：
1. 随机选择起始节点，初始化路径
2. 若当前节点的未访问邻居非空，随机选择一个继续路径
3. 否则断开当前段，随机选择新的未访问节点开始新段
4. 对每个新节点计算其与已访问节点的邻域集

时间和空间复杂度均为 $\mathcal{O}(m)$，$m$ 为边数——**最优线性复杂度**。

### Tokenization

1. **重索引**：按节点首次出现顺序重新编号，保证排列不变性
2. **特殊 token**：`/` 表示段间断开，`<` 和 `>` 标记邻域集的起止
3. 属性图扩展：节点/边属性以交错方式插入 token 序列

### 自回归建模

标准语言模型目标函数：

$$p(s) = \sum_{i=1}^{n} \log p_\theta(s_i \mid s_1, \ldots, s_{i-1})$$

使用 LLaMA 架构（12 层，hidden dim=768），与 GPT-2 smallest 对齐。推理时使用 top-k 采样。

## 实验关键数据

### 合成图生成（Planar & SBM）


### 主实验

| 模型 | Planar Ratio↓ | Planar VUN↑ | SBM Ratio↓ | SBM VUN↑ |
|------|-------------|-------------|-----------|----------|
| DiGress | 6.1% | 77.5% | 1.8 | 60.0% |
| GruM | 1.8 | 90.0% | 1.5 | 85.0% |
| GEEL | 9.5% | 0.0 | 7.3% | 5.0% |
| **AutoGraph** | **1.5** | **87.5%** | **3.4** | **92.5%** |

- SBM 数据集上 VUN（Valid+Unique+Novel）最高
- 仅 AutoGraph 和 GruM 在两个数据集上均达到 VUN≥80

### 效率对比


### 消融实验

| 模型 | 推理 (s/graph) | 训练 (h) |
|------|---------------|---------|
| DiGress | 7.53% | 5.56% |
| GRAN | 1.25 | 12.3% |
| **AutoGraph** | **0.08** | **1.57** |

推理速度比扩散模型快 ~100×，训练快 ~3×。

### 大规模图（Proteins & Point Clouds）

| 模型 | Proteins Ratio↓ | Point Clouds Ratio↓ |
|------|----------------|---------------------|
| GRAN | 77.7% | 19.1% |
| DiGress | OOM | OOM |
| **AutoGraph** | 接近训练集 | 可扩展到 5000+ 节点 |

扩散模型在大图上 OOM，AutoGraph 因线性复杂度可扩展。

### 分子生成

在 QM9 和 MOSES 分子基准上均达到 SOTA 或竞争性能，同时支持**子结构条件生成**（无需微调，类似 prompt）和**迁移学习**能力。

## 评分

⭐⭐⭐⭐⭐

**优点**：
- 理论优美：SENT 的设计建立了图生成与语言建模的严格对应，前缀 = 诱导子图
- 复杂度最优：序列长度和采样复杂度关于边数线性，远优于扩散模型的二次/三次复杂度
- 实践价值巨大：直接使用现有 LLM 基础设施（LLaMA + HuggingFace）做图生成
- 附带子结构条件生成和迁移学习能力，为图基础模型提供了可行路径
- 100× 推理加速和 3× 训练加速使得大规模图生成变得实际可行

**局限**：
- 随机路径采样引入的序列多样性增加了学习难度，可能需要更多训练数据
- 目前仅在无向图上验证，有向图、超图等需要进一步扩展
- 在极小图（节点<10）上 VUN 可能不如扩散模型
- 价值: 


## 亮点与洞察
- 方法设计简洁有效，核心思路清晰
- 实验验证全面，消融分析充分
- 对领域的关键问题提供了新的解决思路


## 局限与展望
- 方法在特定条件下可能存在局限性，泛化性待进一步验证
- 计算效率和可扩展性可做进一步优化
- 与更多相关方法的结合值得探索


## 相关工作与启发
- **vs 同领域代表性方法**：本文在方法设计上有独特贡献，与现有方法形成互补
- **vs 传统方法**：相比传统方案，本文方法在关键指标上取得了显著提升
- **启发**：本文的技术路线对后续相关工作有重要参考价值


## 研究背景与动机

### 领域现状
**领域现状**：本研究关注的领域近年来取得了显著进展，多种方法被提出解决核心问题。

### 现有痛点
**现有痛点**：现有方法在效果、效率或泛化性方面存在明显不足，限制了实际应用。

**核心矛盾**：在性能与效率之间存在根本性权衡，需要在方法论层面进行创新突破。

### 研究目标
**本文目标**：提出新的方法框架解决上述核心挑战。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Scalable GANs with Transformers](../../ICML2026/image_generation/scalable_gans_with_transformers.md)
- [\[NeurIPS 2025\] Graph Distance as Surprise: Free Energy Minimization in Knowledge Graph Reasoning](graph_distance_as_surprise_free_energy_minimization_in_knowledge_graph_reasoning.md)
- [\[ICCV 2025\] Video Motion Graphs](../../ICCV2025/image_generation/video_motion_graphs.md)
- [\[NeurIPS 2025\] Exploring Variational Graph Autoencoders for Distribution Grid Data Generation](exploring_variational_graph_autoencoders_for_distribution_grid_data_generation.md)
- [\[NeurIPS 2025\] Scaling Diffusion Transformers Efficiently via μP](scaling_diffusion_transformers_efficiently_via_μp.md)

</div>

<!-- RELATED:END -->
