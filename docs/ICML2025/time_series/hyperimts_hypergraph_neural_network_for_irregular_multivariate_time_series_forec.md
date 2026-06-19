---
title: >-
  [论文解读] HyperIMTS: Hypergraph Neural Network for Irregular Multivariate Time Series Forecasting
description: >-
  [ICML 2025][时间序列][不规则多元时间序列] 提出 HyperIMTS，利用超图结构表示不规则多元时间序列（IMTS）中的观测值和其依赖关系，通过三种消息传递机制（节点→超边、超边→超边、超边→节点）实现不规则性感知的时间和变量依赖学习，在 5 个 IMTS 数据集上达到 SOTA 且计算效率优于 padding 方法。
tags:
  - "ICML 2025"
  - "时间序列"
  - "不规则多元时间序列"
  - "超图神经网络"
  - "时间感知消息传递"
  - "变量依赖建模"
  - "预测"
---

# HyperIMTS: Hypergraph Neural Network for Irregular Multivariate Time Series Forecasting

**会议**: ICML 2025  
**arXiv**: [2505.17431](https://arxiv.org/abs/2505.17431)  
**代码**: [https://github.com/qianlima-lab/PyOmniTS](https://github.com/qianlima-lab/PyOmniTS)  
**领域**: Time Series Forecasting  
**关键词**: 不规则多元时间序列, 超图神经网络, 时间感知消息传递, 变量依赖建模, 预测

## 一句话总结
提出 HyperIMTS，利用超图结构表示不规则多元时间序列（IMTS）中的观测值和其依赖关系，通过三种消息传递机制（节点→超边、超边→超边、超边→节点）实现不规则性感知的时间和变量依赖学习，在 5 个 IMTS 数据集上达到 SOTA 且计算效率优于 padding 方法。

## 研究背景与动机
多元时间序列（MTS）广泛存在于医疗、天气、生物力学等领域。现实中由于传感器故障、采样频率不一致、人为因素等原因，时间序列往往是**不规则的**（IMTS）：各变量的采样时间间隔不等，且不同变量之间的观测时间点不对齐。

现有处理 IMTS 的方法分为两大类，各有痛点：

**Padding（填充）方法**：将不规则序列填充为规则矩阵。包括时间对齐填充和 patch 对齐填充。问题是大幅增加数据量（如 MIMIC-IV 数据集原始平均 304.8 个观测点，填充后膨胀到 92,000），且可能破坏原始采样模式。

**非填充方法（集合/二部图）**：将观测视为集合元素或用二部图表示。问题是集合无法捕捉观测间相关性；二部图在变量间无共享时间戳时无法传播消息（例如变量 V2 和 V3 没有任何时间对齐点，在二部图中完全断开）。

核心矛盾：如何在**不做填充**（保持高效）的前提下，**完整捕捉所有观测之间的时间和变量依赖关系**？

本文的切入点是**超图**（Hypergraph）：超边可以连接任意数量的节点。将每个观测值作为节点，用变量超边连接同一变量的所有观测，用时间超边连接同一时间点的所有观测，这样即使变量之间没有共享时间戳，也能通过超边→超边的消息传递间接交换信息。

## 方法详解

### 整体框架
HyperIMTS 的 pipeline：（1）将 IMTS 样本转化为超图表示，观测值为节点，时间戳和变量分别作为两类超边；（2）通过三级消息传递更新嵌入：节点→超边（聚合观测信息到时间/变量超边）→超边→超边（变量间交互）→超边→节点（将聚合信息传回观测节点）；（3）最终节点嵌入通过线性映射输出预测值。

### 关键设计

1. **高效超图表示（Efficient Hypergraph Representation）**:

    - 功能：将 IMTS 样本无需填充地表示为超图 $\mathcal{G} = \{\mathcal{V}, \mathcal{E}\}$
    - 核心思路：每个观测 $(t_j, z_j, u_j)$ 对应一个节点 $v_j$。定义两类超边：时间超边 $\mathcal{E}_{\text{time}} = \{e_t | t=1,...,T\}$（同一时间戳的所有观测连接到同一超边）和变量超边 $\mathcal{E}_{\text{var}} = \{e_u | u=1,...,U\}$（同一变量的所有观测连接到同一超边）。拓扑由两个关联矩阵 $\mathbf{H}^T \in \mathbb{R}^{M \times T}$ 和 $\mathbf{H}^U \in \mathbb{R}^{M \times U}$ 表示
    - 节点初始化：$\mathbf{V} = \text{ReLU}(\text{FF}_{\text{obs}}(\mathcal{Z}_i))$，时间超边用正弦编码 $\mathbf{E}_{\text{time}} = \sin(\text{FF}_{\text{time}}(T_i))$，变量超边用可学习参数 $\mathbf{E}_{\text{var}} = \text{ReLU}(\mathbf{W}_{\text{var}})$
    - 设计动机：相比填充方法，超图只处理实际存在的观测（M 个节点），而非 $T \times U$ 的完整矩阵；相比二部图，超边可连接任意数量节点，且支持超边间的消息传递

2. **节点→超边消息传递（Node-to-Hyperedge）**:

    - 功能：将观测节点信息聚合到时间超边和变量超边上
    - 核心思路：用多头注意力实现。以时间超边为例，查询 $\mathbf{q}^h = \text{FF}_q(\mathbf{E}_{\text{time}})$，键值由节点嵌入拼接变量超边嵌入生成 $\mathbf{k}^h = \text{FF}_k(\mathbf{V} || \mathbf{E}_{\text{var}})$：
    $\mathbf{O} = ||_{h=1}^{H} \text{Softmax}\left(\frac{\mathbf{q}^h {\mathbf{k}^h}^\intercal}{\sqrt{d/H}}\right) \mathbf{v}^h$
    $\mathbf{E}_{\text{time}}' = \mathbf{O} + \text{ReLU}(\text{FF}_O(\mathbf{O}))$
      变量超边的更新类似，但将时间信息拼接到键值中以区分时间位置
    - 设计动机：拼接另一维度的超边作为键值的一部分，使得时间超边在聚合时能区分不同变量的观测，反之亦然

3. **超边→超边消息传递（Irregularity-Aware Inter-Variable Message Passing）**:

    - 功能：在变量超边之间传递信息，建模变量间依赖关系
    - 核心思路：计算两种变量相似度并自适应融合。对于变量 $u_a$ 和 $u_b$：
        - 整体变量相似度（series-level）：$\mathbf{S}_{\text{var}} = \text{FF}_q(e_{u_a}) \cdot \text{FF}_k(e_{u_b})^\intercal$
        - 时间感知相似度（仅用对齐观测）：$\mathbf{S}_{\text{obs}} = [v_1^{u_a},...,v_{T_{\text{shared}}}^{u_a}] \cdot [v_1^{u_b},...,v_{T_{\text{shared}}}^{u_b}]^\intercal$
        - 自适应融合：$\mathbf{S}_{\text{IMTS}} = \alpha \mathbf{S}_{\text{obs}} + (1-\alpha) \mathbf{S}_{\text{var}}$
        - 其中 $\alpha = T_{\text{shared}} / T_{\text{total}}$（当 $\mathbf{S}_{\text{var}} > \delta$ 且 $\mathbf{S}_{\text{obs}} \neq 0$ 时），否则 $\alpha = 0$
        - 最终通过注意力传播：$\mathbf{E}_{\text{var}}'' = \text{Softmax}(\mathbf{A}_{\text{var}} / \sqrt{d}) \text{FF}_v(\mathbf{E}_{\text{var}}')$
    - 设计动机：当两个变量有较多时间对齐的观测时，基于对齐观测的细粒度相似度更可靠；当完全不对齐（$\mathbf{S}_{\text{obs}}=0$）时，退回到整体变量级别的相似度，保持变量间的连接不断开。$\delta$ 为可学习阈值（初始化 0.5），控制是否使用时间感知相似度

4. **超边→节点消息传递（Hyperedge-to-Node）**:

    - 功能：将超边聚合的信息传递回节点，更新观测嵌入
    - 核心思路：先对节点做自注意力，再与时间和变量超边信息拼接更新：
    $\mathbf{V}' = \text{SelfAtten}(\mathbf{V})$
    $\mathbf{V}'' = \text{ReLU}(\mathbf{V} + \text{FF}_{\text{node}}(\mathbf{V}' || \mathbf{E}_{\text{time}}' || \mathbf{E}_{\text{var}}'))$
      关键设计：在前 $L-1$ 层使用未经变量间消息传递的 $\mathbf{E}_{\text{var}}'$（学习时间依赖/变量内依赖），仅在最后一层使用 $\mathbf{E}_{\text{var}}''$（引入变量间依赖）
    - 设计动机：先充分学习时间维度的模式，最后才引入变量间信息，避免过早引入跨变量噪声

### 损失函数 / 训练策略
- 输出映射：$\hat{\mathcal{Z}_i} = \text{FF}_{\text{out}}(\mathbf{V}'' || \mathbf{E}_{\text{time}}' || \mathbf{E}_{\text{var}}'')$
- 训练损失：标准 MSE，仅在预测位置的观测值上计算
- 学习率策略：前 3 个 epoch 保持不变，之后按 $\mathcal{L}_n = \mathcal{L}_0 \times 0.8^{n-3}$ 指数衰减
- 最大 300 epoch，early stopping patience=10，5 个随机种子取均值和标准差

## 实验关键数据

### 主实验

| 数据集 | 指标(MSE) | HyperIMTS | 之前SOTA (GraFITi) | 提升 |
|--------|-----------|-----------|-------------------|------|
| MIMIC-III | MSE | **0.4259** | 0.4534 | -6.1% |
| MIMIC-IV | MSE | **0.2174** | 0.2454 | -11.4% |
| PhysioNet'12 | MSE | **0.2996** | 0.3060 | -2.1% |
| Human Activity | MSE | **0.0421** | 0.0435 | -3.2% |
| USHCN | MSE | 0.1738 | 0.2026 | -14.2% |

在 5 个数据集中的 4 个上排名第一，USHCN 上 GRU-D (0.1639) 更好但方差大。总计对比了 27 个 baseline（16 个 regular TS 方法 + 11 个 irregular TS 方法），覆盖面非常广。

### 消融实验

| 配置 | MIMIC-III MSE | MIMIC-IV MSE | 说明 |
|------|-------------|-------------|------|
| Complete | 0.4259 | 0.2174 | 完整模型 |
| w/o VE (去除变量超边) | 0.9556 | 0.6293 | 退化最严重，变量建模不可或缺 |
| w/o IAVD (无不规则感知变量依赖) | 0.4466 | 0.2358 | 证实超边到超边消息传递的必要性 |
| rp IAVD (仅用整体相似度) | 0.4317 | 0.2189 | 时间感知相似度有额外贡献 |
| w/o TE (去除时间超边) | 0.4954 | 0.2652 | 时间信息很重要 |
| rp TE (非可学习时间编码) | 0.4403 | 0.2333 | 可学习的时间编码优于固定编码 |

### 关键发现
- **变量超边是最核心的组件**：去除后 MSE 退化约 2 倍（MIMIC-III: 0.4259→0.9556），说明捕捉变量间依赖对 IMTS 至关重要
- **不规则感知的变量相似度优于仅用整体相似度**：rp IAVD vs Complete，说明利用对齐时间点的细粒度相似度确实有价值
- **效率优势明显**：在 MIMIC-III 上 HyperIMTS 训练速度快于所有 padding 方法（Warpformer、GNeuralFlow），内存占用也更低
- **部分 regular TS 方法在 IMTS 上也有竞争力**：如 Crossformer、PatchTST，说明对 IMTS 方法的对比需要包含更广泛的 baseline
- **预训练模型（MOIRAI、PrimeNet）表现不佳**：regular 和 irregular 数据的分布差异可能需要专门的预训练策略

## 亮点与洞察
- 超图表示是处理 IMTS 的自然选择——同一超边连接任意多节点，完美适配不同变量在同一时间点的不同观测数量
- "不规则性感知"的变量相似度计算设计精巧：根据对齐程度自适应权衡时间感知相似度和整体相似度，$\alpha = T_{\text{shared}} / T_{\text{total}}$ 直觉清晰
- 仅在最后一层引入变量间依赖信息的"渐进式"设计有效避免了跨变量噪声的过早干扰
- 构建了统一基准：27 个 baseline、5 个数据集的公平对比，并开源了 PyOmniTS 流水线，为 IMTS 研究提供了标准化工具
- 不需要填充，数据处理效率高：MIMIC-IV 数据集上原始观测 304.8 个 vs 填充后 92,000 个，差距极大

## 局限与展望
- **不支持多模态数据**：医疗 IMTS 数据集中可能包含文本笔记或图像，当前模型无法利用
- **注意力计算资源消耗**：虽然比 padding 方法高效，但 self-attention 的 $O(M^2)$ 复杂度在观测点非常密集时仍可能成为瓶颈
- USHCN 数据集上结果方差较大（0.0078），稳定性可进一步改善
- 超边→超边消息传递仅在变量维度，未显式建模时间超边之间的交互，可能遗漏某些时间模式
- 当前假设变量超边数量 $U$ 固定，对于变量数量动态变化的场景（如物联网传感器上下线）未做讨论

## 相关工作与启发
- **vs GraFITi (bipartite graph)**: GraFITi 用二部图表示 IMTS，但无法在无共享时间戳的变量间传递消息。HyperIMTS 通过超边→超边消息传递解决了这个问题，MSE 降低最高达 11.4%
- **vs tPatchGNN (patch-aligned padding)**: tPatchGNN 仍需填充，在 MIMIC-III 上内存占用远高于 HyperIMTS，且性能也不如
- **vs Warpformer (canonical padding)**: Warpformer 使用全时间对齐填充，训练速度慢且内存大，精度也低于 HyperIMTS
- **vs 常规时序模型 (PatchTST, Crossformer)**: 这些模型不处理不规则性，但通过填充仍有一定竞争力，说明 IMTS 评估需要包含更广泛的 baseline

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将超图用于 IMTS 预测且设计了不规则感知的消息传递，但超图在其他领域已广泛使用
- 实验充分度: ⭐⭐⭐⭐⭐ 27 个 baseline、5 个数据集、详细消融、效率分析、变长分析，极其全面
- 写作质量: ⭐⭐⭐⭐ 图示清晰（特别是 Figure 1 和 Figure 3），问题动机和方法论述连贯
- 价值: ⭐⭐⭐⭐ 提供了一套完整的 IMTS benchmark 框架和数据处理方案，实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] TQNet: Temporal Query Network for Efficient Multivariate Time Series Forecasting](temporal_query_network_for_efficient_multivariate_time_series_forecasting.md)
- [\[ICML 2025\] IMTS is Worth Time × Channel Patches: Visual Masked Autoencoders for Irregular Multivariate Time Series Prediction](imts_is_worth_time_times_channel_patches_visual_masked_autoencoders_for_irregula.md)
- [\[ICLR 2026\] Learning Recursive Multi-Scale Representations for Irregular Multivariate Time Series Forecasting](../../ICLR2026/time_series/learning_recursive_multi-scale_representations_for_irregular_multivariate_time_s.md)
- [\[AAAI 2026\] Revitalizing Canonical Pre-Alignment for Irregular Multivariate Time Series Forecasting](../../AAAI2026/time_series/revitalizing_canonical_pre-alignment_for_irregular_multivariate_time_series_fore.md)
- [\[NeurIPS 2025\] A Graph Neural Network Approach for Localized and High-Resolution Temperature Forecasting](../../NeurIPS2025/time_series/a_graph_neural_network_approach_for_localized_and_high-resolution_temperature_fo.md)

</div>

<!-- RELATED:END -->
