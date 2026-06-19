---
title: >-
  [论文解读] Bridging Graph and State-Space Modeling for Intensive Care Unit Length of Stay Prediction
description: >-
  [NeurIPS 2025 (GenAI for Health Workshop)][图学习][ICU 住院时长预测] 提出 S2G-Net，将 Mamba 状态空间模型的时序编码与多视图图神经网络（GraphGPS）进行双路融合，用于 ICU 住院时长（LOS）预测，在 MIMIC-IV 数据集上全面超越序列模型、图模型和混合基线。
tags:
  - "NeurIPS 2025 (GenAI for Health Workshop)"
  - "图学习"
  - "ICU 住院时长预测"
  - "图神经网络"
  - "状态空间模型"
  - "Mamba"
  - "多视图图"
---

# Bridging Graph and State-Space Modeling for Intensive Care Unit Length of Stay Prediction

**会议**: NeurIPS 2025 (GenAI for Health Workshop)  
**arXiv**: [2508.17554](https://arxiv.org/abs/2508.17554)  
**代码**: [GitHub](https://github.com/ShuqiZi1/S2G-Net)  
**领域**: 医学图像  
**关键词**: ICU 住院时长预测, 图神经网络, 状态空间模型, Mamba, 多视图图

## 一句话总结
提出 S2G-Net，将 Mamba 状态空间模型的时序编码与多视图图神经网络（GraphGPS）进行双路融合，用于 ICU 住院时长（LOS）预测，在 MIMIC-IV 数据集上全面超越序列模型、图模型和混合基线。

## 研究背景与动机

**领域现状**：ICU 住院时长预测对医院资源配置至关重要。现有方法分为时序模型（LSTM、Transformer）和图模型（GCN、GAT），分别捕捉患者时序轨迹和群体间关联。

**现有痛点**：纯时序模型忽略患者间的临床相似性；纯图模型使用单视图静态图，无法建模多模态异构临床特征；Transformer 类图骨干（如 GraphGPS）的二次复杂度在大规模临床数据上难以扩展。

**核心矛盾**：ICU 数据同时具有长程不规则时序依赖和多模态患者间关系，但缺乏统一建模框架。

**本文目标**：如何在一个端到端框架中同时捕获时序动态和群体级关系结构，并保持计算效率和可解释性。

**切入角度**：Mamba SSM 在线性时间内处理长序列，与多视图 GraphGPS 互补——前者建模时间轴上的患者状态演化，后者建模诊断、语义和管理维度上的患者相似性。

**核心 idea**：双路架构 + 多视图图构建 + 用 SSM 替换 GraphGPS 中的 Transformer 全局层。

## 方法详解

### 整体框架
S2G-Net 包含三个分支：(1) 时序编码器（Mamba SSM）处理 48 小时 ICU 时序数据 $\mathbf{X}_i^{TS} \in \mathbb{R}^{T \times d}$；(2) 图编码器（优化后的 GraphGPS）从多视图患者相似图中学习群体关系；(3) 静态特征编码器处理人口统计和入院元数据 $\mathbf{x}_i^{Flat}$。三路特征通过加权拼接融合后送入回归头，输出 LOS 预测。

### 关键设计

1. **多视图患者相似图构建**:

    - 功能：从诊断码（ICD-9/10）和 BERT 语义嵌入两个视图构建患者相似图
    - 核心思路：诊断码视图用 TF-IDF 余弦相似度 / FAISS 近似 KNN / 惩罚共现三种策略计算相似度，取 top-k 邻居构建 $\mathcal{G}_{diag}$；语义视图用 DistilBERT 编码诊断描述，通过高斯核距离 $w_{ij}^{bert} = \exp(-\|\mathbf{b}_i - \mathbf{b}_j\|_2^2 / 2\sigma^2)$ 构建 $\mathcal{G}_{bert}$
    - 设计动机：单视图图无法捕获诊断码之间的语义近似性（如不同编码但含义相近的疾病），多视图融合可覆盖结构化和非结构化两种临床关系

2. **时序编码器（Mamba SSM）**:

    - 功能：从 48 小时多变量时序数据中提取患者状态表征
    - 核心思路：先线性投影 + RMSNorm + GELU 得到 $\mathbf{H}_i^{(0)}$，然后经 $L$ 层堆叠的 Mamba 块，最后用掩码感知池化 $\mathbf{z}_i^{TS} = \text{MaskPool}(\mathbf{H}_i^{(L)}, \mathbf{m}_i)$ 处理缺失值
    - 设计动机：Mamba 的输入依赖递归在线性时间内捕获长程依赖，比 LSTM 和 Transformer 更适合不规则、长序列 ICU 数据

3. **图编码器（local GENConv + global Mamba）**:

    - 功能：在多视图图上融合局部邻域聚合与全局上下文建模
    - 核心思路：堆叠 $L_g$ 个 GraphGPS 块，每块包含 GENConv（利用类型化加权边属性做局部消息传递）和 Mamba 全局编码器（对按度排序的节点序列建模全局依赖），然后 BN + 残差连接：$\mathbf{x}_i^{(\ell+1)} = \mathbf{x}_i^{(\ell)} + \tilde{\mathbf{u}}_i^{(\ell)}$
    - 设计动机：用 Mamba 替换原 GraphGPS 中的 Transformer 层，将全局注意力从 $O(N^2)$ 降至 $O(N)$，同时保持全局上下文建模能力

4. **加权融合与辅助监督**:

    - 三路特征通过 softmax 归一化权重 $\boldsymbol{\lambda}$ 拼接：$\mathbf{z}_i^{fused} = \text{Concat}(\lambda_{Graph}\mathbf{z}_i^{Graph}, \lambda_{TS}\mathbf{z}_i^{TS}, \lambda_{Flat}\mathbf{z}_i^{Flat})$
    - 在 log 域 $\tilde{y}_i = \log(1+y_i)$ 上用 Huber loss 监督，辅以时序分支辅助 loss 促进梯度流动
    - 样本重加权 $w(y_i) = 1 + \gamma \mathbb{I}(y_i > \tau)$ 增加极端 LOS 值的权重

### 损失函数 / 训练策略
$$\mathcal{L}_i = (1-\alpha)\mathcal{L}_{Huber}(\tilde{\hat{y}}_i^{Main}, \tilde{y}_i) + \alpha \mathcal{L}_{Huber}(\tilde{\hat{y}}_i^{TS}, \tilde{y}_i)$$

使用 AdamW 优化，梯度裁剪，基于验证集 $R^2$ 提前停止。超参数用 Optuna 搜索 75 次 trial。

## 实验关键数据

### 主实验
数据集：MIMIC-IV v3.1，65,347 名成人患者，216 个特征（174 个时序 + 42 个静态）。

| 模型 | R² ↑ | Kappa ↑ | MSE ↓ | MSLE ↓ | MAD ↓ | log-MAPE ↓ |
|------|------|---------|-------|--------|-------|------------|
| **S2G-Net** | **0.43±0.01** | **0.42±0.00** | **14.25±0.18** | **0.25±0.01** | **1.88±0.02** | **35.74±1.24** |
| GraphGPS | 0.40±0.01 | 0.40±0.00 | 15.08±0.24 | 0.27±0.02 | 1.93±0.05 | 37.27±1.26 |
| Mamba | 0.33±0.00 | 0.36±0.00 | 16.89±0.04 | 0.28±0.02 | 2.03±0.01 | 40.46±1.03 |
| BiLSTM | 0.31±0.01 | 0.35±0.01 | 17.23±0.16 | 0.28±0.00 | 2.01±0.01 | 42.09±2.38 |
| LSTM-GAT | 0.31±0.01 | 0.35±0.01 | 17.36±0.21 | 0.28±0.00 | 2.01±0.00 | 41.07±1.58 |
| XGBoost | 0.32±0.00 | 0.39±0.00 | 17.32±0.03 | 0.27±0.00 | 1.96±0.01 | 47.14±0.17 |

S2G-Net 在 R² 上比最佳基线 GraphGPS 提升约 7.5%，log-MAPE 降低 4.1%，且差异具有统计显著性（p<0.05）。

### 消融实验

| 配置 | R² ↑ | MSE ↓ | 说明 |
|------|------|-------|------|
| Baseline (48h) | 0.43 | 14.25 | 完整模型 |
| Last 6h | 0.28 | 18.02 | 时间窗口缩短导致 R² 下降 35% |
| Last 24h | 0.40 | 15.09 | 24h 后收益趋于饱和 |
| No Physiology | 0.39 | 15.40 | 去掉生理特征掉幅最大 |
| No Vitals | 0.40 | 15.16 | 生命体征次之 |
| No Ethnicity | 0.42 | 14.21 | 种族特征贡献最小 |
| Static Only | - | 34.69 | 仅静态特征性能极差 |
| Drop 30% edges | 0.43 | 14.35 | 图结构较鲁棒 |
| Drop 70% edges | 0.38 | 15.63 | 边删除超过 70% 后显著下降 |

### 关键发现
- 观察窗口从 6h→24h 提升显著（R² 0.28→0.40），24h→48h 边际递减
- 生理指标和生命体征对预测贡献最大，种族特征信息量最小（SHAP 验证）
- 图结构在删除 30%-50% 边后仍保持稳健，但 70% 后急剧下降
- 模型参数量 <2.5M，计算效率在同等性能模型中最优

## 亮点与洞察
- **多视图图构建**：诊断码 + BERT 语义嵌入 + MST/GDC 增强，覆盖了编码重合、语义近似和长距离连接三种关系维度，思路可迁移到其他临床预测任务
- **SSM 替换 Transformer 做全局层**：在 GraphGPS 中用 Mamba 替换 Transformer 的全局注意力，复杂度从 $O(N^2)$ 降到 $O(N)$，且性能不降反升——说明全体节点间未必需要全连接注意力
- **掩码感知池化**：针对 ICU 时序数据的大量缺失值设计，比简单均值池化更准确

## 局限与展望
- 仅在 MIMIC-IV 单一数据集上验证，缺少跨医院/跨地域的泛化实验
- 图构建阶段使用静态阈值（top-k、底部 30% 裁剪），未考虑动态图演化
- $R^2=0.43$ 作为最终性能，在临床应用中可能仍然不够精确
- 未探索时间窗口动态扩展（online prediction 场景）
- 可以尝试将 Mamba2 或多尺度 SSM 结构引入时序编码器

## 相关工作与启发
- **vs LSTM-GNN**：LSTM-GNN 用 LSTM 编码时序后在静态图上聚合，但时序和图结构是割裂的两阶段处理。S2G-Net 通过共享 Mamba 在两条路径中统一序列建模范式
- **vs GraphGPS**：原始 GraphGPS 的全局层用 Transformer，计算量大；S2G-Net 用 Mamba 替换后在保持全局建模的同时大幅降低复杂度
- **vs XGBoost**：XGBoost 作为非神经基线（R²=0.32）已相当强，说明特征工程在临床预测中仍有竞争力

## 评分
- 新颖性: ⭐⭐⭐⭐ 多视图图构建和 SSM+GNN 双路融合的组合有新意，但各组件均非全新
- 实验充分度: ⭐⭐⭐⭐ 16 个基线对比 + 详细消融 + 可解释性分析，但仅单数据集
- 写作质量: ⭐⭐⭐⭐ 结构清晰，公式推导完整，图表丰富
- 价值: ⭐⭐⭐⭐ 为 ICU 临床预测提供了高效且可解释的统一框架，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] View Space：跨任意图的表示学习](../../ICML2026/graph_learning/view_space_learning_representation_across_arbitrary_graphs.md)
- [\[ICLR 2026\] Graph Tokenization for Bridging Graphs and Transformers](../../ICLR2026/graph_learning/graph_tokenization_for_bridging_graphs_and_transformers.md)
- [\[NeurIPS 2025\] Graph Neural Networks for Efficient AC Power Flow Prediction in Power Grids](graph_neural_networks_for_efficient_ac_power_flow_prediction_in_power_grids.md)
- [\[NeurIPS 2025\] P-DRUM: Post-hoc Descriptor-based Residual Uncertainty Modeling for Machine Learning Potentials](p-drum_post-hoc_descriptor-based_residual_uncertainty_modeling_for_machine_learn.md)
- [\[NeurIPS 2025\] TAMI: Taming Heterogeneity in Temporal Interactions for Temporal Graph Link Prediction](tami_taming_heterogeneity_in_temporal_interactions_for_temporal_graph_link_predi.md)

</div>

<!-- RELATED:END -->
