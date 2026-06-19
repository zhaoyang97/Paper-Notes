---
title: >-
  [论文解读] Manifolds and Modules: How Function Develops in a Neural Foundation Model
description: >-
  [NeurIPS 2025 Findings][自监督学习][神经基础模型] 从计算神经科学视角"打开黑箱"分析 SOTA 神经活动基础模型 (FNN)，通过构建解码流形和编码流形发现其各处理模块（编码器、循环、读出）展现出质性不同的表征结构，且与生物视觉系统存在关键差异。 领域现状： 神经基础模型（如 FNN）在拟合生物视…
tags:
  - "NeurIPS 2025 Findings"
  - "自监督学习"
  - "神经基础模型"
  - "编码流形"
  - "解码流形"
  - "可解释性"
  - "小鼠视觉系统"
---

# Manifolds and Modules: How Function Develops in a Neural Foundation Model

**会议**: NeurIPS 2025 Findings  
**arXiv**: [2512.07869](https://arxiv.org/abs/2512.07869)  
**代码**: [GitHub](https://github.com/cajal/fnn) (FNN模型) / [GitHub](https://github.com/dyballa/NeuralEncodingManifolds) (分析工具)  
**领域**: 计算神经科学 / 自监督学习  
**关键词**: 神经基础模型, 编码流形, 解码流形, 可解释性, 小鼠视觉系统

## 一句话总结

从计算神经科学视角"打开黑箱"分析 SOTA 神经活动基础模型 (FNN)，通过构建解码流形和编码流形发现其各处理模块（编码器、循环、读出）展现出质性不同的表征结构，且与生物视觉系统存在关键差异。

## 研究背景与动机

**领域现状**: 神经基础模型（如 FNN）在拟合生物视觉系统方面表现出色，但其黑箱性质限制了对脑功能的理解。

**现有痛点**: 现有评估主要关注单元级输出预测精度，未深入分析模型内部表征是否与生物系统具有对应关系。

**核心矛盾**: 即使模型在预测神经活动方面性能优越，其内部计算机制可能与生物系统截然不同——高性能不等于生物学相关性。

**本文目标**: 系统分析 FNN 各处理阶段的内部表征，评估其与生物视觉系统的相似度。

**切入角度**: 像电生理学家一样"逐神经元"表征每个人工神经元的时序响应特性，构建解码与编码流形。

**核心idea**: 联合使用三种分析技术（解码流形、编码流形、解码轨迹）全方位评估基础模型的生物学可信度。

## 方法详解

### 整体框架

以小鼠视觉系统常用刺激（漂移光栅、光流等 88 种独特序列）驱动 FNN，提取各层 2000 个神经元的响应，构建三种分析表示：
1. **解码流形** (Decoding Manifolds): 试验嵌入神经活动空间
2. **编码流形** (Encoding Manifolds): 神经元嵌入刺激-响应空间
3. **解码轨迹** (Decoding Trajectories): 每个刺激下神经活动的时序演化

### 关键设计

1. **解码流形构建**:

    - **功能**: 在神经活动坐标空间中嵌入试验，每个点代表一次刺激试验
    - **怎么做**: 对刺激-时间平均的活动数据执行 PCA 降维，生成 48 个点（6种刺激 × 8个方向）
    - **意义**: 同一刺激的试验应聚在一起，反映对大脑状态的"可读性"

2. **编码流形构建**:

    - **功能**: 构建神经元在刺激-响应框架中的拓扑关系
    - **怎么做**: 三步流程：(1) 对 3-阶张量 $(N \times S \times T)$ 执行非负张量分解得到神经因子 → (2) 用 IAN 算法在神经编码空间构建加权图 → (3) 扩散映射降维得到流形
    - **意义**: 揭示功能相似神经元的全局组织拓扑

3. **解码轨迹分析**:

    - **功能**: 追踪每个时间步的神经活动演化
    - **量化指标**: "管状性"(tubularity) — 紧密度 (Tightness) 和交叉度 (Crossings)
    - **与生物对比**: 评估人工网络的时序动态是否与小鼠视网膜/V1 相似

### 分类准确率评估

使用 Leave-One-Out 3-NN 和逻辑回归分类器在各层激活上评估刺激分类能力。

## 实验关键数据

### 主实验

各层刺激分类准确率（逻辑回归/3-NN）：

| 层 | Enc1 | Enc2 | Enc4 | Enc8 | Rec | RecOut | Readout | Out |
|----|------|------|------|------|-----|--------|---------|-----|
| LR | 0.59 | 0.62 | 0.66 | 0.74 | 0.89 | 0.90 | 0.88 | 0.77 |
| 3-NN | 0.41 | 0.66 | 0.58 | 0.61 | 0.73 | 0.64 | 0.63 | 0.67 |

**循环模块达到最高分类准确率**(0.89/0.90)，显著超过编码器各层。

### 消融实验（各模块表征特性对比）

| 模块 | 编码流形特征 | 解码轨迹特征 | 与生物对比 |
|------|------------|------------|-----------|
| 编码器 (Enc) | 按特征图高度聚类，存在非选择性"强度臂"(β) | 周期刺激呈环状，缺乏刺激依赖的时序模式 | ❌ 与视网膜差异大 |
| 循环模块 (Rec) | 不同区域展示多样的选择性和时序响应 | 出现刺激依赖的轨迹束，分类能力跃升 | ⚠️ 与 V1 有相似但纠缠度高 |
| 读出模块 (Readout) | 高度断裂，每簇几乎来自单一特征图 | 图内不变性，图间多样性 | ❌ 生物系统同类型内更多变 |

### 关键发现

- **循环模块是关键转折**: 刺激表征在此发生质变——不同时序刺激模式被"推开"，分类能力从 0.74 跳至 0.89
- **编码器存在边缘填充伪影**: 特征图边缘的 padding artifacts 产生非选择性的"强度臂"，影响表征
- **读出模块"非生物学"**: 通过大量自相似特征图的线性组合拟合神经数据，而非生物学可信的机制
- 管状性分析量化：FNN 各层的轨迹"交叉度"显著低于生物数据 (p<0.005)
- 输出层通过线性组合实现连续表征——令人惊讶的是拟合神经活动主要发生在读出模块而非整个网络

## 亮点与洞察

- **三种分析工具的首次联合使用**: 解码流形、编码流形、解码轨迹结合提供多维度比较，超越传统 RSA 等成对/平均比较
- **对基础模型的警示**: 即使预测性能优秀，模型内部机制可能偏离生物学——高性能 ≠ 生物可信
- **循环模块 ≈ 通用表征学习**: 类似自监督基础模型中的均匀性和对齐性，其功能角色可推广理解
- **读出模块的"附加模块"角色**: 拟合神经活动的工作集中在读出模块，暗示该架构将表征学习与数据拟合分离

## 局限与展望

- 仅分析了一个基础模型（FNN），缺少其他视频基础模型的对比
- 实验使用了有限的刺激集合（虽然与训练刺激引发相似激活）
- 仅使用了单一 session/scan 的数据，跨动物/跨 session 验证有限
- 编码流形分析依赖采样策略（2000 神经元，40 特征图），不同采样可能影响结果
- 未探讨如何基于这些发现改进模型架构

## 相关工作与启发

- FNN 模型使用高斯读出 + DenseNet 编码器 + ConvLSTM 循环模块，代表了当前神经基础模型的 SOTA 架构
- 编码流形方法来自 Dyballa 等人的工作，扩散映射保留了数据的内在几何
- 与传统 RSA (Representational Similarity Analysis) 相比，编码/解码流形提供了更丰富的全局拓扑信息
- **启发方向**: 未来神经基础模型可考虑在轻量类视网膜编码器之后直接接循环模块，约束特征维度匹配生物细胞类型多样性

## 评分

- 新颖性: ⭐⭐⭐⭐ 三种分析工具的独特组合应用，视角新颖；但主要贡献在分析而非方法创新
- 实验充分度: ⭐⭐⭐ 仅一个模型，刺激集有限，但分析深度值得肯定
- 写作质量: ⭐⭐⭐⭐ 跨学科写作难度大但处理得当，图表信息密度高
- 价值: ⭐⭐⭐⭐ 对理解神经基础模型的生物学相关性有重要贡献，对未来架构设计有指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] How 'Neural' is a Neural Foundation Model?](../../ICML2026/self_supervised/how_neural_is_a_neural_foundation_model.md)
- [\[NeurIPS 2025\] TabSTAR: A Tabular Foundation Model for Tabular Data with Text Fields](tabstar_a_tabular_foundation_model_for_tabular_data_with_text_fields.md)
- [\[AAAI 2026\] Spikingformer: A Key Foundation Model for Spiking Neural Networks](../../AAAI2026/self_supervised/spikingformer_a_key_foundation_model_for_spiking_neural_networks.md)
- [\[CVPR 2025\] Few-Shot Implicit Function Generation via Equivariance](../../CVPR2025/self_supervised/few-shot_implicit_function_generation_via_equivariance.md)
- [\[ICML 2025\] Griffin: Towards a Graph-Centric Relational Database Foundation Model](../../ICML2025/self_supervised/griffin_towards_a_graph-centric_relational_database_foundation_model.md)

</div>

<!-- RELATED:END -->
