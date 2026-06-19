---
title: >-
  [论文解读] DECOR: Deep Embedding Clustering with Orientation Robustness
description: >-
  [AAAI 2026 (KGML Bridge, non-archival)][深度聚类] 提出 DECOR 框架，通过旋转不变的等变卷积自编码器（RCAE）+ 非参数聚类（DeepDPM）+ 集成异常检测，实现晶圆图缺陷模式的方向鲁棒聚类。 半导体制造中，晶圆缺陷的早期检测对产品良率至关重要。然而晶圆质量测试的原始数据面临…
tags:
  - "AAAI 2026 (KGML Bridge, non-archival)"
  - "深度聚类"
  - "旋转不变性"
  - "晶圆缺陷检测"
  - "非参数聚类"
  - "异常检测"
---

# DECOR: Deep Embedding Clustering with Orientation Robustness

**会议**: AAAI 2026 (KGML Bridge, non-archival)  
**arXiv**: [2510.03328](https://arxiv.org/abs/2510.03328)  
**代码**: 无  
**领域**: 其他  
**关键词**: 深度聚类, 旋转不变性, 晶圆缺陷检测, 非参数聚类, 异常检测

## 一句话总结

提出 DECOR 框架，通过旋转不变的等变卷积自编码器（RCAE）+ 非参数聚类（DeepDPM）+ 集成异常检测，实现晶圆图缺陷模式的方向鲁棒聚类。

## 研究背景与动机

半导体制造中，晶圆缺陷的早期检测对产品良率至关重要。然而晶圆质量测试的原始数据面临多个挑战：

**数据复杂且无标签**：晶圆图模式复杂，且通常缺乏人工标注

**类别不平衡**：不同缺陷类型的频率差异巨大

**多缺陷叠加**：单个晶圆可能同时包含多种缺陷模式

**方向变异性**：由于晶圆放置或处理过程中的变化，相同的缺陷模式可能以不同旋转角度出现。标准聚类方法把旋转的实例视为不同模式，导致聚类碎片化

传统聚类方法（K-Means、DBSCAN 等）依赖固定假设（如聚类数 k、距离阈值 ε），在晶圆缺陷分布随时间演变的动态环境中不够实用。这促使作者采用**非参数方法**自适应发现聚类，同时通过**等变网络**处理方向变异性。

核心动机：设计一种无需预设聚类数量、对旋转鲁棒的深度聚类框架，使得空间上相似的缺陷无论旋转角度如何都能被一致地聚类。

## 方法详解

### 整体框架

DECOR 是一个三阶段的框架：(A) 旋转和翻转不变的嵌入提取器 RCAE，(B) 非参数聚类模块 DeepDPM，(C) 集成异常检测机制。

输入晶圆图 → 归一化与掩码 → RCAE 特征提取（128维） → DeepDPM 聚类 → 集成异常检测

### 关键设计

1. **旋转不变卷积自编码器（RCAE）**：

   使用 e2cnn 库中的 R2Conv 层构建等变编码器。编码器包含三个等变块，每个块由 R2Conv + ReLU + PointwiseAvgPool（stride=2）组成，通道数递增为 8→16→32。关键设计是实现对**二面体群 $D_4$ 的等变性**——包括4个离散旋转（0°, 90°, 180°, 270°）和2个镜像翻转。

   通过 **GroupPooling** 将等变特征转换为方向不变描述子（collapse 所有方向通道），然后使用线性层映射到 128 维隐向量。解码器镜像编码器结构，使用 ConvTranspose2D 重建图像。

   设计动机：相比 MoCo 等对比学习方法，RCAE 内建了对称性处理，产生更紧凑且可分离的聚类。实验验证了 RCAE 在降维空间中形成更好分离的聚类。

2. **非参数聚类（DeepDPM）**：

   基于 Dirichlet 过程混合模型（DPMM）的深度聚类框架，**不需要预设聚类数量**。与 K-Means 不同，DeepDPM 从数据分布中推断最优聚类数。

   实现为两层 MLP：Linear(128, 50) → Linear(50, K)，其中 K 在训练过程中自适应确定。模型产生**软聚类成员概率**，通过 argmax 获得硬标签。

   超参数设置：$\nu = d + 2$（d 为输入维度），$k_{init} = 30$，最大训练 epoch = 200。

3. **集成异常检测**：

   组合 Isolation Forest（IF）和 Local Outlier Factor（LOF）两种检测器，只有两者同时判定为异常时才标记为异常（$\mathbf{1}_{final} = \mathbf{1}_{IF} \wedge \mathbf{1}_{LOF}$）。

   关键创新在于**鲁棒的自适应阈值**：
    $\tau = \text{median}(s) + k \cdot \text{MAD}(s)$
   使用中位数和 MAD 而非均值和标准差，确保对重尾分布和已存在异常点的鲁棒性。

   LOF 的邻居数自适应选择：$k_{LOF} = \text{clip}(\sqrt{N}, k_{min}, k_{max})$，根据聚类大小自动调整。IF 使用保守的污染先验（hi_cont = 0.20）。

### 损失函数 / 训练策略

- RCAE 使用 MSE 损失 + Adam 优化器（lr=$10^{-3}$），batch size=128，训练 1000 epochs
- 输入图像归一化并应用边缘掩码和高斯模糊预处理
- 晶圆图统一缩放到 128×128 像素
- 使用 MultilabelStratifiedShuffleSplit 确保训练/测试集中各缺陷类型分布一致

## 实验关键数据

### 主实验

在 MixedWM38 数据集（38000+ 晶圆图，38+ 缺陷模式组合）上评估：

| 嵌入方法 | 聚类方法 | 最终/最优 K | NMI ↑ | ARI ↑ |
|----------|----------|-------------|-------|-------|
| CAE | K-Means^p | 24 | 0.503±0.00 | 0.199±0.00 |
| MoCo | K-Means^p | 18 | 0.409±0.00 | 0.173±0.01 |
| RCAE | K-Means^p | 30 | 0.529±0.00 | 0.205±0.00 |
| CAE | DeepDPM | 25 | 0.498±0.05 | 0.218±0.01 |
| MoCo | DeepDPM | 30 | 0.273±0.00 | 0.117±0.00 |
| **RCAE (Ours)** | **DeepDPM** | **22** | **0.543±0.03** | **0.296±0.00** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|----------|------|
| RCAE vs CAE (K-Means) | NMI: 0.529 vs 0.503 | 等变编码提升聚类质量 |
| RCAE vs MoCo (K-Means) | NMI: 0.529 vs 0.409 | 重建学习优于对比学习 |
| DeepDPM vs K-Means (RCAE) | ARI: 0.296 vs 0.205 | 非参数方法显著优于固定K |
| MoCo+DeepDPM | NMI: 0.273 | 对比学习嵌入与DeepDPM不兼容 |

### 关键发现

1. RCAE 嵌入配合 DeepDPM 在 NMI 和 ARI 上均达到最优
2. RCAE 产生的聚类确实展现了**旋转不变性**——不同旋转角度的相同缺陷模式被正确归入同一聚类
3. 聚类质量与隐空间中缺陷模式的分离度正相关
4. 非参数聚类自动确定了 K=22，无需人工调优
5. 集成异常检测成功识别出了聚类内的异类缺陷（如中心缺陷聚类中的甜甜圈缺陷异常）

## 亮点与洞察

- **等变性 vs 不变性的优雅设计**：编码器使用等变卷积保留方向信息，GroupPooling 将等变变为不变，两者在正确的层次发挥作用
- **无监督的端到端方案**：从嵌入学习到聚类到异常检测，不需要任何标签
- **实用性强**：模型轻量（RCAE 参数量小），训练成本低（单 H100 ~8小时），适合工业部署
- **集成检测策略稳健**：IF 提供全局分区隔离，LOF 捕捉局部密度偏差，AND 融合降低假阳性

## 局限与展望

1. **需要多次运行 DeepDPM**：确定合适的初始 K 和最优训练 epoch 需要反复实验
2. **多标签评估困难**：NMI 和 ARI 需要将多标签简化为主导标签，可能无法全面反映性能
3. **仅测试单一数据集**：仅在 MixedWM38 上验证，泛化能力未知
4. **缺少与更多聚类基线的比较**：如 GMM、HDBSCAN 等
5. **非 archival 论文**：作为 KGML Bridge 论文，研究深度有限

## 相关工作与启发

DECOR 结合了等变网络（e2cnn/D4 群）、非参数贝叶斯聚类（DPMM/DeepDPM）和鲁棒异常检测三个方面的进展。对于其他需要旋转鲁棒聚类的领域（如细胞图像分析、遥感等）具有参考价值。未来可探索多标签感知的聚类指标和时序缺陷追踪。

## 评分

- 新颖性: ⭐⭐⭐⭐ (组件创新有限，组合有新意)
- 实验充分度: ⭐⭐⭐⭐ (单一数据集，基线比较有限)
- 写作质量: ⭐⭐⭐⭐⭐ (清晰易懂，方法描述详细)
- 价值: ⭐⭐⭐⭐ (工业应用有意义，但作为非归档论文贡献有限)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Deep Incomplete Multi-View Clustering via Hierarchical Imputation and Alignment](deep_incomplete_multi-view_clustering_via_hierarchical_imputation_and_alignment.md)
- [\[AAAI 2026\] CAE: Hierarchical Semantic Alignment for Image Clustering](hierarchical_semantic_alignment_for_image_clustering.md)
- [\[ICLR 2026\] Distributed Algorithms for Euclidean Clustering](../../ICLR2026/others/distributed_algorithms_for_euclidean_clustering.md)
- [\[AAAI 2026\] Approximation Algorithm for Constrained k-Center Clustering: A Local Search Approach](approximation_algorithm_for_constrained_k-center_clustering_.md)
- [\[AAAI 2026\] Enhancing Noise Resilience in Face Clustering via Sparse Differential Transformer](enhancing_noise_resilience_in_face_clustering_via_sparse_differential_transforme.md)

</div>

<!-- RELATED:END -->
