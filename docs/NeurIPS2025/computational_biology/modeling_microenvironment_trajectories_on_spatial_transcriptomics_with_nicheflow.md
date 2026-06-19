---
title: >-
  [论文解读] Modeling Microenvironment Trajectories on Spatial Transcriptomics with NicheFlow
description: >-
  [NeurIPS 2025][计算生物][空间转录组学] NicheFlow是一种基于Flow Matching的生成模型，将细胞微环境表示为点云，通过Variational Flow Matching和最优传输联合建模细胞状态与空间坐标的时间演化，在胚胎发育、脑发育和衰老数据集上显著优于单细胞级别的轨迹推断方法。
tags:
  - "NeurIPS 2025"
  - "计算生物"
  - "空间转录组学"
  - "微环境轨迹推断"
  - "Flow Matching"
  - "最优传输"
  - "点云生成"
  - "细胞生态位"
---

# Modeling Microenvironment Trajectories on Spatial Transcriptomics with NicheFlow

**会议**: NeurIPS 2025  
**arXiv**: [2511.00977](https://arxiv.org/abs/2511.00977)  
**作者**: Kristiyan Sakalyan, Alessandro Palma, Filippo Guerranti, Fabian J. Theis, Stephan Günnemann (TUM, Helmholtz Munich)  
**代码**: [项目页面](https://www.cs.cit.tum.de/daml/nicheflow)  
**领域**: 计算生物  
**关键词**: 空间转录组学, 微环境轨迹推断, Flow Matching, 最优传输, 点云生成, 细胞生态位  

## 一句话总结

NicheFlow是一种基于Flow Matching的生成模型，将细胞微环境表示为点云，通过Variational Flow Matching和最优传输联合建模细胞状态与空间坐标的时间演化，在胚胎发育、脑发育和衰老数据集上显著优于单细胞级别的轨迹推断方法。

## 研究背景与动机

### 问题背景
理解细胞微环境在时空数据中的演变对于解读组织发育和疾病进展至关重要。空间转录组学（Spatial Transcriptomics, ST）技术能够在保留空间信息的同时实现单细胞分辨率的基因表达映射，但ST仅能提供动态生物系统的静态快照。时间分辨的空间分析通过捕获发育阶段间的基因表达模式和细胞排列演变，提供了组织发育的关键时序信息。

### 已有工作的不足
- 现有计算方法在**单细胞水平**推断轨迹，使用速度模型（SiRV、SpVelo）或最优传输（moscot、DeST-OT）连接跨时间的单个细胞
- 这些以细胞为中心的方法从根本上**忽略了结构化生态位的协同演化**——细胞并非孤立存在，而是作为空间微环境的一部分协调发展
- 现有单细胞级别的精确OT方法在**可扩展性和泛化能力**上存在局限

### 核心动机
提出一个关键问题：如何在保留局部邻域关系和细胞状态转变的同时，建模细胞微环境的时空演化？NicheFlow直接将细胞邻域作为整体单元建模其动态，而非关注孤立的细胞轨迹。

## 方法详解

### 微环境定义
给定时间分辨的空间转录组数据，每个时间点$s$的组织切片表示为属性点云$\mathcal{P}_s = \{(\boldsymbol{c}_i^s, \boldsymbol{x}_i^s)\}$，其中$\boldsymbol{c}_i^s \in \mathbb{R}^2$为空间坐标，$\boldsymbol{x}_i^s \in \mathbb{R}^D$为基因表达特征。以固定半径$r$定义局部微环境：

$$\mathcal{M}_i^s = \{(\boldsymbol{c}_j^s, \boldsymbol{x}_j^s) \mid \|\boldsymbol{c}_j^s - \boldsymbol{c}_i^s\| \leq r\}$$

### OT耦合策略
为训练条件生成模型，定义源-目标微环境间的最优熵耦合$\pi_{\epsilon,\lambda}^*$。通过加权平均坐标和特征计算微环境的池化表示：

$$\bar{\boldsymbol{m}}_i^s = \left[\frac{1-\lambda}{|\mathcal{M}_i^s|}\sum \boldsymbol{c}_j^s \;\Big\|\; \frac{\lambda}{|\mathcal{M}_i^s|}\sum \boldsymbol{x}_j^s\right]$$

超参数$\lambda \in [0,1]$平衡空间邻近性与特征相似性：$\lambda$越大越优先特征匹配，$\lambda$越小越优先空间位置保持。

### 混合因子化VFM
NicheFlow的核心创新在于对变分后验的因子化设计：
1. **点云级因子化**：后验在点云中的单个点之间完全因子化
2. **特征-坐标因子化**：细胞特征与空间坐标分别建模
3. **混合分布族**：空间坐标使用**Laplace分布**（集中于均值，适合精确空间建模），基因表达使用**Gaussian分布**

训练损失为：

$$\mathcal{L}_{\text{NicheFlow}}(\theta) = \mathbb{E}\left[\sum_{(\boldsymbol{c}_1, \boldsymbol{x}_1) \in \mathcal{M}^1}\left(\|\boldsymbol{c}_1 - \bar{\boldsymbol{f}}_t^\theta\|_1 + \frac{1}{2}\|\boldsymbol{x}_1 - \bar{\boldsymbol{r}}_t^\theta\|_2^2\right)\right]$$

其中$\bar{\boldsymbol{f}}_t^\theta$和$\bar{\boldsymbol{r}}_t^\theta$分别是坐标和特征的后验均值预测。

### Backbone架构：微环境Transformer
- **编码器-解码器结构**：编码器通过自注意力处理源微环境$\mathcal{M}^0$，解码器对噪声目标进行自注意力后通过交叉注意力条件化于编码器输出
- **输入嵌入**：特征和空间坐标分别嵌入后拼接，时间$t$使用正弦编码
- **置换不变性**：天然适应可变大小的点云输入
- **输出投影**：线性投影输出坐标和特征的后验均值估计

### 采样与生成
生成过程：给定源微环境$\mathcal{M}^0$，采样高斯噪声点云$\mathcal{M}^z$，通过求解ODE $\mathcal{M}^1 = \phi_1^\theta(\mathcal{M}^z \mid \mathcal{M}^0)$生成目标微环境。

## 实验关键数据

### 实验1：定量评估——跨数据集空间重建

三个时空数据集：(1) 小鼠胚胎发育（MED, Stereo-seq, 3个时间点）；(2) 蝾螈脑发育（ABD, Stereo-seq, 5个时间点）；(3) 小鼠脑衰老（MBA, MERFISH, 20个时间点）。

| 模型 | 目标函数 | MED 1NN-F1↑ | MED PSD↓ | MED SPD↓ | ABD 1NN-F1↑ | ABD SPD↓ | MBA 1NN-F1↑ | MBA SPD↓ |
|------|---------|-------------|----------|----------|-------------|----------|-------------|----------|
| LUNA | — | 0.540 | — | — | 0.331 | — | 0.222 | — |
| SPFlow | CFM | 0.272 | 1.681 | 0.602 | 0.190 | 1.119 | 0.205 | 0.824 |
| RPCFlow | CFM | 0.546 | 0.981 | 0.564 | 0.524 | 1.015 | 0.271 | 0.810 |
| RPCFlow | GLVFM | 0.586 | 0.979 | 0.586 | 0.554 | 1.038 | 0.265 | 0.779 |
| **NicheFlow** | CFM | 0.609 | 0.979 | 0.402 | 0.604 | 0.568 | 0.283 | 0.556 |
| **NicheFlow** | **GLVFM** | **0.664** | **0.883** | **0.398** | **0.628** | **0.576** | **0.285** | **0.532** |

- NicheFlow+GLVFM在1NN-F1上比最强基线RPCFlow+GLVFM提升13.3%（MED），13.4%（ABD），7.5%（MBA）
- SPD（覆盖度）指标上NicheFlow相比RPCFlow降低约30-45%，说明生成点云对目标区域的覆盖显著更好
- SPFlow（单细胞级别）在所有指标上大幅落后，验证了微环境级别建模的必要性

### 实验2：定性生物学验证——脊髓与神经嵴细胞追踪

在小鼠胚胎数据集上进行两个生物学验证场景：

| 场景 | 源时间点 | 目标时间点 | λ设定 | NicheFlow结果 | moscot结果 |
|------|---------|----------|------|-------------|-----------|
| 脊髓演化 | E10.5 | E11.5 | 低（空间优先） | 正确映射到成熟脊髓区域 | 大量质量错误分配到泌尿生殖嵴和鰓弓 |
| 头部神经嵴分化 | E9.5 | E10.5 | 高（表达优先） | 正确捕获向间充质和颅骨结构的分化 | 显著的质量泄漏到下方不相关区域 |

NicheFlow的轨迹预测在解剖学位置和后代细胞类型一致性上均显著优于基于精确OT的moscot方法。

## 亮点

- **范式创新**：首次提出微环境级别（而非单细胞级别）的时空轨迹推断，将细胞邻域作为点云整体建模，隐式捕获空间相关性
- **混合因子化VFM**：坐标使用Laplace分布、特征使用Gaussian分布的创新设计，比纯Gaussian VFM在准确度上一致提升
- **可扩展性**：基于mini-batch深度学习的框架，相比moscot等精确OT方法可处理更大规模数据
- **双模式推断**：通过调节$\lambda$参数灵活支持"固定结构中的成分变化"和"发育细胞的空间迁移"两种不同的生物学场景
- **跨数据集泛化**：在胚胎发育、脑发育、脑衰老三类不同生物过程上均表现最优

## 局限与展望

- **局部性限制**：固定半径定义微环境，无法捕获跨越较大空间尺度的组织重组事件
- **两两时间点建模**：当前逐对建模相邻时间点，缺乏跨多个时间点的全局时序一致性约束
- **分布族选择启发式**：Laplace/Gaussian的选择基于经验直觉，缺乏自适应机制
- **评估局限**：1NN-F1依赖于预训练分类器的准确性，可能引入偏差
- **仅2D空间**：当前仅处理2D空间坐标，未扩展到3D组织重建
- **基因表达降维**：使用前50个PCA主成分，可能丢失与轨迹相关的细微基因表达差异

## 与相关工作的对比

- **moscot (Klein et al. 2025)**：单细胞级Fused Gromov-Wasserstein OT，精确OT难扩展，且在脊髓/神经嵴追踪实验中出现显著的质量泄漏
- **LUNA (Yu et al. 2025)**：基于扩散模型的空间坐标生成，不建模时间动态，仅作为空间生成参考
- **Wasserstein FM (Haviv et al. 2024)**：用于点云生成的Wasserstein FM，但不处理坐标-特征联合生成和OT时序轨迹
- **DeST-OT (Halmos et al. 2025)**：半松弛OT对齐空间切片，保留转录组和空间邻近性，但不是生成模型
- **SpaTrack (Shen et al. 2025)**：使用Fused Gromov-Wasserstein OT的单细胞轨迹推断，不处理微环境结构
- **VFM (Eijkelboom et al. 2024)**：原始VFM用于图生成，本文将其扩展到混合分布族的点云生成

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次提出微环境级别的时空轨迹推断范式，混合因子化VFM是方法论创新
- 实验充分度: ⭐⭐⭐⭐ — 三个数据集定量评估+两个生物学场景定性验证，消融实验完整
- 写作质量: ⭐⭐⭐⭐⭐ — 理论推导严谨，动机清晰，图表精美，生物学验证令人信服
- 价值: ⭐⭐⭐⭐ — 为空间转录组学提供了新的计算范式，但应用范围限于特定生物信息学领域

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Learning Relative Gene Expression Trends from Pathology Images in Spatial Transcriptomics](learning_relative_gene_expression_trends_from_pathology_images_in_spatial_transc.md)
- [\[CVPR 2026\] Predicting Spatial Transcriptomics from Histology Images via High-Order Multi-Cell Interaction Modeling](../../CVPR2026/computational_biology/predicting_spatial_transcriptomics_from_histology_images_via_high-order_multi-ce.md)
- [\[ICML 2025\] SToFM: a Multi-scale Foundation Model for Spatial Transcriptomics](../../ICML2025/computational_biology/stofm_a_multi-scale_foundation_model_for_spatial_transcriptomics.md)
- [\[CVPR 2026\] Multi-View Hierarchical Alignment Learning for Spatial Transcriptomics](../../CVPR2026/computational_biology/multi-view_hierarchical_alignment_learning_for_spatial_transcriptomics.md)
- [\[CVPR 2026\] FEAST: Fully Connected Expressive Attention for Spatial Transcriptomics](../../CVPR2026/computational_biology/feast_fully_connected_expressive_attention_for_spatial_transcriptomics.md)

</div>

<!-- RELATED:END -->
