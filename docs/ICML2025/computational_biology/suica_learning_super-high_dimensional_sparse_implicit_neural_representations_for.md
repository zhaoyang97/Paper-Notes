---
title: >-
  [论文解读] SUICA: Learning Super-high Dimensional Sparse Implicit Neural Representations for Spatial Transcriptomics
description: >-
  [ICML 2025][计算生物][spatial transcriptomics] 提出 SUICA，通过图增强自编码器将超高维稀疏空间转录组数据压缩到紧凑嵌入空间，再用隐式神经表示（INR）建模坐标到嵌入的连续映射，实现跨多种 ST 平台的空间填补、基因填补和去噪。 领域现状：空间转录组学（Spatial Transcr…
tags:
  - "ICML 2025"
  - "计算生物"
  - "spatial transcriptomics"
  - "implicit neural representations"
  - "graph autoencoder"
  - "gene expression"
  - "spatial imputation"
---

# SUICA: Learning Super-high Dimensional Sparse Implicit Neural Representations for Spatial Transcriptomics

---

**会议**: ICML 2025  
**arXiv**: [2412.01124](https://arxiv.org/abs/2412.01124)  
**代码**: [https://github.com/Szym29/SUICA](https://github.com/Szym29/SUICA)  
**领域**: 计算生物  
**关键词**: spatial transcriptomics, implicit neural representations, graph autoencoder, gene expression, spatial imputation

## 一句话总结

提出 SUICA，通过图增强自编码器将超高维稀疏空间转录组数据压缩到紧凑嵌入空间，再用隐式神经表示（INR）建模坐标到嵌入的连续映射，实现跨多种 ST 平台的空间填补、基因填补和去噪。

---

## 研究背景与动机

**领域现状**：空间转录组学（Spatial Transcriptomics, ST）是一种能在保留组织空间信息的同时定量基因表达的技术。当前主流 ST 平台（如 Stereo-seq、Visium、Slide-seqV2）在各空间位置采集 mRNA 转录本读数，生成超高维（通常 >20,000 个基因通道）的表达矩阵。深度学习方法如 SpaGCN、STAGATE、GraphST 等已用于增强空间分辨率和去噪。

**现有痛点**：ST 数据面临三重挑战。第一，空间采样密度有限（高分辨率 ST 非常昂贵，约 $3,500/cm^2$），导致空间分辨率不足。第二，mRNA 捕获率低加上不同细胞状态的表达模式差异，使数据呈现严重的零膨胀分布（零值占比高达 90%），即 dropout 问题。第三，不同 ST 平台在空间分布规律性、测序深度和 dropout 率上差异巨大，难以用统一框架建模。

**核心矛盾**：隐式神经表示（INR）具有连续建模和内在平滑性的优良特性，天然适合对离散采样点进行空间插值。但 INR 面对 ST 数据有两个根本困难：（1）现有 INR 应用都是低维到低维的映射（如 $\mathbb{R}^2 \to \mathbb{R}^3$），而 ST 要求从 $\mathbb{R}^2$ 映射到超高维空间（>20,000 通道），简单加宽加深 MLP 无法克服维度灾难；（2）INR 的输出倾向于正态分布的平滑值，而 ST 数据是零膨胀的极度稀疏分布，常规回归范式无法保持稀疏性。

**本文目标** 设计一种专门针对 ST 数据特性的 INR 变体，使其能在连续紧凑的表示下进行空间填补、基因填补和去噪，同时保持超高维输出的稀疏性和数值保真度。

**切入角度**：不让 INR 直接映射到超高维原始空间，而是先用图自编码器（GAE）将超高维稀疏数据压缩为低维稠密嵌入，让 INR 仅学习坐标到低维嵌入的映射——将"维度灾难"的负担转移给更擅长处理此问题的图神经网络。

**核心 idea**：用图自编码器桥接 INR 与超高维 ST 数据，让 INR 在紧凑嵌入空间工作，再通过带 Dice Loss 的解码头恢复到原始空间并强制稀疏性。

## 方法详解

### 整体框架

SUICA 的 pipeline 分为三个阶段：（1）预训练图自编码器（GAE）：基于 GCN 编码器和 MLP 解码器，以自回归方式在 ST 切片上训练，获得所有采样点的低维嵌入 $z_{gt}$；（2）INR 训练：优化从空间坐标到嵌入 $z_{gt}$ 的神经映射；（3）解码头微调：冻结 INR，将 GAE 预训练的解码器接入 INR 输出，微调解码器学习从嵌入到原始基因表达的映射，使用包含 Dice Loss 的组合损失来保持稀疏性。

### 关键设计

1. **图增强自编码器 (GAE)**:

    - 功能：将超高维稀疏 ST 数据压缩为低维稠密嵌入，同时保留空间上下文信息
    - 核心思路：编码器采用 GCN，基于 $k$-NN 图（$k=5$）捕获相邻 spot 的上下文信息，使嵌入具备结构感知能力。解码器使用普通 MLP（因为插值生成的新点没有图结构可用）。训练损失为标准 MSE：$\mathcal{L}_{gae} = \frac{1}{|M_y|}\sum_{M_y}(\hat{y} - y_{gt})^2$
    - 设计动机：谱分析实验表明，GAE 生成的嵌入比普通 AE 具有更高的 Graph Total Variation 和更大的通道间方差——即更具结构辨识度和信息量。这种解耦的嵌入更适合 INR 建模

2. **嵌入映射 (INR)**:

    - 功能：学习从空间坐标 $x$ 到紧凑嵌入 $z$ 的连续映射
    - 核心思路：根据 ST 数据的空间稀疏性选择不同 INR 架构——空间稀疏数据用 SIREN（周期性激活函数），空间稠密数据用 FFN（随机傅里叶特征）。训练使用 MSE 损失：$\mathcal{L}_{embd} = \frac{1}{|M_z|}\sum_{M_z}(\hat{z} - z_{gt})^2$。在嵌入空间操作避免了维度灾难（从 >20K 维降到低维），使 INR 的拟合任务大为简化
    - 设计动机：INR 的内在平滑性使其天然具备插值能力，在低维嵌入空间工作时这种平滑先验不会被超高维稀疏性破坏

3. **带 Dice Loss 的解码头**:

    - 功能：将 INR 生成的嵌入解码回超高维原始基因表达空间，同时保持稀疏性
    - 核心思路：先用 $\mathcal{L}_{embd}$ 预热 INR 至稳定，再冻结 INR 接入预训练解码器并单独微调。损失函数包含三项：$\mathcal{L}_{recons} = \frac{1}{|M_y^+|}\sum_{M_y^+}(\hat{y}-y_{gt})^2 + \frac{1}{|M_y|}\sum_{M_y}|\hat{y}-y_{gt}| + \lambda\mathcal{L}_{dice}$。其中 Dice Loss 将回归问题转化为准分类问题，用 $\tanh$ 映射输出到伪概率空间 $[0,1)$，以 IoU 优化预测的非零模式与真实非零模式的重合度
    - 设计动机：（1）分阶段训练避免 INR 映射误差的域偏移直接损害解码性能，也防止预训练解码器的局部最优阻碍 INR 优化；（2）仅在非零值上计算 MSE（$M_y^+$）避免全零预测也获得低损失的问题；（3）Dice Loss 对类别不平衡敏感，有效强制模型保持 ST 数据固有的稀疏模式

### 损失函数 / 训练策略

三阶段顺序训练：GAE 预训练（Adam, lr=$10^{-5}$, 200 epochs）→ INR 训练（Adam, lr=$10^{-4}$, 1K epochs）→ 解码头微调（同样 lr, 1K epochs）。所有实验在单张 RTX 4090 上完成。

## 实验关键数据

### 主实验——空间填补

| 方法 | MAE↓ | MSE↓ | Cosine↑ | Pearson↑ | Spearman↑ | ARI↑ |
|------|------|------|---------|----------|-----------|------|
| FFN | 6.51 | 1.20 | 0.706 | 0.718 | 0.400 | 0.143 |
| SIREN | 7.21 | 1.31 | 0.661 | 0.678 | 0.247 | 0.289 |
| STAGE (SOTA) | 6.52 | 1.11 | 0.732 | 0.747 | 0.365 | 0.139 |
| **SUICA** | **5.66** | **0.85** | **0.797** | **0.792** | **0.447** | **0.343** |

*Stereo-seq MOSTA 数据集，MAE/MSE ×10⁻²。参考 ARI = 0.312*

### 消融实验

| 配置 | MSE↓ (E16.5) | Cosine↑ (E16.5) | MSE↓ (Brain) | Cosine↑ (Brain) |
|------|-------------|-----------------|-------------|-----------------|
| Vanilla INR | 2.35 | 0.668 | 9.33 | 0.756 |
| +AE | 1.60 | 0.789 | 11.27 | 0.695 |
| +Dice | 1.48 | 0.806 | 7.05 | 0.826 |
| +Graph (SUICA) | **1.47** | **0.807** | **5.67** | **0.860** |

### 关键发现

- SUICA 的 ARI 甚至超过了用 ground truth 计算的参考值 3.9%，说明 INR 的平滑先验实际增强了生物信号
- SUICA 成功预测了特定基因的空间表达模式（如 AFP 在肝脏中的定位表达、SEPT3 在脑区的低表达信号恢复）
- GAE vs AE 的效果因 ST 平台而异：空间稀疏数据（Human Brain, Visium）中 Graph 增强贡献最大，空间稠密数据（MOSTA）中 AE+Dice 贡献更大
- 在基因填补和去噪任务上同样优于 STAGE，特别是去噪场景优势明显（Cosine 0.733 vs 0.606）

## 亮点与洞察

- 将 INR 从传统的低维→低维映射拓展到低维→超高维（>20K 通道）的场景，通过中间嵌入空间巧妙解决了维度灾难，这是 INR 应用范围的重要扩展
- Dice Loss 的"回归视为分类"思路非常适合零膨胀数据——这种策略可推广到其他稀疏数据建模，如点云、事件相机数据
- SUICA 是退化类型无关的（degradation-agnostic），同一框架无需任何先验知识即可同时处理空间缺失、基因 dropout 和噪声
- 生物保真性甚至超越原始数据的发现（ARI 超过 ground truth 3.9%），说明 INR 的平滑先验本质上是在做隐式去噪，增强了真实的生物信号
- 图自编码器的谱分析（图 3）直观展示了为什么 GAE 优于普通 AE——GTV 更高意味着嵌入保留了更丰富的空间结构信息
- 论文在多种 ST 平台（Stereo-seq、Visium、Slide-seqV2、MERFISH）上验证，充分证明了方法的通用性

## 局限与展望

- 三阶段串行训练流程较为繁琐，端到端联合训练可能更优，但论文解释了为何需要分阶段（域偏移和局部最优问题）
- GAE 编码器使用的 $k$-NN 图结构在 spot 密度差异极大时可能不稳定，自适应图构建值得探索
- 当前评估主要在小鼠组织上，人类组织上的泛化性未充分验证——人类样本的基因表达分布可能有所不同
- INR 训练速度受限，对于更大规模的 ST 数据（百万级 spot 如 MERFISH 全脑数据）的可扩展性未讨论
- 理论上未分析 INR 从低维到超高维映射的近似误差界，缺乏收敛性保证
- 嵌入维度的选择对性能有显著影响但论文未充分讨论如何自动确定最优维度
- 与有参考图像的超分方法（如 Hist2ST）的比较缺失

## 相关工作与启发

- STAGE (Li et al., 2024) 是当前 SOTA 的基于坐标的 ST 增强方法，使用位置监督的自编码器，但没有利用图结构和稀疏性先验，在空间稀疏数据上表现受限
- SIREN 和 FFN 是经典 INR 架构，SUICA 灵活选用两者作为骨干，并证明其通用性
- NeRF 类工作同样在嵌入空间做 INR 再解码，但映射维度远低于 ST 的 20K+，SUICA 首次将 INR 推向超高维输出
- SpaGCN、STAGATE、GraphST 等图方法专注于空间域聚类，而 SUICA 关注连续信号重建，两者互补
- 本文的 GAE+INR 范式可推广到其他空间组学数据（如空间蛋白质组学、空间代谢组学），只需替换 GAE 的输入维度
- Dice Loss 在医学图像分割中广泛使用，本文将其创新性地引入回归任务处理零膨胀分布

## 评分

⭐⭐⭐⭐ 方法设计针对性强，从三个层面（GAE 维度压缩、INR 连续建模、Dice Loss 稀疏性保持）系统解决 ST 数据建模难题。实验全面覆盖 Stereo-seq、Visium、Slide-seqV2 多平台和空间填补、基因填补、去噪多任务，消融实验充分。ARI 超越 ground truth 的发现也很有趣。但三阶段训练略显复杂，与有组织学图像辅助的方法（Hist2ST、TRIPLEX）的对比缺失，可扩展性讨论不足。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] SToFM: a Multi-scale Foundation Model for Spatial Transcriptomics](stofm_a_multi-scale_foundation_model_for_spatial_transcriptomics.md)
- [\[NeurIPS 2025\] Learning Relative Gene Expression Trends from Pathology Images in Spatial Transcriptomics](../../NeurIPS2025/computational_biology/learning_relative_gene_expression_trends_from_pathology_images_in_spatial_transc.md)
- [\[CVPR 2026\] Multi-View Hierarchical Alignment Learning for Spatial Transcriptomics](../../CVPR2026/computational_biology/multi-view_hierarchical_alignment_learning_for_spatial_transcriptomics.md)
- [\[CVPR 2026\] HyperST: Hierarchical Hyperbolic Learning for Spatial Transcriptomics Prediction](../../CVPR2026/computational_biology/hyperst_hierarchical_hyperbolic_learning_for_spatial_transcriptomics_prediction.md)
- [\[CVPR 2026\] Predicting Spatial Transcriptomics from Histology Images via High-Order Multi-Cell Interaction Modeling](../../CVPR2026/computational_biology/predicting_spatial_transcriptomics_from_histology_images_via_high-order_multi-ce.md)

</div>

<!-- RELATED:END -->
