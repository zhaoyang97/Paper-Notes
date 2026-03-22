# CarGait: Cross-Attention based Re-ranking for Gait Recognition

**会议**: ICCV 2025  
**arXiv**: [2503.03501](https://arxiv.org/abs/2503.03501)  
**代码**: 未知  
**领域**: 计算机视觉 / 步态识别 / 重排序  
**关键词**: 步态识别, 重排序, cross-attention, 步态条带, 细粒度匹配, metric learning  

## 一句话总结

提出CarGait——基于cross-attention的步态识别重排序方法：对任意单阶段步态模型的top-K检索结果，通过probe与候选间步态条带(gait strip)的cross-attention学习细粒度pair-wise交互，生成新的条件化表征并重新计算距离进行重排序。在Gait3D/GREW/OU-MVLP三个数据集、7种基线模型上一致提升Rank-1/5准确率，推理速度6.5ms/probe远超现有重排序方法。

## 背景与动机

步态识别（Gait Recognition）通过行走模式识别个体身份，应用于监控、刑侦、医疗等领域。当前主流方法均为**单阶段**：将步态序列编码为单一全局特征，在gallery中做最近邻检索。这类方法的核心问题在于：

- **Rank-1与Rank-5差距大**：例如GaitPart在Gait3D上Rank-1仅28.2%，但Rank-5达47.6%，说明正确身份往往在top-K里但不在top-1
- **Hard negatives干扰**：全局特征难以区分步态模式高度相似的不同个体
- **单一表征的局限**：一个全局向量缺乏对身体局部运动模式的细粒度区分能力

重排序（re-ranking）在图像检索、行人Re-ID中已被广泛使用，但在步态识别中几乎未被探索。已有重排序方法（k-reciprocal、LBR、GCR）主要基于全局特征相似度矩阵的后处理，未针对步态的时空特性做设计。

## 核心问题

如何设计一个通用的重排序模块，通过probe与候选间的细粒度pair-wise交互，改善步态识别的top-rank精度？

## 方法详解

### 整体流程（两阶段）

1. **初始检索**：任意预训练单阶段步态模型 $M$ 对probe提取全局特征，检索gallery中top-K候选
2. **CarGait重排序**：对probe与每个top-K候选的feature map做cross-attention交互，计算新距离后重新排序

### 步态条带表示

预训练模型 $M$ 将步态序列 $G$ 提取为feature map $F \in \mathbb{R}^{s \times d}$，其中 $s$ 是水平身体条带（strip）数量，$d$ 是特征维度。Strip是时空聚合的单元，弱关联人体空间部位（头、躯干、腿等），携带细粒度步态信息。

### Cross-Attention模块

给定probe特征 $F_p$ 和候选特征 $F_c$：

- **正向cross-attention**：$F_p$ 作为Query，$F_c$ 作为Key/Value → 生成 $E_p \in \mathbb{R}^{s \times d}$
- **反向cross-attention**：$F_c$ 作为Query，$F_p$ 作为Key/Value → 生成 $E_c \in \mathbb{R}^{s \times d}$
- **残差连接**：$F_p \to E_p$、$F_c \to E_c$ 各有skip connection，保留预训练特征空间信息
- 每个strip在 $E_p$ 中被候选所有strip的注意力关系所调节，实现了跨strip交互

模块设计：单block、8头注意力、隐藏维度256。

### 新度量空间与重排序

重排序距离 $d^r_{p,c} = \mathcal{Z}(E_p, E_c)$，即cross-attention后对应strip间欧氏距离的平均值。推理时对top-K候选按新距离升序排列即完成重排序。

### 损失函数

联合优化两个损失：

$$\mathcal{L} = \mathcal{L}_{ranking} + \alpha \mathcal{L}_{CE}$$

- **Ranking loss**：基于triplet，惩罚负样本距probe比正样本更近的情况。引入damping参数 $\beta=0.1$，对已正确排列的triplet降权
- **Classification loss**：$E_p$、$E_c$ 经MLP分类器输出身份logits，标准cross-entropy，作为正则化保持身份区分力（$\alpha=0.01$）

### 训练数据构建

冻结预训练模型 $M$，对训练集每个样本取top-$v=30$最近邻构建训练集 $\mathcal{D}$，包含正样本（同身份）和负样本（不同身份）。验证集同样构建，每10k迭代计算验证ranking loss选最佳checkpoint。

## 实验设置

- **数据集**：Gait3D（4000身份/超市场景/39摄像头）、GREW（26345身份/户外/882摄像头）、OU-MVLP（10307身份/室内/14视角）
- **基线模型**：GaitPart、GaitGL、GaitSet、GaitBase、DeepGaitV2-P3D、SwinGait-3D、SkeletonGait++（7种CNN/Transformer架构）
- **训练**：AdamW lr=1e-5、weight decay=1e-2、100k迭代、4×A100 GPU、平均16小时/实验
- **推理**：K=10，6.5ms/probe（单GPU）

## 实验关键数据

### 主要结果（Rank-1提升）

| 模型 | Gait3D | GREW | OU-MVLP |
|------|--------|------|---------|
| GaitSet | 36.7→41.5 (+4.8) | 48.4→52.0 | 87.1→87.5 |
| GaitBase | 64.6→66.1 (+1.5) | 60.1→67.2 (+7.1) | 90.8→91.1 |
| SwinGait-3D | 75.0→76.3 | 79.3→* | - |
| SG++ | 77.6→78.1 | 85.8→88.2 (+2.4) | - |

### 对比其他重排序方法（Gait3D/SG++）

| 方法 | Rank-1 | Rank-5 | mAP |
|------|--------|--------|-----|
| Initial | 77.6 | 89.4 | 70.30 |
| k-reciprocal | 69.7 | 85.6 | 70.30 |
| LBR | 61.7 | 90.2 | 58.99 |
| GCR | 76.1 | 89.6 | 68.72 |
| **CarGait** | **78.1** | **90.4** | **70.86** |

在OU-MVLP（gallery仅1个正样本）上，KR/LBR甚至降低Rank-1（如GaitPart 88.5→68.4/80.6），而CarGait仍能提升（88.5→89.1）。

### 推理速度

| 方法 | 推理时间(ms/probe) |
|------|-------------------|
| k-reciprocal | 214 |
| GCR | 1866 |
| LBR | 19.81 |
| **CarGait** | **6.52** |

### 消融实验（SwinGait-3D on Gait3D）

- 去掉classification loss（$\alpha=0$）：R1 76.0 vs 76.3，证明CE loss作为正则有帮助
- 去掉loss damping（$\beta=1$）：R1 75.5，验证对已正确triplet降权的有效性
- 二元分类baseline（无cross-attention，简单拼接+MLP）：R1降至71.6，直接证明cross-attention的核心作用
- K=5时R1略高(76.5)但R5不变(86.7)，K=10兼顾两者

## 亮点与启发

- **Strip-wise cross-attention是关键**：不同于全局特征后处理，在身体部件级别做probe-candidate交互建模，学到的cross-strip correlation（图4可视化）是性能提升的来源
- **Metric learning + classification双目标**：ranking loss关注排序正确性，CE loss保持特征的身份区分力，两者互补
- **Loss damping技巧**：用 $\beta<1$ 降权已正确排列的triplet，让优化集中在错误case上
- **即插即用通用性极强**：7种架构（CNN/Transformer/多模态）×3个数据集，全部一致提升
- **推理极快**：6.5ms远快于KR(214ms)和GCR(1866ms)，适合实际部署

## 局限性与改进方向

- 需要为每个预训练模型 $M$ 单独训练一个重排序器，增加离线训练开销
- 在OU-MVLP等已饱和数据集（Rank-1>90%）上提升空间有限
- 仅验证了步态识别，跨strip的cross-attention范式可能推广到行人Re-ID、vehicle Re-ID等检索重排序任务
- 固定K=10、v=30未针对不同模型/数据集调优，可能存在进一步优化空间
- 论文未开源代码

## 评分

- 新颖性: ⭐⭐⭐ cross-attention + re-ranking是成熟技术的有效组合，strip-wise交互设计有新意
- 实验充分度: ⭐⭐⭐⭐⭐ 3数据集×7模型+3种竞争重排序方法+全面消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰、可视化到位（图4的注意力矩阵、图5的蜘蛛图）
- 对我的价值: ⭐⭐ 步态识别细分领域；但strip-wise cross-attention做pair-wise重排序的范式可借鉴到其他检索任务
