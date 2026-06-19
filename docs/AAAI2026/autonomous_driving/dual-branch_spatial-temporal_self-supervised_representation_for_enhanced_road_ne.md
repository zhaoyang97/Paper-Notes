---
title: >-
  [论文解读] Dual-branch Spatial-Temporal Self-supervised Representation for Enhanced Road Network Learning
description: >-
  [AAAI 2026][自动驾驶][路网表示学习] 提出 DST（Dual-branch Spatial-Temporal）路网表示学习框架，通过空间分支（mix-hop 转移矩阵 + 图-超图对比学习）和时间分支（Transformer 编码器 + 下一 token 预测 + 工作日/周末分类）两条支路联合建模路网的空间异质性和时间动态性，在三个城市的三项下游任务上取得 SOTA。
tags:
  - "AAAI 2026"
  - "自动驾驶"
  - "路网表示学习"
  - "自监督学习"
  - "图神经网络"
  - "超图"
  - "时空建模"
---

# Dual-branch Spatial-Temporal Self-supervised Representation for Enhanced Road Network Learning

**会议**: AAAI 2026  
**arXiv**: [2511.06633](https://arxiv.org/abs/2511.06633)  
**代码**: [有](https://github.com/chaser-gua/DST)  
**领域**: 自动驾驶  
**关键词**: 路网表示学习, 自监督学习, 图神经网络, 超图, 时空建模

## 一句话总结

提出 DST（Dual-branch Spatial-Temporal）路网表示学习框架，通过空间分支（mix-hop 转移矩阵 + 图-超图对比学习）和时间分支（Transformer 编码器 + 下一 token 预测 + 工作日/周末分类）两条支路联合建模路网的空间异质性和时间动态性，在三个城市的三项下游任务上取得 SOTA。

## 研究背景与动机

路网表示学习（Road Network Representation Learning, RNRL）旨在为路段学习通用低维向量表示，赋能交通推理、出行时间估计、轨迹目的地预测等下游任务。现有方法面临两大挑战：

**1. 空间异质性**：路段相似性不仅取决于地理距离，还涉及功能属性、轨迹可达性等多维度关系。例如远距离的两条路段可能具有相似的功能（如都是住宅区道路），而相邻路段功能可能完全不同。GNN 的邻域平滑机制难以捕获这种长距离的功能相似性。

**2. 时间动态性**：不同类型路段在不同时段表现出显著差异的交通模式，且工作日与周末的模式截然不同。仅凭路网拓扑无法充分表征这种时间维度的动态变化。

早期方法（Node2Vec, GCN, GAE）局限于简单图结构；近期方法（Toast, JCLRNT, TrajRNE）开始利用轨迹信息但缺乏时间建模；DyToast 加入了三角时间特征但在速度推理任务中表现退化。

## 方法详解

### 整体框架

DST 采用双分支架构，空间分支和时间分支**独立预训练**，最终融合用于下游任务：

- **空间分支**：GNN + 超图对比学习 → 捕获路网空间拓扑和高阶语义关系
- **时间分支**：Transformer 编码器 + 双任务自监督 → 建模 24 小时交通动态模式
- **融合策略**：拼接（concatenation）三种表示用于下游任务

### 关键设计

**1. Mix-hop 转移矩阵加权**

从轨迹数据中提取路段间的多跳到达关系，构建 mix-hop 转移矩阵：

$$P_{hop}[r_i, r_j] = \sum_{\tau \in \mathcal{T}} \sum_{1 \leq i < j \leq m} m - (j - i)$$

权重策略：跳数越小初始权重越大、跳数越大权重越小，既强调相邻连接、又纳入可达的远距离连接。行归一化后作为可学习权重矩阵初始化，用于 GNN 前的特征加权：

$$Z_{hop} = \tilde{P}_{hop} \cdot Z_{\mathcal{I}}$$

**2. 空间语义图-超图对比学习**

两个视角构建路网表示：

**图视角**：多层 GAT 编码路网拓扑结构，同时利用边特征 $X_\mathcal{E}$（路段间连接属性），生成空间表示 $Z_\mathcal{G}$。

**超图视角**：构建三种超边捕获高阶关系：
- $\mathcal{E}_{\mathcal{H}_1}$：功能区域超边——对路段做谱聚类，同一簇的路段共享超边
- $\mathcal{E}_{\mathcal{H}_2}$：同类型超边——相同类型的路段不论距离远近共享超边
- $\mathcal{E}_{\mathcal{H}_3}$：相邻单向超边——地理相邻的单向道路共享超边（Tobler 地理第一定律）

通过 HGNN+ 编码超图，生成语义表示 $Z_\mathcal{H}$。

对比学习最大化两个视角同一路段表示的互信息：

$$\mathcal{L}_{\mathcal{GH}} = -\frac{1}{N}\sum_{r_i \in \mathcal{R}} \left[\frac{1}{|\mathcal{H}(r_i)|}\sum_{r_j \in \mathcal{H}(r_i)} I(v_{r_i}, h_{r_j})\right]$$

**3. 时间动态建模**

交通动态定义为 $\mathcal{D}_\mathcal{R} \in \mathbb{R}^{N \times 24 \times 2}$，即每条路段 24 小时的访问量序列（工作日/周末两通道）。

Transformer 编码器取最后隐状态作为序列压缩表示：

$$Z_\mathcal{D} = \text{TransEnc}(\text{PosEnc}(\mathcal{D}_\mathcal{R}))[-1]$$

联合两个自监督任务：
- **动态预测**（回归）：用历史序列预测下一时步的交通量
$$\mathcal{L}_{reg} = \frac{1}{N \times C}\sum_{i=1}^{N \times C}\|y_i - \hat{y}_i\|^2$$
- **动态分类**：区分输入序列是工作日还是周末
$$\mathcal{L}_{cls} = -\frac{1}{N \times C}\sum_{i}\sum_c y_i(c)\log(\hat{y}_i(c))$$

### 损失函数 / 训练策略

时间分支总损失：$\mathcal{L}_d = \lambda_{reg} \cdot \mathcal{L}_{reg} + \lambda_{cls} \cdot \mathcal{L}_{cls}$

空间分支与时间分支**分别预训练**，最终通过拼接融合三种表示（$Z_\mathcal{G}, Z_\mathcal{H}, Z_\mathcal{D}$）用于下游任务。参数敏感性分析显示增大 $\lambda_{reg}$ 比例可改善性能，因为两个任务初始损失量级差异大，需要通过权重调节实现任务平衡。

## 实验关键数据

### 主实验

**表1：目的地预测和出行时间估计（三个城市）**

| 方法 | Beijing ACC@1↑ | Beijing MRR↑ | Porto ACC@1↑ | Xi'an ACC@1↑ |
|------|---------------|-------------|-------------|-------------|
| Node2Vec | 0.1954 | 0.2884 | 0.2201 | 0.4088 |
| TrajRNE | 0.6728 | 0.7603 | 0.6728 | 0.8260 |
| JCLRNT | 0.4222 | 0.5528 | 0.5133 | 0.6752 |
| **DST** | **0.7288** | **0.8213** | **0.6766** | **0.8335** |

**表2：速度推理任务**

| 方法 | Beijing MAE↓ | Porto MAE↓ | Xi'an MAE↓ |
|------|-------------|-----------|-----------|
| JCLRNT | 2.8512 | 3.7475 | 4.5138 |
| TrajRNE | 3.0756 | 4.7854 | 5.1898 |
| **DST** | **2.4595** | **3.4259** | **4.4987** |

DST 在三个城市、三项下游任务上全面取得最优，目的地预测相比 TrajRNE 在北京提升 8.3%（ACC@1）。

### 消融实验

- w/o $P_{hop}$（去掉 mix-hop 矩阵）：速度推理退化最严重，说明多跳运动关系对理解路段功能至关重要
- w/o $hg_2$（去掉同类型超边）：速度推理退化显著，同类型路段的高阶关系是关键补充信息
- w/o tm（去掉时间分支）：轨迹相关任务（目的地预测、出行时间）退化最严重，时间动态是不可或缺的补充
- w/o $hg_1$、w/o $hg_3$：中等程度退化，三种超边互相补充

### 关键发现

1. 空间语义超图和 mix-hop 矩阵在速度推理中贡献最大——该任务对路段功能理解要求最高
2. 时间分支对轨迹相关任务提升显著——交通动态包含出行模式的关键信息
3. 零样本跨城市迁移实验中 DST 表现竞争力强（北京训练→波尔图测试），ACC@1=0.6424 远超 JCLRNT 的 0.0167
4. DST 对超参不敏感，较小的 traffic batch size 略优（因交通序列的稀疏性，大 batch 引入噪声）

## 亮点与洞察

- **双分支分治策略**清晰且有效：空间/时间各自用最适合的架构（GNN vs Transformer），避免异构输入的耦合干扰
- 三种超边设计覆盖了路网高阶关系的多个维度（功能区域、类型一致、地理邻接），互相补充
- 零样本跨城市迁移能力说明学到的表示具有不错的泛化性，降低新城市部署成本
- 工作日/周末分类作为正则化任务的设计巧妙，引导模型学习有区分度的时间表示

## 局限与展望

- 双分支独立预训练再拼接融合较为简单，联合训练或注意力融合可能效果更好
- mix-hop 转移矩阵依赖轨迹数据质量，GPS 噪声和地图匹配误差可能影响矩阵质量
- 时间分支仅用 24 小时粒度的访问量，未考虑更细粒度（如 15 分钟间隔）或更长时间跨度
- 超图的三种超边类型是人工设计的，可探索数据驱动的超边构建方式

## 相关工作与启发

- **JCLRNT** 和 **TrajRNE** 是利用轨迹增强路网表示的先驱，DST 在其基础上加入超图和时间分支
- 超图对比学习思想可应用于其他空间网络（如电力网、水利网）的表示学习
- 双分支预训练+融合的范式与多模态预训练（如 CLIP 的视觉/文本）有异曲同工之处

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 技术深度 | 4 |
| 实验充分性 | 5 |
| 写作质量 | 4 |
| 实用价值 | 4 |
| 总评 | 4.2 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Self-Supervised Learning of Graph Representations for Network Intrusion Detection](../../NeurIPS2025/autonomous_driving/self-supervised_learning_of_graph_representations_for_network_intrusion_detectio.md)
- [\[AAAI 2026\] Minimum-Cost Network Flow with Dual Predictions](minimum-cost_network_flow_with_dual_predictions.md)
- [\[NeurIPS 2025\] How Different from the Past? Spatio-Temporal Time Series Forecasting with Self-Supervised Deviation Learning](../../NeurIPS2025/autonomous_driving/how_different_from_the_past_spatio-temporal_time_series_forecasting_with_self-su.md)
- [\[CVPR 2026\] STUR3D: Spatio-Temporal Unified Representation Learning for 3D Object Detection](../../CVPR2026/autonomous_driving/stur3d_spatio-temporal_unified_representation_learning_for_3d_object_detection.md)
- [\[CVPR 2026\] TerraSeg: Self-Supervised Ground Segmentation for Any LiDAR](../../CVPR2026/autonomous_driving/terraseg_self-supervised_ground_segmentation_for_any_lidar.md)

</div>

<!-- RELATED:END -->
