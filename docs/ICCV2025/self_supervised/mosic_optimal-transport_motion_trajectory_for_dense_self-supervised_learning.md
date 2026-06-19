---
title: >-
  [论文解读] MoSiC: Optimal-Transport Motion Trajectory for Dense Self-Supervised Learning
description: >-
  [ICCV 2025][自监督学习][稠密自监督学习] MoSiC 利用离线点跟踪器提取长程运动轨迹，通过基于最优传输（Sinkhorn-Knopp）的聚类机制在时间维度上传播聚类分配，从而在视频数据上学习空间-时间一致的稠密表征，仅用视频训练即可将 DINOv2 在多个图像/视频基准上提升 1%–6%。
tags:
  - "ICCV 2025"
  - "自监督学习"
  - "稠密自监督学习"
  - "最优传输"
  - "运动轨迹"
  - "时空一致性"
  - "视频分割"
---

# MoSiC: Optimal-Transport Motion Trajectory for Dense Self-Supervised Learning

**会议**: ICCV 2025  
**arXiv**: [2506.08694](https://arxiv.org/abs/2506.08694)  
**代码**: [github.com/SMSD75/MoSiC](https://github.com/SMSD75/MoSiC)  
**领域**: 自监督学习 / 视频表示学习  
**关键词**: 稠密自监督学习, 最优传输, 运动轨迹, 时空一致性, 视频分割

## 一句话总结

MoSiC 利用离线点跟踪器提取长程运动轨迹，通过基于最优传输（Sinkhorn-Knopp）的聚类机制在时间维度上传播聚类分配，从而在视频数据上学习空间-时间一致的稠密表征，仅用视频训练即可将 DINOv2 在多个图像/视频基准上提升 1%–6%。

## 研究背景与动机

稠密自监督学习在像素/patch 级表征学习上取得了很大进展，但将其扩展到视频仍很困难。核心挑战在于：

**静态增强失效**：图像域中的颜色变换等隐式保持像素对应关系，但视频中的物体运动、相机位移和形变会破坏这种对应；

**遮挡问题**：先前方法（如 TimeTuning）通过掩码传播在帧间建立对应关系，但当物体被暂时遮挡时会产生传播误差；

**漂移累积**：长程跟踪中误差逐帧积累，导致特征表征退化。

MoSiC 的核心洞察来自格式塔心理学原则——"一起运动的点属于同一组"，将这一原则推到更细粒度的 patch 级别。

## 方法详解

### 整体框架

MoSiC 采用教师-学生框架，具体流程为：
1. 对视频片段 $X \in \mathbb{R}^{h \times w \times c \times T}$ 分帧 patchify；
2. 学生网络处理随机遮掩后的 patch，教师网络处理原始 patch；
3. 使用离线点跟踪器（CoTracker-v3）提取长程运动轨迹；
4. 在首帧上通过 Sinkhorn-Knopp 最优传输算法进行聚类；
5. 沿运动轨迹传播聚类分配到后续帧；
6. 通过交叉熵损失对齐学生/教师的聚类分配。

### 关键设计

1. **运动轨迹提取（Motion Trajectories）**：从首帧采样 $N$ 个点构成均匀网格，利用冻结的 CoTracker-v3 跟踪这些点在整个视频片段中的位置 $\text{Traj}_{t,i} \in \mathbb{R}^{T \times N \times 2}$。CoTracker 的关键优势在于它具有长程跟踪能力并且对物体遮挡后的再出现（object permanence）具有鲁棒性。

2. **基于最优传输的聚类（OT-based Clustering）**：给定聚类原型 $P \in \mathbb{R}^{K \times d}$，计算 patch 特征与原型之间的传输代价（负余弦相似度），然后通过 Sinkhorn-Knopp 算法求解熵正则化最优传输问题：

$$M^* = \arg\min_{M \in \mathcal{M}} \langle M, C \rangle - \epsilon \frac{1}{\lambda} H(M)$$

其中 $H(M)$ 是熵正则化项，$\epsilon$ 控制分配的平滑度，同时施加均匀边际约束以防止坍缩。这一过程分别在学生网络和教师网络上进行。

3. **聚类传播（Cluster Propagation）**：在首帧 $t_0$ 上通过 OT 得到聚类分配后，利用运动轨迹将教师网络的聚类分配传播到后续帧：$\mathcal{Q}_t^{\text{teach},i} = \mathcal{Q}_{t_0}^{\text{teach},i}$。这意味着沿同一轨迹运动的点保持相同的聚类身份——即使它们的外观因视角变化而改变。教师端使用双线性插值在连续坐标上采样特征，学生端使用最近邻插值。

### 损失函数 / 训练策略

训练损失为交叉熵形式，在学生首帧的聚类得分 $S_{t_0}^{\text{stu},k,i}$（softmax 后）与教师传播后的 one-hot 聚类分配之间计算：

$$\mathcal{L}_{\text{clust}}(i) = -\sum_{t=1}^{T} \sum_{k=1}^{K} v_{t,i} \cdot \delta(\mathcal{Q}_t^{\text{teach},i} = k) \cdot \log(S_{t_0}^{\text{stu},k,i})$$

关键细节：
- $v_{t,i}$ 是可见性标志，仅对可见轨迹点计算损失，增强对遮挡的鲁棒性；
- 仅使用简单增强（裁剪+遮掩），不需要颜色抖动、灰度等复杂增强；
- 初始化自 DINOv2 预训练权重，在 YouTube-VOS 上训练。

## 实验关键数据

### 主实验

| 基准 | 数据集 | 指标 | MoSiC-S14 | DINOv2-S14 | 提升 |
|------|--------|------|-----------|------------|------|
| In-context Scene Understanding | Pascal VOC (1/128) | mIoU | 62.5 | 56.0 | +6.5 |
| In-context Scene Understanding | Pascal VOC (1/1) | mIoU | 78.2 | 77.0 | +1.2 |
| In-context Scene Understanding | ADE20K (1/1) | mIoU | 40.7 | 38.8 | +1.9 |
| 无监督视频语义分割 | DAVIS (F-Clustering) | mIoU | 58.9 | 57.4 | +1.5 |
| 无监督视频语义分割 | YTVOS (F-Clustering) | mIoU | 60.6 | 56.3 | +4.3 |
| 冻结聚类 | Pascal VOC (K=500) | mIoU | 60.2 | 58.6 | +1.6 |
| 线性分割 | Pascal VOC | mIoU | 79.7 | 78.9 | +0.8 |
| 线性分割 | ADE20K | mIoU | 39.6 | 37.9 | +1.7 |
| 语义分割 | Pascal VOC | mIoU | 51.2 | 37.5 | +13.7 |

MoSiC-B14（85M 参数）进一步提升：Pascal VOC in-context (1/128) 达 65.5，(1/1) 达 80.5。

### 消融实验

| 配置 | Pascal VOC (mIoU) | ADE20K (mIoU) | 说明 |
|------|-------------------|---------------|------|
| 无遮掩 | 51.1 | 18.2 | 基线 |
| 10% 遮掩率 | 51.5 | 18.6 | 最佳 |
| 40% 遮掩率 | 49.9 | 18.5 | 过高遮掩有害 |
| 无 EMA 教师 | 50.5 | 18.2 | EMA 教师有帮助 |
| 有 EMA 教师 | 51.5 | 18.6 | 默认设置 |
| 8×8 网格 | 49.2 | 17.5 | 稀疏网格不足 |
| 16×16 网格 | 51.5 | 18.6 | 默认设置 |

### 关键发现

- MoSiC 可泛化到多种视觉基础模型（DINO、EVA-CLIP、DINOv2-R），均能带来 2%–7% 的提升；
- 低数据量场景下提升更显著（1/128 数据时 +6.5%）；
- 相比 TimeTuning，MoSiC 在 DAVIS 上聚类提升 8.7%，在 YTVOS 上提升 9.4%。

## 亮点与洞察

1. **运动轨迹作为隐式监督信号**：巧妙利用格式塔心理学原则，将"共同运动的点必归类于一组"这一直觉转化为可优化的目标；
2. **仅用视频提升图像表征**：无需图像标注，通过视频时序信号就能改善静态图像的稠密表征质量；
3. **可见性掩码机制**：仅对可见轨迹计算损失，优雅地解决了遮挡导致的伪标签问题，这比 TimeTuning 的掩码传播更鲁棒；
4. **即插即用**：可应用于各种视觉基础模型作为后续微调阶段。

## 局限与展望

- 依赖离线点跟踪器（CoTracker-v3），跟踪器的质量直接影响聚类传播的准确性；
- 训练需要视频数据（YouTube-VOS），无法在纯图像场景下使用；
- 对于极快速运动或相机大幅抖动场景，跟踪器本身可能失效；
- 聚类数 $K$ 需要手动设定，缺乏自适应机制。

## 相关工作与启发

- 与 TimeTuning 的核心区别在于使用鲁棒的长程点跟踪器替代掩码传播，避免了遮挡后的传播误差累积；
- 与 NeCo、CrIBo 等图像域方法互补——它们利用跨图像一致性，MoSiC 利用跨帧时序一致性；
- OT 聚类机制继承自 DINO/SwAV，但创新性地将聚类分配沿运动轨迹传播。

## 评分

- **新颖性**: ⭐⭐⭐⭐ 运动轨迹+OT 聚类传播的组合在稠密自监督学习中是首创
- **实验充分度**: ⭐⭐⭐⭐⭐ 6 个数据集、4 个评估基准、多种 backbone 泛化实验、全面的消融
- **写作质量**: ⭐⭐⭐⭐ 逻辑清晰，公式规范，图示直观
- **价值**: ⭐⭐⭐⭐ 为视频数据如何增强图像表征提供了有效方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] An Optimal Transport-driven Approach for Cultivating Latent Space in Online Incremental Learning](../../CVPR2026/self_supervised/an_optimal_transport_driven_approach_for_cultivating_latent_space_in_online_incr.md)
- [\[CVPR 2026\] Shape-of-You: Fused Gromov-Wasserstein Optimal Transport for Semantic Correspondence in-the-Wild](../../CVPR2026/self_supervised/shape-of-you_fused_gromov-wasserstein_optimal_transport_for_semantic_corresponde.md)
- [\[NeurIPS 2025\] Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning](../../NeurIPS2025/self_supervised/self-supervised_contrastive_learning_is_approximately_supervised_contrastive_lea.md)
- [\[ICML 2025\] ReSA: Clustering Properties of Self-Supervised Learning](../../ICML2025/self_supervised/clustering_properties_of_self-supervised_learning.md)
- [\[ICML 2025\] Collapse-Proof Non-Contrastive Self-Supervised Learning](../../ICML2025/self_supervised/collapse-proof_non-contrastive_self-supervised_learning.md)

</div>

<!-- RELATED:END -->
