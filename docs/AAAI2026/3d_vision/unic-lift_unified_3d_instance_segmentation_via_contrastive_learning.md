---
title: >-
  [论文解读] UniC-Lift: Unified 3D Instance Segmentation via Contrastive Learning
description: >-
  [AAAI 2026][3D视觉][3D实例分割] 提出 UniC-Lift，一个统一的单阶段 3D 实例分割框架，通过在 3DGS 基元中学习可优化的向量嵌入，并利用对比损失和三元组损失训练，最终通过简单的"嵌入到标签"（Embedding-to-Label）过程直接解码出一致的 3D 分割标签，无需 HDBSCAN 等后处理聚类步骤，训练时间从 15+ 小时缩短至 40 分钟以内。
tags:
  - "AAAI 2026"
  - "3D视觉"
  - "3D实例分割"
  - "3D高斯溅射"
  - "对比学习"
  - "多视图一致性"
  - "嵌入到标签"
---

# UniC-Lift: Unified 3D Instance Segmentation via Contrastive Learning

**会议**: AAAI 2026  
**arXiv**: [2512.24763](https://arxiv.org/abs/2512.24763)  
**代码**: [github.com/val-iisc/UniC-Lift](https://github.com/val-iisc/UniC-Lift)  
**领域**: 3D视觉  
**关键词**: 3D实例分割, 3D高斯溅射, 对比学习, 多视图一致性, 嵌入到标签

## 一句话总结

提出 UniC-Lift，一个统一的单阶段 3D 实例分割框架，通过在 3DGS 基元中学习可优化的向量嵌入，并利用对比损失和三元组损失训练，最终通过简单的"嵌入到标签"（Embedding-to-Label）过程直接解码出一致的 3D 分割标签，无需 HDBSCAN 等后处理聚类步骤，训练时间从 15+ 小时缩短至 40 分钟以内。

## 研究背景与动机

### 问题定义

3D 场景理解是 AR/VR、自动驾驶和路径规划等领域的关键任务。现有方法通常通过将 2D 分割标签"提升"（lift）到 3D 表征（如 NeRF、3DGS）来实现 3D 分割。然而，2D 分割模型在不同视角下生成的实例标签是**不一致的**——同一物体在不同视图中可能被分配不同的实例 ID。

### 现有方法的局限

**两阶段方法（预处理+分割）**：如 Panoptic-Lifting 和 DM-NeRF，依赖线性分配问题（Linear Assignment Problem）来匹配 2D 预测和 3D 表示，计算开销极大，单场景训练需 20+ 小时。

**两阶段方法（对比学习+聚类）**：如 Contrastive-Lift，通过对比学习优化特征嵌入后，需要额外使用 HDBSCAN 聚类算法进行后处理来分配标签，引入超参数敏感性，训练需 15+ 小时。

**特征蒸馏方法**：如 DFF，将 CLIP/DINO 等高维特征蒸馏到 3D 表示中，但训练速度极慢（约 2 天）。

### 核心动机

能否将对比学习和标签解码统一为一个单阶段过程？作者观察到，当嵌入空间被约束到 $[0,1]^d$（通过 sigmoid）后，对比损失会自然驱动不同实例的嵌入收敛到超立方体的不同角点（corner），每个角点对应一个唯一的二进制编码，直接解码即可得到离散标签。这一洞察使得后处理聚类变得完全不必要。

## 方法详解

### 整体框架

UniC-Lift 构建在 3DGS 之上。核心思路是为每个 3D 高斯基元添加一个 $d$ 维可学习向量嵌入 $\boldsymbol{v} \in \mathbb{R}^d$，通过可微渲染得到 2D 嵌入图，施加对比损失和三元组损失进行优化，最后通过阈值化和二进制解码直接获得实例标签。

整体流程：
- 输入：多视图 RGB 图像 + 相机位姿 + 2D 分割掩码（可以不一致）
- 3DGS 优化：同时优化颜色参数和向量嵌入
- 损失：渲染损失 + 聚类对比损失 + 三元组损失 + 3D 邻域正则化
- 推理：将渲染的嵌入通过 sigmoid → 阈值化 → 二进制解码 → 实例标签

### 关键设计

#### 1. **可学习向量嵌入的渲染**

每个 3D 高斯基元附加一个 $d$ 维向量嵌入 $\boldsymbol{v}$（视角无关），通过与颜色相同的 alpha 混合方式渲染到 2D 平面：

$$\boldsymbol{\mathcal{V}} = \sum_{i \in T} \boldsymbol{v}_i \alpha'_i \prod_{j=1}^{i-1}(1-\alpha'_j)$$

设计动机：利用 3DGS 的可微渲染框架，使嵌入的优化能自然考虑 3D 几何关系，实现多视图一致性。

#### 2. **聚类对比损失（Cluster Loss）**

对于每个视角的渲染嵌入图 $\mathbb{V} \in \mathbb{R}^{H \times W \times d}$，根据 2D 分割掩码将像素划分为 $K$ 个不相交集合 $\{\Omega_1, ..., \Omega_K\}$，计算每个集合的质心 $\boldsymbol{m}_{\Omega_i}$，然后最小化同类嵌入到质心的距离、最大化不同类质心间的距离：

$$\mathcal{L}_{cluster} = \sum_{\Omega_i} \sum_{u \in \Omega_i} \|\mathbb{V}(u) - \boldsymbol{m}_{\Omega_i}\|_2^2 - \sum_{i \neq j} \|\boldsymbol{m}_{\Omega_i} - \boldsymbol{m}_{\Omega_j}\|_2^2$$

设计动机：拉近同一实例的嵌入，推远不同实例的嵌入，是标准的对比学习目标。

#### 3. **三元组损失与边界硬采样**

直接在渲染嵌入上施加聚类损失无法保证一致的惩罚。因此，先将嵌入通过 sigmoid 约束到 $[0,1]$ 范围，再通过一个**线性层** $\mathbb{W}$ 投影，在投影空间中计算三元组损失：

$$\mathcal{L}_{triplet} = \sum_{(a,p,n) \in \Delta} \max(0, \|a-p\|_2^2 - \|a-n\|_2^2 + \delta)$$

**关键设计**：正样本和负样本从**分割边界**采样而非随机采样。原因是边界处的三元组提供非零梯度，比随机三元组更具信息量，能加速收敛（25k 迭代 vs 50k 迭代达到相同质量）。

**线性层的作用**：直接在特征嵌入上施加硬挖掘被证明不稳定。在渲染嵌入上施加线性变换后再计算三元组损失，能稳定训练并显著提升性能。

#### 4. **3D 邻域正则化**

对每个高斯基元 $i$，构建空间邻域 $\mathcal{N}(i) = \{\|\mu_i - \mu_j\|_2^2 \leq \tau\}$，惩罚邻近高斯的嵌入差异：

$$\mathcal{L}_{3D} = \sum_{i=1}^{|\mathcal{G}|} \sum_{j \in \mathcal{N}(i)} \|\boldsymbol{v}_i - \boldsymbol{v}_j\|_2^2$$

此损失在 15000 次迭代后（自适应密度控制稳定后）才启用，避免干扰高斯基元的分裂/克隆过程。

#### 5. **Embedding-to-Label 过程**

这是本文最核心的创新。整个解码过程极其简洁：

1. 对渲染嵌入施加 sigmoid：$\hat{\boldsymbol{\mathcal{V}}} = \sigma(\boldsymbol{\mathcal{V}})$
2. 阈值化为二进制向量：$\tilde{\boldsymbol{\mathcal{V}}} = \mathbf{1}[\hat{\boldsymbol{\mathcal{V}}} > 0.5]$
3. 二进制解码为标签：$l = \sum_k \tilde{\boldsymbol{\mathcal{V}}}_k \cdot 2^{k-1}$

这使得推理时标签预测的时间复杂度为 $O(n)$（$n$ 为像素数），而 Contrastive-Lift 需要 $O(n \log c)$（$c$ 为聚类数），显著提升推理速度。

### 损失函数 / 训练策略

总损失：

$$\mathcal{L}_{total} = \mathcal{L}_{rendering} + \lambda_{cluster} \mathcal{L}_{cluster} + \lambda_{triplet} \mathcal{L}_{triplet} + \lambda_{3D} \mathcal{L}_{3D}$$

超参数设置：$\lambda_{cluster} = \lambda_{triplet} = \lambda_{3D} = 0.1$，三元组 margin $\delta = 1$，邻域阈值 $\tau = 0.01$。嵌入维度 $d = 12$，最大三元组数为 3000。

训练策略：
- 使用 ADAM 优化器，学习率 $1 \times 10^{-4}$
- 总训练 30k 迭代，单卡 RTX A6000
- 分割损失的梯度不参与 3DGS 的自适应密度控制
- 三元组损失和 3D 损失在自适应密度控制完成后才启用

## 实验关键数据

### 主实验

**ScanNet 和 Replica3D（PQ^scene 指标）**：

| 数据集 | DM-NeRF | Panoptic-Lifting | Contrastive-Lift | Gaussian-Grouping | **UniC-Lift** |
|--------|---------|-------------------|------------------|-------------------|---------------|
| ScanNet | 41.7 | 58.9 | 62.3 | 61.83 | **63.0** |
| Replica3D | 44.1 | 57.9 | 59.1 | 66.52 | **88.7** |

在 Replica3D 上，UniC-Lift 比 Gaussian Grouping 高出 1.3 倍（88.7 vs 66.52）！

**Messy-Rooms 数据集（PQ^scene 指标，不同物体数量）**：

| 方法 | 25obj(Old) | 50obj(Old) | 100obj(Old) | 500obj(Old) | 均值 |
|------|------------|------------|-------------|-------------|------|
| Panoptic-Lifting | 73.2 | 69.9 | 64.3 | 51.0 | 63.2 |
| Contrastive-Lift | 78.9 | 75.8 | 69.1 | 55.0 | 69.0 |
| **UniC-Lift** | **86.0** | **79.1** | **70.8** | **57.4** | **71.5** |

在 8 个场景中的 6 个取得最优。

### 消融实验

**各损失函数的影响（Replica3D）**：

| 配置 | PQ^scene | mIoU | 说明 |
|------|----------|------|------|
| CL + 3D Reg | 88.0 | 94.4 | 无三元组损失 |
| 仅 CL | 83.7 | 91.8 | 无三元组无 3D 正则 |
| CL + Triplet(MLP) | 89.0 | 95.2 | 无 3D 正则 |
| CL + Triplet(无MLP) + 3D | 88.0 | 94.0 | 无线性投影层 |
| **全部（CL+Triplet(MLP)+3D）** | **89.0** | **95.4** | 本文最终配置 |

**训练时间比较（Replica 单场景，NVIDIA A6000）**：

| 方法 | 训练时间 |
|------|----------|
| Panoptic-Lifting | >20 小时 |
| Contrastive-Lift | >15 小时 |
| **UniC-Lift** | **<40 分钟** |

### 关键发现

1. **Embedding-to-Label 过程消除了聚类后处理**：与 Contrastive-Lift+3DGS 相比，UniC-Lift 在保持相当量化指标的同时，完全消除了聚类步骤（42 分钟 vs 85 分钟）。
2. **边界硬采样显著加速收敛**：使用边界三元组在 25k 迭代达到的效果，随机三元组需要 50k 迭代。
3. **低分辨率掩码训练不影响结果**：使用 0.5x 分辨率掩码训练与全分辨率结果视觉上无明显差异。
4. **少量掩码即可训练**：仅使用 5% 的分割掩码就能获得与全量掩码训练接近的结果。

## 亮点与洞察

1. **核心创新的优雅性**：sigmoid 约束空间中对比学习自然驱动嵌入到角点的观察极为巧妙，使得标签解码变成简单的二进制编码，这是一个非常自然且优美的数学发现。
2. **从 $O(n \log c)$ 到 $O(n)$**：推理复杂度的降低具有实际意义，尤其对大规模场景。
3. **训练加速 20-30 倍**：从 15-20 小时缩短至 40 分钟，使得方法具有实际应用价值。
4. **下游应用**：高质量的 3D 分割直接支持物体提取和场景编辑，展示了方法的实用性。

## 局限与展望

1. **嵌入维度限制实例数量**：$d=12$ 维嵌入理论上最多支持 $2^{12} = 4096$ 个实例，对极大规模场景可能不够。
2. **仅限静态场景**：未扩展到动态场景。
3. **依赖 2D 分割质量**：虽然不要求一致性，但仍需合理的 2D 分割作为输入。
4. **边界区域仍存在伪影**：虽然硬采样策略缓解了边界问题，但作者指出这一问题并未完全解决。

## 相关工作与启发

- **Contrastive-Lift**：本文最直接的改进基础，UniC-Lift 将其两阶段合并为单阶段。
- **3DGS 系列**：利用 3DGS 的高效渲染和可扩展属性存储。
- **二进制编码思想**：嵌入到角点的思想可以推广到其他需要离散化表示的场景中。

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — Embedding-to-Label 思想极为优雅和新颖
- **实验充分度**: ⭐⭐⭐⭐ — 三个数据集、充分消融，缺少更大规模场景实验
- **写作质量**: ⭐⭐⭐⭐ — 动机清晰，玩具实验直观展示核心思想
- **实用价值**: ⭐⭐⭐⭐⭐ — 20-30 倍加速使实际应用成为可能

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] PCF-Lift: Panoptic Lifting by Probabilistic Contrastive Fusion](../../ECCV2024/3d_vision/pcf-lift_panoptic_lifting_by_probabilistic_contrastive_fusion.md)
- [\[AAAI 2026\] ASSIST-3D: Adapted Scene Synthesis for Class-Agnostic 3D Instance Segmentation](assist-3d_adapted_scene_synthesis_for_class-agnostic_3d_instance_segmentation.md)
- [\[AAAI 2026\] Retrieving Objects from 3D Scenes with Box-Guided Open-Vocabulary Instance Segmentation](retrieving_objects_from_3d_scenes_with_box-guided_open-vocabulary_instance_segme.md)
- [\[CVPR 2026\] CompetitorFormer: Mitigating Query Conflicts for 3D Instance Segmentation via Competitive Strategy](../../CVPR2026/3d_vision/competitorformer_mitigating_query_conflicts_for_3d_instance_segmentation_via_com.md)
- [\[ICLR 2026\] Learning Unified Representation of 3D Gaussian Splatting](../../ICLR2026/3d_vision/learning_unified_representation_of_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
