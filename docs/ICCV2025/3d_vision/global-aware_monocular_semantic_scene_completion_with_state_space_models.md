---
title: >-
  [论文解读] Global-Aware Monocular Semantic Scene Completion with State Space Models
description: >-
  [ICCV 2025][3D视觉][语义场景补全] 提出GA-MonoSSC，一种结合Transformer（2D全局上下文）和Mamba（3D长程依赖）的混合架构用于室内单目语义场景补全，创新引入Frustum Mamba Layer解决体素序列化中的特征不连续性问题，在Occ-ScanNet和NYUv2上达到SOTA。
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "语义场景补全"
  - "室内场景"
  - "状态空间模型"
  - "Mamba"
  - "单目"
  - "Transformer"
---

# Global-Aware Monocular Semantic Scene Completion with State Space Models

**会议**: ICCV 2025  
**arXiv**: [2503.06569](https://arxiv.org/abs/2503.06569)  
**代码**: 即将公开  
**领域**: 3D视觉  
**关键词**: 语义场景补全, 室内场景, 状态空间模型, Mamba, 单目, Transformer

## 一句话总结

提出GA-MonoSSC，一种结合Transformer（2D全局上下文）和Mamba（3D长程依赖）的混合架构用于室内单目语义场景补全，创新引入Frustum Mamba Layer解决体素序列化中的特征不连续性问题，在Occ-ScanNet和NYUv2上达到SOTA。

## 研究背景与动机

语义场景补全（SSC）从输入中重建完整3D场景并标注语义，在机器人导航、AR和自动驾驶中至关重要。**室内单目SSC**尤为挑战——仅用一张图像推断3D几何和语义。

**CNN架构的两大瓶颈**：

**2D特征提取的非均匀分布问题**：透视投影导致近处点分散、远处点密集。CNN的固定权重无法适应这种不均匀分布，导致2D特征表示不充分

**3D空间中长程依赖的缺失**：3D-to-2D投影天然丢失深度信息。恢复丢失信息需要全局3D依赖建模，但CNN受限于局部感受野

**为什么不直接用Transformer？** 在3D空间中，Transformer的二次复杂度面对大量体素时计算量爆炸。需要更高效的替代方案。

**为什么Mamba直接处理3D体素有问题？** 标准做法是将体素按3D坐标展平为序列。但2D-to-3D反投影过程中，同一射线上的体素共享相似特征，而按3D坐标排列时这些体素可能被分割到序列的不同位置，产生**特征不连续性**。

## 方法详解

### 整体框架

输入单目图像 → **双头多模态编码器(DMenc)**提取全局感知的2D语义/几何特征 →
FLoSP反投影到3D → **Frustum Mamba解码器(FMdec)**捕获3D长程依赖 → 预测头输出语义占据图

### 双头多模态编码器(DMenc)

利用Transformer编码器提取全局上下文的2D特征，然后分两路解码：

$$\mathcal{TK} = \mathbf{Enc}(\mathcal{X}^{2d})$$

- **几何解码器** $\mathbf{Dec}_{\text{geo}}$：提取多尺度几何特征 $\mathcal{F}^{\text{geo}}$，通过深度预测转3D占据进行额外监督
- **语义解码器** $\mathbf{Dec}_{\text{sem}}$：提取多尺度语义特征 $\mathcal{F}^{\text{sem}}$，融入几何信息

设计动机：分离几何和语义学习，让模型捕获更细粒度的表示。语义特征在每个尺度融合几何特征：

$$\mathcal{F}_l^{\text{sem}} = \mathcal{F}_l^{\text{sem}} + \mathcal{F}_l^{\text{geo}}$$

### Frustum Mamba解码器(FMdec)

#### 多尺度信息融合

通过通道注意力自适应加权不同尺度的3D特征：

$$W^{\text{geo}} = \mathbf{MLP}(\mathbf{AvgPool}(\mathcal{F}^{\text{geo},3d}))$$
$$\mathcal{F}^{\text{geo},3d} = \sum_{l=1}^{N_l} \mathcal{F}_l^{\text{geo},3d} \cdot w_l$$

语义和几何特征通过逐元素加法融合：$\mathcal{F}^{3d} = \mathcal{F}^{\text{sem},3d} + \mathcal{F}^{\text{geo},3d}$

#### Frustum Mamba Layer

**核心创新**——在视锥空间而非3D空间中序列化体素：

**问题**：标准方法按3D坐标 $(x,y,z)$ 展平体素。但由于2D-to-3D反投影，相邻体素的特征来自不同图像位置，在序列中产生跳跃式不连续。

**解决**：Frustum Reordering Strategy——将体素按视锥空间（图像平面坐标+深度）排序。这样：
- 同一射线上的体素在序列中相邻（深度方向连续）
- 图像平面上相邻的体素特征相似
- 更好地匹配Mamba的顺序扫描机制

$$\mathcal{SFF}^{3d} = \mathbf{FReorder}(\mathcal{FF}^{3d})$$
$$\mathcal{SFF'}^{3d} = \mathbf{MambaLayers}(\mathcal{SFF}^{3d})$$
$$\mathcal{F'}^{3d} = \mathbf{LN}(\mathbf{Composit}(\mathcal{SFF'}^{3d})) + \mathcal{F}^{3d}$$

每个stage包含Frustum Mamba Layer + 下采样，实现多尺度3D特征提取。

### 监督信号

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{rel}} + \mathcal{L}_{\text{scal}}^{\text{sem}} + \mathcal{L}_{\text{scal}}^{\text{geo}} + \mathcal{L}_{\text{fp}}$$

- $\mathcal{L}_{\text{CE}}$：最终语义预测的交叉熵
- $\mathcal{L}_{\text{BCE}}$：几何解码器的占据预测
- $\mathcal{L}_{\text{scal}}^{\text{sem/geo}}$：场景-类别亲和力损失
- $\mathcal{L}_{\text{fp}}$：视锥比例损失
- $\mathcal{L}_{\text{rel}}$：上下文关系损失

## 实验

### Occ-ScanNet主实验

| 方法 | 输入 | IoU | mIoU |
|:---|:---:|:---:|:---:|
| MonoScene | RGB | 25.41 | 10.66 |
| NDC-Scene | RGB | 30.16 | 14.13 |
| ISO | RGB | 31.96 | 15.43 |
| **GA-MonoSSC** | **RGB** | **33.21** | **17.82** |

### NYUv2主实验

| 方法 | IoU | mIoU |
|:---|:---:|:---:|
| MonoScene | 37.12 | 25.17 |
| NDC-Scene | 43.93 | 29.47 |
| ISO | 42.98 | 29.84 |
| **GA-MonoSSC** | **44.89** | **31.52** |

### 消融实验

| 组件 | IoU | mIoU | 说明 |
|:---|:---:|:---:|:---|
| Baseline (CNN编码+CNN解码) | 30.16 | 14.13 | NDC-Scene |
| +Transformer编码器 | 31.45 | 15.21 | 全局2D上下文有效 |
| +双头分离 | 32.03 | 16.14 | 语义/几何分离提升细节 |
| +Mamba (3D坐标排序) | 32.56 | 16.89 | 长程依赖有效但有不连续问题 |
| **+Frustum Reordering** | **33.21** | **17.82** | **视锥排序解决特征不连续** |

### 关键发现

1. **Transformer vs CNN编码器**：Transformer在2D域建模全局关系显著优于CNN，尤其对非均匀投影点分布
2. **双头解码的价值**：分离几何和语义学习比联合学习更有效，几何特征作为结构先验注入语义
3. **Frustum排序的必要性**：3D坐标排序 vs 视锥排序是~1个mIoU的差异，验证了特征不连续问题的存在
4. **Mamba vs Transformer in 3D**：Mamba以线性复杂度实现与Transformer相当的长程建模能力

## 亮点与洞察

1. **2D Transformer + 3D Mamba的混合范式**：在计算效率和建模能力间取得最优平衡
2. **Frustum空间的洞察**：认识到反投影的特征沿射线方向连续是关键先验
3. **双模态分离**：几何和语义的解耦学习在SSC任务中被证明有效

## 局限性

1. Mamba的单向扫描可能错过某些方向的依赖关系
2. 仅验证室内场景，户外大规模场景的泛化性未知
3. 视锥排序增加了预处理复杂度

## 相关工作

- **单目SSC**：MonoScene, NDC-Scene, ISO
- **状态空间模型**：Mamba, S4, VMamba
- **3D点云SSM**：Hilbert曲线排序, 八叉树排序

## 评分

- 新颖性：⭐⭐⭐⭐ — Frustum Mamba Layer是有洞察力的创新设计
- 技术深度：⭐⭐⭐⭐ — Transformer+Mamba混合架构+双头编码设计完整
- 实验完整性：⭐⭐⭐⭐ — 两个数据集SOTA+充分消融
- 实用价值：⭐⭐⭐ — 室内单目SSC应用明确，但实时性待验证

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Monocular Semantic Scene Completion via Masked Recurrent Networks](monocular_semantic_scene_completion_via_masked_recurrent_networks.md)
- [\[ICCV 2025\] UST-SSM: Unified Spatio-Temporal State Space Models for Point Cloud Video Modeling](ust-ssm_unified_spatio-temporal_state_space_models_for_point_cloud_video_modelin.md)
- [\[ICCV 2025\] MeshMamba: State Space Models for Articulated 3D Mesh Generation and Reconstruction](meshmamba_state_space_models_for_articulated_3d_mesh_generation_and_reconstructi.md)
- [\[NeurIPS 2025\] MVSMamba: Multi-View Stereo with State Space Model](../../NeurIPS2025/3d_vision/mvsmamba_multi-view_stereo_with_state_space_model.md)
- [\[ICCV 2025\] Disentangling Instance and Scene Contexts for 3D Semantic Scene Completion](disentangling_instance_and_scene_contexts_for_3d_semantic_scene_completion.md)

</div>

<!-- RELATED:END -->
