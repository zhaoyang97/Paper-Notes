---
title: >-
  [论文解读] UST-SSM: Unified Spatio-Temporal State Space Models for Point Cloud Video Modeling
description: >-
  [ICCV 2025][3D视觉][点云视频] 提出UST-SSM，通过时空选择扫描(STSS)、时空结构聚合(STSA)和时序交互采样(TIS)三个核心模块，将选择性状态空间模型扩展到点云视频分析，以线性复杂度实现优于Transformer的性能。 现有痛点 现有痛点：点云视频建模面临三重挑战： 时空无序：点云视频缺乏一致…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "点云视频"
  - "状态空间模型"
  - "时空建模"
  - "动作识别"
  - "Mamba"
---

# UST-SSM: Unified Spatio-Temporal State Space Models for Point Cloud Video Modeling

**会议**: ICCV 2025  
**arXiv**: [2508.14604](https://arxiv.org/abs/2508.14604)  
**代码**: [GitHub](https://github.com/wangzy01/UST-SSM)  
**领域**: 3D视觉  
**关键词**: 点云视频, 状态空间模型, 时空建模, 动作识别, Mamba

## 一句话总结

提出UST-SSM，通过时空选择扫描(STSS)、时空结构聚合(STSA)和时序交互采样(TIS)三个核心模块，将选择性状态空间模型扩展到点云视频分析，以线性复杂度实现优于Transformer的性能。

## 研究背景与动机

### 现有痛点

**现有痛点**：点云视频建模面临三重挑战：

**时空无序**：点云视频缺乏一致的时空排序，与SSM的单向扫描范式冲突

**局部几何丢失**：序列化策略破坏邻近点间的几何关系

**时序交互有限**：传统采样策略创建碎片化的时序上下文

现有方案的不足：

### 领域现状

**领域现状**：Mamba4D采用时序顺序扫描，限制查询点只能获取前一帧信息

### 核心矛盾

**核心矛盾**：SSM存在远距离token交互衰减问题

### 解决思路

**解决思路**：语义相似但时空距离远的点被分离到很远位置

## 方法详解

### 时序交互采样 (TIS)

传统单步长采样（stride=2）中锚帧只与直接邻居交互。TIS通过两步采样扩展时序感受野：

第一步（stride=1）：
$$\mathbf{Feat}_{T_i}' = \mathcal{F}(\mathcal{S}(T_{i-1}, T_i, T_{i+1}))$$

第二步（stride=2）：
$$\mathbf{Feat}_{T_{2i}} = \bar{\mathcal{F}}(\bar{\mathcal{S}}(\mathbf{Feat}_{T_{2i-1}}', \mathbf{Feat}_{T_{2i}}, \mathbf{Feat}_{T_{2i+1}}'))$$

非锚帧被多次复用，每个锚帧可访问所有前序帧信息。

### 时空选择扫描 (STSS)

不同于时序顺序扫描，STSS根据语义相似性聚类：

1. 轻量Prompt网络生成分类矩阵 $\mathcal{P} \in \mathbb{R}^{N \times K}$
2. 按语义类别聚类点：$\mathbf{X}_j = \{x_i | \arg\max(p_i) = j\}$
3. 簇内Hilbert排序保留局部几何：$\bar{\mathbf{X}}_j = \text{HilbertSort}(\mathbf{X}_j)$
4. 簇间按时间顺序排列保持运动连续性

### 时空结构聚合 (STSA)

通过4D KNN恢复被序列化破坏的局部几何关系：

**4D邻域构建**：加入时序嵌入 $\mathbf{E}^t$：
$$K = \underset{\mathbf{X}_K \in X}{\text{argmin}}(|\mathbf{X}_C - \mathbf{X}_K| + |\mathbf{E}_C^t - \mathbf{E}_K^t|)$$

**特征传播**：归一化邻居特征并与绝对特征拼接：
$$\mathbf{F}_K' = \frac{\mathbf{F}_K - \mathbf{F}_C}{|\mathbf{F}_K - \mathbf{F}_C|_2 + \epsilon} \oplus \mathbf{F}_C$$

**自适应池化**：指数加权池化替代标准卷积：
$$\mathbf{F}_C' = \text{MLP}\left(\sum_{i=1}^K \frac{e^{\mathbf{F}_K'^i}}{\sum_j e^{\mathbf{F}_K'^j}} \cdot \mathbf{F}_K'^i\right)$$

## 实验

### MSR-Action3D


### 主实验

| 骨干 | 方法 | 24帧 Acc(%) | 36帧 Acc(%) |
|------|------|------------|------------|
| CNN | MeteorNet | 88.50 | - |
| CNN | Kinet | 93.27 | - |
| Transformer | PST-Transformer | 93.73 | 91.15 |
| Transformer | LeaF | 93.84 | - |
| SSM | MAMBA4D | 92.68 | 93.23 |
| SSM | **UST-SSM** | **94.77** | **95.12** |

UST-SSM在所有帧数设置下取得最优，且随帧数增加性能稳定提升（36帧95.12%最优）。

### NTU RGB+D


### 消融实验

| 方法 | Cross-Subject | Cross-View |
|------|--------------|------------|
| P4Transformer | 90.2 | 96.4 |
| PST-Transformer | 91.0 | 96.4 |
| **UST-SSM** | SOTA | SOTA |

### 效率对比

UST-SSM以线性复杂度在运行时间、GPU显存和精度三方面优于P4Transformer和PST-Transformer（二次复杂度）。

## 亮点与洞察

1. **STSS解决根本矛盾**：通过语义聚类将时空距离远但语义相似的点聚近，缓解单向建模的远距离衰减
2. **TIS的非锚帧复用**：巧妙扩展时序感受野而不增加采样复杂度
3. **STSA与SSM互补**：前者处理局部交互，后者处理全局序列依赖
4. **Hilbert排序保留几何**：在语义聚类后仍保持局部空间连贯性

## 局限与展望

- Prompt网络增加额外计算开销
- 聚类数K是超参数，不同数据集可能需要调整
- 4D KNN的时序嵌入需要额外学习
- 未在大规模自动驾驶点云数据上验证

## 相关工作

- P4Transformer, PST-Transformer: Transformer-based点云视频
- Mamba4D: 首个将SSM应用于点云视频
- PointMamba, PCM: SSM用于3D点云（仅空间）

## 评分

- 新颖性: ⭐⭐⭐⭐ (STSS+语义聚类的扫描策略新颖)
- 技术深度: ⭐⭐⭐⭐⭐ (三个模块各解决一个核心问题)
- 实验充分度: ⭐⭐⭐⭐ (三个数据集+效率分析)
- 实用价值: ⭐⭐⭐⭐ (线性复杂度适合长期建模)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Global-Aware Monocular Semantic Scene Completion with State Space Models](global-aware_monocular_semantic_scene_completion_with_state_space_models.md)
- [\[ICCV 2025\] MeshMamba: State Space Models for Articulated 3D Mesh Generation and Reconstruction](meshmamba_state_space_models_for_articulated_3d_mesh_generation_and_reconstructi.md)
- [\[CVPR 2025\] Mesh Mamba: A Unified State Space Model for Saliency Prediction in Non-Textured and Textured Meshes](../../CVPR2025/3d_vision/mesh_mamba_a_unified_state_space_model_for_saliency_prediction_in_non-textured_a.md)
- [\[NeurIPS 2025\] MVSMamba: Multi-View Stereo with State Space Model](../../NeurIPS2025/3d_vision/mvsmamba_multi-view_stereo_with_state_space_model.md)
- [\[ICCV 2025\] StruMamba3D: Exploring Structural Mamba for Self-supervised Point Cloud Representation Learning](strumamba3d_exploring_structural_mamba_for_self-supervised_point_cloud_represent.md)

</div>

<!-- RELATED:END -->
