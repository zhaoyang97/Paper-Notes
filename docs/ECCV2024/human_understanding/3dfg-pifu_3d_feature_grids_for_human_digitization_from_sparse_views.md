---
title: >-
  [论文解读] 3DFG-PIFu: 3D Feature Grids for Human Digitization from Sparse Views
description: >-
  [ECCV 2024][人体理解][人体重建] 本文提出 3DFG-PIFu，通过引入3D特征网格（3D Feature Grids）在整个 pipeline 中全局融合多视图特征，替代传统逐点局部融合方式，并结合迭代网格精炼机制和基于 SDF 的 SMPL-X 特征，显著超越现有稀疏视图人体数字化 SOTA 方法。
tags:
  - "ECCV 2024"
  - "人体理解"
  - "人体重建"
  - "多视图重建"
  - "像素对齐隐式函数"
  - "3D特征网格"
  - "SMPL-X"
---

# 3DFG-PIFu: 3D Feature Grids for Human Digitization from Sparse Views

**会议**: ECCV 2024  
**代码**: [https://github.com/kcyt/3DFG-PIFu](https://github.com/kcyt/3DFG-PIFu)  
**领域**: 其他  
**关键词**: 人体重建, 多视图重建, 像素对齐隐式函数, 3D特征网格, SMPL-X

## 一句话总结

本文提出 3DFG-PIFu，通过引入3D特征网格（3D Feature Grids）在整个 pipeline 中全局融合多视图特征，替代传统逐点局部融合方式，并结合迭代网格精炼机制和基于 SDF 的 SMPL-X 特征，显著超越现有稀疏视图人体数字化 SOTA 方法。

## 研究背景与动机

**领域现状**：从稀疏视图重建穿着衣物的人体3D模型是计算机视觉中的重要问题。当前主流方法基于像素对齐隐式模型（Pixel-aligned Implicit Functions），如 Multi-view PIFu、DeepMultiCap、DoubleField 和 SeSDF 等，这些方法通过将2D图像特征与3D查询点对齐来预测占用场或SDF值。

**现有痛点**：给定 V 张多视图图像，现有方法仅在 pipeline 的最末端以逐点（point-wise）和局部化（localized）的方式融合来自不同视图的特征。换言之，V 张图像在绝大部分处理流程中是被独立处理的，只在最后一步才被非常窄地结合在一起。这在很大程度上违背了使用多视图信息的初衷——多视图任务本质上被当作了单视图任务来处理。

**核心矛盾**：多视图信息在空间上是互补和关联的，但现有方法的融合策略过于局部和延后，无法充分利用不同视角提供的全局上下文信息。这种延迟融合导致重建质量受限于单视图特征的表达能力，跨视图的一致性和完整性难以保证。

**本文目标** (1) 如何在整个 pipeline 中（而非仅在末端）全局地融合多视图特征？(2) 如何利用已有重建结果进行迭代精炼？(3) 如何更有效地将参数化人体模型（SMPL-X）的先验信息融入像素对齐隐式模型？

**切入角度**：作者观察到，如果能将多视图特征统一投射到一个共享的3D特征空间中，就可以在任意处理阶段实现全局融合。3D体素网格天然适合作为这样的共享空间，因为它可以接受任意视角的特征投影。

**核心 idea**：用3D特征网格作为多视图特征的全局融合载体，使多视图信息在 pipeline 的每个阶段都能被充分利用。

## 方法详解

### 整体框架

3DFG-PIFu 的整体 pipeline 如下：输入为 V 张稀疏视图的人体图像及对应的相机参数，输出为重建的3D人体网格。处理流程包含三个核心阶段：(1) 通过图像编码器提取每张图的2D特征，将其反投影到3D特征网格中实现全局融合；(2) 对于每个3D查询点，从3D特征网格和2D像素对齐特征中同时提取信息，通过MLP预测其占用值/SDF值；(3) 通过迭代精炼机制，将初始重建的网格再次投影到各视图获取更新特征，反复迭代提升重建质量。

### 关键设计

1. **3D Feature Grids（3D特征网格）**:

    - 功能：在整个 pipeline 中全局融合多视图特征
    - 核心思路：首先用图像编码器（如 ResNet 或 HRNet）从每张输入图像提取多尺度2D特征图。然后利用已知的相机参数，将每张特征图上的特征反投影（unproject）到一个统一的3D体素网格中。具体来说，对于3D网格中的每个体素，计算其在每个视图中的投影位置，从对应特征图中双线性插值取得特征，再将来自所有视图的特征进行聚合（如均值池化或注意力加权）。这样得到的3D特征网格编码了来自所有视图的全局空间信息。查询任意3D点时，只需在该网格中进行三线性插值即可获得全局多视图特征。
    - 设计动机：传统方法在最后阶段才逐点融合多视图特征，信息融合范围极窄。3D特征网格将融合提前到特征空间构建阶段，且融合范围覆盖整个3D空间而非单个查询点，从根本上解决了多视图信息利用不充分的问题。

2. **Iterative Refinement（迭代精炼机制）**:

    - 功能：利用已有重建结果反复精炼人体网格，逐步提升重建质量
    - 核心思路：第一轮前向传播生成初始人体网格后，将该网格从不同视角渲染（或投影）获取轮廓/深度信息，与原始图像拼接作为新的输入，再次通过编码器和3D特征网格提取更丰富的特征，生成精炼后的网格。这个过程可以迭代多次，每次利用上一轮的重建结果提供更好的几何线索。
    - 设计动机：单次前向传播往往无法准确重建所有细节，尤其是遮挡区域和几何复杂区域。迭代精炼类似于 coarse-to-fine 的策略，让模型有机会在已知粗略几何的基础上修正错误和补全细节。

3. **SDF-based SMPL-X Features（基于SDF的SMPL-X特征）**:

    - 功能：将参数化人体模型 SMPL-X 的先验信息有效融入隐式重建模型
    - 核心思路：不同于以往直接使用 SMPL-X 网格的表面距离或基于体素的 inside/outside 标记，本文计算每个3D查询点到 SMPL-X 网格表面的有符号距离场（SDF）值，作为额外的输入特征提供给占用预测 MLP。SDF 值是连续的且具有明确的几何含义——正值表示点在体外，负值表示在体内，零值表示体表。这种表示比简单的二值标记更加平滑且信息量更丰富。
    - 设计动机：SMPL-X 提供了强大的人体形状先验，但需要合适的方式将其与自由形状的隐式表示结合。SDF 表示是一种自然的选择，因为它本身就是连续的隐式表示，与 PIFu 模型的隐式预测框架在数学形式上一致，可以无缝集成。

### 损失函数 / 训练策略

训练采用标准的点云采样策略，在人体表面附近和空间中随机采样3D查询点，使用二值交叉熵损失（BCE Loss）或 L1 损失来监督占用值/SDF值的预测。训练时使用 ground truth 的 SMPL-X 拟合结果计算 SDF 特征。迭代精炼阶段在训练时采用端到端训练或分阶段训练策略，逐步解锁更多迭代轮次。

## 实验关键数据

### 主实验

实验在 THuman2.0 和 RenderPeople 等标准人体重建数据集上进行评估，使用 Chamfer Distance (CD)、Normal Consistency (NC) 和 Point-to-Surface (P2S) 距离作为指标。

| 数据集 | 指标 | 3DFG-PIFu | Multi-view PIFu | SeSDF | 提升 |
|--------|------|-----------|-----------------|-------|------|
| THuman2.0 | Chamfer Distance ↓ | **最优** | 基线 | 次优 | 显著提升 |
| THuman2.0 | Normal Consistency ↑ | **最优** | 基线 | 次优 | 显著提升 |
| RenderPeople | P2S ↓ | **最优** | 基线 | 次优 | 明显改善 |

3DFG-PIFu 在所有指标上显著超越 Multi-view PIFu、DeepMultiCap、DoubleField 和 SeSDF 等现有 SOTA 方法。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 基线 (无3D特征网格) | CD 较高 | 退化为传统逐点融合方式 |
| +3D Feature Grids | CD 显著降低 | 全局特征融合带来明显提升 |
| +Iterative Refinement | CD 进一步降低 | 迭代精炼有效修正重建错误 |
| +SDF-based SMPL-X | CD 最低 | 人体先验特征进一步改善几何细节 |
| 不同网格分辨率 | 随分辨率增加先升后降 | 存在最优的3D网格分辨率 |
| 不同迭代次数 | 2轮效果最佳 | 过多迭代收益递减 |

### 关键发现

- 3D特征网格是提升最大的单一因素，说明全局多视图融合至关重要
- 迭代精炼在遮挡区域和几何复杂区域（如手部、衣物褶皱）效果尤为明显
- SDF-based SMPL-X 特征对人体各部位的重建均有稳定提升，尤其是四肢区域
- 在极稀疏视图（如2-3视图）条件下，3DFG-PIFu 相比现有方法的优势更加显著

## 亮点与洞察

- **全局融合思路的通用性**：3D特征网格的思路不仅适用于人体重建，可以推广到任意多视图3D重建任务。核心insight是将"延迟融合"变为"早期全局融合"
- **SDF统一表示**：用 SDF 作为参数化模型先验和隐式重建之间的桥梁，是一种优雅的设计选择
- **迭代精炼的轻量实现**：不需要额外训练精炼网络，而是通过重用主网络实现，增加的计算开销可控

## 局限与展望

- 3D特征网格的分辨率受限于 GPU 内存，高分辨率网格虽能捕捉更多细节但计算成本过高
- 迭代精炼增加了推理时间，实时应用场景下可能成为瓶颈
- 依赖 SMPL-X 拟合的准确性，拟合失败时 SDF 特征可能引入噪声
- 对极端姿态和松散衣物的重建仍有改进空间
- 未探索将方法扩展到动态序列重建的可能性

## 相关工作与启发

- **vs Multi-view PIFu**: Multi-view PIFu 仅在最终阶段逐点融合特征，3DFG-PIFu 通过3D特征网格在全 pipeline 中全局融合，从根本上改变了特征融合范式
- **vs SeSDF**: SeSDF 引入了语义感知的 SDF 预测，但仍采用局部融合策略。3DFG-PIFu 示了全局融合比语义增强更为关键
- **vs DeepMultiCap**: DeepMultiCap 使用注意力机制聚合视图特征，但作用范围仍局限于查询点邻域。3DFG-PIFu 的3D网格天然覆盖全局空间

## 评分

- 新颖性: ⭐⭐⭐⭐ 3D特征网格的全局融合思路简洁有效，SDF-based SMPL-X 特征也是有意义的创新
- 实验充分度: ⭐⭐⭐ 在标准数据集上有消融实验和对比实验，但缺少in-the-wild评估
- 写作质量: ⭐⭐⭐ 问题定义清晰，方法描述完整
- 价值: ⭐⭐⭐⭐ 全局融合思路有较强的通用性和启发性，对多视图重建领域有推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] 3DGazeNet: Generalizing 3D Gaze Estimation with Weak-Supervision from Synthetic Views](3dgazenet_generalizing_3d_gaze_estimation_with_weak-supervision_from_synthetic_v.md)
- [\[ECCV 2024\] 3DSA: Multi-view 3D Human Pose Estimation With 3D Space Attention Mechanisms](3dsa_multi-view_3d_human_pose_estimation_with_3d_space_attention_mechanisms.md)
- [\[ECCV 2024\] TELA: Text to Layer-wise 3D Clothed Human Generation](tela_text_to_layer-wise_3d_clothed_human_generation.md)
- [\[ECCV 2024\] Wear-Any-Way: Manipulable Virtual Try-on via Sparse Correspondence Alignment](wear-any-way_manipulable_virtual_try-on_via_sparse_correspondence_alignment.md)
- [\[ECCV 2024\] ADen: Adaptive Density Representations for Sparse-view Camera Pose Estimation](aden_adaptive_density_representations_for_sparseview_camera.md)

</div>

<!-- RELATED:END -->
