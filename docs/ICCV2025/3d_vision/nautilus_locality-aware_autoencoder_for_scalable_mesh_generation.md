---
title: >-
  [论文解读] Nautilus: Locality-aware Autoencoder for Scalable Mesh Generation
description: >-
  [ICCV2025][3D视觉][mesh generation] Nautilus 提出一种局部性感知的自编码器进行可扩展的 artist-like 网格生成，通过 Nautilus 式壳结构网格分词算法将序列长度压缩到 1/4，并结合双流点云条件器提高局部结构保真度，首次实现最多 5000 面的高质量网格直接生成。
tags:
  - "ICCV2025"
  - "3D视觉"
  - "mesh generation"
  - "autoregressive"
  - "tokenization"
  - "locality-aware"
  - "点云"
---

# Nautilus: Locality-aware Autoencoder for Scalable Mesh Generation

**会议**: ICCV2025  
**arXiv**: [2501.14317](https://arxiv.org/abs/2501.14317)  
**代码**: -  
**领域**: 3D视觉  
**关键词**: mesh generation, autoregressive, tokenization, locality-aware, point cloud conditioning  

## 一句话总结

Nautilus 提出一种局部性感知的自编码器进行可扩展的 artist-like 网格生成，通过 Nautilus 式壳结构网格分词算法将序列长度压缩到 1/4，并结合双流点云条件器提高局部结构保真度，首次实现最多 5000 面的高质量网格直接生成。

## 研究背景与动机

- **问题定义**：自动生成高质量三角网格模型，使其具备 artist-like 的简洁、结构化和拓扑正确性
- **现有方法局限**：
    - **中间表示方法**（NeRF、3DGS、SDF 等）：通过 marching cubes 或 Poisson 重建转换为网格，产出过密、次优的网格，缺乏连续表面质量
    - **直接网格生成**（MeshGPT、MeshAnything）：自回归地建模顶点和面，但面临两个根本问题：
    1. **局部结构保真度差**：出现流形缺陷（表面孔洞、重叠面、缺失部件），特别是复杂拓扑时
    2. **面数量受限**：序列过长限制了面数上限，难以捕捉精细拓扑细节
- **核心洞察**：流形网格的局部性（相邻面共享边，每个面簇收敛于一个中心顶点）提供两个关键启发：
  1. 优先建模邻居的局部依赖可确保面精确互连，实现流形性
  2. 显式建模局部共享的边和顶点可显著压缩序列长度

## 方法详解

### 整体框架

1. **Nautilus 式网格分词**：将 artist 网格序列化为紧凑的 token 序列
2. **双流点云条件器**：提供全局一致性和局部结构保真的几何引导
3. **自回归序列解码器**：基于 Transformer 解码，支持点云或单视图图像条件

### 关键设计一：Nautilus 式网格分词

**朴素方法的问题**：将每个面的 3 个顶点展开为 9N 个坐标，无法保持空间邻近性，序列过长。

**Nautilus 壳结构表示**：将网格面组织为多个壳（shell），每个壳围绕一个中心顶点 $O$，包含有序的周围顶点序列 $P$。每个面由 $O$ 和两个相邻 $P$ 构成。这样 $N$ 个面仅需 $N+2$ 个顶点表示，而非 $3N$：

$$S(\mathbf{f}_{OP_1P_2}, \mathbf{f}_{OP_2P_3}, \ldots) = \{O, P_1, P_2, P_3, \ldots\}$$

添加新面只需扩展一个顶点，极大压缩序列。

**连续壳遍历**：完成一个壳后，选择最后遍历顶点的邻居中度数最高的顶点作为下一个壳的中心。这确保了壳间的连续性和空间临近性。

**坐标压缩**：将 3D 坐标 $(x, y, z)$ 映射到 2D 空间 $(u, v)$：

$$x \cdot \alpha^2 + y \cdot \alpha + z = u \cdot \beta + v$$

分辨率 128、乘数 2048，构建 $u$（1024）和 $v$（2048）的编码表。中心顶点 $O$ 的 $u$ 使用独立的 1024 大小编码表以区分壳的开始，避免引入特殊分隔 token。

**压缩效果**：序列长度仅为朴素方法的 **1/4**（压缩比 0.275 vs AMT 的 0.462 和 EdgeRunner 的 0.474）。

### 关键设计二：双流点云条件器

**全局点云编码器**：使用 Michelangelo 编码器提取全局特征 $f_{glb}$，作为解码器交叉注意力层的 key 和 value，提供整体形状信息。

**局部点云编码器**：使用 PointConv 模块 $f_{loc}(\cdot)$ 捕捉局部几何信息。对每个壳的中心顶点 $O_k$ 做 KNN 采样 100 个最近点，提取局部特征 $f_{loc}(O_k)$，注入壳内每个顶点 $P_{k,i}$ 的 token 特征中（受 ControlAR 启发）。

**与壳结构的紧密配合**：局部编码器的特征注入与壳的生成逐步同步，实现渐进式局部几何约束。

### 自回归解码与训练

**生成**：基于 next-token prediction 范式：

$$p(\mathbf{M}) = \prod_{i=1}^{L} p(c_i | c_{<i}), \quad c_i \in \{0, 1, \ldots, \alpha - 1\}$$

**训练损失**：交叉熵损失

$$L_{CE} = \text{CrossEntropy}(\hat{S}, S(\mathbf{M})_{>0})$$

**图像条件**：利用 Michelangelo 的多模态对齐特征空间，训练时用点云编码器（冻结），推理时替换为图像编码器实现单图条件生成。

## 实验关键数据

### 主实验：定量比较

| 方法 | C.Dist. ↓ | H.Dist. ↓ | 用户满意度 ↑ |
|------|-----------|-----------|-------------|
| MeshAnything | 0.133 | 0.293 | 10.27% |
| MeshAnythingV2 | 0.106 | 0.248 | 13.17% |
| **Nautilus** | **0.087** | **0.176** | **88.68%** |

### 分词算法对比

| 指标 | AMT | EdgeRunner | **Nautilus** |
|------|-----|------------|-------------|
| 压缩比 ↓ | 0.462 | 0.474 | **0.275** |
| 局部比 ↑ | 0.378 | 0.461 | **0.554** |

### 关键发现

1. **压缩比领先**：Nautilus 分词压缩比 0.275，远优于 AMT（0.462）和 EdgeRunner（0.474），使 5000 面生成成为可能
2. **局部性保持**：局部比 0.554，说明序列中相邻 token 在网格中也大概率相邻，有利于自回归模型学习局部依赖
3. **用户偏好压倒性**：88.68% 的用户满意度远超 MeshAnything（10.27%）和 V2（13.17%）
4. 仅用坐标压缩或仅用壳结构都不足以达到最佳效果，两者结合才能实现质量和可扩展性
5. 局部点云编码器在复杂拓扑区域（如局部孔洞）提供关键改善

## 亮点与洞察

1. **核心洞察精准**：精确定位了当前直接网格生成方法的根本瓶颈——忽视流形网格的局部性
2. **壳结构分词的优雅设计**：模仿 artist 创建网格的方式（围绕中心顶点扇形扩展），既保持局部性又高效压缩
3. **双流条件机制**：全局保整体形状、局部保拓扑细节，与壳结构的生成范式完美协同
4. **首次实现 5000 面直接生成**：相比 MeshGPT 的 800 面和 MeshAnything 的 1600 面是巨大飞跃
5. **新评价指标 Local Ratio**：量化序列化后局部依赖的保持程度，对分词算法评估有普适价值

## 局限性

1. 推理速度仍然较慢：5000 面生成需要 4 分钟
2. 对下游 image-to-3D 方法生成的测试样本依赖较大，与真实 artist 网格的分布可能不一致
3. 分词算法对非流形网格的处理能力未讨论
4. 训练需要 311K 高质量 artist 网格数据，数据获取成本高
5. 壳结构遍历的贪心策略（选最高度数邻居）可能不是全局最优

## 相关工作与启发

- **网格分词的演进**：MeshGPT（VQ-VAE）→ MeshAnythingV2（AMT）→ EdgeRunner（half-edge）→ **Nautilus（壳结构）**，趋势是从通用分词走向利用网格几何结构的专用分词
- **局部性在序列建模中的重要性**：与 NLP 中位置编码的作用类似，3D 网格分词中的空间局部性保持对建模质量至关重要
- **条件机制的分层设计**：全局+局部双流条件的思路可推广到其他需要多尺度控制的生成任务
- **压缩与质量的平衡**：更高的压缩比使模型能训练在更复杂的样本上，是提升上限的关键

## 评分 ⭐⭐⭐⭐⭐

定位精准（局部性是关键瓶颈），方案优雅（壳结构分词+双流条件），效果显著（压缩 4x、用户满意度 88.68%、首次 5000 面生成）。新指标 Local Ratio 有普适价值。论文完成度高，是直接网格生成领域的重要推进。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] MEGA: Masked Generative Autoencoder for Human Mesh Recovery](../../CVPR2025/3d_vision/mega_masked_generative_autoencoder_for_human_mesh_recovery.md)
- [\[ICCV 2025\] MeshAnything V2: Artist-Created Mesh Generation with Adjacent Mesh Tokenization](meshanything_v2_artist-created_mesh_generation_with_adjacent_mesh_tokenization.md)
- [\[ICCV 2025\] MeshMamba: State Space Models for Articulated 3D Mesh Generation and Reconstruction](meshmamba_state_space_models_for_articulated_3d_mesh_generation_and_reconstructi.md)
- [\[ICCV 2025\] DAP-MAE: Domain-Adaptive Point Cloud Masked Autoencoder for Effective Cross-Domain Learning](dap-mae_domain-adaptive_point_cloud_masked_autoencoder_for_effective_cross-domai.md)
- [\[ICCV 2025\] MeshPad: Interactive Sketch-Conditioned Artist-Reminiscent Mesh Generation and Editing](meshpad_interactive_sketch-conditioned_artist-reminiscent_mesh_generation_and_ed.md)

</div>

<!-- RELATED:END -->
