---
title: >-
  [论文解读] Bi-directional Contextual Attention for 3D Dense Captioning
description: >-
  [ECCV2024 (Oral)][3D视觉][3D Dense Captioning] 提出 BiCA，通过双向上下文注意力机制将 instance query 和 context query 解耦并行解码，解决了 3D 密集描述中定位与描述生成之间的目标冲突，在 ScanRefer 和 Nr3D 两个基准上取得 SOTA。
tags:
  - "ECCV2024 (Oral)"
  - "3D视觉"
  - "3D Dense Captioning"
  - "Transformer"
  - "注意力机制"
  - "点云"
  - "场景理解"
---

# Bi-directional Contextual Attention for 3D Dense Captioning

**会议**: ECCV2024 (Oral)  
**arXiv**: [2408.06662](https://arxiv.org/abs/2408.06662)  
**作者**: Minjung Kim, Hyung Suk Lim, Soonyoung Lee, Bumsoo Kim, Gunhee Kim  
**机构**: Princeton University, LG AI Research, Seoul National University  
**代码**: 待确认  
**领域**: 3D视觉  
**关键词**: 3D Dense Captioning, Transformer, Contextual Attention, Point Cloud, Scene Understanding

## 一句话总结

提出 BiCA，通过双向上下文注意力机制将 instance query 和 context query 解耦并行解码，解决了 3D 密集描述中定位与描述生成之间的目标冲突，在 ScanRefer 和 Nr3D 两个基准上取得 SOTA。

## 背景与动机

3D 密集描述（3D Dense Captioning）要求在 3D 场景中定位所有物体并为每个物体生成自然语言描述。现有方法在构建上下文信息时存在两个关键限制：

1. **上下文范围受限**：已有方法仅通过物体对关系建模或聚合最近邻特征来获取上下文，但物体间的位置关系遍布整个全局场景，不仅限于物体自身附近
2. **目标冲突**：定位任务需要紧凑的局部特征以精确框定物体边界，而描述生成（尤其涉及全局位置关系的描述）需要全局场景的上下文特征。使用单一 query 集同时服务两个任务会导致相互干扰

例如描述"椅子位于房间最西北的位置"需要全局空间理解，但精确定位该椅子需要紧凑的局部特征——这两者在单一表征中难以兼顾。

## 核心问题

如何设计一种架构，在不损害定位性能的前提下，有效聚合全局场景的相关上下文特征，同时提升 3D 密集描述的定位和描述生成性能？

## 方法详解

### 整体架构

BiCA 采用 Transformer encoder-decoder 架构，核心思想是将 instance query（物体）和 context query（非物体上下文）解耦为两个并行流，再通过双向注意力进行信息交互。

### 1. Encoder

沿用 3DETR 的场景编码器。输入点云经 PointNet++ 的 set-abstraction 层分词后，送入含 set-abstraction 的 masked transformer encoder 加两层额外编码层，输出场景 token $p_{enc} \in \mathbb{R}^{1024 \times 3}$，$f_{enc} \in \mathbb{R}^{1024 \times 256}$。

### 2. 双路 Query 生成器

- **Instance Query Generator**：通过 FFN 学习投票偏移量将编码点移向物体中心，然后用半径 0.3 的 set-abstraction 提取 256 个 instance query $(p_o, f_o)$。与 Vote2Cap-DETR 不同，本方法在投票后的候选坐标上提取特征，避免多个 query 聚焦同一物体
- **Context Query Generator**：在编码 token 上用最远点采样（FPS）选取 512 个种子点，用半径 1.2 的 set-abstraction 提取 context query $(p_c, f_c)$。大半径设计使每个 context query 捕获较大范围的几何结构信息，编码物体间和物体与场景的空间关系

### 3. 并行解码器

Instance Decoder 和 Context Decoder 各含 8 层 transformer decoder，使用 Fourier 位置编码对 XYZ 坐标进行编码。两路解码器独立工作：Instance Decoder 聚焦物体检测与属性特征，Context Decoder 捕获非物体区域的结构上下文。

### 4. 双向上下文注意力（Bi-directional Contextual Attention）

这是本文核心贡献，分两个阶段：

**O4C（Objects for Context）**：为每个物体构建 Object-aware Context $V_{ac}$。计算各 instance query 与所有 context query 之间的注意力权重，对 context 特征做加权求和。直觉上，这是在全局 context 中找到与当前物体相关的几何区域信息，通过可学习参数 $\gamma$ 调节。

**C4O（Contexts for Object）**：构建 Context-aware Object $V_{ao}$。用 object-aware context 特征与 instance query 计算注意力，对 instance 特征做加权求和。这一步将"旁边"这种模糊关系具体化为"旁边的红色椅子"，通过可学习参数 $\lambda$ 调节。

最终将 $(V_o, V_{ac}, V_{ao})$ 拼接为 $V_a$，送入描述生成头。

### 5. 定位与描述生成

- **定位**：使用解码后的 instance query $V_o$ 经 5 个 MLP 头回归物体中心偏移和 bounding box 尺寸
- **描述生成**：基于 GPT-2 的 transformer decoder 描述头，包含 2 层 decoder block，用 $V_a$ 替代标准 SOS token 作为前缀，推理时使用 beam search（beam size=5）

### 6. 训练策略

三阶段训练：
1. 在 ScanNet 上预训练检测器（无描述头），1080 epochs
2. 在 ScanRefer/Nr3D 上联合训练，使用交叉熵损失（MLE），720 epochs
3. 使用 Self-Critical Sequence Training（SCST）微调描述头，180 epochs

损失函数为 $\mathcal{L} = \beta_1 \mathcal{L}_o + \beta_2 \sum \mathcal{L}_{det}^i + \beta_3 \mathcal{L}_{cap}$，其中 $\beta_1=10, \beta_2=1, \beta_3=5$。

## 实验关键数据

### ScanRefer 主实验（SCST，无额外 2D 数据）

| 方法 | C@0.25 | C@0.5 | B-4@0.5 | M@0.5 | R@0.5 |
|------|--------|-------|---------|-------|-------|
| Vote2Cap-DETR | 84.15 | 73.77 | 38.21 | 26.64 | 54.71 |
| Vote2Cap-DETR++ | 88.28 | 78.16 | 39.72 | 26.94 | 55.52 |
| **BiCA** | **89.72** | **80.14** | **40.16** | **27.76** | **56.10** |

BiCA 在 CIDEr@0.5 上超过 Vote2Cap-DETR++ **+1.98**，超过 Vote2Cap-DETR **+6.37**。

### Nr3D 主实验（SCST，IoU=0.5）

| 方法 | C | B-4 | M | R |
|------|---|-----|---|---|
| Vote2Cap-DETR++ | 47.62 | 28.41 | 25.63 | 54.77 |
| **BiCA** | **49.81** | **28.83** | **25.85** | **56.46** |

### 消融实验（ScanRefer，SCST，IoU=0.5）

| 配置 | CIDEr | mAP | AR |
|------|-------|-----|-----|
| Vote2Cap-DETR | 73.77 | 45.56 | 67.77 |
| BiCA (仅 $V_o$) | 74.90 | 50.12 | 69.49 |
| BiCA ($V_o$ + KNN($V_c$)) | 79.03 | 55.95 | 69.62 |
| BiCA ($V_o$ + $V_{ac}$) | 81.22 | 56.91 | 70.38 |
| **BiCA ($V_o$ + $V_{ac}$ + $V_{ao}$)** | **85.14** | **57.58** | **72.68** |

每个组件均带来正向增益。完整的 O4C+C4O 比仅用 KNN 上下文高 **+6.11** CIDEr。

### 模型效率

- 参数量：16.9M
- 推理时间：1.8ms/场景（单张 Titan RTX）

## 亮点

1. **解耦设计精巧**：将 instance query 和 context query 分离，从根本上解决了定位与描述生成的目标冲突——定位靠 instance query 保持精度，描述靠融合上下文特征提升质量
2. **双向注意力有效**：O4C 和 C4O 两阶段设计不仅捕获全局几何上下文，还将上下文与具体物体关联，使描述从"旁边"提升为"旁边的红色椅子"
3. **Instance Query Generator 改进**：在投票后坐标上提取特征（而非 FPS 后投票），匹配候选数从 1498 提升至 1540，直接改善检测性能
4. **Context Query 设计**：用大半径（1.2 vs instance 的 0.3）的 set-abstraction 从非物体区域提取结构信息，有效编码场景的全局空间关系
5. **Oral 论文**，在两个基准上全指标 SOTA

## 局限与展望

1. **仅限室内场景**：在 ScanNet 数据集上训练和评估，泛化到大规模户外 3D 场景的能力未知
2. **对点云质量依赖**：稀疏或有噪声的点云可能影响 context query 的质量
3. **Context Query 数量固定**：512 个 context query 和半径 1.2 通过实验确定，不同场景规模可能需要自适应调整
4. **描述头较简单**：仅用 2 层 GPT-2 decoder，引入更强的语言模型或多模态预训练可能进一步提升描述质量
5. **缺少与 LLM 结合的探索**：当前描述生成基于较小的语言模型，与大型语言模型的结合值得探索

## 与相关工作的对比

| 方法 | 上下文范围 | Query 设计 | 定位与描述解耦 | ECCV/CVPR |
|------|-----------|-----------|------------|-----------|
| Scan2Cap | 物体对关系 | 无（两阶段） | ✗ | CVPR 2021 |
| 3DJCG | 物体对 + 图注意力 | 统一 query | ✗ | CVPR 2022 |
| Vote2Cap-DETR | 最近邻 | FPS + 投票 | ✗ | CVPR 2023 |
| Vote2Cap-DETR++ | 最近邻 | 解耦定位/描述 query | 部分 | TPAMI 2024 |
| **BiCA** | **全局场景** | **Instance + Context 双路** | **✓** | **ECCV 2024** |

Vote2Cap-DETR++ 虽然解耦了定位和描述 query，但其解耦 query 仍是物体中心 query 的投影，受限于物体中心设计。BiCA 从结构上实现了物体特征和非物体上下文的真正分离。

## 启发与关联

1. **Query 解耦思想可迁移**：将 query 集按功能分为目标检测和上下文理解两路并行解码，这一思路可应用于其他需要同时处理定位和理解的任务（如 3D Visual Grounding、Open-vocabulary 3D Detection）
2. **双向注意力范式**：O4C → C4O 的两阶段信息流动模式可推广到其他需要局部-全局特征交互的场景
3. **非物体区域的重要性**：显式建模非物体区域的空间结构信息对理解场景关系有帮助，这一观察对 3D 场景理解的其他任务也有参考价值

## 评分
- 新颖性: ⭐⭐⭐⭐ — 双向上下文注意力设计新颖且动机清晰
- 实验充分度: ⭐⭐⭐⭐ — 两个基准、多种设置、详细消融
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，图示直观
- 价值: ⭐⭐⭐⭐ — Oral 论文，SOTA，解耦思想有迁移价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Curvature-Aware Captioning: Leveraging Geodesic Attention for 3D Scene Understanding](../../CVPR2026/3d_vision/curvature-aware_captioning_leveraging_geodesic_attention_for_3d_scene_understand.md)
- [\[ECCV 2024\] View Selection for 3D Captioning via Diffusion Ranking](view_selection_for_3d_captioning_via_diffusion_ranking.md)
- [\[ECCV 2024\] TPA3D: Triplane Attention for Fast Text-to-3D Generation](tpa3d_triplane_attention_for_fast_text-to-3d_generation.md)
- [\[ECCV 2024\] ScatterFormer: Efficient Voxel Transformer with Scattered Linear Attention](scatterformer_efficient_voxel_transformer_with_scattered_linear_attention.md)
- [\[ECCV 2024\] FALIP: Visual Prompt as Foveal Attention Boosts CLIP Zero-Shot Performance](falip_visual_prompt_as_foveal_attention_boosts_clip_zero-shot_performance.md)

</div>

<!-- RELATED:END -->
