---
title: >-
  [论文解读] WIR3D: Visually-Informed and Geometry-Aware 3D Shape Abstraction
description: >-
  [自监督学习] WIR3D 通过优化一组 3D Bézier 曲线参数，在 CLIP 中间层激活的空间引导下，从任意视角忠实表示 3D 形状的几何结构和视觉显著特征（包括纹理），实现稀疏但语义丰富的 3D 形状抽象。 将 3D 形状抽象为一组稀疏的语义有意义的曲线是一个重要但困难的问题。关键挑战在于找到能最佳表示形状视觉特征…
tags:
  - "自监督学习"
---

# WIR3D: Visually-Informed and Geometry-Aware 3D Shape Abstraction

- **会议**: ICCV 2025
- **arXiv**: 2505.04813
- **代码**: 即将发布
- **领域**: 自监督
- **关键词**: 3D形状抽象, Bézier曲线, CLIP引导, 纹理抽象, 形状变形

## 一句话总结

> WIR3D 通过优化一组 3D Bézier 曲线参数，在 CLIP 中间层激活的空间引导下，从任意视角忠实表示 3D 形状的几何结构和视觉显著特征（包括纹理），实现稀疏但语义丰富的 3D 形状抽象。

## 研究背景与动机

将 3D 形状抽象为一组稀疏的语义有意义的曲线是一个重要但困难的问题。关键挑战在于找到能最佳表示形状视觉特征（包括几何和纹理）的稀疏曲线集合。

现有方法的不足：

**遮挡轮廓（Occluding Contours）**：经典的非真实感渲染方法，完全依赖表面几何分析，只能提取低级几何轮廓，无法捕捉高级语义概念和纹理。且是 2D 表示，视角不一致，密集视角渲染时产生闪烁伪影

**反投影方案的局限**：将多视角 2D 轮廓反投影到 3D 会产生密集且不美观的线条簇（如简单圆柱体），且完全无法处理纹理

**3Doodle 等基于 3D 笔画的方法**：主要基于全局 CLIP 监督，对细节（面部特征、纹理模式）不敏感，缺乏空间引导和几何约束

作者指出，真正的 3D 形状抽象需要同时捕捉：视觉显著的几何结构、高级纹理概念（如龙鳞、西瓜籽）以及关键特征（如面部特征），且需要多视角一致性。

## 方法详解

### 整体框架

WIR3D 采用两阶段优化：

- **Stage I（几何抽象）**：优化一组曲线表示形状的整体几何结构
- **Stage II（纹理抽象）**：冻结几何曲线，新增曲线表示纹理和视觉细节

不同阶段使用不同的 CLIP 架构：RN101 对几何结构敏感，RN50x64 对高级视觉概念敏感。

### 曲线表示

3D 笔画建模为三阶 Bézier 曲线集合 $\{B_i\}_{i=1}^n$，每条曲线由 4 个控制点定义：

$$B(t) = (1-t)^3 p^0 + (1-t)^2 t p^1 + (1-t)t^2 p^2 + t^3 p^3$$

通过透视投影 3D 控制点到 2D，使用 DiffVG 进行可微分光栅化。

### 语义损失

基于 CLIPasso 的思路，比较渲染笔画与目标形状的 CLIP 编码：

$$\mathcal{L}_{\text{semantic}} = \lambda_{\text{fc}} \text{dist}(\text{CLIP}(I_{\text{curve}}), \text{CLIP}(I_{\text{target}})) + \sum_{l=3,4} \|\text{CLIP}_l(I_{\text{curve}}) - \text{CLIP}_l(I_{\text{target}})\|_2^2$$

其中 $\text{CLIP}_l$ 是第 $l$ 层中间激活，$\text{dist}$ 为余弦距离。

### 局部关键点损失（核心创新）

为了捕捉细粒度特征，引入基于 3D 关键点的空间加权框架。关键点可由用户指定或通过 Backto3D 特征反投影 + KMeans 自动检测。

为每个视角构建权重图，使用从关键点中心的高斯衰减：

$$I_{\text{weight}}(x,y) = 1 + \sum_p e^{\frac{-\|(x,y) - p\|^2}{2\sigma^2}}$$

加 1 确保远离关键点的区域仍有贡献。维护 z-buffer 处理关键点遮挡。

最终局部关键点损失：

$$\mathcal{L}_{\text{local}} = \lambda_{\text{fc}} \bar{I}_{\text{weight}} \text{dist}(\text{CLIP}(I_{\text{curve}}), \text{CLIP}(I_{\text{target}})) + \sum_{l=3,4} \|I_{\text{weight}} \cdot (\text{CLIP}_l(I_{\text{curve}}) - \text{CLIP}_l(I_{\text{target}}))\|_2^2 + \lambda_{\text{lpips}} \text{LPIPS}(I_{\text{curve}}, I_{\text{target}})$$

### SDF 正则化

使用神经 SDF（MLP 拟合形状的有符号距离场）约束曲线贴近目标表面：

$$\mathcal{L}_{\text{SDF}} = \frac{1}{n \cdot k}\sum_{i=1}^{n}\sum_{k=1}^{s} |\phi(B_i(t_k))|$$

### 视图正则化

确保所有曲线在所有采样视角都可见：

$$\mathcal{L}_{\text{ndc}} = \sum_{i=1}^{n}\sum_t \text{ReLU}(\mathcal{P}(B_i(t)) - 1) + \text{ReLU}(-\mathcal{P}(B_i(t)))$$

### 两阶段优化

**Stage I**（几何）：使用 Freestyle 渲染的无纹理目标，损失为 $\mathcal{L}_I = \mathcal{L}_{\text{semantic}} + 0.1 \cdot \mathcal{L}_{\text{SDF}} + \mathcal{L}_{\text{ndc}}$

**Stage II**（纹理）：冻结 Stage I 曲线，新增曲线使用带纹理渲染的目标，损失为 $\mathcal{L}_{II} = \mathcal{L}_{\text{local}} + \mathcal{L}_{\text{SDF}} + \mathcal{L}_{\text{ndc}}$

## 实验

### 与基线方法的定量比较

| 方法 | LPIPS ↓ | CLIP$^{\text{img}}$ ↑ | User Rank ↑ | Coverage ↓ |
|------|---------|----------------------|-------------|------------|
| NEF | 0.313 | 0.86 | - | 0.056 |
| 3Doodle | 0.246 | 0.900 | 0.12 | 0.020 |
| **WIR3D** | **0.227** | **0.909** | **0.88** | **0.008** |

用户研究（$N=96$）中 WIR3D 被选为更好抽象的比例为 **88%**。Coverage（曲线到表面的单向 Chamfer 距离）比 3Doodle 降低 **2x 以上**。

### 消融实验

| 配置 | LPIPS ↓ | CLIP$^{\text{img}}$ ↑ | Coverage ↓ |
|------|---------|----------------------|------------|
| **WIR3D (Full)** | **0.227** | **0.909** | **0.008** |
| w/o SDF | 0.229 | 0.904 | 0.012 |
| w/o Local Keypoint | 0.233 | 0.905 | 0.009 |
| w/o Stage 1 | 0.248 | 0.900 | 0.016 |
| w/o CLIP Layers | 0.294 | 0.891 | 0.012 |

### 关键发现

1. **CLIP 中间层最关键**：移除中间层激活导致质量下降最大（LPIPS 从 0.227 到 0.294），说明空间化的语义特征比全局 CLIP 编码更具信息量
2. **两阶段设计必要**：去掉 Stage I 导致优化偏向特定视角，产生"扁平化"效果
3. **关键点鲁棒性**：随机关键点的结果类似于不使用关键点损失（退化为 Eq. 2），有意义的关键点则严格提升结果
4. **抽象级别控制**：曲线数量自然控制抽象程度——更多曲线捕捉更多细节
5. **野外重建鲁棒**：对有大量几何缺陷的照片重建模型也能成功抽象

### 应用

- **交互式特征控制**：用户通过选择关键点逐步添加细节（如飞机车轮、Nefertiti 头带）
- **形状变形**：曲线作为直觉的变形手柄，基于欧氏距离蒙皮权重将变形传递到表面。用户研究（$N=42$）中 WIR3D 变形被认为更理想的比例为 **80%**

## 亮点与洞察

- 利用不同 CLIP 架构对几何 vs. 语义概念的差异敏感性是巧妙的设计选择
- 关键点 → 空间权重图 → 加权 CLIP 中间层损失的流程，是将 2D 视觉基础模型知识精确引导到 3D 的优雅方案
- SDF 正则化不仅保证几何保真，还意外赋予了曲线作为变形手柄的能力
- 方法对输入网格质量和拓扑无要求，实用性强

## 局限性

- 质量依赖关键点质量，Random Keypoints 时效果退化到无关键点水平
- 预处理耗时长：神经 SDF 拟合 + 自动关键点检测 + Freestyle 渲染对复杂模型可达 2 小时
- 默认每阶段 20 条曲线的设定可能对几何极复杂的形状不足
- 优化过程纯基于 CLIP 渲染对比，无法保证对所有纹理类型的可靠抽象

## 相关工作

- **形状分解**: Tulsiani（长方体）、CvxNet（凸多面体）——重在重建而非抽象
- **非真实感渲染**: 遮挡轮廓（视角不一致）、NEF/隐式边缘场（关注几何边界，无法抽象）
- **曲线抽象**: CLIPasso（2D草图抽象）、3Doodle（3D曲线但全局监督，对细节不敏感）

## 评分

| 维度 | 分数 |
|------|------|
| 创新性 | ⭐⭐⭐⭐ |
| 有效性 | ⭐⭐⭐⭐ |
| 清晰度 | ⭐⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐ |
| 总评 | 8.0/10 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Teaching DINOv3 About Partial 3D Geometry: A Self-Supervised Geometry-Aware Approach](../../CVPR2026/self_supervised/teaching_dinov3_about_partial_3d_geometry_a_self-supervised_geometry-aware_appro.md)
- [\[ICCV 2025\] Manual-PA: Learning 3D Part Assembly from Instruction Diagrams](manual-pa_learning_3d_part_assembly_from_instruction_diagrams.md)
- [\[CVPR 2025\] Escaping Plato's Cave: Towards the Alignment of 3D and Text Latent Spaces](../../CVPR2025/self_supervised/escaping_platos_cave_towards_the_alignment_of_3d_and_text_latent_spaces.md)
- [\[NeurIPS 2025\] SegMASt3R: Geometry Grounded Segment Matching](../../NeurIPS2025/self_supervised/segmast3r_geometry_grounded_segment_matching.md)
- [\[NeurIPS 2025\] STaRFormer: Semi-Supervised Task-Informed Representation Learning via Dynamic Attention-Based Regional Masking](../../NeurIPS2025/self_supervised/starformer_semi-supervised_task-informed_representation_learning_via_dynamic_att.md)

</div>

<!-- RELATED:END -->
