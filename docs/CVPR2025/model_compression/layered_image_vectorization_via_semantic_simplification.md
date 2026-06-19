---
title: >-
  [论文解读] Layered Image Vectorization via Semantic Simplification
description: >-
  [CVPR 2025][模型压缩][图像矢量化] 本文提出一种渐进式图像矢量化方法，利用 Score Distillation Sampling（SDS）的特征平均效应生成逐级简化的图像序列，以此引导从宏观语义结构到精细细节的分层矢量重建，在视觉保真度、语义对齐和紧凑分层表示上显著优于现有方法。 1. 领域现状：图像矢量化（…
tags:
  - "CVPR 2025"
  - "模型压缩"
  - "图像矢量化"
  - "语义简化"
  - "SDS蒸馏"
  - "分层表示"
  - "SVG生成"
---

# Layered Image Vectorization via Semantic Simplification

**会议**: CVPR 2025  
**arXiv**: [2406.05404](https://arxiv.org/abs/2406.05404)  
**代码**: 无（基于 PyTorch + DiffVG 实现）  
**领域**: 图像生成 / 模型压缩  
**关键词**: 图像矢量化, 语义简化, SDS蒸馏, 分层表示, SVG生成

## 一句话总结
本文提出一种渐进式图像矢量化方法，利用 Score Distillation Sampling（SDS）的特征平均效应生成逐级简化的图像序列，以此引导从宏观语义结构到精细细节的分层矢量重建，在视觉保真度、语义对齐和紧凑分层表示上显著优于现有方法。

## 研究背景与动机

1. **领域现状**：图像矢量化（将栅格图像转换为 SVG 等矢量格式）是计算机图形学的经典问题。近年来基于可微渲染的方法（如 LIVE、DiffVG）通过迭代优化 Bézier 曲线来逼近目标图像，取得了良好效果。

2. **现有痛点**：现有方法以单一目标图像作为优化目标，直接在像素差异最大的区域添加矢量图元。这导致两个问题——(a) 生成的矢量图元过于复杂且缺乏语义结构，难以编辑管理；(b) 无法捕获被遮挡、纹理变化等因素掩盖的隐式语义对象（如被细节打断的完整人体轮廓）。

3. **核心矛盾**：矢量化需要同时满足**视觉保真度**和**结构可管理性**，而直接从细节入手的方法无法建立有意义的语义层次。

4. **本文目标** 如何产生按语义层次组织的紧凑分层矢量表示——从整体轮廓到局部细节逐层构建？

5. **切入角度**：作者发现 SDS 中的"特征平均效应"（feature-average effect）可以用来做图像简化——当 SDS 的条件噪声被消除后，迭代优化会使图像逐渐丢失细节而保留宏观结构。这提供了一种自然的"从详到粗"的简化序列。

6. **核心 idea**：利用 SDS 的特征平均效应生成渐进简化的图像序列，以此作为中间优化目标，引导矢量从宏观语义到精细细节的分层重建。

## 方法详解

### 整体框架
以目标图像为输入，流程分三步：(1) **渐进图像简化**：通过修改 SDS 的 CFG（将条件文本设为空或 CFG scale 设为 0），利用特征平均效应每 20 步迭代生成一级简化图像，形成从原图到粗略轮廓的序列（默认 5 级）；(2) **结构构建（Stage I）**：对简化序列中每张图像做语义分割，提取 mask 并按重叠关系排列为从后到前的层级，为每个 mask 初始化闭合 Bézier 曲线并通过结构损失优化；(3) **视觉精修（Stage II）**：为结构矢量拟合颜色并冻结，在高视觉差异区域添加精修矢量并优化视觉保真度损失。

### 关键设计

1. **SDS-based 渐进图像简化**:

    - 功能：生成一组从详到粗的简化图像序列作为矢量化的中间目标
    - 核心思路：SDS 的更新方向 $(\epsilon_\phi(\mathbf{x}_t, t, y) - \epsilon)$ 中，预训练 DDPM 对输入敏感，预测的噪声存在特征不一致性，导致像素被沿不一致方向更新，产生"特征平均"效果（细节模糊、宏观保留）。为控制简化程度避免形状严重扭曲，将 CFG 中的条件文本设为空字符串 " "（等价于消除条件噪声的引导），这样 SDS 仅依靠无条件噪声预测来更新图像。每隔 $N=20$ 步保存一级简化结果，得到 5 级简化序列。与双边/高斯滤波、超像素等传统简化方法相比，SDS 方法能智能地移除非结构性元素（如房子前的树木），保持清晰的语义边界（如瓢虫的圆形轮廓）。
    - 设计动机：传统简化产生的模糊边界不适合矢量化；SDS 的简化天然产生语义层次且边界平滑，与矢量图形兼容性好。

2. **分层结构构建（带层级 Mask 排列和结构损失）**:

    - 功能：从简化序列中提取语义 mask 并优化为分层矢量结构
    - 核心思路：对每级简化图像做语义分割（SAM），按从最简到最详的顺序依次添加 mask。每个 mask 被放入从后到前的层中，同一层的 mask 不相交。Mask 边界经 Douglas-Peucker 算法简化后初始化为闭合 Bézier 曲线。优化时使用**层级结构损失** $\mathcal{L}_{\text{structure}} = w_1 \mathcal{L}_{\text{mse}} + w_2 \mathcal{L}_{\text{overlap}}$：MSE 项衡量每层 mask 图像与矢量渲染图的差异，overlap 项惩罚同层矢量的重叠（对重叠区域的透明度超过阈值的像素施加 ReLU 惩罚）。优化时每对 mask-矢量使用相同随机颜色，仅关注形状对齐。
    - 设计动机：该方法能发现单张图像分割无法捕获的隐式语义结构——如简化后的"机器人整体"或"无孔洞的脸部"，这些在原图中被纹理/遮挡打断。

3. **视觉精修（Color Fitting + Visual-wise 矢量优化）**:

    - 功能：在保持结构矢量不变的情况下添加精修矢量提升视觉保真度
    - 核心思路：首先为结构矢量拟合颜色——取该矢量覆盖的可见像素中的主色调，或通过 MSE 最小化拟合。然后冻结结构矢量，计算渲染图与目标图像的像素差异，在 Top-K 最大差异连通区域初始化精修矢量（类似 LIVE 的策略），优化视觉保真度损失 $\mathcal{L}_{\text{fidelity}} = \|I_{\text{target}} - I_{\text{vector}}\|_2^2$。优化过程中定期执行矢量清理（合并冗余、删除无用矢量）。
    - 设计动机：两阶段分离确保结构完整性不被颜色/细节优化破坏，同时精修阶段弥补结构矢量的视觉损失。

### 损失函数 / 训练策略
- Stage I 结构损失：$\mathcal{L}_{\text{structure}} = \mathcal{L}_{\text{mse}} + 10^{-8} \cdot \mathcal{L}_{\text{overlap}}$
- Stage II 视觉保真度损失：$\mathcal{L}_{\text{fidelity}} = \|I_{\text{target}} - I_{\text{vector}}\|_2^2$
- Adam 优化器，点坐标学习率 1.0，颜色学习率 0.01
- 简化序列 5 级，每级间隔 20 步 SDS 迭代

## 实验关键数据

### 主实验

| 方法 | MSE ↓ | LPIPS ↓ | VeC (%) ↑ | 说明 |
|------|-------|---------|-----------|------|
| DiffVG | - | - | 41.9 | 基础可微渲染 |
| LIVE | - | - | 43.4 | 渐进添加 |
| O&R | - | - | 39.9 | 优化+剪枝 |
| SGLIVE | - | - | 65.9 | 梯度感知分割 |
| **Ours** | **最低** | **最低** | **73.8** | 语义分层 |

VeC（Vector Compactness）衡量矢量图元被语义 mask 高度包含（>85% 面积重叠）的比例。本文方法 VeC 达 73.8%，标准差最小（11.9），显著优于所有基线。100 张测试图的 MSE 和 LPIPS 在所有矢量数量设置下均最优。

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| Full model | 最佳结构 + 视觉质量 | 包含 SDS 简化序列引导 |
| w/o 简化序列 | 缺失隐式语义结构 | 如"Captain America 的整体"和"草地"无法被捕获 |
| Bilateral filter 替代 SDS | 边界模糊，结构退化 | 瓢虫的圆形边界破坏 |
| Gaussian filter 替代 SDS | 边界更模糊 | 丢失语义信息 |
| Superpixel 替代 SDS | 过度碎片化 | 无法恢复宏观结构 |

### 关键发现
- SDS 简化的核心优势是**语义智能性**：能自动移除非结构性遮挡物（如房子前的树木），恢复被遮挡的完整语义对象（如房子的前墙）
- CLIP 语义相似度显示，在矢量数量较少时（粗略阶段），本文方法的语义保真度远优于其他方法
- Florence-2 模型对粗略矢量层生成的描述性文本与原图内容高度吻合，验证了宏观结构的语义有效性
- 分层矢量表示极大方便了下游编辑（如根据底层结构选择上层图元进行重着色）

## 亮点与洞察
- **巧妙利用 SDS 的"缺陷"**：SDS 的特征平均效应通常被视为质量退化问题（导致过于平滑），本文反其道而行将其作为图像简化工具，将 bug 变成 feature
- **两阶段分离优化策略**将形状和颜色解耦，确保结构完整性，同时简化了优化了过程。这种思路可迁移到其他需要层次化生成的任务
- **从简到繁的矢量化策略**类似于人类画画先画轮廓再填充细节的过程，产生的表示更符合人类直觉和编辑需求
- VeC 指标的提出为矢量化质量提供了新的评价维度（语义紧凑度）

## 局限与展望
- SDS 简化依赖预训练扩散模型的先验，对出分布图像（如特殊风格的画作）效果可能不稳定
- 语义分割模型（SAM）的质量直接影响 mask 层级划分的准确性
- 简化序列的级数和间隔（5 级、20 步）是手动设定的超参数，可能对不同复杂度的图像不够自适应
- 处理照片级真实图像时，矢量数量仍然较多（相对于 clipart/emoji 类图像）
- 未探索渐变填充（gradient fill）等更丰富的矢量图元

## 相关工作与启发
- **vs LIVE**: LIVE 在像素差异最大区域渐进添加矢量，但完全基于低级像素分析，无法感知语义结构。本文引入语义简化序列作为中间引导，从宏观语义到细节构建
- **vs O&R (Optimize & Reduce)**: O&R 通过像素聚类初始化并修剪矢量，也缺乏语义层次。本文的分层策略产生更紧凑、更可编辑的结果
- **vs SGLIVE**: SGLIVE 引入梯度感知分割改善矢量布局，VeC 65.9%。本文通过 SDS 简化引导进一步提升至 73.8%，语义对齐质量明显更高
- SDS 用于图像简化的思路可以迁移到视频矢量化中，生成时间一致的简化序列

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ SDS 特征平均效应用于图像简化的洞察极其巧妙，开辟了矢量化的新方向
- 实验充分度: ⭐⭐⭐⭐ 100 张图的定量对比、消融全面，CLIP 语义评估新颖，但缺少大规模用户研究
- 写作质量: ⭐⭐⭐⭐⭐ 插图出色，简化→分层→精修的流程讲解清晰直观
- 价值: ⭐⭐⭐⭐ 对设计领域实用价值高，紧凑分层 SVG 便于编辑和重着色

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Understanding Multi-layered Transmission Matrices](understanding_multi-layered_transmission_matrices.md)
- [\[NeurIPS 2025\] One-Step Diffusion-Based Image Compression with Semantic Distillation](../../NeurIPS2025/model_compression/one-step_diffusion-based_image_compression_with_semantic_distillation.md)
- [\[CVPR 2025\] CoA: Towards Real Image Dehazing via Compression-and-Adaptation](coa_towards_real_image_dehazing_via_compression-and-adaptation.md)
- [\[NeurIPS 2025\] ReplaceMe: Network Simplification via Depth Pruning and Transformer Block Linearization](../../NeurIPS2025/model_compression/replaceme_network_simplification_via_depth_pruning_and_transformer_block_lineari.md)
- [\[CVPR 2025\] Learned Image Compression with Dictionary-based Entropy Model](learned_image_compression_with_dictionary-based_entropy_model.md)

</div>

<!-- RELATED:END -->
