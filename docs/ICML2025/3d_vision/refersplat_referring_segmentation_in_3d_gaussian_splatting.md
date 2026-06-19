---
title: >-
  [论文解读] ReferSplat: Referring Segmentation in 3D Gaussian Splatting
description: >-
  [ICML 2025 Oral][3D视觉][3D Gaussian Splatting] ReferSplat 提出了 Referring 3D Gaussian Splatting Segmentation（R3DGS）新任务，通过构建 3D Gaussian Referring Fields、位置感知跨模态交互模块和 Gaussian-Text 对比学习，实现了基于自然语言描述在 3DGS 场景中分割目标物体（包括遮挡/不可见物体），在新构建的 Ref-LERF 数据集和开放词汇分割基准上取得 SOTA。
tags:
  - "ICML 2025 Oral"
  - "3D视觉"
  - "3D Gaussian Splatting"
  - "指称分割"
  - "自然语言"
  - "空间推理"
  - "对比学习"
---

# ReferSplat: Referring Segmentation in 3D Gaussian Splatting

**会议**: ICML 2025 Oral  
**arXiv**: [2508.08252](https://arxiv.org/abs/2508.08252)  
**代码**: [https://github.com/heshuting555/ReferSplat](https://github.com/heshuting555/ReferSplat)  
**领域**: 3D视觉  
**关键词**: 3D Gaussian Splatting, 指称分割, 自然语言, 空间推理, 对比学习

## 一句话总结
ReferSplat 提出了 Referring 3D Gaussian Splatting Segmentation（R3DGS）新任务，通过构建 3D Gaussian Referring Fields、位置感知跨模态交互模块和 Gaussian-Text 对比学习，实现了基于自然语言描述在 3DGS 场景中分割目标物体（包括遮挡/不可见物体），在新构建的 Ref-LERF 数据集和开放词汇分割基准上取得 SOTA。

## 研究背景与动机
3DGS 凭借快速训练、实时渲染和显式点表示迅速成为 3D 场景表示的重要方法。开放词汇 3DGS 分割已有初步进展，但仅依赖固定模式的类名输入。自由形式的自然语言与 3D 场景的交互仍很少被探索，这对具身 AI、自动驾驶、VR/AR 等至关重要。

现有开放词汇方法存在两个关键局限：第一，训练时文本查询与 Gaussian 表示缺乏交互——在 2D 渲染特征与文本做匹配而非 3D 空间直接定位；第二，忽略位置信息——渲染特征无法理解空间关系。核心矛盾：需要同时具备 3D 空间推理和细粒度语言理解。

核心 idea：直接在 3D Gaussian 空间建模语言交互，为每个 Gaussian 赋予 referring 特征使其与文本直接关联，并通过位置感知交叉注意力增强空间推理。

## 方法详解

### 整体框架
ReferSplat 含三个核心组件：(1) 3D Gaussian Referring Fields；(2) 位置感知跨模态交互（PCMI）增强空间推理；(3) Gaussian-Text 对比学习（GTCL）区分语义相似表达。训练使用置信度加权 IoU 策略生成的伪标签。

### 关键设计
1. **3D Gaussian Referring Fields**:

    - 为每个 Gaussian 引入 referring 特征 f_r,i，计算与词特征相似度 m_i = sum_j f_r,i * f_w,j，再光栅化渲染 2D 掩码。
    - 设计动机：不同于在 2D 渲染特征上做检索，直接在 3D 空间建模使模型能通过多视角知识识别遮挡物体。

2. **位置感知跨模态交互（PCMI）**:

    - Gaussian 位置特征：将中心坐标通过 MLP 映射为位置嵌入。
    - 文本位置推断：通过词特征与 Gaussian referring 特征的关系间接获取文本位置。
    - 位置引导注意力精化 referring 特征，融合位置和语义信息。
    - 设计动机：理解含空间关系的表达需要语义识别与空间推理并行。

3. **Gaussian-Text 对比学习（GTCL）**:

    - 选择 top-tau 响应的 Gaussian 特征取平均作为正 Gaussian 嵌入，对比学习拉近对应文本、推开不相关文本。
    - 设计动机：区分语义相似但指向不同物体的描述。

### 损失函数 / 训练策略
- 总损失：BCE损失 + lambda * 对比损失，lambda=0.02
- 伪标签：Grounded SAM + 置信度加权 IoU 策略选择最佳掩码
- 两阶段优化精化；BERT 文本嵌入，45,000 迭代

## 实验关键数据

### 主实验（Ref-LERF 数据集）

| 方法 | ramen | figurines | teatime | kitchen | avg mIoU |
|------|-------|-----------|---------|---------|----------|
| Grounded SAM | 14.1 | 16.0 | 16.9 | 16.2 | 15.8 |
| LangSplat | 12.0 | 17.9 | 7.6 | 17.9 | 13.9 |
| GOI | 27.1 | 16.5 | 22.9 | 15.7 | 20.5 |
| **ReferSplat** | **35.2** | **25.7** | **31.3** | **24.4** | **29.2** |

### 消融实验

| 配置 | ramen | kitchen | 说明 |
|------|-------|---------|------|
| Baseline | 28.4 | 18.5 | 仅 Referring Fields |
| + PCMI | 33.5 | 22.8 | +5.1/+4.3，空间推理增强 |
| + GTCL | 32.8 | 21.9 | +4.4/+3.4，细粒度区分 |
| + PCMI + GTCL | 35.2 | 24.4 | 完整 ReferSplat |
| + Two-stage | 36.9 | 25.2 | 进一步精化 |

### 关键发现
- Referring Fields 直接建模 3D-文本关系大幅优于 VLM 特征空间匹配
- 朴素交叉注意力（无位置信息）反而低于 baseline，位置感知至关重要
- 伪标签质量：置信度加权 IoU 优于 Top-1 和 SAM2 方法
- 在开放词汇分割任务上也达到 SOTA，referring 能力可迁移

## 亮点与洞察
- 定义 R3DGS 新任务并构建 Ref-LERF 数据集
- 从 2D 渲染匹配转向 3D 空间直接建模是概念突破
- 文本位置从 Gaussian 间接推断的双向设计巧妙
- 置信度加权 IoU 伪标签策略简单有效

## 局限与展望
- Ref-LERF 规模较小（4 场景、295 条描述），泛化性待验证
- 伪标签上限约 50% mIoU，制约最终性能
- 复杂自然语言表达（否定、条件等）尚未测试

## 相关工作与启发
- 从 2D RES 到 3D 点云 RES 再到 3DGS RES 的自然演进
- 位置推断思路可推广到其他 3D 表示中的跨模态交互
- 对具身 AI 中语言引导导航和操作有直接启发

## 补充分析

### Ref-LERF 数据集特性
- 平均句子长度超过 13.6 词，约为 LERF-OVS 的 8 倍，强调空间推理和细节描述
- 词云显示大量相对位置词（placed、near、next）和细粒度属性词（round、surface）
- 每个物体约有 5 条描述，236 条训练 + 59 条测试

### 3DOVS 任务上的迁移验证
- 在 LERF-OVS 开放词汇分割基准上也达到 SOTA 表现
- LangSplat 平均 51.4 mIoU vs ReferSplat 更高的平均分数
- 说明 Referring Fields + PCMI + GTCL 的组合对通用 3D 语言理解也有效

### 伪标签质量分析
- 手动标注了 Ground Truth 用于评估伪标签质量
- 置信度加权 IoU 策略达到约 50% mIoU vs GT
- 两阶段优化利用第一阶段渲染掩码作为更好的伪标签进一步提升

## 评分
- 新颖性: ⭐⭐⭐⭐ 新任务定义 + 3D Gaussian 空间建模语言交互的范式创新
- 实验充分度: ⭐⭐⭐⭐ 消融全面但数据集规模小
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机充分
- 价值: ⭐⭐⭐⭐ 推动 3DGS 理解向自然语言交互迈进

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Geometry-Aware Cross-Modal Graph Alignment for Referring Segmentation in 3D Gaussian Splatting](../../CVPR2026/3d_vision/geometry-aware_cross-modal_graph_alignment_for_referring_segmentation_in_3d_gaus.md)
- [\[CVPR 2026\] ST4R-Splat: Spatio-Temporal Referring Segmentation in 4D Gaussian Splatting](../../CVPR2026/3d_vision/st4r-splat_spatio-temporal_referring_segmentation_in_4d_gaussian_splatting.md)
- [\[CVPR 2025\] Dr. Splat: Directly Referring 3D Gaussian Splatting via Direct Language Embedding Registration](../../CVPR2025/3d_vision/dr_splat_directly_referring_3d_gaussian_splatting_via_direct_language_embedding_.md)
- [\[CVPR 2026\] Spatial Matters: Position-Guided 3D Referring Expression Segmentation](../../CVPR2026/3d_vision/spatial_matters_position-guided_3d_referring_expression_segmentation.md)
- [\[ICCV 2025\] CLIP-GS: Unifying Vision-Language Representation with 3D Gaussian Splatting](../../ICCV2025/3d_vision/clip-gs_unifying_vision-language_representation_with_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
