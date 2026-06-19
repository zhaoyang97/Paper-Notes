---
title: >-
  [论文解读] See&Trek: Training-Free Spatial Prompting for Multimodal Large Language Model
description: >-
  [NeurIPS 2025][多模态VLM][空间理解] 提出 See&Trek，一个无需训练和GPU的空间提示框架，通过最大语义丰富度采样和运动重建来增强 MLLM 的空间理解能力，在 VSI-Bench 上最高提升 3.5%。 领域现状：多模态大语言模型（MLLM）在图像理解、VQA 等任务上取得了显著进展…
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "空间理解"
  - "多模态大模型"
  - "视觉提示"
  - "视觉里程计"
  - "训练无关"
---

# See&Trek: Training-Free Spatial Prompting for Multimodal Large Language Model

**会议**: NeurIPS 2025  
**arXiv**: [2509.16087](https://arxiv.org/abs/2509.16087)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 空间理解, 多模态大模型, 视觉提示, 视觉里程计, 训练无关

## 一句话总结
提出 See&Trek，一个无需训练和GPU的空间提示框架，通过最大语义丰富度采样和运动重建来增强 MLLM 的空间理解能力，在 VSI-Bench 上最高提升 3.5%。

## 研究背景与动机

**领域现状**：多模态大语言模型（MLLM）在图像理解、VQA 等任务上取得了显著进展，但空间推理仍是短板，尤其是涉及物体定位、运动预测和物理交互的场景。

**现有痛点**：当前 MLLM 处理视频时普遍采用均匀时间采样策略（如抽 8 或 32 帧），这导致两个根本问题：
   - **视觉同质性**：均匀采样容易选到没有显著特征的帧（如墙壁、天花板），降低了输入帧的信噪比
   - **运动未知**：仅依赖采样帧，没有显式自运动信息，模型无法推断物体运动和位移，只能依赖预训练期间获得的常识先验做推测

**核心矛盾**：空间推理需要丰富的视觉语义和明确的运动信息，但现有 pipeline 两者都缺乏

**切入角度**：利用现成的感知模型（YOLO）和视觉里程计（VO），在**无需训练**的情况下为 MLLM 注入空间线索

**核心 idea**：通过最大语义丰富度采样增加视觉多样性，通过运动重建恢复相机轨迹并编码到关键帧中

## 方法详解

### 整体框架
给定视频序列，See&Trek 分三步：(1) 用 YOLO 检测物体并选择语义最丰富的关键帧（Maximum Semantic Richness Sampling）；(2) 用 ORB 特征和本质矩阵估计相机运动轨迹（Motion Reconstruction）；(3) 将运动信息编码到关键帧上作为时空标记（Spatiotemporal Encoding），最终与文本提示组合输入 MLLM。

### 关键设计

1. **最大语义丰富度采样 (Balanced-TopK)**：

    - 功能：从视频中选出 K 个语义最丰富且时间分布均匀的关键帧
    - 核心思路：先用 YOLO 检测每帧中的物体类别集合 $\mathcal{C}_t$，选出物体类别最多的帧作为初始帧；然后将有效区间分为 K-1 个时间段，在每段中选与已有类别重叠最少、物体最多的帧
    - 设计动机：简单的 TopK 会偏向物体最多的时间段，导致时间局部偏差。Balanced-TopK 通过时间分段和去重策略，兼顾语义丰富性和时间多样性

2. **运动重建**：

    - 功能：从单目视频估计相机运动轨迹
    - 核心思路：对相邻帧提取 ORB 特征并匹配，通过 RANSAC 估计本质矩阵 $\mathbf{E}$，SVD 分解得到相对旋转 $\mathbf{R}_t$ 和平移 $\mathbf{T}_t$，递推累积得全局轨迹：$\mathbf{T}_t^{world} = \mathbf{R}_{t-1}^{world}\mathbf{T}_t + \mathbf{T}_{t-1}^{world}$
    - 设计动机：显式的相机运动信息让 MLLM 能基于证据而非推测来理解空间关系

3. **时空编码**：

    - 功能：将轨迹信息视觉化地编码到关键帧上
    - 核心思路：为每个关键帧分配颜色标记（连续色图，反映时间顺序）和帧索引号，直接叠加在图像右上角，同时生成 BEV 和 3D 轨迹可视化图
    - 设计动机：解决关键帧与运动轨迹的关联问题——MLLM 无法将独立的帧与独立的轨迹图联系起来，时空编码通过视觉标记架起桥梁

### 训练策略
完全无训练，只需 CPU 计算几次 ORB 匹配和 YOLO 推理，单次前向传播即可完成。

## 实验关键数据

### 主实验 — VSI-Bench

| 模型 | 原始 Avg. | +See&Trek | 提升 |
|------|----------|-----------|------|
| InternVL3-1B | 29.5 | 32.0 | +3.5% |
| InternVL3-8B | 40.2 | 43.2 | +3.0% |
| InternVL3-14B | 44.2 | 45.6 | +1.4% |
| Qwen2.5-VL-7B | 27.3 | 29.0 | +2.6% |
| LLaVA-OneVision-7B | 31.4 | 33.0 | +1.6% |
| Kimi-VL-A3B | 33.4 | 35.1 | +1.7% |

### 消融实验 — STI-Bench

| 模型 | 原始 | +See&Trek | 静态理解提升 | 动态理解提升 |
|------|------|-----------|-------------|-------------|
| InternVL3-8B | 35.2 | 36.5 | +0.5% | +3.4% |
| Qwen2.5-VL-7B | 30.3 | 33.2 | +1.8% | +4.3% |

### 关键发现
- See&Trek 对**相对方向 (Rel. Dir.)** 和**接近顺序 (Appr. Order)** 提升最显著，这两个任务最依赖运动信息
- 在**物体计数 (Obj. Count)** 上偶有下降，因为语义采样可能丢失部分同类物体出现的帧
- 小模型（1B/3B）受益更大，说明方法能弥补小模型空间理解能力的不足
- 完全不需要训练和 GPU，即插即用，兼容开源和商业模型

## 亮点与洞察
- **零成本增强**：不修改模型参数、不需要额外训练，仅通过输入增强就能提升空间理解，这是一种非常实用的工程思路
- **Balanced-TopK 的设计巧妙**：通过时间分段+去重，在有限帧预算下最大化信息量，该策略可迁移到视频理解的其他场景
- **运动重建作为"免费特征"**：经典 CV 的 VO 管线为 MLLM 提供了一种结构化的空间先验，相比端到端学习更可靠

## 局限与展望
- VO 基于特征匹配，在纹理贫乏或快速运动场景下会失败
- Balanced-TopK 依赖 YOLO 的检测能力，对 YOLO 不认识的物体类别无能为力
- 方法只适用于视频输入，无法处理单图空间推理
- 对 Object Count 类任务偶有负面影响，需要针对任务类型自适应选择是否启用

## 相关工作与启发
- **vs SpatialRGPT/LLaVA-3D**：这些方法需要深度图或点云等额外模态，而 See&Trek 仅用 RGB 视频
- **vs VideoRAG/VideoTree**：这些方法需要微调 MLLM 或依赖 VLM 做检索，计算开销大；See&Trek 完全无训练
- 可以考虑将 See&Trek 的思路与 depth estimation 模型结合，获得更精确的空间信息

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个零训练空间提示框架，思路简洁有效
- 实验充分度: ⭐⭐⭐⭐ 覆盖 10+ 模型和两个 benchmark，结果全面
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，图表信息量大
- 价值: ⭐⭐⭐⭐ 工程实用性强，即插即用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Training-free Online Video Step Grounding](training-free_online_video_step_grounding.md)
- [\[ICCV 2025\] Exploiting Vision Language Model for Training-Free 3D Point Cloud OOD Detection](../../ICCV2025/multimodal_vlm/exploiting_vision_language_model_for_training-free_3d_point_cloud_ood_detection_.md)
- [\[NeurIPS 2025\] Nautilus: A Large Multimodal Model for Underwater Scene Understanding](nautilus_a_large_multimodal_model_for_underwater_scene_understanding.md)
- [\[ACL 2025\] Can Multimodal Large Language Models Understand Spatial Relations?](../../ACL2025/multimodal_vlm/spatialmqa_mllm_spatial_relations.md)
- [\[CVPR 2026\] ROSE: Rotate Your Large Language Model to See](../../CVPR2026/multimodal_vlm/rose_rotate_your_large_language_model_to_see.md)

</div>

<!-- RELATED:END -->
