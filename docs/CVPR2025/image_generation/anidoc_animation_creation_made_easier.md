---
title: "AniDoc: Animation Creation Made Easier"
authors: "Yihao Meng, Hao Ouyang, Hanlin Wang, Qiuyu Wang, Wen Wang, Ka Leong Cheng, Zhiheng Liu, Yujun Shen, Huamin Qu"
affiliations: "HKUST, Ant Group, Nanjing University"
venue: "CVPR 2025"
date: 2024-12-18
tags: ["video colorization", "line art", "animation", "correspondence matching", "diffusion model"]
arxiv: "2412.14173"
code: "https://yihao-meng.github.io/AniDoc_demo"
---

# AniDoc: Animation Creation Made Easier

## 研究背景与动机

动画制作（Animation）是一个高度劳动密集型的产业。传统动画流程中，**上色**（colorization）是最耗时的环节之一。一部标准动画电影包含数万帧，每帧的上色工作需要专业画师手工完成，这不仅成本高昂，而且极度依赖人力资源。

近年来，基于深度学习的视频着色方法取得了进展，但现有方法面临以下关键挑战：

**时序一致性**：逐帧着色容易产生颜色闪烁（flickering），相邻帧之间的颜色分配可能不一致

**稀疏参考**：实际生产中，画师通常只提供关键帧（key frames）的彩色参考，中间帧需要自动推断

**线稿质量差异**：手绘线稿（sketch）与数字线稿在清晰度、线条粗细、封闭性等方面差异巨大

**背景干扰**：训练数据中的复杂背景会干扰模型对前景角色的着色学习

本文提出 AniDoc，一个基于扩散模型的视频线稿着色系统，通过显式的对应关系引导和数据增强策略，解决上述挑战。

## 方法详解

### 整体架构

AniDoc 建立在视频扩散模型之上，核心流程为：给定一组彩色参考帧和灰度线稿视频，生成完整的彩色动画视频。系统包含三个关键模块：

1. **对应关系引导模块** (Correspondence Guidance)
2. **数据增强策略** (Data Augmentation)
3. **稀疏草图训练** (Sparse Sketch Training)

### 对应关系引导 (Correspondence Guidance)

为解决时序一致性问题，AniDoc 构建显式的帧间对应关系：

| 阶段 | 技术 | 作用 |
|------|------|------|
| 特征提取 | SIFT + LightGlue | 在参考帧和目标帧间建立稀疏特征匹配 |
| 稠密追踪 | Co-Tracker | 将稀疏匹配扩展为稠密光流场 |
| 颜色传播 | Warping + Attention | 基于对应关系将参考帧颜色传播到目标帧 |

具体流程：
1. 对参考帧和各目标帧提取 SIFT 特征点
2. 使用 LightGlue 进行特征匹配，获取可靠的稀疏对应点对
3. 将稀疏对应点作为初始化，输入 Co-Tracker 获取稠密的像素级追踪结果
4. 对应关系信息以 feature map 形式注入扩散模型的 cross-attention 层

### 数据增强策略

#### 二值化增强 (Binarization Augmentation)

训练时，对输入线稿进行随机二值化处理，模拟不同风格和质量的线稿：

$$I_{bin} = \begin{cases} 1, & I_{gray} > \tau + \epsilon \ 0, & \text{otherwise} \end{cases}$$

其中 $\tau$ 为自适应阈值，$\epsilon \sim \mathcal{U}(-\delta, \delta)$ 为随机扰动。

#### 背景增强 (Background Augmentation)

为减少背景对前景着色的干扰，训练时随机替换背景：
- 50% 概率使用纯白背景
- 30% 概率使用随机纯色背景
- 20% 概率保留原始背景

这迫使模型关注前景角色的结构和颜色，而非依赖背景线索。

### 稀疏草图训练 (Sparse Sketch Training)

在实际应用中，动画师通常只绘制首尾关键帧的彩色版本。AniDoc 提出一种稀疏训练策略：

- 训练时仅提供视频序列的**首帧和末帧**作为彩色参考
- 中间帧以线稿形式输入
- 模型需要自动推断中间帧的着色方案

这种训练方式使模型学会利用有限的参考信息进行合理的颜色内插。

### Sakuga-42M 数据集

本文收集并整理了 Sakuga-42M 数据集，包含：
- 来源：公开动画数据库和视频平台
- 规模：约4200万帧动画数据
- 处理：自动提取线稿、标注关键帧、过滤低质量样本

## 实验结果

### 定量对比

| 方法 | FID↓ | FVD↓ | LPIPS↓ | 时序一致性↑ |
|------|------|------|--------|-----------|
| Reference-based (baseline) | 78.42 | 312.5 | 0.183 | 0.891 |
| w/o correspondence matching | 75.91 | 298.7 | 0.171 | 0.907 |
| AniDoc (full) | **54.33** | **215.8** | **0.124** | **0.952** |
| AniDoc (sparse, 2-ref) | 58.17 | 234.2 | 0.138 | 0.941 |

### 消融实验

| 组件 | FID↓ | 说明 |
|------|------|------|
| Full Model | 54.33 | 完整AniDoc |
| w/o Correspondence | 75.91 | 无对应关系引导，颜色分配混乱 |
| w/o Binarization Aug | 61.27 | 对手绘线稿泛化性下降 |
| w/o Background Aug | 59.84 | 背景区域着色质量下降 |
| w/o Sparse Training | 63.15 | 仅支持dense参考，实用性降低 |

### 训练细节

- 硬件：16× NVIDIA A100 GPU
- 训练时间：5天
- 视频分辨率：512×512，16帧
- 优化器：AdamW，lr=1e-5
- 批大小：每GPU 2个视频片段

## 总结与展望

AniDoc 通过结合显式对应关系引导、针对性数据增强和稀疏参考训练，显著提升了动画线稿着色的质量和实用性。FID 从75.91降至54.33，证明了对应关系引导的关键作用。该系统可直接集成到现有动画制作流水线中，大幅降低上色环节的人工成本。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Image Referenced Sketch Colorization Based on Animation Creation Workflow](image_referenced_sketch_colorization_based_on_animation_creation_workflow.md)
- [\[CVPR 2025\] Fractals made Practical: Denoising Diffusion as Partitioned Iterated Function Systems](fractals_made_practical_denoising_diffusion_as_partitioned_iterated_function_sys.md)
- [\[CVPR 2025\] SemanticDraw: Towards Real-Time Interactive Content Creation from Image Diffusion](semanticdraw_towards_real-time_interactive_content_creation_from_image_diffusion.md)
- [\[CVPR 2025\] Consistent and Controllable Image Animation with Motion Diffusion Models](consistent_and_controllable_image_animation_with_motion_diffusion_models.md)
- [\[CVPR 2025\] StableAnimator: High-Quality Identity-Preserving Human Image Animation](stableanimator_high-quality_identity-preserving_human_image_animation.md)

</div>

<!-- RELATED:END -->
