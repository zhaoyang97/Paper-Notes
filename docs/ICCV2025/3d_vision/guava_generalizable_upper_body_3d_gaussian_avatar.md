---
title: >-
  [论文解读] GUAVA: Generalizable Upper Body 3D Gaussian Avatar
description: >-
  [ICCV 2025][3D视觉][3D高斯] 提出 GUAVA，首个从单张图像通过前馈推理快速重建可动画上半身3D高斯虚拟人的框架，结合模板高斯和 UV 高斯表示，支持丰富面部表情和手势驱动，约0.1s完成重建并实时渲染。 领域现状 领域现状：创建逼真的、可表达表情的上半身虚拟人对影视、游戏、虚拟会议等应用至关重要…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "3D高斯"
  - "上半身虚拟人"
  - "泛化型重建"
  - "单图重建"
  - "表情驱动"
  - "实时渲染"
  - "SMPLX"
  - "FLAME"
---

# GUAVA: Generalizable Upper Body 3D Gaussian Avatar

**会议**: ICCV 2025  
**arXiv**: [2505.03351](https://arxiv.org/abs/2505.03351)  
**代码**: 无（见项目页面）  
**领域**: 3D视觉  
**关键词**: 3D高斯, 上半身虚拟人, 泛化型重建, 单图重建, 表情驱动, 实时渲染, SMPLX, FLAME

## 一句话总结

提出 GUAVA，首个从单张图像通过前馈推理快速重建可动画上半身3D高斯虚拟人的框架，结合模板高斯和 UV 高斯表示，支持丰富面部表情和手势驱动，约0.1s完成重建并实时渲染。

## 研究背景与动机

### 领域现状

**领域现状**：创建逼真的、可表达表情的上半身虚拟人对影视、游戏、虚拟会议等应用至关重要。现有方法存在多重局限：

**3D方法的局限**：

### 现有痛点

**现有痛点**：需要 per-ID 训练（GART、GaussianAvatar、ExAvatar），每个人需要数分钟到数小时的优化

### 核心矛盾

**核心矛盾**：需要多视图或单目视频输入，采集成本高

### 解决思路

**解决思路**：头部重建方法（GAGAvatar）缺乏身体运动表征，全身方法忽略面部细微表情

**2D扩散方法的局限**：

### 补充说明

**补充说明**：生成质量高但 ID 一致性差（尤其大姿态变化时）

### 补充说明

**补充说明**：推理速度慢（每帧需多步去噪）

### 补充说明

**补充说明**：无法灵活控制视角

GUAVA 的定位独特：**首个从单张图像进行泛化型上半身虚拟人重建的方法**，同时解决了速度（约0.1s重建+52 FPS渲染）、质量（超越2D/3D方法）和表达能力（面部表情+手势）三者的平衡。

## 方法详解

### 整体框架

Pipeline 包含四个阶段：
1. **EHM模板模型与追踪**：获取精确的形状、表情、姿态参数
2. **双分支重建**：模板高斯 + UV高斯构建canonical空间的 Ubody 高斯
3. **动画与变形**：使用追踪参数将canonical空间高斯变形到目标姿态
4. **渲染与精炼**：高斯 splatting 渲染粗特征图，StyleUNet 精炼为最终图像

### 关键设计

**EHM（Expressive Human Model）**：解决 SMPLX 面部表达能力不足的问题。将 SMPLX 头部替换为 FLAME 模型（面部表达更精细），通过眼睛关节位移向量对齐两个模型。追踪分两阶段：预训练模型粗估计加上2D关键点损失精细优化。引入可学习关节偏移提升对齐精度。

**模板高斯（Template Gaussians）**：
- 使用预训练 DINOv2 提取图像特征和全局 ID 嵌入
- 将 EHM 顶点投影到屏幕空间，通过线性插值采样外观特征
- 每个顶点还有可优化的基础特征，学习唯一语义信息
- MLP 解码器将三者拼接后预测高斯属性（旋转、缩放、透明度、颜色特征）
- 位置直接取顶点位置

**UV 高斯（UV Gaussians）**：解决模板顶点数量有限、无法捕捉高频细节的问题
- 在 UV 纹理图每个有效像素处预测一个高斯
- 每个 UV 高斯绑定到对应 mesh 三角面片的局部坐标系
- 通过**逆纹理映射**将屏幕空间特征显式映射到 UV 空间
- 使用 mesh rasterizer 过滤不可见区域
- UV 解码器：先用 StyleUNet 填充不可见区域，再用卷积网络预测高斯属性

**神经精炼器**：有效高斯数量可能不足15万，为每个高斯附加潜在特征，splatting 渲染出粗特征图，StyleUNet 精炼器解码为最终高质量图像。

### 损失函数

总损失由图像损失、位置正则化和尺度正则化组成。图像损失包含全图、面部裁剪和手部裁剪三部分的 L1 + LPIPS，对局部细节给予额外关注。位置正则化约束 UV 高斯不偏离父三角面片过远，尺度正则化防止高斯过大。

## 实验关键数据

### 主实验表格

**自重演（Self-reenactment）vs 2D方法**：

| 方法 | PSNR | L1 | SSIM | LPIPS | FPS |
|------|------|----|------|-------|-----|
| MagicPose | 21.25 | 0.0333 | 0.8661 | 0.0913 | 0.12 |
| Champ | 22.01 | 0.0258 | 0.8643 | 0.1000 | 0.53 |
| MimicMotion | 24.46 | 0.0200 | 0.8768 | 0.0879 | 0.21 |
| **GUAVA** | **25.87** | **0.0162** | **0.9000** | **0.0813** | **52.21** |

**自重演 vs 3D方法**：

| 方法 | PSNR | SSIM | LPIPS | 输入 | 重建时间 |
|------|------|------|-------|------|---------|
| ExAvatar | 24.09 | 0.8783 | 0.1064 | 半段视频 | 约2.4h |
| GaussianAvatar | 23.62 | 0.8780 | 0.1085 | 半段视频 | 约1.3h |
| GART | 24.46 | 0.8805 | 0.1016 | 半段视频 | 约7min |
| **GUAVA** | **25.70** | **0.8976** | **0.0836** | 单帧 | **约98ms** |

**跨身份驱动 IPS（ArcFace身份保持）**：

| GUAVA | MimicMotion | Champ | MagicPose |
|-------|-------------|-------|-----------|
| **0.5554** | 0.1310 | 0.3677 | 0.3277 |

### 消融表格

| 设置 | PSNR | SSIM | LPIPS |
|------|------|------|-------|
| 完整模型 | 25.87 | 0.9000 | 0.0813 |
| 去除精炼器 | 24.93 | 0.8851 | 0.1060 |
| 去除逆纹理映射 | 25.65 | 0.8977 | 0.0864 |
| 去除 UV 高斯 | 25.82 | 0.8971 | 0.0877 |
| 去除 EHM | 25.60 | 0.8950 | 0.0846 |

精炼器影响最大（PSNR: 25.87到24.93），EHM 对面部表情精度贡献显著。

### 关键发现

1. GUAVA 比 2D 方法快 100-400倍（52 FPS vs 0.12-0.53 FPS），同时质量更高
2. 仅需单帧即超越需要半段视频训练的3D方法，重建时间从小时级缩短至 98ms
3. 跨身份驱动中 IPS 达 0.5554，远超 MimicMotion 的 0.1310，证明3D表示的 ID 一致性优势
4. EHM 模型使面部表情（眨眼、说话）和手势驱动更加精确
5. 训练数据规模：超26000视频片段、62万帧，覆盖多样上半身场景

## 亮点与洞察

- **双分支高斯表示设计精巧**：模板高斯负责粗粒度几何、UV高斯负责高频细节，互补性强
- **逆纹理映射**将2D屏幕空间特征显式映射到UV空间，避免了隐式学习对应关系的困难
- **EHM 模型**巧妙融合 SMPLX（身体）和 FLAME（面部），取两者之长
- 98ms 重建 + 52 FPS 渲染 = 真正实时可用的虚拟人系统
- 面部和手部区域的额外裁剪损失是提升局部细节的简单有效技巧

## 局限与展望

- 仅支持上半身，不包含全身（下半身、腿部、脚部）
- 依赖 EHM 追踪的精度，追踪失败（如遮挡严重时）会影响重建质量
- 训练需要156 GPU小时（A6000），成本不低
- 对极端姿态或未见过的服装风格泛化能力未深入分析
- 精炼器引入了额外计算，可能影响极端实时场景的部署

## 相关工作与启发

- **GAGAvatar** 实现了泛化型头部高斯重建，GUAVA 将其扩展到上半身
- **ExAvatar** 结合 mesh 和 Gaussian 表示，GUAVA 的双分支设计是其泛化版本
- **MimicMotion/Champ** 等2D扩散动画方法生成质量高但 ID 一致性差，GUAVA 用3D表示解决
- 逆纹理映射的思路可推广到其他基于参数化模型的3D重建任务

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首个泛化型上半身高斯虚拟人框架，EHM+双分支设计有新意
- **实验充分度**: ⭐⭐⭐⭐⭐ — 与2D/3D方法全面对比，消融充分，速度+质量+ID保持多维评估
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，图示丰富
- **价值**: ⭐⭐⭐⭐⭐ — 实时可用性极强，对虚拟人领域有重要推动作用

## 亮点与洞察

## 局限与展望

## 相关工作与启发

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] MoGA: 3D Generative Avatar Prior for Monocular Gaussian Avatar Reconstruction](moga_3d_generative_avatar_prior_for_monocular_gaussian_avatar_reconstruction.md)
- [\[ICCV 2025\] MuGS: Multi-Baseline Generalizable Gaussian Splatting Reconstruction](mugs_multi-baseline_generalizable_gaussian_splatting_reconstruction.md)
- [\[ICCV 2025\] PHD: Personalized 3D Human Body Fitting with Point Diffusion](phd_personalized_3d_human_body_fitting_with_point_diffusion.md)
- [\[ICCV 2025\] FaceLift: Learning Generalizable Single Image 3D Face Reconstruction from Synthetic Heads](facelift_learning_generalizable_single_image_3d_face_reconstruction_from_synthet.md)
- [\[CVPR 2026\] OMG-Avatar: One-shot Multi-LOD Gaussian Head Avatar](../../CVPR2026/3d_vision/omg-avatar_one-shot_multi-lod_gaussian_head_avatar.md)

</div>

<!-- RELATED:END -->
