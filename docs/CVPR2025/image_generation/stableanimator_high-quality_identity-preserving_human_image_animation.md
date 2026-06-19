---
title: >-
  [论文解读] StableAnimator: High-Quality Identity-Preserving Human Image Animation
description: >-
  [CVPR 2025][图像生成][人物图像动画] StableAnimator 提出首个端到端的身份保持视频扩散框架，通过全局内容感知 Face Encoder 和分布感知 ID Adapter 在训练中维护身份一致性，并在推理时利用 Hamilton-Jacobi-Bellman（HJB）方程优化面部质量，无需任何后处理工具即可生成高保真的人物动画视频。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "人物图像动画"
  - "身份保持"
  - "视频扩散模型"
  - "HJB方程"
  - "面部优化"
---

# StableAnimator: High-Quality Identity-Preserving Human Image Animation

**会议**: CVPR 2025  
**arXiv**: [2411.17697](https://arxiv.org/abs/2411.17697)  
**代码**: [https://francis-rings.github.io/StableAnimator](https://francis-rings.github.io/StableAnimator)  
**领域**: 扩散模型 / 图像生成  
**关键词**: 人物图像动画, 身份保持, 视频扩散模型, HJB方程, 面部优化

## 一句话总结

StableAnimator 提出首个端到端的身份保持视频扩散框架，通过全局内容感知 Face Encoder 和分布感知 ID Adapter 在训练中维护身份一致性，并在推理时利用 Hamilton-Jacobi-Bellman（HJB）方程优化面部质量，无需任何后处理工具即可生成高保真的人物动画视频。

## 研究背景与动机

**领域现状**：扩散模型在图像/视频生成中取得了巨大成功，人物图像动画利用姿势序列来驱动参考图片生成可控的人物动画视频。代表方法包括 AnimateAnyone、MagicAnimate、Champ、MimicMotion、ControlNeXt 等。

**现有痛点**：当姿势序列包含大幅运动变化时，现有方法在面部区域出现严重扭曲和不一致，破坏身份信息。图像域的 ID 保持方法（如 IP-Adapter-FaceID、InstantID、PuLID）无法直接用于视频扩散模型——因为时间层会改变空间分布，导致空间先验不稳定，面部嵌入与扩散潜在变量的分布不匹配。最新的动画模型（MimicMotion、ControlNeXt）依赖第三方换脸工具 FaceFusion 做后处理，但这会破坏原始像素分布、降低视频整体质量。

**核心矛盾**：在视频扩散模型中，时间建模层必然会改变空间特征分布，而图像域的 ID 保持方法依赖稳定的空间分布。这个 ID 一致性与视频保真度之间的冲突是核心难题——保持 ID 意味着约束空间分布，而时间建模需要修改它。

**本文目标**：设计一个端到端的方案，在训练和推理两个阶段都确保 ID 一致性，同时不依赖任何后处理工具。

**切入角度**：观察到 ID 保持失败的根本原因是面部嵌入与经过时间层后的扩散潜在变量之间的分布偏移。如果在每个空间层将两者的分布对齐，就能在时间建模后仍保持 ID 信息。

**核心 idea**：在训练中用分布感知 ID Adapter 通过均值-标准差对齐来弥合面部嵌入与图像嵌入的分布差距；在推理中将 HJB 方程的求解过程整合进扩散去噪，用面部相似度梯度约束去噪路径使其朝向最优 ID 一致性方向。

## 方法详解

### 整体框架

StableAnimator 基于 Stable Video Diffusion（SVD）。参考图像通过三条路径进入模型：(1) VAE 编码为潜在码并复制匹配帧数后与主潜在变量拼接；(2) CLIP Image Encoder 提取图像嵌入送入 U-Net 交叉注意力和 Face Encoder；(3) ArcFace 提取面部嵌入送入 Face Encoder 精炼。PoseNet 从姿势序列提取特征加到噪声潜在变量上。在每个 U-Net 块中，分布感知 ID Adapter 在时间建模前对齐面部和图像嵌入的分布。推理时，HJB 方程优化在每个去噪步骤中嵌入，直接优化预测样本的面部相似度。

### 关键设计

1. **全局内容感知 Face Encoder**:

    - 功能：在注入 U-Net 前增强面部嵌入对参考图像整体布局的感知
    - 核心思路：面部嵌入通过多个交叉注意力块与参考图像嵌入交互。这让面部信息在被注入 U-Net 之前就了解到全局上下文（布局、背景等），避免参考图像中与 ID 无关的元素给面部建模带来噪声
    - 设计动机：直接将 ArcFace 面部嵌入送入 U-Net 缺乏全局感知，参考图中的背景等无关元素会干扰面部建模质量

2. **分布感知 ID Adapter**:

    - 功能：确保视频保真度同时保持 ID 一致性
    - 核心思路：在 U-Net 每个空间层中，扩散潜在变量 $z_i$ 分别与图像嵌入和精炼面部嵌入做交叉注意力，得到 $z_i^{img}$ 和 $z_i^{face}$。计算两者的均值和标准差，通过标准化对齐使两者分布匹配：$\bar{z}_i^{face} = \frac{z_i^{face} - \mu_{face}}{\sigma_{face}} \times \sigma_{img} + \mu_{img}$，然后将对齐后的 $\bar{z}_i^{face}$ 与 $z_i^{img}$ 相加。对齐在时间层之前完成，保证后续时间建模不会破坏 ID 信息
    - 设计动机：时间层会改变空间分布，如果直接将面部嵌入注入会出现分布偏移，导致 ID 信息不稳定。通过均值-标准差对齐让面部嵌入始终与图像嵌入处于同一分布域中，后续时间层的扰动对两者影响一致

3. **HJB 方程面部优化**:

    - 功能：在推理时无需训练额外组件即可增强面部质量
    - 核心思路：在每个去噪步骤中，将扩散模型预测的样本 $x_{pred}$ clone 为可优化变量 $x_{op}$，通过 VAE 解码后用 ArcFace 计算与参考图像的面部余弦相似度损失：$loss = (1 - \text{Cos}(\text{Arc}(f_{pred}), \text{Arc}(y)))$，用 Adam 优化器迭代 10 步。优化后的 $x_{op}$ 替代原始预测参与后续去噪。作者证明这个过程等价于求解 HJB 方程的控制信号 $c_t^* = \gamma = r(x_1 - X_t)/(1+r(1-t))$，且其 SDE 形式与扩散 SDE 结构相同，可无缝整合
    - 设计动机：后处理换脸工具在不同域中操作，破坏原始动画的像素分布和语义一致性。HJB 优化在扩散去噪过程内部进行，始终适应当前去噪潜在变量的分布，避免域外干扰

### 损失函数 / 训练策略

- 使用带面部 mask 加权的重建损失：$\mathcal{L} = \mathbb{E}_\varepsilon(\|(z_{gt} - z_\varepsilon) \odot (1+M)\|^2)$，其中 $M$ 是 ArcFace 提取的面部区域 mask，对面部区域给予双倍权重
- 可训练组件：U-Net、FaceEncoder、PoseNet。ID Adapter 使用 SVD 中预训练的空间交叉注意力权重初始化
- 3K 互联网视频（60-90秒）训练，DWPose 提取骨骼，ArcFace 提取面部嵌入/mask
- 4×A100 80G 训练 20 epochs，batch size 1/GPU，学习率 1e-5

## 实验关键数据

### 主实验

| 模型 | CSIM ↑ | FVD ↓ | L1 ↓ | PSNR ↑ |
|------|--------|-------|------|--------|
| AnimateAnyone | 0.457/0.316 | 171.90/383.45 | -/3.15E-4 | 29.56/27.14 |
| Unianimate | 0.479/0.347 | 148.06/394.32 | 2.66/2.82E-4 | 30.77/27.46 |
| MimicMotion | 0.262/0.242 | 326.57/604.13 | 5.85/3.55E-4 | -/22.94 |
| ControlNeXt | 0.360/0.264 | 326.57/389.45 | 6.20/2.90E-4 | -/25.28 |
| **StableAnimator** | **0.831/0.805** | **140.62/349.94** | 2.87/2.71E-4 | 30.81/28.85 |

格式为 TikTok / Unseen100 数据集结果。CSIM 为面部余弦相似度（越高越好）。

### 消融实验

| 配置 | CSIM ↑ | FVD ↓ |
|------|--------|-------|
| w/o Face Masks | 0.639 | 382.25 |
| w/o Face Encoder | - | - |
| IP-Adapter 替代 | ID 提升但视频质量剧降 |
| FaceFusion 后处理 | 面部改善但视频保真度下降 |
| **StableAnimator 完整** | **0.805** | **349.94** |

### 关键发现

- StableAnimator 在 CSIM 上超越最强竞争者 Unianimate 达 36.9%（TikTok）和 45.8%（Unseen100），同时保持最佳 FVD
- 直接插入 IP-Adapter 改善 ID 但严重损害视频质量和单帧质量——验证了时间层干扰假设
- FaceFusion 后处理改善面部但相对损害视频保真度——后处理工具与扩散模型不在同一分布域
- HJB 面部优化在与去噪同步进行时能有效消除细节扭曲，因为始终适应当前分布

## 亮点与洞察

1. "分布对齐"是一个简洁优雅的解决方案——用均值-标准差归一化即可桥接图像域 ID方法与视频扩散模型的分布差异
2. HJB 方程与扩散 SDE 的结构等价性发现非常有趣，使面部优化可以无缝嵌入去噪过程而非作为外部后处理
3. 首个证明端到端 ID 保持可以完全替代换脸后处理的工作，质量还更好
4. 面部 mask 加权损失是简单但有效的设计，让模型在训练中更关注面部区域

## 局限与展望

- 训练数据量（3K 视频）相对较少，更大规模数据可能进一步提升泛化能力
- HJB 面部优化在每个去噪步骤迭代 10 次，增加推理时间
- GPU 内存需求 12.50G 处于中等水平（处理 16 帧 576×1024）
- 未来可探索将 ID 保持方案推广到更多视频生成任务

## 相关工作与启发

- 与 IP-Adapter-FaceID 相比，分布感知 ID Adapter 避免了时间层造成的分布偏移
- 与 MimicMotion/ControlNeXt 的 FaceFusion 后处理相比，端到端方案产生更一致的视频
- HJB 优化思路可推广到其他需要在扩散推理中施加约束的场景（如风格一致性、语义对齐）

## 评分

- **新颖性**: 8/10 — 分布对齐和 HJB 面部优化都是有说服力的创新，理论推导严谨
- **实验充分度**: 8/10 — 两个数据集+全面消融+8个竞争方法对比
- **写作质量**: 8/10 — 方法动机清晰，数学推导完整
- **价值**: 8/10 — 首个真正端到端 ID 保持的人物动画方案，实用意义大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Free-viewpoint Human Animation with Pose-correlated Reference Selection](free-viewpoint_human_animation_with_pose-correlated_reference_selection.md)
- [\[ICML 2025\] FlexiClip: Locality-Preserving Free-Form Character Animation](../../ICML2025/image_generation/flexiclip_locality-preserving_free-form_character_animation.md)
- [\[CVPR 2025\] EmoDubber: Towards High Quality and Emotion Controllable Movie Dubbing](emodubber_towards_high_quality_and_emotion_controllable_movie_dubbing.md)
- [\[CVPR 2025\] OmniStyle: Filtering High Quality Style Transfer Data at Scale](omnistyle_filtering_high_quality_style_transfer_data_at_scale.md)
- [\[CVPR 2025\] ArtiFade: Learning to Generate High-quality Subject from Blemished Images](artifade_learning_to_generate_high-quality_subject_from_blemished_images.md)

</div>

<!-- RELATED:END -->
