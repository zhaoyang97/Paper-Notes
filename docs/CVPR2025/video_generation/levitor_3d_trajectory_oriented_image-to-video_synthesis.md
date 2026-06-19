---
title: >-
  [论文解读] LeviTor: 3D Trajectory Oriented Image-to-Video Synthesis
description: >-
  [CVPR 2025][视频生成][3D轨迹控制] LeviTor首次在image-to-video合成中引入3D物体轨迹控制，通过将物体mask用K-means聚类为少量代表点并结合深度信息作为控制信号注入SVD模型，实现了遮挡关系、前后移动和环绕等复杂3D运动的精准控制，在DAVIS上FID/FVD分别达到25.41/190.44。
tags:
  - "CVPR 2025"
  - "视频生成"
  - "3D轨迹控制"
  - "图像到视频"
  - "视频扩散模型"
  - "K-means聚类点"
  - "深度信息"
---

# LeviTor: 3D Trajectory Oriented Image-to-Video Synthesis

**会议**: CVPR 2025  
**arXiv**: [2412.15214](https://arxiv.org/abs/2412.15214)  
**代码**: [https://github.com/ant-research/LeviTor](https://github.com/ant-research/LeviTor)  
**领域**: 扩散模型/视频生成  
**关键词**: 3D轨迹控制, 图像到视频, 视频扩散模型, K-means聚类点, 深度信息

## 一句话总结

LeviTor首次在image-to-video合成中引入3D物体轨迹控制，通过将物体mask用K-means聚类为少量代表点并结合深度信息作为控制信号注入SVD模型，实现了遮挡关系、前后移动和环绕等复杂3D运动的精准控制，在DAVIS上FID/FVD分别达到25.41/190.44。

## 研究背景与动机

**领域现状**：轨迹控制的视频生成已经取得了较大进展,DragNUWA将稀疏笔画转为密集光流空间,DragAnything提取实体表示实现entity-level控制,TrackGo通过自由形式mask和箭头定义运动。这些方法都在2D平面上操作轨迹。

**现有痛点**：所有现有方法仅考虑2D轨迹，在真实3D环境中会产生歧义。例如让热气球飞过建筑物时,2D轨迹无法区分气球是从建筑前方还是后方经过;物体前后移动时2D轨迹无法表达深度变化导致大小不合理;环绕运动等复杂3D运动用2D轨迹根本无法表达。

**核心矛盾**：2D轨迹信息维度不足以表达3D空间中的运动——同一条2D轨迹可以对应无穷多条3D轨迹，导致生成结果模糊或不符合透视投影规律。但要求用户输入精确的3D轨迹又太困难。

**本文目标** (1) 设计一种隐式表达3D运动的控制信号，不需要显式3D轨迹标注；(2) 让用户能简便地输入3D控制信息；(3) 准确控制物体间的遮挡关系和深度变化。

**切入角度**：作者发现物体mask的K-means聚类点的分布变化可以隐式编码3D运动信息——点的聚拢表示物体远离、点的分散表示物体靠近、点在另一物体后消失表示遮挡。结合DepthAnythingV2估计的相对深度值，可以用简单的2D标注+深度值近似3D轨迹。

**核心 idea**：用K-means聚类点的空间分布变化+相对深度值作为3D运动的代理表示，通过ControlNet注入视频扩散模型实现3D轨迹控制。

## 方法详解

### 整体框架

训练阶段：从SA-V (VOS)数据集获取高质量视频和物体mask标注，对每帧每个mask做K-means聚类得到代表点，用DepthAnythingV2估计每帧深度图并给聚类点赋深度值，将2D高斯热图轨迹+实例信息+深度信息拼接为控制信号，通过ControlNet注入SVD模型训练。推理阶段：用户在图像上选择要移动的物体mask，绘制2D轨迹并指定深度变化，系统将物体mask的像素点投影到3D空间中移动后再渲染回2D得到逐帧mask，然后用K-means提取控制点生成视频。

### 关键设计

1. **K-means聚类点控制信号**:

    - 功能：将稠密的物体mask压缩为少量代表点，作为运动控制信号
    - 核心思路：对每帧的每个物体mask计算面积比并乘以超参数 $\alpha$ 确定聚类点数 $k$。如果物体mask面积在时间维度上变化超过10倍（说明有遮挡/远近变化），则保证 $k \geq 3$ 以更好表示变化，同时限制 $k \leq 8$。聚类点的**聚拢/分散**隐式反映深度变化，点的**消失/出现**反映遮挡
    - 设计动机：密集mask作为控制会导致物体只能平移无法产生非刚性变形；少量聚类点给了生成模型更多自由度添加运动细节。同时K-means点数随mask大小自适应调整

2. **3D渲染mask推理管线**:

    - 功能：将用户的稀疏3D轨迹输入转换为符合物理规律的逐帧控制信号
    - 核心思路：将起始图像的2D像素+深度值投影到3D相机坐标系 $[X_i, Y_i, Z_i]^T = \mathbf{K}^{-1} \cdot [x_i, y_i, 1]^T \cdot d_i$，在3D空间中按用户指定的位移向量 $\mathbf{T}$ 移动选中物体的点，然后用PyTorch3D渲染器将所有3D点投影回2D图像，得到包含遮挡关系和透视缩放的mask序列
    - 设计动机：在3D空间中移动物体再渲染为2D，天然保证了遮挡关系和大小变化符合透视投影规律，避免用户手动输入复杂的多点轨迹

3. **深度+实例信息融合**:

    - 功能：为控制信号添加深度和物体归属维度
    - 核心思路：用DepthAnythingV2估计相对深度，在每个聚类点位置采样深度值 $d_t^i = D_t(x_t^i, y_t^i)$。同时为每个控制点标注所属实例ID。将高斯热图轨迹、深度图、实例图三种信号拼接后送入ControlNet
    - 设计动机：消融实验证明实例信息是最关键的（没有它FVD从190升到228），因为模型需要知道哪些控制点属于同一物体；深度信息提供辅助的3D位置线索

### 损失函数 / 训练策略

标准扩散模型训练目标：$\mathcal{L} = \mathbb{E}_{z_t, z^0, t, \epsilon}[\|\epsilon - \epsilon_\theta^c(z_t; t, z^0, c_{\text{traj}})\|^2]$。基于SVD模型，用ControlNet注入控制信号。训练200K iterations，AdamW，lr=1e-5，16张A100，batch size 16，分辨率288×512。

## 实验关键数据

### 主实验

| 设置 | 方法 | FID↓ | FVD↓ | ObjMC↓ |
|------|------|------|------|--------|
| Multi-Points | DragAnything | 36.04 | 324.95 | 38.86 |
| Multi-Points | DragNUWA 1.5 | 42.34 | 299.96 | 23.12 |
| Multi-Points | **LeviTor** | **25.41** | **190.44** | 25.97 |
| Single-Point | DragAnything | 36.69 | 327.41 | 42.19 |
| Single-Point | DragNUWA 1.5 | 44.82 | 330.17 | 33.03 |
| Single-Point | **LeviTor** | **28.79** | **226.45** | 37.39 |

### 消融实验

| Depth | Instance | FID↓ | FVD↓ | ObjMC↓ |
|:-----:|:--------:|------|------|--------|
| ✗ | ✗ | 27.83 | 227.58 | 29.82 |
| ✓ | ✗ | 28.04 | 221.29 | 29.13 |
| ✗ | ✓ | 25.45 | 199.44 | 25.40 |
| ✓ | ✓ | **25.41** | **190.44** | 25.97 |

### 关键发现

- 实例信息比深度信息重要得多：加Instance使FVD从228降到199，加Depth仅从228降到221。这说明模型需要明确知道哪些控制点属于同一物体
- 从Single-Point到Multi-Points对LeviTor和DragNUWA均有显著提升，验证了多点控制能更好表达物体大小变化和遮挡
- LeviTor在ObjMC指标上略逊于DragNUWA，因为LeviTor没有用tracking方法获取完整轨迹，但FID/FVD远优于所有baseline
- 控制点数量需要平衡：太少时运动幅度大但可能变形，太多时接近mask控制导致物体只能平移

## 亮点与洞察

- **K-means聚类点作为3D运动的代理表示非常巧妙**——不需要显式3D标注，通过点的聚散变化隐式编码深度变化，既有足够的信息量又给扩散模型留了生成自由度。这种"loose control"的设计思路可迁移到其他需要3D控制的生成任务
- **3D渲染mask管线是让用户易用的关键**——用户只需画一条2D轨迹+调深度值，系统在3D空间完成物理正确的mask变换，大幅降低了3D输入的门槛
- **训练时只用VOS数据+深度估计就能学到3D控制能力**，不依赖任何3D轨迹标注，数据获取成本极低

## 局限与展望

- 受SAM分割质量限制，无法控制未被正确分割的物体
- 不理解物理规律，不能自动生成合理运动，必须依赖用户提供轨迹
- 无法控制物体内部的姿态变化（如行走时腿的运动），因为训练未使用tracking数据
- 基于SVD基础模型，分辨率和帧数受限；小物体面部重建质量差，大运动幅度时出现伪影
- 可以扩展到更强的基础视频模型（如CogVideoX）以提升生成质量和运动范围

## 相关工作与启发

- **vs DragAnything**: DragAnything提取entity representation实现entity-level控制，但仅有2D轨迹，训练时用单点轨迹+首帧mask语义信息，增加轨迹数改善有限。LeviTor通过多点+深度实现了真正的3D控制
- **vs DragNUWA**: DragNUWA将稀疏笔画转为密集光流空间，在ObjMC指标上表现好但不支持选择操作区域，且经常把物体运动误解为相机运动。LeviTor通过包含所有物体mask避免了这一问题
- **vs 3D-TrajMaster**: 3D-TrajMaster用6DoF位姿序列控制多实体3D运动，但需要精确的6DoF输入，用户门槛极高。LeviTor的2D画线+深度值方式更实用

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次在I2V中引入3D轨迹控制，K-means聚类点+深度值的控制信号设计很有创意
- 实验充分度: ⭐⭐⭐⭐ 定量定性对比充分，消融实验设计合理，but仅在DAVIS上定量评估
- 写作质量: ⭐⭐⭐⭐ 论文结构清晰，图示直观
- 价值: ⭐⭐⭐⭐⭐ 开创了I2V中3D控制的新范式，对视频生成领域有重要推动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Tora: Trajectory-Oriented Diffusion Transformer for Video Generation](tora_trajectory-oriented_diffusion_transformer_for_video_generation.md)
- [\[CVPR 2025\] Geometry-guided Online 3D Video Synthesis with Multi-View Temporal Consistency](geometry-guided_online_3d_video_synthesis_with_multi-view_temporal_consistency.md)
- [\[CVPR 2025\] FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance](flashmotion_few-step_controllable_video_generation_with_trajectory_guidance.md)
- [\[CVPR 2025\] IDOL: Instant Photorealistic 3D Human Creation from a Single Image](idol_instant_photorealistic_3d_human_creation_from_a_single_image.md)
- [\[CVPR 2025\] StreetCrafter: Street View Synthesis with Controllable Video Diffusion Models](streetcrafter_street_view_synthesis_with_controllable_video_diffusion_models.md)

</div>

<!-- RELATED:END -->
