---
title: >-
  [论文解读] Motion Prompting: Controlling Video Generation with Motion Trajectories
description: >-
  [CVPR 2025][视频生成][提示学习] 将时空稀疏/稠密点轨迹作为"运动提示"训练ControlNet，用单一模型实现物体控制、相机控制、运动迁移、拖拽编辑等多种运动控制能力，并展现出逼真物理行为的涌现特性。 领域现状： 视频生成模型主要依赖文本提示控制，但文本本质上难以精确描述运动的细节——"a bear quic…
tags:
  - "CVPR 2025"
  - "视频生成"
  - "提示学习"
  - "扩散模型"
  - "point trajectories"
  - "ControlNet"
  - "camera control"
  - "object control"
  - "motion transfer"
---

# Motion Prompting: Controlling Video Generation with Motion Trajectories

**会议**: CVPR 2025  
**arXiv**: [2412.02700](https://arxiv.org/abs/2412.02700)  
**代码**: [https://motion-prompting.github.io/](https://motion-prompting.github.io/)  
**领域**: 视频生成  
**关键词**: motion prompting, video diffusion, point trajectories, ControlNet, camera control, object control, motion transfer

## 一句话总结

将时空稀疏/稠密点轨迹作为"运动提示"训练ControlNet，用单一模型实现物体控制、相机控制、运动迁移、拖拽编辑等多种运动控制能力，并展现出逼真物理行为的涌现特性。

## 研究背景与动机

**领域现状**: 视频生成模型主要依赖文本提示控制，但文本本质上难以精确描述运动的细节——"a bear quickly turns its head"可以有无数种解读。运动的时序、轨迹、加减速等细微差别需要更直接的控制信号。

**现有方案的不足**:
1. **工程复杂度高**: Tora、MotionCtrl、DragNUWA等需要两阶段训练、专用损失函数、特殊架构或多模块联合微调
2. **控制信号受限**: bounding box、分割mask、人体pose等实体级信号只能控制特定类型运动
3. **光流表示不足**: 光流无法跨帧链接（误差累积），不处理遮挡，不适合生成任务
4. **不同运动类型需要不同adapter**: 如MOFA-Video对相机和物体运动需要分别训练

**核心动机**: 点轨迹（point tracks）可以编码任意数量的轨迹、物体或全局运动、时序稀疏运动，是一种足够灵活的"运动语言"——类比文本是"语义语言"，motion prompt是"运动语言"。

## 方法详解

### 整体框架

在预训练视频扩散模型（Lumiere）上训练ControlNet，以点轨迹为条件信号。训练极简：单阶段、均匀采样稠密轨迹、无特殊工程。推理时通过motion prompt expansion将高层用户意图转化为详细轨迹。

### 关键设计

#### 模块一：Motion Prompt表示与编码

- **表示**: N条点轨迹 $p \in \mathbb{R}^{N \times T \times 2}$ + 可见性标记 $v \in \mathbb{R}^{N \times T}$
- **编码**: 每条轨迹分配随机唯一嵌入向量 $\phi^n \in \mathbb{R}^C$（从固定池采样），将嵌入放置在轨迹经过的时空位置，其余位置为零
- **数学形式**: $c[t, x_t^n, y_t^n] = v[n,t] \cdot \phi_n$
- 多条轨迹经过同一位置时嵌入相加；完全稠密轨迹等价于前向warp密集嵌入网格

关键优势：可编码任意密度/时序跨度/空间分布的轨迹，为ControlNet提供统一条件输入。

#### 模块二：Motion Prompt Expansion（运动提示展开）

将高层用户请求转化为详细轨迹的过程（类比text prompt expansion）：
- **图像"交互"**: 鼠标拖拽→周围生成网格轨迹，支持时序稀疏（多次拖拽）、背景静止约束
- **几何体控制**: 将鼠标运动映射到几何代理（如球体）上，实现旋转等复杂运动
- **相机控制**: 单目深度估计→点云→按相机轨迹重投影→得到2D轨迹（含z-buffer遮挡）
- **运动迁移**: 从源视频提取轨迹，应用到新首帧
- **运动组合**: 物体轨迹转位移+叠加到相机轨迹→同时控制相机和物体

#### 模块三：训练策略

- 数据: 对2.2M视频用BootsTAP提取16384条稠密轨迹/视频
- 训练时随机采样轨迹数量（对数均匀分布从 $2^0$ 到 $2^{13}$）
- 标准ControlNet训练，单阶段，无数据过滤
- **关键发现**: 仅用稠密轨迹训练，模型可泛化到稀疏轨迹、空间局部轨迹、非首帧起始轨迹

### 损失函数

标准扩散损失（denoising score matching），无任何额外损失。ControlNet的zero convolution确保训练初期不干扰预训练模型——这也是训练如此简单的原因之一。

## 实验关键数据

### 主实验表

**DAVIS验证集定量评估**:

| 轨迹数 | 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | FVD↓ | EPE↓ |
|--------|------|-------|-------|--------|------|------|
| N=16 | ImageConductor | 12.184 | 0.175 | 0.502 | 1838.9 | 24.263 |
| N=16 | DragAnything | 15.119 | 0.305 | 0.378 | 1282.8 | 9.800 |
| N=16 | **Ours** | **16.618** | **0.405** | **0.319** | **1322.0** | **8.319** |
| N=2048 | DragAnything | 14.845 | 0.286 | 0.397 | 1468.4 | 12.485 |
| N=2048 | **Ours** | **19.327** | **0.608** | **0.227** | **655.9** | **3.887** |

**人类评估** (2AFC, win rate%):

| vs. | 运动一致性 | 运动质量 | 视觉质量 |
|-----|----------|---------|---------|
| ImageConductor | 74.3% | 80.5% | 77.3% |
| DragAnything | 74.5% | 75.7% | 73.7% |

### 消融表

**训练轨迹密度消融** (4轨迹/2048轨迹测试):

| 训练策略 | PSNR (N=4) | PSNR (N=2048) | EPE (N=2048) |
|----------|------------|---------------|--------------|
| Sparse only | 15.075 | 15.697 | 26.724 |
| Dense + Sparse | 15.162 | 15.294 | 27.931 |
| **Dense only** | **15.638** | **19.197** | **4.806** |

### 关键发现

1. **稠密训练最优**: 仅用稠密轨迹训练在稀疏推理时也表现最好——稀疏轨迹训练信号太弱
2. **轨迹越多效果越好**: N从1到2048，PSNR提升约4 dB，FVD降低约3倍
3. **涌现物理行为**: 拖拽头发出现自然摆动、拨弄沙子出现物理合理的散布——模型学到了运动先验
4. **泛化性强**: 训练仅用均匀分布轨迹，推理时泛化到空间局部、时序稀疏、非首帧起始等分布外条件
5. **无需相机标注**: 通过运动轨迹间接实现相机控制，虽未显式训练相机运动

## 亮点与洞察

1. **"运动语言"的概念**: 将运动控制统一为通用的轨迹条件，一个模型覆盖多种控制任务——这是非常优雅的统一框架
2. **极简训练**: 单阶段、无特殊损失、无数据过滤——与之前方法的复杂工程形成鲜明对比
3. **Motion Prompt Expansion**: 类比text prompt rewriting，通过计算机视觉管线将高层意图转化为低层轨迹，桥接用户意图与模型输入
4. **探测视频模型的物理理解**: 通过motion prompt可以"提问"模型对物理世界的理解——拖拽棋子会发生什么？头发被拉会怎样？
5. **实用性**: 提供了鼠标交互GUI，虽非实时（12分钟/视频），但展示了未来与生成世界模型交互的方向

## 局限性

1. 生成不是实时的（约12分钟/视频），远离interactive应用
2. 非因果生成：需要完整轨迹输入才能生成，不支持流式交互
3. 运动条件有时会导致伪影：如cow的角被错误"锁定"到背景
4. 视频模型本身的局限：如移动棋子时突然生成新棋子

## 相关工作与启发

- **DragAnything**: 实体级track控制，用latent warp获得准确运动但产生视觉伪影
- **MotionCtrl**: 显式分离相机和物体运动控制，但需要专用设计
- **MOFA-Video**: 不同运动类型需要不同adapter→Motion Prompting用统一表示解决
- **CoTracker/BootsTAP**: 稠密轨迹估计的进展使高质量训练数据大规模获取成为可能
- **启发**: 未来的世界模型（World Model）可能天然支持motion prompt作为交互接口，用于embodied AI的visual planning

## 评分

⭐⭐⭐⭐⭐ — 概念优美（motion作为运动语言），设计极简（单阶段ControlNet），应用广泛（一个模型多种控制），涌现行为令人惊喜，来自Google DeepMind的扎实工作，对视频生成控制范式有重要影响。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Through-The-Mask: Mask-based Motion Trajectories for Image-to-Video Generation](through-the-mask_mask-based_motion_trajectories_for_image-to-video_generation.md)
- [\[CVPR 2025\] MotionPro: A Precise Motion Controller for Image-to-Video Generation](motionpro_a_precise_motion_controller_for_image-to-video_generation.md)
- [\[ICCV 2025\] Free-Form Motion Control: Controlling the 6D Poses of Camera and Objects in Video Generation](../../ICCV2025/video_generation/free-form_motion_control_controlling_the_6d_poses_of_camera_and_objects_in_video.md)
- [\[CVPR 2025\] MotionStone: Decoupled Motion Intensity Modulation with Diffusion Transformer for Image-to-Video Generation](motionstone_decoupled_motion_intensity_modulation_with_diffusion_transformer_for.md)
- [\[CVPR 2025\] ConMo: Controllable Motion Disentanglement and Recomposition for Zero-Shot Motion Transfer](conmo_controllable_motion_disentanglement_and_recomposition_for_zero-shot_motion.md)

</div>

<!-- RELATED:END -->
