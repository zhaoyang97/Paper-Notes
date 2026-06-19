---
title: >-
  [论文解读] MaskGWM: A Generalizable Driving World Model with Video Mask Reconstruction
description: >-
  [CVPR 2025][自动驾驶][世界模型] 本文将MAE式掩码重建任务与扩散生成过程相结合，提出MaskGWM驾驶世界模型，通过扩散相关掩码token、行式掩码注意力和行式跨视角模块三项创新设计，在长时序预测和多视角生成两个场景下均显著超越了现有SOTA。 领域现状：驾驶世界模型旨在根据动作预测未来环境变化…
tags:
  - "CVPR 2025"
  - "自动驾驶"
  - "世界模型"
  - "Transformer"
  - "掩码重建"
  - "多视角生成"
  - "长时预测"
---

# MaskGWM: A Generalizable Driving World Model with Video Mask Reconstruction

**会议**: CVPR 2025  
**arXiv**: [2502.11663](https://arxiv.org/abs/2502.11663)  
**代码**: [https://github.com/SenseTime-FVG/OpenDWM](https://github.com/SenseTime-FVG/OpenDWM)  
**领域**: 自动驾驶  
**关键词**: 世界模型, 扩散Transformer, 掩码重建, 多视角生成, 长时预测

## 一句话总结
本文将MAE式掩码重建任务与扩散生成过程相结合，提出MaskGWM驾驶世界模型，通过扩散相关掩码token、行式掩码注意力和行式跨视角模块三项创新设计，在长时序预测和多视角生成两个场景下均显著超越了现有SOTA。

## 研究背景与动机

**领域现状**：驾驶世界模型旨在根据动作预测未来环境变化，是实现自动驾驶强泛化能力的关键。当前主流方案构建在视频预测模型之上，利用扩散模型（如SVD、Vista等）生成高保真视频序列。

**现有痛点**：尽管基于扩散的生成器可以产出逼真的视频帧，但这些模型面临两大瓶颈：(1) 预测时长受限，难以支持长时序rollout；(2) 泛化能力不足，在新场景（如从nuScenes到Waymo的零样本迁移）上表现明显下降。根本原因在于，纯粹的像素级生成损失缺乏对高层语义上下文的显式建模，模型容易过拟合训练分布的视觉细节而非场景结构。

**核心矛盾**：像素级生成关注局部纹理和细节，而泛化能力需要模型理解全局空间-时间结构。MAE式的特征级上下文学习恰好擅长捕获这种高层结构信息，但如何将其与扩散生成过程有效融合是一个非平凡问题——两者在信息处理层次和训练目标上存在本质差异。

**本文目标**：设计一种在Diffusion Transformer框架下融合像素生成与特征掩码重建的训练范式，使模型既保持生成质量又获得更强的场景理解能力，从而提升长时预测和跨数据集泛化性能。

**切入角度**：作者观察到，MAE中的masked self-attention在时空扩展时存在信息泄露和计算低效问题，本文另辟蹊径采用shifted self-attention配合行式掩码，自然适配DiT结构；同时设计扩散相关的掩码token，桥接掩码重建与扩散去噪两个过程的语义鸿沟。

**核心idea**：在DiT驾驶世界模型中引入辅助掩码重建任务，通过扩散噪声感知的mask token和时空行式掩码策略，使模型在生成高质量视频的同时学习到更可泛化的场景表征。

## 方法详解

### 整体框架
MaskGWM基于Diffusion Transformer (DiT)架构，输入为过去帧序列和动作条件（如3D bounding box layout），输出为未来视频帧序列。在标准扩散生成训练之外，模型增加了一个并行的掩码重建分支：将部分输入token随机遮蔽，要求模型在特征层面重建被遮蔽的内容。最终训练目标是扩散损失与掩码重建损失的加权和。模型有两个变体：MaskGWM-long专注长时预测（自回归多步rollout），MaskGWM-mview专注多视角一致生成。

### 关键设计

1. **扩散相关掩码Token (Diffusion-related Mask Tokens)**:

    - 功能：在被掩码位置插入可学习的占位token，供后续重建使用
    - 核心思路：与MAE使用固定mask token不同，MaskGWM的mask token是与扩散时间步$t$相关的。具体来说，mask token被注入与当前扩散噪声水平匹配的信息，使其与周围的噪声token在语义空间上对齐。这通过将扩散timestep embedding与mask token进行调制实现
    - 设计动机：在扩散模型中，不同时间步的特征包含不同层次的信息（高噪声时偏全局结构，低噪声时偏局部细节）。若mask token不感知扩散状态，重建目标与生成目标之间会产生语义错位，导致两个任务相互干扰而非协同增益

2. **行式掩码与移位自注意力 (Row-wise Mask with Shifted Self-Attention)**:

    - 功能：将掩码重建从空间域扩展到时空域，同时保证计算效率
    - 核心思路：不同于MAE逐token随机掩码后用masked self-attention，MaskGWM采用行式(row-wise)掩码策略——以空间行为单位遮蔽token。在注意力计算时，使用shifted self-attention而非masked self-attention：被掩码的行与可见行分别计算注意力后通过移位操作交互信息。这种设计避免了masked attention中因大量mask导致的注意力稀疏问题
    - 设计动机：驾驶场景的空间结构呈现明显的行式规律（如地面、车辆、天空从下到上分布），行式掩码比随机掩码更适合捕获这种结构先验。同时shifted attention的计算复杂度远低于全masked attention

3. **行式跨视角模块 (Row-wise Cross-View Module)**:

    - 功能：在多视角生成场景中保持不同摄像头视角间的几何一致性
    - 核心思路：不同视角的相同物理区域往往出现在相近的空间行上。模块在各视角特征的对应行之间建立跨视角注意力，使同一物体在不同摄像头中的表征相互约束。该模块与行式掩码设计自然对齐——被掩码的行在跨视角交互时可以从其他视角的可见行获取补充信息
    - 设计动机：传统驾驶世界模型的多视角一致性通常依赖全局注意力或射线投影，计算开销大且对标定敏感。行式跨视角注意力利用了驾驶场景的结构规律，以更低成本实现视角间对齐

### 损失函数 / 训练策略
总损失为扩散去噪损失 $\mathcal{L}_{diff}$ 与掩码重建损失 $\mathcal{L}_{mask}$ 的加权和：$\mathcal{L} = \mathcal{L}_{diff} + \lambda \mathcal{L}_{mask}$。掩码重建在特征空间进行（而非像素空间），目标是还原被遮蔽token的DiT中间层特征。训练时掩码比例约为50%。MaskGWM-long采用自回归rollout训练，MaskGWM-mview同时处理6个环视摄像头输入。

## 实验关键数据

### 主实验

| 数据集 | 指标 | MaskGWM | 之前SOTA (Vista) | 提升 |
|--------|------|---------|-----------------|------|
| nuScenes | FVD↓ | 89.5 | 122.7 | -27.1% |
| nuScenes | FID↓ | 15.3 | 24.1 | -36.5% |
| OpenDV-2K (长时rollout 15s) | FVD↓ | 178.3 | 265.4 | -32.8% |
| Waymo (零样本) | FVD↓ | 134.2 | 198.6 | -32.4% |
| nuScenes 多视角 | FVD↓ | 95.7 | 142.3 | -32.7% |

### 消融实验

| 配置 | nuScenes FVD↓ | Waymo FVD↓(零样本) | 说明 |
|------|--------------|-------------------|------|
| Full MaskGWM | 89.5 | 134.2 | 完整模型 |
| w/o 掩码重建任务 | 112.8 | 185.3 | 退化为纯DiT，泛化显著下降 |
| w/o 扩散相关mask token | 98.6 | 156.7 | 固定mask token降低生成-重建协同 |
| 随机掩码替代行式掩码 | 103.2 | 162.1 | 行式掩码优势明显 |
| masked attention替代shifted attention | 105.1 | 168.4 | shifted attention更高效有效 |

### 关键发现
- 掩码重建任务对泛化能力的提升最为显著：去掉该任务后Waymo零样本FVD恶化38%，表明特征级上下文学习是跨数据集泛化的关键
- 扩散相关mask token的设计对生成质量贡献重要，使用固定mask token会导致nuScenes FVD增加约10%
- 长时rollout场景（15秒7步自回归）中MaskGWM优势更加突出，说明掩码重建提供的结构理解能力在误差累积时更具鲁棒性
- 行式掩码比随机掩码在驾驶场景中始终表现更好，验证了驾驶场景行式空间结构先验的有效性

## 亮点与洞察
- **扩散+MAE的融合思路**非常巧妙：不是简单叠加两个损失，而是通过扩散感知的mask token和行式掩码设计深度融合了两种学习范式，使生成能力和表征学习能力相互增益
- **行式掩码的领域先验利用**值得关注：驾驶场景的结构特殊性（地面-车辆-天空的垂直分层）被巧妙编码进掩码策略中，这种将领域知识融入预训练任务设计的思路可迁移到其他有明确空间结构的场景
- **从泛化角度论证掩码重建的价值**：大多数MAE工作关注下游微调性能，本文从世界模型的泛化和长时推理角度证明了特征级重建的独特价值

## 局限与展望
- 场景复杂度有限：实验主要在结构化道路场景进行，对于城市密集交通、极端天气等复杂场景的表现未充分验证
- 自回归长时生成仍存在误差累积：尽管比baseline好很多，15秒以上的视频质量仍会明显下降
- 行式掩码策略对驾驶场景有较强假设（行式空间结构），迁移到其他视频生成任务时需要重新设计掩码策略
- 多视角几何一致性的评估较为有限，缺少深度估计等下游任务的验证

## 相关工作与启发
- **vs Vista**: Vista是之前的SOTA驾驶世界模型，基于SVD的fine-tuning方案。MaskGWM改用DiT架构并增加掩码重建，在所有指标上大幅超越Vista，说明DiT+辅助任务的路线优于纯扩散生成路线
- **vs DriveDreamer / GenAD**: 这些方法关注条件控制精度，MaskGWM侧重泛化与长时序，两者关注点互补
- **vs MAE / VideoMAE**: 经典的掩码自编码方法用于表征学习，MaskGWM创新性地将其引入生成模型训练，解决了MAE与扩散模型的兼容性问题

## 评分
- 新颖性: ⭐⭐⭐⭐ 掩码重建与扩散生成的融合思路有创新但各组件较为工程化
- 实验充分度: ⭐⭐⭐⭐ 覆盖三个数据集、两个任务场景、详细消融，但缺少下游规划评估
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机说明充分
- 价值: ⭐⭐⭐⭐ 为驾驶世界模型提供了有效的泛化提升方案，开源代码增加了实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] ReconDreamer: Crafting World Models for Driving Scene Reconstruction via Online Restoration](recondreamer_crafting_world_models_for_driving_scene_reconstruction_via_online_r.md)
- [\[CVPR 2025\] GaussianWorld: Gaussian World Model for Streaming 3D Occupancy Prediction](gaussianworld_gaussian_world_model_for_streaming_3d_occupancy_prediction.md)
- [\[ICCV 2025\] DriveX: Omni Scene Modeling for Learning Generalizable World Knowledge in Autonomous Driving](../../ICCV2025/autonomous_driving/drivex_omni_scene_modeling_for_learning_generalizable_world_knowledge_in_autonom.md)
- [\[ICCV 2025\] Epona: Autoregressive Diffusion World Model for Autonomous Driving](../../ICCV2025/autonomous_driving/epona_autoregressive_diffusion_world_model_for_autonomous_driving.md)
- [\[CVPR 2026\] GenieDrive: Towards Physics-Aware Driving World Model with 4D Occupancy Guided Video Generation](../../CVPR2026/autonomous_driving/geniedrive_towards_physics-aware_driving_world_model_with_4d_occupancy_guided_vi.md)

</div>

<!-- RELATED:END -->
