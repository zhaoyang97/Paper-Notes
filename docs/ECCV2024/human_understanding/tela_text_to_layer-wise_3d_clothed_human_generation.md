---
title: >-
  [论文解读] TELA: Text to Layer-wise 3D Clothed Human Generation
description: >-
  [ECCV 2024][人体理解][3D人体生成] TELA提出了分层的3D穿衣人体表示方法和渐进优化策略，从文本描述生成服装可解耦的3D人体模型，支持逐层穿衣生成和虚拟试衣等编辑应用。 领域现状：文本到3D穿衣人体生成是3D内容创建的重要方向。现有方法（如DreamAvatar、AvatarCLIP等）通常将人体和服装编码…
tags:
  - "ECCV 2024"
  - "人体理解"
  - "3D人体生成"
  - "服装生成"
  - "分层表示"
  - "文本到3D"
  - "虚拟试衣"
---

# TELA: Text to Layer-wise 3D Clothed Human Generation

**会议**: ECCV 2024  
**arXiv**: [2404.16748](https://arxiv.org/abs/2404.16748)  
**代码**: [http://jtdong.com/tela_layer/](http://jtdong.com/tela_layer/)  
**领域**: 人体理解 / 3D生成  
**关键词**: 3D人体生成, 服装生成, 分层表示, 文本到3D, 虚拟试衣

## 一句话总结
TELA提出了分层的3D穿衣人体表示方法和渐进优化策略，从文本描述生成服装可解耦的3D人体模型，支持逐层穿衣生成和虚拟试衣等编辑应用。

## 研究背景与动机

**领域现状**：文本到3D穿衣人体生成是3D内容创建的重要方向。现有方法（如DreamAvatar、AvatarCLIP等）通常将人体和服装编码为一个整体模型，在单阶段优化中生成完整的穿衣人体。

**现有痛点**：（1）整体生成方式无法分离人体和服装——不支持换装、虚拟试衣等编辑操作；（2）单阶段优化对整个生成过程缺乏细粒度控制，容易出现身体-服装的几何耦合问题；（3）不同服装层（如内衣、外套）之间的遮挡关系难以正确建模。

**核心矛盾**：要同时实现高质量生成和服装可编辑性，需要在表示层面就将人体和服装解耦，但解耦会增加生成的难度。

**本文目标**：设计分层的穿衣人体表示和优化策略，生成服装解耦的高质量3D人体模型。

**切入角度**：将穿衣人体分解为最小穿衣的人体层和逐层叠加的服装层，采用渐进式从内到外的生成策略。

**核心 idea**：先生成最小穿衣人体，再逐层添加服装，通过分层组合渲染和解耦损失确保各层之间的正确几何关系。

## 方法详解

### 整体框架
输入文本描述（如"一个穿蓝色连衣裙的女性"），首先生成最小穿衣的人体模型（SMPL-based），然后逐层生成服装——每层衣物作为独立的隐式/显式表面模型。渐进优化策略确保每层在前层基础上正确叠加。

### 关键设计

1. **分层穿衣人体表示**:

    - 功能：将人体和各层服装表示为独立的可编辑模型
    - 核心思路：人体层使用SMPL-based表示，每个服装层用独立的NeRF/SDF表示。各层有自己的几何和外观参数，可以独立编辑或替换。服装层附着于人体层之上，遵循从内到外的物理遮挡关系
    - 设计动机：分层表示是实现服装可编辑性的前提——只有解耦表示才能支持换装和虚拟试衣

2. **分层组合渲染（Stratified Compositional Rendering）**:

    - 功能：将多层模型融合为最终图像进行SDS优化
    - 核心思路：在渲染时按从内到外的顺序组合各层——外层服装遮挡内层人体。使用alpha compositing将各层的颜色和密度组合，确保物理正确的遮挡关系。这个渲染过程是可微的，允许梯度反向传播到各层
    - 设计动机：标准的单层渲染无法处理多层几何的遮挡，分层组合渲染解决了这个问题

3. **服装-人体解耦损失**:

    - 功能：防止服装层和人体层的几何纠缠
    - 核心思路：设计正则化损失确保服装层只在人体表面附近有密度（不渗透到人体内部），同时在服装覆盖区域抑制人体层的外观对最终渲染的贡献。这个损失鼓励"服装在外、人体在内"的正确层次关系
    - 设计动机：没有显式约束的优化容易产生服装-人体的几何融合，解耦损失是保证分层质量的关键

### 损失函数 / 训练策略
使用SDS (Score Distillation Sampling) 损失驱动文本引导的3D生成，加上服装-人体解耦正则化损失。渐进式优化：先优化人体层，再逐层添加和优化服装层。

## 实验关键数据

### 主实验

| 方法 | 生成质量 | 解耦性 | 编辑能力 |
|------|---------|--------|---------|
| TELA | 高 | 强 | 支持换装 |
| DreamAvatar | 高 | 无 | 不支持 |
| AvatarCLIP | 中 | 无 | 不支持 |
| TADA | 高 | 弱 | 有限 |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| Full TELA | 最佳 | 分层+解耦+渐进 |
| 无分层渲染 | 几何糟糕 | 遮挡关系错误 |
| 无解耦损失 | 耦合严重 | 服装人体混合 |
| 同时优化(非渐进) | 质量下降 | 优化不稳定 |

### 关键发现
- 分层表示是服装编辑能力的关键——现有方法因整体表示而完全无法编辑
- 解耦损失对防止几何纠缠至关重要——去掉后服装和人体严重融合
- 渐进式优化比同时优化更稳定，因为内层为外层提供了良好的初始化

## 亮点与洞察
- **从表示出发的问题解决**：不是在生成方法上做改进，而是重新设计表示方式使得编辑成为可能——"right representation makes the right capability"
- **渐进式从内到外的优化**：模拟了真实穿衣过程（先身体再穿衣），这种物理直觉指导的优化策略既自然又有效
- **虚拟试衣的实际应用**：服装解耦后可以直接做虚拟试衣——将A的衣服穿到B身上，有直接的商业价值

## 局限与展望
- 渐进式多层优化速度较慢，每增加一层服装都需要重新优化
- 对于复杂的服装几何（如飘带、褶皱细节），表示能力可能不足
- 服装物理属性（如布料垂坠、弹性）未建模
- 仅支持文本输入，不支持参考图像输入的服装迁移

## 相关工作与启发
- **vs DreamAvatar**：DreamAvatar生成整体人体，TELA分层生成支持编辑
- **vs TADA**：TADA有一定的解耦能力但不够彻底，TELA的分层表示更显式
- **vs AvatarCraft**：AvatarCraft关注动画而非服装编辑，目标不同

## 评分
- 新颖性: ⭐⭐⭐⭐ 分层表示和渐进优化的组合新颖
- 实验充分度: ⭐⭐⭐⭐ 与多个基线对比+消融实验+编辑应用展示
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰系统
- 价值: ⭐⭐⭐⭐ 对3D人体生成和虚拟试衣有实际推动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] FreeCloth: Free-Form Generation Enhances Challenging Clothed Human Modeling](../../CVPR2025/human_understanding/freecloth_free-form_generation_enhances_challenging_clothed_human_modeling.md)
- [\[ECCV 2024\] 3DSA: Multi-view 3D Human Pose Estimation With 3D Space Attention Mechanisms](3dsa_multi-view_3d_human_pose_estimation_with_3d_space_attention_mechanisms.md)
- [\[ECCV 2024\] WorldPose: A World Cup Dataset for Global 3D Human Pose Estimation](worldpose_a_world_cup_dataset_for_global_3d_human_pose_estimation.md)
- [\[ECCV 2024\] 3DFG-PIFu: 3D Feature Grids for Human Digitization from Sparse Views](3dfg-pifu_3d_feature_grids_for_human_digitization_from_sparse_views.md)
- [\[ECCV 2024\] Occlusion Handling in 3D Human Pose Estimation with Perturbed Positional Encoding](occlusion_handling_in_3d_human_pose_estimation_with_perturbed_positional_encodin.md)

</div>

<!-- RELATED:END -->
