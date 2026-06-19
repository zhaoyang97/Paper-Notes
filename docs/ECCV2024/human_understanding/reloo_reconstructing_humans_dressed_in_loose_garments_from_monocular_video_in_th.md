---
title: >-
  [论文解读] ReLoo: Reconstructing Humans Dressed in Loose Garments from Monocular Video in the Wild
description: >-
  [ECCV 2024][人体理解][人体重建] 提出 ReLoo，通过分层神经人体表示和非层级虚拟骨骼变形模块，从单目野外视频中重建穿着宽松服装的高质量3D人体模型。 核心矛盾 核心矛盾：领域现状：现有单目人体重建方法主要关注紧身服装，在宽松服装（如裙子、宽松上衣）上表现不佳。核心原因是这些方法将人体和服装建模为单一实体…
tags:
  - "ECCV 2024"
  - "人体理解"
  - "人体重建"
  - "宽松服装"
  - "虚拟骨骼"
  - "分层神经表示"
  - "单目视频"
---

# ReLoo: Reconstructing Humans Dressed in Loose Garments from Monocular Video in the Wild

**会议**: ECCV 2024  
**arXiv**: [2409.15269](https://arxiv.org/abs/2409.15269)  
**代码**: [有](https://moygcc.github.io/ReLoo/)  
**领域**: 人体理解  
**关键词**: 人体重建, 宽松服装, 虚拟骨骼, 分层神经表示, 单目视频

## 一句话总结

提出 ReLoo，通过分层神经人体表示和非层级虚拟骨骼变形模块，从单目野外视频中重建穿着宽松服装的高质量3D人体模型。

## 研究背景与动机

### 核心矛盾

**核心矛盾**：**领域现状**：现有单目人体重建方法主要关注紧身服装，在宽松服装（如裙子、宽松上衣）上表现不佳。核心原因是这些方法将人体和服装建模为单一实体，仅依赖骨骼变形驱动，无法处理与身体姿态弱相关的宽松服装大幅非刚性形变。模板方法依赖预扫描模板，限制了部署到新主体；NeRF方法缺乏对服装独立运动的建模能力。

## 方法详解

### 整体框架

1. 建立分层神经隐式表示，将人体分解为内层身体和外层服装
2. 对服装层引入非层级虚拟骨骼变形模块
3. 通过多层可微体渲染联合优化形状、外观和变形

### 关键设计

**分层神经表示**：身体层 f^B 和服装层 f^G 分别用神经网络建模SDF和辐射值，通过取最小值组合得到完整穿衣人体。

**虚拟骨骼变形**：定义 n_v 个非层级虚拟骨骼，通过MLP从骨骼位置、姿态和时间嵌入预测其刚性变换。虚拟骨骼不受层级约束可自由运动，适合捕捉高动态宽松服装的变形。skinning权重基于距离计算。

**多层体渲染**：沿射线在身体层和服装层分别采样点，按深度排序后进行体积积分。每层独立处理遮挡，正确处理服装-身体之间的遮挡关系。

**两阶段训练**：第一阶段用骨骼变形驱动两层并热身虚拟骨骼变形场；第二阶段激活虚拟骨骼变形模块，从服装网格中提取虚拟骨骼位置。

### 损失函数

包含重建损失 L_rgb、分割损失 L_seg（使用SAM获取mask）、自适应Eikonal损失 L_eikonal 和虚拟骨骼正则化损失 L_reg。

## 实验关键数据

### 主实验

**MonoLoose数据集表面重建**：

| 方法 | Chamfer-L2↓ | NC↑ | V-IoU↑ |
|------|-------------|-----|--------|
| SelfRecon | 2.22 | 0.788 | 0.844 |
| Vid2Avatar | 2.34 | 0.794 | 0.776 |
| SCARF | 3.13 | 0.711 | 0.691 |
| **ReLoo** | **1.93** | **0.831** | **0.881** |

**新视角合成（MonoLoose / DynaCap）**：

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|-------|-------|--------|
| SelfRecon | 22.5 | 0.953 | 6.08 | 26.8 | 0.982 | 1.56 |
| Vid2Avatar | 25.9 | 0.968 | 4.66 | 27.1 | 0.983 | 1.82 |
| SCARF | 23.3 | 0.953 | 6.59 | 25.5 | 0.979 | 2.55 |
| **ReLoo** | **29.2** | **0.970** | **3.15** | **27.9** | **0.985** | **1.27** |

### 消融实验

**虚拟骨骼数量对性能的影响**：

| 虚拟骨骼数 n_v | LPIPS↓ | 每迭代时间 |
|----------------|--------|----------|
| 20 | ~4.2 | 较快 |
| 40 | ~3.8 | 中等 |
| **80** | **~3.2** | **中等** |
| 160 | ~3.1 | 较慢 |
| 320 | ~3.0 | 很慢 |

- 去除虚拟骨骼：MonoLoose PSNR从29.2降至28.7，DynaCap从27.9降至27.3
- 去除多轮采样：V-IoU从0.881降至0.879，但会导致服装-身体穿透问题
- 虚拟骨骼数量实验：n_v=80为最佳性能-效率折衷点，LPIPS在此处有最大幅度下降

### 关键发现

- SMPL骨骼变形无法处理远离身体的宽松服装，虚拟骨骼模块至关重要
- 分层表示可捕捉服装拓扑变化（如裙子腿间的空隙）
- 在野外视频上具有鲁棒性

## 亮点与洞察

1. 虚拟骨骼是非层级的，不受解剖结构约束，可以自由捕捉服装动态
2. 多层体渲染正确处理了身体-服装之间的遮挡关系
3. 不需要任何3D监督或服装模板先验
4. 使用SAM进行分割提供弱监督

## 局限与展望

- 依赖合理的姿态估计和分割mask，有时需要手动调整SAM结果
- 主要支持最多两件服装的重建
- 复杂度随服装数量线性增长

## 相关工作与启发

- SCARF基于SMPL-X用NeRF重建外层服装，但限于自转运动
- Pan et al. 使用虚拟骨骼进行服装动画，但需要已知模板和3D仿真数据
- 启发：分层表示 + 自由变形模块 是处理复杂服装动态的有效范式

## 评分

- 创新性：★★★★☆ 虚拟骨骼变形模块和分层隐式表示设计巧妙
- 实用性：★★★★☆ 从单目视频重建宽松服装人体，应用前景广
- 实验质量：★★★★★ 提出新数据集MonoLoose，实验对比全面

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Monocular Facial Appearance Capture in the Wild](../../ICCV2025/human_understanding/monocular_facial_appearance_capture_in_the_wild.md)
- [\[CVPR 2025\] D3-Human: Dynamic Disentangled Digital Human from Monocular Video](../../CVPR2025/human_understanding/d3-human_dynamic_disentangled_digital_human_from_monocular_video.md)
- [\[ECCV 2024\] How Video Meetings Change Your Expression](how_video_meetings_change_your_expression.md)
- [\[CVPR 2025\] FATE: Full-head Gaussian Avatar with Textural Editing from Monocular Video](../../CVPR2025/human_understanding/fate_full-head_gaussian_avatar_with_textural_editing_from_monocular_video.md)
- [\[CVPR 2025\] Homogeneous Dynamics Space for Heterogeneous Humans](../../CVPR2025/human_understanding/homogeneous_dynamics_space_for_heterogeneous_humans.md)

</div>

<!-- RELATED:END -->
