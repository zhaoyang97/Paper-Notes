---
title: >-
  [论文解读] VPN: Visual Prompt Navigation
description: >-
  [AAAI 2026][3D视觉][视觉导航] 提出视觉提示导航（VPN）新范式：用户在 2D 俯视图上标注视觉轨迹（箭头连接关键路点）来引导智能体导航，替代自然语言指令和图像目标指令，构建了 R2R-VP 和 R2R-CE-VP 两个数据集及 VPNet 基线模型，结合视图级和轨迹级数据增强后在离散和连续环境中均取得优异性能。
tags:
  - "AAAI 2026"
  - "3D视觉"
  - "视觉导航"
  - "视觉提示"
  - "俯视图"
  - "视觉语言导航"
  - "数据增强"
---

# VPN: Visual Prompt Navigation

**会议**: AAAI 2026  
**arXiv**: [2508.01766](https://arxiv.org/abs/2508.01766)  
**代码**: [github.com/farlit/VPN](https://github.com/farlit/VPN)  
**领域**: 3D视觉  
**关键词**: 视觉导航, 视觉提示, 俯视图, 视觉语言导航, 数据增强

## 一句话总结

提出视觉提示导航（VPN）新范式：用户在 2D 俯视图上标注视觉轨迹（箭头连接关键路点）来引导智能体导航，替代自然语言指令和图像目标指令，构建了 R2R-VP 和 R2R-CE-VP 两个数据集及 VPNet 基线模型，结合视图级和轨迹级数据增强后在离散和连续环境中均取得优异性能。

## 研究背景与动机

### 现有导航范式的局限

视觉导航是 AI 和机器人领域的核心研究方向。现有主流范式包括：

**PointGoal 导航**：给出目标相对方向和距离，缺乏中间引导

**ImageGoal 导航**：提供目标位置图像，但缺少中间导航线索

**视觉语言导航（VLN）**：用自然语言描述导航路径，目前最活跃

**自然语言指令的根本困境**：语言在描述物体位置、方向转换和距离关系时天生存在**歧义性**；追求精确描述则不可避免地导致**冗长性**。这形成了人机交互中的两难。

### 视觉提示的优势

作者提出一个直觉洞察：**在地图上画一条路线**是人类最自然的导航指示方式。核心优势：

**高用户可达性**：非专业用户可通过点击或绘制轨迹自然地指定导航目标

**丰富的空间信息**：俯视图天然保留完整空间布局

**高复用性**：俯视图通过无人机航拍或 3D 重建获取，构建一次即可重复使用

## 方法详解

### 整体框架

VPN 的核心工作包括三部分：
1. **数据集构建**：将 R2R/R2R-CE 的语言指令替换为视觉提示
2. **VPNet 模型**：基于 DUET/ETPNav 架构，用 ViT 编码器替换语言编码器
3. **数据增强策略**：视图级增强和轨迹级增强

### 关键设计

#### 1. **视觉提示构建流程**

四步生成：①生成俯视图 ②用箭头连接路点标注轨迹 ③以轨迹为中心裁剪（+60px 边距）④去除黑边紧密包围视觉提示

设计动机：中心裁剪是关键步骤。消融实验表明，不做裁剪时相同场景的不同 episode 共享同一张俯视图，导致过拟合场景而非学习轨迹信息（SR 仅 31%）。

#### 2. **VPNet 模型架构**

三个核心组件：

**ViT 视觉提示编码器**：使用 ViT-B/16（ImageNet-21k 预训练）编码 224x224 的视觉提示图。多楼层场景使用**顺序感知楼层拼接（OAFC）**：
$$\mathcal{P}_i^o = \text{ViT}(\mathcal{P}_i) + b_i, \quad \mathcal{P} = [\mathcal{P}_1^o, ..., \mathcal{P}_k^o]$$

**节点嵌入模块**：智能体增量构建拓扑图，每个节点由全景视图特征（两层 Transformer 编码）、步骤嵌入和位置嵌入组成。

**图感知跨模态编码器**：多层跨模态图 Transformer，包括跨注意力层和图感知自注意力（GASA）层：
$$\text{GASA}(X) = \text{Softmax}\left(\frac{XW_q(XW_k)^T}{\sqrt{d}} + EW_d\right)XW_v$$
其中 $E$ 是拓扑图的成对距离矩阵。

#### 3. **数据增强策略**

**轨迹级增强**：引入 PREVALENT（178k 轨迹）和 ScaleVLN（1.6M 轨迹）增加训练数据多样性。

**视图级增强**：
- **提示视图增强**：随机旋转俯视图（0°/90°/180°/270°）
- **智能体视图增强**：随机采样初始朝向

设计动机：VPN 中初始朝向与视觉提示无关（不像 VLN 中语言可能隐含初始方向），因此可自由旋转。

### 训练策略

- **行为克隆 + DAgger**：$\mathcal{L} = \lambda \mathcal{L}_{BC} + (1-\lambda) \mathcal{L}_{DAG}$，$\lambda = 0.5$
- 离散环境：单卡 A5000，400k 迭代，batch=10，lr=1.5e-5
- 连续环境：双卡 A5000，400k 迭代，batch=16，lr=1e-5

## 实验关键数据

### 主实验

**离散环境（R2R-VP）**：

| 方法 | 训练数据 | Val Unseen SR↑ | Val Unseen SPL↑ | Test Unseen SR↑ |
|------|---------|----------------|-----------------|-----------------|
| DUET (VLN) | R2R+PRE+SCA | 81 | 70 | 80 |
| VPNet | R2R | 51.23 | 43.47 | 52.40 |
| VPNet | R2R+PRE | 65.92 | 56.17 | 66.38 |
| **VPNet** | **R2R+PRE+SCA** | **96.68** | **94.84** | **97.56** |

VPN 在 val unseen 达 96.68% SR，远超 DUET 的 81%，且仅用 1/3 的 ScaleVLN 轨迹。

**连续环境（R2R-CE-VP）**：

| 方法 | 设置 | Val Seen SR↑ | Val Unseen SR↑ |
|------|------|-------------|----------------|
| ETPNav (VLN) | R2R+PRE | 66 | 57 |
| VPNet | R2R+PRE | **84.11** | 47.96 |

### 消融实验

**不同视觉提示类型的影响（离散环境）**：

| 提示类型 | Val Seen SR | Val Unseen SR | 说明 |
|----------|-------------|---------------|------|
| 未裁剪全俯视图 | 31.68 | 33.94 | 过拟合场景 |
| 仅裁剪俯视图 | 83.56 | 45.83 | 类似 ImageNav |
| 裁剪图+箭头+文字 | 95.74 | 65.36 | 文字遮挡细节 |
| **裁剪图+箭头** | **100** | **65.92** | 最优 |

**视图级增强的影响**：

| 增强方式 | Val Unseen SR↑ | SPL↑ |
|----------|----------------|------|
| 无增强 | 86.33 | 82.92 |
| 仅智能体视图 | 88.18 | 85.02 |
| 仅提示视图旋转 | 96.41 | 94.37 |
| **两者结合** | **96.68** | **94.84** |

提示视图旋转比智能体视图增强效果大得多（+10 SR vs +2 SR）。

### 关键发现

1. **视觉提示的数据效率惊人**：VPNet 用 1.6M 轨迹达 96.68% SR，而 DUET 用 4.9M 轨迹仅 81% SR
2. **裁剪操作至关重要**：不裁剪时严重过拟合场景（31% vs 100% Val Seen）
3. **仅裁剪无轨迹也有效**：模型能从裁剪区域推断大致目的地（类似 ImageNav）
4. **对噪声有一定鲁棒性**：20% 椒盐噪声下 SR 从 96.68% 降至 90.34%

## 亮点与洞察

1. **范式创新**：VPN 是视觉导航的全新范式，填补了语言导航和图像目标导航之间的空白
2. **数据效率优势**：视觉提示包含的空间信息密度远高于语言指令
3. **实用性强**：俯视图可通过无人机或 3D 重建一次获取、多次复用
4. **消融实验设计出色**：系统分析了提示类型、增强策略、编码器设置等多个维度

## 局限与展望

1. **仅在仿真环境中验证**：MP3D/HM3D 场景，未在真实世界测试
2. **依赖高质量俯视图**：某些场景重建质量差无法使用
3. **连续环境性能差距**：Val Unseen SR 仅 47.96%，远低于离散环境的 96.68%
4. **多楼层场景处理粗糙**：简单拼接各楼层特征

## 相关工作与启发

- **VLN (R2R) 系列**：DUET、BEVBert、ScaleVLN 等是主要对比基线
- **DUET**：VPNet 离散版的架构基础
- **ETPNav**：VPNet 连续版的架构基础
- **RoVI**：用手绘符号指导机器人操作，但 VPN 是首个以视觉提示作为唯一导航指令的工作

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 新导航范式，填补重要空白
- **实验充分度**: ⭐⭐⭐⭐ — 消融全面，缺少真实世界实验
- **写作质量**: ⭐⭐⭐⭐ — 动机清晰，图示直观
- **实用价值**: ⭐⭐⭐⭐ — 交互方式友好，连续环境性能有待提升

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] FALIP: Visual Prompt as Foveal Attention Boosts CLIP Zero-Shot Performance](../../ECCV2024/3d_vision/falip_visual_prompt_as_foveal_attention_boosts_clip_zero-shot_performance.md)
- [\[CVPR 2026\] EfficientVPR: Toward Efficient Visual Place Recognition via Scene-Aware Prompt Tuning and Adaptive Feature Enhancement](../../CVPR2026/3d_vision/efficientvpr_toward_efficient_visual_place_recognition_via_scene-aware_prompt_tu.md)
- [\[CVPR 2025\] Vid2Sim: Realistic and Interactive Simulation from Video for Urban Navigation](../../CVPR2025/3d_vision/vid2sim_realistic_and_interactive_simulation_from_video_for_urban_navigation.md)
- [\[ICLR 2026\] OpenFly: A Comprehensive Platform for Aerial Vision-Language Navigation](../../ICLR2026/3d_vision/openfly_a_comprehensive_platform_for_aerial_vision-language_navigation.md)
- [\[CVPR 2025\] RoomTour3D: Geometry-Aware Video-Instruction Tuning for Embodied Navigation](../../CVPR2025/3d_vision/roomtour3d_geometry-aware_video-instruction_tuning_for_embodied_navigation.md)

</div>

<!-- RELATED:END -->
