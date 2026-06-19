---
title: >-
  [论文解读] OCK: Unsupervised Dynamic Video Prediction with Object-Centric Kinematics
description: >-
  [ICCV 2025][视频生成][对象中心学习] 提出 OCK（Object-Centric Kinematics），在以对象为中心的视频预测中引入显式的运动学属性（位置、速度、加速度）作为 Slot 表示的补充，通过 Joint-OCK 和 Cross-OCK 两种 Transformer 变体融合外观与运动信息，在复杂合成和真实场景中显著提升动态视频预测质量。
tags:
  - "ICCV 2025"
  - "视频生成"
  - "对象中心学习"
  - "视频预测"
  - "运动学建模"
  - "注意力机制"
  - "Transformer"
---

# OCK: Unsupervised Dynamic Video Prediction with Object-Centric Kinematics

**会议**: ICCV 2025  
**arXiv**: [2404.18423](https://arxiv.org/abs/2404.18423)  
**代码**: 无  
**领域**: 视频生成  
**关键词**: 对象中心学习, 视频预测, 运动学建模, Slot Attention, 自回归 Transformer

## 一句话总结

提出 OCK（Object-Centric Kinematics），在以对象为中心的视频预测中引入显式的运动学属性（位置、速度、加速度）作为 Slot 表示的补充，通过 Joint-OCK 和 Cross-OCK 两种 Transformer 变体融合外观与运动信息，在复杂合成和真实场景中显著提升动态视频预测质量。

## 研究背景与动机

人类感知将复杂多物体场景分解为**时间不变的外观**（大小、形状、颜色）和**时间变化的运动**（位置、速度、加速度）。基于对象中心的 Transformer 视频预测方法（如 SlotFormer、OCVP）主要依赖 Slot Attention 提取的对象外观表示，存在以下问题：

**忽略显式运动动力学**：仅隐式学习运动变化，难以准确建模动态交互（如碰撞、加减速）

**复杂场景中表现不佳**：在包含多样物体外观、运动模式和背景的场景中（如 MOVi-C/D/E），现有方法预测质量下降甚至发散

**长期预测泛化差**：缺乏显式的运动学先验导致误差快速累积

## 方法详解

### 整体框架

OCK 由三个主要模块组成：
1. **Slot 编码器**：预训练的 SAVi 模型，将视频帧分解为对象 slot $\mathcal{S}_t \in \mathbb{R}^{N \times D_{\text{slot}}}$
2. **运动学编码器**：从视频帧中提取对象运动学 $\mathbf{K}_t \in \mathbb{R}^{N \times D_{\text{kin}}}$
3. **自回归 OCK Transformer**：融合 slot 和运动学信息，预测下一时间步的 slot

### 关键设计

1. **Object Kinematics（对象运动学）**：使用 CNN 提取低层图像特征后定位每个物体质心的 2D 坐标，构建三层运动学状态：
    $\mathbf{K}_t = \begin{bmatrix} \mathbf{x}_t^{\text{pos}} \\ \mathbf{x}_t^{\text{vel}} \\ \mathbf{x}_t^{\text{acc}} \end{bmatrix} = \begin{bmatrix} \phi(\mathbf{o}_t) \\ \lambda(\mathbf{x}_t^{\text{pos}} - \mathbf{x}_{t-1}^{\text{pos}}) \\ \mathbf{x}_t^{\text{vel}} - \mathbf{x}_{t-1}^{\text{vel}} \end{bmatrix}$
   其中 $\lambda$ 为可学习缩放参数。运动学在 2D 图像空间建模（避免 3D 深度估计的计算开销），且不依赖任务特定的损失函数。

2. **两种运动学使用方式**：

    - **分析方法（Analytical）**：根据当前运动学预测下一帧的位置 $\mathbf{x}_{t+1}^{\text{pos}'} = \mathbf{x}_t^{\text{pos}} + \mathbf{x}_t^{\text{vel}} \times \delta$，然后将当前和预测运动学一起送入 Transformer
    - **经验方法（Empirical）**：仅使用当前帧运动学，让 Transformer 隐式学习运动模式

3. **两种 OCK Transformer 架构**：

    - **Joint-OCK**：将 slot 和运动学拼接后联合输入标准 Transformer 编码器进行自注意力
    - **Cross-OCK**：使用交叉注意力机制，slot 作为 query，运动学作为 key/value，并引入温度参数 $\tau$ 调节注意力校准：$\text{Cross-OCK}(\mathbf{v}, \mathbf{k}, \mathbf{q}; \tau) = \mathbf{v} \cdot \text{softmax}(\frac{\mathbf{k}^\top \mathbf{q}}{\tau})$

### 损失函数 / 训练策略

两阶段训练：先训练 SAVi 分解视频帧为 slot，再训练 OCK Transformer。

总损失 $\mathcal{L} = \mathcal{L}_{\text{object}} + \alpha \mathcal{L}_{\text{image}}$：
- **对象重建损失**：预测 slot 与 GT slot 的 L2 距离
- **图像重建损失**：通过冻结的 SAVi 解码器将预测 slot 解码为图像，与 GT 帧的 L2 距离

训练设置为 6 帧输入预测 8 帧，使用时序位置编码保持对象间的置换等变性。

## 实验关键数据

### 主实验 (表格)

6 个合成数据集上的视频预测质量（从简单到复杂）：

| 模型 | OBJ3D PSNR↑ | MOVi-A PSNR↑ | MOVi-C PSNR↑ | MOVi-D PSNR↑ | MOVi-E PSNR↑ |
|------|-------------|-------------|-------------|-------------|-------------|
| SlotFormer | 33.08 | 25.18 | 19.48 | 20.68 | 21.27 |
| OCVP-Seq | 33.10 | 26.24 | 17.95 | 发散 | 发散 |
| Joint-OCK | **35.13** | 27.26 | **21.04** | **22.09** | **22.39** |
| Cross-OCK | 34.10 | **27.58** | 21.04 | 22.34 | 22.34 |

真实场景 Waymo Open Dataset：

| 模型 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| SlotFormer | 19.13 | 0.330 | 0.714 |
| OCVP-Seq | 18.98 | 0.329 | 0.718 |
| Joint-OCK | 25.02 | 0.798 | 0.251 |
| Cross-OCK | **25.98** | **0.728** | **0.220** |

### 消融实验 (表格)

Transformer 组件消融（MOVi-A）：

| 设置 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| Cross-OCK(A) 默认 | **27.58** | **0.812** | **0.123** |
| 输入帧=4 | 27.01 | 0.801 | 0.125 |
| 输入帧=8 | 27.12 | 0.806 | 0.125 |
| Transformer 层=6 | 26.92 | 0.796 | 0.130 |
| Transformer 层=8 | 26.50 | 0.784 | 0.133 |
| 普通位置编码 | 23.60 | 0.591 | 0.205 |
| Teacher Forcing | 23.58 | 0.589 | 0.207 |

### 关键发现

- **运动学的引入对复杂场景至关重要**：在 MOVi-D/E 上 OCVP 完全发散，而 OCK 仍能正常预测
- Waymo 真实场景上 OCK 比 SlotFormer PSNR 提升 ~6.9dB、LPIPS 降低 ~0.49
- **时序位置编码**（保持置换等变性）极为关键，使用普通位置编码 PSNR 下降 4dB
- **Teacher Forcing 有害**：让模型在训练时学会处理自身的不完美预测对长期泛化更重要
- 分析方法略优于经验方法，因为显式预测下一帧的运动状态提供了更准确的引导
- 6 帧输入即足以捕捉对象动力学，过多输入（8帧）反而略有下降

## 亮点与洞察

- **物理学启发**：将经典运动学（位置-速度-加速度）引入对象中心学习，理念直观且有效
- **Cross-OCK 的设计精巧**：slot 作为 query、运动学作为 key/value 的交叉注意力，在计算效率和性能间取得了良好平衡
- **长期泛化能力**：仅用 6 帧训练，可泛化到 18 帧预测且误差增长缓慢
- 在真实驾驶场景（Waymo）上也展现出强大能力

## 局限与展望

- 运动学仅在 2D 图像空间建模，对 3D 遮挡和深度变化的处理有限
- 依赖 SAVi 预训练的 slot 编码器质量，复杂场景的 slot 分解可能不完美
- 未涉及旋转和缩放运动学，限制了对复杂运动的建模
- 对物体数量变化（出现/消失）的处理未讨论
- 可考虑引入物体间交互图（GNN）来显式建模碰撞等事件

## 相关工作与启发

- SlotFormer 是最直接的对比方法，OCK 在其基础上添加运动学维度
- OCVP 探索了时间和关系注意力的分离，但在复杂场景发散
- G-SWM 使用图神经网络建模交互，但性能不如 Transformer 方法
- 对物理模拟器学习、机器人视觉理解有潜在启发

## 评分

- 新颖性: ⭐⭐⭐⭐ 将经典运动学概念引入对象中心学习是自然但有效的创新
- 实验充分度: ⭐⭐⭐⭐ 7 个数据集（含真实场景），详尽的消融分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，两种方法（分析/经验）和两种架构（Joint/Cross）的对比系统
- 价值: ⭐⭐⭐⭐ 解决了对象中心视频预测在复杂场景中的关键瓶颈

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] DreamRelation: Relation-Centric Video Customization](dreamrelation_relation-centric_video_customization.md)
- [\[CVPR 2026\] Ego-InBetween: Generating Object State Transitions in Ego-Centric Videos](../../CVPR2026/video_generation/ego-inbetween_generating_object_state_transitions_in_ego-centric_videos.md)
- [\[CVPR 2025\] Articulated Kinematics Distillation from Video Diffusion Models](../../CVPR2025/video_generation/articulated_kinematics_distillation_from_video_diffusion_models.md)
- [\[ICCV 2025\] FuXi-RTM: A Physics-Guided Prediction Framework with Radiative Transfer Modeling](fuxi-rtm_a_physics-guided_prediction_framework_with_radiative_transfer_modeling.md)
- [\[CVPR 2025\] Unified Dense Prediction of Video Diffusion](../../CVPR2025/video_generation/unified_dense_prediction_of_video_diffusion.md)

</div>

<!-- RELATED:END -->
