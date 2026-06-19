---
title: >-
  [论文解读] MotionDiff: Training-Free Zero-Shot Interactive Motion Editing via Flow-Assisted Multi-View Diffusion
description: >-
  [ICCV 2025][图像生成][运动编辑] MotionDiff 提出一种免训练、零样本的多视图运动编辑方法，通过点运动学模型（PKM）从静态场景估计多视图光流，再利用解耦运动表示引导 Stable Diffusion 生成高质量、多视图一致的运动编辑结果。 现有痛点 现有痛点：领域现状：利用生成模型进行可控编辑是当前的…
tags:
  - "ICCV 2025"
  - "图像生成"
  - "运动编辑"
  - "光流引导"
  - "多视图一致性"
  - "免训练扩散"
  - "3D点云"
---

# MotionDiff: Training-Free Zero-Shot Interactive Motion Editing via Flow-Assisted Multi-View Diffusion

**会议**: ICCV 2025  
**arXiv**: [2503.17695](https://arxiv.org/abs/2503.17695)  
**代码**: 无 (无)  
**领域**: 图像编辑/运动编辑  
**关键词**: 运动编辑, 光流引导, 多视图一致性, 免训练扩散, 3D点云

## 一句话总结

MotionDiff 提出一种免训练、零样本的多视图运动编辑方法，通过点运动学模型（PKM）从静态场景估计多视图光流，再利用解耦运动表示引导 Stable Diffusion 生成高质量、多视图一致的运动编辑结果。

## 研究背景与动机

### 现有痛点

**现有痛点**：**领域现状**：利用生成模型进行可控编辑是当前的热门问题，但在**运动编辑**领域仍面临三大挑战：

**复杂运动处理困难**：现有基于物理的编辑方法（如 DragGAN、DragDiffusion）主要处理简单的拖拽/平移运动，难以处理旋转、缩放、拉伸等复杂运动

**多视图一致性缺失**：大多数方法只关注单视图编辑，缺乏跨视图的运动约束，导致不同视角的编辑结果不一致

**需要重训练或特定表示**：很多方法需要重训练扩散模型或依赖 NeRF/3DGS 等特定场表示，增加了计算成本和数据收集开销

作者的核心洞察：光流是表达像素级运动的天然载体。通过在 3D 空间中建模运动（将用户定义的运动模式作用于 3D 点云），可以自然地获得多视图一致的光流，进而引导扩散模型实现精确的运动编辑。

### 解决思路

**本文目标**：### 整体框架

MotionDiff 包含两个推理阶段：

1. **多视图光流估计阶段（MFES）**：用户从静态场景中选择感兴趣的物体并定义运动模式，通过 PKM 估计多视图光流
2. **多视图运动扩散阶段（MMDS）**：利用估计的光流引导 Stable Diffusion 进行运动编辑，通过解耦运动表示保证多视图一致性

### 关键设计

**1. 点运动学模型（PKM）**

P。


## 方法详解

### 整体框架

MotionDiff 包含两个推理阶段：

1. **多视图光流估计阶段（MFES）**：用户从静态场景中选择感兴趣的物体并定义运动模式，通过 PKM 估计多视图光流
2. **多视图运动扩散阶段（MMDS）**：利用估计的光流引导 Stable Diffusion 进行运动编辑，通过解耦运动表示保证多视图一致性

### 关键设计

**1. 点运动学模型（PKM）**

PKM 是本文的核心创新，用于从用户定义的单视图运动先验估计 3D 点运动后的位置 P_m。基本思路：

- 利用 Mask Clustering 分割 3D 点云，用户交互选择感兴趣的物体 P_o
- 设计 GUI 让用户在选定视角上定义单视图光流 f_s
- 将 f_s 反投影到 3D 空间得到稀疏运动点 P_sm
- 通过 PKM 为不同运动模式估计完整的 P_m，再投影回各视图得到多视图光流 f_m

PKM 支持四种运动模式：

- **平移**：从稀疏点对 (P_so, P_sm) 估计 3D 偏移量 p_off，P_m = P_o + p_off
- **缩放**：根据光流范数比计算缩放因子 s_f，P_m = s_f · P_o
- **旋转**：通过 GUI 获取旋转角 φ，构建旋转矩阵 Rot，P_m = Rot ⊙ (P_o - p_c) + p_c
- **拉伸**：确定拉伸平面，根据点到平面距离计算不同程度的拉伸因子

**2. 解耦运动表示（MMDS 核心）**

MMDS 的关键思想是将运动图像解耦为三部分：

- **静态背景**：运动物体移走后暴露的背景区域
- **运动物体**：光流作用后的前景物体
- **遮挡区域**：运动产生的新遮挡/暴露区域

针对每个部分设计不同的引导策略：
- 背景区域直接从原始图像复制
- 运动物体通过光流 warp 引导扩散模型生成
- 遮挡区域由扩散模型自由补全

**3. 多视图一致性保证**

由于所有视图的光流都源自同一个 3D 点云的运动（PKM），天然保证了多视图之间的几何一致性。无需额外的跨视图约束或重训练。

### 损失函数 / 训练策略

MotionDiff 是**完全免训练**的推理方法。引导策略基于 classifier guidance 范式：

ε̃_θ(x_t; t; y) = ε_θ(x_t; t; y) + σ_t · ∇_{x_t} L(x_t)

其中 L(x_t) 包含光流引导损失和区域解耦损失。DDIM 采样时使用固定的时间步调度。

## 实验关键数据

### 主实验

**与其他运动编辑方法的定性比较**（论文中主要以可视化展示）：

| 方法 | 平移 | 旋转 | 缩放 | 拉伸 | 多视图一致 | 需训练 |
|------|:-:|:-:|:-:|:-:|:-:|:-:|
| DragDiffusion | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Motion Guidance | ✓ | △ | △ | ✗ | ✗ | ✗ |
| DragonDiffusion | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **MotionDiff** | **✓** | **✓** | **✓** | **✓** | **✓** | **✗** |

**多视图编辑一致性评估**：

| 方法 | 多视图一致性 | 纹理保持 | 运动复杂度 |
|------|:-:|:-:|:-:|
| Motion Guidance | 低 | 低 | 平移 |
| DragDiffusion | 低 | 中 | 平移/拖拽 |
| **MotionDiff** | **高** | **高** | **多种模式** |

### 消融实验

**解耦运动表示的消融**：

| 设置 | 背景保持 | 运动准确性 | 遮挡补全 |
|------|:-:|:-:|:-:|
| 无解耦（全局引导） | 差 | 中 | 差 |
| 仅背景+运动 | 好 | 好 | 差 |
| **完整解耦** | **好** | **好** | **好** |

完整的三部分解耦（背景 + 运动物体 + 遮挡区域）对最终质量至关重要。

### 关键发现

1. **PKM 是实现多视图一致运动编辑的关键**：通过 3D 空间建模运动，本质上解决了多视图一致性问题
2. **光流比拖拽点更适合表达复杂运动**：光流可以自然表达旋转、缩放等复杂变换
3. **解耦表示显著提升质量**：分别处理背景、运动物体和遮挡区域，避免了全局引导导致的纹理破坏
4. **用户友好**：交互式 GUI 让用户可以直观地定义运动模式

## 亮点与洞察

1. **3D 空间运动建模的思路很优雅**：从单视图运动先验推导多视图光流，自然保证几何一致性
2. **完全免训练**：不需要准备数据集或重训练任何模型，直接使用预训练 SD 即可
3. **运动模式丰富**：首个同时支持平移、旋转、缩放、拉伸四种运动模式的编辑方法
4. **解耦表示设计巧妙**：将运动编辑问题分解为背景保持、物体变换和遮挡补全三个子问题

## 局限与展望

1. **需要 3D 点云作为输入**：依赖预先重建的静态场景，限制了对任意图像的适用性
2. **刚体运动假设**：PKM 四种运动模式本质上都是刚体变换，不适用于非刚体变形（如人体姿态变化）
3. **实验以定性为主**：缺少充分的定量指标评估（如 FID、LPIPS 等大规模评测）
4. **时间效率**：基于 DDIM 的引导式推理过程可能较慢
5. **遮挡补全质量依赖 SD 的生成能力**：在复杂遮挡场景下补全质量可能不够理想

## 相关工作与启发

- **DragGAN / DragDiffusion**：基于拖拽点的交互编辑，但仅限简单平移
- **Motion Guidance**：利用光流引导扩散模型，但缺乏多视图一致性和纹理保持
- **NeRF-based editing**：需要重训练 NeRF，计算成本高
- **3DGS-based editing**：需要特定的 3DGS 表示
- **启发**：将运动编辑问题提升到 3D 空间是保证多视图一致性的根本解决方案；解耦表示的思想可以推广到更多编辑任务

## 评分

- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Early Timestep Zero-Shot Candidate Selection for Instruction-Guided Image Editing](early_timestep_zero-shot_candidate_selection_for_instruction-guided_image_editin.md)
- [\[CVPR 2026\] Coupled Diffusion Sampling for Training-Free Multi-View Image Editing](../../CVPR2026/image_generation/coupled_diffusion_sampling_for_training-free_multi-view_image_editing.md)
- [\[ICCV 2025\] DynamicID: Zero-Shot Multi-ID Image Personalization with Flexible Facial Editability](dynamicid_zero-shot_multi-id_image_personalization_with_flexible_facial_editabil.md)
- [\[ICCV 2025\] AnyPortal: Zero-Shot Consistent Video Background Replacement](anyportal_zero-shot_consistent_video_background_replacement.md)
- [\[ICCV 2025\] MAVFlow: Preserving Paralinguistic Elements with Conditional Flow Matching for Zero-Shot AV2AV Multilingual Translation](mavflow_preserving_paralinguistic_elements_with_conditional_flow_matching_for_ze.md)

</div>

<!-- RELATED:END -->
