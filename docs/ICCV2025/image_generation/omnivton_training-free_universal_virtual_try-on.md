---
title: >-
  [论文解读] OmniVTON: Training-Free Universal Virtual Try-On
description: >-
  [ICCV2025][图像生成][virtual try-on] OmniVTON 提出首个无需训练的通用虚拟试穿框架，通过解耦服装纹理与姿态条件，利用结构化服装变形、连续边界缝合和频谱姿态注入三大模块，在 in-shop 和 in-the-wild 场景中均实现高保真试穿，并首次支持多人试穿。 - 问题定义：图像虚拟试穿（…
tags:
  - "ICCV2025"
  - "图像生成"
  - "virtual try-on"
  - "training-free"
  - "扩散模型"
  - "garment warping"
  - "pose alignment"
---

# OmniVTON: Training-Free Universal Virtual Try-On

**会议**: ICCV2025  
**arXiv**: [2507.15037](https://arxiv.org/abs/2507.15037)  
**代码**: [GitHub](https://github.com/Jerome-Young/OmniVTON)  
**领域**: 图像生成  
**关键词**: virtual try-on, training-free, diffusion model, garment warping, pose alignment  

## 一句话总结

OmniVTON 提出首个无需训练的通用虚拟试穿框架，通过解耦服装纹理与姿态条件，利用结构化服装变形、连续边界缝合和频谱姿态注入三大模块，在 in-shop 和 in-the-wild 场景中均实现高保真试穿，并首次支持多人试穿。

## 研究背景与动机

- **问题定义**：图像虚拟试穿（VTON）需要将服装图像无缝转移到目标人体上，保持纹理一致性和姿态保真度
- **现有方法局限**：
    - 监督式 in-shop 方法（GP-VTON, IDM-VTON 等）依赖配对训练数据，跨域泛化差
    - 无监督 in-the-wild 方法（StreetTryOn）受数据分布偏差限制，通用性不足
    - 两种范式都需要为特定条件训练专用模型，构建跨类别、跨姿态的大规模数据集不切实际
- **核心挑战**：
  1. **细粒度纹理一致性**：无训练阶段难以建立服装-人体对齐并保留纹理细节
  2. **人体姿态对齐**：现有方法依赖关键点或 DensePose 条件，需重新训练跨模态特征融合
- **动机**：需要一个统一的、无需训练的 VTON 框架，能在不同域和场景间泛化

## 方法详解

### 整体框架

OmniVTON 采用两阶段工作流，利用现成的扩散模型（无需任何训练）：
1. **阶段一**：通过结构化服装变形（SGM）将目标服装变形对齐到人体，生成服装先验
2. **阶段二**：利用服装先验和姿态编码噪声，通过连续边界缝合（CBS）机制逐步修复并完成最终试穿图像

### 关键设计一：结构化服装变形（SGM）

SGM 利用骨架信息和解析图约束服装变形，无需重新训练，适用于不同域。

**伪人物图像生成**：对于 Shop-to-X 场景（仅有服装图像），通过注意力调制生成伪人物图像。具体地，并行去噪服装条件噪声和人物条件噪声，将后者的 K 和 V 注入前者的自注意力层：

$$f_c = \text{Softmax}\left(\frac{Q_c \cdot [K_c \| K_p]^\top}{\sqrt{d}}\right)[V_c \| V_p]$$

**多部件语义对应**：利用 OpenPose 的 25 个关键点，将人体划分为 N 个语义区域（以上衣为例：躯干、左右上臂、左右前臂共 5 个），建立目标服装与源人物图像之间的多部件语义对应。用 TAPPS 生成分割图隔离各区域像素。

**局部化变换**：对每对边界框的角点，用 Levenberg-Marquardt 算法优化单应矩阵 $\mathcal{H}_{o \to p}^i \in \mathbb{R}^{3 \times 3}$，然后执行分段透视变换：

$$\begin{bmatrix} x_o' \\ y_o' \\ 1 \end{bmatrix} = \sum_{i=1}^{5} \mathbb{I}_{\text{Region}_i}(x_o, y_o) H_{o \to p}^i \begin{bmatrix} x_o \\ y_o \\ 1 \end{bmatrix}$$

### 关键设计二：频谱姿态注入（SPI）

DDIM 反演可保留源人物的结构信息，但会引入源服装纹理污染。SPI 利用频域分析解决此问题：

1. 对反演噪声 $z_T^{inv}$ 和随机噪声 $z_T$ 进行 FFT 并中心化
2. 用高斯低通掩模 $G_\tau$ 进行频域加权融合：
$$\hat{f}_T = G_\tau \odot f_T^{inv} + (1 - G_\tau) \odot f_T$$
3. 逆 FFT 得到混合初始噪声 $\hat{z}_T$

**核心思想**：低频保留反演噪声的姿态结构信息，高频替换为随机噪声以消除纹理残留并增强生成灵活性。

### 关键设计三：连续边界缝合（CBS）

SGM 的多区域拼接会产生纹理不连续。CBS 通过双向语义上下文信息改善边界连续性：

- 从 $I_c$ 路径到 $I_p'$ 路径：查询 $Q_p'$ 匹配目标服装纹理，弥合不连续
- 从 $I_p'$ 路径到 $I_c$ 路径：增强两条路径注意力图的相似性，抑制不相似值

### 损失函数

OmniVTON 是无需训练的方法，不涉及损失函数设计。所有组件利用预训练扩散模型的推理管道实现。

## 实验关键数据

### 主实验结果

**VITON-HD 数据集定量比较**（所有 VTON 方法使用 DressCode 预训练模型测试跨数据集泛化）：

| 方法 | 年份 | FID_u ↓ | FID_p ↓ | SSIM_p ↑ | LPIPS_p ↓ |
|------|------|---------|---------|----------|-----------|
| PBE | 2023 | 19.230 | 17.649 | 0.784 | 0.227 |
| AnyDoor | 2024 | 14.830 | 9.922 | 0.796 | 0.164 |
| GP-VTON | 2023 | 51.566 | 49.196 | 0.810 | 0.249 |
| IDM-VTON | 2024 | 23.035 | 20.460 | 0.812 | 0.147 |
| **OmniVTON** | - | **9.621** | **7.758** | **0.832** | **0.145** |

**DressCode 数据集定量比较**（跨服装类型适应性测试）：

| 方法 | FID_u ↓ | FID_p ↓ | SSIM_p ↑ | LPIPS_p ↓ |
|------|---------|---------|----------|-----------|
| CAT-DM | 13.678 | 12.028 | 0.858 | 0.125 |
| IDM-VTON | 9.685 | 8.377 | 0.842 | 0.138 |
| **OmniVTON** | **6.450** | **5.335** | **0.865** | **0.119** |

### 消融实验

| 变体 | SGM | CBS | SPI | FID_u ↓ | FID_p ↓ | SSIM_p ↑ | LPIPS_p ↓ |
|------|-----|-----|-----|---------|---------|----------|-----------|
| Base | - | - | - | 18.445 | 16.878 | 0.773 | 0.222 |
| (A) | ✓ | - | - | 13.303 | 11.475 | 0.809 | 0.177 |
| (B) | ✓ | ✓ | - | 9.799 | 7.993 | 0.824 | 0.158 |
| (C) | ✓ | - | ✓ | 13.148 | 10.767 | 0.813 | 0.180 |
| **OmniVTON** | ✓ | ✓ | ✓ | **9.621** | **7.758** | **0.832** | **0.145** |

### 关键发现

1. SGM 单独即可将 FID_u 从 18.445 降至 13.303，证明无训练服装对齐的有效性
2. CBS 在 SGM 基础上进一步将 LPIPS 提升 0.019，改善感知质量
3. SPI 提供显著的 SSIM 和 FID 增益，有效抑制噪声污染并保持结构一致性
4. 在 DressCode 上 FID_u 相比最佳基线提升 33.4%，展现跨服装类型适应能力
5. 在 StreetTryOn 基准的所有四种跨场景设置中均领先，甚至超过在域内数据上训练的 StreetTryOn

## 亮点与洞察

1. **首个无需训练的通用 VTON 框架**：将 in-shop 和 in-the-wild 场景统一，消除了为特定条件训练专用模型的需要
2. **解耦策略精妙**：将服装纹理保持和姿态对齐解耦为独立模块，避免扩散模型同时处理多个条件时的偏差问题
3. **频域分析的创新应用**：SPI 利用潜空间的频谱特性，低频保持姿态结构、高频增强生成灵活性，思路优雅
4. **多人试穿**：首次实现多人场景试穿，通过沿空间维度拼接多个服装同时生成伪人物图像
5. **跨域泛化能力强**：在 VITON-HD 上使用 DressCode 预训练模型测试，FID 降低 5.209，说明方法不依赖特定域

## 局限性

1. 极端情况下表现受限：高密度人群或目标身体区域极小时，会导致服装对齐失败
2. 多区域拼接仍可能在边界处引入伪影，CBS 并不能完全消除所有不连续
3. 依赖多个预训练模型（OpenPose、TAPPS、扩散模型），推理链较长
4. 无训练方法的推理速度可能较慢，需要逐步去噪和多次注意力调制

## 相关工作与启发

- **服装变形**：从 TPS（VITON）→ 光流（GP-VTON）→ 无训练骨架引导（本文），趋势是减少对配对数据的依赖
- **隐式变形 VTON**：IDM-VTON、StableVITON 通过注意力机制隐式建模变形，但缺乏显式几何约束
- **示例引导修复**：PBE 和 AnyDoor 虽具通用性，但缺乏虚拟试穿的专用设计
- **频域方法启发**：SPI 的频域调制思路可推广到其他需要解耦结构信息和纹理信息的图像生成任务

## 评分 ⭐⭐⭐⭐

创新度高，首次实现无训练通用 VTON，方法设计合理且各模块有明确消融验证。解耦策略和频域分析思路优雅。实验覆盖多数据集、多场景，定量定性结果均优。多人试穿是有意义的扩展。局限主要在极端场景，整体完成度高。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] BooW-VTON: Boosting In-the-Wild Virtual Try-On via Mask-Free Pseudo Data Training](../../CVPR2025/image_generation/boow-vton_boosting_in-the-wild_virtual_try-on_via_mask-free_pseudo_data_training.md)
- [\[CVPR 2025\] Pursuing Temporal-Consistent Video Virtual Try-On via Dynamic Pose Interaction](../../CVPR2025/image_generation/pursuing_temporal-consistent_video_virtual_try-on_via_dynamic_pose_interaction.md)
- [\[CVPR 2025\] Shining Yourself: High-Fidelity Ornaments Virtual Try-on with Diffusion Model](../../CVPR2025/image_generation/shining_yourself_high-fidelity_ornaments_virtual_try-on_with_diffusion_model.md)
- [\[ICCV 2025\] VIGFace: Virtual Identity Generation for Privacy-Free Face Recognition Dataset](vigface_virtual_identity_generation_for_privacy-free_face_recognition_dataset.md)
- [\[CVPR 2026\] PG-VTON: Single-Pass Training-Free Virtual Try-On via Patch-Guided Reference Alignment](../../CVPR2026/image_generation/pg-vton_single-pass_training-free_virtual_try-on_via_patch-guided_reference_alig.md)

</div>

<!-- RELATED:END -->
