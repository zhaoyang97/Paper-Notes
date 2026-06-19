---
title: >-
  [论文解读] ROGR: Relightable 3D Objects using Generative Relighting
description: >-
  [NeurIPS 2025 Spotlight][3D视觉][重光照] 本文提出ROGR，利用多视角扩散重光照模型生成多光照条件下的一致图像，训练一个光照条件化的NeRF，实现任意环境光照下的前馈式3D物体重光照，在TensoIR和Stanford-ORB基准上达到SOTA性能且支持交互式渲染。 将真实物体插入新环境并在不同…
tags:
  - "NeurIPS 2025 Spotlight"
  - "3D视觉"
  - "重光照"
  - "神经辐射场"
  - "生成式重光照"
  - "扩散模型"
  - "环境光照"
---

# ROGR: Relightable 3D Objects using Generative Relighting

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2510.03163](https://arxiv.org/abs/2510.03163)  
**代码**: 有项目页面（Project Page）  
**领域**: 3D视觉 / 重光照  
**关键词**: 重光照, 神经辐射场, 生成式重光照, 扩散模型, 环境光照

## 一句话总结

本文提出ROGR，利用多视角扩散重光照模型生成多光照条件下的一致图像，训练一个光照条件化的NeRF，实现任意环境光照下的前馈式3D物体重光照，在TensoIR和Stanford-ORB基准上达到SOTA性能且支持交互式渲染。

## 研究背景与动机

将真实物体插入新环境并在不同光照条件下正确渲染是计算机图形学的经典问题。当前的3D重光照方法主要有两条技术路线，各有明显不足：

**逆渲染（Inverse Rendering）**：通过优化材质和光照参数来解释观测图像。问题在于：(a) 物理光传输与简化模型之间的不匹配导致优化脆弱；(b) 由于固有歧义，恢复的物体属性在新光照下经常产生不真实的外观；(c) 基于物理的蒙特卡洛光传输模拟计算成本极高，不适合交互应用。

**生成式重光照**：如Neural Gaffer、DiLightNet等扩散模型方法可以生成逼真的单图重光照效果，但**每次只能独立处理一张图像**，导致多视角结果不一致。虽然IllumiNeRF尝试将不一致的重光照图像重建为NeRF，但**每个新目标光照都需要单独优化一个新的3D表示**，无法支持交互式使用。

本文的核心思路是：**先用多视角扩散模型生成一致的重光照数据集，再用这些数据训练一个光照条件化的NeRF，实现一次训练、任意光照的前馈渲染。** 这本质上是将虚拟光台（Light Stage）的理念推广到生成式模型中。

## 方法详解

### 整体框架

1. 使用多视角重光照扩散模型在M=111个环境光照下重光照N=64个视角的输入图像，生成N×M张多光照数据集
2. 用该数据集训练光照条件化的NeRF
3. 推理时，NeRF接受任意环境光照输入，前馈输出重光照结果

### 关键设计

1. **多视角重光照扩散模型**:

    - 基于CAT3D的多视角扩散架构，具有跨视角自注意力层
    - 环境光照编码采用Neural Gaffer的策略：HDR（对数 tone mapping后归一化）和LDR（标准 tone mapping）两个独立latent
    - 对每张输入图像，将环境光照旋转到对应的相机视角，与图像latent和ray map拼接后输入扩散网络
    - 关键优势：**联合降噪多个视角**（64个），确保生成的重光照图像在不同视角间保持一致
    - 在128个TPU v5上训练，总batch size=128

2. **光照条件化NeRF（双分支架构）**:

    - 基于NeRF-Casting，设计了两种光照条件信号：

   **通用条件（General Conditioning）**：
    - 将完整环境光照映射为128维向量
    - 使用基于ViT-S8的Transformer编码器从头训练：f_general = W · ViT(E)
    - 关键区别于NeRF-in-the-Wild：embedding不是针对单张图像优化的，而是环境光照的可学习映射，因此可泛化到新光照

   **镜面条件（Specular Conditioning）**：
    - 沿反射方向查询环境光照值和其模糊版本
    - 使用不同宽度σ的高斯核对环境光照进行预模糊，模拟不同粗糙度材质的反射效果
    - f_i^specular = ∫ E(ω') G(ω'; ω_r, σ_i) dω'
    - 本质是预计算辐射传输（PRT）思想的现代化实现

3. **架构细节**:

    - 几何MLP输出密度、粗糙度、法线和几何特征
    - 颜色MLP接收3D坐标、视角方向、通用条件和镜面条件，输出RGB值
    - 在8块H100上训练50万步
    - 推理时每帧0.384秒，支持交互式重光照

### 训练策略

- 扩散模型：使用40万个合成3D物体（含10万Objaverse）渲染训练数据，每个物体64视角×16 HDR光照
- 渐进式训练：4视角扩散→16视角→64视角
- NeRF：使用111个环境光照生成多光照数据集，在此基础上训练光照条件化NeRF

## 实验关键数据

### TensoIR基准（合成场景）

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| NeRFactor | 23.38 | 0.908 | 0.131 |
| InvRender | 23.97 | 0.901 | 0.101 |
| TensoIR | 28.58 | 0.944 | 0.081 |
| Neural-PBIR | 27.09 | 0.925 | 0.085 |
| NeRO | 27.00 | 0.935 | 0.074 |
| R3DG | 29.05 | 0.937 | 0.080 |
| Neural Gaffer | 27.30 | 0.918 | 0.122 |
| IllumiNeRF | 29.71 | 0.947 | 0.072 |
| **ROGR (Ours)** | **30.74** | **0.950** | **0.069** |

### Stanford-ORB基准（真实场景）

| 方法 | PSNR-H↑ | PSNR-L↑ | SSIM↑ | LPIPS↓ |
|------|---------|---------|-------|--------|
| NVDiffRecMC† | 25.08 | 32.28 | 0.974 | 0.027 |
| Neural-PBIR | 26.01 | **33.26** | 0.979 | **0.023** |
| IllumiNeRF | 25.56 | 32.74 | 0.976 | 0.027 |
| R3DG | 21.25 | 27.50 | 0.962 | 0.063 |
| **ROGR (Ours)** | **26.21** | 32.91 | **0.980** | 0.027 |

### 消融实验（TensoIR hotdog场景）

| 配置 | PSNR↑ | SSIM↑ | LPIPS↓ | 说明 |
|------|-------|-------|--------|------|
| 完整模型（64视角，111环境光） | **31.88** | **0.91** | **0.075** | - |
| (a) 镜面条件无模糊 | 31.35 | 0.90 | 0.080 | 粗糙表面表现下降 |
| (b) 无镜面条件 | 30.00 | 0.88 | 0.079 | 高光精度明显下降 |
| (c) 无通用条件 | 21.59 | 0.70 | 0.130 | 严重伪影 |
| (d) 逐图像embedding | 19.12 | 0.62 | 0.160 | 无法泛化到新光照 |
| (e) 128×128环境光照 | 29.26 | 0.89 | 0.082 | 高光细节损失 |
| (f) 64×64环境光照 | 27.69 | 0.86 | 0.110 | 渲染伪影 |

### 关键发现

- ROGR在TensoIR上PSNR超过IllumiNeRF（之前SOTA）1.03dB，SSIM和LPIPS同时提升
- 在Stanford-ORB上PSNR-H达到最佳，SSIM最佳，综合表现最优
- 通用条件是核心组件——移除后性能骤降（31.88→21.59 PSNR）
- 镜面条件的多尺度模糊设计对不同粗糙度材质都很重要
- 联合降噪更多视角（64 > 16 > 4）显著提升重光照一致性
- 渲染速度0.384秒/帧，与高斯Splatting方法（R3DG，0.415秒/帧）相当

## 亮点与洞察

- 将虚拟光台的思想与生成式模型结合：先生成多光照数据，再蒸馏到高效的前馈模型中，实现了"训练一次、处处重光照"
- 双分支光照条件设计（通用+镜面）精确分工：通用条件负责全局光照效果，镜面条件负责高频反射细节
- 相比逆渲染方法，避免了材质-光照分解的歧义问题；相比IllumiNeRF，避免了每个光照条件都需要单独优化NeRF的开销
- 多视角扩散模型确保了跨视角一致性，而这正是之前单图像重光照方法的主要瓶颈
- 环境光照分辨率的消融实验提供了实用的工程指导（512×512是必要的）

## 局限与展望

- 训练数据未包含次表面散射、折射、体积效果等复杂材质
- 环境光照假设光源无穷远，不支持近场光照
- 仅处理单物体场景，未扩展到大规模场景重光照
- 渲染速度虽然远快于逆渲染方法，但仍不如高斯Splatting的实时渲染
- 扩散模型生成的重光照数据虽然比单图方法一致，但仍可能存在跨视角的微小不一致

## 相关工作与启发

- 将PRT（预计算辐射传输）的"线性叠加"思想与现代生成式模型结合，提供了一个优雅的框架
- 与IllumiNeRF的核心区别在于"泛化"：ROGR训练一个通用的光照条件化NeRF，而非每个光照条件训练一个独立NeRF
- 启发：生成式模型可以作为"虚拟数据工厂"，为下游任务生成高质量训练数据
- NeRF-Casting的反射建模为本文的镜面条件提供了基础，说明精确的反射建模在重光照中的重要性

## 评分

- 新颖性: ⭐⭐⭐⭐ — 生成式蒸馏重光照的框架新颖，双分支条件设计有巧思
- 实验充分度: ⭐⭐⭐⭐⭐ — 合成+真实基准，详尽的消融实验，定性定量全面
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，部分技术细节可更简洁
- 价值: ⭐⭐⭐⭐⭐ — 在3D重光照任务上达到新SOTA，具有实际应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Generative Multiview Relighting for 3D Reconstruction under Extreme Illumination Variation](../../CVPR2025/3d_vision/generative_multiview_relighting_for_3d_reconstruction_under_extreme_illumination.md)
- [\[NeurIPS 2025\] MetaGS: A Meta-Learned Gaussian-Phong Model for Out-of-Distribution 3D Scene Relighting](metags_a_meta-learned_gaussian-phong_model_for_out-of-distribution_3d_scene_reli.md)
- [\[CVPR 2026\] Relightable Holoported Characters: Capturing and Relighting Dynamic Human Performance from Sparse Views](../../CVPR2026/3d_vision/relightable_holoported_characters_capturing_and_relighting_dynamic_human_perform.md)
- [\[CVPR 2025\] RNG: Relightable Neural Gaussians](../../CVPR2025/3d_vision/rng_relightable_neural_gaussians.md)
- [\[CVPR 2025\] ARM: Appearance Reconstruction Model for Relightable 3D Generation](../../CVPR2025/3d_vision/arm_appearance_reconstruction_model_for_relightable_3d_generation.md)

</div>

<!-- RELATED:END -->
