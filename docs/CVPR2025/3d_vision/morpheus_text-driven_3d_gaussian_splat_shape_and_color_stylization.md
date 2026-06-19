---
title: >-
  [论文解读] Morpheus: Text-Driven 3D Gaussian Splat Shape and Color Stylization
description: >-
  [CVPR 2025][3D视觉][3D高斯溅射] 提出Morpheus，一种自回归3DGS风格化方法，核心贡献包括：(1) 新的RGBD扩散模型实现外观和形状风格化的独立强度控制；(2) Warp ControlNet通过变形合成帧传播风格；(3) 深度引导的特征共享确保多视角一致性。 将真实世界的3D场景用不同风格重新想…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "3D高斯溅射"
  - "风格化"
  - "RGBD扩散模型"
  - "形状编辑"
  - "多视角一致性"
  - "ControlNet"
---

# Morpheus: Text-Driven 3D Gaussian Splat Shape and Color Stylization

**会议**: CVPR 2025  
**arXiv**: [2503.02009](https://arxiv.org/abs/2503.02009)  
**代码**: [项目主页](https://nianticlabs.github.io/morpheus/)  
**领域**: 3D Vision  
**关键词**: 3D高斯溅射, 风格化, RGBD扩散模型, 形状编辑, 多视角一致性, ControlNet

## 一句话总结

提出Morpheus，一种自回归3DGS风格化方法，核心贡献包括：(1) 新的RGBD扩散模型实现外观和形状风格化的独立强度控制；(2) Warp ControlNet通过变形合成帧传播风格；(3) 深度引导的特征共享确保多视角一致性。

## 研究背景与动机

将真实世界的3D场景用不同风格重新想象（如"将客厅变成日式茶室"）是一个引人入胜的应用场景，也可用于扩展下游任务的训练分布。基于3DGS的新视角合成近年快速发展，风格化这些3D重建成为新的挑战。

**核心难点**：现有NVS风格化方法主要停留在纹理/颜色编辑层面，**无法令人信服地改变几何形状**。原因在于：任何几何改变需要更大的风格化强度，但增大强度会破坏多视角一致性。现有的2D扩散模型缺乏显式的几何理解和输出修改后几何的能力，也缺乏对外观和形状风格化强度的独立控制。

Instruct-NeRF2NeRF等迭代方法耗时长且结果模糊；GaussCtrl等方法通过跨注意力共享特征，但容易在复杂几何下传递错误信息导致不一致。Morpheus的目标是实现"可控的几何+外观风格化"+"多视角一致"。

## 方法详解

### 整体框架

Morpheus的pipeline：(1) 输入深度正则化的3DGS模型，沿代表性轨迹渲染RGB和深度图序列；(2) 自回归地风格化每一帧——使用RGBD扩散模型进行风格化，使用Warp ControlNet传播前帧风格，使用深度引导的特征共享保持一致性；(3) 用风格化后的RGBD帧重训练一个新的3DGS模型。

### 关键设计

**1. RGBD扩散模型（独立控制形状和外观风格化强度）**

基于Stable Diffusion 2.1扩展为RGBD模型：分别编码RGB和深度图到latent空间，U-Net的输入通道扩展为4×2=8个特征通道+2个时间掩码通道。关键创新在于**分离噪声调度**——为颜色和深度引入独立的最大时间步 T_max^I 和 T_max^D。去噪过程中，在 t > T_max^D 时不修改深度、t > T_max^I 时不修改颜色，通过掩码告知网络哪些通道的修改会被接受。这使得"保持颜色风格不变，单独调整形状变化程度"成为可能。

**2. Warp ControlNet（一致性传播）**

为每个新帧，将之前风格化帧通过深度前向变形（forward-warp）到当前视角，与未风格化的当前帧合成一个composite。训练一个定制ControlNet以composite（RGBD+有效性掩码）为条件，引导RGBD扩散模型：(a) 修正变形区域的伪影；(b) 以与已变形区域一致的方式补全新暴露区域。hint网络通道扩大为(48,96,192,384)以更好理解合成与风格化任务。训练数据由RGBD模型自身生成25万对。

**3. 深度引导的特征共享**

在扩散模型的self-attention层中，将当前帧的query与参考帧的key/value拼接做跨注意力。关键改进：构建4D热图L_ij，通过前向变形参考帧深度来确定"参考帧哪个像素对应当前帧哪个像素"，只在对应位置增强跨注意力权重（而非全局均匀注意），避免美学特征被错误复制到不相关位置。特征注入也使用相同热图的argmax来选择注入哪个参考特征。

### 损失函数

3DGS优化使用尺度不变深度损失（与Metric3D预测深度的median-scaled版本对比）、法线方向损失（点积+余弦相似度）、TVL1正则化减少浮动伪影。最终风格化3DGS重训练时同时使用风格化颜色和深度作为监督。

## 实验关键数据

### 主实验：定量比较（Table 1，53组风格化）

| 方法 | CLIP Dir. Sim.↑ | CLIP Dir. Consist.↑ | RMSE↓ | LPIPS↓ |
|------|:---:|:---:|:---:|:---:|
| Instruct-N2N | 0.098 | 0.531 | 0.0463 | 0.0540 |
| Instruct-GS2GS | 0.097 | 0.519 | 0.0501 | 0.0403 |
| GaussCtrl | 0.123 | 0.590 | 0.0471 | 0.0438 |
| DGE | 0.113 | 0.565 | 0.0384 | 0.0407 |
| **Morpheus** | **0.175** | **0.606** | **0.0370** | **0.0378** |

Morpheus在所有指标上领先。CLIP Direction Similarity高达0.175（第二名0.123），说明风格化与prompt的符合度远超其他方法。处理时间约10分钟，与GaussCtrl和DGE持平。

### 消融实验（Table 2）

| 配置 | Similarity↑ | Consistency↑ | Seq. RMSE↓ | Seq. LPIPS↓ |
|------|:---:|:---:|:---:|:---:|
| 单帧独立风格化 | 0.161 | 0.592 | 0.1170 | 0.0941 |
| w/o Warp ControlNet | 0.170 | 0.604 | 0.0834 | 0.0917 |
| w/o 特征共享 | 0.178 | 0.611 | 0.0817 | 0.0931 |
| 全图跨注意力(无深度引导) | 0.180 | 0.610 | 0.0730 | 0.0914 |
| **完整模型** | **0.175** | **0.606** | **0.0702** | **0.0911** |

- Warp ControlNet删除后Sequential RMSE从0.0702→0.0834，说明它对帧间一致性贡献显著
- 深度引导注意力 vs 全图注意力：防止美学元素错位复制（见Fig. 5的熊眼案例）

### 关键发现

- **形状风格化是独特优势**：通过调节T_max^D可在保持纹理一致的同时产生显著的几何变化（Fig. 3）
- 用户研究（31人，49组）：在"是否符合风格描述"和"美学质量"两个维度上均大幅优于所有基线
- ConsistDreamer只能做局部纹理修改，而Morpheus可实现场景级别的结构性变换

## 亮点与洞察

1. **分离控制是核心突破**：独立的颜色/深度噪声调度使得首次实现了"可控的几何风格化"——以前的方法要么不改几何，要么一改就不一致
2. **深度引导的注意力**：比GaussCtrl的epipolar line方法更精确，因为直接利用了3D几何信息来确定像素对应关系
3. **自回归 vs 全局优化**：相比Instruct-NeRF2NeRF的迭代优化（数小时），自回归管线在10分钟内完成且质量更好

## 局限性

- 风格化质量依赖于原始3DGS模型的质量，原始splat中的错误（如深度不准确、模糊区域）会被继承到风格化结果中
- 自回归pipeline对轨迹选择敏感，需要平滑的相机轨迹以确保帧间重叠足够
- RGBD扩散模型预测的是尺度不变深度，需要用渲染深度进行缩放，可能引入额外误差
- 512×512分辨率限制，高分辨率场景需要进一步适配

## 相关工作与启发

- **GaussCtrl**: 跨注意力共享但不考虑几何，导致特征传递不精确
- **Instruct-NeRF2NeRF/GS2GS**: 迭代优化耗时长且容易产生模糊
- **ConsistDreamer**: 3D噪声表示但只能做表面纹理修改
- **启发**: RGBD扩散模型的分离控制思路可推广到其他3D编辑任务；Warp ControlNet的训练数据生成方式（用自己的模型生成配对数据）值得借鉴

## 评分

⭐⭐⭐⭐ — 方法设计巧妙（分离控制+深度引导注意力），解决了3D风格化中长期存在的"不能改几何"限制，来自Niantic Labs的工程实现也很扎实。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] GaussCtrl: Multi-View Consistent Text-Driven 3D Gaussian Splatting Editing](../../ECCV2024/3d_vision/gaussctrl_multi-view_consistent_text-driven_3d_gaussian_splatting_editing.md)
- [\[ICCV 2025\] MemoryTalker: Personalized Speech-Driven 3D Facial Animation via Audio-Guided Stylization](../../ICCV2025/3d_vision/memorytalker_personalized_speech-driven_3d_facial_animation_via_audio-guided_sty.md)
- [\[CVPR 2025\] Speedy-Splat: Fast 3D Gaussian Splatting with Sparse Pixels and Sparse Primitives](speedy-splat_fast_3d_gaussian_splatting_with_sparse_pixels_and_sparse_primitives.md)
- [\[CVPR 2025\] Geometry in Style: 3D Stylization via Surface Normal Deformation](geometry_in_style_3d_stylization_via_surface_normal_deformation.md)
- [\[CVPR 2025\] Dr. Splat: Directly Referring 3D Gaussian Splatting via Direct Language Embedding Registration](dr_splat_directly_referring_3d_gaussian_splatting_via_direct_language_embedding_.md)

</div>

<!-- RELATED:END -->
