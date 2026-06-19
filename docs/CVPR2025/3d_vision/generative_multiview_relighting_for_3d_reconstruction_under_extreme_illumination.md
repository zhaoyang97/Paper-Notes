---
title: >-
  [论文解读] Generative Multiview Relighting for 3D Reconstruction under Extreme Illumination Variation
description: >-
  [CVPR 2025][3D视觉][多视图重光照] 本文提出先用多视图重光照扩散模型将不同光照下拍摄的图像统一到参考光照条件，再用带有"shading embedding"的鲁棒 NeRF 模型重建 3D 表示，在极端光照变化下实现了远超现有方法的高保真外观重建，尤其擅长恢复镜面/高光效果。 1. 领域现状：三维重建和新视角…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "多视图重光照"
  - "3D重建"
  - "扩散模型"
  - "极端光照变化"
  - "NeRF"
---

# Generative Multiview Relighting for 3D Reconstruction under Extreme Illumination Variation

**会议**: CVPR 2025  
**arXiv**: [2412.15211](https://arxiv.org/abs/2412.15211)  
**代码**: [https://relight-to-reconstruct.github.io/](https://relight-to-reconstruct.github.io/) (项目页)  
**领域**: 3D视觉 / 三维重建  
**关键词**: 多视图重光照, 3D重建, 扩散模型, 极端光照变化, NeRF

## 一句话总结
本文提出先用多视图重光照扩散模型将不同光照下拍摄的图像统一到参考光照条件，再用带有"shading embedding"的鲁棒 NeRF 模型重建 3D 表示，在极端光照变化下实现了远超现有方法的高保真外观重建，尤其擅长恢复镜面/高光效果。

## 研究背景与动机

1. **领域现状**：三维重建和新视角合成（如 NeRF）通常假设输入图像在相同光照条件下拍摄。处理光照变化的常见策略有两种：(a) per-image appearance embedding（如 NeRF-W），让模型学习每张图的外观变化；(b) 逆向渲染，显式恢复材质和逐图光照。
2. **现有痛点**：Appearance embedding 会把视角相关的外观（如高光反射）也"解释掉"，导致重建结果偏漫射。逆向渲染通过低频参数化（球谐函数、球高斯）建模材质和光照，无法恢复高频的镜面反射。近期单图重光照扩散模型（如 IllumiNeRF、Neural Gaffer）虽然利用了强先验，但独立重光照每张图会导致材质解释不一致。
3. **核心矛盾**：从"极端光照变化"的图像中恢复物体外观（特别是视角相关的镜面高光）是高度欠定问题——外观变化可能来自光照变化也可能来自视角变化，二者严重混淆。
4. **本文目标**：(a) 如何从极端不同光照条件的图像中重建出忠实的 3D 外观；(b) 如何正确恢复镜面/高光效果而非退化为漫射结果。
5. **切入角度**：受光度立体法启发——同一材质在多种光照下的观测可以降低材质/光照歧义。将此洞察引入扩散模型，联合重光照所有图像而非独立处理。
6. **核心 idea**：用多视图扩散模型联合重光照所有输入图像到参考光照，再用 shading embedding 机制鲁棒地拟合残余不一致性。

## 方法详解

### 整体框架
两阶段流程：(1) 重光照扩散模型接收 N 张不同光照的输入图像和相机位姿，将所有图像同时重光照到参考图像的光照条件下；(2) 基于 NeRF-Casting 的 3D 重建模型从重光照后的图像中恢复几何和视角相关外观，使用 per-image shading embedding 来吸收扩散模型输出的残余不一致性。

### 关键设计

1. **多视图重光照扩散模型 (Multiview Relighting Diffusion Model)**:
    - 功能：将 N 张不同光照的图像统一到参考图像的光照条件
    - 核心思路：扩展单图重光照模型到多视图场景。架构基于类似 Stable Diffusion 1.5 的 latent diffusion，同时去噪 N 个 latent code $z_1,...,z_N$。去噪第 $i$ 张图的 latent 时，将原始"干净"图像 $I_i$ 和相机位姿 $\pi_i$ 作为输入提供几何信息，使用 3D self-attention blocks 让各视角 latent 之间交互，cross-attention 处理相机位姿。参考图像通过 reference map（全1通道）标识，其他图像用全0通道标识。
    - 设计动机：联合处理多视角利用了"不同光照同一材质"的互补信息（类似光度立体法），比独立处理每张图大幅减少了材质/光照歧义，产生更一致的重光照结果。

2. **Shading Embedding（着色嵌入）**:
    - 功能：吸收扩散模型输出中残余的高光位置偏移
    - 核心思路：为每张训练图学一个嵌入向量 $\mathbf{v}_i$，用它与空间特征一起通过 MLP 预测每张图特有的法线扰动 $\mathbf{n}_i(\mathbf{x}) = \text{normalize}(\text{MLP}(\mathbf{f}(\mathbf{x}), \mathbf{v}_i))$。扰动的法线改变了二次反射光线方向，从而调整高光反射位置。通过与密度场法线的一致性约束防止过大偏移。
    - 设计动机：扩散模型的隐式形状估计存在微小误差导致高光位置偏移。传统 appearance embedding 会把高光当作 per-image 漫射颜色吸收掉（导致漫射化），而 shading embedding 只调整法线方向，保留了物理正确的高光生成机制。

3. **纯合成训练数据与镜面增强**:
    - 功能：提供训练多视图重光照模型的大规模带GT数据
    - 核心思路：从约 30 万高质量 3D 物体资产中渲染训练数据，使用约 700 个环境贴图（Poly Haven）并随机方位旋转增强。关键创新是额外渲染一份将所有材质替换为完美镜面的副本，形成标准材质+镜面材质的混合训练集。模型分四阶段渐进训练（8→16→32→64 frames），在 64 个 TPU v5 上训练约两周。
    - 设计动机：镜面数据增强不仅提升了高光物体的重建质量，有趣的是也提升了普通漫射物体的效果。这可能是因为镜面材质提供了更丰富的光照-表面法线关系的训练信号。

### 损失函数 / 训练策略
重光照模型：标准扩散去噪损失 + classifier-free guidance（CFG=3），随机 mask 参考图注意力做无条件训练。3D 重建：NeRF-Casting 的标准损失，优化约 30 分钟/场景（16 A100 GPU），渲染单张 512×512 约 0.5 秒。推理时同时重光照 N=64 张图像。

## 实验关键数据

### 主实验

**合成数据 (Objaverse)**

| 方法 | Standard PSNR↑ | Standard LPIPS↓ | Shiny PSNR↑ | Shiny LPIPS↓ |
|------|------|------|------|------|
| NeROIC | 26.13 | 0.088 | 22.14 | 0.113 |
| NeRFCast+AE | 27.53 | 0.067 | 21.80 | 0.108 |
| IllumiNeRF (已知GT光照) | 29.22 | 0.057 | 23.46 | 0.095 |
| **Ours** | **31.34** | **0.053** | **26.54** | **0.090** |

**真实照片 (NAVI)**

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|------|------|------|
| NeRFCast+AE | 22.67 | 0.906 | 0.074 |
| NeROIC | 24.01 | 0.918 | 0.079 |
| **Ours** | **25.55** | **0.929** | **0.060** |

### 消融实验

**帧数消融 (Shiny Assets)**

| 同时处理帧数 N | PSNR↑ | SSIM↑ |
|------|---------|------|
| 1 frame | 23.85 | 0.889 |
| 8 frames | 25.62 | 0.901 |
| 32 frames | 26.14 | 0.908 |
| 64 frames (ours) | **26.54** | **0.911** |

**Per-image Embedding 消融**

| 方法 | PSNR↑ | LPIPS↓ |
|------|---------|------|
| No embeddings | 26.02 | 0.094 |
| Appearance embeddings (NeRF-W) | 24.38 | 0.096 |
| **Shading embeddings (ours)** | **26.54** | **0.090** |

### 关键发现
- 同时处理的帧数越多效果越好，从 1 帧到 64 帧 PSNR 提升近 3dB，证明联合重光照的价值
- Appearance embedding 甚至比没有 embedding 更差（24.38 vs 26.02），因为它把高光当 per-image variation 吸收了
- Shading embedding 比无 embedding 好（26.54 vs 26.02），正确地只调整法线方向
- 加入纯镜面训练数据不仅提升 Shiny 物体效果（+0.57 PSNR），连 Standard 物体也有提升（+0.10 PSNR）
- 本方法在真实照片上无需访问 GT 环境贴图，但仍超越了使用 GT 环境贴图的 IllumiNeRF

## 亮点与洞察
- **Shading embedding vs Appearance embedding 的对比揭示了一个重要洞察**：处理光照不一致性时，应当通过调整着色计算的输入（法线方向）而非直接调整输出（颜色），这保留了物理渲染管线的正确性。这一原则可推广到其他需要处理不一致输入的 3D 重建任务
- **多视图联合重光照**：将光度立体法的思想引入扩散模型，利用多视角+多光照的联合观测消解歧义，比独立处理每张图产生更一致的结果
- **镜面材质数据增强**：一个意外的发现——用完美镜面材质渲染额外训练数据不仅帮助镜面物体，也帮助漫射物体重建

## 局限与展望
- 需要输入物体 mask 和准确的相机位姿，高反射物体的位姿估计本身就困难（缺少可靠特征点）
- 完全依赖合成数据训练，虽然实验证明可泛化到真实照片，但 domain gap 仍可能存在
- 重光照模型一次处理 N=64 帧，更大规模（如互联网图片集合的数百张图）的扩展需要进一步研究
- 训练成本高（64 TPU v5 两周），限制了快速迭代
- 当前只处理物体级重建，场景级（如室内/室外大场景）的扩展是未来方向

## 相关工作与启发
- **vs IllumiNeRF**: IllumiNeRF 独立处理每张图的重光照导致严重不一致（表现为重建模糊），且需要 GT 环境贴图作为输入。本文联合处理消除了不一致性，且只需参考图像
- **vs NeROIC**: 基于逆向渲染的方法用低频参数化建模材质/光照，天然限制了高光恢复能力。本文通过生成式重光照绕过了逆向渲染
- **vs NeRF-W**: Appearance embedding 是处理光照变化的通用做法，但本文证明它对高光物体反而有害

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 多视图联合重光照+shading embedding 都是创新性设计
- 实验充分度: ⭐⭐⭐⭐⭐ 合成+真实数据，详尽消融，每个组件都单独验证
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，方法直觉解释到位
- 价值: ⭐⭐⭐⭐⭐ 解决了从网络图片重建高质量3D的关键障碍

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] ROGR: Relightable 3D Objects using Generative Relighting](../../NeurIPS2025/3d_vision/rogr_relightable_3d_objects_using_generative_relighting.md)
- [\[CVPR 2025\] Instant3dit: Multiview Inpainting for Fast Editing of 3D Objects](instant3dit_multiview_inpainting_for_fast_editing_of_3d_objects.md)
- [\[CVPR 2025\] Extreme Rotation Estimation in the Wild](extreme_rotation_estimation_in_the_wild.md)
- [\[CVPR 2025\] Touch2Shape: Touch-Conditioned 3D Diffusion for Shape Exploration and Reconstruction](touch2shape_touch-conditioned_3d_diffusion_for_shape_exploration_and_reconstruct.md)
- [\[CVPR 2025\] Decompositional Neural Scene Reconstruction with Generative Diffusion Prior](decompositional_neural_scene_reconstruction_with_generative_diffusion_prior.md)

</div>

<!-- RELATED:END -->
