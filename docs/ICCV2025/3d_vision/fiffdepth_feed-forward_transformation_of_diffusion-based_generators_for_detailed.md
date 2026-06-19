---
title: >-
  [论文解读] FiffDepth: Feed-forward Transformation of Diffusion-Based Generators for Detailed Depth Estimation
description: >-
  [ICCV 2025][3D视觉][单目深度估计] 提出FiffDepth，将预训练的扩散模型转化为确定性前馈架构进行单目深度估计，通过保持扩散轨迹维持細节生成能力，并引入可学习滤波器蒸馏DINOv2的鲁棒泛化能力到扩散骨干网络，在效率、精度和细节丰富度三方面同时超越现有方法。 - 单目深度估计（MDE）是基础3D视觉问题：…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "单目深度估计"
  - "扩散模型"
  - "前馈架构"
  - "DINOv2蒸馏"
  - "细节保持"
---

# FiffDepth: Feed-forward Transformation of Diffusion-Based Generators for Detailed Depth Estimation

**会议**: ICCV 2025  
**arXiv**: [2412.00671](https://arxiv.org/abs/2412.00671)  
**代码**: [项目主页](https://yunpeng1998.github.io/FiffDepth/)  
**领域**: 3D视觉 / 单目深度估计  
**关键词**: 单目深度估计, 扩散模型, 前馈架构, DINOv2蒸馏, 细节保持

## 一句话总结

提出FiffDepth，将预训练的扩散模型转化为确定性前馈架构进行单目深度估计，通过保持扩散轨迹维持細节生成能力，并引入可学习滤波器蒸馏DINOv2的鲁棒泛化能力到扩散骨干网络，在效率、精度和细节丰富度三方面同时超越现有方法。

## 研究背景与动机

- **单目深度估计（MDE）是基础3D视觉问题**：广泛应用于3D场景重建、自动导航和AI内容创作中
- **现有方法的核心挑战**：
    - 真实世界深度数据集噪声大，合成数据存在域差距
    - 生成式方法（如Marigold）直接微调扩散模型为深度图生成模型，但扩散过程引入噪声和不确定性，对密集预测任务不理想
    - DINOv2等前馈模型（FFN）泛化能力强，但缺乏细节
    - 现有基于扩散的方法效率低（需要多步去噪和测试时组装）
- **关键观察**：
    - 扩散模型的去噪模块以前馈方式直接使用，效果更好更稳定
    - DINOv2能准确预测低频深度分量但缺乏高频细节
    - 扩散模型本身可以学习一个滤波器来分离高低频成分
- **核心思路**：利用扩散模型轨迹的延伸实现高效深度估计，同时引入DINOv2知识来增强泛化能力

## 方法详解

### 整体框架

FiffDepth基于预训练的Stable Diffusion（SD）模型构建，核心创新在于将扩散模型从随机生成框架转化为确定性前馈深度估计器。整个方法包含三个关键组件：

1. **前馈转换**（t=0步的深度预测）
2. **扩散轨迹保持**（训练时同时维持原始去噪能力）
3. **可学习滤波器蒸馏**（t=-1步的DINOv2知识迁移）

### 关键组件1：前馈深度估计

不同于Marigold等方法将深度估计分解为多步去噪过程，FiffDepth直接在t=0时刻进行单步前馈预测：

$$\mathbf{d}_0 = \hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_0, t=0)$$

其中 $\mathbf{x}_0$ 是RGB图像的latent表示，$\mathbf{d}_0$ 是深度图的latent表示。推理时只需一次前向传播，无需迭代去噪。

### 关键组件2：保持扩散轨迹

为防止微调过程中扩散轨迹退化，训练时同时维持前馈步和前序去噪步。核心设计是将目标latent修改为图像和深度表示的混合：

$$\mathbf{b}_0 = \gamma \mathbf{x}_0 + (1 - \gamma) \mathbf{d}_0$$

使用v-prediction参数化定义训练目标：

$$\mathbf{v}_t = \sqrt{\bar{\alpha}_t} \boldsymbol{\epsilon} - \sqrt{1 - \bar{\alpha}_t} \mathbf{b}_0$$

$$L_k = \|\mathbf{v}_t - \hat{\boldsymbol{\epsilon}}_\theta(\mathbf{b}_t, t)\|_2^2, \quad t \in \{1, \ldots, T\}$$

这种混合目标迫使模型保留图像生成和深度估计任务之间的共享特征，使模型在细调过程中自然适应深度估计并保留增强预测精度和细节的特征。γ设为0.5。

### 关键组件3：可学习滤波器蒸馏

**问题**：仅在合成数据上训练限制了泛化能力；DINOv2能够泛化但缺乏细节。直接用DINOv2生成的伪标签监督 $\mathbf{d}_0$ 会破坏细节。

**解决方案**：利用扩散模型自身学习一个滤波器F，在t=-1步产生滤去高频细节的输出：

$$\mathbf{d}_{-1} = \hat{\boldsymbol{\epsilon}}_\theta(\mathbf{d}_0, t=-1)$$

在这一步用DINOv2的预测作为伪标签来监督 $\mathbf{d}_{-1}$，使得：
- DINOv2的鲁棒低频预测能力迁移到模型中
- $\mathbf{d}_0$ 的高频细节不受影响
- 可以使用大量未标注的真实图像数据进行训练

### 损失函数

总损失函数为MAE损失、梯度匹配损失和轨迹保持损失的加权和：

$$L_{\text{final}} = \sum_{t \in \{-1, 0\}} (\lambda_{\text{MAE}} L_{\text{MAE}}(\mathbf{d}_t, \mathbf{d}_t^*) + \lambda_{\text{GM}} L_{\text{GM}}(\mathbf{d}_t, \mathbf{d}_t^*)) + \lambda_k L_k$$

参数设置：$\lambda_{\text{MAE}}=1$，$\lambda_{\text{GM}}=0.5$，$\lambda_k=0.2$。

## 实验关键数据

### 训练设置

- 合成数据：Hypersim + Virtual KITTI，共74K图像
- 真实数据：LAION-Art子集中的20万样本用于t=-1步训练
- 教师模型：Depth Anything V2-Large作为DINOv2模型
- 每个batch中合成与真实数据各占一半

### 主实验：零样本仿射不变深度估计

| 方法 | 训练数据 | NYUv2 AbsRel↓ | NYUv2 δ1↑ | KITTI AbsRel↓ | ETH3D AbsRel↓ | ScanNet AbsRel↓ | DIODE AbsRel↓ | DA-2K Acc |
|------|----------|---------------|-----------|---------------|---------------|-----------------|---------------|-----------|
| Marigold | 74K* | 5.5 | 96.4 | 9.9 | 6.4 | 6.4 | 30.8 | 86.8 |
| GeoWizard | 280K* | 5.2 | 96.6 | 9.7 | 6.4 | 6.1 | 29.7 | 88.1 |
| Lotus-D | 59K* | 5.3 | 96.7 | 8.1 | 6.5 | 5.8 | 29.9 | 86.8 |
| DA v1-L | 62.6M* | 4.3 | 98.1 | 7.6 | 12.7 | 4.2 | 27.7 | 88.5 |
| DA v2-L | 62.6M* | 4.5 | 97.9 | 7.4 | 13.1 | 4.2 | 26.2 | 97.1 |
| **FiffDepth** | **274K*** | **4.4** | **97.8** | **7.3** | **7.1** | **4.2** | **23.9** | **97.1** |

关键发现：FiffDepth仅用274K训练数据，在大部分基准上达到最优或可比结果，尤其在DIODE-Full上显著优于所有方法。

### 边界精度对比

| 方法 | Sintel F1↑ | Spring F1↑ | iBims F1↑ | AM R↑ | P3M R↑ | DIS R↑ |
|------|-----------|-----------|----------|-------|-------|-------|
| DA v2 | 0.228 | 0.056 | 0.111 | 0.107 | 0.131 | 0.056 |
| Depth Pro | 0.409 | 0.079 | 0.176 | 0.173 | 0.168 | 0.077 |
| **FiffDepth** | **0.423** | **0.086** | **0.189** | **0.176** | **0.179** | **0.091** |

### 推理效率对比

| 方法 | Marigold | Marigold(LCM) | GeoWizard | DepthFM | DA v2-L | Depth Pro | FiffDepth |
|------|----------|---------------|-----------|---------|---------|-----------|-----------|
| 时间(s) | 103 | 1.7 | 19 | 0.39 | 0.026 | 0.23 | 0.092 |

FiffDepth比Marigold快1120倍，比GeoWizard快206倍，同时效率接近DA v2。

### 消融实验

各组件对性能的影响（定性分析）：
- 去除轨迹保持但仅预测图像latent → 物体间相对深度关系受损
- 去除轨迹保持 → 部分细节丢失
- 去除DINOv2监督 → 泛化能力显著下降
- 将DINOv2监督应用于d0（而非d-1） → 细节精度降低

## 亮点与洞察

1. **优雅的前馈转换**：将扩散模型轨迹直接延伸到深度域，避免了多步去噪的不确定性和低效率
2. **巧妙的频率解耦蒸馏**：利用扩散模型自身作为滤波器学习器，在t=-1步实现低频知识蒸馏，不损害d0的高频细节
3. **高效的数据利用**：仅用274K数据（合成+真实）即可达到与使用62.6M数据的DA v2相当的性能
4. **确定性推理**：推理时完全确定，无需多次采样和融合，单次前向传播即可输出

## 局限性

- 消融实验主要是定性展示，缺乏定量数据
- 对γ参数的敏感性分析不充分
- 尽管在大多数指标上优秀，在个别benchmark上（如ETH3D和KITTI的δ1）仍略逊于DA v2-L
- 仍依赖合成数据的质量，对于合成数据中未覆盖的场景类型可能存在泛化风险

## 相关工作与启发

- **与Marigold/GeoWizard的对比**：这些方法保持了扩散框架，导致效率低、不确定性高；FiffDepth的前馈方式是更优的选择
- **与DA系列的对比**：DA依赖大规模真实数据实现泛化；FiffDepth用蒸馏方式以少量数据达到同等泛化
- **启发**：生成模型转密集预测的关键不在于保持生成过程，而在于利用其学到的表示；知识蒸馏可以用于频率解耦的方式进行

## 评分 ⭐⭐⭐⭐

创新性好，方法简洁有效，实验充分覆盖多个基准。前馈转换+频率解耦蒸馏的设计思路值得学习。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] AdaHuman: Animatable Detailed 3D Human Generation with Compositional Multiview Diffusion](adahuman_animatable_detailed_3d_human_generation_with_compositional_multiview_di.md)
- [\[ECCV 2024\] DiffusionDepth: Diffusion Denoising Approach for Monocular Depth Estimation](../../ECCV2024/3d_vision/diffusiondepth_diffusion_denoising_approach_for_monocular_depth_estimation.md)
- [\[CVPR 2026\] Iris: Integrating Language into Diffusion-based Monocular Depth Estimation](../../CVPR2026/3d_vision/iris_integrating_language_into_diffusion-based_monocular_depth_estimation.md)
- [\[CVPR 2025\] FLARE: Feed-forward Geometry, Appearance and Camera Estimation from Uncalibrated Sparse Views](../../CVPR2025/3d_vision/flare_feed-forward_geometry_appearance_and_camera_estimation_from_uncalibrated_s.md)
- [\[ECCV 2024\] Diffusion Models for Monocular Depth Estimation: Overcoming Challenging Conditions](../../ECCV2024/3d_vision/diffusion_models_for_monocular_depth_estimation_overcoming_challenging_condition.md)

</div>

<!-- RELATED:END -->
