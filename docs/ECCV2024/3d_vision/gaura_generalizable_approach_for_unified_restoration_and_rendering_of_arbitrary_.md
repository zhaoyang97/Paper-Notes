---
title: >-
  [论文解读] GAURA: Generalizable Approach for Unified Restoration and Rendering of Arbitrary Views
description: >-
  [ECCV 2024][3D视觉][新视角合成] 提出GAURA，一种基于可泛化NeRF的统一复原与渲染框架，通过可学习的退化感知latent codes在特征聚合和渲染阶段动态适应不同图像退化类型，无需逐场景优化即可从退化图像中渲染清晰的新视角。 领域现状： NeRF等神经渲染方法已能从多视角图像实现近真实感的新视角合成…
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "新视角合成"
  - "图像复原"
  - "通用退化处理"
  - "可泛化NeRF"
  - "Transformer"
---

# GAURA: Generalizable Approach for Unified Restoration and Rendering of Arbitrary Views

**会议**: ECCV 2024  
**arXiv**: [2407.08221](https://arxiv.org/abs/2407.08221)  
**代码**: [https://vinayak-vg.github.io/GAURA](https://vinayak-vg.github.io/GAURA)  
**领域**: 3D视觉  
**关键词**: 新视角合成, 图像复原, 通用退化处理, 可泛化NeRF, Transformer

## 一句话总结

提出GAURA，一种基于可泛化NeRF的统一复原与渲染框架，通过可学习的退化感知latent codes在特征聚合和渲染阶段动态适应不同图像退化类型，无需逐场景优化即可从退化图像中渲染清晰的新视角。

## 研究背景与动机

**领域现状**: NeRF等神经渲染方法已能从多视角图像实现近真实感的新视角合成，但要求输入图像质量完美。

**现有痛点**: 现实世界中图像常受到低光照、运动模糊、雾霾、雨雪等退化影响，现有方法需要为每种退化类型显式建模物理退化过程（如Seathru-NeRF建模水下成像、Deblur-NeRF建模模糊核），这不仅需要大量创造性设计，而且只能处理特定退化类型。

**核心矛盾**: 显式建模退化过程导致了**更困难的逆问题**和**零泛化能力**的矛盾——每种新退化都需要重新设计方法，实际应用受限。

**本文目标**: 构建一个能同时泛化到不同场景和不同退化类型的3D复原与渲染框架，无需针对每种退化单独设计或逐场景优化。

**切入角度**: 借鉴2D图像复原领域"all-in-one"方法的成功经验，将退化信息隐式编码到可学习参数中，并结合可泛化NeRF框架（GNT）实现跨场景推理。

**核心 idea**: 用可学习的退化latent codes条件化极线特征聚合和渲染过程，让网络隐式学习不同退化下的图像形成过程。

## 方法详解

### 整体框架

GAURA基于GNT（Generalizable NeRF Transformer）构建，包含三个核心阶段：
1. **特征提取**: UNet从退化输入视角提取卷积特征
2. **极线特征聚合**: View Transformer沿极线聚合多视角特征
3. **光线渲染**: Ray Transformer沿光线累积点特征得到像素颜色

在每个阶段都插入了退化感知模块（DLM），通过退化latent codes动态调整网络行为。

### 关键设计

1. **退化感知Latent模块 (DLM)**: 为$M$种退化类型维护$M$个可学习latent codes $\{\boldsymbol{L}_i\}_{i=1}^M$。受HyperNetwork启发，将latent code映射为MLP权重，对输入特征进行退化特定变换：

$$\text{DLM}(\boldsymbol{X}, \boldsymbol{D}) = W \cdot \boldsymbol{X}, \quad W = \mathcal{F}_{\text{latent}}(\boldsymbol{L}_D | \{\boldsymbol{L}_i\}_{i=1}^M)$$

设计动机：不同退化的图像形成过程存在共性，共享网络主体参数+退化特定latent codes比为每种退化克隆整个网络更高效。

2. **自适应残差模块 (ARM)**: DLM的latent codes独立于实际输入，无法捕获同一退化类型内的变化。ARM从距目标视角最近的输入视角提取残差特征$\boldsymbol{S}$，增强对退化强度变化的适应能力：

$$\text{DLM}_{\text{w/ residue}}(\boldsymbol{X}, \boldsymbol{D}) = \text{DLM}(\boldsymbol{X}, \boldsymbol{D}) + \boldsymbol{S}, \quad \boldsymbol{S} = \text{pool}(\mathcal{F}_{\text{residue}}(\boldsymbol{I}_{\text{nearest}}))$$

其中$\mathcal{F}_{\text{residue}}$是小型卷积网络，pool为全局平均池化。

3. **退化感知的三阶段条件化**: 

    - **特征UNet**: 在decoder每一层的上采样前插入DLM，隐式地用退化信息丰富特征
    - **View Transformer**: 将vanilla MLP的Q/K/V映射全部替换为DLM，在场景表示阶段注入退化先验（因退化导致多视角特征不一致，需要校正）
    - **Ray Transformer**: 仅替换V映射为DLM（因Q/K负责计算注意力分数作为几何blending权重，与退化类型无关）

形式化描述：

$$\boldsymbol{f}(t) = \mathcal{F}_{view}(\{\text{DLM}(\boldsymbol{G})_q, \text{DLM}(\boldsymbol{G})_k, \text{DLM}(\boldsymbol{G})_v\})$$

$$\boldsymbol{c}(r) = \mathcal{F}_{point}(\{\boldsymbol{H}_q, \boldsymbol{H}_k, \text{DLM}(\boldsymbol{H})_v\})$$

### 损失函数 / 训练策略

- **损失函数**: $\mathcal{L} = \mathcal{L}_{\text{MSE}}(\hat{I}_{target}, I_{target}) + \mathcal{L}_{\text{LPIPS}}(\hat{I}_{target}, I_{target})$
- **训练数据**: 使用IBRNet和LLFF训练集，合成生成4种退化（低光照、运动模糊、雾霾、雨），训练时每个迭代随机采样一种退化并随机退化强度
- **初始化**: 复用GNT预训练checkpoint，在此基础上添加退化感知模块
- **优化**: Adam优化器，初始学习率$5 \times 10^{-4}$，共训练400K步
- **推理**: 输入10个源视角，无需任何针对场景的优化
- **微调到新退化**: 仅需增加一个新latent code（<5%额外参数），冻结其余参数，用8个场景微调约10K步

## 实验关键数据

### 主实验（真实世界退化场景）

| 模型 | 泛化(场景/退化) | 低光照 PSNR↑ | 运动模糊 PSNR↑ | 雾霾 PSNR↑ |
|------|:---:|:---:|:---:|:---:|
| NeRF-Restore | ✗/✗ | 15.42 | 23.27 | 13.87 |
| 3D Restore | ✗/✗ | 17.64 | **25.65** | - |
| GNT-Restore | ✓/✗ | 16.36 | 21.97 | 14.16 |
| GNT-(AIO) Restore | ✓/✓ | 17.90 | 20.88 | 16.68 |
| **GAURA (Ours)** | ✓/✓ | **19.91** | 22.12 | **16.82** |

### LLFF-Corrupted基准测试

| 模型 | 低光照 PSNR↑ | 运动模糊 PSNR↑ | 雾霾 PSNR↑ | 雨 PSNR↑ |
|------|:---:|:---:|:---:|:---:|
| GNT-AirNet | 18.20 | 21.08 | 14.55 | 20.71 |
| GNT-PromptIR | 17.67 | 21.01 | 15.81 | 20.73 |
| GNT-DA-CLIP | 12.46 | 21.96 | 8.36 | 20.24 |
| **GAURA** | **21.98** | **22.61** | **18.95** | **22.61** |

### 消融实验（AlethNeRF低光照增强）

| 配置 | PSNR↑ | SSIM↑ | LPIPS↓ | 说明 |
|------|:---:|:---:|:---:|------|
| Vanilla GNT | 17.69 | 0.704 | 0.406 | 无退化感知模块 |
| +DLM in $\mathcal{F}_{conv}$ | 17.98 | 0.680 | 0.427 | 仅特征提取含DLM |
| +DLM in $\mathcal{F}_{conv}$, $\mathcal{F}_{view}$ | 18.73 | 0.727 | 0.394 | +View Transformer |
| +DLM in all three | 19.37 | 0.714 | 0.363 | +Ray Transformer |
| **+ARM (Ours)** | **19.91** | **0.736** | **0.352** | 自适应残差模块 |

### 微调到未见退化

| 模型 | 去雪 PSNR↑ | 散焦去模糊 PSNR↑ |
|------|:---:|:---:|
| Vanilla-GNT | 21.96 | 20.95 |
| GNT-Restore | 20.24 | 21.13 |
| **GAURA (微调)** | **22.61** | **21.34** |

### 关键发现

- GAURA虽然是通用方法，在低光照任务上甚至超过了专门设计的3D Restore方法（AlethNeRF），说明隐式学习优于显式退化建模
- 在LLFF-Corrupted上全面超越所有2D All-in-One + GNT管线，最大提升达10+ dB（去雾任务）
- 微调到新退化仅需<5%参数、8个场景、10K步训练
- 可通过插值latent codes处理多种退化同时出现的情况：$\boldsymbol{L} = \alpha \boldsymbol{L}_{D_1} + (1-\alpha) \boldsymbol{L}_{D_2}$

## 亮点与洞察

- **统一框架的优雅设计**: 将退化信息编码为轻量latent codes，与网络主体解耦，实现了"一个模型处理所有退化"的目标
- **隐式 vs 显式退化建模**: 本文证明了在3D场景复原中，隐式学习退化过程可以超越显式物理建模，与2D图像复原领域的趋势一致
- **View Transformer中Q/K/V全替换 vs Ray Transformer中仅替换V的不对称设计**: 体现了对几何一致性（与退化无关）和外观恢复（与退化有关）的深入理解
- **实用的微调机制**: latent codes的解耦设计使得扩展到新退化类型只需极少数据和计算

## 局限与展望

- 需要预先知道退化类型（非盲复原），未来可探索自动识别退化类型
- 基于极线几何，对稀疏360度场景和复杂光传输场景效果受限
- 可以将退化感知模块集成到3D Gaussian Splatting等更快的表示中
- 缺乏真实世界多退化场景的评测基准

## 相关工作与启发

- GNT为可泛化NeRF提供了强大的Transformer基础架构，本文证明了其适合作为统一复原框架的骨干
- 2D All-in-One复原方法（AirNet, PromptIR）直接应用于多视角导致不一致，说明3D一致性需要在表示层面解决
- HyperNetwork的思路（latent → network weights）在此场景中非常有效
- 未来可结合3DGS的显式表示能力和GAURA的退化感知设计

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个同时泛化场景和退化类型的3D复原方法，DLM+ARM设计巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖6种退化类型、多个基准、完整消融、微调验证、多退化组合
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机阐述充分，公式表述规范
- 价值: ⭐⭐⭐⭐ 对3D场景复原领域有重要启示，统一框架具有很强的实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Flying with Photons: Rendering Novel Views of Propagating Light](flying_with_photons_rendering_novel_views_of_propagating_light.md)
- [\[ECCV 2024\] CaesarNeRF: Calibrated Semantic Representation for Few-Shot Generalizable Neural Rendering](caesarnerf_calibrated_semantic_representation_for_few-shot_generalizable_neural_.md)
- [\[ECCV 2024\] A Direct Approach to Viewing Graph Solvability](a_direct_approach_to_viewing_graph_solvability.md)
- [\[ECCV 2024\] DiffusionDepth: Diffusion Denoising Approach for Monocular Depth Estimation](diffusiondepth_diffusion_denoising_approach_for_monocular_depth_estimation.md)
- [\[ECCV 2024\] TrackNeRF: Bundle Adjusting NeRF from Sparse and Noisy Views via Feature Tracks](tracknerf_bundle_adjusting_nerf_from_sparse_and_noisy_views_via_feature_tracks.md)

</div>

<!-- RELATED:END -->
