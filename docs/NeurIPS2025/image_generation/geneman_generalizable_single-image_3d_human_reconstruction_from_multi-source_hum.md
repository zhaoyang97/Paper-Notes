---
title: >-
  [论文解读] GeneMAN: Generalizable Single-Image 3D Human Reconstruction from Multi-Source Human Data
description: >-
  [NeurIPS 2025][图像生成][单图3D人体重建] GeneMAN 提出一种**无需人体参数模型(如 SMPL)**的通用单图 3D 人体重建框架，通过在大规模多源人体数据上训练人体专属的 2D/3D 扩散先验模型，结合几何初始化-雕刻流水线与多空间纹理精炼，实现了对野外图片中不同体型比例、复杂姿态与个人物品的高保真 3D 人体重建。
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "单图3D人体重建"
  - "扩散模型先验"
  - "无模板重建"
  - "多源数据"
  - "Score Distillation Sampling"
  - "纹理优化"
---

# GeneMAN: Generalizable Single-Image 3D Human Reconstruction from Multi-Source Human Data

**会议**: NeurIPS 2025  
**arXiv**: [2411.18624](https://arxiv.org/abs/2411.18624)  
**代码**: [项目页面](https://roooooz.github.io/GeneMAN/)  
**领域**: 图像生成  
**关键词**: 单图3D人体重建, 扩散模型先验, 无模板重建, 多源数据, Score Distillation Sampling, 纹理优化  

## 一句话总结

GeneMAN 提出一种**无需人体参数模型(如 SMPL)**的通用单图 3D 人体重建框架，通过在大规模多源人体数据上训练人体专属的 2D/3D 扩散先验模型，结合几何初始化-雕刻流水线与多空间纹理精炼，实现了对野外图片中不同体型比例、复杂姿态与个人物品的高保真 3D 人体重建。

## 研究背景与动机

### 问题定义

从单张野外(in-the-wild)人体照片重建高保真 3D 人体模型，是 VR/AR、远程呈现、数字人、影视与游戏等应用中的核心需求，但由于 3D 信息严重缺失，一直是一个高度病态(ill-posed)的问题。

### 现有方法的局限

**基于参数模板的方法**（PaMIR、ICON、ECON、SiTH、SIFU、TeCH 等）依赖 SMPL/SMPL-X 等人体参数模型作为几何先验，但这些模型无法表示宽松衣物的 3D 细节，且在姿态/形状估计不准时产生"弯腿"等严重伪影。

**无模板方法**（PIFu、PIFuHD、PHORHUM 等）虽摆脱了参数约束，但因缺乏充分的人体特定先验，在纹理一致性和几何细节方面仍然不佳。

**通用 image-to-3D 方法**（Zero-1-to-3、Magic123 等）在通用物体上表现不错，但缺少人体特异性先验，导致人体几何不准确、面部和衣物细节丢失。

### 三大核心挑战

| 挑战 | 描述 |
|------|------|
| **体型比例多变** | 野外照片包含全身、半身、头肩特写等不同裁切，现有方法主要面向全身重建 |
| **人物携带物品** | 日常照片中人们经常手持物品、站在物体上或佩戴配饰，极大影响重建质量 |
| **自然姿态与纹理一致性** | 缺乏通用的人体专属几何/纹理模型，导致几何不可信且跨视角纹理不一致 |

此外，高质量 3D 人体数据的稀缺性进一步加剧了问题的难度。

## 方法详解

### 整体框架

GeneMAN 的流水线分为四个核心部分：

```
多源数据构建 → 先验模型训练 → 几何初始化与雕刻 → 多空间纹理精炼
```

#### 1. 多源人体数据集构建

为提升泛化能力，作者收集了超过 **50K 多视角实例**，包含四类数据源：

| 数据源 | 具体数据集 |
|--------|-----------|
| 3D 扫描 | RenderPeople、CustomHumans、HuMMan、THuman2.0/3.0、X-Humans、Objaverse 人体子集 |
| 多视角视频 | DNA-Rendering、ZJU-Mocap、AIST++、Neural Actor、Actors-HQ |
| 2D 图片 | DeepFashion、LAION-5B |
| 合成增广 | ControlNet 生成多视角多服装人体数据 + 图像裁切增广（覆盖半身/特写等体型比例） |

#### 2. GeneMAN 先验模型

**2D 先验**：在全部多源人体数据上微调 Stable Diffusion V1.5，同时加入等量 LAION-5B 图像以保持通用能力。用 AdamW (lr=1e-5)，4×A100 训练 5 天。提供丰富的人体几何与纹理细节。

**3D 先验**：在 3D 扫描 + 多视角视频 + 合成数据 + DeepFashion 图片上微调 Zero-1-to-3，额外加入 20% Objaverse 数据以保持物体重建能力。用 AdamW (lr=1e-4)，8×A100 训练 1 周。确保多视角一致性。

### 关键设计

#### 几何初始化与雕刻 (Geometry Initialization & Sculpting)

**阶段一：NeRF 初始化**

- 使用 Instant-NGP 作为 NeRF 实现，从分辨率 256 逐步提升到 384，训练 5000 步
- 参考视角监督损失 $\mathcal{L}_{\text{ref}}$：RGB MSE + 掩码 MSE
- 深度/法线先验：由人体基础模型 **Sapiens** 推理，分别施加深度损失（归一化负 Pearson 相关）和法线损失（MSE）
- 新视角引导：混合 SDS 损失

$$\mathcal{L}_{\text{guid}} = \mathcal{L}_{\text{2D-SDS}}(\phi_{2d}, g(\theta)) + \mathcal{L}_{\text{3D-SDS}}(\phi_{3d}, g(\theta))$$

**阶段二：DMTet 雕刻**

- 将 NeRF 转换为显式网格，作为 DMTet（混合 SDF-Mesh 表示）的初始化
- 在分辨率 512 下优化 3000 步，使用 MSE + 感知损失监督法线，并引入 HumanNorm 预训练的法向/深度自适应扩散模型作为新视角引导
- SDF 正则化损失 $\mathcal{L}_{\text{sdf}}$ 防止几何偏离过大

#### 多空间纹理精炼 (Multi-Space Texture Refinement)

**潜空间优化（Coarse Texture）**：

- 利用混合 SDS 损失（2D + 3D 先验）优化纹理表示，10000 步
- 参考视角 MSE 损失保证输入图像一致性
- **训练免费多视角策略**：对不同视角渲染的图像加相同高斯噪声，拼接为单张图推理，无需重训扩散模型即可保证视角间纹理一致

$$\mathcal{L}_{\text{coarse}} = \lambda_{\text{ref}}^c (\|{\hat{I}} - g_c(\theta;\hat{c})\|_2 + \|\hat{m} - g_c(\theta;\hat{c})\|_2) + \lambda_{\text{guid}}^c \mathcal{L}_{\text{guid}}^c$$

**像素空间优化（Fine Texture）**：

- 使用 SDEdit 框架：渲染粗纹理图 → 加噪 → 用 GeneMAN 2D prior + ControlNet 多步去噪得到精细图
- UV 贴图优化 1000 步，损失函数为 MSE + LPIPS：

$$\mathcal{L}_{\text{fine}} = \|I_{\text{fine}} - I_{\text{coarse}}\|_2 + \lambda_{LP} \cdot \text{LPIPS}(I_{\text{fine}}, I_{\text{coarse}})$$

### 损失函数

完整训练涉及的损失函数汇总：

| 阶段 | 损失函数 | 作用 |
|------|---------|------|
| NeRF 初始化 | $\mathcal{L}_{\text{ref}}$ (RGB+mask) | 参考视角重建 |
| NeRF 初始化 | $\mathcal{L}_{\text{depth}}$ (Pearson) | 深度一致性 |
| NeRF 初始化 | $\mathcal{L}_{\text{normal}}$ (MSE) | 法线一致性 |
| NeRF/DMTet | $\mathcal{L}_{\text{guid}}$ (2D+3D SDS) | 新视角人体先验引导 |
| DMTet 雕刻 | $\mathcal{L}_{\text{sdf}}$ | 几何正则化 |
| 粗纹理 | $\mathcal{L}_{\text{coarse}}$ (ref+SDS) | 一致性纹理学习 |
| 细纹理 | $\mathcal{L}_{\text{fine}}$ (MSE+LPIPS) | 高保真纹理精炼 |

## 实验关键数据

### 主实验：定量比较

测试集：50 个样本，来自互联网 (in-the-wild) 和 CAPE 数据集，每个方法渲染 360° 共 120 个视角。

| 方法 | in-the-wild PSNR↑ | LPIPS↓ | CLIP-Sim↑ | CAPE PSNR↑ | LPIPS↓ | CLIP-Sim↑ |
|------|:--:|:--:|:--:|:--:|:--:|:--:|
| PIFu | 26.97 | 0.035 | 0.594 | 26.91 | 0.028 | 0.764 |
| GTA | 25.06 | 0.064 | 0.568 | 30.38 | 0.019 | 0.785 |
| TeCH | 25.74 | 0.053 | 0.713 | 27.60 | 0.025 | 0.826 |
| SiTH | 20.41 | 0.129 | 0.608 | 21.99 | 0.048 | 0.815 |
| **GeneMAN** | **32.24** | **0.013** | **0.730** | 28.49 | **0.015** | **0.838** |

- 在野外图像上，GeneMAN 的 PSNR 比第二名高约 **5.3 dB**，LPIPS 降低 **63%**
- CLIP-Sim 最高，表明多视角一致性最优
- 在 CAPE 上 LPIPS 和 CLIP-Sim 均最优，PSNR 有竞争力

### 用户研究

40 人参与，30 个测试案例，共 1200 次对比。**73.08%** 的参与者偏好 GeneMAN 的重建结果（几何+纹理综合评价），远超所有基线方法。

### 消融实验

| 消融项 | 关键发现 |
|--------|---------|
| 几何初始化 vs 雕刻 | DMTet 雕刻平滑了过噪表面，恢复了衣物褶皱、面部特征等高频细节 |
| 潜空间纹理 vs 像素空间精炼 | 潜空间纹理基本合理但背部视角不一致且略模糊，像素空间优化显著提升细节 |
| GeneMAN 2D 先验 vs 原始 DeepFloyd-IF | 原始 2D 先验导致衬衫下摆前后不一致，GeneMAN 2D 先验确保多视角一致 |
| GeneMAN 3D 先验 vs 原始 Zero-1-to-3 | 原始 3D 先验产生不自然姿态（颈部前倾），GeneMAN 3D 先验捕获更自然的体态 |

### 关键发现

- 无模板设计使 GeneMAN 可有效处理宽松衣物（裙子、连衣裙）和个人物品（篮球等），避免了 SMPL 估计失败带来的级联误差
- 收集大规模多源数据并微调扩散模型作为人体先验，是泛化能力的关键来源
- 多视角一致的训练免费策略（共享噪声+拼接推理）在不增加训练开销的前提下有效提升视角一致性

## 亮点与洞察

1. **数据驱动的先验学习**：不依赖手工设计的参数模型，而是通过 50K+ 多源数据学习 2D/3D 人体先验，这是一种更可扩展的范式——数据越多先验越强。
2. **混合先验互补设计**：2D 先验提供细节（纹理、细粒度几何），3D 先验保证一致性（多视角、自然姿态），二者通过 SDS 损失有机融合，各司其职。
3. **从粗到精的分层策略**贯穿始终：NeRF → DMTet 实现几何由粗到细，潜空间 → 像素空间实现纹理由粗到细，每一步都用最合适的表示和监督方式。
4. **无模板（template-free）设计**打破了对 SMPL 精确估计的依赖，使框架天然适应儿童、非标准体型、物品遮挡等现有方法难以应对的场景。
5. **训练免费的多视角一致性策略**是一个巧妙的工程设计——对批量渲染图加相同噪声并拼接推理，无需重训模型即可提升跨视角纹理一致性。

## 局限性

1. **推理速度慢**：完整流水线约 1.4 小时（单张 A100 80G），包含 NeRF 优化、DMTet 雕刻和两阶段纹理优化，远不能满足交互式应用需求。
2. **训练资源昂贵**：3D 先验需 8×A100 训练一周，2D 先验需 4×A100 训练五天，门槛较高。
3. **仍依赖 SDS 优化范式**：SDS 本身存在过度饱和(over-saturation)和模式坍缩等已知问题，虽然通过混合先验缓解，但未从根本上解决。
4. **缺少手部/面部定量评估**：论文的定量指标（PSNR/LPIPS/CLIP-Sim）是全局性的，未专门评估面部表情和手部姿态的重建精度。
5. **数据集偏置风险**：合成数据由 ControlNet 生成，可能引入特定的分布偏差；多源数据间的域差异如何平衡未做深入分析。

## 相关工作与启发

- **LRM (Hong et al., 2023)**：展示了多源数据训练的 transformer 3D 重建器具有强泛化能力，启发了 GeneMAN 的数据策略
- **DreamCraft3D (Sun et al., 2023)**：结合 2D+3D 先验的粗到细优化，GeneMAN 延续并深化了这一思路
- **HumanNorm (Huang et al., 2024)**：预训练法线/深度自适应扩散模型，被 GeneMAN 直接用于 DMTet 雕刻阶段的指导
- **Sapiens (Khirodkar et al., 2024)**：人体基础模型提供法线和深度先验，作为 GeneMAN 的参考视角几何信号
- 整体技术路线可看作 **"人体特化的 DreamCraft3D"**，核心创新在于用多源人体数据替代通用数据来训练更强的域特定先验

## 评分

| 维度 | 分数 (1-5) | 评析 |
|------|:---------:|------|
| 创新性 | 3.5 | 框架各组件（SDS、NeRF→DMTet、SDEdit 纹理精炼）均为现有方法的人体特化版本，核心创新在于多源数据驱动的先验学习和无模板设计 |
| 技术深度 | 4.0 | 系统设计扎实，多阶段流水线中每个模块的选型与损失设计都有合理考量 |
| 实验充分度 | 4.0 | 定量/定性/用户研究/消融实验齐全，野外图像测试覆盖多种挑战场景 |
| 写作质量 | 4.0 | 结构清晰，图示丰富，方法描述详尽 |
| 实用价值 | 3.5 | 效果出色但推理耗时长，需优化到可交互水平才能落地 |
| **总分** | **3.8** | 一篇扎实的系统性工作，通过多源数据+双先验+多阶段优化将单图人体重建推向新水平 |

## 与相关工作的对比

| 方法 | 类型 | 人体先验 | 纹理优化 | 优势 | 局限 |
|------|------|---------|---------|------|------|
| PIFu / PIFuHD | Template-free | 像素对齐隐式场 | 无 | 端到端、不依赖 SMPL | 侧面几何差、纹理不真实 |
| PaMIR / ICON / ECON | Template-based | SMPL 特征 | 无/有限 | 利用人体结构信息 | 衣物细节受限于 SMPL 拓扑 |
| SiTH | Template-based | 微调 diffusion 幻想背面 | SDEdit | 速度较快 | 依赖 HPS 精度，wild 图像泛化差 |
| TeCH | Template-based | SDS 优化 | 扩散精炼 | 细节丰富 | 表面过度嘈杂，纹理不一致 |
| GTA | Template-based | Transformer triplane | 前馈 | 推理快 | 依赖 SMPL，宽松衣物失败 |
| HumanLRM | Template-free | 扩散引导前馈 | 隐式场 | 无需 SMPL、前馈 | 纹理不一致、几何细节不足 |
| **GeneMAN** | **Template-free** | **多源数据微调的 2D+3D 扩散先验** | **潜空间+像素空间** | **泛化强、体型/物品鲁棒** | **推理慢 (~1.4h)** |

**核心差异**：GeneMAN 同时做到了 template-free（避免 SMPL 级联误差）和 prior-rich（50K 多源数据训练的扩散先验），这是其在 wild 图像上大幅超越 template-based 方法的根本原因。

## 启发与关联

1. **数据 > 架构**：GeneMAN 的各模块（NeRF、DMTet、SDS、SDEdit）都是标准组件，核心竞争力来自 50K+ 多源人体数据训练的域特定先验。这启示我们在垂直领域做 3D 重建时，高质量域数据的收集和清洗可能比架构创新更关键。
2. **混合先验范式的通用性**：2D prior 提供细节 + 3D prior 保证一致性的双先验设计，可迁移到其他特定域（动物、车辆、建筑等）的单图 3D 重建任务。
3. **与 feed-forward 方法的互补**：GeneMAN 走优化路线（~1.4h），质量上限高但速度慢；LRM/Instant3D 等前馈方法速度快但质量有限。未来可能的方向是用 GeneMAN 级别的先验初始化前馈网络，兼顾质量与速度。
4. **Sapiens 作为人体基础模型的潜力**：论文中使用 Sapiens 提供参考视角的 depth/normal 先验，提示 foundation model 在下游 3D 任务中可作为即插即用的几何信号源。
5. **训练免费多视角一致性技巧**：对多视角渲染加相同噪声并拼接推理，这一简洁技巧可能适用于其他基于 SDS 的 3D 生成任务。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] DeClotH: Decomposable 3D Cloth and Human Body Reconstruction from a Single Image](../../CVPR2025/image_generation/decloth_decomposable_3d_cloth_and_human_body_reconstruction_from_a_single_image.md)
- [\[NeurIPS 2025\] A Data-Driven Prism: Multi-View Source Separation with Diffusion Model Priors](a_data-driven_prism_multi-view_source_separation_with_diffusion_model_priors.md)
- [\[CVPR 2025\] InterEdit: Navigating Text-Guided Multi-Human 3D Motion Editing](../../CVPR2025/image_generation/interedit_navigating_text-guided_multi-human_3d_motion_editing.md)
- [\[CVPR 2025\] Comprehensive Relighting: Generalizable and Consistent Monocular Human Relighting and Harmonization](../../CVPR2025/image_generation/comprehensive_relighting_generalizable_and_consistent_monocular_human_relighting.md)
- [\[NeurIPS 2025\] Diff-ICMH: Harmonizing Machine and Human Vision in Image Compression with Generative Prior](diff-icmh_harmonizing_machine_and_human_vision_in_image_compression_with_generat.md)

</div>

<!-- RELATED:END -->
