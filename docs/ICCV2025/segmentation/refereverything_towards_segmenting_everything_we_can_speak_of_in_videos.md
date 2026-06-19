---
title: >-
  [论文解读] ReferEverything: Towards Segmenting Everything We Can Speak of in Videos
description: >-
  [ICCV 2025][语义分割][指代视频分割] 利用视频扩散模型中学到的通用视觉-语言映射，通过保留完整生成模型架构并将目标从预测噪声转变为预测掩码潜变量，实现对视频中任意可用语言描述的概念（包括非物体的动态过程）进行开放世界指代分割。 指代视频分割（Referring Video Segmentation…
tags:
  - "ICCV 2025"
  - "语义分割"
  - "指代视频分割"
  - "视频扩散模型"
  - "开放世界分割"
  - "动态过程分割"
  - "生成式预训练"
---

# ReferEverything: Towards Segmenting Everything We Can Speak of in Videos

**会议**: ICCV 2025  
**arXiv**: [2410.23287](https://arxiv.org/abs/2410.23287)  
**代码**: [项目主页](https://refereverything.github.io/)  
**领域**: 图像分割  
**关键词**: 指代视频分割, 视频扩散模型, 开放世界分割, 动态过程分割, 生成式预训练

## 一句话总结

利用视频扩散模型中学到的通用视觉-语言映射，通过保留完整生成模型架构并将目标从预测噪声转变为预测掩码潜变量，实现对视频中任意可用语言描述的概念（包括非物体的动态过程）进行开放世界指代分割。

## 研究背景与动机

指代视频分割（Referring Video Segmentation, RVS）的目标是根据自然语言描述在视频中分割特定区域。然而现有工作几乎都聚焦于指代视频**物体**分割（RVOS），这源于数据的局限性——RVOS 数据集基于物体跟踪基准构建，天然以物体为中心且规模有限。

本文指出一个关键洞察：自然语言不仅能描述物体，还能精确地描述各种动态过程（如烟雾消散、玻璃破碎、雨滴滚落等）。如果一个事件能用语言表达，那么它在视频中就应该可以被时空定位。然而现有 RVOS 方法在训练分布外的泛化能力极差，无法处理罕见物体和非物体概念。

视频扩散模型在互联网级别数据上进行预训练，学到了从语言描述到视频区域的强大映射关系。先前工作（如 VD-IT）尝试利用扩散模型做特征提取器，但替换了部分生成模型架构会破坏预训练学到的对齐表示，导致泛化能力严重损失。本文的核心动机是：**完整保留生成模型架构是释放最强泛化能力的关键**。

## 方法详解

### 整体框架

REM（Refer Everything with Diffusion Models）的核心思路是：不将视频扩散模型仅作为特征提取器使用，而是保留其完整架构（包括去噪网络和 VAE），仅将输出目标从预测噪声切换为预测掩码潜变量。框架输入为带噪声的视频帧和语言表达，输出为分割掩码的潜变量表示。

### 关键设计

1. **从噪声预测到掩码潜变量预测**：传统方法使用扩散模型的中间特征 $\epsilon_\theta^{(n)}$ 并训练新的解码头 $f_{\text{dec}}$ 来预测掩码。REM 的关键创新是直接复用完整的去噪网络 $\epsilon_\theta$ 和冻结的 VAE 解码器 $\mathcal{D}$，将目标从预测噪声改为预测掩码潜变量：

$$\hat{m} = \mathcal{D}(\epsilon_\theta(z_t, e_c, t))$$

这一看似微小的修改能更好地保留生成预训练中学到的通用视觉-语言映射。设计动机是避免用随机初始化的层替换预训练组件，从而防止预训练表示与新学习特征之间的对齐被破坏。

2. **掩码编码与解码策略**：训练时，将单通道 ground-truth 掩码复制三份形成三通道表示，再通过预训练 VAE 编码器映射到潜空间 $\mathcal{E}(m) = z^m$。推理时，将预测的潜变量通过冻结 VAE 解码器解码为三通道掩码，取三通道平均值并以阈值 0.5 二值化。推理是**非迭代**的（单次前向传播），计算成本与其他方法相当。

3. **两阶段训练协议（ModelScope 版本）/ 单阶段训练（Wan 版本）**：基于 ModelScope-1.4B 时采用两阶段训练：Stage I 在 Ref-COCO 图像-文本对上微调空间权重（1 epoch），Stage II 在 Ref-YTB 视频-文本样本上微调所有权重（40 epoch），辅以伪视频增强。基于 Wan-14B DiT 架构时，因其联合建模时空信息无需分阶段，直接在合并数据上训练 80K iterations。文本编码器和 VAE 全程冻结。

### 损失函数 / 训练策略

训练目标为 $\mathcal{L}_2$ 损失，最小化预测潜变量与 GT 掩码潜变量之间的距离：

$$\min_\theta \mathbb{E}_{z^m \sim \mathcal{E}(m), t=0} \|z^m - \epsilon_\theta(z_t, e_c, t)\|_2^2$$

其中时间步固定为 $t = 0$，优先使用尽可能干净的潜变量，确保模型在最小噪声条件下学习精确的掩码预测。

## 实验关键数据

### 主实验

**标准 RVOS 基准 (Ref-DAVIS & Ref-YTB)**

| 方法 | 预训练 | Ref-DAVIS $\mathcal{J\&F}$ | Ref-YTB $\mathcal{J\&F}$ |
|------|--------|---------------------------|--------------------------|
| Referformer | ImageNet+Kinetics | 61.1 | 62.9 |
| UNINEXT | Object365 | 72.5 | 70.1 |
| VD-IT | LAION5B+WebVid | 69.4 | 66.5 |
| **REM (MS-1.4B)** | LAION5B+WebVid | **72.6** | 68.4 |
| **REM (Wan-14B)** | Internal+Public | **75.0** | **71.7** |

**域外泛化 (BURST & VSPW & Ref-VPS)**

| 方法 | VSPW $\mathcal{J}$ | BURST $\mathcal{J}$ | Ref-VPS $\mathcal{J}$ |
|------|-------------------|---------------------|----------------------|
| UNINEXT | 10.1 | 30.2 | 28.7 |
| VD-IT | 12.7 | 29.0 | 37.9 |
| **REM (MS-1.4B)** | 15.2 | 37.5 | **49.0** |
| **REM (Wan-14B)** | **18.5** | **40.9** | **50.0** |

REM 在域外泛化上大幅领先，尤其在 Ref-VPS 上超越 UNINEXT 达 21.3 个 $\mathcal{J}$ 点。

### 消融实验

| 监督空间 | 分辨率 | 解码器 | Ref-YTB $\mathcal{J\&F}$ | Ref-VPS $\mathcal{J}$ |
|---------|--------|--------|--------------------------|----------------------|
| Latent | 512×512 | VAE (冻结) | 63.5 | **40.0** |
| RGB | 256×256 | VAE (冻结) | 58.4 | 31.6 |
| RGB | 256×256 | VAE (微调) | 60.4 | 32.4 |
| RGB | 512×512 | CNN | 59.6 | 29.4 |
| RGB | 512×512 | MLP | 59.3 | 33.1 |

在潜空间中监督掩码预测是泛化的关键；使用预训练 VAE 解码器优于从头训练的 CNN/MLP 解码器。

### 关键发现

- 保留完整生成模型架构（去噪网络 + VAE）是最大化泛化能力的关键
- 视频扩散模型的进步（如从 MS-1.4B 升级到 Wan-14B）直接改善分割性能
- VD-IT 虽利用了同一扩散主干，但因替换了部分架构导致泛化能力受限
- 在 MeViS 运动引导分割数据集上，REM 也达到 SOTA（60.3 $\mathcal{J\&F}$）

## 亮点与洞察

- **范式转换**：将扩散模型从特征提取器转变为端到端掩码预测器，通过一个优雅的目标切换保留了完整的预训练表示
- **新基准 Ref-VPS**：首次定义了指代视频过程分割任务，覆盖 39 种动态过程概念（烟雾、火焰、破碎等），填补了 RVS 评估的空白
- **扩散模型能力的新证据**：证明了视频扩散模型在互联网级预训练中学到的视觉-语言映射是通用的，可直接迁移到分割任务

## 局限与展望

- 推理效率：扩散模型参数量大（1.4B~14B），实际部署成本高
- 对 "Stuff" 类别的分割精度仍有提升空间（VSPW 上仅 18.5）
- 目前仅在 Ref-COCO 和 Ref-YTB 上微调，数据量有限，更多标注数据可能进一步提升效果
- 未探索多轮交互或更复杂的语言表达

## 相关工作与启发

- VD-IT 同样利用 T2V 模型做 RVOS，但因使用中间特征+新解码头，泛化能力不如 REM
- UNINEXT 统一了多种定位任务，域内性能强但域外泛化弱
- 后续可将此策略推广到图像扩散模型用于静态图像分割，或探索其在 3D 感知任务上的适用性

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 将扩散模型从特征提取器升级为保留完整架构的端到端分割器，改动微小但效果显著
- **实验充分度**: ⭐⭐⭐⭐⭐ 6 个基准全面评估，包含新建的 Ref-VPS 基准，消融实验细致
- **写作质量**: ⭐⭐⭐⭐⭐ 动机清晰，逻辑递进，消融分析深入
- **价值**: ⭐⭐⭐⭐⭐ 为开放世界视频分割提供了新范式，新基准有望推动社区发展

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Open-World Skill Discovery from Unsegmented Demonstration Videos](open-world_skill_discovery_from_unsegmented_demonstration_videos.md)
- [\[ICCV 2025\] Learning Precise Affordances from Egocentric Videos for Robotic Manipulation](learning_precise_affordances_from_egocentric_videos_for_robotic_manipulation.md)
- [\[ICCV 2025\] 2HandedAfforder: Learning Precise Actionable Bimanual Affordances from Human Videos](2handedafforder_learning_precise_actionable_bimanual_affordances_from_human_vide.md)
- [\[ICCV 2025\] Can Generative Geospatial Diffusion Models Excel as Discriminative Geospatial Foundation Models?](can_generative_geospatial_diffusion_models_excel_as_discriminative_geospatial_fo.md)
- [\[CVPR 2025\] Segment Any Motion in Videos](../../CVPR2025/segmentation/segment_any_motion_in_videos.md)

</div>

<!-- RELATED:END -->
