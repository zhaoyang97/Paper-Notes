---
title: >-
  [论文解读] Reanimating Images using Neural Representations of Dynamic Stimuli
description: >-
  [CVPR 2025][医学图像][fMRI脑活动解码] 提出 BrainNRDS 框架，将静态图像表征与运动生成解耦，利用 fMRI 脑活动解码光流信息，结合运动条件扩散模型从初始帧生成视频，同时发现视频编码器（VideoMAE）在预测脑活动方面优于图像编码器。 尽管计算机视觉在静态图像识别上已取得巨大突破…
tags:
  - "CVPR 2025"
  - "医学图像"
  - "fMRI脑活动解码"
  - "光流预测"
  - "视频扩散模型"
  - "动态视觉刺激"
  - "神经表征"
---

# Reanimating Images using Neural Representations of Dynamic Stimuli

**会议**: CVPR 2025  
**arXiv**: [2406.02659](https://arxiv.org/abs/2406.02659)  
**代码**: [项目页面](https://brain-nrds.github.io/)  
**领域**: 图像生成  
**关键词**: fMRI脑活动解码, 光流预测, 视频扩散模型, 动态视觉刺激, 神经表征

## 一句话总结

提出 BrainNRDS 框架，将静态图像表征与运动生成解耦，利用 fMRI 脑活动解码光流信息，结合运动条件扩散模型从初始帧生成视频，同时发现视频编码器（VideoMAE）在预测脑活动方面优于图像编码器。

## 研究背景与动机

尽管计算机视觉在静态图像识别上已取得巨大突破，但在理解复杂动态运动的任务中仍远不及人类。对于需要面对运动丰富环境的具身智能体，理解动态场景尤为关键。人类大脑已经进化出同时处理空间和时间信息的高效机制——例如观看一段行人行走的视频，我们不仅能识别视觉特征，还能推断运动模式、意图和场景元素间的关系。

现有的脑活动解码研究大多聚焦于静态图像重建（如利用 fMRI + Stable Diffusion），而对动态视觉刺激的解码工作相对较少。已有的视频解码方法（如 MindVideo）将静态和动态特征混合建模，缺乏可解释性，且难以精确解码运动信息。直接使用图像条件视频扩散模型（如 SVD）生成视频虽然能产生合理运动，但无法与真实观看到的运动对齐——模型只是在"幻想"合理的光流。

BrainNRDS 的核心切入点是：**显式解耦静态图像表征与运动表征**。通过将运动建模为光流（optical flow），以 fMRI 脑活动为条件预测逐帧光流，再利用运动条件扩散模型（DragNUWA）将初始帧按解码出的运动重新动画化。这种解耦设计不仅提升了运动解码精度，更重要的是提供了可解释性——可以直接将预测的光流与真实光流定量比较，并关联到大脑中特定的运动处理区域。

## 方法详解

### 整体框架

BrainNRDS 的流程分为三个方向：（1）从 fMRI 脑活动解码光流（运动预测）；（2）利用解码的光流通过运动条件扩散模型 DragNUWA 重建视频；（3）通过编码模型分析哪些视觉特征最能预测大脑体素级别的活动。输入为 fMRI 体素数据 $B_i \in \mathbb{R}^n$ 和初始帧图像特征，输出为预测的光流场 $O_i \in \mathbb{R}^{(T-1) \times H \times W \times 2}$。

### 关键设计

1. **光流解码模块（Motion Prediction Module）**:
    - 功能：从 fMRI 脑活动中解码视觉运动信息，预测量化的光流场
    - 核心思路：首先使用 RAFT 从真实视频中提取光流作为训练标签，并通过 k-means 将光流向量量化为 $k=40$ 个聚类的 codebook。模型 $M_\theta$ 以 fMRI 体素 $B_i$（时序窗口 $[i-2, i-1, i, i+1, i+2]$ 共5个TR）和 DINOv2 提取的初始帧特征 $G(\mathcal{I}_{i,1})$ 为输入，通过 MLP 处理 fMRI 数据后空间广播，与图像特征拼接，经过三个残差 1×1 卷积块（含 dropout）和全局平均池化，最终对每个空间 patch 分类到 codebook。推理时对所有类别加权求和恢复连续光流。训练使用交叉熵损失
    - 设计动机：将光流预测重新定义为分类问题（而非回归），借鉴先前工作中分类式光流预测优于回归的发现；通过条件化初始帧图像特征，使模型专注于从 fMRI 中提取动态信息

2. **显式运动-外观解耦（Appearance-Motion Disentanglement）**:
    - 功能：将静态图像表征与动态运动表征在架构上分离，提升可解释性
    - 核心思路：不同于 MindVideo 等方法同时预测静态和动态特征，BrainNRDS 在模型输入端就将初始帧特征（来自冻结的 DINOv2）与 fMRI 信号分开处理。模型通过条件化初始帧来"锁定"外观信息，迫使 fMRI 通道专门学习运动信息。使用 FlowSAM 对显著物体进行遮罩，将评估聚焦在关键运动区域
    - 设计动机：混合建模难以区分静态和动态特征的各自贡献，解耦后可以直接将预测光流与 GT 光流比较，获得可量化的运动解码评估

3. **视频重建与脑区编码分析（Video Reanimation & Brain Encoding）**:
    - 功能：将解码的光流可视化为视频，并识别大脑中对动态特征敏感的区域
    - 核心思路：将预测的光流和初始帧送入预训练的 DragNUWA 模型生成视频。同时，使用多种视觉编码器（VideoMAE、CLIP、DINOv2、VC-1 等）提取特征，训练 Ridge 回归模型逐体素预测 fMRI 响应，通过比较不同编码器的预测性能识别对动态特征选择性高的脑区
    - 设计动机：光流本身难以直观理解，通过扩散模型可视化使结果更易评估；脑区编码分析提供了反向验证——确认 fMRI 中确实包含丰富的动态信息

### 损失函数 / 训练策略

- **交叉熵损失**: 将光流预测作为分类问题，对每个空间patch分类到40个光流codebook类别
- **数据预处理**: fMRI 按 session 零均值单位方差归一化，重复观看的响应取平均
- **时序处理**: 光流降采样为3帧（在2秒TR内等间隔选取），空间降采样至 $32 \times 32$
- **fMRI 窗口**: 利用血流动力学响应特性，拼接当前TR前后各2个TR的数据（共5个TR）
- **编码模型**: 使用 Ridge 回归进行体素级编码预测

## 实验关键数据

### 主实验

光流解码端点误差（End Point Error, EPE↓，越低越好）：

| 方法 | S1 | S2 | S3 | 特点 |
|------|-----|-----|-----|------|
| SVD (Best, 10 samples) | 1.192 | 1.192 | 1.192 | 无脑数据，仅初始帧 |
| MindVideo (Best, 100 samples) | — | — | — | 混合解码 |
| **BrainNRDS (Ours)** | **0.543** | **0.572** | **0.634** | 脑数据+GT初始帧 |

视频生成质量（使用 MindVideo 生成的初始帧）：

| 方法 | VideoMAE CosSim↑ | CLIP CosSim↑ | Pixel SSIM↑ |
|------|-----------------|-------------|-------------|
| MindVideo | 0.742±0.006 | 0.879±0.004 | 0.171±0.02 |
| **Ours end-to-end** | **0.769±0.006** | **0.896±0.003** | **0.214±0.01** |

### 消融实验

| 编码模型类别 | 最佳模型 | S1 Pearson r | S2 Pearson r | S3 Pearson r |
|-------------|---------|-------------|-------------|-------------|
| 视频自监督 | VideoMAE Large | 0.285 | 0.314 | 0.324 |
| 具身AI自监督 | VC-1 | 0.260 | 0.290 | 0.294 |
| 图像语义 | CLIP ConvNeXt | 0.219 | 0.263 | 0.272 |

### 关键发现

- 使用脑数据的模型在 EPE 上显著优于不使用脑数据的 SVD（$p \ll 0.001$），证明 fMRI 中确实包含无法仅从静态图像获得的运动信息
- 视频编码器（VideoMAE Large）全面优于图像编码器在预测 fMRI 响应上的表现，表明 fMRI 中包含丰富的动态信息
- 大脑中体感皮层区域（5m, 5mv, 23c 等）由 VideoMAE 预测显著更好，这些区域整合视觉运动和体感信息

## 亮点与洞察

- **运动-外观解耦的核心思想极具价值**：通过将运动显式表示为光流并与外观分离，不仅提升了解码精度，更重要的是提供了科学上的可解释性——可以精确定位大脑中编码运动信息的区域，为理解动态视觉处理的神经机制提供了工具
- **脑数据消歧的三类场景很有说服力**：论文展示了动作歧义（鹰飞行是滑翔还是扇翅）、相机运动歧义（航天飞机的真实运动方向）、静态物体运动歧义（埃菲尔铁塔的相机平移方向）三种情况，脑数据都能消除歧义而纯图像模型不行
- **VideoMAE 与 CLIP 的脑区差异分析**揭示了视频模型额外捕获的运动和动作表征信息，为选择视觉编码器提供了神经科学层面的指导

## 局限与展望

- 仅在单一数据集（Dynamic Natural Vision）上验证，推广性存疑
- 受试者特异性训练，未探索跨受试者对齐方法
- fMRI 时间分辨率低（2秒 TR），限制了精细运动的解码能力
- 光流空间分辨率较低（$32 \times 32$），限制了细粒度运动还原

## 相关工作与启发

- **vs MindVideo**: MindVideo 将外观和运动混合解码，缺乏可解释性；BrainNRDS 通过解耦建模获得更精确的运动预测和更好的视频生成质量
- **vs Stable Video Diffusion**: SVD 仅条件化初始帧生成视频，无法与真实运动对齐；脑数据提供了关键的运动消歧信息
- **vs NeuroClips**: NeuroClips 使用模糊视频作为运动代理，保留大致场景构图；BrainNRDS 使用光流显式编码运动方向和幅度，与神经运动处理研究更直接相关

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次在脑活动解码中显式解耦外观与运动，将光流作为运动表征的桥梁
- 实验充分度: ⭐⭐⭐⭐ 定量（EPE、CosSim、SSIM）+定性可视化+脑区编码分析+多模型比较
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，动机阐述充分，图示精美
- 价值: ⭐⭐⭐⭐ 为神经科学与计算机视觉的交叉领域提供了新框架和新发现

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] NOIR: Neural Operator Mapping for Implicit Representations](noir_neural_operator_mapping_for_implicit_representations.md)
- [\[CVPR 2025\] Unmasking Biases and Reliability Concerns in Convolutional Neural Networks Analysis of Cancer Pathology Images](unmasking_biases_and_reliability_concerns_in_convolutional_neural_networks_analy.md)
- [\[ICML 2025\] Context Matters: Query-aware Dynamic Long Sequence Modeling of Gigapixel Images](../../ICML2025/medical_imaging/context_matters_query-aware_dynamic_long_sequence_modeling_of_gigapixel_images.md)
- [\[CVPR 2025\] T-FAKE: Synthesizing Thermal Images for Facial Landmarking](t-fake_synthesizing_thermal_images_for_facial_landmarking.md)
- [\[CVPR 2026\] Decoding 3D Perception via BrainSSD: Synergistic Fusion of EEG Representations from Static and Dynamic Visual Streams](../../CVPR2026/medical_imaging/decoding_3d_perception_via_brainssd_synergistic_fusion_of_eeg_representations_fr.md)

</div>

<!-- RELATED:END -->
