---
title: >-
  [论文解读] Extreme Rotation Estimation in the Wild
description: >-
  [CVPR 2025][3D视觉][极端旋转估计] 本文提出了一种面向真实互联网图像的极端三维旋转估计方法，构建了ExtremeLandmarkPairs (ELP)基准数据集，通过渐进式学习方案（全景裁剪→FoV+外观增强→真实数据微调）和辅助通道增强的Transformer模型，在无重叠视角的互联网图像对上显著超越现有方法。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "极端旋转估计"
  - "相机位姿估计"
  - "无重叠视角"
  - "Internet图像"
  - "渐进式学习"
---

# Extreme Rotation Estimation in the Wild

**会议**: CVPR 2025  
**arXiv**: [2411.07096](https://arxiv.org/abs/2411.07096)  
**代码**: [https://tau-vailab.github.io/ExtremeRotationsInTheWild/](https://tau-vailab.github.io/ExtremeRotationsInTheWild/)  
**领域**: 3D视觉  
**关键词**: 极端旋转估计, 相机位姿估计, 无重叠视角, Internet图像, 渐进式学习

## 一句话总结

本文提出了一种面向真实互联网图像的极端三维旋转估计方法，构建了ExtremeLandmarkPairs (ELP)基准数据集，通过渐进式学习方案（全景裁剪→FoV+外观增强→真实数据微调）和辅助通道增强的Transformer模型，在无重叠视角的互联网图像对上显著超越现有方法。

## 研究背景与动机

**领域现状**：估计两张图像之间的相对3D旋转是相机定位和3D重建的基础任务。传统方法依赖像素级对应关系（如SIFT、LoFTR）来计算相对位姿，在有充分重叠的场景下效果很好。近期也有工作（DenseCorrVol、CascadedAtt）开始探索极端无重叠视角下的旋转估计。

**现有痛点**：(1) 传统特征匹配方法在极端视角（少/无重叠）下失效，因为无法提取有效对应关系；(2) 现有极端旋转估计方法（DenseCorrVol、CascadedAtt）仅在从全景图裁剪的受控图像上训练和评估——固定90°视场角、一致光照、一致相机内参——无法泛化到真实互联网照片；(3) 真实互联网图像面临巨大的外观多样性挑战：不同光照、天气、季节、动态物体、不同相机内参。

**核心矛盾**：现有方法在"模拟"极端视角上表现不错，但真实互联网图像的极端性远超裁剪图像——不仅视角极端，外观和相机参数的多样性也极端。同时，互联网图像集中大多数相机拍摄的是重叠区域（因为SfM需要密集图像），真正的无重叠图像对数量有限，训练数据不足。

**本文目标** (1) 如何构建真实互联网极端视角图像对的数据集？(2) 如何让模型从受控的全景裁剪数据逐步泛化到真实互联网数据？(3) 如何在无重叠情况下利用隐含线索（消失点、阴影方向、天际线等）推理相对旋转？

**切入角度**：通过构建MNN（互近邻）图识别以旋转为主的图像对，结合FoV自适应的重叠度判断；用渐进式学习弥补真实数据不足——先在全景裁剪数据学习基础能力，再通过FoV增强和扩散模型外观增强逼近真实数据分布，最后在真实数据上微调。

**核心 idea**：通过渐进式跨域学习方案和辅助通道增强的Transformer，将极端旋转估计从受控全景环境推广到真实in-the-wild互联网照片。

## 方法详解

### 整体框架

输入一对可能无重叠的互联网图像。使用预训练LoFTR提取图像特征，与辅助通道（关键点掩码、匹配掩码、语义分割图）结合后，reshape为token序列。加上可学习的欧拉角位置嵌入后输入旋转估计Transformer（Decoder架构）。输出的欧拉角token与平均图像token拼接后，经三个独立MLP头分别预测roll/pitch/yaw的360-bin概率分布。

### 关键设计

1. **ExtremeLandmarkPairs (ELP)数据集构建**:

    - 功能：提供真实互联网极端视角图像对的训练和评估基准
    - 核心思路：从MegaScenes/MegaDepth/Cambridge Landmarks等互联网图像集出发。(a) 通过构建MNN图（K=5最近邻）识别以旋转运动为主的图像对——密集区域的相邻图像通常平移小旋转为主，而稀疏区域的图像不会被纳入MNN图。(b) 由于真实图像FoV各不相同，不能像全景裁剪那样仅用旋转角判断重叠度。设计了基于FoV自适应的重叠度分类：$|γ| < \frac{fov_x^1+fov_x^2}{4}$ 且 $|β| < \frac{fov_y^1+fov_y^2}{4}$ 为大重叠，相应阈值的2倍以上为无重叠。(c) 限制图像对的FoV差异不超5°，过滤roll>10°的图像，排除航拍视角。最终获得约34K无重叠对。
    - 设计动机：之前的评估基准仅用全景裁剪图像，无法反映真实挑战。ELP提供了两个测试集：sELP（单相机恒定光照）和wELP（真正的in-the-wild互联网照片），支持分层评估。

2. **渐进式学习方案**:

    - 功能：从受控数据逐步泛化到真实互联网数据
    - 核心思路：三阶段训练。Stage 1（初始化）：在StreetLearn全景裁剪数据上训练基础能力（~1M图像对，固定90° FoV）。Stage 2（数据增强训练）：两种增强——(a) FoV增强（$\Delta$FoV）：分析ELP训练集的FoV分布，从$\mathcal{N}(\mu, 1.5\sigma)$采样FoV裁剪全景图像，允许图像对间5°的FoV差异，并使用不同宽高比；(b) 外观增强（$\Delta$Im）：用InstructPix2Pix对部分数据应用"Make it snowy/sunset/night/busy street"等文本提示，生成多样化的外观变体。Stage 3（真实数据微调）：在ELP训练集上微调，优先训练无重叠对。
    - 设计动机：真实极端图像对数据稀缺（仅~34K无重叠对），不足以从头训练。渐进式方案让模型先学习基本的旋转估计能力，再逐步适应FoV多样性和外观多样性，最后适配真实数据分布。消融显示每一阶段都有不可替代的贡献。

3. **辅助通道增强**:

    - 功能：为Transformer提供超越像素外观的结构化推理线索
    - 核心思路：在LoFTR提取的图像特征之外，拼接三类辅助通道：(a) 关键点掩码——标记图像中局部特征点的空间分布；(b) 匹配掩码——标记图像对之间成功匹配的关键点位置，对小重叠情况提供对齐线索；(c) 语义分割图——将图像分割为天空、建筑、道路等类别，帮助识别天际线、动态物体等用于推理无重叠对的隐含线索。
    - 设计动机：对于无重叠图像对，像素级特征几乎无法直接提供旋转信息。但语义线索（如两张图的天际线高度比较）和匹配模式（有匹配说明有重叠，无匹配说明可能无重叠）可以辅助推理。结合预训练LoFTR特征（本身编码了互联网图像对的知识），形成了更强的特征表示。

### 损失函数 / 训练策略

使用交叉熵损失分别训练三个欧拉角（roll、pitch、yaw）的360-bin分类。推理时取最高概率bin作为角度估计。评估时报告Top-1和Top-5预测结果（Top-5考虑yaw预测前5个峰值）。渐进式训练中，先处理无重叠batch再处理有重叠batch。

## 实验关键数据

### 主实验

| 方法 | wELP-None MGE↓ | wELP-None RRA30↑ | wELP-Small MGE↓ | wELP-Large MGE↓ |
|------|----------|------------|----------|----------|
| SIFT | 122.84 | 2.0 | 7.27 | 2.94 |
| LoFTR | 56.54 | 33.0 | 6.80 | 2.13 |
| DenseCorrVol | 82.04 | 13.7 | 125.73 | 120.53 |
| CascadedAtt | 78.60 | 20.8 | 139.14 | 170.62 |
| DUSt3R | 81.21 | 26.9 | 2.80 | 1.01 |
| **Ours** | **26.97** | **50.7** | **4.47** | **2.41** |

wELP测试集（真实in-the-wild图像）。在无重叠对上，本文方法MGE=26.97°远优于所有基线（DUSt3R为81.21°，可能受训练数据影响为灰色标注），RRA30从26.9%提升到50.7%。

### 消融实验

| 训练数据 | wELP-None MGE↓ | wELP-None RRA30↑ | wELP-Small MGE↓ | wELP-Large MGE↓ |
|---------|----------|------------|----------|----------|
| 仅全景裁剪[8] | 74.94 | 25.3 | 55.28 | 13.65 |
| +ΔFoV | 61.62 | 38.4 | 12.91 | 4.61 |
| +ΔIm | 68.31 | 36.1 | 11.46 | 4.46 |
| +ELP (完整) | **26.97** | **50.7** | **4.47** | **2.41** |

### 关键发现

- 在全景裁剪数据上训练的模型（DenseCorrVol、CascadedAtt）在真实互联网图像上严重退化——在wELP-Large上MGE超过120°（随机预测水平），说明从全景到真实的domain gap极大。
- FoV增强和外观增强各自贡献显著：仅加FoV增强使None-MGE从74.94→61.62，仅加外观增强使Small-MGE从55.28→11.46。两者互补。
- ELP真实数据微调带来最大提升：None-MGE从68.31→26.97（降低60%），说明真实数据不可替代。
- 模型仅80M参数，远小于DUSt3R的577M参数，但在无重叠极端场景下性能大幅领先。
- LoFTR作为特征提取器比ImageNet预训练的CNN更有效，因为LoFTR本身就在互联网图像对上训练过特征匹配任务。
- 在全景裁剪图像评估（Table 3）中，本文模型与先前方法性能相当（None RRA10=96.4%），说明泛化到真实数据并未牺牲受控场景的能力。

## 亮点与洞察

- **问题的实际价值很高**：互联网照片集的相对旋转估计是3D重建pipeline中的重要组件。现有方法在有重叠时依赖特征匹配，在无重叠时完全失效。本文填补了这个空白。
- **渐进式学习方案设计精巧**：通过FoV分布匹配和扩散模型外观增强，巧妙地弥合了全景裁剪数据与真实互联网数据之间的domain gap。用InstructPix2Pix做外观增强（如将白天变夜晚、添加雪景）是一个值得借鉴的数据增强技巧。
- **ELP数据集构建过程系统化**：从密集SfM重建中自动提取以旋转为主的图像对，FoV自适应的重叠度分类设计合理，可以为后续研究提供标准化基准。

## 局限与展望

- 在无重叠场景下MGE仍有26.97°（sELP上为13.62°），精度有待进一步提高，尤其是wELP的非重叠对。
- 方法假设图像对之间以旋转运动为主（平移可忽略），对平移运动显著的场景不适用。
- 仅处理室外场景，室内场景的结构线索（天际线等）不适用。
- 过滤掉了roll>10°的图像和大FoV差异的图像对，限制了适用范围。
- 三阶段渐进式训练增加了训练复杂度和调参工作量。

## 相关工作与启发

- **vs DenseCorrVol/CascadedAtt**: 这两种方法是最早的极端旋转估计方法，但完全在全景裁剪数据上训练/测试，在真实互联网图像上MGE超过100°。本文通过渐进式学习和辅助通道使模型适配真实场景。
- **vs DUSt3R/Mast3R**: DUSt3R是强大的通用3D重建方法，在有重叠场景下表现出色。但它基于CroCo预训练（假设有重叠），在无重叠对上MGE=81.21°远不如本文方法。且参数量是本文的7倍。
- **vs LoFTR**: LoFTR在有重叠场景是优秀的特征匹配方法，但在小重叠和无重叠场景无法输出可靠估计。本文使用LoFTR作为特征提取器（而非匹配器），发挥了它编码互联网图像对知识的能力。

## 评分

- 新颖性: ⭐⭐⭐⭐ 问题定义有意义，渐进式学习和数据集构建系统化
- 实验充分度: ⭐⭐⭐⭐⭐ 多个测试集（sELP/wELP/全景裁剪），丰富的消融和基线比较
- 写作质量: ⭐⭐⭐⭐ 结构清晰，数据集构建过程描述详细
- 价值: ⭐⭐⭐⭐ ELP数据集和渐进式学习方案对极端位姿估计研究有推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Zero-Shot Monocular Scene Flow Estimation in the Wild](zero-shot_monocular_scene_flow_estimation_in_the_wild.md)
- [\[CVPR 2025\] Reconstructing Animals and the Wild](reconstructing_animals_and_the_wild.md)
- [\[ICCV 2025\] PersPose: 3D Human Pose Estimation with Perspective Encoding and Perspective Rotation](../../ICCV2025/3d_vision/perspose_3d_human_pose_estimation_with_perspective_encoding_and_perspective_rota.md)
- [\[ICCV 2025\] Amodal Depth Anything: Amodal Depth Estimation in the Wild](../../ICCV2025/3d_vision/amodal_depth_anything_amodal_depth_estimation_in_the_wild.md)
- [\[CVPR 2025\] Generative Multiview Relighting for 3D Reconstruction under Extreme Illumination Variation](generative_multiview_relighting_for_3d_reconstruction_under_extreme_illumination.md)

</div>

<!-- RELATED:END -->
