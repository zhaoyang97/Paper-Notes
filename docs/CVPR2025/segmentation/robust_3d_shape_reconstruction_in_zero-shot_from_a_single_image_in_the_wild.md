---
title: >-
  [论文解读] Robust 3D Shape Reconstruction in Zero-Shot from a Single Image in the Wild
description: >-
  [CVPR 2025][语义分割][单视图三维重建] ZeroShape-W 提出了一个遮挡感知的单视图三维形状重建模型，通过联合回归可见掩码、遮挡掩码、深度图和相机内参来估计完整 3D 形状（包括被遮挡部分），同时设计了一个可扩展的合成数据管线来模拟多样化的前景、遮挡物和背景，以仅 194M 参数在 Pix3D 基准上大幅超越了使用 >1100M 参数的 SOTA 方法。
tags:
  - "CVPR 2025"
  - "语义分割"
  - "单视图三维重建"
  - "零样本泛化"
  - "遮挡感知"
  - "合成数据"
  - "域随机化"
---

# Robust 3D Shape Reconstruction in Zero-Shot from a Single Image in the Wild

**会议**: CVPR 2025  
**arXiv**: [2403.14539](https://arxiv.org/abs/2403.14539)  
**代码**: [https://ZeroShape-W.github.io](https://ZeroShape-W.github.io)  
**领域**: 图像分割  
**关键词**: 单视图三维重建, 零样本泛化, 遮挡感知, 合成数据, 域随机化

## 一句话总结

ZeroShape-W 提出了一个遮挡感知的单视图三维形状重建模型，通过联合回归可见掩码、遮挡掩码、深度图和相机内参来估计完整 3D 形状（包括被遮挡部分），同时设计了一个可扩展的合成数据管线来模拟多样化的前景、遮挡物和背景，以仅 194M 参数在 Pix3D 基准上大幅超越了使用 >1100M 参数的 SOTA 方法。

## 研究背景与动机

1. **领域现状**：单视图三维形状重建近年来发展迅速，大规模 3D 数据集（ShapeNet、Objaverse）催生了具有强泛化能力的零样本重建模型（LRM、ZeroShape 等）。这些方法分两类：基于生成模型的方法（如 Zero-1-to-3 扩散多视图合成）和基于回归的方法（如 ZeroShape 单次前向推理）。
2. **现有痛点**：现有 SOTA 方法假设输入是干净的物体分割图像（无背景、无遮挡），因此在真实世界场景中需要先用 SAM 等分割模型处理。这带来两个问题：(a) 分割模型的误差会累积到重建中；(b) 分割会移除被遮挡的物体部分，导致重建不完整。
3. **核心矛盾**：真实图像中物体被遮挡是常态（Pix3D 中 57.4% 的图像含遮挡），但现有方法假设物体完全可见。要处理遮挡需要额外的 amodal completion 模型（如 pix2gestalt），但这进一步增加了误差累积和参数开销。
4. **本文目标**：构建一个端到端的回归模型，直接处理真实世界图像（含背景和遮挡），无需任何外部分割或补全模型。
5. **切入角度**：将分割和重建统一到一个回归框架中，同时显式建模遮挡物的轮廓来帮助推理被遮挡的物体部分。
6. **核心 idea**：联合回归可见掩码+遮挡掩码+深度图+相机内参来构建可见 3D 形状，再利用可见形状和遮挡信息通过 cross-attention 推理完整的 3D 占用场（包括遮挡部分），并设计合成数据管线进行域随机化训练。

## 方法详解

### 整体框架

ZeroShape-W 输入一张包含目标物体的 RGB 图像（224×224），输出物体的完整三维形状（隐式占用场→Marching Cubes 提取网格）。流程分为两阶段：**像素级回归**——使用 DPT-Hybrid 骨干网络从图像提取全局特征和细粒度特征，分别回归相机内参 $K$、深度图 $M_D$、可见掩码 $M_V$（物体可见区域）和遮挡掩码 $M_O$（遮挡物位置）；由 $K$、$M_D$、$M_V$ 通过反投影得到可见 3D 形状 $S_V$。**3D 点级回归**——将 $S_V$ 和 $M_O$ 拼接编码为 key/value，对任意 3D 查询点通过 cross-attention 层预测占用值，恢复完整 3D 形状。可选地利用 VLM 估计物体类别作为先验。

### 关键设计

1. **遮挡感知的像素级回归**:
    - 功能：从单张图像同时回归物体可见区域、遮挡物位置、深度和相机参数
    - 核心思路：使用 DPT-Hybrid 骨干提取全局特征 $X_G$（用于相机内参）和细粒度特征 $X_F$（用于深度/掩码）。可见掩码的回归使用 CLIP 文本编码器提供的类别先验进行特征调制：$\bar{X}_{F_{ij}} = (1+\gamma)X_{F_{ij}} + \beta$，其中 $\gamma, \beta$ 由 CLIP 文本嵌入通过 FFN 估计。遮挡掩码的回归额外利用可见掩码作为输入（拼接到 $X_F$ 通道维度）。可见 3D 形状通过反投影获得：$S_{V_{ij}} = \mathbb{1}_{\{M_{V_{ij}} \geq \eta\}} \cdot (M_{D_{ij}} K^{-1}[i,j,1]^T)$。
    - 设计动机：联合回归共享特征图 $X_F$，使得模型参数极少（194M vs 对手的 >1100M）；CLIP 文本调制帮助在复杂背景或遮挡下更准确地识别目标物体

2. **遮挡感知的 3D 点级回归**:
    - 功能：从可见 3D 形状和遮挡信息推理物体的完整 3D 形状（包括遮挡部分）
    - 核心思路：将 $S_V$（可见 3D 形状）和 $M_O$（遮挡掩码）拼接后编码为 $Z$ 维向量序列作为 cross-attention 的 key/value。每个 3D 查询点构造 query 向量（由点坐标嵌入和 CLIP 文本嵌入拼接），经过 $L=2$ 层 cross-attention 后用 FFN 预测占用值。cross-attention 使得每个查询点能独立关注其相关的空间特征。
    - 设计动机：遮挡掩码告诉模型哪些区域被遮挡，cross-attention 结合学到的 3D 形状先验可以"脑补"被遮挡部分的几何形状——这在之前的回归方法中从未被探索

3. **可扩展的合成数据管线**:
    - 功能：生成大规模、多样化的训练数据，无需真实世界 3D 标注
    - 核心思路：三步流程——(a) **物体渲染**：从 ShapeNet/Objaverse 的 94K 个 3D 物体以多种相机参数渲染，得到 100 万+物体图像及精确的 3D/深度/掩码标注；(b) **外观多样化**：用 ControlNet（以深度图为空间条件、"a [color][material][object]"为文本条件）合成多样化的物体外观。为避免轮廓变形，将渲染图像加噪后作为初始引导（inspired by SDEdit）；(c) **背景多样化**：用 object-aware 背景外扩模型生成不同场景背景，确保不改变物体轮廓；(d) **遮挡增强**：训练时通过 Copy-Paste 在线插入合成前景物体作为遮挡物。
    - 设计动机：真实世界的 3D 标注极度稀缺；传统渲染方法受限于高质量纹理/环境资源；利用生成模型可以模拟几乎无限的外观和背景变化，实现域随机化

### 损失函数 / 训练策略

- **可见掩码和遮挡掩码**：二值交叉熵损失
- **占用值**：二值交叉熵损失
- **深度图**：scale-and-shift-invariant MAE 损失，分两部分——可见区域用 GT 深度、全局区域用 Depth Anything V2 的伪深度（辅助深度损失）
- **相机内参**：MSE 损失（比较可见 3D 形状与 GT）
- 辅助深度损失是关键设计：合成管线仅提供物体区域的深度，辅助损失填补非物体区域的深度监督，防止过拟合

## 实验关键数据

### 主实验

**Pix3D 基准（零样本三维重建）：**

| 模型 | 外部模型 | 参数量 | FS@τ↑ | FS@2τ↑ | CD↓ |
|------|---------|--------|-------|--------|-----|
| LRM (w/ SAM) | SAM | >1100M | 31.0 | 54.5 | 0.121 |
| LRM (w/ SAM+pix2gestalt) | SAM+pix2gestalt | >2400M | 31.1 | 54.9 | - |
| ZeroShape (w/ SAM) | SAM | >800M | 32.1 | 56.8 | 0.116 |
| ZeroShape (w/ SAM+pix2gestalt) | SAM+pix2gestalt | >2100M | 33.6 | 59.0 | 0.110 |
| **ZeroShape-W (Ours)** | **无** | **194M** | **38.2** | **65.3** | **0.097** |

**遮挡/非遮挡分别评估：**

| 模型 | 非遮挡 FS@τ↑ | 非遮挡 CD↓ | 遮挡 FS@τ↑ | 遮挡 CD↓ |
|------|-------------|-----------|-----------|---------|
| LRM (w/ SAM) | 33.5 | 0.111 | 29.1 | 0.128 |
| ZeroShape (w/ SAM) | 34.6 | 0.106 | 30.2 | 0.123 |
| **ZeroShape-W** | **43.6** | **0.082** | **34.2** | **0.107** |

### 消融实验

| 配置 | FS@τ↑ | FS@5τ↑ | CD↓ | 说明 |
|------|-------|--------|-----|------|
| w/o 辅助深度损失 | 35.9 | 90.7 | 0.105 | 非物体区域深度不准 |
| w/o 遮挡物模拟 | 37.7 | 91.2 | 0.102 | 无法重建遮挡部分 |
| w/o 文本提示 | 38.0 | 91.5 | 0.101 | 物体识别受影响 |
| Full (category-specific) | 39.6 | 92.8 | 0.094 | 最优 |
| Full (category-agnostic) | 38.2 | 92.5 | 0.097 | 不需要 VLM |

### 关键发现

- **端到端方法远优于管线方法**：ZeroShape-W 在 FS@τ 上比最强基线（ZeroShape+SAM+pix2gestalt）高 13.7%，同时参数量仅为其 1/12
- **辅助深度损失贡献最大的单项提升**：去掉后 FS@τ 从 39.6 降到 35.9（-3.7），说明全局深度监督对泛化至关重要
- **遮挡增强在遮挡场景中效果显著**：在遮挡图像上 FS@τ 提升 4.0（34.2 vs ~30），但即使在非遮挡场景也有帮助
- **类别先验有帮助但不是必须**：category-specific 比 category-agnostic 高 0.9 FS@τ，说明模型即使不知道物体类别也能很好工作

## 亮点与洞察

- **将分割和重建统一为端到端回归**是最大创新——消除了外部分割模型的误差累积，同时大幅减少了参数量。共享特征图 $X_F$ 用于深度、掩码和遮挡的联合估计是非常优雅的设计
- **遮挡掩码作为 cross-attention 的额外信息源**非常巧妙：它告诉模型"这里有东西挡住了目标物体"，使得模型可以利用学到的形状先验来推理被遮挡部分。这个思路可迁移到任何需要处理遮挡的 3D 任务
- **合成数据管线的初始引导策略**（用渲染图加噪作为 ControlNet 的起点）有效解决了生成模型扭曲物体轮廓的问题，是一个实用的 trick

## 局限与展望

- **仅重建几何形状，不含纹理**：无法输出有纹理的 3D 模型（如 LRM 可以输出 NeRF）
- **输入分辨率受限（224×224）**：可能丢失小物体的细节
- **合成数据与真实数据的域差距**：虽然通过域随机化缓解，但在某些特定领域仍可能不足
- **单物体假设**：仅处理图像中的一个显著物体，无法进行场景级重建
- 改进方向：增加纹理预测分支；提升输入分辨率；扩展到多物体场景重建

## 相关工作与启发

- **vs LRM**: LRM 是基于 Transformer 的大模型方案（>1100M 参数），输出带纹理的 NeRF，但需要 SAM 预处理且参数量大；ZeroShape-W 参数仅为其 1/6 但重建精度更高
- **vs ZeroShape**: ZeroShape 是 ZeroShape-W 的前身，同为回归方法但不处理遮挡和真实背景；ZeroShape-W 通过增加遮挡掩码分支和合成数据管线解决了 in-the-wild 部署问题
- **vs Zero-1-to-3 / Wonder3D 等生成方法**: 生成方法通过扩散合成多视图，质量好但推理开销大；ZeroShape-W 单次前向推理，效率高一个量级
- **vs 域随机化方法**: 传统域随机化随机合成场景，本文通过生成模型有意义地合成外观和背景，更接近真实分布

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个遮挡感知的零样本回归3D重建方法，端到端设计消除了外部模型依赖
- 实验充分度: ⭐⭐⭐⭐ Pix3D 定量+三个额外数据集定性+详尽消融，遮挡/非遮挡单独评估有说服力
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，图示精良，合成数据管线展示直观
- 价值: ⭐⭐⭐⭐⭐ 解决了单视图3D重建真实部署的核心障碍（遮挡和分割依赖），参数量小性能高，实用价值很大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] ZIM: Zero-Shot Image Matting for Anything](../../ICCV2025/segmentation/zim_zero-shot_image_matting_for_anything.md)
- [\[ECCV 2024\] Efficient and Versatile Robust Fine-Tuning of Zero-shot Models](../../ECCV2024/segmentation/efficient_and_versatile_robust_fine-tuning_of_zero-shot_models.md)
- [\[NeurIPS 2025\] HumanCrafter: Synergizing Generalizable Human Reconstruction and Semantic 3D Segmentation](../../NeurIPS2025/segmentation/humancrafter_synergizing_generalizable_human_reconstruction_and_semantic_3d_segm.md)
- [\[CVPR 2025\] Uni4D: Unifying Visual Foundation Models for 4D Modeling from a Single Video](uni4d_unifying_visual_foundation_models_for_4d_modeling_from_a_single_video.md)
- [\[CVPR 2025\] MammAlps: A Multi-view Video Behavior Monitoring Dataset of Wild Mammals in the Swiss Alps](mammalps_a_multi-view_video_behavior_monitoring_dataset_of_wild_mammals_in_the_s.md)

</div>

<!-- RELATED:END -->
