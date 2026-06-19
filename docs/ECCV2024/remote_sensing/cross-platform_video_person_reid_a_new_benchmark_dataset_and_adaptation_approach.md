---
title: >-
  [论文解读] Cross-Platform Video Person ReID: A New Benchmark Dataset and Adaptation Approach
description: >-
  [ECCV 2024][遥感][跨平台行人重识别] 构建首个地面-无人机跨平台视频行人重识别数据集G2A-VReID，并提出VSLA-CLIP方法，通过视觉-语义对齐和参数高效的Video Set-Level-Adapter将CLIP适配到视频ReID任务。 视频行人重识别（VReID）近年来受到广泛关注…
tags:
  - "ECCV 2024"
  - "遥感"
  - "跨平台行人重识别"
  - "视频ReID"
  - "CLIP适配"
  - "视觉-语义对齐"
  - "无人机"
---

# Cross-Platform Video Person ReID: A New Benchmark Dataset and Adaptation Approach

**会议**: ECCV 2024  
**arXiv**: [2408.07500](https://arxiv.org/abs/2408.07500)  
**代码**: [https://github.com/FHR-L/VSLA-CLIP](https://github.com/FHR-L/VSLA-CLIP)  
**领域**: 遥感  
**关键词**: 跨平台行人重识别, 视频ReID, CLIP适配, 视觉-语义对齐, 无人机

## 一句话总结

构建首个地面-无人机跨平台视频行人重识别数据集G2A-VReID，并提出VSLA-CLIP方法，通过视觉-语义对齐和参数高效的Video Set-Level-Adapter将CLIP适配到视频ReID任务。

## 研究背景与动机

视频行人重识别（VReID）近年来受到广泛关注，视频比单帧图像能提供更丰富的外观和时序信息。然而，**现有的VReID数据集和方法几乎全部基于单一平台（地面监控相机）**，在相机视角变化有限的情况下，通过简单的条纹分割即可实现query和gallery视频之间的视觉部件对齐。

在实际应用中存在一个关键场景：当嫌疑人从城市逃入没有部署地面监控的偏远地区时，需要通过无人机（UAV）平台进行跨平台的行人搜索。这一场景带来了**前所未有的挑战**：

**视角剧烈变化**：地面相机约2米高，UAV飞行高度20-60米，俯仰角度差异巨大

**分辨率差异悬殊**：地面相机中行人宽度10-70像素，UAV中仅5-35像素

**严重的自遮挡**：从空中俯拍时行人的外观信息大量丢失，传统时序建模方法反而性能下降

**视觉对齐困难**：跨平台间的视觉特征对齐远比同平台困难

**核心矛盾**：传统方法通过直接对齐视觉部件特征来实现ReID，但在跨平台场景中由于视角和分辨率的巨大差异，视觉层面的直接对齐几乎不可行。

**切入角度**：既然视觉空间的直接对齐困难，那能否借助CLIP等视觉-语言模型，**将跨平台视觉对齐问题转化为视觉-语义对齐问题**？通过学习每个ID的语义描述，让不同平台的视觉特征都对齐到同一语义空间，从而规避直接的跨平台视觉匹配。

**核心Idea**：利用CLIP的视觉-语义对齐能力解决跨平台特征对齐，同时提出参数高效的视频集合级适配器（VSLA），将视频视为无序帧集合而非时序序列，聚合帧间互补信息。

## 方法详解

### 整体框架

方法分为两个主要部分：(1) FT-CLIP基线——通过微调CLIP图像编码器实现视觉-语义对齐；(2) VSLA-CLIP——用参数高效的适配器替代全量微调，同时引入平台桥接提示（PBP）进一步缩小跨平台差距。采用两阶段训练策略：第一阶段学习ID特定描述token和共享文本提示，第二阶段在冻结文本端的情况下训练图像编码器（或适配器）。

### 关键设计

1. **视觉-语义对齐（Visual-Semantic Alignment）**：

   功能：将跨平台视觉对齐转化为视觉-语义对齐。

   核心思路：为每个身份ID学习一组可学习的描述token $[\mathbf{S}]_i$ 和共享文本提示 $[\mathbf{P}]_i$，输入CLIP的Text Encoder生成语义特征 $\mathbf{T}$。然后使用视觉到语义交叉熵损失 $\mathcal{L}_{v2sce}$ 将视频视觉嵌入 $\mathbf{V}_i$ 对齐到语义特征：

    $\mathbf{V}_i = \frac{1}{T}\sum_{j}^{T}\mathbf{E}_i(\mathcal{V}_{ij})$

    $\mathcal{L}_{v2sce}(i) = \sum_{k=1}^{N}-q_k\log\frac{\exp(s(\mathbf{V}_i, \mathbf{T}_{y_k}))}{\sum_{y_j=1}^{N}\exp(s(\mathbf{V}_i, \mathbf{T}_{y_j}))}$

   设计动机：不同平台的视觉特征在视觉空间中差异巨大，但在语义空间中，同一个人的描述应该是一致的。通过语义空间作为桥梁，间接实现跨平台特征对齐。

2. **Video Set-Level-Adapter (VSLA)**：

   功能：参数高效地将预训练图像基础模型适配到视频ReID任务。

   VSLA由两个组件构成：

    - **Intra-Frame Adapter (IFA)**：与ViT每层的MLP并行运行的瓶颈结构，提供帧内外观表示适配：
    $\text{IFA}(\mathbf{x}_i') = \sigma(\mathbf{x}_i'\mathbf{W}_{down})\mathbf{W}_{up}$
      仅占整个Image Encoder参数的5.5%（$\alpha=256$时）。

    - **Cross-Frame Attention Adapter (CFAA)**：跨帧注意力适配器，通过将输入 $\mathbf{x} \in \mathbb{R}^{T\times(N+1)\times\alpha}$ 重塑为 $\mathbb{R}^{(N+1)\times T\times\alpha}$，在帧维度上做注意力，聚合互补信息。

   核心观点：**将视频视为无序帧集合而非时序序列**。航拍视角下时序信息有限（严重自遮挡），互补信息比时序建模更重要。模型对帧顺序不变：$\mathbf{M}(\{\mathcal{V}_{ij}\}) = \mathbf{M}(\{\mathcal{V}_{i\pi(j)}\})$。

3. **Platform-Bridge Prompt (PBP)**：

   功能：进一步弥合地面和航拍平台之间的语义差距。

   核心思路：在Image Encoder的前 $d$ 层MSA中，根据输入来源插入平台特定的可学习提示：

    $f_k(\mathbf{h}, \mathbf{p}_k) = \begin{cases} \text{MSA}_k([\mathbf{h}:\mathbf{p}_k^{ground}]) & \text{if } \mathbf{h} \in Set^{ground} \\ \text{MSA}_k([\mathbf{h}:\mathbf{p}_k^{uav}]) & \text{if } \mathbf{h} \in Set^{uav} \end{cases}$

   设计动机：提示作为显式指令引导预训练模型关注学习平台不变特征，自动抽象出与平台无关的表示。

### 损失函数 / 训练策略

两阶段训练：
- **Stage 1**：冻结Image/Text Encoder，用 $\mathcal{L}_{i2t} + \mathcal{L}_{t2i}$ 优化ID描述token和共享文本提示
- **Stage 2**：冻结Text Encoder和描述token，总损失为：

$$\mathcal{L}_{stage2} = \mathcal{L}_{v2sce} + \beta\mathcal{L}_{tri} + \gamma\mathcal{L}_{id} + \delta\mathcal{L}_{i2t} + \epsilon\mathcal{L}_{t2i}$$

其中 $\beta=1.0, \gamma=0.25, \delta=1.0, \epsilon=1.0$。使用Adam优化器，ViT-Base-16作为Image Encoder，ViFi-CLIP初始化权重。

## 实验关键数据

### 主实验

| 数据集 | 指标 | VSLA-CLIP‡ | 之前SOTA | 提升 |
|--------|------|------------|----------|------|
| MARS | mAP | 88.60 | 87.0 (DenseIL) | +1.60 |
| MARS | Rank-1 | 91.82 | 91.6 (LSTRL) | +0.22 |
| LS-VID | mAP | 85.20 | 82.4 (LSTRL) | +2.80 |
| LS-VID | Rank-1 | 91.66 | 89.8 (LSTRL) | +1.86 |
| iLIDS | Rank-1 | 95.33 | 92.5 (SINet) | +2.83 |
| G2A-VReID | mAP | 79.70 | 76.7 (MGH) | +3.00 |
| G2A-VReID | Rank-1 | 72.55 | 69.9 (MGH) | +2.65 |

### 消融实验

**各组件贡献（在LS-VID和G2A-VReID上）：**

| 配置 | 可调参数(M) | LS-VID mAP | G2A mAP | 说明 |
|------|-----------|------------|---------|------|
| baseline | 86.1 | 76.10 | 72.80 | 仅用triplet+ID loss微调 |
| baseline+VSA (FT-CLIP‡) | 88.0 | 84.07 | 78.11 | 视觉-语义对齐+全量微调 |
| IFA | 4.7 | 77.31 | 73.82 | 仅帧内适配器 |
| IFA+VSA | 6.6 | 84.16 | 79.01 | 适配器+语义对齐 |
| IFA+VSA+CFAA (VSLA-CLIP‡) | 14.5 | 85.20 | 79.70 | 完整适配器 |
| IFA+VSA+CFAA+PBP | 14.5 | - | 81.29 | +平台桥接提示 |

**IFA/CFAA的维度α消融（LS-VID）：**

| α | 可调参数(M) | mAP | Rank-1 |
|---|-----------|-----|--------|
| 64 | 2.6 | 79.58 | 86.71 |
| 128 | 5.6 | 83.64 | 90.00 |
| 256 | 12.6 | 85.20 | 91.66 |

### 关键发现

1. **视觉-语义对齐是最关键的组件**：无论全量微调还是适配器方式，加入VSA后性能提升最显著（LS-VID mAP +7.97/+6.85）
2. **VSLA-CLIP用16.5%的参数超越FT-CLIP**：14.5M vs 88.0M可调参数，LS-VID mAP更高(85.20 vs 84.07)，G2A-VReID mAP更高(79.70 vs 78.11)
3. **时序建模在跨平台场景中表现不佳**：STMN等时序模型在G2A-VReID上表现差于MGH等部件对齐方法
4. **PBP在跨平台数据集上额外带来1.59% mAP提升**（81.29 vs 79.70），但在单平台数据集上无需使用

## 亮点与洞察

- **视角转换很精妙**：将视觉对齐转化为语义对齐，是处理跨模态/跨域对齐问题的通用思路
- **视频=无序集合的观点值得关注**：在航拍等特殊视角下，传统时序建模假设不再合适
- **参数效率极高**：VSLA仅5.5%参数量即可适配基础模型，且集合级注意力天然对帧顺序不变
- **数据集本身具有重要价值**：G2A-VReID是首个地空跨平台视频ReID数据集，包含2788个ID、185907张图像

## 局限与展望

- G2A-VReID数据集规模虽大但场景类型有限（9个类别），未覆盖夜间、恶劣天气
- 仅考虑了两个平台（地面+UAV），未扩展到多平台（如车载、穿戴式相机）
- PBP需要在训练时知道样本来自哪个平台，部署时也需要平台标签
- 可以尝试将VSLA与更大规模的视觉基础模型（如ViT-Large、EVA）结合

## 相关工作与启发

- CLIP-ReID [Li et al.] 首次将CLIP用于图像ReID，本文将其扩展到视频ReID和跨平台场景
- LoRA等参数高效微调方法的思想被借鉴到VSLA设计中，但IFA仅并行于MLP，参数更少
- 集合级视频表示的思想可启发其他视频理解任务（如动作识别中的无序帧聚合）

## 评分

- 新颖性: ⭐⭐⭐⭐ 跨平台数据集+语义对齐思路+集合级适配器，三重贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 四个数据集全面验证，消融充分，SOTA在所有数据集上
- 写作质量: ⭐⭐⭐⭐ 动机清晰，逻辑流畅
- 价值: ⭐⭐⭐⭐ 数据集和方法对跨平台ReID领域有推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Cross-modal Fuzzy Alignment Network for Text-Aerial Person Retrieval and A Large-scale Benchmark](../../CVPR2026/remote_sensing/cross-modal_fuzzy_alignment_network_for_text-aerial_person_retrieval_and_a_large.md)
- [\[CVPR 2026\] WHU-MARS: A Multispectral Aerial-Ground Benchmark Towards Any-Scenario Person Re-Identification](../../CVPR2026/remote_sensing/whu-mars_a_multispectral_aerial-ground_benchmark_towards_any-scenario_person_re-.md)
- [\[CVPR 2026\] Cross-Scale Pansharpening via ScaleFormer and the PanScale Benchmark](../../CVPR2026/remote_sensing/cross-scale_pansharpening_via_scaleformer_and_the_panscale_benchmark.md)
- [\[CVPR 2026\] YieldSAT: A Multimodal Benchmark Dataset for High-Resolution Crop Yield Prediction](../../CVPR2026/remote_sensing/yieldsat_a_multimodal_benchmark_dataset_for_high-resolution_crop_yield_predictio.md)
- [\[ECCV 2024\] Adapting Fine-Grained Cross-View Localization to Areas without Fine Ground Truth](adapting_fine-grained_cross-view_localization_to_areas_without_fine_ground_truth.md)

</div>

<!-- RELATED:END -->
