---
title: >-
  [论文解读] Explicitly Guided Information Interaction Network for Cross-modal Point Cloud Completion
description: >-
  [ECCV 2024][3D视觉][点云补全] 提出EGIInet框架，通过统一编码器实现模态对齐，并利用显式引导的信息交互策略（FT-Loss）让网络精准识别图像中的关键结构信息，在视图引导点云补全任务上以更少参数实现了超越XMFnet 16% CD的性能。 领域现状： 点云补全是3D视觉的基础任务…
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "点云补全"
  - "跨模态融合"
  - "视图引导"
  - "多模态对齐"
  - "信息交互"
---

# Explicitly Guided Information Interaction Network for Cross-modal Point Cloud Completion

**会议**: ECCV 2024  
**arXiv**: [2407.02887](https://arxiv.org/abs/2407.02887)  
**代码**: [https://github.com/WHU-USI3DV/EGIInet](https://github.com/WHU-USI3DV/EGIInet)  
**领域**: 3D视觉  
**关键词**: 点云补全, 跨模态融合, 视图引导, 多模态对齐, 信息交互

## 一句话总结

提出EGIInet框架，通过统一编码器实现模态对齐，并利用显式引导的信息交互策略（FT-Loss）让网络精准识别图像中的关键结构信息，在视图引导点云补全任务上以更少参数实现了超越XMFnet 16% CD的性能。

## 研究背景与动机

**领域现状**: 点云补全是3D视觉的基础任务，由于扫描传感器的固有限制，原始点云往往稀疏、有噪声且存在遮挡。视图引导点云补全（ViPC）通过引入单视角图像辅助补全，是更实际的解决方案。

**现有痛点**: 现有多模态融合方法（ViPC、CSDN）依赖结果级融合或从图像直接估计3D坐标（病态问题）；XMFnet虽采用cross-attention进行潜空间融合，但忽略了模态间的固有差异，且缺乏对信息融合过程的显式引导。

**核心矛盾**: XMFnet倾向于从图像中获取抽象的全局语义特征，而忽视了点云补全任务本质上需要的几何结构信息，导致次优的补全结果。

**本文目标**: 如何找到图像中对点云补全最关键的结构信息，并高效地融合到补全过程中。

**切入角度**: 将补全过程拆分为"模态对齐"和"信息融合"两个阶段，分别用统一编码器和显式引导的特征转移损失来处理。

**核心 idea**: 通过Gram矩阵监督的特征转移损失显式引导间接的模态信息交互，让网络自动识别图像中对补全最有价值的结构信息。

## 方法详解

### 整体框架

EGIInet的pipeline包括三个阶段：(1) **Tokenizer** 将图像和点云统一转为token序列；(2) **Shared Feature Extractor (SFE)** 使用共享ViT块进行模态对齐编码；(3) **Shared Feature Transfer Network (SFTnet)** 在FT-Loss的显式引导下进行信息交互；最后通过一层简单的cross-attention进行特征融合，再由解码器输出完整点云。

### 关键设计

1. **统一编码器 (Unified Encoder)**:

    - **功能**: 将不同模态的输入映射到相邻的潜空间，减少模态差距。
    - **核心思路**: 包含Tokenizer和SFE两部分。图像通过大卷积核分割为token，点云通过多步FPS下采样+Ball-query聚类生成token。两种模态的token序列 $\boldsymbol{F}_{pc}, \boldsymbol{F}_{img} \in \mathbb{R}^{N' \times C'}$ 共享同一个基于自注意力的ViT块进行特征提取：$\boldsymbol{F}_{pc}^{stc} = \text{SFE}(\boldsymbol{F}_{pc})$，$\boldsymbol{F}_{img}^{stc} = \text{SFE}(\boldsymbol{F}_{img})$。
    - **设计动机**: 不同的2D/3D骨干网络在潜空间分布和语义结构上存在差异，使用统一的共享架构可以将不同模态的特征映射到相邻的潜空间，简化后续的信息交互。点云token还额外加入了位置嵌入以弥补点云的不规则性。

2. **共享特征转移网络 (SFTnet)**:

    - **功能**: 提供独立于编码阶段的信息交互过程，让点云特征和图像特征在保持各自信息组织模式的前提下间接交互。
    - **核心思路**: SFTnet同样基于共享的ViT块结构，分别处理两个模态的特征：$\boldsymbol{F}_{pc}' = \text{SFTnet}(\boldsymbol{F}_{pc}^{stc})$，$\boldsymbol{F}_{img}' = \text{SFTnet}(\boldsymbol{F}_{img}^{stc})$。关键在于两个模态的特征不直接接触，而是通过FT-Loss在损失层面进行显式引导的信息交互。
    - **设计动机**: 将信息交互从编码过程中分离出来，使网络在不同阶段有特定的学习目标，降低整体优化难度。直接融合会改变特征的组织模式，增加学习负担。

3. **特征转移损失 (FT-Loss)**:

    - **功能**: 通过Gram矩阵监督，显式引导图像和点云特征之间的结构信息转移。
    - **核心思路**: FT-Loss由信息损失 $\mathcal{L}_{infor}$ 和结构损失 $\mathcal{L}_{stc}$ 组成。信息损失通过对齐Gram矩阵实现跨模态结构信息传递：
    $\mathcal{L}_{infor} = \frac{(\boldsymbol{G}(\boldsymbol{F}_{img}^{stc}) - \boldsymbol{G}(\boldsymbol{F}_{pc}'))^2 + (\boldsymbol{G}(\boldsymbol{F}_{pc}^{stc}) - \boldsymbol{G}(\boldsymbol{F}_{img}'))^2}{N \times C}$
      其中Gram矩阵 $\boldsymbol{G}(\boldsymbol{F}) = \boldsymbol{F}^T \cdot \boldsymbol{F}$。结构损失保持点云特征的信息结构不被破坏：$\mathcal{L}_{stc} = (\boldsymbol{F}_{pc}^{stc} - \boldsymbol{F}_{pc}')^2$。
    - **设计动机**: Gram矩阵可以描述特征的通道级全局结构关键性。通过双向的Gram矩阵对齐，可以将点云特征中的缺失关系传递到图像特征，同时将图像中缺失部分的结构信息传递到点云特征。结构损失则确保3D点云特征的信息结构在转移过程中得以保持（因为2D特征难以直接预测3D坐标）。

### 损失函数 / 训练策略

总损失函数：$\mathcal{L}_{total} = \alpha \times \mathcal{L}_{transfer} + \mathcal{L}_{l_1\text{-}CD}$，其中 $\alpha = 0.01$（因为转移损失数值远大于CD损失）。$\mathcal{L}_{transfer} = \mathcal{L}_{infor} + \mathcal{L}_{stc}$。$\mathcal{L}_{l_1\text{-}CD}$ 为标准的L1 Chamfer Distance。

## 实验关键数据

### 主实验

在ShapeNet-ViPC数据集上与现有方法对比：

| 方法 | Avg CD×10³↓ | Airplane | Lamp | F-Score↑ (Avg) |
|------|------------|----------|------|----------------|
| ViPC | 3.308 | 1.760 | 2.867 | 0.591 |
| CSDN | 2.570 | 1.251 | 2.554 | 0.695 |
| XMFnet | 1.443 | 0.572 | 1.810 | 0.796 |
| **EGIInet (Ours)** | **1.211** | **0.534** | **0.776** | **0.836** |

相比XMFnet，CD指标平均下降16%，F-Score提升5%，且参数量更少（9.03M < 9.57M）。在Lamp类别上提升尤为显著（1.810→0.776，降幅57%）。

### 消融实验

| 配置 | Avg CD×10³↓ | 说明 |
|------|------------|------|
| Full model | 1.211 | 完整模型 |
| w/o sharing | 1.429 | 去掉共享结构，CD上升18% |
| w/o FT-Loss | 1.354 | 去掉特征转移损失，CD上升12% |
| w/o SFTnet | 1.454 | 去掉SFTnet（仅用简化损失），CD上升20% |
| w/o image | 1.383 | 去掉图像输入，CD上升14% |

### 关键发现

- SFTnet的独立信息交互过程是性能提升最大的模块（移除后CD上升20%），验证了将信息交互从编码中分离的必要性。
- 共享结构对模态对齐至关重要，移除后虽然参数翻倍但性能显著下降。
- FT-Loss的可视化证明，有监督时图像特征能准确聚焦于对补全有用的结构区域，无监督时则退化为获取抽象全局特征。
- 即使不用图像（w/o image），单靠点云也比XMFnet有竞争力，说明统一编码器和SFTnet本身对点云特征学习也有帮助。

## 亮点与洞察

- **Gram矩阵用于结构信息转移**是本文最巧妙的设计：它将"哪些图像区域对补全重要"这一问题转化为可微分的损失约束，实现了人为可控的信息交互引导。
- **间接交互优于直接融合**：通过shared network + loss supervision的方式实现信息交互，比直接cross-attention融合更有效，因为保持了各模态特征的组织模式。
- 通过attention weight map可视化，直观展示了方法与XMFnet的差异：EGIInet能精准定位图像中的结构边缘，而XMFnet只能获取模糊的全局语义。

## 局限与展望

- 共享结构在参数有限的情况下，对某些复杂类别（如有效像素较少的类别）表现不如非共享模型好，暗示共享结构可能在不同类别间存在优化冲突。
- 仅在ShapeNet-ViPC上测试，真实场景数据的泛化能力有待验证。
- FT-Loss中 $\alpha$ 固定为0.01，未深入讨论其敏感性。
- 仅使用单张视图，未探索多视图融合的可能性。

## 相关工作与启发

- **vs XMFnet**: XMFnet直接堆叠cross-attention实现特征融合，缺乏显式引导，导致网络倾向于学习全局语义而非局部结构。EGIInet通过FT-Loss显式引导，让网络聚焦于关键的几何结构信息。
- **vs CSDN**: CSDN使用IPAdaIN让图像特征影响点云变形过程，是一种隐式融合方式。EGIInet的显式引导策略更透明且效果更好。
- **vs ViPC**: ViPC将图像直接转为骨架点云再拼接，本质是结果级融合，而非特征级融合。

## 评分

- 新颖性: ⭐⭐⭐⭐ 用Gram矩阵实现跨模态信息交互引导的思路很有创意，但整体框架仍是encoder-decoder范式
- 实验充分度: ⭐⭐⭐⭐ 消融实验覆盖了所有关键组件，可视化直观有说服力，但只在一个数据集上评测
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法描述详尽，图示配合文字解释到位
- 价值: ⭐⭐⭐⭐ "显式引导信息交互"的思想可以推广到其他跨模态融合任务，Gram矩阵损失设计具有启发性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] AEDNet: Adaptive Embedding and Multiview-Aware Disentanglement for Point Cloud Completion](aednet_adaptive_embedding_and_multiview-aware_disentanglement_for_point_cloud_co.md)
- [\[AAAI 2026\] STMI: Segmentation-Guided Token Modulation with Cross-Modal Hypergraph Interaction for Multi-Modal Object Re-Identification](../../AAAI2026/3d_vision/stmi_segmentation-guided_token_modulation_with_cross-modal_hypergraph_interactio.md)
- [\[ECCV 2024\] Equi-GSPR: Equivariant SE(3) Graph Network Model for Sparse Point Cloud Registration](equi-gspr_equivariant_se3_graph_network_model_for_sparse_point_cloud_registratio.md)
- [\[ECCV 2024\] SceneGraphLoc: Cross-Modal Coarse Visual Localization on 3D Scene Graphs](scenegraphloc_cross-modal_coarse_visual_localization_on_3d_scene_graphs.md)
- [\[CVPR 2025\] UniPre3D: Unified Pre-training of 3D Point Cloud Models with Cross-Modal Gaussian Splatting](../../CVPR2025/3d_vision/unipre3d_unified_pre-training_of_3d_point_cloud_models_with_cross-modal_gaussian.md)

</div>

<!-- RELATED:END -->
