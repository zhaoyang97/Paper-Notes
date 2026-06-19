---
title: >-
  [论文解读] A Unified Image-Dense Annotation Generation Model for Underwater Scenes
description: >-
  [CVPR 2025][3D视觉][水下场景] 本文提出TIDE，一种统一的文本到图像和密集标注生成方法，仅以文本为输入就能同时生成高度一致的水下图像、深度图和语义掩码，通过隐式布局共享（ILS）和时间自适应归一化（TAN）机制确保多模态输出的一致性，合成的SynTIDE数据集显著提升了水下深度估计和语义分割性能。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "水下场景"
  - "数据合成"
  - "扩散模型"
  - "深度估计"
  - "语义分割"
---

# A Unified Image-Dense Annotation Generation Model for Underwater Scenes

**会议**: CVPR 2025  
**arXiv**: [2503.21771](https://arxiv.org/abs/2503.21771)  
**代码**: [https://github.com/HongkLin/TIDE](https://github.com/HongkLin/TIDE)  
**领域**: 3D视觉  
**关键词**: 水下场景, 数据合成, 扩散模型, 深度估计, 语义分割

## 一句话总结
本文提出TIDE，一种统一的文本到图像和密集标注生成方法，仅以文本为输入就能同时生成高度一致的水下图像、深度图和语义掩码，通过隐式布局共享（ILS）和时间自适应归一化（TAN）机制确保多模态输出的一致性，合成的SynTIDE数据集显著提升了水下深度估计和语义分割性能。

## 研究背景与动机
水下密集预测（深度估计和语义分割）是水下探索和环境监测的核心技术。然而，高质量、大规模的水下密集标注数据极度稀缺——水下环境复杂、数据采集成本高昂，成为制约技术发展的关键瓶颈。

前人工作Atlantis利用ControlNet以陆地深度图为条件生成水下深度数据，取得了一定效果。但存在两个核心问题：1）使用陆地深度图作为条件是次优方案，生成的数据可能不符合真实水下场景的分布；2）只能生成单一类型的标注（深度图），无法满足水下场景的综合理解需求。

本文的出发点是一个自然的问题：能否只用文本就同时生成高质量的水下图像和多种密集标注？这需要解决的核心难题是：并行生成的图像和标注之间如何保持高度一致性。

## 方法详解

### 整体框架
TIDE基于预训练的PixArt-α文本到图像Transformer构建，并行设置三个去噪分支：text-to-image、text-to-depth、text-to-mask。三个分支共享文本编码器，通过ILS和TAN两个机制实现跨模态对齐。推理时仅需输入文本描述，即可同时输出一致的水下图像、深度图和语义掩码。

### 关键设计

1. **隐式布局共享（Implicit Layout Sharing, ILS）**:

    - 核心观察：在文本到图像模型中，交叉注意力图（cross-attention map）控制着生成图像的布局
    - 将text-to-image分支中计算得到的交叉注意力图$\mathbf{M}_i = \text{softmax}(\mathbf{Q}_i \mathbf{K}_i^\top / \sqrt{c})$直接替换到depth和mask分支中
    - depth和mask分支的交叉注意力简化为$\text{Attn}_d = \mathbf{M}_i \times \mathbf{V}_d$和$\text{Attn}_m = \mathbf{M}_i \times \mathbf{V}_m$
    - 这种设计优雅高效：既保证了布局一致性，又减少了text-to-dense分支的交叉注意力计算量
    - 利用text-to-image模型在大规模数据上预训练获得的强布局控制力

2. **时间自适应归一化（Time Adaptive Normalization, TAN）**:

    - 考虑到不同模态特征之间的互补性，引入跨模态特征交互
    - 将跨模态特征$\mathbf{x}_f$通过MLP映射为两个归一化参数$\gamma$和$\beta$
    - 引入时间嵌入$\mathbf{x}_t$产生自适应系数$\alpha$（通过线性变换+Sigmoid），控制跨模态影响的强度
    - 归一化公式：$\mathbf{x}' = \alpha \cdot \gamma \mathbf{x} + \alpha \cdot \beta$，$\mathbf{x}^* = \mathbf{x}' + \mathbf{x}$（残差连接）
    - 交互方向：depth↔mask双向交互；depth+mask→image的双模态融合（取平均$\bar{\gamma}$和$\bar{\beta}$）
    - TAN与ILS互补：ILS保证宏观布局一致，TAN进一步优化细节层面的特征对齐

3. **数据准备与训练策略**:

    - 基于现有水下分割数据集（SUIM、UIIS、USIS10K）构建约14K个四元组{Image, Depth, Mask, Caption}
    - 深度图由预训练的Depth Anything生成（伪标签）；Caption由BLIP2生成
    - **两阶段训练**：(1) Mini-Transformer预训练：用PixArt-α前10层初始化，在14K图像-文字对上训练60K迭代；(2) TIDE联合训练：用LoRA微调200K迭代，batch size=4
    - LoRA rank分别为text-to-image:32, text-to-depth:64, text-to-mask:64

4. **SynTIDE数据集合成**:

    - 从14K caption中去重得到约5K个非冗余caption
    - 每个caption生成10个样本，构建大规模合成数据集
    - 可生成训练时未见过的水下场景（零样本生成能力，得益于LoRA微调保留了预训练模型的泛化力）

### 损失函数 / 训练策略
总损失为三个分支的去噪MSE损失之和：
$$\mathcal{L} = \mathcal{L}_{mse}^I + \mathcal{L}_{mse}^D + \mathcal{L}_{mse}^M$$

可训练参数仅包括TAN模块和LoRA参数，基础Transformer权重冻结。

## 实验关键数据

### 主实验 — 水下深度估计

| 模型 | 数据集 | 指标 | Atlantis | SynTIDE | 提升 |
|------|--------|------|----------|---------|------|
| NewCRFs | Sea-thru D3+D5 | $SI_{log}$↓ | 37.10 | **22.37** | -14.73 |
| NewCRFs | Sea-thru D3+D5 | $\delta_1$↑ | 0.48 | **0.84** | +0.36 |
| AdaBins | Sea-thru D3+D5 | $SI_{log}$↓ | 38.24 | **26.92** | -11.32 |
| MIM | Sea-thru D3+D5 | $SI_{log}$↓ | 37.01 | **22.49** | -14.52 |
| PixelFormer | SQUID | $SI_{log}$↓ | 21.34 | **19.08** | -2.26 |

### 主实验 — 水下语义分割

| 模型 | 训练数据 | UIIS mIoU | USIS10K mIoU |
|------|----------|-----------|-------------|
| Segformer | Real | 70.2 | 74.6 |
| Segformer | Real+SynTIDE | **75.4(+5.2)** | **76.1(+1.5)** |
| Mask2former | Real | 72.7 | 76.1 |
| Mask2former | Real+SynTIDE | **74.3(+1.6)** | **77.1(+1.0)** |
| ViT-Adapter | Real | 73.5 | 74.6 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 无ILS，无TAN | 一致性低 | 基线并行生成 |
| 有ILS，无TAN | 布局一致 | 宏观对齐有效 |
| 有ILS，有TAN | 一致性最高 | ILS和TAN互补 |

### 关键发现
- SynTIDE在深度估计上全面超越Atlantis，尤其在NewCRFs模型上$SI_{log}$提升14.73
- $\delta_1$指标从0.48提升到0.84（36个百分点），说明合成数据显著提升了模型对水下深度的感知能力
- 单独使用SynTIDE训练分割模型效果接近真实数据，与真实数据联合使用效果最佳
- 零样本生成能力使得TIDE可以生成训练集未覆盖的水下场景

## 亮点与洞察
- 统一框架的设计思路很有前瞻性——一次生成多种标注比分步生成更高效、更一致
- ILS机制的设计非常巧妙：直接复用text-to-image的注意力图，零额外计算开销就获得布局一致性
- TAN引入时间维度的自适应调控，让跨模态交互在不同扩散时间步有不同的影响强度，这个设计很合理
- 仅用14K训练样本+LoRA微调就实现了如此大的性能提升，说明方法有效利用了预训练知识
- 水下场景数据合成的关键洞察：文本条件比深度图条件更灵活，能覆盖更多场景变体

## 局限与展望
- 深度图真值由Depth Anything生成（伪标签），深度精度受限于单目深度估计模型的能力
- 当前仅支持深度和语义掩码两种标注类型，可以扩展到法向量、表面法线等
- 训练数据规模较小（14K），可能限制了生成多样性
- 生成图像的质量和真实感仍依赖于预训练文本到图像模型的能力
- SQUID数据集上部分模型提升较小甚至个别指标略有下降（如S.Rel），说明合成数据分布与某些真实场景仍有差距
- 目前仅验证了水下场景，迁移到其他数据稀缺领域的效果有待验证

## 相关工作与启发
- Atlantis开创了生成式方法解决水下深度数据稀缺的先河，但受限于单一标注类型和陆地深度条件
- FreeMask和SegGen展示了文本条件下的分割数据合成能力，但都是单任务的
- ControlNet系列方法用图像条件控制生成，本文反其道而行之用文本条件实现多标注生成
- PixArt-α的Transformer架构为ILS提供了施展空间——block级别的注意力图共享很自然
- 本文的数据合成范式可以推广到其他数据稀缺领域（如医学影像、遥感等）

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个同时从文本生成图像+多类密集标注的方法，ILS和TAN设计精巧
- 实验充分度: ⭐⭐⭐⭐ 在深度估计和语义分割两个下游任务上充分验证，多模型多数据集
- 写作质量: ⭐⭐⭐⭐ 动机阐述清晰，方法描述直观，图示质量高
- 价值: ⭐⭐⭐⭐ 提供了数据稀缺场景下的有效解决方案，方法可推广性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] UniEgoMotion: A Unified Model for Egocentric Motion Reconstruction, Forecasting, and Generation](../../ICCV2025/3d_vision/uniegomotion_a_unified_model_for_egocentric_motion_reconstruction_forecasting_an.md)
- [\[ICCV 2025\] Relative Illumination Fields: Learning Medium and Light Independent Underwater Scenes](../../ICCV2025/3d_vision/relative_illumination_fields_learning_medium_and_light_independent_underwater_sc.md)
- [\[CVPR 2025\] ODHSR: Online Dense 3D Reconstruction of Humans and Scenes from Monocular Videos](odhsr_online_dense_3d_reconstruction_of_humans_and_scenes_from_monocular_videos.md)
- [\[CVPR 2025\] Wonderland: Navigating 3D Scenes from a Single Image](wonderland_navigating_3d_scenes_from_a_single_image.md)
- [\[CVPR 2025\] MVGenMaster: Scaling Multi-View Generation from Any Image via 3D Priors Enhanced Diffusion Model](mvgenmaster_scaling_multi-view_generation_from_any_image_via_3d_priors_enhanced_.md)

</div>

<!-- RELATED:END -->
