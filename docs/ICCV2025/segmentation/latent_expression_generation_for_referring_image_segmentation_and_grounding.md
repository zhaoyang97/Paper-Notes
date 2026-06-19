---
title: >-
  [论文解读] Latent Expression Generation for Referring Image Segmentation and Grounding
description: >-
  [ICCV 2025][语义分割][指代图像分割] 提出 Latent-VG 框架，通过从单个文本描述生成多个潜在表达式（共享同一主语、但具有不同视觉属性），利用互补的视觉细节弥补稀疏文本与丰富视觉信息之间的语义差距，在指代图像分割和指代表达理解任务上同时达到 SOTA。 指代图像分割（RIS）和指代表达理解（REC）需要根…
tags:
  - "ICCV 2025"
  - "语义分割"
  - "指代图像分割"
  - "视觉定位"
  - "潜在表达式生成"
  - "对比学习"
  - "多模态融合"
---

# Latent Expression Generation for Referring Image Segmentation and Grounding

**会议**: ICCV 2025  
**arXiv**: [2508.05123](https://arxiv.org/abs/2508.05123)  
**代码**: 无  
**领域**: 图像分割  
**关键词**: 指代图像分割, 视觉定位, 潜在表达式生成, 对比学习, 多模态融合

## 一句话总结

提出 Latent-VG 框架，通过从单个文本描述生成多个潜在表达式（共享同一主语、但具有不同视觉属性），利用互补的视觉细节弥补稀疏文本与丰富视觉信息之间的语义差距，在指代图像分割和指代表达理解任务上同时达到 SOTA。

## 研究背景与动机

指代图像分割（RIS）和指代表达理解（REC）需要根据文本描述定位目标物体。视觉区域可以用多种方式描述——例如同一目标可被称为"穿牛仔裤的男人"、"年长的男人"、"左边的男人"——但模型通常只接收一个简短的文本输入，这仅捕获了目标丰富视觉信息的冰山一角。

本文指出了现有方法的核心问题：

**文本-视觉语义鸿沟**：单一稀疏的文本表达无法涵盖目标区域的全部视觉细节（颜色、纹理、空间位置、上下文关系等），导致模型预测严重依赖特定的文本线索

**相似物体混淆**：当场景中存在相似物体时，稀疏的文本描述容易导致错误定位。例如"穿牛仔裤的男人"可能匹配到同样穿牛仔裤的女人

本文的核心 idea 是：如果我们能从一个文本输入生成**多个多样化的潜在表达式**，每个保持相同主语但突出不同的视觉属性，然后集体利用这些表达式进行预测，就能有效弥合语义差距。关键原则是**共享主语 + 差异化属性（shared-subject and distinct-attributes）**。

## 方法详解

### 整体框架

Latent-VG 基于 BEiT-3 单编码器架构，输入图像和文本后，通过潜在表达式初始化生成多个初始潜在文本序列，然后在编码器的每一层中通过 Subject Distributor 和 Visual Concept Injector 模块强制执行"共享主语+差异属性"原则。最终，所有表达式的预测掩码取平均得到最终分割结果。

### 关键设计

1. **潜在表达式初始化（Latent Expression Initialization）**：从原始文本 token 嵌入出发，生成 $N$ 个初始潜在表达式。每个表达式通过 Latent Attribute Initializer 处理：(a) 以不同概率 $p_i$ 随机丢弃语义 token（减少对原始文本的依赖），(b) 通过长度变换层 $\phi^i \in \mathbb{R}^{k^i \times m}$ 将 token 数从 $m$ 转换为预定义长度 $k^i$。然后自动从文本 token 中选择一个"主语 token" $\mathbf{s}$（通过线性层 + Gumbel-softmax），前置到每个潜在表达式中：

$$\mathbf{Z}_0^i = [\mathbf{z}_{[cls]}^i, \mathbf{s}, \mathbf{A}_0^i] \in \mathbb{R}^{(k^i+2) \times d}$$

同时将主语 token 也注入到视觉序列中替换 visual class token。设计动机：通过不同的 dropout 和长度来产生多样性，通过共享主语 token 保持一致的目标指向。

2. **Subject Distributor（主语分发器）**：在编码器每一层的 self-attention 之后，各潜在表达式中的主语 token 可能已偏离原始主语语义。该模块将所有表达式中的主语 token $\{\mathbf{s}^i\}_{i=1}^N$ 重新替换为视觉域中的主语 token $\mathbf{s}^{\mathcal{V}}$。这确保了在整个编码过程中，所有潜在表达式始终指向同一目标主语，类似于"锚点"机制。设计动机是防止 self-attention 混入后主语指向漂移。

3. **Visual Concept Injector（视觉概念注入器）**：在每一层中将独特的视觉概念注入属性 token，实现差异化属性。具体步骤：(a) 初始化 $N_c$ 个正交概念 token $\mathbf{C}$；(b) 根据文本 class token 的相似度从视觉 patch 中筛选目标相关 patch $\mathbf{V}^{tr}$（阈值取平均值）；(c) 概念 token 通过注意力从目标相关 patch 中检索视觉概念 $\mathbf{C}^{\mathcal{V}} = \mathbf{W} \mathbf{V}^{tr}$；(d) 将视觉概念通过**列归一化注意力**注入属性 token $\tilde{\mathbf{A}}$，使每个属性 token 竞争绑定不同的视觉概念（类似 Slot Attention 机制）。设计动机是从视觉域提取属于目标但不在原始文本中的互补信息。

### 损失函数 / 训练策略

损失函数包含两部分：

1. **正间距对比损失（Positive-Margin Contrastive Loss）**：

$$\mathcal{L}_{\text{pos-cont}} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(\min(1, \gamma + s_i)/\tau)}{\sum_{k \in \mathcal{N}_i}\exp(s_k/\tau)}$$

其中 $s_i = \mathbf{t}_o^\top \mathbf{z}_o^i / (\|\mathbf{t}_o\| \|\mathbf{z}_o^i\|)$，$\gamma$ 是间距参数（设为 0.2）。标准对比学习强制正样本完全对齐，会导致所有潜在表达式坍缩为与原始文本相同的表示。正间距允许正样本之间存在一定差异，在保持语义一致性的同时鼓励多样性。

2. **分割损失**：

$$\hat{\mathbf{p}} = \frac{\sigma(\mathbf{F}^{\mathcal{V}} \cdot \mathbf{t}_o) + \sum_{i=1}^{N}\sigma(\mathbf{F}^{\mathcal{V}} \cdot \mathbf{z}_o^i)}{N+1}$$

对平均预测图应用 BCE + Dice 损失，$\lambda_{\text{bce}}=2$, $\lambda_{\text{dice}}=0.5$。

训练使用 AdamW，batch size 64，4 张 A6000 GPU。推理时所有 dropout 设为 0。

## 实验关键数据

### 主实验

**RIS 结果（Combined Dataset 训练，mIoU）**

| 方法 | 编码器 | RefCOCO val | RefCOCO+ val | RefCOCOg val(U) |
|------|--------|-----------|-------------|----------------|
| PolyFormer | Swin-B + BERT | 75.96 | 70.65 | 69.36 |
| EEVG | ViT-B + BERT | 79.49 | 71.86 | 73.56 |
| One-Ref | BEiT3-B | 79.83 | 74.68 | 74.06 |
| **Latent-VG (本文)** | BEiT3-B | **81.01** | **76.92** | **76.10** |

**GRES 结果（gRefCOCO 数据集）**

| 方法 | val mIoU | val N-acc | testA mIoU | testB mIoU |
|------|---------|----------|-----------|-----------|
| GSVA-7B (SAM+CLIP-L) | 66.47 | 62.43 | 71.08 | 62.23 |
| HDC (Swin-B) | 68.23 | 63.38 | 72.52 | 63.85 |
| **Latent-VG (本文)** | **72.45** | **70.42** | **74.51** | **66.12** |

Latent-VG 在 GRES 上仅需为无目标情况添加一个空 token 即可大幅超越 SOTA。

### 消融实验

**各组件贡献（RefCOCO+ val）**

| 配置 | SD | VCI | mIoU | 提升 |
|------|:---:|:---:|------|------|
| 无潜在表达式（基线） | - | - | 68.76 | +0.00 |
| + 朴素潜在表达式 | × | × | 71.03 | +2.27 |
| + Subject Distributor | ✓ | × | 71.59 | +2.83 |
| + Visual Concept Injector | × | ✓ | 71.86 | +3.10 |
| + SD + VCI | ✓ | ✓ | 72.63 | +3.87 |
| + 正间距对比损失 | ✓ | ✓ | **73.19** | **+4.43** |

**对比学习策略对比**

| 损失函数 | mIoU | oIoU | Pr@0.9 |
|---------|------|------|--------|
| InfoNCE | 72.03 | 69.72 | 24.60 |
| Triplet | 72.57 | 70.42 | 23.55 |
| ArcFace | 72.73 | 70.43 | 24.54 |
| **Positive-margin (本文)** | **73.19** | **70.68** | **26.45** |

### 关键发现

- 潜在表达式数量 $N=2$ 且 token 长度 $\{4, 10\}$ 时性能最优，更多表达式反而带来冗余
- 主语 token 选择有效：自动通过 Gumbel-softmax 从文本中选出核心指代词
- Visual Concept Injector 的列归一化（类似 Slot Attention）是差异化属性的关键，去掉后每个表达式趋向相同
- 在 RefCOCO+ 和 RefCOCOg 上性能最突出，表明我们的方法对复杂文本描述更有效
- REC 任务中，无需任何任务特定解码器，直接从 RIS 掩码提取包围框即超越 LLM-based 方法

## 亮点与洞察

- **原创性强**：从文本增强角度出发解决视觉定位中的语义稀疏问题，潜在空间内的表达式生成是新颖的研究方向
- **统一框架**：同一模型同时刷新 RIS、REC 和 GRES 三个任务的 SOTA，无需任务特定设计
- **正间距对比损失**：优雅地解决了潜在表达式坍缩问题，在保持一致性的同时保留多样性
- **Slot Attention 启发的 VCI**：通过竞争机制使属性 token 自动绑定到不同的视觉概念

## 局限与展望

- 计算增量较小（约 12M 参数、3 GFLOPs），但多个潜在表达式仍增加了推理时间
- 仅使用 BEiT3-B 作为骨干，未验证更大模型或其他架构（如 CLIP-L）
- 视觉概念 token 数 $N_c = 100$ 通过消融确定，缺乏自适应机制
- 可探索将此方法推广到视频场景（视频指代分割）或 3D 场景理解
- 主语 token 选择依赖 Gumbel-softmax，对于没有明确主语的复杂表达可能不够鲁棒

## 相关工作与启发

- CRIS 和 CGFormer 使用像素级对比损失做细粒度对齐，本文的正间距对比损失是对传统对比学习的有趣改进
- SimCSE 通过 dropout 在潜在空间生成文本变体，本文在此基础上增加了视觉条件的属性注入
- One-Ref 作为同骨干 SOTA，Latent-VG 在其基础上持续提升 1-3 个点，验证了潜在表达式生成的价值

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 从文本增强角度解决视觉定位的语义稀疏性，潜在表达式生成 + 共享主语/差异属性设计新颖
- **实验充分度**: ⭐⭐⭐⭐⭐ RIS/REC/GRES 三任务全面验证，消融实验覆盖每个组件和超参数
- **写作质量**: ⭐⭐⭐⭐ 动机图示直观，方法描述清晰，公式推导完整
- **价值**: ⭐⭐⭐⭐⭐ 统一框架刷新三个任务 SOTA，正间距对比损失具有广泛应用潜力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] ReferDINO: Referring Video Object Segmentation with Visual Grounding Foundations](referdino_referring_video_object_segmentation_with_visual_grounding_foundations.md)
- [\[ICCV 2025\] DeRIS: Decoupling Perception and Cognition for Enhanced Referring Image Segmentation through Loopback Synergy](deris_decoupling_perception_and_cognition_for_enhanced_referring_image_segmentat.md)
- [\[NeurIPS 2025\] SaFiRe: Saccade-Fixation Reiteration with Mamba for Referring Image Segmentation](../../NeurIPS2025/segmentation/safire_saccade-fixation_reiteration_with_mamba_for_referring_image_segmentation.md)
- [\[ECCV 2024\] ReMamber: Referring Image Segmentation with Mamba Twister](../../ECCV2024/segmentation/remamber_referring_image_segmentation_with_mamba_twister.md)
- [\[ICCV 2025\] Towards Omnimodal Expressions and Reasoning in Referring Audio-Visual Segmentation](towards_omnimodal_expressions_and_reasoning_in_referring_audio-visual_segmentati.md)

</div>

<!-- RELATED:END -->
