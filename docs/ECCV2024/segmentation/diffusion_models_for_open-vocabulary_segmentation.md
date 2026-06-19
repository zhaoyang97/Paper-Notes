---
title: >-
  [论文解读] Diffusion Models for Open-Vocabulary Segmentation
description: >-
  [ECCV 2024][语义分割][开放词汇分割] 本文提出 OVDiff，利用预训练的文本到图像扩散模型为任意文本类别生成支持图像集，从中提取多层次原型（类级、实例级、部件级），结合背景原型实现无训练的开放词汇语义分割，在 PASCAL VOC 上超越先前方法 10% 以上。 领域现状：开放词汇分割要求对图像中任何可用自然…
tags:
  - "ECCV 2024"
  - "语义分割"
  - "开放词汇分割"
  - "扩散模型"
  - "原型学习"
  - "无训练分割"
  - "文本到图像生成"
---

# Diffusion Models for Open-Vocabulary Segmentation

**会议**: ECCV 2024  
**arXiv**: [2306.09316](https://arxiv.org/abs/2306.09316)  
**代码**: 无  
**领域**: 分割 / 开放词汇分割  
**关键词**: 开放词汇分割, 扩散模型, 原型学习, 无训练分割, 文本到图像生成

## 一句话总结

本文提出 OVDiff，利用预训练的文本到图像扩散模型为任意文本类别生成支持图像集，从中提取多层次原型（类级、实例级、部件级），结合背景原型实现无训练的开放词汇语义分割，在 PASCAL VOC 上超越先前方法 10% 以上。

## 研究背景与动机

**领域现状**：开放词汇分割要求对图像中任何可用自然语言描述的物体进行像素级分割。近年来，大规模视觉-语言模型（如 CLIP）推动了该领域的显著进展，但这些方法需要大量图像-文本对的对比训练，代价高昂。

**现有痛点**：(1) 基于对比训练的方法（如 GroupViT、TCL）需要海量图像-文本对进行额外训练，且训练过程中存在噪声——文本可能不完全描述图像内容。(2) 跨模态特征对齐本身存在歧义——视觉外观相似的图像可能对应不同文本描述，反之亦然。(3) 背景分割是一个长期难题，通常需要设定相似度阈值，但最优阈值难以确定且可能因图像而异。

**核心矛盾**：开放词汇分割需要将语言表达的任意类别映射到精确的像素级分割，但直接学习跨模态对齐既昂贵又容易产生歧义。能否利用已有的预训练基础模型，按需合成分割器，避免额外训练？

**切入角度**：作者观察到同一模态内的特征比较比跨模态比较更容易且更可靠。因此，提出用生成式扩散模型将语言查询"翻译"为视觉原型，从而将跨模态问题转化为同模态比较问题。扩散模型不仅能编码物体的视觉外观分布，还能提供上下文先验（如背景），这对分割质量至关重要。

**核心 idea**：用扩散模型为文本类别生成支持图像，提取多层次视觉原型，通过同模态最近邻查找实现无训练的开放词汇分割。

## 方法详解

### 整体框架

OVDiff 分为两个阶段：(1) 原型采样阶段——给定文本查询，用 Stable Diffusion 生成支持图像集，通过预训练特征提取器和无监督分割器将其分解为正向（前景）和负向（背景）原型；(2) 分割阶段——提取目标图像的特征，与预计算的原型进行余弦相似度比较，为每个像素分配类别。整个流程无需任何训练，仅依赖预训练组件。

### 关键设计

1. **支持集生成与多层次原型提取（Multi-level Prototype Extraction）**:

    - 功能：为任意文本类别构建丰富的视觉原型表示
    - 核心思路：对每个查询 $c_i$，用 Stable Diffusion 生成 $N=64$ 张支持图像。利用扩散过程的交叉注意力图获取归因映射，结合 CutLER 等无监督实例分割方法生成前景/背景掩码。在掩码区域内用预训练特征提取器提取特征，构建三类原型：类级原型（所有实例的加权平均）、实例级原型（每张支持图像一个）、部件级原型（通过 $K$-Means 聚类获得 $K=32$ 个聚类中心）
    - 设计动机：单一原型无法捕捉类内变异。类级原型提供全局表示，实例级原型覆盖不同样本的外观，部件级原型关注物体局部特征（如狗的鼻子、脖子等），三者结合确保高质量分割

2. **背景原型机制（Background Prototypes）**:

    - 功能：直接分割背景区域，避免阈值设定问题
    - 核心思路：在支持图像中不仅提取前景原型，还提取背景区域的原型。分割时引入一个额外的"背景"类，其前景原型定义为所有类别的背景原型的并集 $\mathcal{P}_{c_{\text{bg}}}^{\text{fg}} = \bigcup_{c_i \in \mathcal{C}} \mathcal{P}_{c_i}^{\text{bg}}$。这样背景区域可以直接通过原型匹配被识别，无需设定阈值
    - 设计动机：扩散模型生成的图像天然包含类别相关的背景上下文（如船的图像通常包含水面和天空），这提供了宝贵的上下文先验。消融实验显示，去除背景原型后 VOC 上 mIoU 下降 10 个点

3. **类别预过滤与stuff过滤（Category Pre-filtering & Stuff Filtering）**:

    - 功能：减少错误匹配，提升分割精度
    - 核心思路：利用 CLIP 对目标图像进行多标签分类预过滤，保留候选类别并组合为多标签提示，选择最佳匹配的类别集 $\mathcal{C}' \subseteq \mathcal{C}$。对于"stuff"类（如天空、水面），额外过滤可能与其他类背景原型冲突的原型。"thing/stuff"分类通过 ChatGPT 自动完成
    - 设计动机：当类别数量较多时，不同类别的原型可能产生虚假关联。预过滤将候选类别限制在合理范围内。stuff 过滤解决了一类图像的背景可能是另一类的前景的问题

### 损失函数 / 训练策略

OVDiff 是一个完全无训练的方法。分割通过余弦相似度最近邻查找完成：$M = \arg\max_{c \in \hat{\mathcal{C}}} \max_{P \in \mathcal{P}_c^{\text{fg}}} s(\Phi_v(I), P)$。支持图像使用 Stable Diffusion v1.5 生成，classifier-free guidance scale 为 8.0，30 步去噪。特征提取器使用 SD + DINO + CLIP 的集成，取三者余弦距离的平均值。

## 实验关键数据

### 主实验

| 数据集 | 指标 | OVDiff | OVDiff+PAMR | 之前SOTA (TCL) | 之前SOTA+PAMR | 提升 |
|--------|------|--------|------------|--------|--------|------|
| PASCAL VOC | mIoU | 67.1 | 69.0 | 51.2 | 55.0 | +14.0 |
| Pascal Context | mIoU | 30.1 | 31.4 | 24.3 | 30.4 | +1.0 |
| COCO-Object | mIoU | 34.8 | 36.3 | 30.4 | 31.6 | +4.7 |

### 消融实验

| 配置 | VOC mIoU | Context mIoU | 说明 |
|------|---------|-------------|------|
| Full (SD only) | 63.6 | 29.8 | 完整方法 |
| w/o 背景原型 | 53.6 (-10.0) | 28.3 (-1.5) | 背景原型贡献最大 |
| w/o 类别预过滤 | 54.9 (-8.7) | 26.4 (-2.4) | 预过滤同样关键 |
| w/o stuff过滤 | n/a | 27.3 (-2.5) | 对多stuff类数据集重要 |
| w/o CutLER | 60.6 (-3.0) | 27.9 (-1.9) | 无监督分割提升掩码质量 |
| 仅平均原型 | 62.5 (-1.1) | 29.0 (-0.8) | 多层次原型有正向贡献 |

### 关键发现

- 背景原型和类别预过滤是性能提升的两个最大贡献因素
- 不同特征提取器具有互补性：SD（63.6）、CLIP keys（63.2）、DINO（59.6），三者集成达到 67.0
- 支持集大小在 64-128 张图像时效果饱和，更多样本只会产生冗余原型
- 部件级原型可追溯到支持图像的具体区域，具有可解释性

## 亮点与洞察

- **将跨模态对齐转化为同模态比较**：通过生成模型将语言到视觉的桥梁，巧妙避免了跨模态训练的困难
- **背景原型的上下文先验**：利用生成图像天然包含的上下文背景，解决了长期存在的背景分割阈值问题
- **完全无训练**：所有组件都是预训练的，无需额外训练、标注或微调，极大降低了应用门槛
- **可解释性**：分割结果可以追溯到支持集中的具体区域，提供了传统端到端方法所缺乏的可解释性

## 局限与展望

- 受限于特征提取器的分辨率，可能遗漏微小物体
- 无法分割扩散模型不能生成的内容（如清晰文字）
- 支持图像的采样存在计算开销（单个类别约 210 秒），但对整个图像集分割可均摊
- 继承了预训练组件的偏差和限制（如 Stable Diffusion 的潜在偏见）

## 相关工作与启发

- **TCL / GroupViT / OVSegmentor**：基于对比训练的开放词汇分割代表工作
- **CutLER**：无监督实例分割方法，为原型提取提供高质量掩码
- **ReCO**：利用 CLIP 从 ImageNet 检索示例图像进行协同分割，本文用生成模型替代了数据库检索
- 启发：生成式模型可作为"学科专家"，将文本语义转化为视觉表示，未来随生成模型和特征提取器进步，性能将自然提升

## 评分

- 新颖性: ⭐⭐⭐⭐⭐（将扩散模型用于无训练语义分割的思路非常新颖）
- 实验充分度: ⭐⭐⭐⭐（三个数据集、详细消融、多种特征提取器对比）
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐（无训练范式对降低分割应用门槛意义重大）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Can Generative Geospatial Diffusion Models Excel as Discriminative Geospatial Foundation Models?](../../ICCV2025/segmentation/can_generative_geospatial_diffusion_models_excel_as_discriminative_geospatial_fo.md)
- [\[ECCV 2024\] SegGen: Supercharging Segmentation Models with Text2Mask and Mask2Img Synthesis](seggen_supercharging_segmentation_models_with_text2mask_and_mask2img_synthesis.md)
- [\[ECCV 2024\] OpenPSG: Open-set Panoptic Scene Graph Generation via Large Multimodal Models](openpsg_open-set_panoptic_scene_graph_generation_via_large_multimodal_models.md)
- [\[CVPR 2025\] Exploring Simple Open-Vocabulary Semantic Segmentation](../../CVPR2025/segmentation/exploring_simple_open-vocabulary_semantic_segmentation.md)
- [\[CVPR 2025\] Mask-Adapter: The Devil is in the Masks for Open-Vocabulary Segmentation](../../CVPR2025/segmentation/mask-adapter_the_devil_is_in_the_masks_for_open-vocabulary_segmentation.md)

</div>

<!-- RELATED:END -->
