---
title: >-
  [论文解读] Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation?
description: >-
  [CVPR 2025][医学图像][医学图像分割] 通过统一训练与评估协议，在三个异构医学数据集上对比11种专用/通用视觉模型，发现通用视觉模型（GP-VM）在分割精度和可解释性上均可超越多数专用医学分割架构（SMA），挑战了"医学分割必须用专用架构"的传统认知。 领域现状：医学图像分割（MIS）是计算机辅助诊断的基础组件…
tags:
  - "CVPR 2025"
  - "医学图像"
  - "医学图像分割"
  - "通用视觉模型"
  - "跨数据集基准评测"
  - "可解释性分析"
  - "模型选择"
---

# Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation?

**会议**: CVPR 2025  
**arXiv**: [2603.13044](https://arxiv.org/abs/2603.13044)  
**代码**: [GitHub](https://github.com/VanessaBorst/GPVision4MIS/)  
**领域**: 医学图像  
**关键词**: 医学图像分割, 通用视觉模型, 跨数据集基准评测, 可解释性分析, 模型选择

## 一句话总结

通过统一训练与评估协议，在三个异构医学数据集上对比11种专用/通用视觉模型，发现通用视觉模型（GP-VM）在分割精度和可解释性上均可超越多数专用医学分割架构（SMA），挑战了"医学分割必须用专用架构"的传统认知。

## 研究背景与动机

**领域现状**：医学图像分割（MIS）是计算机辅助诊断的基础组件。自U-Net以来，涌现了大量针对医学影像特点（低对比度、小结构、标注稀缺）的专用架构，包括基于Transformer的HiFormer、MISSFormer，基于KAN的U-KAN，以及融合Mamba的Swin-UMamba等。与此同时，通用视觉模型（如SegFormer、InternImage、VWFormer）在自然图像上展现了强大的泛化能力。

**现有痛点**：尽管通用视觉模型在ADE20K等标准基准上表现出色，其在医学影像上的有效性缺乏系统性验证。现有对比研究通常依赖各论文自报的指标，但不同论文在数据预处理、增强策略、优化设置上差异巨大，导致所谓的"架构优势"可能只是实验设计差异的副产品。

**核心矛盾**：缺乏在统一实验条件下的控制性对比——究竟是专用架构的domain-specific设计带来了真正优势，还是通用模型的大规模预训练和充分调优已经足够？

**本文目标**：建立标准化基准框架，在统一训练/评估协议下系统对比通用视觉模型与专用医学分割架构，并结合可解释性分析（Grad-CAM）给出全面评估。

**切入角度**：与其争论架构创新的价值，不如通过严格对照实验展示通用模型的实际表现。选择三个覆盖不同模态（皮肤镜、内窥镜、超声心动图）和类别结构（二分类/多分类）的数据集，消除混杂因素。

**核心 idea**：通用视觉模型在标准化条件下可系统性超越多数专用医学分割架构，且无需domain-specific设计即可捕捉临床相关结构。

## 方法详解

### 整体框架

本文并非提出新模型，而是设计了一套控制性基准评测框架。选取5种专用医学架构（U-Net、HiFormer-B、MISSFormer、Swin-UMamba、U-KAN-L）和6种通用视觉模型（SegFormer-B3、SegNeXt-L、VWFormer×2、InternImage-T、TransNeXt-Tiny），在三个数据集（ISIC'18皮肤病变、BKAI-IGH息肉、CAMUS心脏超声）上统一训练并评估。

### 关键设计

1. **统一训练协议**:

    - 功能：消除实验设计差异带来的混杂因素
    - 核心思路：所有模型使用相同的ImageNet预训练encoder、512×512输入分辨率、AdamW优化器 + REX学习率调度器、batch size 8、统一的数据增强pipeline。对每个模型-数据集组合执行学习率搜索（10⁻⁴/5×10⁻⁵/10⁻⁵），100 epoch选最优后再跑150 epoch五折交叉验证。
    - 设计动机：现有文献中所谓的"架构优势"很大程度上可能来自不同的训练配方而非架构本身，统一协议是公平对比的前提。

2. **多维度评估体系**:

    - 功能：超越单纯的分割精度，全面评估模型行为
    - 核心思路：使用mIoU、mDSC、Recall、Precision四个指标量化分割性能，同时利用Grad-CAM可视化分析模型的注意力模式——考察模型是否真正关注临床相关区域。评估采用五折交叉验证 + 无背景类的全局micro-averaging。
    - 设计动机：在医学场景中，模型不仅需要分割准确，还需要"看对地方"——Grad-CAM分析揭示模型的决策依据是否与临床知识一致。

3. **异构数据集覆盖设计**:

    - 功能：验证结论的跨模态/跨任务泛化性
    - 核心思路：选择ISIC'18（RGB皮肤镜、二分类、3565张）、BKAI-IGH NeoPolyp（RGB内窥镜、三分类、945张）、CAMUS（灰度超声、四分类、1996张）三个差异化数据集。对CAMUS采用患者级别分组以避免信息泄露，对ISIC'18和BKAI-IGH执行去重过滤。
    - 设计动机：涵盖不同模态（RGB/灰度）、不同类别结构（二分类/多分类）、不同成像质量和数据规模，确保结论不是对某个特定场景的过拟合。

### 损失函数 / 训练策略

ISIC'18使用二元交叉熵损失，BKAI-IGH和CAMUS使用标准交叉熵损失。所有模型在同一数据集内使用完全相同的损失函数和增强策略。采用早停机制，模型在PyTorch 2.5.1 + 2×A100 GPU上训练。

## 实验关键数据

### 主实验

| 数据集 | 模型类别 | 最佳模型 | mDSC(%) | 最佳SMA | mDSC(%) | 差距 |
|--------|---------|---------|---------|---------|---------|------|
| ISIC'18 | GP-VM | TransNeXt | 91.9±0.7 | Swin-UMamba | 91.3±0.5 | +0.6 |
| BKAI-IGH | GP-VM | VW-MiT | 89.7±0.8 | Swin-UMamba | 88.9±0.6 | +0.8 |
| CAMUS | GP-VM | SegNeXt/VW-MiT | 91.6±0.1 | Swin-UMamba | 91.3±0.3 | +0.3 |
| 三数据集平均 | GP-VM | VW-MiT | 91.0 | Swin-UMamba | 90.5 | +0.5 |

### 消融实验（各模型三数据集跨类别表现）

| 模型 | BKAI-IGH C₁(非肿瘤) | BKAI-IGH C₂(肿瘤) | CAMUS LV | CAMUS LV Wall | CAMUS LA |
|------|---------------------|---------------------|----------|---------------|----------|
| VW-MiT(GP) | 66.1±4.3 | 92.7±0.9 | 94.6±0.2 | 88.7±0.2 | 91.8±0.2 |
| Swin-UMamba(SMA) | 59.2±3.8 | 92.5±0.6 | 94.4±0.3 | 88.3±0.3 | 91.4±0.4 |
| U-KAN-L(SMA) | 36.9±12.2 | 87.1±0.9 | 94.0±0.2 | 87.4±0.2 | 90.6±0.4 |
| MISSFormer(SMA) | 42.0±6.5 | 87.5±1.8 | 93.8±0.1 | 87.2±0.2 | 90.7±0.2 |

### 关键发现

- **通用模型全面领先**：按三数据集平均mDSC，前6名全部是通用视觉模型（VW-MiT 91.0%、VW-Conv/TransNeXt 90.9%、InternImage-T 90.8%、SegNeXt/SegFormer 90.7%），专用模型中仅Swin-UMamba（90.5%）接近。
- **差距因数据集而异**：在BKAI-IGH上差距最大（非肿瘤息肉类别GP-VM比多数SMA高20+个百分点），在CAMUS上差距最小（约1-2%）。
- **Grad-CAM分析显示GP-VM注意力更聚焦**：通用模型在困难case上能更精确地聚焦临床相关区域，部分GP-VM的注意力图比专用模型更准确——说明大规模预训练获得的通用特征表示在医学场景中同样有效。
- **Swin-UMamba是唯一有竞争力的SMA**：其在每个数据集上都是SMA中最好的，与GP-VM差距最小；其余SMA（尤其是U-KAN-L、MISSFormer）差距显著。

## 亮点与洞察

- **挑战传统假设的实证证据**：不是提出新方法，而是通过严格控制实验回答了一个被长期忽视的基本问题——domain-specific架构是否带来真正优势。结论与直觉相反，这种反直觉发现对社区的价值巨大。
- **Grad-CAM可解释性分析**：不仅看指标，还通过注意力可视化验证了GP-VM确实学到了临床相关的特征表示。这比单纯的数值对比更有说服力，也为临床部署提供了信心。
- **资源配置启示**：既然GP-VM已经够用，那么与其投入大量资源设计新的domain-specific架构，不如把精力放在数据策展、训练优化和OOD泛化评估上——这一洞察对整个医学AI社区有深远影响。

## 局限与展望

- **数据集覆盖有限**：仅涵盖三个2D数据集和两种模态，未扩展到3D医学图像（如CT/MRI体积分割）或极端低数据场景。
- **模型选择不完全均衡**：部分SMA（如U-KAN-L、HiFormer-B）参数量偏小，虽然趋势一致但可能引入轻微偏差。
- **未考虑SAM等基础模型**：Segment Anything Model及其医学变体（MedSAM、SAM-Med2D）是新兴的强大选项，未纳入对比。
- **改进方向**：扩展到更多模态和3D场景；增加OOD评估（如训练在BKAI上、测试在Kvasir-SEG上）；纳入带prompt的基础模型。

## 相关工作与启发

- **vs Swin-UMamba**: Swin-UMamba结合了Mamba的长距离建模和ImageNet预训练的优势，是SMA中唯一能接近GP-VM的模型，说明预训练策略比architecture novelty更重要。
- **vs VWFormer**: 通过多尺度窗口注意力实现的通用分割器，在医学场景无需任何修改就能取得最佳表现，证明general representation已经足够强大。
- **vs SAM系列**: 本文未包含SAM对比，但SAM已被其他工作证明在医学分割中有效，与本文结论一致——通用模型在医学领域可行。

## 评分

- 新颖性: ⭐⭐⭐ 方法上无创新，但问题本身有价值且实验设计严谨
- 实验充分度: ⭐⭐⭐⭐ 11个模型×3个数据集×5折交叉验证+Grad-CAM，设计完善
- 写作质量: ⭐⭐⭐⭐ 条理清晰，实验描述详尽，代码开源
- 价值: ⭐⭐⭐⭐ 为医学AI社区提供了重要的实证指导，推动资源合理配置

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](biclip_bidirectional_and_consistent_language-image_processing_for_robust_medical.md)
- [\[CVPR 2025\] Show and Segment: Universal Medical Image Segmentation via In-Context Learning](show_and_segment_universal_medical_image_segmentation_via_in-context_learning.md)
- [\[CVPR 2025\] Latent Drifting in Diffusion Models for Counterfactual Medical Image Synthesis](latent_drifting_in_diffusion_models_for_counterfactual_medical_image_synthesis.md)
- [\[CVPR 2026\] Simple-ViLMedSAM: Simple Text Prompts Meet Vision-Language Models for Medical Image Segmentation](../../CVPR2026/medical_imaging/simple-vilmedsam_simple_text_prompts_meet_vision-language_models_for_medical_ima.md)
- [\[CVPR 2026\] Revisiting 2D Foundation Models for Scalable 3D Medical Image Classification](../../CVPR2026/medical_imaging/revisiting_2d_foundation_models_for_scalable_3d_medical_image_classification.md)

</div>

<!-- RELATED:END -->
