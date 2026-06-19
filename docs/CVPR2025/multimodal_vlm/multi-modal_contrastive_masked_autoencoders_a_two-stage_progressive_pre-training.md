---
title: >-
  [论文解读] A Two-Stage Progressive Pre-training using Multi-Modal Contrastive Masked Autoencoders
description: >-
  [CVPR 2025][多模态VLM][RGB-D预训练] 本文提出渐进式两阶段预训练策略——第一阶段用patch级对比学习对齐RGB和深度模态的跨模态表示，第二阶段用掩码自编码+受扩散模型启发的去噪+特征蒸馏联合训练，在ScanNet语义分割上比Mask3D提升+1.3% mIoU，在多个RGB-D下游任务上达到SOTA。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "RGB-D预训练"
  - "对比学习"
  - "掩码自编码器"
  - "去噪扩散"
  - "知识蒸馏"
---

# A Two-Stage Progressive Pre-training using Multi-Modal Contrastive Masked Autoencoders

**会议**: CVPR 2025  
**arXiv**: [2408.02245](https://arxiv.org/abs/2408.02245)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: RGB-D预训练, 对比学习, 掩码自编码器, 去噪扩散, 知识蒸馏

## 一句话总结
本文提出渐进式两阶段预训练策略——第一阶段用patch级对比学习对齐RGB和深度模态的跨模态表示，第二阶段用掩码自编码+受扩散模型启发的去噪+特征蒸馏联合训练，在ScanNet语义分割上比Mask3D提升+1.3% mIoU，在多个RGB-D下游任务上达到SOTA。

## 研究背景与动机
1. **领域现状**：自监督学习(SSL)已成为视觉预训练的主流范式，主要分为两大家族：掩码图像建模(MIM，如MAE)学习局部空间统计特征，对比学习(如SimCLR/MoCo)学习对增强不变的判别性表示。对于RGB-D数据，MultiMAE和Mask3D已尝试利用多模态数据预训练ViT。
2. **现有痛点**：(a) MultiMAE需要预训练时用语义分割标签且微调时需要多模态输入；(b) Mask3D虽然通过深度重建编码3D先验但未学习跨模态关系来捕获局部上下文；(c) CoMAE提出了混合框架但仅限小规模数据集且微调时需要所有模态；(d) 现有方法通常无法捕获数据的高频成分。
3. **核心矛盾**：对比学习和MAE学习的是互补特征——前者学习增强不变性的判别性表示，后者学习局部空间依赖。简单地在单个框架中结合两者（如pixel/patch级对比学习+masking）并不容易，且在RGB-D场景下效果不佳。
4. **本文目标** (1) 如何有效结合对比学习和MAE从RGB-D数据中学到互补表示？(2) 如何学习数据的高频成分？(3) 如何在两阶段之间传递知识？
5. **切入角度**：作者受扩散模型去噪成功的启发，假设去噪可以帮助编码器提取高频特征，与MAE的低频重建形成互补。同时设计两阶段渐进式训练而非单框架融合，避免两种SSL范式互相干扰。
6. **核心 idea**：分阶段发挥对比学习的跨模态对齐能力和MAE+去噪的多层次特征学习能力，通过特征蒸馏在阶段间传递知识，实现互补融合。

## 方法详解

### 整体框架
输入为RGB-D图像对。使用模态特有的ViT编码器。第一阶段：用patch级InfoNCE对比损失对齐RGB和深度patch的表示空间。第二阶段：用第一阶段权重初始化编码器，对两个模态进行随机masking后，将未被mask的patch通过编码器，拼接可学习mask token后经轻量级解码器重建深度模态的被mask patch（MAE目标）+预测未被mask patch的噪声（去噪目标）+对齐第二阶段全局嵌入到第一阶段全局嵌入（蒸馏目标）。

### 关键设计

1. **Patch级跨模态对比学习（第一阶段）**:

    - 功能：在patch级别对齐RGB和深度模态的表示，学习局部跨模态对应关系
    - 核心思路：对于batch中的RGB-D对，将RGB和深度图通过各自的ViT编码器得到patch级特征 $\mathbf{z}_i^{rgb}$ 和 $\mathbf{z}_i^{depth}$，使用InfoNCE损失：$\mathcal{L}_{PNCE} = -\frac{1}{N}\sum_{i=1}^N \log \frac{\exp(s_{i,i}/\tau)}{\sum_{k\neq i}\exp(s_{i,k}/\tau) + \exp(s_{i,i}/\tau)}$，其中 $s_{i,j} = \|\mathbf{z}_i^{rgb}\|^T\|\mathbf{z}_i^{depth}\|$。双向计算后取平均
    - 设计动机：实例级对比学习对密集预测任务（如语义分割）帮助有限，因为它只学习高层语义而忽略局部判别性特征。Patch级对比可以捕获跨模态的局部上下文对应关系

2. **掩码自编码+去噪联合训练（第二阶段核心）**:

    - 功能：学习细粒度空间特征（MAE）同时提取高频成分（去噪）
    - 核心思路：MAE部分：随机mask两个模态的patch，仅未被mask的patch经编码器，拼接可学习mask token后解码重建深度模态被mask区域，损失 $\mathcal{L}_{depth} = \frac{1}{n}\sum\|\mathbf{M}_i^{depth} \circ (\mathbf{x}_i^{depth} - \hat{\mathbf{x}}_i^{depth})\|_2^2$。去噪部分：在深度输入上添加高斯噪声 $\mathbf{x}_i^{depth} \leftarrow \mathbf{x}_i^{depth} + \sigma_i^{depth}\mathbf{e}_i^{depth}$，噪声水平 $\sigma$ 通过正弦位置编码+MLP生成嵌入，添加到编码表示中（类似扩散模型的timestep embedding），解码器同时重建被mask patch和预测未被mask patch中的噪声：$\mathcal{L}_{denoise} = \frac{1}{n}\sum\|(1-\mathbf{M}_i^{depth})\circ(\sigma_i^{depth}\mathbf{e}_i^{depth} - \hat{\mathbf{x}}_i^{depth})\|_2^2$
    - 设计动机：MAE的重建任务主要捕获低频空间统计特征，去噪任务迫使编码器学习区分噪声与信号，从而提取数据中的高频成分。去噪利用了原本MAE中被"浪费"的未被mask patch的重建输出，几乎不增加计算开销

3. **全局特征蒸馏（跨阶段知识传递）**:

    - 功能：将第一阶段学到的全局跨模态对应知识传递到第二阶段模型
    - 核心思路：对第二阶段编码器输出做max pooling得到全局嵌入 $\mathbf{f_2}$，与第一阶段冻结模型的全局嵌入 $\mathbf{f_1}$ 通过smooth $\ell_1$ 损失对齐。对RGB和深度模态分别蒸馏，最终蒸馏损失为两个模态的和
    - 设计动机：单纯的MAE+去噪第二阶段可能丢失第一阶段对比学习获得的全局跨模态判别信息，特征蒸馏可以在不限制第二阶段灵活性的前提下保留这些知识

### 损失函数 / 训练策略
第一阶段：$\mathcal{L}_{stage1} = \mathcal{L}_{PNCE}$。第二阶段：$\mathcal{L}_{stage2} = \alpha\mathcal{L}_{depth} + \beta\mathcal{L}_{denoise} + \gamma\mathcal{L}_{distill}$。第二阶段编码器用第一阶段权重初始化。第一阶段在ImageNet上预训练，第二阶段在ScanNet（2.5M RGB-D帧）或SUN RGB-D上继续预训练。仅重建深度模态（实验表明同时重建RGB不提升性能）。

## 实验关键数据

### 主实验

| 数据集 | 任务 | 指标 | Ours | Mask3D | MultiMAE | MAE | 提升vs Mask3D |
|--------|------|------|------|--------|----------|-----|--------------|
| ScanNet | 语义分割 | mIoU | **67.5** | 66.2 | 65.1 | 64.8 | +1.3 |
| SUN RGB-D | 语义分割 | mIoU | **48.7** | 47.4 | 47.1 | 47.3 | +1.3 |
| NYUv2 | 深度估计 | δ₁ | **87.1** | 85.4 | 85.3 | 85.1 | +1.7 |
| ScanNet | 实例分割 | AP | **23.7** | 22.8 | 22.4 | 20.7 | +0.9 |

ViT-L扩展：

| 方法 | ScanNet mIoU (ViT-L) |
|------|---------------------|
| MAE | 68.2 |
| MultiMAE | 69.3 |
| **Ours** | **70.8** |

### 消融实验

| 对比学习 | 重建 | 去噪 | 蒸馏 | ScanNet mIoU |
|---------|------|------|------|-------------|
| ✓ | ✗ | ✗ | ✗ | 63.4 |
| ✓ | ✓ | ✗ | ✗ | 66.3 |
| ✓ | ✓ | ✗ | ✓ | 66.5 |
| ✓ | ✓ | ✓ | ✗ | 67.0 |
| ✓ | ✓ | ✓ | ✓ | **67.5** |

去噪组件消融：

| 配置 | mIoU | 说明 |
|------|------|------|
| 无噪声 | 66.5 | 无去噪组件 |
| 仅加噪声 | 66.9 | 输入加噪但不预测噪声 |
| 完整去噪 | **67.5** | 噪声预测+位置编码 |

### 关键发现
- **对比学习+重建的互补性显著**：仅对比学习63.4→加重建66.3，跳升2.9个点，验证了两种SSL范式学习互补特征的假设
- 去噪贡献+0.7 mIoU，其中仅加噪声就贡献+0.4，完整去噪（含噪声预测和位置编码）再贡献+0.6，证明受扩散模型启发的去噪确实能学习高频特征
- 同时重建RGB和深度不比仅重建深度更好（67.5 vs 66.9），因为3D先验主要通过深度预测编码
- 高掩码率(80%×80%)效果最佳，与MAE的发现一致，高掩码率使任务更具挑战性
- 在低数据场景下优势更明显：仅用60%训练数据即可略超MAE用100%数据的性能

## 亮点与洞察
- **受扩散模型启发的去噪预训练**是本文最创新的贡献：将扩散模型的去噪思想引入自监督表示学习，利用噪声级别的正弦位置编码帮助模型区分噪声与信号。这个思路可以迁移到任何需要学习高频特征的预训练场景
- **两阶段渐进式设计**比单框架融合合理：经验表明在RGB-D场景下单框架混合对比+MAE效果差，而渐进式先对齐再重建各取所长
- 巧妙利用MAE中"被浪费"的未被mask patch来计算去噪损失，几乎不增加额外计算

## 局限与展望
- 仅在RGB-D场景验证，未扩展到其他多模态组合（如RGB-点云、RGB-热红外）
- Patch级对比学习未尝试像素级，作者提到pixel-to-point对比可能更有效但留作未来工作
- 第二阶段在小规模数据集（SUN RGB-D仅10k图像）上也有效，但大规模数据下的scaling行为未充分探讨
- 预训练超参数（α、β、γ）的选择对结果的敏感性分析不够详细
- 去噪的噪声水平范围 $[0, \sigma_{max}]$ 的选择是否有系统研究

## 相关工作与启发
- **vs Mask3D**: Mask3D仅用深度重建编码3D先验，不学跨模态关系；本文通过对比学习显式对齐RGB-深度+重建+去噪，更全面
- **vs MultiMAE**: MultiMAE需要语义标签预训练且微调需多模态输入；本文仅需RGB-D预训练且微调仅用RGB，实用性更强
- **vs CoMAE**: CoMAE也混合对比+MAE但局限于小规模且微调需双模态；本文的渐进式设计在大规模和小规模数据集都有效
- 去噪作为预训练目标的思路值得关注，可能与Diffusion Pre-training等方向产生交叉

## 评分
- 新颖性: ⭐⭐⭐⭐ 扩散去噪引入SSL预训练是新颖点，渐进式两阶段设计有理论依据
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖语义分割、深度估计、实例分割、低数据场景，消融全面彻底
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，但公式较多可能影响可读性
- 价值: ⭐⭐⭐⭐ 渐进式多范式预训练策略和去噪预训练目标可广泛应用于多模态SSL

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] SCAN: Bootstrapping Contrastive Pre-training for Data Efficiency](../../ICCV2025/multimodal_vlm/scan_bootstrapping_contrastive_pre-training_for_data_efficiency.md)
- [\[CVPR 2025\] Post-pre-training for Modality Alignment in Vision-Language Foundation Models](post-pre-training_for_modality_alignment_in_vision-language_foundation_models.md)
- [\[CVPR 2025\] GeoMM: On Geodesic Perspective for Multi-Modal Learning](geomm_on_geodesic_perspective_for_multi-modal_learning.md)
- [\[CVPR 2026\] PowerCLIP: Powerset Alignment for Contrastive Pre-Training](../../CVPR2026/multimodal_vlm/powerclip_powerset_alignment_for_contrastive_pre-training.md)
- [\[CVPR 2025\] Multimodal Autoregressive Pre-training of Large Vision Encoders](multimodal_autoregressive_pre-training_of_large_vision_encoders.md)

</div>

<!-- RELATED:END -->
