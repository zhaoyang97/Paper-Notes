---
title: >-
  [论文解读] Progressive Test Time Energy Adaptation for Medical Image Segmentation
description: >-
  [ICCV 2025][医学图像][test-time adaptation] 提出一种基于能量模型的渐进式测试时自适应方法，训练一个形状能量模型作为分布内/外判别器，在测试时通过最小化能量值引导分割模型适应目标域，在心脏、脊髓、肺部等 8 个公共数据集上持续超越基线。 医学图像分割面临的分布偏移问题： - 不同医院的成像协…
tags:
  - "ICCV 2025"
  - "医学图像"
  - "test-time adaptation"
  - "energy-based model"
  - "图像分割"
  - "domain shift"
  - "shape prior"
---

# Progressive Test Time Energy Adaptation for Medical Image Segmentation

**会议**: ICCV 2025  
**arXiv**: [2503.16616](https://arxiv.org/abs/2503.16616)  
**代码**: 无  
**领域**: 医学图像  
**关键词**: test-time adaptation, energy-based model, medical image segmentation, domain shift, shape prior

## 一句话总结

提出一种基于能量模型的渐进式测试时自适应方法，训练一个形状能量模型作为分布内/外判别器，在测试时通过最小化能量值引导分割模型适应目标域，在心脏、脊髓、肺部等 8 个公共数据集上持续超越基线。

## 研究背景与动机

**医学图像分割面临的分布偏移问题**：
- 不同医院的成像协议不一致（MRI 序列、扫描仪参数）
- 患者群体差异（年龄、病理状态、人口学特征）
- 导致在源域训练的模型在目标域性能显著下降

**现有方法的局限**：

**域适应方法**：需要多次遍历目标数据，在临床场景中不实际（无法预知患者数据）

**测试时训练（TTT）**：需要额外的自监督任务与主任务联合训练

**基于熵的 TTA**（TENT/EATA/SAR）：通用正则化方法，未利用分割任务的形状先验

**CoTTA/MEMO**：基于伪标签或增广一致性，但不够精细

**TEA**：使用能量模型进行分类 TTA，但只输出全局单一能量值，粒度不足

**核心动机**：分割任务有强形状先验（如心脏的解剖结构），可以判断预测形状是否合理。利用能量模型作为形状分类器，在 patch 级别判别预测质量，引导分割模型修正错误区域。

## 方法详解

### 整体框架

方法分为两个阶段：
1. **准备阶段**（源域）：训练形状能量模型 $g_\phi(\cdot)$
2. **适应阶段**（目标域）：冻结能量模型，渐进式更新分割模型 $f_\theta(\cdot)$ 的 BatchNorm 层

### 关键设计

1. **区域能量模型（Region-based Energy Model）**：

    - 使用全卷积网络将分割图 $\hat{S}$ 映射到 $K \times K$ 能量图：$g_\phi(\hat{S}): \mathbb{R}^{H\times W} \mapsto \mathbb{R}^{K\times K}$
    - 每个 patch（大小 $h \times w$，其中 $h=H/K, w=W/K$）对应一个能量值
    - 低能量 = 分布内（正确形状），高能量 = 分布外（错误预测）
    - 建模为二分类任务，训练目标为 patchwise BCE 损失：

    $\mathcal{L}_\phi = \frac{1}{N_p}\sum_{i=1}^{N_p} \left(-y_s^i\log\sigma(-g_\phi(s_s^i)) - (1-y_s^i)\log(1-\sigma(-g_\phi(s_s^i)))\right)$

    - 设计动机：全局单一能量值不够精细，patch 级能量可以定位具体的错误区域

2. **对抗扰动生成负样本**：

    - 源域只有正确的分割结果，缺少分布外（错误）样本
    - 使用 FGSM 对输入图像施加对抗扰动：$\epsilon = \delta \cdot \text{sign}(\nabla_{I_s}\mathcal{L}(f_\theta(I_s), S_s))$
    - 扰动后的输入通过分割网络产生错误分割 $\tilde{S}_s = f_\theta(I_s + \epsilon)$
    - 额外施加空间仿射变换和像素噪声增加多样性
    - 通过比较扰动分割和 ground truth 生成分类标签：$y_s = 1 - \mathbf{1}(d(\tilde{s}_s, s_s) < \tau)$
    - 设计动机：对抗扰动将数据推向低密度区域（自然的 OOD 区域），且通过分割网络的约束产生合理的错误形状

3. **渐进式能量适应**：

    - 测试时，冻结能量模型 $g_\phi$，仅更新分割模型 $f_\theta$ 的 BatchNorm 参数
    - 目标：将预测的能量值对齐到参考低能量（全零矩阵 $\mathbf{0}_{K\times K}$）
    - 适应目标：

    $\theta^* = \arg\min_\theta -\sum_{i=1}^{B_t}\log(1-\sigma(-g_\phi(\hat{s}_t^i)))$

    - 使用 Adam 优化器，每个样本迭代 10 次
    - 每批处理后恢复模型权重
    - 设计动机：通过最小化能量值，鼓励分割模型产生与自然解剖结构一致的预测形状

### 损失函数 / 训练策略

- **能量模型训练**：BCE 损失，patch 大小 $h=w=16$，距离度量使用均值绝对差异，阈值 $\tau=50$
- **测试时适应**：Adam 优化器，10 次迭代/样本
- **仅更新 BatchNorm 层**：遵循 TTA 的标准惯例，处理完每批后恢复权重
- **对抗扰动**：使用 Dice Loss 作为 FGSM 的目标函数

## 实验关键数据

### 主实验 (表格)

**心脏分割（ACDC → 其他数据集, UNet 骨干）**：

| 方法 | LVQuant LV DSC↑ | LVQuant Myo DSC↑ | MyoPS LV DSC↑ | M&M LV DSC↑ | M&M Myo DSC↑ | Avg Rank |
|------|----------------|------------------|--------------|------------|-------------|----------|
| Pretrained | 58.98 | 42.52 | 85.69 | 47.69 | 41.19 | 4.33 |
| TENT | 65.78 | 51.57 | 85.63 | 57.01 | 48.26 | 2.92 |
| CoTTA | 64.58 | 50.52 | 85.64 | 52.98 | 46.72 | 3.67 |
| TEA | 67.96 | 54.10 | 85.88 | 52.83 | 48.06 | 2.92 |
| **Ours** | **76.93** | **59.43** | **86.06** | **61.84** | **53.13** | **1.08** |

**脊髓分割（GMSC Site 1→其他，单类）**：

| 方法 | 1→2 | 1→3 | 1→4 | 4→1 | 4→2 | 4→3 | Avg DSC |
|------|-----|-----|-----|-----|-----|-----|---------|
| TENT | 70.5 | 16.8 | 57.4 | 87.0 | 67.9 | 72.9 | 62.1 |
| CoTTA | 66.1 | 63.3 | 92.1 | 95.0 | 54.7 | 86.7 | 76.4 |
| TEA | 68.4 | 66.5 | 92.4 | 94.9 | 54.7 | 86.7 | 77.3 |
| InTENT | 86.6 | 28.7 | 71.4 | 83.3 | 79.2 | 75.0 | 70.7 |
| **Ours** | 73.6 | **77.7** | **95.3** | **95.1** | 56.2 | **87.2** | **80.9** |

**肺部分割（CHN X-ray → 其他）**：

| 方法 | CHN→MCU DSC | CHN→JSRT DSC | Avg DSC |
|------|------------|-------------|---------|
| TENT | 86.2 | 95.2 | 90.7 |
| CoTTA | 95.8 | 95.2 | 95.5 |
| TEA | 95.7 | 95.5 | 95.6 |
| InTENT | 95.5 | 96.3 | 95.9 |
| **Ours** | **96.1** | **96.3** | **96.2** |

### 消融实验 (表格)

**不同分割骨干的适应效果（ACDC→LVQuant LV DSC）**：

| 骨干 | Pretrained | TENT | CoTTA | TEA | **Ours** | Avg Rank |
|------|-----------|------|-------|-----|---------|----------|
| UNet | 58.98 | 65.78 | 64.58 | 67.96 | **76.93** | **1.08** |
| MedNeXt | 57.55 | 75.10 | 74.57 | 75.85 | **76.22** | **1.00** |
| SwinUNETR | 68.44 | 74.06 | 73.41 | 74.32 | **76.05** | **1.25** |

**不同源域的适应效果（M&M → 其他, UNet）**：

| 方法 | LVQuant LV DSC | MyoPS LV DSC | ACDC LV DSC | Avg Rank |
|------|---------------|-------------|------------|----------|
| Pretrained | 89.08 | 75.80 | 40.84 | 4.08 |
| TENT | 92.03 | 77.34 | 52.74 | 3.67 |
| TEA | 92.27 | 77.75 | 56.68 | 3.00 |
| **Ours** | **93.25** | **79.14** | **59.97** | **1.08** |

### 关键发现

- 在 3 种分割骨干（UNet/MedNeXt/SwinUNETR）上均取得最低平均排名（1.0-1.33），证明模型无关性
- 在心脏分割中，UNet 骨干上 LV DSC 从 58.98%（预训练）提升到 76.93%（适应后），提升近 18 个百分点
- 能量模型的 OOD 检测准确率超过 92%，能有效识别错误区域
- 在脊髓和肺部等单类分割任务上同样有效，平均 DSC 分别达到 80.9% 和 96.2%
- 相比 TENT 等基于熵的方法，利用形状先验的能量方法在大分布偏移场景下优势更明显

## 亮点与洞察

1. **首个能量模型 TTA 用于医学分割**：创新性地将能量模型作为形状先验的隐式编码器，替代传统的显式形状参数化
2. **对抗扰动生成训练数据**：巧妙利用 FGSM 探索错误分割空间，无需额外的 OOD 数据收集
3. **区域级能量 vs 全局能量**：patch 级别的能量判别比 TEA 的全局单值更精细，能定位具体错误区域
4. **模型无关性**：方法可即插即用到任意分割网络，不需要特定架构设计
5. **渐进式适应**：每张图像独立适应后恢复权重，避免误差积累

## 局限与展望

- 每个样本需要 10 次迭代优化 BatchNorm，推理速度会下降
- 仅更新 BatchNorm 层可能限制适应能力，对于 BatchNorm 参数较少的架构效果可能有限
- 能量模型的判别能力取决于源域的多样性和对抗扰动的质量
- 对抗扰动的强度 $\delta$ 和 patch 大小等超参数需要调优
- 未在 3D 体积分割任务上验证

## 相关工作与启发

- 能量模型作为形状先验的思想可推广到其他需要结构先验的密集预测任务
- 对抗扰动生成负样本的策略为一般性的 OOD 检测提供了启发
- 区域级能量判别的思想可与分层能量模型结合，实现更精细的适应

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首次将能量模型引入医学分割 TTA，对抗扰动生成负样本的思路巧妙
- **实验充分度**: ⭐⭐⭐⭐⭐ 8 个数据集、3 种骨干、3 种器官、多种成像模态，非常全面
- **写作质量**: ⭐⭐⭐⭐ 数学推导严谨，方法描述清晰，但部分符号较密集
- **价值**: ⭐⭐⭐⭐⭐ 临床实用价值高，模型无关设计降低使用门槛，在大分布偏移场景效果显著

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] SPEGC: Continual Test-Time Adaptation via Semantic-Prompt-Enhanced Graph Clustering for Medical Image Segmentation](../../CVPR2026/medical_imaging/spegc_continual_test-time_adaptation_via_semantic-prompt-enhanced_graph_clusteri.md)
- [\[AAAI 2026\] Cross-Sample Augmented Test-Time Adaptation for Personalized Intraoperative Hypotension Prediction](../../AAAI2026/medical_imaging/cross-sample_augmented_test-time_adaptation_for_personalized_intraoperative_hypo.md)
- [\[ICCV 2025\] GDKVM: Echocardiography Video Segmentation via Spatiotemporal Key-Value Memory with Gated Delta Rule](gdkvm_echocardiography_video_segmentation_via_spatiotemporal_key-value_memory_wi.md)
- [\[ICCV 2025\] UKBOB: One Billion MRI Labeled Masks for Generalizable 3D Medical Image Segmentation](ukbob_one_billion_mri_labeled_masks_for_generalizable_3d_medical_image_segmentat.md)
- [\[CVPR 2025\] Noise-Consistent Siamese-Diffusion for Medical Image Synthesis and Segmentation](../../CVPR2025/medical_imaging/noise-consistent_siamese-diffusion_for_medical_image_synthesis_and_segmentation.md)

</div>

<!-- RELATED:END -->
