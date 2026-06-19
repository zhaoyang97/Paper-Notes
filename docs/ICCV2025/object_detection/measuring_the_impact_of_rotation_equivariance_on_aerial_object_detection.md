---
title: >-
  [论文解读] Measuring the Impact of Rotation Equivariance on Aerial Object Detection
description: >-
  [ICCV 2025][目标检测][航空图像检测] 提出 MessDet，一个基于旋转等变网络的航空目标检测器，通过新型下采样过程实现严格旋转等变性，并引入旋转等变通道注意力和多分支检测头，在 DOTA 等数据集上以极低参数量达到 SOTA 性能。 航空图像目标检测与通用目标检测的核心区别在于：鸟瞰视角下目标会以任意方向出现…
tags:
  - "ICCV 2025"
  - "目标检测"
  - "航空图像检测"
  - "旋转等变性"
  - "群等变网络"
  - "通道注意力"
  - "多分支检测头"
---

# Measuring the Impact of Rotation Equivariance on Aerial Object Detection

**会议**: ICCV 2025  
**arXiv**: [2507.09896](https://arxiv.org/abs/2507.09896)  
**代码**: [GitHub](https://github.com/Nu1sance/MessDet)  
**领域**: 目标检测  
**关键词**: 航空图像检测, 旋转等变性, 群等变网络, 通道注意力, 多分支检测头

## 一句话总结

提出 MessDet，一个基于旋转等变网络的航空目标检测器，通过新型下采样过程实现严格旋转等变性，并引入旋转等变通道注意力和多分支检测头，在 DOTA 等数据集上以极低参数量达到 SOTA 性能。

## 研究背景与动机

航空图像目标检测与通用目标检测的核心区别在于：鸟瞰视角下目标会以任意方向出现。这要求检测器具备以下性质：
- **分类任务**需要旋转不变性——无论目标朝向如何，分类结果应一致
- **回归任务**需要旋转等变性——当输入旋转时，预测的角度应相应调整

现有航空目标检测器主要通过以下方式隐式学习旋转等变性：增加模型参数、数据增强、新的边界框表示方法或定制损失函数。少量工作（如 ReDet、FRED）尝试使用旋转等变网络（RE-Net）显式实现旋转等变性,但存在关键问题：

**核心发现**：Mohamed 等人证明，在偶数尺寸特征图上使用步长为 2 的卷积核会导致旋转前后采样点不同，从而**打破严格旋转等变性**。之前的工作（如 ReDet）仅实现了近似旋转等变性。FRED 通过单侧零填充将偶数维度变为奇数来保证严格等变性，但可能引入特征错位问题。

**关键开放问题**：对于航空目标检测任务，严格旋转等变性是否真的必要？它比近似旋转等变性带来多大的性能提升？本文首次定量回答这个问题。

## 方法详解

### 整体框架

MessDet 基于 RTMDet 架构重新设计，使用 E2CNN 实现旋转等变的骨干网络（CSPNeXt）和颈部网络（CSPNeXtPAFPN），并引入三项改进：（1）保持旋转等变性的新型下采样过程；（2）旋转等变通道注意力机制（RE-CA）；（3）多分支检测头网络。

### 关键设计

1. **严格旋转等变下采样过程**:

    - 功能：在保持输出尺寸不变的前提下，确保下采样过程中的严格旋转等变性
    - 核心思路：在步长为 2 的下采样卷积层之前，插入一个"调谐层"（tuning layer），将偶数维度的特征图变为奇数维度：
        - 调谐层参数：$k=4, p=1, s=1$，将输入尺寸 $2n$ 变为 $2n-1$
        - 下采样层参数：$k=3, p=1, s=2$，将 $2n-1$ 变为 $n$
        - 输出尺寸计算：$S_{out} = \lfloor((2n-1)-1)/2\rfloor + 1 = n$
      调谐层不改变最终输出尺寸，但确保下采样卷积始终作用于奇数维度特征图
    - 设计动机：避免 FRED 的单侧填充导致的特征错位，用户可通过简单地添加/移除调谐层来控制模型是否具有严格旋转等变性，便于进行定量对比实验

2. **旋转等变通道注意力（RE-CA）**:

    - 功能：在不破坏旋转等变性的前提下引入通道注意力机制
    - 核心思路：旋转等变特征 $\mathbf{X} \in \mathbb{R}^{C \times H \times W}$ 可以重排为 $\mathbb{R}^{\frac{C}{N} \times N \times H \times W}$，其中 $N$ 是旋转方向数。RE-CA 仅产生 $C/N$ 个权重（而非 $C$ 个），每个权重重复 $N$ 次：
    $\boldsymbol{s} = \sigma(\mathbf{W} \cdot \boldsymbol{z}), \quad \mathbf{W} \in \mathbb{R}^{\frac{C}{N} \times C}$
      其中 $\boldsymbol{z}$ 为全局平均池化后的通道描述
    - 设计动机：直接在旋转等变特征上应用标准 SENet 通道注意力会破坏等变性（因为不同旋转方向的特征会被赋予不同权重）。RE-CA 通过共享权重保证等变性，同时参数量降低为原来的 $1/N$

3. **多分支检测头**:

    - 功能：利用旋转等变特征的分组特性减少参数量并提升精度
    - 核心思路：将旋转等变特征 $\mathbf{X} \in \mathbb{R}^{N \times \frac{C}{N} \times H \times W}$ 按旋转方向分为 $N$ 组，每组送入不同的检测头分支，最后拼接输出
    - 设计动机：旋转等变特征天然具有分组属性——同一卷积核在不同旋转方向生成的特征可以独立处理。多分支设计使得每个分支的输入通道数为 $C/N$，显著减少检测头参数量

### 损失函数 / 训练策略

- 使用与 RTMDet 相同的检测损失（GFL 分类损失 + GIoU 回归损失）
- AdamW 优化器，在 DOTA-v1.0/v1.5 和 DIOR-R 上训练 36 个 epoch
- 旋转方向数 $N=8$（参照 ReDet）
- 骨干网络在 ImageNet-1K 上预训练 300 epoch

## 实验关键数据

### 主实验

| 方法 | 参数量 | DOTA-v1.0 mAP | 说明 |
|------|--------|--------------- |------|
| RTMDet | 52.3M | 78.85 | 基线（常规 CNN） |
| ReDet | 31.6M | 76.25 | 近似旋转等变 |
| LSKNet | 31.0M | 77.49 | 大核卷积 |
| PKINet | 30.8M | 78.39 | 当前 CNN SOTA |
| Appr. MessDet | **15.3M** | 78.45 | 近似旋转等变 |
| **Str. MessDet** | **18.1M** | **79.12** | 严格旋转等变 |

MessDet 以 15.3M-18.1M 参数量（仅为 RTMDet 的 1/3）达到 SOTA 性能。严格等变版本（79.12 mAP）比近似等变版本（78.45 mAP）高 0.67 mAP。

### 消融实验

| 配置 | 参数量 | mAP | 说明 |
|------|--------|------|------|
| Str. MessDet + RE-CA | 19.0M | 78.51 | 完整配置 |
| Str. MessDet w/o RE-CA | 18.8M | 76.91 | 无通道注意力，-1.60 |
| Appr. MessDet + RE-CA | 16.2M | 78.15 | 近似等变 |
| Appr. MessDet w/o RE-CA | 16.0M | 77.47 | -0.68 |
| RTMDet Head (2 conv) | 2.4M | 78.15 | 标准检测头 |
| Multi-branch Head (3 conv) | **1.5M** | **78.45** | 参数减少 37%，精度提升 |

RE-CA 为严格等变模型带来 1.60 mAP 提升；多分支头在减少 37% 参数的同时提升 0.30 mAP。

### 关键发现

1. **严格 vs 近似旋转等变**：在 MessDet (RE-Net) 上，严格等变比近似等变提升明显（+0.67 mAP），但在常规 CNN（RTMDet）上影响很小（+0.24 mAP），说明等变性对 RE-Net 更关键
2. **旋转等变误差随训练变化**：近似等变模型在训练过程中，浅层的旋转等变误差逐渐减少（模型学习到近似等变），但深层误差可能增加
3. **参数效率极高**：RE-Net 的权重共享特性（同一卷积核旋转 $N$ 次）+ 多分支头使 MessDet 成为目前参数量最少的 SOTA 航空检测器
4. 在 DOTA-v1.5（包含 <10 像素的小目标）和 DIOR-R 上同样达到 SOTA

## 亮点与洞察

- **首次定量回答了"严格 vs 近似旋转等变性"对航空检测的影响**——这是之前研究的空白
- **工程设计精妙**：通过调谐层实现严格等变的思路简洁有效，避免了 FRED 的单侧填充问题
- **参数效率惊人**：18.1M 参数即达到 79.12 mAP，而 RTMDet 需要 52.3M 才达到 78.85——展示了旋转等变网络在航空场景的巨大优势
- 多分支检测头巧妙利用了旋转等变特征的天然分组属性

## 局限与展望

1. 仅支持 $C_N$ 循环群的离散旋转等变（如 $N=8$ 即 45° 间隔），无法处理连续旋转
2. 调谐层引入额外参数和计算开销（18.1M vs 15.3M），虽然总体仍很轻量
3. 未与 Transformer-based 的航空检测器（如 ViT 变体）进行比较
4. 严格等变性在某些类别上并不一定优于近似等变性（如直升机 HC 类在近似版本中更高）

## 相关工作与启发

- 基于 E2CNN 的群等变卷积理论，将其首次与现代检测架构（RTMDet）深度融合
- ReDet 和 FRED 是该方向的先驱工作，MessDet 在理论分析和架构设计上更进一步
- LSKNet/PKINet 代表了另一条通过大核卷积增强旋转鲁棒性的路线，MessDet 以更少参数实现了相当或更高的精度

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次定量对比严格/近似等变性，RE-CA 和多分支头设计巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ DOTA-v1.0/v1.5、DIOR-R 三个数据集，充分消融，旋转误差追踪分析
- 写作质量: ⭐⭐⭐⭐ 理论清晰，实验设计合理
- 价值: ⭐⭐⭐⭐ 为航空检测的旋转等变性设计提供了重要参考，参数效率优势突出

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Rotation Invariant and Symmetry Aware Pixel Difference Network for Remote Sensing Object Detection](../../CVPR2026/object_detection/rotation_invariant_and_symmetry_aware_pixel_difference_network_for_remote_sensin.md)
- [\[AAAI 2026\] VK-Det: Visual Knowledge Guided Prototype Learning for Open-Vocabulary Aerial Object Detection](../../AAAI2026/object_detection/vk-det_visual_knowledge_guided_prototype_learning_for_open-vocabulary_aerial_obj.md)
- [\[ICCV 2025\] Adversarial Attention Perturbations for Large Object Detection Transformers](adversarial_attention_perturbations_for_large_object_detection_transformers.md)
- [\[ICCV 2025\] SFUOD: Source-Free Unknown Object Detection](sfuod_source-free_unknown_object_detection.md)
- [\[ICCV 2025\] Automated Model Evaluation for Object Detection via Prediction Consistency and Reliability](automated_model_evaluation_for_object_detection_via_prediction_consistency_and_r.md)

</div>

<!-- RELATED:END -->
