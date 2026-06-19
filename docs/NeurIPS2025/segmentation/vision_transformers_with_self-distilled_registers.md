---
title: >-
  [论文解读] Vision Transformers with Self-Distilled Registers
description: >-
  [NeurIPS 2025 Spotlight][语义分割][Transformer] 提出PH-Reg（Post Hoc Registers），一种高效的自蒸馏方法，无需标注数据和完整重训练即可为已有预训练ViT添加register token，通过测试时增强去噪教师特征+自蒸馏学生网络，有效消除ViT密集特征中的伪影token，提升分割和深度估计性能。
tags:
  - "NeurIPS 2025 Spotlight"
  - "语义分割"
  - "Transformer"
  - "Register Token"
  - "自蒸馏"
  - "特征去噪"
  - "开放词汇分割"
---

# Vision Transformers with Self-Distilled Registers

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2505.21501](https://arxiv.org/abs/2505.21501)  
**代码**: [GitHub](https://github.com/yinjiechen/PH-Reg)  
**领域**: 图像分割  
**关键词**: Vision Transformer, Register Token, 自蒸馏, 特征去噪, 开放词汇分割

## 一句话总结

提出PH-Reg（Post Hoc Registers），一种高效的自蒸馏方法，无需标注数据和完整重训练即可为已有预训练ViT添加register token，通过测试时增强去噪教师特征+自蒸馏学生网络，有效消除ViT密集特征中的伪影token，提升分割和深度估计性能。

## 研究背景与动机

Vision Transformer已成为视觉建模的主流架构，在分类、检测、分割等任务上展现出卓越的扩展性。然而近期研究发现ViT的密集特征中会出现**伪影token（artifact tokens）**——这些异常特征与局部图像语义不一致，破坏了精细化空间定位能力，对语义分割、部件对应等需要高空间精度的任务造成负面影响。

现有解决方案是在ViT中添加**register token**——随机初始化的可学习嵌入，参与自注意力计算但不在输出中使用。register token能有效"吸收"伪影项，使密集特征更干净。但这种方法要求**从头重新训练**模型，对于CLIP、DINOv2等大规模预训练模型而言计算成本不可接受。

核心动机是：**能否为已有的大规模预训练ViT事后补充register token？** 这需要解决两个问题：
1. 如何在不引入标注数据的情况下获得干净的训练目标？
2. 如何仅微调极少量参数即可有效消除伪影？

## 方法详解

### 整体框架

PH-Reg是一个自蒸馏框架：教师网络和学生网络从同一组预训练权重初始化。教师网络保持冻结和原始结构不变，通过测试时增强（TTA）生成去噪的密集特征作为蒸馏目标。学生网络仅额外引入register token和极少量可解冻参数，通过蒸馏学习产生干净的密集表征。

### 关键设计

1. **高效的教师特征去噪（Efficient Denoising of Teacher Representations）**: 核心观察是**伪影token不随图像内容静态移动**——如果图像偏移一定量，伪影不会做相同偏移。利用这一特性，对输入图像应用 $n$ 次随机增强（水平/垂直偏移和翻转），每次偏移量为patch大小 $k$ 的整数倍。对每次增强后的图像提取教师特征 $F_i = f_{\text{teacher}}(\mathcal{I}_i)$，然后通过逆变换恢复到原始坐标系，对同一位置的特征做加权平均。最终得到每个位置的去噪特征 $Q/K$（累积特征/计数）。这等价于最小化MSE的最优解，但完全无需梯度计算，**处理速度比基于神经场的DVT方法快约两个数量级**（<200ms）。

2. **学生网络设计（Design of the Student Network）**: 在原始ViT中添加 $m$ 个register token，共 $m + 1 + \frac{H}{k} \times \frac{W}{k}$ 个token参与自注意力。通过消融确定最优解冻策略：除register token外，还解冻位置嵌入、卷积patch embedding层和最后一个attention层。实验表明：(1) 即使只有1个register也能显著提升特征质量（1-register配置的99%分位余弦相似度超过raw情况的50%分位）；(2) 16个register为性价比最优选择；(3) 位置嵌入并非伪影的唯一来源（与先前DVT的观点不同）。

3. **蒸馏学习与优化（Learning and Optimization）**: 使用多目标蒸馏损失，结合余弦相似度和MSE确保方向和幅值的双重对齐：$\text{Loss}_{\text{total}} = 1 - \text{cossim}(\text{target}, \text{predicted}) + \text{MSE}(\text{target}, \text{predicted})$。整个训练过程仅需一组无标注图像。

### 损失函数 / 训练策略

蒸馏使用COCO Caption无标注图像集。教师网络冻结，使用10次增强生成目标。学生网络解冻register token、位置嵌入、卷积patch embedding和最后一个attention层。默认使用16个register token。训练无需分割/深度等标注。

## 实验关键数据

### 主实验

**开放词汇语义分割（mIoU%，OpenAI CLIP ViT-B/16）**:

| 方法 | VOC21 | PC60 | Object | VOC20 | PC59 | Stuff | City | ADE | 平均 |
|------|-------|------|--------|-------|------|-------|------|-----|------|
| MaskCLIP | 49.27 | 25.46 | 26.94 | 66.56 | 28.62 | 18.80 | 28.33 | 13.70 | 32.21 |
| SCLIP | 59.62 | 31.74 | 33.52 | 81.53 | 34.46 | 22.65 | 32.34 | 16.45 | 40.08 |
| NACLIP | 58.88 | 32.20 | 33.15 | 79.70 | 35.16 | 23.30 | 35.48 | 17.42 | 39.41 |
| NACLIP+DVT | 60.25 | 32.73 | 32.89 | 80.26 | 35.91 | 23.41 | 36.31 | 17.54 | 39.91 |
| **PH-Reg** | **63.01** | **34.52** | **35.27** | 83.05 | **37.88** | **24.66** | **37.17** | **19.22** | **41.85** |

PH-Reg在8个基准中的7个取得最佳，平均提升1.94%。

**线性探测分割与深度估计**:

| 方法 | VOC21 mIoU | ADE mIoU | NYUd RMSE↓ | NYUd δ₁↑ |
|------|-----------|---------|-----------|----------|
| CLIP | 73.88 | 35.78 | 0.6843 | 64.93 |
| CLIP+DVT | 74.74 | 36.39 | 0.6800 | 65.07 |
| **PH-Reg (CLIP)** | **75.32** | **38.07** | **0.6746** | **68.17** |
| DINOv2 | 84.13 | 47.82 | 0.4566 | 82.92 |
| DINOv2+DVT | 85.43 | 48.86 | 0.4329 | 85.23 |
| **PH-Reg (DINOv2)** | 84.85 | 48.66 | **0.4306** | **86.35** |

### 消融实验

| 消融配置 | VOC21 | 8基准平均 | 说明 |
|---------|-------|----------|------|
| Vanilla MaskCLIP | 49.27 | 32.21 | 基线 |
| Denoising only (10x aug) | 51.41 | 34.55 | 仅TTA去噪，+2.34 |
| Distill, no reg, no denoise | 61.16 | 40.68 | 蒸馏(无register/去噪) |
| Distill, with reg, no denoise | 61.27 | 40.66 | 加register但无去噪 |
| Distill, no reg, with denoise | 62.48 | 41.48 | 去噪但无register |
| **Full Pipeline** | **63.01** | **41.85** | register+去噪 |

约一半的提升来自register token，另一半来自教师去噪过程。

### 关键发现

- 伪影token并非总是高范数的——部分模型中伪影token范数反而低于正常token，挑战了先前的假设
- 位置嵌入不能完全解释伪影的产生（与DVT的假设不同），但解冻位置嵌入仍有正面效果
- DVT的静态伪影假设在CLIP类模型上不成立，导致DVT对这些模型的提升有限
- 偏移比例（shifting ratio）对去噪效果有影响，10-15%为最优范围
- PH-Reg可泛化应用于多种ViT骨干：OpenAI CLIP、OpenCLIP、DFN-CLIP、DINOv2

## 亮点与洞察

- **"取平均即去噪"**的核心思想极其简洁优雅：通过TTA+平均实现的去噪在数学上等价于MSE最优解，却完全避免了梯度计算
- 自蒸馏设计使得无需任何标注数据即可完成训练，极大降低了实用门槛
- 对伪影token本质的新发现（非静态、非总高范数）推进了社区对ViT内部机制的理解
- Register token引入的参数量极小（仅几千个参数），推理成本增加可忽略

## 局限与展望

- 当前仅在ViT-B/16上验证，更大模型（ViT-L、ViT-G）的效果未知
- 去噪过程需要多次前向传播（默认10次），虽然远快于DVT但仍增加推理成本
- 蒸馏和去噪的交互效应值得进一步分析——为什么两者的提升大致各占一半？
- ClearCLIP在VOC20上略优于PH-Reg，说明q-q attention的定位线索有其独特价值

## 相关工作与启发

- 与DVT（基于神经场的去噪）相比，PH-Reg更高效、更通用、假设更少
- Register token的事后添加范式可推广到其他自注意力架构（不限于视觉领域）
- TTA去噪的思想可与其他密集预测方法组合使用
- SCLIP/NACLIP等注意力修改方法与PH-Reg是正交的，有潜在组合空间

## 评分

- 新颖性: ⭐⭐⭐⭐ register token事后添加的思路实用且有创意，TTA去噪简洁有力
- 实验充分度: ⭐⭐⭐⭐⭐ 8个分割基准+多骨干验证+深度估计+全面消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰，实验设置公正，与基线使用完全相同的背景模型
- 价值: ⭐⭐⭐⭐⭐ 为大量已有预训练ViT提供了低成本的密集特征改善方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] LeGrad: An Explainability Method for Vision Transformers via Feature Formation Sensitivity](../../ICCV2025/segmentation/legrad_an_explainability_method_for_vision_transformers_via_feature_formation_se.md)
- [\[CVPR 2025\] DA-VPT: Semantic-Guided Visual Prompt Tuning for Vision Transformers](../../CVPR2025/segmentation/da-vpt_semantic-guided_visual_prompt_tuning_for_vision_transformers.md)
- [\[ICLR 2026\] Revisiting \[CLS\] and Patch Token Interaction in Vision Transformers](../../ICLR2026/segmentation/revisiting_cls_and_patch_token_interaction_in_vision_transformers.md)
- [\[CVPR 2025\] COSMOS: Cross-Modality Self-Distillation for Vision Language Pre-training](../../CVPR2025/segmentation/cosmos_cross-modality_self-distillation_for_vision_language_pre-training.md)
- [\[ICLR 2026\] Thicker and Quicker: A Jumbo Token for Fast Plain Vision Transformers](../../ICLR2026/segmentation/thicker_and_quicker_a_jumbo_token_for_fast_plain_vision_transformers.md)

</div>

<!-- RELATED:END -->
