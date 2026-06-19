---
title: >-
  [论文解读] Revisiting Adversarial Patch Defenses on Object Detectors: Unified Evaluation, Large-Scale Dataset, and New Insights
description: >-
  [ICCV 2025][目标检测][对抗补丁防御] 系统性重新审视 11 种对抗补丁防御方法，建立首个补丁防御基准（含 13 种攻击、11 个检测器、4 种度量），构建 94,000 张图像的大规模 APDE 数据集，并揭示三个关键新发现：自然补丁防御难点在于数据分布而非高频、补丁检测精度与防御性能不一致、自适应攻击可绕过大多数现有防御。
tags:
  - "ICCV 2025"
  - "目标检测"
  - "对抗补丁防御"
  - "基准评测"
  - "大规模数据集"
  - "自适应攻击"
---

# Revisiting Adversarial Patch Defenses on Object Detectors: Unified Evaluation, Large-Scale Dataset, and New Insights

**会议**: ICCV 2025  
**arXiv**: [2508.00649](https://arxiv.org/abs/2508.00649)  
**代码**: [https://github.com/Gandolfczjh/APDE](https://github.com/Gandolfczjh/APDE)  
**领域**: 对抗鲁棒性 / 目标检测  
**关键词**: 对抗补丁防御, 目标检测, 基准评测, 大规模数据集, 自适应攻击

## 一句话总结

系统性重新审视 11 种对抗补丁防御方法，建立首个补丁防御基准（含 13 种攻击、11 个检测器、4 种度量），构建 94,000 张图像的大规模 APDE 数据集，并揭示三个关键新发现：自然补丁防御难点在于数据分布而非高频、补丁检测精度与防御性能不一致、自适应攻击可绕过大多数现有防御。

## 研究背景与动机

对抗补丁攻击是 DNN 在物理世界中面临的重大安全威胁，尤其在行人检测、自动驾驶等场景。近年来大量防御方法被提出，但现有评估存在四大问题：

**缺乏统一框架**：不同论文使用不一致的参数、攻击方法和补丁放置策略，无法公平比较

**度量不合适**：部分工作仅用补丁检测精度评估防御效果，但高检测精度不等于好的防御性能

**分析不全面**：忽视实时性、不同补丁尺寸/类型的影响、物理世界适用性等关键因素

**攻击不够强**：现有补丁数据集规模小、缺乏按检测器分类的覆盖

## 方法详解

### 整体框架

建立一个完整的评估流水线：攻击生成 → 数据集构建 → 统一评估 → 综合分析。核心贡献是 APDE 数据集和基准评测框架。

### 关键设计

1. **APDE 数据集构建**:

    - 使用 13 种攻击方法在 11 个检测器上进行白盒攻击，生成 94 种补丁
    - 补丁应用于 INRIA-Person 和 MS COCO 测试集，共 94,000 张图像
    - 训练集 56,400 张、测试集 37,600 张（6:4 分割）
    - 对比现有数据集：Apricot (60 种补丁, 1,011 图) 和 GAP (25 种, 9,266 图)
    - 优势：大规模、多样补丁分布、白盒设置（最坏情况评估）

2. **评估度量体系**:

    - **AP@0.5**：被攻击目标的平均精度，直接反映防御效果
    - **ASR（攻击成功率）**：衡量攻击剩余效果
    - **mIoU（SmIoU / NmIoU）**：替代补丁 AP@0.5，适用于不规则形状补丁
    - **推理时间**：评估计算实用性
    - 核心发现：被攻击目标的 AP 比补丁检测精度更能反映真实防御能力

3. **防御方法分类与评估**:

    - 补丁检测/分割型：SAC、PAD、Adyolo、NAPGuard
    - 基于补丁先验知识型：LGS、Zmask、Jedi
    - 基于生成模型型：DIFFender、NutNet
    - 认证防御型：DetectorGuard、ObjectSeeker
    - 覆盖隐藏攻击和出现攻击两种目标

### 损失函数 / 训练策略

对抗补丁生成使用通用目标：$\delta^* = \arg\min_\delta \mathbb{E}_{x \sim X}[\mathcal{L}(f_i(\mathcal{A}(x, \delta, t)), y)] + \lambda L_{tv}(\delta)$，其中 $L_{tv}$ 为全变分损失以生成更平滑补丁。

## 实验关键数据

### 主实验

11 个检测器上的隐藏攻击防御性能（Person AP@0.5）：

| 防御方法 | 类型 | Overall Mean | Overall Min | 推理时间(ms) |
|---------|------|-------------|-------------|-------------|
| w/o defense | - | 30.74 | - | - |
| SAC | 补丁分割 | 60.88 | 20.62 | 44 |
| PAD | 补丁分割 | 76.12 | 40.99 | 32,100 |
| Adyolo | 补丁检测 | 63.12 | 26.59 | 62 |
| NAPGuard | 补丁检测 | 75.94 | 46.93 | 59 |
| DIFFender | 生成模型 | 56.23 | 11.98 | 1,240 |
| **NutNet** | **生成模型** | **76.53** | **55.79** | **71** |
| LGS | 先验知识 | 71.58 | 29.55 | 82 |
| Zmask | 先验知识 | 56.71 | 6.43 | 417 |
| Jedi | 先验知识 | 58.85 | 18.96 | 349 |

### 消融实验

APDE 数据集重训练前后防御性能对比（YOLOv3 + FRCNN 平均 AP@0.5）：

| 攻击方法 | SAC原始 | SAC重训 | NAPGuard原始 | NAPGuard重训 |
|---------|--------|--------|-------------|-------------|
| T-SEA | 51.82 | 71.61 | 83.61 | 86.31 |
| TC-EGA | 58.16 | 71.36 | 68.51 | 85.30 |
| AdvPatch | 56.53 | 73.29 | 78.45 | 85.10 |
| GNAP (自然补丁) | 70.03 | 76.86 | 78.96 | 85.42 |
| AdvCloak (域外) | 4.17 | 71.29 | 52.21 | 73.16 |
| AdvTshirt (域外) | 34.27 | 64.47 | 50.21 | 70.89 |

平均提升 **15.09% AP@0.5**，域外补丁提升尤为显著。

### 关键发现

1. **自然补丁难防御的原因是数据分布而非高频**：NAP（自然对抗补丁）和非 NAP 的高频分量差异不大，但 FID 距离显著不同。防御方法本质上依赖数据分布判断像素是否为补丁
2. **补丁检测精度 ≠ 防御效果**：NAPGuard 检测精度最高但防御效果不如 NutNet；AP 比 mIoU 更能反映防御性能
3. **自适应攻击可绕过大部分防御**：PAD（复杂模型 SAM）和 DIFFender（随机性扩散模型）较鲁棒；利用通用补丁属性（特征过激活、高熵）的 Zmask 和 Jedi 也较鲁棒
4. **物理世界防御有效**：数字域表现好的方法在物理世界也通常有效；距离增大和光照增强有利于防御
5. **多补丁场景**：认证防御的性能随补丁数量增加下降较小，但计算成本呈指数增长

## 亮点与洞察

- **首个系统性补丁防御基准**：统一了评估范式，解决了长期以来各论文自说自话的问题
- **数据分布视角的新洞察**：推翻了"高频特征是自然补丁难防御的原因"这一普遍认知
- **实用性导向**：不仅评估，APDE 数据集还能直接用于提升现有防御性能
- **NutNet 表现最优**：综合防御效果、推理速度和鲁棒性最佳

## 局限与展望

- 主要聚焦行人检测，其他目标类别的泛化性有待验证
- 认证防御受限于严格的威胁模型假设（补丁数量、大小），实用性不足
- 物理世界实验仅使用 iPhone16pro 拍摄，传感器多样性有限
- 未覆盖3D 对抗攻击、视频帧间一致性等更复杂场景

## 相关工作与启发

- 可以借鉴数据分布视角来设计新的防御方法，如用分布距离来检测补丁区域
- 生成模型类防御（NutNet、DIFFender）的成功提示基于扩散模型的图像修复/去噪可能是有前景的防御范式
- APDE 数据集的构建思路可推广到其他对抗鲁棒性研究领域

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首个统一基准，数据分布的新发现有价值
- **实验充分度**: ⭐⭐⭐⭐⭐ 11 种防御 × 13 种攻击 × 11 个检测器，极为全面
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，分析深入，发现有说服力
- **实用价值**: ⭐⭐⭐⭐⭐ 数据集和基准对该领域有巨大推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Kaputt: A Large-Scale Dataset for Visual Defect Detection](kaputt_a_large-scale_dataset_for_visual_defect_detection.md)
- [\[ICCV 2025\] Adversarial Attention Perturbations for Large Object Detection Transformers](adversarial_attention_perturbations_for_large_object_detection_transformers.md)
- [\[ICCV 2025\] Large-scale Pre-training for Grounded Video Caption Generation](large-scale_pre-training_for_grounded_video_caption_generation.md)
- [\[ICCV 2025\] Automated Model Evaluation for Object Detection via Prediction Consistency and Reliability](automated_model_evaluation_for_object_detection_via_prediction_consistency_and_r.md)
- [\[ICLR 2026\] ForestPersons: A Large-Scale Dataset for Under-Canopy Missing Person Detection](../../ICLR2026/object_detection/forestpersons_a_large-scale_dataset_for_under-canopy_missing_person_detection.md)

</div>

<!-- RELATED:END -->
