---
title: >-
  [论文解读] When Person Re-Identification Meets Event Camera: A Benchmark Dataset and An Attribute-guided Re-Identification Framework
description: >-
  [AAAI2026][自动驾驶][person re-identification] 构建首个大规模 RGB-Event 行人重识别数据集 EvReID（1200 ID / 118,988 图像对），并提出基于行人属性引导的三阶段对比学习框架 TriPro-ReID，通过正负属性 prompt 和跨模态 prompt 融合 RGB 与 Event 两种模态特征，mAP 达 69.3%。
tags:
  - "AAAI2026"
  - "自动驾驶"
  - "person re-identification"
  - "event camera"
  - "RGB-Event fusion"
  - "pedestrian attributes"
  - "benchmark dataset"
---

# When Person Re-Identification Meets Event Camera: A Benchmark Dataset and An Attribute-guided Re-Identification Framework

**会议**: AAAI2026  
**arXiv**: [2507.13659](https://arxiv.org/abs/2507.13659)  
**代码**: [Event-AHU/Neuromorphic_ReID](https://github.com/Event-AHU/Neuromorphic_ReID)  
**领域**: 自动驾驶  
**关键词**: person re-identification, event camera, RGB-Event fusion, pedestrian attributes, benchmark dataset

## 一句话总结
构建首个大规模 RGB-Event 行人重识别数据集 EvReID（1200 ID / 118,988 图像对），并提出基于行人属性引导的三阶段对比学习框架 TriPro-ReID，通过正负属性 prompt 和跨模态 prompt 融合 RGB 与 Event 两种模态特征，mAP 达 69.3%。

## 背景与动机

### 领域现状

**领域现状**：基于 RGB 摄像头的行人 ReID 在光照变化、运动模糊和隐私保护方面面临挑战

### 核心矛盾

**核心矛盾**：Event camera 具有低功耗、高动态范围、无运动模糊等优势，但现有 event-based ReID 数据集规模极小（Event-ReID 仅 33 ID / 16,000 样本），无法评估真实性能和泛化能力

### 现有痛点

**现有痛点**：现有方法仅关注 event 特征学习或 RGB-Event 特征融合，忽视了行人属性（长发、戴眼镜等）这类中层语义信息

### 解决思路

**本文目标**：如何构建大规模、真实的 RGB-Event 行人 ReID 基准数据集，并设计有效利用多模态视觉特征与行人属性语义信息的 ReID 框架？

## 方法详解

### EvReID 数据集
- 使用 DVS346 Event camera 采集，分辨率 $346 \times 260$
- 1200 ID、118,988 帧对（是 Event-ReID 的 7 倍图像 / 36 倍 ID）
- 覆盖多季节、多场景、日夜光照条件
- RGB 模态添加 11 种噪声（光照变化、运动模糊、恶劣天气）验证互补学习
- 70%/30% 训练/测试划分，single-shot 评估设置

### TriPro-ReID 框架（三阶段训练）

**Stage 1: Text Prompt Alignment**
- 基于 CLIP-ReID，为每个 ID 学习 text prompt token $[X]_1, ..., [X]_n$
- 冻结视觉/文本编码器，仅优化 ID-specific prompt
- 损失：$L_{stage1} = L_{v2t} + L_{t2v}$

**Stage 2: Multimodal Prompt Alignment**
- 引入 Cross-Modal Prompt (CMP)：在 RGB 分支初始化 learnable prompt token，通过 FC 层投影到 Event 分支
- CMP 在 Transformer 各层同步传播，实现持续的跨模态特征融合

**Stage 3: Visual-Modal Tuning with Attribute Prompts**
- 使用预训练行人属性识别模型 VTFPAR++ 预测属性
- 构建 Positive-Negative Attribute Prompt (PNAP)：正属性 "Male, Jacket, Bald" + 负属性 "Not Female, Not Short Sleeves"
- PNAP 编码后注入 ViT 中间层，动态调制视觉特征
- 损失：$L_{stage3} = L_{id} + L_{tri} + L_{v2t} + L_{t2v}$

## 实验关键数据

**EvReID 数据集上（V+E 模态）：**


### 主实验

| 方法 | mAP | Rank-1 | Rank-5 |
|------|-----|--------|--------|
| CLIMB-ReID | 68.3 | 85.2 | 92.8 |
| AP3D | 66.9 | 86.5 | 95.6 |
| **TriPro-ReID** | **69.3** | **88.6** | **94.3** |

**MARS\* 数据集上（V+E 模态）：**
- TriPro-ReID: mAP 88.4, Rank-1 91.1

**Ablation（EvReID）：**


### 消融实验

| 配置 | mAP | Rank-1 |
|------|-----|--------|
| Base only | 49.2 | 73.0 |
| +PNAP | 62.3 | 81.1 |
| +CMP | 50.2 | 75.2 |
| +PNAP+CMP | **69.3** | **88.6** |


### 关键发现

- PNAP 提升最大（+13.1 mAP），仅用 Positive prompt 为 54.4 mAP，正负属性组合效果显著

## 亮点与洞察
- **首个大规模真实 RGB-Event ReID 数据集**：规模是前作 36 倍 ID 数，多季节/多光照覆盖
- **正负属性 prompt 设计巧妙**：不仅利用 "有什么属性"，还利用 "没有什么属性" 作为判别线索，ablation 验证了负属性的关键作用
- **三阶段渐进训练策略**：从 text alignment → multimodal fusion → attribute tuning，逐步引入信息，训练稳定
- **15 个 SOTA baseline 系统评测**：为社区提供完善的 benchmark

## 局限与展望
- EvReID 仅 $346 \times 260$ 低分辨率，限制了细粒度特征学习
- 行人属性依赖外部预训练模型 VTFPAR++ 的预测质量
- Event 单模态性能远低于 RGB（如 AP3D: 40.6 vs 65.4 mAP），Event 特征利用还有大量空间
- 未探索 open-set 或 cross-domain 评估场景

## 相关工作与启发
- vs. SDCL（CVPR2023）：同为 RGB-Event 融合 ReID，但 SDCL 缺乏属性语义引导，mAP 低 15.1
- vs. CLIP-ReID（AAAI2023）：TriPro-ReID 在其基础上引入 CMP 和 PNAP，V+E 模态 mAP 从 49.2 提升到 69.3
- vs. CLIMB-ReID（AAAI2025）：使用 Mamba 架构，TriPro-ReID 在 mAP 和 Rank-1 上均优

## 相关工作与启发
- 正负属性 prompt 的思路可推广到其他细粒度识别任务（车辆 ReID、动物识别等）
- Event camera 在隐私保护场景（如智能监控）有独特价值，RGB-Event 融合是一个增长中的研究方向
- 三阶段 prompt 学习策略可为其他多模态 CLIP 下游任务提供参考

## 评分
- 新颖性: ⭐⭐⭐⭐ (数据集贡献大 + 正负属性 prompt 新颖)
- 实验充分度: ⭐⭐⭐⭐ (15 baseline + 完整 ablation + 双数据集)
- 写作质量: ⭐⭐⭐⭐ (结构完整，数据集描述详尽)
- 价值: ⭐⭐⭐⭐ (benchmark 数据集对社区有长期价值)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Hierarchical Prompt Learning for Image- and Text-Based Person Re-Identification](hierarchical_prompt_learning_for_image-_and_text-based_person_re-identification.md)
- [\[AAAI 2026\] Debiased Dual-Invariant Defense for Adversarially Robust Person Re-Identification](debiased_dual-invariant_defense_for_adversarially_robust_person_re-identificatio.md)
- [\[CVPR 2026\] FedBPrompt: Federated Domain Generalization Person Re-Identification via Body Distribution Aware Visual Prompts](../../CVPR2026/autonomous_driving/fedbprompt_federated_domain_generalization_person_re-identification_via_body_dis.md)
- [\[NeurIPS 2025\] GSAlign: Geometric and Semantic Alignment Network for Aerial-Ground Person Re-Identification](../../NeurIPS2025/autonomous_driving/gsalign_geometric_and_semantic_alignment_network_for_aerial-ground_person_re-ide.md)
- [\[AAAI 2026\] Difficulty-Aware Label-Guided Denoising for Monocular 3D Object Detection](difficulty-aware_label-guided_denoising_for_monocular_3d_object_detection.md)

</div>

<!-- RELATED:END -->
