---
title: >-
  [论文解读] Nautilus: A Large Multimodal Model for Underwater Scene Understanding
description: >-
  [NeurIPS 2025][多模态VLM][水下场景理解] 构建了首个支持八种水下场景理解任务的大型多模态模型 Nautilus，通过物理先验驱动的视觉特征增强（VFE）模块显式修复水下图像退化，提升 LMM 在水下环境中的鲁棒性。 领域现状：水下场景理解对海洋探索至关重要，涵盖目标检测、计数、图像描述等多粒度任务…
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "水下场景理解"
  - "大型多模态模型"
  - "视觉特征增强"
  - "水下成像模型"
  - "指令微调"
---

# Nautilus: A Large Multimodal Model for Underwater Scene Understanding

**会议**: NeurIPS 2025  
**arXiv**: [2510.27481](https://arxiv.org/abs/2510.27481)  
**代码**: [GitHub](https://github.com/H-EmbodVis/NAUTILUS)  
**领域**: 多模态视觉语言模型 (Multimodal VLM)  
**关键词**: 水下场景理解, 大型多模态模型, 视觉特征增强, 水下成像模型, 指令微调

## 一句话总结

构建了首个支持八种水下场景理解任务的大型多模态模型 Nautilus，通过物理先验驱动的视觉特征增强（VFE）模块显式修复水下图像退化，提升 LMM 在水下环境中的鲁棒性。

## 研究背景与动机

**领域现状**：水下场景理解对海洋探索至关重要，涵盖目标检测、计数、图像描述等多粒度任务。现有水下方法多为单任务设计，通用 LMM 直接应用于水下场景效果差。

**现有痛点**：(1) 通用LMM面临空中-水下域偏移问题；(2) 水下光散射和吸收导致严重图像退化；(3) 缺乏大规模水下多任务指令微调数据集。

**核心矛盾**：水下场景需要多粒度（图像级、区域级、目标级）的综合理解，但既缺数据又缺针对水下退化的有效处理机制。

**本文目标**：构建覆盖八种任务的水下指令微调数据集 NautData，并设计能显式处理水下图像退化的 LMM。

**切入角度**：利用水下成像物理模型的先验知识，在特征空间而非像素空间进行图像增强。

**核心idea**：基于暗像素先验量化后向散射影响、利用深度信息恢复光吸收衰减，构建即插即用的视觉特征增强模块。

## 方法详解

### 整体框架

Nautilus 由五个核心组件构成：图像编码器 $\mathcal{I}_v$、深度编码器 $\mathcal{I}_d$、视觉-语言投影器 $\mathcal{P}_{v-l}$、视觉特征增强（VFE）模块和 LLM。输入水下图像后，分别提取视觉特征和深度特征，VFE模块基于物理先验增强视觉特征，原始和增强特征并行通过共享投影器对齐到语言空间，最终由LLM完成多模态推理。

### 关键设计

1. **NautData 数据集构建**：包含 145 万图像-文本对，覆盖八种水下任务（粗粒度/细粒度分类、计数、VQA、检测、定位、区域描述、图像描述）。数据生成采用三种策略：基于规则的模板生成、整合生成（模板+LMM输出）、自由形式生成（LMM自由问答）。使用 Gemini 2.0 Flash 初始生成 + Qwen2.5-VL-72B 质量评估 + GPT-4o 测试集验证的多阶段质控流程。

2. **水下成像物理先验**：水下图像建模为直接反射 $\bm{D}_c$ 与后向散射 $\bm{B}_c$ 的叠加：
    $\bm{I}_c = \bm{D}_c + \bm{B}_c, \quad \bm{D}_c = \bm{J}_c e^{-\beta_c(\bm{z}) \cdot \bm{z}}$
   增强的目标是恢复无衰减原始色彩 $\bm{J}_c$：
    $\bm{J}_c = \frac{\bm{I}_c - \bm{B}_c}{e^{-\beta_c(\bm{z}) \cdot \bm{z}}}$
   利用暗像素先验量化后向散射强度，用深度信息拟合衰减系数。

3. **视觉特征增强（VFE）模块**：分两步工作——

    - **去后向散射**：找到平均 RGB 值最低的图像 patch 作为暗令牌 $\bm{f}_{v,k}$，通过交叉注意力层提取全局语义 $\bm{q}$，后向散射估计为 $\bm{s} = \bm{f}_{v,k} - \bm{q}$，从全局特征中逐像素减去。
    - **恢复光吸收**：用 MLP 从深度特征预测吸收权重 $\bm{W} = \text{MLP}(\bm{d})$，最终增强特征：
    $\bm{v}_e = (\bm{v} - \bm{s}) \oslash \exp(-\bm{W})$

4. **双路径特征融合**：原始视觉特征保留真实水下环境信息，增强特征减少成像干扰。两者通过共享投影器并行输入 LLM，实现互补理解。

### 损失函数 / 训练策略

- 采用参数高效微调（PEFT）策略，可训练组件包括视觉-语言投影器、LoRA（rank=128）和 VFE 模块
- 在 LLaVA-1.5 和 Qwen2.5-VL 两个基线上适配
- 训练 1 个 epoch，4×A800-80GB GPU，约 3 天

## 实验关键数据

### 主实验

在 NautData 测试集上与知名 LMM 对比：

| 方法 | 粗分类 acc | 细分类 acc | 图像描述 METEOR | 定位 PR@0.5 | 检测 mAP@0.5 | VQA METEOR |
|------|-----------|-----------|----------------|------------|-------------|-----------|
| GPT-4o (零样本) | 55.2 | 54.4 | 0.179 | 4.3 | 1.4 | 0.242 |
| Qwen2.5-VL-72B (零样本) | 55.2 | 54.2 | 0.171 | 46.4 | 14.7 | 0.222 |
| LLaVA-1.5 | 90.0 | 89.8 | 0.208 | 48.2 | 19.0 | 0.359 |
| Qwen2.5-VL | 85.3 | 88.2 | 0.222 | 57.6 | 41.7 | 0.380 |
| **Nautilus (Qwen2.5-VL)** | **90.3** | **93.8** | **0.223** | **58.8** | **45.3** | **0.381** |

### 消融实验

逐步添加模块的消融结果（Qwen2.5-VL 基线）：

| Baseline | 深度编码器 | 恢复光吸收 | 去后向散射 | 粗分类 acc | 细分类 acc | 定位 PR@0.5 | 检测 AP@0.5 |
|----------|-----------|-----------|-----------|-----------|-----------|------------|------------|
| ✔ | - | - | - | 87.9 | 89.1 | 55.4 | 35.9 |
| ✔ | ✔ | - | - | 89.5 | 89.1 | 55.0 | 36.4 |
| ✔ | ✔ | ✔ | - | 85.7 | 91.2 | 53.9 | 34.2 |
| ✔ | ✔ | ✔ | ✔ | **90.0** | **91.4** | **55.9** | **36.2** |

### 关键发现

- 零样本商用 LMM（GPT-4o、Gemini 2.0 Flash）在水下场景表现远不如微调后的开源模型
- VFE 模块在两个基线上均能一致提升多数任务性能（Qwen2.5-VL 上细分类 +5.6%，检测 mAP@0.5 +3.6%）
- 在 MarineInst20M 上的零样本评估验证了泛化能力
- 对比像素空间增强方法（Reti-Diff、SMDR-IS 等），特征空间增强避免了信息损失

## 亮点与洞察

- 首次将物理成像模型的先验知识注入 LMM 的特征空间增强，思路新颖且有物理可解释性
- NautData 构建了覆盖八种任务的大规模水下指令数据集，填补了该领域的数据空白
- VFE 模块设计为即插即用，可灵活适配不同的 LMM 框架
- 暗像素先验→后向散射量化→特征减法的物理推导链路清晰

## 局限与展望

- 多任务联合优化时存在任务间冲突（如计数准确率指标略有下降）
- 仅在 7B/8B 规模模型上验证，更大规模模型的效果未知
- 深度估计依赖冻结的 Depth Anything V2，跨域泛化性待验证
- 暗像素先验假设在极端水下条件（如完全黑暗或强光照）下可能失效

## 相关工作与启发

- MarineGPT：首个公开水下 LMM，但仅支持图像级理解
- MarineInst20M：大规模水下视觉语言数据集，支持目标级描述
- 物理模型指导深度学习的范式可推广到其他退化场景（如雾天、低光照）
- 特征空间增强优于像素空间增强的发现对其他域适应任务有参考价值

## 评分

- 新颖性: ⭐⭐⭐⭐ 物理先验驱动的特征空间增强有创新，但整体框架基于已有 LMM 改造
- 实验充分度: ⭐⭐⭐⭐⭐ 八种任务全面评估，消融充分，零样本泛化验证完整
- 写作质量: ⭐⭐⭐⭐ 结构清晰，物理推导详尽，图表丰富
- 价值: ⭐⭐⭐⭐ 水下场景理解领域开创性工作，数据集和方法均有实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] See&Trek: Training-Free Spatial Prompting for Multimodal Large Language Model](seetrek_training-free_spatial_prompting_for_multimodal_large_language_model.md)
- [\[CVPR 2026\] RE-VLM: Event-Augmented Vision-Language Model for Scene Understanding](../../CVPR2026/multimodal_vlm/re-vlm_event-augmented_vision-language_model_for_scene_understanding.md)
- [\[NeurIPS 2025\] Towards Comprehensive Scene Understanding: Integrating First and Third-Person Views for LVLMs](towards_comprehensive_scene_understanding_integrating_first_and_third-person_vie.md)
- [\[NeurIPS 2025\] Situat3DChange: Situated 3D Change Understanding Dataset for Multimodal Large Language Models](situat3dchange_situated_3d_change_understanding_dataset_for_multimodal_large_lan.md)
- [\[CVPR 2025\] Embodied Scene Understanding for Vision Language Models via MetaVQA](../../CVPR2025/multimodal_vlm/embodied_scene_understanding_for_vision_language_models_via_metavqa.md)

</div>

<!-- RELATED:END -->
