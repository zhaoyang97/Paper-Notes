---
title: >-
  [论文解读] Your Large Vision-Language Model Only Needs a Few Attention Heads for Visual Grounding
description: >-
  [CVPR 2025][多模态VLM][视觉定位] 发现冻结 LVLM 中天然存在少量"定位头"（localization heads）持续捕捉文本语义对应的物体位置，仅用 3 个注意力头的注意力图即可实现超越微调 LISA-7B 的无训练视觉定位，RefCOCO val 达 86.5%。 领域现状：视觉定位（VG）要求模型…
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "视觉定位"
  - "注意力头发现"
  - "无训练方法"
  - "定位头"
  - "LVLM可解释性"
---

# Your Large Vision-Language Model Only Needs a Few Attention Heads for Visual Grounding

**会议**: CVPR 2025  
**arXiv**: [2503.06287](https://arxiv.org/abs/2503.06287)  
**代码**: 无（项目页待确认）  
**领域**: 多模态VLM  
**关键词**: 视觉定位、注意力头发现、无训练方法、定位头、LVLM可解释性

## 一句话总结
发现冻结 LVLM 中天然存在少量"定位头"（localization heads）持续捕捉文本语义对应的物体位置，仅用 3 个注意力头的注意力图即可实现超越微调 LISA-7B 的无训练视觉定位，RefCOCO val 达 86.5%。

## 研究背景与动机

**领域现状**：视觉定位（VG）要求模型定位图像中与文本描述对应的物体。现有方法要么在大量标注数据上训练专门的 grounding 头（如 LISA、CogVLM），要么使用外部检测器+CLIP 匹配。

**现有痛点**：训练定位方法需要大量 bounding box 标注和专门的微调。现有无训练方法（ReCLIP、GroundVLP）依赖外部模块，精度有限。关键问题是：LVLM 内部是否已经具备了空间定位能力？

**核心矛盾**：LVLM 经过大规模图文训练理应理解空间关系，但其平均注意力图通常是模糊的、难以提取定位信息。

**本文目标** 探索 LVLM 内部的空间定位机制——是否存在特定的注意力头天然捕捉物体位置？

**切入角度**：不看所有注意力头的平均，而是逐个分析——用注意力总和（文本 token 对图像 token 的注意力权重之和）和空间熵（注意力图的集中程度）两个指标筛选出"定位头"。

**核心 idea**：在冻结 LVLM 的数千个注意力头中，仅 3 个特定头的注意力图就足以实现高精度视觉定位，无需任何训练或外部模块。

## 方法详解

### 整体框架
输入图文对 → LVLM 前向推理 → 逐个分析每个注意力头 → 两步筛选（注意力总和阈值 + 空间熵排序）→ 选择 top-k 低空间熵头 → 组装其注意力图 → 预测 bounding box / mask。

### 关键设计

1. **注意力总和筛选**:

    - 功能：过滤掉不关注图像 token 的注意力头
    - 核心思路：计算每个头中文本 token 对图像 token 的注意力权重之和 $S_{img}$。在最大曲率处设阈值 $\tau$，$S_{img} < \tau$ 的头被排除
    - 设计动机：很多注意力头主要关注文本 token 之间的关系，它们对定位无用

2. **空间熵筛选**:

    - 功能：找出注意力分布最集中（最定位的）的头
    - 核心思路：将注意力图二值化 → 连通组件分析 → 计算熵 $H$。低熵意味着注意力集中在少数区域（精确定位），高熵意味着分散（模糊）。跨 1000 个样本统计每个头落入 top-10 低熵的频率
    - 设计动机：定位头应该对不同样本都表现出空间集中性，选择频率排序确保一致性

3. **选择频率的稳定性**:

    - 功能：识别跨样本一致的定位头
    - 核心思路：Spearman 相关系数 > 0.7 between 选择频率排名和 IoU，证明被选中频率高的头确实更擅长定位。仅 3 个头就足够（更多头的边际收益递减）
    - 设计动机：证明定位头是 LVLM 的固有属性，而非对特定样本的偶然现象

### 损失函数 / 训练策略
完全无训练——仅利用冻结 LVLM 的注意力图。

## 实验关键数据

### 主实验

| 方法 | 类型 | RefCOCO val | testA | testB | RefCOCO+ val |
|------|------|-----------|-------|-------|-------------|
| ReCLIP | 无训练 | 45.8 | 46.1 | 47.1 | 47.9 |
| GroundVLP | 无训练 | 65.0 | 73.5 | 55.0 | 68.8 |
| LISA-7B | 微调 | 74.1 | 76.5 | 71.1 | 62.4 |
| **Ours (LLaVA-7B)** | **无训练** | **86.5** | **89.8** | **80.2** | **80.1** |
| **Ours (LLaVA-13B)** | **无训练** | **87.2** | **90.0** | **83.3** | **82.7** |

### 消融实验

| 配置 | 效果 |
|------|------|
| 平均所有注意力头 | 极差（模糊） |
| Top-1 定位头 | 不够稳定 |
| Top-3 定位头 | **最优** |
| Top-10 定位头 | 轻微下降 |

### 关键发现
- **无训练超越微调**：仅用 3 个注意力头就在 RefCOCO 上超越 LISA-7B（86.5 vs 74.1），后者经过大量 grounding 数据微调
- **定位头跨架构一致存在**：在 10+ 种 LVLM（LLaVA、InternVL、Qwen-VL 等）中都发现了定位头
- **平均注意力完全无用**：数千头的平均产生模糊噪声图，只有个别头的注意力是锐利且准确的

## 亮点与洞察
- **"LVLM 天生会定位"的发现**极其重要——说明大规模图文训练已经在注意力机制中编码了空间理解，不需要专门的 grounding 训练
- **3 个头就够**：从数千个头中提炼出 3 个，信息利用效率惊人。这暗示可以用定位头做高效的 attention pruning

## 局限与展望
- 定位头的筛选需要一个小的验证集（1000 样本），非完全免数据
- 对复杂推理型定位（如"左边那个较大的物体"）可能不够，需要更深的语义理解
- 仅评估了 REC 任务，对 RES（referring expression segmentation）的效果需要额外的 mask 生成步骤

## 相关工作与启发
- **vs LISA**：LISA 增加 decoder 头并微调 LVLM。本文证明微调不是必要的——冻结模型的注意力就足够
- **vs CLIP 热力图方法**：CLIP 的全局对比特征不适合精确定位。LVLM 的自回归注意力天然编码了空间关系

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "定位头"的发现是对 LVLM 内部机制的重要洞察
- 实验充分度: ⭐⭐⭐⭐⭐ 10+ 种 LVLM、3 个 REC 数据集、详尽的筛选指标分析
- 写作质量: ⭐⭐⭐⭐⭐ 从假设到发现到验证的逻辑极为清晰
- 价值: ⭐⭐⭐⭐⭐ 对 VLM 可解释性和高效定位都有重大贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] VideoGLaMM: A Large Multimodal Model for Pixel-Level Visual Grounding in Videos](videoglamm_a_large_multimodal_model_for_pixel-level_visual_grounding_in_videos.md)
- [\[CVPR 2025\] Generalized Few-Shot 3D Point Cloud Segmentation with Vision-Language Model](generalized_few-shot_3d_point_cloud_segmentation_with_vision-language_model.md)
- [\[CVPR 2025\] MIMO: A Medical Vision Language Model with Visual Referring Multimodal Input and Pixel Grounding Multimodal Output](mimo_a_medical_vision_language_model_with_visual_referring_multimodal_input_and_.md)
- [\[CVPR 2025\] LamRA: Large Multimodal Model as Your Advanced Retrieval Assistant](lamra_large_multimodal_model_as_your_advanced_retrieval_assistant.md)
- [\[CVPR 2025\] ReVisionLLM: Recursive Vision-Language Model for Temporal Grounding in Hour-Long Videos](revisionllm_recursive_vision-language_model_for_temporal_grounding_in_hour-long_.md)

</div>

<!-- RELATED:END -->
