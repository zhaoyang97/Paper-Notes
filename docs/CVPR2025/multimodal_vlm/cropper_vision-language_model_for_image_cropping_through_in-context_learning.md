---
title: >-
  [论文解读] Cropper: Vision-Language Model for Image Cropping through In-Context Learning
description: >-
  [CVPR 2025][多模态VLM][图像裁剪] 本文提出Cropper框架，首次利用大型视觉-语言模型（VLM）的上下文学习（ICL）能力来解决图像裁剪任务，通过高效的prompt检索和基于反馈的迭代裁剪优化策略，无需任何训练即可在自由裁剪、主体感知裁剪和宽高比裁剪三种任务上大幅超越有监督SOTA方法。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "图像裁剪"
  - "视觉上下文学习"
  - "VLM"
  - "提示学习"
  - "迭代优化"
---

# Cropper: Vision-Language Model for Image Cropping through In-Context Learning

**会议**: CVPR 2025  
**arXiv**: [2408.07790](https://arxiv.org/abs/2408.07790)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 图像裁剪, 视觉上下文学习, VLM, prompt检索, 迭代优化

## 一句话总结
本文提出Cropper框架，首次利用大型视觉-语言模型（VLM）的上下文学习（ICL）能力来解决图像裁剪任务，通过高效的prompt检索和基于反馈的迭代裁剪优化策略，无需任何训练即可在自由裁剪、主体感知裁剪和宽高比裁剪三种任务上大幅超越有监督SOTA方法。

## 研究背景与动机
图像裁剪是摄影应用中的关键操作，旨在识别视觉上最具吸引力的裁剪区域。现有裁剪方法依赖在特定数据集上训练神经网络，存在两大问题：(1) **泛化性差**——训练好的模型难以适配新需求或新数据分布，需要重新训练；(2) **任务碎片化**——自由裁剪、主体感知裁剪、宽高比裁剪各需独立的网络架构和训练流程。

随着GPT-4o、Gemini等大型VLM的突破，ICL为下游任务适配提供了无需训练的新范式。然而VLM在图像裁剪上面临两个挑战：(1) ICL性能高度依赖上下文示例的质量，手动选择不可扩展；(2) 如何将美学概念注入VLM并非显而易见。

核心idea：**利用VLM的ICL能力统一解决三种裁剪任务**，通过自动化prompt检索选取语义相似的上下文示例，并用美学评分器提供迭代反馈来逐步优化裁剪质量。

## 方法详解

### 整体框架
Cropper由两个阶段组成：(1) **Visual Prompt Retrieval**——给定输入图像，从训练集中自动检索top-S个语义相似的图像及其GT裁剪坐标作为ICL示例；(2) **Iterative Crop Refinement**——VLM根据ICL示例生成R个候选裁剪，经评分器打分后反馈给VLM迭代优化L轮。所有信息通过结构化prompt传递，无需修改VLM参数。

### 关键设计
1. **Visual Prompt Retrieval（视觉prompt检索）**:

    - 图像相似度度量 $Q$：使用CLIP ViT-B/32对查询图像和数据集图像提取嵌入，以余弦相似度检索top-S相似图像
    - GT裁剪选择度量 $G$：根据任务不同——自由裁剪用MOS分数选最佳裁剪；主体裁剪用mask中心点L2距离选最相近mask；宽高比裁剪用aspect ratio匹配
    - 形式化：$\mathcal{Z} = \arg\max_{z_i \in \mathcal{D}} Q(z_q, z_i)$，$\mathcal{H} = \arg\max_{c_j \in C_j} G(z_q, c_j)$
    - 最优S=30（过少ICL示例信息不足，过多增加噪声和token成本）
    - 设计动机：语义相似的图像更可能有类似的最佳裁剪方式

2. **Iterative Crop Refinement（迭代裁剪优化）**:

    - VLM首先根据ICL示例生成R=6个候选裁剪坐标 $(s, x_1, y_1, x_2, y_2)$
    - 对每个候选裁剪进行三维度评分：VILA-R美学评分 + CLIP内容保留度 + 区域面积
    - 将候选裁剪图像和对应分数反馈给VLM，提示"生成具有高分的相似裁剪"
    - 迭代L=2轮（性能在2轮后饱和）
    - VLM温度设为0.05（低随机性产生更优IoU）
    - 设计动机：VLM缺乏对坐标系和美学标准的深层理解，需要通过显式反馈引导

3. **统一的多任务prompt设计**:

    - 裁剪坐标归一化到1-1000范围
    - 自由裁剪：5-tuple $(s, x_1, y_1, x_2, y_2)$ 含MOS分数
    - 主体裁剪：输入包含subject mask标注，输出4-tuple $(x_1, y_1, x_2, y_2)$
    - 宽高比裁剪：prompt中指定目标宽高比，输出4-tuple
    - 设计动机：统一的坐标表示和prompt模板使同一VLM支持三种裁剪任务

### 损失函数 / 训练策略
Cropper不涉及任何训练，完全基于ICL推理。评分函数是三个归一化到[0,1]的指标的组合：
- **VILA-R美学分**：评估构图、色彩对比、透视等
- **CLIP内容保留分**：原图与裁剪图CLIP嵌入的余弦相似度
- **面积分**：$A = \frac{H_{crop} W_{crop}}{HW}$，防止裁剪区域过小

## 实验关键数据

### 主实验（GAICD自由裁剪）

| 方法 | 是否无训练 | $Acc_{1/5}$ | $\overline{Acc}_5$ | $\overline{Acc}_{10}$ | $\overline{SRCC}$ |
|------|-----------|------------|---------------------|----------------------|-------------------|
| GAIC | ✗ | 68.2 | 63.1 | 81.6 | 0.849 |
| TransView | ✗ | 69.0 | 63.9 | 82.4 | 0.857 |
| Chao et al. | ✗ | 70.0 | 64.8 | 83.3 | 0.872 |
| **Cropper** | **✓** | **88.9** | **84.3** | **96.5** | **0.904** |

### 主体感知裁剪（SACD）

| 方法 | 是否无训练 | IoU↑ | Disp↓ |
|------|-----------|------|-------|
| SAC-Net | ✗ | 0.767 | 0.0491 |
| **Cropper** | **✓** | **0.769** | **0.0372** |

### 宽高比裁剪（FCDB）

| 方法 | 是否无训练 | IoU↑ | Disp↓ |
|------|-----------|------|-------|
| Mars | ✗ | 0.735 | 0.062 |
| **Cropper** | **✓** | **0.756** | **0.053** |

### 消融实验

| 配置 | IoU | $\overline{Acc}_5$ | 说明 |
|------|-----|---------------------|------|
| 完整Cropper (VILA+Area) | 0.748 | 84.3 | 最佳综合配置 |
| 仅VILA | 0.748 | 83.6 | 美学评分为主要贡献 |
| 仅Area | 0.752 | 83.9 | 面积分有独立贡献 |
| 仅CLIP | 0.751 | 81.2 | 内容保留对Acc贡献弱 |
| VILA+Area+CLIP | 0.754 | 84.3 | 与VILA+Area持平 |
| Random prompt替代检索 | - | 显著更差 | 验证了prompt检索的必要性 |
| S=1 (1个ICL示例) | ~0.72 | - | 示例太少效果差 |
| S=30 (最优) | ~0.75 | - | 最佳ICL示例数 |
| L=0 (无迭代) | ~0.72 | - | 验证迭代优化的必要性 |
| L=2 (最优) | ~0.75 | - | 2轮后性能饱和 |

### 关键发现
- **无训练方法首次大幅超越有监督方法**：Cropper在GAICD上 $Acc_{1/5}$ 从SOTA的70.0%提升到88.9%（+18.9个百分点）
- **ICL示例数量和检索策略至关重要**：随机选择ICL示例效果不稳定且差，CLIP检索30个示例最优
- **迭代优化提效显著**：从无迭代到2轮迭代IoU提升约3个百分点
- **GPT-4o零样本裁剪表现差**：直接使用GPT-4o做主体裁剪会裁掉主体关键部分
- **统一框架适用三种任务**：同一VLM + 不同prompt即可处理三种裁剪需求
- **VLM温度影响大**：0.05优于更高温度，低随机性产生更优裁剪

## 亮点与洞察
- **ICL用于坐标回归任务的成功案例**：图像裁剪本质上是预测bounding box坐标，VLM的ICL能力从NLP迁移到精确数值预测
- **反馈驱动迭代优化的普适思路**：用外部评分器对VLM输出打分并反馈，可推广到其他VLM下游任务
- **CLIP检索+VILA评分的组合**是一个实用的图像美学任务工具链
- **无训练超有训练的案例价值**：展示了大型VLM的ICL能力可以替代特定任务的有监督训练

## 局限与展望
- **成本高**：每张图需要30个ICL示例 + 6个候选 × 2轮迭代，对Gemini 1.5 Pro的API调用成本较大
- **依赖闭源VLM**：实验主要基于Gemini 1.5 Pro和GPT-4o，无法在本地部署
- **延迟高**：多轮VLM调用导致推理速度远慢于传统轻量裁剪网络
- GAICD的SRCC（0.904）在部分极端情况下不如PCC表现（0.860低于某些baseline的0.893）
- 未探索开源VLM（如LLaVA、Qwen-VL）在此框架下的表现
- prompt中的坐标归一化方案（1-1000）的选择缺乏消融

## 相关工作与启发
- **vs GAIC/TransView等有监督方法**: Cropper无需训练即大幅超越，说明VLM的通用视觉理解能力+恰当的ICL引导可以替代领域专用网络
- **vs 直接使用GPT-4o/Gemini裁剪**: 裸VLM（即使用CoT提示）在裁剪任务上表现差，说明ICL的示例引导和迭代反馈缺一不可
- **vs NLP ICL方法**: 将Liu et al.的语义相似检索策略从NLP成功迁移到视觉坐标回归任务

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将VLM ICL应用于图像裁剪，统一三种裁剪任务的框架设计巧妙
- 实验充分度: ⭐⭐⭐⭐ 三个数据集、三种裁剪任务、详细消融和user study全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，prompt设计展示完整
- 价值: ⭐⭐⭐⭐ 概念验证价值高，展示了VLM ICL在坐标预测任务上的潜力，但实用部署受限于成本和延迟

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Vision-Language Model IP Protection via Prompt-based Learning](vision-language_model_ip_protection_via_prompt-based_learning.md)
- [\[CVPR 2025\] What's in the Image? A Deep-Dive into the Vision of Vision Language Models](whats_in_the_image_a_deep-dive_into_the_vision_of_vision_language_models.md)
- [\[CVPR 2025\] CoLLM: A Large Language Model for Composed Image Retrieval](collm_a_large_language_model_for_composed_image_retrieval.md)
- [\[CVPR 2025\] Mimic In-Context Learning for Multimodal Tasks](mimic_in-context_learning_for_multimodal_tasks.md)
- [\[ACL 2025\] Transferring Textual Preferences to Vision-Language Understanding through Model Merging](../../ACL2025/multimodal_vlm/transferring_textual_preferences_to_vision-language_understanding_through_model_.md)

</div>

<!-- RELATED:END -->
