---
title: >-
  [论文解读] Multimodal Coreference Resolution for Chinese Social Media Dialogues: Dataset and Benchmark Approach
description: >-
  [多模态VLM] 提出 TikTalkCoref，首个面向中文社交媒体对话的多模态共指消解数据集（基于抖音短视频），并构建了包含文本共指消解、视觉人物追踪和跨模态对齐三个模块的 pipeline benchmark。 问题定义 多模态共指消解（MCR）旨在识别跨文本和视觉模态中指向同一实体的指称（mention）…
tags:
  - "多模态VLM"
---

# Multimodal Coreference Resolution for Chinese Social Media Dialogues: Dataset and Benchmark Approach

## 基本信息

- **会议**: ACL2025
- **arXiv**: [2504.14321](https://arxiv.org/abs/2504.14321)
- **代码**: GitHub (待发布)
- **领域**: 多模态视觉语言模型 (Multimodal VLM)
- **关键词**: 多模态共指消解, 中文社交媒体, 抖音, 视频文本对齐, 数据集构建

## 一句话总结

提出 TikTalkCoref，首个面向中文社交媒体对话的多模态共指消解数据集（基于抖音短视频），并构建了包含文本共指消解、视觉人物追踪和跨模态对齐三个模块的 pipeline benchmark。

## 研究背景与动机

### 问题定义
多模态共指消解（MCR）旨在识别跨文本和视觉模态中指向同一实体的指称（mention），是理解多模态内容的关键任务。例如，对话中的"阿娇"、"她"、"钟欣潼"指向同一个人，同时需要将这些文本指称与视频帧中对应人物的头部区域关联起来。

### 现有不足

**数据匮乏**：现有 MCR 数据集主要集中于人机对话（如 J-CRe3）、电影叙述（如 MPII-MD）或图像描述（如 CIN），无法充分反映真实社交媒体中多模态交互的复杂性和多样性。

**语言覆盖偏向英/日**：大多数研究聚焦英语，中文 MCR 研究几乎空白。

**隐式视觉线索挑战**：在社交媒体真实对话中，说话者往往省略对可见物体外观或位置的描述，使得模型难以从文本中提取视觉线索来定位被提及的对象。

### 动机
构建一个基于真实社交媒体的中文多模态共指消解数据集，填补该领域的数据空白，并提供一个有效的 benchmark。

## 方法详解

### 整体框架
系统是一个三阶段 pipeline：
1. **文本共指消解模块**：从对话文本中提取 mention 并聚类
2. **视觉人物追踪模块**：在视频中检测人物头部区域并追踪聚类
3. **文本-视觉共指对齐模块**：用对比学习将文本 cluster 与视觉 cluster 关联

### 数据集构建 (TikTalkCoref)

**数据来源**：基于 TikTalk 数据集（源自抖音平台），从 367k 对话中随机选取 4000 样本，经人工筛选得到 1,012 个高质量对话。

**筛选标准**：
- 排除包含个人识别信息或敏感内容的样本
- 排除视频严重模糊/噪声、人脸不可辨的样本
- 排除不涉及人物实体共指的对话

**标注内容**：
- 文本 mention 标注（专有名词 33.51%、代词 44.41%、普通名词 22.08%）
- 文本 cluster 标注（含单例 singleton）
- 视频关键帧中人物头部区域 (bbox) 标注
- 名人/非名人分类标注

**标注流程**：三名标注员独立双标注 + 一名专家仲裁，标注间一致性 MUC 78.19。

**数据规模**：

| 统计项 | TikTalkCoref | TikTalkCoref-celeb |
|---|---|---|
| 对话数 | 1,012 | 338 |
| 总时长(min) | 519.65 | 158.33 |
| Mention 数 | 2,179 | 731 |
| Cluster 数 | 1,435 | 488 |
| Bbox 数 | 958 | 426 |

### 关键设计

**文本共指消解**：采用 Maverick 模型（SOTA pipeline 方法），基于 DeBERTa-Chinese-Large 编码器。
- Mention Detection：对每个 token 预测开始/结束概率，阈值 > 0.5 确定候选 mention
- Mention Clustering：coarse-to-fine 方法——先用双线性评分函数粗筛 Top-K 先行词，再用全连接层精细评分

**视觉人物追踪**：
- YOLOv5 头部检测 + DeepSORT 跨帧追踪
- MTCNN + MobileFaceNet 面部特征提取 → 余弦相似度 > 0.6 跨片段聚类

**跨模态对齐**：
- 使用 Chinese CLIP（ViT-B/16 图像编码器 + RoBERTa-wwm-Base 文本编码器）
- 对比学习：最大化匹配文本 cluster 与对应人物头部图像的相似度，最小化非匹配对的相似度

### 损失函数

$$\mathcal{L} = \mathcal{L}_{coref} + \mathcal{L}_{align}$$

- $\mathcal{L}_{coref} = \mathcal{L}_{start} + \mathcal{L}_{end} + \mathcal{L}_{clust}$（均为二元交叉熵）
- $\mathcal{L}_{align}$：归一化温度缩放交叉熵损失（来自 CLIP）

## 实验

### 主实验结果

**文本共指消解**（TikTalkCoref-celeb 测试集）：

| 模型 | MUC F1 | B³ F1 | CEAF_ϕ4 F1 | Avg F1 |
|---|---|---|---|---|
| e2e-coref | 66.02 | 30.61 | 20.59 | 39.07 |
| **Maverick** | **51.92** | **68.14** | **76.30** | **65.46** |

Maverick 显著优于 e2e-coref（p<0.001），尤其在 singleton 处理上（B³ 75.30 vs 17.47）。

**跨模态对齐**：

| 模型 | R@1 | R@2 | R@3 | Mean |
|---|---|---|---|---|
| R2D2 (zero-shot) | 52.50 | 71.50 | 76.25 | 66.67 |
| CN-Clip (zero-shot) | 45.00 | 65.00 | 68.75 | 59.58 |
| R2D2 (fine-tuning) | 56.25 | 73.75 | 80.00 | 70.00 |
| **CN-Clip (fine-tuning)** | **60.83** | **75.83** | **78.75** | **71.81** |

Zero-shot 下 R2D2 更强；fine-tuning 后 CN-Clip 在 R@1 和 Mean 上反超。

### 消融实验

**数据增强影响**：使用非名人数据（Train-all vs Train-celeb）训练文本共指模块：
- Maverick w/ DA: Avg F1 65.46 vs w/o DA: 52.67（+12.79）
- 非名人数据引入更多样化的语言上下文和 mention 类型，提升模型鲁棒性

**不同 Mention 类型的检索性能**：
- Name-central clusters: CN-Clip fine-tuning 70.52%  
- Pronoun-central clusters: CN-Clip fine-tuning 82.22%
- Noun-central clusters: R2D2 fine-tuning 60.61%（R2D2 预训练数据与普通名词-图像对更匹配）

### 关键发现
1. Maverick 的 pipeline 方法在处理大量 singleton 的数据集上表现优异
2. 数据增强（加入非名人数据）可显著提升名人共指消解性能
3. 社交媒体对话中的隐式视觉线索使跨模态对齐仍有较大提升空间

## 亮点与洞察

1. **首创性**：首个中文社交媒体多模态共指消解数据集，填补了该领域空白
2. **实际场景**：基于抖音真实短视频+用户评论对话，比现有数据集（人机对话/电影叙述）更贴近真实应用
3. **标注设计考究**：嵌套 mention 独立标注子 mention（区别于 OntoNotes 取最长 span），更准确反映指称关系
4. **三种 mention 类型全覆盖**：专有名词 + 普通名词 + 代词，比现有数据集更全面
5. **跨模态对齐中负样本设计**：使用同视频非匹配图像作为负样本，CN-Clip 在此设置下适配更好

## 局限性

1. **数据规模有限**：仅 1,012 个对话，且全部来自抖音平台，多样性受限
2. **监督学习局限**：在低资源场景下可能无法充分挖掘模型潜力，未来可探索半监督/无监督方法
3. **Benchmark 聚焦名人域**：仅在名人子集上做了完整评估，限制了泛化分析

## 相关工作

- **文本共指消解**：OntoNotes、LitBank、e2e-coref (Lee et al., 2017)、Maverick (Martinelli et al., 2024)
- **多模态共指消解**：MPII-MD (Rohrbach et al., 2017)、VisPro (Yu et al., 2019)、CIN (Goel et al., 2023)、J-CRe3 (Ueda et al., 2024)
- **视觉-语言对齐**：CLIP (Radford et al., 2021)、Chinese CLIP (Yang et al., 2022)

## 评分 ⭐⭐⭐⭐

- 创新性：⭐⭐⭐⭐ — 首个中文社交媒体 MCR 数据集，填补重要空白
- 实用性：⭐⭐⭐⭐ — 对社交媒体理解有直接应用价值
- 方法新颖度：⭐⭐⭐ — Pipeline 较直接，各模块均基于现有方法
- 实验充分度：⭐⭐⭐⭐ — 多维度分析全面，包括 mention 类型、数据增强、zero-shot/fine-tuning 对比

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Can Multimodal Large Language Models Understand Spatial Relations?](spatialmqa_mllm_spatial_relations.md)
- [\[ACL 2025\] Table Understanding and (Multimodal) LLMs: A Cross-Domain Case Study on Scientific Tables](table_understanding_and_multimodal_llms_a_cross-domain_case_study_on_scientific_.md)
- [\[ICLR 2026\] WebDS: An End-to-End Benchmark for Web-based Data Science](../../ICLR2026/multimodal_vlm/webds_an_end-to-end_benchmark_for_web-based_data_science.md)
- [\[ACL 2025\] Can Vision Language Models Understand Mimed Actions?](can_vision_language_models_understand_mimed_actions.md)
- [\[ACL 2025\] Teaching Vision-Language Models to Ask: Resolving Ambiguity in Visual Questions](teaching_vlm_ask_ambiguity.md)

</div>

<!-- RELATED:END -->
