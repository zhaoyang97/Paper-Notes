---
title: >-
  [论文解读] Beyond Label Semantics: Language-Guided Action Anatomy for Few-shot Action Recognition
description: >-
  [ICCV 2025][视频理解][动作识别] 提出 Language-Guided Action Anatomy (LGA) 框架，利用大语言模型将动作标签解剖为原子级动作描述（主体-动作-对象三要素），同时在视频端通过聚类分割将帧序列划分为对应的原子动作阶段，在原子级别进行多模态融合和匹配，显著提升小样本动作识别性能。
tags:
  - "ICCV 2025"
  - "视频理解"
  - "动作识别"
  - "LLM"
  - "atomic action"
  - "多模态"
  - "metric learning"
---

# Beyond Label Semantics: Language-Guided Action Anatomy for Few-shot Action Recognition

**会议**: ICCV 2025  
**arXiv**: [2507.16287](https://arxiv.org/abs/2507.16287)  
**代码**: 无  
**领域**: 视频理解 / 小样本动作识别  
**关键词**: few-shot action recognition, LLM, atomic action, multimodal fusion, metric learning

## 一句话总结

提出 Language-Guided Action Anatomy (LGA) 框架，利用大语言模型将动作标签解剖为原子级动作描述（主体-动作-对象三要素），同时在视频端通过聚类分割将帧序列划分为对应的原子动作阶段，在原子级别进行多模态融合和匹配，显著提升小样本动作识别性能。

## 研究背景与动机

小样本动作识别（FSAR）旨在用极少标注样本对未见类别视频进行分类。近年来多模态方法（特别是引入文本信息）取得了一定进展，但现有方法通常仅利用动作标签的粗粒度语义。然而，人体动作包含丰富的细粒度信息——姿态变化、运动动态和物体交互在不同阶段呈现不同特征，这些关键知识无法仅通过动作标签充分挖掘。

作者观察到动作的关键因素包括：（1）主体（subject）、运动动态（motion）、对象交互（object）三个核心要素；（2）时间维度上，动作的起始、进行和结束阶段对类别判断至关重要。因此需要在查询-支持视频间进行细粒度对齐，确保动作的每个关键方面都被考虑。

## 方法详解

### 整体框架

LGA 框架包含三大核心模块：**动作解剖**（Action Anatomy）、**细粒度多模态融合**（Fine-grained Multimodal Fusion）和**多模态匹配**（Multimodal Matching）。输入视频经视觉骨干提取特征后，在文本端利用 LLM 将标签分解为原子动作描述，在视频端将帧序列分割为对应的原子动作阶段，随后在原子级别进行特征融合与匹配。

### 关键设计

1. **文本解剖（Textual Anatomy）**:

    - 利用 LLM 将每个动作标签分解为有序的原子动作描述序列
    - 每个描述显式编码主体、动作和对象三要素
    - 例如 "Jump into pool" → 起始："A person standing at the edge..."，进行："The person leaping off..."，结束："The person entering the water..."
    - 将原子描述输入文本骨干提取特征 $\{t_i\}_{i=1}^{L}$

2. **视觉解剖（Visual Anatomy）与 CLUSTER-Segment**:

    - 将帧特征序列 $\{f_i\}_{i=1}^{T}$ 划分为 $L$ 个原子动作阶段
    - 采用 CLUSTER-Segment 策略：初始化每帧为独立簇，计算相邻簇的余弦相似度，反复合并最相似的相邻簇直至剩余 $L$ 个
    - 相邻簇之间添加重叠帧以增强鲁棒性
    - 设计动机：不同于均匀分割，该策略能自适应捕捉不同时长的子动作

3. **细粒度多模态融合模块**:

    - 使用多头交叉注意力机制整合原子级视觉和文本特征
    - Query 由原子视觉特征 $f_{S_i}$ 与对应文本特征 $t_i$ 相加得到：$\mathbf{Q_i} = t_i + f_{S_i}$
    - Key 和 Value 由所有原子视觉特征拼接：$\mathbf{K} = \mathbf{V} = \text{concat}(\{f_{S_i}\}_{i=1}^{L})$
    - 每个原子动作特征既学习局部语义细节，又保持对全局时间结构的感知
    - 最终拼接所有阶段特征得到动作原型：$\tilde{f} = \text{concat}(\tilde{f}_{S_1}, \tilde{f}_{S_2}, \tilde{f}_{S_3})$

4. **多模态匹配模块**:

    - **Video-video 匹配**：提出 Aligned Bidirectional Mean Hausdorff Metric（AB-MHM），在原子动作级别对齐时序，计算查询与支持视频的距离
    - **Video-text 匹配**：将查询视频每阶段的平均池化特征与各类别文本特征计算相似度
    - 最终通过加权几何平均融合两种匹配结果：$p_{(y=i|q)} = (p^{v-v})^{\alpha} \times (p^{v-t})^{(1-\alpha)}$

### 损失函数 / 训练策略

- 训练阶段采用 episode-based 元学习策略
- 使用交叉熵损失 + 对比损失联合优化
- 训练时仅使用 video-video 匹配结果进行分类以保证稳定性
- 推理时使用完整的多模态匹配结果
- 视觉骨干采用 CLIP ViT-B/16 初始化，均匀采样 8 帧

## 实验关键数据

### 主实验

| 数据集 | 设置 | LGA | CLIP-FSAR | EMP-Net | 提升（vs CLIP-FSAR） |
|--------|------|-----|-----------|---------|---------------------|
| HMDB51 | 1-shot | **86.8** | 77.1 | 76.8 | +9.7 |
| HMDB51 | 5-shot | **89.3** | 87.7 | 85.8 | +1.6 |
| Kinetics | 1-shot | **95.2** | 94.8 | 89.1 | +0.4 |
| UCF101 | 1-shot | **98.2** | 97.0 | 94.3 | +1.2 |
| SSv2-Small | 1-shot | **58.9** | 54.6 | 57.1 | +4.3 |
| SSv2-Small | 5-shot | **69.3** | 61.8 | 65.7 | +7.5 |
| SSv2-Full | 1-shot | **63.8** | 62.1 | 63.1 | +1.7 |
| SSv2-Full | 5-shot | **74.4** | 72.1 | 73.0 | +2.3 |

### 消融实验

| Visual-An | Textual-An | V-V匹配 | V-T匹配 | HMDB51 1-shot | HMDB51 5-shot |
|-----------|------------|---------|---------|---------------|---------------|
| ✗ | ✗ | ✓ | ✗ | 75.8 | 87.7 |
| ✓ | ✗ | ✓ | ✗ | 79.9 | 86.0 |
| ✗ | ✓ | ✓ | ✗ | 79.6 | 87.2 |
| ✓ | ✓ | ✓ | ✗ | 80.8 | 88.2 |
| ✓ | ✓ | ✗ | ✓ | 83.1 | 86.2 |
| ✓ | ✓ | ✓ | ✓ | **86.8** | **89.3** |

### 关键发现

- 视觉解剖和文本解剖单独使用就分别带来 4.1% 和 3.8% 的 1-shot 提升
- 多模态匹配（V-V + V-T）在 1-shot 场景下效果尤为显著（+6.0%），说明在视觉信息有限时文本线索更关键
- 原子动作数量为 3（起始/进行/结束）时最优，过多会导致 LLM 幻觉和时序重叠
- CLUSTER-Segment 优于均匀分割（HARD）和 TW-FINCH，因为能自适应不同时长的子动作

## 亮点与洞察

- **LLM 作为动作知识引擎**：巧妙地利用 LLM 的世界知识来解剖动作，而非简单使用标签文本，将隐含在动作标签中的丰富先验知识显式化
- **原子级对齐思想**：不同于全局级别的视觉-文本对齐，在子动作级别进行精细对齐，更符合人类理解动作的方式
- **AB-MHM 度量**：在 Hausdorff 距离中引入原子级时序对齐，非参数化设计使其具有良好的迁移性和计算效率

## 局限与展望

- 原子动作数量固定为 3，对于复杂动作可能不够灵活，可考虑自适应确定分段数
- LLM 生成的描述质量依赖于 LLM 模型能力，存在幻觉风险
- 单独使用视觉或文本解剖在 5-shot 设置下可能导致性能退化（模态不对齐问题）
- 未在更大规模的数据集和骨干网络上验证

## 相关工作与启发

- 与 SAFSAR 等使用扩展描述的方法相比，LGA 的原子级分解能更好地捕捉动作的时序结构
- CLUSTER-Segment 的思想可推广到其他需要视频时序分段的任务
- 多模态匹配的互补性启示：不同数据集对不同匹配模式敏感度不同（HMDB51 更受益于文本匹配，SSv2 更受益于视觉匹配）

## 评分

- 新颖性: ⭐⭐⭐⭐ 动作解剖+原子级融合匹配的创新组合，利用LLM挖掘动作先验知识
- 实验充分度: ⭐⭐⭐⭐⭐ 5个基准、详细消融、多维度分析和可视化
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法阐述系统完整
- 价值: ⭐⭐⭐⭐ 为小样本动作识别中如何利用LLM提供了新范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Trokens: Semantic-Aware Relational Trajectory Tokens for Few-Shot Action Recognition](trokens_semantic-aware_relational_trajectory_tokens_for_few-shot_action_recognit.md)
- [\[CVPR 2026\] MPL: Match-guided Prototype Learning for Few-shot Action Recognition](../../CVPR2026/video_understanding/mpl_match-guided_prototype_learning_for_few-shot_action_recognition.md)
- [\[ICCV 2025\] Frequency-Semantic Enhanced Variational Autoencoder for Zero-Shot Skeleton-based Action Recognition](frequency-semantic_enhanced_variational_autoencoder_for_zero-shot_skeleton-based.md)
- [\[CVPR 2025\] Temporal Alignment-Free Video Matching for Few-Shot Action Recognition](../../CVPR2025/video_understanding/temporal_alignment-free_video_matching_for_few-shot_action_recognition.md)
- [\[CVPR 2025\] TAMT: Temporal-Aware Model Tuning for Cross-Domain Few-Shot Action Recognition](../../CVPR2025/video_understanding/tamt_temporal-aware_model_tuning_for_cross-domain_few-shot_action_recognition.md)

</div>

<!-- RELATED:END -->
