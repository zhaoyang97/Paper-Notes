---
title: >-
  [论文解读] Dynamic Updates for Language Adaptation in Visual-Language Tracking
description: >-
  [CVPR 2025][多模态VLM][视觉语言跟踪] 提出DUTrack，通过动态更新多模态参考信息（模板帧+语言描述）来解决视觉语言跟踪中静态参考与动态目标之间的语义不一致问题，首次让VL跟踪器在LaSOT上超越最佳纯视觉跟踪器。 视觉语言(VL)跟踪依靠自然语言描述和模板帧来定位目标。然而现有方法存在一个根本性缺陷：多…
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "视觉语言跟踪"
  - "动态更新"
  - "多模态参考"
  - "大语言模型"
  - "目标跟踪"
---

# Dynamic Updates for Language Adaptation in Visual-Language Tracking

**会议**: CVPR 2025  
**arXiv**: [2503.06621](https://arxiv.org/abs/2503.06621)  
**代码**: [https://github.com/GXNU-ZhongLab/DUTrack](https://github.com/GXNU-ZhongLab/DUTrack)  
**领域**: 视频理解  
**关键词**: 视觉语言跟踪, 动态更新, 多模态参考, 大语言模型, 目标跟踪

## 一句话总结

提出DUTrack，通过动态更新多模态参考信息（模板帧+语言描述）来解决视觉语言跟踪中静态参考与动态目标之间的语义不一致问题，首次让VL跟踪器在LaSOT上超越最佳纯视觉跟踪器。

## 研究背景与动机

视觉语言(VL)跟踪依靠自然语言描述和模板帧来定位目标。然而现有方法存在一个根本性缺陷：**多模态参考信息是静态的**。具体表现为：

1. **语言描述固定** — 初始语言标注只能描述目标在某一时刻的状态，无法反映整个视频中目标的外观变化（如颜色改变、姿态变化、尺度变化）
2. **模板帧固定** — 初始模板帧只捕获目标开始时的外观，长期跟踪中逐渐偏离实际状态

这导致VL跟踪器的性能一直没能超过最好的纯视觉跟踪器（如ODTrack、AQATrack），使得语言信息在长序列跟踪中被浪费。作者认为核心原因就在于**静态参考与动态目标之间的语义鸿沟**。

## 方法详解

### 整体框架

DUTrack由四个主要组件构成：(1) 多模态交互模块 — 使用one-stream架构统一处理视觉和语言特征；(2) 动态模板捕获模块(DTCM) — 从搜索帧中提取与语言高匹配的区域作为动态模板；(3) 动态语言更新模块(DLUM) — 利用LLM生成当前目标的动态语言描述；(4) 跟踪头 — 输出bbox预测。

### 关键设计

1. **多模态交互模块 (One-Stream Multi-modal Interaction)**:
    - 功能：统一提取和融合视觉与语言特征
    - 核心思路：采用HiViT作为backbone，通过3阶段下采样（$4\times4$ embedding + 两个 $2\times2$ merging）将搜索帧和模板帧转为tokens $S_t \in \mathbb{R}^{N_S \times D}$，语言通过BERT tokenizer转为 $L_t \in \mathbb{R}^{N_L \times D}$（$N_L=16, D=512$），然后拼接后送入统一的多头自注意力进行交互
    - 设计动机：One-stream架构比Two-stream更高效，在相同ViT-base backbone下，DUTrack以69.9M参数实现43.5fps，而JointNLT/MMTrack分别需要153M/176.9M参数

2. **动态模板捕获模块 (DTCM)**:
    - 功能：从搜索帧中捕获与语言描述高度匹配的图像区域，作为动态模板更新
    - 核心思路：利用多头自注意力中[CLS] token对搜索区域的注意力图 $A_{l2s} = \text{Softmax}(\frac{Q_{CLS} \cdot K_S^T}{\sqrt{d}})$，选择注意力得分最高的top-k个patches，将其索引对应的图像区域作为动态模板。这些patches代表了与当前语言描述最匹配的目标最新外观
    - 设计动机：注意力权重天然编码了语言与视觉的匹配程度，无需额外计算即可获得高质量的动态模板；top-k=3为最佳选择

3. **动态语言更新模块 (DLUM)**:
    - 功能：在跟踪过程中动态生成描述目标当前状态的语言标注
    - 核心思路：设计基于目标变化的更新策略，通过比较当前帧结果 $r_i: [x_2, y_2, w_2, h_2]$ 与上次更新时的记录 $r_{stamp}: [x_1, y_1, w_1, h_1]$ 之间的三类变化：尺度变化 $\Delta S = \frac{w_1 h_1}{w_2 h_2}$、位移变化 $\Delta D = \sqrt{(x_1-x_2)^2 + (y_1-y_2)^2}$、颜色变化 $\Delta C = \sqrt{(R_1-R_2)^2 + (G_1-G_2)^2 + (B_1-B_2)^2}$。当变化超过阈值时，使用BLIP生成新的语言描述
    - 设计动机：不是每帧都需要更新语言描述（太频繁会增加开销），而是在目标外观发生显著变化时才更新，既保证信息时效性又控制计算成本

### 损失函数 / 训练策略

**两阶段训练**：
- **第一阶段**（150 epochs）：不使用语言信息，在LaSOT、GOT-10K、COCO、TrackingNet、TNL2K上训练纯视觉跟踪能力，AdamW优化器，学习率和权重衰减均为 $1 \times 10^{-4}$，每epoch 60K样本
- **第二阶段**（50 epochs）：在LaSOT、GOT-10K、TNL2K上引入动态更新多模态参考机制，使用DTLLM-VLT生成的语言标注作为输入

推理时top-k=3，LLM使用BLIP。

## 实验关键数据

### 主实验

| 数据集 | 指标 | DUTrack-384 | 之前最佳VL | 最佳纯视觉 | 提升 |
|--------|------|------|----------|------|------|
| LaSOT | AUC | 74.1% | UVLTrack-L 71.3% | ODTrack 73.2% | +0.9% vs 纯视觉 |
| LaSOT | P | 82.9% | UVLTrack-L 78.3% | ODTrack 80.6% | +2.3% vs 纯视觉 |
| LaSOText | AUC | 52.5% | UVLTrack-L 51.2% | AQATrack 52.7% | 持平 |
| TNL2K | AUC | 65.6% | UVLTrack-L 64.8% | ODTrack 60.9% | +4.7% vs 纯视觉 |
| OTB99-Lang | AUC | 71.3% | MMTrack 70.5% | - | +0.8% |
| GOT-10K | AO | 77.8% | - | ODTrack 77.0% | +0.8% |

### 消融实验

| 配置 | LaSOT AUC | LaSOT P | 说明 |
|------|---------|------|------|
| Baseline (无更新) | 71.0% | 75.9% | 静态参考 |
| +DTCM (top-k=3) | 71.7% | 78.1% | 动态模板+1.8%P |
| +DLUM (静态语言) | 72.4% | 80.3% | 语言信息有效 |
| +DLUM (动态,最高频) | 73.0% | 81.6% | 频繁更新最佳 |
| BLIP作为LLM | 73.0% | 81.6% | 简洁生成最佳 |
| BLIP-2 | 73.2% | 81.7% | 略好 |
| DTLLM-Detailed | 72.5% | 80.6% | 详细描述反而引入噪声 |

### 关键发现

- **历史性突破**：DUTrack首次让VL跟踪器在LaSOT上超越最佳纯视觉跟踪器（74.1% vs ODTrack 73.2%），证明动态更新机制能真正释放语言信息的潜力
- DTCM和DLUM效果互补：单独使用DTCM提升+0.7% AUC，加入DLUM后再提升+1.3% AUC
- 语言描述风格：简洁风格优于详细风格，过于详细的描述会引入不必要的噪声
- 注意力可视化显示，静态语言标注存在明显的注意力错位，而动态语言能纠正这种错位

## 亮点与洞察

- **核心洞察精准**：VL跟踪器不如纯视觉跟踪器的根本原因不是缺乏交互设计，而是静态参考不匹配，这是很好的问题发现
- **实际可用**：推理速度43.5fps，仅69.9M参数，实用性强
- **模块设计简洁**：DTCM直接利用已有的注意力图，几乎零额外计算；DLUM的更新策略基于简单的位移/尺度/颜色变化
- **纯视觉benchmark也有效**：在GOT-10K（无语言标注）上仍能生成语言描述并提升性能

## 局限与展望

- 更新频率的阈值需要手动调参，缺乏自适应机制
- LLM生成的语言描述质量受限于BLIP能力
- 未探索更复杂的更新策略（如基于跟踪置信度的自适应更新）
- 在LaSOText上提升不明显，可能是测试集规模小（仅150序列）导致波动

## 相关工作与启发

- 与STARK等动态参考跟踪器的区别：STARK仅更新视觉模板，DUTrack同时更新视觉和语言参考
- 与DTLLM-VLT的关系：DTLLM也用LLM生成语言描述，但DUTrack把这融入完整的动态更新框架中
- 启发：在任何需要长期参考的任务中（如视频目标分割、重识别），都可以考虑动态更新参考信息

## 评分
- 新颖性: ⭐⭐⭐⭐ 动态更新多模态参考是一个清晰有效的思路，VL跟踪首次超越纯视觉
- 实验充分度: ⭐⭐⭐⭐ 6个benchmark + 多维degree消融，非常全面
- 写作质量: ⭐⭐⭐⭐ 动机清晰，模块介绍条理分明
- 价值: ⭐⭐⭐⭐ 为VL跟踪领域指明了"动态参考"的重要方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Realistic Test-Time Adaptation of Vision-Language Models](realistic_test-time_adaptation_of_vision-language_models.md)
- [\[CVPR 2025\] Improving Personalized Search with Regularized Low-Rank Parameter Updates](improving_personalized_search_with_regularized_low-rank_parameter_updates.md)
- [\[CVPR 2025\] Rethinking Few-Shot Adaptation of Vision-Language Models in Two Stages](rethinking_few-shot_adaptation_of_vision-language_models_in_two_stages.md)
- [\[CVPR 2026\] MVLM: Template-Free Tracking via Vision-Language Margin Confidence and Memory-Gated Tracking](../../CVPR2026/multimodal_vlm/mvlm_template-free_tracking_via_vision-language_margin_confidence_and_memory-gat.md)
- [\[CVPR 2026\] Dynamic Logits Adjustment and Exploration for Test-Time Adaptation in Vision Language Models](../../CVPR2026/multimodal_vlm/dynamic_logits_adjustment_and_exploration_for_test-time_adaptation_in_vision_lan.md)

</div>

<!-- RELATED:END -->
