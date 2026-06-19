---
title: >-
  [论文解读] SAMWise: Infusing Wisdom in SAM2 for Text-Driven Video Segmentation
description: >-
  [CVPR 2025][语义分割][参考视频目标分割] SAMWISE 通过设计跨模态时序适配器（CMT）和条件记忆编码器（CME），在不微调 SAM2 权重的前提下为其注入自然语言理解和显式时序建模能力，以流式处理方式在参考视频目标分割（RVOS）任务上取得了 SOTA 性能，仅增加不到 5M 参数。
tags:
  - "CVPR 2025"
  - "语义分割"
  - "参考视频目标分割"
  - "SAM2"
  - "跨模态时序适配器"
  - "追踪偏差"
  - "流式处理"
---

# SAMWise: Infusing Wisdom in SAM2 for Text-Driven Video Segmentation

**会议**: CVPR 2025  
**arXiv**: [2411.17646](https://arxiv.org/abs/2411.17646)  
**代码**: [https://github.com/ClaudiaCuttano/SAMWISE](https://github.com/ClaudiaCuttano/SAMWISE)  
**领域**: 视频分割 / 多模态VLM  
**关键词**: 参考视频目标分割, SAM2, 跨模态时序适配器, 追踪偏差, 流式处理

## 一句话总结

SAMWISE 通过设计跨模态时序适配器（CMT）和条件记忆编码器（CME），在不微调 SAM2 权重的前提下为其注入自然语言理解和显式时序建模能力，以流式处理方式在参考视频目标分割（RVOS）任务上取得了 SOTA 性能，仅增加不到 5M 参数。

## 研究背景与动机

**领域现状**：参考视频目标分割（RVOS）根据自然语言表达在视频中分割目标对象。现有方法主要有两种范式：一是将视频分成短片段独立处理（如 ReferFormer、MTTR），但这会丢失全局时序上下文；二是离线处理整段视频（如 DsHmp），先建模所有实例的轨迹再选择最匹配的，但无法适用于流式场景。

**现有痛点**：短片段方法在需要长期运动推理的场景（如 MeViS 数据集）中表现很差，因为动作可能跨越多帧；离线方法虽然效果好，但要求一次性访问整个视频，在实时流式场景中不可用。OnlineRefer 尝试在线传播上下文，但仅依赖单帧过去信息，无法捕获长期依赖。

**核心矛盾**：如何在流式处理（不需要整个视频）的同时保留全局上下文信息，这是 RVOS 中的根本矛盾。SAM2 天然适合流式处理且拥有记忆库机制，但它缺乏三个关键能力：(i) 文本理解——只能接受空间点等提示；(ii) 时序建模——逐帧独立提取特征，缺乏运动推理；(iii) 追踪偏差——一旦开始追踪错误目标就会坚持错下去。

**本文目标**：在不微调 SAM2 权重、不依赖外部大模型的前提下，赋予 SAM2 自然语言理解、时序建模和自动纠错追踪的能力。

**切入角度**：利用轻量级适配器模块注入到冻结的 SAM2 中，既保留其强大的分割追踪能力，又增加新功能。这种思路借鉴了 CLIP 等预训练模型的适配器微调范式。

**核心 idea**：设计跨模态时序适配器（CMT）在特征提取阶段同时建模视觉-语言交互和时序演化，并通过条件记忆编码器（CME）检测追踪偏差并软性切换追踪焦点。

## 方法详解

### 整体框架

SAMWISE 建立在冻结的 SAM2 和冻结的文本编码器之上。输入为视频帧序列和文本描述。在每一层特征提取中，CMT 适配器同时嵌入视觉编码器和文本编码器，实现跨模态融合和时序建模。提取的文本特征中的 [CLS] 和动词嵌入分别作为 Contextual Prompt 和 Motion Prompt，经 MLP 投影后送入 SAM2 的 Mask Decoder 生成分割掩码。最终，CME 模块检测当前帧是否出现更匹配文本的新目标，动态调整记忆库中的追踪信息。

### 关键设计

1. **跨模态时序适配器（CMT Adapter）**:

    - 功能：在特征提取阶段同时注入时序信息和跨模态线索
    - 核心思路：CMT 由三个子模块组成——层次化选择注意力（HSA）用于时序建模，视觉到文本注意力（VTA）和文本到视觉注意力（TVA）用于跨模态融合。对于时序部分，将特征体积分解为 $T \times P \times P$ 的时空块，在块内做自注意力，避免全局注意力的高计算开销。块大小 P 随特征分辨率层级递进缩放，实现多尺度时序建模。跨模态部分中，VTA 让视觉特征关注文本表达以识别与描述匹配的候选区域，TVA 让文本 token 吸收视觉信息以根据画面内容调整语义理解
    - 设计动机：SAM2 逐帧独立提取特征，缺乏时序推理。视频中物体运动通常在局部区域，HSA 利用这一先验仅在时空邻域内做注意力，比全 token 自注意力高效得多。跨模态交互使特征在早期就对齐，而非仅在最后阶段融合

2. **双重提示策略（Contextual + Motion Prompt）**:

    - 功能：为 SAM2 Mask Decoder 提供语义和动作双重引导
    - 核心思路：从适配后的文本特征中提取 [CLS] 嵌入作为 Contextual Prompt（编码主体语义），提取动词嵌入作为 Motion Prompt（编码动作线索）。两者拼接后经三层 MLP 投影为 SAM2 的提示向量 $\rho = W_{\text{prompt}}(\text{CAT}[\mathcal{E}_C, \mathcal{E}_M])$。在每一帧都注入该提示，让模型在追踪的同时也关注当前帧内容
    - 设计动机：MeViS 等数据集中文本描述包含动作信息（如"正在攀爬的猫"），仅用全局语义不够，需要显式编码运动相关的动词线索

3. **条件记忆编码器（CME）**:

    - 功能：检测追踪偏差并动态切换追踪焦点
    - 核心思路：从无记忆特征（不受历史预测偏差影响）中通过交叉注意力提取一个 memory-less token $\tau_l$，与 Mask Decoder 输出的 mask token $\tau_m$ 比较。将两者和一个可学习 [DEC] token 拼接做自注意力，再通过线性分类器判断是否检测到新的更匹配目标（$p_{detect} > 0.5$）。若检测到，则计算无偏见 mask $\mathcal{P}_l$ 并与追踪 mask $\mathcal{P}_m$ 在空间上软融合后送入记忆编码器，让 SAM2 "看到"新目标
    - 设计动机：SAM2 存在追踪偏差——当正确目标尚未出现时可能追踪错误目标，且后续即使正确目标出现也不会自动切换。CME 利用无偏特征与文本提示的对齐来检测切换时机，采用软分配而非硬切换以避免误报

### 损失函数 / 训练策略

- 仅训练适配器和 CME 模块（约 4.2-4.9M 参数），SAM2 和文本编码器完全冻结
- 先在 RefCOCO/+/g 上预训练 6 epochs（学习率 1e-4），再在 Ref-Youtube-VOS 上微调 4 epochs（学习率 1e-5），使用 Adam 优化器
- 在 MeViS 上仅微调 1 epoch，clip 长度 T=8
- CME 通过自监督方式用交叉熵损失训练，提供无记忆特征高亮不同目标的样本

## 实验关键数据

### 主实验

| 方法 | MeViS J&F | Ref-YT-VOS J&F | Ref-DAVIS J&F | 总参数 |
|------|-----------|-----------------|---------------|--------|
| DsHmp (前SOTA) | 46.4 | 67.1 | 64.9 | 339M |
| OnlineRefer | 32.3 | 63.5 | 64.8 | 232M |
| VISA (VLM-7B) | 43.5 | 61.5 | 69.4 | 7B |
| **SAMWISE (RoBERTa)** | **49.5** | **69.2** | **70.6** | 202M |
| SAMWISE (CLIP) | 48.3 | 67.2 | 68.5 | 150M |

### 消融实验

| 配置 | MeViS J&F |
|------|-----------|
| MLP-only (无 CMT) | 45.2 |
| + Text-to-Visual | 47.5 |
| + Visual-to-Text | 48.3 |
| + HSA 时序建模 | 50.3 |
| + 全部 CMT | 54.2 |
| + CME | 55.5 |

| HSA 块大小 | 固定P=1 | 固定P=4 | 固定P=8 | 层次 8/4/2 | 层次 16/8/4 |
|-----------|--------|--------|--------|-----------|------------|
| J&F | 49.7 | 52.3 | 53.1 | **54.2** | 53.8 |

### 关键发现

- SAMWISE 在流式处理下超越需要整段视频的离线方法 DsHmp（+3.1% on MeViS），证明流式+记忆库比全局离线更有效
- 仅训练 4.9M 参数（占总量 2.4%）就实现 SOTA，参数效率极高
- CMT 适配器中跨模态交互贡献+5.1%，时序建模贡献+3.9%，CME 贡献+1.3%
- HSA 层次化块大小（8/4/2）比固定大小更好（+1.1%），验证多尺度时序建模的价值
- CME 自适应检测优于固定频率检测：Always 检测反而降 3.5%（50.7 vs 54.2），过频切换引入噪声

## 亮点与洞察

1. "追踪偏差"现象的发现是重要贡献——SAM2 一旦追踪错误目标就不会自我纠正，这在 RVOS 中尤其致命
2. 将适配器范式从图像 CLIP 迁移到视频 SAM2 的思路巧妙，在保留原始能力的同时增加新功能
3. HSA 利用视频运动局部性先验，在时空邻域内做注意力，计算效率远优于全 token 自注意力
4. 150M 参数的 SAMWISE-CLIP 版本超越 7B 参数的 VISA，说明任务特化的轻量适配可能比通用大模型更有效

## 局限与展望

- 流式处理仍需要一定视频片段（T=8 帧），对极端实时场景可能有延迟
- CME 的检测阈值固定为 0.5，不同场景可能需要不同阈值
- 文本编码器冻结可能限制对复杂语义的理解深度
- 未来可探索更长时序窗口和对 SAM2 更大版本（Hiera-L）的适配

## 相关工作与启发

- 与 OnlineRefer 的单帧传播相比，SAMWISE 利用 SAM2 记忆库实现了真正的长程上下文传播
- CMT 的跨模态融合思路可迁移到其他需要文本理解的视觉基础模型（如 DINO、DINOv2）
- 与大 VLM 方法对比表明，任务特化的轻量设计在特定场景中优于通用大模型

## 评分

- **新颖性**: 8/10 — 追踪偏差的发现及 CME 解决方案很有创意，CMT 融合多项创新
- **实验充分度**: 8/10 — 三个主流数据集+详细消融，缺少速度对比
- **写作质量**: 8/10 — 结构清晰，追踪偏差的可视化展示直观
- **价值**: 8/10 — 为 SAM2 扩展到 RVOS 提供了高效强大的方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Style-Editor: Text-driven Object-Centric Style Editing](style-editor_text-driven_object-centric_style_editing.md)
- [\[ICCV 2025\] Correspondence as Video: Test-Time Adaption on SAM2 for Reference Segmentation in the Wild](../../ICCV2025/segmentation/correspondence_as_video_test-time_adaption_on_sam2_for_reference_segmentation_in.md)
- [\[CVPR 2025\] A Distractor-Aware Memory for Visual Object Tracking with SAM2](a_distractor-aware_memory_for_visual_object_tracking_with_sam2.md)
- [\[CVPR 2025\] SAM2-LOVE: Segment Anything Model 2 in Language-Aided Audio-Visual Scenes](sam2-love_segment_anything_model_2_in_language-aided_audio-visual_scenes.md)
- [\[CVPR 2025\] The Devil is in Temporal Token: High Quality Video Reasoning Segmentation](the_devil_is_in_temporal_token_high_quality_video_reasoning_segmentation.md)

</div>

<!-- RELATED:END -->
