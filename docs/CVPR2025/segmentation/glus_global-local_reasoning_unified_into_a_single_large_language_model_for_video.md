---
title: >-
  [论文解读] GLUS: Global-Local Reasoning Unified into A Single Large Language Model for Video Segmentation
description: >-
  [CVPR 2025][语义分割][视频目标分割] 提出GLUS框架，通过"上下文帧（全局推理）+ 查询帧（局部追踪）"的帧划分策略，将全局理解和局部时序一致性统一到单个MLLM中，结合端到端训练的VOS记忆库模块，在MeViS上大幅超越所有MLLM-based方法（J&F 51.3%）。 1. 领域现状：参考视频目标分割（…
tags:
  - "CVPR 2025"
  - "语义分割"
  - "视频目标分割"
  - "多模态大语言模型"
  - "全局局部推理"
  - "记忆库"
  - "参考视频分割"
---

# GLUS: Global-Local Reasoning Unified into A Single Large Language Model for Video Segmentation

**会议**: CVPR 2025  
**arXiv**: [2504.07962](https://arxiv.org/abs/2504.07962)  
**代码**: [https://glus-video.github.io/](https://glus-video.github.io/) (项目页)  
**领域**: 分割  
**关键词**: 视频目标分割, 多模态大语言模型, 全局局部推理, 记忆库, 参考视频分割

## 一句话总结
提出GLUS框架，通过"上下文帧（全局推理）+ 查询帧（局部追踪）"的帧划分策略，将全局理解和局部时序一致性统一到单个MLLM中，结合端到端训练的VOS记忆库模块，在MeViS上大幅超越所有MLLM-based方法（J&F 51.3%）。

## 研究背景与动机

1. **领域现状**：参考视频目标分割（RefVOS）要求根据语言描述在整个视频中定位并持续追踪目标物体。近期方法（VISA、VideoLISA、ViLLa等）将多模态大语言模型（MLLM）引入RefVOS，希望利用LLM的推理能力处理复杂语言表达。
2. **现有痛点**：MLLM的上下文窗口有限（通常N≈16帧），而视频帧数T远大于N。现有方法面临"Ref"与"VOS"的两难困境——用N帧做全局语义理解（uniform sampling），或用N帧做局部连续追踪（continuous sampling），二者无法兼顾。为弥补另一端的不足，它们依赖外部VOS模型或关键帧选择器。
3. **核心矛盾**：全局推理需要覆盖整个视频的稀疏帧来捕捉描述中的行为/属性，局部推理需要连续帧来保证时序一致性。在有限上下文窗口内，两种需求在帧分配上直接冲突。
4. **本文目标** (1) 如何在单个MLLM中同时实现全局理解和局部追踪？(2) 如何摆脱对外部VOS模型的依赖？(3) 如何在有限上下文窗口内最大化信息利用效率？
5. **切入角度**：MLLM的自回归特性天然兼容VOS的流式推理——当前帧的预测基于前序帧的结果。因此只需将帧分为两组：稀疏的上下文帧提供全局信息（像人先快速浏览视频），连续的查询帧逐帧标注分割结果（像人分帧画mask）。
6. **核心 idea**：将N帧显式分为"全局上下文帧"+"局部查询帧"，并端到端集成预训练VOS记忆库，让单个MLLM兼具全局理解和局部追踪能力。

## 方法详解

### 整体框架
输入是视频T帧+语言表达R，输出是所有T帧的分割mask。GLUS基于LISA-7B框架，将N=8帧分为4帧上下文帧（从视频均匀采样）和4帧查询帧（连续采样的视频片段）。上下文帧放在前面提供全局语义，查询帧放在后面自回归生成$\langle\text{SEG}\rangle$ token。推理时用滑动窗口逐段处理所有T帧，上下文帧在整个视频中保持固定。

### 关键设计

1. **上下文帧+查询帧的全局-局部统一策略**:
    - 功能：在单个MLLM的有限上下文窗口内同时容纳全局和局部信息。
    - 核心思路：将N帧分为$N_C$帧上下文帧（从整个视频均匀采样，覆盖全局语义）和$N_Q$帧查询帧（连续采样，支持局部时序推理）。在LLM中，上下文帧排在前面，查询帧在后面交替放置$\langle\text{SEG}\rangle$ token。第t帧的分割token生成为$\langle\text{SEG}\rangle_t = \text{LLM}([R, I^C_{1:N_c}, I^Q_1, \langle\text{SEG}\rangle_1, ..., I^Q_t])$。训练和推理完全一致——训练时上下文帧从每段均匀随机采样，查询帧随机采短片段；推理时上下文帧取每段中心帧，查询帧用stride=1的滑动窗口遍历。
    - 设计动机：之前的方法训练和推理策略不一致（如VISA训练用random、推理用uniform+continuous），导致分布不匹配。GLUS的训练-推理对齐是一个重要优势。

2. **端到端VOS记忆库集成**:
    - 功能：突破MLLM上下文窗口限制，存储和利用长期历史信息，同时消除对外部VOS模型的依赖。
    - 核心思路：将SAM-2的记忆注意力模块端到端集成到分割解码器中。对第t帧查询帧的解码变为$M_t = \text{Dec}(I^Q_t, \langle\text{SEG}\rangle_t, \text{MemBank})$，梯度可以反传到记忆库中存储的特征和预训练的记忆读取注意力。训练时用$N_Q$帧查询帧模拟VOS的流式推理行为。
    - 设计动机：之前的方法（VISA、VideoLISA）在推理时才调用外部VOS模型进行mask传播，训练和推理不一致。端到端训练使MLLM能与VOS记忆库协同优化，且训练分布与推理分布对齐。

3. **目标对比损失（Object Contrastive Loss）**:
    - 功能：增强MLLM对相似外观物体的细粒度区分能力，减少将错误物体匹配到语言描述的情况。
    - 核心思路：对同一物体的不同语言描述生成的$\langle\text{SEG}\rangle$ token构成正样本对（MeViS中约91.5%的batch可采样到正样本对），不同物体的token构成负样本。使用SimCLR风格的对比损失最大化不同物体的$\langle\text{SEG}\rangle$ token间距离，同时维护一个分割token bank来保证充足的负样本。
    - 设计动机：RefVOS中大量视频包含外观相似的多个物体（如"被攻击的大象"需从多只大象中区分），MLLM容易为它们生成相似的$\langle\text{SEG}\rangle$ token。该损失可插拔使用，仅在MeViS数据上施加但对Ref-Youtube-VOS也有提升。

### 损失函数 / 训练策略
文本监督用标准交叉熵损失（强制生成$\langle\text{SEG}\rangle$ token），mask监督沿用SAM-2的per-pixel BCE + DICE损失，加上可选的目标对比损失。MLLM用LoRA微调，SAM-2解码器可训练，骨干冻结。训练4×A100，约25小时，3000步，每步batch=2×10梯度累积。上下文窗口N=8（4上下文+4查询），每帧4倍降采样后64个视觉token。

## 实验关键数据

### 主实验

| 方法 | MeViS J&F | MeViS J | Ref-YT-VOS J&F |
|------|-----------|---------|-----------------|
| VISA-13B | 44.5 | 41.8 | 63.0 |
| VideoLISA-3.8B | 44.4 | 41.3 | 63.7 |
| ViLLa | - | - | 66.5 |
| DsHmp (non-LLM) | 46.4 | 43.0 | 67.1 |
| **GLUS-S (ours)** | **50.3** | **47.5** | **66.6** |
| **GLUS-A (ours)** | **51.3** | **48.5** | **67.3** |

在最具挑战性的MeViS上，GLUS-A的J&F达51.3%，比之前最好的MLLM方法高出约**7个百分点**，比非LLM SOTA DsHmp高约5个百分点。

### 消融实验

| 配置 | MeViS J&F | 说明 |
|------|-----------|------|
| Global-only (uniform) | 44.3 | 仅全局帧，无局部推理 |
| Local-only (continuous) | 44.4 | 仅局部帧，无全局上下文 |
| **GLUS (context+query)** | **48.2** | 全局+局部统一 |
| + Memory Bank | **49.3** | 加记忆库提升 +1.1 |
| + Contrastive Loss | **50.3** | 加对比损失提升 +1.0 |
| + Key Frame Selector | 50.3→refined | 自精炼框架 |

### 关键发现
- **全局+局部统一是核心贡献**：从44.3/44.4跃升到48.2（+3.8/3.9），证明两种推理缺一不可。
- **记忆库的端到端训练是重要增强**：+1.1 J&F，说明MLLM学会了与VOS记忆模块协同工作，不再需要外部VOS。
- **对比损失在MeViS上尤为有效**：+1.0 J&F，因为MeViS中多物体+运动描述的场景特别需要细粒度区分。对比损失仅在MeViS上施加，但对Ref-Youtube-VOS也有间接提升。
- GLUS使用的SFT数据比VISA/ViLLa少（仅MeViS+Ref-YT-VOS），但效果更好，体现了架构设计的效率。

## 亮点与洞察
- **训练-推理一致性的重要性**：GLUS首次在RefVOS的MLLM方法中实现了训练和推理完全对齐的帧采样策略。这个看似简单的设计原则带来了显著的性能提升（vs VISA训练random/推理uniform+continuous）。
- **VOS记忆库作为MLLM的即插即用增强**：端到端集成SAM-2记忆模块的思路非常优雅——既利用了VOS预训练的时序追踪能力，又通过梯度反传让MLLM学会如何"使用"记忆。这个模式可以迁移到其他需要长序列推理的MLLM任务。
- **"人类模拟"的设计直觉**：先快速浏览（上下文帧）再逐帧标注（查询帧）的流程，与人类的视频标注行为一致，提供了清晰的设计哲学。

## 局限与展望
- 上下文窗口仍限制为N=8帧（4+4），更长的视频中上下文帧的采样密度不足，可能遗漏关键动作。
- 每帧仅64个视觉token（4倍降采样），空间分辨率有限，精细边缘的分割质量受限。
- 计算资源限制导致Additional-SFT版本未使用对比损失和关键帧选择器，全部组件叠加的性能未充分探索。
- 对比损失依赖MeViS中同一物体多个referring的特性，在缺少多描述标注的数据集上无法直接使用。
- 关键帧选择器的伪标签来自GLUS自身预测的IoU，存在自强化偏差的风险。

## 相关工作与启发
- **vs VISA**: VISA训练用random sampling，推理用uniform+continuous+外部VOS。GLUS统一了训练和推理，且无需外部VOS，系统更简洁、效果更好。
- **vs ViLLa**: ViLLa专注局部推理（continuous sampling），配合context-aggregation模块。GLUS通过显式的上下文帧引入全局信息，在MeViS上优势明显。
- **vs VideoLISA**: VideoLISA用单个token追踪整个视频，全局推理强但局部弱。GLUS的多token+记忆库方案在时序一致性上更有优势。
- 端到端集成VOS记忆库的思路可推广到其他视频级MLLM任务（视频问答、视频理解等），值得跟进。

## 评分
- 新颖性: ⭐⭐⭐⭐ 上下文帧+查询帧的设计虽简单但非常有效，VOS记忆库的端到端集成是有意义的贡献
- 实验充分度: ⭐⭐⭐⭐⭐ MeViS+Ref-YT-VOS+ReVOS+ReasonVOS多数据集评估，消融全面
- 写作质量: ⭐⭐⭐⭐ 问题分析透彻，Table 1对比前人方法的帧采样策略一目了然
- 价值: ⭐⭐⭐⭐ 为RefVOS的MLLM方法建立了强基线，在最难的MeViS上大幅提升

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] VISA: Reasoning Video Object Segmentation via Large Language Models](../../ECCV2024/segmentation/visa_reasoning_video_object_segmentation_via_large_language_models.md)
- [\[CVPR 2025\] StoryGPT-V: Large Language Models as Consistent Story Visualizers](storygpt-v_large_language_models_as_consistent_story_visualizers.md)
- [\[CVPR 2025\] Uni4D: Unifying Visual Foundation Models for 4D Modeling from a Single Video](uni4d_unifying_visual_foundation_models_for_4d_modeling_from_a_single_video.md)
- [\[CVPR 2025\] The Devil is in Temporal Token: High Quality Video Reasoning Segmentation](the_devil_is_in_temporal_token_high_quality_video_reasoning_segmentation.md)
- [\[CVPR 2025\] DINOv2 Meets Text: A Unified Framework for Image- and Pixel-Level Vision-Language Alignment](dinov2_meets_text_a_unified_framework_for_image-_and_pixel-level_vision-language.md)

</div>

<!-- RELATED:END -->
