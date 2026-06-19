---
title: >-
  [论文解读] OmniPT: Unleashing the Potential of Large Vision Language Models for Pedestrian Tracking and Understanding
description: >-
  [AAAI 2026][多模态VLM][行人跟踪] 本文提出OmniPT，一个基于大视觉语言模型（LVLM）的统一行人跟踪框架，通过RL-Mid Training-SFT-RL四阶段训练策略，同时支持传统MOT、基于语言引用的跟踪（RMOT/CRMOT）和语义理解（SMOT），在多个基准上取得SOTA结果，尤其在BenSMOT上HOTA达75.04，较前SOTA提升3.06。
tags:
  - "AAAI 2026"
  - "多模态VLM"
  - "行人跟踪"
  - "大视觉语言模型"
  - "多目标跟踪"
  - "语义理解"
  - "强化学习"
---

# OmniPT: Unleashing the Potential of Large Vision Language Models for Pedestrian Tracking and Understanding

**会议**: AAAI 2026  
**arXiv**: [2511.17053](https://arxiv.org/abs/2511.17053)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 行人跟踪, 大视觉语言模型, 多目标跟踪, 语义理解, 强化学习

## 一句话总结
本文提出OmniPT，一个基于大视觉语言模型（LVLM）的统一行人跟踪框架，通过RL-Mid Training-SFT-RL四阶段训练策略，同时支持传统MOT、基于语言引用的跟踪（RMOT/CRMOT）和语义理解（SMOT），在多个基准上取得SOTA结果，尤其在BenSMOT上HOTA达75.04，较前SOTA提升3.06。

## 研究背景与动机

行人跟踪（Pedestrian Tracking）是计算机视觉中的经典任务，在自动驾驶、智能监控、运动分析等领域有广泛应用。当前该领域面临两个核心挑战：

**挑战一：复杂场景下的跟踪稳定性**。当行人频繁被遮挡、模糊甚至消失时，如何保持稳定跟踪仍然是一个难题。传统的跟踪器主要依赖外观特征和运动预测，但在外观高度相似（如DanceTrack数据集中的舞者）或长期遮挡的场景下表现不佳。

**挑战二：多模态跟踪新任务的涌现**。近年来MOT领域涌现了许多新的多模态任务：基于语言描述跟踪特定目标的RMOT（Referring MOT）、跨视角的CRMOT（Cross-view RMOT）、以及需要对跟踪目标进行语义理解的SMOT（Semantic MOT）。这些任务强调模型需要在跟踪的同时理解目标的高级语义信息。

**关键洞察**：人类可以轻松跟踪目标，即使目标长时间消失后也能重新识别，这是因为人类会将目标抽象为语义描述进行"潜意识检索"。LVLM在图像级理解任务（如VQA、Caption）上表现出色，但在实例级任务（如目标检测、视觉定位）上与专家模型仍有差距。

**本文切入点**：将LVLM的强大语义理解能力引入行人跟踪任务，用一个统一的"one-for-all"框架同时解决MOT、RMOT、CRMOT和SMOT四类任务。核心idea是将跟踪任务分解为LVLM可以执行的自然语言任务，并通过四阶段训练策略使模型输出格式化的答案。

## 方法详解

### 整体框架

OmniPT基于预训练的LVLM（Qwen2.5-VL），采用**RL → Mid Training → SFT → RL**的四阶段训练策略。整体pipeline将跟踪任务解耦为VQA形式：给定视频帧序列，通过对话式交互获取跟踪结果和语义理解。推理时采用迭代多轮对话实现长期跟踪。

### 关键设计

1. **第一阶段RL：输出格式标准化**

    - **功能**：使用GRPO算法进行轻量级强化学习训练，规范模型输出边界框格式
    - **核心思路**：模型需要输出标准化坐标，严格遵循`<bbox>x,y,w,h</bbox>`格式。设计了分层奖励函数：
        - 格式完全正确：R = 2
        - 包含其他格式（如x,y,x,y）：R = 0.6
        - 不包含bbox标签：R = 0.4
    - 最终奖励 $R_{s1} = R(bbox_p) \times (\frac{IOU(bbox_p, bbox_{gt})}{2} + 0.5)$
    - **设计动机**：LVLM原生输出不可控，需要先统一格式才能进行后续监督训练。第一阶段将IOU映射到0.5-1范围，使模型更关注格式而非精度

2. **第二阶段Mid Training：行人感知增强**

    - **功能**：使用大量行人相关数据和代理任务，增强模型对行人的感知能力
    - **核心模块**：
        - **CLIP对齐训练**：使用SYNTH-PEDES数据集训练视觉编码器，插入`<CLS>` token计算相似度矩阵，用交叉熵损失监督
        - **目标检测代理任务**：随机采样图像，提取GT边界框作为监督
        - **位置预测代理任务**：给定第一帧目标坐标，预测后续帧位置。包含目标消失的特殊样本，增强鲁棒性。主要使用DanceTrack因其运动模式不可预测
        - **行人重识别代理任务**：从不同帧采样同一人和负样本，引导模型识别同一人
    - **设计动机**：Mid Training位于预训练（通用知识）和后训练（特定任务）之间，通过代理任务同步增强目标检测、位置预测和ReID三项核心跟踪能力

3. **第三阶段SFT：任务特定微调**

    - **功能**：在多个行人跟踪数据集上进行监督微调
    - **训练样本分为四类**：
        - **MOT**：连续帧序列 + 三个关键查询（首帧目标位置/序列跟踪/中途新增目标）
        - **RMOT**：用语言描述替代首帧目标位置，跟踪匹配描述的所有目标
        - **CRMOT**：扩展到多视角，将原始序列均匀分配给各视角，用自然语言指定视角
        - **视频/实例描述**：在跟踪完成后进行语义理解，提供整幅图像和目标位置
    - **设计动机**：将跟踪任务解耦为VQA格式，使LVLM能自然地执行不同类型的跟踪任务

4. **第四阶段RL：性能增强**

    - **功能**：额外的RL训练进一步提升跟踪性能和指令遵循能力
    - **与第一阶段的区别**：取消IOU到0.5-1的映射，让模型更关注定位精度

### 推理策略

推理时采用迭代多轮对话实现长期跟踪：前一轮对话最后一帧的跟踪结果作为新一轮的初始先验信息，实现跨对话的连续跟踪。

## 实验关键数据

### 主实验

| 数据集 | 指标 | OmniPT | 之前SOTA | 提升 |
|--------|------|--------|----------|------|
| BenSMOT (跟踪) | HOTA↑ | **75.04** | 71.98 (SMOTer) | +3.06 |
| BenSMOT (视频描述) | CIDEr↑ | **1.826** | 0.343 (SMOTer) | +432% |
| BenSMOT (实例描述) | CIDEr↑ | **0.482** | 0.087 (SMOTer) | +454% |
| DanceTrack | HOTA↑ | **56.4** | 55.1 (OC-SORT) | +1.3 |
| Refer-KITTI-V2 | HOTA↑ | **36.15** | 35.04 (TempRMOT) | +1.11 |
| CRTrack (In-domain) | CVRIDF1↑ | **62.13** | 54.88 (CRTracker) | +7.25 |
| CRTrack (Cross-domain) | CVRIDF1↑ | **46.54** | 12.52 (CRTracker) | +34.02 |

### 消融实验

| 配置 | MOT (HOTA) | RMOT (HOTA) | SMOT (CIDEr) | 说明 |
|------|-----------|-------------|-------------|------|
| SFT only | 47.38 | 30.37 | 0.40 | 基线 |
| MT + SFT | 51.89 | 33.26 | 0.44 | Mid Training显著提升跟踪 |
| SFT + RL | 48.63 | 35.46 | 0.41 | RL在训练不充分时提升有限 |
| MT + SFT + RL | **56.40** | **36.15** | **0.48** | 完整方案效果最佳 |

**不同LVLM对比**（7B规模）：Qwen2.5-VL > LLaVA-NeXT > InternVL2.5，作者认为动态分辨率设计帮助模型更清晰地捕获目标语义特征。

**图像数量影响**：每个训练样本的图像数从2增加到32，描述任务持续提升（CIDEr从0.773→1.826），但跟踪任务先升后降（HOTA在8帧时最优73.04），因为图像过多挑战模型同时处理多图的能力。

### 关键发现

- 跨域CRMOT上提升最为巨大（+34.02 CVRIDF1），说明统一框架在域迁移上具有明显优势
- 语义理解（CIDEr）提升达5倍以上，充分发挥了LVLM在描述任务上的天然优势
- Mid Training阶段对跟踪性能贡献最大，代理任务设计有效

## 亮点与洞察

1. **统一框架新范式**：首次用一个LVLM统一解决MOT/RMOT/CRMOT/SMOT四类任务，体现了"one-for-all"的研究趋势
2. **四阶段训练策略设计精巧**：RL标准化格式→Mid Training增强行人感知→SFT任务对齐→RL增强性能，每个阶段目标明确
3. **Mid Training代理任务设计巧妙**：将检测、位置预测、ReID三项跟踪核心能力拆解为独立的代理任务，特别是位置预测任务中包含目标消失样本的设计
4. **VQA解耦跟踪任务**：将跟踪任务自然地转化为多轮对话，这一思路简洁有效

## 局限与展望

1. **密集场景受限**：在MOT20等密集场景（200+人同时出现）中，LVLM难以准确定位/跟踪所有目标，受限于最大输出长度
2. **类别局限**：当前仅聚焦"行人"类别，尚未扩展到开放词汇目标跟踪
3. **计算资源需求大**：训练使用24块A100 GPU，推理需要多轮对话，效率可能成为瓶颈
4. **实时性未讨论**：论文没有报告推理速度，对实际部署场景很重要

## 相关工作与启发

- LVLM在实例级任务上的能力边界正在被不断拓展，从视觉定位到目标跟踪
- Mid Training（中间训练阶段）的概念来自LLM领域（如OctoThinker、OLMo），在视觉任务中同样有效
- GRPO用于格式标准化是一个实用技巧，解决了LVLM输出不可控的普遍问题
- 启发：是否可以借鉴类似思路，用LVLM统一解决其他视觉任务（如分割+描述+跟踪）

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] VisMem: Latent Vision Memory Unlocks Potential of Vision-Language Models](../../CVPR2026/multimodal_vlm/vismem_latent_vision_memory_unlocks_potential_of_vision-language_models.md)
- [\[CVPR 2026\] MVLM: Template-Free Tracking via Vision-Language Margin Confidence and Memory-Gated Tracking](../../CVPR2026/multimodal_vlm/mvlm_template-free_tracking_via_vision-language_margin_confidence_and_memory-gat.md)
- [\[CVPR 2026\] Unleashing the Intrinsic Visual Representation Capability of Multimodal Large Language Models](../../CVPR2026/multimodal_vlm/unleashing_the_intrinsic_visual_representation_capability_of_multimodal_large_la.md)
- [\[CVPR 2026\] Understanding Counting Mechanisms in Large Language and Vision-Language Models](../../CVPR2026/multimodal_vlm/understanding_counting_mechanisms_in_large_language_and_vision-language_models.md)
- [\[CVPR 2026\] Language-guided Frequency Modulation for Large Vision-Language Models](../../CVPR2026/multimodal_vlm/language-guided_frequency_modulation_for_large_vision-language_models.md)

</div>

<!-- RELATED:END -->
