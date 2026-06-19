---
title: >-
  [论文解读] Reason-before-Retrieve: One-Stage Reflective Chain-of-Thoughts for Training-Free Zero-Shot Composed Image Retrieval
description: >-
  [CVPR 2025][LLM推理][组合图像检索] 本文提出OSrCIR，一种免训练的单阶段零样本组合图像检索方法，利用多模态大语言模型直接处理参考图像和修改文本，并通过反思式链式思维推理准确理解用户隐含意图，在多个基准上比现有免训练方法提升1.80%~6.44%。 1. 领域现状：组合图像检索（CIR）通过参考图像+修改…
tags:
  - "CVPR 2025"
  - "LLM推理"
  - "组合图像检索"
  - "零样本"
  - "链式推理"
  - "多模态大模型"
  - "免训练"
---

# Reason-before-Retrieve: One-Stage Reflective Chain-of-Thoughts for Training-Free Zero-Shot Composed Image Retrieval

**会议**: CVPR 2025  
**arXiv**: [2412.11077](https://arxiv.org/abs/2412.11077)  
**代码**: [https://github.com/Pter61/osrcir2024/](https://github.com/Pter61/osrcir2024/)  
**领域**: LLM推理  
**关键词**: 组合图像检索, 零样本, 链式推理, 多模态大模型, 免训练

## 一句话总结
本文提出OSrCIR，一种免训练的单阶段零样本组合图像检索方法，利用多模态大语言模型直接处理参考图像和修改文本，并通过反思式链式思维推理准确理解用户隐含意图，在多个基准上比现有免训练方法提升1.80%~6.44%。

## 研究背景与动机
1. **领域现状**：组合图像检索（CIR）通过参考图像+修改文本来检索目标图像。零样本CIR（ZS-CIR）利用CLIP等预训练模型，无需大量标注三元组数据。
2. **现有痛点**：当前免训练ZS-CIR方法采用两阶段流程——先用图像描述器生成参考图像的文本描述，再用LLM推理目标描述。这导致两个问题：(1)描述阶段不知道修改意图，遗漏关键视觉细节；(2)简单提示限制了LLM推理能力的发挥。
3. **核心矛盾**：两阶段方法中描述和推理的分离导致信息损失——关键视觉细节在描述阶段就丢失了，后续LLM推理无法弥补。
4. **本文目标**：设计单阶段推理方法，保留完整视觉信息并充分发挥MLLM推理能力。
5. **切入角度**：直接让MLLM同时处理参考图像和修改文本，避免中间描述阶段的信息损失。
6. **核心idea**：单阶段MLLM推理 + 反思式链式思维（Reflective CoT）引导精准理解修改意图。

## 方法详解

### 整体框架
参考图像 $I_r$ + 修改文本 $T_m$ → MLLM（含Reflective CoT提示）→ 目标图像描述 $T_t$ → CLIP文本编码 → 与图像数据库余弦相似度匹配 → 检索结果。所有计算在单块NVIDIA A100上完成，使用PyTorch实现。

### 关键设计

1. **单阶段推理过程**:

    - 功能：消除两阶段方法的信息损失，直接从图像和修改文本推理目标描述。
    - 核心思路：$T_t = \Psi_M(p_c \circ I_r \circ T_m)$，将CoT提示、参考图像和修改文本拼接输入MLLM，一次性生成目标描述。MLLM同时"看到"图像和修改意图，能保留与修改相关的关键视觉细节。
    - 设计动机：两阶段方法中描述器不知道修改意图，无法决定保留哪些细节。单阶段让视觉信息完整参与推理。

2. **反思式链式思维（Reflective CoT）**:

    - 功能：引导MLLM逐步推理用户的修改意图，避免简单提示导致的误解。
    - 核心思路：四步渐进推理——(1)原始图像描述：聚焦与修改文本相关的视觉细节；(2)思考（Thoughts）：分析修改意图和影响的视觉元素；(3)反思（Reflections）：过滤错误意图，识别最相关的修改元素，缓解幻觉问题；(4)目标图像描述：基于过滤后的元素生成最终描述。所有步骤在单个提示中完成，保证效率。
    - 设计动机：用户修改意图往往隐含（如"without human"实际是要保留小狗+模糊人影背景），需要多步推理和反思才能准确理解。

3. **视觉语言上下文学习（Vision-by-Language ICL）**:

    - 功能：让MLLM理解每步CoT的预期输出格式，保持零样本设置。
    - 核心思路：提供纯文本形式的示例输出（不包含参考图像），引导MLLM在每一步产生正确格式的推理输出。
    - 设计动机：仅提供CoT指南不够，MLLM需要具体示例才能理解每步预期行为。

### 损失函数 / 训练策略
无需训练，完全免训练方法。检索使用冻结的CLIP模型，通过目标描述 $T_t$ 的文本编码与候选图像编码的余弦相似度进行排序。在CIRCO和CIRR上通过提交服务器在隐藏测试集上评估，在Fashion-IQ和GeneCIS上直接评估。

## 实验关键数据

### 主实验

| 方法 | CIRCO mAP@5 | CIRCO mAP@25 | CIRR R@1 | CIRR R@5 |
|------|------------|-------------|---------|---------|
| CIReVL (ViT-L/14) | 18.57 | 20.89 | 24.55 | 52.31 |
| CIReVL* (GPT-4o) | 18.92 | 21.15 | 24.83 | 52.68 |
| **OSrCIR** | **23.87** | **27.84** | **29.45** | **57.68** |
| LinCIR | 12.59 | 15.00 | 25.04 | 53.25 |

### 消融实验

| 配置 | CIRCO mAP@5 | 说明 |
|------|------------|------|
| Full OSrCIR | 23.87 | 完整反思式CoT |
| w/o Reflections步骤 | 21.42 | 反思步骤贡献+2.45 |
| 简单提示（无CoT） | 19.85 | CoT框架整体贡献+4.02 |
| 两阶段+GPT-4o | 18.92 | 单阶段优于两阶段 |

### 关键发现
- OSrCIR在所有CLIP架构（ViT-B/32, ViT-L/14, ViT-G/14）上均大幅超越现有免训练方法。
- 反思步骤能有效过滤Thoughts阶段的幻觉，提升推理准确性。
- 在CIRCO（更准确的评估协议）上提升尤为显著（+5.30 mAP@5），而在CIRR上提升较小但仍显著。
- 单阶段方法的推理效率与两阶段方法相当，无额外开销。
- 在Fashion-IQ上OSrCIR（ViT-L/14）平均R@10达33.26%，比CIReVL*高+4.21，比最佳训练方法Context-I2W高+5.46，Shirt/Dress/Toptee分别达33.17/29.70/36.92。
- 仅使用更强MLLM（CIReVL→CIReVL*）带来的提升微小（+0.50 mAP@5），说明两阶段范式本身是瓶颈。
- 默认使用GPT-4o（温度=0），结果取三次平均；也支持GPT-4o-mini、GPT-4V和开源LLaVA、MiniGPT4。

## 亮点与洞察
- **先推理再检索的范式**：OSrCIR将CIR重新定义为MLLM推理问题，而非简单的特征组合问题，更贴近人类的检索认知过程。
- **反思步骤的关键作用**：Reflections步骤类似于人类的"等等，让我再想想"的自我纠错过程，有效减少了推理幻觉。反思步骤贡献了+2.45 mAP@5，是性能提升的核心。
- **可迁移到其他多模态检索任务**：Reflective CoT的设计思路可以应用于任何需要理解隐含意图的多模态任务。
- **信息完整性的重要性**：实验证明单阶段方法比两阶段+GPT-4o还要好（+4.95 mAP@5），说明视觉信息的完整保留比使用更强的LLM更重要。
- **Vision-by-Language ICL的设计巧思**：仅用纯文本示例（不含图像）就能引导MLLM理解CoT每步的输出格式，保持了真正的零样本设置。
- **跨CLIP架构一致性**：从ViT-B/32到ViT-G/14，OSrCIR始终大幅领先，说明方法不依赖特定CLIP版本。定性分析显示OSrCIR能准确捕获CIReVL遗漏的关键细节（如海报类型、犬种、颜色等）。

## 局限与展望
- 依赖MLLM的推理能力，不同MLLM的效果可能差异较大。
- 在CIRR上的提升相对有限（+4.9 R@1），可能因为该基准存在噪声标注。
- 未探索与训练式方法的结合，理论上结合训练可进一步提升。
- 检索仍依赖CLIP的文本-图像对齐质量，CLIP的能力上限制约了最终性能。
- CoT的四步推理增加了MLLM的推理开销，在大规模检索场景下需要考虑效率。
- 当修改文本非常简单（如仅改变颜色）时，反思步骤可能是冗余的。
- 在GeneCIS基准（更泛化的组合检索）上也显著超越所有适应方法，展示了良好的跨基准泛化性。

## 相关工作与启发
- **vs CIReVL**: CIReVL两阶段，描述和推理分离导致信息损失；OSrCIR单阶段保留完整视觉信息。
- **vs LinCIR/Pic2Word**: 这些方法需要训练文本反转网络，OSrCIR完全免训练且效果更好（CIRCO mAP@5: OSrCIR 23.87% vs LinCIR 12.59% vs Context-I2W 13.04%）。
- **vs LDRE**: LDRE使用扩散模型集成，计算开销大；OSrCIR更高效。

## 评分

### 实现细节
默认MLLM为GPT-4o，API温度设为0，所有参数保持默认。
检索模块基于PyTorch在单NVIDIA A100上运行。
CLIP变体使用官方权重（ViT-B/32, ViT-L/14），ViT-G/14使用OpenCLIP权重。
- 新颖性: ⭐⭐⭐⭐ 单阶段+反思式CoT在ZS-CIR中首次应用
- 实验充分度: ⭐⭐⭐⭐ 多基准多架构验证（CIRCO/CIRR/Fashion-IQ/GeneCIS），消融详细
- 写作质量: ⭐⭐⭐⭐ 动机说明清晰，图示直观
- 价值: ⭐⭐⭐⭐ 免训练方法达到新SOTA，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] CoT-RVS: Zero-Shot Chain-of-Thought Reasoning Segmentation for Videos](../../ICLR2026/llm_reasoning/cot-rvs_zero-shot_chain-of-thought_reasoning_segmentation_for_videos.md)
- [\[CVPR 2025\] Enhancing Video-LLM Reasoning via Agent-of-Thoughts Distillation](enhancing_video-llm_reasoning_via_agent-of-thoughts_distillation.md)
- [\[ICML 2026\] Beyond Two-Stage Training: Cooperative SFT and RL for LLM Reasoning](../../ICML2026/llm_reasoning/beyond_two-stage_training_cooperative_sft_and_rl_for_llm_reasoning.md)
- [\[NeurIPS 2025\] Visual Thoughts: A Unified Perspective of Understanding Multimodal Chain-of-Thought](../../NeurIPS2025/llm_reasoning/visual_thoughts_a_unified_perspective_of_understanding_multi.md)
- [\[ICML 2026\] ETS: Energy-Guided Test-Time Scaling for Training-Free RL Alignment](../../ICML2026/llm_reasoning/ets_energy-guided_test-time_scaling_for_training-free_rl_alignment.md)

</div>

<!-- RELATED:END -->
