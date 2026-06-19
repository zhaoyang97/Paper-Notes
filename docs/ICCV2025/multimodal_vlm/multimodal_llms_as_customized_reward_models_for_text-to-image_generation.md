---
title: >-
  [论文解读] Multimodal LLMs as Customized Reward Models for Text-to-Image Generation
description: >-
  [ICCV2025][多模态VLM][奖励模型] 提出 LLaVA-Reward，利用预训练 MLLM 的隐藏状态（而非文本生成）直接输出奖励值，通过 Skip-connection Cross Attention (SkipCA) 增强双向视觉-文本交互，配合 LoRA 适配不同评估维度，在文本-图像对齐、保真度和安全性评估上达到 SOTA，并可用于扩散模型推理时缩放。
tags:
  - "ICCV2025"
  - "多模态VLM"
  - "奖励模型"
  - "文本到图像生成"
  - "多模态评估"
  - "偏好学习"
  - "推理时缩放"
---

# Multimodal LLMs as Customized Reward Models for Text-to-Image Generation

**会议**: ICCV2025  
**arXiv**: [2507.21391](https://arxiv.org/abs/2507.21391)  
**代码**: [GitHub](https://github.com/sjz5202/LLaVAReward)  
**领域**: 多模态VLM  
**关键词**: 奖励模型, 文本到图像生成, 多模态评估, 偏好学习, 推理时缩放  

## 一句话总结

提出 LLaVA-Reward，利用预训练 MLLM 的隐藏状态（而非文本生成）直接输出奖励值，通过 Skip-connection Cross Attention (SkipCA) 增强双向视觉-文本交互，配合 LoRA 适配不同评估维度，在文本-图像对齐、保真度和安全性评估上达到 SOTA，并可用于扩散模型推理时缩放。

## 研究背景与动机

文本到图像生成模型（如 Stable Diffusion）的快速发展催生了对高质量自动评估/奖励模型的强烈需求。现有方法存在以下问题：

**CLIP-based 方法**（CLIPScore、PickScore、HPSv2、ImageReward）：CLIP 表现为词袋模型，对复杂文本-图像关系的建模能力有限，泛化性差

**VQA-based MLLM 方法**（VIEScore、EvalAlign、LlavaGuard）：需要冗长的系统提示和指令微调，推理效率低，评分精度受离散化限制

**Token 概率方法**（Q-ALIGN、VQAScore、LLaVAScore）：依赖特定"golden token"（如 "Yes"/"No"）的概率，难以处理偏好数据中质量差异小的样本

核心问题是：**如何构建一个高效、灵活、多维度的 T2I 评估奖励模型？**

作者发现，可以直接利用 MLLM 的**隐藏状态**来预测奖励，无需生成文本回答或使用复杂评估指令，从而同时实现效率和表达能力的提升。

## 方法详解

### 整体架构

LLaVA-Reward 基于 Phi-3.5-vision (4.2B) 构建，将文本-图像对作为输入，利用 MLLM 最终层的隐藏状态预测奖励。架构包含三个关键组件：

1. **预训练 MLLM 骨干**：冻结的 Phi-3.5-vision，提供视觉和语言表示
2. **LoRA 适配器**：针对不同评估维度（对齐、保真度、安全性）使用独立的 LoRA
3. **SkipCA 奖励头**：替代传统线性层的双向交叉注意力模块

### 关键设计：Skip-connection Cross Attention (SkipCA)

在 decoder-only MLLM 中，因果注意力机制使得视觉 token 不受后续注入的文本 token 影响，这会损害图文相关性推理能力。SkipCA 通过建立**早期视觉特征**和**深层隐藏表示**之间的跳跃连接来解决这一问题：

$$r_\theta(\bm{i}, \bm{t}) = f_r(\mathbf{e}_h, \mathbf{e}_v) = g(f_{\text{SCA}}(\mathbf{e}_h, \mathbf{e}_v))$$

具体地，SkipCA 是一个标准的交叉注意力操作：
- **Query**：最终层 EOS token 的隐藏状态 $\mathbf{e}_h$（融合了文本语义）
- **Key/Value**：视觉投影器输出的视觉 token $\mathbf{e}_v$（视觉特异性强）
- **输出**：经线性投影产生标量奖励（BT 模型）或向量奖励（GPM 模型）

设计动机：视觉 token 在 MLLM 深层中的影响力逐渐减弱，因此直接使用投影后的视觉 token（"跳跃"到浅层）比在深层做交叉注意力更有效。

### 训练目标

**成对偏好数据**：使用 Bradley-Terry 排序损失

$$\mathcal{L}_{\text{rank}} = -\mathbb{E}_{(\bm{i}_c, \bm{i}_r, \bm{t}) \sim \mathcal{D}_p}\left[-\log\sigma\left(\frac{s_{\theta_p}(\bm{i}_c, \bm{t}) - s_{\theta_p}(\bm{i}_r, \bm{t})}{T}\right)\right]$$

**非成对二分类数据**（如安全性标签）：使用交叉熵损失

$$\mathcal{L}_{\text{CE}} = -\mathbb{E}\left[\log\sigma(s(\bm{i}_c, \bm{t})) + \log(1-\sigma(s(\bm{i}_r, \bm{t})))\right]$$

**General Preference Model (GPM)**：对于复杂偏好关系，使用多维奖励向量和反对称偏好算子：

$$s(\bm{i}_c, \bm{t}) - s(\bm{i}_r, \bm{t}) = \langle \mathbf{R}^\succ r_\theta(\bm{i}_c, \bm{t}), r_\theta(\bm{i}_r, \bm{t})\rangle$$

其中 $\mathbf{R}^\succ$ 是 skew-symmetric 矩阵。这使得模型能在隐空间中建模更精细的偏好关系。

### LoRA 适配多维度评估

为每个评估维度（对齐、保真度、安全性、总体排序）训练独立的 LoRA 适配器。所有维度共享 MLLM 骨干参数，通过切换 LoRA 适配器即可快速切换评估维度，兼顾效率和灵活性。

### 训练细节

- 冻结视觉编码器和 LLM 内部参数
- 仅训练视觉投影器、SkipCA 奖励头和 LoRA 适配器（约 8% 额外参数）
- 训练数据：ImageReward 对齐集 158K 对、保真度集 84K 对、UnsafeBench 8.1K 二分类

## 实验结果

### MJ-Bench 多维度评估

| 方法 | 参数量 | 对齐 (Acc w/tie) | 安全 (Acc w/tie) | 保真度 (Acc w/tie) |
|------|-------|-----------------|-----------------|-------------------|
| CLIPScore | 428M | 38.1 | 12.7 | 34.4 |
| ImageReward | 478M | 50.9 | 24.9 | 63.5 |
| HPS-v2.1 | 2B | 47.3 | 18.8 | 67.3 |
| VQAScore | 11B | 63.2 | - | - |
| GPT-4o | - | 61.5 | 35.3 | 97.6 |
| LlavaGuard | 7B | - | 5.6 | - |
| **LLaVA-Reward-Phi** | **4.2B** | **66.1** | **55.2** | **91.1** |
| **LLaVA-Reward-Qwen** | **8.2B** | **67.5** | **59.2** | **94.3** |

LLaVA-Reward-Qwen 在所有三个维度上均优于开源方法，在安全性上大幅超越所有方法（59.2% vs 第二名 37.2%），且仅需 4.2B~8.2B 参数即接近 GPT-4o 级别性能。

### SkipCA 消融

| 配置 | 对齐 Acc | 安全 Acc | 保真度 Acc |
|------|---------|---------|-----------|
| w/o SkipCA (MLP) | 68.2 | 39.7 | 87.3 |
| **w/ SkipCA** | 66.1 | **55.2** | **91.1** |

SkipCA 在安全性评估上提升 15.5 个百分点，在保真度上提升 3.8 个百分点，验证了增强双向视觉-文本交互对 T2I 评估的重要性。

### 推理效率对比

| 方法 | 类型 | 推理时间 (s) |
|------|------|------------|
| Evalalign | VQA | 7.01 |
| LlavaGuard | VQA | 4.30 |
| VQAScore | Token | 2.81 |
| LLaVA-score | Token | 0.26 |
| **LLaVA-Reward** | **隐藏状态** | **0.35** |

LLaVA-Reward 推理速度位列第二（0.35s），远快于 VQA-based 方法，接近效率最高的 Token-based 方法。

### 扩散推理时缩放

使用 FK steering 方法将 LLaVA-Reward 应用于 SD v2.1 和 SDXL 的推理时缩放：

| 奖励模型 | GenEval Overall (SDXL) |
|---------|----------------------|
| 无 | 0.563 |
| CLIPScore | 0.592 |
| ImageReward | 0.627 |
| **LLaVA-Reward** | **0.645** |

LLaVA-Reward 在扩散推理时缩放中实现最佳生成质量提升。

## 亮点与洞察

1. **隐藏状态替代文本生成**：跳过 MLLM 的文本解码过程，直接利用隐藏表示预测奖励，同时实现高精度和高效率
2. **SkipCA 的巧妙设计**：通过跨层跳跃连接解决 decoder-only MLLM 中视觉 token 被"遗忘"的问题
3. **GPM 偏好嵌入**：引入多维向量奖励和反对称偏好算子，能建模比标量奖励更复杂的偏好关系
4. **统一多维度评估**：通过 LoRA 适配器实现单一模型覆盖对齐、保真度、安全性等多个维度

## 局限性

- 在 MJ-Bench 对齐维度上，有 SkipCA 的模型反而比无 SkipCA 的略低（66.1 vs 68.2），可能因为 SkipCA 对对齐任务过度强调视觉特征
- 依赖 ImageReward 作为训练数据源，继承了其标注偏差
- 硬负样本构造策略的有效性高度依赖数据集特性
- 仅在 Phi-3.5-vision 和 Qwen2.5-VL 上验证，对更大规模模型的效果未知

## 相关工作

- **CLIP-based 评估**：CLIPScore, PickScore, HPSv2, ImageReward — 受限于 CLIP 的词袋特性
- **VQA-based MLLM 评估**：VIEScore, EvalAlign, LlavaGuard, ImageGuard — 复杂指令+文本生成，效率低
- **Token 概率评估**：Q-ALIGN, VQAScore, LLaVAScore — 依赖特定 token，离散化限制精度
- **安全性评估**：LlavaGuard, ImageGuard, PerspectiveVision — 专注安全维度

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 技术质量 | 4 |
| 实验充分度 | 5 |
| 写作清晰度 | 4 |
| 实用价值 | 5 |
| 总评 | 4.4 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Multimodal RewardBench 2: Evaluating Omni Reward Models for Interleaved Text and Image](../../CVPR2026/multimodal_vlm/multimodal_rewardbench_2_evaluating_omni_reward_models_for_interleaved_text_and_.md)
- [\[ICCV 2025\] Controlling Multimodal LLMs via Reward-guided Decoding](controlling_multimodal_llms_via_rewardguided_decoding.md)
- [\[ACL 2025\] Enhance Multimodal Consistency and Coherence for Text-Image Plan Generation](../../ACL2025/multimodal_vlm/enhance_multimodal_consistency_and_coherence_for_text-image_plan_generation.md)
- [\[CVPR 2025\] CoMM: A Coherent Interleaved Image-Text Dataset for Multimodal Understanding and Generation](../../CVPR2025/multimodal_vlm/comm_a_coherent_interleaved_image-text_dataset_for_multimodal_understanding_and_.md)
- [\[ACL 2025\] Scaling Text-Rich Image Understanding via Code-Guided Synthetic Multimodal Data Generation](../../ACL2025/multimodal_vlm/code_guided_text_rich_image.md)

</div>

<!-- RELATED:END -->
