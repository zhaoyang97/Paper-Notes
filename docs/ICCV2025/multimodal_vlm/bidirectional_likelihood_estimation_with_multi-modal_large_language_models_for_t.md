---
title: >-
  [论文解读] Bidirectional Likelihood Estimation with Multi-Modal Large Language Models for Text-Video Retrieval
description: >-
  [ICCV 2025][多模态VLM][文本-视频检索] 揭示了基于MLLM的检索系统中"候选先验偏差"问题——候选似然估计倾向于选择先验概率高而非语义最相关的候选，提出BLiM（双向似然估计）和CPN（候选先验归一化）模块来解决此问题，在四个文本-视频检索基准上平均R@1提升6.4。 文本-视频检索旨在给定视频（或文本）查…
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "文本-视频检索"
  - "多模态大语言模型"
  - "双向似然估计"
  - "候选先验偏差"
  - "分数校准"
---

# Bidirectional Likelihood Estimation with Multi-Modal Large Language Models for Text-Video Retrieval

**会议**: ICCV 2025  
**arXiv**: [2507.23284](https://arxiv.org/abs/2507.23284)  
**代码**: [github.com/mlvlab/BLiM](https://github.com/mlvlab/BLiM)  
**领域**: 多模态学习 / 视频检索  
**关键词**: 文本-视频检索, 多模态大语言模型, 双向似然估计, 候选先验偏差, 分数校准

## 一句话总结

揭示了基于MLLM的检索系统中"候选先验偏差"问题——候选似然估计倾向于选择先验概率高而非语义最相关的候选，提出BLiM（双向似然估计）和CPN（候选先验归一化）模块来解决此问题，在四个文本-视频检索基准上平均R@1提升6.4。

## 研究背景与动机

文本-视频检索旨在给定视频（或文本）查询后找到最相关的文本（或视频）候选。现有方法的演进：
- **双编码器架构**（CLIP、BERT）：将查询和候选分别编码为单一嵌入，通过相似度检索，计算高效但token级别对齐不足
- **MLLM基检索**：处理拼接的查询-候选对，实现深层token级交互，对长和复杂的查询-候选对效果更好

然而作者发现MLLM检索中存在**候选先验偏差**：通过贝叶斯分解，
$$P(\mathbf{t}|\mathbf{v}) = \frac{P(\mathbf{v}|\mathbf{t}) P(\mathbf{t})}{P(\mathbf{v})}$$
候选似然 $P(\mathbf{t}|\mathbf{v})$ 同时受查询似然 $P(\mathbf{v}|\mathbf{t})$ 和候选先验 $P(\mathbf{t})$ 影响。MLLM自回归特性倾向于给长且重复的文本分配更高概率（高先验），导致检索结果偏向于高频模式而非真正语义匹配的候选。实验证实：某些高先验文本被37%的视频查询检索到（1003个视频中的374个）。

## 方法详解

### 整体框架

BLiM基于预训练视频MLLM（VideoChat-Flash 7B），由UMT视频编码器、线性投影层和Qwen2 LLM组成。推理时采用两阶段检索流程：InternVideo2 1B先检索top-K候选，BLiM再重排序。

### 关键设计

1. **双向似然估计训练**:

    - **视频到文本生成** $P(\mathbf{t}|\mathbf{v})$：标准MLLM预训练范式，给定视频特征自回归生成文本
    $\mathcal{L}_{t|v} = -\sum_{i=1}^{L_t} \log P(t_i | t_{<i}, \mathbf{v})$
    - **文本到视频特征生成** $P(\mathbf{v}|\mathbf{t})$：给定文本，自回归预测下一个视频clip特征，使用对比softmax
    $\mathcal{L}_{v|t} = -\sum_{i=1}^{L_v} \log \frac{\exp(\tilde{v}_{i-1}^\top v_i)}{\sum_{n=1}^{N} \exp(\tilde{v}_{i-1}^\top v_i^{(n)})}$
    - 总训练目标：$\mathcal{L}_{BLiM} = \mathcal{L}_{t|v} + \mathcal{L}_{v|t}$
    - 两个方向的输入模态顺序互换，使用不同的prompt

2. **候选先验归一化（CPN）**:

    - 免训练的分数校准模块，通过对输入模态应用attention mask来估计候选先验概率
    - 在计算候选似然时，使用attention mask遮蔽所有查询token，使模型在无查询条件下生成候选，得到先验估计
    - 用估计的候选先验对候选似然进行归一化，消除先验偏差
    - CPN的通用性：不仅适用于检索，还能增强VQA、captioning等多模态任务中模型对视觉信息的利用

3. **推理流程**:

    - 视频到文本检索：$n^* = \arg\max_n P(\mathbf{t}^{(n)}|\mathbf{v}) + P(\mathbf{v}|\mathbf{t}^{(n)})$
    - 文本到视频检索：$n^* = \arg\max_n P(\mathbf{t}|\mathbf{v}^{(n)}) + P(\mathbf{v}^{(n)}|\mathbf{t})$
    - 双向似然联合考虑，候选似然找最可能生成的候选，查询似然找最可能生成查询的候选

### 损失函数 / 训练策略

- 仅微调线性投影层 + LoRA，参数高效
- 两阶段检索方案：InternVideo2 1B初检（top-K）→ BLiM精排
- 推理复杂度从 $O(N^2)$ 降至 $O(KN)$（如ActivityNet上快307倍）

## 实验关键数据

### 主实验（Text-to-Video R@1）

| 方法 | DiDeMo | ActivityNet | LSMDC | MSRVTT | 平均 |
|------|--------|-------------|-------|--------|------|
| InternVideo2 1B | 57.0 | 60.4 | 32.0 | 51.9 | 50.3 |
| InternVideo2 6B | 57.9 | 63.2 | 33.8 | 55.9 | 52.7 |
| UMT (fine-tuned) | 70.4 | 66.8 | 43.0 | 58.8 | 59.8 |
| InternVideo2 1B* (fine-tuned) | 75.3 | 68.8 | 44.9 | 59.4 | 62.1 |
| **BLiM (Ours)** | **86.4** | **81.0** | **55.7** | **64.7** | **71.9** |

### 消融实验（Video-to-Text R@1, DiDeMo）

| 配置 | T2V R@1 | V2T R@1 | 说明 |
|------|---------|---------|------|
| 仅候选似然 $P(\mathbf{t}|\mathbf{v})$ | - | 较低 | 受候选先验偏差影响 |
| 仅查询似然 $P(\mathbf{v}|\mathbf{t})$ | - | 较高 | 更准确但单向 |
| 双向似然（BLiM-） | 69.8 | 62.9 | 大幅提升 |
| **BLiM + CPN** | **86.4** | **82.8** | 进一步缓解先验偏差 |

DiDeMo上zero-shot BLiM-（无CPN）即达69.8 T2V R@1，已超越全部fine-tuned基线。

### 关键发现

- 候选先验偏差在MLLM检索中普遍存在（视频到文本和文本到视频两个方向均受影响）
- 查询似然单独使用就能得到较准确的检索结果（对角高相似度），但候选似然的高先验候选会扰乱结果
- CPN不仅改善检索性能，在VQA等其他多模态任务中也有提升，证明其作为通用去偏差工具的价值
- 两阶段检索大幅降低计算成本，使MLLM重排切实可行

## 亮点与洞察

- 对候选先验偏差的formalization非常清晰，Proposition 1严格证明了先验gap超过似然gap时排序会反转
- CPN的设计优雅简洁——仅通过attention mask即可无训练地估计先验，然后归一化
- 文本到视频特征生成的训练目标设计新颖：用对比学习代替真正的视频解码，在LLM输出空间中进行

## 局限与展望

- 依赖两阶段检索（InternVideo2初检），端到端效率仍有提升空间
- 文本到视频生成的对比损失需要batch内所有视频作为负样本，可能受batch size影响
- 当前仅在文本-视频检索上验证，文本-图像检索的效果尚待探索

## 相关工作与启发

- 候选先验偏差与VQA中的language prior（语言偏见）问题本质相近，CPN类似于VCD（Visual Contrastive Decoding）的思路
- 双向似然可视为一种互信息（pointwise mutual information）的近似估计
- attention mask估计先验的trick可推广到其他需要去除条件偏差的场景

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 候选先验偏差的发现和双向似然解决方案都很新颖，CPN设计简洁有效
- 实验充分度: ⭐⭐⭐⭐ 四个检索基准+多模态任务扩展分析，结果提升显著
- 写作质量: ⭐⭐⭐⭐⭐ 问题分析深入、理论证明严谨、可视化直观
- 价值: ⭐⭐⭐⭐⭐ R@1提升6.4的幅度非常大，CPN作为通用模块有广泛应用前景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)
- [\[ICCV 2025\] CapeLLM: Support-Free Category-Agnostic Pose Estimation with Multimodal Large Language Models](capellm_support-free_category-agnostic_pose_estimation_with_multimodal_large_lan.md)
- [\[CVPR 2026\] Gravitation-Driven Semantic Alignment for Text Video Retrieval](../../CVPR2026/multimodal_vlm/gravitation-driven_semantic_alignment_for_text_video_retrieval.md)
- [\[ICCV 2025\] MultiVerse: A Multi-Turn Conversation Benchmark for Evaluating Large Vision and Language Models](multiverse_a_multi-turn_conversation_benchmark_for_evaluating_large_vision_and_l.md)
- [\[ICCV 2025\] CompCap: Improving Multimodal Large Language Models with Composite Captions](compcap_improving_multimodal_large_language_models_with_composite_captions.md)

</div>

<!-- RELATED:END -->
