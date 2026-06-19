---
title: >-
  [论文解读] Vamos: Versatile Action Models for Video Understanding
description: >-
  [ECCV 2024][视频理解][文本表征] 提出 Vamos 框架，以大语言模型作为推理器，灵活统一视觉嵌入和通用文本描述作为视频表征，发现纯文本表征在多个视频理解基准上一致性地取得竞争甚至更优性能，并设计 Token Bottleneck Model 实现可解释证据选择与 5 倍推理加速。 领域现状：视频理解的核心问题…
tags:
  - "ECCV 2024"
  - "视频理解"
  - "文本表征"
  - "大语言模型"
  - "视频问答"
  - "动作预测"
  - "Token Bottleneck"
---

# Vamos: Versatile Action Models for Video Understanding

**会议**: ECCV 2024  
**arXiv**: [2311.13627](https://arxiv.org/abs/2311.13627)  
**代码**: [https://brown-palm.github.io/Vamos/](https://brown-palm.github.io/Vamos/)  
**领域**: 视频理解  
**关键词**: 文本表征, 大语言模型, 视频问答, 动作预测, Token Bottleneck

## 一句话总结

提出 Vamos 框架，以大语言模型作为推理器，灵活统一视觉嵌入和通用文本描述作为视频表征，发现纯文本表征在多个视频理解基准上一致性地取得竞争甚至更优性能，并设计 Token Bottleneck Model 实现可解释证据选择与 5 倍推理加速。

## 研究背景与动机

**领域现状**：视频理解的核心问题包括从视频观察中建模人类活动的时间动态、预测未来行为以及回答视频相关问题。端到端视觉语言模型是主流方向。

**现有痛点**：端到端模型的视觉表征不可解释，难以诊断和修复错误预测；视觉嵌入是任务特定的，不同任务需要重新编码；大规模 VLM 联合训练成本高昂。

**核心矛盾**：视频理解任务可能需要不同粒度的互补表征，但现有方法通常只用单一的视觉嵌入。

**本文目标**：探索通用自然语言描述能否作为有效视频表征，以及预训练 LLM 能否作为动作生成模型。

**切入角度**：将感知和推理解耦，文本描述作为中间表征服务于可解释性和效率。

**核心 idea**：通用视频标题作为文本表征加 LLM 作为推理器，构建高效可解释可泛化的视频理解框架。

## 方法详解

### 整体框架

Vamos 是一个统一框架，接受三种视频表征作为输入：离散动作标签（来自动作识别模型）、通用文本描述（来自 BLIP-2 或 LLaVA）、分布式视觉嵌入（来自 CLIP）。核心思路是将这些表征统一到 LLM 的输入空间，通过 LLM 的序列补全能力完成不同的视频理解任务。

输入序列 $\mathbf{x}_t = [\mathbf{x}_{\text{tvr}}, \mathbf{x}_{\text{task}}]$ 包含文本视频表征和任务相关语言输入。冻结的词嵌入层生成文本 token $\mathbf{z}_t = \mathcal{F}_{\text{emb}}(\mathbf{x}_t) \in \mathbb{R}^{L_t \times D}$。对于视觉嵌入，通过可学习线性投影对齐到语言空间：

$$\mathbf{z}_v = \mathcal{F}_{\text{proj}}(\mathcal{E}(v_1, \ldots, v_{N_v})) \in \mathbb{R}^{N_v \times D}$$

采用早期融合策略拼接视觉和文本 token 作为 LLM 输入。

### 关键设计

1. **文本视频表征（Text-based Video Representation）**：视频的文本表征通过两种方式获取。通用标题方面，使用 BLIP-2 或 LLaVA-1.5 对采样帧生成 image-level caption 并拼接成 video-level caption，LLaVA 生成更详细描述（约 100 token 每帧），效果更好。动作标签方面，使用预训练 Transformer 编码器在 Ego4D 预定义动作空间中预测 verb-noun 对。关键洞察是文本表征通用且可复用，提取一次即可服务于不同下游任务。

2. **LLM 作为时序推理器**：利用 LLM 序列补全能力统一多个任务。视频问答（VQA）中任务输入为指令加问题加答案，训练目标为答案，推理时选择最大化序列似然的答案。长期动作预测（LTA）中任务输入为指令加未来动作序列，LLM 自回归生成未来动作。微调使用 LoRA 或 LLaMA-Adapter 进行参数高效微调。

3. **Token Bottleneck Model（TBM）**：受 Concept Bottleneck Model 启发，进行两个关键泛化。第一，从预定义概念到自由文本 token，直接在分词后的文本 token 上操作。第二，从线性分类器到非线性模型加硬注意力，使用二值注意力选择 token 子集输入 LLM。

   实现方式：将输入序列均匀分为 $k$ 段每段 $n$ 个 token，轻量级 token selector（浅层 Transformer 编码器加线性层）为每段选择 1 个 token，训练时使用 Gumbel-Softmax 保持可微：

    $\mathbf{g}^{(i)} = \text{TokenSelector}(z_1^{(i)}, \ldots, z_n^{(i)}) \in \mathbb{R}^n$

   从 $k$ 段中选出 $k$ 个 token 作为浓缩表征（如 $k=40$ 仅保留约 6% 原始 token）。TBM 实现三重价值：可解释性（直接查看被选中 token 作为决策证据）、可干预性（支持测试时人工修正）、高效性（仅新增 0.7M 参数，推理加速 5 倍）。

4. **模态融合与 Modality Dropout**：同时使用文本和视觉输入微调时模型容易过拟合，解决方案是训练时随机丢弃整个视觉嵌入序列，使 vis+text 模型能稳定训练。

### 损失函数 / 训练策略

- **训练目标**：标准语言建模目标（next token prediction）
- **微调方式**：LoRA 或 LLaMA-Adapter 参数高效微调
- **LLM 选择**：LLaMA2-7B（Ego4D）、LLaMA3-8B（NeXT-QA/IntentQA）、GPT-4o（zero-shot）
- **视觉编码器**：冻结 CLIP ViT-L/14
- **标题生成**：BLIP-2 或 LLaVA-1.5，采样 6-12 帧

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文 (Vamos) | 最佳对比方法 | 提升 |
|--------|------|------|------|------|
| NeXT-QA | Acc (All) | 77.3% | LLaMA-VQA-33B 75.5% | +1.8% (8B 超 33B) |
| IntentQA | Acc (All) | 74.16% | CaVIR 57.64% | +28.7% |
| Ego4D LTA | Edit Dist (Action) | 0.868 | AntGPT 0.877 | +0.009 |
| EgoSchema (zero-shot) | Acc (Full) | 53.55% | InternVideo 32.1% | +66.8% |
| Spacewalk-18 (zero-shot) | Acc | 18.6% | 先前最佳 13.6% | +5.0% |

### 不同表征的对比

| 输入类型 | Ego4D-LTA (Action, 越低越好) | NeXT-QA (All) | IntentQA (All) |
|---------|---------------------|-----------------|-----------------|
| 视觉嵌入 | 0.884 | 69.6% | 66.7% |
| 文本表征 | 0.878 | 75.0% | 73.2% |
| 视觉+文本 | 0.868 | 74.5% | 71.7% |

### Token Bottleneck Model 消融

| 输入 | 选择 token 数 / 总数 | NeXT-QA All | 推理时间 |
|------|---------------------|------------|---------|
| 视觉嵌入 | all | 71.0% | - |
| 文本 | all (644 avg) | 77.3% | 1.41s |
| 文本 + TBM | 20 / 644 | 67.4% | - |
| 文本 + TBM | 40 / 644 | 69.6% | 0.29s (5x) |

### 关键发现

- 文本表征在所有基准上一致性地优于或接近视觉嵌入，这是本文最令人惊讶的发现
- 添加视觉特征在 VQA 任务上几乎无额外增益，说明 CLIP 嵌入无法编码 caption 未捕捉的残余信息
- Vamos 直接受益于更强 LLM：LLaMA3 比 LLaMA1 在 NeXT-QA 上提升 2.3%
- Zero-shot 长视频 QA 上的巨大优势来自感知与推理的解耦，推理模块更容易泛化
- Oracle caption 在 EgoSchema 子集上达到 81.8%，表明更好的 captioning 模型能进一步提升上限

## 亮点与洞察

1. **反直觉的发现**：纯文本表征在视频理解任务上与甚至优于视觉嵌入，挑战了视觉特征不可替代的假设
2. **感知-推理解耦的威力**：解耦后推理模块可零样本泛化，且能直接从 LLM 进步中获益
3. **Token Bottleneck Model**：巧妙地将 CBM 推广到自由文本和非线性模型，兼顾可解释性和效率
4. **测试时干预**：文本表征的可解释性天然支持人工修正错误预测，无需重训模型
5. **框架通用性**：一套文本表征可复用于多个任务，不同任务间可共享推理模块

## 局限与展望

1. 文本表征是对视觉输入的有损压缩，细粒度视觉信息（如物体精确位置、微妙运动）可能丢失
2. 框架性能上限受制于 captioning 模型质量
3. 生成 caption 本身需要大型模型，总体计算并非零成本
4. CLIP 嵌入无法有效补充 caption 信息，需探索替代视觉编码器
5. 需要更好的基准来评估需要细粒度结构化视觉理解的场景

## 相关工作与启发

- **Socratic Models**：同样使用自然语言作为模态间公共接口，Vamos 增加了视觉嵌入融合和 TBM
- **Concept Bottleneck Model**：Vamos 将 CBM 推广为 TBM，从预定义概念到自由文本 token
- **VidIL**：利用专家知识设计概念进行 few-shot 视频理解，Vamos 使用通用 caption 更灵活
- **启发**：在 LLM 能力持续提升的背景下，文本化感知加 LLM 推理的范式是高效可扩展的方向

## 评分

- 新颖性: ⭐⭐⭐⭐ 文本表征替代视觉嵌入的探索有启发性，TBM 设计优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 5 个互补基准、系统表征对比、多 LLM 消融、TBM 分析极其全面
- 写作质量: ⭐⭐⭐⭐⭐ 问题动机清晰，实验设计严谨，可视化分析直观
- 价值: ⭐⭐⭐⭐⭐ 揭示了文本即表征的强大潜力，对视频理解范式有重要启示

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Optimizing Factorized Encoder Models: Time and Memory Reduction for Scalable and Efficient Action Recognition](optimizing_factorized_encoder_models_time_and_memory_reduction_for_scalable_and_.md)
- [\[ECCV 2024\] Referring Atomic Video Action Recognition](referring_atomic_video_action_recognition.md)
- [\[ECCV 2024\] Leveraging Temporal Contextualization for Video Action Recognition](leveraging_temporal_contextualization_for_video_action_recognition.md)
- [\[CVPR 2026\] InternVideo-Next: Towards World-Understanding Video Models](../../CVPR2026/video_understanding/internvideo-next_towards_world-understanding_video_models.md)
- [\[ECCV 2024\] Adapt2Reward: Adapting Video-Language Models to Generalizable Robotic Rewards via Failure Prompts](adapt2reward_adapting_videolanguage_models_to_generalizable.md)

</div>

<!-- RELATED:END -->
