---
title: >-
  [论文解读] CAT: Enhancing Multimodal Large Language Model to Answer Questions in Dynamic Audio-Visual Scenarios
description: >-
  [ECCV 2024][多模态VLM][多模态大语言模型] 提出 CAT 模型，通过设计问题相关线索聚合器（Clue Aggregator）捕获细粒度音视频特征，结合混合多模态训练策略和 AI 辅助的模糊感知直接偏好优化（ADPO）策略，显著提升 MLLM 在动态音视频场景中的问答准确性，在多个 AVQA 基准上达到 SOTA。
tags:
  - "ECCV 2024"
  - "多模态VLM"
  - "多模态大语言模型"
  - "音视频问答"
  - "线索聚合器"
  - "偏好优化"
  - "模糊消除"
---

# CAT: Enhancing Multimodal Large Language Model to Answer Questions in Dynamic Audio-Visual Scenarios

**会议**: ECCV 2024  
**arXiv**: [2403.04640](https://arxiv.org/abs/2403.04640)  
**代码**: [GitHub](https://github.com/rikeilong/Bay-CAT)  
**领域**: 多模态VLM  
**关键词**: 多模态大语言模型, 音视频问答, 线索聚合器, 偏好优化, 模糊消除

## 一句话总结

提出 CAT 模型，通过设计问题相关线索聚合器（Clue Aggregator）捕获细粒度音视频特征，结合混合多模态训练策略和 AI 辅助的模糊感知直接偏好优化（ADPO）策略，显著提升 MLLM 在动态音视频场景中的问答准确性，在多个 AVQA 基准上达到 SOTA。

## 研究背景与动机

现实世界是由声音和视觉信息组合构成的，多模态大语言模型（MLLMs）虽然已能响应音视频内容，但存在两个核心问题：

**音视频与问题的关联不足**：现有 MLLMs（如 Video-LLaMA、ChatBridge）主要使用多分支独立处理各模态，然后将模态嵌入与提示拼接输入 LLM。这种范式无法让文本信息在底层与音视频交互，导致网络无法聚焦于与问题相关的细节。

**多模态对齐困难导致模糊描述**：多模态-文本语料的对齐非常困难，使模型生成的回答常常模糊不清——要么使用含糊的词汇描述音视频内容，要么生成过多无用文本。例如，当问"视频中演奏的是什么乐器"时，Video-LLaMA和ChatBridge都无法正确识别"bagpipes"。

核心矛盾：粗粒度的模态桥接方法无法满足AVQA任务中对特定音视频对象的精准定位需求。本文从三个维度入手：问题引导的特征聚合、混合音视频训练、以及偏好优化消除模糊描述。

## 方法详解

### 整体框架

CAT 由三个信息流组成，输入到冻结的 LLaMA2-7B：
- **视觉分支**：ImageBind编码视频 → 线性投影 → 视觉token $x^{vid}$
- **音频分支**：ImageBind编码音频 → 线性投影 → 音频token $x^{aud}$
- **线索分支**：线索聚合器利用问题文本交互音视频特征 → Q-Former压缩 → 线索token $x^{cue}$

三种token与文本拼接后送入带LoRA的LLM生成回答。

### 关键设计

1. **线索聚合器（Clue Aggregator, CA）**:

    - 功能：从音视频特征中动态提取与当前问题相关的细粒度线索
    - 核心思路：分两步工作——
        - **Step 1 感知**：使用 Perceiver（一个微型Transformer网络），包含正序块 $\mathcal{B}_1$ 和逆序块 $\mathcal{B}_2$。$\mathcal{B}_1$ 以问题文本 $h_t$ 为查询，通过交叉注意力定位问题相关的音视频区域；$\mathcal{B}_2$ 在注意力层面巩固原始音视频表示，恢复原始模态长度。具体地：
       - $\mathcal{B}_1(h_t; X) = \text{XA}(h_t, \text{FFN}(\text{SA}(X)))$
       - $\mathcal{B}_2(X; \mathcal{B}_1) = \text{SA}(\text{FFN}(\text{XA}(\text{FFN}(\text{SA}(X)), \mathcal{B}_1)))$
        - **Step 2 聚合**：使用类似 Q-Former 的架构，设置 K=48 个可学习查询向量，从问题感知的音视频特征中提取固定长度的线索token
    - 设计动机：视觉和音频的 Perceiver 共享参数，以学习跨模态潜在关联；先用问题做交叉注意力定位再恢复原始表示，避免信息丢失

2. **混合多模态训练策略**:

    - 功能：分两阶段训练 CAT 使其能同时理解视觉和音频
    - 核心思路：
        - **Stage-I 特征对齐**：先用 WebVid 2.5M 视频-文本数据训练视觉投影器（冻结音频部分），再用 WavCap 音频-文本数据训练音频投影器（冻结视觉部分），全程冻结 LLM 和 ImageBind
        - **Stage-II 音视频联合指令微调**：冻结视觉/音频投影器，仅微调线索聚合器和 LoRA 参数，使用 100k 视频指令 + 自建 AVinstruct 数据集
    - 设计动机：分阶段训练避免多模态冲突；AVinstruct 收集了YouTube和VGGSound的真实视频，用GPT-4从人写示例中合成音视频联合QA对

3. **AI辅助模糊感知直接偏好优化（ADPO）**:

    - 功能：重训模型使其偏好精确描述、拒绝模糊描述
    - 核心思路：分两步——
        - **收集模糊样本并重写**：用已训练的 CAT 在训练集上生成回答，让 GPT 审查每个回答是否模糊（与标准答案偏差大），将模糊回答标记为负例 $y_{neg}$，GPT 改写为精确的正例 $y_{pos}$
        - **DPO训练**：使用 DPO 损失 + SFT 辅助损失联合优化：$\mathcal{L} = \mathcal{L}_{DPO} + \lambda \mathcal{L}_{SFT}$，其中 $\lambda=0.1$
    - 设计动机：单纯的SFT训练无法解决MLLMs对特定音视频对象描述的模糊问题；DPO损失在正负响应差异小时效果有限，因此引入SFT损失稳定偏向正例的学习

4. **问题标记机制**:

    - 功能：在prompt中用 `<Q>` 和 `</Q>` 标记问题的起止位置
    - 核心思路：重构输入格式为 `USER:<system><Q></Q><video><audio><clues> Assistant:`
    - 设计动机：让线索聚合器能准确定位哪部分输入是问题，从而更精准地进行问题感知的特征提取

### 损失函数 / 训练策略

- Stage-I：标准语言建模损失，仅训练投影器
- Stage-II：标准语言建模损失 + LoRA（r=64, alpha=128），batch size=128，学习率2e-5
- ADPO阶段：$\mathcal{L}_{DPO} + 0.1 \cdot \mathcal{L}_{SFT}$，LoRA（r=64, alpha=16），batch size=1，学习率4e-6，$\beta=0.1$
- 全部训练在单张 NVIDIA A100 GPU 上完成

## 实验关键数据

### 主实验：视频文本生成 & 零样本视频问答

| 方法 | LLM大小 | 正确性 | 细节 | 上下文 | 时序 | 一致性 | MSRVTT-QA Acc | ActivityNet-QA Acc |
|------|---------|--------|------|--------|------|--------|--------------|-------------------|
| Video-LLaMA | 7B | 39.2 | 43.6 | 43.2 | 36.4 | 35.8 | 29.6 | 12.4 |
| Video-ChatGPT | 7B | 48.0 | 50.4 | 52.4 | 39.6 | 47.4 | 49.3 | 35.2 |
| LLaMA-VID | 7B | 59.2 | 60.0 | 70.6 | 49.2 | 50.2 | 57.7 | 47.1 |
| **CAT (Ours)** | **7B** | **61.6** | **62.0** | **69.8** | **56.2** | **57.8** | **62.1** | **50.2** |

### 主实验：闭合式 AVQA（Music-AVQA 全监督）

| 方法 | 语言模型 | 可训参数 | 音频平均 | 视觉平均 | 音视觉平均 | 总体平均 |
|------|---------|---------|---------|---------|-----------|---------|
| PSTP-Net | BERT | 4.3M | 70.9 | 77.3 | 72.6 | 73.5 |
| LAVISH | CLIP | 21.1M | 77.1 | 77.3 | 77.0 | - |
| VAST | BERT | - | - | - | - | 80.7 |
| **CAT-7B** | **LLaMA2** | **5.8M** | **84.9** | **86.1** | **83.2** | **84.3** |

### 消融实验

| 输入模态组合 | 正确性 | 细节 | 时序 |
|-------------|--------|------|------|
| 仅视觉 $x^{vid}$ | 38.6 | 40.4 | 34.0 |
| 视觉+音频 $x^{vid}+x^{aud}$ | 40.6 | 44.8 | 35.2 |
| 仅线索 $x^{cue}$ | 58.6 | 57.4 | 55.8 |
| 全部 $x^{vid}+x^{aud}+x^{cue}$ | **61.6** | **59.0** | **56.2** |

| ADPO消融（AVSD数据集） | B@4 | METEOR | CIDEr |
|----------------------|-----|--------|-------|
| Video-LLaMA 无ADPO | 18.4 | 40.1 | 63.7 |
| Video-LLaMA + ADPO | 22.2 | 48.2 | 69.2 |
| CAT 无ADPO | 28.9 | 56.2 | 74.8 |
| CAT + ADPO | **34.2** | **59.8** | **79.0** |

### 关键发现

- 仅线索token的效果已超过视觉+音频的联合输入，说明问题引导的特征提取极为重要
- 在线索聚合器中，去除视觉特征的影响远大于去除音频（B@4从30.8降到10.0 vs 18.8），视觉包含更多可挖掘的问题相关信息
- ADPO对不同MLLMs具有通用性——在Video-LLaMA上也能带来5.5个CIDEr点的提升
- 最佳线索token数量为K=48，过多或过少都会降低效果

## 亮点与洞察

- **线索聚合器设计精巧**：通过正序-逆序双块结构，既能做问题感知定位又不丢失原始模态信息，比简单的投影器或Q-Former更有针对性
- **ADPO策略务实有效**：将MLLMs的模糊描述问题重新建模为偏好优化，利用GPT自动构造正负样本对，避免人工标注
- **极少可训参数（5.8M）即达SOTA**：得益于LoRA和冻结编码器/LLM的设计，训练成本可控
- **AVinstruct数据集的构建**：为音视频联合问答提供了专门的指令微调数据，填补了该领域的数据空白

## 局限与展望

- ImageBind的音频编码只产生1个token，音频信息可能被过度压缩，限制了模型对复杂音频场景的理解
- ADPO依赖GPT进行模糊审查和重写，引入了外部模型的偏差
- 仅在LLaMA2-7B上验证，未测试更大规模的LLM（如13B、70B）
- 线索聚合器的视觉和音频Perceiver共享参数，可能不是最优设计，因为两种模态的特性差异较大
- 未探索更复杂的时序建模方法（如视频分段、时间戳定位）

## 相关工作与启发

- **Video-LLaMA**（Zhang et al., 2023）：直接用线性层桥接多模态，是CAT的主要对比基线
- **Q-Former**（Li et al., 2023）：BLIP-2提出的查询式特征压缩，被CAT借鉴用于线索聚合的第二步
- **DPO**（Rafailov et al., 2023）：绕过奖励模型直接优化偏好，启发了ADPO策略的设计
- **ChatBridge**（Zhao et al., 2023）：13B模型在AVQA上仍不如CAT-7B，说明模型大小不是唯一关键因素
- 启发：在多模态推理中，**问题引导的特征提取比简单地增加模态输入更重要**——仅用线索token就超过了视觉+音频的组合

## 评分

- 新颖性: ⭐⭐⭐⭐ 线索聚合器的双块设计和ADPO策略均有新意
- 实验充分度: ⭐⭐⭐⭐ 覆盖视频文本生成、零样本VQA、闭合/开放式AVQA四大类基准
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机明确，图表设计直观
- 价值: ⭐⭐⭐⭐ 为音视频问答任务提供了有效的模型架构和训练策略

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Genixer: Empowering Multimodal Large Language Model as a Powerful Data Generator](genixer_empowering_multimodal_large_language_model_as_a_powerful_data_generator.md)
- [\[ICLR 2026\] Can Vision-Language Models Answer Face to Face Questions in the Real-World?](../../ICLR2026/multimodal_vlm/can_vision-language_models_answer_face_to_face_questions_in_the_real-world.md)
- [\[ECCV 2024\] Self-Adapting Large Visual-Language Models to Edge Devices across Visual Modalities](self-adapting_large_visual-language_models_to_edge_devices_across_visual_modalit.md)
- [\[CVPR 2026\] VKG-QA: Visual Knowledge Graph-based Question Answer for Large Multimodal Models](../../CVPR2026/multimodal_vlm/vkg-qa_visual_knowledge_graph-based_question_answer_for_large_multimodal_models.md)
- [\[ECCV 2024\] LoA-Trans: Enhancing Visual Grounding by Location-Aware Transformers](loa-trans_enhancing_visual_grounding_by_location-aware_transformers.md)

</div>

<!-- RELATED:END -->
