---
title: >-
  [论文解读] Burn After Reading: Do Multimodal Large Language Models Truly Capture Order of Events in Image Sequences?
description: >-
  [ACL 2025][多模态VLM][时序推理] 提出 TempVS 基准测试，系统评估 38 个 MLLM 在图像序列中对多事件时序关系的 grounding 和推理能力，揭示 SOTA 模型与人类之间存在巨大性能差距。 领域现状：多模态大语言模型（MLLMs）在单图视觉理解任务上表现出色，但对多图场景下时序理解和推理的评…
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "时序推理"
  - "多图理解"
  - "事件排序"
  - "benchmark"
  - "多模态大语言模型"
---

# Burn After Reading: Do Multimodal Large Language Models Truly Capture Order of Events in Image Sequences?

**会议**: ACL 2025  
**arXiv**: [2506.10415](https://arxiv.org/abs/2506.10415)  
**代码**: [GitHub](https://github.com/yjsong22/TempVS)  
**领域**: 多模态VLM  
**关键词**: 时序推理, 多图理解, 事件排序, benchmark, 多模态大语言模型

## 一句话总结

提出 TempVS 基准测试，系统评估 38 个 MLLM 在图像序列中对多事件时序关系的 grounding 和推理能力，揭示 SOTA 模型与人类之间存在巨大性能差距。

## 研究背景与动机

**领域现状**：多模态大语言模型（MLLMs）在单图视觉理解任务上表现出色，但对多图场景下时序理解和推理的评估相对薄弱。现有多图基准主要关注跨图识别和引用，较少涉及时序关系。

**现有痛点**：已有时序评估存在三大缺陷：(a) 部分任务仅靠单张图片即可回答，无需理解序列；(b) 部分任务过度依赖常识和世界知识（如按烹饪步骤排序）；(c) 部分基准使用图像中不存在的干扰选项，模型可通过目标存在性推断答案。

**核心矛盾**：现有基准无法真正评估模型对视觉故事中多事件时序关系的理解——模型可能通过捷径而非真正的时序推理来"通过"测试。

**本文目标**：构建一个严格防作弊的时序基准，回答"现有 MLLM 是否真正理解图像序列中事件的时序顺序"这一核心问题。

**切入角度**：从视觉故事（visual story）出发，选择事件相对独立且难以从前序事件预测后续事件的图像序列，强制模型整合视觉和语言两种模态来完成任务。

**核心 idea**：通过设计三种主测试（事件关系推断、句子排序、图像排序）及配套 grounding 测试，排除单模态捷径，原子化评估 MLLM 的时序理解能力。

## 方法详解

### 整体框架

TempVS 基准包含 **2,085 个图像序列**（9,803 张图像），覆盖卡通动画（FlintstonesSV、PororoSV）、电影场景（VWP）和日常生活相册（VIST）四个数据源，共生成 **15,192 道多选题**。

基准设计为三层结构：
- **MT1: 事件关系推断** — 判断描述两/三个事件时序关系的陈述是否与图像序列一致（True/False）
- **MT2: 句子排序** — 给定有序图像序列和打乱的句子集，从五个选项中选择正确句子顺序
- **MT3: 图像排序** — 给定文本描述和打乱的图像集，选择正确图像顺序
- **GT: Grounding 测试** — 给定事件描述和图像序列，定位对应图像

### 关键设计

**数据格式化**：每个图像序列表示为 $\mathcal{S} = [(I_1, C_1, E_1), ..., (I_n, C_n, E_n)]$，其中 $I_i$ 为图像，$C_i$ 为原始描述，$E_i$ 为提取的简化事件。

**模板驱动的正负样本构造**：MT1 使用 10 种模板（5 种二事件 + 4 种三事件）生成正负陈述，负样本通过交换事件子句位置构造，保持相同时序连接词，仅改变事件顺序。例如：
- 正: "$E_j$ after $E_i$" → 负: "$E_i$ after $E_j$"
- 正: "$E_i$. Then, $E_j$" → 负: "$E_j$. Then, $E_i$"

**多层过滤机制**：
1. 使用 Detectron2 保留 ≥60% 图像含人物的序列
2. 移除含状态动词（如 belong, love）的序列以避免时间重叠
3. 通过 BERTScore 和 CLIP 余弦相似度移除过于相似的描述和图像
4. 使用 CLIP 跨模态相似度过滤歧义对，确保 $sim(I_i, E_i) > sim(I_i, E_j)$

**防止语言偏差**：使用 Phi-3.5-mini [4B]、Llama-3.1 [8B]、Qwen-2.5 [72B] 三个纯文本 LLM 过滤，在 MT1 和 MT2 中丢弃至少两个 LLM 仅凭文本就能答对的样本。

### 评估指标

- **MT 准确率**：标准多选准确率
- **GT_strict**：模型通过所有对应 grounding 测试的图像序列比例
- **MT|GT_strict**：仅在通过所有 grounding 测试的前提下计算主任务准确率

## 实验关键数据

### 主实验

评估了 38 个 MLLM（参数量从 0.5B 到 78B），主要结果：

| 模型 | 参数 | MT1(二事件) | MT1(三事件) | MT2(事件) | MT2(描述) | MT3 |
|------|------|------------|------------|----------|----------|-----|
| InternVL2.5-78B-MPO | 78B | 58.5 | 61.4 | **79.8** | **86.3** | **53.8** |
| InternVL2.5-26B-MPO | 26B | **60.3** | 62.1 | 69.9 | 76.9 | 34.4 |
| GPT-4o | API | 58.3 | **64.5** | 53.4 | 61.5 | 22.6 |
| LLaVA-OneVision-72B | 72B | 59.3 | 61.5 | 65.2 | 75.1 | 27.6 |
| 随机基线 | - | 50 | 50 | 20 | 20 | 20 |
| **人类** | - | **82.5** | **80.0** | - | - | - |

### 消融实验

**事件距离的影响**：事件间距离越远，MT1 准确率越高（模型更容易区分时间上相距较远的事件）。

**语言结构的影响**：使用原始 caption 比使用简化事件描述效果更好（caption 提供额外上下文和时序线索）。

**Chain-of-Thought (CoT)**：对大多数模型提升有限，某些情况下甚至降低性能。

**Grounding 与时序推理的关系**：
- InternVL2.5-78B-MPO 在 MT2 中，加入 GT_strict 约束后准确率从 79.8% 提升至 **96.6%**（事件）和从 86.3% 提升至 **96.4%**（描述）
- 说明 grounding 能力是时序推理的重要前提

### 关键发现

1. **MT1 接近随机**：大多数 ≤7B 参数模型在 MT1 上准确率约 50%（随机水平），在 MT2/MT3 上约 20%
2. **图像排序最难**：最佳模型 MT3 仅 53.8%，远低于 MT2 的 86.3%
3. **GPT-4o 表现不均**：grounding 最优但排序任务大幅落后于开源最优模型
4. **模型规模 + 后训练双重重要**：InternVL2.5-MPO 在所有任务上系统性超越对应的非 MPO 版本

## 亮点与洞察

- **防作弊设计精妙**：三重过滤（视觉相似度、文本相似度、纯文本 LLM）有效排除捷径，确保评测的纯粹性
- **Grounding-推理解耦分析**：通过 GT + MT|GT_strict 指标清晰展示"能认图 ≠ 能理解时序"
- **跨领域数据源**：覆盖卡通、电影、日常照片，增强评估泛化性
- **模板多样性**：328 种 prompt 变体避免特定 prompt 偏差

## 局限与展望

1. 仅使用多选题格式，未评估开放式生成能力
2. 图像序列大多为 5 张，未测试更长序列（如 10+ 张）的时序推理
3. 仅评估事件的线性时序关系，未涉及并行/重叠等复杂时间关系
4. 评估时将多图水平拼接为单图输入，可能损失单图细节
5. 可在视频理解模型上做进一步验证

## 相关工作与启发

- **MIBench / MuirBench / MMIU** 等多图基准主要关注跨图识别，TempVS 填补了时序推理评估空白
- **Mementos** 关注序列图像的幻觉检测，但不涉及逆向推理（从文本推图序）
- 启发方向：设计专门的时序对齐预训练目标、引入事件图（event graph）结构增强模型时序推理

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首个系统化多事件时序 grounding 与推理基准，防作弊设计创新
- **实验充分度**: ⭐⭐⭐⭐⭐ — 38 个模型、4 个数据源、人类对照、多维度分析
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，任务定义精确
- **价值**: ⭐⭐⭐⭐ — 为 MLLM 时序能力评估提供标准化工具，揭示重要能力缺陷

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Do Vision-Language Models Have Internal World Models? Towards an Atomic Evaluation](do_vision-language_models_have_internal_world_models_towards_an_atomic_evaluatio.md)
- [\[CVPR 2026\] Personalized Image Descriptions from Attention Sequences](../../CVPR2026/multimodal_vlm/personalized_image_descriptions_from_attention_sequences.md)
- [\[CVPR 2026\] Do Vision-Language Models Measure Up? Benchmarking Visual Measurement Reading with MeasureBench](../../CVPR2026/multimodal_vlm/do_vision-language_models_measure_up_benchmarking_visual_measurement_reading_wit.md)
- [\[ACL 2025\] Single-to-mix Modality Alignment with Multimodal Large Language Model for Document Image Machine Translation](single-to-mix_modality_alignment_with_multimodal_large_language_model_for_docume.md)
- [\[ACL 2025\] AlignMMBench: Evaluating Chinese Multimodal Alignment in Large Vision-Language Models](alignmmbench_evaluating_chinese_multimodal_alignment_in_large_vision-language_mo.md)

</div>

<!-- RELATED:END -->
