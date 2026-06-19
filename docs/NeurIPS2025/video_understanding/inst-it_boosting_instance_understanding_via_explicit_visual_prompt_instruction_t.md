---
title: >-
  [论文解读] INST-IT: Boosting Instance Understanding via Explicit Visual Prompt Instruction Tuning
description: >-
  [NeurIPS 2025][视频理解][实例级理解] 提出Inst-IT完整方案：通过GPT-4o辅助的自动标注管线生成实例级细粒度数据，构建Inst-IT Bench评测基准和335K QA对的指令微调数据集，以持续微调范式有效提升LMM的实例级理解能力，同时增强通用图像/视频理解。 大型多模态模型（LMM）在图像和视频…
tags:
  - "NeurIPS 2025"
  - "视频理解"
  - "实例级理解"
  - "视觉提示"
  - "指令微调"
  - "多模态大模型"
  - "时空理解"
---

# INST-IT: Boosting Instance Understanding via Explicit Visual Prompt Instruction Tuning

**会议**: NeurIPS 2025  
**arXiv**: [2412.03565](https://arxiv.org/abs/2412.03565)  
**代码**: [GitHub](https://github.com/inst-it/inst-it) | [HuggingFace](https://huggingface.co/Inst-IT)  
**领域**: 视频理解/多模态学习  
**关键词**: 实例级理解, 视觉提示, 指令微调, 多模态大模型, 时空理解  

## 一句话总结

提出Inst-IT完整方案：通过GPT-4o辅助的自动标注管线生成实例级细粒度数据，构建Inst-IT Bench评测基准和335K QA对的指令微调数据集，以持续微调范式有效提升LMM的实例级理解能力，同时增强通用图像/视频理解。

## 研究背景与动机

大型多模态模型（LMM）在图像和视频的整体理解方面取得了显著突破，但在**实例级理解**（Instance-level Understanding）方面仍然力不从心：

**什么是实例级理解**：识别图像/视频中特定实例（如某个人、某个物体）的属性、行为、关系和时序变化

**现实需求强烈**：用户通常关注的是画面中的特定目标而非整体场景——"那个穿红衣服的人在做什么？"

**现有模型的困境**：当前LMM在整体描述上表现良好，但当需要聚焦到某个特定实例时经常混淆或遗漏

一个有趣的现象推动了本文工作：**SOTA的LMM在给定显式视觉线索（explicit visual cues，如边框、箭头、标号）时，实例理解能力会大幅提升**。这说明模型具备实例理解的"潜力"，只是缺乏相应的训练数据来激活这种能力。

基于此，Inst-IT的核心思路是：**构建大规模的实例级视觉提示指令微调数据**，通过explicitly marking的方式引导模型学习实例级理解。

## 方法详解

### 整体框架

Inst-IT由三部分组成：

1. **Inst-IT Bench**：评测基准，诊断模型的实例级理解能力
2. **Inst-IT Dataset**：大规模指令微调数据集
3. **持续指令微调范式**：有效的训练策略

### 关键设计

**1. 自动标注管线**

利用GPT-4o作为标注引擎，逐帧处理视频：

- **实例标注**：在图像上用显式视觉标记（如 [1], [2], [3]）标注每个实例
- **帧级描述**：为每帧生成三层描述——(a) 各实例的独立描述，(b) 整体场景描述，(c) 与前帧的时序变化
- **视频级描述**：聚合所有帧级标注，生成按时间顺序组织的整体视频描述
- **QA生成**：基于标注生成以实例为中心的开放式问答对

**2. Inst-IT Bench（评测基准）**

- **规模**：~1000个图像QA + ~1000个视频QA
- **评估维度**：图像分支（实例属性、实例关系）+ 视频分支（时序追踪、行为理解）
- **格式**：同时支持开放式和多选题
- **独特性**：使用 `[ID]` 格式引用实例、`<timestamp>` 引用时间点，评估细粒度时空理解

**3. Inst-IT Dataset（微调数据集）**

- 21K视频 + 51K图像
- 21K视频级描述
- 207K帧级描述（51K图像 + 156K视频帧）
- 335K开放式QA对
- 目前最大的实例级视觉提示标注数据集

**4. 持续指令微调范式**

- 将Inst-IT Dataset与原有通用指令微调数据混合
- 采用持续训练（Continual Training）策略而非从头训练
- 仅增加少量实例级数据（~155K），即可在不损害通用能力的前提下大幅提升实例理解

### 损失函数 / 训练策略

- 标准的自回归语言建模损失
- 基于LLaVA-Next框架，分两阶段训练
- 混合比例：原始LLaVA-Next数据（~765K）+ Inst-IT数据（~155K）= ~920K

## 实验关键数据

### 主实验：Inst-IT Bench评测

| 模型 | Backbone | 图像OE | 图像MC | 视频OE | 视频MC |
|------|----------|--------|--------|--------|--------|
| Random Guess | — | — | 25.0 | — | 25.0 |
| GPT-4o | — | 74.1 | 84.8 | 65.5 | 81.0 |
| Gemini-1.5-pro | — | 69.9 | 79.7 | 61.4 | 76.7 |
| LLaVA-1.5 | Vicuna-7B | 41.6 | 32.1 | — | — |
| LLaVA-Next | Vicuna-7B | 46.0 | 42.4 | — | — |
| LLaVA-OV | Qwen2-7B | 48.0 | 71.7 | 33.2 | 45.6 |
| InternVL2 | InternLM2.5-7B | 58.6 | 66.5 | 39.8 | 45.5 |
| Qwen2-VL | Qwen2-7B | 48.3 | 64.9 | 38.2 | 59.4 |
| **LLaVA-Next-Inst-IT** | **Vicuna-7B** | **68.6** | **63.0** | **49.3** | **42.1** |
| **LLaVA-Next-Inst-IT** | **Qwen2-7B** | **67.9** | **75.3** | **45.7** | **53.3** |

关键发现：
- Inst-IT微调后，LLaVA-Next（Vicuna-7B）在图像OE上从46.0提升到68.6（+22.6），接近GPT-4o水平
- 在视频OE上从25.8提升到49.3（+23.5），提升幅度巨大

### 通用基准的表现

Inst-IT微调不仅提升实例理解，还增强了通用图像/视频理解：

| 基准 | LLaVA-Next (原始) | +Inst-IT | 提升 |
|------|-------------------|----------|------|
| AI2D | 65.2 | 68.7 | +3.5 |
| TextVQA | 63.8 | 65.1 | +1.3 |
| EgoSchema | 42.1 | 48.5 | +6.4 |
| MVBench | 56.3 | 60.8 | +4.5 |

### 消融实验

**数据组成的重要性**：

| 数据配置 | Inst-IT Bench (MC) | AI2D | EgoSchema |
|----------|-------------------|------|-----------|
| LLaVA-Next baseline | 42.4 | 65.2 | 42.1 |
| + 仅图像实例数据 | 56.8 | 67.1 | 43.5 |
| + 仅视频实例数据 | 48.2 | 65.8 | 47.2 |
| + 图像+视频实例数据 | 63.0 | 68.7 | 48.5 |

图像和视频实例数据的组合效果最佳，且视频数据对EgoSchema等时序理解任务贡献更大。

**视觉提示方式的影响**：

| 视觉提示类型 | Inst-IT Bench (图像MC) | Inst-IT Bench (视频MC) |
|-------------|----------------------|----------------------|
| 无视觉提示 | 42.4 | 24.8 |
| 边框(Bounding Box) | 55.1 | 35.2 |
| 带标号的标记([ID]) | 63.0 | 42.1 |

显式的[ID]标号标记比简单边框更有效，因为ID系统可以跨帧追踪同一实例。

### 关键发现

1. **实例理解是LMM的显著短板**：即使是GPT-4o在Inst-IT Bench上也仅有74-85分，远非完美
2. **显式视觉线索极其有效**：加入标号后模型的实例理解能力飞跃式提升
3. **实例数据增强通用能力**：实例级理解训练不与通用能力冲突，反而相互促进
4. **少量数据高效**：仅~155K实例数据就能带来巨大提升

## 亮点与洞察

1. **完整的生态系统**：Bench + Dataset + Training = 从评测到数据到训练的完整闭环
2. **自动化标注**：利用GPT-4o的能力自动生成高质量的实例级标注，可扩展性强
3. **通用能力增强**：这一发现违背直觉——细粒度实例训练竟然能提升宏观理解能力，说明实例级理解是一种"基础能力"
4. **简洁有效的视觉提示设计**：[ID]标号系统简单但强大，天然支持跨帧追踪

## 局限与展望

1. **标注成本**：依赖GPT-4o进行标注，成本较高，且可能引入GPT-4o自身的偏差
2. **模型规模有限**：仅在7B模型上验证，更大模型上的效果待确认
3. **实例检测前置**：标注管线依赖现有检测/分割模型提供实例位置，检测失败会级联影响
4. **开放世界局限**：当前数据仅覆盖有限的场景类型，对罕见场景的泛化性不明
5. **训练代码未开源**：目前仅开源了评测工具和模型权重

## 相关工作与启发

- **ViP-LLaVA / SoM-LLaVA**：早期的视觉提示LMM工作，但缺乏大规模实例级数据
- **LLaVA-OneVision**：多阶段指令微调的代表，Inst-IT的持续微调策略受其启发
- **VideoGLaMM**：像素级视频Grounding，关注更细粒度的定位
- **GPT-4o**：既是标注工具也是评测上界，展示了充分训练的实例理解潜力
- 启发：**显式视觉标记是"释放"模型潜在能力的低成本高效手段**，这一思路可推广到其他细粒度任务

## 评分

- 新颖性：⭐⭐⭐⭐（首个大规模实例级视觉提示微调方案）
- 技术深度：⭐⭐⭐⭐
- 实验充分度：⭐⭐⭐⭐⭐
- 实用性：⭐⭐⭐⭐（数据和模型开源）
- 写作质量：⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] MoniTor: Exploiting Large Language Models with Instruction for Online Video Anomaly Detection](monitor_exploiting_large_language_models_with_instruction_for_online_video_anoma.md)
- [\[CVPR 2026\] Beyond Explicit Language: Plug-and-Play Visual-to-Linguistic Modeling Toward General Object Tracking](../../CVPR2026/video_understanding/beyond_explicit_language_plug-and-play_visual-to-linguistic_modeling_toward_gene.md)
- [\[NeurIPS 2025\] TempSamp-R1: Effective Temporal Sampling with Reinforcement Fine-Tuning for Video LLMs](tempsampr1_effective_temporal_sampling_with_reinforcement_fi.md)
- [\[ICCV 2025\] DynImg: Key Frames with Visual Prompts are Good Representation for Multi-Modal Video Understanding](../../ICCV2025/video_understanding/dynimg_key_frames_with_visual_prompts_are_good_representation_for_multi-modal_vi.md)
- [\[ICCV 2025\] RainbowPrompt: Diversity-Enhanced Prompt-Evolving for Continual Learning](../../ICCV2025/video_understanding/rainbowprompt_diversity-enhanced_prompt-evolving_for_continual_learning.md)

</div>

<!-- RELATED:END -->
