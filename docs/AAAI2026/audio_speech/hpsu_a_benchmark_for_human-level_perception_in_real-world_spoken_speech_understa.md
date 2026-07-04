---
title: >-
  [论文解读] HPSU: A Benchmark for Human-Level Perception in Real-World Spoken Speech Understanding
description: >-
  [AAAI2026][音频/语音][Speech LLM] 提出 HPSU 基准，包含 20,000+ 中英文专家标注样本和 16 项任务，系统评估 Speech LLM 在真实口语场景下的深层感知与推理能力，发现最强模型（Gemini 2.5 Pro，62.6%）与人类表现（87.3%）仍有巨大差距。
tags:
  - "AAAI2026"
  - "音频/语音"
  - "Speech LLM"
  - "benchmark"
  - "spoken language understanding"
  - "emotion reasoning"
  - "adversarial evaluation"
---

# HPSU: A Benchmark for Human-Level Perception in Real-World Spoken Speech Understanding

**会议**: AAAI2026  
**arXiv**: [2511.23178](https://arxiv.org/abs/2511.23178)  
**代码**: [Ichen12/HPSU-Benchmark](https://github.com/Ichen12/HPSU-Benchmark)  
**领域**: 音频语音  
**关键词**: Speech LLM, benchmark, spoken language understanding, emotion reasoning, adversarial evaluation

## 一句话总结
提出 HPSU 基准，包含 20,000+ 中英文专家标注样本和 16 项任务，系统评估 Speech LLM 在真实口语场景下的深层感知与推理能力，发现最强模型（Gemini 2.5 Pro，62.6%）与人类表现（87.3%）仍有巨大差距。

## 背景与动机
- 人类听觉理解涉及转录之外的说话人属性、潜在意图、复杂情感等多维度信息，现有 Speech LLM 在 ASR、SER 等基础任务上进步显著，但是否具备**类人水平的深层感知**仍未被充分探索
- 已有基准存在三大不足：(1) 多聚焦粗粒度任务或文本推理，忽视深层感知；(2) 依赖非交互数据或合成语音，脱离真实场景；(3) 多数局限单语言，无法进行跨语言分析
- 缺少一个全面评估 Speech LLM 从基础感知到复杂推理全谱系能力的统一基准

## 核心问题
当前 Speech LLM 能否像人类一样理解真实口语中的隐含意图、情感变化、话语潜台词等高阶语义？如何构建一个覆盖面广、质量高、支持双语的基准来量化这一差距？

## 方法详解

### 数据构建流程（三阶段半自动标注）
1. **数据收集与预处理**：从 CelebV-HQ、MAFW、MELD、MER 等公开视听数据集中筛选高质量片段，用音质评估工具打分，对低于阈值的样本做语音增强去噪（ClearerVoice-Studio）；每个样本保留视频、音频、Whisper/FireRedASR 生成的转录文本三个模态；VCTK 和内部方言数据集作为纯音频来源支持口音任务
2. **信息提取与交叉验证**：先将转录输入 LLM 推断可能的说话人状态（情感、意图等）作为先验知识；再将先验+音频/视频分别输入对应模型提取各模态表征；用**单模态交叉验证**策略（视觉描述喂入音频模型、反之亦然）检验跨模态一致性与逻辑连贯性
3. **信息融合与专家验证**：用 QWQ 推理模型做分层融合，将验证后的信息合成为三层描述——核心层（主情感/意图）、细节层（微表情等细粒度线索）、背景层（上下文）。80% 数据构成 **HPSC 数据集**（50,000 条语音-描述对），剩余 20% 经 Gemini 2.5 Pro 结构化为评测三元组，再由三名标注员独立审核，仅保留一致通过的样本（通过率 81.26%），形成最终 HPSU 基准

### 任务设计（5 大领域 16 项任务）

| 领域 | 任务 | 认知层级 |
|------|------|----------|
| 社会属性 | 年龄、性别、口音 | 基础感知 |
| 情感识别 | 主情感、多情感、情感强度 | 基础感知 |
| 情感推理 | 情感转变（易/难）、表里不一（易/难） | 复杂推理 |
| 非语言行为 | 说话风格、可视行为、描述判断 | 复杂推理 |
| 话语语境 | 意图推断、场景判断、潜台词理解 | 复杂推理 |

- 客观任务用单选，多情感用多选，潜台词等深层语义用是/否判断
- **干扰项生成**：用 Gemini 2.5 Pro 生成分级干扰项（similar / middle / opposite），提升区分度
- **对抗诱导协议**：每个任务实例包含三种 prompt 变体——标准无偏、正向诱导（含正确答案暗示）、负向诱导（含误导线索），用于评估鲁棒性

## 实验关键数据

### 主要结果（13 个模型 + 3 个基线）

| 模型 | 英文平均 | 中文平均 | 备注 |
|------|----------|----------|------|
| 人类 | ~84% | ~90% | 总平均 87.3% |
| Gemini 2.5 Pro | 最高 | 最高 | 总平均 62.6% |
| Qwen2.5-Omni | 次高 | 次高 | 总平均 60.0%，仅落后 Pro 2.6% |
| Whisper+GPT-4o | — | — | 在 Emos/Intent 上与 Gemini Pro 接近 |

### 关键发现
- **人机差距巨大**：最强模型 vs 人类差 24.7 个百分点；基础任务（性别识别 ~94%）接近人类，但高阶推理任务（潜台词、情感转变难题）差距悬殊
- **开源 vs 闭源差距小**：Qwen2.5-Omni 在部分任务上超过 Gemini 2.5 Pro（如英文 Emotion Shift Easy、中文 Visual Behavior），训练数据的多样性可能比模型闭源与否更关键
- **级联方法局限**：Whisper+GPT-4o 在依赖声学细节的任务（描述判断、情感转变难题）上表现较差，说明纯文本模态无法替代声学信息；但在语义推理类任务（Emos、Intent）上与端到端模型持平，反映了文本模态的泛化优势
- **对抗诱导脆弱**：所有模型对误导性 prompt 高度敏感，人类几乎免疫；模型在高歧义任务（年龄、口音、表里不一难题）和开放性任务（意图、场景）上尤其脆弱；原生多模态架构（Gemini）比文本 LLM 微调方案更鲁棒
- **分级评估洞察**：模型错误主多选 "Similar" 选项而非 "Opposite"，说明其理解了大概语义但缺乏细粒度区分能力——障碍不在于理解力缺失，而在于解决微妙歧义的精度不足
- **跨语言不一致**：模型在同一任务上的中英文性能差距远大于人类，例如潜台词理解任务中模型的中文表现显著弱于英文，可能因中文含蓄表达更难建模

## 亮点
- **任务覆盖全面**：16 项任务覆盖从基础属性识别到潜台词理解的完整认知谱系，是目前最全面的语音理解基准
- **标注流程创新**：多模态交叉验证 + 分层融合的半自动标注管线，有效解决高阶任务标注成本高的问题
- **评估维度丰富**：分级干扰项 + 对抗诱导协议 + 多语言对比，提供了远超简单准确率的诊断能力
- **规模与质量兼顾**：20,000+ 样本全部经专家验证，同时发布 50,000 条 HPSC 数据集供社区使用

## 局限与展望
- 数据来源集中于影视和社交媒体，缺乏会议、电话客服、医疗问诊等专业场景的覆盖
- 仅支持中英双语，未涵盖低资源语言（如阿拉伯语、印地语等）
- 对抗诱导仅限于文本 prompt 层面，未探索音频层面的对抗攻击（如添加噪声、变调等）
- 评估依赖 Gemini 2.5 Pro 作为自动裁判，可能引入系统偏差，尤其在 Gemini 自身作为被评模型时存在利益冲突
- HPSC 数据集的 SFT 实验仅在附录简要展示，缺乏对不同训练策略的深入分析
- GPT-4o 因安全策略拒绝回答大部分查询而被排除，限制了与顶级闭源模型的对比完整性
- 未评估模型在多轮对话场景下的深层理解能力

## 与相关工作的对比

| 基准 | 规模 | 情感推理 | 潜台词 | 对抗评估 | 双语 |
|------|------|----------|--------|----------|------|
| MMAU | 3k+ | 部分 | ✗ | ✗ | ✗ |
| AIR-Bench | 9k+ | 部分 | ✗ | ✗ | ✗ |
| MMSU | 5k | ✗ | ✗ | ✗ | ✗ |
| MMAR | 0.6k+ | ✗ | ✗ | ✗ | ✗ |
| **HPSU** | **20k+** | **✓** | **✓** | **✓** | **✓** |

HPSU 在规模、任务深度、对抗评估和多语言支持上全面超越现有基准。

## 启发与关联
- 揭示当前 Speech LLM 的训练数据偏差：模型在 ASR/SER 等低阶任务上过度优化，高阶认知数据严重匮乏，HPSU 为社区指明了需要重点攻克的方向
- 多模态交叉验证标注流程可推广到其他需要细粒度语义标注的任务（如多模态情感计算、话语行为分析）
- 对抗诱导实验表明，原生多模态架构（如 Gemini）比文本 LLM 微调方案更具鲁棒性，为模型架构设计提供参考——端到端多模态训练可能是通向鲁棒语音理解的更优路径
- 分级评估框架（True/Similar/Middle/Opposite）值得在其他主观性强的 NLP 基准中推广，比传统二元评分更能揭示模型的决策模式
- HPSC 的 50,000 条语音-描述对在可控语音生成、语音情感合成等下游任务中有潜在应用价值
- 级联方法在语义任务上的强劲表现提示：未来的端到端模型需要在保留声学细节的同时，匹配文本 LLM 的语义推理能力

## 评分
- 新颖性: ⭐⭐⭐⭐ — 首个系统性评估 Speech LLM 类人深层感知的基准，任务设计和对抗协议有创新
- 实验充分度: ⭐⭐⭐⭐⭐ — 13 个模型全面对比，多维度分析（对抗、分级、跨语言），实验非常扎实
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，图表丰富，但部分分析可进一步深入
- 价值: ⭐⭐⭐⭐ — 对语音理解社区有明确指导意义，HPSC 数据集有额外实用价值

## 补充说明
- 被评估的 13 个模型包括 11 个开源模型（Audio Flamingo 2/3/3.5、Kimi-Audio、Qwen-Audio、Qwen2-Audio、SALMONN、Soundwave、Baichuan-Omni-1.5、Qwen2.5-Omni、Audio-Reasoner）和 2 个闭源模型（Gemini 2.5 Flash/Pro）
- 人类基线由每种语言 10 名母语者在 300 个分层采样实例上完成
- 英文数据 11,580 条，中文数据 8,786 条，总计超过 20,000 条
- 论文同时发布 HPSC 数据集（50,000 条），SFT 实验表明在 HPSC 上微调可提升模型感知与理解能力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] MMSU: A Massive Multi-task Spoken Language Understanding and Reasoning Benchmark](../../ICLR2026/audio_speech/mmsu_a_massive_multi-task_spoken_language_understanding_and_reasoning_benchmark.md)
- [\[ICLR 2026\] EchoMind: An Interrelated Multi-level Benchmark for Evaluating Empathetic Speech Language Models](../../ICLR2026/audio_speech/echomind_an_interrelated_multi-level_benchmark_for_evaluating_empathetic_speech_.md)
- [\[NeurIPS 2025\] VITA-1.5: Towards GPT-4o Level Real-Time Vision and Speech Interaction](../../NeurIPS2025/audio_speech/vita-15_towards_gpt-4o_level_real-time_vision_and_speech_interaction.md)
- [\[ACL 2026\] MSU-Bench: Musical Score Understanding Benchmark](../../ACL2026/audio_speech/musical_score_understanding_benchmark_evaluating_large_language_models39_compreh.md)
- [\[ICLR 2026\] Human Behavior Atlas: Benchmarking Unified Psychological and Social Behavior Understanding](../../ICLR2026/audio_speech/human_behavior_atlas_benchmarking_unified_psychological_and_social_behavior_unde.md)

</div>

<!-- RELATED:END -->
