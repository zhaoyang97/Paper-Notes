---
title: >-
  [论文解读] SpeakerLM: End-to-End Versatile Speaker Diarization and Recognition with Multimodal Large Language Models
description: >-
  [AAAI 2026][多模态VLM][说话人分离与识别] SpeakerLM 是首个专为端到端说话人分离与识别（SDR）设计的多模态大语言模型，通过音频编码器-投影器-LLM 架构和灵活的说话人注册机制，在多个公开基准上大幅超越级联基线系统（cpCER 绝对降低最高达 13.82%），并在域外测试集上展现强鲁棒性。
tags:
  - "AAAI 2026"
  - "多模态VLM"
  - "说话人分离与识别"
  - "多模态大语言模型"
  - "端到端SDR"
  - "说话人注册"
  - "语音理解"
---

# SpeakerLM: End-to-End Versatile Speaker Diarization and Recognition with Multimodal Large Language Models

**会议**: AAAI 2026  
**arXiv**: [2508.06372](https://arxiv.org/abs/2508.06372)  
**代码**: [项目页面](https://sites.google.com/view/speakerlm/overview)  
**领域**: 多模态VLM  
**关键词**: 说话人分离与识别, 多模态大语言模型, 端到端SDR, 说话人注册, 语音理解

## 一句话总结

SpeakerLM 是首个专为端到端说话人分离与识别（SDR）设计的多模态大语言模型，通过音频编码器-投影器-LLM 架构和灵活的说话人注册机制，在多个公开基准上大幅超越级联基线系统（cpCER 绝对降低最高达 13.82%），并在域外测试集上展现强鲁棒性。

## 研究背景与动机

### 领域现状
说话人分离与识别（SDR）任务旨在预测一段音频中"谁在什么时间说了什么"，是会议转录、对话系统等多说话人场景的核心技术。SDR 需要同时完成说话人分离（SD，回答"谁在什么时候说话"）和自动语音识别（ASR，回答"说了什么"）两个子任务。

### 现有痛点

**级联系统的错误传播**: 传统 SDR 系统采用 SD + ASR 级联框架，SD 模块的错误（如说话人边界不准、标签分配错误）会直接传递到 ASR 模块，导致转录质量下降

**重叠语音处理困难**: 传统 SD 系统基于 VAD（语音活动检测），假设每个时间段只有单个说话人，无法有效处理现实中常见的多人同时说话场景

**缺乏联合优化**: SD 和 ASR 模块通常使用不同数据集和框架独立训练，无法充分利用两个任务之间的协同效应

**LLM 后处理的局限性**: 使用 LLM 对级联系统输出进行修正虽有帮助，但受限于前端系统输出质量，且 LLM 的幻觉问题会导致修改原始文本内容

### 核心矛盾
如何将 LLM 不仅作为后处理工具，而是作为端到端 SDR 系统的核心组件，实现 SD 和 ASR 的统一建模和联合优化。

### 本文核心 idea
构建首个端到端多模态 LLM —— SpeakerLM，将说话人信息嵌入作为额外模态注入 LLM 的输入空间，并设计灵活的说话人注册机制以适应多种实际场景。

## 方法详解

### 整体框架
SpeakerLM 采用**编码器-投影器-LLM**架构，由音频编码器、音频投影器、说话人嵌入提取器、说话人投影器和文本 LLM 五大组件构成。多说话人音频经编码器处理后，通过投影器注入预训练文本 LLM 的特征空间；说话人注册信息通过独立的嵌入提取和投影路径融入模型。

### 关键设计

1. **音频编码器与投影器**:

    - **音频编码器**: 采用预训练的 SenseVoice-large 编码器初始化，具备多语言语音识别和音频事件检测能力
    - **音频投影器**: 随机初始化的两层 Transformer + CNN 层进行维度对齐
    - **设计动机**: SenseVoice-large 在多种音频理解任务上表现优异，提供强健的音频表征起点

2. **说话人嵌入提取器与投影器**:

    - **嵌入提取器**: 使用开源的 ERes2NetV2 模型，在多个说话人验证基准上达到 SOTA
    - **投影器**: 单层线性层进行维度对齐
    - **工作流程**: 已注册说话人的语音被分割为 2-10 秒片段 → 提取嵌入 → 多片段平均得到代表性嵌入 → 线性投影到 LLM 空间
    - **设计动机**: 通过冻结的预训练嵌入模型获得稳定且具判别性的说话人表征

3. **灵活说话人注册机制（三种模式）**:

    - **No-Regist（无注册）**: 不提供任何说话人先验信息，输出使用匿名 ID（如 spk 0, spk 1），对应传统级联 SD 系统的设定
    - **Match-Regist（精确注册）**: 所有出现的说话人都已预注册（$N_{rg} = N_{gt}$），模型需将每个说话人关联到正确姓名
    - **Over-Regist（过量注册）**: 注册说话人多于实际出现的说话人（$N_{rg} = N_{gt} + N_{ov}$），模型需判断哪些注册说话人不在当前音频中
    - **设计动机**: 覆盖从匿名转录到个性化说话人转录的多种实际应用场景，且 Over-Regist 更符合真实情况（大量用户池中仅有小部分参与）

4. **四阶段渐进训练策略**:

    - **Stage 1（ASR 预训练）**: 使用 60 万小时公开 ASR 数据训练 SpeakerLM-ASR，对 LLM 使用 LoRA 微调
    - **Stage 2（模拟数据对齐）**: 使用模拟 SDR 数据训练随机初始化的投影器，冻结 LLM 和音频编码器，实现快速音频-文本对齐
    - **Stage 3（真实数据编码器微调）**: 使用真实 SDR 数据联合微调音频编码器和投影器，冻结 LLM
    - **Stage 4（全模块联合微调）**: 联合微调所有模块，LLM 使用 LoRA，实现语言和声学信息的深度整合

### 损失函数 / 训练策略
- LLM 主干: Qwen2.5-7B-Instruct，利用其强指令跟随和通用语言理解能力
- 优化器: AdamW，学习率 1e-5 → 5e-5（warmup）→ 余弦衰减
- 动态批处理策略，最大 token 限制 6K
- 4 × NVIDIA A800 GPU，每阶段训练 1M 步

## 实验关键数据

### 主实验（No-Regist 条件）

| 系统 | 参数量 | AliMeeting cpCER↓ | AISHELL4 cpCER↓ | AISHELL5 cpCER↓ (域外) |
|------|--------|-------------------|-----------------|----------------------|
| 3D-Speaker+Para | 70M (4模型) | 24.94 | 26.01 | 64.12 |
| Pyannote+Para | 70M (4模型) | 24.45 | 28.22 | 68.37 |
| DiariZen-base+Para | 95M (4模型) | 23.97 | 27.27 | 66.89 |
| DiariZen-large+Para | 140M (4模型) | 23.20 | 25.78 | 61.81 |
| ChatGPT4.5 后处理 (零样本) | - (5模型) | 38.64 | 39.21 | 79.05 |
| Qwen2.5-7B 后处理 (微调) | 7B (5模型) | 22.65 | 24.93 | 61.63 |
| **SpeakerLM (7639h)** | **7B (1模型)** | **16.05** | **18.37** | **47.81** |

### 消融实验（说话人注册 & 嵌入模型）

| 配置 | AliMeeting CER | AliMeeting saCER | 说明 |
|------|---------------|------------------|------|
| Match-Regist + ERes2NetV2 | 13.98 | 15.57 | 最优说话人关联 |
| Over-Regist + ERes2NetV2 | 13.96 | 15.71 | 冗余说话人影响小 |
| Match-Regist + CAM++ | 14.74 | 17.23 | 嵌入模型质量影响显著 |
| Over-Regist + CAM++ | 14.71 | 16.92 | CAM++性能较ERes2NetV2差 |
| SA-Transformer (Match-Regist) | - | 41.55 | SpeakerLM 改进25.98% |

### 关键发现

1. **强大的数据扩展能力**: 从 212h 到 7639h 训练数据，cpCER 在 AliMeeting 上从 32.22 降至 16.05，Δcp 从 13.59 降至 2.08
2. **域外泛化性优异**: 在噪声汽车环境（AISHELL5-Eval）中，SpeakerLM 的 Δcp 仅为 0.57，远低于所有级联基线
3. **LLM 后处理不如端到端**: 零样本使用 ChatGPT-4.5 做后处理反而降低性能，因为 LLM 幻觉会修改说话人话语内容
4. **四阶段训练逐步提升**: 每个阶段都带来性能提升，Stage 3 和 4 对域外泛化至关重要
5. **嵌入模型质量直接影响性能**: ERes2NetV2 比 CAM++ 带来 1-2% 的 saCER 改进
6. **对过量注册说话人鲁棒**: 增加 Over-Regist 的冗余说话人数量（1到50）不会显著降低性能

## 亮点与洞察

1. **首个端到端 MLLM 用于 SDR**: 突破了 SD 和 ASR 独立建模的传统范式，实现真正的联合优化
2. **灵活的注册机制设计精巧**: 三种注册模式覆盖了匿名、精确和过量注册的完整谱系，具有很强的实用价值
3. **单模型 vs. 多模型级联**: SpeakerLM 用 1 个 7B 模型即超越了需要 4-5 个独立模型的级联系统
4. **数据扩展分析充分**: 详细展示了从 212h 到 7639h 的扩展曲线，为实际部署提供数据需求参考
5. **实验设计系统**: 覆盖域内/域外、有无注册、不同嵌入模型、数据规模等多个维度

## 局限与展望

1. **仅实验中文普通话**: 未验证多语言场景的适用性
2. **计算资源需求高**: 4 × A800 GPU 训练 1M 步×4阶段，训练成本不低
3. **依赖预训练说话人嵌入**: 嵌入提取器是冻结的，未探索端到端联合训练嵌入提取的可能性
4. **音频长度限制**: 训练和测试均限制在 40-50 秒片段，未验证长音频（如完整会议）的处理能力
5. **Over-Regist 的上限未探索**: 训练时 $N_{ov}$ 最多为 50，实际部署中可能面临更大规模的说话人池

## 相关工作与启发

- **3D-Speaker / Pyannote / DiariZen**: 传统级联 SD 系统的 SOTA 代表，DiariZen 通过引入 WavLM 预训练特征提升了 SD 性能
- **DiarizationLM**: 使用 LLM 后处理 SDR 输出的先驱工作，揭示了 LLM 零样本幻觉的问题
- **SA-Transformer**: 端到端 SA-ASR 的代表，但要求精确的说话人嵌入预注册
- **MinMo / Qwen2-Audio / Kimi-Audio**: 音频-文本 MLLM，主要关注单说话人场景
- **启发**: 将多模态 LLM 的能力从单说话人扩展到多说话人是一个重要且实用的方向；灵活的条件注入机制（如说话人嵌入投影）是实现这种扩展的关键技术

## 评分
- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] MarkushGrapher-2: End-to-end Multimodal Recognition of Chemical Structures](../../CVPR2026/multimodal_vlm/markushgrapher-2_end-to-end_multimodal_recognition_of_chemical_structures.md)
- [\[ACL 2026\] E2E-GMNER: End-to-End Generative Grounded Multimodal Named Entity Recognition](../../ACL2026/multimodal_vlm/e2e-gmner_end-to-end_generative_grounded_multimodal_named_entity_recognition.md)
- [\[ICLR 2026\] WebDS: An End-to-End Benchmark for Web-based Data Science](../../ICLR2026/multimodal_vlm/webds_an_end-to-end_benchmark_for_web-based_data_science.md)
- [\[AAAI 2026\] Positional Bias in Multimodal Embedding Models: Do They Favor the Beginning, the Middle, or the End?](positional_bias_in_multimodal_embedding_models_do_they_favor_the_beginning_the_m.md)
- [\[AAAI 2026\] SDEval: Safety Dynamic Evaluation for Multimodal Large Language Models](sdeval_safety_dynamic_evaluation_for_multimodal_large_language_models.md)

</div>

<!-- RELATED:END -->
