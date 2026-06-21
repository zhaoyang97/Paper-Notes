---
title: >-
  [论文解读] FlexiVoice: Enabling Flexible Style Control in Zero-Shot TTS with Natural Language Instructions
description: >-
  [ICLR 2026][音频/语音][零样本 TTS] FlexiVoice 用一个 LLM 内核同时接收文本、风格指令和音色参考语音，通过"DPO→解耦 GRPO→指令 GRPO"三阶段渐进式后训练，专门破解风格-音色-内容三者纠缠的难题，让零样本 TTS 既能精准跟随自然语言风格指令、又能稳定克隆参考音色。
tags:
  - "ICLR 2026"
  - "音频/语音"
  - "零样本 TTS"
  - "自然语言指令控制"
  - "风格-音色解耦"
  - "DPO"
  - "GRPO"
  - "渐进式后训练"
---

# FlexiVoice: Enabling Flexible Style Control in Zero-Shot TTS with Natural Language Instructions

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=F7GmbfyVg9](https://openreview.net/forum?id=F7GmbfyVg9)  
**代码**: [https://flexi-voice.github.io/](https://flexi-voice.github.io/)（音频样例页）  
**领域**: 语音合成 / 可控 TTS  
**关键词**: 零样本 TTS, 自然语言指令控制, 风格-音色解耦, DPO, GRPO, 渐进式后训练  

## 一句话总结
FlexiVoice 用一个 LLM 内核同时接收文本、风格指令和音色参考语音，通过"DPO→解耦 GRPO→指令 GRPO"三阶段渐进式后训练，专门破解风格-音色-内容三者纠缠的难题，让零样本 TTS 既能精准跟随自然语言风格指令、又能稳定克隆参考音色。

## 研究背景与动机
**领域现状**：零样本 TTS 已能用一小段参考语音克隆说话人音色，而要进一步控制"说话风格"（情绪、语速等）有两条路线——一是像 Vevo、IndexTTS2 那样再用一段风格参考语音，二是指令式 TTS（PromptTTS、VoxInstruct、CosyVoice2 等）用自然语言描述目标风格。

**现有痛点**：指令式模型常常顾此失彼——要么不能忠实跟随指令，要么跟了指令就保不住音色一致性。根因是监督训练里模型会过度依赖参考语音的强声学先验（**音色泄漏**），或从文本内容里推断韵律（**内容泄漏**），结果直接无视显式的风格指令。

**核心矛盾**：作者把它命名为 **Style-Timbre-Content Conflict（风格-音色-内容冲突）**。当指令要"开心"、参考音是"悲伤"、文本内容又透着"难过"时，三个模态相互打架，单纯加一个指令条件根本压不住纠缠的声学线索。

**本文目标**：造一个统一框架，不只是"条件化"模型，而是**主动把风格从音色和内容里解耦出来**，并在冲突声学线索下强制服从指令。

**核心 idea**：**把"跟随指令"从简单的条件输入升级为一个严格的解耦过程**——用一套课程学习式的渐进后训练（PPT），先对齐、再解耦、最后泛化到开放指令，配合一个 LLM 标注的大规模指令-语音数据集 FlexiVoice-Instruct 打底。

## 方法详解

### 整体框架
FlexiVoice 建在一个预训练 LLM 之上：语音 tokenizer 把语音转成离散 token，LLM 内核吃下文本、自然语言指令和参考语音 token，自回归生成语音 token，再经 flow matching 转 Mel 谱、最后由 vocoder 出波形。训练分两段：先用 Emilia + FlexiVoice-Instruct 预训练出 FlexiVoice-Base（只训 LLM 内核，其余冻结，预训练阶段不喂参考语音），再用三阶段 **渐进式后训练（PPT）** 把 Base 打磨成最终的 FlexiVoice。PPT 的灵感来自课程学习——从简单目标起步、逐步推进到困难目标。

```mermaid
flowchart LR
    A[预训练 FlexiVoice-Base<br/>Emilia + FlexiVoice-Instruct] --> B[S1 多模态 DPO<br/>对齐指令+音色参考]
    B --> C[S2 解耦 GRPO<br/>多目标奖励拆开风格/音色/内容]
    C --> D[S3 指令 GRPO<br/>ALM 奖励泛化到开放指令]
    D --> E[FlexiVoice]
```

### 关键设计

**1. FlexiVoice-Instruct 数据集：用 LLM 反推自然指令**：要让模型在预训练阶段就具备基本的指令跟随能力，必须有大规模、风格多样的指令-语音配对。作者构建了 4,316 小时的数据集，巧妙之处在于不直接分析语音，而是利用语音的文本元信息——Emilia 的音频来自视频平台和播客，带有视频标题、标签等源信息，结合转写文本就能让 Deepseek-V3 反推出说话风格和场景，生成开放式指令（如"关于植物园的纪录片旁白，语气平和而富有教育意义"）；另外引入两款热门游戏的配音数据，因为说话风格和角色人格强绑定，LLM 能识别知名角色并捕捉其标志性风格。整个流程还加了"信息价值检查"——让 LLM 先评估元信息对推断风格的价值，过滤掉 URL、元信息与转写冲突等噪声样本。

**2. S1 多模态 DPO：先建立指令+音色的初步对齐**：第一阶段聚焦情绪指令以降低任务复杂度，指令限定为 `Use {label} emotion to read it`（Neutral/Happy/Angry/Sad/Surprised 五类）。情绪任务的偏好数据可直接从语音情绪识别数据集 ESD 拿到——同一说话人读同一句话的不同情绪版本：给定目标情绪标签，选目标情绪的句子做偏好样本 $y_w$、同句不同情绪做非偏好样本 $y_l$、再用同说话人的中性样本做音色参考。DPO 无需显式奖励模型就把模型输出对齐到指令和参考语音：
$$\mathcal{L}_{\text{DPO}}(\pi_\theta;\pi_{\text{ref}}) = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$
其中 $x$ 含指令、文本和参考，$\pi_\theta$ 和 $\pi_{\text{ref}}$ 都从 FlexiVoice-Base 初始化。

**3. S2 解耦 GRPO：用冲突场景强制拆开风格与音色/内容**：DPO 后模型在中性参考下能跟随情绪指令，但当参考语音或文本本身带情绪、且与指令目标情绪冲突时仍会受干扰。S2 主动构造冲突场景（如"Happy 指令 vs Sad 参考"），用多目标 GRPO 让两个奖励互相牵制：风格奖励 $r_{\text{ser}}\in(0,1)$ 由 Emotion2vec-Large 给出目标情绪的概率，惩罚模型从参考/文本泄漏风格；音色奖励 $r_{\text{sv}}\in\{0,1\}$ 由 CAM++ 说话人验证给出，保证音色不丢。联合优势把两路 z-score 归一化后相加：
$$A^i_{\text{emo}} = \frac{r^i_{\text{ser}} - \text{mean}(r_{\text{ser}})}{\text{std}(r_{\text{ser}})} + \frac{r^i_{\text{sv}} - \text{mean}(r_{\text{sv}})}{\text{std}(r_{\text{sv}})}$$
为最大化总奖励，模型被迫把风格、音色、内容三者拆开。

**4. S3 指令 GRPO：用音频语言模型奖励泛化到开放指令**：最后一阶段把能力从情绪扩展到复杂真实指令。这种规模下无法构造偏好对，于是直接上 GRPO；但评判开放指令与语音是否一致很难，作者用开源的 Kimi-Audio-7B-Instruct 当奖励模型，提示它对"生成语音是否匹配指令"输出 yes/no，映射为 $r_{\text{llm}}\in\{1,0\}$。这一阶段只用指令+文本（丢掉参考，因为参考可能与开放约束如性别冲突而破坏训练稳定）。为防灾难性遗忘，混入一小部分 S2 数据，最终成为多任务多目标优化：
$$A^i_{\text{ins}} = \frac{r^i_{\text{llm}} - \text{mean}(r_{\text{llm}})}{\text{std}(r_{\text{llm}})}, \quad A^i = \begin{cases} A^i_{\text{emo}} & \text{S2 输入} \\ A^i_{\text{ins}} & \text{S3 输入} \end{cases}$$

## 实验关键数据

### 主实验表格

多模态可控性与解耦评测（自建集，EN，TR=文本+参考语音，硬难度参考音带冲突情绪）：

| 模型 | TO-Easy ACC-I↑ | TO-Hard ACC-I↑/ACC-T↓ | TR-Hard ACC-I↑/ACC-R↓/SV↑ |
|------|------|------|------|
| VoxInstruct | 70.6 | 17.8 / 41.2 | 23.9 / 0.80 / 90.6 |
| CosyVoice2 | - | - | 14.4 / 0.84 / 99.8 |
| FlexiVoice-Base | 72.4 | 39.4 / 30.6 | 32.2 / 0.78 / 99.4 |
| **FlexiVoice** | **97.4** | **89.4 / 6.6** | **78.2 / 10.6 / 95.8** |

复杂指令跟随（InstructTTSEval，Avg. 宏平均准确率）：

| 模型 | EN Avg. | ZH Avg. |
|------|------|------|
| Gemini-pro（闭源） | 80.3 | 84.8 |
| GPT-4o-mini-TTS（闭源） | 68.5 | 51.1 |
| MiMo-Audio-7B-Instruct | 72.6 | 70.5 |
| VoxInstruct | 50.4 | 47.5 |
| FlexiVoice-Base | 66.4 | 58.4 |
| **FlexiVoice** | **79.3** | **70.8** |

### 消融实验表格

不同训练顺序与策略（EN）：

| 训练策略 | Decoupling Avg. | InstructTTSEval Avg. |
|------|------|------|
| FlexiVoice-Base | 54.9 | 66.4 |
| + S3（直接上复杂指令） | 54.7 | 72.3 |
| + S3→S1→S2（错序） | 84.4 | 74.8 |
| + S1→S2+S3（联合训练） | 84.1 | 75.5 |
| + S1 | 83.3 | 69.0 |
| + S1→S2 | 88.5 | 71.7 |
| **+ S1→S2→S3（PPT 完整）** | **88.7** | **79.3** |

### 关键发现
- **三模态全面解耦**：FlexiVoice 把 TO-Hard 的 ACC-I 从 Base 的 39.4 拉到 89.4、ACC-T 从 30.6 降到 6.6，中文场景 easy/hard 差距仅 1.4%，说明风格几乎完全由指令决定、不再被文本内容或参考音的情绪带跑。
- **训练顺序是刚需**：直接上 S3 只有 72.3，错序 S3→S1→S2 反而引发灾难性遗忘（74.8）。S1 的 DPO 是必要的"冷启动"地基，先建立稳健的多模态响应，模型才能应对后续解耦和泛化。
- **渐进优于联合**：S2（固定分类器压制风格泄漏）和 S3（ALM 奖励鼓励开放风格泛化）梯度方向冲突，联合训练（75.5）干扰彼此，分阶段（79.3）才能各自学好。
- **风格-音色权衡是预期现象**：Base 的 SV 偶尔略高于 FlexiVoice，是因为它直接克隆参考韵律（高相似但 ACC-I 低）；FlexiVoice 为满足风格指令必须改音高、能量等声学特征，导致说话人嵌入余弦相似度小幅下降，但仍保持高 SV 且指令跟随显著更好。
- **可懂度与音质**：WER/CER 比 Base 略升（因 ASR 在高表现力语音上本就退化），但 Q-MOS 更高（EN-TO-Easy 4.08 vs 3.72）、CMOS 最高 +0.9，人耳偏好更好。

## 亮点与洞察
- **把"指令跟随"问题重新定义为"解耦"问题**，这是全文最核心的视角转换——指出单纯条件化压不住模态冲突，必须用奖励信号主动拆开风格/音色/内容，并以 z-score 归一化的多目标优势函数把"跟指令"和"保音色"两个目标平衡起来。
- **课程学习式的 RL 后训练 curriculum 设计扎实**：DPO 做冷启动对齐、解耦 GRPO 在冲突场景下做硬约束、指令 GRPO 用 ALM 奖励泛化，三段顺序经消融严格验证，而非拍脑袋。
- **数据构造很务实**：不分析语音本身，而是用视频标题/标签/角色名等"廉价"文本元信息让 LLM 反推风格，配合信息价值过滤，低成本拿到了自然且多样的指令标注。
- **奖励模型全用开源组件**（Emotion2vec、CAM++、Kimi-Audio），把成本高的 Gemini 评判替换掉，使 GRPO 在大规模指令上可负担。

## 局限与展望
- **奖励模型质量是天花板**：S3 用 Kimi-Audio 的 yes/no 二值判断作奖励，其判别能力直接决定开放指令的泛化上限，二值奖励也比连续分数信息量少。
- **风格仍以情绪为主轴**：S1/S2 都围绕五类情绪标签和 ESD 这类配对数据展开，对语速、音高等非情绪属性的解耦主要靠 S3 间接获得，细粒度连续属性控制仍有提升空间。
- **与闭源差距尚存**：中文 InstructTTSEval 上 70.8 仍落后 Gemini 系列（84+），跨语种一致性和更复杂角色扮演还有差距。
- **音色相似度的小幅牺牲**虽被解释为"必要的声学改动"，但在对音色保真度要求极高的场景（如配音替身）是否可接受仍需评估。

## 相关工作与启发
- **可控语音合成两条线**：零样本 TTS（CosyVoice2、IndexTTS2 等）克隆音色但弱控风格；指令式 TTS（PromptTTS、PromptStyle、Parler-TTS、VoxInstruct、AudioBox、ControlSpeech）用自然语言指风格但缺乏稳健解耦。FlexiVoice 想做统一框架兼顾两者。
- **指令-语音数据集**：TextrolSpeech、Audiobox、Parler-TTS、SpeechCraft、ParaSpeechCaps 等，多为模板化描述、来源单一，FlexiVoice-Instruct 主打更自然、更多样的标注。
- **语音合成中的 RL**：INTP（DPO 提可懂度）、Emo-DPO（情绪偏好对齐）、Vevo2（多目标后训练联合提可懂度和韵律）。FlexiVoice 的差异在于用渐进 curriculum 显式攻克模态解耦+复杂指令跟随。
- **启发**：当多个控制因子在声学上纠缠时，"主动构造冲突场景 + 多目标奖励互相牵制"是比"加条件"更有效的解耦手段；这个思路可迁移到图像/视频生成里的属性解耦、以及多模态可控生成的通用对齐范式。

## 评分
- **新颖性**: ⭐⭐⭐⭐ 把 TTS 的指令跟随重新框定为风格-音色-内容解耦问题，并设计经消融验证的三阶段 RL 课程，视角和方法组合都有原创性。
- **实验充分度**: ⭐⭐⭐⭐ 自建解耦评测+InstructTTSEval 双基准、中英双语、客观+主观指标齐全，训练顺序/联合 vs 渐进/逐阶段累积增益的消融做得相当完整。
- **写作质量**: ⭐⭐⭐⭐ 问题命名清晰（Style-Timbre-Content Conflict）、动机与方法逻辑连贯，公式和图示到位。
- **价值**: ⭐⭐⭐⭐ 在零样本指令式 TTS 这一热点上给出可复现的解耦范式，开源奖励组件降低复现门槛，对工业级可控语音合成有直接参考价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] FC-TTS: Style and Timbre Control in Zero-Shot Text-to-Speech with Disentangled Speech Representations](../../ACL2026/audio_speech/fc-tts_style_and_timbre_control_in_zero-shot_text-to-speech_with_disentangled_sp.md)
- [\[ACL 2026\] ReStyle-TTS: Relative and Continuous Style Control for Zero-Shot Speech Synthesis](../../ACL2026/audio_speech/restyle-tts_relative_and_continuous_style_control_for_zero-shot_speech_synthesis.md)
- [\[ACL 2025\] ControlSpeech: Towards Simultaneous and Independent Zero-shot Speaker Cloning and Zero-shot Language Style Control](../../ACL2025/audio_speech/controlspeech_zero_shot.md)
- [\[ICLR 2026\] From Natural Alignment to Conditional Controllability in Multimodal Dialogue](from_natural_alignment_to_conditional_controllability_in_multimodal_dialogue.md)
- [\[ICLR 2026\] When Style Breaks Safety: Defending LLMs Against Superficial Style Alignment](when_style_breaks_safety_defending_llms_against_superficial_style_alignment.md)

</div>

<!-- RELATED:END -->
