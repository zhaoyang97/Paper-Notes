---
title: >-
  [论文解读] CNSL-bench: Benchmarking the Sign Language Understanding Capabilities of MLLMs on Chinese National Sign Language
description: >-
  [ACL 2026][多模态VLM][中国国家手语] CNSL-bench 是首个基于《国家通用手语词典》的权威中国手语 MLLM 评测基准，覆盖 6,707 个唯一手语词条 × 文本/图片/视频三模态 × 三种手部 articulation（空写/指拼/手指字母）共 20,121 道四选一题，在 21 个 SOTA MLLM 上揭示：GPT-5 文本 89.6%、图片 67.0%、视频 56.7%，相对人类 97% 仍有巨大 gap，且 CoT 推理对视频帮助微弱。
tags:
  - "ACL 2026"
  - "多模态VLM"
  - "中国国家手语"
  - "手语基准"
  - "MLLM 评测"
  - "模态失衡"
  - "manual articulation"
---

# CNSL-bench: Benchmarking the Sign Language Understanding Capabilities of MLLMs on Chinese National Sign Language

**会议**: ACL 2026  
**arXiv**: [2604.22367](https://arxiv.org/abs/2604.22367)  
**代码**: https://github.com/rzhao-zhsq/CNSL-bench  
**领域**: 多模态 VLM / 手语理解  
**关键词**: 中国国家手语, 手语基准, MLLM 评测, 模态失衡, manual articulation

## 一句话总结
CNSL-bench 是首个基于《国家通用手语词典》的权威中国手语 MLLM 评测基准，覆盖 6,707 个唯一手语词条 × 文本/图片/视频三模态 × 三种手部 articulation（空写/指拼/手指字母）共 20,121 道四选一题，在 21 个 SOTA MLLM 上揭示：GPT-5 文本 89.6%、图片 67.0%、视频 56.7%，相对人类 97% 仍有巨大 gap，且 CoT 推理对视频帮助微弱。

## 研究背景与动机

**领域现状**：LLM 推动手语研究从纯 SLR/SLT 流水线进入 LLM-as-decoder 阶段（Sign2GPT、SignLLM 等），近期 MLLM 又把视觉/视频理解能力推得很强，但几乎所有工作都是把 LLM 嵌进特定下游任务（手语翻译、识别），把它当成 semantic enhancement 模块。

**现有痛点**：**MLLM 本身的 intrinsic 手语理解能力从未被系统评测过**。现有手语数据集（WLASL、PHOENIX、CSL-Daily、How2Sign 等）都是给训练特定任务用的，没有跨模态对齐的 lexical 级评测；而通用 MLLM 基准（MME、MMMU、Video-MME 等）压根不含手语。结果是：我们不知道 MLLM 在手语上"到底强在哪、弱在哪、modality gap 多大"。

**核心矛盾**：手语本质是**多模态语言**（视觉空间 + 时间动力学 + 语言学结构），既需要 VLM 的视觉感知又需要 LLM 的语义理解；但既有评测要么只测视觉（不带语义 grounding），要么只测语言（不带视觉），无法回答"MLLM 是真的理解手语 linguistic structure，还是只在做表面视觉相关"。

**本文目标**：构建首个 **(1) 权威 lexical grounding、(2) 多模态对齐（text/image/video）、(3) articulation 多样性（air-writing/finger-spelling/manual-alphabet）** 的手语 MLLM 评测基准，并系统评测 21 个 SOTA MLLM。

**切入角度**：从国家级标准词典出发——《国家通用手语词典》（教育部 + 国家语委 + 中国残联联合发布）是中国唯一的官方手语标准，由它锚定可以消除方言/非规范变体歧义，得到**可控、一致、可复现**的语义参考；再把每个词条与 CNSL-DP（厦大 2025 双视角手语视频数据集）的视频对齐，得到文本/图片/视频三模态完全对齐的 lexical-level 基准。

**核心 idea**：用"权威词典 lexical grounding + 多模态对齐 + manual articulation 细分"三原则构造 4-选-1 选择题基准，把开放式手语理解（当前 MLLM 完全做不了）转化为 closed-form 可控评测，并用 21 个 MLLM × 3 模态 × 3 articulation × 2 视频帧率的密集网格揭示 MLLM 在 sign language understanding 上的系统性失败模式。

## 方法详解

### 整体框架
CNSL-bench 构建分两条主线：

- **Lexical grounding**：从《国家通用手语词典》8,214 条 gloss 出发，经过 sign-level 预处理（合并相同手部动作的不同 gloss、拆分同 gloss 不同含义/动作的多义条目、保留同义不同动作的多变体），得到 6,707 unique sign entries。
- **多模态对齐**：每个 sign entry 三模态对齐——(1) 词典原始的文字描述；(2) 词典插图；(3) 从 CNSL-DP（Jin et al. 2025）选一段代表性视频（24 fps、512×512 中心裁剪、签字者居中）。
- **Articulation 子集**：人工标注 407 条 air-writing、77 条 finger-spelling、592 条 manual-alphabet（基于《中国手指字母方案》），作为 dedicated subset 做细粒度分析。
- **任务形式**：4-way multiple-choice，每题给定一个模态输入 + 一个正确答案 + 3 个随机干扰项（论文也对比了 semantic distractor，结论一致但 random 更易复现）。
- **规模**：6,707 entries × 3 模态 = 20,121 题。
- **评测**：对 21 个 MLLM（含开源 LLaVA-NeXT/Qwen-VL/InternVL-3.5/GLM-4.1V 和闭源 Qwen-Plus/Max、Gemini-2.5、GPT-4o/5）在文本/图片/视频 (2 fps & 10 fps) × AW/FS/MA/All 上跑 zero-shot，并对支持推理的模型测试 fast/slow thinking。

### 关键设计

**1. 权威词典 lexical grounding + sign-level 去重对齐：让每道题都有唯一标准答案**

手语评测最大的噪声来自"同一个意思有多种地方/方言写法"，model 答对了反被判错、或答了某变体被算成错——基准根本不可复现。CNSL-bench 把 lexical 真相源锚定在中国唯一的国家级手语标准上（教育部 2018《国家通用手语常用词表》+ 中国残联 2019《国家通用手语词典》），从源头消除变体歧义。在此之上做 sign-level 预处理，专治三种 redundancy：不同 gloss 共享同一手势的合并成一条；同一 gloss 对应多义（如"安全带"在汽车和飞机语境下是不同手势）的拆成多条；同一含义有多个手势变体的则全部保留。8,214 条 gloss 经此整理为 6,707 unique entries。

视频侧同样严格对齐——CNSL-DP 数据集为每个 sign 提供多 signer 录像，本文挑一个 representative，并补回 CNSL-DP 因合并同义词而遗漏的变体，保证 video 与词典逐条一一对应。正是这套"词典 grounding + 三模态严格对齐"让 CNSL-bench 摆脱了 WLASL、How2Sign 等无标准化基准的 lexical ambiguity，成为可控、可复现的 lexical 级评测。

**2. Manual articulation 三分类细粒度分析（AW / FS / MA）：定位 MLLM 强在哪种手部动作、弱在哪种**

只报一个 overall accuracy 看不出模型到底卡在哪。本文按手语语言学把 articulation 拆成三类：**Air-writing (AW)** 是在空中绘制图形/笔画（如"避雷针"画出 ϟ 形），考验 spatial trajectory tracking；**Finger-spelling (FS)** 用单手或双手 depict 汉字字形（如双手交叉摆出"北"字），强调 graphic cue 而非逐字母拼写；**Manual-alphabet (MA)** 则按《中国手指字母方案》把手势映射到拼音字母（如"二氧化碳"用 C+O+2 的字母手势组合），需要符号识别加组合理解。人工从 6,707 条里标出 407 AW / 77 FS / 592 MA 作为独立 subset，每类都在 text/image/video 三模态下单独报准确率。

这种细分立刻暴露出难度的巨大落差：所有模型在 FS 上都最好（GPT-5 FS 文本 97.4%、视频 53.3%），AW 和 MA 明显更差。原因是 FS 更"离散、字符化"，接近 MLLM 早已熟悉的 OCR 任务；而 AW 是连续 spatial trajectory、MA 是符号+组合，都依赖 MLLM 不具备的 sign-specific reasoning。比起单一 accuracy，这种诊断能明确指出改进方向。

**3. 三模态对齐评测 + modality gap 量化：探明 MLLM 是真懂手语语义，还是靠语言先验蒙**

把同一个 sign entry 在文本、图片、视频三个模态下分别测，就能量化模型的"模态依赖偏倚"——同一份语义内容，换个模态表现会掉多少。评测协议规定每道 4-way MCQ 只用一种模态作输入，其余模态的对齐仅用于排除"跨模态偷懒"；视频还额外测 2 fps 与 10 fps 两档帧率以研究 temporal density 的影响。同时邀请 1 位手语语言学教授 + 3 名手语专业学生（含 1 名 Deaf）做 human baseline（每人每 2 小时 \$30）。

结果一目了然：GPT-5 从 text 89.6% → image 67.0% → video 56.7%，狂掉约 33 个百分点；人类则是 text 96.9% → image 97.4% → video 97.4%，几乎纹丝不动。这说明 modality gap 是 MLLM 独有的毛病——模型主要靠语言先验答题，视觉/时序理解远未达标。这一探针对训练（需要更多手语—视频对齐数据）和评测设计（必须包含视频）都有直接指导意义。

### 损失函数 / 训练策略
本文是评测基准，无训练。评测端固定 temperature=0、max_tokens=2048；reasoning 模型额外测 low/medium/high reasoning effort 三档。人类 baseline 通过专家团队完成。

## 实验关键数据

### 主实验（21 MLLM × 文本/图片/视频 × AW/FS/MA/All，accuracy %）

| Model | Text-All | Image-All | Video 2fps-All | Video 10fps-All | FS-Text | FS-Video10 |
|-------|----------|-----------|-----------------|------------------|---------|------------|
| **GPT-5 (M, slow)** | **89.64** | **66.96** | **53.42** | **56.72** | **97.40** | **53.25** |
| Gemini-2.5-Pro (M, slow) | 84.79 | 61.13 | 48.32 | 48.35 | 94.81 | 35.06 |
| Gemini-2.5-Flash (slow) | 79.95 | 51.62 | 42.28 | 42.63 | 93.51 | 32.47 |
| Qwen3-VL-Plus (slow) | 76.22 | 42.41 | 35.34 | 36.92 | 89.61 | 18.18 |
| GPT-4o | 69.03 | 39.07 | 31.26 | 28.43 | 88.31 | 23.38 |
| InternVL-3.5-8B | 67.53 | 38.36 | 32.26 | 33.59 | 83.12 | 36.36 |
| Qwen3-VL-8B-Instruct | 67.06 | 38.39 | 30.94 | 33.89 | 79.22 | 24.68 |
| GLM-4.1V-9B (slow) | 68.24 | 39.62 | 28.03 | 29.75 | 84.42 | 24.68 |
| Qwen2.5-VL-3B | 60.07 | 34.26 | 28.36 | 30.34 | 72.73 | 28.57 |
| Qwen2-VL-2B | 43.36 | 30.62 | 27.23 | 27.58 | 51.95 | 18.18 |
| LLaVA-NeXT-Video-7B | 1.34 | 12.94 | 15.43 | 15.91 | 2.60 | 14.29 |
| **Random** | 25.23 | 24.73 | 25.03 | 25.04 | 27.27 | 24.67 |
| **Human** | **96.93** | **97.39** | **97.39** | **97.39** | **97.40** | **98.70** |

### 推理消融（test-time scaling，Reasoning effort L/M/H）

| Model | Text-All | Image-All | Video-All | 说明 |
|-------|----------|-----------|-----------|------|
| GPT-5 (L) | 88.94 | 66.77 | 51.89 | 低推理 |
| GPT-5 (M) | **89.64** | 66.96 | **53.42** | 中推理（最强） |
| GPT-5 (H) | 89.95 | **68.34** | 53.09 | 高推理 (text 略升、video 略降) |
| Gemini-2.5-Pro (L) | 81.32 | 58.09 | 48.83 | |
| Gemini-2.5-Pro (M) | 84.79 | 61.13 | 48.32 | |
| Gemini-2.5-Pro (H) | 84.84 | 61.92 | 48.17 | 视频上 high 反而降 |
| Gemini-2.5-Flash fast | 73.04 | 43.57 | 36.62 | |
| Gemini-2.5-Flash slow | **79.95** | **51.62** | **42.28** | 平均增益最大 +6.45% |
| Qwen3-VL-Plus fast | 76.68 | 43.69 | 33.74 | |
| Qwen3-VL-Plus slow | 76.22 | 42.41 | 35.34 | **slow thinking 反而掉点** |

### 关键发现
- **MLLM 显著落后人类**：最强 GPT-5 在 video 模态只有 56.7%，人类 97.4%，gap ~41 个百分点。即便文本模态 GPT-5 也只 89.6% vs 人类 96.9%。
- **强 modality 失衡**：所有模型一致呈现 text >> image > video 的 drop。GPT-5 从文本 89.6% 到视频 56.7% 掉了 33 点，开源模型掉幅更大。说明 MLLM 的 "multimodal alignment" 远未完成，主要靠语言先验。
- **Articulation 不均衡**：FS（finger-spelling）一致最容易（GPT-5 FS-text 97.4%），AW 和 MA 显著难。提示 MLLM 对 "离散、字符化" 手势相对擅长，对连续 spatial trajectory + 符号组合无能。
- **CoT 几乎对 video 无效，对 top-tier 模型有 boundary effect**：Gemini-2.5-Pro 与 GPT-5 从 low → high reasoning，视频 accuracy 几乎不变甚至下降；Qwen3-VL-Plus slow thinking 反而比 fast 掉点。表明手语视频理解的瓶颈在视觉感知层而非推理层。
- **推理 token 严重 modality-biased**：模型在文本上消耗推理 token 远多于图片/视频，反映出 MLLM "习惯用文本思考"，对视觉模态没有充分的内化推理流程。
- **错误答案的推理链更长**：GPT-5-M 在文本上 incorrect/correct 推理长度比 2.89，与人类"难题想得久"行为一致——但这种"想得久"对 video 不带来 accuracy 提升，进一步证实是感知瓶颈。
- **开源 vs 闭源 gap 在缩小**：GLM-4.1V-9B、InternVL-3.5-8B、Qwen3-VL-8B 在多个 subset 上接近甚至超过 GPT-4o-mini / Gemini-2.5-Flash。

## 亮点与洞察
- **首次把 MLLM 的"sign language understanding"形式化为可控的 lexical-level MCQ 评测**：避开了开放式手语生成（当前 MLLM 几乎完全失败，论文 Figure 9 案例显示连 "今天天气不冷不热" 这种简单句子 Gemini-3-Pro / GPT-5.1 都答不对），用 closed-form 让评测变得 reproducible，是手语 MLLM 评测的范式起点。
- **三原则 + 三模态 + 三 articulation 的多维设计**：让基准不仅给 overall accuracy，还能拆出 "modality gap"、"articulation gap"、"reasoning effect"，对 MLLM 改进方向有明确指引——下一步要解决的是 visual perception 不是 reasoning。
- **国家级标准词典作为 lexical truth**：这种"用国家/官方词典做 lexical grounding"的范式对其他低资源语言、专业领域（法律、医学）的 MLLM 评测都有借鉴价值——核心是"用权威 source 消除标准歧义"。
- **CoT 失效 + boundary effect 揭示新方向**：test-time scaling 在 video 输入上几乎无效，说明当前 MLLM 的"多模态推理"主要还是 text reasoning + visual perception 串联——视觉感知的瓶颈无法靠 thinking-time 解决，必须从 vision encoder/training data 层面改进。
- **Modality token consumption 视角**：第一次量化"模型用多少 token 在不同模态上推理"，发现强烈 text bias，给"多模态推理对齐"训练目标提供新维度。

## 局限与展望
- **作者承认**：(1) 只测 lexical-level（单个手语词），不测开放式句子级 SLT（因为 MLLM 此层面尚未可靠）；(2) 只覆盖中国国家手语，未涉及 ASL/PSE/方言变体；(3) MCQ 形式不能完全 capture 开放式理解能力。
- **额外局限**：(1) Random distractor 让题目偏简单，可能高估模型——论文也承认 semantic distractor 更难但结论一致；(2) 视频只测 2 fps 和 10 fps 两档，未系统研究 frame rate × accuracy 曲线；(3) 没测 long video / continuous signing（连续手语序列），现实场景仍未覆盖；(4) 人类 baseline 只 4 人，规模偏小。
- **改进思路**：(1) 拓展到 sentence-level open-ended SLT 评测（待 MLLM 能力提升后）；(2) 加入跨语言对比（ASL、BSL、JSL）；(3) 用 CNSL-bench 数据做 sign-language-specific instruction tuning 验证是否能缩小 modality gap；(4) 研究 vision encoder 端的改进（如 hand-region 注意力、temporal motion encoder）。

## 相关工作与启发
- **vs WLASL / MS-ASL / PHOENIX / How2Sign / CSL-Daily**：它们是手语识别/翻译的训练数据，没有 MLLM-friendly 的多模态对齐评测协议；CNSL-bench 是首个 dedicated MLLM 评测基准。
- **vs MME / MMMU / Video-MME / SEED-Bench**：通用 MLLM 基准侧重 everyday objects/scenes/events，几乎不含手语；CNSL-bench 填补了 sign language 这一专业 modality 的 MLLM 评测空白。
- **vs Sign2GPT / SignLLM / FLa-LLM**：它们把 LLM 作为 SLT pipeline 的 decoder/enhancer，本文则评测 MLLM 的 intrinsic 手语理解，是上游的 capability assessment。
- **vs PRACTIQ / BIRD-INTERACT 等专业基准范式**：与本笔记同 batch 的 CLARITY 一样，CNSL-bench 也走 "权威 source + 自动构造 + 多维细粒度评测 + 人类基线" 路线，是当代专业基准设计的代表性范例。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个中国国家手语 MLLM 评测基准，权威词典 + 多模态对齐 + articulation 细分三原则原创
- 实验充分度: ⭐⭐⭐⭐⭐ 21 个 MLLM × 3 模态 × 3 articulation × 2 帧率 + reasoning 消融 + 人类基线 + case study + prompt token / instruction-following 多维诊断
- 写作质量: ⭐⭐⭐⭐ pipeline 清晰、findings 条理化、case study 直观；可视化（雷达图、KDE）有效
- 价值: ⭐⭐⭐⭐⭐ 对 MLLM 训练、手语 AI for Deaf community、低资源语言专业评测都有直接推动价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Hybrid Autoregressive-Diffusion Model for Real-Time Sign Language Production](hybrid_autoregressive-diffusion_model_for_real-time_sign_language_production.md)
- [\[ACL 2026\] CArtBench: Evaluating Vision-Language Models on Chinese Art Understanding, Interpretation, and Authenticity](cartbench_evaluating_vision-language_models_on_chinese_art_understanding_interpr.md)
- [\[ACL 2026\] GroupToM-Bench: Benchmarking Group Theory of Mind and Nonlinear Social Emergence in MLLMs](grouptom-bench_benchmarking_group_theory_of_mind_and_nonlinear_social_emergence_.md)
- [\[ACL 2026\] VULCA-Bench: A Multicultural Vision-Language Benchmark for Evaluating Cultural Understanding](vulca-bench_a_multicultural_vision-language_benchmark_for_evaluating_cultural_un.md)
- [\[ACL 2026\] AICA-Bench: Holistically Examining the Capabilities of VLMs in Affective Image Content Analysis](aica-bench_holistically_examining_the_capabilities_of_vlms_in_affective_image_co.md)

</div>

<!-- RELATED:END -->
