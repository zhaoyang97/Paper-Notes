---
title: >-
  [论文解读] CaReBench: A Fine-grained Benchmark for Video Captioning and Retrieval
description: >-
  [ICLR2026][视频理解][视频字幕] CaReBench 用 1000 个人工标注、字幕长达 200+ 词且**显式拆成空间/时间两份**的视频，搭起一个能同时考视频细粒度字幕（captioning）和检索（retrieval）的 benchmark，配套两个新指标 ReBias 与 CapST 专门量化 VLM 的时空偏置，并顺手给出一个把字幕和检索统一进单个 MLLM 的两阶段 SFT 基线 CARE。
tags:
  - "ICLR2026"
  - "视频理解"
  - "视频字幕"
  - "视频检索"
  - "细粒度评测"
  - "时空偏置"
  - "多模态大模型"
---

# CaReBench: A Fine-grained Benchmark for Video Captioning and Retrieval

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=OZtGhb9x7C](https://openreview.net/forum?id=OZtGhb9x7C)  
**代码**: https://carebench.github.io （项目主页）  
**领域**: 视频理解  
**关键词**: 视频字幕, 视频检索, 细粒度评测, 时空偏置, 多模态大模型

## 一句话总结
CaReBench 用 1000 个人工标注、字幕长达 200+ 词且**显式拆成空间/时间两份**的视频，搭起一个能同时考视频细粒度字幕（captioning）和检索（retrieval）的 benchmark，配套两个新指标 ReBias 与 CapST 专门量化 VLM 的时空偏置，并顺手给出一个把字幕和检索统一进单个 MLLM 的两阶段 SFT 基线 CARE。

## 研究背景与动机
**领域现状**：视频字幕和视频检索是视频-语言理解的两大主任务，长期由两类模型分头把持——检索靠 CLIP 这类双编码器（dual-encoder）对齐特征，字幕靠多模态大模型（MLLM）逐句生成描述。两者本被当成互不相干的任务，各自有各自的 benchmark 和评测习惯。

**现有痛点**：现有 benchmark 撑不起"细粒度"评测。传统数据集（MSR-VTT、MSVD、DiDeMo）每条字幕只有一句话、十几个词，现代 MLLM 输出的描述比参考字幕还丰富，根本压不出模型的细粒度能力。近期想做长字幕的工作要么用 GPT-4o 自动标注（不可避免引入幻觉和偏置），要么像 DREAM-1K 只盯动作、缺乏对静态物体和动态动作的同时覆盖，也没有层次化结构。指标侧同样尴尬：传统 n-gram 指标（CIDEr）评不了长字幕；基于 LLM 的指标（AutoDQ 只评动作、VDCScore 只算 recall 不管 precision）都不够全面。

**核心矛盾**：视频理解本质要同时看懂**静态场景**和**动态动作**，但现有标注和指标都把这两者混在一起，导致一个被长期忽视的问题被掩盖——VLM 到底是真的理解了动作，还是靠场景线索"抄近路"？字幕和检索混在一句话里，谁也分不清模型的分数来自空间还是时间。

**本文目标**：(1) 造一个字幕足够细、且能把空间/时间解耦评测的 benchmark；(2) 设计能量化时空偏置的专用指标；(3) 验证检索和字幕能否统一进一个模型。

**切入角度**：作者发现检索和字幕其实是同一件事——都是把像素空间映射到一个高维空间 $\phi: \mathbb{R}^{T\times H\times W\times C}\to\mathbb{R}^{D}$，只不过字幕落到词表空间 $\mathbb{R}^{D_v}$、检索落到嵌入空间 $\mathbb{R}^{D_e}$。既然映射同源，就有可能用一个 MLLM 把两件事都干了。

**核心 idea**：用"层次化 + 空间/时间分离"的人工标注撑起细粒度评测，用 ReBias/CapST 把时空偏置量化出来，再用两阶段 SFT 把字幕和检索统一进一个 MLLM。

## 方法详解

### 整体框架
CaReBench 这篇工作有两条主线：**benchmark（数据 + 指标）** 和 **基线模型 CARE**。

benchmark 侧的流水线是：从 FineAction 动作定位数据集里手挑 1000 个视频（106 个子类、每类 10-20 个，同子类视频场景动作高度相似，专门考"区分相似视频"的能力）→ 两阶段人工标注（先写四方面的细粒度详描，再把描述拆成纯空间和纯时间两份）→ 基于这套空间/时间解耦的标注，定义检索指标 ReBias 和字幕指标 CapST。任务上派生出三种检索（general / spatial / temporal）和分物体/事件两类的字幕评测。

模型侧 CARE 建立在 Qwen2-VL 上，做**两阶段渐进式 SFT**：Stage-I 用高质量视频-字幕对把模型输出对齐到细粒度文本空间（练字幕能力）；Stage-II 冻住视觉编码器、用纯文本对比学习把模型输出从词表空间挪到嵌入空间（练检索能力）。两阶段做完，一个 7B 模型既能写细字幕又能抽视频特征。

### 关键设计

**1. 四方面层次化标注：让每条字幕"细到能压出模型"**

针对"现有字幕太短、压不出 MLLM 细粒度能力"的痛点，作者要求每个视频由两名标注员独立写、再由专家合并精修，字幕限定 150-300 词，强制覆盖四个方面：**General Overview**（一句话总览，如"一个人在切西瓜"）、**Object Description**（静态物体的位置/颜色/形状/相对关系，连水印都标）、**Action Description**（动作及其时序，"先…后…"，还要带动作风格如"快速切水果"）、**Misc Description**（视角、视频整体类型等 2-4 句）。最终字幕平均 227.95 词，是 MSR-VTT 的 24.2 倍。这种结构保证每条字幕既有静态物体又有动态动作，比 DREAM-1K（只盯动作）和 VDC（GPT 标、缺动作）都更全面，且全程人工避开了自动标注的幻觉。

**2. 空间/时间分离标注：把"抄近路"暴露出来**

这是整个 benchmark 最关键的设计。Stage-II 标注会把 Stage-I 的详描**手工拆成两份纯净字幕**：Spatial Description 去掉所有动作文本，只留总览 + 主次物体 + 背景，保证能区分同子类的相似视频；Temporal Description 去掉所有静态细节，只留总览 + 动作及其顺序。痛点在于：如果空间和时间信息混在一句话里，模型就算只看场景也能拿高分，没人能判断它到底懂不懂动作。拆开后，"空间检索"和"时间检索"分数的落差，就直接暴露了模型对动态动作的真实理解水平——实验里果然发现所有 VLM 在切到 temporal 时掉得惨烈，证明它们靠场景抄近路。

**3. ReBias：用一个比值量化检索的时空偏置**

针对"没人量化过 VLM 时空偏置"的空白，作者定义检索偏置指标 ReBias，衡量模型偏向静态物体还是动态动作。它看时间召回与空间召回之比偏离 1 的程度：

$$B = \left|\,1 - \frac{\bar{R}_{\text{temporal}}}{\bar{R}_{\text{spatial}}}\,\right|$$

其中 $\bar{R}_{\text{temporal}}$、$\bar{R}_{\text{spatial}}$ 分别是时间/空间检索的平均召回，**越低越好**（0 表示时空平衡）。这个指标的妙处是把"模型偏科"压成一个可比的标量——比如某模型空间召回很高但时间召回塌掉，ReBias 就会很大，一眼看出它在用空间捷径。

**4. CapST：同时评物体与事件、recall 和 precision 都管的字幕指标**

针对"现有字幕指标要么只评动作、要么只算 recall"的缺陷，CapST（Captioning, Spatial + Temporal）让一个强 LLM（实验用 DeepSeek-V3 当裁判）分别从时间字幕里抽**事件**、从空间字幕里抽**物体**，再用自然语言推理（NLI）判断真值描述 $D_{gt}$ 与预测描述 $D_{pred}$ 之间的蕴含关系，算 recall 和 precision：

$$R = \frac{N(D_{gt}\xrightarrow{\text{entail}}E_{pred})}{N(E_{pred})}, \qquad P = \frac{N(D_{pred}\xrightarrow{\text{entail}}E_{gt})}{N(E_{gt})}$$

$E_{pred}$、$E_{gt}$ 是从预测/真值里抽出的元素（物体或事件），$N(\cdot)$ 是元素计数，箭头表示被另一份描述蕴含的元素数。直觉上：recall 看"真值里的内容有多少被预测覆盖到"（漏没漏），precision 看"预测里的内容有多少被真值证实"（编没编，即幻觉）。同时管 recall 和 precision，比只算 recall 的 VDCScore 更能抓幻觉。一个额外细节：当一句描述捆了多个属性（"戴眼镜穿蓝西装的老人"），NLI 会惩罚部分匹配，所以作者让 LLM 抽取时**把属性拆开**（拆成"戴眼镜的老人"和"穿蓝西装的老人"），让部分正确的预测也能拿到应得的分。

### 损失函数 / 训练策略
CARE 的两阶段 SFT：

- **Stage-I（细粒度字幕适配）**：提示词固定为 "Describe the video in detail."，用 Tarsier Recap（动作丰富）+ LLaVA-Video-178k（短视频细节）混合数据全参微调，把输出对齐到细粒度文本空间。约 400 GPU 时，学习率 2e-5、batch 64、16 帧输入。
- **Stage-II（检索适配）**：从 Stage-I 初始化，**冻住视觉编码器只训 LLM**。用 EOL（Explicit One-word Limitation）提示词 "`<sent>` Summary of the above sentence in one word:" 取下一个 token 的隐状态当句子嵌入 $f_i$，在 NLI 数据集上做纯文本对比学习，把输出从词表空间挪到嵌入空间。因为只用文本、不带视频输入，只需 24 GPU 时。对比损失：

$$\mathcal{L} = -\log\frac{e^{\cos(f_i, f_i^+)/\tau}}{\sum_{j=1}^{N}\left(e^{\cos(f_i, f_j^+)/\tau} + e^{\cos(f_i, f_j^-)/\tau}\right)}$$

其中 $f_i, f_i^+, f_i^-$ 是句子 $s_i$ 及其正样本、难负样本的嵌入，$\cos(\cdot)$ 是余弦相似度，$\tau$ 是温度。值得注意的是 Stage-II 全程没有视频参与，却能让模型获得强视频检索能力——因为 Stage-I 已经把视频对齐到了细粒度文本空间，文本侧的对比学习能"借道"迁移到视频侧。

## 实验关键数据

### benchmark 统计对比
CaReBench 的字幕远比传统 benchmark 详细，且唯一同时具备层次化标注、静态物体、动态动作三者，又全程人工标注：

| Benchmark | 样本数 | 平均时长 | 平均词数 | 标注者 | 层次化 | 静态物体 | 动态动作 |
|--------|------|------|------|------|------|------|------|
| MSR-VTT | 1,000 | 15.01s | 9.41 | 人工 | ✗ | ✗ | ✗ |
| DREAM-1K | 1,000 | 8.9s | 59.3 | 人工 | ✗ | ✗ | ✓ |
| VDC | 1,000 | 28.18s | 500.91 | GPT | ✓ | ✓ | ✗ |
| **CaReBench** | 1,000 | 14.35s | 227.95 | 人工 | ✓ | ✓ | ✓ |

### 主实验：检索与字幕
General Retrieval（zero-shot，R@K 越高越好）上，对比学习后的 MLLM 整体反超 CLIP 系，CARE 在两个方向都拿到最佳或并列最佳：

| 模型 | T2V R@1 | T2V R@5 | V2T R@1 | V2T R@5 |
|------|------|------|------|------|
| CLIP L/14 | 51.2 | 83.4 | 54.7 | 86.9 |
| InternVideo2 1B | 72.5 | 93.7 | 69.5 | 94.6 |
| Qwen2-VL 7B（对比训练后） | 76.6 | 95.3 | 77.4 | 95.6 |
| **CARE 7B** | **77.0** | **95.6** | **79.0** | **96.8** |

字幕侧（CapST，DeepSeek-V3 当裁判），CARE 7B 在 Events 和 Objects 上整体超过所有开源模型，连 Qwen2-VL 72B、Qwen2.5-VL 7B 都被压住（如 Events Overall F1：CARE 35.1 vs Qwen2.5-VL 7B 31.1 vs Qwen2-VL 72B 30.5），说明现有模型即便堆到 72B 也仍缺细粒度描述能力。

### 消融实验：两阶段 SFT 的作用
"Unified Score"是检索 Avg. R@1 与字幕 Avg. F1 的平均：

| 配置 | Avg. R@1（检索） | Avg. F1（字幕） | Unified Score |
|------|------|------|------|
| Baseline (Qwen2-VL) | 25.6 | 26.8 | 26.2 |
| + 仅细粒度字幕适配 | 17.6 (−8.0) | 33.8 (+7.0) | 25.7 (−0.5) |
| + 仅检索适配 | 77.0 (+51.4) | 28.2 (+1.4) | 52.6 (+26.4) |
| + 两阶段全做 | **78.0 (+52.4)** | **33.4 (+6.6)** | **55.7 (+29.5)** |

### 关键发现
- **VLM 普遍靠空间捷径抄近路**：从 general 切到 spatial 检索，VLM 几乎不掉分（Qwen2-VL −0.20、Tarsier −2.00）；切到 temporal 却暴跌（Qwen2-VL −24.70、Tarsier −20.75、MiniCPM-V −21.85）。即便动作线索被从字幕里抹掉，模型分数依旧稳，说明它们靠场景线索而非动作信息理解视频——这是 ReBias/空间-时间分离设计才能暴露出来的偏置。
- **检索和字幕能互相增益**：检索适配让 baseline 的字幕 F1 从 26.8 涨到 28.2（+1.4），而字幕适配又把检索后的模型 R@1 从 77.0 推到 78.0（+1）——两个任务确实同源、能彼此促进，支持了"统一框架"的假设。
- **字幕不受 Stage-II 影响**：做不做检索适配几乎不改变字幕性能，说明嵌入空间的对比学习没有损伤词表空间的生成能力。
- **泛化性**：CARE 在 out-of-domain 的 MVBench（60.4）、TVBench（50.1）上同样有竞争力，TVBench 甚至超过 Gemini 1.5 Pro（46.5）。

## 亮点与洞察
- **"空间/时间分离标注"是把隐性问题显性化的杠杆**：很多 benchmark 加难度靠堆数据，CaReBench 靠把字幕物理拆成两份，让"模型靠场景抄近路"这个一直存在却测不出来的现象第一次被量化——这是数据设计驱动洞察的典范，可迁移到任何需要解耦评测某两种能力的场景。
- **检索 = 字幕的统一视角很优雅**：把两任务都写成 $\phi:\mathbb{R}^{T\times H\times W\times C}\to\mathbb{R}^D$（词表空间 vs 嵌入空间）的映射，一句话点破了为什么一个 MLLM 能两头通吃，理论上干净。
- **Stage-II 纯文本对比却能迁移到视频**：冻住视觉塔、只用 NLI 文本数据训 24 GPU 时就拿到 SOTA 检索，本质是借 Stage-I 已建立的视频↔细粒度文本对齐做"免费午餐"，这个 trick 对想低成本给 MLLM 加检索能力的人很有参考价值。
- **CapST 拆属性的小细节很实在**：把"戴眼镜穿蓝西装的老人"拆成两条再做 NLI，避免部分正确被一刀切惩罚，这种对评测公平性的打磨往往被忽视。

## 局限与展望
- 作者明确承认：CARE 只是揭示并量化了 VLM 的时空偏置，**并没有解决**这个偏置——CARE 自己的 ReBias 仍高达 17.53，和主流 MLLM 处在同一水平，没有真的变得更"时空平衡"。
- benchmark 规模 1000 视频、聚焦 5-20 秒短视频（>80% 在此区间），长视频的细粒度理解未覆盖；且全来自 FineAction 的 4 大类日常动作，领域偏窄。
- CapST/ReBias 重度依赖 LLM 裁判（DeepSeek-V3）抽取元素和做 NLI，裁判本身的能力和偏好会影响打分稳定性；人工对齐验证放在附录，正文不易核查。
- 改进方向：设计真正能降低 ReBias 的训练目标（如显式的时间对比/动作监督），把统一框架扩到长视频和更多视频类型。

## 相关工作与启发
- **vs DREAM-1K / VDC（细粒度字幕 benchmark）**：DREAM-1K 人工标但只盯动作、无层次化；VDC 有层次化但用 GPT 标、缺动作覆盖。CaReBench 是唯一同时做到人工标注 + 层次化 + 物体与动作双覆盖 + 空间/时间分离的，且专门服务于时空偏置分析。
- **vs Long-CLIP（长文本视频检索）**：Long-CLIP 把上下文从 77 扩到 248 token 来支持长字幕检索，但它用的 benchmark 由 LLM 标、含粗粒度和错误描述。CaReBench 用人工细标，并发现对比训练后的 MLLM 在检索上整体反超 CLIP 系。
- **vs E5-V / VISTA（统一多模态表示）**：这些工作发现 MLLM 能无 gap 地统一跨模态表示。CARE 把这一思路用 EOL 提示 + 两阶段 SFT 落到视频检索 + 字幕的统一上，并实证两任务可互相增益。

## 评分
- 新颖性: ⭐⭐⭐⭐ 空间/时间分离标注 + ReBias/CapST 把视频时空偏置量化出来，benchmark 设计有真洞察；统一框架属优雅但非颠覆。
- 实验充分度: ⭐⭐⭐⭐ 检索/字幕/QA 多任务、十余个 baseline、消融清晰，但 benchmark 偏短视频单一来源。
- 写作质量: ⭐⭐⭐⭐ 动机—数据—指标—模型逻辑顺，指标公式交代清楚。
- 价值: ⭐⭐⭐⭐ benchmark + 指标 + 统一基线一套打包开源，对评测 VLM 细粒度时空理解很实用。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] FineVAU: A Novel Human-Aligned Benchmark for Fine-Grained Video Anomaly Understanding](../../AAAI2026/video_understanding/finevau_a_novel_human-aligned_benchmark_for_fine-grained_video_anomaly_understan.md)
- [\[ICML 2025\] Fine-Grained Captioning of Long Videos through Scene Graph Consolidation](../../ICML2025/video_understanding/fine-grained_captioning_of_long_videos_through_scene_graph_consolidation.md)
- [\[ICLR 2026\] OmniCVR: A Benchmark for Omni-Composed Video Retrieval with Vision, Audio, and Text](omnicvr_a_benchmark_for_omni-composed_video_retrieval_with_vision_audio_and_text.md)
- [\[ICLR 2026\] Let's Split Up: Zero-Shot Classifier Edits for Fine-Grained Video Understanding](lets_split_up_zero-shot_classifier_edits_for_fine-grained_video_understanding.md)
- [\[CVPR 2026\] Fine-VAD: Towards Fine-Grained Video Anomaly Detection via Progressive Cross-Granularity Learning](../../CVPR2026/video_understanding/fine-vad_towards_fine-grained_video_anomaly_detection_via_progressive_cross-gran.md)

</div>

<!-- RELATED:END -->
