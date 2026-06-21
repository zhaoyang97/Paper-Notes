---
title: >-
  [论文解读] Seq vs Seq: An Open Suite of Paired Encoders and Decoders
description: >-
  [ICLR 2026][预训练][encoder-only] 作者训练了一套从 17M 到 1B、配对的 encoder-only 与 decoder-only 模型（ETTIN suite），二者用**完全相同**的数据、架构和训练配方，只差「目标函数 + 注意力方向」；在公平对比下既各自刷到同尺寸开放数据 SOTA，又证明：分类/检索任务上 encoder 碾压 decoder，生成任务反之，而且**靠继续训练把一种模型改造成另一种（cross-objective）始终补不平这个差距**。
tags:
  - "ICLR 2026"
  - "预训练"
  - "encoder-only"
  - "decoder-only"
  - "配对模型"
  - "MLM vs CLM"
  - "跨目标训练"
---

# Seq vs Seq: An Open Suite of Paired Encoders and Decoders

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=z5Mn8Rxi3l](https://openreview.net/forum?id=z5Mn8Rxi3l)  
**代码**: https://github.com/JHU-CLSP/ettin-encoder-vs-decoder (有)  
**领域**: LLM 预训练 / 表示学习  
**关键词**: encoder-only、decoder-only、配对模型、MLM vs CLM、跨目标训练

## 一句话总结
作者训练了一套从 17M 到 1B、配对的 encoder-only 与 decoder-only 模型（ETTIN suite），二者用**完全相同**的数据、架构和训练配方，只差「目标函数 + 注意力方向」；在公平对比下既各自刷到同尺寸开放数据 SOTA，又证明：分类/检索任务上 encoder 碾压 decoder，生成任务反之，而且**靠继续训练把一种模型改造成另一种（cross-objective）始终补不平这个差距**。

## 研究背景与动机

**领域现状**：LLM 社区几乎只关注 decoder-only（GPT 式）模型，因为它们做文本生成天然方便。但仍有一大批人在用 encoder-only（BERT 式）模型做分类、检索、嵌入这类不需要生成的任务，且因为 encoder 长期缺乏新模型迭代，很多人至今还在用 2019 年的老 BERT。

**现有痛点**：一种流行观点是「decoder 既然更大、训得更久、还能 zero-shot，那它顺手就能接管 encoder 的活，没必要再单独训 encoder」——MTEB 等检索榜单的头部如今确实被 7B+ 的 decoder（或 decoder 继续训练得到的嵌入模型）占据。但这个结论缺乏干净的实验支撑。

**核心矛盾**：过去所有「encoder vs decoder」的对比都是**苹果比橘子**——被比较的两个模型参数量不同、架构不同、预训练数据不同、学习率调度也不同。少数试图控制变量的工作又只在很小的数据规模上做（如 100B token），结论可能只是小数据下 CLM 数据效率更高的假象。于是「到底是架构/目标本身带来差异，还是训练细节带来差异」始终说不清。

**本文目标**：(1) 造一套**唯一变量是训练目标**的配对模型，让 encoder 和 decoder 真正可比；(2) 量化两类目标各自的强弱项以及参数规模缩放的影响；(3) 回答「把 decoder 继续训练成 encoder（或反过来）到底值不值」。

**切入角度**：借鉴 Pythia「开放数据 + 多尺寸 + 全 checkpoint」的思路，但把它扩展到**成对的两种架构**，并且要求模型本身达到 SOTA——只有当两边都是同尺寸最强模型时，对比才有说服力，也才能排除「训练选择偏向了某一方」的质疑。

**核心 idea**：用同一套数据、同一套架构、同一套配方训练 encoder 和 decoder，**只让目标函数（MLM vs CLM）和注意力（双向 vs 因果）两处不同**，从而把架构差异从训练噪声里干净地剥离出来。

## 方法详解

这篇是「开放模型套件 + 受控对比实证研究」，没有复杂的算法 pipeline，核心是**实验设计本身**：如何把变量控制干净、如何复刻一个强配方、如何设计跨目标对照。下面分整体框架和关键设计讲清楚。

### 整体框架

ETTIN suite 一共 10 个模型，构成 5 对（每对一个 encoder、一个 decoder），尺寸从 17M、32M、68M、150M、400M 到 1B，最多训 2T token。整套流程是：先**固定一份开放数据 + 一份模型架构表 + 一套三阶段训练配方**，然后对每个尺寸各训一个 encoder（MLM + 双向注意力）和一个 decoder（CLM + 因果注意力），二者除目标函数和注意力外**逐字相同**（同一份数据、同样的层数/宽度、同样的学习率调度）。训完后，在 encoder 任务（GLUE、MTEB 检索、长上下文、代码检索）和 decoder 任务（ARC、HellaSwag、TriviaQA 等 zero-shot 生成评测）上**交叉评估**——让 decoder 去做分类、让 encoder 去做生成。最后再做一组**跨目标继续训练**：把训好的 decoder 用 MLM 类目标续训成 "encoder-from-decoder"、把 encoder 用 CLM 续训成 "decoder-from-encoder"，看续训能否补上架构差距。整个过程每 8.5B token 存一个 checkpoint（每个模型 236 个），并连同 batch 顺序一起开源。

### 关键设计

**1. 配对受控：把唯一变量锁定在「目标函数 + 注意力」**

这是全文的方法地基，直接针对「过去对比都是苹果比橘子」的痛点。作者让同一对 encoder/decoder 共享一份训练数据、一张架构配置表（同样的层数、隐藏维度、中间层维度、注意力头数、学习率、weight decay、warmup token 数，见原文 Table 1）、同样的三阶段调度，**两者仅有的差别是**：目标函数（encoder 用 MLM、decoder 用 CLM）和注意力模式（encoder 双向、decoder 因果）。这样一来，下游任务上观察到的任何差距都只能归因于这两个因素，而非训练细节。这种「成对、逐字对齐」的设计是过去工作做不到的——它们要么用现成的不可比模型，要么只在玩具规模上控制变量；ETTIN 把控制变量做到了 SOTA 规模，对比才第一次可信。

**2. 开放数据复刻 ModernBERT 配方 + 三阶段训练**

作者选择复刻 ModernBERT（当时最强的公开 encoder）作为跨两种目标的起点配方，但 ModernBERT 的数据不公开，于是改用 Olmo 系列的公开数据（DCLM + Dolma v1.7 的精选源）拼出一份**全开放**的替代。训练分三个阶段、配梯形（trapezoidal）学习率：① 基础预训练（base pre-training），用宽口径混合数据训 1.7T token，含学习率和 batch size 双 warmup；② 中段训练 / 上下文扩展（mid-training），把数据长度提到 8000、RoPE base 调到 160k 以支持长上下文，并换上更高质量的 Dolmino 数据，训 250B token，用 inverse-sqrt 调度降到峰值的一半；③ 衰减阶段（decay），按 ProLong 配方加入长文档（书籍、Wikipedia、教科书），再训 50B token 把学习率衰减到峰值的 0.02。相比 ModernBERT，主要差别是用开放数据、在上下文扩展阶段加衰减、不做模型 merge、衰减阶段把 MLM masking 比例从 30% 降到 15%、以及把 local/global RoPE 设成同值。这套配方的价值不只是训出 ETTIN，更是给社区提供了**第一个 ModernBERT 式模型的可复现公开配方**。

**3. 跨目标继续训练：直接检验「decoder 能否顶替 encoder」**

有了配对模型，作者就能做过去做不到的对照——把成品模型用**反向目标**继续预训练，分两个方向：decoder 续训成 encoder（encoder-from-decoder，沿用 LLM2Vec 的 MNTP 目标，即用前一个 token 的隐状态预测被 mask 的 token，以更贴合 CLM 的因果结构）；encoder 续训成 decoder（decoder-from-encoder，用 CLM）。续训用 50B token，远多于 LLM2Vec 的约 10B，且用的是质量最高的 decay 阶段数据，配新的梯形调度（3B warmup + 10B decay）。这个设计的关键在于：它把「现在很多人把大 decoder 改造成嵌入/分类模型」这一现实做法直接搬进受控实验里去测，从而能定量回答「续训补差距」到底有没有用。

**4. 全工件开源：让套件成为可复用的分析平台**

作者开源了全部工件——训练数据、按 checkpoint 切分的训练数据顺序（batch order）、以及每个模型 200+ 个 checkpoint。这不是锦上添花，而是让 ETTIN 从「一组模型」变成「一个研究平台」：后人可以像 Pythia 那样研究模型在训练过程中**何时**学会了什么、数据顺序如何影响学习、两种目标在性别偏见等维度上如何分化（论文自己就用 WinoGender 做了一个性别偏见的 case study）。开放 batch order 是其中最有分量的一点，它让「某两个相邻 checkpoint 之间模型学到了什么」这种精细分析成为可能。

### 损失函数 / 训练策略
encoder 用 MLM（预训练 masking 比例 30%、decay 阶段降到 15%）+ 双向注意力；decoder 用 CLM + 因果注意力。两者共用梯形学习率调度（warmup → 稳定 → inverse-sqrt 衰减）、学习率和 batch size 双 warmup。跨目标续训中 encoder-from-decoder 用 MNTP（masking 比例 15%）、decoder-from-encoder 用 CLM。

## 实验关键数据

### 主实验

**Encoder 侧（部分代表任务，GLUE Avg / MTEB 检索等）**：ETTIN encoder 在各尺寸普遍打平或超过对应 baseline，包括最新的 ModernBERT。

| 尺寸 | 模型 | CodeSearchNet | MTEB Retrieval | GLUE Avg |
|------|------|------|------|------|
| Base (~150M) | ModernBERT base | 75.9 | 43.9 | 88.4 |
| Base (~150M) | **Ettin-Enc-150m** | 76.3 | 45.7 | **88.9** |
| Large (~400M) | ModernBERT large | 78.3 | 47.0 | 90.4 |
| Large (~400M) | **Ettin-Enc-400m** | 80.7 | 48.4 | **90.8** |
| XL (~1B) | DeBERTa-v1-xl | 75.6 | 47.2 | 90.7 |
| XL (~1B) | **Ettin-Enc-1B** | 82.3 | 50.1 | **91.6** |

**Decoder 侧（10 个 zero-shot 任务平均）**：ETTIN decoder 同样追平或超过开放数据 SOTA。

| 尺寸 | 模型 | Avg |
|------|------|------|
| Base (~135-160M) | SmolLM2-135m | 45.2 |
| Base (~135-160M) | **Ettin-Dec-150m** | **46.2** |
| Large (~360-410M) | SmolLM2-360m | 53.1 |
| Large (~360-410M) | Ettin-Dec-400m | 53.1 |
| XL (~1B) | Llama-3.2-1B | 56.6 |
| XL (~1B) | **Ettin-Dec-1B** | **59.0** |

### Encoder vs Decoder 对照（含跨目标续训）

| 任务类型 | 现象 | 关键数字 |
|------|------|------|
| 分类 (MNLI) | encoder 碾压，且小一档的 encoder 胜过大一档的 decoder | 150M encoder 89.2 > 400M decoder 88.2 |
| 检索 (MS MARCO dev) | encoder 领先；续训能明显帮 decoder，但仍补不平 | 400M：encoder 42.2 vs encoder-from-decoder 41.4 |
| 生成 (平均) | decoder 领先，且差距随尺寸**变大** | 68M 几乎持平 → 1B 差距 >6 分 |
| 1B 难任务 | decoder-from-encoder 在分类强、生成弱 | MMLU：37.0 vs decoder 27.0；GSM8k：18.9 vs 32.0 |

### 关键发现
- **跨目标续训补不平架构差距**：即便续训用了 50B token（远超 LLM2Vec 的 ~10B），encoder-from-decoder 仍打不过原生 encoder，decoder-from-encoder 也打不过原生 decoder——一个 400M 原生 encoder 在 MNLI 上能赢过续训成 encoder 的 1B decoder，生成任务反之亦然。
- **生成任务上 encoder-from-decoder 缩放很差**：decoder-from-encoder 在生成任务的劣势随模型变大而扩大，提示「把 encoder 改造成生成模型」尤其不划算。
- **平均分藏着细节**：在偏分类的「生成」子任务（ARC、SciQ）上，encoder 当生成器用反而能赢 decoder；但 HellaSwag、TriviaQA、SiQA 这类任务 decoder 优势巨大，把平均强拉向 decoder。
- **性别偏见 case study**：同一份数据下，MLM 目标让模型更倾向中性代词，而两类模型对男性代词都有偏见（decoder 略重）；随尺寸增大女性代词使用上升。这是开放数据 + 全 checkpoint 才能做的分析示例。

## 亮点与洞察
- **「只差两处」的极致控制变量**是最大亮点：把 encoder/decoder 之争从「玄学」拉回到可证伪的实验，结论（MLM 强分类检索、CLM 强生成、续训补不平）因此特别有分量。
- **实用结论很硬**：在 1B 以下规模，与其训一个大 decoder 再续训成嵌入模型，不如直接训一个对口的小 encoder——后者更小却更强。作者甚至推断 3B encoder 很可能能超过 MTEB 榜上 7B+ 的 decoder，只是大尺寸 encoder 太稀缺。
- **平台价值**：开放 batch order + 200+ checkpoint 把这套模型变成可做训练动力学、数据顺序、偏见演化研究的基础设施，可迁移性远超单篇论文。
- **MNTP 而非朴素 MLM**：续训用前一 token 隐状态预测被 mask token，是为了对齐 decoder 的因果结构——这种「让续训目标贴合原架构」的细节值得借鉴。

## 局限与展望
- **续训配比不平衡**：50B 续训 token 相比 1.7T+ 预训练量很小，作者自承这模拟了现实中「少量数据做适配」的场景，但也意味着「给足够多续训预算能否补平」仍是开放问题。
- **规模上限到 1B**：多数模型太小，MMLU/GSM8k 这类难任务只有 1B 才有信号；3B encoder 能否超过 7B decoder 只是基于趋势的推断，没有直接训出来验证。
- **与并发工作结论相反需谨慎**：并发工作（Gisserot-Boukhlef et al., 2025）在 100B token 下发现「CLM 起步再续训 MLM 几乎总更好」，本文认为那是小数据下 CLM 数据效率高的假象；两边规模和设置不同，读者不应直接拿结论比大小。
- **偏见分析是单例**：WinoGender 只覆盖三类代词、且用了简化的预测代词任务，性别偏见结论是示例性的，不宜过度外推。

## 相关工作与启发
- **vs Pythia**: Pythia 首创「开放数据 + 多尺寸 + 全 checkpoint」但只有 decoder；本文把这套范式扩展到**成对的 encoder/decoder**，从而能做架构间的受控对比，是直接的精神续作。
- **vs ModernBERT**: ModernBERT 是当时最强 encoder 但数据不开放；本文用开放数据复刻其配方，提供了第一个公开可复现的 ModernBERT 式配方，同时把尺寸谱系补全（17M~1B）。
- **vs LLM2Vec**: LLM2Vec 用 MNTP 把 decoder 续训成嵌入模型（~10B token）；本文沿用其 MNTP 目标但加大到 50B token 做受控对照，结论是即便加大续训也补不平原生 encoder——给「decoder 改造派」泼了冷水。
- **vs Charpentier & Samuel (GPT or BERT)**: 他们在小数据上比 DeBERTa 与 GPT-2；本文在 SOTA 规模上做苹果对苹果的对比，把结论的可信度拉到新高度。

## 评分
- 新颖性: ⭐⭐⭐⭐ 不是新算法，但「成对受控 + SOTA 规模」的实验设计本身是过去做不到的，并给出反直觉且实用的结论。
- 实验充分度: ⭐⭐⭐⭐⭐ 6 个尺寸 × 两种架构 × 多类任务 + 跨目标续训 + 难任务 + 偏见 case study，覆盖全面。
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰、结论明确，表格信息密集；个别细节（如续训配比争议）需读者自行权衡。
- 价值: ⭐⭐⭐⭐⭐ 既给社区一套可复用的开放模型/数据平台，又给「该不该单独训 encoder」这一现实决策提供了硬证据。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Reconstructing CLIP for Open-Vocabulary Dense Perception](../../CVPR2026/llm_pretraining/reconstructing_clip_for_open-vocabulary_dense_perception.md)
- [\[NeurIPS 2025\] Gemstones: A Model Suite for Multi-Faceted Scaling Laws](../../NeurIPS2025/llm_pretraining/gemstones_a_model_suite_for_multi-faceted_scaling_laws.md)
- [\[ICML 2026\] If open source is to win, it must go public](../../ICML2026/llm_pretraining/if_open_source_is_to_win_it_must_go_public.md)
- [\[ACL 2026\] Fine-tuning vs. In-context Learning in Large Language Models: A Formal Language Learning Perspective](../../ACL2026/llm_pretraining/fine-tuning_vs_in-context_learning_in_large_language_models_a_formal_language_le.md)
- [\[ICML 2026\] Names Don't Matter: Symbol-Invariant Transformer for Open-Vocabulary Learning](../../ICML2026/llm_pretraining/names_dont_matter_symbol-invariant_transformer_for_open-vocabulary_learning.md)

</div>

<!-- RELATED:END -->
