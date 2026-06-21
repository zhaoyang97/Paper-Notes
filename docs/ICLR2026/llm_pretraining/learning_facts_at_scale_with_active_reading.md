---
title: >-
  [论文解读] Learning Facts at Scale with Active Reading
description: >-
  [ICLR 2026][预训练][Active Reading] 让模型自己为每篇文档生成一组"学习策略"（释义、自测、知识联想、类比……）再据此合成多样化训练数据，从而把一份封闭知识高效地刻进参数里——8B 的 WikiExpert 在 SimpleQA 上反超 405B 的 Llama 和 236B 的 DeepSeekV2。
tags:
  - "ICLR 2026"
  - "预训练"
  - "Active Reading"
  - "合成数据生成"
  - "事实召回"
  - "知识注入"
  - "持续预训练"
---

# Learning Facts at Scale with Active Reading

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=mRi2cJDtIS](https://openreview.net/forum?id=mRi2cJDtIS)  
**代码/模型**: 公开 WikiExpert-8B 模型与 1T-token 合成数据集  
**领域**: LLM 预训练 / 知识注入 / 合成数据  
**关键词**: Active Reading, 合成数据生成, 事实召回, 知识注入, 持续预训练  

## 一句话总结
让模型自己为每篇文档生成一组"学习策略"（释义、自测、知识联想、类比……）再据此合成多样化训练数据，从而把一份封闭知识高效地刻进参数里——8B 的 WikiExpert 在 SimpleQA 上反超 405B 的 Llama 和 236B 的 DeepSeekV2。

## 研究背景与动机
**领域现状**：LLM 把海量世界知识存进参数里，但"怎么可靠地教模型学会一份指定知识"几乎没有可控手段。预训练阶段，长尾事实因为在语料里只零星出现而学不牢；微调阶段往新知识上灌，又容易引发幻觉、或只是死记硬背而无法迁移使用。

**现有痛点**：实践者提升某份知识源覆盖率的常规做法是简单提高它在数据里的混合权重，但单纯重复读会过拟合到表面形式而不泛化。Allen-Zhu & Li 提出的"释义增强"（paraphrase）能缓解一部分，但释义只是众多学习方式中的一种，且模型能想出的释义花样有限，规模一上去很快饱和；合成 QA（synth QA）在小规模强、放大后同样停滞。

**核心矛盾**：人类学新知识时会主动调用多种策略——主动回忆、间隔重复、画图、自问自答、用类比，而且不同知识适配不同策略（历史用时间线、抽象数学用具体类比）；但现有合成数据方法都钉死在单一固定模板上，缺乏这种"因材施教"的多样性，导致训练信号同质、知识吸收效率低。

**本文目标**：给定一份封闭知识语料，训练模型尽可能完整、可泛化地内化其中事实（逼近完美事实召回），既能做专家域适配，也能放大到预训练规模。

**核心 idea**：**与其人工设计"最好的学习策略"，不如让模型自己针对每篇文档提出一批多样化学习策略，再逐一执行生成自训练数据**——这就是 Active Reading，一个"自生成策略 → 据策略合成"的两阶段管线。

## 方法详解

### 整体框架
Active Reading 是一个简单的两阶段合成数据管线：第一阶段给模型一篇源文档，让它提出若干条针对这篇文档的学习策略；第二阶段把每条策略独立施加到该文档上，生成一篇风格各异的合成文档。所有合成文档汇总成自训练语料，再拿去继续训练（微调或持续预训练）模型。

```mermaid
flowchart LR
    A[源文档<br/>如 Wikipedia 某条目] --> B[阶段一：自生成学习策略<br/>释义/时间线/编歌/类比/自测...]
    B --> C[阶段二：逐策略独立施加<br/>每条策略→一篇合成文档]
    C --> D[多样化合成语料]
    D --> E[继续训练 LLM<br/>+混入预训练数据]
    E --> F[更高事实召回的模型]
```

### 关键设计

**1. 自生成学习策略：把"怎么学"交给模型自己想。** Active Reading 不预设任何固定增强模板，而是先用一个 prompt 让模型读完源文档后列出适合它的学习策略——可能是"给历届获奖者排时间线找规律"、"把人名编成押韵歌谣"、"用你熟悉的人或事件做联想"等。这一步是方法的灵魂：因为策略是针对具体文档内容现场生成的，天然带上下文相关性，且彼此互不重样。作者发现这种自由生成实际上会**自动复现**前人提出的多种增强（释义、合成 QA、概念图 EntiGraph 都被涵盖），说明 Active Reading 是这些方法的超集，多出来的多样性正是它更强的来源。

**2. 任务无关 vs. 任务特定两种 prompt。** 作者用两种 prompt 实例化框架。任务无关版只笼统要求"想办法学透这份材料"；任务特定版则告诉模型下游会被怎样考（如知识竞赛、金融分析），让它**先想象下游可能的考题、再围绕这些考题设计学习策略**。后者带来两个好处：一是数据更贴下游（比如知识竞赛会聚焦长尾事实），二是数据更多样（每次只聚焦文档的某个侧面）。实验里任务特定版在 SimpleWikiQA 上略胜，且 self-BLEU 更低（更多样），印证了"多样性 → 更好扩展性"的判断。

**3. 把微调改造成持续预训练以支撑规模化。** 当训练语料从 SimpleWikiQA 的 ~0.1% Wikipedia 扩到 4×、16× 更多文档时，目标事实的召回会因"干扰文档"急剧下降（类比稠密/生成式检索的扩展难题）。作者发现两处改动能扭转这一恶化趋势：其一，把学习率从微调常用的 $1\text{e-}5$ 大幅拉高到 $3\text{e-}4$，更像持续预训练——大学习率把模型推出局部极小，给学习新事实腾出"弹性容量"；其二，相应**加大预训练数据在混合比例中的权重**来修复因大学习率"打坏"的既有能力。一个反直觉现象是：保持 SimpleWikiQA 的梯度步数不变、只把它的相对占比从 80% 压到 2.5%（同时放大增强版 Wikipedia 与预训练数据），不仅 NaturalQuestions 这类护栏指标恢复，连目标任务 SimpleWikiQA 也一并恢复——说明这里的退化并非经典"灾难性遗忘"能完全解释，混入预训练数据似乎给了模型更强的"可塑性"去组织新知识。

**4. 自生成优于更大模型代生成。** 作者额外用 70B 模型来生成数据训练 8B，结果反而不如 8B 用自己生成的数据（66.25 vs 62.26）。一个假设是：训练数据越贴近模型自身的理解力与已有知识、越不"超纲"，学习效率越高，因而由"待训练的那个模型本身"来生成数据可能很关键。这点把 Active Reading 与"靠更强 teacher 蒸馏"的范式区分开。

## 实验关键数据

设置：从 Llama 3.1 8B Base 继续训练 20,000 步，每个 baseline 固定生成约 40 亿词；训练时混入 10% DCLM 预训练数据防退化；答案用 GPT-4o 判分。

### 主实验表格（专家域，事实召回 %）

| 方法 | SimpleWikiQA | FinanceBench info. | FinanceBench all |
|---|---|---|---|
| Llama 3.1 8B Base | 7.42 | 3.93 | 6.00 |
| repeat（裸文档微调） | 15.92 | 18.43 | 10.49 |
| paraphrase | 25.74 | 43.87 | 17.64 |
| synth QA | 47.87 | 44.23 | 17.16 |
| **Active Reading（任务无关）** | 63.33 | **66.18** | **26.83** |
| **Active Reading（任务特定）** | **66.25** | 61.49 | 25.16 |
| paraphrase+synthQA+AR | 66.66 | 64.45 | 26.12 |
| gold context（8B，上界参考） | 65.85 | 84.71 | 44.36 |
| gold ceiling（70B Instruct） | 90.55 | 92.49 | 57.43 |

SimpleWikiQA 上从裸微调 15.92 提到 66.25（+313% 相对），甚至**追平把文档塞进上下文的 gold context 8B 基线**；FinanceBench 信息抽取子集相对裸微调 +160%。但 FinanceBench 整体仍与 gold ceiling 有大差距——需额外推理的题，纯参数化方法仍输给在线读上下文。

### WikiExpert 规模化结果（1T 合成 token，共 8T token 训练）

| 模型 | SimpleQA | NQ | TQA |
|---|---|---|---|
| Llama 8B | 7.3 | 29.0 | 64.3 |
| **WikiExpert-8B** | **23.5** | 31.2 | 68.5 |
| Qwen2.5 72B | 9.1 | 33.2 | 71.9 |
| DeepSeekV2 236B | 10.2 | 38.6 | 80.0 |
| Llama 405B | 17.1 | 41.5 | 82.7 |
| DeepSeekV3 671B | 24.9 | 40.0 | 82.9 |

8B 的 WikiExpert 在长尾事实 SimpleQA 上 +222%（7.3→23.5），**反超 236B 的 DeepSeekV2 和 405B 的 Llama**，逼近 671B 的 DeepSeekV3。

### 消融与分析

| 分析维度 | 关键发现 |
|---|---|
| 数据扩展（Fig.2） | paraphrase 很快饱和、synth QA 也停滞；Active Reading 持续涨到 4B 词仍不饱和 |
| 答案覆盖率（Fig.5） | synth QA 覆盖率最高却表现最差→优势**不来自**答案覆盖 |
| 数据多样性 self-BLEU（Fig.6） | AR（尤其任务特定）self-BLEU 最低=最多样，与更好扩展性吻合 |
| 模型尺寸（Table 3） | 8B 用自生成数据(66.25) > 用 70B 生成数据(62.26)；自生成更优 |
| 干扰文档（Fig.3/4） | 微调设置下扩 Wikipedia 会崩；大学习率+加重预训练数据可恢复 |

### 关键发现
- Active Reading 的优势**不是**靠更高答案覆盖率，而是靠**数据多样性**（self-BLEU 更低），多样性带来更好的规模化趋势。
- 学知识"用自己生成的数据"比"用更大模型生成的数据"更有效，挑战了"teacher 越强越好"的直觉。
- 混入预训练数据不仅修护栏指标，还能在 SimpleWikiQA 步数不变时恢复目标性能——暗示其作用超出"防遗忘"，更像是给模型"可塑性"。

## 亮点与洞察
- **把人类学习论搬进合成数据**：核心洞见是"别去工程化单一最优策略，让模型自己因材施教地想一堆策略"，这一步把固定模板方法变成它们的超集，多样性自然涌现。
- **小模型靠数据法逆袭大模型**：8B 在事实召回上压过 405B/236B，给"小模型 + 精心合成数据"路线提供了一个有力证据点。
- **规模化的两味药很实用**：拉高学习率（微调→持续预训练）+加重预训练数据混合，是把任何知识注入方法放大时都值得借鉴的配方。
- **开源诚意**：放出 WikiExpert-8B 与完整 1T-token 合成数据集，便于复现与后续研究。

## 局限与展望
- **推理类问题仍是短板**：FinanceBench 整体与 gold ceiling 差距大，纯参数化方法在需要额外推理的题上仍不敌在线读上下文（RAG）。
- **机理未明**："为什么混入预训练数据能恢复并提升知识学习"、"为什么自生成数据优于更大模型生成"都只给了假设，未有机制层面的解释。
- **成本**：1T token 合成 + 8T token 训练规模巨大，中小团队难以复现完整 WikiExpert 路线。
- **评测依赖 LLM 判分**：主结果用 GPT-4o 判分，存在评测器偏差的潜在风险。
- **展望**：作者把"预训练数据如何带来可塑性/逆转知识熵衰减"列为关键未来方向，并视其为通向终身学习（持续吸收新知识）的里程碑。

## 相关工作与启发
- **知识注入**：承接 LLM 即隐式知识库（Petroni、Roberts）一脉，针对长尾事实退化、微调引入幻觉/知识冲突等已知问题，提出可放大到预训练规模的注入方法。
- **合成数据**：相对 Phi 系列（以推理/理解为主、明确不追求知识）与 EntiGraph（概念图，泛化好但弱于 synth QA），本方法定位为这些固定模板的"超集"，靠多样性取胜。
- **域适配**：不同于多数靠人工策展大规模领域数据的工作，Active Reading 既通用（任意域可用）又自适应（自动贴合所施加的域）。
- **启发**：对想把私域/长尾知识灌进自有模型的团队，"让目标模型自己生成多样化学习材料 + 持续预训练式超参 + 加重通用数据混合"是一套可直接迁移的工程范式。

## 评分
- 新颖性: ⭐⭐⭐⭐ "让模型自生成学习策略"这一框架视角新颖且优雅，且证明它是已有增强方法的超集。
- 实验充分度: ⭐⭐⭐⭐⭐ 从专家域到 1T-token 预训练规模、扩展曲线、覆盖率/多样性/模型尺寸消融齐全，并放出模型与数据集。
- 写作质量: ⭐⭐⭐⭐ 动机—方法—规模化—分析层层递进，图表清晰；个别机理只给假设。
- 价值: ⭐⭐⭐⭐⭐ 8B 反超百亿级模型 + 开源数据集，为"合成数据做可靠事实学习"提供了高影响力的实证基线。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Joint Selection for Large-Scale Pre-Training Data via Policy Gradient-based Mask Learning](joint_selection_for_large-scale_pre-training_data_via_policy_gradient-based_mask.md)
- [\[ICLR 2026\] GneissWeb: Preparing High Quality Data for LLMs at Scale](gneissweb_preparing_high_quality_data_for_llms_at_scale.md)
- [\[NeurIPS 2025\] Mouse-Guided Gaze: Semi-Supervised Learning of Intention-Aware Representations for Reading Detection](../../NeurIPS2025/llm_pretraining/mouse-guided_gaze_semi-supervised_learning_of_intention-aware_representations_fo.md)
- [\[ICLR 2026\] Nemotron-CC-Math: A 133 Billion-Token-Scale High Quality Math Pretraining Dataset](nemotron-cc-math_a_133_billion-token-scale_high_quality_math_pretraining_dataset.md)
- [\[NeurIPS 2025\] Memory Mosaics at Scale](../../NeurIPS2025/llm_pretraining/memory_mosaics_at_scale.md)

</div>

<!-- RELATED:END -->
