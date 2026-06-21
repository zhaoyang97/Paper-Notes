---
title: >-
  [论文解读] AlphaBench: Benchmarking Large Language Models in Formulaic Alpha Factor Mining
description: >-
  [ICLR2026][LLM评测][Alpha 因子挖掘] AlphaBench 是第一个系统评测大语言模型在「公式化 Alpha 因子挖掘」（FAFM）能力的基准，把量化研究员的真实工作流拆成因子生成、因子评估、因子搜索三大任务，在 Qlib + CSI300 真实回测环境下横评十余个开源/闭源模型，发现 LLM 能可靠地生成合法因子、却在判断因子好坏（评估任务）上接近随机猜测。
tags:
  - "ICLR2026"
  - "LLM评测"
  - "Alpha 因子挖掘"
  - "LLM Benchmark"
  - "因子生成"
  - "因子评估"
  - "因子搜索"
---

# AlphaBench: Benchmarking Large Language Models in Formulaic Alpha Factor Mining

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=d97Q8r7ZKZ](https://openreview.net/forum?id=d97Q8r7ZKZ)  
**代码**: https://alphabench.cc/  
**领域**: LLM 评测 / 量化金融 / 公式化 Alpha 因子挖掘  
**关键词**: Alpha 因子挖掘, LLM Benchmark, 因子生成, 因子评估, 因子搜索

## 一句话总结
AlphaBench 是第一个系统评测大语言模型在「公式化 Alpha 因子挖掘」（FAFM）能力的基准，把量化研究员的真实工作流拆成因子生成、因子评估、因子搜索三大任务，在 Qlib + CSI300 真实回测环境下横评十余个开源/闭源模型，发现 LLM 能可靠地生成合法因子、却在判断因子好坏（评估任务）上接近随机猜测。

## 研究背景与动机
**领域现状**：在量化投资里，一个 alpha 因子就是一条从行情数据中提取预测信号的数学表达式——由算子（`Mean`、`Corr`、`Std`、`Rank` 等数学/统计/时序变换）和变量（`$close`、`$volume` 等历史价量）组合而成。给每只股票在每个时点算出因子值后，就能排序选股、做多高分股做空低分股。因子好坏通常用信息系数 IC 或排序信息系数 RankIC 衡量，即因子值与未来真实收益的相关性。**公式化 alpha 因子挖掘（FAFM）** 就是不断发现新的、有预测力的因子公式，因为老因子会随市场适应而衰减（alpha decay），必须持续补充。

**现有痛点**：传统因子靠人类专家凭金融直觉手搓（如 Alpha101、Alpha158 因子库），受限于先验知识、且无法在部署后快速适应市场变化。后来用机器学习自动搜索——遗传规划（AutoAlpha）、强化学习（AlphaGen）、符号回归——能挖出新信号，但工程与算力开销大。LLM 擅长符号推理、代码生成和公式合成，天然适合自动设计因子，近两年涌现了 FAMA、QuantAgent、AlphaAgent、Alpha-GPT 等一批 LLM 驱动的因子挖掘工作。

**核心矛盾**：尽管热度很高，LLM 在 FAFM 里的真实能力却是一团迷雾——生成的因子可能带偏见、产出非法/不可执行的公式、在市场 regime 切换下不鲁棒。更关键的是，**整个领域没有标准化基准**来衡量「LLM 到底在因子挖掘的哪个环节强、哪个环节弱」。代码生成、数学推理、科学发现都早有严格 benchmark，唯独 FAFM 没有；同时大规模搜索时，不同 LLM 配置（模型类型、提示范式、推理策略）如何影响表现也没人系统研究过。

**本文目标**：填补这个空白，把「LLM 在 FAFM 中扮演什么角色」形式化，并设计覆盖因子全生命周期的多任务、多指标评测，搞清楚不同 LLM 在各维度的长短板，以及配置变量（model type / reasoning paradigm / prompting strategy）的影响。

**核心 idea**：把量化研究员的真实工作流——生成因子、评估因子、搜索因子——拆成三个可量化的 benchmark 任务，所有产出的因子都在 Qlib 回测引擎 + CSI300 真实历史数据上执行验证，用统一指标横评一大批模型，从而给「LLM 做量化」第一次画出能力地图。

## 方法详解

### 整体框架
AlphaBench 不是一个新模型，而是一套**评测协议 + 数据集 + 指标体系**。它把 FAFM 的工作流抽象成三个核心任务，每个任务对应量化研究员真实流程中的一个关键阶段，并各自拆成子任务来测不同维度的能力：

- **因子生成（Factor Generation）**：把自然语言描述翻译成合法的因子公式。共 687 条生成指令。
- **因子评估（Factor Evaluation）**：让 LLM 当「裁判」，不做完整回测就预判因子质量，用来加速筛选。共 1170 条评估指令。
- **因子搜索（Factor Searching）**：在组合空间里迭代搜索，而非一次性生成。覆盖 3 种搜索算法、27 条搜索指令。

所有因子都要求生成为 **Qlib 兼容格式**，从而能直接被 Qlib 回测框架执行打分；搜索任务以 Qlib 内置的 Alpha158 作为初始因子池，用 2020–2025 年 CSI300 成分股日频数据（含牛、熊、震荡三种市况）做真实市场回测。除了任务级评测，作者还系统消融了两个配置变量：用哪个 LLM（Gemini / GPT / DeepSeek / LLaMA / Qwen 等十余个开源闭源模型）和用什么提示方法（vanilla vs Chain-of-Thought）。

### 关键设计

**1. 因子生成任务：从语言到可执行公式，分难度测语义对齐**

这一任务针对「LLM 能不能把金融意图正确翻译成合法且符合语义的公式」。它拆成两个子任务：**Text2Alpha** 给一个宽泛描述（如动量、均值回归），要模型不依赖示例地翻译成完整公式，测的是对金融概念的理解；**Directional Mining** 给一个具体主题（如波动率类、成交量驱动类信号），要模型产出一组该主题下的多样因子，测约束遵循能力和创造性。评测用三个指标：Reliability（产出的代码是否合法可执行）、Stability（多次输出是否一致）、Accuracy（是否匹配用户意图），Overall 取三者均值。每个子任务的测例还按 easy / medium / hard 分三档难度，专门暴露模型在难指令下的语义对齐能力——这是该任务的关键设计动机：合法性容易达标，难的是语义对齐。

**2. 因子评估任务：让 LLM 当零样本「裁判」，并下钻到原子能力**

回测每个候选因子昂贵又耗时，因此一个诱人的设想是：让 LLM 仅凭因子结构、算子和经济直觉，在**不做回测**的情况下预判质量，从而加速早期筛选。这一任务拆成两个子任务：**Ranking** 给一池候选因子，要模型选出最好的 top-$k$，用 precision@$k$ 和与真实回测结果的 rank correlation 衡量；**Scoring** 给单个因子，要模型打出绝对分数（如预估 IC、Sharpe 或定性评级），用 Signal Accuracy 和 MAE 衡量。为了更精细地剖析模型行为，作者进一步把评估拆成两个**原子任务**：**Signal Classification**（判断一个因子是有意义的信号还是噪声）和 **Pairwise Selection**（两个因子里选更好的那个）。这套从复合任务到原子任务的分解，是为了定位「模型究竟卡在哪一步」——结果证明评估恰恰是 LLM 最弱的环节。

**3. 因子搜索任务：三种搜索范式 + 成本/质量双轴评测**

因子挖掘需要在「广探索」与「快收敛」之间权衡，实践中常用链式、树式或多轮迭代的搜索。AlphaBench benchmark 了三种代表性的 LLM 驱动搜索范式（见原文 Figure 3）：**Chain-of-Experience（CoE）**，从种子因子出发的顺序精炼，把历史轮次的表达式和指标喂回去逐步改进；**Tree-of-Thought（ToT）**，带剪枝的分支探索，逐层扩展；**Evolutionary Algorithm（EA）**，种群迭代，通过 LLM 模拟的交叉（crossover）和变异（mutation）算子在表达式树上做节点交换/替换。搜索任务沿两个轴评测：**Search Cost**（一次有效生成需要多少轮、token 用量多少、产出合法率多高，因为 LLM 没法一步生成所有合法因子）和 **Search Quality**（CoE/ToT 比较发现因子相对初始种子的提升，EA 评估整个因子种群的整体表现）。这个双轴设计直接揭示了「质量与成本的权衡」这一核心实践结论。

**4. 配置变量消融：模型 × 提示 × 温度 × 搜索容量**

AlphaBench 不止做任务级横评，还把影响表现的配置变量当成一等公民来消融。对生成和评估任务，对比 vanilla 提示与 **Chain-of-Thought（CoT）**，看强迫模型逐步推理是否真能提升 FAFM 表现；对搜索任务，扫不同 **temperature**（探索-利用权衡）和 **EA 容量**（种群规模），看输出随机性和候选数量如何影响搜索效率与质量。作者还在 GPT-4.1-Mini 上做了监督微调（SFT）实验，验证少量标注数据能否补上评估能力的短板，以及跨市场（CSI300 ↔ SP500）的迁移性。token 成本上，以 DeepSeek-V3 为例，全任务约消耗 5.5M token（其中 4.2M 在 prompt），支持 prompt caching 的模型缓存命中率达 85%，显著降低重复成本。

## 实验关键数据

实验环境：Qlib 回测框架 + Alpha158 初始因子池 + CSI300 成分股 2020–2025 日频数据（牛/熊/震荡三种市况）。模型覆盖闭源（Gemini-2.5-Pro/Flash、Gemini-1.5-Flash-8B、GPT-4.1-Mini、GPT-5）与开源（DeepSeek-V3、DeepSeek-R1-Distill-Qwen-32B、LLaMA3.1-70B/8B、Qwen2.5-14B）。

### 主实验：三大任务横评

**因子生成**（vanilla 提示，Overall = Reliability/Stability/Accuracy 均值）：

| 模型 | Reliability | Stability | Accuracy | Overall |
|------|------|------|------|------|
| GPT-5 | 1.00 | 0.62 | 0.56 | **0.72** |
| Gemini-2.5-Flash | 0.99 | 0.57 | 0.57 | 0.71 |
| GPT-4.1-Mini | 0.93 | 0.59 | 0.44 | 0.65 |
| LLaMA3.1-70B-Instruct | 0.95 | 0.52 | 0.38 | 0.62 |
| DeepSeek-V3 | 0.91 | 0.35 | 0.31 | 0.52 |
| DeepSeek-R1-Distill-Qwen-32B | 0.35 | 0.19 | 0.14 | 0.23 |

结论：大商用模型整体领先，但所有模型的 **Reliability 普遍很高、Accuracy 明显偏低**——模型能可靠产出语法合法的因子，但把表达式对齐到意图（尤其 hard 指令）才是核心瓶颈。CoT 只带来边际收益，有时反而降低 Stability。专用 coder 模型（Qwen3-Coder-480B 等）虽合法率高，但稳定性和准确率不如同规模通用模型。

**因子评估**（vanilla / CoT，跨 Neutral/Bear/Bull 平均）：

| 模型 | Precision (Rank) | Signal ACC | MAE | Overall |
|------|------|------|------|------|
| Gemini-2.5-Pro | 0.24 | 0.36 | 1.66 | 0.48 |
| GPT-5 | 0.24 | 0.32 | 1.67 | 0.47 |
| Gemini-2.5-Flash | 0.25 / 0.14 | 0.32 / 0.33 | 1.67 / 1.64 | 0.47 / 0.44 |
| DeepSeek-V3 | 0.19 | 0.16 | 1.60 | 0.40 |

结论：**评估任务出奇地差**，没有任何模型在 ranking 和 scoring 上同时表现强，最好的 GPT-5/Gemini-2.5-Pro 的 Overall 也只在 0.40~0.48，CoT 几乎无帮助甚至让 precision 下降。下钻到原子任务（Signal Classification、Pairwise Selection），准确率普遍接近随机猜测。

**因子搜索**（Search Quality / Cost 均缩放到 [0,1]，越高越好）：

| 模型 | Search Quality | Search Cost |
|------|------|------|
| GPT-5 | **0.656** | **0.940** |
| Gemini-2.5-Flash | 0.646 | 0.850 |
| Gemini-2.5-Pro | 0.632 | 0.808 |
| LLaMA3.1-70B-Instruct | 0.624 | 0.850 |
| GPT-4.1-Mini | 0.608 | 0.904 |
| DeepSeek-V3 | 0.494 | 0.800 |

结论：质量与成本存在明显权衡——Gemini-2.5-Pro 原始质量强但 token 不经济；GPT-5 兼顾高搜索效果与高成本效率，最均衡；中型模型居中。

### 消融实验

| 配置 | 关键发现 |
|------|---------|
| vanilla vs CoT | 生成任务 CoT 仅边际收益、有时降稳定性；评估任务 CoT 基本无效甚至掉点 |
| temperature（搜索） | 高温（1.5）提升多样性、偶尔提升表现，但成本更高、稳定性更低；低温（0.75）更高效可靠，多样性略降 |
| EA 容量 | 增大种群仅在一定点前提升表现，之后饱和 |
| SFT（GPT-4.1-Mini） | 少量标注数据即可大幅提升 Pairwise Selection（CSI300 上 0.48→0.83），但对 Signal Classification 收益有限（易过拟合塌缩到单一「noise」标签） |
| 跨市场迁移 | CSI300 上微调的经验迁移到 SP500 的 Pairwise Selection 仍有提升；作者归因于中国市场更严的交易规则、涨跌停和散户驱动波动制造了更多「噪声样态」可学 |

### 关键发现
- **能力割裂**：LLM 在「生成合法因子」上很强（Reliability 接近满分），在「判断因子好坏」上接近随机——生成易、评估难，这是全文最反直觉的结论。
- **规模收益递减**：在复杂评估任务上，增大模型规模/知识容量收益有限，Gemini-2.5-Flash 这类中型模型反而能在 Signal Accuracy 上有竞争力。
- **CoT 不是万灵药**：在 FAFM 这类需要执行验证的任务上，强迫逐步推理常常无益甚至有害。
- **微调能补评估短板**：少量市场内标注数据就能显著提升评估能力，暗示评估弱不是能力天花板而是缺乏对齐数据。

## 亮点与洞察
- **把真实量化工作流拆成可量化任务**：generation / evaluation / searching 三段映射研究员日常，所有产出都过 Qlib 真实回测，避免了「只看自由文本生成」的虚评，是该 benchmark 最扎实的设计。
- **复合任务 → 原子任务的分解诊断法**：评估任务表现差时，进一步拆成 Signal Classification 和 Pairwise Selection 两个原子任务定位病灶，这种「能力解剖」思路可迁移到任何「模型整体差但说不清差在哪」的评测场景。
- **「生成易、评估难」的结论很有指导意义**：它直接告诉量化从业者——可以放心让 LLM 当因子生成器，但别让它当裁判，评估环节仍需真实回测或专门微调。
- **成本/质量双轴 + token 统计**：把 token 用量、合法率、缓存命中率都纳入评测，让 benchmark 不只比「谁强」还比「谁划算」，更贴近工程落地。

## 局限与展望
- **评估指标本身的天花板**：零样本让 LLM 预测 IC/Sharpe 本就极难，接近随机的结果有多少是任务设计过苛、有多少是模型真不行，benchmark 没有完全区分开。
- **市场与时间范围有限**：主战场是 CSI300（2020–2025），SP500 仅在部分原子任务出现；不同市场结构下结论的普适性需进一步验证。
- **搜索范式仍偏经典**：CoE/ToT/EA 都是已有范式的 LLM 化，benchmark 暂未覆盖更前沿的 agent 式自我改进或多模型协作搜索。
- **改进方向**：作者指出 LLM 在鲁棒性、搜索效率、实际可用性上仍有持续挑战；微调实验提示「评估能力可通过少量对齐数据补齐」是值得深挖的方向。

## 相关工作与启发
- **vs 传统/ML 因子挖掘（AutoAlpha 遗传规划、AlphaGen 强化学习、符号回归）**：它们直接搜因子、工程算力开销大；本文不造新挖掘器，而是评测 LLM 在挖掘各环节的能力，定位 LLM 该补位哪一步。
- **vs LLM 驱动的因子挖掘工作（FAMA、QuantAgent、AlphaAgent、Alpha-GPT、LLM-MCTS）**：这些是「用 LLM 挖因子的具体系统」，本文是「衡量这些系统底座能力的标尺」，把零散的系统级声明放到统一基准下可比。
- **vs 金融 LLM benchmark（FinQA、ConvFinQA、FinBen、Pixiu）**：它们聚焦数值推理和问答，评的是自由文本输出；AlphaBench 把 FAFM 当成代码生成问题，要求产出可执行的 DSL 表达式并真实回测，填补了「金融领域可执行因子生成」的评测空白，桥接了程序合成评测与金融 AI。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ 首个 FAFM 的 LLM 系统性 benchmark，任务设计贴合真实工作流，但方法论上是「评测协议」而非新模型
- 实验充分度: ⭐⭐⭐⭐⭐ 十余模型 × 三任务 × 多配置 × 真实回测，还含 SFT 与跨市场迁移，覆盖很全
- 写作质量: ⭐⭐⭐⭐☆ 任务/指标定义清晰，图表丰富，部分附录指标需翻原文才完整
- 价值: ⭐⭐⭐⭐⭐ 「生成易、评估难」的结论对量化落地有直接指导意义，benchmark 可作为后续 FAFM 研究的标准标尺

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] SimBench: Benchmarking the Ability of Large Language Models to Simulate Human Behaviors](simbench_benchmarking_the_ability_of_large_language_models_to_simulate_human_beh.md)
- [\[ACL 2025\] AD-LLM: Benchmarking Large Language Models for Anomaly Detection](../../ACL2025/llm_evaluation/ad-llm_benchmarking_large_language_models_for_anomaly_detection.md)
- [\[ACL 2025\] Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models](../../ACL2025/llm_evaluation/retrieval_models_arent_tool-savvy_benchmarking_tool_retrieval_for_large_language.md)
- [\[ACL 2025\] CodeMEnv: Benchmarking Large Language Models on Code Migration](../../ACL2025/llm_evaluation/codemenv_benchmarking_large_language_models_on_code_migration.md)
- [\[ICML 2026\] PoliticsBench: Benchmarking Political Values in Large Language Models with Multi-Stage Roleplay](../../ICML2026/llm_evaluation/politicsbench_benchmarking_political_values_in_large_language_models_with_multi-.md)

</div>

<!-- RELATED:END -->
