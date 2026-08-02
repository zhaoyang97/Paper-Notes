---
title: >-
  [论文解读] TagRouter: Learning Route to LLMs through Tags for Open-Domain Text Generation Tasks
description: >-
  [ACL 2025][文本生成][model routing] 这篇论文提出 TagRouter，用一个小型标签生成器把开放域文本生成请求先压缩成一组语义标签，再基于标签统计每个候选 LLM 的相对优势并进行路由，从而在不重新训练路由器的前提下，把多模型系统的接受率做得比单个大模型更高，同时显著降低推理成本。
tags:
  - "ACL 2025"
  - "文本生成"
  - "model routing"
  - "tag generation"
  - "LLM ensemble"
  - "cost-efficient inference"
  - "open-domain generation"
---

# TagRouter: Learning Route to LLMs through Tags for Open-Domain Text Generation Tasks

**会议**: ACL 2025  
**arXiv**: [2506.12473](https://arxiv.org/abs/2506.12473)  
**代码**: 无  
**领域**: 文本生成  
**关键词**: model routing, tag generation, LLM ensemble, cost-efficient inference, open-domain generation

## 一句话总结
这篇论文提出 TagRouter，用一个小型标签生成器把开放域文本生成请求先压缩成一组语义标签，再基于标签统计每个候选 LLM 的相对优势并进行路由，从而在不重新训练路由器的前提下，把多模型系统的接受率做得比单个大模型更高，同时显著降低推理成本。

## 研究背景与动机

大模型生态发展得很快，但不同模型并不是简单地按参数量排成一条线。论文一开始就用一个很直观的现象说明问题：即使是更小的模型，在某些具体样本上也会和更大的模型打平，甚至直接胜出。也就是说，平均 benchmark 分数高，不代表对每条用户请求都最合适。

现有 routing 方法虽然已经在尝试把请求分给不同模型，但落地时有几类明显问题。

第一类方法要先让多个候选模型都生成，再在响应层面做选择。这类方法精度可能不错，但延迟和成本都很高，尤其不适合在线场景。

第二类方法依赖解码阶段信息，比如 logits 或中途状态。这对闭源商用模型很不友好，因为你往往拿不到这些内部信息。

第三类方法把 routing 看成一个监督学习分类任务。它们通常需要针对候选模型组合重新训练，一旦模型池变化，就得重做训练，适应不了快速迭代的 LLM 生态。

作者想解决的核心矛盾是：能不能做一种真正实用的路由方法，同时满足下面几个条件。

- 推理前路由，不重复调用多个大模型。
- 支持开放域文本生成，不依赖特定任务模板。
- 支持多个候选模型，而不是只在一大一小两个模型间二选一。
- 支持专有模型，不要求访问 logits。
- 能显式考虑成本控制。
- 当模型池扩充时，不需要把整套路由器重新训练一遍。

作者的切入点很有意思：与其直接用原始 query 训练一个路由器，不如先把 query 抽成标签。因为标签天然更接近“任务语义特征”，能把冗余文本压缩成更容易泛化的结构化表示。于是整篇论文的核心想法可以概括为一句话：先把 query 转成 tag，再基于 tag 去查每个模型在哪些语义模式下更擅长，最后做一个受成本约束的选择。

## 方法详解

TagRouter 由三个顺序执行的模块组成：TagGenerator、TagScorer 和 TagDecider。它不是一个端到端的大型神经路由器，而更像一个“标签抽取 + 能力字典 + 决策规则”的组合系统。这个设计的重点不在于把路由器做得很重，而在于让它可维护、可扩展、可替换。

先看整体流程。给定一个用户请求 $q$，系统先让 TagGenerator 生成标签集合 $\mathcal{T}(q)=\{t_1,t_2,...,t_j\}$。然后 TagScorer 根据这些标签，给每个候选模型 $M$ 算一个累计分数。最后 TagDecider 再结合分数差和成本阈值 $\theta$，决定把请求分给哪个模型。

它解决 routing 问题的形式化定义是：在模型集合 $\mathcal{M}$ 中，为每个 query 选择一个最合适的模型 $M^*(q)$，使系统整体表现最大化。与常见 query-to-label 分类器不同，TagRouter 学的是一个更松弛的结构：并不直接训练“query 到模型”的硬映射，而是用 tags 作为中间语义层，把 query 和 model capability 联系起来。

### 整体框架

整体框架可以拆成三步。

第一步，TagGenerator 把输入请求转成多个细粒度标签。作者没有使用固定标签体系，而是采用 open-tagging。先用 ERNIE-4.0-Turbo-8K 给 BCUQ 数据集中的 query 自动打标签，原始一共得到 14,352 个不同标签。

第二步，对这些标签做规范化和压缩。作者先去掉出现次数少于 5 次的稀有标签，再做规则归一化，然后用 PhraseBERT 表征标签语义，并通过 DBSCAN 聚类把相近标签合并。经过迭代归并后，最终标签词表收缩到 1,601 个。这一步很关键，因为 routing 不是越细越好，过碎的标签只会带来噪声和稀疏性。

第三步，基于标签统计各模型的能力画像。系统把每个标签和每个模型的 win、tie、loss 计数整理成一个分数字典。真正在线路由时，不再需要重复跑多个候选模型，而只需要生成标签、做 embedding 对齐、查表求和，再进行阈值判断。

### 关键设计

1. **TagGenerator**
	- 功能：把自由文本 query 转成一组能概括语义特征的标签。
	- 核心思路：先用强模型生成 teacher tags，再用知识蒸馏把这个能力压到一个小模型上。训练数据形式是 $(q, \mathcal{T}(q))$。
	- 设计动机：如果直接把原始 query 送给路由器，文本冗余、表达多样性和风格差异都会干扰判断；而 tags 更像是面向路由的抽象特征。

作者还专门设计了一个 Hybrid Weight-Based Data Sampling 算法，目的是在训练 TagGenerator 时兼顾高频标签和低频但重要的标签。这个点说明他们不是把 tagging 当成普通 seq2seq 任务，而是真正考虑到后续路由需要“标签覆盖度”和“标签判别力”。

2. **Tag Normalization 与 Tag Alignment**
	- 功能：把生成标签统一到稳定的标签空间里，减少同义词和表面差异导致的错配。
	- 核心思路：一部分在离线阶段做规范化和聚类归并；另一部分在在线阶段对生成的标签和标准标签做 embedding 相似度匹配，映射到统一标签空间。
	- 设计动机：如果 query 里生成了一个未登录标签，或者与标准标签只有表面表达差异，那么直接查表会失效，所以必须有一层语义对齐。

这其实是 TagRouter 比“关键词路由”更可靠的地方。它不是用字符串精确匹配，而是用语义对齐把标签映射到已有能力词表中，因此在新 query 上更稳。

3. **TagScorer**
	- 功能：估计不同模型对当前 query 的适配程度。
	- 核心思路：对每个标签 $t$ 和模型 $M_i$，计算一个分数 $\text{score}(M_i,t)$，分数由该模型在带有标签 $t$ 的样本上的 win、tie、loss 统计决定。
	- 设计动机：作者不想训练一个黑盒打分器，而是希望把“模型在什么语义标签上更强”显式存成键值对，这样既解释性更好，也方便增量更新。

论文里给出的核心公式可以概括为：

$$
\text{score}(M_i,t)=w_t \sum_{r\in\{win,tie,loss\}} \text{count}_{t,M_i}(r)\cdot s_r
$$

这里 $w_t$ 是标签权重，和标签频次相关；$s_r$ 是 win、tie、loss 对应的收益系数。直观理解就是，如果某个模型在某个标签上经常赢或者打平大模型，它在这个标签上的分数就高。

一个细节值得注意：作者把 tie 的贡献单独调优，而不是简单等同于 win。这个决定很合理，因为在路由问题里，能够“打平更大模型但更便宜”其实已经很有价值，但又不能完全等同于明确胜出。

4. **TagDecider**
	- 功能：根据标签分数和成本阈值，最终选择路由目标模型。
	- 核心思路：先对每个候选模型把所有标签分数求和，找出总分最高者；再通过阈值 $\theta$ 控制是否偏向更便宜的小模型。
	- 设计动机：真实部署不仅关心效果，还关心成本。单纯追求最优效果会让系统过度依赖大模型，失去 routing 的意义。

论文把成本控制写成一个很清晰的规则：当系统原本会路由到大模型时，比较小模型和大模型的累计得分差 $\Delta_q$。如果差距没有小到必须上大模型，就可以把 query 交给便宜的小模型。默认情况下，作者认为 $\theta=0$ 已经是一个很稳的设置；若想进一步偏向成本，可以继续调低阈值。

### 损失函数 / 训练策略

严格来说，TagRouter 本身不是端到端训练的神经路由器，所以没有一个统一的“总损失函数”。训练主要发生在 TagGenerator 上，而其余模块更多依赖统计构建和规则决策。

TagGenerator 的训练策略包括三部分。

- 先用强模型生成 teacher tags，构造蒸馏数据集。
- 用混合加权采样提高低频重要标签的出现概率。
- 在小模型上做 instruction tuning，把 tagging 能力压缩到 Qwen2.5-0.5B。

在线路由阶段没有梯度更新，只有三件事：生成 tags、对齐 tags、查表打分。也正因为如此，作者将其称为 training-free routing method。更准确地说，是“路由决策阶段免训练”，而不是系统里完全没有训练过程。

## 实验关键数据

论文的核心 benchmark 是 BCUQ。这个数据集来自百度云 ERNIE Bot 平台的真实用户 query，共 95,559 条，按文中划分为 93,669 个训练样本、1,000 个验证样本和 890 个测试样本，覆盖 brainstorming、classification、close QA、open QA、content creation、rewrite、summarization 和 others 八类任务。

作者主要考察两点：第一，routing 是否真的能让系统优于单个大模型；第二，加入 tags 这层语义抽象以后，是否能比传统 query-level routing 更强。

### 主实验

先看 BCUQ 上最关键的主结果。候选模型是 ERNIE-3.5-8K 和 ERNIE-Speed-8K，其中 EB3.5 既是性能最强模型，也是成本最高模型。

| 方法 | AR(%) | Uplift(%) | Cost | Rank | AUC(%) | PAUC(%) |
|------|------:|----------:|-----:|-----:|-------:|--------:|
| EBspeed | 59.78 | -24.10 | 2.01 | 1.400 | - | 0 |
| EB3.5 | 78.76 | 0.00 | 13.49 | 1.212 | - | 0 |
| RouteLLM-MF | 80.34 | 2.01 | 11.82 | 1.197 | 73.94 | 0.12 |
| RouterBench-KNN | 80.45 | 2.15 | 11.77 | 1.196 | 75.15 | 0.40 |
| FORC | 81.80 | 3.86 | 11.81 | 1.182 | 75.73 | 0.76 |
| RouteLLM-MF + TagGenerator | 82.02 | 4.14 | 11.66 | 1.180 | 76.08 | 0.76 |
| FORC + TagGenerator | 81.91 | 4.00 | 11.79 | 1.181 | 75.97 | 0.59 |
| **TagRouter** | **83.60** | **6.15** | **11.17** | **1.164** | **76.10** | **1.46** |

这张表最重要的结论有三个。

第一，routing 确实有效。无论是传统方法还是 TagRouter，整体都能把系统表现做得超过直接固定调用 EB3.5。

第二，tags 确实有效。把 TagGenerator 接到现有 routing 方法前面后，RouteLLM-MF、RouterBench-KNN、FORC 都有提升，说明标签比原始 query 更利于路由器抓住有效特征。

第三，TagRouter 在效果和成本上同时占优。它把 AR 提高到 83.60%，相对大模型基线提升 6.15%，同时把成本降到 11.17，相比 EB3.5 降低约 17.2%。这说明它不是单纯用更多大模型换性能，而是真的提高了“该上小模型时就上小模型”的能力。

作者还做了跨数据集和跨模型组验证。无论是在 GLM4-9B + Qwen2.5-7B 这组能力更接近的模型上，还是在 Alpaca、Dolly、BCUQ 三个数据集上，TagRouter 的平均 AUC 都高于前三个基线，说明它不是只对一组强弱差异明显的模型有效。

### 消融实验

论文的消融不是只删一个模块看掉点，而是从多个角度验证 TagGenerator、TagScorer 和 TagDecider 的设计是否必要。下面这张表挑了文中最有代表性的分析结果。

| 分析项 | 配置 | 关键结果 | 说明 |
|------|------|---------|------|
| 训练数据规模 | BCUQ 50,000 条 | AR 83.26, AUC 75.90, PAUC 1.30 | 数据减少后性能略降，但下降不大 |
| 训练数据规模 | BCUQ 70,000 条 | AR 83.48, AUC 76.00, PAUC 1.40 | 接近满数据表现 |
| 训练数据规模 | BCUQ 93,669 条 | AR 83.60, AUC 76.10, PAUC 1.46 | 满数据最好，但收益呈边际递减 |
| TagGenerator 基座模型 | Qwen2.5-0.5B | F1 57.75, Inter Rate 0.8686, AUC 76.10 | 最终采用，成本最低且表现强 |
| TagGenerator 基座模型 | Qwen2.5-7B | F1 58.15, Inter Rate 0.8918, AUC 77.48 | 更大模型略强，但不够轻量 |
| TagGenerator 基座模型 | Llama3.2-3B | F1 58.03, Inter Rate 0.8969, AUC 77.26 | 效果也不错，但不是最佳效率点 |
| TagDecider 阈值 | $\theta=0$ | BCUQ 上 AR 82.47, AUC 约 75.95 | 默认值已经很稳，适合直接部署 |
| TagDecider 阈值 | $\theta=\theta^*$ | BCUQ 上 AR 83.60, AUC 76.10 | 进一步调优后更优，但需要数据支持 |

除了表里的数字，正文还给出几条定性但很关键的结论。

- 混合加权采样有效，说明低频标签对 routing 很重要，不能只依赖高频标签。
- tag normalization 和 tag alignment 都能带来收益，说明“标签空间统一”是这类方法成立的基础。
- tie 的权重最优值不是 1，而是约 0.15，说明“打平”应当被奖励，但不该与“胜出”完全等价。

### 关键发现

- 小模型并非全局更差，而是在不同样本和任务类型上与大模型互补，这为 routing 提供了真实收益空间。
- 标签化表示是本文最核心的贡献之一，因为它不仅提升了 TagRouter 本身，也能提升其他现有路由器。
- TagRouter 在大多数任务类型上都优于基线，只有 close QA 这类模式高度固定的任务上，优势没有那么明显。
- 模型池从 2 个扩展到 3 个、5 个时，AUC 从 0.7610 提升到 0.7933，再到 0.8043，说明该方法确实能随模型生态扩展而获益。
- 即使候选模型能力接近，TagRouter 仍然能工作，不依赖“一个明显强、一个明显弱”的设定。

## 亮点与洞察

- 这篇论文最大的亮点不是又造了一个更复杂的神经路由器，而是把路由问题拆成更稳定的三个子问题：标签生成、能力统计、成本决策。这样做的工程价值很高。
- 它把 routing 从“直接在 query 上分类”变成“在语义标签空间做匹配”。这相当于先做一次面向路由的特征抽象，能减少表面表达差异对路由器的干扰。
- TagScorer 用显式键值对记录模型能力，而不是学一个端到端黑盒。这让系统更容易增量扩展，新模型加入时只要补少量标注样本和标签统计，不必整套重训。
- 论文对成本问题的处理比较务实。很多 routing 工作只强调性能，而这篇工作明确把成本阈值 $\theta$ 纳入决策逻辑，因此更接近真实产品系统。
- 一个很有启发的点是：tags 不只是 TagRouter 的中间表示，还是一种能迁移到其他 routing 方法上的“增强特征层”。这说明论文提出的可能不是单一算法，而是一种更通用的 routing 视角。

## 局限与展望

- 作者承认当前 TagGenerator 主要覆盖中文和英文，因为 BCUQ 以这两种语言为主，多语种泛化能力有限。
- 评测主要依赖 LLM-as-a-judge。虽然作者用 50 个样本验证了与人工评估的一致性较高，但这个规模仍然偏小，结论更像“可接受”而非“完全可靠”。
- 当前 TagScorer 仍然以单个最大模型作为参考模型，比较框架比较偏 pairwise。若未来模型池更大，可能需要更细致的多模型相对评分机制，例如 Elo 风格的全局评价。
- 标签质量高度影响最终路由效果，因此 TagGenerator 一旦在新领域产生系统性偏差，后面的查表和决策都会被放大。
- 论文展示了可扩展到更多模型，但大规模模型池下标签冲突、标签稀疏和查表噪声是否会累积，正文还没有完全展开。
- 一个值得继续做的方向是把 tag-based routing 扩展到多模态或 agent 场景。例如把工具需求、上下文长度、推理深度等也编码成 tags，可能会形成更强的统一路由接口。

## 相关工作与启发

- **vs FrugalGPT**：FrugalGPT 更偏“逐级尝试直到满足阈值”的思路，需要多次模型调用。TagRouter 则是在推理前完成选择，因此延迟和成本更可控。
- **vs RouteLLM**：RouteLLM 本质上仍然是 query 到模型的监督式选择，模型池变化时适应性较差。TagRouter 借助标签中间层，让扩展候选模型更自然。
- **vs FORC / RouterBench**：这些方法已经在做推理前 routing，但输入仍然更接近原始 query 表示。本文证明，把输入先转成 tags 后，甚至能直接增强这些已有方法。
- **与 mixture-of-experts 的区别**：MoE 是模型内部专家路由，TagRouter 是系统层模型路由。二者可以结合，前者优化单模型内部计算，后者优化外部服务编排。
- **对我自己的启发**：如果以后做多模型服务系统，不一定先上复杂训练式 router。先建立一个稳定的任务标签体系，再做显式能力建模，往往更可控，也更容易在线维护。

## 评分

- 新颖性: ⭐⭐⭐⭐ 不是全新的 routing 大框架，但把 tag generation、能力键值建模和成本阈值结合起来，形成了很有辨识度的方案。
- 实验充分度: ⭐⭐⭐⭐⭐ 有真实用户数据集、跨任务分析、跨模型组验证、可扩展性实验和多种消融，证据链比较完整。
- 写作质量: ⭐⭐⭐⭐ 方法结构清晰，工程动机讲得比较透；不足是部分附录表格在阅读时更像工程报告，细节稍碎。
- 价值: ⭐⭐⭐⭐⭐ 对真实 LLM 服务编排很有参考意义，尤其适合预算敏感、模型池频繁变化的场景。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Multi-document Summarization through Multi-document Event Relation Graph Reasoning in LLMs](event_graph_bias_mitigation_summarization.md)
- [\[ACL 2025\] Towards Better Open-Ended Text Generation: A Multicriteria Evaluation Framework](towards_better_open-ended_text_generation_a_multicriteria_evaluation_framework.md)
- [\[ACL 2025\] Odysseus Navigates the Sirens' Song: Dynamic Focus Decoding for Factual and Diverse Open-Ended Text Generation](odysseus_dynamic_focus_decoding.md)
- [\[ACL 2025\] Balancing Diversity and Risk in LLM Sampling: How to Select Your Method and Parameter for Open-Ended Text Generation](balancing_diversity_and_risk_in_llm_sampling_how_to_select_your_method_and_param.md)
- [\[ACL 2025\] ATGen: A Framework for Active Text Generation](atgen_a_framework_for_active_text_generation.md)

</div>

<!-- RELATED:END -->
