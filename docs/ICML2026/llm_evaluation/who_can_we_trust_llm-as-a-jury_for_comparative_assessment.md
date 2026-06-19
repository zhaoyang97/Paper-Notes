---
title: >-
  [论文解读] Who can we trust? LLM-as-a-jury for Comparative Assessment
description: >-
  [ICML 2026][LLM评测][LLM-as-a-jury] 这篇论文指出多个 LLM 评审在成对比较中可靠性差异很大，提出带评审判别参数的 BT-σ 模型，在没有人工校准标签的情况下同时学习候选输出排名和每个 LLM judge 的可靠性，从而比简单平均和标准 Bradley-Terry 聚合更接近人类排序。
tags:
  - "ICML 2026"
  - "LLM评测"
  - "LLM-as-a-jury"
  - "Bradley-Terry"
  - "评审可靠性"
  - "成对比较"
  - "无监督校准"
---

# Who can we trust? LLM-as-a-jury for Comparative Assessment

**会议**: ICML 2026  
**arXiv**: [2602.16610](https://arxiv.org/abs/2602.16610)  
**代码**: 无公开代码  
**领域**: LLM 评估 / 比较式自动评价  
**关键词**: LLM-as-a-jury, Bradley-Terry, 评审可靠性, 成对比较, 无监督校准  

## 一句话总结
这篇论文指出多个 LLM 评审在成对比较中可靠性差异很大，提出带评审判别参数的 BT-σ 模型，在没有人工校准标签的情况下同时学习候选输出排名和每个 LLM judge 的可靠性，从而比简单平均和标准 Bradley-Terry 聚合更接近人类排序。

## 研究背景与动机
**领域现状**：LLM-as-a-judge 已经成为 NLG、摘要、对话回复和开放式生成评估中的常用工具。相比直接打分，成对比较通常更稳定，因此很多系统会让一个或多个 LLM 判断候选输出 $i$ 是否优于 $j$，再把这些比较结果聚合成全局排名。

**现有痛点**：多个 LLM judge 的质量并不一致。有的模型更偏好长回答，有的对候选顺序敏感，有的在不同评价维度上循环矛盾严重。常见的概率平均或投票平均默认所有 judge 等可靠，会把噪声模型和高质量模型等权相加，导致最终排名受不一致概率拖累。

**核心矛盾**：成对比较本身应该满足某种全局排序结构，但 LLM 给出的 preference probability 经常违反传递性、交换性和校准一致性。直接使用 soft probabilities 会保留更多信息，也会放大不一致；只使用 hard decisions 更鲁棒，却丢掉概率强度。

**本文目标**：作者希望在不依赖人工标注校准集的情况下，从多个 LLM judge 的成对比较概率中同时恢复候选 item 的全局 skill 排名，以及每个 judge 的可靠性或判别能力。

**切入角度**：论文从 Bradley-Terry 模型出发，先分析 soft BT 何时会自校准、何时会因概率不一致而失效，再把“judge 是否可信”写成模型参数，而不是在聚合前手工指定权重。

**核心 idea**：给每个 LLM judge 加一个可学习的判别尺度 $\sigma_k$，让可靠 judge 对 skill 差异更敏感、噪声 judge 被自然降权，形成无监督的 reliability-aware BT 聚合。

## 方法详解
论文的主线很清楚：先把 LLM 比较概率放进 Bradley-Terry 框架，说明标准 soft BT 在多 judge 场景中等价于匹配平均概率；然后指出平均概率无法表达 judge 间的可靠性差异；最后提出 BT-σ，在同一个似然里学习 item skill 和 judge discriminator。

### 整体框架
输入是一组候选生成结果，以及多个 LLM judge 对所有候选对的 preference probabilities。对于每个 pair $(i,j)$ 和 judge $k$，模型观察到 $p_{ij}^{(k)}$，即 judge $k$ 认为 $i$ 优于 $j$ 的概率。输出包括候选 item 的全局排序分数 $s_i$，以及每个 judge 的可靠性参数 $\sigma_k$。评估时，候选排序与人类评分排序做 Spearman rank correlation。

方法先做一个对称化去偏：如果同一对候选在两个顺序下得到 $p_{ij}$ 和 $p_{ji}$，则用 $p'_{ij}=\frac{1}{2}(p_{ij}+1-p_{ji})$ 强制满足最基本的顺序一致性。之后，hard BT、soft BT、Temp-BT、BT-σ 等方法都在同一组 debiased comparisons 上比较。

### 关键设计

**1. 用概率一致性诊断 hard BT 与 soft BT 的优劣边界**：论文先回答一个反直觉现象——为什么保留概率强度的 soft BT 有时反而不如只看胜负方向的 hard BT。标准 Bradley-Terry 假设 $P(i\succ j)=\sigma(s_i-s_j)$，soft BT 用观测概率 $p_{ij}$ 去拟合这个结构。作者证明：当评审概率本身自洽（确实由某个全局 skill 向量生成）时，对概率做温度缩放只等价于整体缩放 skill 空间、不改变排名，soft BT 会隐式完成自校准，此时 hard BT 与 soft BT 给出相同排序。但真实 LLM 概率常违反传递性、交换性，无法用单一 skill 向量解释；这时 soft BT 必须去拟合互相矛盾的概率强度，反而把噪声放大，而 hard BT 丢掉幅度、只留方向，成了更抗噪的估计量。这个诊断是全文的出发点：问题不在 BT 结构，而在不同评审的概率信号质量参差，不能等权处理。

**2. BT-σ：给每个评审一个可学习的判别尺度 $\sigma_k$**：这是论文的核心。论文证明，直接把所有评审的概率喂给 soft BT，等价于先把各评审概率平均、再拟合一个 soft BT，因此只能学到一个全局排名和一套共享的隐式校准，完全无法表达评审间的可靠性差异。BT-σ 在 soft BT 似然里为每个评审 $k$ 插入一个判别尺度 $\sigma_k$：$\mathcal{L}(\mathbf{s},\{\sigma_k\})\propto\prod_k\prod_{(i,j)}\sigma((s_i-s_j)/\sigma_k)^{p_{ij}^{(k)}}(1-\sigma((s_i-s_j)/\sigma_k))^{1-p_{ij}^{(k)}}$。$\sigma_k$ 控制评审 $k$ 对 skill 差异的敏感度：$\sigma_k$ 越小，说明该评审对候选差异越敏感、概率越自洽、越可信；$\sigma_k$ 越大，说明其概率越平、越噪。所有 $\{s_i\}$ 和 $\{\sigma_k\}$ 在同一个似然里联合最大化、不需要任何人工标签。它本质上是温度缩放的无监督版本——但校准信号不来自人类标注，而来自多评审比较结构本身，从而在聚合时自动给可靠评审更大权重、压低噪声评审。论文也强调 $\sigma_k$ 只在「多评审 + 软概率」场景才有意义：单评审或 hard BT 下，全局尺度 $\sigma_k$ 会被 item skill 吸收、失去信息。

**3. 用相关性验证 $\sigma_k$ 真捕捉到可靠性，并扩展到 aspect 维度**：$\sigma_k$ 可能只是数学上的自由度，论文必须证明它确实对应「可靠性」。作者用学到的 $1/\sigma_k$ 分别与评审自身的独立 SRC、以及 $1-\text{CycleRate}$（循环一致性，CycleRate 统计三元组里出现 $i\succ j\succ k\succ i$ 这类有向环的比例）做相关分析：若越一致的评审学到越大的 $1/\sigma_k$，就说明模型抓到的是真实可靠性而非过拟合某个 benchmark。论文还提出 BT-σ-asp，为每个「评审 × 评价维度」对学一个单独判别尺度，检验可靠性是否随评价维度变化；实验发现单评审一个 $\sigma_k$ 往往已够用，说明评审可靠性大体跨维度稳定。

### 损失函数 / 训练策略
BT-σ 直接最大化上述联合似然，参数包括所有 item skills $\{s_i\}$ 和 judge discriminators $\{\sigma_k\}$。作者用 L-BFGS-B 优化，随机初始化 $s_i$ 和 $\sigma_k$，通常 100 次迭代内收敛。Temp-BT 作为有监督参考，需要用人类标注拟合每个 judge/aspect 的温度；BT-σ 不使用人类标签，只依赖 LLM pairwise probabilities。

## 实验关键数据

### 主实验
论文在 SummEval、Topical-Chat 和 NovelEval 上测试，其中主表详细报告 SummEval 与 Topical-Chat 的 Spearman correlation。SummEval 有 coherence、consistency、fluency、relevance 四个维度；Topical-Chat 有 coherency、continuity、engagingness、naturalness 四个维度。

| 数据集 | 指标 | 本文 BT-σ | 之前强基线 | 提升 |
|--------|------|------|----------|------|
| SummEval COH | SRC | 57.38 | soft BT 53.94 / Temp-BT 56.21 | 优于无监督 soft BT 3.44 点 |
| SummEval FLU | SRC | 42.99 | soft BT 42.69 / Temp-BT 41.88 | 小幅领先 |
| SummEval REL | SRC | 54.15 | soft BT 53.11 / Temp-BT 55.14 | 优于 soft BT，但低于监督 Temp-BT |
| Topical-Chat CNT | SRC | 56.30 | soft BT 53.87 / Temp-BT 52.21 | +2.43 点 vs soft BT |
| Topical-Chat NAT | SRC | 60.56 | soft BT 58.20 / Temp-BT 60.65 | 接近监督校准 |
| SummEval ALL | SRC | 50.50 | soft BT 49.40 / Crowd-BT 48.35 | 总体领先 |

### 消融实验
消融和分析主要围绕两个问题：学到的 discriminator 是否真的代表 judge 可靠性，以及 aspect-specific discriminator 是否必要。

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| SummEval，$1/\sigma_k$ vs judge SRC | ALL PCC 72.21 / SRC 85.71 | discriminator 与独立 judge 表现高度相关 |
| Topical-Chat，$1/\sigma_k$ vs judge SRC | ALL PCC 67.41 / SRC 59.52 | 跨任务仍保持正相关 |
| SummEval，$1/\sigma_k$ vs $1-\text{CycleRate}$ | ALL PCC 90.29 / SRC 95.24 | 更一致的 judge 会学到更大的 $1/\sigma_k$ |
| BT-σ-asp vs BT-σ | SummEval 小幅提升，Topical-Chat 混合 | aspect 相关可靠性存在，但收益有限 |
| hard BT-σ on Topical-Chat ENG | SRC 67.36 | 高循环噪声维度中，hard decision + reliability modeling 更稳 |

### 关键发现
- 单个 LLM judge 上，hard BT 经常能追上甚至超过 soft BT，这说明 raw probabilities 的幅度并不总可信；在多 judge 聚合后，soft BT 又变强，说明不同模型的噪声会部分抵消。
- BT-σ 的优势来自显式建模 judge heterogeneity。它不是简单平均概率，而是在 likelihood 中让不同 judge 的概率曲线有不同温度，从而自然削弱不可靠模型。
- $1/\sigma_k$ 与循环一致性相关性极高，尤其 SummEval 的 ALL SRC 达 95.24。这是很强的证据，说明 discriminator 捕捉到了“是否容易产生 preference cycle”这一可靠性维度。

## 亮点与洞察
- 论文把“LLM judge 可信度”从工程经验变成了可学习参数。很多 evaluation pipeline 会手动挑模型或简单多数投票，BT-σ 给了一个无需人工标签的概率建模替代方案。
- 对 hard BT 和 soft BT 的解释很有价值。它提醒我们，概率输出并不必然比二值偏好更好；当概率本身不满足全局排序结构时，保留概率强度可能是在保留噪声。
- $\sigma_k$ 的可解释性做得比较完整。作者没有只报告聚合分数，而是检查 discriminator 与 judge 表现、cycle inconsistency 的相关性，使方法更像一个诊断工具。

## 局限与展望
- BT-σ 仍然建立在全局 Bradley-Terry skill 的假设上。如果候选输出之间存在上下文依赖、非传递的人类偏好或多峰偏好群体，单一 skill 向量可能过于简单。
- 论文主要面向 NLG benchmark 的离线比较。真实开放式评估中，judge prompt、rubric、候选长度和安全约束会更加复杂，$\sigma_k$ 是否稳定需要进一步测试。
- Temp-BT 在部分维度仍然有优势，说明如果有高质量标注，监督校准依然有价值。未来可以研究少量标注与 BT-σ 的半监督结合。
- BT-σ 估计的是 judge 级可靠性，不直接处理 instance-level reliability。某些 judge 可能只在特定样本类型上失效，这需要更细粒度的条件化 discriminator。

## 相关工作与启发
- **vs Avg-Prob / majority voting**: 简单平均把所有 judge 等权处理，本文通过 $\sigma_k$ 学到软权重，并且强制输出满足全局排序结构。
- **vs hard / soft Bradley-Terry**: 标准 BT 只学习 item skill，本文把 judge 的概率尺度也放入模型，使 soft probability 的可信程度可变。
- **vs supervised temperature scaling**: Temp-BT 需要人类标签拟合温度，BT-σ 用成对比较结构自监督学习 discriminator，更适合 reference-free 评价场景。
- **vs Crowd-BT / annotator aggregation**: 众包模型通常假设重复标注和潜在真值，本文面向 LLM 软概率比较，直接处理生成评价中的 ranking recovery。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ 把 judge reliability 嵌入 Bradley-Terry soft comparison likelihood，问题抓得准，模型也简洁。
- 实验充分度: ⭐⭐⭐⭐☆ 覆盖多个 NLG 数据集、多个 judge、多个评价维度，并有可靠性相关分析；instance-level 失效分析还可加强。
- 写作质量: ⭐⭐⭐⭐☆ 理论动机、方法公式和实验现象衔接自然，hard/soft BT 的解释尤其清楚。
- 价值: ⭐⭐⭐⭐☆ 对自动评测系统很实用，可作为 LLM-as-a-jury 聚合和 judge 诊断的轻量模块。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] CoV-Eval: Can You Really Trust Code Copilots? Evaluating Large Language Models from a Code Security Perspective](../../ACL2025/llm_evaluation/cov_eval_evaluating_llms_from_code_security_perspective.md)
- [\[ACL 2026\] Teaching Language Models to Forecast Research Success Through Comparative Idea Evaluation](../../ACL2026/llm_evaluation/teaching_language_models_to_forecast_research_success_through_comparative_idea_e.md)
- [\[ACL 2026\] SCAN: Structured Capability Assessment and Navigation for LLMs](../../ACL2026/llm_evaluation/scan_structured_capability_assessment_and_navigation_for_llms.md)
- [\[ICML 2026\] Who Flips? Self- and Cross-Model Counterarguments Reveal Answer Instability in LLMs](who_flips_self-_and_cross-model_counterarguments_reveal_answer_instability_in_ll.md)
- [\[ICML 2026\] The ACUTE Protocol: Operationalizing Language Model Activations for Better Calibration, Utility, and Trust](the_acute_protocol_operationalizing_language_model_activations_for_better_calibr.md)

</div>

<!-- RELATED:END -->
