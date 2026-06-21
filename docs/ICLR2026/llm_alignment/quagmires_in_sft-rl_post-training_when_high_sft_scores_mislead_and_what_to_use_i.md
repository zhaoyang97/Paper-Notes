---
title: >-
  [论文解读] Quagmires in SFT-RL Post-Training: When High SFT Scores Mislead and What to Use Instead
description: >-
  [ICLR 2026][LLM对齐][SFT-RL 后训练] 这篇论文用 100 多个模型、超过 100 万 GPU 小时的实验证明：推理大模型后训练里「SFT 分数越高、RL 后效果就越好」是个广泛存在的伪命题，并提出用**验证集泛化损失**和 **Pass@大 k** 两个指标来可靠预测 RL 最终成绩，把预测精度（$R^2$、Spearman 秩相关）相对直接用 SFT 分数最高提升达 0.5（约 2 倍）。
tags:
  - "ICLR 2026"
  - "LLM对齐"
  - "SFT-RL 后训练"
  - "RLVR"
  - "GRPO"
  - "泛化损失"
  - "Pass@k"
---

# Quagmires in SFT-RL Post-Training: When High SFT Scores Mislead and What to Use Instead

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=uLM3BfKo19](https://openreview.net/forum?id=uLM3BfKo19)  
**代码**: https://github.com/feiyang-k/SFT-RL-ICLR  
**领域**: 对齐RLHF / LLM推理 / 后训练分析  
**关键词**: SFT-RL 后训练, RLVR, GRPO, 泛化损失, Pass@k

## 一句话总结
这篇论文用 100 多个模型、超过 100 万 GPU 小时的实验证明：推理大模型后训练里「SFT 分数越高、RL 后效果就越好」是个广泛存在的伪命题，并提出用**验证集泛化损失**和 **Pass@大 k** 两个指标来可靠预测 RL 最终成绩，把预测精度（$R^2$、Spearman 秩相关）相对直接用 SFT 分数最高提升达 0.5（约 2 倍）。

## 研究背景与动机
**领域现状**：自 DeepSeek-R1 之后，推理 LLM 的后训练几乎都遵循「先 SFT 做冷启动、再 RLVR（用可验证奖励的强化学习，通常是 GRPO）」的两段式流水线。工业界里这两段常由不同团队负责，SFT 团队各自把「SFT 后在 benchmark 上的准确率」最大化，然后默认地把 SFT 分数最高的模型交给 RL 团队，相信它也会在 RL 后最强。

**现有痛点**：这个「SFT 分数最高 → RL 后最强」的假设经常不成立。作者给出大量反例：在同一批数据上重复训练更多 epoch 会稳定提升 SFT 分数，但过训练会压缩 RL 的提升空间，有时 RL 后反而更差；只训练「最短/最简单」的样本能让 SFT 分数涨得更快，但这些模型在 RL 阶段几乎学不到东西。极端情况下，在 SFT 提升过的模型上跑 RL，最终成绩还不如直接在 base 模型上跑 RL。

**核心矛盾**：SFT 阶段的优化目标（post-SFT 准确率）和后训练的真正目标（post-RL 准确率）之间存在系统性错位。SFT 分数会偏向**更简单、更同质、重复**的数据——这恰恰是很多 SFT 数据选择方法偏好的方向——于是「优化 SFT 分数」反而把模型推向不利于后续 RL 探索的方向。而 RL 本身极贵（每次跑 3 epoch、单次最多 5 天），又无法靠 early stopping 提前判断，导致后训练团队之间反复扯皮、迭代效率低下，这就是标题里的「quagmire（泥潭）」。

**本文目标**：把问题聚焦在「可预测性」——能不能在不真正花钱跑 RL 的前提下，提前判断一个 post-SFT 模型在 RL 后的潜力？分解为两个研究问题：RQ1：pre-RL 表现更好的模型是否总能在 RLVR 后更好？若不能，失败模式是什么？RQ2：考虑后续 RL 时，什么样的 SFT 范式/数据配方更好，能否在烧 RL 之前就判断 SFT 模型是否合适？

**核心 idea**：抛弃「post-SFT 准确率」这个误导性指标，改用两个能反映模型「潜力」而非「当前成绩」的代理指标——验证集泛化损失（捕捉过拟合）和 Pass@大 k（捕捉模型固有的解题能力上限），用它们来预测和排序 RL 结果。

## 方法详解

### 整体框架
这是一篇分析+方法论的论文，没有提出新的训练算法，而是提出一套「先诊断、再换指标」的预测方法论。整体分三步：第一步系统揭示「SFT 指标陷阱」，在**数据集级**（同分布、变 epoch 数/样本量/学习率）和**实例级**（固定训练配置、换不同数据集/数据筛选策略）两类场景下证明 post-SFT 表现和 post-RL 表现会明显背离；第二步提出两个新预测指标——验证集泛化损失和 Pass@大 k——来替代 post-SFT 准确率；第三步给出实际工作流：先用泛化损失快速滤掉明显差的候选，再用 Pass@大 k 对剩下的排序选最优，必要时在少数 SFT 模型上跑 RL 校准一个线性预测器来估计绝对数值。整套方法的价值在于「不跑（或少跑）RL 就能挑出最该进 RL 的 SFT 模型」。

### 关键设计

**1. SFT 指标陷阱：用数据集级与实例级反例拆穿「SFT 高分=RL 好」**

这是全文的诊断基础，针对的痛点正是「把 SFT 与 RL 分开各自优化」。作者设计两类对照场景：**数据集级**保持 SFT 样本来自同一分布，只改训练配置（独特样本数、epoch 数、学习率），对应工业界「数据有限时多刷几个 epoch vs 数据充足时单 epoch 过全量」的真实抉择；**实例级**固定模型和训练流程，只换 SFT 数据集（如挑最短/最长/随机子集及其混合），对应「SFT 数据选择与策展」。在数据集级，作者拟合 post-SFT 与 post-RL 表现的线性关系，发现 $R^2 = 0.43$——post-SFT 表现只能解释 RL 后成绩 43% 的方差，剩下的鸿沟非常明显；并观察到过训练（最多 8 epoch）能持续抬高 SFT 分数，但 RL 后成绩在 2 epoch 后就饱和甚至（Qwen3）退化。在实例级，只训最短样本让 SFT 涨得最快，但因为这些样本太接近模型原始输出、过于好学，RL 后成绩显著更低。这一节用真实曲线说明：post-SFT 表现会被同质/重复/简单数据系统性地「带偏」，不可靠。

**2. 泛化损失：用验证集 loss 的「翘头」捕捉过训练、预测 RL 潜力**

针对数据集级场景里「过训练害了 RL」这个失败模式，作者提出用 SFT 阶段在留出验证集上的泛化损失作为潜力指标。直觉是：随着 epoch 增多，post-SFT 准确率仍在涨，但验证集泛化损失会先降后升、最终「flare up（骤升）」，这正是强过拟合的信号；而这个 loss 上升与后续 RL 的进一步增益呈强相关。实践用法是：跑完若干不同样本量/epoch 的 SFT 后，**立刻把「准确率又低、泛化损失又高」的模型剔除**，因为它们在 RL 后大概率仍然差，从而高效定位最优 SFT 训练范式。注意它的局限：只适用于数据集级（同分布）选择，一旦换不同数据集，验证 loss 里混入了分布差异分量，无法公平比较。

**3. Pass@大 k：用模型固有解题能力上限预测 RL 成绩**

针对实例级（跨数据集）选择，作者改用 Pass@大 k。出发点来自一个关于 RLVR 的认识：GRPO 的目标是最大化期望奖励、本质上是把 Pass@k 的能力「压缩」成 Pass@1——GRPO 只在一组采样里至少有一个正确才前进，因此 RL 动态与原模型的 Pass@k 强耦合。于是 post-SFT 模型在大 k 下的 Pass@k 更能反映它「天花板有多高、RL 能榨出多少」。为高效估计，作者采用无偏估计公式：对一道题生成 $n$ 个回答、其中 $c$ 个正确，则

$$\text{Pass@}k = \mathbb{E}\left[1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right]$$

对所有 $k \le n$ 都成立。作者用 Pass@大 k 最高的 post-SFT 模型来预判它 RL 后 Pass@1 最高，**完全不需要真跑 RL**。因为它衡量的是模型内在的产生正确解的能力，对训练数据的分布漂移不敏感，所以在实例级排序中尤其稳健，也是泛化损失失效时的替代方案。

### 损失函数 / 训练策略
论文不引入新损失。SFT 用标准监督微调；RL 用 GRPO（Group Relative Policy Optimization）做 RLVR，奖励为数学答案可验证的对/错。预测指标侧：泛化损失即 SFT 数据验证集上的 loss；Pass@大 k 用上面的组合数无偏估计公式（实验取 $k=64$、每题 256 次重复采样）。绝对数值预测时，在少量 SFT 模型上实跑 RL 收集校准点，再以这些指标为自变量拟合线性预测器。

## 实验关键数据

### 主实验
规模：训练上百个最大 12B 的模型，覆盖 Llama3-8B-Instruct、Mistral-Nemo-12B-Instruct、Qwen3-4B-base，SFT 数据用 Llama-Nemotron-SFT / AceReasoner1.1-SFT，RL 数据用 MATH(train) / DeepScaleR，在 7 个数学 benchmark（MATH-500、AIME、GSM8k、AMC、Olympiad、Minerva 等）上用 Pass@1（64 次重复平均）评测，总计 >100 万 A100 GPU 小时。

数据集级预测（Table 1/2，对比预测指标）：

| 模型 | 指标 | post-SFT Pass@1 基线 | 泛化损失 | Pass@大 k (k=64) | 两者平均 |
|------|------|------|------|------|------|
| Llama3-8B | Spearman | 0.75 | 0.94 | 0.95 | 0.97 (+0.22) |
| Mistral-NeMo-12B | Spearman | 0.78 | 0.90 | 0.92 (+0.14) | 0.90 |
| Llama3-8B | $R^2$ | 0.57±0.29 | 0.88±0.09 | 0.87±0.10 | 0.94±0.04 (+0.37) |
| Mistral-NeMo-12B | $R^2$ | 0.29±0.38 | 0.79±0.26 (+0.50) | 0.57±0.32 | 0.72±0.24 |

实例级预测（Table 3/4，泛化损失因跨分布不适用，只比 Pass@大 k）：

| 模型 | 指标 | post-SFT Pass@1 基线 | Pass@大 k (k=64) |
|------|------|------|------|
| Llama3-8B | Spearman | 0.69 | 0.94 (+0.25) |
| Mistral-NeMo-12B | Spearman | 0.70 | 0.98 (+0.28) |
| Llama3-8B | $R^2$ | 0.58±0.20 | 0.92±0.05 (+0.34) |
| Mistral-NeMo-12B | $R^2$ | 0.73±0.16 | 0.98±0.01 (+0.25) |

### 消融实验
论文没有传统意义的模块消融，而是用「指标对比」充当消融——比较三种预测信号在两类场景下的有效性：

| 配置 | 数据集级 | 实例级 | 说明 |
|------|---------|--------|------|
| post-SFT Pass@1（基线） | $R^2$ 低至 0.29 | $R^2$≈0.58–0.73 | 直接用 SFT 分数预测，方差大、不可靠 |
| 泛化损失 | $R^2$ 最高 +0.50 | 不适用 | 同分布下精度最佳，跨数据集失效 |
| Pass@大 k | Spearman≥0.92 | $R^2$ 高至 0.98 | 排序最稳，对分布漂移鲁棒 |

### 关键发现
- **Pass@大 k 是最稳的排序工具**：两类场景 Spearman 都 ≥0.90（最高 0.98），在实例级 $R^2$ 提升达 59%；因为它衡量模型固有能力，对训练数据分布漂移不敏感。
- **泛化损失在数据集级精度最高但有适用边界**：同分布下 $R^2$ 提升最高达 0.50（Mistral 从 0.29→0.79），但跨数据集时验证 loss 混入分布差异，无法公平比较，作者明确指出不适用于实例级选择。
- **「半量数据×两 epoch」常胜过「全量数据×单 epoch」**：相同 SFT 算力预算下，多数实验里前者无论 SFT 后还是 SFT-then-RL 后都更优；只训短样本能拿到更高 SFT 分数，却在 RL 后更差——这些都被新指标捕捉到、却被 post-SFT 分数掩盖。
- **不同模型对 SFT 反应迥异**：Mistral 的 post-SFT 与 post-RL 同涨，Qwen3 则二者几乎无关（RL 后成绩不随 SFT 提升而变），说明单看 SFT 分数无法跨模型外推。

## 亮点与洞察
- **把「指标选错」当成一类系统性失败来研究**：作者不是修某个训练技巧，而是指出整个行业默认的评估代理（post-SFT 准确率）本身有偏，这种「质疑度量」的视角比再调一个超参更有杠杆。
- **Pass@大 k ↔ GRPO 目标的耦合论证很漂亮**：用「GRPO 把 Pass@k 压缩成 Pass@1」这一机制解释为什么大 k 的 Pass@k 能预测 RL 上限，并用组合数无偏估计让它在 $k\le n$ 时高效可算，理论直觉和工程实现都到位。
- **可直接迁移的工程价值**：「先用泛化损失粗筛、再用 Pass@大 k 精排、必要时少量 RL 校准线性预测器」是一套即插即用的 SOP，能在不烧满 RL 的前提下挑模型，给昂贵的后训练流水线显著降本。
- **诚实标注指标边界**：明确说泛化损失不能跨分布用，没有把单一指标包装成万能解，这种 caveat 让方法更可信。

## 局限与展望
- 作者承认：研究只覆盖**数学推理**，能否推广到代码、科学、agentic 等任务待验证。
- 只研究了主流的**在线 RL + GRPO**；offline RL / DPO / 其他 RL 算法下 SFT 特征与 post-RL 表现的关系可能不同。
- **Pass@大 k 评测本身贵**：直接评需要至少重复 k 次采样，长序列下成本高；用小 k 外推估计大 k 是潜在的提效方向。
- 自己发现的局限：横向比较的 $R^2$/Spearman 是在各自模型+数据组合下测的，不同任务难度/算力预算不可直接比大小；上百个模型虽多，但仍集中在 3 个 backbone、少数 SFT/RL 数据集组合，结论的普适性需更多模型谱系验证。

## 相关工作与启发
- **vs 「SFT 冷启动必要」（DeepSeek-R1）**：R1 主张冷启动 SFT 对后续 RL 必要；本文不否定 SFT，而是指出「SFT 做到分数最高」未必是最好的 RL 起点，把焦点从「要不要 SFT」转到「怎么挑 SFT 模型」。
- **vs 「过 SFT 会约束 RL」（Llama-4 报告）/「SFT 泛化差、纯 RL 更好」（Chen et al. 2025a）**：这些工作观点相互矛盾，本文用统一的可预测性框架把这些零散现象（过训练害 RL、简单数据害 RL）解释为「指标偏差」的不同表现，并给出可操作的预测指标，而非再加一句定性论断。
- **vs 基于难度/影响力的 SFT 数据选择（Muennighoff et al. 2025 等）**：这些方法仍以 SFT 后表现为优化目标；本文指出该目标会偏向简单同质数据，应改用泛化损失/Pass@大 k 作为筛选信号。

## 评分
- 新颖性: ⭐⭐⭐⭐ 不是新算法，但「质疑 SFT 指标 + 提出两个可预测代理」的视角扎实且少见
- 实验充分度: ⭐⭐⭐⭐⭐ 上百模型、3 个 backbone、7 benchmark、>100 万 GPU 小时，规模与说服力都很强
- 写作质量: ⭐⭐⭐⭐ 问题驱动、反例清晰，指标适用边界标注诚实
- 价值: ⭐⭐⭐⭐⭐ 直击工业后训练「烧 RL 前不知道该挑谁」的真实痛点，SOP 可直接落地

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] SFT Overtraining Predicts Rank Inversion via Entropy Collapse Under RLVR](../../ICML2026/llm_alignment/sft_overtraining_predicts_rank_inversion_via_entropy_collapse_under_rlvr.md)
- [\[ICLR 2026\] Spectrum Tuning: Post-Training for Distributional Coverage and In-Context Steerability](spectrum_tuning_post-training_for_distributional_coverage_and_in-context_steerab.md)
- [\[ICLR 2026\] Fluent Alignment with Disfluent Judges: Post-training for Lower-Resource Languages](fluent_alignment_with_disfluent_judges_post-training_for_lower-resource_language.md)
- [\[ICLR 2026\] Chasing the Tail: Effective Rubric-based Reward Modeling for Large Language Model Post-Training](chasing_the_tail_effective_rubric-based_reward_modeling_for_large_language_model.md)
- [\[CVPR 2025\] Continual SFT Matches Multimodal RLHF with Negative Supervision](../../CVPR2025/llm_alignment/continual_sft_matches_multimodal_rlhf_with_negative_supervision.md)

</div>

<!-- RELATED:END -->
