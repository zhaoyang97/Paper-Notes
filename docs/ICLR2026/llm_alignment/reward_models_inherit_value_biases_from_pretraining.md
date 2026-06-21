---
title: >-
  [论文解读] Reward Models Inherit Value Biases from Pretraining
description: >-
  [ICLR 2026][LLM对齐][奖励模型] 这篇论文用"穷举 token 搜索 + 心理语言学语料"的可解释性方法系统检查了 10 个主流开源奖励模型（RM），发现 RM 在"能动性 vs 共融性"等多个人类价值维度上的偏好高度取决于它的基座 LLM（Llama 系偏好 agency、Gemma 系偏好 communion），并把这种偏见一路溯源到基座模型的对数概率、证明它在偏好微调过程中很难被"洗掉"。
tags:
  - "ICLR 2026"
  - "LLM对齐"
  - "奖励模型"
  - "价值偏见"
  - "预训练"
  - "可解释性"
  - "心理语言学"
---

# Reward Models Inherit Value Biases from Pretraining

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=dT399j1Azv](https://openreview.net/forum?id=dT399j1Azv)  
**领域**: 对齐RLHF  
**关键词**: 奖励模型, 价值偏见, 预训练, 可解释性, 心理语言学

## 一句话总结
这篇论文用"穷举 token 搜索 + 心理语言学语料"的可解释性方法系统检查了 10 个主流开源奖励模型（RM），发现 RM 在"能动性 vs 共融性"等多个人类价值维度上的偏好高度取决于它的基座 LLM（Llama 系偏好 agency、Gemma 系偏好 communion），并把这种偏见一路溯源到基座模型的对数概率、证明它在偏好微调过程中很难被"洗掉"。

## 研究背景与动机

**领域现状**：奖励模型是 RLHF / DPO 把 LLM 对齐到人类价值的核心部件，但相比预训练和后训练的 LLM 本身，RM 一直被研究得很少。近年随着开源偏好数据、开源 RM 权重和 RewardBench 这类公开基准的出现，针对 RM 的可解释性研究才逐渐起步。

**现有痛点**：已有的 RM 可解释性工作大多关注两件事——要么是如何"主动"用 RM 把后训练模型推向特定偏好（个性化），要么是 RM 如何"无意中"给后训练 LLM 引入偏见。但所有这些工作都把 RM 当成偏见的"传播者"，没人去问一个更前置的问题：RM 本身是从某个 LLM 初始化、再做偏好微调得到的，那么**这个基座 LLM 会不会把自己的价值偏见传染给 RM**？

**核心矛盾**：RM 的设计目标是表征"人类偏好"，理论上它的输出应该只反映偏好数据。但 RM 在结构上继承了基座 LLM 的全部表征（甚至直接复用其权重），这就埋下一个矛盾：RM 的打分到底有多少来自偏好数据、多少来自基座模型的"先天倾向"？如果是后者占了相当比重，那么"换基座 = 换价值观"，而这一点在开源社区选基座时几乎没人当成价值问题考虑。

**本文目标**：拆成三个递进的子问题——(1) 野生的开源 RM 是否真的会按基座分化出系统性的价值差异？(2) 如果会，这种差异能否溯源到基座 LLM 本身（指令微调版乃至预训练版）？(3) 在可控训练下，喂多少偏好数据才能把这种"先天偏见"冲淡，还是说它根本冲不掉？

**切入角度**：作者借用 Christian et al. (2025) 提出的"穷举 token 搜索"——对一个价值导向的 prompt，遍历整个词表给每个 token 打 RM 分，看哪些 token 得分最高/最低，从而读出 RM 的"价值偏好"。再把这些 token 映射到心理学专家标注过的价值维度（如 Big Two 的能动性/共融性），就能把模糊的"价值观"量化成可统计的排名。

**核心 idea**：把 RM 的逐 token 打分接到心理语言学语料上做"价值体检"，再把同一套方法用到基座 LLM 的对数概率上，证明 RM 的价值偏见是从预训练阶段就埋下、并在偏好微调中顽固存留的"遗传病"。

## 方法详解

### 整体框架
这篇是分析型论文，没有要训练的新模型，它的"方法"就是一整套递进的诊断流程，对应正文三个 section：**先在野生 RM 上量出基座层面的价值差异 → 再把差异溯源到基座 LLM 的对数概率（并构造一个"隐式 RM"来刻画两基座之差）→ 最后用自己从零训练的一批 RM 做受控实验，观察这种偏见在训练中如何演化、能否被更多数据洗掉**。

整条链路的输入是"价值导向 prompt + 一个奖励模型/语言模型"，输出是"该模型在某个价值维度（如能动性/共融性）上的偏好强度（中位数排名）"。核心测量工具贯穿始终：对一个 prompt 遍历整个词表算分（RM 用 reward 分，LLM 用对数概率），把得到的逐 token 排名按心理语言学语料归并到价值类目，再做混合效应统计检验。三个阶段层层加码，从"现象存在"一直追到"根因在预训练、且难以消除"。

### 关键设计

**1. 穷举 token 搜索 + 心理语言学语料：把"价值观"变成可统计的排名**

野生 RM 的价值偏好看不见摸不着，作者的做法是对一个价值导向 prompt（如"What, in one word, is the greatest thing ever?"）遍历 RM 的整个词表，给每个 token 当作回答打一个 reward 分，得到全词表的排名。光有排名还不够，作者把它接到两个经心理学专家校验过的语料上：Big Two（263 个词，编码"能动性 agency"如 freedom/success/ability 与"共融性 communion"如 love/family/friendship）和道德基础词典 MFD2（编码 authority/care/fairness/loyalty/sanctity 五个维度）。把"词级 reward"按语料归并成"类目级 reward"，价值偏好就变成了"某类目词的中位数排名"这样一个可统计的量。

为了让结论稳健，作者在 10 个 RewardBench 上的领先 RM（基座要么 Gemma、要么 Llama）上，用 54 个 prompt 变体（27 个正向措辞 + 27 个负向措辞，如"the worst thing ever"）做评测，用混合效应线性模型把"基座选择"作为关键因子。结果很干净：正向 prompt 下 Llama 系 RM 把 agency 词排得更靠前、Gemma 系把 communion 词排得更靠前，负向 prompt 下完全反转（Big Two 类目 × 基座 × 措辞极性的三因子交互 $p < .001$，效应量 Cohen's $d \approx 0.40\text{–}0.43$，属中等效应）。这个偏见还会传到下游：top-k 分析里 Gemma RM 的前 10 高分 token 平均有 5 个是 communion 词、0 个 agency 词，Llama RM 则是 3.67 个 communion + 2.33 个 agency。

**2. 隐式奖励模型与 MWLR 分数：把"两个基座之差"本身当成一个 RM 来读**

证明了野生 RM 有差异后，作者要把根因往前推到基座 LLM。第一步直接看指令微调版 Gemma 2 2B 和 Llama 3.2 3B 给每个 Big Two 名词的对数概率，发现和 RM 一模一样的 agency/communion 分裂（三因子 ANOVA $F(1,208)=58.3$, $p<.001$），而且在**预训练版**上同样成立（$F(1,208)=43.2$, $p<.001$）——说明偏见早在预训练就埋下了。

更巧的一步是：作者把"两个基座之差"本身构造成一个奖励模型来分析。RLHF 的数学告诉我们，微调后的模型可写成 $\pi_r(y|x) = \frac{1}{Z_x}\,\pi_{\text{base}}(y|x)\exp(\beta\cdot r(x,y))$；反过来，对任意两个模型 $\pi_1, \pi_2$，后者都能被看成前者按某个"隐式奖励"微调的结果，这个隐式奖励正是对数概率之差 $r_{1\to2}(x,y) = c(x) + \beta\cdot\log\frac{\pi_2(y|x)}{\pi_1(y|x)}$。于是只要对这个 log 差做穷举 token 搜索，就能读出"从 Gemma 变到 Llama 最该奖励/最该惩罚哪些 token"。但原始 log 差有个毛病：低概率长尾 token 在 log 空间是很大的负值，相减后会让一堆两个模型都根本不会输出的"垃圾 token"占据极值。作者用**混合权重对数比 MWLR** 解决：

$$\text{MWLR} = \tfrac{1}{2}(p+q)\cdot(\log q - \log p),$$

其中 $p \equiv \pi_1(\cdot|x)$、$q \equiv \pi_2(\cdot|x)$。前面的混合概率权重 $\frac{1}{2}(p+q)$ 保证只有至少一个模型赋予非可忽略概率的 token 才会被放大，过滤掉垃圾极值。作者还做了验证：人为对 Gemma 做监督微调注入 10 个 authority 词造一个"威权版"，比较各候选指标谁最能把这些注入词找回来，MWLR 的灵敏度胜过所有对比指标。用 MWLR 算"隐式 Gemma→Llama RM"，最优 token 是"Freedom"、最差 token（去掉 Markdown 格式后）正是"Love"，与前面 RM 的结论惊人一致；把它推广到所有（<405B）Llama 3 与 Gemma 2 指令模型的 21 组两两比较，"Freedom > Love"在全部 21 组成立，且差距随模型规模增大。

**3. 受控 RM 训练实验：偏见随训练演化但冲不干净**

最后作者从零训练自己的 RM 来观察偏见的动态。为排除"是不是某个偏好数据集特有的"，他们用两个不重叠的数据集（Skywork ≈77k、Unified Feedback ≈850k）、各自从 Llama 3.2 3B Instruct 和 Gemma 2 IT 2B 初始化，**所有超参完全一致**（2 epoch、LoRA rank=32/α=64、AdamW lr=1e-5、有效 batch 16、Bradley-Terry 损失、固定随机种子），每 1000 步存一个 checkpoint 做穷举 token 搜索，从而画出偏见随训练步数（模型内）和数据量（模型间）的演化曲线。

三个发现：其一，与前文一致，Llama RM 始终把 agency 排得更高、Gemma RM 把 communion 排得更高；其二，Gemma 与 Llama 之间的差距在训练开始时最大，随后逐渐收窄；其三也是最关键的——**这个差距收窄但不闭合**，在训练约三分之一处就稳定下来。数据量消融显示：数据来源影响不大，但更多偏好数据确实能缓解偏见，大约需要 100k 以上的偏好对才能抹平 Gemma 与 Llama 的差异。不过作者给了两个 caveat：这里只测了 Big Two 两个维度，多维价值空间可能需要更多数据；而且只测了两个基座——附录里基于 Qwen 的探索性实验显示，即便训练 100k 偏好，Qwen 与 Gemma/Llama 之间的差距依然不闭合。更进一步，Yang et al. (2024) 的 GRM（保留语言头并加正则保护生成能力）即便训练超过 630k 偏好，agency/communion 鸿沟依然显著，说明方法学选择会让基座偏见持续得更久。

## 实验关键数据

### 主实验

| 分析对象 | 现象 | 关键统计 |
|--------|------|---------|
| 10 个野生 RM（Big Two，正向 prompt） | Llama 偏 agency、Gemma 偏 communion；负向 prompt 反转 | 三因子交互 $p<.001$，$d\approx0.40\text{–}0.43$ |
| 指令微调版 Gemma 2 2B vs Llama 3.2 3B（对数概率） | 同样的 agency/communion 分裂 | $F(1,208)=58.3$, $p<.001$ |
| 预训练版 Gemma 2 2B vs Llama 3.2 3B | 偏见在预训练阶段已存在 | $F(1,208)=43.2$, $p<.001$ |
| 隐式 Gemma→Llama RM（MWLR） | 最优 token = "Freedom"，最差 = "Love" | 21/21 组比较中 Freedom > Love |

### 消融实验

| 配置 | 关键现象 | 说明 |
|------|---------|------|
| 训练动态（Skywork，每 1000 步 checkpoint） | 起点差距最大 → 收窄 → 约 1/3 处稳定不再闭合 | 偏见持久存在 |
| 数据来源（Unified Feedback vs Skywork） | 来源影响很小 | 偏见不是某数据集特有 |
| 数据量（13k/27k/53k/77k/106k） | ~100k+ 偏好对才能抹平 Gemma/Llama 差距 | 更多数据可部分缓解 |
| 第三方基座（Qwen，附录） | 训练 100k 后差距仍不闭合 | 双基座结论的局限提醒 |
| GRM（保留语言头 + 正则，630k+） | 鸿沟依然显著 | 方法学选择延长偏见寿命 |

### 关键发现
- **基座是价值偏见的真正源头**：同样的偏好数据、同样的微调流程下，只换基座就能稳定改变 RM 的价值偏好，说明偏见来自基座而非偏好数据，这把"选基座"从纯性能问题变成了价值问题。
- **偏见在预训练就埋下、且很顽固**：从野生 RM → 指令微调 LLM → 预训练 LLM 一路都能看到同一个 agency/communion 分裂；训练中差距收窄但不闭合，约 100k 偏好对才能在两个测试维度上抹平，多维或第三方基座下更难。
- **MWLR 的混合权重很关键**：直接用 log 差会被低概率垃圾 token 主导，$\frac{1}{2}(p+q)$ 权重把注意力限制在两模型真正会输出的 token 上，灵敏度实验证明它优于所有对比指标。

## 亮点与洞察
- **把"两个模型之差"当成一个隐式 RM 来读**，是非常优雅的视角迁移：RLHF 的数学本就把"微调"刻画成对基座乘一个 $\exp(\beta r)$，反过来任意两模型的 log 概率差就是隐式奖励，于是同一套穷举 token 搜索可以无缝用到 RM 和 LLM 上。
- **借心理语言学语料给 AI 模型做"价值体检"**，把模糊的"价值观"落到专家标注、可统计检验的维度上，这套思路可迁移到测 LLM 的政治倾向、文化偏好等其他价值轴。
- **"Freedom vs Love"这个对照极具画面感**：同一个"最伟大的东西是什么"的 prompt，Llama 系给"自由"、Gemma 系给"爱"，而且这恰好出现在无约束穷举指标的两个极值上，暗示这可能是两个模型族之间最大的差异之一。

## 局限与展望
- 作者承认只测了 Big Two 两个价值维度和两个主力基座（Gemma/Llama），多维价值空间和更多基座（如 Qwen）下，"~100k 数据可抹平"的结论很可能不成立。
- 受控训练实验用的是标准 Bradley-Terry 损失，而 GRM 这类保留语言头的方法会让基座偏见持续更久，论文只是观察到现象、未深入拆解正则与基座偏见的交互机制。
- 这是诊断性工作，给出了"偏见在预训练、难洗掉"的证据，但没有提出具体的"去偏"训练方法——下一步自然是设计能在偏好微调阶段主动中和基座价值偏见的机制，或把对齐努力前移到预训练阶段。

## 相关工作与启发
- **vs 已有 RM 可解释性工作**：以往工作把 RM 当成偏见的传播者（主动个性化 / 无意引入偏见），本文反过来追问 RM 自身被基座偏染，指出正则化只解决了一半问题，因为 RM 本身就把偏见直接写进了后训练的奖励信号。
- **vs LLM 价值量化（问卷/选择题式）**：Rozado、Santurkar 等用问卷测后训练 LLM 的政治/道德倾向，本文改用心理语言学专家语料、且把镜头对准 RM 而非 LLM，互为补充。
- **vs 模型多样性（model multiplicity）**：Black et al. 指出性能相近的模型内部表征可天差地别，本文的基座差异符合这一框架，但更进一步证明这是"模型族层面"的系统性、持久性差异——跨小版本、跨两个数量级规模都稳定存在，而非随机种子级的特异性偏好。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次把 RM 价值偏见溯源到预训练，并用"隐式 RM + 心理语言学"给出可量化证据
- 实验充分度: ⭐⭐⭐⭐ 10 个野生 RM + 指令/预训练 LLM + 受控训练 + 数据消融，覆盖全面，但价值维度和基座种类偏少
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑层层递进（现象→溯源→动态），"Freedom vs Love"对照清晰有力
- 价值: ⭐⭐⭐⭐⭐ 把"选基座"提升为价值层面的安全考量，强调对齐努力应前移到预训练

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Pretrain Value, Not Reward: Decoupled Value Policy Optimization](pretrain_value_not_reward_decoupled_value_policy_optimization.md)
- [\[ICLR 2026\] Cognitive models can reveal interpretable value trade-offs in language models](cognitive_models_can_reveal_interpretable_value_trade-offs_in_language_models.md)
- [\[AAAI 2026\] BiasJailbreak: Analyzing Ethical Biases and Jailbreak Vulnerabilities in Large Language Models](../../AAAI2026/llm_alignment/biasjailbreakanalyzing_ethical_biases_and_jailbreak_vulnerabilities_in_large_lan.md)
- [\[ACL 2025\] Internal Value Alignment in Large Language Models through Controlled Value Vector Activation](../../ACL2025/llm_alignment/internal_value_alignment_in_large_language_models_through_controlled_value_vecto.md)
- [\[ICLR 2026\] Eliminating Inductive Bias in Reward Models with Information-Theoretic Guidance](eliminating_inductive_bias_in_reward_models_with_information-theoretic_guidance.md)

</div>

<!-- RELATED:END -->
