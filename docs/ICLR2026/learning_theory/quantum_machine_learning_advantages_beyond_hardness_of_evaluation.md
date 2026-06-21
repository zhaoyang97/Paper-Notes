---
title: >-
  [论文解读] Quantum Machine Learning Advantages Beyond Hardness of Evaluation
description: >-
  [ICLR 2026][学习理论][PAC 学习] 本文首次证明：对于由量子函数（BQP-complete）标注的数据，即便不要求模型去"评估"新样本、只要求"识别"出标注函数本身，经典算法也做不到——除非 $\mathsf{BQP}$ 落在多项式层级的低层（一个被普遍认为不成立的塌缩），从而把量子机器学习的优势从"评估难"推进到"学习过程本身难"。
tags:
  - "ICLR 2026"
  - "学习理论"
  - "量子机器学习"
  - "计算复杂度"
  - "PAC 学习"
  - "学习分离"
  - "BQP"
  - "多项式层级"
---

# Quantum Machine Learning Advantages Beyond Hardness of Evaluation

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=on2lie43Kl](https://openreview.net/forum?id=on2lie43Kl)  
**代码**: 无（纯理论论文）  
**领域**: 学习理论 / 量子机器学习 / 计算复杂度  
**关键词**: 量子机器学习, PAC 学习, 学习分离, BQP, 多项式层级

## 一句话总结
本文首次证明：对于由量子函数（BQP-complete）标注的数据，即便不要求模型去"评估"新样本、只要求"识别"出标注函数本身，经典算法也做不到——除非 $\mathsf{BQP}$ 落在多项式层级的低层（一个被普遍认为不成立的塌缩），从而把量子机器学习的优势从"评估难"推进到"学习过程本身难"。

## 研究背景与动机

**领域现状**：量子机器学习（QML）这几年有了一批严格的"量子优势"证明，典型套路是：数据由某个密码学函数或量子函数 $f$ 标注，而经典多项式电路无法高效计算 $f$，于是经典学习器即使学到了"模型"也没法对新输入打标签，量子学习器却可以。这类结果的优势来源是**评估难**（hardness of evaluation）。

**现有痛点**：评估难这件事其实和"学习"关系不大——它只说明经典电路算不出 $f$，并没有说明"从数据里把 $f$ 认出来"这一步有多难。换句话说，已有的量子优势可能全都来自最后那一步"代入新点求值"，而不是来自学习过程。这让人不满意：我们想知道的是，**学习本身**（从带标签样本里恢复出标注函数）是不是也存在量子优势。

**核心矛盾**：要把"学习难"和"评估难"剥离开，自然的思路是只考察 **identification（识别）任务**——给定一堆 $(x, f(x))$，学习器只需输出标注函数 $f$ 的一个描述（即 $f$ 的索引 $\alpha$），完全不要求它能对新 $x$ 求值。这样评估难就被排除在外，剩下的难度只能来自学习。但这里有个根本障碍：已有的识别难证明（密码学函数、DNF 公式）全都依赖 **random generatability（随机可生成性）**——能高效地为随机输入 $x$ 生成带标签样本 $(x, f(x))$。可惜对量子函数，作者猜测（并在本文证明）这个性质**不成立**，于是旧的证明框架直接失效，量子函数的识别优势一直是空白。

**本文目标**：在合理的复杂度假设下，给出量子函数识别任务的**首个**经典难度证明，并据此构造一个真实的量子–经典学习分离。

**切入角度**：既然量子函数不可随机生成、旧证明走不通，就需要一条全新的证明路径——不靠"生成样本"，而是把"假设存在经典识别算法"这件事，通过 NP 神谕反过来用，构造出一个能判定 BQP 语言的经典机器，从而把 $\mathsf{BQP}$ 压进多项式层级，得到矛盾。

**核心 idea**：用"**反演识别算法 + 沿多项式层级向上爬**"代替"随机生成样本"，证明对一大类量子概念类（c-distinct 或 average-case-smooth），经典识别难，除非 $\mathsf{BQP}\subseteq\mathsf{BPP}^{\mathsf{NP}^{\mathsf{NP}^{\mathsf{NP}}}}$。

## 方法详解

### 整体框架

本文是一篇纯复杂度理论论文，"方法"就是一条环环相扣的归约链，目标是把"经典能识别量子函数"这个假设，逐步转化为"$\mathsf{BQP}$ 塌进多项式层级"这种被普遍否定的结论。整条链分三段递进：

**第一段（铺垫，排除旧路）**：先证明量子函数**不可随机生成**——若存在经典算法能为均匀随机 $x$ 高效产出 $(x, f(x))$，则 $\mathsf{BQP}\subseteq \mathsf{P}^{\mathsf{NP}}$（精确版）或 $(\mathsf{BQP},\mathrm{Unif})\in\mathsf{HeurBPP}^{\mathsf{NP}}$（近似版）。这一步的意义是"破旧"：它说明已有识别难证明赖以成立的核心工具对量子函数失效，因此必须另起炉灶。

**第二段（verifiable 识别，热身的强假设版）**：考察"可验证识别"——算法不仅要识别出 $\alpha$，还要能拒绝那些和任何概念都不一致的非法数据集。在这个较强假设下证明：经典可验证识别量子函数 $\Rightarrow (\mathsf{L},\mathrm{Unif})\in\mathsf{HeurBPP}^{\mathsf{NP}}$。这里"可拒绝非法数据"这个"当且仅当"的性质，正是反演归约能跑通的关键。

**第三段（non-verifiable 识别，主结果）**：把"可验证"这个强假设拿掉，只要求一个满足两条温和附加条件的 proper PAC 学习器（approximate-correct identification）。代价是归约要在多项式层级里多爬两层：经典识别 $\Rightarrow (\mathsf{L},\mathrm{Unif})\in\mathsf{HeurBPP}^{\mathsf{NP}^{\mathsf{NP}^{\mathsf{NP}}}}$。最后再给出一个物理动机的具体 c-distinct 概念类（学习可观测量 / 哈密顿量学习相关），它满足主定理条件，从而落地为一个真实的量子–经典识别分离。

输入是"由某个量子函数 $f^\alpha$ 标注的训练集 $T=\{(x_\ell,y_\ell)\}$"，输出是定理形式的结论"经典若能识别则 $\mathsf{BQP}$ 塌缩"。三段的强假设依次放松、结论里需要的 PH 层数依次升高，构成完整论证。

### 关键设计

**1. Identification 任务：把"学习"从"评估"里干净剥离**

要研究"学习过程本身有没有量子优势"，必须设计一个不含评估的学习任务，这就是识别任务。它把概念类写成 $F=\{f^\alpha:\{0,1\}^n\to\{0,1\}\mid \alpha\in\{0,1\}^m\}$（$m=\mathrm{poly}(n)$），学习器收到 $T=\{(x_\ell, f^\alpha(x_\ell))\}$ 后，只需输出标注它的索引 $\alpha$（或一个在 PAC 意义下与之接近的 $\tilde\alpha$，满足 $\mathbb{E}_{x\sim D}|f^\alpha(x)-f^{\tilde\alpha}(x)|\le\epsilon$），**不要求**对新 $x$ 求值。这正对应 proper PAC learning（假设类 $H$ 与概念类 $F$ 相同）。

这个设计是整篇论文的"地基"，因为它精准锁定了优势来源。作者特别强调：若不限制假设类，识别就毫无难度——经典学习器可以直接输出"把数据硬编码进去的整个量子学习器电路"当作答案，于是任何分离都被这个平凡解抹平。所以"$H=F$、只能在给定概念集合里指认"这个约束不是技术细节，而是让分离能够存在的前提。论文进一步区分两个版本：**verifiable**（Def. 7，必须能对非法数据集输出 "invalid"）和 **non-verifiable**（Def. 8，不拒绝任何数据，但满足两条附加条件：永不输出与全部样本都不符的 $\alpha$；数据完全一致时至少有一个随机串能输出 PAC 距离 $\le 1/3$ 的 $\alpha$）。

**2. 量子函数不可随机生成：先拆掉旧证明的承重墙**

旧的识别难证明（密码学函数）有两大支柱：标注函数有简洁表示、以及随机可生成性。random generatability 指存在高效经典算法 $A_D(f_n, r)=(x_r, f_n(x_r))$，能为随机输入产出带标签样本。本文证明这条对量子函数走不通：

$$\text{若 } f \text{ 是 BQP 函数且精确随机可生成，则 } \mathsf{BQP}\subseteq \mathsf{P}^{\mathsf{NP}}.$$

证明思路很巧：假设存在生成算法 $A$，构造 $A'$ 用 NP 神谕去**反演** $A$——给定要判定的输入 $\tilde x$，用 NP 神谕找到一个随机串 $\tilde r$ 使 $A(f_n,\tilde r)=(x_{\tilde r}, f_n(x_{\tilde r}))$ 且 $x_{\tilde r}=\tilde x$；由于 $A$ 是经典多项式时间，NP 神谕能高效验证 $\tilde r$ 是否正确，于是 $A'$ 就能对任意 $x$ 算出 $f_n(x)$，即在 $\mathsf{P}^{\mathsf{NP}}$ 里判定了任意 BQP 语言。近似版（Def. 2，允许在 $x$ 分布和标签上各犯 $\epsilon$ 错误）则得到 $(\mathsf{L},\mathrm{Unif})\in\mathsf{HeurBPP}^{\mathsf{NP}}$，靠的是用 Bellare 等人的 NP 神谕均匀采样多个 $\tilde r_i$ 再对 $f_n(x_{\tilde r_i})$ 取平均，把误分类比例压成 $\epsilon$ 的线性函数。这一结果顺带证明了某类量子生成模型（expectation value samplers）经典不可解（Corollary 2），并刻画了一族经典难以近似到逆多项式 TV 距离的分布。

**3. 反演 + 爬多项式层级：替代随机生成的新证明引擎**

这是全文的核心技术，思路是把"假设存在的识别算法 $A_B$"当成一个可被 NP 神谕反演的预言机来用。先看 verifiable 版（Thm 3）：$A_B$ 在数据集 $T$ 一致时输出接近 $\alpha$ 的 $\tilde\alpha$、不一致时输出 "invalid"，这个"当且仅当"让我们能用 NP 神谕**搜索**出一个数据集 $T$ 使 $A_B(T,\epsilon,\cdot)=\alpha$；拿到 $T$ 后，对任意新 $x$ 只需看 "$x$ 标成 0 还是 1 能让 $T\cup\{(x,y)\}$ 保持一致"，就完成了对 $f^\alpha$ 的求值，于是 $(\mathsf{L},\mathrm{Unif})\in\mathsf{HeurBPP}^{\mathsf{NP}}$。

主结果（Thm 5）把"可拒绝非法数据"这个强假设去掉，难点在于 non-verifiable 算法不再保证一致性，反演不能直接跑。作者分两步搭桥：先（Thm 13/14）证明对 c-distinct 或 average-case-smooth 概念类，一个 approximate-correct 识别算法能在 PH 第一层里被加工成一个"能拒绝含 $>1/\beta$ 比例错标样本的数据集"的算法；再（Thm 11）证明有了这种近似可验证算法，就能在它之上**再爬两层** PH 反演出一个绝大多数样本都被 $f^\alpha$ 正确标注的数据集，错标比例可压到多项式小。两步叠加，结论比 verifiable 版高两层：

$$\text{经典 approximate-correct 识别} \;\Rightarrow\; (\mathsf{L},\mathrm{Unif})\in\mathsf{HeurBPP}^{\mathsf{NP}^{\mathsf{NP}^{\mathsf{NP}}}}.$$

由于 $\mathsf{BQP}$ 被广泛相信不在 PH 低层，这个塌缩等于断言经典识别不可能，从而确立学习分离。

**4. 概念类结构条件（c-distinct / average-case-smooth）：让反演归约真正闭合**

为什么 non-verifiable 版需要额外的概念类条件？因为没有"拒绝非法数据"的保护，必须靠概念之间"足够不同"来保证反演出的数据集能唯一锁定目标概念。本文给出两类可用条件：

- **c-distinct（Def. 5）**：任意两个不同概念 $f^{\alpha_1}\ne f^{\alpha_2}$ 在至少 $c$ 比例的输入上取值不同，$|S|/2^n\ge c$。主定理要求 $c\ge 1/3$。直观上，概念彼此相距够远，错标一小撮样本也不会让识别器把目标认成别的概念。
- **average-case-smooth（Def. 6）**：标签空间带一个距离 $d$，使函数层面接近 $\Rightarrow$ 参数层面接近，$\mathbb{E}_{x\sim\mathrm{Unif}}|f^{\alpha_1}(x)-f^{\alpha_2}(x)|\ge C\, d(\alpha_1,\alpha_2)$。这与机器学习里常见的"参数接近 $\Rightarrow$ 函数接近"恰好反过来。

这两个条件不是可有可无的修饰：作者在 Discussion 指出，哈密顿量学习已知**经典可解**，而它能自然写成识别问题，这说明若把 c-distinct / average-case-smooth 这类结构假设彻底拿掉、还想保持识别难，就会和哈密顿量学习的经典可解性冲突，进而推出复杂度理论里出乎意料的后果。换言之，这些结构条件是分离能成立的必要边界，而非证明的偷懒。附录 F.2 给出了由量子可实现函数构成、满足这两个条件的具体概念类例子。

## 实验关键数据

本文是纯理论工作（Reproducibility Statement 明确说明无实验，全部为定理 + 附录完整证明）。这里以"结论强度对照表"代替实验表，把四条主结果按假设强弱与所得 PH 层数列出。

### 主结果对照

| 结果 | 任务 / 假设 | 经典若可行则推出 | 出处 |
|------|------------|------------------|------|
| Thm 1 | 精确随机可生成量子函数 | $\mathsf{BQP}\subseteq \mathsf{P}^{\mathsf{NP}}$ | §3 |
| Thm 2 | 近似随机可生成量子函数 | $(\mathsf{L},\mathrm{Unif})\in\mathsf{HeurBPP}^{\mathsf{NP}}$ | §3 |
| Thm 3 | 可验证识别（verifiable） | $(\mathsf{L},\mathrm{Unif})\in\mathsf{HeurBPP}^{\mathsf{NP}}$ | §4 |
| Thm 5 | 非验证识别（c-distinct $c\ge\tfrac13$ 或 avg-smooth） | $(\mathsf{L},\mathrm{Unif})\in\mathsf{HeurBPP}^{\mathsf{NP}^{\mathsf{NP}^{\mathsf{NP}}}}$ | §5 |

### 关键命题对照

| 命题 | 设定 | 结论 | 说明 |
|------|------|------|------|
| Prop 1 | singleton 概念类、可验证 | $f\in\mathsf{P/poly}$ | 单概念时"验证数据合法"已蕴含非均匀多项式电路可算 $f$ |
| Thm 4 | 存在多项式子集 $S$ 唯一确定 $f^\alpha$ | $\mathsf{BQP}\subseteq\mathsf{P/poly}$ | 把 $S$ 上标签当 advice，靠"保持数据集合法"逐点求值 |
| Cor 2 | expectation value samplers | 经典不可高效模拟 | Thm 2 的副产物，触及量子生成模型 |
| Cor 1 | 物理动机 c-distinct 概念类 | 量子–经典识别分离 | 落地的可分离实例（学习可观测量 / 哈密顿量相关）|

### 关键发现
- **优势来源被重新定位**：以往量子优势的证据可能都来自"评估难"，本文表明对量子函数而言，"识别 / 学习"这一步本身就难，整个学习流程而非仅最终求值都从量子计算获益。
- **新证明引擎的必要性**：Thm 1/2 证明量子函数不可随机生成，直接说明旧证明框架失效，"反演识别算法 + 爬 PH"是不得不发明的新路径——这是全文最重要的方法论贡献。
- **假设强度↔层级高度的权衡**：去掉"可验证"这个强假设，代价是结论里 PH 从 $\mathsf{NP}$ 一层涨到 $\mathsf{NP}^{\mathsf{NP}^{\mathsf{NP}}}$ 三层；越温和的算法假设，越需要更深的层级塌缩来构成矛盾。
- **结构条件不可轻易松弛**：哈密顿量学习经典可解，反衬出 c-distinct / average-case-smooth 是分离成立的必要边界，去掉它们会与已知经典可解性冲突。

## 亮点与洞察
- **把"假设的算法"当神谕反演**：本文最巧的招数是不去构造样本生成器，而是直接把"假设存在的识别算法 $A_B$"喂给 NP 神谕做反向搜索，搜出一个让 $A_B$ 输出目标 $\alpha$ 的数据集，再借数据集一致性给新点打标签——这把"学习器存在"转化成"求值能力"，思路可迁移到其他"只关心识别、不关心评估"的难度证明。
- **verifiable→non-verifiable 的两步搭桥**：先在 PH 第一层把 approximate-correct 算法升级成"近似可验证"（能拒绝错标过多的数据），再在其上多爬两层反演，是处理"算法不保证一致性"的通用模板。
- **一个负面引理撑起整篇论文**："量子函数不可随机生成"本身是个否定结果，却恰恰是证明"必须用新方法"的关键证据，把"为什么旧方法不行"写成了定理，论证闭环非常干净。
- **理论与物理任务挂钩**：把抽象的识别难度落到哈密顿量学习、学习序参量、学习可观测量这些真实量子任务上，让纯复杂度结果有了可感的应用入口。

## 局限与展望
- **假设依赖较重**：核心结论建立在 $\mathsf{BQP}$ 不在 PH 低层（如 $\mathsf{BQP}\not\subseteq\mathsf{BPP}^{\mathsf{NP}^{\mathsf{NP}^{\mathsf{NP}}}}$）这类启发式复杂度假设上，且正文用的是分布特定（distribution-specific）的启发式版本，精确版（对所有分布）放在附录 C。
- **概念类结构是硬约束**：分离只对 c-distinct（$c\ge 1/3$）或 average-case-smooth 概念类成立；作者自己指出，想去掉这些条件做出更一般的识别难定理，会和哈密顿量学习的经典可解性产生张力，因此短期内难以放松。
- **无实证 / 无算法落地**：全文是存在性与不可能性证明，没有给出可运行的量子识别算法实现或基准实验（仅在附录 G 提出对照 dequantized 经典算法的潜在 benchmark 设想）。
- **可改进方向**：能否把 PH 层数从三层降回更低（向 verifiable 版的一层靠拢）、能否找到不依赖 c-distinct/smoothness 的更自然概念类、以及把 Corollary 1 的可分离实例做成真正可跑的量子–经典对照实验，都是自然的后续。

## 相关工作与启发
- **vs Huang et al. (2021)**：他们指出"有了数据，经典学习器能高效评估某些本来难算的函数"，强调数据对评估的帮助；本文反其道而行，专门研究"数据帮不上忙"的量子函数场景，并把焦点从评估移到识别。
- **vs Gyurik & Dunjko (2023)**：他们在评估层面（要求模型能对新点求值）建立了全量子函数的学习分离，并指出识别分离离不开对假设类的限制；本文承接这一限制（$H=F$），首次在**识别层面**给出量子函数的分离，把优势来源推进到学习本身。
- **vs 密码学函数的识别难证明（Kearns & Vazirani 1994; Jerbi et al. 2024）**：这些证明依赖简洁表示 + 随机可生成性；本文证明量子函数不满足后者，因而必须放弃这套工具，改用反演 + 爬 PH 的新策略。
- **vs 哈密顿量学习（Anshu et al. 2020; Haah et al. 2024）**：哈密顿量学习可写成识别问题且经典可解，正好为本文的结构假设画出边界——说明 c-distinct/smoothness 不可随意去除，否则会与这条已知可解性矛盾。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次证明量子函数识别任务的经典难度，并发明"反演识别算法 + 爬多项式层级"的新证明策略。
- 实验充分度: ⭐⭐⭐⭐ 纯理论论文，定理与附录证明完整自洽，但无实证落地（理论工作正常）。
- 写作质量: ⭐⭐⭐⭐ 归约链条层层递进、动机交代清楚，惟复杂度记号密集对非专业读者门槛偏高。
- 价值: ⭐⭐⭐⭐⭐ 把量子机器学习优势从"评估难"推进到"学习过程难"，为 QML 的根本优势来源提供了新视角与新工具。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Efficient Turing Machine Simulation with Transformers](efficient_turing_machine_simulation_with_transformers.md)
- [\[ICLR 2026\] Learning to Adapt: In-Context Learning Beyond Stationarity](learning_to_adapt_in-context_learning_beyond_stationarity.md)
- [\[ICLR 2026\] The Lie of the Average: How Class Incremental Learning Evaluation Deceives You?](the_lie_of_the_average_how_class_incremental_learning_evaluation_deceives_you.md)
- [\[ICLR 2026\] Subquadratic Algorithms and Hardness for Attention with Any Temperature](subquadratic_algorithms_and_hardness_for_attention_with_any_temperature.md)
- [\[ICLR 2026\] Parameterized Hardness of Zonotope Containment and Neural Network Verification](parameterized_hardness_of_zonotope_containment_and_neural_network_verification.md)

</div>

<!-- RELATED:END -->
