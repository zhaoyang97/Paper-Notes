---
title: >-
  [论文解读] When LLMs Get Significantly Worse: A Statistical Approach to Detect Model Degradations
description: >-
  [ICLR 2026][LLM评测][模型退化检测] 针对"量化/稀疏化后的 LLM 到底有没有真的变差，还是只是评测噪声"这个问题，本文把它形式化为一个统计假设检验，提出**精确单边 McNemar 检验**——不看任务级聚合准确率，而是逐样本对照两个模型的对错，从而能以受控的假阳率把哪怕 0.3% 的准确率下降也判定为"真退化"。
tags:
  - "ICLR 2026"
  - "LLM评测"
  - "模型退化检测"
  - "McNemar 检验"
  - "量化"
  - "假设检验"
  - "逐样本对照"
---

# When LLMs Get Significantly Worse: A Statistical Approach to Detect Model Degradations

**会议**: ICLR 2026  
**论文**: Published as a conference paper at ICLR 2026（Amazon）  
**代码**: https://github.com/amazon-science/LLM-Accuracy-Stats (有)  
**领域**: LLM 评测 / 模型压缩 / 统计假设检验  
**关键词**: 模型退化检测, McNemar 检验, 量化, 假设检验, 逐样本对照

## 一句话总结
针对"量化/稀疏化后的 LLM 到底有没有真的变差，还是只是评测噪声"这个问题，本文把它形式化为一个统计假设检验，提出**精确单边 McNemar 检验**——不看任务级聚合准确率，而是逐样本对照两个模型的对错，从而能以受控的假阳率把哪怕 0.3% 的准确率下降也判定为"真退化"。

## 研究背景与动机
**领域现状**：为了降低 LLM 推理成本和延迟，业界用大量"优化"手段——从理论上无损的高效内核、推测解码，到会轻微改变输出分布的量化（INT4/FP8）、稀疏化。判断一个优化是否"安全"，通常的做法就是在若干 benchmark 上跑准确率，看掉了多少个百分点，再凭经验拍一个阈值（如"掉 ≤2% 就算无损"）。

**现有痛点**：这种"看聚合准确率差"的判断在现代 LLM 上根本站不住脚。即使温度为 0、即使是理论上无损的改动（换硬件、换框架、单卡 vs 张量并行、甚至同一命令重跑一遍），浮点运算的非结合性 $(a+b)+c \neq a+(b+c)$ 会沿计算图累积误差，导致同一个模型生成不同答案。本文实测这种"理论无损"的改动也会带来 1.3%–2.8% 的样本翻转（flip）。于是"理论无损 vs 有损"的界限被模糊了，靠固定阈值既会漏报真退化，也会误报无害噪声。

**核心矛盾**：评测两个模型用的是**同一批样本** $x_1,\dots,x_N$，所以两个准确率估计 $\hat\gamma$ 和 $\hat\beta$ **不独立**。如果错误地当成独立来算差值方差，方差会被高估（本文估计在 LLM 典型场景下高估约 $\sqrt5\approx2.2$ 倍），从而把真实退化误判为统计涨落。$\text{Var}[\hat\gamma-\hat\beta]\neq \tfrac{\gamma(1-\gamma)}{N}+\tfrac{\beta(1-\beta)}{N}$，这正是"在同一组病人身上做用药前后对比"那类配对检验的难点。

**本文目标**：给"模型退化检测"一个严格的统计定义，并给出一个：① 假阳率（type-I error）受控、② 检验功效（power）尽量高、③ 能跨多个 benchmark 聚合成单一决策、④ 还能省评测成本的检验框架。

**切入角度**：1947 年 McNemar 就为"配对比例比较"提出过检验——只看两个模型**意见不一致**的样本。作者把它适配到 LLM 退化检测，并改造成单边、可精确计算 p 值的版本以提升功效。

**核心 idea**：**不要在任务级聚合准确率上做文章，而是逐样本对照两个模型的对错**，在意见不一致的样本上估计"退化概率" $q_\downarrow$，用二项检验判断它是否显著大于 $1/2$。

## 方法详解

### 整体框架
方法本质是一套围绕 $2\times2$ 列联表展开的统计检验流程。输入是：基线模型 $M$ 与优化后模型 $\tilde M$ 在**同一批** $N$ 个评测样本上的逐样本对错（每个样本打分 $L\in\{0,1\}$）；输出是：一个（或多个 benchmark 聚合后的）p 值，据此在受控假阳率下判定"$\tilde M$ 是否真的退化了"。

整条链路是：先把每个样本归入列联表的四个格子 $a,b,c,d$（两模型都错 / 仅基线对 / 仅优化对 / 都对）→ 只在"意见不一致"的 $b+c$ 个样本上定义退化概率 $q_\downarrow=b/(b+c)$，用**精确单边 McNemar 检验**（实为二项检验）算 p 值 → 用检验功效分析推出"该删掉哪些样本来省成本"，并配一个噪声模拟来实际筛样本 → 当有多个 benchmark 时，用三种聚合检验（Pooled / Max Drop / Fisher）+ 组合决策汇成一个判断。这是纯统计方法，无需画 pipeline 图。

### 关键设计

**1. 逐样本对照与退化概率：把"配对相关"显式建模进列联表**

痛点是同批样本导致 $\hat\gamma,\hat\beta$ 相关，聚合准确率差的方差无法由两个边际方差简单相加。本文不再分别估两个准确率，而是把每个样本看成一次"联合对错实验"，落入 $2\times2$ 列联表（见下），并定义两个核心量：翻转概率 $p_\updownarrow := P[L(M(X))\neq L(\tilde M(X))]=P_b+P_c$，以及（条件）退化概率

$$q_\downarrow := P[L(M(X))=1 \mid L(M(X))\neq L(\tilde M(X))] = \frac{P_b}{p_\updownarrow}.$$

直觉上，$p_\updownarrow$ 量化"两模型打分有多频繁地不一致"，$q_\downarrow$ 则在"已经不一致"的前提下量化"有多大比例是基线对、优化错"（即真退化方向）。关键事实（Fact 1）把它和准确率严格挂钩：**优化模型准确率 $\beta<\gamma$ 当且仅当 $q_\downarrow>1/2$**。这一步是整篇的支点——它把"两个相关比例是否相等"的难题，转化为"一个二项参数 $q_\downarrow$ 是否超过 $1/2$"的标准问题，相关性被吸收进了只看 $b,c$ 的设计里（McNemar 早就指出：判断两比例是否相等，只有不一致的 $b,c$ 是相关的，$a,d$ 无关）。

**2. 精确单边 McNemar 检验：用二项检验给出可控假阳率的判定**

有了 $q_\downarrow$，检验统计量就是它的经验估计 $\hat q_\downarrow := \dfrac{b}{b+c}$。在固定 $b+c$ 时，$b\sim\text{Binomial}(b+c,\,q_\downarrow)$，零假设取 $q_\downarrow=1/2$。由于只关心"退化"这一个方向，本文用**单边** p 值（更敏感），并用 `scipy` 直接算精确二项 p 值——所以这个检验实质就是一个标准二项检验。作者还点明它和经典 McNemar 统计量的关系：

$$\frac{(b-c)^2}{b+c}=(2\hat q_\downarrow-1)^2(b+c),$$

即经典 McNemar 统计量不过是 $\hat q_\downarrow$ 仿射变换后的平方缩放。相比经典版（双边、卡方近似），单边 + 精确 p 值带来更高功效，且**只依赖精确二项分布**，因此在小样本/小退化时也能正确控制 type-I error 在 $\alpha$ 内——这正是作者强调的"有用检验的必要条件"。

**3. 检验功效分析与数据集压缩：剔除"不会翻转"的样本不损失信号**

光有正确性还不够，评测很贵，希望别白跑用不上的样本。作者对准确率差 $\delta:=\gamma-\beta=p_\updownarrow(2q_\downarrow-1)$ 做渐近分析，推出检验功效完全由信噪比决定：

$$\text{SNR}:=\sqrt{N/p_\updownarrow}\,\delta.$$

关键观察：若把数据集裁成"只保留两模型不一致的样本"（即 post-hoc 令 $a=d=0$），则 $\delta\to\delta/p_\updownarrow$、$N\to Np_\updownarrow$、$p_\updownarrow\to1$，代入后 **SNR 完全不变**。这给出 Recommendation 1：**对退化检测而言，剔除那些几乎不会翻转的样本只增加成本、不增加信号**。为了真正落地"哪些样本不会翻转"，作者提出一个噪声模拟筛选：拿基线模型在有限温度下（10 次、Temp=0.3，调到每次约 20% 翻转）多跑几遍，统计每个样本被答对的次数——那些 10 次里要么永远对、要么永远错（never-flip）的样本，在真实退化测试里也几乎不翻转，可以删。在 MMLU-Pro（12,032 例）上，5,604 个样本从不翻转，删掉后数据集近乎减半，却保留了绝大多数真正携带退化信号的翻转样本。

**4. 跨任务聚合的三种检验 + 组合决策**

实际评测往往横跨多个 benchmark（BBH、GPQA、IFEval、MATH、MMLU-Pro、MuSR），各任务样本量差异极大，怎么汇成一个判断？作者给每个任务 $i$ 收一张列联表得到 $b_i,c_i$，提出三种互补的聚合：**Pooled**（把所有任务的 $b,c$ 直接相加当一个大数据集，按样本量天然加权，退化均匀分布在多任务时最灵敏，但被大而无信号的任务拖累）；**Max Drop**（对每任务算标准化统计量 $\hat z_i=\frac{\hat q_{\downarrow i}-0.5}{\text{SE}(\hat q_{\downarrow i})}$，取 $z_{\max}=\max_i \hat z_i$，用 Monte Carlo 模拟零分布——适合"只有单个 benchmark 受影响"的情形）；**Fisher's method**（假设各任务样本独立，用 $\chi^2=-2\sum_i\ln p_i$ 合并各任务 p 值，服从 $2T$ 自由度卡方，灵敏度介于前两者之间）。由于这三个 p 值彼此相关、无法紧致合并，最终用一条简单**组合决策**：三者中任一在 $\alpha$ 下拒绝就判退化；由 Bonferroni 论证它把 type-I error 控制在 $3\alpha$，而实验中实际控制往往比 $3\alpha$ 更紧。

### 一个例子：Llama-3.1 8B 的 KV-FP8 到底退化没有
拿 KV-FP8 变体（用 FP8 精度的 FlashAttention3）对比 BF16 基线：聚合准确率只掉了 $\hat\delta=0.79\%$，按"掉 ≤2% 算无损"的硬阈值规则根本不会被标记；翻转概率 $\hat p_\updownarrow=9.03\%$，和"无退化"的 FP8 变体（8.71%）几乎一样，硬翻转规则也分不清。但本文的检验逐样本看下来：在 BBH、MATH、MMLU-Pro 这些大数据集上的下降是强信号，正确加权后 $p_{\text{pool}}=1.69\text{e-}05$——明确判为显著退化。更极端地，70B 的 KV-FP8 只掉 0.3% 也被检出显著，而两个基线规则全都失败。这就是"逐样本对照 + 正确处理样本量"相对"看聚合数字"的威力。

## 实验关键数据

### 主实验
在 Llama-3.1 8B Instruct、Llama-3.3 70B Instruct、Mistral-Small-3.1 上跑 Leaderboard v2（6 个 benchmark，共 25,282 例），基线为 BF16、8×H100、TP8。下表节选 8B 的代表性结果（$\hat\delta$ 为对基线的准确率差，$p_{\text{pool}}$ 为 Pooled 检验 p 值，$\hat p_\updownarrow$ 为翻转概率）：

| 变体 | 类型 | $\hat\delta$ | $p_{\text{pool}}$ | $\hat p_\updownarrow$ | 判定 |
|------|------|------|------|------|------|
| Baseline（重跑） | 理论无损 | -0.02% | 6.29e-01 | 1.31% | 无退化 ✓ |
| A100 换硬件 | 理论无损 | -0.14% | 9.21e-01 | 2.73% | 无退化 ✓ |
| FP8 | 有损量化 | -0.04% | 6.01e-01 | 8.71% | 无退化 ✓ |
| KV-FP8 | 有损量化 | 0.79% | 1.69e-05 | 9.03% | **显著退化** |
| w4a16 (INT4) | 有损量化 | 1.73% | 4.80e-15 | 12.63% | **显著退化** |
| 2:4 Sparse（8B base） | 剪枝蒸馏 | 2.59% | 1.09e-19 | 20.99% | **显著退化** |

关键对照：理论无损变体也有 1.3%–2.8% 翻转，但检验一致判为"无显著退化（甚至在统计涨落内略有提升）"；而 KV-FP8 仅掉 0.79%、70B KV-FP8 仅掉 0.3%，都被检出显著——硬阈值（$\hat\delta>2\%$ 或 $\hat p_\updownarrow\ge5\%$）两条规则在 70B 上无一能检出任何退化。

### 数据集压缩 / 提速
在 MMLU-Pro（12,032 例）上用噪声模拟剔除从不翻转的样本后：

| 配置 | 样本数 | w4a16 翻转率 | KV-FP8 翻转率 |
|------|--------|------|------|
| 全量 | 12,032 | 20.55% | 15.49% |
| 删除 5,604 never-flip | 6,807 | 31.09% | 24.42% |

数据集近乎减半，却保留了绝大多数真正的翻转样本（信号几乎不损），与功效分析中"SNR 不变"的理论一致。

### 关键发现
- **逐样本对照是检测力的来源**：把"无损改动"和"真退化"区分开，靠的不是看 $\hat\delta$ 或 $\hat p_\updownarrow$ 的绝对值（它们在 FP8 与 KV-FP8 上几乎相同），而是逐样本检验 + 正确处理各任务样本量。
- **翻转概率不是好指标**：高翻转既可能来自真退化也可能来自无害噪声；FP8（翻转 8.71%）会被硬翻转规则误报，但统计检验显示它毫无退化迹象。
- **意外发现 bug**：70B 的 w4a16 在 vLLM 上准确率暴跌（$\hat\delta=39.46\%$），换 transformers 后掉点轻得多——检验顺带暴露了推理栈的实现问题（⚠️ 以原文为准）。
- **0.3% 也能下结论**：在正确统计处理下，哪怕 0.3% 的经验准确率下降也能被自信地归因为真退化而非噪声。

## 亮点与洞察
- **把工程直觉变成可控统计判定**：用一个二项检验就把"模型有没有变差"这个长期靠拍阈值的问题，变成假阳率受控、功效可分析的决策——而且实现直接挂在被广泛采用的 LM Evaluation Harness 上，落地成本极低。
- **"无损也会翻转"的实证打脸固定阈值**：理论无损改动带来 1.3%–2.8% 翻转、单任务可达 0.5% 偏差，直接说明任何固定阈值/翻转率规则都不可能跨模型跨数据集稳健。
- **功效分析直接指导省钱**："剔除不会翻转的样本不损失 SNR"这个推论，把统计理论变成了可操作的数据集压缩策略，可迁移到任何需要配对比较的评测压缩场景（如回归测试、A/B 评测的样本精选）。
- **聚合策略的取舍说清了**：Pooled / Max Drop / Fisher 各自适配"退化均匀 / 集中单任务 / 折中"三种情形，组合决策用 Bonferroni 兜底——这套思路可直接搬到任何多 benchmark 聚合判定。

## 局限与展望
- 当前聚焦**二元打分**（对/错），非二元分数的推广放在附录（⚠️ 以原文为准），连续型/部分得分任务的处理还需进一步验证。
- 组合决策为控制假阳用了 Bonferroni（$3\alpha$），偏保守；三个检验 p 值相关无法紧致合并，留有功效改进空间。
- 噪声模拟筛样本依赖温度等超参（Temp=0.3、10 次、约 20% 翻转）的人工设定，对不同任务/模型的可迁移性需更多验证。
- 检验回答的是"是否显著退化"的二元问题，仍存在 type-I / type-II 错误；"显著"不等于"实际影响大"，用户若关心真实准确率掉幅 $\delta$，仍需结合 $p_\updownarrow$ 一起看（功效是 $N,\delta,p_\updownarrow$ 三者的函数，无法只凭 $N,\delta$ 反推所需样本量）。

## 相关工作与启发
- **vs Dutta et al. (2024)**：他们提出用"打分翻转概率"检测退化，并指出准确率不适合做退化判据。本文反驳：翻转在不优化模型时也常发生，且他们的方法缺乏统计显著性量化、$0\!-\!2\%$ 的阈值是任意选的；本文证明正确统计处理下这类小偏差常足以判定显著退化。
- **vs 经典 McNemar / Yang et al. (2025b)**：经典 McNemar 用于"哪个模型在单任务上更好"的双边判断；Yang 等推广到重复交叉验证但界过松、过于保守。本文改造为单边、精确 p 值版本，专为"优化后是否退化"提升功效。
- **vs lm-eval 默认做法**：lm-eval 给每个任务单独的标准误，但把两模型差值当独立来算会高估误差（约 $\sqrt5\approx2.2$ 倍），从而漏检真退化；本文从配对相关性出发修正了这一点。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把经典 McNemar 适配 + 单边精确化到 LLM 退化检测，并配功效分析与聚合策略，组合很扎实（单边 McNemar 本身不claim原创）。
- 实验充分度: ⭐⭐⭐⭐ 覆盖三系列模型、多种无损/有损变体与 6 个 benchmark，且有合成实验佐证聚合策略，顺带发现 vLLM bug。
- 写作质量: ⭐⭐⭐⭐ 推导清晰、动机—理论—实验闭环，符号略密但有附录索引。
- 价值: ⭐⭐⭐⭐⭐ 直接挂 lm-eval、能检出 0.3% 退化、还能压数据集省评测成本，对模型压缩/部署的质量守门极实用。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] LLMs Get Lost In Multi-Turn Conversation](llms_get_lost_in_multi-turn_conversation.md)
- [\[ICLR 2026\] Noisy but Valid: Robust Statistical Evaluation of LLMs with Imperfect Judges](noisy_but_valid_robust_statistical_evaluation_of_llms_with_imperfect_judges.md)
- [\[ICLR 2026\] When to Ensemble: Identifying Token-Level Points for Stable and Fast LLM Ensembling](when_to_ensemble_identifying_token-level_points_for_stable_and_fast_llm_ensembli.md)
- [\[ICLR 2026\] Credit-Budgeted ICPC-Style Coding: When Agents Must Pay for Every Decision](credit-budgeted_icpc-style_coding_when_agents_must_pay_for_every_decision.md)
- [\[ICLR 2026\] Pitfalls in Evaluating Language Model Forecasters](pitfalls_in_evaluating_language_model_forecasters.md)

</div>

<!-- RELATED:END -->
