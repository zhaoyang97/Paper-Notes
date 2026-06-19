---
title: >-
  [论文解读] Position: Beyond Sensitive Attributes, ML Fairness Should Quantify Structural Injustice via Social Determinants
description: >-
  [ICML 2026][AI安全][算法公平] 这是一篇 ICML 立场论文：作者主张 ML 公平性研究不能只盯着 race/sex 这类"敏感属性"，而必须把"社会决定因素"（neighborhood、ADI、学校经费、医疗可及性等情境变量）也纳入审计，并用大学录取理论模型 + 美国人口普查数据 + 乳腺癌筛查半合成实验，证明只围绕敏感属性的缓解策略反而可能制造新的结构性不公。
tags:
  - "ICML 2026"
  - "AI安全"
  - "算法公平"
  - "结构性不公"
  - "社会决定因素"
  - "敏感属性"
  - "因果公平"
---

# Position: Beyond Sensitive Attributes, ML Fairness Should Quantify Structural Injustice via Social Determinants

**会议**: ICML 2026  
**arXiv**: [2508.08337](https://arxiv.org/abs/2508.08337)  
**代码**: 无（立场论文）  
**领域**: AI 安全 / 算法公平 / 立场论文  
**关键词**: 算法公平、结构性不公、社会决定因素、敏感属性、因果公平

## 一句话总结
这是一篇 ICML 立场论文：作者主张 ML 公平性研究不能只盯着 race/sex 这类"敏感属性"，而必须把"社会决定因素"（neighborhood、ADI、学校经费、医疗可及性等情境变量）也纳入审计，并用大学录取理论模型 + 美国人口普查数据 + 乳腺癌筛查半合成实验，证明只围绕敏感属性的缓解策略反而可能制造新的结构性不公。

## 研究背景与动机

**领域现状**：当前 ML 公平性文献几乎把"不公"等价于"沿敏感属性的歧视"——大多数公平度量（Demographic Parity、Equal Opportunity、Conditional Demographic Parity、因果路径效应等）都是先指定 $A$（race/sex/age），再要求预测/决策与 $A$ 解耦或满足某种条件独立。Adult、Folktables、Communities and Crimes 等基准数据集甚至会主动丢掉 address、geolocation 这类情境字段。

**现有痛点**：跨学科文献（政治哲学、社会学、公共卫生）早已指出，真正塑造个体机会与结局的，是 social determinants——neighborhood deprivation、学校经费、空气污染、医院距离、社区资源等情境变量。这些变量在同一族群内部产生异质性（同样是 African American 女性，住在不同 PUMA 的中位年收入从 $38k 跌到 $18.8k），又在不同族群之间产生共同负担（贫困区的非 URM 申请者和 URM 申请者面对同样的社区劣势）。只看敏感属性会把这两种结构性信号同时抹掉。

**核心矛盾**：敏感属性是个体级的、（准）稳态的内禀标识；社会决定因素是情境级的、随空间/时间漂移的结构性变量。现有的个体级因果图（$A \to Y$、$A \to M \to Y$）与去敏感化损失，根本无法承载"邻里-个体"双向影响、社区聚合统计量这种 community-level 结构，把情境当噪声 normalize 掉了。

**本文目标**：把"社会决定因素"作为一个 first-class 的审计对象，回答三个问题：(i) 概念上如何把社会决定因素和敏感属性、敏感属性 proxy 清晰区分？(ii) 现有技术范式为何无法承载？(iii) 只围绕敏感属性的缓解策略到底会带来什么新的结构性不公？

**切入角度**：从一个具体场景出发——历史 redlining 把黑人家庭驱赶进特定社区，使 race、zip code、社区族裔构成长期高度相关；但这三者在公平涵义上完全不同（zip code 是行政标签，不能"改进"；学校经费、空气质量是真正的可干预结构变量）。作者用三个判据（context-level 定义 / social-structural content / exogenous stratification）把它们干净地分类。

**核心 idea**：审计必须先于缓解（auditing must precede mitigation）——在动手"修"模型之前，先要把结构性不公通过社会决定因素显式量化出来，否则盲目地按 race 配额很可能把更弱势的子群（贫困区的非 URM）推向更糟的位置。

## 方法详解

### 整体框架

这篇立场论文的核心主张是：ML 公平性研究不能止步于沿 race/sex 这类敏感属性去歧视，而必须把"社会决定因素"——neighborhood deprivation、学校经费、医疗可及性等情境变量——当成 first-class 的审计对象，而且审计必须先于缓解。为了把这个主张钉死，作者沿"概念 → 理论 → 实证 → 落地"递进论证：先用一套三判据定义把社会决定因素从敏感属性、proxy、行政标签里干净切出来；再用一个大学录取的闭式定理证明只按 race 配额的缓解会反噬贫困区的非 URM 申请者；接着用 Census 人口数据 + OSF HealthCare 真实乳腺癌筛查记录做半合成实验，证明即便统一指南、同一族群，社会决定因素仍制造系统性差距、且干预它能换来可量化的早检收益；最后把论证收成三条可操作 pillar（数据治理、新度量 Social Determinant Parity、多层因果模型），把"应该审计什么"落成"具体怎么做"。

### 关键设计

**1. 社会决定因素的三判据定义（Definition 2.2）：先把审计对象的边界划清，否则后续讨论会被术语模糊吞掉**

作者要求一个变量 $S$ 同时满足三条才算社会决定因素：(a) **Context-level definition**——它定义在某个情境（neighborhood / 机构 / 司法辖区）上，多个个体共享同一个 $S$ 值；(b) **Social-structural content**——跨情境的差异主要由资源配置、机构政策、系统性投资塑造（学校经费 ✓，zip code 这种纯行政标签 ✗）；(c) **Exogenous stratification**——聚合所用的边界（neighborhood / 邮政区）是外生划定的，而不是按被刻画群体自身的特征 endogenously 划定。靠这三条 yes/no，table 1 把容易混为一谈的变量摆进截然不同的格子：race=敏感属性；zip code=非社会决定因素（行政标签）；HOLC redlining 区的族裔构成=敏感属性的 proxy（边界是 endogenous 的）；而 zip code 区的族裔构成、学校经费=真正的社会决定因素。这套切分直接戳破一个常见混淆——"拿 neighborhood 当代理去跑 race 的歧视"是 redlining 的延伸，"审计 neighborhood 本身的结构条件"才是审计结构性不公；不区分两者，就会把"改善学校经费"这种真正可干预的动作排除在公平视野之外。

**2. 配额式录取的结构性不公定理（Theorem 4.5）：把"敏感属性中心化缓解何时反噬最弱势子群"从直觉升级为可证明命题**

作者用一个大学录取的闭式模型，把"只按 race 设配额的 affirmative action 何时会伤害贫困区非 URM 申请者"写成不等式。在 4 个假设下——区域族裔分布失衡、Academic Preparedness $\perp$ Race $\mid$ Region、富区分数 CDF 随机占优贫困区、选拔性大学只有有限名额 $g$——URM 配额可写成 $\eta_{\mathrm{quota}} \cdot \frac{n_a^{(\mathrm{poor})}+n_a^{(\mathrm{rich})}}{n} g$。定理给出的反例条件是：只有当 $\max_q \frac{F^{(\mathrm{rich})}(q)}{F^{(\mathrm{poor})}(q)} \ge \frac{\eta_{\mathrm{quota}}}{1+(1-\eta_{\mathrm{quota}})\frac{n_a^{(\mathrm{poor})}+n_a^{(\mathrm{rich})}}{n_{a'}^{(\mathrm{poor})}+n_{a'}^{(\mathrm{rich})}}}$ 成立，贫困区非 URM 申请者面对的分数阈值才不会被推得比富区 URM 申请者更高。它揭示的悖论很反直觉：结构性不公越严重（左侧随机占优比越大），不等式越容易满足、配额造成的伤害反而越小；可一旦结构正义改善，沿用同样的配额就越可能制造新的不公。而且 $\eta_{\mathrm{quota}}$ 越大、右侧门槛越高，意味着越激进的敏感属性中心化缓解越会放大对贫困区非 URM 的挤压——这正是"为什么必须先审计社会决定因素再谈缓解"的形式化论据。

**3. 乳腺癌筛查半合成实验（Section 5.2）：把理论落进一个高风险医疗场景，实证回应"fairness through unawareness"**

作者用 OSF HealthCare 2012–2022 约 5.4 万次筛查 / 4.5 万患者的真实记录，画出贫困区（ADI ∈ [75,100)）与富区（ADI ∈ [0,25)）白人女性的"首次筛查年龄"分布：同一筛查指南、同一族群，均值仍差 >3 年、中位数差近 5 年——差距只能归因于交通、可及性、信任度这些结构性条件。接着用 100k 粒子模拟、按 SEER 年龄别发病率采样癌症 onset，把 10k 个筛查名额放进"现状分布 vs 改进分布（贫困区改用富区的首次筛查年龄分布）"×"全分给贫困区 vs 两区均分"四种政策组合，各跑 500 次，统计"首次筛查年龄 ≤ onset 年龄 = 早检"的次数。结果是：贫困区一旦采用改进型筛查模式，早检数从 $1367 \pm 33$ 升到 $1461 \pm 36$。这个实验一箭双雕——既证明同族群、同指南都消不掉差距、必须把社会决定因素显式纳入审计，又量化出"干预社会决定因素"相比"调整敏感属性配额"是真正能当政策杠杆操作的（race 不是），为下面三条 pillar 提供经验佐证。

**4. 三条 actionable pillar 与 Social Determinant Parity 度量：把主张落成可操作的技术路线**

立场论文不涉及训练目标，但作者并不止于批判，而是给出三条可落地的 pillar。Pillar 1 是数据治理——别再像 Adult、Folktables 那样主动丢掉 address、geolocation 等情境字段，要把社会决定因素保留进审计数据。Pillar 2 提出新度量 **Social Determinant Parity**：把现有 Demographic Parity 的条件变量从族裔换成 area deprivation index、基础设施可及性、政策暴露等结构变量，其纵向版本进一步要求度量随情境变量做时变追踪。Pillar 3 主张引入多层因果模型 + causal representation learning，让社会决定因素成为显式的干预节点，而不是被压成 race 下游的 mediator——只有当它是干预目标，"改善学校经费 / 筛查可及性"这类动作才进得了因果框架。

## 实验关键数据

### 主实验

| 场景 | 关键数据 | 说明 |
|------|----------|------|
| Census PUMS, 加州 African American 女性年收入中位数 | 低 ADI \$38,000 / 中 ADI \$23,800 / 高 ADI \$18,800 | 同一族裔 × 性别交集，社会决定因素仍带来 >2× 的中位收入差距 |
| OSF HealthCare, 白人女性首次乳腺癌筛查年龄 | 富区 vs 贫困区：均值差 >3 年，中位数差 ≈5 年 | 同一统一筛查指南，差距只能归因于结构性条件 |
| 乳腺癌半合成模拟（10k 名额全分给贫困区，500 次） | 现状模式 $1367 \pm 33$ → 改进模式 $1461 \pm 36$ 次早检 | 仅靠改善"首次筛查年龄分布"这一个社会决定因素相关代理就拿到约 +7% 的早检收益 |

### 消融 / 政策对比

| 配额倍率 $\eta_{\mathrm{quota}}$ | 不等式 (1) 右侧门槛 | 贫困区非 URM 受损概率 |
|---|---|---|
| $\eta=1$（自然比例） | 右侧 = 1 | 与结构性不公严重度直接挂钩 |
| $\eta$ 增大 | 右侧单调升高 | 越大越易违反 → 贫困区非 URM 越被挤压 |
| 结构性不公改善（CDF 比下降） | 左侧下降 | 同 $\eta$ 下反而更易制造新不公 |

### 关键发现
- **同族同性别也能差 2 倍**：Figure 1 直接打脸"交集敏感属性已经足够"——African American 女性这一最常被讨论的交集群体，内部因 ADI 不同收入差距巨大。
- **统一指南消不掉差距**：OSF 数据里富区/贫困区白人女性走的是同一筛查指南，差距完全来自结构性条件（交通、可及性、信任度），说明"指南不感知社会决定因素"本身就是不公的来源。
- **配额悖论**：理论模型证明，结构正义越好的时候，配额反而越容易反噬贫困区非 URM；激进配额放大伤害——这是对"affirmative action 是不是越多越好"的形式化警告。
- **可干预性**：半合成实验显示仅"改进首次筛查年龄分布"就能换来 +94 次早检/10k 筛查，说明社会决定因素是真正可作为政策杠杆操作的，而 race 不是。

## 亮点与洞察
- **三判据定义切得极干净**：用 context-level / social-structural / exogenous stratification 三条 yes/no，能把 race / zip code / HOLC 族裔构成 / zip code 族裔构成 / 学校经费 摆进截然不同的格子，立刻让以后任何"我把 neighborhood 当 sensitive attribute 加进去就完事"的偷懒做法暴露问题，工具性极强。
- **配额悖论的反直觉**：业界长期讨论"affirmative action 是否公平"几乎都停留在哲学/价值判断；这篇用一个 4 假设 + 1 不等式把"何时配额会反过来伤害最弱势子群"变成可验算的条件，把哲学争论拉回数学，是这篇 position paper 最让人"啊哈"的地方。
- **审计先于缓解（auditing must precede mitigation）**：这一句方法论口号可迁移到任何"先评估后干预"的责任 AI 场景——比如把它套到 RLHF 的 reward 数据治理、医疗算法部署，都能直接复用 Pillar 1–3 的三层框架。
- **把医疗 SDoH 拉回 ML 公平**：Obermeyer 等人 2019 年那篇黑人医疗算法的工作之后，ML 社区一直缺一个"如何系统化地把 SDoH 接进公平框架"的入口，这篇把 OSF 真实数据 + 半合成模拟做出来，给后续工作铺了具体的实验范式。

## 局限与展望
- 作者承认：理论模型只刻画了"区域间"结构性不公，没考虑同一学校/机构内部的种族歧视；半合成乳腺癌实验也只覆盖首次筛查年龄这一个杠杆，不能视为对结构性壁垒的因果断言。
- 三判据定义在落地时还有灰色地带——例如"institutional membership 是否 exogenous"在很多就业、教育场景里其实并不清楚（college 录取本身就是 endogenous 选择），实操中如何稳定分类需要更多 case study。
- Social Determinant Parity 作为度量只在概念层提出，尚未给出具体可优化的微分形式，也没和现有 in-processing / post-processing 公平算法做实证对比。
- 多层因果模型 + causal representation learning 的设想要求观测足够多 community-level 协变量，且需要解决 SUTVA / no interference 失效问题，工程化路径仍待后续工作。

## 相关工作与启发
- **vs Conditional Demographic Parity (Žliobaite et al., 2011; Wachter et al., 2021)**: 后者用 region 当 mediator 解释 race 与 outcome 的残差依赖，本质仍是"race 视角"；本文反过来主张把 region 上承载的结构性变量本身设为审计目标，并明确撇清自己不是 Conditional Demographic Parity 的变种。
- **vs Path-specific Causal Fairness (Zhang & Bareinboim, 2018a; Chiappa, 2019; Wu et al., 2019)**: 这些方法虽然可以把社会决定因素塞进 race → SD → Y 的路径，但默认 SD 是 race 的下游 mediator；本文指出环境变量并非 race 的 ancestor，且把 SD 当 mediator 会丢掉它"可被政策直接干预"的杠杆属性。
- **vs Domain Adaptation 类公平 (Madras et al., 2018; Creager et al., 2021)**: 它们把跨情境异质性当作 distribution shift 去 normalize；本文反对这种"把情境视为噪声"的范式，主张情境正是要审计的信号本身。
- **vs Obermeyer et al. (2019) 黑人医疗算法**: 后者实证揭示了"用 cost 代替 need 会带来种族偏差"，是 SDoH 进入 ML 公平讨论的代表作；本文在这条线上更进一步，提供概念定义 + 理论模型 + 通用 pillar，把单点案例升级为方法论。
- **vs Kasirzadeh (2022)** 等结构性不公哲学讨论: 这些工作主要在哲学层呼吁；本文给出三判据 + 闭式定理 + 半合成实验，是哲学呼吁向 ML 工程实践的第一座桥梁。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 在 ML 公平这条已经被刷得很满的赛道上，把"审计对象"本身重新定义并形式化，立场层级的创新。
- 实验充分度: ⭐⭐⭐⭐ 理论闭式 + Census 真实数据 + 真实医疗数据 + 半合成模拟四块齐全；扣一星是没和现有 fairness 算法做端到端对比。
- 写作质量: ⭐⭐⭐⭐⭐ 论证链条 (I)–(V) 极清晰，table 1 / 三判据 / 配额定理彼此呼应，alternative views 一节正面回应了最强反驳。
- 价值: ⭐⭐⭐⭐⭐ 提供了可直接照抄的概念框架（三判据）+ 可计算的工具（Social Determinant Parity / 配额不等式）+ 可落地的三条 pillar，对 ICML 社区方向引导意义大。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Position: Machine Learning for Heart Transplant Allocation Policy Optimization Should Account for Incentives](position_machine_learning_for_heart_transplant_allocation_policy_optimization_sh.md)
- [\[ICML 2026\] LLM Benchmark Datasets Should Be Contamination-Resistant (Position Paper)](llm_benchmark_datasets_should_be_contamination-resistant.md)
- [\[ICML 2026\] Beyond Procedure: Substantive Fairness in Conformal Prediction](beyond_procedure_substantive_fairness_in_conformal_prediction.md)
- [\[ICML 2026\] Extending Fair Null-Space Projections for Continuous Attributes to Kernel Methods](extending_fair_null-space_projections_for_continuous_attributes_to_kernel_method.md)
- [\[AAAI 2026\] Revisiting (Un)Fairness in Recourse by Minimizing Worst-Case Social Burden](../../AAAI2026/ai_safety/revisiting_unfairness_in_recourse_by_minimizing_worst-case_social_burden.md)

</div>

<!-- RELATED:END -->
