---
title: >-
  [论文解读] Correcting Split Selection in Online Decision Trees via Anytime-Valid Inference
description: >-
  [ICML 2026 Spotlight][数据流学习 / 决策树 / 序贯推断][在线决策树] 作者指出经典 Hoeffding Tree（HT）在数据流上分裂时使用的"固定样本量"集中不等式被它自己采用的"数据相关停止规则"破坏，于是用 testing-by-betting + Universal Portfolio 重写分裂判据，让单棵树和 Adaptive Random Forest 都能在任意停止时刻保持 Type-I 错误可控，同时在 12 个真实流上更准且树更小。
tags:
  - "ICML 2026 Spotlight"
  - "数据流学习 / 决策树 / 序贯推断"
  - "在线决策树"
  - "anytime-valid 推断"
  - "testing-by-betting"
  - "Hoeffding Tree"
  - "Adaptive Random Forest"
---

# Correcting Split Selection in Online Decision Trees via Anytime-Valid Inference

**会议**: ICML 2026 Spotlight  
**arXiv**: [2605.31239](https://arxiv.org/abs/2605.31239)  
**代码**: 无  
**领域**: 数据流学习 / 决策树 / 序贯推断  
**关键词**: 在线决策树, anytime-valid 推断, testing-by-betting, Hoeffding Tree, Adaptive Random Forest

## 一句话总结
作者指出经典 Hoeffding Tree（HT）在数据流上分裂时使用的"固定样本量"集中不等式被它自己采用的"数据相关停止规则"破坏，于是用 testing-by-betting + Universal Portfolio 重写分裂判据，让单棵树和 Adaptive Random Forest 都能在任意停止时刻保持 Type-I 错误可控，同时在 12 个真实流上更准且树更小。

## 研究背景与动机

**领域现状**：在数据流场景里，bagging 类集成（尤其是 Adaptive Random Forest，ARF）几乎是事实标准；几乎所有这类方法都以 Hoeffding Tree（VFDT）作为基学习器。HT 的核心机制是：每来一批数据，就用 Hoeffding 不等式（或 McDiarmid、误分类率等修正版）判断"当前最好的候选分裂"是否显著优于第二好的，一旦超过阈值 $\varepsilon(n_t(v),\delta)$ 就提交分裂。

**现有痛点**：所有现存修正版（线性化的不纯度、McDiarmid 边界等）都只能保证"在某个固定样本量 $n$ 下"分裂正确的概率至少 $1-\delta$。可是 HT 实际是"看到证据就停"——这是一个数据相关的停止规则（optional continuation）。

**核心矛盾**：固定样本量不等式不能控制由数据触发的随机停止时刻 $\tau$ 下的错误率。Howard 等人 (2021) 已经展示，在这种 misuse 下，错误分裂的概率甚至可以飙到 1。换句话说，HT 至今没有真正合法的统计保证；当它被嵌进 ARF 跑非平稳流时，问题更严重——独立同分布的假设也被破坏了。

**本文目标**：构造一个分裂判据，在 (i) 数据相关停止、(ii) 非平稳且可能相依的数据流上都能控制 Type-I 错误，同时保证有真正预测优势时能在有限时间内提交分裂。

**切入角度**：放弃"在固定样本下比较哪个候选分裂最优"，转向 anytime-valid 推断（SAVI）——直接用 Ville 不等式把测试构造成"任何时刻都成立"的形式，这天然对 optional stopping 鲁棒。

**核心 idea**：把"是否分裂"重新表述为在线模型对比——把"不分裂的叶子"当 incumbent、把"按候选 $c$ 分裂后的叶子"当 challenger，让它们的 prequential 预测损失差 $\Delta_t^{v,c}$ 进入一个 testing-by-betting 的财富过程，财富超过 $1/\alpha^{v,c}$ 才提交分裂。

## 方法详解

### 整体框架
输入是顺序到达的 $(X_t,Y_t)$，输出是一棵随时间扩展的决策树。每个叶子 $v$ 内部维护一组候选分裂 $C_v$。对每个候选 $c=(j,s)$ 同时跑两个"影子预测器"：incumbent $m^v$ 用 $v$ 上的经验类分布预测；challenger $m^{v_c}$ 沿候选分裂把 $v$ 切成左右两叶，各自维护经验类分布。两者都基于过去信息预测，再用观测到的 $Y_t$ 计算预测损失差 $\Delta_t^{v,c}=\ell(m^v_{t-1}(X_t),Y_t)-\ell(m^{v_c}_{t-1}(X_t),Y_t)\in[-1,1]$。该差值通过一个 anytime-valid 测试累积"证据"，证据足够强时把 $v$ 用 $c^\star$ 真正切开，否则继续观望。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    A["数据流顺序到达 (X_t, Y_t)"] --> B["叶子 v + 候选分裂 c=(j,s)"]
    subgraph CMP["在线模型对比"]
        direction TB
        C1["incumbent：不分裂叶子<br/>用 v 的经验类分布预测"]
        C2["challenger：按 c 切成左右两叶<br/>各维护经验类分布"]
        C1 --> D["预测损失差 Δ_t ∈ [−1,1]"]
        C2 --> D
    end
    B --> CMP
    D --> E["Testing-by-betting + Universal Portfolio<br/>财富 W_t = W_{t-1}(1+β_t·Δ_t)，β_t 免调参"]
    G["置信序列变体 + 全局 α-分配"] -.->|设定阈值 1/α^{v,c}| F
    E --> F{"财富越线？ W_t ≥ 1/α^{v,c}<br/>（Ville 不等式）"}
    F -->|否，继续观望| A
    F -->|是| H["提交分裂 c⋆，把叶子 v 真正切开"]
    H --> A
```

### 关键设计

**1. 从"impurity 最大化"重构为"在线模型对比"：换掉一个在漂移下站不住的目标**

HT 原本要估计种群不纯度差 $\Delta^{v,c}=\mathcal{I}(p(v))-P_L\mathcal{I}(p(v_c^L))-P_R\mathcal{I}(p(v_c^R))$，但在非平稳流里这个目标本身随时间漂移、根本没有"全局最优分裂"可言。本文索性换一个清晰、可检验的零假设：strong null $H_0^{v,c}:\forall t,\;\delta_t^{v,c}=\mathbb{E}[\Delta_t^{v,c}\mid\mathcal{F}_{t-1}]\le 0$，即 challenger 在任何时刻都不优于 incumbent。证据来自 prequential 预测损失差 $\Delta_t^{v,c}=\ell(m^v_{t-1}(X_t),Y_t)-\ell(m^{v_c}_{t-1}(X_t),Y_t)\in[-1,1]$，损失用 log loss（对应 entropy）或 Brier score（对应 Gini），统一裁到 $[0,1]$。直接拿损失差当证据既绕开了不纯度的非线性问题，也不再依赖"种群最优分裂"这个漂移场景下的空概念。

**2. Testing-by-betting + Universal Portfolio：把"是否分裂"变成一场对 optional stopping 天然鲁棒的赌局**

固定样本量不等式扛不住 HT"看到证据就停"的数据相关停止，所以判据换成 anytime-valid 的财富过程。从单位财富 $W_s=1$ 起步，每步选一个 $\mathcal{F}_{t-1}$ 可测的下注比例 $\beta_t\in[0,1]$，财富更新 $W_t=W_{t-1}(1+\beta_t\Delta_t)$。在 $H_0$ 下 $(W_t)$ 是非负超鞅，Ville 不等式 $\mathbb{P}_{H_0}(\sup_t W_t\ge 1/\alpha)\le\alpha$ 保证"财富越线"就是任意时刻都合法的拒绝规则，提交时刻取 $\tau^{v,c}=\inf\{t:W_t^{v,c}\ge 1/\alpha^{v,c}\}$。下注比例不手调——这点很关键，手调 $\beta_t$ 等于偷看测试统计量——而采用 parameter-free 的 Universal Portfolio，用 $\mathrm{Beta}(1/2,1/2)$ Jeffreys 先验对所有常数再平衡组合做混合：

$$\beta_t=\frac{\int_0^1 \beta\prod_{u}(1+\beta\Delta_u)\,dF_+(\beta)}{\int_0^1 \prod_u(1+\beta\Delta_u)\,dF_+(\beta)}.$$

UP 在 i.i.d. 情形下增长率达到最优常数组合的最优值，又完全不需要调参，是目前 SAVI 框架里最强且不依赖独立性假设的工具。

**3. 置信序列变体 + 全局 $\alpha$-分配：补上"平均优势"语义和整棵树的终身错误控制**

strong null 实践中分裂更早、性能更好，但若想在 commit 时刻还保证模型损失单调下降，理论上需要"平均优势"语义，所以本文并提一个 weak null $H_{w,0}^{v,c}:\bar\delta_t^{v,c}=\frac{1}{t-s^v}\sum_u \delta_u^{v,c}\le 0$，用 empirical Bernstein 置信序列 $(L_t,U_t)$ 构造、停在 $\tau_w^{v,c}=\inf\{t:L_t>0\}$，让用户按场景二选一。全局控制部分则把预算 $\alpha$ 拆给所有候选 $(v,c)$：只要 $\sum_{v,c}\alpha^{v,c}\le\alpha$，由 union bound 加 anytime-valid 性质即得 $\mathbb{P}(\exists\text{false split ever})\le\alpha$，于是"一棵树跑一辈子"的 family-wise error 也守得住；要测严格正优势 $\varepsilon>0$，只需把财富换成 $\varepsilon$-shifted 版本 $W_{\varepsilon,t}=\prod_u(1+\beta_u(\Delta_u-\varepsilon))$ 并把 $\beta$ 收缩到 $[0,1/(1+\varepsilon)]$。

### 损失函数 / 训练策略
分类用 log loss、回归用 squared loss（在线维护最大观测损失自适应缩放到 $[0,1]$）。默认超参 $n_{\min}=20$、$\varepsilon=0$、$\alpha$ 即 family-wise 显著水平。理论保证有三条：(i) Thm 4.1，全局 anytime validity；(ii) Thm 4.2，当存在持续优势 $\Delta>0$ 时，提交时刻有限且高概率达到 $\tilde{\mathcal{O}}(\log(1/\alpha^{v,c})/\Delta^2)$；(iii) Thm 4.3，i.i.d.+convex loss 下，deployed 模型的期望损失在两次 commit 之间和 commit 时刻都单调不增。

## 实验关键数据

### 主实验
作者在 6 个回归 + 6 个分类共 12 条数据流上用 prequential（test-then-train）协议跑 10 次，比较 HT、ARF 以及把基学习器换成 anytime-valid 树后的 AVT_B（单树）/ AVF_B（森林）。AVT_B 在大多数流上击败 HT（除 abalone 外），在 bike/fried/hyper100k 上甚至能逼近或超过 ARF；AVF_B 在几乎所有数据集上都做到全局最佳。模型规模上，AVF_B 反而比单树的 AVT_B 还浅、节点数更少；AVT_B 因为没有 bagging 必须自己扛漂移，所以比 HT 稍深。

| 设定 | 在哪占优 | 关键现象 | 备注 |
|------|---------|---------|------|
| 单树 AVT_B vs HT | 12 条流中多数 | 预测曲线全程稳定单调，HT 在 i.i.d. RandomTree 上反复出现性能"塌方" | abalone 是 HT 略优的反例 |
| 森林 AVF_B vs ARF | 几乎全部流 | AVF_B 最佳且树最浅 | 漂移越剧烈差距越大 |

### 消融实验
warm-up（RandomTree，10 数值 + 10 类别特征，深度 4）+ 时间开销对比表给出了 AVT_B / AVT_CS / AVF_B 与 HT/ARF 的代价权衡。

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| HT（warm-up i.i.d.） | 多次塌方 + 叶子数 >> 真值 8 | 固定样本不等式 + optional stopping 失效 |
| AVT_B（betting） | 稳定单调改进，叶子 ≤ 8 | 严格贴合 Thm 4.3 单调性 |
| AVT_CS（weak null + Bernstein CS） | 比 AVT_B 略保守、分裂更晚 | 受历史平均影响，预测略差 |
| 推理时延 AVT_B vs HT | 0.06–0.12 ms vs 0.03–0.12 ms | 同量级，纯树遍历 |
| 更新时延 AVT_B vs HT | 1–25 ms vs 0.07–0.6 ms | UP 财富维护是主要额外开销 |
| 更新时延 AVF_B vs ARF | 同样 1 个数量级以内 | 树更浅抵消部分代价，且天然可并行 |

### 关键发现
- **strong null 比 weak null 更实用**：betting-based 测试分裂更早、损失更低；weak null 因"看平均值"会被历史拖累而保守。
- **AVF_B 树更小是结构性优势**：由于分裂判据更严格，每个 ensemble member 只在真正有持续优势时才长新节点，所以森林整体更紧凑。
- **额外开销来自财富维护而非推理**：更新时间被 UP 积分（实际用离散近似）拉高 1 个数量级，但仍在毫秒级，可实时；候选分裂之间天然并行。

## 亮点与洞察
- **把 SAVI 工具栈搬进流学习**：是第一篇把 testing-by-betting + UP + 经验 Bernstein 置信序列系统化地嫁接到决策树分裂判据的工作；以前的流学习社区基本只在 Hoeffding/McDiarmid 框架里打补丁。
- **"shadow challenger"是可复用的设计模式**：保持 incumbent 不动、并行训练候选模型只用于评估损失差，这套机制可以直接迁移到流式神经网络、流式 GBDT 等需要"在线模型选择"的场景。
- **理论与工程对齐得很干净**：Thm 4.3 的单调性几乎是"convex loss + plug-in 叶子更新"的直接结论，作者把它单独剥出来，强调 anytime-valid 测试只负责"结构性更新（分裂）也不破坏单调性"，让方法层和理论层各司其职。

## 局限与展望
- **更新代价仍是单棵树的 10–100 倍**：UP 用 Beta 先验数值积分逼近，候选越多越贵；作者承认实时性能依赖并行（GPU/多核），单线程在 nzenergy 这类长流上压力明显。
- **strong null 下 commit 时刻的单调性仍需"持续优势"假设**：作者明确说，要在 commit 时刻保证 $\mathbb{E}[\ell(\hat m_{\tau_k})]$ 严格下降，必须用 weak null + $\varepsilon$-shifted 测试，strong null 下只能"实证观察到"单调。
- **$\alpha$-分配粒度未深入讨论**：所有候选分裂共享全局预算 $\alpha$，但候选集 $C_v$ 可能很大（高维 + 多阈值），如何在不同 $(v,c)$ 之间分配 $\alpha^{v,c}$ 才能既保 family-wise 控制又不让有效检测力被过度稀释，论文给的是 schedule 附录，未在主文做敏感性分析。
- **流场景以外的扩展**：方法天然适合扩展到 streaming GBDT、in-stream pruning 等任务，但本文没涉及；非二叉分裂、类别特征、回归树外推等也只是"标准适配可做"一句话带过。

## 相关工作与启发
- **vs Hoeffding Tree / VFDT (Domingos & Hulten 2000)**：HT 是被本文直接 challenge 的对象——固定样本量不等式 + 数据相关停止 = 名义保证作废。本文用 SAVI 替换分裂判据，但保留 HT 的"单 pass、增量、按叶分裂"骨架，因此可以 drop-in 替换。
- **vs McDiarmid-bound 修正流 (Rutkowski 2012; De Rosa & Cesa-Bianchi 2017)**：这些工作只是把 Hoeffding 换成更紧的非线性集中不等式，仍然假设固定 $n$，本文论证它们其实没解决 optional stopping，理论根基同样有问题。
- **vs Orabona & Jun (2023) 的 UP / Choe & Ramdas (2024) 的序贯预测对比**：本文把后者的 sequential forecaster comparison 直接落地为决策树分裂的判据，并强调"strong vs weak null"在树学习里的实证差异。
- **vs Adaptive Random Forest (Gomes 2017/2018)**：本文不替换 ARF 的 bagging/漂移检测层，只换掉它内部 HT 基学习器，证明 AVF_B 在性能和树规模上同时超过 ARF，说明 ARF 的优势很大程度被基学习器的统计不合法性拖累。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把 SAVI/testing-by-betting 系统化地引入流式决策树，并给出全局错误控制 + 单调性双重保证，是这个子领域第一篇真正解决 optional stopping 问题的工作。
- 实验充分度: ⭐⭐⭐⭐ 12 个真实流 + warm-up 合成实验 + 时间开销表，覆盖单树和森林两种部署形式，唯一可惜是没和最新 streaming GBDT/online boosting 横向比（作者把它放到附录）。
- 写作质量: ⭐⭐⭐⭐ 问题陈述非常清楚——把"为什么 HT 不合法"用 Howard et al. (2021) Fig 1 直观说明；定理叙述与算法伪代码一一对应，可读性高。
- 价值: ⭐⭐⭐⭐ 是流学习社区少见的"上层算法不变、底层判据彻底替换"型贡献；对金融、网络入侵、IoT 等流场景特别有直接价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] When Sample Selection Bias Precipitates Model Collapse](when_sample_selection_bias_precipitates_model_collapse.md)
- [\[ICML 2026\] Active Learning with Low-Rank Structure for Data Selection](active_learning_with_low-rank_structure_for_data_selection.md)
- [\[AAAI 2026\] Generalizing Analogical Inference from Boolean to Continuous Domains](../../AAAI2026/learning_theory/generalizing_analogical_inference_from_boolean_to_continuous_domains.md)
- [\[ICLR 2026\] An Improved Model-free Decision-estimation Coefficient with Applications in Adversarial MDPs](../../ICLR2026/learning_theory/an_improved_model-free_decision-estimation_coefficient_with_applications_in_adve.md)
- [\[ICLR 2026\] Branch and Bound Search for Exact MAP Inference in Credal Networks](../../ICLR2026/learning_theory/branch_and_bound_search_for_exact_map_inference_in_credal_networks.md)

</div>

<!-- RELATED:END -->
