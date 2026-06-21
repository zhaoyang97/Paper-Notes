---
title: >-
  [论文解读] Multi-LLM Adaptive Conformal Inference for Reliable LLM Responses
description: >-
  [ICLR 2026][LLM评测][共形推断] 把 LLM 事实性建模成「逐 claim 分数的累积乘积」，用组条件共形校准给出分布无关的覆盖保证，再用多 LLM 集成把事实性分数估得更准，从而在严格控制错误率的同时保留尽可能多的真实信息。 领域现状：在医疗、法律等高风险场景直接用 LLM 输出，前提是能保证其事实性…
tags:
  - "ICLR 2026"
  - "LLM评测"
  - "共形推断"
  - "事实性过滤"
  - "组条件覆盖"
  - "保留率"
  - "多 LLM 集成"
---

# Multi-LLM Adaptive Conformal Inference for Reliable LLM Responses

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=opuQH9Xyu9](https://openreview.net/forum?id=opuQH9Xyu9)  
**代码**: [https://github.com/MLAI-Yonsei/MACI](https://github.com/MLAI-Yonsei/MACI)  
**领域**: LLM 评估 / 事实性保证 / 共形推断（Conformal Inference）  
**关键词**: 共形推断, 事实性过滤, 组条件覆盖, 保留率, 多 LLM 集成  

## 一句话总结
把 LLM 事实性建模成「逐 claim 分数的累积乘积」，用组条件共形校准给出分布无关的覆盖保证，再用多 LLM 集成把事实性分数估得更准，从而在严格控制错误率的同时保留尽可能多的真实信息。

## 研究背景与动机
**领域现状**：在医疗、法律等高风险场景直接用 LLM 输出，前提是能保证其事实性。共形推断（CI）因其分布无关、有限样本的统计保证，成为给 LLM 事实性兜底的热门工具——典型做法是把回答拆成原子 claim，给每个 claim 打一个事实性分数，再按阈值过滤掉不可信的 claim。

**现有痛点**：第一代方法 BCI（Mohri & Hashimoto 2024）对所有数据用**单一全局阈值**，只能给出边际覆盖（marginal coverage），对不同难度的子群会忽高忽低；更糟的是它用单个「最坏 claim 分数」当文档级 conformity score，对该 claim 的估计误差极度敏感，校准时只能把阈值压得很保守，结果**误删大量真实 claim**（保留率常低到 0.01~0.06）。第二代 CCI（Cherian et al. 2024）引入条件阈值函数，但依赖**自适应错误率（adaptive α）**——高风险场景需要的是固定保证而非浮动保证，且其线性阈值函数难以刻画 LLM 回答按语义切分出的复杂组结构。

**核心矛盾**：要么覆盖达标但保留率极低（信息被砍光），要么保留率上去但放松了统计保证。覆盖（validity）与保留（efficiency）这对矛盾在高风险场景里被进一步收紧。

**本文目标**：在严格满足用户指定的低错误率 α、且做到**组条件覆盖**的前提下，尽可能多地保留真实 claim。

**核心 idea**：**① 乘法过滤框架**——把文档级的「整个保留集合都为真」事件写成逐 claim 概率的累积乘积，从而得到一个聚合全体 claim 信息、而非依赖单点最坏值的 conformity score；**② 保留率理论**——首次证明保留率差距由事实性分数的估计 MSE 决定，为「估得越准、保留越多」提供理论依据；**③ 多 LLM 集成**——用集成降低估计方差、逼近 oracle 分数，把理论上的保留上限真正吃到手。

## 方法详解

### 整体框架
MACI 在 Mondrian 共形框架下工作：把回答拆成 claim、用一组 LLM 给每个 claim 打事实性分数、按累积乘积构造文档级 conformity score、在每个组内分别校准阈值得到组条件覆盖保证。整条链路只需要每个 claim 的标量分数，因此可作为即插即用的过滤器套到任意生成器上。

```mermaid
flowchart LR
    A[回答拆成原子claim] --> B[M个LLM各打事实性分数]
    B --> C[优化加权集成 → 单一分数 p_ens]
    C --> D[按分数降序 → 累积乘积过滤规则 F]
    D --> E[conformity score E_i<br/>边界随机化取精确覆盖]
    E --> F[组内校准阈值 Q^k_1-α]
    F --> G[组条件覆盖 ≥ 1-α<br/>保留率最大化]
```

### 关键设计

**1. 乘法过滤框架：把「整个保留集都为真」写成累积乘积。** MACI 的出发点是 oracle 情形下的最优过滤规则。给定 claim 按 oracle 分数降序排列 $p^*_i(c_{\pi(1)})\ge\cdots\ge p^*_i(c_{\pi(N_i)})$，定义前缀乘积 $\Pi_k=\prod_{j=1}^{k}p^*_i(c_{\pi(j)})$，则对阈值 $\tau$ 保留前 $K^*_i(\tau)=\max\{k:\Pi_k\ge\tau\}$ 个 claim。这个乘积恰好是「前 k 个 claim 在独立 Bernoulli 假设下全为真」的概率，因此天然把文档级覆盖约束 $\Pr(\text{保留集}\subseteq\text{真集})\ge\tau$ 转成了一个标量条件。相比 BCI 只看单个最坏 claim，累积乘积聚合了全体保留 claim 的可信度，对单点估计误差鲁棒得多。为了把「覆盖恰好等于 $\tau$」而非保守地超过，作者在边界索引 $K^*_i(\tau)$ 处做随机化：以概率 $\gamma_i(\tau)=\frac{\Pi_{K^*}-\tau}{\Pi_{K^*}-\Pi_{K^*+1}}$ 多收一个 claim，从而在精确覆盖下最大化期望保留率。

**2. 组条件自适应共形校准：固定 α 下的有限样本保证。** 实践中 oracle 分数 $p^*$ 未知，用黑盒 LLM 估计的 $\hat p$ 替代。为把文档级过滤事件压成可校准的标量，作者定义 conformity score $E_i=\inf\{\tau:F(\hat p,\tau,U_i;P_i,C_i)\subseteq A_i\}$，即「能让保留集全为真的最小阈值」。在交换性假设下，对所有样本的 $E_i$ 取 $1-\alpha$ 分位数作阈值，即得 Theorem 1 的边际覆盖保证 $\Pr(\text{保留集}\subseteq\text{真集})\ge 1-\alpha$（且近乎紧）。再仿照 Mondrian 框架，把校准**限制在与测试样本同组**的子集 $I_k$ 内，阈值 $\hat Q^{(k)}_{1-\alpha}=\text{Quantile}(\{E_i:i\in I_k\},1-\alpha)$，即得 Theorem 2 的**组条件覆盖**保证。与 CCI 不同，这里的 α 是用户固定的，符合高风险场景要求。

**3. 保留率随估计误差递减的理论：把「估得准」量化成「保留多」。** 光有 validity 不够——保守规则也能合法却没用。作者在 oracle+Bernoulli 假设下证明此时 conformity score 恰为 $[0,1]$ 上均匀分布，意味着 Theorem 2 能无保守地榨干保留率。更关键的是 Theorem 3：设保留率差距 $\Delta=|R(\hat p,\tau)-R(p^*,\tau)|$，在阈值附近的 margin 条件 $\Pr(|p^*-\tau|\le\epsilon)\le C\epsilon^\beta$ 下，有
$$\Delta\le C'\big(\mathbb{E}[(\hat p-p^*)^2]\big)^{\frac{\beta}{\beta+2}},$$
即保留率差距以估计 MSE 的多项式速率收敛。这首次把共形推断里「估计质量」与「真 claim 保留量」直接挂钩，并直接为下一步集成提供动机。

**4. 多 LLM 集成：用降方差逼近 oracle。** 既然 MSE 越小保留越多，自然想到集成降方差。但直接最小化 MSE 不可行（oracle 不可观测，二元标签会把预测器逼向过自信）。作者改用基于保留率分解的代理目标：把保留率写成 $R(p,\tau)=\rho\cdot\text{TPR}+(1-\rho)\cdot\text{FPR}$，在约束 $\text{TPR}\ge 1-\delta$（防止靠砍 recall 取巧）下最小化 FPR，求加权集成 $p_\text{ens}=\sum_m w_m p_m$ 的最优权重 $w$。这个代理目标既避免过自信，又在实验中确实压低了 MSE，把理论保留上限落到实处。

## 实验关键数据

### 主实验（组条件覆盖 / 保留率，30 次重复均值）
在 MedLFQA、WikiBio、ExpertQA 三个数据集上对比 BCI、CCI；覆盖落在 $1-\alpha\pm0.01$ 记为达标（•），否则为过/欠覆盖。回答固定为各数据集随附的 GPT-4 / GPT-3.5-turbo 生成结果，事实性分数由 Llama-3.3-70B、Qwen-2.5-72B、DeepSeek-V3 三模型集成估得。

| 数据集 / 组 (α=0.1) | BCI Cov./Ret. | CCI Cov./Ret. | MACI Cov./Ret. |
|---|---|---|---|
| MedLFQA (边际) | 0.90• / 0.02 | 0.90• / 0.31 | 0.90• / **0.50** |
| MedLFQA · False-Claim Risk-High | 0.88↓ / 0.01 | 0.89• / 0.22 | 0.89• / **0.41** |
| WikiBio (边际) | 0.90• / 0.01 | 0.89• / 0.11 | 0.90• / **0.25** |
| WikiBio · View Count-Low | 0.87↓ / 0.01 | 0.88↓ / 0.11 | 0.91• / **0.21** |

MACI 在几乎所有组、所有目标覆盖（80/90/95%）下都稳定达标，且保留率最高——常是 BCI 的 10~30 倍、CCI 的 1.5~2 倍。BCI 在 False-Claim Risk / View Count 等异质组上频繁过/欠覆盖，CCI 在固定 α 下也多次欠覆盖。

### 消融与分析
- **集成 vs 单模型**：多 LLM 集成相比任一单模型显著降低事实性分数的 MSE（Figure 3），印证 Theorem 3「MSE 降→保留升」的链路。
- **conformity score 形式**：累积乘积分数相比 BCI 的单点最坏分数对估计误差更鲁棒，避免被极端 claim 拖累出保守阈值。
- 附录还覆盖 MultiValid CI、Group Clustering、Joint Probability Modeling 等变体及协变量偏移下的表现对比。

### 关键发现
- 用单一极值 claim 做 conformity score 是 BCI 低保留的根因；改成聚合全体 claim 的累积乘积即可大幅松绑。
- 在固定 α（而非自适应 α）下同时拿到组条件覆盖与高保留，是 MACI 相对 CCI 的核心实用价值。
- 时间成本也低于基线（无需反复采样一致性检查）。

## 亮点与洞察
- **把统计目标翻译成可优化的工程目标**：从 oracle 最优规则反推出「累积乘积 conformity score」，再用保留率分解把「逼近 oracle」转成「约束 TPR 下最小化 FPR」的可解优化，理论与方法环环相扣。
- **首个共形推断保留率理论**：Theorem 3 把估计 MSE 与真 claim 保留量定量挂钩，给「为什么要集成」提供了严格依据，而非经验直觉。
- **即插即用**：只依赖逐 claim 标量分数，可套任意黑盒生成器，落地门槛低。

## 局限与展望
- **组定义依赖人工先验**：组划分 g 用的是数据集特定的高层类别（如医疗问题类型、实体组），换域需重新设计分组准则。
- **Bernoulli 独立假设**：保留率最优性的推导假设同文档内 claim 标签条件独立，真实 claim 间往往有事实依赖，乘积分数可能偏乐观。
- **集成成本**：主实验用 3 个 70B 级模型打分，推理开销不低；小组样本下组内校准阈值仍偏保守、保留率受限。
- **β/margin 条件难验证**：Theorem 3 的收敛率依赖未知的 margin 指数 β，实际界的紧度不易评估。

## 相关工作与启发
- **BCI**（Mohri & Hashimoto 2024）：单全局阈值 + 单点最坏分数的开山做法，MACI 直接针对其保守性下手。
- **CCI**（Cherian et al. 2024）：条件共形 + 自适应 α，与 MACI 目标相同但保证形式不同，是最直接对手。
- **多校准 / MultiValid CP**（Hébert-Johnson 2018；Jung et al. 2023）：组条件覆盖的理论根基，但通常牺牲效率，MACI 试图在其精神下保住保留率。
- **启发**：在「分布无关保证 vs 信息保留」这类对立约束下，先写出 oracle 最优解、再用一个能量化估计误差影响的理论把工程改进（集成、更聪明的 score 聚合）对齐到统计目标上，是很值得复用的方法论。

## 评分
- **新颖性**: ⭐⭐⭐⭐ 乘法过滤框架 + 首个共形推断保留率理论 + 多 LLM 集成动机，三者构成自洽且有理论支撑的新方法。
- **实验充分度**: ⭐⭐⭐⭐ 三数据集 × 多组划分 × 三档目标覆盖 × 30 次重复，附录含丰富变体与协变量偏移分析，证据扎实。
- **写作质量**: ⭐⭐⭐⭐ 从 oracle 规则到实践方法的推导脉络清晰，定理与设计对应明确。
- **价值**: ⭐⭐⭐⭐ 在高风险场景给出固定 α 下兼顾覆盖与保留的实用过滤器，即插即用、落地价值高。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] AdaBlock-dLLM: Semantic-Aware Diffusion LLM Inference via Adaptive Block Size](adablock-dllm_semantic-aware_diffusion_llm_inference_via_adaptive_block_size.md)
- [\[ICML 2026\] Margin-Adaptive Confidence Ranking for Reliable LLM Judgement](../../ICML2026/llm_evaluation/margin-adaptive_confidence_ranking_for_reliable_llm_judgement.md)
- [\[ACL 2026\] Statistically Reliable LLM-Based Ranking Evaluation via Prediction-Powered Inference](../../ACL2026/llm_evaluation/statistically_reliable_llm-based_ranking_evaluation_via_prediction-powered_infer.md)
- [\[ICLR 2026\] Sci2Pol：评测与微调 LLM 的「科学→政策简报」生成能力](sci2pol_evaluating_and_fine-tuning_llms_on_scientific-to-policy_brief_generation.md)
- [\[ICLR 2026\] Log Probability Tracking of LLM APIs](log_probability_tracking_of_llm_apis.md)

</div>

<!-- RELATED:END -->
