---
title: >-
  [论文解读] Adaptive Conformal Anomaly Detection with Time Series Foundation Models for Signal Monitoring
description: >-
  [ICLR 2026][异常检测][时间序列基础模型] 提出 W1-ACAS：一种 post-hoc、免微调的自适应共形异常检测框架，把预训练时序基础模型（TSFM）的预测误差转成可直接解释为误报率（p-value）的异常分数，并通过最小化 Wasserstein 距离在线学习权重，在非平稳数据下稳定控制误报。
tags:
  - "ICLR 2026"
  - "异常检测"
  - "时间序列基础模型"
  - "共形预测"
  - "自适应异常检测"
  - "信号监测"
  - "误报率控制"
---

# Adaptive Conformal Anomaly Detection with Time Series Foundation Models for Signal Monitoring

**会议**: ICLR 2026  
**代码**: [github.com/ibm-granite/granite-tsfm](https://github.com/ibm-granite/granite-tsfm/tree/main/notebooks/hfdemo/adaptive_conformal_tsad)  
**领域**: 时间序列异常检测 / 共形预测  
**关键词**: 时间序列基础模型, 共形预测, 自适应异常检测, 信号监测, 误报率控制  

## 一句话总结
提出 W1-ACAS：一种 post-hoc、免微调的自适应共形异常检测框架，把预训练时序基础模型（TSFM）的预测误差转成可直接解释为误报率（p-value）的异常分数，并通过最小化 Wasserstein 距离在线学习权重，在非平稳数据下稳定控制误报。

## 研究背景与动机
工业预测性维护与信号监测场景里，往往缺数据、缺数据清洗管线、也缺训练专家，导致依赖大规模训练的异常检测器难以落地。时序基础模型（TSFM，如 Chronos、TTM、TiRex）提供了零样本预测能力，能在数据稀缺时给出"正常信号"的合理期望值，因此用预测残差做异常判定成为自然路径。

**现有痛点**：预测-残差类方法虽然直观，但残差阈值通常依赖全量数据统计、假设固定分布，既不可在线流式使用，也缺乏概率含义——用户看到一个异常分数却不知道它对应多大的误报概率。共形预测（conformal prediction）本可提供分布无关、有限样本保证的不确定性量化，但标准共形依赖**可交换性（exchangeability）** 假设，而时间序列因时间依赖性和分布漂移天然违反该假设。

**核心矛盾**：既要让异常分数可解释（直接等于误报率）、对复杂误差分布鲁棒，又要在分布随时间漂移、样本不可交换的流式设定下保持校准、压低误报。现有共形异常检测要么固定阈值假设可交换，要么只为单一误报率优化，无法同时满足这些诉求。

异常本身又有多种形态：点异常（单点显著偏离正常模式）和上下文异常（仅在特定时间上下文中才算异常），有效检测需要模型既能捕捉时间依赖、又能适配非平稳分布——这进一步加大了"固定阈值"思路的失效风险。

**本文目标**：在不微调 TSFM 的前提下，构造一个对任意误报率 $\alpha$ 都校准、且能在线自适应分布漂移的共形异常分数。**核心 idea**：把异常分数定义为加权共形 p-value，并以"让 p-value 在所有 $\alpha$ 上都服从均匀分布"为目标，用 1-Wasserstein 距离在线学习对历史残差的加权向量。框架命名为 **W1-ACAS**（1-Wasserstein Adaptive Conformal Anomaly Score）。

## 方法详解

### 整体框架
W1-ACAS 接在任意预训练预测器（TSFM）后面：对每个时刻、每个预测步长 $d$，取预测残差 $S_{t+1}^d=|Y_{t+1}-\hat Y_{t+1}^d|$ 作为不一致分数（nonconformity score）；用历史残差和一组权重 $w$ 经加权共形分位数把残差映射成 p-value 形式的异常分数；该映射的权重通过在线最小化"p-value 分布与均匀分布"的 Wasserstein 距离来更新；最后把多个预测步长得到的 p-value 取中位数聚合成最终异常分数，低于阈值即报警。

```mermaid
flowchart LR
    A[预训练 TSFM<br/>多步预测] --> B[残差 S_t+1^d<br/>= |Y - Ŷ^d|]
    B --> C[加权共形分位数<br/>p-value 映射 φ_w]
    C --> D[在线更新权重 w<br/>min W1·均匀分布]
    D --> C
    C --> E[多步长 p-value<br/>取中位数聚合]
    E --> F{p̄ < α ?}
    F -->|是| G[报警]
```

### 关键设计

**1. 把残差归一化成"可解释为误报率"的共形 p-value：** 异常检测的实际困难是残差 $S$ 本身没有概率含义，而且分布会漂移。作者定义加权共形分位数 $Q_{1-\alpha}(s,w)$（在标准共形分位数基础上给每个历史样本 $i$ 赋权 $w_i\in[0,1]$），并把异常分数定义为 $\phi_w(S_{t+1})=\sup\{\alpha\in[0,1]:S_{t+1}\le Q_{1-\alpha}(s,w)\}$。这个 $\phi_w$ 就是加权共形化的 p-value：它把任意实值残差映射进 $[0,1]$，且能直接与任意 $\alpha$ 阈值比较——用户设定的 $\alpha$ 直接就是期望误报率。命题 4.1 证明用 $\mathbb{1}[\phi_w(S_{t+1})<\alpha]$ 做检测器与基于加权分位数的原始检测器等价，因此继承了误报率保证。

**2. 借非可交换共形界，把"该给谁高权重"变成可证明的准则：** 直接套用 Barber 等人（2023）的非可交换共形界（命题 3.1），误报率被界在 $\alpha + \sum_i \frac{w_i}{\|w\|_1+1}d_{TV}(s,s_i)$ 附近，其中 $d_{TV}(s,s_i)$ 是把测试点与第 $i$ 个历史点互换后两个序列的全变差距离。这给出清晰的工程含义：应给那些与测试样本**近似可交换**（互换后联合分布几乎不变）的历史残差更高权重，对偏离者降权；同时下界还鼓励让 $\|w\|_1$ 尽量大、权重尽量贴近 1，以保留有效样本量。这把"自适应加权"从启发式变成有保证可循的优化问题。

**3. 用 Wasserstein 距离把"全误报率校准"写成可微目标：** 理想情况下正常数据上 $\phi_w(S_{t+1})\sim U[0,1]$，即对所有 $\alpha$ 都有 $P(\phi_w\le\alpha)=\alpha$。作者据此最小化 p-value 的 CDF 与均匀分布 CDF 的 1-Wasserstein 距离：$\min_w W_1(F_{\phi_{t+1}(w)},F_U)$，约束有效样本量 $\|w\|>1/\alpha_c-1$（$\alpha_c$ 是用户设的临界误报率）。关键观察是由对偶式 $W_1=\mathbb{E}_{\alpha\sim U}|P(\phi_{t+1}\le\alpha)-\alpha|$，即最小化 Wasserstein 距离**等价于在所有误报率上一致地最小化校准误差**——这正好对上第 1 点想要的"对任意 $\alpha$ 都校准"。

**4. 闭式梯度 + 投影梯度下降的在线流式优化：** 经验 CDF 下，$W_1$ 目标是一串分段线性函数的积分之和，因此对每个 $\phi_{t+i}(w)$ 可微、进而对 $w$ 可微且有闭式表达（式 13、14）。算法维护一个大小为 $n_b$ 的在线缓冲，按式 12 做投影梯度下降（用 ADAM），并把权重投影回 $\{w\in[0,1]^n,\|w\|>n_c\}$ 可行域。梯度含义直观：$\partial W_1/\partial\phi_{t+i}$ 把归一化分数推进其经验分位桶内，$\partial\phi_{t+i}/\partial w_k$ 表明降低"比当前分数更大的历史残差"的权重即可抬高 p-value。多步长则并行跑 $D$ 个实例、对 p-value 取中位数 $\bar\phi_{t+1}=\mathrm{median}_d\phi_{t+1}^d$，要求半数以上步长的检测器都判异常才报警，从而压低误报。

## 实验关键数据

### 主实验表格（univariate，跨 7 个数据集均值±std）

| Forecaster | AD 方法 | PA-F1↑ | Affiliation-F↑ | FPR↓ | CalErr↓ | AUC-PR↑ | VUS-PR↑ |
|---|---|---|---|---|---|---|---|
| TiRex | **W1-ACAS** | **0.925** | **0.897** | 0.084 | **0.025** | 0.344 | 0.438 |
| TiRex | Conformal | 0.878 | 0.890 | 0.107 | 0.038 | 0.308 | 0.429 |
| TiRex | Gaussian | 0.714 | 0.837 | 0.119 | 0.090 | 0.270 | 0.432 |
| Chronos | W1-ACAS | 0.912 | 0.893 | 0.077 | 0.025 | 0.355 | 0.440 |
| TTM | W1-ACAS | 0.889 | 0.886 | 0.082 | 0.029 | 0.342 | 0.449 |
| - | CNN* | 0.858 | 0.881 | 0.083 | 0.643 | 0.269 | 0.423 |
| - | MOMENT ZS | 0.596 | 0.867 | 0.261 | 0.417 | 0.110 | 0.461 |
| - | KShape | 0.533 | 0.789 | 0.508 | 0.176 | 0.125 | 0.303 |
| - | USAD* | 0.498 | 0.809 | 0.425 | 0.324 | 0.088 | 0.398 |

（*为半监督深度方法，需在正常训练集上训练）

### 消融与配置
- **横跨三种 TSFM（Chronos / TTM / TiRex）**：W1-ACAS 在每个基座上都显著优于同基座的 Gaussian 与 Conformal offline 基线（PA-F1 提升约 5~20 点、CalErr 减半以上）。
- **三个基座结果一致**：附录显示三种 TSFM 在各数据集上预测误差相近，对应其异常检测性能也接近，说明增益主要来自共形后处理而非某个特定基座。
- **超参敏感性**：context length 52、horizon $D=15$、$\alpha_c=0.01$、$n_b=10$、$\eta=0.001$；对小学习率和小 batch 表现稳定，$D=15$ 在性能与样本效率间取得平衡。
- **多步长聚合**：附录显示步长聚合（取中位数）相比单步长进一步降低误报，要求半数以上步长检测器都判异常才报警。
- **合成实验**：在可获得真值 p-value 的合成数据上验证了方法在渐变与突变分布漂移下都能保持校准。

### 关键发现
- W1-ACAS 在**阈值相关指标（PA-F1、Affiliation-F）上全面领先**，甚至超过需训练的半监督方法（CNN/USAD/OmniAnomaly），在阈值无关指标（AUC、VUS）上保持竞争力。
- **校准误差 CalErr 极低（0.025）**，远优于深度基线（CNN 高达 0.643）：低 FPR 区间内其阈值最保守、方差最低，使阈值选择在实践中更可靠。
- 经典距离/密度方法（KShape、Sub-KNN、SAND）虽在部分基准表现不俗，但普遍非因果、需全量数据、FPR 高（0.4~0.5），不适合流式场景，被 W1-ACAS 全面压制。
- 学习到的权重能自动捕捉误差中的周期性等时间模式，并在异常区附近快速适配新分布，从而在端到端系统里压低报警数量。

## 亮点与洞察
- **把"可解释性"做成第一公民**：异常分数不是黑盒打分，而是直接等于误报率 p-value，工业用户设 $\alpha$ 即设误报率，决策透明可操作。
- **理论与目标对得很齐**：从非可交换共形界（该给谁高权重）→ Wasserstein 校准目标（最小化它等价于全 $\alpha$ 校准）→ 闭式可微梯度，三步推导逻辑闭环，不是堆 trick。
- **完全 post-hoc + model-agnostic**：不动 TSFM 一根参数，挂在任意预测器或任意异常分数后即可用，特别契合"资源受限、要立即推理"的工业落地。
- **统计后处理也能赢过重模型**：在 CalErr 等校准指标上，轻量的共形后处理把需要训练的深度半监督方法远远甩开，提示"基础模型 + 校准良好的统计层"是被低估的高性价比路线。

## 局限与展望
- **依赖一段初始正常数据做校准**：每个数据集都需要无异常的初始片段来学权重，纯冷启动场景仍受限。
- **多变量仅靠 p-value 聚合**：多变量异常通过 horizon 式聚合处理，未显式建模变量间相关性，复杂耦合异常可能漏检。
- **权重只看残差分布相似度**：作者也指出未来要引入上下文特征来 refine 共形加权，当前对"语义上相似但数值分布不同"的上下文异常适配较弱。
- **临界误报率 $\alpha_c$ 需人为设定**：更小的 $\alpha_c$ 要求更多的历史正常样本 $n_c$，在长期监测中可能带来内存与延迟开销。

## 相关工作与启发
- **共形预测谱系**：从 split conformal（可交换假设）→ 在线自适应共形（Gibbs & Candès，处理漂移但只盯单一误报率）→ 加权共形（按相似度重赋权）→ 本文用 Wasserstein 学权重以"对所有 $\alpha$ 校准"，是对加权共形目标的一次升级。
- **非可交换共形界（Barber 2023）**：本文权重准则的理论根，把"近似可交换样本给高权重"落到可证明的误报率界上。
- **TSFM 用于异常检测**：相比 MOMENT 这类零样本通用异常打分，本文证明"TSFM 预测 + 自适应共形后处理"在校准与误报控制上更可靠，给"基础模型 + 轻量统计后处理"的范式提供了有力样例。
- **流式异常检测评测**：实验沿用 Liu & Paparrizos（2024）的 curated benchmark 与 oracle 阈值策略，并同时报告点级（AUC、PA-F1）与区间级（VUS、Affiliation-F）指标，对"如何公平评测时序异常检测"也有借鉴意义。

## 评分
- **新颖性**: ⭐⭐⭐⭐ — 把加权共形的目标从"单一误报率覆盖"提升为"全 $\alpha$ 均匀校准"，并用 Wasserstein 对偶与闭式梯度落地，组合新颖且推导扎实。
- **实验充分度**: ⭐⭐⭐⭐ — 7 个单变量 + 多个多变量数据集、三种 TSFM、对比经典/半监督/零样本多类基线，附超参敏感性与合成校准验证，较充分；不足是缺更大规模工业部署实证。
- **写作质量**: ⭐⭐⭐⭐ — 动机—理论—算法—实验脉络清晰，命题与梯度推导完整，图示直观；公式密度偏高对非共形背景读者略有门槛。
- **价值**: ⭐⭐⭐⭐ — 免微调、可解释、误报可控，直击工业数据稀缺监测痛点，且有开源代码，落地价值高。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Low Rank Transformer for Multivariate Time Series Anomaly Detection and Localization](low_rank_transformer_for_multivariate_time_series_anomaly_detection_and_localiza.md)
- [\[ICLR 2026\] Foundation Visual Encoders Are Secretly Few-Shot Anomaly Detectors](foundation_visual_encoders_are_secretly_few-shot_anomaly_detectors.md)
- [\[ICLR 2026\] MRAD: Zero-Shot Anomaly Detection with Memory-Driven Retrieval](mrad_zero-shot_anomaly_detection_with_memory-driven_retrieval.md)
- [\[ICLR 2026\] ReTabAD: A Benchmark for Restoring Semantic Context in Tabular Anomaly Detection](retabad_a_benchmark_for_restoring_semantic_context_in_tabular_anomaly_detection.md)
- [\[ICLR 2026\] UniOD: A Universal Model for Outlier Detection across Diverse Domains](uniod_a_universal_model_for_outlier_detection_across_diverse_domains.md)

</div>

<!-- RELATED:END -->
