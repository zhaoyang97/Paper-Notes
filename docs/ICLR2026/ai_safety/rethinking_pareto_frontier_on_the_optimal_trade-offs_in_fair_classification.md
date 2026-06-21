---
title: >-
  [论文解读] Rethinking Pareto Frontier: On the Optimal Trade-offs in Fair Classification
description: >-
  [ICLR 2026][AI安全][公平分类] 本文把"给定模型结构下能达到的最优公平-精度权衡（model-specific Pareto 前沿）"重写成关于混淆向量的凸优化问题，证明了已有后处理前沿其实是次优的，进而提出带组相关偏置的末层重训练框架，并从理论上证明它严格优于随机翻转类后处理基线。
tags:
  - "ICLR 2026"
  - "AI安全"
  - "公平分类"
  - "Pareto 前沿"
  - "公平-精度权衡"
  - "凸优化"
  - "末层重训练"
---

# Rethinking Pareto Frontier: On the Optimal Trade-offs in Fair Classification

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=L8pyycR4wW](https://openreview.net/forum?id=L8pyycR4wW)  
**代码**: https://github.com/cjy24/Pareto-frontier  
**领域**: AI 安全 / 公平性  
**关键词**: 公平分类, Pareto 前沿, 公平-精度权衡, 凸优化, 末层重训练

## 一句话总结
本文把"给定模型结构下能达到的最优公平-精度权衡（model-specific Pareto 前沿）"重写成关于混淆向量的凸优化问题，证明了已有后处理前沿其实是次优的，进而提出带组相关偏置的末层重训练框架，并从理论上证明它严格优于随机翻转类后处理基线。

## 研究背景与动机
**领域现状**：机器学习被广泛用于决策后，公平性成了刚需。社区里有一堆公平概念（demographic parity DP、equal opportunity EOp、equalized odds EOd）和干预手段。大量经验观察表明：提升公平往往以损失精度为代价（公平-精度权衡），而强行同时满足两个公平概念又会互相冲突（即著名的"不可能定理"）。为了量化"在某个固定网络结构下，公平约束收紧时精度最多能保住多少"，已有工作用 **model-specific（MS）Pareto 前沿** 来刻画——以 ResNet-50 为例，"model-specific"指所有施加在 ResNet-50 上的公平干预所能达到的最好权衡曲线。

**现有痛点**：作者指出已有 MS 前沿的刻画有硬伤。Kim et al.(2020)、Jang et al.(2022) 用**后处理（随机翻转）**来逼近前沿，问题有三：① 后处理本身次优，导致画出来的"前沿"偏低，以至于别的方法曲线反而跑到前沿上方——这种"前沿"根本不配当上界基线；② 后处理的可行域由各敏感组的混淆矩阵经验确定，训练集和测试集分布有差异，训练集上可行的解在测试集上可能不可行，前沿不可靠；③ 即便换成 in-processing（Dehdashtian et al. 2024）或直接优化贝叶斯最优分类器（Wang et al. 2024，对应 model-agnostic 前沿），前者作者自己承认次优，后者依赖高方差的联合后验估计。

**核心矛盾**：根本问题在于——**以往都把"求最优前沿"绑死在某个具体公平干预算法上**，于是前沿的质量受限于那个算法的次优性；而且几乎所有讨论只覆盖公平-精度权衡，**公平概念之间的 Pareto 权衡从未被正经刻画**。

**本文目标**：(1) 给出一个不依赖任何特定干预算法、能逼近"最佳可达精度上确界"的 MS Pareto 前沿；(2) 把这套刻画推广到 DP 与 EOd 这两个公平概念之间的权衡，并说清它如何随精度变化；(3) 顺着新公式造一个真能逼近前沿的干预方法。

**切入角度**：作者沿用 Kim et al.(2020) 的关键观察——**精度和各种公平概念都可以写成混淆矩阵（混淆向量）的线性变换**。既然如此，那个原本关于网络参数 $\theta$ 的难解优化，就能转写成关于低维混淆向量 $z$ 的**凸优化**，绕开算法最优性这个坑。

**核心 idea**：把 Pareto 最优权衡重新表述为"关于混淆向量 $z$ 的带约束凸优化"，用 vanilla training 的精度作为上界来锁住可行域，从而直接逼近上确界；再用组相关偏置的末层重训练去贴近这条前沿。

## 方法详解

### 整体框架
整篇工作分两条线。**第一条是"度量"**：怎样画出一条真正可信的 MS Pareto 前沿。作者把"在公平违反度 $\le \epsilon$ 时能达到的最高精度"写成关于混淆向量 $z$ 的凸优化，再把 $\epsilon$ 在 $[0,1]$ 上均匀采样，逐点求解，串成前沿；同一套写法还能换成"固定精度下两个公平概念之间的最优权衡曲线"。**第二条是"干预"**：受这套公式启发，提出末层重训练框架——冻结编码器和投影头，只更新最后一层线性分类头，并给它加上**组相关偏置** $b_a$，直接对公平-精度目标做优化，并证明它严格优于随机翻转后处理。

度量这条线的关键是三个线性表示。设组 $a$ 的混淆向量 $z=[\text{TPR}_0,\text{TNR}_0,\text{TPR}_1,\text{TNR}_1]^T$，基率 $\alpha_a:=\Pr[Y=1\mid A=a]$、敏感属性边缘分布 $\beta:=\Pr[A=1]$，则：

$$\text{Acc}=A_c z,\quad A_c=[\alpha_0(1-\beta),\,(1-\alpha_0)(1-\beta),\,\alpha_1\beta,\,(1-\alpha_1)\beta]$$

$$\text{DP}=|A_{DP}z+A'_{DP}(1-z)|,\qquad \text{EOd}=\|A_{EOd}z\|_1$$

其中 $A_{DP}=[\alpha_0,0,-\alpha_1,0]$，$A'_{DP}=[0,(1-\alpha_0),0,-(1-\alpha_1)]$，$A_{EOd}=\begin{bmatrix}1&0&-1&0\\0&1&0&-1\end{bmatrix}$。有了这些线性式，原本关于 $\theta$ 的难解问题就变成了关于 $z$ 的凸问题。

### 关键设计

**1. 把 MS Pareto 前沿重写为关于混淆向量的凸优化**

针对"前沿绑死在具体算法、因而次优"这个痛点，作者不再去优化某个公平干预，而是直接在混淆向量空间里求上确界。MS 公平-精度权衡被写成

$$\arg\max_{z\in K} A_c z,\quad \text{s.t. } \|A_{EOd}z\|_1\le \epsilon \quad(\text{或 } |A_{DP}z+A'_{DP}(1-z)|\le\epsilon),$$

由于 $A_c,A_{EOd},A_{DP}$ 都由**测试集**边缘分布决定，且 $z$ 只有 4 维，这是个能秒解的低维凸问题。把约束 $\epsilon$ 在 $[0,1]$ 上均匀取 $T$ 个点逐一求解，得到的 $(\epsilon_i, A_c z_i)$ 就连成前沿。因为没对分类器 $f$ 施加任何具体公平松弛，这条前沿天然对应 Pareto 最优权衡的**上确界**，而不是某算法可达的那条偏低曲线。

作者还证明了旧前沿确实次优（Lemma 1）：在 ROC 曲线凹（Assumption 1）的前提下，把随机翻转和阈值调整结合起来构造的可行域 $\hat K$ 里，对 Kim et al.(2020) 那个松弛式（式 4）得到的任意解 $\tilde z$，**总存在一个严格更优的 $\hat z=\tilde z+[0,\delta,0,\delta]^T$**，满足 $A_c\tilde z< A_c\hat z$ 而 EOd 不变，其中

$$\delta=\min\{|\Phi_0^{-1}([1,0,0,0]\tilde z)-(1-[0,1,0,0]\tilde z)|,\ |\Phi_1^{-1}([0,0,1,0]\tilde z)-(1-[0,0,0,1]\tilde z)|\}.$$

这就从理论上说明：旧前沿可以被严格超越，所以不该被当成评判权衡的基线。

**2. 用 vanilla training 的精度上界锁住可行域 $K$**

凸优化要可信，关键在于可行域 $K$ 怎么定。作者用两条"vanilla training 最优"观察来收紧上界。其一是**整体精度上界**：施加非平凡公平约束不可能比无约束训练得到更低的分类损失，故 $A_c z\le A_c z_b$，$z_b$ 是基线模型的混淆向量。其二是**组相关精度上界**：每个敏感组内的精度也不该因为加公平约束而提升，于是 $A^a_c z\le A^a_c z^a_b$，其中 $z^a_b$ 取"只在该组上 vanilla 训练"和"在全量数据上 vanilla 训练"两者所得组内精度的较大值（考虑组间信息重叠）。合起来定义

$$K:=\{z\mid 0\le z\le 1;\ A_c z_b\ge A_c z;\ A^a_c z^a_b\ge A^a_c z\}.$$

为了逼近 $H'$（训练数据决定的可行假设空间）里的最佳测试精度，$z_b$ 和 $z^a_b$ 通过对 $f$ 多次随机初始化来估计。这一步是整套度量可信的根基：它让前沿对齐到"训练能力所允许的上限"，而不是被某个测试集上的可行域人为压低。

**3. 公平概念之间权衡的刻画与"不可能"边界**

针对"DP 与 EOd 之间权衡从未被量化"这个空白，作者把同一套写法套到公平-公平权衡上：

$$\arg\min_{z\in K}|A_{DP}z+A'_{DP}(1-z)|,\quad \text{s.t. }\|A_{EOd}z\|_1\le\epsilon,\ A_c z\ge\eta.$$

通过同时扫 $\epsilon$ 和精度下界 $\eta$，就能画出 EOd-DP 平面上一族随精度变化的等高线，每条线上还有一个 Pareto 拐点（在该精度下没有别的点能同时更优 DP 和 EOd）。更关键的是 Lemma 2 给出了不可能边界：**若 $\alpha_0\ne\alpha_1$，能同时达到 DP 和 EOd 的最优分类器，其精度必然退化到与常数预测器相同**。这意味着评估公平干预时，不该奢望 DP、EOd 同时都很小，而应看它的 EOd-DP 曲线是否贴近自己的 Pareto 最优——这给"两个公平概念都要照顾"的场景提供了一个更合理的评判标准。

**4. 组相关偏置的末层重训练及其理论优越性**

度量给出了"前沿在哪"，但要真贴上去得有方法。作者把分类器看成编码器 $g$ 加分类头 $h$，并把 $h$ 再拆成投影头 $h_1$（降到低维）和分类头 $h_2$；**只更新 $h_2$，冻结 $g$ 和 $h_1$**，使优化保持线性、可控。核心动作是给投影特征拼上组指示位：

$$\hat x_i=[\hat x^0_i,\hat x^1_i,\ \mathbb{1}[a_i=0],\ \mathbb{1}[a_i=1]],$$

于是 $h_2$ 的参数 $[w,b_0,b_1]$ 里的 $b_a$ 就成了**组相关偏置**，等价于给每组调不同阈值，但自由度比纯阈值调整更高。优化目标是

$$\arg\max_{w,b_0,b_1}\ \text{Acc}-\lambda\,\text{EOd}\quad(\text{或 } \text{Acc}-\lambda\,\text{DP}).$$

当把法向量取成基线的 $w^*$ 时，该框架退化为阈值法 $h_2(x_i)=\sigma(h_2^*(x_i)+c_a)$，$c_a=b_a-b^*$ 即组相关阈值；而放开 $w$ 就得到更大灵活性。理论上 Theorem 1 证明：在 Assumption 1 下，本方法的 **EOd 最优点（EOd=0 时精度最高的那个交点）严格优于随机翻转的 EOd 最优点**。进一步在"各子组 logit 服从等方差高斯 $l_i\sim\mathcal N(\mu_{ya},s^2)$"的假设下，Lemma 3 给出达到 EOd 最优时组相关阈值 $c^*_a$ 的闭式，Theorem 2 给出达到 EOd 最优时**精度下降量的解析估计**，让人能在不实际训 $h_2$ 的情况下，仅用潜表示和 $w^*$ 估出某数据集上的精度损失上确界。

### 损失函数 / 训练策略
度量侧无需训练，直接对每个采样的 $\epsilon_i$ 解 4 维凸优化即可。干预侧只重训末层分类头 $h_2$，目标为 $\text{Acc}-\lambda\,\text{EOd}$ 或 $\text{Acc}-\lambda\,\text{DP}$（EOd-DP 联合场景下改成同时考虑两者），$\lambda$ 为可调权重，扫 $\lambda$ 即得方法自身的权衡曲线。

## 实验关键数据

### 主实验
在 COMPAS、Adult、CelebA（二分类）和 Drug（多分类）四个数据集上验证，精度作效用、DP 和 EOd 作公平度量，对比 FACT、Eq. Odds、DFR、SELF、G-STAR、FOC 等后处理/末层重训练基线。

| 权衡类型 | 本文前沿/方法 | 关键观察 |
|----------|--------------|----------|
| EOd-精度 (图2) | Ours-EOd-frontier | 前沿近乎**水平**，收紧 EOd 几乎不掉精度；FACT、FACT+G-STAR 的前沿被多条方法曲线穿过，说明它们不是真上界 |
| DP-精度 (图3) | Ours-DP-frontier | DP 约束收紧时精度**明显下降**，印证 DP-精度的内在权衡 |
| EOd-DP (图4) | Ours (联合目标) | 精度高于常数预测器时两公平概念明显冲突；精度放松时前沿逼近 EOd 轴、冲突消失 |

### 消融实验

| 配置 | 关键效果 | 说明 |
|------|---------|------|
| Ours（组相关偏置 $b_a$ 全开） | 曲线最贴近 MS Pareto 前沿 | 完整方法 |
| 取 $w=w^*$（退化为阈值法 G-STAR） | EOd 最优点更差或仅持平 | 放开 $w$ 带来的自由度是收益来源 |
| FACT / Eq.Odds / DFR / SELF | 离前沿更远 | 受后处理/重训练次优性限制 |

（注：论文主体把详细数值表放在附录 9/10，正文以图 2-4 的前沿曲线呈现，此处归纳趋势性结论。）

### 关键发现
- **EOd-精度并非内在权衡**：因为 EOd 度量的是错误率差异而非预测本身，消除组间差异不必然降精度——这是反直觉但有理论支撑的结论。
- **DP-精度才是真权衡**：与 Zhao & Gordon (2022) 一致，$\alpha_0\ne\alpha_1$ 时 DP 越严精度越低。
- **组相关偏置是收益主来源**：放开 $w$ 比固定 $w^*$（纯阈值）灵活，EOd 最优点严格更优（Theorem 1）。
- **旧前沿确实可被超越**：FACT、FACT+G-STAR 前沿被实际方法曲线穿越，实证了 Lemma 1。

## 亮点与洞察
- **"线性化混淆向量 → 凸优化"是核心杠杆**：把关于 $\theta$ 的不可解问题压到 4 维 $z$ 上秒解，既可信又高效，是整篇能成立的支点。
- **用 vanilla training 当上界锁可行域**很巧：避免了用某个具体干预算法来定前沿，从而绕开算法次优性，直接逼近上确界。
- **把不可能定理"量化"了**：Lemma 2 不只说"DP 与 EOd 不可兼得"，而是精确到"同时满足则精度退化为常数预测器"，并据此提出"看曲线是否贴近自身 Pareto 最优"这一更合理的公平干预评判方式。
- **组相关偏置的可迁移性**：给末层特征拼组指示位、让每组有独立偏置，这个轻量 trick 可迁移到其它需要组级校准的末层重训练任务。

## 局限与展望
- **只做 model-specific，不做 model-agnostic**：前沿绑定在给定网络结构上；作者把 MA Pareto 最优权衡列为未来方向。
- **理论依赖若干假设**：ROC 曲线凹（Assumption 1）、子组 logit 等方差高斯——其中等方差假设偏强，作者用"vanilla training 下组方差收敛"来部分辩护，实际不严格凹时只是经验上仍观察到更好权衡。
- **公平概念覆盖有限**：主要讨论 DP 与 EOd 这类组公平，个体公平、min-max 公平未纳入同一框架。
- **未考虑分布漂移**：测试集边缘分布需可估；作者把"分布漂移下的公平-精度权衡"列为后续工作。

## 相关工作与启发
- **vs FACT / G-STAR (Kim 2020 / Jang 2022)**：他们用随机翻转/阈值后处理逼近前沿，本文证明这类前沿次优（Lemma 1）且在测试集上可能不可行；本文直接在混淆向量上求上确界，前沿更紧更可信，且把权衡推广到公平概念之间。
- **vs Wang et al. (2024)（model-agnostic 前沿）**：他们直接优化贝叶斯最优分类器对应的混淆矩阵，但依赖高方差的联合后验估计；本文走 model-specific 路线，用 vanilla training 上界回避了后验估计的不确定性。
- **vs DFR / SELF（末层重训练）**：DFR 靠特征重加权、SELF 靠选择性微调纠偏，都聚焦 EOd-精度；本文的组相关偏置带来更高自由度，并有 EOd 最优点严格更优的理论保证（Theorem 1），还能同时处理 DP 与 EOd-DP 联合权衡。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把 Pareto 前沿重写成混淆向量凸优化、并量化公平概念之间的权衡，视角新且自洽。
- 实验充分度: ⭐⭐⭐⭐ 四数据集覆盖二分类与多分类，前沿对比清晰，但正文偏图示、数值表多在附录。
- 写作质量: ⭐⭐⭐⭐ 理论推导链条完整、记号统一，但公式密度高，对非理论读者门槛偏陡。
- 价值: ⭐⭐⭐⭐⭐ 给公平干预提供了可信的上界基线和更合理的评判方式，对公平 ML 评估有方法论意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Demystifying the Optimal Fair Classifier in Multi-Class Classification](../../ICML2026/ai_safety/demystifying_the_optimal_fair_classifier_in_multi-class_classification.md)
- [\[ICLR 2026\] Fair Classification by Direct Intervention on Operating Characteristics](fair_classification_by_direct_intervention_on_operating_characteristics.md)
- [\[ICLR 2026\] Fair Conformal Classification via Learning Representation-Based Groups](fair_conformal_classification_via_learning_representation-based_groups.md)
- [\[ICML 2026\] Fair Decisions from Calibrated Scores: Achieving Optimal Classification While Satisfying Sufficiency](../../ICML2026/ai_safety/fair_decisions_from_calibrated_scores_achieving_optimal_classification_while_sat.md)
- [\[ICLR 2026\] Rethinking LoRA for Privacy-Preserving Federated Learning in Large Models](rethinking_lora_for_privacy-preserving_federated_learning_in_large_models.md)

</div>

<!-- RELATED:END -->
