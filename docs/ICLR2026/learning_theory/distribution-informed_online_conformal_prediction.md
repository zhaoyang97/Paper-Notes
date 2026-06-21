---
title: >-
  [论文解读] Distribution-informed Online Conformal Prediction
description: >-
  [ICLR2026][学习理论][在线保形预测] 本文提出 COP（Conformal Optimistic Prediction），在传统在线保形预测的反应式更新之外加一步「乐观修正」——用估计的非一致性分数 CDF 当作对下一步的预判 hint，使预测区间在数据有可预测模式时更窄，同时保留分布无关的有限样本覆盖保证，即使 CDF 估计不准也不破坏长期覆盖。
tags:
  - "ICLR2026"
  - "学习理论"
  - "保形预测"
  - "在线学习"
  - "不确定性量化"
  - "在线保形预测"
  - "乐观梯度下降"
  - "覆盖保证"
  - "后悔界"
  - "时间序列"
---

# Distribution-informed Online Conformal Prediction

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=I69SaLbwqZ](https://openreview.net/forum?id=I69SaLbwqZ)  
**代码**: https://github.com/creator-xi/Conformal-Optimistic-Prediction  
**领域**: 学习理论 / 保形预测 / 在线学习 / 不确定性量化  
**关键词**: 在线保形预测, 乐观梯度下降, 覆盖保证, 后悔界, 时间序列

## 一句话总结
本文提出 COP（Conformal Optimistic Prediction），在传统在线保形预测的反应式更新之外加一步「乐观修正」——用估计的非一致性分数 CDF 当作对下一步的预判 hint，使预测区间在数据有可预测模式时更窄，同时保留分布无关的有限样本覆盖保证，即使 CDF 估计不准也不破坏长期覆盖。

## 研究背景与动机
**领域现状**：保形预测（Conformal Prediction, CP）是做不确定性量化的主流工具——它在「数据可交换」的假设下，能构造一个以指定概率 $1-\alpha$ 包含真值的预测集，且与模型无关、分布无关。但真实的时间序列存在分布漂移和时序相关，可交换性被破坏，于是出现了在线保形预测（online CP）：以 ACI 为先驱，把对抗式在线学习塞进 CP 框架，用 $\hat q_{t+1}=\hat q_t+\eta(\mathrm{err}_t-\alpha)$ 这样的梯度更新动态调阈值，只追求长期覆盖 $\lim_{T\to\infty}\frac1T\sum_t \mathbf 1\{Y_t\notin \hat C_t\}=\alpha$。后续 Conformal PID、ECI、decay-OGD、专家聚合等都在改进步长与更新规则。

**现有痛点**：这些在线学习方法把环境当成**完全对抗**来对待，只根据「上一步对没对」反应式地纠偏，靠正负误差相互抵消来凑出覆盖率。结果是阈值 $q_t$ 在真值附近剧烈震荡，预测区间普遍**过于保守、偏宽**，浪费了序列里本可利用的可预测信息。另一条路线（SPCI、Lee 等、HOP-CPT）直接估计分数的分布函数或给历史分数加权来出区间，能利用结构，但要么依赖模型一致性 / 分布光滑等假设、不是分布无关的，要么干脆没有有限样本覆盖保证、容易过拟合。

**核心矛盾**：「利用可预测模式把区间收窄」和「不依赖分布假设、保住覆盖保证」之间存在张力——纯反应式方法守住了保证但太宽，纯估计分布方法窄了却把命运全押在估计准不准上。

**本文目标**：设计一个在线算法，既动态选阈值实现长期覆盖、不对数据做任何分布假设，又尽量把区间做窄。

**切入角度**：作者注意到，在线 CP 的反应式更新 $\hat q_{t+1}=\hat q_t-\eta\nabla\ell_{1-\alpha}(s_t-\hat q_t)$ 本质是对分位数损失做在线（次）梯度下降（OGD）。而在线优化里早有「乐观在线梯度下降」（OOGD）：在目标里加入对下一步梯度的**预判项（hint）**，预判越准后悔越小。如果能把「估计的分数分布」作为这个 hint 注入进来，就能在保留 OGD 覆盖保证的同时多用一份结构信息。

**核心 idea**：在每步反应式更新之外，加一步基于估计 CDF 的**乐观修正**——把 $\hat F_{t+1}(\hat q_{t+1})-(1-\alpha)$ 当 hint 去预判下一步该往哪挪，从 OOGD 视角统一分析，从而同时收紧后悔界与覆盖界。

## 方法详解

### 整体框架
COP 的输入是逐步到来的数据 $\{(X_t,Y_t)\}$ 和一个基预测器 $\hat f_t$，输出是每个时刻的预测区间 $\hat C_t(X_t)=[\hat f_t(X_t)-q_t,\ \hat f_t(X_t)+q_t]$。每个时刻它维护**两套半径**：一个「主半径」$\hat q_t$ 和一个真正用来出区间的「修正半径」$q_t$。

一个时刻 $t$ 的完整流程是：① 观察输入 $X_t$，用当前修正半径 $q_t$ 输出区间；② 观察真标签 $Y_t$，算出非一致性分数 $s_t$（如 $s_t=|Y_t-\hat f_t(X_t)|$）以及「这次有没有覆盖」的指示 $\mathrm{err}_t=\mathbf 1\{s_t>q_t\}$；③ 用 OGD 反应式更新主半径 $\hat q_{t+1}=\hat q_t+\eta(\mathrm{err}_t-\alpha)$；④ 用最近窗口的分数估一个 CDF $\hat F_{t+1}$；⑤ 在主半径基础上做一步乐观修正得到 $q_{t+1}=\hat q_{t+1}-\lambda_{t+1}(\hat F_{t+1}(\hat q_{t+1})-(1-\alpha))$。注意指示量用的是 $\mathrm{err}_t=\mathbf 1\{s_t>q_t\}$（基于真正输出的修正半径），而非旧方法里的 $\mathbf 1\{s_t>\hat q_t\}$——修正步因此被纳入了覆盖反馈回路。

关键在于：主半径 $\hat q$ 负责「守住长期覆盖」（它就是标准 OGD，保证不丢），修正半径 $q$ 负责「在守住的前提下用分布信息把区间收窄」。当 $\lambda_{t+1}=0$ 时 COP 退化成普通 OGD，所以它是 OGD 的严格超集。这是一个纯在线、纯一阶的迭代方法，没有多模块协同、没有 pipeline 分支，机制靠两条更新公式说清即可，故不画框架图。

### 关键设计

**1. 基于估计 CDF 的乐观修正步：把可预测结构注入而不破坏保证**

针对「反应式更新太保守、区间偏宽」的痛点，COP 在每步主更新后加一个修正。出发点是：若分数 $s_t\mid S_{t-1}$ 的条件分布随 $t$ 大致不变，CP 的目标就近似等于找 $s_t$ 的条件 $(1-\alpha)$ 分位数。于是对修正半径求解一个带二次正则的目标 $q_{t+1}=\arg\min_q \mathbb E_{s_{t+1}}[\ell_{1-\alpha}(s_{t+1}-q)\mid S_t]+\frac{1}{2\lambda_{t+1}}\|q-\hat q_{t+1}\|_2^2$。由于 $\nabla_q L_{t+1}(q)=F_{t+1}(q)-(1-\alpha)$，这个隐式更新无闭式解，作者改为对 $L_{t+1}$ 在 $\hat q_{t+1}$ 处做一阶线性近似，并把真 CDF $F_{t+1}$ 换成估计 $\hat F_{t+1}$，得到闭式修正

$$q_{t+1}=\hat q_{t+1}-\lambda_{t+1}\big(\hat F_{t+1}(\hat q_{t+1})-(1-\alpha)\big).$$

直觉很清晰：若估计 CDF 显示当前半径处的累积概率高于目标 $1-\alpha$（覆盖「过头」、区间偏宽），就把半径往下压；反之则放大。相比 SPCI / Lee 等直接取 $\hat F^{-1}(1-\alpha)$ 的做法，COP **不需要对估计 CDF 求逆**，省去数值误差和计算成本，并且只是「在 OGD 输出上挪一小步」，而非把命运全交给估计——这正是它既窄又稳的来源。默认用经验 CDF $\hat F_{t+1}(\hat q_{t+1})=\frac1w\sum_{i=t-w+1}^{t}\mathbf 1\{s_i\le \hat q_{t+1}\}$（窗口 $w=100$），也可换核密度估计。

**2. OOGD 视角与分布感知 hint：用一个 scale factor 表达「我多信任这份估计」**

COP 的两步更新可以严格改写成乐观在线梯度下降的形式：先 $\hat q_{t+1}=\arg\min_q\{\eta\langle\nabla\ell_{1-\alpha}(s_t-q_t),q\rangle+\frac12\|q-\hat q_t\|^2\}$，再 $q_{t+1}=\arg\min_q\{\eta\langle M_{t+1},q\rangle+\frac12\|q-\hat q_{t+1}\|^2\}$。其中 $M_{t+1}$ 是 OOGD 里的乐观项（对下一步的预判）。经典 OOGD 常把上一步梯度当 hint，而 COP 用的是**分布感知 hint**

$$M_{t+1}=\big(\hat F_{t+1}(\hat q_{t+1})-(1-\alpha)\big)\cdot \lambda_{t+1}/\eta,$$

它刻画了下一个分数可能的分布漂移方向。比例因子 $\lambda_{t+1}/\eta\le 1$ 被命名为 scale factor，直接表达「对 $\hat F_{t+1}$ 的信任程度」：基模型很强、分数高度随机难预测时，调小 $\lambda$（少信估计、退回反应式）；$\hat F$ 可靠时让 scale factor 接近 1。把它统一到 OOGD 框架的意义在于——OOGD 有成熟的「hint 越准、后悔越小」理论，这给后面的联合界提供了分析抓手。

**3. 联合后悔—覆盖界：证明好的 hint 同时收紧两者**

在线 CP 通常分别看覆盖和动态后悔，而且二者一般不互相蕴含。本文从 OOGD 视角导出一个**联合界**（定理 1，常数步长）：

$$\underbrace{\tfrac1T\sum_t[\ell_t(q_t)-\ell_t(u_t)]}_{\text{后悔}}+\underbrace{\tfrac{\eta(1-2\alpha)}{4}\big(\tfrac{\sum_t\mathrm{err}_t}{T}-\alpha\big)}_{\text{覆盖}}\le \tfrac{\eta}{T}\sum_t\|\alpha-\mathrm{err}_t-M_t\|_2^2+\underbrace{\sum_t\tfrac{1}{2\eta}\big(\|u_t-\hat q_t\|^2-\|u_{t-1}-\hat q_t\|^2\big)}_{\text{环境非平稳性}}.$$

右边第一项里出现了 $\|\alpha-\mathrm{err}_t-M_t\|^2$，说明**只要把 $M_t$ 选得接近 $F_t(\hat q_t)-(1-\alpha)$，就能同时压低后悔界和覆盖界**——而这正好对应 scale factor 取 1 时 COP 的选择。这条界从理论上回答了「为什么注入分布 hint 是有益的」，是本文的核心理论贡献，而不仅是经验观察。

**4. 分布无关的有限样本覆盖 + 渐近一致性：估计再差也不丢底线**

COP 的覆盖保证完全不依赖 $\hat F$ 准不准。命题 2（固定步长 $\eta$，分数 $s_t\in[0,B]$、乐观项 $M_t\in[-M,M]$ 有界）给出有限样本界 $\big|\frac1T\sum_t\mathrm{err}_t-\alpha\big|\le \frac{B+(2+6M)\eta}{T\eta}$，且运行算法**无需知道** $B,M$ 的具体上界；$T\to\infty$ 即得长期覆盖 $\frac1T\sum\mathrm{err}_t\to\alpha$。定理 2 把它推广到任意动态步长 $\eta_t$，只要增大步长的次数 $N_T$ 满足 $N_T/(\min_t\eta_t)=o(T)$ 就仍收敛。更进一步，定理 3 证明：当分数 i.i.d.、步长满足 Robbins–Monro 条件 $\sum_t\eta_t=\infty,\ \sum_t\eta_t^2<\infty$ 时，$q_t$ **收敛**到真 $(1-\alpha)$ 分位数 $q^*$——这恰好克服了固定步长 OGD 必然震荡的毛病，而且即便每步都加了有界乐观项也不影响收敛。配合命题 1（在「同号」假设和 $F$ Lipschitz 下，修正后的期望分位数损失不大于修正前），这组结果说明：估计准了能收窄、估计偏了至多退回基线，下限被牢牢守住。

### 一个完整示例
设目标覆盖 $1-\alpha=90\%$，某时刻主半径已更新到 $\hat q_{t+1}=8.5$，scale factor $\lambda_{t+1}=0.5$。若最近 100 个分数里有 95 个 $\le 8.5$，则经验 CDF $\hat F_{t+1}(8.5)=0.95$，hint $=0.95-0.90=0.05>0$，说明这个半径覆盖「过头」、区间偏宽。修正得 $q_{t+1}=8.5-0.5\times0.05=8.475$，区间被略微收窄后输出。下一步看 $s_{t+1}$ 是否落入 $[\,\hat f-8.475,\ \hat f+8.475\,]$ 得到 $\mathrm{err}_{t+1}$，再喂回主更新——反应式的 $\hat q$ 负责长期纠偏不丢覆盖，乐观的 $q$ 负责在此之上借分布信息抠掉多余宽度。

### 损失函数 / 训练策略
COP 无需训练额外模型（不像 Conformal PID 要训 scorecaster）。核心超参三个：基础学习率 $\eta$、scale factor $\lambda=0.5$、窗口长度 $w=100$；并沿用前作的自适应步长 $\eta_t=\eta\cdot(\max\{s_{t-w+1..t}\}-\min\{s_{t-w+1..t}\})$。优化对象是 $(1-\alpha)$ 分位数损失 $\ell_{1-\alpha}(q)=(\mathbf 1\{q>0\}-\alpha)q$，实现上对上下两侧各按 $\alpha/2$ 出非对称区间。

## 实验关键数据

### 主实验
目标覆盖 $1-\alpha=90\%$，对比 7 个 SOTA：ACI、OGD、SF-OGD、decay-OGD、Conformal PID、ECI、LQT(fixed)；三种基预测器 Prophet / AR / Theta。下表节选两个模拟数据集（changepoint、distribution drift）在 Prophet 下的结果，指标为覆盖率与平均/中位区间宽度（宽度越小越好，覆盖越接近 90% 越好）。

| 数据集 | 方法 | 覆盖率(%) | 平均宽度 | 中位宽度 |
|--------|------|-----------|----------|----------|
| Changepoint | ACI | 89.9 | ∞ | 8.20 |
| Changepoint | decay-OGD | 90.0 | 8.30 | 8.22 |
| Changepoint | ECI | 89.9 | 8.16 | 8.25 |
| Changepoint | **COP** | 89.8 | 8.29 | 8.44 |
| Dist. Drift | ACI | 89.9 | ∞ | 6.69 |
| Dist. Drift | decay-OGD | 90.6 | 7.64 | 6.95 |
| Dist. Drift | ECI | 90.0 | 7.27 | 6.98 |
| Dist. Drift | **COP** | 90.6 | **7.07** | **6.89** |

ACI 因更新 $\alpha_t$ 常输出无限宽区间；SF-OGD 宽度明显偏大（changepoint 下 ~12.5）；LQT 依赖网格搜索、不稳定（drift 下覆盖飘到 91.9%）。COP 在维持 ~90% 覆盖的同时拿到最紧或接近最紧的宽度，尤其在分布漂移下平均宽度 7.07 低于所有基线。四个真实数据集（Amazon/Google 股价、新南威尔士电力需求、德里气温）结论一致，且对三种基预测器都稳健。

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| $\lambda=0$（退化为 OGD） | 区间偏宽 | 去掉乐观修正即纯反应式 |
| ECDF（默认） vs 核密度 CDF | 相近 | 估计器选择鲁棒（附录 C） |
| 不准的估计 CDF | 覆盖仍守住 | 对应命题 2，估计差只退回基线（附录 G） |
| scale factor $\lambda$ 取值 | 影响收窄幅度 | $\lambda$ 越大越信估计、收窄越多（附录 H） |

### 关键发现
- 收窄主要来自乐观修正步：$\lambda=0$ 时退回 OGD 的偏宽区间，说明区间变紧确实是分布 hint 的贡献。
- 覆盖保证对估计质量不敏感：即便 CDF 估计不准（附录 G），长期覆盖仍贴近 90%，印证「估计差只退回基线、不破坏底线」。
- 对基预测器通用：Prophet / AR / Theta 三种不同强度的预测器下，COP 都能维持覆盖并收窄，说明改进与基模型正交。

## 亮点与洞察
- **把在线 CP 重新解读为 OOGD，是连接「经验改进」与「理论保证」的桥**：一旦看出反应式更新是分位数损失上的 OGD，乐观修正就自然是 OOGD 的 hint，于是 OOGD 现成的后悔分析直接可用——这种「换个视角统一框架」的招很值得迁移。
- **scale factor 这个旋钮设计得很干净**：用一个 $\lambda/\eta\le 1$ 同时表达「信不信估计」和「插值反应式↔分布式」，$\lambda=0$ 退回 OGD、$\lambda$ 大则更激进，可解释、易调。
- **联合后悔—覆盖界**：在「后悔与覆盖一般互不蕴含」的前提下，证明恰当的 hint 能同时压低两者，把直觉变成了定理。
- **避免对 CDF 求逆**这个工程细节，既省算力又躲开数值误差，是对 SPCI/Lee 路线的实在改进。

## 局限与展望
- 命题 1 的「同号」假设（$\hat F$ 与 $F$ 相对 $1-\alpha$ 的排序一致）较强，作者在附录 A.3 给了一个绕开它的小改版，但主分析仍以此为前提。
- 渐近一致性（定理 3）依赖分数 **i.i.d.** 假设，这与时间序列强相关的实际场景有距离；真正的强相关下只有长期覆盖保证、无收敛保证。
- 估计 CDF 用固定窗口经验 CDF，窗口长度 $w$ 与 scale factor $\lambda$ 仍是需调的超参；论文未深入讨论它们在剧烈漂移下如何自适应。
- 仅在一维区间回归型分数上验证；扩展到分类、多维输出、更复杂分数函数的表现待考。

## 相关工作与启发
- **vs ACI / OGD / decay-OGD**：它们是纯反应式在线 CP，只靠迭代步纠偏、不看数据模式，区间保守且固定步长会震荡；COP 在其上加分布感知修正，既收窄又（i.i.d. 下）收敛，且 $\lambda=0$ 时严格退化为它们。
- **vs Conformal PID / ECI**：PID 需训 scorecaster 且选型缺乏原则、可能引入额外方差，ECI 靠误差量化反应快但抓不住底层分布结构；COP 不训额外模型、直接用估计 CDF 注入结构。
- **vs SPCI / Lee 等（直接估分布求逆）**：它们靠估计分位数/CDF 出区间，依赖模型一致性、分布光滑等假设、非分布无关，且要对 CDF 求逆；COP 只把估计当 hint 做一步修正，分布无关、有限样本覆盖、无需求逆。
- **vs LQT / HOP-CPT / CT-SSF**：LQT 依赖线性自回归结构且超参敏感，后两者用 Hopfield 网络/深度表示加权但缺有限样本保证；COP 在「利用可预测信息」与「保住覆盖保证」之间取得更干净的平衡。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把在线 CP 重述为 OOGD 并用分布感知 hint，是一个清爽且有理论支撑的新角度。
- 实验充分度: ⭐⭐⭐⭐ 5 模拟 + 4 真实数据 × 3 基预测器 × 7 基线，附录还覆盖估计器/scale factor/不准 CDF 等消融。
- 写作质量: ⭐⭐⭐⭐ 从 OGD→OOGD 的动机推导清晰，理论与算法衔接顺畅。
- 价值: ⭐⭐⭐⭐ 给时序不确定性量化提供了一个即插即用、保证完备、能收窄区间的在线方法。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Conformal Prediction for Long-Tailed Classification](conformal_prediction_for_long-tailed_classification.md)
- [\[ICLR 2026\] Online Conformal Prediction with Adversarial Semi-bandit Feedback via Regret Minimization](online_conformal_prediction_with_adversarial_semi-bandit_feedback_via_regret_min.md)
- [\[ICLR 2026\] Conformal Prediction with Corrupted Labels: Uncertain Imputation and Robust Re-weighting](conformal_prediction_with_corrupted_labels_uncertain_imputation_and_robust_re-we.md)
- [\[ICML 2026\] Enhancing Conformal Prediction via Class Similarity](../../ICML2026/learning_theory/enhancing_conformal_prediction_via_class_similarity.md)
- [\[ICLR 2026\] Singleton-Optimized Conformal Prediction](singleton-optimized_conformal_prediction.md)

</div>

<!-- RELATED:END -->
