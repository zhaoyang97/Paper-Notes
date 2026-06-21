---
title: >-
  [论文解读] Off-Policy Evaluation for Ranking Policies under Deterministic Logging Policies
description: >-
  [ICLR 2026][推荐系统][Off-Policy Evaluation] 针对工业排序系统常用「完全确定性」日志策略导致传统 IPS 类估计器严重偏置的痛点，本文提出用「用户点击概率之比」代替「策略概率之比」作为重要性权重的 CIPS 估计器（及其双重稳健扩展 CDR），把无偏性所需的支撑条件从「日志策略要足够随机」放宽到「点击行为本身有随机性」，从而在确定性日志下实现低偏甚至无偏评估。
tags:
  - "ICLR 2026"
  - "推荐系统"
  - "Off-Policy Evaluation"
  - "排序策略"
  - "确定性日志策略"
  - "点击重要性权重"
  - "双重稳健"
---

# Off-Policy Evaluation for Ranking Policies under Deterministic Logging Policies

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=0ZkWWxcHKV](https://openreview.net/forum?id=0ZkWWxcHKV)  
**代码**: 随论文以补充材料形式提供（supplementary material）  
**领域**: 推荐系统 / 离线策略评估 (OPE) / 排序  
**关键词**: Off-Policy Evaluation, 排序策略, 确定性日志策略, 点击重要性权重, 双重稳健

## 一句话总结
针对工业排序系统常用「完全确定性」日志策略导致传统 IPS 类估计器严重偏置的痛点，本文提出用「用户点击概率之比」代替「策略概率之比」作为重要性权重的 CIPS 估计器（及其双重稳健扩展 CDR），把无偏性所需的支撑条件从「日志策略要足够随机」放宽到「点击行为本身有随机性」，从而在确定性日志下实现低偏甚至无偏评估。

## 研究背景与动机
**领域现状**：推荐、搜索、电商等系统本质上是「上下文 bandit」过程——排序策略 $\pi$ 反复观察上下文 $x$、产出一个排序 $A=(a_1,\dots,a_K)$、再观察奖励。离线策略评估（Off-Policy Evaluation, OPE）希望只用旧的日志策略 $\pi_0$ 收集到的数据 $D=\{(x_i,A_i,C_i,C_iR_i)\}$，估计新策略 $\pi$ 上线后的期望价值 $V(\pi)$，从而避免反复做有风险的 A/B 测试。排序场景下主流估计器是 ranking-wise IPS、position-wise 的 IIPS、cascade 模型下的 RIPS——它们都通过「新旧策略选中同一排序/同一位置动作的概率之比」做重要性加权。

**现有痛点**：所有这些 IPS 类估计器都要求日志策略 $\pi_0$ **足够随机**才能无偏。ranking-wise IPS 需要 $\pi_0$ 探索过新策略可能选的所有排序（ranking-wise common support）；IIPS 需要 $\pi_0$ 在每个位置都选过所有动作（position-wise common support）；RIPS 在 cascade 假设下需要前缀支撑。但大规模工业系统由于动作空间巨大，**随机策略既低效又有风险，往往部署完全确定性策略**——给定 $x$ 只输出唯一排序。此时这些支撑条件被严重违反，估计器产生巨大偏置。

**核心矛盾**：现有方法把「重要性权重」的随机性来源**唯一地寄托在日志策略上**。一旦 $\pi_0$ 确定，$\pi_0(A|x)\in\{0,1\}$，绝大多数新策略想评估的排序在日志里概率为 0，权重无定义。论文用 Theorem 2.1 量化了这种偏置：IPS 的偏置等于 $-\mathbb{E}_{p(x)}\big[\sum_{A\in U_0(x,\pi_0)}\pi(A|x)\sum_k q_k(x,A)\big]$，其中 $U_0$ 是不被支撑的排序集合——确定性日志下 $U_0$ 几乎包含所有排序，偏置随之巨大（一个 toy 例子里 6 个排序有 5 个无支撑）。这与传统 OPE 文献关注的「大动作空间导致高方差」完全不同：**确定性日志下方差不是主要矛盾，偏置才是**。

**本文目标**：在完全确定性日志策略下，给出一个偏置可控、理论上可无偏的排序 OPE 估计器。

**切入角度**：作者观察到一个被忽略的随机性来源——**即便日志策略确定地给出一个排序，用户是否点击每个动作仍然是随机的**。排序里每个动作都以某个非零点击概率被用户查看/点击，这种「用户点击行为的内在随机性」与日志策略是否随机无关。

**核心 idea**：用「点击概率之比」代替「策略概率之比」做重要性加权——既然 $\pi_0$ 不再提供随机性，就改用点击概率 $p_c(x,a,\pi)$ 当新的重要性权重的载体，把无偏所需的支撑条件从「策略支撑」换成更容易满足的「点击支撑」。

## 方法详解

### 整体框架
论文要解决的是：在 $\pi_0$ 完全确定、传统重要性权重失效的情况下，如何无偏估计 $V(\pi)=\mathbb{E}_{p(x)\pi(A|x)}\big[\sum_{k=1}^K q_k(x,A)\big]$。整体思路只有一步关键替换，但环环相扣：

第一步，**把价值的写法从「按位置求和」改写成「按动作求和」**。原定义 $V(\pi)=\mathbb{E}[\sum_k C(k)R(k)]$ 按排序位置 $k$ 累加，作者等价改写为 $V(\pi)=\mathbb{E}_{p(x)\pi(A|x)p(C,R|x,A)}\big[\sum_{a\in A}C(a)R(a)\big]$，其中 $C(a)$、$R(a)$ 是动作 $a$ 的点击指示和潜在奖励。这只是同一个量的换元（不引入任何独立性假设），目的是让「点击」成为分析的主角。

第二步，**用边缘化点击概率定义新的重要性权重**。定义 $p_c(x,a,\pi)=\mathbb{E}_{\pi(A|x)}[\,\mathbb{E}[C(a)|x,A]\,]$ 为在策略 $\pi$ 下、上下文 $x$ 的用户点击动作 $a$ 的边缘概率。用 $p_c(x,a,\pi)/p_c(x,a,\pi_0)$ 取代 $\pi(A|x)/\pi_0(A|x)$，得到 CIPS 估计器。

第三步，**理论上确立无偏的新条件**（click-wise common support + 潜在奖励独立），并把估计器扩展成双重稳健版 CDR，在不引入额外偏置的前提下降方差。

这里需要先厘清论文建模的「两阶段奖励」结构：用户先**曝光/点击**（click vector $C$），点击后才产生**下游潜在奖励**（potential reward $R$，如购买、播放时长）。我们只能观察到 $C$ 和 $CR$——即只有 $C(k)=1$ 时才看得到 $R(k)$，$R$ 永远无法单独完整观测。这正是 CIPS 能立足的物理基础：点击这一层天然带噪、带随机性。

### 关键设计

**1. CIPS：用点击概率之比作为新的重要性权重**

传统 IPS 的偏置根源是把随机性押在确定性的 $\pi_0$ 上。CIPS 直接换掉权重的载体——不再问「$\pi_0$ 以多大概率选出这个排序」，而问「在 $\pi_0$ 排出的排序里、用户以多大概率点击动作 $a$」。估计器写作

$$\hat{V}_{\text{CIPS}}(\pi;D)=\frac{1}{n}\sum_{i=1}^n\sum_{a\in A}\frac{p_c(x_i,a,\pi)}{p_c(x_i,a,\pi_0)}\,C_i(a)R_i(a),$$

其中 $p_c(x,a,\pi)=\mathbb{E}_{\pi(A|x)}[\mathbb{E}[C(a)|x,A]]$ 是边缘化点击概率。直觉上，即使 $\pi_0$ 确定地只排出一种 $A_1$，用户在 $A_1$ 里点击 $a_1,a_2,a_3$ 的概率（如 $0.8,0.5,0.2$）都是非零的；新策略 $\pi$ 是多种排序的混合，边缘点击概率 $p_c(x,a,\pi)$ 也非零。于是权重 $p_c(x,a,\pi)/p_c(x,a,\pi_0)$ 几乎处处有定义（toy 例子里 $a_1,a_2,a_3$ 的权重分别是 $0.6875,0.78,2.4$），而同一例子下 IPS 的权重 5/6 无定义、IIPS 6/9 无定义。这就是 CIPS 能在确定性日志下「活下来」的本质原因。

**2. Click-wise common support + 潜在奖励独立：把无偏条件放宽到点击层**

CIPS 无偏需要两个比旧条件温和得多的假设。其一是 **click-wise common support**（Condition 3.1）：$p_c(x,a,\pi)>0\Rightarrow p_c(x,a,\pi_0)>0$，即只要新策略下会被点击的动作、在日志策略下也有非零点击概率即可。由于点击概率本身带随机性，这个条件即便 $\pi_0$ 完全确定也通常成立——而 ranking-wise / position-wise 支撑条件在确定性日志下「永远」不成立。其二是**潜在奖励独立**（Condition 3.2）：$\mathbb{E}[R(a)|x,A]=\mathbb{E}[R(a)|x]=q_r(x,a)$，即点击后的下游奖励只取决于动作 $a$ 自身、与同排序里别的动作无关（如电商里「点进某商品后是否购买」只看该商品属性）。这比 IIPS 要求的「点击层独立」更现实。在这两条下，Theorem 3.1 证明 CIPS 无偏 $\mathbb{E}_{p(D)}[\hat{V}_{\text{CIPS}}]=V(\pi)$，这是已知**首个能在完全确定性日志下给出低偏排序策略评估的方法**。

实践中真实点击概率未知、需从数据估计 $\hat{p}_c$。Theorem 3.2 给出此时的偏置：

$$\text{Bias}[\hat{V}_{\text{CIPS}}]=\mathbb{E}_{p(x)}\Big[\sum_{a\in A}p_c(x,a,\pi_0)\Big(\frac{\hat{p}_c(x,a,\pi)}{\hat{p}_c(x,a,\pi_0)}-\frac{p_c(x,a,\pi)}{p_c(x,a,\pi_0)}\Big)q_r(x,a)\Big],$$

关键结论是：偏置只取决于点击概率**之比**估得准不准，而非点击概率本身估得准不准——即便 $\hat{p}_c$ 有误差，只要新旧之比稳定，偏置就小。这让 CIPS 在仅靠可观测数据估点击概率时仍然实用。

**3. CDR：在不增加偏置的前提下用回归模型降方差**

CIPS 解决了偏置，但其方差（Theorem 3.3）取决于点击重要性权重的量级，且对未点击动作相当于「零填充」奖励，仍有改进空间。作者把 CIPS 套进双重稳健框架，引入一个潜在奖励回归模型 $\hat{q}_r(x,a)$，得到 Click-based Doubly Robust (CDR)：

$$\hat{V}_{\text{CDR}}(\pi;D)=\frac{1}{n}\sum_{i=1}^n\sum_{a\in A}\frac{p_c(x_i,a,\pi)}{p_c(x_i,a,\pi_0)}\big(C_i(a)R_i(a)-p_c(x_i,a,A)\hat{q}_r(x_i,a)\big)+\mathbb{E}_{\pi(A|x_i)}\big[p_c(x_i,a,A)\hat{q}_r(x_i,a)\big].$$

Theorem 4.1/4.2 证明 CDR 与 CIPS **偏置完全相同**（与回归模型是否准确无关），因此回归模型不会破坏无偏性；而 Theorem 4.3 的方差比 CIPS 多出一个由回归误差 $\Delta_r(x,a)=q_r(x,a)-\hat{q}_r(x,a)$ 主导的项——当 $\hat{q}_r$ 比简单零填充更准时，CDR 方差严格小于 CIPS。这就是经典 DR「无偏 + 降方差」思路在点击权重框架下的迁移：用模型预测填补未点击动作的奖励基线，残差仍用点击重要性加权来校正。

### 损失函数 / 训练策略
论文核心是估计器而非训练目标。唯一需要拟合的是点击概率模型 $\hat{p}_c$（实验用 3 层神经网络在日志数据 $D$ 上估计）和 CDR 里的潜在奖励回归 $\hat{q}_r$。CIPS/CDR 本身无训练损失，是闭式的加权平均估计量；如需进一步降方差，可叠加权重截断（clipping）、自归一化（self-normalization）等标准技巧。

## 实验关键数据

实验全部围绕「确定性日志策略」这一核心场景，指标是相对真实价值 $V(\pi)$ 归一化后的 MSE、平方偏置（Squared Bias）、方差（Variance），跑 100 个随机种子取平均、bootstrap 给 95% 置信区间。合成数据默认 $n=1000$、动作数 6、排序长度 $K=6$、独立性违反参数 $\lambda=0.5$、日志策略用 Plackett–Luce 模型并设 $\alpha=\infty$（完全确定）。对比方法为 IPS、IIPS、RIPS，外加 CIPS (true CTR)（用真实点击概率，仅作参考上界）。

### 主实验（合成数据，确定性日志）

| 变化维度 | 关键现象 | 结论 |
|----------|----------|------|
| 日志规模 $n$（500→4000） | CIPS 在所有规模上 MSE 显著低于 baseline，主要靠降偏置；baseline 方差小但偏置巨大 | CIPS 全程占优，且接近 CIPS (true CTR)，估点击概率只引入很小额外偏置 |
| 排序长度 $K$ | baseline 偏置随 $K$ 增大而上升，CIPS 偏置始终低；CIPS 方差略升但不主导 MSE | CIPS 对排序长度稳健 |
| 独立性违反 $\lambda$ | $\lambda$ 增大（违反 Condition 3.2 加重）时 CIPS 的 MSE 仅小幅上升 | Condition 3.2 不严格成立时 CIPS 仍最优、退化温和 |
| 日志随机性 $\alpha$ | 确定性用户比例从 0.07 升到 0.93；$\alpha$ 大（更确定）时 baseline 偏置剧增，CIPS 低偏 | $\alpha$ 小（更随机）时 IIPS 与 CIPS 持平——这正符合 CIPS 主打确定性场景的定位 |

### 真实数据（KuaiRec）
在快手视频推荐数据集 KuaiRec（对部分用户/物品近 100% 稠密的全观测交互矩阵）上构造 OPE 实验，用真实交互矩阵当 $q_r$、按阈值定义点击概率。设 $\epsilon=0.0$ 让新旧策略**都确定**（最难设置），动作数 10、$K=6$、$\alpha=\infty$。

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 变 $n$（图 5） | CIPS 的 MSE 始终低于 baseline | 虽有不可忽略的偏置（源于新旧策略都确定时 click-wise 支撑也被部分违反），但仍远好于完全失效的 baseline |
| 变 $\alpha$（图 6） | 日志越确定，CIPS 相对优势越大 | 所有估计器偏置都随 $\alpha$ 增长，但 CIPS 增长最慢、抑制最有效 |

### 关键发现
- **偏置才是确定性日志下的主敌**：baseline 方差一直很小（确定性日志下方差本就不是问题），MSE 几乎全由偏置贡献，CIPS 的增益来自把这块偏置大幅压下去。
- **点击概率「之比」比「绝对值」更重要**：CIPS 接近 CIPS (true CTR)，印证 Theorem 3.2——只要新旧点击概率之比估得稳，绝对点击概率估不准也无妨。
- **CDR 进一步降方差**（附录 D.2）：在 CIPS 基础上加回归模型，方差进一步下降且不增偏置，并改善了下游的 policy selection。
- **退化边界诚实**：当新旧策略都确定时 click-wise 支撑也会部分失效，CIPS 出现非零偏置——但这是「任何估计器都极难」的极端设置，CIPS 仍是相对最优。

## 亮点与洞察
- **换掉重要性权重的「随机性来源」是一个可迁移的范式**：传统 OPE 默认随机性必须来自行为策略；本文指出在排序/两阶段反馈场景里，用户点击行为本身就是一个独立的随机性来源，可以拿来当权重载体。这个视角对任何「行为侧确定、但反馈侧有噪」的离线评估问题都有启发。
- **把无偏条件「下沉一层」**：从 ranking 层支撑 → position 层支撑 → click 层支撑，支撑条件越往下越容易满足，因为越接近真实带噪的观测层。这是一种用「更弱但更现实的假设」换「在更恶劣环境下可用」的典型权衡。
- **DR 框架的干净嫁接**：CDR 证明了「换权重载体」与「双重稳健降方差」可以正交组合——偏置只由权重决定、方差才由回归模型影响，理论分解非常清爽。
- **挑战长期成见**：作者明确点出本工作动摇了「确定性日志下无法做准确 OPE」这一长期信念，把一个此前被认为 intractable 的设置变得可做。

## 局限与展望
- **依赖点击概率可估且新旧之比稳定**：当点击概率模型在新旧策略下系统性偏差不一致时，Theorem 3.2 的偏置不再小；真实数据上当新旧策略都确定时已观察到非零偏置。
- **Condition 3.2（潜在奖励独立）是简化假设**：现实中点击后的下游奖励可能受同排序其他动作影响（如比价、组合购买），实验显示违反时退化温和，但极端违反下的行为未充分刻画。
- **仅在中小动作空间/排序长度上验证**：合成实验动作数 6、$K=6$，真实数据动作数 10；工业级千万量级动作空间下点击概率模型的可估性与权重稳定性仍需检验。
- **未与「大动作空间 embedding 边缘化」类方法正面对比**：那类方法目标是降方差、需额外 action embedding，本文目标是降偏置、不需 embedding，但两者在「既大又确定」的场景能否结合值得探索。

## 相关工作与启发
- **vs ranking-wise IPS / IIPS / RIPS**：它们用策略概率之比加权，无偏依赖日志策略随机性（ranking/position/cascade 支撑），确定性日志下支撑条件「永远」违反、偏置巨大；CIPS 改用点击概率之比，支撑条件下沉到 click 层，确定性日志下仍低偏。
- **vs deficient support 类方法**（Sachdeva et al. 2020; Felicioni et al. 2022）：它们处理日志策略对部分动作给零概率的「支撑不足」；本文处理更极端的「完全确定性」特例——deficient support 的严格子集，消除了所有随机性，需要专门设计。
- **vs 大动作空间 embedding 边缘化**（Saito & Joachims 2022; Saito et al. 2023; Taufiq et al. 2023）：两者都「边缘化重要性权重」，但前者用观测到的 action embedding 防权重爆炸、目标是降方差且需额外信息；CIPS 用点击概率边缘化、目标是解决确定性日志的偏置且不需 embedding。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个把无偏 OPE 推进到完全确定性日志策略、用点击随机性替代策略随机性，问题与解法都是新的
- 实验充分度: ⭐⭐⭐⭐ 合成数据从 $n$/$K$/$\lambda$/$\alpha$ 四个维度系统验证 + KuaiRec 真实数据，但动作空间规模偏小、缺与大动作空间方法的正面对比
- 写作质量: ⭐⭐⭐⭐⭐ 动机—理论—估计器—实验逻辑清晰，定理对偏置/方差刻画完整，toy 例子直观
- 价值: ⭐⭐⭐⭐⭐ 直击工业排序系统普遍部署确定性策略这一现实痛点，把「确定性日志下无法准评」的成见打破，落地意义强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Learning Design Skills as Memory Policies for Agentic Photonic Inverse Design](../../ICML2026/recommender/learning_design_skills_as_memory_policies_for_agentic_photonic_inverse_design.md)
- [\[ICLR 2026\] GoalRank: Group-Relative Optimization for a Large Ranking Model](goalrank_group-relative_optimization_for_a_large_ranking_model.md)
- [\[ICML 2025\] SIMPLEMIX: Frustratingly Simple Mixing of Off- and On-policy Data in Language Model Preference Learning](../../ICML2025/recommender/simplemix_frustratingly_simple_mixing_of_off-_and_on-policy_data_in_language_mod.md)
- [\[ICLR 2026\] From Evaluation to Defense: Advancing Safety in Video Large Language Models](from_evaluation_to_defense_advancing_safety_in_video_large_language_models.md)
- [\[NeurIPS 2025\] Validating LLM-as-a-Judge Systems under Rating Indeterminacy](../../NeurIPS2025/recommender/validating_llm-as-a-judge_systems_under_rating_indeterminacy.md)

</div>

<!-- RELATED:END -->
