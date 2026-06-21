---
title: >-
  [论文解读] Learning Correlated Reward Models: Statistical Barriers and Opportunities
description: >-
  [ICLR 2026][学习理论][Random Utility Model] 本文证明了 RLHF 里主流的成对偏好数据**根本无法**学到用户效用之间的相关性，而 best-of-three（三选一排序）数据既必要又充分，并据此给出了相关 probit 模型的首个可识别性结果和一个近最优的多项式时间估计器。
tags:
  - "ICLR 2026"
  - "学习理论"
  - "偏好建模"
  - "RLHF 奖励模型"
  - "Random Utility Model"
  - "相关 probit 模型"
  - "IIA 假设"
  - "best-of-three 偏好"
  - "可识别性"
  - "样本复杂度"
---

# Learning Correlated Reward Models: Statistical Barriers and Opportunities

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=TbEyl6krsY](https://openreview.net/forum?id=TbEyl6krsY)  
**代码**: 待确认  
**领域**: 学习理论 / 偏好建模 / RLHF 奖励模型  
**关键词**: Random Utility Model, 相关 probit 模型, IIA 假设, best-of-three 偏好, 可识别性, 样本复杂度  

## 一句话总结
本文证明了 RLHF 里主流的成对偏好数据**根本无法**学到用户效用之间的相关性，而 best-of-three（三选一排序）数据既必要又充分，并据此给出了相关 probit 模型的首个可识别性结果和一个近最优的多项式时间估计器。

## 研究背景与动机
**领域现状**：随机效用模型（Random Utility Model, RUM）是建模人类偏好的经典框架，从 1960 年代的数学心理学、计量经济学一路用到今天的 RLHF 奖励建模。最常用的是 logit / Bradley-Terry 模型——它是唯一满足"无关备选独立性"（Independence of Irrelevant Alternatives, IIA）的 RUM，可以用成对比较学习、用 softmax 高效计算。

**现有痛点**：IIA 假设在 LLM 语境下意味着所有用户共享同一个潜在效用函数，把丰富的人类偏好坍缩成一个粗糙的全局排序。经典的"红巴士/蓝巴士"悖论说明：当效用之间存在相关性时，IIA 会给出自相矛盾的反事实预测。要捕捉个性化偏好就得放弃 IIA、显式建模效用相关性（例如相关 probit 模型 $X\sim\mathcal{N}(\mu,\Sigma)$）。

**核心矛盾**：相关 probit 模型虽然表达力强，但学习它的统计与计算保证几乎是一片空白——在本文之前，连"它到底能不能被识别"（identifiability，撇开样本量不谈）都没有答案。两个最基本的问题悬而未决：① 需要什么样的数据？② 需要多少样本？

**本文目标**：回答这两个问题，建立相关 probit 模型从偏好数据中学习的理论边界。

**核心 idea**：**数据阶数决定了能否学到相关性**——成对数据丢失相关信息（统计障碍），而三选一数据恰好补回这一信息（机会）。作者把"学相关协方差"这件事还原成"从三元排序概率反推高斯分布参数"的几何问题，再用归纳聚合把三元结论拓展到任意 $n$ 个备选。

## 方法详解

### 整体框架
全文围绕"成对不够、三选一够"展开，沿三条线推进：先用一个不可能性定理钉死成对数据的天花板，再在 $n=3$ 的最小情形把可识别性几何化地证出来，最后把三元结论聚合成任意 $n$ 的估计器并配上匹配的上下界。归一化约定贯穿始终：因为效用整体平移/缩放不改变选择分布，所以固定 $\langle\mu,\mathbf{1}\rangle=0$、$\Sigma\mathbf{1}=0$、$\mathrm{Tr}(\Sigma)=n$，使 $X$ 落在超平面 $\mathbf{1}^\top X=0$ 上。

```mermaid
flowchart TD
    A[相关 probit 模型 X~N(mu, Sigma)<br/>归一化: 1·mu=0, Sigma·1=0, Tr=n] --> B[Thm 3.2 成对数据不可能性<br/>无穷多组 mu,Sigma 给出同样的成对概率]
    A --> C[n=3 三元情形 Thm 4.1<br/>投影到 2D 平面 + 各向同性化]
    C --> D[Thm 4.4 聚合到任意 n<br/>用 O(n^3) 个 3x3 子矩阵拼回全局]
    D --> E[Thm 5.2 估计器<br/>O(n^2) 子矩阵, N=O(n^2/eps^2) 样本]
    E --> F[Thm 5.3 下界<br/>每对至少出现 Omega(eps^-2) 次, 证明近最优]
```

### 关键设计

**1. 成对数据的不可能性定理：钉死统计障碍。** 这是全文的出发点，也是最具冲击力的结论。作者证明（Theorem 3.2）：对任意 $n\geq 3$ 和满足归一化的 $(\mu,\Sigma)$，都存在一个**无穷集合** $S$，其中所有 $(\mu',\Sigma')$ 给出完全相同的成对概率 $\mathbb{P}\{X_i\geq X_j\}$。直觉上，取 $\mu=0$ 时对任意 $i,j$ 都有 $\mathbb{P}\{X_i\geq X_j\}=1/2$，无论 $\Sigma$ 是什么——成对比较的对称性把所有相关信息都抹平了。这一步从根上解释了"为什么相关 probit 模型一直缺统计/计算保证"：不是没人会算，而是数据本身就不含答案。

**2. 三元情形的几何识别：把概率反推成半空间。** 在 $n=3$ 这个最小可学情形（Theorem 4.1），作者把落在平面 $\mathbf{1}^\top X=0$ 上的退化高斯投影到二维 $V\sim\mathcal{N}(\dot\mu,\dot\Sigma)$，于是六种全排序事件 $\{X_i\geq X_j\geq X_k\}$ 对应平面上六个扇区的概率质量。再做白化变换 $\dot\Sigma^{-1/2}$ 把分布各向同性化，三个事件边界变成单位向量 $\tilde c_i$ 张成的半空间。这样一来，单变量的"翻越概率" $\mathbb{P}\{c_i^\top V\geq 0\}=\Phi(\langle\tilde c_i,\tilde\mu\rangle)$ 由于 $\Phi$ 严格单调，直接给出 $\alpha_i=\langle\tilde c_i,\tilde\mu\rangle$（Lemma 4.2）；而两两夹角 $\alpha_{ij}=\langle\tilde c_i,\tilde c_j\rangle$ 则通过双变量事件概率 $\gamma_{12}=\mathbb{P}\{\langle X,v_1\rangle,\langle X,v_2\rangle\leq 0\}$ 单调反解出来（Lemma 4.3）。拿到所有 $\alpha_i,\alpha_{ij}$ 后，利用 $c_3=c_1+c_2$ 解一个可逆线性系统就能复原 $\dot\Sigma^{1/2}$ 与 $\dot\mu$，进而还原 $\mu,\Sigma$。**这正是 best-of-three 必要又充分的来源**：三元排序概率恰好提供了夹角信息，而成对只提供边界。

**3. 从三元聚合到任意 n：用局部子矩阵拼出全局。** 单看三个备选，Theorem 4.1 只能把 $(\bar\mu_{ijk},\bar\Sigma_{ijk})$ 识别到一个未知缩放 $t_{ijk}>0$（因为每个三元组各自的归一化未必和全局一致）。聚合的关键是用**共享的成对二次型**把不同三元组的尺度对齐：由 $c_{ij}^\top\bar\Sigma_{ijk}c_{ij}=c_{ij}^\top\Sigma c_{ij}=\sigma_{ii}+\sigma_{jj}-2\sigma_{ij}$ 这个不随 $k$ 变的量，先令 $t_{123}=1$，再链式传播确定所有 $t_{ijk}$。随后用 $\sum_j c_{ij}^\top\Sigma c_{ij}=n\sigma_{ii}+\mathrm{Tr}(\Sigma)$ 并对 $i$ 再求和恢复 $\mathrm{Tr}(\Sigma)$，从而解出每个 $\sigma_{ij}$；均值则由 $c_{ij}^\top(t_{ijk}\tilde\mu_{ijk})=\mu_i-\mu_j$ 配合 $\sum_j(\mu_1-\mu_j)=n\mu_1$ 复原（Theorem 4.4）。这把"局部可识别"升级成"全局可识别"。

**4. 有限样本估计器与匹配下界：证明近最优。** 朴素聚合要估全部 $O(n^3)$ 个 $3\times 3$ 子矩阵，浪费。作者改用 $\tilde O(n^2)$ 个子矩阵，在可观测性假设 $\mathbb{P}\{i\succ j\succ k\}\geq\gamma$ 下给出多项式时间估计器（Theorem 5.2）：只需 $N\geq C n^2\varepsilon^{-2}\gamma^{-24}\log(n/\delta)\log^6(n/(\gamma\varepsilon))$ 个三元排序观测，就能保证 $\|\mu-\mu^*\|_\infty\leq\varepsilon$ 且 $\|\Sigma-\Sigma^*\|_\infty\leq\varepsilon$。配套的下界（Theorem 5.3）说明：任何成功估计器都必须让**每一对** $(i,j)$ 至少出现 $\Omega(\varepsilon^{-2})$ 次，于是独立实验数 $\Omega(n^2)$、总样本 $\Omega(n^2\varepsilon^{-2}\log(1/\delta))$，在 $n$、$\varepsilon$、$\delta$ 三个量级上都与上界吻合，确立了近最优性。

## 实验关键数据

### 主实验表格（真实数据，预测准确率，中位数 quantile）
六个备选里给 4 个的排序作上下文，预测剩下两个的偏好。对比 logit / 矩阵补全 / direct / probit(成对) / probit(best-of-three)。

| 数据集 (变体, 特征) | logit | 矩阵补全 | probit(成对) | probit(best-of-three) |
|---|---|---|---|---|
| sushi (B, default) | 0.65 | 0.66 | 0.64 | **0.65** |
| sushi (A, onehot) | 0.66 | 0.67 | 0.65 | **0.68** |
| sushi (B, onehot) | 0.67 | 0.68 | 0.66 | **0.67** |
| MovieLens 1k (onehot) | 0.62 | 0.60 | 0.60 | **0.61** |
| Netflix 100k (onehot) | 0.62 | 0.61 | 0.59 | **0.62** |
| Netflix 150k (onehot) | 0.61 | 0.60 | 0.56 | **0.61** |
| jokes (onehot) | 0.61 | 0.61 | 0.59 | **0.61** |

关键趋势：best-of-three probit **一致地优于**成对 probit（差距在多个数据集达 3–5 个百分点），并普遍追平或略超 logit；成对 probit 往往是表中最差，印证了理论上的"成对信息不足"。

### 消融实验表格（合成数据，对照 ground truth）
均值 $\mu\in\{0,r\}$，协方差 $\Sigma\in\{I,\,\text{bin}(\pm1),\,rI,\,r\}$。direct 为带模拟的理想上界。

| $\mu$ | $\Sigma$ | logit | probit(成对) | probit(best-of-three) |
|---|---|---|---|---|
| 0 | $I$ | 0.50 | 0.50 | 0.50 |
| 0 | bin | 0.50 | 0.45–0.54 | **0.79** |
| 0 | $r$ | 0.50 | 0.47–0.55 | **0.67** |
| $r$ | bin | 0.79 | 0.95 | **0.95** |
| $r$ | $r$ | 0.63 | 0.67–0.69 | **0.71** |

关键发现：在**纯相关**（$\mu=0$、$\Sigma=$bin）场景，logit 和成对 probit 都困在 0.50（瞎猜），唯有 best-of-three probit 跳到 0.79，**直接可视化地复现了"成对学不到相关性"**。Figure 2 进一步显示成对 probit 在无相关时会幻觉出虚假相关，而 best-of-three 恢复的协方差矩阵与 ground truth 几乎一致。

### 关键发现
- **可解释的相关性**：在 Netflix/MovieLens 上学到的协方差有清晰语义——续集（Spider-Man 1 & 2、Kill Bill Vol.1 & 2，相关 0.56–0.65）强正相关；文艺片（Memento、Eternal Sunshine）与好莱坞爆米花片（Pearl Harbor、Cheaper by the Dozen）强负相关（约 −0.40）。寿司数据里海胆极具分裂性，黄瓜卷与 toro（金枪鱼腹）偏好负相关。
- **福利最大化实验**：用学到的模型求固定大小子集的期望最大效用 $\max_{R:|R|=N}\mathbb{E}[\max_{i\in R}X_i]$，best-of-three 模型在选取多样化推荐集合上更合理。
- best-of-three 对 logit 的提升在均值效应强的数据集上较温和——相关性是"锦上添花"，但在相关主导的场景下是决定性的。

## 亮点与洞察
- **把"数据采集范式"上升为一等公民**：以往 RLHF 讨论都在卷模型/算法，本文指出**数据的阶数本身就是信息瓶颈**——再强的算法也救不回成对数据里缺失的相关信息。这对"怎么标注偏好数据"有直接的工程指导：想要个性化，就该收集三选一而非两两比较。
- **首个可识别性 + 近最优估计器的完整闭环**：从不可能性（下界）到可识别性（存在性）到多项式时间估计器（构造性）到样本下界（最优性），四块拼图齐全，理论上干净利落。
- **几何化的证明很优雅**：把高维退化高斯投影到 2D、各向同性化后用半空间概率反解夹角，把抽象的协方差识别变成可画图的扇区问题（Figure 1）。
- **理论直接落到真实推荐数据**并产出可解释的相关结构，避免了纯理论论文"只证不验"的常见短板。

## 局限与展望
- **可观测性假设较强**：Theorem 5.2 依赖 $\mathbb{P}\{i\succ j\succ k\}\geq\gamma$，且样本复杂度里 $\gamma^{-24}$ 这个指数极其悲观——当某些三元序极罕见时实际样本需求会爆炸，论文未讨论该指数能否收紧。
- **probit 高斯假设**：相关性被锁定在高斯协方差结构里，对重尾、多峰或非高斯的真实偏好可能失配；扩展到更一般的相关 RUM 仍是开放问题。
- **三选一数据的获取成本**：理论上 best-of-three 必要，但现实标注里让用户对三项做完整排序比两两比较更累、更易噪声，论文未量化这一标注代价与收益的权衡。
- **best-of-three 对 logit 的真实增益偏温和**：除合成的强相关场景外，真实数据上提升多在 1–3 个百分点，相关性是否值得改造整条 RLHF 数据管线，需要更大规模、更贴近 LLM 对齐的验证。
- 实验规模仍停留在传统推荐数据集（寿司/电影/笑话），尚未在真正的 LLM RLHF（prompt-response 对）上端到端验证，而这恰是 motivation 里反复强调的应用场景。

## 相关工作与启发
- **RUM 与 IIA 谱系**：从 Luce-Shephard-McFadden（logit）、Bradley-Terry，到 Debreu 的红/蓝巴士悖论、Benson et al. 的实证 IIA 违背，本文站在这条"批判 IIA"的长链末端，但首次从**可学习性**角度给出量化刻画。
- **RLHF 偏好建模**：呼应 Christiano et al. 的偏好对齐管线，以及 Bai/Casper/Sorensen/Conitzer 等对"单一全局奖励"局限的批评——本文为"多元/个性化偏好不可学"提供了理论根据，并给出补救方案。
- **方法学启发**：① "higher-order data unlocks higher-order structure"是个可迁移的范式，类似思路或许适用于学习其它需要高阶矩的潜变量模型；② 对工程界，最直接的 takeaway 是**重新设计偏好标注协议**，在关键场景引入三选一/小集合排序；③ 把可识别性分析（而非只看 loss 下降）作为评估新偏好模型的前置检验，值得 RLHF 社区借鉴。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次回答相关 probit 模型"能否学/需多少数据"，"成对不可能 + 三选一近最优"是干净且有冲击力的新结论。
- 实验充分度: ⭐⭐⭐ 合成数据漂亮地佐证理论，真实数据有可解释性，但规模偏小、未触及真正的 LLM RLHF 端到端验证。
- 写作质量: ⭐⭐⭐⭐ 理论叙述层层递进、几何直觉清晰（Figure 1/2 很有帮助），证明组织得当；公式密度高但脉络明确。
- 价值: ⭐⭐⭐⭐ 对"如何采集偏好数据"提供了原则性指导，可能影响 RLHF 数据标注实践；理论闭环完整，是偏好学习理论的扎实一步。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Barriers for Learning in an Evolving World: Mathematical Understanding of Loss of Plasticity](barriers_for_learning_in_an_evolving_world_mathematical_understanding_of_loss_of.md)
- [\[ICLR 2026\] Statistical and Structural Identifiability in Representation Learning](statistical_and_structural_identifiability_in_representation_learning.md)
- [\[ICLR 2026\] Near-Optimal Sample Complexity Bounds for Constrained Average-Reward MDPs](near-optimal_sample_complexity_bounds_for_constrained_average-reward_mdps.md)
- [\[ICLR 2026\] Automata Learning and Identification of the Support of Language Models](automata_learning_and_identification_of_the_support_of_language_models.md)
- [\[ICLR 2026\] Interactive Learning of Single-Index Models via Stochastic Gradient Descent](interactive_learning_of_single-index_models_via_stochastic_gradient_descent.md)

</div>

<!-- RELATED:END -->
