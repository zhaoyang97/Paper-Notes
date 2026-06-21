---
title: >-
  [论文解读] Full-Batch Gradient Descent Outperforms One-Pass SGD: Sample Complexity Separation in Single-Index Learning
description: >-
  [ICML2026][优化/理论][单指标模型] 本文在二次激活的高斯单指标模型里严格证明：朴素二次激活下"重复使用全部数据"的全批量梯度下降并不比一遍式 SGD 更省样本（都要 $n\gtrsim d\log d$），但只要把激活**截断**一下，全批量 GD 就能在 $n\gtrsim d$（线性样本量）下实现弱恢复甚至强恢复，从而与仍需 $d\log d$ 的一遍式 SGD 拉开一个 $\log d$ 的样本复杂度差距。
tags:
  - "ICML2026"
  - "优化/理论"
  - "单指标模型"
  - "样本复杂度"
  - "全批量梯度下降"
  - "相位恢复"
  - "BBP 相变"
---

# Full-Batch Gradient Descent Outperforms One-Pass SGD: Sample Complexity Separation in Single-Index Learning

**会议**: ICML2026  
**arXiv**: [2602.02431](https://arxiv.org/abs/2602.02431)  
**代码**: 待确认  
**领域**: 学习理论 / 优化  
**关键词**: 单指标模型, 样本复杂度, 全批量梯度下降, 相位恢复, BBP 相变

## 一句话总结
本文在二次激活的高斯单指标模型里严格证明：朴素二次激活下"重复使用全部数据"的全批量梯度下降并不比一遍式 SGD 更省样本（都要 $n\gtrsim d\log d$），但只要把激活**截断**一下，全批量 GD 就能在 $n\gtrsim d$（线性样本量）下实现弱恢复甚至强恢复，从而与仍需 $d\log d$ 的一遍式 SGD 拉开一个 $\log d$ 的样本复杂度差距。

## 研究背景与动机

**领域现状**：机器学习里有个民间共识——把训练数据多用几遍（multi-pass / full-batch）能提升梯度法的统计效率。这一点在线性回归里被研究透了，但在**非线性、非凸**的特征学习里，多遍 GD 到底比一遍式在线 SGD 强在哪、强多少，一直没说清楚。

**现有痛点**：单指标模型 $y=\sigma(\langle x,\theta^\star\rangle)$ 是分析浅层网络非凸特征学习的标准沙盒。对于信息指数为 2 的激活（包括所有偶函数链接，如二次激活、相位恢复），已知一遍式 SGD 需要 $n\gtrsim d\log d$ 个样本才能弱恢复 $\theta^\star$，而信息论极限只需 $n\gtrsim d$，中间差了一个 $\log d$。谱方法 + 近似消息传递（AMP）能在 $n\gtrsim d$ 下成功，但它们不是梯度法。

**核心矛盾**：人们想知道**标准的全批量梯度下降**（不改算法、不改损失）能不能也吃掉这个 $\log d$。难点在于：① 现有对 full-batch GD 的分析大多还需要 $n\gtrsim d\,\mathrm{polylog}\,d$，并不能证明它优于一遍式 SGD；② 此前用"重复两遍数据等价于改损失/做标签预处理"来解释多遍优势的机制（Dandi et al. 2024b 等），对偶激活根本不适用——偶链接的生成指数仍 $\ge 2$，预处理拿不掉 $\log d$。

**本文目标**：在 $n,d$ 成比例的最优区间里，严格刻画 full-batch GD 何时能以**线性样本量**实现弱恢复与强恢复，以及对应需要多少步迭代。

**切入角度**：作者盯住"球面上最小化相关损失"这一最常被研究、且 SGD 下 $\log d$ 障碍被认为不可避免的设定，先证一个负结果说明朴素二次激活无济于事，再用一个极简改动（截断激活）把局面翻转。

**核心 idea**：朴素二次激活无界，导致经验矩阵谱被噪声主导、顶特征向量与信号失相关；只要**截断激活让测量有界**，沿优化轨迹的经验 Hessian 就出现一致的 BBP 相变（谱隙 + 信号对齐），从而在 $n\gtrsim d$ 就能恢复——而一遍式 SGD 对同一截断链接仍受 $d\log d$ 下界约束，分离由此产生。

## 方法详解

### 整体框架
这是一篇纯理论论文，"方法"就是三组定理把"数据复用是否有用"这个问题层层切开。设定固定：$n$ 个 i.i.d. 样本 $x_i\sim\mathcal N(0,I_d)$，$y_i=\sigma(\langle x_i,\theta^\star\rangle)$，$\|\theta^\star\|=1$，激活取二次 $\sigma(z)=z^2$ 或其截断版。评价用两档：**弱恢复** $\lim_{n\to\infty}\frac{|\langle\hat\theta,\theta^\star\rangle|}{\|\hat\theta\|\,\|\theta^\star\|}=\epsilon>0$（拿到非平凡相关），**强恢复** $\lim_{n\to\infty}\min_{s\in\{\pm1\}}\|\hat\theta-s\theta^\star\|=0$（精确恢复，差一个符号歧义）。对照基线是一遍式 SGD，每个点只用一次，其在信息指数 2 下有 $n\gtrsim d\log d$ 的下界（Ben Arous et al. 2021）。

这里有两个贯穿全文的关键量需要先讲清。一是**信息指数**（information exponent）：它刻画激活在随机初始化附近"逃离平庸方向"的难度，信息指数为 2 的激活（包括所有偶链接，如 $z^2$、相位恢复）下，总体梯度流要花 $T\gtrsim\log d$ 时间才能逃离，折算到在线 SGD 就是 $n\gtrsim d\log d$ 的样本量。二是**生成指数**（generative exponent，Damian et al. 2024）：它刻画任意标签预处理之后还剩多难，本文的二次/截断二次激活生成指数仍是 2，这正是"重复两遍数据等价改损失"那套机制对本设定失效的根本原因——预处理拿不掉 $\log d$。

作者沿三步推进：先在球面 + 相关损失下给出**负结果**（朴素二次激活），再在同一设定下用**截断**给出**正结果**（弱恢复 @ $n\gtrsim d$），最后切到欧式 + 平方损失 + 小初始化拿到**强恢复 + 迭代复杂度**。注意"损失/几何"的选择不是装饰：相关损失天然定义在球面上、范数恒定，适合做谱/景观分析但给不出范数信息；平方损失 + 小初始化则能同时学方向和范数，是拿到强恢复的必要切换。三组结论的对照见下表。

| 定理 | 损失 / 几何 | 激活 | 结论 | 样本量 |
|------|------------|------|------|--------|
| Thm 3.1（负） | 相关损失 / 球面梯度流 | $z^2$ | $n=o(d\log d)$ 时 overlap $\to 0$，无任何优势 | 需 $\gtrsim d\log d$ |
| Thm 3.2（正） | 相关损失 / 球面梯度流 | 截断 $z^2$ | 弱恢复，overlap $\to 1$（随 $M,n/d$ 增大） | $n\gtrsim d$ |
| Thm 4（正） | 平方损失 / 欧式 GD + 小初始化 | 截断 $z^2$ | 强（精确）恢复 | $n\gtrsim d$，$T\gtrsim\log d$ 步 |

### 关键设计

**1. 负结果：朴素二次激活下，全批量 GD 退化成固定矩阵上的幂迭代，仍要 $d\log d$**

针对的痛点是"凭直觉以为复用全部数据一定有用"。对 $\sigma(z)=z^2$，相关损失 $\hat L(\theta)=-\frac1n\sum_i y_i\sigma(\langle x_i,\theta\rangle)$ 的球面梯度流恰好是对一个**固定矩阵** $A_\star=\frac2n\sum_i y_i x_i x_i^\top$ 的幂迭代 $\frac{d\theta}{dt}=(I-\theta\theta^\top)A_\star\theta$。于是 $\theta(t)$ 收敛到 $A_\star$ 的主特征向量 $v_1(A_\star)$，问题就变成"$v_1(A_\star)$ 和 $\theta^\star$ 有多相关"。作者用随机矩阵分析证明（Thm 3.1）：当 $n=o(d\log d)$ 时，因为二次激活**无界**，把 $\sum_i y_i x_i x_i^\top$ 写成分块形式 $\begin{psmallmatrix}a&q^\top\\ q&P\end{psmallmatrix}$ 后，$P$ 的最大特征值至少 $d\log n$ 量级、压过了携带信号的秩一更新 $\mu qq^\top$，导致 $|\langle v_1(A_\star),\theta^\star\rangle|\to 0$，几乎完全失相关。结论很反直觉：复用全部数据的 full-batch GD 在这里**对一遍式 SGD 毫无统计优势**，$\log d$ 因子照样存在。

**2. 截断激活触发一致 BBP 相变，把弱恢复阈值从 $d\log d$ 压到 $d$，构成与 SGD 的样本复杂度分离**

负结果的病根是"无界激活让谱被重尾噪声主导"。作者的修法极简：把二次激活**截断**——$\sigma(z)=\min\{z^2,M\}$（或式 (3.6) 的光滑版 $\sigma(z)=\int_0^{z^2}\varphi(u)\,du$，$\varphi$ 是 $[0,1]$ 的光滑截断核）。截断后球面梯度流变成对**时变矩阵** $A(\theta(t))$ 的幂迭代。关键观察是把它拆成 $A(\theta)=A_\star-B(\theta)$，其中 $B(\theta)\succeq0$ 只支撑在罕见事件 $\{\langle x_i,\theta\rangle^2>M\}$ 上、且可被 VC 维论证一致地控住。由于测量 $y_i$ 现在有界，子高斯协方差矩阵的集中给出**一致的 BBP 相变**：以高概率，对所有 $\theta$，$A(\theta)$ 的前两个特征值分别接近 $6$ 和 $2$、有常数级谱隙，顶特征向量与 $\theta^\star$ 的重叠随 $n/d$ 增大趋于 1，即 $|\langle v_1(A_\star),\theta^\star\rangle|\le C_M\sqrt{d/n}$。再用经验损失作 Lyapunov 函数 + 中心稳定流形定理排除收敛到非主特征向量，得到 Thm 3.2：当 $n\ge CM^4 d$ 时，以概率 $\ge1-\exp(-cd^{1/5})$ 有
$$\lim_{t\to\infty}|\langle\theta(t),\theta^\star\rangle|\ge 1-C\big(e^{-M/2}+(d/n)^{1/5}\big).$$
要害在于：**同样的截断链接信息指数仍是 2**，所以一遍式 SGD 的 $d\log d$ 下界依旧成立。于是同一问题、同一激活下，full-batch GD 用 $n\gtrsim d$、SGD 要 $n\gtrsim d\log d$——这就是干净的样本复杂度**分离**，且分离来自"数据复用"本身而非过参数化。

**3. 小初始化 + 平方损失，把弱恢复升级为强（精确）恢复，并给出 $T\gtrsim\log d$ 的迭代复杂度**

球面 + 相关损失的景观分析有两个盲区：① 因 $A(\theta(t))$ 时变、谱控制误差比初始重叠大 $\sqrt d$ 量级，**说不清收敛要多少步**；② 球面上范数恒定，**给不出强恢复**。为补上，作者改用欧式 GD 最小化平方损失 $\hat L(\theta)=\frac1{2n}\sum_i(\sigma(\langle x_i,\theta\rangle)-\sigma(\langle x_i,\theta^\star\rangle))^2$，并从**小初始化**出发。直觉是 Stöger–Soltanolkotabi 的"早期阶段 $\approx$ 在 $A_\star$ 上做幂迭代"——截断让 $A_\star$ 在有限 $n/d$ 就有信息性 BBP 相变；同时小初始化迫使算法还要学对范数（$\|\theta^\star\|=1$），这正是相关损失在球面上捕捉不到的。Thm 4 证明：截断二次激活下，$n\gtrsim d$、$T\gtrsim\log d$ 步即可强恢复——先经历至多 $O(\log d/\eta)$ 步的"搜索阶段"，之后 $\min_{s}\|\theta_t-s\theta^\star\|^2$ 几何收敛到 0。作者强调这是据其所知**第一个**在信息论最优的成比例 $n,d$ 区间里、对未经算法或损失改造的 full-batch GD 给出的强恢复 + 收敛速率保证。

## 实验关键数据

这是理论论文，"实验"是对定理阈值的数值验证（Figure 1）：在球面上以学习率 $\eta=0.1$、跑 $T=1000\log^2 d$ 步最小化经验相关损失，看重叠随 $\delta=n/d$ 的变化。

### 主结果：阈值随维度的行为

| 设置 | 现象 | 印证的定理 |
|------|------|-----------|
| 二次激活 $\sigma(z)=z^2$（图 1a） | $d$ 越大，达到固定重叠所需的 $\delta=n/d$ 阈值越大，阈值不收敛 | Thm 3.1（需 $d\log d$） |
| 截断激活 $\sigma=\min\{z^2,M\}$，$M=8$（图 1b） | 不同 $d$ 的学习曲线几乎重合，固定 $\delta$ 处重叠与 $d$ 无关，阈值 $\delta=\Theta(1)$ | Thm 3.2（$n\gtrsim d$ 即可） |
| 二次激活的阈值拟合（图 1c） | 达到目标平方重叠 $\{0.1,\dots,0.5\}$ 所需 $\delta$ 与 $\log d$ 高度线性 | 定量验证 $\delta\simeq\log d$ |

### 关键发现
- **截断是分水岭**：仅仅把激活截断（一个看似无关紧要的改动），就让弱恢复阈值从随 $d$ 增长变成常数级——这与理论里"无界→谱被噪声主导 / 有界→一致 BBP 相变"的机制完全吻合。
- **$\log d$ 因子可被定量"看见"**：图 1c 把朴素二次激活的样本复杂度阈值漂亮地拟合成 $\log d$，说明 $\log d$ 障碍不是分析松弛的产物，而是真实存在。
- **重叠随 $M$ 与 $n/d$ 单调改善**：截断阈值 $M$ 越大、$n/d$ 越大，Thm 3.2 的重叠下界 $1-C(e^{-M/2}+(d/n)^{1/5})$ 越接近 1，实验里不同 $d$ 曲线的塌缩也印证了这一点。
- **优势源自数据复用而非过参数化**：与 Sarao Mannelli et al. 的过参数化景观分析不同，本文学生模型不过参数化，分离纯粹来自"每步复用全部数据"。

## 亮点与洞察
- **"负结果 + 正结果"成对呈现**：先证朴素 full-batch GD 没用，再证截断后翻盘——这种对照让"截断"这一改动的因果作用无可辩驳，比单给一个正结果更有说服力。
- **一致 BBP 相变是技术核心**：把时变矩阵 $A(\theta)$ 拆成固定 $A_\star$ 减去支撑在罕见事件上的扰动 $B(\theta)$，再用 VC 维一致地控住扰动，是绕开"矩阵随轨迹变化"这一难点的关键技巧，可迁移到其他沿轨迹分析谱的非凸问题。
- **稳定流形定理排除坏不动点**：用经验损失作 Lyapunov 函数保证收敛到某个特征向量，再借中心稳定流形定理证明坏不动点的吸引域是零测集，这套"景观良性 + 随机初始化几乎必收敛到主特征向量"的组合拳值得借鉴。
- **截断作为"廉价正则"的统计意义**：截断不改信息指数（所以 SGD 占不到便宜），却让 full-batch 的经验谱变良性——这提示"有界化激活/损失"在非凸特征学习里可能是被低估的免费午餐。
- **回应大模型多 epoch 训练的理论缺口**：动机直接对标 LLM 在数据受限时单 epoch vs 多 epoch 的经验现象，给"数据复用何时真能省样本"提供了干净的非线性可证样例。

## 局限与展望
- **模型高度受限**：只覆盖高斯单指标、二次/截断二次激活（信息指数 2、偶链接，紧贴相位恢复），是否推广到一般信息指数 $k$、多指标模型、真实分布仍未知。
- **截断常数 $M$ 与样本量耦合**：Thm 3.2 要求 $n\ge CM^4 d$（可细化到 $M^{2.01}$），$M$ 太小则截断误差项 $e^{-M/2}$ 主导、重叠上界打折，存在 $M$ 与 $n/d$ 的权衡。
- **梯度流 vs 实际 GD**：球面部分用连续时间梯度流刻画，离散步长、有限步效应主要靠平方损失那节补；学习率 $\eta$ 取很小才有干净保证。
- **结论是渐近的**：弱/强恢复都在 $n,d\to\infty$ 成比例极限下陈述，有限维波动（图中小 $\delta$ 处的偏差）未被覆盖。
- **正则性靠光滑截断换来**：球面部分用光滑截断 (3.6) 是为保证梯度流解的良定与正则，硬截断 (4.4) 只在平方损失那节用作技术便利，两种截断的等价性是默认而非证明。
- 一个自然方向：是否存在"自适应截断阈值 $M(\theta)$"或其他有界化手段，在更弱条件下保住分离；另一方向是把分析推广到一般信息指数 $k>2$ 与多指标模型，看 full-batch 优势是否依旧成立。

## 相关工作与启发
- **vs 一遍式在线 SGD（Ben Arous et al. 2021）**：SGD 每点用一次、受 $d\log d$ 下界；本文 full-batch 复用全部数据 + 截断激活打到 $d$，构成分离。核心区别是"数据复用 + 有界测量"让经验谱良性。
- **vs 两遍数据 = 改损失（Dandi/Lee/Arnaboldi 2024）**：他们解释多遍优势靠"重复两遍 ≈ 标签预处理降低信息指数"，但对偶链接生成指数仍 $\ge2$，拿不掉 $\log d$；本文机制完全不同，靠的是全批量轨迹上的一致 BBP 相变，并能处理需多遍才分离的"硬"目标。
- **vs 谱方法 + AMP（Mondelli–Venkataramanan 2021 等）**：这些非梯度算法早已知道在 $n\gtrsim d$ 成功；本文的价值是证明**标准梯度下降**（不改算法/损失）也能达到同等样本量，并给出迭代复杂度。
- **vs 过参数化景观分析（Sarao Mannelli et al. 2020b）**：他们靠学生过参数化得到良性景观，分不清优势来自复用还是过参数化；本文不过参数化，把功劳明确归给数据复用，并补上 SGD 的下界对照。
- **vs 并行工作（Montanari & Wang 2026）**：他们用动态平均场理论刻画 full-batch GD 恢复多指标目标首个非平凡子空间的尖锐阈值；本文聚焦单指标 + 截断激活，额外给出强恢复与显式迭代复杂度。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次为未改造的 full-batch GD 在最优 $n,d$ 区间证明强恢复 + 与 SGD 的样本复杂度分离。
- 实验充分度: ⭐⭐⭐⭐ 纯理论，数值实验精准卡住理论阈值，但无真实任务（理论论文本该如此）。
- 写作质量: ⭐⭐⭐⭐⭐ 负/正结果对照清晰，证明思路（幂迭代 + 一致 BBP + 稳定流形）层层递进。
- 价值: ⭐⭐⭐⭐ 给"数据复用何时省样本"提供干净可证样例，并把 LLM 多 epoch 直觉锚到非线性特征学习理论。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Implicit Bias of Per-sample Adam on Separable Data: Departure from the Full-batch Regime](../../ICLR2026/optimization/implicit_bias_of_per-sample_adam_on_separable_data_departure_from_the_full-batch.md)
- [\[NeurIPS 2025\] Learning Single-Index Models via Harmonic Decomposition](../../NeurIPS2025/optimization/learning_single-index_models_via_harmonic_decomposition.md)
- [\[ICML 2025\] Nearly Optimal Sample Complexity for Learning with Label Proportions](../../ICML2025/optimization/nearly_optimal_sample_complexity_for_learning_with_label_proportions.md)
- [\[ICML 2026\] Convex Basins in Single-Index Model Loss Landscapes: Applications to Robust Recovery under Strong Adversarial Corruption](convex_basins_in_single-index_model_loss_landscapes_applications_to_robust_recov.md)
- [\[AAAI 2026\] Beyond the Mean: Fisher-Orthogonal Projection for Natural Gradient Descent in Large Batch Training](../../AAAI2026/optimization/beyond_the_mean_fisher-orthogonal_projection_for_natural_gradient_descent_in_lar.md)

</div>

<!-- RELATED:END -->
