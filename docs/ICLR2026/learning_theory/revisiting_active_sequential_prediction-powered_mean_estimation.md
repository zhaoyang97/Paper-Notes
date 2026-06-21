---
title: >-
  [论文解读] Revisiting Active Sequential Prediction-Powered Mean Estimation
description: >-
  [ICLR 2026][学习理论][主动序贯均值估计] 本文重新审视"主动序贯预测增强均值估计"：先给出此前只有渐近保证的估计量的**非渐近、任意时刻成立**的数据相关置信界，再用 FTRL 在线学习去选每轮的标签查询概率，理论与实验共同表明——当查询概率对**当前**协变量不可见时，最优策略就是简单地令查询概率收敛到预算上界 $T_b/T$，精心设计的不确定性加权几乎不带来额外收益。
tags:
  - "ICLR 2026"
  - "学习理论"
  - "主动统计推断"
  - "预测增强推断（PPI）"
  - "主动序贯均值估计"
  - "非渐近分析"
  - "Freedman 不等式"
  - "FTRL"
  - "查询概率"
---

# Revisiting Active Sequential Prediction-Powered Mean Estimation

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=Iw0tMeLed8](https://openreview.net/forum?id=Iw0tMeLed8)  
**代码**: 随补充材料提供（Jupyter notebook，复现 Figure 1 等）  
**领域**: 学习理论 / 主动统计推断 / 预测增强推断（PPI）  
**关键词**: 主动序贯均值估计, 非渐近分析, Freedman 不等式, FTRL, 查询概率

## 一句话总结
本文重新审视"主动序贯预测增强均值估计"：先给出此前只有渐近保证的估计量的**非渐近、任意时刻成立**的数据相关置信界，再用 FTRL 在线学习去选每轮的标签查询概率，理论与实验共同表明——当查询概率对**当前**协变量不可见时，最优策略就是简单地令查询概率收敛到预算上界 $T_b/T$，精心设计的不确定性加权几乎不带来额外收益。

## 研究背景与动机

**领域现状**：均值估计是经典推断任务，近年在"主动统计推断 / 预测增强推断（PPI）"框架下复兴。设有大量无标签样本 $x_1,\dots,x_T$，目标是估计标签均值 $\mu_y=\mathbb{E}[y_t]$，但真值标签 $y_t$ 获取代价高、预算有限（要求 $\mathbb{E}[T_{\text{lab}}]\le T_b$，且 $T_b\ll T$）。同时手上有一个不断更新的黑盒预测模型 $f_t(\cdot)$。Zrnic & Candes (2024)（下称 ZC24）提出在每轮以概率 $\pi_t(x_t)$ 决定是否查询真值：查到就用真标签、查不到就用模型预测，得到无偏估计量

$$\hat w=\frac{1}{T}\sum_{t=1}^{T}\Big(f_t(x_t)+\big(y_t-f_t(x_t)\big)\frac{\xi_t}{\pi_t(x_t)}\Big),\quad \xi_t\sim\text{Bernoulli}(\pi_t(x_t)).$$

**现有痛点**：ZC24 的最优策略正比于预测不确定性 $\pi^{\text{opt}}_t\propto\sqrt{\mathbb{E}[(y_t-f_t(x_t))^2\mid\mathcal{F}_{t-1}]}$，但真分布未知，实践里只能用一个不确定性预测器 $u_t(x_t)$ 近似 $|y_t-f_t(x_t)|$，再裁剪到预算内；又因近似可能不准、预算可能被浪费，最后还要和均匀策略 $\pi^{\text{unif}}=T_b/T$ 做线性混合 $\pi^{(\lambda)}_t=(1-\lambda)\pi_t+\lambda\pi^{\text{unif}}_t$，ZC24 直接取 $\lambda=0.5$。更关键的是，**该估计量此前只有渐近正态性，没有非渐近的任意时刻置信界**。

**核心矛盾**：作者跑 ZC24 的公开实现、扫不同混合系数 $\lambda$，发现一个反直觉现象——$\lambda=1$（纯均匀、**完全忽略**不确定性）得到的置信区间宽度反而和 $\lambda=0.5$ 相当、甚至略窄（Figure 1）。这说明"用当前协变量的不确定性来调查询概率"这个看似聪明的设计，在实践中是**脆弱、可有可无**的。

**本文目标**：(1) 补上估计量的非渐近分析，给出随观测累积的数据相关界；(2) 理论上解释为什么不确定性分量"没用"，并给出一个有 no-regret 保证的查询概率在线学习方案。

**核心 idea**：把序贯估计写成**在线更新**、用 Freedman 不等式刻画其收敛；再把"选查询概率"建模成一个**不依赖当前协变量**的在线凸优化问题，用 FTRL 求解——而 FTRL 的 sublinear regret 恰恰迫使查询概率收敛到预算上界 $\tau=T_b/T$，从理论上印证了"均匀即最优"。

## 方法详解

### 整体框架
本文不是提出新的网络或模块，而是给主动序贯均值估计这个**统计/在线学习问题**做两件事：先把 ZC24 的估计量改写成逐轮的在线更新步并证一个非渐近界，再设计一个在线规则去**主动控制这个界里的方差项**。两条线索串成一个闭环：每观察一个样本 $x_t$ → 用 FTRL 给出查询概率 $p_t$（只看历史、不看当前 $x_t$）→ 以 $\text{Bernoulli}(p_t)$ 决定查不查真值 → 更新无偏估计 $w_{t+1}$，并每攒满一个 batch 就更新预测模型 $f_t$ 与近似 oracle。最终输出 $(1-\alpha)$ 置信区间 $\mathrm{CI}_\alpha=\big[w_{T+1}\pm z_{1-\alpha/2}\,\hat\sigma/\sqrt{T}\big]$。

理论主线是：估计误差 $|w_{t+1}-\mu_y|$ 被一个含**累积条件方差** $S_t=\sum_{s\le t}\sigma_s^2$ 的量控制（Theorem 1）；而条件方差里唯一受查询概率影响的项是 $\tfrac{1}{p_t}\mathbb{E}[(y_t-f_t(x_t))^2\mid\mathcal{F}_{t-1}]$（Lemma 2）；于是"选 $p_t$ 让界更小"就被还原成一个在线优化问题（Section 5）。

### 关键设计

**1. 把序贯主动估计改写成在线更新，用 Freedman 不等式给出非渐近、任意时刻置信界**

针对"此前只有渐近保证"这个空白，作者先把估计量重写成 $w_{t+1}=w_t+\tfrac{1}{T}g_t$，其中 $g_t:=f_t(x_t)+(y_t-f_t(x_t))\tfrac{\xi_t}{\pi_t(x_t)}$ 满足 $\mathbb{E}[g_t]=\mu_y$，初值 $w_1=0$。由于 $g_t-\mu_y$ 是一个鞅差序列，作者套用 Freedman 鞅集中不等式（Lemma 1）——它给出的偏差界**自适应于累积条件方差**而非最坏方差，这正是数据相关界的来源。最终 Theorem 1 给出：对任意 $\delta\in(0,1/e)$，以概率至少 $1-\delta$，对所有 $t\in[T]$，

$$|w_{t+1}-\mu_y|\le \frac{2\max\big\{2\sqrt{S_t},\,(G+|\mu_y|)\sqrt{\log(\log(T)/\delta)}\big\}\sqrt{\log(\log(T)/\delta)}}{T}+\Big(1-\frac{t}{T}\Big)|\mu_y|,$$

其中 $S_t=\sum_{s=1}^{t}\sigma_s^2$，$\sigma_s^2=\mathbb{E}[(g_s-\mu_y)^2\mid\mathcal{F}_{s-1}]$，$|g_t|\le G$。这个界"任意时刻成立"（uniform over $t$），且分两段：早期 $t\ll T$ 时 $(1-t/T)|\mu_y|$ 的 burn-in 项主导，速率可能慢于 $O(1/\sqrt t)$；burn-in 之后第一项主导，用平凡界 $S_t\le 2t(G^2+\mu_y^2)$ 即得 $|w_{t+1}-\mu_y|=O(1/\sqrt t)$。它的价值在于把"误差大小"显式挂到 $S_t$ 上，从而提示：**只要能压低累积条件方差 $S_t$，就能比 $O(1/\sqrt t)$ 更快**——这就给下一步"主动选查询概率"留出了发力点。

**2. 条件方差分解 + 近似 oracle：把"该不该多查"还原成一个可在线优化的单一标量**

要压 $S_t$，得先看清查询概率藏在哪儿。Lemma 2 把单步条件方差展开，证明其中**唯一**与查询概率 $\pi_t(x_t)$ 有关的项是 $\mathbb{E}\big[(y_t-f_t(x_t))^2\tfrac{1}{\pi_t(x_t)}\mid\mathcal{F}_{t-1}\big]$；并且当查询策略只依赖历史（$\pi_t(x_t)=p_t$ 是 $\mathcal{F}_{t-1}$-可测）时，它进一步化简为 $\tfrac{1}{p_t}\mathbb{E}[(y_t-f_t(x_t))^2\mid\mathcal{F}_{t-1}]$。也就是说，$p_t$ 越大方差项越小，但 $p_t$ 受预算约束不能太大——这是一个干净的 trade-off。可惜 $\mathbb{E}[(y_t-f_t(x_t))^2\mid\mathcal{F}_{t-1}]$ 依赖未知分布，无法直接拿到。作者引入一个**近似 oracle** $\Phi_t(x_t)\in\mathbb{R}_+$，满足夹逼关系

$$\tfrac{1}{c_1}\Phi_t(x_t)\le \mathbb{E}\big[(y_t-f_t(x_t))^2\mid\mathcal{F}_{t-1}\big]\le c_0\,\Phi_t(x_t),\quad c_0,c_1>0.$$

实践里 $\Phi_t$ 就用"对平方残差 $(f_t(x_t)-y_t)^2$ 做线性回归"来实现，每个 batch 更新一次。这样"控制方差项"就被还原成"在每轮挑一个 $p_t$ 去最小化与 $\Phi_t$ 有关的累积损失"，为 FTRL 登场铺好了路。

**3. 用 FTRL 在线决定查询概率：闭式解，且 sublinear regret 逼着 $p_t$ 收敛到预算上界 $\tau=T_b/T$**

作者把每轮的损失定义为 $\tilde\ell_t(p)=\Phi_t(x_t)/p$（在 $(0,1]$ 上凸），用经典的 Follow-the-Regularized-Leader 在可行域 $[\beta,\tau]$ 上在线选 $p_t$：

$$p_t\leftarrow\arg\min_{p\in[\beta,\tau]}\;\gamma\theta_{t-1}p+\tfrac{1}{2}p^2,\qquad \theta_{t-1}:=-\sum_{s=1}^{t-1}\frac{\Phi_s(x_s)}{p_s^2}.$$

Lemma 3 给出闭式解 $p_t=\max\{\beta,\min\{\tau,-\gamma\theta_{t-1}\}\}$；其中上界取 $\tau=T_b/T$ 对应预算约束（保证 $p_t$ 不超过预算比例），下界 $\beta>0$ 强制每轮保留一点探索、不让查询概率塌到 0。FTRL 对凸损失有 sublinear regret（Lemma 4–5：取 $\gamma=\tfrac{1}{\sqrt T}\tfrac{\beta^2}{B}$ 时 regret 为 $O(\sqrt T)$，前提是 oracle 输出有界 $\Phi_t\le B$）。**关键洞察**在于：因为 $\Phi_s(\cdot)\ge 0$，单步损失 $\Phi_s/p$ 在 $[\beta,\tau]$ 上的最小值点恒为 $p=\tau$，所以要想跟上"事后最优固定概率"、维持 sublinear regret，学习器的 $p_t$ 必须**最终收敛到上界 $\tau=T_b/T$**。把 FTRL 策略代回 Theorem 1，得到 Theorem 2 的数据相关界

$$|w_{t+1}-\mu_y|\le \frac{2\max\{2\sqrt{\Psi_t},(G+|\mu_y|)\sqrt{\log(\log(T)/\delta)}\}\sqrt{\log(\log(T)/\delta)}}{T}+\Big(1-\frac{t}{T}\Big)|\mu_y|,$$

其中 $\sqrt{\Psi_t}\le\sqrt{c_0c_1\,\sigma^{*2}_{1:t}}+O(T^{1/4})$，$\sigma^{*2}_{1:t}$ 是"事后最优固定查询概率"下的累积条件方差。这条线最终给出全文最反直觉的结论：**当查询概率对当前协变量 $x_t$ 不可见（只能依赖历史）时，最优做法就是把 $p_t$ 设成常数 $T_b/T$**；任何形式的不确定性估计（哪怕依赖过去的协变量/不确定性）都不带来明显优势。FTRL 的作用就是快速收敛到这个常数并保持住。

### 损失函数 / 训练策略
- **估计量更新**：$w_{t+1}=w_t+\tfrac{1}{T}\big(f_t(x_t)+(y_t-f_t(x_t))\tfrac{\xi_t}{p_t}\big)$，输出方差估计 $\hat\sigma^2=\tfrac{1}{T}\sum_t\big(f_t(x_t)+(y_t-f_t(x_t))\tfrac{\xi_t}{p_t}-w_{T+1}\big)^2$。
- **FTRL 损失**：$\tilde\ell_t(p)=\Phi_t(x_t)/p$，正则 $R(p)=\tfrac12 p^2$，步长 $\gamma=\tfrac{1}{\sqrt T}\tfrac{\beta^2}{B}$。
- **模型/oracle 更新**：与 ZC24 的差别之一是每次更新预测模型 $f$ 时也更新不确定性预测器/oracle；并把查到的带标签数据**等分成两份不相交子集** $D^\clubsuit,D^\spadesuit$，分别用于更新 $f_{t+1}$ 与 $u_{t+1}/\Phi_{t+1}$，避免 oracle 在见过的数据上低估测试期不确定性。

## 实验关键数据

### 数据集与协议
三个真实数据集 + 一个合成数据集：礼貌度评分（21 维特征 + ChatGPT 分，回归）、红酒评分（GPT-4o mini 预测，线性回归）、选举后调查（二分类，XGBoost 作 $f$，$u_t=2\min\{f_t,1-f_t\}$）。对比三种策略：本文 **FTRL**、ZC24 混合策略、以及 **uniform sampling 基线**（注意：该基线用**固定**预测器 $f$ 且 $\xi_t\sim\text{Bernoulli}(T_b/T)$，见式 6，比"均匀查询但仍更新模型"更弱）。每个设置跑 50 次。

### 主结果（置信区间宽度，越小越好；趋势源自 Figure 2–4）

| 数据集 | 指标 | FTRL（本文） | ZC24 混合 | uniform sampling 基线 |
|--------|------|--------------|-----------|------------------------|
| 礼貌度评分 | 区间宽度随 $T_b$ | 与 ZC24 相当，部分略窄 | 与 FTRL 相当 | 明显更宽 |
| 红酒评分 | 区间宽度随 $T_b$ | 与 ZC24 相当 | 与 FTRL 相当 | 明显更宽 |
| 选举后调查 | 区间宽度随 $T_b$ | 与 ZC24 相当 | 与 FTRL 相当 | 明显更宽 |
| 覆盖率 | 包含真均值比例 | 高 | 高 | 高 |

结论：四个数据集上 FTRL 与 ZC24 混合策略**宽度相当**，其中两个数据集 FTRL 略窄；二者都优于固定预测器的 uniform sampling 基线；三种策略覆盖率都高（区间有效）。

### 混合系数消融（Figure 1 / Figure 5，扫 $\lambda\in\{0.05,0.1,0.5,0.8,1.0\}$）

| 混合系数 $\lambda$ | 含义 | 区间宽度（相对 0.5） |
|--------------------|------|----------------------|
| 0.5 | ZC24 默认 | 基准 |
| 1.0 | 纯均匀、忽略不确定性 | 相当或**略窄** |
| 0.05 / 0.1 | 偏重不确定性 | 不更优 |

四个数据集上 $\lambda=1$（完全不用不确定性）都给出与 $\lambda=0.5$ 相当甚至更窄的区间，这正是触发整篇理论分析的实验观察。

### 关键发现
- **不确定性分量是脆弱的**：把权重全压到均匀策略（$\lambda=1$）不仅不掉点、有时还更好——理论（Theorem 2 + FTRL 收敛到 $\tau$）解释了原因：对当前协变量不可见时，常数 $T_b/T$ 就是最优查询概率。
- **收益来自"更新模型"而非"聪明查询"**：FTRL 很快收敛到 $T_b/T$（与 uniform sampling 基线**同样的查询概率**），却仍显著优于该基线——差距来自持续更新 $f$ 与 oracle，而不是不确定性加权的查询策略。
- **数据相关界更紧的条件**：当 $\sum_s\sigma_s^2\ll 2t(G^2+\mu_y^2)$ 时，速率可快于 $O(1/\sqrt t)$；burn-in 阶段则受 $(1-t/T)|\mu_y|$ 项拖累。

## 亮点与洞察
- **用 no-regret 反推"均匀最优"**：把"选查询概率"写成损失 $\Phi_t/p$ 的在线凸优化后，由于 $\Phi\ge0$ 使单步最优恒在上界 $\tau$，sublinear regret 直接逼着 $p_t\to T_b/T$。这是一个很漂亮的论证——不靠分布假设，仅凭 regret 几何就锁定了最优策略形态。
- **"任意时刻成立"的非渐近界**：借 Freedman 鞅不等式得到对所有 $t$ 一致成立、且自适应累积方差的界，把 ZC24 只有渐近正态的结果补成了 anytime-valid 形式，对序贯/流式场景更实用。
- **可迁移的判别经验**：当一个主动采样/标注方案"看起来"应当利用样本级不确定性时，先做一个"对当前样本盲、只看历史"的退化版本对照——如果两者差不多，说明真正起作用的是模型更新而非查询策略，能省掉脆弱且难调的不确定性预测器。

## 局限与展望
- **结论的边界条件很关键**："均匀最优"只在查询概率**对当前协变量 $x_t$ 不可见**时成立；若允许 $p_t$ 真正条件于 $x_t$（如 ZC24 实际做的），理论上仍可能有收益，本文未否定这一情形，只是说明"依赖历史的不确定性"无用。
- **依赖近似 oracle 的有界与夹逼假设**：Theorem 2 需要 $\Phi_t\le B$ 且 $\Phi_t$ 在常数因子内逼近真条件方差；现实中残差回归的 oracle 是否满足、$c_0,c_1$ 多大，文中未量化。
- **一维、固定分布设定**：分析限于标量标签、固定（非分布漂移）数据流；高维、多组或分布漂移下结论是否延续值得探究。
- **改进思路**：把 FTRL 框架推广到"可条件于当前协变量"的策略类（如上下文在线学习），看能否在理论上重新夺回不确定性分量的收益；或与 PPI++ 的功效系数调参结合。

## 相关工作与启发
- **vs Zrnic & Candes (2024, ZC24)**：本文沿用其无偏估计量与预算约束，但 ① 补上非渐近 anytime 界（ZC24 只有渐近正态）；② 用 FTRL 在线学查询概率替代"不确定性混合均匀 + 固定 $\lambda$"；③ 训练上额外更新 oracle、并把带标签数据等分给 $f$ 与 oracle。核心反转：ZC24 强调用当前 $x_t$ 的不确定性，本文证明对当前协变量盲时均匀即最优。
- **vs PPI / PPI++ (Angelopoulos et al. 2023a,b)**：PPI 假设有一小批预标注数据来校正预测偏差，PPI++ 用功效系数控制预测影响；本文属于其"主动 + 序贯 + 在线更新"的分支，关注的是**查询概率**而非功效系数的选取。
- **vs 非渐近 PPI 分析 (Mani et al. 2025)**：同样关注 PPI 类估计量的非渐近行为，本文专注于"主动序贯 + 在线学习查询概率"这一具体设置，并给出 FTRL 收敛到 $T_b/T$ 的结构性结论。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ 在已有估计量上补非渐近界并不算颠覆，但"用 FTRL regret 几何反推均匀最优"的视角新颖且有说服力
- 实验充分度: ⭐⭐⭐⭐☆ 三真实 + 一合成数据集、含混合系数扫描与覆盖率，足以支撑理论结论；但都是低维标量任务
- 写作质量: ⭐⭐⭐⭐☆ 从实验异象出发推动理论、逻辑闭环清晰，公式与假设交代完整
- 价值: ⭐⭐⭐⭐☆ 给主动序贯估计补上 anytime 保证，并提供一条"何时不必做不确定性建模"的实用判据

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Multiple-Prediction-Powered Inference](multiple-prediction-powered_inference.md)
- [\[ICLR 2026\] Mean Estimation from Coarse Data: Characterizations and Efficient Algorithms](mean_estimation_from_coarse_data_characterizations_and_efficient_algorithms.md)
- [\[ICLR 2026\] Revenue Maximization under Sequential Price Competition via the Estimation of s-Concave Demand Functions](revenue_maximization_under_sequential_price_competition_via_the_estimation_of_s-.md)
- [\[ICLR 2026\] Revisiting Tree-Sliced Wasserstein Distance through the Lens of the Fermat–Weber Problem](revisiting_tree-sliced_wasserstein_distance_through_the_lens_of_the_fermatweber_.md)
- [\[ICLR 2026\] Singleton-Optimized Conformal Prediction](singleton-optimized_conformal_prediction.md)

</div>

<!-- RELATED:END -->
