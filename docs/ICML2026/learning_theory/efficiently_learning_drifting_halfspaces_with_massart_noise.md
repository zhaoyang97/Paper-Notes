---
title: >-
  [论文解读] Efficiently Learning Drifting Halfspaces with Massart Noise
description: >-
  [ICML2026][学习理论][分布漂移] 在分布随时间漂移、标签又被 Massart 噪声污染的在线学习场景下，本文给出第一个**多项式时间**学习 $\gamma$-间隔半空间的算法，误差为 $\eta+\tilde{O}(\Delta^{1/3}/\gamma)$，并用低次多项式下界证明 $\Delta^{1/3}$ 这个指数对高效算法来说是无法绕过的，从而揭示出一条信息-计算鸿沟。
tags:
  - "ICML2026"
  - "学习理论"
  - "分布漂移"
  - "半空间学习"
  - "Massart 噪声"
  - "信息-计算鸿沟"
  - "低次多项式下界"
---

# Efficiently Learning Drifting Halfspaces with Massart Noise

**会议**: ICML2026  
**arXiv**: [2606.11149](https://arxiv.org/abs/2606.11149)  
**代码**: 无（纯理论文）  
**领域**: 学习理论  
**关键词**: 分布漂移, 半空间学习, Massart 噪声, 信息-计算鸿沟, 低次多项式下界

## 一句话总结
在分布随时间漂移、标签又被 Massart 噪声污染的在线学习场景下，本文给出第一个**多项式时间**学习 $\gamma$-间隔半空间的算法，误差为 $\eta+\tilde{O}(\Delta^{1/3}/\gamma)$，并用低次多项式下界证明 $\Delta^{1/3}$ 这个指数对高效算法来说是无法绕过的，从而揭示出一条信息-计算鸿沟。

## 研究背景与动机
**领域现状**：经典统计学习理论假设训练样本来自一个固定分布，且测试也在同一分布上。但金融预测、消费者偏好、天气、乃至语言模型这类应用里，数据分布是随时间**漂移**的——模型在旧快照上训练，却要在新数据上服役。形式化的模型是"分布漂移下的二分类"：相邻两轮分布的总变差距离被漂移率约束 $d_{\mathrm{TV}}(D^{(t)},D^{(t+1)})\le\Delta$，学习器在线地每轮先输出一个假设、再收到一个带噪样本。

**现有痛点**：早期工作（HLO91、Bar92、HL94 等）已经刻画了**统计最优**误差——可实现情形 $\Theta((d\Delta)^{1/2})$、不可知情形 $\Theta((d\Delta)^{1/3})$——但这些结果都依赖经验风险最小化（ERM），**一般是指数时间**的。换句话说，统计上知道"能学到多好"，计算上却不知道"高效算法能学到多好"。对半空间这个最基本的类，已有的高效算法要么误差很差（HL94 的 $\tilde{O}(d\sqrt{\Delta})$，差了 $\sqrt{d}$ 因子），要么需要很强的边缘分布假设（CMEDV10、HKY15 要求样本均匀分布在单位球面）。

**核心矛盾**：漂移本身不增加可学性的"信息"门槛（Massart/RCN 的统计最优误差仍是 $\Delta^{1/2}$ 量级，和可实现情形同档），但**高效**算法能否达到这个统计最优却完全没人知道；而且一旦在无结构边缘分布下，$d_{\mathrm{TV}}$ 小并不能推出真实权重向量 $w^{*(i)}$ 与 $w^{*(i+1)}$ 几何距离小——这是把无噪声分析搬过来的最大障碍。

**本文目标**：在**最小分布假设**（只要求对目标半空间存在间隔 $\gamma$）+ **Massart 标签噪声**下，设计多项式时间（关于 $d,1/\gamma,1/\Delta$）的漂移半空间学习器，并搞清楚高效算法能达到的误差极限。

**切入角度**：选 Massart 噪声（每个样本以未知概率 $\eta(x)\le\eta<1/2$ 翻转标签）是因为它在无漂移情形下已有高效学习器（DGT19），而不可知（对抗）噪声即便有间隔也被证明无法高效学习。作者借用 DKTZ25 在固定半空间上的漏 ReLU 梯度，再想办法把它迁移到漂移设定。

**核心 idea**：把"在线投影梯度下降 + 漏 ReLU 噪声鲁棒梯度"按 epoch 分块运行，关键在于一个**Regret-to-Error 引理**——它绕开了追踪 $w^*$ 漂移这件难事，直接用"每轮回报 + 漂移误差项"控住预测误差，从而把高效算法的误差钉在 $\eta+\tilde{O}(\Delta^{1/3}/\gamma)$。

## 方法详解

### 整体框架
算法 `DriftedMassart`（Algorithm 1）以**长度 $W$ 的 epoch** 为单位运行。在第 $i$ 个 epoch 内，学习器一直用上个 epoch 学到的假设 $\hat{h}^{(i-1)}$ 做预测，同时把这个 epoch 收到的所有带噪样本攒进集合 $S^{(i)}$；epoch 结束时（$t=iW$）调用子程序 `DriftPerceptron`（Algorithm 2）在 $S^{(i)}$ 上重新学一个半空间，作为下一 epoch 的预测假设。epoch 长度是整个设计的命门，取 $W=2\lfloor\Delta^{-2/3}\log(1/\Delta)\rfloor$。

`DriftPerceptron` 内部把 $S^{(i)}$ 平分两半：前一半 $S_1$ 用来**在线更新**权重向量 $w$（每个样本走一步投影梯度下降，留下一条 $w^{(1)},w^{(2)},\dots$ 的更新轨迹），后一半 $S_2$ 用来**从这条轨迹里挑出经验误差最小的那个假设**输出。整篇论文的技术核心不在算法形式（它就是带特定梯度的在线 PGD），而在**分析**：怎么证明这条轨迹里存在一个误差小到 $\eta+\tilde{O}(\Delta^{1/3}/\gamma)$ 的假设，并且能被 $S_2$ 的经验误差可靠地选出来。

整个误差预算被拆成三块——优化误差 $\tfrac{1}{2T\mu}+\tfrac{2\mu}{\gamma^2}$、漂移误差 $O(T\Delta/\gamma)$、集中误差 $\sum_i\xi_i$——前两块靠调 $W$ 和步长 $\mu=\gamma/\sqrt{T}$ 配平，第三块靠鞅集中控住。三者在 $T=\tilde{\Theta}(\Delta^{-2/3})$ 时同时被压到 $\tilde{O}(\Delta^{1/3}/\gamma)$。

### 关键设计

**1. Epoch 分块 + epoch 长度的漂移-估计权衡：把"样本越多估计越准"和"样本越旧越失真"配平**

漂移学习的根本张力是：要降低估计/集中误差就得用更多历史样本，但样本越旧、它对应的目标分布离当下越远，引入的漂移误差越大。本文把这个权衡量化进 epoch 长度 $W$。一个 epoch 内用 $T=W/2$ 个样本做梯度更新，优化误差 $\sim 1/(\sqrt{T}\gamma)$ 随 $T$ 增大而下降，漂移误差 $\sim T\Delta/\gamma$ 随 $T$ 增大而上升。令两者相等解出 $T=\Theta(\Delta^{-2/3})$，于是 $W=\Theta(\Delta^{-2/3}\log(1/\Delta))$，此时两项都落在 $\tilde{O}(\Delta^{1/3}/\gamma)$。这就是 $\Delta^{1/3}$ 指数的算术来源：它正是 $1/\sqrt{T}$ 与 $T\Delta$ 平衡点的值 $\Delta^{1/3}$。

**2. 漏 ReLU 噪声鲁棒梯度 + 在线投影梯度下降：让 Massart 噪声下的更新仍朝正确方向走**

每轮对样本 $(x,y)$ 构造"梯度"向量
$$g(w^{(i)};x,y)=\frac{\big((1-2\eta)\,\mathrm{sign}(w^{(i)}\cdot x)-y\big)\,x}{\max\{|w^{(i)}\cdot x|,\gamma\}},$$
然后做一步投影梯度下降 $w^{(i+1)}=\mathrm{proj}_{\mathbb{B}(1)}(w^{(i)}-\mu g)$。这个 $g$ 可以看成漏 ReLU 损失 $\ell_\eta(w;x,y)=(1-2\eta)|yw\cdot x|+yw\cdot x$ 的梯度，再被 $\max\{|w\cdot x|,\gamma\}^{-1}$ 加权——权重的作用是给不同间隔的错分样本不同惩罚，避免靠近决策边界的样本主导更新。$(1-2\eta)$ 因子是处理 Massart 噪声的关键：它把翻转概率 $\le\eta$ 的期望偏置正好抵消掉，使期望更新方向仍指向真实半空间。这套梯度借鉴自固定半空间设定（DKTZ25），但分析必须全部重做。

**3. Regret-to-Error 引理：绕过"追踪 $w^*$ 漂移"，直接用回报项控住每轮预测误差**

把固定半空间分析搬到漂移设定的最大障碍是：无结构边缘分布下，$d_{\mathrm{TV}}(D^{(i)},D^{(i+1)})$ 小**不**蕴含 $\|w^{*(i)}-w^{*(i+1)}\|$ 小——真实权重可以剧烈摆动。先前工作（CMEDV10、HKY15）正是靠均匀球面假设才得到 $\|w^{*(i)}-w^{*(i+1)}\|$ 小，本文要在没有这个假设的情况下做。作者的破局点是 **Lemma 3.1**：存在一个有界函数 $F_i(w^*,w,x)$ 使得
$$\mathbf{E}_{y\sim D^{(i)}\mid x}[g(w;x,y)\cdot(w-w^*)]\ge 2(\mathrm{err}^{(i)}(w,x)-\eta)-F_i,$$
其中 $|F_i|\le 1/\gamma$ 且 $\mathbf{E}_x F_i=O(\Delta m/\gamma)$。直觉是：哪怕 $w^*$ 漂移得很厉害，**每一轮的预测超额误差 $(\mathrm{err}-\eta)$ 都能被该轮的期望回报项加上一个"漂移误差"$F_i$ 控住**，于是根本不需要逐轮追踪 $w^*$ 的几何变化。这个 $F_i$（"漂移误差"）的期望被进一步归结到两件事：**不一致区域** $\{x:(w^*\cdot x)(w^{*(i)}\cdot x)<0\}$ 的变化（Claim 3.1，量级 $O(\Delta m/(1-2\eta))$）和**噪声率** $\eta_i(x)-\eta_{i+m}(x)$ 的变化（Claim 3.2，量级 $O(\Delta m)$）——两者都只随漂移总量 $m\Delta$ 增长，从而把分布漂移翻译成可控的代数项。

**4. 信息-计算鸿沟：用低次多项式下界证明 $\Delta^{1/3}$ 对高效算法无法绕过**

算法误差 $\Delta^{1/3}$ 在质上**劣于**统计最优的 $\Delta^{1/2}$，这看起来像是分析不够紧。本文反而证明这是高效算法的内在代价。作者在最特化的 RCN（随机分类噪声，$\eta(x)\equiv\eta$）情形下，针对**低次多项式检验**这一被广泛研究、能覆盖大量学习算法的计算模型，建立下界 Theorem 1.2：在低次硬度猜想下，没有多项式时间算法能对一族实例都达到 $\mathrm{opt}_T+\Delta^{1/3}\gamma^{-1/6}$ 的误差。结论非常干净——就漂移 $\Delta$ 的标度而言，有界噪声（Massart/RCN）在**信息论**上和可实现情形同档（$\Delta^{1/2}$），但**计算上**却被推到不可知情形的下限（$\Delta^{1/3}$）。算法上界与该下界相互匹配，说明本文算法在标度意义上本质最优。

### 损失函数 / 训练策略
没有可训练参数，"训练策略"即在线 PGD 的超参选取：步长 $\mu=\gamma/\sqrt{T}$、每 epoch 用前半 $S_1$ 跑更新轨迹、后半 $S_2$ 按经验误差 $\hat{\mathrm{err}}(\hat{h}_j)=\frac{1}{|S_2|}\sum_{(x,y)\in S_2}\mathbb{1}\{\hat{h}_j(x)\ne y\}$ 选出最优假设。最后用 Hoeffding 不等式 + 并集界保证经验误差能以高概率（$0.9$）选出真正好的假设。RCN 情形再补一招：在每个 epoch 内存在某 $\eta^*$ 使 $\mathrm{opt}_t\le\eta^*+O(\Delta W)$，于是用不同 $\eta$ 反复跑 Algorithm 1 即可把保证从 $\eta+\dots$ 升级为 $\mathrm{opt}_t+\tilde{O}(\Delta^{1/3}/\gamma)$（Theorem 3.1）。

## 实验关键数据
本文是纯理论文，没有数值实验。下面两张表汇总它的**理论结果矩阵**，便于和已有工作对照。

### 主结果：各设定下高效算法的误差保证

| 设定 | 本文高效算法误差 | 先前最好（高效） | 说明 |
|------|------------------|------------------|------|
| Massart 漂移 + $\gamma$ 间隔 | $\eta+\tilde{O}(\Delta^{1/3}/\gamma)$ | 无（首个带噪声保证） | Theorem 1.1，$T=\tilde\Omega(\Delta^{-2/3})$ |
| RCN 漂移 + $\gamma$ 间隔 | $\mathrm{opt}_t+\tilde{O}(\Delta^{1/3}/\gamma)$ | 无 | Theorem 3.1 |
| 可实现 + $\gamma$ 间隔 | $\tilde{O}(\sqrt{\Delta}\,\gamma^{-3/2})$ | $\tilde{O}(\sqrt{\Delta}\,\gamma^{-2})$（HL94） | 间隔依赖从 $\gamma^{-2}$ 改进到 $\gamma^{-3/2}$ |
| 可实现（一般 VC 维 $d$） | $\tilde{O}(d\sqrt{\Delta})$（HL94，指数级 ERM 才统计最优） | — | 作为对照基线 |

### 信息论 vs 计算高效：$\Delta$ 标度的鸿沟

| 噪声模型 | 信息论最优（可能指数时间） | 高效算法（本文上界 + 下界） | 结论 |
|----------|----------------------------|------------------------------|------|
| 可实现 | $\Theta((d\Delta)^{1/2})$ | $\Delta^{1/2}$ 量级 | 无鸿沟 |
| Massart / RCN | $\tilde\Theta((d\Delta)^{1/2})$（Thm 2.1/2.2） | $\Delta^{1/3}$（Thm 1.1 上界 + Thm 1.2 低次下界） | **存在信息-计算鸿沟** |
| 不可知（对抗） | $\Theta((d\Delta)^{1/3})$ | 即便有间隔也无高效算法（Dan16） | 计算上不可学 |

### 关键发现
- $\Delta^{1/3}$ 这个指数不是分析松弛的产物，而是 $1/\sqrt{T}$ 与 $T\Delta$ 配平点的代数必然，并被低次多项式下界证明对高效算法无法绕过。
- 有界标签噪声（Massart/RCN）在信息论上和可实现同档（$\Delta^{1/2}$），但高效算法被迫支付不可知情形的代价（$\Delta^{1/3}$）——这是本文最反直觉的洞察。
- 无结构边缘分布下不能再假设 $\|w^{*(i)}-w^{*(i+1)}\|$ 小，Regret-to-Error 引理是绕过这一障碍的唯一支点。

## 亮点与洞察
- **把"追踪漂移的最优解"换成"每轮误差被回报项控住"**：Lemma 3.1 是整篇论文最巧的一步，它让无结构边缘分布下的漂移分析成为可能，思路可迁移到其他漂移 + 噪声的在线学习问题。
- **epoch 长度 $\Delta^{-2/3}$ 的来历透明**：从优化误差 $1/\sqrt{T}$ 与漂移误差 $T\Delta$ 的平衡直接读出指数，让人一眼看穿 $\Delta^{1/3}$ 从何而来。
- **上界与下界对仗工整**：用低次多项式检验这一标准计算模型给出匹配下界，把"算法本质最优"坐实，而不是停留在"我们暂时只能做到这么好"。

## 局限与展望
- 保证形式是 $\eta+\tilde{O}(\dots)$ 而非 $\mathrm{opt}_t+\tilde{O}(\dots)$（一般 Massart 情形），因为 Massart 下即便无漂移，输出误差小于 $\eta-o(1)$ 也是计算困难的（NT22）；只有 RCN 才能升级为 $\mathrm{opt}_t$ 形式。
- 间隔依赖仍是 $1/\gamma$（噪声情形）/ $\gamma^{-3/2}$（可实现），是否最优、能否进一步降到 $\gamma^{-1}$ 量级仍未解决。
- 下界建立在低次硬度猜想之上，属于"条件下界"；将其升级为无条件下界或对更广算法族的下界是自然的后续方向。
- 只处理了半空间这一类；推广到更一般的几何概念类（如多面体、低度多项式阈值函数）的漂移 + 噪声学习仍开放。

## 相关工作与启发
- **vs HL94（基于 LP 的高效漂移半空间）**: 他们在可实现情形给出 $\tilde{O}(d\sqrt{\Delta})$（差 $\sqrt{d}$）且不容噪声；本文容忍 Massart 噪声、并把可实现误差改进到 $\tilde{O}(\sqrt{\Delta}\gamma^{-3/2})$。
- **vs CMEDV10 / HKY15（均匀球面假设下的高效漂移学习）**: 他们靠均匀分布把 $\|w^{*(i)}-w^{*(i+1)}\|$ 控小，本文用 Regret-to-Error 引理在**最小分布假设**（只要间隔）下做到，且额外容忍标签噪声。
- **vs DGT19 / DKTZ25（无漂移 Massart 半空间高效学习）**: 借用了它们的漏 ReLU 噪声鲁棒梯度，但把分析从固定目标搬到漂移目标，核心难点（无单一 ground-truth $w^*$）是本文独有贡献。
- **vs Lon99（不可知漂移的 $\Theta((d\Delta)^{1/3})$ 统计最优）**: 本文揭示有界噪声下高效算法被迫支付与不可知情形相同的 $\Delta^{1/3}$ 标度，把统计与计算两条线串了起来。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个漂移 + 标签噪声下的高效学习保证，并配出匹配的信息-计算鸿沟下界。
- 实验充分度: ⭐⭐⭐⭐ 纯理论文无实验，但理论结果矩阵完整、上下界对仗。
- 写作质量: ⭐⭐⭐⭐⭐ 误差预算拆解清晰，$\Delta^{1/3}$ 的来历讲得透。
- 价值: ⭐⭐⭐⭐⭐ 为非平稳数据下的可学性给出计算视角的标尺，对漂移学习理论有奠基意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Semi-Supervised Noise Adaptation: Transferring Knowledge from Noise Domain](semi-supervised_noise_adaptation_transferring_knowledge_from_noise_domain.md)
- [\[ICML 2026\] Robustness of Mixtures of Experts to Feature Noise](robustness_of_mixtures_of_experts_to_feature_noise.md)
- [\[ICML 2026\] Performative Learning Theory](performative_learning_theory.md)
- [\[ICML 2026\] Bandit Social Learning with Exploration Episodes](bandit_social_learning_with_exploration_episodes.md)
- [\[ICML 2026\] Towards Optimal Robustness in Learning-Augmented Paging](towards_optimal_robustness_in_learning-augmented_paging.md)

</div>

<!-- RELATED:END -->
