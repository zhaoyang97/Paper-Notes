---
title: >-
  [论文解读] Tighter Performance Theory of FedExProx
description: >-
  [ICLR 2026][优化/理论][联邦学习] 本文重新审视联邦外推近端方法 FedExProx，指出其原有理论在二次问题上竟然不比最朴素的梯度下降（GD）更好，进而用「先证距离收敛、再翻译成函数值收敛」的新分析绕开了导致松弛的 $L_{\max}$ 依赖，首次严格证明在通信主导计算的真实联邦场景下 FedExProx 的总时间复杂度可以**严格优于 GD**，并把结论推广到部分参与、自适应外推与 PŁ 条件。
tags:
  - "ICLR 2026"
  - "优化/理论"
  - "联邦学习"
  - "外推"
  - "近端算子"
  - "收敛速率"
  - "通信复杂度"
---

# Tighter Performance Theory of FedExProx

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=PHsFWKJhGv](https://openreview.net/forum?id=PHsFWKJhGv)  
**代码**: 无  
**领域**: 联邦优化 / 凸优化理论  
**关键词**: 联邦学习, 外推, 近端算子, 收敛速率, 通信复杂度

## 一句话总结
本文重新审视联邦外推近端方法 FedExProx，指出其原有理论在二次问题上竟然不比最朴素的梯度下降（GD）更好，进而用「先证距离收敛、再翻译成函数值收敛」的新分析绕开了导致松弛的 $L_{\max}$ 依赖，首次严格证明在通信主导计算的真实联邦场景下 FedExProx 的总时间复杂度可以**严格优于 GD**，并把结论推广到部分参与、自适应外推与 PŁ 条件。

## 研究背景与动机

**领域现状**：联邦学习要在「不集中数据」的前提下求解 $\min_x f(x) = \frac{1}{n}\sum_{i=1}^n f_i(x)$，核心瓶颈是通信。一类强力做法是**近端方法**：客户端不传梯度，而是计算近端算子 $\mathrm{prox}_{\gamma f_i}(x) = \arg\min_z \{ f_i(z) + \frac{1}{2\gamma}\|z-x\|^2 \}$，再在服务器聚合。FedProx 把 FedAvg 的局部梯度步换成近端步，而 **FedExProx**（Li et al., 2024）在 FedProx 之上加了**外推**：用更新 $x^{k+1} = x^k + \alpha_k(\frac{1}{n}\sum_i \mathrm{prox}_{\gamma f_i}(x^k) - x^k)$，取 $\alpha_k > 1$ 比简单平均更快。它的迭代复杂度 $O(L_\gamma(1+\gamma L_{\max})R^2/\varepsilon)$ 被证明优于 FedProx 和 FedExP。

**现有痛点**：作者把这个看似漂亮的界放到二次优化任务上一算，发现一件尴尬的事——它**不比 vanilla GD 更好**。GD 的通信复杂度是 $O(LR^2/\varepsilon)$，而 FedExProx 的界里带着 $L_{\max}$（所有客户端光滑常数的最大值），在异质数据下常有 $L_{\max} \gg L$。一旦把「一步要花多少时间」也考虑进去，最优策略竟然是令 $\gamma \to 0$，此时 FedExProx 退化成 GD，近端算子白用了。

**核心矛盾**：到底是 FedExProx 这个方法本身就不行，还是只是 Li et al. (2024) 的分析太松？换言之，$L_{\max}$ 的依赖是方法的内在代价，还是分析的人为产物？

**本文目标**：（1）严格刻画原分析为何只能给出「不优于 GD」的结论；（2）造一套更紧的分析，让 FedExProx 在理论上真正赢过 GD；（3）把结论扩展到部分参与、自适应外推、以及超出二次的 PŁ 函数类。

**切入角度**：作者追查发现 $L_{\max}$ 来自原证明所依赖的一条引理（Lemma A.9）——它把 Moreau 包络的下降量翻译成原函数下降量时引入了 $\frac{1}{1+\gamma L_{\max}}$ 因子。如果换一条不经过这条引理的证明路径，就有望甩掉 $L_{\max}$。

**核心 idea**：**先在「距离」意义下证收敛 $\mathbb{E}\|\bar x - x^*\|^2 \le \varepsilon$，再用 $f$ 的 $L$-光滑性翻译成函数值收敛**，从而绕开那条引入 $L_{\max}$ 的引理，得到只依赖 $L_\gamma/\mu_\gamma^+$ 的更紧线性速率。

## 方法详解

### 整体框架

这是一篇纯理论论文，没有新算法，「方法」即一套**新的收敛分析框架**。整条论证沿四步推进：① **诊断**——把 Li et al. (2024) 的界翻译到二次问题，证明在引入时间模型后它恒不优于 GD（Theorem 3.2），定位病根是 $L_{\max}$；② **重证**——改走距离收敛路径，给非强凸二次问题建立线性速率 $O(\frac{L_\gamma}{\mu_\gamma^+}\log\frac{1}{\varepsilon})$（Theorem 4.1），其中 $L_\gamma$、$\mu_\gamma^+$ 是平均 Moreau 包络 $M^\gamma$ 的最大/最小非零特征值；③ **时间建模**——给单步成本一个显式模型 $\pi(\gamma) = \eta + \tau(\gamma L_{\max}+1)$（通信 $\eta$ + 计算 $\tau$），把迭代数乘上去得到总时间，证明存在 $\gamma>0$ 使 FedExProx 严格快于 GD（Theorem 4.3）；④ **推广**——把同一套距离分析搬到部分参与（nice sampling）、自适应外推（GraDS / StoPS）和满足 PŁ 条件的一般函数。四步共享同一个核心技巧（距离收敛），层层放宽设定。

### 关键设计

**1. 诊断旧分析：FedExProx 在二次问题上并不强于 GD**

要证明自己的新分析有价值，作者先把旧分析逼到墙角。把 Li et al. (2024) 的迭代复杂度 $O(L_\gamma(1+\gamma L_{\max})R^2/\varepsilon)$ 配上一个合理的单步时间假设——「一步耗时 $\pi(\gamma)$ 关于 $\gamma$ 单调不减」（Assumption 3.1，因为 $\gamma$ 越大、局部近端子问题越难解：子问题 $f_i(z)+\frac{1}{2\gamma}\|z-x\|^2$ 是 $(L_i+1/\gamma)$-光滑、$1/\gamma$-强凸，GD 解它要 $\tilde O(\gamma L_i + 1)$ 步）。于是总时间 $T(\gamma) = \pi(\gamma)\cdot\frac{L_\gamma(1+\gamma L_{\max})R^2}{\varepsilon}$。Theorem 3.2 证明对一切 $\gamma>0$ 都有

$$T(\gamma) \;\ge\; \pi(0)\cdot\frac{L R^2}{\varepsilon},$$

且当 $\gamma\to 0$ 时取等、FedExProx 退化为 GD。换句话说，在旧分析下最优策略就是不用近端算子。这一步的价值不在「贬低」FedExProx，而在精准定位：不等式恒成立的根源是界里那个 $L_\gamma(1+\gamma L_{\max}) \ge L$ 的 $L_{\max}$ 因子，而 GD 只依赖 $L$。病根锁定，才知道该往哪里动刀。

**2. 距离收敛绕开 $L_{\max}$：建立更紧的线性速率**

旧分析为何带 $L_{\max}$？因为它要直接证 $\mathbb{E}[f(\bar x)] - f(x^*)\le\varepsilon$，中途必须用 Lemma A.9：$M^\gamma(x)-M^\gamma(x^*) \ge \frac{1}{1+\gamma L_{\max}}(f(x)-f(x^*))$，这一步把 $L_{\max}$ 塞了进来。作者的关键一招是**换一个收敛度量**：先证迭代在距离意义下线性收缩 $\mathbb{E}\|\bar x - x^*\|^2\le\varepsilon$，再仅用 $f$ 的全局 $L$-光滑性把它翻译成函数值差。这条路径根本不经过 Lemma A.9，因此 $L_{\max}$ 不再出现。Theorem 4.1 给出非强凸二次问题的新迭代复杂度

$$O\!\left(\frac{L_\gamma}{\mu_\gamma^+}\log\frac{1}{\varepsilon}\right),$$

其中 $L_\gamma$ 是 $M^\gamma$ 的光滑常数、$\mu_\gamma^+$ 是 $\nabla^2 M^\gamma$ 的**最小非零特征值**。作为对照，GD 在同设定下需要 $O(\frac{L}{\mu^+}\log\frac{1}{\varepsilon})$（$\mu^+$ 是 $A=\frac1n\sum_i A_i$ 的最小非零特征值）。两者形式同构，但 FedExProx 的条件数 $L_\gamma/\mu_\gamma^+$ 可以远小于 $L/\mu^+$——这正是外推+近端算子带来的真实增益，旧分析因为 $L_{\max}$ 把它掩盖了。

**3. 计算+通信时间模型：证明通信主导时严格快于 GD**

光有更小的迭代数还不够，必须把「每步多贵」算进去才公平。作者给单步时间一个显式分解：局部计算受最慢客户端约束、耗时 $\tilde O(\tau(\gamma L_{\max}+1))$，通信耗时 $\eta$，故

$$\pi(\gamma) = \eta + \tau(\gamma L_{\max}+1),\qquad T_\eta(\gamma) = \tilde O\!\left((\eta + \tau(\gamma L_{\max}+1))\cdot\frac{L_\gamma}{\mu_\gamma^+}\right).$$

GD 对应 $T_{\mathrm{GD}} = T_\eta(0) = \tilde O((\eta+\tau)\cdot L/\mu^+)$。Theorem 4.3 证明：最优 $\gamma$ 落在一个显式区间内，并且总有 $T_\eta(\gamma)\le T_{\mathrm{GD}}$；当通信主导（$\eta/\tau \ge 2$）时最优 $\gamma$ 严格大于 0，FedExProx 严格更快。Example 4.4 用对角矩阵给出量化：取 $\gamma = 1/\min_{i,j} a_{ij}$ 可让 $L_\gamma/\mu_\gamma^+\le 2$，当 $\eta$ 足够大时 $T_\eta(\gamma)=\tilde O(\eta)$，而 GD 是 $\tilde\Omega(\eta\cdot \frac{\max_j\sum_i a_{ij}}{\min_j\sum_i a_{ij}})$，后者可大出一个条件数倍。这与旧分析（Theorem 3.2）「永远不优于 GD」形成鲜明反差，是本文最核心的「翻盘」结论。

**4. 推广到部分参与、自适应外推与 PŁ 条件**

同一套距离分析被层层加码。**部分参与**：用 nice sampling（每轮均匀抽 $S$ 个客户端），Theorem 5.1 给出复杂度 $O(\frac{L_{\gamma,S}}{\mu_\gamma^+}\log\frac{1}{\varepsilon})$，其中 $L_{\gamma,S}=\frac{n-S}{S(n-1)}\frac{L_{\max}}{1+\gamma L_{\max}} + \frac{n(S-1)}{S(n-1)}L_\gamma$，且 Theorem 5.2 表明随机设定下的最优 $\gamma$ 与确定性情形一致，结论照搬不误。**自适应外推**（Theorem 6.1）：GraDS 用梯度多样性 $\alpha_k = \frac{\frac1n\sum\|\nabla M^\gamma_{f_i}\|^2}{\|\frac1n\sum\nabla M^\gamma_{f_i}\|^2}\ge 1$ 自动定外推系数，StoPS 用 Polyak 风格步长，二者都对 $\gamma$ **半自适应**——对任意 $\gamma>0$ 收敛，只要 $\gamma$ 取够大就接近最优，无需预先知道最优 $\alpha=1/\gamma L_\gamma$（Remark 6.3：StoPS 在 $O(\frac{L_\gamma}{\mu_\gamma^+}\log\frac1\varepsilon)$ 内收敛，代价是要知道平均 Moreau 包络的最小值）。**超出二次**（Theorem 7.2 / 7.5）：把 $\mu_\gamma^+$ 从「特征值」换成「PŁ 常数」，对满足 PŁ 条件的一般函数得到同构的线性速率。相比 Li et al. (2024) 需要 $f$ **强凸**（解唯一），本文只要 PŁ（允许多解），假设更弱、对问题常数的依赖更好，还能容忍近端算子的不精确计算（Theorem E.1）。

### 损失函数 / 训练策略
本文为理论分析，无训练损失。唯一的「超参」是近端正则 $\gamma$ 与外推系数 $\alpha$：常数外推取最优 $\alpha=1/(\gamma L_\gamma)$；$\gamma$ 的最优取值由时间模型 Theorem 4.3 给出的显式区间决定，并随通信/计算耗时比 $\eta/\tau$ 移动。

## 实验关键数据

实验为验证理论而设，并非刷 benchmark。目标函数取随机二次 $f(x)=\frac1n\sum_i\frac12 x^\top A_i x$，$A_i\in\mathrm{Sym}^7_+$ 且最小特征值为 0（即非强凸），$n=14$ 个客户端用 GD 解局部近端子问题，外推系数取 Theorem 4.1 的最优值。核心是观察「**总经验时间复杂度随 $\gamma$ 如何变化**」。

### 主实验：最优 $\gamma$ 的 U 形曲线

| 通信耗时 $\eta$ | 最优 $\gamma$ 位置 | 理论预测一致性 | 含义 |
|------|------|------|------|
| 小（$\eta\propto 100$） | 接近 0 | 落在 Thm 4.3 边界内 | 计算贵→少用近端→近似 GD |
| 中 | 向 $>0.1$ 偏移 | 一致 | 通信渐贵→近端开始划算 |
| 大（$\eta\propto 10^5$） | 明显 $>0$ | 一致 | 通信主导→外推近端显著占优 |

随着 $\eta$ 增大，总时间-$\gamma$ 曲线呈现明显的 **U 形**，最优 $\gamma$ 从 0 附近逐步右移到大于 0.1 的位置，且始终落在 Theorem 4.3 给出的理论上下界（图中虚线）之间。这直接印证了核心论断：**通信越主导，非平凡的 $\gamma>0$ 越有利，FedExProx 越能赢过 GD**。

### 关键发现
- **U 形是关键证据**：若像旧分析那样最优 $\gamma$ 永远在 0，曲线应单调、最优点贴边；实测的 U 形说明存在内部最优 $\gamma^*>0$，与新分析吻合、与旧分析矛盾。
- **理论边界很紧**：经验最优 $\gamma$ 落在 Theorem 4.3 预测的窄区间内，说明新界不只是定性正确，量级也对得上。
- **现象超出二次**：作者指出在 ARCENE 等真实数据集上也观察到类似规律（附录 G），暗示「外推在通信主导下有利」的结论在二次之外仍成立。

## 亮点与洞察
- **「换收敛度量」是四两拨千斤的关键技巧**：直接证函数值收敛会被迫引入 $L_{\max}$；改为先证距离收敛、再用全局 $L$-光滑翻译，一步绕开病根。这种「绕道证明」的思路对其他被 $L_{\max}$ 卡住的联邦/分布式分析有直接借鉴价值。
- **把「时间」写进复杂度**：很多优化论文只比迭代数，本文坚持用 $\pi(\gamma)=\eta+\tau(\gamma L_{\max}+1)$ 把通信与计算耗时显式建模，才让「FedExProx vs GD」的比较有现实意义——这也是它能给出「通信主导时严格更快」这种可落地结论的前提。
- **诚实地拿自己开刀**：论文先花一整节证明「在旧分析下该方法不如 GD」，把 baseline 逼到极致再翻盘，论证张力强、可信度高。
- **更弱假设换更强结论**：把强凸换成 PŁ，既扩大了适用范围（允许过参数化模型的多解），又改善了对问题常数的依赖，是「假设更弱、结论更好」的少见双赢。

## 局限与展望
- **局限在二次/PŁ**：最干净的结果（含时间最优 $\gamma$ 的显式区间、U 形验证）都在非强凸二次问题上；PŁ 推广虽更一般，但 Theorem 4.3 那种精细的时间复杂度刻画尚未完全搬过去。一般非凸深度网络仍是开放问题。
- **依赖插值假设**：全文建立在 Assumption 1.3（存在 $x^*$ 使所有 $\nabla f_i(x^*)=0$）之上，这对过参数化模型常成立，但对一般异质联邦数据未必。
- **时间模型理想化**：$\pi(\gamma)=\eta+\tau(\gamma L_{\max}+1)$ 假设通信时间 $\eta$ 恒定、计算被最慢客户端线性 gating；真实网络的抖动、异步、掉线只在部分参与里部分覆盖。
- **StoPS 的代价**：自适应外推免去了对最优 $\alpha$ 的预知，但 StoPS 需要知道平均 Moreau 包络的最小值，实际中未必易得。
- **展望**：作者明确把本文定位为「垫脚石」，下一步是放宽插值/二次假设，在更一般、更贴近实践的设定下证明外推的同类增益。

## 相关工作与启发
- **vs Li et al. (2024)（FedExProx 原作）**: 同一个算法，本文给出**更紧的分析**。原作证 $O(L_\gamma(1+\gamma L_{\max})R^2/\varepsilon)$，被 $L_{\max}$ 拖累到不优于 GD；本文换距离收敛路径得 $O(\frac{L_\gamma}{\mu_\gamma^+}\log\frac1\varepsilon)$，首次证明可严格快于 GD。强凸假设也被弱化为 PŁ。
- **vs GD（Nesterov, 2018）**: GD 通信复杂度 $O(LR^2/\varepsilon)$、不需近端 oracle；本文以 GD 为「教学性最简 baseline」，证明在通信主导的联邦场景 FedExProx 总时间严格更优，量化增益可达一个条件数倍（Example 4.4）。
- **vs FedProx（Li et al., 2020）/ FedExP（Jhunjhunwala et al., 2023）**: FedProx 仅在服务器平均近端步、FedExP 把外推用于 FedAvg；FedExProx 把外推用到近端步上，本文进一步把它的理论保证从「优于 FedProx/FedExP」推进到「优于 GD」。

## 评分
- 新颖性: ⭐⭐⭐⭐ 「换收敛度量绕开 $L_{\max}$」的技巧巧妙，把一个被认为不优于 GD 的方法翻盘
- 实验充分度: ⭐⭐⭐ 实验只为验证理论（二次+少量真实数据），U 形证据有力但规模小
- 写作质量: ⭐⭐⭐⭐⭐ 「先自我证伪再翻盘」的叙事清晰，病根定位与修法逻辑环环相扣
- 价值: ⭐⭐⭐⭐ 为联邦外推近端方法提供了更可信的理论基础，分析技巧可迁移

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Elastic Optimal Transport: Theory, Application, and Empirical Evaluation](elastic_optimal_transport_theory_application_and_empirical_evaluation.md)
- [\[ICLR 2026\] Bilevel Optimization with Lower-Level Uniform Convexity: Theory and Algorithm](bilevel_optimization_with_lower-level_uniform_convexity_theory_and_algorithm.md)
- [\[NeurIPS 2025\] Learning Theory for Kernel Bilevel Optimization](../../NeurIPS2025/optimization/learning_theory_for_kernel_bilevel_optimization.md)
- [\[ICML 2026\] Towards Understanding Continual Factual Knowledge Acquisition of Language Models: From Theory to Algorithm](../../ICML2026/optimization/towards_understanding_continual_factual_knowledge_acquisition_of_language_models.md)
- [\[ICML 2026\] Adaptive Sharpness-Aware Minimization with a Polyak-type Step size: A Theory-Grounded Scheduler](../../ICML2026/optimization/adaptive_sharpness-aware_minimization_with_a_polyak-type_step_size_a_theory-grou.md)

</div>

<!-- RELATED:END -->
