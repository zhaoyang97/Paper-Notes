---
title: >-
  [论文解读] Stability Analysis of Sharpness-Aware Minimization
description: >-
  [ICML 2026][优化/理论][SAM] 本文从动力系统视角剖析 SAM 在鞍点附近的收敛不稳定：先在确定性梯度流下证明只要邻域半径 $\rho > -1/\lambda_1$，鞍点就会变成 SAM 的吸引子；再在随机扩散框架下证明 SAM 的鞍点逃逸均方位移比 SGD 小 $2\eta t^2|\lambda_j|^3\rho/B$；最后用 SAM 扩散公式解释 momentum 和 batch size 为什么是 SAM 取得 SOTA 泛化性能的真正幕后功臣。
tags:
  - "ICML 2026"
  - "优化/理论"
  - "SAM"
  - "鞍点逃逸"
  - "动力系统"
  - "扩散方程"
  - "momentum 与 batch size"
---

# Stability Analysis of Sharpness-Aware Minimization

**会议**: ICML 2026  
**arXiv**: [2301.06308](https://arxiv.org/abs/2301.06308)  
**代码**: 无  
**领域**: 优化 / 训练动力学  
**关键词**: SAM、鞍点逃逸、动力系统、扩散方程、momentum 与 batch size

## 一句话总结
本文从动力系统视角剖析 SAM 在鞍点附近的收敛不稳定：先在确定性梯度流下证明只要邻域半径 $\rho > -1/\lambda_1$，鞍点就会变成 SAM 的吸引子；再在随机扩散框架下证明 SAM 的鞍点逃逸均方位移比 SGD 小 $2\eta t^2|\lambda_j|^3\rho/B$；最后用 SAM 扩散公式解释 momentum 和 batch size 为什么是 SAM 取得 SOTA 泛化性能的真正幕后功臣。

## 研究背景与动机
**领域现状**：以 Foret 等人 2020 年提出的 Sharpness-Aware Minimization (SAM) 为代表的"平坦极小"训练框架，已经成为 CIFAR、ImageNet、ViT、NLP 等多个领域提升泛化性能的标配。SAM 的核心是先沿当前梯度方向走一小步得到对抗扰动权重 $\bm{w}^p = \bm{w} + \rho \nabla \ell(\bm{w})$，再用扰动点的梯度更新原参数 $\bm{w}_{t+1} = \bm{w}_t - \eta \nabla \ell(\bm{w}_t^p)$，强迫优化器找一个邻域内 worst-case loss 也小的解，从而趋向于平坦的极小值。

**现有痛点**：作者把 SAM 和 vanilla GD 同时放到 Beale 函数（一个经典优化测试函数）上跑，GD 顺利收敛到全局最小，而 SAM 卡死在鞍点不动了；Kaddour 等人 2022 年也曾报告 SAM 在某些设置下表现异常。这种"卡鞍点"的现象在深度学习高度非线性的 loss landscape 上代价更大——网络训练里鞍点数量远超极小值，如果 SAM 真的会被它们捕获，那它在大规模实验中仍然 work 就需要新的解释。

**核心矛盾**：sharpness 的几何动机鼓励 SAM 朝"邻域内 loss 都小"的方向走，但这恰好和"沿不稳定流形快速逃离鞍点"的需求冲突——鞍点附近邻域内 loss 起伏小、worst-case 不大，于是 SAM 觉得"挺平坦"，便停下了脚步；而 GD 因为只看当前点梯度，会被鞍点周围微小的扰动顺势带出去。

**本文目标**：(1) 在确定性动力学下，找到 SAM 把鞍点当吸引子的精确条件；(2) 在随机扩散下，定量比较 SAM 与 SGD 的鞍点逃逸速度；(3) 解释 momentum 与 batch size 为何能缓解这种不稳定，进而成为 SAM 成功的隐藏关键。

**切入角度**：用动力系统的 qualitative theory 把 SAM 在鞍点附近的轨迹拆成 Case-I/II/III 三类，再用 Lambda Lemma 论证 Case-III 必然在两个吸引域之间来回震荡。随后切到 Fokker-Planck 框架，用 Fisher 信息矩阵近似把 SAM 的扩散张量写出来，把鞍点逃逸问题变成均方位移问题。

**核心 idea**：扰动 $\bm{w}^p$ 可能掉进相邻极小值的吸引域，使 SAM 的更新方向在鞍点附近反复倒戈；而当 $\rho|\lambda_j|$ 足够大时，原本沿 Hessian 负特征方向的逃逸力 $\lambda_j$ 被 $\rho\lambda_j^2$ 反向覆盖，鞍点几何上从"双曲不稳定点"变为"稳定吸引子"。

## 方法详解

### 整体框架
本文不是提一个新算法，而是给 SAM 做一次完整的"稳定性病理报告"。论文分四步推进：(a) 用 Lambda Lemma + 三 Case 几何分析说明 SAM 为什么在鞍点周围会出现梯度震荡；(b) 用 Hessian 特征值给出鞍点变吸引子的解析条件（Theorem 1）；(c) 用 Fokker-Planck 方程导出 SAM 的扩散公式并证明逃逸更慢（Theorem 2 + Corollary 1）；(d) 把 momentum 和 batch size 写进扩散公式，给出"为何调大 momentum、调小 batch 能救 SAM"的精确量化（Theorem 3）。最终把 CIFAR-10/100 的实验作为这套理论的实证 closure。

### 关键设计

**1. Lambda Lemma 三 Case 几何分析 + 鞍点吸引子条件（Theorem 1）：给出 SAM 卡鞍点的 closed-form 临界值**

要解释 SAM 为什么会在鞍点不动，先得把它在鞍点附近的轨迹看清楚。考虑梯度流 $d\bm{w}/dt = -\nabla \ell(\bm{w})$ 与两个相邻极小值 $\bm{s}_1, \bm{s}_2$ 之间的 index-1 鞍点 $\bm{d}$，按 $\bm{w}_t$ 距 $\bm{d}$ 的远近分三类：Case-I 远离 $\bm{d}$ 与稳定流形 $W^s(\bm{d})$，此时 $-\nabla\ell(\bm{w}_t^p) \sim -\nabla\ell(\bm{w}_t)$，SAM 和 GD 行为一致；Case-II 在 $\bm{d}$ 附近但仍在 $A(\bm{s}_1)$ 内，按 Lambda Lemma 轨迹沿不稳定流形 $W^u(\bm{d})$ 走；Case-III 才是病灶——一旦落入 $\bm{d}$ 的 $\rho$-邻域，扰动 $\bm{w}_t^p$ 可能跨过吸引域边界进入 $A(\bm{s}_2)$，于是 $-\nabla\ell(\bm{w}_t^p)$ 指向 $\bm{s}_2$、下一步又被拉回 $A(\bm{s}_1)$，形成沿 $W^u(\bm{d})$ 的左右震荡。把这个震荡解析化的关键在于：SAM 的扰动 $\rho\nabla\ell$ 会把动力系统的 Hessian 项从 $\Lambda$ 改写成 $\Lambda + \rho\Lambda^2$，而二次项 $\rho\lambda^2$ 恒正、能把原本的负特征值反向。于是 Theorem 1 给出干净的临界条件：对 index-1 鞍点、负特征值 $\lambda_1<0$，只要 $\rho > -1/\lambda_1$（等价于 $\lambda_1 + \rho\lambda_1^2 > 0$），鞍点几何上就从"双曲不稳定点"升级成 SAM 动力学下的吸引子。Beale 函数与 $f(x,y)=x^2-y^2$ 的数值实验恰好同时复现 Case-III 与该条件。

**2. Fokker-Planck 推导 SAM 扩散与逃逸速度比较（Theorem 2 + Corollary 1）：证明 SAM 逃鞍点比 SGD 慢**

确定性分析无法解释 SAM 在真实（noisy SGD）训练里仍然 work，所以要把"扰动如何与 mini-batch 噪声相互作用"写进随机框架。把 SAM 写成 SDE $d\bm{w} = -\nabla\ell(\bm{w}^p)dt + [\eta C(\bm{w}^p)]^{1/2} dW_t$，用 Fisher 信息矩阵把噪声协方差近似为 $C(\bm{w})\approx \frac{1}{B}[H(\bm{w})]^+$，再对损失做二阶 Taylor 展开化成局部二次形式。代入 Fokker-Planck 方程解得 $\bm{w} \sim \mathcal{N}(\bm{d}, Q\,\mathrm{diag}(\bm{\sigma}^2(t))Q^T)$，每个本征方向的方差为

$$\sigma_j^2(t) = \frac{\eta|\lambda_j|}{2B \lambda_j (1+\rho\lambda_j)^2}\Big[1 - \exp\big(-2\lambda_j(1+\rho\lambda_j)^2 t\big)\Big]$$

把 SAM 和 SGD（即 $\rho=0$）的均方位移相减、在 $|\lambda_j|t \ll 1$ 的小时间窗下展开，得到 $\Delta_{SGD} - \Delta_{SAM} = 2\eta t^2 |\lambda_j|^3 \rho / B + \mathcal{O}(B^{-1}\eta t^3 \lambda_j^4)$。这个差值恒正、随 $\rho$ 线性增长，机理也一目了然：扰动 $\rho$ 把分母里的 $(1+\rho\lambda_j)^2$ 撑大，压制了随机扩散，于是噪声本身的逃逸能力被削弱。量纲式 $\rho|\lambda_j|^3/B$ 干净地说明：越大的 $\rho$、越尖锐的 Hessian、越大的 batch，都会把 SAM 与 SGD 的逃逸差距拉大。

**3. 含 momentum + batch size 的 SAM 扩散公式（Theorem 3）：量化两个"工程超参"为何是隐藏支柱**

最后把 momentum $\gamma$ 与 batch size $B$ 同时纳入扩散公式，看它们怎么救鞍点逃逸。加入 momentum 项后均方位移变为

$$\Delta_{SAM} = C_1 \frac{(1-e^{-C_2(1-\gamma)})^2}{(1-\gamma)^3 B} + C_3 \frac{(1-e^{-C_4/(1-\gamma)})}{(1-\gamma)B}$$

其中 $C_1=\eta^2|\lambda_j|/2$、$C_2=\eta/t$、$C_3=\eta|\lambda_j|/[2\lambda_j(1+\rho\lambda_j)^2]$、$C_4=2\lambda_j(1+\rho\lambda_j)^2 t$。$(1-\gamma)$ 在分母里最高到 3 次方，所以 $\gamma\to 1^-$ 会显著抬高 $\Delta_{SAM}$（加快逃逸）；$B$ 只在分母一次方，减小 batch 也能加速逃逸。更关键的是 $\rho$ 会缩小 $C_3$ 这一项，意味着 SAM 用的 $\rho$ 越大，就越需要更大的 $\gamma$ 来把扩散补回到与 SGD 同等水平。这就把 momentum 和 batch size 从"默认或粗调的工程超参"重新定位成 SAM 能 work 的隐藏支柱——CIFAR-10 + ResNet-18 上关掉 BN/aug 只调 $\gamma$ 与 $B$ 即验证：$B=512$ 时 SAM 训练 loss 卡在 1 以上、test acc < 60%；$\gamma=0.9$ 能把 SAM 的 acc 抬升 20%+，而 SGD 只涨 5%，且 $\rho=0.1,\gamma=0.95$ 是最佳组合，印证"$\rho$ 越大越需要 $\gamma$ 补"的预测。

### 损失函数 / 训练策略
本文不引入新损失函数；所用实验损失仍是标准交叉熵（CIFAR-10/100）或均方误差（玩具神经网络）。理论部分的所有推导都基于 (1) 二阶 Taylor 展开围绕鞍点 $\bm{d}$、(2) Fisher 信息矩阵 $\frac{1}{N}\sum_i \nabla\ell_i \nabla\ell_i^T \approx [H]^+$ 这两条近似，附录 A 给出完整证明。

## 实验关键数据

### 主实验（玩具与神经网络验证）

| 实验 | 设置 | GD/SGD 结果 | SAM 结果 | 解读 |
|------|------|-------------|----------|------|
| Beale 函数 | $\eta=10^{-4}$，单鞍点在 $(0,1)$ | 收敛到全局最小 | 卡在鞍点 $(0,1)$ 不动 | 直接验证 Case-III 与 Theorem 1 |
| $f(x,y)=x^2-y^2$ | 初始点 $(-3,-\epsilon), \epsilon=0.01$ | 越过鞍点收敛 | 被吸到鞍点 | Hessian 特征值 $\{2,-2\}$ 满足 $\lambda+\rho\lambda^2 > 0$ |
| Toy NN (Ziyin et al.) | 两层一神经元，$\varphi(x)=x^2$ | 多数种子收敛到全局最小 | 多数种子滞留鞍点区域 | 验证随机扩散下 Theorem 2 |
| 增大 $\rho$ | 同上 toy NN | — | average loss 单调上升 | 与 Corollary 1 一致：$\rho$ 越大越难逃 |

### 消融实验（CIFAR-10/100，ResNet-18，关闭 BN + data augmentation）

| 配置 | CIFAR-10 测试结果 | 说明 |
|------|-------------------|------|
| SAM, $B=512$, $\gamma=0$ | train loss > 1, test acc < 60% | 大 batch + 无 momentum 直接崩 |
| SAM, 减小 $B$ | 训练 loss 下降，test acc 显著上升 | 与 Theorem 3 中 $\Delta_{SAM} \propto 1/B$ 一致 |
| SAM, 调大 $\gamma$ | 训练 loss 下降，test acc 显著上升 | 与 Theorem 3 中 $\Delta_{SAM}$ 随 $(1-\gamma)$ 倒数升高一致 |
| SGD vs SAM, $\gamma=0 \to 0.9$ | SGD +5%, SAM +20%+ | momentum 对 SAM 的边际收益远大于 SGD |
| SAM, $\rho=0.1, \gamma=0.95$ (BN+aug 开) | 95.08%（最佳） | $\rho=0.5, \gamma=0$ 仅 86.20%，Max-Min gap 5.79 |

### 关键发现
- SAM 卡鞍点不是边角案例：Beale、$x^2-y^2$、toy NN、CIFAR 都能复现，且本文用 $\rho>-1/\lambda_1$ 给出了精确的发生条件。
- 鞍点逃逸差距 $\Delta_{SGD}-\Delta_{SAM} \propto \rho|\lambda_j|^3/B$ 是干净的量纲式，能解释为什么大 $\rho$+大 $B$ 的组合最容易在 SAM 上翻车。
- 实践中 momentum 几乎是 SAM 能 work 的必要条件：$\gamma=0.9$ 让 CIFAR-10 上 SAM 的 acc 跳 20+，这是论文第一次把"经验上调 momentum 有效"上升到理论解释。
- $\rho$ 越大越需要更大的 $\gamma$ 才能维持与 SGD 同等的扩散，这给出超参联合搜索的明确方向。
- Beale 函数上 $\cos(\nabla\ell(\bm{w}_t), \nabla\ell(\bm{w}_t^p))$ 在收敛到鞍点后持续在 $-1$ 和 $1$ 之间振荡，给 Case-III-(ii) 的几何描述提供了一张直接可视化证据。
- Table 1 中 ρ=0.5 时 Max-Min gap 达到 5.79%，说明 SAM 越激进越对 momentum 选择敏感，这一对 ρ-γ 的耦合关系完全符合 Theorem 3 的预测。

## 亮点与洞察
- 用 Lambda Lemma + 三 Case 把"SAM 为何卡鞍点"从一个含糊的"扰动越界"直觉落到了几何严格的 Case-III，配上 Beale 函数 cos 震荡的实验图，论证非常干净。
- Theorem 1 的 $\rho > -1/\lambda_1$ 是一个意外简洁的临界条件，意味着 $\rho$ 选大就会把所有 $|\lambda_1| > 1/\rho$ 的鞍点全部吸引化，给"自适应/层归一化 $\rho$"等改进方法提供了直接的设计目标。
- 把 momentum 写进扩散公式给出 $(1-\gamma)^{-3}$ 这种强依赖，是为"为什么 SAM 论文里 momentum 一直默认开 0.9"提供了第一份理论解释；这种"事后才解释清楚的工程超参"在深度学习里非常少见。
- $\Delta_{SGD}-\Delta_{SAM} = 2\eta t^2|\lambda_j|^3\rho/B$ 这种形式的 closed-form 差异公式具有可迁移性，可以直接套到任何带"邻域扰动 + 一次梯度"的变种 SAM 上做稳定性分析。

## 局限与展望
- 全部理论建立在鞍点 $\bm{d}$ 附近的二阶 Taylor 展开 + Fisher 信息近似上，跳出局部邻域后的全局动力学只能靠实验外推。
- 假设 $\bm{w}^p$ 是用一阶近似精确计算的扰动，没有处理 ASAM / GSAM / Lookahead-SAM 这类变种的归一化或二阶修正。
- 实验主要在 CIFAR-10/100 + ResNet-18 这种中等规模上做，没有 ImageNet 或 LLM 训练验证；与 Bartlett et al. 2023 的"bouncing across ravines" / Chen et al. 2023 的 "transient saddle attraction is beneficial" 这两条互补理论也只在 Limitations 一笔带过，缺少统一比较。
- 论文证明了 SAM 逃鞍点慢，但没有给出"SAM 在哪些 setting 下反而因为滞留鞍点附近而提升泛化"的可证伪边界——这正是 Chen et al. 2023 的视角。
- 给出的 $\rho > -1/\lambda_1$ 临界条件依赖准确的局部 Hessian 信息，实际深度网络中精确估计 Hessian 谱是另一项难题，所以"如何在线判断已经进入危险 $\rho$ 区"仍然开放。
- 文章没有给出基于其理论的自动 $\rho$/$\gamma$/$B$ 协同调度算法，把"诊断"与"治疗"之间的工程缝合留给了后续工作。

## 相关工作与启发
- **vs Compagnoni et al. 2023**：他们用 Lipschitz 假设 + 小学习率把噪声当成隐式平滑项；本文用动力系统局部分析把扰动当作驱动轨迹远离不稳定 attractor 的随机激励，两者互补，本文更贴近 edge-of-stability 的 large Hessian 情境。
- **vs Andriushchenko & Flammarion 2022**：他们经验上发现小 batch 显著抬 SAM 性能，本文用 Theorem 3 给出 $\Delta_{SAM} \propto 1/B$ 的解析式，把经验观察上升为机理解释。
- **vs Kaddour et al. 2022**：他们最早报告 SAM 在鞍点附近行为异常，本文是对这一现象的首次系统理论刻画（Theorem 1 + Lambda Lemma 几何）。
- **vs Bartlett et al. 2023 / Chen et al. 2023**：前者从 sharpness 的角度刻画 SAM 沿 ravine 的 bouncing 行为，后者认为 transient 鞍点滞留有助于泛化；本文与它们在结论上互补——指出"永久陷入鞍点"是失败模式，需要 momentum/batch size 来平衡 exploration 与 escape。
- **vs Long & Bartlett 2024 (edge-of-stability)**：edge-of-stability 关心 Hessian 最大特征值在训练中漂到 $2/\eta$ 附近的行为，本文 Theorem 1 的临界 $\rho > -1/\lambda_1$ 与 EoS 在结构上互补，可视作 SAM 版的"边缘临界点"。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Adaptive Sharpness-Aware Minimization with a Polyak-type Step size: A Theory-Grounded Scheduler](adaptive_sharpness-aware_minimization_with_a_polyak-type_step_size_a_theory-grou.md)
- [\[ICML 2025\] Tilted Sharpness-Aware Minimization](../../ICML2025/optimization/tilted_sharpness-aware_minimization.md)
- [\[ICLR 2026\] Minor First, Major Last: A Depth-Induced Implicit Bias of Sharpness-Aware Minimization](../../ICLR2026/optimization/minor_first_major_last_a_depth-induced_implicit_bias_of_sharpness-aware_minimiza.md)
- [\[ICML 2026\] Conflicting Biases at the Edge of Stability: Norm versus Sharpness Regularization](conflicting_biases_at_the_edge_of_stability_norm_versus_sharpness_regularization.md)
- [\[ICML 2026\] Cost-Aware Stopping for Bayesian Optimization](cost-aware_stopping_for_bayesian_optimization.md)

</div>

<!-- RELATED:END -->
