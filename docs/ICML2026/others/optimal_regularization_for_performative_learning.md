---
title: >-
  [论文解读] Optimal Regularization for Performative Learning
description: >-
  [ICML2026][表演性学习] 在高维岭回归框架下首次系统刻画了"模型部署反过来推动数据分布漂移"（performativity）场景中最优正则强度的标度律：最优 $\lambda$ 与表演性强度 $\bar b$ 成正比，并且在过参数化区域里恰当的正则甚至能利用表演效应反向降低风险。 领域现状：表演性学习（perfor…
tags:
  - "ICML2026"
  - "表演性学习"
  - "岭正则化"
  - "高维统计"
  - "重复风险最小化"
  - "谬误特征"
---

# Optimal Regularization for Performative Learning

**会议**: ICML2026  
**arXiv**: [2510.12249](https://arxiv.org/abs/2510.12249)  
**代码**: https://github.com/totilas/regularization-vs-perf  
**领域**: others（高维学习理论 / 表演性学习 / 岭回归）  
**关键词**: 表演性学习, 岭正则化, 高维统计, 重复风险最小化, 谬误特征

## 一句话总结
在高维岭回归框架下首次系统刻画了"模型部署反过来推动数据分布漂移"（performativity）场景中最优正则强度的标度律：最优 $\lambda$ 与表演性强度 $\bar b$ 成正比，并且在过参数化区域里恰当的正则甚至能利用表演效应反向降低风险。

## 研究背景与动机

**领域现状**：表演性学习（performative learning, Perdomo et al. 2020）研究的是一个反馈回路——部署的模型 $\theta$ 会改变下一次采样到的数据分布 $\mathcal{D}(\theta)$，最典型的例子是策略性用户为了拿到贷款而修改自己的特征。处理方式主要分两条线：一是显式估计表演算子（Miller 2021、Izzo 2022、Cyffers 2024），二是直接做重复风险最小化（RRM）。

**现有痛点**：第一条线只能在低维小例子上跑得动，需要多轮部署反复对齐分布；第二条线 RRM 虽然贴近实际（部署常常只有一次），但分析基本只覆盖强凸损失加上低维场景。一旦进入"特征维度 $p$ 与样本数 $n$ 同阶"的现代过参数化区，已有理论几乎沉默——而过参数化恰恰是双下降、benign overfitting 等深度学习现象的栖息地。

**核心矛盾**：正则化看起来是低成本的应对手段，但在高维下它会鼓励模型依赖谬误特征（spurious features, Bombari & Mondelli 2025），如果表演效应又恰好放大了这些谬误特征，盲目加 $\lambda$ 可能反而把模型推到更糟的方向。因此"加多少正则、朝哪个方向加"在表演性学习里是一个被悬空的问题。

**本文目标**：在高维线性回归中刻画 (i) 总体（population）极限和 (ii) 比例区 $p/n=\kappa>1$ 两种情形下，岭正则化对 RRM 不动点风险的影响，并给出最优 $\lambda^*$ 的封闭形式。

**切入角度**：作者把表演效应建模成"标签里多出一个线性项" $y = x^\top \theta^*_{\text{pop}} + x^\top D\theta + w$，其中 $D=\text{diag}(b,c)$ 把预测特征和谬误特征的表演强度分开建模。这样既能复用 Han & Xu (2023) 那一套高维随机矩阵工具来求确定性等价，又能保留对"哪类特征被表演性放大"的解析控制。

**核心 idea**：把表演效应当成一种已知方向的扰动，证明在高维线性回归中"最优正则随表演强度 $\bar b$ 等比缩放"，并据此给出无需估计 $D$ 的实操选 $\lambda$ 法则。

## 方法详解

整篇论文是理论分析，方法的"管线"就是把 RRM 不动点的风险写成 $\lambda$、$D$、$\Sigma$ 的解析函数，再对 $\lambda$ 极小化。

### 整体框架

设特征 $x\in\mathbb{R}^p$（$p=2d$），前 $d$ 维是真正预测的，后 $d$ 维是谬误特征。地真参数为 $\theta^*_{\text{pop}} = (a^\top, 0)^\top$；表演性矩阵 $D=\text{diag}(b,c)$，$b$ 对应预测特征、$c$ 对应谬误特征；标签生成 $y = x^\top \theta^*_{\text{pop}} + x^\top D\theta + w$，$w\sim\mathcal{N}(0,\sigma^2)$。

RRM 第 $k$ 轮解 $\theta_k = \arg\min_\theta \tfrac{1}{2n}\sum_i \ell(x_i^{(k-1)}, y_i^{(k-1)};\theta) + \tfrac{\lambda}{2}\|\theta\|_2^2$，数据来自上一轮分布 $\mathcal{D}(\theta_{k-1})$。评测风险定义在未被表演性污染的初始分布 $\mathcal{D}(\theta=0)$ 上，超额风险为 $\mathcal{R}(\Sigma,\theta,\theta^*_{\text{pop}}) = \|\Sigma^{1/2}(\theta-\theta^*_{\text{pop}})\|_2^2$。

总体（population）情形下 RRM 收敛到不动点 $\theta^\infty = (I_p + \lambda\Sigma^{-1} - D)^{-1}\theta^*_{\text{pop}}$；过参数化情形下数据有限，需要借助高维随机矩阵理论给出"确定性等价"。

### 关键设计

**1. 总体极限：最优 $\lambda$ 与平均表演强度 $\bar b$ 成正比**

在 $n\to\infty$ 的总体情形下，作者想要的是一个能直接拿来用的选 $\lambda$ 公式。做法是令 $F = D - \lambda\Sigma^{-1}$，把超额风险关于 $F$ 做二阶 Taylor 展开，得到主导项

$$\widetilde{\mathcal{R}}_{\text{pop}}(D,\lambda,\Sigma) = \tfrac{1}{d}\text{Tr}[\text{diag}(b^2)\Sigma_1] - 2\lambda\bar b + \tfrac{\lambda^2}{d}\text{Tr}(S_1),$$

其中 $\bar b = \tfrac{1}{d}\sum_i b_i$ 是预测特征侧的平均表演强度，$S_1$ 是协方差的 Schur 补。这对 $\lambda$ 是显式二次型，一阶条件直接给出极小点 $\lambda^*_{\text{pop}} = \bar b\, d / \text{Tr}(S_1)$。它的实操价值在于：最优正则只取决于表演性的"平均强度" $\bar b$ 和协方差结构，**完全不用逐坐标估计 $D$**——而以往的表演性算法在 $p>100$ 时几乎都要崩在这一步。结论也很直观："富者愈富"式的正反馈（$\bar b>0$）需要更强的正则去压，"自我衰减"式的表演性（$\bar b<0$）反而需要负正则，这给了过参数化里时常冒出的负 $\lambda$ 一个全新的物理解释。

**2. 过参数化区：方向一致的表演性反而能降低风险**

到了 $p/n=\kappa>1$ 的比例区，数据有限，总体公式失效，得换高维随机矩阵的"确定性等价"来刻画。作者在 Han & Xu (2023)、Ildiz et al. (2025) 的高维风险框架上对 RRM 做两步迭代（一阶意义上已足够逼近不动点），得到风险的确定性等价 $\mathcal{R}_{\text{eq}}(\Sigma,\theta^*_{\text{pop}},D,\lambda)$，它依赖一个辅助标量 $\tau$，由定点方程 $\kappa^{-1} - \lambda/\tau = \tfrac{1}{p}\text{Tr}[(\Sigma+\tau I_p)^{-1}\Sigma]$ 确定。把风险展开到 $\bar b,\bar c$ 一阶后，四个辅助函数 $B_1,B_2,C_1,C_2$ 的符号就决定了表演性是帮忙还是添乱。关键结果是 $B_2(\kappa,\sigma)\le 0$：只要表演性放大已有趋势（$\bar b>0$），最优正则下的风险**反而比无表演性时更低**。这与总体情形截然相反，机制是过参数化区方差占主导，表演性把信号往同一方向再叠一遍，相当于偷偷抬高了信噪比。

**3. 噪声水平翻转正则的方向**

同样是加正则，噪声大小不同时该往哪个方向加却相反，这一点本文也讲清楚了。分析 $B_1(\sigma,\kappa)$ 的符号给出一个临界噪声 $\sigma_{B_1}^2(\kappa) = 1/2 - 7\kappa^{-1}/18 + O(\kappa^{-2})$：低噪声（$\sigma < \sigma_{B_1}$）时 $B_1\ge 0$，最优 $\lambda$ 随 $\bar b$ 同向移动；高噪声（$\sigma > \sigma_{B_1}$）时 $B_1\le 0$，方向反转。对谬误特征侧的表演性 $\bar c$，只要 $\kappa\ge 2$ 就总是与 $\bar c$ 反向，但影响被谬误—预测特征相关系数 $\rho^2$ 压得很低，实证上几乎看不见。作者给的贝叶斯式直觉很顺：噪声大时模型本应"回归先验"，所以表演性带来的修正要往反方向走，免得过度自信——这条法则在 LSAC 小样本（$n=100, d=22$）的实验曲线翻转上得到了直接验证。

### 损失函数 / 训练策略
全文使用平方损失 $\ell(x,y;\theta) = (y - x^\top\theta)^2$ 加岭正则 $\tfrac{\lambda}{2}\|\theta\|_2^2$。理论分析无需训练超参；实验里 RRM 跑 4–5 轮足以逼近不动点，与"两步迭代足够"的理论预测一致。

## 实验关键数据

### 主实验：合成数据 + 真实数据集

| 设置 | 数据 | 关键观察 | 与理论一致 |
|------|------|---------|------------|
| 总体区, $d=100, \Sigma=I_p$ | 合成 | $\lambda^*$ 随 $\bar b$ 线性增长；$\bar b<0$ 时 $\lambda^*<0$ | Corollary 4.2 |
| 比例区低噪声 $\kappa=1.1, \sigma=0.2$ | 合成 | $\bar b=0.2$ 比 $\bar b=0$ 时 $\lambda^*$ 更大且风险**更低** | Theorem 5.2 ($B_2\le 0$) |
| 比例区高噪声 $\kappa=1.1, \sigma=0.7$ | 合成 | $\bar b=0.2$ 让 $\lambda^*$ 减小、风险仍降低 | $B_1\le 0$ 翻转 |
| Housing ($n=4000, d=8$) | 真实 | $\lambda^*$ 随 $\bar b$ 增大，风险随 $\bar b$ 变差 | 总体区行为 |
| LSAC ($n=4000, d=22$) | 真实 | 同上 | 总体区行为 |
| LSAC ($n=100, d=22$) | 真实 | $\lambda^*$ 随 $\bar b$ 减小，风险随 $\bar b$ 改善 | 过参数化大噪声预测 |

### 消融实验：替换正则形式

| 配置 | 现象 | 说明 |
|------|------|------|
| Ridge (本文主线) | $\lambda^*\propto\bar b$ | 理论目标对象 |
| Dropout | 同样规律 | 表明关系不依赖具体范数 |
| Lasso | 同样规律 | $\ell_1$ 下定性结论保留 |
| Elastic Net | 同样规律 | 混合正则一致 |
| 神经网络 + GiveMeSomeCredit (Mofakhami 2023) | $\ell_2$ 正则缓解 $\delta$ 增大带来的精度下降，最优 $\lambda$ 随 $\delta$ 增大 | 定性结论外推到非线性模型 |

### 关键发现
- **最重要的结论**：过参数化 + 强信号方向的表演性 ($\bar b>0$) 联手会让最优风险**优于**无表演性的基线，颠覆了"表演性总是坏事"的直觉。
- **谬误特征的表演性 $\bar c$ 影响很小**：因为它在风险展开里乘了 $\rho^2$，实证上几乎看不到；这也间接说明实践中可以放心忽略谬误侧的表演性建模。
- **样本量决定走哪条曲线**：同一数据集（LSAC），$n\gg d$ 时遵循总体区的"加正则"逻辑，$n$ 与 $d$ 接近时切换到过参数化大噪声逻辑——这是一个直接的实操指导。
- **方法对 RRM 收敛速度的要求很弱**：两步迭代足以触达不动点（一阶意义上），与实际中"部署次数极少"的工业场景天然契合。

## 亮点与洞察
- **第一个用高维统计工具系统刻画表演性学习**：把 Han & Xu (2023) 的 deterministic equivalent 框架嫁接到 RRM 不动点，开了一条新分析路径。这套工具显然可以平移到表演性分类、表演性对抗鲁棒性等场景。
- **"最优 $\lambda$ 正比于平均表演强度"的实操法则**：不需要逐坐标估计 $D$，只需要估 $\bar b$（一个标量），这让方法在高维下真正能用——而以往的表演性算法在 $p>100$ 时几乎都要崩。
- **负正则的物理意义被解释**：以前文献里负 $\lambda$ 总被当作过参数化的怪现象，本文给出一个新解释——当表演性"自我衰减"时，负正则正好抵消这个衰减。
- **失败案例的诚实呈现**：作者明确说 $c$（谬误特征表演性）的影响在实验中观测不到，没有强行宣称该项是关键贡献——这种自我克制在 ICML 投稿里值得鼓励。

## 局限与展望
- **线性假设很强**：分析依赖标签关于 $\theta$ 线性、特征高斯（子高斯可扩展），真实表演性反馈大概率是非线性的。神经网络实验只能给定性证据。
- **只考虑标签漂移**：不涵盖特征漂移；作者承认特征漂移可以通过 centering 部分吸收，但策略性分类的真实场景常常涉及特征改变。
- **优化方法没动**：分析的是"加正则"，但没探讨早停、剪枝、数据增强等其它隐式正则手段——这是作者明确点出的未来方向。
- **测试分布的二选一**：评测必须选 $\mathcal{D}(0)$ 或 $\mathcal{D}(\theta)$ 之一，前者忽视社会演化、后者鼓励操纵分布；如何定义"公正"的表演性评测仍是开放问题。

## 相关工作与启发
- **vs Perdomo et al. (2020)**: 他们证明了 RRM 在强凸性下收敛；本文进一步给出最优正则的标度律，并把分析推到了过参数化区。
- **vs Cyffers et al. (2024)**: 他们在分类里给出"表演性最优 ≈ 正则化非表演性最优"的数值证据；本文把这种直觉在回归里证成了定理，且揭示了过参数化下的反向现象。
- **vs Hastie et al. (2022) / Patil et al. (2024)**: 这些工作研究了 ridge 在标准 / OOD 回归里的最优性，包括负 $\lambda$ 的出现条件；本文表明负 $\lambda$ 在总体区也会因为表演性出现，扩展了负正则的成因谱。
- **vs Bombari & Mondelli (2025)**: 他们指出高维正则可能加剧对谬误特征的依赖；本文在表演性场景中验证了"谬误特征侧的最优 $\lambda$ 方向相反"，与他们的担忧形成对话。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次把高维统计工具引入表演性学习，并发现"表演性可改善风险"这一反直觉现象。
- 实验充分度: ⭐⭐⭐⭐ 合成 + Housing + LSAC + 神经网络都跑了，但真实表演性数据仍缺。
- 写作质量: ⭐⭐⭐⭐⭐ 定理与图示一一对应，直觉与公式都讲得很清楚。
- 价值: ⭐⭐⭐⭐ 给出可执行的选 $\lambda$ 法则，对策略性分类、推荐系统的工程师都有借鉴价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Revisiting the Predictability of Performative, Social Events](../../ICML2025/others/revisiting_the_predictability_of_performative_social_events.md)
- [\[ICML 2026\] Guaranteed Optimal Compositional Explanations for Neurons](guaranteed_optimal_compositional_explanations_for_neurons.md)
- [\[ICML 2025\] Cross-regularization: Adaptive Model Complexity through Validation Gradients](../../ICML2025/others/cross-regularization_adaptive_model_complexity_through_validation_gradients.md)
- [\[NeurIPS 2025\] Tight Lower Bounds and Improved Convergence in Performative Prediction](../../NeurIPS2025/others/tight_lower_bounds_and_improved_convergence_in_performative_prediction.md)
- [\[ICML 2026\] Learning Permutation-Invariant Macroscopic Dynamics](learning_permutation-invariant_macroscopic_dynamics.md)

</div>

<!-- RELATED:END -->
