---
title: >-
  [论文解读] Corner Gradient Descent
description: >-
  [ICLR2026][优化/理论][随机梯度下降] 本文提出"围道视角"，把带任意线性内存的广义 (S)GD 等价于复平面上的一条围道（响应映射 $\Psi=P/Q$），证明只要让围道在原点处形成一个外角为 $\theta\pi$（$1<\theta<2$）的"尖角"，就能把幂律二次问题上 SGD 的损失收敛率从 $O(t^{-\zeta})$ 加速到 $O(t^{-\theta\zeta})$，并给出了最优加速因子 $\theta_{\max}=\min(2,\nu,\frac{2}{\zeta+1/\nu})$ 的精确相图，最后用有理逼近把理想的无穷内存"尖角算法"压成可落地的有限内存算法，在合成问题和 MNIST 上验证了加速…
tags:
  - "ICLR2026"
  - "优化/理论"
  - "随机梯度下降"
  - "收敛加速"
  - "幂律谱"
  - "复平面围道"
  - "无穷内存优化器"
---

# Corner Gradient Descent

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=nOXCfIdhD9](https://openreview.net/forum?id=nOXCfIdhD9)  
**代码**: 论文随补充材料附 Jupyter notebook（无独立仓库）  
**领域**: optimization  
**关键词**: 随机梯度下降, 收敛加速, 幂律谱, 复平面围道, 无穷内存优化器  

## 一句话总结
本文提出"围道视角"，把带任意线性内存的广义 (S)GD 等价于复平面上的一条围道（响应映射 $\Psi=P/Q$），证明只要让围道在原点处形成一个外角为 $\theta\pi$（$1<\theta<2$）的"尖角"，就能把幂律二次问题上 SGD 的损失收敛率从 $O(t^{-\zeta})$ 加速到 $O(t^{-\theta\zeta})$，并给出了最优加速因子 $\theta_{\max}=\min(2,\nu,\frac{2}{\zeta+1/\nu})$ 的精确相图，最后用有理逼近把理想的无穷内存"尖角算法"压成可落地的有限内存算法，在合成问题和 MNIST 上验证了加速。

## 研究背景与动机
**领域现状**：在病态二次问题上，确定性梯度下降（GD）的损失收敛是幂律 $L_t=O(t^{-\zeta})$。经典结论（Polyak 的 Heavy Ball + Nemirovskiy–Polyak/Brakhage 的非平稳 Jacobi 调度）告诉我们，给 GD 加上动量并用一套随时间变化的"Jacobi 时间表"，可以把指数翻倍，加速到 $O(t^{-2\zeta})$，而且这是非自适应调度能达到的最好结果。

**现有痛点**：现代机器学习用的是小批量随机梯度下降（SGD），而上面这套漂亮的加速在 SGD 里失效。非平稳 Jacobi Heavy Ball 在 SGD 下会因为采样噪声不断累积而**最终发散**。换句话说，确定性世界里"指数翻倍"的加速，搬到随机世界里就崩了，至今没有已知的、能在无穷维问题上把 SGD 加速到 $O(t^{-2\zeta})$ 的算法。

**核心矛盾**：问题的根源是 SGD 里同时存在两股力量——**信号传播**（在无噪声下纯粹地缩小误差）和**噪声传播**（采样噪声在各个时刻注入并被算法放大）。任何想加快信号收敛的手段（比如增大有效学习率、增大动量），往往同时放大噪声，于是收敛曲线在某个时刻被噪声主导而停滞甚至发散。Yarotsky & Velikanov (2024) 已证明：所有**平稳**的有限内存广义 SGD（动量及其各种推广）都跟普通 SGD 共享同一张相图，在信号主导相里都只能给出 $O(t^{-\zeta})$，**无法加速指数**。

**本文目标**：在信号主导相里，构造一个能真正把指数从 $\zeta$ 提升到 $\theta\zeta$（$\theta$ 最高接近 2）的平稳算法，并刻画在采样噪声约束下"加速到底能走多远"的精确边界。

**切入角度**：作者不再从矩阵/递推形式（$\alpha,b,c,D$ 这些算法参数）出发去硬调，而是换一个完全几何的视角——把广义 SGD 的全部"指纹"压缩进一个复平面上的响应映射 $\Psi(\mu)=P(\mu)/Q(\mu)$，它对应一条把单位圆映成的围道 $\gamma$。损失的所有信息都由这条围道的形状决定。这样设计算法就变成了"设计一条围道"，而围道的**局部几何**（尤其是它在原点附近长什么样）直接决定收敛指数。

**核心 idea**：让围道在原点处带一个外角 $\theta\pi$ 的尖角，用"尖角的锐度 $\theta$"这个单一几何参数同时刻画加速和噪声放大，再通过平衡两者解析地求出最优 $\theta$。

## 方法详解

### 整体框架
本文的对象是无穷维 Hilbert 空间上的二次损失 $L(w)=\frac12 w^T H w - w^T q$，其中 Hessian $H$ 有离散正谱 $\lambda_k\downarrow 0$，并满足两条幂律谱条件：容量条件 $\lambda_k=\Lambda k^{-\nu}(1+o(1))$ 和源条件 $\sum_{k:\lambda_k<\lambda}\lambda_k(e_k^Tw^*)^2=Q\lambda^\zeta(1+o(1))$。直观上，$\zeta$ 像"有效条件数的倒数"（越小越难），$\nu$ 像"有效维数的倒数"。

整条工作线是：**（1）把带 $M$ 大小内存的平稳 (S)GD 等价成复平面围道**——通过特征多项式 $\chi(\mu,\lambda)=P(\mu)-\lambda Q(\mu)$ 和响应映射 $\Psi=P/Q$，证明损失演化所需的全部信息都装在 $\Psi$ 里；**（2）在围道空间里设计"尖角"围道**，证明外角 $\theta\pi$ 直接把信号传播子的衰减指数从 $\zeta$ 拉到 $\theta\zeta$，同时把噪声传播子的衰减指数改成 $2-\theta/\nu$；**（3）平衡信号与噪声**，对两个指数取下界求最优 $\theta$，得到加速相图；**（4）有理逼近落地**，因为理想尖角映射是无理的、需要无穷内存，作者用 $z^\theta$ 的快速有理逼近把它压成有限内存的可执行算法。

理论的技术骨架是 SGD 平均损失的"传播子展开"：在 Spectrally-Expressible (SE) 近似（且取 $\tau_2=0$）下，

$$L_t=\tfrac12\Big(V_{t+1}+\sum_{m=1}^{t}\sum_{0<t_1<\dots<t_m<t+1}U_{t+1-t_m}\cdots U_{t_2-t_1}V_{t_1}\Big),$$

其中**信号传播子** $V_t$ 描述无噪声下的误差缩减，**噪声传播子** $U_t$ 描述各时刻注入的采样噪声的扰动。$L_t$ 的渐近指数由 $U_t,V_t$ 两者中**衰减更慢**的那个决定（Theorem 1），这正是"加速被噪声卡住"的数学来源。批量大小 $|B|$ 只通过 $U_t$ 的系数（分母 $|B|$）起作用，$|B|\to\infty$ 时 $U_t\equiv 0$，退化为确定性 GD。

### 关键设计

**1. 围道视角：把广义 (S)GD 编码成一个响应映射 $\Psi$**

痛点是：直接在算法参数 $(\alpha,b,c,D)$ 上调来调去，既看不清"什么决定收敛率"，也难以系统地设计加速。作者用一个留数/围道积分的恒等式把这件事几何化。注意到传播子里反复出现 $(\,1\ 0^T)(\mu-S_\lambda)^{-1}(\cdot)$ 这种量，通过简单计算可得

$$(\,1\ 0^T)(\mu-S_\lambda)^{-1}\binom{-\alpha}{c}=\frac{Q(\mu)}{P(\mu)-\lambda Q(\mu)}=\frac{1}{\Psi(\mu)-\lambda},\qquad \Psi(\mu)=\frac{P(\mu)}{Q(\mu)}.$$

于是噪声传播子可写成只依赖 $\Psi$ 的围道积分 $U_t=\frac{\tau_1}{|B|}\sum_k\lambda_k^2\big|\frac{1}{2\pi i}\oint_{|\mu|=1}\frac{\mu^{t-1}d\mu}{\Psi(\mu)-\lambda_k}\big|^2$，信号传播子 $V_t$ 同理（只是核里多一个 $\frac{\Psi(\mu)}{\mu-1}$ 因子）。关键洞察是：$P$ 可以是任意 $P(1)=0$ 的 $(M{+}1)$ 阶首一多项式、$Q$ 可以是任意 $\le M$ 阶多项式，所以"设计一个平稳带内存 SGD"本质上等价于"设计一个有理函数 $\Psi$"。再把 $\Psi$ 看成把单位圆映成的曲线 $\gamma=\Psi(\{|\mu|=1\})$，稳定性条件（所有 $S_{\lambda_k}$ 严格稳定）等价于 $\gamma$ 不碰谱段 $[0,\lambda_{\max}]$。反过来，给定一条 Jordan 围道 $\gamma$，由 Riemann 映射定理与 Carathéodory 定理，存在唯一满足 $\Psi(\infty)=\infty$、$\Psi(1)=0$ 的共形映射 $\Psi_\gamma$，从而能从围道**反构出算法**。这一步的价值在于：它把一个高维参数调优问题，降维成"在复平面上画一条好曲线"。

**2. 尖角算法：让围道在原点带一个外角 $\theta\pi$ 的尖角，直接把指数从 $\zeta$ 拉到 $\theta\zeta$**

有了围道视角，作者问：什么样的局部几何能加速？信号主导相里的损失系数 $C_L$ 随有效学习率 $\alpha_{\text{eff}}=-(\frac{d\Psi}{d\mu}(1))^{-1}$ 增大而减小，而 $\alpha_{\text{eff}}$ 增大意味着让 $-\frac{d\Psi}{d\mu}(1)$ 尽量小。于是自然的选择是让 $\Psi$ 在 $\mu=1$（对应围道原点）处呈幂律奇性：

$$\Psi(\mu)=-c_\Psi(\mu-1)^\theta(1+o(1)),\qquad \mu\to 1,$$

这对应围道在原点处张开一个外角 $\theta\pi$。由 $-\frac{d\Psi}{d\mu}\sim c\theta(\mu-1)^{\theta-1}$，要让导数在 $\mu=1$ 处趋于 $+0$（即 $\alpha_{\text{eff}}$ 变大）就需要 $\theta>1$；而稳定性（围道不能穿过谱段）又禁止 $\theta>2$，于是 $\theta$ 的可行区间恰是 $[1,2]$。作者的主技术结果（Theorem 3）证明：在幂律谱下，尖角映射给出严格幂律的传播子——**信号** $V_t=C_V t^{-\theta\zeta}(1+o(1))$，**噪声** $U_t=C_U t^{\theta/\nu-2}(1+o(1))$，两个系数 $C_U,C_V$ 都有限且可用 Mittag-Leffler 函数 $E_{\theta,\theta},E_\theta$ 表达。一句话：尖角越锐（$\theta$ 越大），信号衰减越快（好），但噪声衰减越慢（坏）——加速和噪声放大被同一个 $\theta$ 牵住。

**3. 加速相图：平衡信号与噪声，解析求出最优加速因子 $\theta_{\max}$**

既然 $L_t$ 的指数取 $\theta\zeta$ 和 $2-\theta/\nu$ 中更小者，最优 $\theta$ 来自让两者相等：$\theta\zeta=2-\theta/\nu$；同时还要噪声指数 $2-\theta/\nu>1$（否则总噪声 $U_\Sigma=\sum_t U_t=\infty$，损失对任意有限 $|B|$ 都发散）。综合这些约束，作者得到信号主导相（$\nu>1,\,0<\zeta<2-1/\nu$）下可行加速的精确刻画（Theorem 4）：

$$\theta_{\max}=\min\Big(2,\ \nu,\ \frac{2}{\zeta+1/\nu}\Big),$$

并把信号主导相切成三块子相：**(I) 完全加速** $\theta_{\max}=2$（$\nu>2,\,0<\zeta<1-1/\nu$，能逼近确定性 GD 的指数翻倍）；**(II) 信号/噪声平衡** $\theta_{\max}=\frac{2}{\zeta+1/\nu}<2$；**(III) 受 $U_\Sigma$ 有限性所限** $\theta_{\max}=\nu<2$（此时继续增大 $\theta$ 会让总噪声发散）。这张相图的意义在于：它第一次把"SGD 在病态二次问题上到底能加速多少"给出了**封闭表达式**，而不是个别算法的经验上界。

**4. 有限内存逼近：用 $z^\theta$ 的快速有理逼近把无穷内存尖角算法落地**

痛点很现实：理想尖角映射 $\Psi$ 是无理函数，对应的算法需要无穷大内存，不能直接跑。作者借助 Newman (1964) / Gopal & Trefethen (2019) 关于幂函数 $z\mapsto z^\theta$ 可被 $M$ 阶有理函数以 $O(e^{-c\sqrt{M}})$ 误差逼近的经典结果。具体地，先显式写出一个 $\theta$-尖角映射 $\Psi(\mu)=A\big[(\theta-2)\int_0^\infty\frac{e^{-(2-\theta)s}ds}{\mu-1+e^{-s}}-1\big]\frac{\mu-1}{\mu}$，再用步长 $h=l/\sqrt M$ 的（单侧）梯形求积把积分离散成 $M$ 个节点，得到有理函数 $\Psi^{(M)}=P/Q$（$\deg P=M{+}1,\ \deg Q\le M$）。Proposition 2 进一步给出对应的显式 memory-$M$ 算法参数：$D$ 取对角阵 $\mathrm{diag}(1-e^{-h/2},\dots,1-e^{-(M-1/2)h})$，$b=(1,\dots,1)^T$，$c$ 与 $\alpha$ 由 $A,\theta,h$ 的指数公式给出。代价分析很关键：相比普通 SGD，额外开销主要在内存（多存 $MW$ 个标量，$W$ 是权重数），但每步算术运算只多 $O(MW)$；当 $|B|\gg M$（实践常见）时，这点开销相比批量梯度估计的 $|B|W$ 可忽略——也就是说尖角加速几乎是"免费"的。需要诚实指出的一个理论局限：有限内存的平稳算法在 $t\to\infty$ 时仍只能给出 $O(t^{-\zeta})$（这是平稳有限内存算法的固有上限），但凭借 $O(e^{-c\sqrt M})$ 的逼近精度，在实际关心的有限 $t$ 区间内，曲线会贴近理想的 $O(t^{-\theta\zeta})$。

## 实验关键数据

实验用 Proposition 2 的有限内存逼近，内存 $M=5$、间距参数 $l=5$，单卡 RTX 4070 半小时内训完全部模型。

### 合成指标问题
拟合指示函数 $y(x)=\mathbf{1}_{[1/4,3/4]}(x)$，用只训输出层的浅层 ReLU 网络（$N=10^5$），其极限是一个精确线性模型，可解析算出谱指数 $\zeta=1/4,\ \nu=4$，落在子相 I（完全加速，理论 $\theta_{\max}=2$）。实验取 $\theta=1.8$。

| 算法 | 实测损失指数 | 理论值 |
|------|------|------|
| Plain SGD | $-0.247$ | $\zeta=0.25$ |
| Corner SGD ($\theta=1.8$) | $-0.408$ | $\theta\zeta=1.8\times0.25=0.45$ |

普通 SGD 实测指数贴近理论 $\zeta=0.25$；尖角 SGD 把指数推到 $\approx0.41$，接近理论加速值 $0.45$，确认了加速。

### MNIST
单隐层 ReLU 网络（宽度 $H=1000$，one-hot 目标 + 二次逐点损失），大宽度下处于近似线性的 NTK 区。MNIST 经验谱 $\zeta\approx0.25,\ \nu\approx1.3$，落在子相 III（受 $U_\Sigma$ 有限性所限，$\theta_{\max}=\nu$）。

| 配置 | $\lvert B\rvert=1000$ 损失指数 | $\lvert B\rvert=100$ 损失指数 |
|------|------|------|
| Plain SGD | $-0.274$ | $-0.271$ |
| Corner SGD $\theta=1.3$ | $-0.409$ | $-0.411$ |
| Corner SGD $\theta=1.8$ | $-0.414$ | $-0.281$（不稳定）|

### 关键发现
- $\theta=1.3$ 的尖角 SGD 在两种批量下都稳定地把普通 SGD 的指数加速约 $\times1.5$（$0.27\to0.41$），且测试集损失/错误率也同步更快下降，说明加速不是过拟合训练损失。
- $\theta=1.8$ 在 $|B|=1000$ 下损失更低但加速因子不再明显提升，到 $|B|=100$ 时直接变得不稳定——印证了"$\theta$ 越大噪声放大越严重，需要更大批量来压住 $U_\Sigma<1$"的理论。
- 有趣的边界现象：MNIST 是有限维问题，$U_\Sigma$ 总是有限，因此实践中允许用 $\theta>\nu$ 去博取超出理论 $\theta_{\max}=\nu$ 的加速（在 $|B|$ 够大时），这是无穷维理论的边界在有限维松动后的额外空间。

## 亮点与洞察
- **把"调算法"变成"画围道"**：用响应映射 $\Psi=P/Q$ + Riemann 映射，把任意大小线性内存的广义 SGD 一一对应到复平面围道，是一个干净且可复用的统一框架——动量、Heavy Ball 都是它的低阶特例（围道分别是圆、椭圆）。
- **用一个几何参数 $\theta$ 同时编码加速与噪声**：尖角外角 $\theta$ 既决定信号衰减 $t^{-\theta\zeta}$ 又决定噪声衰减 $t^{\theta/\nu-2}$，于是"加速 vs 噪声"的权衡被压缩成对 $\theta$ 的一元优化，最后得到封闭的 $\theta_{\max}=\min(2,\nu,\frac{2}{\zeta+1/\nu})$，这是本文最"啊哈"的地方。
- **无理目标 + 有理逼近的落地范式**：理想算法无穷内存不可执行，但借幂函数 $z^\theta$ 的 $O(e^{-c\sqrt M})$ 有理逼近，$M=5$ 就够在实用区间逼出加速——"理论上的极限对象用逼近论压成工程可用算法"这套思路可迁移到其他需要无穷内存/无理生成函数的优化器设计。

## 局限与展望
- **理论局限（作者承认）**：有限内存逼近的尖角 SGD 在 $t\to\infty$ 时并不真正加速指数（平稳有限内存算法的固有限制），它只在有限 $t$ 区间内贴近 $O(t^{-\theta\zeta})$；真正的渐近加速预期要靠非平稳近似（Yarotsky & Velikanov 2024 启发式给出 $\theta$ 到 $2-1/\nu$，Ferbach et al. 2025 在类似非平稳算法上给了严格证明）。
- **假设条件较强**：主结果依赖 SE 近似且取 $\tau_2=0$，这对平移不变/高斯分布成立，但不是普遍自然的分布族；作者在附录用微扰论 sketch 了 $\tau_2\neq0$ 的推广（声称在大批量下 Theorem 3 仍成立、系数不变），但这部分尚属论证而非完整证明。
- **适用范围**：理论建立在二次损失/线性模型（或大宽度 NTK 近似线性区）上；当模型有强非线性、出现特征学习时，收敛图景会更复杂（如 Bordelon et al. 2024 的 rich training 给出不同的加速因子 $\frac{2}{1+\zeta}$，机制与本文不同）。
- **改进方向**：把围道视角与非平稳调度结合，争取在保留有理实现的同时获得真正的渐近指数加速；以及把谱指数 $\zeta,\nu$ 在线估计出来以自适应选 $\theta$。

## 相关工作与启发
- **vs 非平稳 Jacobi Heavy Ball**：经典 Jacobi 调度在确定性 GD 上把 $O(t^{-\zeta})$ 加速到 $O(t^{-2\zeta})$，但在 SGD 下因噪声累积发散；本文换成**平稳无穷内存**算法，在信号主导相里稳定加速到 $O(t^{-\theta\zeta})$，并精确刻画了噪声约束下的可行上界。
- **vs Yarotsky & Velikanov (2024)**：该工作证明了所有平稳有限内存 SGD 共享同一相图、无法加速指数，本文正是从"有限内存不行→走向无穷内存 + 围道几何"突破，并复用了它的特征多项式与传播子展开。
- **vs Varre & Flammarion (2022)**：他们的非平稳 SGD 修改只在**有限维**问题上实现二次加速；本文针对**无穷维**问题，且给出可有理逼近的实用算法。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用复平面围道 + 尖角几何统一刻画广义 SGD 加速，是全新的视角与封闭相图。
- 实验充分度: ⭐⭐⭐ 合成 + MNIST 验证了加速且与理论吻合，但规模有限、只覆盖近似线性区。
- 写作质量: ⭐⭐⭐⭐ 理论推导严谨、动机层层递进，但数学门槛高，对非优化背景读者不友好。
- 价值: ⭐⭐⭐⭐ 给"SGD 能加速多少"提供了原理性答案，框架与逼近范式有较强可迁移性。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] On the Convergence Direction of Gradient Descent](on_the_convergence_direction_of_gradient_descent.md)
- [\[ICLR 2026\] Fast Data Mixture Optimization via Gradient Descent](fast_data_mixture_optimization_via_gradient_descent.md)
- [\[ICLR 2026\] Adaptive gradient descent on Riemannian manifolds and its applications to Gaussian variational inference](adaptive_gradient_descent_on_riemannian_manifolds_and_its_applications_to_gaussi.md)
- [\[ICLR 2026\] Egalitarian Gradient Descent: A Simple Approach to Accelerated Grokking](egalitarian_gradient_descent_a_simple_approach_to_accelerated_grokking.md)
- [\[ICML 2026\] On the Convergence Rate of LoRA Gradient Descent](../../ICML2026/optimization/on_the_convergence_rate_of_lora_gradient_descent.md)

</div>

<!-- RELATED:END -->
