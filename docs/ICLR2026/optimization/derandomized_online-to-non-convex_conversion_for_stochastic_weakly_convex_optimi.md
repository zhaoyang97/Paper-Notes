---
title: >-
  [论文解读] Derandomized Online-to-Non-convex Conversion for Stochastic Weakly Convex Optimization
description: >-
  [ICLR2026][优化/理论][弱凸优化] 这篇论文证明在随机弱凸优化中，可以去掉 O2NC 依赖的随机插值或随机缩放，直接在当前迭代点取随机次梯度，并通过带二次正则的在线增量学习得到最优的 Goldstein stationarity 复杂度，同时导出一个几乎等价于周期性重启动量的 SGDM 变体。
tags:
  - "ICLR2026"
  - "优化/理论"
  - "弱凸优化"
  - "Goldstein stationarity"
  - "O2NC"
  - "SGDM"
  - "动量重启"
---

# Derandomized Online-to-Non-convex Conversion for Stochastic Weakly Convex Optimization

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=Snfqe4lU3G](https://openreview.net/forum?id=Snfqe4lU3G)  
**代码**: 无  
**领域**: 随机弱凸优化 / 非光滑非凸优化  
**关键词**: 弱凸优化, Goldstein stationarity, O2NC, SGDM, 动量重启  

## 一句话总结
这篇论文证明在随机弱凸优化中，可以去掉 O2NC 依赖的随机插值或随机缩放，直接在当前迭代点取随机次梯度，并通过带二次正则的在线增量学习得到最优的 Goldstein stationarity 复杂度，同时导出一个几乎等价于周期性重启动量的 SGDM 变体。

## 研究背景与动机
**领域现状**：非光滑非凸优化里，普通的“梯度范数趋近于零”很难作为可计算目标，因为目标函数可能没有经典梯度，甚至在一般 Lipschitz 非凸情形下寻找近似驻点本身就有不可避免的复杂性障碍。近几年常用的替代收敛概念是 Goldstein $(\delta, \epsilon)$-stationarity：不要求当前点的单个次梯度小，而是看当前点半径 $\delta$ 邻域内 Clarke 次梯度凸包到零的距离是否不超过 $\epsilon$。

**现有痛点**：Cutkosky 等人的 online-to-non-convex conversion（O2NC）给出了非常漂亮的理论框架：把一个在线凸优化算法嵌入随机次梯度迭代，就能以 $O(\delta^{-1}\epsilon^{-3})$ 的最优 oracle 复杂度找到 Goldstein 驻点。问题在于，原始 O2NC 为了证明下降，需要在相邻迭代点之间随机插值，例如在 $v_n=w_{n-1}+s_n\Delta_n$ 处取梯度，$s_n\sim\mathrm{Uniform}([0,1])$。后续 E-O2NC 又引入随机缩放。这些随机化让理论算法能够解释 SGDM/Adam 一类优化器，却又和真实深度学习训练里的 SGDM 有偏差。

**核心矛盾**：如果完全去掉随机化，在一般非光滑非凸函数上会撞上 Jordan 等人的不可能性结果：确定性算法无法获得 dimension-free rate。也就是说，“像实际 SGDM 一样确定性更新”和“保持 O2NC 的维度无关最优保证”在最一般情形下不能同时成立。

**本文目标**：作者把问题限定到一个仍然很宽、但结构更可用的函数类：随机弱凸优化。风险函数 $R$ 是 $\rho$-weakly convex，即 $R(\cdot)+\frac{\rho}{2}\|\cdot\|^2$ 是凸函数。论文要回答三件事：第一，弱凸性能否替代随机插值来给出下降不等式；第二，去随机化后的 O2NC 是否还能达到 $O(\delta^{-1}\epsilon^{-3})$ 复杂度；第三，这个理论算法能否更接近实际 SGDM，并在神经网络训练中有可见效果。

**切入角度**：弱凸性最关键的作用是提供一个二次上界式的“一步变化控制”。对任意 $\gamma\ge\rho$，若 $g_n\in\partial R(w_n)$ 且 $\Delta_n=w_n-w_{n-1}$，弱凸性给出类似
$$
R(w_n)-R(w_{n-1})\le \langle g_n,\Delta_n\rangle+\frac{\gamma}{2}\|\Delta_n\|^2.
$$
这正好把每一步参数更新 $\Delta_n$ 的选择，变成在线学习一串二次损失 $\langle \hat g_n,\Delta\rangle+\frac{\gamma}{2}\|\Delta\|^2$ 的问题。

**核心 idea**：用弱凸性的二次正则项替代 O2NC 的随机插值项，让在线学习器直接学习“下一步增量”，从而在不引入辅助随机化的情况下恢复最优 Goldstein 驻点复杂度，并自然得到 clipped SGDM 或 momentum-restarted SGDM。

## 方法详解

### 整体框架
本文方法叫 Derandomized O2NC（D-O2NC）。输入是初始点 $w_0$、总迭代预算 $N=KT$、弱凸性相关的二次正则系数 $\gamma$、在线学习步长 $\eta$，以及可选的增量半径 $D$。每轮先用当前增量更新参数 $w_n=w_{n-1}+\Delta_n$，然后就在这个真实当前点 $w_n$ 上采样数据 $z_n$ 并计算随机次梯度 $\hat g_n\in\partial \ell(w_n;z_n)$，最后把 $\hat g_n$ 送给在线学习器，更新下一步增量 $\Delta_{n+1}$。

它和原始 O2NC 的最大差别是：没有 $v_n=w_{n-1}+s_n\Delta_n$ 这个随机中间点，也没有指数随机缩放。理论分析不再靠随机插值把函数差分和路径积分接起来，而是靠弱凸性把一步函数变化压到二次 online loss 上。输出仍沿用 O2NC 的分块平均思想：把 $N$ 次迭代分成 $K$ 个长度为 $T$ 的 block，每个 block 输出平均点 $\bar w^{(k)}=\frac{1}{T}\sum_{t=1}^T w_t^{(k)}$，最终从这些平均点中均匀随机选一个。

整体流程可以概括为四步：先维护增量 $\Delta_n$，再用它更新真实参数点 $w_n$；在 $w_n$ 处查询随机次梯度；把这个次梯度看成在线二次损失的一部分；用 OGD 类在线算法决定下一轮的增量。在线学习器有两个版本：Option-I 是带球约束的 clipped OGD，理论上直接给 Goldstein stationarity；Option-II 是无约束但每 $T$ 步重启一次的 OGD，实践上更像真实 SGDM，并用新的 regularized stationarity 来分析。

### 关键设计
**1. 弱凸二次化：用 $\gamma\|\Delta\|^2/2$ 替代随机插值**

原始 O2NC 在 $w_{n-1}$ 和 $w_n$ 之间随机取点，本质上是为了让随机次梯度能够代表沿路径的平均变化；本文发现，在弱凸函数上可以完全绕开这一步。因为 $R$ 是 $\rho$-weakly convex，只要取 $\gamma\ge\rho$，就在当前点 $w_n$ 的次梯度处有
$$
R(w_n)-R(w_{n-1})\le \langle g_n,\Delta_n\rangle+\frac{\gamma}{2}\|\Delta_n\|^2.
$$
这个式子把“优化目标函数下降”转译成“在线地选择增量，让线性项加二次项的累计损失小”。随机次梯度 oracle 满足 $\mathbb E[\hat g_n\mid\mathcal F_{n-1}]=g_n\in\partial R(w_n)$，所以在线学习器看到的 $\hat g_n$ 可以作为这个二次损失的随机估计。

这个设计的好处不是简单地少了一个随机数，而是把 O2NC 的理论依赖点换了：原来靠随机路径采样获得维度无关的 Goldstein 收敛，现在靠弱凸结构获得一个自然的 quadratic loss。也正因为这个替换只对弱凸类成立，论文能避开一般 Lipschitz 非凸函数上的确定性不可能性结果。

**2. Clipped OGD：保守增量带来直接的 Goldstein 保证**

第一个版本用投影 OGD 更新增量：
$$
\Delta_{n+1}=\mathrm{clip}_D\big((1-\eta\gamma)\Delta_n-\eta\hat g_n\big).
$$
这里的 $D$ 是增量半径，保证每一步更新不会太大。若定义 $m_n=-\gamma\Delta_n$、$\beta=\eta\gamma$，这个更新可以重写成
$$
w_n=w_{n-1}-\gamma^{-1}m_n,
\qquad
m_{n+1}=\mathrm{clip}_D\big((1-\beta)m_n+\beta\hat g_n\big),
$$
所以它就是一个 clipped SGDM：$m_n$ 是带动量的搜索方向，$\gamma^{-1}$ 类似学习率，$\beta$ 类似动量混合系数。

这个版本的关键是用小增量保证 block 内所有 $w_t^{(k)}$ 都落在平均点 $\bar w^{(k)}$ 的 $\delta$ 邻域里。当 $TD\le\delta$ 时，block 内次梯度的平均值属于 $\partial_\delta R(\bar w^{(k)})$ 的凸包，在线 regret bound 又能控制这个平均次梯度的范数。最后得到的核心界是
$$
\mathbb E\left[\frac{1}{K}\sum_{k=1}^K \mathrm{dist}(0,\partial_\delta R(\bar w^{(k)}))\right]
\le
\frac{\eta G^2}{D}+\left(\gamma T+\frac{2}{\eta}\right)\frac{D}{T}+\frac{G}{\sqrt T}+\frac{\Delta R_0}{DKT}.
$$
合理设定 $T,K,\gamma,\eta,D$ 后，复杂度是
$$
O\left(\delta^{-1}\epsilon^{-3}+\rho^3\delta^2+\delta^{-1}\right),
$$
主导项仍是 O2NC 的最优 $O(\delta^{-1}\epsilon^{-3})$。

**3. 周期重启 OGD：把理论算法推近标准 SGDM**

Clipped OGD 的理论很干净，但显式限制 $\Delta_n$ 的半径在实践中偏保守，也不像常见的 SGDM 实现。第二个版本去掉投影，改用
$$
\Delta_{n+1}=(1-\eta\gamma)\Delta_n-\eta\hat g_n,
$$
但每经过一个长度为 $T$ 的 block，就把下一轮增量重置为 $0$。换成动量变量 $m_n=-\gamma\Delta_n$ 后，更新为
$$
w_n=w_{n-1}-\gamma^{-1}m_n,
\qquad
m_{n+1}=\big((1-\beta)m_n+\beta\hat g_n\big)\mathbf 1\{\mathrm{mod}(n+1,T)\ne 1\}.
$$
这几乎就是标准 SGDM，只多了一个周期性动量清零。

重启的作用可以从两个角度理解。实践上，它让算法在远离驻点时可以用无约束动量更积极地走，而不是被小球投影卡住；理论上，它保证每个 block 的初始增量为零，从而在线 regret 的初始项可控。因为不再有显式 $D$ 约束，作者引入了 $(\mu,\epsilon)$-regularized stationarity：
$$
\|\partial f(w)\|_{+\mu}=\inf_{V\subseteq\mathbb R^d}\left\{\mathrm{dist}(0,\partial_V f)+\mu\sup_{v\in V}\|v-w\|^2\right\}.
$$
这个指标同时惩罚“邻域内次梯度凸包离零多远”和“这个邻域离当前点多远”。它不是另起炉灶，因为论文证明了它和 Goldstein stationarity 可以互相转换：$(\mu,\epsilon)$-regularized stationary 蕴含 $(\sqrt{\epsilon/\mu},\epsilon)$-stationary。

**4. 分块平均输出：把在线 regret 变成非光滑驻点证书**

O2NC 类算法的输出不是最后一个 iterate，而是 block 平均点。这一点在本文里仍然很重要。对第 $k$ 个 block，记 $\bar g^{(k)}=\frac{1}{T}\sum_{t=1}^T g_t^{(k)}$。在线学习器保证二次损失 regret 小，弱凸下降式把累计 regret 和 $R(w_T^{(k)})-R(w_0^{(k)})$ 连接起来；跨 block 求和后，函数值下降项 telescoping，只剩初始 gap $\Delta R_0$。

对于 clipped 版本，$\|\Delta_t\|\le D$ 进一步说明 block 内点离 $\bar w^{(k)}$ 不超过 $TD$，于是 $\bar g^{(k)}$ 可以视为 Goldstein $\delta$-subdifferential 里的一个候选元素。对于重启版本，作者不要求所有点都被硬塞进固定半径球，而是把 $\max_t\|w_t^{(k)}-\bar w^{(k)}\|^2$ 放进 regularized stationarity 的惩罚项里。这就是两条理论线的分叉：一个直接控制 $\partial_\delta R$，一个先控制 $\|\partial R\|_{+\mu}$ 再转回 Goldstein。

### 一个完整示例
可以把一次 block 看成长度为 $T$ 的“小训练阶段”。开始时 $\Delta_1=0$，因此第一步从 $w_0$ 出发查询随机次梯度 $\hat g_1$。如果使用周期重启版本，在线学习器会给出 $\Delta_2=-\eta\hat g_1$ 的缩放形式；第二步参数移动到 $w_1=w_0+\Delta_1$ 或下一真实迭代点后，继续在当前点直接取 $\hat g_2$，并把新梯度和旧增量按 $(1-\eta\gamma)$ 衰减后相加。这样重复 $T$ 步，得到一串真实训练轨迹 $w_1,\ldots,w_T$。

在这个 block 结束时，算法不输出 $w_T$，而是输出平均点 $\bar w=\frac{1}{T}\sum_t w_t$ 作为候选。直观上，如果这个 block 里的平均次梯度仍然很大，那么在线学习器本可以选择一个反方向的固定增量来获得更小二次损失，从而和 regret bound 矛盾；因此平均次梯度必须小。若 block 内点没有离平均点太远，这个小平均次梯度就变成了 Goldstein 意义下的驻点证书。

### 损失函数 / 训练策略
理论部分的“损失”不是神经网络训练 loss 的改造，而是在线学习器内部面对的二次损失
$$
f_n(\Delta)=\langle \hat g_n,\Delta\rangle+\frac{\gamma}{2}\|\Delta\|^2.
$$
Option-I 在半径 $D$ 的球上对 $f_n$ 做 OGD，Corollary 1 给出一组依赖总预算 $N$ 和目标精度 $(\delta,\epsilon)$ 的参数：
$$
T=\lceil(\delta N)^{2/3}\rceil,
\quad
K=\left\lfloor\frac{N}{T}\right\rfloor,
\quad
\gamma=N^{1/3}\delta^{-2/3},
\quad
\eta=\frac{1}{8N},
\quad
D=\delta^{1/3}N^{-2/3}.
$$
在 $N\ge O(\delta^{-1}\epsilon^{-3})+\rho^3\delta^2+\delta^{-1}$ 时，它输出满足 $\mathbb E[\mathrm{dist}(0,\partial_\delta R(\bar w_T))]\le\epsilon$ 的点。

Option-II 使用同一个二次 loss，但不投影，改为周期重启。Corollary 2 选择
$$
T=\lceil N^{4/7}\mu^{-2/7}\rceil,
\quad
K=\left\lfloor\frac{N}{T}\right\rfloor,
\quad
\gamma=N^{3/7}\mu^{2/7},
\quad
\eta=\frac{1}{8N},
$$
并得到 $(\mu,\epsilon)$-regularized stationarity 的复杂度
$$
O\left(\mu^{1/2}\epsilon^{-7/2}+\rho^{7/3}\mu^{-2/3}+\mu^{1/2}\right).
$$
利用 $\mu=\delta^{-2}\epsilon$ 转换到 Goldstein 指标后，主导项仍是 $O(\delta^{-1}\epsilon^{-3})$。

## 实验关键数据

### 主实验
论文用 D-O2NC 的周期重启版本训练 CIFAR-10 图像分类模型，并和标准 SGDM 在相同学习率、动量和 weight decay 下比较。主表报告每个 trial 的最优 iterate 表现，数值显示 D-O2NC 不只是理论上更贴近 SGDM，也在训练速度和测试精度上有小但稳定的收益。

| Backbone | 指标 | SGDM | D-O2NC $(T=20)$ | D-O2NC $(T=50)$ | 主要观察 |
|--------|------|------|----------------|----------------|----------|
| ResNet-101 | Train Loss $(\times 10^{-3})$ | $2.55\pm0.08$ | $0.55\pm0.02$ | $1.38\pm0.01$ | 重启版收敛到更低训练损失 |
| ResNet-101 | Test Accuracy | $93.91\pm0.55$ | $94.64\pm0.07$ | $94.81\pm0.34$ | 最好约提升 $0.90$ 个百分点 |
| ViT | Train Accuracy | $90.16\pm0.51$ | $92.38\pm0.28$ | $91.36\pm0.22$ | ViT 上训练精度更高 |
| ViT | Test Accuracy | $84.26\pm0.41$ | $84.92\pm0.17$ | $86.22\pm0.29$ | 最好约提升 $1.96$ 个百分点 |

作者还在 CIFAR-100 + ResNet-152 上做了补充实验。图 4 的趋势是 D-O2NC 在 train loss、train top-1 accuracy 和 test top-1 accuracy 上都比 SGDM 更快达到较好水平，说明结论不是 CIFAR-10 单一设置下的偶然现象。

### 消融实验
论文最有用的消融是重启频率 $T$、学习率和动量的敏感性。$T$ 太小会频繁清空动量，几乎无法积累有用方向；$T$ 太大又接近标准 SGDM，重启带来的稳定化和探索节奏会变弱。因此性能通常随 $T$ 增大先改善、再下降。

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| $T=2$ | 多数曲线表现较差 | 重启太频繁，动量刚形成就被清空 |
| $T=20$ | ResNet-101 测试精度 $94.64\pm0.07$ | 训练损失最低，收敛很快 |
| $T=50$ | ResNet-101 测试精度 $94.81\pm0.34$，ViT 测试精度 $86.22\pm0.29$ | 主实验中综合最强，尤其 ViT 明显优于 SGDM |
| $T=196$ | 附录显示性能开始回落 | 重启间隔接近一个 epoch，算法更像普通 SGDM |
| 动量/学习率 $(0.95,0.05)$ 与 $(0.9,0.1)$ | D-O2NC 仍保持收敛速度和预测精度优势 | 说明收益不完全依赖主实验的 $0.99$ 动量与 $0.01$ 学习率 |
| Robust phase retrieval | 与 SGDM 表现相当 | 在弱凸复合问题上没有明显退化，但优势不如神经网络实验突出 |

### 关键发现
- 理论主结论是 clipped D-O2NC 在随机弱凸优化中达到 $O(\delta^{-1}\epsilon^{-3})$ 主导复杂度，且弱凸参数 $\rho$ 只出现在非主导项 $\rho^3\delta^2$ 中；当 $\rho$ 可扩展到 $O(\delta^{-1}\epsilon^{-1})$ 量级时，仍不压过最优项。
- 周期重启版本虽然理论界更复杂，但实践形态最接近 SGDM，只比标准 SGDM 多一个每 $T$ 步清空动量的操作。
- CIFAR-10 上，D-O2NC 对 ResNet-101 和 ViT 都提高测试精度；ViT 的提升更明显，说明周期重启可能对 Transformer 类训练动态更有帮助。
- 重启频率是主要超参数。过小的 $T$ 会破坏动量积累，过大的 $T$ 会削弱重启机制，论文实验中 $20$ 到 $50$ 是相对稳健的区间。
- Robust phase retrieval 结果更保守：D-O2NC 与 SGDM 大体相当，这提醒读者不要把神经网络实验里的提升解读成所有弱凸任务上的普遍优势。

## 亮点与洞察
- 最巧妙的理论动作是把随机插值替换成弱凸二次上界。它不是重新发明一个优化器，而是指出“在弱凸类里，O2NC 需要随机化的那部分可以由结构性不等式接管”。
- 论文对 SGDM 的解释更贴近实际训练代码。原始 O2NC/E-O2NC 能解释动量，但要引入随机插值或随机缩放；本文的 Option-II 只需要周期性重启，和工程实现之间的距离明显更短。
- $(\mu,\epsilon)$-regularized stationarity 是一个有用的分析桥。它把“平均梯度小”和“block 内轨迹不能离输出点太远”放进一个量里，使无投影、无显式半径约束的重启算法也能接上 Goldstein stationarity。
- 复杂度里 $\rho$ 的位置很有启发性。弱凸参数没有进入主导项，意味着本文并不是简单把弱凸问题退化成 Moreau envelope 的传统分析，而是在 Goldstein 指标下利用了 O2NC 的最优结构。
- 从优化器设计看，周期性清空动量是一个很便宜的 trick。它不像新 optimizer 那样引入大量状态变量，也不需要额外 oracle，却在实验里给出更快收敛和更好测试精度。

## 局限与展望
- 理论适用范围依赖弱凸性。一般 Lipschitz 非光滑非凸函数上的确定性 dimension-free 不可能性仍然存在，因此本文的去随机化不能直接推广到所有非光滑非凸目标。
- Clipped OGD 版本虽然理论最干净，但需要显式约束增量半径 $D$，并且参数设定依赖目标精度 $(\delta,\epsilon)$ 与总预算 $N$；这和真实训练中调学习率、动量的方式仍有距离。
- 周期重启版本的理论指标先落在 regularized stationarity，再转回 Goldstein stationarity。这个桥接很自然，但相比直接 Goldstein bound，多了一层参数选择和解释成本。
- 实验规模相对初步。CIFAR-10、CIFAR-100、ResNet/ViT 和 robust phase retrieval 能说明方法有效，但还不足以判断在大规模语言模型、扩散模型或强化学习训练中的收益。
- 论文主要比较 SGDM，没有系统比较 AdamW、schedule-free SGD、Lion 等现代训练常用优化器。作者在结论中也把扩展到 Adam 和 schedule-free SGD 分析作为未来方向。

## 相关工作与启发
- **vs 原始 O2NC**: 原始 O2NC 通过随机插值把在线学习和 Goldstein stationarity 连接起来，达到最优 $O(\delta^{-1}\epsilon^{-3})$ 复杂度；本文在弱凸函数上去掉随机插值，仍保持主导复杂度，并且更像真实 SGDM。
- **vs E-O2NC**: E-O2NC 通过指数随机缩放与无约束 OMD 进一步解释标准 SGDM；本文的 Option-II 同样使用无约束增量更新，但不需要随机缩放，也不需要 EMA 输出，工程实现更直接。
- **vs SGD/SGDM 的 Moreau envelope 分析**: 经典弱凸优化通常证明 SGD/SGDM 找到 Moreau envelope 的 $\epsilon$-stationary point，复杂度约为 $O(\epsilon^{-4})$；本文关注 Goldstein stationarity，并给出和 O2NC 一致的 $O(\delta^{-1}\epsilon^{-3})$ 主导项，两类指标不能简单等同。
- **vs INGD / gradient sampling**: INGD 面向非光滑 stationarity，但通常有随机化、维度依赖或难以扩展到随机 oracle 的问题；本文的 D-O2NC 在弱凸假设下保持 deterministic（除数据采样外）、dimension-free 和 stochastic oracle 可用。
- **启发**: 对其他优化器的理论化可以沿着“把 optimizer 的状态更新解释成在线学习器”这条线继续走。若能找到类似弱凸二次上界的结构，也许可以把 Adam、schedule-free SGD 或动量重启策略的经验设计转化成更清晰的 stationarity guarantee。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 去随机化 O2NC 在一般情形下受不可能性限制，本文用弱凸性打开一个可行窗口，理论切入点很清楚。
- 实验充分度: ⭐⭐⭐⭐☆ 有 ResNet、ViT、CIFAR-10/100 和 robust phase retrieval，但规模仍偏 preliminary，缺少更大模型和现代 optimizer 横向比较。
- 写作质量: ⭐⭐⭐⭐☆ 理论结构完整，定理、推论和 SGDM 对应关系讲得清楚；附录证明较长，对非优化背景读者门槛较高。
- 价值: ⭐⭐⭐⭐⭐ 对非光滑非凸优化理论和实际动量优化器之间的连接很有价值，尤其是周期重启 SGDM 这个简单变体值得进一步实证验证。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] High Probability Bounds for Non-Convex Stochastic Optimization with Momentum](high_probability_bounds_for_non-convex_stochastic_optimization_with_momentum.md)
- [\[ICLR 2026\] Fast Frank–Wolfe Algorithms with Adaptive Bregman Step-Size for Weakly Convex Functions](fast_frankwolfe_algorithms_with_adaptive_bregman_step-size_for_weakly_convex_fun.md)
- [\[ICLR 2026\] Non-Convex Federated Optimization under Cost-Aware Client Selection](non-convex_federated_optimization_under_cost-aware_client_selection.md)
- [\[ICLR 2026\] Improving Online-to-Nonconvex Conversion for Smooth Optimization via Double Optimism](improving_online-to-nonconvex_conversion_for_smooth_optimization_via_double_opti.md)
- [\[NeurIPS 2025\] Stochastic Momentum Methods for Non-smooth Non-Convex Finite-Sum Coupled Compositional Optimization](../../NeurIPS2025/optimization/stochastic_momentum_methods_for_non-smooth_non-convex_finite-sum_coupled_composi.md)

</div>

<!-- RELATED:END -->
