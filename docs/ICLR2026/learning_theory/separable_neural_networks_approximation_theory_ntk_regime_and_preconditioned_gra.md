---
title: >-
  [论文解读] Separable Neural Networks: Approximation Theory, NTK Regime, and Preconditioned Gradient Descent
description: >-
  [ICLR 2026][学习理论][可分神经网络] 这篇论文系统补齐了可分神经网络（SepNN）的理论底座：证明 CP/TT/Tucker 型 SepNN 具备普适逼近能力，推导其无限宽/无限秩与固定秩下的 NTK regime，并提出 SepPGD 用低维可分预条件矩阵调节 NTK 谱，从而加速 INR、PINN 等网格坐标任务中的训练收敛。
tags:
  - "ICLR 2026"
  - "学习理论"
  - "NTK"
  - "优化"
  - "可分神经网络"
  - "普适逼近"
  - "神经切线核"
  - "谱偏置"
  - "预条件梯度下降"
---

# Separable Neural Networks: Approximation Theory, NTK Regime, and Preconditioned Gradient Descent

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=FlcMckO6x5](https://openreview.net/forum?id=FlcMckO6x5)  
**论文**: OpenReview  
**代码**: https://github.com/YisiLuo/SepPGD  
**领域**: 学习理论 / NTK / 优化  
**关键词**: 可分神经网络, 普适逼近, 神经切线核, 谱偏置, 预条件梯度下降  

## 一句话总结
这篇论文系统补齐了可分神经网络（SepNN）的理论底座：证明 CP/TT/Tucker 型 SepNN 具备普适逼近能力，推导其无限宽/无限秩与固定秩下的 NTK regime，并提出 SepPGD 用低维可分预条件矩阵调节 NTK 谱，从而加速 INR、PINN 等网格坐标任务中的训练收敛。

## 研究背景与动机
**领域现状**：SepNN 的基本思路是把多变量函数 $f(x_1,\dots,x_D)$ 拆成若干一维因子函数的组合，例如 CP 形式写作 $f_\Theta(x)=\sum_{r=1}^R\prod_{d=1}^D (f_{\Theta_d}(x_d))_r$。这类结构在隐式神经表示（INR）、可分 PINN、NeRF / tensor factorization 类坐标网络里很自然，因为训练点经常落在规则网格上，可以先分别计算每个维度的因子输出，再通过张量式组合得到全网格预测。

**现有痛点**：实践上，SepNN 的吸引力很明确：对 $D$ 维每维 $n$ 个点的网格，普通 MLP 需要直接处理 $n^D$ 个坐标点，而 SepNN 只需对每个维度的一维坐标做因子网络前向，再复用这些因子输出。问题是，已有工作更多停留在结构设计和应用层面：它到底能不能表示任意连续多变量函数、梯度下降训练时是否也有类似普通宽网络的 NTK 动力学、为什么会在高频细节上收敛慢，这些基础问题都没有被系统回答。

**核心矛盾**：SepNN 的效率来自强可分结构，但可分结构也天然让人担心表达力不足；同时，NTK 谱偏置告诉我们小特征值方向收敛慢，而 SepNN 在 INR/PINN 中恰好常需要拟合图像纹理、曲面细节和 PDE 解的高频成分。也就是说，SepNN 既要保持可分结构带来的 $O(nD)$ 级计算优势，又要避免这种结构在表示和优化上变成瓶颈。

**本文目标**：作者把问题拆成三层：第一层是表示能力，证明 CP、TT、Tucker 三类 SepNN 都能在紧集上任意逼近连续多变量函数；第二层是训练动力学，推导 CP SepNN 的 NTK 形式，并区分无限宽无限秩与无限宽固定秩两种 regime；第三层是算法改进，用 SepNN 的可分结构设计低复杂度预条件梯度下降，让 NTK 谱更均匀，从而缓解谱偏置。

**切入角度**：论文的关键观察是，SepNN 虽然不是普通 MLP，但它的结构与张量分解和一维 MLP 组合高度契合。表示论可以从 Stone-Weierstrass 定理入手，先证明连续可分函数族在 $C(X)$ 中稠密，再用一维 MLP 逼近每个因子；训练论则从 NTK 的参数梯度展开入手，把整体核写成各维因子 MLP 的 NTK 与其他维因子输出的乘积加权和。

**核心 idea**：**用“可分函数族稠密性 + 因子 NTK 分解 + 可分预条件”统一解释 SepNN 为什么能表示、怎么训练、如何加速**。

## 方法详解
### 整体框架
这篇论文不是单纯提出一个新网络，而是围绕 SepNN 建立一条完整理论-算法链：先证明可分结构没有牺牲普适逼近性，再用 NTK 刻画梯度下降下的误差收敛方向，最后顺着 NTK 谱偏置提出 SepPGD。由于核心对象是理论推导和优化器，方法更适合用概念流而不是复杂 pipeline 图来理解。

可以把整篇工作的逻辑写成：给定 CP/TT/Tucker 型 SepNN，作者先把它视为由一维连续函数构成的代数；证明该代数满足 Stone-Weierstrass 的条件后，再把每个一维连续函数替换成 MLP 因子网络。随后，作者对 CP SepNN 加上 $1/\sqrt{R}$ 缩放，展开 $K_\Theta(x,x')=\langle\nabla_\Theta f_\Theta(x),\nabla_\Theta f_\Theta(x')\rangle$，得到整体 NTK 可分解为每个维度的因子 NTK 加权求和。最后，SepPGD 不再构造 $n^D\times n^D$ 的大预条件矩阵，而是在每个维度上构造 $n\times n$ 的因子预条件矩阵 $S_d$，再通过张量 unfold、mode product 和因子输出把预条件后的残差送回对应因子网络。

### 关键设计
**1. Stone-Weierstrass 到 SepNN：先证明可分函数足够密，再交给一维 MLP 拟合**

SepNN 最容易被质疑的点是表达力：既然它把多变量函数拆成一维因子再相乘/相加，是不是只能表示低秩结构？论文的回答是，在秩 $R$ 可以增长时，这种担心在普适逼近层面并不成立。以 CP SepNN 为例，作者定义可分函数类 $\mathcal{A}=\{g(x)=\sum_{r=1}^R\prod_{d=1}^D (g_d(x_d))_r\}$，其中每个 $(g_d)_r$ 是定义在紧集 $X_d$ 上的一维连续函数。

关键不是直接构造某个具体逼近，而是证明 $\mathcal{A}$ 在连续函数空间 $C(X)$ 中稠密。作者逐项验证 Stone-Weierstrass 定理需要的性质：常数函数包含在内；不同点可以通过某个维度上的一维连续函数分开；加法、乘法和数乘在 CP 形式下仍能表示。这样任意连续多变量函数都先被某个可分连续函数逼近。之后再调用非多项式激活 MLP 的普适逼近定理，把每个一维因子 $(g_d)_r$ 替换成 MLP 输出 $(f_{\Theta_d})_r$，并用乘积误差界控制整体误差：若每个因子误差小于 $\delta$，则一项乘积误差可被 $DM^{D-1}\delta$ 控制，总误差随 rank 项数线性累积。TT 和 Tucker 的证明路线类似，只是闭包性质要在张量链式矩阵或 core tensor 上展开。

**2. 两个 NTK regime：无限秩给确定核，固定秩保留随机核**

有了表示能力之后，论文转向训练行为。对 CP SepNN

$$
f_\Theta(x_1,\dots,x_D)=\frac{1}{\sqrt{R}}\sum_{r=1}^R\prod_{d=1}^D (f_{\Theta_d}(x_d))_r,
$$

整体 NTK 可以写成

$$
K_\Theta(x,x')=\frac{1}{R}\sum_{d=1}^D a_d(x)^\top K_{\Theta_d}(x_d,x'_d)a_d(x'),
$$

其中 $K_{\Theta_d}$ 是第 $d$ 个因子 MLP 的多输出 NTK，$a_d(x)$ 则收集除第 $d$ 维以外所有因子输出的逐 rank 乘积。这个公式把 SepNN 的训练动力学拆开了：某一维参数的梯度贡献，不只取决于该维输入的因子 NTK，也取决于其他维度当前输出形成的乘积权重。

在无限宽 $W\to\infty$ 且无限秩 $R\to\infty$ 时，每个因子 MLP 的 NTK 收敛到确定核 $k(x_d,x'_d)$，其他因子输出的 rank 平均也由大数定律收敛到协方差乘积 $\prod_{d'\ne d}c_{d'}(x_{d'},x'_{d'})$，于是整体 SepNN NTK 收敛为确定核

$$
K(x,x')=\sum_{d=1}^D k(x_d,x'_d)\prod_{d'\ne d}c_{d'}(x_{d'},x'_{d'}).
$$

但实践中 rank 往往不会无限大。固定 $R$ 而令宽度 $W\to\infty$ 时，因子 MLP 输出收敛为高斯过程，但 rank 平均无法消掉随机性，所以 NTK 只收敛到一个随机核。这一点很重要：它解释了为什么小 rank SepNN 的训练行为不能完全用一个固定确定核统一描述，也提醒后续理论还需要非渐近或随机 NTK 分析。

**3. 谱偏置诊断：用 NTK 特征值直接解释 SepNN 的慢方向**

论文把 SepNN 放进平方损失和小学习率梯度流中考察。若 NTK 在训练中近似固定，预测向量 $u(t)$ 的动力学满足 $du(t)/dt=-K(u(t)-y)$。对 $K=\sum_i \lambda_i v_iv_i^\top$ 做特征分解后，每个特征方向上的残差满足

$$
v_i^\top(u(t)-y)=\exp(-\lambda_i t)v_i^\top(u(0)-y).
$$

这给了一个非常直接的解释：标签中投影到大特征值方向的成分先被学会，投影到小特征值方向的成分收敛慢。对 INR 来说，这常对应图像细节和高频纹理；对 PINN 来说，则可能对应 PDE 解中更难拟合的局部结构或高频残差。论文的 NTK 实验也验证了这一点：随着宽度和 rank 同时增加，初始化 NTK 趋于确定，训练中 NTK 变化变小；但 NTK 矩阵谱衰减很快，说明 SepNN 同样有明显谱偏置。

这一步的意义在于把“SepNN 训练慢、细节糊”从经验现象变成可分析对象。SepPGD 不是凭空加速，而是瞄准 $K$ 的特征值分布：如果能让有效核 $KS$ 的谱更均匀，小特征值方向的残差就不会长期拖后腿。

**4. SepPGD：把大 NTK 预条件拆成每维小预条件**

已有 NTK 预条件方法会构造一个作用在全部 $n^D$ 个训练样本上的矩阵 $S\in\mathbb{R}^{n^D\times n^D}$，更新形如 $\Theta\leftarrow\Theta-\eta\nabla_\Theta f_\Theta(X)^\top Sr$。这在理论上清楚，但在网格任务里太贵：构造和分解大 NTK 矩阵都随 $n^D$ 爆炸。SepPGD 利用 SepNN 的结构反过来做：对每个维度的因子 MLP 分别计算 $n\times n$ 的 pseudo NTK $K_{\Theta_d}$，对其特征值做调制，得到因子预条件矩阵 $S_d$。

具体调制沿用“压平头部特征值”的思路：若 $K_{\Theta_d}$ 的前 $k$ 个特征对为 $(\lambda_i,v_i)$，设 $g(\lambda_i)=\lambda_k$（$i\le k$），则

$$
S_d=I-\sum_{i=1}^k\left(1-\frac{g(\lambda_i)}{\lambda_i}\right)v_iv_i^\top.
$$

这样做会削弱过大的头部特征值影响，让谱分布更平滑。训练更新时，SepPGD 不是把 $S$ 乘到 $n^D$ 维残差向量上，而是把残差张量 $\mathcal{R}=Z_\Theta-Y$ 沿每个 mode 乘以 $S_d$，再结合其他维度的因子输出构造 $M_d$，最后更新第 $d$ 个因子网络的参数：

$$
\Theta\leftarrow\Theta-\eta\bigoplus_{d=1}^D \nabla_{\Theta_d}\langle f_{\Theta_d}(\hat{x}_d),M_d\rangle.
$$

在二维情况下，作者进一步证明 SepPGD 等价于一个经典 NTK-PGD 更新，只不过大预条件矩阵具有 Kronecker 和结构 $\tilde{S}=S_1\otimes I_n+I_n\otimes S_2$。借助 $(C^\top\otimes A)\mathrm{vec}(B)=\mathrm{vec}(ABC)$，大矩阵乘法可以化成低维矩阵乘法，因此保住了 SepNN 在网格输入上的效率优势。

### 一个完整示例
以一张 $512\times512$ 彩色图像的 INR 表示为例，输入不是整张图像本身，而是二维坐标网格 $(x,y)$，输出是对应像素值。普通 MLP 会把每个像素坐标都当作一个二维样本来前向和反向；SepNN 则把横坐标集合 $\hat{x}$ 和纵坐标集合 $\hat{y}$ 分别送入两个一维因子 MLP，得到两个因子矩阵 $F_x,F_y$，再用 $Z=F_x^\top F_y$ 组合成整张图像的预测。

训练早期，SepNN 往往先学到大块颜色和低频轮廓，纹理、边缘和细线条收敛较慢。按照 NTK 视角，这不是某个像素单独难，而是这些细节主要投影到小特征值方向，残差衰减因子 $\exp(-\lambda_i t)$ 太慢。SepPGD 的处理流程是：先分别在 $\hat{x}$ 和 $\hat{y}$ 上计算两个因子 pseudo NTK，得到 $S_x,S_y$；再把图像残差矩阵 $R=Z-Y$ 分别做横向和纵向的预条件变换，构造给 $F_x$ 和 $F_y$ 的目标矩阵 $M_x,M_y$；最后更新两个因子 MLP。读者可以把它理解成：原来整张图像的残差直接回传，现在先沿每个坐标轴重新加权，让那些被 NTK 谱压住的细节方向更早参与更新。

这个例子也说明了 SepPGD 为什么依赖网格结构。图像、体素曲面和规则 PDE collocation 点都能 reshape 成张量残差，因子预条件矩阵只需在每个坐标轴上作用；如果训练点是完全散乱的非网格样本，SepPGD 仍可用逐点 Einstein product 写出来，但就不再拥有同样干净的 Kronecker / mode product 加速。

### 损失函数 / 训练策略
论文主要讨论平方损失下的全批量优化。对网格训练点 $\hat{x}_1\times\cdots\times\hat{x}_D$，标签被 reshape 成 $D$ 阶张量 $Y\in\mathbb{R}^{n\times\cdots\times n}$，SepNN 输出张量为 $Z_\Theta$，残差张量为 $\mathcal{R}=Z_\Theta-Y$。SepPGD 对数据拟合项使用预条件更新；在 PINN 实验中，作者还把 SepPGD 用到数据项、初值项和边界项，但没有用于 PDE residual 的导数项，因为导数损失的 PGD 扩展需要不同处理。

预条件器构造使用每个因子 MLP 的 pseudo NTK，并做特征分解；训练中通常每 10 次迭代更新一次预条件器，PINN 中为了省计算则在初始化处固定。实验优化器多用 Adam：KRR 中学习率 $0.001$ 并带 weight decay，INR 图像任务中学习率 $0.0001$，曲面和 PINN 中学习率 $0.005$。因子网络常用 SIREN 结构，以适配 INR/PINN 的高频表示需求。

## 实验关键数据

### 主实验
论文实验覆盖四类任务：NTK/KRR 验证、图像 INR 表示与修复、3D 曲面 occupancy 表示、以及 3D PDE 的 separable PINN。作者更强调随时间的收敛曲线，因为 SepNN/SepPGD 的优势主要来自每步复杂度和预条件器构造成本，而不是单纯相同迭代数下的误差。

| 任务 | 指标 | SepNN | SepNN + SepPGD | 对比结论 |
|------|------|-------|----------------|----------|
| 图像表示 Plane | PSNR | 26.48 | 33.30 | 同等迭代下细节恢复明显更好 |
| 3D 曲面表示 Thai statue | IoU | 0.983 | 0.992 | SepPGD 更好捕捉表面纹理与细节 |
| 3D diffusion PINN | MSE | 0.042 | 0.037 | 在 separable PINN 上继续降低误差 |
| 3D Klein-Gordon PINN | MSE | 0.029 | 0.018 | 对 PDE 解的收敛加速更明显 |
| 3D Helmholtz PINN | MSE | 0.018 | 0.016 | 相比 separable PINN 仍有稳定增益 |

| 方法 | 预条件应用复杂度 | 预条件构造对象 | 适用含义 |
|------|------------------|----------------|----------|
| Hessian-based methods | $O(P)$ | 参数空间 Hessian | 参数量 $P$ 大时昂贵 |
| Modified NTK spectrum | $O(n^D)$ | 一个 $n^D\times n^D$ NTK | 理论直接但网格大样本难扩展 |
| Mini-batch NTK PGD | $O(n^D/p)$ | mini-batch NTK | 牺牲全批量结构换可计算性 |
| SepPGD | $O(nD)$ | $D$ 个 $n\times n$ 因子 NTK | 直接利用 SepNN 和网格可分性 |

### 消融实验
| 配置 | 关键指标 | 说明 |
|------|----------|------|
| 图像表示，不同 rank $R=100\sim700$ | SepNN 约 26.36-26.74 PSNR；SepPGD 约 31.59-33.65 PSNR | rank 改变时 SepPGD 仍稳定提升，说明并非只在某个秩上偶然有效 |
| 调制参数 $k=30\sim100$ | $k\ge60$ 时 PSNR 从 33.04 到 34.28 附近 | 需要调制足够多头部特征值，但在合理区间内不太敏感 |
| 预条件更新频率 10 到 1100 | PSNR 从 33.30 缓慢降到 33.06 | NTK 不会剧烈变化，因此不必每步重建预条件器 |
| 激活函数替换 | Sin: 26.48→33.30；Cos: 21.83→30.06；Fourier+ReLU: 30.89→40.49 | SepPGD 对不同因子网络结构都能改善收敛 |
| 高噪声图像表示 | 噪声 0.01 时 26.40→33.26；噪声 0.09 时 25.29→25.06 | 低中噪声下鲁棒，高噪声时会更快拟合噪声 |
| 非网格输入 INR | SepPGD 与 MSK 接近，均优于 SepNN | 非网格时失去 Kronecker/网格效率优势，但预条件仍有一定效果 |

### 关键发现
- **理论验证与实验现象一致**：固定 rank 时即使宽度变大，初始化 NTK 仍保留随机性；宽度和 rank 同时增大时，NTK 才趋向确定核，这和定理/推论的区分相符。
- **SepPGD 的收益主要来自谱调节**：NTK 特征值快速衰减对应慢收敛方向，SepPGD 通过因子预条件矩阵平滑谱分布，在图像纹理、曲面细节、PDE 解细节上都有改善。
- **效率优势依赖网格结构**：在规则坐标网格上，SepPGD 能把大预条件拆成每维小预条件；非网格点上仍可做点式 Einstein product，但效率优势会接近普通 NTK 预条件。
- **加速不是无条件好事**：高噪声图像实验显示，缓解谱偏置会更快追上高频成分，但高频成分也可能是噪声，说明实际应用中可能需要 TV、早停或其他正则。

## 亮点与洞察
- **把 SepNN 从经验结构推进到理论对象**：论文没有只证明某个二变量特殊结构，而是同时覆盖 CP、TT、Tucker 三类可分结构，并给出统一的 Stone-Weierstrass 证明路线。这让 SepNN 不再只是“低秩结构可能够用”的工程假设，而有了普适逼近层面的保证。
- **固定秩随机 NTK 是一个很有启发的 caveat**：很多 NTK 分析默认无限宽后得到确定核，但 SepNN 还有 rank 这个维度。论文指出无限宽不等于确定核，必须让 rank 也进入大数定律，这对理解小 rank SepNN 的随机性和泛化很关键。
- **SepPGD 的设计非常贴合结构**：它不是把通用 PGD 生硬套到 SepNN 上，而是把残差张量、mode product、因子 NTK 和 Kronecker 等价关系连起来。这个设计保留了 SepNN 在网格任务上的低成本优势，也解释了为什么它比全局 NTK 预条件更可扩展。
- **谱偏置解释连接了 INR/PINN 的真实痛点**：图像纹理、3D 曲面细节、PDE 局部结构都常表现为慢收敛成分。SepPGD 用统一的 NTK 谱视角解释这些任务中的加速收益，比单独报告 PSNR 或 MSE 更有迁移价值。

## 局限与展望
- **逼近理论还没有给出速率**：定理证明了任意精度可逼近，但没有说明误差如何随 rank、宽度、维度增长而下降。实际使用 SepNN 时，rank 选多大才足够仍主要靠经验。
- **NTK 理论集中在 CP SepNN**：论文认为 TT/Tucker 可以类似扩展，但核心 NTK regime、谱偏置和 SepPGD 等价性主要针对 CP 形式展开；更复杂张量结构下的精确核形式还需要补完。
- **固定秩训练和泛化仍缺少完整理论**：固定 rank 是实践中更常见的设置，但论文只给出随机 NTK 的极限描述，没有进一步给出非渐近收敛率或泛化误差界。
- **SepPGD 对噪声敏感**：它会加快高频方向拟合，在真实数据带强噪声时可能更快过拟合。未来可以把谱调制和显式正则、早停、自适应 $k$ 选择结合起来。
- **PDE residual 的预条件还没覆盖**：PINN 实验中 SepPGD 没有作用到导数型 PDE residual loss，这正是 PINN 训练难点之一。若能设计面向导数算子的可分预条件，可能更直接改善科学计算任务。

## 相关工作与启发
- **vs CoordX / split MLP INR**: CoordX 等工作强调把坐标网络拆分以加速 INR 训练，本文则解释这类可分结构为什么具备足够表示力，并进一步分析其 NTK 训练动力学。
- **vs Separable PINN**: Separable PINN 已展示了在 PDE 网格 collocation 上的效率，本文补上理论分析，并用 SepPGD 继续改善 separable PINN 的收敛速度。
- **vs 标准 NTK 谱调制方法**: Geifman 等方法直接修改大 NTK 的谱，理论上清楚但对 $n^D$ 网格样本非常重；SepPGD 用每维因子 NTK 近似实现类似谱调节，代价更贴合 SepNN。
- **vs mini-batch inductive gradient adjustment**: IGA/MSK 通过 mini-batch 降低预条件成本，SepPGD 则利用网格可分性在全批量结构下构造低维预条件，两者适合的计算场景不同。
- **对 KAN / 张量分解网络的启发**: 论文附录指出 KAN 每层也可看成一类一维函数线性组合。Stone-Weierstrass + 因子 NTK 的分析路线可能迁移到更一般的可加/可分函数网络。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ 把 SepNN 的普适逼近、NTK regime 和可分预条件串成一体，问题选择很扎实；核心工具本身多来自经典理论，但组合得有价值。
- 实验充分度: ⭐⭐⭐⭐☆ 覆盖 KRR、INR、曲面和 PINN，并有多组敏感性分析；不过主要是数值验证，真实大规模科学应用还不多。
- 写作质量: ⭐⭐⭐⭐☆ 主线清楚，定理和实验互相支撑；部分符号和复杂张量公式对非张量分解背景读者门槛较高。
- 价值: ⭐⭐⭐⭐⭐ 对使用 SepNN/可分 PINN/坐标网络的人很有参考价值，既回答“能不能表示”，也给出“为什么慢”和“怎么加速”的可操作方案。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Scaling Laws and Spectra of Shallow Neural Networks in the Feature Learning Regime](scaling_laws_and_spectra_of_shallow_neural_networks_in_the_feature_learning_regi.md)
- [\[ICLR 2026\] Interactive Learning of Single-Index Models via Stochastic Gradient Descent](interactive_learning_of_single-index_models_via_stochastic_gradient_descent.md)
- [\[ICLR 2026\] Gradient Descent Dynamics of Rank-One Matrix Denoising](gradient_descent_dynamics_of_rank-one_matrix_denoising.md)
- [\[ICLR 2026\] A New Initialization to Control Gradients in Sinusoidal Neural Networks](a_new_initialization_to_control_gradients_in_sinusoidal_neural_networks.md)
- [\[ICLR 2026\] Transformers Trained via Gradient Descent Can Provably Learn a Class of Teacher Models](transformers_trained_via_gradient_descent_can_provably_learn_a_class_of_teacher_.md)

</div>

<!-- RELATED:END -->
