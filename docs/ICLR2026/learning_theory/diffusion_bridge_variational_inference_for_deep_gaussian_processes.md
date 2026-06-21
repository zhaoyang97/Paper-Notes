---
title: >-
  [论文解读] Diffusion Bridge Variational Inference for Deep Gaussian Processes
description: >-
  [ICLR2026][概率方法][扩散桥] 针对深度高斯过程（DGP）诱导变量的后验推断，本文把 DDVI（去噪扩散变分推断）那个"从固定高斯先验出发的逆向扩散"改造成"从一个可学习、依赖数据的初始分布出发的扩散桥"，用 Doob h-变换在保持 Girsanov-ELBO 数学框架不变的前提下缩短推断轨迹，从而在回归、分类、图像重建任务上比 DDVI 收敛更快、后验更准。
tags:
  - "ICLR2026"
  - "概率方法"
  - "变分推断"
  - "深度高斯过程"
  - "扩散桥"
  - "Doob h-变换"
  - "摊销推断"
---

# Diffusion Bridge Variational Inference for Deep Gaussian Processes

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=zyRmy0Ch9a](https://openreview.net/forum?id=zyRmy0Ch9a)  
**代码**: 待确认  
**领域**: 概率方法 / 变分推断 / 深度高斯过程  
**关键词**: 深度高斯过程, 变分推断, 扩散桥, Doob h-变换, 摊销推断

## 一句话总结
针对深度高斯过程（DGP）诱导变量的后验推断，本文把 DDVI（去噪扩散变分推断）那个"从固定高斯先验出发的逆向扩散"改造成"从一个可学习、依赖数据的初始分布出发的扩散桥"，用 Doob h-变换在保持 Girsanov-ELBO 数学框架不变的前提下缩短推断轨迹，从而在回归、分类、图像重建任务上比 DDVI 收敛更快、后验更准。

## 研究背景与动机

**领域现状**：深度高斯过程（Deep Gaussian Process, DGP）把多层高斯过程级联起来，获得了远强于单层 GP 的层次化贝叶斯表达力。但它的后验推断出了名地难——似然非共轭、层间强耦合、为了可扩展性每层又要引入大量诱导变量 $u^{(l)}$（位于诱导输入 $Z^{(l)}$ 处）。主流做法是带诱导点的随机变分推断（SVI），把后验近似成因子化高斯，但这种简单分布往往装不下深层模型那种复杂、多峰的真实后验。

**现有痛点**：最近提出的 DDVI（Denoising Diffusion Variational Inference）换了个思路——把诱导变量的变分后验建模成一个**逆时扩散 SDE 在 $t=1$ 时刻的边缘分布**，用神经网络参数化逆向漂移里的 score 函数，从而表达任意复杂的后验，同时还能继承 Girsanov 定理给出的可解 ELBO。问题在于：DDVI 的逆向扩散**永远从一个固定的、无条件的高斯 $U_0\sim\mathcal N(0,\sigma^2 I)$ 出发**。

**核心矛盾**：诱导变量的真实后验通常离这个固定高斯起点很远，逆时 SDE 就必须走一条又长又绕的轨迹才能到达目标分布。轨迹长 → 推断低效、方差大、收敛慢；而且这个起点不看观测数据，采样是"输入无关"的，谈不上摊销（amortization）与可扩展。

**本文目标**：让逆向扩散的**起点离后验更近**，同时让起点**依赖观测**从而支持摊销推断，并且不破坏 DDVI 那套优雅的逆时 SDE + Girsanov-ELBO 数学机制。

**切入角度**：既然问题出在"起点固定且离后验远"，那就把起点变成一个**可学习、依赖数据的分布**，让 ELBO 的梯度自动把它推向后验；起点动了，整个扩散过程就从普通逆向扩散变成了一个**桥过程（bridge process）**——而扩散桥恰好有成熟的 Doob h-变换刻画。

**核心 idea**：用一个摊销网络 $\mu_\theta(x)$ 给逆向扩散一个**数据相关的起点**，并用 **Doob h-变换**把它重新解释成端点受约束的扩散桥，从而在 DDVI 框架内推出可训练的桥式 ELBO——这就是 Diffusion Bridge Variational Inference（DBVI）。

## 方法详解

### 整体框架
DBVI 要解决的就是"DDVI 起点太烂"这一件事，但解法牵动了三处：起点、动力学、训练目标。整条 pipeline 是：给定一个 mini-batch，先用**摊销网络**根据诱导输入算出一个数据相关的初始分布 $p_0^\theta(U_0\mid x)=\mathcal N(\mu_\theta(x),\sigma^2 I)$；从这个起点出发跑一条**观测条件的逆向桥 SDE**，其漂移用条件 score $s_{\text{cond}}=s_\phi+h$（$h$ 来自 Doob h-变换）；逆向桥在 $t=1$ 的终点 $U_1$ 就是诱导变量的后验样本，喂进 DGP 前向得到 $f^{(L)}$ 算似然；最后用一个**桥式 ELBO**（把 KL 写成 score-matching 形式）联合训练摊销网络 $\mu_\theta$、score 网络 $s_\phi$ 和 DGP 超参 $\gamma$。

关键是，相比 DDVI，DBVI 只改了"起点 + 桥修正项"，整套逆时 SDE、Girsanov-ELBO、SVI 可扩展性原封不动——并且当 $\mu_\theta(x)\equiv 0$（起点退回原点）时，DBVI 的 loss **精确退化为 DDVI 的 loss**，是 DDVI 的严格推广。这是一篇理论/方法论文，核心在模型与推断推导，故不强行画 pipeline 图。

### 关键设计

**1. 摊销的数据相关初始分布：把固定起点变成可学习起点**

DDVI 慢的根因是起点 $\mathcal N(0,\sigma^2 I)$ 离后验远。DBVI 直接把起点的均值变成依赖数据的摊销输出：
$$p_0^\theta(U_0\mid x)=\mathcal N\big(U_0;\ \mu_\theta(x),\ \sigma^2 I\big),$$
其中只有均值 $\mu_\theta(x)$ 由网络给出、方差 $\sigma^2$ 固定。这样起点本身就贴近后验，逆向 SDE 要走的路被显著缩短。更妙的是，因为起点是 ELBO 目标里的可学参数，**ELBO 的梯度会自然地把初始分布推向后验**，训练越久起点越准，推断间隙（inference gap）越小。同时"起点依赖观测"这件事天然引出摊销推断：每个后验样本只需把数据过一遍漂移网络的前向，不必为每个数据集单独优化。

**2. Doob h-变换的扩散桥重述：起点一动，过程就成了桥**

起点从固定改成依赖数据后，过程不再是普通逆向扩散，而是端点受初始分布约束的桥过程。本文用 **Doob h-变换**把这个约束形式化（命题 1）：令
$$h(U_t,t,U_0)=\nabla_{U_t}\log p(U_0\mid U_t),$$
则前向桥的漂移在原漂移 $f(U_t,t)$ 上加一项 $g(t)^2 h$，把路径"掰"向目标端点；逆时桥 SDE 写作
$$dU_t=\big[f(U_t,t)-g(t)^2 s_{\text{cond}}(U_t,t,U_0)\big]dt+g(t)\,dW_t,$$
其中条件 score $s_{\text{cond}}=s(U_t,t,U_0)+h(U_t,t,U_0)$。这一步的意义在于：它给"依赖数据的起点"提供了严格的随机过程语言，说明起点的改变如何同时修正前向与逆向动力学，而不是拍脑袋加个偏置。

**3. 桥边缘的闭式高斯与 score-matching ELBO：让桥能被高效训练**

光有桥过程还不够，得能算 ELBO。命题 2 证明，在线性漂移 + Doob 桥修正下，桥过程在每个 $t$ 的边缘仍是高斯 $p_t(U_{\text{Bri}}\mid x)=\mathcal N(U_{\text{Bri}};m_t,\kappa_t I)$，其均值 $m_t$、方差 $\kappa_t$ 由一对耦合 ODE 决定（初值 $m_0=\mu_\theta(x),\ \kappa_0=\sigma^2$），ODE 里多出一个修正系数 $c(t)$；当 $c(t)\equiv0$ 时就退回 DDVI 的"桥过程 trick"。有了这个闭式边缘，命题 3 把变分逆向桥 $Q_\phi$ 与参考桥之间的路径 KL 写成可解的 **score-matching** 形式，得到每个 mini-batch 的 ELBO：
$$\ell_{\text{DBVI}}=\mathbb E_{Q_\phi}\Big[-\log p_0^\theta(U_1)+\tfrac{N}{B}\log p(y_I\mid f^{(L)})-\tfrac12\!\int_0^1\! g(t)^2\big\|\tfrac{1}{\kappa_t}(U_t-m_t)+s_{\text{cond}}\big\|^2 dt+\log p_{\text{prior}}(U_1)-\mathrm{KL}\big(\mathcal N(\mu_\theta,\sigma^2 I)\,\|\,\mathcal N(m_1,\kappa_1 I)\big)\Big].$$
和 DDVI 相比，本质差异有两点：(i) 起点被摊销均值 $\mu_\theta(x)$ 参数化，诱导出时间相关的参考均值 $m_t$；(ii) loss 里用的是条件 score $s_{\text{cond}}=s_\phi+h$，显式吃进了桥修正。当 $\mu_\theta\equiv0$ 时 $m_t\equiv0$，整个目标精确还原 DDVI，因此 DBVI 是把 DDVI 当特例包含的严格扩展。

**4. 以诱导输入 $Z^{(l)}$ 为摊销输入：解决维度错配与可扩展**

摊销网络 $\mu_\theta$ 理想上应吃整个数据集 $x$ 输出诱导变量参数，但这在内存/计算上不可行；只喂 mini-batch 又会因看不到全局而产生偏差，而且存在根本的**维度错配**——mini-batch 输入是 $[B,d_{\text{in}}]$，第 $l$ 层诱导变量却是 $[M_l,d_{\text{out}}]$，硬把 $x$ 映到 $u^{(l)}$ 要展平高维张量、破坏高效 batching 且随深度恶化。DBVI 的做法是**用每层的诱导输入 $Z^{(l)}\in\mathbb R^{M_l\times d_{\text{in}}}$ 作为摊销器的输入**：稀疏 GP 的直觉里 $Z$ 本就是数据集的代表性特征，且它的形状天然和 $u^{(l)}$ 对齐。于是定义层级网络 $\mu_\theta^{(l)}:\mathbb R^{d_{\text{in}}}\to\mathbb R^{d_{\text{out}}}$ 逐点作用得到 $\mu_\theta^{(l)}(Z^{(l)})\in\mathbb R^{M_l\times d_{\text{out}}}$。$Z^{(l)}$ 在训练中本身也会更新，这套摊销既保留全局数据集结构、又让输出维度自动匹配诱导变量，且无需访问全量数据。

### 损失函数 / 训练策略
训练目标即上文命题 3 的 mini-batch ELBO $\ell_{\text{DBVI}}(\theta,\phi,\gamma)$。算法（Algorithm 1）流程为：先数值积分命题 2 的 ODE 预计算参考桥边缘 $(m_t,\kappa_t)$；每步采 mini-batch $I$，从 $p_0^\theta(\cdot\mid X_I)$ 摊销采起点 $U_0$；用 Euler-Maruyama 离散逆向桥 SDE 走 $K$ 步（每步漂移含 $s_{\text{cond}}=s_\phi+h$）并累加 score-matching 项 $L_t$；终点 $U_1$ 拆成各层诱导变量 $u^{(l)}$，按稀疏 GP 条件逐层前向采 $f^{(l)}$；最后组装 ELBO，用 Adam（学习率 0.01）联合更新 $\theta,\phi,\gamma$。所有模型用 RBF 核、每层 $M=128$ 个诱导点。

## 实验关键数据

### 主实验
覆盖 UCI 回归（10 个数据集，2–5 层 DGP，报告 RMSE/NLL）、图像分类（MNIST/Fashion-MNIST/CIFAR-10，CIFAR-10 用 ResNet-20 特征）、大规模物理分类（SUSY 5.5M、HIGGS 11M，报告 AUC）、Frey Faces 无监督重建。基线为 DSVI、IPVI、SGHMC、DDVI。

图像分类测试准确率（%，节选 3 层/4 层）：

| 数据集 | 方法 | Acc(L=3) | Acc(L=4) |
|--------|------|----------|----------|
| MNIST | DDVI | 98.84 | 99.01 |
| MNIST | **DBVI** | **99.02** | **99.10** |
| Fashion | DDVI | 90.36 | 90.85 |
| Fashion | **DBVI** | **90.53** | **91.07** |
| CIFAR-10 | DDVI | 95.23 | 95.56 |
| CIFAR-10 | **DBVI** | **95.42** | **95.68** |

大规模物理分类 AUC（$M=128$，节选）：

| 数据集 | 方法 | L=2 | L=4 | L=5 |
|--------|------|-----|-----|-----|
| SUSY | DDVI | 0.883 | 0.887 | 0.886 |
| SUSY | **DBVI** | **0.885** | **0.889** | **0.889** |
| HIGGS | DDVI | 0.849 | 0.856 | 0.857 |
| HIGGS | **DBVI** | **0.851** | **0.858** | **0.859** |

Frey Faces 重建（掩盖 75% 像素，括号内为标准差）：

| 方法 | RMSE | NLL | 每迭代耗时 |
|------|------|-----|-----------|
| DDVI | 7.64 (0.20) | 1.17 (0.01) | 0.36s |
| **DBVI** | **7.52 (0.18)** | **1.12 (0.01)** | 0.40s |

### 消融实验
本文最重要的"消融"是退化关系本身：当摊销均值 $\mu_\theta(x)\equiv0$（即 $m_t\equiv0$、$c(t)\equiv0$）时，DBVI 的 ELBO 精确还原 DDVI 的 ELBO。因此 DBVI 相对 DDVI 的全部增益，都可归因于"可学习、数据相关的起点 + Doob 桥修正项 $h$"这一组件——这等价于一个把摊销初始化整体去掉的 w/o 消融。

| 配置 | 等价于 | 效果 |
|------|--------|------|
| 完整 DBVI（$\mu_\theta\neq0$ + 桥修正 $h$）| —— | 各任务一致优于 DDVI |
| 去掉摊销起点（$\mu_\theta\equiv0$）| 退化为 DDVI | 起点远离后验 → 轨迹变长、收敛变慢 |

### 关键发现
- **增益主要来自起点而非更大的网络**：DBVI 每迭代耗时只比 DDVI 略高（如 CIFAR-10 4 层 0.74s vs 0.69s），但收敛更快、终点更准——说明改善来自"缩短扩散路径"而非堆算力。
- **大规模数据上优势更明显**：作者指出在 YearMSD、Airline 这类大数据集上，无条件 DDVI 收敛慢的毛病最严重，而 DBVI 的摊销桥初始化在这里增益最突出。
- **绝对提升偏小但全面一致**：分类准确率、AUC 多在小数点后第二位的提升，幅度不大，但在所有数据集、所有层数、所有任务（回归/分类/重建）上方向一致，且 RMSE/NLL 标准差也更小，后验质量（uncertainty 校准）更好。

## 亮点与洞察
- **"起点对了，路就短了"是个朴素却有效的洞察**：扩散类变分推断的瓶颈常被放在 score 网络表达力上，本文把矛头指向被忽视的"固定起点"，用一个摊销网络就把推断间隙压下来——视角干净。
- **退化保证让方法"只赚不赔"**：DBVI 严格包含 DDVI 为特例（$\mu_\theta\equiv0$），意味着引入摊销起点至多不变、通常变好，理论上不会比 DDVI 差，这种"扩展即下界更紧"的结构很讨喜。
- **Doob h-变换在变分推断里的复用很巧**：把生成式扩散桥里成熟的 h-变换搬到 DGP 的后验推断上，既保住了 Girsanov-ELBO 的可解性，又把"依赖数据的起点"严格落到随机过程语言里，而不是经验性加偏置。
- **用 $Z^{(l)}$ 当摊销输入解决了维度错配**：这是个可迁移的工程 trick——凡是诱导点框架，诱导输入天然是"对齐了维度的数据摘要"，用它当摊销器输入既省内存又免展平。

## 局限与展望
- **绝对增益偏小**：相对 DDVI 多为小数点后第二位的提升，是否值得多引入一个摊销网络与桥修正、在实际部署中性价比如何，文中没有充分讨论。
- **理论保证仍待补全**：作者在结论里把"扩散桥在变分推断中的理论保证（如收敛性/偏差界）"列为未来工作，说明当前主要是经验性改进 + 形式化推导，缺端到端的近似误差刻画。
- **依赖线性漂移假设**：命题 2 的闭式高斯边缘建立在线性前向 SDE 上，更一般的非线性漂移下桥边缘是否还可解、ODE 系统怎么算，没有展开。
- **摊销输入限定为 $Z^{(l)}$**：用诱导输入当摘要虽优雅，但当诱导点本身没学好或层很深时，$Z^{(l)}$ 是否仍是数据集的好代表、会不会把误差逐层传递，值得进一步检验。

## 相关工作与启发
- **vs DDVI**：同一套逆时 SDE + Girsanov-ELBO + SVI 框架，DDVI 从固定无条件高斯起点出发，DBVI 改成可学习、依赖观测的起点并加 Doob 桥修正；DBVI 把 DDVI 当 $\mu_\theta\equiv0$ 的特例严格包含，优势是更短的推断轨迹、更快收敛与摊销可扩展。
- **vs IPVI**：IPVI 用神经网络表示诱导点后验、靠 GAN 式对抗目标训练，优化不稳定、易得有偏后验；DBVI 走变分扩散路线，有可解 ELBO，训练更稳。
- **vs DSVI**：DSVI 是标准均场高斯变分近似，简单但装不下深层多峰后验；DBVI 用扩散桥构造灵活后验，表达力更强。
- **vs SGHMC**：SGHMC 是采样式推断，CIFAR-10 上每迭代耗时高达 8s 量级；DBVI 在可比或更优精度下计算高效得多。
- **vs 扩散桥 / Schrödinger 桥模型**：He 等的一致性扩散桥、Schrödinger 桥多用于生成建模中两端分布间的直接转移；本文把"观测条件桥 + 摊销参数化"迁移到 DGP 的后验推断，是扩散桥思想在变分推断中的一次落地。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把扩散桥 + Doob h-变换引入 DGP 变分推断，并以可学习摊销起点严格推广 DDVI，角度清晰
- 实验充分度: ⭐⭐⭐⭐ 回归/分类/大规模/重建四类任务、2–5 层、四个基线齐全，但绝对增益偏小、缺更细的消融
- 写作质量: ⭐⭐⭐⭐ 三条命题把模型→边缘→ELBO 串得很顺，退化关系交代清楚
- 价值: ⭐⭐⭐⭐ 给扩散式变分推断指出"起点"这一被忽视的优化维度，方法可复用、理论自洽

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Variational Inference for Cyclic Learning](variational_inference_for_cyclic_learning.md)
- [\[ICLR 2026\] Variational Deep Learning via Implicit Regularization](variational_deep_learning_via_implicit_regularization.md)
- [\[ICLR 2026\] A Unification of Discrete, Gaussian, and Simplicial Diffusion](a_unification_of_discrete_gaussian_and_simplicial_diffusion.md)
- [\[ICLR 2026\] Deep FlexQP: Accelerated Nonlinear Programming via Deep Unfolding](deep_flexqp_accelerated_nonlinear_programming_via_deep_unfolding.md)
- [\[ICLR 2026\] Alternating Diffusion for Proximal Sampling with Zeroth Order Queries](alternating_diffusion_for_proximal_sampling_with_zeroth_order_queries.md)

</div>

<!-- RELATED:END -->
