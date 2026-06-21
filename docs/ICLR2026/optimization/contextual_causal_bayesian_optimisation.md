---
title: >-
  [论文解读] Contextual Causal Bayesian Optimisation
description: >-
  [ICLR2026][优化/理论][贝叶斯优化] 本文提出 CoCa-BO，把"在哪些变量上做干预"（因果作用域选择）和"干预到什么值"（上下文贝叶斯优化）统一进同一个搜索过程，用多臂老虎机在所有"可能最优混合策略作用域"（POMPS）之间挑作用域、用高斯过程在每个作用域内部挑干预值，并给出了首个同时覆盖因果 BO 与上下文 BO 的亚线性后悔界。
tags:
  - "ICLR2026"
  - "优化/理论"
  - "贝叶斯优化"
  - "因果推断"
  - "上下文优化"
  - "混合策略作用域"
  - "多臂老虎机"
  - "后悔界"
---

# Contextual Causal Bayesian Optimisation

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=QW0PchhVaD](https://openreview.net/forum?id=QW0PchhVaD)  
**代码**: 有（Pyro 实现，论文承诺公开）  
**领域**: optimization  
**关键词**: 贝叶斯优化, 因果推断, 上下文优化, 混合策略作用域, 多臂老虎机, 后悔界

## 一句话总结
本文提出 CoCa-BO，把"在哪些变量上做干预"（因果作用域选择）和"干预到什么值"（上下文贝叶斯优化）统一进同一个搜索过程，用多臂老虎机在所有"可能最优混合策略作用域"（POMPS）之间挑作用域、用高斯过程在每个作用域内部挑干预值，并给出了首个同时覆盖因果 BO 与上下文 BO 的亚线性后悔界。

## 研究背景与动机

**领域现状**：贝叶斯优化（Bayesian Optimisation, BO）是优化昂贵黑盒函数的主流框架，目标函数常被写成"给可干预变量赋某些值后目标变量的条件期望"。当这些可干预变量之间存在因果关系时，**因果贝叶斯优化（CaBO）** 利用已知的因果图，只在被精心挑选的变量子集上做干预，从而拿到更低的后悔；与之并行的 **上下文贝叶斯优化（CoBO）** 则利用可观测的上下文变量，把干预值写成上下文的函数。

**现有痛点**：两条线各有死穴。CaBO 把所有非干预变量都当作不可观测，**完全丢掉了上下文信息**；CoBO 则被一个**固定的策略作用域**（一组写死的可干预变量 + 上下文变量）束缚住——一旦这个作用域选得不对，它在某些结构因果模型下会产生**可证明是线性的后悔**（regret），也就是根本收敛不到最优。更麻烦的是，可能的策略作用域数量随变量个数**指数增长**，穷举不可行。

**核心矛盾**：因果信息能帮你判断"该在哪些变量上干预"，上下文信息能帮你判断"在给定观测下干预到什么值"，但已有方法只用其中一种，缺一就可能掉进次线性都达不到的线性后悔。本文用一个具体 SCM（图 1）证明：忽略上下文则后悔至少 $T/12$，忽略因果结构则后悔至少 $T/3$，而存在一个同时利用两者的策略能拿到期望奖励 $1/3$。

**本文目标**：设计一个框架，(1) 在多个候选作用域之间联合优化，跳出固定作用域的线性后悔；(2) 把指数级的作用域搜索压缩到可控规模；(3) 让因果图与上下文变量同时参与决策，并给出收敛保证。

**核心 idea**：把"选作用域"和"选干预值"**拼成一个两层搜索**——外层用多臂老虎机在"可能最优混合策略作用域（POMPS）"集合里选作用域，内层用高斯过程贝叶斯优化在选定作用域里选上下文相关的干预值；二者用一个新的、基于 UCB 滑动平均的作用域选择准则衔接起来。

## 方法详解

### 整体框架

CoCa-BO 解决的是一个序贯决策问题：给定一个已知拓扑、但机制未知的结构因果模型（SCM）$M=(\mathbf V,\mathbf U,\mathbf F,P_{\mathbf U})$ 的因果图 $G$，目标变量 $Y\in\mathbf V$、可干预变量集 $\mathbf X^*$，每一步 $t$ 智能体要：(a) 选一个**混合策略作用域** $S_t$ 和一个建立在它之上的策略 $\pi_t$；(b) 环境按干预后的分布 $P_{\mathbf V}^{\pi_t}$ 生成样本；(c) 智能体观测到上下文 $c_t$ 和奖励 $y_t$。目标是最小化累积后悔 $R_T=T\sup_{\pi}\mu(\pi)-\sum_{t=1}^T\mu(\pi_t)$，其中 $\mu(\pi)=\mathbb E_{P_{\mathbf V}^\pi}[Y]$。

整个算法（论文 Alg. 1）跑一个外层循环，里面嵌着三个组件，三者协同把"作用域 + 干预值"一起优化：

1. **POMPS 枚举**：先一次性算出与 $G$ 相容的所有"可能最优混合策略作用域"集合 $S^*[G]$，把上下文集固定为 $\mathbf C^*=\mathbf V\setminus(\mathbf X^*\cup\{Y\})$，保证不会漏掉最优作用域。
2. **多臂老虎机选作用域**：把 $S^*$ 里每个 POMPS 当成一个"臂"，用一个 MAB 策略 $A:S^*\to\mathbb R$ 在每一步挑出当前最值得试的作用域 $S_t$。
3. **每作用域的高斯过程贝叶斯优化**：为每个 POMPS 维护一个独立的上下文 BO 优化器（用 GP 代理 + HEBO 采集），在 $S_t$ 选定、上下文 $c_t$ 观测后，给出该作用域内的干预值 $x_{S_t}$。

每步观测到 $(c_t,x_t,y_t)$ 后，被选中作用域的 BO 优化器和 MAB 的臂价值都被更新；其余作用域保持不变。本文把这套结构抽象成一个更一般的 **多函数贝叶斯优化（multi-function BO）** 设定来做理论分析：$m$ 个臂对应 $m$ 个 POMPS，每个臂有自己的函数 $f_i$、自己的上下文分布 $P_i$、自己的动作空间，奖励 $Y=f_{i_t}(X_t,C_t)+\epsilon_t$。

### 关键设计

**1. 混合策略作用域（MPS）与可能最优作用域（POMPS）：把"在哪些变量上干预"也变成可优化对象**

这是本文统一 CaBO 与 CoBO 的根基。一个**混合策略作用域** $S$ 是一组配对 $(X;\mathbf C_X)$，其中 $X$ 是某个可干预变量、$\mathbf C_X$ 是用来条件化它的上下文变量集；约束是把 $G$ 中进入 $X$ 的边删掉、再从 $\mathbf C_X$ 向 $X$ 连边后得到的图 $G_S$ 仍然无环（Def. 1）。建立在 $S$ 上的**混合策略** $\pi=\{\pi_{X\mid \mathbf C_X}\}$ 则把每个上下文取值映射到一个干预值（Def. 2）。这样一来，CoBO（固定作用域、只优化策略）和 CaBO（只挑干预子集、不要上下文）都成了它的特例。

直接在所有 MPS 上搜代价太大，作者借用 Lee & Bareinboim (2020) 的 **POMPS（possibly-optimal MPS）** 概念把候选收窄：$S$ 是 POMPS，当且仅当存在某个与 $G$ 相容的 SCM 使得 $S$ 严格优于其他所有作用域（Def. 3）。算法只在 $S^*[G]$ 里选。关键的诚实之处在于：**这个集合不能再被进一步裁剪**——对任意 $S\in S^*[G]$，都存在一个相容 SCM 让 $S$ 恰好最优；只给 $G$、不知道真实机制时，砍掉任何 POMPS 都可能丢掉最优策略、从而招致线性后悔。所以"枚举 POMPS"既是降复杂度的手段，也是保证不漏最优的安全边界。

**2. 嵌套式三组件求解：POMPS 枚举 × MAB 臂选择 × 每作用域 GP 贝叶斯优化**

POMPS 集合 $S^*$ 的计算是对所有 MPS 的暴力搜索，复杂度随 $|\mathbf V|$ 指数增长，但这一步高度可并行，实现中被并行化掉。拿到 $S^*$ 后，外层把每个 POMPS 当老虎机的一个臂，目标是找到平均奖励 $\mu_S^*=\sup_{\pi\in\Pi_S}\mu(\pi)$ 最大的那个作用域；内层对每个作用域 $S$ 单独维护一个上下文 GP。具体地，记 $[t]_S=\{i\le t:S_i=S\}$ 为历史上选了 $S$ 的那些步，GP 假设 $y_t=g(x_t,C_t)+\sigma_n\xi_t$，对测试点 $(x,c)$ 给出闭式后验

$$\mu_{\text{post}}(x,c)=k_{[t]_S}^\top(x,c)\big(K_{[t]_S}+\sigma_n^2 I\big)^{-1}y_{[t]_S},$$
$$\sigma_{\text{post}}^2(x,c)=k\big((x,c),(x,c)\big)-k_{[t]_S}^\top(x,c)\big(K_{[t]_S}+\sigma_n^2 I\big)^{-1}k_{[t]_S}(x,c).$$

每个 POMPS 内部用 **HEBO** 作 BO 子程序，因为它能处理上下文、连续/离散混合变量，并对异方差、非平稳鲁棒。后验更新要对 $T\times T$ 核矩阵求逆，因此单步代价 $O(T^3)$。这套"外层离散选臂、内层连续选值"的嵌套，正是把因果（哪些变量）和上下文（条件化什么）拆成两层分别求解的工程落地。

**3. 用 UCB 滑动平均替代因果采集函数来选作用域：让"跨上下文比较"重新变得合法**

这是本文相对 CaBO 最要害的修正。CaBO 选下一个干预子集时，直接比各个 GP 的**采集函数（causal acquisition function, cAF）** 值，直觉是"最优子集 + 最优干预值应当给出最高奖励，采集函数会在低不确定区反映这一点"。但在上下文设定里，采集函数的最优值**本质上依赖当前观测到的上下文**——于是 CaBO 等于在"不同上下文下评估出的采集函数"之间做比较，这种比较**根本不可比**，会让优化在不同作用域间随上下文胡乱切换，造成不稳定甚至发散。

CoCa-BO 改用一个新的选择准则 $\bar U_{it}$：它是每个 POMPS 实例在其历史评估点上 **UCB 值的滑动平均**，再加一个探索项 $\rho_i(n_i)/\sqrt{n_i}$（$n_i$ 是该作用域已被选中的次数）。论文 Lemma 2 证明这个量能**同时从上下界夹住最优策略下目标的经验均值**（误差为可控的加性项），而这些加性项衰减得足够快，使得次优 POMPS 只会被极少次选到（Lemma 4）。换言之，对作用域取"上下文平均后的 UCB"，恰好绕开了"跨上下文比较采集函数"的非法性，把作用域选择重新放回一个可比、可证收敛的尺度上。

**4. 多函数 BO 的双重后悔界：worst-case 与 instance-dependent**

本文在更一般的多函数 BO 设定下给出收敛保证。这里的后悔 $R_T=\sum_{t=1}^T\big(\mu^*-\mathbb E[f_{i_t}(X_t,C_t)\mid f,D_{t-1}]\big)$ 比较的是"始终选上下文平均奖励最大的臂、并打出上下文最优动作"的 oracle 与算法的差距——它既不同于 MAB（那里臂的奖励跨轮恒定），也不同于标准 BO（这里 $f_{i_t}$ 的上下文平均最大值要和别的函数 $f_j$ 的最优值比），因此分析更难。引入第 $i$ 个 GP 在 $n$ 步上的**最大信息增益** $\gamma_i(n)=\max_{|A|=n}\tfrac12\log\det(I+\sigma^{-2}K_{i,A})$ 后，Theorem 1 在温和的核光滑/有界假设（Asm. 1–2）下给出两条高概率界：

$$R_T\le A_2\sqrt{mT}\big(\sqrt{\beta_T\bar\gamma_T}+\rho_T\big)+\|\Delta\|_1,$$

$$R_T\le A_3\Big\{\sqrt{|I^*|T}\big(\sqrt{\beta_T\gamma_T}+\rho_T\big)+(\beta_T\gamma_T+\rho_T^2)\textstyle\sum_{i\notin I^*}\tfrac{1}{\Delta_i}+m(\sqrt{\beta_1}+\rho_T)\Big\}+\|\Delta\|_1.$$

第一条是 **worst-case 界**，与具体函数无关：对指数特征衰减核（如平方指数核）$\gamma_T$ 至多对数级，于是后悔约为 $\sqrt{mT}\max_i\bar d_i$（差对数因子），即 $\sqrt T\,\text{polylog}(T)$，相对线性后悔是质变。第二条是 **instance-dependent 界**：当最优臂集 $I^*$ 几乎必然是单点时，主导项降到 $\sqrt T\max_i\bar d_i$，比 worst-case 改进 $\sqrt m$ 倍（前提是次优间隙 $\Delta_i$ 不趋于 0，否则 $\sum_{i\notin I^*}1/\Delta_i$ 会反过来主导）。两条界里的 $\|\Delta\|_1$ 来自开头 $m$ 轮每个臂各试一次，是 $O(m)$、与 $T$ 无关、可忽略。作者强调这是**首个**不仅覆盖本文框架、连更成熟的 CaBO 设定也一并补上的后悔界。

### 一个完整示例

用论文图 1 的 SCM 走一遍就能看清"为什么必须同时要因果和上下文"。该图中 $U_1,U_2\sim\text{Uniform}([-1,1])$，机制为 $X_1=U_1,\;C=U_1,\;X_2=U_2 e^{-(X_1+C)^2},\;Y=U_2X_2+C$；上下文 $C$ 同时影响 $X_2$ 和 $Y$。

- **只用因果、丢掉上下文**：可选项只有"不干预 / 干预 $X_1$ / 干预 $X_2$"。不干预时 $\mathbb E[Y]=\tfrac13\mathbb E[e^{-4U_1^2}]\le 1/6$；干预 $X_2$ 会让它与 $U_2$ 独立，期望降为 0；把 $X_1$ 固定到 $x$ 则 $\mathbb E[Y]\le\tfrac13\mathbb E[e^{-U_1^2}]<1/4$。于是任何无上下文策略期望都 $<1/4$，累积后悔至少 $T/12$。
- **只用上下文、丢掉因果**：策略退化为 $C\mapsto(x_1,x_2)$ 的确定函数，由于 $\mathbb E[U_2]=\mathbb E[C]=0$ 且 $\pi^{(2)}_2(C)$ 与 $U_2$ 独立，$\mathbb E[Y]=0$，后悔至少 $T/3$。
- **同时用两者**：在作用域 $S_1=\{(X_1;C)\}$ 上取策略 $\pi^{(1)}_{X_1\mid C}(c)=-c$，可算得 $\mathbb E_{\pi^{(1)}}[Y]=\mathbb E[U_2^2]=1/3$——这正是 CoCa-BO 通过 POMPS 搜索能找到、而 CaBO/CoBO 都够不着的最优策略。

这个例子也说明了 CoCa-BO 的运行画面：MAB 在 $\langle X_1\mid C\rangle$ 与 $\langle X_2\mid C\rangle$ 两个 POMPS 之间靠 $\bar U_{it}$ 逐渐把选择频率压向 $\langle X_1\mid C\rangle$，内层 HEBO 同时把该作用域里的干预函数拟合到 $-c$ 附近，整体后悔随步数下降。

## 实验关键数据

实验在 **110 个随机种子、700 次迭代（即 700 次干预）** 下进行，结果取平均、误差棒为 1.96 倍标准误；评价指标用时间归一化累积后悔 $\bar R_T=\tfrac1T\sum_t\hat r_t$（$\hat r_t=\mathbb E_{\pi^*}[Y]-y_t$），并跟踪每个作用域的累计选择频率。对照方法是 CaBO 与 CoBO，分两种配置：配置 I 下基线也能达到最优（考察额外样本代价），配置 II 下基线无法达到最优（验证 CoCa-BO 仍收敛）。

### 主实验

| 对比 / 配置 | 基线能否收敛 | CoCa-BO 表现 | 关键现象 |
|------|------|------|------|
| CaBO vs CoCa-BO，配置 I（PSA 图） | 仅在 SCM 参数特定对齐时勉强收敛 | 后悔显著更小 | CaBO 靠 cAF 选 POMIS，选择高度抖动，拖累收敛 |
| CaBO vs CoCa-BO，配置 II | 不收敛（无法按 Age/BMI 调药） | 收敛、亚线性后悔 | CaBO 边缘化上下文，无法做上下文相关干预 |
| CoBO vs CoCa-BO，配置 I（图 1） | 收敛，且比 CoCa-BO 略快 | 收敛 | 变量少时 CoBO 更快；CoCa-BO 优势在高维 |
| CoBO vs CoCa-BO，配置 II（图 1 原 SCM） | 线性后悔（不收敛） | 收敛到 $\langle X_1;C\rangle$、亚线性 | 验证理论：固定作用域致线性后悔 |

PSA 例子里 6 个变量（PSA、age、BMI、cancer、aspirin、statin）中只有 aspirin、statin 可干预，目标是前列腺特异抗原 PSA；POMIS 集为 $\{\varnothing,\{\text{Aspirin}\},\{\text{Statin}\},\{\text{Aspirin,Statin}\}\}$，而 POMPS 只有一个 $\{(\text{Aspirin};\text{Age,BMI}),(\text{Statin};\text{Age,BMI})\}$。

### 消融 / 鲁棒性

| 配置 | 现象 | 说明 |
|------|------|------|
| 高维环境（附录 A.3） | 优化域维度被压缩约 **22 倍** | 利用因果图编码的独立性裁掉无关维 |
| 噪声 $\sigma(\epsilon_Y)$ 0.01→1.25 | 仍收敛 | 对 $Y$ 的观测噪声鲁棒 |
| 噪声 $\sigma(\epsilon_Y)=6.25$ | 收敛趋势变模糊 | 极端噪声下退化（$Y$ 噪声直接干扰作用域选择） |

### 关键发现
- **作用域选择的稳定性是胜负手**：CaBO 用因果采集函数选作用域，在上下文环境里选择频率剧烈抖动，直接拖垮内层优化器收敛；CoCa-BO 的 UCB 滑动平均把选择稳定下来。
- **CoCa-BO 的优势随变量数放大**：变量少时 CoBO 甚至更快，但变量一多，CoBO 的优化域爆炸，CoCa-BO 靠因果独立性把域压小（约 22×），样本复杂度优势才显现。
- **理论与实验互证**：配置 II 中 CoBO 的归一化后悔曲线保持水平（线性后悔），与"固定作用域必招线性后悔"的分析完全吻合。

## 亮点与洞察
- **把"作用域选择"提升为一等公民**：以往 CaBO 选子集、CoBO 选策略各管一摊，本文用 MPS/POMPS 把二者塞进同一个搜索空间，再用"外层 MAB + 内层 BO"的嵌套结构求解——这个分层视角很可能能迁移到其他"先选结构、再选参数"的优化问题（如神经架构搜索里选监控信号 + 调超参）。
- **诊断并修好了一个具体 bug**：指出 CaBO 在上下文设定下"跨上下文比较采集函数"是非法操作，并用"上下文平均后的 UCB"这一可比量替换，配 Lemma 2/4 给出严格保证；这种"先定位旧方法为何失效、再给可证补丁"的写法很扎实。
- **理论补白**：给出首个同时适用于上下文-因果 BO 和经典 CaBO 的后悔界，且同时有 worst-case 和 instance-dependent 两条，区分了"间隙明显时能再快 $\sqrt m$ 倍"的情形。
- **可复现性意识强**：用 Pyro 实现、自带一套可基准测试的环境，并明确批评了 Functional Causal BO 因关键超参（$N,\alpha_i,c_i$）未指定而难复现。

## 局限与展望
- **POMPS 枚举随 $|\mathbf V|$ 指数增长**：尽管高度可并行，变量规模一大仍是硬瓶颈；论文靠"可并行"缓解而非根除。
- **依赖已知且正确的因果图**：方法假设 $G$ 给定、做精确干预（precise intervention），因果发现被明确排除在范围之外；图错了保证就失效。
- **GP 后验 $O(T^3)$**：长视野下核矩阵求逆代价高，缺乏对大 $T$ 的可扩展近似（如归纳点/稀疏 GP）的讨论。
- **极端噪声退化**：$\sigma(\epsilon_Y)=6.25$ 时收敛已不明朗，说明对目标变量噪声的鲁棒性有限。
- **跨 POMPS 的信息共享未解决**：作者自己指出，如何在不加强假设的前提下让不同 POMPS 之间共享信息（multitask GP 方向），仍是开放问题。

## 相关工作与启发
- **vs Causal BO (Aglietti et al., 2020b)**：CaBO 用因果图 + 因果采集函数搜作用域、把非干预变量当不可观测；本文显式利用上下文、在干预×上下文联合空间搜索，既解更优也修掉了 cAF 在上下文下的不稳定。
- **vs Contextual BO**：CoBO 把干预值条件化在上下文上，但通常把所有可控变量都当可干预、作用域固定，可能根本够不到最优策略；本文通过在 POMPS 上搜索规避这一次优性。
- **vs Functional Causal BO (Gultchin et al., 2023)**：后者用有限 RKHS 展开给 $\mu(\pi)$ 加 GP 先验，但 $N,\alpha_i,c_i$ 未指定、采集函数优化也没讲，难复现；本文不需要这种有限展开、有理论保证且有公开实现。
- **vs Model-Based Causal BO (Sussex et al., 2023)**：后者假设加性噪声 SCM、不缩减搜索空间，后悔界含 $(K\beta_T)^N$ 与 $|\mathbf V|$ 这类可能很大的因子；本文条件更一般、在无未观测混杂时 POMPS 退化为单点，scaling 明显更优。
- **vs Zhang & Bareinboim (2022) / Lattimore et al. (2016)**：这两者也用因果知识做 MAB，但都要求变量域离散有限、且不考虑上下文干预/假设无混杂；本文不要求离散有限域。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次把因果作用域选择与上下文 BO 统一进一个可证框架，并补上首个覆盖 CaBO 的后悔界。
- 实验充分度: ⭐⭐⭐⭐ 两类配置 + PSA 真实结构 + 噪声鲁棒性，110 种子×700 步较扎实；但环境偏小、缺真正大规模系统验证。
- 写作质量: ⭐⭐⭐⭐⭐ 用反例把动机讲透，定义—算法—定理层层递进，对旧方法失效点定位精准。
- 价值: ⭐⭐⭐⭐ 给因果 + 上下文决策提供了清晰的理论框架与可复现实现，但 POMPS 指数枚举与已知图假设限制了即用范围。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Reducing Contextual Stochastic Bilevel Optimization via Structured Function Approximation](reducing_contextual_stochastic_bilevel_optimization_via_structured_function_appr.md)
- [\[ICLR 2026\] Combinatorial Bandit Bayesian Optimization for Tensor Outputs](combinatorial_bandit_bayesian_optimization_for_tensor_outputs.md)
- [\[ICLR 2026\] BoGrape: Bayesian optimization over graphs with shortest-path encoded](bogrape_bayesian_optimization_over_graphs_with_shortest-path_encoded.md)
- [\[ICML 2026\] Efficient Stochastic Optimisation via Sequential Monte Carlo](../../ICML2026/optimization/efficient_stochastic_optimisation_via_sequential_monte_carlo.md)
- [\[ICLR 2026\] Adaptive Acquisition Selection for Bayesian Optimization with Large Language Models](adaptive_acquisition_selection_for_bayesian_optimization_with_large_language_mod.md)

</div>

<!-- RELATED:END -->
