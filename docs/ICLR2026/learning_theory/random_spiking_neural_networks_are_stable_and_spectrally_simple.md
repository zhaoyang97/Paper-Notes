---
title: >-
  [论文解读] Random Spiking Neural Networks are Stable and Spectrally Simple
description: >-
  [ICLR 2026][学习理论][LIF 神经元] 本文把离散时间 LIF 脉冲神经网络（SNN）分类器看成布尔函数的组合，用布尔函数分析证明了随机初始化的宽 SNN「平均意义下是稳定的」——输入扰动到 $O(\sqrt{n})$ 个坐标时输出大概率不变，并由此提出「谱简单性」概念，证明随机 SNN 偏向傅里叶谱集中在低频的简单函数，实验进一步表明训练会让稳定性更强。
tags:
  - "ICLR 2026"
  - "学习理论"
  - "脉冲神经网络"
  - "LIF 神经元"
  - "布尔函数分析"
  - "噪声稳定性"
  - "简单性偏差"
---

# Random Spiking Neural Networks are Stable and Spectrally Simple

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=Ochp5HHp46](https://openreview.net/forum?id=Ochp5HHp46)  
**代码**: 待确认  
**领域**: 学习理论 / 脉冲神经网络  
**关键词**: 脉冲神经网络, LIF 神经元, 布尔函数分析, 噪声稳定性, 简单性偏差

## 一句话总结
本文把离散时间 LIF 脉冲神经网络（SNN）分类器看成布尔函数的组合，用布尔函数分析证明了随机初始化的宽 SNN「平均意义下是稳定的」——输入扰动到 $O(\sqrt{n})$ 个坐标时输出大概率不变，并由此提出「谱简单性」概念，证明随机 SNN 偏向傅里叶谱集中在低频的简单函数，实验进一步表明训练会让稳定性更强。

## 研究背景与动机
**领域现状**：脉冲神经网络靠事件驱动的稀疏脉冲通信，被视作低功耗、可上神经形态硬件（Loihi 2、SpiNNaker 2）的节能替代品。但相比传统人工神经网络（ANN），SNN 的理论基础仍很薄弱：训练算法和硬件实现进展不少，可稳定性、鲁棒性、泛化这些核心性质几乎没人从理论上刻画。

**现有痛点**：「稳定性」在 SNN 里连统一定义都没有——可以指学习算法的算法稳定性，可以指动力系统稳定性，也可以指本文关心的「对输入扰动的敏感度」。已有从动力系统角度做的工作（Ding et al. 2024）只分析单个神经元、且用的是简化的 reset-to-zero 机制，停留在「输出脉冲序列差多少」这一层，没法回答「分类预测会不会变」这个更实际的问题。

**核心矛盾**：SNN 的神经元只在膜电位越过阈值时才发放脉冲，这是一个二值事件，天然适合用布尔函数刻画；但真实的 LIF 模型带「reset-by-subtraction（减阈值复位）」和时间上的自回归衰减，会在不同时间步之间引入复杂的概率依赖，使得经典布尔函数分析没法直接照搬。

**本文目标**：(1) 给出宽 LIF-SNN 分类器在随机初始化下的稳定性定量界；(2) 把这种稳定性翻译成一种「简单性」并形式化；(3) 用实验验证理论、并考察训练的影响。

**切入角度**：作者第一次把布尔函数分析（O'Donnell 的噪声敏感度 / 傅里叶–沃尔什展开框架）引入 SNN，并把研究对象限定在「初始化时的随机网络」——既绕开了尚不成熟的 SNN 训练理论、隔离出模型本身的内禀稳定性，又因为随机网络在 PAC-Bayes 泛化界里能当先验而具有独立价值。

**核心 idea**：把多层 sign-LIF SNN 分类器写成布尔函数的迭代组合，用噪声敏感度量化其稳定性，再通过「噪声稳定 ⟹ 傅里叶谱集中在低频」把稳定性升级为「谱简单性」，从而把 SNN 接入深度网络的简单性偏差叙事。

## 方法详解

### 整体框架
全文是一条「建模 → 单神经元界 → 多层界 → 谱简单性 → 实验」的理论链条，没有可训练 pipeline，因此不画流程图。

输入是 $T$ 个时间步的二值序列 $(x_t)_{t\in[T]}\in(\{-1,1\}^n)^T$，输出是分类标签。作者先把单个 **sign-LIF（sLIF）神经元**定义为一个随时间递归演化的计算单元：膜电位 $u_t=\beta u_{t-1}+w^\top x_t-\frac{\theta}{2}(s_{t-1}+1)$，脉冲 $s_t=\mathrm{sign}(u_t-\theta)$，其中 $\beta\in[0,1]$ 是泄漏系数、$\theta>0$ 是阈值、权重按 $w\sim N(0,I_n/n)$ 随机初始化。多个 sLIF 神经元按层全连接堆叠成 $L$ 层网络，分类器取输出层「脉冲计数最大」的神经元为预测类别：$f^{L,T}=\arg\max_{i\in[n_L]}\sum_{t=1}^T s^{(L)}_{t,i}$。

关键观察是：固定权重后，每个脉冲 $s_t:\{-1,1\}^n\to\{-1,1\}$ 本身就是一个布尔函数，整个分类器是布尔函数的组合。于是稳定性问题被翻译成布尔函数的**噪声敏感度** $\mathrm{NS}_\nu(f)=\Pr_{x,\xi}[f(x)\neq f(x\odot\xi)]$，对随机权重族再取期望得到 ENS（期望噪声敏感度）。最后利用「噪声敏感度低 ⟺ 傅里叶谱集中在低频」这条经典桥梁，把稳定性结论升级为谱简单性。

### 关键设计

**1. 把 SNN 翻译成布尔函数组合：sign-LIF 神经元 + reset-by-subtraction**

要用布尔函数分析，第一步得把连续动力学的 LIF 神经元离散成「输入二值序列 → 输出二值脉冲」的布尔映射。作者采用 sign 激活（$s_t=\mathrm{sign}(u_t-\theta)$ 取值 $\{-1,1\}$）而非经典的 Heaviside，纯粹是为了让后续傅里叶分析更干净（$\{-1,1\}$ 是傅里叶–沃尔什展开的自然定义域）。权重按 $w\sim N(0,I_n/n)$ 初始化保证 $w^\top x=O(1)$，避免「从不发放」或「过度发放」的退化区。

真正的技术难点在 **reset-by-subtraction**：每次发放后膜电位减去 $\theta$（对应递归式里 $-\frac{\theta}{2}(s_{t-1}+1)$ 这一项），而不是简单清零。这使得阈值随过程动态自适应、不同时间步之间产生非平凡的概率依赖，是本文区别于 Ding et al. (2024) 简化模型的核心，也是后面证明里最棘手的地方。为了把主分析做干净，作者把理论限定在 $\beta=1$（无泄漏的 IF）且静态输入（同一样本在 $T$ 步内重复呈现，正是 MNIST/CIFAR 等静态数据集在 SNN 里的常用编码），$\beta\neq1$ 的推广放到附录。

**2. 单神经元稳定性界：随机线性阈值函数的高斯分解（Theorem 1）**

针对「单个神经元在输入扰动下输出会不会翻转」，作者给出第一条定量界。设两条输入序列在每个时间步的相对汉明距离为 $\nu_t=d_H(x_t,y_t)/n$、其均值 $\bar\nu_t$，当 $\max_t\nu_t=O(1/\sqrt n)$ 时，对所有 $t$：

$$\Pr_w\!\big[s_t(x_1,\dots,x_t)\neq s_t(y_1,\dots,y_t)\big]\le C(1+\theta)\,t^2\sqrt{\bar\nu_t}\,\log n,$$

静态输入下还能去掉 $\log n$ 因子。证明思路从 $t=1$ 切入：此时问题退化为两个高斯量 $X=w^\top x_1,\,Y=w^\top y_1$ 的符号是否一致。用经典高斯分解 $Y=\rho X+\sqrt{1-\rho^2}\,Z$（$\rho=1-\nu_1$），把翻转事件写成 $\{X>\theta,Y\le\theta\}\cup\{X\le\theta,Y>\theta\}$，再用相关系数为 $2\nu_1-1$ 的二元高斯 CDF $\Phi_2$ 及尾界，得到 $\Pr[\cdot]\le C_\theta\sqrt{\nu_1}$。$t\ge2$ 时用归纳 + 并集界处理时间依赖，由此带来 $t^2$ 因子；reset 机制让阈值动态变化，是收紧时间依赖的主要障碍。直观结论：宽神经元平均很稳定，扰动到 $O(\sqrt n)$ 个坐标输出才大概率改变。

**3. 多层分类器稳定性界：吸收马尔可夫链 + Chernoff（Theorem 2）**

把单神经元的界推到 $L$ 层分类器才是真正有用的结果。作者跟随 Jonasson et al. (2023) 的思路，把「两条输入在第 $l$ 层产生的脉冲差异」建成一条马尔可夫链 $D^{(l)}_1(x,y)=\frac14\|s^{(l)}_1(x)-s^{(l)}_1(y)\|^2$，它有 $n+1$ 个状态、$0$ 是吸收态。在条件 $D^{(l-1)}_1=\lfloor\nu_1 n\rfloor$ 下，下一层差异 $D^{(l)}_1\sim\mathrm{Bin}(n,p_{\nu_1})$ 且 $p_{\nu_1}\le C_\theta\sqrt{\nu_1}$（由 Theorem 1 给出），于是差异被 $\mathrm{Bin}(n,C_\theta\sqrt{\nu_1})$ 随机控制，逐层用 Chernoff 界即得：当 $\nu=O(1/\sqrt n)$、$n$ 足够大时，

$$\Pr_W\!\big[f^{L,T}((x_t))\neq f^{L,T}((y_t))\big]\le n_L T^4 C(1+\theta)\,\nu^{\frac{1}{2^{2L+1}}}\log^{3/2}n+(L-1)e^{-c\,\nu^{\frac{1}{2^{2L-1}}}n}.$$

界随层数 $L$、延迟 $T$、阈值 $\theta$ 增大而变松，与布尔函数组合「深度越大敏感度越高」的一般规律一致；作者指出 $\theta$ 依赖和 $\log^{3/2}n$ 多半是证明的 artifact，$L,T$ 依赖是否内禀留作开放问题（实验里考察）。

**4. 谱简单性：从噪声稳定到傅里叶谱集中（Corollary 1）**

这是把「稳定」升级为「简单」的关键一跃。任意 $f:\{-1,1\}^n\to\mathbb R$ 有唯一傅里叶–沃尔什展开 $f(x)=\sum_{S\subseteq[n]}\hat f(S)\chi_S(x)$，低阶项 $|S|$ 小对应低频。作者定义**期望谱集中**：若 $\mathbb E_{w\sim\mu}\big[\sum_{|S|>k}\hat f_w^2(S)\big]\le\epsilon$，则称该函数族在期望意义下「谱 $\epsilon$-集中到 $k$ 阶」。借助经典命题「取 $\epsilon=3\,\mathrm{NS}_\nu(f)$，则 $f$ 的谱 $\epsilon$-集中到 $1/\nu$ 阶」（线性地推广到 ENS），把 Theorem 2 的稳定性界直接翻译成谱集中界。Corollary 1 给出：二分类 sLIF-SNN 在期望下谱 $\epsilon$-集中到 $1/\nu'$ 阶，$\epsilon=C_{T,\theta}\,\nu'^{\,1/2^{2L+1}}\log^{3/2}n$。取 $\nu'=\frac{1}{\sqrt n\log n}$，则网络是 $O(n^{1/2^{2(L+1)}})$-集中到 $O(\sqrt n\log n)$ 阶——只有消失比例的高频对谱有贡献，故称「谱简单」。值得注意的是，集中的**最大阶数**与架构参数无关，而集中**程度**随 $L,T,\theta$ 恶化。这条性质比 De Palma et al. (2019) 的「到最近异类点的平均汉明距离大」更弱，但在脉冲网络里是自然涌现的。

### 损失函数 / 训练策略
理论部分针对随机初始化网络、无训练。实验里用 ADAM + 代理梯度（surrogate gradient）训练 sLIF / IF SNN（如三层网络在 MNIST 上训到 98% 训练精度），用来对比训练前后噪声敏感度的变化。

## 实验关键数据

### 主实验（噪声敏感度 ENS 验证）
用蒙特卡洛估计 $\mathrm{ENS}_{1/\sqrt n}$，验证 Theorem 1、2 的界并考察训练影响。

| 设置 | 网络 | 关键观察 |
|------|------|----------|
| 单神经元 | sIF / IF，$n=100/1000/10000$，$\theta=0.5,T=10$ | 各 $t$ 下敏感度都很低；Theorem 1 的界对 sIF 与 IF 都成立 |
| 5 层网络 | sIF / IF，宽度=输入维 | 深度对敏感度影响比延迟更强，但 Theorem 2 的界**高估**了这种影响 |
| 训练前后（MNIST，3 层） | sLIF / IF，$n=784$ | 训练后敏感度显著下降（最终精度足够高时） |
| 训练前后（CIFAR-10） | $n=3072$ | 训练同样降敏感度，但幅度小于 MNIST（CIFAR-10 训练精度仅 84.38%） |
| 神经形态数据（NMNIST） | 卷积 SNN，$n=2312$ | 训练前后 ENS 都很小，训练对 ENS 影响不如静态数据明显 |

### 消融 / 扰动方式对比（5 层网络，$n=1000$，$\beta=0.5,\theta=1$）

| 模型 / 扰动 | 随机翻转 | 丢弃 5% 输入（dropout） |
|------------|---------|------------------------|
| sLIF | 0.19 | **0.16** |
| LIF | 0.28 | **0.16** |

### 关键发现
- **稳定性是内禀的、且随宽度增强**：随机初始化的宽 SNN 平均敏感度就已很低，印证了「宽网络谱简单」的理论。
- **训练让网络更稳**：在静态数据（MNIST/CIFAR-10）上训练显著降低噪声敏感度，且精度越高降得越多；但在事件型数据 NMNIST 上训练的去敏感效果弱很多。
- **理论界偏松、尤其在深度上**：5 层实验显示 Theorem 2 高估了深度的负面影响，作者明确把 $L,T,\theta$ 依赖是否内禀列为开放问题。
- **dropout 比随机翻转更鲁棒**：因为 $\{0,1\}^n$ 输入里 dropout 不改变零分量，所以扰动更温和。

## 亮点与洞察
- **跨领域工具迁移**：第一次把成熟的布尔函数分析（噪声敏感度 + 傅里叶–沃尔什谱）搬到 SNN，给一个理论薄弱的方向接上了 ANN 已有的「简单性偏差」叙事，思路本身就很值得借鉴。
- **reset-by-subtraction 的正面硬刚**：没有像前人那样简化成 reset-to-zero，而是直面减阈值复位带来的时间依赖，用「条件二项分布 + 吸收马尔可夫链 + Chernoff」逐层传播误差，是方法上的硬骨头。
- **「稳定 ⟹ 谱集中 ⟹ 简单」三段桥**：把一个鲁棒性结论（输出不翻转）转化成一个表示论结论（谱在低频），可复用到其他二值/阈值网络的简单性分析。
- **随机网络当 PAC-Bayes 先验**：把分析锁在初始化网络，既隔离了模型内禀性质，又留了一个接 PAC-Bayes 泛化界的接口。

## 局限与展望
- **理论假设较强**：主结果限定 $\beta=1$（无泄漏 IF）+ 静态输入 + sign 激活 + 大宽度，$\beta\neq1$、动态输入只在附录/实验里触及；动态输入下「输入和落在超立方体外」会带来额外技术困难。
- **界偏松**：$t^2$、$T^4$、$\log^{3/2}n$ 及 $\theta$ 依赖多被作者自认为是证明 artifact，$L,T$ 依赖是否内禀未解决，实验也显示界高估了深度影响。
- **只到初始化，训练无理论**：训练让稳定性变强是纯实验观察，缺少理论刻画——这是作者明确留给未来的工作。
- **谱简单性是较弱概念**：比 De Palma et al. 的平均汉明距离定义弱，能否推出更强的泛化结论尚不清楚。

## 相关工作与启发
- **vs Ding et al. (2024)（动力系统视角）**：他们分析单神经元、reset-to-zero、看脉冲序列差异；本文在分类器层面（预测可不变即便脉冲不同）、用更复杂的 reset-by-subtraction、并扩展到多神经元多层网络，方法基底是布尔函数分析而非 Lyapunov。
- **vs Jonasson et al. (2023)（布尔阈值网络）**：方法上最接近，借用了其马尔可夫链 + Chernoff 框架；但 SNN 的 reset 动力学和时间演化引入了前者没有的概率依赖，是本文新增的技术挑战。
- **vs De Palma et al. (2019)（随机 ANN 的简单性偏差）**：同样研究随机网络偏向简单函数，但他们用高斯过程论证（难推广到 SNN），本文用布尔函数分析（自然适配脉冲设定）；代价是「谱简单」比他们的「平均汉明距离大」更弱。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次把布尔函数分析引入 SNN 稳定性，并提出谱简单性概念
- 实验充分度: ⭐⭐⭐⭐ 覆盖单/深层、静态/事件数据、训练前后，但都是 ENS 验证型实验，规模偏小
- 写作质量: ⭐⭐⭐⭐ 理论链条清晰、坦诚标注了哪些依赖是证明 artifact
- 价值: ⭐⭐⭐⭐ 为理论薄弱的 SNN 提供了稳定性/简单性的严格刻画，并接上简单性偏差与 PAC-Bayes

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Overparametrization bends the landscape: BBP transitions at initialization in simple Neural Networks](overparametrization_bends_the_landscape_bbp_transitions_at_initialization_in_sim.md)
- [\[ICLR 2026\] Random Label Prediction Heads for Studying Memorization in Deep Neural Networks](random_label_prediction_heads_for_studying_memorization_in_deep_neural_networks.md)
- [\[ICLR 2026\] From Neural Networks to Logical Theories: The Correspondence between Fibring Modal Logics and Fibring Neural Networks](from_neural_networks_to_logical_theories_the_correspondence_between_fibring_moda.md)
- [\[ICLR 2026\] Reducing Symmetry Increase in Equivariant Neural Networks](reducing_symmetry_increase_in_equivariant_neural_networks.md)
- [\[ICLR 2026\] Proper Velocity Neural Networks](proper_velocity_neural_networks.md)

</div>

<!-- RELATED:END -->
