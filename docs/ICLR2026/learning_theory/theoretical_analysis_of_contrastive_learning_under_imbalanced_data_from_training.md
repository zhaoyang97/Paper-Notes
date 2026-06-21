---
title: >-
  [论文解读] Theoretical Analysis of Contrastive Learning under Imbalanced Data: From Training Dynamics to a Pruning Solution
description: >-
  [ICLR 2026][对比学习理论][对比学习] 本文给出对比学习在**不平衡数据**下的训练动力学理论：以「Transformer-MLP + 稀疏编码数据模型」为分析对象，证明神经元权重经历三阶段演化、少数特征因频率低而被学得更弱更混杂，并从理论上说明**幅值剪枝**能放大少数特征方向的梯度更新，从而恢复被不平衡损害的表征质量（CIFAR-LT / ImageNet-LT 线性探针实验验证）。
tags:
  - "ICLR 2026"
  - "对比学习理论"
  - "表示学习"
  - "特征学习"
  - "对比学习"
  - "数据不平衡"
  - "训练动力学"
  - "幅值剪枝"
---

# Theoretical Analysis of Contrastive Learning under Imbalanced Data: From Training Dynamics to a Pruning Solution

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=DUXG9E8dEO](https://openreview.net/forum?id=DUXG9E8dEO)  
**代码**: 待确认  
**领域**: 对比学习理论 / 表示学习 / 特征学习  
**关键词**: 对比学习, 数据不平衡, 训练动力学, 特征学习, 幅值剪枝

## 一句话总结
本文给出对比学习在**不平衡数据**下的训练动力学理论：以「Transformer-MLP + 稀疏编码数据模型」为分析对象，证明神经元权重经历三阶段演化、少数特征因频率低而被学得更弱更混杂，并从理论上说明**幅值剪枝**能放大少数特征方向的梯度更新，从而恢复被不平衡损害的表征质量（CIFAR-LT / ImageNet-LT 线性探针实验验证）。

## 研究背景与动机

**领域现状**：对比学习（如 SimCLR、CLIP 背后的 InfoNCE 范式）通过「拉近正对、推远负对」从无标签数据里学通用表征，已经成为视觉、多模态、视觉-语言模型预训练的主力。理论侧近年也有进展，但大多在解释「为何数据增广必要」「为何优于 GAN」「为何能降低下游样本复杂度」这些问题。

**现有痛点**：真实数据几乎都是长尾/不平衡的——多数类主导了正负对的构成，少数类被严重低估，导致少数类的判别特征学不好、整体表征质量下降。监督学习里靠 re-weighting / re-sampling 缓解，但这些方法都**依赖准确的类别标签**，而自监督对比学习里根本没有标签，于是社区转向了**剪枝（pruning）**这类经验方法来隐式地照顾长尾数据。

**核心矛盾**：所有这些缓解手段——无论 re-sampling 还是 pruning——都是**启发式**的，只在实践中观察到涨点，却**说不清「不平衡究竟从哪个机制上、以何种方式」损害了表征**，也说不清剪枝为什么有用。理论与实践之间存在一道鸿沟。

**本文目标**：把这道鸿沟补上，具体拆成三个子问题——(1) 对比学习在不平衡数据下，神经元到底怎么逐步学到特征？(2) 少数特征的低频率，量化地如何拖垮表征能力？(3) 幅值剪枝在动力学层面为什么能救回少数特征？

**切入角度**：作者借用**特征学习（feature learning）范式**，把数据建模成**稀疏编码模型**（每个 token 是字典特征的稀疏线性组合 + 噪声），用特征出现频率 $\epsilon_j$ 来刻画 majority / minority，然后**逐神经元、逐阶段地追踪权重与各特征方向的内积如何演化**。这个角度有希望，是因为它把「不平衡」精确编码成了频率参数 $\epsilon_j$，使得「频率→学习速率→表征纯度」的因果链可被严格推导。

**核心 idea**：用「神经元在特征方向上的投影动力学」这把尺子，证明**低频少数特征会被学得更弱、更混杂、专精神经元更少**，再证明**幅值剪枝恰好偏向性地放大少数特征方向的梯度**，从而把不平衡造成的损害定量地补回来。

## 方法详解

### 整体框架

本文不是一个新算法，而是一套**分析框架 + 一个被重新解释的剪枝过程**。整体逻辑链是：先定义一个**可解析的简化模型**（Transformer-MLP 编码器 + 稀疏编码数据），用 InfoNCE 损失训练；然后把训练过程**切成三个阶段**，逐阶段给出神经元权重的演化界；据此**定量刻画不平衡（频率比 $\epsilon_{\min}/\epsilon_{\max}$）如何在三个维度上削弱表征**；最后把幅值剪枝塞进同一框架，证明它如何**改写少数特征方向的更新速率**，把劣势补回。

具体设定：输入序列 $X=[x^{(1)},\dots,x^{(L)}]$ 过一个**单头自注意力**再接一个带**双边 ReLU（BReLU）** 的 MLP，输出嵌入 $f_\theta(X)\in\mathbb{R}^m$。BReLU 定义为 $\mathrm{BReLU}_b(s)=\mathrm{ReLU}(s-b)-\mathrm{ReLU}(-s-b)$，即一个带阈值 $b$ 的对称激活。第 $i$ 个隐藏单元为

$$h_i(X_n)=\sum_{r=1}^{L}\mathrm{BReLU}_{b_i^{(t)}}\big(\langle w_i^{(t)},\ \mathrm{Attention}(W_Q x_n^{(r)}, W_K X_n, W_V X_n)\rangle\big).$$

训练目标是带 $\ell_2$ 正则的 InfoNCE 经验风险：

$$\widehat{L}_{\mathrm{aug}}(f_\theta)=\frac{1}{K}\sum_{k=1}^{K}\ell\big(f_\theta,X_k,Y_k,N_k\big)+\frac{\lambda}{2}\|\theta\|_F^2,$$

其中相似度用 $\mathrm{sim}_{f_\theta}(X_n,Y_n)=\langle f_\theta(X_n),\ \mathrm{StopGrad}(f_\theta(Y_n))\rangle$，StopGrad 在前向是恒等、反向阻断梯度。整套分析的「显微镜」是追踪内积 $\langle w_i^{(t)}, M_j\rangle$——即神经元 $i$ 在第 $j$ 个特征方向 $M_j$ 上的投影——随训练步 $t$ 如何变化。这是纯机制/数学分析，没有可画成 pipeline 的多模块流程，故不配框架图。

### 关键设计

**1. 稀疏编码数据模型 + 频率参数 $\epsilon_j$：把「不平衡」翻译成可解析的频率**

要在理论上谈「不平衡损害表征」，先得有一个能把「多数/少数」精确量化的数据模型。作者采用**稀疏编码模型**（Assumption 3.1）：每个 token $x^{(\ell)}_n = M z^{(\ell)}_n + \xi^{(\ell)}_n$，其中 $M=[M_1,\dots,M_d]$ 是**列正交**的字典矩阵，$z$ 是稀疏隐信号，$\xi\sim\mathcal{N}(0,\sigma_\xi^2 I)$ 是噪声。关键之处在于噪声方差 $\sigma_\xi^2=\Theta(\sqrt{\log d}/d)$ 允许**噪声量级与信号相当甚至超过信号**（当 $d_1\gg d$），所以没有线性映射能直接从输入恢复隐信号——这让问题「形式简单但本质困难」，正适合检验非线性网络。正负对由「token 聚合后的隐信号支撑集与符号是否一致」定义（Assumption 3.4），把对比学习的语义结构形式化。

不平衡通过 **Definition 3.1** 注入：特征 $j$ 的激活概率为 $\Pr(z^{(i)}_j\neq 0)=\Theta(\epsilon_j \log\log d / d)$。$\epsilon_{\max}=\max_j\epsilon_j$ 对应**多数特征**，$\epsilon_{\min}=\min_j\epsilon_j$ 对应**少数特征**。于是「数据有多不平衡」被压缩成一个干净的比值 $\epsilon_{\min}/\epsilon_{\max}$，后续所有定量结论都挂在这个比值上——这正是整套分析能推下去的支点。

**2. 三阶段训练动力学：神经元如何从「乱长」到「专精」**

这是全文的核心机制结论：尽管算法看上去只是一致的梯度下降，神经元权重其实分**三阶段**演化（对应 Lemma 3.1、Lemma 3.2、Theorem 3.1）。

*Stage 1（Lemma 3.1）*：神经元在**特征方向上增长、在非特征方向被压制**，且增速由频率决定。形式上，

$$|\langle w_i^{(t+1)},M_j\rangle|\ \ge\ |\langle w_i^{(t)},M_j\rangle|\Big(1-\eta\lambda+\epsilon_j\frac{\eta C_z\log\log d}{d}\Big)-\widetilde{O}\Big(\frac{\eta\|w_i^{(t)}\|^2}{\mathrm{poly}(d_1)}\Big),$$

而非特征方向 $M_j^\perp$ 上的投影只会衰减。这里增长因子里直接带着 $\epsilon_j$：**频率高的多数特征长得快，频率低的少数特征早期就难以被捕捉**。

*Stage 2（Lemma 3.2）*：神经元分化成两类——**lucky 神经元** $M^\star_j$（只对齐单一特征方向）和**ordinary 神经元** $M_j$（对齐某方向但更混）。lucky 神经元显著强化与 $M_j$ 的对齐：$|\langle w^{(T_2)}_i,M_j\rangle|^2\ge 2\cdot\frac{\epsilon_j}{\epsilon_{\max}}\|w^{(T_1)}_i\|_2^2$，且其数量 $|M^\star_j|\ge m\cdot d^{-(\epsilon_{\max}/\epsilon_{\min})^2}$；ordinary 神经元则被 lucky 神经元的特征分量在常数倍内界住。结果是**学到的特征变纯、非特征分量持续被压**。

*Stage 3（Theorem 3.1，收敛）*：训练误差收敛到 $o(1)$，每个神经元收敛成

$$w_i^{(t)}=\sum_{j\in N_i}\alpha_{i,j}M_j+\sum_{j\notin N_i}\alpha'_{i,j}M_j+\sum_{j\in[d_1]\setminus[d]}\beta_{i,j}M_j^\perp,$$

即**强对齐一小簇主特征 $N_i$、弱对齐其余特征、在非特征方向上几乎为零**。主系数满足 $\alpha_{i,j}\in[\frac{\epsilon_j}{\epsilon_{\max}}\frac{\tau}{\Xi_2},\ \frac{\epsilon_j}{\epsilon_{\max}}\tau]$。这条收敛刻画是后面所有「不平衡危害」与「剪枝救援」结论的落脚点。

**3. 不平衡的三重危害：频率比如何同时压低幅值、扩大混杂、减少专精神经元**

把 Stage 3 的结论代入，作者定量给出不平衡损害表征的**三条相互纠缠的路径**（K2 与 Theorem 3.1 的 Remark）：

- **少数特征学得更弱**：神经元在 $N_i$ 上的幅值 $\alpha_{i,j}\propto \epsilon_j/\epsilon_{\max}$，故 $\epsilon_j$ 越小、该特征被学得越弱。
- **神经元更易混杂**：主特征集合大小 $|N_i|=O(d^{1-(\epsilon_{\min}/\epsilon_{\max})^2})$，比值 $\epsilon_{\min}/\epsilon_{\max}$ 越小，$|N_i|$ 越大，意味着一个神经元被迫**混学多个特征**而非保持纯净。
- **专精神经元数量下降**：对每个特征 $M_j$，纯净专精它的神经元数下界为 $\Omega(m\cdot d^{-(\epsilon_{\max}/\epsilon_{\min})^2})$，随 $\epsilon_{\max}$ 与 $\epsilon_{\min}$ 差距拉大而锐减。

由于对比学习的成功**依赖 lucky 神经元（专精单一纯特征）**——混学多特征的神经元只对少数下游任务有用——这三条合起来就是：不平衡让少数特征又弱、又混、专精它的神经元又少，从而**逼迫模型用更多神经元/更复杂结构**才能覆盖所有特征，代价是更高的算力。Theorem 3.1 的 Remark 3 进一步指出：当上游学到的特征方向纯净可分时，下游只需线性探针即可轻松抽取对应特征，**神经元专精度越高，线性可分性与下游泛化越好**——这把「上游表征结构」和「下游可用性」连成了一条因果链。

**4. 幅值剪枝放大少数特征：用「谁更易被剪」改写更新速率**

既然小幅值神经元正好是学少数特征的那批（由设计 3 可知少数特征幅值更小），作者重新审视幅值剪枝并给出动力学解释（Algorithm 1 + Theorem 3.2）。算法是**前向带掩码、反向不带掩码**：每个 epoch 把幅值最小的 $\alpha$ 比例神经元用二值掩码 $M^{(t)}$ 临时屏蔽（$\theta^{(t)}_{mk}=\theta^{(t)}\odot M^{(t)}$）来前向编码、算损失，但**梯度施加到完整参数集**上：

$$\theta^{(t+1)}\leftarrow(1-\eta\lambda)\theta^{(t)}-\eta\cdot g(\theta^{(t)}_t,M^{(t)}).$$

机制上（Theorem 3.2）：被剪掉的少数特征神经元在该样本上**正 logit 变小、负 logit 变大**，于是含少数特征的样本在梯度更新里被放大。定量地，对齐少数方向 $M_{j^\star}$ 的 lucky 神经元增长速率被抬到 $\Omega(\eta\epsilon_{j^\star}^2\alpha\,C_z\log\log d/d)$（即量级 $\alpha/d$），而非少数特征方向只以 $\alpha/d^2$ 量级微增——两者**相差一个 $1/d$ 因子**。最关键的是收敛系数 $\alpha_{i,j^\star}\in(\tau/\Xi_2,\ \tau)$ **不再依赖比值 $\epsilon_{\min}/\epsilon_{\max}$**：少数特征的表征不再被不平衡压制，专精它的神经元下界从 $\Omega(m\cdot d^{-(\epsilon_{\max}/\epsilon_{\min})^2})$ 抬升到 $\Omega(m\cdot d^{-1})$。这就从理论上解释了「剪枝→放大少数特征→更多专精神经元→更鲁棒平衡的表征」这条经验观察。

### 损失函数 / 训练策略

训练用带 $\ell_2$ 正则（权重衰减 $\lambda$）的 InfoNCE，温度 $\tau$，正负对相似度用内积 + StopGrad。剪枝侧（Algorithm 1）：MLP 权重高斯初始化、注意力权重初始化为单位阵，掩码初始全 1；每步剪 $\alpha$ 比例最小幅值神经元，前向用掩码参数、反向更新完整参数。理论分析的关键尺度参数：Stage 1 时长 $T_1=\Theta(d_1\log d/(\eta\log\log d))$，Stage 2 时长再加 $\Theta(d\tau\log d/(\epsilon_{\max}\eta\log\log d))$，收敛区间 $T\in[T_3,T_4]=[d^{1.01}/\eta,\ d^{1.99}/\eta]$。

## 实验关键数据

实验为「理论验证」性质：CIFAR10-LT / CIFAR100-LT / ImageNet-LT 上做**线性探针**评测，对比 vanilla 对比学习（w/o pruning）与加剪枝（w/ pruning）。不平衡比 $\rho$ = 多数类样本数 / 少数类样本数，越大越不平衡；指标为总体准确率与 $\Delta_{20}$（前 20% head 类与后 20% tail 类的准确率差距）。

### 主实验

| 数据集 | $\rho$ | Acc (w/o 剪枝) | Acc (w/ 剪枝) | $\Delta_{20}$ (w/o) | $\Delta_{20}$ (w/ 剪枝) |
|--------|------|------|------|------|------|
| CIFAR10-LT | 1 | 90.93 | **91.52** | 1.54 | **1.28** |
| CIFAR10-LT | 10 | 79.25 | **84.92** | 3.42 | **2.99** |
| CIFAR10-LT | 50 | 75.58 | **83.60** | 3.92 | **3.35** |
| CIFAR10-LT | 100 | 74.24 | **81.31** | 5.69 | **5.62** |
| CIFAR100-LT | 10 | 51.21 | **56.33** | 2.45 | **1.37** |
| CIFAR100-LT | 50 | 49.32 | **56.12** | 4.95 | **2.57** |
| CIFAR100-LT | 100 | 47.12 | **54.93** | 7.11 | **4.38** |
| ImageNet-LT | 256 | 63.21 | **65.12** | 8.47 | **7.21** |

### 关键发现（剪枝的增益随不平衡加剧而放大）

| 现象 | 数据 | 说明 |
|------|------|------|
| 不平衡越重、剪枝增益越大 | CIFAR10-LT $\rho{=}10$ 涨 +5.67，$\rho{=}50$ 涨 +8.02 | 与理论一致：剪枝专门补少数特征 |
| 头尾差距 $\Delta_{20}$ 缩小 | CIFAR100-LT $\rho{=}100$ 从 7.11 → 4.38 | 剪枝改善 head/tail 平衡，而非只抬总体 |
| 平衡场景增益小 | CIFAR10-LT $\rho{=}1$ 仅 +0.59 | $\rho{=}1$ 几乎无少数特征可救，符合机制预期 |

### 关键发现
- **剪枝增益随 $\rho$ 单调放大**：$\rho$ 越大意味着少数特征越稀有、vanilla 学得越差，而剪枝恰好偏向放大这些方向的梯度，所以涨点更猛——这是「$\alpha/d$ vs $\alpha/d^2$ 速率差」的实证体现。
- **不只是涨总体，更是缩差距**：$\Delta_{20}$ 普遍下降，说明剪枝把性能补在了尾部类上，而非整体平移，印证了「专精少数特征神经元增多」的理论。
- 论文另有合成数据实验（附录 A.2）直接验证三阶段动力学与神经元投影曲线（正文 Figure 1）。

## 亮点与洞察
- **把「不平衡」压成一个频率比 $\epsilon_{\min}/\epsilon_{\max}$**：整套危害（弱幅值、混杂、专精神经元减少）都解析地挂在这一个比值上，可读性极高，也让「剪枝令系数不再依赖该比值」成为一句话就能说清的强结论。
- **给剪枝一个动力学层面的因果解释**，而不止于「实践有效」：小幅值神经元 ↔ 少数特征 ↔ 前向被剪后该样本 logit 失衡 ↔ 梯度被放大，链条闭环，这是把经验 trick 理论化的范式。
- **lucky 神经元 / ordinary 神经元的二分**很可迁移：把「表征好不好」归结到「有多少神经元专精单一纯特征」，这个视角可以搬到分析其他自监督/长尾方法。
- 用 Transformer-MLP（单头注意力）而非以往的单隐层前馈网络做特征学习分析，是把这类理论往现代架构推的一小步。

## 局限与展望
- **剪枝比率与方案的精确刻画缺失**：作者承认无法给出「性能如何随剪枝比 $\alpha$ 与剪枝方案变化」的完整刻画，需要更精确的数据分布假设，留作未来工作。
- **架构极度简化**：单头注意力 + 单层 MLP + BReLU + 列正交字典 + 高斯噪声，离真实的深层 Transformer / 真实图像分布很远；推广到更复杂模型可能需要全新的分析工具。
- **理论与实验的桥接偏松**：理论建立在稀疏编码 + 特征频率模型上，实验却直接用 CIFAR/ImageNet 长尾，二者之间「特征频率 ≈ 类频率」是隐含假设，严格对应性未充分论证。
- **改进思路**：可探索剪枝之外、同样能偏向性放大少数特征梯度的机制（如频率自适应的温度或正则），并尝试把分析扩到多头/多层以检验三阶段结论是否稳健。

## 相关工作与启发
- **vs Wen & Li (2021) / Sun et al. (2025)**：他们分析单隐层前馈网络下对比学习的训练动力学；本文换成 Transformer 架构、用不同数据模型，**并显式引入数据不平衡**，给出不平衡如何影响特征解耦的系统分析，而非仅靠特征幅值变化做直接外推。
- **vs Allen-Zhu & Li（feature purification）/ 监督特征学习系列**：那些工作把特征绑在 ground-truth 标签上，无法直接搬到无标签的对比学习；本文在自监督设定下追踪同类「神经元 ↔ 特征方向对齐」动力学。
- **vs HaoChen et al. (2021) / 图谱视角的对比学习理论**：他们用谱聚类/图论解释对比学习为何有效；本文不解释「为何有效」，而是聚焦「不平衡如何破坏有效性、剪枝如何修复」，是互补的问题面。
- **vs Jiang et al. (2021) / Qian et al. (2022) 的经验剪枝**：他们经验上发现剪枝能改善低估类表征；本文为这一现象提供了**首个动力学层面的理论解释**（$\alpha/d$ vs $\alpha/d^2$ 的速率差）。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次在 Transformer 编码器 + 不平衡数据下刻画对比学习训练动力学，并给剪枝以理论根据
- 实验充分度: ⭐⭐⭐ 三个长尾基准 + 合成数据验证趋势到位，但本质是理论验证，规模与架构有限
- 写作质量: ⭐⭐⭐⭐ 三阶段叙事清晰、频率比贯穿全文，公式较密但逻辑链完整
- 价值: ⭐⭐⭐⭐ 为「无标签长尾 + 剪枝」这条经验路线补上理论解释，对理解自监督表征结构有指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] A Theoretical Analysis of Mamba's Training Dynamics: Filtering Relevant Features for Generalization in State Space Models](a_theoretical_analysis_of_mambas_training_dynamics_filtering_relevant_features_f.md)
- [\[ICLR 2026\] Reshaping Reasoning in LLMs: A Theoretical Analysis of RL Training Dynamics through Pattern Selection](reshaping_reasoning_in_llms_a_theoretical_analysis_of_rl_training_dynamics_throu.md)
- [\[ICLR 2026\] Fast Escape, Slow Convergence: Learning Dynamics of Phase Retrieval under Power-Law Data](fast_escape_slow_convergence_learning_dynamics_of_phase_retrieval_under_power-la.md)
- [\[ICLR 2026\] Convergence Analysis of Tsetlin Machines under Noise-Free and Noisy Training Conditions: From 2 Bits to k Bits](convergence_analysis_of_tsetlin_machines_under_noise-free_and_noisy_training_con.md)
- [\[ICLR 2026\] A Generalized Geometric Theoretical Framework of Centroid Discriminant Analysis for Linear Classification of Multi-dimensional Data](a_generalized_geometric_theoretical_framework_of_centroid_discriminant_analysis_.md)

</div>

<!-- RELATED:END -->
