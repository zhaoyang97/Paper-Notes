---
title: >-
  [论文解读] Dynamics and Representation Structure of Local Approximations to Gradient-Based Learning in Linear Recurrent Neural Networks
description: >-
  [ICML 2026][优化/理论][线性 RNN] 本文在 student–teacher 数据对齐的线性 RNN 上，把 BPTT、one-step tBPTT、RFLO 的更新写成可解析的 ODE，比较它们的不动点流形、稳定性、收敛速率，发现 RFLO 缺少 BPTT/tBPTT 那条非最优鞍流形但代价是稳定性依赖符号、收敛更慢，并且**局限于初始权重的低秩扰动**——这一低秩限制可推广到非数据对齐的设定。
tags:
  - "ICML 2026"
  - "优化/理论"
  - "线性 RNN"
  - "RFLO"
  - "tBPTT"
  - "学习动力学"
  - "低秩约束"
---

# Dynamics and Representation Structure of Local Approximations to Gradient-Based Learning in Linear Recurrent Neural Networks

**会议**: ICML 2026  
**arXiv**: [2606.00243](https://arxiv.org/abs/2606.00243)  
**代码**: 待确认  
**领域**: 优化 / 学习理论 / 神经科学  
**关键词**: 线性 RNN, RFLO, tBPTT, 学习动力学, 低秩约束

## 一句话总结
本文在 student–teacher 数据对齐的线性 RNN 上，把 BPTT、one-step tBPTT、RFLO 的更新写成可解析的 ODE，比较它们的不动点流形、稳定性、收敛速率，发现 RFLO 缺少 BPTT/tBPTT 那条非最优鞍流形但代价是稳定性依赖符号、收敛更慢，并且**局限于初始权重的低秩扰动**——这一低秩限制可推广到非数据对齐的设定。

## 研究背景与动机

**领域现状**：训练 RNN 的金标准是 BPTT (Backpropagation Through Time)，但 BPTT 在空间和时间上都不局部——更新要依赖远距离的隐状态和远早时刻的误差。对神经科学来说，这种非局部性使 BPTT 难以解释生物大脑的学习；对神经形态硬件来说，非局部访存也是落地的硬伤。为此社区造出一批「局部近似」算法：把 BPTT 截到 $\tau$ 步的 one-step tBPTT，把 RTRL 的 Jacobian 乘积换成对角矩阵 + 随机反馈的 RFLO，以及把 RFLO 改成对角 $W$ 的 e-prop。

**现有痛点**：这些局部算法**并不是任何目标函数的真梯度**，所以理论上没保证它们会沿损失下降，更没保证它们会收敛到与 BPTT 相同的解。社区只有零星的实证比较，缺乏对「不动点结构、稳定性、收敛速率」这套基本学习动力学性质的系统分析。

**核心矛盾**：要分析非梯度的、非线性的学习动力学，必须有一个足够 tractable 的设置——既要保留 RNN 时间结构带来的核心难度，又要让 ODE 推导可解。

**本文目标**：把 BPTT、one-step tBPTT、RFLO 三者放在同一个数学框架里，回答：(i) 不动点长什么样？(ii) 哪些不动点稳定？(iii) 在最优流形附近收敛速度谁快谁慢？(iv) 学到的解在表示空间里有什么结构特征？

**切入角度**：复用 Proca 等人为 BPTT 设计的「数据对齐线性 RNN」框架，把学生和教师的输入/输出/递归矩阵在同一组正交基下联合对角化，使 $n$ 维 RNN 的学习动力学退化为 $n$ 个互不耦合的三维 ODE（每个模式只有 $(a,b,w)$ 三个标量参数）。

**核心 idea**：把 tBPTT 和 RFLO 的更新规则也搬到这套对角化框架里，取 $T\to\infty$ 与 $\eta\to0$ 双重极限，拿到三组可比较的 ODE，然后用动力系统标准武器（不动点 + 雅可比线性化 + 数值积分）逐项对比。

## 方法详解

### 整体框架
学生–教师线性 RNN 用同一段高斯白噪声 $x_t\sim\mathcal{N}(0,\mathbf{I})$ 驱动，学生 $h_{t+1}=Wh_t+Bx_t,\;y_{t+1}=Ah_{t+1}$；教师参数加 $\star$ 上标。损失是末时刻 $L_T=\tfrac{1}{2}\|y_T-y_T^\star\|^2$ 的期望。三种算法的更新都按 $\theta_{k+1}=\theta_k-\eta\Delta\theta_k$ 形式：

- **BPTT**：$\Delta W=\sum_{t=1}^{T}(W^{T-t})^\top A^\top\mathbb{E}[\varepsilon_T h_{t-1}^\top]$。
- **one-step tBPTT** ($\tau=1$)：只保留 $t=T$ 那一项，即 $\Delta_\tau W=A^\top\mathbb{E}[\varepsilon_T h_{T-1}^\top]$。
- **RFLO**：把 $W^{T-t}$ 换成 $\widehat{W}^{T-t}=\hat w^{T-t}\mathbf{I}$（标量乘单位阵）、把 $A^\top$ 换成固定随机反馈 $R^\top$，得到 $\Delta_{\mathrm{RFLO}} W=\sum_t(\widehat W^{T-t})^\top R^\top\mathbb{E}[\varepsilon_T h_{t-1}^\top]$。

数据对齐假设要求：输入–输出相关矩阵 $\Sigma_t^\star=\mathbb{E}[y_T^\star x_t^\top]$ 可分解为 $\Sigma_t^\star=U S_t V^\top$（$U,V$ 正交、$S_t$ 对角），并且教师 $(A_\star,W_\star,B_\star)$ 与学生 $(A_0,W_0,B_0)$ 在初始化时都按 $(U,V,P_\star)$ 联合对角化（$P_\star,P$ 是正交、$\bar{\cdot}$ 是对角部分）。对齐后每个学生模式 $(a,b,w)$ 与对应教师模式 $(a_\star,b_\star,w_\star)$ 独立演化，把高维问题拆成 $n$ 个互不耦合的三维 ODE。RFLO 的随机反馈 $R=U\bar R P^\top$（$\bar R$ 对角随机）也被纳入这套对角化。

取 $T\to\infty$ 极限把更新求和写成闭式（利用几何级数 $\sum w^t$），再取 $\eta\to0$ 把离散更新升级成 ODE $\dot\theta=-\Delta\theta$，得到三组解析 ODE（公式 19–22）。共同的 $a$-方向更新是 $\Delta a\to \tfrac{ab^2}{1-w^2}-\tfrac{a_\star b b_\star}{1-w w_\star}$；$w,b$ 方向因算法而异，例如 RFLO 的 $\Delta_{\mathrm{RFLO}} b\to \tfrac{\hat a a b}{1-\hat w w}-\tfrac{\hat a a_\star b_\star}{1-\hat w w_\star}$ 因为 $\hat a$ 在分子里，$b=0$ 不会自动让 $\Delta b=0$，这是 RFLO 缺少非最优流形的根本原因。

### 关键设计

**1. 数据对齐 + 双重极限把 RNN 学习对角化为 3D ODE：给三个算法一个统一基座**

直接分析高维、非线性、时间耦合的学习动力学几乎不可能，必须先化简。作者从 Proca 等 (2025) 给 BPTT 设计的数据对齐对角化出发，关键拓展是证明 tBPTT 和 RFLO 也能在同一组正交基下被对角化——只要把 RFLO 的随机反馈规定为 $R=U\bar R P^\top$、$\widehat W=\hat w\mathbf{I}$ 即可。对角化一旦成立，每个模式的更新 $(\Delta W,\Delta A,\Delta B)$ 都是对角矩阵，模式之间真的不交换信息，于是 $n$ 维 RNN 退化成 $n$ 个独立的 $(a,b,w)$ 三维系统。再取 $T\to\infty$ 用几何级数 $\sum_{t=0}^\infty w^t=1/(1-w)$ 闭式求和，得到形如

$$\Delta a=\frac{ab^2}{1-w^2}-\frac{a_\star b b_\star}{1-w w_\star}$$

的有理分式 ODE，最后 $\eta\to 0$ 升级到连续时间。这个 3D ODE 既保留了 RNN 时间依赖的核心难度（递归项 $W^t$ 通过几何级数留在分母里），又允许逐点雅可比线性化，是后面所有结论的统一数学基座。

**2. 三算法不动点结构对比：RFLO 缺一条非最优鞍流形**

不动点回答的是"学习能停在哪里"。把各算法 ODE 右端置零求解，三者都有一条最优流形 $\{ab=a_\star b_\star,\ w=w_\star\}$（这条一维曲线上损失最小，参数化为 $(s,a_\star b_\star/s,w_\star),\ s\ne 0$）；但 BPTT 和 tBPTT 还额外有一条非最优流形 $\{a=b=0,\ w\text{ 任意}\}$，损失停在严格正值 $\sum a_\star^2 b_\star^2/[2(1-w_\star^2)]$。RFLO 没有这条线，因为它的 $\Delta_{\mathrm{RFLO}} b$ 即使在 $a=b=0$ 处也不为零——被 $\hat a a_\star b_\star/(1-\hat w w_\star)$ 这一项撑住。少一条鞍线乍看是好事（不会在鞍点附近卡出慢相位），但代价藏在下一条结果里。

**3. 稳定性 + 收敛速率：RFLO 牺牲速度换来低秩解**

把雅可比在最优流形上线性化，就能量化"停在最优解附近时学得多快、稳不稳"。BPTT 雅可比特征值实部为负、最大特征值 $\lambda_+$ 决定速度，在 $s=\pm\sqrt{|a_\star b_\star|}$ 处最慢；tBPTT 在 $|w_\star|$ 小时贴近 BPTT；RFLO 的特征值显式依赖 $\mathrm{sgn}(\hat a s)$——$\hat a s>0$ 稳定，$\hat a s<0$ 时 $\lambda_+$ 可能转正、出现不稳定甚至振荡区（公式 28–29），而且几乎整条流形上都比 BPTT 慢，只在 $s\to 0$（小 $a^\star$、大 $b^\star$ 模式）时反超。另一面，Proposition 3.1 证明 RFLO 的更新必然低秩：$W_K=W_0+\sum_{i=1}^o r_i q_i^\top$、$B_K=B_0+\sum_{i=1}^o r_i q_i^{(b)\top}$，秩至多为输出维度 $o$。把这两条放一起，就同时解释了 Fig 4 里 RFLO"绕远路到对面分支"的行为，和 Fig 6 里局部规则都学出更低秩 $W_K-W_0$ 的现象——表达力被局部约束按住了。

### 损失函数 / 训练策略
理论部分用 $L_T=\tfrac{1}{2}\|y_T-y_T^\star\|^2$，附录 H 把所有主结果推广到序列损失 $\mathcal{L}=\tfrac{1}{2T}\sum_{t=1}^T\|y_t-y_t^\star\|^2$。实验部分用小方差高斯初始化学生（Saxe 等 2019 风格），学生递归维度允许 $\ge$ 教师维度以保证普适。

## 实验关键数据

### 主实验
非数据对齐学生 RNN 学习模态对齐教师 (4 个模式)，比较实验轨迹与理论 ODE 预测的吻合度。

| 算法 | 不动点流形 | 稳定性 | 收敛速度（最优流形上） |
|------|-----------|--------|----------|
| BPTT | 最优 (cyan) + 非最优 (red, 鞍) | 最优稳定、非最优鞍 | 最快，最慢点在 $s=\pm\sqrt{|a_\star b_\star|}$ |
| tBPTT ($\tau=1$) | 同 BPTT | 同 BPTT | 当 $\|w_\star\|$ 小时接近 BPTT |
| RFLO | 仅最优 | 符号相关：$\hat a s>0$ 稳定 / $\hat a s<0$ 不稳定/振荡 | 多数 $s$ 处最慢，仅 $s\to 0$ 时反超 |

### 消融实验（学到的解的秩）

| 算法 | $W_K-W_0$ 谱形状 | 解释 |
|------|----------|------|
| BPTT | 高秩（多个显著奇异值） | 无局部性约束 |
| e-prop | 中秩 | 用对角 $W$ 反传 |
| tBPTT ($\tau=1$) | 接近 RFLO 的低秩 | 只用末步误差 |
| RFLO | 严格秩 $\le o$ (本例 $=1$) | Proposition 3.1：$W_K=W_0+\sum_{i=1}^o r_iq_i^\top$ |

### 关键发现
- **理论可外推到非数据对齐设定**：经过短暂瞬态后，对齐度（recurrent、input/output、random feedback）持续上升，ODE 预测与数值实验在两个主导模式上吻合很好；这说明数据对齐不是必要条件，而是一个「学到的对齐」结果。
- **RFLO 牺牲速度换稳定性**：不动点更少（没有非最优鞍线）听上去是优点，但稳定性变得依赖 $\mathrm{sgn}(\hat a s)$，且 Fig 4 显示从 $(1,1,0.6)$ 出发 RFLO 不会收敛到最近的最优分支，反而绕一大圈到对面分支，wall-clock 学习时间剧增。
- **局部规则 ⇒ 低秩解**：所有局部算法（RFLO、tBPTT、e-prop）学到的 $W_K-W_0$ 都比 BPTT 低秩；20 次模拟中 RFLO 仅 13 次、e-prop 仅 12 次收敛，而 tBPTT/BPTT 全部收敛，秩与稳定性/性能高度相关（Supp Fig 10）。
- **可迁移到非线性 / 状态空间模型？** 作者指出线性 RNN 与现代 SSM (Mamba 等) 在表达力上属于同一类，因此 RFLO 等局部规则在 SSM 上也可能受同样的低秩约束——这对神经形态硬件训练线性 SSM 是值得警惕的暗示。

## 亮点与洞察
- **一个理论框架同时覆盖三个算法 + 三个性质**：把不动点几何、雅可比稳定性、解的秩这三件事在同一个数据对齐线性 RNN 上一次性回答清楚——这是这篇论文最值钱的地方，远比单做一项实证比较深刻。
- **「为什么 RFLO 训出来的解很穷」有了机制解释**：Proposition 3.1 的证明只需要 $\widehat W=\hat w\mathbf{I}$，与是否数据对齐无关，把秩界限严格压到输出维度 $o$。这条结果直接告诉硬件研究者：要让神经形态系统能学复杂解，不能用纯标量反馈，必须给随机反馈更高的有效秩。
- **数据对齐假设的「合法性」被数值证伪 + 再证立**：作者承认不是所有场景都自然对齐，但实验发现训练过程会自发对齐，这把过去被批评「数据对齐太强」的理论一族重新合法化——「假设是结果」是漂亮的反转。
- **可迁移设计思路**：「把非梯度更新规则装回某个 ODE 框架，比较其与真梯度的差异」是研究一切 surrogate gradient（脉冲网络、量子化反传、合成梯度）的通用模板，本文给出了在线性 RNN 上的完整 case study。

## 局限与展望
- 仅限线性 RNN。作者承认非线性 RNN 的学习动力学复杂得多，文中数学技巧（几何级数求和、对角化）不能直接搬过去。
- $T\to\infty$ + $\eta\to0$ 双重极限丢掉了有限批次/有限步长效应；现实训练中 mini-batch 噪声和学习率调度可能改写 RFLO 的稳定性结论。
- 关于 RFLO 「绕远路到对面最优分支」的现象只在小例子里展示，对实际 SSM 的影响需要在更大规模、更长序列上验证。
- e-prop 用对角 $W$ 反馈而非随机反馈，秩约束应该不同；本文只做了实证比较，没给 e-prop 的 closed-form 秩界，是自然的下一步。

## 相关工作与启发
- **vs Proca et al. (2025)**：Proca 把 BPTT 在数据对齐线性 RNN 上对角化，本文是把这个框架推广到 tBPTT 和 RFLO 两个非梯度算法上，并把分析对象从「BPTT 自身」扩展到「BPTT 与局部近似的差异」。
- **vs Saxe et al. (2014, 2019)**：Saxe 系列在前馈线性网络上做学习动力学，本文把同样的「线性可解、动力学非线性」哲学搬到 RNN 域，并显式处理时间维度带来的几何级数。
- **vs Murray (2019) RFLO 原作**：Murray 给 RFLO 数值证据，但缺少机制解释；本文给出 RFLO 没有非最优流形 + 局限于秩 $\le o$ 的两条结构性结论。
- **vs Bellec et al. (2020) e-prop**：e-prop 是脉冲网络的局部规则，本文在 rate 网络上的版本（对角 $W$ 而非纯标量）显示秩高于 RFLO 但仍低于 BPTT，给「局部 ⇒ 低秩」家族提供了梯度。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把 tBPTT/RFLO 拉入数据对齐对角化框架并对比三种几何性质是新工作；秩定理是 elegant 的新观察。
- 实验充分度: ⭐⭐⭐⭐ 理论 + 数值 ODE 积分 + 真实学生–教师训练三层验证，Fig 1–6 + 多个附录把每条结论都覆盖。
- 写作质量: ⭐⭐⭐⭐ 数学密度高但符号统一、推导节奏清晰，每个定理都配对应图示。
- 价值: ⭐⭐⭐⭐ 对神经科学（解释生物可塑性的能力上限）和神经形态硬件（怎么设计反馈结构才能突破低秩约束）都有直接指导意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Balancing Learning Rates Across Layers: Exact Two-Step Dynamics and Optimal Scaling in Linear Neural Networks](balancing_learning_rates_across_layers_exact_two-step_dynamics_and_optimal_scali.md)
- [\[AAAI 2026\] On the Learning Dynamics of Two-Layer Linear Networks with Label Noise SGD](../../AAAI2026/optimization/on_the_learning_dynamics_of_two-layer_linear_networks_with_label_noise_sgd.md)
- [\[ICML 2026\] Learning-Augmented Scalable Linear Assignment Problem Optimization via Neural Dual Warm-Starts](learning-augmented_scalable_linear_assignment_problem_optimization_via_neural_du.md)
- [\[ICML 2026\] Ubiquity of Emergent Hebbian Dynamics in Regularized Learning](ubiquity_of_emergent_hebbian_dynamics_in_regularized_learning.md)
- [\[ICML 2026\] The Implicit Bias of Adam and Muon on Smooth Homogeneous Neural Networks](the_implicit_bias_of_adam_and_muon_on_smooth_homogeneous_neural_networks.md)

</div>

<!-- RELATED:END -->
