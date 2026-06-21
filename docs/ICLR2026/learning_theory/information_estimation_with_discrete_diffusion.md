---
title: >-
  [论文解读] Information Estimation with Discrete Diffusion
description: >-
  [ICLR 2026][学习理论][互信息估计] 提出 INFO-SEDD：把离散扩散（连续时间马尔可夫链）的 score 函数接到 Dynkin 公式上，直接在离散数据上估计 KL 散度、互信息与熵，绕开了"先嵌入到连续空间再估计"的老套路，在高维高互信息场景下显著更准、更稳。 领域现状：互信息（MI）、熵这类信息论度量是…
tags:
  - "ICLR 2026"
  - "学习理论"
  - "信息论估计"
  - "离散扩散"
  - "互信息估计"
  - "熵估计"
  - "连续时间马尔可夫链"
  - "KL 散度"
---

# Information Estimation with Discrete Diffusion

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=m18MXVdrV9](https://openreview.net/forum?id=m18MXVdrV9)  
**代码**: [https://github.com/AlbertoForesti/mutinfo-diffusion](https://github.com/AlbertoForesti/mutinfo-diffusion)  
**领域**: 学习理论 / 信息论估计 / 离散扩散  
**关键词**: 互信息估计, 熵估计, 离散扩散, 连续时间马尔可夫链, KL 散度  

## 一句话总结
提出 INFO-SEDD：把离散扩散（连续时间马尔可夫链）的 score 函数接到 Dynkin 公式上，直接在离散数据上估计 KL 散度、互信息与熵，绕开了"先嵌入到连续空间再估计"的老套路，在高维高互信息场景下显著更准、更稳。

## 研究背景与动机
**领域现状**：互信息（MI）、熵这类信息论度量是刻画变量间非线性关系的核心工具，广泛用于机器学习训练目标、模型选择、神经科学、基因组学等场景。神经估计器（MINE、NWJ、SMILE、F-DIME 等）近年取代了经典参数/非参数方法，但它们绝大多数针对**连续分布**设计。

**现有痛点**：真实世界里大量数据是**离散且高维**的（DNA 序列、文本 token、Ising 自旋）。处理这类数据的通行做法是"嵌入技巧"——先把离散数据投影到连续空间，再套用连续估计器。这条路有三个硬伤：(1) 需要为嵌入模型和估计器架构做大量针对性工程；(2) 嵌入可能丢掉数据本身的离散结构；(3) 基于变分下界的估计器在**高 MI 区**会失效——所需样本量随真实 MI 指数增长（McAllester & Stratos 的著名负面结论），实践中 MI 估计被批大小的对数 $\log(\text{batch size})$ 卡住天花板。

**核心矛盾**：信息论度量的应用价值（基因测序、文本摘要、神经科学）与"缺乏可扩展、能直接吃离散数据的高维估计器"之间的鸿沟。连续域的强估计器（如 MINDE）一搬到离散数据上就崩。

**本文目标**：造一个**直接在离散空间工作**、可扩展、统计一致、且能复用预训练模型的 MI/熵估计器。

**核心 idea**：**信息估计 = 生成建模的副产品**。离散扩散模型（SEDD/掩码扩散）训练出的 score 函数本就编码了不同时刻的概率分布信息；把它代进由 Dynkin 公式推出的 KL 积分公式，就能把 MI、熵这些量算出来——**训练一个生成模型，顺手就拿到了估计器**。

## 方法详解

### 整体框架
INFO-SEDD 把信息估计完全建立在连续时间马尔可夫链（CTMC）的时间反演框架上。先构造两条只在初始分布不同的 CTMC，用 Dynkin 公式把 $\mathrm{KL}[\vec{p}_0\|\vec{q}_0]$ 写成一个对时间积分的期望，积分核里出现的概率比值正好可以用离散扩散的 score 模型 $s_\theta$ 来近似；于是 KL、MI、熵全部归约成"采样时刻 + 模拟前向过程 + 查 score"的蒙特卡洛估计。落地时再借**吸收态（absorbing）转移矩阵**这个关键选型，把"本来要训两个 score 模型"压缩成"只训一个联合模型"，并支持直接接管预训练掩码扩散模型。

```mermaid
flowchart TD
    A[离散数据 X, Y] --> B[构造两条 CTMC<br/>仅初始分布不同 p0 vs q0]
    B --> C[Dynkin 公式 + 后向算子<br/>把 KL 写成时间积分期望]
    C --> D[score 模型 s_theta 近似<br/>概率比值 p_t·x/p_t·Xt]
    D --> E{吸收态转移矩阵选型}
    E -->|单模型出边际 score| F[INFO-SEDD-J 联合法<br/>KL pXY‖pX⊗pY]
    E -->|条件分布建模| G[INFO-SEDD-C 条件法<br/>E·KL pY|X‖pY]
    F --> H[蒙特卡洛积分 → MI / 熵估计]
    G --> H
```

### 关键设计

**1. 用 Dynkin 公式把 KL 散度变成可估的时间积分**：这是整套方法的数学地基。给定同支撑上的两个分布 $\vec{p}_0,\vec{q}_0$，构造两条共享生成元、只是初值不同的 CTMC。利用反演过程在终点都收敛到同一参考分布 $\pi$ 这一性质，KL 可写成对反演轨迹的期望。再套 Dynkin 公式（对函数 $f$ 有 $\mathbb{E}[f(\overleftarrow{X}_T,T)|\overleftarrow{X}_0]-f(\overleftarrow{X}_0,0)=\mathbb{E}[\int_0^T \partial_t f + \mathcal{B}[f]\,dt]$，其中后向算子 $\mathcal{B}[f](a,t)=\sum_{b\neq a}\overleftarrow{Q}_t(b,a)(f(b)-f(a))$），最终把 KL 整理成只依赖**概率比值** $\frac{\vec{p}_t(x)}{\vec{p}_t(\vec{X}_t)}$ 的积分式，核函数 $K(a)=a(\log a-1)$。这一步的妙处在于：KL 不再需要显式知道分布，只需要知道相邻状态的概率比值——而这正是离散扩散 score 函数的定义。

**2. 用离散扩散 score 替换不可知的概率比值**：积分式里的真实比值 $\frac{\vec{p}_t(x)}{\vec{p}_t(\vec{X}_t)}$ 当然算不出来，于是用 SEDD 的参数化 score $s_\theta^p(\vec{X}_t)_x$ 替换，得到可计算估计量
$$\mathrm{KL}[\vec{p}_0\|\vec{q}_0]\approx\mathbb{E}\Big[\int_0^T\!\!\sum_{x\neq\vec{X}_t}\!\vec{Q}_t(\vec{X}_t,x)\big(K(s_\theta^p)+s_\varphi^q - s_\theta^p\log s_\varphi^q\big)dt\Big].$$
score 通过 SEDD 原生的 DWDSE（Diffusion Weighted Denoising Score Entropy）损失训练。蒙特卡洛实现极其直接：在 $[0,T]$ 均匀采时刻 $t$、模拟前向过程 $\vec{X}_t$、查 score 即可。MI 有两种表达对应两个变体——**联合法 INFO-SEDD-J**（$I(X,Y)=\mathrm{KL}[p_{XY}\|p_X\otimes p_Y]$）和**条件法 INFO-SEDD-C**（$I(X,Y)=\mathbb{E}[\mathrm{KL}[p_{Y|X}\|p_Y]]$），后者在标签维度远低于序列维度时（如 DNA→标签）优化更容易。

**3. 吸收态转移矩阵：一个模型算尽所有边际 score**：朴素实现里转移矩阵 $\vec{Q}_t$ 的规模随状态空间 $|\chi|^2$ 爆炸。借助"序列可分解成 $D$ 个子分量"的结构，约束 CTMC 每步只改一个分量（单位 Hamming 距离），转移由共享的局部矩阵 $\vec{Q}^{tok}$ 决定，复杂度大降。更关键的是选 $\vec{Q}^{tok}_t=\sigma(t)\vec{Q}^{tok}_{absorb}$ 这种**吸收态**矩阵：子分量只能转入吸收态 $\varnothing$。这一选型让"联合分布上训练的单个 score 模型"也能直接读出边际 score（当 $Y$ 全被吸收为 $\varnothing$ 时，联合 score 比值自动退化为 $X$ 的边际 score 比值），从而把本需两个模型的 MI 估计压成单模型，并天然兼容预训练掩码扩散模型（MDLM、Caduceus、MD4、LLaDA）。

**4. 一致性与误差分解的理论保证**：在 score 有界（常数 $C_1,C_2$）和网络近似误差 $\epsilon_p,\epsilon_q$ 的温和假设下，估计偏差被分解成两项：
$$\big|\mathbb{E}\,\mathcal{E}(s_\theta^p,s_\varphi^q)-\mathrm{KL}[p\|q]\big|\le\underbrace{\bar\sigma(T)D|\chi|(1+\tfrac{C_2}{C_1})(\epsilon_p+\epsilon_q)}_{\text{估计误差}}+\underbrace{(1-\vec{p}_T(\varnothing^D))DC_2\log|\chi|}_{\text{截断偏差}}.$$
估计误差随 score 误差线性增长；截断偏差源于有限时间 $T$，随吸收态概率 $\vec{p}_T(\varnothing^D)\to1$ **指数衰减**。因此 INFO-SEDD 是一致估计器（差一个指数小的截断项），关键是它**没有重要性采样估计器那种随 MI 指数爆炸的方差**——这正是它在高 MI 区碾压变分方法的根因。熵估计也被纳入同一框架：$H(\vec{p}_0)=\log N-\mathrm{KL}[\vec{p}_0\|\vec{u}_0]$，对均匀分布做 KL 即可。

## 实验关键数据

### 主实验：高维合成基准（已知真值 MI）
所有方法用同一骨干网络、$10^5$ 样本、批大小 1024、训练 $10^5$ 步，10 个种子取均值±标准差。

| Estimator | MI=10,D=10 | MI=20,D=20 | MI=30,D=30 | MI=40,D=40 | MI=50,D=50 |
|-----------|-----------|-----------|-----------|-----------|-----------|
| **INFO-SEDD** | **9.92±0.12** | **20.02±0.21** | **29.83±0.54** | **39.11±0.65** | **47.77±1.18** |
| GAN-DIME | 12.15±0.89 | 22.09±1.75 | 20.74±1.75 | 19.64±1.33 | 17.27±1.46 |
| MINDE | 14.01±2.91 | 26.98±3.16 | 31.08±4.33 | 33.97±3.32 | 32.60±3.93 |
| SMILE | 12.83±0.95 | 23.11±1.41 | 21.79±1.08 | 20.13±1.27 | 18.97±1.05 |
| MINE | 10.21±6.33 | 8.82±0.80 | 7.41±1.23 | 6.91±0.66 | 7.21±1.14 |
| KL-DIME | 8.38±0.90 | 7.51±0.56 | 7.02±0.43 | 6.52±0.32 | 6.41±0.62 |

随着 MI 和维度同步升高，所有竞争者要么严重低估（变分法被 $\log(\text{batch})$ 卡死）、要么剧烈高估方差（MINDE/MINE），唯独 INFO-SEDD 始终贴近真值且标准差极小。

### 下游应用：文本摘要的人类指标对齐（Pearson 相关）
在 SUMMEVAL 上估计模型摘要与原文的 MI，与人类评分相关性：

| 方法 | 连贯性 COH | 一致性 CON | 流畅性 FLU | 相关性 REL | 总体 OVR |
|------|----------|----------|----------|----------|---------|
| **INFO-SEDD-C** | 0.209 | **0.740** | **0.679** | **0.411** | **0.568** |
| INFO-SEDD-J | -0.091 | 0.550 | 0.455 | 0.288 | 0.322 |
| KL-DIME | 0.170 | 0.214 | 0.194 | 0.076 | 0.193 |
| HD-DIME | -0.243 | 0.331 | 0.281 | -0.145 | 0.063 |
| SMILE | -0.367 | -0.074 | -0.162 | -0.149 | -0.221 |

MI 与"一致性（consistency，衡量摘要被原文蕴含的程度）"相关性最高（0.740），符合直觉——一致性正是文本与摘要共享信息量的体现。

### 关键发现
- **一致性测试（文本/基因组）**：把 BART 摘要以概率 $\rho$ 配真原文、$1-\rho$ 配随机文本，理论上 MI 应随 $\rho$ 线性增长；INFO-SEDD 两个变体的曲线最贴合经验推导（256–303 nats 区间），变分法因 $\log(\text{batch})$ 上限严重低估，MINDE 因高维嵌入完全失效。
- **基因组 motif 发现**：在 Arabidopsis thaliana 启动子序列上，用滑动窗口 + 掩码估计 MI 曲线，INFO-SEDD-J 能精准定位 TATA-BOX 基序（-39 到 -26 区间 MI 显著升高），且单次训练即可估计任意子序列子集的 MI，对相关基序鲁棒（逐个解蔽而不受其他基序干扰）。
- **样本效率与收敛**：合成实验消融显示 INFO-SEDD 仅用 $10^3$ 样本即准确，对支撑大小 $|\chi|$ 鲁棒，且收敛比 GAN-DIME/SMILE 更快。

## 亮点与洞察
- **范式统一**：把"信息论估计"和"生成建模"焊在一起——训练离散扩散模型本身就产出了估计器，score 函数一物两用，无需额外的估计器架构工程。
- **绕过维度诅咒的根因**：变分下界估计器在高 MI 区的指数样本复杂度是结构性缺陷；INFO-SEDD 走 KL 积分 + 一致估计这条路，方差不随 MI 爆炸，理论上界给得很干净。
- **吸收态选型是工程画龙点睛**：用一个数学技巧（吸收态让边际成为联合的特例）把"训两个模型"省成"训一个模型"，且无缝继承预训练掩码扩散模型，复用性极强。
- **C 变体 vs J 变体的实践智慧**：当一侧维度远低于另一侧（DNA→二分类标签），条件法只需建模低维标签的边际/条件 score，优化难度骤降，这个观察对落地很有指导性。

## 局限与展望
- **截断偏差依赖吸收态收敛**：一致性保证里那个指数衰减项要求 $\vec{p}_T(\varnothing^D)$ 足够接近 1，意味着时间horizon $T$ 要够长、扩散过程要充分吸收，否则偏差不可忽略。
- **score 模型质量是天花板**：误差界线性依赖网络近似误差 $\epsilon_p,\epsilon_q$，估计精度本质上被离散扩散训练的好坏锁死，复杂分布上 score 难训会直接拖累估计。
- **仍需训练/微调**：虽然能复用预训练模型，但多数场景仍要在目标数据上训练或微调 score 模型，相比即插即用的经典估计器有计算开销。
- **展望**：作者指出借 Generator Matching 框架可扩展到**混合连续/离散数据**，并兼容 MD4、LLaDA 等更强的掩码扩散骨干，在科学发现（基因组、神经科学）场景潜力很大。

## 相关工作与启发
- **离散扩散 / SEDD 谱系**：方法直接站在 Lou et al. (SEDD)、Sahoo et al. (MDLM)、Campbell/Austin 的掩码扩散与 CTMC 工作之上，把生成建模的 score 复用为估计工具。
- **扩散估计器**：延续 Franzese et al. (MINDE)、Kong et al. 用扩散估计信息量的思路，但把战场从连续域搬到离散域，正面解决 MINDE 在离散数据上的失效。
- **变分 MI 估计的负面结论**：McAllester & Stratos、Song & Ermon 关于变分下界高 MI 失效的批判，是本文动机的理论支点，也是它一致估计设计的对照面。
- **启发**：这项工作示范了"把生成模型的训练副产品转化为统计量估计器"的通用思路，对任何已有强生成模型（语言、单细胞 RNA、蛋白序列）的领域，都提示了一条几乎零额外成本的信息论分析路径。

## 评分
- **新颖性**: ⭐⭐⭐⭐⭐ 首个直接在高维离散数据上工作、有一致性保证、可复用预训练掩码扩散模型的 MI/熵估计器，Dynkin + 吸收态的组合干净优雅。
- **实验充分度**: ⭐⭐⭐⭐ 合成基准 + 文本摘要 + 基因组 motif + Ising 熵四类实验覆盖广，对比 8 个强 baseline，但真实任务多为"一致性/相关性"间接验证，缺更多带硬真值的下游评测。
- **写作质量**: ⭐⭐⭐⭐ 数学推导清晰、动机与误差分析讲得透，两个变体的取舍解释到位；公式密度较高，对不熟 CTMC 的读者门槛偏陡。
- **价值**: ⭐⭐⭐⭐⭐ 解决了离散数据信息估计这一长期痛点，对基因组学、NLP 评测等离散密集型领域有直接落地价值，开源代码加分。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] InfoBridge: Mutual Information Estimation via Bridge Matching](infobridge_mutual_information_estimation_via_bridge_matching.md)
- [\[ICLR 2026\] A Unification of Discrete, Gaussian, and Simplicial Diffusion](a_unification_of_discrete_gaussian_and_simplicial_diffusion.md)
- [\[ICLR 2026\] Characterizing the Discrete Geometry of ReLU Networks](characterizing_the_discrete_geometry_of_relu_networks.md)
- [\[ICLR 2026\] Polynomial Convergence of Riemannian Diffusion Models](polynomial_convergence_of_riemannian_diffusion_models.md)
- [\[ICLR 2026\] On the Interpolation Effect of Score Smoothing in Diffusion Models](on_the_interpolation_effect_of_score_smoothing_in_diffusion_models.md)

</div>

<!-- RELATED:END -->
