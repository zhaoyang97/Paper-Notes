---
title: >-
  [论文解读] Interpretability and Generalization Bounds for Learning Spatial Physics
description: >-
  [ICML 2026][优化/理论][神经算子] 论文用数值分析工具证明：在线性 PDE（1D Poisson 等）上学到的解算子 $\mathbf{W}$ 只会收敛到真算子 $\mathbf{A}$ 在训练函数空间上的投影 $\mathbf{A}\mathbf{U}\mathbf{U}^\top$，所以**函数空间本身**——而非数据量或网格细度——决定 OOD 泛化；并提出一种把权重矩阵作用在 one-hot 上即可看出"是否学到 Green 函数结构"的机械可解释技术，用 25×25 跨数据集 cross-evaluation 把 8 类 SciML 模型（含 PINN/DeepONet/FNO/PI-DeepONet）的失败…
tags:
  - "ICML 2026"
  - "优化/理论"
  - "神经算子"
  - "Green 函数"
  - "泛化界"
  - "机械可解释性"
  - "PINN"
---

# Interpretability and Generalization Bounds for Learning Spatial Physics

**会议**: ICML 2026  
**arXiv**: [2506.15199](https://arxiv.org/abs/2506.15199)  
**代码**: 待确认  
**领域**: 科学计算 / SciML / 泛化理论  
**关键词**: 神经算子、Green 函数、泛化界、机械可解释性、PINN  

## 一句话总结
论文用数值分析工具证明：在线性 PDE（1D Poisson 等）上学到的解算子 $\mathbf{W}$ 只会收敛到真算子 $\mathbf{A}$ 在训练函数空间上的投影 $\mathbf{A}\mathbf{U}\mathbf{U}^\top$，所以**函数空间本身**——而非数据量或网格细度——决定 OOD 泛化；并提出一种把权重矩阵作用在 one-hot 上即可看出"是否学到 Green 函数结构"的机械可解释技术，用 25×25 跨数据集 cross-evaluation 把 8 类 SciML 模型（含 PINN/DeepONet/FNO/PI-DeepONet）的失败模式逐个标出来。

## 研究背景与动机

**领域现状**：把 ML 用到科学计算里有两条主流路线——白盒（SINDy、符号回归，给闭式公式）和黑盒（DeepONet、Fourier Neural Operator、PINN，灵活但不可解释）。中间还有一类"物理感知"模型（PINN、PI-DeepONet）通过把 PDE 损失放进训练来注入先验。这些模型在自己设计的训练分布上往往能拿到机器精度级别的 MSE。

**现有痛点**：低训练误差**不**等于学到了正确的物理。已有工作零星地观察到 Neural ODE 会过拟合时序、PINN 在某些训练策略下会失败、PINO 在跨分辨率时崩溃——但这些都是"经验现象"，缺少能解释**为什么失败**的理论刻画；更没有一种统一框架去同时评价"参数学习 + 算子学习 + 物理感知"三大类模型。

**核心矛盾**：传统 ML 直觉认为"数据更多、表达力更强"应该单调改善泛化；而经典数值分析告诉我们近似误差由**离散化阶数和函数空间**决定。两者在 SciML 上正面冲突——本文要把数值分析的 a priori estimate 搬进 ML，刻画清楚冲突的边界。

**本文目标**：（1）在最简单的 1D Poisson 方程上对参数拟合和线性算子学习给出严格的收敛/泛化界；（2）把"训练函数空间"作为一阶变量纳入分析；（3）提供一种不依赖 loss、只看权重的机械可解释方法，能直观判断模型是否真学到了物理。

**切入角度**：从 Poisson 的 Green 函数 $G(s, x)$ 出发——它是 PDE 解算子的"标准答案"。如果学到的矩阵 $\mathbf{W}$ 真在逼近 $\mathbf{A} = \int G \psi$ 的离散化，那么 $\mathbf{W} \mathbf{e}_j$ 应该长得像 Green 函数的 impulse response。这给了一个既能做理论分析又能做可视化诊断的统一抓手。

**核心 idea**：把训练数据建模成"采样函数空间 $\mathcal{F}(\mathrm{type}, p)$"的随机过程，证明 GD 在线性模型上的解 $\mathbf{W}^*$ 是真算子向训练空间正交投影的结果，并把这个投影残差作为先验泛化界；再用 $\mathbf{W} \mathbf{e}_j$（或非线性模型上的 $\mathrm{Model}(\mathbf{e}_j)$）作为"Green 函数提取器"做机械可解释检查。

## 方法详解

### 整体框架
研究对象是 1D Poisson 方程 $-k \, d^2 u / dx^2 = f(x)$ 在 $[0,1]$ 上配齐次 Dirichlet 边界，对应的解算子可由 Green 函数 $G(s, x)$ 表出。论文构造了 25 个数据集——多项式 / 正弦 / 余弦 $\mathcal{F}(\mathrm{type}, p)$ ($p = 1..8$) 加一个 FEM 分段线性，每个 10,000 例——然后训练 8 类模型并做 25×25 的 cross-evaluation（行=训练集，列=测试集），产出错误矩阵热图。

理论侧聚焦两个有解析解的设置：
- **设置 A**（参数拟合 + 已知 PDE 结构）：固定有限差分 stencil 阶 $q$，只学标量 $w \approx k$；
- **设置 B**（黑盒线性算子）：学整个矩阵 $\mathbf{u} = \mathbf{W} \mathbf{f}$。

经验侧把同一框架推广到 deep linear / MLP / DeepONet / FNO / PINN / PI-DeepONet 等无解析解的架构。

### 关键设计

**1. 有限差分参数学习的 a priori 估计（Theorem 3.1）：揭穿"数据越丰富越好"的直觉**

第一个有解析解的设置是已知 PDE 结构、只学一个标量参数 $w\approx k$。用 $q$ 阶 stencil（如三点 FD-2，$d^2 u/dx^2 \approx (u_{i-1}-2u_i+u_{i+1})/\Delta x^2 + \mathcal{O}(\Delta x^q)$）去解 MSE 最小化，论文证明：当训练多项式阶 $p<q$ 时 $w=k$ 精确；一旦 $p\geq q$ 就出现不可约偏置

$$\frac{|w-k|}{|k|} = \mu_q \Delta x^q + \sum_{m=q+1}^p \mu_m \Delta x^m \approx \mu_q \Delta x^q$$

其中 $\mu_m$ 是 stencil 的截断误差系数。这个结论直接打脸 ML 直觉——每加一阶高于 stencil 阶的多项式数据，就再叠一项 $\mathcal{O}(\Delta x^m)$ 的偏置，因为高阶多项式让 stencil 的截断误差有机会被"吸进" $w$ 里。这与有没有无限数据无关，是离散化阶导致的硬天花板，而且在 PINN 的 inverse problem 上观察到同样趋势。

**2. 线性算子的子空间投影定理（Theorem 3.2）：把"训练函数空间"写进泛化界**

第二个设置是黑盒线性模型 $\mathbf{u}=\mathbf{W}\mathbf{f}$。设训练 forcing 由 $\mathbf{f}^{(n)}=\mathbf{B}\mathbf{c}^{(n)}$ 采样（$\mathbf{B}$ 形如 Vandermonde，秩 $p+1$），各坐标零均值独立，则零均值初始化下 GD 的极限是

$$\mathbf{W}^* = \mathbf{A}\,\mathbf{U}\mathbf{U}^\top + \mathbf{W}^0(\mathbf{I}-\mathbf{U}\mathbf{U}^\top)$$

其中 $\mathbf{U}$ 是 $\mathbf{B}$ 的左正交基。这是一个异常悲观的结果：收敛点与数据量、网格细度都无关，只取决于训练空间的秩——当且仅当 $\dim\mathcal{F}_{\mathrm{train}}\geq\mathrm{rank}(\mathbf{A})$ 才有机会学到真算子，否则 $\mathbf{W}$ 永远只是 $\mathbf{A}$ 在训练空间上的投影、正交方向上残留任意初始噪声。它干净地解释了 Fig. 1 的反直觉现象（训练误差到机器精度、但矩阵和真 $\mathbf{A}$ 差很远），而且预测 MSE 下界就是 $\|\mathbf{A}-\mathbf{A}\mathbf{U}\mathbf{U}^\top\|_F^2$，可由 $\mathbf{B}$ 的 SVD 直接算出来验证。

**3. Green 函数机械可解释探针：不看 loss，喂 one-hot 看响应**

训练/测试 MSE 没法区分"过拟合到某个函数空间"和"真学到了算子"——两种情况在训练集上 loss 都能一样低。这里给出一个正交于 loss 的诊断：因为对任何把 forcing 映成 solution 的模型都有 $\mathbf{A}_{ij}\leftrightarrow\mathrm{Model}(\mathbf{f}=\mathbf{e}_j)_i$，所以只要喂 one-hot $\mathbf{f}=\mathbf{e}_j$ 就能把模型"扫描"成一个矩阵：线性模型直接看权重列，MLP/DeepONet/FNO 这类非线性算子就喂 25 个 one-hot 把响应画出来。学得对的话，列向量长得像 Green 函数的 impulse response，整体呈现接触状的分段线性结构；学不对就是噪声。还能进一步求逆 $\hat{\mathbf{L}}=\mathbf{W}^{-1}$ 看是否恢复出三对角的局部 stencil。这套思路类似在 DINO 里看 attention map——训得动不等于学得对，能不能在 impulse response 上看出 PDE 结构是更高的 bar。

### 实验与交叉验证协议
作者引入**function-space cross-evaluation**：用 25 个数据集分别训练同一模型，再在另外 24 个上测试，形成 25×25 MSE 热图。差异既来自网格 $\Delta x$ 也来自函数类 $\mathcal{F}(\mathrm{type}, p)$。这构成新的 SciML benchmark，"OOD" 的定义从"换分布"严格化为"换函数子空间"。模型覆盖 5 大类共 8 种：有限差分 + PINN（参数拟合）；线性 / deep linear / MLP（黑盒）；DeepONet / FNO（SciML 黑盒）；PI-DeepONet（物理感知）。

## 实验关键数据

### 主实验（25×25 OOD cross-evaluation 的关键发现）

| 模型族 | 是否 subspace generalize | 训练 MSE 量级 | OOD 失败方式 |
|---|---|---|---|
| 线性模型 $\mathbf{u} = \mathbf{W}\mathbf{f}$ | 是（三块下三角块，与 Thm 3.2 一致） | $\sim 10^{-20}$ | 跨函数族（poly→sin 等）训练分布外失败 |
| Deep Linear | 部分（sin/cos 上有，poly 上无） | 中等 | 子空间外不一致 |
| MLP | 否，强对角 | 中等 | 几乎不泛化，纯过拟合 |
| 有限差分参数拟合 | — | 随 $p$ 升高而升 | 训多项式阶越高、$w$ 越偏（Thm 3.1） |
| PINN inverse problem | — | 与 FD 同趋势 | 训练阶 $p$ ↑ 则 $w$ 误差 ↑ |
| DeepONet | 块下三角但对角偏低 | 较低 | 训练分布上轻微过拟合 |
| FNO | 类似 DeepONet | 不稳定，部分函数类训不下来 | — |
| PI-DeepONet | 块下三角但 baseline error 高 | $\sim 10^{-6}$ | PDE loss 抬高 floor，但不消除子空间限制 |

> 关键反差：训练误差差 $10^{14}$ 数量级（$10^{-20}$ vs $10^{-6}$），OOD 跨子空间时所有模型都跳到 $10^{-2}$ 同一量级——证明**测试空间是否落在训练子空间里，比模型复杂度更决定性**。

### 鲁棒性 & 推广实验

| 设置 | 现象 | 结论 |
|---|---|---|
| 加测量噪声（Fig. 7） | floor 从 $10^{-20}$ 抬到 $10^{-9}$，但下三角块结构不变 | 噪声只抬地板不模糊边界，子空间泛化定理仍成立 |
| 1D Biharmonic $-k d^4 u/dx^4 = f$（Fig. 8a） | 同样三块下三角结构 | Thm 3.2 不限于 Poisson |
| 2D Poisson 用张量积基（Fig. 8b） | Sierpiński-三角形状的下三角图案 | 子空间泛化在每个空间维度独立成立 |
| FEM 分段线性数据训线性模型 | 学到完整 $\mathbf{A}$，求逆得到三对角 stencil | "数据空间稠密"是学到真算子的充分条件 |

### 关键发现
- **不可约偏置的反直觉来源**：在参数学习里，每加一阶高于 stencil 阶 $q$ 的多项式都让误差更大（Thm 3.1），与 "数据越丰富越好" 的 ML 经验完全相反——因为高阶多项式让截断误差有机会"伪装"成可调参数。
- **物理感知 ≠ 物理正确**：PI-DeepONet 把 PDE 写进 loss，却没有更好的子空间外泛化，只是把训练误差地板从机器精度抬到 $10^{-6}$。先验项防不住函数空间限制。
- **FEM 数据是黄金通行证**：所有黑盒模型在 FEM 分段线性训练数据下都展现最广泛的跨子空间泛化，因为分段线性基足够稠密、能跨越其它函数族的子空间。这给出了 SciML 数据采集的实用指导。

## 亮点与洞察
- **把"函数空间"提升到一阶变量**：传统 OOD 分析关注分布偏移、噪声、协变量漂移；本文把"训练分布所张的函数子空间"显式写进泛化界，给出 $\mathbf{A} \mathbf{U} \mathbf{U}^\top$ 这个干净的闭式，能直接被实验验证。
- **三块下三角热图是新的 SciML benchmark**：这种 25×25 cross-evaluation 比单个测试 MSE 信息量大得多，建议未来 SciML 论文都该跑这种图。
- **Green 函数探针的可迁移性**：核心思想"喂 one-hot，看响应"对任意把空间函数映射成空间函数的模型都适用——把它推广到时间算子、3D 算子、attention map 解读都很自然。
- **悲观结论的建设性**：理论给的不是 negative result，而是"如何采集训练数据"的建设性指导——选择分段线性或宽函数族（如稠密多项式高阶）做训练集，可以本质上扩张可学子空间。

## 局限与展望
- **限定线性 PDE**：Theorem 3.2 严格只对线性算子成立，非线性 PDE（如 Navier-Stokes）的子空间投影界还没有；作者只能用经验观察推广。
- **限定 1D / 张量积 2D**：高维非分离 PDE 上 Green 函数离散化会爆炸，$\mathbf{A}$ 的存储和分析都受限。
- **没有分析正则化 / 优化器选择**：实验都是零初始化 + GD，但实际工程里大量用 Adam + weight decay + dropout，是否改变子空间投影行为没回答。
- **改进方向**：把"子空间秩 ≥ 算子秩"推广为非线性算子的局部线性化条件、把 cross-evaluation 协议扩到时间算子（Neural ODE）、研究主动学习如何选最小的训练函数集去覆盖目标算子。

## 相关工作与启发
- **vs Boullé 等 (Green-function ML)**：他们用 Green 函数构造解算子或证明椭圆 PDE 的数据需求下界；本文把 Green 函数同时用作**理论锚** + **可视化探针**两个角色。
- **vs Krishnapriyan PINN failure 系列**：那些工作说明 PINN 在某些训练策略下失败；本文给出了更结构化的解释——PINN 的 PDE loss 不能扩张子空间，所以仍受 Thm 3.2 约束。
- **vs Sakarvadia PINO 跨分辨率失败**：本文明确**不**研究跨分辨率，而是跨函数空间，是正交但互补的角度。
- **vs Nanda 等 mechanistic interpretability**：他们在 Transformer 上发现可识别算法；本文把"看权重 → 找已知算子"的方法论搬到 SciML，Green 函数是这里的"已知算法"。
- **vs ReNO / RINO**：那些做 discretization-invariant 架构；本文说就算分辨率不变，function space 不匹配仍会崩，是设计 invariant 架构之外更基础的问题。
- **启发**：可以把"function-space cross-evaluation"作为评测 LLM in-context learning 泛化能力的范式——把 prompt 的"任务空间"看成 $\mathcal{F}_{\mathrm{train}}$，测试集去测真正不同 task-subspace 上的泛化。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ — 把数值分析的 a priori estimate 严格搬进 ML 解算子的泛化分析，并给出实验可证伪的闭式预测。
- 实验充分度: ⭐⭐⭐⭐⭐ — 8 类模型 × 25 数据集 × 噪声 / 2D / 双调和扩展，加 Green 函数探针可视化，覆盖罕见地全面。
- 写作质量: ⭐⭐⭐⭐⭐ — Fig. 1 的"训练精度高 vs. 真算子差远"对照极其有冲击力，理论与实验穿插推进。
- 价值: ⭐⭐⭐⭐⭐ — 给出"如何采集 SciML 训练数据"的直接指导，并把 cross-evaluation 协议立为新 benchmark，对实践和理论两端都有持续影响。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Pseudospectral Bounds for Transient Amplification in Coupled Gradient Descent](pseudospectral_bounds_for_transient_amplification_in_coupled_gradient_descent.md)
- [\[ICML 2026\] Taming the Loss Landscape of PINNs with Noisy Feynman-Kac Supervision: Operator Preconditioning and Non-Asymptotic Error Bounds](taming_the_loss_landscape_of_pinns_with_noisy_feynman-kac_supervision_operator_p.md)
- [\[NeurIPS 2025\] Learning at the Speed of Physics: Equilibrium Propagation on Oscillator Ising Machines](../../NeurIPS2025/optimization/learning_at_the_speed_of_physics_equilibrium_propagation_on_oscillator_ising_mac.md)
- [\[ICML 2025\] A Generalization Result for Convergence in Learning-to-Optimize](../../ICML2025/optimization/a_generalization_result_for_convergence_in_learning-to-optimize.md)
- [\[ICLR 2026\] High Probability Bounds for Non-Convex Stochastic Optimization with Momentum](../../ICLR2026/optimization/high_probability_bounds_for_non-convex_stochastic_optimization_with_momentum.md)

</div>

<!-- RELATED:END -->
