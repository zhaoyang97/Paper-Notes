---
title: >-
  [论文解读] A Schrödinger Eigenfunction Method for Long-Horizon Stochastic Optimal Control
description: >-
  [ICLR2026][优化/理论][随机最优控制] 对于「无控漂移是某势函数梯度」这一类随机最优控制（SOC）问题，本文证明其线性化后的 HJB 算子与一个谱纯离散的薛定谔算子酉等价，于是长程最优控制可由该算子的**最大特征函数**直接给出（修正项随时间跨度指数衰减）；据此给出对称 LQR 的闭式解，并提出去掉「隐式重加权」偏差的相对特征函数损失，把长程 SOC 的内存/时间复杂度从 $O(Td)$ 降到 $O(d)$，控制精度提升约一个数量级。
tags:
  - "ICLR2026"
  - "优化/理论"
  - "随机最优控制"
  - "HJB 方程"
  - "薛定谔算子"
  - "特征函数学习"
  - "长程规划"
---

# A Schrödinger Eigenfunction Method for Long-Horizon Stochastic Optimal Control

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=lcEw5NcSij](https://openreview.net/forum?id=lcEw5NcSij)  
**代码**: 待确认  
**领域**: 优化 / 随机最优控制 / 谱方法  
**关键词**: 随机最优控制, HJB 方程, 薛定谔算子, 特征函数学习, 长程规划

## 一句话总结
对于「无控漂移是某势函数梯度」这一类随机最优控制（SOC）问题，本文证明其线性化后的 HJB 算子与一个谱纯离散的薛定谔算子酉等价，于是长程最优控制可由该算子的**最大特征函数**直接给出（修正项随时间跨度指数衰减）；据此给出对称 LQR 的闭式解，并提出去掉「隐式重加权」偏差的相对特征函数损失，把长程 SOC 的内存/时间复杂度从 $O(Td)$ 降到 $O(d)$，控制精度提升约一个数量级。

## 研究背景与动机
**领域现状**：随机最优控制研究的是如何驱动一个由随机微分方程（SDE）刻画的系统，使期望总代价最小，广泛见于分子动力学稀有事件采样、机器人、金融等领域。在「仿射控制」设定下（控制以线性方式进入状态），最优控制恰好等于价值函数梯度，对应的 Hamilton-Jacobi-Bellman（HJB）方程可被大幅简化。高维问题无法用网格 PDE 求解器（维数灾难），主流是用神经网络去解 HJB，分两条路线：前向-后向 SDE（FBSDE）方法，和迭代扩散优化（IDO，靠模拟受控轨迹 + 自动微分更新参数）。

**现有痛点**：这些方法在**时间跨度 $T$ 变长时全面变差**。内存和单步运行时间至少随 $T$ 线性增长；理论上 FBSDE 的误差估计随 $T$ 恶化，IDO 用重要性采样时权重方差可能随 $T$ 指数爆炸。换句话说，长程规划（long-horizon）是这一族方法的共同短板，作者在实验里也复现了这种随 $T$ 退化的现象。

**核心矛盾**：现有方法都把 SOC 当成一个**沿时间轴往回滚的动态规划**问题来解，于是计算量天然绑死在 $T$ 上。但直觉上，当离终止时刻还很远（$t \ll T$）时，最优控制应当趋于某种「稳态」，几乎与 $T$ 无关——现有方法没有利用这一点。

**本文目标**：找到一个子类 SOC，使长程控制能被一个**与 $T$ 无关的稳态对象**刻画，从而摆脱「计算量随 $T$ 增长」的诅咒。

**切入角度**：在仿射控制下，HJB 可经 Cole–Hopf 变换 $\psi := \exp(-V)$ 线性化为 $\partial_t\psi = L\psi$，最优控制 $u^* = \partial_x\log\psi$。线性 PDE 的解可以借用有限维线性 ODE 的思路——把初值在算子特征基上展开，再让指数算子作用在特征值上。若 $L$ 有**离散谱**，$\psi$ 就能写成特征函数级数 $\psi_\tau=\sum_i e^{-\lambda_i\tau}\langle\phi_i,\psi_0\rangle\phi_i$，其中最低特征值对应的「最大特征函数」$\phi_0$ 主导长程行为。

**核心 idea**：在「漂移为梯度场」的假设下，$L$ 与一个谱纯离散的薛定谔算子 $S=-\Delta+V$ **酉等价**，从而保证了离散谱的存在；于是长程最优控制可由单个最大特征函数 $\phi_0$ 给出，修正项随 $T-t$ 指数衰减——把一个「随 $T$ 滚动」的控制问题，换成一个「与 $T$ 无关」的特征值问题。

## 方法详解

### 整体框架
本文的方法是一条「理论约化 + 数值求解」的链条。输入是一个仿射控制 SOC 问题：SDE 动力学 $dX_t^u=(b(X_t^u)+\sigma u)\,dt+\sqrt{\beta^{-1}}\sigma\,dW_t$，代价泛函含运行代价 $f$ 与终止代价 $g$；输出是高维、长程下的最优反馈控制 $u^*(x,t)$。

整条链路分四步：（1）**线性化**——用 Cole–Hopf 型变换把非线性 HJB 方程化成由线性算子 $L$ 主导的线性 PDE；（2）**谱验证**——在「漂移为梯度场」假设下证明 $L$ 酉等价于薛定谔算子，从而拥有离散谱、存在正交特征基，且最低特征值单重、对应特征函数严格为正（即「最大特征函数」$\phi_0$）；（3）**特征函数控制**——证明最优控制由特征系统给出，长程部分仅由 $\phi_0$ 决定，修正项指数衰减，于是只需学好 $\phi_0$（对称 LQR 甚至有闭式解）；（4）**混合求解**——远离终止时刻直接用 $\phi_0$ 给出的控制，接近终止时刻（短程）切回已有的 FBSDE/IDO 求解器补一个修正项，使总计算量只与短程区间挂钩。

### 关键设计

**1. Cole–Hopf 线性化 + 薛定谔酉等价：把 SOC 变成离散谱特征值问题**

痛点在于：HJB 方程一般是非线性的，难以利用「特征展开」这类线性结构，长程下又只能逐步往回滚。作者先做参数化 $V(x,t)=-\beta^{-1}\log\psi$，让 HJB 的非线性项相消，得到线性 PDE
$$\partial_\tau\psi + L\psi = 0,\quad L\psi = -\mathrm{Tr}(\sigma\sigma^\top\nabla^2\psi) - 2\beta\, b^\top\nabla\psi + 2\beta^2 f\cdot\psi,$$
其中 $\tau=(2\beta)^{-1}(T-t)$，最优控制 $u^*=\partial_x\log\psi$。关键的一步是假设 **(A1) 漂移是某势函数的梯度** $b=-\nabla E$。在加权空间 $L^2(\mu)$（$\mu(x)=e^{-2\beta E(x)}$）上，$L$ 变得对称，并且存在酉算子 $U:\psi\mapsto e^{-\beta E}\psi$ 使得
$$ULU^{-1} = -\Delta + \beta^2\|\nabla E\|^2 - \beta\Delta E + 2\beta^2 f =: S = -\Delta + \mathcal V.$$
也就是说 $L$ 与量子力学里研究透彻的薛定谔算子 $S$ 酉等价。在温和条件 (A2)（有效势 $\mathcal V$ 局部 $L^2$、下有界、在无穷远趋于 $+\infty$）下，$S$（从而 $L$）拥有可数正交特征基、特征值下有界且无有限聚点、最低特征值单重、对应特征函数可取严格正——这正是把 SOC 化为特征值问题所需的全部谱性质。这一步是全文的理论地基：它不是「借物理做比喻」，而是用酉等价把控制算子的谱问题**精确归约**到一个已被量子力学完整刻画的算子上。

**2. 特征函数控制公式与对称 LQR 闭式解：长程控制由单个 $\phi_0$ 决定**

有了离散谱，作者证明（Theorem 3）最优控制可写成特征系统的级数：
$$u^*(x,t)=\beta^{-1}\Big(\nabla\log\phi_0(x) + \nabla\log\big(1+\textstyle\sum_{i>0}c_i\, e^{-\frac{1}{2\beta}(\lambda_i-\lambda_0)(T-t)}\tfrac{\phi_i(x)}{\phi_0(x)}\big)\Big),$$
其中 $\lambda_0<\lambda_1\le\cdots$。当 $t\ll T$ 时，所有 $i>0$ 的修正项以速率 $\lambda_i-\lambda_0$（谱隙）指数衰减，于是**长程最优控制就是 $\beta^{-1}\nabla\log\phi_0$**——一个与 $T$ 无关的量。这正面解决了「计算量绑死在 $T$ 上」的痛点：长程部分根本不需要随 $T$ 滚动，只要算出一个最大特征函数。

进一步，当 $E$、$f$ 都是二次型时（对称 LQR），对应的薛定谔算子恰好是**量子谐振子的哈密顿量**，其特征值与特征函数有经典闭式解（用厄米多项式 $H_{\alpha_i}$ 表示）：
$$\phi_\alpha(x)\propto \exp\!\big(-\tfrac{\beta}{2}x^\top(-A+U^\top\Lambda^{1/2}U)x\big)\prod_i H_{\alpha_i}\big(\sqrt{\beta}(\Lambda^{1/4}Ux)_i\big),\quad \lambda_\alpha=\beta\big(-\mathrm{Tr}(A)+\textstyle\sum_i\Lambda_{ii}^{1/2}(2\alpha_i+1)\big).$$
结合 Theorem 3，这给出对称线性漂移 + 二次运行代价 + **任意终止代价**下的闭式最优控制——而经典 LQR 解要求终止代价也是二次的，本文去掉了这一限制。

**3. 相对特征函数损失：消除已有损失的隐式重加权偏差**

对一般梯度漂移没有闭式解，需用神经网络学 $\phi_0$。已有两类损失都有致命缺陷。作者把特征函数参数化为 $\phi=\exp(-\beta V_0)$（$V_0$ 是网络，天然保证 $\phi>0$，与 $\phi_0$ 严格正一致）。代入后可算出，PINN 残差损失 $\|L\phi-\lambda\phi\|^2$ 和变分/Ritz 损失 $\langle\phi,L\phi\rangle$ 都会带上一个 $e^{-\beta V_0}$ 因子：
$$R^\rho_{\mathrm{PINN}}(e^{-\beta V_0}) = 4\beta^4\big\|e^{-\beta V_0}(\mathcal K V_0-\tfrac{\lambda_0}{2\beta^2})\big\|_\rho^2 + \alpha R_{\mathrm{reg}}.$$
在 $V_0$ 很大的区域（即价值函数大、最该控制好的区域）这个因子趋于 0，损失对这些区域的误差**视而不见**，导致控制在高代价区崩坏（论文 Figure 3 的 RING 例子里，已有损失学出的控制在高 $V_0$ 区严重偏离切向真值）。作者的修法非常直接——把残差改成**相对残差**：
$$R^\rho_{\mathrm{Rel}}(\phi)=\Big\|\tfrac{L\phi}{\phi}-\lambda\Big\|_\rho^2 + \alpha R_{\mathrm{reg}}(\phi).$$
同样代入 $\phi=e^{-\beta V_0}$ 后，那个使损失「失明」的指数因子被消掉了：$R^\rho_{\mathrm{Rel}}(e^{-\beta V_0})=4\beta^4\|\mathcal K V_0-\tfrac{\lambda_0}{2\beta^2}\|_\rho^2+\alpha R_{\mathrm{reg}}$，于是即便 $\phi$ 很小的区域，损失仍然敏感。实践中先用变分损失训出 $\lambda_0$ 的估计和 $V_0$ 的好初值，再用相对损失微调（作者发现这个初始化对相对损失收敛是必要的）。

**4. 混合求解器：长程用特征函数、短程切回 SDE 求解器，复杂度 $O(Td)\to O(d)$**

特征函数控制在 $t\to 0$（远离终止）表现极好，但在接近终止时刻 $T$ 时不准（因为此时被丢掉的高阶修正项还没衰减）。FBSDE/IDO 则相反——它们在整个 $[0,T]$ 上误差大致恒定，但每步都要模拟一遍 SDE，计算量随 $T$ 线性增长。作者据此取长补短：选一个截断时刻 $T_{\mathrm{cut}}<T$，控制写成
$$u_\theta(x,t)=\begin{cases}\beta^{-1}\nabla\log\phi_0^{\theta_0}, & t\le T_{\mathrm{cut}},\\[2pt]\beta^{-1}\big(\nabla\log\phi_0^{\theta_0}+e^{-\frac{1}{2\beta}(\lambda_1-\lambda_0)(T-t)}v^{\theta_1}(x,t)\big), & T_{\mathrm{cut}}<t\le T.\end{cases}$$
长程段直接用学好的 $\phi_0$；短程段只在区间 $[T_{\mathrm{cut}},T]$ 上跑 IDO/FBSDE 去学一个加性修正网络 $v^{\theta_1}$。由于只需在短程区间模拟 SDE，总时间复杂度从 $O(Td)$ 降到 $O(d)$。$T_{\mathrm{cut}}$ 的选取有一个基于谱隙的经验法则：令修正项 $\exp(-\tfrac{1}{2\beta}(\lambda_1-\lambda_0)(T-T_{\mathrm{cut}}))=\varepsilon$，解出 $T-T_{\mathrm{cut}}=-\tfrac{2\beta}{\lambda_1-\lambda_0}\log\varepsilon$，用经验估计的 $\lambda_0,\lambda_1$ 即可定出量级。

### 损失函数 / 训练策略
训练分两阶段：先以变分损失（deep Ritz）粗训得到 $\lambda_0$ 估计与 $V_0$ 初值，再以相对损失 $R_{\mathrm{Rel}}$ 微调最大特征函数；同时估出前两个特征值 $\lambda_0,\lambda_1$ 用于设定 $T_{\mathrm{cut}}$ 与修正项衰减因子。短程修正网络 $v^{\theta_1}$ 则嵌入标准 IDO/FBSDE 训练流程，只在 $[T_{\mathrm{cut}},T]$ 上优化。

## 实验关键数据

### 主实验
作者在四个高维（$d=20$）长程基准上评测：QUADRATIC（各向同性/各向异性）、DOUBLE WELL、RING，均改造自 Nüsken & Richter (2021) 的短程问题、加长时间跨度并保留可计算的真值。结论是：相对损失在逼近 $\nabla\log\phi_0$ 上显著优于已有特征函数损失；把学到的特征函数嵌入 IDO 的混合算法，在每个设定上的 $L^2$ 控制误差都更低，**通常领先一个数量级**，同时把内存/运行复杂度从 $O(Td)$ 降到 $O(d)$。

误差随 $t\in[0,T]$ 的分布印证了设计动机：纯特征函数法在 $t\to0$（长程）最优、接近 $T$ 时变差；IDO 全程误差恒定；混合法兼具两者之长，整体 $L^2$ 误差最低。

| 方法 | 算法 / 损失 | 控制目标（越小越好） |
|------|-------------|----------------------|
| EIGF（本文） | 相对损失 | **73.33 ± 0.02** |
| IDO | Log-variance | 74.52 ± 0.02 |
| IDO | Adjoint Matching | 74.69 ± 0.02 |
| IDO | Relative Entropy | 75.63 ± 0.02 |
| IDO | SOCM | 未收敛 |

上表为「舆论动力学（De Groot 模型）」任务：$N=10$ 个智能体、状态 $X_t\in\mathbb R^{10}$、对称交互矩阵 $L$、非二次运行代价 $f(x)=\sum_i(x_i^2-1)^2$、终止代价为 0、$T=10$，训练 80,000 步后的最终控制目标。本文方法取得最低目标值，且 SOCM 基线在此设定下直接不收敛。

### 消融实验

| 配置 | 关键现象 | 说明 |
|------|----------|------|
| 相对损失（本文） | 高 $V_0$ 区控制方向正确 | 消除了 $e^{-\beta V_0}$ 隐式重加权 |
| PINN 残差损失 | 高 $V_0$ 区控制崩坏 | 损失对高代价区「失明」 |
| 变分 / Ritz 损失 | 高 $V_0$ 区控制崩坏 | 同样带 $e^{-\beta V_0}$ 因子 |
| 纯特征函数法 | $t\to0$ 优、近 $T$ 差 | 缺短程修正 |
| 混合法（本文） | 全程最低 $L^2$ 误差 | 长程用 $\phi_0$ + 短程切 IDO |

### 关键发现
- **隐式重加权是已有损失的根因**：PINN/变分损失因带 $e^{-\beta V_0}$ 因子而在最该控制好的高代价区失效；相对损失消去该因子后偏差消失（Figure 3 的 RING 例子直观可见）。
- **长程靠 $\phi_0$、短程靠 SDE 求解器**是最优分工：误差随 $t$ 的曲线显示两者互补，混合后整体最优。
- **增加特征函数数目收益递减**：作者指出学全谱代价高且回报快速递减，只用最大特征函数 $\phi_0$（加短程修正）即足够（Figure 2）。

## 亮点与洞察
- **酉等价把控制问题「搬」到量子力学的成熟工具箱里**：不是类比，而是精确归约——一旦证明 $L\cong S=-\Delta+\mathcal V$，离散谱、正交基、最低特征值单重等全套结论可直接调用，这是全文最漂亮的一步。
- **「长程 = 稳态特征函数」的视角**很可迁移：任何「随时间往回滚、长程退化」的问题，只要能线性化出一个有谱隙的算子，就可能用最大特征函数刻画长程行为，把 $T$ 依赖换成谱隙依赖。
- **相对损失是一个通用的小 trick**：把残差 $\|L\phi-\lambda\phi\|$ 换成相对残差 $\|L\phi/\phi-\lambda\|$，专治「参数化中指数因子让损失对某些区域失明」的毛病，可复用到其他特征函数/PDE 学习任务。
- **对称 LQR 任意终止代价的闭式解**本身是一个独立的理论贡献，去掉了经典 LQR「终止代价必须二次」的约束。

## 局限与展望
- **作者承认**：方法目前局限于梯度漂移问题；当 $L$ 连对称都不满足时，可能不再有实特征值（不过最大特征函数仍可能为实、非退化，长程行为仍由其刻画）。此外 $T_{\mathrm{cut}}$ 没有先验确定方法，是依赖谱隙 $\lambda_1-\lambda_0$ 与具体应用调的超参。
- **自己发现的局限**：实验集中在 $d=20$ 的合成基准与一个 $N=10$ 的舆论模型，规模仍偏小；「最大特征函数主导长程」的优势在谱隙很小（弱混合）的系统里会削弱——此时修正项衰减慢，需要更大的 $T_{\mathrm{cut}}$，混合法的复杂度优势随之缩水。
- **改进思路**：把酉等价框架推广到非梯度漂移（一般非对称 $L$）是最自然的下一步；以及为 $T_{\mathrm{cut}}$ 设计自适应、可在线估计谱隙的选取策略。

## 相关工作与启发
- **vs IDO / FBSDE 求解器**：它们把 SOC 当动态规划逐步往回滚，计算量随 $T$ 线性增长且长程退化；本文用特征函数刻画长程稳态，只在短程区间调用它们，复杂度 $O(Td)\to O(d)$。
- **vs 已有特征函数学习损失（PINN、deep Ritz、变分）**：这些损失在 $\phi=e^{-\beta V_0}$ 参数化下带隐式重加权、在高代价区失明；本文相对损失消去该因子。
- **vs 把薛定谔算子用于控制的早期工作（如稳态 HJB、Fokker–Planck 层面的分布控制）**：本文针对的是**有限时间、带任意终止代价**的 SOC，并落到可学习的最大特征函数 + 短程修正的实用算法上。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用酉等价把长程 SOC 精确归约到薛定谔算子谱问题，并给出对称 LQR 任意终止代价闭式解，视角新颖且有理论深度。
- 实验充分度: ⭐⭐⭐⭐ 多设定 + 舆论模型，一个数量级提升且有误差随 $t$ 的机理分析；但规模偏合成、$d=20$ 略小。
- 写作质量: ⭐⭐⭐⭐⭐ 理论链条（线性化→谱验证→特征函数控制→混合算法）层层递进，动机与失效分析（隐式重加权）讲得很清楚。
- 价值: ⭐⭐⭐⭐ 为长程随机最优控制提供了一个能摆脱 $T$ 依赖的新范式，相对损失这一 trick 也有独立复用价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] MDNS: Masked Diffusion Neural Sampler via Stochastic Optimal Control](../../NeurIPS2025/optimization/mdns_masked_diffusion_neural_sampler_via_stochastic_optimal_control.md)
- [\[ICLR 2026\] Dual Optimistic Ascent (PI Control) is the Augmented Lagrangian Method in Disguise](dual_optimistic_ascent_pi_control_is_the_augmented_lagrangian_method_in_disguise.md)
- [\[ICLR 2026\] Nesterov Finds GRAAL: Optimal and Adaptive Gradient Method for Convex Optimization](nesterov_finds_graal_optimal_and_adaptive_gradient_method_for_convex_optimizatio.md)
- [\[ICLR 2026\] Differentiable Model Predictive Control on the GPU](differentiable_model_predictive_control_on_the_gpu.md)
- [\[ICLR 2026\] Multilevel Control Functional](multilevel_control_functional.md)

</div>

<!-- RELATED:END -->
