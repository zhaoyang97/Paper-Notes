---
title: >-
  [论文解读] A Block Coordinate Descent Method for Nonsmooth Composite Optimization under Orthogonality Constraints
description: >-
  [ICLR2026][优化/理论][正交约束] 本文提出 OBCD，一种在正交约束（Stiefel 流形）下求解"光滑 + 非光滑"复合优化的块坐标下降算法：每次只更新解矩阵的 $k\ge 2$ 行、把问题压成一个 $k\times k$ 的小型正交约束子问题精确求解，从而做到天然可行、单步开销低，同时给出比经典临界点更强的"block-$k$ 稳定点"最优性、$O(1/\epsilon)$ 迭代复杂度以及 KL 条件下的末迭代收敛率。
tags:
  - "ICLR2026"
  - "优化/理论"
  - "正交约束"
  - "Stiefel流形"
  - "块坐标下降"
  - "非光滑复合优化"
  - "稀疏PCA"
---

# A Block Coordinate Descent Method for Nonsmooth Composite Optimization under Orthogonality Constraints

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=L3Or2mhuCH](https://openreview.net/forum?id=L3Or2mhuCH)  
**代码**: 待确认  
**领域**: optimization  
**关键词**: 正交约束, Stiefel流形, 块坐标下降, 非光滑复合优化, 稀疏PCA  

## 一句话总结
本文提出 OBCD，一种在正交约束（Stiefel 流形）下求解"光滑 + 非光滑"复合优化的块坐标下降算法：每次只更新解矩阵的 $k\ge 2$ 行、把问题压成一个 $k\times k$ 的小型正交约束子问题精确求解，从而做到天然可行、单步开销低，同时给出比经典临界点更强的"block-$k$ 稳定点"最优性、$O(1/\epsilon)$ 迭代复杂度以及 KL 条件下的末迭代收敛率。

## 研究背景与动机

**领域现状**：很多统计学习与数据科学模型都能写成"正交约束下的非光滑复合优化"：$\min_{X\in\mathbb{R}^{n\times r}} F(X)=f(X)+h(X)$，s.t. $X^\top X=I_r$，其中 $f$ 光滑、$h$ 非光滑且逐坐标可分（如 $\ell_0$、$\ell_1$、capped-$\ell_1$、非负指示函数）。典型实例包括稀疏 PCA、非负 PCA、字典学习、正交非负矩阵分解、K-indicators 聚类、深度网络的正交正则等。约束集 $\{X:X^\top X=I_r\}$ 就是 Stiefel 流形 $\mathrm{St}(n,r)$。

**现有痛点**：这个问题同时被两件事卡住——正交约束本身是非凸的、投影/收缩代价高；目标里还带非光滑项。现有方法各有短板：测地线法要解 ODE、投影/收缩法每步要做 SVD 或极分解、乘子修正与 landing 方法常常只在极限处才满足可行性（迭代过程中是"几乎正交"），近端梯度/黎曼次梯度/算子分裂（ADMM、RADMM、PSM）要么依赖全梯度、单步开销大，要么缺乏严格的末迭代收敛保证。

**核心矛盾**：几乎所有现有方法都依赖**全梯度信息**并在**整个 $n\times r$ 空间**里处理正交约束与非光滑项，单步既贵又难精确求解；而它们给出的最优性多半只是"临界点"（一阶稳定）这种**弱**保证，无法刻画解相对全局最优还差多少。

**本文目标**：设计一个 (i) 每步只动少量坐标、单步便宜，(ii) 迭代过程始终严格落在流形上（真·可行、真·下降），(iii) 能精确求解子问题，(iv) 同时具备更强最优性刻画与严格收敛率的算法。

**切入角度**：作者把经典的块坐标下降（BCD）搬到 Stiefel 流形上——既然全空间难处理，那就每次只挑 $k$ 行更新，并设计一个**保正交**的更新格式，使得"只动 $k$ 行"等价于在一个小的 $k\times k$ Stiefel 流形 $\mathrm{St}(k,k)$ 上解子问题。这样一来非光滑项与正交约束都被压进低维空间，反而可以精确求解。

**核心 idea**：用"行块更新 $X^{t+1}(B,:)=V\,X^{t}(B,:)$（$V\in\mathrm{St}(k,k)$）"取代全空间更新，把高维难题化成可精确求解的低维 $k\times k$ 正交子问题，并据此建立 block-$k$ 稳定点这一更强的最优性层级。

## 方法详解

### 整体框架

OBCD（Orthogonality-constrained Block Coordinate Descent）是一个在 Stiefel 流形子流形上沿块坐标方向逐次最小化目标的迭代算法。输入是初始可行点 $X^0\in\mathrm{St}(n,r)$、块大小 $k\ge 2$、近端参数 $\alpha>0$；输出是收敛到 block-$k$ 稳定点的序列 $\{X^t\}$。

每一轮 $t$ 做四步：(S1) 从 $\{1,\dots,n\}$ 里选一个大小为 $k$ 的**工作集** $B$（随机或循环）；(S2) 按式 $Q=(Z^\top\!\otimes U_B)^\top H (Z^\top\!\otimes U_B)$ 或对角近似 $Q=\varsigma I$ 构造曲率矩阵；(S3) 用多数化-最小化（MM）造出一个上界代理函数 $K(V;X^t,B)$，并在 $V\in\mathrm{St}(k,k)$ 上求全局最小 $\bar V^t$；(S4) 用 $X^{t+1}(B,:)\leftarrow \bar V^t X^t(B,:)$ 只更新这 $k$ 行。整个过程的关键是：无论怎么选 $V\in\mathrm{St}(k,k)$，更新后的 $X^{t+1}$ 自动还在 $\mathrm{St}(n,r)$ 上，因此每一步都是可行且下降的，不需要任何收缩/投影回流形的步骤。

这是一个**部分梯度法**：算 $K$ 里的线性项 $\langle [\nabla f(X^t)(X^t)^\top]_{BB}, V\rangle = \langle [\nabla f(X^t)]_{B,:}[X^t]_{B,:}^\top, V\rangle$ 只需要 $\nabla f$ 的 $k$ 行和 $X^t$ 的 $k$ 行，单步开销远低于全梯度方法。

### 关键设计

**1. 保正交的行块更新格式：把"只动 k 行"做成天然可行**

正交约束非凸、收缩/投影昂贵，是这类问题的第一道坎。本文给出的更新格式直接绕开收缩。设工作集 $B$（$|B|=k$）对应的选择矩阵为 $U_B\in\mathbb{R}^{n\times k}$，更新写成

$$X^{+}=X+U_B(V-I_k)U_B^\top X,\qquad V\in\mathrm{St}(k,k).$$

它等价于只把 $B$ 这 $k$ 行替换成 $V X(B,:)$、其余行不动。关键引理（Lemma 2.1）证明：对任意 $X$ 都有 $[X^{+}]^\top X^{+}=X^\top X$，因此**只要原来 $X$ 正交，更新后 $X^{+}$ 仍然正交**。直觉上，因为 $V$ 本身是 $k\times k$ 正交阵，对一组正交行做正交线性组合不破坏整体正交性。这样每一步迭代严格落在流形上，是"feasible method"，从根上避免了 landing/乘子法那种"只在极限处可行"的毛病。配套引理还给出 $\|X^{+}-X\|_F^2=2\langle I_k-V,\,U_B^\top X X^\top U_B\rangle\le \|V-I_k\|_F^2$，为后面的充分下降性铺路。

**2. MM 多数化把子问题压成小型 $k\times k$ 正交优化**

直接在 $V$ 上最小化 $F(X_B^t(V))$ 仍含光滑 $f$ 和非光滑 $h$，不好解。本文对光滑部分用 $H$-光滑性做二次上界、对非光滑 $h$ 保持不动，构造多数化代理

$$K(V;X^t,B)\triangleq \tfrac12\|V-I_k\|_{Q+\alpha I}^2+\langle V,[\nabla f(X^t)(X^t)^\top]_{BB}\rangle+h(VU_B^\top X^t)+\ddot c,$$

满足 $F(X_B^t(V))\le K(V;X^t,B)$。利用 $h$ 的逐坐标可分性，$h(X_B^t(V))=h(U_{B^c}^\top X^t)+h(VU_B^\top X^t)$，于是只剩 $h(VU_B^\top X^t)$ 这一项与 $V$ 相关。最终子问题 $\bar V^t\in\arg\min_{V\in\mathrm{St}(k,k)} K(V;X^t,B)$ 进一步化为标准形式 $\min_{V\in\mathrm{St}(k,k)} \tfrac12\|V\|_{\tilde Q}^2+\langle V,P\rangle+h(VZ)$（$\tilde Q=Q+\alpha I$）。这一步的巧妙之处在于：它**同时**处理非光滑项 $h$ 和正交约束这两个非光滑成分，但是在一个 $k\times k$ 的低维空间里——这比标准近端梯度只在全空间处理单个非光滑项更"重"，却因为维度极低反而能精确求解。当 $h\equiv 0$、$Q$ 取对角时，子问题闭式解为 $\bar V^t=-\mathcal{P}_M(P)$（$P$ 的最近正交矩阵）。

**3. $k=2$ 旋转+反射 + 断点搜索：把子问题精确解出来**

子问题最干净的情形是 $k=2$。Lemma 2.5 指出任何 $V\in\mathrm{St}(2,2)$ 都能写成 Givens 旋转 $V_\theta^{\mathrm{rot}}=\big(\begin{smallmatrix}\cos\theta & \sin\theta\\ -\sin\theta & \cos\theta\end{smallmatrix}\big)$（$\det=1$）或 Jacobi 反射 $V_\theta^{\mathrm{ref}}=\big(\begin{smallmatrix}-\cos\theta & \sin\theta\\ \sin\theta & \cos\theta\end{smallmatrix}\big)$（$\det=-1$），从而把子问题降成关于单个角度 $\theta$ 的**一维**问题。即便 $h\ne 0$ 使目标非光滑，作者也用一种新的**断点搜索（breakpoint search）**方法精确定位最优 $\bar\theta$（非光滑函数在若干断点处不可导，逐段求极值即可全局求解，用 C++ 实现内层逐元素循环）。

一个容易被忽视但关键的点是：必须**同时**用旋转和反射两族矩阵。以往工作（如对称特征值、稀疏 PCA）只用旋转 $\{V_\theta^{\mathrm{rot}}\}$，但本文用反例说明，单靠旋转会漏掉更优解——例如 $\min_{V\in\mathrm{St}(2,2)}\|V-A\|_F^2$ 在某些 $A$ 下，反射 $V_\theta^{\mathrm{ref}}$ 才能取到更低目标值。这是因为反射矩阵无法由任何旋转序列表示，二者覆盖 $\mathrm{St}(2,2)$ 的两个连通分支，缺一不可。

**4. block-$k$ 稳定点：比临界点更强的最优性 + 严格收敛率**

这是理论核心，也是本文相对已有方法"只给弱保证"的关键升级。作者定义 **block-$k$ 稳定点（BS$k$-point）**：$\ddot X$ 是 BS$k$-point 当且仅当对所有大小为 $k$ 的工作集 $B$，恒等阵 $I_k$ 都是子问题代理 $K(V;\ddot X,B)$ 的全局最小点——直观就是"无论挑哪 $k$ 行去优化，都已经没法再降目标"。再借助"任意正交矩阵可由一串 $2\times 2$ 旋转/反射复合表示"（Theorem 3.1 / Corollary 3.2，说明该更新格式从任意初值可达任意可行点），证明了如下最优性层级（Theorem 3.8）：

$$\{\text{临界点}\}\ \supseteq\ \{\text{BS}_2\}\ \supseteq\ \cdots\ \supseteq\ \{\text{BS}_k\}\ \supseteq\ \{\text{BS}_{k+1}\}\ \supseteq\ \{\text{全局最优点}\},$$

即 $k$ 越大、稳定性越强，且 BS$2$ 已严格强于标准临界点（反向包含一般不成立）。收敛性方面：先证充分下降 $\tfrac{\alpha}{2}\|X^{t+1}-X^t\|_F^2\le F(X^t)-F(X^{t+1})$，由此得到找到 $\epsilon$-BS$k$-point 的迭代复杂度为 $O(1/\epsilon)$（Theorem 4.2）；进一步在黎曼次梯度度量下给出 $\epsilon$-临界点复杂度（Theorem 4.6）；最后在 Kurdyka-Łojasiewicz（KL）条件下建立**非遍历（末迭代）收敛率**（Theorem 4.10–4.11）：依 KL 指数 $\sigma$ 不同，分别得到有限步收敛（$\sigma=0$）、线性收敛（$\sigma\in(0,\tfrac12]$，$\mathbb{E}\|X^t-X^\infty\|_F\le \dot c\,\dot\tau^{\,t}$）和次线性收敛（$\sigma\in(\tfrac12,1)$，$\mathbb{E}\|X^t-X^\infty\|_F\le \dot c/t^{\dot\tau}$）。

### 损失函数 / 训练策略

无需训练。算法只有两个主要超参：近端参数 $\alpha>0$（控制充分下降步长，越大越保守）与块大小 $k\ge 2$（越大每步子问题越强、最优性越高但单步越贵，实践常取 $k=2$ 以享受闭式/一维精确解）。工作集选择用随机或循环策略，理论分析在随机策略下展开。

## 实验关键数据

### 主实验

任务为 $\ell_0$-正则稀疏 PCA：$\min_{X\in\mathrm{St}(n,r)} -\langle X,CX\rangle+\lambda\|X\|_0$（$C$ 为协方差矩阵），另有 $\ell_1$-SPCA、非负 PCA 见附录。对比对象为三类算子分裂法：LADMM、RADMM、PSM，各配恒等/随机初始化共六个变体；OBCD 用随机工作集 + 恒等初始化（OBCD-R(id)）。**指标为目标函数值（越低越好），统一 40 秒时间上限**。下表节选 $r=20,\ \lambda=10$ 的结果（数值越小越优）：

| 数据集 | LADMM(id) | RADMM(id) | PSM(id) | OBCD-R(id) |
|--------|-----------|-----------|---------|------------|
| w1a (2477×300) | 199.897 | 219.698 | 199.897 | **199.667** |
| TDT2 (500×1000) | 199.997 | 359.382 | 199.997 | **199.258** |
| 20News (8000×1000) | 199.995 | 219.673 | 199.995 | **199.222** |
| MNIST (60000×784) | 199.985 | 379.893 | 199.985 | **199.896** |
| Cifar (1000×1000) | 199.979 | 479.979 | 199.979 | **199.974** |

在 $\lambda=50$ 的更难设置下，差距进一步拉大（如 MNIST：LADMM 999.985 / RADMM 1699.913 / PSM 2849.852 / OBCD **999.896**；randn：其他方法普遍 1300–3900，OBCD 仅 **999.977**）。

### 消融 / 分析

| 配置 | 作用 | 说明 |
|------|------|------|
| 仅用旋转 $V^{\mathrm{rot}}$ | $k=2$ 子问题 | 在特定 $2\times 2$ 算例上漏掉更优解，目标值偏高 |
| 旋转 + 反射 $V^{\mathrm{rot}},V^{\mathrm{ref}}$ | $k=2$ 子问题 | 覆盖 $\mathrm{St}(2,2)$ 两个分支，取到更低目标 |
| 块大小 $k$ 增大 | 最优性强度 | BS$k$ 层级随 $k$ 单调变强（BS$_{k}\supseteq$ BS$_{k+1}$） |

### 关键发现

- OBCD-R(id) 在几乎所有数据集与正则强度下都取得**最低目标值**，尤其在 $\lambda$ 大、问题更非凸时优势最明显——说明 block-$k$ 稳定点确实比 ADMM 类方法落入的临界点更优。
- RADMM/PSM 在随机初始化下波动大、常陷入差解；OBCD 的"每步可行 + 充分下降"使其表现稳定。
- 反射矩阵不可省：仅用旋转会系统性地高估目标，验证了同时使用两族正交矩阵的必要性。

## 亮点与洞察

- **"行块更新天然保正交"是全文的支点**：$X^{+}=X+U_B(V-I_k)U_B^\top X$ 让"只动 $k$ 行"与"留在 Stiefel 流形上"自动等价，省掉了所有收缩/投影步骤——这正是它能做成可行下降法的根本，也是把高维非凸约束化简为低维子问题的钥匙。
- **把非光滑 + 正交两个硬骨头一起塞进 $k\times k$ 小空间**：维度低到可以精确求解，反而比在全空间近端更彻底；$k=2$ 时进一步降成一维角度问题，配断点搜索精确解非光滑子问题，很优雅。
- **block-$k$ 稳定点给出可调节的最优性层级**：$k$ 越大保证越强，在临界点与全局最优之间架了一座可控的"梯子"，这种"用块大小换最优性强度"的视角可迁移到其他流形约束优化。
- **反射矩阵的必要性论证**：一个常被忽视的细节——单用 Givens 旋转覆盖不了 $\mathrm{St}(2,2)$ 的负行列式分支，作者用具体反例点破，体现了对几何结构的细致把握。

## 局限与展望

- **子问题全局可解性受限**：当 $k>2$ 且 $h\ne 0$ 时，$k\times k$ 子问题未必有全局解，只能退而求其次找满足 $K(\bar V^t)\le K(I_k)$ 的局部解，此时只能保证较弱的最优性——实践中真正"精确"的甜区主要在 $k=2$。
- **断点搜索依赖底层实现**：非光滑子问题的精确求解要逐元素循环，作者用 C++ 才跑得动，纯 MATLAB 实现会很慢；可扩展性与并行化（相比列式并行 BCD）还需进一步工作。
- **检验 BS$k$ 代价高**：确定性判定一个解是否为 BS$k$-point 需解全部 $C_n^k$ 个子问题，只能靠随机抽样在期望意义下检验。
- **实验聚焦稀疏/非负 PCA**：虽然问题框架覆盖字典学习、ONMF、聚类等，但论文主要在 PCA 系列上验证，更大规模深度网络正交正则等场景的实证尚缺。

## 相关工作与启发

- **vs 投影/收缩类（黎曼梯度、近端梯度）**：它们每步沿（黎曼）梯度走一步再投影/收缩回流形，单步常含 SVD/极分解、依赖全梯度；OBCD 用保正交的行块更新天然落在流形上、只需部分梯度，单步更便宜且严格可行。
- **vs 算子分裂（LADMM / RADMM / PSM）**：分裂法引入辅助变量与线性约束，常只在极限处可行、缺末迭代收敛率且只达临界点；OBCD 每步可行下降，给出 $O(1/\epsilon)$ 复杂度、KL 末迭代率，并落到更强的 block-$k$ 稳定点。
- **vs 已有列式 BCD（Shalit & Chechik 2014；Gao et al. 2019）**：它们多限于光滑目标、$k=2$ 且 $r=n$ 的列式更新，且只用旋转矩阵；本文是**行式** BCD，支持逐坐标非光滑 $h$、$k\ge 2$、$r\le n$，并同时使用旋转与反射矩阵，适用面更广。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 保正交行块更新 + block-$k$ 稳定点层级，把 BCD 干净地搬上 Stiefel 流形且给出更强最优性
- 实验充分度: ⭐⭐⭐⭐ 多数据集、多正则强度对比充分，但任务集中在 PCA 系列
- 写作质量: ⭐⭐⭐⭐⭐ 算法-最优性-收敛三段论证完整严谨，引理层层递进
- 价值: ⭐⭐⭐⭐ 为正交约束非光滑优化提供了可行、可精确求解且理论扎实的新框架

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Clipped Gradient Methods for Nonsmooth Convex Optimization under Heavy-Tailed Noise: A Refined Analysis](clipped_gradient_methods_for_nonsmooth_convex_optimization_under_heavy-tailed_no.md)
- [\[ICML 2026\] Mirror Descent Under Generalized Smoothness](../../ICML2026/optimization/mirror_descent_under_generalized_smoothness.md)
- [\[ICML 2026\] Multi-Objective Bayesian Optimization via Adaptive ε-Constraints Decomposition](../../ICML2026/optimization/multi-objective_bayesian_optimization_via_adaptive_varepsilon-constraints_decomp.md)
- [\[ICLR 2026\] Arbitrary-Order Block SignSGD for Memory-Efficient LLM Fine-Tuning](arbitrary-order_block_signsgd_for_memory-efficient_llm_fine-tuning.md)
- [\[ICLR 2026\] Decentralized Nonconvex Optimization under Heavy-Tailed Noise: Normalization and Optimal Convergence](decentralized_nonconvex_optimization_under_heavy-tailed_noise_normalization_and_.md)

</div>

<!-- RELATED:END -->
