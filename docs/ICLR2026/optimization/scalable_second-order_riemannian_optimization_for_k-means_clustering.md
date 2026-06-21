---
title: >-
  [论文解读] Scalable Second-Order Riemannian Optimization for K-means Clustering
description: >-
  [ICLR 2026][优化/理论][K-means] 本文把带非负约束的低秩 K-means SDP 松弛重写成一个**乘积流形上的无约束光滑优化**问题，用立方正则化的黎曼牛顿法求解，并通过流形分解把每步牛顿子问题的代价降到关于样本数 $n$ 的线性级，从而在 $n\cdot\epsilon^{-3/2}\cdot\mathrm{poly}(r,d)$ 时间内收敛到二阶临界点（在良性非凸假设下即全局最优），比一阶 SOTA 方法快 2–4 倍且统计精度相当。
tags:
  - "ICLR 2026"
  - "优化/理论"
  - "K-means"
  - "黎曼优化"
  - "半定规划松弛"
  - "二阶临界点"
  - "Burer-Monteiro 分解"
---

# Scalable Second-Order Riemannian Optimization for K-means Clustering

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=FNWNG1ftuw](https://openreview.net/forum?id=FNWNG1ftuw)  
**领域**: 优化 / 聚类  
**关键词**: K-means、黎曼优化、半定规划松弛、二阶临界点、Burer-Monteiro 分解

## 一句话总结
本文把带非负约束的低秩 K-means SDP 松弛重写成一个**乘积流形上的无约束光滑优化**问题，用立方正则化的黎曼牛顿法求解，并通过流形分解把每步牛顿子问题的代价降到关于样本数 $n$ 的线性级，从而在 $n\cdot\epsilon^{-3/2}\cdot\mathrm{poly}(r,d)$ 时间内收敛到二阶临界点（在良性非凸假设下即全局最优），比一阶 SOTA 方法快 2–4 倍且统计精度相当。

## 研究背景与动机
**领域现状**：K-means 聚类的目标是把 $n$ 个点划分成 $K$ 组使簇内相似度最大。Lloyd 算法、谱聚类等主流方法本质上都是这个离散问题的启发式，没有局部最优、更没有全局最优的保证。一个有理论保证的路线是把 K-means 写成半定规划（SDP）松弛：在数据来自分离良好的高斯混合模型（GMM）时，Peng & Wei (2007) 的 SDP 松弛在 $\mu\to 0^+$ 时被证明能多项式时间恢复全局最优簇，且一旦簇间分离超过信息论阈值就能精确恢复。

**现有痛点**：SDP (2) 要在 $n\times n$ 的矩阵 $Z$ 上优化，变量数 $O(n^2)$，对 $n$ 个样本的聚类完全不实用。自然的替代是 Burer-Monteiro 分解 $Z=UU^\top$，把变量降到 $O(n)$ 的因子矩阵 $U\in\mathbb{R}^{n\times r}$，但代价是放弃了凸性，一般只能求到可能是伪局部极小或鞍点的临界点。更麻烦的是，K-means 的非负 BM 形式额外带了逐元素约束 $U\ge 0$（保证 $Z$ 双重非负），而非负 BM 分解**普遍被认为不具备良性非凸性**——copositive / completely positive 检验都是 NP-hard，伪局部极小会大量涌现。

**核心矛盾**：尽管理论上非负形式应该满地是伪解，作者却**经验性地反复观测到**：在 SDP 能全局求解 K-means 的平均情形下，低秩非负形式 (3) 的所有二阶临界点都落在某个全局最优的邻域内（论文记为「良性非凸」假设 1）。如果这个假设成立，那么全局求解 K-means 就**等价于**找到 (3) 的一个二阶临界点。但在带约束的非凸场景里，没有通用算法能保证收敛到二阶临界点：核心障碍是每步迭代都要让 $U$ 满足非凸约束 $UU^\top\mathbf{1}_n=\mathbf{1}_n$、$\mathrm{tr}(UU^\top)=K$ 的**可行性**，同时还要朝最优推进。`fmincon`、`knitro` 只保证收敛到某个 merit 函数的临界点（对原问题可能不可行）；增广拉格朗日只保证局部一阶收敛。

**本文目标**：在保持严格可行的前提下，设计一个能**保证收敛到二阶临界点**、且每步代价低、能 scale 到大 $n$ 的算法。

**切入角度**：无约束非凸优化里，从任意初值出发，梯度下降、信赖域牛顿等一大批算法都能全局收敛到二阶临界点。如果能把 (3) 的约束「内化」成流形结构，让问题变成流形上的**无约束**光滑优化，就能直接把这套成熟算法连同它们的一/二阶收敛保证搬过来。

**核心 idea**：把约束集重写成一个**乘积流形**（投影超球 × 正交群），它带有廉价的二阶 retraction，再在其上跑立方正则化黎曼牛顿法，并利用黎曼 Hessian 的「块对角 + 低秩」结构把牛顿子问题压到线性时间。

## 方法详解

### 整体框架
方法是一条「松弛 → 分解 → 流形化 → 二阶求解」的重写链，目标是把 (3) 这个**带非凸约束的非负低秩问题**变成可以用现成黎曼二阶算法高效求解的形式。

第一步，约束集
$$\mathcal{M}=\{U\in\mathbb{R}^{n\times r}: UU^\top\mathbf{1}_n=\mathbf{1}_n,\ \mathrm{tr}(UU^\top)=K\}$$
本身可验证是一个流形（满足线性独立约束规范 LICQ）。直接在 $\mathcal{M}$ 上跑黎曼优化会得到一个与 Carson et al. (2017) 很像的算法，但它的致命问题是缺一个高效的 retraction——为保持 $U\in\mathcal{M}$ 每步都得做一次，而 Carson 用的指数 retraction 要 $O(n^2)$，直接卡死了 scalability。

第二步是本文的关键转折：不直接在 $\mathcal{M}$ 上做，而是构造一个从**乘积流形** $\widetilde{\mathcal{M}}=\mathcal{V}\times O_r$ 到 $\mathcal{M}$ 的 submersion（淹没映射）$\varphi$，其中 $\mathcal{V}$ 是投影超球、$O_r$ 是 $r\times r$ 正交矩阵群。乘积流形上每个分量的 retraction 都只是简单的欧氏投影，整体代价骤降到 $O(nr+r^3)$。问题随之等价转写为在 $\widetilde{\mathcal{M}}$ 上优化目标 (13)。

第三步，在 $\widetilde{\mathcal{M}}$ 上跑**立方正则化黎曼牛顿法**（定理 1，$O(\epsilon^{-3/2})$ 次迭代收敛到二阶临界点）。朴素实现每步牛顿子问题极贵，本文利用黎曼 Hessian 的块对角 + 低秩结构 + 二分搜索，把每步压到关于 $n$ 线性，最终在 $(n/\epsilon^{1.5})\cdot\mathrm{poly}(r,d,K)$ 时间内拿到 $\epsilon$ 二阶临界点。

### 关键设计

**1. 把带约束 K-means 写成流形上的无约束光滑优化：搬来二阶收敛保证**

针对「带约束非凸场景没有通用算法能保证收敛到二阶临界点」这个痛点，作者的做法是把可行性约束彻底吸收进流形的几何里。一般约束优化 $\min_{U\in\mathcal{M}} f(U)$，$\mathcal{M}=\{U: \mathcal{A}(UU^\top)+\mathcal{B}(U)=c\}$ 在 LICQ (8) 下，每个局部极小都是 $\epsilon$ 二阶临界点（同时满足一阶条件 (9) 和切空间上的二阶条件 (10)）。黎曼算法通过问题专属的 retraction 算子让迭代点始终落在可行集 $\mathcal{M}$ 上：从可行点 $U$ 出发沿可行集上的光滑曲线 $\gamma(t)=R_U(t\dot U)$ 前进，用黎曼梯度 $\mathrm{grad}\,f$ 和黎曼 Hessian $\mathrm{Hess}\,f$ 做局部二阶泰勒展开 (11) 选下降方向。这样就第一次为 K-means 打开了「从任意初值全局收敛到一/二阶最优」的可能——之前的 Carson、NLR 都做不到这点（前者非光滑目标 + 平滑罚 + 一阶法可能困在鞍点，后者只有全局解邻域内的局部线性收敛）。

**2. 乘积流形 submersion：把 $O(n^2)$ 的 retraction 砍到 $O(nr+r^3)$**

直接在 $\mathcal{M}$ 上做的瓶颈是 retraction 太贵，这一设计正是 scalability 的命门。作者证明（定理 2）存在从乘积流形 $\widetilde{\mathcal{M}}=\mathcal{V}\times O_r$ 到 $\mathcal{M}$ 的 submersion
$$\varphi(V,Q)=\hat V Q,\quad \hat V=\begin{bmatrix}\hat{\mathbf 1}_n & V\end{bmatrix},\ \hat{\mathbf 1}_n=n^{-1/2}\mathbf 1_n,$$
其中投影超球 $\mathcal{V}=\{V\in\mathbb{R}^{n\times(r-1)}: \mathbf 1_n^\top V=0,\ \mathrm{tr}(VV^\top)=K-1\}$、正交群 $O_r=\{Q: QQ^\top=I_r\}$。submersion 的标准性质保证：$\widetilde{\mathcal{M}}$ 上每个 $\epsilon$ 二阶点也是 $\mathcal{M}$ 上的 $c\epsilon$ 二阶点（$c$ 为常数缩放），所以在 $\widetilde{\mathcal{M}}$ 上求解与原问题等价。乘积流形的妙处在于它的二阶 retraction 就是两个分量各自的欧氏投影：
$$\mathrm{Proj}_{\mathcal V}(V)=\sqrt{K-1}\,\frac{V-n^{-1}\mathbf 1_n\mathbf 1_n^\top V}{\lVert V-n^{-1}\mathbf 1_n\mathbf 1_n^\top V\rVert},\qquad \mathrm{Proj}_{O_r}(Q)=(QQ^\top)^{-1/2}Q,$$
整体只要 $O(nr+r^3)$，相比 Carson 的 $O(n^2)$ 指数 retraction 是数量级的改进。等价目标变为
$$\min_{(V,Q)\in\widetilde{\mathcal{M}}}\ \langle C, VV^\top\rangle-\mu\sum_{i,j}\log\big(\varphi_{i,j}(V,Q)\big)_+,\qquad C=-XX^\top.$$
对数罚带来两个附加要求：算法必须从严格可行点 $\varphi(V_0,Q_0)>0$ 起步（作者给出了一个好的严格可行初值，并证明这种点只在搜索秩**过参数化** $r>K$ 时存在），且因对数罚只在闭的严格可行子集上 Lipschitz，需要特别处理才能套用收敛保证。

**3. 立方正则化黎曼牛顿 + 线性时间牛顿子问题：用块对角加低秩结构换速度**

有了流形和廉价 retraction，原则上黎曼梯度下降能在 $(n/\epsilon)\cdot\mathrm{poly}$ 时间拿到一阶临界点，但实践中对数罚导致景观**极度病态**，梯度法、以及 MANOPT/PYMANOPT 里的共轭梯度（CG）和黎曼信赖域（RTR）都收敛得惨不忍睹。作者最终用立方正则化黎曼牛顿（定理 1，$O(\epsilon^{-3/2})$ 迭代、与维度无关），关键是把每步的牛顿子问题
$$\min_{Ap=0}\ g^\top p+\tfrac12 p^\top H p+\tfrac{L}{6}\lVert p\rVert^3$$
高效解出来。这里 $g,H$ 是向量化的黎曼梯度与 Hessian，$A$ 实现切空间约束；子问题有 $O(nr)$ 个变量、$O(r^2)$ 个线性约束。由于只有线性约束，其局部极小必满足一/二阶最优条件，可写成带正则 $\lambda I$ 的 KKT 系统并配 $\lambda=\tfrac{L}{2}\lVert p\rVert$。引理 2 证明在 $\lambda>-\lambda_{\min}$ 时参数化解 $p(\lambda)$ 良定义且 $\lVert p(\lambda)\rVert$ 关于 $\lambda$ 单调递减，于是可对单调方程 $2\lambda=L\lVert p(\lambda)\rVert$ 做**二分搜索**求 $\lambda_{\mathrm{opt}}$。二分的主要代价是算 $p(\lambda)$，朴素地要 $O(n^3 r^3)$，但本文证明黎曼 Hessian $H$ 具有「块对角 + 低秩」结构，借此把单次求解降到 $n\cdot\mathrm{poly}(r,d)$。综合定理 1，整套方法在 $(n/\epsilon^{1.5})\cdot\mathrm{poly}(r,d,K)$ 时间内得到 $\epsilon$ 二阶临界点——即二阶法的迭代复杂度配上一阶法量级的单步代价。

### 损失函数 / 训练策略
优化目标即上文 (13)：数据项 $\langle C,VV^\top\rangle$（$C=-XX^\top$ 为负 Gram 矩阵）加对数障碍罚 $-\mu\sum_{i,j}\log(\varphi_{i,j})_+$。两个核心超参：罚系数 $\mu$ 和搜索秩 $r$。实践建议——$\mu$ 取得足够小以保证落在「相变」良性区间；$r$ 在 $\mu$ 小时不敏感，取最小可行的 $r=K+1$ 即可，仅当 $\mu$ 偏大、算法陷入伪解区时调大 $r$ 帮助逃逸。收敛后由块对角隶属矩阵 $Z=UU^\top$ 经引理 1 唯一恢复出独热的簇分配矩阵。

## 实验关键数据

数据集：合成高斯混合模型（GMM，质心放在单纯形顶点、分离度 $\gamma\Theta/2$）与真实质谱流式（CyTOF，265,627 个细胞 × 32 标记，按 Zhuang et al. 2024 采样 $K=4$ 不平衡簇、$n=1800$）。

### 主实验

CyTOF 真实数据上与一阶 SOTA（NLR）及经典基线对比（误聚类率、到 oracle $Z^\star$ 的 Frobenius 距离）：

| 方法 | 误聚类率 | 到 $Z^\star$ 的 Frobenius 距离 | 说明 |
|------|---------|------------------------------|------|
| 本文（二阶黎曼牛顿） | ≈0，方差最小 | **最小** | 同模型但用二阶 Hessian 更新，恢复更准 |
| NLR (Zhuang 2024) | ≈0 | 较大 | 一阶 SOTA，误聚类同样低但真值恢复差 |
| 谱聚类 SC / NMF / KM++ | 明显更高、方差大 | — | 经典基线 |

GMM 上与 NLR 的收敛效率对比（$n=100$, $\gamma=0.8$, $\mu=0.1$）：

| 指标 | 本文 | NLR |
|------|------|-----|
| 收敛所需迭代数 | **152** | ≈80,000 |
| 单步代价 | 约为 NLR 单步的 25–100 倍 | 1×（单次矩阵-向量积） |
| 单步随 $n$ 的复杂度 | $O(n)$（与理论一致） | $O(n)$ |
| 总墙钟时间 | 快 **2–4 倍** | 基准 |

### 消融 / 对比分析

| 配置 / 对比对象 | 关键结果 | 说明 |
|------|---------|------|
| 验证假设 1（图 1） | 损失降到 $10^{-10}$、Hessian 最小特征值由负转正 | 二阶临界点≈全局最优，零聚类误差 |
| vs Carson et al. 2017（图 4） | 误聚类 >30%，不可行性 $\lVert U_-\rVert$ 始终不消 | 罚系数 $\lambda$ 调大改善可行性却恶化聚类，鱼与熊掌不可兼得 |
| vs 经典 RTR（图 5） | 本文≈360 次迭代到机器精度，RTR 停滞 >21,000 次 | 对数罚致极度病态，通用信赖域吃不消 |
| 超参 $\mu$（图 6 左） | 存在尖锐相变阈值，超过则失效 | $\mu$ 需取够小 |
| 搜索秩 $r$（图 6 中/右） | $\mu$ 小时对 $r$ 不敏感；$\mu$ 大时增大 $r$ 助逃逸 | 推荐 $r=K+1$ |
| 错设簇数 $K_{\mathrm{mis}}=3,5$（图 7） | NMI 仍高，局部收敛行为与 $K=4$ 类似 | 对簇数误设鲁棒 |

### 关键发现
- **二阶法的迭代优势压倒单步开销**：单个牛顿步比 NLR 单步贵 25–100 倍，但迭代数从 8 万降到 152（约三个数量级），净收益是总时间快 2–4 倍——这正是「把二阶法单步代价降到一阶量级」后才成立的结论。
- **良性非凸假设得到强经验支持**：50 次随机初始化下损失稳定收敛到零、Hessian 最小特征值非负，二阶临界点几乎都是全局最优。
- **病态来自对数障碍**：CG/RTR 失败、本文用立方正则化 + 块对角低秩结构才稳住，说明算法选择对这个 landscape 至关重要。
- **超参鲁棒性**：$r=K+1$ 这一最省配置在 $\mu$ 选得当的情况下就够用，且对簇数误设鲁棒（用 NMI 衡量）。

## 亮点与洞察
- **用乘积流形换掉昂贵 retraction**：把 $\mathcal{M}$ 经 submersion 抬升到「投影超球 × 正交群」，retraction 退化成两个简单欧氏投影，是把 $O(n^2)$ 砍到 $O(nr+r^3)$ 的关键。这种「找一个更好参数化的覆盖流形」的思路可迁移到其它带复杂约束的低秩 SDP。
- **块对角 + 低秩 Hessian 结构 → 线性时间牛顿步**：让二阶法获得一阶法的单步成本，是「二阶法不实用」这一常识的有力反例，对任何结构化 Hessian 的二阶优化都有借鉴意义。
- **把组合聚类问题转成有保证的连续优化**：在良性非凸假设下，全局解 K-means 被归约为「找一个二阶临界点」，再用黎曼算法严格保证收敛——这条「离散→SDP→低秩→流形→二阶」的链条本身就很漂亮。
- **二分搜索解立方正则子问题**：靠引理 2 的单调性把子问题求解变成对 $2\lambda=L\lVert p(\lambda)\rVert$ 的一维二分，干净且可证。

## 局限与展望
- **依赖良性非凸假设**：全局最优保证建立在假设 1（所有近似二阶临界点都在全局最优邻域）之上，本文只给出强经验证据，并未在 K-means 非负形式下给出严格证明——而非负 BM 一般是已知会有伪解的。
- **依赖分离良好的 GMM 平均情形**：理论保证绑定在「SDP 能全局求解」的可分离 GMM 区间，簇间分离不够时仍无能为力（信息论意义上本就不可恢复）。
- **对超参 $\mu$ 敏感**：存在尖锐相变，$\mu$ 一旦超阈值就失效，实际需要调参定位良性区间。
- **规模验证有限**：真实数据实验在 $n=1800$ 量级，单步 $O(n)$ 的优势在更大规模、更高秩 $r$ 下的常数因子表现还需进一步检验。

## 相关工作与启发
- **vs Carson et al. (2017，一阶黎曼)**：同样走非负 BM 路线，但用非光滑目标 + 平滑罚 + 一阶法，既无法真正强制 $U\ge 0$ 可行、又可能困在鞍点，且指数 retraction $O(n^2)$ 不可扩展；本文用乘积流形廉价 retraction + 二阶法，可证收敛到二阶临界点且 scale。
- **vs NLR (Zhuang et al. 2024)**：投影梯度 + 增广拉格朗日的一阶原对偶法，只能在全局解邻域局部线性收敛、无二阶保证；本文二阶更新单步虽贵但迭代数低三个量级，总时间快 2–4 倍且真值恢复更准。
- **vs 混合方法（Wang & Hu 2025、Monteiro 2025 等）**：这些方法用黎曼优化处理简单流形约束、增广拉格朗日处理其余约束；但 K-means 要求对非负性和单纯形型流形约束**同时严格可行**（恢复出的因子必须编码合法划分），投影/增广拉格朗日方案会破坏这一结构性保证，本文则全程保持严格可行。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次为 K-means 给出全局收敛到二阶临界点的可扩展黎曼算法，乘积流形 submersion + 线性时间牛顿步都很巧
- 实验充分度: ⭐⭐⭐⭐ 合成 + 真实数据、与一阶 SOTA / 经典黎曼 / 多基线全面对比，但真实数据规模偏小
- 写作质量: ⭐⭐⭐⭐ 动机—理论—算法链条清晰，公式严谨；细节多在附录
- 价值: ⭐⭐⭐⭐⭐ 把「二阶法不实用」的常识打破，为带保证的聚类提供了实用工具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] The Potential of Second-Order Optimization for LLMs: A Study with Full Gauss-Newton](the_potential_of_second-order_optimization_for_llms_a_study_with_full_gauss-newt.md)
- [\[ICLR 2026\] Angle k-means：用角度关系加速精确 k-means](angle_k-means.md)
- [\[ICLR 2026\] Riemannian Optimization on Relaxed Indicator Matrix Manifold](riemannian_optimization_on_relaxed_indicator_matrix_manifold.md)
- [\[ICLR 2026\] Landing with the Score: Riemannian Optimization through Denoising](landing_with_the_score_riemannian_optimization_through_denoising.md)
- [\[AAAI 2026\] Convex Clustering Redefined: Robust Learning with the Median of Means Estimator](../../AAAI2026/optimization/convex_clustering_redefined_robust_learning_with_higher_order_norms_and_beyond.md)

</div>

<!-- RELATED:END -->
