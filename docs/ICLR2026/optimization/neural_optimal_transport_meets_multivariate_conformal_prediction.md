---
title: >-
  [论文解读] Neural Optimal Transport Meets Multivariate Conformal Prediction
description: >-
  [ICLR 2026][优化/理论][向量分位数回归] 用神经最优传输：学一个连续、循环单调的「向量分位数函数」（把参考分布搬运到条件分布），再把它诱导出的多元秩：当作共形分数，构造出既有有限样本覆盖保证、又能自适应条件分布几何形状的多元预测区域。 - 领域现状：标量分位数回归（Koenker）是刻画异方差、偏态、尾部行为的…
tags:
  - "ICLR 2026"
  - "优化/理论"
  - "向量分位数回归"
  - "神经最优传输"
  - "多元共形预测"
  - "输入凸神经网络"
  - "摊销优化"
---

# Neural Optimal Transport Meets Multivariate Conformal Prediction

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=ylaKpd7tmA](https://openreview.net/forum?id=ylaKpd7tmA)  
**代码**: 待确认（作者承诺接收后开源）  
**领域**: 优化 / 最优传输 / 不确定性量化  
**关键词**: 向量分位数回归, 神经最优传输, 多元共形预测, 输入凸神经网络, 摊销优化  

## 一句话总结
用**神经最优传输**学一个连续、循环单调的「向量分位数函数」（把参考分布搬运到条件分布），再把它诱导出的**多元秩**当作共形分数，构造出既有有限样本覆盖保证、又能自适应条件分布几何形状的多元预测区域。

## 研究背景与动机
- **领域现状**：标量分位数回归（Koenker）是刻画异方差、偏态、尾部行为的基石；共形预测能给出分布无关、有限样本的边际覆盖保证。但这两套工具在**多元响应** $Y\in\mathbb{R}^d$ 上都不好用。
- **现有痛点**：$\mathbb{R}^d$ 上没有自然的全序，导致分位数难以定义。已有的多元共形方法要么**逐坐标**处理（产生保守的矩形盒子，忽略坐标间相关性），要么把多维问题**压成一维标量分数**（球/盒形状受限），要么依赖深度生成嵌入的**启发式分数**，缺乏理论指导，也不显式利用联合条件分布的几何。
- **核心矛盾**：最优传输理论已经给出了多元秩/分位数的「正确」定义（把分位数看成从参考分布到 $Y$ 律的传输映射，恢复中心向外的秩与嵌套分位数区域），但**之前没人用神经 OT 把向量分位数回归（VQR）规模化**——已有连续 VQR 都困在「分位映射对 $X$ 嵌入仿射」的假设里，表达力受限，且只能给出离散逐点解，无法得到连续的秩函数。
- **本文目标**：学一个**连续、参数化、循环单调**的条件向量分位数函数（CVQF）及其逆（多元秩），并把秩函数无缝接入共形预测，得到自适应几何的有效预测区域。
- **为什么是 OT**：把分位数看成「从参考分布到目标分布的传输映射」，自然恢复中心向外的秩、嵌套的分位数区域，把一维序统计量干净地推广到高维。
- **核心 idea**：**(1) 用输入凸神经网络（PICNN）+ 摊销优化把神经 OT 扩展到「条件」VQR**，从单一联合采样直接学凸势函数；**(2) 把学到的多元秩范数 $\|\hat Q^{-1}_{Y|X}(y,x)\|$ 当共形分数**，并证明在径向结构下这种「拉回球」就是体积最优的最高密度区域（HPD）。

## 方法详解

### 整体框架
方法分两段：先用神经 OT 学一个把参考分布 $F_U$（如球内均匀/高斯）搬到条件分布 $F_{Y|X=x}$ 的向量分位数映射 $Q_{Y|X}(u,x)=\nabla_u\varphi(u,x)$，它是某个凸势 $\varphi$ 的梯度（保证循环单调/可逆）；其逆 $Q^{-1}_{Y|X}(y,x)=\nabla_y\psi(y,x)$ 就是多元秩，且条件于 $X$ 服从 $F_U$。第二段把秩范数 $S=\|\hat Q^{-1}_{Y|X}(Y,X)\|$ 作为共形分数，用校准集分位数定一个半径 $\rho_{1-\alpha}$，预测集就是秩落在半径球内的所有 $y$。

```mermaid
flowchart LR
    A[参考分布 F_U<br/>球内均匀/高斯] -->|凸势 φθ 的梯度<br/>PICNN 参数化| B[向量分位数<br/>Q_Y|X = ∇φ]
    B -->|Legendre 共轭<br/>c-transform| C[多元秩<br/>Q⁻¹ = ∇ψ]
    C -->|秩范数当共形分数<br/>S = ‖Q⁻¹‖| D[校准集求<br/>1-α 分位 ρ]
    D --> E[拉回球预测集<br/>‖Q⁻¹y,x‖ ≤ ρ]
    C -.可选 reranking R.-> E
```

### 关键设计

**1. 半对偶 + PICNN 把条件 VQR 变成单凸势优化（C-NQR）：** 出发点是 Carlier 等人给出的 OT 对偶——条件向量分位数由一对互为 Legendre 共轭的凸势 $\varphi(u,x),\psi(y,x)$ 决定，$Q_{Y|X}=\nabla_u\varphi$、$Q^{-1}_{Y|X}=\nabla_y\psi$。本文把对偶问题改写成只对**单个**凸势 $\varphi_\theta$ 的优化，通过 c-transform 自动得到共轭：目标 $\mathcal V(\theta)=\mathbb E_{F_U\otimes F_X}[\varphi_\theta(U,X)]+\mathbb E_{F_{YX}}[\varphi_\theta^*(Y,X)]$，其中共轭 $\varphi^*_\theta(y,x)=\max_u\{u^\top y-\varphi_\theta(u,x)\}$。$\varphi_\theta$ 用 **PICNN（部分输入凸网络）**参数化，对第一参数 $u$ 保持凸性，从而梯度天然循环单调、映射可逆。靠 Danskin 定理，对共轭求导只需 $\varphi_\theta$ 的导数。训练时用 **L-BFGS 精确求内层 argmax**（式 7），外层对 PICNN 做 SGD。这一版直接但贵——每个 mini-batch 的每个 $x$ 都要解一次内层凸优化。

**2. 摊销优化甩掉内层求解（AC-NQR）：** 为消除反复解 argmax 的开销，引入一个**摊销网络** $\tilde u_\vartheta(y,x)\approx\check u_{\varphi_\theta(\cdot,x)}(y)$ 直接预测近似最大化点，用二次损失逼着它对齐真实 argmax，并采用**两时间尺度**训练（摊销网络更新更快、$\varphi_\theta$ 更慢）。这样把「解优化」换成「前向预测一次」，实验里 AC-NQR 训练/推理都最快（约 8.9 sec/epoch、1.1 sec 推理 8192 点），是后续共形实验的默认基模型。作者还区分 $U$/$Y$ 两种参数化方向（下标 U/Y），因为对偶的对称性允许参数化任意一侧的势。

**3. 熵正则化的可扩展变体（EC-NQR）：** 当维度更高、解凸共轭仍嫌贵时，给原始 OT 问题加**熵正则项**，使目标平滑、共轭的 argmax 退化成 softmax 闭式，内层优化变成可采样逼近的期望，从而能用纯随机梯度求解、扩展性更好。代价是熵会引入偏差、可能扭曲分位映射几何，所以它是「扩展性 vs 几何保真」的折中选项。作者特别在附录用反例论证：**循环单调性不可丢**——不能拿普通非凸的 normalizing flow 替掉凸势，否则得不到统计上有意义的多元秩。

**4. 秩范数共形分数 + 拉回球的体积最优性：** 把多元秩范数 $S_i=\|\hat Q^{-1}_{Y|X}(Y_i,X_i)\|$ 当共形分数，取 $\{S_i\}$ 的 $\lceil(n{+}1)(1{-}\alpha)\rceil$ 阶统计量 $\rho_{1-\alpha}$，预测集 $\hat C^{pb}_\alpha(x)=\{y:\hat Q^{-1}_{Y|X}(y,x)\in B(0,\rho_{1-\alpha})\}$ 立刻享有有限样本边际覆盖 $\ge 1-\alpha$。这是 CQR 从一维到多维的自然推广。**理论亮点（定理 3）**：当逆传输的 Jacobian 行列式具径向结构（$\det\nabla_yQ^{-1}=j_x(\|Q^{-1}\|)$ 且 $r\mapsto\phi(r)j_x(r)$ 严格递减，涵盖椭圆/高斯情形）时，这个拉回球在所有满足条件覆盖 $\ge1-\alpha$ 的集合里**体积最小**，即它就是 $Y|X=x$ 的最高密度区域（HPD）——把「有效率」做到了理论极致。

**5. Reranking 修正各向异性（RPB）：** 拉回球隐含假设秩 $U=\hat Q^{-1}(Y,X)$ 径向对称；当模型被误设、秩呈各向异性时欧氏半径不可靠。本文把向量秩本身视作多元分数，套上 Thurin 等人的 OT-CP reranking 算子 $R$ 校正对参考分布 $F_U$ 的偏离，得到新分数 $S^{rpb}_i=\|R(U_i)\|$ 再做共形校准。实验显示 reranking 能让条件覆盖更锐利，但预测集体积变大，性价比存疑——侧面说明对作者的 VQR 模型而言「裸的 split conformal 已经够用」。

## 实验关键数据

### 主实验表格（生成质量，S-W2 越小越好；合成数据）

| 数据集 | AC-NQRU（本文） | VQR | FN-VQR | CPQ | CVQR |
|---|---|---|---|---|---|
| Star | **0.182** | 0.270 | 0.271 | 0.274 | 0.443 |
| Glasses | **0.771** | 1.964 | 2.017 | 0.931 | 1.170 |
| Banana | 0.073 | 0.389 | 0.398 | 0.237 | 0.401 |
| Convex Glasses | **0.657** | 1.961 | 1.954 | 0.793 | 0.953 |

- 训练时间 AC-NQRU 约 8.89 sec/epoch、推理 1.12 sec（8192 点），是所有变体里最快的；本文系列在多数数据集上 S-W2 同时拿到最佳/次佳。

### 消融实验表格（L2-UV，恢复真实分位算子，越小越好）

| 函数 | 数据集 | C-NQRU | C-NQRY | AC-NQRU | AC-NQRY | CPF |
|---|---|---|---|---|---|---|
| $Q^{-1}_{Y|X}$ | Convex Banana | 3.784 | 0.212 | **0.106** | 0.206 | 9.479 |
| $Q^{-1}_{Y|X}$ | Convex Glasses | 0.332 | **0.068** | 0.203 | 0.109 | 2.268 |
| $Q_{U|X}$ | Convex Banana | 7.665 | 0.660 | **0.545** | 0.569 | 16.537 |

- 本文模型在重建真实分位算子上对 CPF 等基线有数量级优势（如 Convex Banana 的 9.479 → 0.106）。

### 关键发现
- **共形实验（scm20d/sgemm/blog/bio，$\alpha=0.1$）**：PB / PBS 在 4 个数据集中的 3 个上同时给出有竞争力的条件覆盖**和**最小预测集体积（以 $\log V/d_y$ 衡量），明显优于 OT-CP、OT-CP+ 与局部椭球（ELL）。
- **残差版更稳**：在信号残差 $s=y-\hat f(x)$（$\hat f$ 为随机森林）上再拟合 VQR 的 PBS/RPBS 变体进一步提升表现，说明 VQR 与点预测器可正交组合。
- **Reranking 是把双刃剑**：RPB/RPBS 让 worst-slab 覆盖更锐利，但体积增大，作者判定对自家 VQR 模型「split conformal 就够」。
- **自适应性**：与 Thurin/Klein 等并发的离散 OT-CP 不同（其集合大小不随 $X$ 变），本文直接学神经 VQR、不依赖条件密度估计，预期在高维更占优。
- **多模态可扩展**：附录给出基于变量替换公式的密度型分数，能刻画 $F_{Y|X=x}$ 为高斯混合时的**不连通**几何，弥补球形拉回集的局限。

## 亮点与洞察
- **把三条线拧成一股绳**：神经 OT（学凸势）+ 向量分位数回归（多元秩）+ 共形预测（覆盖保证），各取所长——OT 给几何，VQR 给连续可逆映射，CP 给有限样本保证。
- **理论与实用兼得**：定理 3 把「拉回球 = HPD = 体积最优」钉死，给了「为什么用秩范数当分数」一个干净的最优性解释，而不只是工程启发。
- **摊销是关键工程支点**：AC-NQR 用一个前向网络替掉每 batch 的内层凸优化，是把 OT-VQR 规模化、进而能跑真实多目标回归基准的现实前提。
- **强调凸性不可省**：用反例说明不能拿任意 flow 顶替凸势，点出「循环单调性」才是多元秩有统计意义的根。
- **两种参数化方向对称**：可参数化 $\varphi_\theta$（绑 $F_U$）或 $\psi_\theta$（绑 $F_{Y|X}$），实验里两侧都试，给了实践者按数据几何选侧的余地。
- **可与点预测器解耦**：既能直接拟合 $y$，也能拟合随机森林残差 $y-\hat f(x)$，让 VQR 当「不确定性外壳」套在任意回归器上。

## 局限与展望
- **模型类受限于凸势**：PICNN 的凸性既是保证也是枷锁，表达力天花板可能在复杂多模态条件分布上吃亏（虽附录给了密度法处理多峰）。
- **熵正则的偏差**：EC-NQR 扩展性最好但会扭曲分位几何，高维下「扩展性 vs 保真」仍需更细的权衡。
- **响应维度仍偏低**：真实实验维度只到 16/4/2/2，「高维优势」更多是论证/预期而非大规模验证。
- **依赖可交换性假设**：共形保证建立在校准/测试样本可交换之上，分布漂移或时序场景下覆盖可能失效。
- **效率优势依赖建模正确**：定理 3 的体积最优只在径向/椭圆假设下成立，模型误设时拉回球未必最优，这也正是引入 reranking 的原因。
- **未来方向**：作者自陈要扩到更广的生成模型类、并探索高维下更紧的效率（体积）保证。

## 相关工作与启发
- **多元分位数谱系**：从空间分位数（Chaudhuri）、深度分位数（Hallin）到 OT 测度传输视角（Chernozhukov、Hallin & Konen），本文继承 Carlier 的 CVQF 并打破其「对 $X$ 仿射」假设。
- **神经 OT**：熵正则 Sinkhorn 一脉（Cuturi、Seguy）扩展性好但有偏；ICNN 凸势一脉（Amos、Makkuva、Bunne）保单调可逆，本文属后者但首次从「单一联合采样」学条件势。
- **多元共形**：相对逐坐标/标量化/生成嵌入（Feldman、Dheur）的启发式分数，本文给出显式生成模型 + 最优性理论；与并发的 OT-CP（Thurin）、熵 OT-CP（Klein）相比，本文的连续神经 VQR 自带自适应性、不需条件密度估计。
- **与 CQR 的关系**：把 Romano 等的一维 CQR「用残差经验分位替名义分位」原理直接搬到拉回集上，秩范数即多元版「残差」，是该思路最自然的高维推广。
- **启发**：「先用 OT 学一个把简单参考分布搬到复杂条件分布的可逆映射，再在参考空间里做几何简单的统计推断（球/秩）」是一个可迁移的范式，可推广到其它需要分布无关保证又想利用几何的任务。

## 评分
- **新颖性**: ⭐⭐⭐⭐ — 神经 OT 首次规模化到「条件」VQR，并把多元秩干净地接入共形预测，定理 3 的 HPD 最优性是扎实的理论贡献；但底层 OT/CVQF/共形各组件均有成熟前作，属高质量整合而非全新范式。
- **实验充分度**: ⭐⭐⭐ — 合成数据上生成质量与算子恢复对比详尽，真实多目标回归覆盖-体积权衡也站得住，但响应维度偏低、缺真正高维压力测试，未能充分验证「高维占优」的主张。
- **写作质量**: ⭐⭐⭐⭐ — 从一维直觉过渡到多元、再到共形，逻辑链清晰，理论与算法层次分明（C/AC/EC-NQR + PB/RPB）。
- **价值**: ⭐⭐⭐⭐ — 为「带覆盖保证又自适应几何」的多元不确定性量化提供了一个可扩展、有理论支撑的实用工具，对 UQ/共形预测社区有实际参考价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Neural Hamilton–Jacobi Characteristic Flows for Optimal Transport](neural_hamilton--jacobi_characteristic_flows_for_optimal_transport.md)
- [\[ICLR 2026\] HOTA: Hamiltonian Framework for Optimal Transport Advection](hota_hamiltonian_framework_for_optimal_transport_advection.md)
- [\[ICLR 2026\] Elastic Optimal Transport: Theory, Application, and Empirical Evaluation](elastic_optimal_transport_theory_application_and_empirical_evaluation.md)
- [\[ICLR 2026\] A Memory-Efficient Hierarchical Algorithm for Large-scale Optimal Transport Problems](a_memory-efficient_hierarchical_algorithm_for_large-scale_optimal_transport_prob.md)
- [\[ICLR 2026\] A Scalable Constant-Factor Approximation Algorithm for $W_p$ Optimal Transport](a_scalable_constant-factor_approximation_algorithm_for_w_p_optimal_transport.md)

</div>

<!-- RELATED:END -->
