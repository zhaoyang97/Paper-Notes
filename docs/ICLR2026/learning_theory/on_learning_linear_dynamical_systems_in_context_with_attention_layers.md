---
title: >-
  [论文解读] On learning linear dynamical systems in context with attention layers
description: >-
  [ICLR 2026][学习理论][in-context learning] 本文给出单层线性注意力在「噪声线性动力系统（LDS）」上下文学习任务里的**最优权重显式解**，证明它在一阶自回归近似（AR(1)）下等价于对自回归最小二乘损失做一步梯度下降，并通过实验把 AR(s)（$s\ge2$）的最优解结构与预条件共轭梯度（PCG）方法联系起来，从而为「Transformer 预测精度能逼平 Kalman 滤波」这一经验现象提供了理论解释。
tags:
  - "ICLR 2026"
  - "学习理论"
  - "上下文学习"
  - "Transformer 表达力"
  - "in-context learning"
  - "线性注意力"
  - "线性动力系统"
  - "系统辨识"
  - "梯度下降等价"
---

# On learning linear dynamical systems in context with attention layers

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=os7OLubIMI](https://openreview.net/forum?id=os7OLubIMI)  
**代码**: https://github.com/XHZhang01/icl-for-lds-data  
**领域**: 学习理论 / 上下文学习 / Transformer 表达力  
**关键词**: in-context learning, 线性注意力, 线性动力系统, 系统辨识, 梯度下降等价

## 一句话总结
本文给出单层线性注意力在「噪声线性动力系统（LDS）」上下文学习任务里的**最优权重显式解**，证明它在一阶自回归近似（AR(1)）下等价于对自回归最小二乘损失做一步梯度下降，并通过实验把 AR(s)（$s\ge2$）的最优解结构与预条件共轭梯度（PCG）方法联系起来，从而为「Transformer 预测精度能逼平 Kalman 滤波」这一经验现象提供了理论解释。

## 研究背景与动机

**领域现状**：把 Transformer 的上下文学习（ICL）看成「前向传播里执行隐式优化」是近年理论界的主流视角。已有大量工作证明：在 **i.i.d. 数据**（如各 token 独立的线性回归）下，单层线性自注意力的最优权重恰好实现对上下文诱导的最小二乘损失做（预条件）梯度下降，Ahn、Mahankali、von Oswald 等人把这个图景刻画得相当完整。

**现有痛点**：真实序列（语言、时间序列）几乎都不是 i.i.d.，而是**当前 token 统计上依赖整段历史**。一旦数据来自动力系统，i.i.d. 设定下的证明就失效了：损失里会冒出高阶数据矩，token 之间的统计耦合让一阶最优性条件难以求解。因此「非 i.i.d. 设定下注意力到底在做什么优化」基本是空白。

**核心矛盾**：经验上 Du 等人发现 GPT-2 在预测 LDS 下一观测时，精度能和 **Kalman 滤波（KF）**——这个已知系统参数下可证最优的预测器——打平，甚至在 KF 可证最优的区间也如此；但其内部机制完全没人解释清楚。理论（注意力做什么）和经验（为何这么强）之间缺一座桥。

**本文目标**：刻画「单层线性自注意力被训练去预测 LDS 序列下一观测 $y_T$」时，预训练损失的**全局最优权重**长什么样，并把它解释成对某个上下文损失的算法步骤。

**切入角度**：作者借用系统辨识里的**「非正常学习（improper learning）」**思路——不去估计 $A,c,\Sigma_w,\sigma_v$ 这些系统参数，而是直接把「下一观测」近似成「最近 $s$ 个观测的线性函数」，即一个 AR(s) 过程。在 KF 收敛条件下，这个自回归近似能以指数速度逼近真实条件期望 $\mathbb{E}[y_{t+1}\mid y_t,\dots,y_1]$，于是问题就化归为「带非 i.i.d. 噪声的线性回归」，从而能接上「注意力=优化器」的分析框架。

**核心 idea**：用「非正常学习的 AR(s) 自回归损失」当作上下文损失，证明线性注意力的最优解就是在这个损失上跑优化步——AR(1) 时是一步梯度下降，AR(s) 时是 PCG 型 Krylov 子空间迭代。

## 方法详解

### 整体框架

这是一篇**理论分析**论文，没有训练新模型，整条「方法」是一条**化归—求解—解释**的推理链，目标是把「线性注意力在 LDS 上学到了什么」翻译成「它在某个显式损失上跑了哪种优化算法」。整体分四步走：

1. **数据与近似**：序列由双重高斯噪声污染的 LDS（公式 $x_{t+1}=Ax_t+w_t,\ y_t=c^\top x_t+v_t$）产生。借 Tsiamis–Pappas 的结果把「未来观测」写成「过去 $s$ 个观测的线性组合 + 指数衰减偏置 + 噪声」，从而定义 AR(s) 自回归损失 $L_{\mathrm{AR}(s)}(w)$ 作为上下文目标。
2. **架构与 token**：只保留**单层、单头、无 MLP、带因果掩码的线性注意力**（去掉 softmax 与投影矩阵 $W_O$，把 $W_QW_K^\top$ 合并成 $W_{QK}$），并用前人的 token 构造把序列拼成输入矩阵 $Y_0$，只关心最后一个「测试 token」的预测 $\hat y_T$。
3. **求最优权重**：写出上下文损失的一阶最优性（驻点）条件，先用一个**带状稀疏结构引理**把候选最优权重缩到一个小参数类里，再在 AR(1) 情形下解出**闭式全局最优解**。
4. **解释**：把最优解代回前向传播——AR(1) 恰好是「从 0 出发对 $L_{\mathrm{AR}(1)}$ 做一步 GD」；AR(s)（$s\ge2$）则由实验观测到的权重稀疏模式 + Hessian/梯度的分块结构，对上「两步 PCG / 增广 Krylov」迭代。

整条链里真正贡献性的环节是：**AR(s) 损失化归（设计 1）、最优性条件的带状结构引理（设计 2）、AR(1) 闭式最优解定理（设计 3）、以及 AR(s) 的 PCG 解释（设计 4）**。下面逐个讲。

### 关键设计

**1. 把 LDS 预测化归为 AR(s) 自回归最小二乘：让非 i.i.d. 问题接上「注意力即优化」框架**

直接分析 LDS 数据的难点在于 $y_T$ 依赖整段历史 $x_0,w_{1:T},v_{1:T}$，损失里出现高阶矩。作者不走「先辨识系统参数再用 KF」的正常学习路线，而是采用非正常学习：把未来观测写成最近 $s$ 个观测的线性函数。基于 Tsiamis–Pappas 的展开式，在可观测、边缘稳定（$\rho(A)\le1$）等假设下，$\rho(A-kc^\top)<1$ 使偏置项随窗口 $s$ 指数衰减、可忽略，于是序列退化成一个 **AR(s) 噪声线性回归**，对应优化目标

$$\min_{w\in\mathbb{R}^s}\ L_{\mathrm{AR}(s)}(w):=\frac{1}{2(T-s-1)}\sum_{t=1}^{T-s-1}\big(y_{t+s}-w^\top\bar y_t\big)^2,$$

其中 $\bar y_t:=[y_t,\dots,y_{t+s-1}]^\top$。这一步的价值在于：它把「带历史耦合的非 i.i.d. 序列」变成了一个**凸**的最小二乘问题，从而能复用 i.i.d. 设定下「注意力实现对上下文最小二乘损失做优化」的整套分析。同时它给出了一个可证的精度承诺——Kozdoba 等人的结果表明，对任意有限 LDS 族存在窗口长度 $s(\varepsilon)$ 使 AR(s) 最优预测器的平均误差逼近最优 KF，这正是「Transformer 逼平 KF」的理论入口。

**2. 一阶最优性条件里的带状稀疏结构：把高维参数搜索缩到一个小类**

求最优权重要解损失的驻点方程，但因为数据全历史耦合，方程里项数爆炸、高阶矩难处理。作者把损失重写成只含两个有效参数块 $b$（来自 $W_V$）和 $z_1,\dots,z_s$（来自 $W_{QK}$）的形式，并发现驻点条件（公式 12）的右端因为 **$A$ 分布的中心对称性**而呈现一种**带状（banded）结构**：随 $s+j$ 的奇偶性在两种棋盘式零模式 $B_0(s,j)$、$B_1(s,j)$ 之间交替——即在 $r+l$ 为偶（或奇）的位置被强制为零。Lemma 4.1 据此构造出一类满足该零模式的参数（$W_{QK}$、$W_V$ 取特定 Kronecker 张量结构），证明这类参数能让方程左端复现右端的稀疏零结构。其意义是：它把「在整个权重空间里找最优解」缩小成「在这个结构化小类里找」，对 $s\ge2$ 时定位可证最优解至关重要；实验进一步验证训练得到的极小点确实落在这个稀疏模式上（用支撑集的 Jaccard 距离度量）。

**3. AR(1) 闭式全局最优解 = 一步梯度下降：第一个 LDS 数据下的最优性结果**

在 $s=1$（AR(1)）情形，作者用 Lemma 4.1 的结构简化 + **Isserlis 定理**（把高斯数据的高阶矩拆成二阶矩乘积）攻克全历史依赖，给出单层线性自注意力关于损失 $L(\theta)$ 的**全局最优权重闭式解**（公式 15，差一个非零常数缩放）：

$$W^\star_{QK}=\begin{pmatrix}\dfrac{(T-2)\,\mathbb{E}[y_{T-1}y_T\sum_{i=1}^{T-2}y_iy_{i+1}]}{\mathbb{E}\!\big[y_{T-1}^2(\sum_{i=1}^{T-2}y_iy_{i+1})^2\big]}&0\\[2pt]0&0\end{pmatrix},\qquad W^\star_V=\begin{pmatrix}0&0\\0&1\end{pmatrix}.$$

关键洞察是：用这组最优参数跑一次前向传播，输出**恰好等于**从 $w_0=0$ 出发对 $L_{\mathrm{AR}(1)}(w)$ 做一步梯度下降后给出的预测。这就把 i.i.d. 设定下「注意力实现 GD」的结论**首次**推广到了 LDS 产生的非 i.i.d. 数据上，是已知第一个针对 LDS 数据的最优性结果，也直接为 GPT-2 逼平 KF 的现象提供了机制假说。

**4. AR(s)（$s\ge2$）的 PCG / 增广 Krylov 解释：超出一步 GD 的更强算法**

对 $s\ge2$，简单算一下前向传播会发现：最优权重**不再**实现标准 GD。作者注意到前向传播里的因子 $\frac{1}{T-s-1}\bar Y$ 具有有意义的分块结构（公式 16）——它把 $L_{\mathrm{AR}(s)}$ 的 **Hessian $\nabla^2 L_{\mathrm{AR}(s)}$、零点梯度 $\nabla L_{\mathrm{AR}(s)}(0)$** 和一个标量 $\gamma$ 排成一个对称分块矩阵。结合 Lemma 4.1 的参数结构，注意力诱导的预测器可写成（公式 17）

$$P\nabla^2 L_{\mathrm{AR}(s)}\,q+\xi_1 P\nabla L_{\mathrm{AR}(s)}(0)+\xi_2\,p,$$

而对 $L_{\mathrm{AR}(s)}$ 以 $P^{-1}$ 为预条件、从 $w_0=0$ 跑**两步 PCG** 得到的预测器（公式 18）是 $\tau_1 P\nabla^2 L_{\mathrm{AR}(s)}P\nabla L_{\mathrm{AR}(s)}(0)+\tau_2 P\nabla L_{\mathrm{AR}(s)}(0)$。两者的第二项在缩放意义下一致；第一项都落在 Krylov 子空间方向 $P\nabla^2 L_{\mathrm{AR}(s)}$ 上，若把 PCG 的共轭方向初始化为 $q$ 则首项也对齐——实验测得两方向的余弦相似度在 AR(4) 上达 $0.88\pm0.05$（设定 a）和 $0.93\pm0.01$（设定 b）。额外的 $p$ 向量则像**增广 Krylov 方法**里的「修正方向」，用来补偿 $P\nabla^2 L_{\mathrm{AR}(s)}$ 的病态模式，并顺带允许 $P$ 非对称；测得它与预条件残差方向反对齐（余弦 $\approx-0.99$），说明它确实把预测器往降残差方向推。这个解释自洽地涵盖了 AR(1)：因为一维协变量下 PCG 变体退化为 GD。

### 损失函数 / 训练策略
训练在线生成数据：每条轨迹采样不同的 $A,c,x_0$，跑 $T=30$ 步、隐状态维度 $d=5$，每个 iteration 重新采一批 LDS；目标是上下文损失 $L(\theta)=\mathbb{E}\big[\tfrac12(T_\theta(Y_0)_{s+1,T-s}-y_T)^2\big]$。优化器用 AdamW + 梯度裁剪 + 线性 warmup 后接 cosine 退火，共 8000 步，窗口越大 batch 越大（AR(1) 从 3000 起），结果对 3 个随机种子取平均。

## 实验关键数据

实验是**纯验证理论**性质的（没有跟其他模型刷 benchmark），核心是看「训练得到的最优注意力权重是否符合 Theorem 4.1 / Lemma 4.1 预测的结构」。

### 主结果：最优权重符合理论预测

| 设定 | 窗口 | 验证的理论 | 现象 |
|------|------|-----------|------|
| (a) $A$ 对角、$c=\mathbf{1}$ | AR(1) | Theorem 4.1 | $W^\star_{QK}$、$W^\star_V$ 收敛到定理给出的单元素结构（图 1b,c） |
| (a) | AR(2–4) | Lemma 4.1 | 权重落在带状奇偶交替的稀疏模式上（图 1e,f,h,i,k,l） |
| (b) $A=Q^\top\mathrm{diag}(v)Q$ 一般正交 | AR(1–4) | Thm 4.1 + Lemma 4.1 | 同样吻合（附录图 3） |
| (c) 非各向同性 $\Sigma_w$ | AR(1–3) | Lemma 4.1 | 吻合（附录图 4） |
| (d) $A=P^{-1}\mathrm{diag}(v)P$ 非正规 | AR(1–3) | Lemma 4.1 | 吻合（附录图 5） |

### 关键发现
- **AR(1) 闭式解可被训练复现**：在所有四种数据设定下，AdamW 训练出的 $W^\star_{QK}$/$W^\star_V$ 都收敛到 Theorem 4.1 的结构，说明「最优=一步 GD」不只是理论存在性，而是优化器实际能找到的解。
- **带状稀疏模式稳健**：$s\ge2$ 时权重的非零位置稳定落在 Lemma 4.1 预测的奇偶棋盘格上，作者用支撑集 Jaccard 距离量化了这一收敛。
- **PCG 解释有数值支撑**：AR(4) 上注意力预测方向与两步 PCG 方向余弦相似度 $0.88\sim0.93$，修正向量 $p$ 与残差方向余弦 $\approx-0.99$，说明「注意力在做 PCG 型 Krylov 迭代」这一解释站得住脚。

## 亮点与洞察
- **把非 i.i.d. 难点转嫁给系统辨识工具**：用「非正常学习 + AR(s) 近似」这一步，巧妙地把全历史耦合的难题转成凸最小二乘，使 i.i.d. 设定的整套「注意力即优化器」分析得以复用——这是全文最关键的一招。
- **中心对称性 → 带状结构**：把 $A$ 分布的对称性这个看似纯技术的假设，转化成最优性条件里可利用的稀疏零模式，从而把高维搜索缩成小参数类，是「用分布对称性换可解性」的漂亮案例。
- **Isserlis 定理破高阶矩**：面对高斯数据的高阶矩，直接调用 Isserlis（Wick）定理做矩分解，是处理非 i.i.d. 期望损失的可迁移技巧。
- **从 GD 升级到 PCG 的视角**：揭示「窗口越大，注意力实现的隐式优化越强（GD→PCG→Krylov）」，为「单层注意力为何能逼平 KF」给出了算法层面的解释，这个「数据窗口决定隐式算法阶数」的直觉可迁移到其他 ICL 分析。

## 局限与展望
- **AR(s)（$s\ge2$）只有结构刻画 + 经验解释，没有闭式最优解**：Theorem 4.1 只完整解决了 AR(1)；$s\ge2$ 仅由 Lemma 4.1 缩小了候选类、并靠实验对上 PCG，缺一个像定理那样的可证全局最优。作者把它列为最紧迫的后续工作（可先在 $\bar Y$ 为 Toeplitz 的平稳 LDS 区间下简化求解）。
- **设定理想化**：单层、单头、线性注意力、无 MLP、噪声为高斯、$A$ 分布需中心对称——离实际多层 softmax Transformer 还有距离；PCG 解释也只在 AR(4) 等少数情形做了余弦相似度抽查，不是严格证明。
- **训练目标简化**：用的是 few-shot 上下文损失（只预测最后一位），未覆盖工程里常用的「对所有位置求预测误差」的标准因果预训练目标；作者把扩展到该目标、以及对多层 Transformer 的 Krylov 类比做经验验证列为未来方向。

## 相关工作与启发
- **vs i.i.d. 线性回归 ICL（Ahn / Mahankali / von Oswald / Zhang）**：他们证明 i.i.d. 高斯协变量下单层线性注意力的全局最优实现（预条件）GD；本文把这条线**首次推到非 i.i.d. 的 LDS 数据**，难点在于 token 的全历史依赖，靠 AR(s) 化归 + Isserlis 定理解决。
- **vs Cole et al. (2025)**：他们也处理噪声 LDS，但给的是「存在性构造」——一个至少两层、首层固定的 attention-only Transformer，且无训练能恢复它的保证、无实验佐证；本文聚焦**单层权重的可证最优性**且有实验验证，互为补充。
- **vs Sander et al. (2024) / von Oswald et al. (2023b)**：他们用标准因果预训练损失刻画无噪声 $y_{t+1}=Ay_t$ 下的最优解；本文处理的是**双重高斯噪声**的完整系统 (1)，更贴近真实。
- **vs Transformer 模拟 KF（Goel & Bartlett / Akram & Vikalo / Du et al.）**：前两者构造能逼近/模拟 KF 的注意力，但依赖已知系统参数与精巧 token 增广；Du 等人给出 GPT-2 逼平 KF 的经验现象却未解释机制。本文不假设已知参数，从「ICL 即隐式优化」角度为这一经验现象提供了首个机制假说。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个 LDS 非 i.i.d. 数据下单层线性注意力的可证最优性结果，并把最优解与 GD/PCG 对应起来。
- 实验充分度: ⭐⭐⭐⭐ 多种数据设定下验证了 AR(1) 闭式解与 AR(s) 稀疏结构，但 PCG 解释仅靠少量余弦相似度抽查、未严格证明。
- 写作质量: ⭐⭐⭐⭐ 化归—求解—解释的逻辑清晰，符号体系完整；但理论密度高，对非系统辨识背景读者门槛偏大。
- 价值: ⭐⭐⭐⭐⭐ 在「Transformer 表达力 / ICL 即优化」理论与「Transformer 逼平 KF」经验之间架了第一座桥，奠定后续 $s\ge2$ 与多层分析的基础。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] A New Approach to Controlling Linear Dynamical Systems](a_new_approach_to_controlling_linear_dynamical_systems.md)
- [\[ICLR 2026\] Understanding In-Context Learning on Structured Manifolds: Bridging Attention to Kernel Methods](understanding_in-context_learning_on_structured_manifolds_bridging_attention_to_.md)
- [\[ICLR 2026\] In-Context Algorithm Emulation in Fixed-Weight Transformers](in-context_algorithm_emulation_in_fixed-weight_transformers.md)
- [\[ICLR 2026\] Pretrain–Test Task Alignment Governs Generalization in In-Context Learning](pretraintest_task_alignment_governs_generalization_in_in-context_learning.md)
- [\[ICLR 2026\] Transformers with Endogenous In-Context Learning: Bias Characterization and Mitigation](transformers_with_endogenous_in-context_learning_bias_characterization_and_mitig.md)

</div>

<!-- RELATED:END -->
