---
title: >-
  [论文解读] Continuum Transformers Perform In-Context Learning by Operator Gradient Descent
description: >-
  [ICLR2026][学习理论][Transformer] 这篇论文给"连续 Transformer"（处理无限维函数输入、用于 PDE 代理建模的 Transformer 变体）的上下文学习现象提供了首个理论刻画：证明它在前向传播中等价于在一个**算子 RKHS** 上做梯度下降，无限深度时恢复贝叶斯最优预测子，并且这套实现梯度下降的参数恰好是预训练目标的稳定点。
tags:
  - "ICLR2026"
  - "学习理论"
  - "上下文学习"
  - "神经算子"
  - "Transformer"
  - "算子 RKHS"
  - "算子梯度下降"
  - "贝叶斯最优预测子"
---

# Continuum Transformers Perform In-Context Learning by Operator Gradient Descent

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=X63V2CWjj3](https://openreview.net/forum?id=X63V2CWjj3)  
**代码**: https://github.com/yashpatel5400/opicl  
**领域**: 学习理论 / 上下文学习 / 神经算子  
**关键词**: 上下文学习, 连续 Transformer, 算子 RKHS, 算子梯度下降, 贝叶斯最优预测子

## 一句话总结
这篇论文给"连续 Transformer"（处理无限维函数输入、用于 PDE 代理建模的 Transformer 变体）的上下文学习现象提供了首个理论刻画：证明它在前向传播中等价于在一个**算子 RKHS** 上做梯度下降，无限深度时恢复贝叶斯最优预测子，并且这套实现梯度下降的参数恰好是预训练目标的稳定点。

## 研究背景与动机
**领域现状**：标准 Transformer 早已被观察到具备"上下文学习"（in-context learning, ICL）能力——不更新任何参数，仅把若干训练样本 $(x^{(i)}, y^{(i)})$ 放进上下文窗口，模型在新任务上的预测精度就能提升。一系列理论工作（Akyürek 2022、Garg 2022、Dai 2022、Cheng et al. 2023）证明了这背后的机制：在特定的 $W_k, W_q, W_v$ 选择下，一次前向推理等价于对上下文任务做若干步梯度下降，其中 Cheng et al. 2023 进一步用核方法把它刻画成在某个 RKHS 上的函数梯度下降。

**现有痛点**：所有这些结果都被限制在**标准 Transformer**上，即处理有限维向量输入。但机器学习里有一个看似正交的子领域——用神经算子加速求解偏微分方程（PDE）。这里序列的元素不再是有限维向量，而是**无限维函数**。为此，Calvello et al. 2024 提出了"连续 Transformer"（continuum transformer），把注意力推广到函数输入；更令人意外的是，这类 Transformer **同样表现出上下文学习**——把若干 PDE 解对放进上下文窗口，就能高效求解相关的新 PDE，催生了"上下文算子网络"（ICON）这一分支。

**核心矛盾**：尽管 ICON 在经验上效果很好，这种**泛化的、函数式的上下文学习从未被理论刻画**过。有限维下"前向传播 = 梯度下降"的证明工具（经典表示定理、有限维高斯过程、对矩阵求导的优化分析）无法直接搬到无限维函数空间，因为很多在有限维下显然的步骤（梯度的显式形式、对参数求导、无穷层组合的收敛性）在算子空间里都要重新严格化。

**本文目标**：把 Cheng et al. 2023 这条"ICL = 核梯度下降"的理论线，从有限维向量空间提升到无限维算子空间，回答三个子问题——连续 Transformer 的前向传播在做什么？无限深度时收敛到什么？以及这套参数能否通过训练自然得到？

**切入角度**：作者发现，只要把连续注意力里的相似度非线性从"标量值"改造成"算子值"，整个上下文学习过程就能被装进一个**算子 RKHS** 的框架里分析，从而能复用（推广后的）表示定理和希尔伯特空间上的高斯测度这两件数学工具。

**核心 idea**：证明连续 Transformer 的逐层推理 = 在算子 RKHS 上做逐步**算子梯度下降**，无限深度恢复贝叶斯最优预测子，且实现这一行为的参数是训练目标的稳定点；并顺带交付一套能让"理论家凭有限维直觉、用我们的无限维结论严格背书"的数学框架。

## 方法详解

### 整体框架
这是一篇纯理论论文，主线是把"上下文学习 = 梯度下降"的刻画从有限维 Transformer 平移到处理无限维函数的连续 Transformer。整篇论文的逻辑链可以这样鸟瞰：先**重新建模**连续注意力层，把相似度非线性 $H$ 从标量值改写成算子值，使其能在算子 RKHS 里被分析（这是后续一切的前提）；然后证明在一组特定参数下，**逐层推理恰好等于逐步算子梯度下降**（定理 3.1）；接着证明**无限深度极限下这套梯度下降收敛到贝叶斯最优预测子**（命题 3.3）；最后证明实现梯度下降的那组参数**确实是训练目标的稳定点**（定理 3.6），从而闭环——训练能把模型推到"做梯度下降"的配置上。

设定上，神经算子要学的是函数空间之间的映射 $\widehat{G}: \mathcal{A}\to\mathcal{U}$（多数情形是 PDE 的时间推演算子，输入输出在同一希尔伯特空间 $X$）。连续注意力（Calvello et al.）把标准注意力的 $W_k, W_q, W_v$ 矩阵换成**线性算子**，实践中用 FNO 式的核积分变换实现，即 $W_q x_i = \mathcal{F}^{-1}(R_q \odot \mathcal{F} x_i)$，其中 $R_q$ 是查询核的傅里叶参数化。上下文窗口仿照有限维 ICL：把 $n$ 个输入输出函数对 $(f^{(i)}, u^{(i)})$ 排成一列，最后一列的输出位置填 0，模型从第 $(n{+}1)$ 个输入 $f$ 预测 $u^{(n+1)}$。

### 关键设计

**1. 算子值非线性：把连续注意力改写成能在 RKHS 里分析的形式**

原始连续注意力 $\mathrm{ContAttn}(X) = (W_v X)\,M\,\mathrm{softmax}((W_q X),(W_k X))$ 里，注意力权重矩阵仍然落在 $\mathbb{R}^{n\times n}$，即相似度是**标量值**的。作者指出：如果照搬这种标量相似度 $H: Q^{n+1}\times K^{n+1}\to \mathbb{R}^{(n+1)\times(n+1)}$，整个问题**无法被刻画成 RKHS 上的梯度下降**，后续分析就无从下手。于是他们做了一个并不显然的推广——让非线性取**算子值**：$H: Q^{n+1}\times K^{n+1}\to (\mathcal{L}(V))^{(n+1)\times(n+1)}$，其中 $\mathcal{L}(V)$ 是 $V\to V$ 的有界线性算子集合（标量 $c$ 被理解为算子 $c\cdot\mathrm{Id}$，所以这包含了原 softmax 形式作为特例）。基于此，$m$ 层连续 Transformer 的逐层更新写成

$$Z_{\ell+1} = Z_\ell + \Big(\widetilde{H}(W_{q,\ell} X_\ell, W_{k,\ell} X_\ell)\, M\, (W_{v,\ell} Z_\ell)^{\top}\Big)^{\top},$$

其中键/查询算子只作用在输入函数行 $X_\ell$ 上，值算子作用在完整的 $Z_\ell$ 上，掩码 $M$ 取分块 $\big[\begin{smallmatrix}I_{n\times n}&0\\0&0\end{smallmatrix}\big]$。这一步"算子值非线性"看似只是符号上的推广，实则是整篇论文能成立的钥匙——作者明确说这本身就是一项可供社区复用的建模贡献。上下文损失定义为 $L(W_v,W_q,W_k)=\mathbb{E}\big[\,\|[Z_{m+1}]_{2,n+1}+u^{(n+1)}\|_X^2\,\big]$（加号源于最终预测带负号的约定）。

**2. 算子梯度下降等价：逐层推理 = 在算子 RKHS 上走一步梯度下降**

上下文任务是从样本 $\{(f^{(i)},u^{(i)})\}$ 求 $O^* = \arg\min_O \sum_i \|u^{(i)} - O f^{(i)}\|_X^2$，自然可以用迭代 $O_{\ell+1}=O_\ell - \eta_\ell \nabla L(O_\ell)$ 求解。定理 3.1 给出核心等价：设 $\kappa: X\times X\to\mathcal{L}(X)$ 是任意**算子值核**、$\mathcal{O}$ 是它诱导的算子 RKHS，令 $O_\ell$ 为第 $\ell$ 步算子梯度下降的结果（$O_0=0$）。那么存在标量步长 $r'_0,\dots,r'_m$，使得当连续 Transformer 取参数

$$[\widetilde{H}(U,W)]_{i,j}=\kappa(u^{(i)},w^{(j)}),\quad W_{v,\ell}=\begin{bmatrix}0&0\\0&-r'_\ell I\end{bmatrix},\quad W_{q,\ell}=I,\quad W_{k,\ell}=I$$

时，对任意测试函数 $f$ 都有 $T_\ell(f)= -O_\ell f$。换言之，**第 $\ell$ 层的上下文预测恰好等于 $\ell$ 步算子梯度下降的结果**。证明思路与 Cheng et al. 2023 平行，但关键难点在于：算子空间上梯度下降的显式表达式不像有限维向量空间那样直接可得，作者必须调用**广义的表示定理（Representer Theorem）**——把经典表示定理推广到算子 RKHS——才能把这一步的梯度写成可与 Transformer 逐层推理对齐的显式形式。

**3. 无限深度恢复贝叶斯最优预测子：把高斯测度搬上希尔伯特空间**

有限维下，Cheng et al. 证明若输出来自核 $\kappa$ 的高斯过程边缘、深度 $m\to\infty$，Transformer 恢复贝叶斯最优预测子。要在算子设定下复现这一结论，必须先定义"采样真实算子 $O$ 的高斯测度"。作者借助 Jorgensen & Tian 2024 把高斯过程推广到希尔伯特空间：称 $U\,|\,F\sim\mathcal{N}(0, K(F))$ 是 $\kappa$ 高斯随机变量，当 $[K(F)]_{i,j}=\kappa(f^{(i)},f^{(j)})$ 且任意投影 $\langle v, u^{(i)}\rangle_X$ 都服从对应方差的高斯（定义 3.2）。命题 3.3 随即证明：在 $U|F$ 是 $\kappa$ 高斯随机变量、且非线性 $\widetilde{H}$ 与核 $\kappa$ 匹配的条件下，当层数 $m\to\infty$，连续 Transformer 的预测收敛到**最优线性无偏预测子（BLUP）**；而由希尔伯特空间 kriging 理论（Menafoglio & Petris、Luschgy），BLUP 在 MSE 意义下恰好就是**贝叶斯最优预测子**。这里的技术新意在于：要小心处理无穷层组合 $T_\infty := \dots\circ T_m\circ\dots\circ T_0$ 的收敛性，并把它与希尔伯特空间 kriging 的已知结果接上——这两步都是无限维下独有的难点。

**4. 预训练收敛到梯度下降参数：希尔伯特空间泛函上的梯度流分析**

前两个结果都假设"参数已经长成梯度下降的样子"。定理 3.6 补上最后一环：那组实现算子梯度下降的参数其实是**训练目标的稳定点**，因此训练过程会自然把模型推向它。证明难点在于，要对"希尔伯特空间上泛函"做梯度流分析，需要用比"对矩阵求导"更一般的 **Fréchet 可微性**，并把训练目标改写成一个等价的期望表达（需小心操控数据分布的协方差算子）。作者在一个**旋转对称**假设下完成分析（假设 3.4：存在自伴可逆算子 $\Sigma$，使分布在 $\Sigma^{1/2} M \Sigma^{-1/2}$ 变换下不变）：把分析放到被 $\Sigma^{-1/2}$ 旋转的参考系里做，那里数据分布和注意力权重都被保持，于是结论可平移回原坐标。最终得到的固定点形如 $W_{q,\ell}=b_\ell \Sigma^{-1/2}$、$W_{k,\ell}=c_\ell \Sigma^{-1/2}$，其中 $W_v$ 取 $\big[\begin{smallmatrix}0&0\\0&r_\ell I\end{smallmatrix}\big]$；当对称算子 $\Sigma=I$ 时，正好退化回定理 3.1 里执行梯度下降的那套参数配置。

### 损失函数 / 训练策略
上下文损失见上文 $L(W_v,W_q,W_k)=\mathbb{E}\big[\|[Z_{m+1}]_{2,n+1}+u^{(n+1)}\|_X^2\big]$。定理 3.6 把训练表述为最小化该损失对 $(r, W_q, W_k)$ 的梯度范数之和（Hilbert–Schmidt 范数 $\|\cdot\|_{HS}$），并证明在前述固定点处该和为 0，即这些参数是稳定点。

## 实验关键数据
实验目的不是刷指标，而是**经验验证三条理论断言**。数据生成统一在 $X=L^2(\mathbb{T}^2)$ 上：用 Hilbert–Schmidt 积分算子核 $[\kappa(f^{(1)},f^{(2)})u](y)=k_x(f^{(1)},f^{(2)})\int k_y(y',y)u(y')\,dy'$ 构造算子值核，函数从协方差为 $\alpha(-\Delta+\beta I)^{-\gamma}$ 的高斯随机场采样。

### 主实验

| 实验 | 验证的理论 | 设置 | 结论 |
|------|-----------|------|------|
| BLUP 收敛（图 1） | 命题 3.3 | 固定定理 3.1 参数，4 组 $(k_x,k_y)$ 核，50 次独立采样 | 当非线性与数据生成核匹配时，上下文损失随层数单调下降，并收敛到 BLUP（式 15）的误差水平 |
| Poisson 方程（图 2） | 命题 3.3 鲁棒性 | 2D Poisson $\Delta u=f$，真实核未知，FFT 解析求解器算 $u$ | 即使无法精确选核，参数仍呈现期望的优化特性；线性核表现最优；BLUP 近乎完美（Poisson 解算子对 $f$ 线性） |
| 参数收敛（图 3） | 定理 3.6 | 250 层连续 Transformer，随机抽 10 层算两两余弦相似度，5 次训练 | 键/查询算子的两两 HS 余弦相似度（式 16）随训练 $\to 1$，验证固定点刻画 |

### 关键发现
- **核匹配是最优性的前提**：只有当注意力非线性 $\widetilde{H}$ 与数据生成核 $\kappa$ 一致时，每加一层才对应"多走一步算子梯度下降"，损失才单调下降并触达 BLUP；附录 I 进一步显示此时预测场在结构上贴合真值 $Of$。
- **鲁棒性强于理论保证**：图 3 中即便某些 $k_x$ 核**违反**了证明所需的假设（仅线性核满足 Assumption G.4），参数仍稳健地两两收敛——暗示定理 3.6 可能存在更强的、与层 $\ell$ 无关的版本。
- **可复现**：BLUP 实验跨 50 次算子采样、优化实验跨 5 次训练初始化都稳定复现，说明结论对随机性不敏感。

## 亮点与洞察
- **"算子值非线性"是四两拨千斤的建模选择**：把标量相似度换成算子值，看似只改了符号，却是让整个无限维问题落进算子 RKHS 框架的唯一入口——作者诚实地把这件事本身当作一项独立贡献，因为它决定了后续能不能用表示定理。
- **一套可迁移的"有限→无限"证明工具箱**：广义表示定理、希尔伯特空间上的高斯测度、泛函上的梯度流（Fréchet 可微）三件套，让原本只在有限维成立的 ICL 理论能干净地提升到函数空间。作者明确希望经典优化社区能"凭有限维直觉、用这套无限维结论背书"，不必再纠缠有限投影带来的误差收敛细节。
- **理论直接指向实践改进**：与语言模型不同，这里 RKHS 由 PDE 参数分布诱导，因此可以**针对具体 PDE 元学习任务估计 $\kappa$ 并直接用它参数化 $\widetilde{H}$**，把"理解 ICL"变成"改进 ICL"的可操作路径。

## 局限与展望
- **理论依赖一组结构性假设**：旋转对称（假设 3.4）、$\widetilde{H}$ 的不变性（假设 3.5）等是这类优化分析的惯例，但现实 PDE 数据未必满足；实验里违反假设仍收敛，说明理论尚未完全解释观察到的鲁棒性。
- **核选择在实践中困难**：最优性建立在"非线性匹配数据核"之上，而真实 RKHS 的核往往难以估计；作者把"如何系统地选 $\kappa$"留作未来工作。
- **定理 3.6 可能偏弱**：实验提示存在与层 $\ell$ 无关的更强收敛结论，证明它将是有价值的推广。
- **验证规模有限**：实验局限于 $L^2(\mathbb{T}^2)$ 上的合成高斯随机场与 Poisson 方程，尚未在更复杂、更高维或非周期边界的真实 PDE 基准上检验。

## 相关工作与启发
- **vs Cheng et al. 2023（有限维 ICL = 核梯度下降）**：本文的直接母本。Cheng 证明有限维 Transformer 逐层推理等价于在标量 RKHS 上做函数梯度下降并恢复贝叶斯最优；本文把同一刻画提升到**无限维算子空间**，需要把经典表示定理、有限维高斯过程、对矩阵求导分别换成广义表示定理、希尔伯特空间高斯测度、泛函梯度流——并非平凡照搬。
- **vs Cole et al. 2024（连续 ICL 的另一条线）**：同样研究连续 Transformer 的 ICL，但关注的是线性椭圆 PDE 的**样本复杂度与泛化**，与本文"前向传播在做什么（梯度下降机制刻画）"是互补而非重叠的视角。
- **vs Calvello et al. 2024（连续 Transformer 架构）**：提供了被本文分析的架构对象；本文在其连续注意力上引入算子值非线性这一推广，使其可被理论刻画。
- **vs ICON 系列（Cao 2024、Yang 2023、Meng 2025）**：这些工作在连续注意力上搭建上下文算子网络并展示了强经验性能，但都"只观察、未刻画"；本文正是补上其缺失的理论解释。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个对连续 Transformer 上下文学习的理论刻画，并交付一套可复用的无限维分析工具
- 实验充分度: ⭐⭐⭐⭐ 三组实验精准对应三条理论断言且可复现，但仅限合成 GRF 与 Poisson，缺更复杂真实 PDE
- 写作质量: ⭐⭐⭐⭐ 逻辑链清晰、对证明难点的取舍交代诚实，但数学密度高、依赖大量附录
- 价值: ⭐⭐⭐⭐⭐ 既解释了 ICON 为何有效，又给出"估计 $\kappa$ 来改进 PDE 元学习"的可操作方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Transformers Learn Latent Mixture Models In-Context via Mirror Descent](transformers_learn_latent_mixture_models_in-context_via_mirror_descent.md)
- [\[ICLR 2026\] Interactive Learning of Single-Index Models via Stochastic Gradient Descent](interactive_learning_of_single-index_models_via_stochastic_gradient_descent.md)
- [\[ICLR 2026\] Transformers with Endogenous In-Context Learning: Bias Characterization and Mitigation](transformers_with_endogenous_in-context_learning_bias_characterization_and_mitig.md)
- [\[ICLR 2026\] Adversarially Pretrained Transformers May Be Universally Robust In-Context Learners](adversarially_pretrained_transformers_may_be_universally_robust_in-context_learn.md)
- [\[ICLR 2026\] In-Context Algorithm Emulation in Fixed-Weight Transformers](in-context_algorithm_emulation_in_fixed-weight_transformers.md)

</div>

<!-- RELATED:END -->
