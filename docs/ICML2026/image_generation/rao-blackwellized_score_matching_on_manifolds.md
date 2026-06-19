---
title: >-
  [论文解读] Rao-Blackwellized Score Matching on Manifolds
description: >-
  [ICML 2026][图像生成][去噪 score matching] 当数据分布落在嵌入流形 $M\subset\mathbb{R}^D$ 上时，环境空间高斯加噪做 DSM 学到的切向目标含有方差以 $d/\sigma^2$ 发散的法向噪声通道；本文证明对最近点投影 $\pi(X)$ 做一次 Rao-Blackwell 条件化即可干净地去掉这个奇异通道，并把剩下的目标精确展开为「内蕴 Riemannian score + $\sigma^2$ 阶 Tweedie 校正 + $\sigma^2$ 阶 Weingarten/Ricci 外蕴曲率校正」。
tags:
  - "ICML 2026"
  - "图像生成"
  - "去噪 score matching"
  - "流形假设"
  - "Rao-Blackwell"
  - "黎曼 score"
  - "外蕴曲率"
---

# Rao-Blackwellized Score Matching on Manifolds

**会议**: ICML 2026  
**arXiv**: [2605.25567](https://arxiv.org/abs/2605.25567)  
**代码**: 待确认  
**领域**: 扩散模型 / 流形上的 score matching / 生成式建模理论  
**关键词**: 去噪 score matching, 流形假设, Rao-Blackwell, 黎曼 score, 外蕴曲率

## 一句话总结
当数据分布落在嵌入流形 $M\subset\mathbb{R}^D$ 上时，环境空间高斯加噪做 DSM 学到的切向目标含有方差以 $d/\sigma^2$ 发散的法向噪声通道；本文证明对最近点投影 $\pi(X)$ 做一次 Rao-Blackwell 条件化即可干净地去掉这个奇异通道，并把剩下的目标精确展开为「内蕴 Riemannian score + $\sigma^2$ 阶 Tweedie 校正 + $\sigma^2$ 阶 Weingarten/Ricci 外蕴曲率校正」。

## 研究背景与动机

**领域现状**：score-based 生成模型靠 DSM 在带噪样本上回归残差 $(Z-X)/\sigma^2$，配合 Tweedie 公式刻画 $\nabla\log p_\sigma$。但实际数据普遍服从「流形假设」——分布集中在低维子流形 $M$ 上，对环境 Lebesgue 测度奇异，所以严格意义上的 $\nabla\log q$ 根本不存在，DSM 只在 $\sigma>0$ 时良定。

**现有痛点**：两条路线都不令人满意。**内蕴方法**（RSGM、Riemannian SDE）改用流形上的 Brownian motion，但需要 exp map / 热核模拟等流形专属基础设施，对一般嵌入流形几乎不可用；**环境方法**继续跑 Euclid DSM，仅靠经验启发——「投到切空间后 $\sigma\to 0^+$ 应该会收敛到内蕴 score」，但既没有刻画到底学到了什么，也无法解释为什么 $\sigma$ 接近 0 时泛化界普遍崩坏。

**核心矛盾**：环境 DSM 的切向回归目标 $T_\sigma=P_T(\pi(X))(Z-X)/\sigma^2$ 在 $\sigma\to 0^+$ 时条件方差以 $d/\sigma^2$ 发散——这不是参数化的副作用，而是法向纤维上的高斯噪声本身贡献的不可压缩噪声通道。任何对 $T_\sigma$ 的直接回归都被这个发散方差"喂垃圾"。

**本文目标**：(i) 在环境 DSM 框架内给出一个统计上规范、信号无损、方差有界的切向目标；(ii) 把它展开到 $\sigma^2$ 阶，看清楚它与真正的内蕴 Riemannian score 差多少、差在哪里。

**切入角度**：注意到法向纤维上的噪声只通过 $X-\pi(X)\in N_{\pi(X)}M$ 这一维进入观测，而切向信号可以被最近点投影 $\pi(X)$ 完整保留。这启发用 $\pi(X)$ 作为充分统计量做 Rao-Blackwell：把 $T_\sigma$ 对 $\pi(X)$ 条件化，凡是只依赖法向纤维噪声的成分都会被条件期望平均掉。

**核心 idea**：定义 $r_\sigma(z)=\mathbb{E}[T_\sigma\mid\pi(X)=z]$。一句话——"用 nearest-point projection 做一次 Rao-Blackwell 把奇异法向噪声碾平，剩下的就是 $O(\sigma^2)$ 接近内蕴 score 的规范目标"。

## 方法详解

### 整体框架

要解决的是流形数据上"环境 DSM 切向目标被法向噪声污染"这个问题：数据 $Z\sim q\,d\mathrm{Vol}_M$ 落在紧致 $C^5$、正 reach 的嵌入子流形 $M\subset\mathbb{R}^D$（维度 $d$）上，加噪 $X=Z+\sigma\xi$（$\xi\sim\mathcal{N}(0,I_D)$），对小 $\sigma$ 事件 $X\in\mathrm{Tub}_{r_0}(M)$ 以 $1-e^{-c/\sigma^2}$ 概率成立，故最近点投影 $\pi:\mathrm{Tub}_{r_0}(M)\to M$ 几乎处处良定。本文不改训练流程，而是把"应该回归什么目标"这件事彻底想清楚：用 $\pi(X)$ 作充分统计量对原始切向目标做一次 Rao-Blackwell 条件期望，先证明所得的规范目标在统计上最优、方差有界，再把它展开到 $\sigma^2$ 阶看清它离真正的内蕴 score 差在哪。

### 关键设计

**1. Rao-Blackwell 化的规范切向目标 $r_\sigma$：把奇异噪声通道平均掉**

原始环境方法直接对切向 DSM 目标 $T_\sigma=P_T(\pi(X))(Z-X)/\sigma^2$ 回归，等价于让网络逼近一个方差发散的目标——本质上是把法向高斯噪声当成监督信号硬学。本文的关键观察是法向噪声只通过 $X-\pi(X)\in N_{\pi(X)}M$ 一维进入观测，而切向信号被 $\pi(X)$ 完整保留，于是可以把 $T_\sigma$ 改写成 $\sigma^{-2}P_T(\pi(X))(Z-\pi(X))$，再沿 $\pi(X)$ 做条件期望，定义规范目标 $r_\sigma(z)=\mathbb{E}[T_\sigma\mid\pi(X)=z]$，它是一个只依赖 $z\in M$ 的切向场 $r_\sigma:M\to TM$。在 $\pi(X)=z$ 上做 $L^2$ 投影分解可证：对任意切向场 $h$ 有 $\mathcal{R}_\sigma(h)=\mathcal{R}_\sigma(r_\sigma)+\mathbb{E}\|r_\sigma(\pi(X))-h(\pi(X))\|^2$，故 $r_\sigma$ 是唯一的风险极小化器（Theorem 4.1）；更一般地，对任何比 $\pi(X)$ 更粗的 fiber-collapsing 统计量 $S$（$\sigma(S)\subseteq\sigma(\pi(X))$），分解式 $\mathbb{E}\|T_\sigma-\eta\|^2=\mathbb{E}\|T_\sigma-r_\sigma(\pi(X))\|^2+\|r_\sigma(\pi(X))-\eta_S\|^2+\|\eta_S-\eta\|^2$ 表明 $\pi(X)$ 是"最细的折叠纤维统计量"、$r_\sigma$ 是它配套的最优预测器，正是经典 Rao-Blackwell 定理在流形 DSM 设定下的对应版本。这一步只需要 $\pi(X)$ 这个比 exp map 廉价得多的几何操作，就同时拿到了内蕴方法的信号纯净度和环境方法的零基础设施成本。

**2. 方差坍缩定理与 $d/\sigma^2$ 的 Bayes 下界：证明这一步不可省**

为了量化"做不做 Rao-Blackwell 差多少"，本文在管状坐标里把 $T_\sigma$ 拆成切向信号加法向噪声两部分，法向部分在条件分布上是各向同性高斯且与 $\pi(X)$ 独立，直接算得 $\mathrm{Var}(T_\sigma\mid\pi(X)=z)=d/\sigma^2+O(1)$；再由全方差公式 $\mathrm{Var}(T_\sigma)=\mathrm{Var}(r_\sigma(\pi(X)))+d/\sigma^2+O(1)$ 看出 $r_\sigma(\pi(X))$ 方差有界（$O(1)$），而 $T_\sigma$ 方差按 $d/\sigma^2$ 发散（Theorem 4.2）。同一分解还给出对任何 $S$-可测预测器 $\eta$ 的下界 $\inf_\eta\mathbb{E}\|T_\sigma-\eta\|^2\geq d/\sigma^2+O(1)$，且等号只在 $\sigma(S)=\sigma(\pi(X))$ 时取到。这把"为什么必须先做投影再回归"从经验启发抬到了信息论层面：不做这步，无论网络多大、训得多久都要为发散方差付出 $d/\sigma^2$ 的不可降低风险；做了这步方差立刻塌缩到 $O(1)$，监督信噪比在 $\sigma\to 0$ 时仍然有意义。Figure 1 在 $S^2$ + von Mises-Fisher 上验证了 $\log\mathbb{E}\|T_\sigma\|^2$ 对 $\log\sigma$ 的斜率正是 $-2$，而 $r_\sigma$ 曲线持平在 $\mathbb{E}\|\nabla_M\log q\|^2$。

**3. 外蕴 $\sigma^2$ 校正展开：看清规范目标离内蕴 score 差在哪**

最后本文在 graph 坐标下对 $r_\sigma$ 做 Bayes 计算，把诱导体积元修正和管状 Jacobian 的平均曲率修正分别算清，得到精确到二阶的展开 $r_\sigma(z)=\nabla_M\log q(z)+\sigma^2[b_q(z)+g_M^{\mathrm{ext}}(z)]+o(\sigma^2)$（Theorem 5.2）：领头项就是真正的内蕴 Riemannian score，偏置由两块构成——内蕴 Tweedie 项 $b_q(z)=\tfrac{1}{2}\nabla_M[\Delta_M\log q+\|\nabla_M\log q\|^2](z)$，以及外蕴曲率项 $g_M^{\mathrm{ext}}(z)=(\tfrac{1}{2}W_{H(z)}-\mathrm{Ric}_z^\sharp)\nabla_M\log q(z)$，其中 $W_u$ 是法向 $u$ 方向的 Weingarten 算子、$H(z)$ 是平均曲率向量、$\mathrm{Ric}_z^\sharp$ 是 Ricci 自同态。在球面 $S^d$ 上代入 $W_\nu=-\mathrm{Id}$、$H=-dz$、$\mathrm{Ric}^\sharp=(d-1)\mathrm{Id}$，外蕴系数塌缩成标量 $\alpha_d=1-d/2$（Corollary 5.4），即 $\alpha_1=+1/2,\alpha_2=0,\alpha_3=-1/2,\alpha_4=-1$。这条展开一次回答了两个老问题：环境 DSM 学到的就是内蕴 score 加一个完全显式、可由两个曲率张量算出的偏置；而 $S^2$ 上之所以"看起来效果出奇地好"，是因为它恰是 Einstein 流形、$\tfrac{1}{2}W_H=\mathrm{Ric}^\sharp=\mathrm{Id}$ 让外蕴项相消、只剩 $\sigma^2 b_q$ 这一内蕴偏置——这个相消在 $S^1$、$S^3$、$S^d(d\geq3)$ 都不再成立，理论预言的二阶偏置应当可被实验测到。平坦情形 $M=V$ 下两项都为零，环境 DSM 严格退化为低维 Gaussian DSM（Proposition 5.1），给整套理论提供了干净基线。

### 损失函数 / 训练策略

本文是 population-level identification theorem，没有定义新损失。隐含训练策略是把网络回归目标从原始切向残差 $T_\sigma$ 换成 $r_\sigma$ 的有限样本估计 $\widehat{r}_{\sigma,i}$，例如对邻域样本做局部线性回归（Appendix G 给出 $O(N^{-2/(d+4)})$ 的非参收敛率）。Figure 3(b) 在多个 Einstein 流形上以相同 MLP 架构和训练预算对比"回归 $T_{\sigma,i}$ vs 回归 $\widehat{r}_{\sigma,i}$"，后者 score MSE 显著更低且差距随 $d$ 增大而扩大，与 Theorem 4.2 中的 $d/\sigma^2$ 因子完全一致。

## 实验关键数据

> 这是纯理论论文，实验仅用于数值验证。三张关键 figure 充当"主结果"。

### 主实验（数值验证理论预言）

| 验证目标 | 设定 | 预言 | 数值结果 |
|---|---|---|---|
| 方差坍缩率（Theorem 4.2） | $S^2$ + vMF$(\mu,\kappa=2)$ | $\log\mathbb{E}\|T_\sigma\|^2$ 对 $\log\sigma$ 斜率 $-2$；$r_\sigma$ 持平 | 黑线斜率 $-2$（$d=2$ 时 $d/\sigma^2$）；蓝线持平于 $\mathbb{E}\|\nabla_M\log q\|^2$（Fig 1） |
| 外蕴系数（Corollary 5.4） | $S^1,S^2,S^3,S^4,T^2$，$\sigma\in\{0.05,0.06,0.08\}$ | $\alpha_1=+1/2,\alpha_2=0,\alpha_3=-1/2,\alpha_4=-1$；$T^2$ 取 $+1/2$ | Gauss-Hermite 数值 $\alpha_{\mathrm{ext}}$ 落在预测值上（Fig 2） |
| 采样去偏（Corollary 5.4 应用） | 闭式 Langevin drift，$\sigma=0.3$ | 用 $(1-\sigma^2\alpha_d)(1+\sigma^2\alpha_d)\nabla_M\log q$ 可纠正环境 DSM 偏差 | 去偏 drift（蓝）的平衡分布密度匹配内蕴 score（黑），原 ambient drift（橙）偏离（Fig 3a） |

### 消融实验（监督目标对比）

| 目标 | 流形 | Score MSE 趋势 | 说明 |
|---|---|---|---|
| 回归 $T_{\sigma,i}$（原始） | 多个 Einstein 流形 | 高，随 $d$ 显著恶化 | 受 $d/\sigma^2$ 发散方差污染 |
| 回归 $\widehat{r}_{\sigma,i}$（RB） | 同上同 MLP 同预算 | 显著更低，随 $d$ 增大优势扩大 | 验证 Theorem 4.2 的预测 |
| 平坦情形 $M=V$（Proposition 5.1） | $\mathbb{R}^d$ 嵌入 | 严格退化为低维 Gaussian DSM | 两项校正都消失，给出干净基线 |

### 关键发现

- **方差坍缩在 $S^2$ 上斜率精确为 $-2$**：直接确认 $d/\sigma^2$ 不是宽松上界而是渐近精确率，且对常用流形 DSM testbed 给出可量化的"先做 Rao-Blackwell 再回归"价值。
- **$T^2$ 是关键的非球对照**：环面内蕴平坦（$\mathrm{Ric}=0$），但外蕴系数仍预测为 $+1/2$ 且数值确认——证明偏置真的来自嵌入方式，而不是内蕴曲率伪影。
- **$S^2$ 的相消是 Einstein 流形的巧合**：$\tfrac{1}{2}W_H=\mathrm{Ric}^\sharp=\mathrm{Id}$ 才让外蕴项归零，$d\neq 2$ 时偏置存在且方向取决于 $\alpha_d$ 的符号——预言了为何 $S^3$ 上环境 DSM 系统性向"反向"偏移。

## 亮点与洞察

- **把 Rao-Blackwell 定理搬到流形 DSM 的"正确版本"**：传统 Rao-Blackwell 需要"充分统计量 + 无偏估计"的设定，本文用 $L^2$ 投影分解 + fiber-collapsing summary 的语言把它包装成一个对管状邻域上观测都成立的工具，挑出 $\pi(X)$ 这个嵌入流形天然给出的"最细折叠统计量"，论证逻辑很干净。
- **把 $\sigma\to 0$ 的发散从"经验上的脆弱"升级为"信息论意义上的不可避免"**：$d/\sigma^2$ 不是某个算法的弱点，而是任何只看 $\pi(X)$ 的预测器的 Bayes 下界——这给"先做投影再回归"的工程实践提供了硬理论支撑。
- **外蕴校正项 $\sigma^2(\tfrac{1}{2}W_H-\mathrm{Ric}^\sharp)\nabla_M\log q$ 是可计算的**：意味着工程上有一条新选择——继续用 Euclid pipeline 训练（最便宜），训完再把 closed-form 外蕴校正减回去，就能拿到接近内蕴 score 的目标，不需要任何热核模拟。这给 ambient vs intrinsic 之争一个全新的中间路线。
- **$S^2$ 这个"看起来效果好"的现象终于有了非偶然的解释**：所有在 $S^2$ 上做 manifold DSM 的论文其实都活在一个 Einstein 巧合里，迁移到非 $d=2$ 的球面或一般曲面时会立刻显现 $\sigma^2$ 阶系统偏置——这条预言可以指导后续 benchmark 设计。

## 局限与展望

- **作者承认的局限**：population-level 结果，没给完整 finite-sample minimax 率（只在 Appendix G 给了非参局部线性回归的初步收敛率）；要求紧致、$C^5$、正 reach 的嵌入流形，对一般非紧或低正则流形的扩展未触及。
- **本文未覆盖的局限**：所有结果都建立在"$\pi(X)$ 可计算"上，对未知流形的实际工作流意味着先要做 manifold estimation/projection，这本身又是一个有 sample complexity 的问题；外蕴校正项要求显式的 $W_H$、$\mathrm{Ric}^\sharp$，对未知流形得先做曲率估计，闭环工作流尚未给出。另外整篇没有真正在图像/3D mesh 等高维实际数据上的端到端实验，所有验证都在低维显式流形上。
- **改进思路**：(i) 把 $\widehat{r}_\sigma$ 的非参估计替换为更高效的 kernel ridge 或 neural local estimator，给出可证收敛率的端到端 manifold DSM 训练方案；(ii) 用 $\sigma^2$ 校正项做一个"后处理"模块插到已有 RSGM/ambient-DSM 上，量化校正后采样质量提升；(iii) 把推导推广到带边流形（如 simplex 上的概率分布），覆盖 categorical diffusion 等场景。

## 相关工作与启发

- **vs Riemannian SGM (Bortoli et al., 2022)**：他们用流形上的热核扩散完全避开环境奇异，本文走环境路线但通过 Rao-Blackwell 拿到同样的 $O(\sigma^2)$ 接近内蕴 score 的目标。优势：不需要 exp map / 热核模拟；劣势：受 $\sigma^2 g_M^{\mathrm{ext}}$ 外蕴偏置约束，必须显式去除才能匹敌 RSGM。
- **vs Pidstrigach (2022) / Bortoli (2023) 流形假设下的收敛分析**：那两篇刻画 ambient score 在流形附近的"对齐法向"行为，本文进一步定量回答"对齐之后切向部分到底学到什么"。
- **vs Vincent (2011) 经典 DSM + Tweedie 公式**：本文把"密度对 Lebesgue 测度绝对连续"这条隐含假设暴露出来，给出在流形支撑上的对应版本——$r_\sigma$ 取代 Tweedie score 成为新的规范目标。
- **vs Levy-Jurgenson et al. (2026) 环境投影启发**：他们经验性地用"投到切空间"作为环境 DSM 的后处理，本文给出该启发的统计学原理（Rao-Blackwell）和精确偏置（$\sigma^2 g_M^{\mathrm{ext}}$）。
- **启发迁移**：Rao-Blackwell + fiber-collapsing 框架天然适用于任何"观测含可分离 nuisance 通道"的回归设定——例如对带噪 3D 点云做表面法向估计、对含 view-dependent 噪声的多视图特征做投影回归，都可以套用类似的"选最细可计算充分统计量、做条件期望"的逻辑去掉显式干扰维度。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Rao-Blackwell + $\pi(X)$ 这条路线在流形 DSM 文献里没人正面走过，外蕴 $\sigma^2$ 校正项是新的可验证理论结果。
- 实验充分度: ⭐⭐⭐ 数值验证支持理论但只在低维显式流形上，缺图像/真实高维数据的端到端验证。
- 写作质量: ⭐⭐⭐⭐⭐ 三个定理一条主线推得非常干净，$S^2$ 巧合的解释给得很漂亮。
- 价值: ⭐⭐⭐⭐ 给"为什么 ambient DSM 在 $S^2$ 上好用、迁移到别处会偏"提供了完整解释；为后续 manifold-aware 训练/采样去偏提供了 closed-form 工具。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Alignment-Guided Score Matching for Text-to-Image Alignment in Diffusion Models](alignment-guided_score_matching_for_text-to-image_alignment_in_diffusion_models.md)
- [\[ICML 2026\] Riemannian MeanFlow for One-Step Generation on Manifolds](riemannian_meanflow_for_one-step_generation_on_manifolds.md)
- [\[ICCV 2025\] Balanced Image Stylization with Style Matching Score](../../ICCV2025/image_generation/balanced_image_stylization_with_style_matching_score.md)
- [\[NeurIPS 2025\] A Connection Between Score Matching and Local Intrinsic Dimension](../../NeurIPS2025/image_generation/a_connection_between_score_matching_and_local_intrinsic_dimension.md)
- [\[ICML 2026\] Offline Multi-agent Reinforcement Learning via Sequential Score Decomposition](offline_multi-agent_reinforcement_learning_via_sequential_score_decomposition.md)

</div>

<!-- RELATED:END -->
