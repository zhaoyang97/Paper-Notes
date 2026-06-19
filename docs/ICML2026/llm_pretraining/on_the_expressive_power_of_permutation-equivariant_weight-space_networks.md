---
title: >-
  [论文解读] On the Expressive Power of Permutation-Equivariant Weight-Space Networks
description: >-
  [ICML 2026 Spotlight][预训练][weight-space learning] 本文为操作在 MLP 权重上的置换等变 weight-space 网络（DWS / NFN / GMN / NG-GNN 等）建立了首个系统的表达力理论，证明这些架构在表达力上几乎完全等价，并在"general position"假设下对四种逼近场景（函数空间泛函/算子、置换不变泛函、置换等变算子）给出了普适性刻画；由理论得出的简单修改 OCE（输出端 ensemble 多个 MLP）在 INR 编辑基准上相对 SOTA 提升 34%。
tags:
  - "ICML 2026 Spotlight"
  - "预训练"
  - "weight-space learning"
  - "permutation equivariance"
  - "universality"
  - "INR editing"
  - "OCE"
---

# On the Expressive Power of Permutation-Equivariant Weight-Space Networks

**会议**: ICML 2026 Spotlight  
**arXiv**: [2602.01083](https://arxiv.org/abs/2602.01083)  
**代码**: https://github.com/dayanadir/capacity_increase_inr_editing_experiment  
**领域**: 权重空间学习 / 表达力理论 / 等变神经网络  
**关键词**: weight-space learning, permutation equivariance, universality, INR editing, OCE

## 一句话总结
本文为操作在 MLP 权重上的置换等变 weight-space 网络（DWS / NFN / GMN / NG-GNN 等）建立了首个系统的表达力理论，证明这些架构在表达力上几乎完全等价，并在"general position"假设下对四种逼近场景（函数空间泛函/算子、置换不变泛函、置换等变算子）给出了普适性刻画；由理论得出的简单修改 OCE（输出端 ensemble 多个 MLP）在 INR 编辑基准上相对 SOTA 提升 34%。

## 研究背景与动机

**领域现状**：weight-space learning 把训练好的神经网络当成"结构化数据"，用一个 meta-network 直接吃另一个 MLP 的参数 $v=(W_1,b_1,\dots,W_L,b_L)$，做精度预测、INR 编辑、meta-optimization 等下游任务。由于 MLP 隐藏层神经元置换 $\tau$ 满足 $f_{\rho(\tau)v}=f_v$（函数不变），主流 SOTA 架构（DWS / NP-NFN / HNP-NFN / GMN / NG-GNN / NFT）均显式构造为对 $G_A=S_{d_1}\times\cdots\times S_{d_{L-1}}$ 置换等变的。

**现有痛点**：(1) 这些架构看起来差别很大（DWS 是手工对齐的等变线性层，GMN/NG-GNN 是把网络当成图跑 GNN，NFT 是 attention），但社区不清楚它们到底谁强谁弱；(2) 对称性约束本身会削弱表达力，但既有理论仅给出零散的 forward-pass 模拟结论 (Navon et al., 2023; Lim et al., 2023; Kalogeropoulos et al., 2024)，没有统一的 universality 刻画；(3) 一些自然的目标，例如 INR 的"放大/zoom-out"算子，输出函数的复杂度可能超过输入 MLP 的容量上限，从理论上看根本不可能用同架构输出去逼近——但这件事此前没有被明确指出。

**核心矛盾**：weight-space 的逼近问题天然横跨两层语义——一边是参数空间 $\mathcal V$，一边是它们实现的函数空间 $\mathcal F$。"等变 weight-to-weight 映射"和"函数空间算子"是不同的目标类型，必须分开讨论；而对称性约束在某些设定下足够，在另一些设定下不够（即非普适），需要精确刻画"边界"。

**本文目标**：(a) 把所有主流置换等变 weight-space 架构在表达力意义下放进同一个等价类；(b) 把"逼近目标"系统分成 4 类并给出 universality / non-universality 的完整图谱；(c) 把理论翻译成可落地的架构改动。

**切入角度**：作者注意到，几何深度学习里其它对称域（图、点云）常用"在退化子集 $\mathcal E$ 外才普适"的 *general position*（GP）范式 (Maron et al., 2020; Finkelshtein et al., 2025)；权重空间中天然的退化集合就是"某一隐藏层存在两个相同偏置 $b_i=b_j$"——这是 Lebesgue 测度零的子集，几乎所有训练得到的 MLP 都落在 GP 上。以 GP 为抓手，等变性带来的退化就能被分离出来。

**核心 idea**：先证所有主流架构表达力等价 → 把分析对象统一成"通用置换等变 weight-space 网络" → 在 4 个逼近场景里逐个判定 universality；同时由"函数空间算子在固定架构下不普适"这一不可能性结论，反推出一个简单破解方法：**让输出比输入更大**（OCE 直接输出 $k$ 个 MLP 取平均）。

## 方法详解

本文是纯理论 + 一个理论驱动的简单架构改动，"方法"指的是定理框架与证明骨架，而不是一个新模型。

### 整体框架

全文围绕一张二维表展开：横轴是"想逼近什么目标"，纵轴是"输入是否落在 general position（GP）上"。目标按 Definition 4.1 分成四类——函数空间泛函 $\Psi:\mathcal C(X,\mathbb R^{d_L})\to\mathbb R^n$（如精度预测、INR 分类）、置换不变泛函 $\Psi:\mathcal V_A\to\mathbb R^n$（如权重 $\ell_2$ 范数、loss landscape 曲率）、函数空间算子 $\Psi:\mathcal C\to\mathcal C$（如 INR 编辑、domain adaptation）、置换等变算子 $\Psi:\mathcal V_A\to\mathcal V_A$（如 pruning mask 预测、meta-optimization 的梯度预测）。输入域则分整个参数空间 $\mathcal V$ 与去掉退化集合的 GP 子集 $\mathcal V\setminus\mathcal E$，其中 $\mathcal E_A=\{v\mid\exists\ell\in[L-1],\,i\ne j,\,(b_\ell)_i=(b_\ell)_j\}$ 是"某隐藏层存在两个相同偏置"的测度零集合。逼近误差对泛函与等变算子用 $L^2$、对函数空间算子用 $L^\infty$ 度量，后者即要求权重到权重映射 $\Phi$ 经 realization map $R(v)=f_v$ 拉回函数空间后满足 $\sup_v\|\Psi(f_v)-f_{\Phi(v)}\|_\infty<\epsilon$。整套论证先证所有主流架构表达力等价、把分析对象统一，再逐格判定普适性，最后由一条不可能性结论反推出可落地的修复。

### 关键设计

**1. 架构等价定理：把六个看起来天差地别的架构折叠成一个等价类（Theorem 5.2 + Proposition 5.3）**

社区一直说不清 DWS（手工对齐的等变线性层）、GMN/NG-GNN（把网络当图跑 GNN）、NFT（attention）这些架构到底谁强谁弱。本文直接证明它们表达力几乎相同：对任意紧集 $K\subseteq\mathcal V$ 和任意 $\pi,\pi'\in\Pi\setminus\{\text{NFT}\}$（$\Pi=\{\text{DWS, NP-NFN, HNP-NFN, GMN, NG-GNN, NFT}\}$），都有 $\mathcal N^\pi_{\text{inv}}(K)=\mathcal N^{\pi'}_{\text{inv}}(K)$ 且 $\mathcal N^\pi_{\text{equi}}(K)=\mathcal N^{\pi'}_{\text{equi}}(K)$，证明手法是让一个架构的基本层去"模拟"另一个架构的基本层（mutual approximation）。NFT 因为 attention 机制非标准，在全空间上掉队（Proposition 5.3 给出反例 $K$），但只要把输入限制在 GP 子集 $K\subset\mathcal V\setminus\mathcal E$ 上，它也回到等价类。这一步的意义是把"该用哪个架构"从一个理论选型问题降级为纯工程偏好，后续所有定理只需针对"通用置换等变 weight-space 网络"陈述一次，且分析时挑最方便的 DWS 来证就够了。

**2. GP 假设下的四象限普适性地图：精确标出哪些场景现成架构够用、哪些必须换架构（Theorems 6.1 / 6.3 / 7.2 / 7.4）**

四类目标各被判定一次，结论彼此不同，GP 假设是把"理论非普适"与"实际几乎总能逼近"两边都说清楚的关键抓手。**函数空间泛函**（Thm 6.1）在全空间 $K\subseteq\mathcal V$ 上就普适，证明是先用 DWS 能模拟 MLP forward pass (Navon et al., 2023) 推出 separation 性质——DWS 区分不开 $v,v'$ 就意味着 $f_v=f_{v'}$——再套 Pacini et al. (2025b) 的 separation-to-approximation 定理收尾。**置换不变泛函**（Prop 6.2 + Thm 6.3）在全空间下反而不普适：可以构造 $v,v'$ 使 $W_2,W_2'$ 秩不同却被 1-WL 区分不开（Figure 3），但限制到 GP 上就普适。Thm 6.3 的核心构造是**连续 canonization 映射** $\operatorname{canon}:K\to\mathcal V$——因 $K\cap\mathcal E=\varnothing$，每层偏置 $b_\ell$ 元素两两不同，用 $\operatorname{argsort}(b_\ell)$ 当作该层置换就能得到既唯一又连续的轨道代表，而 DWS 自带 DeepSets 原语、对 ranking 普适 (Segol & Lipman, 2019)，再接一个 MLP head 即可。**函数空间算子**（Prop 7.1 + Thm 7.2）在固定 ReLU 架构上不普适：ReLU MLP 的 linear region 数被 (Montúfar et al., 2014) 卡死，放大 / zoom-out 这类会增加几何复杂度的算子根本不可能用同容量的输出去逼近；但只要放开"输出架构 $A$ 足够大"就普适，证明分三步——用 partition of unity 把 $\Psi(f_v)$ 写成 $M$ 个参考 MLP 的连续凸组合，用 canonization 转成置换等变，再套 Thm 7.4。**置换等变算子**（Prop 7.3 + Thm 7.4）与不变情形对偶：通过 broadcasting，invariant 普适性蕴含 equivariant 普适性，Thm 7.4 用"广播 canonization" $\widetilde{\operatorname{canon}}(v)$（把 $\operatorname{Flat}(\operatorname{canon}(v))$ 拼到每个权重 / 偏置 entry 上）加 pointwise MLP 完成构造。四格全部判定后就拼出 Figure 1 的"表达力地图"，既告诉实践者哪些场景现成架构够用，也精确标出哪些场景需要新架构，避免悲观的非普适结论被无差别滥用。

**3. OCE（Output Capacity Expansion）：把 Thm 7.2 的"输出要更大"做成几乎零成本的即插即用改动（Section 8）**

Prop 7.1 的不可能性根子在于"输出 MLP 容量 $=$ 输入 MLP 容量"，导致输出端的 linear region 数不够。OCE 的破解极其朴素：在任何 weight-space 网络的最后特征维加一个 $k>1$ 的维度，把输出张量解读成 $k$ 个并行 MLP 的参数，最终预测取这 $k$ 个 MLP 输出的平均。因为共享了 backbone、只把输出 head 通道扩 $k$ 倍，参数量基本不变、代码上几乎只改一行；而 ensemble 后等效的 ReLU region 数翻了 $k$ 倍，恰好绕开容量瓶颈，并且天然保持置换等变（每个分支各自等变）。这相当于把第 2 点里"输出架构要更大"的理论指引直接兑换成工程提升。

### 损失函数 / 训练策略
理论部分无训练目标；OCE 实验沿用 INR dilation benchmark (Zhou et al., 2023a) 的标准 MSE 监督。

## 实验关键数据

实验只有一个：MNIST INR dilation benchmark，用于验证 OCE 带来的 SOTA 增益，从而间接验证 Thm 7.2 的实践价值。

### 主实验

| 方法 | 参考 | MSE ($\times 10^{-2}$, ↓) |
|------|------|---------------------------|
| NFT | Zhou et al. 2023b | 5.10 ± 0.04 |
| NP-NFN | Kofinas et al. 2024 | 2.55 ± 0.00 |
| NG-GNN-64 | Kofinas et al. 2024 | 2.06 ± 0.01 |
| ScaleGMN-B | Kalogeropoulos et al. 2024 | 1.89 ± 0.00 |
| NG-T-64 | Kofinas et al. 2024 | 1.75 ± 0.01 |
| ScaleGMN + GradMetaNet++ | Gelberg et al. 2026 | 1.60 ± 0.01 |
| DWS (k=1, baseline) | Gelberg et al. 2026 | 2.29 ± 0.01 |
| GMN (k=1, baseline) | Gelberg et al. 2026 | 1.96 ± 0.02 |
| **DWS + OCE (k=8)** | This paper | **1.36 ± 0.03** |
| **GMN + OCE (k=8)** | This paper | **1.06 ± 0.13** |

GMN+OCE 相对此前 SOTA（ScaleGMN+GradMetaNet++ 的 1.60）下降 34%；DWS 和 GMN 自身相对 $k=1$ 分别下降 41% 和 46%。

### 消融实验

附录 Table 2 的趋势（正文转述）：

| 配置 | 关键现象 | 说明 |
|------|---------|------|
| DWS, $k=1\to 8$ | MSE 下降 ~41% | 不增加额外参数（共享 backbone） |
| GMN, $k=1\to 8$ | MSE 下降 ~46% | 验证 Thm 7.2 的"扩输出架构"指引 |
| 对照 baseline | 大量利用 gradient / probe 等额外信号 | OCE 不用任何额外信号即超过 |

### 关键发现

- **理论 → 实验闭环**：性能瓶颈定位为"输出表示容量不足"而非"backbone 弱"，这是 Prop 7.1 直接预言并由 OCE 验证的。
- **OCE 是 weight-space 学习的 free lunch**：参数量几乎不变、对 DWS / GMN 都通用、不需要额外监督信号，却能拉开数十百分点的 MSE 差距，对未来基线设置有强示范作用。
- **NFT 出乎意料地排名靠后**（5.10 vs DWS+OCE 1.36），暗示 attention 在 weight-space 的优势远不如在序列建模——这与 Prop 5.3 中"NFT 在全空间下与其它架构不等价"的理论观察相呼应。

## 亮点与洞察

- **把一堆看起来差异巨大的架构折叠成一个等价类**，是这篇 paper 最高密度的洞察：以后做 weight-space 研究时，"选 DWS 还是 GMN 还是 NG-GNN"几乎只剩工程偏好；论文里只需选一个最方便分析的（这里是 DWS）来证一切。
- **GP 假设的双重用法**：既用来把"反例"和"普适"切开（Prop 6.2 vs Thm 6.3 配对出现），又用来把 NFT 拉回等价类（Prop 5.3）。"在测度零退化集外讨论普适性"这套方法论可以迁移到其它对称域（如带共享参数的 transformer、scale-equivariant 网络）。
- **连续 canonization 是一把万能钥匙**：因为 GP 下 $\operatorname{argsort}(b_\ell)$ 唯一且局部常数，连续 canonization 自然存在；这把"等变普适性"问题归约为"DeepSets 普适性"，证明骨架因此异常干净，几乎所有 4 个 cell 都共用这一步。
- **Prop 7.1 与 OCE 的工程意义**：以前社区觉得"INR 编辑做不上去"是模型不够强；本文指出根本原因是输出 MLP 容量被锁死了。这个观察直接催生了 OCE 这种"几乎零成本"的改动——可类比到任何"输入小网络→输出小网络"的 meta-learning 设定。
- **与过参数化的隐喻**：作者明确把"扩大输出架构"与"过参数化更易优化、泛化更好" (Du et al., 2019; Belkin et al., 2019) 类比，提示 weight-space 学习的下一个突破口可能在 input-smaller-than-output 的非对称设计上。

## 局限性 / 可改进方向

- **只覆盖 MLP 权重空间**，不含 transformer / conv 权重（虽然作者在附录 H 给了 transformer 的延伸 sketch），CNN 上的卷积+池化对称群尚未处理。
- **明确排除了 scale-equivariant 架构**（如 ScaleGMN, Kalogeropoulos et al. 2024; Tran et al. 2024），但实验 baseline 里 ScaleGMN 表现最强，理论与实践之间有一段缺口。
- **理论只谈表达力，不谈优化与泛化**——作者自己也强调这点；某个映射"可被某网络逼近"不等于"梯度下降能找到它"，在 weight-space 这种 loss landscape 极不规则的领域，gap 可能很大。
- **不可能性结论依赖 ReLU**（Prop 7.1 用 linear region 计数），其它激活下需要重新构造退化函数族；论文承认"相信能推广"但未给完整证明。
- **OCE 的 ensemble 数 $k$ 是超参**，且只在 INR dilation 这一个 benchmark 上验证，更多 function-space operator 任务（domain adaptation、NeRF 编辑）尚无实证。

## 相关工作与启发

- **vs Navon et al. 2023 (DWS)、Lim et al. 2023 (GMN)、Kalogeropoulos et al. 2024 (ScaleGMN)**：这些工作各自给出"forward-pass 模拟"或对特定目标类的部分 expressivity；本文把它们装到统一框架里，先证彼此等价，再补全四象限的 universality 地图。
- **vs Maron et al. 2020 / Finkelshtein et al. 2025 / Gelberg et al. 2026（GDL 中的 GP 方法论）**：本文把"GP 外普适"的范式首次系统地搬进 weight-space，并给出权重空间下最自然的 GP 定义（hidden bias 两两不同）。
- **vs Pacini et al. 2025b（separation-to-approximation）**：本文是这套 Stone–Weierstrass 风格定理在 weight-space 的非平凡应用——separation 性质用 DWS 的 forward-pass 模拟来证，是本文证明骨架的关键拼图。
- **vs Bronstein 等 GDL 综述**：本文相当于在"对称域 = 图/点云/集合"之外，把 weight space 作为第四类对称结构化数据，搭出了对应的 expressivity 工具箱，是 GDL 在 meta-learning 时代的延展。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次为整个 weight-space 网络家族给出统一的表达力刻画，且把架构等价性、四象限普适性、不可能性、工程修复闭环写在一篇里。
- 实验充分度: ⭐⭐⭐ 理论占绝对主体，实证只有一个 benchmark；好在结果硬（34% SOTA gain）且直接呼应理论。
- 写作质量: ⭐⭐⭐⭐⭐ Figure 1 的"表达力地图"一图胜千言，定理-反例-定理的节奏清晰，证明 sketch 自洽可读。
- 价值: ⭐⭐⭐⭐⭐ 既给后续 weight-space 论文省去"选架构"这一节，也提示了 input-smaller-than-output 的新设计方向；OCE 本身就是即插即用的实用 trick。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Different Layers, Different Manifolds: Module-Wise Weight-Space Geometry in Transformer Optimization](different_layers_different_manifolds_module-wise_weight-space_geometry_in_transf.md)
- [\[NeurIPS 2025\] Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training](../../NeurIPS2025/llm_pretraining/power_lines_scaling_laws_for_weight_decay_and_batch_size_in_llm_pre-training.md)
- [\[ICLR 2026\] Steering Language Models with Weight Arithmetic](../../ICLR2026/llm_pretraining/steering_language_models_with_weight_arithmetic.md)
- [\[ICLR 2026\] Intrinsic Training Dynamics of Deep Neural Networks](../../ICLR2026/llm_pretraining/intrinsic_training_dynamics_of_deep_neural_networks.md)
- [\[ACL 2026\] Data Mixing Agent: Learning to Re-weight Domains for Continual Pre-training](../../ACL2026/llm_pretraining/data_mixing_agent_learning_to_re-weight_domains_for_continual_pre-training.md)

</div>

<!-- RELATED:END -->
