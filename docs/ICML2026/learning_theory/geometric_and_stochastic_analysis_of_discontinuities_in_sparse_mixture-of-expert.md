---
title: >-
  [论文解读] Geometric and Stochastic Analysis of Discontinuities in Sparse Mixture-of-Experts
description: >-
  [ICML 2026][学习理论][Sparse MoE] 首次对稀疏专家混合（SMoE）中 Top-$k$ 路由带来的"输入—输出映射不连续性"做严格的几何 + 随机分析——按"打平专家个数"给不连续面分阶，证明低阶（order-1）不连续面占据几乎全部"近不连续"体积、高阶面体积可忽略，并用扩散过程证明随机扰动几乎必然首次撞上 order-1 面；据此提出一个即插即用的 $\ell_\infty$ 局部平滑机制 SmoothSMoE，在几乎不增算力的前提下让 SMoE 映射连续并提升语言/视觉任务性能。
tags:
  - "ICML 2026"
  - "学习理论"
  - "稀疏专家混合"
  - "路由连续性"
  - "Sparse MoE"
  - "Top-k 路由"
  - "不连续性分析"
  - "测度论"
  - "扩散过程"
  - "局部平滑"
---

# Geometric and Stochastic Analysis of Discontinuities in Sparse Mixture-of-Experts

**会议**: ICML 2026  
**arXiv**: [2606.19036](https://arxiv.org/abs/2606.19036)  
**代码**: https://github.com/thotranhuu99/Smooth_SMoE  
**领域**: 学习理论 / 稀疏专家混合 / 路由连续性  
**关键词**: Sparse MoE, Top-k 路由, 不连续性分析, 测度论, 扩散过程, 局部平滑

## 一句话总结
首次对稀疏专家混合（SMoE）中 Top-$k$ 路由带来的"输入—输出映射不连续性"做严格的几何 + 随机分析——按"打平专家个数"给不连续面分阶，证明低阶（order-1）不连续面占据几乎全部"近不连续"体积、高阶面体积可忽略，并用扩散过程证明随机扰动几乎必然首次撞上 order-1 面；据此提出一个即插即用的 $\ell_\infty$ 局部平滑机制 SmoothSMoE，在几乎不增算力的前提下让 SMoE 映射连续并提升语言/视觉任务性能。

## 研究背景与动机
**领域现状**：SMoE 把 Transformer 的 FFN 层换成一堆稀疏激活的专家模块，靠 Top-$k$ 门控只激活 $k$ 个专家，从而在控制算力的同时把模型规模做大，已广泛用于大语言模型（DeepSeek、Switch Transformer 等）和视觉模型。

**现有痛点**：Top-$k$ 门控的稀疏性是靠"硬选 top-$k$"实现的，这天然让 SMoE 的输入—输出映射**不连续**——两个几乎一模一样的输入，只要落在路由切换边界附近，就可能被分给完全不同的专家集合，输出随之剧烈跳变。这对鲁棒性、对抗稳定性都是隐患。已有工作要么只是"承认存在这种不连续"却没系统分析其结构，要么用免硬切换的可微路由来回避（SMEAR 合并专家、Soft MoE 跨 token 混合），但这些做法破坏了自回归生成所需的因果结构；ReMoE 用 ReLU 门控替代 Top-$k$ 又需要从头重训门控、初始化代价几乎等于训一个稠密模型。

**核心矛盾**：稀疏性（=算力可控）与连续性（=稳定性）在 Top-$k$ 门控下是对立的——要保留 Top-$k$ 的稀疏与因果性，就得直面它的不连续；而要消除不连续，现有方案往往要付出"破坏因果"或"重训门控"的高昂代价。更根本的是，此前**没人定量回答**过：这些不连续面到底长什么样、有多大、随机扰动会不会撞上它们。

**本文目标**：从两个视角严格刻画 SMoE 的不连续性——几何上"不同打平模式有多频繁、其边界附近占多大空间"，随机上"随机扰动的轨迹会不会撞边界、撞的是哪一阶"，并把理论洞察转化为一个低成本、保因果的平滑机制。

**切入角度**：把门控打分写成仿射函数后，Top-$k$ 把输入空间切成一堆"激活专家集固定"的开胞腔，不连续面就是这些胞腔的边界（即出现并列打分之处）。按"同时打平的专家个数"给边界分阶（order-$n$），就能用测度论的切片论证去算每一阶边界的体积，再把随机扰动建模成扩散过程去算撞击/驻留时间。

**核心 idea**：用"几何测度（体积估计）+ 随机过程（撞击/驻留时间）"两套工具证明"低阶不连续主导、输入更可能落在低阶面附近"，从而只需在 $\ell_\infty$ 意义下对那条窄边界带做局部软化即可恢复连续——花小钱办大事。

## 方法详解

### 整体框架
全文是"先分析、后落地"的结构。**分析侧**：把 Top-$k$ 门控的不连续集 $\Gamma$ 按打平专家数分成 order-1、order-2……，用切片论证给每一阶不连续面的 $\epsilon$-加厚（thickening）体积一个渐近上界，证明阶数越高体积越可忽略；再把输入受到的随机扰动建模成布朗扩散，证明轨迹几乎必然在有限时间撞上不连续面、且首次撞击几乎必然发生在 order-1 面，并给出在各阶邻域内的驻留时间界。**落地侧**：既然输入最可能贴着低阶面，就只在 $\ell_\infty$ 意义下离边界很近的那条窄带里，把刚好落选的非 top-$k$ logit 软性抬升进来（log-smoothstep），从而让映射连续；再配一个边界损失自适应地把窄带宽度 $\epsilon$ 调到"平均只多激活半个专家"的预算上。这是一篇以理论为主体、平滑机制是理论推论的工作，因此不强塞流程图，下面用定义—定理—公式讲清。

### 关键设计

**1. 按"打平专家个数"给不连续面分阶**

不连续性来自门控打平。门控打分是仿射的 $z_i(x)=\langle W_g^{(i)},x\rangle+b_g^{(i)}$，对每个 $k$-子集 $\mathbb{S}$ 定义开胞腔 $\mathcal{C}_{\mathbb{S}}=\{x: z_i(x)>z_j(x),\ \forall i\in\mathbb{S}, j\notin\mathbb{S}\}$，所有这种胞腔把输入空间剖分成"激活集恒定"的区域，SMoE 映射在每个区域内是光滑的；它们的补集 $\Gamma$ 就是不连续集。论文进一步细分：一个**order-$n$ 不连续点**是指有 $n+1$ 个打分恰好在第 $k$ 与第 $k+1$ 大之间打平：

$$z_{i_1}(x)=\cdots=z_{i_{n+1}}(x)=z_{[k]}(x)=z_{[k+1]}(x)$$

把这 $n$ 个独立等式 $(W_g^{(i_s)}-W_g^{(i_1)})^\top x=b_g^{(i_1)}-b_g^{(i_s)}$ 摞成线性系统 $A_J x=d_J$，order-$n$ 面就落在一个**余维 $n$**（即 $D-n$ 维）的仿射平面 $S_J^{(n)}$ 上。最简单的 order-1 就是第 $k$ 与第 $k+1$ 名打平、一个微小扰动就能交换 top-$k$ 成员资格。这套"按阶分类"是后面所有体积/时间估计的骨架。

**2. 加厚体积的渐近估计：低阶主导、高阶可忽略**

不连续集 $\Gamma$ 本身是测度零（不占体积），但"近不连续"的邻域可能很大（类比有理数测度零、其 $\epsilon$-邻域却是整条实线），所以真正该量化的是 $\epsilon$-加厚 $T_\epsilon(\Gamma^{(n)})=\{x:\mathrm{dist}(x,\Gamma^{(n)})<\epsilon\}$ 的体积。核心定理证明：在半径 $R$ 的球内，

$$\frac{\lambda^D(T_\epsilon(\Gamma^{(n)})\cap B^D(0,R))}{\lambda^D(B^D(0,R))}\ \lesssim\ \Big(\frac{\epsilon}{R}\Big)^{n}$$

直觉是 order-$n$ 面落在余维 $n$ 的平面上，法向贡献 $\epsilon^n$ 体积、切向贡献 $R^{D-n}$，归一化后随 $(\epsilon/R)^n$ 衰减。论文还给出**跨阶之比** $U_n(R)/U_m(R)\sim(\epsilon/R)^{n-m}$，意味着随 $R$ 增大高阶面相对低阶面体积可忽略——**绝大多数"近不连续"的输入都贴着 order-1 面**。此外，由于直接做欧氏邻近测试要对所有子空间求距离、太贵，论文引入更易算的 $\ell_\infty$ 加厚 $\mathrm{dist}_\infty(x,\Gamma)=\inf_y\|z(x)-z(y)\|_\infty$，它只需检查"是否有某个非 top-$k$ logit 落在 $z_{[k]}(x)$ 的 $\epsilon$ 之内"，把邻近测试直接搬到 logit 空间，且保持同样的跨阶衰减率（只是前因子多了个 $\kappa_{J,r}$ 反映 $\ell_\infty$ 管的轴对齐性）。

**3. 随机扰动的撞击与驻留时间：首撞几乎必为 order-1**

光有几何还不够，得知道随机扰动下输入会不会真的撞到边界。论文把扰动（如对抗者做的小随机更新）建模成 Itô 扩散 $dx_t=\sigma\,dB_t$，$x_0$ 从某个开胞腔 $\mathcal{C}_{\mathbb{S}}$ 出发。两个核心结论：(i) **撞击时间**——轨迹几乎必然在有限时间内撞上不连续边界，且**首次撞击几乎必然发生在 order-1 面**，并给出显式的有限时间概率界；(ii) **驻留时间**——扩散在 order-$n$ 面 $\epsilon$-邻域内停留的期望时间上界含主因子 $\epsilon^n$，随阶数升高指数衰减（$0<\epsilon<1$ 时），即停在高阶邻域的时间远少于低阶。两者合起来给出一个清晰的物理图景：**随机轨迹大部分时间在低阶面附近游走**，这正是平滑机制只需照顾低阶窄带的理论依据。

**4. SmoothSMoE：$\ell_\infty$ 局部平滑 + 自适应边界损失**

既然输入最常贴着 order-1 面、且 $\ell_\infty$ 邻近可在 logit 空间高效检测，平滑就只在窄带里动手：对满足 $0<z_{[k]}(x)-z_i(x)<\epsilon$ 的非 top-$k$ logit（刚好落选的那几个）做软性抬升，低于这条带的直接丢弃，top-$k$ logit 原封不动。用 log-smoothstep $h(u)$（$u\le0$ 时 $-\infty$、$u\ge1$ 时 $0$、中间平滑过渡）定义平滑系数 $m_i(x)=h((z_i(x)-z_{[k]}(x)+\epsilon)/\epsilon)$，再令平滑后 logit 为 $\hat z_i(x)=z_i(x)+m_i(x)$。一个关键性质：若 $x$ 落在 order-$n$ 面的 $\ell_\infty$ 加厚里，最多只会多激活 $n$ 个专家；而高阶加厚体积衰减极快（设计 2），所以小 $\epsilon$ 下平均多激活的专家数极少，算力开销可控、且**保留了自回归所需的因果结构**（不像 Soft MoE 跨 token 混合）。由于 $\epsilon$ 直接作用在 logit 空间、难手调，论文再加一个边界损失 $\mathcal{L}_{\text{boundary}}=\alpha\,\epsilon\,(\mathcal{K}-k^*)$ 自适应调 $\epsilon$：$\mathcal{K}$ 是当前平均激活专家数、$k^*$ 是目标预算；$\mathcal{K}>k^*$ 时损失把 $\epsilon$ 压小、反之拉大，实践取 $k^*=k+0.5$（平均只允许多半个专家）。

## 实验关键数据

### 主实验
在语言（WikiText-103 / EnWiki-8）、自然语言理解（GLUE）、视觉（DomainBed）多任务上，把 SmoothSMoE 接到现有 SMoE / GMoE 上对比基线（含连续路由方法 ReMoE、Soft MoE、Expert Choice）。

| 任务 / 指标 | SMoE 基线 | ReMoE | SmoothSMoE（本文）|
|------|------|------|------|
| WikiText-103 Test PPL ↓ | 35.52 | 35.35 | **34.35** |
| Attacked WikiText-103 Test PPL ↓ | 44.18 | 44.00 | **42.85** |
| EnWiki-8 BPC ↓ | 1.153 | — | **1.122** |
| GLUE 平均（K=16,k=2）↑ | 81.17 | 81.18 | **81.65** |
| GLUE 平均（K=16,k=4）↑ | 81.14 | 81.16 | **81.73** |
| DomainBed 平均 acc ↑ | GMoE 基线 | — | **+0.56%（vs GMoE）** |

SmoothSMoE 在干净与受攻击的语言建模、语言理解、域泛化视觉任务上都稳定优于 Top-$k$ SMoE 与连续路由基线 ReMoE：WikiText-103 测试 PPL 降 1.17、受攻击版降 1.33，GLUE 平均提升 0.47%（k=2）/0.57%（k=4），DomainBed 平均涨 0.56%（TerraInc 涨 2.1%）。

### 连续性验证 / 分析

| 实验 | 观察 | 说明 |
|------|------|------|
| CIFAR-10 上 4 层 SMoE，Layer 3 | SMoE 在边界处输出"跳变" | 沿法向走，微小扰动→大输出变化 |
| 同权重换成 SmoothSMoE | 跳变消失、映射连续 | 验证平滑确实恢复连续 |
| 最大输出差 vs 扰动 $\|\Delta x\|$ | SMoE 不随 $\|\Delta x\|\to0$ 消失，SmoothSMoE 消失 | 直接证明连续性 |
| 边界损失对 $\epsilon$ 的调节（附录 B.4）| 自适应把平均激活专家数控在 $k^*$ 附近 | 验证预算控制有效 |

### 关键发现
- **理论预测被实验印证**：跳变只发生在边界、且平滑后消失，最大输出差随扰动趋零而消失，正对应"order-1 面主导 + 局部软化恢复连续"的理论。
- **大数据集增益更稳**：DomainBed 上越大的数据集提升越一致，作者解释为大数据下"贴近打平边界"的输入更多，平滑通过在 tie 附近多激活专家而更有效。
- **几乎零额外算力**：因高阶加厚体积衰减极快，小 $\epsilon$ 下平均只多激活半个专家，开销可忽略，却同时拿到连续性 + 鲁棒性 + 性能三重收益。

## 亮点与洞察
- **把"不连续"做成可量化对象**：用"按阶分类 + 切片测度"把抽象的路由不连续变成有渐近体积公式的几何对象，这套刻画此前是空白，很可能成为后续 MoE 理论的基础工具。
- **几何 + 随机两条腿走路**：体积估计回答"哪里不连续多"，扩散撞击/驻留时间回答"扰动会不会撞、撞哪一阶"，两者交叉验证出"低阶主导"这一可操作结论。
- **理论直接生出方法**：$\ell_\infty$ 加厚之所以被选为平滑判据，正是因为它能在 logit 空间高效检测又保持同样的衰减率——这是"理论指导工程"的干净范例，且方法即插即用、保因果，可直接套到现有 SMoE/GMoE 上。

## 局限与展望
- 分析建立在门控打分**仿射**这一假设上（线性 scoring + 偏置），对非线性/带归一化的门控是否成立未充分讨论。
- 随机扰动建模为各向同性常系数布朗扩散 $dx_t=\sigma\,dB_t$，与真实对抗扰动或训练动态的相关性是简化的；"对抗者"动机更多是叙事而非实测。
- 实验规模偏中小（Switch Transformer 16 专家、BERT-large 单层换 MoE、ViT-S/16），在百亿级真·大 MoE 上的连续性收益与算力开销尚待验证。
- 性能提升幅度温和（PPL 降 ~1、GLUE/DomainBed 提升 <1%），更像"稳定性副产物"而非大幅 SOTA，价值更多在理论刻画本身。

## 相关工作与启发
- **vs SMEAR / Soft MoE（可微路由）**：它们靠合并专家或跨 token 混合消除硬切换，但破坏自回归因果结构、限制生成任务；本文只在窄带做局部平滑、保留 top-$k$ 因果性。
- **vs ReMoE（ReLU 门控）**：ReMoE 用 ReLU 替代 Top-$k$，需从头重训门控且有昂贵初始化；SmoothSMoE 即插即用、不改门控本体、几乎零重训成本，且实验上全面优于 ReMoE。
- **vs 此前"承认不连续"的工作（Chen et al. 2022 等）**：它们只指出不连续存在；本文首次给出其结构（分阶）、体积（渐近测度）、与随机行为（撞击/驻留时间）的系统理论。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次对 SMoE 不连续性做严格几何+随机刻画，分阶测度与撞击时间界都是新结果
- 实验充分度: ⭐⭐⭐⭐ 跨语言/理解/视觉三类任务验证，但规模偏中小、增益温和
- 写作质量: ⭐⭐⭐⭐ 定义—定理—推论层层递进，理论自洽；公式密度高、OCR 略有噪声
- 价值: ⭐⭐⭐⭐ 既补上理论空白，又给出即插即用、保因果的低成本平滑机制

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] A Generalized Geometric Theoretical Framework of Centroid Discriminant Analysis for Linear Classification of Multi-dimensional Data](../../ICLR2026/learning_theory/a_generalized_geometric_theoretical_framework_of_centroid_discriminant_analysis_.md)
- [\[NeurIPS 2025\] Finite-Time Analysis of Stochastic Nonconvex Nonsmooth Optimization on the Riemannian Manifolds](../../NeurIPS2025/learning_theory/finite-time_analysis_of_stochastic_nonconvex_nonsmooth_optimization_on_the_riema.md)
- [\[ICML 2026\] Robustness of Mixtures of Experts to Feature Noise](robustness_of_mixtures_of_experts_to_feature_noise.md)
- [\[ICLR 2026\] Lipschitz Bandits with Stochastic Delayed Feedback](../../ICLR2026/learning_theory/lipschitz_bandits_with_stochastic_delayed_feedback.md)
- [\[ICLR 2026\] Better Bounds for the Distributed Experts Problem](../../ICLR2026/learning_theory/better_bounds_for_the_distributed_experts_problem.md)

</div>

<!-- RELATED:END -->
