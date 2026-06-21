---
title: >-
  [论文解读] Exploring Mode Connectivity in Krylov Subspace for Domain Generalization
description: >-
  [ICLR 2026][优化/理论][mode connectivity] 本文跳出"找平坦极小值"的主流思路，转而利用损失曲面的**全局几何性质——模态连通性（mode connectivity）**，提出模拟台球运动的 Billiard Optimization Algorithm (BOA)，在低维 Krylov 子空间里沿低损失隧道从一个普通极小值"走"到泛化更强的极小值，在 DomainBed 上全面超越 SAM 等锐度感知方法。
tags:
  - "ICLR 2026"
  - "优化/理论"
  - "mode connectivity"
  - "Krylov subspace"
  - "domain generalization"
  - "loss landscape"
  - "sharpness-aware minimization"
  - "billiard dynamics"
---

# Exploring Mode Connectivity in Krylov Subspace for Domain Generalization

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=fpH2GYXJwD](https://openreview.net/forum?id=fpH2GYXJwD)  
**代码**: 待确认  
**领域**: 优化 / 域泛化（Optimization, Domain Generalization）  
**关键词**: mode connectivity, Krylov subspace, domain generalization, loss landscape, sharpness-aware minimization, billiard dynamics  

## 一句话总结
本文跳出"找平坦极小值"的主流思路，转而利用损失曲面的**全局几何性质——模态连通性（mode connectivity）**，提出模拟台球运动的 Billiard Optimization Algorithm (BOA)，在低维 Krylov 子空间里沿低损失隧道从一个普通极小值"走"到泛化更强的极小值，在 DomainBed 上全面超越 SAM 等锐度感知方法。

## 研究背景与动机
**领域现状**：理解深度网络损失曲面的几何结构已成为解释泛化的有力工具。主流共识是"平坦极小值泛化更好"，由此催生了 SAM、GSAM、SAGM、DISAM 等一大批锐度感知（sharpness-aware）方法，并被广泛迁移到域泛化（Domain Generalization, DG）任务中。

**现有痛点**：近期理论与实验（Dinh et al. 2017；Wen et al. 2023）指出，**平坦性并不普遍等价于更好的泛化**——既存在很尖锐却泛化良好的极小值，平坦性的泛化收益在标准神经网络（哪怕两层 ReLU）上也无法由线性模型结论直接推广。在分布偏移更剧烈的 DG 场景里，参数空间的平坦度甚至无法刻画模型对特征空间域偏移的脆弱性。换言之，只盯着**局部锐度**这一个指标，已经触到了天花板。

**核心矛盾**：损失曲面并非由孤立的盆地组成，不同极小值之间其实由连续的低损失路径相连（模态连通性）。这意味着一个 DG 准确率很差的"非理想模型"和一个准确率接近 100% 的"理想模型"，可能**就在同一条低损失隧道的两端**。可现有优化器（SGD、Langevin 动力学等）容易困在局部，根本无法主动沿这条隧道行走；而高维参数空间的"维度灾难"又让随机方向搜索几乎必然失效。

**本文目标**：把模态连通性首次用于 DG 算法设计，提供一个能在高维损失隧道里高效"导航"、从任意起点走到强泛化解的优化器。

**核心 idea**：**[几何洞察]** 把优化轨迹想象成台球在"球桌"（训练损失的低于阈值子水平集）内运动——直线滚向边界、撞到边界后镜面反射，从而在保持训练损失几乎不变的前提下遍历低损失区域；**[降维关键]** 进一步发现"测试梯度与训练梯度张成的 Krylov 子空间高度对齐"，据此把搜索约束到仅 5–20 维的子空间，既给出近最优初始方向、又破解维度灾难。

## 方法详解

### 整体框架
BOA 把 DG 优化建模成一场台球游戏：先用 ERM/SAM 预热得到起点 $\theta_0$，再把训练损失低于阈值 $\ell_{th}=\ell_{\text{train}}(\theta_0)+\Delta\ell$ 的参数区域定义为"球桌" $\mathcal{T}$；随后在球桌内交替执行两个动作——**线搜索**（球滚向桌边，定位损失等高线边界）与**反射**（撞边后按物理规则改变方向）——生成一条参数轨迹，最后用验证集从轨迹里挑出最优模型。关键的提速在于：所有方向搜索和反射都被约束在由训练梯度生成的低维 **Krylov 子空间** 内进行，从而避开高维参数空间的维度灾难。

```mermaid
flowchart LR
    A[ERM/SAM 预热<br/>起点 θ₀] --> B[定义球桌 T<br/>ℓ_train ≤ ℓ_th]
    B --> C[构造 Krylov 子空间<br/>K_K = span g, Hg, …, H^{K-1}g]
    C --> D[确定初始方向 p₀<br/>≈ -∇ℓ_test 的 Krylov 近似]
    D --> E[线搜索<br/>滚到损失等高线边界 θ_i]
    E --> F[反射<br/>p_i = I-2ñ_iñ_iᵀ · p_{i-1}]
    F --> E
    F --> G[沿轨迹用验证集<br/>挑最优模型]
```

### 关键设计

**1. 球桌定义：把"训练损失基本不变"变成硬约束**
台球桌被数学化为训练损失的子水平集 $\mathcal{T}:=\{\theta\in\mathbb{R}^d \mid \ell_{\text{train}}(\theta)\le \ell_{th}\}$，阈值取 $\ell_{th}=\ell_{\text{train}}(\theta_0)+\Delta\ell$，其中 $\Delta\ell>0$ 是一个很小的损失增量。这一构造保证预热模型 $\theta_0$ 严格落在球桌内部，更重要的是把"在训练集上表现几乎恒定"变成了一个有界的活动空间——算法只能在这个低损失盆地里折腾，因此它找到的所有解都共享差不多的训练损失，差异只体现在对未见域的泛化上。这正是用模态连通性做 DG 的前提：在等损失面上平移，本质上就是在不同极小值之间沿低损失隧道穿行。

**2. 线搜索：像球滚向桌边一样定位等高线边界**
给定当前参数 $\theta_{i-1}$ 和方向 $p_{i-1}$，线搜索要解非线性方程 $\ell(\theta_{i-1}+\alpha p_{i-1})=\ell_{th}$，即沿射线找到撞上"桌边"（损失等高线）的步长。BOA 用自适应括弧策略先粗定位区间：当 $\ell(x_k)<\ell_{th}$（还在桌内）时步长按 $h_{k+1}=(2k-1)h$ 指数放大，越过阈值即停；当 $\ell(x_k)>\ell_{th}$（冲出桌外）时步长按 $h_k=h_{k-1}/2$ 几何收缩。锁定区间 $[h_L,h_R]$ 后再用黄金分割搜索精修，$\psi=(1+\sqrt5)/2$，最终 $\theta_i=\theta_{i-1}+\alpha^\star p_{i-1}$ 把参数精确放到等高线上。整个过程只需若干次前向损失评估，不碰二阶量。

**3. 反射：动量守恒的镜面弹跳，O(d) 拿到曲率信息**
到达边界点 $\theta_i$ 后，BOA 用局部损失几何决定新方向。把单位法向取为最速下降方向 $n_i=-\nabla\ell(\theta_i)/\|\nabla\ell(\theta_i)\|_2$（相当于桌边法线），入射方向 $p_{i-1}$ 经镜面反射更新为

$$p_i=p_{i-1}-2(p_{i-1}^\top n_i)n_i=(I-2n_in_i^\top)p_{i-1}.$$

这一变换减去 $p_{i-1}$ 在法向上投影的两倍，**严格保持动量大小** $\|p_i\|_2=\|p_{i-1}\|_2$ 并满足反射定律（入射角=反射角）。反射算子 $R[n_i]=I-2n_in_i^\top$ 是一个 $\det=-1$ 的非正常旋转，由于 $n_i\propto\nabla\ell(\theta_i)$，它在**不重算 Hessian** 的情况下隐式注入了曲率信息，每步只需 $O(d)$ 复杂度即可沿等高线高效探索。

**4. Krylov 子空间对齐：维度灾难的"免费午餐"**
高维下维度灾难体现在两点——随机向量几乎正交于 oracle 最优方向（随机初始方向几乎无效），以及轨迹稀疏（需极多步才够泛化。解法是把一切约束进由训练导数生成的 Krylov 子空间 $\mathcal{K}_K(H_{\text{train}},g_{\text{train}})=\text{span}\{g_0,Hg_0,\dots,H^{K-1}g_0\}$，其中 $g_0=\nabla\ell_{\text{train}}(\theta_0)$、$H=\nabla^2\ell_{\text{train}}(\theta_0)$。本文的核心实证发现是：**测试梯度 $\nabla\ell_{\text{test}}(\theta_0)$ 与该 Krylov 子空间高度对齐**，于是初始方向可直接近似为 $p_0=-\sum_{k=0}^{K-1}\beta_k H^k g_0\approx-\nabla\ell_{\text{test}}(\theta_0)$，而且实验显示简单取 $\beta_k=1$ 就能让 $p_0$ 与 oracle 负测试梯度的夹角足够小。Hessian-向量积用有限差分 $Hg_0\approx[\nabla\ell(\theta_0+\epsilon g_0)-\nabla\ell(\theta_0)]/\epsilon$ 近似，保持 $O(d)$ 且无需显式二阶导。反射也被投影回子空间：$p_i=(I-2\tilde n_i\tilde n_i^\top)p_{i-1}$，$\tilde n_i=\text{proj}_{\mathcal{K}_K}n_i$。这样既无需访问测试数据就拿到近最优初始方向，又把搜索空间压到 5–20 维，真正架起了训练域与未见测试域之间的桥梁。

## 实验关键数据

### 主实验表格
ViT-B/16（CLIP 预训练 + VPT）在 DomainBed 五数据集上的 out-of-domain 准确率（%），采用 test-domain validation set 公平复现所有基线：

| 方法 | VLCS | PACS | OfficeHome | TerraInc | DomainNet | Avg. |
|------|------|------|-----------|----------|-----------|------|
| ERM (VPT) | 81.9 | 95.9 | 84.1 | 56.1 | 59.5 | 75.5 |
| CORAL | 82.6 | 96.4 | 83.8 | 57.5 | 59.8 | 76.0 |
| SAM | 82.9 | 96.6 | 85.4 | 56.2 | 59.8 | 76.2 |
| GSAM | 82.9 | 96.6 | 85.6 | 55.4 | 59.8 | 76.1 |
| SAGM | 82.8 | 96.8 | 85.2 | 58.0 | 59.1 | 76.4 |
| DISAM | 82.7 | 97.1 | 85.4 | 57.3 | 59.8 | 76.5 |
| **BOA (Ours)** | **86.5** | **97.4** | **86.0** | **60.3** | **60.2** | **78.1** |

BOA 平均 78.1%，较最强基线 DISAM (76.5%) 提升 1.6%，在 VLCS 上相比 SAM 大涨 **3.6%**。

### 消融实验表格
用 oracle 负测试梯度 $-\nabla\ell_{\text{test}}(\theta_0)$ 作初始方向验证台球动力学可行性（Table 4 摘要）：

| 配置 | 五数据集平均提升 |
|------|------|
| ERM + BOA (oracle 方向) | +4.9% |
| SAM + BOA (oracle 方向) | +4.8% |

说明只要给对初始方向，沿低损失隧道行走确实能稳定捞到更强泛化模型；而 Krylov 近似（$\beta_k=1$）正是在无测试数据时逼近这个 oracle 方向。

### 关键发现
- **Krylov 对齐普遍成立**：在 Caltech101/LabelMe/SUN09/VOC2007 上，测试梯度在 Krylov 子空间的投影余弦相似度随维度迅速逼近 1，而随机子空间几乎为 0——对齐性跨数据集、跨架构稳健。
- **低维即够**：仅需 5–20 维 Krylov 子空间即可包含强泛化解，验证了降维的正确性。
- **验证集选择陷阱**：作者发现验证准确率相近的模型，DG 性能沿轨迹可相差 10%+，故采用 test-domain validation set 并对所有基线统一复现以保证公平。
- **步长敏感性**：步长 $h$ 存在明确最优区间（如 Caltech101 在 $h\approx0.5$ 达 99.2%），过大反而掉点。

## 亮点与洞察
- **换了一个几何视角**：从局部平坦性转向全局模态连通性，是对"flat minima 万能论"的有力反驳，并首次把连通性落地为可执行的 DG 优化器。
- **台球隐喻好用且严谨**：线搜索+反射两个直观动作，对应到有界子水平集上的等损失面遍历，物理直觉与数学约束严丝合缝。
- **"免费午餐"很漂亮**：测试梯度与训练 Krylov 子空间对齐这一实证规律，让算法在不碰测试数据的前提下拿到近最优方向，是全文最有价值的发现。
- **计算友好**：反射靠有限差分 HVP，全程 $O(d)$，无需显式 Hessian，工程上可落地。

## 局限与展望
- **依赖预热 + 验证集选模**：BOA 在 SAM 预热之后运行，且需沿轨迹用验证集挑模型；论文也坦承标准验证集会误导选择，被迫改用 test-domain validation set，这在严格 DG 协议下的公平性值得商榷。
- **超参较多**：步长 $h$、反射次数、Krylov 维度 $K$、损失增量 $\Delta\ell$ 等需网格搜索，对新任务的调参成本未充分讨论。
- **理论缺口**：Krylov 对齐目前是强实证规律，为何测试梯度会落进训练 Krylov 子空间、对齐何时失效，缺乏理论刻画。
- **范围局限**：实验集中在 CLIP-ViT + VPT 的视觉 DG，是否能推广到 NLP、检测/分割等更复杂任务或全量微调场景尚待验证。

## 相关工作与启发
- **局部结构 / 锐度感知**：SAM、GSAM、SAGM、DISAM、SWAD 等追求平坦极小值；本文站在它们的对立面，用连通性而非平坦性来选解。
- **模态连通性**：Garipov et al. (2018)、Draxler et al. (2018) 发现极小值间的低损失曲线；此前多用于模型合并、机器遗忘、持续学习，本文是首次用于 OOD/DG 优化。
- **Krylov 子空间方法**：经典数值线代工具被巧妙借来做高维优化降维，提示我们"训练侧二阶信息张成的子空间能近似测试侧梯度"这一桥梁或可推广到更广的迁移/适应问题。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把模态连通性首次落地为 DG 优化器，并发现测试梯度—训练 Krylov 子空间对齐这一非平凡规律，视角和机制都很新。
- 实验充分度: ⭐⭐⭐⭐ DomainBed 五数据集 + 三种 ViT backbone + oracle/步长/维度多组消融，较扎实；但局限于 CLIP-ViT+VPT、未覆盖更广任务。
- 写作质量: ⭐⭐⭐⭐ 台球隐喻清晰、公式与图示配合到位，叙述有画面感。
- 价值: ⭐⭐⭐⭐ 提供了超越"找平坦极小"的新工具，Krylov 对齐的"免费午餐"对迁移学习社区有启发意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Entropic Confinement and Mode Connectivity in Overparameterized Neural Networks](entropic_confinement_and_mode_connectivity_in_overparameterized_neural_networks.md)
- [\[ICML 2025\] Understanding Mode Connectivity via Parameter Space Symmetry](../../ICML2025/optimization/understanding_mode_connectivity_via_parameter_space_symmetry.md)
- [\[ICLR 2026\] Exploring Diverse Generation Paths via Inference-time Stiefel Activation Steering](exploring_diverse_generation_paths_via_inference-time_stiefel_activation_steerin.md)
- [\[ICLR 2026\] Trion: FFT-based Dynamic Subspace Selection for Low-Rank Adaptive Optimization of LLMs](trion_fft-based_dynamic_subspace_selection_for_low-rank_adaptive_optimization_of.md)
- [\[ICLR 2026\] Bayesian Evidence-Driven Prototype Evolution for Federated Domain Adaptation](bayesian_evidence-driven_prototype_evolution_for_federated_domain_adaptation.md)

</div>

<!-- RELATED:END -->
