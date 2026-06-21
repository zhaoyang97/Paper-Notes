---
title: >-
  [论文解读] Efficient Sliced Wasserstein Distance Computation via Adaptive Bayesian Optimization
description: >-
  [ICLR 2026][图像生成][Sliced Wasserstein] 这篇论文把 Sliced Wasserstein 距离中的“投影方向选择”从固定的低差异采样改成可学习的贝叶斯优化过程，提出 BOSW/RBOSW/ABOSW/ARBOSW 四种可插拔策略，在不改下游损失和梯度公式的前提下，在多个 SW-in-the-loop 任务上达到或逼近 SOTA。
tags:
  - "ICLR 2026"
  - "图像生成"
  - "Sliced Wasserstein"
  - "Bayesian Optimization"
  - "QSW"
  - "方向采样"
  - "最优传输"
---

# Efficient Sliced Wasserstein Distance Computation via Adaptive Bayesian Optimization

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=6IZAOTfXUT](https://openreview.net/forum?id=6IZAOTfXUT)  
**代码**: 论文称已提供可复现代码与固定随机种子（具体仓库链接见补充材料）  
**领域**: image generation / optimal transport  
**关键词**: Sliced Wasserstein、Bayesian Optimization、QSW、方向采样、最优传输  

## 一句话总结
这篇论文把 Sliced Wasserstein 距离中的“投影方向选择”从固定的低差异采样改成可学习的贝叶斯优化过程，提出 BOSW/RBOSW/ABOSW/ARBOSW 四种可插拔策略，在不改下游损失和梯度公式的前提下，在多个 SW-in-the-loop 任务上达到或逼近 SOTA。

## 研究背景与动机
**领域现状**：Sliced Wasserstein（SW）通过把高维最优传输投影到一维来降复杂度。对离散分布而言，单次 1D Wasserstein 主要开销是排序，整体代价约为 $O(n\log n)$，因此比直接高维 WD 更适合生成建模、配准、梯度流等任务。当前主流做法是给定 $L$ 个方向后做平均。

**现有痛点**：方向集 $\Theta_L$ 的质量直接决定 SW 估计误差。纯 MC 的误差随样本数只按 $O(L^{-1/2})$ 收敛；QSW/RQSW 虽然用低差异点列改善了“均匀覆盖”，但本质仍是数据无关的几何采样，无法利用当前任务里已观察到的切片代价信息。

**核心矛盾**：SW 的积分域是整个球面，但“有信息量”的方向往往集中在局部结构区域。固定预算下，如果仍追求全局均匀覆盖，就会把不少切片浪费在低贡献方向；而优化回路中的分布又会随迭代变化，最优方向集合也应随时间更新。

**本文目标**：作者把问题拆成两个子目标：
1. 在固定切片预算 $L$ 下，找到更“任务相关”的方向集以加速收敛。
2. 在 SW 反复被调用的优化过程中，允许方向集进行轻量重学习，同时保持对现有 QSW 管线的兼容。

**切入角度**：作者将 $f(\theta;\mu,\nu)=W_p^p(\theta_\sharp\mu,\theta_\sharp\nu)$ 视为定义在单位球面上的黑盒函数，使用 GP + UCB 的 Bayesian Optimization（BO）按“观测到的切片代价”自适应选方向，而不是只看几何均匀性。

**核心 idea**：用 BO 学习“哪里的投影方向更值得评估”，并把这套学习器做成 drop-in 的方向选择模块，可与 QSW 种子方向组合（ABOSW/ARBOSW），达到“低差异覆盖 + 任务自适应”的折中。

## 方法详解

### 整体框架
论文先回到 SW 定义：
$$
SW_p^p(\mu,\nu)=\mathbb{E}_{\theta\sim U(S^{d-1})}\left[W_p^p(\theta_\sharp\mu,\theta_\sharp\nu)\right],
$$
实践中以 $L$ 个方向近似：
$$
\widehat{SW}_p^p(\mu,\nu;\Theta_L)=\frac{1}{L}\sum_{\ell=1}^{L}W_p^p\big((\theta_\ell)_\sharp\mu,(\theta_\ell)_\sharp\nu\big).
$$
因此关键不是改 SW 损失本身，而是改“如何选 $\Theta_L$”。作者把每个方向对应的一维代价记作 $f(\theta)$，然后在球面上做 BO，输出方向集合再交给原 SW 估计器与下游优化器。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
	A[输入分布对 μ, ν] --> B[评估已有方向的切片代价 f(θ)]
	B --> C[球面 GP 代理建模]
	C --> D[UCB 采集函数选新方向]
	D --> E[去重与批量更新方向集]
	E --> F[计算 SW 估计并驱动下游优化]
	F --> G{是否周期刷新}
	G -->|是| B
	G -->|否| H[保持方向集继续迭代]
```

### 关键设计
**1. 球面核 GP：在方向空间上建模“切片代价地形”**

BO 的核心是对 $f(\theta)$ 做后验建模。作者使用球面角距离上的 RBF 核：
$$
k(\theta,\theta')=\exp\left(-\frac{1}{2}\left(\frac{d_S(\theta,\theta')}{\ell}\right)^2\right),\quad
d_S(\theta,\theta')=\arccos\langle \theta,\theta'\rangle.
$$
这里长度尺度 $\ell$ 用当前样本对之间的球面距离中位数启发式设定。这样做的意义是：相近方向的切片代价共享统计相关性，BO 能据此推断“尚未采样但潜在高价值”的方向区域。

**2. UCB 批量提案 + 近重复抑制：在开销可控前提下做自适应探索**

每轮 BO 先在球面均匀采样候选池（默认 $n_c=4096$），再用 $\alpha_{\text{UCB}}(\theta)=\mu(\theta)+\beta\sigma(\theta)$ 打分（文中默认 $\beta=0.7$），选择小批量方向（默认 $b=5$）。
为避免方向扎堆，作者加入余弦相似度阈值（>0.98 则剔除）抑制近重复。该设计把每轮附加开销压在 $O(n_c n_t)$ 量级，并且新增真值评估点只有 $b$ 个，实测 overhead 维持在“可接受的轻量级”区间。

**3. 四种 BO 变体：一次性学习、周期刷新、QSW 混合与重启混合**

论文不是只给一个算法，而是给四个可替换选择器：
- BOSW：一次性 BO 选满 $L$ 个方向，后续固定。
- RBOSW：每隔 $R$ 步（实验常用 $R=25$）重新运行 BOSW。
- ABOSW：先用强 QSW 集合作种子，再做少量 BO 微调（文中常用 $r=2,b=5$，即最多替换 10% 方向）。
- ARBOSW：周期性重启 ABOSW，每次都从 QSW 重新播种再短程 BO。

这组设计对应了不同任务形态：若分布变化快，刷新/重启更有利；若训练分布稳定，固定或轻微微调方向通常更稳。

**4. 与 QSW/RQSW 的组合策略：不改目标函数，只换方向生成器**

论文强调 BO 模块是“插拔式”的：不需要改 SW 目标、反向传播公式或优化器超参，只在“方向采样器”这一层替换。ABOSW/ARBOSW 就是 QSW（低差异几何先验）与 BO（任务自适应）的组合。

同时作者明确：BO 导向的方向集通常引入偏差，因此它不追求“有限样本下的无偏 SW 估计”；若任务必须依赖无偏随机梯度，可继续用 RQSW 机制，再叠加 BO 作为可选 refinement。

### 一个完整示例
以文中的点云梯度流任务为例，假设预算 $L=100$：
1. 先初始化方向集（例如 CQSW 种子）并计算当前每个方向的一维 OT 代价。
2. ABOSW 用这些观测训练球面 GP，在 4096 个候选方向中找 UCB 值最高的 5 个新方向。
3. 将方向集中贡献最低的 5 个方向替换掉；重复 2 轮后，最多替换 10 个方向。
4. 用更新后的方向集继续做梯度流迭代。
5. 若采用 ARBOSW，则每 25 步重新执行“QSW 播种 + 2 轮 BO 微调”。

这个流程的关键不是“让每个方向都最优”，而是把有限预算尽量放到对当前优化最有区分度的切片区域。

### 损失函数 / 训练策略
论文并未提出新的下游损失，核心是替换 SW 估计时的方向选择器。实验设置尽量与 QSW 论文保持一一对应：同数据、同优化器、同学习率、同停止准则，仅替换方向构造方法，保证横向比较公平。

## 实验关键数据

### 主实验
论文在多类任务评估：合成方向搜索、点云插值梯度流、图像风格迁移、点云自编码器。最核心结论是：
- 纯“近似 SW 积分”场景里，QSW 依旧强（因为它擅长均匀覆盖）。
- SW-in-the-loop 的动态优化任务里，BO 混合方法（特别 ARBOSW、ABOSW）更有优势或至少持平。

下表摘自点云梯度流（$L=100$，指标为 $W_2\downarrow$，文中表值按 $10^2$ 缩放）：

| 方法 | Step 100 | Step 200 | Step 300 | Step 400 | Step 500 | 时间(s) |
|---|---:|---:|---:|---:|---:|---:|
| MCSW | 5.749 | 0.187 | 0.031 | 0.013 | 0.006 | 4.06 |
| CQSW | 5.603 | 0.183 | 0.078 | 0.073 | 0.071 | 3.96 |
| RCQSW | 5.708 | 0.181 | 0.027 | 0.011 | 0.005 | 3.95 |
| RBOSW (ours) | **2.213** | **0.083** | 0.047 | 0.033 | 0.025 | 44.58 |
| ARBOSW (ours) | 5.717 | 0.186 | **0.025** | 0.012 | **0.003** | 6.91 |

可见 RBOSW 在早期迭代显著领先，但代价是高刷新开销；ARBOSW 在后期达到最优或并列最优，同时保持中等额外耗时。

### 消融实验
论文还在点云自编码器上比较了多种估计器（$L=100$，400 epoch，表值同样按 $10^2$ 缩放）：

| 方法 | SW2 @100 | W2 @100 | SW2 @200 | W2 @200 | SW2 @400 | W2 @400 |
|---|---:|---:|---:|---:|---:|---:|
| MCSW | 2.25 | 10.58 | 2.11 | 9.92 | 1.94 | 9.21 |
| CQSW | 2.22 | 10.54 | 2.05 | 9.81 | 1.84 | 9.06 |
| BOSW | 2.20 | 10.34 | 2.02 | 9.78 | 1.80 | 9.01 |
| ABOSW | **2.18** | **10.27** | **2.01** | **9.76** | 1.81 | **9.01** |
| ARBOSW | 2.21 | 10.44 | 2.04 | 9.80 | 1.85 | 9.07 |

这里 ABOSW 在多个检查点上给出最低重建误差，说明在“分布较稳定、长周期训练”场景，QSW 播种 + 少量 BO 微调比频繁刷新更合适。

### 关键发现
- 方向选择不是单一最优策略：静态积分近似偏向 QSW，动态优化回路更适合 BO 自适应机制。
- “刷新频率”是性能-效率关键旋钮：RBOSW 精度收益高但时间成本大，ARBOSW 提供更平衡的折中。
- ABOSW 的性价比很高：仅替换少量方向（最多约 10%）就能在多任务中得到稳定收益。

## 亮点与洞察
- 亮点 1：把 SW 的改进点精准定位在“方向集设计”而非“损失重写”。这让方法能无缝接入现有 OT/SW 训练代码，工程迁移成本低。
- 亮点 2：提出“QSW 几何先验 + BO 数据自适应”的混合范式。它不是抛弃低差异采样，而是把其作为稳定起点，再用 BO 做局部预算再分配。
- 亮点 3：作者对偏差问题态度诚实。论文明确 BO 方向会引入偏置，不把它包装成无偏估计器，而是把目标设为“优化收敛效率”。
- 洞察：在 SW 被反复调用的场景里，“估计器与任务共适应”比“一次性全局好积分规则”更重要，这一点对生成模型与配准任务都很有启发。

## 局限与展望
- 偏差问题：BO 选方向不是经典无偏球面积分，理论上不保证有限样本无偏估计 SW 真值；论文也把这一点列为未来研究重点。
- 维度扩展：尽管文中讨论了高维 BO 进展，但 GP 代理在超高维和超大预算下仍可能受限，需更强 surrogate（如神经代理）或并行加速。
- 超参依赖：刷新间隔 $R$、候选池大小 $n_c$、批量大小 $b$、UCB 的 $\beta$ 都会影响效果，跨任务迁移时仍需调参。
- 计算开销：在强调快速收敛的同时，RBOSW 这类高频刷新方案在时间上明显更重，真实部署需要基于任务时延预算选型。

## 相关工作与启发
- **vs QSW/RQSW（Nguyen et al., 2024）**：QSW 系列优势是球面覆盖质量与稳定实现；本文优势是任务自适应。ABOSW/ARBOSW 证明两者可以组合而非二选一。
- **vs Importance-weighted SW（Nguyen & Ho, 2023）**：后者通过重加权切片构造新的切片分布；本文通过 BO 直接学习方向集，本质更接近“自适应实验设计”。
- **vs Bayesian Quadrature**：BQ 面向积分本身，但在“每步积分对象都变”的优化回路中需要频繁重拟合权重，代价较高；本文改为学习方向选择器，更适配迭代任务。
- 启发：未来可把“方向学习器”做成元学习模块，在一类任务上预训练方向策略，再迁移到新数据分布，实现 warm-start 的自适应 SW。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ 方向选择器 BO 化并与 QSW 组合的思路清晰且有增量创新。
- 实验充分度: ⭐⭐⭐⭐☆ 覆盖合成验证与三类下游任务，且与强基线保持同协议比较。
- 写作质量: ⭐⭐⭐⭐☆ 方法定义和工程细节交代完整，偏差与适用边界说明较诚实。
- 价值: ⭐⭐⭐⭐☆ 对需要 SW-in-the-loop 的实践场景有较高可落地价值，尤其是 ABOSW/ARBOSW 的折中方案。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Tree-Sliced Wasserstein Distance: A Geometric Perspective](../../ICML2025/image_generation/tree-sliced_wasserstein_distance_a_geometric_perspective.md)
- [\[ICML 2025\] Tree-Sliced Wasserstein Distance with Nonlinear Projection](../../ICML2025/image_generation/tree-sliced_wasserstein_distance_with_nonlinear_projection.md)
- [\[ICLR 2026\] Learning a Distance Measure from the Information-Estimation Geometry of Data](learning_a_distance_measure_from_the_information-estimation_geometry_of_data.md)
- [\[ICLR 2026\] ToProVAR: Efficient Visual Autoregressive Modeling via Tri-Dimensional Entropy-Aware Semantic Analysis and Sparsity Optimization](toprovar_efficient_visual_autoregressive_modeling_via_tri-dimensional_entropy-aw.md)
- [\[ICLR 2026\] ViPO: Visual Preference Optimization at Scale](vipo_visual_preference_optimization_at_scale.md)

</div>

<!-- RELATED:END -->
