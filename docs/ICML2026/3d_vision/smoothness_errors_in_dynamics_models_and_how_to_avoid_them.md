---
title: >-
  [论文解读] Smoothness Errors in Dynamics Models and How to Avoid Them
description: >-
  [ICML 2026][3D视觉][图神经网络] 作者从理论上指出 Kiani 等人的 "unitary GNN" 因为强行保持 Rayleigh 商而对热扩散这类"天然会变光滑"的物理系统过度约束，进而提出"松弛 unitary 卷积"（R-UniGraph / R-UniMesh）并把整套 Rayleigh 商-unitary 卷积框架从图扩展到三角网格，在 MeshPDE 与 WeatherBench22 上同时超越多类强基线。
tags:
  - "ICML 2026"
  - "3D视觉"
  - "图神经网络"
  - "网格学习"
  - "过平滑/欠平滑"
  - "Unitary 卷积"
  - "Rayleigh 商"
  - "天气预报"
---

# Smoothness Errors in Dynamics Models and How to Avoid Them

**会议**: ICML 2026  
**arXiv**: [2602.05352](https://arxiv.org/abs/2602.05352)  
**代码**: 有（论文末尾提供）  
**领域**: 3D 视觉 / 几何深度学习 / PDE 神经求解器  
**关键词**: GNN, 网格学习, 过平滑/欠平滑, Unitary 卷积, Rayleigh 商, 天气预报

## 一句话总结
作者从理论上指出 Kiani 等人的 "unitary GNN" 因为强行保持 Rayleigh 商而对热扩散这类"天然会变光滑"的物理系统过度约束，进而提出"松弛 unitary 卷积"（R-UniGraph / R-UniMesh）并把整套 Rayleigh 商-unitary 卷积框架从图扩展到三角网格，在 MeshPDE 与 WeatherBench22 上同时超越多类强基线。

## 一句话补：核心命题
GNN 既不能过平滑也不能不平滑——要让架构的平滑倾向恰好匹配真实物理过程的平滑倾向。

## 研究背景与动机

**领域现状**：用神经网络求解定义在网格/流形上的 PDE（热扩散、波动、Cahn–Hilliard、地球大气）是科学计算近两年最活跃的方向。主流做法是把流形离散成 mesh，再用支持高阶连接性的 mesh-GNN（GCN、MPNN、EGNN、Gauge-Equivariant CNN、Hermes 等）做消息传递。但 GNN 普遍有过平滑问题：层数一多，相邻节点特征趋同。Kiani 等人最近提出 "unitary graph convolution"，把权重矩阵约束成酉矩阵，使卷积保持 Rayleigh 商 $R_\mathcal{G}(X) = \mathrm{Tr}(X^\dagger L X)/\|X\|_F^2$，从而严格不过平滑。

**现有痛点**：unitary 卷积之于 GCN，是"完全不平滑"对"过度平滑"。可现实物理系统大多自带"恰好的平滑度"——热扩散就是让特征变得越来越平滑、波动方程则要求保持高频结构。把 unitary 卷积强行套到这些系统上反而会"欠平滑"——网络无法学到"按物理需要"的中等平滑过程。

**核心矛盾**：GCN 与 unitary 卷积在 Rayleigh 商上是两个极端：GCN 严格降低 Rayleigh 商（持续平滑），unitary 卷积严格保持 Rayleigh 商（完全不平滑）。任何真实的物理动力学需要的是"可调"的平滑率，而不是两个极端任选其一。

**本文目标**：(i) 在理论上给出 unitary 函数的近似误差下界，证明它在角度依赖强的目标上过约束；(ii) 设计可控的 "松弛 unitary 卷积"，让网络能在两个极端之间自由切换；(iii) 把 Rayleigh 商与 unitary 卷积从图扩展到 mesh，使其适用于 PDE/天气预报这类真正的物理任务。

**切入角度**：作者发现 Lie unitary 卷积 $f(X) = \exp(AXW)$，$W = -W^\dagger$ 的 unitary 性来自 Taylor 展开"展到无穷阶"。如果只展到第 $T_\max$ 阶就截断，得到的层不再严格 unitary，但仍以可调的方式接近 unitary——这就是天然的"连续松弛旋钮"。

**核心 idea**：用 Taylor 截断的 Lie 卷积或 "zero-pad + unitary 编码器 + 任意解码器" 两种方式松弛掉严格的 Rayleigh 商保持性，让网络在训练中自适应匹配物理过程的真实平滑度；并通过用 Robust Laplacian 与切线权重把整套理论搬到 mesh 上。

## 方法详解

### 整体框架
这套方法要解决的是"GNN 的平滑倾向和真实物理过程对不上"这件事——既不能像 GCN 那样越叠越平滑，也不能像严格 unitary 卷积那样完全不平滑，而要让平滑率可调。整体思路围绕 Rayleigh 商 $R_\mathcal{G}(X) = \mathrm{Tr}(X^\dagger L X)/\|X\|_F^2$ 这一统一的平滑度度量展开：先在理论上证明严格 unitary 函数对"模长随角度变化"的目标存在不可消除的近似误差下界，说清"为什么要松弛"；再给出两条把"严格保 Rayleigh 商"变成"可调平滑率"的松弛路径——Taylor 截断（得到图上的 R-UniGraph）和零填充编码-解码（得到 mesh 上的 R-UniMesh）；最后用 Robust Laplacian 把 Rayleigh 商与 unitary 卷积从图搬到三角网格，让整套框架第一次能落到真实流形 PDE 任务上。最终模型用 GroupSort 作激活（不破坏模长）、用 MLP/GCN 解码器主动打破 unitary 约束来换取灵活性。

### 关键设计

**1. Taylor 截断的松弛 Lie 卷积（R-UniGraph）：用一个旋钮在过平滑与不平滑之间无级切换**

痛点在于 GCN 严格降低 Rayleigh 商（持续平滑）、unitary 卷积严格保持 Rayleigh 商（完全不平滑），两者是写死的两个极端，而真实物理动力学需要一个介于中间、可调的平滑率。作者注意到 Lie unitary 卷积 $\exp(AXW)$（$W=-W^\dagger$）的 unitary 性其实来自矩阵指数 Taylor 展开"展到无穷阶"，于是把它在第 $T_\max$ 阶截断，得到 $f_{\text{Relaxed}}(X; A, T_\max) = \sum_{i=0}^{T_\max} \frac{1}{i!} L^i(X)$，其中 $L(X)=AXW$。这样 $T_\max$ 就成了平滑率的连续旋钮：$T_\max=1$ 时近似 GCN 行为，$T_\max\to\infty$ 时恢复严格 Lie unitary 卷积，中间值（热扩散实验取 $T_\max=3$、其余实验取 $T_\max=10$）则允许网络在大体保持 Rayleigh 商的前提下做小幅平滑修正。

与 Kiani 等人此前提出的 separable unitary 卷积"放松"相比，这里的关键区别是把放松的来源做了隔离。他们的放松同时动了两处——既截断 Taylor、又让 $U$ 不再 unitary，导致无法定量分析松弛究竟来自哪里；R-UniGraph 单独保留 Lie 形式的反对称 $W$，使 $T_\max$ 成为 Rayleigh 商保持性的唯一旋钮，物理上若已知目标过程有多平滑，甚至可以直接查表选 $T_\max$。

**2. 零填充编码-解码松弛（R-UniMesh）：把容量集中到解码器，绕开深 unitary 堆叠的训练不稳定**

Taylor 截断那一路不能改变通道维数，想加大参数量只能靠加深，而深的 unitary 堆叠会遇到 Balduzzi 等人指出的"shattered gradients"训练不稳定问题。R-UniMesh 改走"宽 + 浅"：先用零填充 $f_{\text{pad}}: \mathbb{R}^{n\times d_{in}}\to\mathbb{R}^{n\times d_{out}}$ 把节点特征补到隐藏维度（零填充保模长，因而天然保持 Rayleigh 商），再堆 $k$ 层 Lie unitary mesh 卷积 $f_{\text{UniMeshConv}}^{\text{Lie}}(X; A, \mathcal{W}) = \exp(\tilde A X W)$ 作编码器 $E$，其中 $\tilde A = D^{-1/2}(\mathcal{W}\odot A)D^{-1/2}$ 引入 cotangent 权重 $\mathcal{W}$，最后接一层 MLP 或 GCN 解码器 $D$。

解码器在这里身兼两职：既把特征映射到目标通道数，又主动打破 unitary 约束。这等价于一种清晰的分工——unitary 编码器负责保持几何与平滑结构，解码器负责把输出拟合到任意标签平滑度，参数自由度集中在不受 unitary 约束的解码端，因此既保留了主干的强归纳偏置，又把训练的不稳定挡在了浅层 unitary 主干之外。

**3. Mesh Rayleigh 商与 Robust Laplacian 上的 unitary 卷积：把整套平滑度分析搬到三角网格**

要让上面两路落到真实流形 PDE 任务，必须先把 Rayleigh 商和 unitary 卷积从图扩展到 mesh，而这里的障碍是：传统对称 cotangent Laplacian $\tilde L$ 在非 Delaunay 三角化下会出现负权，使 Rayleigh 商失去正定意义。作者改用 Sharp & Crane 的 Robust Laplacian——它通过最少边重连保证所有 cotangent 权重 $\mathcal{W}_{ij} = \frac{1}{2}(\cot\alpha_{ij} + \cot\beta_{ij})$ 满足 Delaunay 准则（$\alpha_{ij}+\beta_{ij}\le\pi$），从而所有非对角元非负。

$$R_\mathcal{M}(X) = \frac{\mathrm{Tr}(X^\dagger \tilde L X)}{\|X\|_F^2}$$

在这个 mesh Rayleigh 商之上，只需把 separable/Lie unitary 卷积里原来的 $\tilde A$ 换成带 cotangent 权重的归一化邻接矩阵，Corollary 1 即可证明两种 mesh 版本的 unitary 卷积同样保持 mesh Rayleigh 商。这一步的巧妙之处在于：以前 mesh-GNN 用 cotangent 权重只是为了改善数值精度，没人把它和"严格平滑度保持"挂钩；作者用 Delaunay 假设连同 Robust Laplacian 把"权重为正"这一关键条件坐实，于是 unitary 框架在图上的所有数学结论自动迁移到 mesh，省去了重做一遍代数证明。

### 损失函数 / 训练策略
所有任务都直接最小化 MSE/NRMSE 等回归损失，没有引入额外的 Rayleigh 损失项——作者的关键论点正是要让"是否保持平滑"由架构本身的归纳偏置决定而非软约束。R-UniMesh 用 GroupSort（Anil 等 2019）作激活以保证激活不破坏模长，用正交权重（实数任务下已足够），并与 GCN 解码器搭配做端到端反向传播。

## 实验关键数据

### 主实验
作者在两类任务上做评估：(1) MeshPDE（在 PyVista 复杂网格上自回归求解 heat、wave、Cahn–Hilliard 方程）；(2) WeatherBench22 全球天气预报（T850 温度、Z500 位势）。

| 数据集 | 任务 | 指标 | R-UniMesh | 最强基线 | 备注 |
|--------|------|------|-----------|----------|------|
| MeshPDE / Heat | 自回归 196 步 | NRMSE ↓ | **51.9 ± 3.6** | 73.0 ± 4.7（Hermes） | 几乎降一半 |
| MeshPDE / Heat | 同上 | RE ↓ | **9.1 ± 7.4** | 14.2 ± 1.4（EMAN） | 平滑度匹配最准 |
| MeshPDE / Wave | 自回归 196 步 | NRMSE ↓ | **236.5 ± 6.4** | 281.3 ± 15.5（EMAN） | 仍领先 |
| MeshPDE / Cahn–Hilliard | 同上 | NRMSE ↓ | 123.9 ± 2.6 | **121.2 ± 1.8**（GemCNN） | 接近 SOTA |
| WB22 / T850 | RMSE @ 1-10 d | RMSE / ACC | 早期与 SOTA 相当 | Pangu/GraphCast | 训练数据受限下仍可比 |

### 消融实验
作者在 motivating experiment 中用 2D 网格上的热扩散对比 GCN、Lie unitary、R-UniGraph：

| 配置 | MSE ($\times 10^{-2}$) ↓ | MRE ($\times 10^{-2}$) ↓ | 解读 |
|------|--------------------------|---------------------------|------|
| GCN | 1.08 | 5.99 | 过平滑、误差大 |
| Lie Uni | 0.14 | 8.86 | 完全不平滑，欠平滑 |
| **R-UniGraph (Ours, $T_\max=3$)** | **0.11** | **2.07** | MSE 与 Rayleigh 商误差同时最优 |

### 关键发现
- R-UniGraph 在 MSE 和 Rayleigh 误差两项上同时优于 GCN 与严格 unitary，说明"恰好的平滑率"比"完全不变 / 总是变"都更接近物理真值。
- 在 mesh 上的热扩散预测里，R-UniMesh 的 Rayleigh 误差几乎在每个时间步都与 ground truth 一致，可视化显示其 rollout 不像 EMAN 那样过平滑、也不像 Hermes 那样欠平滑。
- 在简单几何（Cahn–Hilliard 用 toroid mesh）上几乎所有等价/unitary/MPNN 都打平，差异主要出现在复杂几何泛化（不同 PyVista mesh）上——证明"几何归纳偏置"在跨网格泛化时才发挥关键作用。
- GCN 与 EGNN 在所有任务上都垫底，说明仅仅靠消息传递或欧氏等价性不足以模拟流形上的 PDE，必须显式考虑 mesh 上的平滑结构。

## 亮点与洞察
- "近似误差下界 + 旋钮式松弛"的设计逻辑非常优雅：先用 Theorem 1（在 fundamental domain 上对模长方差积分给出 $\int p(\|te\|)\mathbb{V}_{Gz}[\|f\|]dz$ 下界）说明严格 unitary 的"代价"，再用 Taylor 截断把"严格"变成"可调"——这种"诊断—对症"的写作模板对其他归纳偏置的研究也有借鉴价值。
- Rayleigh 商误差（RE）这一新指标本身就是论文级别的贡献：它给 PDE neural surrogate 提供了一个比 RMSE 更有物理含义的平滑度对齐度量，未来 mesh-GNN 论文应该把它作为标准指标。
- 把 Robust Laplacian + 切线权重 + unitary 卷积三者打包成完整 mesh 框架，等于给后续做流形 PDE 求解器的研究者提供了"现成的脚手架"。

## 局限与展望
- $T_\max$ 与 zero-pad 维度仍需根据任务先验调参；作者建议在已知目标平滑度时查表选 $T_\max$，但对未知 PDE 没有自动调度策略，未来可考虑可学习的 $T_\max$ 或自适应注意力来动态决定截断阶。
- 实验在 Cahn–Hilliard 这类既不严格平滑也不严格保平滑的方程上优势不明显，说明现有"两端拉松"的二元视角对真正中间态系统仍偏粗；可能需要更细的 Rayleigh 商谱分析。
- WB22 实验受算力限制只用了 $1.5°$ 分辨率和小规模训练，与 ECMWF SOTA 相比仍有差距，未来在大规模上的可扩展性需要进一步验证。

## 相关工作与启发
- **vs Kiani et al. 2024 (Unitary GNN)**：本文是它的直接延伸与"反向修正"——既证明其严格保 Rayleigh 商在动力学任务上是缺陷，又用 Taylor 截断给出可控松弛，并把整套理论推广到 mesh。
- **vs Hermes / EMAN / GemCNN (Gauge equivariant mesh GNN)**：这些方法靠 gauge equivariance 处理 mesh 上的方向不变性，R-UniMesh 用 Rayleigh 商保持性提供了一个正交且互补的归纳偏置，在热扩散等强平滑任务上明显更优。
- **vs Subich 2025 / Bonev 2025（在频谱域做训练目标）**：这些工作通过软约束（频谱损失）改善天气模型的 effective resolution；R-UniMesh 把同样目标通过架构约束达成，避免了损失加权调参，且对 PDE 任务有更明确的物理意义。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 第一次从 Rayleigh 商角度把"过平滑 vs 欠平滑"统一刻画，并给出可控松弛与 mesh 扩展，理论+方法双新。
- 实验充分度: ⭐⭐⭐⭐ 在 motivating 实验、MeshPDE 多 PDE 多 mesh 与 WB22 真实数据上都验证；WB22 受算力限制略遗憾。
- 写作质量: ⭐⭐⭐⭐⭐ 理论与方法、动机与实验衔接非常顺畅，定理与命题的引用关系都标注得很清楚。
- 价值: ⭐⭐⭐⭐ 既是一篇可用的 PDE neural surrogate（在热扩散等任务上 SOTA），也是理论上具有指导意义的 mesh-GNN 归纳偏置研究。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] One Diffusion to Generate Them All](../../CVPR2025/3d_vision/one_diffusion_to_generate_them_all.md)
- [\[CVPR 2026\] Charge: A Comprehensive Novel View Synthesis Benchmark and Dataset to Bind Them All](../../CVPR2026/3d_vision/charge_a_comprehensive_novel_view_synthesis_benchmark_and_dataset_to_bind_them_a.md)
- [\[NeurIPS 2025\] Temporal Smoothness-Aware Rate-Distortion Optimized 4D Gaussian Splatting](../../NeurIPS2025/3d_vision/temporal_smoothness-aware_rate-distortion_optimized_4d_gaussian_splatting.md)
- [\[CVPR 2026\] LumiMotion: Improving Gaussian Relighting with Scene Dynamics](../../CVPR2026/3d_vision/lumimotion_gaussian_relighting_dynamics.md)
- [\[ICML 2026\] FoundObj: Self-supervised Foundation Models as Rewards for Label-free 3D Object Segmentation](foundobj_self-supervised_foundation_models_as_rewards_for_label-free_3d_object_s.md)

</div>

<!-- RELATED:END -->
