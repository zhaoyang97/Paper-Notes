---
title: >-
  [论文解读] Conservation Laws for Modern Neural Architectures
description: >-
  [ICML2026][优化/理论][守恒律] 这篇论文把"刻画训练动力学中所有守恒量"的问题重述为求解一个与数据无关的偏微分方程，并借助复分析里的亚纯延拓技巧，第一次给出了 GELU/SiLU/SwiGLU 前馈网、多头注意力（含正弦 PE 与 RoPE）、以及各种门控的 MoE 的**完整守恒律清单**，顺手解决了 Marcotte et al. (2025) 留下的多头注意力开放问题。
tags:
  - "ICML2026"
  - "优化/理论"
  - "守恒律"
  - "梯度流"
  - "隐式偏置"
  - "注意力机制"
  - "RoPE"
  - "MoE"
---

# Conservation Laws for Modern Neural Architectures

**会议**: ICML2026  
**arXiv**: [2606.17816](https://arxiv.org/abs/2606.17816)  
**代码**: 待确认  
**领域**: 优化理论 / 隐式偏置  
**关键词**: 守恒律, 梯度流, 隐式偏置, 注意力机制, RoPE, MoE

## 一句话总结
这篇论文把"刻画训练动力学中所有守恒量"的问题重述为求解一个与数据无关的偏微分方程，并借助复分析里的亚纯延拓技巧，第一次给出了 GELU/SiLU/SwiGLU 前馈网、多头注意力（含正弦 PE 与 RoPE）、以及各种门控的 MoE 的**完整守恒律清单**，顺手解决了 Marcotte et al. (2025) 留下的多头注意力开放问题。

## 研究背景与动机

**领域现状**：守恒律（conservation law）指的是沿优化轨迹保持不变的量。在梯度流 $\dot\theta(t)=-\nabla L_{\mathcal D}(\theta(t))$ 下，这类不变量暴露了架构与优化算法共同施加的几何约束，是理解过参数化模型"隐式偏置"的一把钥匙——它解释了为什么从初始化到收敛某些性质始终被保持，也被用于分析收敛性、稳定性，甚至被反过来用来设计加速训练的优化器。

**现有痛点**：虽然人们零散地发现了不少守恒量，但**完整性**结果（证明"这些就是全部"）极其稀少。Marcotte et al. (2023/2024/2025) 用李代数框架把浅层线性网、ReLU 网、ICNN、NMF、ResNet 乃至单头注意力的守恒律刻画清楚了，但对**多头注意力**只找到了若干守恒量、无法证明已穷尽，作者明确把它列为开放问题。而现代大模型早已不用 ReLU 和单头注意力——它们用 GELU/SiLU/SwiGLU、用 RoPE、用稀疏 MoE，这些组件的守恒律几乎是空白。

**核心矛盾**：原有的李代数机器对"非多项式、含位置依赖、含门控路由"的现代算子并不顺手；多头注意力里每个头的 $Q_i^\top Q_i-K_i^\top K_i$ 看似是守恒量，但要证明再没有别的不变量，需要一套能处理超越函数和不连续路由的新工具。

**本文目标**：对现代深度学习里反复出现的关键构件，给出守恒律的**完整刻画**（不仅找到，还要证明穷尽），并附严格证明与实验验证。

**切入角度**：作者不再套用李代数，而是回到第一性原理问"刻画所有守恒律到底意味着什么"——答案是：求解一个把输入 $x$ 消掉之后、只约束 $h$ 本身的偏微分方程。

**核心 idea**：守恒律 $h$ 的梯度必须正交于每个 $\nabla_\theta f_i(x;\theta)$；把这个含 $x$ 的条件通过复分析"延拓—消元"变成与 $x$ 无关的 PDE 约束，再读出 PDE 的特征不变量，就能枚举出全部守恒律。

## 方法详解

### 整体框架

整篇论文是一条"先建统一判据，再用一套证明引擎逐架构推导"的理论流水线。出发点是 Marcotte 等人给出的正交性判据：在"守恒律须对任意数据集成立 + 损失满足 $\mathcal V_\ell=\mathbb R^{d_\text{out}}$"的假设下，$C^1$ 函数 $h$ 是守恒律当且仅当对所有 $i\in[d_\text{out}]$、$\theta$、$x$ 都有

$$\langle\nabla_\theta h(\theta),\,\nabla_\theta f_i(x;\theta)\rangle=0.$$

难点在于这个条件**显式依赖输入 $x$**，而 $h$ 只定义在参数空间上。作者的整体策略是：把上式看成关于 $x$ 的函数，想办法把 $x$ 消掉，得到一组只约束 $\nabla_\theta h$ 的偏微分方程，再解出特征不变量。论文用一个玩具模型 $f(x;a,b)=abx$ 演示了全流程——约束化简成 $b\,\partial_a h+a\,\partial_b h=0$，沿特征曲线发现 $a^2-b^2$ 是不变量，于是 $h$ 在 $\{a^2-b^2=c\}$ 的每个连通分支上为常数。后面四节就是把这套"归约到 PDE → 读出不变量"的方法分别用到 FFN、MHA、RoPE-MHA、MoE 上。整个方法是纯机制推导，没有可串成 pipeline 的多模块结构，因此不配框架图。

### 关键设计

**1. 把"刻画所有守恒律"归约为偏微分方程：回到第一性原理替代李代数机器**

痛点是李代数框架在处理现代算子时既不直观也难推广。作者改用一个更朴素的视角：正交性判据 $\langle\nabla_\theta h,\nabla_\theta f_i(x;\theta)\rangle=0$ 对所有 $x$ 成立，本质上要求把"对 $x$ 的依赖"消成"对 $h$ 的约束"。由于 $h$ 不含 $x$，这些约束自然落到 $\nabla_\theta h$ 的各分量之间，形成一阶 PDE 系统。判据是否"成功"以能否把条件化简成**与 $x$ 无关**、且足以完全确定 $h$ 的条件为准。玩具例子里 $a^2-b^2$ 就是这样被读出来的特征不变量。这个视角的好处是：不变量不再靠猜，而是 PDE 特征曲线的守恒量，因此能保证"穷尽"。

**2. 亚纯延拓 + identity theorem + 极点阶数比较：消去输入依赖的证明引擎**

这是全文证明能跑通的核心机关。对固定的 $\theta,i$，定义 $F_i(x;\theta)\coloneqq\langle\nabla_\theta h,\nabla_\theta f_i(x;\theta)\rangle$。对 FFN、MHA、dense MoE 这些算子，$f_i(\cdot;\theta)$ 是**亚纯（meromorphic）**的，于是 $F_i$ 也是亚纯函数，可以延拓到复平面。把输入限制成 $x=t e_j$（$t$ 在实轴一段开区间上取值），由复分析的 **identity theorem**——一个在带聚点集合上取零的亚纯函数必恒为零——可知 $F_i$ 沿这些子空间恒等于零，从而得到与输入无关的约束。再分析 $F_i$ 各亚纯分量的**极点结构**：若一组亚纯函数的线性组合恒为零，比较极点阶数会强迫主极点对应的系数为零，反复施用就逐条逼出 $\nabla_\theta h$ 必须满足的等式。SMoE 的 Top-$k$ 路由会在激活专家集合切换的边界引入不连续，需要额外论证把不同激活区域连起来（这也是为何 Theorem 4.6 要求 $k>1$：连通性才能强迫各专家门控梯度一致）。

**3. 现代架构的完整守恒律清单：逐个算子读出不变量**

把上面两件事落到具体架构，就得到本文的主结果（均为"完整刻画"，即证明再无其他守恒律）。带 GELU 或 SiLU 的单隐层 FFN $f=A\cdot\text{act}(Bx)$ **没有非平凡守恒律**，所有守恒律都是常数——这本身有点反直觉。SwiGLU 因为有 $A$ 与 $C$ 的乘性交互，反而产生非平凡不变量 $\lVert A_{:,i}\rVert^2-\lVert C_{i,:}\rVert^2$。多头注意力（无 PE）确证了 Marcotte 等人的猜想：守恒律完全由 $Q_i^\top Q_i-K_i^\top K_i$ 与 $V_i^\top V_i-O_i^\top O_i$ 刻画——这正是被悬置的开放问题。正弦 PE 只是对输入做了一个双射平移、不改变 MHA 内部结构，所以守恒律与 vanilla 相同；RoPE 则因位置旋转 $Q_iR_{p-q}K_i^\top$ 把位置耦合进打分，**实质改变**了不变量结构，变成逐 $2\times2$ 块的 $\lVert Q_i^{(j)}\rVert_F^2-\lVert K_i^{(j)}\rVert_F^2$（值-输出对仍保持 $V_i^\top V_i-O_i^\top O_i$）。MoE 的不变量"局部化"在每个专家上（沿用 SwiGLU 的结论），外加一个来自门控的整体不变量 $\sum_{i}W_i$；而且 dense、sparse（$k>1$）、softmax 门控、归一化 sigmoid 门控四种变体**守恒律完全一致**。

各架构的不变量汇总如下：

| 架构 | 完整守恒不变量 |
|------|---------------|
| FFN（GELU / SiLU） | 无非平凡守恒律（$h$ 只能是常数） |
| FFN（SwiGLU） | $\lVert A_{:,i}\rVert^2-\lVert C_{i,:}\rVert^2,\ i\in[d_1]$ |
| MHA（无 PE / 正弦 PE） | $Q_i^\top Q_i-K_i^\top K_i$，$V_i^\top V_i-O_i^\top O_i$，$i\in[n]$ |
| MHA + RoPE | $\lVert Q_i^{(j)}\rVert_F^2-\lVert K_i^{(j)}\rVert_F^2$（逐块 $i,j$），$V_i^\top V_i-O_i^\top O_i$ |
| MoE（dense / sparse $k>1$ / sigmoid 门控） | 各专家的 SwiGLU 不变量 + 门控 $\sum_{i=1}^n W_i$ |

### 损失函数 / 训练策略

本文是理论刻画，不训练新模型。守恒律的成立前提是欧氏梯度流 ODE，并采用 Marcotte et al. (2025) 的结论：**带不带权重衰减，守恒函数是对应的**，因此分析中省略权重衰减、只看纯梯度流。实验侧用真实离散优化（见下）验证理论。

## 实验关键数据

### 主实验

实验目的不是刷指标，而是验证"连续梯度流里的守恒量在离散 SGD 下还近似守恒吗"。理论给出的误差界（基于 Marcotte et al. 2025 的 Proposition 5.1，假设 Hessian 与梯度期望有界）是：

$$\mathbb E\,\big|h(\theta_k)-h(\theta_0)\big|\le\frac{C_hC_L}{2}\sum_{i=0}^{k-1}\tau_i^2.$$

由此推出两种步长策略下守恒误差截然不同的行为：

| 步长策略 | 守恒误差界 | 含义 |
|----------|-----------|------|
| 常数步长 $\tau_k=\tau$ | $\mathcal O(\tau^2 k)$，随迭代线性增长 | 离散化误差逐步累积，但增速可控 |
| 衰减步长 $\tau_k=\tau_0/(k+1)$ | $\mathcal O(\tau_0^2)$，一致有界 | 保证 SGD 收敛时，守恒律近似始终保持 |

作者在语言建模（Qwen-3 架构 + WikiText-103 / Penn Treebank，含 RoPE、dense/sparse MoE、softmax 与归一化 sigmoid 门控）和视觉（ViT + CIFAR-10 / ImageNet-1K，含绝对 PE 与 SwiGLU）两域上验证。每个配置用 10 个随机种子独立训练，按块级相对偏差监控守恒误差：

$$\epsilon_\text{block}(k)=\frac1N\sum_{i=1}^N\frac{\lVert h_i(\theta_k)-h_i(\theta_0)\rVert_2}{\lVert h_i(\theta_0)\rVert_2}.$$

### 消融实验

| 监控对象 | 现象 | 说明 |
|----------|------|------|
| FFN / MHA / RoPE / MoE 门控的守恒量 | 误差随学习率增大而增大，且随迭代温和增长 | 与 $\mathcal O(\tau^2 k)$ 的理论预测一致 |
| 非守恒量（baseline） | 参数微小变化即出现大幅漂移 | 作为定性对照，凸显守恒量演化被严格约束 |

### 关键发现

- 守恒误差的**量级随学习率放大**，跨 MHA、SwiGLU FFN、RoPE、MoE 门控四类组件都吻合 $\mathcal O(\tau^2 k)$ 的标度，说明理论不变量在真实离散训练中确实近似守恒。
- 对照的非守恒量在同样训练下大幅波动，从反面证明被识别出的量"守恒"不是平凡现象。
- 证明框架的边界很清楚：层归一化引入的 $\sqrt{x}$、注意力层堆叠产生的 $e^{1/x}$ 这类含分支点或本质奇点的函数**没有全平面亚纯延拓**，现有引擎不能直接处理——这既是诚实的局限交代，也指出了后续方向。

## 亮点与洞察
- **把"找守恒量"变成"解 PDE"**：最漂亮的一步是用复分析的 identity theorem 把含输入 $x$ 的无穷多约束塌缩成有限的 PDE 约束，从而能证明"穷尽"而不仅是"找到"——这是完整性结果稀少的根本难点所在。
- **GELU/SiLU 没有非平凡守恒律、SwiGLU 却有**：差别只在 SwiGLU 多了一条 $C$ 路径带来的乘性交互，提示"门控/乘性结构"才是非平凡隐式偏置的来源，这个对比很有启发。
- **RoPE 与正弦 PE 的本质差异被量化**：正弦 PE 只是双射平移、守恒结构不变；RoPE 把位置旋转耦合进打分，不变量从整矩阵层面降到逐 $2\times2$ 块层面——这把"RoPE 不只是换了个位置编码"讲清楚了。
- 这套"亚纯延拓 + 极点阶数比较"的证明范式可迁移到其他亚纯算子的隐式偏置分析，是可复用的技术资产。

## 局限与展望
- 作者承认框架不能直接处理含 $\sqrt{x}$（层归一化）或 $e^{1/x}$（多层注意力复合）的算子，因为它们有分支点/本质奇点、无全平面亚纯延拓；需要把 $F_i$ 只延拓到 $\mathbb C^d$ 的合适区域来改进。
- 守恒律基于欧氏梯度流的连续时间理想化；离散 SGD 下只是"近似守恒"，常数步长会线性累积误差，真实大模型训练（含动量、自适应优化器、各种正则）下的偏离程度还需更多验证。
- FFN 分析限制在单隐层；多层堆叠的守恒律刻画仍未解决。
- 改进思路：把奇点分析做得更精细以覆盖 LayerNorm 与深层 Transformer，是把该框架推向"整网"刻画的关键一步。

## 相关工作与启发
- **vs Marcotte et al. (2023/2024)**：他们用李代数框架刻画浅层线性/ReLU 网、ICNN、NMF、动量动力学；本文改用复分析的 PDE 归约，覆盖到 GELU/SiLU/SwiGLU、多头注意力、RoPE、MoE 等现代组件，方法论上更适配非多项式与门控结构。
- **vs Marcotte et al. (2025)**：他们把框架推到 ResNet 与 Transformer，但注意力只做到单头、明确把多头列为开放问题；本文证明了多头注意力守恒律由 $Q_i^\top Q_i-K_i^\top K_i$ 与 $V_i^\top V_i-O_i^\top O_i$ 完全刻画，**直接解决该开放问题**，并进一步给出 RoPE 与各类 MoE 门控的结果。
- **vs Kunin et al. (2021) / Zhang et al. (2025) 等**：这些工作发现了不少具体守恒量但缺完整性证明；本文的价值在于"穷尽性"——证明所列不变量就是全部。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用复分析归约把守恒律刻画推广到现代架构并保证完整性，思路新颖且解决了公开难题。
- 实验充分度: ⭐⭐⭐⭐ 跨语言/视觉、多架构、多种子验证守恒误差标度，作为理论论文的实证已较充分；但缺真实大规模训练下的偏离量化。
- 写作质量: ⭐⭐⭐⭐⭐ 用玩具例子搭直觉、定理与证明草图层次清晰，局限交代诚实。
- 价值: ⭐⭐⭐⭐ 为现代架构隐式偏置研究提供可复用工具与完整结论，理论价值高、直接应用偏间接。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Transformative or Conservative? Conservation Laws for ResNets and Transformers](../../ICML2025/optimization/transformative_or_conservative_conservation_laws_for_resnets_and_transformers.md)
- [\[ICLR 2026\] Saddle-to-Saddle Dynamics Explains A Simplicity Bias Across Neural Network Architectures](../../ICLR2026/optimization/saddle-to-saddle_dynamics_explains_a_simplicity_bias_across_neural_network_archi.md)
- [\[ICML 2026\] Muon in Associative Memory Learning: Training Dynamics and Scaling Laws](muon_in_associative_memory_learning_training_dynamics_and_scaling_laws.md)
- [\[NeurIPS 2025\] Emergence and Scaling Laws in SGD Learning of Shallow Neural Networks](../../NeurIPS2025/optimization/emergence_and_scaling_laws_in_sgd_learning_of_shallow_neural_networks.md)
- [\[NeurIPS 2025\] Learning Quadratic Neural Networks in High Dimensions: SGD Dynamics and Scaling Laws](../../NeurIPS2025/optimization/learning_quadratic_neural_networks_in_high_dimensions_sgd_dynamics_and_scaling_l.md)

</div>

<!-- RELATED:END -->
