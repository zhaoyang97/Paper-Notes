---
title: >-
  [论文解读] AdS-GNN - a Conformally Equivariant Graph Neural Network
description: >-
  [ICLR2026][图学习][共形对称性] 这篇论文把点云从平直欧氏空间"抬升"到高一维的反德西特（AdS）空间，借助物理里 AdS 等距变换 ⇔ 边界共形变换的对应关系，构造出第一个对**完整共形群**（含平移、旋转、缩放，乃至非仿射的特殊共形变换）等变的图神经网络 AdS-GNN，并在超像素 MNIST、形状分割和 Ising 模型关联函数等任务上展现出更强的尺度泛化能力，还能从训练好的网络里直接读出共形维数这种物理上有意义的普适量。
tags:
  - "ICLR2026"
  - "图学习"
  - "共形对称性"
  - "图神经网络"
  - "Anti-de Sitter 空间"
  - "AdS/CFT"
  - "消息传递"
---

# AdS-GNN - a Conformally Equivariant Graph Neural Network

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=EIyvsL5Cue](https://openreview.net/forum?id=EIyvsL5Cue)  
**代码**: 待确认  
**领域**: 图学习 / 等变神经网络 / 几何深度学习  
**关键词**: 共形对称性、等变 GNN、Anti-de Sitter 空间、AdS/CFT、消息传递

## 一句话总结
这篇论文把点云从平直欧氏空间"抬升"到高一维的反德西特（AdS）空间，借助物理里 AdS 等距变换 ⇔ 边界共形变换的对应关系，构造出第一个对**完整共形群**（含平移、旋转、缩放，乃至非仿射的特殊共形变换）等变的图神经网络 AdS-GNN，并在超像素 MNIST、形状分割和 Ising 模型关联函数等任务上展现出更强的尺度泛化能力，还能从训练好的网络里直接读出共形维数这种物理上有意义的普适量。

## 研究背景与动机
**领域现状**：等变神经网络是几何深度学习的主线之一，从 Cohen & Welling 的群等变卷积开始，社区已经把平移、旋转、反射（E(n)）、一般等距变换、乃至各种李群（仿射群、半黎曼流形上的等距群）的等变性都研究得相当透彻。E(n) 等变 GNN（EGNN）这类方法用节点间的欧氏距离 $\|p_i-p_j\|$ 来条件化消息，天然对旋转平移不变。

**现有痛点**：但这些方法都停在"距离保持"型的对称性上。共形变换是一类**保角但不保距**的更大对称群——它额外包含缩放（dilatation）和**特殊共形变换**（special conformal transformation，一种非仿射的"先反演、再平移、再反演"操作）。尺度等变本身已被零散研究过（多尺度卷积、Fourier 层、尺度空间理论），但没有任何方法处理完整共形群，尤其是那个棘手的非仿射特殊共形变换。

**核心矛盾**：共形群 $\mathrm{Conf}(\mathbb{R}^d)$ 的元素在欧氏空间里的作用是**非线性、坐标依赖**的（见特殊共形变换那一项 $\frac{x'}{\|x'\|^2}=\frac{x}{\|x\|^2}-b$），直接在 $\mathbb{R}^d$ 上设计对它等变的算子非常困难——这正是过去方法绕不过去的坎。

**切入角度**：作者从理论物理的 AdS/CFT 对应里借了一个关键事实——$d$ 维平直空间的全局共形群 $\mathrm{Conf}_g(\mathbb{R}^d)$，恰好**同构于** $(d{+}1)$ 维 Anti-de Sitter 空间 $\mathrm{AdS}_{d+1}$ 的等距群 $PO(d{+}1,1)$。也就是说，平直空间里"难搞"的共形变换，在高一维的 AdS 空间里变成了"好搞"的等距变换，而等距群上的等变网络在几何深度学习里已经有成熟工具。

**核心 idea**：把数据从 $\mathbb{R}^d$ 抬升（lift）到 $\mathrm{AdS}_{d+1}$，用 AdS 上的不变测地距离（proper distance）来条件化消息传递，从而"免费"获得对完整共形群的等变性。

## 方法详解

### 整体框架
AdS-GNN 的输入是 $\mathbb{R}^d$ 里的一个点云 $\{x_i\}_{i=1}^N$（可带特征 $h_i$），输出是每个节点的共形不变表示，可用于分类，也可还原成带指定共形维数的边界场用于回归。整条管线分三段：先把每个平直空间的点抬升（lift）到 $\mathrm{AdS}_{d+1}$（多算出一个尺度坐标 $z$），把点云变成 AdS 流形上的一张图；再在这张图上做消息传递，但消息只依赖 AdS 不变测地距离 $D(X_i,X_j)$（标量任务用 AdS-GNN，向量任务用基于 Clifford 代数的 AdS-CEGNN）；最后按任务读出——分类直接对节点求和，回归则乘上 $z$ 的幂把不变特征还原成带共形维数 $\Delta$ 的场。

直观上，多出来的那一维 $z$ 编码的是"系统自由度的长度尺度"：$z$ 越大代表越粗的尺度。共形数据被想象成"住在 $\mathrm{AdS}_{d+1}$ 的边界（$z=0$）上"，而网络是把边界数据延拓进 AdS 体内（bulk）来计算。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["输入点云<br/>x_i ∈ R^d (+ 特征 h_i)"] --> B["抬升到 AdS<br/>KNN 求质心定 z_i<br/>特征乘 z^Δ"]
    B --> C["按 proper distance<br/>建 KNN 图"]
    C --> D["不变消息传递<br/>标量: AdS-GNN<br/>向量: AdS-CEGNN"]
    D -->|分类: 求和| E["共形不变输出"]
    D -->|回归: 乘 z^{-Δ}| F["共形场 O(x)<br/>可读出维数 Δ"]
```

### 关键设计

**1. 抬升到 AdS 空间：把非线性的共形变换换成线性的等距变换**

这是整篇论文的基石，直接针对"共形变换在 $\mathbb{R}^d$ 上非线性、难以直接设计等变算子"这个核心痛点。$\mathrm{AdS}_{d+1}$ 可以看成嵌在 $\mathbb{R}^{d+1,1}$ 里、满足约束 $\|Y\|=-1$（度规 $\eta=\mathrm{diag}(-1,1,\dots,1)$）的双曲面的一支。用坐标 $X=(x,z)\in\mathbb{R}^d\times\mathbb{R}_{>0}$ 解这个约束后，AdS 上的黎曼度规是 $ds^2=\frac{1}{z^2}\big(\sum_a (dx^a)^2+dz^2\big)$。关键事实是：$PO(d{+}1,1)$ 既是 $\mathrm{AdS}_{d+1}$ 的等距群，又正好是 $\mathbb{R}^d$ 的全局共形群——这不是巧合，而是量子引力里 AdS/CFT 对应的运动学内核。AdS 流形在 $z=0$ 处有一个 $d$ 维边界，等距群作用在边界点 $(x,z=0)$ 上的方式，正是平直空间里那套共形变换（平移/旋转/缩放/特殊共形）。所以"在 AdS 体内做局部操作 ⇒ 在边界得到共形等变操作"。

**2. AdS 嵌入算法：用近邻质心确定尺度坐标 $z$，精确保尺度、温和破坏特殊共形**

天真地把点直接放在边界 $z=0$ 上行不通——度规在 $z=0$ 有奇异性，任意两个边界点的 proper distance 都是无穷大，强行取一个固定的小 $z$ 又会破坏对称性。作者的做法（Algorithm 1）是：先把每个点临时放在 $z=z_0$（小正则量），对它的 $k_{\text{lift}}$ 个近邻算一个 **AdS 质心**（欧氏质心在双曲空间的推广，用 Galperin 1993 的构造），质心的 $z$ 坐标在 $z_0\to 0$ 时有有限极限，且依赖于这些点之间的相对间距；最终把点嵌入到 $X_i=(x_i,\hat z_i)$。直觉上"一个点合适的长度尺度 = 它到邻居的距离"。这个选择**精确保持尺度不变**，但会**温和破坏特殊共形变换**——这在物理上是预期的，因为任何正则化都必然破坏共形不变性（Cardy），实验里作者也实测了这种破坏很小。论文形式化为 Proposition 4.1.1：抬升过程对平移、旋转、缩放生成的子群严格等变。

此外，输入特征要当作底层共形场 $O(x)$ 的采样，体内特征是标量、不带 $\lambda^\Delta$ 因子，所以抬升特征时要乘一个尺度因子：$h_i^{\text{lifted}}=\hat z_i^{\Delta}\, h_i^{\text{input}}$（图像数据 $\Delta=0$，可跳过）。这一步对应 AdS/CFT 里把体内物理与边界物理联系起来的 bulk-to-boundary propagator。

**3. 基于 proper distance 的不变消息传递：把欧氏距离换成 AdS 测地距离**

有了 AdS 上的点 $\{X_i\}$，就在上面跑 GNN。作者以 EGNN 为蓝本——EGNN 的消息是 $m_{ij}=\psi_e(h_i^l,h_j^l,\|p_i-p_j\|^2)$，只依赖欧氏距离。AdS-GNN 只改一处：把欧氏距离换成 $PO(d{+}1,1)$ 不变的 AdS proper distance

$$\cosh D(X,X')=\frac{z^2+z'^2+\sum_a (x^a-x'^a)^2}{2zz'},$$

即 $m_{ij}=\psi_e(h_i^l,h_j^l,D(X_i,X_j))$。若图未给边，就用 proper distance 取 $k_{\text{con}}$ 近邻建图。这样几乎不增加计算开销就得到了共形等变 GNN，而且 proper distance 同时在普通空间和尺度（$z$ 方向）上引入了局部性概念。值得强调：虽然抬升步骤温和破坏了特殊共形，但 GNN 本身对整个 $\mathrm{Conf}(\mathbb{R}^d)$ 是**精确不变**的。

**4. AdS-CEGNN 处理向量特征 + 输出层还原共形场（可解释性来源）**

上面的 proper-distance 消息只依赖距离，故只能产出不变特征。要做更强的等变（比如预测向量场），作者利用 $\mathrm{AdS}_{d+1}$ 是 $\mathbb{R}^{d+1,1}$ 里共形等变嵌入的子流形这一点，套用 $O(d{+}1,1)$ 等变的 Clifford 群网络（Ruhe et al. 2023），消息变成在多向量上运算的 $M_{ij}=\psi_e(H_i^l,H_j^l,X_i,X_j)$，记作 AdS-CEGNN，对受限共形群 $\mathrm{Conf}(\mathbb{R}^d)=PO_0(d{+}1,1)$ 等变。输出端则是抬升的逆操作：若要让输出是带共形维数 $\Delta$ 的边界场，就取 $O(x_i)=\hat z_i^{-\Delta} h_i^{l_{\text{final}}}$，保证它在缩放下按 $O'(\lambda x)=\lambda^{-\Delta}O(x)$ 变换。这里 $\Delta$ 可设为**可训练参数**，训练完直接读出来就是网络"学到的共形维数"——这正是模型可解释性的来源。

### 损失函数 / 训练策略
计算复杂度由两部分组成：AdS 嵌入用 KNN，$O(N\log N)$；MPNN 部分随节点数线性。回归任务用相对 $L2$ 损失。Ising 关联函数任务里，输出是 $N$ 点共形场的乘积，用 $N$ 份输出公式组合：

$$\log\big(\mathrm{Pred}_a(\{x_i\})\big)=\mathrm{AdSGNN}_a(\{x_i\})-\Delta_a\sum_{i=1}^N \log(\hat z_i),$$

其中 $a\in\{\sigma,\epsilon\}$，$\Delta_a$ 是可训练参数，训练后即为学到的自旋/能量场共形维数。

## 实验关键数据

### 主实验
作者把任务分成计算机视觉与物理两类。

| 任务 | 对比基线 | 关键结果 |
|------|----------|----------|
| SuperPixel MNIST 分类 | EGNN / PΘNITA 等 | 同分布 4.09% 错误率，与 roto-等变方法持平；**在旋转+缩放增广测试集上仍保持 4.09%，而 EGNN/PΘNITA 退化到随机猜测** |
| Shape 分割 | EGNN / MPNN | 同分布即超过 EGNN，尤其训练点少时优势明显 |
| 2d Ising 关联函数回归 | EGNN / MPNN | 所有规模下相对 $L2$ 最低；**2 点函数比基线好一个数量级以上**（因其形式被共形不变性完全固定，只需学两个数） |
| 3d Ising（非可解，用共形 bootstrap 造数据） | EGNN / MPNN | 明显更优，且恢复出 $\Delta_\sigma=0.518$ |
| N-body 带电粒子动力学（向量任务） | CEGNN (Ruhe et al., SOTA) | AdS-CEGNN 更优，并正确恢复加速度场维数 $\Delta_a=2$ |

最具说服力的对比是 SuperPixel MNIST 的增广测试：当测试集叠加旋转+缩放后，EGNN 这类只对旋转平移不变的方法直接崩到随机水平，而 AdS-GNN 因为精确尺度不变，错误率纹丝不动。

### 消融 / 分析

| 配置 / 维度 | 现象 | 说明 |
|------|------|------|
| 同分布 vs 增广测试 | AdS-GNN 几乎不掉点，EGNN 崩盘 | 尺度等变带来的鲁棒性 |
| 训练点数 64→32768（Shapes/Ising） | AdS-GNN 在小样本区优势最大 | 等变性带来更高样本效率 |
| 外推到训练范围外的坐标 / 不同 $N$ | AdS-GNN 泛化更好，连通度越密优势越大 | 说明它真正学到底层物理 |
| 读出 $\Delta_\sigma,\Delta_\epsilon$ | 非常接近真值（2d $\Delta_\sigma=1/8$、$\Delta_\epsilon=1$；3d $\Delta_\sigma\approx0.518$） | 可解释性 |

### 关键发现
- **尺度泛化是最大卖点**：只要任务带尺度变化，AdS-GNN 相对 EGNN 的优势就拉开，且不需要多尺度增广训练。
- **可解释性是副产品但很珍贵**：能从网络里直接读出共形维数这种与微观细节无关的普适物理量；N-body 里若训练后 $\Delta$ 恰好正确，等于给了一个"模型是否正确处理数据"的额外验证信号——这是一般深度模型（只看性能）没有的。
- **轨道信息是短板**：在 MNIST 上略逊于带朝向信息的 PΘNITA，在 Shapes 上落后于带相对位移 $x_i-x_j$ 的 MPNN，因为 AdS-GNN 只用不变描述子、缺少朝向。

## 亮点与洞察
- **把物理对偶当成工程工具**：AdS/CFT 通常是量子引力里的深奥话题，作者只取其"运动学"——边界共形 ⇔ 体内等距的同构，把一个难造的等变性化简成已有成熟工具能处理的等距等变性，这种"换空间换难度"的思路非常漂亮，可迁移到其他难处理的对称群。
- **几乎零额外开销**：相比欧氏 EGNN，AdS-GNN 只是把距离公式从欧氏换成 proper distance，复杂度同阶，却换来完整共形等变。
- **尺度坐标 $z$ 的物理直觉**：把多出来的一维解释为"长度尺度"，并用近邻质心自适应地确定每个点的 $z$，既保尺度不变又只温和破坏特殊共形——这个工程折中既诚实又实用。
- **可训练的共形维数 $\Delta$**：把物理量塞进可学习参数，使网络兼具预测力与可解释性，值得借鉴到其他"有已知守恒量/标度律"的科学机器学习任务。

## 局限与展望
- **特殊共形变换并非精确等变**：抬升步骤温和破坏了它（正则化的必然代价），只在实验上验证破坏很小，缺少误差上界的理论刻画。
- **缺少朝向信息**：只用不变描述子，导致在需要方向判别的任务（MNIST、Shapes）上不如带朝向/相对位移的方法，这是不变性换来的代价。
- **物理任务偏重**：最亮眼的结果集中在 Ising、N-body 等物理任务上；视觉侧除增广鲁棒性外，常规分割（PascalVOC）与 EGNN 基本持平，实际视觉收益还需更多验证。
- **可扩展信息有限**：作者提到共形场论还由 3 点系数 $c_{abc}$ 等普适量刻画，能否从网络里也提取出来仍是开放问题。
- 展望：作者看好把这种对局部尺度变化天然一致的性质用到计算机视觉与机器人——物体在场景中以不同尺度出现时预测保持一致，无需大量多尺度训练数据。

## 相关工作与启发
- **vs EGNN / E(n) 等变网络**：EGNN 用欧氏距离条件化消息，只对旋转平移反射等变；AdS-GNN 把距离换成 AdS proper distance，等变群扩大到完整共形群，在尺度增广下不崩盘，这是本质区别。
- **vs 尺度等变方法（多尺度卷积 / Fourier 层 / 尺度空间）**：这些只处理尺度子群，且常把额外维当尺度；AdS-GNN 的 AdS 几何**按构造**强制对全部共形变换（含非仿射的特殊共形）等变，而不仅是尺度或等距。
- **vs CEGNN (Ruhe et al. 2023)**：CEGNN 用 Clifford 群网络做 $O(n)$ 等变；AdS-CEGNN 把它搬到 $\mathbb{R}^{d+1,1}$ 上、利用 AdS 是共形等变子流形，从而升级到共形等变，在 N-body 上超过作为 SOTA 的 CEGNN。
- **vs 半黎曼流形等变网络（Weiler/Zhdanov 等）**：本文可视为他们一般框架在 AdS 这一特例上的具体落地，并补上了"如何把平直数据抬升进 AdS"这关键一环。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个对完整共形群（含特殊共形变换）等变的 GNN，思路从 AdS/CFT 借力，原创性强
- 实验充分度: ⭐⭐⭐⭐ 覆盖视觉与物理多类任务并验证泛化/可解释性，但视觉侧常规收益有限、缺破坏性理论界
- 写作质量: ⭐⭐⭐⭐ 物理动机讲得清楚、图文对应好，但 AdS/CFT 背景门槛偏高
- 价值: ⭐⭐⭐⭐⭐ 既给等变学习开了共形这条新线，又示范了从网络提取普适物理量的可解释范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] On The Expressive Power of GNN Derivatives](on_the_expressive_power_of_gnn_derivatives.md)
- [\[AAAI 2026\] On Stealing Graph Neural Network Models](../../AAAI2026/graph_learning/on_stealing_graph_neural_network_models.md)
- [\[AAAI 2026\] Magnitude-Modulated Equivariant Adapter for Parameter-Efficient Fine-Tuning of Equivariant Graph Neural Networks](../../AAAI2026/graph_learning/magnitude-modulated_equivariant_adapter_for_parameter-efficient_fine-tuning_of_e.md)
- [\[ICLR 2026\] GNN-as-Judge: Unleashing the Power of LLMs for Graph Learning with GNN Feedback](gnn-as-judge_unleashing_the_power_of_llms_for_graph_learning_with_gnn_feedback.md)
- [\[ICML 2026\] Full-Spectrum Graph Neural Network: Expressive and Scalable](../../ICML2026/graph_learning/full-spectrum_graph_neural_network_expressive_and_scalable.md)

</div>

<!-- RELATED:END -->
