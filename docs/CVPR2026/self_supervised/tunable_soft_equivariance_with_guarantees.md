---
title: >-
  [论文解读] Tunable Soft Equivariance with Guarantees
description: >-
  [CVPR 2026][自监督学习][软等变性] 本文提出一个架构无关的"软等变"框架：把任意预训练模型的权重投影到一个由群的李代数表示决定的子空间里，用一个截断阈值 $b$ 连续地调节模型从「完全等变」到「完全不等变」，并给出等变误差的可证明上界；在 ImageNet/分割/轨迹预测上同时提升精度并降低等变误差。
tags:
  - "CVPR 2026"
  - "自监督学习"
  - "软等变性"
  - "等变误差界"
  - "李代数投影"
  - "Schur 分解"
  - "预训练模型适配"
---

# Tunable Soft Equivariance with Guarantees

**会议**: CVPR 2026  
**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Rahman_Tunable_Soft_Equivariance_with_Guarantees_CVPR_2026_paper.html)  
**代码**: 待确认  
**领域**: 表示学习 / 等变性理论  
**关键词**: 软等变性, 等变误差界, 李代数投影, Schur 分解, 预训练模型适配

## 一句话总结
本文提出一个架构无关的"软等变"框架：把任意预训练模型的权重投影到一个由群的李代数表示决定的子空间里，用一个截断阈值 $b$ 连续地调节模型从「完全等变」到「完全不等变」，并给出等变误差的可证明上界；在 ImageNet/分割/轨迹预测上同时提升精度并降低等变误差。

## 研究背景与动机
**领域现状**：等变性（输入做某个变换、输出可预测地跟着变）是计算机视觉里很基础的归纳偏置，比如分割任务里物体平移、预测掩码也该平移。专门设计的群等变网络（群卷积、等变 Transformer/GNN）理论漂亮、在很多任务上有效，但在主流大模型里几乎没被采用。

**现有痛点**：真实数据往往只是"近似等变"，严格强加等变会牺牲模型表达力。于是出现了"软等变"路线，但现有做法都有硬伤——① 数据增广和正则化（在 loss 里惩罚等变误差）训练完**对最终的等变性没有任何保证**；② 往等变骨架里掺非等变分支（残差混合 RPP 等）虽然能权衡表达力与等变性，但**仍然不给等变保证**，且依赖专门的等变架构，没法直接套用 ViT/ResNet 这类现成大模型。

**核心矛盾**：表达力与等变性之间存在 trade-off，而既有方法要么不可控、要么不可证、要么不通用——你拿到一个预训练 ViT，没有办法"既保留它、又精确地给它注入可调且有界的等变性"。

**本文目标**：构造一种软等变层，要同时满足三点：(a) 能套在**任意预训练模型**上、不引入新参数；(b) 软硬程度**连续可调**；(c) 调到任何程度都有**等变误差的理论上界**。

**切入角度**：作者从 CNN 抗锯齿（anti-aliasing/blurring）能提升平移不变性这一特例出发（Zhang 2019）。信号处理告诉我们：低通滤波等价于"投影到带限子空间"，而带限子空间在平移下保持带限——所以模糊滤波本质是一个投影算子。作者把这一"投影=软等变"的观点从平移群推广到任意紧致连通李群。

**核心 idea**：用"广义模糊"——即把权重投影到由群作用决定的子空间——来实现软等变，投影保留多少个方向由截断阈值 $b$ 决定，$b$ 越小越偏等变，并能反过来给出等变误差的闭式界。

## 方法详解

### 整体框架
方法可以一句话概括：**给定一个群 $G$ 和一个预训练层，先离线算出该群李代数表示的奇异/特征结构，构造一个投影算子，训练时把层的可学习参数强制投影到这个子空间里**。整条流水线是"群结构 → 投影算子 → 套到任意线性层上"，不增加任何可学习参数，因此可无缝接进 ViT 的 patch embedding、位置编码、卷积、以及点特征的全连接层；ReLU 这类逐点非线性本身就等变，无需改动。

作者先把"软等变"重新定义成一个尺度无关的相对量（关键设计 1），然后在连续群上用李代数表示的 SVD 构造投影算子并证出误差界（关键设计 2），再针对旋转这类"正规"群用 Schur 分解把构造代价从 $O((d d')^3)$ 降到 $O(\max(d,d')^3)$（关键设计 3），最后用"群前向差分算子"把整套理论搬到离散群上（关键设计 4）。截断阈值 $b$ 是唯一的软硬旋钮，可按验证集调。

### 关键设计

**1. $\epsilon$-软等变：用雅可比归一化定义一个尺度无关、可解释的等变误差**

先前工作把软等变写成绝对误差约束 $\lVert F(\rho_X(g)x) - \rho_Y(g)F(x)\rVert \le \epsilon$，但这个量随输出 $F(x)$ 的尺度漂移，$\epsilon$ 大小本身没法解释。本文把它改成一个相对量：

$$\frac{\lVert F(\rho_X(g)x) - \rho_Y(g)F(x)\rVert}{\lVert J_F(x)\rVert_F \,\lVert x\rVert} \le \epsilon, \quad \forall g\in G,\ x\in X.$$

这里 $J_F(x)$ 是 $F$ 在 $x$ 处的雅可比，$\lVert J_F\rVert_F$ 表示局部的输出变化幅度。直觉上，分母把"违反量"除以"模型自己在该点的局部尺度"，于是 $\epsilon$ 衡量的是相对于模型自身输出变化的等变破坏程度，跨任务、跨模型都可比；$\rho_Y$ 取恒等时则退化为软不变。这个定义是后面所有误差界的落脚点，让"软到什么程度"有了一把可解释的尺子。

**2. 李代数投影算子：把权重限制到"群作用影响小"的子空间，并证出闭式误差界**

这是全文的核心。对一个全连接不变层 $y=w^\top x$，作者不直接学 $w$，而是令 $w = B_{\text{inv}}\theta$，$\theta$ 才是可学习参数，$B_{\text{inv}}$ 是固定的投影算子。它怎么来？先取群 $G$ 的李代数表示 $\bar A = d\rho_X(A)$（无穷小生成元，刻画恒等附近的一阶作用），做 SVD $\bar A = U\Sigma V^\top$，奇异值升序排列；然后只保留奇异值小于阈值 $b$ 的那些左奇异向量张成的子空间：

$$B_{\text{inv}} = \sum_{i:\,\sigma_i < b} u_i u_i^\top.$$

奇异值大的方向正是被群作用"搅动"得最厉害的方向，把它们滤掉，权重就只活在对群作用不敏感的子空间里——这正是"广义模糊"。误差界（Claim 1）给出 $\epsilon_b = b\sqrt{n_G}\, r_G + \delta_G$：$n_G$ 是生成元个数、$r_G$ 是单射半径（刻画群的"大小/复杂度"，如连续 2D 旋转 $r_G=\pi,\ n_G=1$），$\delta_G$ 是一阶泰勒展开的残差。界随 $b$ 线性增长，于是 $b$ 直接、可证地控制了软硬程度。对等变（而非不变）层，把约束通过 Kronecker 积合并成矩阵 $L = d\rho_X(A)^\top\!\otimes I - I\otimes d\rho_Y(A)$，对 $L$ 做 SVD 后用其右奇异向量构造 $B_{\text{eq}}$，对 $\text{vec}(W)=B_{\text{eq}}\theta$ 同样得到 $\epsilon_b = b\sqrt{n_G d'}\,r_G + \delta_G$（Claim 2）。多生成元情形则把各生成元的表示横向拼接后再取左奇异向量。关键好处：它不依赖任何特定架构，套在现成预训练权重上即可，且不新增参数。

**3. Schur 分解的高效实现：把旋转这类正规群的投影构造代价降一个量级**

对 $L$ 做 SVD 的复杂度是 $O((d\cdot d')^3)$，虽然每个群只需训练前算一次，但 $d\cdot d'$ 一大就吃不消（14×14 输入 SVD 要近 15 分钟）。作者注意到：当李代数表示是**正规矩阵**（与其共轭转置可交换，2D/3D 旋转就是）时，可以改用实 Schur 分解 $d\rho_X = U_X\Lambda_X U_X^\top$、$d\rho_Y = U_Y\Lambda_Y U_Y^\top$，$\Lambda$ 是由 $1\times1$ 或 $2\times2$ 块组成的块对角阵，复杂度降到 $O(\max(d,d')^3)$。先把权重换到 Schur 基 $\Delta' = U_Y^\top \Delta U_X$，再按块做投影 $B_{\text{Schur}}$：当两个块不共享特征值（$T_l\not\sim S_k$）且最大特征值之和超过 $b$ 时整块置零，共享特征值时用 $\text{Sym}(\cdot)$ 取对称化形式 $\bigl(\begin{smallmatrix}\frac{a+d}{2}&\frac{b-c}{2}\\ -\frac{b-c}{2}&\frac{a+d}{2}\end{smallmatrix}\bigr)$（来自 Sylvester 方程/Schur 引理给出的 $2\times2$ 等变解 $\bigl(\begin{smallmatrix}\alpha&\beta\\-\beta&\alpha\end{smallmatrix}\bigr)$），其余保留。Claim 3 证明这样构造仍满足同样形式的 $\epsilon_b$ 界。实测 14×14 时 Schur 从近 15 分钟降到不到 1 秒（见 Tab. 6），让框架在较大维度上真正可用。

**4. 离散群扩展：用群前向差分算子替代李代数表示**

前三个设计都建立在李群的泰勒展开和李代数表示上，离散群（如有限旋转群）没有李代数。作者引入"群前向差分算子"$\Delta_s f(g) = f(sg) - f(g)$ 作为李代数表示的离散类比（沿生成元 $s$ 的差分），并据此给出离散群上的一阶泰勒近似（Lemma 2）：$\hat f(g) = f(e) + \sum_i n_{s_i}\Delta_{s_i}f(e)$，逐点误差被字度量 $d_S$ 和 Lipschitz 常数 $h$ 界住，$\lvert f(g)-\hat f(g)\rvert \le 2h\cdot d_S(e,g)$。有了这个差分版泰勒，只要把前面公式里的 $d\rho$ 换成 $\Delta_s$，投影算子和误差界的构造就原样迁移到离散群。这一步让框架既能处理连续旋转，也能处理离散旋转/反射等。

### 损失函数 / 训练策略
本方法不改训练目标，只把每个被加固层的权重替换成"投影 × 可学习参数"，正常微调即可。唯一的旋钮是截断阈值 $b$：$b$ 越小越偏等变（保留更少方向）、越大越灵活但等变误差越大，作者把 $b$ 当超参按验证集调。投影既可用硬阈值，也可用平滑（soft）阈值，后者实测更好（见 Tab. 7）。

## 实验关键数据

### 主实验
在 ImageNet-1K 上微调三种骨架，Ours 在精度与不变误差上同时最优，且不像 canonicalizer 那样掉精度（iErr 单位 $\times10^{-2}$）：

| 骨架 | 方法 | Acc↑ | aAcc↑ | cAcc↑ | iErr↓ |
|------|------|------|-------|-------|-------|
| ViT | Base | 81.67 | 77.29 | 79.40 | 0.36 |
| ViT | Canon. | 76.51 (-5.16) | 75.81 | 76.15 | 0.15 |
| ViT | **Ours** | **82.28 (+0.61)** | **80.56** | **81.40** | **0.15** |
| DINOv2 | Base | 84.27 | 82.82 | 83.52 | 0.13 |
| DINOv2 | **Ours** | **85.31 (+1.04)** | **84.44** | **84.87** | **0.05** |
| ResNet-50 | Base | 77.91 | 75.12 | 76.48 | 0.24 |
| ResNet-50 | **Ours** | **77.96 (+0.06)** | **75.52** | **76.72** | **0.11** |

语义分割（PASCAL VOC，mIoU / 等变误差 eErr $\times10^{-2}$）也是同向改善：

| 骨架 | 方法 | mIoU↑ | aIoU↑ | cIoU↑ | eErr↓ |
|------|------|-------|-------|-------|-------|
| ViT | Base | 73.40 | 70.09 | 71.73 | 12.31 |
| ViT | Canon. | 65.36 (-8.03) | 61.93 | 63.62 | 20.39 |
| ViT | **Ours** | **74.78 (+1.38)** | **71.61** | **73.18** | **11.12** |
| SegFormer | **Ours** | **66.34 (+0.99)** | **62.52** | **64.40** | **10.64** |

人体轨迹预测（ETH/UCY，cADE/cFDE/eErr 越低越好）上，Ours 在 4/5 个场景拿到最佳 cADE/cFDE，且优于"完全等变"的 EqAuto——后者强等变反而精度更差，eErr 为 0 但预测变糟：

| 场景 | 方法 | cADE↓ | cFDE↓ | eErr↓ |
|------|------|-------|-------|-------|
| ETH | Base | 4.73 | 6.15 | 1.68 |
| ETH | EqAuto | 5.40 | 7.33 | 0.00 |
| ETH | **Ours** | **4.58** | 6.23 | 1.42 |
| ZARA1 | **Ours** | **3.40** | **4.67** | 0.39 |
| ZARA2 | **Ours** | **2.91** | **3.60** | 0.24 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| SVD（14×14） | ~890 s（≈15 分钟） | 构造投影算子，维度大时极慢 |
| Schur（14×14） | 0.25 s | 正规生成元下等价构造，快约 3500× |
| 硬阈值（ViT 分割） | mIoU 73.92 / cIoU 71.78 / eErr 11.74 | 直接截断 |
| 平滑阈值（ViT 分割） | mIoU 74.78 / cIoU 73.18 / eErr 11.12 | 软截断，精度更高、误差更低 |
| 可调软度（MNIST，vs RPP） | 全软度区间 iErr 更低、cAcc 更高 | 比残差混合 RPP 的 trade-off 更优 |

### 关键发现
- **ImageNet 上没有 trade-off**：精度与不变误差同时改善，这在软等变里不常见——通常增强等变会牺牲表达力，而投影到"群作用不敏感子空间"反而既稳又准。
- **强等变未必更好**：轨迹预测里完全等变的 EqAuto（eErr=0）精度反不如可调软等变，印证"真实数据只近似等变"的出发点。
- **Schur 是落地关键**：把构造代价降一个量级，使框架在较大维度可用；同时平滑阈值优于硬阈值，说明"渐变模糊"比"一刀切"更利于优化。
- **canonicalizer 的局限**：旋转会让角落像素移出视野，导致规范化网络预测漂移、无法达到零不变误差且大幅掉点（-5～-10），反衬本文方法的稳健。

## 亮点与洞察
- **把"抗锯齿提升不变性"上升为一般理论**：Zhang 2019 是经验观察，本文用"低通滤波=投影到带限子空间"的信号处理视角，把它推广到任意紧致连通李群，并给出数学解释，这是很漂亮的"经验→原理"升格。
- **架构无关 + 零新增参数**：直接把现成预训练 ViT/ResNet/DINOv2/SegFormer 改造成可控软等变，不像以往软等变必须从等变骨架长出来——这是真正能用在大模型上的工程意义。
- **可证误差界把"软"变成可控旋钮**：$\epsilon_b = b\sqrt{n_G}\,r_G+\delta_G$ 把抽象的"软硬程度"和一个标量阈值线性挂钩，给了实践者一个可解释、可调、可保证的接口。
- **可迁移的思路**：用"群作用的谱结构 → 选低响应方向 → 投影约束权重"这套范式，原则上可推广到点云、几何特征、甚至生成模型中需要近似对称性的场景。

## 局限性 / 可改进方向
- **一阶泰勒残差 $\delta_G$ 不可控**：误差界里 $\delta_G$ 来自一阶近似，论文未量化它有多大，群较大或变换角度大时界可能松（Tab. 3 也显示大角度旋转性能普遍下滑）。
- **依赖正规矩阵才高效**：Schur 加速只对李代数表示为正规矩阵的群（旋转等）成立，一般群仍要回退到昂贵的 SVD，限制了对更复杂群的可扩展性。
- **主要验证 2D 旋转**：分类/分割/轨迹实验基本聚焦 2D 旋转等变，更高维群（如 O(5)）只在合成回归上点到，更大规模、更复杂对称的实证还不充分。
- **改进方向**：自动按层/按任务选择 $b$（而非手调）、把投影从线性层推广到注意力等更复杂算子、以及给出更紧的高阶误差界。

## 相关工作与启发
- **vs 群等变架构（群卷积/等变 Transformer/GNN）**：它们强加严格等变、需专门设计；本文是软等变、架构无关、可套预训练模型，且把"严格 vs 不等变"做成连续谱。
- **vs 增广 / 正则化软等变**：那类方法训练后**对等变性无保证**；本文给出闭式误差界，软硬可证可控。
- **vs 残差混合 RPP/ResEq**：它们往等变骨架掺非等变分支、约把模型规模翻倍且仍无保证；本文 MNIST 实验显示在全软度区间 iErr 更低、cAcc 更高（Fig. 2），且不新增参数、不需等变骨架。
- **vs 规范化（canonicalizer）**：用等变网络预测并标准化输入再喂主干；本文无需额外预测、不受旋转边界效应拖累，精度与一致性都更稳。
- **vs Finzi et al.（Kronecker 等变约束）**：本文借用了 Kronecker 积合并等变约束的工具，但把它从"精确等变"推广到"软等变"，并对接现代预训练视觉模型、显式可控软度。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把"投影=软等变"从平移特例升格为带可证误差界的通用李群框架，架构无关且零新增参数，角度新。
- 实验充分度: ⭐⭐⭐⭐ 覆盖分类/分割/轨迹三任务、四骨架、ImageNet 级别且做了 Schur/阈值消融；但主要限于 2D 旋转、复杂群实证偏少。
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰、定义到三条 Claim 层层递进、图表呼应到位；公式密度高，对非等变背景读者门槛偏高。
- 价值: ⭐⭐⭐⭐⭐ 给出能直接套在大预训练模型上、可调可证的软等变接口，且在 ImageNet 上同时提精度降误差，实用意义强。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Soft Equivariance Regularization for Invariant Self-Supervised Learning](../../ICLR2026/self_supervised/soft_equivariance_regularization_for_invariant_self-supervised_learning.md)
- [\[CVPR 2025\] Few-Shot Implicit Function Generation via Equivariance](../../CVPR2025/self_supervised/few-shot_implicit_function_generation_via_equivariance.md)
- [\[NeurIPS 2025\] Soft Task-Aware Routing of Experts for Equivariant Representation Learning](../../NeurIPS2025/self_supervised/soft_task-aware_routing_of_experts_for_equivariant_representation_learning.md)
- [\[NeurIPS 2025\] Asymptotic and Finite-Time Guarantees for Langevin-Based Temperature Annealing in InfoNCE](../../NeurIPS2025/self_supervised/asymptotic_and_finite-time_guarantees_for_langevin-based_temperature_annealing_i.md)
- [\[CVPR 2026\] Learning from Semantic Dictionaries: Discriminative Codebook Contrastive Learning for Unified Visual Representation and Generation](learning_from_semantic_dictionaries_discriminative_codebook_contrastive_learning.md)

</div>

<!-- RELATED:END -->
