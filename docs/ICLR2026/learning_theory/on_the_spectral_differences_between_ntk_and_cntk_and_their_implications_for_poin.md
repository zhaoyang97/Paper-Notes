---
title: >-
  [论文解读] On the Spectral Differences Between NTK and CNTK and Their Implications for Point Cloud Recognition
description: >-
  [ICLR 2026][学习理论][NTK] 本文在任意维度的张量数据假设下，给出 NTK 与 CNTK 两个**与数据分布无关**的谱差异定理（NTK 特征值均值更大、谱更集中），据此定义衡量数据"是否适合卷积"的指标"卷积适配度"，并由此推断点云比图像更依赖卷积结构，最终用 CNTK-NTK 混合核（PointNTK）在小样本点云识别上显著超过 NTK 基线。
tags:
  - "ICLR 2026"
  - "学习理论"
  - "神经正切核"
  - "NTK"
  - "CNTK"
  - "谱分析"
  - "卷积适配度"
  - "点云识别"
---

# On the Spectral Differences Between NTK and CNTK and Their Implications for Point Cloud Recognition

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=U6SnDgI3gG](https://openreview.net/forum?id=U6SnDgI3gG)  
**代码**: 待确认  
**领域**: 学习理论 / 神经正切核  
**关键词**: NTK, CNTK, 谱分析, 卷积适配度, 点云识别

## 一句话总结
本文在任意维度的张量数据假设下，给出 NTK 与 CNTK 两个**与数据分布无关**的谱差异定理（NTK 特征值均值更大、谱更集中），据此定义衡量数据"是否适合卷积"的指标"卷积适配度"，并由此推断点云比图像更依赖卷积结构，最终用 CNTK-NTK 混合核（PointNTK）在小样本点云识别上显著超过 NTK 基线。

## 研究背景与动机

**领域现状**：神经正切核（NTK）把无限宽神经网络的训练动力学转化为一个固定核，使我们能用核回归的视角精确分析网络的归纳偏置与收敛性。其卷积变体 CNTK 则对应无限宽 CNN，已有不少工作研究它作为核的性质并向各方向扩展。

**现有痛点**：尽管 NTK 与 CNTK 各自被研究得很透，但**在同一份数据上两者的谱（特征值分布）到底差在哪里**，却几乎没有被系统讨论。更麻烦的是，已有谱分析大多假设数据均匀分布在高维球面上，完全忽略了真实数据的**张量结构**（图像是 2D 张量、点云是 1D 序列），导致结论难以解释"为什么卷积网络泛化更好""为什么深层 CNN 后面要接几层 MLP"这类经验现象。

**核心矛盾**：卷积结构带来的归纳偏置体现在核矩阵的谱上，但缺少一个**不依赖具体数据分布**的、能直接对比 NTK 与 CNTK 的谱量度；同时也缺一个能判断"某份数据到底有多适合卷积"的可计算指标。

**本文目标**：(1) 在一般维度张量数据假设下，刻画 NTK 与 CNTK 的谱差异；(2) 把谱差异转成一个数据层面的"卷积适配度"指标；(3) 用这个理论解释并指导点云识别——一个此前几乎没人用核回归碰过的领域。

**切入角度**：作者放弃球面均匀假设，转而假设 $d{+}1$ 维数据落在 $d$ 维张量空间上、服从协方差矩阵 $\sigma_H$ 的任意分布（协方差一般非对角），并把 CNTK 推广到**任意卷积维度**（$d=0$ 时退化回 NTK）。这样 NTK 与 CNTK 就被放进同一框架，谱的差异可以被协方差矩阵直接刻画。

**核心 idea**：用三个谱统计量（均值 $m_K$、平方均值 $s_K$、离散度 $\beta_K$）作为 NTK/CNTK 的"统一标尺"，证明 $\beta_K$ 对任意层 CNTK 恒等于 1、对 NTK 恒小于 1，从而 $1-\beta^{(0)}_{\mathrm{NTK}}$ 天然就是"数据有多适合卷积"的度量。

## 方法详解

### 整体框架
本文不是一个"网络 pipeline"，而是一条**理论推导链**：先把 CNTK 推广到任意维度并与 NTK 统一（一般维 CNTK 递推）→ 定义三个谱统计量 $m_K, s_K, \beta_K$ → 证明 NTK 与 CNTK 在这三个量上的分布无关差异（两个谱定理）→ 用首层谱差 $1-\beta^{(0)}_{\mathrm{NTK}}$ 定义"卷积适配度"，并据此判断点云比图像更吃卷积 → 给出 CNTK 接 NTK 的闭式混合核，落地为 PointNTK 做点云核回归。

整条链的逻辑是"谱性质决定泛化、谱性质由数据协方差决定、协方差结构在点云上更接近对角"，因此理论结论一路推到"点云应当用卷积核回归"的实践建议。由于核心是矩阵谱与协方差的运算，不存在可图示的数据流分支，这里不画框架图，用公式把每一步讲清。

### 关键设计

**1. 一般维度 CNTK 递推：把 NTK 与 CNTK 统一进同一框架**

要比较 NTK 与 CNTK，第一步是让它们"可比"——本文把通常只写到 2D 的 CNTK 推广到任意卷积维度 $d$。对张量样本 $x, x' \in \mathbb{R}^{\prod_{i=1}^{d} h_i \times n_0}$，在位置 $p$ 及其卷积邻域 $N(p)$ 上，逐层维护协方差 $\Lambda^{(l)}_{p,q}$、零阶/一阶期望 $\Sigma^{(l)}_{p,q}, \dot\Sigma^{(l)}_{p,q}$，张量核的递推为

$$K^{(l)}_{p,q}(x,x') = \mathrm{tr}\!\left(K^{(l-1)}_{N(p),N(q)} \odot \dot\Sigma^{(l-1)}_{N(p),N(q)}(x,x')\right) + E^{(l-1)}_{p,q}(x,x'),$$

其中 $\odot$ 为 Schur（逐元素）积，递推基底 $\Sigma^{*(0)}_{p,q}(x,x') = x \otimes x'$ 用 Kronecker 积刻画张量内部元素的相互作用。带池化时核为 $K_{wp}=P_{av}(K^{(L)})$，不带池化时 $K_{wop}=\mathrm{tr}(K^{(L)})$。关键之处在于：**当 $d=0$ 时 $p=q$ 固定、$N(p)=p$，递推精确退化回标准 NTK**。这一退化关系正是后面所有"NTK vs CNTK"对比的公共地基——两者不再是两套独立公式，而是同一递推在不同卷积维度下的特例。

**2. 三个谱统计量与两个分布无关的谱差异定理：解释卷积为何泛化更好**

针对"NTK 与 CNTK 谱差在哪"，本文先定义三个可计算的谱统计量：

$$m_K = \tfrac{1}{N}\mathrm{tr}(K),\quad s_K = \tfrac{1}{N^2}\mathrm{tr}(KK^\top),\quad \beta_K = \frac{s_K}{m_K^2}.$$

$m_K$ 是特征值均值（反映收敛速度），$s_K$ 是特征值平方均值，$\beta_K$ 刻画特征值的离散程度——且 $1/\beta_K$ 恰好等同于 Bartlett 等人定义的 effective rank，可上界核回归的泛化误差。在此基础上给出两条**不需要任何具体数据分布假设**的结论：

- **定理 2（均值差）**：当 $N\to\infty$，$m^{(L)}_{\mathrm{NTK}} \ge m^{(L)}_{\mathrm{CNTK}}$，即 NTK 平均特征值更大。证明里用到更强的逐点结论 $K^{(L)}_{\mathrm{NTK}}(x,x)\ge K^{(L)}_{\mathrm{CNTK}}(x,x)$，且只要激活函数的对偶函数对对角项非凸即可（不限于 ReLU）。
- **定理 3（离散度差）**：对任意层 $L,L'$，$\beta^{(L)}_{\mathrm{NTK}} \le \beta^{(L')}_{\mathrm{CNTK}} = 1$。即 **CNTK 的 $\beta_K$ 对任意层恒为 1**，而 NTK 每层都严格小于 1——这仅来自随机初始化下协方差矩阵的半正定性，普适成立。

两条定理把经验现象讲圆了：$\beta_K$ 越小说明核矩阵越"对角化"、泛化越差，NTK 的 $\beta$ 恒小于 CNTK，所以**卷积网络泛化优于 MLP**；而 NTK 的 $m_K$ 更大、对应更快收敛，这正解释了**为什么深层卷积网络后接几层全连接能加速训练又不损泛化**。

**3. 卷积适配度（Convolutional Suitability）：一个只看数据协方差的可计算指标**

定理给的是"CNTK 总比 NTK 谱更散"，但不同数据散的程度不同——本文用**首层谱差**把它量化成数据指标。命题 4 给出首层（$L{=}0$）的闭式：

$$\beta^{(0)}_{\mathrm{NTK}} = \frac{\mathrm{mean}(\sigma_H \odot \sigma_H)}{\mathrm{mean}(\mathrm{diag}(\sigma_H)\otimes\mathrm{diag}(\sigma_H))},\qquad \beta^{(0)}_{\mathrm{CNTK}}=1,$$

其中 $\sigma_H$ 是张量输入的协方差矩阵。由于 CNTK 的 $\beta$ 恒为 1，作者把

$$\text{卷积适配度} = \beta^{(L)}_{\mathrm{CNTK}} - \beta^{(0)}_{\mathrm{NTK}} = 1 - \beta^{(0)}_{\mathrm{NTK}}$$

定义为"数据有多适合卷积网络"的度量：$\beta^{(0)}_{\mathrm{NTK}}$ 越小（数据协方差越接近对角），适配度越高。这个指标只依赖数据协方差、无需训练任何网络，等号当且仅当 $\sigma_H$ 为常数矩阵时成立。把它套到真实数据上得到关键判断：**点云因为点的无序性，样本间相关弱、协方差近对角、$\beta_K$ 更小，所以比图像更适合卷积结构**（Table 1 中 ModelNet 系列 $\beta_K\!\sim\!10^{-3}$，而 CIFAR10 达 $0.137$）。

**4. CNTK-NTK 混合核与 PointNTK：把理论落到点云核回归**

理论说"卷积后接 MLP 兼得 CNTK 的宽谱(II)和 NTK 的大偏移(I)"，本文据此推导**卷积接全连接**这一实用架构的闭式核。命题 1 给出 $d_1$ 维卷积子网 $f_1$ 接 $d_2$ 维子网 $f_2$（经局部平均池化 $P^{d_1\to d_2}_{av}$ 连接）的组合核 $K_c = P_{av}(K^{(L_1+L_2)})$，连接点处 $\Sigma^{(L_1+1)} = P^{d_1\to d_2}_{av}(\Sigma^{(L_1)})$；当 $d_2=0$（接 MLP）即得 **CNTK-NTK**。

落地到点云时，作者注意到 PointNet 的 shared-MLP 本质上就是**核大小为 1 的一维卷积**，于是完全照搬 PointNet 架构得到 **PointNTK**：取 $L_1{=}4$ 层 1D 卷积、$L_2{=}3$ 层 MLP、$d_1{=}1, d_2{=}0$。这是 CNTK 第一次被用到点云任务上，且核回归在输入固定时结果**确定且唯一**（无随机性）。

### 损失函数 / 训练策略
核回归方法本身无需训练，给定输入即得闭式解。对照组 PointNetm 把训练损失换成最小二乘、把 max pooling 换成 average pooling 以贴近核回归设定。消融用固定 500 样本训练、全测试集评估；小样本实验固定 100 个覆盖全类别的样本。实现基于 Python 3.7 + PyTorch 1.8，单张 RTX 3090，不做数据增强与后处理（仅全局归一化）。

## 实验关键数据

### 主实验
核回归直接对比：点云用 1dCNTK、图像用 2dCNTK，统一 $L=7$。CNTK 在点云上对 NTK 的提升远大于图像，印证"点云更吃卷积"。

| 数据集 | NTK | CNTK | 提升 | $\beta_K$ |
|--------|-----|------|------|-----------|
| ModelNet103 | 17.58 | 76.65 | +59.07 | 1.25e-3 |
| ModelNet106 | 16.19 | 91.96 | +75.77 | 1.12e-3 |
| ModelNet40 | 11.10 | 60.62 | +49.52 | 1.53e-3 |
| MNIST | 94.00 | 96.00 | +2.00 | 3.38e-2 |
| CIFAR10 | 59.19 | 76.79 | +17.60 | 1.37e-1 |

PointNTK vs PointNet（节选，$f$ 表示仅用 100 样本训练）：

| 方法 | ModelNet103 | ModelNet10$^f_3$ | ModelNet106 | ModelNet10$^f_6$ | ModelNet40 | ModelNet40$^f$ |
|------|------|------|------|------|------|------|
| PointNet | 91.12 | 68.92 | 92.08 | 70.22 | 88.71 | 38.32 |
| NTK | 17.58 | 11.12 | 16.19 | 11.78 | 11.10 | 3.69 |
| 1dCNTK | 76.65 | 71.03 | 91.96 | 81.16 | 60.62 | 45.22 |
| PointNTK | 86.56 | 73.68 | 91.19 | 78.52 | 80.47 | 45.71 |

关键现象：PointNTK 全面优于同层 1dCNTK；在**充分数据**下略逊于训练式 PointNet，但在**所有数据集的小样本（100 样本）设定下都反超** PointNet/PointNetm。

### 消融实验
固定 500 样本，分别消融网络深度与卷积核大小。

| 配置 | 结论 |
|------|------|
| 仅加深 1D 卷积（1dCNTK/PNTK1） | 收益甚微甚至变差 |
| 固定卷积、加深 FC（PNTK0） | 回归性能下降 |
| 卷积加深 + 末端接 FC | 全数据集提升，最有效 |
| 两部分同时加深（PNTK01） | 不如只加深 1D 卷积 |
| 核大小 1 vs >1 | 核大小 1 最优 |

### 关键发现
- **末端 MLP 是关键**：单纯加深一维卷积几乎无益，但"加深卷积 + 末端接全连接"在所有数据集都涨点——这正是定理(I)(II)预测的"宽谱 + 大偏移"组合，也解释了 PointNet 为何要在 pooling 后接 MLP。
- **核大小必须为 1**：点云无序，核大小超过 1 时被卷进同一核的点之间没有有意义的空间关系，因此 size=1（即 shared-MLP）最优。
- **小样本是 CNTK 的主场**：核回归无需训练、结果确定，在 100 样本极小数据下稳定超过训练式 PointNet，验证了"低数据场景下卷积核回归更优"的论断。

## 亮点与洞察
- **分布无关的谱定理**：$\beta^{(L')}_{\mathrm{CNTK}}\equiv 1$、$\beta^{(L)}_{\mathrm{NTK}}<1$ 仅靠协方差半正定就成立，不需要任何数据分布假设，结论非常干净且普适。
- **把"适合卷积"变成可算的数 $1-\beta^{(0)}_{\mathrm{NTK}}$**：只看数据协方差就能预判"这份数据用卷积还是 MLP"，且与 effective rank/泛化界挂钩——这是可迁移到任意张量数据的诊断工具。
- **理论直接指导架构**：从"宽谱(II)+大偏移(I)"推出"卷积后接 MLP"，再认出 PointNet 的 shared-MLP 就是 size-1 1D 卷积，把抽象谱结论落成 PointNTK，理论与实践闭环。
- **首次把 CNTK 用于点云核回归**，填补了核方法在点云领域的空白。

## 局限与展望
- 作者承认：主定理虽分布无关，但**计算任意 $L$ 层 NTK 的 $\beta_K$ 仍需分布假设**，因此实践中只能退而用首层（$L{=}0$）评估适配度，多层谱的上下界仍是开放问题。
- 充分数据下 PointNTK 仍**逊于训练式 PointNet**（尤其 ModelNet40），核回归的优势集中在小样本，方法的适用边界较窄。
- "卷积适配度"建立在"数据服从高斯分布、协方差能良好拟合"的前提上，对强非高斯/重尾数据指标是否仍有判别力未充分验证。
- 数据集仅限 ModelNet10/40 的点云与 CIFAR/MNIST 的图像，未在更大规模或更复杂点云任务上检验。

## 相关工作与启发
- **vs 标准 NTK（Jacot et al.）/ CNTK（Arora et al.）**: 前人分别给出 NTK、CNTK 的递推与性质，但未在同一数据上系统对比两者的谱；本文把 CNTK 推广到任意维并证明 $d=0$ 退化回 NTK，第一次给出两者分布无关的谱差异定理。
- **vs Bartlett et al.（benign overfitting / effective rank）**: 借用其 $1/\beta_K =$ effective rank 上界泛化误差的结论，但把它从"分析工具"转成"数据适配度指标"并用于架构选择。
- **vs PointNet（Qi et al.）**: PointNet 是经验性提出 shared-MLP + pooling；本文从核理论解释 shared-MLP 等价 size-1 1D 卷积、pooling 后接 MLP 的合理性，并给出其无限宽对应核 PointNTK，在小样本上反超 PointNet。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 分布无关谱定理 + 卷积适配度 + CNTK 首用于点云，理论与应用都新。
- 实验充分度: ⭐⭐⭐⭐ 谱直方图、核回归、深度/核大小消融较完整，但数据集与规模有限。
- 写作质量: ⭐⭐⭐ 理论密度高、记号繁，部分命题表述偏紧凑，初读门槛较高。
- 价值: ⭐⭐⭐⭐ "卷积适配度"作为数据诊断工具具备较强可迁移性，对小样本结构化数据建模有指导意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Sublinear Spectral Clustering Oracle with Little Memory](sublinear_spectral_clustering_oracle_with_little_memory.md)
- [\[ICLR 2026\] Provable Separations between Memorization and Generalization in Diffusion Models](provable_separations_between_memorization_and_generalization_in_diffusion_models.md)
- [\[ICLR 2026\] From Neural Networks to Logical Theories: The Correspondence between Fibring Modal Logics and Fibring Neural Networks](from_neural_networks_to_logical_theories_the_correspondence_between_fibring_moda.md)
- [\[ICLR 2026\] Training-Free Determination of Network Width via Neural Tangent Kernel](training-free_determination_of_network_width_via_neural_tangent_kernel.md)
- [\[ICLR 2026\] From Predictors to Samplers via the Training Trajectory](from_predictors_to_samplers_via_the_training_trajectory.md)

</div>

<!-- RELATED:END -->
