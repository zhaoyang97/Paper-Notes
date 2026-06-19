---
title: >-
  [论文解读] Hyperbolic Busemann Neural Networks
description: >-
  [CVPR 2026][计算生物][双曲神经网络] 利用 Busemann 函数将多类逻辑回归（MLR）和全连接层（FC）内蕴地提升到双曲空间，提出 BMLR 和 BFC 两个统一组件，在 Poincaré 球和 Lorentz 模型上同时适用，且在图像分类、基因组序列、节点分类、链接预测四类任务上均优于已有双曲层。
tags:
  - "CVPR 2026"
  - "计算生物"
  - "双曲神经网络"
  - "Busemann函数"
  - "双曲分类"
  - "全连接层"
  - "流形学习"
---

# Hyperbolic Busemann Neural Networks

**会议**: CVPR 2026  
**arXiv**: [2602.18858](https://arxiv.org/abs/2602.18858)  
**代码**: [有](https://github.com/GitZH-Chen/HBNN)  
**领域**: 计算生物  
**关键词**: 双曲神经网络, Busemann函数, 双曲分类, 全连接层, 流形学习  

## 一句话总结

利用 Busemann 函数将多类逻辑回归（MLR）和全连接层（FC）内蕴地提升到双曲空间，提出 BMLR 和 BFC 两个统一组件，在 Poincaré 球和 Lorentz 模型上同时适用，且在图像分类、基因组序列、节点分类、链接预测四类任务上均优于已有双曲层。

## 研究背景与动机

### 1. 领域现状

双曲空间因其指数级体积增长特性，能低失真地嵌入树状与层次结构数据，近年来在计算机视觉、图学习、多模态学习、推荐系统、基因组学、NLP 等领域取得广泛成功。为支撑双曲深度学习，MLR（多类逻辑回归）和 FC（全连接层）这两个核心组件已被多次推广到 Poincaré 球和 Lorentz 模型。

### 2. 痛点

现有双曲 MLR 和 FC 层存在若干共性问题：

- **过度参数化**：Ganea 等人的 Poincaré MLR 每类需额外的流形参数 $p_k \in \mathbb{P}_K^n$，参数量翻倍
- **批计算效率低**：部分方法（如 PBMLR-P）需逐类循环计算，无法高效矩阵化
- **模型特异性**：Poincaré FC 仅适用于 Poincaré 模型，Lorentz FC 仅适用于 Lorentz 模型，缺乏统一框架
- **几何失真**：Möbius FC 和 Lorentz FC 在切空间或环境 Minkowski 空间做欧式变换后投影，扭曲了内蕴几何

### 3. 核心矛盾

实践需要一个**内蕴、高效、统一**的双曲 MLR/FC 层，但已有方案要么不内蕴（依赖切空间/环境空间近似）、要么不高效（过度参数化/不支持批处理）、要么不统一（绑定单一模型）。

### 4. 要解决什么

在 Poincaré 和 Lorentz 两大双曲模型上提供统一的、参数紧凑的、批高效的 MLR 和 FC 层，且保留真实的几何距离解释。

### 5. 切入角度

**Busemann 函数**——双曲空间中内积的内蕴推广。欧式内积 $\langle v, x \rangle$ 在双曲空间的对应是 Busemann 函数 $-B^v(x)$；欧式超平面的对应是**horosphere（极球面）**。这对概念在 Poincaré 和 Lorentz 模型上均有解析闭式。

### 6. 核心 idea

用 Busemann 函数直接替换欧式 MLR/FC 中的内积运算，得到 **BMLR**（Busemann MLR）和 **BFC**（Busemann FC），一套公式同时覆盖两种双曲模型，且当曲率 $K \to 0^-$ 时自然退化回欧式对应物。

## 方法详解

### 整体框架

这篇论文想解决的是：双曲神经网络里最基础的两个零件——分类头（MLR）和全连接层（FC）——一直缺一个既内蕴、又紧凑、还能跨 Poincaré 和 Lorentz 两种模型通用的写法。作者的破题点是一个观察：欧式空间里这两个零件本质都建立在「内积 $\langle v, x\rangle$」和「超平面」之上，而内积在双曲空间的内蕴对应物正是 **Busemann 函数** $-B^v(x)$，超平面的对应物则是 **horosphere（极球面）**。于是只要把欧式公式里的内积换成 Busemann 函数、把超平面换成 horosphere，就能整体平移到双曲空间。

按这个思路落成两个组件：**BMLR** 接在网络末端做分类，把欧式 logit $u_k(x)=\langle a_k,x\rangle+b_k$ 改写成基于 Busemann 函数的形式；**BFC** 替换中间的全连接层，把欧式 FC「输出第 $k$ 维 = 到坐标超平面的有符号距离」这一等式的右端换成 Busemann logit，再反解出双曲点 $y$。两者共用同一套「内积→Busemann、超平面→horosphere」的字典，所以一套公式同时覆盖两种模型，且曲率 $K\to 0^-$ 时自动退回欧式版本。

### 关键设计

**1. Busemann MLR：用 Busemann 函数替掉分类头里的内积**

欧式 MLR 的 logit $u_k(x)=\alpha_k\langle v_k,x\rangle+b_k$ 里那个内积，在双曲空间没有现成对应物，过去的双曲 MLR 只能靠每类额外挂一个流形参数 $p_k$ 来补偿，参数翻倍。这里换个角度：欧式空间下 $B^v(x)=-\langle x,v\rangle$，也就是 Busemann 函数本就是内积的负数，那直接把内积替成 Busemann 函数就行，得到

$$u_k(x) = -\alpha_k B^{v_k}(x) + b_k$$

其中 $\alpha_k>0$、$v_k\in\mathbb{S}^{n-1}$、$b_k\in\mathbb{R}$，每类只需 $(\alpha_k,v_k,b_k)$ 共 $C(n+2)$ 个参数，不再有额外的流形值参数。$B^v(x)$ 在两种模型上都有闭式：Poincaré 球上 $B^v(x)=\frac{1}{\sqrt{-K}}\log\frac{\|v-\sqrt{-K}x\|^2}{1+K\|x\|^2}$，Lorentz 模型上 $B^v(x)=\frac{1}{\sqrt{-K}}\log(\sqrt{-K}(x_t-\langle x_s,v\rangle))$。这两个表达式都能整批矩阵化，所有类的 logit 一次算完，不像 PBMLR-P 那样要逐类循环。而且 $K\to 0^-$ 时 Poincaré 版趋于 $2\alpha_k\langle v_k,x\rangle+b_k$、Lorentz 版趋于 $\alpha_k\langle v_k,x_s\rangle+b_k$，干净地退化回欧式 MLR，保证了它是欧式情形的真推广而非另起炉灶。

**2. 点到极球面的距离解释：让这个 logit 有几何含义**

把内积换成 Busemann 函数后，自然要问换出来的 logit 到底代表什么。答案是它就是「点到 horosphere 的有符号测地距离」。关键依据是 Hadamard 空间（涵盖欧式与双曲的广义度量空间）里 Busemann 函数的等值面——也就是 horosphere——彼此等距：$d(H_{\tau_1}^\gamma,H_{\tau_2}^\gamma)=|\tau_2-\tau_1|$，于是点到任意 horosphere 的距离写成 $d(x,H_\tau^v)=|B^v(x)-\tau|$。把这个代回去，BMLR 的 logit 恰好等于有符号点到 horosphere 距离再乘 $\alpha_k$。这就把欧式 MLR 里「logit = 点到决策超平面的有符号距离」（Lebanon & Lafferty 的经典解释）原封不动搬到了双曲空间：样本离哪一类的 horosphere 越近，属于那类的概率越大。区别在于它用的是真实测地距离，而不是切空间投影出来的伪距离。

**3. Busemann FC：把同一套字典用到全连接层**

全连接层比分类头多一步——它要输出一个新的点，而不只是一个标量 logit。作者先把欧式 FC 重写成距离等式 $\bar{d}(y,H_{e_k,0})=\langle a_k,x\rangle+b_k$，即输出 $y$ 的第 $k$ 维等于它到第 $k$ 个坐标超平面的有符号距离。然后两端各替一次：右端的内积换成 Busemann logit $u_k(x)$，左端的欧式距离换成双曲点到超平面距离，得到隐式方程 $\bar{d}(y,H_{e_k,e})=u_k(x)$，再反解出 $y$。两种模型都有闭式解：Poincaré 上 $y=\omega/(1+\sqrt{1-K\|\omega\|^2})$，其中 $\omega_k=\sinh(\sqrt{-K}\,u_k(x))/\sqrt{-K}$；Lorentz 上 $y_s=\sinh(\sqrt{-K}\,u(x))/\sqrt{-K}$、$y_t=\sqrt{1/(-K)+\|y_s\|^2}$。整个过程始终在双曲流形上完成，不经切空间或环境 Minkowski 空间的欧式近似，所以不引入 Möbius FC / Lorentz FC 那类几何失真。这个写法还很好扩展：把 $u_k(x)$ 套个激活 $\phi(-\alpha_k B^{v_k}(x)+b_k)$ 就能加非线性，也可以再附一个 gyroaddition 偏置；开销上 FLOPs 为 $O(nm)$，与已有方法持平，Lorentz 版本约 $O(2nm)$。

### 损失函数 / 训练策略

- **分类任务**（BMLR）：标准交叉熵损失
- **链接预测**（BFC）：Fermi-Dirac 解码器配合交叉熵，按 HGCN 原始设置
- **参数约束**：$v_k$ 需保持单位球约束 $v_k \in \mathbb{S}^{n-1}$，通过归一化实现；$\alpha_k > 0$ 通过 softplus 保证
- **曲率**：各任务曲率 $K$ 作为可学习参数或交叉验证选取
- **特征映射**：混合架构中，欧式 backbone 输出通过指数映射投射到双曲空间后再接 BMLR/BFC

## 实验关键数据

### 主实验

**表1：图像分类准确率（ResNet-18 backbone，Top-1 %）**

| 空间 | 方法 | CIFAR-10 (10类) | CIFAR-100 (100类) | Tiny-ImageNet (200类) | ImageNet-1k (1000类) |
|------|------|-----------------|--------------------|-----------------------|----------------------|
| $\mathbb{R}^n$ | MLR | 95.14 | 77.72 | 65.19 | 71.87 |
| $\mathbb{P}_K^n$ | PMLR | 95.04 | 77.19 | 64.93 | 71.77 |
| $\mathbb{P}_K^n$ | PBMLR-P | 95.23 | 77.78 | 65.43 | 71.46 |
| $\mathbb{P}_K^n$ | **BMLR-P** | **95.32** | **78.10** | **66.16** | **73.36** |
| $\mathbb{L}_K^n$ | LMLR | 94.98 | 78.03 | 65.63 | 72.46 |
| $\mathbb{L}_K^n$ | **BMLR-L** | **95.25** | **78.07** | **65.99** | **73.24** |

**关键发现**：BMLR 相对已有双曲 MLR 的优势随类别数增大而增大——在 ImageNet-1k（1000类）上 BMLR-P 比 PMLR 高 **1.59%**，比 PBMLR-P 高 **1.90%**。PBMLR-P 参数量为其他方法两倍且训练速度最慢。

**表2：节点分类 F1（HGCN backbone）与链接预测 AUC**

| 空间 | 方法 | Disease (δ=0) | Airport (δ=1) | PubMed (δ=3.5) | Cora (δ=11) |
|------|------|---------------|---------------|-----------------|-------------|
| **节点分类 F1** | | | | | |
| $\mathbb{P}_K^n$ | HGCN (tangent) | 86.87 | 85.34 | 76.29 | 76.56 |
| $\mathbb{P}_K^n$ | HGCN-BMLR-P | **92.45** | **86.02** | **77.36** | **78.48** |
| $\mathbb{L}_K^n$ | HGCN-LMLR | 89.72 | 82.61 | 75.44 | 69.91 |
| $\mathbb{L}_K^n$ | HGCN-BMLR-L | **90.80** | **85.27** | **77.30** | **77.65** |
| **链接预测 AUC** | | | | | |
| $\mathbb{P}_K^n$ | Poincaré FC | 79.45 | 94.31 | 94.24 | 88.21 |
| $\mathbb{P}_K^n$ | BFC-P | **80.45** | **94.88** | **94.85** | **91.94** |
| $\mathbb{L}_K^n$ | Lorentz FC | 72.78 | 92.99 | 94.20 | 92.06 |
| $\mathbb{L}_K^n$ | BFC-L | **78.36** | **95.37** | **94.90** | **92.28** |

### 消融实验

- **类别数效应**：从 CIFAR-10（10类）到 ImageNet-1k（1000类），BMLR 的优势从 ~0.2% 扩大到 ~1.6%，说明 Busemann 函数在高维分类上的表达能力优势
- **双曲度效应**：在节点分类中，LMLR 在 Cora（$\delta=11$，最不双曲）上严重退化（69.91 vs tangent 的 77.37），但 BMLR-L 依然保持 77.65，显示出对图双曲度的鲁棒性
- **链接预测中 Disease（$\delta=0$，最双曲）**：BFC-L 比 Lorentz FC 高 5.58%，在最双曲的数据上 Busemann 几何优势最大

### 关键发现

1. **类别数越多优势越大**：BMLR 在 1000 类的 ImageNet-1k 上比 PMLR 高 1.59%，比 LMLR 高 0.78%
2. **训练速度最快**：Lorentz BMLR 在所有双曲 MLR 中 FLOPs 最低，fit time 最短；PBMLR-P 因不支持批计算，在 16 个基因组数据集上稳定最慢
3. **几何越双曲增益越大**：链接预测中 Disease（$\delta=0$）上 BFC-L 比 Lorentz FC 高 5.58%，但在较平坦的 Cora（$\delta=11$）上差距缩小到 0.22%
4. **鲁棒性**：已有双曲 MLR 在非双曲图上可能不如 tangent baseline（如 LMLR 在 Cora 上大幅退化），BMLR 在所有 $\delta$ 下均为最佳

## 亮点与洞察

- **数学优雅**：用 Busemann 函数统一了欧式内积 → 双曲空间的推广，一个公式同时涵盖 Poincaré 和 Lorentz 两种模型
- **理论完整**：证明了 Hadamard 空间中极球面等距性质（Thm 3.3），给出了 BMLR 的点到 horosphere 距离解释，以及 $K \to 0^-$ 的极限定理
- **实用性强**：BMLR-L 的 FLOPs 为 $C(2n+12)$，接近欧式 MLR 的 $C(2n)$，几乎零额外开销
- **跨领域验证**：四类任务（视觉、基因组、图节点分类、图链接预测）覆盖面广，说明方法的通用性

## 局限与展望

1. **仅覆盖 MLR 和 FC**：注意力、归一化、残差等其他网络组件未用 Busemann 函数重构，是否能构建完整的 Busemann 网络？
2. **曲率固定或手工选择**：虽提到可学习曲率，但实验中主要通过交叉验证选取，自适应曲率学习有待探索
3. **仅限常曲率空间**：真实数据可能具有变曲率结构（如乘积空间 $\mathbb{H} \times \mathbb{E}$），Busemann 函数在混合曲率空间的推广值得研究
4. **大规模 GNN 实验不足**：图学习实验仅用小规模数据集（最大 PubMed ~20K节点），在百万级图上的表现未验证

## 相关工作与启发

- **承接自**：Ganea et al. (NeurIPS'18) Poincaré MLR/FC → Shimizu et al. (NeurIPS'21) 重参数化 → Bdeir et al. (ICLR'24) Lorentz MLR/CNN
- **Busemann 函数在 ML 中的应用**：Fan et al. 双曲 SVM、Chami et al. 双曲 PCA、Bonet et al. Sliced-Wasserstein
- **启发**：Busemann 函数作为"内蕴内积"的角色可类推到其他 Hadamard 流形（如 SPD 矩阵空间），为设计更通用的流形神经网络组件提供模板

## 评分

- ⭐⭐⭐⭐ 新颖性：以 Busemann 函数为工具统一构建双曲 MLR 和 FC 层，数学动机清晰、理论框架优雅，但核心思路是已有工具的组合应用
- ⭐⭐⭐⭐ 实验充分度：覆盖 4 类任务 20+ 数据集、两种双曲模型的系统对比，含效率分析；但图实验仅用经典小规模数据集，缺少 OGB 等大规模基准
- ⭐⭐⭐⭐⭐ 写作质量：定理-证明结构严谨，对比表格清晰全面，欧式-双曲类比的叙述脉络流畅易懂
- ⭐⭐⭐⭐ 实用性：代码已开源，BMLR/BFC 即插即用，Lorentz BMLR 速度接近欧式 MLR，实际部署门槛低

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Hyperbolic Neural Population Geometry Benefits Computation](../../ICML2026/computational_biology/hyperbolic_neural_population_geometry_benefits_computation.md)
- [\[CVPR 2026\] HyperST: Hierarchical Hyperbolic Learning for Spatial Transcriptomics Prediction](hyperst_hierarchical_hyperbolic_learning_for_spatial_transcriptomics_prediction.md)
- [\[NeurIPS 2025\] Random Search Neural Networks for Efficient and Expressive Graph Learning](../../NeurIPS2025/computational_biology/random_search_neural_networks_for_efficient_and_expressive_graph_learning.md)
- [\[CVPR 2026\] CryoKRAQEN: Kernel-Regularized Annealing for Quantized Embedding Networks in Cryo-EM Heterogeneous Reconstruction](cryokraqen_kernel-regularized_annealing_for_quantized_embedding_networks_in_cryo.md)
- [\[CVPR 2026\] Cell-Type Prototype-Informed Neural Network for Gene Expression Estimation from Pathology Images](cell-type_prototype-informed_neural_network_for_gene_expression_estimation_from_.md)

</div>

<!-- RELATED:END -->
