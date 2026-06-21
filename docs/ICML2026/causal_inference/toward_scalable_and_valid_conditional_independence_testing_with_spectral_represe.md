---
title: >-
  [论文解读] Toward Scalable and Valid Conditional Independence Testing with Spectral Representations
description: >-
  [ICML2026][因果推理][条件独立检验] SpectralCIT 把核方法里刻画条件独立的「偏协方差算子」用神经网络学到的低维谱特征来近似，再用一个形如 HSIC 的简单统计量做条件独立检验——它用一个双层对比算法学出算子的领先奇异特征，证明零假设下统计量渐近服从卡方分布、备择假设下有功效保证，从而把核方法的理论扎实性和现代表示学习的可扩展性接上。
tags:
  - "ICML2026"
  - "因果推理"
  - "条件独立检验"
  - "偏协方差算子"
  - "谱表示学习"
  - "对比学习"
  - "卡方零分布"
---

# Toward Scalable and Valid Conditional Independence Testing with Spectral Representations

**会议**: ICML2026  
**arXiv**: [2512.19510](https://arxiv.org/abs/2512.19510)  
**代码**: [github.com/alekfrohlich/SCIT](https://github.com/alekfrohlich/SCIT)  
**领域**: 因果推断 / 条件独立检验 / 表示学习  
**关键词**: 条件独立检验, 偏协方差算子, 谱表示学习, 对比学习, 卡方零分布

## 一句话总结
SpectralCIT 把核方法里刻画条件独立的「偏协方差算子」用神经网络学到的低维谱特征来近似，再用一个形如 HSIC 的简单统计量做条件独立检验——它用一个双层对比算法学出算子的领先奇异特征，证明零假设下统计量渐近服从卡方分布、备择假设下有功效保证，从而把核方法的理论扎实性和现代表示学习的可扩展性接上。

## 研究背景与动机
**领域现状**：条件独立（CI）检验——给定 $X,Y,Z$ 判断 $X\perp\!\!\!\perp Y\mid Z$ 是否成立——是因果推断、图模型、变量选择的基石。但在非参数设定下它出了名地难：Shah 和 Peters（2020）的 no-free-lunch 定理证明，任何能在所有条件独立分布上一致控制第一类错误的检验，对任何备择假设都没有功效。直觉是 CI 分布太「富」了，任何强相关的样本都能被一个条件独立的模型任意逼近，除非对 $P_{X,Y,Z}$ 施加结构假设（如 $P_{X,Y\mid Z=z}$ 随 $z$ 连续变化）。

**现有痛点**：这个困境逼着大家放弃「普适有效」、转向「针对特定设定」的检验，但每条路线都有短板。核方法（KCIT、RCIT）靠 $Z\to X,Y$ 的回归，要求回归学得足够好；Model-X 方法（GCIT、DGCIT）要求拿得到 $P(X\mid Z)$ 或可靠近似；局部置换检验（NNLSCIT）按分箱/聚类的 $Z$ 置换样本，但计算代价大且依赖平滑性假设。其中**经典核方法**通过偏协方差算子来刻画条件依赖，这个算子隐式编码了平滑、稀疏、低秩、隐变量等一大类结构假设、最具普适性；可惜核方法**缺乏自适应性和可扩展性**（核选择难、维度一高就掉功效、计算贵），实际影响有限。

**核心矛盾**：偏协方差算子框架理论上很美（统一刻画 CI 且不依赖单一结构假设），但它绑死在核方法上，而核方法在高维和大样本下既不 scalable 也不 adaptive——理论的普适性和实现的可扩展性凑不到一起。

**本文目标**：在保留偏协方差算子这套普适刻画的前提下，把核方法换成可学习、可扩展的表示，造一个既有效（控第一类错误）又有功效、还能 scale 到高维的 CI 检验。

**切入角度**：近年「学习统计算子的领先谱特征」在因果效应估计、强化学习、动力系统等非参数任务上都见效，且因为和对比学习相通而保持简单可扩展。作者顺着这条线问：能不能用谱表示学习去学偏协方差算子的领先特征，从而治好核方法的毛病？

**核心 idea**：用神经网络学出偏协方差算子 $\Sigma_{X\ddot{Y}\cdot Z}$ 的截断 SVD 谱特征（左/右奇异函数 + 残差化用的 $w$ 特征），再用这些特征构造一个形如 HSIC 的简单统计量；难点在于偏协方差里的残差化项 $\Sigma_{XZ}\Sigma_{ZY}$ 无法直接从数据估，作者用一个**双层（bi-level）对比公式**把它绕过去。

## 方法详解

### 整体框架
SpectralCIT 把样本 $\{(X_i,Y_i,Z_i)\}_{i=1}^N$ 切成训练集和测试集。**训练阶段**用一个双层对比算法（Algorithm 1）在训练集上学三组神经网络特征：$u_\theta(X)$、$v_\theta(\ddot{Y})$（其中 $\ddot{Y}=(Y,Z)$）、$w_\theta(Z)$，再对它们做白化（whitening）让特征在整个训练集上经验正交归一。**检验阶段**用学到的特征在测试集上算统计量 $\widehat{T}_n$（Eq. 10），它本质是「白化后的经验偏协方差矩阵」的 Frobenius 范数平方乘以 $n$。**决策阶段**把 $\widehat{T}_n$ 和自由度为 $d^2$ 的卡方分布的 $1-\alpha$ 分位数比较，超过就拒绝零假设（$d$ 是网络输出维度）。整条管线把「学表示」和「做检验」用 train/test split 解耦，既给了渐近理论一个干净的零分布，又避免数据复用带来的偏差。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["i.i.d. 样本切 train/test"] --> B["偏协方差的双层变分公式<br/>把不可估的残差化项转成低秩内层问题"]
    B --> C["双层对比表示学习<br/>学谱特征 u(X),v(Y,Z),w(Z)"]
    C --> D["白化后处理<br/>训练集上强制经验正交归一"]
    D --> E["谱统计量 T_n<br/>白化偏协方差矩阵 Frobenius 范数²×n"]
    E -->|T_n ≥ χ²(d²) 的 1-α 分位| F["拒绝 H0：条件依赖"]
    E -->|否则| G["不拒绝：条件独立"]
```

### 关键设计

**1. 用偏协方差算子的截断 SVD 把「条件依赖」压成一个低维矩阵**

条件独立 $X\perp\!\!\!\perp Y\mid Z$ 等价于偏协方差算子的 Hilbert–Schmidt 范数为零：$\|\Sigma_{X\ddot{Y}\cdot Z}\|_{\mathrm{HS}}=0$，其中 $\Sigma_{AB\cdot C}=\Sigma_{AB}-\Sigma_{AC}\Sigma_{CB}$。作者把检验重述为 $\mathcal{H}_0:\|\Sigma_{X\ddot{Y}\cdot Z}\|_{\mathrm{HS}}^2=0$ vs. $\mathcal{H}_{1,n,d}:\|\Sigma_{X\ddot{Y}\cdot Z}\|_{\mathrm{HS}}^2\geq\epsilon_n$。这个算子框架的妙处在于它**隐式编码了平滑、稀疏、低秩、隐变量等一大类结构假设**，不需要像回归型或局部置换型检验那样显式假定 $P_{X\mid Z=z}$ 的 Lipschitz 性。为了让它可计算又可扩展，作者只学算子的**最佳 rank-$d$ 截断 SVD** $[\![\Sigma_{X\ddot{Y}\cdot Z}]\!]_d=\sum_{i=1}^d \sigma_i u_i\otimes v_i$——把无穷维算子压成 $d$ 维子空间。这一步把「估计整个条件分布」降级成「捕获算子的一个低维谱子空间」，按经典的「检验比估计容易」（Ingster 1993）的道理，要求比回归型检验弱得多。

**2. 双层变分公式：绕开无法直接估计的残差化项**

把截断 SVD 写成变分（最小化）问题时，目标里会冒出一项 $\mathrm{tr}[\mathsf{M}^\top\mathsf{U}^*\Sigma_{XZ}\Sigma_{Z\ddot{Y}}\mathsf{V}]$——它是两个协方差算子的复合 $\Sigma_{XZ}\Sigma_{Z\ddot{Y}}$，**不能从数据直接估计**，这正是学偏协方差（相比无条件协方差）的核心技术障碍。作者的招数是利用迹的循环性：注意到算子 $\Sigma_{Z\ddot{Y}}\mathsf{V}\mathsf{M}^\top\mathsf{U}^*\Sigma_{XZ}:L^2(Z)\to L^2(Z)$ 至多是 rank-$d$ 的，其对称化至多 rank-$2d$，于是有 SVD 形式 $\mathsf{W}\mathsf{N}\mathsf{W}^*$（$\mathsf{W}=[w_1|\cdots|w_{2d}]$ 是 $L^2(Z)$ 上一组正交系）。把这个低秩辅助分解套进同一套变分原理，那个讨厌的复合项就被一个**内层优化问题**（关于 $(\mathsf{W},\mathsf{N})$）替代了。最终偏协方差的截断 SVD 可写成 $[\![\Sigma_{X\ddot{Y}\cdot Z}]\!]_d=\mathsf{U}[C_{UV}-C_{UW}C_{WV}]\mathsf{V}^*$——条件依赖结构全装进矩阵 $C_{UV}-C_{UW}C_{WV}$ 里，只要有合适的表示 $(U,V,W)=(u(X),v(\ddot{Y}),w(Z))$ 就能算。把外层（学 $u,v$）和内层（学 $w$）合起来就是一个**双层优化问题**，这是整个方法相对已有谱学习文献的实质差别所在。

**3. 双层对比算法 + 白化后处理：把变分公式落成可训练的神经网络损失**

作者把 $u,v,w$ 用神经网络 $u_\theta:\mathcal{X}\to\mathbb{R}^d$、$v_\theta:\mathcal{Y}\times\mathcal{Z}\to\mathbb{R}^d$、$w_\theta:\mathcal{Z}\to\mathbb{R}^{2d}$ 参数化，并用 U-统计量把外层、内层目标写成可对 mini-batch 估计的经验损失 $\widehat{\mathcal{L}}_{\mathrm{out}}$、$\widehat{\mathcal{L}}_{\mathrm{in}}$（形式上是对比学习里常见的正对吸引、负对平方项）。Algorithm 1 交替优化：每轮先跑 $n_{\texttt{steps\_inner}}$ 步更新内层网络 $w_\theta$，再用一个新 batch 更新外层网络 $u_\theta,v_\theta$。训练时还加**正交归一（白化）正则** $\widehat{\Omega}(\theta)=\|\widehat{C}_{UU}-I_d\|_F^2+\|\widehat{C}_{VV}-I_d\|_F^2$（内层是 $\|\widehat{C}_{WW}-I_{2d}\|_F^2$），强度 $\gamma>0$，目的是让经验协方差矩阵良态、白化时可安全求逆。由于 batch 级正交只是软约束且和别的目标竞争，训练后还做一步**白化后处理**：用全训练集估的协方差做 $\widehat{u}_\theta(X)=\widehat{C}^{-1/2}_{\widetilde{U}\widetilde{U}}\widetilde{u}_\theta(X)$（$v,w$ 同理），它保持学到的子空间不变（range 不变）、只改善基函数的几何，使表示在经验上严格正交归一——这一步对零分布精确收敛到卡方至关重要。

**4. 谱统计量与卡方零分布 / 功效保证：把表示质量直接挂到检验性能上**

检验阶段用测试集算 $\widehat{T}_n=n\,\|\widehat{C}_{\widehat{U}\widehat{V}}-\widehat{C}_{\widehat{U}\widehat{W}}\widehat{C}_{\widehat{W}\widehat{V}}\|_F^2$（Eq. 10），它就是 $[\![\Sigma_{X\ddot{Y}\cdot Z}]\!]_d$ 那个矩阵 $C_{UV}-C_{UW}C_{WV}$ 的经验版范数，形式上与 HSIC 同源。理论把检验性能直接绑到两个**表示误差**上：validity 误差 $\mathcal{E}_m^{\mathrm{val}}$（白化后特征离正交归一有多远）和 power 误差 $\mathcal{E}_m^{\mathrm{pow}}$（学到的 rank-$d$ 表示离真算子截断 SVD 有多远）。**有效性**（Theorem 4.1）：只要 $\mathcal{E}_m^{\mathrm{val}}\to0$，零假设下 $\widehat{T}_n\stackrel{d}{\to}\chi^2(d^2)$——直觉是白化后特征近似正交归一，于是该矩阵在零假设下近似单位协方差，$\widehat{T}_n$ 近似 $d^2$ 个独立标准高斯的平方和；这比回归型检验「以指定速率估条件期望」的要求**更弱**，因为只需抓住一个低维谱子空间。**功效**（Theorem 4.2）：当信号强度 $\epsilon_n^2\gtrsim d(\mathcal{E}_m^{\mathrm{pow}})^2+\frac{d^2+d\log(\delta^{-1})}{n}$ 时，备择假设下以至少 $1-\delta$ 的概率拒绝零假设。两个定理合起来说明：$\mathcal{E}_m^{\mathrm{val}}$ 控零假设下的校准、$\mathcal{E}_m^{\mathrm{pow}}$ 控备择假设下的信号保留，检验好不好就看表示学得好不好。

### 损失函数 / 训练策略
外层损失 $\widehat{\mathcal{L}}_{\mathrm{out}}$ 含三项：负对的平方项 $\frac{1}{m(m-1)}\sum_{i\neq j}\langle\bar{u}_i,M\bar{v}_j\rangle^2$、正对的吸引项 $-\frac{2}{m}\sum_i\langle\bar{u}_i,M\bar{v}_i\rangle$，以及携带 $w$ 残差化的耦合项 $\frac{2}{m(m-1)}\sum_{i\neq j}\langle\bar{u}_i,M\bar{v}_j\rangle\langle\bar{w}_i,\bar{w}_j\rangle$；内层损失 $\widehat{\mathcal{L}}_{\mathrm{in}}$ 则围绕 $w$ 的二次项与耦合项最小化（$\bar{\cdot}$ 表示中心化，$M=M_\theta$、$N=(N_\theta+N_\theta^\top)/2$）。激活函数取有界的（如 Tanh）以满足理论里的 sub-Gaussian 假设；超参（含 $d$）按附录 C 选取。

## 实验关键数据

### 主实验（合成数据：post-nonlinear 模型 + 高维核对比）
在固定样本量、变化条件维度 $d_Z$ 的多组合成设定下，对比 KCIT/RCIT/GCIT/DGCIT/NNLSCIT 及核方法 LPCIT/GCM 的第一类错误与功效（$\alpha=0.05$，每设定重复 100 次）。

| 设定 / 方法 | 第一类错误控制 | 功效 | 说明 |
|------|------|------|------|
| post-nonlinear · KCIT/RCIT/GCIT | 失控 | 高 | 功效高但 type I error 控不住 |
| post-nonlinear · DGCIT | 严重失控 | — | 完全失去 type I error 控制 |
| post-nonlinear · NNLSCIT | 稳健 | 高 | 与 Li et al. 2023 一致 |
| post-nonlinear · **SpectralCIT** | 稳健 | 高 | 各维度都同时控错且高功效 |
| 非平滑高维（$h_k$ 振荡）· NNLSCIT | 崩溃 | — | 振荡破坏平滑性假设，type I error 完全失控 |
| 非平滑高维 · **SpectralCIT** | 稳健 | — | 算子框架不靠平滑性，仍稳 |
| 高维核对比（$d_Z$ 50→300）· KCIT | 失控 | 高 | $d_Z\in\{250,300\}$ 功效更高但 type I error 失控 |
| 高维核对比 · GCM | 有效 | 极低 | 控错但几乎没功效 |
| 高维核对比 · **SpectralCIT** | 稳健 | 高 | 同时兼顾有效性与功效 |

关键对照：在 $X=f(Z/2+\varepsilon_X),Y=g(Z/2+\varepsilon_Y)$ 且 $f,g$ 在原点附近高度振荡（$\cos(2\pi/w)$）的非平滑设定下，NNLSCIT 的核心平滑假设被违反、第一类错误彻底崩溃，而 SpectralCIT 的算子框架不依赖这类平滑性，仍保持稳健——这正体现了「偏协方差算子隐式涵盖更广结构」的优势。

### 真实数据（乳腺癌：分子谱 vs 组织学影像）
在 TCGA-BRCA 上构造 $N=1341$ 个三元组：$X\in\mathbb{R}^3$ 是 Her2/Luminal/Basal 三个 metagene 分数，$Y\in\{0,1\}$ 是生存结局，$Z\in\mathbb{R}^{384}$ 是 Path Foundation 模型提取的组织学影像特征。问的是「在影像特征已知的前提下，分子谱还能否提供增量预测信息」。

| 检验 | P 值 | 结论 |
|------|------|------|
| **SpectralCIT** | $<10^{-3}$ | 强烈拒绝条件独立 |
| KCIT | $6.8\times10^{-2}$ | 未能在 $\alpha=0.05$ 拒绝 |
| NNLSCIT | $3.8\times10^{-1}$ | 未能拒绝 |

线性偏相关几乎为零（$-0.006,-0.088,0.07$），逻辑回归加 $X$ 反而略降准确率（0.86 vs 0.87），但 XGBoost 把 $X$ 加进去后准确率从 0.91 升到 0.95——证实存在**非线性的残余预测信息**。只有 SpectralCIT 检出了这种被影像特征漏掉的复杂依赖，凸显其在高维生物数据上捕获非线性条件依赖的能力。

### 关键发现
- 几乎所有核方法（KCIT、RCIT、GCIT）在高维下要么失控第一类错误、要么掉功效；DGCIT 直接崩；GCM 控错但几乎无功效——SpectralCIT 是少数能同时兼顾两者的。
- 非平滑设定是 NNLSCIT 的死穴而非 SpectralCIT 的：算子框架不显式假定条件分布平滑，更稳。
- 真实乳腺癌数据上，线性方法和两个强基线都判「条件独立」，唯独 SpectralCIT 检出非线性残余信号，并被 XGBoost 的预测增益佐证，体现可落地价值。

## 亮点与洞察
- **把核方法的偏协方差算子换成可学习谱特征**是核心洞察：保留了算子框架「不靠单一结构假设」的普适性，同时借对比学习拿到可扩展性，治好了核方法高维掉功效、计算贵的老毛病。
- **双层变分把不可估的残差化项绕过去**很巧：用低秩辅助 SVD（$\mathsf{W}\mathsf{N}\mathsf{W}^*$）把 $\Sigma_{XZ}\Sigma_{Z\ddot{Y}}$ 这个复合算子替换成一个内层优化，是「学偏协方差 vs 学无条件协方差」差别的关键，可迁移到其他需要残差化的算子学习任务。
- **把检验有效性/功效直接挂到表示误差 $\mathcal{E}_m^{\mathrm{val}},\mathcal{E}_m^{\mathrm{pow}}$ 上**，给「表示学习质量 → 统计检验性能」建了一座桥，比「以指定速率估回归」的要求更弱、更贴近实践。
- **白化后处理**这个工程细节对「零分布精确收敛到 $\chi^2(d^2)$」至关重要——它把 batch 级软正交补成全集级硬正交，是理论能落地的隐形功臣。

## 局限与展望
- 检验依赖**表示学得足够好**（$\mathcal{E}_m^{\mathrm{val}}\to0$ 才有效），若网络欠训练或容量不足，零分布近似卡方会偏、校准会坏；理论是渐近的，有限样本行为仍受表示质量牵制。
- 需要选维度 $d$ 等超参，作者也承认 CI 检验文献「缺标准化基准与成熟调参协议」，$d$ 的选取在实践中可能敏感。
- 训练/检验 split 牺牲了一部分样本效率；双层 + 内外层交替优化相比核方法实现更复杂、训练成本更高。
- 真实数据实验规模有限（135 张 WSI 扩成 1341 个 patch 级三元组），单一癌种、单一结局划分，结论的临床普适性仍需更多验证。

## 相关工作与启发
- **vs 核方法（KCIT/RCIT, Zhang 2011 / Strobl 2019）**：同样基于偏协方差算子的 HS 范数，但核方法用固定核 + 随机傅里叶特征近似，高维掉功效、计算贵；SpectralCIT 用学到的谱特征替代核，scalable 且 adaptive。
- **vs Model-X（GCIT/DGCIT, Bellot 2019 / Shi 2021）**：它们要求拿到或学到 $P(X\mid Z)$ 再做置换/生成，bound 在简单线性设定都可能松；SpectralCIT 不建模条件分布，只学算子子空间，避开了对 $P(X\mid Z)$ 的依赖。
- **vs 局部置换（NNLSCIT, Li 2023）**：靠分箱/聚类 $Z$ 内置换，依赖平滑性且计算重；SpectralCIT 在非平滑振荡设定下仍控错，而 NNLSCIT 崩溃。
- **vs GCM（Shah & Peters 2020）**：GCM 有效但高维功效极低；SpectralCIT 在同设定下兼顾有效与功效。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把谱表示学习引入偏协方差算子的 CI 检验，双层公式绕开残差化项，思路新且接上了两条本不相通的线。
- 实验充分度: ⭐⭐⭐⭐ 三组合成设定（含针对性的非平滑反例）+ 真实乳腺癌数据，对照面广；但真实实验规模偏小。
- 写作质量: ⭐⭐⭐⭐ 算子/变分推导严谨、定理陈述清楚，但理论密度高、对读者数学背景要求不低。
- 价值: ⭐⭐⭐⭐ 给高维非参 CI 检验提供了可扩展又有理论保证的新工具，对因果发现、特征选择有实际意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Efficient Ensemble Conditional Independence Test Framework for Causal Discovery](../../ICLR2026/causal_inference/efficient_ensemble_conditional_independence_test_framework_for_causal_discovery.md)
- [\[ICML 2026\] Outcome-Aware Spectral Feature Learning for Instrumental Variable Regression](outcome-aware_spectral_feature_learning_for_instrumental_variable_regression.md)
- [\[ICLR 2026\] Learning Robust Intervention Representations with Delta Embeddings](../../ICLR2026/causal_inference/learning_robust_intervention_representations_with_delta_embeddings.md)
- [\[ICLR 2026\] Journey to the Centre of Cluster: Harnessing Interior Nodes for A/B Testing under Network Interference](../../ICLR2026/causal_inference/journey_to_the_centre_of_cluster_harnessing_interior_nodes_for_ab_testing_under_.md)
- [\[ICLR 2026\] Direct Doubly Robust Estimation of Conditional Quantile Contrasts](../../ICLR2026/causal_inference/direct_doubly_robust_estimation_of_conditional_quantile_contrasts.md)

</div>

<!-- RELATED:END -->
