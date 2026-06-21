---
title: >-
  [论文解读] On the Alignment Between Supervised and Self-Supervised Contrastive Learning
description: >-
  [ICLR 2026][自监督学习][对比学习] 本文从理论上证明：在共享随机性下，自监督对比学习（CL）与一种监督替身——"仅负样本监督对比"（NSCL）——在**表示相似度空间**会全程保持高度对齐（CKA/RSA 有高概率下界），而它们的**参数**却可能指数级发散，从而把 NSCL 确立为连接自监督与监督学习的一座有原则的桥梁。
tags:
  - "ICLR 2026"
  - "自监督学习"
  - "对比学习"
  - "自监督"
  - "NSCL"
  - "表示对齐"
  - "CKA/RSA"
---

# On the Alignment Between Supervised and Self-Supervised Contrastive Learning

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=JkitQScjuL](https://openreview.net/forum?id=JkitQScjuL)  
**代码**: https://dlfundamentals.github.io/cl-nscl-representation-alignment  
**领域**: 自监督 / 表示学习 / 对比学习理论  
**关键词**: 对比学习, 自监督, NSCL, 表示对齐, CKA/RSA

## 一句话总结
本文从理论上证明：在共享随机性下，自监督对比学习（CL）与一种监督替身——"仅负样本监督对比"（NSCL）——在**表示相似度空间**会全程保持高度对齐（CKA/RSA 有高概率下界），而它们的**参数**却可能指数级发散，从而把 NSCL 确立为连接自监督与监督学习的一座有原则的桥梁。

## 研究背景与动机
**领域现状**：自监督对比学习（SimCLR、MoCo、CPC 等）已经能学出可媲美甚至超过监督预训练的表示，核心机制是"拉近同一样本的增广视图、推远其它样本"。一个长期谜题是：明明没有标签，CL 为什么能学出与语义类别边界高度吻合的特征？

**现有痛点**：近期 Luthra et al. (2025) 给出了**损失层面**的解释——CL 的 InfoNCE 目标在类别数 $C$ 增大时，会以 $O(1/C)$ 的速度逼近一个监督变体 NSCL（NSCL 在分母里剔除同类样本，只对负样本归一化）。但这只说明两个**目标函数**接近，并不能保证两条**优化轨迹**走到一起：曲率差异、梯度噪声、学习率调度都可能把微小的损失差放大，让 SGD 轨迹分道扬镳。

**核心矛盾**：损失接近 $\neq$ 表示接近。我们真正关心的是下游行为，而下游行为由表示几何决定，不由损失值决定。因此"CL 究竟只是收敛到一个与 NSCL 相似的解，还是在整个训练过程中与 NSCL 始终耦合"这个问题悬而未决。

**本文目标**：在**共享随机性**（同一初始化、同一 mini-batch、同一增广）下，刻画 CL 与 NSCL 在训练全程的对齐：(1) 它们的表示是否始终相似？(2) 在什么条件下对齐、哪些因素控制对齐强度？(3) 参数空间是否也同样耦合？

**切入角度**：放弃在参数空间分析（非凸动力学下参数漂移不可控、且对重参数化敏感），转而在**相似度矩阵空间**分析——这个视角对重参数化不变，且直接刻画表示几何。

**核心 idea**：用一个"相似度下降"替代动力学跟踪 CL/NSCL 的相似度矩阵演化，证明二者的 Frobenius 漂移被一个随类别数、batch、温度系统变化的界控制，再把这个界翻译成 CKA/RSA 的高概率下界。

## 方法详解

### 整体框架
这是一篇**纯理论分析**论文（无网络结构创新），其"方法"是一条环环相扣的证明链。设有数据集 $S=\{(x_i,y_i)\}_{i=1}^N$ 含 $C$ 个类，编码器 $f_w$ 把输入映射为嵌入，相似度用 $\ell_2$ 归一化后的余弦相似度。对一个 batch 内的锚点 $i$，CL 的逐锚损失把所有其它样本（含同类）当负样本放进分母，而 NSCL 只把**异类**样本放进分母：

$$\ell^{CL}_i = -\log\frac{\exp(\mathrm{sim}(z_i,z_i')/\tau)}{\sum_{t\neq i}\exp(\mathrm{sim}(z_i,z_t)/\tau)+\exp(\mathrm{sim}(z_i,z_t')/\tau)},\quad \ell^{NSCL}_i=-\log\frac{\exp(\mathrm{sim}(z_i,z_i')/\tau)}{\sum_{j\in I_i^-}[\cdots]}$$

其中 $I_i^-=\{j:y_j\neq y_i\}$ 是异类负样本下标集。整条分析从"在参数空间分析会爆炸"这个观察出发，转入相似度空间，分三步走：① 把参数 SGD 诱导的相似度更新近似为只更新当前 batch 触及条目的"相似度下降"替代动力学；② 证明两条相似度轨迹的 Frobenius 漂移满足一个递推、解出指数耦合界（定理 1）；③ 把这个界翻译成 CKA、RSA 两个标准表示对齐指标的高概率下界（推论 1–2）。最后用参数空间的不稳定性结果（定理 2）作为对照，说明"参数发散但表示对齐"二者不矛盾。

为量化表示相似度，文中用两个指标：线性 CKA 是两个**中心化**相似度矩阵的归一化 Frobenius 内积，$\mathrm{CKA}(Z,Z')=\frac{\langle H\Sigma(Z)H,\,H\Sigma(Z')H\rangle_F}{\|H\Sigma(Z)H\|_F\,\|H\Sigma(Z')H\|_F}$，其中 $H=I-\frac1N\mathbf{1}\mathbf{1}^\top$ 是中心化投影；RSA 是两个相异度矩阵 $\mathrm{RDM}=\mathbf{1}\mathbf{1}^\top-\Sigma$ 上三角非对角元之间的 Pearson 相关。两者都落在 $[0,1]$，越接近 1 表示相似结构越一致。

### 关键设计

**1. 相似度空间替代动力学：绕开参数空间的不可控性**

作者首先论证：直接在参数空间研究 CL 与 NSCL 的轨迹会"崩"——在没有凸性/强凸性假设的非凸损失上，小的重参数化就能扭曲距离，参数漂移随时间不受控地增长。于是把分析对象从权重 $w$ 换成相似度矩阵 $\Sigma_t\in[-1,1]^{N\times N}$（固定参考集嵌入的两两余弦相似度）。为让分析可解析，定义一个"相似度下降"替代：每步只更新当前 mini-batch 触及的那些条目，$\Sigma^{CL}_{t+1}=\Sigma^{CL}_t-\eta_t G^{CL}_t$，$\Sigma^{NSCL}_{t+1}=\Sigma^{NSCL}_t-\eta_t G^{NSCL}_t$，其中 $G_t=\nabla_\Sigma\bar\ell_{B_t}(\Sigma_t)$ 是把损失写成相似度条目函数后的 batch 梯度图，未触及条目置零。附录 D 证明：在 Jacobian 谱范数有界 $\|J(w)\|_{2\to2}\le L_\Sigma$、二阶 Taylor 余项有界、学习率调度满足 $\sum_t\eta_t/(\tau^2 B)$ 与 $\sum_t\eta_t^2$ 有界等正则条件下，这个替代轨迹与真实参数 SGD 诱导的相似度轨迹 $\hat\Sigma_t=\Sigma(w_t)$ 一致逼近。直觉是：小步长、足够大 batch、适中温度时，参数 SGD 推动相似度的方式几乎等同于直接在相似度空间做梯度下降。这一步是整篇分析能成立的支点——它把一个对重参数化敏感、会爆炸的问题，换成了对重参数化不变、可控的问题。

**2. 相似度耦合界：用离散 Grönwall 把漂移控成指数界**

这是核心定理（定理 1）。在共享随机性下，作者先给出每步梯度失配的估计：把 CL–NSCL 的 batch 梯度差分解为 (i) 一个**重加权误差**（NSCL 剔除同类带来的归一化差异，按总变差被 $\Delta_{\pi,\delta}(B;\tau)$ 界住）与 (ii) 一个**稳定性项**（梯度图对当前相似度的依赖，由 batch 梯度图的 $\frac{1}{2\tau^2 B}$-Lipschitz 性控制），借助跨锚点的块正交性把重加权贡献按平方和合成，得到

$$\big\|G^{CL}_t-G^{NSCL}_t\big\|_F\le\frac1\tau\cdot\frac{\Delta_{\pi,\delta}(B;\tau)}{\sqrt B}+\frac{1}{2\tau^2 B}\big\|\Sigma^{CL}_t-\Sigma^{NSCL}_t\big\|_F.$$

代入更新式得到漂移的递推 $\|\Sigma^{CL}_{t+1}-\Sigma^{NSCL}_{t+1}\|_F\le(1+\frac{\eta_t}{2\tau^2 B})\|\Sigma^{CL}_t-\Sigma^{NSCL}_t\|_F+\eta_t\frac{\Delta_{\pi,\delta}(B;\tau)}{\tau\sqrt B}$——第一项传播已有误差，第二项注入本步新差异。展开这个递推（离散 Grönwall 不等式）即得**至少 $1-\delta$ 概率**成立的界：

$$\big\|\Sigma^{CL}_T-\Sigma^{NSCL}_T\big\|_F\le\exp\!\Big(\frac{1}{2\tau^2 B}\sum_{t=0}^{T-1}\eta_t\Big)\Big(\frac{1}{\tau\sqrt B}\sum_{t=0}^{T-1}\eta_t\Big)\,\Delta_{\pi,\delta}(B;\tau),$$

其中 $\Delta_{\pi,\delta}(B;\tau)=\dfrac{2\,e^{2/\tau}(\pi_{\max}+\epsilon_{B,\delta})}{1-\pi_{\max}-\epsilon_{B,\delta}}$，$\epsilon_{B,\delta}=\sqrt{\frac{1}{2B}\log(TB/\delta)}$，$\pi_{\max}=\max_c\pi_c$ 是最大类先验。这个界的价值在于它**把抽象的"何时对齐"翻译成了 CL 常见超参的单调关系**：类别数 $C$ 越大（均衡类下 $\pi_{\max}\approx1/C$ 越小，$\Delta$ 越小）、batch $B$ 越大（同时压低浓度误差 $\epsilon_{B,\delta}$、前因子 $1/\sqrt B$ 和指数里的 $\frac{1}{2\tau^2 B}$）、温度 $\tau$ 越高（压低 $1/\tau$、$1/\tau^2$ 与 $e^{2/\tau}$）、有效步长 $\sum_t\eta_t$ 越小，界都越紧——而这些恰好就是"CL 表现得像 NSCL"的真实经验区间。关键洞察是：相似度空间的"不稳定率"只有 $\frac{1}{2\tau^2 B}$，对典型 $B\sim10^2$–$10^3$ 几乎可忽略，与参数空间里以光滑系数 $\beta$ 为增长率截然不同。

**3. 从相似度漂移到 CKA/RSA 保证：让理论落到可观测指标上**

定理 1 控制的是 Frobenius 漂移，但实践中人们看的是 CKA、RSA。这一步把界翻译过去。因为中心化是收缩映射 $\|HXH\|_F\le\|X\|_F$，对 $\Sigma$ 的界自动控制中心化 Gram 矩阵 $K=H\Sigma H$ 的差。定义相对偏差 $\rho_T=\|K^{CL}_T-K^{NSCL}_T\|_F/\|K^{CL}_T\|_F$，推论 1 给出 **CKA 高概率下界** $\mathrm{CKA}_T\ge\frac{1-\rho_T}{1+\rho_T}$；同理对 RSA，定义 $r_T=\|b_T-a_T\|_2/(\sqrt M\,\sigma_{D,T})$（$a_T,b_T$ 是两模型 RDM 上三角向量，$M=\binom N2$，$\sigma_{D,T}$ 是 $a_T$ 的经验标准差），推论 2 给出 $\mathrm{RSA}_T\ge\frac{1-r_T}{1+r_T}$。在现实区间（$C\sim10^3$、$B\sim10^2$–$10^3$）里 $\rho_T,r_T\ll1$，于是两个指标都被钉在接近 1。这恰恰解释了图 1 的现象：**即便权重发散，诱导出的表示却以耦合且稳定的方式演化**。

**4. 参数空间的内在不稳定性：发散与对齐并不矛盾**

为完整性，作者在 $\beta$-光滑假设下给出参数漂移界（定理 2），证明 CL 与 NSCL 的**权重差可随训练时间指数增长**——这里的增长率由光滑系数 $\beta$ 主导，远比相似度空间的 $\frac{1}{2\tau^2 B}$ 凶猛。这一对照构成本文的概念主张：参数耦合本质不稳定，而表示耦合本质稳定，所以"两个模型权重越走越远、表示却始终对齐"完全可以同时发生。这也从根本上说明了为什么应该在相似度空间而非参数空间衡量自监督与监督学习的关系。

### 损失函数 / 训练策略
分析的对象就是上面的 CL（InfoNCE 型）与 NSCL（剔除同类的负样本归一化）两个损失本身；实验中 CL 采用解耦的 DCL 损失（避免正负耦合），监督对照包含 NSCL、SCL（监督对比）与 CE（交叉熵）。证明的关键技术假设是共享随机性、$\beta$-光滑、Jacobian 谱范数有界，以及一个高概率 batch 组成保证（推论 3：每个锚点分母里负样本占比与期望偏差不超过 $\epsilon_{B,\delta}$，排除"正样本过多"的坏 batch）。

## 实验关键数据

实验用 ResNet-50 编码器 + 两层 MLP 投影头（2048→2048→ReLU→128），LARS 优化器，batch $B=1024$，按 SimCLR 配方训练，覆盖 CIFAR-10/100、Mini-ImageNet、Tiny-ImageNet、ImageNet-1K。

### 主实验
下游评估用最近类心分类器（NCCC）与线性探针（LP）准确率（%）。注意：本文论点不是"NSCL 下游最强"，而是"NSCL 与 CL 表示最对齐"——所以 SCL/CE 下游更高并不削弱主张。

| 数据集 | 指标 | CL | NSCL | SCL | CE |
|--------|------|----|------|-----|-----|
| CIFAR-10 | NCCC / LP | 88.37 / 90.16 | 94.47 / 94.09 | 94.93 / 94.67 | 92.97 / 93.39 |
| CIFAR-100 | NCCC / LP | 54.62 / 65.65 | 60.14 / 68.38 | 64.06 / 69.52 | 67.35 / 68.04 |
| Mini-ImageNet | NCCC / LP | 60.78 / 65.30 | 63.92 / 72.60 | 74.78 / 76.00 | 75.20 / 74.00 |
| Tiny-ImageNet | NCCC / LP | 40.59 / 44.61 | 40.76 / 45.79 | 48.63 / 48.73 | 48.28 / 52.57 |

最有说服力的对齐数据：在 Tiny-ImageNet 上训练 1000 epoch 后，**CL–NSCL 的 CKA 达 0.87，而 CL–SCL 仅 0.043**——NSCL 比其它任何监督目标都更紧地跟踪 CL。

### 消融实验（理论预测的逐项验证）

| 控制变量 | 现象 | 对应理论 |
|----------|------|----------|
| 类别数 $C'$（$C'$-way 子集训练 1000 ep） | CKA/RSA 随 $C'$ 单调上升（图 3，所有数据集一致） | $\Delta$ 中 $1/C$ 项随 $C$ 减小 |
| 温度 $\tau\in\{0.1,0.5,1.0\}$（训 300 ep） | $\tau=1.0$ 全程对齐最高 | 前因子/指数里 $1/\tau$、$1/\tau^2$ 随 $\tau$ 减小 |
| batch $B$ 与学习率缩放 | $\eta=O(B)$ 时对齐随 $B$ **下降**；$\eta=O(\sqrt B)/O(\sqrt[4]B)/$ 常数时随 $B$ **上升** | 界随 $B$ 升降取决于 $\eta$ 如何缩放，符号完全对上 |
| 训练时长 | 前 ~1000 ep CL 最贴 NSCL；继续训很久后对齐下降 | NSCL 比 SCL/CE 更晚进入神经坍缩区 |
| 权重空间 | NSCL、SCL 的权重都随训练越来越远离 CL | 定理 2：参数发散可指数增长 |

### 关键发现
- **NSCL 始终是与 CL 最对齐的监督目标**，远超 SCL 与 CE；机制是三者都诱导神经坍缩，但 NSCL 与 CL 结构最像（都只拉一个正样本、推开负样本、做实例级判别），而 SCL 施加更强的类级约束、更快形成紧类簇，CE 居中。
- batch 实验是最精细的验证：界对 $B$ 的影响方向**随学习率缩放翻转**，实验四种缩放下对齐升降方向与定理符号逐一吻合，说明界刻画的不是平凡单调性。
- 表示对齐与权重发散在同一组实验里**同时观测到**，直接坐实了"参数发散 ≠ 表示发散"。

## 亮点与洞察
- **换空间换出可解性**：把"参数轨迹是否耦合"这个会爆炸的问题，换成"相似度矩阵是否耦合"这个对重参数化不变、可被 Grönwall 控住的问题——这是全篇最巧的一步，思路可迁移到任何"两个目标的优化轨迹是否接近"的分析。
- **界即旋钮**：定理 1 的界把类别数、batch、温度、学习率全写成显式单调因子，理论直接给出"想让自监督更像监督，就调大 $C$/$B$/$\tau$、调小步长"的可操作处方，而非只给存在性结论。
- **视角转移**：以往理论关心"自监督损失最小化能否保证下游分类准确率"，本文把焦点移到"CL 与 NSCL 是否诱导相似的相似度结构"，这对依赖表示几何的任务（可解释性、图像分割）比纯分类准确率更贴切。
- **一个反直觉结论被讲清**：权重指数发散与表示稳定对齐可以并存，提醒后续做模型相似性/模型融合的工作不要用参数距离当表示相似度的代理。

## 局限与展望
- 理论建立在共享随机性（同初始化、同 batch、同增广）这一强假设上；现实里两个独立训练的模型不共享随机性，对齐是否仍成立、强度如何，文中未理论覆盖。
- 替代动力学与真实参数 SGD 的一致逼近依赖一组正则条件（Jacobian 谱范数有界、二阶余项可控、特定学习率调度），这些在深网上是否严格成立缺乏直接验证。
- 实验观察到"训练极久后对齐下降"，但定理 1 的界只随 $\sum_t\eta_t$ 增长而变松、并不预测对齐**回落**的精细形状，理论与长程经验之间留有缝隙。
- 全部实验为视觉分类 + ResNet-50 + SimCLR 配方，对语言/语音/多模态等其它对比学习场景的可迁移性尚待检验。

## 相关工作与启发
- **vs Luthra et al. (2025)**：他们证明 CL 与 NSCL 在**损失层面**以 $O(1/C)$ 收敛、并刻画 NSCL 极小点几何；本文接着回答"损失接近是否意味着**表示**全程接近"，把分析从目标函数推进到训练动力学与相似度几何。
- **vs Balestriero & LeCun (2024)**：他们在线性模型里证明 VICReg 等自监督目标等价于监督二次损失；本文不限于线性、不依赖架构，给出训练全程一致成立、与标签无关的相似度耦合界。
- **vs alignment/uniformity 路线（Wang & Isola 2020 等）**：那条线用球面上正样本聚集、负样本铺开来刻画 CL 几何，但不解释不同语义类如何被组织；本文直接用 CL 与监督目标的相似度结构对齐来回答"监督信号如何隐含在 CL 里"。
- **vs Grigg et al. (2021)**：他们经验性观察到监督与自监督模型表示大致几何对齐；本文给出该现象的理论解释，并量化对齐受类别数、batch、温度、学习率控制的条件。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把表示对齐分析从损失层面推到训练全程的相似度空间，"换空间求可解性 + 界即超参旋钮"的组合很有原创性。
- 实验充分度: ⭐⭐⭐⭐ 五个数据集逐项验证类别数/温度/batch/时长四类预测，但全为视觉单架构，缺跨模态。
- 写作质量: ⭐⭐⭐⭐⭐ 问题动机清晰、定理与经验一一对应，理论叙事干净。
- 价值: ⭐⭐⭐⭐ 为"自监督为何媲美监督"提供有原则的桥梁，并给出可操作的对齐调参处方，对表示相似性研究有指导意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning](../../NeurIPS2025/self_supervised/self-supervised_contrastive_learning_is_approximately_supervised_contrastive_lea.md)
- [\[ICLR 2026\] Dual Perspectives on Non-Contrastive Self-Supervised Learning](dual_perspectives_on_non-contrastive_self-supervised_learning.md)
- [\[ICLR 2026\] Understanding the Robustness of Distributed Self-Supervised Learning Frameworks Against Non-IID Data](understanding_the_robustness_of_distributed_self-supervised_learning_frameworks_.md)
- [\[ICLR 2026\] Soft Equivariance Regularization for Invariant Self-Supervised Learning](soft_equivariance_regularization_for_invariant_self-supervised_learning.md)
- [\[ICLR 2026\] Equivariant Splitting: Self-supervised learning from incomplete data](equivariant_splitting_self-supervised_learning_from_incomplete_data.md)

</div>

<!-- RELATED:END -->
