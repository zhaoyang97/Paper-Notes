---
title: >-
  [论文解读] Rényi Sharpness: A Novel Sharpness That Strongly Correlates with Generalization
description: >-
  [ICLR 2026][学习理论][锐度] 本文指出真正决定泛化的是 Hessian 谱的"平均散度/不均匀度"，于是用信息论里的 Rényi 熵把它定义成 **Rényi sharpness**（Hessian 归一化谱的负 Rényi 熵），证明它与泛化在大量场景下都强相关（Kendall τ 普遍 0.6–0.9，远超 trace/SAM/PAC-Bayes 等旧度量），并据此给出泛化界和一个有竞争力的 RSAM 训练正则化算法。
tags:
  - "ICLR 2026"
  - "学习理论"
  - "泛化"
  - "损失景观"
  - "锐度"
  - "泛化界"
  - "Hessian 谱"
  - "Rényi 熵"
  - "Sharpness-Aware Minimization"
---

# Rényi Sharpness: A Novel Sharpness That Strongly Correlates with Generalization

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=rRIhUpobCv](https://openreview.net/forum?id=rRIhUpobCv)  
**代码**: 有（论文中提供匿名链接）  
**领域**: 学习理论 / 泛化 / 损失景观  
**关键词**: 锐度, 泛化界, Hessian 谱, Rényi 熵, Sharpness-Aware Minimization

## 一句话总结
本文指出真正决定泛化的是 Hessian 谱的"平均散度/不均匀度"，于是用信息论里的 Rényi 熵把它定义成 **Rényi sharpness**（Hessian 归一化谱的负 Rényi 熵），证明它与泛化在大量场景下都强相关（Kendall τ 普遍 0.6–0.9，远超 trace/SAM/PAC-Bayes 等旧度量），并据此给出泛化界和一个有竞争力的 RSAM 训练正则化算法。

## 研究背景与动机

**领域现状**：深度网络为何在过参数化下还能泛化，长期被归因于损失景观的"平坦性"——直觉上，落在平坦极小点（锐度低）的模型，数据轻微扰动只引起很小的 loss 变化，因此泛化好。锐度通常用损失 Hessian $H$ 的某个泛函来量化，最常见的两个是迹锐度 $\mathrm{tr}(H)$ 和最大特征值锐度 $\lambda_{\max}(H)$。

**现有痛点**：近年来大量实证工作发现这些锐度度量与泛化的相关性其实很弱，甚至出现矛盾——Dinh et al. (2017) 能构造出锐度任意大但泛化照样好的等价模型；Andriushchenko et al. (2023) 在现代架构上发现锐度几乎和泛化不相关；Kaur et al. (2023) 指出 $\lambda_{\max}(H)$ 即便在标准训练下也预测不准泛化。直觉和现实之间裂了一道口子。

**核心矛盾**：作者认为问题出在"只看了谱的一部分"。Hessian 谱可分三类特征值——顶部（数量少但对 loss 变化影响大）、中部（单个影响小但数量极多）、尾部（接近 0、几乎不影响）。$\mathrm{tr}(H)$ 本质只反映中部特征值的平均量级，$\lambda_{\max}(H)$ 只盯着顶部最大那个，两者都丢失了大量信息，自然预测不准。

**本文目标**：找到一个能**同时综合三类特征值贡献**的锐度度量，让它与泛化既强相关又稳定，并能反过来指导训练。

**切入角度**：作者的关键观察是——真正重要的不是谱的平均值或最大值，而是谱的**不均匀度（unevenness）/平均散度**。一个越均匀的谱（所有特征值接近相等、没有特别大的方向）越利于泛化，因为任意方向的数据扰动都只引起小 loss 变化。如何刻画"一个非负向量有多不均匀"？信息论里的熵正是干这个的。

**核心 idea**：把 Hessian 归一化谱当成一个"虚拟概率向量"，用 **Rényi 熵**度量它的不均匀度，并取其负号定义为 Rényi sharpness——熵越大（谱越均匀）锐度越小、泛化越好。相比方差（尾部近零特征值会主导方差却对泛化无关）和 Shannon 熵（无可调参数），Rényi 熵多一个自由阶数 $\alpha$，能按谱的形态灵活调节对大小特征值的敏感度。

## 方法详解

### 整体框架
本文是一篇"提出新度量 + 理论分析 + 算法落地"的工作，没有传统意义上的多阶段 pipeline，而是围绕一个核心定义层层展开：先**定义** Rényi sharpness 并证明它具有重参数化不变性；再用这个不变性**推导**两条以 Rényi sharpness 为主项的泛化界，把"度量小 → 泛化好"严格化；接着解决两个实用问题——**阶数 $\alpha$ 怎么选**（按谱形态在 0.5 与 1.5 间二选一）和**怎么快速估计**（用 SLQ 把 $O(n^3)$ 谱分解降成矩阵-向量乘）；最后把它做成训练正则化器 **RSAM**。下面四个关键设计依次对应"定义—理论—估计与选阶—正则化"这条逻辑链。

### 关键设计

**1. Rényi sharpness：用 Rényi 熵把"谱的不均匀度"变成可量化的锐度**

针对旧度量只看谱的局部（中部或顶部）的痛点，本文要的是一个能反映整条谱不均匀度的量。先回顾 Rényi 熵：对概率向量 $p=[p_1,\dots,p_n]$，阶数 $\alpha$ 的 Rényi 熵为

$$H_\alpha(p) = \frac{1}{1-\alpha}\log\sum_{i=1}^n p_i^\alpha,\quad 0<\alpha<\infty,\ \alpha\neq 1$$

它是 $p$ 的凹函数，在 $p$ 均匀时取最大值，恰好能刻画"有多均匀"。把它推广到正定矩阵 $H$：先把特征值归一化成虚拟概率 $\lambda_i(H)/\mathrm{Tr}(H)$，再代入

$$H_\alpha(H) = \frac{1}{1-\alpha}\log\sum_{i=1}^n\Big(\frac{\lambda_i(H)}{\mathrm{Tr}(H)}\Big)^\alpha$$

**Rényi sharpness 定义为该归一化谱的负 Rényi 熵 $-H_\alpha(H)$**（取单层权重的 Hessian；实际中偶现的负特征值取绝对值处理）。谱越均匀 → 熵越大 → 锐度越小 → 越利于泛化。和方差相比，它的好处是尾部近零特征值不会像在方差里那样主导结果（那些值对泛化几乎无贡献却会撑大方差）；和 Shannon 熵相比，多出的自由阶数 $\alpha$ 让它能按谱形态调焦（见设计 3）。

**2. 重参数化不变性与两条泛化界：把"度量"升格为"有理论保证的指标"**

一个好的泛化指标必须对"换汤不换药"的网络变换鲁棒——Dinh et al. (2017) 攻击旧锐度靠的正是 ReLU 网络的逐层缩放等价变换。Rényi sharpness 恰好对这类变换免疫：对正齐次激活（$\sigma(cx)=c\sigma(x)$）的 $L$ 层网络，做满足 $\prod_{l=1}^L c_l = 1$ 的逐层缩放 $\tilde W_l = c_l W_l$，则任意层 Hessian 的归一化谱 Rényi 熵保持不变，$H_\alpha(H_{\tilde\theta}) = H_\alpha(H_\theta)$（命题 2.2；ViT 的 GELU 近似齐次，故近似不变）。

借助这个不变性以及"数据扰动可转化为乘性权重扰动"的技巧（命题 3.1：把总体分布 $D$ 与训练集 $S$ 的差异看成对 $S$ 的扰动，而 $g(W(h(x)+\rho A h(x))) = g((W+\rho WA)h(x))$ 说明输入/特征空间的乘性扰动能完全转嫁到某一层的权重 $W$ 上，于是只研究单层锐度就够了），本文给出两条泛化界。其一（定理 3.2）形如

$$\mathbb{E}_Q[L(D,\theta)] \le \mathbb{E}_Q[L(S,\theta)] + 2\sqrt{\frac{2L_0 + CV^2/n\,\exp\!\big(-\tfrac1n(H_\alpha(H)-A)\big) + \log\frac{2N}{\delta}}{N-1}}$$

其二（定理 3.3）给出更直接的形式，其平方根项里含 $-\tfrac12 H_\alpha(H)$。两条界都表明：泛化 gap 被 Hessian 归一化谱的 Rényi 熵所控制——熵越大（锐度 $-H_\alpha$ 越小），界越紧，从而把"低 Rényi sharpness → 好泛化"从直觉变成了严格陈述。

**3. 阶数 $\alpha$ 的选择：按 Hessian 谱形态在 0.5 与 1.5 间二选一**

$\alpha$ 是 Rényi 熵相对 Shannon 熵多出的自由度，但用错会适得其反，所以必须给出选取规则。作者用 PyHessian 实测各层谱，发现深网 Hessian 谱虽都重尾，但形态分两类：① **零主导、多簇谱**——除大量近零值（Part 1）和少数大特征值（Part 3）外，中间还有一批"不可忽略但明显小于大特征值"的中部特征值（Part 2）；② **零主导、均匀谱**——只有近零值和少数大特征值，Part 2 几乎消失。

选阶逻辑由 $\alpha$ 对均匀度的惩罚强度决定（$\alpha$ 越大越偏重高概率质量、越放大大特征值）：对多簇谱，需要分辨 Part 2 与 Part 3 的细微差别，若 $\alpha>1$ 会过度放大大特征值而吞掉小的，故取 $\alpha\in(0,1)$、实测 $\alpha=0.5$ 最稳定；对均匀谱，关键反而是分辨那几个主导特征值之间的差异，$\alpha<1$ 会抹平差异，故取 $\alpha\ge 1$、但 $\alpha\to 1$ 数值不稳，折中取 $\alpha=1.5$。一句话概括：簇间分离大于簇内放大时选 $\alpha<1$，单簇情形选 $\alpha>1$。作者在 60 个模型、共 1630 个案例上统计"经验最优 $\alpha$ 是否与该规则预测一致"，1451 匹配、179 不匹配，验证了选阶规则的有效性。

**4. SLQ 快速估计与 RSAM 正则化：让 Rényi sharpness 既算得起又用得上**

直接对巨型 Hessian 做谱分解复杂度不可接受。作者先把 Rényi 熵改写成矩阵函数迹的形式

$$H_\alpha(H) = \frac{1}{1-\alpha}\log\frac{\mathrm{Tr}(H^\alpha)}{\mathrm{Tr}(H)^\alpha}$$

于是只需估计 $\mathrm{Tr}(H)$ 与 $\mathrm{Tr}(H^\alpha)$。再用随机迹估计（Hutchinson：$\mathrm{Tr}(f(H)) = \mathbb{E}[v^\top f(H) v]$，$v$ 取 Rademacher）加随机 Lanczos 求积（SLQ：用 Lanczos 把矩阵三对角化、用 Gauss 求积算二次型期望），把谱分解化为若干次矩阵-向量乘（算法 1），复杂度大幅下降。

把它做成训练正则化器时，为避免反复求 Hessian，进一步用梯度幅值近似 Hessian $H\approx \mathrm{GM} = \big(\mathrm{Diag}(\tfrac1N\sum_i\nabla_\theta l)\big)^2$，得到可微的 Rényi 正则项 $-\mathrm{sign}(1-\alpha)\frac{\sum_j|g_j|^{2\alpha}}{(\sum_j g_j^2)^\alpha}$。为绕开保留计算图的开销，RSAM 仿照 SAM 走"先沿正则梯度方向扰动权重、再在扰动点求 loss"的两步式（公式 13），一阶展开后恰等于原 loss 减去该 Rényi 正则项。训练时还有两个稳定性 trick：先用纯 SGD warm-up（简单任务≤5 epoch，TinyImageNet 等难任务等验证 Top-1 超 30% 约第 20 epoch 再切换），以及用单一全局正则项作用于所有层（附录证明全局目标蕴含逐层目标），避免逐层调强度的组合爆炸。

### 损失函数 / 训练策略
RSAM 的训练目标是原分类 loss 加 Rényi 正则项，实现上通过公式 13 的两步扰动近似完成；阶数 $\alpha$ 按设计 3 规则取（CIFAR 类任务通常 0.5/1.5）；关键超参为扰动半径 $\rho$ 与 warm-up 长度。固定总 epoch 数（含 SGD warm-up），未额外补偿切换后的计算预算。

## 实验关键数据

### 主实验

**相关性实验**：在 ResNet18/34、Simple ViT，覆盖 CIFAR10/100、TinyImageNet，通过改变学习率/优化器/权重衰减生成大量不同极小点，用 Kendall rank 相关系数 $\tau$ 衡量锐度与泛化 gap（训练 loss 与测试 loss 之差）的相关性。Rényi sharpness 全面碾压旧度量（下表为 ResNet18-CIFAR10 各代表度量的 $\tau$）：

| 度量 | Kendall τ | 说明 |
|------|-----------|------|
| Rényi sharpness（逐层，最佳层） | +0.77 | 多数层在 0.6–0.78 区间 |
| trace $\mathrm{tr}(H)$ | -0.05 | 几乎无关 |
| parameter norm | -0.01 | 无关 |
| Fisher-Rao norm | -0.25 | 弱负相关 |
| PAC-Bayes flat | +0.13 | 弱 |
| SAM $\ell_\infty$ | -0.17 | 弱负相关 |
| ASAM $\ell_\infty$ | +0.20 | 弱 |

跨任务汇总（图 3）显示 Rényi sharpness 在 CIFAR10/100 ResNet、ViT、TinyImageNet 上 τ 普遍落在 0.6–0.9，而 trace、SAM、ASAM、PAC-Bayes 等大多在 ±0.3 内甚至变号。

**RSAM 训练实验**：作为正则化器与 SAM/ASAM/Eigen-SAM/FSAM/SSAM 对比，多数任务上取得最高测试精度：

| 数据集 | 模型 | SGD | SAM | ASAM | RSAM(本文) |
|--------|------|-----|-----|------|-----------|
| CIFAR10 | WRN-28-10 | 96.36 | 96.95 | 96.79 | **97.13** |
| CIFAR100 | ResNet56 | 72.60 | 74.86 | 75.20 | **75.71** |
| TinyImageNet | ResNet50 | 59.62 | 60.70 | 62.56 | **63.33** |
| CIFAR100 | ViT-B-16(微调) | 88.27 | 89.38 | 88.78 | **89.58** |

### 消融实验

| 分析维度 | 关键结果 | 说明 |
|----------|---------|------|
| 选阶规则匹配率 | 1451/1630 匹配 | 60 模型上经验最优 α 与谱形态预测规则高度一致 |
| α 取值 | 0.5（多簇）/ 1.5（均匀） | 跨数据集/模型稳健，错配会削弱相关性 |
| warm-up | 必要 | 训练早期直接加 Rényi 正则不稳定，需先 SGD 预热 |
| 全局 vs 逐层正则 | 全局 | 单一全局正则项即可，蕴含逐层目标，避免调参爆炸 |

### 关键发现
- Rényi sharpness 与泛化的相关性（τ≈0.6–0.9）远强于所有旧度量，且符号稳定、跨架构跨数据集一致——这是旧度量做不到的。
- 选阶规则不是拍脑袋：谱形态（多簇 vs 均匀）唯一决定 $\alpha$ 该大于还是小于 1，且 1630 案例统计支持该规则。
- RSAM 对 ASAM 的优势在部分任务上较小，作者归因于用了梯度幅值的近似 Hessian 而非精确 Rényi sharpness，留作改进方向。

## 亮点与洞察
- **把"谱的不均匀度"这一抽象直觉精确化为 Rényi 熵**：旧度量各执一端（trace 看中部、$\lambda_{\max}$ 看顶部），Rényi 熵一口气综合三类特征值的贡献，这是相关性大幅提升的根本原因。
- **重参数化不变性是理论与实证的双重支点**：它既让 Rényi sharpness 免疫 Dinh et al. 那类缩放攻击（实证上稳），又是推导泛化界的关键工具（理论上严），一个性质两头吃。
- **自由阶数 $\alpha$ + 谱形态选阶**是可迁移的思路：当一个度量需要"按数据分布形态调焦"时，用带参数的 Rényi 熵替代固定的 Shannon 熵/方差，并给出基于经验谱形态的选参规则，这套范式可搬到其他需要刻画向量不均匀度的场景。
- **SLQ + 梯度幅值近似**让一个看似昂贵的谱泛函真正落到训练里，是"理论度量 → 可训练正则"的务实桥梁。

## 局限与展望
- RSAM 用的是 Rényi sharpness 的近似（梯度幅值代 Hessian + 一阶展开），并非精确度量，作者承认这是它在某些任务上对 ASAM 优势有限的原因；用精确 Rényi sharpness 或更紧的估计器有望进一步提升。
- 泛化界依赖较强假设：数据可分（类间无重叠、可沿类内流形扰动）、Hessian 在极小点正定、激活正齐次（ViT 上只是近似），实际网络未必严格满足。
- 实验集中在图像分类（CIFAR/TinyImageNet/ViT 微调）中等规模，是否在大规模预训练、NLP、检测分割等任务同样强相关，尚待验证。
- 选阶规则把谱简化为"多簇/均匀"两类、$\alpha$ 二选一（0.5/1.5），较粗；介于两者之间或更复杂的谱形态如何选 $\alpha$ 留有空间。

## 相关工作与启发
- **vs trace 锐度 $\mathrm{tr}(H)$ / $\lambda_{\max}(H)$**：旧度量只反映谱的中部均值或顶部极值，信息损失大、与泛化弱相关甚至变号；本文用 Rényi 熵刻画整条谱的不均匀度，相关性从近 0 提到 0.6–0.9。
- **vs SAM / ASAM (Foret 2020 / Kwon 2021)**：SAM 系列优化最坏方向的扰动 loss，本质仍偏向顶部特征值；RSAM 把整条谱的 Rényi 熵作正则项，训练精度多数任务更高，且其锐度度量本身与泛化相关性远胜 SAM/ASAM 定义的锐度。
- **vs Eigen-SAM (Luo 2024) / Fisher-SAM (Kim 2022)**：前者专门正则顶部特征值、后者在 Riemannian 度量下做 SAM，仍是"局部视角"；本文是"全谱视角"，并在 Table 1 上整体领先。
- **vs PAC-Bayes / Fisher-Rao norm 等泛化度量 (Jiang 2019)**：这些在大规模研究里相关性也不稳；Rényi sharpness 在同等 Kendall τ 协议下显著更强更稳。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用 Rényi 熵把"谱不均匀度"定义成锐度，角度新且打通了直觉与现实的裂缝
- 实验充分度: ⭐⭐⭐⭐ 多架构多数据集的相关性 + 选阶统计 + RSAM 对比都有，但规模偏中等、未覆盖 NLP/大模型
- 写作质量: ⭐⭐⭐⭐ 逻辑链清晰（定义→理论→选阶→算法），理论与实证衔接好
- 价值: ⭐⭐⭐⭐⭐ 给"锐度预测不准泛化"这一长期争议提供了有理论保证、可估计、可训练的新答案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Quantitative Bounds for Length Generalization in Transformers](quantitative_bounds_for_length_generalization_in_transformers.md)
- [\[ICLR 2026\] Understanding the Dynamics of Forgetting and Generalization in Continual Learning via the Neural Tangent Kernel](understanding_the_dynamics_of_forgetting_and_generalization_in_continual_learnin.md)
- [\[ICLR 2026\] Provable Separations between Memorization and Generalization in Diffusion Models](provable_separations_between_memorization_and_generalization_in_diffusion_models.md)
- [\[ICLR 2026\] Pretrain–Test Task Alignment Governs Generalization in In-Context Learning](pretraintest_task_alignment_governs_generalization_in_in-context_learning.md)
- [\[ICLR 2026\] Bound by Semanticity: Universal Laws Governing the Generalization-Identification Tradeoff](bound_by_semanticity_universal_laws_governing_the_generalization-identification_.md)

</div>

<!-- RELATED:END -->
