---
title: >-
  [论文解读] Variational Deep Learning via Implicit Regularization
description: >-
  [ICLR 2026][学习理论][隐式正则化] 这篇论文提出 Implicit Bias VI（IBVI）：训练权重上的变分分布时**直接扔掉 ELBO 里的 KL 正则项**，只靠 SGD 自身的隐式偏置来"挑"分布；并在过参数化线性模型上严格证明，这种隐式偏置等价于以 **2-Wasserstein 距离**（而非 KL）为正则项的广义变分推断——既保留了标准神经网络的泛化能力，又免费拿到了校准良好的不确定性，且几乎不增加计算开销。
tags:
  - "ICLR 2026"
  - "学习理论"
  - "贝叶斯深度学习"
  - "隐式正则化"
  - "SGD 隐式偏置"
  - "变分推断"
  - "2-Wasserstein"
  - "不确定性量化"
---

# Variational Deep Learning via Implicit Regularization

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=WsN88Ns0i6](https://openreview.net/forum?id=WsN88Ns0i6)  
**代码**: https://github.com/inferno-ml/inferno  
**领域**: 学习理论 / 贝叶斯深度学习  
**关键词**: 隐式正则化, SGD 隐式偏置, 变分推断, 2-Wasserstein, 不确定性量化

## 一句话总结
这篇论文提出 Implicit Bias VI（IBVI）：训练权重上的变分分布时**直接扔掉 ELBO 里的 KL 正则项**，只靠 SGD 自身的隐式偏置来"挑"分布；并在过参数化线性模型上严格证明，这种隐式偏置等价于以 **2-Wasserstein 距离**（而非 KL）为正则项的广义变分推断——既保留了标准神经网络的泛化能力，又免费拿到了校准良好的不确定性，且几乎不增加计算开销。

## 研究背景与动机
**领域现状**：现代深度网络明明过参数化、几乎不加显式正则却能很好地泛化，主流解释是"隐式正则化"——架构、超参、优化器（尤其是 SGD）共同施加的归纳偏置在无数个零训练误差的全局最优里偏好了某一类解。这种偏置不需要额外计算就能生效。

**现有痛点**：但标准网络在分布外（OOD）数据上非常脆弱——预测过度自信、泛化急剧下降。贝叶斯深度学习用"模型平均"来缓解，可它有三个老毛病：先验难以设定（prior elicitation）、难以扩展到大模型、显式的先验正则配上近似推断常常产生病态的归纳偏置和不确定性。换句话说，**为了拿到不确定性，贝叶斯方法用一个显式先验正则把标准训练的隐式正则给"盖掉"了**。

**核心矛盾**：标准 VI 的目标函数 $\ell_r(\theta)=\mathbb{E}_{q_\theta}(-\log p(y\mid w)) + \mathrm{KL}(q_\theta\,\|\,p)$ 里，KL 项是一把双刃剑——它提供了对先验的正则，却也覆盖了 SGD 本身那套被证明对泛化至关重要的隐式偏置。能不能**不要这个显式 KL，而让优化器的隐式偏置来扮演正则角色**？

**切入角度**：作者注意到，过参数化下"损失最小"对应的不是一个点而是一整族解。如果直接最小化期望损失 $\bar\ell(\theta)=\mathbb{E}_{q_\theta}(\ell(y,f_w(X)))$（没有任何散度项），表面上看最优解会塌缩成一个 delta 点质量（退化、没有不确定性）；但在过参数化设定里，**delta 点质量只是众多最优之一**，到底收敛到哪个，由 SGD 的初始化与参数化决定。

**核心 idea**：用 SGD 的隐式偏置代替显式 KL 正则来训练变分网络——把变分分布初始化在先验，然后只最小化期望损失。理论上，这等价于在所有零训练误差的分布里选出**离先验 2-Wasserstein 距离最近**的那个，即广义变分推断。

## 方法详解

### 整体框架
IBVI 的整体思路是把"标准网络怎么训练"这套照搬到变分网络上，只是把点估计的权重换成权重上的高斯变分分布 $q_\theta(w)=\mathcal{N}(w;\mu, SS^\top)$。具体地，给定架构 $f_w$ 和变分族，训练目标退化为只优化期望损失（对比标准 VI 公式 (3)，把散度项 $\lambda D(q_\theta,p)$ 拿掉）：

$$\theta^\star \in \arg\min_\theta\ \mathbb{E}_{q_\theta(w)}\big(\ell(y, f_w(X))\big).$$

整条 pipeline 只有三件事：(1) 用先验初始化变分参数 $\theta_0=(\mu_0, S_0)$；(2) 用带动量的 SGD 最小化上面的期望损失，每次前向只采**一个**参数样本 $w_m\sim q_\theta$；(3) 训练收敛后得到 $q_{\theta^\star}$，预测时对权重分布做模型平均得到带不确定性的输出。关键在于：因为没有 KL，所有解都把训练数据插值到零误差（在训练点上不确定性塌缩，和标准网络一模一样）；而在偏离训练数据流形的方向上，分布会自然回退到先验，从而给出 OOD 不确定性。这套机制不是凭直觉，而是被第 4 节的两个定理严格刻画——这也是本文的核心贡献，因此它本质是一个**理论刻画 + 配套训练配方**，而非多模块串行 pipeline。

### 关键设计

**1. 去 KL 的期望损失训练：让 delta 塌缩不再是唯一最优**

针对的痛点是标准 VI 的 KL 项盖掉了 SGD 隐式偏置。做法上，作者把变分目标里的散度正则整个删掉，只留期望损失 $\bar\ell(\theta)=\mathbb{E}_{q_\theta(w)}(\ell(y,f_w(X)))$。乍看这有问题——目标在 $q_\theta=\delta_{w^\star}$（点质量于损失最小解）时取到最小，预测毫无方差，贝叶斯框架形同虚设。但作者点破：在过参数化（$P>N$）下，零训练误差对应一整族变分分布 $q_{\theta^\star}$，点质量只是其中之一，**究竟落到哪个由优化器的隐式偏置决定**。于是问题从"目标会不会塌缩"转成了"SGD 的隐式偏置把我们带到哪个最优"——这正好接到第 2 点的定理。这一设计的价值在于：它把训练流程拉回到和标准网络几乎一致（同一个 SGD、同一种损失），从而**继承**而非**覆盖**标准训练的隐式正则。

**2. 把 SGD 隐式偏置刻画为 2-Wasserstein 广义变分推断**

这是全文的理论支柱。作者在过参数化线性模型 $f_w(x)=x^\top w$、高斯先验/变分族下证明：若 SGD 从先验初始化 $q_{\theta_0}=p$ 出发，其隐式偏置等价于在所有最小化期望损失的解里，挑出离先验 2-Wasserstein 距离最近的那个：

$$q_{\theta^{\mathrm{GD}}_\star} = \arg\min_{q_\theta:\ \theta\in\arg\min\bar\ell(\theta)} W_2^2(q_\theta, p).$$

这正是广义变分推断目标 (3) 在"正则项取 $W_2^2$ 而非 KL"时的形式。**回归**（定理 1）下，这一刻画对 SGD 乃至带动量的 SGD 都成立，且 $q_{\theta^{\mathrm{GD}}_\star}$ 恰好等于"从先验独立初始化并训练的一组线性模型集成"的权重分布——这解释了实验里 IBVI 与 Deep Ensembles 表现相近。**二分类**（定理 2，线性可分、指数损失）下结论类似但更微妙：经过 $\theta^{\mathrm{rGD}}_t=(\frac{1}{\log t}\mu^{\mathrm{GD}}_t+P_{\mathrm{null}(X)}\mu_0,\ S^{\mathrm{GD}}_t)$ 的重缩放后，均值参数收敛到训练数据张成空间内的 **L2 最大间隔向量** $\hat w$（即硬间隔 SVM 解），在数据流形上不确定性塌缩为零，而在零空间（偏离数据流形处）由 $W_2$ 正则把分布拉回先验。为什么用 $W_2$ 而非 KL 是关键：对一个方差趋于零的高斯，KL 散度会发散到无穷，所以 KL 正则**绝不会**允许在训练点上不确定性塌缩；而 $W_2$ 允许塌缩，从而让 IBVI 的预测能像标准网络一样精确插值训练数据，又在远处保留先验不确定性。

**3. 变分版 μP：让超参可从小模型迁移到大模型**

隐式偏置依赖初始化和参数化，而参数化被列为"贝叶斯计算的重大挑战"之一。标准参数化（SP）下，模型变宽时最优学习率会漂移，必须对每个尺寸重新调参。作者把**最大更新参数化（μP）**扩展到变分网络：第 $l$ 层第 $i$ 个隐单元 $h^{(l)}_i(x)=(\mu_i+S_i z)h^{(l-1)}(x)$ 是变分均值/协方差参数、前向噪声 $z$、前层激活四者的函数。由于 $S_i z$ 是对 $S\in\mathbb{R}^{P\times R}$ 的 $R$ 项求和，用中心极限定理把这一项按 $R^{-1/2}$ 缩放，再对均值和协方差参数施加 μP（实现上落到协方差初始化和学习率的调整）。这样就把 μP 的"宽度无关特征学习 + 超参迁移"能力带进了概率模型——在小模型上调好的学习率可直接迁移到大模型，CIFAR-10 实验里 SP 的最优学习率随宽度下降、而 μP 保持不变。

**4. 单样本 + 低秩协方差：把开销压到接近标准网络**

为了让"拿到不确定性"几乎不额外花钱，作者做两件事。其一，前向只采**单个**参数样本（$M=1$）：这会让期望损失估计更噪（类似更小 batch），但只要把学习率调得足够小、或配合动量/多训几步，就能照常收敛——这样每步前向反向的开销和标准网络几乎一致。其二，协方差用**低秩分解** $\Sigma=SS^\top$（$S\in\mathbb{R}^{P\times R}$，$R\le P$），且只把输入层和输出层做成概率层。理论上也正好需要这种分解协方差，定理 2 才把 SGD 隐式偏置刻画为广义 VI；实践上把相对标准网络的内存开销压到约 10%、训练时间相近。

### 损失函数 / 训练策略
训练目标就是 minibatch 化的期望损失：

$$\bar\ell(\theta)\approx \frac{1}{N_b M}\sum_{n=1}^{N_b}\sum_{m=1}^{M}\ell\big(y_n, f_{w_m}(x_n)\big),\quad w_m\sim q_\theta(w).$$

实验统一用带动量 $\gamma=0.9$、batch $N_b=128$ 的 SGD 训 200 epoch，单精度，$M=1$，最后两层用低秩协方差，必要时配 μP。

## 实验关键数据

### 主实验（分布内泛化 + 不确定性）
在 MNIST / CIFAR10 / CIFAR100 / TinyImageNet 上对比标准网络与多种不确定性基线（Temperature Scaling、Laplace、Weight-space VI、SWAG、Deep Ensembles）。

| 维度 | IBVI 表现 | 对比与代价 |
|------|-----------|------------|
| 测试误差 | 与 SWAG 相当 | 只有 Ensembles 能提精度，但内存开销大得多 |
| NLL（似然） | 与 TS、DE 一同显著改善 | LA、WSVI 偶尔反而变差 |
| 校准（ECE） | 与 TS、DE 一同最优 | —— |
| 计算开销 | 内存仅 +≈10%，训练时间相近 | 显著小于 Ensembles / WSVI |

IBVI 与 Deep Ensembles 在分布内表现相近，这与理论一致（线性模型下二者等价，见 Proposition S1）。

### 鲁棒性（输入腐蚀 OOD）
在 MNISTC / CIFAR10C / CIFAR100C / TinyImageNetC（15 种腐蚀、取最大严重度后平均）上评测。

| 配置 | 关键发现 |
|------|----------|
| 精度 | 除 DE 外，IBVI 在腐蚀数据上的精度优于所有其他方法 |
| 不确定性（NLL/ECE）| TS、DE、IBVI 一致表现好，LA-ML 在 NLL 上有一定竞争力 |
| IBVI vs Ensembles | 在所有数据集上，IBVI 的 OOD 不确定性量化都**优于** Ensembles |

### 关键发现
- **去掉 KL 不会让不确定性塌缩**：只要正确初始化（设在先验）和参数化，SGD 隐式偏置会自动在训练数据外保留先验不确定性——这是反直觉但被定理保证的核心结论。
- **$W_2$ vs KL 的本质差别**：KL 对零方差高斯发散，永远不允许训练点上的不确定性塌缩；$W_2$ 允许塌缩，因此 IBVI 能精确插值训练数据又在远处回退先验。
- **超参迁移只对 μP 成立**：隐藏维度 >256 后，"小模型调参迁移到大模型"在 μP 下有效、在 SP 下失效。
- **最佳性价比**：IBVI 在校准/NLL/OOD 鲁棒性上几乎全面位居第一梯队，却几乎不增加内存与时间。

## 亮点与洞察
- **"减法"式创新**：别人在变分目标上加更精巧的正则，本文反其道把 KL 整个删掉，靠优化器自带的隐式偏置兜底——既省算力又省调参，思路非常干净。
- **把隐式偏置从点估计推广到分布**：已有结论说 SGD 选离初始化欧氏距离最近的极小点；本文把它升级为"选离先验 2-Wasserstein 最近的分布"，这是从非概率模型到概率模型的严格推广，理论价值高。
- **可迁移的 trick**：单样本 $M=1$ + 低秩协方差 + 只做首尾两层概率化，这套"几乎零开销拿不确定性"的配方可以直接搬到其他需要轻量 UQ 的网络上。
- **免去先验设定**：把先验只用于初始化，初始化后即可释放其超参内存，绕开了贝叶斯深度学习里最棘手的 prior elicitation。

## 局限与展望
- **理论只覆盖线性模型**：定理 1/2 严格成立于过参数化线性模型；深度网络上隐式正则更复杂，作者只能在实验上观察、未给出刻画。
- **协方差结构受限**：刻画依赖分解协方差 $\Sigma=SS^\top$；任意协方差参数化下 SGD 的隐式偏置仍是开放问题。
- **分类需要重缩放与额外假设**：定理 2 依赖指数损失、线性可分、支持向量张成数据等假设，并需对均值参数做 $1/\log t$ 重缩放才能收敛；推广到交叉熵、带动量 SGD 仍是猜想。
- **单样本训练需小学习率**：$M=1$ 时训练不稳，需调小学习率或加动量、多训步数来补偿。

## 相关工作与启发
- **vs 标准变分推断（mean-field VI / ELBO）**：他们用 KL 散度正则到先验，本文用 $W_2$ 正则且只靠 SGD 隐式施加；区别在于 KL 不允许训练点不确定性塌缩、$W_2$ 允许，因此 IBVI 更贴近标准网络的训练与泛化行为。
- **vs 广义变分推断 + Wasserstein 正则**：与本文理论最相关——本文证明 SGD 的隐式偏置**恰好实现**了 $W_2$ 广义 VI，从而无需显式计算正则项。
- **vs Deep Ensembles**：DE 用多次随机初始化独立训练，开销随集成数线性增长；IBVI 在线性模型下与 DE 等价，但只训一个变分模型、内存开销小得多，且 OOD 不确定性更好。
- **vs 非概率隐式偏置理论**：已有工作证明 SGD 选离初始化最近（欧氏）/最大间隔解，本文把这些结论提升到变分（概率）模型，并扩展了 μP 到概率网络。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "删掉 KL、用隐式偏置当正则"并给出 2-Wasserstein 广义 VI 的严格刻画，角度新且有理论深度
- 实验充分度: ⭐⭐⭐⭐ 覆盖 4 数据集 + 腐蚀 OOD + 多基线，但理论与深度网络实验之间存在鸿沟
- 写作质量: ⭐⭐⭐⭐⭐ 动机—理论—实验逻辑紧密，反直觉结论解释清楚
- 价值: ⭐⭐⭐⭐⭐ 近乎零开销拿到校准良好的不确定性，对落地贝叶斯深度学习很有吸引力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Variational Inference for Cyclic Learning](variational_inference_for_cyclic_learning.md)
- [\[ICLR 2026\] Diffusion Bridge Variational Inference for Deep Gaussian Processes](diffusion_bridge_variational_inference_for_deep_gaussian_processes.md)
- [\[ICLR 2026\] Implicit bias produces neural scaling laws in learning curves, from perceptrons to deep networks](implicit_bias_produces_neural_scaling_laws_in_learning_curves_from_perceptrons_t.md)
- [\[ICLR 2026\] Memory-Statistics Tradeoff in Continual Learning with Structural Regularization](memory-statistics_tradeoff_in_continual_learning_with_structural_regularization.md)
- [\[ICLR 2026\] Testing Fourier Sparsity via Implicit Sensing](testing_fourier_sparsity_via_implicit_sensing.md)

</div>

<!-- RELATED:END -->
