---
title: >-
  [论文解读] Completed Hyperparameter Transfer across Modules, Width, Depth, Batch and Duration
description: >-
  [ICLR2026][优化/理论][超参迁移] 作者把 μP/CompleteP 这套"小模型调参、大模型直接迁移"的缩放规则补全到四条最关键的训练轴——宽度、深度、批量、训练时长——并进一步证明：在正确的参数化下，连"逐模块"（每种张量、每层各自一组学习率/权重衰减/Adam 参数）的细粒度超参也能从 5000 万参数的小模型直接迁移到 72 亿参数的大模型，带来约 1.3× 的训练加速。
tags:
  - "ICLR2026"
  - "优化/理论"
  - "超参迁移"
  - "μP"
  - "CompleteP"
  - "SDE 缩放律"
  - "逐模块超参"
---

# Completed Hyperparameter Transfer across Modules, Width, Depth, Batch and Duration

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=elB9k4nTL1](https://openreview.net/forum?id=elB9k4nTL1)  
**代码**: 待确认  
**领域**: 优化 / 超参迁移 / 缩放律  
**关键词**: 超参迁移, μP, CompleteP, SDE 缩放律, 逐模块超参

## 一句话总结
作者把 μP/CompleteP 这套"小模型调参、大模型直接迁移"的缩放规则补全到四条最关键的训练轴——宽度、深度、批量、训练时长——并进一步证明：在正确的参数化下，连"逐模块"（每种张量、每层各自一组学习率/权重衰减/Adam 参数）的细粒度超参也能从 5000 万参数的小模型直接迁移到 72 亿参数的大模型，带来约 1.3× 的训练加速。

## 研究背景与动机
**领域现状**：大模型训练对超参（学习率、权重衰减、初始化尺度、AdamW 的 $\beta_1,\beta_2,\epsilon$）极其敏感，调错就可能训不收敛甚至崩掉。为了不在昂贵的大规模训练上反复试错，近年提出了一类"原则化参数化"——以 μP（Maximal Update Parameterisation）为代表，把无穷宽极限作为参照，让有限规模的模型都逼近同一个无病态的极限，从而**在小模型上搜到的最优全局超参可以直接搬到大模型**。Depth-μP、CompleteP 又把这套迁移从"宽度"推广到"深度"。

**现有痛点**：这套方法目前只解决了两件事的一半。第一，μP 及其变体的缩放规则**只对模型尺寸（宽度、深度）推导过**；但训练算力的另外两根杠杆——批量大小（batch size）和训练 token 数（训练时长）——同样会破坏超参迁移，沿这两根轴改训练配置，小模型的最优超参就不再是大模型的最优了。第二，迄今迁移的都是**全局超参**：整个模型共用一个学习率。但既然不同张量在 μP 下本就要按各自的架构角色乘不同的缩放因子（embedding 和 hidden 权重的学习率缩放方式天然不同），就没有理由相信它们的"基准最优值"应该在某个宽度下全部坍缩到同一个数——逐模块单独调参很可能还有可观的空间没挖。

**核心矛盾**：逐模块调参带来一个维度爆炸的难题。一个 Transformer block 里有十几种模块类型，再乘上层数，自由超参数量是 $|M|\times L$ 级别，直接在大模型上搜是彻底不可行的。于是问题变成：**能不能既享受逐模块调参的收益，又只在小模型上付出搜索成本，再靠参数化把结果迁移上去？**

**本文目标**：(1) 补全缩放规则，让超参迁移沿宽度、深度、批量、训练时长四条轴都成立；(2) 验证逐模块超参在正确参数化下也能迁移；(3) 给出一套在高维、布满"训练发散悬崖"的逐模块超参空间里真正能用的搜索配方。

**核心 idea**：提出 **Complete(d)P** 参数化——在 CompleteP（宽度+深度）基础上，补上基于随机微分方程（SDE）极限推导的批量与训练时长缩放规则，并用一个"深度×类型"的 Kronecker 因子化把逐模块超参压成 $|M|+L$ 个可迁移乘子，从而把"小模型上昂贵搜索 + 直接迁移"这条路打通到逐模块粒度。

## 方法详解

### 整体框架
整篇工作可以看成两层。**下层是参数化（缩放规则）**：把训练里那些需要随规模改变的设置——模型宽度 $m_N$、深度 $m_L$、批量比 $m_B$、token 数比 $m_D$——和我们真正想搜的超参（学习率 $\eta$、权重衰减 $\lambda$、$\beta_1,\beta_2,\epsilon$、初始化尺度、残差块乘子）分开。参数化规定了"当训练配置沿某根轴放大 $\kappa$ 倍时，每个超参该乘什么因子"，使得无论规模如何变化，模型行为都贴着同一个无穷极限走，于是**最优超参在不同规模下保持不变**，可以直接迁移。**上层是优化**：在一个 5000 万参数 / 16 亿 token 的小代理模型上，用信赖域随机搜索把逐模块超参搜到最优，再用下层的缩放规则一次性迁移到 72 亿参数 / 大 token 预算的目标模型，全程不在大模型上做任何二次调参。

整个迁移规则汇总在原文 Table 1 里：它给出 μP 与 Complete(d)P 对每一类张量（输入 embedding、hidden 权重、QK norm、unembedding 等）在初始化方差、学习率、AdamW $\epsilon$、权重衰减上各自的缩放因子，灰色部分标出了相对 CompleteP 的改动。下面按"宽度深度 → 批量 → 时长 → 逐模块"四块拆开讲。

### 关键设计

**1. Complete(d)P：把宽度/深度的参数化补全并修正**

宽度深度迁移的核心思想是把有限模型当作无穷极限的离散化：两个不同尺寸的模型只要都足够接近同一个无穷极限、且在所考察的超参范围内行为相近，它们就该共享相近的最优超参。难点在于不同参数化会导向不同的无穷极限，多数极限是病态的（标准参数化 SP 让特征随规模爆炸，NTK 参数化则没有特征学习），μP 正是被证明能在规模放大时排除这些病态的那个参数化。本文在 CompleteP（Depth-μP 对 Transformer 的改编）上做了三处修补：其一，把参数化**扩展到 QK 归一化层**——QK norm 在现代 Transformer 里已是标配，但它有别于任何其他组件，权重在各注意力头之间共享，当用"增加头数、固定头维度"的方式扩宽度时就引入了跨缩放维度的权重共享，需要单独的缩放考量；其二，**修正 CompleteP 对输入 embedding 的 AdamW $\epsilon$ 缩放错误**——改动虽小，但不修就足以让 Yang 等人的坐标检查（coordinate check）扫不过；其三，**消掉最终线性投影输出上的显式标量乘子**，把它的作用重参数化进学习率和初始化尺度里，这样就能无缝接入 Cut Cross-Entropy 这类不显式构造完整 logit 矩阵、能大幅省显存的算法。

深度迁移的关键是残差连接上那个依赖深度的重缩放因子：第 $\ell$ 层的输出满足

$$h_{\ell+1} = h_\ell + m_L^{-\alpha}\, F_\ell(h_\ell),\qquad \ell\in\{1,\dots,L\}$$

由单一参数 $\alpha\in[\tfrac12,1]$ 控制。本文的一个明确观察是：**Complete(d)P 在 $\alpha=\tfrac12$ 和 $\alpha=1$ 下都允许跨深度迁移**，覆盖了所有理论上合理的 $\alpha$ 取值；这与 CompleteP 报告的"$\alpha=0.5$ 时迁移退化"相反，作者把差异归因于自己加入的 QK norm 提升了稳定性（即便去掉 QK norm 也未出现对方报告的崩溃），并指出 CompleteP 的公开实现违背了自己的建议、对所有权重（含 embedding）用了同一个 $\epsilon$。有意思的是 $\alpha=\tfrac12$ 在最大模型上最终损失略好，暗示其"特征多样性"的理论动机在语言 Transformer 上可能确有收益。

**2. 批量缩放：把 AdamW 的 SDE 重参数化扩展到权重衰减**

批量大小在实践中受显存、并行架构、甚至月度算力波动牵制，调参时往往想用小批量以省显存、并行跑多组搜索，放大训练时又想用大批量吃满算力——可超参在批量轴上同样不会自动迁移。本文沿用 Malladi 等人的"把训练看成离散化一个 SDE"的思路：当"有效步长"（按已观测数据量归一化的时间里的步长）趋于无穷小时，离散的 Adam 迭代收敛到一个连续随机过程，找到能保证一致 SDE 极限的参数化，就能让超参跨批量迁移。作者用一个简化的 RMSPropW 例子推导：设每步梯度是固定方向加噪声 $g_k=g+\sigma e_k$，带权重衰减 $\lambda$ 的迭代

$$\theta_{k+1} = \theta_k - \eta\Big(\frac{g_k}{\sigma} + \lambda\theta_k\Big)$$

是步长为 $\eta^2$ 的 SDE $d\Theta_t = \tfrac{1}{\eta\sigma}(g+\lambda\sigma\Theta_t)dt + dW_t$ 的离散化。要保持迭代分布不变，需满足 $m_k m_\eta^2=1$、$m_\eta m_\sigma=1$、$m_\lambda=m_\eta$；当批量乘以 $\kappa$ 时 $m_\sigma=\kappa^{-1/2}$，于是得到**平方根缩放律**：

$$\eta' = \sqrt{\kappa}\,\eta,\quad k' = \kappa k,\quad \lambda' = \sqrt{\kappa}\,\lambda.$$

把 SDE 缩放律扩展到 **AdamW 的权重衰减**是本文自称的首创。作者还区分了两种实现：PyTorch 的 AdamW 把 $\lambda$ 乘了学习率，得到上式的 $m_\lambda=m_\eta$；而 Loshchilov-Hutter 原版（AdamLH）的迭代是 $\theta_{k+1}=\theta_k-\eta\tfrac{g_k}{\sigma}-\lambda\theta_k$，对应 $m_\lambda=m_\eta^2$——这意味着批量翻倍时权重衰减要翻倍，若不正确缩放，AdamLH 的漂移会比 PyTorch 版更大，作者推测这正是 PyTorch 实现更流行的原因之一。

**3. 训练时长缩放：等时间轴的 SDE 学习率衰减规则**

训练 token 数（训练时长）这根轴此前关注最少，但它是缩放算力的两大杠杆之一。作者观察到：**固定其他一切，最优学习率随训练迭代数衰减，且大致正比于 $1/\sqrt{\kappa}$**（$\kappa$ 为迭代数放大倍数）。把最短训练时长下的最优学习率 $\eta_{\text{opt}}$ 按 $\eta_{\text{opt}}/\sqrt{\kappa}$ 迁移，几乎完美对上真实最优；相比之下 Bjorck 等人是去拟合一条 $\eta_{\text{opt}}/\kappa^\beta$ 的缩放律、$\beta$ 落在 $0.3\text{–}0.7$，与本文规则吻合。本文进一步用 SDE 视角给出更优解释：固定批量、把学习率按 $1/\sqrt\kappa$ 缩放，等价于在 SDE 里**降低信噪比（SNR）而保持时间horizon不变**；他们独立观察到在模拟 AdamW SDE 时、其他参数不变只提升信噪比会一致地改善性能，于是提出"**等时间轴缩放规则**"：缩放 token 数时只调 SNR 参数、其余项不动。验证方式很巧——只靠增大批量来增加 token 数（这恰好只改 SNR），结果学习率近乎完美迁移，且由此得到的缩放律在拟合的缩放律上给出更低的渐近损失下界（Complete(d)P+SDE 的下界系数优于纯 CompleteP）。作者也诚实指出这可能与他们固定用的 cosine 调度有关，并通过一个 4842 次运行的贪心调度搜索发现：**短时长下的最优学习率退火曲线从来不是长时长最优曲线的前缀**，说明最优调度不是数据无关的、不能用贪心增量地拼出来。

**4. 逐模块超参的深度×类型 Kronecker 因子化与信赖域搜索**

逐模块调参的维度本是 $|M|\times L$，无法直接搜。作者把每个模块 $m$、深度 $\ell$ 的超参 $\zeta_{m,\ell}$ 在对数空间分解成四项相加：

$$\log\zeta_{m,\ell}(T,N,L,B) = \underbrace{\log\zeta^{\text{type}}_m}_{\text{类型}} + \underbrace{\log\zeta^{\text{depth}}_\ell(L)}_{\text{深度}} + \underbrace{\log\mathrm{SDE}(T,B)}_{\text{批量/时长}} + \underbrace{\log\mathrm{CP}_m(N,L)}_{\text{Complete(d)P}}$$

其中后两项（SDE 的批量/时长依赖、Complete(d)P 的宽度/深度调整）由前面的缩放律自动给出，真正要搜的只剩 $\zeta^{\text{type}}_m$ 和 $\zeta^{\text{depth}}_\ell$ 这些**无量纲、与时间无关的乘子**。这个因子化把自由乘子从 $|M|L$ 压到 $|M|+L$。迁移到更大深度 $L'$ 时，按 $\ell/L$ 对深度乘子做线性插值——只要基准超参随深度足够规则地连续变化，深度-SDE/ODE 极限就仍存在，于是有限深度的乘子可视为连续极限超参的离散化。

为什么不能用现成的搜索方法？因为逐模块超参的损失地形虽然**接近 invex**（驻点即全局最优），却布满麻烦：不同模块的"训练发散学习率阈值"能差好几个数量级；稳定/不稳定的边界形状复杂、模块间有非平凡耦合，边界甚至呈分形，简单的线性模型或高斯过程都拟合不了。这导致随机搜索（缺乏局部性偏置，搜索框稍微定不好就要么全发散、要么框不住最优）和标准贝叶斯优化（GP 在高度非平稳数据上失效）都频频失败。作者因此采用**信赖域随机搜索**——只在已知好解的邻域里搜，避开那些悬崖。

### 损失函数 / 训练策略
所有实验用 decoder-only Transformer（pre-norm + QK norm），在 RedPajama 上训练，损失是交叉熵加 Z-loss，统一用 cosine 学习率调度，性能指标统一报告预训练数据上的最终验证损失（被证明是下游性能的强指标）。逐模块超参在 5000 万参数 / 16 亿 token 尺度上、从最优全局超参出发，用信赖域随机搜索在对数空间联合优化逐模块的学习率、权重衰减、$\beta_1,\beta_2,\epsilon$、初始化尺度与残差块乘子。

## 实验关键数据

### 主实验

| 场景 | 对比 | 结果 |
|------|------|------|
| 50M / 1.6B token（搜索处） | 逐模块超参 vs 最优全局超参 | 达到同等性能快约 **2.3×** |
| 7.2B（迁移目标，约 1e4× FLOP） | 迁移的逐模块超参 vs 全局超参 | 保留 **1.32×** 加速 + benchmark 提升 |
| token 时长缩放律拟合 | Complete(d)P+SDE vs CompleteP | 损失下界系数更优（2.253 vs 2.482） |

宽度/深度迁移上，全局学习率在宽度（128→2048）与深度（4→128）放大时，最优学习率位置稳定对齐（$\alpha=\tfrac12$ 与 $\alpha=1$ 均成立）。批量迁移上，用平方根规则后学习率与权重衰减都能跨批量（$\tfrac18\times$ 到 $4\times$）迁移，不调整则明显漂移。逐模块超参的迁移则通过"穿过迁移后逐模块最优学习率与全局最优学习率的超平面切片"展示——该地形随模型尺寸保持稳定。

### 消融实验

| 配置 | 结论 |
|------|------|
| 去掉逐深度乘子（只按层角色搜） | 仍显著优于全局最优，但**差于含逐深度乘子**的版本 |
| 从 Kronecker 最优继续搜完全解耦的逐层学习率 | **几乎无提升**，说明大部分收益已被 Kronecker 因子化捕获 |
| 批量轴不做缩放 vs 平方根缩放 | 不缩放时学习率/权重衰减明显漂移、迁移失败 |

### 关键发现
- **逐模块收益的主体来自残差块内不同模块类型拿到不同学习率**，逐深度乘子是锦上添花但仍有可见贡献。
- **深度×类型的 Kronecker 因子化基本不损失性能**：完全解耦的逐层搜索几乎追不上它，说明最优逐模块学习率近似可被 Kronecker 结构表达——这正是把维度从 $|M|L$ 压到 $|M|+L$ 还能work的实证基础。
- **逐模块加速随规模缓慢衰减**（50M 的 2.3× 到 7.2B 的 1.32×），但作者坦言无法区分这是迁移在非渐近区不完美、还是无穷规模模型的渐近性质。

## 亮点与洞察
- **把"训练配置"和"超参"显式解耦**是整套方法的认知支点：前者是为了缩放而不得不改的（数据量、模型尺寸、批量），后者才是要搜的最优值，参数化的职责就是吸收前者对后者的影响。这个区分让"四条轴的缩放律可以正交叠加"成为可能。
- **SDE 视角同时解释了批量和时长两根轴**：批量缩放是保持 SDE 不变，时长缩放被重新理解为"只动信噪比、不动时间horizon"，并据此设计出"靠增大批量来增 token"的巧妙验证实验——把两个看似不同的现象统一进同一个连续过程框架。
- **逐模块损失地形的刻画本身就很有价值**：指出它接近 invex 但边界呈分形、模块间阈值差数量级，直接解释了为什么随机搜索和标准贝叶斯优化在这里会翻车、为什么得用信赖域方法——这对任何想做高维超参搜索的人都是可迁移的经验。
- **工程上的两个小修补很实在**：消掉最终投影的标量乘子以接入 Cut Cross-Entropy 省显存、修正 embedding 的 $\epsilon$ 缩放以通过坐标检查，都是"理论正确才能在大词表大模型上落地"的细节。

## 局限与展望
- **搜索效率仍有空间**：虽然用合理次数的试验就找到了改进的逐模块超参，但作者认为针对这种地形的信赖域贝叶斯优化、早停、利用训练过程结构的方法可能更试验高效。
- **只验证了单一训练设置**：自回归 Transformer + RedPajama，虽实用但缩放原则（尤其时长迁移规则、逐模块收益）理应在其他设置上再验证。
- **只用了 cosine 一种调度**：而 4842 次贪心搜索恰恰显示最优调度形状随 token 时长变化，固定调度可能并非最优；很多结论可能与该调度绑定。
- **加速随规模衰减的成因未明**：到底是迁移在有限规模不完美，还是无穷规模模型的固有性质，作者明确表示不知道，留给后续工作找计算可行的回答方式。

## 相关工作与启发
- **vs μP / Depth-μP**: μP 解决宽度迁移、Depth-μP 加上深度，但都只覆盖模型尺寸两轴、且只迁移全局超参；本文补上批量与时长两轴，并推进到逐模块粒度。
- **vs CompleteP（Dey et al., 2025）**: 本文直接以它为基座，但修正了 embedding 的 AdamW $\epsilon$ 缩放、扩展到 QK norm、消掉最终投影的标量乘子，并报告 $\alpha=0.5$ 下不出现对方所称的迁移退化。
- **vs Malladi et al. (2022) 的 SDE 缩放律**: 沿用其"训练即离散化 SDE"的框架，但**首次把 SDE 重参数化扩展到 AdamW 的权重衰减**，并用 SNR 视角推出训练时长的迁移规则。
- **vs Bjorck et al. (2025) 的时长缩放**: 对方拟合 $\eta_{\text{opt}}/\kappa^\beta$ 经验律（$\beta\in[0.3,0.7]$），本文从 SDE 原理直接给出 $1/\sqrt\kappa$，与其经验区间吻合但更有理论解释。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把超参迁移补全到宽度/深度/批量/时长四轴，并首次打通逐模块粒度迁移，AdamW 权重衰减的 SDE 缩放律为首创
- 实验充分度: ⭐⭐⭐⭐ 四轴迁移、逐模块消融、最高到 7.2B 规模都有验证，但只覆盖单一架构 + 单一数据集 + 单一调度
- 写作质量: ⭐⭐⭐⭐ 理论推导与工程修补都交代清楚，缩放规则汇总成表；信息密度高、部分推导需对照附录
- 价值: ⭐⭐⭐⭐⭐ 直接降低大模型预训练的调参成本，给出可落地的缩放规则与搜索配方，逐模块迁移在 1e4× FLOP 下仍带 1.3× 加速

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Wasserstein Transfer Learning](../../NeurIPS2025/optimization/wasserstein_transfer_learning.md)
- [\[ICLR 2026\] AutoEP: LLMs-Driven Automation of Hyperparameter Evolution for Metaheuristic Algorithms](autoep_llms-driven_automation_of_hyperparameter_evolution_for_metaheuristic_algo.md)
- [\[ICLR 2026\] Weight Decay May Matter More Than µP for Learning Rate Transfer in Practice](weight_decay_may_matter_more_than_µp_for_learning_rate_transfer_in_practice.md)
- [\[ICLR 2026\] ConRep4CO: Contrastive Representation Learning of Combinatorial Optimization Instances across Types](conrep4co_contrastive_representation_learning_of_combinatorial_optimization_inst.md)
- [\[ICLR 2026\] Saddle-to-Saddle Dynamics Explains A Simplicity Bias Across Neural Network Architectures](saddle-to-saddle_dynamics_explains_a_simplicity_bias_across_neural_network_archi.md)

</div>

<!-- RELATED:END -->
