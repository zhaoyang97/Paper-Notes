---
title: >-
  [论文解读] A Diffusive Classification Loss for Learning Energy-based Generative Models
description: >-
  [ICML2026][图像生成][扩散分类损失] 这篇论文提出 DiffCLF，把时间噪声层级之间的能量估计改写成分类问题，并与 DSM 联合训练，从而在不引入昂贵最大似然采样的情况下学习更可靠的能量函数，尤其改善了分数匹配在多模态权重上的模式盲区。 领域现状：扩散模型和 stochastic interpolants 通常…
tags:
  - "ICML2026"
  - "图像生成"
  - "扩散分类损失"
  - "能量模型"
  - "去噪分数匹配"
  - "模式盲区"
  - "Boltzmann Generator"
---

# A Diffusive Classification Loss for Learning Energy-based Generative Models

**会议**: ICML2026  
**arXiv**: [2601.21025](https://arxiv.org/abs/2601.21025)  
**代码**: https://github.com/h2o64/diffclf  
**领域**: 图像生成 / 扩散模型 / 能量模型  
**关键词**: 扩散分类损失、能量模型、去噪分数匹配、模式盲区、Boltzmann Generator  

## 一句话总结
这篇论文提出 DiffCLF，把时间噪声层级之间的能量估计改写成分类问题，并与 DSM 联合训练，从而在不引入昂贵最大似然采样的情况下学习更可靠的能量函数，尤其改善了分数匹配在多模态权重上的模式盲区。

## 研究背景与动机
**领域现状**：扩散模型和 stochastic interpolants 通常只学习 score，也就是对数密度关于输入的梯度。这样的训练目标很高效，因为不需要计算归一化常数，也不需要从当前模型中做内层 MCMC 采样；采样阶段只要把 score 塞进反向 SDE/ODE 就能生成数据。

**现有痛点**：很多下游任务并不只需要 score，而是需要能量本身。例如模型组合要把多个模型的密度做乘积或混合，Boltzmann Generator 要用 learned energy 做重加权和 SMC，free energy difference estimation 也依赖中间分布的 log-density。只靠 DSM 学到的能量只在每个连通模式内部可靠，对不同模式之间的相对权重并不敏感。

**核心矛盾**：最大似然能约束全局密度比例，但需要对 $p^\theta_t$ 采样；DSM 便宜，却只约束局部梯度，容易把“形状相同但混合权重不同”的分布看成几乎一样。作者要解决的是“既要学到可用于下游的全局能量，又不能退回昂贵的 EBM 最大似然训练”。

**本文目标**：第一，设计一个只用已有 noising/interpolation 样本即可计算的能量监督信号；第二，让它能和 DSM 兼容，而不是牺牲原本的生成质量；第三，在理论上解释为什么这个信号能恢复真实边缘分布，并在实验上证明它能改善组合、BG 和分子能量学习。

**切入角度**：扩散过程天然给出一串时间边缘分布 $p_{t_1},\ldots,p_{t_N}$。如果给定一个 noisy sample，让模型判断它来自哪个时间层级，那么分类器的 softmax logit 就必须比较不同时间的能量值；这种比较直接涉及 log-density 的相对高度，而不是只看梯度。

**核心 idea**：用“样本来自哪个扩散时间”的分类损失来监督 EBM 的能量尺度，再用 DSM 固定局部 score，使模型同时获得全局密度比例和局部生成动力学。

## 方法详解
这篇论文的出发点是一个统一的 noising 框架：给定可采样的随机过程 $X_t$ 和独立高斯噪声 $Z$，观测变量写成 $Y_t=X_t+\gamma(t)Z$。作者想学习每个时间 $t$ 上的边缘密度 $p_t(y)$，但只允许学习未归一化能量 $U^\theta_t(y)$ 与一个可学习的 free-energy/bias 项 $F^\theta_t$。如果直接最大似然，梯度里会出现模型分布下的期望，需要从当前 EBM 采样；DiffCLF 的核心就是绕开这个内层采样。

### 整体框架
训练时，先采样一组时间 $t_{1:N}$，再从对应边缘分布中产生 noisy samples $Y_{t_i}$。模型对同一个样本 $y$ 计算所有时间层级的 logit $-U^\theta_{t_j}(y)+F^\theta_{t_j}$，再做 $N$ 类 softmax，目标是预测它真实来自的时间 $t_i$。这相当于把每个 $p_t$ 当成一个类别条件密度，做多类 logistic regression。

DiffCLF 本身提供跨时间的 log-density 比较；DSM 则继续提供 score 监督。联合目标 $L_{DSM}+L_{clf}$ 的作用可以理解为：DSM 让每个时间切片内的能量斜率对齐，DiffCLF 让不同模式和不同时间之间的能量高度对齐。论文证明真实边缘分布是 DiffCLF 的最优解之一，而联合 DSM 后可以消除只乘上共同正函数导致的非唯一性。

### 关键设计
**1. 时间层级分类化的能量学习：把"估计密度"换成"判断它来自哪个时间"。** score matching 只看 $\nabla_y\log p_t(y)$，对两个模式位置相同、混合权重不同的多模态分布，score 几乎一模一样——局部梯度看不见全局比例，这正是模式盲区的根源。DiffCLF 把"估计 $p_t(y)$"改写成一个监督分类问题：给定噪声样本 $Y_{t_i}$，让模型判断它来自哪个时间边缘，用 $p^\theta(c=i\mid y)=\exp(-U^\theta_{t_i}(y)+F^\theta_{t_i})/\sum_j\exp(-U^\theta_{t_j}(y)+F^\theta_{t_j})$ 做交叉熵。由于 softmax 比较的是各时间层级在同一点 $y$ 上的能量值，模型被迫学会不同时间密度的相对高度；而分类 posterior 会随混合权重变化，于是把 score 看不见的全局密度比例补了回来。

**2. 与 DSM 联合而非替代：局部斜率交给 DSM，全局高度交给分类。** 只用 DiffCLF 仍存在非唯一解——所有时间密度同时乘上一个共同的正函数，分类 posterior 可能保持不变，能量形状就被"放飞"了。作者因此把它和去噪分数匹配（DSM）联合：DSM 约束 $-\nabla_y U^\theta_t(y)$ 逼近真实 score，固定每个时间切片内部的能量斜率；DiffCLF 约束不同 $t$ 之间的未归一化密度比，固定跨时间、跨模式的能量高度。两者合起来把能量的形状和相对高度都钉住，理论上唯一最优解回到真实 $p_t$，同时不牺牲扩散/插值模型原有的生成质量。

**3. 二分类与多分类两条计算路径：在监督丰富度和算力预算间留出旋钮。** 训练中能量网络的前向次数是主要成本，所以作者给出两个版本。多分类版本一次比较 $N$ 个时间层级，监督信息更丰富、靠更多层级降低估计方差、增强自一致性；二分类版本只比较一对 $(t,t')$，损失化简成两个 softplus 项，额外计算量约为 DSM 的 50%，让整体预算接近纯 DSM。论文还指出二分类极限与 time-score matching 存在联系——当时间采样过密时，它会退化回局部导数式的约束。

### 损失函数 / 训练策略
训练总目标是 $L_{DSM}+L_{clf}$。DSM 部分使用扩散模型或 stochastic interpolant 中已有的 denoising regression；DiffCLF 部分对每个 batch 采样多个时间层级，计算所有候选时间上的 energy logit 并做交叉熵。作者在伪代码里强调，为了和 DSM-only 做公平比较，DiffCLF 训练会调整 batch size，使 DSM 更新次数一致。对 diffusion model，还采用 Thornton/Karras 风格的 energy preconditioning，让零网络时对应一个简单高斯先验能量。

## 实验关键数据

### 主实验
论文的主实验覆盖合成高维高斯混合、stochastic interpolant、分子系统、模型组合、Boltzmann Generator 和自由能估计。下面选取最能体现 DiffCLF 价值的结果：在 MOG-40 上，它在不明显恶化 FD/MMD 的情况下大幅降低分类一致性损失；在分子系统上，它接近 FPE 正则的质量但训练更快；在 ALDP 自由能估计上，它比原始 Lbase 更接近参考值。

| 任务 / 数据集 | 指标 | DiffCLF / 本文 | 主要对比 | 结论 |
|--------|------|------|----------|------|
| MOG-40, DM, 128 维 | 分类损失 $L_{clf}$ | 4.40±1.00 | DSM 383.53±35.99；CtSM 20.86±4.93 | DiffCLF 显著修正跨模式/跨时间的能量比例 |
| MOG-40, DM, 128 维 | MMD ×100 | 3.54±1.34 | DSM 1.99±0.35；CtSM 5.20±0.34 | 生成质量保持在可比范围，目标不是牺牲采样换能量 |
| ALDP 分子系统 | Langevin PMF | 0.094±0.001 | DSM 1.047±0.924；FPE 0.104±0.004 | 用 learned energy 做 Langevin 时明显优于 DSM，略好于 FPE |
| Chignolin 分子系统 | 训练时间 | 18.9 GPU h | FPE 49.6 GPU h | 相比 FPE 正则更轻量，约 2.6 倍更快 |
| ALDP 溶剂化自由能 | 估计值 | 29.02±0.41 | Lbase 27.30±0.45；参考 29.43±0.01 | 加入 DiffCLF 后 TI 估计更接近参考 |

### 消融实验
论文没有传统视觉模型式的“去掉模块 A/B”消融，但给了多个替代训练目标和层级数分析。核心对照是 DSM-only、DSM+CtSM 与 DSM+DiffCLF：DSM 负责 score，CtSM 试图用 time-score 补充信息，而 DiffCLF 直接用分类 posterior 约束能量。

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| DSM only | MOG-40 128 维 $L_{clf}=383.53±35.99$ | score 和生成指标可以不错，但能量自一致性最差，体现模式权重盲区 |
| DSM + CtSM | MOG-40 128 维 $L_{clf}=20.86±4.93$ | time-score 约束有帮助，但仍依赖局部导数，不能完全解决全局比例 |
| DSM + DiffCLF | MOG-40 128 维 $L_{clf}=4.40±1.00$ | 分类目标直接比较能量高度，最能恢复正确 log-density |
| FPE regularization | ALDP train time 8.1 GPU h | 能学好分子能量，但要反传时间导数、score、Laplacian，成本高 |
| DiffCLF | ALDP train time 5.6 GPU h | 保持接近 FPE 的 Langevin JS/PMF，同时训练更快 |

### 关键发现
- DiffCLF 对“能量是否正确”比对“样本是否看起来正确”更敏感。MOG 实验里 DSM 的 MMD 并不总是很差，但分类损失和 log-density scatter 说明它没有学好密度高度。
- 在模式组合和 BG 任务中，DiffCLF 的优势会被放大，因为这些任务直接使用 learned marginal energy；DSM 的模式权重错误会导致组合分布比例明显偏掉。
- 分子系统结果说明 DiffCLF 不只是 toy loss：在 ALDP 和 Chignolin 上，用 $U^\theta_{t=0}$ 跑 Langevin dynamics 的统计量明显改善，同时比 FPE 正则更省训练时间。

## 亮点与洞察
- 这篇论文最巧妙的地方是把“扩散时间”当成分类标签。扩散训练本来就会采样不同噪声层级，DiffCLF 只是让模型回答这些样本来自哪个层级，却间接获得了能量比例监督。
- 它把 mode blindness 讲得很清楚：score 是局部斜率，不能可靠编码 disconnected modes 的质量比例；分类 posterior 依赖不同密度值的比值，因此天然能看见混合权重。
- 方法对现有 score-based 训练侵入很小。它没有要求放弃 DSM，也不需要从 EBM 内层采样，只是多做若干时间条件的前向评估，因此容易嵌入扩散模型、stochastic interpolants 甚至离散 CTMC。
- 下游实验选得比较到位。模型组合、Boltzmann Generator、free energy difference 都是“生成图片好不好看”以外更依赖能量数值的任务，能说明本文不是只在指标上做修补。

## 局限与展望
- 实验规模还偏中小。作者也承认图像建模的大规模 SMC composition 仍是未来方向；当前主要证据来自 MOG、分子系统和 toy composition。
- DiffCLF 虽然比最大似然便宜，但多分类版本仍需要 $N+1$ 次网络评估。对大模型，时间层级数、batch size 和训练吞吐之间还需要更系统的工程权衡。
- 方法依赖不同时间边缘分布之间有足够可分类性。如果时间采样过密，二分类极限会退化到 time-score matching 风格的局部约束，可能重新接近模式盲区。
- 学到的能量仍是未归一化的，虽然很多下游任务可接受这一点，但如果任务需要精确 normalizing constant，还需要额外估计或校准。

## 相关工作与启发
- **vs DSM / score matching**: DSM 学的是 score，训练便宜但对 disconnected modes 的相对权重不敏感；DiffCLF 通过分类 posterior 比较能量值，补上全局密度比例，同时继续保留 DSM 的局部 score 监督。
- **vs Conditional Time Score Matching**: CtSM 用可求的 conditional time-score 约束 $\partial_t\log p_t$，更像导数层面的修补；DiffCLF 直接在有限时间层级之间做密度比分类，因此对模式权重更直接。
- **vs Fokker-Planck regularization**: FPE 正则从边缘密度 PDE 出发，理论上强但计算重；DiffCLF 的监督信号来自普通交叉熵，实验中在分子系统达到接近或更好的统计质量，同时训练更快。
- **对后续工作的启发**: 很多生成模型训练其实有天然的“辅助标签”，例如时间、噪声强度、温度、退火层级。把这些标签转成密度比分类，可能是给 score model 补全能量尺度的一条通用路线。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把扩散时间层级分类用于 EBM 能量学习，抓住了 score matching 模式盲区的根因。
- 实验充分度: ⭐⭐⭐⭐ 覆盖合成、分子、组合、BG、自由能估计，但大规模图像实验仍不足。
- 写作质量: ⭐⭐⭐⭐ 理论动机和连接工作讲得清楚，附录细节很全；主文实验较密，需要读表格时来回对照。
- 价值: ⭐⭐⭐⭐⭐ 对需要 energy 而不只是 sample 的扩散模型应用很有价值，尤其适合组合生成和物理采样场景。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] $f$-Trajectory Balance: A Loss Family for Tuning GFlowNets, Generative Models, and LLMs with Off- and On-Policy Data](f-trajectory_balance_a_loss_family_for_tuning_gflownets_generative_models_and_ll.md)
- [\[AAAI 2026\] Symmetrical Flow Matching: Unified Image Generation, Segmentation, and Classification with Score-Based Generative Models](../../AAAI2026/image_generation/symmetrical_flow_matching_unified_image_generation_segmentation_and_classificati.md)
- [\[NeurIPS 2025\] Latent Zoning Network: A Unified Principle for Generative Modeling, Representation Learning, and Classification](../../NeurIPS2025/image_generation/latent_zoning_network_a_unified_principle_for_generative_modeling_representation.md)
- [\[ICML 2026\] A Kinetic Energy Perspective of Flow Matching](a_kinetic_energy_perspective_of_flow_matching.md)
- [\[CVPR 2026\] Transition Models: Rethinking the Generative Learning Objective](../../CVPR2026/image_generation/transition_models_rethinking_the_generative_learning_objective.md)

</div>

<!-- RELATED:END -->
