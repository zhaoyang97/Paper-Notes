---
title: >-
  [论文解读] Over-Alignment vs Over-Fitting: The Role of Feature Learning Strength in Generalization
description: >-
  [ICML2026][特征学习强度] 首次在标准分类任务里实证发现"特征学习强度（FLS）存在最优值"——既不是越大越好也不是越小越好——并用两层 ReLU 网络在 logistic loss 下的有限时间梯度流分析，把过大 FLS 引起的过拟合与过小 FLS 引起的"过对齐"分解为可量化的两个对立项，从而严格刻画最优 FLS 的存在性。
tags:
  - "ICML2026"
  - "特征学习强度"
  - "隐式偏置"
  - "神经元对齐"
  - "梯度流"
  - "过对齐"
---

# Over-Alignment vs Over-Fitting: The Role of Feature Learning Strength in Generalization

**会议**: ICML2026  
**arXiv**: [2602.00827](https://arxiv.org/abs/2602.00827)  
**代码**: 待确认  
**领域**: 深度学习理论 / 泛化与隐式偏置  
**关键词**: 特征学习强度, 隐式偏置, 神经元对齐, 梯度流, 过对齐

## 一句话总结
首次在标准分类任务里实证发现"特征学习强度（FLS）存在最优值"——既不是越大越好也不是越小越好——并用两层 ReLU 网络在 logistic loss 下的有限时间梯度流分析，把过大 FLS 引起的过拟合与过小 FLS 引起的"过对齐"分解为可量化的两个对立项，从而严格刻画最优 FLS 的存在性。

## 研究背景与动机

**领域现状**：理解过参数化神经网络为何能泛化是深度学习的核心谜题。学界一种主流解释是**隐式偏置**——梯度下降会偏好某些特定解，从而在没有显式正则的情况下也能挑出"好的" minimizer。其中 **feature learning strength（FLS）**——定义为模型输出有效缩放的倒数，可通过初始化尺度 $\alpha$ 或输出乘子 $c$ 控制——被广泛认为是决定学习动力学走向"特征学习 regime"还是"NTK / kernel regime"的关键旋钮。

**现有痛点**：现存理论几乎一致地告诉读者"更强的特征学习总是带来更好的泛化"，这一结论的证据多来自**极限分析**：要么 $\alpha \to 0$ 的 mean-field 极限，要么训练时间 $t \to \infty$ 的隐式偏置极限。可现实里的训练永远是有限时间 + 有限样本——通常在训练损失达到某阈值（或预算耗尽）时就早停。在这种"目标训练风险 $\eta$ 达成后立刻停"的实际语境下，理论的"越大越好"与工程上看到的"中等温度最香"形成明显矛盾。

**核心矛盾**：FLS 决定了两件相互拉扯的事——(i) FLS 越大（$\alpha$ 越小），权重在 Phase 1 越能精准对齐到经验类均值方向 $\mathbf{x}_+/\|\mathbf{x}_+\|$；(ii) 但经验类均值方向并不等于贝叶斯最优方向 $\mathbf{s}_+$，在有限样本下二者夹角 $\phi > 0$，过强对齐反而把预测器钉在一个偏离 Bayes 最优的方向上。这就是"过对齐"的本质。

**本文目标**：分解为两个研究问题——Q1：实证上 FLS 与泛化的关系是不是单调的？Q2：如果存在最优 FLS，其数学起源是什么？

**切入角度**：作者把"目标训练风险 $\eta$ 达成时刻"作为停止时间 $t_{\eta, \alpha}$，研究两层 ReLU 网络 + logistic loss 在 Gaussian mixture 数据下的梯度流。借助 min2024early、boursier2025early 关于 Phase 1 神经元对齐 ODE 的结果，把权重的角度偏差严格刻画为 $\alpha$ 的函数；再把超额误差分解为**过对齐项 $\mathsf{OA}(\alpha)$** 与**过拟合项 $\mathsf{OF}(\alpha)$**，发现两者随 $\alpha$ 反向单调，从而最优 FLS 必然存在于内部。

**核心 idea**：在有限时间训练范式下，泛化误差 = 过对齐 + 过拟合，二者随 FLS 反向变化，最优 FLS 来自这一权衡，理论与跨架构（VGG/ResNet）实证一致。

## 方法详解

### 整体框架

文章先做实证（Section 3），再做理论（Section 5）。实证部分用一个统一的"输出乘子 + 学习率"重参数化—— $f \mapsto cf$ 同时把 $\eta \mapsto \eta / c$，于是更小的 $c$ 等价于更大的 FLS；在 CIFAR-10/100 + BigGAN 合成数据上扫 $(c, \eta/c)$ 平面的测试精度热图，揭示"最优 FLS"的普遍存在。理论部分专攻两层 ReLU + logistic loss + 二分类 Gaussian mixture，把训练拆成两阶段：Phase 1 神经元对齐、Phase 2 margin 最大化；分别给出两阶段权重方向 $\psi_j(t)$ 与有效预测器方向 $\Psi(t)$ 关于 $\alpha$ 的下界，最后得到超额误差的上界分解。

### 关键设计

**1. FLS 的统一参数化与"训到 $\eta$ 就停"的停止时间：把理论拉回早停现实**

既往 FLS 理论几乎都讲渐近极限——要么 $\alpha\to0$ 的 mean-field、要么 $t\to\infty$ 的隐式偏置——和"训练损失降到某阈值就早停"的实际训练严重脱节，于是得出了"FLS 越大越好"的与工程直觉相悖的结论。本文先把 FLS 抽象成一个标量：既可用初始化尺度 $\mathbf{W}(0)=\alpha\mathsf{W}$ 控制，也可用输出乘子 $c$ 控制，并证明 $f\mapsto cf$ 同时把 $\eta\mapsto\eta/c$、二者在分析上等价。关键一步是把训练终点固定到训练损失第一次降到 $\eta$ 的时刻

$$t_{\eta,\alpha}:=\inf\{t\ge t_\alpha:\hat{L}_+(\theta_t)\le\eta\}$$

而不是 $t\to\infty$。这么做让不同 $\alpha$ 在相同 $\eta$ 处公平对比，剥离了"FLS 改变收敛速度"这一干扰，也正是后面把"过对齐 vs 过拟合"权衡显式化的前提——没有这个停止时间，两项就没法各自写成 $\alpha$ 的可微函数。

**2. 两阶段神经元对齐分析与角度下界：把"$\alpha$ 越小对齐越强"量化成 $\sqrt{\alpha}$ 标度**

训练被拆成 Phase 1（神经元对齐，长度 $t_\alpha=\Theta(\log(1/\alpha)/n)$）与 Phase 2（margin 最大化）。Phase 1 末，权重方向与经验类均值方向 $\mathbf{x}_+/\|\mathbf{x}_+\|$ 的内积有下界

$$\psi_j(t_\alpha)\ge\sqrt{\zeta(\alpha)}\tanh\big((t_\alpha-t_1)\|\mathbf{x}_+\|\sqrt{\zeta(\alpha)}\big),\quad \zeta(\alpha)=1-\frac{4\alpha n\sqrt{h}\,\mathbf{x}_{max}^2\mathsf{W}_{max}^2}{\|\mathbf{x}_+\|}$$

由此推出权重方向与 $\mathbf{x}_+/\|\mathbf{x}_+\|$ 的夹角与 $\sqrt{\alpha}$ 成正比（推论 5.3）。进入 Phase 2 后，借 conic-hull 性质把单神经元的对齐传递到有效预测器 $\hat{\mathbf{w}}_\alpha(t)$，并证明 $\Psi(t_{\eta,\alpha})\approx\Psi(t_\alpha)$——也就是说 Phase 2 几乎只继承 Phase 1 的对齐结果。这条 $\sqrt{\alpha}$ 角度下界是把超额误差写成 $\alpha$ 可微函数的骨架，也直接给出了后面最优 FLS 标度律的来源。

**3. 超额误差分解：过对齐 + 过拟合的反向单调**

这是全文最核心的概念创新。把超额误差写成两项之和 $\mathcal{E}(\hat{\mathbf{w}}_\alpha)-\mathcal{E}^*=\mathsf{OA}(\alpha)+\mathsf{OF}(\alpha)$，其中有效预测器被约束在圆锥 $H(\alpha)=\{\mathbf{v}\in\mathbb{S}^{d-1}:\langle\mathbf{x}_+/\|\mathbf{x}_+\|,\mathbf{v}\rangle\ge\Psi(t_{\eta,\alpha})\}$ 内。过对齐项 $\mathsf{OA}(\alpha)=\inf_{\mathbf{v}\in H(\alpha)}\mathcal{E}(\mathbf{v})-\mathcal{E}^*$ 度量"即便在锥内挑最优，仍偏离 Bayes 最优 $\mathbf{s}_+$ 多远"——$\alpha$ 越小、锥越收缩，锥内最优方向越偏离 $\mathbf{s}_+$，此项单调增大；过拟合项 $\mathsf{OF}(\alpha)=\mathcal{E}(\hat{\mathbf{w}}_\alpha)-\inf_{\mathbf{v}\in H(\alpha)}\mathcal{E}(\mathbf{v})$ 度量"锥内随机性带来的额外误差"——$\alpha$ 越大、锥越宽，候选解空间越大，此项也单调增大。两项随 $\alpha$ 反向单调，于是 FLS 太小会把预测器钉到一个偏离 Bayes 方向的窄锥（over-alignment）、太大让锥宽到容纳过多候选（over-fitting），二者的对立单调性在数学上保证了最优 FLS 必然存在于内部。

### 损失函数 / 训练策略

理论假设：(i) 数据 $\mathbf{x}_i = \kappa y_i \mathbf{s}_i + \sigma \mathbf{z}_i$，$\mathbf{z}_i \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$，二分类对称 Gaussian mixture；(ii) 训练集满足正交可分性 $y\tilde{y}\langle \mathbf{x}, \tilde{\mathbf{x}}\rangle / (\|\mathbf{x}\|\|\tilde{\mathbf{x}}\|) \geq \lambda$；(iii) 损失为 logistic loss，梯度流优化；(iv) 第二层权重初始化 $v_j(0) \sim \text{Unif}(\{\|\mathbf{w}_j(0)\|, -\|\mathbf{w}_j(0)\|\})$ 以利用平衡性性质。实证训练直接 SGD（无动量、无 augmentation、无 weight decay、无 lr scheduler），训到 train acc $\geq 99\%$ 比较 peak test acc。

## 实验关键数据

### 主实验

| 架构 | 数据集 | 默认 FLS ($c=2^0$) | 最优 FLS | 提升 |
|------|--------|-------------------|----------|------|
| ResNet-50 | CIFAR-100 | 53.57% | 59.76% ($c=2^{-4}$) | +6.19% |
| ResNet-18 | BigGAN edim=128 | 59.95% | 76.62% ($c=2^{-6}$) | +16.67% |
| VGG-19 / ResNet-18/34 | CIFAR-100 | 中等 | 内部最优 c | 普遍存在 U 形 |
| 5 层 CNN | BigGAN edim=128 | — | $c^* \propto n^{-2} h^{-1}$ | 实测与理论标度律一致 |

### 消融 / 鲁棒性

| 设定 | 结果 | 含义 |
|------|------|------|
| 训练风险作为停止准则 | 最优 FLS 存在 | 主结论成立 |
| 验证风险作为早停准则（Table 1） | 最优 $c$ 不变 | 不依赖具体停止方式 |
| 数据难度 edim 32 → 64 → 128 | 最优 FLS 收益从小到大 | 任务越难，调 FLS 越值得 |
| 跨宽度 / 跨数据集大小 | $c^* \propto n^{-2}h^{-1}$ | 最优 FLS 可跨规模迁移 |

### 关键发现
- **U 形泛化曲线在所有架构上都出现**：VGG-19、ResNet-18/34/50 在 CIFAR-100 上都呈现"中等 $c$ 最好、两端都差"的热图，说明这不是单一架构现象
- **任务越难，调 FLS 收益越大**：BigGAN 的 effective dimension 从 32 涨到 128，最优 FLS 相对默认 FLS 的精度差距从几个点扩到 16 点以上，意味着难任务上"是否调 FLS"几乎决定能不能用
- **理论标度律可迁移**：5 层 CNN 上扫宽度 $h$ 和数据量 $n$，最优输出乘子 $c^*$ 实测与理论预测的 $O(n^{-2}h^{-1})$ 高度吻合，提示 FLS 调参可像 $\mu$P 一样按规则迁移而非每次重扫
- **数值仿真验证分解**：对 $\mathsf{OA}(\alpha)$ 与 $\mathsf{OF}(\alpha)$ 直接数值评估（图 5），两条曲线确实呈反向单调，其和恢复了实际超额误差曲线，从而把理论分解从形式上的合法分解上升为实证可观察的双成分

## 亮点与洞察
- **理论概念的命名**：把"小 $\alpha$ 失效"命名为 over-alignment（过对齐），与传统 over-fitting 对位，让"为什么不能无限增大 FLS"变成一句话能讲清的几何故事——这种命名本身就是贡献，未来文献会反复引用
- **从极限到有限**：把 FLS 分析从渐近极限拉到"训到 $\eta$ 就停"的有限时间，是连接理论与工程的重要一步；停止时间 $t_{\eta, \alpha}$ 这一手术刀让 Phase 2 几乎只继承 Phase 1，简化了证明同时贴近实际
- **可操作的 takeaway**：明确建议把 FLS 列为正式超参数轴（与 lr、weight decay 同级），并给出 $c^* \propto n^{-2} h^{-1}$ 的预测标度律，对实际调参直接可用

## 局限与展望
- 理论严格依赖正交可分性假设（Assumption 4.1），对一般数据分布是否仍能保证 OA/OF 的反向单调性未证明；放宽到真实数据是关键开放问题
- 只覆盖两层 ReLU + 梯度流 + Gaussian mixture 的最小模型，未触及 BN、dropout、Adam、动量等真实训练成分；这些组件可能改变 Phase 1 ODE 的不动点
- 仅在视觉分类小模型上做了实验，是否在 Transformer / LLM 这种大模型 + 复杂数据上仍出现最优 FLS、且最优 $c$ 是否仍服从同一标度律是接下来最值得验证的方向

## 相关工作与启发
- **vs woodworth2020kernel / atanasov2025the**: 这些工作主张"更强的特征学习总是更好"，但都是渐近 / 在线设定；本文用"训到 $\eta$ 就停"的有限时间分析直接反驳，并给出最优 FLS 存在的几何机制
- **vs petrini2022learning**: 同样研究 FLS 与泛化，但只在球面回归任务和无限宽两个极端 regime 之间二选一；本文证明最优 FLS 在两个 regime 之间，且在标准分类任务上普遍成立
- **vs masarczyk2025unpacking / agarwala2023temperature**: 前作经验观察到 temperature scaling 存在最优值，但缺乏理论解释；本文用 OA/OF 分解给出了第一手严格框架，把经验观察提升为可预测的标度律

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ over-alignment vs over-fitting 的概念分解是真正的新洞察，且首次在标准分类下严格证明最优 FLS 存在
- 实验充分度: ⭐⭐⭐⭐ 多架构 + 多数据集 + 多停止准则 + 跨规模标度律验证，实证非常扎实
- 写作质量: ⭐⭐⭐⭐ 实证→理论→标度律的脉络清晰，几何图 4 把抽象证明可视化
- 价值: ⭐⭐⭐⭐⭐ 既改写隐式偏置文献的常识，又直接给出可操作的调参规则，对理论与工程双向有用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Revisiting Weak-to-Strong Generalization: Reverse KL vs. Forward KL](../../ACL2025/others/revisiting_weak-to-strong_generalization_in_theory_and_practice_reverse_kl_vs_fo.md)
- [\[ICML 2026\] Return-to-Go is More Than a Number: Q-Guided Alignment for Return-Conditioned Supervised Learning](return-to-go_is_more_than_a_number_q-guided_alignment_for_return-conditioned_sup.md)
- [\[ICLR 2026\] Noisy-Pair Robust Representation Alignment for Positive-Unlabeled Learning](../../ICLR2026/others/noisy-pair_robust_representation_alignment_for_positive-unlabeled_learning.md)
- [\[CVPR 2026\] Data-Centric Meta-Learning for Robust Few-Shot Generalization](../../CVPR2026/others/data-centric_meta-learning_for_robust_few-shot_generalization.md)
- [\[CVPR 2025\] TraF-Align: Trajectory-aware Feature Alignment for Asynchronous Multi-agent Perception](../../CVPR2025/others/traf-align_trajectory-aware_feature_alignment_for_asynchronous_multi-agent_perce.md)

</div>

<!-- RELATED:END -->
