---
title: >-
  [论文解读] How to Cure Newton for Unlearning Neural Networks? An Empirical Study from the Hessian Perspective
description: >-
  [ICLR 2026][AI安全][机器遗忘] 本文发现牛顿型二阶遗忘（Newton unlearning）在真实神经网络/LLM 上会因 Hessian 退化（大量零/负特征值）而失效，提出基于三次正则化的 CuReNU 及其随机 Hessian-free 变体 CuReNUS，能自动确定阻尼因子 $\gamma$、保证收敛到二阶驻点，并在批量与序列遗忘乃至 LLM 规模上达到与 SOTA 经验方法相当的遗忘效果。
tags:
  - "ICLR 2026"
  - "AI安全"
  - "机器遗忘"
  - "二阶遗忘"
  - "牛顿法"
  - "Hessian 退化"
  - "三次正则化"
---

# How to Cure Newton for Unlearning Neural Networks? An Empirical Study from the Hessian Perspective

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=dHz2LBCyTh](https://openreview.net/forum?id=dHz2LBCyTh)  
**领域**: AI安全 / 机器遗忘  
**关键词**: 机器遗忘, 二阶遗忘, 牛顿法, Hessian 退化, 三次正则化

## 一句话总结
本文发现牛顿型二阶遗忘（Newton unlearning）在真实神经网络/LLM 上会因 Hessian 退化（大量零/负特征值）而失效，提出基于三次正则化的 CuReNU 及其随机 Hessian-free 变体 CuReNUS，能自动确定阻尼因子 $\gamma$、保证收敛到二阶驻点，并在批量与序列遗忘乃至 LLM 规模上达到与 SOTA 经验方法相当的遗忘效果。

## 研究背景与动机

**领域现状**：机器遗忘（machine unlearning）要让模型在不重训的前提下抹掉某些训练样本 $D_e$ 的影响，以满足 GDPR/CCPA 的"被遗忘权"或清除噪声/恶意数据。重训虽然精确但代价高昂（GPT-4 训练超 1 亿美元），因此主流转向**近似遗忘**——让遗忘后的模型尽量逼近重训模型的行为。其中**二阶遗忘**利用损失的一阶（梯度）和二阶（Hessian）信息，天然与影响函数（influence function）相连，并能提供"收敛到与重训相同损失"的理论保证，比经验启发式（最大化遗忘集损失、随机标签等）更严谨。

**现有痛点**：经典牛顿遗忘的更新式 $w_{t+1}=w_t-(H_{w_t}^{D_r})^{-1}g_{w_t}^{D_r}$ 默认 Hessian 可逆。但作者在 CNN×FMNIST、Llama-2×TOFU 上实测发现：训练收敛后 Hessian 的特征谱呈典型的"spiked model"——大量特征值堆在 0 附近（bulk），只有少数大特征值（spike），Hessian 秩随训练收敛快速塌缩，甚至出现负特征值（鞍点）。这使得 Hessian 不可逆，牛顿遗忘直接不可用，且 LLM 上退化程度更严重。

**核心矛盾**：常见补救手段都治标不治本。用伪逆（PINV-Newton），更新范数 $\|\Delta\|^2=\sum_{\lambda_i\neq0}\frac{1}{\lambda_i^2}(u_i^\top g)^2$ 会被近零特征值放大到爆炸；加阻尼（Damped Newton）$\|\Delta\|^2=\sum_i\frac{1}{(\gamma+\lambda_i)^2}(u_i^\top g)^2$，若 $\gamma$ 太小同样产生超大范数更新，导致冲过局部极小、遗忘性能崩溃（实验中两者准确率掉到 ~9%）。但更新范数关于 $\gamma$ 单调递减，意味着存在一个"恰到好处"的 $\gamma$。

**本文目标**：把问题重新表述为——如何**自动**找到一个 $\gamma$，既大到避免超大范数更新、又小到避免平凡更新拖慢收敛，从而让牛顿遗忘对 NN/LLM 真正有效，同时满足无需手调超参的可用性。

**切入角度**：作者借用优化中的**三次正则化牛顿法**（Nesterov & Polyak），它的二阶近似带一个 $\frac{L}{6}\|\Delta\|^3$ 的三次项，能给出全局上界，从而在非凸损失下也提供全局收敛保证；而其对偶变量恰好隐式定义了最优阻尼因子 $\gamma$。

**核心 idea**：用三次正则化替代裸牛顿/简单阻尼，让最优 $\gamma$ 由优化问题自动解出，并设计随机 Hessian-free 版本把方法推到 LLM 规模。

## 方法详解

### 整体框架
方法围绕一条主线：**给定一个已训练、Hessian 退化的模型 $w^*$，如何稳健地做二阶遗忘**。作者先把牛顿遗忘失败的根因锁定在 Hessian 退化上，再用三次正则化把"找最优阻尼 $\gamma$"变成一个可解的对偶优化问题（CuReNU），最后把昂贵的显式 Hessian 计算替换成 Hessian-向量积（HVP）的随机迭代（CuReNUS），从而可扩展到 LLM。整个流程从原模型出发，迭代地把参数从遗忘集 $D_e$ 上"拉开"、向重训损失 $L(w;D_r)$ 的二阶驻点收敛，输出遗忘后的模型。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["已训练模型 w*<br/>(Hessian 退化)"] --> B["Hessian 退化诊断<br/>裸牛顿/伪逆/阻尼都失效"]
    B --> C["CuReNU：三次正则化牛顿遗忘<br/>对偶求解最优 γ"]
    C -->|需显式 Hessian, O(d²) 不可扩展| D["CuReNUS：随机 Hessian-free 变体<br/>HVP + GD 内循环 + 扰动梯度"]
    D --> E["遗忘后模型<br/>逼近重训, ε-SOSP"]
```

### 关键设计

**1. Hessian 退化诊断：把牛顿遗忘失败归因到特征谱塌缩**

这一步回答"为什么现成的牛顿遗忘在真实网络上不灵"。作者通过实测特征谱（Observation 4.1）刻画训练后 Hessian 的结构：存在 $k\ll d$ 使得 $\lambda_i>0$ 仅对前 $k$ 个成立，其余 $\lambda_j\le0$，即大量零特征值、还可能有负特征值。这直接破坏了牛顿法的可逆性前提。更关键的是作者用范数公式定量解释了为何两个标准补救都不行：伪逆下 $\|\Delta_{t+1}\|^2=\sum_{i:\lambda_i\neq0}\frac{1}{\lambda_i^2}(u_i^\top g_{w_t}^{D_r})^2$，近零特征值会让某些项 $\gg0$；阻尼下 $\|\Delta_{t+1}\|^2=\sum_{i=1}^d\frac{1}{(\gamma+\lambda_i)^2}(u_i^\top g_{w_t}^{D_r})^2$，小 $\gamma$ 同样压不住。超大范数更新会冲过局部极小，造成遗忘后模型性能崩塌——这把后续设计的目标精确地定位为"如何选对 $\gamma$"。

**2. CuReNU：用三次正则化的对偶解自动求最优阻尼 $\gamma$**

针对"$\gamma$ 该取多少"的痛点，CuReNU 不再手调，而是最小化重训损失的**三次正则化近似**：

$$\tilde{L}(w_{t+1};D_r)=L(w_t;D_r)+\langle g_{w_t}^{D_r},\Delta_{t+1}\rangle+\tfrac{1}{2}\langle H_{w_t}^{D_r}\Delta_{t+1},\Delta_{t+1}\rangle+\tfrac{L}{6}\|\Delta_{t+1}\|^3$$

这个三次项让 $\tilde{L}$ 成为真实损失的**全局上界**，从而获得裸牛顿（仅二次近似、只能局部收敛甚至发散）拿不到的全局收敛性。但该问题不能直接用一阶条件求解（因为 $\|\Delta\|$ 未知会让更新 ill-defined），作者转而求其**强对偶**：引入对偶变量 $\alpha_{t+1}\triangleq\|w_{t+1}-w_t\|$，得到关于 $\alpha$ 的凸约束优化

$$\sup_{\alpha_{t+1}\in Q}\;-\tfrac{1}{2}\Big\langle\big(H_{w_t}^{D_r}+\tfrac{L}{2}\alpha_{t+1}I\big)^{-1}g_{w_t}^{D_r},\,g_{w_t}^{D_r}\Big\rangle-\tfrac{L}{12}\alpha_{t+1}^3$$

由于它在 $\alpha_{t+1}$ 上是凸的，可用现成的信赖域（trust-region）求解器高效解出。关键在于 $\alpha_{t+1}$ **隐式地定义了最优阻尼因子** $\gamma=\frac{L}{2}\alpha_{t+1}$，于是更新式 $w_{t+1}=w_t-(H_{w_t}^{D_r}+\frac{L}{2}\alpha_{t+1}I)^{-1}g_{w_t}^{D_r}$ 用的就是这个自适应阻尼。理论上 CuReNU 在 $O(\varepsilon^{-1.5})$ 次迭代内全局收敛到 $\varepsilon$-二阶驻点（$\varepsilon$-SOSP），比一阶方法只到 $\varepsilon$-FOSP 的保证更强。

**3. CuReNUS：随机 Hessian-free 变体，把方法推到 LLM 规模**

CuReNU 虽有快收敛保证，却要显式存 Hessian（$O(d^2)$ 空间）、算 Hessian 及其逆（$O(nd^2+d^3)$ 时间），在 LLM 上完全不可行（违反效率诉求）。CuReNUS 改用三次正则化的**随机近似**：在两个小批 $B_1,B_2\subset D_r$ 上估计随机梯度 $g^{B_1}$ 与随机 Hessian $H^{B_2}$，并用梯度下降解内层问题，因为 $\tilde{L}_\text{sto}$ 的梯度可通过 **HVP（Hessian-向量积）** 高效计算（Pearlmutter trick），从而**完全绕开显式 Hessian**。内层第 $s$ 步迭代为

$$\Delta_{s+1}=\Delta_s-\eta\big(\tilde{g}_{w_t}^{B_1}+H_{w_t}^{B_2}\Delta_s\big)$$

为应对三次正则化优化中的"hard case"（当 $\lambda_d<0$ 且 $\langle u_d,g^{B_1}\rangle=0$，梯度始终落在与 $u_d$ 正交的子空间、走不到最优方向），作者对梯度加微小扰动 $\tilde{g}^{B_1}=g^{B_1}+\sigma\zeta$（$\zeta\sim\text{Unif}(S^{d-1})$，$\sigma<1$ 以保留大部分一阶信息）。外层每轮重采新小批、跑 $T_\text{inner}$（约 5–10）步内层，共 $T_\text{outer}$ 轮。其内存仅 $O(2d)$（一个梯度 + 一个 HVP），相比已有 Hessian-free 方法的 $O(dn)$（随样本数线性增长）大幅降低，并在 $\tilde{O}(\varepsilon^{-3.5})$ 次梯度/HVP 评估内收敛到 $\varepsilon$-SOSP。

### 损失函数 / 训练策略
两个算法都以最小化**重训损失** $L(w;D_r)$ 为目标（作者论证最小化保留集损失等价于遗忘 $D_e$）。利普希茨常数 $L$（三次项系数）按数据集经验取值：FMNIST 用 5、CIFAR-10 用 50、AG-News 用 80、TOFU 用 400，并证明算法对 $L$ 的不同取值鲁棒。CuReNUS 取 $\sigma=0.1$、$\eta$ 与训练学习率相同；小批 $n_1,n_2$ 在 LLM 上取 10/5、其余取 128/64；序列遗忘时 $T_\text{outer}=10\text{–}20,\ T_\text{inner}=5$。收敛保证对比（非凸损失下）：

| 算法 | 收敛率 | 保证 |
|------|--------|------|
| GD | $O(\varepsilon^{-2})$ | $\varepsilon$-FOSP |
| SGD | $O(\varepsilon^{-4})$ | $\varepsilon$-FOSP |
| Newton | 局部二次 | $\varepsilon$-FOSP |
| CuReNU | 全局 $O(\varepsilon^{-1.5})$ | $\varepsilon$-SOSP |
| CuReNUS | 全局 $\tilde{O}(\varepsilon^{-3.5})$ | $\varepsilon$-SOSP |

## 实验关键数据

### 主实验
数据集/模型涵盖 CNN×FMNIST、ResNet-18×CIFAR-10、Llama-2-7B(+LoRA)×AG-News/TOFU。评测以"逼近重训"为准（与重训差距越小越好），用 ToW（Tug-of-War，越高越好）、JS 散度、Truth Ratio、MIA 等指标。

CNN×FMNIST 批量遗忘（样本级 / 类别级，越接近重训越好）：

| 方法 | 样本级 ToW (↑) | 类别级 ToW (↑) | 类别级 $D_e$ Acc (→重训 0) |
|------|------|------|------|
| Retraining | 1.00 | 1.00 | 0.00 |
| PINV-Newton | 0.01 | 0.05 | 1.44 |
| Damped Newton | 0.01 | 0.05 | 0.52 |
| SCRUB (SOTA) | 0.94 | 0.97 | 0.00 |
| DELETE (SOTA) | 0.85 | 0.99 | 0.00 |
| **CuReNU** | 0.98 | 0.93 | 1.37 |
| **CuReNUS** | **0.98** | **0.99** | 0.14 |

裸补救法（PINV/Damped）彻底崩溃，印证 Sec. 4 的范数分析：类别级遗忘中 PINV-Newton 更新范数高达 $3708.78$、Damped Newton $838.68$，而 CuReNU/CuReNUS 仅 $0.36/0.38$。CuReNUS 在两种设置都取得最佳/并列最佳 ToW，与 SOTA 经验方法相当甚至更优。

### 序列遗忘与效率
Llama-2×TOFU（样本级）与 ResNet-18×CIFAR-10（类别级）各 5 轮序列遗忘，最后一轮：

| 方法 | TOFU ToW (↑) | TOFU Truth Ratio (↑) | CIFAR ToW (↑) |
|------|------|------|------|
| Retraining | 1.00 | 0.658 | 1.000 |
| GD | 0.60 | 0.538 | 0.057 |
| SCRUB | 0.72 | 0.512 | 0.944 |
| NPO | 0.08 | 0.831 | 0.732 |
| **CuReNUS** | **0.80** | 0.591 | 0.909 |

效率（Table 4）：在 Llama-2×TOFU 上，重训需 900s，SCRUB 178s，而 CuReNUS 340s 但内存远低；CuReNU 在 CNN 上就要 6355s 且 $O(d^2)$ 内存，无法上 LLM——印证只有 CuReNUS 可扩展。

### 关键发现
- **退化诊断是全文支点**：更新范数从 ~3700 降到 ~0.36，定量证明"选对 $\gamma$"才是真正的病因与解法。
- 一阶方法（GD）表现为**欠遗忘**（$D_e$ 仍接近原模型），印证二阶信息对高保真逼近重训的必要性。
- CuReNU 与 CuReNUS 性能接近，说明随机 Hessian-free 近似几乎不损失遗忘质量，却换来 LLM 可扩展性。
- MIA 在正则化良好的模型上普遍区分度低，作者另在过拟合模型上补充验证。

## 亮点与洞察
- **把遗忘失败追溯到 Hessian 退化**：用特征谱 + 更新范数公式给出"为什么 PINV/阻尼都爆炸"的定量诊断，比泛泛说"牛顿法不稳定"深刻得多，是可复用的分析范式。
- **对偶变量即最优阻尼**：$\gamma=\frac{L}{2}\alpha_{t+1}$ 这一关系把"手调阻尼"变成"自动解出"，把优化里的三次正则化优雅地迁移到遗忘场景。
- **HVP + 扰动梯度解 hard case**：用 Pearlmutter trick 绕开显式 Hessian、$O(2d)$ 内存上 LLM，并用随机扰动跳出与负特征向量正交的子空间，这套组合可迁移到其他需要二阶信息的大规模场景。

## 局限与展望
- 三次项系数 $L$（Lipschitz Hessian 常数）实际无法精确求得，只能按数据集经验取值；虽证明鲁棒，但跨任务的取值仍需经验。
- CuReNU 因 $O(d^2)$ 内存只能用于小模型，真正上 LLM 的只有 CuReNUS；二者性能虽接近，但随机近似在更极端退化/更长序列下的稳定性仍待考。
- MIA 在常规模型上几乎失效，遗忘的隐私层面验证依赖过拟合模型这一特设场景，现实遗忘的隐私保证仍不够直接。
- 收敛保证基于光滑性假设（Assumption 3.1–3.3），与 LLM 真实损失景观的吻合度有限。

## 相关工作与启发
- **vs 经验近似遗忘（GA / SCRUB / NPO / DELETE）**：它们靠启发式（最大化遗忘集损失、加保留集正则等），缺乏严格收敛保证，在序列遗忘/LLM 等困难设置易退化；本文走二阶路线、有 $\varepsilon$-SOSP 全局收敛保证，且在这些设置上达到相当甚至更好的 ToW。
- **vs 经典牛顿/二阶遗忘（Guo et al. 2020；Golatkar et al. 2020）**：前作假设线性模型凸损失、Hessian 半正定可逆，本文指出该假设在 NN/LLM 上不成立，并用三次正则化正面解决 Hessian 退化。
- **vs 已有 Hessian-free 遗忘（Qiao et al. 2025）**：对方空间复杂度 $O(dn)$ 随样本数线性增长，CuReNUS 仅 $O(2d)$ 常数内存，可扩展性显著更好。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把"遗忘失败"系统归因到 Hessian 退化并用三次正则化自动求阻尼，视角新；但算法本身改编自已有优化方法。
- 实验充分度: ⭐⭐⭐⭐ 覆盖 CNN/ResNet/LLM、批量与序列遗忘、效率与多指标，较全面。
- 写作质量: ⭐⭐⭐⭐ 诊断—方法—验证逻辑清晰，公式与范数分析到位。
- 价值: ⭐⭐⭐⭐ 为二阶遗忘上 LLM 提供了可扩展且有理论保证的实用路径。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Designing Affine-Invariant Neural Networks for Photometric Corruption Robustness and Generalization](designing_affine-invariant_neural_networks_for_photometric_corruption_robustness.md)
- [\[ICLR 2026\] ATEX-CF: Attack-Informed Counterfactual Explanations for Graph Neural Networks](atex-cf_attack-informed_counterfactual_explanations_for_graph_neural_networks.md)
- [\[ICLR 2026\] Fisher-Rao Sensitivity for Out-of-Distribution Detection in Deep Neural Networks](fisher-rao_sensitivity_for_out-of-distribution_detection_in_deep_neural_networks.md)
- [\[ICML 2026\] Singular Bayesian Neural Networks](../../ICML2026/ai_safety/singular_bayesian_neural_networks.md)
- [\[ICLR 2026\] Robust Spiking Neural Networks Against Adversarial Attacks](robust_spiking_neural_networks_against_adversarial_attacks.md)

</div>

<!-- RELATED:END -->
