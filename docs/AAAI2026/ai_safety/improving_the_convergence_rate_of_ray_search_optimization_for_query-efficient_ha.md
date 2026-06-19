---
title: >-
  [论文解读] Improving the Convergence Rate of Ray Search Optimization for Query-Efficient Hard-Label Attacks
description: >-
  [AAAI 2026 (Oral)][AI安全][硬标签攻击] 本文针对硬标签黑盒对抗攻击中的查询效率瓶颈，提出基于 Nesterov 加速梯度的动量算法 ARS-OPT，并引入代理模型先验得到增强版 PARS-OPT，在理论上证明了更快的收敛率，在 ImageNet 和 CIFAR-10 上超越 13 种 SOTA 方法。
tags:
  - "AAAI 2026 (Oral)"
  - "AI安全"
  - "硬标签攻击"
  - "黑盒对抗"
  - "查询效率"
  - "动量加速"
  - "代理模型"
---

# Improving the Convergence Rate of Ray Search Optimization for Query-Efficient Hard-Label Attacks

**会议**: AAAI 2026 (Oral)  
**arXiv**: [2512.21241](https://arxiv.org/abs/2512.21241)  
**代码**: 无  
**领域**: AI 安全 / 对抗攻击  
**关键词**: 硬标签攻击, 黑盒对抗, 查询效率, 动量加速, 代理模型

## 一句话总结

本文针对硬标签黑盒对抗攻击中的查询效率瓶颈，提出基于 Nesterov 加速梯度的动量算法 ARS-OPT，并引入代理模型先验得到增强版 PARS-OPT，在理论上证明了更快的收敛率，在 ImageNet 和 CIFAR-10 上超越 13 种 SOTA 方法。

## 研究背景与动机

**领域现状**：对抗攻击研究是 AI 安全的重要组成部分。黑盒攻击中，硬标签设置（hard-label setting）只能获取模型的 top-1 预测标签，是最具实际意义也最具挑战性的攻击场景。基于射线搜索（ray search）的方法通过寻找最优方向来最小化对抗扰动的 $\ell_2$ 范数，是硬标签攻击的代表性方法。

**现有痛点**：硬标签攻击的核心障碍是极高的查询复杂度——由于只能获得离散的标签信息（而非连续的梯度），每次估计梯度需要大量查询。现有的射线搜索优化方法收敛速度慢，需要数千甚至上万次查询才能找到足够小的对抗扰动。

**核心矛盾**：信息量极低（仅标签）vs 搜索空间极大（高维图像空间）——需要在极有限的信息下高效导航高维空间。

**本文目标**：（1）提升射线搜索优化的收敛速率；（2）减少达到同等攻击性能所需的查询次数；（3）保证理论收敛性。

**切入角度**：从优化理论出发，将 Nesterov 加速梯度（NAG）的动量思想引入射线搜索——利用累积动量预测未来梯度方向，加速收敛。

**核心 idea**：用动量加速的射线搜索优化（ARS-OPT）前瞻性地估计未来方向的梯度，再结合代理模型先验（PARS-OPT）进一步加速，实现更快的收敛和更少的查询。

## 方法详解

### 整体框架

攻击流程：给定良性图像 $x$ 和目标模型 $f$（只返回 top-1 标签），目标是找到最小 $\ell_2$ 范数的扰动 $\delta$ 使得 $f(x + \delta) \neq f(x)$。方法将此转化为在单位球面上搜索最优射线方向——每个方向对应一条从 $x$ 出发的射线，沿该射线寻找分类边界的最近点。优化目标是找到使边界点距离 $x$ 最近的方向。

### 关键设计

1. **动量加速射线搜索（ARS-OPT）**:

    - 功能：利用历史梯度信息加速当前方向的优化。
    - 核心思路：受 NAG 启发，在每一步不是直接估计当前方向的梯度，而是先用累积动量预测一个"未来"方向，然后估计在该未来方向处的梯度。具体而言，如果当前方向为 $d_t$，动量为 $m_t$，则在 $d_t + \beta m_t$（前瞻方向）处估计梯度，使得梯度估计更好地反映优化的走势。
    - 设计动机：标准射线搜索每步仅用当前信息决策，在曲率变化的区域容易振荡。动量累积了历史趋势信息，前瞻式梯度估计使优化路径更平滑、收敛更快。作者提供了理论分析证明 ARS-OPT 比标准射线搜索具有更好的收敛率。

2. **代理模型先验增强（PARS-OPT）**:

    - 功能：利用代理模型（如ResNet）的梯度信息加速搜索。
    - 核心思路：在 ARS-OPT 的梯度估计中融入代理模型的梯度作为先验（prior）。代理模型虽然与目标模型不同，但其梯度方向通常具有迁移性——提供了关于对抗方向的先验知识。具体实现为将代理梯度与差分估计的梯度加权平均。
    - 设计动机：纯粹依靠差分梯度估计噪声大且查询量高。代理模型先验提供了低噪的方向指引，减少了差分估计所需的查询次数。

3. **收敛性理论分析**:

    - 功能：为方法的优越性提供数学保证。
    - 核心思路：在标准假设下推导 ARS-OPT 和 PARS-OPT 的收敛率上界，证明其优于无动量的射线搜索方法。分析涵盖了动量参数选择和代理先验权重对收敛的影响。
    - 设计动机：硬标签攻击领域很多方法靠实验调参，缺乏理论指导。有理论保证的方法更可靠。

### 损失函数 / 训练策略

优化目标为最小化沿射线方向的边界距离：$\min_{d \in \mathbb{S}^{n-1}} g(d)$，其中 $g(d)$ 是方向 $d$ 对应的边界距离。使用差分方式估计 $g$ 的梯度。动量参数 $\beta$ 和学习率通过理论分析确定。

## 实验关键数据

### 主实验

| 数据集 | 指标 | PARS-OPT | 最佳基线 | 对比方法数 |
|--------|------|----------|----------|-----------|
| ImageNet | 查询效率($\ell_2$扰动) | 最佳 | 次优 | 超越13种SOTA |
| CIFAR-10 | 查询效率($\ell_2$扰动) | 最佳 | 次优 | 超越13种SOTA |

### 消融实验

| 配置 | 查询效率 | 说明 |
|------|---------|------|
| PARS-OPT | 最佳 | 动量+代理先验 |
| ARS-OPT | 次优 | 仅动量，无代理先验 |
| 无动量射线搜索 | 基线 | 标准方法 |
| 无动量+代理先验 | 中等 | 代理先验有帮助但不如动量明显 |

### 关键发现

- 动量加速是查询效率提升的主要来源——ARS-OPT 相比无动量版本有显著提升。
- 代理模型先验进一步降低了查询需求，PARS-OPT 取得最佳整体性能。
- 理论收敛率与实验结果一致，验证了分析的准确性。
- 方法在 13 种 SOTA 方法中取得最佳成绩，且以 Oral 形式被接收。

## 亮点与洞察

- **将 NAG 加速思想引入硬标签攻击**是理论与实践的优雅结合——用成熟的优化理论解决实际的攻击效率问题。
- **理论保证**在对抗攻击领域较为少见，增强了方法的可信度和可解释性。
- 方法论具有通用性——动量加速的差分梯度估计可以用于其他黑盒优化问题。

## 局限与展望

- 代理模型的选择对性能有影响，代理与目标模型差异过大时先验可能不准。
- 目前聚焦于 $\ell_2$ 范数攻击，对 $\ell_\infty$ 攻击的适用性有待验证。
- 对防御模型（如对抗训练模型）的攻击效率有待测试。
- 可以探索自适应动量参数——根据搜索历史动态调整动量大小。

## 相关工作与启发

- **vs SignOPT/HSJA**: 这些方法是经典射线搜索方法，本文在其基础上引入动量加速。
- **vs 基于迁移的攻击**: 迁移攻击利用代理模型梯度直接攻击，本文将代理梯度作为先验融入射线搜索，更加鲁棒。
- **vs 基于查询的梯度估计**: 标准差分估计每次需要很多查询，动量的历史信息积累减少了每步的查询需求。

## 评分

- 新颖性: ⭐⭐⭐⭐ NAG加速思想用于硬标签攻击新颖，理论分析扎实
- 实验充分度: ⭐⭐⭐⭐⭐ 超越13种方法，涵盖两个数据集，有理论支撑
- 写作质量: ⭐⭐⭐⭐ 理论与实验并重，Oral级别水平
- 价值: ⭐⭐⭐⭐ 对AI安全/对抗鲁棒性研究有重要贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Rethinking Target Label Conditioning in Adversarial Attacks: A 2D Tensor-Guided Generative Approach](rethinking_target_label_conditioning_in_adversarial_attacks_a_2d_tensor-guided_g.md)
- [\[ICCV 2025\] Membership Inference Attacks with False Discovery Rate Control](../../ICCV2025/ai_safety/membership_inference_attacks_with_false_discovery_rate_control.md)
- [\[AAAI 2026\] Easy to Learn, Yet Hard to Forget: Towards Robust Unlearning Under Bias](easy_to_learn_yet_hard_to_forget_towards_robust_unlearning_under_bias.md)
- [\[CVPR 2026\] SEBA: Sample-Efficient Black-Box Attacks on Visual Reinforcement Learning](../../CVPR2026/ai_safety/seba_sample-efficient_black-box_attacks_on_visual_reinforcement_learning.md)
- [\[CVPR 2026\] Improving Adversarial Transferability with Local Perturbation Augmentation](../../CVPR2026/ai_safety/improving_adversarial_transferability_with_local_perturbation_augmentation.md)

</div>

<!-- RELATED:END -->
