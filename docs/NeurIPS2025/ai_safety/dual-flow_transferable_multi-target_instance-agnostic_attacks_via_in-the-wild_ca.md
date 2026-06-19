---
title: >-
  [论文解读] Dual-Flow: Transferable Multi-Target, Instance-Agnostic Attacks via In-the-wild Cascading Flow Optimization
description: >-
  [NeurIPS 2025][AI安全][对抗攻击] 本文提出 Dual-Flow 框架，利用预训练扩散模型的正向 ODE 流和微调 LoRA 速度函数的逆向流进行多目标实例无关对抗攻击，通过级联分布偏移训练策略显著提升迁移攻击成功率（从 Inc-v3 到 Res-152 成功率提升 34.58%），在防御模型上也表现出强鲁棒性。
tags:
  - "NeurIPS 2025"
  - "AI安全"
  - "对抗攻击"
  - "黑盒迁移攻击"
  - "扩散模型"
  - "流匹配"
  - "多目标攻击"
---

# Dual-Flow: Transferable Multi-Target, Instance-Agnostic Attacks via In-the-wild Cascading Flow Optimization

**会议**: NeurIPS 2025  
**arXiv**: [2502.02096](https://arxiv.org/abs/2502.02096)  
**代码**: [github.com/Chyxx/Dual-Flow](https://github.com/Chyxx/Dual-Flow)  
**领域**: AI安全 / 对抗攻击  
**关键词**: 对抗攻击, 黑盒迁移攻击, 扩散模型, 流匹配, 多目标攻击

## 一句话总结
本文提出 Dual-Flow 框架，利用预训练扩散模型的正向 ODE 流和微调 LoRA 速度函数的逆向流进行多目标实例无关对抗攻击，通过级联分布偏移训练策略显著提升迁移攻击成功率（从 Inc-v3 到 Res-152 成功率提升 34.58%），在防御模型上也表现出强鲁棒性。

## 研究背景与动机

**领域现状**：对抗攻击分为实例特定和实例无关两类。实例无关方法通过学习数据分布级别的扰动，具有更好的黑盒迁移性。生成模型方法又分为单目标（需为每个目标类训练一个模型）和多目标（条件生成一个模型攻击所有类别）。

**现有痛点**：多目标生成式攻击面临模型容量限制导致迁移成功率低的问题；现有扩散模型用于攻击都是实例特定的（推理时需目标模型梯度）；ODE/SDE 采样选择缺乏理论依据。

**核心矛盾**：训练逆向流时无法获取中间时间步的真实分布（正向 ODE 轨迹是 in-the-wild 的），标准扩散训练算法不适用。

**本文目标** (a) 如何利用扩散模型做实例无关的多目标攻击？(b) 如何在无法访问中间分布时训练逆向流？

**切入角度**：将攻击分解为两个流——正向流（预训练扩散模型产生扰动分布）和逆向流（微调 LoRA 映射回约束空间）。

**核心 idea**：用预训练扩散模型的前向 ODE 产生中间表示，再用 LoRA 微调的速度函数逆向映射为受 l-inf 约束的对抗样本，通过级联优化渐进改善攻击效果。

## 方法详解

### 整体框架
输入图像 x 通过正向流映射到扰动分布 X_tau，再通过逆向流映射到 l-inf 约束空间。推理时完全不需要目标模型梯度。

### 关键设计

1. **正向流（Forward Flow）**:

    - 功能：将干净图像映射到中间扰动分布
    - 核心思路：使用预训练扩散模型的速度函数 v_phi，通过 ODE 积分从 t=0 到 t=tau
    - 设计动机：预训练扩散模型本身就能产生有结构的扰动分布，不需要额外训练

2. **逆向流（Reverse Flow）**:

    - 功能：将扰动分布映射为有效对抗样本
    - 核心思路：微调 LoRA 得到新速度函数 v_theta，通过 ODE 积分从 t=tau 到 t=0
    - 优化目标：最小化交叉熵 j = -CE(f(x), c)，其中 f 是源模型，c 是目标类别

3. **级联分布偏移训练 (Cascading Distribution Shift Training)**:

    - 功能：解决训练时中间时间步分布不可访问的问题
    - 核心思路 (Algorithm 1)：从 t=N 到 t=1 逐步回溯，每步先估计 x_0_hat，然后 clip 到约束范围，用交叉熵更新 theta
    - 理论保证 (Theorem 2)：级联改善性质——在时间步 t 更新 theta 后，时间步 t-delta 的交叉熵不会变差（delta 足够小时）
    - 设计动机：保证训练过程与采样过程一致

4. **Morse Flow 构造 (Proposition 1)**:

    - 核心理论：证明在 X_epsilon 和函数 j 的温和假设下，存在唯一光滑流 Phi，速度函数 v 几乎处处等于 alpha(x) * grad_x j(x)
    - 意义：保证沿梯度方向的流可改善攻击目标，流映射是微分同胚

5. **动态梯度裁剪与 ODE vs SDE 选择**:

    - 训练时对估计的 x_0_hat 做 clip + stop gradient
    - 级联 ODE 优于级联 SDE（随机项破坏级联关系）和随机 SDE（分布不匹配）

### 损失函数 / 训练策略
- 交叉熵损失 CE(f(x_0_hat), c)
- l-inf <= 16/255 扰动约束
- LoRA 微调减少参数量

## 实验关键数据

### 主实验：多目标攻击成功率 (%) — 正常训练模型

| 源模型 | 方法 | Inc-v3* | Inc-v4 | Res-152 | DN-121 | VGG-16 | 黑盒平均 |
|--------|------|---------|--------|---------|--------|--------|---------|
| Inc-v3 | C-GSP | 93.40 | 66.90 | 41.60 | 46.40 | 45.00 | 51.08 |
| Inc-v3 | CGNC | 96.03 | 59.43 | 42.48 | 62.98 | 52.54 | 52.80 |
| Inc-v3 | Dual-Flow | 90.08 | 77.19 | 77.06 | 82.64 | 67.09 | 73.96 |

### 防御模型攻击成功率 (%) — 源模型 Inc-v3

| 方法 | Inc-v3_adv | IR-v2_ens | Res50_SIN | Res50_Aug | 平均 |
|------|-----------|-----------|-----------|-----------|------|
| C-GSP | 20.41 | 18.04 | 6.96 | 21.95 | 24.28 |
| CGNC | 24.36 | 22.54 | 8.85 | 22.85 | 28.60 |
| Dual-Flow | 51.54 | 55.62 | 45.86 | 67.56 | 62.28 |

### 关键发现
- 黑盒迁移攻击成功率大幅提升：Inc-v3 -> Res-152 从 42.48% (CGNC) 提升到 77.06%，绝对提升 34.58%
- 对防御模型的攻击优势更大：平均成功率 62.28% vs CGNC 的 28.60%（+33.68%）
- 与单目标攻击相比，多目标版本仅低 ~3%，但省去为每个目标类单独训练的开销
- 级联 ODE 显著优于级联 SDE 和随机 SDE，验证了确定性轨迹的必要性

## 亮点与洞察
- 首次将 flow-based ODE 速度训练用于对抗攻击（区别于传统的 score function 训练），为扩散模型在安全领域的应用开辟了新方向
- 级联分布偏移训练的设计很巧妙——通过先前向积分再逐步反向优化，保证了训练与推理的一致性，且有理论支撑
- LoRA 微调使得模型仅增加极少参数就完成对抗适配，部署友好

## 局限与展望
- 需要源模型白盒训练（源模型梯度用于训练），迁移到目标模型是黑盒
- 实验仅在 ImageNet 分类任务上验证，未扩展到检测/分割等下游任务
- 扰动约束固定为 l-inf <= 16/255，未探索其他约束或更小扰动预算
- 正向流的时间步 tau 的选择可能需要调参

## 相关工作与启发
- **vs CGNC (2024)**: 同为多目标条件生成攻击，但 CGNC 用 UNet-GAN，Dual-Flow 用扩散 ODE + LoRA；黑盒迁移率 Dual-Flow 平均高出 20+%
- **vs C-GSP**: 也是生成式方法，但迁移率低于 CGNC 和 Dual-Flow

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将 flow-based 速度训练用于多目标对抗攻击，级联训练方法有创新
- 实验充分度: ⭐⭐⭐⭐ 覆盖正常/防御模型、多/单目标、ODE vs SDE 对比
- 写作质量: ⭐⭐⭐⭐ 理论与实验结合好，直觉解释清晰
- 价值: ⭐⭐⭐⭐ 显著推进了多目标迁移攻击的 SOTA，对模型鲁棒性评估有重要意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] AdvFM: Lookahead Flow-Matching Velocity-Field Attacks for Imperceptible and Transferable Adversarial Examples](../../CVPR2026/ai_safety/advfm_lookahead_flow-matching_velocity-field_attacks_for_imperceptible_and_trans.md)
- [\[CVPR 2025\] Lyapunov Stable Graph Neural Flow](../../CVPR2025/ai_safety/lyapunov_stable_graph_neural_flow.md)
- [\[CVPR 2026\] WaTeRFlow: Watermark Temporal Robustness via Flow Consistency](../../CVPR2026/ai_safety/waterflow_watermark_temporal_robustness_via_flow_consistency.md)
- [\[CVPR 2026\] FlowHijack: A Dynamics-Aware Backdoor Attack on Flow-Matching Vision-Language-Action Models](../../CVPR2026/ai_safety/flowhijack_a_dynamics-aware_backdoor_attack_on_flow-matching_vision-language-act.md)
- [\[NeurIPS 2025\] Towards Unsupervised Open-Set Graph Domain Adaptation via Dual Reprogramming](towards_unsupervised_open-set_graph_domain_adaptation_via_dual_reprogramming.md)

</div>

<!-- RELATED:END -->
