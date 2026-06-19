---
title: >-
  [论文解读] The Power of Decaying Steps: Enhancing Attack Stability and Transferability for Sign-based Optimizers
description: >-
  [CVPR 2026][优化/理论][adversarial attack] 将 sign-based 对抗攻击优化器重构为坐标级梯度下降，揭示其非衰减步长是导致不收敛和不稳定的根因，提出单调递减坐标步长策略 MDCS，理论证明 MDCS-MI 达到最优 $O(1/\sqrt{T})$ 收敛率，在图像分类和跨模态检索任务上显著提升攻击迁移性与稳定性。
tags:
  - "CVPR 2026"
  - "优化/理论"
  - "adversarial attack"
  - "transferability"
  - "sign-based optimizer"
  - "step-size scheduling"
  - "convergence guarantee"
---

# The Power of Decaying Steps: Enhancing Attack Stability and Transferability for Sign-based Optimizers

**会议**: CVPR 2026  
**arXiv**: [2602.19096](https://arxiv.org/abs/2602.19096)  
**作者**: Wei Tao, Yang Dai, Jincai Huang, Qing Tao (国防科技大学, 军事科学院, 合肥工业大学)
**代码**: [AndssY/MDCS_attack](https://github.com/AndssY/MDCS_attack)  
**领域**: 优化  
**关键词**: adversarial attack, transferability, sign-based optimizer, step-size scheduling, convergence guarantee

## 一句话总结

将 sign-based 对抗攻击优化器重构为坐标级梯度下降，揭示其非衰减步长是导致不收敛和不稳定的根因，提出单调递减坐标步长策略 MDCS，理论证明 MDCS-MI 达到最优 $O(1/\sqrt{T})$ 收敛率，在图像分类和跨模态检索任务上显著提升攻击迁移性与稳定性。

## 研究背景与动机

对抗样本生成本质上是一个约束优化问题。Sign-based 优化器（I-FGSM、MI-FGSM、PGD）是当前对抗攻击的事实标准，但存在两个核心问题：

**理论缺陷**：Karimireddy et al. 已证明 sign-based 梯度即使在简单凸问题中也不能收敛到最优解，sign 算子丢弃了梯度幅度信息，扭曲了真正的下降方向

**实践不稳定**：违反直觉地，增加迭代次数反而可能导致攻击成功率急剧下降——I-FGSM 在迭代 $t=2$ 时成功率达 37.1%，到 $t=20$ 时骤降至 15.5%

**关键洞察**：将 I-FGSM 重构为坐标级梯度下降后，其第 $i$ 个坐标的步长为 $\alpha/|\partial J/\partial x_i|$。当攻击趋近局部最优时梯度趋零，步长会发散并剧烈波动，违反了优化理论中步长衰减的基本要求。MI-FGSM 虽通过动量积累缓解了单个梯度的波动，但同样存在非衰减步长的固有问题。

这一发现直接启发了从 AdaGrad/AMSGrad 的成功经验——单调递减坐标步长 (MDCS) 可修复 Adam 的收敛缺陷——迁移到对抗攻击优化器的方案。

## 方法详解

### 整体框架

对抗样本生成本质是在 $\ell_\infty$ 球内最大化分类损失的约束优化问题，而 I-FGSM、MI-FGSM、PGD 这类 sign-based 优化器是事实标准——每步只取梯度符号、按固定步长 $\alpha$ 前进。本文的整体思路是先换一个等价视角：把这些优化器看成"坐标级梯度下降"，从而暴露出它们藏在 sign 算子背后的真实步长其实是发散的；再把 AdaGrad/AMSGrad 修复 Adam 的那套招数搬过来，给每个坐标的步长套上"只能越来越小"的约束（MDCS），把原本不收敛、不稳定的老优化器改造成有收敛保证的新优化器。整条改造不碰动量、输入变换等原有组件，只替换"步长"这一个零件。

### 关键设计

**1. 把 sign 操作重写成坐标级梯度下降：让"隐藏步长"现形**

sign 算子之所以难分析，是因为它丢掉了全部梯度幅度信息，表面上步长是固定的 $\alpha$。本文把 I-FGSM 第 $i$ 个坐标的更新换个写法拆开看：

$$x_{t+1,i}^{adv} = x_{t,i}^{adv} + \frac{\alpha}{|\partial J / \partial x_i|} \cdot \partial J / \partial x_i$$

这一拆就看清了：真正作用在第 $i$ 个坐标梯度方向上的步长不是 $\alpha$，而是 $\alpha/|\partial J/\partial x_i|$，它与梯度幅度成反比。当攻击逼近局部最优时梯度趋零，这个等效步长就会发散并剧烈震荡，恰好违反了优化里"步长应随迭代衰减"的基本要求——这也正解释了为什么 I-FGSM 迭代到后期成功率会从 $t=2$ 的 37.1% 崩到 $t=20$ 的 15.5%。MI-FGSM 用动量平滑了单步梯度的抖动，但步长非衰减这个根子没动，问题依旧。

**2. MDCS：给坐标步长强行套上单调递减约束**

既然病根是步长不衰减，那就直接给它加一道闸。受 AMSGrad 用单调递减坐标步长（Monotone Decreasing Coordinate Step）修好 Adam 的启发，本文对每个坐标维护一个步长变量，每步只允许它变小、不许变大：

$$d_{t,i} = \min(1/|m_{t+1,i}|, \; d_{t-1,i})$$

这个 min 保证了 $d_{t,i} \leq d_{t-1,i} \leq 1$，于是无论梯度怎么波动，坐标步长都被钉成单调下降。以 MDCS-MI 为例，先用归一化梯度更新动量 $\mathbf{m}_{t+1} = \beta_t \mathbf{m}_t + \nabla J(\mathbf{x}_t^{adv}) / \|\nabla J(\mathbf{x}_t^{adv})\|_1$，再把上面的步长组成对角阵 $\mathbf{D}_t$ 代入参数更新：

$$\mathbf{x}_{t+1}^{adv} = \text{Clip}_\mathbf{x}^\epsilon[\mathbf{D}_t^{-1/2}(\mathbf{x}_t^{adv} + \alpha_t \mathbf{D}_t \mathbf{m}_{t+1})]$$

这一步把"发散的等效步长"换成了"受控收缩的步长"，是整篇方法治好不稳定优化器的核心手术。

**3. 衰减动量加 $O(1/\sqrt{T})$ 收敛保证：让改造有理论兜底**

光给步长加约束还不够稳，本文再让动量系数也随迭代衰减——取 $\beta_t = \beta\lambda^{t-1}$（$0 < \beta < 1$，$0 < \lambda < 1$），步长缩放设为 $\alpha_t = \gamma/\sqrt{t}$。在目标函数于约束域 $\mathbf{Q}$ 上局部凹、梯度有界 $\|\nabla J\|_1 \leq M$ 的假设下（Theorem 3），MDCS-MI 生成的平均迭代点满足：

$$J(\mathbf{x}^*) - J(\bar{\mathbf{x}}^{adv}_T) \leq O(1/\sqrt{T})$$

这是 sign-based 攻击优化器第一次拿到最优阶的收敛率保证，证明框架直接借用了 AdaGrad/AMSGrad 的分析。它的意义在于把"多迭代反而更稳"从经验现象变成了有理论预期的结果，让前面那条不稳定曲线有了可被修复的依据。

**4. 即插即用：只换步长这一个零件**

MDCS 不是一个完整的新攻击算法，而是一个能拧进任意 sign-based 攻击的步长模块——原方法里的动量、方差调优、输入变换、跨模态对齐统统保留，只把"取 sign 加固定步长"那一环换成 MDCS 的单调递减步长。所以它能近乎零改造成本地派生出图像分类侧的 MDCS-MI / MDCS-MEF / MDCS-OPS，以及跨模态检索侧的 MDCS-SGA / MDCS-DRA / MDCS-SAAET，相当于给一批现有攻击批量做收敛性升级。

## 实验关键数据

### 实验设置
- **图像分类**：NIPS2017 数据集，1000 张图像，$\epsilon = 16/255$，$T = 10$
- **代理模型**：ResNet-50 / ViT-B/16
- **目标模型**：CNN（VGG16, MobileNet-v2, Inc-v3）+ ViT（ViT-B, PiT-B, Vis-S）+ 防御模型
- **跨模态检索**：Flickr30K，$\epsilon = 8/255$，ALBEF/TCL/CLIP_CNN/CLIP_ViT

### Table 1: 单模型攻击迁移性（代理模型 Res50，攻击成功率 %）

| 类型 | 方法 | Res50 | VGG16 | Mob-v2 | Inc-v3 | ViT-B | PiT-B | Vis-S |
|---|---|---|---|---|---|---|---|---|
| ② | MI | 100.0 | 59.4 | 53.1 | 36.3 | 12.5 | 23.1 | 26.6 |
| ② | **MDCS-MI** | 100.0 | **67.2** | **60.3** | **41.3** | **13.4** | 23.3 | **30.8** |
| ② | MEF | 99.3 | 94.9 | 94.4 | 91.2 | 65.3 | 81.1 | 88.2 |
| ② | **MDCS-MEF** | 100.0 | **96.4** | **95.5** | **93.4** | 58.7 | 78.8 | **91.0** |
| ③ | OPS | 99.5 | 98.0 | 97.8 | 98.2 | 88.8 | 93.8 | 96.7 |
| ③ | **MDCS-OPS** | **99.9** | **98.9** | **99.0** | **99.1** | **89.3** | **94.7** | **97.9** |

MDCS 一致性提升：MDCS-MI 较 MI 在 VGG16 上提升 **+7.8%**，Inc-v3 上提升 **+5.0%**；MDCS-OPS 在所有目标模型上均刷新 SOTA。

### Table 4: 跨模态检索攻击（Flickr30K，代理 ALBEF，黑盒 R@1 %）

| 方法 | TCL TR | TCL IR | CLIP_CNN TR | CLIP_CNN IR | CLIP_ViT TR | CLIP_ViT IR |
|---|---|---|---|---|---|---|
| SGA | 87.67 | 87.88 | 38.04 | 46.17 | 41.63 | 50.36 |
| **MDCS-SGA** | **91.78** | **91.24** | **41.35** | **49.71** | **45.08** | **53.93** |
| DRA | 89.78 | 90.52 | 46.63 | 57.28 | 50.32 | 59.11 |
| **MDCS-DRA** | **93.26** | **92.98** | **49.94** | **59.31** | **55.56** | **62.44** |
| SA-AET | 96.31 | 96.19 | 54.23 | 63.50 | 58.88 | 65.18 |
| **MDCS-SAAET** | **96.52** | **96.71** | **60.25** | **67.01** | **60.54** | **67.89** |

跨模态场景中 MDCS 同样有效：MDCS-SAAET 较 SA-AET 在 CLIP_CNN TR 上提升 **+6.02%**，CLIP_ViT IR 上提升 **+2.71%**。在 CLIP 到 TCL 的跨架构迁移中，SA-AET 的 IR R@1 从 25.18% 提升至 33.40%（**+8.22%**）。

### 防御模型实验
在对抗训练和防御模型上，MDCS 同样有效。MDCS-OPS 在 8 个防御模型中的多数上取得最优，验证了 MDCS 在对抗场景下的鲁棒性。

### 稳定性验证
随着迭代数 $T$ 增大，MI-FGSM 的攻击成功率剧烈波动甚至下降，而 MDCS-MI 在不同 $T$ 下保持单调稳定的性能提升，验证了收敛理论的正确性。

## 亮点与洞察

- **优化视角的根因分析**：将 sign 操作重构为坐标级梯度下降，揭示步长非衰减是不稳定的本质原因，分析视角新颖且有说服力
- **理论与实践闭环**：首次为 sign-based 攻击提供 $O(1/\sqrt{T})$ 最优收敛保证，且实验中的稳定性提升与理论预测一致
- **即插即用**：MDCS 可无缝替换任意 sign-based 攻击中的步长策略，无需修改其他设计（动量、输入变换等），实用性极强
- **跨任务通用性**：在图像分类（CNN/ViT）和跨模态检索（VLM）两类任务上均有效，覆盖单模型攻击、集成攻击、防御模型攻击多个场景

## 局限性

- **局部凹性假设**：理论分析依赖目标函数在约束域上的局部凹性，虽然作者论证了该假设的合理性，但在高度非凸损失曲面上的保证仍有限
- **超参数 $\gamma$ 需调**：步长缩放因子 $\gamma$ 需在 [2, 4] 范围内网格搜索，增加了使用成本
- **仅限非定向攻击**：实验仅评估了非定向攻击，定向攻击场景下的效果未验证
- **对已强方法的增益递减**：当基线方法已非常强时（如 OPS 在部分模型上接近 99%），MDCS 的提升空间有限

## 相关工作

- **Sign-based 攻击**：FGSM → I-FGSM/PGD → MI-FGSM → VMI/GRA/PGN/MEF/OPS，逐步引入动量、方差调优、输入变换等技术，但均保持固定步长
- **自适应步长优化**：AdaGrad 引入 MDCS 解决稀疏学习，AMSGrad 用 MDCS 修复 Adam 的收敛缺陷 → 本文将此思路迁移到对抗攻击
- **VLM 攻击**：Co-Attack → SGA → DRA → SA-AET，策略越来越复杂但底层仍依赖 PGD → MDCS 可作为通用升级模块
- **GRA 的 decay indicator**：GRA 已观察到 sign 扰动的频繁翻转，提出 decay indicator 动态调步长，但缺乏理论保证 → MDCS 提供了更系统的解决方案

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将优化理论（MDCS/AMSGrad）与对抗攻击连接的视角新颖，重构分析有洞察力
- 实验充分度: ⭐⭐⭐⭐⭐ — 覆盖图像分类+跨模态检索，CNN+ViT+VLM，正常模型+防御模型，稳定性+消融实验全面
- 写作质量: ⭐⭐⭐⭐ — 动机清晰，理论推导严谨，实验部分组织良好
- 价值: ⭐⭐⭐⭐ — 即插即用的通用攻击增强策略，有理论保证，对对抗鲁棒性研究有直接参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Unveiling the Power of Multiple Gossip Steps: A Stability-Based Generalization Analysis in Decentralized Training](../../NeurIPS2025/optimization/unveiling_the_power_of_multiple_gossip_steps_a_stability-based_generalization_an.md)
- [\[ICML 2026\] AdaGC: Enhancing LLM Pretraining Stability via Adaptive Gradient Clipping](../../ICML2026/optimization/adagc_enhancing_llm_pretraining_stability_via_adaptive_gradient_clipping.md)
- [\[AAAI 2026\] Cost-Minimized Label-Flipping Poisoning Attack to LLM Alignment](../../AAAI2026/optimization/cost-minimized_label-flipping_poisoning_attack_to_llm_alignment.md)
- [\[CVPR 2026\] Enhancing Visual Representation with Textual Semantics: Textual Semantics-Powered Prototypes for Heterogeneous Federated Learning](enhancing_visual_representation_with_textual_semantics_textual_semantics_powered_p.md)
- [\[ICML 2026\] On the Expressive Power of GNNs to Solve Linear SDPs](../../ICML2026/optimization/on_the_expressive_power_of_gnns_to_solve_linear_sdps.md)

</div>

<!-- RELATED:END -->
