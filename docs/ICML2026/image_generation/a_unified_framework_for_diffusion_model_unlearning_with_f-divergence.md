---
title: >-
  [论文解读] A Unified Framework for Diffusion Model Unlearning with f-Divergence
description: >-
  [ICML 2026][图像生成][扩散模型] 这篇论文把扩散模型概念遗忘中的 MSE/KL 对齐推广到任意 $f$-divergence，提出 f-DMU 框架，并发现 closed-form Hellinger loss 往往比 MSE 更稳、更能保留非目标概念。 领域现状：文本到图像扩散模型可以生成高质量图像…
tags:
  - "ICML 2026"
  - "图像生成"
  - "扩散模型"
  - "概念擦除"
  - "模型遗忘"
  - "f-divergence"
  - "Hellinger距离"
---

# A Unified Framework for Diffusion Model Unlearning with f-Divergence

**会议**: ICML 2026  
**arXiv**: [2509.21167](https://arxiv.org/abs/2509.21167)  
**代码**: https://github.com/tonellolab/f-DMU  
**领域**: 图像生成 / 扩散模型遗忘  
**关键词**: 扩散模型、概念擦除、模型遗忘、f-divergence、Hellinger距离  

## 一句话总结
这篇论文把扩散模型概念遗忘中的 MSE/KL 对齐推广到任意 $f$-divergence，提出 f-DMU 框架，并发现 closed-form Hellinger loss 往往比 MSE 更稳、更能保留非目标概念。

## 研究背景与动机
**领域现状**：文本到图像扩散模型可以生成高质量图像，但也会记住 NSFW 内容、版权艺术风格、角色形象或个人信息。模型遗忘希望在不重训整个模型的情况下，从已训练模型中定向删除某个概念。

**现有痛点**：主流 fine-tuning 遗忘方法通常把 target concept 的 denoiser 输出拉向 anchor concept，例如空概念、父类概念或语义相近概念。这类目标常写成 MSE，本质上对应两个 Gaussian reverse-process 分布之间的 KL divergence。问题是，KL/MSE 只是一种 divergence 选择，不同任务可能需要不同的擦除强度和保真度权衡。

**核心矛盾**：强擦除容易伤害非目标概念和整体画质，温和擦除又可能保留 target 特征。现有方法缺少一个统一视角来解释“为什么某个 loss 更稳”以及“什么时候该用更激进的 loss”。

**本文目标**：作者希望把扩散模型概念遗忘写成一般 $f$-divergence 最小化问题，既覆盖原有 KL/MSE 方法，也提供一组可选择的 closed-form 和 variational loss。

**切入角度**：论文从概率分布而非具体网络结构出发，把原模型在 anchor concept 下的 reverse process 分布，与待遗忘模型在 target concept 下的 reverse process 分布对齐。

**核心 idea**：用 $f$-divergence 替代固定 KL divergence，让 divergence 的梯度形态控制遗忘过程中的稳定性、擦除强度和先验保持能力。

## 方法详解
f-DMU 的出发点是：许多概念擦除方法其实都在做“把 target 的生成分布改得像 anchor”。如果把原模型记为 $\Phi$，遗忘后的模型记为 $\hat{\Phi}$，那么可以比较 $p_{\Phi}(x_{t-1}|x_t,c)$ 与 $p_{\hat{\Phi}}(x_{t-1}|x_t,c^*)$ 之间的 divergence，其中 $c$ 是 anchor，$c^*$ 是要擦除的 target。

### 整体框架
论文把遗忘目标写成对时间步、样本、target-anchor pair 的期望：最小化 reverse-process 条件分布之间的 $D_f$。当两个条件分布可以近似为同协方差 Gaussian 时，一部分 $f$-divergence 有 closed-form，于是 loss 仍然像 MSE 一样便宜；当没有 closed-form 时，使用 variational representation，把问题写成 $\min_{\hat{\Phi}} \max_T$ 的 min-max 目标，由判别函数 $T$ 估计 divergence。

具体实例包括 KL/MSE、Jeffreys、squared Hellinger、Pearson $\chi^2$ 以及更一般的 $\alpha$-divergence。论文重点比较 closed-form Hellinger、closed-form $\chi^2$、标准 MSE/KL 和 variational losses。

### 关键设计
**1. $f$-divergence 统一遗忘目标：把各式概念擦除 loss 收进同一个分布对齐框架。** 主流方法其实都在做同一件事——把目标概念的生成分布改得像锚概念，并习惯性地写成 MSE。本文指出，这个 MSE 不过是两个高斯 reverse-process 分布之间的 KL，于是把目标推广为最小化任意 $f$-divergence $D_f$，KL/MSE 只是其中一个特例。关键观察是：换用不同的 $f$ 并不改变全局最优点（最优解仍是让两个分布对齐），但会改变优化路径和梯度幅度，也就是改变“擦除有多激进、先验保留得多好”。所以扩散遗忘真正该调的旋钮是分布迁移的强弱，而不是被默认锁死在 MSE 这一种几何上。

**2. closed-form Hellinger / $\chi^2$ loss：不加网络、不做 min-max，只换梯度形态。** 当目标与锚概念的两个条件分布近似为同协方差高斯时，一部分 $f$-divergence 有闭式解，loss 仍然像 MSE 一样便宜，差别全落在梯度缩放上。Hellinger loss 的梯度约为 $e^{-\text{MSE}}\nabla \text{MSE}$，会给大误差样本更小的权重，从而自然压住离群更新、减少 fine-tuning 途中图像突然崩坏；$\chi^2$ loss 的梯度约为 $e^{\text{MSE}}\nabla \text{MSE}$，反而放大大误差样本，适合需要更强擦除、可接受一定质量损失的场景。这组闭式 loss 因此排成一条“偏保真 ↔ 偏强擦除”的可选谱系，且都不增加训练成本。

**3. variational f-DMU：用变分形式把框架推广到没有闭式解的任意 $f$-divergence。** 不是每个 $f$-divergence 都有高斯闭式解；这时改用变分表示 $D_f(p\|q)=\sup_T \mathbb{E}_p[T]-\mathbb{E}_q[f^*(T)]$，引入一个判别函数 $T$ 去估计两个输出分布之间的散度，遗忘模型则最小化这个估计，整体写成 $\min_{\hat{\Phi}}\max_T$ 的 min-max 目标。代价是它需要额外训练判别函数，而且扩散 fine-tuning 常用的小 batch 下散度估计噪声更大、训练更激进、生成质量风险也更高——通用性是用稳定性换来的。

### 损失函数 / 训练策略
实验覆盖 Stable Diffusion 1.4、1.5、2.1、XL，并在附录扩展到 SD3 和 FLUX。评估概念是否被擦除使用 CLIP Score 和 CLIP Accuracy：对 target 概念越低越好，对保留概念越高越好；KID 用于衡量图像分布和质量变化。closed-form loss 与 MSE 一样只需单模型 fine-tuning，variational loss 需要额外训练判别函数。

## 实验关键数据

### 主实验
Van Gogh 风格擦除在 SD 2.1 上展示了 closed-form divergence 的主效果。

| 方法 | 擦除概念 CS↓ | 擦除概念 CA↓ | 保留概念 CS↑ | 保留概念 CA↑ | KID↓ |
|--------|------|------|------|------|------|
| ESD | 0.657 | 0.6 | 0.668 | 0.74 | 0.027 |
| CAbl | 0.635 | 0.2 | 0.668 | 0.78 | 0.028 |
| DoCo | 0.737 | 0.9 | 0.691 | 0.86 | 0.033 |
| Hellinger closed-form | 0.624 | 0.2 | 0.672 | 0.78 | 0.027 |
| $\chi^2$ closed-form | 0.628 | 0.1 | 0.672 | 0.76 | 0.028 |
| Hellinger variational | 0.645 | 0.5 | 0.702 | 0.88 | 0.051 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 梯度幅度：H-DMU | 梯度始终小于 MSE 和 $\chi^2$ | 大误差样本被指数衰减，fine-tuning 过程更平滑 |
| 梯度幅度：$\chi^2$-DMU | 梯度显著大于 MSE | 擦除更激进，但更容易损害保留概念和生成质量 |
| 10 个艺术风格顺序擦除 | H2 在多个 retained artist 上 KID 更低 | 多概念场景下 Hellinger 更利于 prior preservation |
| Nudity erasure: H-DMU | I2P 0.063, MMA Adv. 0.049, MMA S.Adv. 0.042 | 在主表中对非对抗和对抗提示均表现很强 |
| Nudity erasure: CAbl | I2P 0.120, MMA Adv. 0.118, MMA S.Adv. 0.141 | 标准 MSE 类方法在鲁棒性上弱于 H-DMU |
| Variational losses | 擦除快但 KID 往往更高 | 小 batch divergence 估计较粗，会造成更大分布扰动 |

### 关键发现
- Hellinger closed-form 是默认推荐：它通常保留非目标概念更好，且不增加训练开销。
- $\chi^2$ closed-form 更像“强擦除模式”：目标概念下降快，但会更明显改变周边分布。
- variational f-DMU 提供最大通用性和更激进擦除，但需要额外 min-max 训练，生成质量风险更高。
- 论文的理论梯度分析与实验现象一致，说明 divergence 选择确实改变了 fine-tuning 动力学，而不只是换了 loss 名字。

## 亮点与洞察
- 这篇论文最有价值的是把很多看似经验性的 unlearning loss 还原为 divergence choice，让方法选择有了可解释坐标系。
- Hellinger 的保守梯度不是手工调学习率，而是 per-sample 自适应缩放；这解释了为什么它能在保持 target 擦除的同时减少中途图像崩坏。
- anchor 选择和 divergence 选择是两个正交旋钮：anchor 决定 target 被替换到哪里，divergence 决定替换过程有多激进。

## 局限与展望
- 评估主要依赖 CLIP Score、CLIP Accuracy 和 KID，这些代理指标不能完全捕捉人类对“概念是否仍存在”的细粒度判断。
- 多概念、跨语言 prompt、组合概念和风格-对象纠缠场景仍可能让单一 divergence 难以稳定处理。
- Variational framework 虽然通用，但在扩散 fine-tuning 常见的小 batch 下噪声较大，需要更稳的估计或正则化策略。
- 后续可以把 f-DMU 与更好的 anchor selection、closed-form weight editing 或安全过滤结合，形成可控的模型治理工具链。

## 相关工作与启发
- **vs ESD / CAbl**: 它们使用 KL/MSE 类对齐，f-DMU 说明这些方法只是 $f$-divergence 家族中的特例，并给出更稳的 Hellinger 替代。
- **vs UCE / RECE / MACE**: 这些方法更多依赖结构化编辑或 closed-form 权重更新，f-DMU 保持 fine-tuning 框架，模型架构适用性更广。
- **vs DoCo**: DoCo 引入 GAN-like 变分思想，f-DMU 从 variational $f$-divergence 角度统一这类 min-max 目标，并指出其激进但质量风险更高。
- **启发**: 在模型遗忘中，不同任务应明确声明“偏保真”还是“偏强擦除”；loss 选择应服务于这个应用目标，而不是默认沿用 MSE。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用 $f$-divergence 统一扩散遗忘目标，理论和实践连接很强。
- 实验充分度: ⭐⭐⭐⭐☆ 模型、概念和攻击鲁棒性覆盖丰富；人类评估和真实合规场景还可加强。
- 写作质量: ⭐⭐⭐⭐☆ 数学动机清楚，但符号和附录表较多，读者需要一定扩散模型背景。
- 价值: ⭐⭐⭐⭐⭐ 对扩散模型安全、版权风格擦除和可控遗忘都有直接方法价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] GUDA: Counterfactual Group-wise Training Data Attribution for Diffusion Models via Unlearning](guda_counterfactual_group-wise_training_data_attribution_for_diffusion_models_vi.md)
- [\[ICCV 2025\] Holistic Unlearning Benchmark: A Multi-Faceted Evaluation for Text-to-Image Diffusion Model Unlearning](../../ICCV2025/image_generation/holistic_unlearning_benchmark_a_multi-faceted_evaluation_for_text-to-image_diffu.md)
- [\[CVPR 2026\] UniPercept: A Unified Diffusion Model for Generalizable Visual Perception](../../CVPR2026/image_generation/unipercept_a_unified_diffusion_model_for_generalizable_visual_perception.md)
- [\[ICML 2026\] Quantifying Error Propagation and Model Collapse in Diffusion Models](quantifying_error_propagation_and_model_collapse_in_diffusion_models.md)
- [\[CVPR 2025\] SVFR: A Unified Framework for Generalized Video Face Restoration](../../CVPR2025/image_generation/svfr_a_unified_framework_for_generalized_video_face_restoration.md)

</div>

<!-- RELATED:END -->
