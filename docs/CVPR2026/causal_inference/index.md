---
title: >-
  CVPR2026 因果推理论文汇总 · 4篇论文解读
description: >-
  4篇CVPR2026的因果推理方向论文解读，涵盖域适应、扩散模型等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2026"
  - "因果推理"
  - "论文解读"
  - "论文笔记"
  - "域适应"
  - "扩散模型"
item_list:
  - u: "a_polynomial_chaos_framework_for_causal_discovery_in_nonlinear_uncertain_systems/"
    t: "A Polynomial Chaos Framework for Causal Discovery in Nonlinear Uncertain Systems"
  - u: "cgu-bayes_causal_graph_uncertainty-guided_bayesian_inference_for_domain_generali/"
    t: "CGU-Bayes: Causal Graph Uncertainty-Guided Bayesian Inference for Domain Generalization"
  - u: "maskdime_adaptive_masked_diffusion_for_precise_and_efficient_visual_counterfactu/"
    t: "MaskDiME: Adaptive Masked Diffusion for Precise and Efficient Visual Counterfactual Explanations"
  - u: "retrieving_counterfactuals_improves_visual_in-context_learning/"
    t: "Retrieving Counterfactuals Improves Visual In-Context Learning"
item_total: 4
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔗 因果推理

**📷 CVPR2026** · **4** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (63)](../../ICLR2026/causal_inference/index.md) · [💬 ACL2026 (7)](../../ACL2026/causal_inference/index.md) · [🧪 ICML2026 (19)](../../ICML2026/causal_inference/index.md) · [🤖 AAAI2026 (7)](../../AAAI2026/causal_inference/index.md) · [🧠 NeurIPS2025 (20)](../../NeurIPS2025/causal_inference/index.md) · [📹 ICCV2025 (2)](../../ICCV2025/causal_inference/index.md)

**[A Polynomial Chaos Framework for Causal Discovery in Nonlinear Uncertain Systems](a_polynomial_chaos_framework_for_causal_discovery_in_nonlinear_uncertain_systems.md)**

:   把噪声项用多项式混沌展开（PCE）嵌进结构方程，得到 PCE-LiNGAM，证明在轻度稀疏条件下因果 DAG 可唯一辨识，并用「PCE 签名污染检验 + 递归找 sink」的多项式时间算法在极端非高斯工业数据上把平均 F1 从 0.50 提到 0.756，同时顺手给出基于 Sobol 指数的不确定性量化。

**[CGU-Bayes: Causal Graph Uncertainty-Guided Bayesian Inference for Domain Generalization](cgu-bayes_causal_graph_uncertainty-guided_bayesian_inference_for_domain_generali.md)**

:   针对"用结构因果模型（SCM）做领域泛化时、因果图在数据稀缺/含噪下估不准"的问题，本文不再点估计单一因果图，而是**对因果图的后验做贝叶斯推断**，从采样出的多张图里各选一套因果马尔可夫毯（CMB）特征训练预测器，再用每张图与测试样本的"对齐不确定性"当权重做加权集成，在 BLT、CMNIST 等强分布偏移数据集上拿到 SOTA。

**[MaskDiME: Adaptive Masked Diffusion for Precise and Efficient Visual Counterfactual Explanations](maskdime_adaptive_masked_diffusion_for_precise_and_efficient_visual_counterfactu.md)**

:   提出 MaskDiME，一个免训练的扩散框架，通过自适应双掩码机制将全局分类器引导转化为决策驱动的局部编辑，实现精确高效的视觉反事实解释，推理速度比 DiME 快 30 倍以上，GPU 内存仅为 ACE/RCSB 的十分之一。

**[Retrieving Counterfactuals Improves Visual In-Context Learning](retrieving_counterfactuals_improves_visual_in-context_learning.md)**

:   提出 CIRCLES 框架，通过属性引导的 composed image retrieval 检索反事实示例，构建因果+相关性双通道 in-context demonstration，显著提升 VLM 的细粒度视觉推理能力。
