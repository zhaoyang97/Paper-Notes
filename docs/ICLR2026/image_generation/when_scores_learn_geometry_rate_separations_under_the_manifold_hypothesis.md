---
title: >-
  [论文解读] When Scores Learn Geometry: Rate Separations under the Manifold Hypothesis
description: >-
  [ICLR 2026][图像生成][Score Learning] 在流形假设下揭示score学习中几何信息与分布信息的尺度分离现象——流形几何信息强度为 $\Theta(\sigma^{-2})$，比分布信息强 $O(\sigma^{-2})$ 倍，由此证明扩散模型的成功主要来自学习数据流形而非完整分布，并提出一行代码修改即可生成流形上的均匀分布。
tags:
  - "ICLR 2026"
  - "图像生成"
  - "Score Learning"
  - "流形假设"
  - "几何学习"
  - "分布学习"
  - "率分离"
  - "均匀采样"
---

# When Scores Learn Geometry: Rate Separations under the Manifold Hypothesis

**会议**: ICLR 2026  
**arXiv**: [2509.24912](https://arxiv.org/abs/2509.24912)  
**代码**: 无  
**领域**: 扩散模型理论  
**关键词**: Score Learning, 流形假设, 几何学习, 分布学习, 率分离, 均匀采样

## 一句话总结
在流形假设下揭示score学习中几何信息与分布信息的尺度分离现象——流形几何信息强度为 $\Theta(\sigma^{-2})$，比分布信息强 $O(\sigma^{-2})$ 倍，由此证明扩散模型的成功主要来自学习数据流形而非完整分布，并提出一行代码修改即可生成流形上的均匀分布。

## 研究背景与动机

**领域现状**：Score-based方法（扩散模型、贝叶斯逆问题等）通常被解释为在低噪声极限 $\sigma \to 0$ 时学习数据分布。流形假设——数据支持在高维空间的低维流形上——被广泛采用。

**现有痛点**：(1) 低温极限下score估计极其困难，实践中需要大量post-training工程来稳定；(2) 现有理论分析将几何和分布信息混合在误差界中，未揭示尺度分离；(3) 即使score近似很好，恢复的分布可能在流形上是任意的。

**核心矛盾**：分布学习（完全恢复 $\mu_{\text{data}}$）要求 $o(1)$ 的score精度，但实际score误差通常远大于此。然而扩散模型仍能生成逼真样本——这说明它们可能并非真正学会了分布。

**本文目标**：(1) 为什么扩散模型在score不完美时仍然有效？(2) 能否利用这种不完美的score做有用的事情（如均匀采样）？

**切入角度**：在 $\sigma \to 0$ 时对score函数进行渐近展开，分离几何项（leading order, $\Theta(\sigma^{-2})$）和分布项（sub-leading order, $\Theta(1)$）。

**核心 idea**：score函数中流形几何信息比分布信息强 $\sigma^{-2}$ 倍，因此几何学习允许 $o(\sigma^{-2})$ 大的误差，远比分布学习所需的 $o(1)$ 宽松。

## 方法详解

### 整体框架

本文不提出新模型，而是在流形假设（数据支持在 $C^4$ 紧致无边界低维流形 $\mathcal{M}$ 上）下，对高斯平滑后的 score 函数在 $\sigma \to 0$ 时做渐近展开，把它拆成强度不同的两层，再据此证明"几何学习比分布学习宽松 $\sigma^{-2}$ 倍"，并给出一行代码改成均匀采样的算法。整套理论建立在三个量上：平滑测度 $\mu_\sigma = \text{law}(X + \sigma Z)$（VE）或 $\text{law}(\sqrt{1-\sigma^2}X + \sigma Z)$（VP）、半平方距离函数 $d_\mathcal{M}(x) = \tfrac{1}{2}\text{dist}^2(x, \mathcal{M})$，以及流形上的 Riemannian 体积测度与度量张量 $g(u)$。

### 关键设计

**1. Score 的渐近展开：把几何项和分布项按尺度拆开**

现有 score learning 理论的根本问题是把"点离流形多远"和"流形上密度多大"两类信息混在同一个误差界里，让人看不清扩散模型到底学到了什么。本文在 $\sigma \to 0$ 时对 score 做展开（Theorem 3.1），证明它由两层叠加：leading order 项强度为 $\Theta(\sigma^{-2})$，本质是到流形的投影算子，把任意点沿法向拉回最近流形点，编码的是纯几何信息；sub-leading order 项强度为 $\Theta(1)$，才编码流形上的密度 $p_{\text{data}}$。两层之间相差 $\sigma^{-2}$ 这个巨大的尺度间隔，就是全文的"率分离"。它之所以成立，是因为小噪声下离流形哪怕一点点的偏移都会被 $1/\sigma^2$ 急剧放大，几何信号因此压倒性地主导了 score 的量级。

**2. 流形集中 vs 分布恢复：两类任务对 score 精度的要求差 $\sigma^{-2}$ 倍**

既然 score 分两层，那"学几何"和"学分布"要求的精度自然不同（Theorem 4.1）。要让生成样本集中落在流形附近（流形集中），只需 score 误差控制在 $o(\sigma^{-2})$ 即可——因为主导的几何项本身就有 $\Theta(\sigma^{-2})$ 的量级，再大的误差只要不淹没它就行；而要精确恢复数据分布 $\mu_{\text{data}}$，误差必须压到 $o(1)$，才能让微弱的 $\Theta(1)$ 分布项不被噪声盖住。两者差距是 $O(\sigma^{-2})$，在小 $\sigma$ 下天差地别。这恰好解释了为什么实践中 score 估计远不完美、扩散模型却仍能生成逼真样本：它们达到的精度足以学会流形几何（"什么看起来像真实数据"），但远不足以恢复真实分布（"真实数据出现的频率"）。

**3. 一行代码改成均匀采样：用距离梯度抵消几何项**

既然不完美的 score 已经把流形几何学得够好，就能直接拿它做一件分布学习做不到的事——在流形上均匀采样（Theorems 5.1–5.2）。做法是把 Langevin 动力学里的 score 换成 $\tilde{s}_\sigma(x) = \nabla \log p_\sigma(x) + \tfrac{1}{\sigma^2} \nabla d_\mathcal{M}(x)$，作者称之为 **Tempered Score (TS) Langevin dynamics**，即

$$dX_t = \Big[\nabla \log p_\sigma(X_t) + \tfrac{1}{\sigma^2}\nabla d_\mathcal{M}(X_t)\Big] dt + \sqrt{2}\, dW_t.$$

新增的 $\tfrac{1}{\sigma^2}\nabla d_\mathcal{M}$ 项恰好抵消 score 中那个 $\Theta(\sigma^{-2})$ 的几何（投影）分量，剩下的只有 $\Theta(1)$ 的分布项，而它在小 $\sigma$ 下被噪声淹没，于是动力学收敛到流形上的均匀分布——也就是流形几何的最大熵表示。关键是这一步所需 score 精度同样只要 $o(\sigma^{-2})$，与几何学习一致，因此无需任何微调，纯在推理时实现。一个技术难点是：当 score 来自参数化模型、不再是严格的梯度场时，Langevin 动力学的稳态分布没有闭式解，本文用非可逆动力学的工具补全了这部分论证。

**4. 贝叶斯逆问题：均匀先验对 score 误差更鲁棒**

率分离的实践含义还延伸到逆问题（Theorem 6.1）。若把先验取成流形上的均匀分布，后验采样只需 $o(\sigma^{-2})$ 的 score 精度即可；而若坚持用数据分布 $\mu_{\text{data}}$ 当先验，则要更苛刻的 $o(1)$。也就是说，最大熵（均匀）先验天然对 score 估计误差更宽容，这为依赖 score-based 先验的下游逆问题提供了"该用什么先验更稳"的直接指导。

## 实验关键数据

### 合成实验
- 1D圆嵌入2D空间：验证几何恢复和均匀采样的可行性
- 不同score误差水平：$o(\sigma^{-2})$ 足以恢复流形支持，但分布任意
- 一行修改后：生成流形上的均匀分布

### Stable Diffusion 1.5实验
- 在大规模预训练模型上验证理论预测
- 修改采样算法后生成更均匀覆盖流形的样本
- 多样性增加，偏差减少

### 关键发现
- 实验验证了 $\Theta(\sigma^{-2})$ 的率分离在实际模型中成立
- 均匀采样算法在不需要微调的情况下有效工作
- 标准扩散模型确实主要学习了流形几何而非精确分布

## 亮点与洞察
- **范式转换的主张**：从分布学习→几何学习，这不仅是理论发现，更是实践指导——不必追求完美的分布恢复
- **一行代码的力量**：均匀采样仅需修改一行（添加距离梯度项），实用性极强
- **对扩散模型成功的新解释**：是因为它们学会了"什么看起来像真实数据"（流形），而非"真实数据的频率分布"
- **贝叶斯逆问题的启示**：均匀先验（最大熵）比数据先验对误差更鲁棒，这对下游应用有重要指导

## 局限与展望
- 理论分析依赖流形假设（$C^4$紧致无边界流形），实际数据可能不严格满足
- $\nabla d_\mathcal{M}$ 在实际中需要估计，估计精度影响均匀采样质量
- 初步实验在Stable Diffusion 1.5上进行，更大模型和更多任务待验证
- 率分离的精确常数（前因子）未讨论，实际gap可能小于理论预测

## 相关工作与启发
- **vs Stanczuk et al. 2024**：他们估计内在维度，不涉及率分离
- **vs Ventura et al. 2024**：仅分析线性流形（子空间），本文适用于一般光滑流形
- **vs De Santi et al. 2025**：他们通过微调实现均匀采样，本文在推理时实现（无需微调）
- **vs 现有score learning理论**：它们将几何和分布误差混合，本文首次分离两者

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 率分离现象的发现和几何学习范式的主张具有突破性
- 实验充分度: ⭐⭐⭐⭐ 合成+Stable Diffusion验证，但缺少更多定量评估
- 写作质量: ⭐⭐⭐⭐⭐ 理论展开清晰，图示直观，主要贡献突出
- 价值: ⭐⭐⭐⭐⭐ 对理解扩散模型原理和改进采样有深远影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Generative Model Inversion Through the Lens of the Manifold Hypothesis](../../NeurIPS2025/image_generation/generative_model_inversion_through_the_lens_of_the_manifold_hypothesis.md)
- [\[ICLR 2026\] Dragging with Geometry: From Pixels to Geometry-Guided Image Editing](dragging_with_geometry_from_pixels_to_geometry-guided_image_editing.md)
- [\[ICLR 2026\] TempFlow-GRPO: When Timing Matters for GRPO in Flow Models](tempflow-grpo_when_timing_matters_for_grpo_in_flow_models.md)
- [\[ICLR 2026\] Learning a Distance Measure from the Information-Estimation Geometry of Data](learning_a_distance_measure_from_the_information-estimation_geometry_of_data.md)
- [\[ICML 2026\] Let EEG Models Learn EEG](../../ICML2026/image_generation/let_eeg_models_learn_eeg.md)

</div>

<!-- RELATED:END -->
