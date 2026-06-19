---
title: >-
  [论文解读] Learning to Steer: Input-dependent Steering for Multimodal LLMs
description: >-
  [NeurIPS 2025][多模态VLM][steering] 针对现有模型引导(steering)方法使用固定方向向量无法适配不同输入的局限，提出 L2S (Learn-to-Steer)：先通过输入特定的对比提示生成理想的引导向量（P2S），再训练一个轻量 2 层 MLP 从输入上下文预测该向量，以极低开销实现了输入依赖的行为引导，在安全执行和幻觉缓解两个应用上显著优于静态 steering 基线。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "steering"
  - "input-dependent"
  - "hallucination mitigation"
  - "safety enforcement"
  - "提示学习"
---

# Learning to Steer: Input-dependent Steering for Multimodal LLMs

**会议**: NeurIPS 2025  
**arXiv**: [2508.12815](https://arxiv.org/abs/2508.12815)  
**代码**: [https://github.com/jayneelparekh/learn-to-steer](https://github.com/jayneelparekh/learn-to-steer)  
**领域**: 多模态VLM / 模型安全 / 幻觉缓解 / 表示引导  
**关键词**: steering, input-dependent, hallucination mitigation, safety enforcement, contrastive prompting  

## 一句话总结
针对现有模型引导(steering)方法使用固定方向向量无法适配不同输入的局限，提出 L2S (Learn-to-Steer)：先通过输入特定的对比提示生成理想的引导向量（P2S），再训练一个轻量 2 层 MLP 从输入上下文预测该向量，以极低开销实现了输入依赖的行为引导，在安全执行和幻觉缓解两个应用上显著优于静态 steering 基线。

## 研究背景与动机

**领域现状**：Steering 通过在 LLM/MLLM 的潜在表示上施加线性偏移来引导模型行为，是一种轻量级的 post-hoc 控制方法。主流做法（如 CAA/mean-steering）计算正负行为表示的均值差作为固定引导向量，对所有输入统一施加。

**现有痛点**：固定引导向量的致命问题在于——**期望行为的实例化是输入依赖的**。例如，对于非法活动查询，安全回答应该是拒绝回答；对于医疗咨询，安全回答应该是建议咨询专家。这两种"安全"行为完全不同，一个固定向量无法兼顾。

**核心矛盾**：理想的输入特定引导向量（P2S）需要知道期望的答案内容才能计算——但推理时我们正是不知道答案才需要引导，形成了鸡和蛋的问题。

**切入角度**：虽然我们在推理时不知道期望答案，但可以用训练数据中的对比提示构建 P2S 向量作为"教师信号"，然后训练一个极简网络从输入上下文预测这些向量。

**核心 idea**：用 2 层 MLP 从输入的中间层表示预测输入特定的 steering 向量，将 P2S 的理论优势转化为实际可用的 L2S 方法。

## 方法详解

### 整体框架
分为训练和推理两个阶段：
- **训练阶段**：对每个样本 $X=(I,T)$，构建输入特定的正/负对比提示 $(T_X^+, T_X^-)$，通过 teacher forcing 提取 $L^*$ 层最后 token 的表示差作为 P2S 向量 $z_{X,L^*}$，同时提取 $L'$ 层输入上下文表示 $h_{X,L'}$，训练 MLP $g_{\Theta}$ 使得 $g_\Theta(h_{X,L'}) \approx z_{X,L^*}$
- **推理阶段**：对新输入提取 $h_{X,L'}$，用训练好的 $g_{\Theta^*}$ 预测引导向量，施加到 $L^*$ 层所有生成 token 的表示上

### 关键设计

1. **输入特定对比提示 (P2S)**：

    - 功能：为每个输入生成反映期望/非期望行为的提示补全
    - 核心思路：构建 $X^+ = (I, T||T_X^+)$ 和 $X^- = (I, T||T_X^-)$，在 teacher forcing 下提取 $L^*$ 层最后 token 表示的差：$z_{X,L^*} = h_{L^*}^{q^+}(X^+) - h_{L^*}^{q^-}(X^-)$
    - 设计动机：不同于 CAA 的固定提示对，P2S 允许不同输入使用不同的行为描述。例如安全场景中，非法活动用"拒绝"模板，医疗咨询用"推荐专家"模板

2. **Learn-to-Steer (L2S) 辅助网络**：

    - 功能：从输入上下文预测 P2S 引导向量，使推理时不需要知道对比提示
    - 核心思路：定义输入上下文为 $L'$ 层最后 input token 的表示 $h_{X,L'} = h_{L'}^{N_V+N_T}(X)$。训练目标为均方误差 $\Theta^* = \arg\min_\Theta \mathbb{E}_X[\|z_{X,L^*} - g_\Theta(h_{X,L'})\|_2^2]$。推理时对生成 token $p$ 施加 $h_{L^*}^p \leftarrow h_{L^*}^p + \alpha g_{\Theta^*}(h_{X,L'})$
    - 设计动机：2 层 MLP（hidden size 100）极其轻量，训练只需表示空间操作无需加载主模型梯度，内存开销可忽略

3. **多行为场景处理**：

    - 功能：在同一 steering 框架中处理多种不同的期望行为
    - 关键示例（安全场景）：前 9 类有害活动→用"拒绝/回避"模板的 $(T_X^+, T_X^-)$；后 3 类敏感咨询→用"建议咨询专家"模板的 $(T_X^+, T_X^-)$。L2S 通过学习不同输入到不同向量的映射，自然支持多行为
    - 关键对比：mean-steering 如果混合不同模板的向量会互相干扰（Mean-S 差于 Mean-S(BA)）

### 训练策略
- 辅助网络：2 层 MLP，hidden size 100
- 训练 100 epochs，Adam 优化器，学习率 $10^{-4}$ 或 $5\times10^{-5}$
- Cosine 学习率调度 + plateau 自适应
- 引导强度 $\alpha \in [1, 3.0)$（LLaVA），保证响应质量下降 <10%
- 安全任务：$L^*=15$（引导层），$L'=30$（上下文提取层）
- 幻觉任务：$L^*=14, L'=14$
- 实验模型：LLaVA-v1.5-7B 和 Qwen2-VL-7B，单张 RTX5000 (24GB) 即可运行

## 实验关键数据

### 安全执行 — MMSafetyBench (LLaVA-v1.5)

| 指标 | No-steering | Prompt | Mean-S | Mean-S(BA) | L2S | P2S* |
|------|------------|--------|--------|-----------|-----|------|
| $\mathbb{E}_{p\geq0.5}$[Unsafe]↓ | 0.276 | 0.248 | 0.161 | 0.089 | **0.082** | 0.094 |
| $\mathbb{E}_{p\geq0.7}$[Unsafe]↓ | 0.234 | 0.207 | 0.129 | 0.066 | **0.057** | 0.064 |
| $\mathbb{E}_{p\geq0.9}$[Unsafe]↓ | 0.204 | 0.183 | 0.102 | 0.041 | **0.034** | 0.042 |
| ED-score↑ | 0.250 | 0.197 | 0.329 | 0.276 | **0.395** | 0.382 |
| Response quality↑ | 6.92 | 7.34 | 6.61 | 6.42 | 6.56 | 6.49 |

### 幻觉缓解 — POPE (LLaVA-v1.5)

| 子集 | 指标 | No-steering | Prompt | Norm-Rnd | Mean-S | L2S | P2S* |
|------|------|------------|--------|---------|--------|-----|------|
| Random | Accuracy↑ | 82.73 | 84.91 | 82.38 | 84.29 | **86.46** | 89.26 |
| Random | F1↑ | 90.55 | 91.84 | 90.34 | 91.47 | **92.74** | 94.33 |
| Popular | Accuracy↑ | 80.40 | 83.35 | 80.36 | 82.11 | **82.58** | 88.64 |
| Adversarial | Accuracy↑ | 76.82 | 76.36 | 75.77 | 76.36 | **77.76** | 82.58 |

### CHAIR 评估 (LLaVA-v1.5, 500 COCO 图像)

| 方法 | CHAIR_s↓ | CHAIR_i↓ | Recall↑ | Gemini Win Rate↑ |
|------|---------|---------|---------|-----------------|
| No-steering | 17.31 | 52.80 | 71.23 | 35.80% |
| **L2S** | **16.10** | **51.80** | **73.50** | **64.20%** |

### 关键发现
- **L2S 超越 P2S oracle**在安全任务上（Unsafe-score 0.082 vs 0.094），说明 L2S 的泛化能力甚至优于逐样本计算的理想方法
- Mean-S 在混合多种行为模板时性能下降（Mean-S 0.161 vs Mean-S(BA) 0.089），但 L2S 可以同时处理多行为（ED-score 0.395 远超所有基线）
- 随机方向 steering (Norm-Rnd) 可以降低有害内容但无法引导专家推荐行为，证明 steering 方向的精确性至关重要
- 在幻觉任务上，Mean-S 和 Prompt 不能在所有子集上一致提升，但 L2S 全面优于所有可用基线
- Gemini Win Rate 64.20% 表明 L2S 不仅减少幻觉，还提高了描述质量

## 亮点与洞察
- **输入依赖 steering 的核心洞察**精准：期望行为不是一个固定方向，而是取决于输入语境的流形——这在安全场景中尤其明显（拒绝 vs 推荐专家 vs 不介入）
- **2 层 MLP 替代 teacher forcing**的巧妙之处在于：将一个理论上不实用的方法（需要知道答案才能引导）转化为实际可部署的轻量方案
- **训练成本极低**：只需在表示空间训练小网络，无需主模型梯度，单 24GB GPU 即可完成全流程
- L2S 在安全任务上**超越 oracle P2S** 说明学到的映射具有正则化效果，泛化性好

## 局限与展望
- 对比提示的选择仍需人工设计，不同应用场景需要定制不同的 $(T_X^+, T_X^-)$ 模板
- 目前只在单层 $L^*$ 施加线性偏移，多层/非线性 steering 可能更有效
- 辅助网络容量（hidden 100）可能限制了对复杂行为的建模能力
- 主要在 LLaVA-v1.5 和 Qwen2-VL 上验证，需要更多模型和任务的验证
- $\alpha$ 值对性能-质量权衡非常敏感（$\alpha \geq 3$ 明显退化），自动化选择 $\alpha$ 是一个开放问题
- 恶意使用风险：同样的方法可以被用来引导模型产生有害行为

## 相关工作与启发
- **vs CAA (Contrastive Activation Addition)**：CAA 用固定的均值差向量，适合行为实例化单一的场景；L2S 扩展为输入依赖，覆盖多行为场景
- **vs CAST**：CAST 根据条件向量的相似度缩放固定 steering 向量，但方向仍不变；L2S 的方向和大小都是输入依赖的
- **vs PAI / AD-HH (注意力头干预)**：这些方法直接操纵注意力权重，L2S 操纵残差流表示；两者互补，可以组合使用
- **vs 微调（SFT/RLHF）**：微调成本高且可能遗忘；L2S 是 post-hoc 方法，不修改模型权重

## 评分
- 新颖性: ⭐⭐⭐⭐ 输入依赖 steering 的想法自然但此前未被充分探索，P2S→L2S 的两步设计巧妙
- 实验充分度: ⭐⭐⭐⭐ 安全+幻觉两个应用，两个模型，多个指标维度，消融全面
- 写作质量: ⭐⭐⭐⭐ 动机阐述清晰，例子直观，方法描述简洁
- 价值: ⭐⭐⭐⭐ 实用性强——极低成本的 post-hoc 行为控制方法，可直接部署于生产环境

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Analyzing Finetuning Representation Shift for Multimodal LLMs Steering](../../ICCV2025/multimodal_vlm/analyzing_finetuning_representation_shift_for_multimodal_llms_steering.md)
- [\[NeurIPS 2025\] Vision Function Layer in Multimodal LLMs](vision_function_layer_in_multimodal_llms.md)
- [\[NeurIPS 2025\] ADMN: A Layer-Wise Adaptive Multimodal Network for Dynamic Input Noise and Compute Resources](admn_a_layerwise_adaptive_multimodal_network_for_dynamic_inp.md)
- [\[NeurIPS 2025\] BridgeVLA: Input-Output Alignment for Efficient 3D Manipulation Learning with Vision-Language Models](bridgevla_input-output_alignment_for_efficient_3d_manipulation_learning_with_vis.md)
- [\[NeurIPS 2025\] VT-FSL: Bridging Vision and Text with LLMs for Few-Shot Learning](vt-fsl_bridging_vision_and_text_with_llms_for_few-shot_learning.md)

</div>

<!-- RELATED:END -->
