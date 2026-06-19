---
title: >-
  [论文解读] Matryoshka Pilot: Learning to Drive Black-Box LLMs with LLMs
description: >-
  [NeurIPS 2025][模型压缩][Black-Box LLM] 提出 Matryoshka Pilot (M-Pilot)，用轻量级白盒 LLM 作为控制器，通过生成中间引导（任务分解、高层计划、用户画像）来驱动黑盒 LLM 在推理、规划和个性化等复杂长程任务上的性能，并通过迭代 DPO 实现自我改进。
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "Black-Box LLM"
  - "White-Box Controller"
  - "迭代DPO"
  - "中间引导"
  - "多轮交互"
---

# Matryoshka Pilot: Learning to Drive Black-Box LLMs with LLMs

**会议**: NeurIPS 2025  
**arXiv**: [2410.20749](https://arxiv.org/abs/2410.20749)  
**代码**: [GitHub](https://github.com/lichangh20/Matryoshka)  
**领域**: Model Compression / LLM Control  
**关键词**: Black-Box LLM, White-Box Controller, 迭代DPO, 中间引导, 多轮交互

## 一句话总结

提出 Matryoshka Pilot (M-Pilot)，用轻量级白盒 LLM 作为控制器，通过生成中间引导（任务分解、高层计划、用户画像）来驱动黑盒 LLM 在推理、规划和个性化等复杂长程任务上的性能，并通过迭代 DPO 实现自我改进。

## 研究背景与动机

**领域现状**：商用 LLM（GPT-4、Gemini 等）多为黑盒模型，用户无法访问模型参数、结构甚至输出 logits。

**现有痛点**：增强黑盒 LLM 能力的现有方法主要分两类——(a) ICL 方法需要精心设计的示例和提示，依赖人工启发；(b) 适配器方法从多个候选中选最优，但受限于黑盒 LLM 本身的生成能力。两类方法在长程任务（多步推理、长期规划）上均表现不佳。

**核心矛盾**：黑盒 LLM 的不透明性使得直接优化其能力不可行，但用户又需要针对特定任务提升其性能。

**本文目标**：如何在不访问黑盒 LLM 参数的前提下，系统性地增强其在复杂长程任务上的推理、规划和个性化能力。

**切入角度**：将黑盒 LLM 视为"环境"，训练一个白盒 LLM 作为"策略"来生成中间引导。

**核心idea**：用小模型驱动大模型——轻量白盒控制器生成中间引导来指导黑盒 LLM 的行为，并通过迭代偏好优化实现持续改进。

## 方法详解

### 整体框架

M-Pilot 采用控制器-生成器框架：白盒 LLM（如 LLaMA-3-8B-Instruct）作为控制器，生成 $T$ 步中间引导 $\{g_t\}_{t=1}^T \sim f_\theta(x)$；黑盒 LLM（如 GPT-4o-mini）作为生成器/环境，接收引导后生成最终答案 $\hat{y} \sim g_{\text{LLM}}(x, \{g_t\}_{t=1}^T)$。

### 关键设计

1. **任务特定的中间引导实例化**：

    - **推理任务**：控制器输出问题分解的子任务序列，帮助黑盒 LLM 进行逐步推理
    - **规划任务**：控制器生成高层计划，将复杂任务分解为子目标
    - **个性化任务**：控制器对用户历史记录进行摘要总结，生成用户画像
   
   这三种引导形式统一在同一框架下，体现了方法的通用性。

2. **多轮交互与 MDP 建模**：将控制器-环境交互建模为马尔可夫决策过程。在每个步骤 $t$，控制器根据当前状态 $s_{t-1}$ 生成动作 $a_t$（即引导），环境返回观察 $o_t$。状态转移为：
$$s_t = (s_{t-1}, a_t, o_t) = (x, a_1, o_1, \cdots, a_t, o_t)$$
整个轨迹的奖励由最终答案的正确性评估函数 $u(\tau) = \text{eval}(\hat{y}, y)$ 决定。

3. **数据收集**：对每个输入 $x_i$ 进行 $K$ 次多轮交互采样，获得多个轨迹及其评估信号，区分正例（引导成功）和负例（引导失败）。通过引入随机性增加引导多样性。

4. **迭代直接偏好优化（IDPO）**：

    - **SFT 热启动**：用行为克隆（利用 GPT-3.5 生成的引导）初始化控制器策略
    - **迭代偏好对收集**：在第 $m$ 次迭代中，用 $\theta^{(m)}$ 生成新轨迹，按成功/失败分组，与历史数据合并
    - **偏好优化**：使用 Bradley-Terry 模型建模偏好，更新参考策略为上一轮模型 $\pi_{\text{ref}} = \pi_\theta^{(m)}$，训练目标为：
$$\mathcal{L}_{\text{IDPO}} = \mathbb{E}_{(x, \tau^+, \tau^-) \sim \mathcal{D}} \left[ -\log\sigma\left(\eta^{-1}\left(\log\frac{p_{\theta^{(m+1)}}(\{g_t^+\}|x)}{p_{\theta^{(m)}}(\{g_t^+\}|x)} - \log\frac{p_{\theta^{(m+1)}}(\{g_t^-\}|x)}{p_{\theta^{(m)}}(\{g_t^-\}|x)}\right)\right)\right]$$
   关键推导：由于黑盒 LLM 的生成概率在正负样本比值中可以消除，优化目标仅涉及白盒控制器的参数。

### 训练策略

- SFT 热启动 → 迭代数据采样 + DPO 训练（bootstrapping 方式累积数据）
- 每轮迭代更新参考策略，实现自我改进

## 实验关键数据

### 主实验

**个性化任务 (LaMP)**：使用 gpt-4o-mini 作为黑盒 LLM

| 方法 | LaMP-1 Acc | LaMP-2N Acc | LaMP-2M Acc | LaMP-3 MAE↓ | LaMP-4 BLEU |
|------|-----------|-------------|-------------|-------------|-------------|
| gpt-4o-mini | 0.514 | 0.655 | 0.413 | 0.371 | 0.992 |
| RAG (k=4) | 0.632 | 0.792 | 0.502 | 0.272 | 2.953 |
| **M-Pilot** | **0.640** | **0.823** | **0.527** | **0.277** | **4.298** |

**推理任务 (GSM8K)**：

| 方法 | GSM8K (gpt-3.5) | GSM-HARD (gpt-3.5) |
|------|-----------------|---------------------|
| CoT | 0.809 | 0.406 |
| PAL_SelfDebug | 0.864 | 0.701 |
| **M-Pilot** | **0.931** | **0.761** |

**规划任务 (ALFWorld)**：使用 gpt-3.5-turbo

| 方法 | Overall Success Rate |
|------|---------------------|
| ReAct | 47.76% |
| AdaPlanner | 88.06% |
| **M-Pilot** | **96.27%** |

### 消融实验

| 变体 | ALFWorld 成功率 |
|------|----------------|
| M-Pilot (完整) | 96.27% |
| w/o 2nd-round IDPO | 94.78% |
| w/o 1st, 2nd-round IDPO | 88.06% |
| w/o Guidance Optimization | 81.34% |

### 关键发现

- M-Pilot 在三类任务上平均提升：推理 +3.19%、规划 +7.46%、个性化 +5.82%
- 即插即用能力：在 gpt-4o-mini 上训练的控制器可直接迁移到 gpt-3.5 和 gemini-1.5-flash
- 样本效率高：仅用 1/4 训练数据即可超越最强基线 AdaPlanner
- 迭代 DPO 的自改进效果显著，每增加一轮 IDPO 训练均带来稳定提升

## 亮点与洞察

- **"小模型驱动大模型"的范式创新**：将白盒-黑盒 LLM 的交互形式化为 MDP，用强化学习方法优化
- **通用性强**：同一框架在推理、规划、个性化三类截然不同的任务上均有效
- **即插即用**：训练好的控制器可无需额外训练直接迁移到其他黑盒 LLM
- **自我改进**：迭代 DPO 使控制器在没有人类标注的情况下持续提升引导质量

## 局限与展望

- 控制器需要针对不同任务类型设计特定的引导格式（问题分解、高层计划、用户摘要）
- 数据收集阶段需要大量与黑盒 LLM 的交互，API 调用成本较高
- 中间引导的质量上限受制于白盒 LLM 的能力
- 目前仅在 NLP 任务上验证，未扩展到多模态场景
- 安全风险：恶意用户可能利用白盒控制器来越狱黑盒 LLM

## 相关工作与启发

- **o1-preview 的 Scratchpad**：M-Pilot 与之类似，但将中间推理过程外化到独立的白盒模型中
- **STaR (Zelikman et al.)**：M-Pilot 借鉴了自引导推理的 bootstrapping 思想
- **RLPrompt / TEMPERA**：现有 RL prompt 优化方法局限于分类任务，M-Pilot 扩展到长程生成
- 启发：轻量模型作为"大脑前额叶"指导大模型执行，可能是未来黑盒 LLM 优化的重要方向

## 评分

- 新颖性: ⭐⭐⭐⭐ 控制器-生成器框架的MDP建模新颖，但"小模型帮大模型"的想法非首创
- 实验充分度: ⭐⭐⭐⭐ 三类任务覆盖全面，消融和即插即用实验充分
- 写作质量: ⭐⭐⭐⭐ 数学推导清晰，框架叙述条理分明
- 价值: ⭐⭐⭐⭐ 对黑盒LLM增强有实用价值，即插即用特性尤为实用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Rank-Guided Pseudo-Bias Learning for Robust Black-Box Adaptation](../../CVPR2026/model_compression/rank-guided_pseudo-bias_learning_for_robust_black-box_adaptation.md)
- [\[NeurIPS 2025\] TokenSqueeze: Performance-Preserving Compression for Reasoning LLMs](tokensqueeze_performance-preserving_compression_for_reasoning_llms.md)
- [\[NeurIPS 2025\] Robust Federated Finetuning of LLMs via Alternating Optimization of LoRA](robust_federated_finetuning_of_llms_via_alternating_optimization_of_lora.md)
- [\[NeurIPS 2025\] DenoiseRotator: Enhance Pruning Robustness for LLMs via Importance Concentration](denoiserotator_enhance_pruning_robustness_for_llms_via_importance_concentration.md)
- [\[NeurIPS 2025\] Few-Shot Knowledge Distillation of LLMs With Counterfactual Explanations](few-shot_knowledge_distillation_of_llms_with_counterfactual_explanations.md)

</div>

<!-- RELATED:END -->
