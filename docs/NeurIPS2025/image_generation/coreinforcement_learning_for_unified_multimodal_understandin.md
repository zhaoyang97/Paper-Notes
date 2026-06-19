---
title: >-
  [论文解读] Co-Reinforcement Learning for Unified Multimodal Understanding and Generation
description: >-
  [NeurIPS 2025][图像生成][统一多模态模型] 提出 CoRL（Co-Reinforcement Learning）框架，通过"统一RL→精细化RL"两阶段策略对统一多模态模型（ULM）同时进行理解和生成能力的强化学习优化，实现理解生成双能力的协同进化，在 1.5B 参数量下生成提升 7%、理解提升 23%。
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "统一多模态模型"
  - "强化学习"
  - "GRPO"
  - "文本生成图像"
  - "多模态理解"
---

# Co-Reinforcement Learning for Unified Multimodal Understanding and Generation

**会议**: NeurIPS 2025  
**arXiv**: [2505.17534](https://arxiv.org/abs/2505.17534)  
**代码**: [https://github.com/mm-vl/ULM-R1](https://github.com/mm-vl/ULM-R1)  
**领域**: 图像生成  
**关键词**: 统一多模态模型, 强化学习, GRPO, 文本生成图像, 多模态理解

## 一句话总结

提出 CoRL（Co-Reinforcement Learning）框架，通过"统一RL→精细化RL"两阶段策略对统一多模态模型（ULM）同时进行理解和生成能力的强化学习优化，实现理解生成双能力的协同进化，在 1.5B 参数量下生成提升 7%、理解提升 23%。

## 研究背景与动机

**领域现状**：统一多模态大语言模型（ULM）能同时处理视觉理解和图像生成任务，代表性工作包括 Janus-Pro（全自回归 F-AR 方案）和 Show-o（自回归+扩散混合方案）。强化学习（RL）后训练已在纯文本 LLM（如 DeepSeek-R1）上展示出显著效果，但在多模态领域的应用主要局限于理解任务的推理增强。

**现有痛点**：(1) RL 在视觉生成上的应用非常有限，仅 SimpleAR 用 CLIP Score 做过初步探索，效果一般；(2) 更关键的是，将 RL 同时用于 ULM 的理解和生成两种能力的协同优化完全未被探索；(3) 直接对单一任务做 RL 不仅在生成任务上提升有限，还可能损害另一个任务的能力。

**核心矛盾**：ULM 的理解和生成共享同一个 LLM backbone，分别优化容易产生冲突。现有 RL 方法（如 GRPO）的 reward 设计主要面向文本输出，缺乏适用于图像生成的可验证奖励信号。

**本文目标** 设计一套适合 ULM 的强化学习框架，让理解和生成两种能力在共享的策略优化中互利共进而非互相伤害。

**切入角度**：作者做了系统的 pilot study，比较了四种 RL 策略（分别RL/分别RL后权重合并/交替RL/统一RL），发现统一 RL 显著优于其他策略，证明双能力可以在共享优化中协同进化。基于此洞察设计两阶段方案：先统一建立跨任务协同，再分别精细化。

**核心 idea**：通过统一 GRPO 框架同时优化 ULM 的理解和生成能力，利用跨任务奖励信号的协同效应实现双能力的共同提升。

## 方法详解

### 整体框架

CoRL 采用"基础→专精"（Foundation-then-Specialization）的两阶段 RL 流程。第一阶段（统一 RL）：在包含 22K 理解+生成样本的混合数据集上，用联合奖励函数进行 GRPO 优化，同时提升两种能力。第二阶段（精细化 RL）：分别用任务特定的奖励和数据对理解（MCQ/OE 两种）和生成能力进行针对性增强。基础模型为 Janus-Pro-1B/1.5B，8 卡 H20 训练。

### 关键设计

1. **双向循环一致性奖励（Bidirectional Cycle Consistency Reward）**:

    - 功能：为文本到图像生成任务提供可验证的语义保真度奖励
    - 核心思路：从两个方向评估生成质量——视觉一致性用 LPIPS 衡量生成图与真实图的感知相似度；文本一致性先用 BLIP 对生成图 re-caption，再用 SPICE 衡量 re-caption 与原始 prompt 的语义匹配度。$\mathcal{R}_{cycle} = 1 - \text{LPIPS}(\mathcal{I}_{real}, \mathcal{I}_{gen}) + \text{SPICE}(\mathcal{P}_{org}, \mathcal{C}_{re-cap})$，归一化到 [0,1]
    - 设计动机：单一的 CLIP Score 评估太粗糙，且在 pilot study 中效果不佳。双向循环形成闭环反馈——同时惩罚视觉幻觉（通过 LPIPS）和语义偏离（通过 SPICE），比单向度量更全面

2. **文本-图像匹配奖励（Text-Image Matching Reward）**:

    - 功能：在 token 级别细粒度评估跨模态对齐
    - 核心思路：利用 ULM 自身的特征空间，将 prompt 的文本 token 表示 $\mathbf{T}$ 和生成图的视觉 token 表示 $\mathbf{I}$ 做双向最大余弦相似度匹配：$\mathcal{R}_{TIM} = \frac{1}{2}(\frac{1}{L_i}\sum_j \max_k \cos(\mathbf{i}_j, \mathbf{t}_k) + \frac{1}{L_t}\sum_k \max_j \cos(\mathbf{t}_k, \mathbf{i}_j))$
    - 设计动机：CLIP Score 只给出全局匹配分数，无法捕捉细粒度的概念-视觉元素对应关系。利用 ULM 自身表示空间做 token 级匹配，既更细粒度又不依赖外部模型

3. **统一 RL（Stage 1）+ 精细化 RL（Stage 2）的两阶段策略**:

    - 功能：先建立跨任务协同基础，再做任务特定增强
    - 核心思路：Stage 1 使用联合奖励 $\mathcal{R}_{Uni} = \mathcal{R}_{cycle} + \mathcal{R}_{TIM} + \lambda(\mathcal{R}_{Acc} + \mathcal{R}_{Format})$，对理解和生成同时优化，采用标准 GRPO 但去掉 KL 散度约束以提升泛化。Stage 2 分三路独立精细化——生成用 $\mathcal{R}_{cycle} + \mathcal{R}_{TIM}$，MCQ 理解用 $\mathcal{R}_{MCQ-Acc} + \mathcal{R}_{Format}$，OE 理解用 $\mathcal{R}_{OE-Acc} + \mathcal{R}_{Format}$，此阶段重新引入 KL 约束防止偏移
    - 设计动机：Pilot study 明确证明统一 RL 优于分别 RL/交替 RL/权重合并等方案。两阶段设计让 Stage 1 建立共享的能力基础和跨任务知识迁移，Stage 2 再做精准优化而不会破坏已建立的协同关系

### 损失函数 / 训练策略

基于 GRPO，每个 prompt 采样 8 个（Stage 1）或 16 个（Stage 2）候选响应，计算组内归一化优势值。Stage 1 学习率 4e-6，batch size 16，$\lambda=0.8$；Stage 2 学习率降至 1e-6。理解任务的最终模型通过高斯分布权重合并策略组合 MCQ 和 OE 两个精细化模型。

## 实验关键数据

### 主实验

| Benchmark | Janus-Pro-1.5B | ULM-R1 | 提升 |
|-----------|---------------|--------|------|
| GenEval ↑ | 0.73 | **0.77** | +4.3 |
| DPG ↑ | 82.63 | **83.92** | +1.3 |
| WISE ↑ | 0.26 | **0.33** | +7 |
| MMMU ↑ | 36.3 | **42.3** | +6.0 |
| WeMath ↑ | 5.9 | **21.1** | +15.2 |
| LogicVista ↑ | 23.9 | **34.5** | +10.6 |
| MathVerse ↑ | 13.5 | **25.4** | +11.9 |
| MMVet ↑ | 39.8 | **43.9** | +4.1 |
| POPE ↑ | 86.2 | **88.9** | +2.7 |

### 消融实验

| 配置 | GenEval | DPG | MMMU | WeMath | LogicVT |
|------|---------|-----|------|--------|---------|
| Baseline | 73.0 | 82.6 | 36.3 | 5.9 | 23.9 |
| + Cold-SFT (S1) | 72.8 | 82.5 | 41.0 | 18.0 | 27.9 |
| + Unified-RL (S1) | 75.9 | 83.3 | 40.3 | 14.0 | 30.2 |
| + Refined-RL w/ Cold-SFT | 74.5 | 82.8 | 41.8 | 22.5 | 35.9 |
| **CoRL (Unified+Refined)** | **77.3** | **83.9** | **42.3** | **21.1** | **34.5** |

### 关键发现

- **统一 RL 是协同进化的关键**：对比 #1（Cold-SFT 基础）和 #2（Unified-RL 基础），统一 RL 在生成任务上显著更好（GenEval 75.9 vs 72.8），且理解任务的 LogicVista 提升更大（+6.3 vs +4.0），说明跨任务 RL 带来的协同效应超越了简单的监督学习
- **两阶段优于单阶段**：CoRL (#7) 全面优于仅有统一 RL 的 #2，精细化阶段在不破坏协同关系的前提下进一步提升各任务性能
- **数学推理提升最显著**：WeMath +15.2、MathVerse +11.9，表明 RL 的长链思维能力提升对数学推理尤其有效
- 1.5B 的 ULM-R1 在多个 benchmark 上超越了 7B 的 Janus-Pro，说明 RL 后训练的效率极高

## 亮点与洞察

- **首次系统验证 GRPO 对 ULM 双任务协同优化的有效性**：pilot study 清晰地展示了四种策略的优劣，统一 RL 的优越性给出了一个明确的设计原则——共享优化优于分离优化
- **自监督式的生成质量奖励设计**：cycle consistency + TIM 奖励不依赖外部大模型打分，而是利用 ULM 自身的表示空间和简单的 re-captioning 流程，降低了对外部 reward model 的依赖
- **小模型 + RL 后训练的效率**：1.5B 模型通过 CoRL 在多个 benchmark 上达到甚至超越 7B 模型的水平，说明 RL 后训练在效率/性能 trade-off 上极具吸引力

## 局限与展望

- 生成分辨率受限于 Janus-Pro 的 384×384，远低于主流扩散模型的 512/1024
- 图像生成的采样质量仍需 CFG（guidance weight=5），增加推理成本
- RL 训练过程中需要从 ULM 实时采样图像进行奖励评估，训练效率远低于纯文本 RL
- 未探索 RL 对视频生成/理解的适用性
- 权重合并策略（用于组合 MCQ/OE 精细化模型）的最优配置可能因任务而异

## 相关工作与启发

- **vs SimpleAR**：SimpleAR 只用 CLIP Score 做自回归生成的 RL，效果一般。CoRL 的双向循环一致性 + token 级匹配奖励设计更精细，且同时处理理解和生成
- **vs R1-like MLLMs**（如 Vision-R1、LMM-R1）：这些工作只用 RL 增强理解/推理能力，CoRL 首次将 RL 扩展到理解+生成的联合优化
- **vs DPO-based ULMs**（如 Emu3-DPO、HermesFlow）：DPO 需要偏好数据对，CoRL 使用可验证的 rule-based rewards，数据需求更低且更灵活

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统研究 RL 对 ULM 双能力的联合优化，但方法框架本身基于已有的 GRPO
- 实验充分度: ⭐⭐⭐⭐⭐ pilot study 设计扎实，12 个 benchmark 覆盖全面，消融详细
- 写作质量: ⭐⭐⭐⭐ 整体清晰，pilot study 部分特别有说服力
- 价值: ⭐⭐⭐⭐⭐ 为 ULM 的 RL 后训练提供了清晰的路线图，跨任务协同的发现具有广泛启发性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Learning to Generate via Understanding: Understanding-Driven Intrinsic Rewarding for Unified Multimodal Models](../../CVPR2026/image_generation/learning_to_generate_via_understanding_understanding-driven_intrinsic_rewarding_.md)
- [\[CVPR 2025\] TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation](../../CVPR2025/image_generation/tokenflow_unified_image_tokenizer_for_multimodal_understanding_and_generation.md)
- [\[NeurIPS 2025\] Show-o2: Improved Native Unified Multimodal Models](show-o2_improved_native_unified_multimodal_models.md)
- [\[NeurIPS 2025\] RLVR-World: Training World Models with Reinforcement Learning](rlvr-world_training_world_models_with_reinforcement_learning.md)
- [\[NeurIPS 2025\] Mitigating Intra- and Inter-modal Forgetting in Continual Learning of Unified Multimodal Models](mitigating_intra-_and_inter-modal_forgetting_in_continual_learning_of_unified_mu.md)

</div>

<!-- RELATED:END -->
