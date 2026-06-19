---
title: >-
  [论文解读] Janus-Pro-R1: Advancing Collaborative Visual Comprehension and Generation via Reinforcement Learning
description: >-
  [NeurIPS 2025][图像生成][MLLM] 提出 Janus-Pro-R1，通过两阶段训练（SFT + RL）实现视觉理解与生成的协同共进，让 MLLM 在文本到图像生成中形成真正的 Chain-of-Thought 并触发 Aha 时刻，在 GenEval 上超越 GPT-4o，同时拓展到图像编辑任务。
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "MLLM"
  - "视觉生成"
  - "reinforcement-learning"
  - "Chain-of-Thought"
  - "Aha Moment"
---

# Janus-Pro-R1: Advancing Collaborative Visual Comprehension and Generation via Reinforcement Learning

**会议**: NeurIPS 2025  
**arXiv**: [2506.01480](https://arxiv.org/abs/2506.01480)  
**代码**: [https://janus-pro-r1.github.io](https://janus-pro-r1.github.io) (项目主页)  
**领域**: 多模态大语言模型 / 图像生成  
**关键词**: MLLM, 视觉生成, reinforcement-learning, Chain-of-Thought, Aha Moment

## 一句话总结

提出 Janus-Pro-R1，通过两阶段训练（SFT + RL）实现视觉理解与生成的协同共进，让 MLLM 在文本到图像生成中形成真正的 Chain-of-Thought 并触发 Aha 时刻，在 GenEval 上超越 GPT-4o，同时拓展到图像编辑任务。

## 研究背景与动机

当前多模态大语言模型（MLLMs）虽然将视觉理解和生成统一在同一个 next-token prediction 框架中，但这两种能力实际上仍然是独立运作的——视觉理解并不能增强视觉生成，LLM 强大的推理机制也没有被充分整合到图像生成中。例如 Janus-Pro 这样的 SOTA 模型在文本到图像生成上仍不令人满意，且仅支持纯文本输入进行生成。

核心矛盾在于：**视觉理解和生成之间缺乏协同效应**。本文的切入角度是让 MLLM 能够将理解和生成能力自然地协同合作，将图像生成变成一个**迭代内省过程**——生成图像后自我评估，发现问题后重新生成，形成真正的 CoT 推理链并触发"Aha 时刻"。

## 方法详解

### 整体框架

Janus-Pro-R1 采用两阶段训练：
1. **监督微调（SFT）阶段**：通过混合子任务训练，教会 MLLM 构建视觉生成 CoT 的基本能力
2. **强化学习（RL）阶段**：通过 GRPO 算法在探索-利用权衡中释放模型全部潜力，从模仿升级为真正的推理

### 关键设计

1. **混合子任务训练（SFT 阶段）**：将视觉生成 CoT 分解为三个子任务进行混合训练——

    - **Task-I 文本到图像生成**：选择语义一致性分数 S≥0.8 的文本-图像对进行标准 T2I 训练
    - **Task-II 文本-图像一致性自评估**：训练模型判断生成图像是否与文本语义一致，并给出判断理由
    - **Task-III 图像重新生成**：给定之前错误生成的上下文，训练模型纠正错误并重新生成更准确的图像
    - 训练数据使用 200K prompt，每个 prompt 由 FLUX 和 Janus-Pro 生成 18 张图像，用 InternVL2.5-26B 评估语义一致性分数

2. **双层 QA 奖励的强化学习（RL 阶段）**：

    - 使用 GRPO 算法，将图像生成视为长 token 级马尔可夫决策过程
    - 设计**双层奖励**：生成奖励 R^Gen（评估生成图像质量）和理解奖励 R^Comp（评估自我评估的准确性）
    - 生成奖励对最终输出图像赋予更高权重；理解奖励衡量模型自评估与外部评估器的一致程度
    - 使用 InternVL2.5-26B 作为奖励模型，无需真实图像标注

3. **从文本到图像生成扩展到图像编辑**：

    - 图像编辑本质上也需要理解编辑指令 + 生成新图像，与内省式生成共享相同基础
    - 同样采用 SFT + RL 两阶段训练
    - RL 奖励包括跟随分数 R^flw（是否准确执行编辑指令）和保持分数 R^psv（未编辑区域是否保持不变）

### 损失函数 / 训练策略

- SFT 阶段：三个子任务按 0.2:0.3:0.5 比例混合训练，50K 步，学习率 2e-5
- RL 阶段：GRPO 目标函数带 KL 散度约束（β=0.05），组大小 7，3K 步
- 训练稳定性技巧：采用线性+余弦学习率调度器；奖励曲线下降时降低学习率或更新参考模型

## 实验关键数据

### 主实验（文本到图像生成）

| 基准 | 指标 | Janus-Pro-R1 (Aha) | Janus-Pro-7B | GPT-4o | 提升 |
|------|------|-----|------|------|------|
| GenEval | Overall↑ | **0.86** | 0.80 | 0.85 | +7.5% |
| T2I-CompBench | Avg↑ | **72.7** | 49.4 | - | +47.0% |
| DPG-Bench | Score↑ | **85.57** | 84.17 | - | +1.7% |
| GenEval | Counting↑ | 0.66 | 0.59 | 0.85 | +11.9% |
| GenEval | Position↑ | **0.87** | 0.79 | 0.75 | +10.1% |
| GenEval | ColorAttr↑ | **0.78** | 0.66 | 0.66 | +18.2% |

### 消融实验

| 配置 | GenEval Overall | 说明 |
|------|--------|------|
| Janus-Pro-7B 基线 | 0.80 | 原始模型 |
| SFT (w/o aha) | 0.81 | SFT 仅带来微小提升 |
| SFT (with aha) | 0.81 | 内省机制初步生效 |
| R1 (w/o aha) | 0.83 | RL 显著提升初始生成质量 |
| R1 (with aha) | **0.86** | 内省+RL 达到最佳 |
| R1-1B (with aha) | 0.71 | 小模型无法有效激活 Aha |
| 仅 Task-I SFT | 0.79 | 单任务效果不如混合 |
| 仅 Task-II+III SFT | 0.76 | 缺少 T2I 基础能力 |

### 关键发现

- Janus-Pro-R1 在 GenEval Overall 上超越 GPT-4o（0.86 vs 0.85）
- SFT 偏向模仿记忆，RL 才能实现真正泛化（反事实生成如"方形苹果"可验证）
- 1B 参数模型无法有效激活 Aha 时刻，体现 scaling law
- 数据质量比数量更重要：高阈值筛选虽减少数据量但显著提升性能
- Janus-Pro-R1 作为图像语义评估器，与 GenEval 标准的一致率达 81.1%，超越 InternVL2.5-8B

## 亮点与洞察

- **"Aha 时刻"的概念非常精炼**：将扩散模型/AR 模型的图像生成变成迭代内省过程，让模型自己发现并纠正问题
- **双层奖励设计巧妙**：同时奖励生成质量和自评估准确性，推动理解与生成的真正协同
- **RL 是关键催化剂**：SFT 只是冷启动，RL 才是让模仿变为真正推理的关键
- **统一生成的正确路径**：证明了理解+生成的协同可以自然地扩展到图像编辑等高级任务
- 反事实生成案例（方形苹果）很好地展示了 RL 后模型的推理泛化能力

## 局限与展望

- Counting 相关训练数据较少，计数能力落后于 GPT-4o
- 图像编辑数据集美学质量不高，影响编辑结果的美观度
- 当前仅在简单 T2I 任务上验证了理解-生成协同，更复杂的交错图文生成任务有待探索
- 1B 模型无法受益于此训练范式，小模型适用性受限
- 训练需要 32 张 A800 GPU，计算资源需求较大

## 相关工作与启发

- 与 T2I-R1、MINT、GOT 等同期引入 CoT 的工作相比，本文强调 CoT 应由模型深度思考自发产生，而非强制文本规划
- 借鉴 DeepSeek-R1 的 GRPO 算法思路，将 RL 用于视觉生成推理
- 对统一多模态模型的启发：RL 可能是实现真正理解-生成协同的关键训练范式

## 评分

- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Co-Reinforcement Learning for Unified Multimodal Understanding and Generation](coreinforcement_learning_for_unified_multimodal_understandin.md)
- [\[NeurIPS 2025\] Towards Robust Zero-Shot Reinforcement Learning](towards_robust_zero-shot_reinforcement_learning.md)
- [\[NeurIPS 2025\] RLVR-World: Training World Models with Reinforcement Learning](rlvr-world_training_world_models_with_reinforcement_learning.md)
- [\[CVPR 2026\] PaCo-RL: Advancing Reinforcement Learning for Consistent Image Generation with Pairwise Reward Modeling](../../CVPR2026/image_generation/paco-rl_advancing_reinforcement_learning_for_consistent_image_generation_with_pa.md)
- [\[CVPR 2025\] GraphGPT-o: Synergistic Multimodal Comprehension and Generation on Graphs](../../CVPR2025/image_generation/graphgpt-o_synergistic_multimodal_comprehension_and_generation_on_graphs.md)

</div>

<!-- RELATED:END -->
