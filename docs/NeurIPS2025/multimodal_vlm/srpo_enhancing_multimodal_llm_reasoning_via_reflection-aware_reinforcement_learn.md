# SRPO: Enhancing Multimodal LLM Reasoning via Reflection-Aware Reinforcement Learning

**会议**: NeurIPS 2025  
**arXiv**: [2506.01713](https://arxiv.org/abs/2506.01713)  
**代码**: https://srpo.pages.dev/  
**领域**: 多模态VLM / LLM推理  
**关键词**: 多模态推理, 自我反思, 强化学习, GRPO, Reflection-Aware RL

## 一句话总结

提出 SRPO（Self-Reflection enhanced reasoning with Group Relative Policy Optimization），一个两阶段反思感知 RL 框架：第一阶段用大模型生成反思数据做 SFT cold-start，第二阶段设计反思感知奖励函数在 GRPO 中强化简洁有效的自我反思能力，在 MathVista/MathVision/MMMU-Pro 等多模态推理基准上以 7B/32B 模型显著超越同规模 SOTA。

## 研究背景与动机

1. **领域现状**: 多模态大模型（MLLM）在推理任务中展现了潜力，DeepSeek-R1 等工作将 RL-based 推理从文本扩展到多模态场景。但现有方法（MM-Eureka、Vision-R1、VL-Rethinker 等）在 7B 规模模型上仍难以匹配闭源模型的推理表现。
2. **现有痛点**: (1) MLLM 生成过程遵循 token 级 Markov 过程，依赖局部依赖关系，导致冗余、重复或错误的推理步骤；(2) GPT-o1 在 MathVista 上 73.9% 甚至不如 Qwen2.5-VL-72B 的 74.8%，说明错误和冗余步骤会拖累最终性能。
3. **核心矛盾**: 自我反思是解决冗余/错误推理的有效手段，但预训练阶段已经基本固定了模型的推理能力上限。RL 只能激活已有的决策结构而非获取新知识——要突破这个上限需要外部干预（如高质量反思经验的注入）。
4. **本文要解决什么？** 如何让 MLLM 学会有效的自我反思和自我纠错，从而突破预训练阶段设定的推理能力上界。
5. **切入角度**: 受认知科学启发——人类的稳健推理涉及主动自我反思和迭代纠错。将显式反思方法同时融入 SFT 和 RL 两个阶段。
6. **核心idea一句话**: 两阶段训练——先用大模型蒸馏的反思数据做 SFT 注入反思能力，再用反思感知奖励函数在 GRPO 中强化简洁有效的反思行为。

## 方法详解

### 整体框架

SRPO 分两个阶段：
- **Stage 1 (Reflection-oriented Cold-start SFT)**: 构建高质量反思数据集，训练 policy model 获得基础反思能力
- **Stage 2 (Reflection-aware RL)**: 在 GRPO 框架中引入专门的反思奖励函数，进一步强化反思行为

生成格式：first solution → `<reflection>...</reflection>` → second refined solution

### 关键设计

1. **Self-Reflection SFT 数据构建**:
   - 从 LLaVA-CoT (100K)、Mulberry (260K)、MathV360K 中精选 N=10,000 个多模态推理样本
   - 用 policy model (如 Qwen-2.5-VL-7B) 生成初始回答
   - 用大模型 (GPT-o4-mini) 基于 ground truth 生成反思过程
   - 两种互补策略：**正确 CoT 的精简**（去除冗余）+ **错误 CoT 的修正**（纠错）
   - 数据中约 30% 初始回答正确、70% 包含错误，凸显反思的必要性
   - "Less is More"——仅用 10K 样本即可有效注入反思能力

2. **Cold-start SFT 训练**: 目标函数 L = -E[log π(a1, `<reflection>...</reflection>`, a2 | q)]，其中 a1 是策略模型初始回答，reflection 是大模型生成的反思，a2 是 ground truth。模型同时学习：(1) 通过反思从 a1 修正到 a2；(2) 利用 a2 中的推理知识指导未来预测。

3. **Reflection-Aware Reward (SRPO 核心)**:
   - **总奖励** R_total = R_task + R_reflection
   - **Task Reward**: R_format (0.5 if 格式正确) + R_accuracy (0.5 if 第一次解答正确)
   - **Reflection Reward** = I_eff + I_ref + α·f_len(L)
     - **I_eff (有效性指标)**: 反思纠正错误答案 +0.5，保持正确答案 +0.25，未能纠正 0，把正确改错 -0.25
     - **I_ref**: 反思格式正确 +0.25
     - **f_len**: 长度奖励，以 exp(-|L-T_target|/(T_max-T_target))^2 鼓励简洁输出

### 损失函数 / 训练策略

- RL 阶段使用 GRPO 优化，组内归一化计算 advantage：A_i = (r_i - mean(r)) / std(r)
- 关键改进：奖励信号不仅关注准确率，还直接对反思行为的「有效性」、「简洁性」、「格式」给予细粒度奖励
- SFT 数据来自 ScienceQA、Geometric Math QA、ChartQA、DVQA、AI2D、MATH、Virgo、R1-OneVision、MMK12、PhyX 等多来源聚合

## 实验关键数据

### 主实验

#### 7B 模型对比

| 模型 | MathVista | MathVerse | MathVision | MMMU-Pro | EMMA |
|------|-----------|-----------|------------|----------|------|
| Qwen-2.5-VL-7B | 68.2 | 46.3 | 25.1 | 36.9 | 21.5 |
| VL-Rethinker-7B | 74.9 | 54.2 | 32.3 | 41.7 | 29.7 |
| Vision-R1-7B | 73.5 | 52.4 | 27.2 | 37.7 | 22.4 |
| MM-Eureka-7B | 73.0 | 50.3 | 26.9 | 37.6 | 23.5 |
| **SRPO-7B** | **75.8** | **55.8** | **32.9** | **42.3** | **29.6** |

#### 32B 模型对比

| 模型 | MathVista | MathVerse | MathVision | MMMU-Pro | EMMA |
|------|-----------|-----------|------------|----------|------|
| Qwen-2.5-VL-32B | 74.7 | 48.5 | 38.4 | 49.5 | 31.1 |
| MM-Eureka-32B | 74.8 | 56.5 | 34.4 | 50.4 | 34.5 |
| **SRPO-32B** | **78.5** | **58.9** | **39.6** | **51.3** | **38.2** |

SRPO-7B 在所有基准上全面超越同规模开源推理模型；SRPO-32B 在 MMMU-Pro 上达到 51.3，接近闭源 Claude3.7-Sonnet 的 51.5。

### 消融实验

| 消融维度 | 关键发现 |
|----------|----------|
| 去掉 I_eff (有效性奖励) | 性能显著下降，模型生成空洞反思 |
| 去掉 f_len (长度奖励) | 输出冗余增加，反思段落过长 |
| 仅用 SFT 不做 RL | 性能低于完整 SRPO，验证 RL 阶段的必要性 |
| 标准 GRPO 无反思奖励 | 模型不生成有意义的反思行为 |

### 关键发现

1. 两阶段训练缺一不可：SFT 注入反思能力，RL 强化反思质量
2. I_eff 奖励设计是核心——区分"保持正确"和"纠正错误"给予不同奖励，防止模型把对的改错
3. 长度奖励有效防止 reward hacking（通过膨胀输出长度获得高奖励）
4. 仅用 10K SFT 数据即可实现有效的反思能力注入

## 亮点与洞察

1. **反思感知奖励设计精巧**: I_eff 的四档奖励（+0.5/+0.25/0/-0.25）直接与反思是否真正改善结果挂钩，避免"为反思而反思"
2. **SFT 数据构建策略新颖**: "Less is More"——10K 样本覆盖正确精简和错误修正两种场景，比前人的大规模 CoT 蒸馏更高效
3. **跨规模一致性**: 7B 和 32B 上均有显著提升，说明方法的通用性
4. **补充了多模态推理中的反思空白**: 前人（VL-Rethinker、Vision-R1）未在 SFT+RL 两阶段都显式强化反思

## 局限性 / 可改进方向

1. 反思数据构建依赖闭源大模型 (GPT-o4-mini)，蒸馏成本存在
2. 反思格式固定为 `<reflection>` 标签包裹，未探索更灵活的反思形式
3. 当前仅在数学和通用科学推理上验证，未涵盖开放式视觉推理
4. I_eff 的四档阈值为手动设定，未探索更细粒度或自适应的奖励
5. 未与 RL-based 方法（如 DAPO、Dr.GRPO）做深度对比

## 相关工作与启发

- **VL-Rethinker**: 用选择性样本回放和文本 rethinking trigger 改善推理，但未在 SFT 阶段注入反思知识
- **MM-Eureka**: 提出 MMK12 数据集 + 两阶段 RL，但无显式反思机制
- **Vision-R1**: 用 DeepSeek-R1 增强 CoT 数据 + 渐进思维抑制，仅在 SFT 端改进
- **DeepSeek-R1**: 验证了 RL 在文本推理上的有效性，SRPO 将此扩展到多模态 + 反思
- 启发：反思机制可推广到视觉生成、具身智能等需要多步纠错的场景

## 评分

- 新颖性: ⭐⭐⭐⭐ — 反思感知奖励函数设计在 GRPO 框架中是首创，SFT+RL 双阶段反思注入新颖
- 实验充分度: ⭐⭐⭐⭐⭐ — 8 个基准、7B/32B 两个规模、详细消融和对比
- 写作质量: ⭐⭐⭐⭐ — 动机清晰，公式完整，图表丰富
- 价值: ⭐⭐⭐⭐ — 为多模态推理中的自我反思提供了可复现的训练范式
