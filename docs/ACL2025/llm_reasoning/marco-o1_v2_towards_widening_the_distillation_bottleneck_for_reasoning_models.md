---
title: >-
  [论文解读] Marco-o1 v2: Towards Widening The Distillation Bottleneck for Reasoning Models
description: >-
  [ACL 2025][LLM推理][Reasoning Distillation] 揭示了直接蒸馏大推理模型（如 DeepSeek-R1）的长 CoT 数据到小模型时的「形式化长时间思考」瓶颈，提出基于 MCTS 从头构造树状 CoT 数据并结合思维长度平衡、细粒度 DPO 和联合训练目标来缓解该问题。
tags:
  - "ACL 2025"
  - "LLM推理"
  - "Reasoning Distillation"
  - "MCTS"
  - "Chain-of-Thought"
  - "DPO"
  - "Formalistic Thinking"
---

# Marco-o1 v2: Towards Widening The Distillation Bottleneck for Reasoning Models

**会议**: ACL 2025  
**arXiv**: [2503.01461](https://arxiv.org/abs/2503.01461)  
**代码**: [AIDC-AI/Marco-o1](https://github.com/AIDC-AI/Marco-o1)  
**领域**: LLM Reasoning / 知识蒸馏  
**关键词**: Reasoning Distillation, MCTS, Chain-of-Thought, DPO, Formalistic Thinking  

## 一句话总结

揭示了直接蒸馏大推理模型（如 DeepSeek-R1）的长 CoT 数据到小模型时的「形式化长时间思考」瓶颈，提出基于 MCTS 从头构造树状 CoT 数据并结合思维长度平衡、细粒度 DPO 和联合训练目标来缓解该问题。

---

## 研究背景与动机

### 问题背景
大推理模型（LRM）如 OpenAI o1、DeepSeek-R1 通过扩展测试时计算和生成长 Chain-of-Thought (CoT) 展现出强大推理能力。将这些推理能力蒸馏到更小的模型中（如直接在 LRM 生成的数据上微调 Qwen2.5 7B）是一种高效策略。例如 DeepSeek-R1 蒸馏模型在 AIME 上从 GPT-4 的 9.3% 提升到 55.5%。

### 核心问题：形式化长时间思考
作者发现蒸馏后的小模型经常出现**形式化长时间思考（Formalistic Long-time Thinking）**——机械模仿大模型的推理模式但未真正内化推理逻辑。具体表现为三类错误：

**内容重复（Content Repetition）**：模型反复生成相同的文本片段，无法推进推理（如 "positions are considered up to consider that the positions are..." 的死循环）

**过度反思（Over-Reflection）**：模型不断用 "Wait, perhaps..."、"Alternatively,..." 等模式自我质疑但无法收敛到答案

**指令失败（Instruction Failure）**：在翻译等简单任务中陷入不必要的长推理，最终无法给出答案

### 根本原因
- 蒸馏数据中的长 CoT 对小模型而言存在**学习困难**
- SFT 和 RL 方法导致**偏差继承**（如过度思考模式）
- DPO 训练对响应长度敏感，加剧形式化思考

### 研究问题
如何通过数据构造、SFT 和 RL 方法有效地将长 CoT 推理迁移到小模型？

---

## 方法详解

### 整体框架
分为两部分：
1. **数据侧**：基于 MCTS 从头构造树状 CoT 数据（而非从 LRM 蒸馏）
2. **方法侧**：CoT 感知的后训练技术（Thoughts Length Balance + Fine-grained DPO + Joint Objective）

### 关键设计

#### 1. 基于 MCTS 的 CoT 数据构造

**思维节点（Thought Node）定义**：

| 节点类型 | 作用 | 前缀提示 |
|---------|------|---------|
| Thinking | 开放式推理延续 | （无，直接续写） |
| Sub-Task | 任务分解 | "Firstly, I need to break down this task." |
| Reflection | 检查与纠错 | "Let's check the result. Wait! something is wrong..." |
| Hypothesis | 假设提出 | "I propose the following hypothesis:" |
| Double-Check | 验证 | "Now, I need to check whether all requirements are met." |
| Reclarify | 重新阐明 | "To ensure clarity, let me restate..." |
| Answer | 给出答案 | "The answer is:" |

**MCTS 搜索过程**：
1. **节点选择**：使用 UCB 公式平衡探索与利用
    $UCB(n_i) = \frac{v(n_i)}{n_{\text{visits}}(n_i)} + C\sqrt{\frac{\ln(n_{\text{visits}}(n_{\text{parent}}))}{n_{\text{visits}}(n_i)}}$
2. **扩展**：按预定义的节点转移矩阵，提示 LLM 生成该节点类型的内容
3. **Rollout**：到达 Answer 节点时基于规则计算正确性奖励
4. **回传**：将奖励回传到树中

**多模型协作**：
- Thinking 节点使用 Qwen2.5-72B-Instruct
- Reflection 节点切换到 Llama3.1-70B-Instruct
- 当同一模型自纠错时容易复用相同的错误分布，切换模型可减少重复错误

**推理模式多样性**：
设计 4 种不同的节点转移模式（如 Sub-Task→Thinking→Answer、Sub-Task→Hypothesis→Thinking→Answer 等），随机采样以产生多样化推理路径。

**数据提取**：
- **SFT 数据**：选择到达正确答案的成功路径（最高奖励路径或特定长度路径）
- **DPO 数据**：正例为正确路径，负例为与正例共享最短公共前缀的错误路径

#### 2. Thoughts Length Balance（思维长度平衡）

- 发现 CoT 长度对 DPO 阶段影响显著但对 SFT 影响不大
- 策略：**SFT 阶段用最长 CoT，DPO 阶段用最短 CoT**
- 从 CoT 树中按相对长度（短/中/长）提取路径，而非设定固定 token 阈值
- 较短推理路径减少了无效输出，缓解形式化长时间思考

#### 3. Fine-grained DPO（细粒度 DPO）

**Conservative DPO (cDPO)**：
- 处理噪声偏好标签，设置偏好概率 $p(y_w \succ y_l) = 1 - \epsilon$
- 修正后的损失函数：
  $$\mathcal{L}_{\text{DPO}}^{\epsilon}(\theta, y_w, y_l) = -(1-\epsilon)\log\hat{p}_\theta(y_w \succ y_l) - \epsilon\log(1-\hat{p}_\theta(y_w \succ y_l))$$
- 通过软化梯度更新减少噪声标签的影响

**Masking-based DPO**：
- 识别正负样本的公共前缀 token 数量
- 将公共前缀 token 的损失掩码设为零（类似 padding token 处理）
- 确保模型聚焦于区分性部分而非共享前缀

#### 4. Joint Post-training Objective（联合后训练目标）

- 纯 DPO 训练导致灾难性遗忘和分布偏移
- 在 DPO 损失中加入 SFT 损失：$\mathcal{L} = \mathcal{L}_{\text{DPO}} + \alpha \mathcal{L}_{\text{SFT}}$
- $\alpha = 1$ 为最佳权衡点

---

## 实验

### 实验设置
- **基础模型**：Llama-3.1-8B-Instruct、Llama-3.2-1B、Qwen2.5-7B/1.5B-Instruct
- **基准**：GSM8K（初等数学）、MATH500（高级数学）、AIME（竞赛数学）、Blocksworld（规划）、Multi-IF（8种语言指令跟随）
- **对比**：Sky-T1 数据集（基于 QwQ 32B 蒸馏）

### SFT 数据对比

| 模型 | 数据 | GSM8K | MATH | AIME | Blocksworld | IF(Zh) | IF(En) | IF(Other) |
|------|------|-------|------|------|-------------|--------|--------|-----------|
| Llama-3.1-8B | 基线 | 85.5 | 47.0 | 11.7 | 10.0 | 61.5 | 76.2 | 67.1 |
| | +Sky-T1 | 84.8 | 44.0 | 6.7 | 2.0 | 25.4 | 31.6 | 29.7 |
| | **+Our Data** | **87.4** | **51.4** | **15.0** | **12.4** | **69.2** | **76.6** | **79.1** |
| Qwen2.5-7B | 基线 | 90.4 | 62.0 | 15.0 | 10.6 | 69.6 | 72.8 | 74.4 |
| | +Sky-T1 | 89.6 | 61.6 | 9.4 | 0.4 | 26.2 | 24.5 | 30.6 |
| | **+Our Data** | **90.7** | **64.0** | **15.0** | **12.0** | **73.1** | **73.4** | **78.8** |

关键观察：
- Sky-T1 数据在 8B 模型上**全面降低性能**（IF 任务降幅达 35-50%），验证了蒸馏瓶颈
- 本文构造的数据在所有任务上提升性能，对小模型（1B）改进更为显著

### 后训练方法逐步叠加（Llama-3.1-8B）

| 方法 | GSM8K | MATH | AIME | Plan. | IF(Zh) | IF(En) | IF(Other) |
|------|-------|------|------|-------|--------|--------|-----------|
| SFT 基线 | 87.4 (0.23%) | 51.4 (5.4%) | 15.0 (30%) | 12.4 (1.8%) | 69.2 (0.77%) | 76.6 (1.69%) | 79.1 (1.08%) |
| + DPO | 86.2 (6.37%) | 41.8 (31.8%) | 8.3 (55%) | 2.0 (93.6%) | 5.7 (91.5%) | 6.3 (90.9%) | 6.7 (92.2%) |
| + Data Balance | 86.8 (5.08%) | 28.0 (46.4%) | 6.6 (65%) | 6.8 (44.6%) | 43.4 (30.8%) | 44.7 (44.7%) | 42.4 (45.3%) |
| + cDPO | 87.5 (3.71%) | 48.6 (15%) | 15.0 (45%) | 4.4 (47.4%) | 61.9 (11.2%) | 66.4 (15.6%) | 67.7 (15.4%) |
| + Joint Loss | 86.8 (0.38%) | 48.6 (8.6%) | 10.0 (31.7%) | 8.6 (9%) | 72.3 (1.15%) | 78.9 (1.9%) | 78.1 (2.22%) |
| + Masking | **87.2** (0.15%) | **51.0** (5.8%) | 8.0 (38.3%) | **12.6** (10.2%) | **72.0** (1.15%) | **77.2** (1.9%) | **79.1** (1.36%) |

*（括号内为无答案输出的比例）*

关键发现：
1. **纯 DPO 灾难性**：在 Planning 和 IF 任务上无答案比例达 90%+，性能崩塌
2. **逐步修复有效**：每个技术正交互补，最终恢复到接近/超过 SFT 基线水平
3. **改进主要来自减少无答案输出**：Joint Loss 和 Masking 将无答案比例从 90%+ 降到 <10%

### Joint Loss α 超参数

| α | GSM8K | MATH | Plan. | IF(Zh) |
|---|-------|------|-------|--------|
| cDPO (α=0) | 87.5 | 48.6 | 4.4 | 61.9 |
| α=0.5 | 86.5 | 50.0 | 7.8 | 68.8 |
| **α=1.0** | **86.8** | **48.6** | **8.6** | **72.3** |
| α=1.5 | 85.5 | 48.4 | 7.6 | 68.4 |
| α=2.0 | 85.6 | 48.0 | 8.4 | 70.7 |

α=1 为最佳平衡点。

### MCTS 推理探索

| 模型 | Test@1 | Test@8 | Test@32 |
|------|--------|--------|---------|
| Llama-3.1-8B 基线 | 47.0 | 67.6 | 75.8 |
| Our Best Model | 51.0 | 70.2 | 79.2 |
| + MCTS Decode | 51.0 | **70.8** | **82.8** |

MCTS 推理在 Test@32 上额外提升 3.6%，展示了测试时计算扩展的潜力。

---

## 亮点与洞察

1. **揭示了蒸馏瓶颈的本质**：形式化长时间思考不是简单的性能不足，而是小模型机械模仿推理模式的表现。这比"小模型推理能力弱"的泛泛说法更具洞察力
2. **从头构造 CoT 数据**优于从 LRM 蒸馏数据，是本文最重要的实证贡献
3. **多模型协作的 MCTS 框架**设计精巧：Qwen 负责推理，Llama 负责反思，避免同模型自纠错的分布偏差
4. **DPO 在长 CoT 上的失效**是一个重要发现（Planning 和 IF 上无答案比例 90%+），揭示了标准 DPO 不适合直接用于推理模型
5. **五种技术正交互补**：数据平衡、cDPO、Joint Loss、Masking 各解决不同问题，组合使用效果显著
6. **定量分析形式化思考**：通过无答案比例精确衡量问题严重程度

## 局限性

1. 基础模型仅测试了 Llama 和 Qwen 系列，未覆盖其他模型家族
2. MCTS 数据构造需要大量 LLM 推理调用（Qwen-72B + Llama-70B），成本较高
3. AIME 上最终性能（8.0-15.0%）仍然较弱，复杂数学推理的瓶颈未完全突破
4. Masking DPO 在 AIME 上反而降低了性能（从 10.0% 到 8.0%），说明技术组合并非对所有任务都是正向的
5. 未与 DeepSeek-R1 蒸馏模型直接对比

## 相关工作

- **推理模型**：OpenAI o1、DeepSeek-R1 (Guo et al. 2025)、QwQ (Qwen Team 2024)
- **知识蒸馏**：直接蒸馏 (DeepSeek-R1)、Sky-T1
- **MCTS 用于推理**：Tian et al. 2024、RStar (Qi et al. 2024)、Math-Shepherd (Wang et al. 2024)
- **DPO 改进**：cDPO (Mitchell 2023)、Joint SFT+DPO (Fernando et al. 2024)

---

## 评分 ⭐⭐⭐⭐

对推理蒸馏瓶颈的分析深入且有实证支撑，提出的 MCTS CoT 构造框架具有原创性。技术组合全面且每个组件都有合理动机。不足在于对竞赛级数学推理的改进有限，以及缺少与主流蒸馏方法（如 DeepSeek-R1 蒸馏）的直接对比。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Revisiting the Test-Time Scaling of o1-like Models: Do they Truly Possess Test-Time Scaling Capabilities?](revisiting_the_test-time_scaling_of_o1-like_models_do_they_truly_possess_test-ti.md)
- [\[CVPR 2025\] Enhancing Video-LLM Reasoning via Agent-of-Thoughts Distillation](../../CVPR2025/llm_reasoning/enhancing_video-llm_reasoning_via_agent-of-thoughts_distillation.md)
- [\[ACL 2025\] ClozeMath: Improving Mathematical Reasoning in Language Models by Learning to Fill Equations](clozemath_improving_mathematical_reasoning_in_language_models_by_learning_to_fil.md)
- [\[ICML 2026\] The Role of Feedback Alignment in Self-Distillation](../../ICML2026/llm_reasoning/the_role_of_feedback_alignment_in_self-distillation.md)
- [\[ACL 2025\] Improve Vision Language Model Chain-of-thought Reasoning](improve_vlm_cot_reasoning.md)

</div>

<!-- RELATED:END -->
