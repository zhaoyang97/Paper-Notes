# Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning

**会议**: NeurIPS 2025  
**arXiv**: [2502.18080](https://arxiv.org/abs/2502.18080)  
**代码**: [RUCBM/TOPS](https://github.com/RUCBM/TOPS)  
**领域**: model_compression / LLM reasoning  
**关键词**: test-time compute, Chain-of-Thought, reasoning scaling, overthinking, self-improvement, optimal reasoning effort

## 一句话总结

揭示了过度延长 CoT 长度会损害 LLM 推理性能，并提出 Thinking-Optimal Scaling (TOPS) 策略，让模型为每道题选择最短正确响应进行自我提升，在效果和效率上同时优于现有蒸馏方法。

## 研究背景与动机

1. **System-2 思维范式兴起**：以 OpenAI o1 为代表的推理模型通过延长 CoT 来实现搜索、反思、回溯等深度推理，在复杂任务上取得显著提升。
2. **后续工作追求更长 CoT**：QwQ-32B-Preview、DeepSeek-R1 等模型通过蒸馏或 RL 进一步扩展推理 token 数量，试图获得更好的性能。
3. **"overthinking" 效率问题**：已有并发工作指出 o1-like 模型对简单问题生成过多冗余 token，但仅关注效率，未探讨对正确率的影响。
4. **作者核心关切**：过度追求更长 CoT 是否会在某些领域反而 *降低* 推理准确率？这是一个比效率更深层的问题。
5. **初步观察**：对比 QwQ-32B-Preview 与 Qwen2.5-32B-Instruct，前者使用显著更多 token 但性能提升有限，暗示长 CoT 并非总有益。
6. **研究目标**：系统性地研究 CoT 长度缩放对推理性能的影响，并设计"思考最优"的缩放策略，让模型自适应决定每道题所需的推理深度。

## 方法详解

### 整体框架：TOPS (Thinking-OPtimal Scaling)

TOPS 包含三个阶段，核心思想是让模型为每道题找到"最短正确响应"作为训练目标：

#### 阶段一：Format Imitation（格式模仿）

- 使用少量种子数据（约 1.3K 题，每题 3 种推理深度的响应，共约 3.9K 样本）训练 **tag 模型**
- 通过 3 种 system prompt 控制低/中/高推理努力（Low/Medium/High），引导 QwQ-32B-Preview 生成不同长度的正确 CoT
- 对同一题的 3 个响应按实际长度重排，要求相邻长度差 > 300 tokens，确保推理深度确实不同
- 用种子数据微调基座模型，使其学会根据不同 system prompt 采用不同推理深度

#### 阶段二：Reasoning Effort-Conditioned Generation（条件生成）

- 用 tag 模型在额外 50K 数学问题上分别以低/中/高推理努力各生成 1 个响应
- 对每道题在 3 个响应中选出 **最短的正确响应** 作为 thinking-optimal 响应
- 与种子数据中低推理努力的响应合并，得到约 26K 样本的 thinking-optimal 数据集

#### 阶段三：Self-Improvement（自我提升）

- 用 thinking-optimal 数据集对基座模型做 SFT（学习率 $1 \times 10^{-5}$，batch size 96，2 epoch）
- 得到最终的 TOPS 模型，能自适应为简单题分配少量 token、为难题分配更多 token

### 关键设计

- **不同于固定长度蒸馏**：STILL-2 / Sky-T1 直接用 o1-like 模型的原始长度分布做蒸馏，TOPS 则通过多推理深度生成 + 最短正确选择，获得更优的长度分布
- **不同于随机选择**：对照实验 Qwen2.5-32B-Random 随机选正确响应，性能一致低于最短正确响应选择

### 训练策略：迭代自我提升

- **Iter-SFT**：在额外 4500 道 MATH 问题 + AIME1983-2023 上，从 TOPS 模型采样 8 个响应，选最短正确响应继续 SFT
- **Iter-DPO**：构建偏好对——chosen 为最短正确响应，rejected 为最长错误响应（提升推理能力） + 最短错误但比最短正确更短的响应（避免过度简化），做 DPO 优化

## 实验关键数据

### 主实验表（Qwen2.5-32B 系列）

| 模型 | GSM8K Acc | GSM8K #Tokens | MATH500 Acc | MATH500 #Tokens | AIME2024 Acc | AIME2024 #Tokens |
|---|---|---|---|---|---|---|
| Qwen2.5-32B-Instruct (T=0) | 95.91 | 295 | 84.20 | 577 | 16.67 | 1407 |
| QwQ-32B-Preview | 95.23 | 761 | 92.02 | 2416 | **45.33** | 7637 |
| STILL-2-32B | 95.47 | 571 | 91.40 | 2005 | **45.33** | 6656 |
| Sky-T1-32B-Preview | 94.82 | 696 | 89.48 | 2022 | 35.33 | 5351 |
| **Qwen2.5-32B-TOPS** | **95.82** | **412** | 91.48 | 1883 | 43.33 | 7260 |
| **TOPS-Iter-DPO** | 95.80 | 385 | **91.60** | 1732 | **46.00** | 6427 |

### 消融与分析

| 分析维度 | 关键发现 |
|---|---|
| 推理努力 vs 难度 | 简单任务（GSM8K）Low effort 最优，困难任务（AIME2024）High effort 更优 |
| 长 CoT 的负面机制 | 更长 CoT 中错误推理步骤的数量和占比均显著增加 |
| Loss masking 验证 | 对错误步骤不计算 loss → 性能优于全步骤计算 loss，验证错误步骤有害 |
| 答案一致性 | 最优推理努力下，多次采样的不同答案数最少，说明模型最稳定 |
| TOPS vs Random 选择 | 每个 benchmark 上 TOPS（最短正确）均优于 Random（随机正确） |
| 迭代 DPO | 同时提升效果和效率，AIME2024 达到 46.00%（超越 QwQ-32B-Preview） |

### 关键发现

- 仅 1.3K 种子样本 + 自我提升即可达到甚至超越使用 3.9K 高质量蒸馏样本的 STILL-2
- TOPS 在 GSM8K 上仅用 412 tokens（QwQ 用 761），有效缓解 overthinking
- 在 LLaMA3.1-8B-Instruct 上同样有效，说明方法具有跨架构泛化能力

## 亮点与洞察

- **核心洞察：更长 ≠ 更好**。首次系统性证明过长 CoT 会引入更多错误推理步骤，反而降低准确率，颠覆了"越长越好"的朴素直觉
- **最短正确响应的优雅设计**：无需额外奖励模型或复杂搜索，仅通过多深度采样 + 最短正确选择即可自动获得 thinking-optimal 分布
- **自我提升闭环**：种子数据极少（1.3K），模型自己生成大规模训练数据并筛选，实现了从 System-1 到高效 System-2 的低成本升级
- **答案一致性指标**：发现最优推理深度下多次采样答案分布最集中，提供了一种无需 ground truth 评估推理深度适当性的间接信号
- **迭代 DPO 中的双向偏好对设计**：同时避免 overthinking 和 underthinking，比一般 DPO 考虑更周全

## 局限性 / 可改进方向

1. **领域局限**：分析和实验主要在数学推理上，因为数学有精确的正确性验证；在代码、科学推理、开放式问题等领域是否有同样的 overthinking 现象尚待探索
2. **仅 SFT 设定**：未在 RL 训练（如 GRPO、PPO）场景中验证 TOPS 策略，RL 中过度奖励长正确响应可能也有类似问题
3. **推理努力离散化**：仅 3 级推理深度（Low/Medium/High），更细粒度的连续控制可能带来更优结果
4. **单次采样**：每种推理努力仅采样 1 个响应，多次采样后取最短正确可能进一步提升
5. **依赖教师模型生成种子数据**：种子数据仍由 QwQ-32B-Preview 生成，探索无教师的纯 RL 自我进化是重要方向
6. **未结合 PRM/ORM**：如果在选择最短正确响应时结合过程奖励模型，可能获得更精细的质量-长度权衡

## 相关工作与启发

- **与 STILL-2 / Sky-T1 的关系**：这两种方法直接蒸馏 o1-like 响应，继承了教师模型的长度分布；TOPS 通过自适应选择打破了这一限制
- **与 overthinking 研究的互补**：Chen et al. (2024) 关注效率问题，TOPS 进一步揭示了效果问题，并提供了解决方案
- **与 RL-based scaling 的联系**：作者指出 RL 中给所有正确答案相同奖励（如 1.0）也存在类似问题，更短正确响应应获得更高偏好
- **对 DeepSeek-R1 等后续工作的启示**：在 RL 训练中引入长度惩罚或基于推理步骤质量的差异化奖励，可能是更优方向
- **Process Reward Model 的互补**：TOPS 选择最短正确响应是一种粗粒度的过程质量代理，与 PRM 的细粒度步骤评估可以互补

## 评分

- ⭐ 新颖性: 4/5 — 首次系统分析 CoT 长度过度缩放的负面效应，TOPS 方法虽简单但洞察深刻
- ⭐ 实验充分度: 4/5 — 多个基座模型、多难度benchmark、详尽消融、机制分析全面，但领域覆盖略窄
- ⭐ 写作质量: 4/5 — 逻辑清晰，从现象观察到原因分析再到方法设计，叙事流畅
- ⭐ 价值: 4.5/5 — 对 test-time scaling 社区提供了重要的反思视角和实用的训练策略
