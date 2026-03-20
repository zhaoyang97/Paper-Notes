# ARM: Adaptive Reasoning Model

**会议**: NeurIPS 2025  
**arXiv**: [2505.20258](https://arxiv.org/abs/2505.20258)  
**代码**: [https://team-arm.github.io/arm](https://team-arm.github.io/arm)  
**领域**: LLM推理效率 / 强化学习  
**关键词**: 自适应推理, overthinking, GRPO, 推理格式选择, test-time compute  

## 一句话总结
ARM 通过让模型自适应地选择四种推理格式（直接回答、短CoT、代码、长CoT），配合改进的 Ada-GRPO 训练算法解决 format collapse 问题，在保持与纯长CoT模型持平的准确率的同时平均节省 ~30% token，最多节省 ~70%。

## 研究背景与动机

1. **领域现状**：以 OpenAI-o1、DeepSeek-R1 为代表的大型推理模型（LRM）通过长链式思维（Long CoT）在复杂任务上取得突破。这些模型通过 test-time scaling——生成更多 token——来提升推理能力。

2. **现有痛点**：LRM 对所有任务一律使用 Long CoT，导致严重的"overthinking"问题。简单的常识问答（如"maid 在哪里倒垃圾？"）也会生成数百 token 的推理过程，不仅浪费计算资源，冗长输出甚至会引入噪声导致错误答案。实验显示 DeepSeek-R1 在简单任务上使用近 4× token 却没有性能提升甚至出现退化。

3. **核心矛盾**：Long CoT 不是万能药——简单任务不需要复杂推理，但现有 GRPO 训练只优化准确率，会导致模型收敛到 Long CoT 这一单一格式（format collapse），完全放弃其他更高效的推理方式。现有缓解方案（如长度惩罚）依赖人工预估 token 预算，估计不准时性能会大幅下降。

4. **本文要解决什么**：训练一个能自主根据任务难度选择合适推理格式的模型，无需人工干预 token 预算。

5. **切入角度**：将推理效率问题转化为"推理格式选择"问题——不是缩短 Long CoT 的长度，而是让模型学会在简单任务上用 Direct Answer，中等任务上用 Short CoT 或 Code，复杂任务上用 Long CoT。

6. **核心idea一句话**：通过 Ada-GRPO 中的格式多样性奖励缩放因子，防止 GRPO 训练中的 format collapse，使模型能根据任务难度自适应切换四种推理格式。

## 方法详解

### 整体框架
ARM 采用两阶段训练框架。输入是问题，输出是模型自动选择的推理格式（Direct Answer / Short CoT / Code / Long CoT）所产生的答案。四种格式通过特殊 token（如 `<Code></Code>`）来标识。

- **Stage 1 (SFT)**：用 10.8K 问题（每题配四种格式的标注答案）做监督微调，让模型理解四种推理格式的基本使用方式。
- **Stage 2 (Ada-GRPO)**：用 19.8K 可验证问答对做强化学习训练，通过格式多样性奖励机制教会模型根据任务难度自适应选择格式。

### 关键设计

1. **四种推理格式分级**：
   - 做什么：将推理策略分为 Direct Answer（零推理）、Short CoT（简短推理链）、Code（代码推理）、Long CoT（深度推理含反思和回溯）四档。
   - 核心思路：不同难度的任务需要不同复杂度的推理。常识题用 Direct Answer 即可（~10 token），数学题可能需要 Code（~300 token），竞赛题则需要 Long CoT（~3000 token）。
   - 设计动机：这比单纯缩短推理链更优雅——不是"少想"，而是"选对方式想"。

2. **Ada-GRPO 中的格式多样性缩放因子**：
   - 做什么：解决标准 GRPO 中的 format collapse 问题——训练约 10 步后模型就会收敛到 Long CoT。
   - 核心思路：对每个 rollout 的 reward 乘以缩放因子 $\alpha_i(t) = \frac{G}{F(o_i)} \cdot decay_i(t)$，其中 $F(o_i)$ 是该格式在 group 中出现的次数，$G$ 是 group size。出现越少的格式获得越高的 reward 放大，鼓励模型探索不同格式。
   - 设计动机：标准 GRPO 只优化准确率，Long CoT 通常准确率最高因此被不断强化。Ada-GRPO 通过反比例放大稀有格式的 reward，防止多样性丧失。

3. **Cosine Decay 机制**：
   - 做什么：让格式多样性奖励的影响随训练进程逐渐衰减。
   - 核心思路：$decay_i(t) = \frac{F(o_i)}{G} + 0.5 \cdot (1 - \frac{F(o_i)}{G}) \cdot (1 + \cos(\frac{\pi t}{T}))$。训练初期（$t=0$）格式多样性 reward 最大，训练末期（$t=T$）衰减到 1，回归纯准确率优化。
   - 设计动机：避免长期过度奖励低频格式导致的训练不稳定。实验证实去掉 decay 会导致测试准确率大幅波动。

4. **三种推理模式**：
   - Adaptive Mode（默认）：模型自主选择推理格式
   - Instruction-Guided Mode：用户通过特殊 token 指定格式，适合已知任务类型的批量推理
   - Consensus-Guided Mode：先用三个高效格式分别推理，不一致时回退到 Long CoT，以性能为优先

### 损失函数 / 训练策略
- Stage 1：标准 SFT，使用 LoRA + DeepSpeed ZeRO-3，学习率 2e-4，6 epochs
- Stage 2：Ada-GRPO 目标函数与 GRPO 相同（PPO-clip 风格 + KL 正则），核心区别在于 reward 经过格式多样性缩放。batch size 1024，每个 prompt 8 个 rollout，最大 rollout 长度 4096 token，9 epochs，8 张 A800 GPU。

## 实验关键数据

### 主实验

| 模型 (7B, maj@8) | CSQA (easy) | GSM8K (medium) | MATH (medium) | AIME'25 (hard) | 平均准确率 | 平均Token |
|---|---|---|---|---|---|---|
| Qwen2.5-7B Base | 82.0 | 89.9 | 64.7 | 3.3 | 68.3 | 260 |
| Qwen2.5-7B SFT+GRPO | 83.7 | 94.8 | 84.9 | 20.0 | 76.1 | 1164 |
| **ARM-7B** | **85.7** | **93.7** | **82.6** | **20.0** | **75.9** | **786** |
| DS-R1-Distill-7B | 64.9 | 90.0 | 93.6 | 40.0 | 75.5 | 2797 |

ARM-7B vs SFT+GRPO：准确率仅降 0.2%，token 节省 32.5%。

### 消融实验

| 配置 | 效果 | 说明 |
|---|---|---|
| SFT only → 格式分布 | 四格式近均匀分布 | SFT 教会格式但不教选择，简单题也用 Direct Answer 导致中等题准确率暴跌 |
| GRPO → 格式分布 | ~100% Long CoT | 约 10 步即 collapse，完全放弃高效格式 |
| Ada-GRPO → 格式分布 | Easy 多用 DA/Short CoT，Hard 多用 Long CoT | 自适应选择，简单题 token 节省 >70% |
| 去掉 Decay | 测试准确率大幅波动 | 持续过度奖励稀有格式导致训练不稳 |
| 去掉 Direct Answer (CSQA) | +0.1% acc, +29.4% token | DA 对简单任务效率至关重要 |
| 去掉 Long CoT (AIME'25) | -8.4% acc | 复杂任务必须用 Long CoT |
| 加入 function-calling 格式 | +0.4% avg acc | 更细粒度格式有帮助，但增加工程复杂度 |

### 关键发现
- Ada-GRPO 的 2× 训练加速来自 rollout 阶段的 response 长度减半，因为大量简单任务不再生成 Long CoT
- Backbone 选择影响有限：Base 和 Instruct 模型表现接近；R1-Distill 在难题上更强但简单题反而 overthinking
- 与 L1、ThinkPrune 等长度惩罚方法相比，ARM 在所有任务上保持稳定性能，不依赖人工 token 预算估计

## 亮点与洞察
- **格式选择 > 长度控制**：将推理效率问题从"控制推理链长度"重新定义为"选择推理格式"，是一个更自然更robust的范式。长度惩罚方法需要人工估计 token 预算且估计不准时性能崩溃，ARM 的自适应选择则完全自主。
- **Ada-GRPO 的简洁设计**：只用一个反比例缩放因子 $G/F(o_i)$ + cosine decay 就解决了 format collapse 问题，无需改 GRPO 的核心架构。这个思路可以迁移到任何需要防止多目标优化中某个目标主导的场景。
- **训练效率副产品**：Ada-GRPO 的 2× 训练加速是一个意外但有价值的副产品——因为模型更多使用短格式推理，rollout 生成速度大幅提升。

## 局限性 / 可改进方向
- 依赖预定义的四种推理格式，无法自动发现新的推理策略。未来可探索模型自主创造推理格式。
- 训练数据不包含 AIME 等竞赛级难题，导致在 hard task 上提升空间受限。
- 在 LLaMA backbone 上 token 节省不如 Qwen 显著（15.7% vs 55.2%），可能与 LLaMA 的重复生成倾向有关。
- Consensus-Guided Mode 需要运行三个高效格式 + 可能的 Long CoT，总 token 开销反而比纯 Long CoT 更高。

## 相关工作与启发
- **vs L1/ThinkPrune（长度惩罚）**：它们通过 RL 训练模型缩短推理链长度，但需要人工指定 token 预算。ARM 不控制长度而是选择格式，更加自主且鲁棒。在 CSQA 上，L1 用 512 token 时性能仅 45.8%，ARM 用 136 token 达到 86.1%。
- **vs DeepSeek-R1-Distill**：R1-Distill 是知识蒸馏路线，获得更强推理能力但也继承了 overthinking 问题。ARM-7B 与 DS-R1-Distill-7B 准确率持平但仅用 27.8% 的 token。
- **vs Qwen3 Thinking Mode**：Qwen3 支持 thinking/non-thinking 模式切换，但每次切换需要人工指定。ARM 的自适应选择更加自动化。

## 评分
- 新颖性: ⭐⭐⭐⭐ 将推理效率从"长度控制"重新定义为"格式选择"是很好的 insight，Ada-GRPO 的设计简洁有效，但四种格式是手工预定义的
- 实验充分度: ⭐⭐⭐⭐⭐ 3/7/14B 三个尺度，7个评测集，多种 backbone，多种推理模式，消融实验非常全面
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，Figure 2 的格式分布可视化非常直观
- 价值: ⭐⭐⭐⭐ 实用价值高，Ada-GRPO 的 2× 训练加速和推理节省对部署意义大，但受限于预定义格式
