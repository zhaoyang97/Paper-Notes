# MindPower: Enabling Theory-of-Mind Reasoning in VLM-based Embodied Agents

**会议**: CVPR 2026  
**arXiv**: [2511.23055](https://arxiv.org/abs/2511.23055)  
**代码**: [https://zhangdaxia22.github.io/MindPower/](https://zhangdaxia22.github.io/MindPower/) (Benchmark)  
**领域**: 具身智能 / 心智理论 / VLM Agent  
**关键词**: Theory of Mind, BDI推理, 具身Agent, Mind-Reward, GRPO  

## 一句话总结
MindPower 提出了以机器人为中心的心智理论（ToM）推理框架，将感知→信念→欲望→意图→决策→行动组织为六层推理层级，并用 Mind-Reward（基于 GRPO）优化推理一致性，在决策和动作生成上分别超过 GPT-4o 12.77% 和 12.49%。

## 背景与动机
现有 VLM-based 具身 agent 只能执行显式指令，缺乏推断人类信念、欲望和意图的能力。现有的 ToM benchmark 只关注推断视频中人物的心理状态（角色中心），不涉及 agent 自身的视角推理，也不要求生成决策和动作。这意味着 agent 无法做到"理解人类在想什么，然后主动帮忙"。

## 核心问题
如何让具身 agent 从自身视角出发，推断人类的心理状态（信念、欲望、意图），并基于此推理做出主动的决策和行动？

## 方法详解

### 整体框架
MindPower 包含三部分：(1) MindPower Benchmark——590 个家庭场景（VirtualHome + ThreeDWorld），含两个任务（错误信念纠正、隐式目标推断）；(2) MindPower Reasoning Hierarchy——三级六层推理层级结构；(3) Mind-Reward——基于 GRPO 的强化学习优化，用 SFT+RL 两阶段训练。基础模型为 Qwen2.5-VL-7B。

### 关键设计
1. **MindPower Reasoning Hierarchy（六层推理结构）**:
   - Level-1 **感知** `<Perception>`：观察环境和人类行为
   - Level-2 **心智推理**: `<Belief>`（推断自己和人类的信念，含二阶信念——"我认为Alice认为苹果在桌上"）→ `<Desire>`（确定辅助目标）→ `<Intention>`（形成具体行动意图）
   - Level-3 **决策与行动**: `<Decision>`（选择计划）→ `<Action>`（输出原子操作序列如 `walk(fridge), open(fridge), pick(apple)`）

2. **Robot-Centric 视角（区别于现有 Role-Centric benchmark）**: 现有 ToM benchmark 只推断视频中人物的心理状态。MindPower 要求 agent 同时推断自己的信念和人类的信念，形成完整的推理闭环。例如："我认为 Alice 在找苹果" + "我知道苹果实际在冰箱里" → "我应该帮她从冰箱拿苹果"

3. **Mind-Reward（基于 ROUGE 的原子动作匹配奖励）**: 将每层推理输出转换为原子动作序列（由 Qwen3-Max 提取），然后计算三种对齐指标：原子准确度（ROUGE-1）、局部一致性（ROUGE-2）、全局一致性（ROUGE-L）。最终 $R_{\text{Mind}} = \alpha_1 R_{\text{atomic}} + \alpha_2 R_{\text{local}} + \alpha_3 R_{\text{global}}$，辅以格式奖励，用 GRPO 优化。

### 损失函数 / 训练策略
两阶段训练：(1) SFT 冷启动（5 epochs），建立基本推理能力；(2) GRPO 强化（400 iterations，8 个生成样本），用 Mind-Reward + Format-Reward。训练在单卡 H800 上完成。

## 实验关键数据
| 方法 | Decision (S) | Action SR | Action AC | BPC |
|------|------|------|------|------|
| GPT-4o (图像输入) | 34.35 | 1.82 | 2.91 | 8.05 |
| Gemini-2.5 Pro | 33.87 | 2.08 | 2.54 | 8.56 |
| Video-R1 | 30.33 | 1.43 | 1.72 | 6.45 |
| Qwen2.5-VL-7B (base) | 26.56 | 0.29 | 0.22 | 6.07 |
| **Ours (SFT+Mind-Reward)** | **47.12** | **11.75** | **15.40** | **8.87** |
| Human Baseline | 56.66 | 19.37 | 26.26 | 8.19 |

相对 GPT-4o: 决策 +12.77pp，动作准确率 +12.49pp。

### 消融实验要点
- **仅 SFT（无 RL）**: 已有较大提升（Action AC: 0.22→10.48），说明推理层级结构本身有效
- **仅 Mind-Reward（无 SFT）**: 效果有限（AC: 0.40），说明需要 SFT 冷启动
- **SFT+Mind-Reward**: 最优（AC: 15.40），RL 在 SFT 基础上进一步提升约 5 个点
- **MindPower Hierarchy vs 直接输出**: GPT-4o 去掉推理层级后决策准确率下降 1.24%，动作下降更多
- **MindPower Hierarchy vs 标准 CoT**: MindPower 的结构化 BDI 推理比通用 `<think>` 推理好 4.89%

## 亮点
- 将认知科学中的 BDI 框架（信念-欲望-意图）系统化地引入具身 agent，形成可解释的推理链
- Robot-Centric 视角是核心创新——agent 不仅推断他人心理状态，还显式建模自己的信念，实现二阶推理
- Mind-Reward 将推理质量分解为原子-局部-全局三个粒度的一致性评估，比黑盒 LLM 评分更可控
- 两个任务设计很有洞察力：错误信念纠正（agent 知道物体被移动了但人不知道）和隐式目标推断（从人搜索行为推断需求）

## 局限性 / 可改进方向
- 数据集仅 590 个场景，且全部来自模拟器（VirtualHome + ThreeDWorld），场景多样性受限
- 动作空间较粗（高层原子操作如 `walk(fridge)`），未涉及底层运动控制
- Mind-Reward 依赖 Qwen3-Max 提取原子动作，引入了额外的 LLM 依赖
- 开放式评估的自动指标（BERTScore、ROUGE）是否能真正反映推理质量存疑

## 与相关工作的对比
- **vs MuMA-ToM/MMToM-QA**: 这些只做角色心理状态推断（选择题），MindPower 要求 agent 从自身视角做完整的 BDI 推理 + 动作生成
- **vs Smart-Help/AToM-Bot**: 这些做人机交互辅助但缺乏显式心智推理。MindPower 明确建模了信念不一致的检测与纠正
- **vs Video-R1/VideoChat-R1**: 这些专注视频理解的 RL 训练，但不涉及 ToM 推理和具身决策

## 启发与关联
- BDI 推理层级可以作为一种"结构化 CoT"推广到其他需要推理他人意图的任务（如社交对话、协作博弈）
- Mind-Reward 的设计思路——将过程拆解为原子操作再评估一致性——对其他需要过程奖励的 RL 任务有参考价值
- 可以探索将 MindPower 的高层推理与 VLA 模型（如 MergeVLA）的底层执行结合，构建端到端具身系统

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Robot-Centric ToM + BDI 推理层级是全新视角
- 实验充分度: ⭐⭐⭐⭐ 对比了多个闭源/开源 VLM + 人类基线，但数据集规模偏小
- 写作质量: ⭐⭐⭐⭐ 概念清晰、层次分明，但附录过长
- 价值: ⭐⭐⭐⭐ 为具身 agent 赋予 ToM 能力是重要方向，但距离实际应用还有距离
