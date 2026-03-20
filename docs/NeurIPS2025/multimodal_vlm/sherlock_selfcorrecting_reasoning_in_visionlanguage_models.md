# Sherlock: Self-Correcting Reasoning in Vision-Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2505.22651](https://arxiv.org/abs/2505.22651)  
**代码**: [https://dripnowhy.github.io/Sherlock/](https://dripnowhy.github.io/Sherlock/)  
**领域**: 多模态VLM / 自我纠正 / 视觉推理  
**关键词**: self-correction, preference learning, trajectory-level, VLM reasoning, self-improvement  

## 一句话总结
首个系统研究VLM推理自纠正能力的框架：发现现有推理VLM几乎不能自纠正（<10%出现aha moment），提出Sherlock三阶段训练框架（SFT冷启动→离线轨迹级偏好学习→在线自我迭代）仅用20K标注数据超越使用100K-260K数据的LLaVA-CoT/Mulberry/LlamaV-o1。

## 背景与动机
推理VLM（如LLaVA-CoT、VL-Rethinker）虽然能做长链推理，但存在三个关键问题：(1) 对推理错误极其敏感——一步错则链式传播导致最终错误；(2) 需要海量标注数据（100K-260K）；(3) 泛化性差，难以超越训练领域。作者的关键洞察：**自纠正**可以同时解决这三个问题——部分正确的推理通过纠正比从头生成更容易，纠正前后的response自然形成偏好对，减少对外部标注的依赖。

## 核心问题
现有推理VLM能否自纠正？如果不能，如何教会它们自纠正？自纠正能力能否反过来提升直接推理性能？

## 方法详解

### 整体框架
三阶段训练：
1. **SFT Cold-Start**：用10K标注数据同时训练推理和纠正能力
2. **Offline Preference Training**：用轨迹级自纠正目标做偏好学习
3. **Online Iterative Self-Improvement**：无需外部标注，用自生成偏好数据迭代提升

### 关键设计

1. **深入分析现有VLM的自纠正能力（Section 3）**：
   - Step-wise：在LLaVA-CoT/VL-Rethinker上修改一个推理步后让模型续写，<10%出现aha moment（反思信号），且即使出现也仅约50%能纠正到正确答案
   - Response-wise：无论是用self-correction prompt还是外部critic（Critic-V、Qwen2.5-VL），都无法有效改善推理，甚至可能下降
   - 结论：当前推理VLM本质上不具备自纠正能力

2. **Trajectory-level Self-Correction Objective**：不是修改整个response，而是只修改从错误步开始的suffix。给定preference pair (Y_w, Y_l)，在随机truncation点i处分割，只对≥i的suffix做偏好学习。这避免了对已正确prefix的无意义更新，提供更清晰的学习信号。

3. **Visual Perturbation构造偏好数据**：用视觉噪声扰动生成低质量response——在随机truncation点后给图像加噪声再让模型续写，产生可控质量差距的rejected response。无需外部验证器。

4. **Dynamic β for DPO**：根据truncation位置i和噪声强度ε自适应调整β值：β(i,n,ε) = 1/4 * (0.5 + (i/n)^0.5 + ε/2)。质量差距大（早期truncation+强噪声）时β大→保守更新；差距小时β小→强学习。

5. **Online Self-Improvement（Stage 3）**：利用自纠正能力自举——对每个输入做3轮self-correction产生Y2/Y3/Y4，如果最终答案一致则Y4作为preferred。用Y1+视觉扰动构造rejected。每轮仅需5K无标注问题。

### 训练策略
- 基座：Llama3.2-Vision-11B-Instruct
- 仅需20K标注数据（从LLaVA-CoT 100K中随机采样2个10K子集DA、DB）
- 在线阶段每轮5K无标注数据，做2轮迭代
- 训练时间128 GPU·h，少于LLaVA-CoT(160h)和LlamaV-o1(288h)

## 实验关键数据

| 方法 | 标注数据 | 直接推理Avg | 自纠正后Avg |
|------|----------|-------------|------------|
| LLaVA-CoT | 100K | 63.2 | 63.0 (↓0.2) |
| Mulberry | 260K | 63.9 | 63.8 (↓0.1) |
| LlamaV-o1 | 175K | 63.4 | 48.2 (↓15.2!) |
| **Sherlock Iter2** | **20K** | **64.1** | **65.4 (+1.3)** |

关键对比：Sherlock是唯一一个自纠正后性能提升的模型。LlamaV-o1纠正后崩溃是因为其多轮推理格式与correction prompt冲突。

推理时间缩放：Sherlock + MM-Verify验证器，MathVista上52.0→55.9（+3.9），GPU时间仅8.7h vs Majority Vote的40.2h。

### 消融实验要点
- 自纠正和推理不是正交能力——学习自纠正也提升直接推理（Finding 1）
- 轨迹级目标 >> 全response级目标——在线迭代时全response纠正反而退化（Finding 2）
- Dynamic β稳定训练并持续提升两个能力（Finding 3）
- 20K数据下Sherlock SFT已比相同数据量的LLaVA-CoT好+0.8（因为self-correction是free lunch）
- 每轮self-correction都有提升：64.1→64.5→65.2→65.4（3轮）

## 亮点 / 我学到了什么
- **"现有推理VLM不能自纠正"这个实证发现**——quantify了10%以下的aha moment率，非常有说服力
- **轨迹级纠正**而非全response纠正——只改错误suffix保留正确prefix，是一个精细且合理的设计
- **视觉扰动构造偏好数据**——巧妙利用图像噪声创造可控质量差距，不需要任何外部验证器
- 自纠正 → 自我迭代的闭环：学会纠正后就能自己生成偏好数据继续进步，非常优雅
- 20K数据超过260K——极高的数据效率，核心在于充分利用每个样本

## 局限性 / 可改进方向
- 仅在Llama3.2V-11B上验证，更大模型（如72B）和其他VLM家族未测试
- 自纠正提升幅度有限（+1.3%），对于已经接近正确的case纠正空间小
- 视觉扰动只用高斯噪声，语义级扰动（如occlude关键区域）可能更有效
- 在线阶段用self-consistency做过滤，但一致性≠正确性
- 与RL方法（如GRPO）的结合未探索——Sherlock是纯SFT+DPO路线

## 与相关工作的对比
- vs **LLaVA-CoT**：LLaVA-CoT只做SFT无自纠正，自纠正后性能轻微下降；Sherlock用1/5数据超越且自纠正有效
- vs **Mulberry**：Mulberry用MCTS生成260K CoT数据+step-wise反思SFT，但自纠正仍失败；Sherlock的轨迹级偏好学习更effective
- vs **NoisyRollout (2504.13055)**：NoisyRollout用视觉扰动增强RL exploration，Sherlock用视觉扰动构造偏好数据——同一策略的不同应用场景
- vs **VL-Rethinker**：RL-based但自纠正也失败（<10% aha），说明RL alone不够

## 与我的研究方向的关联
- 与ideas/multimodal_vlm/20260316_causal_process_reward_vision.md相关——Sherlock证明过程级（轨迹级）信号比结果级更有效
- 与NoisyRollout互补：NoisyRollout在RL阶段加视觉扰动做exploration，Sherlock在偏好学习阶段用视觉扰动做数据构造，可组合
- "自纠正即free lunch"的发现很有启发——任何推理VLM都应该配备self-correction training

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个系统研究VLM推理自纠正的工作，轨迹级纠正+视觉扰动偏好数据是novel的组合
- 实验充分度: ⭐⭐⭐⭐⭐ 8个benchmark、详细消融、与MM-Verify的结合、case study极其细致
- 写作质量: ⭐⭐⭐⭐⭐ Section 3的分析→Section 4的方法设计逻辑非常流畅，4个Takeaway清晰
- 对我的价值: ⭐⭐⭐⭐⭐ VLM self-correction是关键能力，20K数据的高效训练范式对资源受限场景极有价值
