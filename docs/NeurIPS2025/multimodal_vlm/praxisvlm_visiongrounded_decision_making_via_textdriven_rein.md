# Praxis-VLM: Vision-Grounded Decision Making via Text-Driven Reinforcement Learning

**会议**: NeurIPS 2025  
**arXiv**: [2503.16965](https://arxiv.org/abs/2503.16965)  
**代码**: [https://github.com/Derekkk/Praxis-VLM](https://github.com/Derekkk/Praxis-VLM)  
**领域**: 多模态VLM / Agent决策 / 强化学习  
**关键词**: VLM decision-making, text-driven RL, GRPO, cross-modal transfer, embodied reasoning  

## 一句话总结
发现VLM的决策推理能力可以与视觉感知解耦——用文本描述替代图像时决策性能不降反升，据此提出Praxis-VLM：在纯文本场景上用GRPO训练决策推理能力，然后零样本迁移到视觉输入推理，在VIVA/PCA-Bench/EgoNormia三个决策benchmark上超越SFT基线且泛化性更强。

## 背景与动机
VLM在视觉理解上表现好，但在复杂情境决策（如"看到交通事故应该做什么"）上缺乏显式推理能力。现有方法（如R1-OneVision、OpenVLThinker）用RL增强推理，但严重依赖大规模图文配对数据——在决策场景中这类数据极度稀缺。作者的关键发现：当把视觉情境替换为文本描述时，VLM的决策性能甚至更好——说明决策推理能力的核心在语言域，可以与视觉感知解耦学习。

## 核心问题
VLM的决策推理能力能否从纯文本中学习并迁移到视觉输入场景？如何用数据高效的方式（不需要图文配对数据）增强VLM的情境决策能力？

## 方法详解

### 整体框架
三阶段流程：
1. 构造纯文本决策数据（GPT-4o合成10K样本）
2. 多阶段GRPO训练（Stage1: geometry3k数学冷启动 → Stage2: 文本决策场景RL）
3. 推理时直接用视觉输入——文本中学到的推理能力自动迁移

训练时只更新LLM参数，不碰vision encoder。推理时完整VLM架构处理图像。

### 关键设计

1. **核心发现：决策推理与视觉感知可解耦**：在VIVA和PCA-Bench上，用textual situation（GPT-4o caption或标注文本）替代原始图像，Qwen2.5-VL的决策准确率与用图像输入持平甚至更高。这说明VLM的决策瓶颈不在视觉感知，而在推理能力。

2. **Multi-Stage GRPO with Adaptive R1 Reward**：
   - Stage 1（冷启动）：用geometry3k数学数据训练格式遵从和基础逻辑推理。Reward = R_accuracy + R_format + 0.5·R_tag。模型学会<think></think><answer></answer>格式后去掉R_tag。
   - Stage 2（决策RL）：在合成文本决策数据上训练。Reward = R_accuracy + 0.8·R_format + 0.5·R_len。R_len鼓励模型生成更长、更充分的推理链（word_count/250, capped at 1.0），促进多角度分析。
   - 关键发现：可以跳过SFT冷启动直接GRPO——只要有adaptive reward策略。

3. **文本决策数据构造**：用GPT-4o批量生成（每次10个+去重），每个样本包含文本情境描述+多选题+答案。10K训练+1K验证。不需要图像，不需要人工过滤。

4. **推理维度分析**：通过GPT-4o分析Praxis-VLM的推理链，识别出4个核心决策维度：① 情境分析 ② 行动与结果评估 ③ 安全与风险管理 ④ 规则与规范遵从

### 训练策略
- 基座：Qwen2.5-VL-3B/7B-Instruct
- GRPO rollout N=5, KL系数0.01, lr=1e-6
- 训练硬件：4×A100/H100 GPU
- 推理：vLLM + greedy decoding

## 实验关键数据

| 模型 | VIVA | PCA-Bench | EgoNormia (OOD) |
|------|------|-----------|-----------------|
| Qwen2.5-VL-7B | 80.97 | 46.37 | 46.19 |
| + SFT | 81.13 | 45.74 | 34.83 |
| + Reason SFT | 78.79 | 53.00 | 34.08 |
| **Praxis-VLM-7B** | **83.87** | **58.99** | **49.57** |
| Praxis-VLM-7B (w/o cold start) | 82.66 | 55.21 | 47.10 |

关键对比：SFT在OOD的EgoNormia上严重退化（34.83），Praxis-VLM反而提升（49.57 > 46.19），说明RL学到的推理更可迁移。

Majority Vote (8 samples)：Praxis-VLM-7B在VIVA 84.36, PCA-Bench 61.83, EgoNormia 55.08——全面超越。

### 消融实验要点
- Math cold start提升OOD泛化（EgoNormia: 47.10→49.57），域内影响小
- 更长推理链对应更难样本，但在同等难度下Praxis-VLM始终优于baseline
- 超长推理可能"overthinking"——最长20%的样本准确率下降
- Pass@1 (8 samples) 非常高：VIVA 89.27%, PCA-Bench 77.92%——说明正确推理路径充分存在

## 亮点 / 我学到了什么
- **决策推理与视觉感知可解耦**——这个发现很有认知科学意义，呼应"心理模型理论"（人类通过语言构建内部表示来推理）
- **纯文本训练→视觉推理迁移**：训练时不用任何图文配对数据，推理时直接处理视觉输入，极其数据高效
- 无需SFT冷启动——直接用adaptive reward的GRPO就能工作，简化了pipeline
- R_len reward促进更充分的推理分析——与NoisyRollout中"更长不一定更好"的观点形成对比，说明在决策任务中更长推理是有益的
- 错误分析很有价值：情境误解、安全优先级错误、规范对齐缺失是三大失败模式

## 局限性 / 可改进方向
- 仅在3B/7B模型上验证，更大模型效果未知
- 文本决策数据由GPT-4o合成，可能存在域偏差
- EgoNormia用视频帧拼接为单图——对视频理解能力的评估不够原生
- 推理链长度限制（1024 tokens）导致部分回答被截断
- 未与其他VLM决策方法（如VLA模型）直接比较

## 与相关工作的对比
- vs **NoisyRollout**：NoisyRollout用视觉扰动增强exploration，Praxis-VLM完全绕开视觉域用纯文本训练——两种数据效率策略形成互补
- vs **R1-OneVision/Vision-R1等**：这些方法用图文配对数据做RL，Praxis-VLM证明对决策任务可以纯文本训练
- vs **Sherlock**：Sherlock关注推理中的自纠正，Praxis-VLM关注决策场景的推理泛化——可以组合

## 与我的研究方向的关联
- "文本训练→视觉迁移"范式对VLM训练效率有深远启示——可推广到更多视觉推理任务
- 与CoRL (2505.17534)的"跨任务协同"互补：CoRL让generation和understanding协同，Praxis-VLM让text和vision协同
- 4个决策维度（情境分析、结果评估、安全考量、规范遵从）可用于设计更好的reward

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "纯文本训练VLM决策推理"是独特且有说服力的范式，preliminary analysis设计精妙
- 实验充分度: ⭐⭐⭐⭐ 3个benchmark+多种baseline+diverse sampling+错误分析，但模型规模和任务覆盖可更广
- 写作质量: ⭐⭐⭐⭐⭐ 从preliminary finding到方法设计的叙事逻辑极佳，"Language is the dress of thought"引用贴合
- 对我的价值: ⭐⭐⭐⭐⭐ VLM agent决策+数据高效RL是核心方向，文本迁移范式可直接复用
