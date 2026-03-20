# Nudging the Boundaries of LLM Reasoning

**会议**: ICLR2026  
**arXiv**: [2509.25666](https://arxiv.org/abs/2509.25666)  
**代码**: [GitHub](https://github.com/SalesforceAIResearch/NuRL)  
**领域**: llm_reasoning  
**关键词**: 强化学习推理, GRPO改进, 自生成hint, 上界突破, 近侧发展区  

## 一句话总结
指出GRPO无法从"不可解"问题(0% pass rate)学习的根本局限，提出NuRL方法在训练时对难题注入自生成的抽象hint(不泄露答案)使其变为可学习样本，在6个benchmark和3个模型上一致超越GRPO且能提升模型能力上界(pass@k)。

## 背景与动机
1. 在线RL(GRPO)的核心限制：对于模型完全无法解出的难题(pass rate=0%)，advantage=0无梯度，无法学习
2. RL后训练主要做"分布锐化"——提高已知解法的生成概率，而非发现新推理能力
3. pass@k(大k)在RL训练后常不变，说明模型的能力上界未被突破
4. 这些难题虽"不可解"但包含丰富学习信号——正对应Vygotsky的"近侧发展区"概念
5. 难题比简单题提供更有价值的训练信号(暴露模型弱点)
6. 需要一种不依赖外部强模型的方式来"nudge"模型突破能力边界

## 方法详解
**NuRL (Nudging LLM with Reinforcement Learning)**：
1. **离线Hint收集**：给定(问题q, 正确答案a)，先让模型生成解释为什么答案正确的CoT，再从CoT抽象出高层hint(核心知识线索，不包含具体答案或步骤)
2. **在线Rollout增强**：GRPO训练中，对每个问题生成$\mathcal{G}$个rollout；若全部失败(pass rate=0%)，将hint拼接到问题末尾，重新生成$\mathcal{G}-1$个带hint的rollout + 1个不带hint的rollout
3. **推理时不用hint**：训练时的hint帮助模型内化推理模式，推理时只用原始问题

**Hint类型探索**：
- 抽象线索(最佳) > 部分步骤 > 解释 > 直接答案(最差)
- 关键发现：暴露越多答案信息，性能越差——与人类学习规律一致

**两阶段训练**：Stage 1: 标准GRPO训练至收敛；Stage 2: NuRL继续训练(对Stage 1的unsolvable问题注入hint)

## 实验关键数据
| 模型 | 方法 | MATH500 | AIME | GPQA | MMLU-Pro | 平均 |
|------|------|---------|------|------|----------|------|
| Llama-3B | GRPO | 56.92 | 8.33 | 27.98 | 34.78 | 35.87 |
| Llama-3B | **NuRL(Self)** | **58.04** | **9.17** | **28.28** | **36.18** | **37.49** |
| OctoThinker-3B | GRPO | 68.81 | 8.33 | 23.26 | 44.25 | 42.63 |
| OctoThinker-3B | **NuRL(Self)** | **70.13** | **9.66** | **27.15** | **45.54** | **44.38** |
| Qwen3-4B | GRPO | 96.52 | 60.83 | 62.50 | 72.65 | 79.31 |
| Qwen3-4B | **NuRL(Self)** | **96.46** | **63.54** | **62.88** | **72.83** | **80.10** |

- NuRL(教师hint)进一步提升至+3.44%(Llama)
- NuRL提升pass@1024而GRPO无法——证明能力上界被突破
- 与Self-Consistency互补：NuRL+SC提升9.4% vs GRPO+SC 7.8%

## 亮点
- 清晰指出GRPO无法学习不可解问题的根本限制——insight深刻
- Vygotsky近侧发展区类比精准且有启发性
- 自生成hint无需外部模型——避免分布偏移且自给自足
- "越抽象的hint越好"的发现反直觉但有力
- 两阶段策略(GRPO收敛→NuRL)简洁实用

## 局限性 / 可改进方向
- 改进幅度相对温和(+1-2%平均)，在强模型(Qwen3-4B)上提升有限
- 自生成hint质量受限于模型本身能力——极难问题可能生成不了有用hint
- 仅用二值判断(全失败/部分成功)决定是否注入hint，缺乏更细粒度的策略
- 离线hint收集需要gold answer，限制了在无答案场景的适用性
- 未探索hint的质量评估和动态更新机制

## 与相关工作的对比
- vs GRPO/DAPO/Dr.GRPO: 这些方法改进advantage估计/KL/采样，NuRL正交地解决"不可解样本"问题
- vs STaR(Zelikman等2022): STaR用answer-conditioned reasoning，NuRL进一步抽象为不泄露答案的hint
- vs SFT+RL混合方法: NuRL不需要SFT阶段扩展知识范围，纯RL框架
- vs TBA(Bartoldson等2025): TBA用多搜索节点生成多样轨迹，NuRL用hint降低问题难度

## 评分
- 新颖性: ⭐⭐⭐⭐ (GRPO上界限制的insight + 自生成hint方案)
- 实验充分度: ⭐⭐⭐⭐ (3模型6benchmark+多hint类型消融+pass@k分析)
- 写作质量: ⭐⭐⭐⭐⭐ (ZPD类比优美，动机→方法→实验逻辑流畅)
- 价值: ⭐⭐⭐⭐ (解决RL推理训练的实际瓶颈，方法简洁可落地)
