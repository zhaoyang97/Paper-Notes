---
title: "AutoTool: Automatic Scaling of Tool-Use Capabilities in RL via Decoupled Entropy Constraints"
authors: "Yirong Zeng, Xiao Ding, Yufei Liu, Yuxian Wang, Qunyao Du, Yutai Hou, Wu Ning, Haonan Song, Duyu Tang, Dandan Tu, Bing Qin, Ting Liu"
affiliations: "Harbin Institute of Technology, Peking University, Huawei Technologies"
venue: "ICLR 2026"
arxiv: "2603.13348"
code: ""
tags: ["tool use", "reinforcement learning", "test-time scaling", "entropy constraint", "GRPO", "agentic LLM"]
rating:
  novelty: 4
  experiments: 4
  writing: 4
  value: 4
---

# AutoTool: Automatic Scaling of Tool-Use Capabilities in RL via Decoupled Entropy Constraints

## 一句话总结

提出解耦自适应熵约束 (Decoupled Adaptive Entropy Constraints) 的强化学习策略，使 LLM 在工具调用任务中根据问题难度自动切换长/短推理模式，在提升 9.8% 准确率的同时减少约 81% 的推理 token 开销。

## 研究动机

将大语言模型与外部工具集成是迈向 AGI 的关键路径。当前通过强化学习 (RL) 实现测试时缩放 (Test-time Scaling, TTS) 在数学推理中已取得显著成功——RL 训练能让模型的响应长度随准确率同步增长。然而，作者发现在工具调用任务中存在两个核心挑战：

1. **推理模式坍塌 (Reasoning Collapse)**：直接用 GRPO 等 RL 算法训练工具调用模型时，模型的响应长度不升反降——随着训练步数增加，准确率提升但响应长度急剧缩短。模型"懒"于展开长链推理，导致在复杂多轮工具调用场景中性能下降。
2. **过度思考 (Overthinking)**：通过蒸馏获得的长推理模型会对所有问题生成冗长推理轨迹，即使简单问题也如此，造成约 10 倍的 token 开销浪费。

作者进一步分析发现，推理坍塌与信息熵高度正相关——训练中策略模型的熵迅速下降，模型失去探索能力，默认使用短推理。而直接加 length penalty 无法缓解低熵问题，静态熵约束又对系数 $\beta$ 高度敏感。

## 方法详解

### 整体框架

AutoTool 采用两阶段训练流程：**热身 SFT → 解耦自适应熵约束 RL**。

### 阶段一：数据准备与热身 SFT

- **PubTool 数据集**：整合 ToolACE、xLAM、Hermes Function-Calling 三个公开数据源，经下采样和质量筛选得到 SFT 数据 8.2k 条、RL 数据 7k 条
- **混合长短推理数据**：对训练数据分别用 no-thinking 模型 (Qwen2.5-7B-Instruct) 和 thinking 模型 (Qwen3-32B) 做 pass@8 推理。若 no-thinking 模型答对，采用短推理作为标签；否则采用 thinking 模型的长推理标签
- **RL 数据质量优化**：移除一半过于简单/困难样本平衡分布，并基于多轮 GRPO 训练中 reward 方差筛选与模型学习轨迹对齐的高质量样本
- 热身 SFT 让模型初步感知数据难度，学会区分需要长推理和短推理的问题

### 阶段二：解耦自适应熵约束 RL

核心是在 GRPO 的策略损失中加入**解耦的熵正则项**：

$$\beta_i = \beta_s \cdot m_i \cdot \mathbb{I}\{H_i \leq H_s\} + \beta_l \cdot (1-m_i) \cdot \mathbb{I}\{H_i \leq H_l\}$$

其中 $m_i \in \{0,1\}$ 标识当前轨迹是短推理 ($m_i=1$) 还是长推理 ($m_i=0$)：

- **短推理 $\beta_s$**：固定系数，防止过度探索，保持简洁响应
- **长推理 $\beta_l$**：自适应系数，通过额外损失动态调节

自适应熵系数损失：

$$\mathcal{L}_{\beta}^l = \frac{1}{\sum_j(1-m_j)} \sum_{i=1}^{N} (1-m_i) \cdot \beta_l \cdot (H_i - H_l)$$

当 $H_i < H_l$ 时 $\beta_l$ 增大以鼓励探索，当 $H_i > H_l$ 时 $\beta_l$ 减小以抑制过度随机性。

### 自动思考奖励模块

设计非对称奖励机制激励效率与准确的平衡：

| 情形 | 奖励 |
|------|------|
| 正确 + no-think | +1.0 |
| 正确 + think | +0.5 |
| 错误 + think | -0.5 |
| 错误 + no-think | -1.0 |

简单问题答对用短推理获得更高奖励；复杂问题答错则鼓励切换到长推理。推理时可通过前缀 token 可控地切换推理模式。

## 实验结果

### 基准与设置

- 基座模型：Qwen2.5-7B-Instruct
- 评测基准：BFCL（Non-Live / Live / Multi-Turn）、API-Bank（L-1 / L-2）、ACEBench

### 主要结果（BFCL）

| 模型 | Non-Live | Live | Multi-Turn | Overall |
|------|----------|------|------------|---------|
| Qwen2.5-7B-Instruct | 86.46 | 67.44 | 7.62 | 53.69 |
| PubTool-SFT | 88.98 | 77.28 | 9.68 | 58.17 |
| PubTool-Distilled | 87.73 | 78.64 | 15.65 | 60.30 |
| Qwen3-8B | 88.81 | 78.54 | 33.00 | 66.34 |
| **AutoTool-7B (auto)** | **89.76** | **80.22** | **38.18** | **70.12** |

- 相比 PubTool-SFT 提升 +11.95%，相比 Base 提升 +16.43%
- Multi-Turn 复杂场景提升最为显著（+28.5% vs SFT）
- 性能与 GPT-4o（70.42）、o3（70.32）等前沿模型相当

### 推理效率分析

- AutoTool 平均仅需约 183 tokens，蒸馏模型约 966 tokens，**token 成本降低 81%**
- Multi-Turn 场景 thinking rate 45%，Non-Live 简单场景 thinking rate 0%——模型真正学会按难度自适应
- 强制 no-think 模式下 ACU（Accuracy per Computation Unit）达到最优 0.97

### 消融实验

| 变体 | Overall 变化 |
|------|-------------|
| 完整模型 | 70.12 |
| w/o data refine | -6.43 |
| w/o decouple | -5.89 |
| w/o adapt coeff | -2.34 |

数据质量筛选影响最大，解耦设计次之，自适应系数对 Multi-Turn 稳定性贡献显著（移除后 Multi-Turn 下降 10.53%）。

## 优点

1. **问题分析深入**：系统性地分析了推理坍塌现象与熵的关系，揭示了数据分布并非根因，信息熵才是关键
2. **方法设计合理**：解耦长短推理的熵约束避免目标干扰，自适应系数免去敏感的手动调参
3. **实用价值高**：7B 模型达到 GPT-4o 级别工具调用性能，同时大幅降低推理成本
4. **可控推理模式**：推理时可通过前缀灵活切换 think / no-think / auto 模式
5. **奖励设计精巧**：非对称奖励自然引导模型对简单问题用短推理、复杂问题用长推理

## 局限性

1. 仅在 7B 规模验证，更大模型上推理坍塌现象是否同样存在、方法是否仍然有效未探讨
2. PubTool 数据集规模有限（7k RL 数据），更大规模训练的 scalability 未验证
3. 热身 SFT 依赖 Qwen3-32B 的蒸馏数据，需要强大的 teacher 模型
4. 长短推理的划分依赖是否包含 think 标记，对更细粒度的推理深度控制有限
5. 评测主要面向函数调用类工具，对代码执行、网页浏览等更广泛工具场景的泛化能力未测试
