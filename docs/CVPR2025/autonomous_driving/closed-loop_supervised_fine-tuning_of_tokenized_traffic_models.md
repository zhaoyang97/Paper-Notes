---
title: "Closed-Loop Supervised Fine-Tuning of Tokenized Traffic Models"
authors: "Zhejun Zhang, Peter Karkus, Xavi Puig, Maximilian Igl, Yue Wang, Marco Pavone"
affiliations: "NVIDIA, Stanford University"
venue: "CVPR 2025"
date: 2024-12-06
tags: ["autonomous driving", "traffic simulation", "closed-loop", "tokenization", "fine-tuning"]
arxiv: "2412.05334"
code: "https://github.com/NVlabs/catk"
---

# Closed-Loop Supervised Fine-Tuning of Tokenized Traffic Models

## 研究背景与动机

自动驾驶中的**交通仿真**（traffic simulation）是评估规划算法安全性的核心工具。高质量的交通仿真需要生成真实、多样且符合物理规律的交通参与者行为轨迹。

近年来，基于 Transformer 的**token化交通模型**（如 SMART）取得了显著进展。这类方法将连续的轨迹离散化为 token 序列，利用自回归语言模型的架构进行轨迹预测和生成。然而，这些模型面临一个根本性问题——**分布偏移**（distribution shift）。

### 分布偏移问题

在标准的**行为克隆**（Behavior Cloning, BC）训练中，模型在专家轨迹上以 teacher forcing 方式训练：每一步的输入都是真实的历史状态。但在**闭环推理**（closed-loop inference）时：

1. 模型的预测误差会累积到后续步骤的输入中
2. 随着时间推移，模型的输入分布逐渐偏离训练分布
3. 模型对偏离分布的输入缺乏应对能力，导致行为退化

| 训练方式 | 输入来源 | 分布偏移 | 计算开销 |
|---------|---------|---------|---------|
| Behavior Cloning (开环) | 真实历史 | 严重 | 低 |
| DAgger | 混合 | 中等 | 高（需要交互） |
| RL fine-tuning | 模型预测 | 低 | 极高 |
| **CLSFT (本文)** | **CAT-K rollouts** | **低** | **中等** |

现有的缓解方案（DAgger、RL微调）要么需要在线交互的高计算成本，要么需要设计复杂的奖励函数。

本文提出 **Closed-Loop Supervised Fine-Tuning (CLSFT)**，通过一种巧妙的 rollout 策略（CAT-K）将闭环信息引入监督学习框架，以极低的额外成本显著缓解分布偏移。

## 方法详解

### Token化交通模型基础

SMART 等模型的工作流程：
1. 将每个交通参与者的轨迹点 $(x, y, 	heta)$ 量化为离散 token
2. 使用 VQ-VAE 学习 token codebook
3. 以 GPT 风格的自回归模型预测下一个 token

标准训练使用交叉熵损失：

$$\mathcal{L}_{BC} = -\sum_t \log p_	heta(z_t^* | z_{<t}^*)$$

其中 $z_t^*$ 是专家轨迹对应的真实 token。

### CAT-K Rollouts

CAT-K（Closest-Among-Top-K）是 CLSFT 的核心创新。在生成训练数据时：

#### 步骤1：Top-K 采样

在每个时间步，模型预测 token 的概率分布 $p_	heta(z_t | z_{<t})$，取概率最高的 $K$ 个候选 token：

$$	ext{Top-K}(p_	heta) = \{z_t^{(1)}, z_t^{(2)}, ..., z_t^{(K)}\}$$

#### 步骤2：选择最接近专家的 token

在 $K$ 个候选中，选择与专家 token 最接近的那个：

$$z_t^{CAT} = rg\min_{z \in 	ext{Top-K}(p_	heta)} d(z, z_t^*)$$

其中 $d(\cdot)$ 是 token 空间中的距离度量（通常是对应轨迹点的欧氏距离）。

#### 步骤3：闭环展开

将 $z_t^{CAT}$ 作为下一步的输入（而非真实的 $z_t^*$），继续生成后续 token。

| 方法 | 第$t$步输入 | 第$t$步目标 |
|------|-----------|-----------|
| BC (开环) | $z_{<t}^*$ (真实) | $z_t^*$ |
| 纯闭环 | $z_{<t}^{model}$ (模型) | $z_t^*$ |
| **CAT-K** | $z_{<t}^{CAT}$ (近似专家) | $z_t^*$ |

### CAT-K 的设计直觉

CAT-K 的关键优势：

1. **受控偏离**：CAT-K 产生的轨迹与专家轨迹接近但不完全相同，模型在训练中接触到轻微偏离的输入分布
2. **稳定性**：由于每步选择最接近专家的 token，偏离不会失控地累积
3. **多样性**：$K$ 的选择控制了偏离程度，$K=1$ 退化为贪心解码，$K 	o \infty$ 接近随机采样

### CLSFT 训练流程

1. 用预训练的 SMART 模型在训练集上执行 CAT-K rollouts
2. 收集 (CAT-K轨迹, 专家标签) 对
3. 以标准交叉熵损失进行微调

$$\mathcal{L}_{CLSFT} = -\sum_t \log p_	heta(z_t^* | z_{<t}^{CAT})$$

## 实验结果

### WOSAC Leaderboard

| 方法 | 参数量 | RMM↑ | Kinematic↑ | Interactive↑ | Map↑ |
|------|--------|------|-----------|-------------|------|
| SMART-7M | 7M | 0.7302 | 0.821 | 0.683 | 0.712 |
| SMART-102M | 102M | 0.7614 | 0.849 | 0.715 | 0.738 |
| MotionLM | 45M | 0.7489 | 0.837 | 0.701 | 0.729 |
| **SMART-7M + CLSFT** | **7M** | **0.7702** | **0.856** | **0.728** | **0.749** |

SMART-7M + CLSFT 以仅7M参数**超越了SMART-102M**（RMM 0.7702 vs 0.7614），展示了 CLSFT 的巨大潜力。

### K值消融

| K值 | RMM↑ | 说明 |
|-----|------|------|
| K=1 (贪心) | 0.7412 | 偏离太小，接近开环 |
| K=5 | 0.7589 | 适度偏离 |
| **K=10** | **0.7702** | 最佳平衡点 |
| K=50 | 0.7645 | 偏离过大，不稳定 |
| K=∞ (随机) | 0.7301 | 退化，偏离失控 |

K=10 提供了最佳的偏离-稳定性平衡。

### 闭环仿真质量

| 指标 | SMART (BC) | SMART + DAgger | SMART + CLSFT |
|------|-----------|---------------|--------------|
| Collision Rate↓ | 5.23% | 3.87% | **3.21%** |
| Off-road Rate↓ | 2.14% | 1.62% | **1.38%** |
| Progress↑ | 0.891 | 0.912 | **0.927** |
| Comfort↑ | 0.834 | 0.856 | **0.871** |

CLSFT 在所有闭环仿真指标上均优于 DAgger，且不需要在线交互。

## 总结与展望

CAT-K 通过在 Top-K 候选中选择最接近专家的 token 来构建近似闭环的 rollouts，为 token 化交通模型提供了一种高效且有效的闭环监督微调方法。SMART-7M + CLSFT 以7M参数在 WOSAC 排行榜上超越了102M参数的 SMART-102M（RMM 0.7702 vs 0.7614），证明了在正确的训练范式下，小模型可以超越大模型。CLSFT 的核心思想——通过受控偏离引入闭环信息——可以推广到其他序列预测任务中。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] DrivingSphere: Building a High-fidelity 4D World for Closed-loop Simulation](drivingsphere_building_a_high-fidelity_4d_world_for_closed-loop_simulation.md)
- [\[ECCV 2024\] Safe-Sim: Safety-Critical Closed-Loop Traffic Simulation with Diffusion-Controllable Adversaries](../../ECCV2024/autonomous_driving/safe-sim_safety-critical_closed-loop_traffic_simulation_with_diffusion-cont.md)
- [\[CVPR 2026\] RLFTSim: Realistic and Controllable Multi-Agent Traffic Simulation via Reinforcement Learning Fine-Tuning](../../CVPR2026/autonomous_driving/rlftsim_realistic_and_controllable_multi-agent_traffic_simulation_via_reinforcem.md)
- [\[NeurIPS 2025\] Model-Based Policy Adaptation for Closed-Loop End-to-End Autonomous Driving](../../NeurIPS2025/autonomous_driving/model-based_policy_adaptation_for_closed-loop_end-to-end_autonomous_driving.md)
- [\[ICLR 2026\] SMART-R1: Advancing Multi-agent Traffic Simulation via R1-Style Reinforcement Fine-Tuning](../../ICLR2026/autonomous_driving/advancing_multi-agent_traffic_simulation_via_r1-style_reinforcement_fine-tuning.md)

</div>

<!-- RELATED:END -->
