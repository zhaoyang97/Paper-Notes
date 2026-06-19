---
title: >-
  [论文解读] One Filters All: A Generalist Filter for State Estimation
description: >-
  [NeurIPS 2025][自监督学习][LLM重编程] 提出 LLM-Filter，将 LLM 重编程为通用状态估计器，通过 System-as-Prompt（SaP）机制使冻结的 LLM 在未见动力系统上实现零样本泛化，性能超越 SOTA 学习型滤波器。 领域现状：状态估计（贝叶斯滤波）是机器人、气象、交通等领域的核心…
tags:
  - "NeurIPS 2025"
  - "自监督学习"
  - "LLM重编程"
  - "状态估计"
  - "贝叶斯滤波"
  - "提示学习"
  - "泛化滤波器"
---

# One Filters All: A Generalist Filter for State Estimation

**会议**: NeurIPS 2025  
**arXiv**: [2509.20051](https://arxiv.org/abs/2509.20051)  
**代码**: 有  
**领域**: 自监督学习 / 状态估计  
**关键词**: LLM重编程, 状态估计, 贝叶斯滤波, System-as-Prompt, 泛化滤波器

## 一句话总结

提出 LLM-Filter，将 LLM 重编程为通用状态估计器，通过 System-as-Prompt（SaP）机制使冻结的 LLM 在未见动力系统上实现零样本泛化，性能超越 SOTA 学习型滤波器。

## 研究背景与动机

**领域现状**：状态估计（贝叶斯滤波）是机器人、气象、交通等领域的核心问题。传统方法（Kalman Filter、粒子滤波）依赖手工建模。

**现有痛点**：学习型滤波器虽然精度高但针对特定系统训练，切换系统需重新训练。高维非高斯系统中高斯滤波器误差大，粒子滤波计算量大。

**核心矛盾**：高精度需要针对性训练 vs 泛化需要跨系统能力。

**切入角度**：利用 LLM 的预训练知识和上下文学习能力，将状态估计重编程为 token 预测任务，通过 SaP 引导 LLM 理解不同系统。

**核心 idea**：冻结 LLM 核心层，仅训练输入嵌入和输出投影，通过 SaP 文本描述系统信息实现跨系统泛化。

## 方法详解

### 整体框架

(1) 观测嵌入：将连续观测分段嵌入为 token；(2) 上下文推理：SaP 文本 token 与观测 token 拼接送入冻结 LLM；(3) 状态投影：LLM 输出 token 投影为状态估计。

### 关键设计

1. **观测嵌入（Observation Embedding）**

    - 功能：将滑动窗口内的观测序列 $\boldsymbol{Y}_t \in \mathbb{R}^{T \times N}$ 分段嵌入
    - 核心思路：按段长 $L$ 分段保持多维结构，通过 ObsEmbedding 映射到 LLM 的 $D$ 维隐空间
    - 设计动机：避免单序列展平破坏变量间的固有关联（如位置-速度关系）

2. **System-as-Prompt（SaP）**

    - 功能：用自然语言描述当前系统的任务指令和示例
    - 核心思路：Task Instruction（系统方程、噪声特性）+ Task Examples（示例输入输出对），tokenize 后与观测 token 拼接
    - 设计动机：利用 LLM 的上下文学习能力，无需重训即可适应新系统

3. **状态投影（State Projection）**

    - 功能：从 LLM 输出 token 投影为状态估计
    - 核心思路：去除 LLM 原始 embedding/projection 层，仅用核心 Transformer 层，输出 token 通过 StateProjection 映射到 $\mathbb{R}^{L \times M}$

### 损失函数 / 训练策略

MSE 损失：$\mathcal{L}(\boldsymbol{\theta}) = \|\boldsymbol{x}_t - \hat{\boldsymbol{x}}_t\|_2^2$。仅训练 ObsEmbedding 和 StateProjection 参数，LLM 核心层完全冻结。

## 实验关键数据

### 主实验（多个动力系统）

| 系统 | KF/EKF | 粒子滤波 | 学习型SOTA | **LLM-Filter** |
|------|--------|---------|-----------|---------------|
| 线性系统 | 0.12 | 0.15 | 0.08 | **0.06** |
| Lorenz-63 | 2.45 | 1.82 | 0.95 | **0.72** |
| Hopf 系统 | 1.89 | 1.34 | 0.78 | **0.61** |
| 未见系统(零样本) | N/A | N/A | 3.45 | **1.23** |

### 消融实验

| 配置 | MSE | 说明 |
|------|------|------|
| 无 SaP | 1.85 | 缺少系统信息 |
| 仅 Task Instruction | 1.12 | 有指令无示例 |
| 仅 Task Examples | 0.95 | 有示例无指令 |
| **完整 SaP** | **0.72** | **指令+示例** |
| 微调 LLM | 0.68 | 微调略优但失去泛化 |

### 关键发现

- LLM-Filter 在已知系统上超越所有 SOTA 学习型滤波器
- 零样本泛化到未见系统时，LLM-Filter 显著优于需要重训的学习型方法
- 观察到 scaling law：模型越大、训练越长，精度越高
- SaP 中 Task Instruction 和 Task Examples 缺一不可

## 亮点与洞察

- **控制-滤波对偶性的利用**：大控制模型（如 RT-2、OpenVLA）的成功启发了大滤波模型的设计。这是对偶性在深度学习时代的新应用。
- **冻结 LLM 策略**：仅训练输入/输出适配层，保留 LLM 的泛化能力。这个模式可迁移到其他连续信号处理任务。
- **Scaling Law**：LLM 规模越大滤波精度越高，预示着通用滤波基础模型的可能性。

## 局限与展望

- SaP 的设计需要对系统有先验知识（方程形式等）
- 高频低延迟场景下 LLM 推理速度可能不足
- 仅在几个经典系统上验证，高维复杂系统需更多实验

## 相关工作与启发

- **vs Kalman Filter**：KF 需精确线性模型，LLM-Filter 无需手工建模
- **vs 学习型滤波器**：学习型方法切换系统需重训；LLM-Filter 通过 SaP 实现零样本泛化

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ LLM 重编程为滤波器是全新思路
- 实验充分度: ⭐⭐⭐⭐ 多个系统，消融充分
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰，方法优雅
- 价值: ⭐⭐⭐⭐⭐ 开辟了通用滤波基础模型的新方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Towards One-for-All Anomaly Detection for Tabular Data](../../ICML2026/self_supervised/towards_one-for-all_anomaly_detection_for_tabular_data.md)
- [\[NeurIPS 2025\] Implicit Modeling for Transferability Estimation of Vision Foundation Models](implicit_modeling_for_transferability_estimation_of_vision_foundation_models.md)
- [\[CVPR 2025\] Spectral State Space Model for Rotation-Invariant Visual Representation Learning](../../CVPR2025/self_supervised/spectral_state_space_model_for_rotation-invariant_visual_representation_learning.md)
- [\[CVPR 2026\] NitroGen: An Open Foundation Model for Generalist Gaming Agents](../../CVPR2026/self_supervised/nitrogen_an_open_foundation_model_for_generalist_gaming_agents.md)
- [\[ICML 2025\] Foundation Model Insights and a Multi-Model Approach for Superior Fine-Grained One-shot Subset Selection](../../ICML2025/self_supervised/foundation_model_insights_and_a_multi-model_approach_for_superior_fine-grained_o.md)

</div>

<!-- RELATED:END -->
