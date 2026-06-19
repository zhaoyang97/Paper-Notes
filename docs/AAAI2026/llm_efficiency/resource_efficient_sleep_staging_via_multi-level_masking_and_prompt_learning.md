---
title: >-
  [论文解读] Resource Efficient Sleep Staging via Multi-Level Masking and Prompt Learning
description: >-
  [AAAI2026][LLM效率][sleep staging] 提出 MASS (Mask-Aware Sleep Staging) 框架，通过多层级 masking 策略和层次化 prompt learning 机制，仅用 **10% 的原始 EEG 信号**即可实现可靠的睡眠分期，为资源受限的可穿戴睡眠监测系统提供方案。
tags:
  - "AAAI2026"
  - "LLM效率"
  - "sleep staging"
  - "EEG"
  - "masking"
  - "提示学习"
  - "wearable monitoring"
  - "resource efficiency"
---

# Resource Efficient Sleep Staging via Multi-Level Masking and Prompt Learning

**会议**: AAAI2026  
**arXiv**: [2511.06785](https://arxiv.org/abs/2511.06785)  
**代码**: [AnsonAiTRAY/MASS](https://github.com/AnsonAiTRAY/MASS)  
**领域**: LLM效率  
**关键词**: sleep staging, EEG, masking, prompt learning, wearable monitoring, resource efficiency

## 一句话总结
提出 MASS (Mask-Aware Sleep Staging) 框架，通过多层级 masking 策略和层次化 prompt learning 机制，仅用 **10% 的原始 EEG 信号**即可实现可靠的睡眠分期，为资源受限的可穿戴睡眠监测系统提供方案。

## 研究背景与动机
自动睡眠分期对睡眠质量评估和障碍诊断至关重要。AASM 标准将 30 秒 EEG 信号（sleep epoch）分为 W/REM/N1/N2/N3 五个阶段。

- **资源限制**：可穿戴/家庭睡眠监测设备电池容量有限，长时间连续 EEG 采集的功耗是核心瓶颈
- **现有方法局限**：DeepSleepNet/AttnSleep/NeuroNet 等 SOTA 方法依赖完整连续 EEG 信号，数据缺失时性能急剧下降
- **硬件基础**：ADS1299 等信号放大器支持正常/待机模式微秒级切换，为按需采集提供硬件支撑
- **核心定位**：首个从数据效率角度出发的神经网络睡眠分期方法

## 方法详解

### 多层级 Masking 策略
在 epoch 和 patch 两个层级施加随机 masking：
1. **Epoch-level masking**：以 mask ratio $r_e$ 随机遮蔽整个 30 秒 epoch
2. **Patch-level masking**：每个未遮蔽 epoch 分为 30 个 1 秒 patch，以 mask ratio $r_a$ 进一步遮蔽
3. 剩余可见 patch 经 PSD 频域变换 + 线性投影得到特征表示 $E_{vis} \in \mathbb{R}^{e(1-r_e) \times 30(1-r_a) \times d_a}$

训练和推理阶段均施加 masking，模拟真实部分观测场景。

### Global Prompt Learning
将所有可见 patch 展平为全局序列，加入 learnable CLS token，用固定正弦位置编码保留原始序列中的绝对位置，经浅层 Transformer 编码器（$L_p$ 层）生成 global prompt $z_{prompt}$，作为语义锚引导后续建模。

### 分层特征建模
- **Patch-level**：每个可见 epoch 内，将 CLS token、global prompt、未遮蔽 patch 拼接输入 Transformer 编码器，提取 epoch 内细粒度模式
- **Epoch-level**：将 patch-level CLS 输出与 global prompt 拼接，遮蔽 epoch 填零，经 Bi-GRU 建模 epoch 间时序转换
- **辅助任务**：阶段转换预测（binary），增强 epoch 间动态捕捉

### 训练目标
$\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda_1 \mathcal{L}_{Cos} + \lambda_2 \mathcal{L}_{Trans}$

## 实验关键数据

### 不同信号完整度下的 macro-F1 (%)对比

| 方法 | DREAMS-SUB 100% | 10% | Sleep-EDF-20 100% | 10% | Sleep-EDF-78 100% | 10% | SHHS 100% | 10% |
|---|---|---|---|---|---|---|---|---|
| DeepSleepNet | 67.02 | 11.08 | 76.65 | 10.27 | 71.81 | 13.97 | 73.91 | 12.88 |
| TinySleepNet | 78.49 | 8.81 | 78.41 | 9.85 | 74.11 | 26.87 | 75.22 | 26.49 |
| NeuroNet | 79.51 | 6.69 | 78.65 | 8.94 | 75.73 | 20.18 | 76.87 | 10.65 |
| **MASS** | **81.14** | **75.58** | **80.11** | **76.62** | **75.02** | **71.61** | **76.87** | **70.25** |

10% 信号下 MASS 仅损失 3-7% macro-F1，竞争方法则崩溃至个位数或二十几。

### 资源效率分析
- 10% 信号下，ADS1299-4 功耗从 22mW 降至 6.79mW（降低 **69%**），ADS131A04 从 15.8mW 降至 3.92mW（降低 **75%**）
- 模型参数效率 $\eta_p = 0.73$，推理时间效率 $\eta_t = 16.08$，均为最优

### 消融实验 (40% 信号, DREAMS-SUB)

| 变体 | ACC | macro-F1 |
|---|---|---|
| MASS-Base | 43.2 | 15.9 |
| MASS-Prompt | 45.6 | 18.5 |
| MASS-Mask | 85.8 | 79.9 |
| **MASS (完整)** | **86.4** | **80.0** |

多层级 masking 是性能的核心驱动因素。

## 亮点
- **极端数据效率**：10% 信号仅损失约 5% macro-F1，这是现有方法完全无法达到的
- **训练-推理一致**：masking 在训练和推理阶段均施加，模型天然适应部分观测
- **实际功耗验证**：结合真实放大器功耗数据证明可节省 60-75% 功耗
- **global prompt 设计**：通过保留绝对位置编码的 CLS token 聚合全局上下文，弥补 masking 造成的信息损失

## 局限性
- 仅验证单通道 EEG，多通道场景下 masking 策略需重新设计
- masking pattern 固定为随机，未探索自适应/学习式 masking
- 30 个 1 秒 patch 的分割粒度是否最优未充分讨论
- 仅在公开数据集验证，缺少真实可穿戴设备的端到端部署实验

## 评分
- 新颖性: ⭐⭐⭐⭐ — 首次从数据采集效率角度设计睡眠分期模型，masking+prompt 组合有新意
- 实验充分度: ⭐⭐⭐⭐ — 4 数据集 x 4 信号比例 x 6 基线，含消融和资源效率分析
- 写作质量: ⭐⭐⭐⭐ — 问题定义清晰，方法推导严谨
- 价值: ⭐⭐⭐⭐ — 对可穿戴睡眠监测的实际部署有直接价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] One-Prompt Strikes Back: Sparse Mixture of Experts for Prompt-based Continual Learning](../../ICLR2026/llm_efficiency/one-prompt_strikes_back_sparse_mixture_of_experts_for_prompt-based_continual_lea.md)
- [\[ICLR 2026\] Efficient Resource-Constrained Training of Transformers via Subspace Optimization](../../ICLR2026/llm_efficiency/efficient_resource-constrained_training_of_transformers_via_subspace_optimizatio.md)
- [\[ACL 2026\] Task-Aware LLM Routing with Multi-Level Task-Profile-Guided Data Synthesis for Cold-Start Scenarios](../../ACL2026/llm_efficiency/task-aware_llm_routing_with_multi-level_task-profile-guided_data_synthesis_for_c.md)
- [\[ICML 2026\] Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing](../../ICML2026/llm_efficiency/efficient_training-free_multi-token_prediction_via_embedding-space_probing.md)
- [\[CVPR 2026\] JUMP-Hand: Learning Joint-wise Uncertainty to Gate Mixture of View Experts for Multi-View 3D Hand Reconstruction](../../CVPR2026/llm_efficiency/jump-hand_learning_joint-wise_uncertainty_to_gate_mixture_of_view_experts_for_mu.md)

</div>

<!-- RELATED:END -->
