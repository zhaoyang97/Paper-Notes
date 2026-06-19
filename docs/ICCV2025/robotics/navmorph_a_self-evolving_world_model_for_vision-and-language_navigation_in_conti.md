---
title: >-
  [论文解读] NavMorph: A Self-Evolving World Model for Vision-and-Language Navigation in Continuous Environments
description: >-
  [ICCV 2025][机器人][视觉语言导航] 提出 NavMorph，一种基于 RSSM 的**自进化世界模型**，通过 World-aware Navigator 和 Foresight Action Planner 在隐空间建模连续环境动态，并引入上下文进化记忆（CEM）实现在线测试时的快速适应。
tags:
  - "ICCV 2025"
  - "机器人"
  - "视觉语言导航"
  - "世界模型"
  - "连续环境"
  - "自进化"
  - "RSSM"
---

# NavMorph: A Self-Evolving World Model for Vision-and-Language Navigation in Continuous Environments

**会议**: ICCV 2025  
**arXiv**: [2506.23468](https://arxiv.org/abs/2506.23468)  
**代码**: [https://github.com/Feliciaxyao/NavMorph](https://github.com/Feliciaxyao/NavMorph)  
**领域**: 机器人导航 / 具身智能  
**关键词**: 视觉语言导航, 世界模型, 连续环境, 自进化, RSSM

## 一句话总结

提出 NavMorph，一种基于 RSSM 的**自进化世界模型**，通过 World-aware Navigator 和 Foresight Action Planner 在隐空间建模连续环境动态，并引入上下文进化记忆（CEM）实现在线测试时的快速适应。

## 研究背景与动机

VLN-CE（连续环境中的视觉语言导航）要求智能体在真实 3D 环境中执行低级动作（前进 0.25m、转向 15°等）。与离散 VLN 不同，连续环境的状态空间巨大且连续，带来独特挑战：

**现有世界模型局限**：PathDreamer、DreamWalker 等先行工作要么只支持离散状态转换，要么依赖像素级预测（计算昂贵），无法有效建模连续时空动态

**在线适应能力不足**：这些世界模型在预训练阶段学习后固定不变，面对新环境的分布偏移时性能退化

**缺乏前瞻规划**：传统 VLN 方法基于当前观测做决策，缺少对未来环境变化的预测能力

NavMorph 的核心思想：受人类"心理表征"启发，在隐空间中构建一个**持续进化的环境动态模型**，使智能体具备前瞻规划和在线适应能力。

## 方法详解

### 整体框架

NavMorph 由三个核心组件构成：
- **World-aware Navigator（推断网络）**：从观测推断环境隐状态
- **Foresight Action Planner（预测网络）**：基于隐状态预测未来场景并决策
- **Contextual Evolution Memory（CEM）**：跨 episode 积累导航经验，支持在线自进化

### 关键设计

#### 1. 基于 RSSM 的隐空间建模

采用 Recurrent State-Space Model，将隐状态分解为两部分：
- **确定性历史 $\mathbf{h}_t$**：通过循环模块编码时序动态，$\mathbf{h}_t = f(\mathbf{h}_{t-1}, \mathbf{s}_{t-1})$
- **随机状态 $\mathbf{s}_t$**：建模环境不确定性

推断网络从后验分布采样：
$$q(\mathbf{s}_t | \mathbf{o}_{1:t}, \mathbf{a}_{1:t-1}) \sim \mathcal{N}(\mu_\phi(\mathbf{h}_t, \mathbf{a}_{t-1}, \mathbf{x}_t), \sigma_\phi(\mathbf{h}_t, \mathbf{a}_{t-1}, \mathbf{x}_t)\mathbf{I})$$

预测网络从先验分布采样：
$$p(\hat{\mathbf{s}}_t | \mathbf{h}_t, \hat{\mathbf{s}}_{t-1}) \sim \mathcal{N}(\mu_\theta(\mathbf{h}_t, \hat{\mathbf{a}}_{t-1}), \sigma_\theta(\mathbf{h}_t, \hat{\mathbf{a}}_{t-1})\mathbf{I})$$

关键区别：推断网络可利用真实观测，预测网络仅依赖以往隐状态和预测动作。动作定义为 $\Delta position_t$（相邻步之间的位移），而非低级控制指令。

#### 2. 上下文进化记忆（CEM）

CEM 维护 $N_m$ 个场景上下文特征 $\{\mathbf{v}_m\}_{m=1}^{N_m}$，代替传统 RNN/LSTM 作为循环模块的核心增强：

**检索增强**：从 CEM 中检索 Top-K 相关特征融入当前状态
$$\tilde{\mathbf{h}}_t = (1-\alpha)\mathbf{h}_t + \alpha \sum_{k=1}^{K} \mu_k \mathbf{v}_k$$

**前向更新**（非梯度回传）：
$$\mathbf{v}_k \leftarrow (1-\beta)\mathbf{v}_k + \beta \mathbf{h}_t$$

这种设计有两大优势：
- **训练时**：通过梯度优化积累多环境的导航知识
- **测试时**：通过前向更新快速适应新环境，无需反向传播，效率极高

#### 3. 特征级预测而非像素级

视觉解码器 $d_\theta$ 预测的是**视觉嵌入** $\hat{\mathbf{x}}_t$（特征向量），而非像素图像。这避免了生成模型的高计算成本，同时保留了足够的语义信息用于动作规划。

### 损失函数 / 训练策略

总损失 $\mathcal{L} = \mathcal{L}_W + \mathcal{L}_{IL}$，其中世界模型损失：

$$\mathcal{L}_W = \underbrace{\ell_{re}}_{\text{重建}} + \underbrace{\ell_{ac}}_{\text{动作预测}} + \gamma \cdot \underbrace{\ell_{kl}}_{\text{KL 散度}}$$

- $\ell_{re}$：视觉嵌入重建（包含 NDTW 正则化保证时序一致性）
- $\ell_{ac}$：动作预测（L2 损失 + NDTW 正则化）
- $\ell_{kl}$：后验与先验分布对齐（可闭式计算）
- $\mathcal{L}_{IL}$：模仿学习损失（DAgger 式教师动作监督）

预训练时观测 $T$ 步后预测未来 $T_p$ 步，形成"观测-预测"两阶段训练。

## 实验关键数据

### 主实验（R2R-CE Val Unseen）

| 方法 | 摄像头 | NE↓ | SR(%) | SPL(%) |
|------|--------|:---:|:---:|:---:|
| VLN-3DFF | 单目 | 6.05 | 43.8 | 29.4 |
| **NavMorph** | **单目** | **5.75** | **47.9** | **33.2** |
| ETPNav | 全景 | 4.69 | 57 | 49 |
| **NavMorph** | **全景** | **4.62** | **59** | **50** |
| HNR | 全景 | 4.57 | 61 | 51 |
| **NavMorph** | **全景** | **4.37** | **64** | **53** |

### 消融实验（R2R-CE Val Unseen，单目 VLN-3DFF 基线）

| 方法 | TL↓ | NE↓ | SR(%) | SPL(%) |
|------|:---:|:---:|:---:|:---:|
| Base model | 26.16 | 6.05 | 43.77 | 29.39 |
| NavMorph (full) | **22.54** | **5.75** | **47.91** | **33.22** |
| w/o $\ell_{re}$ | 20.25 | 5.85 | 45.51 | 32.38 |
| w/o $\ell_{ac}$ | 25.14 | 5.96 | 44.81 | 31.22 |
| w/o $\ell_{kl}$ | 25.69 | 6.30 | 44.10 | 30.44 |
| NavMorph w/o SE | 23.34 | 5.92 | 45.08 | 31.19 |
| CEM vs LSTM | CEM: 21.22s | - | 47.91 | 33.22 |
| | LSTM: 44.56s | - | 43.67 | 29.81 |

### 关键发现

1. NavMorph 在单目设置下 SR 提升 **4.1%**，SPL 提升 **3.8%**，同时**轨迹长度减少 14%**，说明规划更高效
2. CEM 比 LSTM 自进化快 **2.1 倍**且指标全面领先，验证了前向更新优于梯度回传的适应策略
3. 最优 CEM 大小为 1000：太小不够存储，太大引入噪声冗余
4. 即使不做自进化（w/o SE），世界模型架构本身也带来 +1.3% SR / +1.8% SPL 的提升

## 亮点与洞察

- **CEM 的设计哲学**：前向更新（momentum-style）避免了测试时反向传播的计算开销，实现了真正的"零额外训练成本的在线适应"
- 特征级预测避免了像素生成的复杂度，同时 NDTW 正则化保证了时序连贯——这是一个实用的工程取舍
- 世界模型与导航策略的解耦设计使 NavMorph 可作为即插即用模块增强多种现有 VLN 方法

## 局限与展望

- CEM 仍依赖余弦相似度检索，对语义理解深度有限
- 奖励函数缺乏直接来自真实目标的信号，未能做显式规划（acknowledged by 作者）
- 前向更新的 $\alpha, \beta$ 超参敏感度未充分探讨
- 仅在 Matterport3D 环境测试，对真实机器人的迁移能力未验证

## 相关工作与启发

- 与 Dreamer 系列（Hafner et al.）的世界模型设计一脉相承，但首次系统适配 VLN-CE 任务
- CEM 机制与 Neural Episodic Control / MERLIN 等记忆增强 RL 方法异曲同工
- 启发未来工作：将基础模型（如 VLM）作为世界模型的一部分，或设计模块化可组合的进化架构

## 评分

- 新颖性: ⭐⭐⭐⭐ （自进化世界模型 + CEM 设计新颖）
- 实验充分度: ⭐⭐⭐⭐ （R2R-CE / RxR-CE + 多基线 + 充分消融）
- 写作质量: ⭐⭐⭐⭐ （框架图清晰，推导完整）
- 价值: ⭐⭐⭐⭐ （对 VLN-CE 领域有明确推进）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] C-NAV: Towards Self-Evolving Continual Object Navigation in Open World](../../NeurIPS2025/robotics/c-nav_towards_self-evolving_continual_object_navigation_in_open_world.md)
- [\[CVPR 2026\] Semantic Audio-Visual Navigation in Continuous Environments](../../CVPR2026/robotics/semantic_audio-visual_navigation_in_continuous_environments.md)
- [\[ICCV 2025\] COSMO: Combination of Selective Memorization for Low-cost Vision-and-Language Navigation](cosmo_combination_of_selective_memorization_for_low-cost_vision-and-language_nav.md)
- [\[CVPR 2026\] NavForesee: A Unified Vision-Language World Model for Hierarchical Planning and Dual-Horizon Navigation Prediction](../../CVPR2026/robotics/navforesee_a_unified_vision-language_world_model_for_hierarchical_planning_and_d.md)
- [\[ICCV 2025\] DexVLG: Dexterous Vision-Language-Grasp Model at Scale](dexvlg_dexterous_vision-language-grasp_model_at_scale.md)

</div>

<!-- RELATED:END -->
