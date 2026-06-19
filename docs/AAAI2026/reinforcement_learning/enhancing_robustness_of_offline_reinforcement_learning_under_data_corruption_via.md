---
title: >-
  [论文解读] Enhancing Robustness of Offline RL Under Data Corruption via SAM
description: >-
  [AAAI 2026 (Student Abstract, Oral)][强化学习][离线RL] 首次将 Sharpness-Aware Minimization (SAM) 作为即插即用优化器应用于离线 RL，假设数据损坏导致损失景观中出现尖锐极小值从而泛化差，SAM 通过寻找平坦极小值提升鲁棒性，在 D4RL 基准上 IQL+SAM 平均得分从 34.47 提升到 44.40。
tags:
  - "AAAI 2026 (Student Abstract, Oral)"
  - "强化学习"
  - "离线RL"
  - "数据损坏"
  - "SAM优化器"
  - "平坦极小值"
  - "鲁棒性"
---

# Enhancing Robustness of Offline RL Under Data Corruption via SAM

**会议**: AAAI 2026 (Student Abstract, Oral)  
**arXiv**: [2511.17568](https://arxiv.org/abs/2511.17568)  
**代码**: 无  
**领域**: 强化学习  
**关键词**: 离线RL, 数据损坏, SAM优化器, 平坦极小值, 鲁棒性

## 一句话总结
首次将 Sharpness-Aware Minimization (SAM) 作为即插即用优化器应用于离线 RL，假设数据损坏导致损失景观中出现尖锐极小值从而泛化差，SAM 通过寻找平坦极小值提升鲁棒性，在 D4RL 基准上 IQL+SAM 平均得分从 34.47 提升到 44.40。

## 研究背景与动机

**领域现状**：离线 RL 从静态数据集学习策略，避免在线交互。IQL 通过 expectile 回归学习 Q 函数和 V 函数，展现出对部分数据损坏的固有鲁棒性。RIQL 在 IQL 基础上加入 Huber 损失和分位数估计器专门处理动态损坏。

**现有痛点**：即使 IQL 和 RIQL 在动态损坏下表现不错，但在观测损坏和混合损坏下性能显著下降。

**核心假设**：数据损坏在损失景观中创建了尖锐的、不可靠的极小值。收敛到这些尖锐极小值的模型不够鲁棒——输入数据的小扰动可能导致值估计的大误差。

**切入角度**：不修改算法的损失函数，而是替换优化器。SAM 通过两步 minimax 过程寻找平坦区域：先找到局部最大化损失的对抗性权重扰动，再在该"最坏情况"点计算梯度更新原始参数。

**核心 idea**：用 SAM 替换 Adam 作为价值函数优化器，寻找平坦极小值而非尖锐极小值。

## 方法详解

### 整体框架
SAM 作为 PyTorch 自定义 Optimizer 类包装 Adam。消融发现仅应用于价值函数网络效果最稳定。

### 关键设计

1. **SAM 两步优化过程**:

    - **上升步**：计算对抗性扰动 $\hat{\epsilon}(\theta) = \rho \frac{\nabla_\theta L(\theta)}{\|\nabla_\theta L(\theta)\|_2}$，其中 $\rho$ 控制邻域大小
    - **下降步**：在扰动后的参数 $\theta' = \theta + \hat{\epsilon}(\theta)$ 处计算梯度，用该梯度更新原始 $\theta$
    - 直觉：惩罚尖锐度——最小化邻域内最高损失值

2. **仅应用于价值函数**:

    - 消融实验发现：同时应用于策略网络反而不稳定
    - 价值函数的尖锐极小值直接影响策略提取质量

### 损失函数 / 训练策略
不改变 IQL/RIQL 的损失函数，仅替换优化器。3 个随机种子，30% 数据损坏率。

## 实验关键数据

### 主实验（D4RL medium-replay，随机损坏）

| 环境 | 损坏类型 | IQL | IQL+SAM | RIQL | RIQL+SAM |
|------|---------|-----|---------|------|----------|
| HalfCheetah | 观测 | 21.01 | **33.33** | 26.03 | **33.74** |
| HalfCheetah | 混合 | 20.93 | **33.02** | 22.08 | **32.06** |
| Walker2d | 观测 | 24.74 | **31.75** | 30.48 | **30.93** |
| Hopper | 观测 | 58.42 | **73.21** | 44.09 | **53.19** |
| **平均** | | 34.47 | **44.40** | 33.97 | **39.47** |

### 对抗损坏结果

| 方法 | 平均得分 |
|------|---------|
| IQL | 22.45 |
| **IQL+SAM** | **36.03** (+60%) |
| RIQL | 38.20 |
| **RIQL+SAM** | **40.09** |

### 奖励曲面可视化
IQL 收敛到尖锐高峰和深谷区域；IQL+SAM 学到显著更平滑更平坦的奖励曲面——直观确认 SAM 引导代理到更鲁棒的解。

### 关键发现
- SAM 对混合损坏的改进尤为明显——混合损坏是最困难的设定
- 对抗损坏下 IQL+SAM 提升 60%+，效果惊人
- 仅需替换优化器，不改变任何算法逻辑——真正的即插即用
- 计算开销约 2x（每步需两次前向+反向传播）

## 亮点与洞察
- **从优化器几何性质入手**的视角很独特——不是设计新算法/新损失，而是直接改优化器
- **"数据损坏→尖锐极小值→泛化差"**的因果假设得到可视化验证，说服力强
- **即插即用特性**使其可以与任何离线 RL 算法组合

## 局限与展望
- Student Abstract 篇幅限制，仅 3 个环境的 medium-replay 数据集
- SAM 的计算开销（2x）在实时应用中需要考虑
- $\rho$ 参数的敏感性分析不够
- 未与其他鲁棒RL方法（如 UWMSG, TRACER, ADG）全面对比

## 相关工作与启发
- **vs RIQL**: RIQL 修改损失函数处理特定损坏类型；SAM 从优化层面通用增强，互补
- **vs ADG**: ADG 用扩散模型恢复干净数据（预处理方法）；SAM 从优化角度处理（训练方法），可组合
- **vs SAF**: SAF 提供更高效的 SAM 变体；可替换以减少计算开销

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将 SAM 用于离线 RL，视角新颖
- 实验充分度: ⭐⭐⭐ Student Abstract 篇幅限制，环境有限
- 写作质量: ⭐⭐⭐⭐ 清晰简洁，可视化有说服力
- 价值: ⭐⭐⭐⭐ 通用即插即用方案，启发性强

## 补充分析
- 本文提出的方法在其特定子领域代表了一种有意义的技术进步
- 核心创新点在于将领域特有的结构性先验知识编码到模型设计中，而非完全依赖数据驱动的端到端学习
- 与同期发表的其他顶会工作相比，本文在问题定义和方法设计的系统性上展现了较高水平的研究素养
- 在实际部署场景中，还需综合考虑计算效率、实时性要求、数据隐私保护以及系统可扩展性等工程因素
- 方法的核心思想具有一定的可迁移性——类似的设计范式可能在相关但不同的任务和数据模态上发挥作用
- 消融实验的设计较为合理，为理解各组件对整体性能的贡献提供了清晰的分析视角
- 未来可以考虑与大规模预训练模型（LLM/VLM/基础模型）结合，利用其强大的表征学习能力进一步提升方法的性能上限

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] A Survey of Reinforcement Learning for Large Language Models under Data Scarcity: Challenges and Solutions](../../ACL2026/reinforcement_learning/a_survey_of_reinforcement_learning_for_large_language_models_under_data_scarcity.md)
- [\[ICML 2026\] Trajectory-Level Data Augmentation for Offline Reinforcement Learning](../../ICML2026/reinforcement_learning/trajectory-level_data_augmentation_for_offline_reinforcement_learning.md)
- [\[ICML 2025\] Automatic Reward Shaping from Confounded Offline Data](../../ICML2025/reinforcement_learning/automatic_reward_shaping_from_confounded_offline_data.md)
- [\[CVPR 2026\] AnyDoc: Enhancing Document Generation via Large-Scale HTML/CSS Data Synthesis and Height-Aware Reinforcement Optimization](../../CVPR2026/reinforcement_learning/anydoc_enhancing_document_generation_via_large-scale_htmlcss_data_synthesis_and_.md)
- [\[ICLR 2026\] Less is More: Clustered Cross-Covariance Control for Offline RL](../../ICLR2026/reinforcement_learning/less_is_more_clustered_cross-covariance_control_for_offline_rl.md)

</div>

<!-- RELATED:END -->
