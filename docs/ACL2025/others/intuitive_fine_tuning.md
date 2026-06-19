# Intuitive Fine-Tuning: Towards Simplifying Alignment into a Single Process

| 会议 | 领域 | 关键词 |
|------|------|--------|
| ACL 2025 | 其他 / 对齐与微调 | LLM对齐, SFT, 偏好优化, MDP, 残差连接, 统一框架 |

> **一句话总结**: 通过MDP框架统一分析SFT和偏好优化（PO），揭示SFT只是PO的特例，提出Intuitive Fine-Tuning（IFT）方法，利用时间残差连接融合SFT的数据效率和PO的对齐效果，仅用正样本和单策略模型即可实现接近或超越SFT+PO的性能。

## 研究背景与动机

**研究问题**: SFT和偏好优化（PO，如DPO/PPO）通常作为对齐的两个独立阶段顺序执行，存在范式鸿沟（损失函数、数据格式、辅助模型不同），能否将两者统一为一个过程？

**现有方法的不足**:
- **SFT**: 使用ground truth前缀预测下一token，但这些前缀偏离模型自身分布，导致偏好估计有偏、转移优化次优
- **PPO**: 无偏的模型偏好估计，但需要奖励模型、在线采样，计算代价高
- **DPO**: 理论最优估计，但需要配对偏好数据（正+负样本），数据收集成本高；离线变体使用非当前策略生成的负样本，估计有偏
- **现有统一尝试**: 如ORPO、SimPO仍需偏好标注数据或参考模型

**核心动机**: SFT的偏好估计为什么有偏？因为在预测第$n$个token时，用的是ground truth的前$n-1$个token作为上下文，而非模型自己生成的前缀。能否在不增加数据和计算成本的情况下修正这个偏差？

## 方法详解

### 整体框架

IFT分三步：
1. **前向推理一步**: 对每个ground truth前缀，用当前模型预测下一个token（获取模型偏好）
2. **直觉偏好估计**: 将模型预测token的embedding与ground truth token的embedding按$\lambda$加权混合，构建更接近模型分布的先验状态
3. **动态关系传播**: 通过累积求和重构损失函数，使当前token的梯度受未来token准确性影响

### 关键设计

- **时间残差连接**: $\hat{s_i^{\theta}} = (1-\lambda) \cdot s_i^* + \lambda \cdot \pi_\theta(s_{i-1}^*)$，将模型生成的embedding残差传递给下一个token，让模型在ground truth上下文中感知自己的"整体回答直觉"
- **MDP统一视角**: 定义偏好估计（Preference Estimation）和转移优化（Transition Optimization），表明SFT偷偷假设 $T_\theta(s_{n-1}^*, \rho_0)=1$（前缀一定是模型会生成的），导致过估计
- **仅需正样本**: 不同于DPO需要正负配对数据，IFT只需SFT同等格式和规模的数据

### 损失函数

IFT的损失函数在标准交叉熵基础上引入累积求和实现动态关系传播：

$$\mathcal{L}_{IFT} = \mathbb{E}\left[-\sum_{n=0}^{N}\sum_{i=n}^{N} \log \mathcal{T}_\theta(a_i^*, \delta_\theta(s_i^*))\right]$$

其中$\delta_\theta$为直觉偏好估计函数。该损失隐式满足Bellman方程，兼具RLHF的有效性和SFT的效率。可选加入衰减因子$\alpha$处理长序列。

## 实验

### 主实验（Open-LLM Leaderboard，Mistral-7B基座）

| 方法 | ARC | MMLU | TruthfulQA | WinoGrande | GSM8K | 平均 |
|------|-----|------|------------|------------|-------|------|
| SFT | 56.49 | 60.44 | 55.57 | 77.90 | 42.84 | 58.65 |
| DPO | 61.86 | 61.02 | 47.98 | 76.64 | 43.89 | 58.28 |
| ORPO | 56.66 | 60.57 | 51.77 | 77.19 | 42.30 | 57.70 |
| SimPO | 59.90 | 52.61 | 47.25 | 78.30 | 37.53 | 55.15 |
| **IFT** | 56.74 | 60.49 | **57.65** | **78.45** | **44.73** | **59.61** |

### 生成质量评估（Alpaca-Eval）

| 方法 | 数据量 | 偏好数据 | 参考模型 | Win Rate | LC Win Rate |
|------|--------|----------|----------|----------|-------------|
| SFT | 120k | ✗ | ✗ | 82.56 | 78.32 |
| DPO | 120k | ✓ | ✓ | 74.00 | 73.12 |
| ORPO | 120k | ✗ | ✓ | 85.14 | 76.60 |
| **IFT** | 120k | **✗** | **✗** | **85.18** | **78.78** |
| SFT+DPO | 320k | ✓ | ✓ | 91.62 | 81.54 |
| SFT+IFT | 260k | **✗** | **✗** | 88.37 | **81.29** |

### 关键发现

1. **单阶段IFT即超越SFT**: 在6个基准上，IFT平均59.61 vs SFT的58.65，且不需要偏好数据
2. **IFT接近SFT+DPO序贯训练效果**: SFT+IFT（260k数据）在LC Win Rate上达81.29，接近SFT+DPO（320k数据）的81.54
3. **TruthfulQA上优势显著**: IFT的57.65大幅超过DPO的47.98和ORPO的51.77，说明IFT更擅长事实跟随
4. **数据效率极高**: 用120k非偏好数据就能达到需要320k配对数据方法的水平
5. **Frozen Lake实验验证**: 在可解释环境中确认IFT学到了竞争性策略

## 亮点

- 理论优雅：通过MDP框架统一理解SFT和PO，揭示SFT偏差的根本原因
- 时间残差连接的设计简洁而有效，不引入额外模型或数据需求
- 仅需正样本+单策略模型，大幅降低对齐门槛
- 在事实性（TruthfulQA）和生成质量（Alpaca-Eval）上表现突出

## 局限性

- 理论分析依赖MDP的理想化假设，实际语言生成的状态空间远比MDP复杂
- $\lambda$超参数的选择需要调优，论文未充分讨论敏感度
- 前向推理一步增加了约1倍的计算量（虽然比PPO/DPO的在线采样少）
- 主要在7B-8B模型上验证，未在更大规模模型上充分测试

## 相关工作

- **SFT**: 标准监督微调，用ground truth做teacher forcing
- **PPO**: Schulman et al. (2017)，在线策略优化+奖励模型
- **DPO**: Rafailov et al. (2024)，合并奖励建模和策略优化
- **ORPO/SimPO/TDPO**: 尝试统一SFT和PO的中间方案，但仍需偏好数据
- **Unlikelihood Training**: Welleck et al. (2019)，在SFT中引入负样本惩罚

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 新颖性 | 5 |
| 理论深度 | 5 |
| 实验充分性 | 4 |
| 写作质量 | 4 |
| 实用价值 | 4 |
| 总评 | 4.4 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] SoRFT: Issue Resolving with Subtask-oriented Reinforced Fine-Tuning](sorft_issue_resolving_with_subtask-oriented_reinforced_fine-tuning.md)
- [\[ACL 2025\] A Spatio-Temporal Point Process for Fine-Grained Modeling of Reading Behavior](a_spatio-temporal_point_process_for_fine-grained_modeling_of_reading_behavior.md)
- [\[ACL 2025\] AIDE: Attribute-Guided Multi-Hop Data Expansion for Data Scarcity in Task-Specific Fine-tuning](aide_attribute-guided_multi-hop_data_expansion_for_data_scarcity_in_task-specifi.md)
- [\[ECCV 2024\] Dropout Mixture Low-Rank Adaptation for Visual Parameters-Efficient Fine-Tuning](../../ECCV2024/others/dropout_mixture_low-rank_adaptation_for_visual_parameters-efficient_fine-tuning.md)
- [\[ACL 2025\] Instruction-Tuning Data Synthesis from Scratch via Web Reconstruction](instruction-tuning_data_synthesis_from_scratch_via_web_reconstruction.md)

</div>

<!-- RELATED:END -->
