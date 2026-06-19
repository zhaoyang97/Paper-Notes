---
title: >-
  [论文解读] Distilling Deep Reinforcement Learning into Interpretable Fuzzy Rules: An Explainable AI Framework
description: >-
  [AAAI 2026][强化学习][模糊逻辑] 提出层次化 Takagi-Sugeno-Kang (TSK) 模糊分类器系统，将深度 RL 的神经网络策略蒸馏为人类可读的 IF-THEN 模糊规则，引入三个量化可解释性度量（FRAD、FSC、ASG），在 Lunar Lander 连续控制任务上以 81.48% 的保真度超越决策树 21 个百分点。
tags:
  - "AAAI 2026"
  - "强化学习"
  - "模糊逻辑"
  - "TSK系统"
  - "策略蒸馏"
  - "可解释AI"
  - "连续控制"
---

# Distilling Deep Reinforcement Learning into Interpretable Fuzzy Rules: An Explainable AI Framework

**会议**: AAAI 2026  
**arXiv**: [2603.13257](https://arxiv.org/abs/2603.13257)  
**代码**: 无  
**领域**: 可解释强化学习  
**关键词**: 模糊逻辑, TSK系统, 策略蒸馏, 可解释AI, 连续控制

## 一句话总结

提出层次化 Takagi-Sugeno-Kang (TSK) 模糊分类器系统，将深度 RL 的神经网络策略蒸馏为人类可读的 IF-THEN 模糊规则，引入三个量化可解释性度量（FRAD、FSC、ASG），在 Lunar Lander 连续控制任务上以 81.48% 的保真度超越决策树 21 个百分点。

## 研究背景与动机

### 问题定义

深度强化学习（DRL）智能体在连续控制中表现出色，但策略编码在深度神经网络权重中，对人类观察者来说是黑盒的。例如，月球着陆器 PPO 策略包含数百个参数，当智能体输出推力向量 $[0.36, 0.71]$ 时，决策理由不可获取：是基于倾斜角？还是位置与速度的复杂交互？

### 现有方法的局限

**局部解释方法（SHAP/LIME）**：仅回答"哪些特征贡献了此动作"，不揭示全局策略结构或操作模式；实例特定，不可移植

**符号蒸馏方法（决策树/VIPER）**：以分段常数近似光滑控制函数，需要极深的树才能达到可接受保真度，牺牲可解释性；边界处动作突变

**已有神经模糊方法**：在训练中集成模糊逻辑（如 Fuzzy Q-Learning），但性能不如现代深度 RL；后置模糊代理方法缺乏层次化策略、严格量化指标和全面基线比较

### 核心动机

连续控制任务需要近似光滑的非线性映射，模糊逻辑通过语言变量（如"速度高"）和 IF-THEN 规则自然地桥接数值计算与语言推理。TSK 系统直接输出连续函数而非离散模糊集，提供适合控制的光滑插值预测。

## 方法详解

### 整体框架

两级层次化分解：
- **Level 1（前件学习）**：K-Means 聚类将状态空间 $\mathcal{S}$ 划分为 $N$ 个操作区域（如"悬停""纠正漂移"）
- **Level 2（后件学习）**：每个区域内用加权 Ridge 回归学习局部 TSK 后件函数
- **全局推理**：归一化加权聚合活跃局部模型的输出

### 关键设计

#### 1. **隶属函数设计**

为每个聚类 $i$ 和状态维度 $k$ 定义隶属函数 $\mu_{i,k}: \mathbb{R} \to [0,1]$。比较两种类型：

**高斯隶属函数**：

$$\mu_{i,k}^{\text{Gauss}}(s_k) = \exp\left(-\frac{(s_k - c_{i,k})^2}{2\sigma_{i,k}^2}\right)$$

特点：无限支撑，光滑过渡；每个状态在所有聚类中都有非零隶属度，导致许多规则同时激活。

**三角隶属函数**：

$$\mu_{i,k}^{\text{Tri}}(s_k) = \max\left(0, \min\left(\frac{s_k - l_{i,k}}{c_{i,k} - l_{i,k}}, \frac{r_{i,k} - s_k}{r_{i,k} - c_{i,k}}\right)\right)$$

其中 $l_{i,k} = c_{i,k} - \beta\sigma_{i,k}$, $r_{i,k} = c_{i,k} + \beta\sigma_{i,k}$（$\beta = 1.5$）。特点：紧凑支撑，稀疏激活，解释更聚焦。

设计动机：三角函数创建局部化激活（更少规则活跃），使决策逻辑更透明（假设：更高 FRAD 而不牺牲保真度）。

#### 2. **规则激发与全局推理**

规则 $i$ 对状态 $\mathbf{s}$ 的激发强度：

$$\alpha_i(\mathbf{s}) = \prod_{k=1}^d \mu_{i,k}(s_k)$$

三角函数中，任一维度零隶属度使整个规则激活为零→稀疏规则集便于人类理解。

TSK 后件函数（线性模型）：

$$f_i(\mathbf{s}) = \mathbf{w}_i^T \mathbf{s} + b_i$$

全局推理：

$$a_{\text{FCS}}(\mathbf{s}) = \frac{\sum_{i=1}^N \alpha_i(\mathbf{s}) \cdot f_i(\mathbf{s})}{\sum_{i=1}^N \alpha_i(\mathbf{s})}$$

#### 3. **三个量化可解释性度量**

**FRAD（模糊规则激活密度）**：受 HHI 指数启发，衡量解释聚焦度：

$$\text{FRAD}(\mathbf{s}) = \sum_{i=1}^N \left(\frac{\alpha_i(\mathbf{s})}{\sum_j \alpha_j(\mathbf{s})}\right)^2$$

范围 $[1/N, 1]$，越高表示越聚焦（单一规则主导）。

**FSC（模糊集覆盖度）**：验证语言词汇完整性，检查是否存在所有模糊集隶属度均低的区域：

$$\text{FSC} = \frac{1}{|\mathcal{D}|} \sum_{\mathbf{s}} \frac{1}{d} \sum_{k=1}^d \max_i \mu_{i,k}(s_k)$$

**ASG（动作空间粒度）**：衡量规则后件多样性：

$$\text{ASG} = \text{Var}(\{b_1, \ldots, b_N\})$$

高 ASG 表示规则代表不同动作模式（推进/悬停/纠正）。

### 损失函数 / 训练策略

- Level 1：K-Means 聚类（标准设置）
- Level 2：加权 Ridge 回归，每个规则 $i$ 的样本权重为 $\alpha_i(\mathbf{s}_j)$：

$$\min_{\mathbf{w}_i, b_i} \sum_j \alpha_i(\mathbf{s}_j)(a_j - \mathbf{w}_i^T \mathbf{s}_j - b_i)^2 + \lambda\|\mathbf{w}_i\|^2$$

Ridge 正则化防止规则支撑数据有限时的过拟合。$\lambda = 0.1$。

- 行为保真度验证：使用 DTW（动态时间规整）比较教师与代理的时序轨迹相似性。

## 实验关键数据

### 实验设置

- 环境：LunarLanderContinuous-v3（状态 $\mathbb{R}^8$，动作 $[-1,1]^2$）
- 教师：PPO（64单元×2层 MLP），50000步训练
- 数据：5000个状态-动作对，80/20训练/验证
- 基线：Decision Tree (16叶)、Simple MLP (32隐藏单元)
- FCS 变体：Gaussian-16, Triangular-16/8/4
- 5个随机种子(42-46)，配对 t 检验

### 主实验

| 模型 | 保真度(%) | MSE | DTW | FRAD | FSC |
|------|-----------|-----|-----|------|-----|
| Simple-MLP | 96.84±1.80 | 0.0016 | 0.55 | N/A | N/A |
| FCS-Tri-16 | **81.48±0.43** | 0.0053 | 1.05 | **0.814** | 0.933 |
| FCS-Gaus-16 | 81.38±0.64 | 0.0037 | 0.87 | 0.723 | **0.974** |
| DT-16 | 60.14±1.27 | 0.0074 | 1.32 | N/A | N/A |

### 消融实验（规则数量影响）

| 规则数 | 保真度(%) | MSE | FRAD | FSC |
|--------|-----------|------|------|-----|
| 4 | **97.83±0.5** | 0.00069 | **0.863** | 0.937 |
| 8 | 95.83±0.7 | 0.00119 | 0.801 | 0.963 |
| 16 | 81.48±0.4 | 0.00534 | 0.814 | 0.933 |

### 关键发现

1. **模糊代理桥接差距**：FCS-Tri 比 DT 保真度高 21.34 个百分点（81.48% vs 60.14%），验证模糊插值优于分段常数近似
2. **三角函数更聚焦**：三角与高斯保真度相似（p>0.05），但 FRAD 显著更高（0.814 vs 0.723，t=14.5, p<0.001）
3. **"少即是多"现象**：4规则模型反而达最高保真度（97.83%），说明 PPO 策略可被蒸馏为极简结构
4. **语义可解释规则**：提取的规则如"IF 着陆器向左漂移在高空 THEN 施加向上推力+向右纠正"具有真正的操作含义
5. **时序保真度验证**：DTW 距离仅 1.05，代理轨迹与教师紧密跟踪

## 亮点与洞察

1. **从定性到定量的可解释性评估**：FRAD/FSC/ASG 三个指标从不同角度量化解释质量，超越了"可解释"的主观断言
2. **层次化解耦设计**：将状态分区（K-Means）与动作推理（Ridge Regression）解耦，使模糊系统可扩展到连续控制
3. **"少即是多"的深刻发现**：复杂神经网络策略可被4条简单规则近乎完美地概括，暗示 PPO 策略的内在低维性
4. **实用安全审计价值**：16条规则可被文档化、审查和认证——对万参数神经网络来说不可能

## 局限与展望

1. **单一环境验证**：仅在 Lunar Lander 上测试，未扩展到更复杂的连续控制任务（机器人操作、自动驾驶）
2. **教师策略简单**：50000步训练的 PPO 策略可能本身就较简单，对更复杂策略的蒸馏效果未知
3. **状态空间维度限制**：8维状态空间较低，高维视觉输入场景的性能有待验证
4. **固定聚类数**：K-Means 需预设规则数量 $N$，缺乏自动确定最优规则数的机制
5. **无在线适应**：蒸馏后规则固定，无法在线更新以适应环境变化

## 相关工作与启发

- **VIPER (Bastani 2018)**：用决策树蒸馏 Q-network，本文改进在于用模糊系统替代分段常数
- **SHAP/LIME**：局部解释方法，本文提供全局结构性解释
- **Fuzzy Q-Learning (Glorennec)**：在训练中集成模糊逻辑，但性能弱于现代深度RL
- **TSK 系统 (Takagi-Sugeno-Kang)**：输出连续函数的模糊推理系统
- HHI指数启发的 FRAD 设计巧妙地将经济学市场集中度概念迁移到解释聚焦度

## 评分

- 新颖性: ⭐⭐⭐⭐ — 层次化 TSK + 三个新指标，框架合理创新
- 实验充分度: ⭐⭐⭐ — 单一环境验证不够，但指标设计和统计检验严谨
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，规则示例生动，公式推导完整
- 价值: ⭐⭐⭐⭐ — 将可解释RL从主观评估推向量化评估的实质性进步

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Toward a Dynamic Stackelberg Game-Theoretic Framework for Agent-Based Conversational AI Defense Against LLM Jailbreaking](../../ICLR2026/reinforcement_learning/toward_a_dynamic_stackelberg_game-theoretic_framework_for_agent-based_conversat.md)
- [\[AAAI 2026\] DRMD: Deep Reinforcement Learning for Malware Detection under Concept Drift](drmd_deep_reinforcement_learning_for_malware_detection_under_concept_drift.md)
- [\[AAAI 2026\] ChartEditor: A Reinforcement Learning Framework for Robust Chart Editing](charteditor_a_reinforcement_learning_framework_for_robust_chart_editing.md)
- [\[AAAI 2026\] MARS: A Meta-Adaptive Reinforcement Learning Framework for Risk-Aware Multi-Agent Portfolio Management](mars_a_meta-adaptive_reinforcement_learning_framework_for_risk-aware_multi-agent.md)
- [\[AAAI 2026\] ReGal: A First Look at PPO-based Legal AI for Judgment Prediction and Summarization in India](regal_a_first_look_at_ppo-based_legal_ai_for_judgment_prediction_and_summarizati.md)

</div>

<!-- RELATED:END -->
