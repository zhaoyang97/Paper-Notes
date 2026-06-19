---
title: >-
  [论文解读] MARS: A Meta-Adaptive Reinforcement Learning Framework for Risk-Aware Multi-Agent Portfolio Management
description: >-
  [AAAI 2026][强化学习][投资组合管理] 提出 MARS 框架，通过异构多智能体集成（每个智能体有不同风险偏好和 Safety-Critic）与元自适应控制器（MAC）的两层架构，在动态市场条件下实现风险感知的投资组合管理，显著降低最大回撤和波动率。 深度强化学习（DRL）在自动化投资组合管理中的应用已经取得了显著…
tags:
  - "AAAI 2026"
  - "强化学习"
  - "投资组合管理"
  - "多智能体强化学习"
  - "风险管理"
  - "元学习"
  - "安全评论家"
---

# MARS: A Meta-Adaptive Reinforcement Learning Framework for Risk-Aware Multi-Agent Portfolio Management

**会议**: AAAI 2026  
**arXiv**: [2508.01173](https://arxiv.org/abs/2508.01173)  
**代码**: 无  
**领域**: 强化学习  
**关键词**: 投资组合管理, 多智能体强化学习, 风险管理, 元学习, 安全评论家

## 一句话总结

提出 MARS 框架，通过异构多智能体集成（每个智能体有不同风险偏好和 Safety-Critic）与元自适应控制器（MAC）的两层架构，在动态市场条件下实现风险感知的投资组合管理，显著降低最大回撤和波动率。

## 研究背景与动机

深度强化学习（DRL）在自动化投资组合管理中的应用已经取得了显著进展，但面临两个核心挑战：

**非平稳性问题**：金融市场的统计特性随时间变化，违反了 MDP 的平稳环境假设。在某一市场条件下（如低波动牛市）训练的模型，在市场体制转变时（如熊市）往往会灾难性失败，之前学到的模式变得过时。

**风险管理表面化**：许多 DRL 模型通过奖励整形（如 Sharpe Ratio 作为奖励信号）隐式处理风险，这种方式本质上是"反应式"——在风险发生后才施加惩罚，而不能像人类交易者那样在决策过程中主动管理风险。这导致智能体容易受到尾部风险和突发市场冲击的影响。

**两个挑战的深度交织**：无法适应市场体制变化的智能体也无法有效管理风险。现有单体模型（monolithic model）难以同时解决这两个问题。

MARS 的核心设计动机是：用多智能体的行为多样性（behavioral diversity）替代传统的显式特征工程，通过"保守型"到"激进型"智能体的动态编排来应对不同市场环境。

## 方法详解

### 整体框架

MARS 采用两层架构：
- **下层**：异构智能体集成（HAE）—— $N$ 个具有不同风险偏好的 Safety-Critic 智能体
- **上层**：元自适应控制器（MAC）—— 动态分配各智能体权重

在每个时间步 $t$，市场状态 $s_t$（包含当前持仓、现金余额、技术指标）被同时输入 MAC 和 HAE。MAC 输出权重向量 $w_t$，HAE 中每个智能体输出动作 $a_t^i$，加权聚合后经过风险管理覆盖层产生最终执行动作 $A_t'$。

问题被建模为 MDP $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$：
- **状态空间**：现金余额 + 每个资产的持仓量和特征向量（价格 + MACD、RSI、CCI、ADX 四个技术指标）
- **动作空间**：连续向量 $A_t \in [-1,1]^D$，表示每个资产的目标配置变化
- **奖励函数**：$R_t = \frac{V_{t+1} - V_t}{V_t} - C_t - \rho_t$，其中 $\rho_t = w_{vol} \cdot \sigma_{30d} + w_{dd} \cdot DD_{30d}$ 是基于30天滚动波动率和最大回撤的风险惩罚

### 关键设计

1. **异构智能体集成（HAE）**：集成 $\mathcal{E} = \{\mathcal{A}_1, ..., \mathcal{A}_N\}$ 由 $N$ 个独立的 Safety-Critic 智能体组成。每个智能体 $\mathcal{A}_i$ 由唯一的风险参数对 $(\theta_i, \lambda_i)$ 定义——$\theta_i$ 是风险容忍阈值，$\lambda_i$ 是风险厌恶惩罚权重。这些智能体的风险偏好从"超保守"到"高激进"均匀分布，创造了一个行为多样的"专家"池。每个智能体基于 DDPG 架构，包含三个网络：

    - **Actor 网络**：策略梯度包含条件安全惩罚（CSP）：$\nabla_{\phi_i} J(\phi_i) \approx \mathbb{E}[\nabla_{\phi_i} Q_{\psi_i}(s_t, \pi_{\phi_i}(s_t)) - \lambda_i \cdot \nabla_{\phi_i} \text{ReLU}(C_{\xi_i}(s_t, \pi_{\phi_i}(s_t)) - \theta_i)]$。CSP 项仅在预测风险超过特定阈值 $\theta_i$ 时才惩罚策略。
    - **Critic 网络**：标准 TD 误差训练，估计状态-动作价值函数。
    - **Safety-Critic 网络**：预测动作的外部风险分数。训练目标是环境风险函数 $\mathcal{C}_{env}$，该函数从三个维度综合计算 $[0,1]$ 的风险分数——投资组合集中度（30%）、杠杆率（30%）和模拟波动率（40%），通过 MSE 损失训练。

2. **元自适应控制器（MAC）**：MAC 是一个神经网络 $M_\omega$，学习元策略 $\pi_\omega(w_t | s_t)$，根据当前市场状态动态分配智能体权重。输出权重分布通过 softmax 生成：$w_t = \text{softmax}(M_\omega(s_t))$。最终动作为加权平均：$A_t = \sum_{i=1}^{N} w_t^i \cdot \pi_{\phi_i}(s_t)$。

   MAC 的训练目标是最大化风险调整收益函数：$\mathcal{L}(\omega) = -\left(\frac{\mathbb{E}[\bar{Q}_t]}{\text{Std}(\bar{Q}_t) + \epsilon} - \lambda_{meta} \cdot \mathbb{E}[\bar{C}_t]\right)$，其中第一项类似 Sharpe 比率（收益/风险比），第二项惩罚集成的预测风险。这使 MAC 学会偏好"高收益、低风险且稳定"的智能体组合。

3. **风险管理覆盖层**：作为最终安全网，执行基于规则的约束——单一资产仓位不超过总组合价值的 20%、维持现金缓冲、禁止卖空。任何违规动作被调整为合规动作 $A_t'$。注意 Safety-Critic 是训练时模块，部署时风险由 MAC 动态加权和规则层管理。

### 损失函数 / 训练策略

- 所有网络为三层 MLP（256-128-64，ReLU 激活）
- $N=10$ 个智能体，$\theta_i$ 范围 0.10-0.55，$\lambda_i$ 范围 1.0-5.5
- 奖励函数参数：$w_{vol}=0.5$，$w_{dd}=2.0$，$\gamma=0.99$
- MAC 风险惩罚参数：$\lambda_{meta}=0.5$
- 每个资产的特征向量维度 $K=5$（价格 + 4个技术指标）
- 随机种子固定为 42

训练流程（Algorithm 1）：每个 episode 中，HAE 各智能体提出动作 → MAC 计算权重 → 加权聚合 → 风险覆盖 → 执行 → 收集经验 → 更新各智能体的 Actor/Critic/Safety-Critic → 定期更新 MAC。

## 实验关键数据

### 主实验

实验在 DJI（道琼斯，50只美股）和 HSI（恒生指数，50只港股）上进行，设置两个测试期：2022年（熊市）和2024年（牛市）。

| 环境 | 模型 | CR% | AR% | SR | AVol% | MDD% |
|------|------|-----|-----|-----|-------|------|
| DJI 2024 | **MARS** | **29.50** | **31.19** | **2.84** | 10.99 | **-5.39** |
| DJI 2024 | MARS-Static | 17.10 | 17.17 | 1.71 | 10.04 | -6.79 |
| DJI 2024 | DeepTrader | 13.30 | 14.01 | 1.18 | 11.92 | -6.84 |
| DJI 2024 | HRPM | 19.11 | 20.16 | 0.99 | 20.43 | -7.90 |
| DJI 2024 | DJI Index | 15.36 | 16.19 | 1.41 | 11.51 | -6.06 |
| DJI 2022 | **MARS** | **-0.86** | -0.93 | -0.05 | 19.83 | **-16.77** |
| DJI 2022 | DeepTrader | -10.70 | -11.43 | -0.46 | 25.07 | -21.32 |
| DJI 2022 | AlphaStock | -36.37 | -38.42 | -1.03 | 37.35 | -46.17 |
| HSI 2024 | **MARS** | 16.24 | 17.84 | **1.49** | 12.00 | -7.38 |
| HSI 2022 | **MARS** | **-14.50** | -14.88 | -0.66 | **22.56** | **-32.72** |
| HSI 2022 | DeepTrader | -26.69 | -27.34 | -0.86 | 31.93 | -48.02 |

关键结果：MARS 在 DJI 2022 和 2024 的 Sharpe Ratio 相对最佳 baseline 分别提升 70.6% 和 101.4%。

### 消融实验

| 配置 | CR% | SR | MDD% | 说明 |
|------|-----|-----|------|------|
| MARS（完整） | 29.50 | 2.84 | -5.39 | 完整框架 |
| MARS-Static | 17.10 | 1.71 | -6.79 | 移除 MAC 动态加权（均匀权重） |
| MARS-Homo | 22.21 | 1.85 | -7.81 | 移除智能体异构性（相同风险参数） |
| MARS-Div5 | 12.02 | 1.08 | -6.19 | 仅5个智能体 |
| MARS-Div15 | 19.70 | 1.67 | -7.26 | 15个智能体（收益递减） |

### 关键发现

1. **熊市资本保全**：2022年 MARS 亏损仅 0.86%，而 AlphaStock 亏损 36.37%，回撤 -46.17%
2. **自适应策略切换**：2022年 MAC 权重波动大、频繁切换保守/进取；2024年权重稳定、保守-激进负相关从 -0.788 加深到 -0.968
3. **集成规模**：10个智能体最优，5个不够多样，15个收益递减
4. **MAC 是关键**：移除 MAC 后 Sharpe Ratio 从 2.84 降到 1.71，说明动态编排是核心

## 亮点与洞察

- **架构创新**：将风险管理从"奖励惩罚"的被动方式，提升为"多智能体行为多样性 + 元控制器动态编排"的主动方式
- **Safety-Critic 的独特定位**：训练时用来塑造每个智能体的风险偏好，部署时不使用——风险完全由 MAC 和规则层管理
- **三维风险信号**：环境风险函数融合投资组合集中度、杠杆率和模拟波动率，比简单价格惩罚更全面
- **MAC 训练目标设计巧妙**：类似 Sharpe 比率的公式让 MAC 同时追求高收益和低风险

## 局限与展望

1. 仅验证了股票市场，未涉及其他资产类别（如加密货币、外汇、商品期货）
2. Safety-Critic 的环境风险函数权重（40%-30%-30%）基于敏感性分析固定，可能不适用于所有市场
3. 交易成本和滑点的建模较简化
4. 仅选择 50 只股票，扩展到更大资产池的效果未知
5. 缺乏与更新的 DRL 方法（如基于 Transformer 的策略、扩散模型等）的比较

## 相关工作与启发

- **DeepTrader**（Wang et al. 2021b）：双模块架构（市场条件嵌入 + 回撤惩罚），MARS 的直接对比对象
- **HRPM**（Wang et al. 2021a）：层次化 RL 框架，高层设战略配置、低层优化执行成本
- **EarnHFT**（Qin et al. 2024）：三层层次化高频交易，用元控制器选择最佳专家智能体——与 MARS 的 MAC 思路最接近
- **MAPS**（Lee et al. 2020）：多智能体协作 + 分散化惩罚鼓励策略多样性——启发了 MARS 的异构设计

## 评分

- 新颖性: ⭐⭐⭐⭐ （两层架构设计新颖，Safety-Critic 与元控制器的结合有创意）
- 实验充分度: ⭐⭐⭐⭐ （消融完整，多市场多时期验证，但 baseline 略少）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，公式规范）
- 价值: ⭐⭐⭐⭐ （在金融 RL 领域有实际应用价值）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] MARS: Multi-Agent Adaptive Reasoning with Socratic Guidance for Automated Prompt Optimization](mars_multi-agent_adaptive_reasoning_with_socratic_guidance_f.md)
- [\[AAAI 2026\] Explaining Decentralized Multi-Agent Reinforcement Learning Policies](explaining_decentralized_multi-agent_reinforcement_learning_policies.md)
- [\[AAAI 2026\] Think, Speak, Decide: Language-Augmented Multi-Agent Reinforcement Learning for Economic Decision-Making](think_speak_decide_language-augmented_multi-agent_reinforcement_learning_for_eco.md)
- [\[ICML 2026\] Interaction-Breaking Adversarial Learning Framework for Robust Multi-Agent Reinforcement Learning](../../ICML2026/reinforcement_learning/interaction-breaking_adversarial_learning_framework_for_robust_multi-agent_reinf.md)
- [\[ICLR 2026\] BoreaRL: A Multi-Objective Reinforcement Learning Environment for Climate-Adaptive Boreal Forest Management](../../ICLR2026/reinforcement_learning/borearl_a_multi-objective_reinforcement_learning_environment_for_climate-adaptiv.md)

</div>

<!-- RELATED:END -->
