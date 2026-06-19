---
title: >-
  [论文解读] Predictive Preference Learning from Human Interventions
description: >-
  [NeurIPS 2025 Spotlight][自动驾驶][交互式模仿学习] PPL通过轨迹预测模型预见智能体未来状态，并将人类单次干预信号"扩展"到预测的未来状态上构建对比偏好数据，结合行为克隆和偏好优化双损失训练策略，大幅减少了人类干预次数和示范数据需求。 交互式模仿学习(IIL)让人类专家在训练过程中实时监控并纠正智…
tags:
  - "NeurIPS 2025 Spotlight"
  - "自动驾驶"
  - "交互式模仿学习"
  - "偏好学习"
  - "人类干预"
  - "轨迹预测"
  - "DPO"
---

# Predictive Preference Learning from Human Interventions

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2510.01545](https://arxiv.org/abs/2510.01545)  
**代码**: [https://metadriverse.github.io/ppl](https://metadriverse.github.io/ppl)  
**领域**: 自动驾驶 / 模仿学习  
**关键词**: 交互式模仿学习, 偏好学习, 人类干预, 轨迹预测, DPO

## 一句话总结
PPL通过轨迹预测模型预见智能体未来状态，并将人类单次干预信号"扩展"到预测的未来状态上构建对比偏好数据，结合行为克隆和偏好优化双损失训练策略，大幅减少了人类干预次数和示范数据需求。

## 研究背景与动机

交互式模仿学习(IIL)让人类专家在训练过程中实时监控并纠正智能体的错误行为，相比离线模仿学习能有效缓解分布偏移问题。然而现有IIL方法面临三大痛点：

**仅纠正当前状态**：HG-DAgger等方法只在专家干预的当前状态进行行为克隆，但智能体可能在后续 $t+1, \cdots, t+L$ 步重复类似错误，需要专家反复纠正

**极高的认知负担**：专家需要持续监控训练全过程，预判智能体未来轨迹，并在安全关键状态及时干预

**数据效率低**：只利用了干预时刻的示范数据，未利用干预隐含的偏好信息——即"专家选择干预"本身就意味着智能体的动作是不可取的

核心idea：利用轨迹预测模型可视化智能体未来状态，帮助专家提前预判；同时将每次干预"展开"(bootstrap)到未来L步，构建偏好数据集，利用对比偏好优化(CPO)传播专家纠正意图到安全关键区域。

## 方法详解

### 整体框架

PPL工作流程：(1) 智能体在每个决策点提出动作 $a_n$ → (2) 轨迹预测模型 $f(s, a_n, H)$ 生成H步未来状态并可视化 → (3) 人类专家观察未来轨迹，决定是否干预 → (4) 若干预，产生的示范数据存入 $\mathcal{D}_h$，同时在预测的L步未来状态上构建偏好对存入 $\mathcal{D}_{pref}$ → (5) 双损失训练策略优化策略网络

### 关键设计

1. **轨迹预测与可视化**：

    - 给定当前状态s和智能体动作 $a_n$，轨迹预测模型生成 $f(s, a_n, H) = (s, \tilde{s}_1, \cdots, \tilde{s}_H)$
    - 预测结果实时可视化给专家（如驾驶场景中显示红色预测轨迹），当预测轨迹指向碰撞等危险时，专家提前干预
    - 实现方式：可使用模拟器rollout（1000fps）或运动学自行车模型（3000fps，无需模拟器）
    - 设计动机：减少专家的认知负担——专家无需自己预判未来，系统帮助预判

2. **偏好展开(Preference Bootstrapping)**：

    - 当专家在状态s干预时，为预测的前L步状态构建偏好三元组：$(\tilde{s}_i, a^+ = a_h, a^- = a_n)$，$i = 1, \cdots, L$
    - 偏好horizon $L \leq H$ 控制展开长度，是关键超参数
    - 核心假设：专家在s的纠正动作 $a_h$ 在近未来状态 $\tilde{s}_i$ 中仍优于智能体动作 $a_n$
    - 设计动机：单次干预生成L个偏好样本，大幅提升数据效率；将纠正信号传播到智能体可能探索的安全关键区域

3. **双损失训练策略**：

    - 行为克隆损失：$\mathcal{L}_{BC} = -\mathbb{E}_{(s,a_h) \sim \mathcal{D}_h}[\log \pi_\theta(a_h|s)]$
    - 对比偏好优化(CPO)损失：$\mathcal{L}_{pref} = -\mathbb{E}_{(s,a^+,a^-) \sim \mathcal{D}_{pref}}[\log \sigma(\beta \log \pi_\theta(a^+|s) - \beta \log \pi_\theta(a^-|s))]$
    - 总损失：$\mathcal{L} = \mathcal{L}_{pref} + \mathcal{L}_{BC}$
    - BC损失正则化策略不偏离专家示范太远，CPO损失利用偏好信号抑制危险行为

### 损失函数 / 训练策略
- 训练参数：$\beta = 0.1$，MetaDrive中 $L = 4$，Nut Assembly中 $L = 6$，$H = 10$
- CPO不需要参考策略（相比DPO的优势），无需预训练
- 每H=10步更新一次预测轨迹（约1秒），专家通过Xbox手柄或键盘干预

## 实验关键数据

### 主实验

**MetaDrive (真人实验, 10K步)**

| 方法 | 人类数据量 | 成功率 | 回报 | 路线完成率 |
|------|------|------|----------|------|
| BC | 20K(离线) | 0.0 | 53.5 | 0.16 |
| PVP | 4.9K(0.49) | 0.46 | 267.3 | 0.71 |
| Ensemble-DAgger | 3.8K(0.38) | 0.36 | 233.8 | 0.70 |
| **PPL(Ours)** | **2.9K(0.29)** | **0.76** | **324.8** | **0.90** |
| Human Expert | 20K | 0.95 | 349.2 | 0.98 |

PPL以最少的人类数据(2.9K)达到最高成功率(76%)，仅需12分钟在单卡RTX4080上完成。

### 消融实验

| 配置 | 成功率 | 路线完成 | 说明 |
|------|---------|------|------|
| PPL完整版 | 0.81 | 0.92 | 最优 |
| 仅模仿a+ | 0.36 | 0.65 | 偏好对比信息很重要 |
| 随机a+ | 0.45 | 0.73 | a+质量影响大 |
| 随机a- | 0.38 | 0.69 | a-质量也很重要 |
| 仅BC损失 | 0.42 | 0.72 | 缺少偏好优化 |
| 仅CPO损失 | 0.04 | 0.31 | 缺少行为克隆的正则化 |
| PPL+DPO | 0.80 | 0.91 | DPO需要参考策略,效果相当 |
| 规则轨迹预测 | 0.78 | 0.91 | 不依赖模拟器rollout |

### 关键发现
- **偏好horizon L的选择很关键**：L=4在MetaDrive中最优；L过小覆盖不够（高 $\delta_{dist}$），L过大标签质量下降（高 $\delta_{pref}$）
- 轨迹预测器即使有噪声（$\epsilon \leq 0.25$），PPL仍优于所有基线
- PPL产生更平滑的控制序列和更符合人类偏好的轨迹
- 理论上界(Theorem 4.1)：$J(\pi_h) - J(\pi_n) = O(\sqrt{\epsilon + \delta_{pref}} + \delta_{dist})$，L需平衡两项误差
- 在RoboSuite的桌面擦拭和螺母装配任务中同样有效，展示了方法的通用性

## 亮点与洞察
- 将RLHF/DPO思路巧妙地迁移到实时控制问题中——通过轨迹预测构建偏好数据，无需人工标注偏好对
- 偏好展开是一个简洁有效的idea：一次干预生成多个训练样本，显著提升样本效率
- 轨迹可视化同时服务于人类（减少认知负担）和算法（创建偏好数据），双赢设计
- 理论分析严谨，给出了偏好horizon选择的原则性指导

## 局限与展望
- 假设专家总是知道最优纠正动作且准确执行，但真实人类示范可能次优或不一致
- 所有实验在仿真环境中进行，真实机器人和物理环境的效果未知
- 偏好展开的核心假设（$a_h$在未来状态仍优于$a_n$）随L增大越来越弱
- 需要可用的轨迹预测模型或物理模型，在某些场景中可能不易获得

## 相关工作与启发
- HG-DAgger、PVP等IIL方法是直接前作，PPL弥补了它们只纠正当前状态的不足
- DPO/CPO等偏好优化方法从LLM对齐领域迁移而来
- 轨迹预测在自动驾驶中有大量成熟工作可供利用
- 偏好展开的思路可推广到其他需要稀疏人类反馈的场景

## 评分
- 新颖性: ⭐⭐⭐⭐ 轨迹预测+偏好展开+IIL的组合是全新的，创意出色
- 实验充分度: ⭐⭐⭐⭐⭐ 驾驶+机器人操作双场景，真人+模拟专家，消融和鲁棒性分析全面
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法直观，理论分析完整
- 价值: ⭐⭐⭐⭐ 为IIL提供了实用高效的方法，桥接了偏好RL和模仿学习

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Certified Human Trajectory Prediction](../../CVPR2025/autonomous_driving/certified_human_trajectory_prediction.md)
- [\[ECCV 2024\] Progressive Pretext Task Learning for Human Trajectory Prediction](../../ECCV2024/autonomous_driving/progressive_pretext_task_learning_for_human_trajectory_prediction.md)
- [\[NeurIPS 2025\] Towards Physics-Informed Spatial Intelligence with Human Priors: An Autonomous Driving Perspective](towards_physics-informed_spatial_intelligence_with_human_priors_an_autonomous_dr.md)
- [\[NeurIPS 2025\] Towards Predicting Any Human Trajectory in Context](towards_predicting_any_human_trajectory_in_context.md)
- [\[CVPR 2026\] Towards Balanced Multi-Modal Learning in 3D Human Pose Estimation](../../CVPR2026/autonomous_driving/towards_balanced_multi-modal_learning_in_3d_human_pose_estimation.md)

</div>

<!-- RELATED:END -->
