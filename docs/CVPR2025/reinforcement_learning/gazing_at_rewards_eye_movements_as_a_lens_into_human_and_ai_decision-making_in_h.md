---
title: >-
  [论文解读] Gazing at Rewards: Eye Movements as a Lens into Human and AI Decision-Making in Hybrid Visual Foraging
description: >-
  [CVPR 2025][强化学习][视觉搜索] 提出Visual Forager（VF）模型，通过目标特征调制、目标价值调制和ViT-based Actor-Critic决策网络模拟人类混合视觉搜索任务中的眼动策略，在归一化得分上达到72.6%（人类87.4%），扫视大小仅差0.01°（4.06° vs 人类4.05°），首次揭示目标价值和出现率如何联合影响人类搜索决策。
tags:
  - "CVPR 2025"
  - "强化学习"
  - "视觉搜索"
  - "眼动建模"
  - "注视策略"
  - "混合搜索"
---

# Gazing at Rewards: Eye Movements as a Lens into Human and AI Decision-Making in Hybrid Visual Foraging

**会议**: CVPR 2025  
**arXiv**: [2411.09176](https://arxiv.org/abs/2411.09176)  
**代码**: [https://github.com/ZhangLab-DeepNeuroCogLab/visual-forager](https://github.com/ZhangLab-DeepNeuroCogLab/visual-forager)  
**领域**: 强化学习  
**关键词**: 视觉搜索、眼动建模、强化学习、注视策略、混合搜索

## 一句话总结
提出Visual Forager（VF）模型，通过目标特征调制、目标价值调制和ViT-based Actor-Critic决策网络模拟人类混合视觉搜索任务中的眼动策略，在归一化得分上达到72.6%（人类87.4%），扫视大小仅差0.01°（4.06° vs 人类4.05°），首次揭示目标价值和出现率如何联合影响人类搜索决策。

## 研究背景与动机

**领域现状**：混合视觉搜索（同时搜索多种目标物体，各目标具有不同价值和出现概率）是人类日常活动（如超市购物、驾驶扫描）的核心能力。现有计算模型主要关注简单的固定目标搜索，忽略了目标价值对眼动策略的影响。

**现有痛点**：缺少同时考虑目标特征匹配、目标价值偏好和注视决策的统一计算模型。现有RL搜索模型不模拟人类的中心凹视觉（越远离注视点分辨率越低）。

**核心矛盾**：人类在搜索时同时优化多个目标——找到相关物品、优先高价值目标、管理注意力资源——这种多目标权衡如何形式化建模是开放问题。

**本文目标** 建立一个能模拟人类混合视觉搜索的计算模型，揭示目标价值和出现率对眼动决策的联合影响机制。

**切入角度**：将搜索建模为RL问题——智能体通过注视动作（眼跳）和点击动作（拾取目标）最大化回报，用VGG16+离心率依赖池化模拟中心凹视觉。

**核心 idea**：用目标特征调制+价值调制+RL Actor-Critic框架统一建模混合视觉搜索中的注视-拾取决策，复现人类的价值偏好和扫视模式。

## 方法详解

### 整体框架
输入搜索场景图像和目标模板，VGG16提取特征后经离心率依赖池化（模拟中心凹视觉），与目标特征相似度图和目标价值编码融合，送入ViT Actor-Critic网络输出注视位置概率图和点击动作概率。

### 关键设计

1. **目标特征调制 + 离心率依赖池化**:

    - 功能：模拟人类中心凹视觉下的目标匹配
    - 核心思路：VGG16提取搜索场景特征，经多层池化模拟离心率——越远离当前注视点池化越粗（分辨率越低）。然后计算目标模板与场景特征图的相似度，生成多目标的空间匹配热力图
    - 设计动机：人类视觉中心凹区域分辨率最高，外围急剧下降，这决定了扫视策略——需要移动目光才能看清外围物体

2. **目标价值调制**:

    - 功能：将目标价值信息融入决策
    - 核心思路：将每个目标的价值通过学习的编码器转为嵌入向量，与特征匹配图逐元素相加。高价值目标的匹配信号被放大，低价值目标被抑制
    - 设计动机：人类搜索时优先关注高价值目标——早餐时更优先找牛奶而非餐巾纸

3. **ViT Actor-Critic决策网络**:

    - 功能：输出注视位置和点击决策
    - 核心思路：ViT处理融合后的特征图，Actor头输出空间注视概率分布（决定下一步看哪里），Critic头估计状态价值。训练时用PPO优化。点击动作头单独决定是否拾取当前注视位置的物品
    - 设计动机：ViT的全局注意力能整合远距离目标信息，帮助规划高效的扫视路径

### 损失函数 / 训练策略
PPO强化学习，奖励信号为成功拾取目标的价值、步数惩罚$-0.01$。15名人类被试产生750次搜索trial、50514次注视、12851次点击作为评估基准。

## 实验关键数据

### 主实验

| 方法 | 归一化得分 (UnValEqPre) | 扫视大小 |
|------|----------------------|---------|
| 人类 | 87.4% | 4.05° |
| VF (ours) | 72.6% | 4.06° |
| FeatOnly | 52.3% | - |
| MaxVal | 44.7% | - |
| DQN | 36.5% | - |

### 关键发现
- VF的注视热点与人类高度一致——都优先注视高价值+高匹配度区域
- 人类和VF都展现了"Click Bias Ratio"——在高价值目标上有更高的点击倾向
- 去除离心率依赖池化后扫视大小显著变化，验证了中心凹视觉对扫视策略的决定性影响
- VF在OOD条件（未见目标、未见价值、未见场景大小）下保持合理的搜索行为

## 亮点与洞察
- **认知科学与CV的交叉**：首次用RL模型定量复现人类混合视觉搜索中的价值-注意力权衡机制
- **扫视大小精确匹配**：VF 4.06° vs 人类 4.05°，说明中心凹视觉建模是扫视策略的关键
- **可解释性**：通过比较人类和模型的注视/点击模式，揭示了搜索决策的计算原理

## 局限与展望
- 与人类的性能差距仍约15%，可能因为人类有更好的工作记忆避免重复搜索
- 目前假设目标价值已知，现实中价值可能需要推断
- 仅在2D静态场景中验证，真实世界的3D动态搜索更复杂

## 相关工作与启发
- **vs DQN搜索模型**: DQN做视觉搜索仅36.5% vs VF 72.6%，因为DQN不模拟中心凹视觉和价值调制
- **vs 显著性模型**: 显著性模型预测注视"在哪"但不预测"做什么"；VF统一了注视和点击决策

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次建模混合视觉搜索中价值-注意力交互，认知启发性强
- 实验充分度: ⭐⭐⭐⭐ 15名被试数据、多条件消融、OOD泛化
- 写作质量: ⭐⭐⭐⭐ 认知科学和计算模型的对比分析清晰
- 价值: ⭐⭐⭐⭐ 对主动视觉搜索和认知建模有重要价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Structured Reinforcement Learning for Combinatorial Decision-Making](../../NeurIPS2025/reinforcement_learning/structured_reinforcement_learning_for_combinatorial_decision-making.md)
- [\[ICML 2025\] Enhancing Decision-Making of Large Language Models via Actor-Critic](../../ICML2025/reinforcement_learning/enhancing_decision-making_of_large_language_models_via_actor-critic.md)
- [\[ICML 2025\] Counterfactual Effect Decomposition in Multi-Agent Sequential Decision Making](../../ICML2025/reinforcement_learning/counterfactual_effect_decomposition_in_multi-agent_sequential_decision_making.md)
- [\[ICLR 2026\] EMFuse: Energy-based Model Fusion for Decision Making](../../ICLR2026/reinforcement_learning/emfuse_energy-based_model_fusion_for_decision_making.md)
- [\[ICML 2025\] The Sample Complexity of Online Strategic Decision Making with Information Asymmetry and Knowledge Transportability](../../ICML2025/reinforcement_learning/the_sample_complexity_of_online_strategic_decision_making_with_information_asymm.md)

</div>

<!-- RELATED:END -->
