---
title: >-
  [论文解读] Improving Transferable Targeted Attacks with Feature Tuning Mixup
description: >-
  [CVPR 2025][对抗攻击] 提出 FTM（Feature Tuning Mixup）通过在代理模型的特征空间中混合优化的攻击专用扰动和随机干净扰动来提升有目标对抗攻击的迁移性，使用动量式随机更新策略保持计算效率，14 个黑盒模型上平均成功率从 74.6% 提升到 77.4%。 领域现状：有目标迁移攻击要求在代理模型上…
tags:
  - "CVPR 2025"
  - "对抗攻击"
  - "迁移攻击"
  - "特征扰动"
  - "Mixup"
  - "黑盒攻击"
---

# Improving Transferable Targeted Attacks with Feature Tuning Mixup

**会议**: CVPR 2025  
**arXiv**: [2411.15553](https://arxiv.org/abs/2411.15553)  
**代码**: [https://github.com/uhiu/feature-tuning-mixup](https://github.com/uhiu/feature-tuning-mixup)  
**领域**: 其他  
**关键词**: 对抗攻击、迁移攻击、特征扰动、Mixup、黑盒攻击

## 一句话总结
提出 FTM（Feature Tuning Mixup）通过在代理模型的特征空间中混合优化的攻击专用扰动和随机干净扰动来提升有目标对抗攻击的迁移性，使用动量式随机更新策略保持计算效率，14 个黑盒模型上平均成功率从 74.6% 提升到 77.4%。

## 研究背景与动机

**领域现状**：有目标迁移攻击要求在代理模型上生成的对抗样本能在未知黑盒模型上攻击到指定目标类别。现有方法通过输入增强（DI、SI）或特征混合（CFM）提升迁移性。

**现有痛点**：Clean Feature Mixup（CFM）在特征空间中混入干净图像特征来增强多样性，但仅使用随机干净特征——没有针对攻击目标进行优化，限制了扰动多样性的提升。

**核心矛盾**：特征扰动需要足够多样化以提升迁移性，但又需要与攻击目标相关——随机干净特征和攻击优化特征各有所长。

**本文目标** 在特征空间中引入攻击优化的可学习扰动，与干净特征混合使用，进一步提升迁移性。

**切入角度**：设计可学习的特征扰动（element-wise 加到中间层输出），通过 min-max 目标优化——扰动最大化对抗损失，对抗图像最小化损失。动量式随机更新避免额外的前向/反向计算。

**核心 idea**：在代理模型特征空间中混合攻击优化的可学习扰动和随机干净扰动，通过 min-max 优化+动量随机更新实现零额外开销的迁移性提升。

## 方法详解

### 整体框架
在每次迭代中：当前对抗图像前向传播 → 中间层 feature 加上可学习扰动 δ → δ 通过 min-max 优化（对抗图像 min 损失，δ max 损失）→ 动量式随机更新（只更新随机选择的层子集，用先前迭代的 δ 作 momentum 初始化）→ 最终将 FTM 和 CFM 混合使用。

### 关键设计

1. **可学习攻击扰动**:

    - 功能：产生攻击目标相关的特征多样性
    - 核心思路：在中间层输出上 element-wise 加入可学习扰动 δ。δ 通过 min-max 目标优化——内层最大化对抗损失（使扰动最"破坏性"），外层最小化对抗损失（使对抗图像适应这种破坏性扰动）
    - 设计动机：CFM 的随机干净扰动与攻击目标无关，攻击优化扰动更有针对性地增加多样性

2. **动量式随机更新**:

    - 功能：零额外前向/反向开销地更新多层扰动
    - 核心思路：每次迭代只随机选择一个层子集（概率 p）更新 δ，其余层用上一迭代的 δ（momentum）。扰动和对抗图像的梯度在同一次前向/反向中联合计算
    - 设计动机：如果每层每步都独立更新 δ 需要多次前向/反向——随机选择+momentum 以零额外开销近似全更新。消融显示同时更新所有层反而更差

3. **FTM-E（集成变体）**:

    - 功能：进一步提升迁移性
    - 核心思路：使用代理模型的多个副本，每个副本独立应用 FTM，然后集成攻击。2 个副本产生最佳性价比（79.5% vs 单副本 77.4%）
    - 设计动机：不同副本的随机扰动路径不同，集成增加了攻击的鲁棒性

### 损失函数 / 训练策略
对抗图像用 PGD 迭代优化。Min-max 目标：$\min_x \max_\delta \mathcal{L}(f_\theta(x + \delta), y_{target})$。无额外前向/反向开销。

## 实验关键数据

### 主实验

| 方法 | 耗时 | RN-50→14 模型平均成功率 |
|------|------|----------------------|
| RDI | 1.23s | 49.4% |
| ODI | 4.38s | 69.2% |
| RDI-CFM | 1.39s | 74.6% |
| **RDI-FTM** | **1.54s** | **77.4%** |
| **RDI-FTM-E** | **2.92s** | **79.5%** |

### 消融实验

| 配置 | 成功率 |
|------|-------|
| CFM alone | 74.6% |
| FTM alone | 优于 CFM |
| **FTM + CFM** | **77.4%** |
| 同时更新所有层 | 降低（过拟合代理） |
| 随机层选择 + momentum | **最优** |

### 关键发现
- **攻击优化扰动 > 随机干净扰动**：FTM 单独即超 CFM，组合更好
- **随机层选择关键**：全层更新导致过拟合代理模型，随机选择增加隐式正则
- **零额外计算**：1.54s vs CFM 1.39s，开销增加仅 11%

## 亮点与洞察
- **Min-max 特征扰动**在特征空间做对抗训练提升了攻击鲁棒性——本质上让对抗图像对特征空间中的扰动更"免疫"
- **动量随机更新**是一个优雅的效率解决方案——避免了多层更新的计算爆炸

## 局限与展望
- 仅在图像分类攻击上验证，目标检测/分割攻击未测
- 代理模型和目标模型差异大时迁移性仍有限
- 有目标攻击比无目标更具挑战，成功率仍有提升空间

## 相关工作与启发
- **vs CFM**：仅用随机干净扰动。FTM 引入攻击优化扰动进一步提升 2.8%
- **vs ODI**：需要多次前向（4.38s）。FTM 仅 1.54s 且效果更好

## 评分
- 新颖性: ⭐⭐⭐⭐ 特征空间 min-max 扰动的想法有创意
- 实验充分度: ⭐⭐⭐⭐ 14 个黑盒模型、多代理模型、效率分析
- 写作质量: ⭐⭐⭐⭐ 方法和消融分析清楚
- 价值: ⭐⭐⭐ 对对抗攻击研究有贡献但应用场景有限

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Towards Million-Scale Adversarial Robustness Evaluation With Stronger Individual Attacks](towards_million-scale_adversarial_robustness_evaluation_with_stronger_individual.md)
- [\[CVPR 2025\] Tuning the Frequencies: Robust Training for Sinusoidal Neural Networks](tuning_the_frequencies_robust_training_for_sinusoidal_neural_networks.md)
- [\[CVPR 2025\] Feature Selection for Latent Factor Models](feature_selection_for_latent_factor_models.md)
- [\[CVPR 2025\] Improving Accuracy and Calibration via Differentiated Deep Mutual Learning](improving_accuracy_and_calibration_via_differentiated_deep_mutual_learning.md)
- [\[ACL 2025\] TARGA: Targeted Synthetic Data Generation for Practical Reasoning over Structured Data](../../ACL2025/others/targa_targeted_synthetic_data_generation_for_practical_reasoning_over_structured.md)

</div>

<!-- RELATED:END -->
