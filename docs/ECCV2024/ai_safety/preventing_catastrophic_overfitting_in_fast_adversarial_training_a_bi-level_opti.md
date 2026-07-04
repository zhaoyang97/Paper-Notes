---
title: >-
  [论文解读] Preventing Catastrophic Overfitting in Fast Adversarial Training: A Bi-level Optimization Perspective
description: >-
  [ECCV2024][AI安全][fast adversarial training] 从双层优化视角分析快速对抗训练中灾难性过拟合的成因，提出 FGSM-PCO 方法，通过自适应融合历史与当前对抗样本并配合定制正则化损失，有效防止并纠正内层优化崩溃。 对抗训练（Adversarial Training, AT）是抵御对抗样…
tags:
  - "ECCV2024"
  - "AI安全"
  - "fast adversarial training"
  - "catastrophic overfitting"
  - "bilevel optimization"
  - "FGSM"
  - "adversarial examples"
---

# Preventing Catastrophic Overfitting in Fast Adversarial Training: A Bi-level Optimization Perspective

**会议**: ECCV2024  
**arXiv**: [2407.12443](https://arxiv.org/abs/2407.12443)  
**代码**: [HandingWangXDGroup/FGSM-PCO](https://github.com/HandingWangXDGroup/FGSM-PCO)  
**领域**: AI安全  
**关键词**: fast adversarial training, catastrophic overfitting, bilevel optimization, FGSM, adversarial examples

## 一句话总结
从双层优化视角分析快速对抗训练中灾难性过拟合的成因，提出 FGSM-PCO 方法，通过自适应融合历史与当前对抗样本并配合定制正则化损失，有效防止并纠正内层优化崩溃。

## 背景与动机
对抗训练（Adversarial Training, AT）是抵御对抗样本的有效手段，可建模为双层优化问题：内层最大化扰动以生成对抗样本，外层最小化模型在对抗样本上的损失。标准 PGD-AT 采用多步攻击求解内层问题，计算代价高。快速对抗训练（FAT）使用单步 FGSM 替代 PGD，大幅降低训练开销，但面临严重的**灾难性过拟合**问题——模型在 FGSM 攻击下准确率飙升，而在多步 PGD 攻击下准确率骤降至 0%。

现有 FAT 方法（如 FGSM-RS 随机初始化、FGSM-GA 梯度对齐正则、FGSM-MEP 动量扰动初始化）虽能延缓过拟合发生，但在复杂任务（如 Tiny-ImageNet）或大参数模型（如 WideResNet34-10）上**仍无法完全避免**灾难性过拟合。更关键的是，一旦过拟合发生，这些方法缺乏**纠正机制**来恢复有效训练。

## 核心问题
1. **灾难性过拟合的根因**：FGSM 的单步大步长攻击与交替优化机制耦合，极易导致内层优化崩溃——生成的对抗样本对当前模型无效，进而导致整个双层优化失效
2. **现有方法缺陷**：已有 FAT 方法只能推迟过拟合发生，无法在过拟合**已发生**后将训练拉回正轨
3. **目标**：设计一种 FAT 框架，既能预防灾难性过拟合，又能在过拟合趋势出现时自动纠正

## 方法详解

### 整体框架：FGSM-PCO
FGSM-PCO（Preventing Catastrophic Overfitting）的核心思想是：不直接使用当前 FGSM 生成的对抗样本训练，而是将**历史对抗样本**与**当前对抗样本**自适应融合后再用于训练。

### 1. 对抗样本生成与融合
给定上一轮对抗样本 $\boldsymbol{x}_{t-1}^*$，当前阶段的处理流程为：

- **计算梯度方向**：$\mathbf{g}_t = \text{sign}(\nabla_{\mathbf{x}} \mathcal{L}(f_\theta(\boldsymbol{x}_{t-1}^*), \mathbf{y}))$
- **生成放大对抗样本**：$\boldsymbol{x}_{am}^* = \boldsymbol{x}_{t-1}^* + \gamma \epsilon \mathbf{g}_t$，其中 $\gamma$ 为放大因子（默认 $\gamma=2$），补偿融合带来的扰动衰减
- **自适应融合**：$\boldsymbol{x}_{train} = \lambda_t \boldsymbol{x}_{t-1}^* + (1-\lambda_t) \boldsymbol{x}_{am}^*$

### 2. 自适应融合比例
融合因子 $\lambda_t$ 由模型对当前对抗样本的分类置信度决定：

$$\lambda_t = f_\theta^k(\boldsymbol{x}_{t-1}^* + \boldsymbol{\delta}_t)$$

其中 $k$ 是真实标签的索引。关键直觉：

- **正常训练时**：对抗样本有效，模型对真实类的置信度低 → $\lambda_t$ 小 → 更多使用当前对抗样本
- **过拟合趋势出现时**：对抗样本失效，模型对真实类置信度高 → $\lambda_t$ 大 → 更多保留历史对抗样本，避免依赖无效的当前对抗样本

这种机制确保了在过拟合发生时，训练样本自动偏向历史有效样本，从而纠正训练方向。

### 3. 定制正则化损失
为配合融合框架，提出 PCO 损失函数：

$$\mathcal{L}_{PCO} = \mathcal{L}_{CE}(f_\theta(\boldsymbol{x}_{train}), \mathbf{y}) + \beta[\mathcal{L}_1(f_\theta(\boldsymbol{x}_t^*), f_\theta(\boldsymbol{x}_{t-1}^*)) - \mathcal{L}_1(f_\theta(\boldsymbol{x}_{train}), f_\theta(\boldsymbol{x}_t^*))]$$

- 第一项：融合样本上的交叉熵损失，保证模型在对抗样本上的判别能力
- 第二项正则化：要求融合后样本的预测与前后两阶段对抗样本的预测保持一致，防止内层优化崩溃。$\beta=10$ 为默认权重

### 4. 纠正能力
与其他 FAT 方法不同，FGSM-PCO 具备**纠正已发生过拟合**的能力。实验显示：当 FGSM-AT 在第 16 轮、FGSM-MEP 在第 50 轮发生过拟合后切换到 FGSM-PCO，模型均能恢复有效训练。

## 实验关键数据

### CIFAR-10 + ResNet18

| 方法 | Clean Acc | PGD10 | PGD50 | AA | 训练时间 |
|------|-----------|-------|-------|-----|---------|
| PGD-AT (best) | 82.57 | 53.19 | 52.21 | 48.77 | 199 min |
| TRADES (best) | 82.03 | 54.06 | 53.16 | 49.47 | 241 min |
| FGSM-MEP (best) | 81.72 | 55.13 | 54.29 | 48.23 | 57 min |
| **FGSM-PCO (best)** | **82.05** | **56.32** | **55.67** | 48.04 | 60 min |

- PGD10 准确率比 PGD-AT 高 **3.1%**，比 FGSM-MEP 高 **1.2%**
- 最后一轮 checkpoint 与最佳 checkpoint 结果一致，证明无过拟合

### CIFAR-100 + WideResNet34-10

| 方法 | Clean Acc | PGD10 | 训练时间 |
|------|-----------|-------|---------|
| PGD-AT | 62.45 | 32.36 | 1397 min |
| FGSM-MEP | 43.42 | 23.77 | 407 min |
| **FGSM-PCO** | **65.80** | **29.80** | 421 min |

- 10 次独立重复实验中，FGSM-PCO **0/10 次**发生过拟合，而 FGSM-AT/FGSM-RS 均为 10/10，FGSM-MEP 为 6/10

### Tiny-ImageNet + PreActResNet18

| 方法 | Clean Acc | PGD10 | PGD50 |
|------|-----------|-------|-------|
| PGD-AT (best) | 33.99 | 15.35 | 15.16 |
| FGSM-MEP (best) | 31.70 | 16.81 | 16.69 |
| **FGSM-PCO (best)** | **34.96** | **18.17** | **17.99** |

### 消融实验（CIFAR-10 + ResNet18）
- 仅融合（无自适应、无正则）：PGD10 = 39.91%，发生过拟合
- 融合 + 正则损失：PGD10 = 54.27%，显著提升
- 融合 + 自适应：PGD10 = 50.67%
- 全部组件：PGD10 = **56.12%**，三个组件缺一不可

## 亮点
1. **从双层优化理论角度**清晰解释了灾难性过拟合的本质——内层优化崩溃导致的连锁反应
2. **自适应融合机制**设计精巧：利用模型自身的分类置信度作为信号，无需额外超参数调节
3. **首个具备纠正能力的 FAT 方法**：不仅能预防过拟合，还能在过拟合发生后恢复训练
4. 在 WideResNet34-10 + CIFAR-100 这一公认困难设置下，10/10 次完全避免过拟合
5. 比 FGSM-MEP 仅多 3 分钟训练时间，但节省 1/3 显存

## 局限与展望
1. **训练开销仍高于最简单的 FAT**：需要存储上一轮对抗样本并做额外前向传播，比 FGSM-RS 慢约 50%
2. **放大因子 $\gamma$ 固定为 2**：未探索动态调整策略，不同数据集/模型可能需要不同设置
3. **AA（AutoAttack）指标上未超越 TRADES**：在最强攻击下的鲁棒性仍有差距（48.04% vs 49.47%）
4. **仅验证了 $l_\infty$ 范数约束**：未讨论对 $l_2$ 等其他范数约束的适用性
5. **数据集规模有限**：最大仅验证到 Tiny-ImageNet（64×64），未在 ImageNet 全尺寸上测试

## 与相关工作的对比

| 方法 | 核心策略 | 能否防止过拟合 | 能否纠正过拟合 | 额外开销 |
|------|----------|:---:|:---:|----------|
| FGSM-RS | 随机初始化 + 大步长 | 部分 | 否 | 无 |
| FGSM-GA | 梯度对齐正则 | 部分 | 否 | 中等 |
| FGSM-MEP | 动量扰动初始化 | 大部分 | 否 | 显存高 |
| **FGSM-PCO** | 自适应融合 + 正则 | **完全** | **是** | 显存低 |

与 FGSM-MEP 的关键区别：MEP 通过累积梯度动量初始化扰动，降低内层优化失败风险但无法纠正；PCO 通过融合历史样本直接参与训练，当过拟合趋势出现时自动增加历史样本比例来纠正方向。

## 启发与关联
1. **自适应融合思想**可推广到其他容易崩溃的训练场景（如 GAN 训练、强化学习中的策略崩溃）
2. 利用模型**自身置信度**作为训练状态的监控信号，是一种轻量且通用的诊断机制
3. **历史样本复用**的思路与 experience replay（经验回放）异曲同工，可探索在对抗训练中引入更丰富的历史信息
4. 正则化项要求"融合前后预测一致"，与知识蒸馏中的 consistency regularization 有联系

## 评分
- 新颖性: ⭐⭐⭐⭐ — 双层优化视角的分析有深度，自适应融合机制简洁有效
- 实验充分度: ⭐⭐⭐⭐ — 三个数据集三个模型，含消融、敏感性分析和纠正能力验证
- 写作质量: ⭐⭐⭐⭐ — 问题动机清晰，方法推导完整
- 价值: ⭐⭐⭐⭐ — FAT 领域的实用改进，解决了长期困扰的灾难性过拟合问题

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Mitigating Error Amplification in Fast Adversarial Training](../../CVPR2026/ai_safety/mitigating_error_amplification_in_fast_adversarial_training.md)
- [\[ECCV 2024\] Bi-TTA: Bidirectional Test-Time Adapter for Remote Physiological Measurement](bi-tta_bidirectional_test-time_adapter_for_remote_physiological_measurement.md)
- [\[ICML 2026\] SORA: Free Second-Order Attacks in Fast Adversarial Training](../../ICML2026/ai_safety/sora_free_second-order_attacks_in_fast_adversarial_training.md)
- [\[CVPR 2026\] A Unified Perspective on Adversarial Membership Manipulation in Vision Models](../../CVPR2026/ai_safety/a_unified_perspective_on_adversarial_membership_manipulation_in_vision_models.md)
- [\[ICML 2026\] PRPO: Paragraph-level Policy Optimization for Vision-Language Deepfake Detection](../../ICML2026/ai_safety/prpo_paragraph-level_policy_optimization_for_vision-language_deepfake_detection.md)

</div>

<!-- RELATED:END -->
