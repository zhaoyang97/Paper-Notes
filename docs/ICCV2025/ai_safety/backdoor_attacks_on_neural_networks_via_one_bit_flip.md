---
title: >-
  [论文解读] Backdoor Attacks on Neural Networks via One-Bit Flip
description: >-
  [ICCV 2025][AI安全][backdoor attack] 提出SOLEFLIP，首个在量化模型上仅翻转一个比特位即可注入后门的推理阶段攻击方法，通过高效算法识别可利用的权重和比特位，并生成对应触发器，在CIFAR-10/SVHN/ImageNet上实现平均98.9%的攻击成功率且对正常精度零影响。
tags:
  - "ICCV 2025"
  - "AI安全"
  - "backdoor attack"
  - "bit-flip attack"
  - "Rowhammer"
  - "quantized model"
  - "one-bit flip"
---

# Backdoor Attacks on Neural Networks via One-Bit Flip

**会议**: ICCV 2025  
**代码**: 无  
**领域**: AI安全 / 后门攻击  
**关键词**: backdoor attack, bit-flip attack, Rowhammer, quantized model, one-bit flip

## 一句话总结

提出SOLEFLIP，首个在量化模型上仅翻转一个比特位即可注入后门的推理阶段攻击方法，通过高效算法识别可利用的权重和比特位，并生成对应触发器，在CIFAR-10/SVHN/ImageNet上实现平均98.9%的攻击成功率且对正常精度零影响。

## 研究背景与动机

后门攻击是DNN面临的隐蔽安全威胁——模型在正常输入上表现正常，但遇到特定触发器时产生攻击者期望的输出。传统后门攻击需要操纵训练数据或过程，但训练环境可被保护。近期研究引入了更实际的推理阶段威胁模型——利用Rowhammer等内存故障注入技术翻转模型权重比特位。然而现有方法需要翻转大量比特位（10-100个），在实践中极具挑战且常不可行。量化模型的定点整数表示限制了比特翻转的效果范围（不像全精度模型中翻转指数位可造成巨大变化），使单比特攻击更加困难。

## 方法详解

### 整体框架

SOLEFLIP分三步：(1) 可利用权重识别——设计算法找到一个适合后门注入的权重及其特定比特位；(2) 触发器生成——给定选定权重，生成能以大值激活该权重对应神经元的触发器；(3) 后门激活——翻转目标比特位后，含触发器的输入驱动模型产生攻击者期望输出。前两步离线完成，第三步在线执行。

### 关键设计

1. **可利用权重识别算法**: 在量化模型的所有层中搜索权重，评估每个权重的每个比特位被翻转后的影响。关注高位（如符号位或高值位）翻转能产生大权重变化的情况。通过评估翻转后权重值的变化量、该权重连接的神经元对输出的影响链，以及目标类别的可达性，高效筛选出最具"可利用性"的(权重, 比特位)对。与ONEFLIP（针对全精度模型）不同，量化模型的权重被约束在有界范围[−1,1]内，翻转效果受限，需要更精巧的选择策略。

2. **触发器生成**: 给定选定的权重w和比特位，优化一个小patch触发器使得：当触发器加入输入图像时，翻转后的权重w'激活对应神经元产生异常大的输出，进而通过网络传播到目标类别。优化目标是最大化触发器在翻转后权重下的激活值。

3. **单比特翻转实现后门**: 仅翻转一个比特位即完成后门注入。Rowhammer攻击已被证明可以精确翻转特定的单个比特位，使攻击在实际中可行。与需要翻转数十个比特位的现有方法相比，SOLEFLIP大幅降低了攻击门槛。

### 损失函数 / 训练策略

触发器生成使用梯度优化。攻击本身不涉及模型训练——是对部署后模型的推理阶段攻击。仅需少量良性样本用于评估。

## 实验关键数据

### 主实验

| 数据集 | 模型 | 翻转比特数 | 攻击成功率 | 良性精度下降 |
|--------|------|----------|----------|------------|
| CIFAR-10 | ResNet | 1 | 99.9% | 0.0% |
| SVHN | VGG | 1 | ~99% | 0.0% |
| ImageNet | ViT | 1 | ~98% | 0.0% |
| **平均** | - | **1** | **98.9%** | **0.0%** |

相比之下，TBT需要~100个翻转，ProFlip需要~10个翻转。

### 消融实验

- 不同量化精度(4/8比特)：均有效
- 不同模型架构(CNN/ViT)：普适
- 对后门防御的鲁棒性：SOLEFLIP对现有防御方法表现出强抗性
- 触发器大小和位置的影响

### 关键发现

- 仅翻转一个比特位就足以在量化模型中植入成功的后门
- 量化模型虽比全精度模型对参数攻击更鲁棒，但仍存在单比特漏洞
- SOLEFLIP对后门防御方法具有较强抗性
- 揭示了DNN部署中一个严重的安全威胁

## 亮点与洞察

- 将后门攻击的实际可行性推到极限——单比特翻转
- 从全精度到量化模型的扩展解决了实际中更常见的场景
- 攻击效果惊人——98.9%成功率且零良性精度损失
- 对安全研究社区的重要预警

## 局限与展望

- 攻击假设白盒访问（知道模型结构和权重），限制了部分场景
- Rowhammer攻击本身的成功率受DRAM硬件特性限制
- 针对此类攻击的新防御方法亟待研究
- 仅在分类任务上验证，对检测、分割等任务的适用性未知

## 相关工作与启发

- TBT、ProFlip、HPT等推理阶段后门攻击是直接对比方法
- ONEFLIP针对全精度模型的单比特攻击是最近相关工作
- Rowhammer攻击的硬件安全研究为本文提供了威胁模型基础
- 防御侧需要关注量化模型的比特级脆弱性

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首个量化模型单比特后门攻击
- 技术深度: ⭐⭐⭐⭐ — 权重识别算法设计精妙
- 实验充分性: ⭐⭐⭐⭐ — 多数据集、多架构、防御评估
- 写作质量: ⭐⭐⭐⭐ — 工作流程图清晰，对比分析到位
- 实用价值: ⭐⭐⭐⭐ — 对DNN安全部署有重要警示意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Robust Spiking Neural Networks Against Adversarial Attacks](../../ICLR2026/ai_safety/robust_spiking_neural_networks_against_adversarial_attacks.md)
- [\[ICML 2026\] Singular Bayesian Neural Networks](../../ICML2026/ai_safety/singular_bayesian_neural_networks.md)
- [\[ICML 2025\] Adversarial Inception Backdoor Attacks against Reinforcement Learning](../../ICML2025/ai_safety/adversarial_inception_backdoor_attacks_against_reinforcement_learning.md)
- [\[ICCV 2025\] Mind the Cost of Scaffold! Benign Clients May Even Become Accomplices of Backdoor Attack](mind_the_cost_of_scaffold_benign_clients_may_even_become_accomplices_of_backdoor.md)
- [\[ICML 2025\] Solving Probabilistic Verification Problems of Neural Networks Using Branch and Bound](../../ICML2025/ai_safety/solving_probabilistic_verification_problems_of_neural_networks_using_branch_and_.md)

</div>

<!-- RELATED:END -->
