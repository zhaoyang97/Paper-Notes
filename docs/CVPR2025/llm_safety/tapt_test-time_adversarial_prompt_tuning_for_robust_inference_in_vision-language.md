---
title: >-
  [论文解读] TAPT: Test-Time Adversarial Prompt Tuning for Robust Inference in Vision-Language Models
description: >-
  [CVPR 2025][LLM安全][对抗鲁棒性] 首个 VLM 测试时对抗防御方法，通过最小化多视图增强的熵一致性 + 对抗-干净 embedding 统计对齐来学习每个测试样本的防御性 prompt，仅需一步优化即可将 CLIP 对 AutoAttack 的鲁棒性从 0.1% 提升到 48.9%。
tags:
  - "CVPR 2025"
  - "LLM安全"
  - "对抗鲁棒性"
  - "测试时防御"
  - "提示学习"
  - "VLM安全"
  - "零样本推理"
---

# TAPT: Test-Time Adversarial Prompt Tuning for Robust Inference in Vision-Language Models

**会议**: CVPR 2025  
**arXiv**: [2411.13136](https://arxiv.org/abs/2411.13136)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 对抗鲁棒性、测试时防御、prompt tuning、VLM安全、零样本推理

## 一句话总结
首个 VLM 测试时对抗防御方法，通过最小化多视图增强的熵一致性 + 对抗-干净 embedding 统计对齐来学习每个测试样本的防御性 prompt，仅需一步优化即可将 CLIP 对 AutoAttack 的鲁棒性从 0.1% 提升到 48.9%。

## 研究背景与动机

**领域现状**：VLM 面临对抗攻击威胁——微小的像素扰动可使 CLIP 的分类准确率从 68% 降到 0%。现有防御方法需要在特定任务数据上训练对抗性 prompt（APT），但推理时不知道目标任务。

**现有痛点**：APT 需要任务特定的标注数据进行对抗训练，且对未见过的攻击类型泛化差。测试时无标签、无训练数据，如何防御？

**核心矛盾**：有效的对抗防御需要知道攻击模式，但推理时对攻击一无所知；同时不能牺牲清洁样本的性能。

**本文目标** 在不依赖任何任务特定数据的前提下，在推理时自适应地为每个测试样本学习防御性 prompt。

**切入角度**：对抗样本的不同增强视图应产生不一致的预测（高熵），而干净样本的增强视图预测一致。利用这一差异最小化多视图熵来恢复对抗样本的正确预测。

**核心 idea**：每个测试样本用一步梯度优化学习防御性双模态 prompt，通过熵最小化 + 对抗-干净统计对齐同时提升鲁棒性和保持清洁性能。

## 方法详解

### 整体框架
预计算：在 ImageNet 上用对抗训练预调 prompt 初始化 → 计算逐层的对抗/干净 embedding 统计（均值/方差）。推理时：测试样本生成 $M$ 个增强视图 → 选 top-$K$ 低熵视图 → 一步优化 $\mathcal{L}_{TAPT} = \mathcal{L}_{entropy} + \alpha \cdot \mathcal{L}_{adv} + (1-\alpha) \cdot \mathcal{L}_{clean}$ → 用优化后的 prompt 推理。

### 关键设计

1. **多视图熵最小化**:

    - 功能：利用预测一致性检测和修复对抗扰动
    - 核心思路：对测试样本生成 $M$ 个随机增强视图（裁剪/翻转），选 top-$K$ 预测熵最低的视图，最小化它们的联合预测熵。对抗扰动被增强"稀释"后，正确类别的信号在多视图中更一致
    - 设计动机：增强视图中对抗扰动的效果不确定（可能增强也可能减弱），选择低熵视图过滤掉扰动效果被放大的视图

2. **对抗-干净 embedding 对齐**:

    - 功能：将测试样本的 embedding 统计从对抗域移向干净域
    - 核心思路：预计算 ImageNet 样本在每层的对抗/干净 embedding 的均值和方差。测试时将当前样本的逐层 embedding 统计与预计算统计对齐：$\mathcal{L}_{adv}$ 拉向对抗统计（因为输入可能已被攻击），$\mathcal{L}_{clean}$ 拉向干净统计（保持原始语义）。$\alpha$ 控制两者平衡
    - 设计动机：仅用熵最小化可能不够（某些攻击能保持低熵），统计对齐从表征空间层面提供额外约束

3. **单步优化**:

    - 功能：确保推理效率
    - 核心思路：整个 prompt 优化只需一步梯度下降即可完成，因为 APT 预初始化已经提供了良好的起点
    - 设计动机：测试时防御必须实时，多步优化不可接受

### 损失函数 / 训练策略
$\mathcal{L}_{TAPT} = \mathcal{L}_{entropy} + \alpha \cdot \mathcal{L}_{adv} + (1-\alpha) \cdot \mathcal{L}_{clean}$。无任务特定训练数据需求。

## 实验关键数据

### 主实验

| 方法 | ImageNet 鲁棒Acc | 11 数据集平均鲁棒Acc |
|------|-----------------|-------------------|
| Vanilla CLIP | 0.0% | 0.1% |
| APT-V | 14.8% | 16.3% |
| **TAPT-V** | **49.2%** | **48.9%** |
| 提升 vs APT | **+34.4%** | **+32.6%** |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 仅熵最小化 | ~35% | 有效但不够 |
| +对抗对齐 | ~43% | 显著提升 |
| +干净对齐 | ~48.9% | 保持清洁性能 |
| 视图数 M=64, K=32 | 最优 | 更多视图更稳定 |

### 关键发现
- **从 0% 到 49%**：CLIP 对 AutoAttack 的鲁棒性从几乎为零提升到接近50%，无需任何任务数据
- **不牺牲清洁性能**：在干净样本上的准确率基本保持不变
- **任务无关**：用 ImageNet 预计算的统计量对 11 个不同领域的数据集都有效

## 亮点与洞察
- **首个测试时 VLM 对抗防御**填补了重要的安全空白——之前的防御方法都需要训练时数据
- **统计对齐的想法**巧妙——用领域内（ImageNet）的统计量作为锚点，将未知攻击的 embedding 拉回正常范围

## 局限与展望
- 预初始化仍依赖 ImageNet 对抗训练，对与 ImageNet 差异很大的领域可能不够
- 单步优化可能不足以应对强适应性攻击
- $\alpha$ 值需要手动设定，自适应确定可能更好

## 相关工作与启发
- **vs APT**：APT 需要任务数据做对抗训练，TAPT 在任何任务上零样本工作，且鲁棒性提升超过 30 个点
- **vs TPT**：TPT 是测试时 prompt tuning 但不针对对抗攻击。TAPT 引入对抗-干净统计对齐使其可对抗

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个测试时 VLM 对抗防御，熵+统计对齐的组合新颖
- 实验充分度: ⭐⭐⭐⭐ 11 个数据集、多种攻击方法，但缺少自适应攻击评估
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法讲解到位
- 价值: ⭐⭐⭐⭐⭐ 对 VLM 安全部署有重要实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] CleanSight: Test-Time Attention Purification for Backdoored Large Vision Language Models](test-time_attention_purification_for_backdoored_large_vision_language_models.md)
- [\[ICCV 2025\] LATTE: Collaborative Test-Time Adaptation of Vision-Language Models in Federated Learning](../../ICCV2025/llm_safety/latte_collaborative_test-time_adaptation_of_vision-language_models_in_federated_.md)
- [\[CVPR 2025\] Hyperbolic Safety-Aware Vision-Language Models](hyperbolic_safety-aware_vision-language_models.md)
- [\[NeurIPS 2025\] Buffer Layers for Test-Time Adaptation](../../NeurIPS2025/llm_safety/buffer_layers_for_test-time_adaptation.md)
- [\[ICCV 2025\] Adversarial Robust Memory-Based Continual Learner](../../ICCV2025/llm_safety/adversarial_robust_memory-based_continual_learner.md)

</div>

<!-- RELATED:END -->
