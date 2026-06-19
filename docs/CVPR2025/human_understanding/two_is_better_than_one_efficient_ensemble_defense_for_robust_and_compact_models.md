---
title: >-
  [论文解读] Two is Better than One: Efficient Ensemble Defense for Robust and Compact Models
description: >-
  [CVPR 2025][人体理解][集成防御] 提出 EED（Efficient Ensemble Defense），从单个基础模型通过不同剪枝策略（NIS/ERM/ASE/BNSF）生成多个子模型并动态集成——在 80% 稀疏度下 CIFAR-10 PGD 鲁棒准确率 55.71%（接近未压缩基线），推理加速 1.86 倍。
tags:
  - "CVPR 2025"
  - "人体理解"
  - "集成防御"
  - "模型剪枝"
  - "对抗鲁棒性"
  - "鲁棒多样性"
  - "压缩"
---

# Two is Better than One: Efficient Ensemble Defense for Robust and Compact Models

**会议**: CVPR 2025  
**arXiv**: [2504.04747](https://arxiv.org/abs/2504.04747)  
**代码**: 无  
**领域**: 人体理解 / 对抗鲁棒性  
**关键词**: 集成防御, 模型剪枝, 对抗鲁棒性, 鲁棒多样性, 压缩

## 一句话总结

提出 EED（Efficient Ensemble Defense），从单个基础模型通过不同剪枝策略（NIS/ERM/ASE/BNSF）生成多个子模型并动态集成——在 80% 稀疏度下 CIFAR-10 PGD 鲁棒准确率 55.71%（接近未压缩基线），推理加速 1.86 倍。

## 研究背景与动机

### 领域现状

**领域现状**：对抗鲁棒模型（如对抗训练的 ResNet）本身就计算密集。模型压缩（剪枝/量化）是部署鲁棒模型的关键手段，但压缩通常会显著降低鲁棒性。

**现有痛点**：（1）鲁棒性和压缩率存在严重trade-off——80% 稀疏度通常导致 PGD 下降 5-10%；（2）单模型剪枝丢失多样的鲁棒特征——对抗鲁棒性需要模型对不同攻击方向都有应对。

**核心矛盾**：单次剪枝必然丢弃某些鲁棒特征方向，但不同剪枝策略保留的方向不同——可以集成互补。

**切入角度**：从同一个预训练模型用 4 种不同的重要性度量剪枝，生成 12 个稀疏子模型（4 度量 × 不同随机种子），动态集成时用 Robust Diversity (RD) 指标选择互补子集。

**核心 idea**：多剪枝策略生成多样子模型 + RD 引导的动态集成 = 压缩且鲁棒。

### 解决思路

**本文目标**：### 关键设计

1. **多剪枝策略子模型池**：NIS（神经重要性）/ERM（经验风险）/ASE（对抗显著性）/BNSF（BN 缩放因子）各保留不同的"鲁棒方向"

2. **Robust Diversity (RD) 指标**：衡量子模型间在对抗样本上预测的多样性，优选高 RD 的子集进行集成

3. **三项损失**：$\mathcal{L}_{E}$（集成分类）+ $\mathcal{。


## 方法详解

### 关键设计

1. **多剪枝策略子模型池**：NIS（神经重要性）/ERM（经验风险）/ASE（对抗显著性）/BNSF（BN 缩放因子）各保留不同的"鲁棒方向"

2. **Robust Diversity (RD) 指标**：衡量子模型间在对抗样本上预测的多样性，优选高 RD 的子集进行集成

3. **三项损失**：$\mathcal{L}_{E}$（集成分类）+ $\mathcal{L}_{R}$（误分类正则化）+ $\mathcal{L}_{C}$（稀疏约束）

### 损失函数 / 训练策略

$\mathcal{L}_{EED} = \mathcal{L}_E + \omega\mathcal{L}_R + \gamma\mathcal{L}_C$。N=12 子模型，动态推理选最优子集。

## 实验关键数据

| 方法 | CIFAR-10 PGD | AA | 稀疏度 | 速度 |
|------|-------------|-----|--------|------|
| 未压缩基线 | ~56% | ~49% | 0% | 1× |
| 单模型剪枝 | ~45-52% | ~40% | 80% | — |
| **EED** | **55.71%** | **48.13%** | **80%** | **1.86×** |

### 消融实验
- $\mathcal{L}_C$ 对鲁棒性关键：无 $\mathcal{L}_C$ PGD 从 55.71% 降到 39.10%
- 动态推理在 50% 稀疏度效果更好（+1.81% PGD）
- RD 指标在小集成中特别重要

### 关键发现
- **4 种剪枝策略的多样性是关键**——同一策略的 4 个副本远不如 4 种不同策略
- **80% 稀疏度下几乎不损鲁棒性**——55.71% vs 未压缩 ~56%
- **加速 1.86×**——稀疏集成比密集单模型更快

## 亮点与洞察
- **压缩+鲁棒的新范式**——不是"压缩后恢复鲁棒性"，而是"通过集成获得超越单模型的鲁棒性"
- **不同剪枝策略保留不同鲁棒方向**——这一观察对理解对抗鲁棒性有启发

## 局限与展望
- 极端稀疏度（95%）性能显著下降
- 训练 12 个子模型的开销
- 动态选择子集增加推理逻辑

## 评分
- 新颖性: ⭐⭐⭐⭐ 多剪枝集成+RD指标有新意
- 实验充分度: ⭐⭐⭐⭐ CIFAR-10/SVHN/多攻击
- 写作质量: ⭐⭐⭐⭐ 清晰
- 价值: ⭐⭐⭐⭐ 为鲁棒模型部署提供了实用方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Two by Two: Learning Multi-Task Pairwise Objects Assembly for Generalizable Robot Manipulation](two_by_two_learning_multi-task_pairwise_objects_assembly_for_generalizable_robot.md)
- [\[CVPR 2025\] One2Any: One-Reference 6D Pose Estimation for Any Object](one2any_one-reference_6d_pose_estimation_for_any_object.md)
- [\[CVPR 2025\] Pose Priors from Language Models](pose_priors_from_language_models.md)
- [\[CVPR 2025\] Efficient Video Face Enhancement with Enhanced Spatial-Temporal Consistency](efficient_video_face_enhancement_with_enhanced_spatial-temporal_consistency.md)
- [\[ICCV 2025\] OpenAnimals: Revisiting Person Re-Identification for Animals Towards Better Generalization](../../ICCV2025/human_understanding/openanimals_revisiting_person_re-identification_for_animals_towards_better_gener.md)

</div>

<!-- RELATED:END -->
