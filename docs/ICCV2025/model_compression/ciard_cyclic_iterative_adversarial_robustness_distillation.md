---
title: >-
  [论文解读] CIARD: Cyclic Iterative Adversarial Robustness Distillation
description: >-
  [ICCV 2025][模型压缩][adversarial robustness distillation] 提出CIARD，通过对比推离损失（Contrastive Push Loss）解决双教师ARD框架中clean teacher和robust teacher的优化目标冲突，并设计迭代教师训练（ITT）策略持续更新robust teacher以防止性能退化，在CIFAR-10/100和Tiny-ImageNet上同时提升对抗鲁棒性+3.53%和干净准确率+5.87%。
tags:
  - "ICCV 2025"
  - "模型压缩"
  - "adversarial robustness distillation"
  - "知识蒸馏"
  - "对抗训练"
  - "dual-teacher"
  - "contrastive push loss"
---

# CIARD: Cyclic Iterative Adversarial Robustness Distillation

**会议**: ICCV 2025  
**arXiv**: [2509.12633](https://arxiv.org/abs/2509.12633)  
**代码**: [https://github.com/eminentgu/CIARD](https://github.com/eminentgu/CIARD)  
**机构**: NJUST, HKUST(GZ), INSAIT Sofia University, Peking University
**领域**: 模型压缩 / 对抗鲁棒蒸馏 / 知识蒸馏  
**关键词**: adversarial robustness distillation, 知识蒸馏, 对抗训练, dual-teacher, contrastive push loss, 模型压缩

## 一句话总结
提出CIARD，通过对比推离损失（Contrastive Push Loss）解决双教师ARD框架中clean teacher和robust teacher的优化目标冲突，并设计迭代教师训练（ITT）策略持续更新robust teacher以防止性能退化，在CIFAR-10/100和Tiny-ImageNet上同时提升对抗鲁棒性+3.53%和干净准确率+5.87%。

## 背景与动机

### 领域现状

**领域现状**：边缘设备部署需要高效且鲁棒的模型。知识蒸馏(KD)可压缩教师到学生，但传统KD不关心鲁棒性。对抗训练(AT)增强鲁棒性但对小模型效果有限。对抗鲁棒蒸馏(ARD)将两者结合。

双教师ARD的两个关键挑战：
1. **优化目标冲突**：clean teacher关注干净样本准确率，robust teacher关注对抗鲁棒性，学生难以调和
2. **对抗教师性能退化**：训练中学生生成的对抗样本越来越强，持续侵蚀robust teacher性能

### 解决思路

**本文目标**：如何在双教师ARD框架中同时提升学生的对抗鲁棒性和干净准确率？

## 方法详解

### 整体框架
CIARD包含固定的clean teacher + 持续更新的robust teacher + push loss机制。

### 关键设计

1. **对比推离损失（Contrastive Push Loss）**：

    - 核心洞察：不是让学生同时靠近两个教师，而是让学生**主动远离**clean teacher的错误预测
    - 当clean teacher在对抗样本上预测错误时，push loss让学生偏离错误方向
    - 学生更有效地吸收robust teacher的专业知识
    - 解耦了clean knowledge和robust knowledge的传递路径

2. **迭代教师训练（ITT）**：

    - 阶段1（预热）：冻结两个教师参数，让学生建立基本知识
    - 阶段2（迭代更新）：周期性地用当前学生生成的对抗样本重训练robust teacher
    - 持续对抗重训练确保robust teacher始终能有效防御当前最强对抗样本
    - 类似GAN中判别器-生成器交替更新

3. **训练流程**：

    - 用PGD等攻击方法对学生生成对抗样本
    - 蒸馏损失 = KL(student, robust_teacher on adv) + push_loss + CE(student, GT)
    - 周期性触发ITT更新robust teacher

## 实验关键数据

### CIFAR-10上MobileNet-V2

| 框架类型 | 方法 | Clean(%) | Robust(%) |
|---------|------|----------|-----------|
| Single-Teacher | ARD | 83.43 | 57.03 |
| Dual-Teacher | MTARD | 89.26 | 57.84 |
| Dual-Teacher | B-MTARD | 89.09↓ | 58.79↑ |
| **Dual-Teacher** | **CIARD** | **89.51↑** | **59.10↑** |

- CIARD是唯一同时提升clean和robust accuracy的方法

### 跨数据集
- CIFAR-100：平均对抗防御率+3.53%
- Tiny-ImageNet：干净准确率+5.87%
- PGD-20/AutoAttack/C&W等多种攻击下均保持优势

### 消融
- Push loss去掉后clean accuracy -1.2%，robust accuracy -0.8%
- ITT去掉后robust accuracy在训练后期明显下降
- ITT更新频率需要平衡计算成本和收益

## 亮点与洞察
- **打破精度-鲁棒性权衡**：同时提升两者对实际部署意义重大
- **Push Loss的反直觉设计**：推远清洁教师的错误预测——"负样本"思路简单有效
- **教师退化问题的识别和解决**：首次明确指出并用ITT直接解决
- **理论分析+实证验证**：清楚展示了教师退化现象

## 局限与展望
- ITT需要周期性重训练robust teacher，增加额外计算开销
- 仅在分类任务上验证
- Push loss权重系数需手动调节
- 未与最新对抗训练方法充分对比

## 相关工作与启发
- **vs. ARD/RSLAD**：单教师框架，鲁棒性提升有限且干净准确率低
- **vs. MTARD/B-MTARD**：双教师但未解决优化冲突和教师退化
- **CIARD优势**：push loss解耦 + ITT动态更新

## 相关工作与启发
- Push loss的"远离错误"策略可迁移到其他多教师蒸馏场景
- 教师模型动态退化问题在其他在线蒸馏/对抗学习中也可能存在
- 与curriculum learning有相似性：ITT本质上是让教师适应学生的"课程"难度

## 技术细节补充
- Clean teacher：标准训练的大模型（如WRN-34-10），仅在干净样本上训练
- Robust teacher：经PGD-AT训练的同架构大模型，具有对抗鲁棒性
- Student：轻量模型（MobileNet-V2/ResNet-18等）
- ITT更新周期：通常每10个epoch重训练robust teacher一次
- Push loss权重通过验证集上的鲁棒性指标自适应调整
- 对抗样本生成使用PGD-10（ε=8/255，步长2/255）
- CIARD的额外训练开销主要来自ITT的教师重训练，约增加15-20%总训练时间
- 但相比从头做adversarial training，ARD+CIARD仍然更高效

## 评分
- 新颖性: ⭐⭐⭐⭐ Push loss和ITT有效但每个单独看并非非常新颖
- 实验充分度: ⭐⭐⭐⭐ 多数据集+多攻击+消融，但学生架构多样性不足
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，Figure 1的框架对比图直观
- 价值: ⭐⭐⭐⭐ 同时提升精度和鲁棒性对实际部署有重要意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Toward Understanding Adversarial Distillation: Why Robust Teachers Fail](../../ICML2026/model_compression/toward_understanding_adversarial_distillation_why_robust_teachers_fail.md)
- [\[ICCV 2025\] Knowledge Distillation with Refined Logits](knowledge_distillation_with_refined_logits.md)
- [\[CVPR 2025\] IterIS: Iterative Inference-Solving Alignment for LoRA Merging](../../CVPR2025/model_compression/iteris_iterative_inference-solving_alignment_for_lora_merging.md)
- [\[NeurIPS 2025\] DenoiseRotator: Enhance Pruning Robustness for LLMs via Importance Concentration](../../NeurIPS2025/model_compression/denoiserotator_enhance_pruning_robustness_for_llms_via_importance_concentration.md)
- [\[ICCV 2025\] A Good Teacher Adapts Their Knowledge for Distillation](a_good_teacher_adapts_their_knowledge_for_distillation.md)

</div>

<!-- RELATED:END -->
