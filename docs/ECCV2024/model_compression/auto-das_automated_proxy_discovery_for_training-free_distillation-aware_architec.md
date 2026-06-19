---
title: >-
  [论文解读] Auto-DAS: Automated Proxy Discovery for Training-free Distillation-aware Architecture Search
description: >-
  [ECCV 2024][模型压缩][知识蒸馏] 本文提出 Auto-DAS，一个基于进化算法的自动化代理发现框架，用于免训练的蒸馏感知架构搜索（DAS），通过在由学生内在统计量和师生交互统计量构成的搜索空间中自动发现最优代理指标，避免了手工设计代理的局限性，在 ResNet、ViT、NAS-Bench-101/201 等多种架构和搜索空间上达到了 SOTA 的排序相关性和搜索精度。
tags:
  - "ECCV 2024"
  - "模型压缩"
  - "知识蒸馏"
  - "架构搜索"
  - "免训练代理"
  - "进化算法"
  - "自动代理发现"
---

# Auto-DAS: Automated Proxy Discovery for Training-free Distillation-aware Architecture Search

**会议**: ECCV 2024  
**代码**: [https://github.com/lliai/Auto-DAS](https://github.com/lliai/Auto-DAS)  
**领域**: 模型压缩 / 神经架构搜索  
**关键词**: 知识蒸馏, 架构搜索, 免训练代理, 进化算法, 自动代理发现

## 一句话总结

本文提出 Auto-DAS，一个基于进化算法的自动化代理发现框架，用于免训练的蒸馏感知架构搜索（DAS），通过在由学生内在统计量和师生交互统计量构成的搜索空间中自动发现最优代理指标，避免了手工设计代理的局限性，在 ResNet、ViT、NAS-Bench-101/201 等多种架构和搜索空间上达到了 SOTA 的排序相关性和搜索精度。

## 研究背景与动机

**领域现状**：知识蒸馏（Knowledge Distillation, KD）是模型压缩的核心技术，通过让小型学生模型模仿大型教师模型的行为来提升学生性能。蒸馏感知架构搜索（Distillation-aware Architecture Search, DAS）进一步优化这个过程——对于给定的教师模型，搜索能从蒸馏中获得最大收益的最优学生架构。这比随意选择学生架构再蒸馏要有效得多。

**现有痛点**：传统 DAS 方法需要对每个候选学生架构进行完整的蒸馏训练来评估其性能，搜索成本极高（可能需要数百 GPU-days）。近期的 DisWOT 方法提出了基于 KD 的免训练代理（training-free proxy），通过少量前向传播就能预测蒸馏后的精度排序，大幅加速搜索。但 DisWOT 存在两个关键问题：(1) 代理指标是手工设计的，依赖研究者对蒸馏机制的理解，搜索空间非常有限；(2) 手工设计的代理在 CNN 架构上有效，但对 Vision Transformer（ViT）等新型架构泛化性差。

**核心矛盾**：免训练代理需要在"预测准确性"和"设计泛化性"之间取得平衡。手工设计可以针对特定架构优化预测准确性，但很难覆盖多样化的架构搜索空间。如何自动发现既准确又通用的代理指标？

**本文目标** (1) 如何自动发现免训练的蒸馏评估代理，避免手工设计；(2) 如何让发现的代理在 CNN 和 ViT 等不同架构类型上都有良好的泛化性。

**切入角度**：作者观察到有效的蒸馏代理通常依赖两类信息——学生模型的内在统计量（如特征分布、梯度信息等，反映学生的学习潜力）和师生交互统计量（如特征相似度、KL 散度等，反映师生兼容性）。基于此，可以将代理指标建模为这两类统计量的函数，用进化算法在函数空间中搜索最优代理。

**核心 idea**：用进化算法在学生内在统计量和师生交互统计量构成的计算图空间中自动发现最优的免训练蒸馏评估代理。

## 方法详解

### 整体框架

Auto-DAS 的流程分为三个阶段：(1) **代理搜索空间构建**——定义以学生内在统计量和师生交互统计量作为输入的计算图空间，包含各种变换操作和距离计算操作；(2) **代理搜索**——使用进化算法（EA）在搜索空间中搜索最优代理，以候选代理与真实蒸馏精度之间的排序相关性（如 Kendall's Tau）作为适应度函数；(3) **蒸馏感知架构搜索**——用发现的最优代理对候选学生架构进行免训练评估，选择最优学生架构进行正式蒸馏训练。

### 关键设计

1. **代理搜索空间（Proxy Search Space）**:

    - 功能：定义蒸馏代理的所有可能形式
    - 核心思路：搜索空间以计算图表示代理函数。输入节点有两类：(a) 学生内在统计量——包括学生各层输出的特征图、梯度、特征图的 Gram 矩阵、BatchNorm 统计量、各层权重的奇异值分布等；(b) 师生交互统计量——包括师生各层特征的余弦相似度、KL 散度、CKA（Centered Kernel Alignment）等。中间操作节点包含基本变换（log、exp、abs、normalize 等）和网络距离算子（Frobenius 范数、核范数、trace 等），这些操作受前人代理设计和 KD 损失函数的启发。最终输出节点将中间计算结果聚合为单个标量作为代理得分
    - 设计动机：将手工设计问题转化为搜索问题。搜索空间的设计兼顾了覆盖面（包含足够多的统计量和操作）和可行性（计算图的结构约束保证了输出是合法的标量值）

2. **自适应精英选择策略（Adaptive-Elite Selection Strategy）**:

    - 功能：在进化搜索过程中平衡探索（exploration）与利用（exploitation）
    - 核心思路：标准进化算法通常保留固定比例的精英个体（如 top-20%）。Auto-DAS 提出自适应调整精英比例：搜索初期增大精英群体（保留更多候选以维持多样性——exploration），搜索后期缩小精英群体（集中在高适应度区域——exploitation）。具体来说，精英比例 $p$ 随迭代次数 $t$ 按退火策略递减：$p(t) = p_{max} - (p_{max} - p_{min}) \cdot t / T$
    - 设计动机：固定精英比例容易导致早期收敛（过度 exploitation）或后期发散（过度 exploration）。自适应策略在搜索早期广泛探索代理空间，后期精细优化最有前景的代理形式

3. **多架构泛化验证（Multi-Architecture Generalization）**:

    - 功能：确保发现的代理在不同架构搜索空间上都有效
    - 核心思路：在代理搜索过程中，适应度函数不仅评估候选代理在单一架构族（如 ResNet 系列）上与蒸馏精度的排序相关性，还评估在多种架构族上的平均相关性。这使得搜索到的代理具有跨架构的预测能力。评估时，先对少量代表性架构进行蒸馏训练获得真实精度（作为搜索的"ground truth"），然后候选代理在这些架构上计算排序相关性
    - 设计动机：如果代理只在 CNN 上有效但对 ViT 无效，就严重限制了实用性。通过多架构联合优化，发现的代理天然具有更强的泛化性

### 损失函数 / 训练策略

代理搜索阶段的"损失"即候选代理与真实蒸馏精度排序之间的 Kendall's Tau（$\tau$）或 Spearman 相关系数（$\rho$），进化算法最大化此指标。正式蒸馏阶段沿用标准 KD 训练策略，损失为硬标签交叉熵 + 软标签 KL 散度的加权和。搜索效率极高——代理评估只需一次前向传播（约几秒），整个代理搜索过程可在单 GPU 上几小时内完成。

## 实验关键数据

### 主实验（排序相关性与最终精度）

| 搜索空间 | 指标 | Auto-DAS | DisWOT（之前SOTA）| 提升 |
|---------|------|---------|----------|------|
| NAS-Bench-201 (CIFAR-10) | Kendall's τ | SOTA | 次优 | 显著提升 |
| NAS-Bench-201 (CIFAR-100) | Kendall's τ | SOTA | 次优 | 显著提升 |
| NAS-Bench-101 | Kendall's τ | SOTA | 次优 | 显著提升 |
| ResNet-family | 搜索到的学生 Top-1 Acc | 最优 | 次优 | 提升 |
| ViT-family | 搜索到的学生 Top-1 Acc | 最优 | 不支持 | - |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 仅内在统计量 | τ 中等 | 缺少师生交互信息 |
| 仅交互统计量 | τ 中等 | 缺少学生自身信息 |
| 内在 + 交互 | τ 最高 | 两类信息互补 |
| 固定精英比例 | τ 略低 | 搜索不够充分 |
| 自适应精英 | τ 最高 | 更好的探索-利用平衡 |

### 关键发现

- 学生内在统计量和师生交互统计量都是有效代理的必要组成部分，两者提供互补信息
- 自动发现的代理在 ViT 搜索空间上的泛化性远超手工设计的 DisWOT 代理
- 自适应精英选择策略带来稳定的搜索提升
- 代理搜索成本极低（单 GPU 数小时），而正式蒸馏训练需要数十 GPU-days，搜索加速比超过 100x

## 亮点与洞察

- 将"设计代理"本身作为优化问题来解决是一个非常有见地的元思路（meta approach），避免了手工设计的试错过程
- 搜索空间设计充分借鉴了代理设计和 KD 损失函数的先验知识，不是盲目搜索而是有指导的搜索
- 自适应精英选择策略虽然简单，但非常有效，值得在其他进化搜索场景中使用
- Auto-DAS + AttnZero（同一团队的两个工作）构成了"自动化发现"的系列贡献，展示了进化搜索在多个 AI 设计问题上的强大能力

## 局限与展望

- 代理搜索仍依赖少量架构的蒸馏训练结果作为 ground truth，对于全新的架构搜索空间需要预先蒸馏训练获取标注
- 搜索空间中的操作类型是预定义的，可能遗漏一些新型的统计量或变换操作
- 当前主要验证了图像分类的蒸馏场景，对于检测、分割等任务的蒸馏感知搜索效果未验证
- 搜索到的代理的可解释性有限——很难直观理解为什么某个计算图组合能更好地预测蒸馏效果
- 未探索多教师场景（多个教师同时蒸馏到一个学生）下的代理搜索

## 相关工作与启发

- **DisWOT** 是本文的直接前驱工作，首次提出了 KD-based 免训练代理概念
- **ZenNAS、NASWOT** 等免训练 NAS 代理工作为代理设计提供了方法论基础
- **NAS-Bench-101/201** 提供了标准化的架构搜索评估基准
- 与 AttnZero 类似，Auto-DAS 也使用进化算法在结构化搜索空间中搜索最优组件，体现了"用搜索替代设计"的方法论
- 该方法的自动代理发现思路可推广到其他需要代理指标的场景（如数据质量评估、模型可迁移性预测）

## 评分
- 新颖性: ⭐⭐⭐⭐ 将代理设计建模为搜索问题很有创意，自适应精英选择是加分项
- 实验充分度: ⭐⭐⭐⭐ 覆盖多种架构和搜索空间，消融全面
- 写作质量: ⭐⭐⭐ 动机清晰，但搜索空间细节可更详细
- 价值: ⭐⭐⭐⭐ 对免训练 NAS 和蒸馏感知搜索领域有较好参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Category Adaptation Meets Projected Distillation in Generalized Continual Category Discovery](category_adaptation_meets_projected_distillation_in_generalized_continual_catego.md)
- [\[CVPR 2026\] TAS-LoRA: Transformer Architecture Search with Mixture-of-LoRA Experts](../../CVPR2026/model_compression/tas-lora_transformer_architecture_search_with_mixture-of-lora_experts.md)
- [\[CVPR 2025\] DKDM: Data-Free Knowledge Distillation for Diffusion Models with Any Architecture](../../CVPR2025/model_compression/dkdm_data-free_knowledge_distillation_for_diffusion_models_with_any_architecture.md)
- [\[ECCV 2024\] PaPr: Training-Free One-Step Patch Pruning with Lightweight ConvNets for Faster Inference](papr_training-free_one-step_patch_pruning_with_lightweight_convnets_for_faster_i.md)
- [\[ICCV 2025\] Time-Aware Auto White Balance in Mobile Photography](../../ICCV2025/model_compression/time-aware_auto_white_balance_in_mobile_photography.md)

</div>

<!-- RELATED:END -->
