---
title: >-
  [论文解读] ACE-G: Improving Generalization of Scene Coordinate Regression Through Query Pre-Training
description: >-
  [ICCV 2025][预训练][场景坐标回归] 将场景坐标回归器拆分为「场景无关的Transformer」和「场景特定的map code」，通过在数万场景上进行交替的mapping/query预训练，显著提升SCR方法在光照、视角变化下的泛化能力，同时保持轻量化的计算开销。 领域现状 领域现状：场景坐标回归（Scene C…
tags:
  - "ICCV 2025"
  - "预训练"
  - "场景坐标回归"
  - "视觉重定位"
  - "Transformer"
  - "泛化性"
  - "map code"
---

# ACE-G: Improving Generalization of Scene Coordinate Regression Through Query Pre-Training

**会议**: ICCV 2025  
**arXiv**: [2510.11605](https://arxiv.org/abs/2510.11605)  
**代码**: [nianticspatial/ace-g](https://github.com/nianticspatial/ace-g) (有，含预训练模型)  
**领域**: 视觉定位  
**关键词**: 场景坐标回归, 视觉重定位, Transformer预训练, 泛化性, map code  

## 一句话总结
将场景坐标回归器拆分为「场景无关的Transformer」和「场景特定的map code」，通过在数万场景上进行交替的mapping/query预训练，显著提升SCR方法在光照、视角变化下的泛化能力，同时保持轻量化的计算开销。

## 背景与动机

### 领域现状

**领域现状**：场景坐标回归（Scene Coordinate Regression, SCR）是一种基于学习的视觉重定位方法，其核心思路是训练一个网络，将图像patch特征映射到对应的3D坐标，进而通过PnP求解器估计相机位姿。代表性工作ACE仅需几分钟的场景特定训练即可达到很高精度。

**现有方法的痛点**：

### 现有痛点

**现有痛点**：SCR方法的泛化能力远不如经典的特征匹配方法（如hloc）。当查询图像的成像条件（光照、视角、物体摆放）与训练视图差异较大时，SCR模型容易失效。

### 解决思路

**解决思路**：关键原因在于训练机制本身：以往SCR的训练目标是把场景的训练视图编码到回归器的网络权重中，回归器本质上是在「过拟合」训练视图——这是设计使然（by design），而非bug。因此网络对未见过的查询条件缺乏泛化能力。

### 核心矛盾

**核心矛盾**：传统特征匹配方法通过大规模数据上预训练的特征提取器天然具有对外观变化的鲁棒性，但SCR方法缺乏这种跨场景预训练的能力。

### 解决思路

**本文目标**：如何让SCR方法在保持快速场景适配（几分钟训练）和轻量化优势的同时，获得对光照变化、大视角差异、季节/天气变化等条件的泛化能力？

核心挑战在于：传统SCR把"理解场景"和"回归坐标"两个任务耦合在同一个网络中，导致网络必须为每个场景从头训练，无法积累跨场景的泛化知识。

## 方法详解

### 整体框架
ACE-G将传统的单一场景坐标回归器拆分为两部分：
- **场景无关的Transformer回归器**：一个通用的坐标回归网络，负责"理解如何从特征推断3D坐标"的通用能力，可以在大量场景上预训练
- **场景特定的map code**：一组可学习的编码向量，存储特定场景的3D结构信息，通过反向传播优化获得

整体流程分为三个阶段：
1. **预训练阶段**：在数万个场景上预训练Transformer，学习跨场景泛化能力
2. **Mapping阶段**：对新场景，固定Transformer，只优化map code来编码该场景
3. **重定位阶段**：给定查询图像，Transformer结合map code预测2D-3D对应关系，通过PnP求解器估计位姿

### 关键设计

1. **Map Code与回归器的分离**：把场景特定信息从网络权重中剥离到外部的map code中。Map code是一组可学习的向量，通过与Transformer的交互来表示场景的3D结构。这种分离使得Transformer可以在大量场景上共享训练，积累通用的坐标回归能力。传统ACE方法中，整个网络权重都是场景特定的；而ACE-G中只有map code是场景特定的，Transformer权重是固定的。

2. **Query Pre-Training（查询预训练）**：这是本文最核心的创新。预训练过程模拟了方法的实际使用场景，在数百个场景上并行进行，交替执行两种迭代：

    - **Mapping迭代**：使用mapping图像同时优化map code和Transformer权重——让map code学会编码场景，Transformer学会如何读取map code
    - **Query迭代**：使用与mapping图像不同的query图像（包含新视角、不同光照/物体摆放等变化），但**只优化Transformer权重**，不更新map code——这强迫Transformer学会从map code中提取信息来泛化到未见过的查询条件
   
   这种交替训练策略的关键洞察是：query迭代中，场景信息已经"锁"在了map code中，Transformer必须学会利用这些固定的map code来处理变化的查询图像，从而获得泛化能力。

3. **基于不确定性的对应关系筛选**：在重定位阶段，网络不仅预测3D坐标，还估计预测的不确定性。利用不确定性来预筛选2D-3D对应关系，过滤掉低置信度的预测，再送入鲁棒PnP求解器求解位姿。

4. **DINOv2特征骨干**：使用预训练的DINOv2作为图像特征提取器，提供高质量的patch-level特征作为Transformer的输入。

### 损失函数 / 训练策略
- **训练损失**：使用重投影误差的负对数似然（Negative Log-Likelihood of Reprojection Error）作为损失函数。将预测的3D坐标投影回图像平面，计算与真实2D位置的误差。
- **预训练数据**：在数万个场景上进行预训练，使用3D监督（深度图或SfM点云提供Ground Truth 3D坐标）
- **Mapping阶段无需3D信息**：在对新场景做mapping时，虽然预训练使用了3D监督，但mapping阶段不需要3D Ground Truth，只需要带位姿标注的图像即可
- **配置选项**：提供5分钟（ace_g_5min）和25分钟（ace_g_25min）两种mapping配置
- **训练参数**：默认batch_size=40960，max_buffer_size=4000000，num_iterations=1000

## 实验关键数据

| 数据集 | 指标 | 说明 |
|--------|------|------|
| Indoor-6 | 位姿精度 | 室内大规模场景，包含显著光照和视角变化 |
| 7Scenes | 中位误差 | 经典室内重定位基准 |
| 12Scenes | 中位误差 | 室内场景 |
| Cambridge Landmarks | 位姿精度 | 室外大规模场景，季节/天气/光照变化 |
| RIO-10 | 位姿精度 | 包含物体摆放变化的室内场景 |

论文在多个挑战性重定位数据集上展示了ACE-G相比ACE和ACE-DINOv2在泛化性方面的显著提升，同时保持了有竞争力的计算开销。具体数值请参阅原文Table。

### 消融实验要点
- **Query预训练的贡献**：query pre-training是泛化性提升的核心来源。去掉query迭代（只保留mapping迭代）会导致性能显著下降。
- **Map code设计**：map code的容量和结构对场景表示质量有影响。
- **预训练场景规模**：使用更多场景进行预训练通常会带来更好的泛化效果。
- **5min vs 25min mapping**：更长的mapping时间（25分钟）可以带来更好的场景编码质量。

## 亮点与洞察
- **思路极其清晰**：将"场景记忆"和"坐标回归能力"巧妙解耦。过拟合问题不是bug而是feature——既然无法避免过拟合，就把需要过拟合的部分（map code）和需要泛化的部分（Transformer）分开。
- **预训练策略设计精妙**：通过mapping/query交替训练模拟实际使用场景，让Transformer在预训练阶段就学会泛化，而非仅仅学会过拟合。
- **实用性强**：mapping阶段不需要3D GT，只需带位姿的图像；代码和预训练模型均已开源；5分钟即可完成场景mapping。
- **可迁移的设计思路**：将"场景特定信息"外化为可学习编码的思路可以推广到其他需要场景/任务适配的视觉任务中。

## 局限与展望
- **预训练代码未开源**：虽然提供了预训练模型，但预训练代码未公布，限制了社区在新数据/新架构上的扩展。
- **预训练需要3D监督**：预训练阶段需要大量带3D标注的场景数据，数据获取成本较高。
- **Map code固定后不可更新**：mapping完成后map code固定，如果场景发生物理变化（如装修、搬家），需要重新mapping，缺乏增量更新能力。
- **计算成本方面**：虽然比特征匹配方法轻量，但Transformer + map code的组合仍比原始ACE更重。
- **户外大规模场景**：在城市级别的大规模户外定位场景中的表现有待进一步验证。

## 相关工作与启发
- **vs ACE / ACE-DINOv2**：ACE使用轻量MLP作为场景特定回归器，所有权重都是场景特定的，无法利用跨场景知识。ACE-G通过分离map code和Transformer，引入预训练，显著提升泛化性。ACE-DINOv2使用DINOv2特征但仍采用ACE的场景特定训练范式，泛化性不如ACE-G。
- **vs 特征匹配方法（hloc等）**：传统特征匹配方法通过预训练特征提取器天然具有泛化性，但需要显式地存储和检索场景特征地图，存储和计算开销大。ACE-G在保持SCR方法紧凑表示的同时，缩小了与特征匹配方法的泛化性差距。
- **vs NeRF/3DGS-based定位**：基于神经渲染的定位方法也需要场景特定训练，泛化性同样有限。ACE-G的分离思路对这类方法也有启发意义。

## 相关工作与启发
- **"记忆外化"思路的普适性**：将任务特定知识从网络权重中分离到外部可学习编码的思路，和Prompt Tuning、LoRA等参数高效微调方法有异曲同工之妙。可以考虑将类似思路应用到其他需要快速适配的视觉任务中。
- **预训练策略的创新**：交替mapping/query训练来模拟真实使用场景的预训练策略，可以推广为一种通用的"模拟部署条件的预训练"范式。
- **与3D基础模型的结合**：如果用更强大的3D基础模型替代DINOv2作为特征骨干，或用更大规模的3D数据预训练，有望进一步提升性能。

## 评分
- 新颖性: ⭐⭐⭐⭐ map code与回归器分离的架构设计和query pre-training策略都是对SCR领域的重要创新
- 实验充分度: ⭐⭐⭐⭐ 在5个数据集上进行验证，含消融实验和可视化分析
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，方法动机解释得非常好，项目页面也做得很优秀
- 价值: ⭐⭐⭐⭐ 开源代码和预训练模型，对视觉定位领域有实际推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Improving Continual Pre-training Through Seamless Data Packing](../../ACL2025/llm_pretraining/improving_continual_pre-training_through_seamless_data_packing.md)
- [\[ICCV 2025\] ConstStyle: Robust Domain Generalization with Unified Style Transformation](conststyle_robust_domain_generalization_with_unified_style_transformation.md)
- [\[ICCV 2025\] Dataset Ownership Verification for Pre-trained Masked Models](dataset_ownership_verification_for_pre-trained_masked_models.md)
- [\[ICCV 2025\] SynCity: Training-Free Generation of 3D Worlds](syncity_training-free_generation_of_3d_worlds.md)
- [\[ACL 2025\] Stealing Training Data from Large Language Models in Decentralized Training through Activation Inversion Attack](../../ACL2025/llm_pretraining/stealing_training_data_from_large_language_models_in_decentralized_training_thro.md)

</div>

<!-- RELATED:END -->
