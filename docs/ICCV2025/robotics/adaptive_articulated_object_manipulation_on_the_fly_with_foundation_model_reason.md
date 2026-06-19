---
title: >-
  [论文解读] Adaptive Articulated Object Manipulation On The Fly with Foundation Model Reasoning and Part Grounding
description: >-
  [ICCV 2025][机器人][铰接物体操作] 本文提出 AdaRPG 框架，利用基础视觉-语言模型对铰接物体进行零件级分割和可操作性推理，并借助 GPT-4o 生成高层控制代码以自适应调度原子操作技能，在仿真和真实环境中实现了跨类别零样本泛化操作。 铰接物体（如瓶子、门、保险柜、微波炉等）由多个可动部件和关节组成…
tags:
  - "ICCV 2025"
  - "机器人"
  - "铰接物体操作"
  - "基础模型"
  - "零件分割"
  - "可操作性预测"
  - "自适应策略"
---

# Adaptive Articulated Object Manipulation On The Fly with Foundation Model Reasoning and Part Grounding

**会议**: ICCV 2025  
**arXiv**: [2507.18276](https://arxiv.org/abs/2507.18276)  
**代码**: 无  
**领域**: 机器人操作  
**关键词**: 铰接物体操作, 基础模型, 零件分割, 可操作性预测, 自适应策略

## 一句话总结
本文提出 AdaRPG 框架，利用基础视觉-语言模型对铰接物体进行零件级分割和可操作性推理，并借助 GPT-4o 生成高层控制代码以自适应调度原子操作技能，在仿真和真实环境中实现了跨类别零样本泛化操作。

## 研究背景与动机
铰接物体（如瓶子、门、保险柜、微波炉等）由多个可动部件和关节组成，其操作是机器人领域的核心挑战。现实场景中，机器人需要处理复杂的自适应操作，比如保险柜需要先解锁才能打开，但锁的状态对机器人不可见，需反复尝试并根据反馈调整策略。

现有研究面临两大瓶颈：（1）真实铰接物体的几何多样性极大，导致视觉感知和可操作性学习难以泛化到新类别；（2）不同类别物体的操作机制（关节约束、锁定机制等）差异显著，使得操作策略难以直接迁移。这两者共同阻碍了构建统一的跨类别自适应操作策略。

核心矛盾在于：整个物体级别的几何变化太大难以学习，但不同类别物体之间的局部零件（如把手、旋钮、按钮）却共享相似的几何特征。本文正是基于这一关键洞察，提出以"零件"作为中间表征来提升可操作性预测的泛化能力。

切入角度：融合基础模型在视觉感知和语言推理方面的强大泛化能力，构建一个从零件分割到可操作性推理再到高层策略生成的完整框架。核心idea：零件级可操作性建模 + 基础模型推理 = 跨类别自适应操作。

## 方法详解

### 整体框架
AdaRPG 包含三个核心组件：（1）零件级可操作性数据集构建与学习；（2）基础模型引导的零件定位与分割；（3）GPT-4o 驱动的高层控制代码生成与原子技能执行。整个流程为：相机获取 RGB-D 图像 → GPT-4o 识别零件描述 → GroundingDINO 定位 → SAM 精细分割 → 可操作性模型推理 → 原子技能执行 → GPT-4o 生成自适应控制代码。

### 关键设计

1. **零件级可操作性数据集与学习**:

    - 功能：从 PartNet-Mobility 数据集提取 11 类物体的功能性零件（把手、按钮、旋钮等），构建零件级点云-可操作性标注数据集
    - 核心思路：给定零件点云 $O_i$，利用自动算法标注高可操作性区域。高可操作性点位于零件中心附近，基于质心估计和包围盒中心精化确定。训练使用 PointNet++ 作为特征提取器，输出 per-point 可操作性分数 $V(O_i, p_i) \in [0,1]$，损失函数为二元交叉熵：$L_V = \text{BCELoss}(r_i, V(O_i, p_i))$
    - 设计动机：零件级建模相比整体物体级建模，几何变异性大幅降低，使得可操作性预测在新类别物体上具有更好的泛化能力。数据集仅包含零件点云而不含完整物体，训练时不暴露完整形状

2. **基础模型引导的零件定位与分割**:

    - 功能：利用三个冻结的基础模型级联工作，实现对未见物体零件的精准定位和分割
    - 核心思路：RGB 图像 → GPT-4o 生成详细零件描述（不超过三句话，区分功能性部件如固定手柄 vs 可动手柄）→ 描述作为 prompt 输入 GroundingDINO 生成边界框 → 边界框作为 prompt 输入 SAM 进行精细分割 → 反投影到深度图获得 3D 零件点云
    - 设计动机：详细文本描述比单词 prompt 显著提升 GroundingDINO 的检测精度。三个基础模型均以训练冻结方式使用，无需额外微调，唯一可训练的组件是可操作性模型

3. **原子技能函数与高层代码生成**:

    - 功能：定义六种基于末端执行器坐标系的原子操作函数（z 轴推/拉、自转，y 轴平移），并利用 GPT-4o 生成 Python 控制代码
    - 核心思路：通过可操作性分数确定末端执行器位姿（平均高分点作为平移位置，点云表面法线确定方向）。GPT-4o 根据自然语言输入生成结构化 Python 代码，形成自适应控制循环（如先抓取 → 反复旋转 → 概率性尝试拉取 → 成功则持续拉取直至完成）
    - 设计动机：将操作抽象为原子函数使得 GPT-4o 只需进行高层推理而非底层控制，所有移动通过阻抗控制保证平滑稳定，实时角度偏差修正增强鲁棒性

### 损失函数 / 训练策略
可操作性模型使用二元交叉熵损失训练。阻抗控制用于所有运动的动态力-柔顺性调节。GPT-4o 仅使用单一通用 prompt 即可为所有类别生成控制代码，无需针对特定物体编写脚本。

## 实验关键数据

### 主实验：仿真操作成功率

| 方法 | Bottle | Pen | PC | CM | Window | Door | Lamp |
|------|--------|-----|-----|-----|--------|------|------|
| SAGE | 0.21 | 0.40 | 0.00 | 0.30 | 0.38 | 0.39 | 0.40 |
| CoPa | 0.58 | 0.47 | 0.17 | 0.40 | 0.08 | 0.39 | 0.20 |
| AdaManip | 0.46 | 0.53 | 0.50 | 0.60 | 0.46 | 0.44 | 0.30 |
| **AdaRPG** | **0.84** | **0.73** | **1.00** | **0.80** | **0.84** | **0.78** | **0.70** |

### 消融实验

| 配置 | Bottle | Pen | PC | CM | Window | Door | Lamp | 说明 |
|------|--------|-----|-----|-----|--------|------|------|------|
| AdaRPG (完整) | 0.84 | 0.73 | 1.00 | 0.80 | 0.84 | 0.78 | 0.70 | 完整模型 |
| w/o prompt | 0.11 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.20 | 无GPT描述引导，几乎全部失败 |
| w/o affordance | 0.78 | 0.58 | 0.83 | 0.60 | 0.57 | 0.63 | 0.40 | 使用均匀可操作性分数，平均下降15% |

### 关键发现
- AdaRPG 在所有 7 个类别上均超越所有基线方法，在 Pressure Cooker 上达到 100% 成功率
- 零件分割的 IoU 在所有类别上均超过 80%（Bottle 高达 0.99）
- 零件级可操作性 F1 分数远超整体物体级方法 VAT-MART（平均 0.74 vs 0.27）
- 真实世界实验成功率甚至优于仿真结果，因为基础模型本身是在真实数据上训练的

## 亮点与洞察
- **零件作为中间表征** 是一个优雅的解耦设计：零件的局部几何在不同类别间更相似，自然支持泛化
- **三阶段冻结基础模型级联**（GPT-4o → GroundingDINO → SAM）实现零训练的高质量零件分割
- **真实世界性能优于仿真**，揭示了基础模型在弥合 sim-to-real 差距中的潜力

## 局限与展望
- 原子技能设计较为简单（6种），面对更复杂的工具使用场景可能不足
- GPT-4o 的推理延迟可能限制实时性
- 可操作性模型仍需在特定数据集上训练，未实现完全零样本

## 补充实验：真实世界结果

| 方法 | Bottle | Pressure Cooker | Microwave | Lamp |
|------|--------|-----------------|-----------|------|
| SAGE | 5/10 | 6/10 | 3/10 | 6/10 |
| CoPa | 4/10 | 2/10 | 3/10 | 1/10 |
| AdaManip | 8/10 | 5/10 | 7/10 | 5/10 |
| **AdaRPG** | **9/10** | **10/10** | **9/10** | **8/10** |

真实世界性能全面优于仿真结果，归因于所有基础模型都是在真实数据上预训练的，仿真环境引入的域差距反而略微削弱了性能。这一发现表明，在机器人系统中引入更多基础模型可以更好地应对真实世界挑战。

## 相关工作与启发
- **vs AdaManip**: AdaManip 使用模仿学习训练扩散策略，需要大量专家演示，且跨类别泛化能力有限；AdaRPG 借助基础模型实现零样本跨类别泛化
- **vs SAGE**: SAGE 依赖 GAPartNet 进行零件姿态估计，姿态估计精度不如可操作性表征，且错误会在流水线中传播
- **vs CoPa**: CoPa 使用 GraspNet 进行通用抓取，但通用抓取无法对齐铰接物体的功能性零件，导致早期执行失败

## 评分
- 新颖性: ⭐⭐⭐⭐ 零件级可操作性 + 基础模型组合的框架设计巧妙且有效
- 实验充分度: ⭐⭐⭐⭐ 7类仿真+4类真实物体，消融完整，基线对比全面
- 写作质量: ⭐⭐⭐⭐ 图文配合好，流水线清晰
- 价值: ⭐⭐⭐⭐ 为机器人自适应操作提供了实用的零样本泛化方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Physically Ground Commonsense Knowledge for Articulated Object Manipulation with Analytic Concepts](../../CVPR2026/robotics/physically_ground_commonsense_knowledge_for_articulated_object_manipulation_with.md)
- [\[ICLR 2026\] From Spatial to Actions: Grounding Vision-Language-Action Model in Spatial Foundation Priors](../../ICLR2026/robotics/from_spatial_to_actions_grounding_vision-language-action_model_in_spatial_founda.md)
- [\[ICCV 2025\] Selective Contrastive Learning for Weakly Supervised Affordance Grounding](selective_contrastive_learning_for_weakly_supervised_affordance_grounding.md)
- [\[CVPR 2025\] Magma: A Foundation Model for Multimodal AI Agents](../../CVPR2025/robotics/magma_a_foundation_model_for_multimodal_ai_agents.md)
- [\[ICCV 2025\] Self-supervised Learning of Hybrid Part-aware 3D Representations of 2D Gaussians and Superquadrics](self-supervised_learning_of_hybrid_part-aware_3d_representations_of_2d_gaussians.md)

</div>

<!-- RELATED:END -->
