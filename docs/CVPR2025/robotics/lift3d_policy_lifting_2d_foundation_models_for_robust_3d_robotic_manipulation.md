---
title: >-
  [论文解读] Lift3D Foundation Policy: Lifting 2D Large-Scale Pretrained Models for Robust 3D Robotic Manipulation
description: >-
  [CVPR 2025][机器人][3D机器人操控] Lift3D提出了一个两阶段框架，先通过任务感知MAE重建深度信息增强2D基础模型的隐式3D感知能力，再通过将3D点云投影到虚拟平面建立与2D位置嵌入的映射关系来直接让2D模型编码点云数据，在MetaWorld上平均成功率达83.9%（超越前SOTA DP3的65.3%达18.6个百分点）。
tags:
  - "CVPR 2025"
  - "机器人"
  - "3D机器人操控"
  - "2D基础模型"
  - "点云编码"
  - "位置嵌入提升"
  - "隐式3D表示"
---

# Lift3D Foundation Policy: Lifting 2D Large-Scale Pretrained Models for Robust 3D Robotic Manipulation

**会议**: CVPR 2025  
**arXiv**: [2411.18623](https://arxiv.org/abs/2411.18623)  
**代码**: [https://lift3d-web.github.io](https://lift3d-web.github.io)  
**领域**: 3D视觉/机器人操控  
**关键词**: 3D机器人操控, 2D基础模型, 点云编码, 位置嵌入提升, 隐式3D表示

## 一句话总结

Lift3D提出了一个两阶段框架，先通过任务感知MAE重建深度信息增强2D基础模型的隐式3D感知能力，再通过将3D点云投影到虚拟平面建立与2D位置嵌入的映射关系来直接让2D模型编码点云数据，在MetaWorld上平均成功率达83.9%（超越前SOTA DP3的65.3%达18.6个百分点）。

## 研究背景与动机

**领域现状**：视觉机器人操控策略需要理解3D场景才能完成复杂操控任务。目前方法分两类：一类直接编码点云数据（PointNet++、PointNext等），但缺乏大规模机器人3D数据和基础模型，泛化能力受限；另一类做模态转换——将3D点云投影为多视图图像输入2D模型，或将2D特征提升到3D空间，但模态转换不可避免地丢失空间信息。

**现有痛点**：直接训练3D策略模型面临数据不足和计算成本高的问题；而模态转换方法在3D→2D或2D→3D的过程中会丢失空间几何信息，影响机器人对3D空间关系的理解。现有2D基础模型（CLIP、DINOV2）虽然有强大的预训练知识，但不具备3D空间感知能力。

**核心矛盾**：大规模预训练知识存在于2D模型中，而机器人操控需要3D空间理解——如何在不丢失空间信息的前提下将2D预训练知识用于3D操控？

**本文目标** (1) 如何增强2D基础模型的3D空间感知能力；(2) 如何让2D模型直接编码3D点云数据而不做模态转换；(3) 构建一个利用大规模预训练知识的鲁棒3D操控策略。

**切入角度**：作者观察到Transformer的位置嵌入（PE）是连接2D和3D的关键桥梁——如果能建立3D点与2D PE之间的映射关系，就能让2D模型直接理解3D输入。先用MAE重建深度赋予模型隐式3D感知，再用PE映射实现显式3D编码。

**核心 idea**：通过将3D点投影到多个虚拟平面获取预训练2D位置嵌入的映射，让2D基础模型无需模态转换即可直接编码点云数据进行操控策略学习。

## 方法详解

### 整体框架

两阶段训练框架。**Stage 1（隐式3D表示）**：使用任务感知MAE对2D基础模型进行自监督微调。利用CLIP提取任务文本描述的注意力图指导masking策略，遮挡任务相关区域后让模型重建深度信息，同时通过蒸馏损失保留预训练知识。**Stage 2（显式3D表示）**：将点云通过轻量3D tokenizer编码为128个token，投影到6个虚拟平面建立与原始2D PE的映射，平均多个平面的PE得到统一3D位置指示器(PE_3D)，加在3D token上送入2D基础模型编码，最后通过3层MLP策略头预测7-DoF末端执行器位姿。

### 关键设计

1. **任务感知MAE (Task-aware Masked Autoencoder)**:

    - 功能：增强2D基础模型的隐式3D空间感知能力
    - 核心思路：利用CLIP根据任务文本描述（如"Robot arm take the red bowl"）生成图像注意力图，用阈值 $\theta=0.5$ 过滤出任务相关的affordance区域进行重点遮挡（保持75%总遮挡率）。重建目标是深度信息而非RGB，因为消融实验证明深度重建比RGB重建对操控任务更有价值（+6 vs +1）。同时添加蒸馏损失约束可见token输出与原始预训练模型保持一致，防止灾难性遗忘
    - 设计动机：传统MAE随机遮挡可能大量遮盖无关背景，效率低；聚焦任务affordance区域可以更高效地学习操控相关的几何信息

2. **2D模型提升策略 (2D Model-Lifting Strategy)**:

    - 功能：让2D基础模型直接编码3D点云数据
    - 核心思路：将输入点云通过轻量3D tokenizer（FPS下采样+KNN局部聚合+线性层）编码为128个3D token。然后将每个3D token的坐标投影到6个虚拟平面（立方体投影），得到对应的6个2D坐标。通过这些坐标查找原始预训练2D PE，取平均得到统一的3D位置嵌入：$PE_{3D} = \frac{1}{n}\sum_{j=1}^{n} PE_{2D}(C_{2D}^{ij})$。这样投影过程仅用于建立位置映射，不用于构造模型输入，避免了模态转换
    - 设计动机：引入全新的3D PE会与预训练2D模型产生语义不匹配；复用原始预训练PE能最大限度保留大规模预训练知识

3. **蒸馏+冻结训练策略**:

    - 功能：在微调过程中保留2D基础模型的预训练知识
    - 核心思路：Stage 1用L1蒸馏损失约束微调后模型的可见token输出与原始模型一致。Stage 2冻结2D基础模型参数，只更新3D tokenizer、注入的adapter和策略头。消融显示蒸馏带来+8%成功率提升
    - 设计动机：避免在小规模机器人数据上微调时灾难性遗忘大规模预训练知识

### 损失函数 / 训练策略

Stage 1：$\mathcal{L}_{\text{implicit}} = \|2D_e(x_{\text{vis}}) - 2D_e^{\text{pre}}(x_{\text{vis}})\|_1 + \|2D_d(\cdot) - D_{\text{target}}\|_1$（蒸馏+深度重建）。Stage 2：$\mathcal{L}_{\text{explicit}} = \text{MSE}(T) + (1 - \cos(R)) + \text{BCE}(G)$（平移MSE+旋转余弦+夹爪二分类）。Stage 1在100万样本上自监督训练，Stage 2在25-100 demo上进行模仿学习。

## 实验关键数据

### 主实验

| 方法 | 类型 | 输入 | MetaWorld Mean S.R. | Adroit Mean S.R. |
|------|------|------|-------------------|-----------------|
| CLIP | 2D Rep. | RGB | 65.3 | 84.0 |
| R3M | 2D Rep. | RGB | 75.1 | 85.3 |
| PointNet++ | 3D Rep. | PC | 61.6 | 76.0 |
| SPA | 3D Rep. | RGB | 69.5 | 81.3 |
| DP3 | 3D Policy | PC | 65.3 | 66.7 |
| **Lift3D (CLIP)** | **Ours** | **PC** | **83.9** | **88.0** |

### 消融实验

| 配置 | Mean Accuracy | Gain |
|------|:---:|---:|
| Baseline (无MAE无Lifting) | 62 | +0 |
| + Depth reconstruction | 68 | +6 |
| + Affordance masking + Depth | 72 | +10 |
| + Visual distillation | 80 | +18 |
| + 2D Model-Lifting (our PE) | **96** | **+34** |
| + Learnable PE (替代our PE) | 90 | +28 |

### 关键发现

- 2D Model-Lifting是最关键的组件（从80→96，+16），证明直接用2D预训练PE编码3D数据比引入新PE有效6个百分点
- 深度重建比RGB重建更重要（+6 vs +1），说明几何信息对操控任务至关重要
- Affordance masking比随机masking多带来4%提升，验证了聚焦任务相关区域的价值
- 蒸馏防止灾难性遗忘带来+8%提升
- 真实世界30个demo即可学到新操控技能，泛化到不同物体/背景/光照

## 亮点与洞察

- **"投影只为建映射，不做模态转换"的设计极为精巧**——将点云投影到虚拟平面仅用于查找对应的2D PE，实际输入仍然是3D token，完全避免了信息丢失。这种设计可迁移到任何需要将预训练2D模型用于3D数据的场景
- **两阶段渐进增强策略清晰有效**——先隐式（MAE+深度重建）提升3D感知，再显式（PE映射+点云编码）实现3D输入，每个阶段都有明确的目标和验证
- **任务感知masking**引入了CLIP的语义先验来指导几何学习，跨模态的信息融合思路值得借鉴

## 局限与展望

- 仅使用简单MLP作为策略头，未探索扩散策略等更强的action decoder
- 需要单视角RGBD输入，在纯RGB或多视角设置下的表现未知
- Stage 1的MAE训练需要100万样本的大规模数据集，数据准备成本不低
- 虚拟平面数量和位置对性能有影响，但论文中未给出充分的敏感性分析
- 可以探索与VLA模型结合，利用语言指令实现更灵活的任务泛化

## 相关工作与启发

- **vs DP3 (3D Diffusion Policy)**: DP3直接在点云上用扩散模型生成动作，但从头训练3D编码器缺乏预训练知识；Lift3D利用2D预训练知识在MetaWorld上超越DP3 18.6个百分点
- **vs RVT-2**: RVT-2将点云投影为多视图图像送入2D模型，模态转换丢失空间信息；Lift3D通过PE映射避免了模态转换
- **vs Act3D/ChainedDiffuser**: 这些方法将2D特征提升到3D空间进行多尺度表示，需要复杂的3D特征构建；Lift3D更简洁地复用了2D模型本身

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次提出通过PE映射让2D模型直接编码3D点云的思路，MAE设计也有创新
- 实验充分度: ⭐⭐⭐⭐⭐ 三个仿真benchmark+真实场景+30+任务+详细消融+泛化测试+扩展性分析
- 写作质量: ⭐⭐⭐⭐ 方法动机推导清晰，两阶段框架描述完整
- 价值: ⭐⭐⭐⭐⭐ 解决了3D操控中利用2D预训练知识的核心问题，具有很强的实际意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] 3D-MVP: 3D Multiview Pretraining for Robotic Manipulation](3d-mvp_3d_multiview_pretraining_for_manipulation.md)
- [\[NeurIPS 2025\] RoboCerebra: A Large-scale Benchmark for Long-horizon Robotic Manipulation Evaluation](../../NeurIPS2025/robotics/robocerebra_a_large-scale_benchmark_for_long-horizon_robotic_manipulation_evalua.md)
- [\[CVPR 2026\] Localizing, Structuring, and Rendering: Bridging 3D and 2D Vision-Language-Action Models for Robotic Manipulation](../../CVPR2026/robotics/localizing_structuring_and_rendering_bridging_3d_and_2d_vision-language-action_m.md)
- [\[ICLR 2026\] Rethinking Policy Diversity in Ensemble Policy Gradient in Large-Scale Reinforcement Learning](../../ICLR2026/robotics/rethinking_policy_diversity_in_ensemble_policy_gradient_in_large-scale_reinforce.md)
- [\[CVPR 2025\] UniAct: Universal Actions for Enhanced Embodied Foundation Models](universal_actions_for_enhanced_embodied_foundation_models.md)

</div>

<!-- RELATED:END -->
