---
title: >-
  [论文解读] Can Generative Video Models Help Pose Estimation?
description: >-
  [CVPR 2025][图像生成][位姿估计] 提出 InterPose，利用预训练视频生成模型在两张少/无重叠图像之间"幻想"中间帧，配合自一致性评分选择最佳视频，在 DUSt3R 基础上一致提升四个数据集的位姿估计精度。 领域现状：成对相机位姿估计是 3D 视觉的基础任务。传统方法依赖特征匹配和对应关系计算…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "位姿估计"
  - "视频生成模型"
  - "帧插值"
  - "自一致性评分"
  - "少重叠图像"
---

# Can Generative Video Models Help Pose Estimation?

**会议**: CVPR 2025  
**arXiv**: [2412.16155](https://arxiv.org/abs/2412.16155)  
**代码**: [https://Inter-Pose.github.io](https://Inter-Pose.github.io) (项目页)  
**领域**: 图像生成  
**关键词**: 位姿估计, 视频生成模型, 帧插值, 自一致性评分, 少重叠图像

## 一句话总结
提出 InterPose，利用预训练视频生成模型在两张少/无重叠图像之间"幻想"中间帧，配合自一致性评分选择最佳视频，在 DUSt3R 基础上一致提升四个数据集的位姿估计精度。

## 研究背景与动机

**领域现状**：成对相机位姿估计是 3D 视觉的基础任务。传统方法依赖特征匹配和对应关系计算，需要图像间有足够重叠。DUSt3R 等深度学习方法在大规模 3D 数据上训练，泛化能力好，但在视角差异大、几乎无重叠的情况下仍然困难。

**现有痛点**：当两张图像视角差异极大（如教室的两面墙），无法建立可靠的特征对应，传统和学习方法都会失败。根本原因是缺少中间视角的"桥梁"信息。训练更强的 3D 模型需要大量 3D 标注数据，而 3D 数据的多样性和规模远不如视频数据。

**核心矛盾**：位姿估计需要几何一致的视觉对应，但少重叠场景缺乏对应；视频生成模型包含丰富的场景先验但不保证几何一致性，生成的视频可能有伪影和不一致几何。

**本文目标** 如何利用视频生成模型的场景先验来改善少重叠图像对的位姿估计。

**切入角度**：人类能从两张几乎不重叠的教室图片推断空间关系，是因为拥有关于典型教室布局的先验知识。类似地，视频模型在海量视频上训练，学到了强大的场景运动先验。用视频模型在两张图之间插帧，创建密集的视觉过渡，将少重叠问题转化为多帧密集重叠问题。

**核心 idea**：用视频插帧模型在两张图之间幻想过渡帧来"填充"视觉空白，通过自一致性评分从多个视频中选出几何最一致的，辅助 DUSt3R 提升位姿估计。

## 方法详解

### 整体框架
给定图像对 $(I_A, I_B)$，用 GPT-4o 生成两个描述性 caption，结合正序和翻转序（$A \to B$ 和 $B \to A$），生成 $n=4$ 个插帧视频。对每个视频，随机采样 $m=11$ 组帧子集（每组 $k=5$ 帧含原始两帧），分别用 DUSt3R 估计位姿，计算自一致性评分选出最佳视频，输出其 medoid 位姿作为最终结果。

### 关键设计

1. **视频插帧作为场景先验**

    - 功能：在少/无重叠图像对之间创建密集视觉过渡，将困难的宽基线问题转化为简单的窄基线多帧问题
    - 核心思路：使用现成的视频插帧模型 $f_{vid}(I_A, I_B, p) = [I_1, ..., I_N]$，其中 $I_1=I_A, I_N=I_B$。测试了三个模型：DynamiCrafter（开源）、Runway Gen-3（商用）、Luma Dream Machine（商用）。生成的中间帧与原始图像对一起输入 DUSt3R 的多帧扩展版本进行位姿估计。视频模型不需要任何修改或微调
    - 设计动机：视频模型在比 3D 数据集大几个数量级的视频数据上训练，学到了更强的场景布局和运动先验。即使生成的帧不完美，也为 DUSt3R 提供了比仅有两帧更多的几何线索

2. **自一致性评分（Self-Consistency Score）**

    - 功能：从多个生成视频中选出几何最一致的视频
    - 核心思路：对每个视频，随机采样 $m$ 组帧子集，计算每组的位姿估计 $\hat{T}^{(i)}$。用 medoid 距离衡量一致性：$D_{med} = \min_i \frac{1}{m-1}\sum_{j \neq i} dist(\hat{T}^{(i)}, \hat{T}^{(j)})$。低 medoid 距离意味着不同帧子集给出几乎相同的位姿，说明视频几何一致。为防止退化情况（低质量视频恰好一致地给出错误预测），加入锚定项：$D_{total} = D_{med} + dist(\hat{T}_{med}, f_{pose}(\{I_A, I_B\}))$。最终选 $D_{total}$ 最低的视频，输出其 medoid 位姿
    - 设计动机：视频模型常产生伪影（物体突然出现/消失、morphing、几何不一致），简单平均所有视频的预测反而会变差（实验中 Dream Machine 平均后 MRE 从 13.28° 恶化到 21.85°）。medoid 评分无需真值标签即可区分好坏视频

3. **对称生成策略**

    - 功能：缓解视频模型固有的运动偏置
    - 核心思路：对每个图像对同时生成正序（$A \to B$）和翻转序（$B \to A$）的视频。用两个不同的 caption 进一步增加多样性，总共 4 个视频/对
    - 设计动机：视频模型通常偏向生成从左到右的相机运动，翻转输入顺序可以补偿这一偏置

### 损失函数 / 训练策略
InterPose 是完全无训练的推理时方法——不修改任何模型，仅是将视频生成模型和 DUSt3R 作为黑盒组合使用。位姿距离用旋转测地距离 + 平移角度误差的和来衡量。

## 实验关键数据

### 主实验

| 数据集 | 方法 | MRE↓ | MTE↓ | AUC30↑ |
|--------|------|------|------|--------|
| Cambridge (室外) | DUSt3R only | 13.28° | — | 77.23 |
| Cambridge | **InterPose+Runway Medoid** | **10.78°** | — | **80.59** |
| ScanNet (室内) | DUSt3R only | 21.31° | 24.72° | 60.34 |
| ScanNet | **InterPose+DreamMachine Medoid** | **17.65°** | **15.88°** | **63.06** |
| DL3DV-10K | DUSt3R only | — | 13.08° | 66.99 |
| DL3DV-10K | **InterPose+DreamMachine** | — | **8.72°** | **69.44** |
| NAVI (物体) | DUSt3R only | 8.65° | 7.88° | — |
| NAVI | **InterPose+DreamMachine** | **7.85°** | **6.51°** | — |

### 消融实验

| 配置 | Cambridge MRE↓ | 说明 |
|------|---------------|------|
| DUSt3R only | 13.28° | 基线 |
| + DreamMachine Avg | 21.85° | 简单平均反而变差 |
| + DreamMachine Medoid | **11.96°** | 自一致性评分恢复提升 |
| Oracle（选最佳视频） | 3.65° | 上界，说明更好的视频选择空间巨大 |

### 关键发现
- 自一致性评分是关键：简单平均所有视频的预测在某些模型上反而变差（Cambridge MRE 从 13.28° 恶化到 21.85°），但 medoid 选择可以稳定提升
- Oracle 上界（选择真值位姿最近的视频）显示 MRE 可低至 3.65°，说明视频模型确实包含正确的几何先验，瓶颈在于视频选择
- Runway 和 DreamMachine 通常优于开源的 DynamiCrafter，商用模型质量更稳定
- 翻转输入顺序对缓解向右运动偏置有帮助
- 方法在四种不同类型场景（室外/室内/中心聚焦/物体中心）上一致有效

## 亮点与洞察
- **将视频模型作为场景先验**的跨模态思路极具启发性：视频数据远多于 3D 数据，这种利用方式开辟了"用生成模型 bootstrap 3D 理解"的新方向
- **自一致性评分**不需要任何标签就能评估视频质量，基于"好的重建应该对输入扰动不敏感"这一简洁原理
- **完全即插即用**：不修改任何模型，仅组合使用，方法极其简单但一致有效

## 局限与展望
- 视频生成模型**昂贵且慢**——商用模型总花费约 $5,500，限制了实验规模
- 视频模型不保证多视图一致性，有时所有生成视频都质量差
- 对 prompt、相机内参、宽高比敏感，需要精心设计
- Oracle 与实际性能差距大（3.65° vs 10.78°），更好的视频选择/评分是重要的未来方向
- 目前仅测了 DUSt3R 作为位姿估计器，与 COLMAP 等传统方法的组合未探索

## 相关工作与启发
- **vs DUSt3R**: DUSt3R 在少重叠时性能下降。InterPose 通过提供中间帧扩大有效重叠，不修改 DUSt3R 即可提升。二者是互补关系
- **vs 流匹配/关键点方法**: SIFT+NN 和 LOFTR 在少重叠时基本失效（MRE > 30°/64°），需要对应点才能工作
- **vs 概率位姿方法**: 扩散位姿模型等可以处理歧义，但不利用额外的场景先验。InterPose 通过生成先验提供更多信息

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次利用视频生成模型改善位姿估计，方向新颖且有重要启示意义
- 实验充分度: ⭐⭐⭐⭐ 四个数据集、三个视频模型、消融和 oracle 分析完整，但受限于成本样本量偏小
- 写作质量: ⭐⭐⭐⭐⭐ 动机描述直观（教室例子），图表出色，方法简洁易懂
- 价值: ⭐⭐⭐⭐ 开辟了视频模型→3D 理解的新范式，但实用性受限于视频生成成本

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] AniMer: Animal Pose and Shape Estimation Using Family Aware Transformer](animer_animal_pose_and_shape_estimation_using_family_aware_transformer.md)
- [\[CVPR 2025\] Goku: Flow Based Video Generative Foundation Models](goku_flow_based_video_generative_foundation_models.md)
- [\[CVPR 2025\] PhysicsGen: Can Generative Models Learn from Images to Predict Complex Physical Relations?](physicsgen_can_generative_models_learn_from_images_to_predict_complex_physical_r.md)
- [\[CVPR 2025\] Pursuing Temporal-Consistent Video Virtual Try-On via Dynamic Pose Interaction](pursuing_temporal-consistent_video_virtual_try-on_via_dynamic_pose_interaction.md)
- [\[CVPR 2025\] VLog: Video-Language Models by Generative Retrieval of Narration Vocabulary](vlog_video-language_models_by_generative_retrieval_of_narration_vocabulary.md)

</div>

<!-- RELATED:END -->
