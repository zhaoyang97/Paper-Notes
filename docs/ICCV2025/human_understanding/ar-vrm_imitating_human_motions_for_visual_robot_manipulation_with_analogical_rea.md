---
title: >-
  [论文解读] AR-VRM: Imitating Human Motions for Visual Robot Manipulation with Analogical Reasoning
description: >-
  [ICCV 2025][人体理解][视觉机器人操控] 提出 AR-VRM，首个通过显式模仿人类手部关键点来增强视觉机器人操控的方法，采用关键点视觉语言模型预训练从大规模人类动作视频中学习动作知识，并通过类比推理(Analogical Reasoning)建立人手关键点与机器人组件的映射。 视觉机器人操控(VRM)需要配对的图…
tags:
  - "ICCV 2025"
  - "人体理解"
  - "视觉机器人操控"
  - "类比推理"
  - "人体关键点"
  - "视觉语言预训练"
  - "人机动作迁移"
---

# AR-VRM: Imitating Human Motions for Visual Robot Manipulation with Analogical Reasoning

**会议**: ICCV 2025  
**arXiv**: [2508.07626](https://arxiv.org/abs/2508.07626)  
**代码**: [https://github.com/idejie/ar](https://github.com/idejie/ar)  
**领域**: 人体理解 / 机器人操控  
**关键词**: 视觉机器人操控, 类比推理, 人体关键点, 视觉语言预训练, 人机动作迁移

## 一句话总结

提出 AR-VRM，首个通过显式模仿人类手部关键点来增强视觉机器人操控的方法，采用关键点视觉语言模型预训练从大规模人类动作视频中学习动作知识，并通过类比推理(Analogical Reasoning)建立人手关键点与机器人组件的映射。

## 研究背景与动机

视觉机器人操控(VRM)需要配对的图像、语言指令和机器人动作状态数据，采集成本极高，数据稀缺严重制约了性能。现有方法虽利用大规模数据预训练弥补不足，但存在两个核心问题：

**数据域不匹配**：使用 VQA 等 Web 数据预训练，与机器人操控任务差异大，知识迁移受限

**隐式学习的局限**：部分方法使用人类动作视频（如 Ego4D），但通过对比学习或像素级未来帧预测进行隐式学习，不可避免地引入无关背景信息和像素噪声

**关键洞察**：人手关键点和机器人动作在操控物体时具有底层相似性。应该显式地聚焦于动作本身（关键点），忽略无关视觉信息。核心挑战在于：(1) 如何从大规模人类视频中以关键点形式提取动作知识？(2) 如何建立形态差异巨大的人手和机器人臂之间的对应关系？

## 方法详解

### 整体框架

AR-VRM 由两部分组成：预训练阶段学习关键点视觉语言模型(Keypoint VLM)，微调阶段引入类比推理模块。

### 关键设计

1. **关键点视觉语言模型预训练 (Keypoint VLM Pretraining)**

    - 使用大规模人类第一人称视频数据集 Ego4D（3,500小时、800K视频片段、8M帧）
    - 使用 InterHand 离线检测3D手部关键点 $k_t \in \mathbb{R}^K$（图像坐标系下的3D坐标）
    - 三模态 token 对齐：
        - 语言：CLIP 文本编码器 + MLP → $z_l \in \mathbb{R}^d$
        - 视觉：ViT(MAE预训练) + Perceiver Resampler → $z_o^{CLS}, z_o^{p_{1:M}} \in \mathbb{R}^d$
        - 关键点：HandFormer 编码器 + MLP → $z_k \in \mathbb{R}^d$
    - 三模态 token 拼接后送入 Transformer 层进行 next-token prediction：
    $\hat{z_{k_t}} = h^{Atten}(z_l, z_{o_1}, z_{k_1}, ..., z_{o_{t-1}}, z_{k_{t-1}})$
    - 预训练损失为 MSE：$\mathcal{L}_{pretrain} = \sum_{t=1}^T \mathcal{L}_{MSE}(\text{KeypointHead}(\hat{z_{k_t}}), k_t)$
    - 固定 CLIP 文本编码器和 ViT 参数，仅训练交互层

2. **机器人微调与类比推理 (Analogical Reasoning)**

    - **人类动作检索**：基于语言相似度和视觉帧特征的余弦相似度检索相关人类动作视频：
    $sim(\tau^R, \tau^H) = cos(z_l^R, z_l^H) + \sum_t cos(z_{o_t}^R, z_{o_t}^H)$
    - **类比映射矩阵**：引入可学习矩阵 $m \in \mathbb{R}^{S \times K}$ 表示人手关键点到机器人组件的映射，$S$ 为机器人组件数，$K$ 为手部关键点数
    - 通过映射矩阵将人类关键点特征变换为模拟的机器人状态特征：
    $f_{s_j}^* = (1-\alpha) \cdot m \cdot f_{k_j} + \alpha \cdot f_s$
    - 类比推理损失：$\mathcal{L}_{AR} = \sum_{j=1}^J \mathcal{L}_{MSE}(\text{Linear}(f_{s_{j,T}}^*), s_T)$
    - 总微调损失：$\mathcal{L}_{finetune} = \mathcal{L}_{state} + \beta \cdot \mathcal{L}_{AR}$

3. **微调策略设计**

    - 冻结关键点编码器和关键点预测头，微调 VLM 的 Transformer 层
    - 冻结关键点模块可保持预训练获得的关键点编解码能力
    - 微调 VLM 防止在小规模机器人数据上过拟合并遗忘预训练知识
    - 人类视频样本在微调阶段起到数据回放的作用

### 损失函数总结

| 阶段 | 损失函数 | 说明 |
|------|---------|------|
| 预训练 | $\mathcal{L}_{pretrain} = \sum \mathcal{L}_{MSE}$ | 预测人手关键点 |
| 微调-状态 | $\mathcal{L}_{state} = \mathcal{L}_{MSE}$ | 预测机器人状态 |
| 微调-类比 | $\mathcal{L}_{AR} = \sum \mathcal{L}_{MSE}$ | 类比映射监督 |

## 实验关键数据

### 主实验（CALVIN ABCD→D）

| 方法 | 1任务 | 2任务 | 3任务 | 4任务 | 5任务 | Avg.Len | Avg.Rate |
|------|-------|-------|-------|-------|-------|---------|----------|
| MCIL | 0.373 | 0.027 | 0.002 | 0.000 | 0.000 | 0.40 | 8.0% |
| RT-1 | 0.844 | 0.617 | 0.438 | 0.323 | 0.227 | 2.45 | 49.0% |
| GR-1 | 0.949 | 0.896 | 0.844 | 0.789 | 0.731 | 4.21 | 84.2% |
| **AR-VRM** | **0.951** | **0.915** | **0.855** | **0.800** | **0.751** | **4.27** | **85.4%** |

未见场景泛化（ABC→D）：AR-VRM 65.9% vs GR-1 61.2%（**+4.7%**）

### 消融实验

| Pretrain | Retrieval | AR | Avg.Len | Avg.Rate |
|----------|-----------|-----|---------|----------|
| ✗ | ✗ | ✗ | 3.00 | 60.0% |
| ✓ | ✗ | ✗ | 4.06 | 81.3% |
| ✓ | ✓ | ✗ | 4.21 | 84.3% |
| ✓ | ✓ | ✓ | **4.27** | **85.4%** |

Few-shot (10% data)：AR-VRM 45.6% vs GR-1 40.0%（**+5.6%**）

真实机器人实验：

| 任务 | RT-1 | GR-1 | AR-VRM |
|------|------|------|--------|
| 已见物体 | 0.27 | 0.79 | **0.95** |
| 未见实例 | 0.13 | 0.73 | **0.91** |
| 未见类别 | 0.00 | 0.30 | **0.53** |
| 关节操控 | 0.35 | 0.75 | **0.82** |

### 关键发现

- 预训练带来最大提升（60% → 81.3%），证明大规模人类视频预训练的有效性
- 检索 + 数据回放进一步提升至 84.3%，防止遗忘预训练知识
- 类比推理锦上添花（84.3% → 85.4%），但在少数据场景贡献更大（10%数据时提升5.6%）
- 冻结关键点参数 + 微调 VLM 是最优组合（85.4%），反之则显著下降
- 类比映射的可视化表明学到了合理的人机对应关系：机器人夹爪对应人类指尖，机器人臂根部对应手掌

## 亮点与洞察

- **"显式 vs 隐式"的核心洞察**：前人通过像素级预测或对比学习隐式学习人类动作知识，不可避免地混入背景噪声；本文聚焦关键点，直击动作本质
- **类比推理的优雅设计**：可学习映射矩阵自动发现人手-机器人的功能对应关系，无需人工指定
- **数据回放的巧妙用途**：微调时加入检索到的人类视频不仅提供动作引导，还防止 VLM 过拟合

## 局限与展望

- 类比映射矩阵是全局共享的，不同任务可能需要不同的映射关系
- 依赖 InterHand 的关键点检测质量，遮挡或快速运动下可能不稳定
- Ego4D 数据偏向日常操作，对于精细工业操控的适用性未知
- 5 任务序列的成功率(75.1%)仍有提升空间
- 真实机器人中未见类别泛化(53%)尚有较大差距

## 相关工作与启发

- 与 ATM（需要配对人机数据）不同，AR-VRM 仅需分别收集人类视频和机器人演示
- 关键点预测的预训练思路可扩展到全身关键点，支持更多类型的人机交互
- 为"从人类视频学习机器人"这一研究方向提供了显式学习的新范式

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个通过显式关键点模仿学习的VRM方法，类比推理设计新颖
- 实验充分度: ⭐⭐⭐⭐⭐ CALVIN全设置+真实机器人+完整消融+可视化分析
- 写作质量: ⭐⭐⭐⭐ 动机和方法描述清晰，对比分析到位
- 价值: ⭐⭐⭐⭐ 为数据稀缺下的机器人操控提供了实用的预训练方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Two by Two: Learning Multi-Task Pairwise Objects Assembly for Generalizable Robot Manipulation](../../CVPR2025/human_understanding/two_by_two_learning_multi-task_pairwise_objects_assembly_for_generalizable_robot.md)
- [\[ICML 2025\] Scaling Large Motion Models with Million-Level Human Motions](../../ICML2025/human_understanding/scaling_large_motion_models_with_million-level_human_motions.md)
- [\[ICCV 2025\] What's Making That Sound Right Now? Video-centric Audio-Visual Localization](whats_making_that_sound_right_now_video-centric_audio-visual_localization.md)
- [\[CVPR 2025\] StickMotion: Generating 3D Human Motions by Drawing a Stickman](../../CVPR2025/human_understanding/stickmotion_generating_3d_human_motions_by_drawing_a_stickman.md)
- [\[CVPR 2026\] EgoPoseFormer v2: Accurate Egocentric Human Motion Estimation for AR/VR](../../CVPR2026/human_understanding/egoposeformer_v2_accurate_egocentric_human_motion_estimation_for_arvr.md)

</div>

<!-- RELATED:END -->
