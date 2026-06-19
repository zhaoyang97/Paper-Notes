---
title: >-
  [论文解读] Situat3DChange: Situated 3D Change Understanding Dataset for Multimodal Large Language Models
description: >-
  [NeurIPS 2025][多模态VLM][3D 场景变化理解] 构建 Situat3DChange 数据集（174K 数据实例），统一了动态场景变化与情境感知理解的感知-行动范式，并提出 SCReasoner——一种高效的 3D MLLM 用于点云对比推理。 物理环境本质是动态的，但现有 3D 数据集要么关注动态场景（物…
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "3D 场景变化理解"
  - "情境感知"
  - "多模态大语言模型"
  - "点云比较"
  - "数据集"
---

# Situat3DChange: Situated 3D Change Understanding Dataset for Multimodal Large Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2510.11509](https://arxiv.org/abs/2510.11509)  
**代码**: [https://github.com/RuipingL/Situat3DChange](https://github.com/RuipingL/Situat3DChange)  
**领域**: 多模态VLM  
**关键词**: 3D 场景变化理解, 情境感知, 多模态大语言模型, 点云比较, 数据集

## 一句话总结

构建 Situat3DChange 数据集（174K 数据实例），统一了动态场景变化与情境感知理解的感知-行动范式，并提出 SCReasoner——一种高效的 3D MLLM 用于点云对比推理。

## 研究背景与动机

物理环境本质是动态的，但现有 3D 数据集要么关注动态场景（物体变化检测）要么关注动态情境（视角-情境推理），缺乏将两者结合的综合框架。具体来说：3D 情境推理数据集（SQA3D、MSQA）基于静态场景；3D 变化理解数据集（Dy2Change、ChangeSim）使用合成数据且缺乏情境上下文。

作者提出两个关键质疑：(Q1) 现有 LLM 生成的数据是否真正反映人类的共享心理地图和情境感知？采访 30 位不同背景人士（含两位盲人）发现，人类以**圆柱坐标系**感知空间（说"左/右"时需指定参考方向，"前"指离观察者更近的），而机器人/工程采用笛卡尔坐标系——两者存在根本性认知差异。(Q2) 场景图能否有效引用空间变化？3DSSG 缺乏对物体旋转和约 10cm 微小位移的感知灵敏度。

## 方法详解

### 整体框架

Situat3DChange 遵循感知-行动模型：**感知**包括 121K 问答对 + 36K 变化描述，**行动**包括 17K 重排指令。基于 3RScan 的 903 个真实世界扫描对构建，由 11K 人工标注为基础，融合自中心视角和分配视角以及分类/坐标空间关系，用 LLM 扩展为完整的情境数据。

### 关键设计

1. **人类认知对齐的数据构建**: 7 位有辅助视障人士经验的合作者为 3RScan 中标记为变化的每个物体标注四个字段：变化原因（Reason）、警告信息（Warning）、变化描述（Description）、重排指令（Rearrangement）。标注融合了分配视角的水平线索。在此基础上，提取垂直关系和物体属性，计算物体的自中心位置变化，用 GPT-4 生成完整的情境变化描述和重排指令。变化描述按顺时针排列并以米为单位；重排指令以步数为单位（对盲人更友好）。

2. **独特查询生成（Distinctive Feature Query）**: 为了唯一标识场景中被变化的物体，先将场景中唯一的物体提升为地标（landmark），然后为其余物体提取三种候选区分特征：独特颜色、水平极端性（最近/最远于地标）、垂直空间关系。人工审核并在必要时补充特征，确保每个查询唯一指向目标物体。

3. **SCReasoner 架构**: 针对"两个高度相似的点云如何有效比较"的挑战，现有方法将所有模态 token 拼接在解码器输入前——对相似点云造成冗余。SCReasoner 利用 **Mamba 的选择性** 从前一场景点云中筛选信息 token，再用 **star operation（逐元素乘法）** 与当前场景 token 融合，无需额外 token 传入语言解码器，参数开销极小。架构基于 LEO 框架，仅增加选择性比较投影器。

### 损失函数 / 训练策略

保持与 LEO 一致的训练设置，在 Situat3DChange 三个任务上联合训练 5 个 epoch。评估采用多维度指标体系：长文本任务用 CIDEr/BLEU-4/METEOR/ROUGE/句嵌入相似度/GPT 评分；QA 任务用 GPT 评分（1-5 分归一化）；距离类问题设计了改进的 REL 指标——解决了当真值距离为 0 时传统 REL 分母为零的问题。

## 实验关键数据

### 主实验

| 任务 | 指标 | SCReasoner (mamba*) | LEO | InternVL2-7B (FT) |
|------|------|-------------------|-----|------------------|
| 变化描述 | GPT 评分 | 13.9% | 12.7% | 8.2% |
| 重排指令 | GPT 评分 | 30.7% | 30.1% | 16.3% |
| QA (平均) | GPT 评分 | 最优 | 次优 | — |

2D MLLM vs 3D MLLM：3D 在分配理解上更优，2D 微调后在自中心任务（距离/方向）上略优。

### 消融实验

| 配置 | 变化描述 GPT | 重排指令 GPT | 说明 |
|------|-------------|-------------|------|
| LEO (baseline) | 12.7 | 30.1 | 简单拼接两个点云 |
| SCReasoner (linear+) | 12.6 | 30.3 | 线性投影+加法融合 |
| SCReasoner (linear*) | 13.4 | 30.3 | 线性投影+乘法融合 |
| SCReasoner (mamba+) | 13.3 | 30.5 | Mamba 投影+加法 |
| SCReasoner (mamba*) | **13.9** | **30.7** | Mamba 投影+乘法，最优 |

### 关键发现

- Mamba 的选择性特性确实优于简单线性投影，有助于从相似点云中筛选出变化相关的信息 token。
- Star operation（乘法融合）优于加法融合，其将输入映射到高维表示的特性有助于突显差异。
- 全景图（panorama）作为 2D MLLM 输入有一定可行性，但缺乏全面的分配上下文导致长文本任务表现不佳。
- 零样本和单样本 MLLM 在变化理解任务上表现很差，说明场景变化理解需要专门的微调。
- 数据量扩展实验显示正向 scaling 效应，且跨域迁移实验证明数据集具有任务无关的训练价值。

## 亮点与洞察

- **以人为中心的数据构建**：开创性地融合自中心/分配视角、分类/坐标空间关系，通过有辅助视障人士经验的标注者确保认知对齐。
- **对现有 LLM 数据生成的深刻批判**：揭示了笛卡尔 vs 圆柱坐标认知差异、场景图对微小变化的感知不灵敏等问题。
- **极简高效的 SCReasoner**：仅增加少量参数就实现了有效的点云对比，无需为解码器增加额外 token。
- 改进的 REL 距离评估指标解决了零距离除零问题，对场景理解领域具有通用价值。
- 三个任务（QA、变化描述、重排指令）统一在感知-行动模型下，评估体系完整。
- 数据质量控制机制（人工标注+自动交叉验证+GPT 扩展）多层保障。

## 局限与展望

- 依赖 3RScan 数据集，场景规模有限（903 扫描对），室内场景为主。
- 人工标注仅来自 7 位合作者，可能存在标注者偏差和文化背景限制。
- SCReasoner 的提升相对 LEO 较小（约 1-2%），点云对比方法仍有较大改进空间。
- 未探索视频序列或连续帧的动态变化理解。
- 3RScan 的测试集标签未公开，只能在验证集上评估，可能影响评估公平性。
- 全景图渲染方法依赖六面体投影，可能引入几何畸变。
- 距离评估的改进 REL 指标虽解决了零距离问题，但对极远距离的容错可能过大。
- 缺乏与最新 3D 基础模型（如 PointLLM、3D-LLM）的更广泛对比。

## 相关工作与启发

- 与 SQA3D、MSQA 等情境 QA 数据集互补，填补了"动态场景+情境感知"的空白。
- SCReasoner 中 Mamba+star operation 的组合可推广至其他需要比较两个相似输入的场景（如视频理解、before/after 对比等）。
- 对辅助技术（视障导航、室内重排机器人）具有直接应用价值。- 人类认知对齐的数据构建思路（圆柱坐标系感知、步数单位指引）对具身 AI 数据集设计有重要参考价值。
- 改进的距离 REL 评估指标对所有涉及距离回归的任务都有通用参考意义。

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首个融合动态场景+情境感知的 3D 数据集，认知对齐视角独特
- **实验充分度**: ⭐⭐⭐⭐ 多任务/多基线评测全面，含 Scaling 和迁移实验
- **写作质量**: ⭐⭐⭐⭐ 动机阐述深刻，数据构建流程详尽
- **价值**: ⭐⭐⭐⭐ 数据集填补重要空白，对具身 AI 和辅助技术有重大意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] SldprtNet: A Large-Scale Multimodal Dataset for CAD Generation in Language-Driven 3D Design](../../CVPR2025/multimodal_vlm/sldprtnet_a_large-scale_multimodal_dataset_for_cad_generation_in_language-driven.md)
- [\[NeurIPS 2025\] Nautilus: A Large Multimodal Model for Underwater Scene Understanding](nautilus_a_large_multimodal_model_for_underwater_scene_understanding.md)
- [\[NeurIPS 2025\] SmartWilds: Multimodal Wildlife Monitoring Dataset](smartwilds_multimodal_wildlife_monitoring_dataset.md)
- [\[ACL 2025\] ViGiL3D: A Linguistically Diverse Dataset for 3D Visual Grounding](../../ACL2025/multimodal_vlm/vigil3d_a_linguistically_diverse_dataset_for_3d_visual_grounding.md)
- [\[NeurIPS 2025\] Better Tokens for Better 3D: Advancing Vision-Language Modeling in 3D Medical Imaging](better_tokens_for_better_3d_advancing_vision-language_modeling_in_3d_medical_ima.md)

</div>

<!-- RELATED:END -->
