---
title: >-
  [论文解读] Spatial Understanding from Videos: Structured Prompts Meet Simulation Data
description: >-
  [NeurIPS 2025 Spotlight][机器人][视觉-空间理解] 提出 SpatialMind 结构化提示策略与 ScanForgeQA 合成QA数据集的双管齐下方案，在不修改VLM架构的前提下显著增强其从扫描视频进行3D空间推理的能力。 视觉-空间理解（从视觉输入推断物体间的空间关系与布局）是机器人导航、自动驾…
tags:
  - "NeurIPS 2025 Spotlight"
  - "机器人"
  - "视觉-空间理解"
  - "链式思维提示"
  - "合成数据"
  - "视觉语言模型"
  - "3D推理"
---

# Spatial Understanding from Videos: Structured Prompts Meet Simulation Data

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2506.03642](https://arxiv.org/abs/2506.03642)  
**代码**: [GitHub](https://github.com/Hyu-Zhang/SpatialMind)  
**领域**: 机器人  
**关键词**: 视觉-空间理解, 链式思维提示, 合成数据, 视觉语言模型, 3D推理

## 一句话总结

提出 SpatialMind 结构化提示策略与 ScanForgeQA 合成QA数据集的双管齐下方案，在不修改VLM架构的前提下显著增强其从扫描视频进行3D空间推理的能力。

## 研究背景与动机

视觉-空间理解（从视觉输入推断物体间的空间关系与布局）是机器人导航、自动驾驶和增强现实等应用的基础能力。虽然点云是3D场景理解的主流表示，但其获取依赖昂贵传感器且计算开销大。因此，研究者开始探索仅基于扫描视频的纯视觉方案。

然而，从扫描视频进行3D空间推理面临两大核心挑战：

**空间不确定性**：缺乏显式深度信息时，模型需从本质受限的2D观测中推断3D结构，遮挡、透视畸变和纹理歧义带来大量不确定性，需要跨帧的多步逻辑推理。

**数据稀缺**：现有数据集规模小、多样性不足，且均源自真实场景扫描、难以扩展，限制了VLM获取鲁棒空间知识的能力。

现有2D空间理解方法（如SpatialVLM、SpatialBot）在复杂3D环境中性能显著下降。已有的3D方法大多依赖点云，实用性和可扩展性受限。因此需要一种可扩展的视觉纯方案来提升VLM的空间推理能力。

## 方法详解

### 整体框架

框架分两大组件：(1) **SpatialMind** — 结构化链式思维(CoT)提示策略，引导VLM执行分步空间推理；(2) **ScanForgeQA** — 从3D仿真场景自动构建的大规模合成QA数据集，用于微调。两者可独立使用也可组合，不修改VLM底层架构。

### 关键设计

1. **场景分解 (Scene Decomposition)**

   场景分解包含三个步骤：
    - **局部建模 (Local Modeling)**：对每帧视频，利用VLM检测候选目标物体并估计其相对于参考物体的局部3D坐标 $\mathbf{p}_{ij}^{\text{local}} \in \mathbb{R}^3$，构建局部3D地图 $\mathcal{L}_i$。
    - **坐标映射 (Coordinate Mapping)**：通过VLM推断相邻帧间的相对旋转 $\mathbf{R}_{k,k-1}$ 和平移 $\mathbf{t}_{k,k-1}$，累积计算每帧的全局变换 $\mathbf{T}_i = \prod_{k=1}^{i} \begin{bmatrix} \mathbf{R}_{k,k-1} & \mathbf{t}_{k,k-1} \\ \mathbf{0} & 1 \end{bmatrix}$，将局部坐标通过齐次变换转换为全局坐标。基于空间邻近性和语义一致性合并跨帧重复检测，得到全局3D地图 $\mathcal{G}$。
    - **认知生成 (Cognition Generation)**：探索三种表示：3D地图、2D空间网格（将物体映射到离散格子 $(i_k,j_k) = (\lfloor x_k/s \rfloor, \lfloor y_k/s \rfloor)$）、自然语言位置描述。实验发现VLM最擅长理解文本描述。

2. **问题分解 (Question Decomposition)**

   将空间问题按类型分类（如物体大小、相对距离、相对方向等），为每种类型设计专用推理流程。例如"相对距离"类问题按四步推理：识别物体 → 估计坐标 → 计算两两距离 → 选最小值。推理时根据问题类型自动选择对应的推理方案。

3. **ScanForgeQA 数据集构建**

   三阶段流水线：
    - **场景构建**：(a) 从3D-FRONT数据集拆分出34,116个单房间场景；(b) 用LLM引导的HoloDeck工具合成160个新场景。
    - **扫描生成**：在Unity引擎中用两种策略模拟扫描视频：轨道扫描（定高圆形轨迹，每5度拍一帧，72帧/圈）和导航扫描（在可行走区域规划路径，起止点各做360度旋转，共72帧/路径）。
    - **QA生成**：自动生成三类问题——属性估计（物体数量、大小、房间面积）、空间推理（相对距离、绝对距离、方向、接触关系）、假设分析（操作可行性）。最终包含34,276场景、103K扫描视频、925K QA对。

### 损失函数 / 训练策略

采用标准SFT微调策略在ScanForgeQA上训练VLM。提示策略为无训练方案。为避免微调影响通用能力，可混入少量传统视频理解数据（如5%-10% ShareGPT4Video）实现能力平衡。

## 实验关键数据

### 主实验

| 模型 | 方法 | VSI-Bench Avg | 提升 |
|------|------|:---:|:---:|
| Qwen2.5-VL-7B | Baseline | 37.2 | - |
| Qwen2.5-VL-7B | +SpatialMind | 39.2 | ↑2.0% |
| Qwen2.5-VL-7B | +ScanForgeQA | 43.3 | ↑6.1% |
| Qwen2.5-VL-7B | +Both | 43.9 | ↑6.7% |
| InternVL2-40B | Baseline | 36.0 | - |
| InternVL2-40B | +Both | 44.5 | ↑8.5% |
| Qwen2.5-VL-72B | Baseline | 39.2 | - |
| Qwen2.5-VL-72B | +Both | 47.1 | ↑7.9% |
| GPT-4o | +SpatialMind | 40.8 | ↑6.8% |
| Gemini-1.5 Pro | +SpatialMind | 52.8 | ↑7.4% |

| 模型 | 方法 | OpenEQA Acc | ScanQA BLEU-1 | SQA3D EM-1 |
|------|------|:---:|:---:|:---:|
| Qwen2.5-VL-7B | Baseline | 50.1 | 32.5 | 17.2 |
| Qwen2.5-VL-7B | +Both | 58.6 | 37.9 | 24.5 |
| Qwen2.5-VL-72B | Baseline | 53.8 | 35.4 | 34.8 |
| Qwen2.5-VL-72B | +Both | 60.4 | 44.1 | 46.3 |

### 消融实验

| 配置 | Room Size | VSI-Bench Avg | 说明 |
|------|:---:|:---:|------|
| Qwen2.5-VL-7B baseline | 38.9 | 37.2 | 基线 |
| +SQA3D 微调 | 38.8 | 38.9 | 已有数据集效果有限 |
| +ScanQA 微调 | 38.5 | 39.1 | 已有数据集效果有限 |
| +ScanForgeQA 微调 | 44.9 | 43.3 | 合成数据显著更优 |
| CoT-Question only | 50.6 | 41.3 | 仅问题分解 |
| CoT-Scene only | 52.1 | 42.7 | 场景描述贡献更大 |
| Full SpatialMind | 53.8 | 44.0 | 两者互补 |

### 关键发现

- 文本描述是VLM最易理解的场景表示格式，优于3D地图和2D网格
- 大模型从提示策略中受益更多，小模型从微调中收益更大（7B微调+6.1%，提示仅+2.0%）
- 提示+微调两者互补，组合使用持续带来增益
- ScanForgeQA微调对通用视频能力影响轻微（MVBench略升，Video-MME略降），混合数据可缓解

## 亮点与洞察

- 纯视觉方案不修改模型架构，通用性强，可适配各种规模和类型的VLM
- 合成数据管线可扩展性好，避免了真实场景数据获取的高成本
- 人类与VLM各有所长：人类在定性任务（如外观排序100%准确率）上表现优异，VLM在精确定量估计上反超人类，两者互补

## 局限与展望

- 场景分解依赖VLM自身的位姿估计能力，视角变化剧烈时误差可能累积
- 仿真数据与真实世界仍存在域差距
- 文本描述格式可能在物体密集场景中信息压缩不足
- 可探索与深度估计模型或SLAM技术结合以提升坐标精度

## 相关工作与启发

- 对比SpatialVLM、SpatialRGPT等2D空间理解方法，指出其在复杂3D场景中的局限
- CoT提示思路可推广到其他需要多步推理的视觉任务
- 合成数据+微调的组合策略为数据稀缺领域提供了可复制的范式

## 评分

- 新颖性: ⭐⭐⭐⭐ 提示+合成数据双管齐下是合理创新，但各组件并非全新
- 实验充分度: ⭐⭐⭐⭐⭐ 多个基准、多种模型、充分消融，实验非常全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表丰富
- 价值: ⭐⭐⭐⭐ 提供了实用的空间推理增强方案，对具身智能领域有直接参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Rethinking the Simulation vs. Rendering Dichotomy: No Free Lunch in Spatial World Modelling](rethinking_the_simulation_vs_rendering_dichotomy_no_free_lunch_in_spatial_world_.md)
- [\[ECCV 2024\] Hierarchically Structured Neural Bones for Reconstructing Animatable Objects from Casual Videos](../../ECCV2024/robotics/hierarchically_structured_neural_bones_for_reconstructing_animatable_objects_fro.md)
- [\[ICLR 2026\] UrbanVerse: Scaling Urban Simulation by Watching City-Tour Videos](../../ICLR2026/robotics/urbanverse_scaling_urban_simulation_by_watching_city-tour_videos.md)
- [\[NeurIPS 2025\] Talk2Event: Grounded Understanding of Dynamic Scenes from Event Cameras](talk2event_grounded_understanding_of_dynamic_scenes_from_event_cameras.md)
- [\[CVPR 2026\] Spatial-Aware VLA Pretraining through Visual-Physical Alignment from Human Videos](../../CVPR2026/robotics/spatial-aware_vla_pretraining_through_visual-physical_alignment_from_human_video.md)

</div>

<!-- RELATED:END -->
