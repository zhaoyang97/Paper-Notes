---
title: >-
  [论文解读] RoboSpatial: Teaching Spatial Understanding to 2D and 3D Vision-Language Models for Robotics
description: >-
  [CVPR 2025][多模态VLM][空间推理] RoboSpatial 构建了一个包含 1M 图像、5k 3D 扫描和 3M 空间关系标注的大规模机器人空间理解数据集，通过自动化 pipeline 从已有 3D 场景数据中生成三类空间问答对（空间上下文/兼容性/配置），并引入三种参考坐标系（自我/世界/物体），在多个 2D 和 3D VLM 上训练后显著提升空间推理性能，并在真实机器人操作实验中验证了有效性。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "空间推理"
  - "机器人操作"
  - "VLM微调"
  - "3D空间理解"
  - "参考坐标系"
---

# RoboSpatial: Teaching Spatial Understanding to 2D and 3D Vision-Language Models for Robotics

**会议**: CVPR 2025  
**arXiv**: [2411.16537](https://arxiv.org/abs/2411.16537)  
**代码**: [https://github.com/chanh-ee/RoboSpatial](https://github.com/chanh-ee/RoboSpatial)  
**领域**: 多模态VLM  
**关键词**: 空间推理, 机器人操作, VLM微调, 3D空间理解, 参考坐标系

## 一句话总结
RoboSpatial 构建了一个包含 1M 图像、5k 3D 扫描和 3M 空间关系标注的大规模机器人空间理解数据集，通过自动化 pipeline 从已有 3D 场景数据中生成三类空间问答对（空间上下文/兼容性/配置），并引入三种参考坐标系（自我/世界/物体），在多个 2D 和 3D VLM 上训练后显著提升空间推理性能，并在真实机器人操作实验中验证了有效性。

## 研究背景与动机
VLM 在机器人领域的应用日益广泛，但空间理解能力仍然是瓶颈——现有 VLM 可以描述"桌上有碗"，却无法推理碗在桌上的具体位置、能否放下新物品等复杂空间关系。核心矛盾在于：VLM 的训练数据来自通用图像数据集，缺乏机器人场景中的精细空间标注，尤其缺乏对参考坐标系（自我中心/世界中心/物体中心）的理解能力。现有空间推理数据集（如 SpatialVLM、BLINK）要么规模小、要么不支持多坐标系、要么不适用于具身场景。本文的切入角度是：**空间理解的瓶颈是缺乏合适的训练数据**，因此构建大规模、多坐标系、机器人导向的空间 QA 数据集来弥补这一缺口。

## 方法详解

### 整体框架
RoboSpatial 的核心是一个自动化数据生成 pipeline：输入带有 3D 标注框、相机位姿和语义标签的场景数据集（如 ScanNet、Matterport3D、GraspNet-1B），输出包含 $(I_i, q_i, a_i, l_i)$（图像、问题、答案、参考坐标系标签）的空间推理 QA 数据集。Pipeline 分两阶段：先在 3D 空间提取空间关系，再映射到 2D 图像空间生成 QA 对。

### 关键设计
1. **三类空间关系定义**:
    - 功能：覆盖机器人场景中最核心的空间推理需求
    - 核心思路：将空间理解分解为三个层次—— (1) Spatial Context（空间上下文）：识别环境中的空闲空间，输出可放置位置的 2D 坐标；(2) Spatial Compatibility（空间兼容性）：判断目标物体能否放进指定区域，通过模拟虚拟包围盒放置并检测碰撞来回答 True/False；(3) Spatial Configuration（空间配置）：判断两个物体之间的相对空间关系（左/右/上/下/前/后），输出 True/False
    - 设计动机：距离度量难以跨场景归一化，而这三种关系直接对应机器人的路径规划、物品放置和导航需求

2. **三种参考坐标系**:
    - 功能：让模型理解同一空间关系在不同视角下的差异
    - 核心思路：每个 QA 对从三个坐标系分别生成—— (a) Ego-centric（以相机位姿为中心），(b) World-centric（全局坐标系），(c) Object-centric（以锚物体的朝向为中心，如"车前方"指车头方向）
    - 设计动机：自然语言中的空间描述隐含了参考坐标系，"桌子前面"在不同视角下完全不同，模型必须学会区分

3. **两阶段数据生成 Pipeline**:
    - 功能：从 3D 标注场景中自动生成大规模 QA 对
    - 核心思路：Stage 1 在 3D 空间中基于 oriented bounding box 的位置和朝向，计算物体间的六方向关系 $r_i \in \{left, right, above, below, front, behind\}$；Stage 2 在 2D 图像空间中通过俯视占据地图采样空闲点，利用射线检测过滤遮挡点，并通过虚拟碰撞检测判断兼容性（要求各轴至少 10cm 余量）
    - 设计动机：利用精确 3D 几何避免感知模型的噪声标注，同时通过相机投影桥接 2D/3D 模态

### 损失函数 / 训练策略
在已有 VLM（如 VILA-1.5-8B、LLaVA-NeXT-8B）上进行微调，使用 RoboSpatial 数据集 + 辅助物体定位数据集（将物体描述映射到 2D bounding box）进行联合训练。辅助定位数据集用于缓解物体指代解析错误带来的级联失败。

## 实验关键数据

### 主实验（RoboSpatial-Val）

| 模型 | Indoor 均分 | Tabletop 均分 | 总均分 | 提升 |
|------|-----------|-------------|-------|------|
| VILA (baseline) | 43.1 | 37.4 | 40.2 | - |
| VILA + RoboSpatial | 64.8 | 62.9 | 63.9 | +23.7 |
| LLaVA-NeXT (baseline) | 31.4 | 29.2 | 30.3 | - |
| LLaVA-NeXT + RoboSpatial | 60.4 | 60.5 | 60.5 | +30.2 |
| LEO (3D, baseline) | 41.9 | 43.7 | 42.8 | - |
| LEO + RoboSpatial | 73.1 | 70.7 | 71.9 | +29.1 |
| GPT-4o (zero-shot) | 49.3 | 52.3 | 50.8 | - |

### 域外泛化（RoboSpatial-Home / BLINK / SpatialBench）

| 模型 | Home Config | Home Compat | BLINK Acc | SpatialBench |
|------|------------|------------|-----------|-------------|
| LLaVA-NeXT | 68.3 | 70.5 | 71.3 | 55.9 |
| LLaVA-NeXT + RoboSpatial | 78.9 | 80.1 | 79.0 | 70.6 |
| SpaceLLaVA + RoboSpatial | 71.6 | 72.4 | 81.8 | 67.7 |
| GPT-4o | 77.2 | 58.1 | 76.2 | 70.6 |

### 机器人实验

| 模型 | 成功率 (%) |
|------|-----------|
| LLaVA-NeXT | 23.7 |
| LLaVA-NeXT + RoboSpatial | **52.6** |
| RoboPoint | 44.7 |
| GPT-4o | 46.9 |

### 关键发现
- 所有 2D 和 3D VLM 在 RoboSpatial 微调后，所有任务性能均大幅提升（↑20-30%）
- 模型能泛化到训练中未见过的空间介词（如 "next to"、"under"），因为训练覆盖了 3D 空间的六个主方向
- 3D VLM（如 LEO）通常优于 2D VLM，但公平性比较受限于预训练数据差异
- 跨环境迁移有正向协同效应：在 indoor 数据上训练也能提升 tabletop 性能

## 亮点与洞察
- **数据驱动的空间理解**：证明了空间推理的瓶颈在数据而非模型架构，通用 VLM + 好的空间数据即可大幅提升
- **参考坐标系的引入**是关键创新点，让模型学会"车前方"和"我前方"的区别
- 自动化 pipeline 可扩展到新场景和新空间关系，具有良好的可扩展性
- 微调后的 LLaVA-NeXT 在真实机器人实验中超过 GPT-4o，显示了领域数据的价值

## 局限与展望
- 空间上下文（context）任务中使用凸包判定正确性标准过于严格，导致分数偏低
- 2D 到 3D 的投影误差（2 像素 → 5-10cm）仍是机器人操作中的关键瓶颈
- 3D VLM 目前需要完整 3D 扫描作为输入，难以在真实场景中实时获取
- 模板化 QA 可能限制语言多样性，未来可引入 LLM 改写

## 相关工作与启发
- **vs SpatialVLM/SpatialRGPT**: 它们基于互联网图像和感知模型标注，无法泛化到具身场景；RoboSpatial 基于真实 3D 扫描，精度更高
- **vs RoboPoint/Molmo**: 这些指向模型缺乏参考坐标系和物体兼容性理解，只能预测点而不能回答空间关系
- **vs EmbSpatial-Bench**: 规模太小（4k QA vs 3M QA），且不支持多坐标系
- **vs 3D-LLM/LEO**: 3D VLM 能直接利用深度信息，但需要完整扫描输入，RoboSpatial 同时支持 2D 和 3D
- **vs BLINK-Spatial**: BLINK 仅包含 286 个样本的评测，RoboSpatial 是训练+评测一体的完整方案

## 评分
- 新颖性: ⭐⭐⭐⭐ 参考坐标系和三类空间关系的定义很有见地，但整体方法偏数据工程
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖多个 VLM、多个 benchmark、域外泛化、交叉环境迁移、真实机器人实验
- 写作质量: ⭐⭐⭐⭐ 结构清晰，pipeline 描述详尽
- 价值: ⭐⭐⭐⭐ 对机器人领域的 VLM 空间推理有直接推动作用，数据集开源价值很高

## 补充说明
- 数据集规模：3M QA 对来自 5 个源数据集（ScanNet、Matterport3D、3RScan、HOPE、GraspNet-1B）
- 评测中凸包判定标准偏严格，实际准确率可能高于报告值
- 跨环境实验表明 indoor 和 tabletop 之间存在正向迁移效应，一起训练效果更好

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] MM-Spatial: Exploring 3D Spatial Understanding in Multimodal LLMs](../../ICCV2025/multimodal_vlm/mm-spatial_exploring_3d_spatial_understanding_in_multimodal_llms.md)
- [\[CVPR 2026\] HiSpatial: Taming Hierarchical 3D Spatial Understanding in Vision-Language Models](../../CVPR2026/multimodal_vlm/hispatial_taming_hierarchical_3d_spatial_understanding_in_vision-language_models.md)
- [\[CVPR 2026\] Abstract 3D Perception for Spatial Intelligence in Vision-Language Models](../../CVPR2026/multimodal_vlm/abstract_3d_perception_for_spatial_intelligence_in_vision-language_models.md)
- [\[CVPR 2025\] LayoutVLM: Differentiable Optimization of 3D Layout via Vision-Language Models](layoutvlm_differentiable_optimization_of_3d_layout_via_vision-language_models.md)
- [\[NeurIPS 2025\] SD-VLM: Spatial Measuring and Understanding with Depth-Encoded Vision-Language Models](../../NeurIPS2025/multimodal_vlm/sd-vlm_spatial_measuring_and_understanding_with_depth-encoded_vision-language_mo.md)

</div>

<!-- RELATED:END -->
