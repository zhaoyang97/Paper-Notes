---
title: >-
  [论文解读] Dynamic Group Detection using VLM-augmented Temporal Groupness Graph
description: >-
  [ICCV 2025][多模态VLM][群组检测] 本文提出基于VLM增强的时序群组图（temporal groupness graph）进行视频中的动态人群群组检测，核心创新是用CLIP提取包含人对和背景的groupness-augmented特征来估计成组概率，并通过全帧时序图的Louvain聚类实现动态变化群组的检测。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "群组检测"
  - "CLIP"
  - "时序图聚类"
  - "社交行为分析"
  - "Louvain算法"
---

# Dynamic Group Detection using VLM-augmented Temporal Groupness Graph

**会议**: ICCV 2025  
**arXiv**: [2509.04758](https://arxiv.org/abs/2509.04758)  
**代码**: [https://github.com/irajisamurai/VLM-GroupDetection.git](https://github.com/irajisamurai/VLM-GroupDetection.git)  
**领域**: 多模态VLM  
**关键词**: 群组检测, CLIP, 时序图聚类, 社交行为分析, Louvain算法

## 一句话总结
本文提出基于VLM增强的时序群组图（temporal groupness graph）进行视频中的动态人群群组检测，核心创新是用CLIP提取包含人对和背景的groupness-augmented特征来估计成组概率，并通过全帧时序图的Louvain聚类实现动态变化群组的检测。

## 研究背景与动机

**领域现状**：群组检测（group detection）旨在将视频中观察到的所有人划分为不同的群组，广泛用于群体活动识别、轨迹预测和异常检测。传统方法依赖人的空间位置+面部朝向+姿态等手工特征，近期方法开始用深度学习联合学习这些特征。

**现有痛点**：
   - **局部特征不够**：现有方法独立提取每个人的特征然后比较/合并，忽略了**空间场景上下文**——两个人可能不面对面但在浏览同一个背景物体（如商店橱窗），此时他们是一组的
   - **只做静态群组检测**：现有方法假设视频中群组结构不变，只输出一个固定的群组集合。但现实中人们会在不同群组间流动（如Fig.1b所示，3人组在 $t_1$ 分裂为两组在 $t_2$）
   - **鸡生蛋问题**：要把视频分割成群组结构不变的短片需要先知道群组何时变化，但这正是要检测的内容

**核心矛盾**：需要同时利用空间场景上下文（理解复杂互动）和时序上下文（检测动态变化），但之前方法只能做到其中之一。

**本文目标** 设计能同时利用空间和时序场景上下文的方法，实现视频中动态变化群组的检测。

**切入角度**：(a) 用VLM（CLIP）从包含人对+背景的bounding box直接估计成组概率，天然融合了空间场景上下文；(b) 将所有帧的成组概率构建为时序图，用Louvain算法做全局最优聚类来检测动态群组。

**核心 idea**：用CLIP增强的图像特征捕获空间场景上下文估计两两成组概率，构建全帧时序图后用Louvain聚类实现动态群组检测。

## 方法详解

### 整体框架
如Fig.3所示：对视频每帧中的所有人对，提取轨迹特征和图像特征（GA-CLIP），融合后估计每对的成组概率 $P_g$。所有帧的成组概率构建时序群组图（temporal groupness graph），通过Louvain算法聚类得到动态群组。

### 关键设计

1. **Groupness-Augmented CLIP (GA-CLIP)**:

    - 功能：从包含目标人对和背景的bounding box中提取群组相关的视觉特征
    - 核心思路：
      1) **红圈标注**：在bounding box中用红色圆圈标注目标人对（Fig.2a），引导CLIP关注这两个人而非其他人。受[3]启发但扩展到标注两人
      2) **三分类微调**：在CLIP图像编码器上用三分类任务微调——"a group of people"、"individual people"、"occlusion"。加入遮挡类是因为遮挡会使图像特征不可靠（Fig.4d）
      3) 微调后只保留图像编码器提取特征 $Z_{app} = \psi(I_{app})$，不用分类层
    - 设计动机：预训练CLIP已经具备一定的群组理解能力（Fig.4a-b验证），但对三人以上场景和遮挡情况处理不好。红圈+微调解决这两个问题

2. **轨迹特征提取**:

    - 功能：从人的运动轨迹中提取时序特征
    - 核心思路：两步提取——(1) 帧内：将人对的位置属性（bbox位置、面部朝向等）$X_{traj}$ 通过编码器 $\phi$ 得到帧内特征 $E_{traj}$；(2) 时序：将 $T$ 帧的帧内特征拼接后通过Transformer编码器 $\chi$ 得到时序轨迹特征 $Z_{traj}$
    - 设计动机：运动模式（同步行走、停留位置接近等）是判断群组的重要线索，图像特征无法完全捕获

3. **图像-轨迹融合与成组概率估计**:

    - 功能：融合视觉和轨迹特征估计每对的成组概率
    - 核心思路：将 $Z_{app}$ 和 $Z_{traj}$ 拼接得到 $Z$，通过全连接层+softmax得到成组概率 $R = (P_i, P_g) = SM(\rho(Z))$。当GA-CLIP检测到遮挡时，模型自动更多依赖轨迹特征
    - 设计动机：图像和轨迹提供互补信息，遮挡时图像不可靠需转向轨迹

4. **时序群组图构建与Louvain聚类**:

    - 功能：从帧内成组概率构建跨帧时序图并聚类得到动态群组
    - 核心思路：
        - Step 1：每帧构建帧内群组图，节点为人，边权重为成组概率 $P_g$
        - Step 2：相邻帧间通过身份概率 $P_t$（来自MOT或GT跟踪ID）连接同一个人的节点，构建时序群组图
        - Step 3：用Louvain算法最大化图的模块度来聚类。Louvain不需要预设聚类数量等超参数
    - 设计动机：之前方法要么合并所有帧取平均（无法检测动态变化），要么需要手动指定聚类数。时序图+Louvain既能检测动态变化又自动确定群组数量

### 损失函数 / 训练策略
- GA-CLIP：三分类交叉熵损失微调
- 联合训练：轨迹编码器 $\phi$、时序编码器 $\chi$ 和融合网络 $\rho$ 联合训练，GA-CLIP编码器固定
- 训练目标：成组概率 $R$ 与GT $R_{gt} \in \{(1,0)^T, (0,1)^T\}$ 的交叉熵损失

## 实验关键数据

### 主实验（静态群组检测）

| 方法 | JRDB F1 | Café F1 | 特征类型 |
|------|---------|---------|---------|
| GDet (位置+朝向) | ~72 | ~65 | 手工特征 |
| PCTDM | ~75 | ~68 | DINOv2 |
| SIDNet | ~77 | ~70 | 学习特征 |
| **Ours (GA-CLIP)** | **~82** | **~74** | **GA-CLIP** |

### 消融实验

| 配置 | JRDB F1 | Café F1 | 说明 |
|------|---------|---------|------|
| Full (GA-CLIP + 轨迹 + 时序图) | ~82 | ~74 | 完整模型 |
| w/o GA-CLIP (仅轨迹) | ~75 | ~68 | 视觉特征贡献显著 |
| w/o 红圈标注 | ~79 | ~71 | 红圈引导CLIP注意力有效 |
| w/o 遮挡类 | ~80 | ~72 | 遮挡检测有帮助 |
| w/o 时序图 (静态合并) | ~80 | ~72 | 动态检测需要时序图 |
| 用DINOv2替换CLIP | ~78 | ~70 | CLIP比DINOv2更好 |

### 关键发现
- GA-CLIP特征显著优于DINOv2等视觉特征，验证了VLM对群组理解的优越性
- 红圈标注是让CLIP在多人场景中关注目标人对的关键技巧
- 时序群组图+Louvain实现了动态群组检测，这是之前方法不支持的新能力
- 遮挡检测类使模型在遮挡场景下更鲁棒
- 动态检测结果可以简单地转换为静态检测结果（取帧内聚类即可）

## 亮点与洞察
- **VLM用于群组理解**：巧妙利用CLIP预训练的社交场景理解能力，通过红圈标注引导CLIP关注特定人对。将VLM的零样本能力适配到群组检测是非常新颖的思路
- **动态群组检测新任务**：之前方法都假设群组不变（静态检测），本文首次实现了动态变化群组的检测。时序群组图是一个优雅的形式化方式
- **Louvain聚类**：相比谱聚类和标签传播，Louvain不需要预设聚类数量等超参数，更适合大规模时序图的高效聚类
- **遮挡类设计**：在二分类基础上加入遮挡类，让模型知道什么时候图像特征不可靠从而转向轨迹特征，简单但有效

## 局限与展望
- 依赖准确的人物检测和跟踪作为前置条件
- 红圈标注是手工设计的视觉prompt，更自动化的注意力引导方式值得探索
- Louvain算法在分辨率参数上可能影响聚类粒度
- 在人群非常密集的场景下bbox重叠严重，可能影响GA-CLIP特征质量
- 未使用视频理解VLM（如VideoCLIP），可能进一步提升时序建模能力

## 相关工作与启发
- **vs PCTDM/SIDNet**: 传统方法独立提取每个人的特征，忽略空间场景上下文。本文直接从包含人对+背景的bbox提取特征，天然融合了上下文
- **vs 静态群组检测**: 之前方法通过合并/平均所有帧的结果输出固定群组。本文通过时序图实现了动态检测
- 红圈引导注意力的技巧可以迁移到其他需要VLM关注特定目标的任务

## 评分
- 新颖性: ⭐⭐⭐⭐ VLM+动态群组检测是新颖的组合，但各组件相对标准
- 实验充分度: ⭐⭐⭐⭐ JRDB和Café数据集，静态+动态评估，消融充分
- 写作质量: ⭐⭐⭐⭐ 条理清晰，实验分析详细
- 价值: ⭐⭐⭐⭐ 动态群组检测新任务有实际应用价值，VLM应用思路有启发性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] CLIPSym: Delving into Symmetry Detection with CLIP](clipsym_delving_into_symmetry_detection_with_clip.md)
- [\[ICCV 2025\] Dynamic Multimodal Prototype Learning in Vision-Language Models](dynamic_multimodal_prototype_learning_in_vision-language_models.md)
- [\[CVPR 2026\] Beyond Graph Model: Reliable VLM Fine-Tuning via Random Graph Adapter](../../CVPR2026/multimodal_vlm/beyond_graph_model_reliable_vlm_fine-tuning_via_random_graph_adapter.md)
- [\[ICCV 2025\] NegRefine: Refining Negative Label-Based Zero-Shot OOD Detection](negrefine_refining_negative_label-based_zero-shot_ood_detection.md)
- [\[CVPR 2026\] DynamicGTR: Leveraging Graph Topology Representation Preferences to Boost VLM Capabilities on Graph QAs](../../CVPR2026/multimodal_vlm/dynamicgtr_leveraging_graph_topology_representation_preferences_to_boost_vlm_cap.md)

</div>

<!-- RELATED:END -->
