---
title: >-
  [论文解读] Global-Local Tree Search in VLMs for 3D Indoor Scene Generation
description: >-
  [CVPR 2025][多模态VLM][3D场景生成] 提出全局-局部树搜索算法，利用VLM的空间推理能力，通过层次化场景表示和emoji网格的视觉提示，实现高质量3D室内场景布局生成，在用户研究中平均排名第一。 1. 领域现状：3D室内场景生成的核心挑战是建模物体间合理的空间关系。早期方法（如基于GAN/VAE的数据驱动方…
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "3D场景生成"
  - "视觉语言模型"
  - "树搜索"
  - "室内场景"
  - "布局规划"
---

# Global-Local Tree Search in VLMs for 3D Indoor Scene Generation

**会议**: CVPR 2025  
**arXiv**: [2503.18476](https://arxiv.org/abs/2503.18476)  
**代码**: [https://github.com/dw-dengwei/TreeSearchGen](https://github.com/dw-dengwei/TreeSearchGen)  
**领域**: 多模态VLM  
**关键词**: 3D场景生成, 视觉语言模型, 树搜索, 室内场景, 布局规划

## 一句话总结
提出全局-局部树搜索算法，利用VLM的空间推理能力，通过层次化场景表示和emoji网格的视觉提示，实现高质量3D室内场景布局生成，在用户研究中平均排名第一。

## 研究背景与动机

1. **领域现状**：3D室内场景生成的核心挑战是建模物体间合理的空间关系。早期方法（如基于GAN/VAE的数据驱动方法）受限于3D数据集规模小（3D-FRONT仅18K+），鲁棒性不足。近期方法转向利用VLM的常识推理能力来生成场景布局。
2. **现有痛点**：LayoutGPT等直接用VLM生成布局的方法采用从左到右的自回归推理，无法修正先前的错误输出。HoloDeck和AnyHome虽然生成场景图再转为布局，但其规则式转换算法不完善，导致多样性低且结果不够真实。
3. **核心矛盾**：VLM的链式推理（Chain-like）本质上是贪心策略，一旦某个物体放置不当，错误会不断累积，且无法回溯修正，导致物体超出房间、互相重叠等问题。
4. **本文目标** (1) 如何让VLM具备回溯修正的推理能力？(2) 如何降低复杂场景中搜索树的深度？(3) 如何让VLM准确感知2D俯视图中的空间位置？
5. **切入角度**：人在布置房间时，会逐个放置物体，每个物体有多个候选位置；如果当前放置不合适，会调整之前的决定。这个过程本质上是一个树搜索问题。
6. **核心 idea**：将3D场景生成建模为约束满足的规划问题，用"全局树搜索（物体级）+ 局部树搜索（参数级）"的层次化搜索框架配合VLM推理来求解。

## 方法详解

### 整体框架
输入是文本描述（如"一个带有queen-size床的卧室"），输出是3D室内场景（每个物体的类别、尺寸、位置、朝向）。方法分两阶段：(1) 先用VLM从文本生成层次化场景表示（代理P），将场景分解为房间→区域→地面物体→承载物体四级；(2) 再用全局-局部树搜索算法确定每个物体的具体位置和朝向。搜索过程中，VLM通过emoji网格的视觉提示来感知空间位置。

### 关键设计

1. **层次化场景表示（Hierarchical Scene Representation）**:
    - 功能：将用户文本输入分层转化为结构化的场景描述，作为文本与3D场景之间的语义桥梁。
    - 核心思路：逐级提示VLM生成四个层级——房间层（生成房间尺寸）、区域层（划分功能区域如睡眠区/工作区，各区域共享房间宽度）、地面物体层（每个区域内的主要物体，选定锚点物体并确定其他物体与锚点的空间关系如place_front/place_beside）、承载物体层（桌上物品等）。物体的3D模型通过CLIP视觉相似度+Sentence-BERT文本相似度+尺寸匹配从Objaverse数据库中检索。
    - 设计动机：直接从文本到场景的语义跳跃太大；层次化分解将复杂问题切分为多个独立的小区域子问题，显著降低搜索树深度和计算成本。

2. **全局树搜索（Global Tree Search）**:
    - 功能：管理物体级别的放置顺序和回溯逻辑，确保所有物体都能合理放置。
    - 核心思路：以区域为根节点，每层代表一个物体。先放置锚点物体，其余物体按尺寸从大到小排序依次放置。对每个物体，调用局部树搜索生成候选放置方案。若成功则进入下一层；若失败（尝试k次后仍无法放置），则回溯到上一层重新选择该层的放置方案。搜索采用DFS算法。锚点物体的最大尝试次数$k=3$，其他物体$k=1$。
    - 设计动机：模拟人类布置房间的行为——逐个放置，发现不合适就调整之前的决策。DFS优先探索最有希望的方案，比BFS更高效。

3. **局部树搜索 + Emoji网格视觉提示（Local Tree Search with Emoji Grid）**:
    - 功能：确定单个物体的具体位置参数（侧面选择→行坐标→列坐标三步分解）。
    - 核心思路：将物体放置分解为三步——先确定放在锚点的哪一侧（左/右/上/下），再确定行或列坐标，最后确定另一个轴的坐标。每步通过文本+视觉双重提示VLM，其中视觉提示是将俯视图离散化为一个密集网格，每个单元格填充不同的emoji符号。VLM通过识别emoji名称来指示物体应放置的位置。每步还有VLM评估器检查中间结果是否合理（如碰撞检测）。
    - 设计动机：VLM难以直接输出精确的坐标数值，但能很好地在视觉网格中识别位置关系。Emoji的多样性使每个单元格具有明确的标识，避免VLM混淆相邻位置。参数级分解降低了每步的决策复杂度。

### 损失函数 / 训练策略
本方法是无训练的推理时方法（test-time reasoning），不涉及损失函数或训练策略。使用GPT-4o API作为VLM骨干，通过精心设计的prompt和搜索算法在推理时完成场景生成。约束条件包括：空间范围约束、放置常识约束、无重叠约束、无悬浮约束。

## 实验关键数据

### 主实验
使用ChatGPT生成120条文本提示（4种场景类型×30条），用CLIP分数和用户研究的倒数排名（Reciprocal Rank）评估。

| 方法 | 浴室CLIP | 卧室CLIP | 厨房CLIP | 客厅CLIP | 平均RR |
|------|---------|---------|---------|---------|--------|
| AnyHome | 26.28 | 27.22 | 27.75 | 27.02 | 0.443 |
| HoloDeck | 28.46 | 29.33 | 29.82 | 28.87 | 0.596 |
| **Ours** | **29.37** | **29.93** | **29.58** | **30.18** | **0.793** |

用户研究（15名标注者）中，本文方法的平均倒数排名0.793，即平均排名约1.26（3种方法中最优），相比AnyHome和HoloDeck分别提升+0.360和+0.197。

### 消融实验

| 配置 | 浴室RR | 卧室RR | 厨房RR | 客厅RR | 平均RR |
|------|--------|--------|--------|--------|--------|
| IO (直接输出) | 0.437 | 0.350 | 0.441 | 0.356 | 0.396 |
| CoT (链式推理) | 0.681 | 0.702 | 0.676 | 0.685 | 0.686 |
| **Full (树搜索)** | **0.714** | **0.780** | **0.714** | **0.790** | **0.750** |

### 关键发现
- IO方法表现最差（RR仅0.396），说明VLM的训练数据中缺乏3D布局样本，直接输出布局效果很差。
- CoT继承了层次化场景表示和任务分解，已经取得了不错效果（0.686），但缺乏回溯能力。
- 树搜索在CLIP分数上相比CoT提升有限（+0.21），原因是为控制API成本，设置了较小的k值，搜索空间未被充分探索。
- 在卧室（0.834 RR）和客厅（0.868 RR）等空间关系明显的场景中表现最佳；在厨房中略逊于HoloDeck，因为厨房常沿墙放置物品，不存在明显的空间关系。

## 亮点与洞察
- **Emoji网格作为视觉空间接口**：这是一个非常巧妙的设计——利用emoji的多样性和VLM对emoji的识别能力，将连续空间离散化为可由VLM准确操作的网格。这个trick可以迁移到任何需要VLM输出空间位置的任务。
- **层次化分解策略**：将搜索树分为物体级（全局）和参数级（局部）两层，同时在物体级又通过区域划分降低每个子问题的复杂度，是搜索空间管理的经典思路。
- **训练-free的方案**：完全不需要收集数据或训练模型，通过算法设计和prompt工程最大化利用VLM的常识推理能力，对资源有限的场景特别友好。

## 局限与展望
- 受限于API调用成本，树搜索的宽度k设置较小，未能充分探索搜索空间，在部分场景中与CoT差距不大。
- 目前仅处理室内场景，未扩展到室外场景或AR/VR应用。
- Emoji网格的分辨率有限，可能影响物体放置的精度。
- 物体检索依赖Objaverse数据库，若库中缺少匹配物体则无法处理。
- 未考虑物体的精确几何形状（仅用bounding box），可能在形状复杂物体间仍出现穿模。

## 相关工作与启发
- **vs LayoutGPT**: LayoutGPT直接用LLM以CSS格式输出布局参数，属于chain-like方法，无法修正错误。本文的树搜索有回溯能力，但需要更多API调用。
- **vs HoloDeck**: HoloDeck先用LLM生成场景图，再规则式转换为布局。本文让VLM在视觉空间中直接推理位置，更灵活但依赖VLM的视觉理解质量。
- **vs Tree-of-Thoughts (ToT)**: 本文是ToT在3D场景生成任务上的具体实例化，贡献在于设计了层次化表示和emoji提示来适配这个具体问题。

## 评分
- 新颖性: ⭐⭐⭐⭐ 树搜索+emoji网格的组合颇具创意，但核心框架是ToT的直接应用
- 实验充分度: ⭐⭐⭐ 定量评估仅CLIP分数和用户研究，缺少大规模自动化指标
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，动机阐述充分，图示直观
- 价值: ⭐⭐⭐⭐ 展示了VLM在3D场景规划中的潜力，emoji网格思路有较好的可迁移性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] HOG-Layout: Hierarchical 3D Scene Generation, Optimization and Editing via Vision-Language Models](../../CVPR2026/multimodal_vlm/hog_layout_hierarchical_3d_scene_generation_optimization_and_editing.md)
- [\[ICCV 2025\] Global and Local Entailment Learning for Natural World Imagery](../../ICCV2025/multimodal_vlm/global_and_local_entailment_learning_for_natural_world_imagery.md)
- [\[ICML 2026\] Pair2Scene: Learning Local Object Relations for Procedural Scene Generation](../../ICML2026/multimodal_vlm/pair2scene_learning_local_object_relations_for_procedural_scene_generation.md)
- [\[CVPR 2025\] Improving Personalized Search with Regularized Low-Rank Parameter Updates](improving_personalized_search_with_regularized_low-rank_parameter_updates.md)
- [\[CVPR 2025\] SldprtNet: A Large-Scale Multimodal Dataset for CAD Generation in Language-Driven 3D Design](sldprtnet_a_large-scale_multimodal_dataset_for_cad_generation_in_language-driven.md)

</div>

<!-- RELATED:END -->
