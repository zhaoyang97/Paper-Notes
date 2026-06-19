---
title: >-
  [论文解读] From Objects to Anywhere: A Holistic Benchmark for Multi-level Visual Grounding in 3D Scenes
description: >-
  [NeurIPS 2025][3D视觉][3D视觉定位] 提出 Anywhere3D-Bench，首个涵盖区域/空间/物体/部件四个层级的 3D 视觉定位基准，揭示即使最强的 Gemini-2.5-Pro 和 o3 在空间级任务上仅达约 30% 准确率、部件级约 40%，远低于人类的 95%。 领域现状：3D 视觉定位（vi…
tags:
  - "NeurIPS 2025"
  - "3D视觉"
  - "3D视觉定位"
  - "多层级定位"
  - "benchmark"
  - "空间推理"
  - "MLLM"
---

# From Objects to Anywhere: A Holistic Benchmark for Multi-level Visual Grounding in 3D Scenes

**会议**: NeurIPS 2025  
**arXiv**: [2506.04897](https://arxiv.org/abs/2506.04897)  
**代码**: [https://anywhere-3d.github.io](https://anywhere-3d.github.io)  
**领域**: 3D视觉 / 多模态VLM / 视觉定位  
**关键词**: 3D视觉定位, 多层级定位, benchmark, 空间推理, MLLM

## 一句话总结
提出 Anywhere3D-Bench，首个涵盖区域/空间/物体/部件四个层级的 3D 视觉定位基准，揭示即使最强的 Gemini-2.5-Pro 和 o3 在空间级任务上仅达约 30% 准确率、部件级约 40%，远低于人类的 95%。

## 研究背景与动机

**领域现状**：3D 视觉定位（visual grounding）已在物体级别取得显著进展，现有基准（ScanRefer、Nr3D、Sr3D）和方法主要关注将语言表达对齐到场景中的物体。

**现有痛点**：(a) 现有基准仅限于物体级别，忽略了物体之外的空间区域（如"把台灯放在扶手椅旁边"需要推理未被占据的空间）；(b) 对物体部件级（拉开抽屉）和活动区域级（协同学习区）的定位未被评估；(c) 现有基准很少测试模型对定量尺寸和距离的推理能力。

**核心矛盾**：人类能自然地在 3D 场景中"任意位置"进行视觉定位（估计尺寸、空间关系、物体部件），但当前模型和基准都局限于物体级语义。

**本文目标**：构建一个全面评估模型在四个层级（area/space/object/part）3D 视觉定位能力的基准。

**切入角度**：设计分层的表达类型（活动→空间→物体→部件），每个层级包含多种子类型来考察不同能力。

**核心 idea**：通过多层级多类型的定位基准，系统暴露当前 3D 视觉定位模型在空间推理和细粒度感知上的根本不足。

## 方法详解

### 整体框架
Anywhere3D-Bench 包含 2,886 个表达-3D 包围盒对，来自 276 个场景（ScanNet、MultiScan、3RScan、ARKitScenes），分为四个层级。数据通过 GPT-4o 生成表达 + 人工标注包围盒 + 迭代验证的流程创建。

### 关键设计

1. **四层级视觉定位体系**:

    - **区域级（Area, 189 条）**：描述室内活动区域（如"协同学习区"），需推理多个物体组成的功能区
    - **空间级（Space, 1,209 条）**：指向场景中物体之外的空间区域，包含 5 种子类型——尺寸（调整物体大小）、距离（指定距离放置）、情境（自我中心视角推理）、常识（如"40寸电视"）、轨迹（沿路径放置终点物体）
    - **物体级（Object, 954 条）**：类似此前基准但特别强调定量尺寸/形状/距离推理
    - **部件级（Part, 534 条）**：指向物体的特定部件，包含运动（预测部件移动后的位置）、关系、功能三种类型

2. **自适应 IoU 评估指标**:

    - 功能：针对不同层级的几何特性设计分段 IoU 公式
    - 核心思路：区域级使用 2D XY 平面 IoU（忽略高度）；当某一维度 ground truth 尺寸极小（< 阈值 t）时，使用另外两个维度的 2D IoU 并检查该小维度的中心距离和预测尺寸是否在阈值内；其他情况使用标准 3D IoU。主指标为 $\mathrm{Acc}@0.25\mathrm{IoU}$
    - 设计动机：空间级定位可能涉及平面区域（如放一本书），某一维度极薄，标准 3D IoU 会过于严苛

3. **数据生成与质量控制流程**:

    - 功能：GPT-4o 生成 + 人工标注 + 迭代验证
    - 核心思路：为每个场景构建 3D 场景图（含物体标签、ID、包围盒、描述、关系），按照特定层级和类型的人工 prompt 让 GPT-4o 生成表达。标注员在交互界面中调整 3D 包围盒（支持缩放移动和距离测量），并可修改表达。所有标注经二次人工验证
    - 设计动机：保证每条表达与唯一包围盒无歧义对应

### 三类基线模型评估

评估了三类模型：(1) **LLM**（纯文本输入场景图）：GPT-4.1、o4-mini、Qwen 系列、DeepSeek；(2) **MLLM**（文本+BEV+视频帧）：GPT-4.1、o3、Gemini-2.5-Pro 等；(3) **3D 视觉定位专家模型**（点云+视频帧）：3D-VisTA、PQ3D、Chat-Scene、Grounded 3D-LLM。

## 实验关键数据

### 主实验（Acc@0.25 IoU）

| 模型 | 类型 | Area↑ | Space↑ | Object↑ | Part↑ | Overall↑ |
|------|------|-------|--------|---------|-------|----------|
| GPT-4.1 | LLM | 76.19 | 17.28 | 48.00 | 22.94 | 32.34 |
| DeepSeek-R1-671B | LLM(thinking) | 71.96 | 14.61 | 47.76 | 20.92 | 30.49 |
| GPT-4.1 | MLLM | 81.48 | 19.03 | 53.88 | 25.85 | 35.90 |
| o3 | MLLM(thinking) | 87.83 | 31.26 | 60.27 | 38.77 | 45.94 |
| **Gemini-2.5-Pro** | **MLLM(thinking)** | **83.60** | **29.86** | **64.47** | **38.77** | **46.47** |
| Chat-Scene* | 3D专家 | 49.10 | 6.55 | 31.73 | 22.99 | 22.90 |
| **Human** | — | **100** | **92** | **98** | **97** | **95** |

### 细分分析：空间级子类型

| 表达类型 | Gemini-2.5-Pro | o3 | GPT-4.1(MLLM) | 说明 |
|----------|---------------|-----|----------------|------|
| Size | ~50% | ~45% | ~40% | 相对简单 |
| Distance | ~25% | ~28% | ~15% | 需距离推理 |
| Situation | ~25% | ~30% | ~15% | 需第一人称视角转换 |
| Commonsense | ~30% | ~32% | ~20% | 需常识推理 |
| Trajectory | ~15% | ~20% | ~10% | 最难：需综合理解 |

### 关键发现
- **空间级是最大瓶颈**：最强模型仅约 30%，人类 92%，差距 60+ 百分点
- **Thinking 模型显著优于 non-thinking**：Gemini-2.0-Flash thinking vs non-thinking 差距约 9%；Qwen3-32B thinking 在 space 上从 9.6% 提升到 12.57%
- **视觉输入帮助有限**：LLM→MLLM 在 object 级平均提升 8.19%，但 space 级仅 3.47%
- **3D 专家模型表现最差**：Chat-Scene 在 space 级仅 6.55%，远低于 MLLM
- **错误分析（Gemini-2.5-Pro）**：space 级错误主要是语言推理+空间推理错误；part 级主要是视觉感知错误

## 亮点与洞察
- **四层级定位体系**：超越物体级的全面评估框架，space 级定位是全新的评估维度。这种分层设计思路可迁移到具身智能和机器人操作评估
- **暴露 MLLM 的 3D 空间推理短板**：即使 o3+视觉思维，在从 2D 图像理解 3D 空间关系（尤其是方向映射：左/右→空间轴）仍很弱
- **Thinking 能力对空间推理至关重要**：thinking 模型通过显式推理步骤（如"厚度是垂直于墙面的维度"）获得更准确的空间理解
- 人类 vs 模型的巨大差距表明当前 AI 的 3D 空间智能仍有基础性不足

## 局限与展望
- 基准规模相对较小（2,886 条），space 级虽最多（1,209）但子类型分布不均
- 仅涵盖室内场景，未验证室外场景
- 评估依赖特定 IoU 阈值，不同阈值下排名可能不同
- 未评估将 3D 输入（点云）直接提供给 MLLM 的方案
- 数据生成依赖 GPT-4o，可能存在表达偏差
- 可作为空间推理增强训练数据来改进 MLLM

## 相关工作与启发
- **vs ScanRefer/Nr3D**：经典物体级基准，Chat-Scene 在 ScanRefer 上 >50% 但在本基准物体级仅 31.73%，说明本基准对定量推理要求更高
- **vs MMScan**：包含区域级但无空间级和部件级
- **vs SceneFun3D**：有部件级但仅限于预定义功能元素（把手、按钮），本基准涵盖更开放的物体部件并测试运动推理
- **vs VSI-Bench/Space3D-Bench**：视频级空间推理基准，不涉及 3D 包围盒预测
- 该基准可作为评估未来具身智能系统空间理解能力的核心测试集

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个四层级 3D 视觉定位基准，space 级定位是全新评估维度
- 实验充分度: ⭐⭐⭐⭐⭐ 20+ 模型（LLM/MLLM/3D专家）× 四层级 × 多子类型 + 人类评估 + 错误分析
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，图表信息量大，定性案例分析直观
- 价值: ⭐⭐⭐⭐⭐ 系统暴露了当前 AI 在 3D 空间推理中的根本不足，对社区有重要指引意义

## 与相关工作的对比

## 启发与关联

## 评分

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] S$^2$-MLLM: Boosting Spatial Reasoning Capability of MLLMs for 3D Visual Grounding with Structural Guidance](../../CVPR2026/3d_vision/s2-mllm_boosting_spatial_reasoning_capability_of_mllms_for_3d_visual_grounding_w.md)
- [\[ICCV 2025\] Articulate3D: Holistic Understanding of 3D Scenes as Universal Scene Description](../../ICCV2025/3d_vision/articulate3d_holistic_understanding_of_3d_scenes_as_universal_scene_description.md)
- [\[CVPR 2026\] EG-3DVG: Expression and Geometry Aware Grounding Decoder for 3D Visual Grounding](../../CVPR2026/3d_vision/eg-3dvg_expression_and_geometry_aware_grounding_decoder_for_3d_visual_grounding.md)
- [\[NeurIPS 2025\] OpenLex3D: A Tiered Evaluation Benchmark for Open-Vocabulary 3D Scene Representations](openlex3d_a_tiered_evaluation_benchmark_for_open-vocabulary_3d_scene_representat.md)
- [\[NeurIPS 2025\] HouseLayout3D: A Benchmark and Training-Free Baseline for 3D Layout Estimation in the Wild](houselayout3d_a_benchmark_and_training-free_baseline_for_3d_layout_estimation_in.md)

</div>

<!-- RELATED:END -->
