---
title: >-
  [论文解读] MapEval: A Map-Based Evaluation of Geo-Spatial Reasoning in Foundation Models
description: >-
  [遥感] 提出 MapEval 基准，通过 700 道涵盖文本、API 和视觉三类任务的多选题，系统评估 30 个基础模型在地图场景下的地理空间推理能力，发现最强模型准确率不超过 67%，且所有模型落后人类表现 20% 以上。 基础模型（如 GPT-4o、Claude-3.5-Sonnet、Gemini-1.5-Pro）在自…
tags:
  - "遥感"
---

# MapEval: A Map-Based Evaluation of Geo-Spatial Reasoning in Foundation Models

- **会议**: ICML 2025 (Spotlight)
- **arXiv**: [2501.00316](https://arxiv.org/abs/2501.00316)
- **代码**: [GitHub - MapEval](https://github.com/MapEval)
- **领域**: 遥感 / 地理空间推理
- **关键词**: Geo-Spatial Reasoning, Benchmark, Foundation Models, Map-Based QA, LLM Evaluation

## 一句话总结

提出 MapEval 基准，通过 700 道涵盖文本、API 和视觉三类任务的多选题，系统评估 30 个基础模型在地图场景下的地理空间推理能力，发现最强模型准确率不超过 67%，且所有模型落后人类表现 20% 以上。

## 研究背景与动机

基础模型（如 GPT-4o、Claude-3.5-Sonnet、Gemini-1.5-Pro）在自然语言推理、工具使用等方面取得了显著进展，但其在**地图场景下的地理空间推理**能力尚未被充分探索。现有的地理空间 QA 基准存在以下不足：

**任务类型单一**：大多聚焦于简单的位置查询（如 POI 检索），缺乏对复杂空间关系、导航规划等任务的覆盖

**缺少多模态评估**：未同时考察文本、视觉和 API 交互三种模态下的推理能力

**地理覆盖不足**：现有数据集往往集中于少数城市/国家，缺乏全球多样性

**缺少工具交互评估**：真实的地图使用场景涉及 API 调用（如 Google Maps API），但现有基准未涉及此类评估

MapEval 的动机是构建一个**全面、多模态、全球覆盖**的地图推理基准，系统揭示当前基础模型在空间推理方面的短板。

## 方法详解

### 整体框架

MapEval 包含三个子任务，分别评估模型在不同信息输入条件下的地理空间推理能力：

| 子任务 | 输入形式 | 评估重点 | 题目数量 |
|---|---|---|---|
| MapEval-Textual | 结构化文本（地名、坐标、营业时间等） | 长上下文推理、空间关系理解 | 300 |
| MapEval-API | 模型通过工具函数调用地图 API | Agent 工具使用、API 交互推理 | 300 |
| MapEval-Visual | 地图截图（Google Maps 视觉快照） | 视觉地图理解、地图信息提取 | 100 |

整体数据集覆盖 **180 个城市、54 个国家**，共 700 道多选题。

### 关键设计

**1. 问题类别体系**

MapEval 将地图推理任务划分为 5 大类别：

- **Place Info**：关于特定地点的属性信息（评分、营业时间、地址等）
- **Nearby**：基于空间邻近性的 POI 搜索和推荐
- **Route/Navigation**：路线规划、距离计算、导航方向判断
- **Trip**：多站点旅行规划、时间预算、行程优化
- **Unanswerable**：信息不足、无法作答的问题（考察模型拒答能力）

**2. 数据构建流程**

- 由专家标注员基于 Google Maps 手动创建问题，确保真实性和多样性
- 使用 MapQaTor 工具缓存 API 调用结果，构建静态评估数据库，确保可复现性
- 通过 LLM 过滤器剔除可仅凭预训练知识回答的简单问题（无上下文基线仅 6.67% 准确率）

**3. MapEval-API 的 Agent 评估框架**

在 API 任务中，模型作为 Agent 可调用以下简化工具函数：
- `PlaceDetailsTool(placeId)` — 获取地点详情
- `NearbySearchTool(location, keyword, radius)` — 搜索附近地点
- `TravelTimeTool(origin, destination, travelMode)` — 查询出行时间
- `DirectionsTool(origin, destination)` — 获取导航路线

这些工具封装了实际的 Google Maps API 调用，减少了 API 参数变异导致的评估偏差。

### 评估指标

采用多选题准确率（Accuracy）作为主要指标，按子任务和类别分别统计，并与人类表现进行对比。

## 实验关键数据

### 主实验：30 个模型的整体表现

| 模型 | Textual 总体 | Place Info | Nearby | Route | Trip | Unans. |
|---|---|---|---|---|---|---|
| Claude-3.5-Sonnet | **66.33** | 73.44 | 73.49 | 75.76 | 49.25 | 40.00 |
| GPT-4o | ~64 | — | — | — | — | — |
| Gemini-1.5-Pro | 66.33 | 65.63 | 74.70 | 69.70 | 47.76 | 85.00 |
| Llama-3.2-90B | 58.33 | 68.75 | 66.27 | 66.67 | 38.81 | 30.00 |
| Gemma-2.0-27B | 49.00 | 39.07 | 71.08 | 59.09 | 31.34 | 15.00 |
| 人类表现 | **>86** | — | — | — | — | 65.00 |

**MapEval-API 关键发现**：Claude-3.5-Sonnet Agent 在 API 任务中分别超出 GPT-4o 和 Gemini-1.5-Pro 约 **16%** 和 **21%**，开源模型差距更大。

### 消融与深入分析

| 分析维度 | 关键发现 |
|---|---|
| 无上下文基线 | Claude-3.5-Sonnet 在无上下文时仅 6.67%，证明外部上下文必要 |
| 开放式 vs MCQ | 开放式回答准确率显著低于 MCQ（Textual: 55.33% vs 66.33%） |
| 微调效果 | 在 MapEval-Textual 上微调开源小模型（Phi-3.5-mini、Llama-3.2-3B 等）提升不足 5% |
| 大型 VLM | Qwen2.5-VL-72B 在 Visual 任务达 60.35%，缩小了与闭源模型的差距（vs 61.65%） |
| 计算器辅助 | 添加计算器工具可改善涉及距离/时间计算的题目表现 |

### 关键发现

1. **所有模型准确率均不超过 67%**，最强闭源模型与人类仍有 **20%+** 差距
2. **开源模型显著落后闭源模型**，尤其在 API 交互和视觉推理任务中
3. 模型在**距离估算、方向判断、路线规划**方面表现最弱
4. **Unanswerable 类别**表现两极化：Claude-3.5-Sonnet（90%）远超人类（65%），因为模型严格依赖上下文而人类倾向猜测
5. 微调小模型无法显著改善表现，说明问题根源在于模型**地理空间推理能力的根本缺陷**

## 亮点与洞察

1. **三合一评估框架独特**：首次在同一基准中整合文本、API 和视觉三种地图推理模态，提供了全面的评估视角
2. **Agent 评估设计精巧**：通过封装工具函数和缓存 API 响应，既保证了评估的真实性又确保了可复现性
3. **揭示了根本能力缺陷**：微调实验证明性能差距并非训练数据不足，而是模型在空间推理能力上的根本局限
4. **MapQA 生态系统**：MapEval 是更大的 MapQA 生态的一部分（MapQaTor → MapEval → MapAgent），形成了从数据构建到评估到 Agent 的完整研究链

## 局限性

1. **数据规模有限**：700 道题在某些细分类别中样本量不足（如 Place Info 仅 64 题），模型间 1-2% 的差异缺乏统计显著性
2. **MCQ 格式限制**：多选题无法完全反映真实场景中的开放式空间推理需求
3. **地图来源单一**：仅基于 Google Maps，未覆盖 OpenStreetMap、百度地图等其他地图服务
4. **时效性问题**：地图数据会随时间变化（如商店关闭、道路变更），静态缓存可能导致评估偏差
5. **缺少遥感影像**：仅使用数字地图截图，未包含卫星影像等遥感模态

## 相关工作与启发

- **MapQaTor (ACL 2025)**：MapEval 的数据标注工具，用于缓存 API 调用并构建静态评估数据库
- **TravelPlanner (ICML 2024)**：另一个关注旅行规划的基准，但仅聚焦于文本推理
- **GeoQuestions1089**：早期地理空间 QA 数据集，但缺乏多模态和工具交互评估
- **MapAgent (EACL 2026)**：后续工作，在 MapEval 基础上构建更强的地图推理 Agent

**启发**：地理空间推理是基础模型的一个关键薄弱环节，未来可考虑结合遥感影像、GIS 工具和结构化地理知识图谱来增强模型的空间理解能力。

## 评分

- **创新性**: ⭐⭐⭐⭐ — 首个综合文本/API/视觉三模态的地图推理基准
- **实用性**: ⭐⭐⭐⭐ — 直接服务于导航、旅行规划等高频应用场景的评估
- **实验充分度**: ⭐⭐⭐⭐⭐ — 评估了 30 个模型，包含多维度消融分析
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，实验详实

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Causal Foundation Models: Disentangling Physics from Instrument Properties](causal_foundation_models_disentangling_physics_from_instrument_properties.md)
- [\[AAAI 2026\] Consistency-based Abductive Reasoning over Perceptual Errors of Multiple Pre-trained Models in Novel Environments](../../AAAI2026/remote_sensing/consistency-based_abductive_reasoning_over_perceptual_errors_of_multiple_pre-tra.md)
- [\[CVPR 2026\] WRIVINDER: Towards Spatial Intelligence for Geo-locating Ground Images onto Satellite Imagery](../../CVPR2026/remote_sensing/wrivinder_towards_spatial_intelligence_for_geo-locating_ground_images_onto_satel.md)
- [\[CVPR 2026\] IMAIA: Interactive Maps AI Assistant for Travel Planning and Geo-Spatial Intelligence](../../CVPR2026/remote_sensing/imaia_interactive_maps_ai_assistant_for_travel_planning_and_geo-spatial_intellig.md)
- [\[CVPR 2026\] GeoBridge: A Semantic-Anchored Multi-View Foundation Model Bridging Images and Text for Geo-Localization](../../CVPR2026/remote_sensing/geobridge_a_semantic-anchored_multi-view_foundation_model_bridging_images_and_te.md)

</div>

<!-- RELATED:END -->
