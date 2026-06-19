---
title: >-
  [论文解读] HyperGLM: HyperGraph for Video Scene Graph Generation and Anticipation
description: >-
  [CVPR 2025][视频理解][视频场景图] HyperGLM 提出将实体场景图（捕捉空间关系）和程序图（建模因果时序转换）统一为超图 (HyperGraph)，并将其注入多模态 LLM 实现视频场景图的生成、预测和推理，同时发布包含 190 万帧的 VSGR 数据集支持五类任务。 领域现状 领域现状：视频场景图生成 (…
tags:
  - "CVPR 2025"
  - "视频理解"
  - "视频场景图"
  - "超图"
  - "大语言模型"
  - "关系推理"
  - "场景图预测"
---

# HyperGLM: HyperGraph for Video Scene Graph Generation and Anticipation

**会议**: CVPR 2025  
**arXiv**: [2411.18042](https://arxiv.org/abs/2411.18042)  
**代码**: [https://uark-cviu.github.io/projects/HyperGLM](https://uark-cviu.github.io/projects/HyperGLM)  
**领域**: 视频理解/场景图生成  
**关键词**: 视频场景图, 超图, 大语言模型, 关系推理, 场景图预测

## 一句话总结

HyperGLM 提出将实体场景图（捕捉空间关系）和程序图（建模因果时序转换）统一为超图 (HyperGraph)，并将其注入多模态 LLM 实现视频场景图的生成、预测和推理，同时发布包含 190 万帧的 VSGR 数据集支持五类任务。

## 研究背景与动机

### 领域现状

**领域现状**：视频场景图生成 (VidSGG) 旨在建模视频帧间多物体关系，是自动驾驶、智能监控、视频问答等高层任务的基础。近年来利用 Transformer 和时空上下文的方法取得了进展。

**现有痛点**：(1) 传统场景图方法仅建模成对 (pairwise) 的物体关系，无法表达高阶多物体交互（如"一个人坐在沙发上拿着吉他弹奏"涉及人、沙发、吉他三者的链式关系）；(2) 渐进特征融合和批处理 Transformer 方法无法捕捉长程时序依赖；(3) 现有数据集仅支持场景图生成和预测两个任务，缺少推理能力的评估（VQA、视频描述、关系推理）。

**核心矛盾**：现实视频中的交互关系本身是多对多的、高阶的、时序演化的，但传统图结构只能表达成对连接，导致表达能力不足。

**本文目标** (1) 设计能表达高阶关系的统一图结构；(2) 将结构化图知识注入 LLM 实现推理；(3) 提供全面的评估基准。

## 方法详解

### 整体框架

HyperGLM 包含五个组件：图像编码器、MLP 投影器、时序聚合器、统一超图、语言模型。处理流程：(1) 图像编码器逐帧提取特征 → MLP 投影到语言空间；(2) 时序聚合器压缩 T×N 个 embedding；(3) 基于检测到的物体构建实体场景图和程序图 → 通过随机游走算法统一为超图；(4) 超图以 token 形式注入 LLM，LLM 自回归生成答案。

### 关键设计

1. **统一超图 (Unified HyperGraph)**：
    - 功能：融合空间关系和时序因果关系到一个统一表示中
    - 核心思路：超图由两部分构成——
        - **实体场景图**：每帧的物体及其成对关系（subject → relation → object）
        - **程序图**：关系之间的因果转换概率（如 "holding" → "playing" 的转换频率），通过统计训练数据中相邻帧的关系变化计算
    - 超图的核心优势：超边 (hyperedge) 可以连接多个节点（而非仅两个），从而自然表达高阶关系，如"人坐在沙发上、手持吉他、弹奏吉他"可以用一条超边表达

2. **随机游走超图构建算法**：
    - 功能：从统一超图中采样代表性子结构，生成新的超边
    - 核心思路：在超图上交替执行"节点→超边→节点"的随机游走，每次游走收集经过的所有节点组成新超边。参数 Nw（游走次数）和 Nl（游走长度）控制超边数量和深度，实验表明 Nw=60、Nl=7 效果最优
    - 设计动机：精确的子图匹配是 NP-hard 问题，随机游走提供了高效的近似方案，同时能捕捉跨帧的高阶连接模式

3. **程序图的因果转换概率**：
    - 功能：建模关系在时间维度上的演化规律，支持场景图预测 (SGA)
    - 核心思路：统计训练集中相邻帧的关系转换频率，归一化为概率分布。预测时选择最可能的下一步关系。移除自环（同一关系保持不变），使概率聚焦在关系变化上

### 损失函数

场景图生成和预测任务使用交叉熵损失最小化预测关系类别与真实标签的负对数似然。LLM 部分使用标准的自回归语言建模损失。训练采用 LoRA（rank=128, scaling=256）对 Mistral-7B-Instruct 微调，4×GPU 约 6 小时。

## 实验关键数据


### 主实验

| 任务/数据集 | HyperGLM | 最佳基线 | 指标 |
|------|------|------|------|
| SGA@R50 (Action Genome, F=0.5) | **53.5** | 51.4 (SceneSayerSDE) | Recall@50 |
| SGA@mR50 (Action Genome, F=0.5) | **40.5** | 39.9 (SceneSayerSDE) | mRecall@50 |
| SGA@R50 (Action Genome, F=0.9) | **50.0** | 47.4 (SceneSayerSDE) | Recall@50 |
| SGA@mR50 (Action Genome, F=0.9) | **38.0** | 37.1 (SceneSayerSDE) | mRecall@50 |
| VSGR 数据集规模 | **190 万帧** | ASPIRe: 160 万帧 | 帧数 |
| VSGR 支持任务数 | **5 个** | 其他数据集: ≤3 个 | SGG+SGA+VQA+VC+RR |
| 最优超边数量 | 60 | - | Nw=60, Nl=7 |
| 训练配置 | 4×GPU, 约6小时 | - | LoRA rank=128 |
| VQA 问答对数 | 74,856 | - | 约 20 问题/视频 |
| VC 描述对数 | 82,532 | - | 约 22 描述/视频 |
| RR 推理任务数 | 61,120 | - | 约 16 任务/视频 |

## 亮点与洞察

1. **超图是表达视频中高阶关系的自然选择**：传统成对图无法表达"人-物体1-物体2"的链式交互，超图的超边天然支持多实体连接，这一建模方式的优势在实验中得到了一致验证
2. **实体图 + 程序图的统一是精巧的设计**：空间关系（"谁在和谁交互"）和时序演化（"关系如何变化"）被分别建模后统一，类似于知识图谱中 schema 和 instance 的结合
3. **关系转换概率的统计方法简单有效**：不需要学习复杂的时序模型，仅通过统计训练集中的转换频率就能为预测提供可靠的先验，且可以减少对低频关系类别的偏见
4. **VSGR 数据集的全面性**：首次支持 SGG、SGA、VQA、VC、RR 五个任务，且涵盖第三人称、自我中心、无人机三种视角

## 局限与展望

1. 随机游走的超参数（Nw, Nl）需要手动调整，不同数据集可能需要不同设置
2. 程序图的转换概率是全局统计的，对于特定场景或稀有交互可能不够准确
3. 超图构建和随机游走增加了推理时的计算开销
4. LLM 的推理成本较高，实际部署时可能不如轻量级方法高效

## 相关工作

- **场景图生成**：STTran（时空Transformer）、DSGDetr（基于DETR的场景图检测）、SceneSayer（ODE/SDE建模时序演化）、ASPIRe（大规模空间感知场景图）
- **超图应用**：HyperGraph Convolution（超图卷积）、HyperGraph Attention（超图注意力）、事故预测和群体活动识别中的超图方法
- **多模态 LLM**：LLaVA/Video-LLaVA（视觉语言推理）、Mistral-7B（基础语言模型）、CLIP-ViT（视觉编码）
- **数据集**：Action Genome（234K帧，SGG+SGA）、PVSG（153K帧，SGG+VQA+VC）、ASPIRe（160万帧，仅SGG）、SportsHHI（11.4K帧，仅SGG）
- **开放词汇方法**：利用视觉-语言模型处理未见过的物体和关系类别，增强泛化能力

## 评分

- 新颖性：⭐⭐⭐⭐（超图+LLM 的组合在视频场景图领域是全新尝试）
- 实用性：⭐⭐⭐⭐（VSGR 数据集支持五类任务有广泛价值）
- 技术深度：⭐⭐⭐⭐（统一超图设计有理论基础，随机游走算法有数学性质保证）
- 表达清晰度：⭐⭐⭐（内容丰富但结构复杂，多任务评估增加了理解成本）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] VGEnt: Graph-Based Retrieval-Reasoning-Augmented Generation for Long Video Understanding](../../NeurIPS2025/video_understanding/vgent_graph-based_retrieval-reasoning-augmented_generation_for_long_video_unders.md)
- [\[ICML 2025\] Fine-Grained Captioning of Long Videos through Scene Graph Consolidation](../../ICML2025/video_understanding/fine-grained_captioning_of_long_videos_through_scene_graph_consolidation.md)
- [\[CVPR 2025\] EgoTextVQA: Towards Egocentric Scene-Text Aware Video Question Answering](egotextvqa_towards_egocentric_scene-text_aware_video_question_answering.md)
- [\[CVPR 2025\] GG-SSMs: Graph-Generating State Space Models](gg-ssms_graph-generating_state_space_models.md)
- [\[CVPR 2026\] Hypergraph-State Collaborative Reasoning for Multi-Object Tracking](../../CVPR2026/video_understanding/hypergraph-state_collaborative_reasoning_for_multi-object_tracking.md)

</div>

<!-- RELATED:END -->
