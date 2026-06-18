---
title: >-
  [论文解读] Taming the Untamed: Graph-Based Knowledge Retrieval and Reasoning for MLLMs to Conquer the Unknown
description: >-
  [ICCV 2025][多模态VLM][多模态知识图谱] 以《怪物猎人：世界》为测试平台，构建了包含文本、图像、视频和复杂实体关系的多模态知识图谱(MH-MMKG)，设计了238个复杂查询和多智能体知识检索方法，揭示了当前MLLM在领域特定任务中的知识检索与推理能力不足。 领域现状 近年来MLLM（如GPT-4o、Claud…
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "多模态知识图谱"
  - "多智能体检索"
  - "MLLM"
  - "知识增强推理"
  - "游戏认知"
---

# Taming the Untamed: Graph-Based Knowledge Retrieval and Reasoning for MLLMs to Conquer the Unknown

**会议**: ICCV 2025  
**arXiv**: [2506.17589](https://arxiv.org/abs/2506.17589)  
**代码**: [github.com/wbw520/MH-MMKG](https://github.com/wbw520/MH-MMKG)  
**领域**: 多模态VLM  
**关键词**: 多模态知识图谱, 多智能体检索, MLLM, 知识增强推理, 游戏认知

## 一句话总结

以《怪物猎人：世界》为测试平台，构建了包含文本、图像、视频和复杂实体关系的多模态知识图谱(MH-MMKG)，设计了238个复杂查询和多智能体知识检索方法，揭示了当前MLLM在领域特定任务中的知识检索与推理能力不足。

## 研究背景与动机

### 领域现状

近年来MLLM（如GPT-4o、Claude）在常识性VQA等通用任务上表现优异，但在领域特定、罕见数据的任务上，仅依靠自身感知和内置知识远远不够，且缺乏可解释性。

### 现有痛点

**多模态RAG的知识冗余与低相关性**：启发式和基于Agent的mRAG方法检索到的知识往往不精确，存在信息冗余

**现有MMKG基准不够挑战**：多数MMKG使用VQA数据集或网络资源构建，这些知识很可能已被现有MLLM学习，无法真正评估MMKG的增强效果

**现有MMKG模态单一**：主要整合文本和图像，缺少视频等更丰富的模态

**训练检索器成本高**：在低资源领域训练有效的图谱检索器不现实，且无法将新图谱整合到闭源模型中

### 核心矛盾

需要一个MLLM确实不具备内置知识的领域来评估MMKG的有效性，同时需要免训练的检索方法来适配闭源模型。

### 切入角度

选择电子游戏《怪物猎人：世界》作为测试平台——游戏的视觉世界由计算机图形生成，与真实世界差异大；游戏世界的知识与现实知识不同，MLLM的内置知识几乎无法帮助回答。

**核心idea**：通过构建游戏领域的多模态知识图谱，配合多智能体自搜索检索器，让MLLM无需训练即可从结构化知识中检索相关信息并增强推理。

## 方法详解

### 整体框架

给定输入查询Q（包含问题q、视频/图像d、辅助信息z），系统分三步工作：(1) 感知器将视觉参考转为文本描述；(2) 多智能体检索器在MH-MMKG上搜索相关知识子图；(3) 摘要器利用检索到的知识生成最终答案。

### 关键设计

#### 1. **MH-MMKG 多模态知识图谱构建**

- **功能**：由3位资深玩家手工构建22个子图（每个怪物一个），合并为完整图谱 $\mathcal{G} = (\mathcal{E}, \mathcal{V}, \mathcal{R})$
- **核心思路**：实体分为怪物实体 $\mathcal{E}_o$ 和其他实体 $\mathcal{E}_a$，边 $v = (e, r, e')$ 表示关系。每个实体附带属性 $A(e) = (c, u)$，包括视频片段（60fps 4K录制）和文本上下文。同时提供人工选择的关键帧和人工撰写的字幕
- **设计动机**：游戏世界的知识高度结构化（攻击动作、连招条件、弱点等），适合用知识图谱表示。多模态属性（视频+文本+图像）使MMKG更贴近复杂的多模态推理需求

#### 2. **多智能体检索器**

- **功能**：三个Agent协作，从知识图谱中自动检索回答问题所需的子图
- **核心思路**：
    - **主题选择Agent** $L$：分析问题确定根实体 $e_0 = L(q, z, \mathcal{E}_o)$
    - **扩展Agent** $W$：以BFS方式扩展子图，判断邻居实体是否对回答有用：$\mathcal{N}(e) = W(e, \mathcal{K}_t; Q, \mathcal{G}, \mathcal{A})$
    - **验证Agent** $U$：检查从根到当前开放实体的路径是否提供了足够知识：$o(e) = U(e, \mathcal{K}_{t+1}; Q, \mathcal{G}, \mathcal{A})$
    - 迭代直到没有开放实体需要扩展：$\mathcal{O}_t = \emptyset$
- **设计动机**：利用MLLM的自搜索能力，无需额外训练即可在图谱上导航检索，适配闭源模型

#### 3. **知识增强推理**

- **功能**：将检索到的子图 $\hat{\mathcal{I}}$ 上的路径转为文本，输入MLLM生成答案
- **核心公式**：$\hat{y} = \text{MLLM}(Q, \aleph(\hat{\mathcal{I}}, \mathcal{G}, \mathcal{A}, \alpha))$
- **设计动机**：模型不需依赖内置知识，只需要纯推理能力。限制路径数量 $\alpha=5$ 以防信息过载

### MH Benchmark 设计

6个子任务覆盖不同认知层次：

| 子任务 | 描述 | 样本数 |
|--------|------|--------|
| I: 个体信息 | 检索怪物的文本信息（昵称、栖息地等） | 24 |
| II: 攻击识别 | 识别怪物正在/即将/完成的攻击动作 | 109 |
| III: 连招预测 | 根据当前动作预测后续攻击序列 | 28 |
| IV: 条件感知 | 检测怪物/环境状态（是否愤怒、地形变化等） | 29 |
| V: 效果洞察 | 分析环境和攻击对怪物状态的影响 | 35 |
| VI: 跨怪物分析 | 比较不同怪物间的攻击模式 | 13 |

三个任务变体：Knowledgeable（完美知识+完美感知）、Perceptive（无知识但完美感知）、Unaided（完全依赖模型自身）。

## 实验关键数据

### 主实验

| 模型 | Vanilla Acc. | Know. Acc. | Unaided-Online Acc. | Online Pre. | Online Rec. |
|------|-------------|-----------|---------------------|-------------|-------------|
| GPT-4o | 0.312 | 0.856 | **0.510** | 0.276 | **0.563** |
| Claude 3.7 Sonnet | 0.283 | 0.899 | 0.439 | **0.291** | 0.403 |
| Claude 3.5 Sonnet | 0.287 | 0.878 | 0.397 | 0.233 | 0.327 |
| Gemini 1.5 Pro | 0.219 | 0.844 | 0.405 | 0.212 | 0.459 |
| Qwen2.5-VL-72B | 0.148 | 0.873 | 0.316 | 0.162 | 0.181 |
| InternVL2.5-78B | 0.160 | 0.865 | 0.308 | 0.156 | 0.238 |
| Human (知识型) | 0.525 | — | 0.903 | 0.921 | 0.854 |

### 消融实验

| 配置 | Acc. | Pre. | Rec. | 说明 |
|------|------|------|------|------|
| 使用5条检索路径 | **0.510** | 0.276 | 0.563 | GPT-4o最优设置 |
| 使用3条路径 | ~0.48 | — | — | 信息不足 |
| 使用10条路径 | ~0.49 | — | — | 信息冗余 |
| BFS搜索（3路径） | **更优** | — | — | 短路径更有效 |
| DFS搜索（3路径） | 较差 | — | — | 长路径初期不利 |
| Human关键帧选择 | 0.510 | 0.276 | 0.563 | 更有信息量 |
| 均匀采样关键帧 | 0.484 | 0.239 | 0.525 | 差距不大但有影响 |

### 关键发现

- **Vanilla设置下所有模型表现极差**（GPT-4o仅0.31），证明游戏领域确实是MLLM的知识盲区
- **Knowledgeable设置下准确率达到~0.9**，证明MH-MMKG包含了足够的知识来回答问题
- **Online vs Offline**：知道"要找什么"时，模型的视觉感知能力显著提升（GPT-4o提升0.1）
- **闭源模型整体优于开源模型**，显示闭源模型内置了更丰富的游戏相关知识
- **模型倾向于检索过多知识**（Recall > Precision），而人类检索更精准

## 亮点与洞察

1. **测试平台选择精妙**：游戏世界既有丰富的结构化知识又确保MLLM缺乏内置知识，是评估知识增强推理的理想场景
2. **多模态KG融合视频**：超越传统的文本+图像MMKG，引入视频属性，更贴近复杂多模态任务
3. **免训练检索方法**：多智能体自搜索无需训练嵌入，可直接用于闭源模型
4. **"知道要找什么"增强感知能力**：Online模式下模型在知道查询和检索上下文后，视觉描述更精准

## 局限与展望

1. **基准规模有限**：仅238个查询，22个怪物子图，扩大规模可能发现更多模式
2. **BFS优于DFS说明查询较简单**：多数问题仅需少数跳步即可回答，未充分测试复杂推理链
3. **视频模态未被充分利用**：当前MLLM无法直接处理视频输入，只能通过关键帧/字幕间接使用
4. **人类基线效果有限**：即使知识型玩家在vanilla设置下也只有0.53准确率，表明视觉领域差异确实很大
5. **单一游戏的代表性**：仅基于一款游戏，泛化性有待验证

## 相关工作与启发

- **GraphRAG** 将图谱检索引入RAG范式，本文是该思路在多模态领域的扩展
- **StructGPT** 利用LLM直接在结构化数据上推理，本文的多智能体设计是其在MMKG上的应用
- 游戏认知是评估AI智能的一个有趣维度，类似思路可扩展到其他虚拟世界（如电影宇宙、小说世界）

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 游戏领域MMKG+多智能体检索是新颖组合，但单个技术组件相对成熟
- **实验充分度**: ⭐⭐⭐⭐ — 14个模型、6个子任务、多种设置，分析全面深入
- **写作质量**: ⭐⭐⭐⭐ — 问题定义清晰，任务变体设计合理，可读性好
- **价值**: ⭐⭐⭐⭐ — 提供了评估MLLM图谱检索推理能力的新平台，基准和代码开源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Training-Free Personalization via Retrieval and Reasoning on Fingerprints](training-free_personalization_via_retrieval_and_reasoning_on_fingerprints.md)
- [\[ICCV 2025\] ReasonVQA: A Multi-hop Reasoning Benchmark with Structural Knowledge for Visual Question Answering](reasonvqa_a_multi-hop_reasoning_benchmark_with_structural_knowledge_for_visual_q.md)
- [\[ICCV 2025\] ChartPoint: Guiding MLLMs with Grounding Reflection for Chart Reasoning](chartpoint_guiding_mllms_with_grounding_reflection_for_chart_reasoning.md)
- [\[ICCV 2025\] Visual-Oriented Fine-Grained Knowledge Editing for MultiModal Large Language Models](visual-oriented_fine-grained_knowledge_editing_for_multimodal_large_language_mod.md)
- [\[ICCV 2025\] GRAB: A Challenging GRaph Analysis Benchmark for Large Multimodal Models](grab_a_challenging_graph_analysis_benchmark_for_large_multimodal_models.md)

</div>

<!-- RELATED:END -->
