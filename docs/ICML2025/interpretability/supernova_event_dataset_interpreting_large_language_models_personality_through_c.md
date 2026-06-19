---
title: >-
  [论文解读] Supernova Event Dataset: Interpreting Large Language Models' Personality through Critical Event Analysis
description: >-
  [ICML 2025][可解释性][LLM人格分析] 提出 Supernova Event Dataset（包含传记、历史事件、新闻、科学发现的 Wikipedia 文章），通过让 LLM 从长文本中抽取并排序关键事件，再由另一个 LLM 作为评判者推断目标模型的"人格特质"，揭示不同 LLM 在主观决策中的一致性行为模式差异。
tags:
  - "ICML 2025"
  - "可解释性"
  - "LLM人格分析"
  - "事件抽取与排序"
  - "LLM-as-Judge"
  - "主观推理"
---

# Supernova Event Dataset: Interpreting Large Language Models' Personality through Critical Event Analysis

**会议**: ICML 2025  
**arXiv**: [2506.12189](https://arxiv.org/abs/2506.12189)  
**代码**: 无  
**领域**: 可解释性  
**关键词**: LLM人格分析, 事件抽取与排序, 可解释性, LLM-as-Judge, 主观推理

## 一句话总结

提出 Supernova Event Dataset（包含传记、历史事件、新闻、科学发现的 Wikipedia 文章），通过让 LLM 从长文本中抽取并排序关键事件，再由另一个 LLM 作为评判者推断目标模型的"人格特质"，揭示不同 LLM 在主观决策中的一致性行为模式差异。

## 研究背景与动机

当前 LLM 基准测试主要聚焦于有客观正确答案的任务（如问答、推理），但随着 LLM 被部署到医疗、法律、金融等高风险领域，仅评估事实准确性已不够——理解模型的**主观判断**和**价值倾向**变得至关重要。

已有工作表明 LLM 在被显式 prompt 为特定角色时能模拟人格特质，但本文的核心发现是：**即使没有角色扮演 prompt，LLM 在处理复杂主观任务时也会展现出一致的行为模式**，这些模式可被解读为"人格"。

关键事件识别与排序是一个天然的主观任务：
- 需要跨越长上下文进行推理
- 需要建模因果链和非线性事件交互
- 不同人（和模型）会因为价值观差异而做出不同选择

这使得该任务成为探测 LLM 内在决策倾向的理想工具。

## 方法详解

### 整体框架

框架分为三个阶段：

1. **数据集构建**：构建 Supernova Event Dataset，包含四类 Wikipedia 文章（传记、历史事件、新闻、科学发现）
2. **事件抽取与排序**：目标 LLM 通过 RAG 接收文章，提取并排序 5 个最关键事件
3. **人格评判**：另一个 LLM（Judge）分析目标模型的事件选择和排序，推断其人格类型

### 关键设计

#### 数据集构建（Supernova Event Dataset）

| 类别 | 数据源 | 最低字数 | 最低浏览量 | 额外筛选 | 文章数 |
|------|--------|---------|-----------|---------|--------|
| 传记 | 英文 Wikipedia | 3000 | 50000 | Infobox 模板过滤 | 150 |
| 历史事件 | 英文 Wikipedia | 500 | 5000 | ORES ≥B + LLM 验证 + 年份<2000 | 150 |
| 新闻事件 | 英文 Wikipedia | 500 | 5000 | ORES ≥B + LLM 验证 + 年份>2000 | 150 |
| 科学发现 | Gemini Deep Research | - | - | Nobel Prize API + Gemini 扩写 | 25 |

数据集设计亮点：
- **传记**：要求 ≥3000 词确保覆盖人物完整生涯，需有标准化 infobox 模板
- **历史/新闻事件**：两阶段筛选——先用启发式规则过滤歧义页，再用本地 LLaMA-3-8B 做语义验证（置信度 >0.9）
- **科学发现**：从 Nobel Prize REST API 提取 384 条获奖记录（1901-2024），用 Gemini 2.5 Pro Deep Research 扩写为百科全书式文章

#### RAG 管道与事件抽取

文档处理流程：
1. **分块**：将文档分为 1000 token 的语义块（100 token 重叠）
2. **嵌入**：使用 nomic-embed-text-v1 模型生成高维向量
3. **索引**：存入 FAISS 向量数据库
4. **检索**：MultiQueryRetriever 将查询改写为多个搜索查询，提高召回率

**两阶段 prompt 策略**：
- **第一阶段 prompt**：引导检索器关注"转折点""级联效应"等关键事件特征，而非仅拉取主题相关内容
- **第二阶段 prompt**：引导 LLM 进行结构化分析，要求其识别并排序 5 个最关键事件

对于科学发现类别，额外使用反事实测试（"如果没有这个事件，结果是否改变？"）作为选择标准。

#### 人格评判框架

- **Judge 模型**：使用 Qwen-2.5 14B 作为外部评判者
- **评估方式**：Judge 接收目标 LLM 的完整事件选择和排序输出，分析其决策模式
- **人格编码**：使用 sentence-transformers（all-MiniLM-L6-v2）对识别出的人格特质进行语义嵌入
- **可视化**：对聚合嵌入进行 PCA 降维，在二维空间中展示模型人格位置
- **相似度度量**：使用余弦相似度量化模型间人格相似性

### 损失函数 / 训练策略

本文**不涉及模型训练**，而是一个评估框架。核心要素包括：

- **推理时策略**：结构化 prompt 引导 + RAG 检索增强
- **人格量化**：通过频率加权的特质嵌入聚合
- **科学发现分析**：结合关键词计数 + 开放编码（open coding），收敛出三类决策原则：
    - **因果中心型**（causality-centric）：关注机制和因果路径
    - **赋能中心型**（enablement-centric）：关注基础、障碍消除、验证
    - **综合中心型**（synthesis-centric）：强调概念整合和范式级连接

## 实验关键数据

### 主实验

**被评估模型**：
- 小模型：Phi-4, Orca 2 (13B), Qwen 2.5 (14B)
- 大模型（科学发现类）：Claude Sonnet 3.7, Gemini 2.5 Pro, OpenAI o3

**人格类别分布结果**（七类人格维度）：

| 模型 | 战略成就者 | 创意创新者 | 情感型 | 社区支持 | 意识形态 | 观察型 | 影响者 |
|------|-----------|-----------|-------|---------|---------|-------|-------|
| Phi-4 | **最高** | **高** | 中等 | 低 | 低 | 低 | 低 |
| Orca 2 | 中等 | 低 | **最高** | 中等 | 低 | 低 | 低 |
| Qwen 2.5 | **最高** | 高 | 中等 | 中等 | 中等 | 中等 | 中等 |

**科学发现类模型决策原则分布**：

| 模型 | 因果中心型 | 赋能中心型 | 综合中心型 |
|------|-----------|-----------|-----------|
| o3 | **主导** | 中等 | 低 |
| Gemini 2.5 Pro | 中等 | **主导** | 低 |
| Claude 3.7 Sonnet | 低 | 明显 | **主导** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 电影剧本数据集（1172 部） | 人格模式一致 | 验证人格在不同领域间的稳定性 |
| Phi-4 在电影中的表现 | 战略/情节导向 | 优先选"Jafar 的阴谋""Zuckerberg 创建 Facebook 的决定" |
| Orca 2 在电影中的表现 | 情感/关系导向 | 优先选"Aladdin 遇见 Jasmine""Mark 与 Eduardo 的决裂" |
| Qwen 2.5 在电影中的表现 | 里程碑导向 | 优先选"Facemash 的创建""林肯中心最终演出" |

### 关键发现

1. **模型人格可重现**：不同领域（传记、金融危机、电影剧本、科学发现）下，模型展现一致的行为偏好
2. **小模型差异显著**：Phi-4 偏"战略成就"、Orca 2 偏"情感推理"、Qwen 2.5 最均衡
3. **大模型推理风格分化**：o3 因果推理（step-by-step）、Gemini 实证验证、Claude 概念整合
4. **语义空间分离清晰**：PCA 可视化显示三个小模型占据截然不同的人格区域
5. **无需角色扮演**：人格特质在无显式 personality prompt 的情况下自然涌现

## 亮点与洞察

- **任务设计精巧**：关键事件排序是一个天然主观的任务，不存在唯一正确答案，因此能直接反映模型的价值偏好，这比传统基准测试更深层地探测模型行为
- **Prompt-agnostic**：本文的人格识别方法不依赖于特定 prompt 设计，模型的行为模式在不同 prompt 下保持一致
- **科学发现分析有启发性**：三类推理原则（因果/赋能/综合）提供了选择 LLM 的实用参考——需要因果分析用 o3，需要方法论基础评估用 Gemini，需要跨领域概念整合用 Claude
- **反事实测试**：用"如果没有这个事件，结果是否改变？"来筛选关键事件，方法论上很严谨
- **对 AI 辅助科研有意义**：理解 LLM 的推理人格有助于设计更好的人机协作科研工作流

## 局限与展望

1. **数据偏差**：Wikipedia 天然存在编辑偏见和西方中心主义，可能影响人格标签推断
2. **LLM-as-Judge 偏差**：评判者模型自身存在风格偏好（stylistic bias），缺乏人类验证
3. **人格框架非标准化**：人格类别是经验性推导的，未基于 Big Five 等成熟心理学框架
4. **科学发现样本量小**：仅 25 篇文章，统计显著性有限
5. **缺乏对抗性测试**：未检验模型人格在对抗性 prompt 下是否仍然稳定
6. **未考虑温度等推理参数的影响**：不同采样策略可能改变事件选择
7. **评判者单一**：仅用 Qwen 2.5 作为 judge，未使用多 judge 委员会交叉验证

## 相关工作与启发

- **LLM 人格研究**：Jiang et al. (2023) 和 Bodroža et al. (2024) 使用 Big Five 等心理测量工具评估 LLM 的行为特质，本文将其扩展到无显式人格 prompt 的场景
- **事件抽取**：DDEE (Liu & Luo, 2024)、ULTRA (Zhang et al., 2024)、EventRL (Gao et al., 2024) 聚焦事件抽取准确性，本文转向事件**重要性排序**这一主观维度
- **长上下文推理**：NoLiMa (Modarressi et al., 2025) 和 BABILong (Kuratov et al., 2024) 测试长上下文能力，本文从人格视角补充了这一评估维度
- **对 idea 的启发**：可以将人格分析扩展到更多模型和领域；结合 mechanistic interpretability 分析模型内部如何表示人格相关特征

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 新颖性 | 4 | 事件排序→人格推断的任务设计新颖 |
| 技术深度 | 3 | 方法本身较直接（RAG+prompt+judge），无复杂模型设计 |
| 实验充分性 | 3 | 跨领域验证充分，但样本量有限且缺乏人类评估 |
| 实用价值 | 4 | 对模型选择和人机协作有实际指导意义 |
| 写作质量 | 4 | 结构清晰，案例丰富，可读性强 |
| **综合** | **3.5** | 理念有价值，但需更严格的验证框架 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Inference-Time Decomposition of Activations (ITDA): A Scalable Approach to Interpreting Large Language Models](inference-time_decomposition_of_activations_itda_a_scalable_approach_to_interpre.md)
- [\[ACL 2025\] Cracking Factual Knowledge: A Comprehensive Analysis of Degenerate Knowledge Neurons in Large Language Models](../../ACL2025/interpretability/degenerate_knowledge_neurons.md)
- [\[ACL 2026\] DPN-LE: Dual Personality Neuron Localization and Editing for Large Language Models](../../ACL2026/interpretability/dpn-le_dual_personality_neuron_localization_and_editing_for_large_language_model.md)
- [\[ICML 2026\] Scalable Circuit Learning for Interpreting Large Language Models](../../ICML2026/interpretability/scalable_circuit_learning_for_interpreting_large_language_models.md)
- [\[NeurIPS 2025\] Table as a Modality for Large Language Models](../../NeurIPS2025/interpretability/table_as_a_modality_for_large_language_models.md)

</div>

<!-- RELATED:END -->
