---
title: >-
  [论文解读] MIRAGE: A Benchmark for Multimodal Information-Seeking and Reasoning in Agriculture
description: >-
  [NeurIPS 2025][多模态VLM][benchmark] MIRAGE 是首个基于真实农业专家咨询对话（35,000+条）构建的多模态基准，评估视觉语言模型在领域级实体识别、因果推理和"澄清还是回答"决策方面的能力，揭示了即使 GPT-4.1 识别准确率也仅 43.9% 的严峻挑战。 1. 现有基准局限：主流 VL…
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "benchmark"
  - "多模态"
  - "VLM"
  - "agriculture"
  - "visual grounding"
  - "multi-turn dialogue"
---

# MIRAGE: A Benchmark for Multimodal Information-Seeking and Reasoning in Agriculture

**会议**: NeurIPS 2025  
**arXiv**: [2506.20100](https://arxiv.org/abs/2506.20100)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: benchmark, multimodal, VLM, agriculture, visual grounding, multi-turn dialogue  

## 一句话总结
MIRAGE 是首个基于真实农业专家咨询对话（35,000+条）构建的多模态基准，评估视觉语言模型在领域级实体识别、因果推理和"澄清还是回答"决策方面的能力，揭示了即使 GPT-4.1 识别准确率也仅 43.9% 的严峻挑战。

## 背景与动机

1. **现有基准局限**：主流 VLM 基准（MMMU、VQA）聚焦短文本问答或受限选择题，无法反映真实专家咨询场景的交互性和决策需求。
2. **领域知识缺口**：农业等知识密集型领域中，用户查询经常带有模糊性、上下文不完整性和开放世界实体，现有数据集未充分覆盖这些场景。
3. **多模态融合不足**：现实中农民发送作物照片、描述症状、提供地理位置和时间信息，但现有基准很少同时整合文本、图像和元数据。
4. **交互式决策未被评估**：专家在咨询中需要判断是直接回答还是先追问澄清，这种"clarify-or-respond"决策能力之前未被系统评估。
5. **开放世界泛化挑战**：现有分类数据集采用封闭类别体系，而真实场景涉及 7,000+ 种生物实体（含大量罕见物种），模型泛化能力未知。
6. **错误输出风险高**：农业 AI 的不准确推荐可能导致严重后果（如误判病害、错误用药），凸显了严格评估框架的必要性。

## 方法详解

### 整体框架

- **功能**：构建 MIRAGE 基准，包含两大评估任务——MMST（多模态单轮）和 MMMT（多模态多轮），覆盖实体识别、管理建议生成和澄清决策。
- **为什么**：传统基准缺乏真实交互场景、领域专家标注和开放世界设置，无法全面评估 VLM 在知识密集领域的实际能力。
- **怎么做**：从 AskExtension 平台收集 218,000+ 交互记录，经四步流水线（清洗→分类→格式化→划分）筛选出高质量数据，构建标准子集（8,184 条）、上下文子集（3,934 条）和多轮对话集（861 段对话）。

### 关键设计

#### MMST 单轮推理任务
- **功能**：给定用户查询 + 图片 + 元数据，模型需生成结构化响应，包括实体识别（ID）和管理建议（MG）。
- **为什么**：评估模型从视觉症状中进行因果推理和生成可操作建议的能力，模拟真实的长文本视觉问答。
- **怎么做**：将数据分为标准子集（自包含查询）和上下文子集（需推断隐含的时间/地点/农业背景信息）；ID 任务用二元匹配准确率，MG 任务用 4 维评分（Accuracy/Relevance/Completeness/Parsimony）。

#### MMMT 多轮决策任务
- **功能**：在多轮对话中，模型需判断在当前上下文下应澄清（Clarify）还是回答（Respond），并生成对应话语。
- **为什么**：真实专家咨询中，用户信息经常不完整，专家需要主动发现知识缺口并引导对话，这种决策能力是 AI 助手的核心。
- **怎么做**：将多轮对话截断到某个用户发言处，利用后续用户回复作为"揭示事实"来重建决策点；使用 GPT-4o 生成结构化标注；评估决策准确率和生成内容的目标相关性。

#### 评估框架
- **功能**：采用多推理 LLM 集成评判方案，替代单模型单次评估。
- **为什么**：单一评判器存在偏差，集成方案提高可靠性和可复现性。
- **怎么做**：使用 DeepSeek-R1-Distilled、Qwen3-32B、Phi-4-Reasoning 三个推理 LLM 作为评判者，每个样本 3 次生成 × 3 个评判者 = 9 次评估；用 Fleiss' κ（0.75-0.88）验证一致性。

## 实验结果

### 实验一：MMST 单轮识别与管理任务

| 模型 | ID Acc (%) | Reasoning | MG-Acc | MG-Rel | MG-Comp | MG-Pars | W-Sum |
|------|-----------|-----------|--------|--------|---------|---------|-------|
| GPT-4.1 | **43.9** | **3.01** | **3.24** | **3.60** | **3.22** | **3.01** | **0.82** |
| Claude-3.7-Sonnet | 33.9 | 2.64 | 2.82 | 3.23 | 2.69 | 2.88 | 0.72 |
| Qwen2.5-VL-72B | 29.8 | 2.47 | 2.72 | 3.09 | 2.56 | 2.61 | 0.69 |
| Qwen2.5-VL-32B | 25.1 | 2.43 | 2.87 | 3.19 | 2.88 | 2.43 | 0.71 |
| InternVL3-78B | 22.4 | 2.24 | 2.60 | 2.98 | 2.31 | 2.87 | 0.67 |
| LLaVA-v1.6-7B | 7.1 | 1.34 | 2.20 | 2.50 | 1.86 | 2.20 | 0.55 |

**发现**：即使最强的 GPT-4.1 识别准确率也仅 43.9%，开源模型最高不到 30%，说明开放世界农业实体识别极具挑战性。闭源与开源模型差距约 14 个百分点。

### 实验二：MMMT 多轮决策任务

| 模型 | Zero-Shot Acc% | CoT Acc% | Clarify Rel | Respond Rel |
|------|---------------|---------|-------------|-------------|
| GPT-4o | 62.98 | **65.50** | 72.80 | 78.50 |
| Claude-3.7-Sonnet | 57.80 | 62.40 | 34.90 | 28.70 |
| LLaMA-4-Maverick | 53.75 | 59.80 | 69.10 | 74.20 |
| Qwen-72B | 31.33 | 37.40 | 63.90 | 76.50 |

**发现**：在部分可观测条件下，即使 GPT-4o 决策准确率也仅约 63-65%，表明推断用户隐含目标和知识缺口仍然是核心难点。CoT 提示一致性地提升各模型表现（+2.5%~+6%）。

### 微调实验

LoRA 微调 Qwen2.5-VL-3B 后，已见实体准确率从 22.3% 提升至 28.4%，但未见实体仅 14.6%，存在约 14 个百分点的泛化差距；32B 模型峰值达 37.6%。

## 亮点

- 首个基于 35,000+ 真实专家对话的农业多模态基准，涵盖 7,000+ 生物实体，生态效度极高
- 引入"clarify-or-respond"决策评估维度，超越传统问答范式
- 三评判器集成框架具有高一致性（κ=0.75-0.88），评估方案可复用
- 标准/上下文/多轮三个子集设计，层层递进地揭示模型不同维度的短板

## 局限性

- 数据偏向美国小型种植和家庭园艺场景，未覆盖大规模工业农业
- MMMT 任务未模拟真正的动态交互（如用户模拟器对话），仅评估单步决策
- 元数据（时间/位置）对模型性能提升有限（GPT-4.1 仅 +1.6%），当前模型未能有效利用
- 多轮对话中视觉跟进未被建模，首轮之后假设均为纯文本

## 相关工作对比

| 维度 | MIRAGE | AgMMU |
|------|--------|-------|
| 数据来源 | 真实专家-用户对话 | 合成短文本选择题 |
| 任务类型 | 开放式长文本问答 + 多轮决策 | 封闭式选择题 |
| 多模态 | 图像 + 文本 + 元数据 | 图像 + 文本 |
| 生物实体覆盖 | 7,000+ 种 | 有限类别 |
| 评估维度 | 识别 + 推理 + 决策 + 生成质量 | 准确率 |

| 维度 | MIRAGE | CROP |
|------|--------|------|
| 数据来源 | AskExtension 真实对话 | 两种作物数据 |
| 多模态支持 | ✓（图像） | ✗（纯文本） |
| 作物覆盖 | 数千种作物/害虫/病害 | 仅 2 种作物 |
| 多轮对话 | ✓（861 段，含决策标注） | ✓（但无决策维度） |

## 评分

- ⭐⭐⭐⭐ 新颖性：首个将"澄清决策"纳入多模态基准评估，填补重要空白
- ⭐⭐⭐⭐ 技术质量：数据筛选流程严谨，三评判器集成方案设计合理，系统评估 22 个模型
- ⭐⭐⭐⭐ 实用价值：对农业 AI 系统开发有直接指导意义，基准和代码已开源
- ⭐⭐⭐ 表达清晰度：论文结构清晰但内容量大，部分实验细节需参考附录

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Can Multimodal Foundation Models Understand Schematic Diagrams? An Empirical Study on Information-Seeking QA over Scientific Papers](../../ACL2025/multimodal_vlm/can_multimodal_foundation_models_understand_schematic_diagrams_an_empirical_stud.md)
- [\[ICML 2025\] Context is Key: A Benchmark for Forecasting with Essential Textual Information](../../ICML2025/multimodal_vlm/context_is_key_a_benchmark_for_forecasting_with_essential_textual_information.md)
- [\[CVPR 2025\] Beyond Final Answers: CRYSTAL Benchmark for Transparent Multimodal Reasoning Evaluation](../../CVPR2025/multimodal_vlm/beyond_final_answers_crystal_benchmark_for_transparent_multimodal_reasoning_eval.md)
- [\[ICLR 2026\] Decoding Open-Ended Information Seeking Goals from Eye Movements in Reading](../../ICLR2026/multimodal_vlm/decoding_open-ended_information_seeking_goals_from_eye_movements_in_reading.md)
- [\[NeurIPS 2025\] MMPerspective: Do MLLMs Understand Perspective? A Comprehensive Benchmark for Perspective Perception, Reasoning, and Robustness](mmperspective_do_mllms_understand_perspective_a_comprehensive_benchmark_for_pers.md)

</div>

<!-- RELATED:END -->
