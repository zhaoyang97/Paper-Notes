---
title: >-
  [论文解读] MTBBench: A Multimodal Sequential Clinical Decision-Making Benchmark in Oncology
description: >-
  [NeurIPS 2025][医学图像][多模态基准] 提出MTBBench——首个同时覆盖多模态、纵向时序和交互式Agent工作流三个维度的临床基准，模拟分子肿瘤委员会（MTB）的决策流程，评估并增强AI Agent在肿瘤学精准医疗中的多模态纵向推理能力。 多模态大语言模型在生物医学推理中展现了良好前景…
tags:
  - "NeurIPS 2025"
  - "医学图像"
  - "多模态基准"
  - "肿瘤学"
  - "分子肿瘤委员会"
  - "纵向推理"
  - "临床决策Agent"
---

# MTBBench: A Multimodal Sequential Clinical Decision-Making Benchmark in Oncology

**会议**: NeurIPS 2025  
**arXiv**: [2511.20490](https://arxiv.org/abs/2511.20490)  
**代码**: [GitHub](https://github.com/bunnelab/MTBBench) / [HuggingFace](https://huggingface.co/datasets/EeshaanJain/MTBBench)  
**领域**: 医疗NLP  
**关键词**: 多模态基准, 肿瘤学, 分子肿瘤委员会, 纵向推理, 临床决策Agent

## 一句话总结

提出MTBBench——首个同时覆盖多模态、纵向时序和交互式Agent工作流三个维度的临床基准，模拟分子肿瘤委员会（MTB）的决策流程，评估并增强AI Agent在肿瘤学精准医疗中的多模态纵向推理能力。

## 研究背景与动机

多模态大语言模型在生物医学推理中展现了良好前景，但现有评估基准与真实临床工作流存在严重脱节。当前评估主要聚焦于单模态、去上下文化的静态问答，忽略了像分子肿瘤委员会（MTB）这样的多专家、多轮决策环境。MTB是肿瘤学中的典型决策场景：肿瘤学家、放射科医生、病理学家和遗传学家共同分析一个不断演变的患者病例，需要整合H&E染色、免疫组化（IHC）、血液学、基因组学等多模态数据，并在不同时间点做出决策。

现有基准如MedAgentBench、MediQ等仅覆盖交互性或纵向性的一个方面，且通常限于单一模态（如文本EHR）。主要缺口在于：（1）缺乏同时评估多模态理解+纵向推理+Agent交互的基准；（2）Agent面对部分数据、顺序更新、冲突信息和高风险结果时的认知能力未被测试。

## 方法详解

### 整体框架

MTBBench包含两个轨道：多模态轨道（MTBBench-Multimodal）和纵向轨道（MTBBench-Longitudinal），加上一个Agent框架将基础模型作为可调用工具。Agent在多轮对话中处理时序演变的患者数据，选择性地请求和推理不同模态的文件来回答临床问题。

### 关键设计

1. **MTBBench-Multimodal**: 从HANCOCK数据集遴选26例头颈癌患者，每例平均40个模态文件（~1.2张H&E切片、~26.2张IHC图像、1份血液学报告），生成390道多模态问答对（15道/患者）。任务分三阶段：病理图像解读（H&E/IHC组织亚型和免疫浸润空间分布）→ 血液学推理（术前生化分析感染风险、出血倾向等）→ 术后预后整合（5年生存率和2年复发预测）。

2. **MTBBench-Longitudinal**: 从MSK-CHORD数据集遴选40例患者，每例约5个关联文件（拷贝数变异、体细胞突变、病理报告、临床时间线），手动构建183道纵向问答对。Agent需在时序分段的决策节点上回答诊断轨迹、生存预测、复发预测和治疗进展映射问题，基因组数据在关键阶段引入以支持耐药模式分析。

3. **Agent工作流与基础模型工具**: Agent在每轮 $t$ 接收临床查询 $q_t$ 和模态文件集 $\mathcal{F}_t$，可请求子集 $\mathcal{R}_t \subseteq \mathcal{F}_t$ 来检索信息。文件不自动跨轮持续存在，必须主动重新请求。工具包括：

    - **CONCH**（H&E视觉语言模型）：基于图像-文本嵌入相似度返回最匹配描述
    - **UNI2 + ABMIL**（IHC定量工具）：基础模型嵌入 + 注意力多实例学习回归阳性染色比例
    - **PubMed检索**：自然语言查询 → BAAI-bge重排 → 返回top-3摘要
    - **DrugBank**：自动链接药物名称到治疗适应症、作用机制、药物相互作用信息

### 损失函数 / 训练策略

- 专家验证流程：开发了一个Web端伴侣应用，允许临床医生检查临床上下文、浏览H&E/IHC图像、对每道Q&A进行反馈标注
- IHC定量工具的训练：256×256 patch → UNI2编码为1536维特征 → ABMIL回归阳性染色百分比，训练数据通过QuPath手动标注
- 评估指标：bootstrap重采样1000次估计95%置信区间

## 实验关键数据

### 主实验：多模态轨道（无工具）

| 模型 | 数字病理 | 血液学 | 预后与复发 | 总体 |
|------|---------|--------|-----------|------|
| gpt4o | 63.2±6.0 | 76.9±7.7 | 59.9±13.5 | 66.7±8.1 |
| internvl3-78b | 62.0±6.4 | 79.7±7.7 | **65.6±11.5** | **69.1±8.4** |
| qwen25-7b | 42.3±6.2 | 61.1±9.1 | 53.9±12.5 | 52.4±9.0 |
| llama90b | 54.6±6.2 | **82.8±7.2** | 51.7±13.5 | 63.0±14.8 |
| o4-mini | 59.5±6.4 | 77.8±8.2 | 55.7±14.4 | 64.3±10.5 |

### 消融实验：工具增强效果

| 配置 | 关键指标提升 | 说明 |
|------|------------|------|
| 多模态+视觉工具 (CONCH/UNI) | 数字病理准确率↑最高9% | FM工具显著提升病理图像理解 |
| 纵向+知识工具 (PubMed/DrugBank) | 进展/复发预测↑>5% | 外部知识增强时序推理 |
| 文件访问数量 vs 准确率 | 正相关（多模态和纵向） | 信息获取能力比模型规模更关键 |
| 模型规模 vs 准确率 | 无一致正相关 | gemma-3-12b在部分任务优于27b |

### 纵向轨道关键结果

| 模型 | 预后 | 进展 | 复发 | 总体 |
|------|------|------|------|------|
| qwen3-32b | **83.0±9.2** | 63.3±12.3 | 54.6±13.6 | **67.0±13.5** |
| llama33-70b | 73.2±9.9 | **68.2±13.2** | **56.7±13.6** | 66.0±7.8 |
| gpt4o | 72.9±10.6 | 64.8±13.2 | 54.8±13.6 | 64.2±8.6 |

### 关键发现

- internvl3-78b总体最优(69.1%)，超越闭源gpt4o(66.7%)2.5个百分点
- 模型规模不是决定因素——文件访问数量与准确率的正相关更强
- 预后/复发预测对所有模型都极具挑战性，准确率接近随机(~50%)
- 工具增强在所有任务上均有提升，病理任务受益最大
- 纵向轨道中，粗粒度生存信号可被检测，但细粒度时序推理（进展/复发）仍困难

## 亮点与洞察

- 首个在单一基准中联合评估多模态、纵向性和Agent交互的临床AI基准
- 将领域特定基础模型（CONCH、UNI2）作为Agent可调用工具的设计理念新颖，模拟了临床中专家咨询专业系统的真实流程
- 揭示了"信息获取能力 > 模型规模"的重要发现——Agent有效检索和整合信息比参数量更重要
- 专家验证环节（Web应用）提升了基准的临床可信度

## 局限与展望

- 仍是离线控制基准，Agent未在真实交互式临床工作流中测试
- 预后和复发预测性能低下，目前所有模型在此类高层推理任务上能力不足
- 纵向轨道缺乏专用的纵向推理基础模型，主要依赖通用知识工具
- 数据规模有限（26+40例患者），可扩展到更多肿瘤类型
- IHC定量工具依赖手动QuPath标注训练，可扩展性受限

## 相关工作与启发

- 与MedAgentBench、MediQ、MedJourney等相比，MTBBench在多模态+纵向+交互三维度上更完整
- Agent框架中基础模型工具的设计可推广到其他多专科临床场景（如心脏病学MDT、神经科影像讨论）
- 启示：临床AI评估应从静态QA转向动态、过程导向的决策评估

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首个三维度联合的临床Agent基准，定位精准
- **实验充分度**: ⭐⭐⭐⭐ 多模型广泛评估，含工具/无工具对比，但数据规模偏小
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，临床流程描述专业，表格和图表信息丰富
- **价值**: ⭐⭐⭐⭐⭐ 填补了临床AI评估的重要空白，对推动精准肿瘤学AI有直接意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Sequential Attention-based Sampling for Histopathological Analysis](sequential_attention-based_sampling_for_histopathological_analysis.md)
- [\[NeurIPS 2025\] SMMILE: An Expert-Driven Benchmark for Multimodal Medical In-Context Learning](smmile_an_expert-driven_benchmark_for_multimodal_medical_in-context_learning.md)
- [\[CVPR 2026\] OmniBrainBench: A Comprehensive Multimodal Benchmark for Brain Imaging Analysis Across Multi-stage Clinical Tasks](../../CVPR2026/medical_imaging/omnibrainbench_a_comprehensive_multimodal_benchmark_for_brain_imaging_analysis_a.md)
- [\[NeurIPS 2025\] QoQ-Med: Building Multimodal Clinical Foundation Models with Domain-Aware GRPO Training](qoq-med_building_multimodal_clinical_foundation_models_with_domain-aware_grpo_tr.md)
- [\[ICML 2026\] CASCADE Conformal Prediction: Uncertainty-Adaptive Prediction Intervals for Two-Stage Clinical Decision Support](../../ICML2026/medical_imaging/cascade_conformal_prediction_uncertainty-adaptive_prediction_intervals_for_two-s.md)

</div>

<!-- RELATED:END -->
