# Improving Automatic Evaluation of LLMs in Biomedical Relation Extraction via LLMs-as-the-Judge

**会议**: ACL 2025  
**arXiv**: [2506.00777](https://arxiv.org/abs/2506.00777)  
**代码**: [https://github.com/tahmedge/llm_judge_biomedical_re](https://github.com/tahmedge/llm_judge_biomedical_re)  
**领域**: 医学图像 / NLP理解  
**关键词**: LLM-as-Judge, biomedical relation extraction, structured output, domain adaptation, evaluation  

## 一句话总结
本文首次系统研究了 LLM-as-Judge 在生物医学关系抽取评估中的表现，发现其准确率通常低于 50%，并提出结构化输出格式（JSON）和域适应技术来提升约 15% 的评估准确率。

## 研究背景与动机

1. **领域现状**：LLM 在生物医学关系抽取（BioRE）中已展现出强大的零样本能力，但评估仍依赖昂贵的人工标注。LLM-as-Judge 范式在通用 NLP 任务中已证明有效，但在生物医学领域尚未深入探索。
2. **现有痛点**：(1) LLM 生成的回答经常包含金标准的同义词或缩写（如"dex" vs "dexamethasone"），传统精确匹配指标无法正确评估；(2) 人工评估成本高、耗时长、不可扩展；(3) 直接将 LLM-Judge 应用到生物医学关系抽取时，准确率出奇地低（通常 <50%）。
3. **核心矛盾**：LLM 生成器的自由格式输出导致 LLM 评判者难以准确解析关系，而生物医学领域的专业术语和命名规范进一步加剧了这一问题。
4. **本文要解决什么？** (1) 量化 LLM-Judge 在 BioRE 评估中的实际能力；(2) 找出失败原因；(3) 提出改进方案使 LLM-Judge 更可靠。
5. **切入角度**：从 LLM 生成器的输出格式入手——非结构化文本是评判困难的根因。
6. **核心idea一句话**：通过将 LLM 生成器的输出规范为 JSON 结构化格式，显著提升 LLM-Judge 在 BioRE 评估中的准确率，结合域适应微调进一步增强效果。

## 方法详解

### 整体框架
输入文本 → LLM-Generator（关系抽取）→ 输出关系 → LLM-Judge（替代人工评估）→ 判定正确关系数和总预测关系数 → 计算 Precision/Recall/F1。核心改进在两个环节：(1) LLM-Generator 的输出格式（非结构化→JSON结构化）；(2) LLM-Judge 的域适应微调。

### 关键设计

1. **结构化输出格式（Structured Output Formatting）**:
   - 做什么：要求 LLM-Generator 以 JSON 格式输出关系抽取结果，而非自由文本
   - 核心思路：原始 LLM 输出如"The drug X treats disease Y"是自由文本，LLM-Judge 需要从中解析关系对，容易出错。改为 JSON 格式 `{"relations": [{"entity1": "X", "relation": "treats", "entity2": "Y"}]}` 后，评判变为结构化比对
   - 设计动机：实验观察到 LLM-Judge 的低准确率主要源于无法正确解析非结构化输出中的关系。JSON 格式选择基于近期研究表明 LLM 在 JSON 格式上比 YAML 等更可靠
   - 效果：平均提升约 15% 精确匹配准确率

2. **域适应（Domain Adaptation via Transfer Learning）**:
   - 做什么：在缺乏目标领域人工评估数据时，用其他数据集的人工标注数据微调 LLM-Judge
   - 核心思路：假设目标数据集 $X$ 无训练数据，但有另一数据集 $Y$ 的人工评估数据可用，则先在 $Y$ 上微调 LLM-Judge，使其学习"如何评判关系抽取"的通用能力，再迁移到 $X$
   - 设计动机：开源 LLM-Judge（如 Prometheus-2）仅针对通用评估维度（helpfulness、factual correctness）微调，不适用于关系抽取评估；而人工标注的判断数据稀缺
   - 解决了域特异性问题和数据稀缺问题的双重挑战

3. **评估指标设计**:
   - **精确匹配准确率（EM）**：LLM-Judge 标注的正确关系数和总预测关系数都与人工标注完全一致
   - **RMSE**：计算 LLM-Judge 标注与人工标注之间的均方根误差，惩罚大偏差

## 实验关键数据

### 主实验：LLM-Judge 零样本准确率

| LLM-Judge | BC5CDR EM↑ | DDI EM↑ | KD-DTI EM↑ | BC5CDR RMSE↓ |
|-----------|-----------|---------|-----------|-------------|
| GPT-4o-Mini | **48.35** | **59.03** | **53.11** | 2.33 |
| Gemini-Flash | 42.55 | 47.12 | 40.68 | **2.09** |
| Qwen-2.5-7B | 45.25 | 46.60 | 49.98 | 2.42 |
| Claude-3-Haiku | 29.50 | 31.15 | 40.27 | 2.26 |
| LLaMA-3.1-8B | 29.45 | 29.32 | 36.73 | 2.40 |
| DeepSeek-R1-Qwen-7B | 30.60 | 42.67 | 42.45 | 2.76 |

### 消融实验：结构化输出的影响

| 配置 | BC5CDR EM | DDI EM | KD-DTI EM | 说明 |
|------|----------|--------|----------|------|
| 非结构化输出 | ~48% (GPT-4o-Mini) | ~59% | ~53% | 基线，自由文本格式 |
| 结构化 JSON 输出 | ~63% | ~74% | ~68% | 平均提升约 15% |
| + 域适应微调 | 进一步提升 | 进一步提升 | 进一步提升 | 跨数据集知识迁移 |

### 关键发现
- **LLM-Judge 在 BioRE 中表现很差**：最好的 GPT-4o-Mini 准确率也仅约 50%，远低于通用 NLP 任务中的表现
- **推理型 LLM 无优势**：DeepSeek-R1 蒸馏版本并未优于标准指令微调模型，说明"推理"能力不直接迁移到评估任务
- **结构化格式是关键**：统一输出为 JSON 格式后，LLM-Judge 更容易准确解析和比对关系，一致性提升显著
- **生物医学专用 LLM 不适合做 Judge**：BioMistral-7B 完全无法遵循评判指令
- **域适应有效但有限**：跨数据集迁移在关系类型相似时效果更好

## 亮点与洞察
- **问题诊断精准**：通过大量实验（100+轮）准确定位了 LLM-Judge 失败的根因——非结构化输出导致解析困难，而非评判能力不足。这一洞察简单但影响深远
- **结构化输出的通用性**：JSON 结构化输出不仅帮助评估，也是提升 LLM 关系抽取质量的good practice，可迁移到其他信息抽取任务
- **域适应策略**：用异域人工标注数据微调 LLM-Judge 的思路可推广到其他缺乏评估数据的领域（如法律、金融实体关系）
- **大规模标注数据开源**：36K 标注样本（4K 人工 + 32K LLM）的公开对社区有实际价值

## 局限性 / 可改进方向
- 仅评估了 3 个 BioRE 数据集，覆盖的关系类型有限（药物-疾病、药物-药物、药物-靶点），未涉及基因-疾病等更广泛的关系类型
- 域适应仅使用了简单的跨数据集微调，未探索更先进的迁移学习技术（如 adapter、LoRA）
- 结构化输出依赖 LLM-Generator 的 JSON 生成能力，早期 LLM 无法生成结构化格式
- 评估聚焦于关系抽取，未扩展到其他生物医学 NLP 任务（NER、事件抽取等）

## 相关工作与启发
- **vs LLM-as-Judge (Zheng et al., 2023)**: 通用 LLM-Judge 在对话生成评估中表现良好，但本文揭示其在结构化任务（关系抽取）中的严重不足
- **vs Prometheus-2 (Kim et al., 2024)**: Prometheus-2 是专门微调的开源评估 LLM，但其评估维度（helpfulness, correctness）不适用于关系抽取，凸显了任务特定评估的必要性
- **vs Jahan et al. (2024)**: 提供了 LLM 在 BioRE 中的零样本基准，但依赖人工评估，本文尝试用 LLM-Judge 替代

## 评分
- 新颖性: ⭐⭐⭐ 首次系统研究 LLM-Judge 在 BioRE 中的表现，但方法（结构化输出+域适应）相对直接
- 实验充分度: ⭐⭐⭐⭐⭐ 100+ 实验，8 个 Judge × 5 个 Generator × 3 个数据集，极其全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，问题-分析-解决的逻辑链完整
- 价值: ⭐⭐⭐⭐ 实用价值高，开源数据集对社区有贡献
