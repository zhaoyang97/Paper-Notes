---
title: >-
  [论文解读] Chinese SimpleQA: A Chinese Factuality Evaluation for Large Language Models
description: >-
  [ACL 2025][LLM安全][中文基准] 提出 Chinese SimpleQA——首个全面的中文事实性评估基准，包含 3000 个高质量短问答（覆盖 6 大主题、99 个子主题），评估 41 个 LLM 后发现仅 o1-preview（63.8%）和 Doubao-pro-32k（61.9%）能通过，并系统揭示了"大模型更好"、"RAG缩小差距"、"对齐降低事实性"等关键洞察。
tags:
  - "ACL 2025"
  - "LLM安全"
  - "中文基准"
  - "事实性评估"
  - "SimpleQA"
  - "知识边界"
  - "RAG"
  - "对齐税"
  - "校准性"
---

# Chinese SimpleQA: A Chinese Factuality Evaluation for Large Language Models

**会议**: ACL 2025  
**arXiv**: [2411.07140](https://arxiv.org/abs/2411.07140)  
**代码**: [https://openstellarteam.github.io/ChineseSimpleQA/](https://openstellarteam.github.io/ChineseSimpleQA/)  
**作者**: Yancheng He, Shilong Li, Jiaheng Liu 等  
**机构**: 阿里巴巴淘天集团  
**领域**: LLM评估 / 事实性  
**关键词**: 中文基准, 事实性评估, SimpleQA, 知识边界, RAG, 对齐税, 校准性

## 一句话总结

提出 Chinese SimpleQA——首个全面的中文事实性评估基准，包含 3000 个高质量短问答（覆盖 6 大主题、99 个子主题），评估 41 个 LLM 后发现仅 o1-preview（63.8%）和 Doubao-pro-32k（61.9%）能通过，并系统揭示了"大模型更好"、"RAG缩小差距"、"对齐降低事实性"等关键洞察。

## 研究背景与动机

**领域现状**：
   - LLM 生成事实性不一致的内容（幻觉问题）严重阻碍了通用 AI 的广泛应用
   - OpenAI 发布的 SimpleQA 基准为英文事实性评估提供了简洁可靠的工具，但主要面向英文
   - 现有中文 LLM 基准（C-Eval、CMMLU）主要测试推理能力，未专门评估中文事实性知识边界

**核心问题**：
   - 缺乏聚焦中文语言的事实性评估基准
   - LLM 在中文知识领域（特别是中国文化相关知识）的表现与英文存在显著差异
   - 对齐训练（RLHF/DPO等）是否降低模型事实性尚缺系统性验证

**研究目标**：构建一个中文、多样、高质量、静态、易评估的事实性基准，全面评估现有 LLM 在中文知识上的边界

## 方法详解

### 数据集设计原则

Chinese SimpleQA 遵循五个核心原则：

1. **中文（Chinese）**：聚焦中文语言知识评估
2. **多样（Diverse）**：6 大主题 + 99 个细分子主题
3. **高质量（High-quality）**：严格的自动化+人工质控流程
4. **静态（Static）**：所有答案不随时间变化（常青属性）
5. **易评估（Easy-to-evaluate）**：问答极短，可用 LLM API 快速评分

### 六大主题覆盖

| 主题 | 样本数 | 示例 |
|------|--------|------|
| 中国文化 (Chinese Culture) | 323 | 非中文特有不可的文化知识 |
| 人文 (Humanities) | 623 | 历史、哲学、语言等 |
| 工程技术与应用科学 (ETAS) | 473 | 计算机、工程、医学等 |
| 生活、艺术与文化 (LAC) | 602 | 日常生活、艺术、体育等 |
| 社会 (Society) | 450 | 政治、经济、法律等 |
| 自然科学 (Natural Science) | 529 | 物理、化学、生物等 |

### 数据构建流程

**自动化阶段**：
1. **知识内容提取**：从 Wikipedia 等知识丰富文本中提取高质量内容
2. **问答对生成**：用 LLM 基于高质量知识内容自动生成问答对
3. **LLM 质量验证**：按预定义标准（答案唯一性、静态性等）自动过滤
4. **RAG 验证**：使用 LlamaIndex + Google/Bing 搜索引擎进行检索增强验证
5. **难度过滤**：如果四个强力模型（GPT-4o、Llama3-70B、Qwen2.5-72B、GLM-4-Plus）全部答对，则剔除该问题

**人工验证阶段**：
- 每个问题由 **2 名标注员独立评估**
- 标注员使用搜索引擎查找答案，每人提供至少 2 个支持 URL
- 答案不一致时由 **第 3 名标注员仲裁**
- 最终仅保留与 LLM 答案一致的样本

### 问答构建标准（四条核心规则）

1. **答案必须客观唯一**：排除主观问题和多解问题（如"朱祁镇何年登基"有两个答案）
2. **答案不随时间变化**：排除时事问题（如"某国现任总统"）
3. **问题需有挑战性**：不能过于简单
4. **截止至 2023 年可回答**：确保训练数据截止日期后的模型可公平评估

### 数据量变化

| 阶段 | 样本数 | 保留比例 |
|------|--------|----------|
| 初始生成 | 10,000 | 100% |
| 难度过滤后 | 6,310 | 63.1% |
| 规则+RAG 验证后 | 3,470 | 34.7% |
| 人工审核后 | **3,000** | **30.0%** |

### 评估指标

- **Correct (CO)**：预测答案完全包含参考答案且无矛盾
- **Not Attempted (NA)**：未给出参考答案，也无矛盾
- **Incorrect (IN)**：预测答案与参考答案矛盾
- **Correct Given Attempted (CGA)**：在已作答的问题中回答正确的比例
- **F-score**：CO 和 CGA 的调和平均数

### 数据统计

| 统计项 | 数值 |
|--------|------|
| 问题平均长度 | 23.6 字 |
| 答案平均长度 | 6.1 字 |
| 问题最长 | 81 字 |
| 答案最长 | 47 字 |

## 实验

### 评估规模

评估 **41 个 LLM**：17 个闭源 + 24 个开源，涵盖 o1、GPT-4o、Qwen2.5、InternLM、Yi、LLaMA3、DeepSeek、Baichuan、Mistral 等系列。

### 主实验结果（整体排名，部分展示）

| 模型 | CO↑ | NA | IN↓ | CGA | F-score |
|------|-----|-----|-----|-----|---------|
| o1-preview | **63.8** | 12.2 | 24.0 | 72.7 | **67.9** |
| Doubao-pro-32k | 61.9 | 10.3 | 27.8 | 69.1 | 65.3 |
| GLM-4-Plus | 58.7 | 7.4 | 33.9 | 63.4 | 60.9 |
| GPT-4o | 59.3 | 1.4 | 39.3 | 60.1 | 59.7 |
| Qwen-Max | 54.1 | 11.3 | 34.6 | 61.0 | 57.4 |
| Qwen2.5-72B (开源) | 48.4 | 7.1 | 44.5 | 52.1 | 50.2 |
| DeepSeek-67B | 43.5 | 14.8 | 41.7 | 51.1 | 47.0 |
| LLaMA3.1-70B | 38.3 | 9.4 | 52.3 | 42.3 | 40.2 |
| GPT-3.5 | 29.7 | 2.9 | 67.4 | 30.6 | 30.1 |

**基准难度验证**：仅 o1-preview 和 Doubao-pro-32k 超过 60%（及格线）。

### 关键发现 1：大模型更好

Qwen2.5 系列模型规模效应：

| 模型 | CO |
|------|-----|
| Qwen2.5-72B | 48.4% |
| Qwen2.5-32B | 38.8% |
| Qwen2.5-14B | 35.4% |
| Qwen2.5-7B | 26.6% |
| Qwen2.5-3B | 16.2% |
| Qwen2.5-1.5B | 11.1% |

从 1.5B 到 72B，CO 准确率从 11.1% 提升到 48.4%，呈近线性增长。

### 关键发现 2：中国文化主题上中文模型优势显著

在 "Chinese Culture" 子主题上的 F-score：

| 模型 | Chinese Culture | 整体 |
|------|----------------|------|
| Doubao-pro-32k | **61.8** | 65.3 |
| GLM-4-Plus | 56.5 | 60.9 |
| DeepSeek-V2.5 | 50.4 | 55.7 |
| o1-preview | 45.7 | **67.9** |
| GPT-4o | 39.4 | 59.7 |

Doubao-pro-32k 和 GLM-4-Plus 在中国文化上大幅领先 o1-preview（+16/+11 个百分点），而整体排名落后。

### 关键发现 3：RAG 大幅缩小模型差距

引入 RAG 后的性能变化：

| 模型对比 | 无 RAG 差距 | 有 RAG 差距 |
|----------|-----------|-----------|
| GPT-4o vs Qwen2.5-3B | 42.4% | **9.3%** |

RAG 策略使弱模型获益更大，极大缩小了不同规模模型间的性能差距。

### 关键发现 4：对齐税存在

对齐和后训练策略通常会降低模型的事实性表现——模型在对齐过程中可能牺牲知识准确性来换取安全性和有用性。

### 关键发现 5：大模型校准性更好

| 模型 | CO | NA | 解读 |
|------|-----|-----|------|
| o1-preview | 63.8 | 12.2 | 不确定时会拒绝回答 |
| o1-mini | 39.5 | 20.6 | 拒绝更多但也答错更多 |
| GPT-4o | 59.3 | 1.4 | 几乎从不拒绝 |
| GPT-4o-mini | 37.6 | 0.9 | 也不拒绝，但错得更多 |
| Claude-3.5-Sonnet | 46.2 | **27.4** | 最谨慎的模型 |

Claude-3.5-Sonnet 拒绝率最高（27.4%），但也因此避免了大量错误回答。

### SimpleQA vs Chinese SimpleQA 排名差异

英文 SimpleQA 排名和中文 Chinese SimpleQA 排名不一致：专注中文的模型（如 Doubao、GLM-4-Plus）在中文版本上排名显著提升，说明中英知识评估不可互相替代。

## 亮点与洞察

1. **首个系统的中文事实性基准**：填补了中文 LLM 事实性评估的空白，与 OpenAI SimpleQA 形成互补
2. **严格的数据质量保障**：三轮过滤（LLM自动/RAG验证/双人人工审核）仅保留 30% 原始数据，确保高质量
3. **全面的模型评估生态**：覆盖 41 个模型，闭源+开源，多尺度（0.5B-671B），提供了目前最全面的中文事实性能力画像
4. **对齐税的系统性验证**：首次在中文事实性基准上证实对齐训练可能降低事实准确性，为后训练策略设计提供参考
5. **RAG 的均衡化效应发现**：RAG 使模型间差距从 42.4% 缩至 9.3%，对资源受限场景（只能用小模型）有重要实践指导意义
6. **中文文化知识的差异性**：揭示了国际化 LLM（GPT、o1）在中国文化领域的短板，说明数据来源对知识覆盖的关键影响

## 局限性

1. **评估成本低但构建成本高**：双人标注+第三人仲裁的人工质控成本较高，难以快速扩展
2. **时间截止限制**：所有问题须在 2023 年底前可回答，无法评估模型对更新知识的掌握
3. **领域覆盖不均**：中国文化（323 题）明显少于生活文化（602 题），可能低估中文文化知识的评测深度
4. **仅评估事实性**：不涉及推理能力、创造性写作等其他维度，无法全面评估 LLM
5. **评分依赖 OpenAI API**：使用 LLM-as-a-Judge 进行自动评分，评分器本身的准确率未充分验证
6. **静态设计的双面性**：答案不随时间变化虽保证了基准稳定性，但也意味着无法捕捉 LLM 对动态世界知识的理解能力

## 相关工作

- **事实性基准**：SimpleQA (Wei et al., 2024)、TruthfulQA、FreshQA
- **中文 LLM 基准**：C-Eval (Huang et al., 2023)、CMMLU (Li et al., 2023)、WebQA (Li et al., 2016)
- **通用评估**：MMLU (Hendrycks et al., 2021)、GSM8K (Cobbe et al., 2021)、AlpacaEval
- **LLM-as-Judge**：MT-Bench (Zheng et al., 2023)、Arena-Hard (Li et al., 2024)

## 评分

⭐⭐⭐⭐⭐ — 填补中文事实性评估空白的重要工作，数据质量极高、评估覆盖全面、发现（对齐税、RAG均衡效应、中文文化差异）有深刻实践价值，是中文 LLM 开发者的必读基准。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] UAlign: Leveraging Uncertainty Estimations for Factuality Alignment on Large Language Models](ualign_leveraging_uncertainty_estimations_for_factuality_alignment_on_large_lang.md)
- [\[ACL 2025\] Ensemble Watermarks for Large Language Models](ensemble_watermarks_llm.md)
- [\[ACL 2025\] Real-time Factuality Assessment from Adversarial Feedback](real-time_factuality_assessment_from_adversarial_feedback.md)
- [\[ACL 2026\] Permutation-Consensus Listwise Judging for Robust Factuality Evaluation](../../ACL2026/llm_safety/permutation-consensus_listwise_judging_for_robust_factuality_evaluation.md)
- [\[ACL 2025\] Ewe: Improving Factuality with Explicit Working Memory](improving_factuality_with_explicit_working_memory.md)

</div>

<!-- RELATED:END -->
