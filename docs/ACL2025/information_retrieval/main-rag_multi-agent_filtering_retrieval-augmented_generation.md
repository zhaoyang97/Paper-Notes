---
title: >-
  [论文解读] MAIN-RAG: Multi-Agent Filtering Retrieval-Augmented Generation
description: >-
  [ACL 2025][信息检索/RAG][多Agent过滤] 提出 MAIN-RAG，一个无需训练的多 Agent RAG 过滤框架，通过 Predictor→Judge→Final-Predictor 三个 LLM Agent 协作评估检索文档的相关性，并设计自适应阈值（基于分数均值和标准差）动态过滤噪声文档，在 4 个 QA 基准上实现 2-11% 的准确率提升。
tags:
  - "ACL 2025"
  - "信息检索/RAG"
  - "多Agent过滤"
  - "文档噪声"
  - "自适应阈值"
  - "training-free"
  - "相关性评分"
---

# MAIN-RAG: Multi-Agent Filtering Retrieval-Augmented Generation

**会议**: ACL 2025  
**arXiv**: [2501.00332](https://arxiv.org/abs/2501.00332)  
**代码**: 未公开  
**作者**: Chia-Yuan Chang, Zhimeng Jiang, Vineeth Rakesh, Menghai Pan, Chin-Chia Michael Yeh, Guanchu Wang, Mingzhi Hu, Zhichao Xu, Yan Zheng, Mahashweta Das, Na Zou  
**机构**: Texas A&M University, Visa Research, WPI, University of Utah, University of Houston  
**领域**: 检索增强生成 (RAG)  
**关键词**: 多Agent过滤, 文档噪声, 自适应阈值, training-free, 相关性评分

## 一句话总结

提出 MAIN-RAG，一个无需训练的多 Agent RAG 过滤框架，通过 Predictor→Judge→Final-Predictor 三个 LLM Agent 协作评估检索文档的相关性，并设计自适应阈值（基于分数均值和标准差）动态过滤噪声文档，在 4 个 QA 基准上实现 2-11% 的准确率提升。

## 研究背景与动机

**RAG 中的噪声问题**：
   - 检索器返回的文档常包含不相关或有噪声的内容
   - 噪声文档可能误导 LLM，降低回答准确性
   - 现有研究（Chen et al., 2024; Yu et al., 2024）表明 LLM 对噪声缺乏鲁棒性

**现有方案不足**：
   - **训练型 RAG**（Self-RAG、REALM）：效果好但需大量计算资源和训练数据
   - **无训练型 RAG**：简单高效但对噪声敏感，简单拼接 top-k 文档到 prompt 中
   - 缺乏有效的后处理过滤机制

**文档顺序影响**：
   - LLM 存在"lost in the middle"问题——倾向关注输入的开头和结尾
   - 随机打乱文档顺序导致性能方差很大（max 远高于 min），说明存在最优排序

**核心动机**：设计一个 training-free 的多 Agent 框架，通过协作评估和自适应过滤提升 RAG 的噪声鲁棒性

## 方法详解

### 整体框架

MAIN-RAG 在标准 RAG 流程的检索阶段之后增加一个多 Agent 过滤层，由三个 LLM Agent 协作完成：

### Agent 定义

#### Agent-1: Predictor（预测者）

- 对每个查询 q，逐一读取每个检索文档 dᵢ
- 生成基于每个文档的初步回答 aᵢ
- 形成 Document-Query-Answer 三元组 (dᵢ, q, aᵢ)

#### Agent-2: Judge（评判者）

- 接收每个 (dᵢ, q, aᵢ) 三元组
- 判断文档是否为查询和回答提供了相关支持信息
- 输出 "Yes" 或 "No" 的判断

**关键创新——相关性分数量化**：
- 不使用 "Yes"/"No" 的离散判断
- 而是计算 **log P("Yes") - log P("No")** 的差值
- 这个差值作为文档的连续相关性分数 rᵢ
- 使文档可排序，为过滤提供连续阈值基础

#### Agent-3: Final-Predictor（最终预测者）

- 接收过滤和排序后的文档列表
- 基于高质量文档生成最终回答

### 自适应阈值 τ_q

**核心观察**：
- 相关文档的分数分布：偏高、标准差小（LLM 对相关文档更确定）
- 噪声文档的分数分布：更均匀、标准差大（LLM 不确定，可能误判）
- 最优过滤阈值随查询不同而变化

**设计**：
- 对每个查询 q，计算所有候选文档的平均相关性分数作为自适应阈值 τ_q
- 保留分数 rᵢ ≥ τ_q 的文档
- 可通过 τ_q - n·σ 引入灵活性（n 是唯一超参数）
- 直觉：相关文档多时平均分高→过滤掉低分异常值；相关文档少时平均分低→约过滤一半文档

### 文档排序

过滤后的文档按相关性分数**降序**排列——高分文档排在前面，利用 LLM 倾向关注上下文开头的偏置。

## 实验结果

### 数据集与设置

- **4 个 QA 基准**：TriviaQA (开放域)、PopQA (长尾实体)、ARC-Challenge (科学推理)、ALCE-ASQA (长篇 QA)
- **Agent 实例化**：预训练 Mistral-7B 或 Llama3-8B（无微调）
- **检索器**：Contriever-MS MARCO，每个查询检索最多 20 个文档
- **零样本评估**

### 主实验结果

| 方法 | TriviaQA | PopQA | ARC-C | ASQA (em/rg/mau) |
|------|:---:|:---:|:---:|:---:|
| **无检索** |||||
| Mistral-7B | 54.8 | 26.2 | 55.5 | 11.2/18.1/27.6 |
| Llama3-8B | 68.4 | 29.2 | 58.8 | 19.4/30.3/54.3 |
| **标准 RAG** |||||
| Mistral-7B + RAG | 69.4 | 55.5 | 57.1 | 32.4/34.8/54.3 |
| Llama3-8B + RAG | 73.1 | 61.8 | 55.6 | 37.1/36.5/63.0 |
| **训练型方法** |||||
| Self-RAG-7B | 66.4 | 54.9 | 67.3 | 30.0/35.7/74.3 |
| **MAIN-RAG** |||||
| MAIN-RAG (Mistral) | **71.0** | **58.9** | **58.9** | 35.7/36.2/60.0 |
| MAIN-RAG (Llama3) | **74.1** | **64.0** | **61.9** | **39.2/42.0/70.6** |

**核心结论**：
1. MAIN-RAG 在所有基准上超越 training-free 基线，提升 2-11%
2. 在 TriviaQA 和 PopQA 上，training-free 的 MAIN-RAG 接近甚至超越 training-based 的 Self-RAG
3. PopQA（长尾实体）上优势最大——因为检索器未在目标数据上微调，噪声文档更多，过滤的价值更大

### 变体消融（Figure 7）

| 变体 | 作用 |
|------|------|
| Naïve Multi-agent RAG | Judge 用 Yes/No 离散判断替代连续分数 → 性能下降，证明分数量化的必要性 |
| MAIN-RAG (Random) | 过滤后随机排序 → 性能下降，证明按分数降序排列的重要性 |
| 标准 RAG | 无过滤 → 基线 |

### 自适应阈值消融

| 方法 | TriviaQA | PopQA | ARC-C |
|------|:---:|:---:|:---:|
| τ_q (默认) | **71.0** | **58.9** | **58.9** |
| τ_q - 0.5σ | 71.2 | 58.6 | 59.0 |
| τ_q - 1.0σ | 70.8 | 58.0 | 58.5 |
| τ_q - 1.5σ | 70.4 | 58.4 | 57.7 |
| 升序排列 | 70.2 | 53.5 | 57.4 |

**发现**：
- 默认 τ_q 在所有基准上至少排名第二，是最稳定的选择
- 降序排列一致优于升序排列，验证了 LLM "primacy bias" 的存在
- 调整 σ 的效果因数据集而异，默认设置最通用

### Case Study：τ_q 值的直觉

- **Case 1** (τ_q = 9.575)：高置信度，大多数文档相关 → 严格过滤 → 正确回答
- **Case 2** (τ_q = -8.425)：低置信度，大多数文档噪声 → 宽松保留 → 从少量有信息文档中找到答案
- **Case 3** (τ_q = 0.4875)：中等置信度 → 部分文档相关但缺少目标信息 → 回答错误

## 亮点与洞察

1. **Training-free 的竞争力**：无需微调或额外标注数据，仅通过推理时的多 Agent 协作就能接近 Self-RAG 等 training-based 方法的性能
2. **相关性分数量化的巧妙设计**：利用 log P("Yes") - log P("No") 将二值判断转为连续分数，是连接 Agent 判断与传统信息检索排序的桥梁
3. **自适应阈值的鲁棒性**：唯一超参数 n 默认为 0 就能稳定工作，且对不同查询自动调整，比固定阈值更通用
4. **文档排序的实证验证**：系统验证了文档顺序对 RAG 性能的显著影响，降序排列一致最优
5. **可扩展性**：三个 Agent 可用不同 LLM 实例化，框架对 LLM 选择无特殊要求

## 局限性

1. 三个 Agent 各需一次 LLM 推理，计算开销约为标准 RAG 的 3 倍（每个文档需 Agent-1 推理一次 + Agent-2 判断一次）
2. 仅在 QA 任务上验证，未测试摘要、对话等其他 RAG 应用场景
3. 仅测试了 Mistral-7B 和 Llama3-8B 两种模型
4. 未考虑文档压缩、更高级的解码策略等正交优化方向
5. 自适应阈值基于分数均值，对分数分布严重偏斜的情况可能不理想
6. 未考虑检索器选择和重排序器的影响

## 相关工作

- **训练型 RAG**: Self-RAG (Asai et al., 2024) 通过反思 token 学习检索和自我批评；REALM (Guu et al., 2020) 在预训练中引入检索
- **无训练型 RAG**: In-context RALM (Ram et al., 2023) 动态检索；FLARE (Jiang et al., 2023) 主动判断何时检索
- **噪声鲁棒性**: Chen et al. (2024) 的 RGB 基准评估 RAG 噪声鲁棒性；Yu et al. (2024) 用 context ranking 提升鲁棒性

## 评分 ⭐⭐⭐⭐

- **创新性**: ⭐⭐⭐⭐ 多 Agent 过滤 + 自适应阈值的组合新颖实用，log-prob 差值量化相关性简洁有效
- **实验充分性**: ⭐⭐⭐⭐ 4 个基准、多种基线对比、消融实验充分、case study 直观
- **实用价值**: ⭐⭐⭐⭐⭐ Training-free、即插即用、超参数少（仅 n=0），对 RAG 实践有直接价值
- **写作质量**: ⭐⭐⭐⭐ 框架图清晰，自适应阈值的动机从分数分布观察自然推导

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] MASS-RAG: Multi-Agent Synthesis Retrieval-Augmented Generation](../../ACL2026/information_retrieval/mass-rag_multi-agent_synthesis_retrieval-augmented_generation.md)
- [\[ACL 2025\] GeAR: Generation Augmented Retrieval](gear_generation_augmented_retrieval.md)
- [\[ACL 2025\] Mitigating Lost-in-Retrieval Problems in RAG Multi-Hop QA](mitigating_lost-in-retrieval_problems_in_retrieval_augmented_multi-hop_question_.md)
- [\[ACL 2025\] Typed-RAG: Type-Aware Decomposition of Non-Factoid Questions for Retrieval-Augmented Generation](typed-rag_type-aware_decomposition_of_non-factoid_questions_for_retrieval-augmen.md)
- [\[ACL 2025\] HASH-RAG: Bridging Deep Hashing with Retriever for Efficient, Fine Retrieval and Augmented Generation](hash-rag_bridging_deep_hashing_with_retriever_for_efficient_fine_retrieval_and_a.md)

</div>

<!-- RELATED:END -->
