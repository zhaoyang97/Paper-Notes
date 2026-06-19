---
title: >-
  [论文解读] Pap2Pat: Benchmarking Outline-Guided Long-Text Patent Generation with Patent-Paper Pairs
description: >-
  [ACL 2025][LLM评测][专利生成] 构建了包含 1.8k 专利-论文配对的 Pap2Pat 基准，提出基于大纲的分块专利描述生成方法 COPGen，并设计了基于 NLI 的事实性/覆盖率/风格评估指标，系统评测了当前 LLM 在超长专利文档生成上的能力与不足。 专利申请是一个漫长而昂贵的流程…
tags:
  - "ACL 2025"
  - "LLM评测"
  - "专利生成"
  - "长文档生成"
  - "大纲引导"
  - "Patent-Paper Pairs"
  - "分块生成"
---

# Pap2Pat: Benchmarking Outline-Guided Long-Text Patent Generation with Patent-Paper Pairs

**会议**: ACL 2025  
**arXiv**: [2410.07009](https://arxiv.org/abs/2410.07009)  
**代码**: [有](https://github.com/boschresearch/Pap2Pat)  
**领域**: LLM评测  
**关键词**: 专利生成, 长文档生成, 大纲引导, Patent-Paper Pairs, 分块生成

## 一句话总结

构建了包含 1.8k 专利-论文配对的 Pap2Pat 基准，提出基于大纲的分块专利描述生成方法 COPGen，并设计了基于 NLI 的事实性/覆盖率/风格评估指标，系统评测了当前 LLM 在超长专利文档生成上的能力与不足。

## 研究背景与动机

专利申请是一个漫长而昂贵的流程，需要深厚的技术知识和专利法专长。尽管 NLP 已在专利检索、专利全景分析等任务上得到应用，但专利文档的自动撰写仍基本依赖人工。在一份专利中，**描述部分（description）平均占据超过 90% 的篇幅**（涵盖技术领域、背景、摘要、详细描述等章节），对这部分的自动生成支持将带来最大的生产力提升，但也面临最大的挑战：

- **极端的文档长度**：Pap2Pat 中专利描述平均 18k tokens，部分超过 180k tokens，远超当前 LLM 的生成能力上限
- **先前工作的局限**：大多数专利生成研究只关注标题、摘要和权利要求（claims），忽视了占比最大的描述部分；任务设定不够现实，缺乏开放基准
- **数据获取困难**：发明人通常提交的发明报告（IR）是机密的。但在研究实验室中，预发表论文常作为发明报告使用，形成了"专利-论文对"（Patent-Paper Pairs），这为构建开放数据集提供了可能
- **评估标准缺失**：专利文本长且技术性强，传统文本相似度指标（ROUGE）在长文档上效果不佳，缺乏针对事实性、覆盖率和语言风格的专用评估方法

## 方法详解

### 整体框架

本文工作分为三大部分：(1) Pap2Pat 基准数据集构建，(2) COPGen 分块大纲引导生成方法，(3) 专利生成评估指标设计。

### 关键设计

1. **Pap2Pat 数据集构建**：

    - 从 USPTO 670 万专利申请中出发，使用 SemOpenAlex 查询具有作者重叠和日期相关的论文
    - 通过标题/摘要术语重叠、候选唯一性、开放许可证等多级过滤，从 93 万初始候选缩减至 1.8k 高质量专利-论文对
    - 人工验证精度：随机抽取 60 对，55/60 (91.7%) 为精确匹配
    - 使用 Llama-3 70B 从原始专利自动生成三种粒度的大纲（短/中/长，平均 37/74/150 个要点），大纲要点平均 5.4 词
    - 数据划分：train 1000, val 242, test 500, 另有 nc-test 71（2024年专利，避免数据泄露）

2. **COPGen：分块大纲引导专利生成**：

    - **核心思想**：将长大纲分割为多个"块"（chunks），每个块独立生成一段专利文本，最后拼接
    - **Token 分配与分块**：默认每块分配 2k 指令 tokens + 3k 论文上下文 tokens + 2k 专利输出 tokens；根据每个要点平均对应的字符数决定每块包含多少要点
    - **论文上下文选择**：使用 BM25 以当前块的大纲要点为查询，从论文中检索最相关段落。始终包含论文摘要和所有标题，按相关性排序填充至 token 上限
    - **长度控制机制**：基于 Bai et al. 的发现——LLM 生成长度只在较短时可控——通过减少每块的输出 token 分配（增加总块数）来精确控制总输出长度。校准后使用 400 tokens/块的设定使平均长度匹配参考专利
    - **全局上下文**：每块生成时还包含前序块的大纲，提供全局文档结构感知

3. **评估指标设计**：

    - **事实性（Factuality）**：使用 NLI 基础的 SCALE 指标，计算参考专利（和论文）对生成文本的蕴含分数（Ref→Gen, Ref+Pap→Gen）
    - **覆盖率（Coverage）**：反向计算，以生成文本为前提、参考专利为假设（Gen→Ref）
    - **语言风格**：使用 n-gram 档案相似度（1-4 gram）+ StyloMetrix 语言特征（196 种规则检测）衡量专利风格相似度
    - **重复度**：使用滑动窗口的 Repetition Rate (RR)，区分合理的法律语言重复与无意义的循环生成

### 损失函数 / 训练策略

- COPGen 本身不涉及训练，是一个推理时框架
- 实验中同时测试了 Llama-3 8B 在 Pap2Pat 训练集上的监督微调（SFT），探索微调对风格和事实性的影响

## 实验关键数据

### 主实验：不同模型与方法的生成效果

| 方法 | Tokens | Coverage (Gen→Ref) | Factuality (Ref→Gen) | Style |
|------|--------|-------------------|---------------------|-------|
| 参考专利（上界） | 18.1k (100%) | 88.6 | 88.5 | 100 |
| 论文原文 | 8.1k (45%) | 44.8 | 46.5 | 47.2 |
| Qwen2-72B（单次调用） | 2.8k (16%) | 40.3 | 65.8 | 39.6 |
| COPGen + Llama-3 8B | 9.6k (53%) | 40.3 | 60.8 | 43.2 |
| COPGen + Llama-3 70B | 6.1k (34%) | 42.7 | 64.5 | 49.5 |
| COPGen + Qwen2-72B | 8.1k (45%) | 44.1 | 62.5 | 47.5 |
| COPGen + Qwen2-72B (校准长度) | **18.1k (100%)** | **46.8** | 59.7 | 47.8 |
| COPGen + Llama-3 8B SFT | 27.5k (152%) | 42.0 | 49.3 | **59.4** |

### 消融实验：Input 组件的影响

| 变体 | Coverage | Factuality (Ref+Pap→Gen) |
|------|----------|-------------------------|
| COPGen + Qwen2-72B（完整） | 44.1 | 67.9 |
| 去掉论文输入 | 38.9 | 66.6 |
| 去掉大纲输入 | 34.9 | 75.3 |

### 关键发现

1. **COPGen 显著提升覆盖率**：同一模型（Qwen2-72B）从单次调用的 40.3 提升至分块生成的 44.1（Coverage），说明分块策略有效缓解了 LLM 长文本生成的困难
2. **长度控制有效**：通过校准 token 分配，可以将输出长度精确匹配到参考专利长度（18.1k），且覆盖率进一步提升至 46.8
3. **微调双刃剑**：SFT 使风格分数从 43.2 跃升至 59.4（更像专利），但事实性从 60.8 暴跌至 49.3，出现更多幻觉。重复率也大幅上升
4. **论文是关键信息源**：去掉论文输入后覆盖率下降 5.2 个点；去掉大纲后覆盖率下降更多（9.2 个点），且事实性反而升高（模型更"保守"但内容更少）
5. **4-gram 重叠仅 8.3%**：专利和论文虽描述同一发明，但语言风格差异巨大，任务远非简单的改写

## 亮点与洞察

- **数据集构建方法论价值高**：将"论文常作为发明报告使用"这一学术界/工业界的未被充分利用的特点转化为数据资源，构建方法可迁移到其他领域
- **分块策略优雅有效**：COPGen 无需特殊训练，通过工程化的上下文管理和长度控制实现了可控的超长文档生成，且支持并行生成各块
- **评估指标设计细致**：将事实性、覆盖率和风格解耦为独立指标，比单一 ROUGE 分数信息量大得多。SCALE 的 BM25 预筛 + 采样策略也使其可扩展到超长文档
- **微调导致幻觉增加的发现**：这提供了一个重要警示——在专业领域微调 LLM 可能学到风格但牺牲事实准确性

## 局限与展望

- 数据集规模受限于论文开放许可证的限制（仅 1.8k 对），扩展需要学术出版界更广泛的开放获取
- COPGen 各块独立生成可能导致块间连贯性不足，未来可引入块间依赖建模
- 评估仍依赖自动指标，人类专家评估成本高昂，本文的人类案例研究仅为小规模验证
- 仅评测了开源 LLM，未包含 GPT-4 等闭源模型（出于可复现性考虑）
- 大纲由参考专利自动生成，而非由真实专利律师提供，与实际场景仍有差距

## 相关工作与启发

本文与长文档生成（Sun et al. 2022; Wang et al. 2024d）、大纲引导生成（Shao et al. 2024）、以及专利 NLP（Christofidellis et al. 2022; Wang et al. 2024c）三个方向紧密相关。启发性在于：长文档生成的"分治"策略（先规划大纲、再分块生成）可能是当前 LLM 能力边界下的最优实践，该思路可推广到学术论文、技术报告等其他长文档生成任务。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 数据集构建方法新颖（PPP 匹配管线），分块生成框架虽简单但针对性强，评估指标设计有创意  
- **实验充分度**: ⭐⭐⭐⭐ — 多模型对比、消融实验、长度控制分析、微调分析、人类评估，维度全面  
- **写作质量**: ⭐⭐⭐⭐⭐ — 结构极为清晰，动机-方法-评估逻辑通畅，图表设计精美  
- **价值**: ⭐⭐⭐⭐ — 填补了专利描述自动生成的基准空白，开放数据和代码，对法律 AI 和长文档生成社区均有直接帮助

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] EditInspector: A Benchmark for Evaluation of Text-Guided Image Edits](editinspector_a_benchmark_for_evaluation_of_text-guided_image_edits.md)
- [\[ACL 2025\] Improving the Calibration of Confidence Scores in Text Generation Using the Output Distribution's Characteristics](calibration_confidence_text_gen.md)
- [\[ACL 2025\] Atomic Calibration of LLMs in Long-Form Generations](atomic_calibration_of_llms_in_long-form_generations.md)
- [\[ACL 2026\] Illusions of the Gold Standard: A Large-scale Analysis of Human Evaluation Protocols for Long-form Text Generation](../../ACL2026/llm_evaluation/illusions_of_the_gold_standard_a_large-scale_analysis_of_human_evaluation_protoc.md)
- [\[ACL 2025\] MARS: Benchmarking the Metaphysical Reasoning Abilities of Language Models with a Multi-task Evaluation Dataset](mars_benchmarking_the_metaphysical_reasoning_abilities_of_language_models_with_a.md)

</div>

<!-- RELATED:END -->
