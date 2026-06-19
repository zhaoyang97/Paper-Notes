---
title: >-
  [论文解读] Evaluation of LLMs in Medical Text Summarization: The Role of Vocabulary Adaptation in High OOV Settings
description: >-
  [ACL 2025 (Findings)][医疗NLP][vocabulary adaptation] 系统性基准研究发现 LLM 在高 OOV（词汇外词）和高新颖性医学文本摘要场景下性能显著下降，并通过多种词汇适配策略（MEDVOC、MEDVOC-LLM、ScafFix）证明即使 Llama-3.1（128K 词汇量）仍受过度分片问题困扰，词汇适配可带来显著改善。
tags:
  - "ACL 2025 (Findings)"
  - "医疗NLP"
  - "vocabulary adaptation"
  - "medical summarization"
  - "OOV"
  - "LLM"
  - "tokenization"
  - "continual pretraining"
---

# Evaluation of LLMs in Medical Text Summarization: The Role of Vocabulary Adaptation in High OOV Settings

**会议**: ACL 2025 (Findings)  
**arXiv**: [2505.21242](https://arxiv.org/abs/2505.21242)  
**代码**: [GitHub](https://github.com/gb-kgp/LLM-MedicalSummarization-Benchmark)  
**领域**: 医疗NLP  
**关键词**: vocabulary adaptation, medical summarization, OOV, LLM, tokenization, continual pretraining

## 一句话总结

系统性基准研究发现 LLM 在高 OOV（词汇外词）和高新颖性医学文本摘要场景下性能显著下降，并通过多种词汇适配策略（MEDVOC、MEDVOC-LLM、ScafFix）证明即使 Llama-3.1（128K 词汇量）仍受过度分片问题困扰，词汇适配可带来显著改善。

## 研究背景与动机

**领域现状**: LLM 已在医学文本摘要中取得成功，主要依赖上下文学习（ICL）和参数高效微调（QLoRA）。近期 ClinSumm 等工作探索了多种 LLM 适配策略，但这些研究仅报告全测试集的聚合性能分数。

**现有痛点**: 现有研究缺乏在困难场景下的细粒度评估。LLM 的 tokenizer 对医学专业词汇存在严重的**过度分片**（over-fragmentation）问题——例如"cardiomyopathy"被 Llama-2 切分成 6 个 token（'_card', 'iom', 'y', 'op', 'ath', 'y'），导致语义信息丢失。

**核心矛盾**: 即使是拥有 128K 词汇量的 Llama-3.1，其在医学领域的 fragment score 仍比通用领域高 **13.08%**，33% 以上的医学词汇被切分超过 3 次。这种词汇不匹配在编码阶段损失语义，在生成阶段增加 token 数量。

**本文目标**: (1) 在高 OOV 和高新颖性（参考摘要中出现源文档未出现的领域词）场景下系统评估 LLM 性能；(2) 验证词汇适配策略能否有效缓解这一问题。

**切入角度**: 设计 360 种评估组合（4 模型 × 3 词汇策略 × 3 数据集 × 2 预训练策略 × 5 细粒度场景），全面基准测试词汇适配对 LLM 医学摘要的影响。

**核心 idea**: 通过细粒度基准测试证明 LLM 在高 OOV 医学摘要场景下性能大幅下降，并提出 ScafFix（无脚手架 token 的词汇适配）来有效改善。

## 方法详解

### 整体框架

三步词汇适配流程：(1) 从目标领域数据集生成候选词汇 token；(2) 通过效用函数（如 fragment score）选择重要词汇；(3) 学习新增词汇的嵌入并整合到 LLM。

### 关键设计

1. **MEDVOC**: 原始 SOTA 词汇适配策略——从医学语料（PubMed Abstract Collection, PAC）构建候选词汇 $V_{PAC}$，从目标任务数据集构建 $V_{TGT}$，通过超参搜索在 $V_{PAC} \cap V_{TGT}$ 中找到最优词汇集。效用函数为 fragment score $= \frac{1}{|\mathcal{C}|}\sum_{w \in \mathcal{C}} \text{subwords}(w, \mathcal{V})$，即平均每个词被分成的子词数。

2. **MEDVOC-LLM**: 针对 LLM tokenizer 清洗 MEDVOC 结果——移除在目标任务训练集参考摘要中一次都未出现的词汇，以及数字和标点混合的无效 token（如"-9,"），使词汇适配更契合 LLM 的分词方案。

3. **ScafFix（核心贡献）**: 解决现有词汇适配的**脚手架 token 开销**问题。以"cholesterol"为例，添加该词需要额外添加中间 token "chole"（因为 BPE 合并规则逐对操作），这些中间 token 在整词添加后几乎不再使用，反而被欠训练。ScafFix 直接按词频选取前 $x$ 个医学词（$x=500$, 步长 50），跳过分词阶段的子词构建，然后使用 **AdaptBPE** 分词方案——先检查输入 token 的最长前缀是否在新增词汇中，保留不切分，再对剩余部分运行标准 BPE。Llama-3.1 中约 **20%** 的添加 token 属于这种冗余脚手架 token。

4. **持续预训练策略**: 新增词汇嵌入初始化为现有子词嵌入的平均值，然后用 LoRA (rank=32, alpha=64) 在 20K PubMed 文档上训练。两种策略：

    - **End-to-End**: 冻结基础层，解冻输入/输出嵌入层和 LoRA adapter，端到端训练 5 epochs
    - **Two-Stage**: 先冻结 LoRA 只训练嵌入层 2 epochs/10K 样本，再解冻 LoRA 一起训练 3 epochs/20K 样本。更稳定，避免嵌入空间过拟合

### 细粒度评估设置

5 种困难场景（取前 10% 分位）：DifficultRS（参考摘要中高 OOV 浓度）、DifficultSD（源文档中高 OOV 浓度）、NovelRS（参考摘要中高新颖词浓度）、AllSD/AllRS（所有 OOV 浓度）。

## 实验关键数据

### 主实验（Rouge-L，选关键结果）

| 数据集 | 模型 | BASE | CPT-only | MEDVOC-LLM | ScafFix |
|--------|------|------|----------|------------|---------|
| PubMedQA | Llama-2 | 26.33 | 27.12 | 26.90 | **27.61** |
| PubMedQA | Llama-3.1 | 28.10 | 26.62 | 27.69 | 27.67 |
| EBM | Llama-2 | 18.56 | 19.13 | 19.27 | 18.65 |
| EBM | Llama-3.1 | 20.04 | 20.13 | **20.75** | 20.79 |
| BioASQ-S | Llama-2 | 32.12 | 33.30 | 32.40 | **32.88** |
| BioASQ-S | Llama-3.1 | 35.25 | 36.01 | **37.15** | 36.70 |
| BioASQ-M | Llama-2 | 28.50 | 27.22 | 24.50 | **26.16** |
| BioASQ-M | Llama-3.1 | 29.28 | 27.56 | 27.45 | **28.91** |

### 困难场景下的性能（Overall = 四个场景平均）

| 数据集 | Llama-2 BASE → 最佳适配 | 改善 | Llama-3.1 BASE → 最佳适配 | 改善 |
|--------|------------------------|------|--------------------------|------|
| PubMedQA | 23.51 → 26.18 | +11.4% | 24.58 → 25.65 | +4.4% |
| EBM | 15.77 → 16.46 (CPT) | +4.4% | 15.44 → 17.15 | +11.1% |
| BioASQ-S | 28.62 → 30.21 | +5.6% | 27.52 → 32.04 | +16.4% |
| BioASQ-M | 26.33 → 26.78 | +1.7% | 26.51 → 26.94 | +1.6% |

### 关键研究问题发现

- **RQ1**: 词汇适配在 5/8 全测试集设置上超越 BASE，平均提升 Llama-2 3.68%、Llama-3.1 4.57%
- **RQ2**: CPT-Only（不做词汇适配）在高 OOV 场景有改善但不如词汇适配——词汇适配在 13/16 高 OOV 设置超越 CPT-Only
- **RQ3**: 高 OOV 场景下词汇适配在 14/16 设置超越 BASE，平均改善 Llama-2 8.74%、Llama-3.1 14.64%
- **RQ4**: 高新颖性场景词汇适配在 6/8 设置超越 BASE，平均改善 Llama-2 11.92%、Llama-3.1 18.03%
- **人工评估**: 医学专家认为词汇适配产生了更相关、更连贯、更忠实的摘要
- **Concept Score** 显示词汇适配在 5/6 设置上提升了事实忠实度：Llama-2 平均 18.75%，Llama-3.1 14.82%

### 关键发现

- **即使 128K 词汇量的 Llama-3.1 仍需要词汇适配**——医学域 fragment score 比通用域高 13.08%
- **ScafFix 在高新颖性场景表现最好**（5/8 设置优胜），**MEDVOC-LLM 在高 OOV 场景最好**
- **脚手架 token 是真实问题**：Llama-3.1 添加词汇时产生约 20% 冗余中间 token
- Two-Stage 预训练通常比 End-to-End 更稳定

## 亮点与洞察

- **细粒度评估框架**非常有价值——现有 LLM 医学评估几乎都只报告全集性能，本文系统揭示了性能在 OOV/新颖词密集区域大幅下降的现象
- **ScafFix 的洞察很实用**：直接将完整医学词加入词汇而非依赖 BPE 子词分解，避免了欠训练的脚手架 token 问题
- **Fragment score 作为诊断工具**：量化词汇不匹配的严重程度，可指导是否需要词汇适配
- **20K 样本持续预训练即可**（6 小时 on A100），100K 样本（40 小时）性能相当，性价比极高

## 局限与展望

- 仅评估了 ICL（上下文学习），未测试 QLoRA 微调后词汇适配的效果
- 词汇适配在 BioASQ-M 上改善有限，说明在多文档长上下文场景下效果减弱
- ScafFix 的 AdaptBPE 方案在推理时引入了额外的分词逻辑，与标准 HuggingFace 生态的兼容性未讨论
- 仅考虑英语医学文本，多语言场景的词汇适配可能更有意义
- 缺少与指令微调模型（如 Med-Gemini、Med-PaLM）的对比

## 相关工作与启发

- **MEDVOC** (Balde et al., 2024b) 是本文最重要的基线和出发点，ScafFix 是其在 LLM 上的改进版
- **ClinSumm** (Van Veen et al., 2024) 提供了 LLM 医学摘要的基准测试框架
- **AdaptBPE** (Balde et al., 2024a) 提供了 ScafFix 使用的替代分词方案
- 启发：LLM 的 tokenizer 是一个被严重低估的瓶颈，尤其在专业领域——改善 tokenization 可能比增大模型规模更有效

## 评分

- **新颖性**: ⭐⭐⭐ — ScafFix 有新意，但整体是基准测试研究，方法贡献有限
- **实验充分度**: ⭐⭐⭐⭐⭐ — 360 种评估组合，4 个模型 × 3 数据集 × 5 场景 × 多策略，极其全面
- **写作质量**: ⭐⭐⭐⭐ — 结构化 RQ 分析清晰，表格丰富
- **价值**: ⭐⭐⭐⭐ — 揭示了 LLM 在医学领域的 tokenization 瓶颈，对实际应用有指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Improving Automatic Evaluation of LLMs in Biomedical Relation Extraction via LLMs-as-the-Judge](biore_llm_judge_evaluation.md)
- [\[NeurIPS 2025\] Shallow Robustness, Deep Vulnerabilities: Multi-Turn Evaluation of Medical LLMs](../../NeurIPS2025/medical_nlp/shallow_robustness_deep_vulnerabilities_multi-turn_evaluation_of_medical_llms.md)
- [\[NeurIPS 2025\] Faithful Summarization of Consumer Health Queries: A Cross-Lingual Framework with LLMs](../../NeurIPS2025/medical_nlp/faithful_summarization_of_consumer_health_queries_a_cross-lingual_framework_with.md)
- [\[ICML 2026\] MedCase-Structured: A Text-to-FHIR Dataset for Benchmarking Diagnostic Reasoning in Clinically Realistic EHR Settings](../../ICML2026/medical_nlp/medcase-structured_a_text-to-fhir_dataset_for_benchmarking_diagnostic_reasoning_.md)
- [\[ACL 2026\] MHSafeEval: Role-Aware Interaction-Level Evaluation of Mental Health Safety in Large Language Models](../../ACL2026/medical_nlp/mhsafeeval_role-aware_interaction-level_evaluation_of_mental_health_safety_in_la.md)

</div>

<!-- RELATED:END -->
