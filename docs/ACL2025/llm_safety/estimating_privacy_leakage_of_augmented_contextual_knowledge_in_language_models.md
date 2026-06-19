---
title: >-
  [论文解读] Estimating Privacy Leakage of Augmented Contextual Knowledge in Language Models
description: >-
  [ACL 2025][LLM安全][隐私泄露] 本文提出context influence指标，基于差分隐私框架量化语言模型在解码时对增强上下文知识的隐私泄露程度，并系统分析了模型大小、上下文大小、生成位置等因素对隐私泄露的影响。 - 领域现状：LLM依赖参数知识（预训练编码）和上下文知识（prompt中的信息）完成QA等任…
tags:
  - "ACL 2025"
  - "LLM安全"
  - "隐私泄露"
  - "上下文知识"
  - "差分隐私"
  - "语言模型"
  - "RAG安全"
---

# Estimating Privacy Leakage of Augmented Contextual Knowledge in Language Models

**会议**: ACL 2025  
**arXiv**: [2410.03026](https://arxiv.org/abs/2410.03026)  
**代码**: [james-flemings/context_influence](https://github.com/james-flemings/context_influence)  
**领域**: AI安全  
**关键词**: 隐私泄露、上下文知识、差分隐私、语言模型、RAG安全

## 一句话总结
本文提出context influence指标，基于差分隐私框架量化语言模型在解码时对增强上下文知识的隐私泄露程度，并系统分析了模型大小、上下文大小、生成位置等因素对隐私泄露的影响。

## 研究背景与动机
- **领域现状**：LLM依赖参数知识（预训练编码）和上下文知识（prompt中的信息）完成QA等任务。RAG（检索增强生成）等方法在prompt中注入外部上下文已成为主流。
- **核心矛盾**：上下文可能包含敏感信息，LM在回答时可能泄露这些私密数据。但直接比较LM输出与上下文的方法会**高估**隐私风险——因为LM参数知识中可能已包含相同信息。
- **关键例子**：如果上下文包含John Doe的地址，且LM输出了该地址，直接比较会判定为"上下文泄露"。但若移除上下文后LM仍输出该地址，说明泄露来自参数知识而非上下文。
- **核心idea**：需要一个能**分离参数知识贡献**的隐私度量——比较有/无特定上下文子集时输出分布的差异，借鉴差分隐私框架。

## 方法详解

### 整体框架
1. 定义context influence指标（基于ex-post per-instance DP）
2. 提出Context Influence Decoding (CID)控制上下文影响程度
3. 建立context influence与PMI（点互信息）的理论联系
4. 系统实验分析影响隐私泄露的各种因素

### 关键设计
1. **Context Influence (Definition 3.1)**

    - 核心思路：度量移除上下文中第 $i$ 个 $n$-gram $D_{i,n}$ 后输出概率的变化
    - 定义：$\tau_{i,n} = |\log p_\theta(y_t | D, \mathbf{x}, \mathbf{y}_{<t}) - \log p_\theta(y_t | D \setminus D_{i,n}, \mathbf{x}, \mathbf{y}_{<t})|$
    - $n=1$ 对应词级隐私，$n=|D|$ 对应文档级隐私
    - 对整个响应的context influence为各token的求和（类比DP的组合性质）

2. **Context Influence Decoding (CID)**

    - 将CAD（Context-aware Decoding）重新表述为参数 $\lambda$ 控制的上下文影响
    - $\bar{p}_{\theta,\lambda}(y_t) = \sigma[(\lambda \cdot \text{logit}_\theta(y_t|D,\mathbf{x}) + (1-\lambda) \cdot \text{logit}_\theta(y_t|\mathbf{x})) / T]$
    - $\lambda=0$：仅用参数知识（无上下文隐私泄露）
    - $\lambda=1$：正常解码
    - $\lambda>1$：放大上下文影响（减少幻觉但增加隐私风险）

3. **理论联系 (Theorem 3.1)**

    - Context influence正比于 $\lambda \cdot |\text{pmi}(D) - \text{pmi}(D \setminus D_{i,n})|$
    - 两个关键泄露因素：(a) 上下文相对参数知识的OOD程度（PMI差异大），(b) 解码时上下文放大程度（$\lambda$ 大）

### 损失函数 / 训练策略
本文不涉及模型训练，是一个分析/度量框架。关键公式：
- DP保证：CID可通过选择合适 $\lambda^*$ 满足 $\epsilon$-DP（Theorem B.1）
- 实际使用：通过 $\hat{\tau}_{i,n}(p_\theta) = \frac{1}{|\mathcal{D}|}\sum_{(D,\mathbf{x})} \tau_{i,n}(\cdot)$ 估计期望context influence

## 实验关键数据

### 主实验：Context Influence与输入反刍

| 模型 | 数据集 | $\lambda$ | Context Influence | Repeat Prompts | Rouge Prompts |
|------|--------|-----------|-------------------|----------------|---------------|
| LLaMA 3 8B | CNN-DM | 0.5 | 15.97 | 8 | 109 |
| LLaMA 3 8B | CNN-DM | 1.0 | 64.61 | 285 | 632 |
| LLaMA 3 8B | CNN-DM | 1.5 | 98.99 | 429 | 882 |
| OPT 1.3B | PubMedQA | 1.0 | 45.66 | 47 | 251 |
| GPT-Neo 1.3B | PubMedQA | 1.0 | 38.79 | 54 | 268 |

### 关键对比：参数知识的影响

| 对比 | 说明 |
|------|------|
| OPT 1.3B vs GPT-Neo 1.3B (PubMedQA) | GPT-Neo预训练含PubMed → context influence更低(38.79 vs 45.66) → 但Repeat/Rouge Prompts反而更高 → 传统度量高估了上下文泄露 |

### 因素分析

| 因素 | 关键发现 |
|------|---------|
| $\lambda$ (上下文放大) | $\lambda$从1.0→1.5：ROUGE-L提升10%但输入反刍增50% |
| 模型大小 | 更大模型context influence更低（可更多依赖参数知识） |
| 上下文大小 | $|D| \leq 32$ 时context influence极低；$|D| \geq 256$ 后趋于稳定 |
| 响应位置 | 前10个token受上下文影响最大，之后逐渐减弱 |
| 预训练 vs 微调 | LLaMA 3 IT远比LLaMA 3受上下文影响大 |
| n-gram位置 | 上下文前部的n-gram影响最大（位置偏差） |

### 关键发现
- Context influence能正确归因隐私泄露：OPT在PubMedQA上context influence更高（因为预训练不含PubMed），而传统指标错误地显示GPT-Neo泄露更多
- 放大上下文（$\lambda>1$）可减少幻觉但显著增加隐私风险——存在隐私-效用权衡
- 模型微调（SFT+RLHF）提高了上下文利用能力，同时也增大了隐私泄露

## 亮点与洞察
- **理论基础扎实**：将DP分析框架引入上下文隐私度量，用ex-post per-instance DP替代粗粒度的 $\epsilon$-DP，能针对具体上下文和输出计算隐私损失
- **区分两种知识来源**：核心贡献是分离参数知识和上下文知识的贡献，避免传统方法的高估问题
- **分析全面**：系统考察了模型大小、上下文大小、生成位置、n-gram粒度等多维因素
- **实用指导**：(1)敏感信息放在上下文后部可降低泄露；(2)可采用自适应隐私级别（开头严格、后期放松）；(3)预训练数据选择影响隐私保证的可信度

## 局限与展望
- 未考虑模型解码时的entropy对context influence的影响（更自信的模型influence更小，可能产生误导）
- 仅关注上下文隐私泄露，不涉及参数知识的隐私泄露（记忆化问题）
- 仅使用temperature sampling，未分析top-p/top-k等采样策略的影响
- **可改进方向**：能否设计一种自适应解码策略，在检测到高context influence时自动降低 $\lambda$，实现实时隐私保护？

## 相关工作与启发
- 与参数知识泄露研究互补：Carlini et al. 研究训练数据提取，本文研究推理时上下文泄露
- 与RAG安全相关：Zeng et al., Qi et al. 研究RAG数据提取攻击，但隐含假设上下文不在参数知识中，本文指出这会高估风险
- 与Context-aware Decoding的关系：本文将其重新表述为隐私控制工具（CID）
- 对隐私保护LLM部署的启示：在RAG场景中需要同时考虑参数知识和上下文知识的隐私属性

## 评分
- 新颖性: ⭐⭐⭐⭐ 将DP框架应用于上下文隐私度量是新颖的，区分两种知识来源的思路好
- 实验充分度: ⭐⭐⭐⭐ 多模型多数据集多因素分析详尽，定性定量结合
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，motivation用Figure 1说明直观，整体逻辑严密
- 价值: ⭐⭐⭐⭐ 对RAG隐私评估有直接指导意义，为隐私保护解码策略提供理论基础

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Robust Data Watermarking in Language Models by Injecting Fictitious Knowledge](robust_data_watermarking_in_language_models_by_injecting_fictitious_knowledge.md)
- [\[ACL 2026\] Privacy Collapse: Benign Fine-Tuning Can Break Contextual Privacy in Language Models](../../ACL2026/llm_safety/privacy_collapse_benign_fine-tuning_can_break_contextual_privacy_in_language_mod.md)
- [\[AAAI 2026\] Privacy-protected Retrieval-Augmented Generation for Knowledge Graph Question Answering](../../AAAI2026/llm_safety/privacy-protected_retrieval-augmented_generation_for_knowledge_graph_question_an.md)
- [\[ACL 2025\] Answer When Needed, Forget When Not: Language Models Pretend to Forget via In-Context Knowledge Unlearning](answer_when_needed_forget_when_not_language_models_pretend_to_forget_via_in-cont.md)
- [\[ACL 2025\] The Tug of War Within: Mitigating the Fairness-Privacy Conflicts in Large Language Models](tug_of_war_fairness_privacy.md)

</div>

<!-- RELATED:END -->
