---
title: >-
  [论文解读] Are LLM Belief Updates Consistent with Bayes' Theorem?
description: >-
  [ICML 2025 (Workshop on Assessing World Models)][LLM评测][Bayesian coherence] 本文提出贝叶斯一致性系数（BCC）来量化 LLM 的信念更新是否符合贝叶斯定理，发现更大、更强的预训练模型在给定新证据时，其信念更新与贝叶斯定理更一致。
tags:
  - "ICML 2025 (Workshop on Assessing World Models)"
  - "LLM评测"
  - "Bayesian coherence"
  - "belief update"
  - "LLM evaluation"
  - "probabilistic reasoning"
  - "scaling"
---

# Are LLM Belief Updates Consistent with Bayes' Theorem?

**会议**: ICML 2025 (Workshop on Assessing World Models)  
**arXiv**: [2507.17951](https://arxiv.org/abs/2507.17951)  
**代码**: 无  
**领域**: LLM评测  
**关键词**: Bayesian coherence, belief update, LLM evaluation, probabilistic reasoning, scaling

## 一句话总结
本文提出贝叶斯一致性系数（BCC）来量化 LLM 的信念更新是否符合贝叶斯定理，发现更大、更强的预训练模型在给定新证据时，其信念更新与贝叶斯定理更一致。

## 研究背景与动机

### 领域现状

**领域现状**：领域现状**: LLM 在推理和决策任务中表现出色，但其内部是否遵循概率推理的基本原则尚不清楚。

**现有痛点**: 缺乏系统的方法来评估 LLM 是否能在接收到新证据后合理地更新其"信念"（对命题的置信度）。

**核心矛盾**: LLM 虽然不是显式的概率模型，但在实际使用中经常被要求做出涉及不确定性的判断。如果信念更新不符合贝叶斯原则，可能导致不一致和不可靠的推理。

**本文解决什么**: 量化 LLM 的 in-context 信念更新与贝叶斯定理的一致性程度。

**切入角度**: 构造专门的数据集，通过比较 LLM 在看到证据前后的置信度变化与贝叶斯定理的预测值来计算一致性。

**核心 idea**: 提出 BCC 指标，系统测量多个模型家族的贝叶斯一致性，发现模型规模和能力与一致性正相关。

### 解决思路

**本文目标**：### 整体框架
作者设计了评估流程：(1) 构造包含命题、证据和先验/后验概率的数据集；(2) 让 LLM 在有/无证据条件下分别输出对命题的置信度；(3) 用贝叶斯定理计算理论后验并与 LLM 实际更新进行比较。


## 方法详解

### 整体框架
作者设计了评估流程：(1) 构造包含命题、证据和先验/后验概率的数据集；(2) 让 LLM 在有/无证据条件下分别输出对命题的置信度；(3) 用贝叶斯定理计算理论后验并与 LLM 实际更新进行比较。

### 关键设计
1. **Bayesian Coherence Coefficient (BCC)**: 衡量 LLM 置信度更新与贝叶斯定理预测之间的一致性。给定先验 $P(H)$、似然 $P(E|H)$ 和 LLM 的后验 $P_{LLM}(H|E)$，BCC 衡量后者与贝叶斯后验 $P(H|E) = \frac{P(E|H)P(H)}{P(E)}$ 之间的偏差。设计动机：需要一个可量化的、与模型无关的指标来跨模型家族比较。

2. **数据集构造**: 生成包含多种领域（科学、医学、日常推理等）的命题-证据对，确保覆盖不同先验概率和证据强度。使用人工和自动化方法确保数据质量和多样性。

3. **多维度评估**: 将 BCC 与模型参数量、训练数据量以及在常见 benchmark 上的分数进行关联分析，探索哪些因素最能预测贝叶斯一致性。

### 损失函数 / 训练策略
本文是评估性工作，核心在于评测协议的设计。通过 prompt 工程让 LLM 输出 0-1 之间的置信度数值。

## 实验关键数据

### 主实验

| 模型家族 | 参数量 | BCC ↑ | MMLU | 趋势 |
|----------|--------|-------|------|------|
| 小模型 | <7B | 较低 | 较低 | 基线 |
| 中等模型 | 7B-30B | 中等 | 中等 | 提升 |
| 大模型 | 30B-70B | 较高 | 较高 | 显著提升 |
| 最大模型 | >70B | 最高 | 最高 | 趋于饱和 |

### 消融实验

| 配置 | BCC 变化 | 说明 |
|------|---------|------|
| 不同 prompt 格式 | 有波动 | prompt 设计影响置信度提取的稳定性 |
| 不同领域命题 | 领域依赖 | 科学领域一致性通常较高 |
| 证据强度变化 | 一致性下降 | 强证据下更一致，弱证据时偏差增大 |

### 关键发现
- 更大和更有能力的 LLM 信念更新与贝叶斯定理更一致
- 模型 benchmark 分数与 BCC 正相关
- 即使最好的模型也远未达到完美的贝叶斯一致性

## 亮点与洞察
- 首次系统性地用贝叶斯定理评估 LLM 的信念更新一致性
- BCC 指标设计简洁有效，可用于未来的模型评估
- 为理解 LLM 是否"隐式学习"了概率推理提供了新证据
- 对 AI 治理有重要意义

## 局限与展望
- 仅评估了预训练模型，未测试 RLHF/instruction-tuned 模型
- 置信度提取依赖 prompt 设计，可能引入系统偏差
- 数据集规模和领域覆盖可以进一步扩展

## 相关工作与启发
- 与 calibration 相关工作互补
- 启发：可以将贝叶斯一致性作为训练目标之一

## 评分
- 新颖性: ⭐⭐⭐⭐ 贝叶斯视角评估 LLM 信念更新是新颖的切入点
- 实验充分度: ⭐⭐⭐ 覆盖多个模型家族但整体规模有限（Workshop paper）
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，论述简洁
- 价值: ⭐⭐⭐⭐ 对 LLM 可信赖性评估有参考价值

---

## 补充思考

### 与领域发展趋势的关系
本文的研究方向与当前 AI 研究的几个大趋势密切相关：(1) 对 LLM 内部机制的深入理解需求日益增长；(2) 模型效率和可访问性的重要性不断提升；(3) AI 安全和可靠性成为核心关注点。从方法论角度看，本文代表了一种从"黑盒使用"到"白盒理解"的研究范式转变。

### 对未来研究的具体建议
1. 可以将本文的核心思路与其他模态（视觉、语音）结合
2. 考虑在更大规模的模型和数据上验证结论的普适性
3. 探索与强化学习和在线学习结合的可能性
4. 开发自动化的评估和优化工具链


---

## 补充思考

### 与领域发展趋势的关系
本文的研究方向与当前 AI 研究的几个大趋势密切相关：模型能力评估与可靠性保证、参数高效微调与模型压缩、以及 AI 安全与对齐。从方法论角度看，本文代表了对 LLM 深层机制的探索，有助于推动从经验驱动到理论驱动的研究范式转变。

### 对未来研究的具体建议
1. 可以将核心思路与其他模态（视觉、语音、多模态）结合，验证方法的跨模态通用性
2. 在更大规模模型（70B+）和更新的架构（Mixture-of-Experts 等）上验证结论
3. 探索与强化学习、在线学习结合的可能性，实现动态适应
4. 开发自动化评估和优化工具，降低方法的使用门槛
5. 考虑与 LLM alignment 研究的交叉，探索安全性和性能的协同优化

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Ineq-Comp: Benchmarking Human-Intuitive Compositional Reasoning in Automated Theorem Proving on Inequalities](../../NeurIPS2025/llm_evaluation/ineq-comp_benchmarking_human-intuitive_compositional_reasoning_in_automated_theo.md)
- [\[ICML 2025\] LLM-SRBench: A New Benchmark for Scientific Equation Discovery with LLMs](llm-srbench_a_new_benchmark_for_scientific_equation_discovery_with_large_languag.md)
- [\[ACL 2025\] RealHiTBench: A Comprehensive Realistic Hierarchical Table Benchmark for Evaluating LLM-Based Table Analysis](../../ACL2025/llm_evaluation/realhitbench_a_comprehensive_realistic_hierarchical_table_benchmark_for_evaluati.md)
- [\[ACL 2026\] 正确信念的瓦解：临床压力下 LLM 的认知韧性研究](../../ACL2026/llm_evaluation/when_correct_beliefs_collapse_epistemic_resilience_of_llms_under_clinical_pressu.md)
- [\[ACL 2025\] ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming](../../ACL2025/llm_evaluation/elaboration_competitive_programming.md)

</div>

<!-- RELATED:END -->
