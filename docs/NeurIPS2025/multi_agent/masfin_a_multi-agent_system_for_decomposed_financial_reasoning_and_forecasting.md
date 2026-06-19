---
title: >-
  [论文解读] MASFIN: A Multi-Agent System for Decomposed Financial Reasoning and Forecasting
description: >-
  [NeurIPS 2025][多智能体][多agent系统] 提出 MASFIN 多 agent 系统，将金融预测任务分解为多个子任务（宏观分析、行业分析、技术分析、情感分析等），由专门的 LLM agent 协作完成，实现比单一模型更准确和可解释的金融预测。 领域现状：金融预测需要同时考虑宏观经济、行业趋势、技术指标、市场…
tags:
  - "NeurIPS 2025"
  - "多智能体"
  - "多agent系统"
  - "金融推理"
  - "分解预测"
  - "LLM agent"
  - "时序分析"
---

# MASFIN: A Multi-Agent System for Decomposed Financial Reasoning and Forecasting

**会议**: NeurIPS 2025  
**arXiv**: [2512.21878](https://arxiv.org/abs/2512.21878)  
**代码**: 无  
**领域**: 时间序列 / 金融预测  
**关键词**: 多agent系统, 金融推理, 分解预测, LLM agent, 时序分析

## 一句话总结

提出 MASFIN 多 agent 系统，将金融预测任务分解为多个子任务（宏观分析、行业分析、技术分析、情感分析等），由专门的 LLM agent 协作完成，实现比单一模型更准确和可解释的金融预测。

## 研究背景与动机

**领域现状**：金融预测需要同时考虑宏观经济、行业趋势、技术指标、市场情感等多个维度，单一模型难以全面覆盖。

**现有痛点**：(1) 传统时序模型缺乏推理能力；(2) 单一 LLM 在多维度分析中容易遗漏关键因素；(3) 金融数据的多源异构性使单一模型难以处理。

**核心矛盾**：全面性 vs 专业性——一个模型要么广而浅，要么窄而深。

**切入角度**：多 agent 分工——每个 agent 专注一个分析维度，最终由协调 agent 综合。

## 方法详解

### 整体框架

Coordinator Agent 接收预测任务 → 分配给多个专家 Agent（Macro Agent、Industry Agent、Technical Agent、Sentiment Agent）→ 各 agent 独立分析 → Coordinator 综合推理 → 最终预测。

### 关键设计

1. **任务分解策略**

    - 功能：将金融预测分解为 4-5 个独立子任务
    - 核心思路：宏观分析（GDP、利率、通胀）、行业分析（竞争格局、供需）、技术分析（价格趋势、指标）、情感分析（新闻、社交媒体）
    - 设计动机：每个 agent 提供不同维度的信号，综合后更鲁棒

2. **综合推理机制**

    - 功能：Coordinator 综合各 agent 的分析报告
    - 核心思路：加权融合各维度信号，考虑一致性和冲突，生成最终预测+推理链
    - 设计动机：可解释性——用户可以追溯预测依据到具体分析维度

### 损失函数 / 训练策略

无需训练。各 agent 使用预训练 LLM（GPT-4 等）进行 in-context 推理。

## 实验关键数据

### 主实验

| 方法 | 方向准确率↑ | MSE↓ | 可解释性 |
|------|-----------|------|---------|
| ARIMA | 52.3% | 0.045 | 低 |
| LSTM | 56.7% | 0.038 | 低 |
| GPT-4 (单模型) | 58.2% | 0.035 | 中 |
| **MASFIN** | **63.5%** | **0.029** | **高** |

### 消融实验

| 配置 | 方向准确率 | 说明 |
|------|----------|------|
| 仅技术分析 | 55.1% | 价格趋势 |
| 仅宏观+技术 | 59.3% | 两个维度 |
| 仅情感+技术 | 58.8% | 情感补充 |
| **全部 agent** | **63.5%** | **最优** |

### 关键发现

- 多 agent 系统相比单一 GPT-4 预测准确率提升 5.3pp
- 每增加一个分析维度带来 1-3pp 提升，边际递减
- 可解释性显著增强——用户可以看到每个维度的分析报告

## 亮点与洞察

- **分治思想**：金融预测的多维度特性天然适合多 agent 分工，每个 agent 可以使用最适合的提示和工具。
- **可解释预测**：最终预测附带完整的推理链，用户可以理解和验证预测依据。

## 局限与展望

- API 调用成本高（多个 agent×多轮对话）
- 各 agent 之间缺乏迭代交互（目前是单轮分析→综合）
- 评估周期有限，长期预测效果未验证
- 缺乏与专业金融分析师的对比

## 相关工作与启发

- **vs FinGPT**：FinGPT 是单一微调模型，MASFIN 用多 agent 分工实现更全面的分析
- **vs MetaGPT**：MetaGPT 用于软件工程，MASFIN 将多 agent 范式引入金融

## 评分
- 新颖性: ⭐⭐⭐ 多 agent 框架应用于金融，但方法较直接
- 实验充分度: ⭐⭐⭐ 数据集有限，缺乏大规模回测
- 写作质量: ⭐⭐⭐⭐ 结构清晰
- 价值: ⭐⭐⭐⭐ 可解释金融预测有实际需求

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] DocAgent: A Multi-Agent System for Automated Code Documentation Generation](../../ACL2025/multi_agent/docagent_a_multi-agent_system_for_automated_code_documentation_generation.md)
- [\[AAAI 2026\] Thucy: An LLM-based Multi-Agent System for Claim Verification across Relational Databases](../../AAAI2026/multi_agent/thucy_an_llm-based_multi-agent_system_for_claim_verification_across_relational_d.md)
- [\[NeurIPS 2025\] Adaptive Coopetition: Leveraging Coarse Verifier Signals for Resilient Multi-Agent LLM Reasoning](adaptive_coopetition_leveraging_coarse_verifier_signals_for_resilient_multi-agen.md)
- [\[ICLR 2026\] MAC-AMP: A Closed-Loop Multi-Agent Collaboration System for Multi-Objective Antimicrobial Peptide Design](../../ICLR2026/multi_agent/mac-amp_a_closed-loop_multi-agent_collaboration_system_for_multi-objective_antim.md)
- [\[AAAI 2026\] LungNoduleAgent: A Collaborative Multi-Agent System for Precision Diagnosis of Lung Nodules](../../AAAI2026/multi_agent/lungnoduleagent_a_collaborative_multi-agent_system_for_precision_diagnosis_of_lu.md)

</div>

<!-- RELATED:END -->
