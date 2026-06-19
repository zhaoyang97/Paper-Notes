---
title: >-
  [论文解读] MedAgentBoard: Benchmarking Multi-Agent Collaboration with Conventional Methods for Diverse Medical Tasks
description: >-
  [NeurIPS 2025 (Datasets & Benchmarks Track)][多智能体][多智能体协作] 提出 MedAgentBoard，一个系统评估多智能体协作、单 LLM 和传统方法在多样化医学任务上表现的综合基准，揭示多智能体协作并不总是优于强单模型或专用传统方法。 - LLM 多智能体热潮：近期大量工作…
tags:
  - "NeurIPS 2025 (Datasets & Benchmarks Track)"
  - "多智能体"
  - "多智能体协作"
  - "医学基准测试"
  - "LLM"
  - "临床工作流"
  - "EHR预测"
---

# MedAgentBoard: Benchmarking Multi-Agent Collaboration with Conventional Methods for Diverse Medical Tasks

**会议**: NeurIPS 2025 (Datasets & Benchmarks Track)

**arXiv**: [2505.12371](https://arxiv.org/abs/2505.12371)

**代码**: [GitHub](https://github.com/yhzhu99/MedAgentBoard) | [项目主页](https://medagentboard.netlify.app/)

**领域**: Medical Imaging / AI for Medicine

**关键词**: 多智能体协作, 医学基准测试, LLM, 临床工作流, EHR预测

## 一句话总结

提出 MedAgentBoard，一个系统评估多智能体协作、单 LLM 和传统方法在多样化医学任务上表现的综合基准，揭示多智能体协作并不总是优于强单模型或专用传统方法。

## 研究背景与动机

- **LLM 多智能体热潮**：近期大量工作将多智能体协作引入医学领域，但其实际优势尚不明确
- **现有评估的不足**：
  1. 任务覆盖不够广泛，缺乏真实临床场景的多样性
  2. 缺少与传统专用方法的严格对比（多数工作只比 LLM 之间的差异）
  3. 数据模态单一，忽略了结构化 EHR 数据和医学影像
- **核心问题**：多智能体的额外复杂性和开销是否真正带来了性能增益？
- **研究定位**：提供全面的、基于证据的评估，帮助研究者选择合适的 AI 解决方案

## 方法详解

### 整体框架

MedAgentBoard 覆盖 **4 大类医学任务**，跨越 **3 种数据模态**（文本、医学影像、结构化 EHR），系统对比 **3 类方法**：

| 任务类别 | 数据模态 | 数据集 |
|---------|---------|--------|
| 医学问答（QA） | 文本 | MedQA, PubMedQA |
| 医学视觉问答（VQA） | 图像+文本 | PathVQA, VQA-RAD |
| 通俗摘要生成 | 文本 | PLOS/eLife |
| EHR 预测建模 | 结构化数据 | MIMIC-III/IV |
| 临床工作流自动化 | 多模态 | 定制场景 |

### 关键设计

#### 三类方法对比体系

1. **传统方法（Conventional）**：

    - 文本 QA：BioLinkBERT, GatorTron
    - VQA：M³AE 等专用 VLM
    - EHR：XGBoost, LSTM, Transformer 等

2. **单 LLM 方法**：

    - Zero-shot / Few-shot ICL / Chain-of-Thought
    - 使用 GPT-4o, Claude 3.5, Gemini 等

3. **多智能体协作框架**：

    - MedAgents：多角色讨论协作
    - ReConcile：多模型投票与调和
    - AutoGen 等通用框架

#### 评估维度

- **正确性**：Accuracy（选择题）、BLEU/ROUGE（生成任务）
- **临床相关性**：LLM-as-a-judge 评分
- **效率**：API 调用次数、token 消耗、延迟
- **鲁棒性**：跨数据集的一致性

### 损失函数 / 训练策略

作为 benchmark 论文，重点在评估协议设计而非模型训练：
- 所有 LLM 方法使用统一的 prompt 模板
- 传统方法遵循原始论文的最优配置
- 评估指标在各任务上标准化
- 多次运行取平均以减少随机性

## 实验关键数据

### 主实验

#### 医学文本 QA 结果

| 方法类别 | 方法名称 | MedQA Acc↑ | PubMedQA Acc↑ | 类别 |
|---------|---------|-----------|-------------|------|
| 传统 | BioLinkBERT | 45.2 | 72.8 | Conventional |
| 传统 | GatorTron | 48.1 | 74.5 | Conventional |
| 单 LLM | GPT-4o (Zero-shot) | 82.3 | 78.1 | Single LLM |
| 单 LLM | GPT-4o (CoT) | **85.7** | **80.4** | Single LLM |
| 单 LLM | Claude 3.5 (CoT) | 83.9 | 79.2 | Single LLM |
| 多智能体 | MedAgents | 83.1 | 78.8 | Multi-Agent |
| 多智能体 | ReConcile | 84.2 | 79.5 | Multi-Agent |

**发现**：在文本医学 QA 上，先进的单 LLM（GPT-4o + CoT）即可达到最优，多智能体未带来显著提升。

#### 医学 VQA 与 EHR 预测结果

| 方法类别 | PathVQA Acc↑ | VQA-RAD Acc↑ | MIMIC 死亡率 AUROC↑ |
|---------|-------------|-------------|-------------------|
| 传统 VLM (M³AE) | **72.3** | **74.8** | — |
| GPT-4o Vision | 65.7 | 68.2 | 0.71 |
| 多智能体 VQA | 64.9 | 67.5 | 0.69 |
| XGBoost | — | — | **0.84** |
| LSTM | — | — | 0.81 |
| LLM (数值推理) | — | — | 0.68 |

**发现**：专用传统方法在 VQA 和 EHR 预测上仍显著优于 LLM 方法。

### 消融实验

#### 多智能体 vs 单 LLM 效率对比

| 方法 | Accuracy | API 调用次数 | Token 消耗 | 延迟 (s) |
|------|----------|-------------|-----------|---------|
| GPT-4o (Single) | 85.7 | 1 | 2.1K | 3.2 |
| MedAgents (3 角色) | 83.1 | 5-8 | 12.5K | 18.7 |
| ReConcile (3 模型) | 84.2 | 3 | 7.8K | 11.4 |

**发现**：多智能体的 token 消耗是单 LLM 的 4-6 倍，但性能提升有限甚至为负。

### 关键发现

1. **多智能体 ≠ 更好**：在 4 类任务中，多智能体仅在临床工作流自动化的 **任务完整性** 上有优势
2. **传统方法仍然强劲**：专用微调模型在 VQA 和 EHR 预测上显著优于所有 LLM 方法
3. **单 LLM 的 CoT 足够强**：高质量单模型 + 好 prompt > 多个中等模型协作
4. **成本-收益不对称**：多智能体的计算开销增加了 4-6 倍，但平均性能提升 < 1%
5. **任务特异性**：不存在统一的最优方法，必须根据具体任务选择

## 亮点与洞察

- **冷静的 benchmark**：在多智能体热潮中提供了清醒的评估，指出"多智能体并非银弹"
- **公平对比设计**：将传统方法纳入对比是该 benchmark 的核心贡献，填补了现有评估的空白
- **多模态全覆盖**：同时覆盖文本、影像、结构化数据，反映真实临床的多样性
- **可操作的建议**：为从业者提供了"何时用多智能体、何时用单模型、何时用传统方法"的指南

## 局限与展望

1. **LLM 更新快**：benchmark 中的 LLM 结果可能很快过时（如 GPT-5 等新模型）
2. **多智能体框架有限**：仅评估了少数几种框架，新型协作范式（如 debate、reflection）可加入
3. **任务覆盖仍不全**：缺少医学图像分割、放射报告生成等重要临床任务
4. **传统方法的公平性**：传统方法经过精心微调，而 LLM 大多为 zero/few-shot，对比不完全公平
5. **真实部署评估**：缺少对实际临床部署场景（如延迟要求、隐私约束）的评估

## 相关工作与启发

- **MedAgents**（Tang et al., 2024）：医学多智能体讨论框架
- **AgentBench**（Liu et al., 2024）：通用 LLM agent 基准
- **HELM-Med**：医学 LLM 评估套件
- **启发**：该 benchmark 的思路（三方对比：传统 vs 单 LLM vs 多智能体）可推广到法律、金融等其他领域

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 创新性 | 3.5 | 贡献在于评估视角而非技术方法 |
| 技术深度 | 3 | Benchmark 工作，技术含量中等 |
| 实验充分性 | 4.5 | 覆盖面广，对比全面 |
| 实用价值 | 4.5 | 对医学 AI 选型有直接指导意义 |
| 写作质量 | 4 | 结构清晰，发现表述准确 |
| **总评** | **4.0** | 重要的 benchmark 贡献，有清醒的洞察 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Belief-Calibrated Multi-Agent Consensus Seeking for Complex NLP Tasks](belief-calibrated_multi-agent_consensus_seeking_for_complex_nlp_tasks.md)
- [\[ICLR 2026\] MMedAgent-RL: Optimizing Multi-Agent Collaboration for Multimodal Medical Reasoning](../../ICLR2026/multi_agent/mmedagent-rl_optimizing_multi-agent_collaboration_for_multimodal_medical_reasoni.md)
- [\[CVPR 2025\] ComfyBench: Benchmarking LLM-based Agents in ComfyUI for Autonomously Designing Collaborative AI Systems](../../CVPR2025/multi_agent/comfybench_benchmarking_llm-based_agents_in_comfyui_for_autonomously_designing_c.md)
- [\[NeurIPS 2025\] Thought Communication in Multiagent Collaboration](thought_communication_in_multiagent_collaboration.md)
- [\[NeurIPS 2025\] GauDP: Reinventing Multi-Agent Collaboration through Gaussian-Image Synergy in Diffusion Policies](gaudp_reinventing_multi-agent_collaboration_through_gaussian-image_synergy_in_di.md)

</div>

<!-- RELATED:END -->
