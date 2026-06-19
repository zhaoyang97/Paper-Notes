---
title: >-
  [论文解读] VisionArena: 230K Real World User-VLM Conversations with Preference Labels
description: >-
  [CVPR 2025][多模态VLM][视觉语言模型] VisionArena 构建了一个包含 230K 条真实用户与 VLM 交互记录的大规模数据集（含偏好标签），涵盖 73K 用户、45 个 VLM、138 种语言，揭示了当前 VLM 在空间推理和规划任务上的不足，并展示了用真实对话数据微调可显著超越 LLaVA-Instruct。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "视觉语言模型"
  - "人类偏好"
  - "基准评测"
  - "真实用户交互"
  - "Chatbot Arena"
---

# VisionArena: 230K Real World User-VLM Conversations with Preference Labels

**会议**: CVPR 2025  
**arXiv**: [2412.08687](https://arxiv.org/abs/2412.08687)  
**代码**: [https://huggingface.co/lmarena-ai](https://huggingface.co/lmarena-ai)  
**领域**: 推荐系统 / VLM评估  
**关键词**: 视觉语言模型, 人类偏好, 基准评测, 真实用户交互, Chatbot Arena

## 一句话总结
VisionArena 构建了一个包含 230K 条真实用户与 VLM 交互记录的大规模数据集（含偏好标签），涵盖 73K 用户、45 个 VLM、138 种语言，揭示了当前 VLM 在空间推理和规划任务上的不足，并展示了用真实对话数据微调可显著超越 LLaVA-Instruct。

## 研究背景与动机

**领域现状**：VLM 能力快速增长，但现有基准多为人工构造，无法反映用户真实使用场景和偏好。Chatbot Arena 已成功用于 LLM 评测，但视觉领域缺乏类似的大规模真实交互数据集。

**现有痛点**：(1) 人工基准与真实使用场景存在差距；(2) 缺乏大规模用户偏好数据来指导 VLM 训练；(3) 不清楚用户实际如何使用 VLM、在哪些任务上模型表现不佳。

**核心矛盾**：需要大规模、真实、多样化的用户-VLM 交互数据集，但此类数据收集成本高且涉及隐私问题。

**本文目标**：构建首个大规模真实用户-VLM 对话数据集，包含偏好标签，支持训练和评估。

**切入角度**：利用 Chatbot Arena 开源平台的真实用户交互记录，附加偏好投票功能。

**核心 idea**：从 Chatbot Arena 收集 230K 真实对话，划分为 Chat（200K 对话）、Battle（30K 偏好对比）和 Bench（500 条自动基准）三个子集。

## 方法详解

### 整体框架
数据集包含三个子集：(1) VisionArena-Chat：200K 单轮/多轮用户-VLM 对话；(2) VisionArena-Battle：30K 条用户同时与两个匿名 VLM 对话并投票选偏好的记录；(3) VisionArena-Bench：500 条自动基准提示，可高效近似 Chatbot Arena 在线排名。

### 关键设计

1. **VisionArena-Chat（200K 对话）**:

    - 功能：提供大规模真实 VLM 训练数据
    - 核心思路：从 Chatbot Arena 平台收集用户主动提交的对话记录，覆盖 138 种语言和 45 个 VLM。对话包含用户上传的图像和文本查询，以及 VLM 的响应
    - 设计动机：真实数据比人工指令数据更符合实际使用分布，用于微调可产生更好效果

2. **VisionArena-Battle（30K 偏好对比）**:

    - 功能：提供高质量偏好标签用于 RLHF 训练和模型排名
    - 核心思路：用户同时向两个匿名 VLM 发送相同查询，然后投票选择更好的响应（或选择平局）。这种 side-by-side 对比是获取可靠偏好信号的黄金标准
    - 设计动机：偏好数据可直接用于 RLHF 训练或构建奖励模型

3. **VisionArena-Bench（500 自动基准）**:

    - 功能：提供自动化评估工具
    - 核心思路：从 Battle 数据中精选 500 条多样化提示，使用强 VLM 作为评判者自动评分，能够高效近似在线 Arena 的 ELO 排名
    - 设计动机：在线 Arena 排名需要大量人力，自动基准可以快速评估新模型

### 损失函数 / 训练策略
用 VisionArena-Chat 进行标准的指令微调。实验显示，在相同基模型上，用 VisionArena-Chat 微调比用 LLaVA-Instruct-158K 微调在 MMMU 上提升 17 分，在 WildVision 上提升 46 分。

## 实验关键数据

### 主实验

| 训练数据 | MMMU | WildVision |
|---------|------|-----------|
| LLaVA-Instruct-158K | 基线 | 基线 |
| VisionArena-Chat | **+17** | **+46** |

### 关键发现

| 发现 | 详情 |
|------|------|
| 回复风格偏好 | 开放式任务（描述、幽默）高度依赖回复风格 |
| 模型弱点 | 当前 VLM 在空间推理和规划任务上表现差 |
| 数据质量 | 真实用户数据用于训练远优于人工指令数据 |

### 关键发现
- 开放式任务（如 captioning、幽默生成）的用户偏好高度依赖回复风格而非内容准确性
- 当前 VLM 在空间推理和规划任务上普遍表现不佳
- 真实用户查询的分布与现有基准差异显著，有大量非英语查询

## 亮点与洞察
- **首个大规模真实 VLM 交互数据集**：230K 对话、73K 用户、138 种语言的多样性远超现有数据集
- **训练数据质量的重要性**：仅更换训练数据（VisionArena-Chat 替代 LLaVA-Instruct）就带来了巨大提升，说明真实分布数据的价值
- **偏好信号的复杂性**：style vs. content 的偏好分离发现对 RLHF 训练策略有重要启示

## 局限与展望
- 数据来自 Chatbot Arena 用户，可能存在用户群体偏差（技术用户居多）
- 部分对话可能包含隐私敏感内容，数据清洗流程未详细描述
- VisionArena-Bench 的 500 条提示可能不够全面

## 相关工作与启发
- **vs WildVision**: 更早的小规模真实 VLM 评测数据集。VisionArena 在规模和多样性上大幅超越
- **vs Chatbot Arena (LLM)**: VisionArena 将 Arena 范式成功迁移到视觉领域
- 支撑了后续 VLM 训练和对齐研究的重要数据资源

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个大规模真实 VLM 对话数据集
- 实验充分度: ⭐⭐⭐⭐ 数据分析详细，训练实验验证了数据价值
- 写作质量: ⭐⭐⭐⭐ 数据集构建和分析描述清晰
- 价值: ⭐⭐⭐⭐⭐ 数据集对 VLM 社区有极高价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] STING-BEE: Towards Vision-Language Model for Real-World X-ray Baggage Security Inspection](sting-bee_towards_vision-language_model_for_real-world_x-ray_baggage_security_in.md)
- [\[ACL 2025\] REAL-MM-RAG: A Real-World Multi-Modal Retrieval Benchmark](../../ACL2025/multimodal_vlm/real-mm-rag_a_real-world_multi-modal_retrieval_benchmark.md)
- [\[ICCV 2025\] AdvDreamer Unveils: Are Vision-Language Models Truly Ready for Real-World 3D Variations?](../../ICCV2025/multimodal_vlm/advdreamer_unveils_are_visionlanguage_models_truly_ready_for.md)
- [\[NeurIPS 2025\] WearVQA: A Visual Question Answering Benchmark for Wearables in Egocentric Authentic Real-world scenarios](../../NeurIPS2025/multimodal_vlm/wearvqa_a_visual_question_answering_benchmark_for_wearables_in_egocentric_authen.md)
- [\[CVPR 2025\] Debiasing Multimodal Large Language Models via Noise-Aware Preference Optimization](debiasing_multimodal_large_language_models_via_noise-aware_preference_optimizati.md)

</div>

<!-- RELATED:END -->
