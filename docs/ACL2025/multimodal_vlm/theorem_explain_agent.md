---
title: >-
  [论文解读] TheoremExplainAgent: Towards Video-based Multimodal Explanations for LLM Theorem Understanding
description: >-
  [ACL 2025][多模态VLM][theorem explanation] 提出 TheoremExplainAgent，一个双 Agent 系统（Planner + Coder），通过 Manim 动画脚本自动生成长达 10 分钟的定理讲解视频，配套 TheoremExplainBench（240 个 STEM 定理 × 5 维评估指标），证明 agentic planning 是长视频生成的关键，且视觉解释能暴露文本评估无法发现的推理缺陷。
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "theorem explanation"
  - "video generation"
  - "Manim animation"
  - "LLM agent"
  - "STEM education"
---

# TheoremExplainAgent: Towards Video-based Multimodal Explanations for LLM Theorem Understanding

**会议**: ACL 2025  
**arXiv**: [2502.19400](https://arxiv.org/abs/2502.19400)  
**代码**: [GitHub](https://tiger-ai-lab.github.io/TheoremExplainAgent/)  
**领域**: Multimodal VLM  
**关键词**: theorem explanation, video generation, Manim animation, LLM agent, STEM education

## 一句话总结

提出 TheoremExplainAgent，一个双 Agent 系统（Planner + Coder），通过 Manim 动画脚本自动生成长达 10 分钟的定理讲解视频，配套 TheoremExplainBench（240 个 STEM 定理 × 5 维评估指标），证明 agentic planning 是长视频生成的关键，且视觉解释能暴露文本评估无法发现的推理缺陷。

## 研究背景与动机

**领域现状**：理解领域特定定理通常不仅需要文本推理，还需要结构化的视觉解释来加深理解。LLM 在文本推理、定理证明等任务上已表现出色，现有基准如 TheoremQA、GSM8K 主要通过选择题或短答题评估定理理解能力。

**现有痛点**：❶ 评估形式单一——选择题容易被表面线索（如选项顺序）利用，无法真正衡量概念理解深度。❷ 缺少视觉维度——定理推理本质上是多模态的，几何、拓扑、代数等领域的理解高度依赖视觉表示，但现有评估完全是纯文本的。❸ AI 生成多模态解释的能力是开放性挑战——尽管 LLM 文本能力强大，能否生成连贯、有教学意义的视觉解释尚未探索。

**核心问题**：AI 系统能否有效生成多模态定理解释？更关键的是，视觉化生成过程能否暴露出文本评估所掩盖的更深层推理缺陷？

**切入角度**：将定理理解评估从"选择题/短答题"提升到"生成视频讲解"，通过代码驱动的 Manim 动画生成长形式视频，同时构建标准化评估框架。

## 方法详解

### 整体框架

TheoremExplainAgent（TEA）采用双 Agent 流水线：

- **输入**：定理名称 + 简短描述
- **Planner Agent**：生成高层视频计划（story plan）→ 划分多个场景 → 细化每个场景的视觉元素、动画和过渡效果 → 生成旁白文字
- **Coding Agent**：将场景规格转换为 Manim Python 脚本 → 代码执行 → 错误修复循环（最多 N=5 次重试）→ TTS 生成语音旁白
- **输出**：包含动画 + 结构化推导 + 语音旁白的定理讲解视频（>1 分钟，最长可达 10 分钟）

### 关键设计

1. **Manim 代码驱动的视频生成**:

    - 功能：通过生成可执行的 Python 脚本来创建数学动画，而非直接生成像素级视频
    - 核心思路：Manim 是 3Blue1Brown 使用的开源数学动画库，代码驱动的方式天然适合 LLM 生成——LLM 擅长代码生成但不擅长像素控制
    - 设计动机：对比实验中，纯文本到视频模型（LTXVideo、Veo2）生成的内容完全不可用（视觉上不连贯、与定理无关），证明代码驱动路线的必要性

2. **Agentic 错误修复循环（N=5 重试）**:

    - 功能：Coding Agent 执行代码后如遇错误，自动审查错误信息并生成修正版本代码
    - 核心思路：代码生成本身容易出错（Manim API 幻觉、LaTeX 渲染错误、Python 通用错误），但通过多次重试可大幅提升成功率
    - 设计动机：N=0 时成功率仅 3-7%，N=5 时 o3-mini 达到 91-96%，证明重试机制至关重要

3. **Agentic RAG（检索增强生成）**:

    - 功能：以 Manim 文档为知识库，在三个阶段动态检索
    - 核心思路：❶ 故事板生成阶段检索视觉示例和相关概念 → ❷ 技术实现阶段检索代码片段和用法模式 → ❸ 错误修正阶段检索诊断信息和解决方案
    - 设计动机：理论上应帮助代码生成，但实验发现 RAG 对强模型（o3-mini）反而有害（93.8% → 82.1%），因检索结果常不匹配具体场景，引入噪声

### 损失函数 / 训练策略

本文不涉及模型训练。TheoremExplainBench 的评估体系包含 5 个维度：
- **准确度与深度**、**逻辑流程**：基于 GPT-4o 对 SRT 字幕的文本评估
- **视觉相关性**、**元素布局**：关键帧提取 + GPT-4o 图像评估
- **视觉一致性**：Gemini 2.0-Flash 分析视频片段
- 综合分数 = 各维度的几何平均值（0-1 范围），使用 greedy decoding（temperature=0）确保输出稳定

## 实验关键数据

### 主实验

**视频生成成功率**（4 个 Agent × 3 个难度 × 4 个学科）：

| Agent | Easy | Medium | Hard | Math | Phys | CS | Chem | Overall |
|-------|------|--------|------|------|------|----|------|---------|
| o3-mini | **93.8%** | **91.2%** | **96.2%** | 95.0% | 93.3% | 93.3% | 93.3% | **93.8%** |
| GPT-4o | 61.3% | 57.5% | 46.2% | 61.7% | 55.0% | 58.3% | 45.0% | 55.0% |
| Gemini 2.0-Flash | 20.0% | 11.2% | 12.5% | 16.7% | 8.3% | 21.7% | 11.7% | 14.6% |
| Claude 3.5-Sonnet v1 | 2.5% | 1.2% | 2.5% | 1.7% | 1.7% | 1.7% | 3.3% | 2.1% |

**视频质量评分**（满分 1.0，仅统计成功生成的视频）：

| Agent | 准确度 | 视觉相关性 | 逻辑流程 | 元素布局 | 视觉一致性 | 综合 |
|-------|--------|-----------|---------|---------|-----------|------|
| GPT-4o | **0.79** | 0.79 | 0.89 | 0.59 | 0.87 | 0.78 |
| o3-mini | 0.76 | 0.76 | **0.89** | **0.61** | **0.88** | **0.77** |
| 人工 Manim 视频 | 0.80 | **0.81** | 0.70 | **0.73** | 0.87 | 0.77 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| N=0（无重试） | 成功率 3-7% | 单次代码生成几乎不可能成功 |
| N=1 | 成功率 33-51% | 第一次重试带来最大提升 |
| N=5 | 成功率 91-96% (o3-mini) | 收敛点，进一步增加收益递减 |
| o3-mini + RAG | 成功率 82.1%（降低 11.7%） | RAG 对强模型反而有害 |
| GPT-4o + RAG | 成功率 45.8%（降低 9.2%） | RAG 普遍降低成功率 |
| Agentless 方法 | 视频 ≤20 秒 | 无法生成长视频，证明 agentic planning 必要性 |
| Text-to-Video 模型 | 视觉不连贯，无关内容 | LTXVideo/Veo2 完全不可用 |

### 关键发现

- **o3-mini 在成功率上碾压其他模型**：93.8% vs GPT-4o 的 55.0%，强推理模型在代码驱动视觉生成上有绝对优势
- **Claude 3.5-Sonnet 几乎完全失败**：仅 2.1% 成功率，暴露其在 Manim 代码生成上的严重不足
- **RAG 反而有害**：对 o3-mini 成功率从 93.8% 降至 82.1%，检索到的文档常不匹配具体场景引入噪声
- **元素布局是所有模型的短板**：最高仅 0.61（o3-mini），人工视频 0.73，空间推理能力仍是瓶颈
- **视频解释暴露更深层推理缺陷**：15 名参与者先看文本解释时全部判为正确，看到视频后 60% 改判为错误——视觉化迫使 AI 显式编码结构知识，错误更容易被发现
- **人工视频逻辑流程反而更低（0.70 vs 0.89）**：人工视频更追求直觉性和互动性，AI 视频严格遵循逻辑结构
- **化学领域最难**：复杂对象（烧瓶、分子）比数学中的简单几何图元更难以代码方式可视化

## 亮点与洞察

- **任务定义是核心贡献**：将定理理解评估从"选择题回答"提升到"生成讲解视频"，完全不同的评估维度，更接近"真正理解"
- **"生成即理解"的评估范式**：如果 AI 能生成正确的动画讲解，说明它确实理解了定理的结构和过程
- **多模态解释作为推理缺陷探测器**：视觉化能暴露文本中隐藏的错误——这一发现对 AI 评估和教育应用都有深远意义
- **Agentic 方法的必要性**：agentless 只能生成 ≤20 秒视频，agent 可达 10 分钟，规划能力是长内容生成的基石

## 局限与展望

- 视觉布局质量仍不理想，文本重叠、形状错位、大小不一致问题频繁出现
- 依赖 Manim 库的能力边界：某些复杂可视化（3D 交互、化学分子结构）受限于 Manim 的表达能力
- 评估指标与人类判断的对齐度有限：准确度与深度的 Spearman ρ=0.14，逻辑流程 ρ=0.16，只有视觉相关性 ρ=0.72 较好
- 仅测试英语，STEM 教育有强地域性，多语言适用性未探索
- 计算成本高：每个定理需要多次 LLM 调用+代码执行+TTS，约 1500 美元 API 费用
- 缺乏用户学习效果研究：视频是否真正帮助学生理解定理，未做受控实验

## 相关工作与启发

- **vs TheoremQA**: TheoremQA 通过选择题/短答题评估，容易被表面线索利用；本文要求生成完整视频，更严格
- **vs MatPlotAgent/PlotGen**: 前人 AI 可视化研究聚焦数据图表生成；本文拓展到数学动画和教育视频，复杂度更高
- **vs 3Blue1Brown/Manim**: 3Blue1Brown 手动编写 Manim 脚本制作高质量视频；本文探索 AI 自动化这一过程的可行性和边界
- **vs Text-to-Video 模型**: LTXVideo/Veo2 等生成模型缺乏推理能力，无法生成结构化教育内容，证明了 LLM agent 路线的优势

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 任务定义本身极具创新性，首次将定理理解评估提升到视频生成维度
- 实验充分度: ⭐⭐⭐⭐ 4 个 Agent、240 个定理、重试消融、RAG 对比、人类研究，较为全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，案例分析和人类研究增强了说服力
- 价值: ⭐⭐⭐⭐ 对 AI 教育、多模态评估和 agent 设计都有启发，但实用性受限于当前视觉质量

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Breaking Multimodal LLM Safety via Video-Driven Prompting](../../CVPR2026/multimodal_vlm/breaking_multimodal_llm_safety_via_video-driven_prompting.md)
- [\[NeurIPS 2025\] Watch and Listen: Understanding Audio-Visual-Speech Moments with Multimodal LLM](../../NeurIPS2025/multimodal_vlm/watch_and_listen_understanding_audio-visual-speech_moments_with_multimodal_llm.md)
- [\[ACL 2025\] GODBench: A Benchmark for Multimodal Large Language Models in Video Comment Art](godbench_a_benchmark_for_multimodal_large_language_models_in_video_comment_art.md)
- [\[ACL 2026\] ViLL-E: Video LLM Embeddings for Retrieval](../../ACL2026/multimodal_vlm/vill-e_video_llm_embeddings_for_retrieval.md)
- [\[ACL 2025\] Evaluating Multimodal Large Language Models on Video Captioning via Monte Carlo Tree Search](mcts_video_captioning_eval.md)

</div>

<!-- RELATED:END -->
