---
title: >-
  [论文解读] Enhancing Video-LLM Reasoning via Agent-of-Thoughts Distillation
description: >-
  [CVPR 2025][LLM推理][Agent蒸馏] AoTD 用 LLM agent 将复杂视频问题分解为子任务、调用专家视觉模型执行并收集中间结果作为推理链（CoT），经 LLM 质量过滤后蒸馏到 Video-LLM 中，让端到端模型同时获得准确答案和可解释的多步推理能力。 领域现状：视频问答（VideoQA）领域存在…
tags:
  - "CVPR 2025"
  - "LLM推理"
  - "Agent蒸馏"
  - "视频问答"
  - "推理链"
  - "时空定位"
  - "多步推理"
---

# Enhancing Video-LLM Reasoning via Agent-of-Thoughts Distillation

**会议**: CVPR 2025  
**arXiv**: [2412.01694](https://arxiv.org/abs/2412.01694)  
**代码**: [https://zhengrongz.github.io/AoTD/](https://zhengrongz.github.io/AoTD/)  
**领域**: Video Understanding / Agent  
**关键词**: Agent蒸馏、视频问答、推理链、时空定位、多步推理

## 一句话总结
AoTD 用 LLM agent 将复杂视频问题分解为子任务、调用专家视觉模型执行并收集中间结果作为推理链（CoT），经 LLM 质量过滤后蒸馏到 Video-LLM 中，让端到端模型同时获得准确答案和可解释的多步推理能力。

## 研究背景与动机

**领域现状**：视频问答（VideoQA）领域存在两条路线——(1) 端到端的 Video-LLM（如 VideoLLaMA2、LLaVA-NeXT-Video），在 QA pair 上直接训练，性能好但缺乏可解释性和时空定位能力；(2) 基于 agent 的系统，用 LLM 分解问题再调用专家模型，可解释性好但推理慢（47s vs 10s）、占内存大（65GB vs 18GB）、且受限于工具模型的能力。

**现有痛点**：端到端模型只学到"问题→答案"的映射，没有学到中间推理过程，遇到复杂的组合型、时序型、因果型问题容易出错且无法解释推理路径。而 agent 系统虽然有推理链，但实际部署不可行（太慢太重）。

**核心矛盾**：agent 系统的推理过程有价值（可解释、有时空定位），但其形式（多模型串行调用）不适合实际部署。能否把 agent 的推理能力"教"给端到端模型？

**本文目标** 如何自动为任何 VideoQA 数据集生成高质量的多步推理链（CoT），并将其蒸馏到 Video-LLM 中以增强推理能力？

**切入角度**：与直接用 MLLM 生成 CoT（可能产生幻觉）不同，作者用可靠的专家视觉模型作为"思考的代理"——每个子任务的输出都是实际的视觉分析结果（检测框、时间窗口等），比纯文本推理更可靠。

**核心 idea**：用 agent 系统的执行轨迹（而非 MLLM 想象）构造视频推理链，经 LLM 验证后蒸馏到端到端 Video-LLM。

## 方法详解

### 整体框架
AoTD 分四步：(1) 评估并选择各子任务最佳视觉模型；(2) 用 LLM 将视频问题分解为 Python 程序，调用专家模型顺序执行并记录中间结果；(3) 用 LLM 将执行轨迹转为自然语言 CoT 并做两步质量验证；(4) 将验证过的 CoT 与 QA pair 一起蒸馏到 Video-LLM。最终模型可根据 prompt 选择输出简洁答案或详细推理链。

### 关键设计

1. **子任务专家模型选择与 agent 执行**:

    - 功能：将复杂视频问题自动分解为可执行的子任务链，并用最优模型依次解决
    - 核心思路：定义 5 类子任务——问题分解（DeepSeek-Coder 85.7% Acc）、目标检测（OWL-ViT v2 63.0% IoU）、时序定位（UniVTG 24.7% IoU）、动作识别（LLaVA-NeXT-Video-DPO 18.2% Top1）、问答（LLaVA-NeXT-Video-DPO 53.4% Acc）。用 STAR 数据集的带标注 program 做评估，为每个子任务选最佳模型。执行时 LLM 读取模型文档，将问题分解为 Python 代码调用相应模型
    - 设计动机：不依赖单一模型的"想象"构造 CoT，而是用专家模型的实际输出（检测框、时间段等）作为推理依据，更可靠。每个子任务独立评估也暴露了当前视觉模型的能力边界（如时序定位只有 24.7% IoU）

2. **两步 CoT 质量验证**:

    - 功能：过滤掉错误或低质量的推理链，确保蒸馏数据可靠
    - 核心思路：第一步——执行结果过滤：多选题要求 agent 输出与正确答案完全匹配，开放题用 LLM 验证一致性；第二步——逻辑质量过滤：用 LLM 评估 CoT 是否遵循清晰的逐步推理过程、是否包含解答所需的关键信息（二分类判断 Yes/No）。从 158.6K QA pair 中最终保留 32.3K 高质量 CoT（约 20% 通过率）
    - 设计动机：不过滤直接蒸馏会导致性能下降（消融实验证明：过滤后 MVBench 55.6% vs 不过滤 53.7%），说明低质量 CoT 会误导模型学习

3. **双模式蒸馏训练**:

    - 功能：让模型同时支持直接回答和生成推理链两种输出模式
    - 核心思路：训练时将有 CoT 的样本用"Explain the rationale"作后缀 prompt，无 CoT 的样本用标准 QA prompt。损失函数 $\mathcal{L} = \mathcal{L}_{label} + \lambda \mathcal{L}_{rationale}$，其中 $\lambda=1$。推理时根据不同 prompt 选择输出模式——需要快速回答就直接输出答案，需要解释就生成完整推理链
    - 设计动机：推理链训练不仅提升可解释性，还反哺了直接回答的准确率（因为模型内化了推理过程），同时保持了部署灵活性

### 损失函数 / 训练策略
标准交叉熵 loss，answer loss 和 rationale loss 等权（λ=1）。无 CoT 的样本 rationale loss 设为 0。基于 LLaVA-NeXT-Video-7B 做指令微调。训练数据包括 STAR、NExT-QA、AGQA、ANetQA、CLEVRER 等 VideoQA 数据集。

## 实验关键数据

### 主实验

| 基准 | 指标 | LNV-AoTD | LNV-Instruct | 提升 |
|--------|------|------|----------|------|
| STAR (组合型) | Acc | **74.3%** | 72.2% | +2.1% |
| NExT-QA (因果型) | Acc | **81.2%** | 79.7% | +1.5% |
| Perception-Test | Acc | **58.8%** | 57.1% | +1.7% |
| MVBench | Acc | **55.6%** | 53.1% | +2.5% |
| AGQA (开放) | Acc/Score | 60.9/3.6 | 59.3/3.4 | +1.6/+0.2 |
| ActivityNet-QA | Score | **3.55** | 3.52 | +0.03 |

### 消融实验

| 配置 | MVBench | STAR | AGQA |
|------|---------|------|------|
| LNV-AoTD (w/ filtering) | **55.6** | **74.3** | 60.9/3.6 |
| LNV-AoTD (w/o filtering) | 53.7 | 73.3 | 59.5/3.5 |
| LLaVA-OneVision + AoTD | **60.5** | **76.6** | 65.7/3.7 |
| LLaVA-OneVision Instruct | 59.2 | 75.8 | 65.6/3.7 |
| Qwen2-VL + AoTD | **66.5** | **73.1** | 61.2/3.7 |
| Qwen2-VL Instruct | 65.6 | 71.4 | 59.8/3.6 |

### 关键发现
- **CoT 过滤至关重要**：不过滤的 CoT 蒸馏反而可能带来噪声（MVBench 降 1.9%），说明 agent 系统产生的推理链约 80% 质量不达标
- **方法可迁移到不同 Video-LLM**：在 LLaVA-OneVision、VideoLLaMA2、Qwen2-VL 上一致提升，验证了 AoTD 的通用性
- **蒸馏后模型真的学到了时空推理**：在 STAR 上评估 rationale 中的时序定位（IoU 21.7% vs UniVTG 22.8%）和空间定位（IoU 45.2% vs OWL-ViT 64.7%），端到端模型的定位能力接近专家模型
- **效率提升显著**：agent 系统 47.9s/65GB → 蒸馏后 10.6s/18GB，推理延迟降低 4.5 倍，内存减少 3.6 倍

## 亮点与洞察
- **"用 agent 的执行轨迹做 CoT"比"让 MLLM 自己编 CoT"更可靠**：因为中间结果是视觉模型的实际输出，不是凭空想象。这种"以工具执行结果为依据"的 CoT 构造范式可以推广到其他多步推理任务
- **蒸馏方法的优雅性**：不需要改动 Video-LLM 的架构，只是在训练数据层面加入了推理链，就能同时提升准确率和可解释性。成本极低，收益明确
- **子任务评估暴露了视觉模型的短板**：时序定位（24.7% IoU）和动作识别（18.2% Top1-Acc）仍然很弱，是 agent 系统的瓶颈。随着这些基础模型的进步，AoTD 的效果还有很大提升空间

## 局限与展望
- 时序定位和动作识别模型还太弱（IoU 仅 24.7%），CoT 中的时序信息可能不够准确，限制了蒸馏效果的天花板
- CoT 通过率仅约 20%（158.6K→32.3K），大量 QA pair 没有 CoT 辅助训练
- 蒸馏后模型的空间定位能力（IoU 45.2%）远弱于专家模型（64.7%），说明端到端蒸馏仍有信息损失
- 开放式 VQA 的评估（GPT 评分）存在偏差，难以准确反映模型真实能力
- 没有探索迭代蒸馏——用蒸馏后的模型生成更好的 CoT 再训练

## 相关工作与启发
- **vs VPD (Visual Program Distillation)**: VPD 在图像上做类似的事；AoTD 是首次将 agent 蒸馏扩展到视频领域，需要额外处理时序定位和动作识别
- **vs Video-STaR**: Video-STaR 用视频和已有标签构造 CoT，不需要 agent 系统；AoTD 的优势是 CoT 基于视觉模型的实际观测，更有根据
- **vs MoReVQA/VURF**: 这些是 agent 系统本身，推理慢且重；AoTD 把 agent 的能力蒸馏到轻量模型，实际部署更可行

## 评分
- 新颖性: ⭐⭐⭐⭐ 将 agent 执行轨迹蒸馏为 CoT 是一个自然但有效的想法，视频领域首次
- 实验充分度: ⭐⭐⭐⭐ 多个 VideoQA 基准、多模型迁移测试、CoT 质量评估、效率对比都有
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，子任务评估/过滤机制描述详细
- 价值: ⭐⭐⭐⭐ 方法通用、成本低、效果一致，对 Video-LLM 社区有指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Video-T1: Test-Time Scaling for Video Generation](../../ICCV2025/llm_reasoning/video-t1_test-time_scaling_for_video_generation.md)
- [\[NeurIPS 2025\] Atom of Thoughts for Markov LLM Test-Time Scaling](../../NeurIPS2025/llm_reasoning/atom_of_thoughts_for_markov_llm_testtime_scaling.md)
- [\[ACL 2025\] Unifying Language Agent Algorithms with Graph-based Orchestration Engine for Reproducible Agent Research](../../ACL2025/llm_reasoning/unifying_language_agent_algorithms_with_graph-based_orchestration_engine_for_rep.md)
- [\[ICML 2025\] ProofCompass: Enhancing Specialized Provers with LLM Guidance](../../ICML2025/llm_reasoning/proofcompass_enhancing_specialized_provers_with_llm_guidance.md)
- [\[CVPR 2025\] VideoEspresso: A Large-Scale Chain-of-Thought Dataset for Fine-Grained Video Reasoning via Core Frame Selection](videoespresso_a_large-scale_chain-of-thought_dataset_for_fine-grained_video_reas.md)

</div>

<!-- RELATED:END -->
