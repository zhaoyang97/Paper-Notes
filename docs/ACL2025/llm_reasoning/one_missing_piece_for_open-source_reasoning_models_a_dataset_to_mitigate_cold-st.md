---
title: >-
  [论文解读] One Missing Piece for Open-Source Reasoning Models: A Dataset to Mitigate Cold-Starting Short CoT LLMs in RL
description: >-
  [ACL 2025][LLM推理][长链式思维] 提出 Long CoT Collection——一个由短链式思维 LLM（如 GPT-4o）标注的 100K 长链式推理数据集，通过从 o1 提取推理流程作为引导，使短 CoT LLM 也能生成长 CoT 数据，从而解决强化学习中的冷启动问题，训练在该数据上初始化的模型在后续 RL 中获得 2-3 倍的性能提升。
tags:
  - "ACL 2025"
  - "LLM推理"
  - "长链式思维"
  - "冷启动"
  - "强化学习"
  - "推理模型"
  - "思维预算控制"
---

# One Missing Piece for Open-Source Reasoning Models: A Dataset to Mitigate Cold-Starting Short CoT LLMs in RL

**会议**: ACL 2025  
**arXiv**: [2506.02338](https://arxiv.org/abs/2506.02338)  
**代码**: 有（代码、数据集和模型公开发布）  
**领域**: LLM推理  
**关键词**: 长链式思维, 冷启动, 强化学习, 推理模型, 思维预算控制

## 一句话总结

提出 Long CoT Collection——一个由短链式思维 LLM（如 GPT-4o）标注的 100K 长链式推理数据集，通过从 o1 提取推理流程作为引导，使短 CoT LLM 也能生成长 CoT 数据，从而解决强化学习中的冷启动问题，训练在该数据上初始化的模型在后续 RL 中获得 2-3 倍的性能提升。

## 研究背景与动机

### 问题现状

大型推理模型（LRM）如 OpenAI o1 通过测试时缩放（生成长链式思维推理）在推理任务上取得突破性进展。DeepSeek-R1 公开了构建方法，关键发现是：

**冷启动问题**：直接在短 CoT LLM 上做强化学习（RLVR）效果不佳，学习信号太弱

**关键步骤**：需要先在长 CoT 数据上做 SFT，让模型学会推理结构

**现有依赖**：当前方法依赖收集 R1 等已有 LRM 的输出作为训练数据

### 核心问题

构建长 CoT 数据的方法尚不明确——现有工作要么依赖闭源 LRM（o1/R1）的蒸馏，要么数据集规模和质量不足。这构成了开源推理模型发展的关键瓶颈。

### 核心动机

**能否用短 CoT LLM 构建长 CoT 数据集？** 如果可以，就能摆脱对已有 LRM 的依赖，实现独立的 LRM 开发。同时，这种数据构建过程天然支持思维预算控制，可以应对 LRM 的"过度思考"问题（如 QwQ-32B 对 "1+1+3?" 也产出约 1500 tokens）。

## 方法详解

### 整体框架

数据构建流水线分为四步：
1. 收集 1K 种子数据集（o1 的推理流程 + 思维预算）
2. 对新问题检索相似的推理流程作为示范
3. 引导短 CoT LLM 生成新问题的推理流程
4. 沿推理流程逐步生成长 CoT 推理

### 关键设计

1. **种子数据集收集（1K demonstrations）**

    - 从 ChatGPT 网站手动收集 o1 对 1K 推理问题的**推理流程**（Reasoning Flow）
    - 推理流程定义为推理步骤大纲的序列 $S = \{s_1, s_2, ..., s_n\}$
    - 同时记录**思维预算** $b_{ref}$（thought token 数量 = 总 completion tokens - 返回响应 tokens）
    - 来源问题取自 magpie-reasoning-V1 数据集
    - 作用：捕获 LRM 的新型推理策略（验证、回溯、多路径探索等）

2. **推理流程检索（Reasoning Flow Retrieval）**

    - 对新问题动态检索种子数据集中的相似示例作为 ICL demonstration
    - 两个检索维度：
        - **领域匹配**：相同/相似领域的问题共享推理模式
        - **思维预算控制**：用相似度函数 $1 - |\frac{\min(x,y)}{\max(x,y)} - 1|$ 匹配相近长度的推理流程
    - 设计动机：确保检索到的 demonstration 在推理策略和深度上与目标问题匹配

3. **推理流程生成（Reasoning Flow Generation）**

    - 以检索到的 demonstration 为引导，让 GPT-4o 预测大纲步数 $|S|$ 并生成推理大纲序列
    - **关键发现**：无 demonstration 时，LLM 只能线性思考；有 demonstration 后能模仿验证、回溯等高级策略
    - 这是从 LRM 到短 CoT LLM 的"间接知识迁移"——仅通过推理流程模板传递

4. **逐步长 CoT 生成（Step-by-step Long CoT Generation）**

    - 对推理流程的每个步骤 $\hat{s}_i$，基于已有推理历史、当前步骤和下一步骤生成详细推理
    - 所有步骤完成后生成最终答案
    - 最后聚合为完整推理链
    - 设计动机：分步生成解决短 CoT LLM 直接生成长推理时的连贯性崩溃问题

5. **正确性过滤**

    - 用 GPT-4o 验证答案一致性
    - 过滤后保留 76% 的实例
    - 确保不在错误推理上训练

### 思维预算控制

通过调整推理大纲数量实现对推理长度的控制：
- 100% 预算：完整推理流程
- 50% 预算：压缩推理
- 25% 预算：最精简推理

这提供了应对"过度思考"问题的实用手段。

## 实验关键数据

### 数据质量评估（vs R1，o3-mini 评判）

在 100 个样本上的 head-to-head 比较：
- **推理流程**：Long CoT Collection **优于** R1
- **推理策略**：略低于 R1，但竞争力强
- **正确性**：略低于 R1，整体可比

### 主实验 — 通用推理能力

| 模型 | 参数 | GPQA Diamond | MMLU-Pro |
|------|------|-------------|----------|
| o1-mini | - | 60.0 | 80.3 |
| R1 | 671B | 71.5 | 84.0 |
| Bespoke-7B (R1蒸馏) | 7B | 38.9 | - |
| Llama-3.1-8B-Instruct | 8B | 22.7 | 43.7 |
| **Llama-3.1-8B-LC** | **8B** | **36.4** | **44.5** |
| Qwen-2.5-7B-Instruct | 7B | 37.6 | 49.9 |
| **Qwen-2.5-7B-LC** | **7B** | **39.9** | **51.4** |

GPQA 上 Llama-8B 提升 60%（22.7→36.4），Qwen-7B-LC 超越 Bespoke-7B（直接蒸馏 R1 的模型）。

### 核心实验 — RL 冷启动缓解（Figure 1）

| 设置 | MATH-500 RL增益 | GPQA RL增益 |
|------|----------------|------------|
| Qwen-0.5B → RLVR | 基准增益 | 基准增益 |
| **Qwen-0.5B-LC → RLVR** | **2-3x 增益** | **2-3x 增益** |

在 Long CoT Collection 上初始化后再做 RL，性能增益达到直接 RL 的 **2-3 倍**。

### 思维预算消融

| 预算比例 | MATH-500 |
|---------|----------|
| 100% | 66.6 |
| 50% | 60.7 |
| 25% | 57.6 |

### Best-of-N 采样分析

在 Llama-3.1-8B 上：
- LC 模型在 Pass@1 到 Pass@32 上持续优于基线
- Qwen-2.5-7B-LC 在大 N（16、32）时优势更明显
- 表明 SFT 后模型不仅准确率更高，且能探索更多样的推理路径

### 关键发现

1. **短 CoT LLM 可生成高质量长 CoT**：GPT-4o 在推理流程引导下产出与 R1 质量相当的推理数据
2. **冷启动缓解效果显著**：2-3x RL 性能增益是核心贡献
3. **推理策略有效迁移**：数据中包含丰富的推理触发词（如 "Wait"、"To verify"），促进多样推理路径
4. **思维预算可控**：通过调整大纲数量精确控制推理长度
5. **泛化到通用推理**：不仅数学，GPQA 和 MMLU-Pro 也有显著提升
6. **过度缩减预算有害**：25% 预算导致信息过度压缩，推理变得混乱

## 亮点与洞察

- **流水线而非数据本身是核心贡献**：揭示了用短 CoT LLM + 少量 LRM 引导构建长 CoT 的可行路径
- **推理流程作为中间表示**：将长 CoT 分解为"先规划后执行"，分层生成保证连贯性
- **Best-of-N 作为 RL 潜力预评估**：用 BoN 采样预判模型在 RL 中的上限，是实用的评估方法论
- **过度思考的解决思路**：预算控制不仅是数据构建技巧，也是部署时的实用工具
- **间接知识蒸馏**：只传递"怎么思考"（推理流程），而非"思考了什么"（具体推理内容）

## 局限与展望

1. **种子数据依赖 o1**：仍需 1K 条 o1 推理流程，未完全脱离 LRM 依赖
2. **RL 实验规模受限**：GPU 限制下仅在 0.5B 模型上验证 RL 效果，更大模型待验证
3. **GPT-4o 标注成本**：100K 数据的构建成本未详细披露
4. **领域局限**：主要验证数学和通用推理，代码、科学等专家领域有待探索
5. **仅考虑 o1 作为参考 LRM**：可扩展到其他部分公开推理的模型

## 相关工作与启发

- 与 DeepSeek-R1 的关系：R1 指出冷启动问题和长 CoT SFT 的重要性，本文提供了独立构建数据的方案
- 与 Sky-T1、Bespoke-7B 的比较：后者直接蒸馏 LRM 输出，本文用短 CoT LLM 间接生成
- 启发：推理流程的解耦思路可扩展到其他需要长序列推理的场景（如规划、代码生成）

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 新颖性 | ⭐⭐⭐⭐ | 用短CoT LLM生成长CoT的思路新颖且实用 |
| 实验充分度 | ⭐⭐⭐⭐ | BoN/RL/通用推理多角度验证，但RL规模受限 |
| 写作质量 | ⭐⭐⭐⭐ | 问题动机清晰，流水线描述详细 |
| 实用价值 | ⭐⭐⭐⭐⭐ | 解决开源推理模型的关键瓶颈 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Improve Vision Language Model Chain-of-thought Reasoning](improve_vlm_cot_reasoning.md)
- [\[ACL 2025\] Fine-Tuning on Diverse Reasoning Chains Drives Within-Inference CoT Refinement in LLMs](dcot_diverse_cot_refinement.md)
- [\[ACL 2025\] CoT-UQ: Improving Response-wise Uncertainty Quantification in LLMs with Chain-of-Thought](cot-uq_improving_response-wise_uncertainty_quantification_in_llms_with_chain-of-.md)
- [\[ACL 2025\] CoT-Valve: Length-Compressible Chain-of-Thought Tuning](cot-valve_length-compressible_chain-of-thought_tuning.md)
- [\[ACL 2025\] CoT-based Synthesizer: Enhancing LLM Performance through Answer Synthesis](cot-based_synthesizer_enhancing_llm_performance_through_answer_synthesis.md)

</div>

<!-- RELATED:END -->
