---
title: >-
  [论文解读] CoRVid: Improving Multimodal Large Language Models Towards Chain-of-Thought Reasoning
description: >-
  [ICCV 2025][LLM推理][多模态大语言模型] 提出 Corvid，通过混合视觉编码器 + GateMixer 连接器 + 高质量 CoT 数据集 + 推理时自验证策略，全面提升 MLLM 的链式推理能力，在数学推理和科学问题求解上超越同参数量级的开源模型。 多模态大语言模型（MLLMs）在感知和理解方面已展现出色…
tags:
  - "ICCV 2025"
  - "LLM推理"
  - "多模态大语言模型"
  - "链式推理"
  - "视觉编码器"
  - "推理时扩展"
  - "CoT"
---

# CoRVid: Improving Multimodal Large Language Models Towards Chain-of-Thought Reasoning

**会议**: ICCV 2025  
**arXiv**: [2507.07424](https://arxiv.org/abs/2507.07424)  
**代码**: [项目主页](https://mm-vl.github.io/corvid)  
**领域**: LLM推理  
**关键词**: 多模态大语言模型, 链式推理, 视觉编码器, 推理时扩展, CoT

## 一句话总结

提出 Corvid，通过混合视觉编码器 + GateMixer 连接器 + 高质量 CoT 数据集 + 推理时自验证策略，全面提升 MLLM 的链式推理能力，在数学推理和科学问题求解上超越同参数量级的开源模型。

## 研究背景与动机

多模态大语言模型（MLLMs）在感知和理解方面已展现出色能力，但在需要**复杂结构化推理**的任务上（如数学推理、科学问题求解）仍表现不佳。以 Ovis2 和 Qwen2.5-VL 为代表的领先模型在需要深度思考和外推的任务上表现次优。

**三大挑战**：

**高质量多模态 CoT 数据稀缺**：人工创建的 CoT 通常简短，AI 生成的 CoT 噪声大且有重复。研究表明用直接回答训练的 MLLM 无法进行逐步推理。

**视觉表征不足和跨模态对齐不佳**：复杂推理需要准确捕获视觉信息并高效转换到语言嵌入空间。现有连接器（简单 MLP 投影或跨注意力层）在构建推理所需的充分视觉表征方面不够。

**推理时过度推理与不足推理**：当前 o1-like MLLM 对所有样本统一执行深度推理，但简单任务上 CoT 推理反而可能因上下文丢失和幻觉而降低准确率。

本文的核心思路是**多管齐下**：架构上用混合编码器+门控混合连接器加强视觉表征，数据上精炼 287K 高质量 CoT 指令跟随数据，推理上用自验证策略自适应选择直接回答或 CoT 推理。

## 方法详解

### 整体框架

Corvid 包含三个核心模块：混合视觉编码器（SigLIP + ConvNeXt-XXL）、GateMixer 连接器、Llama3-8B LLM。训练分三阶段：多粒度对齐预训练 → CoT 增强微调 → 纯 CoT 指令调优。

### 关键设计

1. **混合视觉编码器**：

    - SigLIP ViT-SO400M：384×384 输入，输出 729×1152 的语义丰富特征
    - OpenCLIP ConvNeXt-XXL：同分辨率输入，输出 729×5760 的多尺度空间细节特征
    - 两个编码器互补：ViT 擅长语义，CNN 保留空间细节

2. **GateMixer 连接器**：

    - 分别通过线性层将两组特征映射到统一空间 $\{\mathbf{h}_v, \mathbf{h}_c\} \in \mathbb{R}^{729 \times d}$
    - 借鉴 LSTM 输入门机制，用门控注意力逐元素混合：
        - $\boldsymbol{\alpha} = \sigma(\mathbf{W}_g[\mathbf{h}_v; \mathbf{h}_c] + \mathbf{b}_g)$
        - $\mathbf{h} = (1 - \boldsymbol{\alpha}) \odot \mathbf{h}_v + \boldsymbol{\alpha} \odot \mathbf{h}_c$
    - 插入可学习前缀 token $\mathbf{h}_p \in \mathbb{R}^{24 \times d}$ 增强上下文捕获
    - 最终线性投影到语言嵌入空间

3. **MCoT-Instruct-287K 数据集**：

    - 从多个公开推理数据集中精炼和标准化，覆盖数学、科学、逻辑等多种推理类型
    - 人工创建的 CoT 虽然准确但简短 → 用 GPT 辅助扩展和标准化
    - AI 生成的 CoT 虽然详细但含错误 → 用 GPT 辅助纠错和去重
    - 训练数据包括 MGA-1M（对齐预训练）、Corvid-1M（SFT）、o1-320K（纯 CoT）

4. **推理时自验证策略**：

    - 同时生成直接回答 $\mathcal{R}_{\text{direct}}$ 和 CoT 回答 $\mathcal{R}_{\text{CoT}}$
    - 如果两者答案一致，直接采用
    - 如果不一致，计算加权分数 $\mathcal{SC} = (1-\alpha)\mathcal{S} + \alpha\mathcal{C}$
        - $\mathcal{S}$：LLM 后图像-文本表征的余弦相似度（跨模态对齐质量）
        - $\mathcal{C}$：归一化困惑度（模型信心）
    - 选择 $\mathcal{SC}$ 更高的答案

### 损失函数 / 训练策略

- **Stage 1**（多粒度对齐预训练）：冻结编码器+LLM，仅训练 GateMixer，在 MGA-1M 上训练。额外使用对比正则化 $\mathcal{L}_{\text{CReg}}$ 促进图文语义关联。
- **Stage 2**（CoT 增强 SFT）：联合训练 GateMixer + LLM，在 Corvid-1M 上训练，得到 Corvid-base。
- **Stage 3**（纯 CoT 调优）：在 o1-320K 上微调 Corvid-base，得到 Corvid-o1。

## 实验关键数据

### 主实验

与同参数量级 MLLM 对比（数学推理基准）：

| 模型 | MathVista | MathVerse | WeMath | MathVision | DynaMath |
|------|-----------|-----------|--------|------------|---------|
| Qwen2.5-VL-7B | 65.8 | 31.5 | 36.4 | 25.8 | 14.8 |
| Ovis2-8B | 71.8 | 42.3 | 27.2 | 25.9 | 20.4 |
| InternVL2.5-8B | 64.5 | 22.8 | 23.5 | 17.0 | 9.4 |
| **Corvid-o1-8B** | **72.0** | **40.1** | **59.8** | **30.1** | **33.2** |

与 o1-like MLLM 对比：

| 模型 | MMStar | MMB | MathV | AI2D | Avg. |
|------|--------|-----|-------|------|------|
| LLaVA-o1 | 58.1 | 75.6 | 56.1 | 78.8 | 63.1 |
| LlamaV-o1 | 59.5 | 79.9 | 54.4 | 81.2 | 67.3 |
| Mulberry-o1-7B | 61.3 | 75.3 | 57.5 | 79.0 | 62.8 |
| **Corvid-o1-8B** | **65.2** | **82.9** | **72.0** | **85.0** | **67.5** |

### 消融实验

自验证策略的有效性（Corvid-o1）：

| 推理方式 | 平均 | MMStar | MMMU | MathVista | WeMath |
|---------|------|--------|------|-----------|--------|
| 直接推理 | 48.9 | 54.5 | 47.6 | 49.2 | 41.8 |
| CoT 推理 | 56.2 | 61.1 | 55.7 | 67.2 | 57.1 |
| **自验证** | **59.9** | **65.2** | **59.7** | **72.0** | **59.8** |

连接器消融（平均精度）：FC_GELU_FC: 54.0, 2×FC_GELU_FC: 53.7, GateMixer: **55.6**。

### 关键发现

- 高质量 CoT 数据至关重要：直接回答数据 (45.7%) vs 原始理由 (51.2%) vs 精炼 CoT (55.6%)
- 数据多样性同样关键：仅 CoT 推理数据时平均 48.0，加入直接推理和 OCR 等数据后升至 55.6
- 自验证策略比纯 CoT 推理提升 3.7 分（Corvid-o1），比直接推理提升 11.0 分
- 混合编码器 (55.6) > 单 SigLIP (54.7) > 单 ConvNeXt (52.3)

## 亮点与洞察

- 自验证策略解决了"何时该深度推理"的问题，无需外部验证器
- GateMixer 的门控机制保持了视觉 token 长度不变，同时实现选择性表征融合
- CoT 数据的标准化和精炼流程比简单的数据量扩展更重要
- WeMath 上 59.8 分远超其他模型（Ovis2-8B 仅 27.2），展示了 CoT 训练在复杂数学推理上的巨大潜力

## 局限与展望

- 推理时需要生成两次响应（直接+CoT），推理成本翻倍
- 缺乏世界知识和常识推理能力（如典型车道宽度等）
- 384×384 的输入分辨率限制了高分辨率图像的细节捕获
- 未探索强化学习（如 GRPO）的推理能力增强
- 最大生成长度仅 1024 tokens，限制了超长推理链

## 相关工作与启发

- LLaVA-o1 和 LlamaV-o1 的 beam search 选择最优推理路径 vs 本文的自验证策略各有优劣
- Mulberry-o1 的蒙特卡洛树搜索方法更重量级，但潜力也更大
- URSA-8B 在数学推理上也有关注，可与本文方法结合
- DeepSeek-R1 的 RL 驱动推理训练方向值得 MLLM 探索

## 评分

- 新颖性：⭐⭐⭐⭐ 自验证策略和 GateMixer 设计有新意
- 技术深度：⭐⭐⭐⭐ 多模块协同，训练流程完整
- 实验充分度：⭐⭐⭐⭐⭐ 13 个基准全面评估
- 实用价值：⭐⭐⭐⭐ 开源且性能领先
- 总体推荐：⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] MM-Verify: Enhancing Multimodal Reasoning with Chain-of-Thought Verification](../../ACL2025/llm_reasoning/mm-verify_enhancing_multimodal_reasoning_with_chain-of-thought_verification.md)
- [\[ACL 2025\] Can Large Language Models Detect Errors in Long Chain-of-Thought Reasoning?](../../ACL2025/llm_reasoning/can_large_language_models_detect_errors_in_long_chain-of-thought_reasoning.md)
- [\[ACL 2025\] Improving Chain-of-Thought Reasoning via Quasi-Symbolic Abstractions](../../ACL2025/llm_reasoning/improving_chain-of-thought_reasoning_via_quasi-symbolic_abstractions.md)
- [\[ACL 2025\] Unlocking General Long Chain-of-Thought Reasoning Capabilities of Large Language Models via Representation Engineering](../../ACL2025/llm_reasoning/glore_long_cot_representation.md)
- [\[ACL 2025\] ClozeMath: Improving Mathematical Reasoning in Language Models by Learning to Fill Equations](../../ACL2025/llm_reasoning/clozemath_improving_mathematical_reasoning_in_language_models_by_learning_to_fil.md)

</div>

<!-- RELATED:END -->
