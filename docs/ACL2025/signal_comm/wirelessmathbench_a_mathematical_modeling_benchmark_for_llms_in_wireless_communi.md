---
title: >-
  [论文解读] WirelessMathBench: A Mathematical Modeling Benchmark for LLMs in Wireless Communications
description: >-
  [ACL 2025][信号/通信][无线通信] 本文提出WirelessMathBench，一个包含587道题目的无线通信数学建模基准，从40篇前沿论文中提取，系统评估LLM在领域特定数学推导上的能力，揭示即使最强的DeepSeek-R1平均准确率也仅38.05%，完整公式推导仅7.83%。 领域现状： LLM在通用数学推理…
tags:
  - "ACL 2025"
  - "信号/通信"
  - "无线通信"
  - "数学推理"
  - "LLM评估基准"
  - "领域特定推理"
  - "公式推导"
---

# WirelessMathBench: A Mathematical Modeling Benchmark for LLMs in Wireless Communications

**会议**: ACL 2025  
**arXiv**: [2505.14354](https://arxiv.org/abs/2505.14354)  
**代码**: [https://lixin.ai/WirelessMathBench](https://lixin.ai/WirelessMathBench)  
**领域**: 信号通信  
**关键词**: 无线通信, 数学推理, LLM评估基准, 领域特定推理, 公式推导

## 一句话总结

本文提出WirelessMathBench，一个包含587道题目的无线通信数学建模基准，从40篇前沿论文中提取，系统评估LLM在领域特定数学推导上的能力，揭示即使最强的DeepSeek-R1平均准确率也仅38.05%，完整公式推导仅7.83%。

## 研究背景与动机

**领域现状**: LLM在通用数学推理上取得了显著进展（如GSM8K、MATH等基准），OpenAI-o1和DeepSeek-R1等推理模型进一步推动了多步推理能力。然而，这些进展主要集中在通用数学领域。

**现有痛点**: 现有数学基准（GSM8K、MATH、OlympiadBench等）集中在中小学到竞赛级别的纯数学问题，缺乏对工程领域（尤其是无线通信）的复杂数学建模能力的评估。无线通信涉及严格的物理约束、维度一致性和领域特定符号系统。

**核心矛盾**: LLM可能在选择题上表现良好（>75%），但在需要重构完整公式推导时能力急剧下降，说明"理解"和"推导"之间存在巨大鸿沟。

**本文目标**: 构建一个专门针对无线通信数学建模的专家级基准，全面评估LLM的符号推理和领域知识运用能力。

**切入角度**: 从真实的前沿研究论文中提取数学模型，设计多层次任务——从选择题到渐进遮盖填空再到完整公式推导——提供递进式难度评估。

**核心 idea**: 通过渐进式公式遮盖策略评估LLM在无线通信中的数学推导能力，暴露当前模型在领域特定符号推理上的根本性不足。

## 方法详解

### 整体框架

WirelessMathBench围绕两个设计原则构建：（1）真实世界复杂性——题目直接来源于同行评审论文；（2）多层递进——从基础选择题到完整推导，覆盖不同难度级别。数据采集流程包括：论文选择→系统模型提取→任务策划→领域专家审核。

### 关键设计

#### 1. 数据来源与覆盖

- **功能**: 从40篇顶级期刊/会议论文中提取数学模型
- **核心思路**: 覆盖核心模型类别（RIS 19篇、MIMO 12篇、UAV 6篇、ISAC 6篇、Satellite 4篇、SIM 3篇、NOMA 2篇）和问题类别（Beamforming 18篇、Channel Estimation 12篇、Performance Analysis 8篇等）
- **设计动机**: 确保评估覆盖无线通信主流研究方向的真实工程挑战

#### 2. 三层任务设计

- **选择题（MCQ）**: 从几个紧密相关的干扰项中选择正确的数学表达式，测试模型的公式识别和回忆能力
- **渐进遮盖填空**: 系统模型公式被渐进式遮盖，分为三个级别——从单变量缺失到多变量遮盖，每级作为独立子问题
- **完整公式推导（FEC）**: 整个公式完全隐藏，仅提供场景描述，要求从基本定义推导完整表达式

#### 3. 数据质量保障

- **功能**: 多轮专家审核确保准确性
- **核心思路**: 半自动提取（LLM初步提取+专家审核修正）+刻意改写避免数据污染（重新表述论文上下文、重组公式呈现方式）
- **设计动机**: 防止LLM通过记忆训练语料而非真正推理来作答

#### 4. 评估管线

- **功能**: 统一的prompt模板 + 两阶段评估
- **核心思路**: MCQ直接比对答案；渐进遮盖和FEC使用GPT-4o作为评估器判断符号等价性
- **设计动机**: 多项式可能有多种等价表示形式，需要语义级别的比较

### 损失函数/训练策略

本文为评估基准，不涉及训练。所有实验采用zero-shot设置，使用各模型默认超参数，不提供额外的chain-of-thought提示。

## 实验关键数据

### 主实验

**16个LLM在WirelessMathBench上的表现**：

| 模型 | MCQ | Level 1 | Level 2 | Level 3 | FEC | 平均 |
|------|-----|---------|---------|---------|-----|------|
| DeepSeek-R1 | 76.00% | 60.00% | 34.91% | 12.50% | 7.83% | **38.05%** |
| OpenAI-o1 | 66.40% | 59.17% | 32.17% | 8.04% | 6.96% | 34.55% |
| GPT-4o | 72.80% | 42.50% | 28.70% | 6.25% | 4.35% | 30.92% |
| DeepSeek-V3 | **78.40%** | 50.00% | 24.35% | 6.25% | 6.96% | 33.19% |
| Gemini-1.5-pro | 65.60% | 43.33% | 29.57% | 9.82% | 6.09% | 30.88% |
| Qwen2.5-Math-72B | 70.40% | 37.50% | 26.09% | 7.14% | 6.09% | 29.44% |
| LLaMA-3.3-70B | 65.60% | 38.33% | 17.39% | 2.68% | 6.09% | 26.02% |
| GPT-3.5-turbo | 45.60% | 7.50% | 10.43% | 1.79% | 1.74% | 13.41% |
| LLaMA-3-8B-Tele | 40.80% | 11.67% | 4.35% | 2.68% | 0.87% | 12.07% |

### 消融实验

**DeepSeek-R1的40个错误案例分析**：

| 错误类型 | 占比 | 说明 |
|----------|------|------|
| 部分填充不匹配 | 31% | 正确填一个遮盖但错误填其他关联遮盖 |
| 符号误解 | 29% | 选错符号或遗漏关键符号元素（如 $\mathbf{H}_{BR}$ vs $\mathbf{H}_{BR}^H$） |
| 推导路径错误 | 24% | 遗漏关键中间步骤或引入无关组件，早期错误传播 |
| 无关系统混入 | 11% | 引入不相关的系统设定（如在RIS-MIMO中插入NOMA干扰因子） |
| 其他 | 4% | 表达式不完整或重复占位符 |

### 关键发现

1. **推理模型的优势**: DeepSeek-R1（38.05%）和OpenAI-o1（34.55%）显著优于其他模型，显式推理策略对多步符号推导至关重要
2. **MCQ强但推导弱**: DeepSeek-V3的MCQ最高达78.40%，但Level 3仅6.25%，FEC仅6.96%，"理解"与"推导"存在巨大鸿沟
3. **渐进式退化**: 随遮盖程度增加，性能急剧下降——DeepSeek-R1从Level 1的60%降到Level 3的12.50%
4. **领域微调收益有限**: LLaMA-3-8B-Tele（电信微调版）反而不如原版LLaMA-3-8B，因为电信微调数据偏重协议知识而非数学推理
5. **数学专用模型有优势**: Qwen2.5-Math-72B（29.44%）在同参数量级中表现突出

## 亮点与洞察

1. **首个工程级数学评估基准**: 不同于纯数学问题，WirelessMathBench要求满足物理约束和维度一致性，更贴近真实科研需求
2. **渐进遮盖策略精巧**: 从MCQ到FEC的递进设计让我们能精确定位模型能力的断裂点
3. **揭示了LLM辅助科研的基本限制**: 即使最强模型在FEC任务上也仅约8%准确率，距离替代人类工程师还有巨大差距
4. **数据污染控制得当**: 专家刻意改写论文内容，确保模型不能靠记忆作弊

## 局限与展望

1. 仅覆盖文本型问题，未包含天线图、仿真图等多模态数据
2. 虽然覆盖了MIMO/RIS等主流方向，但缺少量子通信、太赫兹等新兴领域
3. 自动评估检查最终符号等价性，可能忽略中间推理步骤的错误
4. 所有实验为zero-shot设置，未探索fine-tuning或RAG方法的潜力
5. 587道题的规模相对有限，可进一步扩展

## 相关工作与启发

- **GSM8K / MATH / OlympiadBench**: 通用数学推理基准，WirelessMathBench填补了工程领域数学推理评估的空白
- **TelecomGPT (Zou et al., 2024)**: 探索LLM在无线通信中的应用，但侧重知识检索而非数学推导
- **Maatouk et al. (2023, 2024)**: LLM在电信领域的知识提取，本文在此基础上提出更高层次的推理要求
- **启发**: 领域特定基准对于理解LLM的真实能力边界至关重要，其他工程领域也需要类似基准

## 评分

| 维度 | 评分 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 价值 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Large Language Model (LLM)-enabled In-context Learning for Wireless Network Optimization](../../ICML2025/signal_comm/large_language_model_llm-enabled_in-context_learning_for_wireless_network_optimi.md)
- [\[NeurIPS 2025\] Masked Symbol Modeling for Demodulation of Oversampled Baseband Communication Signals](../../NeurIPS2025/signal_comm/masked_symbol_modeling_for_demodulation_of_oversampled_baseband_communication_si.md)
- [\[ICLR 2026\] Enhancing Instruction Following of LLMs via Activation Steering with Dynamic Rejection](../../ICLR2026/signal_comm/enhancing_instruction_following_of_llms_via_activation_steering_with_dynamic_rej.md)
- [\[ICLR 2026\] Mamba-3: Improved Sequence Modeling using State Space Principles](../../ICLR2026/signal_comm/mamba-3_improved_sequence_modeling_using_state_space_principles.md)
- [\[CVPR 2025\] Neural Video Compression with Context Modulation](../../CVPR2025/signal_comm/neural_video_compression_with_context_modulation.md)

</div>

<!-- RELATED:END -->
