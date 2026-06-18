---
title: >-
  [论文解读] MaTVLM: Hybrid Mamba-Transformer for Efficient Vision-Language Modeling
description: >-
  [多模态VLM] 提出MaTVLM，将预训练VLM中部分Transformer层替换为Mamba-2层并通过单阶段知识蒸馏训练，在保持竞争性性能的同时实现3.6倍推理加速和27.5%显存降低。 当前视觉语言模型（VLM）主要基于Transformer架构，其自注意力机制的二次复杂度在处理长序列时带来严重的计算和内存瓶颈…
tags:
  - "多模态VLM"
---

# MaTVLM: Hybrid Mamba-Transformer for Efficient Vision-Language Modeling

## 元信息
- **会议**: ICCV 2025
- **arXiv**: [2503.13440](https://arxiv.org/abs/2503.13440)
- **代码**: [https://github.com/hustvl/MaTVLM](https://github.com/hustvl/MaTVLM)
- **领域**: 多模态VLM
- **关键词**: Mamba-Transformer混合架构, 知识蒸馏, 高效VLM, 线性复杂度, 推理加速

## 一句话总结

提出MaTVLM，将预训练VLM中部分Transformer层替换为Mamba-2层并通过单阶段知识蒸馏训练，在保持竞争性性能的同时实现3.6倍推理加速和27.5%显存降低。

## 研究背景与动机

当前视觉语言模型（VLM）主要基于Transformer架构，其自注意力机制的二次复杂度在处理长序列时带来严重的计算和内存瓶颈。尽管Mamba等RNN模型以线性复杂度展现了作为替代方案的潜力，但现有的Mamba-based VLM存在三个关键问题：

**全局上下文捕获不足**：Mamba的顺序处理限制了其捕获长距离依赖的能力，导致在复杂推理任务上表现不佳

**收敛困难**：顺序处理导致梯度传播低效，从头训练VLM收敛缓慢，计算成本高昂

**训练流程复杂**：现有Mamba VLM需要多阶段训练才能达到最优性能，难以扩展

作者观察到Attention和Mamba-2之间存在内在的数学联系——移除softmax后的注意力本质上等价于线性RNN，这为两者的权重初始化提供了理论基础。基于此，作者提出将Mamba与Transformer有机结合，在效率和性能之间取得平衡。

## 方法详解

### 整体框架

MaTVLM建立在预训练VLM（TinyLLaVA）之上，包含视觉编码器、连接器和语言模型三个组件。核心改动在语言模型部分：将部分Transformer decoder层替换为Mamba-2 decoder层（仅替换注意力模块，其他组件保持不变），按比例均匀分布（如12.5%、25%、50%）。

### Attention与Mamba-2的数学联系

标准注意力的输出为：

$$y_n = \sum_{t=1}^{n} \text{softmax}\left(\frac{Q_n K_t^\top}{\sqrt{d}}\right) V_t$$

移除softmax后，可以重新表述为线性RNN形式：

$$h_n = h_{n-1} + K_n^\top W_V x_n, \quad y_n = \frac{Q_n}{\sqrt{d}} h_n$$

将此与Mamba-2的SSM表达式 $h_t = A_t h_{t-1} + B_t x_t$，$y_t = C_t^\top h_t$ 对比，可得到映射关系：

$$x_t \leftrightarrow W_V x_t, \quad B_t \leftrightarrow W_K x_t, \quad C_t \leftrightarrow W_Q x_t$$

因此Mamba-2中 $x$、$B$、$C$ 的线性权重可以从注意力的 $V$、$K$、$Q$ 权重初始化，加速收敛。

### 知识蒸馏策略

采用单阶段蒸馏，仅训练Mamba-2层和连接器，Transformer层冻结。损失函数包含三个部分：

**1. 概率分布蒸馏**（KL散度）：

$$\mathcal{L}_{\text{prob}} = T^2 \cdot \text{KL}(P_t \| P_s) = T^2 \cdot \sum_i P_t(i) \log\frac{P_t(i)}{P_s(i)}$$

其中 $P_t$ 和 $P_s$ 为温度缩放后的教师和学生模型输出分布。

**2. 层级蒸馏**（L2对齐）：

$$\mathcal{L}_{\text{layer}} = \sum_{i=1}^{m} \|T_{l_i}(x) - S_{l_i}(x)\|_2$$

确保学生模型Mamba层的输出与教师模型对应Transformer层输出对齐。

**3. 总损失**：

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{layer}} + \beta \cdot \mathcal{L}_{\text{prob}} + \gamma \cdot \mathcal{L}_{\text{ce}}$$

实验中设 $\alpha = \beta = 1.0$，$\gamma = 0$（省略序列预测损失）。

## 实验

### 主实验结果

| 方法 | LLM | MME-P | MMB | TextVQA | GQA | MM-Vet | SQA-I | POPE | MMMU |
|------|-----|-------|-----|---------|-----|--------|-------|------|------|
| TinyLLaVA (Teacher) | Phi2-2.7B | 1466.4 | 66.1 | 60.3 | 62.1 | 37.5 | 73.0 | 87.2 | 38.4 |
| **MaTVLM (Ours)** | Hybrid | **1484.0** | 61.2 | 57.7 | 60.4 | 35.3 | 68.1 | **87.4** | **40.0** |
| Cobra (Mamba) | Mamba 2.8B | - | - | 57.9 | - | - | - | 88.2 | - |
| LLaVA-Phi | Phi2-2.7B | 1335.1 | 59.8 | 48.6 | - | 28.9 | 68.4 | 85.0 | - |
| MoE-LLaVA-2.7Bx4 | Phi2-2.7B | 1396.4 | 65.5 | 50.2 | 61.1 | 31.1 | 68.7 | 85.0 | - |

MaTVLM在MME上比教师模型提升17.6分，POPE和MMMU分别提升0.2和1.6分。相比同等规模模型，MME提升87.7分，TextVQA提升7.0分。

### 效率对比

| 指标 | TinyLLaVA | MaTVLM (25% Mamba) | MaTVLM (50% Mamba) |
|------|-----------|--------------------|--------------------|
| 推理加速 | 1.0× | ~2.0× | **3.6×** |
| 显存降低 | 0% | ~15% | **27.5%** |

随着生成token长度增加，MaTVLM与TinyLLaVA的推理时间差距持续扩大。

### 消融实验

| 蒸馏策略 | MME | GQA | TextVQA | POPE |
|---------|-----|-----|---------|------|
| 无蒸馏 | 1320.5 | 56.8 | 52.1 | 83.4 |
| 仅 $\mathcal{L}_{\text{prob}}$ | 1445.2 | 59.6 | 56.4 | 86.8 |
| 仅 $\mathcal{L}_{\text{layer}}$ | 1401.3 | 58.7 | 55.2 | 85.9 |
| $\mathcal{L}_{\text{prob}} + \mathcal{L}_{\text{layer}}$ | **1484.0** | **60.4** | **57.7** | **87.4** |

两种蒸馏损失的组合效果最优，单独使用概率分布蒸馏比层级蒸馏效果更好。

### 关键发现

1. 混合架构在多数benchmark上达到了与纯Transformer教师模型竞争性的性能
2. 25%的Mamba-2替换比例在性能-效率权衡上最优
3. 从Attention权重初始化Mamba-2是加速收敛的关键
4. 单阶段蒸馏避免了多阶段训练的复杂性

## 亮点与洞察

1. **理论贡献**：揭示了Attention与Mamba-2之间Q↔C、K↔B、V↔x的数学对应关系，为权重迁移提供理论依据
2. **实用价值**：3.6倍加速+27.5%显存降低使得VLM在资源受限环境中的部署成为可能
3. **训练效率**：单阶段蒸馏极大简化了训练流程，降低了技术门槛
4. **可扩展性**：方法可以推广到其他VLM架构和更大规模模型

## 局限性

1. 仅在TinyLLaVA（3.1B参数）上验证，未扩展到更大规模模型
2. 替换比例的选择（12.5%/25%/50%）缺乏自适应策略
3. Mamba-2在需要强全局推理的任务上仍存在性能差距（如ScienceQA下降4.9分）
4. 蒸馏依赖于高质量的预训练教师模型

## 相关工作

- **高效VLM**：TinyLLaVA、MobileVLM、Qwen2.5-VL等
- **SSM模型**：Mamba、Mamba-2及其在VLM中的应用（Cobra、ML-Mamba）
- **混合架构**：MambaInLlama、MOHAWK、MambaVision
- **知识蒸馏**：DistillVLM、MAD、LLAVADI

## 评分

- **创新性**: ★★★★☆ — Attention-Mamba权重映射和单阶段蒸馏策略新颖
- **实用性**: ★★★★★ — 显著的推理加速和显存节省具有强实用价值
- **实验完整度**: ★★★★☆ — 多benchmark验证+效率分析+消融实验
- **写作质量**: ★★★★☆ — 逻辑清晰，数学推导严谨

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Physics Context Builders: A Modular Framework for Physical Reasoning in Vision-Language Models](physics_context_builders_a_modular_framework_for_physical_reasoning_in_vision-la.md)
- [\[ICCV 2025\] R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization](r1-vl_learning_to_reason_with_multimodal_large_language_models_via_step-wise_gro.md)
- [\[ICCV 2025\] OrderChain: Towards General Instruct-Tuning for Stimulating the Ordinal Understanding Ability of MLLM](orderchain_towards_general_instruct-tuning_for_stimulating_the_ordinal_understan.md)
- [\[ICCV 2025\] Training-Free Personalization via Retrieval and Reasoning on Fingerprints](training-free_personalization_via_retrieval_and_reasoning_on_fingerprints.md)
- [\[ICCV 2025\] MetaMorph: Multimodal Understanding and Generation via Instruction Tuning](metamorph_multimodal_understanding_and_generation_via_instruction_tuning.md)

</div>

<!-- RELATED:END -->
