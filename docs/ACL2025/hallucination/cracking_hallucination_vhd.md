---
title: >-
  [论文解读] Cracking the Code of Hallucination in LVLMs with Vision-aware Head Divergence
description: >-
  [ACL 2025][幻觉检测][幻觉缓解] 提出 VHD 指标量化每个注意力头输出对视觉输入的敏感程度，发现仅少数注意力头对视觉信息高度敏感而模型过度依赖语言先验是导致幻觉的关键因素，进而设计 VHR 免训练方法逐层自适应增强视觉感知头的贡献（$\alpha=2$），在 CHAIR 上将 LLaVA-1.5 的 CHAIR$_S$ 从 49.68 降至 33.32，且几乎无额外推理开销。
tags:
  - "ACL 2025"
  - "幻觉检测"
  - "幻觉缓解"
  - "Vision-aware Head Divergence"
  - "注意力头分析"
  - "语言偏置"
  - "免训练解码"
---

# Cracking the Code of Hallucination in LVLMs with Vision-aware Head Divergence

**会议**: ACL 2025  
**arXiv**: [2412.13949](https://arxiv.org/abs/2412.13949)  
**代码**: [VHR](https://github.com/jinghan1he/VHR)  
**领域**: 幻觉检测  
**关键词**: 幻觉缓解, Vision-aware Head Divergence, 注意力头分析, 语言偏置, 免训练解码

## 一句话总结

提出 VHD 指标量化每个注意力头输出对视觉输入的敏感程度，发现仅少数注意力头对视觉信息高度敏感而模型过度依赖语言先验是导致幻觉的关键因素，进而设计 VHR 免训练方法逐层自适应增强视觉感知头的贡献（$\alpha=2$），在 CHAIR 上将 LLaVA-1.5 的 CHAIR$_S$ 从 49.68 降至 33.32，且几乎无额外推理开销。

## 研究背景与动机

**领域现状**：大型视觉语言模型（LVLMs）在多模态推理上取得显著进展，但幻觉问题——生成文本与视觉内容不符——严重损害模型的准确性和可靠性。

**现有痛点**：(1) 现有方法（对齐训练、后处理、对比解码）主要在输出层面干预，未深入探究幻觉的内部机制；(2) 对比解码（VCD、DoLa 等）直接操纵 logits 分布，引入生成不稳定性；(3) 缺乏对模型内部驱动幻觉的注意力机制的系统分析。

**核心矛盾**：LVLMs 倾向于优先使用语言模式（语言偏置），即使不提供图像输入，模型也能生成高度一致的内容描述。这种偏置嵌入模型参数中，导致输出更依赖内部知识而非视觉上下文。

**本文目标** 从多头注意力机制的角度探究幻觉的内部驱动因素，并基于发现设计主动缓解方案。

**切入角度**：受 LLM 中"上下文头"和"记忆头"研究的启发，探索不同注意力头对视觉内容的差异化敏感性。

**核心 idea**：量化注意力头对视觉上下文的敏感度差异，自适应放大高敏感度注意力头的输出以增强模型的视觉依赖。

## 方法详解

### 整体框架

分两阶段：(1) 通过 VHD 指标分析注意力头的视觉敏感性，并聚合为 T-VHD 指标建立幻觉与语言偏置的定量关联；(2) 基于 VHD 分数逐层自适应选择并放大关键注意力头（VHR），整个过程在一次前向传播中完成，仅在第一个生成步骤需要额外一次前向传播计算 VHD。

### 关键设计

1. **Vision-aware Head Divergence (VHD) 指标**
    - **功能**：量化每个注意力头输出对视觉输入的敏感度
    - **核心思路**：在生成步骤 $t$，分别计算有图像输入和无图像输入时第 $l$ 层第 $i$ 个注意力头的输出，取欧氏距离：$\text{VHD}_{l,i} = d(A_{l,i}(y_t|y_{<t}, x_V, x_T),\ A_{l,i}(y_t|y_{<t}, x_T))$。聚合每层 top-$k$ 的 VHD 得到 Token-VHD (T-VHD)
    - **关键发现**：仅少数注意力头展现显著高 VHD 分数；幻觉词/句对应更低的 T-VHD 分数，统计验证了语言偏置与幻觉的关联
    - **设计动机**：需要一个无需标注、逐样本自适应的指标来捕获模型对视觉信息的依赖程度

2. **Vision-aware Head Reinforcement (VHR)**
    - **功能**：免训练地增强视觉感知注意力头的贡献以主动缓解幻觉
    - **核心思路**：对每一层，先计算 VHD 分数并剔除异常值（VHD 高但源于无图像时激活激增的"负面视觉敏感"头），然后选择 VHD 分数超过中位数的前半数注意力头，将其输出放大 $\alpha$ 倍：$\widetilde{A}_{l,i} = \alpha \cdot A_{l,i}$ if $i \in H_l$
    - **三个关键实现细节**：(a) 逐层应用——在处理当前层时前序层已被增强，保证 VHD 计算一致性；(b) 仅在第一个生成步骤确定关键头并在后续步骤复用，与 KV cache 兼容；(c) 无需额外标注，每个样本自适应选择不同的头
    - **设计动机**：直接在模型内部干预优于在输出层面修正，放大视觉感知头可将注意力输出方向重新朝向视觉证据

3. **注意力输出重定向理论分析**
    - **功能**：证明放大操作有效重定向 MHA 输出方向
    - **核心思路**：由于 RMSNorm 的归一化，只有方向影响后续 FFN 输入。Proposition 1 证明：放大第 $h$ 个头的输出 $\alpha$ 倍后，FFN 输入 $\widetilde{Z}_l$ 与仅包含该头贡献的 $Z_{l,h}$ 的余弦相似度严格增大：$\cos(\widetilde{Z}_l, Z_{l,h}) > \cos(Z_l, Z_{l,h})$
    - **设计动机**：为放大操作的合理性提供数学保证

## 实验关键数据

### 主实验——CHAIR 基准（MSCOCO 500 图像，5 次随机采样平均）

| 方法 | InstructBLIP CHAIR$_S$↓ | InstructBLIP CHAIR$_I$↓ | LLaVA-1.5 CHAIR$_S$↓ | LLaVA-1.5 CHAIR$_I$↓ | LLaVA-NeXT CHAIR$_S$↓ | LLaVA-NeXT CHAIR$_I$↓ |
|------|:-:|:-:|:-:|:-:|:-:|:-:|
| Greedy | 45.32 | 12.98 | 49.68 | 14.32 | 29.08 | 8.08 |
| DoLa | 46.00 | 13.00 | 50.88 | 14.64 | 28.76 | 8.12 |
| VCD | 50.72 | 14.42 | 51.92 | 15.42 | 30.80 | 8.72 |
| OPERA | 45.76 | 13.06 | 44.28 | 13.36 | - | - |
| EAH | 46.40 | 13.13 | 38.76 | 11.05 | 28.13 | 6.62 |
| **VHR** | **37.76** | **9.75** | **33.32** | **9.71** | **24.96** | **6.80** |

### 消融实验——注意力头选择策略（CHAIR 基准）

| 配置 | InstructBLIP CHAIR$_S$↓ | LLaVA-1.5 CHAIR$_S$↓ | LLaVA-NeXT CHAIR$_S$↓ |
|------|:-:|:-:|:-:|
| VHR（自适应选头） | 37.76 | 33.32 | 24.96 |
| fixed VHR（固定选头） | 45.40 | 44.72 | 36.96 |
| outlier VHR（不剔除异常） | 37.76 | 36.88 | 24.64 |

### 关键发现

- VHR 在三个 LVLM 上全面超越所有免训练基线，LLaVA-1.5 上 CHAIR$_S$ 降低 16.36 个点（49.68→33.32）
- 自适应逐样本选头至关重要——固定选头性能大幅下降（33.32→44.72）
- 异常 VHD 分值剔除有效，防止放大"负面视觉敏感"头
- 放大因子 $\alpha=2$ 效果最佳，$\alpha=4$ 导致模型行为异常，$\alpha<1$（削弱视觉头）则幻觉显著加重
- POPE F1 和 LLaVA-Bench 准确性也有提升，且生成长度和自然度基本不受影响
- 推理时间开销可忽略：第一步多一次前向传播，后续仅需缩放操作

## 亮点与洞察

1. **VHD/T-VHD 指标设计优雅**：仅通过移除图像输入比较注意力头输出差异，无需任何标注，可在每个样本上即时计算，兼具分析能力和实用价值
2. **从"事后修正"到"事前干预"的范式转变**：不同于对比解码在 logits 层面修正，VHR 直接在模型内部增强视觉感知，理论更清晰
3. **理论证明有力**：Proposition 1 严格证明放大操作的方向重定向效果，不仅是经验trick
4. **实验中语言偏置的可视化**：移除图像后模型仍生成高度一致的描述，强有力地展示了语言偏置现象

## 局限与展望

1. 仅关注多头注意力机制，视觉编码器和 FFN 模块对幻觉的贡献未探讨
2. VHD 需要额外一次无图像前向传播，虽然开销小但不为零
3. 未在更大规模模型（如 >13B）和更多 LVLM 架构上验证
4. 对幻觉类型（对象/属性/关系）未做细粒度分析

## 相关工作与启发

- VHD 的"有无视觉对比"思路可推广到其他模态（如音频）的注意力头分析
- 逐层自适应选择-增强的策略可应用于其他需要增强特定信息流的场景（如安全对齐中的safety heads）
- T-VHD 指标可作为幻觉检测的实时信号，用于推理时自动识别不可靠 token

## 评分

⭐⭐⭐⭐

- **新颖性** ⭐⭐⭐⭐：从注意力头内部机制切入幻觉分析，VHD 指标和 VHR 方法新颖
- **实验充分度** ⭐⭐⭐⭐：三个 LVLM、多个基准、消融实验覆盖全面
- **写作质量** ⭐⭐⭐⭐：理论推导清晰，可视化分析直观
- **价值** ⭐⭐⭐⭐：免训练、高效、有理论支撑的幻觉缓解方案，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] ETF: An Entity Tracing Framework for Hallucination Detection in Code Summaries](etf_an_entity_tracing_framework_for_hallucination_detection_in_code_summaries.md)
- [\[CVPR 2026\] CausalLens: Sensitivity-Guided Multi-Head Causal Intervention for Hallucination Mitigation in Large Vision-Language Models](../../CVPR2026/hallucination/causallens_sensitivity-guided_multi-head_causal_intervention_for_hallucination_m.md)
- [\[ACL 2026\] Hallucination Detection in LLMs with Topological Divergence on Attention Graphs](../../ACL2026/hallucination/hallucination_detection_in_llms_with_topological_divergence_on_attention_graphs.md)
- [\[CVPR 2026\] Residual Decoding: Mitigating Hallucinations in Large Vision-Language Models via History-Aware Residual Guidance](../../CVPR2026/hallucination/residual_decoding_mitigating_hallucinations_in_large_vision-language_models_via_.md)
- [\[ACL 2025\] Activation Steering Decoding: Mitigating Hallucination in Large Vision-Language Models through Bidirectional Hidden State Intervention](activation_steering_decoding_mitigating_hallucination_in_large_vision-language_m.md)

</div>

<!-- RELATED:END -->
