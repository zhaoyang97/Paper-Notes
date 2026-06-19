---
title: >-
  [论文解读] Seeing Far and Clearly: Mitigating Hallucinations in MLLMs with Attention Causal Decoding
description: >-
  [CVPR 2025][幻觉检测][多模态大模型] 提出 FarSight，一种即插即用的无训练解码策略，通过在因果掩码的上三角矩阵中引入注意力寄存器来吸收异常 token 的过度注意力，并设计递减掩蔽率的位置感知编码增强远距离视觉 token 的信息传播，有效缓解多模态大模型中的初始幻觉和雪球幻觉。
tags:
  - "CVPR 2025"
  - "幻觉检测"
  - "多模态大模型"
  - "幻觉缓解"
  - "注意力机制"
  - "因果解码"
  - "位置编码"
---

# Seeing Far and Clearly: Mitigating Hallucinations in MLLMs with Attention Causal Decoding

**会议**: CVPR 2025  
**arXiv**: [2505.16652](https://arxiv.org/abs/2505.16652)  
**代码**: [https://mllms-farsight.github.io/](https://mllms-farsight.github.io/)  
**领域**: 幻觉检测  
**关键词**: 多模态大模型, 幻觉缓解, 注意力机制, 因果解码, 位置编码

## 一句话总结
提出 FarSight，一种即插即用的无训练解码策略，通过在因果掩码的上三角矩阵中引入注意力寄存器来吸收异常 token 的过度注意力，并设计递减掩蔽率的位置感知编码增强远距离视觉 token 的信息传播，有效缓解多模态大模型中的初始幻觉和雪球幻觉。

## 研究背景与动机

**领域现状**：多模态大语言模型（MLLMs）如 LLaVA、InternVL 等在视觉问答任务中表现出色，但普遍存在幻觉问题——生成与图像内容矛盾的文本描述。

**现有痛点**：现有缓解方法要么需要额外训练/数据（指令微调、外部知识检索），要么只关注结果层面的对比解码而未深入分析幻觉的根因。已有方法对"雪球幻觉"（模型为保持前文一致性而连续产生错误）的缓解效果有限。

**核心矛盾**：幻觉的根本原因在于多模态 token 间的交互不充分。作者发现两个关键问题：(1) **注意力坍塌**——softmax 机制迫使所有注意力分数非零且归一化，导致低信息量的异常 token（标点符号、视觉背景）获得不成比例的高注意力；(2) **位置信息衰减**——RoPE 的长程衰减特性使视觉 token 的信息随生成文本长度增加而逐渐消失。

**本文目标**：设计一种无需训练、即插即用的解码策略，通过优化因果掩码来缓解注意力坍塌和位置信息衰减。

**切入角度**：直接在注意力机制的因果掩码上做文章——传统因果掩码的上三角全为负无穷（屏蔽未来 token），作者提出利用这一"空闲空间"来放置注意力寄存器，吸纳异常 token 的过度注意力。

**核心 idea**：在因果掩码的上三角矩阵中设置线性衰减的注意力寄存器分数，softmax 归一化时这些寄存器参与计算（吸收多余注意力），归一化后再清零以保持因果性，从而让有效 token 获得更均衡的注意力分配。

## 方法详解

### 整体框架
FarSight 作为插件替换 MLLM 解码器中每一层的标准因果掩码操作。输入原始注意力分数矩阵 $\omega$，先用下三角矩阵 $C$ 清零上三角的注意力值，然后加入预计算的注意力寄存器矩阵 $\mathcal{P}$（上三角部分为线性衰减值），经过 softmax 归一化后再乘以 $C$ 清除上三角的概率值。整个过程只修改掩码操作，不改变模型权重。

### 关键设计

1. **上三角注意力寄存器（Attention Registers）**:

    - 功能：提供专用的"注意力接收槽"来吸收本应分配给异常 token 的过度注意力
    - 核心思路：构建上三角注意力寄存器矩阵 $\mathcal{P}$，对位置 $j > i$ 的寄存器值设为 $\mathcal{P}_{i,j} = -(j-i) \cdot \sigma$，其中 $\sigma$ 为衰减率超参数。最终注意力矩阵 $\mathbf{W} = \omega \cdot C + \mathcal{P}$，经 softmax 后再乘 $C$ 清零上三角：$\tilde{\mathbf{W}} = \text{SoftMax}(\mathbf{W}) \cdot C$。线性衰减确保寄存器与 RoPE 的相对位置编码一致
    - 设计动机：softmax 要求所有注意力分数归一化为 1，导致即使低信息量 token 也会获得显著注意力。寄存器提供了一个"安全出口"让多余的注意力被吸收，而不是被迫分散到无关 token 上

2. **位置感知编码（Positional Awareness Encoding）**:

    - 功能：增强生成文本对远距离视觉 token 的关注能力，缓解位置信息衰减
    - 核心思路：通过注意力寄存器的线性衰减设计，使得每行有效注意力的累积和随行索引 $i$ 单调递增（$\beta_1 < \beta_2 < \cdots < \beta_n = 1$）。这意味着后面位置的 token 能聚合更多前文历史信息，实现了绝对位置信息的隐式编码。越靠后的生成 token 对早期视觉 token 的总关注度越高，弥补了 RoPE 长程衰减的不足
    - 设计动机：RoPE 只编码相对位置，随距离增大视觉-文本 token 间的信息流逐渐消失。对于视频任务中的长序列，这个问题尤为严重

3. **动态寄存器注意力分配**:

    - 功能：在每个解码步自适应优化注意力分配
    - 核心思路：寄存器分数通过类 ALiBi 的偏置生成，但方向翻转（应用于上三角而非下三角），不同注意力头使用不同的衰减率。每个解码步都重新计算完整的 softmax 归一化，使寄存器动态适应当前上下文
    - 设计动机：不同层、不同头、不同时间步的注意力分布差异很大，需要灵活的动态机制而非固定规则

### 损失函数 / 训练策略
FarSight 完全不需要训练。只需在推理时修改因果掩码操作即可。主要超参数为衰减率 $\sigma$，控制寄存器吸收注意力的强度。

## 实验关键数据

### 主实验

| 模型 | 方法 | CHAIR_S ↓ | CHAIR_I ↓ | POPE-R ↑ | POPE-P ↑ |
|------|------|-----------|-----------|----------|----------|
| LLaVA-1.5-7B | Baseline | 48.0 | 13.9 | 87.0 | 82.8 |
| LLaVA-1.5-7B | + FarSight | 显著降低 | 显著降低 | 提升 | 提升 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 仅 RoPE（原始） | CHAIR_S=48.0 | 基线 |
| 固定视觉位置编码 (FixVPE) | 有所改善 | 仅固定视觉 token 的位置编码 |
| 仅文本位置编码 (EDVT) | 部分改善 | 去除视觉 token 的位置编码 |
| FarSight（完整） | 最优 | 注意力寄存器 + 位置感知编码 |

### 关键发现
- 幻觉可分为初始幻觉和雪球幻觉两类，雪球幻觉在视频字幕任务中占比特别高
- 现有对比解码方法无法有效减少雪球幻觉比例，而 FarSight 通过改善信息传播同时缓解两类幻觉
- 注意力寄存器和位置感知编码两个组件各有独立贡献且协同效果更优
- 方法在图像和视频两类 benchmark 上均有效，对视频长序列任务改善尤为显著

## 亮点与洞察
- **上三角空间的巧妙利用**：因果掩码的上三角传统上被完全屏蔽（负无穷），本文发现可以在 softmax 前利用这一空间放置"注意力缓冲区"，softmax 后再清零，既不违反因果性又能调节注意力分配。这一思路非常巧妙且通用
- **理论与实践结合**：对注意力坍塌和位置信息衰减的分析有严格的数学推导（互信息不等式、单调递增证明），不同于多数解码策略论文的纯经验做法
- **即插即用的实用价值**：不需要任何训练或额外数据，只修改注意力掩码操作，计算开销极小，可直接应用于任何基于 transformer 的 MLLM

## 局限与展望
- 衰减率 $\sigma$ 需要手动调节，不同模型/任务可能需要不同的最优值
- 论文主要在 7B 级别模型上验证，对更大规模模型（如 70B+）的效果未知
- 虽然理论分析了位置信息衰减，但寄存器机制能否完全补偿 RoPE 在超长序列上的局限尚未充分验证
- 仅对因果解码进行干预，未探索 prefill 阶段的注意力优化

## 相关工作与启发
- **vs OPERA**: OPERA 也关注了"summary token"的注意力汇聚问题，但通过惩罚特定 token 来处理。FarSight 则从掩码层面提供了更优雅的解决方案，不直接修改注意力分数
- **vs VCD (Visual Contrastive Decoding)**: VCD 通过对比有无视觉输入的输出分布来减少幻觉，增加了推理开销。FarSight 不需要额外的前向传播
- **vs ALiBi/StableMask**: 这些方法主要改善单模态文本的长度外推，FarSight 专门针对多模态场景的视觉-语言 token 交互进行优化

## 评分
- 新颖性: ⭐⭐⭐⭐ 上三角注意力寄存器的设计新颖，对幻觉原因的分析深入
- 实验充分度: ⭐⭐⭐⭐ 多模型多 benchmark 验证，有消融和位置编码方法对比
- 写作质量: ⭐⭐⭐⭐ 理论推导严谨，伪代码清晰，图表直观
- 价值: ⭐⭐⭐⭐ 即插即用的实用方案，对理解和缓解 MLLM 幻觉有较高参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Tell Model Where to Look: Mitigating Hallucinations in MLLMs by Vision-Guided Attention](../../CVPR2026/hallucination/tell_model_where_to_look_mitigating_hallucinations_in_mllms_by_vision-guided_att.md)
- [\[NeurIPS 2025\] Seeing is Believing? Mitigating OCR Hallucinations in Multimodal Large Language Models](../../NeurIPS2025/hallucination/seeing_is_believing_mitigating_ocr_hallucinations_in_multimodal_large_language_m.md)
- [\[CVPR 2026\] COPO: Causal-Oriented Policy Optimization for Hallucinations of MLLMs](../../CVPR2026/hallucination/copo_causal-oriented_policy_optimization_for_hallucinations_of_mllms.md)
- [\[CVPR 2025\] Octopus: Alleviating Hallucination via Dynamic Contrastive Decoding](octopus_alleviating_hallucination_via_dynamic_contrastive_decoding.md)
- [\[ACL 2025\] Mixture of Decoding: An Attention-Inspired Adaptive Decoding Strategy to Mitigate Hallucination in Multimodal LLMs](../../ACL2025/hallucination/mixture_of_decoding_an_attention-inspired_adaptive_decoding_strategy_to_mitigate.md)

</div>

<!-- RELATED:END -->
