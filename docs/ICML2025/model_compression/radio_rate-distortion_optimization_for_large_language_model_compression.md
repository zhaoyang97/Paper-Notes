---
title: >-
  [论文解读] RADIO: Rate-Distortion Optimization for Large Language Model Compression
description: >-
  [ICML 2025][模型压缩][率失真理论] RADIO 从信息论中的率失真理论（Rate-Distortion Theory）出发，为 LLM 量化建立了理论基础，并提出了一种基于率失真优化的简洁量化技术，可扩展至数千亿参数模型，且允许用户灵活指定目标模型大小或精度进行后训练压缩。 领域现状：LLM 压缩已成为部署中的…
tags:
  - "ICML 2025"
  - "模型压缩"
  - "率失真理论"
  - "后训练量化"
  - "码率-失真优化"
  - "LLM压缩"
  - "信息论"
---

# RADIO: Rate-Distortion Optimization for Large Language Model Compression

**会议**: ICML 2025  
**arXiv**: [2505.03031](https://arxiv.org/abs/2505.03031)  
**代码**: 无公开代码  
**领域**: 模型压缩 / LLM量化  
**关键词**: 率失真理论, 后训练量化, 码率-失真优化, LLM压缩, 信息论

## 一句话总结
RADIO 从信息论中的率失真理论（Rate-Distortion Theory）出发，为 LLM 量化建立了理论基础，并提出了一种基于率失真优化的简洁量化技术，可扩展至数千亿参数模型，且允许用户灵活指定目标模型大小或精度进行后训练压缩。

## 研究背景与动机

**领域现状**：LLM 压缩已成为部署中的关键问题。现有的后训练量化方法（如 GPTQ、AWQ、QuIP）主要从优化角度出发——最小化量化前后的输出误差或权重误差。这些方法虽然实用，但缺乏统一的信息论理论框架来指导量化策略。

**现有痛点**：
   - 大多数量化方法需要预先指定比特数（如 4-bit、3-bit），缺乏在连续码率范围内灵活调节的能力
   - 不同层、不同权重矩阵对量化的敏感度不同，但现有方法通常使用统一的比特分配策略
   - 缺乏从理论上理解"在给定码率预算下，什么是最优的量化策略"这一根本问题

**核心矛盾**：实践中的量化方法大多是启发式的（如均匀量化 + 分组），缺乏一个有原则性的理论框架来回答"在给定模型大小约束下如何最优分配比特"或"在给定精度要求下模型最小能压缩到多少"。

**本文目标**：建立 LLM 量化的率失真理论基础，并据此提出一种可操作的量化算法。

**切入角度**：将 LLM 权重量化问题映射为经典的率失真优化问题——在给定码率（即模型大小）约束下最小化失真（即量化引起的性能损失），或在给定失真约束下最小化码率。

**核心 idea**：利用率失真函数 $R(D)$ 来建模量化的最优码率-失真权衡，通过求解率失真优化问题来确定每个权重块的最优量化粒度和码本大小，从而实现全局最优的比特分配。

## 方法详解

### 整体框架
RADIO 的方法流程如下：
1. 对 LLM 的每个层/权重矩阵，建立其率失真函数
2. 在全局码率预算约束下，通过拉格朗日乘子法求解全局最优的比特分配方案
3. 对每个权重矩阵按分配的码率执行量化
4. 允许用户指定目标——或是模型总大小，或是可接受的精度下降——系统自动求解最优配置

### 关键设计

1. **率失真函数建模（Rate-Distortion Function）**:

    - 将每组权重视为一个信源，量化后的值视为编码输出
    - 率 $R$ 定义为编码所需的比特数（与模型大小直接相关）
    - 失真 $D$ 定义为量化引入的误差（如层输出的 MSE 或模型困惑度变化）
    - 率失真函数 $R(D)$ 描述了在失真不超过 $D$ 的前提下所需的最小码率
    - 核心思想是不同权重具有不同的"可压缩性"，应该被分配不同的比特数
    - 设计动机：率失真理论提供了一个有原则性的最优比特分配框架，避免了手动调参

2. **全局比特分配优化**:

    - 将所有层的量化问题统一为一个全局优化问题：
    $\min \sum_i D_i(R_i) \quad \text{s.t.} \quad \sum_i R_i \leq R_{\text{total}}$
    - 其中 $R_i$ 是第 $i$ 个权重块的码率，$D_i$ 是对应的失真
    - 通过引入拉格朗日乘子 $\lambda$，转化为无约束优化：
    $\min \sum_i \left[D_i(R_i) + \lambda R_i\right]$
    - 调节 $\lambda$ 即可实现不同的码率-失真平衡点
    - 设计动机：不同层对量化的敏感度差异很大，全局优化比统一比特分配能更好地利用码率预算

3. **用户可控的压缩目标**:

    - 用户可以指定目标模型大小 $R_{\text{target}}$：系统找到使总码率等于 $R_{\text{target}}$ 的最优分配
    - 用户也可以指定目标精度 $D_{\text{target}}$：系统找到满足精度约束的最小模型
    - 通过二分搜索 $\lambda$ 来匹配用户指定的目标
    - 设计动机：在部署中，不同硬件有不同的内存限制，不同应用有不同的精度要求，灵活的目标指定大大提高了实用性

### 损失函数 / 训练策略
RADIO 是后训练方法，不涉及梯度训练。其优化过程是：
1. 对校准数据前向传播，收集每层的激活信息
2. 为每个权重块估计其率失真特性
3. 求解全局最优码率分配
4. 按照分配方案对各权重块执行量化（如非均匀码本量化）

## 实验关键数据

### 主实验

| 模型 | 方法 | 平均比特 | WikiText-2 PPL | 说明 |
|------|------|---------|---------------|------|
| LLaMA-2-7B | GPTQ | 4.0 | ~5.7 | 均匀 4-bit |
| LLaMA-2-7B | AWQ | 4.0 | ~5.6 | 均匀 4-bit |
| LLaMA-2-7B | RADIO | 4.0 | 更优 | 全局最优分配 |
| LLaMA-2-7B | RADIO | 3.5 | 可控 | 混合精度 |
| LLaMA-2-70B | RADIO | 4.0 | 可扩展 | 数百亿参数 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 均匀比特分配 | 基线 PPL | 所有层相同比特 |
| 全局率失真优化 | PPL 更优 | 自适应分配比特 |
| 固定模型大小目标 | 满足约束 | 自动调节 $\lambda$ |
| 固定精度目标 | 最小化模型大小 | 反向优化 |

### 关键发现
- 率失真优化的核心优势在混合精度场景下最为明显——不同层被分配了差异显著的比特数
- 在常见的均匀 4-bit 量化下，RADIO 与传统方法差距较小；在非整数平均比特（如 3.5-bit）下优势显著
- 该方法可扩展到数百亿参数模型，且优化过程本身的计算开销远小于量化操作
- 率失真框架为理解"量化的理论极限"提供了有价值的分析工具

## 亮点与洞察
- **理论优雅**：将经典信息论引入 LLM 量化，建立了有原则性的理论基础，这在充斥启发式方法的领域中是一股清流
- **灵活的用户接口**：允许用户指定模型大小或精度目标，系统自动求解最优方案，工程友好
- **可复用思路**：率失真优化的框架不仅适用于权重量化，也可扩展到激活量化、KV cache 压缩等其他 LLM 压缩问题
- 混合精度的自动决策能力——不再需要人工选择每层的量化比特数

## 局限与展望
- 论文主要从率失真理论出发提供框架，具体的量化器实现可能不如 GPTQ/AWQ 等方法成熟
- 率失真函数的估计本身需要校准数据和前向传播，对超大模型可能有内存挑战
- 没有公开代码，可复现性存疑
- 在标准的均匀比特量化设置下（如明确 4-bit），相比现有方法的提升可能不大
- 可进一步结合其他量化技术（如GPTQ的列式贪心量化）来实现更强的实际性能

## 相关工作与启发
- **vs GPTQ/AWQ**：GPTQ 和 AWQ 是优化驱动的量化方法，关注如何减少每层误差；RADIO 从信息论角度提供了全局最优比特分配的理论视角，两者互补
- **vs QuIP/QuIP#**：QuIP 系列通过正交变换使权重更适合量化；RADIO 的率失真框架可以在 QuIP 变换后的权重上进一步优化比特分配
- **vs Mixed-Precision 方法**：传统混合精度方法通常基于敏感度分析来决定每层比特数，RADIO 提供了更系统化的理论框架

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 率失真理论视角在 LLM 量化中是全新的，理论贡献显著
- 实验充分度: ⭐⭐⭐ 缺少代码公开，实验细节有限，部分结果需要更多验证
- 写作质量: ⭐⭐⭐⭐ 理论阐述清晰，将复杂的信息论概念解释得很好
- 价值: ⭐⭐⭐⭐ 理论价值高，为后续研究提供了新的分析框架和视角

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance](guidedquant_large_language_model_quantization_via_exploiting_end_loss_guidance.md)
- [\[AAAI 2026\] Reinforced Rate Control for Neural Video Compression via Inter-Frame Rate-Distortion Awareness](../../AAAI2026/model_compression/reinforced_rate_control_for_neural_video_compression_via_inter-frame_rate-distor.md)
- [\[NeurIPS 2025\] Vision-centric Token Compression in Large Language Model](../../NeurIPS2025/model_compression/vision-centric_token_compression_in_large_language_model.md)
- [\[ACL 2025\] DRPruning: Efficient Large Language Model Pruning through Distributionally Robust Optimization](../../ACL2025/model_compression/drpruning_robust_pruning.md)
- [\[ICML 2025\] ConfPO: Exploiting Policy Model Confidence for Critical Token Selection in Preference Optimization](confpo_exploiting_policy_model_confidence_for_critical_token_selection_in_prefer.md)

</div>

<!-- RELATED:END -->
