---
title: >-
  [论文解读] OmniBal: Towards Fast Instruction-Tuning for Vision-Language Models via Omniverse Computation Balance
description: >-
  [ICML 2025][多模态VLM][VLM训练加速] 针对大规模视觉语言模型 instruction-tuning 训练中因数据和模型异构性导致的计算不平衡问题，提出 OmniBal 框架从数据、模型、内存三个层面系统性平衡跨设备计算负载，在 InternVL-Chat 上实现约 1.8× 训练加速。
tags:
  - "ICML 2025"
  - "多模态VLM"
  - "VLM训练加速"
  - "3D并行"
  - "计算平衡"
  - "Pipeline Parallelism"
  - "动态批处理"
---

# OmniBal: Towards Fast Instruction-Tuning for Vision-Language Models via Omniverse Computation Balance

**会议**: ICML 2025  
**arXiv**: [2407.20761](https://arxiv.org/abs/2407.20761)  
**代码**: [github.com/ModelTC/OmniBal](https://github.com/ModelTC/OmniBal)  
**领域**: 多模态/视觉语言模型, 分布式训练, 系统优化  
**关键词**: VLM训练加速, 3D并行, 计算平衡, Pipeline Parallelism, 动态批处理

## 一句话总结

针对大规模视觉语言模型 instruction-tuning 训练中因数据和模型异构性导致的计算不平衡问题，提出 OmniBal 框架从数据、模型、内存三个层面系统性平衡跨设备计算负载，在 InternVL-Chat 上实现约 1.8× 训练加速。

## 研究背景与动机

VLM 的训练规模持续扩大：InternVL-Chat 相比 LLaVA-1.5 将数据从 665K 增至 5M，图像分辨率从 336×336 增至 3840×2160，视觉编码器从 ~300M 的 ViT-L 增至 ~6B 的 InternViT。

然而，大规模 3D 并行训练 VLM 时存在严重的**计算不平衡**问题：

**数据不平衡**：VLM 输入包含变长文本和变数量图像，导致 mini-batch 大小剧烈波动（输入大小标准差高达 $1.4K \pm 0.9K$ tokens）

**模型不平衡**：ViT 和 LLM 的 transformer block 计算量差异大（前向时间标准差达 $85 \pm 93$ ms）

**内存不平衡**：动态输入让 GPU 内存需求波动（$39 \pm 23$ G），迫使使用最激进的重计算策略

这些问题在 LLM 文本预训练中不存在（输入固定、可 packing、模型同构），是 VLM 训练特有的挑战。

## 方法详解

### 整体框架

OmniBal 从三个紧密关联的角度解决计算不平衡：数据→模型→内存，形成递进关系。

### 1. 平衡动态 Mini-Batch（数据层面）

定义两个衡量数据不平衡的指标：
- **Pad Ratio**（设备内）：$\text{PadRatio} = \frac{\sum_i^B(t_{max} - t_i)}{t_{max} \times B}$
- **Dist Ratio**（跨设备）：$\text{DistRatio} = \frac{\sum_i^N(T_{max} - T_i)}{T_{max} \times N}$

提出 **ISF (Iterative Sampling and Filtering)** 算法：
- **采样阶段**：随机将样本加入当前组，直到图像数 $I_v$ 或文本长度 $I_t$ 达到预定义上限 $Q_v, Q_t$
- **过滤阶段**：移除不满足下限 $Q'_v, Q'_t$ 的组

交替迭代 $T$ 次，将 Pad Ratio 从 0.31 降至 0，Dist Ratio 从 0.34 降至 0.02。

### 2. 平衡模型分区（模型层面）

目标：找到最优 pipeline 分区策略 $P^* = \arg\min_{P_i} f(P_i)$。

由于 ViT 和 LLM 的异构性，简单的参数或层数均分无效。提出基于搜索的方法：
- Profile 每层前向时间 $\text{FWD}(l_i)$
- 用贪心算法计算锚点分区 $P^+$
- 在 $P^+$ 周围半径 $r$ 内生成候选集
- 使用两个排序指标：前向时间方差 $\text{VAR(fwd\_time)}$ 和通信量 $\text{SUM(comm)}$

### 3. 平衡自适应重计算（内存层面）

得益于前两步稳定了计算负载和内存需求：
1. 开启全部重计算，记录各阶段剩余 GPU 内存 $M_r$
2. 手动关闭部分层的重计算，记录内存变化 $\Delta M_r$
3. 估计 ViT 和 LLM 各层的内存节省量 $M_v, M_t$
4. 根据估计为每个阶段自适应选择最优重计算策略

## 实验关键数据

### 主实验

| 模型 | 平衡? | Backend | GPU Days | 加速比 |
|------|-------|---------|----------|--------|
| 6+20B | ✗ | DeepSpeed | 38.9 | 1.00× |
| 6+20B | ✓ | DeepSpeed | 25.3 | **1.54×** |
| 6+20B | ✗ | Megatron | 61.8 | 0.63× |
| 6+20B | ✓ | Megatron | 21.3 | **1.83×** |
| 6+34B | ✗ | DeepSpeed | 54.3 | 1.00× |
| 6+34B | ✓ | DeepSpeed | 35.5 | **1.53×** |
| 6+34B | ✗ | Megatron | 75.4 | 0.72× |
| 6+34B | ✓ | Megatron | 30.5 | **1.80×** |

### 模型性能保持

平衡训练不影响模型效果，在 MMBench、ChartQA、AI2D、MMVet、MME 五个 benchmark 上与基线持平或略优。

### 消融实验

- 数据平衡：ISF 将 GPU Days 从 61.8 减至 51.9（仅数据平衡）
- 模型平衡：进一步降至 29.0（Balanced Model Partitioning）
- 内存平衡：最终降至 21.3（自适应重计算）

## 亮点与洞察

1. **系统性解决方案**：首次全面识别并解决 VLM instruction-tuning 中的计算不平衡问题
2. **三层递进设计**：数据平衡使模型分区可行，两者使内存分析可行，形成统一框架
3. **显著加速**：1.8× 以上加速且无性能损失，实际节省大量 GPU 资源
4. **通用性强**：在不同模型规模、数据集、硬件上一致有效

## 局限性

- ISF 算法的 $Q_v, Q_t$ 需要根据数据集统计确定
- 搜索空间在大 pipeline 并行度下可能增大
- 目前仅验证了 InternVL-Chat 架构

## 相关工作

- 3D 并行（Megatron-LM、DeepSpeed ZeRO）
- Pipeline 并行优化（GPipe、PipeDream、AdaPipe）
- VLM 训练（LLaVA、InternVL-Chat、BLIP）

## 评分

⭐⭐⭐⭐ — 工程贡献扎实，1.8× 加速对大规模 VLM 训练有实际价值。方法设计系统、实验全面。偏系统优化方向，理论新颖性有限但实用性很高。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Parrot: Multilingual Visual Instruction Tuning](parrot_multilingual_visual_instruction_tuning.md)
- [\[ICML 2025\] Understanding and Mitigating Miscalibration in Prompt Tuning for Vision-Language Models](understanding_and_mitigating_miscalibration_in_prompt_tuning_for_vision-language.md)
- [\[ICML 2025\] Dynamic Mixture of Curriculum LoRA Experts for Continual Multimodal Instruction Tuning](dynamic_mixture_of_curriculum_lora_experts_for_continual_multimodal_instruction_.md)
- [\[ICML 2025\] Streamline Without Sacrifice — Squeeze out Computation Redundancy in LMM](streamline_without_sacrifice_--_squeeze_out_computation_redundancy_in_lmm.md)
- [\[NeurIPS 2025\] Visual Instruction Bottleneck Tuning](../../NeurIPS2025/multimodal_vlm/visual_instruction_bottleneck_tuning.md)

</div>

<!-- RELATED:END -->
