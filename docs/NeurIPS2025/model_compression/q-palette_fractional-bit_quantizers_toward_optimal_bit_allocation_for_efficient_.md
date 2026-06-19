---
title: >-
  [论文解读] Q-Palette: Fractional-Bit Quantizers Toward Optimal Bit Allocation for Efficient LLM Deployment
description: >-
  [NeurIPS 2025][模型压缩][模型量化] 从信息论角度推导高斯化权重的最优比特分配,提出 Q-Palette 分数位量化器集合和混合方案量化框架,在 LLM 推理中实现近最优的量化性能和推理加速。 权重后训练量化（Weight-only PTQ）是降低 LLM 推理延迟和内存占用的关键技术,尤其适用于内存受限的边…
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "模型量化"
  - "LLM推理"
  - "分数位量化"
  - "失真率"
  - "CUDA优化"
---

# Q-Palette: Fractional-Bit Quantizers Toward Optimal Bit Allocation for Efficient LLM Deployment

**会议**: NeurIPS 2025

**arXiv**: [2509.20214](https://arxiv.org/abs/2509.20214)

**代码**: [GitHub](https://github.com/snu-mllab/Q-Palette)

**领域**: 模型压缩

**关键词**: 模型量化, LLM推理, 分数位量化, 失真率, CUDA优化

## 一句话总结

从信息论角度推导高斯化权重的最优比特分配,提出 Q-Palette 分数位量化器集合和混合方案量化框架,在 LLM 推理中实现近最优的量化性能和推理加速。

## 研究背景与动机

权重后训练量化（Weight-only PTQ）是降低 LLM 推理延迟和内存占用的关键技术,尤其适用于内存受限的边缘设备。然而,LLM 权重中的重尾离群值使量化变得复杂。

近期研究进展与挑战：

**旋转方法**: 通过 Hadamard 旋转将权重变换为近高斯分布,减少离群值影响

**整数位限制**: 现有量化器仅支持整数位宽(2/3/4-bit),无法实现细粒度的比特分配

**最优性差距**: 实际量化器与信息论下界（高斯失真率界）之间存在较大差距

**混合精度局限**: 现有混合精度方法仅在有限选项间选择,未考虑量化器类型的混合

## 方法详解

### 整体框架

Q-Palette 包含两大组件：(1) 一组覆盖不同比特宽和失真水平的分数位量化器；(2) 一个联合优化量化器选择和层融合决策的混合方案框架。

### 关键设计

**1. 信息论最优比特分配**

- 对高斯化（旋转后）的权重,推导 Rate-Distortion 函数
- 最优分配要求使用**任意精度**的量化器,而非仅整数位宽
- 推导出每层的最优比特数: $b_l^* = \frac{1}{2}\log_2\frac{\sigma_l^2}{\lambda}$, 其中 $\lambda$ 由总比特预算确定

**2. 分数位量化器集合**

Q-Palette 提供从近最优失真到快速推理的多种量化器:
- **Trellis-coded quantizers (TCQ)**: 通过有限状态机实现,失真接近高斯失真率界
- **向量量化器 (VQ)**: 利用多维编码获得分数位宽 (如 2.5-bit)
- **标量量化器**: 简单高效,适合对速度要求高的场景
- 所有量化器均有优化的 **CUDA kernel** 实现

**3. 混合方案量化框架**

- 联合优化两个层面: (a) 每层选择哪种量化器 (b) 是否融合相邻层
- 使用动态规划在给定资源约束下求解最优方案
- 目标: $\min \sum_l D_l(q_l) \quad \text{s.t.} \quad \sum_l R_l(q_l) \leq B$

### 损失函数 / 训练策略

- 无需重新训练: 纯后训练量化 (PTQ)
- 校准: 使用少量（或零）校准数据
- 分配优化: 通过拉格朗日乘子法迭代求解

## 实验关键数据

### 主实验

LLaMA-2-7B 在不同平均比特宽下的困惑度 (PPL, WikiText-2):

| 方法 | 2-bit | 2.5-bit | 3-bit | 4-bit |
|------|-------|---------|-------|-------|
| GPTQ | 发散 | 12.85 | 8.32 | 6.09 |
| QuIP# | 9.15 | 7.85 | 6.83 | 5.98 |
| AQLM | 8.78 | 7.52 | 6.71 | 5.95 |
| Q-Palette (Ours) | **8.21** | **7.18** | **6.52** | **5.92** |

LLaMA-2-13B (PPL, WikiText-2):

| 方法 | 2-bit | 2.5-bit | 3-bit | 4-bit |
|------|-------|---------|-------|-------|
| GPTQ | 发散 | 9.15 | 6.85 | 5.42 |
| QuIP# | 7.52 | 6.58 | 5.92 | 5.35 |
| AQLM | 7.28 | 6.35 | 5.85 | 5.32 |
| Q-Palette (Ours) | **6.85** | **6.12** | **5.73** | **5.30** |

### 消融实验

不同量化器类型的失真-速度权衡 (LLaMA-2-7B, 3-bit):

| 量化器 | PPL | 解码延迟 (ms/token) | 与DRB差距 |
|--------|-----|-------------------|----------|
| 标量 (Uniform) | 7.15 | 2.1 | +1.23 dB |
| 向量 (Group-VQ) | 6.78 | 3.5 | +0.52 dB |
| TCQ | 6.55 | 5.2 | +0.08 dB |
| 高斯失真率界 | - | - | 0 dB |

### 关键发现

1. 分数位量化在低比特（< 3-bit）场景下优势最明显,缩小与理论最优的差距
2. TCQ 量化器的失真仅比高斯失真率界高 0.08 dB,接近信息论极限
3. 混合方案比统一使用单一量化器进一步降低 0.3-0.5 PPL
4. 优化的 CUDA kernel 使 TCQ 的推理开销控制在合理范围内

## 亮点与洞察

- **信息论视角**: 从 Rate-Distortion 理论出发,提供了量化的理论最优目标
- **工程完备**: 不仅有理论,还有完整的 CUDA kernel 实现
- **灵活组合**: Q-Palette 允许不同层使用不同量化策略,最大化整体效率

## 局限与展望

1. TCQ 量化器的编解码复杂度较高,实际推理加速有限
2. 当前仅支持权重量化,未扩展到 KV cache 量化
3. 混合方案的搜索空间随模型规模增大而增大
4. 在极低比特（< 2-bit）场景下性能仍有较大退化

## 相关工作与启发

- **QuIP#** (Tseng et al.): 基于随机旋转的量化方法
- **AQLM** (Egiazarian et al.): 自适应分组量化
- **Rate-Distortion 理论**: Shannon 信息论提供的量化下界

## 评分

- ⭐ 创新性: 8/10 — 将信息论工具引入LLM量化是新颖视角
- ⭐ 实用性: 9/10 — 开源代码+CUDA kernel,可直接部署
- ⭐ 写作质量: 8/10 — 理论到工程的衔接流畅

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Learning Grouped Lattice Vector Quantizers for Low-Bit LLM Compression](learning_grouped_lattice_vector_quantizers_for_low-bit_llm_compression.md)
- [\[NeurIPS 2025\] ParetoQ: Improving Scaling Laws in Extremely Low-bit LLM Quantization](paretoq_improving_scaling_laws_in_extremely_low-bit_llm_quantization.md)
- [\[NeurIPS 2025\] Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference](ada-kv_optimizing_kv_cache_eviction_by_adaptive_budget_allocation_for_efficient_.md)
- [\[NeurIPS 2025\] LittleBit: Ultra Low-Bit Quantization via Latent Factorization](littlebit_ultra_low-bit_quantization_via_latent_factorization.md)
- [\[NeurIPS 2025\] FALQON: Accelerating LoRA Fine-tuning with Low-Bit Floating-Point Arithmetic](falqon_accelerating_lora_fine-tuning_with_low-bit_floating-point_arithmetic.md)

</div>

<!-- RELATED:END -->
