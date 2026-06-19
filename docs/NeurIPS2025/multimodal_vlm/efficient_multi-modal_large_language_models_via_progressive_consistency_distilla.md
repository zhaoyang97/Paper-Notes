---
title: >-
  [论文解读] Efficient Multi-modal Large Language Models via Progressive Consistency Distillation
description: >-
  [NeurIPS 2025][多模态VLM][多模态LLM] 提出EPIC框架，通过渐进式一致性蒸馏（Token和Layer两个维度）解决视觉token压缩训练中特征空间扰动导致的学习困难，在不修改模型架构的前提下实现高效多模态LLM。 多模态大语言模型（MLLMs）将视觉编码器提取的视觉token输入LLM进行理解和推理…
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "多模态LLM"
  - "视觉token压缩"
  - "渐进式蒸馏"
  - "一致性蒸馏"
  - "推理效率"
---

# Efficient Multi-modal Large Language Models via Progressive Consistency Distillation

**会议**: NeurIPS 2025  
**arXiv**: [2510.00515](https://arxiv.org/abs/2510.00515)  
**代码**: [有](https://github.com/ZichenWen1/EPIC)  
**领域**: 多模态VLM  
**关键词**: 多模态LLM, 视觉token压缩, 渐进式蒸馏, 一致性蒸馏, 推理效率

## 一句话总结

提出EPIC框架，通过渐进式一致性蒸馏（Token和Layer两个维度）解决视觉token压缩训练中特征空间扰动导致的学习困难，在不修改模型架构的前提下实现高效多模态LLM。

## 研究背景与动机

多模态大语言模型（MLLMs）将视觉编码器提取的视觉token输入LLM进行理解和推理。然而，视觉token数量巨大（如LLaVA-v1.5使用576个token），带来显著计算开销：
- 注意力机制的二次复杂度使长token序列成为推理瓶颈
- 高分辨率图像和多帧视频进一步加剧问题

现有视觉token压缩方法分两类：

**免训练方法**（FastV、SparseVLM等）：基于重要性或冗余性剪枝，性能损失明显

**训练感知方法**（MQT-LLaVA、TokenPacker等）：通过架构修改实现灵活压缩

**核心问题**：现有训练感知方法主要依赖架构改进，**忽视了token压缩带来的训练困难**。如图1所示：
- token压缩改变了特征空间分布（引入扰动）
- 扰动导致参数空间的最优点偏移
- 压缩比越高，最优点偏移越大
- 直接训练容易陷入局部最优

## 方法详解

### 整体框架

EPIC基于标准MLLM架构（CLIP + MLP投影器 + Vicuna LLM），**不修改任何架构组件**。核心创新在于训练策略：将特征空间扰动分解为token维度和layer维度，分别提出Token一致性蒸馏（TCD）和Layer一致性蒸馏（LCD）。

共享权重的单一模型同时扮演教师和学生角色。

### 关键设计

**1. Token一致性蒸馏（TCD）**

核心思想：**渐进式增加压缩比**，使每一步的最优点偏移较小，优化更容易。

- **学生模型**：在训练迭代 $t$ 时，从范围 $[R_{\min,t}^{\text{stu}}, R_{\max,t}^{\text{stu}}]$ 采样压缩比
- **教师模型**：使用比学生稍低的压缩比（差值为 $\Delta_t$），提供更好的特征指导
- **渐进策略**：
    - 训练初期：教师和学生都使用低压缩比（简单任务）
    - 训练后期：压缩比逐渐增大，教师-学生差距 $\Delta_t$ 也渐进增大
- 当差距过大时，学生难以从教师获得有效指导，因此 $\Delta_t$ 也遵循渐进策略

任意即插即用的token压缩器（FastV、DART、随机剪枝）均可作为压缩算子。

**2. Layer一致性蒸馏（LCD）**

基于观察：视觉token在LLM深层注意力显著降低，深层压缩对输出影响小。

- 定义归一化训练进度 $\beta_t = t/T$
- 压缩层位置：$\ell_t = \text{Round}(L - \beta_t(L - \ell_{\min}))$
- **渐进策略**：训练初期在最深层压缩（影响最小），逐渐移到浅层（影响增大）
- 同样维持教师-学生的压缩比差距

### 损失函数 / 训练策略

**TCD/LCD训练目标**：
$$\mathcal{L}_{\text{total}}(\theta) = (1-\lambda) \cdot \mathcal{L}_{\text{SFT}}(\theta) + \lambda \cdot \mathcal{L}_{\text{TCD/LCD}}(\theta)$$

其中 $\lambda = 0.7$，$\mathcal{L}_{\text{SFT}}$ 是标准自回归交叉熵损失。

蒸馏损失为温度缩放的KL散度：
$$\mathcal{L}_{\text{TCD}}(\theta) = \mathbb{E}_{I,P,t}\left[\text{KL}(p_{\text{tea}} \| p_{\text{stu}})\right]$$
$$p_{\text{tea}} = \text{Softmax}(h_{\text{tea}}/\tau), \quad p_{\text{stu}} = \text{Softmax}(h_{\text{stu}}/\tau)$$

训练仅需一个阶段（视觉指令微调），约12小时在8×A100 GPU上完成。

## 实验关键数据

### 主实验

**表1：10个视觉理解基准测试性能（部分，与LLaVA-v1.5-7B对比）**

| 方法 | #Visual Tokens | VQAv2 | GQA | MME | MMB | 平均(%) | vs LLaVA |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| LLaVA-v1.5 | 576 | 72.2 | 61.9 | 1785 | 64.1 | 61.4 | — |
| MQT-LLaVA | 64 | 65.6 | 58.7 | 1810 | 61.3 | 56.8 | -4.6 |
| MQT-LLaVA | 256 | 68.3 | 60.1 | 1740 | 61.7 | 57.8 | -3.6 |
| TokenPacker | 144 | 71.3 | 62.0 | 1716 | 63.9 | 59.9 | -1.5 |
| **TCD (Ours)** | 256 | 72.7 | 61.4 | 1807 | **66.1** | **61.7** | **+0.3** |
| **TCD (Ours)** | 128 | 69.7 | 59.9 | **1861** | 65.6 | 61.3 | -0.1 |
| **TCD (Ours)** | 64 | 66.1 | 57.1 | 1809 | 64.2 | 59.4 | -2.0 |
| **LCD (Ours)** | 256 | 72.6 | 62.0 | 1834 | 64.3 | **62.2** | **+0.8** |
| **LCD (Ours)** | 128 | 69.2 | 60.6 | 1832 | 64.1 | 61.3 | -0.1 |
| **LCD (Ours)** | 64 | 66.0 | 58.3 | 1794 | 62.1 | 59.4 | -2.0 |

关键发现：
- 保留128个token（压缩77.8%），性能几乎无损（-0.1%）
- 保留256个token时**超越**原始LLaVA-v1.5（+0.3 ~ +0.8%）
- 仅64个token也只下降2%，远优于MQT-LLaVA、LLaVA-Mini等

**表2：推理效率分析**

| 方法 | Visual Tokens | KV Cache (MB) ↓ | CUDA Time (s) ↓ | FLOPs (T) ↓ |
|:---:|:---:|:---:|:---:|:---:|
| LLaVA-v1.5 | 576 | 367.2 | 1103.5 | 9.3 |
| EPIC + DART | 64 | 40.9 (↓88.9%) | 744.3 (↓32.6%) | 1.5 (↓83.9%) |
| EPIC + Random | 64 | 40.9 (↓88.9%) | 697.3 (↓36.8%) | 1.5 (↓83.9%) |

KV缓存减少 **88.9%**，FLOPs减少 **83.9%**，实际加速约 **1.6×**。

### 消融实验

**表3：TCD消融**

| 方法 | VQAv2 | MME | MMB | 平均(%) |
|:---:|:---:|:---:|:---:|:---:|
| TCD (128 tokens) | 69.7 | **1861** | **65.6** | **61.3** |
| 去掉蒸馏损失 | 67.2 | 1745 | 63.8 | 59.8 (-1.5) |
| 去掉渐进式压缩比 | 67.1 | 1788 | 63.8 | 59.1 (-2.2) |

**表4：LCD消融**

| 方法 | VQAv2 | MME | MMB | 平均(%) |
|:---:|:---:|:---:|:---:|:---:|
| LCD (128 tokens) | 69.2 | **1832** | 64.1 | **61.3** |
| 去掉蒸馏损失 | 67.1 | 1761 | 62.9 | 60.5 (-0.8) |
| 去掉渐进式压缩层 | 68.7 | 1776 | 63.1 | 60.3 (-1.0) |

关键消融结论：
- **渐进式策略不可或缺**：去掉后TCD平均下降2.2%，LCD下降1.0%
- **教师指导有效**：去掉蒸馏损失后TCD下降1.5%，LCD下降0.8%
- TCD对渐进策略更敏感，LCD对两者都需要

### 关键发现

1. **极端压缩不划算**：从576到128个token的FLOPs急剧下降，但从64到36→18的FLOPs减少极小，且性能急剧下降。作者定义了"高ROI区域"（≥64 tokens）和"低ROI区域"（<64 tokens）
2. **跨压缩策略泛化良好**：用DART训练的模型在FastV和Random推理时也表现良好
3. **训练成本低**：仅需一阶段微调（12小时 vs 30-48小时用于架构修改方法）
4. **暗示大量视觉token冗余**：128个token即可匹配576个token的性能

## 亮点与洞察

1. **问题洞察深刻**：首次明确指出token压缩的核心困难是特征空间扰动→参数空间最优点偏移→优化陷入局部最优
2. **无需架构修改**：纯训练策略改进，与任何即插即用的token压缩器兼容
3. **自蒸馏设计简洁**：教师和学生共享权重，无需维护额外模型
4. **渐进式学习有理论直觉**：每一步最优点偏移较小，类似课程学习的思想

## 局限与展望

1. 目前仅在LLaVA-v1.5（7B）上验证，更大模型（13B+）和更新架构（LLaVA-Next等）的适用性待验证
2. TCD和LCD目前独立使用，联合使用的效果未探索
3. 渐进式调度（线性增长）可能不是最优的，自适应调度策略值得研究
4. 蒸馏超参数 $\lambda=0.7$ 和温度 $\tau$ 的敏感性分析不充分
5. 视频理解场景（token更多）的评估未涉及

## 相关工作与启发

- **LLaVA** [Liu et al., 2023/2024]：本文的基础架构，全量576个视觉token
- **FastV** [Chen et al., 2024]：基于注意力分数的免训练token剪枝，本文的压缩组件之一
- **MQT-LLaVA** [Li et al., 2024]：动态Q-former编码可变长度token，训练感知方法的主要对比
- **TokenPacker** [Li et al., 2024]：粗到精视觉投影器，通过架构修改压缩token
- **VoCo-LLaMA** [Ye et al., 2024]：通过注意力修改将视觉信息转移到少量VoCo token

## 评分

- **新颖性**: ★★★★☆ — 渐进式蒸馏框架新颖，自蒸馏+渐进学习的结合巧妙
- **技术深度**: ★★★★☆ — 设计直觉清晰，token维度和layer维度的分解有洞察力
- **实验充分性**: ★★★★★ — 10个基准、3种压缩策略、详尽消融、效率分析齐全
- **写作质量**: ★★★★☆ — 损失景观可视化（图1）直观有效，表格清晰
- **实用性**: ★★★★★ — 代码开源，不修改架构，训练成本低，直接可用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](../../ICCV2025/multimodal_vlm/large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)
- [\[NeurIPS 2025\] VaMP: Variational Multi-Modal Prompt Learning for Vision-Language Models](vamp_variational_multi-modal_prompt_learning_for_vision-language_models.md)
- [\[CVPR 2025\] A Two-Stage Progressive Pre-training using Multi-Modal Contrastive Masked Autoencoders](../../CVPR2025/multimodal_vlm/multi-modal_contrastive_masked_autoencoders_a_two-stage_progressive_pre-training.md)
- [\[NeurIPS 2025\] mmWalk: Towards Multi-modal Multi-view Walking Assistance](mmwalk_towards_multi-modal_multi-view_walking_assistance.md)
- [\[ACL 2025\] Jailbreak Large Vision-Language Models Through Multi-Modal Linkage](../../ACL2025/multimodal_vlm/jailbreak_large_vision-language_models_through_multi-modal_linkage.md)

</div>

<!-- RELATED:END -->
