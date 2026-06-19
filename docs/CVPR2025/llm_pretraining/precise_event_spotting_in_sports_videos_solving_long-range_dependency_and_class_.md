---
title: >-
  [论文解读] Precise Event Spotting in Sports Videos: Solving Long-Range Dependency and Class Imbalance
description: >-
  [CVPR 2025][预训练][精确事件定位] 提出端到端可训练的精确事件定位框架，通过自适应时空精炼模块（ASTRM）增强特征的时空信息，并引入Soft Instance Contrastive（SoftIC）损失解决类别不平衡问题，在SoccerNet V2 tight设置上以73.74 mAP超越SOTA。
tags:
  - "CVPR 2025"
  - "预训练"
  - "精确事件定位"
  - "体育视频"
  - "类别不平衡"
  - "时序依赖"
  - "对比学习"
---

# Precise Event Spotting in Sports Videos: Solving Long-Range Dependency and Class Imbalance

**会议**: CVPR 2025  
**arXiv**: [2503.00147](https://arxiv.org/abs/2503.00147)  
**代码**: 无  
**领域**: LLM预训练  
**关键词**: 精确事件定位, 体育视频, 类别不平衡, 时序依赖, 对比学习

## 一句话总结

提出端到端可训练的精确事件定位框架，通过自适应时空精炼模块（ASTRM）增强特征的时空信息，并引入Soft Instance Contrastive（SoftIC）损失解决类别不平衡问题，在SoccerNet V2 tight设置上以73.74 mAP超越SOTA。

## 研究背景与动机

精确事件定位（PES）要求在长未裁剪视频中识别事件发生的精确帧，在体育视频领域尤为重要。面临两大核心挑战：

1. **长程时序依赖**：许多事件的类别只能通过观察远距离帧来确定。例如"射门中靶"和"射门偏靶"在射门瞬间标注，但判断需要看球的后续轨迹。仅靠邻近帧不足以做出正确预测
2. **类别不平衡**：体育事件的发生频率天然不均。足球中"球出界"远多于"红牌"，导致模型在少数类上性能很差

现有方法的不足：
- Transformer方法（如ASTRA）依赖大型预训练特征提取器，self-attention在长视频上权重矩阵过大难以学好
- E2E-Spot/T-DEED等CNN方法时序信息提取能力有限
- 几乎没有方法专门处理类别不平衡，仅ASTRA尝试了mixup变体

## 方法详解

### 整体框架

端到端可训练网络：CNN时空特征提取器（RegNetY + ASTRM）→ Bi-GRU长程时序模块 → 线性分类器。使用ASAM优化器训练，配合SoftIC损失处理类别不平衡。

### 关键设计

**设计一：自适应时空精炼模块（ASTRM）**

- **功能**：在特征提取过程中同时增强空间和时序信息
- **核心思路**：三个子模块串联——(a) 局部空间：通道维度MaxPool+AvgPool后7×7卷积+Sigmoid，生成空间注意力权重；(b) 局部时序：3×1×1的3D卷积捕获局部时序变化+Sigmoid；(c) 全局时序：全局平均池化+FC层生成动态卷积核，与前两阶段增强后的特征做卷积

$$\Psi(\mathbf{x}) = ((\mathbf{x} \odot (1 + \mathcal{F}_s(\mathbf{x}))) \odot (1 + \mathcal{F}_t(\mathbf{x}))) * \mathcal{G}_t(\mathbf{x})$$

- **设计动机**：空间信息对事件判断同样重要（如球的位置），不能只用时序。全局时序通过动态卷积核注入全局信息，简洁高效

**设计二：Soft Instance Contrastive（SoftIC）损失**

- **功能**：在mixup增强下有效促进类内紧凑和类间分离，缓解类别不平衡
- **核心思路**：在Instance Contrastive Loss基础上扩展以支持soft label（mixup产生的非one-hot标签）。记忆库中存储特征及其类别权重，对比时用权重 $w_j$ 缩放特征相似度
- **设计动机**：IC Loss假设每个样本只有单一标签，与mixup不兼容。SoftIC通过权重化处理混合标签，同时利用mixup的正则化和对比学习的特征分离能力

$$\mathcal{L}_{SIC}(\mathbf{z}_i, \omega) = \frac{1}{\omega|M(\mathbf{c}_i)|} \sum_{(\mathbf{z}_j, w_j) \in M(\mathbf{c}_i)} \log \frac{\exp(\mathbf{z}_i \cdot w_j \mathbf{z}_j / \tau)}{\sum_{(\mathbf{z}_k, w_k) \in A(\mathbf{c}_i)} \exp(\mathbf{z}_i \cdot w_k \mathbf{z}_k / \tau)}$$

**设计三：ASAM优化策略**

- **功能**：提升训练模型的泛化能力
- **核心思路**：使用Adaptive Sharpness-Aware Minimization替代标准Adam，考虑损失流形的尺度自适应邻域来计算"平坦"最小值
- **设计动机**：SAM的固定邻域不够灵活，ASAM的自适应邻域在事件定位任务上更有效

### 损失函数

$$\mathcal{L}_{final} = \mathcal{L}_{BCE} + \lambda_{SIC} \cdot \mathcal{L}_{SIC}$$

其中 $\lambda_{SIC} = 0.001$。

## 实验关键数据

### SoccerNet V2测试集

| 方法 | Params(M) | GFLOPs | Tight mAP | Loose mAP |
|------|-----------|--------|-----------|-----------|
| E2E-Spot (RegNet-Y 200) | 4.46 | 39.61 | 61.19 | 73.25 |
| Spivak | 17.46 | 461.89 | 65.10 | 78.50 |
| ASTRA | 44.33 | 8.83 | 66.63 | 78.14 |
| COMEDIAN (ViSwin-T) | 70.12 | 222.76 | 71.33 | 77.07 |
| **Proposed** | **6.46** | **60.25** | **73.74** | **79.11** |

### 少数类性能对比（tight设置）

| 类别 | E2E-Spot | COMEDIAN | Proposed |
|------|----------|----------|----------|
| Red card | ~30% | ~55% | **~72%** |
| Yellow→Red | ~20% | ~45% | **~65%** |
| Penalty | ~50% | ~60% | **~75%** |

### 消融实验

| 组件 | SoccerNet V2 Tight |
|------|-------------------|
| Baseline (RegNetY + GRU) | 61.19 |
| + ASTRM | 66.82 |
| + SoftIC | 70.41 |
| + ASAM | **73.74** |

### 关键发现

1. 仅6.46M参数即超越70M参数的COMEDIAN，提升2.41 tight mAP，证明端到端训练+精心设计优于堆叠大模型
2. SoftIC在少数类上的提升尤为显著（Red card类AP翻倍以上），有效缓解了类别不平衡
3. ASTRM贡献+5.63 mAP，SoftIC贡献+3.59 mAP，ASAM贡献+3.33 mAP——三个组件贡献均显著
4. 端到端训练的CNN+GRU架构在tight设置下优于使用预训练冻结特征+Transformer的方案

## 亮点与洞察

1. **高效即是王道**：6.46M参数击败70M参数的模型，说明适当设计的轻量网络+端到端训练>堆叠大模型
2. **SoftIC损失的通用性**：将对比学习扩展到支持soft label，对任何使用mixup的不平衡分类任务都有价值
3. **ASTRM的三阶段设计**：空间→局部时序→全局时序的级联精炼，充分利用了不同维度的信息

## 局限与展望

1. 处理128帧的窗口大小可能不足以覆盖某些需要更长上下文的事件
2. SoftIC需要额外的记忆库存储，增加内存开销
3. 仅在体育视频上验证，未测试监控视频等其他事件检测场景
4. 可以探索将ASTRM插入更大的backbone以进一步提升性能

## 相关工作与启发

- **E2E-Spot**：端到端事件定位的先驱，本文在其架构基础上大幅增强
- **COMEDIAN**：通过3阶段训练+大backbone达到高性能，但效率低
- **Instance Contrastive Loss**：SoftIC直接扩展自IC Loss以适配mixup场景
- 启发：类别不平衡问题的对比学习解法（SoftIC）可以迁移到其他长尾分布任务

## 评分

⭐⭐⭐⭐ — 高效的端到端设计以极少参数达到SOTA，SoftIC损失对不平衡问题的解决方案优雅实用。ASTRM虽不算全新设计但组合有效。在tight设置下的大幅提升说明少数类改进是关键。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Beyond Length: Quantifying Long-Range Information for Long-Context LLM Pretraining Data](../../ICLR2026/llm_pretraining/beyond_length_quantifying_long-range_information_for_long-context_llm_pretrainin.md)
- [\[ICCV 2025\] Synchronization of Multiple Videos](../../ICCV2025/llm_pretraining/synchronization_of_multiple_videos.md)
- [\[NeurIPS 2025\] CLIMB: Class-Imbalanced Learning Benchmark on Tabular Data](../../NeurIPS2025/llm_pretraining/climb_class-imbalanced_learning_benchmark_on_tabular_data.md)
- [\[NeurIPS 2025\] Deep Compositional Phase Diffusion for Long Motion Sequence Generation](../../NeurIPS2025/llm_pretraining/deep_compositional_phase_diffusion_for_long_motion_sequence_generation.md)
- [\[ACL 2025\] Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset](../../ACL2025/llm_pretraining/nemotron_cc_pretraining_data.md)

</div>

<!-- RELATED:END -->
