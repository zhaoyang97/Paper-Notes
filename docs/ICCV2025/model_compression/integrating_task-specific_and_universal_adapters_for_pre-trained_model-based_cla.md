---
title: >-
  [论文解读] Integrating Task-Specific and Universal Adapters for Pre-Trained Model-based Class-Incremental Learning
description: >-
  [ICCV 2025][模型压缩][类增量学习] 提出 TUNA 方法，通过为每个增量任务训练正交的 task-specific adapter，并将它们融合为一个 universal adapter，结合基于熵的 adapter 选择机制和双 adapter 集成推理策略，在无 exemplar 的 PTM-based CIL 中实现 SOTA。
tags:
  - "ICCV 2025"
  - "模型压缩"
  - "类增量学习"
  - "适配器"
  - "预训练模型"
  - "模型融合"
  - "持续学习"
---

# Integrating Task-Specific and Universal Adapters for Pre-Trained Model-based Class-Incremental Learning

**会议**: ICCV 2025  
**arXiv**: [2508.08165](https://arxiv.org/abs/2508.08165)  
**代码**: [https://github.com/LAMDA-CL/ICCV2025-TUNA](https://github.com/LAMDA-CL/ICCV2025-TUNA)  
**领域**: 模型压缩  
**关键词**: 类增量学习, 适配器, 预训练模型, 模型融合, 持续学习

## 一句话总结

提出 TUNA 方法，通过为每个增量任务训练正交的 task-specific adapter，并将它们融合为一个 universal adapter，结合基于熵的 adapter 选择机制和双 adapter 集成推理策略，在无 exemplar 的 PTM-based CIL 中实现 SOTA。

## 研究背景与动机

类增量学习（CIL）要求模型不断学习新类别而不遗忘旧类别。在预训练模型（PTM）时代，主流做法是冻结 PTM 权重，通过轻量模块（如 prompt、adapter）进行增量适配。然而现有方法存在两大问题：

**模块选择不准确**：L2P 等方法依赖 key-query 匹配来选择 task-specific prompt，匹配过程脆弱，容易选错模块导致性能下降

**忽略跨任务共享知识**：现有方法只关注 task-specific 知识，无法区分跨任务的高度相似类别（例如不同任务中学到的猫和狗可能外观相似）

这两个问题导致推理时既可能选错 adapter，又无法利用通用知识来辅助分类。

## 方法详解

### 整体框架

TUNA 包含三个核心组件：（1）正交约束的 task-specific adapter 训练；（2）多阶段 adapter 融合构建 universal adapter；（3）基于预测不确定性的 adapter 选择与双 adapter 集成推理。

### 关键设计

1. **正交 Task-Specific Adapter 训练**：对每个增量任务 $t$，初始化新 adapter $\mathcal{A}_t$（bottleneck 结构：$W_{down} \in \mathbb{R}^{d \times r}$, ReLU, $W_{up} \in \mathbb{R}^{r \times d}$），通过残差连接注入 MLP 层。为防止任务间特征冗余，在 up-projection 权重上施加正交约束：
    $\mathcal{L}_{orth} = \sum_{i=1}^{t-1} \|W_{up}^t \cdot {W_{up}^i}^\top\|_1$
   仅对 up-projection 权重施加正交（消融实验表明同时约束 down-projection 反而有害），因为 up-projection 负责将特征投影到高维空间、编码 task-specific 信息。

2. **多阶段 Adapter 融合（Universal Adapter）**：训练完 $t$ 个任务后，将所有 adapter 权重展平为向量 $\{\mathbf{v}^1, \ldots, \mathbf{v}^t\}$，通过两步操作融合：

    - **符号聚合**：对每个参数位置取所有任务向量的符号投票 $\mathbf{s}^{uni} = \text{sgn}(\sum_i \mathbf{v}^i)$
    - **幅值选择**：对每个参数位置，在保持共识符号方向的前提下取最大绝对值
    - 最终 universal task vector $\mathbf{v}^{uni} = \epsilon^{uni} \odot \mathbf{s}^{uni}$，reshape 回 adapter 形状
   
   该融合策略基于模型合并技术的两个操作：符号求和作为投票系统保持主导特征方向，最大绝对值选择抑制噪声并保留具有判别力的特征幅值。

3. **基于熵的 Adapter 选择与双 Adapter 集成**：

    - **选择机制**：通过实验发现低熵（高置信度）与高准确率强相关。推理时计算每个 task-specific adapter 的预测熵，选择熵最小的 adapter $\mathcal{A}^*$
    $\mathcal{A}^* = \arg\min_{\mathcal{A}_i} \left(-\sum_c f_c(\mathbf{x}; \mathcal{A}_i) \log f_c(\mathbf{x}; \mathcal{A}_i)\right)$
    - **集成推理**：将选中的 task-specific adapter 与 universal adapter 的预测概率相加：$y^* = \arg\max_y (f_y(\mathbf{x}; \mathcal{A}^*) + f_y(\mathbf{x}; \mathcal{A}_{uni}))$

### 损失函数 / 训练策略

- 分类损失：标准交叉熵 $\mathcal{L}_{cls}$
- 正交损失：$\mathcal{L}_{orth}$ 仅约束 up-projection 权重
- 总损失：$\mathcal{L} = \mathcal{L}_{cls} + \lambda \mathcal{L}_{orth}$，$\lambda$ 初始化为 1e-3 并指数衰减
- 每个任务训练完后计算类别均值和方差，用于后续任务中 replay 以校准分类器
- 训练细节：SGD + momentum，lr=0.01 cosine annealing，20 epochs，batch size=48，adapter projection dim $r$=16

## 实验关键数据

### 主实验

| 方法 | CIFAR $\bar{\mathcal{A}}$ | CIFAR $\mathcal{A}_B$ | IN-R $\bar{\mathcal{A}}$ | IN-R $\mathcal{A}_B$ | IN-A $\bar{\mathcal{A}}$ | IN-A $\mathcal{A}_B$ | ObjectNet $\bar{\mathcal{A}}$ | ObjectNet $\mathcal{A}_B$ |
|------|------|------|------|------|------|------|------|------|
| L2P | 85.94 | 79.93 | 75.46 | 69.77 | 49.39 | 41.71 | 63.78 | 52.19 |
| CODA-Prompt | 89.11 | 81.96 | 77.97 | 72.27 | 53.54 | 42.73 | 66.07 | 53.29 |
| SLCA | 92.49 | 88.55 | 81.17 | 77.00 | 68.66 | 58.74 | 72.55 | 61.30 |
| RanPAC | 94.00 | 90.62 | 82.98 | 77.94 | 69.32 | 61.82 | 72.76 | 62.02 |
| EASE | 91.51 | 85.80 | 81.74 | 76.17 | 65.34 | 55.04 | 70.84 | 57.86 |
| **TUNA** | **94.44** | **90.74** | **84.22** | **79.42** | **73.78** | **64.78** | **76.46** | **66.32** |

ViT-B/16-IN21K backbone, B0 Inc 设置。TUNA 在所有四个数据集上均为 SOTA，尤其在挑战性较大的 ImageNet-A 和 ObjectNet 上提升显著。

### 消融实验

| 配置 | ImageNet-A B0 Inc20 $\mathcal{A}_B$ |
|------|------|
| Baseline（多 adapter max logit） | ~55 |
| + entropy-based adapter selection | ~59 |
| + orthogonal loss | ~62 |
| + universal adapter（完整 TUNA） | ~65 |

每个组件均有正向贡献。三种推理策略对比（ImageNet-A）：
- Variation-1（TUNA 完整策略）：最优
- Variation-2（仅 entropy 选择 task-specific）：缺乏共享知识
- Variation-3（仅 universal adapter）：缺乏 task-specific 细粒度

正交损失消融（ObjectNet B0 Inc20）：仅约束 up-projection（Variation-1）效果最好，同时约束 up+down 会过度刚性导致欠拟合。

### 关键发现

- TUNA 在无 exemplar 设置下超越使用 20/class exemplar 的传统 CIL 方法（iCaRL、DER、FOSTER 等）
- 超参数（$r \in \{8,16,32,64,128\}$, $\lambda \in \{0.001,...,0.1\}$）对性能影响较小，鲁棒性好
- 可视化表明：task-specific adapter 对域内类别区分力强但跨域混淆（如金毛犬→狮子），universal adapter 能捕获跨任务共享特征纠正此类错误

## 亮点与洞察

- **adapter 融合策略简洁有效**：基于符号投票+最大绝对值选择的融合方式，无需额外训练即可从多个 adapter 中提取通用知识
- **熵作为 adapter 选择代理**：相比 key-query 匹配更稳定可靠，实验验证了低熵↔高准确率的强相关性
- **双 adapter 互补推理**：task-specific 提供细粒度区分，universal 提供跨任务泛化能力
- **正交约束设计精细**：仅约束 up-projection 这一选择体现了对 adapter 结构功能的深入理解

## 局限与展望

- **推理效率**：选择最优 adapter 需要多次前向传播（等于任务数量），计算开销随任务增多线性增长
- universal adapter 融合是静态的（每次新任务后重新融合），未探索在线融合或动态调整
- 未探索与其他 PEFT 方法（如 LoRA、VPT）的深度结合，虽然论文提到框架兼容

## 相关工作与启发

- 与模型合并（model merging）方法有天然联系，符号投票策略类似于 TIES-Merging
- 启发：adapter 融合的思路可推广到多任务学习、联邦学习等需要聚合多个模型的场景
- 与 EASE（串联多 adapter 特征）的关键区别：TUNA 融合为一个 universal adapter 而非简单拼接

## 评分

- **新颖性**: ⭐⭐⭐⭐ 双 adapter 推理策略和 adapter 融合方法设计巧妙
- **实验充分度**: ⭐⭐⭐⭐⭐ 4个数据集，多种 baseline，详细消融和分析
- **写作质量**: ⭐⭐⭐⭐ 逻辑清晰，动机明确，实验展示完整
- **价值**: ⭐⭐⭐⭐ 为 PTM-based CIL 提供了有效的 adapter 管理方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Mixture of Noise for Pre-Trained Model-Based Class-Incremental Learning](../../NeurIPS2025/model_compression/mixture_of_noise_for_pre-trained_model-based_class-incremental_learning.md)
- [\[CVPR 2026\] QKD: Quantum-Gated Task-interaction Knowledge Distillation for Class-Incremental Learning](../../CVPR2026/model_compression/qkd_quantum_gated_incremental_learning.md)
- [\[ICCV 2025\] Achieving More with Less: Additive Prompt Tuning for Rehearsal-Free Class-Incremental Learning](achieving_more_with_less_additive_prompt_tuning_for_rehearsal-free_class-increme.md)
- [\[AAAI 2026\] Compensating Distribution Drifts in Class-incremental Learning of Pre-trained Vision Transformers](../../AAAI2026/model_compression/compensating_distribution_drifts_in_class-incremental_learning_of_pre-trained_vi.md)
- [\[ICCV 2025\] Efficient Adaptation of Pre-Trained Vision Transformer Underpinned by Approximation Theory](efficient_adaptation_of_pre-trained_vision_transformer_underpinned_by_approximat.md)

</div>

<!-- RELATED:END -->
