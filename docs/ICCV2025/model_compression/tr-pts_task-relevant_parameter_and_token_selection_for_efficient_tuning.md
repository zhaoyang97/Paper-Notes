---
title: >-
  [论文解读] TR-PTS: Task-Relevant Parameter and Token Selection for Efficient Tuning
description: >-
  [ICCV 2025][模型压缩][PEFT] 提出 TR-PTS 框架，通过 Fisher 信息矩阵进行任务驱动的逐层参数选择，同时利用 CLS 注意力分数动态筛选/合并 token，在仅微调 0.34%-0.60% 参数的情况下超越全量微调 3.40%（FGVC）和 10.35%（VTAB）。 大规模预训练 ViT 在下…
tags:
  - "ICCV 2025"
  - "模型压缩"
  - "PEFT"
  - "Transformer"
  - "Token Selection"
  - "Fisher Information Matrix"
  - "Parameter Selection"
---

# TR-PTS: Task-Relevant Parameter and Token Selection for Efficient Tuning

**会议**: ICCV 2025  
**arXiv**: [2507.22872](https://arxiv.org/abs/2507.22872)  
**代码**: [https://github.com/synbol/TR-PTS](https://github.com/synbol/TR-PTS)  
**领域**: 模型压缩 / 参数高效微调  
**关键词**: PEFT, Vision Transformer, Token Selection, Fisher Information Matrix, Parameter Selection

## 一句话总结

提出 TR-PTS 框架，通过 Fisher 信息矩阵进行任务驱动的逐层参数选择，同时利用 CLS 注意力分数动态筛选/合并 token，在仅微调 0.34%-0.60% 参数的情况下超越全量微调 3.40%（FGVC）和 10.35%（VTAB）。

## 研究背景与动机

大规模预训练 ViT 在下游视觉任务上表现优异，但全量微调代价高昂。现有参数高效微调（PEFT）方法存在三大问题：

**推理开销**：VPT 等方法引入额外可学习模块，导致推理时计算增加

**缺乏任务感知**：大多数方法对不同任务使用统一的调优策略，忽略不同层和参数对特定任务的重要性差异

**参数与 token 优化脱节**：已有工作将参数选择和 token 处理分开考虑，但 token 的信息量本身高度依赖于任务

作者观察到：**不同任务依赖不同的 token 子集进行最终预测**（图 2 可视化证实了这一点），因此需要一个统一框架同时完成任务相关的参数选择和 token 精炼。

## 方法详解

### 整体框架

TR-PTS 包含两个协同模块：Task-Relevant Parameter Selection 和 Task-Relevant Token Selection，二者通过联合优化实现互相增强。

### 关键设计

1. **Task-Relevant Parameter Selection（基于 FIM 的逐层参数分配）**:

    - 利用 Fisher 信息矩阵（FIM）量化每个参数对任务的敏感度。FIM 通过交叉熵损失的梯度平方近似：$\mathcal{F}(\theta) \approx \mathbb{E}_{(x,y)\sim D}\left[\left(\frac{\partial \mathcal{L}_{CE}}{\partial \theta}\right)^2\right]$
    - 选出 top-M% 高 FIM 值参数后，计算每层的贡献权重 $w_l$，再归一化得到每层神经元的可训练连接数 $C_l = \max(1, \frac{w_l}{\min(w)} \cdot C_{\min})$
    - 每层内进一步按 FIM 分数选出每个神经元的 top-$C_l$ 个连接作为可训练参数
    - 设计动机：相比 GPS 使用的梯度幅值，FIM 不受优化噪声影响，能更准确反映参数对任务的重要性；逐层分配保证每层至少保留一个活跃连接，避免网络局部失活

2. **Task-Relevant Token Selection（基于 CLS 注意力的动态 token 筛选与合并）**:

    - 利用 CLS token 对各 image token 的注意力分数 $a_i$ 衡量 token 重要性
    - 按选择率 $\rho$ 保留 top-$\lfloor\rho N\rfloor$ 个高注意力 token
    - 未被选中的 token 并不丢弃，而是通过注意力加权平均合并为一个聚合 token：$x_{\text{merged}} = \frac{\sum_{i\in\mathcal{I}} a_i x_i}{\sum_{i\in\mathcal{I}} a_i}$
    - 合并后的精炼序列为 $X_{\text{refined}} = \{x_{\text{CLS}}, X_{\text{selected}}, x_{\text{merged}}\}$
    - 设计动机：结合了 token pruning（减少计算量）和 token merging（保留全局信息）的优势

3. **参数-Token 协同选择策略**:

    - 关键发现：参数稀疏的层倾向于编码信息量较少的 token
    - 因此将 token reduction 优先应用于参数稀疏层（"sparse insertion"策略）
    - 用二值掩码 $M$ 控制梯度更新：$\Theta^{(t+1)} = \Theta^{(t)} - \eta(M \odot \nabla_\Theta \mathcal{L})$

### 损失函数 / 训练策略

- 使用标准交叉熵损失训练
- Adam 优化器 + cosine 学习率衰减，训练 100 个 epoch
- 骨干网络：ViT-B/16，预训练于 ImageNet-21K

## 实验关键数据

### 主实验

**VTAB-1k 基准（19 个视觉分类任务）**:

| 方法 | Natural 均值 | Specialized 均值 | Structured 均值 | 总均值 | 参数量(%) |
|---|---|---|---|---|---|
| Full Fine-tuning | - | - | - | 65.57 | 100.00 |
| LoRA | - | - | - | 72.63 | 0.90 |
| GPS | - | - | - | 75.18 | 0.25 |
| **TR-PTS** | - | - | - | **75.92** | **0.34** |

**FGVC 基准（5 个细粒度分类数据集）**:

| 方法 | CUB-200 | NABirds | Flowers | Dogs | Cars | 均值 | 参数量(%) |
|---|---|---|---|---|---|---|---|
| Full | 87.3 | 82.7 | 98.8 | 89.4 | 84.5 | 88.54 | 100.00 |
| GPS | 89.9 | 86.7 | 99.7 | 92.2 | 90.4 | 91.78 | 0.77 |
| **TR-PTS** | **90.0** | **87.1** | 99.6 | **92.4** | **90.6** | **91.94** | **0.60** |

### 消融实验

**各组件贡献（VTAB-1k 子集）**:

| Token Selection | Parameter Selection | dSprites/loc | Flower102 | Sun397 |
|---|---|---|---|---|
| ✗ | ✗ | 12.5 | 97.0 | 51.0 |
| ✓ | ✗ | 14.8 | 98.8 | 51.2 |
| ✗ | ✓ | 85.1 | 99.4 | 54.2 |
| ✓ | ✓ | **87.7** | **99.5** | **54.5** |

**Token 选择位置策略对比**:

| 策略 | 选择率 | Sun397 | Flower102 | Loc | Camelyon |
|---|---|---|---|---|---|
| Dense | 0.95 | 53.5 | 99.3 | 85.2 | 87.3 |
| Random | 0.95 | 54.0 | 99.3 | 85.9 | 87.9 |
| **Sparse** | **0.95** | **54.5** | **99.4** | **87.7** | **88.1** |

### 关键发现

- TR-PTS 的 FLOPs 和推理内存消耗在所有 PEFT 方法中最低
- 不同任务的 FIM 关键参数层分布差异显著：Flower102 集中在 Block 8/10，Sun397 集中在 Block 0，Patch/Camelyon 分布均匀
- 不同任务间的参数选择集重叠率低（如 Sun397 vs Patch/Camelyon 仅 0.17），验证了任务自适应选择的必要性
- Token 可视化显示：浅层保留较多 token，深层逐渐聚焦在前景目标上

## 亮点与洞察

- **参数与 token 联合优化**的思路新颖，发现了"参数稀疏层→token 冗余度高"的内在关联，并据此设计协同策略
- 实验覆盖 24 个数据集，结果全面且强一致
- 不引入任何额外参数，训练和推理阶段均无额外开销
- FIM 比梯度幅值更稳定地反映参数重要性

## 局限与展望

- 仅在分类任务上验证，尚未拓展到检测、分割等密集预测任务
- Token 选择率 $\rho$ 和最小连接数 $C_{\min}$ 为超参，需手动调节
- FIM 计算需要前向+反向传播，增加了初始化阶段的计算成本
- 未探索跨层自适应 token 选择率（当前各层使用固定 $\rho$）

## 相关工作与启发

- GPS 是最接近的前身工作，同样做参数选择但使用梯度幅值且无 token 压缩
- ToMe（Token Merging）提出了基于相似度的 token 合并，但不考虑任务相关性
- 本文的"稀疏层做 token 压缩"发现可启发未来工作：在其他模型压缩场景中，寻找计算稀疏区域进行更激进的资源节省

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 参数与 token 联合选择框架、FIM 逐层分配策略有创意
- **实验充分度**: ⭐⭐⭐⭐⭐ — 24 个数据集 + 多维度消融 + 计算开销分析
- **写作质量**: ⭐⭐⭐⭐ — 层次清晰，图表丰富
- **价值**: ⭐⭐⭐⭐ — 实用性强，PEFT 领域有参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Generalized Tensor-based Parameter-Efficient Fine-Tuning via Lie Group Transformations](generalized_tensor-based_parameter-efficient_fine-tuning_via_lie_group_transform.md)
- [\[CVPR 2025\] Faster Parameter-Efficient Tuning with Token Redundancy Reduction (FPET)](../../CVPR2025/model_compression/faster_parameter-efficient_tuning_with_token_redundancy_reduction.md)
- [\[CVPR 2025\] Expert Pyramid Tuning: Efficient Parameter Fine-Tuning for Expertise-Driven Task Allocation](../../CVPR2025/model_compression/expert_pyramid_tuning_efficient_parameter_fine-tuning_for_expertise-driven_task_.md)
- [\[ACL 2025\] C3A: Parameter-Efficient Fine-Tuning via Circular Convolution](../../ACL2025/model_compression/parameter-efficient_fine-tuning_via_circular_convolution.md)
- [\[CVPR 2026\] FAAR: Efficient Frequency-Aware Multi-Task Fine-Tuning via Automatic Rank Selection](../../CVPR2026/model_compression/faar_efficient_frequency-aware_multi-task_fine-tuning_via_automatic_rank_selecti.md)

</div>

<!-- RELATED:END -->
