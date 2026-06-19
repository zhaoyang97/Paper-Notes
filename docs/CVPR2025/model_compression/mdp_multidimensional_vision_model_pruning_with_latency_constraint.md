---
title: >-
  [论文解读] MDP: Multidimensional Vision Model Pruning with Latency Constraint
description: >-
  [CVPR 2025][模型压缩][结构化剪枝] MDP 提出多维度剪枝范式，将通道、注意力头、Q/K/V、嵌入维度和整个 block 等不同粒度的结构化剪枝统一建模为混合整数非线性规划(MINLP)问题，在严格延迟约束下联合求解全局最优剪枝结构，在高剪枝比下大幅超越已有方法。 领域现状：结构化剪枝是部署大模型的主流手段…
tags:
  - "CVPR 2025"
  - "模型压缩"
  - "结构化剪枝"
  - "延迟约束"
  - "混合整数非线性规划"
  - "多维度剪枝"
  - "硬件感知"
---

# MDP: Multidimensional Vision Model Pruning with Latency Constraint

**会议**: CVPR 2025  
**arXiv**: [2504.02168](https://arxiv.org/abs/2504.02168)  
**代码**: 无（论文提及 acceptance 后开源）  
**领域**: 模型压缩  
**关键词**: 结构化剪枝, 延迟约束, 混合整数非线性规划, 多维度剪枝, 硬件感知

## 一句话总结
MDP 提出多维度剪枝范式，将通道、注意力头、Q/K/V、嵌入维度和整个 block 等不同粒度的结构化剪枝统一建模为混合整数非线性规划(MINLP)问题，在严格延迟约束下联合求解全局最优剪枝结构，在高剪枝比下大幅超越已有方法。

## 研究背景与动机
**领域现状**：结构化剪枝是部署大模型的主流手段，通过移除通道、注意力头等结构化参数来减少计算量。现有方法通常在单一粒度（如通道级或层级）上操作。

**现有痛点**：(1) 大多数方法只能在中等剪枝比（30-50%）下有效，因为只剪通道或注意力头难以达到 70-90% 的高剪枝比；(2) 现有延迟感知方法（如 HALP）使用过度简化的线性延迟模型，只考虑输出通道数，忽略输入通道变化对延迟的影响，对 Transformer 完全不适用（Transformer 有 5 个互相关联的可剪维度）。

**核心矛盾**：高剪枝比需要同时在多种粒度上剪枝（通道+block），但多粒度剪枝的搜索空间极大；延迟对多个维度非线性依赖，简单线性模型无法准确建模。

**本文目标** (1) 支持跨多种粒度（通道、头、Q/K/V、嵌入、block）的联合剪枝；(2) 建立准确的多维度延迟约束模型；(3) 在延迟预算下找到全局最优剪枝结构。

**切入角度**：作者认识到问题的关键在于"多维度"——无论是实现高剪枝比需要的多粒度组合，还是准确建模延迟需要的多维度交互。将多维度剪枝变量和 block 保留/移除决策统一编码为数学优化问题，利用成熟的 MINLP 求解器（Pyomo + MindtPy）一次性找到全局最优解。

**核心 idea**：用 MINLP 统一建模多粒度剪枝和多维延迟约束，在延迟预算下直接求解全局最优剪枝结构。

## 方法详解

### 整体框架
给定预训练模型和目标延迟预算 Ψ，MDP 首先识别所有可剪维度并编码为独热变量 ω，然后构建重要性目标函数和延迟约束函数，接着将 block 移除决策建模为二值变量 κ，最后将问题统一为 MINLP 求解。求解后提取剪枝子网络并微调恢复精度。

### 关键设计

1. **多维度变量编码与重要性目标函数**:

    - 功能：用统一的数学形式描述所有可剪维度的配置及其重要性
    - 核心思路：对每个可剪维度用独热向量 $\omega_b^i$ 编码，"热"位的索引表示该维度保留的参数数量。CNN 中每层有一个可剪维度（输出通道数），Transformer 每个 block 有 5 个可剪维度（嵌入、头数、Q/K、V、MLP）。重要性向量 $\vec{I_b^i}$ 的第 j 个元素为保留 Top-j 个参数的累积重要性分数（CNN 用 Taylor 分数，Transformer 用 Hessian 分数）。总重要性函数为所有维度的线性组合：$\mathcal{I}(\omega) = \sum_b \sum_i \omega_b^{i\top} \cdot \vec{I_b^i}$
    - 设计动机：不同于传统方法给单个通道打分再排序，MDP 评估的是"保留 j 个通道的总重要性"，这使得重要性可以与延迟约束放在同一个优化框架中联合求解

2. **多维度延迟约束建模**:

    - 功能：准确建模延迟随多个可剪维度同时变化的关系
    - 核心思路：为每个 block 预计算延迟查找表（LUT），表的维度等于可剪维度数。直接计算外积链开销太大，因此进行模型特定的分解：CNN 分解为每层单独的 2D LUT（输入通道 × 输出通道）；Transformer 分解为 QK（嵌入×头×qk 的 3D LUT）、V-Proj（嵌入×头×v 的 3D LUT）和 MLP（嵌入×mlp 的 2D LUT）三部分。总延迟为所有 block 贡献之和
    - 设计动机：HALP 等之前方法只用与输出通道数线性的延迟模型，忽略了输入通道的影响。对于 Transformer 有 5 个维度交互影响延迟，线性模型完全不可行。分解后的 LUT 既保证精度又可高效计算

3. **Block 级移除与 MINLP 求解**:

    - 功能：支持整个 block 的移除决策，并与细粒度剪枝联合优化
    - 核心思路：为每个 block 引入二值决策变量 κ_b。当 κ_b=0 时，该 block 被完全移除（依靠残差连接保持信息流），其重要性和延迟贡献归零。最终优化问题为最大化 $\mathcal{I}(\omega, \kappa)$ s.t. $\mathcal{C}'(\omega, \kappa) \leq \Psi$，其中 ω 是独热向量（整数），κ 是二值（整数），重要性和延迟函数含多项式项（非线性），构成 MINLP。使用 Pyomo + MindtPy 配合 Feasibility Pump 求解
    - 设计动机：高剪枝比下仅靠细粒度剪枝不够，需要移除整个 block 才能大幅减少延迟。但 block 移除和通道剪枝需要联合优化才能找到最佳平衡——这正是 MINLP 的优势

### 损失函数 / 训练策略
MINLP 求解得到最优结构后，按照 Top-j（重要性排序）保留对应参数，提取剪枝后的子网络 Θ̂。然后在标准训练集上微调 E 个 epoch 恢复精度。整个流程无需额外模块或训练阶段。

## 实验关键数据

### 主实验

| 模型/数据集 | 方法 | Top-1(%) | 加速比 | 说明 |
|------------|------|---------|--------|------|
| ResNet50/ImageNet | HALP-85% | 68.1 | 3.90× | 之前SOTA |
| ResNet50/ImageNet | **MDP-85%** | **70.0** | **5.21×** | +1.9% Top-1, +28%速度 |
| ResNet50/ImageNet | HALP-70% | 74.5 | 2.55× | 之前SOTA |
| ResNet50/ImageNet | **MDP-70%** | **74.8** | **3.06×** | +0.3% Top-1, +20%速度 |
| DeiT-B/ImageNet | Isomorphic-S | 82.41 | 2.56× | 最新Transformer剪枝 |
| DeiT-B/ImageNet | **MDP-39%** | **82.65** | **2.73×** | +0.24% Top-1, +7%速度 |
| StreamPETR/NuScenes | Dense | 0.449 mAP | 1.0× | 未剪枝基线 |
| StreamPETR/NuScenes | **MDP-45%** | **0.451 mAP** | **1.18×** | 剪枝后mAP反超Dense |

### 消融实验

| 配置 | 说明 | 优势 |
|------|------|------|
| 仅通道剪枝 | HALP等传统方法 | 中等剪枝比有效 |
| 通道+Block | MDP 多粒度 | 高剪枝比下精度远优于仅通道 |
| 线性延迟模型 | HALP 的 1D LUT | 不精确，Transformer不适用 |
| 多维延迟模型 | MDP 的分解 LUT | 精确建模，支持CNN+Transformer |

### 关键发现
- MDP 的优势在高剪枝比下尤为显著：85% 剪枝率下比 HALP 多 28% 的速度提升和 1.9% 的 Top-1 改进
- Block 移除对高剪枝比至关重要——仅靠通道剪枝无法在 >70% 比率下保持合理精度
- 在 3D 检测任务中（StreamPETR），MDP 剪枝 45% 后 mAP 反而超过原始模型，说明去除冗余结构可能有正则化效果
- 首次实现了 Transformer 的严格延迟约束剪枝，之前方法（如 NViT）只用延迟作为软正则项

## 亮点与洞察
- **用运筹学的方法解决深度学习问题**：将网络剪枝形式化为 MINLP，利用成熟的数值优化工具（Pyomo/MindtPy）求解，而非贪心搜索或启发式方法。MINLP 保证了全局最优性，这是贪心方法无法做到的
- **多维延迟建模的通用性**：LUT 分解思路可以轻松扩展到新的网络架构——只需识别可剪维度并预计算对应的 LUT 即可。这让 MDP 成为一个真正通用的剪枝框架
- **Block 移除的优雅处理**：通过二值变量 κ 将 block 移除自然地纳入优化框架，无需额外的线性探针或两阶段流程

## 局限与展望
- MINLP 求解器的计算开销未明确报告，对于超大模型（如 LLaMA-70B）MINLP 的求解时间可能成为瓶颈
- LUT 需要在目标硬件上预先测量大量延迟数据点，部署到新硬件需要重新构建
- 当前仅验证了视觉模型（CNN + ViT），未扩展到 NLP（如 LLM 剪枝）或多模态模型
- 微调策略较简单（标准训练），可以探索知识蒸馏等更高效的恢复方法

## 相关工作与启发
- **vs HALP**: HALP 也做延迟约束剪枝，但只用 1D 线性延迟模型（仅考虑输出通道），且不支持 block 移除。MDP 用多维 LUT + MINLP 同时优化多粒度，在所有比较中均超越 HALP
- **vs NViT**: NViT 对 Transformer 剪枝用 Hessian 重要性全局打分，但只将延迟作为软正则项，无法保证满足延迟约束。MDP 通过硬约束严格满足延迟预算
- **vs Isomorphic Pruning**: 最新的 Transformer 剪枝方法，MDP 在相同模型上进一步提速 37% 且精度更高

## 评分
- 新颖性: ⭐⭐⭐⭐ 将剪枝问题形式化为 MINLP 的视角新颖，多维延迟建模填补了Transformer延迟剪枝的空白
- 实验充分度: ⭐⭐⭐⭐⭐ 涵盖 CNN、Transformer、3D 检测多种架构和任务，对比方法全面
- 写作质量: ⭐⭐⭐⭐ 数学推导清晰严谨，但符号较多读起来有一定门槛
- 价值: ⭐⭐⭐⭐ 对高剪枝比场景和 Transformer 延迟剪枝有重要实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] HiAP: A Multi-Granular Stochastic Auto-Pruning Framework for Vision Transformers](hiap_a_multi-granular_stochastic_auto-pruning_framework_for_vision_transformers.md)
- [\[ECCV 2024\] Isomorphic Pruning for Vision Models](../../ECCV2024/model_compression/isomorphic_pruning_for_vision_models.md)
- [\[CVPR 2025\] BHViT: Binarized Hybrid Vision Transformer](bhvit_binarized_hybrid_vision_transformer.md)
- [\[ACL 2025\] BeamLoRA: Beam-Constraint Low-Rank Adaptation](../../ACL2025/model_compression/beamlora_beam_constraint_lora.md)
- [\[NeurIPS 2025\] Vision-centric Token Compression in Large Language Model](../../NeurIPS2025/model_compression/vision-centric_token_compression_in_large_language_model.md)

</div>

<!-- RELATED:END -->
