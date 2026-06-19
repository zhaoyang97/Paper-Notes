---
title: >-
  [论文解读] One-stage Prompt-based Continual Learning
description: >-
  [ECCV 2024][AI安全][提示学习] 提出 OS-Prompt 框架，通过直接使用 ViT 中间层 token embedding 作为 prompt query（而非额外的 query ViT 前向传播），将 Prompt-based Continual Learning 的计算成本降低约 50%，并通过 Query-Pool Regularization (QR) loss 补偿表征能力损失，在 CIFAR-100、ImageNet-R、DomainNet 上超越 CodaPrompt 约 1.4%。
tags:
  - "ECCV 2024"
  - "AI安全"
  - "提示学习"
  - "Transformer"
  - "计算效率"
  - "Query-Pool Regularization"
  - "类增量学习"
---

# One-stage Prompt-based Continual Learning

**会议**: ECCV 2024  
**arXiv**: [2402.16189](https://arxiv.org/abs/2402.16189)  
**代码**: 无  
**领域**: 持续学习 / 高效学习  
**关键词**: Prompt-based Continual Learning, Vision Transformer, 计算效率, Query-Pool Regularization, 类增量学习

## 一句话总结

提出 OS-Prompt 框架，通过直接使用 ViT 中间层 token embedding 作为 prompt query（而非额外的 query ViT 前向传播），将 Prompt-based Continual Learning 的计算成本降低约 50%，并通过 Query-Pool Regularization (QR) loss 补偿表征能力损失，在 CIFAR-100、ImageNet-R、DomainNet 上超越 CodaPrompt 约 1.4%。

## 研究背景与动机

Prompt-based Continual Learning (PCL) 是当前持续学习的 SOTA 方案，通过在预训练 ViT 上训练可学习的 prompt token 来避免灾难性遗忘，无需存储历史数据（隐私友好、内存开销小）。然而，现有 PCL 方法（如 L2P、DualPrompt、CodaPrompt）都需要**两阶段 ViT 前向传播**：第一阶段用一个冻结的 query ViT 生成 prompt query 来选择 prompt pool 中的 prompt；第二阶段用骨干 ViT 将选中的 prompt 与图像 token 融合进行分类。这种双 ViT 架构导致训练和推理的计算成本翻倍（约 35 GFLOPs 推理），严重限制了在资源受限设备上的部署。

**核心矛盾**：PCL 的高精度与双 ViT 带来的高计算成本之间的矛盾。

**切入角度**：作者观察到 prompt 持续学习过程中，ViT 早期层的 token embedding 变化极小（cosine distance ≤ 0.1），这意味着可以直接用中间层的 embedding 作为 prompt query，从而省去整个 query ViT。

**核心 idea**：一阶段 PCL，用中间层 [CLS] token 替代额外 query ViT，推理 GFLOPs 减半，精度损失 ≤ 1%。

## 方法详解

### 整体框架

OS-Prompt 框架只需**一个 ViT 前向传播**。图像输入骨干 ViT 后，在第 1-5 层每层直接使用当前层的 [CLS] token embedding 作为 prompt query，与 prompt pool 中的 key 计算相似度，生成该层的 prompt token，然后通过 prefix-tuning 方式注入 self-attention。最终由分类头输出预测。

OS-Prompt++ 在训练时额外引入一个冻结的 reference ViT 提取最终层 [CLS] token，用于 QR loss 正则化，但推理时不使用 reference ViT。

### 关键设计

1. **中间层 Token Embedding 作为 Prompt Query**:

    - 功能：用骨干 ViT 第 $l$ 层的 [CLS] token $q_l = x_{l_{[CLS]}}$ 直接作为 prompt query，替代原先需要额外 query ViT 前向传播的 $q = Q(x)_{[CLS]}$
    - 核心思路：由于 prompt 仅添加在第 1-5 层且骨干 ViT 权重冻结，早期层 token embedding 在持续学习过程中变化极小
    - 设计动机：作者通过实验测量了 CIFAR-100 10-task 设置下各层 token embedding 的 cosine distance 变化——第 1-5 层的距离始终 ≤ 0.1，而最后一层 ≥ 0.1 且随任务增多持续增大。这证明早期层的 embedding 足够稳定，适合作为 query
    - 与之前方法的区别：之前方法用独立冻结 ViT 保证 query 一致性，本文直接用骨干 ViT 内部 embedding，省去一半计算量

2. **逐层 Prompt 生成（CodaPrompt 式加权求和）**:

    - 功能：对第 $l$ 层，计算 query $q_l$ 与 prompt pool keys $\{k_l^1, ..., k_l^M\}$ 的 cosine similarity $\gamma(\cdot)$，加权求和得到 prompt $\phi_l = \sum_m \gamma(q_l, k_l^m) p_l^m$
    - 核心思路：沿用 CodaPrompt 的 soft matching 策略实现端到端训练
    - 设计动机：与 L2P 的 hard top-k 选择相比，加权求和允许梯度流过所有 prompt 分量

3. **Query-Pool Regularization (QR) Loss**:

    - 功能：在训练时通过 reference ViT 提取最终层 [CLS] token $r$，约束中间层 query 与 prompt pool 的相似度分布逼近 reference 的分布
    - 核心思路：定义两个 softmax 归一化的相似度向量 $A_{query}^l = \text{Softmax}(\frac{K_l q_l^T}{\|K_l\|_2 \|q_l\|_2})$ 和 $A_{ref}^l = \text{Softmax}(\frac{K_l r^T}{\|K_l\|_2 \|r\|_2})$，QR loss 为 $\mathcal{L}_{QR} = \sum_l \|A_{query}^l - A_{ref}^l\|_2^2$
    - 设计动机：中间层 token 的表征能力弱于最终层，直接使用会导致约 1% 精度下降。QR loss 通过知识蒸馏的思路让 prompt pool 学到与最终层 query 一致的表征关系
    - 关键点：QR loss **仅在训练时使用**，推理时不需要 reference ViT，因此推理成本仍为原始的 50%

### 损失函数 / 训练策略

总损失为交叉熵分类损失与 QR loss 的加权和：

$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{QR}$$

其中 $\lambda$ 为超参数（默认 1e-4），通过 20% 训练集验证调优。训练时仅更新 prompt pool 中的 key 和 prompt，骨干 ViT 参数冻结。采用 prefix-tuning 方式将 prompt 拆分为 $[\phi_k, \phi_v]$ 并 prepend 到 self-attention 的 key 和 value。

## 实验关键数据

### 主实验

**ImageNet-R 10-task 设置（class-incremental，5 次不同种子平均）：**

| 方法 | $A_N$ (↑) | $F_N$ (↓) | 推理 GFLOPs |
|------|-----------|-----------|-------------|
| L2P | 69.29 | 2.03 | 35.1 (100%) |
| DualPrompt | 71.32 | 1.71 | 35.1 (100%) |
| CodaPrompt | 75.45 | 1.64 | 35.1 (100%) |
| **OS-Prompt** | 74.58 | 1.92 | **17.6 (50.1%)** |
| **OS-Prompt++** | **75.67** | **1.27** | **17.6 (50.1%)** |

**CIFAR-100 10-task 设置：**

| 方法 | $A_N$ (↑) | $F_N$ (↓) |
|------|-----------|-----------|
| CodaPrompt | 86.25 ± 0.74 | 1.67 ± 0.26 |
| OS-Prompt | 86.42 ± 0.61 | 1.64 ± 0.14 |
| OS-Prompt++ | **86.68 ± 0.67** | **1.18 ± 0.21** |

**DomainNet 5-task 设置：**

| 方法 | $A_N$ (↑) | $F_N$ (↓) |
|------|-----------|-----------|
| CodaPrompt | 73.24 ± 0.59 | 3.46 ± 0.09 |
| OS-Prompt++ | **73.32 ± 0.32** | **2.07 ± 0.06** |

### 消融实验

**QR Loss 设计消融（ImageNet-R 10-task）：**

| 配置 | $A_N$ (↑) | $F_N$ (↓) | 说明 |
|------|-----------|-----------|------|
| 无 CosSim 无 Softmax | 75.00 | 1.68 | baseline |
| 仅 CosSim | 75.47 | 1.38 | +0.47 |
| 仅 Softmax | 75.51 | 1.28 | +0.51 |
| CosSim + Softmax | **75.67** | **1.27** | 两者协同，最佳 |

**超参数 $\lambda$ 敏感性（ImageNet-R 5/10/20-task）：**

| $\lambda$ | Task-5 | Task-10 | Task-20 |
|-----------|--------|---------|---------|
| 1e-5 | 77.03 | 75.63 | 73.63 |
| 1e-4 | 77.07 | 75.67 | 73.77 |
| 5e-4 | 77.13 | 75.68 | 73.68 |

### 关键发现

- **QR loss 贡献最大**：OS-Prompt → OS-Prompt++ 在 ImageNet-R 20-task 上提升 1.77%（72.00 → 73.77），同时遗忘率从 1.09 降至 0.79
- **超参数不敏感**：$\lambda$ 在 1e-5 到 5e-4 范围内波动极小（< 0.1%）
- **Prompt 数量**：OS-Prompt++ 在 50 个 prompt 后即达到性能平台，OS-Prompt 则需要更多 prompt 才能饱和
- **推理延迟**：在 RTX2080ti、RTX3090、A100 三种 GPU 上延迟均降低约 50%
- **与不同 prompt formation 策略兼容**：OS-Prompt 框架与 L2P、DualPrompt 策略结合时也优于原始方法

## 亮点与洞察

- **极简但有效的观察**：早期层 embedding 的稳定性是整个方法的基石，这个观察非常简洁且 general——任何冻结骨干 + prompt tuning 的场景都可能成立
- **训练-推理解耦设计**：QR loss 仅训练时用，reference ViT 仅训练时需要，推理时完全是 one-stage，实现了精度和效率的双赢
- **可迁移思路**：这种"用中间层表征替代额外前向传播"的思路可以推广到其他需要两阶段推理的 prompt learning 方法中

## 局限与展望

- **训练成本未降低（OS-Prompt++ 版本）**：虽然推理省 50%，但训练时仍需 reference ViT 前向传播，训练 GFLOPs 与原方法相同
- **CodaPrompt 的 soft matching 在中间层 query 下略有退化**：实验显示 hard matching（L2P/Dual 的 top-k）对中间层 query 更鲁棒
- **仅验证了 ViT-B/16**：未在更大模型（ViT-L）或其他架构上验证
- **class-incremental 设置下仍有精度天花板**：与 UB（77.13%）仍有差距

## 相关工作与启发

- **vs CodaPrompt**: CodaPrompt 通过加权求和实现端到端 prompt pool 训练，是 SOTA；OS-Prompt++ 在其基础上用一阶段框架降低 50% 推理成本的同时精度更高
- **vs L2P / DualPrompt**: 这些方法提出了 prompt pool 的概念，但都需要双 ViT；OS-Prompt 框架与它们的 prompt formation 策略兼容且更优
- **vs DINO 预训练**: 在无监督预训练权重下，OS-Prompt 仍然保持优势，说明方法不依赖特定的预训练方式

## 评分

- 新颖性: ⭐⭐⭐⭐ 核心观察（早期层稳定性）简洁有力，one-stage 框架设计自然
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集、多种 task 配置、详细消融、GFLOPs/延迟对比、不同预训练权重、不同 prompt formation 策略
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，图表丰富，动机-观察-方法链条流畅
- 价值: ⭐⭐⭐⭐ 50% 推理加速对 PCL 的实际部署有重要意义，QR loss 的设计思路有启发性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Noise-Assisted Prompt Learning for Image Forgery Detection and Localization](noise-assisted_prompt_learning_for_image_forgery_detection_and_localization.md)
- [\[ICML 2026\] Active Continual Learning with Metaplastic Binary Bayesian Neural Networks](../../ICML2026/ai_safety/active_continual_learning_with_metaplastic_binary_bayesian_neural_networks.md)
- [\[ICML 2026\] HEDP: A Hybrid Energy-Distance Prompt-based Framework for Domain Incremental Learning](../../ICML2026/ai_safety/hedp_a_hybrid_energy-distance_prompt-based_framework_for_domain_incremental_lear.md)
- [\[CVPR 2026\] COPYLENS: Towards Copyrighted Characters Infringement Detection via Copyright-Aware Prompt Learning](../../CVPR2026/ai_safety/copylens_towards_copyrighted_characters_infringement_detection_via_copyright-awa.md)
- [\[NeurIPS 2025\] Model Inversion with Layer-Specific Modeling and Alignment for Data-Free Continual Learning](../../NeurIPS2025/ai_safety/model_inversion_with_layer-specific_modeling_and_alignment_for_data-free_continu.md)

</div>

<!-- RELATED:END -->
