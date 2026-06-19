---
title: >-
  [论文解读] Random Initialization of Gated Sparse Adapters (RIGSA)
description: >-
  [ICML 2025][模型压缩][sparse fine-tuning] 提出 RIGSA，一种基于随机初始化全秩适配器 + ReZero 门控 + 迭代幅度剪枝的稀疏微调方法，在学习新任务的同时比 QLoRA 更好地保留源任务性能。 现有痛点 现有痛点：微调 LLM 学习新任务时面临灾难性遗忘：问题 领域现状 领域现状：…
tags:
  - "ICML 2025"
  - "模型压缩"
  - "sparse fine-tuning"
  - "PEFT"
  - "lottery ticket hypothesis"
  - "catastrophic forgetting"
  - "LoRA"
---

# Random Initialization of Gated Sparse Adapters (RIGSA)

**会议**: ICML 2025  
**arXiv**: [2511.01794](https://arxiv.org/abs/2511.01794)  
**代码**: -  
**领域**: Parameter-Efficient Fine-Tuning / Sparse Adaptation  
**关键词**: sparse fine-tuning, PEFT, lottery ticket hypothesis, catastrophic forgetting, LoRA  

## 一句话总结

提出 RIGSA，一种基于随机初始化全秩适配器 + ReZero 门控 + 迭代幅度剪枝的稀疏微调方法，在学习新任务的同时比 QLoRA 更好地保留源任务性能。

## 研究背景与动机

### 现有痛点

**现有痛点**：微调 LLM 学习新任务时面临**灾难性遗忘**问题

### 领域现状

**领域现状**：LoRA 通过低秩约束实现参数高效微调，但低秩限制在复杂任务上表现欠佳

### 核心矛盾

**核心矛盾**：稀疏微调提供了一种替代方案，不施加秩约束，可能允许更具表达力的适应

### 解决思路

**解决思路**：彩票假说 (LTH) 表明密集网络中存在稀疏子网络能匹配全网性能

### 补充说明

**补充说明**：现有稀疏微调方法（如 LT-SFT）将差分矩阵 $\Delta W$ 初始化为零，无法包含"幸运"初始化

## 方法详解

### 核心思想

学习 $W = W_0 + \alpha \Delta W$，其中：
- $W_0$：冻结的预训练权重
- $\Delta W$：随机初始化的全秩差分矩阵
- $\alpha$：ReZero 风格的可学习门控参数，初始化为 $10^{-6}$

**关键设计**：$\alpha$ 近零初始化确保训练从预训练权重 $W_0$ 出发，而 $\Delta W$ 的随机初始化允许偏离预训练权重，权重衰减又引导回到该区域。

### 迭代幅度剪枝 (IMP)

1. 训练 $W_0 + \alpha \Delta W$ 一个 epoch
2. 剪枝：在未改变符号的参数中，保留幅度最大的 80%，其余重置为初始值并冻结
3. 重复 5 次 → 最终稀疏度约 3.46%
4. 用最终稀疏掩码训练得到 winning ticket

### 与其他方法的区别

| 方法 | 初始化 | 秩约束 | 剪枝策略 |
|------|--------|--------|----------|
| LoRA/QLoRA | B 初始化为零 | 低秩 | 无 |
| LT-SFT | $\Delta W = 0$ | 无 | 单次剪枝 |
| RoSA | 梯度累积 | 低秩+稀疏联合 | 基于梯度 |
| **RIGSA** | 随机 + ReZero 门控 | 无（全秩→稀疏） | IMP + 符号保留 |

## 实验结果

### 目标任务：Textual MNIST

将 MNIST 图像转为数字文本矩阵（每像素量化为 0-9），作为纯文本图像分类任务：
- SmolLM2-1.7B-Instruct 零样本准确率仅约 10%（随机水平）
- 微调后可有效学习

### 主实验对比

| 方法 | Textual MNIST | PIQA | HellaSwag | GSM8k |
|------|--------------|------|-----------|-------|
| 基线（未微调） | ~10% | 75.4 | 51.7 | 43.7 |
| RIGSA (step 1, dense) | 99.05% | - | - | 40.7↓ |
| RIGSA (step 3, sparse) | 98.37% | ~75 | ~52 | **45.1↑** |
| QLoRA (rank=16) | **99.46%** | ~75 | ~51 | 14.18↓↓ |
| Random Mask | ~97% | - | - | ~43 |

### 关键发现

- **目标任务**：QLoRA 略优于 RIGSA（99.46% vs 98.37%）
- **遗忘**：RIGSA 在 GSM8k 上遗忘远少于 QLoRA（下降 ~0 vs 下降 ~6%）
- 稀疏微调（包括随机掩码）在保留源任务性能方面系统性优于 QLoRA
- 高秩 QLoRA 反而保留源任务更好——可能因为高秩适应更"自然"

## 亮点与洞察

- 创新地将 LTH 思想应用于 LLM 适配器，使用 ReZero 门控解决随机初始化的不稳定性
- 提出 Textual MNIST 作为标准化的 OOD 视觉文本任务
- 揭示了稀疏微调在减少遗忘方面的系统性优势
- 方法简洁，不需要复杂的掩码选择策略

## Textual MNIST 任务设计

论文提出的 Textual MNIST 作为 OOD 评估基准具有独特价值：

- 将 28×28 灰度图像每个像素量化为 0-9，形成纯文本表示
- SmolLM2 零样本/5-shot 准确率均约 10%（随机水平）
- 任务与预训练数据分布完全不同，清晰度量学习新能力
- 与 BigBench 的 ASCII art 方法不同，使用单字符数字编码更适合 tokenizer

这种设计有效隔离了迁移学习的影响，是评估适配器学习能力的理想测试平台。

## 局限与展望

- 仅在 1.7B 参数模型上实验，未验证更大模型
- 每轮剪枝 80% 过于激进，可能导致次优性能
- 单一目标任务（Textual MNIST）的验证不够充分
- 缺乏多次实验的统计显著性分析
- 与 RoSA 等更强的稀疏微调基线缺乏直接对比
- 高权重衰减 (1.0) 的选择缺乏理论支撑

## 评分

⭐⭐⭐ — 思路清晰有趣，ReZero 门控+IMP 的组合新颖，但实验规模和深度不足以充分验证方法的优势。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Beyond Zero Initialization: Investigating the Impact of Non-Zero Initialization on LoRA Fine-Tuning Dynamics](beyond_zero_initialization_investigating_the_impact_of_non-zero_initialization_o.md)
- [\[ICML 2025\] Neutral Residues: Revisiting Adapters for Model Extension](neutral_residues_revisiting_adapters_for_model_extension.md)
- [\[AAAI 2026\] Put the Space of LoRA Initialization to the Extreme to Preserve Pre-trained Knowledge](../../AAAI2026/model_compression/put_the_space_of_lora_initialization_to_the_extreme_to_preserve_pre-trained_know.md)
- [\[NeurIPS 2025\] Gated Integration of Low-Rank Adaptation for Continual Learning of Large Language Models](../../NeurIPS2025/model_compression/gated_integration_of_low-rank_adaptation_for_continual_learning_of_large_languag.md)
- [\[ACL 2025\] Limited-Resource Adapters Are Regularizers, Not Linguists](../../ACL2025/model_compression/limited-resource_adapters_are_regularizers_not_linguists.md)

</div>

<!-- RELATED:END -->
