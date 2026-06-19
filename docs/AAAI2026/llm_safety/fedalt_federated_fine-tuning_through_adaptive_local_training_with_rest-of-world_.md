---
title: >-
  [论文解读] FedALT: Federated Fine-Tuning through Adaptive Local Training with Rest-of-World LoRA
description: >-
  [AAAI 2026][LLM安全][联邦学习] 提出 FedALT，通过为每个客户端维护独立的 Individual LoRA（本地训练更新）和冻结的 Rest-of-World (RoW) LoRA（其他客户端平均），配合自适应 MoE 混合器动态平衡本地知识与全局知识，彻底避免 FedAvg 聚合导致的跨客户端干扰，在异构任务联邦 LLM 微调上显著优于 SOTA。
tags:
  - "AAAI 2026"
  - "LLM安全"
  - "联邦学习"
  - "LoRA微调"
  - "个性化"
  - "跨客户端干扰"
  - "MoE"
---

# FedALT: Federated Fine-Tuning through Adaptive Local Training with Rest-of-World LoRA

**会议**: AAAI 2026  
**arXiv**: [2503.11880](https://arxiv.org/abs/2503.11880)  
**代码**: 无  
**领域**: AI安全/联邦学习  
**关键词**: 联邦学习, LoRA微调, 个性化, 跨客户端干扰, MoE

## 一句话总结
提出 FedALT，通过为每个客户端维护独立的 Individual LoRA（本地训练更新）和冻结的 Rest-of-World (RoW) LoRA（其他客户端平均），配合自适应 MoE 混合器动态平衡本地知识与全局知识，彻底避免 FedAvg 聚合导致的跨客户端干扰，在异构任务联邦 LLM 微调上显著优于 SOTA。

## 研究背景与动机

**领域现状**：联邦 LoRA 微调已成为隐私保护 LLM 适应的主流范式。FedIT 等方法沿用 FedAvg 框架——聚合本地 LoRA→用聚合模型初始化下轮训练。FedDPA 引入全局+本地双 LoRA 组件但仍基于 FedAvg。

**现有痛点**：
   - **有害的跨客户端干扰**：当客户端任务差异大时（如文本摘要 vs 情感分析），FedAvg 聚合会抵消各个客户端在本地微调中取得的进步
   - **缺乏有效的全局-本地平衡机制**：FedDPA 等用固定权重组合全局和本地 LoRA，无法针对不同输入动态调整
   - 实验验证：FedIT 在 Commonsense Reasoning 和 Text Classification 上反而不如纯本地微调

**核心矛盾**：如何从其他客户端获取有用知识，同时避免聚合破坏本地适应？

**本文目标** 设计一种摆脱 FedAvg 范式的个性化联邦 LoRA 微调方法。

**切入角度**：不再用聚合模型初始化本地训练。每个客户端继续在自己之前训练的本地模型上学习，全局知识通过一个冻结的"其余世界" LoRA 注入，配合自适应混合器按输入动态加权。

**核心 idea**：用冻结的 RoW LoRA 提供全局知识 + 可训练的 Individual LoRA 做本地适应 + MoE 混合器动态平衡，完全避免 FedAvg 聚合干扰。

## 方法详解

### 整体框架
每个客户端 $k$ 维护两个 LoRA 模块和一个混合器：
- **Individual LoRA** $\mathbf{A}_k^L / \mathbf{B}_k^L$：本地训练更新，捕获客户端特有知识
- **RoW LoRA** $\mathbf{A}_k^R / \mathbf{B}_k^R$：所有其他客户端 Individual LoRA 的平均，本地训练时冻结
- **Mixer** $\mathbf{G}_k$：动态学习两个 LoRA 的输入相关权重

前向传播：$y = \mathbf{W}_0 x + \alpha_k(x) \mathbf{B}_k^L \mathbf{A}_k^L x + (1-\alpha_k(x)) \mathbf{B}_k^R \mathbf{A}_k^R x$

### 关键设计

1. **Individual LoRA + RoW LoRA 分离**:

    - 功能：将本地知识和全局知识显式分离到两个独立 LoRA
    - 核心思路：RoW LoRA 计算为 $\mathbf{A}_k^R = \frac{1}{K-1} \sum_{m \neq k} \mathbf{A}_m^L$。关键：RoW LoRA 在本地训练时完全冻结，不参与梯度更新
    - 设计动机：FedAvg 范式中干扰来自两个环节——(1) 聚合抵消本地改进 (2) 用聚合模型初始化覆盖本地适应。冻结 RoW 完全消除这两个问题。且跳过 RoW 的本地训练将客户端计算量减半

2. **自适应 MoE 混合器**:

    - 功能：按输入动态调整 Individual LoRA 和 RoW LoRA 的贡献权重
    - 核心思路：$\alpha(x), 1-\alpha(x) = \text{softmax}(\mathbf{G}_k x)$，其中 $\mathbf{G}_k \in \mathbb{R}^{2 \times d}$ 是可训练的线性层
    - 设计动机：不同输入从本地模型和全局模型获益程度不同。固定权重（如 FedDPA）是次优的。MoE 范式提供了输入自适应的灵活权重
    - 重要：混合器是个性化的（不在客户端间平均），确保反映各客户端独特的数据分布

3. **为什么不直接把 RoW 加到预训练模型**:

    - 论文专门讨论了这个替代方案并指出两个问题：(1) 如果 RoW 性能差会"污染"预训练模型，难以纠正 (2) 失去灵活性——不同输入需要不同的全局-本地平衡

### 训练策略
- 服务器端：收集所有客户端的 Individual LoRA，计算每个客户端的 RoW LoRA 并分发
- 客户端：用新 RoW LoRA 替换旧的，然后更新 Individual LoRA 和 Mixer（RoW 和预训练模型冻结）
- 上传：仅上传 Individual LoRA，Mixer 留在本地

## 实验关键数据

### 主实验（LLaMA2-7B，8个异构任务）

| 方法 | 常识推理 | 指代消解 | 文本分类 | 平均 |
|------|---------|---------|---------|------|
| Local Only | 73.83 | 74.62 | 67.18 | 62.86 |
| FedIT (FedAvg) | 72.82 | 77.14 | 66.39 | 62.19 |
| FedDPA | 74.81 | 81.88 | 65.42 | 64.64 |
| FDLoRA | 76.29 | 75.60 | 67.59 | 65.17 |
| **FedALT** | **76.12** | **83.04** | **71.60** | **67.55** |

FedALT 平均性能 67.55%，比最佳基线 FDLoRA 高 2.38%，比 Local Only 高 4.69%。

### 消融实验

| 配置 | 平均性能 |
|------|----------|
| FedALT (Full) | 67.55 |
| w/o Mixer (固定 α=0.5) | 65.82 |
| w/o RoW LoRA (Local Only) | 62.86 |
| 用 FedAvg 聚合 | 62.19 |

### 关键发现
- FedAvg 在部分任务上反而比纯本地差（常识推理：72.82 < 73.83），验证了跨客户端干扰的存在
- 简单拆分单个大 LoRA 为多个小 LoRA（FedIT-split）无法缓解干扰——干扰的根源在服务器聚合而非模型内部
- Mixer 贡献显著（+1.73%），验证了动态输入自适应加权的价值
- Bloom-560M 上同样有效，说明方法对模型规模不敏感

## 亮点与洞察
- **彻底摆脱 FedAvg 范式**的思路很大胆——不再用聚合模型初始化，而是让每个客户端持续训练自己的模型。这从根本上消除了跨客户端干扰
- **RoW LoRA 冻结+Mixer 动态加权**的组合设计优雅：冻结保证不干扰→Mixer 保证灵活利用全局知识→两者互补
- **Motivational Study 做得好**：用 FedIT 和 Local Only 对比8个任务，清晰展示了干扰和收益共存的现实问题

## 局限与展望
- Mixer 是简单的 2×d 线性层+softmax，更复杂的路由机制可能带来进一步提升
- 每轮通信量与客户端数目无关（只传 Individual LoRA），但 RoW 计算需要所有客户端的 LoRA
- 仅在 NLP 任务上验证，多模态或视觉 LLM 的联邦微调有待探索
- 客户端部分参与（partial participation）场景下 RoW LoRA 的计算需要额外处理

## 相关工作与启发
- **vs FedDPA**: FedDPA 的全局 LoRA 仍用 FedAvg 训练→受干扰；FedALT 的 RoW 冻结完全避免
- **vs FDLoRA**: FDLoRA 依赖服务器端数据集且全局 LoRA 聚合仍有干扰；FedALT 无此依赖
- **vs HydraLoRA**: HydraLoRA 在集中式设置中用多 LoRA 减少干扰，但在联邦设置中同样无效——干扰来自聚合而非模型内部

## 评分
- 新颖性: ⭐⭐⭐⭐ 摆脱 FedAvg 范式+RoW 冻结+MoE 混合器，设计逻辑清晰
- 实验充分度: ⭐⭐⭐⭐ 2个LLM+8个任务+6个基线+充分消融
- 写作质量: ⭐⭐⭐⭐⭐ Motivational Study→问题定义→解决方案的叙事非常流畅
- 价值: ⭐⭐⭐⭐ 为异构联邦 LLM 微调提供了有效的个性化方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Adaptive LoRA Experts Allocation and Selection for Federated Fine-Tuning](../../NeurIPS2025/llm_safety/adaptive_lora_experts_allocation_and_selection_for_federated_fine-tuning.md)
- [\[ICLR 2026\] SHE-LoRA: Selective Homomorphic Encryption for Federated Tuning with Heterogeneous LoRA](../../ICLR2026/llm_safety/she-lora_selective_homomorphic_encryption_for_federated_tuning_with_heterogeneou.md)
- [\[AAAI 2026\] TOFA: Training-Free One-Shot Federated Adaptation for Vision-Language Models](tofa_training-free_one-shot_federated_adaptation_for_vision-language_models.md)
- [\[NeurIPS 2025\] FedSVD: Adaptive Orthogonalization for Private Federated Learning with LoRA](../../NeurIPS2025/llm_safety/fedsvd_adaptive_orthogonalization_for_private_federated_learning_with_lora.md)
- [\[AAAI 2026\] Federated CLIP for Resource-Efficient Heterogeneous Medical Image Classification](federated_clip_for_resource-efficient_heterogeneous_medical_image_classification.md)

</div>

<!-- RELATED:END -->
