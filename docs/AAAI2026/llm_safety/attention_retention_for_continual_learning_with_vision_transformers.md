---
title: >-
  [论文解读] Attention Retention for Continual Learning with Vision Transformers
description: >-
  [AAAI 2026][LLM安全][持续学习] 提出ARCL-ViT框架，通过注意力掩码生成和梯度掩码两步策略防止ViT在持续学习中的注意力漂移，在ImageNet-R和CIFAR-100上取得SOTA结果，证明保持注意力模式是解决灾难性遗忘的关键。 领域现状：持续学习要求模型在学习新任务时保持对旧任务的性能…
tags:
  - "AAAI 2026"
  - "LLM安全"
  - "持续学习"
  - "Transformer"
  - "注意力保持"
  - "灾难性遗忘"
  - "梯度掩码"
---

# Attention Retention for Continual Learning with Vision Transformers

**会议**: AAAI 2026  
**arXiv**: [2602.05454](https://arxiv.org/abs/2602.05454)  
**代码**: 无  
**领域**: LLM安全  
**关键词**: 持续学习, Vision Transformer, 注意力保持, 灾难性遗忘, 梯度掩码

## 一句话总结
提出ARCL-ViT框架，通过注意力掩码生成和梯度掩码两步策略防止ViT在持续学习中的注意力漂移，在ImageNet-R和CIFAR-100上取得SOTA结果，证明保持注意力模式是解决灾难性遗忘的关键。

## 研究背景与动机
**领域现状**：持续学习要求模型在学习新任务时保持对旧任务的性能。ViT在CL中的应用日益增多。

**现有痛点**：(a) 灾难性遗忘在ViT中表现为**注意力漂移**；(b) 正则化方法（EWC）对ViT效果有限；(c) 扩展方法（DualPrompt）增加大量参数。

**核心矛盾**：更新参数学习新任务时可能破坏对旧任务特征的注意力分配。

**本文目标** 直接防止ViT中旧任务的注意力模式被破坏。

**切入角度**：受人类V1视觉皮层选择性注意启发——保持对重要特征的持续关注。

**核心 idea**：生成前一任务的注意力掩码，在新任务训练时零化对应区域的Q/K/V梯度，直接防止注意力漂移。

## 方法详解

### 整体框架
输入：连续到达的任务序列。输出：能处理所有已学任务的ViT。两步：(1) 层级rollout提取注意力图 → 自适应阈值 → 二值掩码；(2) 掩码零化Q/K/V权重梯度。

### 关键设计

1. **注意力掩码生成**：

    - 功能：从前一任务提取需保护的注意力区域
    - 核心思路：层级rollout提取 $\mathbf{U}_{t-1}$，实例自适应阈值生成 $\bar{\mathbf{M}}_{t-1}$
    - 设计动机：识别对旧任务判别性特征至关重要的注意力区域

2. **梯度掩码**：

    - 功能：新任务训练时保护旧注意力模式
    - 核心思路：$\nabla \mathbf{W}'_{\theta,t} = \nabla \mathbf{W}_{\theta,t} \odot (1 - \bar{\mathbf{M}}_{t-1})$，配合Adam缩放 $\Delta\mathbf{W}'_{\theta,t} = (\nabla\mathbf{W}'_{\theta,t} / \nabla\mathbf{W}_{\theta,t}) \odot \Delta\mathbf{W}_{\theta,t}$
    - 设计动机：直接在梯度层面阻止对旧任务关键区域的修改，与Adam兼容

3. **实例自适应阈值**：

    - 功能：为不同样本生成不同的二值化阈值
    - 设计动机：不同任务/样本的注意力分布差异很大

### 损失函数 / 训练策略
标准交叉熵损失，梯度掩码在反向传播后应用，无需修改损失函数。

## 实验关键数据

### 主实验

| 方法 | 10S-ImageNet-R | 20S-ImageNet-R | 10S-CIFAR-100 |
|------|---------------|---------------|--------------|
| CODA-Prompt | 75.45% | - | 86-89% |
| OS-Prompt++ | - | 73.77% | - |
| **ARCL-ViT** | **SOTA** | **SOTA** | **~87%** |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 完整模型 | 最优 | 注意力掩码+梯度掩码 |
| w/o 梯度掩码 | 严重退化 | 等同Seq-FT |
| w/o 自适应阈值 | 轻微下降 | 全局阈值不够灵活 |
| 不同预训练方案 | 鲁棒 | 对预训练不敏感 |

### 关键发现
- 注意力漂移是ViT灾难性遗忘的主因，可视化清楚证明
- 梯度掩码比正则化和扩展方法更有效
- 对长序列任务（20S）和不同预训练方案均鲁棒

## 亮点与洞察
- **问题定义精准**：将灾难性遗忘归因为注意力漂移，可视化证据非常有说服力。
- **生物启发的优雅解法**：简单的梯度掩码实现注意力保持，方法简洁但有效。
- **Adam兼容性设计**：梯度/更新缩放trick是工程上的关键贡献。

## 局限与展望
- 需要存储前一任务的注意力掩码，随任务增多线性增长
- 二值掩码可能过于粗糙，软掩码可能更灵活
- 仅在ViT上验证，CNN适用性未知
- 需要知道任务边界

## 相关工作与启发
- **vs EWC**：EWC在参数空间做正则化，ARCL-ViT在注意力空间直接保护，对ViT更有效
- **vs DualPrompt/CODA-Prompt**：扩展方法增加参数，ARCL-ViT不增模型大小
- **vs PackNet**：类似思路从CNN剪枝迁移到ViT注意力保护

## 评分
- 新颖性: ⭐⭐⭐⭐ 注意力漂移视角新颖
- 实验充分度: ⭐⭐⭐⭐ 多数据集多设置加可视化
- 写作质量: ⭐⭐⭐⭐ 清晰直观
- 价值: ⭐⭐⭐⭐ 对ViT持续学习有实用贡献

## 补充说明
- 该工作的方法论和实验设计对相关领域有参考价值
- 后续工作可在更多场景和更大规模上验证方法的泛化性和可扩展性
- 与近期相关工作的结合（如与 RL/MCTS/多模态方法的交叉）有潜在研究价值
- 建议结合实际应用需求评估该方法的部署可行性和计算效率
- 数据集和评估指标的选择可能影响结论的普适性，需在更多 benchmark 上交叉验证

## 补充说明
- 该工作的方法论和实验设计对相关领域有参考价值
- 后续工作可在更多场景和更大规模上验证方法的泛化性和可扩展性
- 与近期相关工作的结合（如与 RL/MCTS/多模态方法的交叉）有潜在研究价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Elastic Weight Consolidation Done Right for Continual Learning](../../CVPR2026/llm_safety/elastic_weight_consolidation_done_right_for_continual_learning.md)
- [\[NeurIPS 2025\] Finding Structure in Continual Learning](../../NeurIPS2025/llm_safety/finding_structure_in_continual_learning.md)
- [\[AAAI 2026\] PANDA: Patch and Distribution-Aware Augmentation for Long-Tailed Exemplar-Free Continual Learning](panda_--_patch_and_distribution-aware_augmentation_for_long-tailed_exemplar-free.md)
- [\[NeurIPS 2025\] Attention! Your Vision Language Model Could Be Maliciously Manipulated](../../NeurIPS2025/llm_safety/attention_your_vision_language_model_could_be_maliciously_manipulated.md)
- [\[ACL 2026\] Seeing No Evil: Blinding Large Vision-Language Models to Safety Instructions via Adversarial Attention Hijacking](../../ACL2026/llm_safety/seeing_no_evil_blinding_large_vision-language_models_to_safety_instructions_via_.md)

</div>

<!-- RELATED:END -->
