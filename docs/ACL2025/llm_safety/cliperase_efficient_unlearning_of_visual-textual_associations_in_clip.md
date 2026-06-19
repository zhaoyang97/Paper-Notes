---
title: >-
  [论文解读] CLIPErase: Efficient Unlearning of Visual-Textual Associations in CLIP
description: >-
  [ACL 2025][LLM安全][机器遗忘] 提出 CLIPErase，一种专为 CLIP 多模态模型设计的机器遗忘框架，通过遗忘模块、保留模块和一致性模块三部分协同，选择性地移除特定视觉-文本关联，同时保持模型在保留数据上的性能。 机器遗忘（Machine Unlearning）旨在从已训练模型中移除特定数据的影响…
tags:
  - "ACL 2025"
  - "LLM安全"
  - "机器遗忘"
  - "CLIP"
  - "多模态"
  - "视觉-文本对齐"
  - "隐私保护"
---

# CLIPErase: Efficient Unlearning of Visual-Textual Associations in CLIP

**会议**: ACL 2025  
**arXiv**: [2410.23330](https://arxiv.org/abs/2410.23330)  
**代码**: [https://tianyu-yang-anna.github.io/ClipErase-ACL/](https://tianyu-yang-anna.github.io/ClipErase-ACL/)  
**领域**: LLM安全  
**关键词**: 机器遗忘, CLIP, 多模态, 视觉-文本对齐, 隐私保护

## 一句话总结

提出 CLIPErase，一种专为 CLIP 多模态模型设计的机器遗忘框架，通过遗忘模块、保留模块和一致性模块三部分协同，选择性地移除特定视觉-文本关联，同时保持模型在保留数据上的性能。

## 研究背景与动机

机器遗忘（Machine Unlearning）旨在从已训练模型中移除特定数据的影响，而无需完全重新训练。此前的工作主要集中在单模态领域（如文本分类或图像分类），但对 CLIP 等多模态模型的遗忘研究相对匮乏。

CLIP 模型通过对比学习在共享嵌入空间中对齐图像和文本表示。这种跨模态对齐带来了独特的遗忘挑战：

**单模态遗忘方法的失效**：如果只在文本模态上使用梯度上升（Gradient Ascent），会导致跨模态关系被意外破坏，下游任务（如扩散模型生成）产生扭曲图像甚至无法生成有意义的内容。

**多义概念的精确遗忘**：例如 "apple" 既可指水果也可指科技公司。单模态方法会不加区分地擦除 "apple" 的所有含义，而无法选择性地只遗忘其中一个语义。

**合规与隐私需求**：大规模多模态训练数据常包含敏感或受版权保护的内容，需要在不重新训练的情况下移除这些数据的影响。

## 方法详解

### 整体框架

CLIPErase 修改 CLIP 原始模型 Θ 的图像编码器和文本编码器，通过三个核心模块协同工作来实现遗忘：遗忘模块（FM）负责破坏遗忘集的跨模态关联，保留模块（RM）保持保留集性能，一致性模块（CM）确保遗忘后模型与原始模型在保留集上行为一致。

### 关键设计

1. **遗忘模块（Forgetting Module）**：通过最小化遗忘集中图像与文本嵌入的点积来破坏它们的对齐关系。损失函数为 $\mathcal{L}_{FM} = \frac{1}{N_f} \sum_{n=1}^{N_f} f_{img}(x_i^n) \cdot f_{txt}(x_t^n)$，将点积驱向零或负值，使遗忘集中的图文不再互相检索。这种设计直接且高效，避免了复杂的软标签或对抗训练。

2. **保留模块（Retention Module）**：在保留集上应用 CLIP 原始的对比损失函数 $\mathcal{L}_{RM}$，确保保留集中每张图像仍然与其对应文本紧密对齐，同时与其他图文对保持区分。选择对比损失而非 MSE 的原因是它能有效维护结构化的成对关系，避免引入冲突的学习信号。

3. **一致性模块（Consistency Module）**：通过 KL 散度惩罚遗忘模型 $\Theta_u$ 与原始模型 $\Theta$ 在保留集上的单模态输出分布差异。它同时考虑图像分布和文本分布的一致性：$\mathcal{L}_{CM} = \frac{1}{N_r} \sum_{n=1}^{N_r} [KL(p_o^{img} \| p_u^{img}) + KL(p_o^{txt} \| p_u^{txt})]$。这防止了因不同优化目标干扰而引入的预测偏差。

### 损失函数 / 训练策略

总体损失为三个模块的加权组合：

$$\mathcal{L} = \lambda_1 \mathcal{L}_{RM} + \lambda_2 \mathcal{L}_{FM} + \lambda_3 \mathcal{L}_{CM}$$

实验中设置 $\lambda_1 = 1, \lambda_2 = \lambda_3 = 3$。使用 Adam 优化器，学习率在 CIFAR-100 和 Conceptual 12M 上为 $1 \times 10^{-6}$，在 Flickr30K 上为 $1 \times 10^{-8}$。批大小为 16，训练 20 epochs，在验证集上选择最优 checkpoint，使用 NVIDIA V100 训练。

## 实验关键数据

### 主实验

| 数据集 | 任务 | 指标 | CLIPErase | 之前最优 | 说明 |
|--------|------|------|-----------|----------|------|
| CIFAR-100 | 零样本预测 | Df准确率↓ | **0.00%** | 0.00%(ENMN) | 完全遗忘 |
| CIFAR-100 | 零样本预测 | Dr准确率↑ | **90.99%** | 89.96%(GradDiff) | 保留集性能最优 |
| Conceptual 12M | 零样本预测 | Df准确率↓ | **0.74%** | 4.96%(GradDiff) | 近乎完全遗忘 |
| Conceptual 12M | 零样本预测 | Dr准确率↑ | **97.10%** | 97.01%(GradDiff) | 保留集性能持平 |
| Flickr30K | 图像检索 R@10 | Df↓/Dr↑ | 10.55/50.35 | 7.82/47.13(GradDiff) | 遗忘效果与保留性能平衡最佳 |

### 消融实验

| 配置 | Df准确率↓ | Dr准确率↑ | 说明 |
|------|----------|----------|------|
| 无任何模块 | 86.08% | 72.85% | 原始CLIP |
| FM only | 18.57% | 64.12% | 遗忘有效但破坏保留集 |
| FM + RM | 9.40% | 73.14% | RM 恢复保留集性能 |
| FM + RM + CM | **0.00%** | **90.80%** | CM 大幅提升保留集到90%以上 |

### 关键发现

- ENMN 虽然也能在遗忘集上达到 0% 准确率，但保留集性能暴跌至 12.46%，实际不可用。
- CLIPErase 在不同遗忘类别比例（3%～30%）下均保持稳健性，而 GA 和 GradDiff 在遗忘比例增大时遗忘效果反而下降。
- 在扩散模型实验中，CLIPErase 将 "apple" 的检测率从 100% 降至 2%，"bicycle" 从 90% 降至 8%，同时不影响其他概念的生成。
- 可推广到 BLIP 等其他 VLM 模型，在 BLIP 上也实现 Df=0%、Dr=83.12%。

## 亮点与洞察

- **模块化设计**：三个模块各司其职，消融实验清楚展示了每个组件的贡献。特别是 CM 模块带来的 Dr 从 73% 到 91% 的飞跃，说明维护单模态分布一致性对遗忘质量至关重要。
- **精细化遗忘能力**：能够区分 "apple" 水果与苹果公司，这在实际隐私合规中非常关键。
- **可视化验证充分**：注意力热图和 t-SNE 嵌入空间可视化直观展示了遗忘效果。
- **模型无关性**：框架不依赖 CLIP 特定组件，已验证可扩展到 BLIP-1。

## 局限与展望

1. 缺乏专门为多模态遗忘设计的标准化数据集和评估基准。
2. 目前仅适用于多模态嵌入模型，尚未扩展到 VLM 等生成式多模态模型。
3. 超参数 $\lambda_1, \lambda_2, \lambda_3$ 的选择依赖人工调优，缺少自适应机制。
4. 遗忘集的定义假设用户明确知道要遗忘哪些样本，实际场景中遗忘目标可能更模糊。

## 相关工作与启发

- 与 MultDelete（Cheng & Amiri, 2023）的比较：MultDelete 依赖随机采样的不相关对，可能在小数据集上失败且仅针对特定任务。CLIPErase 直接在 CLIP 共享嵌入空间中操作，适用于更广泛的下游任务。
- Gradient Ascent 方法虽直觉上合理但过于激进，容易破坏保留集。GradDiff 尝试平衡但缺乏一致性约束。
- 该工作对理解多模态模型中知识存储和跨模态关联的本质有重要启发。遗忘本质上是在共享嵌入空间中精确地"解绑"特定关联。

## 补充细节

- 在 BLIP 实验中，GA 方法虽然将遗忘集准确率降至 0%，但保留集暴跌至 42.89%；GradDiff 遗忘集仍高达 89.73%，基本无效。CLIPErase 在 BLIP 上也维持了 0% 遗忘和 83.12% 保留的良好平衡。
- 可视化分析中，t-SNE 图清楚显示遗忘后 "apple" 类的图像和文本嵌入距离显著增大，而其他类别的聚类结构完全不受影响。
- 实验在不同遗忘比例（3%、10%、20%、30%）下验证了鲁棒性。随遗忘比例增大，GA 的保留集准确率急剧下降，而 CLIPErase 始终维持与原始 CLIP 相当的保留集性能。

## 评分

- 新颖性: ⭐⭐⭐⭐ 多模态遗忘是新兴方向，三模块设计合理但每个模块技术上并非全新
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖5个下游任务、3个数据集、扩散模型和BLIP扩展，消融和可视化完善
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机阐述充分，图表设计直观
- 价值: ⭐⭐⭐⭐ 解决多模态隐私合规的实际需求，模型无关框架有推广价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Federated CLIP for Resource-Efficient Heterogeneous Medical Image Classification](../../AAAI2026/llm_safety/federated_clip_for_resource-efficient_heterogeneous_medical_image_classification.md)
- [\[ACL 2025\] ZJUKLAB at SemEval-2025 Task 4: Unlearning via Model Merging](zjuklab_at_semeval-2025_task_4_unlearning_via_model_merging.md)
- [\[AAAI 2026\] AUVIC: Adversarial Unlearning of Visual Concepts for Multi-modal Large Language Models](../../AAAI2026/llm_safety/auvic_adversarial_unlearning_of_visual_concepts_for_multi-mo.md)
- [\[NeurIPS 2025\] Enhancing CLIP Robustness via Cross-Modality Alignment](../../NeurIPS2025/llm_safety/enhancing_clip_robustness_via_crossmodality_alignment.md)
- [\[ACL 2025\] ReLearn: Unlearning via Learning for Large Language Models](relearn_unlearning_via_learning_for_large_language_models.md)

</div>

<!-- RELATED:END -->
