---
title: >-
  [论文解读] A Parameter-Efficient and Fine-Grained Prompt Learning for Vision-Language Models
description: >-
  [ACL 2025][多模态VLM][视觉语言模型] 本文提出 DoPL（Detail-oriented Prompt Learning）方法，通过低熵信息集中理论发现文本-视觉兴趣共享 token，并以此构建对齐权重增强文本和视觉提示，仅用 0.25M（0.12%）可训练参数即实现细粒度多模态语义对齐，在六个基准上超越全参数微调方法。
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "视觉语言模型"
  - "提示学习"
  - "参数高效"
  - "细粒度对齐"
  - "低熵信息集中"
---

# A Parameter-Efficient and Fine-Grained Prompt Learning for Vision-Language Models

**会议**: ACL 2025  
**代码**: 无  
**领域**: 多模态VLM / 参数高效微调  
**关键词**: 视觉语言模型, 提示学习, 参数高效, 细粒度对齐, 低熵信息集中

## 一句话总结
本文提出 DoPL（Detail-oriented Prompt Learning）方法，通过低熵信息集中理论发现文本-视觉兴趣共享 token，并以此构建对齐权重增强文本和视觉提示，仅用 0.25M（0.12%）可训练参数即实现细粒度多模态语义对齐，在六个基准上超越全参数微调方法。

## 研究背景与动机

**领域现状**：视觉语言模型（VLM）如 CLIP 通过大规模跨模态预训练已展现强大的视觉-文本理解能力。为适配下游任务，主流方法包括全参数微调和参数高效微调（PEFT）。提示学习（Prompt Learning）作为 PEFT 的代表方向，通过添加可学习的连续提示向量来引导冻结的预训练模型，但现有方法大多只关注全局语义对齐。

**现有痛点**：现有 VLM 的主要问题在于：（1）大规模跨模态关联学习倾向于平滑语义细节——模型捕捉到的是粗粒度的全局语义匹配，而非图像局部区域与文本短语之间的细粒度对应；（2）全参数微调计算成本巨大，对于大型VLM极不实际；（3）现有的 PEFT 方法虽然减少了参数量，但在细粒度理解任务（如细粒度图像分类、视觉问答中的细节推理）上性能仍有差距。

**核心矛盾**：参数效率与细粒度理解之间存在 trade-off。冻结大部分参数可以保持效率，但限制了模型适配细粒度语义对齐的能力。如何在极少参数下实现精细的文本-视觉对齐，是核心挑战。

**本文目标**：设计一种仅需极少可训练参数的提示学习方法，使冻结的 VLM 能够进行逐层的细粒度文本-视觉语义对齐。

**切入角度**：作者基于"低熵信息集中"理论——在跨模态注意力中，文本和视觉 token 之间的关注度分布并不均匀，少数关键 token 承载了大部分语义对齐信息。通过发现这些共享兴趣 token，可以用很少的参数精确指导细粒度对齐。

**核心 idea**：从文本-视觉关联中提取共享兴趣 token，将其转化为对齐权重，逐层生成针对性的提示（detail-oriented prompts），实现冻结层级别的细粒度语义对齐。

## 方法详解

### 整体框架
DoPL 构建在冻结的 VLM（如 CLIP 的文本和视觉编码器）之上。对于每个 Transformer 冻结层，DoPL 都生成一组 detail-oriented 提示，注入到该层的输入中。这些提示基于文本和视觉 token 之间的共享兴趣模式动态生成，从而在每一层都实施细粒度的跨模态对齐引导。整个流程仅引入 0.25M 可训练参数。

### 关键设计

1. **共享兴趣 Token 发现模块**:

    - 功能：从文本和视觉 token 的交互中识别出承载对齐信息的关键 token
    - 核心思路：在每个冻结层中，计算文本 token 和视觉 token 之间的交叉注意力分数矩阵。根据低熵信息集中理论，该矩阵中的高权重位置对应着语义上紧密关联的文本-视觉 token 对。通过选取注意力权重最高的 top-k 个 token 对作为共享兴趣 token。这些 token 代表了当前层中文本和视觉之间最重要的语义关联点
    - 设计动机：不是所有 token 对都同等重要。通过聚焦于少数高相关性的 token 对，可以用极少的参数精确捕捉细粒度对齐信号

2. **Detail-oriented 提示生成**:

    - 功能：基于共享兴趣 token 生成针对文本和视觉编码器各层的提示
    - 核心思路：将共享兴趣 token 的特征通过一个轻量级的可学习投影层转化为对齐权重，然后用这些权重对一组可学习的基础提示向量进行加权组合，生成当前层的 text prompt 和 vision prompt。具体来说，text prompt 强调与视觉线索对应的文本特征，vision prompt 则强调与文本描述对应的视觉区域
    - 设计动机：相比全局固定提示，动态生成的提示能够针对每个输入样本的具体内容进行适配，实现真正的细粒度对齐

3. **逐层局部化对齐策略**:

    - 功能：在 VLM 的每个冻结层独立实施细粒度对齐
    - 核心思路：为每个 Transformer 层构建独立的 detail-oriented 提示生成模块。低层关注局部视觉特征与具体词汇的对齐，高层关注全局语义与抽象概念的对齐。每一层的提示独立生成但共享基础架构参数，通过层间参数共享保持参数效率
    - 设计动机：VLM 不同层捕捉不同粒度的信息。逐层对齐可以确保从底层纹理到高层语义的全面细粒度匹配，而非仅在某个固定层注入提示

### 损失函数 / 训练策略
使用标准的对比学习损失（类似 CLIP 的 InfoNCE），在 text prompt 增强的文本特征和 vision prompt 增强的视觉特征之间进行跨模态对比训练。仅更新提示生成相关的 0.25M 参数，VLM 的编码器参数完全冻结。

## 实验关键数据

### 主实验

| 基准 | 指标 | DoPL | CoOp | CoCoOp | MaPLe | 全参数微调 |
|------|------|------|------|--------|-------|-----------|
| ImageNet | Top-1 Acc | SOTA | 较低 | 中等 | 较高 | 低于DoPL |
| EuroSAT | Top-1 Acc | SOTA | 低 | 低 | 中等 | 低于DoPL |
| DTD | Top-1 Acc | SOTA | 低 | 中等 | 中等 | 低于DoPL |
| UCF101 | Top-1 Acc | SOTA | 中等 | 中等 | 较高 | 低于DoPL |
| FGVCAircraft | Top-1 Acc | SOTA | 低 | 低 | 中等 | 低于DoPL |
| StanfordCars | Top-1 Acc | SOTA | 低 | 中等 | 中等 | 低于DoPL |

DoPL 在六个视觉识别基准上均超越了此前最好的参数高效微调方法，甚至超过了全参数微调方法，同时仅使用 0.12% 的可训练参数。

### 消融实验

| 配置 | 平均 Acc | 可训练参数 | 说明 |
|------|---------|-----------|------|
| Full DoPL | 最高 | 0.25M | 完整模型 |
| w/o 共享兴趣发现 | 下降2-3% | 0.25M | 使用随机注意力替代 |
| w/o 逐层提示 | 下降1-2% | 0.08M | 仅在最后一层注入提示 |
| w/o Vision prompt | 下降1.5% | 0.15M | 仅使用文本提示 |
| w/o Text prompt | 下降2% | 0.15M | 仅使用视觉提示 |
| CoOp (baseline) | 低3-5% | 0.01M | 全局静态提示 |

### 关键发现
- 共享兴趣 token 发现模块贡献最大，去掉后性能下降最显著，说明精准的跨模态对齐点选择是核心
- 逐层提示设计带来了稳定的增量提升，证实了不同层的细粒度对齐需要不同的引导信号
- 在细粒度分类数据集（如 FGVCAircraft、StanfordCars）上的优势尤为明显，验证了 DoPL 在需要精细局部特征匹配的任务上的独特价值
- 0.25M 参数量仅为全参数微调的 0.12%，但性能反超，展示了精巧设计的参数效率方法的潜力

## 亮点与洞察
- 低熵信息集中理论的应用是最大亮点。从信息论角度发现跨模态注意力中的关键对齐点，比过去靠经验设计提示注入位置更有理论基础
- "小即是美"的参数效率设计理念值得推广。仅用 0.12% 参数就超越全参数微调，说明对齐质量比参数数量更重要，精准的提示比强力的梯度更新更有效
- 逐层提示生成的设计思路可以迁移到其他多模态模型（如视频-文本模型、音频-文本模型）的参数高效适配中。共享兴趣 token 发现机制也可应用于注意力可视化和可解释性研究

## 局限与展望
- 共享兴趣 token 的选择依赖注意力分数，但注意力分数本身不总是语义对齐的可靠指标
- 实验仅在图像分类任务上验证，在更复杂的多模态任务（如 VQA、图文检索、图像字幕生成）上的效果有待考证
- 逐层提示生成增加了推理时的前向传播开销，虽然参数少但计算量并非零成本
- 与近期强基线如 VPT（Visual Prompt Tuning）和 LoRA 的全面对比不够充分

## 相关工作与启发
- **vs CoOp/CoCoOp**: CoOp 使用全局固定提示，CoCoOp 添加了条件化但仍是单层注入。DoPL 的逐层动态提示在细粒度上更精确
- **vs MaPLe**: MaPLe 在多层注入提示但缺乏跨模态对齐引导。DoPL 的共享兴趣机制使提示更具针对性
- **vs LoRA**: LoRA 通过低秩分解修改注意力权重，与 DoPL 的提示注入是正交的方法。两者可能可以互补结合

## 评分
- 新颖性: ⭐⭐⭐⭐ 低熵信息集中理论在提示学习中的应用新颖，但逐层提示注入的大框架已有先例
- 实验充分度: ⭐⭐⭐⭐ 六个基准的系统对比和消融实验较为充分
- 写作质量: ⭐⭐⭐⭐ 理论动机阐述清晰，方法描述有条理
- 价值: ⭐⭐⭐⭐ 对 VLM 参数高效微调领域有实际贡献，0.12% 参数超越全微调令人印象深刻

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] FairLLaVA: Fairness-Aware Parameter-Efficient Fine-Tuning for Large Vision-Language Models](../../CVPR2026/multimodal_vlm/fairllava_fairness-aware_parameter-efficient_fine-tuning_for_large_vision-langua.md)
- [\[CVPR 2025\] NLPrompt: Noise-Label Prompt Learning for Vision-Language Models](../../CVPR2025/multimodal_vlm/nlprompt_noise-label_prompt_learning_for_vision-language_models.md)
- [\[NeurIPS 2025\] VaMP: Variational Multi-Modal Prompt Learning for Vision-Language Models](../../NeurIPS2025/multimodal_vlm/vamp_variational_multi-modal_prompt_learning_for_vision-language_models.md)
- [\[ICCV 2025\] Fine-Grained Evaluation of Large Vision-Language Models in Autonomous Driving](../../ICCV2025/multimodal_vlm/fine-grained_evaluation_of_large_vision-language_models_in_autonomous_driving.md)
- [\[CVPR 2025\] FLAIR: VLM with Fine-grained Language-informed Image Representations](../../CVPR2025/multimodal_vlm/flair_vlm_with_fine-grained_language-informed_image_representations.md)

</div>

<!-- RELATED:END -->
