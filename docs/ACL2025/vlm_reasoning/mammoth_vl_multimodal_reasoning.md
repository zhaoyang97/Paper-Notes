---
title: >-
  [论文解读] MAmmoTH-VL: Eliciting Multimodal Reasoning with Instruction Tuning at Scale
description: >-
  [多模态VLM] 提出一种可扩展、低成本的方法，仅使用开源模型构建含 1200 万条富含中间推理过程 (CoT) 的多模态指令微调数据集 MAmmoTH-VL-Instruct，训练的 MAmmoTH-VL-8B 在推理基准上达到 SOTA（MathVerse +8.1%, MMMU-Pro +7%, MuirBench +13.3%）。
tags:
  - "多模态VLM"
---

# MAmmoTH-VL: Eliciting Multimodal Reasoning with Instruction Tuning at Scale

| 会议 | arXiv | 代码 | 领域 | 关键词 |
|------|-------|------|------|--------|
| ACL 2025 | [2412.05237](https://arxiv.org/abs/2412.05237) | [项目主页](https://mammoth-vl.github.io) | multimodal_vlm | 多模态推理, 指令微调, CoT, 数据重写, 大规模训练数据 |

## 一句话总结

提出一种可扩展、低成本的方法，仅使用开源模型构建含 1200 万条富含中间推理过程 (CoT) 的多模态指令微调数据集 MAmmoTH-VL-Instruct，训练的 MAmmoTH-VL-8B 在推理基准上达到 SOTA（MathVerse +8.1%, MMMU-Pro +7%, MuirBench +13.3%）。

## 研究背景与动机

- **现有问题**: 现有多模态指令微调数据集主要来自学术 VQA 数据集（如 VQA, AI2D, ChartQA），这些数据集目标简单，仅提供短语级答案而无中间推理过程，限制了模型的推理能力。
- **核心差距**: Chain-of-Thought (CoT) 推理在纯文本 LLM 中效果显著，但构建大规模多模态 CoT 数据集面临两大障碍：(1) 确保指令的多样性和复杂性, (2) 生成带详细理由的连贯响应。人工标注成本过高，依赖 GPT-4 等闭源模型又涉及高成本和版权问题。
- **研究动机**: 用开源模型实现低成本、可扩展的多模态 CoT 数据集构建，降低开源社区的门槛。

## 方法详解

### 整体框架

三步数据构建流水线：
1. **数据收集与分类**: 从 153 个公开数据集收集，按10大类（General, OCR, Chart, Caption, Domain-specific, Code&Math, Language, Detection, Multi-Image, Video）组织
2. **指令数据重写**: 使用开源模型将短答案扩展为含 CoT 推理的详细响应
3. **自过滤**: 用同一 MLLM 作为裁判 (Model-as-Judge) 过滤幻觉内容

### 关键设计

1. **数据源三级分组**:
    - Group A (58 个): 高质量，直接保留原始数据
    - Group B (60 个): 有潜力但回答简略，进行重写增强
    - Group C (35 个): 过于模糊/简短，直接丢弃

2. **任务感知重写策略**: 为每个数据类别设计定制化 prompt。Caption 类数据用纯文本模型 (Llama-3-70B) 生成任务导向 QA 对；其他类型用多模态模型 (InternVL2-Llama3-76B) 确保视觉-文本对齐。

3. **数据混合比例**: 70% 重写数据 + 30% 原始数据，t-SNE 分析表明重写数据既保留原始分布的核心特征，又拓展了覆盖范围。

### 训练配置

三阶段训练（基于 LLaVA-OneVision 架构）：
- **Stage-1**: 语言-图像对齐预训练（558K, 仅训练 Projector）
- **Stage-2**: 单图视觉指令微调（10M, 全参数训练）
- **Stage-3**: One Vision 多图/视频微调（2M, 全参数训练）

LLM 骨干: Qwen2.5-7B-Instruct, 视觉编码器: SigLIP-so400m-patch14-384

## 实验

### 主实验：多学科知识与数学推理

| 模型 | MMStar | MMMU (val) | MMMU-Pro | MathVerse | MathVista |
|------|--------|-----------|----------|-----------|-----------|
| GPT-4o | 64.7 | 69.1 | 49.7 | 50.2 | 63.8 |
| Qwen2-VL-7B | 60.7 | 52.1 | 26.9 | 28.2 | 58.2 |
| LLaVA-OV-7B | 61.7 | 48.8 | 18.7 | 26.2 | 63.2 |
| Llava-CoT-11B | 57.6 | 48.9 | 18.5 | 24.2 | 54.8 |
| **MAmmoTH-VL-8B** | **63.0** | **50.8** | **25.3** | **34.2** | **67.6** |
| Δ vs 最佳开源 (~10B) | +1.3 | +1.9 | **+7.1** | **+8.1** | +4.4 |

### 文档/图表理解

| 模型 | AI2D | ChartQA | DocVQA | RealWorldQA |
|------|------|---------|--------|-------------|
| LLaVA-OV-7B | 81.4 | 80.0 | 87.5 | 66.3 |
| InternVL-2-8B | 83.8 | 83.3 | 91.6 | 64.4 |
| **MAmmoTH-VL-8B** | **84.0** | **86.2** | **93.7** | 69.9 |
| Δ vs 最佳开源 (~10B) | +2.4 | +2.1 | +1.6 | +0.6 |

### 关键消融发现

1. **自过滤至关重要**: OCR 和图表数据的幻觉过滤率最高，移除过滤步骤会导致模型性能显著下降。
2. **重写数据质量提升**: 重写后数据在信息含量和相关性评分上均高于原始数据 (5 分制)。
3. **数据规模效应显著**: 从 2M 到 10M 训练数据，性能持续提升，说明大规模 CoT 数据的可扩展性。
4. **非推理任务也受益**: 非推理基准上也有最高 4% 的提升，表明 CoT 训练的泛化效益。

## 亮点

- 仅用开源模型构建 12M 大规模多模态 CoT 数据集，打破了对 GPT-4 等闭源模型的依赖
- 三步流水线（收集-重写-过滤）简洁可扩展，方法论可复用于其他领域
- 8B 规模模型在推理密集型任务上大幅超越同规模甚至更大规模模型（MathVerse +8.1%）
- Model-as-Judge 自过滤方法与人类评估一致性达 Kappa 0.64（良好水平）

## 局限性

- 自过滤使用同一生成模型作为裁判，可能对自身生成的特定错误模式存在盲点
- 重写过程中 OCR/图表类数据幻觉率较高，说明开源 MLLM 在细粒度视觉理解上仍有不足
- 训练成本虽低于使用 GPT-4，但 10M 规模训练仍需大量计算资源
- 数据收集依赖已有公开数据集，新领域/新任务的覆盖可能不足

## 相关工作

- **多模态指令微调**: LLaVA (Liu et al. 2024b) 开创视觉指令微调范式；LLaVA-OneVision (Li et al. 2024b) 扩展到多图/视频
- **推理增强**: Chain-of-Thought (Wei et al. 2022) 逐步推理；Llava-CoT (Xu et al. 2024a) 在单一模型中引入 CoT 但依赖 GPT-4 数据
- **数据质量与过滤**: Cambrian (Tong et al. 2024) 多源数据融合训练；InternVL2 (Chen et al. 2023b) 大规模预训练；本工作强调开源模型自过滤的可行性

## 评分

| 维度 | 分数 (1-10) |
|------|------------|
| 创新性 | 7 |
| 技术深度 | 7 |
| 实验充分性 | 9 |
| 写作质量 | 8 |
| **综合** | **7.5** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] MetaMorph: Multimodal Understanding and Generation via Instruction Tuning](../../ICCV2025/multimodal_vlm/metamorph_multimodal_understanding_and_generation_via_instruction_tuning.md)
- [\[ICCV 2025\] MM-IFEngine: Towards Multimodal Instruction Following](../../ICCV2025/multimodal_vlm/mm-ifengine_towards_multimodal_instruction_following.md)
- [\[ICCV 2025\] R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization](../../ICCV2025/multimodal_vlm/r1-vl_learning_to_reason_with_multimodal_large_language_models_via_step-wise_gro.md)
- [\[ICCV 2025\] OrderChain: Towards General Instruct-Tuning for Stimulating the Ordinal Understanding Ability of MLLM](../../ICCV2025/multimodal_vlm/orderchain_towards_general_instruct-tuning_for_stimulating_the_ordinal_understan.md)
- [\[ACL 2025\] Can Multimodal Large Language Models Understand Spatial Relations?](spatialmqa_mllm_spatial_relations.md)

</div>

<!-- RELATED:END -->
