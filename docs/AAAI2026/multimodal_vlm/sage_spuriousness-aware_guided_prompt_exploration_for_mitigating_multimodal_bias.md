---
title: >-
  [论文解读] SAGE: Spuriousness-Aware Guided Prompt Exploration for Mitigating Multimodal Bias
description: >-
  [AAAI 2026][多模态VLM][虚假相关性] 提出SAGE，一种无需训练、微调或外部标注的提示选择方法，通过计算提示模板在类别间的分离度得分来缓解CLIP模型中的多模态虚假偏差，在四个基准+五个骨干模型上一致提升最差组准确率（WGA）和调和均值（HM）。 领域现状 CLIP等预训练视觉-语言模型通过在共享嵌入空间对齐…
tags:
  - "AAAI 2026"
  - "多模态VLM"
  - "虚假相关性"
  - "CLIP"
  - "零样本分类"
  - "提示选择"
  - "鲁棒性"
---

# SAGE: Spuriousness-Aware Guided Prompt Exploration for Mitigating Multimodal Bias

**会议**: AAAI 2026  
**arXiv**: [2511.13005](https://arxiv.org/abs/2511.13005)  
**代码**: [https://github.com/wenqian-ye/spurious_vlm](https://github.com/wenqian-ye/spurious_vlm)  
**领域**: 多模态VLM  
**关键词**: 虚假相关性, CLIP, 零样本分类, 提示选择, 鲁棒性

## 一句话总结
提出SAGE，一种无需训练、微调或外部标注的提示选择方法，通过计算提示模板在类别间的分离度得分来缓解CLIP模型中的多模态虚假偏差，在四个基准+五个骨干模型上一致提升最差组准确率（WGA）和调和均值（HM）。

## 研究背景与动机

### 领域现状
CLIP等预训练视觉-语言模型通过在共享嵌入空间对齐图像和文本取得了强大的零样本分类能力。然而，CLIP模型常常发展出**多模态虚假偏差**（multimodal spurious bias），即不合理地依赖非本质特征进行预测。

### 核心痛点

**虚假相关性本质**：预训练数据中，"陆鸟"（landbird）与"陆地背景"频繁共现，导致CLIP将"陆地背景"对齐到"landbird"的文本表示。当遇到"陆地背景中的水鸟"时，模型错误预测为"陆鸟"。这种跨模态虚假关联严重损害了分布外数据上的零样本泛化能力。

### 已有方法的局限
- **微调方法**（Yang et al. 2023; You et al. 2024）：需要下游标注数据，无法处理零样本场景
- **ROBOSHOT**（Adila et al. 2024）：需要用LLM（大语言模型）为每个任务明确指定虚假属性
- **TIE***：直接使用虚假属性生成伪标签，同样依赖先验知识

**核心矛盾**：现有方法要么需要训练数据，要么需要先验知识——这都破坏了CLIP"开箱即用"的零样本优势。

### 本文切入点
能否仅通过**选择合适的文本提示模板**来缓解虚假偏差？SAGE的核心洞察是：**分离度越大的提示越能捕捉类别核心语义，从而减少对虚假特征的依赖**。当某个提示下不同类别的相似度差异很大时，说明这个提示更关注本质特征；反之，相似度接近则可能是虚假特征在"混淆"分类。

## 方法详解

### 整体框架
SAGE的流程极其简洁：(a) 准备M个候选提示模板；(b) 对每张测试图像，计算每个提示在类别间的分离度；(c) 选取分离度最高的K个提示进行集成零样本分类。

### 关键设计

1. **多模态虚假偏差的理论分析**:

    - 引入潜在虚假特征 $\mathbf{u}_s$ 的概念
    - **定义1（多模态虚假偏差）**：当 $p(\mathbf{u}_s|\mathbf{u}_1) \approx 1$ 且 $p(\mathbf{u}_s|\mathbf{v}) \approx 1$ 时，虚假特征与类别标签和图像表示高度关联
    - **定理1**：证明在虚假偏差下，模型被偏向预测为错误类别 $c_1$，因为 $\frac{p(\mathbf{u}_1|\mathbf{v})}{p(\mathbf{u}_2|\mathbf{v})} \approx \frac{p(\mathbf{u}_s|\mathbf{u}_1)p(\mathbf{u}_1)}{p(\mathbf{u}_2|\mathbf{v})} > 1$
    - **关键推论**：提示模板直接控制 $p(\mathbf{u}_s|\mathbf{u}_1)$。受虚假偏差影响大的提示会使类间相似度趋近→分离度低；反之分离度高

2. **分离度得分（Separation Score）**:

    - 对提示模板 $T_j$ 和图像 $x_n$，计算：$\sigma_j^n = \max_i \frac{\mathbf{v}_n^T\mathbf{u}_i^j}{\|\mathbf{v}_n\|_2\|\mathbf{u}_i^j\|_2} - \min_i \frac{\mathbf{v}_n^T\mathbf{u}_i^j}{\|\mathbf{v}_n\|_2\|\mathbf{u}_i^j\|_2}$
    - 即：该提示下最高类别相似度与最低类别相似度之差
    - 高分离度→提示更能区分核心语义→更不受虚假特征干扰

3. **模板选择与集成推理**:

    - 对每张图像，按分离度排序所有M个提示模板
    - 选取top-K个提示构建K个零样本分类器
    - 集成预测：$\hat{y_n} = \arg\max_i \frac{1}{K}\sum_{k=1}^K \frac{\mathbf{v}_n^T\mathbf{u}_i^k}{\|\mathbf{v}_n\|_2\|\mathbf{u}_i^k\|_2}$
    - 默认K=1（即选单个最优提示）

### 损失函数 / 训练策略
SAGE完全**无需训练**。不涉及任何参数更新、微调或外部模型调用。仅在推理时进行提示选择计算。

## 实验关键数据

### 主实验

在四个基准数据集上的平均结果（跨5个骨干模型）：

| 方法 | Waterbirds WGA | Waterbirds HM | CelebA WGA | CelebA HM | PACS WGA | VLCS WGA |
|------|---------------|---------------|------------|-----------|----------|----------|
| ZS (基线) | 36.7 | 51.1 | 75.3 | 78.1 | 75.5 | 23.0 |
| ROBOSHOT | 41.5 | 52.5 | 79.5 | 81.9 | 78.0 | 30.1 |
| TIE* | 38.4 | 52.8 | 69.8 | 73.1 | 77.2 | 31.7 |
| **SAGE** | **44.9** | **59.7** | **80.6** | **82.0** | **81.9** | **33.8** |

SAGE在所有数据集上WGA和HM均领先，且**不需要任何先验知识**。

### 消融实验（对比随机选择和全集成）

| 策略 | Waterbirds WGA | Waterbirds HM | CelebA WGA | CelebA HM |
|------|---------------|---------------|------------|-----------|
| 全集成(K=80) | 36.2 | 51.6 | 73.2 | 76.5 |
| 随机选择 | 40.1 | 55.1 | 74.9 | 77.3 |
| **SAGE** | **44.9** | **59.7** | **80.6** | **82.0** |

关键发现：**全集成（K=80）效果最差**，因为集成了大量受虚假偏差影响的提示反而稀释了好提示的效果。

### 各模型上的具体提升（Waterbirds WGA）

| 模型 | ZS | SAGE | 提升 |
|------|-----|------|------|
| CLIP-RN-50 | 41.0 | 41.3 | +0.3 |
| CLIP-ViT-B/32 | 27.5 | **46.0** | **+18.5** |
| CLIP-ViT-L/14 | 27.6 | **47.8** | **+20.2** |
| ALIGN | 50.0 | 47.0 | -3.0 |
| AltCLIP | 37.2 | **42.6** | **+5.4** |

### 关键发现

1. **分离度得分是虚假偏差的有效代理指标**：理论和实验一致表明，高分离度的提示更关注核心语义
2. **提示选择的重要性**：不同提示模板对虚假偏差的敏感度差异巨大——选对提示就能大幅提升鲁棒性
3. **全集成反而有害**：集成80个提示比单个最优提示效果差，说明"多即是好"在虚假偏差场景不成立
4. **模型越大、提升越明显**：CLIP-ViT-L/14的WGA从27.6涨到47.8（+20.2），而ResNet-50仅+0.3
5. **无需任何先验知识**：SAGE是唯一完全不需要虚假属性信息的方法，但效果最好

## 亮点与洞察

- **理论驱动的方法设计**：从形式化定义虚假偏差到证明分离度与偏差的关系，理论推导完整且优雅
- **极致简洁**：整个方法只需计算余弦相似度的差值并排序，零训练成本
- **通用性**：适用于任何基于CLIP的零样本视觉-语言模型，无需修改
- **调和均值（HM）作为评价指标**的提出：同时衡量平均准确率和最差组准确率，比单独报告两者更全面
- 揭示了一个反直觉现象：**更多提示的集成反而降低鲁棒性**

## 局限与展望

- SAGE依赖预定义的候选提示集合，提示集质量可能影响效果
- 在ALIGN模型上Waterbirds的WGA略有下降（50.0→47.0），说明并非所有模型-数据组合都适用
- K=1的默认设置可能在某些场景不是最优，需要数据集级别的超参调优
- 计算开销随候选提示数M和类别数C线性增长，对大规模分类可能需要优化
- 仅在分类任务上验证，未扩展到检索、分割等其他CLIP应用场景

## 相关工作与启发

- 与ROBOSHOT和TIE*的本质区别：后者需要"知道什么是虚假特征"，SAGE完全不需要
- 与提示工程的联系：SAGE从鲁棒性角度为"如何选提示"提供了理论依据
- 虚假偏差问题在单模态中已被广泛研究（DRO、重加权等），本文将其扩展到多模态零样本场景
- 启发：未来可以将分离度得分用于**自动化提示筛选**或**提示学习的正则化**

## 评分
- 新颖性: ⭐⭐⭐⭐（理论分析新颖，方法极为简洁）
- 实验充分度: ⭐⭐⭐⭐⭐（4个数据集×5个模型×3个对比方法+消融）
- 写作质量: ⭐⭐⭐⭐⭐（理论推导清晰，动机链完整）
- 价值: ⭐⭐⭐⭐（对CLIP零样本部署有直接实用价值）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Mitigating Perceptual Judgment Bias in Multimodal LLM-as-a-Judge via Perceptual Perturbation and Reward Modeling](../../ICML2026/multimodal_vlm/mitigating_perceptual_judgment_bias_in_multimodal_llm-as-a-judge_via_perceptual_.md)
- [\[ICML 2025\] Understanding and Mitigating Miscalibration in Prompt Tuning for Vision-Language Models](../../ICML2025/multimodal_vlm/understanding_and_mitigating_miscalibration_in_prompt_tuning_for_vision-language.md)
- [\[AAAI 2026\] Positional Bias in Multimodal Embedding Models: Do They Favor the Beginning, the Middle, or the End?](positional_bias_in_multimodal_embedding_models_do_they_favor_the_beginning_the_m.md)
- [\[CVPR 2026\] LOREAL: Mitigating Low-Resolution Challenges in Vision-Language Models with Attribute-driven Prompt Self-Distillation](../../CVPR2026/multimodal_vlm/loreal_mitigating_low-resolution_challenges_in_vision-language_models_with_attri.md)
- [\[ICLR 2026\] Capacity-Aware Inference: Mitigating the Straggler Effect in Mixture of Experts](../../ICLR2026/multimodal_vlm/capacity-aware_inference_mitigating_the_straggler_effect_in_mixture_of_experts.md)

</div>

<!-- RELATED:END -->
