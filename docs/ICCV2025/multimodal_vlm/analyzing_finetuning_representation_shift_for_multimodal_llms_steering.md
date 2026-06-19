---
title: >-
  [论文解读] Analyzing Finetuning Representation Shift for Multimodal LLMs Steering
description: >-
  [ICCV 2025][多模态VLM][MLLM可解释性] 提出一个无需训练的框架，通过概念级别分析揭示多模态大语言模型微调时的表征偏移，并利用偏移向量实现模型行为的轻量级引导（去偏、安全控制）。 多模态大语言模型（MLLMs）在图像描述、视觉问答等任务上表现出色，但理解其内部行为仍是挑战。现有工作大多以"事后分析"的方式考…
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "MLLM可解释性"
  - "概念漂移"
  - "表征偏移"
  - "模型引导"
  - "去偏"
---

# Analyzing Finetuning Representation Shift for Multimodal LLMs Steering

**会议**: ICCV 2025  
**arXiv**: [2501.03012](https://arxiv.org/abs/2501.03012)  
**代码**: [项目页面](https://pegah-kh.github.io/projects/lmm-finetuning-analysis-and-steering/)  
**领域**: 多模态VLM  
**关键词**: MLLM可解释性, 概念漂移, 表征偏移, 模型引导, 去偏

## 一句话总结

提出一个无需训练的框架，通过概念级别分析揭示多模态大语言模型微调时的表征偏移，并利用偏移向量实现模型行为的轻量级引导（去偏、安全控制）。

## 研究背景与动机

多模态大语言模型（MLLMs）在图像描述、视觉问答等任务上表现出色，但理解其内部行为仍是挑战。现有工作大多以"事后分析"的方式考察最终训练好的模型，忽略了 **微调过程中隐含概念发生的动态变化**。例如，当对图像描述模型做"场所聚焦"微调后，原本与"人"相关的概念可能悄然引入场所关键词，甚至部分概念完全消失或新概念涌现。

作者指出两个核心问题：

**微调引入不可控的概念漂移**：不同概念受微调影响程度不同，一些被精化，一些被彻底重塑。这些变化可能引入偏见或不安全行为。

**缺少解释和控制手段**：现有可解释方法多面向单模态（视觉或文本），对 MLLM 几乎空白；而引导（steering）方法也仅在纯文本 LLM 上有探索。

因此本文希望：(a) 在可读的概念层面监控微调带来的变化，(b) 利用发现的偏移向量以零额外训练代价引导 MLLM 行为。

## 方法详解

### 整体框架

框架分三部分：**概念提取与比较**（Section 3.1）→ **微调概念偏移恢复**（Section 3.2）→ **模型引导应用**（Section 3.3）。

### 关键设计

1. **概念提取（K-Means 字典学习）**  
   给定一组图像，通过 MLLM 某一层的残差流提取表征 $\bm{Z} \in \mathbb{R}^{D \times M}$，对其做 K-Means 分解 $\bm{Z} \approx \bm{U}\bm{V}$。$\bm{U}$ 的每一列 $\bm{u}_k$ 即一个概念，通过 **图像 grounding**（最大激活样本）和 **文本 grounding**（unembedding 矩阵映射到词表）进行人类可读解释。概念间相似度用 **Text Grounding Overlap (T-Overlap)** 衡量。

2. **概念偏移向量（Concept Shift Vectors）**  
   对比原始模型 $f^a$ 与微调模型 $f^b$ 在同一数据集上的表征，为每个原始概念 $\bm{u}_k^a$ 找到其关联样本集 $\bm{A}_k$，计算偏移向量：
    $\bm{\Delta}_k^{a \to b}(\bm{u}_k^a) = \frac{1}{|\bm{A}_k|} \sum_{m \in \bm{A}_k} (\bm{b}_m - \bm{a}_m)$
   然后通过 $\bm{u}_k^s = \bm{u}_k^a + \alpha \bm{\Delta}_k^{a \to b}$ 得到偏移概念。关键发现：**个体偏移一致性越高，概念恢复越好**（正相关且统计显著）。

3. **粗粒度与细粒度模型引导**  

    - **粗粒度引导**：计算目标样本集与原始样本集均值之差作为引导向量 $\bm{s}_c$，在推理时加到特征上 $\tilde{f_l}(x) = f_l(x) + \alpha \bm{s}_c$。
    - **细粒度引导**：分解概念后，计算概念对之间的差 $\bm{s}_{ij}^f = \bm{u}_j - \bm{u}_i$，仅对激活特定概念的样本施加引导，实现定向修改（如将"yes"答案引导为"no"）。

### 损失函数 / 训练策略

本方法 **无需训练**。偏移向量和引导向量均在推理时直接加到残差流表征上，不改变任何模型参数。$\alpha$ 默认为 1。引导在深层（尤其最后一层）最有效，因为深层概念近似线性可分。

## 实验关键数据

### 主实验（表格）

| 引导方向 | Yes/No 准确率 | Number 准确率 | Other 准确率 | 原始答案变化 | 目标答案变化 |
|---------|-------------|-------------|------------|-----------|-----------|
| 无引导 | 90.82 | 58.47 | 71.10 | 0 | 0 |
| Yes→No | 69.03 | 56.82 | 68.99 | -828 | +828 |
| 1→3 | 90.71 | 54.52 | 71.12 | -215 | +144 |
| White→Black | 90.40 | 58.42 | 58.36 | -98 | +441 |

在 VQAv2 上，引导可显著改变目标答案数量，同时 **其他答案类型的准确率和数量基本不变**，体现了引导的定向性。

### 消融实验（表格）

| 模型 | 总性别表达数 | 方法 | 性别→中性转换数 |
|------|-----------|------|--------------|
| LLaVA-1.5 | 794 | 粗粒度引导 | 232 |
| LLaVA-1.5 | 794 | 细粒度引导 | **632** |
| Idefics2 | 815 | 粗粒度引导 | 237 |
| Idefics2 | 815 | 细粒度引导 | **315** |
| Qwen2-VL | 926 | 粗粒度引导 | 134 |
| Qwen2-VL | 926 | 细粒度引导 | **300** |

细粒度引导在性别去偏任务中远优于粗粒度引导，且在三个不同 MLLM 上均有效。

### 关键发现

- 微调使概念逐渐偏离原始状态（T-Overlap 随迭代递减），不同概念受影响程度差异大。
- 偏移向量能部分恢复微调后的概念（$\alpha=1$ 通常最优），恢复效果与个体偏移一致性正相关。
- 安全引导：Qwen2-VL 的 ASR（攻击成功率）从 45/100 降至 5/100，效果突出。

## 亮点与洞察

- **概念即向量**的线性假设在 MLLM 深层得到实证验证，为轻量引导奠定理论基础。
- 将可解释性与可控性统一在同一框架中：先理解（分析偏移），后操控（应用偏移向量）。
- 方法零训练成本、即插即用，可扩展到多种 MLLM（LLaVA、Idefics2、Qwen2-VL）。

## 局限与展望

- 依赖线性表征假设，对非线性编码的特征可能失效。
- 概念匹配使用余弦相似度 + 最优传输，更复杂的匹配算法可能改善结果。
- 引导强度 $\alpha$ 增大时，输出多样性下降甚至退化为重复词。
- 仅在图像描述/VQA 场景验证，更复杂的多模态交互（视频、对话）有待探索。

## 相关工作与启发

- 与 ReFT 等表征微调方法思路相通，但本文不需要任何额外训练。
- 与 SAE（稀疏自编码器）用于 LLM 可解释性的路线互补：SAE 更精细但计算成本高，本文方法更轻量。
- 引导向量方法（Activation Addition）此前仅在文本 LLM 上验证，本文首次扩展至多模态场景。

## 评分

- 新颖性: ⭐⭐⭐⭐ （首个系统性分析 MLLM 微调概念偏移并实现引导的工作）
- 实验充分度: ⭐⭐⭐⭐ （多模型/多任务/多场景，包含去偏和安全应用）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，图表丰富）
- 价值: ⭐⭐⭐⭐ （零成本引导对 MLLM 安全和去偏有实际意义）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Learning to Steer: Input-dependent Steering for Multimodal LLMs](../../NeurIPS2025/multimodal_vlm/learning_to_steer_input-dependent_steering_for_multimodal_llms.md)
- [\[ICCV 2025\] Hints of Prompt: Enhancing Visual Representation for Multimodal LLMs in Autonomous Driving](hints_of_prompt_enhancing_visual_representation_for_multimodal_llms_in_autonomou.md)
- [\[ICCV 2025\] Controlling Multimodal LLMs via Reward-guided Decoding](controlling_multimodal_llms_via_rewardguided_decoding.md)
- [\[ICCV 2025\] Enrich and Detect: Video Temporal Grounding with Multimodal LLMs](enrich_and_detect_video_temporal_grounding_with_multimodal_llms.md)
- [\[ICCV 2025\] Visual Chronicles: Using Multimodal LLMs to Analyze Massive Collections of Images](visual_chronicles_using_multimodal_llms_to_analyze_massive_collections_of_images.md)

</div>

<!-- RELATED:END -->
