---
title: >-
  [论文解读] ViSpec: Accelerating Vision-Language Models with Vision-Aware Speculative Decoding
description: >-
  [NeurIPS 2025][多模态VLM][推测解码] 针对VLM推测解码（speculative decoding）中草稿模型难以处理冗余视觉token的问题，提出ViSpec框架，通过视觉适配器压缩图像token+全局视觉特征注入+合成训练数据，首次在VLM推测解码中实现了显著加速（最高3.22×）。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "推测解码"
  - "VLM加速"
  - "图像token压缩"
  - "草稿模型"
  - "推理加速"
---

# ViSpec: Accelerating Vision-Language Models with Vision-Aware Speculative Decoding

**会议**: NeurIPS 2025  
**arXiv**: [2509.15235](https://arxiv.org/abs/2509.15235)  
**代码**: [GitHub](https://github.com/KangJialiang/ViSpec)  
**领域**: 多模态VLM  
**关键词**: 推测解码, VLM加速, 图像token压缩, 草稿模型, 推理加速

## 一句话总结

针对VLM推测解码（speculative decoding）中草稿模型难以处理冗余视觉token的问题，提出ViSpec框架，通过视觉适配器压缩图像token+全局视觉特征注入+合成训练数据，首次在VLM推测解码中实现了显著加速（最高3.22×）。

## 研究背景与动机

推测解码在纯文本LLM中已取得3-4×加速，但在VLM中仅实现<1.5×的微弱加速（如SpecLLaVA仅1.5×）。作者深入分析了这一差距的根本原因：

**文本 vs. 视觉数据的本质差异**：文本经过千年进化，信息密度高且抽象；而图像虽然视觉丰富，但包含大量冗余（如相同颜色的色块）。这导致浅层草稿模型难以从大量冗余中提取关键信息。

**理论分析**：作者给出了形式化证明。考虑 $R+1$ 个token，其中 $R$ 个为相同的冗余token（如统一色块的图像patch），1个为独特token。对于单层Transformer，当 $R \to \infty$ 时，独特token的注意力权重趋近于零：

$$\alpha_{iu} = \frac{\exp(B)}{R\exp(A) + \exp(B)} \to 0$$

输出退化为对冗余token的平均，完全忽略了有价值的独特视觉信息。理论上处理嵌套复杂度 $K$ 需要 $K+1$ 层网络，但草稿模型通常很浅。

**Lost-in-the-Middle问题**：当生成的文本序列变长时，位于中间位置的图像token会被淹没，浅层草稿模型的U型性能曲线导致视觉信息进一步丢失。

## 方法详解

### 整体框架

ViSpec由三个核心组件组成：(1) 视觉适配器压缩图像embedding，(2) 全局视觉特征注入，(3) 合成长回答训练数据集生成。草稿模型接收压缩后的视觉token和增强后的文本token，预测下一个token序列，目标模型并行验证。

### 关键设计

1. **视觉适配器（Vision Adaptor）**：受Q-Former启发的轻量级Transformer编码器。使用固定数量的可学习query向量作为查询，原始视觉特征作为key和value。通过交叉注意力机制，将数千个image embedding压缩为少量（实验中仅1个）紧凑视觉token，并保留原始图像的位置信息。最终实验表明1个压缩embedding即可充分捕获视觉信息，增加数量对接受长度 $\tau$ 影响甚微，反而因增加草稿模型计算量而降低实际加速比。

2. **全局视觉特征注入（Global Visual Feature Integration）**：从视觉适配器的最终输出中提取一个全局特征向量 $g$，通过学习的投影矩阵 $W_g$ 注入到所有后续文本token的隐状态中：

$$f_t^{\text{aug}} = f_t + W_g g$$

这确保草稿模型在生成长文本时始终能访问全局视觉上下文，有效缓解了Lost-in-the-Middle效应。

3. **训练数据生成与多token预测**：针对公开多模态数据集缺少长回答样本的问题，通过修改prompt（如追加"请用至少1000字回答"）让目标VLM生成长回答，构造合成训练数据。采用采样策略（非贪婪）打破隐状态与embedding间的一一对应关系，结合多token预测（受DeepSeek-V3启发），防止草稿模型产生shortcut learning。

$$L = \text{CrossEntropy}(p_i, \hat{p}_i)$$

### 损失函数 / 训练策略

- 两阶段训练：先在68K ShareGPT纯文本数据上训练文本基础，再在多模态数据上微调
- ViSpec草稿模型为单层，镜像目标模型的decoder层结构
- 学习率3e-6，AdamW优化器，batch size 8，训练20 epoch
- 推理时使用EAGLE-2的上下文感知动态草稿树（30 draft tokens，深度3，8节点扩展）

## 实验关键数据

### 主实验（Temperature=0, LLaVA-1.6-7B）

| 方法 | SQA加速 | TextVQA加速 | COCO Caps加速 | GQA加速 | 平均加速 | 平均 $\tau$ |
|------|--------|-----------|-------------|--------|---------|------------|
| Medusa | 1.41× | 1.46× | 1.61× | 1.29× | 1.42× | 0.72 |
| EAGLE-2 | 2.14× | 1.25× | 1.80× | 1.64× | 1.62× | 1.31 |
| **ViSpec** | **2.37×** | **2.90×** | **3.22×** | **2.22×** | **2.58×** | **2.98** |

### 跨模型验证（Temperature=0, 平均）

| 模型 | Medusa | EAGLE-2 | ViSpec |
|------|--------|---------|--------|
| LLaVA-1.6-7B | 1.42× | 1.62× | **2.58×** |
| LLaVA-1.6-13B | 1.48× | 1.86× | **2.38×** |
| Qwen2.5-VL-3B | 1.14× | 1.39× | **1.87×** |
| Qwen2.5-VL-7B | 1.11× | 1.40× | **1.80×** |

### 消融实验

| 组件 | COCO Caps加速 | GQA加速 | MME加速 |
|------|-------------|--------|--------|
| EAGLE-2基线 | 1.80× | 1.64× | 1.68× |
| +图像压缩 | 2.37× (+30%) | 1.92× | 1.83× |
| +全局视觉注入 | 2.42× (+7%) | 2.03× | 1.95× |
| +数据集生成 | **3.22×** (+30%) | **2.22×** | **2.55×** |

| 压缩embedding数 | COCO τ | COCO加速 | GQA τ | GQA加速 |
|-----------------|--------|---------|-------|--------|
| 1 | 3.30 | **3.22×** | 2.88 | **2.22×** |
| 4 | 3.24 | 3.24× | 2.84 | 2.24× |
| 64 | 3.25 | 2.71× | 2.86 | 1.91× |

### 关键发现

- ViSpec在所有模型和任务上一致超越Medusa和EAGLE-2，加速比范围1.37×-3.22×
- 更长的输出序列带来更高的加速比（TextVQA 353.58 tokens → 2.90×，GQA 46.25 tokens → 2.22×）
- LLaVA系列的加速效果优于Qwen2.5-VL，后者更大的词表增加了token预测复杂度
- 视觉适配器不增加显著的prefill延迟（测量噪声级别）
- 在视频任务（MSVD-QA, MVBench）上无需视频特定训练即可获得1.32×-1.46×加速

## 亮点与洞察

1. **理论与实践结合**：从注意力机制角度清晰解释了为什么浅层草稿模型处理冗余视觉token效果差，动机论证扎实
2. **极简设计原则**：仅用1个压缩视觉embedding即可捕获关键信息，体现了"less is more"
3. **全局特征注入思路巧妙**：简单但有效地解决了长文本生成中的视觉遗忘问题
4. **首次在VLM推测解码中突破2×加速壁垒**，建立了该方向的新基准

## 局限与展望

- 绝对加速比仍落后于纯文本推测解码的最优方法
- 训练数据依赖目标模型生成，合成数据的质量和多样性有限
- 视觉编码器架构未做优化（如动态patch缩减、神经压缩）
- Qwen2.5-VL系列加速效果明显弱于LLaVA系列，大词表场景需要进一步研究
- 高分辨率图像由于prefill时间增长，端到端加速比被稀释

## 相关工作与启发

- EAGLE系列（目标感知特征注入）为ViSpec的草稿模型隐状态输入提供了灵感
- Q-Former（BLIP-2）的查询-key-value压缩思路被用于视觉适配器设计
- DeepSeek-V3的多token预测策略用于防止训练时的shortcut learning
- 为视频VLM的加速提供了直接的扩展方向

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次针对VLM设计推测解码框架，理论分析+方案设计完整
- 实验充分度: ⭐⭐⭐⭐⭐ 4个模型×8个数据集×2个温度，消融分析详尽
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，但符号较多
- 实用价值: ⭐⭐⭐⭐⭐ 直接可部署的VLM推理加速方案，代码开源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Balanced Token Pruning: Accelerating Vision Language Models Beyond Local Optimization](balanced_token_pruning_accelerating_vision_language_models_b.md)
- [\[NeurIPS 2025\] Enhancing Vision-Language Model Reliability with Uncertainty-Guided Dropout Decoding](enhancing_visionlanguage_model_reliability_with_uncertaintyg.md)
- [\[NeurIPS 2025\] Test-Time Spectrum-Aware Latent Steering for Zero-Shot Generalization in Vision-Language Models](test-time_spectrum-aware_latent_steering_for_zero-shot_generalization_in_vision-.md)
- [\[ICCV 2025\] FOLDER: Accelerating Multi-modal Large Language Models with Enhanced Performance](../../ICCV2025/multimodal_vlm/folder_accelerating_multi-modal_large_language_models_with_enhanced_performance.md)
- [\[CVPR 2025\] Taxonomy-Aware Evaluation of Vision-Language Models](../../CVPR2025/multimodal_vlm/taxonomy-aware_evaluation_of_vision-language_models.md)

</div>

<!-- RELATED:END -->
