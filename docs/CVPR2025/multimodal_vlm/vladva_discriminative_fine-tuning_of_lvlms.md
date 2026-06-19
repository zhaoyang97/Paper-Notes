---
title: >-
  [论文解读] VladVA: Discriminative Fine-tuning of LVLMs
description: >-
  [CVPR 2025][多模态VLM][判别式微调] 提出VladVA框架，通过混合短/长caption数据策略、对比损失+自回归损失的联合训练、以及soft prompting+LoRA的参数高效适配，将生成式LVLM（LLaVA）转化为强判别式模型，在图文检索和组合性理解基准上大幅超越CLIP类模型和18B EVA-CLIP。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "判别式微调"
  - "LVLM"
  - "对比学习"
  - "图文检索"
  - "组合性理解"
---

# VladVA: Discriminative Fine-tuning of LVLMs

**会议**: CVPR 2025  
**arXiv**: [2412.04378](https://arxiv.org/abs/2412.04378)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 判别式微调, LVLM, 对比学习, 图文检索, 组合性理解

## 一句话总结

提出VladVA框架，通过混合短/长caption数据策略、对比损失+自回归损失的联合训练、以及soft prompting+LoRA的参数高效适配，将生成式LVLM（LLaVA）转化为强判别式模型，在图文检索和组合性理解基准上大幅超越CLIP类模型和18B EVA-CLIP。

## 研究背景与动机

当前视觉-语言模型有两大流派各有短板：

**对比训练的VLM（如CLIP）**：判别能力强但语言理解有限，表现出"bag of words"行为——即使打乱词序也不影响匹配分数，在组合性理解（空间关系、属性绑定）上表现差。更大的模型和数据集也无法根本解决。

**生成式LVLM（如LLaVA）**：结合视觉编码器和LLM，具备强推理和细粒度理解能力，但自回归训练方式使其不适合直接用于判别式任务（如图文检索）。

关键问题：能否把两者的优势结合？作者发现LVLM天然具有zero-shot判别能力（通过特定prompt提取summary token），但性能远不及CLIP。此前E5-V工作认为图文对比微调有害，VladVA用精心设计的框架证明这一结论是错误的。

## 方法详解

### 整体框架

VladVA采用two-tower架构：图像侧将图像通过完整LVLM（视觉编码器+投影层+LLM）得到图像嵌入 $\mathbf{f}_v$（取last token的hidden state作为summary token），文本侧将文本通过LLM得到文本嵌入 $\mathbf{f}_t$。两者用cosine similarity计算相似度。对比训练在short caption上进行，自回归训练在long caption上进行，通过soft prompt和LoRA实现参数高效。

### 关键设计

1. **数据策略：短/长caption分工协作**:
    - 功能：让模型同时学习粗粒度和细粒度的图文匹配
    - 核心思路：将训练数据按caption长度分为短caption（<30 tokens，标题级）和长caption（30-500 tokens，详细描述级）。短caption用于对比学习教模型做高层图文匹配；长caption用于自回归学习教模型理解细粒度细节和组合关系。缺失类型的图像用BLIP2生成短caption、ShareGPT-4V生成长caption
    - 设计动机：直接用对比损失训练长caption会崩溃——长caption太具体，几乎没有hard negative，loss在几百次迭代内就归零。按长度分工配合不同损失是解决这一矛盾的关键

2. **混合训练损失：对比 + 自回归**:
    - 功能：统一框架内同时强化判别能力和语言理解能力
    - 核心思路：对比损失 $\mathcal{L}_c = \frac{1}{b}\sum_{k=1}^{b}(-\log\frac{\exp(s_v^{k,k})}{\sum_j\exp(s_v^{k,j})} - \log\frac{\exp(s_t^{k,k})}{\sum_j\exp(s_t^{j,k})})$ 应用于短caption的summary token对齐。自回归损失 $\mathcal{L}_{CE} = \sum_{i=1}^{L}\log p_\theta(u_i | \mathbf{x}_v, \mathbf{x}_p^v, \mathbf{x}_{q,<i}^{long})$ 应用于长caption的逐token预测
    - 设计动机：自回归loss有三大优势：(a) 逐token预测是challenging任务，不会崩溃；(b) 预测过程鼓励summary token压缩更多信息；(c) 保持了模型的生成能力

3. **参数高效适配：Soft Prompting + LoRA**:
    - 功能：低成本微调LVLM
    - 核心思路：用可学习向量替换手工prompt的token embedding（用手工prompt embedding初始化），image和text模态使用不同的soft prompt。LLM线性层加LoRA adapter（rank=16, $\alpha$=16）。分析显示训练后soft prompt解码为语义基本不变的句子，仅首尾标记字符改变
    - 设计动机：Soft prompt的核心作用不是改变语义，而是"标记哪个token应该收集判别信息"。LoRA补充了soft prompt有限的表征能力

### 行为变化分析

训练后模型展现三个关键行为变化：(1) summary token与vision token之间的attention map变得更密集——生成模式可以逐步回看vision token，判别模式必须一次性压缩所有信息；(2) output分布熵增加——summary token编码了更丰富的信息；(3) embedding矩阵累积方差更分散——embedding空间利用更充分，对应更高的矩阵秩。

### 损失函数 / 训练策略

总损失 = 短caption对比损失 + 长caption自回归损失，batch内联合优化。训练7 epoch，batch size 1024，学习率 $10^{-4}$，AdamW优化器，余弦调度，最多32块A100 GPU。训练数据约8.1M样本（OpenImages 4M + CC3M 2.8M + ShareGPT-4V 1.3M）。

## 实验关键数据

### 主实验（Zero-shot Image-Text Retrieval R@1）

| 方法 | 参数量 | Flickr IR | COCO IR | nocaps IR | Flickr TR | COCO TR |
|------|--------|-----------|---------|-----------|-----------|---------|
| CLIP (ViT-L) | 0.43B | 67.3 | 37.0 | 48.6 | 87.2 | 58.1 |
| EVA-CLIP (18B) | 18B | 83.3 | 55.6 | 69.3 | 95.3 | 72.8 |
| E5-V (8B) | 8.36B | 79.5 | 52.0 | 65.9 | 88.2 | 62.0 |
| **VladVA (7B)** | **7.06B** | **85.0** | **59.0** | **72.3** | **94.3** | **72.9** |

### 组合性理解（SugarCrepe）

| 类别 | VladVA | EVA-CLIP(18B) | E5-V(8B) | CLIP(ViT-L) | 提升vs EVA |
|------|--------|---------------|----------|-------------|-----------|
| Object Swap | **79.0** | 65.3 | 75.0 | 60.2 | +13.7 |
| Attribute Swap | **82.9** | 76.0 | 70.1 | 62.3 | +6.9 |
| Relation Replace | **86.8** | 76.1 | 85.3 | 65.2 | +10.7 |
| Attribute Add | **95.8** | 85.0 | 83.5 | 71.5 | +10.8 |

### 消融实验（1M样本训练）

| 配置 | SugarCrepe(Rep/Swp/Add) | Flickr T2I/I2T | 说明 |
|------|------------------------|----------------|------|
| LLaVA原始 | 81.9/59.8/64.7 | 59.6/65.6 | 无适配基线 |
| +soft prompt | 86.4/66.9/89.3 | 76.7/91.7 | prompt alone很有效 |
| +LoRA | 87.0/69.8/88.8 | 79.1/91.4 | LoRA容量更大 |
| +两者结合 | 87.1/72.0/88.6 | 79.6/92.9 | 互补提升 |
| +AR Loss | **89.5/75.5/89.5** | **80.6/91.8** | AR loss关键 |
| 数据1M→8.1M | 持续提升 | 无饱和迹象 | 扩展空间大 |

### 关键发现

- 7B VladVA超越18B EVA-CLIP：Flickr IR 85.0% vs 83.3%，COCO IR 59.0% vs 55.6%
- Object Swap类别提升最大（+13.7%），直接度量"bag of words"行为，VladVA显著减轻了CLIP家族的根本缺陷
- 对比损失和自回归损失承担互补角色：移除AR loss后组合性大幅下降，移除对比loss后检索性能下降
- 数据从1M到8.1M持续提升无饱和，仍有scaling潜力
- Qwen2-VL-2B版本也有效（Flickr IR从54.1→80.4），说明框架泛化性强

## 亮点与洞察

- **推翻了E5-V的核心结论**：证明对比图文微调不仅不有害，反而是释放LVLM判别能力的关键——前提是配合合理的数据策略和自回归loss
- "什么是好prompt"的分析非常深刻：高熵output分布 → 高秩embedding矩阵 → 更好检索性能
- **attention densification**的解释力：优雅说明了为什么LVLM需要特殊训练——生成模式可"逐步偷看"，判别模式必须"一次看完"
- 7B超越18B的效率故事：不需要更大模型，需要更好的训练策略

## 局限与展望

- 主实验仅基于LLaVA-1.5-7B，更大LVLM（13B/70B）的效果未验证
- Text R@1在Flickr上略低于EVA-CLIP(18B)（94.3 vs 95.3），文本侧还有优化空间
- 训练成本不低（32 A100），全量复现有门槛
- 未探索E5-V的text-text对比损失与VladVA框架的融合（论文已提及留future work）
- 推理时需分别计算图文embedding各一次forward，LVLM推理成本高于CLIP

## 相关工作与启发

- 与E5-V形成直接对比：E5-V只用text-text loss，VladVA证明image-text对比+AR loss更优
- 与VLM2Vec相比（无生成loss和soft prompt），VladVA在相同设定下大幅超越
- "将生成模型转为判别模型"思路可能适用于其他生成模型（如diffusion for retrieval）
- 短/长caption分治策略可迁移到其他需要多粒度监督的VL任务

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 将LVLM转化为判别式模型的完整框架非常新颖，混合损失+数据策略设计精巧
- 实验充分度: ⭐⭐⭐⭐⭐ 检索+组合性+消融+数据扩展+prompt分析+attention可视化极为全面
- 写作质量: ⭐⭐⭐⭐⭐ 动机分析透彻，每个设计都有充分的实验和理论支撑
- 价值: ⭐⭐⭐⭐⭐ 方向性贡献——证明LVLM在判别任务上的巨大潜力，影响深远

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Advancing Compositional Awareness in CLIP with Efficient Fine-Tuning](../../NeurIPS2025/multimodal_vlm/advancing_compositional_awareness_in_clip_with_efficient_fin.md)
- [\[CVPR 2025\] Beyond Words: Augmenting Discriminative Richness via Diffusions in Unsupervised Prompt Learning](beyond_words_augmenting_discriminative_richness_via_diffusions_in_unsupervised_p.md)
- [\[CVPR 2025\] FLAIR: VLM with Fine-grained Language-informed Image Representations](flair_vlm_with_fine-grained_language-informed_image_representations.md)
- [\[ICCV 2025\] The Inter-Intra Modal Measure: A Predictive Lens on Fine-Tuning Outcomes in Vision-Language Models](../../ICCV2025/multimodal_vlm/the_inter-intra_modal_measure_a_predictive_lens_on_fine-tuning_outcomes_in_visio.md)
- [\[ICML 2025\] CoMemo: LVLMs Need Image Context with Image Memory](../../ICML2025/multimodal_vlm/comemo_lvlms_need_image_context_with_image_memory.md)

</div>

<!-- RELATED:END -->
