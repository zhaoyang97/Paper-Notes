---
title: >-
  [论文解读] Guiding Cross-Modal Representations with MLLM Priors via Preference Alignment
description: >-
  [NeurIPS 2025][多模态VLM][跨模态检索] 提出 MAPLE 框架，利用现成 MLLM 的内在模态对齐能力自动构建偏好数据，通过 Relative Preference Alignment（RPA）损失引导跨模态表示学习，在细粒度检索任务上取得显著提升。 CLIP 等对比学习模型在跨模态检索中表现出色…
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "跨模态检索"
  - "偏好对齐"
  - "DPO"
  - "模态间隙"
  - "MLLM"
---

# Guiding Cross-Modal Representations with MLLM Priors via Preference Alignment

**会议**: NeurIPS 2025  
**arXiv**: [2506.06970](https://arxiv.org/abs/2506.06970)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 跨模态检索, 偏好对齐, DPO, 模态间隙, MLLM

## 一句话总结

提出 MAPLE 框架，利用现成 MLLM 的内在模态对齐能力自动构建偏好数据，通过 Relative Preference Alignment（RPA）损失引导跨模态表示学习，在细粒度检索任务上取得显著提升。

## 研究背景与动机

CLIP 等对比学习模型在跨模态检索中表现出色，但其特征空间中存在显著的**模态间隙**（modality gap）——图像和文本嵌入在共享空间中存在系统性分离，限制了检索效果。

作者有一个关键发现：**现成的 MLLM（如 Qwen2-VL）天然具有强大的模态对齐能力**。通过提出基于 1-Wasserstein 距离（WD）的统一度量，可以同时比较基于 logits 的模型（MLLM）和基于嵌入的模型（CLIP）的模态间隙，发现 MLLM 的对齐质量远优于 CLIP。

然而，将 MLLM 微调为检索模型时，这种内在对齐能力会被削弱。因此问题是：**如何在将 MLLM 转为检索器的同时保留其强大的跨模态对齐能力？**

两个核心挑战：
1. 标准对比学习是粗粒度对齐，均匀推开所有负样本，忽略细粒度语义差异
2. 直接对 MLLM 做对比微调会损失其原有的模态对齐先验

## 方法详解

### 整体框架

MAPLE（Modality-Aligned Preference Learning for Embeddings）包含两大组件：
1. **偏好数据构建**：离线挖掘难负样本 + 在线用 MLLM 评分
2. **偏好对齐训练**：用 RPA 损失对齐嵌入空间与 MLLM 偏好

### 关键设计

#### 1. MLLM-based 检索器架构

- 从预训练 MLLM 初始化模型
- 将因果注意力掩码替换为**双向注意力**
- 添加 **mean pooling** 聚合最终隐状态为检索表示
- 用 LoRA 微调

#### 2. 偏好数据构建

**离线阶段 — 候选生成**：
- 用 DINOv2 提取图像嵌入，通过 SemDedup 去重
- 为每个图像检索 top-K 近邻作为难负样本集 $\mathcal{C}_i^{img}$
- 用 MLLM 的多图推理能力生成区分性描述，构建文本候选集 $\mathcal{C}_i^{txt}$

**在线阶段 — 评分与结构化**：
- 计算对齐分数：Prompt MLLM 对图文对输出 "yes"/"no"，用 softmax 得到对齐分数 $\alpha_{ii}$
- 按对齐分数排序候选，构建两种偏好结构：
    - **Pairwise 偏好**：所有满足 $a < b$ 的候选对 $(x_{r_a}, x_{r_b})$
    - **Listwise 偏好**：利用完整排名列表的每个后缀结构

#### 3. Relative Preference Alignment (RPA) 损失

从 DPO 出发，做两个关键改进：

**消除参考模型**：采用均匀先验 $U$ 作为参考模型 $\pi_w$，简化为：
$$\mathcal{L}_{\text{DPO-simplified}} = -\mathbb{E}[\log \sigma(\beta \log \pi_\theta(y_w|x) - \beta \log \pi_\theta(y_l|x))]$$

**适配嵌入模型**：用缩放相似度分数 $\beta(z^{anchor} \cdot z^{candidate})$ 替换对数概率。

**Pairwise RPA 损失**：
$$\mathcal{L}_{\text{RPA-Pairwise}}^{txt2img} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{0 \le k < l \le K} (\alpha_{i,r_k} - \alpha_{i,r_l}) \log \sigma(s_{ik} - s_{il})$$

偏好权重 $(\alpha_{r_k} - \alpha_{r_l})$ 使得偏好差距越大的样本对获得更多关注。

**Listwise RPA 损失**：
$$\mathcal{L}_{\text{RPA-Listwise}}^{txt2img} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=0}^{K-1} w_{ik} \log \frac{\exp(s_{ik})}{\sum_{j=k}^{K} \exp(s_{ij})}$$

其中权重 $w_{ik}$ 是第 $k$ 个候选与所有后续候选的平均 MLLM 对齐分数差。

### 损失函数 / 训练策略

**正则化联合优化**：
$$\mathcal{L} = \lambda \mathcal{L}_{RPA} + (1 - \lambda) \mathcal{L}_{contrast}$$

- $\mathcal{L}_{contrast}$ 是标准对比损失，作为正则化防止过度对齐导致特征坍缩
- $\lambda$ 平衡偏好对齐强度与通用检索能力
- 训练数据：OpenImage 子集
- 扩展负样本池策略：隐式增大有效 batch size 而不增加计算开销

## 实验关键数据

### 主实验

| 模型 | COCO T/I R@1 | Flickr30k T/I R@1 | Winoground T/I | NaturalBench T/I |
|------|-------------|-------------------|----------------|-----------------|
| SigLIPv2 (2B) | 72.8/56.1 | 95.4/86.0 | 39.8/17.0 | 65.5/68.7 |
| VladVA (LLaVA-7B) | 72.9/59.0 | 94.3/83.3 | 40.5/17.5 | -/- |
| **MAPLE (Qwen2-VL-7B)** | **75.5/60.3** | **94.3/86.1** | **56.0/32.7** | **76.1/76.8** |

细粒度检索提升尤为显著：Winoground Text +13.5, Image +15.2（vs VladVA）。

### 消融实验

| 方法 | COCO T/I | Winoground T/I | NaturalBench T/I |
|------|---------|----------------|-----------------|
| Baseline ($\mathcal{L}_{contrast}$) | 74.0/54.4 | 42.5/20.5 | 61.4/62.5 |
| + RPA-Pairwise only | 51.9/52.4 | 48.8/34.7 | 70.1/77.3 |
| $\mathcal{L}_{contrast}$ + RPA-Listwise | 71.9/58.6 | 51.0/28.2 | 69.2/71.2 |
| + 扩展负样本 + RPA | **75.5/60.3** | **56.0/32.7** | **76.1/76.8** |

### 关键发现

1. **单独使用 RPA 损失**大幅提升细粒度检索但损害通用检索；加 $\mathcal{L}_{contrast}$ 正则化后两者兼得
2. **Listwise RPA 优于 Pairwise**：因为 listwise 考虑了完整排名结构而非独立的对
3. **扩展负样本池**对通用和细粒度检索均有提升，有效增大了 batch size
4. **模态间隙度量**：MAPLE + RPA 显著降低分布间隙（$W_{dist-gap}$）同时提升判别间隙（$W_{disc-gap}$）

## 亮点与洞察

1. **发现 MLLM 天生具有强模态对齐能力**：这一观察本身就有价值，提示可以将 MLLM 作为其他模型的"对齐教师"
2. **DPO → 嵌入空间的迁移巧妙**：将对数概率替换为相似度得分，消除参考模型需求，适配检索场景
3. **偏好权重设计细腻**：用 MLLM 对齐分数差作为偏好强度权重，更关注有明确偏好的样本对
4. **统一的模态间隙度量**：基于 Wasserstein 距离的度量可比较不同架构（logit-based vs embedding-based）

## 局限与展望

1. 跨模态表示可能继承 MLLM 的内在偏见
2. 未在更复杂的任务上验证（如组合检索 composed retrieval）
3. 在线阶段计算 MLLM 对齐分数增加训练成本
4. 通用检索性能相比 baseline 仍有轻微下降（COCO Text R@1 从 74.0 到 75.5 提升有限，部分配置下甚至下降）
5. 可探索更多 MLLM backbone 和更大规模训练数据

## 相关工作与启发

- 与 DPO/RLHF 的桥接：将 LLM 对齐技术迁移到视觉检索领域
- CLIP 的改进路线：不再改编码器架构，而是利用外部 MLLM 知识来改进对齐
- 启发：可将类似思路用于视频检索、多模态RAG中的细粒度匹配

## 评分

- 新颖性: ⭐⭐⭐⭐ (DPO→嵌入空间的迁移和MLLM对齐先验发现)
- 实验充分度: ⭐⭐⭐⭐ (多个通用+细粒度基准，完整消融)
- 写作质量: ⭐⭐⭐⭐ (技术细节丰富但结构略显复杂)
- 价值: ⭐⭐⭐⭐ (细粒度检索提升显著，MLLM先验的发现有启发性)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Instruction-Oriented Preference Alignment for Enhancing Multi-Modal Comprehension Capability of MLLMs](../../ICCV2025/multimodal_vlm/instruction-oriented_preference_alignment_for_enhancing_multi-modal_comprehensio.md)
- [\[NeurIPS 2025\] FOCUS: Internal MLLM Representations for Efficient Fine-Grained Visual Question Answering](focus_internal_mllm_representations_for_efficient_fine-grained_visual_question_a.md)
- [\[ICML 2025\] Vision-Language Models Create Cross-Modal Task Representations](../../ICML2025/multimodal_vlm/vision-language_models_create_cross-modal_task_representations.md)
- [\[NeurIPS 2025\] Learning from Videos for 3D World: Enhancing MLLMs with 3D Vision Geometry Priors](learning_from_videos_for_3d_world_enhancing_mllms_with_3d_vision_geometry_priors.md)
- [\[NeurIPS 2025\] On the Value of Cross-Modal Misalignment in Multimodal Representation Learning](on_the_value_of_cross-modal_misalignment_in_multimodal_representation_learning.md)

</div>

<!-- RELATED:END -->
