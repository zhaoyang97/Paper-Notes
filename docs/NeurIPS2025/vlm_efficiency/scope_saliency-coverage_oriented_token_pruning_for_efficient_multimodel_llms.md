---
title: >-
  [论文解读] SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodal LLMs
description: >-
  [NeurIPS 2025][多模态VLM][视觉Token剪枝] 提出 SCOPE，一种联合建模显著性和覆盖率的视觉 Token 剪枝策略，通过迭代选择 SCOPE 得分最高的 Token 来保持语义完整性，在 9 倍 Token 缩减下保留 LLaVA-1.5 96% 的性能。 领域现状：MLLM 将图像编码为大量视觉…
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "视觉Token剪枝"
  - "多模态大模型推理加速"
  - "语义覆盖率"
  - "子模函数"
  - "训练无关"
---

# SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodal LLMs

**会议**: NeurIPS 2025  
**arXiv**: [2510.24214](https://arxiv.org/abs/2510.24214)  
**代码**: [https://github.com/kinredon/SCOPE](https://github.com/kinredon/SCOPE)  
**领域**: 多模态VLM  
**关键词**: 视觉Token剪枝, 多模态大模型推理加速, 语义覆盖率, 子模函数, 训练无关

## 一句话总结
提出 SCOPE，一种联合建模显著性和覆盖率的视觉 Token 剪枝策略，通过迭代选择 SCOPE 得分最高的 Token 来保持语义完整性，在 9 倍 Token 缩减下保留 LLaVA-1.5 96% 的性能。

## 研究背景与动机

**领域现状**：MLLM 将图像编码为大量视觉 Token（如 576 或 2000+），与文本 Token 一起输入 LLM，自注意力的二次复杂度导致巨大计算开销

**现有痛点**：基于显著性的剪枝（如 FastV、SparseVLM、VisionZip）只保留注意力分数最高的 Token，存在两个问题：
   - **语义不完整**：高显著性 Token 往往集中在少数物体上，丢失了上下文信息（如对 "猫在哪" 的回答需要猫和它周围的环境）
   - **注意力分布偏斜**：只有极少数 Token 获得高注意力，其余 Token 注意力几乎均匀分布，难以区分信息性 Token 和冗余 Token

**核心矛盾**：显著性优先会导致所选 Token 语义高度重叠，覆盖率低

**切入角度**：借鉴子模函数优化（submodular optimization）中的覆盖函数思想，提出联合考虑显著性和覆盖率的选择策略

## 方法详解

### 整体框架
在 MLLM 的指定层（如第2层）进行 Token 剪枝：计算所有视觉 Token 对之间的余弦相似度，然后迭代地选择 SCOPE 得分最高的 Token 加入保留集合，直到达到预算 K。

### 关键设计

1. **集合覆盖率（Set-Coverage）**：

    - 功能：量化已选 Token 集合对全部 Token 的语义覆盖程度
    - 核心思路：对每个 Token $u$，其被覆盖程度定义为它与已选集合中最相似 Token 的余弦相似度：$C(u,\mathcal{S}) = \max_{s \in \mathcal{S}} \text{sim}(u,s)$。总覆盖率：$f(\mathcal{S}) = \sum_{u \in \mathcal{V}} \max_{s \in \mathcal{S}} \text{sim}(u,s)$
    - 设计动机：鼓励选择语义多样化的 Token，确保每个未选 Token 都有至少一个相似的代表

2. **Token 覆盖增益（Token-Coverage Gain）**：

    - 功能：量化新增一个 Token 带来的额外覆盖
    - 核心思路：边际增益 $\Delta(v;\mathcal{S}) = \sum_{u \in \mathcal{V}}[\max(C(u,\mathcal{S}), \text{sim}(u,v)) - C(u,\mathcal{S})]$
    - 设计动机：贪心选择增益最大的 Token，这是经典的次模函数最大化策略，有 $(1-1/e)$ 的近似保证

3. **SCOPE 得分**：

    - 功能：融合显著性和覆盖增益
    - 核心思路：$\Delta(v, A_v^\alpha; \mathcal{S}) = \Delta(v; \mathcal{S}) \cdot A_v^\alpha$，其中 $A_v$ 是注意力分数，$\alpha$ 是缩放因子
    - 设计动机：纯覆盖增益忽略了 Token 的内在信息量，乘以注意力分数后可以在覆盖和重要性之间取得平衡

### 训练策略
完全无需训练，即插即用。在推理时的指定 Transformer 层执行一次剪枝即可。

## 实验关键数据

### 主实验 — LLaVA-1.5 7B, 保留 64 Token (↓88.9%)

| Benchmark | Vanilla (576) | FastV | SparseVLM | VisionZip | SCOPE | 相对性能 |
|-----------|---------------|-------|-----------|-----------|-------|---------|
| GQA | 61.9 | 52.7 | 57.6 | 59.3 | **60.3** | 97.4% |
| MME | 1862 | 1612 | 1721 | 1783 | **1805** | 97.0% |
| POPE | 85.9 | 64.8 | 83.6 | 85.3 | **85.6** | 99.7% |
| TextVQA | 58.2 | 52.5 | 56.1 | 56.3 | **57.0** | 97.9% |
| Avg.(相对) | 100% | 89.5% | 96.5% | 97.5% | **98.2%** | - |

### 消融实验

| 策略 | θ-Coverage (θ=0.95) | GQA | MME |
|------|---------------------|-----|-----|
| Saliency Only | 18.2% | 57.6 | 1721 |
| Coverage Only | 52.3% | 59.1 | 1778 |
| Random | 23.5% | 50.2 | 1512 |
| SCOPE (ours) | 48.7% | **60.3** | **1805** |

### 关键发现
- 纯显著性方法的 θ-覆盖率甚至**低于随机选择**，说明高注意力 Token 高度集中
- 192 Token 时 SCOPE 保留 96.0% 原始性能，64 Token 时保留 98.2%
- 在 LLaVA-Next 上同样有效，说明方法具有通用性
- $\alpha$ 参数控制显著性权重，$\alpha=0.5$ 在大多数任务上最优

## 亮点与洞察
- **θ-覆盖率指标**的定义非常优雅，为视觉 Token 剪枝领域提供了一个新的定量评估维度，可以用来分析任何 Token 选择策略
- **子模函数最大化的贪心策略**天然适合 Token 选择问题，这个连接虽然自然但之前没人做过，是本文核心贡献
- 方法的时间复杂度是 $O(NK)$，对于 $N=576, K=64$ 只需约 36K 次相似度比较，几乎不影响推理速度

## 局限与展望
- 当前 SCOPE 在 MLLM 的某一层做一次剪枝，渐进式多层剪枝可能更好（如 PyramidDrop 的思路）
- 覆盖率基于余弦相似度来衡量语义接近性，但余弦相似度在高维空间不一定能精确反映语义关系
- 未考虑文本 Token 与视觉 Token 的交互，可以结合问题内容做自适应剪枝
- 对视频理解的扩展尚未验证

## 相关工作与启发
- **vs FastV**：FastV 用文本到视觉的早期层注意力做剪枝，只选显著 Token；SCOPE 额外考虑覆盖率
- **vs DivPrune**：DivPrune 最大化多样性但不考虑显著性；SCOPE 统一两者
- **vs VisionZip**：VisionZip 用 CLS Token 注意力 + Token 合并；SCOPE 的覆盖增益是更优的选择准则
- 这个方法的思路可以迁移到 NLP 中的长文本 Token 压缩

## 评分
- 新颖性: ⭐⭐⭐⭐ 子模函数覆盖 + 显著性的结合简洁优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 多个 MLLM、多个 benchmark、详细消融
- 写作质量: ⭐⭐⭐⭐⭐ 数学推导完整、可视化直观
- 价值: ⭐⭐⭐⭐⭐ 无训练即插即用，实用性极强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Balanced Token Pruning: Accelerating Vision Language Models Beyond Local Optimization](balanced_token_pruning_accelerating_vision_language_models_b.md)
- [\[ICCV 2025\] METEOR: Multi-Encoder Collaborative Token Pruning for Efficient Vision Language Models](../../ICCV2025/multimodal_vlm/meteor_multi-encoder_collaborative_token_pruning_for_efficient_vision_language_m.md)
- [\[CVPR 2026\] HAWK: Head Importance-Aware Visual Token Pruning in Multimodal Models](../../CVPR2026/multimodal_vlm/hawk_head_importance-aware_visual_token_pruning_in_multimodal_models.md)
- [\[CVPR 2026\] TransPrune: Token Transition Pruning for Efficient Large Vision-Language Model](../../CVPR2026/multimodal_vlm/transprune_token_transition_pruning_for_efficient_large_vision-language_model.md)
- [\[ACL 2026\] HiPrune: Hierarchical Attention for Efficient Token Pruning in Vision-Language Models](../../ACL2026/multimodal_vlm/hiprune_hierarchical_attention_for_efficient_token_pruning_in_vision-language_mo.md)

</div>

<!-- RELATED:END -->
