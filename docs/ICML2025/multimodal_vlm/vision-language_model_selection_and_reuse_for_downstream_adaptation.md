---
title: >-
  [论文解读] Vision-Language Model Selection and Reuse for Downstream Adaptation
description: >-
  [ICML 2025][多模态VLM][Model Selection] 提出 Model Label Learning (MLL) 范式，通过构建语义图对 49 个预训练 VLM 进行离线"标注"（描述各模型在不同视觉概念上的能力），面对新任务时通过语义匹配选择和集成最合适的模型，实现数据高效、计算高效且可扩展的 VLM 选择与复用。
tags:
  - "ICML 2025"
  - "多模态VLM"
  - "Model Selection"
  - "VLM"
  - "Model Hub"
  - "Semantic Graph"
  - "Ensemble"
  - "模型标签学习"
---

# Vision-Language Model Selection and Reuse for Downstream Adaptation

**会议**: ICML 2025  
**arXiv**: [2501.18271](https://arxiv.org/abs/2501.18271)  
**代码**: 未公开  
**领域**: VLM 模型选择, 模型复用, 零样本视觉任务  
**关键词**: Model Selection, VLM, Model Hub, Semantic Graph, Ensemble, 模型标签学习

## 一句话总结

提出 Model Label Learning (MLL) 范式，通过构建语义图对 49 个预训练 VLM 进行离线"标注"（描述各模型在不同视觉概念上的能力），面对新任务时通过语义匹配选择和集成最合适的模型，实现数据高效、计算高效且可扩展的 VLM 选择与复用。

## 研究背景与动机

### 领域现状

**领域现状**：开源 VLM（如 CLIP 及其变体）数量快速增长，open-clip 库已有 100+ 模型。然而：

### 现有痛点

**现有痛点**：没有单一 VLM 在所有任务上最优**：不同模型在不同任务甚至同一任务的不同类别上表现差异巨大

### 核心矛盾

**核心矛盾**：评估所有模型不现实**：受时间和数据限制

### 解决思路

**解决思路**：现有模型选择方法**（NCE、LEEP、LogME）**面向单模态模型**，不适用于 VLM

首个 VLM 选择工作 LOVM 提出用文本数据评估 VLM，但依赖 ImageNet 上的真实性能，在下游任务与通用数据集存在域偏移时失效。

## 方法详解

### 整体框架 — MLL 三模块

**模块 1：模型标注 (Model Labeling)**

构建**语义图 $\mathcal{G}$**：
- 节点：WordNet 的 synset（>9000 视觉概念）
- 边：上下位关系
- 每个节点关联代表性图像样本 $X_v$
- Caption："{synset name} which is {synset definition}"

每个 VLM $f_m$ 在语义图上预测试，生成模型标签：

$$s_{m,x}^v = \text{sim}(\mathcal{I}_m(x), \mathcal{T}_m(d_v)), \quad S_m = \{s_m^v | v \in V_\mathcal{G}\}$$

标签描述了模型在各视觉概念上的能力分布。**此过程与目标任务无关，一次预计算。**

**模块 2：模型选择 (Model Selection)**

给定目标任务类别 $Y_T$：
1. GPT-4 为每个类别生成扩展描述 $D_T$
2. 语言模型计算 $D_T$ 与 $D_\mathcal{G}$ 的相似度，为每个类选 top-$k$ 语义节点
3. 构建迁移矩阵 $Z$
4. 由模型标签估计每个模型在每个目标类上的精度：$p_{m,y} = \sum_v p_{m,v} \cdot z_{vy}$
5. 综合类别精度和整体精度：$r_{m,y} = \alpha \cdot p_{m,y} + \frac{1-\alpha}{|Y_T|}\sum_{y'} p_{m,y'}$

**模块 3：模型复用 (Model Reuse)**

对每个类选 top-$k$ 模型组成集成预测器：

$$p_y^k(x) = \sum_{f_m \in \mathcal{F}_y^k} w_{m,y} \cdot \frac{\exp(\text{sim}(\mathcal{I}_m(x), \mathcal{T}_m(y)))}{\sum_{y'} \exp(\text{sim}(\mathcal{I}_m(x), \mathcal{T}_m(y')))}$$

权重 $w_{m,y}$ 基于预测概率熵——高置信度（可能过度自信）的模型权重降低。

最终预测：$\hat{y} = \arg\max_y p_y^k(x)$

## 实验关键数据

### Benchmark

49 个预训练 VLM + 17 个下游数据集。

### 单模型选择（k=1）


### 主实验

| 方法 | CIFAR100 | Flowers102 | MNIST | FER2013 | StanfordCars | 平均 |
|------|---------|-----------|-------|---------|-------------|------|
| INB (ImageNet最优) | 0.860 | 0.876 | 0.796 | 0.286 | 0.949 | 0.643 |
| ModelGPT | 0.860 | 0.876 | 0.565 | 0.401 | 0.949 | 0.637 |
| **MLL** | **0.877** | **0.891** | **0.810** | **0.493** | **0.957** | **0.662** |

### 集成 3 模型（k=3）

17 个数据集平均准确率：MLL 最优，超越 INB 和 ModelGPT 基线。

### 关键发现

- ImageNet 上最优的模型在特定任务上不一定最好（如 FER2013 上差距巨大）
- 每类选不同模型（细粒度选择）比全局选一个模型效果更好
- 模型 hub 越大，MLL 性能越好（可扩展性强）

## 亮点与洞察

1. **目标任务无关的标注**：模型标签在上传时一次性计算，选择过程无需运行候选模型
2. **细粒度类别级选择**：不同类别可选不同模型，充分利用各模型专长
3. **可扩展性**：语义图可持续扩展节点，模型 hub 越大能力越强
4. **完整 Benchmark**：49 模型 × 17 数据集的系统评估推动了 VLM 选择领域的研究

## 局限与展望

- 语义图覆盖度依赖于 WordNet，可能遗漏领域特定概念
- GPT-4 生成 caption 引入了对闭源模型的依赖
- 每类样本数较少时模型标签可能不够准确
- 集成多模型增加推理开销

## 相关工作

- Model Selection（NCE、LEEP、LogME、Model Spider）
- LOVM（首个 VLM 选择工作）
- Learnware 范式（模型规约）
- VLM 模型库（open-clip、HuggingFace）

## 评分

⭐⭐⭐⭐ — 新颖的"模型标签"概念，将模型选择从在线评估转为离线预计算+语义匹配。49 VLM×17 数据集的大规模 benchmark 有独立价值。实践层面的贡献大于理论深度。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Metacognitive Sensitivity for Test-Time Dynamic Model Selection](../../NeurIPS2025/multimodal_vlm/metacognitive_sensitivity_for_test-time_dynamic_model_selection.md)
- [\[CVPR 2026\] Vision-Language Model Guided Source-Free Domain Adaptation via Optimal Transport](../../CVPR2026/multimodal_vlm/vision-language_model_guided_source-free_domain_adaptation_via_optimal_transport.md)
- [\[CVPR 2025\] Realistic Test-Time Adaptation of Vision-Language Models](../../CVPR2025/multimodal_vlm/realistic_test-time_adaptation_of_vision-language_models.md)
- [\[NeurIPS 2025\] DOTA: DistributiOnal Test-time Adaptation of Vision-Language Models](../../NeurIPS2025/multimodal_vlm/dota_distributional_testtime_adaptation_of_visionlanguage_mo.md)
- [\[CVPR 2026\] Rethinking Model Selection in VLM Through the Lens of Gromov-Wasserstein Distance](../../CVPR2026/multimodal_vlm/rethinking_model_selection_in_vlm_through_the_lens_of_gromov-wasserstein_distanc.md)

</div>

<!-- RELATED:END -->
