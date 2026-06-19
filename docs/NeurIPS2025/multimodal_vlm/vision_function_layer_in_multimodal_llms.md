---
title: >-
  [论文解读] Vision Function Layer in Multimodal LLMs
description: >-
  [NeurIPS 2025][多模态VLM][MLLM内部机制] 发现MLLM中视觉相关的功能解码分布在特定的窄层块中（Vision Function Layer），且跨模型家族呈现一致的层级顺序（识别→计数→定位→OCR），据此提出VFL-LoRA（仅用1/3参数匹配full-LoRA性能）和VFL-select（20%数据达98%全量性能）。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "MLLM内部机制"
  - "视觉功能层"
  - "Token Swapping"
  - "LoRA"
  - "数据选择"
---

# Vision Function Layer in Multimodal LLMs

**会议**: NeurIPS 2025  
**arXiv**: [2509.24791](https://arxiv.org/abs/2509.24791)  
**代码**: [GitHub](https://github.com/ChengShiest/Vision-Function-Layer)  
**领域**: 多模态大模型 / 可解释性  
**关键词**: MLLM内部机制, 视觉功能层, Token Swapping, LoRA, 数据选择

## 一句话总结

发现MLLM中视觉相关的功能解码分布在特定的窄层块中（Vision Function Layer），且跨模型家族呈现一致的层级顺序（识别→计数→定位→OCR），据此提出VFL-LoRA（仅用1/3参数匹配full-LoRA性能）和VFL-select（20%数据达98%全量性能）。

## 研究背景与动机

- **核心问题**: MLLM在视觉理解上取得了显著进展，但其内部如何处理和推理视觉token仍是"黑盒"
- **现有方案缺陷**: 已有可解释性研究主要关注token重要性和跨模态交互，忽略了不同视觉功能如何在层级间被内部表征和协调
- **关键差距**: 缺乏能够隔离单个视觉功能的诊断框架——大多数通用任务同时需要多种能力，导致只能得出粗糙结论（如"浅层提取特征、深层做推理"）
- **额外挑战**: 不同MLLM使用不同视觉编码器和连接模块，使内部机制分析更加复杂

## 方法详解

### Vision Token Swapping分析框架

核心思路：在解码时第 $k$ 层将原始图像的视觉token KV缓存替换为另一张图像的视觉token，观察输出变化率。精心设计最小差异图像对来隔离单个视觉功能：

- **OCR**: 不同单词渲染在空白画布上
- **识别（Recognition）**: COCO图像 vs 空白画布，问是否存在某物体
- **计数（Counting）**: CLEVR数据集，仅物体数量不同
- **定位（Grounding）**: 相同物体在不同位置

### 关键发现：Vision Function Layer

以Qwen-2.5-VL-7B（28层）为例的功能层定位：

| 视觉功能 | 峰值层 | 峰值变化率 | 特征 |
|----------|--------|-----------|------|
| 识别 | 0-10层 | 分布式 | 早期建立，持续影响 |
| 计数 | 第12层 | 87.4% | 集中在中间层 |
| 定位 | 第18层 | 100.0% | 集中在中深层 |
| OCR | 第22层 | 92.8% | 集中在深层 |

**与人类认知一致**: 先识别→再计数→再定位→最后读字，跨LLaVA和Qwen系列呈现一致模式。

### Vision Token Dropping验证

对通用VQA benchmark逐步移除深层视觉token，验证功能层发现：
- 移除最后4层：OCR/TextVQA急剧下降（Qwen-7B: 82.8→74.1），其他任务几乎不变
- 移除最后8层：OCR类任务近乎崩溃（82.8→15.3），Recognition/Spatial开始下降
- 移除最后12层：所有视觉任务显著下降

### 应用1：VFL-LoRA

仅在目标视觉功能对应的层上施加LoRA，而非全层。以空间推理为例，对Qwen2.5-VL-7B仅使用计数功能层（10-17, 20-23层）训练LoRA：

- 可训练参数：155M vs 全量LoRA 309M（减少50%）
- In-domain平均: 85.0% vs 全量LoRA 84.4%（持平或略优）
- Out-of-domain平均: 75.0% vs 全量LoRA 74.3%（更好的泛化，减少灾难性遗忘）

### 应用2：VFL-select数据选择

通过分析特定VFL被消融时训练数据上的性能差异，自动将数据按功能分类。以20%数据量达到98%全量数据性能，超越人工专家数据选择。

## 实验关键数据

### Vision Token Dropping对各任务的影响（Qwen2.5-VL-7B）

| Drop层数 | SQA-I | POPE | TextVQA | OCR | ChartQA |
|---------|-------|------|---------|-----|---------|
| 0 (baseline) | 87.2 | 86.1 | 82.8 | 82.2 | 83.2 |
| drop 4 | 87.4 | 86.3 | 74.1↓ | 76.3↓ | 82.7↓ |
| drop 8 | 87.4 | 86.2 | 15.3↓↓ | 5.5↓↓ | 20.5↓↓ |
| drop 12 | 87.2 | 79.5↓ | 13.8 | 3.7 | 17.4 |

### VFL-LoRA vs Full-LoRA（Qwen2.5-VL-7B）

| 方法 | 参数量(%) | CV-Count | CV-Avg | ChartQA | MMMU | POPE |
|------|----------|----------|--------|---------|------|------|
| 基线 | - | 68.0 | 82.1 | 83.2 | 50.7 | 86.1 |
| Full-LoRA | 1.9% | 70.9 | 84.4 | 86.2 | 50.1 | 86.6 |
| **VFL-LoRA** | **0.9%** | **72.6** | **85.0** | **86.4** | **51.7** | **86.9** |
| Reversed-VFL | 0.9% | 69.0 | 82.7 | 85.9 | 51.2 | 84.9 |

### VFL-select数据选择

- 20%数据实现98%全量性能
- 超越人类专家在相同预算约束下的数据选择结果

## 亮点与洞察

1. **发现跨模型一致的功能层级**: 从LLaVA到Qwen，从3B到13B，视觉功能层的层级顺序惊人地一致（识别→计数→定位→OCR），暗示MLLM可能发展出类人的层级视觉处理策略
2. **Token Swapping比传统探针更精确**: 通过最小差异图像对实现功能级因果分析，而非仅做相关性分析
3. **实用价值显著**: VFL-LoRA用一半参数超越full-LoRA且减少遗忘；VFL-select用1/5数据达98%性能——两个应用都有明确的工程价值
4. **反转实验（Reversed-VFL）提供强反证**: 在非功能层上做LoRA性能显著差于VFL-LoRA，证实功能层定位的有效性

## 局限性

1. **功能粒度有限**: 仅分析了4种视觉功能（识别/计数/定位/OCR），更复杂的推理、因果理解等未覆盖
2. **需要精心设计图像对**: Token Swapping依赖最小差异图像对的构造，这对新功能的分析构成瓶颈
3. **层级划分的清晰度因功能而异**: Recognition呈分布式而非局部化，说明并非所有功能都有清晰的VFL
4. **VFL-LoRA的层选择依赖先验知识**: 需要预先通过Token Swapping分析确定功能层，增加了使用门槛
5. **CV-Distance子任务下降**: 该子任务更依赖语言先验而非视觉，VFL-LoRA对此类任务帮助有限

## 相关工作与启发

- **LLM可解释性**: 文本LLM中层级功能分工（如浅层语法→深层语义）在MLLM中找到了视觉对应物
- **LoRA变种**: VFL-LoRA是基于机制理解而非经验搜索的层选择策略，比随机选层或按梯度选层更有理论支撑
- **启发**: 功能层发现可指导MLLM剪枝——若某下游场景仅需识别和计数，可安全跳过OCR功能层以加速推理

## 评分

⭐⭐⭐⭐⭐ — 科学发现深刻（功能层级的跨模型一致性），方法设计巧妙（Token Swapping），实用应用强（VFL-LoRA和数据选择），实验覆盖全面（多模型×多任务×消融+应用）。是MLLM可解释性领域的重要工作。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Multi-Layer Visual Feature Fusion in Multimodal LLMs: Methods, Analysis, and Best Practices](../../CVPR2025/multimodal_vlm/multi-layer_visual_feature_fusion_in_multimodal_llms_methods_analysis_and_best_p.md)
- [\[NeurIPS 2025\] ADMN: A Layer-Wise Adaptive Multimodal Network for Dynamic Input Noise and Compute Resources](admn_a_layerwise_adaptive_multimodal_network_for_dynamic_inp.md)
- [\[NeurIPS 2025\] Learning to Steer: Input-dependent Steering for Multimodal LLMs](learning_to_steer_input-dependent_steering_for_multimodal_llms.md)
- [\[NeurIPS 2025\] MME-VideoOCR: Evaluating OCR-Based Capabilities of Multimodal LLMs in Video Scenarios](mme-videoocr_evaluating_ocr-based_capabilities_of_multimodal_llms_in_video_scena.md)
- [\[NeurIPS 2025\] VT-FSL: Bridging Vision and Text with LLMs for Few-Shot Learning](vt-fsl_bridging_vision_and_text_with_llms_for_few-shot_learning.md)

</div>

<!-- RELATED:END -->
