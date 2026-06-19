---
title: >-
  [论文解读] COUNTS: Benchmarking Object Detectors and Multimodal Large Language Models under Distribution Shifts
description: >-
  [CVPR 2025][多模态VLM][分布偏移] 本文构建了COUNTS——一个包含14种自然分布偏移、222K+样本和119万+标注框的大规模OOD数据集，并提出O(OD)²和OODG两个基准，系统评估了目标检测器和多模态大模型在分布偏移下的泛化能力，发现即使是GPT-4o也仅能达到56.7%的定位准确率。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "分布偏移"
  - "目标检测鲁棒性"
  - "视觉定位"
  - "OOD泛化"
  - "多模态基准"
---

# COUNTS: Benchmarking Object Detectors and Multimodal Large Language Models under Distribution Shifts

**会议**: CVPR 2025  
**arXiv**: [2504.10158](https://arxiv.org/abs/2504.10158)  
**代码**: [GitHub](https://github.com/jiansheng-li/COUNTS_benchmark)  
**领域**: 多模态VLM  
**关键词**: 分布偏移, 目标检测鲁棒性, 视觉定位, OOD泛化, 多模态基准

## 一句话总结
本文构建了COUNTS——一个包含14种自然分布偏移、222K+样本和119万+标注框的大规模OOD数据集，并提出O(OD)²和OODG两个基准，系统评估了目标检测器和多模态大模型在分布偏移下的泛化能力，发现即使是GPT-4o也仅能达到56.7%的定位准确率。

## 研究背景与动机
现有目标检测器在实际部署中常因数据分布偏移导致性能大幅退化。分布外（OOD）泛化问题已在图像分类领域有充分研究，但在更复杂的目标检测和视觉定位任务上，缺乏大规模、细粒度标注的评估基准。现有的检测鲁棒性数据集存在以下问题：COCO-C使用合成损坏（不反映真实场景）、COCO-O规模小（仅6,782张图）且域数量有限、驾驶场景数据集（如Cityscapes等）场景单一缺乏领域多样性。

更关键的是，多模态大语言模型（MLLMs）近年来发展迅猛，但其**视觉定位（grounding）**能力在分布偏移下的表现几乎未被研究。由于MLLM的训练数据不透明，传统定义分布偏移的方式不适用，需要新的评估范式。

本文的切入点是：构建一个真实世界采集的、涵盖丰富域类型的大规模细粒度标注数据集，同时提出适用于目标检测器和MLLM的两个评估基准，从而系统性地揭示两类模型在OOD条件下的弱点。

## 方法详解

### 整体框架
COUNTS项目包含三个核心组成部分：
1. **COUNTS数据集**：14个域、35个类别、222,234张真实世界图像、1,196,114个标注框
2. **O(OD)²基准**：评估目标检测器的OOD泛化能力
3. **OODG基准**：评估MLLM视觉定位的OOD泛化能力

### 关键设计
1. **COUNTS数据集构建**:

    - 三阶段流程：首先从Open Images、Visual Genome、RefCOCO等公开数据集的500万+图像中筛选候选样本；然后由两名独立标注员验证域标签（只保留一致的样本）；最后对测试/验证集（23,000张图）进行人工重新标注bbox
    - 14个域的选择标准：在现实中常见、对像素分布有显著影响、域间尽量独立
    - 域列表：dim（昏暗）、painting（绘画）、snow（雪景）、sand（沙地）、handmade（手工制品）、street（街道）、road（公路）、water（水域）、grass（草地）、indoor（室内）、mountain（山景）、sky（天空）、tree（树木）、occlusion（遮挡）
    - 所有域共享完整的35个类别空间，保证跨域测试的有效性

2. **O(OD)²基准设计**:

    - 核心思路：选择sky、occlusion、grass、water、dim、handmake六个域作为目标域（代表不同类型的分布偏移），其余域作为训练集
    - 使用多个目标域同时评估（而非单一目标域），更真实地反映部署场景的不确定性
    - 评估对象涵盖两阶段检测器（Faster R-CNN + 各种backbone/neck/head组合）、单阶段检测器（RetinaNet、YOLOv9）和Transformer检测器（DETR、DINO、DINOv2）

3. **OODG基准设计**:

    - 创新定义：将分布偏移定义为ICL示例与测试样本之间的分布差异（而非训练/测试偏移）
    - 五个评估设置：零样本能力、IID上下文学习、协变量偏移泛化、标签偏移泛化、虚假相关偏移泛化
    - 三种测试任务：视觉定位（给区域问类别）、识别与定位（给问题预测框）、视觉-语义映射（给描述匹配区域）
    - 虚假相关偏移的设计尤其巧妙：通过在ICL样本中故意引入"猫=昏暗室内、狗=户外"等统计关联来测试模型是否被误导

### 损失函数 / 训练策略
本文为基准论文，不涉及新的训练策略。所有检测器实验使用MMDetection框架，遵循各原始论文的最优配置，使用RTX 4090 GPU。

## 实验关键数据

### 主实验——目标检测器OOD泛化（O(OD)²）

| 检测器 | 验证mAP | OOD平均mAP | 最差域 | 说明 |
|--------|--------|-----------|--------|------|
| Faster R-CNN (RN-50) | 0.279 | 0.135 | Water: 0.103 | 基线 |
| Faster R-CNN (RN-101) | 0.325 | 0.148 | Water: 0.110 | backbone更强有帮助 |
| DINO | 0.384 | 0.213 | Dim: 0.169 | 显著领先 |
| DINOv2 | 0.389 | 0.213 | Dim: 0.165 | 最佳 |
| YOLOv9 | 0.282 | 0.150 | Grass: 0.118 | 单阶段表现一般 |

### 主实验——MLLM视觉定位能力（OODG零样本）

| 模型 | 整体准确率 | 说明 |
|------|-----------|------|
| GPT-4o | 65.9% | 最强但仍有限 |
| GLaMM | 62.5% | 开源模型中表现较好 |
| Gemini-1.5-Flash | 59.1% | 中等水平 |
| Qwen2-VL | 54.4% | 明显弱于GPT-4o |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Neck对比（FPN vs PAFPN vs NAS-FPN） | OOD mAP差异<2% | Neck改变对OOD泛化影响有限 |
| Head对比（Standard vs Cascade vs SABL） | SABL>Cascade>Standard | Head设计是提升OOD关键 |
| 预训练数据（IN-1K vs IN-21K） | +0.5% OOD mAP | 数据量增加收益有限 |
| Sup_timm vs 标准supervised | +18.9%相对提升 | 高级训练方法比大数据更有效 |
| ICL设置3（协变量偏移）GPT-4o | 准确率降至55.2% | 分布偏移显著影响ICL效果 |
| ICL设置5（虚假相关）GPT-4o | 准确率降至50.3% | 模型易被虚假相关误导 |

### 关键发现
- IID性能的提升并不必然转化为OOD泛化的提升；某些增强IID性能的组件（如NAS-FPN、FreeAnchor）反而削弱OOD性能
- 对于两阶段检测器，改进head比改进backbone更有效；对于单阶段检测器，两者都有帮助
- DINO/DINOv2显著优于其他检测器，说明训练策略（自监督+对比学习）比单纯的架构改进更有助于OOD泛化
- 自监督预训练（MoCov2）没有带来OOD泛化的提升，但先进的训练方法（Sup_timm）显著提升了泛化能力
- 即使在ICL设置下，MLLM的视觉定位也非常脆弱，GPT-4o在协变量偏移和虚假相关设置下准确率大幅下降

## 亮点与洞察
1. 首个同时支持目标检测和MLLM视觉定位OOD评估的大规模真实世界数据集
2. 对MLLM定义分布偏移的方式非常巧妙——利用ICL样本与测试样本之间的分布差异，避开了训练数据不透明的问题
3. 虚假相关偏移设置的设计令人印象深刻：揭示了MLLM不仅会利用ICL示例学习，还会学到示例中的统计偏差
4. "IID强不等于OOD强"的结论虽然不新，但在检测任务上的系统性验证填补了重要空白
5. 34页的论文（含附录）数据量巨大，提供了检测器设计的详细指导

## 局限与展望
- 域的选择主要基于场景/背景级别的分布偏移，缺少对象级别的偏移（如姿态变化、形变等）
- 14个域之间的独立性假设未被严格验证，某些域可能存在强相关（如road和street）
- OODG基准目前仅定义了ICL阶段的分布偏移，未覆盖微调阶段的偏移
- 数据集的类别数（35类）相对于COCO（80类）仍较少，可能限制了某些分析的推广性
- 对开源MLLM的评估仅覆盖了有限的模型（GLaMM、Qwen2-VL），缺少对更多最新模型的评估

## 相关工作与启发
- 与COCO-O的对比表明，真实世界的域多样性和数据规模是建设性基准的关键
- OODG的ICL偏移定义可以推广到其他MLLM评估场景（如VQA、image captioning的OOD评估）
- 检测器head的重要性启发了一个方向：设计专门针对OOD鲁棒性的检测头
- 与DomainBed等分类OOD基准相比，检测任务的OOD泛化远未得到充分研究

## 评分
- 新颖性: ⭐⭐⭐⭐ 数据集构建和OODG定义有创新；但核心是基准论文，方法创新有限
- 实验充分度: ⭐⭐⭐⭐⭐ 涵盖大量检测器变体和MLLM模型，分析角度多（backbone/neck/head/预训练）
- 写作质量: ⭐⭐⭐⭐ 结构清晰、数据丰富，但论文过长（34页）且部分分析可以更精炼
- 价值: ⭐⭐⭐⭐ 填补了检测和MLLM定位OOD基准的空白，对未来研究有重要参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] On the Out-of-Distribution Generalization of Multimodal Large Language Models](on_the_out-of-distribution_generalization_of_large_multimodal_models.md)
- [\[CVPR 2026\] ORIC: Benchmarking Object Recognition under Contextual Incongruity in Large Vision-Language Models](../../CVPR2026/multimodal_vlm/oric_benchmarking_object_recognition_under_contextual_incongruity_in_large_visio.md)
- [\[CVPR 2025\] Teaching Large Language Models to Regress Accurate Image Quality Scores Using Score Distribution](teaching_large_language_models_to_regress_accurate_image_quality_scores_using_sc.md)
- [\[CVPR 2025\] VideoGLaMM: A Large Multimodal Model for Pixel-Level Visual Grounding in Videos](videoglamm_a_large_multimodal_model_for_pixel-level_visual_grounding_in_videos.md)
- [\[CVPR 2025\] EventGPT: Event Stream Understanding with Multimodal Large Language Models](eventgpt_event_stream_understanding_with_multimodal_large_language_models.md)

</div>

<!-- RELATED:END -->
