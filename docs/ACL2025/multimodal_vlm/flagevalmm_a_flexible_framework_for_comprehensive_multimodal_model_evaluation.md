---
title: >-
  [论文解读] FlagEvalMM: A Flexible Framework for Comprehensive Multimodal Model Evaluation
description: >-
  [ACL 2025][多模态VLM][多模态评估] 提出 FlagEvalMM，一个开源的多模态模型评估框架，通过将模型推理与评估过程解耦的架构设计，统一支持视觉语言理解（VQA）、文生图/文生视频生成和图文检索等多种多模态任务的评估。 随着多模态模型的快速发展，业界需要一个能够全面、高效、便捷地评估各类多模态能力的统一框架…
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "多模态评估"
  - "视觉语言模型"
  - "文生图评估"
  - "评估框架"
  - "解耦架构"
---

# FlagEvalMM: A Flexible Framework for Comprehensive Multimodal Model Evaluation

**会议**: ACL 2025  
**arXiv**: [2506.09081](https://arxiv.org/abs/2506.09081)  
**代码**: [https://github.com/flageval-baai/FlagEvalMM](https://github.com/flageval-baai/FlagEvalMM)  
**领域**: Multimodal / VLM 评估  
**关键词**: 多模态评估, 视觉语言模型, 文生图评估, 评估框架, 解耦架构

## 一句话总结

提出 FlagEvalMM，一个开源的多模态模型评估框架，通过将模型推理与评估过程解耦的架构设计，统一支持视觉语言理解（VQA）、文生图/文生视频生成和图文检索等多种多模态任务的评估。

## 研究背景与动机

随着多模态模型的快速发展，业界需要一个能够全面、高效、便捷地评估各类多模态能力的统一框架。然而现有方案存在明显不足：

**任务覆盖不全**：VLMEvalKit 和 Lmms-Eval 主要面向 VLM 理解任务；VBench 专注视频生成评估。没有一个框架能同时覆盖理解和生成任务。

**推理与评估耦合**：现有框架在同一运行环境中执行模型推理和评估，导致环境冲突（如模型推理和 LLM-as-Judge 评估的依赖冲突）、资源利用效率低。

**扩展性差**：VLMEvalKit 需要侵入式代码修改来添加新 benchmark；VHELM 基于 HELM 架构复杂、主要依赖 API 调用；Lmms-Eval 仅支持 Transformers 和 vLLM 推理框架。

FlagEvalMM 的动机在于通过解耦架构和模块化设计解决上述问题，提供一站式多模态评估体验。

## 方法详解

### 整体框架

FlagEvalMM 由两大组件构成：**评估服务器 (Evaluation Server)** 和 **模型运行器 (Model Runner)**，两者通过轻量级 HTTP RESTful 协议通信。这种解耦设计使推理环境和评估环境完全独立。

### 关键设计

1. **评估服务器 (Evaluation Server)**：每个评估任务是最小执行单元，包含三个核心组件——Processor（数据预处理，将不同来源的数据集转为标准化格式）、Config（配置参数如评估指标和提示模板）、Evaluator（评估模型输出并生成性能指标）。设计可扩展，用户可注册自定义 Dataset 和 Evaluator。

2. **模型运行器 (Model Runner)**：包含 Model Adapter 和 Backend。Model Adapter 作为评估服务器和推理引擎之间的桥梁，内置了对 OpenAI REST API、Gemini、Anthropic 等的适配器。Backend 是实际的推理引擎，官方支持 vLLM、SGLang、LMDeploy 等高性能后端。实现了基于 SQLite 的缓存机制——对输入数据（文本、图像、参数）计算哈希值作为唯一键，避免重复推理。

3. **通信协议**：六步生命周期——get_tasks（获取可用任务）→ task_info（查询任务信息）→ get_meta（获取元数据）→ get_data(i)（获取评估项）→ 推理 → submit(result)（提交结果）。每步支持分布式和并行化评估。

4. **评估加速**：利用 vLLM、SGLang 等推理加速工具，配合异步数据加载（如数据预取），显著减少等待时间。

### 评估能力覆盖

- **多模态理解**：覆盖通用知识、数学、图表理解、视觉感知、文字识别五大能力维度。数据集包括 MMMU、MMMU-Pro、MathVision、MathVerse、Blink、CharXiv 等公开数据集 + 自建主观评测集和文字识别评测集。区分中英文。
- **多模态生成**：评估提示一致性、真实性、美学质量、安全性四个维度。自动指标包括 VQAScore、Q-Align、VideoScore，结合人工评估（3位评估者打分取平均）。

## 实验关键数据

### VLM 理解评估

| 模型 | 平均排名 | 通用知识 | 数学 | 图表 | 视觉感知 | 文字识别 |
|------|----------|---------|------|------|----------|---------|
| Gemini-2.0-pro | 2.1 | 64.00 | 52.18 | 67.06 | 62.73 | 78.22 |
| Qwen2.5-VL-72B | 4.6 | 61.30 | 35.45 | 67.00 | 60.90 | 77.63 |
| Claude-3.7-Sonnet | 6.9 | 58.98 | 49.31 | 71.19 | 66.55 | 67.69 |
| GPT-4o-2024-11 | 8.1 | 58.39 | 30.82 | 65.50 | 62.02 | 70.31 |
| InternVL2.5-78B | 6.9 | 61.31 | 37.80 | 60.14 | 62.97 | 70.87 |

### 文生图评估

| 模型 | 加权分 | 一致性 | 真实性 | 美学 | 安全性 | VQAScore |
|------|--------|--------|--------|------|--------|----------|
| Hunyuan-Image | 73.00 | 67.93 | 66.67 | 78.50 | 100.0 | 73.76 |
| DALL-E 3 | 70.12 | 70.24 | 57.51 | 68.38 | 98.21 | 81.82 |
| FLUX.1 schnell | 68.39 | 61.95 | 64.34 | 73.18 | 99.11 | 77.95 |
| Midjourney v6.1 | 65.91 | 67.56 | 46.95 | 64.58 | 98.21 | 77.63 |

### 消融/分析

| 分析维度 | 关键发现 |
|---------|---------|
| 开源 vs 闭源 VLM | Qwen2.5 系列超越多个早期闭源模型，差距缩小 |
| 跨语言表现 | Mistral-3.1、Claude-3.7 在中文评估上明显弱于英文 |
| 自动 vs 人工评估 | 一致性维度 VQAScore 与人工评估 Pearson 相关仅 0.76 |
| 美学评估 | OneAlign-Aesthetic 与人工评估相关仅 0.59 |

### 关键发现

- **开源模型进步显著**：Qwen2.5-VL-72B 在多项能力上超越 GPT-4o 和 Claude-3.5-Sonnet
- **跨语言泛化仍是挑战**：部分模型英文表现远优于中文，文化适应性不足
- **VLM 在空间推理、计数、遮挡等经典CV任务上仍不稳定**
- **闭源 T2I 模型整体优于开源**，但自动评估指标与人工判断存在明显差距
- **解耦架构**有效解决环境冲突问题，支持灵活的资源分配

## 亮点与洞察

- 解耦推理与评估的架构设计是核心创新，解决了实际工程中模型评估的痛点
- 统一覆盖理解和生成两大类多模态任务，填补了现有框架的空白
- 提供了目前最全面的 VLM 和 T2I 模型横向比较，数据具有参考价值
- 基于 SQLite 的缓存机制有效避免重复推理，实用性强
- 已集成到 FlagEval 平台和 HuggingFace Spaces，可直接使用

## 局限与展望

- 评估方法的覆盖仍有限——尚未纳入多轮对话、交互式游戏和高级推理能力评估
- 生成任务中自动评估与人工评估差距大，仍需依赖人工标注
- 自建数据集未公开具体构成细节，可复现性有限
- 文生视频评估仅涉及 148 条 prompt，规模较小
- 未对评估框架本身的效率（如通信开销）做定量分析

## 相关工作与启发

- 与 Lmms-Eval 相比，FlagEvalMM 的主要优势在于支持生成任务评估和更灵活的后端选择
- 与 VLMEvalKit 相比，插件式设计避免了侵入式代码修改
- 评估结果揭示的自动指标与人工评估的差距，对设计更好的自动评估指标有启示
- 解耦架构的设计思路可推广到其他大模型评估场景

## 评分

- **新颖性**: ⭐⭐⭐ — 解耦架构思路有工程创新，但核心是系统设计而非方法创新
- **实验充分度**: ⭐⭐⭐⭐ — 覆盖 50+ VLM 和 30+ 生成模型的大规模评测
- **写作质量**: ⭐⭐⭐ — 架构描述清晰但部分实验细节（自建数据集、具体流程）不够详尽
- **价值**: ⭐⭐⭐⭐ — 作为开源评估工具具有很高的实用价值，横向评测数据有参考意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] UPME: An Unsupervised Peer Review Framework for Multimodal Large Language Model Evaluation](../../CVPR2025/multimodal_vlm/upme_an_unsupervised_peer_review_framework_for_multimodal_large_language_model_e.md)
- [\[CVPR 2026\] 4DWorldBench: A Comprehensive Evaluation Framework for 3D/4D World Generation Models](../../CVPR2026/multimodal_vlm/4dworldbench_a_comprehensive_evaluation_framework_for_3d4d_world_generation_mode.md)
- [\[CVPR 2026\] SpatialScore: Towards Comprehensive Evaluation for Spatial Intelligence](../../CVPR2026/multimodal_vlm/spatialscore_towards_comprehensive_evaluation_for_spatial_intelligence.md)
- [\[ACL 2025\] Hidden in Plain Sight: Evaluation of the Deception Detection Capabilities of LLMs in Multimodal Settings](hidden_in_plain_sight_evaluation_of_the_deception_detection_capabilities_of_llms.md)
- [\[ACL 2025\] Scalable Vision Language Model Training via High Quality Data Curation](scalable_vision_language_model_training_via_high_quality_data_curation.md)

</div>

<!-- RELATED:END -->
