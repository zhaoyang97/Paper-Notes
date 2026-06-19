---
title: >-
  [论文解读] SPA-VL: A Comprehensive Safety Preference Alignment Dataset for Vision Language Models
description: >-
  [CVPR 2025][多模态VLM][安全对齐] SPA-VL 构建了一个包含 100,788 个四元组（问题、图像、优选回答、劣选回答）的大规模VLM安全偏好对齐数据集，覆盖6大领域/13类/53子类有害内容，基于12个VLM的多样化回答和全自动化标注流程，使用DPO/PPO训练后模型在安全性上大幅提升同时保持帮助性。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "安全对齐"
  - "偏好学习"
  - "RLHF"
  - "VLM安全"
  - "数据集"
---

# SPA-VL: A Comprehensive Safety Preference Alignment Dataset for Vision Language Models

**会议**: CVPR 2025  
**arXiv**: [2406.12030](https://arxiv.org/abs/2406.12030)  
**代码**: [https://sqrtizhang.github.io/SPA-VL/](https://sqrtizhang.github.io/SPA-VL/)  
**领域**: 多模态VLM  
**关键词**: 安全对齐, 偏好学习, RLHF, VLM安全, 数据集

## 一句话总结
SPA-VL 构建了一个包含 100,788 个四元组（问题、图像、优选回答、劣选回答）的大规模VLM安全偏好对齐数据集，覆盖6大领域/13类/53子类有害内容，基于12个VLM的多样化回答和全自动化标注流程，使用DPO/PPO训练后模型在安全性上大幅提升同时保持帮助性。

## 研究背景与动机
- **领域现状**: VLM（如GPT-4V、LLaVA）在多模态理解上取得显著进展，但面临严重的安全问题。虽然LLM已经过安全对齐，但视觉编码器的对齐较弱，容易通过视觉模态发起攻击
- **现有痛点**: (1) 大多数VLM安全工作集中在评估基准或攻击检测，缺少用于训练的大规模高质量数据集；(2) 现有安全方法多基于SFT或推理时防护，RLHF方法探索不足；(3) 无害输入也可能导致不安全输出，需要全面的对齐数据
- **核心矛盾**: 安全性 vs 帮助性——过度安全会导致模型拒绝正常请求，需要在两者间取得平衡
- **本文解决什么**: 构建首个大规模、全自动化的VLM安全偏好对齐数据集，使 RLHF 安全对齐成为可能
- **切入角度**: 从数据覆盖广度（53个子类）、回答多样性（12个模型）、问题复杂度（3种问题类型）三个维度确保数据质量
- **核心idea**: 大规模 + 多样化 + 多目标（无害+有用）的偏好数据 = 更好的安全对齐

## 方法详解

### 整体框架
SPA-VL 的构建分三阶段：(1) 图像收集：基于层级化有害内容分类体系从 LAION-5B 中检索相关图像；(2) 问题构造：为每张图像生成3类问题（简单问题、困难问题、困难陈述），使用 Gemini 生成并精炼；(3) 偏好标注：从12个VLM收集多样化回答，使用 MD-Judge 进行安全性分类，再用 GPT-4V 基于无害+有用标准进行偏好排序。整个流程全自动化。

### 关键设计
1. **层级化有害内容分类体系**:
    - 功能：确保数据集对有害内容的全面覆盖
    - 核心思路：建立6个主领域（毒性、不公平、信息侵蚀、安全威胁、非法活动、欺诈等）→ 15个二级类别 → 53个三级类别的层级结构。参考 OpenAI、LLaMA、Gemini、Claude 的使用政策及 Llama Guard 的安全指南
    - 设计动机：有害内容类型繁多且层次丰富，粗粒度分类会导致特定类型覆盖不足，三级分类确保细粒度和系统性

2. **多模型多类型问答生成**:
    - 功能：提供多样化的问题和回答以增强对齐效果
    - 核心思路：(1) 问题方面：为每张图像用 Gemini 生成 easy question（直接相关）、hard question（精炼后更复杂）、hard statement（陈述形式）3种问题，覆盖不同攻击难度；(2) 回答方面：从12个不同VLM（含开源和闭源）收集回答，每个问题随机选2个不同安全等级的模型回答作为偏好对
    - 设计动机：单一问题类型训练的模型鲁棒性不足（消融实验证实）；多源回答减少特定模型偏见，确保优选/劣选回答涵盖不同安全水平

3. **全自动偏好标注流程**:
    - 功能：无需人工标注即可生成高质量偏好数据
    - 核心思路：先用 MD-Judge 对回答进行安全/不安全分类，然后从不同安全等级的模型组中选取回答对，最后用 GPT-4V 同时基于无害性和有用性两个标准评估偏好（交换回答顺序查询两次以消除位置偏见）
    - 设计动机：人工标注成本极高且涉及有害内容，全自动流程不仅可扩展，还避免了标注者接触有害内容

### 损失函数 / 训练策略
- **DPO训练**: 冻结视觉编码器，更新投影层和LLM权重
- **PPO训练**: 同样冻结视觉编码器，使用奖励模型指导训练
- 基础模型：LLaVA-1.5 (7B)
- 训练数据从100到90K样本进行了完整的数据规模消融

## 实验关键数据

### 主实验 — 安全性评估

| 模型 | MM-SafetyBench Avg↓ | AdvBench vanilla↓ | AdvBench suffix↓ | HarmEval USR↓ |
|------|---------------------|-------------------|-------------------|--------------|
| LLaVA-7B (baseline) | 20.54 | 98.08 | 99.81 | 44.15 |
| LLaVA + SPA-VL-DPO | **0.60** | **0.00** | **0.00** | **0.00** |
| LLaVA + SPA-VL-PPO | 0.45 | 0.19 | 2.12 | 0.00 |
| Gemini-1.5-pro | 0.00 | 0.38 | 0.38 | 1.89 |
| mPLUG-Owl-7B | 21.88 | 100 | 100 | 52.45 |

### 消融实验 — 数据规模影响

| 数据规模 | MM-SafetyBench Avg↓ | AdvBench vanilla↓ | HarmEval USR↓ | Help Score↑ |
|---------|---------------------|-------------------|--------------|-------------|
| 100 (DPO) | 19.94 | 97.89 | 43.40 | 51.00 |
| 1K | 18.75 | 91.54 | 26.04 | 58.50 |
| 10K | 2.53 | 5.77 | 0.38 | 63.00 |
| 90K | 1.49 | 0.00 | 0.75 | 70.00 |

### 关键发现
- 训练后模型的攻击成功率从20%+骤降至接近0%，安全性提升极其显著
- 数据规模是关键因素：从100→90K，安全性和帮助性同时持续提升，不存在此消彼长
- 回答来源多样性至关重要：仅用安全模型的回答对训练（Safe Group）在AdvBench suffix上仍有65%攻击成功率，而使用全部12个模型的回答（All Group）降至6.54%
- 混合3种问题类型比单一类型更有效，安全性全面优于任何单一类型
- DPO和PPO方法均有效，DPO在安全性上略优，PPO在帮助性上略优
- 训练时同时更新投影层和LLM比仅更新LLM效果更好

## 亮点与洞察
- 数据集规模（100K+）和覆盖面（53子类）在VLM安全领域目前最全面
- 全自动构建流程的设计极具实用价值，避免了人工接触有害内容的伦理问题
- "安全性和帮助性同时提升"的实验结论打破了两者对立的直觉，说明好的偏好数据可以两全
- 12个不同模型作为回答源的设计确保了偏好数据的多样性和泛化性

## 局限与展望
- 数据中的有害图像来自 LAION-5B 通过CLIP检索，可能存在检索噪声
- GPT-4V 作为偏好标注器本身也有偏见，无完全可靠的ground truth
- 仅在 LLaVA-1.5 (7B) 上验证，对更大模型和更新架构的效果未知
- 安全分类体系虽然全面但偏静态，随着社会文化变化需要持续更新
- Gemini 的jailbreak策略用于生成困难问题存在一定的安全和伦理争议

## 相关工作与启发
- **vs VLGuard**: VLGuard 仅有2000张SFT训练图像，SPA-VL提供100K+ RLHF偏好对，规模和方法论深度都更进一步
- **vs RLAIF-V**: RLAIF-V 主要解决幻觉问题，SPA-VL 专注安全对齐，目标不同但DPO方法论互补
- **vs MLLM-Protector**: MLLM-Protector 是推理时的即插即用防护，SPA-VL 是训练时的根本性对齐，两者可结合使用
- **vs LLaVA-RLHF**: LLaVA-RLHF仅训练LLM层，SPA-VL证明同时更新投影层和LLM效果更好
- **vs MM-SafetyBench**: MM-SafetyBench是评估基准，SPA-VL是训练数据集，两者角色互补

## 评分
- 新颖性: ⭐⭐⭐ 核心贡献是数据集而非方法创新，但数据集的规模和系统性确实填补了空白
- 实验充分度: ⭐⭐⭐⭐⭐ 数据规模、模型多样性、问题类型、训练策略的消融极其充分
- 写作质量: ⭐⭐⭐⭐ 数据集构建流程描述清晰，统计分析详尽
- 价值: ⭐⭐⭐⭐ 作为首个大规模VLM安全偏好数据集，对后续安全对齐研究有重要支撑作用

---

> 本笔记基于论文全文阅读生成，覆盖了 Dataset Construction、Experiments 和 Analysis 全部内容。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Task Preference Optimization: Improving Multimodal Large Language Models with Vision Task Alignment](task_preference_optimization_improving_multimodal_large_language_models_with_vis.md)
- [\[CVPR 2025\] Florence-VL: Enhancing Vision-Language Models with Generative Vision Encoder and Depth-Breadth Fusion](florence-vl_enhancing_vision-language_models_with_generative_vision_encoder_and_.md)
- [\[CVPR 2025\] Debiasing Multimodal Large Language Models via Noise-Aware Preference Optimization](debiasing_multimodal_large_language_models_via_noise-aware_preference_optimizati.md)
- [\[CVPR 2025\] Post-pre-training for Modality Alignment in Vision-Language Foundation Models](post-pre-training_for_modality_alignment_in_vision-language_foundation_models.md)
- [\[CVPR 2025\] SmartCLIP: Modular Vision-language Alignment with Identification Guarantees](smartclip_modular_vision-language_alignment_with_identification_guarantees.md)

</div>

<!-- RELATED:END -->
