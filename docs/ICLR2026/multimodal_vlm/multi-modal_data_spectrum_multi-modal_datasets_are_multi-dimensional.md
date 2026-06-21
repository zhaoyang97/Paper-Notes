---
title: >-
  [论文解读] Multi-modal Data Spectrum: Multi-modal Datasets are Multi-dimensional
description: >-
  [ICLR 2026][多模态VLM][多模态学习] 通过大规模实证研究量化了23个VQA基准中的模态内依赖和模态间依赖，揭示许多旨在消除文本偏置的基准反而引入了图像偏置，提出了多模态数据集的多维度刻画框架。 1. 领域现状： 多模态大语言模型（MLLMs）快速发展，伴随超过200个评测基准的涌现，但对这些基准实际测量什么缺…
tags:
  - "ICLR 2026"
  - "多模态VLM"
  - "多模态学习"
  - "基准评测"
  - "模态依赖"
  - "VQA"
  - "MLLM"
---

# Multi-modal Data Spectrum: Multi-modal Datasets are Multi-dimensional

**会议**: ICLR 2026  
**arXiv**: [2509.23499](https://arxiv.org/abs/2509.23499)  
**代码**: [GitHub](https://github.com/divyam3897/multimodal-spectrum)  
**领域**: 信号通信  
**关键词**: 多模态学习, 基准评测, 模态依赖, VQA, MLLM

## 一句话总结

通过大规模实证研究量化了23个VQA基准中的模态内依赖和模态间依赖，揭示许多旨在消除文本偏置的基准反而引入了图像偏置，提出了多模态数据集的多维度刻画框架。

## 研究背景与动机

1. **领域现状**: 多模态大语言模型（MLLMs）快速发展，伴随超过200个评测基准的涌现，但对这些基准实际测量什么缺乏系统性理解。

2. **现有痛点**: 基准选择缺乏科学依据——Gemini 1.5和2.5评测使用的数据集集合不同且缺乏为何更换的说明。模型性能提升是真正的多模态能力进步还是对单模态捷径的利用，难以判断。

3. **核心矛盾**: 基准开发陷入"猫鼠游戏"循环——新数据集被设计来消除特定的单模态偏置，但随后被发现引入了新的偏置（如VQA→VQAv2→VQA-CP→MMMU→MMMU-Pro）。

4. **本文目标**: 对现有多模态基准进行系统化的模态依赖分析，提供定量刻画框架。

5. **切入角度**: 通过模态置换（shuffling）方法，在保持单模态边际分布不变的前提下，破坏模态间依赖，测量模型在四种输入条件下的性能变化。

6. **核心 idea**: 多模态数据集本质上是多维度的，其中模态内依赖（单模态即可回答）和模态间依赖（需联合推理）的强度在基准内部和基准之间都有显著差异。

## 方法详解

### 整体框架

诊断的核心是把一条多模态样本拆成图像、文本两路输入，再用四种"输入配置"喂给同一个模型，看准确率怎么塌。配置依次是：正常配对 $\mathcal{M}(f_\theta(\mathbf{x_1}, \mathbf{x_2}), \mathbf{y})$、仅图像（文本换成数据集内随机样本的文本）、仅文本（图像换成随机样本的图像）、以及两路都打乱的全随机基线。某一路被打乱后准确率掉得越少，说明模型越能绕过另一模态、靠这一路单独答题，各模态的独立贡献与二者的交互贡献便由此被量化出来。

### 关键设计

**1. 用模态置换而非置零来切断模态间依赖**

要分离某个模态的贡献，最直接的想法是把它抹掉——给空白图像或空字符串。但这会把输入推到训练分布之外（out-of-distribution），模型面对没见过的"空输入"会产生不可预测的行为，污染测量结果。本文沿用并改造感知分数（Perceptual Score, Gat et al. 2021）的置换思路：把目标模态替换成同一数据集中另一条样本的对应模态，图像还是真实图像、问题还是合法问题，只是它与标签 $\mathbf{y}$ 的对齐关系被打断。这样每个模态的边际分布原封不动，被破坏的只有模态间的关联，性能落差才干净地对应到"模态间依赖"这一项上——而置零或扰动会让落差里混进"输入异常"的噪声。

**2. 下沉到子类别粒度，避免聚合分数掩盖局部偏置**

在整库上算一个平均落差，很容易得出"这个数据集是平衡的"的错觉，因为强单模态依赖的子群会被弱依赖的子群稀释掉。本文按问题类型、对象类别等属性把数据集切成子集，在每个子集上独立跑一遍四配置诊断。结果显示，全局看似需要双模态交互的数据集，在特定子类别里可能几乎只靠一种模态——比如 ScienceQA 的高年级问题其实强依赖文本先验，这种结构性偏置只有放大到子群层面才看得见。

**3. 跨模型规模与架构做边际化，确认依赖来自数据本身**

单看一个模型的落差，分不清是数据集天生有捷径，还是这个模型恰好偏科。本文把模态依赖看成数据和模型的联合函数，要拿到数据的固有特性就得对模型这一维做边际化（marginalize）：用 Cambrian-1 的 8B/13B/34B 三个规模做多数投票（majority-vote）集成，再额外用 LLaVA-Next、Qwen 系列等不同架构复核。当偏置模式在这些差异巨大的模型上保持一致时，才有底气说它是基准的属性而非某个模型的怪癖。

### 度量与读数

本文是诊断性工作，不训练任何模型，全部结论来自对现成模型的评测，度量统一用多选题（MCVQA）准确率。四种配置给出四个准确率：以两路都打乱的全随机配置作底线，正常配对相对底线的提升衡量数据集整体的可解性，而仅图像、仅文本两种配置相对底线的提升则分别量化图像内依赖与文本内依赖的强度；当两种单模态配置相对底线几乎不涨、只有正常配对显著高出时，剩下的提升才归到真正需要联合推理的模态间依赖。

## 实验关键数据

### 主实验

23个基准的模态依赖分类：

| 依赖类型 | 代表数据集 | 特征 |
|---------|-----------|------|
| 仅模态间依赖 | MME, POPE, COCO, V*Bench | 极少，仅4/23个数据集 |
| 含文本内依赖 | GQA(+26%), ScienceQA(+17.5%), MMMU(+11.35%), AI2D(+34.94%) | 仅靠文本即可大幅超越随机 |
| 含图像内依赖 | MMBench(+41%), SEED, TextVQA, MMMU-Pro, MMVP | 消除文本偏置反而引入了图像偏置 |

### 消融实验

| 配置 | 关键发现 | 说明 |
|------|---------|------|
| 模型规模增大(8B→34B) | 单模态偏置不减反增 | MMMU上更大模型增加了图像和文本依赖 |
| 不同模型类型 | 偏置模式跨模型一致 | Cambrian、LLaVA-Next、Qwen模型表现类似 |
| 子类别分析 | 聚合指标掩盖子群偏置 | ScienceQA高年级问题几乎全靠文本 |

### 关键发现

- **仅4/23基准** 表现出纯粹的模态间依赖，远少于预期
- 旨在消除文本偏置的新基准（如MMBench、SEED等）反而引入了图像偏置——用一种单模态捷径替换了另一种
- 模型规模增大不能缓解单模态偏置，反而可能加剧
- 子类别分析显示，即使全局平衡的数据集在特定子集上仍有强烈偏置

## 亮点与洞察

- **揭示了多模态评测的根本问题**: 用单一聚合分数评价模型是不够的，需要同时报告单模态基线性能
- **为"模型进步是否真实"提供了判断工具**: 性能提升可能只是模型更善于利用单模态依赖
- **设计新基准的实用指南**: 核心目标应是需要两个模态共同回答，而非仅消除某一模态的依赖
- **"猫鼠游戏"的深刻洞察**: 只有系统化量化模态依赖才能打破这一循环

## 局限与展望

- 分析限于多选VQA格式，未涵盖开放式生成任务
- 模态置换方法在选项本身包含模态信息时可能有局限
- 需要扩展到模型主动弃权（abstention）能力的评估
- 未来应推进开放式答案生成和评估的基准设计

## 相关工作与启发

- Perceptual Score (Gat et al., 2021) 提供了基础方法论，本文将其扩展到23个基准的大规模分析
- 与NAS中"搜索空间设计比搜索算法重要"的类似洞察——基准设计比模型改进更根本
- 启发：评价多模态模型时应同时报告模态特定基线，形成社区规范

## 评分

- 新颖性: ⭐⭐⭐⭐ 系统性分析揭示了被忽视的重要问题
- 实验充分度: ⭐⭐⭐⭐⭐ 23个基准、多个模型规模和类型、子类别分析
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，可视化出色
- 价值: ⭐⭐⭐⭐ 对多模态评测社区具有重要的方法论指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Patch-as-Decodable-Token: Towards Unified Multi-Modal Vision Tasks in MLLMs](patch-as-decodable-token_towards_unified_multi-modal_vision_tasks_in_mllms.md)
- [\[ICLR 2026\] TableDART: Dynamic Adaptive Multi-Modal Routing for Table Understanding](tabledart_dynamic_adaptive_multi-modal_routing_for_table_understanding.md)
- [\[AAAI 2026\] ImageBindDC: Compressing Multi-modal Data with ImageBind-based Condensation](../../AAAI2026/multimodal_vlm/imagebinddc_compressing_multi-modal_data_with_imagebind-based_condensation.md)
- [\[CVPR 2026\] Hierarchical Attacks for Multi-Modal Multi-Agent Reasoning](../../CVPR2026/multimodal_vlm/hierarchical_attacks_for_multi-modal_multi-agent_reasoning.md)
- [\[ICLR 2026\] Calibrated Information Bottleneck for Trusted Multi-modal Clustering](calibrated_information_bottleneck_for_trusted_multi-modal_clustering.md)

</div>

<!-- RELATED:END -->
