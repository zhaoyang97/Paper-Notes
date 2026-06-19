---
title: >-
  [论文解读] MODA: MOdular Duplex Attention for Multimodal Perception, Cognition, and Emotion Understanding
description: >-
  [ICML 2025 (Spotlight, Top 2.6%)][多模态VLM][多模态大语言模型] 针对多模态大语言模型中跨模态注意力不一致与逐层衰减的"注意力缺失障碍"问题，提出模块化双工注意力机制MODA，通过将注意力解耦为模态内自精炼与模态间交互两路，并借助Duplex Aligner和自适应掩码注意力实现"先对齐再校正"的策略，在21个感知、认知与情感基准上验证了有效性。
tags:
  - "ICML 2025 (Spotlight, Top 2.6%)"
  - "多模态VLM"
  - "多模态大语言模型"
  - "注意力机制"
  - "跨模态对齐"
  - "情感理解"
  - "认知理解"
---

# MODA: MOdular Duplex Attention for Multimodal Perception, Cognition, and Emotion Understanding

**会议**: ICML 2025 (Spotlight, Top 2.6%)  
**arXiv**: [2507.04635](https://arxiv.org/abs/2507.04635)  
**作者**: Zhicheng Zhang, Wuyou Xia, Chenxi Zhao, Zhou Yan, Xiaoqiang Liu, Yongjie Zhu, Wenyu Qin, Pengfei Wan, Di Zhang, Jufeng Yang
**代码**: [https://zzcheng.top/MODA](https://zzcheng.top/MODA)  
**领域**: 多模态VLM  
**关键词**: 多模态大语言模型, 注意力机制, 跨模态对齐, 情感理解, 认知理解

## 一句话总结

针对多模态大语言模型中跨模态注意力不一致与逐层衰减的"注意力缺失障碍"问题，提出模块化双工注意力机制MODA，通过将注意力解耦为模态内自精炼与模态间交互两路，并借助Duplex Aligner和自适应掩码注意力实现"先对齐再校正"的策略，在21个感知、认知与情感基准上验证了有效性。

## 研究背景与动机

### 问题背景
多模态大语言模型（MLLMs）在整合视觉、语言等多模态数据方面展现了强大能力，已成为通往通用人工智能（AGI）的重要路径。现有MLLM主要通过注意力机制实现多模态token的混合与融合，在基本感知任务上表现良好。然而，更高层次的认知理解（如角色扮演、代码生成中的决策判断）和情感理解（如讽刺检测、微表情识别）对细粒度跨模态信息融合的要求远超当前模型的能力。

### 已有工作的不足
- **语言主导偏差**：主流方法过度聚焦于以语言为中心的微调，忽视了视觉模态在注意力混合中的充分利用
- **视觉模态被抑制**：MMVP等工作揭示现有MLLM无法充分激活视觉模态，尤其在低层视觉属性处理上存在问题
- **公开基准表现堪忧**：3个SOTA模型在HFM数据集的二分类讽刺检测中仅达到约50%的准确率（接近随机猜测）
- **Cambrian-1**通过空间视觉聚合器增强视觉特征，缓解但未从根本上解决注意力不均衡问题

### 核心发现：注意力缺失障碍（Attention Deficit Disorder）
作者深入分析了MLLM中多模态token通过注意力混合的过程，发现了两个关键问题：

**跨模态注意力不一致**：在MLLM的不同层中，视觉和语言模态的注意力分配呈现严重偏斜，注意力分数偏向语言成分

**逐层注意力衰减**：随着网络层数加深，视觉模态的注意力激活逐层衰减，进一步加剧模态失衡

以《教父》片段为例，SOTA MLLM无法捕捉角色的细微眼神等细粒度视觉特征，导致情感理解错误。定量分析表明跨层注意力存在高达63%的差异，不同模态间的注意力分数差距可达10倍。

### 核心动机
多模态注意力在自模态交互与跨模态交互间天然存在不平衡，简单的统一注意力无法兼顾两者。需要一种能够显式分离和调控这两个组件的机制，在保持各模态独立特性的同时实现有效的跨模态特征对齐。

## 方法详解

### 整体框架
MODA基于标准MLLM架构（视觉编码器 + 连接器 + LLM），核心创新在于用MOdular Duplex Attention替换LLM中的标准注意力层。整体遵循"先对齐再校正"（Correct-after-Align）的策略，将模态对齐与跨层token混合有效解耦。

### 关键设计1：Duplex (V/T)-Aligner

Duplex Aligner是MODA的核心组件，负责将多模态token映射到共享的双模态表示空间。

- **双工空间构建**：定义视觉（V）和文本（T）两组基向量（basis vectors），分别构建对应的Gram矩阵
- **对齐阶段**：输入的视觉token和语言token被分别映射到这两个双工模态空间中。在视觉空间中，视觉token保持自身特征的同时，语言token被投影到视觉空间以实现跨模态交互；文本空间同理
- **双向交互**：通过双工映射，视觉与语言模态能够在各自的"主场"空间中进行交互，避免单一空间带来的模态偏向

### 关键设计2：模块化注意力分解

MODA将传统的统一注意力显式拆分为两个模块：

1. **自模态注意力（Self-Modal Attention）**：
    - 专注于捕获单一模态内部的固有关系
    - 视觉token之间、语言token之间分别进行自注意力计算
    - 保留各模态的独立表征能力，防止跨模态干扰

2. **跨模态注意力（Cross-Modal Attention）**：
    - 负责不同模态间的特征对齐与信息交换
    - 在Duplex Aligner映射后的双工空间中进行
    - 确保视觉和语言特征能够有效地相互补充

### 关键设计3：Modular Masked Attention

为进一步增强模型灵活性，MODA引入自适应掩码注意力机制：

- **自定义掩码模式**：根据不同模态和不同任务类型，设计可定制的注意力掩码
- **自适应聚焦**：模型能够根据任务需求动态调整对不同模态的关注程度
- **校正阶段**：在对齐完成后，通过掩码注意力对注意力分数进行校正，确保跨模态注意力的一致性，解决注意力衰减问题

### 训练策略

MODA采用多阶段训练策略，在预训练的MLLM基础上进行：

- **阶段一：对齐预训练** — 固定视觉编码器和LLM权重，训练Duplex Aligner和连接器，使用大规模图文对齐数据建立初始的双工模态空间映射
- **阶段二：指令微调** — 解冻MODA注意力层，在多样化的指令跟随数据上进行端到端微调，覆盖感知、认知和情感三类任务
- **模块化设计优势** — MODA的注意力模块可以作为即插即用组件替换到现有MLLM中，不改变原有架构的其余部分

## 实验关键数据

### 感知任务评估

MODA在多个视觉感知基准上进行了全面评估，涵盖一般视觉问答、视觉推理和幻觉检测等任务：

| 基准类型 | 基准名称 | 评估能力 | MODA表现 |
|---------|---------|---------|---------|
| 一般视觉问答 | MMMU | 多学科理解 | 领先基线 |
| 一般视觉问答 | MMBench | 多维感知 | 领先基线 |
| 视觉推理 | MMVP | 视觉模式 | 显著提升 |
| 幻觉检测 | POPE | 对象幻觉 | 有效改善 |
| 文档理解 | DocVQA | OCR理解 | 具竞争力 |
| 细粒度感知 | 多个基准 | 细节识别 | 优于基线 |

MODA在MMVP等要求细粒度视觉感知的基准上提升尤为显著，直接验证了解决注意力缺失障碍的有效性。

### 认知与情感任务评估

MODA在高级理解任务上相比基线有质的提升：

| 任务类型 | 基准名称 | 具体能力 | 关键发现 |
|---------|---------|---------|---------|
| 讽刺检测 | HFM | 多模态讽刺 | 从约50%随机水平大幅提升 |
| 情感分析 | 多个基准 | 细粒度情感 | 显著优于SOTA |
| 角色认知 | 角色扮演基准 | 决策判断 | 一致性改善 |
| 情感推理 | 情感推理基准 | 因果推理 | 跨模态推理能力增强 |
| 微表情识别 | 微表情基准 | 细微特征 | 视觉注意力改善明显 |

共21个基准全面覆盖感知（perception）、认知（cognition）和情感（emotion）三大维度，MODA在各维度均表现出一致的优势。

### 消融实验关键发现

- **注意力分解的必要性**：去除自模态/跨模态分解后，性能显著下降，验证了模块化设计的有效性
- **Duplex Aligner的贡献**：双工空间映射相比单一空间映射带来明显提升
- **Modular Mask的作用**：自适应掩码在不同任务类型上提供不同程度的增益，情感任务受益最大
- **注意力一致性改善**：可视化显示MODA有效降低了跨层注意力差异，从63%差异降至可控范围

## 亮点与洞察

- **问题发现深刻**：首次系统化定义并量化了MLLM中的"注意力缺失障碍"问题，63%跨层差异和10倍模态间注意力差距的发现具有重要的诊断价值
- **设计理念优雅**：将注意力解耦为自模态和跨模态两路的思路简洁有效，"先对齐再校正"策略具有很强的直觉性
- **双工空间创新**：通过Gram矩阵构建的双工模态空间，让每个模态都有"主场优势"，避免了传统方法中单一共享空间导致的模态偏向
- **任务覆盖全面**：21个基准横跨感知、认知、情感三级能力层次，体系化验证了方法的泛化性
- **即插即用设计**：MODA注意力模块可替换现有MLLM的注意力层，工程友好度高
- **Spotlight接收**：ICML 2025 Top 2.6%，学术认可度很高

## 局限与展望

- **计算开销**：双工注意力在自模态和跨模态分别计算，理论上比标准注意力增加计算量，效率-性能权衡待评估
- **模态扩展性**：当前设计针对视觉-语言双模态，扩展到音频、视频等更多模态时，双工空间的构建方式需要重新设计
- **基向量选择**：Duplex Aligner中基向量的初始化和学习策略对最终性能的影响值得进一步研究
- **训练数据依赖**：多阶段训练策略需要大量多样化数据，对于资源受限场景可能不够友好
- **理论分析缺乏**：对注意力缺失障碍的分析以实验观察为主，缺乏对注意力衰减机制的理论解释
- **缓存不完整**：本笔记基于截断的arXiv缓存撰写，具体实验数值和更多细节可参阅原文

## 相关工作与启发

### MLLM注意力机制改进
- **Cambrian-1 (Tong et al., 2024a)**：引入空间视觉聚合器增强视觉特征，但未从注意力机制层面解决模态失衡
- **MMVP (Tong et al., 2024b)**：揭示MLLM未能充分激活视觉模态的问题，本文在此基础上进一步定位到注意力机制
- **Flamingo (Alayrac et al., 2022)**：通过交叉注意力实现多模态集成，但交叉注意力仍存在模态偏向

### 情感与认知理解
- **EmotionCLIP (Yang et al., 2024)**：情感理解在多模态MLLM中的挑战，本文扩展到认知维度
- **角色认知研究 (Wang et al., 2024; Chen et al., 2024)**：认知任务要求细粒度跨模态推理，MODA通过改善注意力分配提供支撑

### 启发
MODA的注意力分解思路可推广到其他多模态融合场景。在注意力设计中，不应简单地将所有模态token统一混合，而是考虑模态特异性，为不同类型的交互（自模态 vs 跨模态）设计专门的机制。这一设计哲学对于构建更强的多模态agent（如具备情感感知能力的对话系统）有直接启发。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 注意力缺失障碍的概念新颖，双工注意力设计有创意
- 实验充分度: ⭐⭐⭐⭐⭐ — 21个基准覆盖三大维度，消融实验充分
- 写作质量: ⭐⭐⭐⭐ — 问题分析深入，论述清晰，图示直观
- 价值: ⭐⭐⭐⭐ — Spotlight论文，对MLLM注意力改进有重要参考价值，但计算开销待评估

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Seeing is Understanding: Unlocking Causal Attention into Modality-Mutual Attention for Multimodal LLMs](../../ICML2026/multimodal_vlm/seeing_is_understanding_unlocking_causal_attention_into_modality-mutual_attentio.md)
- [\[ACL 2025\] Aligning VLM Assistants with Personalized Situated Cognition](../../ACL2025/multimodal_vlm/aligning_vlm_assistants_with_personalized_situated.md)
- [\[CVPR 2025\] SmartCLIP: Modular Vision-language Alignment with Identification Guarantees](../../CVPR2025/multimodal_vlm/smartclip_modular_vision-language_alignment_with_identification_guarantees.md)
- [\[ACL 2025\] AkaCE: A Multimodal Multi-party Dataset for Emotion Recognition in Movie Dialogues](../../ACL2025/multimodal_vlm/akan_cinematic_emotions_ace_a_multimodal_multi-party_dataset_for_emotion_recogni.md)
- [\[ACL 2026\] Dynamic Emotion and Personality Profiling for Multimodal Deception Detection](../../ACL2026/multimodal_vlm/dynamic_emotion_and_personality_profiling_for_multimodal_deception_detection.md)

</div>

<!-- RELATED:END -->
