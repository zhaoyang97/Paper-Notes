---
title: >-
  [论文解读] Octopus: Alleviating Hallucination via Dynamic Contrastive Decoding
description: >-
  [CVPR 2025][幻觉检测][幻觉缓解] 本文通过大量实验揭示了 LVLM 幻觉成因的混合性——不同样本和不同生成步骤面临不同类型的幻觉挑战，据此提出 Octopus 框架，利用可学习的 decision token 和 transformer block 在每个生成步自适应选择最合适的对比解码（CD）策略，通过 DPO 优化，在四个基准上全面超越现有 CD 方法。
tags:
  - "CVPR 2025"
  - "幻觉检测"
  - "幻觉缓解"
  - "对比解码"
  - "动态策略选择"
  - "大视觉语言模型"
  - "DPO"
---

# Octopus: Alleviating Hallucination via Dynamic Contrastive Decoding

**会议**: CVPR 2025  
**arXiv**: [2503.00361](https://arxiv.org/abs/2503.00361)  
**代码**: [https://github.com/LijunZhang01/Octopus](https://github.com/LijunZhang01/Octopus)  
**领域**: 多模态VLM  
**关键词**: 幻觉缓解, 对比解码, 动态策略选择, 大视觉语言模型, DPO

## 一句话总结

本文通过大量实验揭示了 LVLM 幻觉成因的混合性——不同样本和不同生成步骤面临不同类型的幻觉挑战，据此提出 Octopus 框架，利用可学习的 decision token 和 transformer block 在每个生成步自适应选择最合适的对比解码（CD）策略，通过 DPO 优化，在四个基准上全面超越现有 CD 方法。

## 研究背景与动机

**领域现状**：大视觉语言模型（LVLMs）如 LLaVA、InstructBLIP 在视觉理解和多模态推理上表现优异，但普遍存在幻觉问题——生成虚构的物体、错误的属性和不存在的关系。对比解码（Contrastive Decoding, CD）作为一种无需重训练的后处理方法成为缓解幻觉的重要方向。

**现有痛点**：
1. **单一策略的局限**：现有 CD 方法（VCD、M3ID、AVISC）各自针对特定类型的幻觉设计——VCD 对抗语言先验、M3ID 缓解视觉信息丢失、AVISC 减少注意力偏差。但它们都采用"一刀切"的方式，对所有样本和所有生成步使用相同的干扰策略。
2. **幻觉成因的复杂性被忽视**：没有工作系统研究过不同样本和不同 token 是否面临相同类型的幻觉。

**核心矛盾**：幻觉的成因是混合的（语言先验 + 视觉信息丢失 + 注意力偏差），但现有方法只能"头痛医头"，用单一策略应对所有情况，必然导致次优结果。

**本文切入角度**：先通过诊断实验证明幻觉的混合性，然后设计一个自适应框架，让模型在每个生成步自动选择最合适的 CD 策略。

**核心 idea**：像章鱼（Octopus）一样，用"眼睛"（decision token）识别幻觉类型，用多条"触手"（多种 CD 策略）分别应对不同的幻觉挑战。

## 方法详解

### 整体框架

Octopus 框架包含两个核心组件：(1) **决策模块**（"眼睛"）——一个 transformer block + 可学习 decision token，负责在每个生成步判断当前 token 面临哪种幻觉类型；(2) **执行模块**（"触手"）——多种现成的 CD 策略（VCD、M3ID、AVISC + null 动作），根据决策结果执行对应的对比操作。通过 DPO 优化决策模块参数，LVLM 参数保持冻结。

### 关键设计

1. **样本级幻觉诊断实验**:
    - 功能：证明单一 CD 策略无法覆盖所有幻觉样本
    - 核心思路：在 AMBER、Object-HalBench、MMHalBench 三个数据集上，分别使用 VCD、M3ID、AVISC 三种 CD 方法对 LLaVA-1.5-7B 的每个样本进行干预，统计每种方法有效纠正的样本比例。结果显示~60% 的样本只能被某一种特定 CD 策略纠正，三种策略同时有效的重叠区域仅约 10%
    - 设计动机：为动态策略选择提供实证依据——如果一种策略就能解决所有问题，就没必要做动态选择

2. **Token 级幻觉诊断实验**:
    - 功能：证明同一个样本中不同 token 的幻觉成因也不同
    - 核心思路：在 AMBER 数据集上，对每个描述中前 3 个幻觉名词使用枚举法测试不同 CD 策略组合。量化结果表明组合多种策略（如 strategy-1+3，strategy-1+2+3）显著优于单一策略。定性分析中，通过注意力图发现同一句话中 "sitting" 受注意力偏差影响（关注了视觉盲 token），"lying" 是因为对视觉信息关注不足，"person" 则完全依赖语言 token——三个词对应三种不同的幻觉成因
    - 设计动机：进一步将动态策略选择的粒度从样本级细化到 token 级

3. **Octopus 决策与执行架构**:
    - 功能：在每个生成步自适应选择最合适的 CD 策略
    - 核心思路：构建一个轻量 transformer block $\mathcal{O}_\phi$，将 LVLM 的隐状态序列 $H_t = \{h_i\}_{i=1}^t$（包含视觉、文本和已生成 token 的信息）与一个可学习的 decision token $eye \in \mathbb{R}^d$ 拼接，加上位置编码后送入 transformer block：$[h_{eye}^t; H_t'] = \mathcal{O}_\phi(\text{concat}[eye; H_t] + E_{pos})$。通过自注意力机制，$h_{eye}^t$ 聚合来自全序列的信息。然后经 MLP 映射为动作向量 $h_{act}^t \in \mathbb{R}^k$（$k=4$，对应三种 CD 策略 + null 动作），取 argmax 得到当前步的策略选择 $a_t$。最终生成一个完整的工作流 $\mathcal{A} = \{a_t\}_{t=1}^N$
    - 设计动机：利用 transformer 的自注意力机制让 decision token 综合考虑视觉输入、文本指令和已生成内容来做出全局性的策略决策

### 损失函数 / 训练策略

**DPO 优化**：由于 argmax 操作不可微分，且缺乏显式的决策标签，采用 Direct Preference Optimization 进行训练。

- **数据构建**：对每个样本随机生成 10 个不同的动作序列（每步随机选择 4 种动作之一），根据 CHAIR 指标将序列分为正样本（减少幻觉的工作流 $\mathcal{A}^+$）和负样本（增加幻觉的工作流 $\mathcal{A}^-$）
- **优化目标（去除参考模型的 DPO）**：$\max_{\mathcal{O}_\phi} \mathbb{E} \log \sigma(\beta \log \mathcal{O}_\phi(\mathcal{A}^+ | x) - \beta \log \mathcal{O}_\phi(\mathcal{A}^- | x))$，其中 $x = (v, q)$ 是视觉-文本输入，$\beta = 1$
- **关键特性**：仅优化 Octopus 的参数 $\phi$，LVLM 权重完全冻结，保证部署灵活性

**训练数据**：生成任务使用 MSCOCO 的 10,000 个偏好数据对，判别任务使用 7,000 个幻觉数据。训练在 4×3090 GPU 上完成，batch size 为 4。

## 实验关键数据

### 主实验（生成任务，LLaVA-1.5-7B）

| 数据集 | 指标 | LLaVA Base | +VCD | +M3ID | +AVISC | +Octopus | 提升 vs 最佳CD |
|--------|------|-----------|------|-------|--------|----------|---------------|
| AMBER | CHAIR↓ | 8.0 | 6.7 | 6.0 | 6.3 | **4.8** | -1.2 |
| AMBER | Cover↑ | 44.5 | 46.5 | 48.9 | 46.6 | **49.2** | +0.3 |
| AMBER | HalRate↓ | 31.0 | 27.8 | 26.0 | 25.6 | **23.4** | -2.2 |
| Object-HalBench | CHAIRs↓ | 25.0 | 23.6 | 23.2 | 22.1 | **20.8** | -1.3 |
| Object-HalBench | CHAIRi↓ | 9.2 | 8.4 | 7.3 | 7.8 | **6.6** | -0.7 |
| MMHalBench | Score↑ | 1.59 | 1.96 | 2.14 | 2.19 | **2.61** | +0.42 |

### 消融实验（AMBER 数据集）

| 配置 | CHAIR↓ | Cover↑ | Hal↓ | Cog↓ |
|------|--------|--------|------|------|
| LLaVA Base（无 CD） | 8.0 | 44.5 | 31.0 | 2.2 |
| 随机选择三种 CD 策略 | 6.9 | 46.2 | 26.1 | 2.2 |
| Octopus (Str1+Str2) | 5.5 | 48.7 | 25.8 | 1.5 |
| Octopus (Str1+Str3) | 5.7 | 48.2 | 25.3 | 1.5 |
| Octopus (Str2+Str3) | 5.5 | 48.4 | 26.2 | 1.6 |
| **Octopus (全部三种+null)** | **4.8** | **49.2** | **23.4** | **1.2** |

### 判别任务（LLaVA-1.5-7B）

| 数据集 | 指标 | LLaVA Base | +VCD | +Octopus | 提升 vs Base |
|--------|------|-----------|------|----------|-------------|
| AMBER | Acc | 67.00 | 67.30 | **76.70** | +9.70 |
| AMBER | F1 | 71.10 | 71.10 | **82.70** | +11.60 |
| POPE (ALL) | Acc | 82.04 | 82.96 | **85.79** | +3.75 |
| POPE (ALL) | F1 | 80.42 | 81.81 | **83.44** | +3.02 |

### 关键发现

- Octopus 在 AMBER 数据集上将 CHAIR 指标从 8.0 降至 4.8，相比 Base 减少约 40% 的幻觉
- 相比需要重训练整个模型的方法（如 HA-DPO、HALVA），Octopus 仍大幅领先，且无需修改 LVLM 权重
- 消融实验证明：(1) 即使随机选择 CD 策略也有帮助，但远不如 Octopus 的自适应选择；(2) 增加更多"触手"（CD 策略）可以持续提升性能，框架具有良好的可扩展性
- 不同 RL 优化方法（DPO、Monte-Carlo、PPO）都能获得满意结果，说明框架对优化算法不敏感
- 不同的评判标准（CHAIR、Cover、平均分）都可作为正负样本划分依据，框架具有跨领域适应性

## 亮点与洞察

1. **诊断先于治疗的研究范式**：先通过系统的样本级和 token 级诊断实验揭示幻觉的混合成因，再据此设计解决方案，这种"先理解问题再解决问题"的思路值得借鉴
2. **优雅的"元策略"设计**：不是发明新的 CD 方法，而是设计一个"策略选择器"来组合现有 CD 方法，这种元学习的思想使框架具有天然的可扩展性——未来任何新 CD 方法都可以直接作为新的"触手"接入
3. **LVLM 权重完全冻结**：仅训练轻量的 decision module，不修改部署模型的任何参数，实用性极强
4. **DPO 的巧妙应用**：将策略选择问题转化为偏好学习问题，通过随机采样 + CHAIR 评估自动构造正负样本对，避免了人工标注

## 局限与展望

- 目前仅集成了三种 CD 策略（VCD、M3ID、AVISC），随着新 CD 方法的出现，候选策略空间可以进一步扩大
- DPO 训练数据通过随机采样构造，质量可能不够理想，可探索更高效的数据构造方式
- 多次前向传播（每种 CD 策略需要额外的 distorted input 推理）带来推理延迟，实际部署时需要权衡效率
- token 级动态选择的计算开销较大——每个 token 都需要运行 decision module + 对应的 CD 前向传播
- 框架的有效性依赖于候选 CD 策略的多样性和互补性，如果候选策略高度同质化则收益有限

## 相关工作与启发

- **vs VCD/M3ID/AVISC**：这三种 CD 方法是 Octopus 的"触手"，各自只能覆盖~60% 的幻觉样本。Octopus 的核心贡献是学会在不同情况下选择最合适的策略
- **vs 重训练方法（HACL, POVID, HA-DPO）**：这些方法需要构造高质量数据并重训练 LVLM 参数，成本高且不适用于已部署模型。Octopus 作为即插即用方案，甚至性能优于这些重量级方法
- **vs OPERA/LCD/ICD**：这些也是后处理方法，但仍采用单一策略。Octopus 的动态组合思路是本质性的提升
- **启发**：这种"元策略"的思想可以推广到其他领域——在任何存在多种互补解决方案的场景中，学一个策略选择器可能比设计一个更好的单一策略更有效

## 评分
- 新颖性: ⭐⭐⭐⭐ 将幻觉诊断和动态策略选择结合的思路很新颖，但核心 CD 策略仍是现有方法
- 实验充分度: ⭐⭐⭐⭐⭐ 诊断实验充分有说服力，主实验覆盖生成和判别两种任务，消融全面
- 写作质量: ⭐⭐⭐⭐ "章鱼"的类比贯穿全文，结构清晰，诊断实验的呈现方式很直观
- 价值: ⭐⭐⭐⭐ 作为通用框架具有良好的扩展性和实用性，对幻觉缓解研究有重要指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Retrieval Visual Contrastive Decoding to Mitigate Object Hallucinations in Large Vision-Language Models](../../ACL2025/hallucination/retrieval_visual_contrastive_decoding_to_mitigate_object_hallucinations_in_large.md)
- [\[CVPR 2025\] Seeing Far and Clearly: Mitigating Hallucinations in MLLMs with Attention Causal Decoding](seeing_far_and_clearly_mitigating_hallucinations_in_mllms_with_attention_causal_.md)
- [\[ICLR 2026\] Dynamic Multimodal Activation Steering for Hallucination Mitigation in Large Vision-Language Models](../../ICLR2026/hallucination/dynamic_multimodal_activation_steering_for_hallucination_mitigation_in_large_vis.md)
- [\[ACL 2025\] Alleviating Hallucinations from Knowledge Misalignment in Large Language Models via Selective Abstention Learning](../../ACL2025/hallucination/alleviating_hallucinations_from_knowledge_misalignment_in_large_language_models_.md)
- [\[ACL 2025\] Mixture of Decoding: An Attention-Inspired Adaptive Decoding Strategy to Mitigate Hallucination in Multimodal LLMs](../../ACL2025/hallucination/mixture_of_decoding_an_attention-inspired_adaptive_decoding_strategy_to_mitigate.md)

</div>

<!-- RELATED:END -->
