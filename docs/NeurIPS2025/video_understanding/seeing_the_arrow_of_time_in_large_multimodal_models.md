---
title: >-
  [论文解读] Seeing the Arrow of Time in Large Multimodal Models
description: >-
  [NeurIPS 2025][视频理解][时间箭头] 本文揭示当前大多模态模型（LMMs）对视频时间方向性（时间箭头）出人意料地不敏感——正放/倒放时答案几乎相同，提出基于 GRPO 的 ArrowRL 训练策略引入反向视频奖励来激发时间方向感知，并构建 AoTBench 基准，在多个 VQA 基准上取得显著提升（Vinoground 上相对提升 65.9%）。
tags:
  - "NeurIPS 2025"
  - "视频理解"
  - "时间箭头"
  - "大模型时间感知"
  - "强化学习微调"
  - "视频理解基准"
  - "GRPO"
---

# Seeing the Arrow of Time in Large Multimodal Models

**会议**: NeurIPS 2025  
**arXiv**: [2506.03340](https://arxiv.org/abs/2506.03340)  
**代码**: [项目主页](https://vision.cs.utexas.edu/projects/SeeAoT)  
**领域**: 视频理解 / 多模态时间感知  
**关键词**: 时间箭头, 大模型时间感知, 强化学习微调, 视频理解基准, GRPO

## 一句话总结

本文揭示当前大多模态模型（LMMs）对视频时间方向性（时间箭头）出人意料地不敏感——正放/倒放时答案几乎相同，提出基于 GRPO 的 ArrowRL 训练策略引入反向视频奖励来激发时间方向感知，并构建 AoTBench 基准，在多个 VQA 基准上取得显著提升（Vinoground 上相对提升 65.9%）。

## 研究背景与动机

时间箭头（Arrow of Time, AoT）是物理世界的基本属性——奶油溶入咖啡、玻璃碎裂等事件逆转会立即显得不自然。人类天然具有时间方向感知能力，但当前的大模态模型却缺乏这种能力。

核心发现令人震惊：将视频帧打乱或倒放后，SOTA LMMs（如 LLaVA-OV-7B）在多个标准 VQA 基准上的性能几乎没有下降——说明这些模型根本没有真正利用视频的时间顺序信息。更具体地，当展示正放和倒放视频时，模型常给出完全相同的描述（如对正放和倒放都说"ignite"），暴露出根本性的时间不敏感。

本文的切入角度是双管齐下：(1) 在模型端，通过强化学习让模型对正放和倒放视频产生不同的响应；(2) 在评估端，构建真正考验时间方向感知的基准。核心 idea：利用反转视频作为天然的对比信号，通过奖励机制强制模型区分正反时序。

## 方法详解

### 整体框架

ArrowRL 是基于 GRPO（Group Relative Policy Optimization）的后训练策略。给定一个预训练 LMM $\pi_\theta$，对每个问题 $q=(v, l)$ 生成一组候选响应 $\{o_i\}_{i=1}^G$，同时用反转视频 $\tilde{v}$ 生成反向响应 $\tilde{o}$。双重奖励信号（保真度奖励 + 反向奖励）指导模型优化，然后通过 GRPO 目标函数更新策略。

### 关键设计

1. **时间散度评分（TDS）**:

    - 为量化样本和基准的时间敏感性，提出基于 KL 散度的评分方法
    - 对每个样本，比较模型对正放和倒放视频的首 token 概率分布：$\text{TDS}_i = D_{KL}[p_i \| \tilde{p}_i]$
    - 比简单的准确率差异更细粒度，能捕捉模型置信度的变化
    - 用于系统分析 8 个主流 VQA 基准的时间敏感性，发现 TVBench、Vinoground、TempCompass 时间敏感度高，而 VITATECS、TemporalBench 等低

2. **保真度奖励（Target Fidelity Reward）**:

    - 衡量候选响应 $o_i$ 与目标响应 $o^*$ 的相似度
    - MCQ 任务：精确匹配（1.0 或 0.0）
    - 开放式 QA 和视频描述：使用 LLM 评分
    - 确保模型输出与正确答案对齐

3. **反向奖励（Reverse Reward）**:

    - 最大化正向候选响应 $o_i$ 与反向响应 $\tilde{o}$ 之间的差异：$r_i^{rev} = 1 - \text{Similarity}(o_i, \tilde{o})$
    - 动机：AoT 敏感的模型应对正放和倒放视频产生不同的响应
    - 动态加权机制：当 $\text{Similarity}(\tilde{o}, o^*) > \gamma$ 时（说明该样本时间不敏感），设 $\alpha_i = 0$ 禁用反向奖励
    - 最终奖励：$r_i = r_i^{fid} + \alpha_i \cdot r_i^{rev}$

4. **AoTBench 基准**:

    - 三个任务：(a) 序列方向分类（613 个视频判断正放/倒放）；(b) 方向性描述匹配（2000 个视频的 V2T 和 T2V 任务）；(c) AoT 敏感 VQA（从 8 个基准中挑选高 TDS 的 1800 个样本）
    - 专门针对时间方向感知能力的评估

### 损失函数 / 训练策略

采用标准 GRPO 目标函数，组大小 $G=8$，反向奖励权重 $\alpha=0.25$，动态阈值 $\gamma=0.75$。训练数据包括：1.1K MCQ（UCF101 正反分类）、11.8K 高时间性开放式 QA（从 LLaVA-Video-178K 按困惑度差异筛选）、11.7K 视频描述（RTime 数据集）。仅需 2000 步 RL 训练，6 张 GH200 GPU。无需 SFT 阶段直接应用于预训练模型。

## 实验关键数据

### 主实验

**AoTBench 结果**：

| 模型 | 方向分类RFilm | 方向分类UCF | 描述匹配T2V | 描述匹配V2T | AoT-VQA |
|------|-------------|------------|------------|------------|---------|
| GPT-4o | 52.8 | 54.0 | 56.5 | 69.5 | 67.8 |
| Qwen2.5-VL-7B | 50.0 | 51.6 | 53.4 | 66.6 | 49.6 |
| + ArrowRL | **51.4** | **54.8** | **55.6** | **69.6** | **58.8** |
| Qwen2-VL-7B | 50.0 | 51.6 | 56.3 | 62.3 | 44.3 |
| + ArrowRL | **69.1** | **72.6** | **57.1** | **68.8** | **51.1** |

**现有时间敏感基准**：

| 模型 | TempCompass | TVBench | Vinoground Group |
|------|------------|---------|-----------------|
| Qwen2.5-VL-7B | 73.8 | 54.7 | 16.4 |
| + ArrowRL | **75.5** | **56.2** | **27.2** (+65.9%相对提升) |

### 消融实验

| 配置 | AoTBench平均准确率 | 说明 |
|------|------------------|------|
| Qwen2.5-VL-7B 基线 | 56.2% | 未训练 |
| + SFT（同数据） | 57.4% | 监督微调效果有限 |
| + ArrowRL（LLaVA字幕） | 57.7% | 非高时间性数据 |
| + ArrowRL（RTime字幕） | 60.4% | 高时间性数据 |
| + ArrowRL（完整数据） | **61.4%** | 多任务组合最佳 |

### 关键发现
- 几乎所有开源 LMMs 在方向分类任务上表现为纯随机（50%），对正放/倒放视频给出完全相同的回答
- ArrowRL 远优于 SFT（+4.0 vs +1.2），RL 策略在激发时间方向感知上明显更有效
- 反向奖励是核心组件：移除后 ($\alpha=0$) 性能降至基线以下
- ArrowRL 不损害通用视频理解能力：在 VideoMME、NExT-QA 等非时间敏感基准上性能持平或小幅提升

## 亮点与洞察

- **揭示了一个根本性问题**：SOTA LMMs 对视频时间方向几乎完全不敏感，这是之前被忽视的重要缺陷
- **反向奖励设计巧妙**：利用反转视频作为天然信号，不需要额外标注数据，通过推动正向/反向响应分离来获得时间感知
- **TDS 评分的方法论贡献**：为评估视频基准的时间敏感性提供了系统性工具，发现许多号称测试时间理解的基准实际上对时间顺序不敏感
- **极高效训练**：仅 2000 步 RL 训练就能显著提升时间感知，可直接应用于预训练模型无需 SFT

## 局限与展望

- 训练限制最大视频帧数为 16 帧，对长视频的时间感知提升有限
- 反向奖励的前提假设是正放和倒放视频应有语义差异，对静态或循环视频不适用（虽有动态加权缓解）
- 方向分类任务中部分基线仍接近随机，说明仅靠 ArrowRL 还不足以完全解决该问题
- 未探索与 Chain-of-Thought 推理的结合

## 相关工作与启发

- **vs Video-R1**: Video-R1 专注于视频推理，但在 AoT 任务上同样表现为随机水平（方向分类 50%），而 ArrowRL 能有效提升
- **vs 早期 AoT 自监督**（Pickup 2014, Wei 2018）: 早期工作仅在视觉特征学习中使用 AoT 作为 pretext task，本文首次在 LMM 的语言生成层面解决时间方向感知
- **对 RL 微调的启发**: 反向奖励的思路可推广——利用输入的特定变换（如翻转、裁剪、速度变化）作为对比信号来强化模型对特定属性的感知

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 揭示了 LMM 时间感知的根本缺陷，反向奖励设计极具原创性
- 实验充分度: ⭐⭐⭐⭐⭐ 8 个基准的系统性分析、3 个基座模型验证、充分的消融和超参分析
- 写作质量: ⭐⭐⭐⭐⭐ 论证逻辑严密，问题发现→分析→方案→验证的脉络非常清晰
- 价值: ⭐⭐⭐⭐⭐ 指出了领域的根本性缺陷并提供了有效解决方案，对整个视频 LMM 领域有深远影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] DisTime: Distribution-based Time Representation for Video Large Language Models](../../ICCV2025/video_understanding/distime_distribution-based_time_representation_for_video_large_language_models.md)
- [\[CVPR 2025\] DivPrune: Diversity-Based Visual Token Pruning for Large Multimodal Models](../../CVPR2025/video_understanding/divprune_diversity-based_visual_token_pruning_for_large_multimodal_models.md)
- [\[NeurIPS 2025\] SAMA: Towards Multi-Turn Referential Grounded Video Chat with Large Language Models](sama_towards_multi-turn_referential_grounded_video_chat_with_large_language_mode.md)
- [\[NeurIPS 2025\] MoniTor: Exploiting Large Language Models with Instruction for Online Video Anomaly Detection](monitor_exploiting_large_language_models_with_instruction_for_online_video_anoma.md)
- [\[NeurIPS 2025\] Self-alignment of Large Video Language Models with Refined Regularized Preference Optimization](self-alignment_of_large_video_language_models_with_refined_regularized_preferenc.md)

</div>

<!-- RELATED:END -->
