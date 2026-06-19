---
title: >-
  [论文解读] Video-T1: Test-Time Scaling for Video Generation
description: >-
  [ICCV 2025][LLM推理][测试时缩放] 将LLM中的测试时缩放(TTS)思想迁移到视频生成领域，将TTS重新定义为从高斯噪声空间到目标视频分布的搜索问题，提出Tree-of-Frames (ToF)搜索算法实现高效的推理时计算扩展，在VBench上持续稳定提升各类视频生成模型的质量。 领域现状 视频生成在训练时扩…
tags:
  - "ICCV 2025"
  - "LLM推理"
  - "测试时缩放"
  - "视频生成"
  - "Tree-of-Frames"
  - "扩散模型"
  - "自回归视频"
---

# Video-T1: Test-Time Scaling for Video Generation

**会议**: ICCV 2025  
**arXiv**: [2503.18942](https://arxiv.org/abs/2503.18942)  
**代码**: [liuff19.github.io/Video-T1](https://liuff19.github.io/Video-T1)  
**领域**: LLM推理  
**关键词**: 测试时缩放, 视频生成, Tree-of-Frames, 扩散模型, 自回归视频

## 一句话总结

将LLM中的测试时缩放(TTS)思想迁移到视频生成领域，将TTS重新定义为从高斯噪声空间到目标视频分布的搜索问题，提出Tree-of-Frames (ToF)搜索算法实现高效的推理时计算扩展，在VBench上持续稳定提升各类视频生成模型的质量。

## 研究背景与动机

### 领域现状

视频生成在训练时扩展（更多数据、更大模型、更多计算）方面取得了显著进步，但训练开销巨大，进一步扩展面临瓶颈。与此同时，LLM领域（如DeepSeek-R1、OpenAI o1）已证明测试时缩放(TTS)可以通过增加推理计算显著提升模型性能。

### 现有痛点

**训练扩展成本极高**：视频生成模型的训练需要海量数据和计算资源，扩展代价远超LLM

**推理时可能性未被探索**：视频生成领域几乎没有系统性研究如何通过增加推理计算来提升质量

**视频生成的特殊挑战**：不同于LLM的token序列，视频需要同时保证空间质量和时间一致性；扩散模型的多步去噪过程使计算扩展更复杂

### 核心研究问题

如果允许视频生成模型使用更多的推理时计算，能多大程度地提升生成质量？

### 切入角度

将视频生成的TTS重新定义为搜索问题：在高斯噪声空间到目标视频分布的轨迹中搜索更好的路径。通过测试时验证器提供反馈、启发式算法引导搜索过程。

## 方法详解

### 整体框架

Video-T1框架包含三个核心组件：
- **视频生成器** $\mathcal{G}$：从文本条件生成视频的预训练模型
- **测试验证器** $\mathcal{V}$：评估生成视频质量的多模态评估模型
- **启发式搜索算法** $f$：利用验证器反馈引导搜索的优化方法

提出两种搜索策略：随机线性搜索和Tree-of-Frames (ToF)搜索。

### 关键设计

#### 1. **随机线性搜索 (Random Linear Search)**

- **功能**：随机采样N个高斯噪声，分别执行完整去噪生成N个视频，选择验证器评分最高的一个（Best-of-N策略）
- **核心思路**：可视为N棵退化树（每棵树只有一条路径）组成的森林，从中选择最优路径
- **时间复杂度**：$O(TN)$，T为帧数，N为噪声样本数
- **局限**：线性结构简单，缺乏高效优化机制；各树独立，无交互反馈，增加了冗余计算

#### 2. **Tree-of-Frames (ToF) 搜索**

- **功能**：利用自回归模型的逐帧生成特性，在时间维度上引入推理时推理，通过树结构自适应扩展和剪枝视频分支
- **核心思路**：分三阶段生成——
    - **(a) 图像级对齐**：首帧与文本在核心语义上对齐（颜色、对象计数、位置），影响后续帧
    - **(b) 层次化提示**：中间帧关注运动稳定性、物理合理性，动态调整验证器的评估重点
    - **(c) 整体评估**：最终评估整体视频质量和文本对齐
- **三大技术**：
    - **图像级对齐**：在去噪过程中逐步评估帧质量，早期拒绝低质量候选
    - **层次化提示**：不同阶段的验证器使用不同prompt，首帧关注语义一致、中间帧关注运动连续、末尾关注整体质量
    - **启发式剪枝**：每个时间步保留top-$k_t$个节点，动态分支因子$b_t$，平衡探索与收敛
- **复杂度**：实际操作中$b_t$在大多数时间步为1，仅在阶段切换时分支，复杂度简化为$O(N+T)$

#### 3. **多验证器集成**

- **功能**：融合多个验证器的评分来选择最终视频
- **核心公式**：$\hat{i} = \arg\max_{0<i<n} \frac{1}{|\mathcal{M}|} \sum_{v \in \mathcal{M}} c_v \text{Rank}_v(f^{(i)})$
- **设计动机**：不同验证器关注不同评估维度，融合可以减少偏差、选择综合最优的视频

### 验证器选择

使用三个多模态奖励模型：VisionReward（29个维度的人类偏好）、VideoScore（基于LMM训练的多维评分）、VideoLLaMA3（SOTA多模态理解模型）。以VBench作为ground-truth验证器来测量上界。

## 实验关键数据

### 主实验（VBench总分提升）

| 模型 | 无TTS | +TTS | 提升% | 语义分提升% |
|------|-------|------|-------|------------|
| CogVideoX-5B | 81.61 | **84.42** | +3.44% | +10.1% |
| CogVideoX-2B | 80.91 | 83.89 | +3.68% | +3.38% |
| OpenSora-v1.2 | 79.76 | 81.65 | +2.37% | +9.87% |
| Pyramid-Flow (FLUX) | 81.61 | **86.51** | +5.86% | **+18.6%** |
| Pyramid-Flow (SD3) | 81.72 | 85.31 | +4.39% | +13.8% |
| NOVA | 78.56 | 79.80 | +1.58% | +2.43% |

**维度级提升亮点**（Pyramid-Flow FLUX）：
- Multiple Objects: 61.08 → 88.93 (+45.6%)
- Scene: 47.65 → 56.07 (+17.7%)
- Object Class: 93.49 → 99.69 (+6.6%)

### 消融实验

| 配置 | 说明 | 效果 |
|------|------|------|
| ToF vs 线性搜索 | 相同计算预算下比较 | ToF达到同等性能，GFLOPs减少约70% |
| 单验证器 vs 多验证器 | VisionReward/VideoScore/VideoLLaMA3 | 多验证器集成进一步提升TTS曲线 |
| 小模型(NOVA 0.6B) vs 大模型(CogVideoX-5B) | 参数量对TTS效果的影响 | 大模型从TTS获益显著更多 |
| 不同生成维度 | Scene/Object/Motion等 | 常见语义维度提升大，Motion Smoothness等提升有限 |

**GFLOPs对比**（N=7时）：

| 模型 | 线性搜索 | ToF搜索 | 比率 |
|------|---------|--------|------|
| Pyramid-Flow(FLUX) | 5.22×10⁷ | 1.62×10⁷ | 31% |
| Pyramid-Flow(SD3) | 3.66×10⁷ | 1.13×10⁷ | 31% |
| NOVA | 4.02×10⁶ | 1.41×10⁶ | 35% |

### 关键发现

- **TTS在所有视频生成模型上持续稳定提升质量**，最终收敛到一个上限
- **更大更好的基础模型从TTS获益更多**：CogVideoX-5B比NOVA获得更大的提升空间和更高效率
- **ToF搜索远比线性搜索高效**：在计算量仅约1/3的情况下达到相当甚至更好的性能
- **语义对齐维度提升最显著**（Object Class +19.5%，Scene +18.6%），而运动平滑等隐式属性提升有限
- **多验证器集成能进一步推高TTS天花板**

## 亮点与洞察

1. **范式迁移**：首次系统性将TTS从LLM迁移到视频生成，建立了通用的搜索框架
2. **ToF的设计巧妙**：利用自回归模型的逐帧特性，在时间维度上构建树搜索，将复杂度从$O(TN)$降到$O(N+T)$
3. **层次化提示策略**：不同阶段关注不同质量维度，模拟人类对视频质量的多层次判断
4. **验证器是关键瓶颈**：框架的上限取决于验证器的质量，强调了视频评估模型的重要性
5. **隐含结论**：推理时计算可能比增加训练更划算——在生成质量提升上提供了训练扩展之外的新路径

## 局限与展望

1. **仅适用于自回归模型的ToF**：全帧扩散模型（如CogVideoX）只能使用线性搜索，无法利用ToF的效率优势
2. **验证器能力制约上限**：当前验证器对运动平滑、时间闪烁等维度评估能力不足，导致这些维度提升有限
3. **计算开销仍然可观**：即使ToF效率更高，大规模部署时的推理成本仍远超单次生成
4. **未探索更复杂的搜索策略**：如MCTS、beam search等LLM中常用的策略
5. **仅在VBench上评估**：缺少人类偏好评估的验证

## 相关工作与启发

- **OpenAI o1 / DeepSeek-R1** 证明TTS在LLM中的巨大潜力，本文将该思想迁移到视觉生成
- **图像生成TTS** 的先期工作为本文提供了直接启发，但视频的时间维度带来新挑战
- **NOVA、Pyramid-Flow** 等自回归视频模型天然适配ToF的树搜索结构
- 验证器的质量是TTS效果的关键——改进视频评估模型（如VideoLLaMA3）会直接带动TTS效果

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首次将TTS系统性引入视频生成，ToF搜索设计新颖且高效
- **实验充分度**: ⭐⭐⭐⭐ — 6个生成模型、3个验证器、多维度分析，但缺少人类评估
- **写作质量**: ⭐⭐⭐⭐ — 概念清晰，框架完整，算法伪代码规范
- **价值**: ⭐⭐⭐⭐⭐ — 为视频生成提供了训练扩展之外的全新优化路径，影响深远

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] TTS-VAR: A Test-Time Scaling Framework for Visual Auto-Regressive Generation](../../NeurIPS2025/llm_reasoning/tts-var_a_test-time_scaling_framework_for_visual_auto-regressive_generation.md)
- [\[CVPR 2025\] Enhancing Video-LLM Reasoning via Agent-of-Thoughts Distillation](../../CVPR2025/llm_reasoning/enhancing_video-llm_reasoning_via_agent-of-thoughts_distillation.md)
- [\[NeurIPS 2025\] VideoRFT: Incentivizing Video Reasoning Capability in MLLMs via Reinforced Fine-Tuning](../../NeurIPS2025/llm_reasoning/videorft_incentivizing_video_reasoning_capability_in_mllms_via_reinforced_fine-t.md)
- [\[NeurIPS 2025\] Rethinking Optimal Verification Granularity for Compute-Efficient Test-Time Scaling](../../NeurIPS2025/llm_reasoning/rethinking_optimal_verification_granularity_for_compute-efficient_test-time_scal.md)
- [\[ACL 2025\] Revisiting the Test-Time Scaling of o1-like Models: Do they Truly Possess Test-Time Scaling Capabilities?](../../ACL2025/llm_reasoning/revisiting_the_test-time_scaling_of_o1-like_models_do_they_truly_possess_test-ti.md)

</div>

<!-- RELATED:END -->
