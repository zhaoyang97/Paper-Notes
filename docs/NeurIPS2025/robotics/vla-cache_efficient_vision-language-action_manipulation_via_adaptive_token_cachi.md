---
title: >-
  [论文解读] VLA-Cache: Efficient Vision-Language-Action Manipulation via Adaptive Token Caching
description: >-
  [NeurIPS 2025][机器人][VLA加速] 提出VLA-Cache，一种免训练的VLA推理加速方法，通过跨帧识别并缓存静态视觉token的KV表示、过滤任务相关token并按层自适应调整复用比例，实现1.7倍加速且几乎不损失任务成功率。 VLA模型（如OpenVLA）将视觉和语言整合为端到端的动作生成…
tags:
  - "NeurIPS 2025"
  - "机器人"
  - "VLA加速"
  - "Token缓存"
  - "推理加速"
  - "免训练"
  - "机器人操作"
---

# VLA-Cache: Efficient Vision-Language-Action Manipulation via Adaptive Token Caching

**会议**: NeurIPS 2025  
**arXiv**: [2502.02175](https://arxiv.org/abs/2502.02175)  
**代码**: [项目页面](https://vla-cache.github.io)  
**领域**: 多模态VLM  
**关键词**: VLA加速, Token缓存, 推理加速, 免训练, 机器人操作

## 一句话总结

提出VLA-Cache，一种免训练的VLA推理加速方法，通过跨帧识别并缓存静态视觉token的KV表示、过滤任务相关token并按层自适应调整复用比例，实现1.7倍加速且几乎不损失任务成功率。

## 研究背景与动机

VLA模型（如OpenVLA）将视觉和语言整合为端到端的动作生成，但其庞大的计算开销成为实时机器人控制的主要瓶颈。具体问题如下：

**视觉token的时间冗余严重**：在闭环机器人操作中，相邻帧的观测图像大部分区域（背景、静止物体等）几乎不变化，但VLA模型在每个时间步都从头计算所有视觉token，造成大量冗余计算。

**现有加速方法不适配VLA特性**：模型轻量化、量化、early-exit等通用方法需要修改架构或重新训练，且缺乏针对VLA时序特性的设计。FastV、SparseVLM等VLM加速方法在单帧内剪枝/合并token，破坏了空间保真度，在操作精度要求高的VLA任务中效果不佳。

**朴素复用导致性能骤降**：不加区分地复用所有视觉上静态的token会严重损害性能（成功率从84.4%降至74.2%），因为某些视觉上变化不大但语义上至关重要的区域（如夹爪附近、目标物体）需要每步重新计算。

VLA-Cache的关键洞察是：利用VLA任务中的**时间连续性**，跨帧缓存复用静态token的KV表示，同时通过解码器注意力分数识别并保护**任务相关token**，再按层自适应调整复用比例以优化精度-效率平衡。

## 方法详解

### 整体框架

VLA-Cache包含两个核心步骤：(a) **动态Token选择**——识别跨帧静态token并过滤掉任务相关token；(b) **自适应Token缓存**——按层依据注意力分布动态调整复用比例。整个方法无需修改模型架构或重新训练，直接作为即插即用模块嵌入VLA推理流程。

### 关键设计

1. **静态Token选择**：将输入图像划分为 $N \times N$ 个patch，计算当前帧和上一帧对应patch的余弦相似度：$\text{Sim}(\mathbf{P}_t^{i,j}, \mathbf{P}_{t-1}^{i,j}) = \frac{\mathbf{P}_t^{i,j} \cdot \mathbf{P}_{t-1}^{i,j}}{\|\mathbf{P}_t^{i,j}\|_2 \cdot \|\mathbf{P}_{t-1}^{i,j}\|_2}$。相似度超过阈值 $\tau$（默认0.996）的patch被标记为静态，再通过Top-k（默认100）进一步筛选最稳定的token。这一步在像素空间操作，开销极低（$\mathcal{O}(H^2)$）。

2. **任务相关Token过滤**：从语言解码器中提取文本到视觉的交叉注意力矩阵 $\mathbf{A}_{\text{vis-text}}^l$，对多层多头取平均得到每个视觉token的任务相关性分数 $\mathbf{S}_{\text{task-relevance}}$。分数超过阈值 $\tau_{\text{task}}$（默认0.5）的token被标记为任务相关，从可复用集合中移除：$\mathcal{P}_{\text{reuse}} = \mathcal{P}_{\text{static}} \setminus \mathcal{P}_{\text{task-relevant}}$。这确保了夹爪、目标物体等关键区域始终使用最新特征。此机制将成功率从74.2%恢复到82.6%。

3. **层自适应Token复用**：不同解码器层的注意力分布差异显著——浅层注意力分散，深层注意力集中。作者提出基于注意力熵的自适应策略：计算相邻层的熵比率 $R^l = (\mathcal{E}^{l-1} - \mathcal{E}^l) / \mathcal{E}^{l-1}$，累积后确定各层的复用比例 $\alpha^l = \min(k \sum_{j=1}^l R^j, 1)$。累积熵下降越大的层（注意力越集中），允许复用更多token。这进一步将成功率提升至83.8%。

### 损失函数 / 训练策略

VLA-Cache是**完全免训练**的方法，不涉及任何损失函数设计或额外训练。它直接修改VLA解码器的前向推理过程：对被标记为可复用的token，直接从上一帧的KV缓存中读取对应的Key和Value向量；对需要重新计算的token，正常执行前向计算并更新KV缓存。由于Transformer的置换不变性，这种部分更新不影响注意力计算的正确性。

## 实验关键数据

### 主实验

**LIBERO基准 - OpenVLA**

| 方法 | Spatial | Object | Goal | Long | 均值 | FLOPs(T)↓ | 延迟(ms)↓ | 控制频率(Hz)↑ |
|------|---------|--------|------|------|------|-----------|-----------|--------------|
| OpenVLA | 84.4% | 86.6% | 75.6% | 53.2% | 75.0% | 1.864 | 51.91 | 4.23 |
| +SparseVLM | 79.8% | 67.0% | 72.6% | 39.4% | 64.7% | 1.407 | 83.39 | 3.72 |
| +FastV | 83.4% | 84.0% | 74.2% | 51.6% | 73.3% | 1.864 | 53.28 | 4.19 |
| **+VLA-Cache** | **83.8%** | **85.8%** | **76.4%** | **52.8%** | **74.7%** | **1.355** | **31.83** | **4.59** |

**OpenVLA-OFT（高频VLA架构）**

| 方法 | 均值成功率 | FLOPs(T) | 延迟(ms) | 控制频率(Hz) |
|------|-----------|----------|----------|-------------|
| OpenVLA-OFT | 96.8% | 4.013 | 79.05 | 65.10 |
| +VLA-Cache | **97.4%** | **3.097** | **62.59** | **78.98** |

**真实机器人（Kinova Jaco2）**

| 方法 | PickPot | PlaceCube | PutSausage | WipeTable | 均值 | 延迟(ms) |
|------|---------|-----------|------------|-----------|------|----------|
| OpenVLA | 95.0% | 83.3% | 80.0% | 70.0% | 82.1% | 64.16 |
| +VLA-Cache | 90.0% | **90.0%** | **85.0%** | **73.3%** | **84.6%** | **51.85** |

### 消融实验

| Token选择策略 | 成功率(%) | FLOPs↓ | 延迟(ms)↓ | 说明 |
|--------------|----------|--------|-----------|------|
| 基线OpenVLA | 84.4 | 1.888 | 52.37 | 无缓存 |
| +静态Token复用 | 74.2 | – | 31.03 | 朴素复用，性能骤降 |
| +过滤任务相关 | 82.6 | – | 31.03 | 注意力过滤恢复性能 |
| +层自适应 | **83.8** | – | 32.22 | 最终版本，精度最优 |

### 关键发现

- **SparseVLM和FastV在VLA上失效**：因为VLA的动作输出很短（仅7个token），VLM加速方法在长序列解码上的优势不适用，SparseVLM甚至导致延迟增加。
- **VLA-Cache与高频架构兼容**：在OpenVLA-OFT上进一步提升了约14Hz的控制频率（65→79Hz），证明VLA-Cache直接加速解码瓶颈。
- **动态背景下依然鲁棒**：引入背景运动干扰后，VLA-Cache维持成功率不变，同时FLOPs减少42%、延迟减少35%。

## 亮点与洞察

- **免训练即插即用**：无需重新训练或修改模型架构，直接对现有VLA的推理过程进行加速，部署门槛极低。
- **利用VLA特有的时间冗余**：区别于通用VLM加速，VLA-Cache专门利用了机器人操作中相邻帧高度相似这一独特性质。
- **注意力分数作为任务相关性代理**：巧妙地利用解码器自身的注意力模式来判断哪些token不可复用，无需额外的检测器或分割模型。

## 局限与展望

- 在环境高度动态（大量运动、背景剧变）的场景中，可复用token减少，加速效果会打折。
- 目前验证了LLaMA2为backbone的VLA架构（OpenVLA、CogAct、OpenVLA-OFT），对Gemma2等其他backbone的适用性未验证。
- 不直接适用于没有VLM backbone的纯扩散策略模型。
- 层自适应策略中的超参数 $k$ 需要根据模型结构调整。

## 相关工作与启发

- 与DeeR-VLA（动态深度控制）和TinyVLA（模型蒸馏）等需要重训练的方法形成互补。
- 与π0-FAST、HiRT等高频VLA架构兼容，VLA-Cache加速的是这些架构的解码瓶颈。
- 启发：未来可将跨帧token缓存策略扩展到多视角输入、3D体素表征等场景，或探索基于学习的动态阈值调整。

## 评分

- **新颖性**: ⭐⭐⭐⭐ 利用VLA时间冗余进行跨帧token缓存的思路简洁有效，注意力过滤机制设计精巧
- **实验充分度**: ⭐⭐⭐⭐⭐ 三个VLA模型×两个仿真平台+真实机器人，消融全面，含敏感性分析和动态背景测试
- **写作质量**: ⭐⭐⭐⭐⭐ 层层递进的方法呈现，每步都有表格验证效果，逻辑清晰
- **价值**: ⭐⭐⭐⭐ 解决了VLA部署中的实际痛点，免训练特性使其容易被社区采用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning](autovla_a_vision-language-action_model_for_end-to-end_autonomous_driving_with_ad.md)
- [\[AAAI 2026\] TTF-VLA: Temporal Token Fusion via Pixel-Attention Integration for Vision-Language-Action Models](../../AAAI2026/robotics/ttf-vla_temporal_token_fusion_via_pixel-attention_integratio.md)
- [\[CVPR 2025\] CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models](../../CVPR2025/robotics/cot-vla_visual_chain-of-thought_reasoning_for_vision-language-action_models.md)
- [\[CVPR 2026\] Adaptive Action Chunking at Inference-time for Vision-Language-Action Models](../../CVPR2026/robotics/adaptive_action_chunking_at_inference-time_for_vision-language-action_models.md)
- [\[ICCV 2025\] Moto: Latent Motion Token as the Bridging Language for Learning Robot Manipulation from Videos](../../ICCV2025/robotics/moto_latent_motion_token_as_the_bridging_language_for_learning_robot_manipulatio.md)

</div>

<!-- RELATED:END -->
