---
title: >-
  [论文解读] SAMA: Towards Multi-Turn Referential Grounded Video Chat with Large Language Models
description: >-
  [NeurIPS 2025][视频理解][视频grounding] 提出 SAMA 框架，通过构建统一的数据集（SAMA-239K）、模型（时空上下文聚合器 + SAM）和基准（SAMA-Bench），首次实现了多轮引用式视频对话中细粒度时空理解与grounding的联合建模。 当前 Video Large Multimod…
tags:
  - "NeurIPS 2025"
  - "视频理解"
  - "视频grounding"
  - "多轮对话"
  - "时空理解"
  - "SAM"
  - "视频LMM"
---

# SAMA: Towards Multi-Turn Referential Grounded Video Chat with Large Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2505.18812](https://arxiv.org/abs/2505.18812)  
**作者**: Ye Sun, Hao Zhang, Henghui Ding, Tiehua Zhang, Xingjun Ma, Yu-Gang Jiang
**代码**: 无  
**领域**: 视频理解 / 视频对话  
**关键词**: 视频grounding, 多轮对话, 时空理解, SAM, 视频LMM

## 一句话总结

提出 SAMA 框架，通过构建统一的数据集（SAMA-239K）、模型（时空上下文聚合器 + SAM）和基准（SAMA-Bench），首次实现了多轮引用式视频对话中细粒度时空理解与grounding的联合建模。

## 研究背景与动机

当前 Video Large Multimodal Models (Video LMMs) 在细粒度视频时空理解方面仍然面临巨大挑战。实现这一目标需要同时掌握两项核心能力：

**视频引用理解 (Video Referring Understanding)**：捕获视频区域的语义信息

**视频grounding**：根据自然语言描述分割目标区域

然而，现有方法大多将这两个任务独立处理，存在以下关键瓶颈：

- **缺乏高质量的统一视频指令数据**：现有数据集要么只关注引用理解，要么只关注grounding，缺少支持联合学习的大规模数据集
- **缺乏综合评测基准**：没有统一的benchmark来评估引用式视频对话中的多轮时空理解能力
- **模型设计局限**：现有模型难以同时处理视频级别的时空理解和精确的区域级grounding

## 方法详解

### 整体框架

SAMA 从三个核心维度——数据集、模型、基准——全面解决上述问题：

1. **SAMA-239K 数据集**：包含 15K 个精心策划的视频，共 239K 条指令数据，支持视频引用理解、grounding 和多轮视频对话的联合学习
2. **SAMA 模型**：集成通用时空上下文聚合器和 Segment Anything Model (SAM)
3. **SAMA-Bench**：包含 522 个视频中的 5,067 个问题

### 关键设计

#### 时空上下文聚合器 (Spatio-Temporal Context Aggregator)

- 设计了一种通用的时空上下文聚合模块，能够在不同时间帧和空间区域之间进行信息交互
- 支持将用户通过点击/框选指定的视频区域编码为上下文表示
- 实现了跨帧的时序关联，使模型能够追踪对象在时间维度上的变化

#### SAM 集成

- 将 Segment Anything Model 集成到视频LMM管道中
- SAM 负责生成精确的区域分割mask
- 模型能够在理解对话语义的同时输出精确的空间定位

#### SAMA-239K 数据集构建

- 从 15K 个多样化视频中收集数据
- 涵盖多种任务类型：视频引用理解、空间grounding、时间grounding、多轮对话
- 精心设计的数据采样策略，确保任务类型的均衡分布

### 损失函数 / 训练策略

- 采用多任务联合训练策略
- 同时优化视频理解loss、grounding loss 和对话生成loss
- 使用分阶段训练：先预训练基础能力，再进行指令微调

## 实验关键数据

### 主实验

| 模型 | SAMA-Bench (Overall) | Video Referring | Video Grounding | Multi-Turn Chat |
|------|---------------------|----------------|-----------------|-----------------|
| Video-ChatGPT | 32.1 | 28.5 | 18.3 | 41.2 |
| VideoChat2 | 38.7 | 35.2 | 22.1 | 46.8 |
| LLaVA-Video | 42.3 | 40.1 | 25.7 | 49.6 |
| VISA | 45.1 | 43.8 | 31.2 | 51.4 |
| **SAMA** | **56.8** | **54.2** | **48.6** | **58.3** |

> SAMA 在 SAMA-Bench 上全面超越现有方法，尤其在 Video Grounding 上提升显著（+17.4pp vs VISA）。

| 方法 | MeViS val J&F | Ref-YouTube-VOS J&F | Ref-DAVIS J&F |
|------|-------------|---------------------|---------------|
| UNINEXT | 56.8 | 64.3 | 65.2 |
| OnlineRefer | 55.6 | 63.5 | 64.1 |
| TrackGPT | 58.3 | 65.8 | 66.7 |
| **SAMA** | **62.1** | **68.5** | **69.3** |

> 在通用 grounding benchmark 上，SAMA 同样取得新的 SOTA。

### 消融实验

| 配置 | SAMA-Bench | Grounding | Referring |
|------|-----------|-----------|-----------|
| 无 SAM | 48.2 | 35.1 | 50.3 |
| 无时空聚合器 | 50.6 | 40.8 | 48.7 |
| 无 SAMA-239K（仅公开数据） | 47.5 | 36.2 | 46.5 |
| **完整 SAMA** | **56.8** | **48.6** | **54.2** |

### 关键发现

1. **SAM 的引入**对 grounding 性能提升最为关键（+13.5pp）
2. **SAMA-239K 数据集**比仅使用公开数据提升了 9.3 个百分点
3. 时空上下文聚合器在引用理解任务上贡献最大
4. SAMA 在标准视觉理解 benchmark 上保持了高度竞争性的性能，说明 grounding 能力不会牺牲通用理解性能

## 亮点与洞察

1. **系统性贡献**：同时在数据集、模型和benchmark三个维度进行贡献，构建了完整的研究闭环
2. **统一框架**：首次将视频引用理解、grounding和多轮对话统一到一个模型中
3. **高质量数据集**：SAMA-239K 的构建方法论具有借鉴意义——从 15K 视频中生成 239K 条多样化指令
4. **SAM 集成范式**：展示了如何将视觉基础模型（SAM）有效集成到视频LMM中

## 局限与展望

1. **计算开销**：集成 SAM 增加了推理时的计算成本
2. **长视频支持**：当前实验主要集中在中等长度的视频片段
3. **实时交互**：多轮对话中的实时响应能力有待提升
4. **开放域泛化**：在wild场景下的泛化性能需要进一步验证

## 相关工作与启发

- **VideoChat / Video-ChatGPT** 系列：早期视频对话模型，但缺乏 grounding 能力
- **SAM / SAM 2**：视觉分割基础模型，为区域级理解提供了强力支撑
- **UNINEXT / TrackGPT**：视频 grounding 专用模型
- **启发**：统一数据集 + 模型 + benchmark 的三位一体方法论，可以推广到其他多模态任务

## 评分

- **创新性**: ⭐⭐⭐⭐ — 首次完整解决多轮引用式视频对话
- **技术贡献**: ⭐⭐⭐⭐ — 数据集+模型+benchmark系统贡献
- **实验充分度**: ⭐⭐⭐⭐ — 多个benchmark验证，消融充分
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，动机明确
- **影响力**: ⭐⭐⭐⭐ — 为视频理解社区提供了重要资源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Self-alignment of Large Video Language Models with Refined Regularized Preference Optimization](self-alignment_of_large_video_language_models_with_refined_regularized_preferenc.md)
- [\[NeurIPS 2025\] MoniTor: Exploiting Large Language Models with Instruction for Online Video Anomaly Detection](monitor_exploiting_large_language_models_with_instruction_for_online_video_anoma.md)
- [\[NeurIPS 2025\] FastVID: Dynamic Density Pruning for Fast Video Large Language Models](fastvid_dynamic_density_pruning_for_fast_video_large_languag.md)
- [\[ICCV 2025\] Factorized Learning for Temporally Grounded Video-Language Models](../../ICCV2025/video_understanding/factorized_learning_for_temporally_grounded_video-language_models.md)
- [\[CVPR 2025\] Video Summarization with Large Language Models](../../CVPR2025/video_understanding/video_summarization_with_large_language_models.md)

</div>

<!-- RELATED:END -->
