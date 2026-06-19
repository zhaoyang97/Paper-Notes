---
title: >-
  [论文解读] VideoLucy: Deep Memory Backtracking for Long Video Understanding
description: >-
  [NeurIPS 2025][视频理解][长视频理解] 提出VideoLucy框架，通过层次化记忆结构和基于Agent的迭代回溯机制，模拟人类从粗到细的回忆过程，在多个长视频理解基准上大幅超越现有方法，甚至超过GPT-4o等商业模型。 长视频理解要求系统对视频中几乎所有细节具有全面的记忆和把握，任何信息缺失都可能导致回答不准…
tags:
  - "NeurIPS 2025"
  - "视频理解"
  - "长视频理解"
  - "层次化记忆"
  - "Agent系统"
  - "记忆回溯"
  - "视频问答"
---

# VideoLucy: Deep Memory Backtracking for Long Video Understanding

**会议**: NeurIPS 2025  
**arXiv**: [2510.12422](https://arxiv.org/abs/2510.12422)  
**代码**: [https://videolucy.github.io](https://videolucy.github.io)  
**领域**: 视频理解  
**关键词**: 长视频理解, 层次化记忆, Agent系统, 记忆回溯, 视频问答

## 一句话总结

提出VideoLucy框架，通过层次化记忆结构和基于Agent的迭代回溯机制，模拟人类从粗到细的回忆过程，在多个长视频理解基准上大幅超越现有方法，甚至超过GPT-4o等商业模型。

## 研究背景与动机

长视频理解要求系统对视频中几乎所有细节具有全面的记忆和把握，任何信息缺失都可能导致回答不准确。现有Agent系统虽然绕过了传统视频MLLM的长输入限制，但面临两个关键挑战：

**挑战一：逐帧建模无法捕捉时间上下文。** 现有Agent系统通常对单帧生成文本描述，然后通过信息检索循环获取关键帧。但实际应用中，许多问题与连续帧的时间上下文密切相关，逐帧处理的时间理解能力较弱。

**挑战二：稀疏采样导致关键信息丢失。** 为降低密集逐帧描述的计算成本，现有系统采用稀疏帧采样（如VideoTree在Video-MME上仅用0.125 FPS）。即使对1小时视频（1 FPS）也需生成3600个描述，因此不得不妥协使用稀疏采样，这将导致大量关键细节信息的丢失。

VideoLucy的灵感来源于电影《超体》中主角Lucy获得了完美记忆能力——能回溯到生命中每一个细节。VideoLucy模拟人类从模糊到清晰的回忆过程，通过层次化记忆和迭代回溯实现对长视频的全面、深入理解。

## 方法详解

### 整体框架

VideoLucy包含三个核心组件：(1) 层次化记忆结构，实现从粗到细的多级视频表示和全面信息覆盖；(2) 赋予不同角色的Agent，负责描述、定位、指令和回答；(3) 迭代回溯机制，通过多阶段循环动态挖掘问题相关的深层记忆。

### 关键设计

1. **层次化记忆结构**: 定义三层记忆，时间感知范围递减、细节粒度递增：

    - **粗粒度长程记忆**：大时间范围的概括描述（如每60秒一段）
    - **细粒度短程记忆**：较短时间段的详细描述
    - **帧级超细记忆**：单帧或极短片段的精细描述

   每段视频的记忆通过 $m_k = VidCap(v_k, p_k)$ 获得，其中 $VidCap$ 为视频MLLM（如Qwen2.5-VL-7B），$p_k$ 为指令提示。当 $K=1$ 时退化为全局概览，$K=N$ 时表示逐帧描述。该结构同时实现了多级表示和全面信息覆盖。

2. **四类Agent**:

    - **Captioning Agent**：系统的"眼睛"，用MLLM将视频片段转换为文本描述
    - **Localization Agent**：用LLM（DeepSeek-R1）在当前记忆中定位与问题最相关的时间段
    - **Instruction Agent**：分析当前记忆中缺失的关键信息，生成引导性描述指令
    - **Answering Agent**：基于当前记忆判断是否能自信回答，若不能则输出不确信标志

3. **迭代回溯机制（Algorithm 1）**: 

    - **稀疏粗粒度初始化**：先生成全局粗粒度记忆，然后用Localization Agent筛选出与问题最相关的时间段
    - **深度和广度探索**：每次迭代中，Localization Agent定位最相关时间段 → Instruction Agent分析缺失信息并生成指令 → Captioning Agent重新生成当前深度记忆（更新）和更深层记忆（下探）→ 更新当前记忆列表
    - **Agent驱动循环**：迭代持续直到Answering Agent认为信息充足可以自信回答，或达到最大迭代次数（默认5次）

### 损失函数 / 训练策略

VideoLucy是一个无需额外训练的推理时Agent系统，直接利用现成的开源模型（Qwen2.5-VL-7B用于描述，DeepSeek-R1用于文本推理）。不同基准的时间范围参数 $T_c, T_f, T_{uf}$ 有不同设置。Agent通过精心设计的prompt实现角色分工。

## 实验关键数据

### 主实验

| 基准 | 指标 | VideoLucy | 之前SOTA | 提升 |
|------|------|-----------|---------|------|
| Video-MME (长视频) | Acc | **66.8** | 65.0 (AdaReTake-72B) | +1.8 |
| Video-MME (平均) | Acc | **72.5** | 71.9 (GPT-4o) | +0.6 |
| LVBench (整体) | Acc | **58.8** | 53.3 (AdaReTake-72B) | +5.5 |
| LVBench (KIR) | Acc | **75.6** | 62.2 (AdaReTake-72B) | +13.4 |
| MLVU | M-Avg | **76.1** | 74.7 (VideoChat-Flash-7B) | +1.4 |
| EgoMem (整体) | Acc | **56.7** | 46.4 (VideoChat-Flash-7B) | +10.3 |

注：VideoLucy使用7B开源模型，超越了使用72B模型和GPT-4o等商业模型的方法。

### 消融实验

| 记忆深度配置 | Video-MME长视频Acc | 说明 |
|-------------|-------------------|------|
| 仅视频摘要 | ~52 | 信息严重不足 |
| 粗粒度记忆 | ~58 | 基础改善 |
| +细粒度记忆 | ~63 | 显著提升 |
| +帧级超细记忆 | **~67** | 最优，每层记忆都有贡献 |

| 最大迭代次数 | Video-MME长视频Acc | 说明 |
|-------------|-------------------|------|
| 1次 | ~60 | 探索不足 |
| 3次 | ~64 | 改善明显 |
| 5次 | **~67** | 最优（默认值） |
| 7次 | ~66 | 过多迭代无额外增益 |

### 关键发现

- VideoLucy在Agent系统中的性能远超已有方法（Video-MME平均+8.5% vs MemVid），接近甚至超过商业闭源模型
- 在关键信息检索（KIR）任务上表现尤其突出（75.6%），说明迭代回溯机制对精确信息搜索极其有效
- "Needle-in-A-Video-Haystack"实验表明VideoLucy的细节感知能力几乎不受视频长度影响
- 记忆回溯过程中，信息丰富度和问题相关性持续提升，验证了分层探索的有效性
- EgoMem基准揭示现有MLLM在超长视频理解上严重不足（接近随机猜测水平）

## 亮点与洞察

- 层次化记忆结构的设计与人类认知过程高度吻合——从粗到细的回忆模式
- 无需额外训练，纯推理时框架，基于7B开源模型即可超越72B和商业模型
- 提出EgoMem基准填补了超长视频（平均6.33小时）理解评测的空白
- 迭代回溯机制提供了可解释的推理过程，增强了用户信任

## 局限与展望

- 迭代回溯的计算成本较高（每次迭代需要多次MLLM和LLM调用）
- 最大迭代次数是固定的硬限制，更智能的停止条件有待研究
- 对Captioning Agent（MLLM）的描述质量高度依赖，描述错误会在后续传播
- 文本形式的记忆存在信息损失，直接处理视觉特征可能更有效

## 相关工作与启发

- **vs DrVideo**: DrVideo也用Agent循环更新关键帧信息，但仅在帧级别操作；VideoLucy通过层次化记忆在多个时间尺度上建模
- **vs VideoTree**: VideoTree构建层次化视频表示但采用稀疏采样（0.125 FPS）；VideoLucy通过按需下探实现密集覆盖
- **vs 传统视频MLLM**: 传统方法受限于最大上下文长度和稀疏采样；VideoLucy通过文本化记忆和迭代检索突破了这些限制
- **vs LangRepo**: LangRepo維护结构化语言仓库但缺乏按需深度探索；VideoLucy的回溯机制更灵活高效

## 评分

- 新颖性: ⭐⭐⭐⭐ 层次化记忆+迭代回溯的组合设计有效且直观，但核心组件（Agent循环、层次结构）并非全新
- 实验充分度: ⭐⭐⭐⭐⭐ 四个基准全面评测+EgoMem新基准+Needle实验+丰富消融+定性对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机富于感染力（Lucy电影引喻），图示丰富
- 价值: ⭐⭐⭐⭐⭐ 在长视频理解领域树立了新的性能标杆，EgoMem基准对社区有重要贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] ReWind: Understanding Long Videos with Instructed Learnable Memory](../../CVPR2025/video_understanding/rewind_understanding_long_videos_with_instructed_learnable_memory.md)
- [\[ICCV 2025\] VideoLLaMB: Long Streaming Video Understanding with Recurrent Memory Bridges](../../ICCV2025/video_understanding/videollamb_long_streaming_video_understanding_with_recurrent_memory_bridges.md)
- [\[NeurIPS 2025\] Unleashing Hour-Scale Video Training for Long Video-Language Understanding](unleashing_hour-scale_video_training_for_long_video-language_understanding.md)
- [\[NeurIPS 2025\] InfiniPot-V: Memory-Constrained KV Cache Compression for Streaming Video Understanding](infinipot-v_memory-constrained_kv_cache_compression_for_streaming_video_understa.md)
- [\[CVPR 2025\] MLVU: Benchmarking Multi-task Long Video Understanding](../../CVPR2025/video_understanding/mlvu_benchmarking_multi-task_long_video_understanding.md)

</div>

<!-- RELATED:END -->
