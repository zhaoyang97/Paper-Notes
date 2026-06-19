---
title: >-
  [论文解读] TSPO: Temporal Sampling Policy Optimization for Long-form Video Language Understanding
description: >-
  [AAAI 2026][视频理解][长视频理解] 将视频关键帧选择和语言生成建模为联合决策过程，通过基于GRPO的强化学习端到端优化轻量级时序智能体的采样策略，在四个长视频理解基准上取得SOTA（LLaVA-Video-7B上LongVideoBench +5.0%、MLVU +6.0%），且可零样本迁移到其他Video-MLLM。
tags:
  - "AAAI 2026"
  - "视频理解"
  - "长视频理解"
  - "关键帧采样"
  - "强化学习"
  - "时序策略优化"
  - "视频多模态大模型"
---

# TSPO: Temporal Sampling Policy Optimization for Long-form Video Language Understanding

**会议**: AAAI 2026  
**arXiv**: [2508.04369](https://arxiv.org/abs/2508.04369)  
**代码**: [https://github.com/Hui-design/TSPO](https://github.com/Hui-design/TSPO)  
**领域**: 视频理解 / 多模态大模型  
**关键词**: 长视频理解, 关键帧采样, 强化学习, 时序策略优化, 视频多模态大模型

## 一句话总结

将视频关键帧选择和语言生成建模为联合决策过程，通过基于GRPO的强化学习端到端优化轻量级时序智能体的采样策略，在四个长视频理解基准上取得SOTA（LLaVA-Video-7B上LongVideoBench +5.0%、MLVU +6.0%），且可零样本迁移到其他Video-MLLM。

## 研究背景与动机

**领域现状**：Video-MLLM（如LLaVA-Video、Qwen2.5-VL）在视频理解任务中取得了显著进展，但处理长视频时受限于LLM的上下文长度和训练成本，必须先进行稀疏帧采样。当前主流做法是均匀采样（uniform sampling），或使用DINOv2、LLaVA-1.5等预训练模型做无训练的关键帧搜索。

**现有痛点**：（1）均匀采样经常遗漏与查询相关的关键信息；（2）无训练的关键帧搜索方法（如LongVU用DINOv2-1B、CoS用LLaVA-1.5-13B）受限于预训练特征提取器的跨模态理解能力，且推理开销大（CoS需要13B级别模型）；（3）这些方法都无法通过训练进一步优化。

**核心矛盾**：构建可训练的帧采样方法面临两个根本挑战——（1）无监督性：通用视频理解训练数据缺乏帧级标注，没有精确的定位监督；（2）不可微性：帧采样是离散子集选择问题，输出是帧索引而非连续变量，无法通过反向传播在SFT中优化。

**本文目标** 如何设计一个可训练的稀疏帧采样方法，使其能端到端优化，不需要帧级标注，且在长视频理解中显著提升性能？

**切入角度**：受DeepSeek-R1用GRPO增强推理能力的启发，作者将关键帧选择和语言生成建模为联合决策过程——帧采样的离散选择可以类比为RL中的动作，MLLM的回答准确性直接作为奖励信号反馈给采样策略。

**核心 idea**：用一个轻量级事件感知时序智能体做概率性关键帧选择，通过GRPO将其与Video-MLLM的语言生成联合优化，利用回答准确性作为奖励信号端到端训练采样策略。

## 方法详解

### 整体框架

TSPO框架由三部分组成：（1）事件感知时序智能体（Temporal Agent）：基于CLIP特征，通过局部窗口注意力捕获事件-查询相关性，输出每帧的选择概率；（2）TSPO强化学习优化：将帧选择和语言生成建模为联合策略，冻结Video-MLLM只优化时序智能体；（3）双风格训练数据+双奖励机制。输入是长视频+文本查询，时序智能体从候选帧中选出关键帧，送入冻结的Video-MLLM生成回答。

### 关键设计

1. **事件感知时序智能体（Event-aware Temporal Agent）**:

    - 功能：从候选帧中以概率方式选择与查询最相关的关键帧
    - 核心思路：以CLIP帧级视觉特征 $\mathbf{F}_f \in \mathbb{R}^{T_c \times D}$ 和文本特征 $\mathbf{F}_t \in \mathbb{R}^{1 \times D}$ 为输入。通过局部窗口注意力（窗口大小 $w=12$）加正弦位置编码学习帧内事件依赖关系，得到事件表征 $\mathbf{F}_e$。最终相似度融合事件级和帧级两种相似度 $S = \text{Sim}_{event}(\mathbf{F}_e, \mathbf{F}_t) + \text{Sim}_{frame}(\mathbf{F}_f, \mathbf{F}_t)$。使用Gumbel-Softmax采样进行探索：$\mathcal{P}, \mathcal{I} = \text{TopK}(\text{Softmax}(S/\tau + \gamma))$，其中 $\gamma \sim \text{Gumbel}(0,1)$
    - 设计动机：CLIP仅有帧级特征，缺乏事件级（跨帧）理解。局部窗口注意力让相邻帧能互相感知，形成事件级的时序理解。Gumbel-Softmax既保证可微性又提供RL所需的探索性。温度退火（$\tau$ 从0.025退火到0.01）使训练从探索逐步转向利用。整个智能体仅有3.5M可学习参数，极其轻量

2. **时序采样策略优化（TSPO Algorithm）**:

    - 功能：端到端优化时序智能体的采样策略，无需帧级标注
    - 核心思路：将联合策略分解为 $\pi(\mathbf{o}, \mathbf{V_s} | \mathbf{q}, \mathbf{V}_c) = \pi_l(\mathbf{o} | \mathbf{q}, \mathbf{V}_s, \mathbf{V}_c) \cdot \pi_{ts}(\mathbf{V}_s | \mathbf{q}, \mathbf{V}_c)$。冻结Video-MLLM使得语言生成策略比值为1，TSPO目标简化为只优化时序智能体：$\mathcal{J}^*_{tspo} = \frac{1}{G}\sum_{i=1}^{G}\frac{\pi_{ts}(\mathbf{V}_s | \mathbf{q}, \mathbf{V}_c)}{\pi_{ts_{old}}(\mathbf{V}_s | \mathbf{q}, \mathbf{V}_c)} A_i$。每个问题采样 $G$ 组不同的关键帧组合，各自生成回答并获得奖励，计算组内相对优势 $A_i$ 来更新策略
    - 设计动机：直接用SFT训练帧选择器（实验中的SFT*方案）效果不如TSPO，因为SFT梯度来自语言生成的交叉熵损失，对帧选择的监督是间接的。GRPO框架让采样策略直接receive奖励信号，且通过多组采样来估计优势函数，实现更有效的策略优化

3. **双风格训练数据构建管道**:

    - 功能：为TSPO训练提供高质量的长视频训练数据
    - 核心思路：（1）综合时序数据：从LLaVA-Video-178K中筛选1分钟以上的多选QA，过滤掉4帧就能答对（太简单）和64帧仍答不对（太难）的样本；（2）视频"大海捞针"数据：将目标视频与无关视频拼接打乱形成10~60分钟长视频，用Qwen2.5-VL对目标视频生成事件描述并转为多选题，模拟长程时序定位场景。两种数据合计构成TSPO-10K
    - 设计动机：综合时序数据提供一般视频理解能力，大海捞针数据训练长程时序定位能力。两者互补——消融实验显示单用大海捞针提升LongVideoBench但降低VideoMME，组合使用效果最佳

### 损失函数 / 训练策略

双奖励机制：回答准确性奖励 $R_A = \mathbf{1}(y = \bar{y})$（选项是否正确）和时序定位奖励 $R_T = T_t / T_a$（采样帧中落在目标视频内的比例）。综合时序数据的总奖励为 $R_A + 1$，大海捞针数据为 $R_A + R_T$。训练使用8×A800 GPU，学习率5e-4，batch size为1，训练1个epoch。训练时选16帧，推理时选64帧。

## 实验关键数据

### 主实验

| 模型 | LongVideoBench | MLVU | VideoMME(Long) | LVBench |
|------|--------------|------|-------------|---------|
| LLaVA-Video-7B (uniform) | 58.9 | 70.3 | 53.6 | 40.2 |
| LLaVA-Video-7B + TSPO | **63.9** (+5.0) | **76.3** (+6.0) | **54.7** (+1.1) | **45.3** (+5.1) |
| Qwen2.5VL (uniform) | 59.0 | 65.1 | 53.3 | 38.3 |
| Qwen2.5VL + TSPO | **64.2** (+5.2) | **74.3** (+9.2) | **56.4** (+3.1) | **46.4** (+8.1) |
| LLaVA-Video + CoS (13B selector) | 58.9 | 71.4 | 53.8 | - |
| LLaVA-Video + AKS (0.5B selector) | 62.7 | - | 54.0 | - |

### 消融实验

| 配置 | LongVideoBench | VideoMME | 说明 |
|------|--------------|---------|------|
| 基线 (uniform) | 58.9 | 64.4 | 无训练 |
| 仅综合时序数据 + $R_A$ | 62.8 | 65.5 | 一般理解提升显著 |
| 仅大海捞针 + $R_T$ | 63.4 | 64.6 | 定位提升但理解略降 |
| 双数据 + 双奖励 | **63.9** | **65.5** | 最优组合 |
| SFT* (30K数据) | 62.8 | 64.8 | TSPO(10K)仍更优 |

### 关键发现

- **跨架构零样本迁移**：在LLaVA-Video上训练的时序智能体，直接迁移到Qwen2VL/Qwen2.5VL上平均提升4.5%/6.3%，甚至迁移到LLaVA-Video-72B也有3.6%提升
- **推理效率优势**：TSPO的关键帧提取仅需1.2秒（CoS需28.4秒），节省90%时间。用32帧即可接近64帧uniform的性能，Token数量和LLM推理时间减半
- **TSPO优于SFT**：即使SFT*用了3倍数据量（30K vs 10K），TSPO仍优于SFT*，验证了RL探索+奖励反馈比监督学习更适合离散采样优化
- **VideoMME提升较小**：因为VideoMME侧重整体视频理解而非关键帧定位，这恰好说明TSPO的优势在于长程时序定位

## 亮点与洞察

- **将帧采样问题转化为RL问题**是本文最核心的贡献。帧采样天然是离散、无监督的，而GRPO框架完美匹配了这个问题——不需要帧级标签，用回答准确性作为奖励信号，通过多组采样估计优势。这个建模思路可以推广到任何需要从大量候选中做离散选择的场景
- **冻结MLLM只优化采样器**的策略非常聪明：（1）避免了大模型的训练开销；（2）已经SFT过的MLLM在选对帧时本就能回答正确，所以只需优化"选帧"；（3）训练好的采样器可以即插即用到不同MLLM上
- **3.5M参数的超轻量时序智能体**对比CoS的13B参数MLLM选择器，参数量小3700倍，推理时间快23倍，性能还更好

## 局限与展望

- 候选帧数固定为128（1FPS），对于数小时级别的超长视频可能不够，需要更高效的多级采样策略
- Video-MLLM始终冻结，未探索采样器和MLLM联合优化的潜力（可能因为计算成本太高）
- 训练数据仅10K样本、最长60分钟视频，对更长视频（如电影级别）的泛化能力未验证
- 奖励设计相对简单（准确性+时序定位），可以引入更细粒度的奖励（如视觉信息量、帧多样性）

## 相关工作与启发

- **vs LongVU (shen2024longvu)**：LongVU用DINOv2-1B做跨帧去重，属于无训练方法，参数量是TSPO时序智能体的285倍，但MLVU上被TSPO超越10.9%
- **vs CoS (hu2025cos)**：CoS用LLaVA-1.5-13B做逐帧打分，推理代价极高（28.4秒 vs TSPO的1.2秒），且无法训练优化。TSPO在LongVideoBench上超越5.0%
- **vs TPO**：TPO用DPO优化LLM的时序推理能力，但不优化帧选择。TSPO直接在帧选择层面优化，与TPO互补

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将帧采样建模为RL决策过程并用GRPO端到端优化，解决了无监督+不可微的核心难题
- 实验充分度: ⭐⭐⭐⭐⭐ 4个基准、跨3个MLLM迁移验证、完整消融（数据/奖励/SFT对比/效率分析）
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，但GRPO推导可以更简洁
- 价值: ⭐⭐⭐⭐⭐ 高度实用——轻量采样器即插即用，对长视频理解有直接工程价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] T*: Re-thinking Temporal Search for Long-Form Video Understanding](../../CVPR2025/video_understanding/re-thinking_temporal_search_for_long-form_video_understanding.md)
- [\[NeurIPS 2025\] LongVPO: From Anchored Cues to Self-Reasoning for Long-Form Video Preference Optimization](../../NeurIPS2025/video_understanding/longvpo_from_anchored_cues_to_selfreasoning_for_longform_vid.md)
- [\[ICCV 2025\] VideoMiner: Iteratively Grounding Key Frames of Hour-Long Videos via Tree-based Group Relative Policy Optimization](../../ICCV2025/video_understanding/videominer_iteratively_grounding_key_frames_of_hour-long_videos_via_tree-based_g.md)
- [\[ICCV 2025\] HERMES: temporal-coHERent long-forM understanding with Episodes and Semantics](../../ICCV2025/video_understanding/hermes_temporal-coherent_long-form_understanding_with_episodes_and_semantics.md)
- [\[CVPR 2026\] DIvide, then Ground: Adapting Frame Selection to Query Types for Long-Form Video Understanding](../../CVPR2026/video_understanding/divide_then_ground_adapting_frame_selection_to_query_types_for_long-form_video_u.md)

</div>

<!-- RELATED:END -->
