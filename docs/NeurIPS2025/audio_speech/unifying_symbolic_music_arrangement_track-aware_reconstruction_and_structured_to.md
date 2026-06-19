---
title: >-
  [论文解读] Unifying Symbolic Music Arrangement: Track-Aware Reconstruction and Structured Tokenization
description: >-
  [NeurIPS 2025][音频/语音][音乐编排] 提出一个统一的符号音乐编排框架，通过段级自监督重建目标（解耦内容和乐器风格）和新的多轨token化方案REMI-z，使单个预训练模型能够处理乐队编排、钢琴缩编和鼓编排等多种编排任务，并在三个典型任务上超越了任务特定的SOTA。 音乐编排(arrangement)是将作品…
tags:
  - "NeurIPS 2025"
  - "音频/语音"
  - "音乐编排"
  - "符号音乐"
  - "多轨音乐生成"
  - "音乐token化"
  - "自监督学习"
---

# Unifying Symbolic Music Arrangement: Track-Aware Reconstruction and Structured Tokenization

**会议**: NeurIPS 2025  
**arXiv**: [2408.15176](https://arxiv.org/abs/2408.15176)  
**代码**: [https://www.oulongshen.xyz/automatic_arrangement](https://www.oulongshen.xyz/automatic_arrangement)  
**领域**: 音频与语音 / 符号音乐生成  
**关键词**: 音乐编排, 符号音乐, 多轨音乐生成, 音乐token化, 自监督学习

## 一句话总结
提出一个统一的符号音乐编排框架，通过段级自监督重建目标（解耦内容和乐器风格）和新的多轨token化方案REMI-z，使单个预训练模型能够处理乐队编排、钢琴缩编和鼓编排等多种编排任务，并在三个典型任务上超越了任务特定的SOTA。

## 研究背景与动机

音乐编排(arrangement)是将作品改编为不同演奏形式的艺术，包括：
- **重新诠释(reinterpretation)**：为不同乐器重新编写（如管弦乐→爵士乐队）
- **简化(simplification)**：简化为独奏（如乐队→钢琴）
- **增补生成(additive generation)**：添加新音轨（如为歌曲添加鼓）

现有研究针对每种任务独立设计模型，存在三个核心问题：

**缺乏跨任务通用性**：每种编排任务使用专用架构和训练方案，无法共享音乐知识

**数据瓶颈**：端到端方法依赖平行数据集（同一作品的不同编排版本），这类数据极度稀缺

**token化的局限**：现有REMI+等严格时序化方案将不同乐器的音符交错排列，导致内容碎片化，难以实现乐器级别的控制

核心切入点：所有编排任务共享同一结构——基于已有轨道，在内容和乐器约束下生成新的音乐轨道。可以通过统一的自监督重建目标和结构化token化来实现。

## 方法详解

### 整体框架

采用80M参数的Decoder-only Transformer。先在大规模MIDI数据集(Los Angeles MIDI, 405K文件)上进行标准next-token预训练，然后通过统一的重建目标在Slakh2100数据集上微调，实现多种编排任务。输入序列格式为 `[condition]<SEP>[target]`，仅在target子序列上计算交叉熵损失。

### 关键设计

1. **段级解耦重建目标**：

    - 输入音乐被分解为三个token流：
        - **乐器条件 I(·)**：指定目标片段使用的乐器，按音高寄存器排序（高音在前），实现乐器控制和声部控制
        - **内容条件 C(·)**：从源音乐中提取的纯音符序列（去除乐器标记），按时间排序后去重，编码"弹什么"而非"谁弹、怎么弹"
        - **历史条件**：上一段的完整token序列，提供跨段一致性
    - 微调目标：$\mathcal{L}(\theta) = -\log p_\theta(\mathcal{T}_{task}(y^{(t)}) | I(\mathcal{T}_{task}(y^{(t)})), C(\mathcal{S}_{task}(y^{(t)})), \mathcal{T}_{task}(y^{(t-1)}))$
    - 设计动机：训练时三个条件均从同一音乐中提取（自监督），无需平行数据；推理时可自由组合任意乐器和内容

2. **REMI-z token化方案**：

    - 核心改变：放弃全局严格时序，优先保证轨道内连续性
    - 结构：每个小节包含多个轨道序列，每个轨道对应一个乐器；轨道内按onset排序，轨道间按平均音高从高到低排列
    - "锯齿形"编码模式：同一乐器的音符在token序列中保持连续，不会被其他乐器的音符打断
    - 优势：
        - 序列长度减少32.9%（151.68 vs 225.91 tokens/小节），降低计算成本
        - 清晰的轨道边界，便于乐器级别控制
        - 同一乐器的内容编码不受并发乐器影响，增加训练数据中的模式重复性
    - 小节级Shannon熵降低（29.43 vs 41.68 bits/token），信息冗余减少

3. **任务实例化**：

    - **乐队编排**：$\mathcal{S}_{task} = \mathcal{T}_{task} = identity$（去除鼓轨）；训练时随机删除部分非旋律轨道、去除时值token，倒逼模型创造性重写
    - **钢琴缩编**：$\mathcal{T}_{task}$选择钢琴轨，$\mathcal{S}_{task} = identity$；过滤仅保留钢琴轨覆盖>40%音高范围的片段
    - **鼓编排**：$\mathcal{S}_{task}$提取所有有音高乐器轨，$\mathcal{T}_{task}$提取鼓轨；使用4小节段以捕获跨小节鼓模式

### 损失函数 / 训练策略
- 预训练：标准next-token prediction，4卡RTX A5000，batch 12，1 epoch
- 微调：段级重建目标，单卡A40，3 epoch，AdamW + 线性warmup
- 关键正规化：key normalization（将所有歌曲转调到C大调/A小调），随机轨道删除（Poisson采样删除数量）

## 实验关键数据

### 主实验

**乐队编排(客观)**

| 模型 | I-IoU↑ | VER↓ | Note F1↑ | Notei F1↑ | Mel F1↑ |
|------|------|------|----------|------|------|
| Transformer-VAE | 97.5 | 35.0 | 49.5 | 40.0 | 24.7 |
| REMI+ Transformer | 95.0 | 18.2 | 94.4 | 76.0 | 68.8 |
| REMI-z(无预训练) | 99.5 | 9.9 | 97.8 | 77.5 | 77.8 |
| **Ours(+预训练)** | **99.8** | **7.6** | **97.5** | **87.0** | **84.5** |

**鼓编排(主观, 5分制)**

| 模型 | 兼容性 | 连贯性 | 过渡 | 创意 | 音乐性 |
|------|------|------|----------|------|------|
| CA v2 | 3.82 | 4.05 | 2.86 | 2.58 | 3.19 |
| **Ours** | **3.91** | 4.03 | **3.77** | **3.27** | **3.57** |
| Ground Truth | 4.31 | 4.18 | 3.36 | 3.16 | 3.78 |

### 消融实验

| 配置 | Notei F1 | Mel F1 | 说明 |
|------|---------|------|------|
| 完整模型 | 87.0 | 84.5 | — |
| 去除voice控制 | 84.3(-2.7) | 81.5(-3.0) | 声部信息有助推断乐器角色 |
| 去除history条件 | 77.4(-9.6) | 79.4(-5.1) | 历史上下文对跨段一致性关键 |
| 无预训练(w/o PT) | 77.5 | 77.8 | 预训练带来+9.5 Notei F1提升 |

**Token化对比(无条件生成)**

| Token化方案 | tokens/小节 | tokens/音符 | 音符PPL↓ |
|------|---------|------|------|
| REMI+ | 225.91 | 4.03 | 116.20 |
| REMI-z | 151.68 | 2.77 | **84.11** |

### 关键发现
- REMI-z相对REMI+在所有客观指标上统计显著(p<0.001)，验证了结构化token化的优势
- 预训练知识迁移主要提升了乐器级别建模(Notei F1 +9.5%)和旋律保持(Mel F1 +6.7%)
- 在鼓编排中，模型接近Ground Truth的音乐性评分(3.57 vs 3.78)
- REMI-z不仅在编排任务中更好，在无条件生成中也有更低的音符级困惑度

## 亮点与洞察
- 统一的自监督重建目标是核心贡献——通过内容-风格解耦，让单一模型处理多种编排场景，完全无需平行数据
- REMI-z token化方案解决了多轨音乐建模中长期存在的内容碎片化问题，且优势延伸至通用符号音乐建模
- 从预训练大模型到微调的范式在符号音乐领域取得了与NLP类似的成功
- 80M参数的小模型就能超越任务特定SOTA，说明方法设计比模型规模更重要

## 局限与展望
- 模型规模仅80M，扩大规模可能带来显著提升
- 预训练数据量(405K MIDI)相对有限，更大数据集可能更有效
- 未处理速度(velocity)信息，限制了演奏表现力的建模
- 内容条件中去除了时值(duration)信息，可能丢失重要的乐器特定节奏模式
- 钢琴缩编使用原始钢琴轨道而非人工编排的目标，可能不完全反映真实编排需求

## 相关工作与启发
- Q-Transformer(Zhao et al., 2023)用高级描述符引导编排，但将风格建模与内容实现分离
- Composer's Assistant 2用infilling方式做鼓编排，但不保留原曲核心
- REMI/REMI+系列token化方案是本文REMI-z的直接改进基础
- 类BART的span infilling去噪思路在随机删除轨道中有所体现
- 预训练→微调范式的成功印证了符号音乐领域的"基础模型"路线

## 评分
- 新颖性: ⭐⭐⭐⭐ 统一编排框架和REMI-z token化都是新颖的贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 三个task，主客观评估都有，消融和token化分析详尽
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，任务定义精确，图示直观
- 价值: ⭐⭐⭐⭐ 为符号音乐编排提供了实用且通用的解决方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Segment-Factorized Full-Song Generation on Symbolic Piano Music](segment-factorized_full-song_generation_on_symbolic_piano_music.md)
- [\[ICLR 2026\] SyncTrack: Rhythmic Stability and Synchronization in Multi-Track Music Generation](../../ICLR2026/audio_speech/synctrack_rhythmic_stability_and_synchronization_in_multi-track_music_generation.md)
- [\[ACL 2026\] Jamendo-MT-QA: A Benchmark for Multi-Track Comparative Music Question Answering](../../ACL2026/audio_speech/jamendo-mt-qa_a_benchmark_for_multi-track_comparative_music_question_answering.md)
- [\[ACL 2026\] Anchored Cyclic Generation: A Novel Paradigm for Long-Sequence Symbolic Music Generation](../../ACL2026/audio_speech/anchored_cyclic_generation_a_novel_paradigm_for_long-sequence_symbolic_music_gen.md)
- [\[NeurIPS 2025\] MEGADance: Mixture-of-Experts Architecture for Genre-Aware 3D Dance Generation](megadance_mixture-of-experts_architecture_for_genre-aware_3d_dance_generation.md)

</div>

<!-- RELATED:END -->
