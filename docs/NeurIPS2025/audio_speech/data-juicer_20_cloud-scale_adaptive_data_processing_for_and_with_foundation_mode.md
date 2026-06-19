---
title: >-
  [论文解读] Data-Juicer 2.0: Cloud-Scale Adaptive Data Processing for and with Foundation Models
description: >-
  [NeurIPS 2025 Spotlight][音频/语音][数据处理系统] Data-Juicer 2.0 是面向基础模型的云规模多模态数据处理系统，150+ 跨文本/图像/视频/音频算子，支持自适应分布式执行（Ray/MaxCompute），在 10000+ CPU 核心上高效处理 TB 级数据，已广泛应用于阿里云 PAI 等产品。
tags:
  - "NeurIPS 2025 Spotlight"
  - "音频/语音"
  - "数据处理系统"
  - "多模态数据"
  - "分布式计算"
  - "数据质量"
  - "基础模型"
---

# Data-Juicer 2.0: Cloud-Scale Adaptive Data Processing for and with Foundation Models

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2501.14755](https://arxiv.org/abs/2501.14755)  
**代码**: [GitHub](https://github.com/modelscope/data-juicer)  
**领域**: 音频语音  
**关键词**: 数据处理系统, 多模态数据, 分布式计算, 数据质量, 基础模型

## 一句话总结
Data-Juicer 2.0 是面向基础模型的云规模多模态数据处理系统，150+ 跨文本/图像/视频/音频算子，支持自适应分布式执行（Ray/MaxCompute），在 10000+ CPU 核心上高效处理 TB 级数据，已广泛应用于阿里云 PAI 等产品。

## 研究背景与动机

**领域现状**：基础模型训练需要海量多模态数据处理 pipeline。RedPajama、Dolma 主要面向文本，Spark 面向传统大数据。

**现有痛点**：(a) 多模态处理不足：缺乏跨模态对齐和语义变换；(b) 效率-扩展性矛盾：基础模型工作负载是简单但海量的逐样本操作；(c) 生态碎片化：API 不兼容阻碍跨平台优化。

**核心矛盾**：功能（多模态+语义）、规模（PB 级）和易用性（Python 原生）的同时挑战无统一框架满足。

**本文目标** 构建统一、可扩展的多模态数据处理系统覆盖清洗/标注/合成/后训练。

**切入角度**：从 Data-Juicer 1.0（50 文本算子）出发，通过分层架构系统扩展到多模态、大规模、多引擎。

**核心 idea**：分层自适应架构 + 150+ 多模态算子 + 多引擎统一抽象 = 覆盖基础模型全生命周期的云规模数据处理。

## 方法详解

### 整体框架
三层架构：(1) **能力层**——150+ 多模态算子（7 种类型）；(2) **接口层**——Python API / RESTful / Web UI / NL Agent；(3) **运行时层**——统一 Dataset 抽象、自适应执行、容错。

### 关键设计

1. **150+ 多模态算子体系**:

    - 功能：覆盖文本/图像/视频/音频的清洗、分析、合成、标注
    - 核心思路：原有 5 种原子算子 + 5 种组合型算子（Grouper/Aggregator/FusedOP/ScriptOP/HumanOP）。HumanOP 基于 Label Studio 支持 RLHF 人在回路
    - 设计动机：模型驱动算子（SDXL/GPT/Qwen）占多数，反映语义感知处理趋势

2. **统一 Data-Juicer-Dataset 抽象**:

    - 功能：屏蔽底层引擎差异（HF Dataset / Ray / MaxFrame）
    - 核心思路：Facade 模式，Token 对齐中间 Schema（`<__dj__image>` 等特殊 token 表示多模态数据）
    - 设计动机：同一 pipeline 可在单机/Ray/MaxCompute 间无缝切换

3. **自适应运行时优化**:

    - 功能：自动配置资源、批次、执行顺序
    - 核心思路：Adapter 类 probe_small_batch() 探测算子速度，快算子前置。GPU 算子自动配置量化，I/O 算子用多层并行。自适应分片对 Ray 提供 2-3x 加速
    - 设计动机：统一并行粒度导致 OOM 或浪费，自适应同时优化效率和稳定性

4. **容错与恢复**:

    - 功能：算子级检查点和细粒度恢复
    - 核心思路：LLM 输出预验证+自动重试，算子级断点续传替代 Ray 粗粒度全重启
    - 设计动机：大规模后期错误可致 TB 计算浪费

## 实验关键数据

### 不同规模性能

| 规模 | 样本量 | 推荐引擎 | 关键发现 |
|------|--------|---------|---------|
| 小 | 560K-2.24M | 单机 HF | 单机高效，Ray 4节点加速 138-226% |
| 中 | 5.6M-56M | Ray-DLC | Ray 优于单机，DLC 比 ECS 快 24.8% |
| 大 | 56M-70B | 多模态:Ray / 文本:MaxCompute | MaxCompute 文本仅需 Ray 1/4 时间 |

### 大规模去重性能

| 数据量 | CPU 核心 | 时间 |
|--------|---------|------|
| 200GB | 640 | 11.13 min |
| 1TB | 640 | 50.83 min |
| 5TB | 1280 | 168.10 min |

### 优化效果

| 优化项 | 效果 |
|--------|------|
| 自适应分片 | 2-3x 加速，网络 I/O 从 160MB/s 降至 60MB/s |
| 算子重排+融合 | 最多减少 70.22% |
| 自动 GPU 分配 | 节省最多 99% |
| 批量处理 | 最多减少 84% |

### 关键发现
- 存储-计算-软件协同设计在 >10M 样本后变关键：AI-CPFS（3x 带宽）加速 2.7x
- 70B 样本用 6400 核 Ray 仅需 7611 秒
- MaxCompute 纯文本远优于 Ray，得益于计算-存储协同

## 亮点与洞察
- **生产级规模**：12800 核、TB 数据、70B 样本的工程实践，不是玩具系统
- **模型即算子**：将大模型作为一等公民处理算子，"用 AI 处理 AI 数据"范式可迁移

## 局限与展望
- GPU 后端支持有限，未充分利用 NeMo Curator GPU 加速
- 多语言支持不够
- 缺乏处理后数据质量的系统评估

## 相关工作与启发
- **vs NeMo Curator**: GPU 加速效率更高（1.1TB 64 A100 仅 1.8h），但 Data-Juicer 2.0 更通用功能更丰富
- **vs 1.0**: 50 -> 150+ 算子，1000+ -> 10000+ 核，70M -> 70B 样本

## 评分
- 新颖性: ⭐⭐⭐ 系统集成和工程为主，单技术点创新有限
- 实验充分度: ⭐⭐⭐⭐ 覆盖多规模多引擎的全面评估
- 写作质量: ⭐⭐⭐ 内容丰富但略冗长
- 价值: ⭐⭐⭐⭐ 作为开源基础设施对数据处理生态有重要推动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Sensorium Arc: AI Agent System for Oceanic Data Exploration and Interactive Eco-Art](sensorium_arc_ai_agent_system_for_oceanic_data_exploration_and_interactive_eco-a.md)
- [\[ACL 2025\] SpeechWeave: Diverse Multilingual Synthetic Text & Audio Data Generation Pipeline for Training Text to Speech Models](../../ACL2025/audio_speech/speechweave_diverse_multilingual_synthetic_text_audio_data_generation_pipeline_f.md)
- [\[ICML 2026\] Algorithmic Recourse of In-Context Learning for Tabular Data](../../ICML2026/audio_speech/algorithmic_recourse_of_in-context_learning_for_tabular_data.md)
- [\[ACL 2025\] Distilling an End-to-End Voice Assistant Without Instruction Training Data](../../ACL2025/audio_speech/distilling_an_end-to-end_voice_assistant_without_instruction_training_data.md)
- [\[ICCV 2025\] VGGSounder: Audio-Visual Evaluations for Foundation Models](../../ICCV2025/audio_speech/vggsounder_audio-visual_evaluations_for_foundation_models.md)

</div>

<!-- RELATED:END -->
