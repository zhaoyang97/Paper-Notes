---
title: >-
  [论文解读] DanmakuTPPBench: A Multi-modal Benchmark for Temporal Point Process Modeling and Understanding
description: >-
  [NeurIPS 2025][多模态VLM][temporal point process] 构建首个多模态时间点过程基准DanmakuTPPBench：DanmakuTPP-Events提供7250个序列共1080万弹幕事件（时间-文本-视频三模态天然对齐），DanmakuTPP-QA通过多智能体pipeline自动生成10类推理问答，系统暴露了经典TPP模型和MLLM在多模态事件动态理解上的显著短板。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "temporal point process"
  - "多模态"
  - "Danmaku"
  - "LLM evaluation"
  - "multi-agent pipeline"
---

# DanmakuTPPBench: A Multi-modal Benchmark for Temporal Point Process Modeling and Understanding

**会议**: NeurIPS 2025  
**arXiv**: [2505.18411](https://arxiv.org/abs/2505.18411)  
**代码**: [GitHub](https://github.com/FRENKIE-CHIANG/DanmakuTPPBench)  
**领域**: 多模态时序建模 / TPP / LLM Benchmark  
**关键词**: temporal point process, multimodal benchmark, Danmaku, LLM evaluation, multi-agent pipeline

## 一句话总结

构建首个多模态时间点过程基准DanmakuTPPBench：DanmakuTPP-Events提供7250个序列共1080万弹幕事件（时间-文本-视频三模态天然对齐），DanmakuTPP-QA通过多智能体pipeline自动生成10类推理问答，系统暴露了经典TPP模型和MLLM在多模态事件动态理解上的显著短板。

## 研究背景与动机

**领域现状**：时间点过程（TPP）是建模连续时间事件序列的经典框架，广泛应用于社交媒体预测、医疗监测和金融分析。近年LLM/MLLM在多模态推理上取得巨大进展，将TPP建模引入多模态语言模型是值得探索的方向。

**现有痛点**：现有TPP数据集（Retweet、StackOverflow、Taobao等）仅包含时间戳和事件类别，完全缺乏文本语义和视觉上下文。RNCNIX和Amazon Review虽有文本但缺少视觉信息。没有任何TPP数据集同时包含时间、文本和视觉三个模态，也没有面向TPP理解的QA评估基准。

**核心矛盾**：多模态能力强（如MLLM在VQA上表现出色）不等于多模态时间事件建模能力强——当前没有基准来评估模型能否联合理解时间动态、文本语义和视觉内容的交互关系。

**本文切入**：B站弹幕系统是理想的多模态TPP数据源——每条弹幕天然携带精确时间戳（与视频帧对齐）、文本内容（用户即时反应）和对应视频帧，形成完美的三模态对齐。基于此构建DanmakuTPP-Events用于经典TPP建模评估，再通过多Agent pipeline自动构建DanmakuTPP-QA用于深层时序推理评估。

## 方法详解

### 整体框架

DanmakuTPPBench包含两个互补组件：(1) DanmakuTPP-Events——面向经典TPP建模的多模态事件数据集（7250序列、1080万事件、14个视频分类）；(2) DanmakuTPP-QA——面向深层时序推理的多任务问答基准（2605个视频、10类任务），由5个专业化Agent协作自动构建。

### 关键设计

1. **DanmakuTPP-Events数据集构建**:
    - 功能：从B站2024年Top 100创作者的所有视频中收集弹幕数据，构建首个三模态TPP数据集
    - 核心思路：每个弹幕事件建模为四元组 $(t_i, e_i, m_i^{\text{text}}, m_i^{\text{image}})$——时间戳、事件类型（9类：吐槽、弹幕梗、情感表达等）、弹幕文本和对应视频帧。7250个序列、平均长度1494（远超现有TPP数据集的27-197），覆盖14个视频分类（游戏23%、教育18%、生活12%、动画10%等）
    - 设计动机：弹幕是极少数天然同时具备时间-文本-视觉三要素的数据源，无需人工对齐。数据量（1080万事件）和序列长度（平均1494）远超所有现有TPP数据集，为研究长序列多模态事件建模提供了理想的测试平台

2. **DanmakuTPP-QA多智能体构建Pipeline**:
    - 功能：自动构建10类评估任务（8闭合题+2开放题），涵盖时间预测、情感分析、事件归因等
    - 核心思路：5个专业化Agent分工协作——**Task-Design Agent**（Deepseek-R1）从数据结构出发设计10类任务及其input/output格式；**Annotation Agent**（Qwen2.5文本+Qwen2.5-VL视觉+RAM物体标注）为事件打标签；**Quality-Control Agent**（Qwen3）通过多数投票和gap-filling协调并验证标注一致性；**Visualization Agent**（Qwen2.5-Coder）自动生成Python可视化脚本；**Task-Solve Agent**（Qwen3+Qwen2.5-VL+Gemma-3多模型投票）生成参考答案
    - 设计动机：人工构建大规模多模态QA成本极高，多Agent pipeline在规模和质量间取得平衡。交叉验证+多数投票+人工抽检保证数据质量。测试集答案经过严格人工验证

3. **10类评估任务体系**:
    - 功能：设计从简单预测到复杂因果推理的梯度式评估任务
    - 核心思路：**闭合题**（8类）——弹幕爆发计数(ACC)、下一弹幕/爆发时间预测(RMSE)、平均情感极性评估(RMSE)、情感极性预测(RMSE)、事件类型推断(ACC)、爆发触发类型预测(ACC)。**开放题**（2类）——全局情感动态分析(LLM-Eval打分0-1)、爆发因果归因分析(LLM-Eval)。数据划分：训练集2005、验证集300、测试集300
    - 设计动机：任务难度从数值预测到逻辑推理再到因果归因层层递进，全面评估模型的时间感知、文本理解和跨模态推理能力

### 损失函数 / 训练策略

作为benchmark论文，不提出新训练方法。经典TPP模型使用EasyTPP框架默认设置训练。MLLM评估采用zero-shot推理，随机采样3个视频帧作为视觉输入。Fine-tuning实验用LoRA在Qwen2.5-VL-3B上进行（单GPU RTX 4090，3 epoch，lr=1e-4）。

## 实验关键数据

### 主实验

| 模型 | T-1(ACC↑) | T-2(RMSE↓) | T-4(RMSE↓) | T-7(ACC↑) | T-8(ACC↑) |
|------|-----------|-----------|-----------|-----------|-----------|
| Qwen2.5-7B | 0.33 | 27.64 | 0.65 | 10.67 | 32.67 |
| Qwen2.5-72B | 0.67 | 1.28 | 0.30 | 16.00 | 43.83 |
| Qwen3-30B-A3B | 0.67 | 1.33 | **0.20** | **23.00** | 43.67 |
| DeepSeek-V3 | **25.00** | 1.30 | 0.34 | 13.67 | 34.50 |
| Qwen2.5-VL-72B | 0.33 | **1.14** | 0.28 | 15.98 | **47.17** |
| Gemma3-27B | 0.33 | 1.33 | 0.28 | 15.67 | 36.17 |
| Fine-tuned VL-3B | **27.0** | 1.35 | **0.05** | 15.33 | 43.00 |

### 消融实验

| 配置 | T-4 RMSE↓ | T-5 RMSE↓ | T-6 RMSE↓ | 说明 |
|------|-----------|-----------|-----------|------|
| 最佳预训练模型 | 0.20 | 0.26 | 0.20 | Qwen3-30B/DeepSeek-V3/Gemma3 |
| Fine-tuned VL-3B | **0.05** | **0.16** | **0.08** | 误差降低4-6倍，小模型微调大幅超越 |
| Fine-tuned VL-3B@T-3 | 220.43 | — | — | 但时间预测任务过拟合严重 |

经典TPP模型：NHP在log-likelihood（0.799）和RMSE（0.932）上均最优，但注意力模型（THP 0.619、AttNHP 0.550）log-likelihood较低。

### 关键发现

- MLLM在多模态TPP理解上并不一致优于纯文本LLM：Qwen2.5-VL-72B仅在部分任务上最优，Llama-3.3-70B在时间预测（T-2 RMSE=1.11）上更好
- 模型规模效应显著：Qwen2.5系列从7B到72B，T-2 RMSE从27.64降至1.28
- 小模型微调在情感任务上大幅超越大模型零样本（T-4 RMSE: 0.05 vs 0.20），但在时间预测上可能过拟合（T-3严重退化）
- 开放题上Qwen2.5-VL-72B和Qwen3-235B领先，任务9（情感动态分析）得分0.48、任务10（因果归因）得分0.52——远未饱和
- 弹幕爆发因果归因（Task-10）最具挑战性，需同时理解视频内容变化、用户情感演变和事件触发机制

## 亮点与洞察

- 数据源选择精妙：弹幕是极少数天然三模态对齐的数据源，每条弹幕自带精确时间戳、文本和对应视频帧，无需任何人工对齐操作。这种数据设计的insight具有示范意义。
- 多Agent pipeline设计成熟：5个专业化Agent各司其职+交叉验证+多数投票，可推广到其他领域的benchmark自动构建。特别是Task-Design Agent用推理模型（R1）设计任务、Task-Solve Agent用多模型投票生成答案的范式很有参考价值。
- 揭示了一个重要gap："多模态理解能力强"≠"多模态时间事件建模能力强"——当前MLLM虽在VQA/图像理解上强大，但面对长序列时间动态分析仍有根本局限。

## 局限与展望

- 数据域集中在B站弹幕生态，文化背景和用户行为有平台偏见，泛化到其他语言/文化场景需验证
- 自动QA生成依赖LLM能力上限，任务定义和参考答案可能存在噪声
- 仅提供benchmark未提出新的强建模方法——缺少在此数据上设计的多模态TPP模型
- 事件类型仅9类，不同视频分类的弹幕行为差异可能需要更细粒度的类型划分
- MLLM仅采样3帧作为视觉输入，更密集的帧采样或视频编码器可能改善效果

## 相关工作与启发

- **vs 传统TPP数据集**（Retweet/Taobao/StackOverflow）：均为单模态，DanmakuTPP-Events是首个完整三模态TPP数据集，且平均序列长度1494远超它们（27-197）
- **vs TSQA**（Kong et al.）：TSQA面向时间序列QA但不涉及点过程和多模态，DanmakuTPP-QA专注TPP理解且带视觉信息
- **vs Amazon Review**：有文本但无视觉，且平均序列长度仅27，DanmakuTPP-Events在模态完整性和序列规模上全面超越

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个多模态TPP benchmark，数据源选择独特且自然
- 实验充分度: ⭐⭐⭐⭐ 覆盖经典TPP模型和多种LLM/MLLM，任务设计丰富
- 写作质量: ⭐⭐⭐⭐ 数据构建流程清晰完整，实验分析到位
- 价值: ⭐⭐⭐⭐ 填补多模态TPP评估空白，连接TPP和LLM两个社区

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Face-Human-Bench: A Comprehensive Benchmark of Face and Human Understanding for Multi-modal Assistants](face-human-bench_a_comprehensive_benchmark_of_face_and_human_understanding_for_m.md)
- [\[NeurIPS 2025\] mmWalk: Towards Multi-modal Multi-view Walking Assistance](mmwalk_towards_multi-modal_multi-view_walking_assistance.md)
- [\[ACL 2025\] AGRI-CM3: A Chinese Massive Multi-Modal Multi-Level Benchmark for Agricultural Understanding](../../ACL2025/multimodal_vlm/agri-cm3_a_chinese_massive_multi-modal_multi-level_benchmark_for_agricultural_un.md)
- [\[ACL 2025\] MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark](../../ACL2025/multimodal_vlm/mmmu_pro_robust_benchmark.md)
- [\[CVPR 2026\] Point Cloud as a Foreign Language for Multi-modal Large Language Model](../../CVPR2026/multimodal_vlm/point_cloud_as_a_foreign_language_for_multi-modal_large_language_model.md)

</div>

<!-- RELATED:END -->
