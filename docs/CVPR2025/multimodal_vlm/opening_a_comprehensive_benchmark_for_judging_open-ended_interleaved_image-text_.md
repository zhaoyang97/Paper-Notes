---
title: >-
  [论文解读] OpenING: A Comprehensive Benchmark for Judging Open-ended Interleaved Image-Text Generation
description: >-
  [CVPR 2025][多模态VLM][图文交错生成] 本文提出 OpenING 基准（5,400 条人工标注实例、56 个真实场景任务）和 IntJudge 评判模型（与人类判断一致率 82.42%），填补了开放式图文交错生成评估的真空，发现当前集成管线（如 Gemini+Flux）大幅领先端到端模型，但所有方法仍远不及人类标注质量。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "图文交错生成"
  - "基准评估"
  - "评判模型"
  - "多模态生成"
  - "人类对齐"
---

# OpenING: A Comprehensive Benchmark for Judging Open-ended Interleaved Image-Text Generation

**会议**: CVPR 2025  
**arXiv**: [2411.18499](https://arxiv.org/abs/2411.18499)  
**代码**: [https://opening-benchmark.github.io](https://opening-benchmark.github.io)  
**领域**: 多模态VLM  
**关键词**: 图文交错生成, 基准评估, 评判模型, 多模态生成, 人类对齐

## 一句话总结

本文提出 OpenING 基准（5,400 条人工标注实例、56 个真实场景任务）和 IntJudge 评判模型（与人类判断一致率 82.42%），填补了开放式图文交错生成评估的真空，发现当前集成管线（如 Gemini+Flux）大幅领先端到端模型，但所有方法仍远不及人类标注质量。

## 研究背景与动机

**领域现状**：多模态大模型在视觉理解和生成方面取得了快速进展，图文交错生成（interleaved image-text generation）成为通向通用人工智能的重要能力。早期模型如 DALL-E、Stable Diffusion 聚焦单向任务（文生图或图理解），近期出现了 Emu-3、Chameleon 等原生自回归模型和 SEED-X 等两阶段模型，能够交替生成文本和图像。

**现有痛点**：评估体系严重滞后于模型进展。现有基准（OpenLEAF 仅 660 实例、InterleavedBench 仅 815 实例）规模小、覆盖场景窄、查询多样性不足。更关键的是，现有评估严重依赖 GPT-based 评分，而 GPT 存在偏向自身生成内容的偏差、数据泄露风险和 API 隐私问题。传统指标如 BLEU/ROUGE 无法衡量视觉质量，FID/IS 忽略文本元素，CLIPScore 无法全面评估开放式交错内容。

**核心矛盾**：社区缺乏一个规模足够大、任务足够丰富、且有可靠离线评判模型的图文交错生成基准。没有有效的评估，模型的进步方向就不明确。

**本文目标** (1) 构建大规模、高质量、涵盖真实场景的图文交错生成基准；(2) 训练一个与人类判断高度一致的离线评判模型；(3) 系统评估当前方法的优劣势。

**切入角度**：从真实日常场景出发（旅行指南、设计、头脑风暴等），用自顶向下的方式设计 23 个元主题和 56 个具体任务，并组织 50 人团队进行高质量标注。

**核心 idea**：构建覆盖 56 个真实任务的大规模图文交错基准 OpenING 和与人类对齐度 82.42% 的评判模型 IntJudge，系统评估图文交错生成方法。

## 方法详解

### 整体框架

OpenING 项目包含三个核心贡献：(1) OpenING 基准——5,400 条人工标注的多步图文交错实例；(2) IntJudge——基于 Qwen2-VL-7B 训练的评判模型；(3) Interleaved Arena——成对比较评估框架。数据从 20+ 来源收集，经概念化→采集→标注→过滤→处理五阶段构建。

### 关键设计

1. **自顶向下的任务概念化与数据标注**:

    - 功能：确保基准覆盖真实世界场景的广度和深度
    - 核心思路：借助 AI agent 头脑风暴确定 23 个元主题（时尚、烹饪、旅行、设计等），细分为 56 个具体任务。从小红书、YouTube、Google、OpenDataLab 等 20+ 来源收集数据。28 名专业标注员在 14 名数据专家监督下使用自研 IntLabel 工具进行标注，每个实例限制在 10 步以内。交叉检查确保一致性，不合格数据淘汰后用 GPT-4o+SDXL 生成内容补充。中文文本由 GPT-4o 翻译为英文并人工校验
    - 设计动机：图文交错数据的收集和标准化极为困难——不同领域的数据格式差异大，质量参差不齐，需要严格的流程把控

2. **Interleaved Arena 成对评估框架**:

    - 功能：通过成对比较实现更稳定的开放式评估
    - 核心思路：从测试集中抽取数据实例，对两个匿名模型的输出进行成对比较。评估基于七个维度：正确性、图文一致性、多步连贯性、内容质量、人类偏好对齐、完整性、内容丰富度。使用轮盘匹配算法为每个数据实例采样 $E$ 个不同的对战对，覆盖时间 $T_k = \lceil \frac{|\mathcal{M}|(|\mathcal{M}|-1)}{2E} \cdot \frac{D_k}{|\mathcal{P}_k|} \rceil$，确保所有模型都被评估到
    - 设计动机：成对比较比主观评分更稳定（先前研究已证明过多平局会降低评估效率），Arena 式评估在 LLM 评估中已被验证有效

3. **IntJudge 评判模型训练**:

    - 功能：提供离线、可复现、与人类高度对齐的自动评估
    - 核心思路：基于 Qwen2-VL-7B 训练。数据来源两部分：(1) Dev Set 上的人工标注成对比较数据；(2) Reference-Augmented Generation (RAG) 扩充数据——给模型提供黄金答案后生成 RAG 结果，与普通生成结果配对（RAG 结果为赢者）。训练损失结合四项：$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{CE}} + \lambda_2 \mathcal{L}_{\text{CT}} + \lambda_3 \mathcal{L}_{\text{MSE}} + \lambda_4 \mathcal{L}_{\text{PR}}$（交叉熵 + 对比 + MSE + 成对排序损失）
    - 设计动机：GPT-based 评估存在偏向自身内容的偏差、API 隐私和成本问题。离线评判模型可控、可复现、无数据泄露风险

### 损失函数 / 训练策略

IntJudge 使用四损失加权训练：CE 保证分类准确性，对比损失区分好坏输出，MSE 拉近预测分数与实际分数，成对排序损失确保正确的偏好排序。

## 实验关键数据

### 主实验（模型胜率排名 — IntJudge 评估）

| 方法 | 类型 | FDT 胜率 | w/ Tie(.5) 胜率 |
|------|------|---------|----------------|
| Human | 标注 | 87.46% | 84.23% |
| GPT-4o+DALL-E3 | 集成管线 | 85.02% | 80.68% |
| Gemini1.5+Flux | 集成管线 | 68.30% | 65.41% |
| SEED-X | 两阶段 | 49.86% | 49.72% |
| Anole | 端到端 | 53.42% | 51.33% |
| SEED-LLaMA | 端到端 | 50.13% | 48.48% |
| Show-o | 两阶段 | 31.49% | 32.87% |
| NExT-GPT | 端到端 | 30.96% | 32.58% |
| MiniGPT-5 | 端到端 | 24.47% | 27.85% |
| GILL | 端到端 | 24.87% | 30.32% |

### 评判模型一致性

| 评判者 | 与人类一致率 (FDT) | 与人类一致率 (w/o Tie) |
|--------|-------------------|---------------------|
| GPT-4o | 71.08% | 74.58% |
| **IntJudge** | **82.42%** | **87.46%** |
| 提升 | **+11.34%** | **+12.88%** |

### 关键发现

- 集成管线（GPT-4o+DALL-E3）在所有评估方式下都大幅领先，胜率 85%+，说明目前图文交错生成仍需强力的独立文本和图像生成模型配合
- 端到端模型（Anole、SEED-LLaMA 等）胜率集中在 25-53%，与人类标注（87%+）差距巨大
- IntJudge 以 82.42% 的与人类一致率显著超越 GPT-4o 的 71.08%，作为离线评判模型实现了更好的人类对齐
- 文本方面 GPT 生成可以比人工标注更丰富信息，但图像方面人工标注的自然图像仍优于生成图像
- IntJudge 在未见过的模型（unseen models）上也保持了较好的泛化性能

## 亮点与洞察

- **从评估真空中建立标准**：图文交错生成领域几乎没有可靠的评估体系，OpenING 一次性提供了数据、评判模型和排行榜，构建了完整的评估基础设施。50 人团队 3 个月的投入保证了数据质量
- **RAG 数据扩充策略**：用黄金答案作为参考让模型生成 RAG 结果，与普通结果配对训练评判模型——这种自举式的训练数据扩充方式巧妙且低成本，可迁移到其他需要评判模型的场景
- **七维度评估体系**：从正确性到人类偏好对齐的七个维度，比简单的单一分数提供了更细粒度的评估信号，有助于诊断模型的具体弱点

## 局限与展望

- 5,400 实例虽然比前作大了一个数量级，但对 56 个任务来说平均每任务仅 ~96 个实例，某些任务覆盖可能不足
- 数据中文翻英文可能引入翻译偏差，影响非中文背景任务的自然度
- IntJudge 基于 Qwen2-VL-7B，模型容量有限，对非常复杂的交错内容判断可能不够精确
- 评估框架侧重于内容质量，缺少对生成效率（延迟、成本）的考量
- 部分数据用 GPT-4o+SDXL 补充，可能引入分布偏差

## 相关工作与启发

- **vs InterleavedBench**: 仅 815 实例、10 个任务、无离线评判模型；OpenING 在规模（5,400 实例）、覆盖面（56 任务）和评估工具（IntJudge）上全面升级
- **vs OpenLEAF**: 仅 660 实例、2 个元主题、不开源、无离线评判；OpenING 在各维度全面超越
- **vs LMSYS Arena**: Arena 式评估在纯文本领域已很成熟，OpenING 将其扩展到多模态交错生成领域是自然的延伸
- 集成管线远超端到端模型的发现很关键——意味着当前图文交错生成的瓶颈不在框架设计而在基础模型能力（尤其是图像生成质量）

## 评分

- 新颖性: ⭐⭐⭐⭐ 填补了图文交错生成评估的空白，但核心方法（基准构建+评判模型训练）偏工程
- 实验充分度: ⭐⭐⭐⭐⭐ 10 个模型、3 种评判方式、多维度对比，非常全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，数据统计详实，任务设计合理
- 价值: ⭐⭐⭐⭐⭐ 为快速发展的图文交错生成领域提供了急需的评估基础设施

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] CoMM: A Coherent Interleaved Image-Text Dataset for Multimodal Understanding and Generation](comm_a_coherent_interleaved_image-text_dataset_for_multimodal_understanding_and_.md)
- [\[ICML 2026\] ECA: Efficient Continual Alignment for Open-Ended Image-to-Text Generation](../../ICML2026/multimodal_vlm/eca_efficient_continual_alignment_for_open-ended_image-to-text_generation.md)
- [\[CVPR 2025\] Mosaic of Modalities: A Comprehensive Benchmark for Multimodal Graph Learning](mosaic_of_modalities_a_comprehensive_benchmark_for_multimodal_graph_learning.md)
- [\[ACL 2025\] CrafText Benchmark: Advancing Instruction Following in Complex Multimodal Open-Ended World](../../ACL2025/multimodal_vlm/craftext_benchmark_advancing_instruction_following_in_complex_multimodal_open-en.md)
- [\[CVPR 2026\] Multimodal RewardBench 2: Evaluating Omni Reward Models for Interleaved Text and Image](../../CVPR2026/multimodal_vlm/multimodal_rewardbench_2_evaluating_omni_reward_models_for_interleaved_text_and_.md)

</div>

<!-- RELATED:END -->
