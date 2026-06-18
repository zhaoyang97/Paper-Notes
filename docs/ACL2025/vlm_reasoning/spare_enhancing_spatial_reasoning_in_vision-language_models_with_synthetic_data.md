---
title: >-
  [论文解读] SpaRE: Enhancing Spatial Reasoning in Vision-Language Models with Synthetic Data
description: >-
  [ACL 2025][多模态VLM][空间推理] 本文发现现有VLM数据集中空间关系数据严重匮乏（前17%的关系占据90%以上样本），提出从DOCCI、Localized Narratives和PixMo-Cap等超详细图像描述数据集中，利用LLM自动提取45.5万样本（340万QA对）的空间推理合成数据，微调后的SpaRE模型在What's Up基准上实现最高49%的性能提升，同时不损害通用VL能力。
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "空间推理"
  - "合成数据"
  - "超详细描述"
  - "视觉问答"
  - "VLM微调"
---

# SpaRE: Enhancing Spatial Reasoning in Vision-Language Models with Synthetic Data

**会议**: ACL 2025  
**arXiv**: [2504.20648](https://arxiv.org/abs/2504.20648)  
**代码**: 无  
**领域**: 多模态VLM / 空间推理  
**关键词**: 空间推理, 合成数据, 超详细描述, 视觉问答, VLM微调

## 一句话总结

本文发现现有VLM数据集中空间关系数据严重匮乏（前17%的关系占据90%以上样本），提出从DOCCI、Localized Narratives和PixMo-Cap等超详细图像描述数据集中，利用LLM自动提取45.5万样本（340万QA对）的空间推理合成数据，微调后的SpaRE模型在What's Up基准上实现最高49%的性能提升，同时不损害通用VL能力。

## 研究背景与动机

**领域现状**：视觉语言模型（VLM）在图像描述、视觉问答等任务上表现优异，但在空间推理方面持续挣扎。多项研究（VSR、What's Up等）已证实，即使是SOTA VLM在判断基本空间关系（如左右、上下）时也接近随机水平。

**现有痛点**：空间关系在现有VL数据集中极其稀少。作者分析InternVL2使用的SFT数据集发现，VQAv2仅1.44%、GQA仅3.07%的样本涉及空间关系。更严重的是分布极度不均衡：前17%的常见关系（如on、left、under）占据了90%以上的空间关系样本，大量关系（如facing、opposite、surrounding）严重不足。

**核心矛盾**：先前的解决方案要么使用合成场景图像（如CLEVR、STUPD），由于简单几何形状导致domain gap而难以泛化到真实世界；要么使用人工标注的真实数据，但受限于规模和多样性（如VSR+What's Up共约8K样本）。

**本文目标** 如何大规模生成覆盖丰富空间关系类型的真实世界图像训练数据，以提升VLM的空间推理能力。

**切入角度**：注意到近年来出现的超详细图像描述数据集（DOCCI、PixMo-Cap、Localized Narratives）中包含丰富的空间关系描述，可以作为空间推理QA对的天然来源。

**核心 idea**：从真实世界图像的超详细描述中，用小型LLM自动提取空间推理QA对作为训练数据，绕过合成图像的domain gap问题。

## 方法详解

### 整体框架

整体流程是一个三阶段pipeline：（1）从三个超详细描述数据集中筛选含空间信息的描述；（2）用Qwen2.5-3B-Instruct生成空间推理QA对；（3）通过多重质量保证过滤后用于VLM微调。输入是真实世界图像+超详细描述，输出是455K样本、340万QA对的训练集。

### 关键设计

1. **数据源选择与预过滤**:

    - 功能：从158万图像描述对中筛选含空间信息的子集
    - 核心思路：选择三个互补的超详细描述数据集：DOCCI（1.5万张,136词/描述,人工标注的精细描述）、Localized Narratives（84.9万张,42词/描述,语音转录+鼠标轨迹结合COCO/Flickr30k等图像）、PixMo-Cap（71.7万张,196词/描述,跨70个主题的稠密描述）。使用Qwen2.5-3B-Instruct对描述进行空间关系分类，过滤掉约65%不含空间信息的描述。
    - 设计动机：超详细描述天然包含物体间空间关系的显式描述，避免了合成图像的domain gap，三个数据集在描述长度、风格和图像来源上互补

2. **LLM驱动的QA对生成**:

    - 功能：从筛选后的描述中自动提取多样化的空间推理QA对
    - 核心思路：构建详细的ICL prompt指导Qwen2.5-3B-Instruct从描述中提取空间推理QA对，覆盖位置、方向和距离等多种空间关系。以temperature=0生成结构化JSON输出，每个描述可产生多个QA对（平均约7.4对）。生成范围限于描述中提到的空间关系，确保有据可依。
    - 设计动机：使用小型LLM（3B参数）即可胜任QA提取任务，降低了生成成本，同时结构化输出便于自动化处理

3. **多重质量保证体系**:

    - 功能：确保生成的QA对质量和准确性
    - 核心思路：按计算成本从低到高依次应用五项检查：(1) 去重：全字匹配+CLIP语义相似度(阈值0.95)检测重复问题；(2) 引用检查：过滤引用"描述"而非直接问图像的QA对；(3) 答案-描述一致性：验证答案关键词存在于原始描述中；(4) 图像-问题一致性：CLIPScore(阈值0.25)检查图像与问题的语义对齐；(5) 空间关系验证：确认QA对确实涉及空间推理。人工抽样400条QA对，错误率约4%。
    - 设计动机：合成数据的质量是关键瓶颈，多层次过滤既保证了精度又控制了计算成本

### 损失函数 / 训练策略

使用标准交叉熵loss微调VLM，仅计算文本token的loss，不计算视觉token的loss。2B模型全参数训练，7B模型使用LoRA以节省显存。训练使用bfloat16精度，前1000步线性warmup后余弦衰减，梯度裁剪最大范数1.0。在4×NVIDIA A40（48GB）上训练，每个配置跑5个随机种子取均值。

## 实验关键数据

### 主实验

| 模型 | VSR | What's Up A | What's Up B | 3DSRBench | RealWorldQA | 空间平均 |
|------|-----|------------|------------|-----------|-------------|---------|
| Qwen2VL-2B | 70.3 | 44.6 | 79.1 | 46.5 | 58.6 | 59.8 |
| **SpaRE-2B** | **80.8** | **93.4** | **95.1** | **54.4** | **63.5** | **77.6** |
| 提升 | +10.5 | **+48.8** | +16.0 | +7.9 | +4.9 | +17.8 |
| Qwen2VL-7B | 82.3 | 99.5 | 99.3 | 49.2 | 67.7 | 79.2 |
| **SpaRE-7B** | **85.4** | **100.0** | **100.0** | **57.5** | **68.8** | **82.3** |
| GPT-4o | 79.0 | 100.0 | 100.0 | 45.3 | 61.0 | 77.9 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| SpaRE-2B MMMU | 40.0 vs 34.0 (Qwen2VL-2B) | 通用能力不降反升 |
| SpaRE-2B MMBench | 71.6 vs 72.0 (Qwen2VL-2B) | 通用性能基本持平 |
| SpaRE-7B MMMU | 51.0 vs 51.0 (Qwen2VL-7B) | 7B模型通用能力完全保持 |
| SpaRE-2B vs InternVL2-2B | 77.6 vs 68.9 空间平均 | 在同规模中全面领先 |
| SpaRE-7B vs GPT-4o | 82.3 vs 77.9 空间平均 | 7B开源模型超越GPT-4o |

### 关键发现

- SpaRE-2B在What's Up A上实现49%的绝对提升（44.6→93.4），这是所有基准中最大的单项提升
- 空间推理增强不以通用能力为代价：SpaRE模型在MMMU、MMBench等通用基准上与原始模型持平甚至略有提升
- SpaRE-7B在空间推理平均指标上（82.3%）超越了GPT-4o（77.9%），证明了合成数据方法的有效性
- 训练过程中观察到的"良性幻觉"（与图像相关但非空间推理的QA对）被保留在训练中，反而有助于保持通用性能

## 亮点与洞察

- 问题定义精准：量化了空间关系数据稀缺问题（前17%关系占90%样本），为解决方案提供了明确目标
- 方法极其简洁有效：无需合成图像、无需大型模型、无需复杂训练流程，仅通过从现有描述中提取QA对就实现了巨大提升
- 使用仅3B参数的LLM进行QA生成，成本极低但效果显著
- "良性幻觉"的发现是一个有趣的副产品——与空间关系无关但与图像相关的QA对实际上帮助维持了模型的泛化能力

## 局限与展望

- 代码和数据集尚未开源（论文中声称将在适当时候共享）
- 仅在2B和7B规模上实验，更大模型上的效果未知
- QA生成依赖于描述中已有的空间信息，对于描述中未提及的空间关系无法覆盖
- 约4%的QA对错误率意味着训练数据中仍存在噪声
- 数据集以英文为主，多语言空间推理的泛化性未验证
- 空间关系分类的长尾问题虽然改善但可能仍不完全均衡

## 相关工作与启发

- VSR (Liu et al., 2023a) 和 What's Up (Kamath et al., 2023) 是空间推理评测的核心基准
- CLEVR (Johnson et al., 2017) 和 STUPD (Agrawal et al., 2023) 的domain gap问题正是本文方法要克服的
- DOCCI、PixMo-Cap等超详细描述数据集的出现为本文方法提供了基础
- 该方法的思路可推广到其他VLM薄弱领域：识别数据稀缺问题 → 找到信息丰富的数据源 → 合成训练数据

## 评分

- **新颖性**: 7/10 — 方法思路清晰但技术贡献相对增量
- **技术深度**: 6/10 — 方法简单直接，缺少深层技术创新
- **实验充分性**: 8/10 — 多基准多模型评估，5种子均值可靠
- **写作质量**: 8/10 — 问题分析深入，论述逻辑清晰
- **应用价值**: 8/10 — 方法简单有效，易于复现和扩展到其他场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Synthetic Data is an Elegant GIFT for Continual Vision-Language Models](../../CVPR2025/multimodal_vlm/synthetic_data_is_an_elegant_gift_for_continual_vision-language_models.md)
- [\[ACL 2025\] Scaling Text-Rich Image Understanding via Code-Guided Synthetic Multimodal Data Generation](code_guided_text_rich_image.md)
- [\[ICML 2026\] Left-Right Symmetry Breaking in CLIP-style Vision-Language Models Trained on Synthetic Spatial-Relation Data](../../ICML2026/multimodal_vlm/left-right_symmetry_breaking_in_clip-style_vision-language_models_trained_on_syn.md)
- [\[ACL 2025\] VReST: Enhancing Reasoning in Large Vision-Language Models through Tree Search and Self-Reward Mechanism](vrest_tree_search_vlm_reasoning.md)
- [\[CVPR 2025\] ESPIRE: A Diagnostic Benchmark for Embodied Spatial Reasoning of Vision-Language Models](../../CVPR2025/multimodal_vlm/espire_a_diagnostic_benchmark_for_embodied_spatial_reasoning_of_vision-language_.md)

</div>

<!-- RELATED:END -->
