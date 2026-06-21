---
title: >-
  [论文解读] VisioMath: Benchmarking Figure-based Mathematical Reasoning in LMMs
description: >-
  [ICLR 2026][Reasoning][数学推理基准] 提出VisioMath基准，包含1800道K-12数学题目，所有选项均为高度视觉相似的图表，揭示了LMM在多图像-文本对齐上的核心短板，并探索三种对齐策略实现+12.6%的提升。 现有多模态数学推理基准大多关注单图场景或文本选项，忽略了一类重要且常见的题型：所有答…
tags:
  - "ICLR 2026"
  - "Reasoning"
  - "数学推理基准"
  - "多图像推理"
  - "视觉相似性"
  - "图文对齐"
  - "LMM评测"
---

# VisioMath: Benchmarking Figure-based Mathematical Reasoning in LMMs

**会议**: ICLR 2026  
**arXiv**: [2506.06727](https://arxiv.org/abs/2506.06727)  
**代码**: [GitHub](https://github.com/Nefefilibata/VisioMath)  
**领域**: 多模态VLM  
**关键词**: 数学推理基准, 多图像推理, 视觉相似性, 图文对齐, LMM评测

## 一句话总结

提出VisioMath基准，包含1800道K-12数学题目，所有选项均为高度视觉相似的图表，揭示了LMM在多图像-文本对齐上的核心短板，并探索三种对齐策略实现+12.6%的提升。

## 研究背景与动机

现有多模态数学推理基准大多关注单图场景或文本选项，忽略了一类重要且常见的题型：**所有答案选项均为图表**的题目。这类题目在K-12数学教育中普遍存在，要求对视觉极为相似的几何图形、函数曲线等进行精细比较推理。

已有多图像基准（如MathVerse-mv、MV-Math）缺乏对**高度视觉相似**的系统考量。VisioMath的核心观察：LMM在区分几乎相同的图表选项时系统性失败，其主要失败模式是**图文错位**——模型依赖位置启发式而非文本线索进行推理。

## 方法详解

### 整体框架

VisioMath 把一类被以往基准忽略的题型——所有答案选项都是图表、且这些图表彼此高度相似——固化成一个 1800 题的多选数学基准，再围绕它做大规模 LMM 评测，最后探索能缓解模型短板的对齐策略。整条主线是「造一个专挑视觉相似性的难基准 → 量化模型在相似度上的退化 → 用对齐策略验证瓶颈可被部分修复」。

### 关键设计

**1. 选项即图表的基准构建：把「比较几乎一样的图」做成可控难度**

普通数学基准的选项是文本或单张配图，模型只要读懂题干就够；VisioMath 要求模型在 8070 张图表选项中分辨出哪一张才是答案，而这些图往往只差一条曲线的斜率或一个顶点的位置。作者从 2002–2023 年中国高中与高考真题中筛出 1800 道这类多选题，覆盖几何、代数可视化、数值比较、函数模式识别等 K-12 主题，保证题目的代表性；同时用 JSON 结构化存储、LaTeX 规范数学公式、手工裁剪保证严格的一图一选项，并经人工交叉审核，避免脏数据稀释难度。这样基准的难度不是来自题目本身有多难，而是来自「选项之间太像」这一受控变量。

**2. 用最小余弦相似度量化视觉相似性：让「有多像」变成可分箱的连续轴**

要研究相似度对推理的影响，先得能给每道题的相似度打分。作者用 Qwen multimodal-embedding-v1 把每个选项图 $x_i$ 编码成向量 $f(x_i)$，取所有选项两两余弦相似度中的最小值作为该题的相似度 $\text{Sim}(Q) = \min_{i \neq j} \cos(f(x_i), f(x_j))$——之所以取最小值而非平均，是因为只要存在一对极像的干扰项就足以让模型混淆。基准刻意保留完整的相似度谱、不做截断筛选，避免选择偏差，再把全部题目按相似度分到四个四分位区间 Q1–Q4，从而能系统观察模型准确率随相似度升高如何下滑。此外约一半题目的题干本身也含图，进一步把「多源视觉信息整合」这一维度叠加进来。

**3. 三种对齐策略：从免训练改写到 CoT 微调，逐级验证瓶颈可修**

既然主要失败模式是图文错位（模型把文字选项标签和对应的图对不上），作者就设计三种针对性干预来佐证这一诊断。前两种免训练：**图像合并**把多张选项图拼到单一布局里，减少模型跨图切换的负担；**显式视觉-文本锚点**在每张图和它对应的文本选项之间打上明确的对应标记，强制建立图文绑定。第三种是**对齐导向的 CoT 微调**，构建一份多图像思维链数据集做标准 SFT，让模型显式地一步步描述「哪张图对应哪个选项、为什么」，仅用少量数据即可带来 +12.6% 的提升。三种策略由轻到重，共同说明这道短板不是无解，而是当前模型缺乏显式的多图-文本对齐机制。

## 实验关键数据

### 主实验

| 模型 | VisioMath均分 | 无图题干 | 有图题干 |
|------|-------------|---------|---------|
| Human | 91.3 | 92.3 | 89.7 |
| Gemini 2.5 Pro | **80.9** | **86.3** | **75.2** |
| Seed1.6-Thinking | 72.3 | 83.9 | 58.0 |
| GPT-4.1 | 52.6 | 56.1 | 42.8 |
| GLM-4.5V (开源最佳) | 53.7 | 61.2 | 37.2 |
| Qwen2.5-VL-72B | 43.7 | 49.8 | 33.0 |
| Vision-R1-7B | 36.7 | 33.7 | 29.2 |
| Random | 25.6 | - | - |

| 视觉相似度区间 | Q1 (低) | Q2 | Q3 | Q4 (高) |
|---------------|---------|-----|-----|---------|
| Human | 95.7 | 91.2 | 87.6 | 89.0 |
| Gemini 2.5 Pro | 86.2 | 83.8 | 76.7 | 76.9 |
| GLM-4.5V | 68.7 | 59.3 | 44.2 | 44.7 |
| Qwen2.5-VL-7B | 33.6 | 37.8 | 29.8 | 29.6 |

### 消融实验

| 策略 | 准确率 | 提升 | 说明 |
|------|--------|------|------|
| Baseline (无策略) | 基准 | - | 原始推理 |
| 选项重排(Shuffling) | -8.7% (Gemini) | 显著下降 | 证明模型依赖位置启发式 |
| 对齐导向CoT微调 | +12.6% | 最大提升 | 少量CoT数据即可 |

| 误差分析(GLM4.5V, 50样本) | 占比 | 说明 |
|---------------------------|------|------|
| 图文错位 | **36%** | 最主要错误来源 |
| 其他推理错误 | 64% | 含计算错误、概念错误等 |

### 关键发现

- **有图题干更难**: 几乎所有LMM在题干含图的题目上准确率显著下降（Gemini下降11.1%，GLM下降24%），说明多源视觉信息整合是瓶颈
- **高相似度严重退化**: 从最低到最高相似度四分位，模型准确率下降12-15个百分点
- **图文错位是主因**: 36%的错误来自图文对齐失败，LMM倾向用位置启发式代替语义推理
- **人类vs LMM的差异**: 人类在高相似度区间准确率仅轻微下降后趋于稳定，说明人类错误更多来自概念理解，而LMM错误来自感知-对齐失败
- 开源最佳GLM-4.5V (53.7%) 与人类 (91.3%) 差距达37.6%，说明该任务远未解决

## 亮点与洞察

- 填补了图表选项数学推理评测的空白，首个系统化研究视觉相似度对多模态推理的影响
- 选项重排实验精妙地证明了LMM依赖位置启发式而非真正的语义对齐
- 视觉相似度量化方法（最小余弦相似度 + Qwen嵌入）经过严格验证
- +12.6%的CoT微调收益说明问题可通过数据策略部分缓解

## 局限与展望

- 题目来源仅限中国高考/高中（虽有英文翻译），文化和课程覆盖有限
- 基准规模1800题相对中等，细分领域样本可能不足
- 对齐策略仅为初步探索，更系统的架构级改进有待研究
- 仅评测多选题，开放式图表推理未覆盖

## 相关工作与启发

- 与MathVista、MathVerse等互补——VisioMath专注于多图选项的细粒度区分
- 图文错位问题可能普遍存在于需处理多图的VLM任务中（如文档理解、医学影像对比）
- 为LMM训练提供启示：需加强多图-文本显式对齐能力

## 评分

- 新颖性: ⭐⭐⭐⭐ 填补图表选项推理评测空白，视觉相似度量化新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖20+个模型（闭源+开源+数学专用），误差分析和对照实验充分
- 写作质量: ⭐⭐⭐⭐⭐ 结构严谨，观察-分析-策略逻辑清晰，图表直观
- 价值: ⭐⭐⭐⭐ 揭示了LMM的核心短板，但作为benchmark论文实用性取决于社区采纳

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] GeoGramBench: Benchmarking the Geometric Program Reasoning in Modern LLMs](geogrambench_benchmarking_the_geometric_program_reasoning_in_modern_llms.md)
- [\[ICLR 2026\] LEXam: Benchmarking Legal Reasoning on 340 Law Exams](lexam_benchmarking_legal_reasoning_on_340_law_exams.md)
- [\[ICLR 2026\] USTBench: Benchmarking and Dissecting Spatiotemporal Reasoning Capabilities of LLMs as Urban Agents](ustbench_benchmarking_and_dissecting_spatiotemporal_reasoning_capabilities_of_ll.md)
- [\[ICLR 2026\] FaithCoT-Bench: Benchmarking Instance-Level Faithfulness of Chain-of-Thought Reasoning](faithcot-bench_benchmarking_instance-level_faithfulness_of_chain-of-thought_reas.md)
- [\[ICLR 2026\] RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Reasoning Intervention in Large Reasoning Models](rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_reasoning_interv.md)

</div>

<!-- RELATED:END -->
