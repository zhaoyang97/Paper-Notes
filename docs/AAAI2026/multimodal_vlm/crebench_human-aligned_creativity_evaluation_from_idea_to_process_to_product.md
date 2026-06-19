---
title: >-
  [论文解读] CreBench: Human-Aligned Creativity Evaluation from Idea to Process to Product
description: >-
  [AAAI 2026][多模态VLM][创造力评估] 构建 CreBench 创造力评估基准和 CreMIT 多模态指令微调数据集（2.2K 样本、79.2K 人类反馈、4.7M 条指令），从创意想法→过程→产品三个维度 12 项指标评估 MLLM 的创造力对齐能力，微调得到的 CreExpert 大幅超越 GPT-4V。
tags:
  - "AAAI 2026"
  - "多模态VLM"
  - "创造力评估"
  - "多模态大模型"
  - "人类对齐"
  - "指令微调"
  - "benchmark"
---

# CreBench: Human-Aligned Creativity Evaluation from Idea to Process to Product

**会议**: AAAI 2026  
**arXiv**: [2511.13626](https://arxiv.org/abs/2511.13626)  
**代码**: [项目主页](https://kaixuewen.github.io/Crebench)  
**领域**: 多模态VLM  
**关键词**: 创造力评估, benchmark, 多模态指令微调, 人类对齐, LLaVA

## 一句话总结
提出 CreBench，一个覆盖创意想法→创作过程→创意产品三个维度、12个细粒度指标的多模态创造力评估基准，配套构建 CreMIT（2.2K样本、79.2K人工评价、4.7M指令）并微调出 CreExpert，在创造力评估上显著优于 GPT-4V 和 Gemini-Pro-Vision。

## 研究背景与动机

**领域现状**：MLLM 在视觉问答、图像描述等客观任务上进步显著，但创造力是高度抽象、主观、多维的人类认知能力，现有 MLLM 还无法与人类判断对齐。

**现有痛点**：
   - 缺乏专门针对创造力的评估基准
   - 现有指标（BLEU, CIDEr, CLIPScore）无法捕捉新颖性、实用性等创造力维度
   - 之前的创造力数据集覆盖面窄（无多模态、无过程数据、无人工反馈）

**核心矛盾**：创造力的开放性和主观性使得自动评估极其困难，但这恰恰是 AI 系统需要具备的核心能力。

**本文目标** 构建一个多维度、人类对齐的创造力评估基准+数据集+专家模型。

**切入角度**：从认知科学和设计理论出发，将创造力分解为想法→过程→产品三个维度，12个细粒度指标（原创性、适当性、沉浸度、发散性、结构化、评估、精化、有效性、美学、新颖性、可制造性、系统复杂性）。

**核心 idea**：构建人类对齐的多维创造力评估基准，通过专家标注+GPT-4o指令生成训练 CreExpert。

## 方法详解

### 整体框架
三阶段流水线：
1. **数据收集**：512名中学生和 AI 完成 4 个开放式创造性任务，获取文本想法+行为日志+视觉作品
2. **专家评估**：3名专家按 12 个指标评分，产生 79.2K 条评价（Fleiss' κ=0.71, ICC=0.78）
3. **指令生成**：用 GPT-4o 将专家反馈转化为 4.7M 条指令-响应对（6种 QA 格式）

### 关键设计

1. **12维创造力评估框架**:

    - 创意想法（2个指标）：原创性、适当性
    - 创作过程（5个指标）：沉浸/准备、发散、结构化、评估、精化
    - 创意产品（5个指标）：有效性、美学、新颖性、可制造性、系统复杂性
    - 每个指标 5 分制行为锚定量表

2. **CreMIT 指令数据集**:

    - 6 种 QA 格式：Reasoning, What, How, Why, Y/N, MCQ
    - 涉计人类和 AI 的创作、多模态输入（文本+过程日志+图像）

3. **CreExpert 模型**:

    - 基于 LLaVA-1.5 微调
    - 保留原始知识同时获得创造力理解能力

## 实验关键数据

### 主实验
CreExpert 在 12 个创造力维度上均显著优于 GPT-4V 和 Gemini-Pro-Vision，尤其在创作过程评估和创意产品评估上优势明显。

### 消融实验
- 多维度评估比单维度更稳定且与人类判断更一致
- 6 种 QA 格式的多样性显著提升了模型的创造力感知能力
- 人工评价的质量（κ=0.71）保证了训练数据的可靠性

### 关键发现
- 现有 MLLM（包括 GPT-4V）在创造力评估上与人类判断存在显著差距
- 通过专家标注+指令微调可以有效弥补这一差距
- 创作过程的评估是现有方法最弱的环节——因为需要理解时序行为数据

## 亮点与洞察
- **“从想法到过程到产品”的三维框架**是对创造力研究的系统化操作化，比单纯看结果新颖性更全面
- **数据集质量很高**：512 名学生的真实创作数据 + 专家多轮标准化标注，质量远超众包标注
- **6 种 QA 格式**确保模型能处理多样化的创造力相关查询

## 局限与展望
- 数据规模较小（2.2K 实例），可能不足以覆盖创造力的全部复杂性
- 只考虑了视觉设计任务，未涉及文本创作、音乐创作等其他创造力形式
- 基于 LLaVA-1.5 微调，更强的基座模型可能带来进一步提升
- 专家标注成本高，难以扩展到更大规模

## 相关工作与启发
- **vs AesBench**：AesBench 评估美学，本文评估创造力——创造力更强调原创性和想象深度而非视觉美感
- **vs DALL-E/SD 的创造力评估**：以前用多样性/新颖性近似创造力，本文用 12 维人类对齐评估更全面

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个多维度人类对齐创造力评估基准，填补重要空白
- 实验充分度: ⭐⭐⭐⭐ 多基线对比，但受限于数据规模
- 写作质量: ⭐⭐⭐⭐ 框架设计系统化，但文章很长
- 价值: ⭐⭐⭐⭐⭐ 对创造力研究和 MLLM 评估体系都有重要贡献
**领域**: 多模态VLM  
**关键词**: 创造力评估, 多模态大模型, 人类对齐, 指令微调, benchmark

## 一句话总结

构建 CreBench 创造力评估基准和 CreMIT 多模态指令微调数据集（2.2K 样本、79.2K 人类反馈、4.7M 条指令），从创意想法→过程→产品三个维度 12 项指标评估 MLLM 的创造力对齐能力，微调得到的 CreExpert 大幅超越 GPT-4V。

## 研究背景与动机

创造力是人类认知的核心能力，随着 MLLM 的快速发展，一个关键问题浮现：**这些模型理解的"创造力"是否与人类一致？**

现有问题：
1. **创造力天然高度抽象、主观、多维**，现有 MLLM 在创造力评估上表现不佳
2. **缺乏系统的评估基准**：现有视觉-语言基准（VQA、Captioning）关注客观任务，有明确的标准答案，而创造力是开放式的
3. **传统指标不足**：BLEU、CIDEr、CLIPScore 无法捕捉新颖性、实用性和想象力等创造力维度
4. **现有创造力相关数据集的局限**：缺少人类反馈、缺少指令微调数据、缺少过程类数据、缺少 AIGC 内容

**核心贡献**：首个系统化多模态创造力评估基准，覆盖从"想法"到"过程"到"产品"的完整创造力链条。

## 方法详解

### 整体框架

CreBench 由两部分组成：

**1. 评估维度体系（12 项指标，3 大维度）**

**Creative Idea（创意想法）**：
- Originality（原创性）：偏离常规方法的新颖程度
- Appropriateness（适切性）：与任务需求的相关性和可行性

**Creative Process（创造过程）**：
- Immersion/Preparation（沉浸/准备）：初始反思、观察和策略规划
- Divergence（发散）：开放式探索中的多样化和实验性想法
- Structuring（结构化）：将视觉元素整合为连贯构图
- Evaluation（评估）：持续评估和优化想法
- Elaboration（精细化）：最终视觉输出中的细节和表现力

**Creative Product（创造产品）**：
- Effectiveness（有效性）：传达解决方案的清晰度和连贯性
- Aesthetic（美学）：视觉吸引力、构图平衡和表现力
- Novelty（新颖性）：形式、内容或符号表达的原创性
- Manufacturability（可制造性）：现实世界中的可行性和功能性
- Systemic Complexity（系统复杂度）：多功能组件的整合度

每个维度采用 5 分行为锚定量表评分。

**2. CreMIT 数据集构建流水线**

### 关键设计

#### 数据收集（Stage 1）

- **任务设计**：4 个开放式现实问题解决场景（如"货物过河"），激发非常规创造性响应
- **参与者**：512 名中学生，来自 5 所学校，通过分层整群抽样确保人口统计和认知多样性
- **数据模态**：每人完成 3 个创造性任务，贡献文本想法、行为日志和视觉产出
- **AIGC 内容**：同时收集 AI 生成的创造性解决方案进行对比

#### 专家标注（Stage 2）

- 遵循 **共识评估技术（CAT）**：3 名创造力教育专家
- **两轮校准培训**后独立标注，确保评估者间一致性
- **质量控制**：持续一致性监测、定期校准会议、自动检查 + 人工复核
- **结果**：79.2K 条人类反馈，覆盖 2.2K 多维评估实例
- **信度指标**：Fleiss's $\kappa = 0.71$，ICC(2,1) = 0.78，达到"显著一致"水平

#### 指令数据生成（Stage 3）

用 GPT-4o 将专家反馈转化为 6 种类型的指令-响应对：
- **Reasoning**：分析专家评分背后的逻辑
- **What**：调查创意想法的关键特征和表达元素
- **How**：解释创意如何实现或执行
- **Why**：揭示评估结果背后的原因
- **Yes/No**：对新颖性、相关性等进行二元判断
- **MCQ**：将评估场景转化为多选评分（优秀到差）

最终生成 **4.7M 条多类型指令**。

#### CreExpert 模型

基于 LLaVA-1.5-7B 架构：
- 视觉编码器：CLIP-ViT-L14（336×336，576 visual tokens）
- 模态桥接：两层 MLP
- 语言解码器：Vicuna-v1.5

训练策略：冻结视觉编码器，仅微调投影模块和语言模型（LoRA），在 LLaMA-Factory 框架下训练。

### 损失函数 / 训练策略

- 采用监督指令微调（SFT）
- 保留通用知识的同时赋予创造力评估能力
- 在 8 × NVIDIA A40 48GB 上训练
- 数据集 50/50 划分为微调集和评估集
- 评估指标：Pearson 相关系数（模型预测与人类反馈的一致性）

## 实验关键数据

### 主实验

**表2：CreExpert 与 11 个 MLLM 的对比（Pearson 相关系数 %）**

| 模型 | Creative Idea | Creative Process | Creative Product | Overall | 排名 |
|---|---|---|---|---|---|
| **CreExpert** | **84.14%** | **72.19%** | **40.18%** | **65.50%** | 1 |
| GPT-4V | 15.16% | 45.01% | 27.64% | 29.27% | 2 |
| Gemini-Pro-Vision | 11.47% | 54.39% | 17.50% | 27.78% | 3 |
| mPLUG-Owl2 | 14.34% | 29.31% | 23.76% | 22.47% | 4 |
| LLaVA-1.5-7B | 13.06% | 28.78% | 19.87% | 20.57% | 5 |
| Qwen2.5-VL | 12.36% | 23.34% | 22.66% | 19.45% | 8 |
| TinyGPT | 3.29% | 8.15% | 7.89% | 6.44% | 12 |

**CreExpert 超越 GPT-4V 超过 35%（Overall），超越 baseline LLaVA-1.5-7B 近 45%。**

**表3：跨任务消融（Creative Idea 维度）**

| 任务 | Baseline Ori. | CreExpert Ori. | 提升 |
|---|---|---|---|
| Transport | 12.80% | 72.42% | +59.62% |
| Parking | 14.98% | 69.08% | +54.10% |
| Reach | 14.12% | 83.91% | +69.79% |
| Fence | 11.90% | 80.28% | +68.38% |

### 消融实验

- **各任务一致提升**：Originality 维度提升最大（+54%~+70%），因为创意想法主要涉及文本表达，LLM 更易对齐
- **Creative Process 提升最稳定**：因为过程数据（行为日志）提供了丰富的训练信号
- **Creative Product 相对最难**：涉及视觉评估的主观性最强，仍有较大提升空间

### 关键发现

1. **现有 MLLM 在创造力评估上表现极差**：GPT-4V 的 Overall 仅 29.27%，说明通用大模型远未理解人类创造力
2. **领域微调效果惊人**：仅用 CreMIT 微调 LLaVA-1.5-7B 就从 20.57% 提升到 65.50%
3. **Creative Product 是最难的维度**：最高也仅 40.18%，因为涉及对视觉设计的整体主观评判
4. **Immersion/Preparation 指标最易对齐**：CreExpert 达 92.24%，因为这是最结构化的过程指标
5. **开源模型之间差异不大**，但与 CreExpert 差异巨大，说明通用预训练对创造力评估几乎无帮助

## 亮点与洞察

- **首个系统化多模态创造力评估基准**：填补了重要的空白，从认知科学、设计理论出发构建维度体系
- **"从想法到过程到产品"的三维评估**打破了仅关注最终产出的传统，更符合创造力研究的理论框架（Wallas 四阶段模型等）
- **数据集规模可观**：79.2K 人类反馈 + 4.7M 指令，是创造力领域罕见的大规模标注数据
- **6 种指令类型的设计**确保模型能处理多样化的创造力查询场景
- **评估者间一致性**（$\kappa = 0.71$, ICC = 0.78）达到可接受水平，增强了基准的可信度

## 局限与展望

1. **参与者主要为中学生**：创造力水平和多样性可能有限，不代表专业设计师或艺术家
2. **仅 4 个任务场景**：覆盖面有限，可扩展到更多创造力领域（音乐、写作、编程等）
3. **Creative Product 评估仍然困难**（最高仅 40%），需要更好的视觉理解能力
4. **人类标注的主观性**：尽管 $\kappa = 0.71$，但创造力的主观本质意味着标注噪声不可避免
5. **GPT-4o 生成指令数据的质量**：可能引入偏差或不忠实于原始专家反馈
6. **可探索与创造力生成（而非仅评估）的结合**：让模型不仅能评估还能辅助创造

## 相关工作与启发

- **LLaVA 系列**：CreExpert 的基础架构，展示了指令微调的强大适应能力
- **AesBench**：美学评估基准，但美学≠创造力，创造力更强调原创性和想象力
- **DALL-E/Stable Diffusion**：图像生成模型的创造力常被简化为多样性指标，本文提出更系统的评估
- **共识评估技术（CAT）**：Amabile 提出的创造力评估黄金标准
- 启发：人类对齐不仅是偏好对齐，还包括高阶认知能力的对齐，创造力评估是一个重要方向

## 评分

| 维度 | 分数 (1-5) |
|---|---|
| 新颖性 | 4.0 |
| 技术深度 | 3.0 |
| 实验充分性 | 4.0 |
| 写作质量 | 3.5 |
| 实用价值 | 3.5 |
| **总评** | **3.6** |

## 与相关工作的对比

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] VLIC: Vision-Language Models As Perceptual Judges for Human-Aligned Image Compression](../../CVPR2026/multimodal_vlm/vlic_vision-language_models_as_perceptual_judges_for_human-aligned_image_compres.md)
- [\[AAAI 2026\] ClearAIR: A Human-Visual-Perception-Inspired All-in-One Image Restoration](clearair_a_human-visual-perception-inspired_all-in-one_image_restoration.md)
- [\[AAAI 2026\] SDEval: Safety Dynamic Evaluation for Multimodal Large Language Models](sdeval_safety_dynamic_evaluation_for_multimodal_large_language_models.md)
- [\[AAAI 2026\] Towards Human-AI Accessibility Mapping in India: VLM-Guided Annotations and POI-Centric Analysis in Chandigarh](towards_human-ai_accessibility_mapping_in_india_vlm-guided_annotations_and_poi-c.md)
- [\[AAAI 2026\] DisCode: Distribution-Aware Score Decoder for Robust Automatic Evaluation of Image Captioning](discode_distribution-aware_score_decoder_for_robust_automatic_evaluation_of_imag.md)

</div>

<!-- RELATED:END -->
