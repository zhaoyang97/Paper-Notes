---
title: >-
  [论文解读] VISA: Retrieval Augmented Generation with Visual Source Attribution
description: >-
  [ACL 2025][信息检索/RAG][视觉来源归因] VISA 提出了一种基于视觉来源归因的 RAG 方法，利用大型视觉语言模型（VLM）在检索到的文档截图中用 bounding box 高亮支持生成答案的精确区域，并构建了 Wiki-VISA 和 Paper-VISA 两个数据集验证其有效性。
tags:
  - "ACL 2025"
  - "信息检索/RAG"
  - "视觉来源归因"
  - "检索增强生成"
  - "视觉语言模型"
  - "文档截图"
  - "边界框定位"
---

# VISA: Retrieval Augmented Generation with Visual Source Attribution

**会议**: ACL 2025  
**arXiv**: [2412.14457](https://arxiv.org/abs/2412.14457)  
**代码**: 有（论文中提到将释出代码、数据和模型检查点）  
**领域**: 信息检索  
**关键词**: 视觉来源归因, 检索增强生成, 视觉语言模型, 文档截图, 边界框定位

## 一句话总结

VISA 提出了一种基于视觉来源归因的 RAG 方法，利用大型视觉语言模型（VLM）在检索到的文档截图中用 bounding box 高亮支持生成答案的精确区域，并构建了 Wiki-VISA 和 Paper-VISA 两个数据集验证其有效性。

## 研究背景与动机

1. **领域现状**：RAG 系统通过检索外部文档来增强生成的可靠性。近期工作引入了"带引用的生成"（generation with citation），让模型在生成答案的同时引用源文档。但现有方法主要是**文本级引用**——将答案链接到文档标识符。

2. **现有痛点**：文档级引用给用户带来了沉重的认知负担。用户在获得一个引用后，仍需在长篇、多页的文档中手动寻找支持答案的具体段落、表格或图片。即使是段落级引用也存在问题——需要额外的工程开发来匹配 chunk 到原始文档位置，且无法自然地在 PDF 等格式中高亮显示。

3. **核心矛盾**：现有 RAG 的来源归因粒度太粗（文档级），且受限于文本格式无法直观展示证据位置。另一方面，近期出现的**文档截图检索范式**（如 DSE）直接用截图做检索，保留了视觉信息但缺少归因能力。

4. **本文目标** 能否在视觉化 RAG 流程中实现端到端的来源归因——让模型不仅生成答案，还精确标出答案在文档截图中的位置？

5. **切入角度**：利用 VLM 已具备的图像理解和 bounding box 预测能力，将来源归因定义为在文档截图中输出支持证据的 bounding box 坐标。

6. **核心 idea**：将 RAG 的来源归因从文本引用范式转变为视觉定位范式——VLM 直接在文档截图上画 bounding box 指向答案的证据来源。

## 方法详解

### 整体框架

输入为用户文本查询 + 检索到的文档截图（1-多张）。VLM 处理多模态输入后，以自回归方式同时生成三个输出：（1）文本答案；（2）相关文档的标识符；（3）bounding box 坐标（左上角和右下角 $(x_1,y_1,x_2,y_2)$）。最终将 bounding box 画在文档截图上展示给用户。

### 关键设计

1. **视觉来源归因任务定义**:

    - 功能：形式化定义 RAG 中的视觉归因任务
    - 核心思路：给定查询 $q$，检索候选文档集 $D=\{d_1,...,d_n\}$，系统需同时返回答案 $a$、最相关文档标识 $i$、以及该文档中证据的 bounding box $B_{d^*} = [(x_1,y_1),(x_2,y_2)]$。所有输入都是截图图像，整个过程可用 next-token prediction 统一建模
    - 设计动机：将归因和生成统一为一个自回归生成任务，避免多阶段管道

2. **Wiki-VISA 和 Paper-VISA 数据集构建**:

    - 功能：提供高质量的训练和评估数据
    - 核心思路：**Wiki-VISA** 基于 Natural Questions 数据集，用 Selenium 渲染原始 Wikipedia 网页截图（980px 宽 × 最多 3920px 高），以 NQ 的 short answer 作为答案目标，long answer 对应的 HTML 元素在截图中的位置作为 bounding box 目标（87k 训练 / 3000 测试）。**Paper-VISA** 基于 PubLayNet（生物医学论文 PDF 页面），利用 VLM 为每个预标注的布局元素合成查询和答案（100k 训练 / 2160 测试）。另外还构建了 **FineWeb-VISA**（60k 爬取网页截图）作为补充训练数据
    - 设计动机：Wiki-VISA 提供人工标注质量的通用知识评估，Paper-VISA 覆盖科学论文领域，两者布局差异大可测试泛化能力

3. **多候选文档训练设定**:

    - 功能：模拟真实 RAG 场景中存在多个检索候选
    - 核心思路：从 DSE 检索器的 top-20 结果中随机采样 $m-1$ 个非正确文档作为 hard negatives，与正确文档混合输入。20% 概率随机替换正确文档测试模型识别"无答案"场景的能力。先在单文档上训练两个 epoch，再用单文档权重初始化多文档训练一个 epoch
    - 设计动机：随机采样而非直接取 top-m 避免模型依赖特定检索器和文档位置；加入无答案场景测试拒绝能力

### 损失函数 / 训练策略

标准 next-token prediction + cross-entropy loss。使用 LoRA 微调 Qwen2-VL-2B 和 Qwen2-VL-7B，学习率 1e-4，batch size 64，4×H100 GPU。训练时采用随机裁剪增强（在 bounding box 外裁剪），提高模型对不同输入尺寸的泛化能力。多候选训练时冻结图像编码器以节省显存。

## 实验关键数据

### 主实验

**单文档设定**：

| 模型 | Wiki-VISA bbx Avg | Wiki-VISA ans Avg | Paper-VISA bbx Avg | Paper-VISA ans Avg |
|------|-------------------|-------------------|--------------------|--------------------|
| QWen2-VL-72B (zero-shot) | 1.5% | 60.4% | 1.5% | 43.1% |
| VISA-2B-single | 37.5% | 57.1% | 63.0% | 38.3% |
| VISA-7B-single | 54.2% | 65.2% | 68.2% | 43.8% |

**多文档设定**（VISA-7B，3个候选，含无答案样本）：

| 设定 | Wiki-VISA bbx | Wiki-VISA ans | Paper-VISA bbx | Paper-VISA ans |
|------|---------------|---------------|----------------|----------------|
| Multi-candidate, Full | 41.6% | 51.1% | 66.8% | 50.3% |

### 消融实验

| 训练数据 | Wiki-VISA bbx | Wiki-VISA ans | Paper-VISA bbx | Paper-VISA ans |
|----------|---------------|---------------|----------------|----------------|
| Wiki only | 54.2% | 65.2% | 27.8% | 36.2% |
| Paper only | 0.2% | 42.6% | 68.2% | 43.8% |
| FineWeb only | 37.6% | 50.2% | 22.0% | 43.3% |
| Wiki+Paper+FineWeb | 58.1% | 64.8% | 67.6% | 44.3% |

### 关键发现

- **Zero-shot 极其困难**：QWen2-VL-72B 虽能生成合理答案（60.4%），但 bounding box 准确率仅 1.5%，表明现有 VLM 远不具备 zero-shot 视觉归因能力
- **文档位置对性能影响大**：首页段落 bbx 准确率 75.6% vs 非首页段落 50.1%（Wiki-VISA），多页长文档的归因是主要挑战
- **跨域泛化困难**：Paper→Wiki 迁移 bbx 准确率几乎为零（0.2%），Wiki→Paper 为 27.8%；Wiki 的多页性质提供了更丰富的训练信号
- **多候选 vs 单文档**：从单文档到三候选文档，bbx 准确率从 54.2% 降至 37.7%（下降 17 个百分点），多文档归因显著更难
- **FineWeb 数据增强有效**：Wiki+FineWeb 在 Wiki-VISA 上 bbx 从 54.2% 提升到 58.2%，多样化布局有助于泛化

## 亮点与洞察

- **视觉归因范式的开创性定义**：将 RAG 的来源归因从文本引用转变为视觉定位，这是首次在 VLM-RAG 框架中将答案生成和精确视觉定位统一为一个任务。这个思路可迁移到任何需要追溯信息来源的场景
- **数据集构建策略的巧妙之处**：Wiki-VISA 利用 NQ 的 long answer HTML 元素自然获得 bounding box 标注，无需额外人工标注；Paper-VISA 用 VLM 合成问答避免了手动标注科学论文的高成本
- **对 VLM 能力的有意义评估**：VISA 不仅是一个应用，也是检验 VLM 自解释能力和精确定位能力的 benchmark

## 局限与展望

- 跨域泛化能力严重不足，Paper→Wiki 几乎为零，限制了实际部署
- 目前只支持单 bounding box，真实场景中答案证据可能分散在多个区域
- 只测试了 Qwen2-VL 系列，未测试其他 VLM（如 InternVL、LLaVA）
- 多页长文档上的定位仍然是主要瓶颈（非首页段落准确率大幅下降）
- 当前只支持截图级输入，未探索与 OCR pipeline 的结合

## 相关工作与启发

- **vs 传统文本 RAG 归因**: Gao et al. 的 citation generation 只提供文档 ID，用户需自行查找；VISA 直接定位到文档截图中的具体位置，认知负担大幅降低
- **vs DSE (Ma et al.)**: DSE 用文档截图做检索但没有归因能力；VISA 在 DSE 基础上补齐了视觉归因环节，可构成端到端视觉 RAG 系统
- **vs GUI grounding**: Lin et al., Cheng et al. 在 GUI 界面中做 UI 元素定位；VISA 将类似能力应用到内容密集的文档中，难度更大（布局更复杂、内容更密集）

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次在 VLM-RAG 框架中提出视觉来源归因，问题定义有开创性
- 实验充分度: ⭐⭐⭐⭐ 两个数据集 + 多种设定（单文档/多文档/跨域），分析细致
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，数据集构建过程详尽
- 价值: ⭐⭐⭐⭐ 为 RAG 可验证性提出新范式，但跨域泛化问题限制了即时实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Evaluation of Attribution Bias in Generator-Aware Retrieval-Augmented Large Language Models](evaluation_of_attribution_bias_in_generator-aware_retrieval-augmented_large_lang.md)
- [\[ACL 2025\] GeAR: Generation Augmented Retrieval](gear_generation_augmented_retrieval.md)
- [\[ACL 2025\] Accelerating Adaptive Retrieval Augmented Generation via Instruction-Driven Representation Reduction of Retrieval Overlaps](accelerating_adaptive_retrieval_augmented_generation_via_instruction-driven_repr.md)
- [\[ACL 2025\] Unanswerability Evaluation for Retrieval Augmented Generation](unanswerability_evaluation_for_retrieval_augmented_generation.md)
- [\[ACL 2025\] MT-RAIG: Novel Benchmark and Evaluation Framework for Retrieval-Augmented Insight Generation over Multiple Tables](mt-raig_novel_benchmark_and_evaluation_framework_for_retrieval-augmented_insight.md)

</div>

<!-- RELATED:END -->
