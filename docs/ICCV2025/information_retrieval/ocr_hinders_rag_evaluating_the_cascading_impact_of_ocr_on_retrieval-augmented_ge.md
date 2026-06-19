---
title: >-
  [论文解读] OCR Hinders RAG: Evaluating the Cascading Impact of OCR on Retrieval-Augmented Generation
description: >-
  [ICCV 2025][信息检索/RAG][OCR] 提出 OHRBench——首个评估 OCR 对 RAG 系统级联影响的基准，包含 7 个领域的 8561 张文档图像和 8498 个 QA 对，系统性地揭示了 OCR 产生的语义噪声（Semantic Noise）和格式噪声（Formatting Noise）对检索和生成两阶段的不同影响模式。
tags:
  - "ICCV 2025"
  - "信息检索/RAG"
  - "OCR"
  - "RAG系统"
  - "文档解析"
  - "知识库质量"
  - "benchmark"
---

# OCR Hinders RAG: Evaluating the Cascading Impact of OCR on Retrieval-Augmented Generation

**会议**: ICCV 2025  
**arXiv**: [2412.02592](https://arxiv.org/abs/2412.02592)  
**代码**: [GitHub](https://github.com/opendatalab/OHR-Bench)  
**领域**: 信息检索  
**关键词**: OCR噪声, RAG系统, 文档解析, 知识库质量, benchmark

## 一句话总结

提出 OHRBench——首个评估 OCR 对 RAG 系统级联影响的基准，包含 7 个领域的 8561 张文档图像和 8498 个 QA 对，系统性地揭示了 OCR 产生的语义噪声（Semantic Noise）和格式噪声（Formatting Noise）对检索和生成两阶段的不同影响模式。

## 研究背景与动机

RAG（检索增强生成）系统的一个被忽视的关键环节是**外部知识库的构建质量**。在实际应用中大量知识以非结构化 PDF 文档形式存在，需要通过 OCR 解析为结构化数据才能被 RAG 使用。然而：

**OCR 不完美**：即使是最先进的 OCR 方案，在复杂文档（扫描件、手写体、多语言、多列排版）上仍会出错

**结构化数据的表示不唯一**：同一个表格可以用 Markdown、LaTeX 或 HTML 表示，这种不一致本身就是噪声

**RAG 对噪声敏感**：已有研究（InfoRAG、RAAT）表明 RAG 对输入噪声很敏感，但只关注了 chunk 级别的检索噪声，**OCR 阶段引入的噪声被完全忽略**

**现有评估缺口**：
   - 已有 RAG benchmark 不评估 OCR 的影响
   - 已有 OCR benchmark 不评估对 RAG 下游的影响
   - 缺乏覆盖多种 OCR 挑战（扫描、手写、多语言、历史文档）的数据

核心观点：OCR 噪声会**级联放大**——OCR 错误导致检索错误，检索错误导致生成错误，三级错误叠加远超单级影响。

## 方法详解

### 整体框架

OHRBench 由三部分组成：
1. **文档数据集**：7 个领域的 PDF + ground truth 结构化数据 + QA 对
2. **OCR 噪声建模**：语义噪声和格式噪声的定义与扰动生成
3. **三级评估协议**：分别评估 OCR 对检索、生成、端到端 RAG 的影响

### 关键设计

1. **文档数据收集（7 领域覆盖）**：

    - 教科书（Textbook）、法律（Law）、金融（Finance）、报纸（Newspaper）、手册（Manual）、学术论文（Academic）、行政文档（Administration）
    - 来源：DUDE、OmniDocBench、FinanceBench、CUAD、GNHK 等公开数据集 + 网络公开资源
    - 包含 9 类 OCR 挑战特征：手写、扫描、历史文档、多语言、表格、公式、图表、多列布局、水印
    - 共 1261 个 PDF，8561 页，ground truth 由 Mathpix 初提取 + 专家级标注员修正

2. **QA 对生成与质量控制**：

    - 围绕 5 种证据源：纯文本（41.5%）、表格（27.8%）、公式（14.9%）、图表（9.0%）、阅读顺序（8.1%）
    - 任务类型：理解（71.9%）和推理（28.1%）；单页（90.1%）和多页（9.9%）
    - 使用 GPT-4o 生成，经三重过滤：
        - **RAG 兼容性**：问题需上下文依赖，不能被模型内部知识回答
        - **任务定义忠实度**：与定义的任务类型匹配
        - **正确性**：提供 oracle 上下文重复采样验证
    - 最终从 15317 候选中筛选出 8498 道高质量 QA

3. **两类 OCR 噪声定义与扰动**：

    - **Semantic Noise（语义噪声）**：OCR 预测错误导致内容语义改变
        - 通过对文档图像添加扰动（背景噪声、水印、膨胀/腐蚀等 8 种），再用 MinerU/GOT/Qwen2.5-VL 重新 OCR
        - 迭代交叉验证确保扰动逼真：一人加扰动，另一人盲评是否能区分
        - 生成三级（mild/moderate/severe）扰动数据
    - **Formatting Noise（格式噪声）**：样式命令（空格、粗体斜体）和格式不一致（MD/LaTeX/HTML 互转）
        - 基于启发式规则，通过添加/删除/转换格式命令引入控制性扰动
        - 同样生成三级扰动数据

4. **噪声程度量化**：

    - 传统编辑距离（edit distance）不能准确反映 OCR 对 RAG 的实际影响
    - 定义 $r_{noise}$：受 OCR 噪声影响的 QA 对比例（LCS < 0.95 视为受影响）

### 损失函数 / 训练策略

本文是 benchmark，不涉及模型训练。评估指标包括：
- OCR 质量：编辑距离
- 检索性能：LCS（最长公共子序列）
- 生成性能：F1 分数
- 检索器：BGE-M3（稠密）+ BM25（稀疏）
- LLM：Qwen2-7B、Qwen2-72B、Llama-3.1-8B

## 实验关键数据

### 主实验（各 OCR 方案对 RAG 的影响）

| OCR 方案 | 编辑距离↓ | 检索 ALL↑ | 生成 ALL↑ | 端到端 ALL↑ |
|---------|---------|----------|----------|-----------|
| Ground Truth | - | 70.0 | 43.9 | 36.1 |
| Qwen2.5-VL-72B (VLM) | **0.18** | **59.2** | **37.5** | **31.1** |
| Marker (Pipeline) | 0.28 | 56.6 | 35.9 | 29.5 |
| MinerU (Pipeline) | 0.24 | 50.1 | 36.7 | 30.0 |
| InternVL2.5-78B (VLM) | 0.28 | 55.8 | 35.8 | 29.6 |
| GOT (End-to-end) | 0.27 | 45.4 | 27.8 | 24.6 |
| Nougat (End-to-end) | 0.34 | 40.9 | 25.5 | 14.5 |

### 消融实验（语义噪声 vs 格式噪声的影响）

| 噪声类型 | 噪声程度 | 对检索的影响 | 对生成的影响 | 总结 |
|---------|---------|-----------|-----------|------|
| Semantic Noise | mild→severe | 线性下降约 50% | 全面下降，多模态问题更严重 | 对 RAG 各环节一致性强烈影响 |
| Formatting Noise | mild→severe | 多模态问题下降 12.7% | 小模型影响有限，大模型更敏感 | 主要影响表格/公式/图表相关问题 |

### 表格格式影响

| 格式 | 检索（BGE-M3）↑ | 生成（Qwen2-72B）↑ | 端到端↑ |
|------|---------------|-------------------|--------|
| Markdown | **最优** | 较差（不支持合并单元格）| 最优（BGE+Qwen2-72B）|
| LaTeX | 中等 | 与 HTML 相当 | 中等 |
| HTML | 最差 | 与 LaTeX 相当 | 最差 |

### 关键发现

- **最佳 OCR 仍有 14% 性能差距**：即使 Qwen2.5-VL-72B 表现最好，端到端 F1 仍比 GT 低 5 分（31.1 vs 36.1）
- **编辑距离不是好的 OCR-for-RAG 指标**：MinerU 编辑距离比 Marker 低但 RAG 检索性能更差
- **语义噪声是主要威胁**：影响持续且一致，稠密/稀疏检索器均不免疫
- **格式噪声影响因模型而异**：小 LLM 反而不受影响（因为信息整合能力本身就差），大 LLM 更敏感
- **阅读顺序是所有方案的短板**：VLM 和端到端 OCR 在阅读顺序上仅得约 5 分，pipeline OCR 靠规则策略稍好
- **图表解析全面失败**：除 VLM 外，其他 OCR 方案几乎无法解析图表

## 亮点与洞察

- **系统性揭示了一个被忽视的关键问题**：RAG 研究社区关注检索器和 LLM 的改进，却忽略了"垃圾进→垃圾出"的知识库质量问题
- **两类噪声的分离分析**有重要实际价值：语义噪声需要提升 OCR 本身准确性，格式噪声则需要标准化的输出格式或噪声鲁棒的检索器
- **扰动数据的生成方法论**（迭代交叉验证确保逼真度）可以推广到其他数据增强场景
- **表格格式影响的发现**直接指导实际部署：BGE-M3 更理解 Markdown，LLM 更擅长 HTML/LaTeX 的合并单元格

## 局限与展望

- 仅覆盖中英文两种语言，未包含阿拉伯语、日语等 OCR 更具挑战的语言
- 扰动方法基于图像层面，未考虑 PDF 原生文本提取（非 OCR 路径）的噪声
- 检索只用了 top-2 chunk，未评估不同 top-k 设置的影响
- 未评估最新的多模态 RAG 方案（如 VisRAG 直接用图像检索跳过 OCR）

## 相关工作与启发

- 对 RAG 系统构建者："OCR 质量是你的 RAG 系统的天花板"——在优化检索器和 LLM 之前，先确保知识库质量
- 对 OCR 研究者：需要开发"RAG-aware"的 OCR 指标，编辑距离不够
- 两类噪声的框架可以推广到其他管道式 AI 系统的错误传播分析

## 评分

- 新颖性：⭐⭐⭐⭐ （首次系统评估 OCR 对 RAG 的级联影响，填补重要空白）
- 技术深度：⭐⭐⭐⭐ （噪声建模+三级评估+扰动生成方法论严谨）
- 实验充分度：⭐⭐⭐⭐⭐ （6 种 OCR 方案 × 2 检索器 × 3 LLM × 多级噪声 × 7 领域）
- 实用价值：⭐⭐⭐⭐⭐ （直接指导 RAG 系统的 OCR 选型和噪声处理策略）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] HoH: A Dynamic Benchmark for Evaluating the Impact of Outdated Information on Retrieval-Augmented Generation](../../ACL2025/information_retrieval/hoh_a_dynamic_benchmark_for_evaluating_the_impact_of_outdated_information_on_ret.md)
- [\[ACL 2025\] MAIN-RAG: Multi-Agent Filtering Retrieval-Augmented Generation](../../ACL2025/information_retrieval/main-rag_multi-agent_filtering_retrieval-augmented_generation.md)
- [\[ACL 2025\] GeAR: Generation Augmented Retrieval](../../ACL2025/information_retrieval/gear_generation_augmented_retrieval.md)
- [\[ACL 2025\] Typed-RAG: Type-Aware Decomposition of Non-Factoid Questions for Retrieval-Augmented Generation](../../ACL2025/information_retrieval/typed-rag_type-aware_decomposition_of_non-factoid_questions_for_retrieval-augmen.md)
- [\[ACL 2025\] HASH-RAG: Bridging Deep Hashing with Retriever for Efficient, Fine Retrieval and Augmented Generation](../../ACL2025/information_retrieval/hash-rag_bridging_deep_hashing_with_retriever_for_efficient_fine_retrieval_and_a.md)

</div>

<!-- RELATED:END -->
