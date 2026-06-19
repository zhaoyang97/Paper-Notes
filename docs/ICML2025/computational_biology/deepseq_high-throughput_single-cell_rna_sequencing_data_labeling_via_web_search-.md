---
title: >-
  [论文解读] DeepSeq: High-Throughput Single-Cell RNA Sequencing Data Labeling via Web Search-Augmented Agentic Generative AI Foundation Models
description: >-
  [ICML 2025][计算生物][单细胞RNA测序] 提出 DeepSeq 流水线，利用大语言模型（尤其是具备实时网络搜索能力的 Agentic GPT-4o）对单细胞RNA测序数据进行自动化细胞类型标注，最高准确率达 82.5%，解决了大规模组学数据标注的吞吐量瓶颈。 单细胞RNA测序的规模化挑战 单细胞RNA测序（sc…
tags:
  - "ICML 2025"
  - "计算生物"
  - "单细胞RNA测序"
  - "大语言模型"
  - "细胞类型标注"
  - "智能体 AI"
  - "基础模型"
---

# DeepSeq: High-Throughput Single-Cell RNA Sequencing Data Labeling via Web Search-Augmented Agentic Generative AI Foundation Models

**会议**: ICML 2025  
**arXiv**: [2506.13817](https://arxiv.org/abs/2506.13817)  
**代码**: [有](https://github.com/saleemaldajani/deepseq)  
**领域**: 医学影像/生物信息学  
**关键词**: 单细胞RNA测序, 大语言模型, 细胞类型标注, 智能体 AI, 基础模型

## 一句话总结

提出 DeepSeq 流水线，利用大语言模型（尤其是具备实时网络搜索能力的 Agentic GPT-4o）对单细胞RNA测序数据进行自动化细胞类型标注，最高准确率达 82.5%，解决了大规模组学数据标注的吞吐量瓶颈。

## 研究背景与动机

### 单细胞RNA测序的规模化挑战

单细胞RNA测序（scRNA-seq）已彻底改变了我们在细胞分辨率水平理解生物系统的能力。与传统的 bulk 测序不同，scRNA-seq 保留了细胞多样性，支持谱系追踪、扰动推断和细胞类型识别等下游分析。

然而，随着条形码技术和实验方案的改进，scRNA-seq 数据集已从数千个细胞增长到每次实验数百万个细胞。根据 Svensson et al. (2018) 的统计，自 2009 年以来单细胞测序数量呈指数增长，预计到 2030 年单个研究将超过 $10^9$ 个细胞。

### 标注瓶颈

当前的核心瓶颈在于**细胞类型标注**：

- 手动标注速度远跟不上数据增长
- 随着聚类数量随数据量增加，人工标注复杂度急剧上升
- 监督学习、伪时间排序、扰动建模等下游任务都依赖于准确的细胞类型标签
- 人工标注不可避免地引入人为误差

### 现有方法的不足

传统的自动标注方法（如参考图谱映射）受限于参考数据的覆盖范围和跨组织泛化能力。最近 Hou & Ji (2024) 在 Nature Methods 上展示了 GPT-4 在单细胞标注中的初步能力，但缺乏系统性的流水线设计和多模型对比。

### 本文动机

作者提出：能否构建一个模块化的、可扩展的 LLM 标注系统，同时支持本地轻量推理和在线 Agentic 推理，以解决高通量标注需求？

## 方法详解

### 整体框架

DeepSeq 是一个端到端的模块化流水线，整体流程为：

$$\text{原始数据} \xrightarrow{\text{过滤}} \text{清洗后数据} \xrightarrow{\text{降维+聚类}} \text{细胞簇} \xrightarrow{\text{提取标志基因}} \text{结构化提示} \xrightarrow{\text{LLM推理}} \text{细胞类型标签}$$

系统支持两种推理路径：
1. **本地推理**：通过 Ollama 客户端部署轻量级模型（如 LLaMA3）进行设备端推理
2. **Agentic 推理**：通过 GPT-4o + Web Search 进行在线推理，Agent 可自主检索和总结外部生物学知识

### 关键设计

#### 1. **数据预处理与过滤模块**：去除低质量细胞和基因→三种互补的过滤策略→确保输入数据质量

原始数据被处理为基因×细胞矩阵，转换为 AnnData 格式。过滤采用三种策略：

- **标准阈值过滤**：每个细胞至少表达 $\geq 200$ 个基因
- **自动拐点检测**：使用 KneeLocator 算法自动确定过滤阈值
- **平滑拐点过滤**：基于平滑后的分布曲线检测拐点

三种策略可以生成质控诊断图，供用户选择最合适的过滤方案。

#### 2. **聚类与标志基因提取模块**：将细胞分群并提取每群的特征基因→Leiden 算法聚类 + Scanpy 基因排序→为 LLM 提供结构化输入

具体步骤：
- 使用 PCA 进行降维
- 基于邻域图，使用 Leiden 算法聚类
- UMAP 嵌入到 2D 空间进行可视化
- 对每个聚类 $C_i$，使用 Scanpy 的排序函数提取 top 标志基因 $G_i = \text{rank\_genes}(C_i)$

#### 3. **LLM 标注模块**：基于标志基因生成细胞类型预测→结构化提示 + 双推理路径→平衡效率与准确性

核心算法流程：

对于每个聚类 $C_i$：
1. 提取 top 标志基因 $G_i$
2. 构造结构化提示 $P_i = \text{format}(G_i)$
3. 根据推理模式：
    - **Ollama 路径**：$\hat{y}_i = \text{local\_LLM}(P_i)$
    - **GPT-4o 路径**：先执行网络搜索获取上下文，再 $\hat{y}_i = \text{gpt4o}(P_i, \text{web results})$

提示工程参考了 Hou & Ji (2024) 的格式设计，并适配到结构化转录组数据。LangChain 负责编排提示和后处理。

#### 4. **评估模块**：量化标注准确性→两阶段验证协议→确保可重复性

- **阶段一：标志基因验证** — 确认每个聚类的 top 标志基因与已知的典型标志基因充分匹配，确保评估在生物学上有意义
- **阶段二：标签准确性评估** — 将 LLM 生成的标签与人工标注的 ground truth 进行比较，使用模糊字符串匹配和同义词解析来鲁棒地评估聚类级别的一致性

### 损失函数 / 训练策略

本文方法不涉及模型训练，是一个推理时（inference-time）流水线。不需要损失函数或反向传播，LLM 模型直接作为"零样本"标注器使用。系统的核心在于：

- **提示工程**：精心设计的结构化提示，将标志基因信息转换为 LLM 可理解的格式
- **Agentic 增强**：通过实时网络搜索引入外部生物学知识，增强 LLM 在特定领域的推理能力
- **后处理**：模糊匹配和同义词解析来处理标签格式不一致的问题

## 实验关键数据

### 主实验

实验在标准的 scRNA-seq 数据集上进行，使用 top 标志基因作为提示输入，比较不同 LLM 的标注准确率。

| 模型 | 参数量 | 推理方式 | 标注准确率 | 特点 |
|------|--------|----------|-----------|------|
| LLaMA3-2-1B | ~1B | 本地 (Ollama) | 较低 | 轻量级，可离线部署 |
| GPT-3.5-turbo | ~175B | Agentic (Web Search) | 中等 | 有网络搜索能力 |
| GPT-4o | ~1.8T | Agentic (Web Search) | **82.5%** | 最高准确率 |

关键观察：
- GPT-4o 实现了 **82.5%** 的最高准确率
- 从 LLaMA3-2-1B 到 GPT-3.5 的提升幅度 > 从 GPT-3.5 到 GPT-4o 的提升幅度
- 这表明 Agentic 能力（网络搜索）提供了基线提升，但架构优化和参数扩展的收益递减

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 无 Web Search (LLaMA3-1B) | 明显低于有搜索的模型 | 网络搜索是关键增益来源 |
| 有 Web Search (GPT-3.5) | 比无搜索大幅提升 | Agentic 能力提供基线增益 |
| 有 Web Search + 更大模型 (GPT-4o) | 82.5% | 参数量翻倍但提升有限 |
| 三种过滤策略 | 各有优劣 | 提供多种质控选项 |

### 关键发现

1. **Agentic 能力 > 模型规模**：网络搜索带来的增益大于单纯增加参数量的增益。在没有领域特定数据的情况下，模型规模扩展的收益递减。

2. **Scaling Law 延伸到生物数据**：类似于语言模型中数据量与性能的正相关关系，细胞类型标注的准确性也依赖于实验数据的规模和多样性。

3. **轻量模型具有竞争力**：LLaMA3-1B 在考虑到其体积的情况下表现不俗，证实了在受限环境中轻量级部署的可行性。

4. **标志基因质量是瓶颈**：提示的信息量取决于每个聚类中标志基因的区分性，在生物学上模糊的情况下模型仍表现脆弱。

## 亮点与洞察

1. **系统设计理念出色**：将 LLM 与单细胞分析完整集成，形成端到端可复现的流水线，而非简单的"把基因列表丢给 ChatGPT"的概念验证。

2. **双路径推理架构**：同时支持离线本地推理和在线 Agentic 推理，适应不同的部署场景（隐私敏感环境 vs 追求最高准确率）。

3. **将 Scaling Law 类比到生物标注**：提出随着单细胞数据集规模增长到 $10^9$ 量级，LLM 标注可能超越人类水平的洞察。

4. **实用性强**：代码开源，流水线每一步都输出可解释的日志，支持不同 LLM 配置和评估策略的扩展。

5. **Web Search 作为知识增强**的有效性被量化验证 — 这为 Agentic AI 在科学领域的应用提供了数据支持。

## 局限与展望

1. **数据集单一**：仅在一个 scRNA-seq 数据集上验证，缺乏跨组织、跨物种的泛化性测试。82.5% 的准确率在不同数据集上是否稳定需要更多实验。

2. **评估粒度有限**：仅评估聚类级别的标签准确率，缺少细胞级别的准确率分析和混淆矩阵。

3. **缺乏与专用工具的对比**：未与 CellTypist、scType、SingleR 等专门的细胞注释工具进行对比，难以判断 LLM 方法是否真的优于现有专业工具。

4. **成本分析缺失**：GPT-4o 的 API 调用成本在大规模数据集上可能很高，论文未讨论经济可行性。

5. **提示敏感性未探究**：不同的提示格式、标志基因数量选择对结果的影响未做系统探究。

6. **82.5% 准确率的实际意义**：在需要高精度的临床和研究场景中，17.5% 的错误率可能仍然太高，论文对此讨论不足。

7. **未考虑多模态整合的实验**：虽然讨论了与 scATAC-seq 和空间转录组学的潜在整合，但缺乏实际实验。

## 相关工作与启发

- **Hou & Ji (2024), Nature Methods**：首先评估了 GPT-4 在单细胞标注中的能力，是本文提示设计的基础
- **Svensson et al. (2018)**：记录了单细胞测序的指数增长趋势，支撑了本文的 scaling law 讨论
- **Human Cell Atlas / Human Tumor Atlas Network**：大规模细胞图谱项目，为 DeepSeq 提供了应用场景
- **Wang et al. (2025)**：SpatialAgent，自主 AI 代理用于空间生物学，与本文的 Agentic 范式呼应

**启发**：
- Agentic AI + 领域知识检索在科学数据标注中的范式值得推广到其他组学数据（蛋白质组、代谢组）
- 将 Scaling Law 的框架用于分析生物信息学工具的性能边界是一个有趣的视角
- 混合推理架构（本地 + 云端）的设计思路可以应用到隐私敏感的医疗场景

## 评分

- 新颖性: ⭐⭐⭐⭐ 方法本身是已有技术的工程化组合（LLM + Scanpy + LangChain），核心创新在系统设计而非算法
- 实验充分度: ⭐⭐⭐ 单数据集验证，缺乏与专用工具对比，消融实验有限
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，系统架构描述详细，图表丰富
- 价值: ⭐⭐⭐⭐ 展示了 LLM 在生物数据标注中的可行性，但实际影响力受限于准确率上限和缺乏广泛验证

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] A Survey on Foundation Language Models for Single-cell Biology](../../ACL2025/computational_biology/foundation_lm_single_cell_survey.md)
- [\[ICML 2025\] scSSL-Bench: Benchmarking Self-Supervised Learning for Single-Cell Data](scssl-bench_benchmarking_self-supervised_learning_for_single-cell_data.md)
- [\[ICML 2025\] PolyConf: Unlocking Polymer Conformation Generation through Hierarchical Generative Models](polyconf_unlocking_polymer_conformation_generation_through_hierarchical_generati.md)
- [\[NeurIPS 2025\] Steering Generative Models with Experimental Data for Protein Fitness Optimization](../../NeurIPS2025/computational_biology/steering_generative_models_with_experimental_data_for_protein_fitness_optimizati.md)
- [\[NeurIPS 2025\] scMRDR: A Scalable and Flexible Framework for Unpaired Single-Cell Multi-Omics Data Integration](../../NeurIPS2025/computational_biology/scmrdr_a_scalable_and_flexible_framework_for_unpaired_single-cell_multi-omics_da.md)

</div>

<!-- RELATED:END -->
