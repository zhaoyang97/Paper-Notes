---
title: >-
  [论文解读] Pharos-ESG: A Framework for Multimodal Parsing, Contextual Narration, and Hierarchical Labeling of ESG Reports
description: >-
  [AAAI 2026][多模态VLM][ESG报告解析] 本文提出Pharos-ESG框架，通过基于版面流的阅读顺序建模、目录锚点引导的层次结构重建、上下文感知的多模态图像描述转换、以及多级金融标签预测四个核心模块，实现对ESG报告的结构化解析，在全面评估中F1达93.59、ROKT达0.92、TBTA达92.46%，显著超越MinerU、GPT-4o、Gemini 2.5 Pro等基线，并发布了首个大规模公开ESG报告数据集Aurora-ESG（24K+报告）。
tags:
  - "AAAI 2026"
  - "多模态VLM"
  - "ESG报告解析"
  - "文档理解"
  - "阅读顺序建模"
  - "层次结构重建"
  - "多级标签预测"
---

# Pharos-ESG: A Framework for Multimodal Parsing, Contextual Narration, and Hierarchical Labeling of ESG Reports

**会议**: AAAI 2026  
**arXiv**: [2511.16417](https://arxiv.org/abs/2511.16417)  
**代码**: [https://github.com/liucun-zy/Pharos-ESG](https://github.com/liucun-zy/Pharos-ESG)  
**领域**: 多模态VLM  
**关键词**: ESG报告解析, 文档理解, 阅读顺序建模, 层次结构重建, 多级标签预测

## 一句话总结
本文提出Pharos-ESG框架，通过基于版面流的阅读顺序建模、目录锚点引导的层次结构重建、上下文感知的多模态图像描述转换、以及多级金融标签预测四个核心模块，实现对ESG报告的结构化解析，在全面评估中F1达93.59、ROKT达0.92、TBTA达92.46%，显著超越MinerU、GPT-4o、Gemini 2.5 Pro等基线，并发布了首个大规模公开ESG报告数据集Aurora-ESG（24K+报告）。

## 研究背景与动机

**领域现状**：ESG（环境、社会与治理）原则正在重塑全球金融治理，从自愿披露转向强制披露，成为连接企业、投资者和监管机构的关键基础设施。ESG报告是评估企业ESG表现的核心媒介。

**核心痛点**：ESG报告通常以视觉密集的长PDF形式发布，存在两个关键技术挑战：

**混乱的阅读顺序**：布局高度异质化，文本、表格、图表交错排列，常呈类幻灯片格式，即使在看似结构化的目录部分也存在不一致

**隐式的层次结构**：报告通常超过50页，缺少编号标题或一致的格式化等标准化结构指示器，难以恢复层次组织

**为何现有方法不够**：
- **间接代理**：金融研究被迫使用简单的披露指标、小规模案例研究或第三方评级，绕过了报告本身丰富的语义
- **通用文档解析器**（MinerU、Docling、Marker）：针对学术论文、法律合同等结构规则的格式设计，在ESG报告的不规则布局上性能严重下降
- **通用多模态模型**（GPT-4o、Gemini）：容易产生幻觉，在重建长篇弱结构文档的隐式层次时计算成本高昂

**本文切入角度**：设计一个专门针对ESG报告特点的统一框架，从阅读顺序→层次结构→多模态内容→金融标签四个维度完成结构化解析。同时发布首个大规模公开ESG报告数据集，填补数据资源空白。

## 方法详解

### 整体框架

Pharos-ESG包含四个核心模块：(a) 基于版面流的阅读顺序建模；(b) 目录（ToC）锚点引导的结构重建；(c) 视觉元素到自然语言的上下文转换；(d) ESG/GRI/情感多级标签标注。

### 关键设计

1. **阅读顺序建模（Reading Order Modeling）**

    - **功能**：从ESG报告中提取多模态元素后，将全局序列排序转化为成对后继分类问题
    - **Block内容编码**：每页表示为内容块集合 $\mathcal{D}_p = \{(w_i, b_i, c_i, p)\}$，包含内容、边界框、块类型、页码
    - **多模态特征构建**：对每对有序Block $(i,j)$，构建特征向量 $\varphi_{ij}$，整合语义（LayoutLMv3编码）、空间（中心偏移、IoU、距离）和类别信号
    - **关系预测**：关系感知Transformer（RAT）通过交叉注意力预测Block $j$ 是否直接跟随Block $i$，计算后继概率 $s_{ij} = \sigma(\mathbf{W} \cdot \text{Transformer}(\varphi_{ij}) + \mathbf{b})$
    - **拓扑排序**：构建有向加权图，通过拓扑排序获得全局一致的无环阅读序列
    - **设计动机**：ESG报告的类幻灯片布局使得传统从上到下、从左到右的规则失效，需要学习型方法建模复杂阅读流

2. **ToC引导的层次结构重建**

    - **RAP（Region-Aware Prompting）**：一种视觉提示策略，利用颜色相似性、空间邻近性和文本邻接性引导MLLM推断隐式层次。包含四个组件：跨区域条目聚合、上下文感知标签丰富、基于区域的层次推断、多行合并
    - **ALIGN（Anchor-based Linguistic Indexing for Granular Navigation）**：多阶段对齐算法，三个阶段：
        - 精确匹配：字符级匹配识别高置信度锚点
        - 模糊/包含匹配：Levenshtein相似度和子串包含扩展覆盖
        - 上下文感知插入：LLM推理解决未匹配标题（摘要段落→评估是否缺少概述标题→识别改善位置→选择最优插入点）
    - **设计动机**：ESG报告的ToC格式多样且层次隐式，传统基于字体大小、缩进等视觉线索的方法不可靠

3. **图像到文本转换**

    - **层次引导的多模态聚合**：将目标图像与其周围内容整合为连贯的多模态簇，由附近章节标题引导，保持阅读顺序
    - **上下文化图像描述生成**：两阶段——结构化语义建模（编码层次路径、元素声明）+ 多模态嵌入及语义生成（ViT提取视觉特征 → Q-Former投影 → Qwen2.5-VL-Instruct生成描述）
    - **设计动机**：ESG报告中的图表、图像需要结合上下文才能准确理解，孤立描述会丢失关键语境

4. **MLPDH多级金融标签预测**

    - **三元嵌入**：文本语义（Chinese-RoBERTa-wwm-ext的[CLS] token）+ 层次上下文（GRU编码标题路径）+ 全局阅读顺序位置
    - **层次注意力**：堆叠注意力传播层次信号 $\mathbf{v}^{(h)}_{blk} = \text{softmax}(\frac{(\mathbf{W}_q \mathbf{e}_{blk})^\top (\mathbf{W}_k \mathbf{v}^{(h-1)}_{ref})}{\sqrt{d}}) \cdot \mathbf{W}_v \mathbf{v}^{(h-1)}_{ref}$
    - **层次一致性约束**：父子标签依赖惩罚 + BCE损失联合优化
    - **标签结构**：ESGN类别 → GRI指标 → 情感标签三级预测
    - **设计动机**：ESG分析需要多维度标注，层次结构信息与金融标签密切相关

## 实验关键数据

### 主实验

| 方法 | 类型 | 解析F1↑ | ROKT↑ | TBTA(%)↑ |
|------|------|---------|-------|----------|
| Marker | 专用解析器 | 39.88 | 0.34 | 3.79 |
| MinerU | 专用解析器 | 76.89 | 0.82 | 6.94 |
| Textin | 专用解析器 | 82.55 | 0.80 | 9.68 |
| GPT-4o | 通用多模态 | 65.17 | 0.75 | 43.55 |
| Gemini 2.5 Pro | 通用多模态 | 87.50 | 0.75 | 64.30 |
| **Pharos-ESG** | **本文** | **93.59** | **0.92** | **92.46** |

### 消融实验

| 配置 | 组件 | F1↑ | 说明 |
|------|------|-----|------|
| Config 1 | 无模块 | 76.95 | 与通用解析器水平相当 |
| Config 2 | +GP-based ToC | 78.79 | +1.84，通用提示效果有限 |
| Config 3 | +RAP替代GP | 83.57 | +4.78，RAP显著优于通用提示 |
| Config 4 | +阅读顺序建模 | 88.14 | +4.57，阅读顺序对召回率提升大 |
| Config 5 | +ALIGN前两步 | 90.05 | +1.91，精确/模糊匹配提升 |
| Config 6 | +完整ALIGN | **93.59** | +3.54，上下文感知插入补全未匹配标题 |

**多级标签预测**：

| 方法 | ESGN F1 | GRI F1 | 情感F1 | Macro-F1 | HLA |
|------|---------|--------|--------|----------|-----|
| SVM+TF-IDF | 72.14 | 61.59 | 68.31 | 67.35 | - |
| BERT-base | 80.21 | 72.30 | 77.61 | 76.71 | 81.31 |
| HMCN | 82.70 | 76.86 | 79.07 | 79.54 | 88.15 |
| **MLPDH** | **85.62** | **84.23** | **89.11** | **86.32** | **94.78** |

**跨市场泛化**：

| 市场 | 解析F1 | ROKT | TBTA | Macro-F1 |
|------|--------|------|------|----------|
| 中国A股 | 92.04 | 0.92 | 92.46 | 86.32 |
| 香港 | 89.05 | 0.88 | 89.50 | 87.20 |
| 美国 | **94.30** | **0.94** | **94.80** | **87.60** |

### 关键发现

1. **专用解析器在TBTA任务上几乎完全失败**（< 20%），因为依赖字体大小、缩进等启发式特征在ESG报告中不可靠
2. **RAP策略提升显著**：相比通用提示GP，RAP在CC/RC/HC上平均提升+9.89%/+12.02%/+14.51%
3. **阅读顺序建模与结构提取强相关**：ROKT与解析指标呈正相关
4. **美国市场最好解析**：因格式更标准化、层次更清晰、分块更一致
5. **MLPDH的三元嵌入和层次约束是关键**：BERT-base的HLA仅81.31，缺少跨层约束导致父子标签不一致

## 亮点与洞察

1. **面向垂直领域的端到端文档理解**：不是通用解析器，而是针对ESG报告特点定制化设计，这种思路对其他专业文档也有参考价值
2. **RAP视觉提示策略**：巧妙利用颜色/空间/文本邻接性引导MLLM理解隐式层次，可推广到其他弱结构化文档
3. **ALIGN三阶段对齐算法**：从精确匹配到模糊匹配到LLM推理，渐进式处理对齐难度递增的情况
4. **Aurora-ESG数据集贡献**：24K+报告、800万+内容块、覆盖中美港三个市场，填补了领域数据空白
5. **完整的金融标注体系**：ESGN→GRI→情感三级标签，直接服务于金融分析需求

## 局限与展望

1. **评估主要聚焦中文ESG报告**：虽然测试了跨市场泛化，但50份中文报告的训练集较小
2. **计算资源需求**：完整pipeline涉及多个模型（LayoutLMv3、Qwen2.5-VL等），部署成本较高
3. **ALIGN的LLM依赖**：上下文感知插入阶段依赖LLM进行推理，可能在复杂情况下产生错误
4. **数据集质量**：Aurora-ESG基于框架自动生成，可能包含系统性偏差
5. **可扩展性**：对于非ESG类型的长文档（如年报、法律文件），方法的迁移效果未验证

## 相关工作与启发

- **文档解析领域**：从PubLayNet到DocLayNet，布局分析从规则文档扩展到不规则文档，ESG报告是新的挑战前沿
- **LayoutLMv3等跨模态模型**：在语义对齐上有效，但在ESG报告中位置分散的相关元素同步理解上受限
- **长文档理解**：注意力衰减问题（超20页后识别失败）在ESG报告中被放大
- 启发：垂直领域文档理解可能需要定制化的"领域知识注入"策略，通用multi-modal大模型并不能解决所有问题
- ToC作为文档理解的"锚点"是一个值得推广的思路

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Multimodal DeepResearcher: Generating Text-Chart Interleaved Reports From Scratch with Agentic Framework](multimodal_deepresearcher_generating_text-chart_interleaved_.md)
- [\[ACL 2026\] SlideAgent: Hierarchical Agentic Framework for Multi-Page Visual Document Understanding](../../ACL2026/multimodal_vlm/slideagent_hierarchical_agentic_framework_for_multi-page_visual_document_underst.md)
- [\[AAAI 2026\] Plug-and-Play Clarifier: A Zero-Shot Multimodal Framework for Egocentric Intent Disambiguation](plug-and-play_clarifier_a_zero-shot_multimodal_framework_for_egocentric_intent_d.md)
- [\[ICML 2026\] FlowNar: Scalable Streaming Narration for Long-Form Videos](../../ICML2026/multimodal_vlm/flownar_scalable_streaming_narration_for_long-form_videos.md)
- [\[AAAI 2026\] PlantTraitNet: An Uncertainty-Aware Multimodal Framework for Global-Scale Plant Trait Inference from Citizen Science Data](planttraitnet_an_uncertainty-aware_multimodal_framework_for_global-scale_plant_t.md)

</div>

<!-- RELATED:END -->
