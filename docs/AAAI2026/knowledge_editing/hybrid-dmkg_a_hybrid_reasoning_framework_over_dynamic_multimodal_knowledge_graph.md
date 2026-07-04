---
title: >-
  [论文解读] Hybrid-DMKG: A Hybrid Reasoning Framework over Dynamic Multimodal Knowledge Graphs for Multimodal Multihop QA with Knowledge Editing
description: >-
  [AAAI2026][知识编辑][多模态知识编辑] 提出MMQAKE基准和Hybrid-DMKG框架，在动态多模态知识图谱上构建"关系链接预测 + RAG增强LVLM推理"双通道混合推理机制，配合背景反思决策模块，在2-5跳多模态知识编辑问答中显著超越现有方法（LLaVA上H-Acc达29.90%，超IKE 13.52个百分点）。
tags:
  - "AAAI2026"
  - "知识编辑"
  - "多模态知识编辑"
  - "多跳问答"
  - "动态知识图谱"
  - "跨模态检索"
  - "RAG"
  - "混合推理"
---

# Hybrid-DMKG: A Hybrid Reasoning Framework over Dynamic Multimodal Knowledge Graphs for Multimodal Multihop QA with Knowledge Editing

**会议**: AAAI2026  
**arXiv**: [2512.00881](https://arxiv.org/abs/2512.00881)  
**作者**: Li Yuan, Qingfei Huang, Bingshan Zhu, Yi Cai, Qingbao Huang, Changmeng Zheng, Zikun Deng, Tao Wang (SCUT等)  
**代码**: [YuanLi95/Hybrid-DMKG](https://github.com/YuanLi95/Hybrid-DMKG)  
**领域**: 知识编辑  
**关键词**: 多模态知识编辑, 多跳问答, 动态知识图谱, 跨模态检索, RAG, 混合推理  

## 一句话总结

提出MMQAKE基准和Hybrid-DMKG框架，在动态多模态知识图谱上构建"关系链接预测 + RAG增强LVLM推理"双通道混合推理机制，配合背景反思决策模块，在2-5跳多模态知识编辑问答中显著超越现有方法（LLaVA上H-Acc达29.90%，超IKE 13.52个百分点）。

## 背景与动机

### 知识编辑面临多模态挑战

大语言模型编码的知识可能过时或错误，知识编辑（Knowledge Editing）旨在修正这些知识而不影响无关内容。随着LVLM的发展，多模态知识编辑（MKE）将编辑扩展到文本+视觉，但现有MKE基准（如VLKEB）主要评估最终答案正确性，忽略了多跳推理过程的质量和对视觉输入变化的鲁棒性。

### 现有评估的三大局限

(1) **缺乏中间推理步骤评估**：模型可能通过错误推理路径得到正确最终答案，仅评估最终答案会掩盖推理错误；(2) **缺乏视觉重述鲁棒性评估**：同一实体的不同图片应得到一致结果，但现有基准未测试；(3) **忽略答案别名多样性**："Buenos Ayres"与"Buenos Aires"语义等价但未被识别为正确。这些局限使得MKE方法的真实推理能力被高估。

### 多跳推理对知识编辑的特殊挑战

当知识链中的第一个事实被编辑后（如将人名从Roy Bittan改为Gustavo Santaolalla），模型需要正确传播修改信息并在后续推理步骤中使用更新后的知识。这要求模型不仅能编辑单一事实，还能在整个推理链上保持一致性——这对现有parameter-update和parameter-retention方法都是巨大挑战。

## 核心问题

如何在多模态多跳问答中，使模型在知识编辑后能够在每个推理步骤上正确使用更新知识，同时对视觉输入的变化保持鲁棒？

## 方法详解

### 整体框架

Hybrid-DMKG是一个无需修改模型参数的框架，包含四个核心组件：(a) 动态多模态知识图谱（DMKG）构建与维护；(b) LLM问题分解；(c) 跨模态实体检索；(d) DMKG引导的混合推理。

### DMKG构建与更新

多模态知识图谱 $\mathcal{G}$ 中每条记录表示为 $(\mathcal{G}_i^e, \mathcal{G}_i^r, \mathcal{G}_i^o)$，部分实体关联图像 $\mathcal{G}_i^v$。当接收编辑四元组 $(x, v, o, \tilde{o})$ 时，将其整合到 $\mathcal{G}$ 生成动态图 $\tilde{\mathcal{G}}$，同时保留原始和编辑后的事实。

### 问题分解

利用LLM（无需fine-tuning）将多跳问题 $Q$ 分解为子问题序列：
$$\{q_1, q_2, \ldots, q_n\} = \text{LLM}(Q, P_{\text{Dec}})$$
视觉子问题使用[IMAGE]占位符，相关实体用[ENT]标记以维持一致性。

### 跨模态实体检索

对视觉子问题，使用跨模态检索模型 $\text{M}_u$ 联合编码实体名称和图像：
$$z_m = \text{M}_u([\tilde{\mathcal{G}}_m^e, \tilde{\mathcal{G}}_m^v])$$
$$s = \text{M}_u([q_1, \tilde{v}])$$
通过余弦相似度Top-1检索最相关实体作为答案：$a_1 = \arg\text{Top1}_{m} \frac{s^T z_m}{\|s\|_2 \|z_m\|_2}$

### DMKG引导的混合推理

对推理型子问题，采用双通道并行推理：

**通道1: 关系链接预测** — 使用fine-tuned关系提取器 $\text{M}_e$ 从查询中提取关系关键词 $k_2^q$，与DMKG中候选关系计算Sense2Vec嵌入的余弦相似度，超过阈值 $\alpha$ 则选择对应实体作为答案 $a_2^{\text{link}}$。

**通道2: RAG增强LVLM推理** — 从DMKG检索Top-K相关三元组作为上下文，输入LVLM生成答案：
$$a_2^{\text{model}} = \text{LVLM}(q_2, \tilde{v}, \mathcal{K}_{\text{Ret}}(q_2, C_2), P_{\text{Ans}})$$

**背景反思决策** — 当两通道答案不一致时，从DMKG提取两个候选答案的背景知识，由LVLM综合评估选择最可信答案：
$$a_2 = \text{LVLM}(q_2, \tilde{v}, [a_2^{\text{link}}, C_2^{\text{link*}}], [a_2^{\text{model}}, C_2^{\text{modal*}}], P_{\text{Cho}})$$

## 实验关键数据

### MMQAKE基准统计

| 指标 | 数值 |
|------|------|
| 知识编辑数 | 1,278 |
| 2-hop问题数 | 1,278 |
| 3-hop问题数 | 1,238 |
| 4-hop问题数 | 1,193 |
| 5-hop问题数 | 1,110 |
| 子问题总数 | 11,773 |
| 平均答案别名数 | 9.49 |

### 主实验 (Original Image)

| 方法 | BLIP-2 M-Acc | BLIP-2 H-Acc | LLaVA M-Acc | LLaVA H-Acc | MiniGPT-4 M-Acc | MiniGPT-4 H-Acc |
|------|-------------|-------------|------------|------------|----------------|----------------|
| FT(QFor) | 3.73 | 0.20 | 4.63 | 0.44 | 4.69 | 0.44 |
| MEND | 0.04 | 0.00 | 0.70 | 0.00 | 0.07 | 0.00 |
| SERAC | 5.75 | 0.00 | 6.58 | 0.00 | 0.27 | 0.00 |
| IKE | 16.64 | 6.16 | 38.93 | 16.38 | 15.48 | 6.14 |
| **Hybrid-DMKG** | **47.55** | **28.88** | **53.75** | **29.90** | **35.86** | **24.73** |

### 消融实验 (LLaVA, Original Image)

| 变体 | M-Acc | H-Acc |
|------|-------|-------|
| Hybrid-DMKG (full) | **53.75** | **29.90** |
| w/o Linking | 47.68 | 23.15 |
| w/o Decision | 52.71 | 28.36 |

### 不同Hop数的H-Acc表现 (LLaVA)

在4-hop和5-hop的H-Acc上，Hybrid-DMKG达到约5%以上准确率，而其他方法通常低于2%，差距近乎翻倍。

## 亮点

- **首个多模态多跳知识编辑基准MMQAKE**：支持2-5跳推理链、逐步评估、视觉重述鲁棒性测试和答案别名匹配，填补了MKE评估的空白
- **双通道混合推理设计**：关系链接预测擅长处理DMKG中有明确关系的查询，RAG增强推理弥补背景知识不完整的情况，两者互补提升鲁棒性
- **背景反思决策模块**：当双通道给出不同答案时，利用DMKG中的邻域背景知识让LVLM进行"反思式"决策，有效过滤错误候选
- **参数无修改框架**：不需要修改LVLM参数，通过外部知识图谱实现知识更新，避免了catastrophic forgetting问题
- **大幅超越基线**：在LLaVA上H-Acc比最强基线IKE高出13.52个百分点（29.90% vs 16.38%）

## 局限与展望

- **依赖外部组件较多**：需要LLM问题分解、CLIP检索模型、关系提取器、Wiki Linker等多个外部模块，系统复杂度高，任一模块失败可能导致级联错误
- **H-Acc绝对值仍然较低**：即使是最好的结果（29.90%），说明多模态多跳推理仍远未解决，尤其在5-hop时H-Acc仅约5%
- **未支持开放式问答**：MMQAKE仅覆盖事实型QA，未涉及开放式或生成式问答场景
- **DMKG规模受限**：当前实验中知识图谱包含约5.8万实体和68.6万三元组，在更大规模KG上的效率和准确性有待验证

## 与相关工作的对比

- **vs IKE**: IKE基于检索增强的in-context learning，能维持较稳定的baseline但在多跳传播上能力不足，Hybrid-DMKG通过结构化KG遍历实现显式推理链
- **vs MEND/SERAC**: 参数修改类方法在多跳推理上几乎完全失败（H-Acc ≈ 0%），说明单跳编辑能力无法泛化到多跳场景
- **vs MQUAKE**: MMQAKE将纯文本多跳知识编辑评估扩展到多模态场景，增加了视觉重述和逐步评估维度

## 启发与关联

- 知识图谱作为外部知识存储的范式在知识编辑场景中展现独特优势——可精确定位并修改特定三元组
- 双通道推理+反思决策的设计模式可推广到其他需要多源证据融合的推理任务
- MMQAKE的逐步评估协议为未来多跳推理研究提供了更严格的评估标准

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首个多模态多跳知识编辑基准+DMKG混合推理框架
- 实验充分度: ⭐⭐⭐⭐ — 多个backbone、消融实验、hop-wise分析、无别名对照实验
- 写作质量: ⭐⭐⭐⭐ — 问题定义清晰，Figure 1/2展示直观
- 价值: ⭐⭐⭐⭐ — 问题重要且提出了可用的benchmark，但H-Acc绝对值仍有较大提升空间

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] KORE: Enhancing Knowledge Injection for Large Multimodal Models via Knowledge-Oriented Controls](../../ICML2026/knowledge_editing/kore_enhancing_knowledge_injection_for_large_multimodal_models_via_knowledge-ori.md)
- [\[ICLR 2026\] When Large Multimodal Models Confront Evolving Knowledge: Challenges and Explorations](../../ICLR2026/knowledge_editing/when_large_multimodal_models_confront_evolving_knowledge_challenges_and_explorat.md)
- [\[ACL 2025\] A General Knowledge Injection Framework for ICD Coding](../../ACL2025/knowledge_editing/a_general_knowledge_injection_framework_for_icd_coding.md)
- [\[ACL 2025\] Memorizing is Not Enough: Deep Knowledge Injection Through Reasoning](../../ACL2025/knowledge_editing/memorizing_is_not_enough_deep_knowledge_injection_through_reasoning.md)
- [\[NeurIPS 2025\] KScope: A Framework for Characterizing the Knowledge Status of Language Models](../../NeurIPS2025/knowledge_editing/kscope_a_framework_for_characterizing_the_knowledge_status_of_language_models.md)

</div>

<!-- RELATED:END -->
