---
title: >-
  [论文解读] MAGIC-VQA: Multimodal and Grounded Inference with Commonsense Knowledge for Visual Question Answering
description: >-
  [ACL 2025][多模态VLM][视觉问答] 提出MAGIC-VQA框架，通过三阶段流程（显式知识检索→按类型后处理→GNN隐式增强）将外部常识知识系统地注入LVLM，在ScienceQA、TextVQA、MMMU等基准上实现即插即用的常识推理增强，仅需0.33M可训练参数。 问题背景 视觉问答（VQA）要求模型同时理解…
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "视觉问答"
  - "常识知识"
  - "知识图谱"
  - "图神经网络"
  - "大型视觉语言模型"
---

# MAGIC-VQA: Multimodal and Grounded Inference with Commonsense Knowledge for Visual Question Answering

**会议**: ACL 2025  
**arXiv**: [2503.18491](https://arxiv.org/abs/2503.18491)  
**作者**: Shuo Yang, Soyeon Caren Han (University of Melbourne), Siwen Luo (UWA), Eduard Hovy  
**代码**: [adlnlp/magic_vqa](https://github.com/adlnlp/magic_vqa)  
**领域**: 多模态VLM  
**关键词**: 视觉问答, 常识知识, 知识图谱, 图神经网络, 大型视觉语言模型  

## 一句话总结

提出MAGIC-VQA框架，通过三阶段流程（显式知识检索→按类型后处理→GNN隐式增强）将外部常识知识系统地注入LVLM，在ScienceQA、TextVQA、MMMU等基准上实现即插即用的常识推理增强，仅需0.33M可训练参数。

## 研究背景与动机

### 问题背景
视觉问答（VQA）要求模型同时理解视觉和文本信息，近年来大型视觉语言模型（LVLM）如LLaVA、Qwen2VL、GPT-4o等在VQA上取得了显著进展，但在需要常识推理的问题上仍然表现不佳——例如隐含的上下文线索或日常世界知识。

### 已有方法的不足
- **多模态RAG方法**：通过密集检索注入外部信息，但检索是静态的、与输入无关的，容易引入噪声
- **提示调优方法**：依赖精心设计的prompt激发模型内在常识，但静态prompt缺乏对新场景的动态适应能力
- **图方法**：利用GNN整合结构化常识知识，但忽略了外部知识与模型内在知识之间的动态交互
- **核心缺失**：没有一个统一框架能将动态、上下文对齐的常识整合与结构化图推理相结合

### 核心动机
设计一个轻量级即插即用框架，无需大规模预训练或复杂prompt调优，就能系统性地将显式和隐式常识知识注入任意LVLM，提升常识推理能力。

## 方法详解

### 整体框架
MAGIC-VQA采用三阶段流水线架构：
1. **显式常识知识检索**：从外部知识图谱中提取与输入相关的知识三元组
2. **按类型常识后处理**：根据数据集特性过滤和分配三元组的相关性等级
3. **隐式常识知识增强**：通过GNN构建异构多模态图，生成置信度分数

最终将原始输入（图像、问题、图说）、带相关性等级的知识三元组、GNN置信度分数统一送入LVLM做推理。

### 关键设计1：显式常识知识检索

选用ATOMIC2020作为外部知识源，因其覆盖133万三元组、23种关系类型，涵盖三大类常识：
- **物理实体（PE）**：对象属性和功能，如"纸由纤维素制成"
- **事件中心（EC）**：情境序列，如"X吃早餐"通常在"X去上班"之前
- **社交互动（SI）**：人际交互与情感，如"X送礼物"导致"Y感到感激"

检索流程：给定图像$I$和问题$Q$，先用BLIP2生成图说$C$，然后将$\{I, Q, C\}$编码到共享嵌入空间，与ATOMIC2020中所有三元组的头尾实体计算余弦相似度，为每个输入源选取Top-K最相关的三元组。

### 关键设计2：按类型常识后处理

该阶段包含两步：

**按类型过滤**：根据每个数据集的需求定制常识类型比例。通过分析发现：ScienceQA需要更多PE知识（比例0.7:0.15:0.15），TextVQA更依赖EC知识（0.2:0.6:0.2），MMMU需要均衡混合（0.33:0.33:0.33）。先丢弃低于阈值$\tau$的三元组，再按目标比例从各类型中选取得分最高的$k_t = \lfloor p_t \times k \rfloor$个三元组。

**相关性等级分配**：使用动态阈值机制，基于每个数据集的余弦相似度均值$\mu_f$和标准差$\sigma_f$，将三元组分为High（$\geq \mu_f + \sigma_f/2$）、Medium和Low三个等级，帮助LVLM在推理时优先关注最有意义的知识。

### 关键设计3：GNN隐式常识增强

构建异构多模态图$G_n = \{V, E\}$：
- **节点**：图像、问题、图说各1个节点 + $k$个常识节点（由过滤后的三元组展平为自然语言句子）
- **边**：基于节点嵌入间的余弦相似度构建
- **推理**：使用两层GCN迭代更新节点嵌入（$H^{(l+1)} = \rho(\widetilde{A}H^{(l)}W_l)$），池化后通过MLP生成候选答案的置信度分数

这些置信度分数作为额外信号注入LVLM，使其能优先选择有常识支撑的答案。

## 实验关键数据

### 表1：显式常识知识的消融实验（各配置准确率%）

| 模型 | 无常识 | CS-Q | CS-I | CS-C | CS-PE | CS-EC | CS-SI | 全部CS |
|------|--------|------|------|------|-------|-------|-------|--------|
| **ScienceQA** | | | | | | | | |
| LLaVA1.6 | 67.50 | 68.83 | 71.56 | 70.35 | 71.12 | 69.01 | 70.83 | 72.30 |
| Qwen2VL | 71.39 | 72.21 | 74.83 | 71.86 | 74.22 | 72.03 | 72.57 | 75.95 |
| GPT4o-mini | 76.45 | 77.34 | 79.83 | 77.17 | 79.63 | 77.52 | 78.87 | 81.22 |
| **TextVQA** | | | | | | | | |
| Qwen2VL | 75.30 | 76.07 | 77.63 | 77.05 | 76.57 | 78.02 | 76.85 | 78.90 |
| GPT4o-mini | 78.98 | 79.34 | 81.25 | 80.63 | 80.93 | 81.51 | 81.22 | 82.13 |
| **MMMU** | | | | | | | | |
| Qwen2VL | 51.10 | 52.69 | 55.89 | 54.83 | 53.60 | 54.57 | 54.10 | 57.42 |
| GPT4o-mini | 55.89 | 56.53 | 58.79 | 56.21 | 58.12 | 57.57 | 57.89 | 60.87 |

### 表2：各组件消融（Qwen2VL / GPT4o-mini）

| 显式CS | 相关性等级 | GNN置信度 | SQA | MMMU | TextVQA |
|--------|-----------|----------|------|------|---------|
| ✗ | ✗ | ✗ | 71.39 / 76.45 | 51.10 / 55.89 | 75.30 / 78.98 |
| ✓ | ✗ | ✗ | 75.11 / 80.07 | 56.00 / 59.30 | 78.50 / 81.73 |
| ✗ | ✗ | ✓ | 72.88 / 77.02 | 53.41 / 57.64 | 76.42 / 79.55 |
| ✓ | ✓ | ✗ | 75.95 / 81.22 | 57.42 / 60.87 | 78.90 / 82.13 |
| ✓ | ✓ | ✓ | **77.12 / 82.50** | **58.72 / 61.03** | **79.80 / 83.37** |

### 表3：与知识增强基线对比

| 模型 | 知识源 | A-OKVQA | VCR |
|------|--------|---------|-----|
| VLC-BERT | COMET | 38.05 | 79.24 |
| KAT | Wikidata+GPT3 | 49.74 | 83.18 |
| KRISP | Wikidata+CNet | 27.10 | 65.12 |
| MAGIC-VQA (GPT4o-mini) | ATOMIC2020 | **76.55** | **93.42** |

## 关键发现

- **图像驱动常识贡献最大**：CS-I（图像相关常识）在所有模型和数据集上提升最显著，例如LLaVA1.6在MMMU上从48.38%跳至53.52%，说明与图像对齐的常识能提供更接地的推理线索
- **常识类型需按数据集定制**：ScienceQA从PE知识获益最多（科学概念），TextVQA从EC知识获益最多（上下文理解），MMMU则需要均衡分布
- **显式+隐式互补**：单独使用GNN置信度也能带来提升（MMMU从51.10%→53.41%），但与显式知识结合后效果最佳（58.72%），说明两者捕获了不同维度的常识信息
- **具象物体比抽象概念更易受益**：TextVQA中"uniform""books"等具象类别的提升比"persons"等抽象类别更大
- **简单题获益大于难题**：MMMU中easy级别问题提升明显，hard级别问题需要超越常识的复杂推理

## 亮点与洞察

- **极致轻量**：整个框架仅0.33M可训练参数（GNN部分），相比LLaVA的7B或GPT-4的175B+，参数量降低了数万倍，这使得快速适配到新的LVLM成为可能
- **即插即用设计**：不修改LVLM本身，不需要微调或预训练，通过外部知识检索+GNN生成的置信度分数组装输入prompt即可，架构上解耦了知识获取与模型容量
- **按类型过滤的insight**：不同任务对常识类型的需求差异很大，暴力注入所有知识反而可能引入噪声，按数据集固有分布定制过滤比例是关键
- **相关性等级的soft signal**：用High/Medium/Low标注而非直接截断，让LVLM自行判断知识可靠性，是一种优雅的不确定性传递机制

## 局限性

- **依赖固定知识图谱**：使用ATOMIC2020作为唯一外部知识源，其覆盖面有局限，遇到未收录的专业领域知识可能失效
- **预定义常识类别的局限**：PE/EC/SI三类划分较粗，可能无法精确匹配所有VQA场景的知识需求
- **实验多集中于基准选择题**：ScienceQA和MMMU为选择题格式，TextVQA用验证集，缺乏在开放式VQA或更复杂场景上的验证
- **常识类型比例需人工设定**：虽然提出了按类型过滤，但最优比例需要通过实验搜索或GPT-4分析确定，自动化程度有限
- **GNN的贡献相对显式知识较小**：消融实验显示GNN置信度单独使用提升有限（1-2个点），性价比值得商榷

## 相关工作与启发

- **VLC-BERT (Ravi et al. 2023)**：将常识知识编码为额外文本特征微调VL-BERT，但需要修改模型架构；MAGIC-VQA的即插即用设计更灵活
- **MM-CoT / KAM-CoT**：在CoT数据上微调模型注入常识，但需要大量训练数据和计算资源；MAGIC-VQA的零样本方式更高效
- **VQA-GNN (Wang et al. 2022)**：用GNN做多模态语义图推理，但未与LVLM结合；MAGIC-VQA将GNN作为LVLM的辅助信号更具扩展性
- **启发**：将结构化知识以"soft prompt"方式注入LVLM是一个值得探索的范式——既保持了LVLM的泛化能力，又补充了其缺失的具体知识

## 评分

- 新颖性: ⭐⭐⭐⭐ — 三阶段框架设计系统性强，按类型过滤和相关性等级分配机制有新意
- 实验充分度: ⭐⭐⭐⭐ — 覆盖5个LVLM、5个数据集、多维消融实验，定量定性分析充分
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，图表详尽，框架图直观
- 价值: ⭐⭐⭐⭐ — 0.33M参数的即插即用方案实用性高，对知识增强VQA有参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] WikiMixQA: A Multimodal Benchmark for Question Answering over Tables and Charts](wikimixqa_a_multimodal_benchmark_for_question_answering_over_tables_and_charts.md)
- [\[ACL 2025\] MTVQA: Benchmarking Multilingual Text-Centric Visual Question Answering](mtvqa_benchmarking_multilingual_text-centric_visual_question_answering.md)
- [\[CVPR 2026\] SEA: Evaluating Sketch Abstraction Efficiency via Element-level Commonsense Visual Question Answering](../../CVPR2026/multimodal_vlm/sea_evaluating_sketch_abstraction_efficiency_via_element-level_commonsense_visua.md)
- [\[ACL 2026\] WikiSeeker: Rethinking the Role of Vision-Language Models in Knowledge-Based Visual Question Answering](../../ACL2026/multimodal_vlm/wikiseeker_rethinking_the_role_of_vision-language_models_in_knowledge-based_visu.md)
- [\[CVPR 2026\] VQ-VA World: Towards High-Quality Visual Question-Visual Answering](../../CVPR2026/multimodal_vlm/vq-va_world_towards_high-quality_visual_question-visual_answering.md)

</div>

<!-- RELATED:END -->
