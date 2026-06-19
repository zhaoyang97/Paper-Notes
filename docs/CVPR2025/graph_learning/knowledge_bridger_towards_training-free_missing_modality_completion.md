---
title: >-
  [论文解读] Knowledge Bridger: Towards Training-Free Missing Modality Completion
description: >-
  [CVPR 2025][图学习][缺失模态补全] 本文提出 Knowledge Bridger，一个免训练的缺失模态补全框架，通过利用大型多模态模型（LMM）自动挖掘多模态知识、构建知识图谱来指导缺失模态的生成与排序，在通用场景和医学OOD场景下均超越了现有方法。 1. 领域现状：缺失模态补全（MMC）是多模态学习中的关键问…
tags:
  - "CVPR 2025"
  - "图学习"
  - "缺失模态补全"
  - "知识图谱"
  - "免训练"
  - "大型多模态模型"
  - "OOD泛化"
---

# Knowledge Bridger: Towards Training-Free Missing Modality Completion

**会议**: CVPR 2025  
**arXiv**: [2502.19834](https://arxiv.org/abs/2502.19834)  
**代码**: [https://github.com/Guanzhou-Ke/Knowledge-Bridger](https://github.com/Guanzhou-Ke/Knowledge-Bridger)  
**领域**: 医学图像 / 多模态学习  
**关键词**: 缺失模态补全, 知识图谱, 免训练, 大型多模态模型, OOD泛化

## 一句话总结
本文提出 Knowledge Bridger，一个免训练的缺失模态补全框架，通过利用大型多模态模型（LMM）自动挖掘多模态知识、构建知识图谱来指导缺失模态的生成与排序，在通用场景和医学OOD场景下均超越了现有方法。

## 研究背景与动机

1. **领域现状**：缺失模态补全（MMC）是多模态学习中的关键问题，现有的成功方法依赖精心设计的融合技术和大量完整数据的预训练，如 MMIN 利用模态间插值恢复，DiCMoR 利用跨模态解耦进行补全。
2. **现有痛点**：这些方法在域外（OOD）场景下泛化能力有限，切换到医学影像或自动驾驶等新领域时需要重新训练，带来大量人力和计算成本。虽然 prompt-learning 和 missing modality tag 等方案降低了适配成本，但仍依赖大量训练数据。
3. **核心矛盾**：如何同时实现低资源依赖（无需针对新领域的训练数据和大量计算）和强 OOD 泛化能力？
4. **本文目标** 两个子问题：(1) 如何约束生成过程保证缺失模态的保真度；(2) 如何从多个生成候选中挑选最佳结果。
5. **切入角度**：大型多模态模型（如 Qwen2-VL）具有强大的 OOD 能力和 in-context learning 能力，可以在不微调的情况下理解和生成多模态内容。
6. **核心 idea**：利用 LMM 从可用模态自动构建知识图谱，用知识驱动的方式指导缺失模态生成，并通过图相似度+语义相似度进行排序选出最佳补全。

## 方法详解

### 整体框架
Knowledge Bridger 是一个三阶段的免训练管线：输入是可用的多模态数据和领域先验知识，输出是高质量的缺失模态数据。整个过程不需要任何训练，完全利用预训练 LMM 的能力。三个阶段分别是：知识图谱建模 → 知识驱动生成 → 知识驱动排序。

### 关键设计

1. **知识图谱建模（Knowledge Graph Modeling）**:

    - 功能：从可用模态中自动提取结构化知识，构建知识图谱用于指导后续的生成和排序。
    - 核心思路：利用 LMM 的 Chain-of-Thought (CoT) 推理能力，通过设定 {Entity: Reasoning Prompts} 形式的提取规则（如 "识别图中的主要对象"），让 LMM 逐步分析可用模态中的实体、关系和属性。CoT 分两步：先让 LMM 对每条规则生成简洁回答，再从回答中提取唯一的实体-关系对。对于医学等专业领域，还可注入领域先验知识（如组织学、临床诊断信息）减少幻觉。
    - 设计动机：直接让 LMM 一次性提取所有实体关系容易受上下文窗口限制而遗漏，CoT 的分步策略既提高单条规则的回答准确性，又能更好地综合信息。

2. **知识驱动生成（Knowledge-driven Generation）**:

    - 功能：利用知识图谱中的结构化信息，引导 LMM 精确生成缺失模态的候选内容。
    - 核心思路：提出知识驱动的实体交替策略——从知识图谱中选取与缺失模态相关的实体，以每个实体为主题采用多视角生成方式，让 LMM 生成包含所有节点和属性信息的标准化文本描述。然后将这些描述传递给模态生成器（图像缺失时用 SDXL 1.0，医学图像用 Cheff 模型，文本缺失时由 LMM 直接生成）。默认生成 5 个候选。
    - 设计动机：直接让 LMM 描述图像具有很大随机性——既不知道缺失文本的形式（标题/摘要/描述），也无法精确指定描述重点。通过知识图谱引导的多视角生成，可以减少随机性并提高可控性。

3. **知识驱动排序（Knowledge-based Ranking）**:

    - 功能：自动评估生成的缺失模态质量，选出最佳候选。
    - 核心思路：综合两种相似度计算质量分数 $QS(x_a, x_m) = \cos_{graph}(f_a(x_a), f_a(x_m)) + [\cos(f_c(x_a), f_c(x_m)) + \cos(f_b(x_a), f_b(x_m))]$，其中 $f_a$ 提取邻接矩阵计算图相似度（衡量知识结构一致性），$f_c$ 和 $f_b$ 分别利用 CLIP 和 BLIP 提取语义嵌入计算表示相似度（衡量语义一致性）。最终选取 QS 最高的候选作为输出。
    - 设计动机：单纯依赖生成质量无法保证准确性，需要从结构和语义两个互补维度进行评估，图相似度捕捉知识层面的对齐，语义相似度捕捉表示层面的一致。

### 损失函数 / 训练策略
本方法为免训练框架，无需任何训练损失。整个管线完全依赖预训练模型（Qwen2-VL 作为 LMM，SDXL/Cheff 作为图像生成器，CLIP/BLIP 作为语义评估器），通过 prompt 工程和 in-context learning 实现零样本补全。

## 实验关键数据

### 主实验

| 数据集 | 缺失率 | 指标 | Knowledge Bridger (7B) | 之前SOTA | 提升 |
|--------|--------|------|----------------------|----------|------|
| COCO-2014 | η=0.7 | F1 | 77.9% | 72.3% (MPLMM) | +5.6% |
| COCO-2014 | η=0.7 | AP | 83.5% | 80.1% (MPLMM) | +3.4% |
| MM-IMDb | η=0.7 | F1 | 55.2% | 49.1% (MPLMM) | +6.1% |
| MM-IMDb | η=0.7 | AP | 61.8% | 56.2% (MPLMM) | +5.6% |
| IU X-ray (OOD) | η=0.7 | F1 | 46.3% | 36.8% (MPMM) | +9.5% |
| IU X-ray (OOD) | η=0.7 | AP | 70.5% | 61.9% (MPLMM) | +8.6% |

在 OOD 医学数据集 IU X-ray 上改进尤为显著，高缺失率下 F1 提升 9.5%，AP 提升 8.6%。

### 消融实验

| 配置 | F1 (MM-IMDb) | F1 (IU X-ray) | 说明 |
|------|-------------|---------------|------|
| Full model (Qwen-VL-7B) | 55.2 | 46.3 | 完整模型 |
| w/o Knowledge Modeling | -1.3 | -17.5 | 不用知识建模直接生成，OOD 大幅下降 |
| w/o Knowledge + Random Ranking | -1.6 | -19.3 | 同上再加随机排序 |
| Random Ranking | -0.5 | -3.8 | 只去掉知识排序 |
| w/o Knowledge Ranking | -0.2 | -1.9 | 去掉图相似度排序 |
| w/o Semantic Ranking | -0.2 | -3.6 | 去掉语义相似度排序 |

### 关键发现
- **知识建模是最关键组件**：在 OOD 场景（IU X-ray）去掉知识建模后 F1 暴降 17.5%，说明领域知识对跨域泛化至关重要。
- **方法随 LMM 规模有效scaling**：从 2B 到 7B 模型，所有指标持续提升，使用 GPT-4o 时效果更佳。
- **合成数据增强下游任务**：生成的合成模态数据不仅能补全缺失，还能提升其他 MMC 模型的性能。
- 语义排序在 OOD 场景中比图结构排序更关键（-3.6 vs -1.9），而在通用场景中两者贡献相当。

## 亮点与洞察
- **免训练范式**：完全不需要在目标域训练，而是巧妙利用 LMM 的 in-context learning 和 CoT 推理能力完成跨域模态补全，打破了 MMC 领域对大量训练数据的依赖。
- **知识图谱作为桥梁**：将 LMM 的非结构化理解转化为结构化知识图谱，再用图谱指导生成和排序，这个「知识桥梁」的设计让整个管线既可解释又可控。
- **生成+排序的解耦设计**：生成多个候选再排序选择的策略可以迁移到其他需要高精度生成的任务，如文本到图像、跨模态检索等。

## 局限与展望
- **依赖 LMM 质量**：性能与 LMM 强相关，小模型（2B）效果明显弱于大模型（7B），而大模型的推理成本较高。
- **生成速度较慢**：每个样本需要生成 5 个候选并分别构建知识图谱和计算相似度，实时性不足。
- **知识提取规则的可扩展性**：当前的实体-关系提取规则需要人工定义，对于全新领域可能需要调整。
- 图像生成质量受限于底层生成器（SDXL/Cheff），在某些细粒度医学图像场景可能不够精确。
- 可以考虑引入主动学习策略，让模型自动决定哪些候选需要更多轮生成。

## 相关工作与启发
- **vs MMIN (ACL'21)**: MMIN 通过模态间插值学习缺失表示，需要训练且在 OOD 下表现差；本文免训练且 OOD 泛化强，但推理成本更高。
- **vs MPLMM (ACL'24)**: MPLMM 用 prompt-learning 动态调整融合策略，仍需训练数据；本文完全免训练，在高缺失率下优势尤为明显。
- **vs DiCMoR (CVPR'23)**: DiCMoR 是插值方法，在高缺失率下性能严重退化；本文利用生成模型保持稳定性能。
- 知识图谱引导的生成范式可以启发其他需要可控生成的场景，如医学报告自动生成和多模态推理。

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将 LMM 应用于 MMC 任务，知识图谱桥梁的设计新颖
- 实验充分度: ⭐⭐⭐⭐ 覆盖通用和 OOD 场景，消融实验系统
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法描述系统
- 价值: ⭐⭐⭐⭐ 免训练范式对数据稀缺的医学领域有重要实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Beyond Completion: A Foundation Model for General Knowledge Graph Reasoning](../../ACL2025/graph_learning/beyond_completion_a_foundation_model_for_general_knowledge_graph_reasoning.md)
- [\[NeurIPS 2025\] Uncertain Knowledge Graph Completion via Semi-Supervised Confidence Distribution Learning](../../NeurIPS2025/graph_learning/uncertain_knowledge_graph_completion_via_semi-supervised_confidence_distribution.md)
- [\[ICML 2026\] GILT: An LLM-Free, Tuning-Free Graph Foundational Model for In-Context Learning](../../ICML2026/graph_learning/gilt_an_llm-free_tuning-free_graph_foundational_model_for_in-context_learning.md)
- [\[ACL 2026\] GS-Quant: Granular Semantic and Generative Structural Quantization for Knowledge Graph Completion](../../ACL2026/graph_learning/gs-quant_granular_semantic_and_generative_structural_quantization_for_knowledge_.md)
- [\[AAAI 2026\] MyGram: Modality-aware Graph Transformer with Global Distribution for Multi-modal Entity Alignment](../../AAAI2026/graph_learning/mygram_modality-aware_graph_transformer_with_global_distribution_for_multi-modal.md)

</div>

<!-- RELATED:END -->
