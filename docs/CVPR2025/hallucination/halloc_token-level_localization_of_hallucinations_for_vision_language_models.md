---
title: >-
  [论文解读] HalLoc: Token-Level Localization of Hallucinations for Vision Language Models
description: >-
  [CVPR 2025][幻觉检测][Token级定位] 提出HalLoc，一个15.5万样本、覆盖VQA/指令跟随/图像描述三类任务的token级幻觉标注数据集，并基于此训练了轻量级幻觉检测模型HalLocalizer，可在不影响效率的前提下即插即用地集成到现有VLM中实现实时概率化幻觉检测。 领域现状： 视觉语言模型（VL…
tags:
  - "CVPR 2025"
  - "幻觉检测"
  - "Token级定位"
  - "概率化检测"
  - "即插即用"
  - "VLM可靠性"
---

# HalLoc: Token-Level Localization of Hallucinations for Vision Language Models

**会议**: CVPR 2025  
**arXiv**: [2506.10286](https://arxiv.org/abs/2506.10286)  
**代码**: [GitHub](https://github.com/dbsltm/cvpr25_halloc)  
**领域**: 幻觉检测  
**关键词**: 幻觉检测、Token级定位、概率化检测、即插即用、VLM可靠性

## 一句话总结

提出HalLoc，一个15.5万样本、覆盖VQA/指令跟随/图像描述三类任务的token级幻觉标注数据集，并基于此训练了轻量级幻觉检测模型HalLocalizer，可在不影响效率的前提下即插即用地集成到现有VLM中实现实时概率化幻觉检测。

## 研究背景与动机

**领域现状**：
视觉语言模型（VLMs）虽然在多项多模态任务上表现出色，但容易生成与视觉内容不一致的描述（即幻觉），包括对象幻觉、属性幻觉、关系幻觉和场景幻觉。

**现有痛点**：
1. 现有幻觉检测方法（如GAVIE、FAITHScore、UNIHD）依赖GPT-4等大模型进行交叉验证，计算开销大、延迟高
2. 现有方法仅提供二元（幻觉/真实）的判定结果，无法处理现实中幻觉与真实信息边界模糊的情况
3. 已有数据集（HaELM 56K、MHalDetect 16K）粒度粗（句子/段落级别），且不包含幻觉类型标注
4. MHaluBench虽有类型标注但仅400样本，规模太小不足以训练模型

**核心矛盾**：
实际应用需要高效、实时、细粒度且概率化的幻觉检测，但现有方法要么计算代价太高，要么粒度太粗，要么数据规模不足。

**本文目标**
构建一个大规模、token级粒度、带类型标注的幻觉数据集，并基于此训练轻量级的即插即用幻觉检测模块。

**切入角度**：
设计HQA注入管道（Hallucinated Question-Answer Injection），通过可控方式将幻觉注入到来源文本中，实现大规模的token级幻觉标注。

**核心 idea**：
先构建大规模幻觉问答数据库，再将幻觉答案注入到不同任务的源文本中，从而获得token级标注的多任务幻觉数据集。

## 方法详解

### 整体框架

HalLoc的构建分两阶段：(1) 幻觉生成——基于概念关联偏差和统计偏差构建大规模幻觉问答数据库（HQA Database, 16万条）；(2) 幻觉注入——将HQA数据库中的幻觉答案系统性地注入到VQA、指令跟随和图像描述的源文本中，形成三个子集。基于此训练的HalLocalizer使用VisualBERT编码器+4个线性分类头实现token级四类幻觉检测。

### 关键设计

1. **四类幻觉类型体系**:
    - 功能：提供细粒度的幻觉类型分类
    - 核心思路：
        - **对象幻觉** `<obj>`：引用不存在的对象
        - **属性幻觉** `<attr><obj>`：与单一对象关联的错误属性（颜色、位置、动作等）
        - **关系幻觉** `<obj1><rel><obj2>`：两个对象间的错误交互关系
        - **场景幻觉** `<sce>`：错误的场景描述（天气、地点等）
    - 设计动机：不同类型的幻觉有不同的成因和检测难度，细粒度分类有助于针对性处理

2. **HQA注入管道（HQA-Injection Pipeline）**:
    - 功能：可控、可扩展地生成大规模token级幻觉标注数据
    - 核心思路：
        - 从GQA数据集选取问题，按关于对象/属性/关系/场景分类
        - 通过概念关联偏差（借用图中其他对象的属性）和统计偏差（高共现频率的错误属性）构造幻觉答案
        - 用GPT-4将幻觉答案注入到源文本的合适位置，替换正确答案或插入新句子
        - 附加VLM验证（InternVL、LLaVA、InstructBLIP三重验证）
    - 设计动机：直接让VLM生成幻觉文本很难控制类型和位置，HQA注入管道实现了精确控制

3. **HalLocalizer模型架构**:
    - 功能：轻量级token级幻觉检测
    - 核心思路：使用VisualBERT双向编码器处理VLM的最后隐层状态或直接处理文本响应，后接4个独立的线性分类头分别预测四类幻觉概率
    - 设计动机：VisualBERT参数量小、推理快，4个独立分类头支持多类型并发检测

### 损失函数 / 训练策略

- 标准二元交叉熵损失，四个分类头独立训练
- AdamW优化器，学习率$1 \times 10^{-6}$，余弦退火
- 25 epochs，4×A6000 GPU，约10小时训练
- 序列长度512，batch size 64
- 每个分类头的阈值在验证集上独立调优

## 实验关键数据

### 主实验

**VQA子集（token级幻觉检测F1）**

| 方法 | 对象 F1 | 属性 F1 | 关系 F1 | 场景 F1 |
|------|---------|---------|---------|---------|
| CHAIR | 0.19 | - | - | - |
| Always 1 | 0.44 | 0.44 | 0.44 | 0.12 |
| HalLocalizer (InternVL) | **0.71** | **0.94** | **0.71** | **0.93** |

**Instruct子集**

| 方法 | 对象 F1 | 属性 F1 | 关系 F1 | 场景 F1 |
|------|---------|---------|---------|---------|
| HalLocalizer (w/o Embed.) | **0.82** | **0.97** | 0.83 | **0.94** |
| HalLocalizer (InternVL) | 0.79 | 0.95 | **0.84** | 0.94 |

**Caption子集**

| 方法 | 对象 F1 | 属性 F1 | 关系 F1 | 场景 F1 |
|------|---------|---------|---------|---------|
| HalLocalizer (w/o Embed.) | **0.68** | **0.64** | **0.71** | **0.76** |
| HalLocalizer (InternVL) | 0.58 | 0.37 | 0.46 | 0.25 |

### 消融实验

**HalLocalizer vs Token Log Probability（总体幻觉检测F1）**

| 子集 | HalLocalizer | LogProb (LLaVA) |
|------|-------------|-----------------|
| VQA | 0.95 | 0.95 |
| Instruct | **0.91** | 0.47 |
| Caption | **0.71** | 0.17 |

### 关键发现

1. **VLM embedding并非总是帮助**：在Instruct和Caption子集上，不使用VLM embedding的纯文本模式反而表现更好，特别是Caption子集差异显著（F1: 0.68 vs 0.58）
2. **Log probability远不如专用检测器**：在较长响应中（Instruct/Caption），token log probability的幻觉检测精度急剧下降（Caption F1仅0.17 vs HalLocalizer的0.71）
3. **属性和场景幻觉更容易检测**：VQA和Instruct子集中属性/场景F1 >0.90，对象和关系F1 ~0.70左右，说明后者更具挑战性
4. **Caption子集最难**：所有方法在Caption上表现最差，因为paragraph长度长（平均57.53词）且幻觉密度低（仅5%）
5. **人工评估验证高质量**：幻觉生成准确率91%，注入准确率98%，标注者完全一致

## 亮点与洞察

1. **首个大规模token级幻觉标注数据集**：15.5万样本，覆盖VQA/指令跟随/图像描述三大核心VLM任务，token级四类标注，远超现有数据集
2. **概率化检测范式**：不同于现有方法的二元判定，HalLocalizer输出幻觉概率，更适合边界模糊的实际场景
3. **即插即用设计**：HalLocalizer作为附加模块集成到VLM中，不影响原模型生成能力，可实现实时伴随式幻觉检测
4. **HQA注入管道的巧妙设计**：将幻觉数据构建分解为"构造幻觉答案"和"注入到源文本"两步，实现了类型可控、位置可控的大规模标注

## 局限与展望

1. HQA数据库基于GQA数据集构建，场景和对象类别覆盖受限
2. 场景幻觉样本相对稀少（仅4084条），因为具有明确环境设定的图像较少
3. Caption子集上检测性能仍有较大提升空间（最佳F1仅0.68/0.64/0.71/0.76）
4. 未探索HalLocalizer对下游VLM生成质量的反馈机制（如拒绝采样或重生成）
5. 仅在英文数据上验证，多语言场景下的幻觉检测有待探索

## 相关工作与启发

- **CHAIR**: 最早的对象幻觉检测方法，基于规则匹配，粒度粗且仅支持对象类型
- **UNIHD**: 使用链式大模型进行段落级幻觉分类，计算代价高
- **MHalDetect**: 训练专用检测模型但仅16K数据、句子级粒度
- 启发：幻觉检测的关键瓶颈是训练数据，而非模型架构；HQA注入的思路可推广到其他NLP数据增强场景

## 评分

- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] One Token, Two Fates: A Unified Framework via Vision Token Manipulation Against MLLMs Hallucination](one_token_two_fates_a_unified_framework_via_vision_token_manipulation_against_ml.md)
- [\[ICLR 2026\] Token-Guard: Towards Token-Level Hallucination Control via Self-Checking Decoding](../../ICLR2026/hallucination/token-guard_towards_token-level_hallucination_control_via_self-checking_decoding.md)
- [\[CVPR 2025\] ODE: Open-Set Evaluation of Hallucinations in Multimodal Large Language Models](ode_open-set_evaluation_of_hallucinations_in_multimodal_large_language_models.md)
- [\[ICCV 2025\] Mitigating Object Hallucinations via Sentence-Level Early Intervention](../../ICCV2025/hallucination/mitigating_object_hallucinations_via_sentence-level_early_intervention.md)
- [\[ACL 2025\] Visual Evidence Prompting Mitigates Hallucinations in Large Vision-Language Models](../../ACL2025/hallucination/visual_evidence_prompting.md)

</div>

<!-- RELATED:END -->
