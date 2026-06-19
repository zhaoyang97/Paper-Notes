---
title: >-
  [论文解读] MegaPairs: Massive Data Synthesis For Universal Multimodal Retrieval
description: >-
  [ACL 2025][多模态VLM][多模态] 提出 MegaPairs 数据合成方法，利用异构 KNN 三元组从开放域图像语料中挖掘相关图像对，结合 VLM/LLM 生成检索指令，合成 2600 万多模态训练实例，训练的 MMRet 模型仅用 0.5M 数据即超越使用 36.7M 数据的 MagicLens（70× 数据效率），在 4 个 CIR 基准和 MMEB 36 个数据集上达到 SOTA。
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "多模态"
  - "Data Synthesis"
  - "Composed Image Retrieval"
  - "对比学习"
  - "instruction tuning"
  - "Hard Negatives"
---

# MegaPairs: Massive Data Synthesis For Universal Multimodal Retrieval

**会议**: ACL 2025  
**arXiv**: [2412.14475](https://arxiv.org/abs/2412.14475)  
**代码**: 将公开（数据集、模型、pipeline 全套公开）  
**领域**: Multimodal VLM / 多模态检索  
**关键词**: Multimodal Retrieval, Data Synthesis, Composed Image Retrieval, Contrastive Learning, instruction tuning, Hard Negatives

## 一句话总结

提出 MegaPairs 数据合成方法，利用异构 KNN 三元组从开放域图像语料中挖掘相关图像对，结合 VLM/LLM 生成检索指令，合成 2600 万多模态训练实例，训练的 MMRet 模型仅用 0.5M 数据即超越使用 36.7M 数据的 MagicLens（70× 数据效率），在 4 个 CIR 基准和 MMEB 36 个数据集上达到 SOTA。

## 研究背景与动机

**领域现状**: 多模态检索需求快速增长（图像搜索、VQA、RAG 等），基于预训练视觉-语言模型（CLIP/ALIGN/SigLIP）的方法已建立初步能力，但仅限于文本-图像匹配，无法处理更复杂的多模态任务。

**指令微调瓶颈**: 指令微调可增强多任务能力，但多模态检索的指令数据极度稀缺。现有方法（如 MagicLens）从共存于同一网页的图像对中合成数据，面临四大限制：
   - **可扩展性差**: 仅少量网页包含多图像
   - **质量低**: 共存图像常不相关或高度重复
   - **多样性不足**: 相关图像关系单调（如同一物体不同角度）
   - **可获取性差**: 大规模数据集通常私有不公开

**核心 idea**: 利用开放域图像语料（如 DataComp）+ 多个相似度模型挖掘异构图像对 + 开源 VLM/LLM 生成检索指令 → 大规模、高质量、多样化、公开可用的多模态检索训练数据。

## 方法详解

### 整体框架

MegaPairs 分为两个阶段：(1) 挖掘相关图像对——使用三种相似度模型发现异构关联；(2) 生成开放式检索指令——VLM 描述关系 + LLM 精炼为指令。

### 阶段一：挖掘相关图像对

- **三种相似度模型引入异构关联**:
  1. **视觉-语义关联** (EVA-CLIP 图像编码器): 捕捉语义相关但视觉不同的图像对（如同一辆车的不同视角）
  2. **视觉-模式关联** (DINOv2): 捕捉视觉相似但语义可能不同的图像对（如不同车在相似背景中）
  3. **标题关联** (EVA-CLIP 文本编码器): 基于图像标题的文本相似性
- **过滤策略**: 保留相似度在 (0.8, 0.96) 范围内的图像对，排除弱关联和近重复
- **Hard Negatives**: 对每个图像对 (I_q, I_ti)，检索集中的其他图像 {I_tj | tj ≠ ti} 自然充当 hard negatives，每对引入 5 个 hard negatives

### 阶段二：生成开放式指令

- **Step 1**: VLM (InternVL2-26B) 输入图像对，生成两张图像之间共同概念和差异的详细描述 D_i
- **Step 2**: LLM (LLaMA3-8B) 将描述精炼为检索指令 T_{q→ti}，每对生成至少 3 条不同指令增加多样性
- **最终三元组**: (I_q, T_{q→ti}, I_ti)，其中 (I_q, T_{q→ti}) 用于检索 I_ti

### 实现规模

- 图像语料: Recap-DataComp-1B 的 2000 万带标题图像子集
- 合成结果: 26,235,105 个图像对，全面开源

### MMRet 模型

设计了两种架构：

1. **CLIP-based MMRet** (Base/Large):
    - 双编码器架构独立编码图像和文本
    - 多模态嵌入使用 score-fusion: e_it = Φ_I(I) + Φ_T(T)

2. **MLLM-based MMRet** (基于 LLaVA-1.6 Mistral 7B):
    - 将图像 token 直接输入 LLM 处理
    - 使用任务特定指令: <instruct> {task_inst} <query> {q_t} {q_i} [EOS]
    - 以 [EOS] token 的归一化最后隐藏状态作为嵌入

### 训练目标

标准 InfoNCE 对比损失:
$$\mathcal{L} = -\frac{1}{|\mathcal{Q}|}\sum_{q_i \in \mathcal{Q}} \log \frac{\exp(\mathbf{e}_{q_i} \cdot \mathbf{e}_{c_i^+}/\tau)}{\sum_{c_j \in \mathcal{C}} \exp(\mathbf{e}_{q_i} \cdot \mathbf{e}_{c_j}/\tau)}$$
温度参数 τ = 0.02。query 和 candidate 可以是图像、文本或图文组合。

## 实验

### 零样本 CIR 性能（Table 1）

| 方法 | Backbone | Params | CIRCO mAP@5 | CIRR R@1 | FashionIQ R@10 | GeneCIS Rs@1 |
|------|----------|--------|-------------|----------|----------------|--------------|
| MagicLens-L‡ | CoCa-L | 613M | 34.1 | 33.3 | 38.0 | 16.7 |
| IP-CIR | CLIP-G | 43.8B† | 32.8 | 39.3 | 45.7 | - |
| MMRet-Base | CLIP-B | 149M | 34.3 | 36.1 | 31.9 | 18.0 |
| MMRet-Large | CLIP-L | 428M | 39.2 | 38.0 | 34.6 | 18.1 |
| **MMRet-MLLM** | LLaVA-1.6 | 7.57B | **42.2** | **46.7** | 35.6 | **21.1** |

**关键发现**:
- MMRet-MLLM 在 CIRCO 上超越前 SOTA 8.1 个百分点（42.2 vs 34.1）
- MMRet-Base (149M) 甚至超越了多数大模型（包括 MagicLens-L 613M）
- 所有 MMRet 模型在所有规模上均领先

### MMEB 零样本性能（Table 2）

| 模型 | Classification | VQA | Retrieval | Grounding | Overall |
|------|---------------|-----|-----------|-----------|---------|
| UniIR | 42.1 | 15.0 | 60.1† | 62.2 | 42.8 |
| MMRet-MLLM | **47.2** | **18.4** | **56.5** | **62.2** | **44.0** |

在 36 个数据集的综合评估中，MMRet-MLLM 取得最高整体得分 44.0，且 UniIR 的检索元任务包含 10/12 个 MMEB 检索数据集（非严格零样本）。

### MMEB 微调性能（Table 3）

| 模型 | IND | OOD | Overall |
|------|-----|-----|---------|
| VLM2Vec (LLaVA-1.6) | 61.0 | 47.5 | 55.0 |
| VLM2Vec (Phi-3.5-V) | 66.5 | 52.0 | 60.1 |
| **MMRet-MLLM** | **68.0** | **59.1** | **64.1** |

微调后 OOD 性能提升 11.6%（vs LLaVA-1.6 baseline），展现 MegaPairs 预训练赋予的强泛化能力。

### 数据质量与可扩展性（Figure 2）

- **70× 数据效率**: 仅 0.5M MegaPairs 样本即超越用 36.7M MagicLens 数据训练的模型
- **持续缩放**: 性能随数据量增长持续提升，未观察到饱和

### 消融实验

1. **Hard Negatives 效果 (Table 4)**: 使用挖掘的 hard negatives 在 CIRCO 上提升 2.6 个百分点（29.7→32.3）
2. **搜索策略 (Table 5)**: 三种相似度模型联合使用效果最佳（32.3 mAP@5），文本相似度单独使用优于视觉相似度

## 亮点与洞察

1. **数据合成范式创新**: 从"依赖网页共存图像"转向"从开放域语料主动挖掘"，彻底解决了可扩展性瓶颈
2. **异构关联是关键**: 三种不同相似度模型引入的多样化关联是数据质量的核心保障——单一模型的关系太单调
3. **极致的数据效率**: 0.5M 数据胜 36.7M，说明数据质量远比数量重要，这是检索领域的经典洞察
4. **全开源**: 数据集 + 模型 + pipeline 全套开源，对社区贡献巨大
5. **小模型强势**: MMRet-Base (149M) 超越多数大模型，证明正确的训练数据可以弥补模型规模差距

## 局限性

1. 仅使用三种检索器构建图像对，更多检索策略（如 BGE、图文交叉检索）可能进一步提升多样性
2. 图像来源经过 Datacomp 团队筛选但可能不完全干净
3. CLIP-based MMRet 在 FashionIQ 等垂直领域性能不如 CIR 专用模型

## 相关工作

- **多模态检索**: CLIP/ALIGN/SigLIP 等预训练模型；UniIR、E5-V 等通用多模态嵌入
- **指令微调**: 从 LLM (FLAN/InstructGPT) 到 Embedding (Su et al., GTE) 再到多模态 (MagicLens, UniIR)
- **CIR 方法**: SEARLE, CIReVL, LDRE, CompoDiff 等零样本/训练方法

## 评分 ⭐⭐⭐⭐⭐

- **创新性**: ⭐⭐⭐⭐⭐ 异构 KNN 三元组 + VLM/LLM 注释的数据合成范式新颖且有效
- **实验完备性**: ⭐⭐⭐⭐⭐ 4 个 CIR 基准 + MMEB 36 数据集 + 零样本/微调 + 数据缩放/消融
- **实用性**: ⭐⭐⭐⭐⭐ 全套开源，2600 万数据立即可用，解决领域痛点
- **写作**: ⭐⭐⭐⭐ 方法描述清晰，图示丰富

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Universal Retrieval for Multimodal Trajectory Modeling](../../ICML2025/multimodal_vlm/universal_retrieval_for_multimodal_trajectory_modeling.md)
- [\[NeurIPS 2025\] Generalized Contrastive Learning for Universal Multimodal Retrieval](../../NeurIPS2025/multimodal_vlm/generalized_contrastive_learning_for_universal_multimodal_re.md)
- [\[CVPR 2025\] Img-Diff: Contrastive Data Synthesis for Multimodal Large Language Models](../../CVPR2025/multimodal_vlm/img-diff_contrastive_data_synthesis_for_multimodal_large_language_models.md)
- [\[ICLR 2026\] U-MARVEL: Unveiling Key Factors for Universal Multimodal Retrieval via Embedding Learning](../../ICLR2026/multimodal_vlm/u-marvel_unveiling_key_factors_for_universal_multimodal_retrieval_via_embedding_.md)
- [\[ICCV 2025\] Oasis: One Image is All You Need for Multimodal Instruction Data Synthesis](../../ICCV2025/multimodal_vlm/oasis_one_image_is_all_you_need_for_multimodal_instruction_data_synthesis.md)

</div>

<!-- RELATED:END -->
