---
title: >-
  [论文解读] Transferring Textual Preferences to Vision-Language Understanding through Model Merging
description: >-
  [ACL 2025][多模态VLM][模型合并] 提出一种免训练方法，通过模型参数合并（model merging）将纯文本奖励模型（RM）的偏好能力迁移到大视觉语言模型（LVLM）中，构建视觉语言奖励模型（VLRM），在多个多模态评估基准上超越LVLM直接评分和纯文本RM。 问题背景 大视觉语言模型（LVLM）在多模态任务…
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "模型合并"
  - "视觉语言奖励模型"
  - "偏好迁移"
  - "免训练"
  - "RLHF"
---

# Transferring Textual Preferences to Vision-Language Understanding through Model Merging

**会议**: ACL 2025  
**arXiv**: [2502.13487](https://arxiv.org/abs/2502.13487)  
**作者**: Chen-An Li, Tzu-Han Lin, Yun-Nung Chen, Hung-yi Lee (National Taiwan University)
**代码**: [GitHub](https://github.com/lca0503/MergeToVLRM)  
**领域**: 多模态VLM  
**关键词**: 模型合并, 视觉语言奖励模型, 偏好迁移, 免训练, RLHF

## 一句话总结

提出一种免训练方法，通过模型参数合并（model merging）将纯文本奖励模型（RM）的偏好能力迁移到大视觉语言模型（LVLM）中，构建视觉语言奖励模型（VLRM），在多个多模态评估基准上超越LVLM直接评分和纯文本RM。

## 研究背景与动机

### 问题背景
大视觉语言模型（LVLM）在多模态任务上表现出色，但其评估生成内容质量的能力仍然有限。训练专门的视觉语言奖励模型（VLRM）需要收集昂贵的多模态偏好数据并进行大规模训练，成本极高。与此同时，文本领域已经积累了丰富的偏好数据集和高质量的文本奖励模型。

### 已有工作的不足
- 现有VLRM训练依赖大量多模态偏好数据，收集和标注成本极高
- LVLM直接做评分（LVLM-as-a-Judge）在VL-RewardBench等挑战性任务上表现不佳，尤其在评分和批量排序任务中
- 文本RM虽然偏好判断能力强，但完全缺乏视觉理解能力，无法处理多模态场景
- 级联方法（先用LVLM描述图片，再用文本RM评分）信息传递有损失

### 核心动机
许多LVLM（如Llama-3.2-Vision）的语言模块本身就是在预训练语言模型基础上扩展视觉编码器和适配器而来。这种架构特点意味着：如果文本RM也来自相同的预训练语言模型，那么两者的transformer参数在同一参数空间中，可以通过合并来组合各自的能力——LVLM提供视觉理解，文本RM提供偏好判断。

## 方法详解

### 整体框架

核心思想极为简洁：将LVLM和文本RM的共享transformer层进行参数合并，保留LVLM的视觉编码器和适配器，保留文本RM的奖励头，组合成一个完整的VLRM。

具体地，合并后的VLRM由五个部分组成：
$$\theta^{\text{MERGE}} = \{\theta^{\text{LVLM}}_{\text{venc}}, \theta^{\text{LVLM}}_{\text{adapt}}, \theta^{\text{MERGE}}_{\text{emb}}, \theta^{\text{MERGE}}_{\text{trans}}, \theta^{\text{RM}}_{\text{rm}}\}$$

- 视觉编码器和适配器：完全来自LVLM，保持视觉理解能力
- embedding层和transformer层：通过合并策略融合两个模型
- 奖励头：完全来自文本RM，将隐藏状态映射为标量奖励值

前提条件是两个模型必须共享相同的预训练语言模型基座（本文中均基于Llama-3.1-8B）。

### 关键设计1：四种合并策略

**加权平均（Linear）**：最直接的合并方式，将两个模型的transformer参数按权重λ线性组合。简单但可能导致参数干扰。

**任务算术（Task Arithmetic）**：先分别计算LVLM和RM相对于预训练模型的"任务向量"（即参数增量），再将两个任务向量按比例叠加到预训练参数上。这种方式避免了直接平均带来的参数抵消问题。

**TIES**：在任务算术基础上增加了三步处理——按幅度修剪小参数、解决符号冲突（保留累积幅度更大方向的参数）、对保留参数取均值。通过密度参数$d$控制保留比例。

**DARE**：随机以概率$p$丢弃任务向量中的delta参数，并将保留参数放大$1/(1-p)$倍进行重缩放。可与Task Arithmetic或TIES组合使用。

### 关键设计2：Embedding层合并策略

由于LVLM和RM可能有不同的词表扩展（如LVLM增加了视觉token），embedding层的合并需要特殊处理。采用MergeKit的策略：
1. 预训练模型中已有的token使用预训练embedding
2. 仅出现在某个模型中的token直接使用该模型的embedding
3. 同时出现在多个模型中的token取embedding均值

### 关键设计3：超参数选择

使用从RLAIF-V训练集采样400个实例作为验证集进行超参数搜索。对于Linear和Task Arithmetic，搜索$\lambda \in [0.0, 0.1, ..., 1.0]$；对于TIES和DARE，搜索$\lambda \in [0.5, 0.7, 1.0]$和$d \in [0.2, 0.4, 0.6, 0.8]$。

## 实验关键数据

### 实验1：主要基准结果（使用Tulu-2.5-RM合并）

| 方法 | VL-RB General | VL-RB Hallucination | VL-RB Reasoning | VL-RB Overall | TextVQA | MMMU-Pro Std | MMMU-Pro Vision |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Llama-3.2-Vision | 33.3 | 38.4 | 56.6 | 42.9 | 46.4 | 28.8 | 19.8 |
| Tulu-2.5-RM (纯文本) | 43.2 | 31.4 | 54.1 | 38.9 | 42.6 | 29.8 | 21.4 |
| Cascade | 44.8 | 37.8 | 57.2 | 43.8 | 43.2 | 30.9 | 23.4 |
| Linear | 39.3 | 52.3 | 54.4 | 51.0 | 54.7 | 27.8 | 22.1 |
| Task Vec. | 48.6 | 59.4 | 59.7 | 57.9 | 59.0 | 31.0 | 22.7 |
| TIES | 43.7 | 58.2 | 58.5 | 56.2 | 64.2 | 29.1 | 22.6 |
| DARE + Task Vec. | **49.2** | **61.7** | **61.0** | **59.7** | 58.8 | 30.3 | 22.4 |
| DARE + TIES | 49.2 | 59.1 | 58.2 | 57.4 | 57.3 | **31.6** | 22.0 |

合并后的VLRM在VL-RewardBench上将Overall从42.9%提升至59.7%（+16.8），在TextVQA上从46.4提升至64.2（+17.8），提升幅度显著。

### 实验2：与大规模模型和商业模型对比

| 方法 | VL-RB General | VL-RB Hallucination | VL-RB Reasoning |
|------|:---:|:---:|:---:|
| Llama-3.2-Vision (11B) | 33.3 | 38.4 | 56.6 |
| Llama-3.2-Vision (90B) | 42.6 | 57.3 | 61.7 |
| GPT-4o-mini | 41.7 | 34.5 | 58.2 |
| Gemini-1.5-Flash | 47.8 | 59.6 | 58.4 |
| Gemini-1.5-Pro | 50.8 | 72.5 | 64.2 |
| GPT-4o | 49.1 | 67.6 | 70.5 |
| DARE + Task Vec. (本文) | 49.2 | 61.7 | 61.0 |

合并后的11B VLRM超越了90B LVLM，在General和Hallucination维度上可与GPT-4o和Gemini-1.5-Pro竞争。

### 实验3：视觉编码器消融（去除图像输入）

| 方法 | VL-RB (有图) | VL-RB (无图) | TextVQA (有图) | TextVQA (无图) |
|------|:---:|:---:|:---:|:---:|
| Task Vec. | 57.9 | 44.9 | 59.0 | 38.7 |
| DARE + Task Vec. | 59.7 | 44.5 | 58.8 | 36.2 |
| TIES | 56.2 | 42.7 | 64.2 | 40.9 |

去除图像后性能大幅下降（VL-RB约降13-15点，TextVQA约降20-28点），证明合并后的VLRM确实在利用视觉编码器，而非仅靠文本RM贡献。

## 关键发现

- 高级合并策略（Task Arithmetic, TIES, DARE）明显优于简单加权平均，表明处理参数干扰至关重要
- 即使将任务向量修剪至仅保留20%-40%的参数，合并后模型仍能保持强劲性能，与LLM合并领域的已有发现一致
- 不同benchmark的最优超参数不同——VL-RewardBench对$\lambda$不敏感，而MMMU-Pro在$\lambda=1.0$时最优
- 合并方法优于级联方法（Cascade），说明参数空间的直接融合比"先描述图片再评分"的流水线方式捕获了更多信息

## 亮点与洞察

- **极简但有效的设计理念**：整个方法零训练、零多模态偏好数据，仅通过参数合并就实现了能力组合，计算开销仅需CPU即可完成
- **架构洞察深刻**：利用LVLM语言模块与文本RM共享预训练基座这一事实，将跨模态偏好迁移转化为同一参数空间内的向量运算
- **实用价值高**：整个合并流程在CPU上1.5-6小时即可完成，验证推理仅需约1.5小时GPU时间，远低于训练VLRM的成本
- **11B合并模型击败90B模型**：证明了能力组合的效率优于单纯的规模扩大

## 局限性

- **模型架构限定**：仅在LLaMA架构上验证，要求LVLM和RM共享同一预训练基座，限制了适用范围
- **规模单一**：仅测试了11B LVLM + 8B RM的组合，未探索更大或更小模型的效果
- **超参数敏感**：最优超参数因任务而异，需要精心构建验证集来选择，增加了实际使用的额外成本
- **未与训练式方法对比**：缺少与直接在多模态偏好数据上训练的VLRM的系统对比
- **未实验RLHF集成**：未将合并后的VLRM用于PPO等RLHF算法，无法评估端到端效果

## 相关工作与启发

本文处于**模型合并**和**奖励模型**的交叉地带。与DogeRM（Lin et al., 2024）将领域知识通过合并注入RM相似，但本文沿相反方向——将RM能力注入LVLM，且是跨模态的。REMEDY（Zhu et al., 2025）也研究LVLM合并，但聚焦于同类LVLM之间的合并，而非异构模型（LVLM + RM）。

启发：这种"借用已有能力通过合并免训练注入"的范式可以推广——例如将代码生成RM合并到通用LLM中、将安全对齐RM合并到特定领域模型中。关键前提是共享基座模型。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 跨模态免训练偏好迁移的视角新颖，但合并技术本身已有
- 实验充分度: ⭐⭐⭐⭐ — 多基准、消融实验、超参数分析完整，但缺少与训练式方法的对比
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，动机表述合理，定性案例分析有助于理解
- 价值: ⭐⭐⭐⭐ — 实用性强，提供了一种低成本构建VLRM的可行路径

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Value-Spectrum: Quantifying Preferences of Vision-Language Models via Value Decomposition](value_spectrum_vlm_pref.md)
- [\[NeurIPS 2025\] RobustMerge: Parameter-Efficient Model Merging for MLLMs with Direction Robustness](../../NeurIPS2025/multimodal_vlm/robustmerge_parameter-efficient_model_merging_for_mllms_with_direction_robustnes.md)
- [\[CVPR 2025\] Cropper: Vision-Language Model for Image Cropping through In-Context Learning](../../CVPR2025/multimodal_vlm/cropper_vision-language_model_for_image_cropping_through_in-context_learning.md)
- [\[ACL 2025\] Jailbreak Large Vision-Language Models Through Multi-Modal Linkage](jailbreak_large_vision-language_models_through_multi-modal_linkage.md)
- [\[ACL 2025\] SPHERE: Unveiling Spatial Blind Spots in Vision-Language Models Through Hierarchical Evaluation](sphere_unveiling_spatial_blind_spots_in.md)

</div>

<!-- RELATED:END -->
