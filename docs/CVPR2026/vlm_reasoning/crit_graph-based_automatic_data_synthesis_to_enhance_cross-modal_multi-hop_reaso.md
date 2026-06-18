---
title: >-
  [论文解读] CRIT: Graph-Based Automatic Data Synthesis to Enhance Cross-Modal Multi-Hop Reasoning
description: >-
  [CVPR 2026][多模态VLM][跨模态] 提出基于图结构的自动数据生成 pipeline，构建了 CRIT 数据集与 benchmark，用于训练和评测 VLM 在交错图文内容上的跨模态多跳推理能力，训练后的模型在 SPIQA 等多个基准上取得显著提升。 现实世界推理常需跨模态整合信息：例如阅读 DIY 教程时需不断…
tags:
  - "CVPR 2026"
  - "多模态VLM"
  - "跨模态"
  - "Multi-Hop Reasoning"
  - "Data Synthesis"
  - "Graph-Based Pipeline"
  - "VLM Benchmark"
---

# CRIT: Graph-Based Automatic Data Synthesis to Enhance Cross-Modal Multi-Hop Reasoning

**会议**: CVPR 2026  
**arXiv**: [2604.01634](https://arxiv.org/abs/2604.01634)  
**代码**: 无  
**领域**: Multimodal VLM  
**关键词**: Cross-Modal Reasoning, Multi-Hop Reasoning, Data Synthesis, Graph-Based Pipeline, VLM Benchmark

## 一句话总结

提出基于图结构的自动数据生成 pipeline，构建了 CRIT 数据集与 benchmark，用于训练和评测 VLM 在交错图文内容上的跨模态多跳推理能力，训练后的模型在 SPIQA 等多个基准上取得显著提升。

## 研究背景与动机

现实世界推理常需跨模态整合信息：例如阅读 DIY 教程时需不断在文字指令和配图之间交叉参照。然而，现有多模态基准存在严重缺陷：

**评估端**：大多数 benchmark 仅涉及单图或一组图片，答案往往可从单一模态推断，无法测试真正的跨模态推理

**训练端**：虽然大量交错图文数据用于预训练，但其中真正需要互补跨模态推理的数据极少

**模型端**：即使 SOTA 模型（GPT-4o）在需要 CoT 推理时，也经常产生与视觉/文本证据脱节的幻觉

直接用 VLM 生成复杂推理数据会引入循环偏差（用同类模型生成和评测）和幻觉问题。本文通过图结构作为中间表示，**全程仅需 LLM（无需 VLM）**即可生成问答对，避免了上述问题。

## 方法详解

### 整体框架

CRIT 要解决的是跨模态多跳推理「既缺训练数据又缺评测基准」的双重短板，而直接用 VLM 生成这类数据又会带来循环偏差（同类模型自产自评）和幻觉。它的关键思路是引入**图结构作为中间表示**，让全流程仅靠 LLM（无需 VLM）就能造出真正需要图文互补的问答对：先把带场景图标注的图像构建成统一的多模态内容图，再基于子图生成互补性文本，最后采样跨模态子图链生成多跳 QA。整条管线还能通过适配「图结构构建」这一步，把数据源从自然图像扩展到视频帧和科学论文，最终产出的 CRIT 数据集既作为训练集（LoRA SFT）也作为评测 benchmark。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["带场景图标注的图像<br/>（随机采样 1–6 张）"] --> B
    A2["扩展到视频和科学论文<br/>视频帧→字幕转场景图；论文→段落/图表转统一图"] --> B
    B["多模态内容图<br/>规则过滤唯一实体→合并场景图→LLM 加桥接文本节点"] --> C
    C["互补性文本生成<br/>抽子图（排除图像属性/跨图关系）→LLM 多风格叙事"] --> D
    D["QA 生成与多层过滤<br/>采样 1–5 跳跨模态子图链→LLM 生成问题+CoT→三层过滤"] --> E
    E["CRIT 数据集 + Benchmark"] --> F["LoRA SFT 训练 VLM"]
```

### 关键设计

**1. 多模态内容图：用可编程的图结构锁死「跨模态多跳」约束**

直接 prompt VLM 生成数据，难以保证问题真的需要跨图、跨模态推理。CRIT 把内容组织成有向图 $G=(\mathcal{V}, \mathcal{E})$，节点是实体（视觉对象或文本实体）、边是关系：随机采样 1-6 张带场景图标注的图像，用规则过滤只保留能被属性或关系唯一标识的实体以避免歧义，再用 LLM 为每个图像节点生成新的文本实体和关系作为跨图像的桥接节点。这样多跳和跨模态的约束就被编程式地写进了图里，而非靠运气。

**2. 互补性文本生成：让文本只补图像没说的，逼出真正的跨模态依赖**

如果文本把图里的信息也说全了，问题就退化成单模态可答。CRIT 为每张图像提取关联子图，但**排除图像节点的属性和跨图关系**（这些留给模型推理时从图像里取），再让 LLM 以故事、日记、纪录片等多种叙事风格生成文本，且只描述增强的文本节点及其与图像节点的连接、不泄露需从图像推理的信息。文本与图像因此严格互补，缺一不可。

**3. QA 生成与多层过滤：保证每个问题都「非跨模态不可答」**

采样包含 1-5 条边的跨模态子图链、且终端节点必须来自图像，LLM 根据序列化的子图 JSON 和目标答案生成问题（约束中间实体不可在问题里直接提及），同时产出 CoT 推理链。再叠三层过滤：(a) 剔除问题中显式提到中间实体的样本；(b) 用 3 个不同 LLM 检验「单模态即可回答」的问题并删除；(c) 修剪过长的 CoT。多重把关确保留下的题目确实依赖跨模态多跳。

**4. 扩展到视频和科学论文：同一图范式迁移到新模态**

整条管线只需适配「图结构构建」这一步即可扩展：视频上利用密集字幕数据集、挑与字幕高 CLIP 相似度的帧，再让 LLM 把字幕转成场景图；科学论文上把段落/图表/表格转成统一图结构，标记视觉实体后从文本中移除对应描述。这使方法从自然图像一路覆盖到视频和论文域。

### 损失函数 / 训练策略

- 使用 LoRA 对 Qwen2.5-VL-7B 和 Idefics2-8B 进行 SFT
- 每个训练样本同时包含直接回答和 CoT 两种格式
- 数据生成 LLM：Qwen3-30B-A3B-Instruct-2507
- 过滤 LLM：Qwen3-30B + Gemma-3-27b-it + Mistral-Small-3.2-24B

## 实验关键数据

### 主实验

CRIT Benchmark 结果（CoT 评测，EM/F1）：

| 模型 | NI-EM | NI-F1 | VF-EM | VF-F1 | SP-EM | SP-F1 |
|------|-------|-------|-------|-------|-------|-------|
| GPT-4o | 35.1 | 37.7 | 32.0 | 38.9 | 8.4 | 14.0 |
| Qwen2.5-VL-7B | 28.3 | 29.1 | 24.0 | 27.8 | 6.8 | 9.6 |
| Qwen2.5-VL-72B | 38.0 | 39.4 | 30.1 | 33.9 | 9.4 | 12.3 |
| **Qwen2.5-VL_CRIT** | **58.6** | **59.5** | **38.8** | **42.2** | **15.9** | **22.5** |
| **Idefics2_CRIT** | **54.1** | **54.9** | **31.2** | **33.9** | **12.3** | **20.2** |

训练后的 7B 模型大幅超越 GPT-4o 和 72B 模型。

跨基准迁移效果（Idefics2 + Mantis-Instruct + CRIT vs. Mantis-Instruct only）：

| 基准 | 指标 | +CRIT | 仅 Mantis | 提升 |
|------|------|-------|-----------|------|
| SPIQA | METEOR | 10.53 | 3.60 | +192% |
| SPIQA | CIDEr | 67.93 | 23.83 | +185% |
| VEGA | ROUGE-L | 35.1 | 29.5 | +19% |
| MMQA | EM | 30.0 | 27.3 | +10% |
| FCMR | F1 | 50.5 | 44.9 | +12% |

### 消融实验

| 配置 | NI-EM | VF-EM | SP-EM | 说明 |
|------|-------|-------|-------|------|
| No Fine-tuning | 28.3 | 24.0 | 6.8 | 基线 |
| CRIT (84k) | 58.6 | 38.8 | 15.9 | 标准训练集 |
| CRIT Augmented (210k) | 62.6 | 45.6 | 16.7 | 扩展训练集，视频域提升最大 |

使用模型生成标注的扩展数据能进一步提升性能，且科学论文域也受益于自然图像/视频域的数据扩展（跨域迁移）。

### 关键发现

- **SOTA 模型在跨模态多跳推理上表现很差**：GPT-4o 在自然图像域仅 35.1% EM，科学论文域仅 8.4%
- **错误分析**（75 个 GPT-4o 错误样本）：55% 为证据定位错误（模型找错了图片或文本段落），视觉感知错误是文本理解错误的 4 倍
- **训练后不损害通用能力**：加入 CRIT 后在 MME、SeedBench 等通用基准上保持甚至提升性能

## 亮点与洞察

1. **图结构作为中间表示的设计极为精巧**：通过子图采样可编程式地保证多跳、跨模态约束，比直接 prompt VLM 生成数据质量高得多
2. **全程无需 VLM，仅用 LLM**：避免了用 VLM 生成 VLM 评测数据的循环偏差问题
3. **单模态过滤设计巧妙**：用 3 个不同 LLM 分别验证文本和视觉模态，确保问题确实需要跨模态推理
4. **管道高度可扩展**：从标注图像扩展到视频帧和科学论文，仅需适配图结构构建阶段

## 局限与展望

- 科学论文域表现仍然较低（15.9% EM），长文本 + 复杂图表的精确跨模态对齐仍是挑战
- 图结构构建依赖已有的场景图标注（GQA）或密集字幕标注（ActivityNet），完全无标注场景的适用性有待验证
- 当前仅评测了手动验证的 1,446 个测试样本，规模相对有限
- 未探索 CoT 推理链质量对训练效果的影响

## 相关工作与启发

- 图结构中间表示 → LLM 生成 QA 的范式可推广到其他需要结构化推理的数据合成任务
- "互补性"约束（排除图像属性和跨图关系从文本中泄露）是保证跨模态推理质量的关键
- 错误分析揭示"证据定位"是当前 VLM 最大瓶颈，而非推理能力本身

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 图结构数据生成 pipeline 设计精巧，解决了数据合成中的循环偏差问题
- 实验充分度: ⭐⭐⭐⭐ — 多模型对比 + 多基准迁移 + 数据扩展 + 错误分析
- 写作质量: ⭐⭐⭐⭐ — 三阶段 pipeline 描述清晰，Fig.2 的流程图信息量大
- 价值: ⭐⭐⭐⭐⭐ — 开创性地定义并解决了跨模态多跳推理的数据和评测瓶颈

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Socratic-Geo: Synthetic Data Generation and Cross-Modal Geometric Reasoning via Multi-Agent Interaction](socratic-geo_synthetic_data_generation_and_cross-modal_geometric_reasoning_via_m.md)
- [\[ACL 2025\] FCMR: Robust Evaluation of Financial Cross-Modal Multi-Hop Reasoning](../../ACL2025/multimodal_vlm/fcmr_robust_evaluation_of_financial_cross-modal_multi-hop_reasoning.md)
- [\[CVPR 2026\] Multi-Crit: Benchmarking Multimodal Judges on Pluralistic Criteria-Following](multi-crit_benchmarking_multimodal_judges_on_pluralistic_criteria-following.md)
- [\[CVPR 2026\] Hierarchical Attacks for Multi-Modal Multi-Agent Reasoning](hierarchical_attacks_for_multi-modal_multi-agent_reasoning.md)
- [\[CVPR 2026\] OmniZip: Learning a Unified and Lightweight Lossless Compressor for Multi-Modal Data](omnizip_learning_a_unified_and_lightweight_lossless_compressor_for_multi-modal_d.md)

</div>

<!-- RELATED:END -->
