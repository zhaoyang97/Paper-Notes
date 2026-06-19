---
title: >-
  [论文解读] OpenVision: A Fully-Open, Cost-Effective Family of Advanced Vision Encoders for Multimodal Learning
description: >-
  [ICCV 2025][多模态VLM][CLIP] 本文发布 OpenVision——一个完全开源（数据、训练代码、权重）的视觉编码器家族（5.9M-632.1M参数），基于 CLIPS 框架和 Recap-DataComp-1B 数据集训练，在集成到 LLaVA 等多模态框架时匹配甚至超越 OpenAI CLIP 和 Google SigLIP 的性能，为社区提供透明、灵活的视觉骨干替代方案。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "CLIP"
  - "视觉编码器"
  - "开源"
  - "多模态学习"
  - "LLaVA"
---

# OpenVision: A Fully-Open, Cost-Effective Family of Advanced Vision Encoders for Multimodal Learning

**会议**: ICCV 2025  
**arXiv**: [2505.04601](https://arxiv.org/abs/2505.04601)  
**代码**: [https://github.com/UCSC-VLAA/OpenVision](https://github.com/UCSC-VLAA/OpenVision)  
**领域**: 多模态VLM / 视觉编码器  
**关键词**: CLIP, 视觉编码器, 开源, 多模态学习, LLaVA

## 一句话总结

本文发布 OpenVision——一个完全开源（数据、训练代码、权重）的视觉编码器家族（5.9M-632.1M参数），基于 CLIPS 框架和 Recap-DataComp-1B 数据集训练，在集成到 LLaVA 等多模态框架时匹配甚至超越 OpenAI CLIP 和 Google SigLIP 的性能，为社区提供透明、灵活的视觉骨干替代方案。

## 研究背景与动机

**领域现状**：几乎所有多模态基础模型（LLaVA、Mini-GPT-4、Falcon2 VLM、Eagle 等）都依赖 OpenAI 的 CLIP-L/336 作为视觉编码器。这个"事实标准"统治了整个多模态领域。

**现有痛点**：
   - **不透明**：OpenAI CLIP 的训练数据和完整训练方案未公开，限制了可重复性和透明度
   - **规模受限**：仅有 Base 和 Large 两个参数规模，无法满足轻量级边缘部署或更大模型的探索需求
   - **已知缺陷**：CLIP 在空间关系理解和目标计数上存在已知幻觉问题
   - **开源替代不达标**：现有开源 CLIP（OpenCLIP、DataComp、DFN 等）虽然零样本性能好，但在多模态下游任务（MME、ChartQA、TextVQA）上明显落后

**核心矛盾**：社区需要一个完全开放的视觉骨干来推进多模态研究，但至今没有一个从头训练的开源编码器能在多模态基准上持续匹配或超越 OpenAI CLIP。

**本文目标** 提供一个完全开放（数据+代码+权重）、性价比高、多尺度的视觉编码器家族，使其在多模态学习场景下达到或超越闭源水平。

**切入角度**：基于最新的开源进展——CLIPS 训练框架（多正样本对比+文本生成器）和 Recap-DataComp-1B（重新标注的十亿级数据集），系统性分析关键设计选择。

**核心 idea**：站在 CLIPS+Recap-DataComp-1B 肩上，系统调优训练策略，发布25+个完全开源视觉编码器checkpoint，全面匹配/超越 OpenAI CLIP。

## 方法详解

### 整体框架

OpenVision 的训练流程遵循标准的双塔 CLIP 架构：视觉编码器+文本编码器，使用对比损失训练。关键增强来自 CLIPS 框架——引入多正样本损失（将原始标注和合成标注都视为正样本）并联合训练轻量文本解码器生成新标注。训练完成后仅保留视觉骨干，丢弃文本塔和解码器。

### 关键设计

1. **多阶段分辨率训练**

    - 功能：按照从低到高的分辨率阶段逐步训练，提高训练效率。
    - 核心思路：对于 Large/SoViT-400M/Huge 变体，依次在 $84 \times 84$、$224 \times 224$、$336 \times 336$（或 $384 \times 384$）三个阶段训练；更小的 Tiny/Small/Base 模型从 $160 \times 160$ 开始。三个阶段分别处理 12.8B、1.024B 和 256M 图文对，全局 batch size 分别为 32K、16K 和 8K。
    - 设计动机：遵循 CLIPA 的高效训练课程，低分辨率阶段快速学习语义表示，高分辨率阶段精炼细节，大幅降低计算成本。

2. **CLIPS 多正样本对比学习**

    - 功能：扩展标准 CLIP 的对比学习，将原始标注和合成标注都视为正样本。
    - 核心思路：CLIPS 使用 Recap-DataComp-1B（用 LLaVA 重新标注的 DataComp-1B）提供更丰富的合成标注，并联合训练文本解码器生成新标注，进一步丰富训练信号。
    - 设计动机：单一标注可能信息不足或有噪声，多正样本策略增加训练信号多样性，合成标注通常更详细准确。

3. **灵活的模型规模和 Patch 大小**

    - 功能：发布从 Ti（5.9M）到 H（632.1M）的完整编码器系列，并探索不同 patch 大小（8×8 vs 16×16）。
    - 核心思路：使用固定正弦余弦位置编码允许适应不同序列长度。更小 patch 提供更细的空间分辨率但计算成本更高。
    - 设计动机：满足从边缘设备到服务器的不同部署需求。8×8 patch 在 TextVQA 上带来显著提升（Tiny +4.4%，Small +5.0%，Base +3.3%）但大幅增加视觉 token 数量。

4. **合成标注和辅助解码器的作用**

    - 功能：消融分析合成标注和辅助文本解码器对编码器质量的贡献。
    - 核心思路：实验表明两者都对多模态下游任务有正面贡献。去掉合成标注在大多数基准上掉点，去掉辅助解码器同样有明显负面影响。
    - 设计动机：验证 CLIPS 框架中各组件对多模态场景的具体贡献。

### 损失函数 / 训练策略

- 对比损失 + 多正样本损失 + 文本生成损失（来自 CLIPS）
- 三阶段学习率：cosine 衰减，基础学习率依次为 $8 \times 10^{-6}$、$4 \times 10^{-7}$、$1 \times 10^{-7}$
- 文本编码器输入 80 token，文本解码器生成 128 token
- 下游评估使用 LLaVA-1.5（冻结编码器）和 Open-LLaVA-Next（全微调）两个框架

## 实验关键数据

### 主实验：LLaVA-1.5 框架下多模态性能对比

| 视觉编码器 | 分辨率 | TextVQA | ChartQA | OCR | MME | SEED | GQA | POPE |
|-----------|-------|---------|---------|-----|-----|------|-----|------|
| OpenAI-CLIP L/14 | 224 | 56.1 | 13.2 | 177 | 1443/306 | 66.0 | 60.8 | 85.0 |
| LAION-2B-CLIP L/14 | 224 | 54.2 | 12.8 | 165 | 1434/298 | 65.5 | 59.0 | 84.5 |
| DataComp-1B-CLIP L/14 | 224 | 53.0 | 12.3 | 131 | 1382/312 | 62.4 | 57.8 | 83.0 |
| DFN-2B-CLIP L/14 | 224 | 53.2 | 12.4 | 246 | 1447/306 | 65.6 | 59.1 | 85.0 |
| **OpenVision L/14** | **224** | **57.7** | **13.9** | **315** | **1487/317** | **69.5** | **62.9** | **86.4** |
| OpenAI-CLIP L/14 | 336 | 59.1 | 13.8 | 201 | 1475/288 | 67.5 | 61.1 | 85.7 |
| **OpenVision L/14** | **336** | **61.2** | **15.7** | **339** | **1525/315** | **70.5** | **63.7** | **87.2** |
| SigLIP SoViT-400M/14 | 384 | 62.6 | 14.5 | 338 | 1481/347 | 69.4 | 63.3 | 87.0 |
| **OpenVision SoViT-400M/14** | **384** | **62.4** | **16.1** | **357** | **1493/320** | **70.4** | **63.8** | **88.0** |

### 消融实验：不同 Patch 大小的影响（LLaVA-1.5）

| 编码器 | Patch | TextVQA | ChartQA | OCR | MME | SEED | GQA |
|--------|-------|---------|---------|-----|-----|------|-----|
| Ti/16 | 16 | 50.2 | 11.6 | 139 | 1329/280 | 62.0 | 58.0 |
| Ti/8 | 8 | **54.6** | **12.9** | **223** | **1383/310** | **66.3** | **59.7** |
| S/16 | 16 | 54.3 | 12.0 | 235 | 1393/343 | 67.5 | 61.6 |
| S/8 | 8 | **59.3** | **15.9** | **310** | **1449/303** | **70.3** | **62.0** |
| B/16 | 16 | 57.9 | 14.5 | 293 | 1432/333 | 69.8 | 62.8 |
| B/8 | 8 | **61.2** | **17.2** | **345** | **1545/299** | **71.8** | **63.0** |

### 关键发现
- **OpenVision 全面超越开源 CLIP 替代品**：在同等设置下，LAION-2B、DataComp-1B、DFN-2B 的 CLIP 在多模态任务上均明显落后于 OpenVision，尽管它们的零样本分类性能可能更高（如 DFN-2B 的 ImageNet 81.4% vs OpenVision 78.4%）。这说明**零样本分类性能不是多模态质量的充分指标**。
- **匹配/超越闭源**：OpenVision L/14@336 在大多数 LLaVA-1.5 基准上超越 OpenAI CLIP L/14@336，在 Open-LLaVA-Next 下也达到可比水平。
- **小 patch 大幅提升细粒度理解**：8×8 patch 在 TextVQA 和 OCR 上带来 +3~5% 提升，但以视觉 token 数量增加（计算成本翻4倍）为代价。
- **极小模型仍有竞争力**：Ti/16（5.9M）+ Qwen2.5-0.5B 的~250M 模型虽然性能下降，但保留了 87% 的平均表现。
- **扩大训练规模有益**：H/14（632.1M参数）在多个指标上进一步提升，证明更大编码器对多模态有价值。

## 亮点与洞察

- **完全透明性带来的价值**：当数据、代码、权重全部公开时，研究者才能做系统消融来理解"什么因素真正重要"。本文填补了开源视觉编码器在多模态场景下全面达标的空白。
- **CLIP 零样本性能 ≠ 多模态性能**：这是一个重要发现——DFN-2B CLIP 在 ImageNet 上高达 81.4% 但在多模态任务上明显不如 OpenVision（78.4%）。训练策略（如合成标注、多正样本损失）对多模态下游的影响比分类精度更大。
- **实用价值极高**：25+ checkpoints + 从 5.9M 到 632.1M 的完整规模谱系 + 不同 patch 大小 + 不同分辨率，为社区提供了多模态视觉骨干的完整"自助餐"。

## 局限与展望

- 训练数据（Recap-DataComp-1B）虽然开源，但重标注质量受限于用于标注的 LLaVA 模型能力。
- 未探索更先进的架构变体（如 EVA-CLIP 的架构改进）。
- 在视频理解等时序多模态任务上的评估缺失。
- 与最新的 SigLIP2 等更新模型的对比不够充分。
- 多模态评估主要基于 LLaVA 框架，在 Qwen-VL、InternVL 等其他框架上的表现未知。

## 相关工作与启发

- **vs OpenAI CLIP**: OpenAI 的 CLIP 训练数据不公开，仅有 B/L 两个规模；OpenVision 提供完全透明的训练流程和更丰富的模型规模。在多模态下游任务上性能匹配或超越。
- **vs SigLIP**: Google 的 SigLIP 训练数据同样不公开；OpenVision SoViT-400M/14 与 SigLIP 同等规模下性能相当甚至更好。
- **vs OpenCLIP/LAION/DataComp/DFN**: 这些开源 CLIP 在零样本分类上表现好，但在多模态下游任务上明显不足；OpenVision 通过 CLIPS 框架和合成标注弥补了这一差距。

## 评分

- 新颖性: ⭐⭐⭐ 主要是对已有技术（CLIPS+DataComp）的系统整合和工程优化，方法创新有限
- 实验充分度: ⭐⭐⭐⭐⭐ 多框架(LLaVA-1.5/Open-LLaVA-Next)、多规模、多分辨率、多patch的全面对比和消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰，表格信息密度高，结论明确
- 价值: ⭐⭐⭐⭐⭐ 对社区贡献巨大——25+开源checkpoint为多模态研究提供了急需的透明、灵活的视觉骨干

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Multimodal Autoregressive Pre-training of Large Vision Encoders](../../CVPR2025/multimodal_vlm/multimodal_autoregressive_pre-training_of_large_vision_encoders.md)
- [\[ICCV 2025\] Dynamic Multimodal Prototype Learning in Vision-Language Models](dynamic_multimodal_prototype_learning_in_vision-language_models.md)
- [\[ICCV 2025\] On Large Multimodal Models as Open-World Image Classifiers](on_large_multimodal_models_as_open-world_image_classifiers.md)
- [\[ICCV 2025\] Effective Training Data Synthesis for Improving MLLM Chart Understanding](effective_training_data_synthesis_for_improving_mllm_chart_understanding.md)
- [\[CVPR 2025\] BadVision: Stealthy Backdoor Attack in Self-Supervised Learning Vision Encoders for Large Vision Language Models](../../CVPR2025/multimodal_vlm/stealthy_backdoor_attack_in_self-supervised_learning_vision_encoders_for_large_v.md)

</div>

<!-- RELATED:END -->
