---
title: >-
  [论文解读] ADIEE: Automatic Dataset Creation and Scorer for Instruction-Guided Image Editing Evaluation
description: >-
  [ICCV 2025][图像生成][image editing evaluation] 提出ADIEE，通过自动化方法构建超过10万样本的图像编辑评估训练数据集，微调LLaVA-NeXT-8B作为编辑质量评分模型，在多个基准上超越开源VLM和Gemini-Pro 1.5，并可作为奖励模型提升编辑模型性能。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "image editing evaluation"
  - "automated scoring"
  - "VLM fine-tuning"
  - "reward model"
  - "instruction-guided editing"
---

# ADIEE: Automatic Dataset Creation and Scorer for Instruction-Guided Image Editing Evaluation

**会议**: ICCV 2025  
**代码**: [GitHub](https://github.com/SherryXTChen/ADIEE.git)  
**领域**: 图像编辑评估 / 视觉语言模型  
**关键词**: image editing evaluation, automated scoring, VLM fine-tuning, reward model, instruction-guided editing

## 一句话总结

提出ADIEE，通过自动化方法构建超过10万样本的图像编辑评估训练数据集，微调LLaVA-NeXT-8B作为编辑质量评分模型，在多个基准上超越开源VLM和Gemini-Pro 1.5，并可作为奖励模型提升编辑模型性能。

## 研究背景与动机

指令引导的图像编辑快速发展，但有效的自动化评估仍是痛点。传统指标（MSE、PSNR、SSIM、LPIPS）无法捕捉语义对齐；CLIP相关指标难以理解长文本和组合关系。VLM作为评判者是有前景的方向，但开源模型与人类判断对齐差，闭源模型（如GPT-4o）不透明且成本高。最关键的瓶颈在于：**没有公开的训练数据集**来微调开源VLM作为编辑评分器——现有的只有小规模评估基准，各自采用不同的评估方案。

## 方法详解

### 整体框架

ADIEE包含三部分：(1) 基于编辑模型的训练数据生成——利用9种文本引导编辑方法对MagicBrush数据集生成多样化编辑输出并启发式打分；(2) 基于多轮编辑序列的数据生成——利用SEED-Data-Edit的多轮序列构建连续分数样本；(3) 评分模型架构——在LLaVA-NeXT-8B上扩展[SCORE] token，通过MLP解码器输出连续评分。训练后的评分器还可作为奖励模型微调编辑模型。

### 关键设计

1. **基于编辑模型的自动标注**: 利用图像编辑训练数据集天然包含正样本（ground-truth编辑结果，高分）和负样本（原始输入图像作为编辑输出，低分）。用9种编辑方法生成中间质量样本，通过CLIP方向相似度、CLIP图像相似度和DINO相似度的组合启发式规则自动分配0/0.5/1三级分数。还微调了Qwen2-VL生成与编辑指令对齐的输入/目标caption，解决了现有VLM生成的caption与编辑意图不一致的问题。

2. **多轮编辑序列数据增强**: 在多轮编辑序列S = [I₀, p₁, I₁, ..., pₗ, Iₗ]中，任意两个位置的图像对都可以构成评估样本。中间帧的分数按完成编辑步骤的比例线性分配f = (k-j₁)/(j₂-j₁)，实现了连续分数分布的训练数据，而非仅有0/0.5/1三个离散值。

3. **VLM评分架构与奖励模型**: 在LLaVA-NeXT-8B的词表中添加[SCORE]特殊token，其嵌入通过MLP解码器映射为连续评分值。支持两种奖励微调方式：(a) 奖励条件训练——将评分附加到编辑指令后，推理时设为最高分；(b) 奖励反馈学习——将(10-评分)作为额外损失项直接参与编辑模型梯度更新。

### 损失函数 / 训练策略

评分模型在6×H100 GPU上训练30 epoch，batch size 192，LoRA rank 8，学习率2e-5。奖励微调MagicBrush模型10000步，batch size 384，学习率5e-5。数据集共93,915个编辑模型生成样本 + 21,382个多轮序列。

## 实验关键数据

### 主实验

| 方法 | AURORA-Bench相关性 | GenAI-Bench准确率 | AURORA-Bench准确率 |
|------|-------------------|------------------|-------------------|
| CLIP-D | 0.3080 | - | - |
| Gemini-Pro 1.5 | 0.4038 | 55.87% | 50.76% |
| InternVL2-8B | 0.3342 | 52.43% | 45.07% |
| **ADIEE** | **0.4734** | **59.90%** | **55.51%** |

与SOTA相比：AURORA-Bench相关性+0.0696(+17.24%)，GenAI-Bench准确率+4.03%(+7.21%)，AURORA-Bench配对准确率+4.75%(+9.35%)。

### 消融实验

- 奖励模型微调MagicBrush：ImagenHub平均分从5.90提升至6.43(+8.98%)
- 多轮序列数据贡献了连续分数分布，显著提升了评分粒度
- 编辑prompt微调（VLM caption对齐）对数据质量至关重要

### 关键发现

- 编辑训练数据集可以"免费"转化为评估训练数据，无需额外人工标注
- 连续分数分布（来自多轮序列）优于离散分数
- 评分器作为奖励模型的效果显著，验证了评估与生成的闭环

## 亮点与洞察

- 数据构建思路巧妙——利用现有编辑数据集的固有结构免费获取评估训练数据
- 多轮序列的连续分数分配方式简洁而有效
- 评估→奖励模型→改进编辑模型的闭环验证了方法的实用价值
- [SCORE] token设计简洁，避免了从自然语言中解析分数的不稳定性

## 局限与展望

- 自动分数分配依赖CLIP/DINO的启发式规则，可能引入偏差
- 仅在instruction-guided编辑上验证，未涉及text-guided（需prompt pair）编辑的评估
- 评分模型基于LLaVA-NeXT-8B，推理开销较大，不适合大规模实时评估
- 训练数据主要来自MagicBrush，泛化到其他编辑分布的能力待验证

## 相关工作与启发

- VIEScore、ImagenHub、AURORA-Bench等评估基准提供了评估标准
- 奖励模型思路与RLHF类似，但直接用评分而非偏好对
- 自动化数据构建思路可推广到视频编辑、3D编辑等领域的评估

## 评分

- 新颖性: ⭐⭐⭐⭐ — 自动化评估数据构建思路新颖实用
- 技术深度: ⭐⭐⭐ — 方法核心是启发式数据构建+标准微调
- 实验充分性: ⭐⭐⭐⭐ — 多基准、多模型对比完整
- 写作质量: ⭐⭐⭐⭐ — 数据构建流程描述清晰
- 实用价值: ⭐⭐⭐⭐⭐ — 评分器和数据集都开源，直接可用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Early Timestep Zero-Shot Candidate Selection for Instruction-Guided Image Editing](early_timestep_zero-shot_candidate_selection_for_instruction-guided_image_editin.md)
- [\[ICCV 2025\] SuperEdit: Rectifying and Facilitating Supervision for Instruction-Based Image Editing](superedit_rectifying_and_facilitating_supervision_for_instruction-based_image_ed.md)
- [\[ICCV 2025\] AID: Adapting Image2Video Diffusion Models for Instruction-guided Video Prediction](aid_adapting_image2video_diffusion_models_for_instruction-guided_video_predictio.md)
- [\[ICCV 2025\] CAP: Evaluation of Persuasive and Creative Image Generation](cap_evaluation_of_persuasive_and_creative_image_generation.md)
- [\[ICCV 2025\] LUSD: Localized Update Score Distillation for Text-Guided Image Editing](lusd_localized_update_score_distillation_for_text-guided_image_editing.md)

</div>

<!-- RELATED:END -->
