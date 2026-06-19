---
title: >-
  [论文解读] MIMO: A Medical Vision Language Model with Visual Referring Multimodal Input and Pixel Grounding Multimodal Output
description: >-
  [CVPR 2025][多模态VLM][医学视觉问答] 本文提出MIMO——首个同时支持"视觉引用多模态输入"（用户通过点/框指定感兴趣区域）和"像素级定位多模态输出"（模型在文本回答中嵌入分割mask）的医学视觉语言模型，并构建了895K样本的MIMOSeg数据集，在多种医学VQA和分割任务上展示了独特的referring+grounding能力。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "医学视觉问答"
  - "视觉引用"
  - "像素级定位"
  - "多模态输入输出"
  - "医学图像分割"
---

# MIMO: A Medical Vision Language Model with Visual Referring Multimodal Input and Pixel Grounding Multimodal Output

**会议**: CVPR 2025  
**arXiv**: [2510.10011](https://arxiv.org/abs/2510.10011)  
**代码**: [https://github.com/pkusixspace/MIMO](https://github.com/pkusixspace/MIMO)  
**领域**: 多模态VLM  
**关键词**: 医学视觉问答, 视觉引用, 像素级定位, 多模态输入输出, 医学图像分割

## 一句话总结

本文提出MIMO——首个同时支持"视觉引用多模态输入"（用户通过点/框指定感兴趣区域）和"像素级定位多模态输出"（模型在文本回答中嵌入分割mask）的医学视觉语言模型，并构建了895K样本的MIMOSeg数据集，在多种医学VQA和分割任务上展示了独特的referring+grounding能力。

## 研究背景与动机

当前医学视觉语言模型(MVLM)在VQA中主要依赖纯文本指令输入+纯文本回答输出。→ 痛点1（输入侧）：医学图像中精确描述感兴趣区域非常困难，如"右肺中下叶肺门区"这种术语对非专业用户不友好，且不同患者的影像表现存在个体差异。→ 痛点2（输出侧）：纯文本回答缺乏与图像关键区域的关联，无法像医生之间的交流那样"边指边说"。→ 核心矛盾：缺乏统一模型架构同时支持visual referring输入和pixel grounding输出，且缺少对应的医学多模态训练数据。→ 本文切入角度：设计MIMO架构（Multi-modal Input Aligner + grounding tokens + SAM解码器）+ 构建MIMOSeg数据集（895K，覆盖4个视角），从输入端和输出端同时丰富医学VLM的多模态能力。

## 方法详解

### 整体框架

MIMO由6个核心模块组成：(1) CLIP ViT-H/14图像编码器产生图像嵌入 $X_{img} \in \mathbb{R}^{l_1 \times d}$；(2) 视觉提示编码器将点/框编码为 $X_q^v \in \mathbb{R}^{l_3 \times d}$；(3) Multi-modal Input Aligner通过交叉注意力融合多模态输入；(4) Vicuna-7B LLM生成含特殊token的文本回答；(5) SAM分割编码器提取图像分割特征；(6) SAM mask解码器根据 `<SEG>` token的embedding生成分割mask。

### 关键设计

1. **Multi-modal Input Aligner（多模态输入对齐器）**:
    - 功能：从多模态输入中提取指令引导的视觉信息
    - 核心思路：使用可学习query embedding $X_q$ 通过交叉注意力与 $(X_{img}, X_q^t, X_q^v)$ 交互，提取文本和视觉提示共同关注的图像特征，经线性投影后送入LLM
    - 设计动机：直接拼接多模态特征无法有效对齐视觉提示与图像区域；cross-attention可以让模型选择性关注视觉提示指向区域；消融实验表明移除后在含视觉提示的视角II/IV中性能显著下降（mIoU下降0.04~0.06）

2. **Grounding Token机制（语言到像素的映射）**:
    - 功能：在LLM文本输出中标记可定位的医学实体，并关联到分割mask
    - 核心思路：在词表中引入 `<p>`, `</p>`, `<SEG>` 三个特殊token；`<p>...</p>` 包裹可定位实体，`<SEG>` 紧跟其后表示需要分割；提取 `<SEG>` 对应的LLM末层隐状态 $r_{seg}$，通过投影层送入SAM mask解码器：$\mathcal{M} = \mathcal{V}(\mathcal{G}(I), \text{proj}(r_{seg}))$
    - 设计动机：借鉴LISA/GLaMM的做法将LLM的语义理解与像素级分割无缝连接，使文本回答中每个医学实体都可"落地"到图像区域

3. **MIMOSeg数据集构建（895K样本，4个视角）**:
    - 功能：提供覆盖8种影像模态（CT、X光、眼底、病理等）的大规模训练数据
    - 核心思路：设计4个递进难度的视角——I: 语言引导分割（纯文本→mask）；II: 视觉提示感知（文本+视觉→mask+实体识别）；III: 带分割的复杂推理问答；IV: 视觉提示辅助的推理+分割。通过知识库检索（Wikipedia/UMLS）+ GPT-4o生成复杂场景的QA
    - 设计动机：医学领域无同时包含referring和grounding标注的现成数据集；4个视角覆盖从基础到复杂的全部交互场景

### 损失函数 / 训练策略

总损失 $L_{total} = \lambda_1 L_{text} + \lambda_2 L_{bce} + \lambda_3 L_{dice}$：
- $L_{text}$：文本生成交叉熵损失
- $L_{bce}$：像素级二元交叉熵损失（分割mask监督）
- $L_{dice}$：Dice损失（处理前景/背景不平衡）
- 训练数据混合比：4个视角 + LLaVA-Med VQA = 1:2:2:1:1，训练3个epoch

## 实验关键数据

### 主实验（Held-in：MIMOSeg测试集，分割性能）

| 模型 | 视角I(mIoU) | 视角II(mIoU) | 视角III(mIoU) | 视角IV(mIoU) |
|------|------------|-------------|--------------|-------------|
| SAM-h | × | 0.571 | × | 0.583 |
| GLaMM | 0.556 | 0.496 | 0.421 | 0.564 |
| MIMO(w/o Aligner) | 0.607 | 0.622 | 0.468 | 0.526 |
| **MIMO** | **0.639** | **0.665** | **0.531** | **0.586** |

### Held-out零样本分割实验

| 数据集 | 条件 | MIMO(AP50) | GLaMM(AP50) | SAM-h(AP50) |
|--------|------|-----------|-------------|-------------|
| X-ray | w/o bbox | **0.507** | 0.335 | × |
| X-ray | with bbox | **0.989** | 0.859 | 0.989 |
| Fundus | with bbox | **0.988** | 0.946 | 0.973 |
| Skin Lesion | w/o bbox | **0.787** | 0.458 | × |
| Skin Lesion | with bbox | **0.985** | 0.719 | 0.982 |

### 消融实验

| 配置 | 视角II(mIoU) | 视角IV(mIoU) | 说明 |
|------|-------------|-------------|------|
| MIMO | **0.665** | **0.586** | 完整模型 |
| MIMO(w/o Aligner) | 0.622 | 0.526 | 移除对齐器，视觉提示场景性能下降明显 |

| 训练数据比例 | mIoU趋势 | AP50趋势 | F1趋势 |
|-------------|---------|---------|--------|
| 25%→50%→75%→100% | 稳步提升 | 稳步提升 | 稳步提升 |

### 关键发现

- MIMO是唯一能同时做visual referring + pixel grounding的医学VLM
- 在纯文本指令分割（w/o bbox）场景下MIMO大幅超越GLaMM，表明其文本指令跟随能力更强
- Held-out VQA平均准确率56.1%，不及HuatuoGPT-Vision（66.1%），但后者使用了更多VQA训练数据且不支持分割
- Multi-modal Input Aligner在含视觉提示的场景中带来一致的增益

## 亮点与洞察

- **首创性**：首个在医学VLM中同时实现visual referring输入和pixel grounding输出的统一模型
- **数据构建方法论**：4个视角覆盖从基础到复杂的交互场景，知识库+GPT-4o的生成流程可复用
- **实用性强**：`<p>平滑肌<SEG></p>` 这种文本+分割无缝结合的输出非常符合医生的实际工作流
- **开源**：模型和数据集均已开源

## 局限与展望

- Held-out VQA性能不及HuatuoGPT-Vision，通用VQA能力被分割任务稀释
- QA生成依赖预定义知识库，知识覆盖范围有限
- 仅使用7B的Vicuna作为LLM backbone，更大模型可能效果更好
- 视觉提示仅支持点和框，不支持自由形状涂鸦（如Ferret的free-form）
- 分割mask只配对单个实体，不支持复杂的多实体关系建模

## 相关工作与启发

- **GLaMM**：通用域的referring+grounding多模态模型，MIMO将思路迁移到医学场景并加入Multi-modal Input Aligner
- **LISA**：推理分割模型，用`<SEG>` token连接LLM和SAM，MIMO借鉴了这一核心设计
- **Ferret**：支持自由形状referring的通用VLM，启发了MIMO的视觉提示设计
- 启发：医学VLM的"输入-输出模态丰富度"是被忽视的重要方向；多视角数据集构建的方法论可推广

## 评分
- 新颖性: ⭐⭐⭐⭐ 在医学领域首创referring+grounding统一模型，但技术组件多为已有方法组合
- 实验充分度: ⭐⭐⭐⭐ 覆盖held-in/held-out、分割/VQA多维评估，消融也较充分
- 写作质量: ⭐⭐⭐⭐ 4个视角设计逻辑清晰，图示丰富
- 价值: ⭐⭐⭐⭐ 对医学AI有直接应用价值，数据集贡献重要

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] VideoGLaMM: A Large Multimodal Model for Pixel-Level Visual Grounding in Videos](videoglamm_a_large_multimodal_model_for_pixel-level_visual_grounding_in_videos.md)
- [\[CVPR 2025\] Your Large Vision-Language Model Only Needs a Few Attention Heads for Visual Grounding](your_large_vision-language_model_only_needs_a_few_attention_heads_for_visual_gro.md)
- [\[ICCV 2025\] DOGR: Towards Versatile Visual Document Grounding and Referring](../../ICCV2025/multimodal_vlm/dogr_towards_versatile_visual_document_grounding_and_referring.md)
- [\[CVPR 2025\] ReVisionLLM: Recursive Vision-Language Model for Temporal Grounding in Hour-Long Videos](revisionllm_recursive_vision-language_model_for_temporal_grounding_in_hour-long_.md)
- [\[NeurIPS 2025\] BridgeVLA: Input-Output Alignment for Efficient 3D Manipulation Learning with Vision-Language Models](../../NeurIPS2025/multimodal_vlm/bridgevla_input-output_alignment_for_efficient_3d_manipulation_learning_with_vis.md)

</div>

<!-- RELATED:END -->
