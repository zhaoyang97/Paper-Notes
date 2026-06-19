---
title: >-
  [论文解读] Img-Diff: Contrastive Data Synthesis for Multimodal Large Language Models
description: >-
  [CVPR 2025][多模态VLM][对比数据合成] 提出一种受对比学习启发的数据合成方法，自动生成包含细微物体差异的相似图像对及其差异描述，用于微调MLLM后在MMVP上超越GPT-4V和Gemini达12个点，并在8个通用MLLM基准上平均提升3.06%。 领域现状：多模态大语言模型（MLLM）的性能高度依赖训练数据质…
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "对比数据合成"
  - "图像差异描述"
  - "多模态大语言模型"
  - "细粒度视觉理解"
  - "数据增强"
---

# Img-Diff: Contrastive Data Synthesis for Multimodal Large Language Models

**会议**: CVPR 2025  
**arXiv**: [2408.04594](https://arxiv.org/abs/2408.04594)  
**代码**: [https://github.com/modelscope/data-juicer/tree/ImgDiff](https://github.com/modelscope/data-juicer/tree/ImgDiff)  
**领域**: 多模态VLM  
**关键词**: 对比数据合成、图像差异描述、多模态大语言模型、细粒度视觉理解、数据增强

## 一句话总结
提出一种受对比学习启发的数据合成方法，自动生成包含细微物体差异的相似图像对及其差异描述，用于微调MLLM后在MMVP上超越GPT-4V和Gemini达12个点，并在8个通用MLLM基准上平均提升3.06%。

## 研究背景与动机

**领域现状**：多模态大语言模型（MLLM）的性能高度依赖训练数据质量。当前MLLM的训练数据主要包含预训练阶段的图-文对和微调阶段的VQA指令数据，在细粒度图像识别上仍有短板。

**现有痛点**：(1) 现有VQA数据集关注单张图的理解，缺少对相似图像间细微差异的辨别训练；(2) CLIP等视觉编码器存在"CLIP-blind"问题——对CLIP特征相似但内容不同的图像无法区分；(3) 已有图像差异数据集（Spot-the-Diff、CLEVR-Change）规模小、领域窄，且差异描述覆盖整张图不够精确。

**核心矛盾**：要提升MLLM的细粒度视觉辨别能力，需要大量"相似但不同"的图像对数据，但人工标注成本极高，且现有合成方法（如InstructPix2Pix）的差异描述不够精确、缺乏区域聚焦。

**本文解决什么**：设计一套全自动的数据合成流程，生成高质量的"物体替换"图像对及精确的区域级差异描述。

**切入角度**：借鉴对比学习的思路——让模型通过比较相似图像来学习分辨细节差异。

**核心 idea**：用SDXL+Prompt-to-Prompt生成相似图像对，经多级过滤提取差异区域，由MLLM生成精确的区域级差异描述，形成高质量对比训练数据。

## 方法详解

### 整体框架
三步流程：(1) 图像对生成——用LLM做物体替换产生caption对，用SDXL+Prompt-to-Prompt生成相似图像对；(2) 差异区域生成器（Difference Area Generator）——通过图像相似度过滤、FastSAM分割、图文匹配过滤、差异检测定位物体差异区域；(3) 差异描述生成器（Difference Captions Generator）——两阶段流程用MLLM标注区域内容并生成精确的差异描述。

### 关键设计

1. **图像对生成（Prompt-to-Prompt + SDXL）**:
    - 功能：生成仅有少量物体被替换的高度相似图像对
    - 核心思路：(i) 从caption数据库（MSCOCO/LLaVA预训练数据）获取图像描述；(ii) 用LLM替换描述中的一个物体名（如"dog"→"cat"）；(iii) 基于原始和替换后的caption，使用Stable-Diffusion-XL配合Prompt-to-Prompt技术生成图像对——该技术通过共享cross-attention层确保只改变被替换物体，其余保持一致
    - 设计动机：直接对比InstructPix2Pix的方法：(1)使用更先进的SDXL（而非SD-1.5）生成更真实的图像；(2)Prompt-to-Prompt比编辑方法更好地控制变化范围。从118K图像对中，经过过滤最终得到38,533个高度相似但非完全相同的图像对

2. **差异区域生成器（Difference Area Generator）**:
    - 功能：精确定位图像对中物体差异的区域（bounding box）
    - 核心思路：四步级联过滤：(i) **图像相似度过滤**：用CLIP计算图像对余弦相似度，保留相似度在阈值范围内的对（太相似=无差异，太不同=整体不同）；(ii) **FastSAM分割**：对每张图做实例分割获取所有bounding box；(iii) **图文匹配过滤**：用BLIP对裁剪的子图和物体名做图文匹配，确认子图中确实包含目标物体；(iv) **差异检测**：对同一bounding box位置裁剪两张图，用CLIP计算子图相似度，仅保留差异显著的区域，再做IoU过滤去除重叠框
    - 设计动机：不直接用物体检测（如YOLO）是因为物体类别有限；基于分割+相似度比较的方案不受已知类别限制，能检测任意物体的差异。117,779个有效bounding box经多级过滤确保质量

3. **差异描述生成器（Difference Captions Generator）**:
    - 功能：为差异区域生成精确的文本描述
    - 核心思路：**Stage-1（物体标注+过滤）**：选取相似度最低的top-5 bounding box，用 LLaVA-NEXT 描述每个区域内容，再通过图文匹配过滤和描述相似度过滤（用CLIP计算两个描述的相似度，太相似说明没差异）。**Stage-2（差异描述生成）**：在图像上画红框标记差异区域，将两张图+区域描述+红框图一起输入LLaVA-NEXT，让它描述红框区域内两张图的具体差异
    - 设计动机：关键创新在于**区域级差异描述**而非全图差异——全图描述容易遗漏或笼统，红框标注+局部描述确保每个差异都被精确捕捉。最终从117,779个bounding box中筛选出12,688个高质量"物体替换"样本

### 损失函数 / 训练策略
- 将 Img-Diff 数据混入 MLLM 原始视觉指令微调数据进行微调
- 对 LLaVA-1.5 和 MGM：混合后重新微调
- 对 InternVL2：在原始微调基础上做第二次微调
- 在 Spot-the-Diff 和 Image-Edit-Request 上再做 2 个 epoch 的领域适配微调
- 无额外损失函数设计，使用标准的 VQA 训练目标

## 实验关键数据

### MMVP 基准（图像差异识别）

| 模型 | 原始分数 | +Img-Diff | 提升 |
|------|---------|----------|------|
| LLaVA-1.5-7B | ~18 | ~32 | +14 |
| MGM-7B | ~27 | ~42 | +15 |
| InternVL2-8B | ~40 | ~45 | +5 |
| GPT-4V | ~30 | - | - |
| Gemini | ~30 | - | - |

### 8个MLLM通用基准（平均提升 Δ）

| 模型 | VQAv2 | GQA | POPE | MMBench | MM-Vet | SciQA | SEED | Δ平均 |
|------|-------|-----|------|---------|--------|-------|------|--------|
| LLaVA-1.5-7B | 78.5→79.3 | 62→62.8 | 85.9→86.4 | 64.3→66.1 | 30.5→33.2 | 66.8→68.2 | 58.6→61.7 | **+3.06%** |
| MGM-7B | 80.4→80.7 | 62.6→62.7 | 86→86.2 | 69.3→68.7 | 40.8→44.1 | 70.6→71.7 | 63.5→63.2 | **+1.28%** |
| InternVL2-8B | 81.8→81.8 | 62.6→62.6 | 87.7→88.0 | 82.5→82.7 | 49.2→52.6 | 96.5→96.6 | 69.5→69.9 | **+1.01%** |

### Spot-the-Diff 基准

| 模型 | BLEU | METEOR | CIDEr-D | ROUGE-L |
|------|------|--------|---------|---------|
| MGM-7B | 9.9 | 12.0 | 46.3 | 31.5 |
| MGM-7B + RP | **10.8** | **13.1** | **53.5** | **33.0** |
| VACC (之前SOTA) | 9.7 | 12.6 | 41.5 | 32.1 |

### 关键发现
- **Img-Diff 对弱模型提升更大**：LLaVA-1.5-7B 提升3.06%，InternVL2-8B（已有大量高质量数据）仅提升1.01%，说明数据的边际效益递减
- **质量评估结果**：人工标注1000个样本——79.6%的bounding box存在有效物体差异，80.1%的区域描述准确，70%+的差异描述完全正确
- **多样性可观**：数据集覆盖1,203个物体类别和3,680个独特的"物体替换对"
- **数据量vs质量权衡**：附录实验显示适当扩大数据量可进一步提升性能，但质量优先于数量
- **"物体移除"数据也有效**：除了"物体替换"，"物体移除"变体也能带来额外提升（附录Section 16）

## 亮点与洞察
- **数据合成的完整pipeline设计非常精巧**——从图像对生成到多级过滤再到区域级描述，每一步都有质量保证机制
- **区域级差异描述**（画红框+局部描述）比全图描述更精确——解决了"一段描述无法涵盖所有差异"的问题
- **作为通用微调数据集，Img-Diff 是"无害的"**——在提升差异识别能力的同时，不会损害（甚至轻微提升）模型在通用VQA上的表现
- **超越 GPT-4V 和 Gemini**这一结果非常有说服力——12个点的提升完全来自数据，不改模型架构

## 局限与展望
- 数据规模相对较小（12,688个样本），扩大规模+保持质量是挑战
- 依赖SDXL的生成质量——合成图像可能存在伪影，影响差异检测
- 差异描述依赖LLaVA-NEXT，该模型本身在细粒度描述上也有不足
- 仅关注"物体替换/移除"类差异，对属性变化（颜色、纹理、姿态）的覆盖不够系统
- 过滤流程较复杂，端到端效率有待提升

## 相关工作与启发
- **vs InstructPix2Pix**：同样用Prompt-to-Prompt生成图像对，但Img-Diff使用更好的SDXL、多级过滤、区域级描述，数据质量显著更高
- **vs Spot-the-Diff 数据集**：真实场景固定摄像头拍摄，差异是自然的时间变化。Img-Diff是合成的物体变化，覆盖范围更广但可能偏离真实分布
- **vs VIXEN**：首个用MLLM做图像差异描述的方法，侧重模型设计。Img-Diff侧重数据合成，二者互补

## 评分
- 新颖性: ⭐⭐⭐⭐ 用对比学习思想做数据合成的思路新颖，但基础组件（SDXL、P2P、FastSAM）均为现有工具
- 实验充分度: ⭐⭐⭐⭐⭐ 3个差异基准+8个通用基准+3个MLLM+数据质量/多样性评估+消融+多种变体，实验极为全面
- 写作质量: ⭐⭐⭐⭐ 流程图清晰，每个组件都有明确说明，但正文与附录的分割略重
- 价值: ⭐⭐⭐⭐ 提供了一种通用的数据增强策略，对提升MLLM细粒度视觉能力有直接价值，代码和数据开源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] On the Out-of-Distribution Generalization of Multimodal Large Language Models](on_the_out-of-distribution_generalization_of_large_multimodal_models.md)
- [\[CVPR 2025\] Active Data Curation Effectively Distills Large-Scale Multimodal Models](active_data_curation_effectively_distills_large-scale_multimodal_models.md)
- [\[CVPR 2025\] 4D LangSplat: 4D Language Gaussian Splatting via Multimodal Large Language Models](4d_langsplat_4d_language_gaussian_splatting_via_multimodal_large_language_models.md)
- [\[ACL 2025\] MegaPairs: Massive Data Synthesis For Universal Multimodal Retrieval](../../ACL2025/multimodal_vlm/megapairs_massive_data_synthesis_for_universal_multimodal_retrieval.md)
- [\[ICCV 2025\] Effective Training Data Synthesis for Improving MLLM Chart Understanding](../../ICCV2025/multimodal_vlm/effective_training_data_synthesis_for_improving_mllm_chart_understanding.md)

</div>

<!-- RELATED:END -->
