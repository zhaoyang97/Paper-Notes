---
title: >-
  [论文解读] Advancing Visual Large Language Model for Multi-granular Versatile Perception
description: >-
  [ICCV 2025][语义分割][VLLM] 本文提出 MVP-LM，一个基于视觉大语言模型的多粒度通用感知框架，通过创新的多粒度解码器和 CoT 启发的数据统一策略，首次在单一模型中同时支持词级/句级指令下的框/掩膜预测四种感知组合，在全景分割、目标检测、视觉定位和指示表达分割等任务上取得有竞争力的性能。
tags:
  - "ICCV 2025"
  - "语义分割"
  - "VLLM"
  - "多粒度感知"
  - "统一框架"
  - "CoT数据策划"
  - "全景分割"
---

# Advancing Visual Large Language Model for Multi-granular Versatile Perception

**会议**: ICCV 2025  
**arXiv**: [2507.16213](https://arxiv.org/abs/2507.16213)  
**代码**: 无（论文提到 "The code will be available here" 但未给出链接）  
**领域**: 分割/目标检测  
**关键词**: VLLM, 多粒度感知, 统一框架, CoT数据策划, 全景分割

## 一句话总结
本文提出 MVP-LM，一个基于视觉大语言模型的多粒度通用感知框架，通过创新的多粒度解码器和 CoT 启发的数据统一策略，首次在单一模型中同时支持词级/句级指令下的框/掩膜预测四种感知组合，在全景分割、目标检测、视觉定位和指示表达分割等任务上取得有竞争力的性能。

## 研究背景与动机
视觉感知任务可按两个维度分类：预测类型（边界框 vs 分割掩膜）和指令类型（词级 vs 句级），共产生四种组合。然而，现有方法通常只覆盖其中部分组合，限制了模型的通用性。

具体来看：传统检测器（如 GLIP、Grounding DINO）擅长词级+框预测，但不做掩膜；分割方法（如 X-Decoder、OpenSeeD）处理词级+掩膜但不擅长句级理解；VLM（如 QwenVL、InternVL）能理解句级指令输出框坐标，但无法做像素级预测；近期如 LISA、PixelLM 能做句级+掩膜，但忽略了词级感知。

核心矛盾：联合训练某些组合已有研究，但所有四种组合的协同训练效果尚未被充分探索。特别是：（1）如何让 VLLM 同时输出框和掩膜？（2）如何统一词级和句级指令的处理？（3）如何利用 LLM 的解码和生成能力增强感知？

MVP-LM 的核心思路：利用 VLLM 的语言理解和生成能力，设计多粒度解码器实现框+掩膜双输出，通过 CoT 启发的数据策划将异构数据集统一为"先思考后感知"的格式。

## 方法详解

### 整体框架
MVP-LM 由四个核心组件组成：（1）Swin-B Transformer 作为图像编码器；（2）连接模块对齐视觉-文本特征；（3）Phi-1.5 作为语言模型；（4）基于 OpenSeeD 设计的多粒度解码器同时输出框和掩膜。VLLM 先生成图像描述caption，然后输出 summary token，其 hidden state 被投影为视觉查询，再送入解码器进行检测和分割。

### 关键设计

1. **动态查询生成**:

    - 功能：根据输入指令动态生成用于检测/分割的视觉查询，而非使用固定的可学习查询
    - 核心思路：每个查询由两部分组成——（a）**上下文感知基础查询**：VLLM 生成 summary token `<PER>`，其 hidden state 通过 MLP 投影为 $N$ 个基础查询向量；（b）**语言引导残差**：计算多尺度视觉特征与输入指令文本 embedding 的相似度，选取 Top-$N$ 最相似的视觉特征作为残差，与基础查询相加得到最终查询
    - 设计动机：动态查询使模型能自适应关注图像中与指令最相关的区域。LLM 生成的 summary token 编码了输入的全局上下文信息，而语言引导的视觉特征选择引入了空间先验

2. **多粒度解码器**:

    - 功能：基于 OpenSeeD 架构，同时处理框预测和掩膜预测
    - 核心思路：从查询选择机制生成内容查询和参考点，经过多层可变形注意力（deformable attention）与多尺度视觉特征交叉注意，每层输出通过三个共享头处理——跨模态相似度计算头、框回归头和掩膜预测头。对于词级感知，通过文本 embedding 与预测区域的相似度匹配类别；对于句级感知，使用 BCE 损失直接匹配目标
    - 设计动机：框和掩膜共享表征可以相互促进——掩膜标注可转化为框标注，联合训练能利用更丰富的数据源

3. **CoT 启发的数据统一策略**:

    - 功能：将来自不同任务（全景分割、检测、定位、指示分割）的异构数据集统一为单一 SFT 数据格式
    - 核心思路：每个训练样本由三部分组成——任务描述（如"请根据给定短语列表识别所有物体"或"请根据以下指令识别目标"）、指令（词列表或指示表达）、回答（"[图像描述]. 感知结果是 `<PER>`"）。训练时在回答中预置图像描述 caption，鼓励模型"先思考后感知"
    - 设计动机：使用多个开源 VLLM（VILA-3B/13B, InternVL2-8B/26B）自动生成多样化 caption，避免过拟合单一描述风格。词级任务中随机打乱类别顺序并加入负类别，防止学习虚假关联

### 损失函数 / 训练策略
综合损失函数：$\mathcal{L} = \mathcal{L}_{llm} + \lambda_{word}\mathcal{L}_{word} + \lambda_{sent}\mathcal{L}_{sent} + \mathcal{L}_{mask} + \mathcal{L}_{box}$，其中 $\mathcal{L}_{mask} = 5 \cdot L_{BCE} + 5 \cdot L_{DICE}$，$\mathcal{L}_{box} = 5 \cdot L_{L1} + 2 \cdot L_{GIoU}$。训练分两阶段：Stage 1 仅训练连接模块（CC3M 数据），Stage 2 对全模型（除视觉编码器）联合训练 80K 步。使用匈牙利匹配和去噪策略稳定优化。

## 实验关键数据

### 主实验：闭集全景分割与开集分割

| 方法 | 类型 | COCO PQ | COCO mIoU | ADE-OV PQ | ADE-OV mIoU | PC59 mIoU | PAS20 mIoU |
|------|------|---------|-----------|-----------|-------------|-----------|-----------|
| PSALM | VLLM | 55.9 | 66.6 | 13.7 | 18.2 | 48.5 | 81.3 |
| OMG-LLaVA | VLLM | 53.8 | - | - | - | - | - |
| **MVP-LM** | VLLM | **56.1** | **66.8** | **19.4** | **20.5** | 44.1 | **85.7** |
| OpenSeeD | 专家 | 59.5 | 68.6 | 19.7 | 23.4 | - | - |

### 消融实验

| 训练数据集配置 | RefCOCO val (cIoU) | COCO PQ | COCO mIoU | 说明 |
|--------------|-------------------|---------|-----------|------|
| C, R | 77.6 | 56.4 | 66.3 | 基础配置 |
| C, R, O | 81.8 | 55.8 | 65.9 | +O365，句级大幅提升 |
| C, R, O, G | 83.6 | 56.1 | 66.8 | +Grounding，全面最优 |

| 回答设置 | RefCOCO cIoU | COCO PQ | COCO mIoU |
|----------|-------------|---------|-----------|
| 无描述（仅 summary） | 75.6 | 55.3 | 65.7 |
| 生成已有物体名称 | 75.6 | 54.9 | 65.6 |
| 生成图像 caption | 75.7 | 55.6 | 66.2 |

### 关键发现
- MVP-LM 是首个在单一模型中覆盖所有四种感知组合的 VLLM 方法
- 仅 1.3B 参数的 MVP-LM 在 REC 任务上超越多数 7B/13B 模型（RefCOCO val 93.5）
- 多数据集联合训练将句级感知（RefCOCO）提升了 6.0 个点，同时保持词级感知性能稳定
- 开集分割（ADE-OV）上相比 PSALM 提升 5.7 PQ / 2.3 mIoU，体现了框架的泛化能力
- "先描述后感知"策略确实优于直接感知，验证了 CoT 范式对感知任务的有效性

## 亮点与洞察
- **四合一统一框架**：首次论证联合训练所有四种感知组合的可行性和互惠效果，特别是框标注数据对分割的增强作用
- **动态查询生成**：将 LLM 的生成能力与传统检测器的查询机制巧妙结合，base query 来自语言理解，residual 来自视觉-语言匹配
- **CoT 范式迁移到感知**：将推理领域的"先思考"范式应用于感知任务，是一个有启发性的设计

## 局限与展望
- 模型规模较小（Phi-1.5 + Swin-B），扩大规模可能带来进一步提升
- 在 PC59 等部分开集分割基准上弱于专家模型
- 尚未探索视频感知和 3D 感知的扩展
- 论文提到可探索 R1-like RL 训练用于感知任务，是值得关注的方向

## 补充实验：指示表达理解（REC）

| 方法 | 参数量 | RefCOCO val | RefCOCO testA | RefCOCO testB |
|------|--------|------------|--------------|---------------|
| Shikra | 13B | 87.8 | 91.1 | 81.8 |
| MiniGPTv2 | 7B | 88.7 | 91.7 | 85.3 |
| **MVP-LM** | **1.3B** | **93.5** | **94.5** | **91.6** |
| DeepSeek-VL2 | 200B+ | 95.1 | 96.7 | 95.1 |

仅 1.3B 参数的 MVP-LM 在 RefCOCO 上全面超越 7B/13B 模型，甚至接近 200B+ 的 DeepSeek-VL2，体现了统一框架设计的高效性。

## 相关工作与启发
- **vs PSALM**: PSALM 使用 VLLM 作为通用解码器实现词级框+掩膜预测，但忽略了句级框预测；MVP-LM 覆盖所有四种组合
- **vs LISA**: LISA 通过附加掩膜解码器实现句级分割，但不处理词级任务；MVP-LM 的多粒度解码器统一处理两种指令类型
- **vs OMG-LLaVA**: OMG-LLaVA 在全景分割上性能稍弱（53.8 PQ），且未报告开集分割和指示分割结果

## 评分
- 新颖性: ⭐⭐⭐⭐ 四种感知组合统一框架概念清晰，动态查询生成和 CoT 数据策划设计巧妙
- 实验充分度: ⭐⭐⭐⭐ 多基准评估 + 详尽消融，但部分开集基准性能不够突出
- 写作质量: ⭐⭐⭐⭐ 分类体系清晰，方法表述条理好，Tab.1 的能力对比表直观
- 价值: ⭐⭐⭐⭐ 为 VLLM 在统一感知方向提供了完整的基线和设计范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] HiMTok: Learning Hierarchical Mask Tokens for Image Segmentation with Large Multimodal Model](himtok_learning_hierarchical_mask_tokens_for_image_segmentation_with_large_multi.md)
- [\[CVPR 2026\] RealVLG-R1: A Large-Scale Real-World Visual-Language Grounding Benchmark for Robotic Perception and Manipulation](../../CVPR2026/segmentation/realvlg-r1_a_large-scale_real-world_visual-language_grounding_benchmark_for_robo.md)
- [\[CVPR 2025\] GLUS: Global-Local Reasoning Unified into A Single Large Language Model for Video Segmentation](../../CVPR2025/segmentation/glus_global-local_reasoning_unified_into_a_single_large_language_model_for_video.md)
- [\[CVPR 2025\] SAM2-LOVE: Segment Anything Model 2 in Language-Aided Audio-Visual Scenes](../../CVPR2025/segmentation/sam2-love_segment_anything_model_2_in_language-aided_audio-visual_scenes.md)
- [\[ICCV 2025\] Prompt Guidance and Human Proximal Perception for HOT Prediction with Regional Joint Loss](prompt_guidance_and_human_proximal_perception_for_hot_prediction_with_regional_j.md)

</div>

<!-- RELATED:END -->
