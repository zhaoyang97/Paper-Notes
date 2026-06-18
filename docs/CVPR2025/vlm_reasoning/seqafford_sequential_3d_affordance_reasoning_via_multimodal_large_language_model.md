---
title: >-
  [论文解读] SeqAfford: Sequential 3D Affordance Reasoning via Multimodal Large Language Model
description: >-
  [CVPR 2025][多模态VLM][3D affordance] 提出 Sequential 3D Affordance Reasoning 任务，构建180K指令-点云对基准，通过在3D MLLM中引入 `<SEG>` token 和多粒度语言-点云融合模块，从复杂人类指令中推理并分割出序列化的affordance区域。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "3D affordance"
  - "序列推理"
  - "多模态大语言模型"
  - "点云分割"
  - "具身智能"
---

# SeqAfford: Sequential 3D Affordance Reasoning via Multimodal Large Language Model

**会议**: CVPR 2025  
**arXiv**: [2412.01550](https://arxiv.org/abs/2412.01550)  
**代码**: [https://github.com/seq-afford](https://github.com/seq-afford)  
**领域**: 多模态VLM  
**关键词**: 3D affordance、序列推理、多模态大语言模型、点云分割、具身智能

## 一句话总结

提出 Sequential 3D Affordance Reasoning 任务，构建180K指令-点云对基准，通过在3D MLLM中引入 `<SEG>` token 和多粒度语言-点云融合模块，从复杂人类指令中推理并分割出序列化的affordance区域。

## 研究背景与动机

现有3D affordance工作局限于"单物体-单affordance"范式：给定一条简单指令（如"打开门"），模型只需找到对应的单一affordance区域（如门把手）。但真实场景中，人类指令往往涉及多步骤、多物体的长horizon任务——例如"用微波炉加热碗里的食物"需要依次"抓碗→开微波炉门→放入碗"。现有方法无法主动推理这种隐含的序列affordance。

此外，之前的方法要么依赖独立的语言编码器（如BERT/RoBERTa），缺乏推理和世界知识；要么用纯LLM定位2D物体后再检索3D物体，缺乏视觉-语言联合对齐能力。因此需要一个统一的3D多模态大语言模型来弥合推理与细粒度分割之间的鸿沟。

## 方法详解

### 整体框架

SeqAfford由三大组件构成：(1) 基于大规模3D表征学习的3D视觉编码器（Uni3D），为密集预测提供基础特征；(2) 3D多模态大语言模型（ShapeLLM），利用世界知识进行affordance推理并输出含 `<SEG>` token的文本；(3) 多粒度语言-点云融合模块（MGLP），将LLM的推理结果注入点云密集特征以生成affordance mask。

### 关键设计

1. **Sequential Affordance Reasoning（序列affordance推理）**:
    - 功能：将复杂人类指令分解为多步affordance序列
    - 核心思路：在3D MLLM词汇表中添加特殊的 `<SEG>` token。模型接收点云 $\mathbf{X}_{\text{point}}$ 和指令 $\mathbf{X}_{\text{txt}}$ 后，输出含若干 `<SEG>` 的文本 $\tilde{\mathbf{y}}_{\text{txt}} = \mathcal{F}(\mathbf{X}_{\text{point}}, \mathbf{X}_{\text{txt}})$，每个 `<SEG>` 代表一个affordance分割结果。提取其last-layer embedding后通过MLP投影为分割向量 $\mathbf{H}_{\text{seg}}^{(i)} = \text{Proj}(\mathbf{h}_{\text{seg}}^{(i)})$
    - 设计动机：受LISA等2D分割MLLM启发，将分割能力嵌入LLM的生成过程，使推理与分割在统一框架中完成

2. **Multi-Granular Language-Point Integration（多粒度语言-点云融合）**:
    - 功能：将LLM推理出的抽象语义注入3D点云密集特征，实现affordance mask预测
    - 核心思路：分两阶段——(a) 多粒度特征传播：通过层级上采样和FPS将中间特征逐步传播为密集特征 $\mathbf{f}_{\text{dense}}$；(b) 点-语言融合：以 $\mathbf{H}_{\text{seg}}^{(i)}$ 为Query，$\mathbf{f}_{\text{dense}}$ 为Key/Value做cross-attention，再与稀疏特征 $\mathbf{f}_{\text{sparse}}$ 融合得到 $\mathbf{A}_f^{(i)} = \mathcal{G}(\mathbf{f}_{\text{dense}}, \mathbf{f}_{\text{sparse}}, \mathbf{H}_{\text{seg}}^{(i)})$，最终由解码器输出mask
    - 设计动机：仅用LLM的全局语义embedding无法直接做密集预测，需要多粒度的点云特征来补充空间细节，同时语义token提供"在哪里分割"的指导

3. **大规模Benchmark构建（180K指令-点云对）**:
    - 功能：提供单affordance和序列affordance两种设定的训练/测试数据
    - 核心思路：基于3D AffordanceNet的点云和PartNet的mesh渲染图，结合IAGNet的HOI图像，用4种方式prompt GPT-4o生成多样化指令（纯文本、mesh图、mesh+HOI图、mesh+场景描述）
    - 设计动机：解决现有数据集指令单一（同类物体共享同一文本）的问题，为每个点云实例生成个性化指令

### 损失函数 / 训练策略

总损失由三部分组成：$\mathcal{L} = \lambda_c \mathcal{L}_c + \lambda_b \mathcal{L}_b + \lambda_d \mathcal{L}_d$，其中 $\mathcal{L}_c$ 是自回归交叉熵损失（文本生成），$\mathcal{L}_b$ 是二元交叉熵损失（mask预测），$\mathcal{L}_d$ 是Dice损失（mask预测）。采用LoRA（rank=8）高效微调ShapeLLM-7B，冻结3D编码器，AdamW优化器，学习率2e-4，cosine scheduler，1×A100训练10 epochs。

## 实验关键数据

### 主实验

| 设定 | 方法 | mIoU↑ | AUC↑ | SIM↑ | MAE↓ |
|------|------|-------|------|------|------|
| Seen | PointRefer (SOTA) | 16.3 | 84.3 | 0.568 | 0.108 |
| Seen | **SeqAfford** | **19.5** | **86.9** | **0.594** | **0.098** |
| Unseen | PointRefer | 12.4 | 76.1 | 0.502 | 0.132 |
| Unseen | **SeqAfford** | **13.8** | **82.4** | **0.518** | **0.128** |
| Sequential | PointRefer* | 14.3 | 80.7 | 0.521 | 0.124 |
| Sequential | **SeqAfford** | **14.6** | **84.2** | **0.573** | **0.118** |

*注：\*表示baseline使用ground-truth序列顺序，因其本身不具备序列预测能力*

### 消融实验

| 配置 | mIoU↑ (Single) | AUC↑ (Single) | mIoU↑ (Seq) | AUC↑ (Seq) |
|------|----------------|---------------|-------------|------------|
| w/o MGLP | 12.1 | 83.4 | 11.7 | 80.3 |
| w/ MGLP (Ours) | **19.5** | **86.9** | **14.6** | **84.2** |

| 3D视觉编码器 | mIoU↑ | AUC↑ | SIM↑ | MAE↓ |
|-------------|-------|------|------|------|
| ULIP | 17.9 | 84.8 | 0.574 | 0.109 |
| OpenShape | 18.4 | 85.3 | 0.582 | 0.103 |
| Recon++ | 19.1 | 86.4 | 0.588 | 0.099 |
| **Uni3D** | **19.5** | **86.9** | **0.594** | **0.098** |

### 关键发现

- MGLP模块对mIoU提升巨大（Single: 12.1→19.5, +61%），说明多粒度融合是弥合推理与分割的关键
- 在Unseen设定下AUC从76.1提升到82.4（+8.3%），展示了3D MLLM利用世界知识进行开放世界泛化的能力
- 序列任务中，即使给baseline提供GT序列顺序，SeqAfford仍然在AUC和SIM上大幅领先

## 亮点与洞察

- **任务定义新颖**：首次将3D affordance从"单指令→单affordance"扩展到"复杂指令→序列affordance"，更贴近真实机器人操作需求
- **统一框架设计**：将推理（LLM世界知识）和分割（dense prediction）融合在同一模型中，避免了pipeline式方法的信息丢失
- **数据构建巧妙**：利用4种模态组合prompt GPT-4o生成多样化指令，比之前对每类物体共享同一文本的方式更真实

## 局限与展望

- 目前仅在物体级别做affordance分割，尚未扩展到场景级别的fine-grained reasoning
- 依赖ShapeLLM（7B）作为backbone，计算开销较大
- Sequential任务的mIoU（14.6）仍有较大提升空间，说明多步推理的准确性仍是挑战
- 数据集基于3D AffordanceNet的23个类别，覆盖范围有限

## 相关工作与启发

- LISA将 `<SEG>` token引入2D MLLM实现推理式分割，本文将其扩展到3D领域
- ShapeLLM/PointLLM等3D MLLM虽然理解3D物体，但缺乏dense prediction能力
- 对于具身AI，affordance推理是连接感知与操作的关键环节，序列affordance更是长horizon规划的基础

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次提出序列3D affordance推理任务，任务定义和benchmark构建都有新意
- 实验充分度: ⭐⭐⭐⭐ 多设定对比+消融实验充分，但缺少下游机器人操作的实际验证
- 写作质量: ⭐⭐⭐⭐ 结构清晰，任务动机阐述透彻
- 价值: ⭐⭐⭐⭐ 为具身AI的长horizon操作提供了affordance推理的新范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Generalized Few-Shot 3D Point Cloud Segmentation with Vision-Language Model](generalized_few-shot_3d_point_cloud_segmentation_with_vision-language_model.md)
- [\[CVPR 2025\] Period-LLM: Extending the Periodic Capability of Multimodal Large Language Model](period-llm_extending_the_periodic_capability_of_multimodal_large_language_model.md)
- [\[CVPR 2025\] SketchAgent: Language-Driven Sequential Sketch Generation](sketchagent_language-driven_sequential_sketch_generation.md)
- [\[CVPR 2025\] SldprtNet: A Large-Scale Multimodal Dataset for CAD Generation in Language-Driven 3D Design](sldprtnet_a_large-scale_multimodal_dataset_for_cad_generation_in_language-driven.md)
- [\[CVPR 2025\] Distraction is All You Need for Multimodal Large Language Model Jailbreaking](distraction_is_all_you_need_for_multimodal_large_language_model_jailbreaking.md)

</div>

<!-- RELATED:END -->
