---
title: >-
  [论文解读] GiT: Towards Generalist Vision Transformer through Universal Language Interface
description: >-
  [ECCV 2024][语义分割][generalist vision model] 提出 GiT 框架，通过通用语言接口将图像描述、目标检测、实例分割、语义分割和视觉定位五大视觉任务统一为自回归序列生成，仅用纯 ViT（无任何任务特定模块）实现多任务联合训练，且任务间互相增强。 领域现状：LLM 已经证明多层 Transf…
tags:
  - "ECCV 2024"
  - "语义分割"
  - "generalist vision model"
  - "universal language interface"
  - "multi-task learning"
  - "ViT"
  - "auto-regressive"
---

# GiT: Towards Generalist Vision Transformer through Universal Language Interface

**会议**: ECCV 2024  
**arXiv**: [2403.09394](https://arxiv.org/abs/2403.09394)  
**代码**: [https://github.com/Haiyang-W/GiT](https://github.com/Haiyang-W/GiT)  
**领域**: 语义分割 / 多任务视觉建模  
**关键词**: generalist vision model, universal language interface, multi-task learning, ViT, auto-regressive

## 一句话总结
提出 GiT 框架，通过通用语言接口将图像描述、目标检测、实例分割、语义分割和视觉定位五大视觉任务统一为自回归序列生成，仅用纯 ViT（无任何任务特定模块）实现多任务联合训练，且任务间互相增强。

## 研究背景与动机
**领域现状**：LLM 已经证明多层 Transformer 架构（如 GPT）可以用简单的堆叠处理各种任务。视觉领域试图复制这一成功，但一直受限于任务特定模块（检测头、像素解码器等）。

**现有痛点**：LLaVA、Unified-IO、OFA 等统一模型仍保留了视觉编码器、RPN、感知头等特定组件；这些模块导致多阶段训练流程复杂、模型扩展困难；且它们主要聚焦图像级视觉-语言任务，忽略了检测和分割等经典感知。

**核心矛盾**：视觉任务输出格式差异巨大——检测输出可变数量的框、分割输出二值掩码、描述生成文本，难以用单一模型同时处理。密集预测的序列长度又使自回归解码计算量爆炸。

**本文目标**：设计一个架构极简的通用视觉模型，不用任何视觉专用模块（除 patch 投影），用纯多层 Transformer 处理从图像级到像素级的所有任务。

**切入角度**：所有视觉任务的目标都转化为语言 token 序列，利用标准词表而非额外 token；通过网格采样把密集预测拆解为 N 个并行子过程，解决像素级自回归的效率问题。

**核心 idea**：用通用语言接口 + 并行网格解码，让纯 ViT 完成从描述到分割的全任务统一建模。

## 方法详解

### 整体框架
GiT 的架构极度简洁：一个 window-based ViT（与 SAM 使用的相同结构）作为共享骨干，加上文本 embedding 和 out-of-vocabulary 压缩模块。没有 RPN、没有像素解码器、没有 FPN。共享参数占整体 >98%。输入由图像 patch、指令文本、N 个网格点的局部特征和任务标识符组成。图像部分用双向自注意力（类似编码器），局部预测部分用单向因果注意力（类似解码器），但共用同一组 Transformer 层。

### 关键设计
1. **统一输入输出表示 (Unified Representation)**

    - 功能：将所有模态（图像、文本、边界框、掩码）映射到统一 token 空间
    - 核心思路：文本用 WordPiece 分词（~30K 词表）；多片段概念（如 "traffic light"）通过单层注意力压缩成单个 out-of-vocabulary token：$\mathcal{F}_{\text{traffic light}} = \text{Attention}(\text{TE}(\mathcal{I}_0)+\text{PE}(0), \text{TE}(\mathcal{I}_1)+\text{PE}(1))$；稀疏目标（框、多边形）表示为 $(C, P=\{x_i,y_i\}_{i=1}^N)$；密集标签按栅格顺序展平为 1D 序列
    - 设计动机：避免扩展词表引入额外 token，简化后处理；统一表示使所有任务可共用同一个自回归损失

2. **多任务模板与并行解码 (Multi-Task Template with Parallel Decoding)**

    - 功能：将不同粒度的视觉任务嵌入统一的指令模板，并通过网格采样并行处理
    - 核心思路：模板定义为 $\langle\text{Image}\rangle\langle\text{Instruction}\rangle + N \times [\langle\text{LocalFeature}_i\rangle\langle\text{Task}_i\rangle:\langle\text{Response}_i\rangle]$。图像级任务 $N=1$；目标检测 $N=625$（25×25 网格）；语义分割 $N=1764$（42×42 网格）。每个网格点的局部特征通过双线性插值获得，各子过程独立并行解码
    - 设计动机：直接对全图做像素级自回归序列太长（672×672 有 45 万像素），拆成 1764 个子过程每个只需 16 步解码，大幅降低计算量。同时网格并行避免了非并行解码的效率瓶颈

3. **Out-of-Vocabulary 压缩模块**

    - 功能：将类别名（如 "traffic light"）、坐标值等多 token 概念压缩为单个 token
    - 核心思路：将多片段文本先用标准分词器生成子词索引，再用单层注意力融合为一个特征向量，取首个输出作为压缩表示
    - 设计动机：避免引入分隔符带来的序列膨胀，避免多 token 变长输出导致的复杂后处理

4. **注意力掩码策略**

    - 功能：在同一组 Transformer 层内实现编码器和解码器的双重功能
    - 核心思路：图像 patch 和 instruction 之间使用双向注意力（含 image-to-text 注意力）；局部特征和目标预测使用因果单向注意力。不同网格子过程的 patch token 在窗口注意力中只与同窗口内的网格点交互
    - 设计动机：保持架构纯粹性（不需要单独的编码器/解码器结构），同时让 image-to-text 注意力增强文本条件化能力

### 损失函数 / 训练策略
- **损失函数**：所有任务统一使用标准 CrossEntropy 损失做 next-token prediction，仅在词表上做动态控制——每个任务在训练和推理时使用对应的任务特定词表子集
- **多任务采样**：5 个任务均匀采样（各 1/5），与数据量大小无关，防止小数据集被淹没；任务内部按域（日常、室内、户外）平衡，域内按数据量比例采样
- **模型规模**：Base(131M) / Large(387M) / Huge(756M)，初始层继承 SAM 预训练参数
- **联合训练**：27 个公开数据集（17M 样本），不做任务特定微调

## 实验关键数据

### 主实验（Multi-task Generalist）

| 任务 | 指标 | GiT-Huge (multi) | 单任务训练 | 多任务提升 |
|------|------|-------------------|-----------|-----------|
| 目标检测 (COCO) | AP | 45.1 | 43.5 | +1.6 |
| 实例分割 (COCO) | AP₅₀ | 54.2 | 52.6 | +1.6 |
| 语义分割 (ADE20K) | mIoU | 47.7 | 47.6 | +0.1 |
| 图像描述 (COCO) | CIDEr | 133.0 | 128.3 | +4.7 |
| 视觉定位 (RefCOCO) | Acc@0.5 | 82.4 | 79.9 | +2.5 |

### 与其他通用模型对比

| 方法 | 特定模块数 | Det AP | InsSeg AP | SemSeg mIoU | Caption CIDEr | Grounding Acc |
|------|-----------|--------|-----------|-------------|---------------|---------------|
| Pix2Seq v2 | 2 | 46.5 | 38.2 | - | - | - |
| VisionLLM | 6 | 44.8 | - | - | - | - |
| Uni-Perceiver v2 | 5 | 42.0/52.0† | - | - | - | - |
| **GiT-Huge** | **0** | **45.1** | **42.6** | **47.7** | **133.0** | **82.4** |

### 消融实验

| 配置 | 检测 AP | 描述 CIDEr | 说明 |
|------|---------|-----------|------|
| 无 image-to-text 注意力 | 44.3 | 130.2 | 移除文本条件化 |
| 有 image-to-text 注意力 | **45.1** | **133.0** | 默认设置 |
| 单任务训练 | 43.5 | 128.3 | 无跨任务增强 |
| 多任务联合训练 | **45.1** | **133.0** | 任务互增强 |

### 关键发现
- 多任务联合训练带来的增益在所有任务上都为正值，证实了类似 LLM 的多任务互增强效应
- 不同类型的能力（图像理解、定位、分割、语言）可以在共享参数中协同学习
- 27 数据集联合训练使 GiT 在未见数据上展现出强 zero-shot 和 few-shot 迁移能力
- 架构简洁性（>98% 共享参数）使模型扩展极为直接

## 亮点与洞察
- **极简架构首次覆盖全视觉任务**：GiT 是第一个无任何视觉专用模块即可同时支持检测、分割（实例+语义）、描述、定位的通用模型，证明了纯多层 Transformer 在视觉的可行性。
- **并行网格解码是关键瓶颈突破**：把密集预测拆成上千个独立子过程并行解码，既保持了自回归语言接口的统一性，又解决了像素级序列过长的效率问题，这一设计颇具工程优雅。
- **多任务互增强**模式在视觉中首次系统验证：检测+描述共享定位能力后分别提升 1.6AP 和 4.7CIDEr，反映了视觉与语言能力的底层共性。

## 局限与展望
- 语义分割的 mIoU (47.7) 与专用模型 Mask2Former (47.2) 差距不大但也没有显著优势，密集预测任务的多任务增益相对有限 (+0.1)
- 目前仅处理 2D 视觉任务，3D 感知、视频理解等方向尚未覆盖
- 多边形表示的实例分割（polygon-based）与主流 mask-based 方法差异较大，限制了直接对比

## 相关工作与启发
- **vs VisionLLM**: VisionLLM 需要 6 个特定模块且分多阶段训练，GiT 用 0 个特定模块、一阶段联合训练
- **vs Pix2Seq v2**: Pix2Seq v2 不支持语义分割，且使用非并行解码效率低；GiT 覆盖更全且并行
- **vs LLaVA**: LLaVA 需要外部视觉编码器+LLM两阶段，且不支持检测分割；GiT 端到端更简洁

## 评分
- 新颖性: ⭐⭐⭐⭐ 纯ViT统一全视觉任务的简洁设计令人眼前一亮，并行网格解码很有创意
- 实验充分度: ⭐⭐⭐⭐ 5任务27数据集全面评估，消融充分，zero-shot验证完整
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，表格信息量大
- 价值: ⭐⭐⭐⭐ 缩小视觉与语言架构差距的重要探索，对通用视觉基础模型有启发意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Rotary Position Embedding for Vision Transformer](rotary_position_embedding_for_vision_transformer.md)
- [\[ECCV 2024\] SiLC: Improving Vision Language Pretraining with Self-Distillation](silc_improving_vision_language_pretraining_with_self-distillation.md)
- [\[ECCV 2024\] SCLIP: Rethinking Self-Attention for Dense Vision-Language Inference](sclip_rethinking_self-attention_for_dense_vision-language_inference.md)
- [\[ICLR 2026\] Locality-Attending Vision Transformer](../../ICLR2026/segmentation/locality-attending_vision_transformer.md)
- [\[CVPR 2025\] SketchFusion: Learning Universal Sketch Features through Fusing Foundation Models](../../CVPR2025/segmentation/sketchfusion_learning_universal_sketch_features_through_fusing_foundation_models.md)

</div>

<!-- RELATED:END -->
