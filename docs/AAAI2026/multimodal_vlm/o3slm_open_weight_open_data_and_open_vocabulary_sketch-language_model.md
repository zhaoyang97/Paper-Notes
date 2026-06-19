---
title: >-
  [论文解读] O3SLM: Open Weight, Open Data, and Open Vocabulary Sketch-Language Model
description: >-
  [AAAI 2026][多模态VLM][草图理解] 本文构建了大规模草图-图像-指令三元组数据集SketchVCL（包含600K预训练 + 215K微调数据），并训练了O3SLM——首个能够流畅理解手绘草图并完成检测、计数、检索和VQA四大任务的开源大视觉语言模型，在所有任务上大幅超越现有LVLM。
tags:
  - "AAAI 2026"
  - "多模态VLM"
  - "草图理解"
  - "大视觉语言模型"
  - "草图-图像-文本对齐"
  - "开放词汇"
  - "指令微调"
---

# O3SLM: Open Weight, Open Data, and Open Vocabulary Sketch-Language Model

**会议**: AAAI 2026  
**arXiv**: [2511.14368](https://arxiv.org/abs/2511.14368)  
**代码**: [项目主页](https://vcl-iisc.github.io/O3SLM/)  
**领域**: 多模态VLM  
**关键词**: 草图理解, 大视觉语言模型, 草图-图像-文本对齐, 开放词汇, 指令微调

## 一句话总结
本文构建了大规模草图-图像-指令三元组数据集SketchVCL（包含600K预训练 + 215K微调数据），并训练了O3SLM——首个能够流畅理解手绘草图并完成检测、计数、检索和VQA四大任务的开源大视觉语言模型，在所有任务上大幅超越现有LVLM。

## 研究背景与动机

**领域现状**：大视觉语言模型（LVLM）在VQA、文档理解等任务上取得了巨大成功，但它们几乎完全依赖自然图像和文本输入。手绘草图作为一种直观的视觉表达方式，可以轻松传达文字难以描述的空间布局和形状信息，且跨越语言障碍，是更通用的沟通工具。

**核心痛点**：现有开源LVLM（LLaVA、Qwen-VL、DeepSeek-VL2等）在理解粗糙手绘草图时几乎完全失败。如图1所示，即使模型能勉强识别某些视觉线索，也无法利用这些信息完成检测、推理等下游任务。闭源模型（GPT-4o、Gemini）虽有初步的草图理解能力，但多模态定位能力弱，且不可访问和解释。

**根本原因**：缺乏大规模、开源的草图-图像-文本联合训练数据集。现有草图数据集（QuickDraw、Sketchy、TU-Berlin等）要么只有类级草图没有配对图像，要么只针对单一任务（SBIR），且都缺少文本描述和问答对——而这些对训练LVLM至关重要。

**核心矛盾**：草图具有高度抽象性和变异性（风格、文化、绘画技能差异大），这使其与自然图像之间存在巨大的域差距。要让LVLM理解草图，需要同时解决数据缺乏和模态对齐两大挑战。

**本文切入角度**：
1. 构建自动化草图生成流水线，从大规模图像数据集批量生成实例级草图
2. 设计两阶段训练策略：先做大规模草图-图像-文本对齐预训练，再做任务特定指令微调
3. "三个Open"原则：Open Weight（开源权重）、Open Data（开源数据）、Open Vocabulary（开放词汇）

## 方法详解

### 整体框架
O3SLM的架构简洁：CLIP ViT-L/336作为视觉主干（编码草图和自然图像）→ 两层MLP多模态连接器 → Vicuna v1.5 LLM。草图、图像和文本token拼接后送入LLM，利用自注意力隐式学习跨模态对齐。模型权重从LLaVA-1.5初始化以继承其文本-图像对齐能力。

### 关键设计

#### 1. **SketchVCL数据集与自动化草图生成流水线**

**草图生成流水线（图3）**：
- 对每个目标对象实例，使用SAM2生成分割掩码
- 掩去背景，将前景通过Photo2Sketch（Pix2Pix方法）生成草图
- 使用形态学梯度边缘检测增强草图
- 最终草图 = Pix2Pix草图 + 边缘检测结果的聚合

在Object365和OpenImages上分别生成了19M和14M个实例级草图。

**数据集组成**：

| 阶段 | 任务 | 图像数据集 | 草图来源 | 数量 |
|------|------|-----------|---------|------|
| 预训练 | 详细描述+边界框 | Objects365 | SketchVCL-O365 | 300K |
| 预训练 | 详细描述+边界框 | OpenImages | SketchVCL-OI | 300K |
| 微调 | 目标检测 | COCO | SketchMIX | 110K |
| 微调 | VQA | COCO | SketchMIX | 50K |
| 微调 | 计数 | PixMo Count | SketchMIX | 30K |
| 微调 | SBIR | Sketchy | SketchMIX | 25K |

设计动机：Photo2Sketch方法比CLIP-based方法质量更好，比扩散模型快得多，适合大规模数据生成。SketchMIX混合多个草图源（Sketchy + QuickDraw + 生成的草图）以增加多样性，故意排除TU-Berlin用于测试泛化能力。

#### 2. **两阶段训练策略**

**Stage I：草图对齐预训练**（600K）
目标是让模型学会三模态对应关系——草图↔图像↔文本：
- 识别草图中的对象
- 将草图与对应图像中的对象关联
- 发展精细空间理解能力（检测所需）
- 保持自然语言描述能力

每张图像配一个目标类别，DeepSeek-VL2生成描述性标题，LLaMA-3-8B Instruct进一步精炼为包含草图识别、物体描述、空间关系和边界框坐标的结构化回答。

**Stage II：指令微调**（215K）
为四个任务设计了任务特定前缀描述符（类似Molmo的做法）：
- **COUNT**：草图引导的对象计数，输出整数
- **BBOX**：草图引导的目标检测，输出 $[x_1, y_1, x_2, y_2]$
- **VQA**：草图辅助的视觉问答（25K草图QA + 25K普通QA平衡）
- **SBIR**：草图-图像检索，训练目标类似二元交叉熵

设计动机：两阶段分离使模型先建立通用草图理解，再适配特定任务。任务前缀避免了混淆，随机选择提示模板防止提示过拟合。

#### 3. **SBIR的LVLM适配方案**

创新性地将SBIR转化为LLM可直接训练的二分类任务：
$$\arg\min_\theta -\sum_{i=1}^N [y_i \log(p_\theta(\texttt{<yes>}|X_i)) + (1-y_i)\log(p_\theta(\texttt{<no>}|X_i))]$$

推理时按 $p_\theta(\texttt{<yes>}|X_i)$ 的概率降序排列，取Top-K。

设计动机：传统SBIR需要特殊的度量学习架构。将其转化为<yes>/<no>二分类完美融入LLM训练框架，无需修改模型结构。

### 损失函数 / 训练策略
- 使用LoRA（rank=64）进行参数高效训练
- 2张NVIDIA H100 GPU
- 学习率 $2 \times 10^{-5}$，余弦衰减，3%预热
- 训练1个epoch，batch size 24
- 模型规模：7B和13B两个版本

## 实验关键数据

### 主实验——草图引导计数（Accuracy）

| 模型 | PixMo-Count Avg | COCO Avg |
|------|----------------|----------|
| GPT-4o | 33.6 | 16.4 |
| Gemini 1.5 Pro | 32.5 | 17.0 |
| LLaVA-1.5-7B | 16.0 | 12.1 |
| Qwen2.5-VL-7B | 17.7 | 24.6 |
| Molmo-7B-D | 30.3 | 12.0 |
| **O3SLM-7B** | **43.5** | **31.3** |
| **O3SLM-13B** | **44.0** | **31.7** |

### 草图引导目标检测（Acc@0.5，COCO val2017）

| 模型 | Sketchy | QuickDraw | TU-Berlin† | SketchVCL-C |
|------|---------|-----------|------------|-------------|
| LLaVA-1.5-7B | 29.1 | 26.9 | 29.7 | 27.4 |
| Molmo-7B-D | 25.3 | 27.9 | 27.5 | 25.3 |
| **O3SLM-7B** | **33.9** | **23.8** | **29.4** | **21.5** |
| **O3SLM-13B** | **35.6** | **28.1** | **31.5** | **24.8** |

（注：†TU-Berlin为训练未见过的数据集，测试泛化能力）

### SBIR检索（Sketchy数据集）

| 模型 | Acc@1 | Acc@5 | Acc@10 |
|------|-------|-------|--------|
| LLaVA-1.5-7B | 11.0 | 14.4 | 13.0 |
| **O3SLM-7B** | **65.0** | **59.2** | **39.4** |
| LLaVA-1.5-13B | 10.0 | 29.2 | 28.3 |
| **O3SLM-13B** | **55.0** | **46.4** | **32.9** |

### 消融实验

| 配置 | 关键发现 | 说明 |
|------|---------|------|
| 无预训练 | SBIR大幅下降，计数影响小 | 预训练对高度依赖草图的任务（检索）至关重要 |
| 冻结多模态连接器 | 性能显著下降 | 7B调连接器 > 13B冻结连接器 |
| 纯图像任务 | VQAv2: 76.6 vs 80.0 (LLaVA) | 草图训练仅损失<5%图像理解能力 |
| 文本检测 | 21.0 vs 13.4 (LLaVA) | 草图训练反而提升了文本引导的检测 |

### 关键发现
1. O3SLM在所有草图任务上大幅超越现有开源LVLM，甚至在多项任务上超越GPT-4o和Gemini 1.5 Pro
2. 在训练未见过的TU-Berlin草图上表现出强泛化能力，证明模型学到了通用草图理解而非过拟合特定风格
3. SBIR Acc@1从11.0%提升到65.0%，5.9倍提升，说明原始LLaVA几乎无法理解草图
4. 调整多模态连接器是关键——7B模型调连接器优于13B冻结连接器，说明草图-图像对齐需要在投影层完成
5. 模型展现出涌现能力：虽然只用单独草图训练，但能处理草图+文本的细粒度联合查询

## 亮点与洞察
1. **填补了LVLM在草图理解上的空白**：首个专门为草图设计的开源LVLM，开放了权重、数据和模型
2. **大规模自动化草图生成流水线**：在Object365和OpenImages上生成了33M+实例级草图，解决了数据瓶颈
3. **SBIR的LLM适配方案优雅简洁**：将检索问题转化为二分类完美融入LLM框架
4. **涌现的细粒度理解**：通过VQA辅助监督，模型自动学会了利用文本描述补充草图难以表达的属性（颜色、纹理等）
5. **几乎不损失原有能力**：图像任务性能仅下降<5%，说明草图训练与图像能力互补

## 局限与展望
1. Photo2Sketch生成的草图可能不完全模拟真实手绘草图的多样性和噪声
2. 基于LLaVA-1.5架构，CLIP ViT-L/336分辨率对细粒度草图可能不够
3. 训练仅1个epoch，更多epoch是否能进一步提升？
4. SBIR推理需要对gallery中每张图做前向传播（10K次），效率较低
5. 未探索草图生成能力（image→sketch），仅聚焦于草图理解

## 相关工作与启发
- 自动化数据生成流水线（SAM2+Pix2Pix+边缘检测）可以推广到其他抽象视觉模态
- 两阶段训练策略（先对齐再微调）是新模态接入LVLM的通用范式
- 将检索任务转化为二分类的思路可以推广到其他"匹配"类任务
- 验证了在投影层进行模态对齐的重要性（vs 仅调LLM）

## 评分
- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Vocabulary Scaling Law: Tuning Open-vocabulary Predictors for Their Openness](../../CVPR2026/multimodal_vlm/vocabulary_scaling_law_tuning_open-vocabulary_predictors_for_their_openness.md)
- [\[CVPR 2025\] Compositional Caching for Training-free Open-vocabulary Attribute Detection](../../CVPR2025/multimodal_vlm/compositional_caching_for_training-free_open-vocabulary_attribute_detection.md)
- [\[CVPR 2026\] Towards Open-Vocabulary Industrial Defect Understanding with a Large-Scale Multimodal Dataset](../../CVPR2026/multimodal_vlm/towards_open-vocabulary_industrial_defect_understanding_with_a_large-scale_multi.md)
- [\[CVPR 2025\] Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models](../../CVPR2025/multimodal_vlm/molmo_and_pixmo_open_weights_and_open_data_for_state-of-the-art_vision-language_.md)
- [\[CVPR 2026\] SynCLIP: Synonym-Coherent Language-Image Pretraining for Robust Open-Vocabulary Dense Perception](../../CVPR2026/multimodal_vlm/synclip_synonym-coherent_language-image_pretraining_for_robust_open-vocabulary_d.md)

</div>

<!-- RELATED:END -->
