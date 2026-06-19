---
title: >-
  [论文解读] Implicit Concept Removal of Diffusion Models
description: >-
  [ECCV 2024][图像生成][概念消除] 提出 Geom-Erasing 方法，通过引入外部分类器/检测器提供隐式概念的存在性和几何位置信息，将其编码为文本条件中的位置 token 并作为负提示使用，有效消除扩散模型中水印、不安全内容等"隐式概念"的生成，在 I2P 和自建 ICD 基准上达到 SOTA。
tags:
  - "ECCV 2024"
  - "图像生成"
  - "概念消除"
  - "隐式概念"
  - "扩散模型"
  - "几何驱动控制"
  - "负提示"
---

# Implicit Concept Removal of Diffusion Models

**会议**: ECCV 2024  
**arXiv**: [2310.05873](https://arxiv.org/abs/2310.05873)  
**代码**: [https://kaichen1998.github.io/projects/geom-erasing/](https://kaichen1998.github.io/projects/geom-erasing/)  
**领域**: 目标检测 / 图像生成安全  
**关键词**: 概念消除, 隐式概念, 扩散模型, 几何驱动控制, 负提示

## 一句话总结

提出 Geom-Erasing 方法，通过引入外部分类器/检测器提供隐式概念的存在性和几何位置信息，将其编码为文本条件中的位置 token 并作为负提示使用，有效消除扩散模型中水印、不安全内容等"隐式概念"的生成，在 I2P 和自建 ICD 基准上达到 SOTA。

## 研究背景与动机

文本到图像（T2I）扩散模型（如 Stable Diffusion）在生成高质量图像的同时，经常不可控地生成水印、不安全内容等未在文本提示中指定的概念。本文首次将这类概念定义为**隐式概念（Implicit Concepts, IC）**——不在文本提示中显式出现但仍被模型生成的概念。评估显示，SD 生成的图片中约 11% 含有水印，39% 含有不安全内容。

现有概念消除方法的核心假设是：**需消除的概念可以被模型可控生成或识别**。但对隐式概念而言，这一假设不成立：

**隐式概念无法被可控生成**：在文本提示中加入"watermark"与生成图像中是否出现水印几乎无相关性（相关系数 $r=-0.08$, $p=0.21$），因此无法构造可靠的有/无概念配对图像用于微调。

**隐式概念无法被模型识别**：SD 的交叉注意力图无法定位到水印区域，即模型根本"看不见"自己生成的隐式概念。

**根本原因**：训练数据中包含含有隐式概念的图像，但对应的文本条件中没有描述这些概念，导致模型学会了生成它们但无法感知它们。

**核心 idea**：既然模型自己识别不了隐式概念，就借助外部分类器/检测器让模型"重新认知"这些概念，并利用几何位置信息精确定位和消除。

## 方法详解

### 整体框架

Geom-Erasing 的流程为：(1) 使用外部分类器/检测器识别图像中隐式概念的存在性和位置；(2) 将概念名称和位置信息编码为特殊 token 追加到文本条件中；(3) 通过区域损失重加权微调模型；(4) 推理时将学到的概念+位置 token 作为负提示，引导生成远离隐式概念。

### 关键设计

1. **隐式概念识别**：利用现成的分类器或检测器（如 LAION 水印检测器、NSFW 分类器、OCR 文本检测器）获取隐式概念的检测结果 $L = [p_i, (o_i)]_{i=1}^N$，其中 $p_i$ 为置信度，$o_i = [a_i^1, b_i^1, a_i^2, b_i^2]$ 为边界框坐标。核心优势在于只需检测器的输出结果，无需访问其参数。

2. **几何驱动消除（Geometry-driven Removal）**：这是本文的核心贡献。将连续坐标离散化为 bin，每个 bin 对应一个特殊位置 token $\langle l\{m,n\}\rangle$ 加入文本词表。被隐式概念覆盖的 bin 对应的位置 token 被追加到文本条件中：
    $y' = y \oplus y_{\text{im}} \oplus \langle l\{m,n\}\rangle_{m=A_{\text{bin}}^1, n=B_{\text{bin}}^1}^{m=A_{\text{bin}}^2, n=B_{\text{bin}}^2}$
   其中 $A_{\text{bin}}^1 = \lfloor a_i^1/W_{\text{bin}}\rfloor$ 等为离散化后的 bin 索引。这种设计让模型学会将概念名称与其空间位置关联，从而在推理时能够精确"抹除"。
    - 设计动机：仅添加概念名称（如"watermark"）远不足以消除隐式概念（消融实验验证），几何位置信息是成功消除的关键。

3. **损失重加权策略**：对隐式概念区域降低损失权重，让模型更关注非概念区域的生成质量：
    $\mathcal{L}_{\text{Geom-Erasing}} = \mathbb{E}_{z,y,\epsilon,t}\left[w \odot \|\epsilon - \epsilon_\theta(z_t, t, c_\theta(y'))\|_2^2\right]$
   其中权重 $w_{m,n}$ 在隐式概念区域内取 $\frac{T}{K+\alpha(T-K)}$，区域外取 $\frac{\alpha T}{K+\alpha(T-K)}$，保证权重总和不变（$\sum w_{m,n}=T$），$\alpha$ 为超参数。
    - 设计动机：隐式概念区域的像素本身是"脏数据"，降低其权重可避免模型学习生成这些内容，同时不影响整体损失尺度。

### 损失函数 / 训练策略

- **Model Removal 设置**：仅优化新增的位置 token 的嵌入向量，不修改扩散模型参数
- **Data Removal 设置**：同时微调扩散模型参数和位置 token
- 推理时，将概念名称+位置 token 作为负提示（Negative Prompt）使用

## 实验关键数据

### 主实验

**Model Removal 设置**（消除预训练 SD 中的水印和不安全内容）：

| 方法 | 水印 FID↓ | 水印 ICR(%)↓ | I2P Overall↓ | I2P Sexual↓ | I2P Inappro.↓ |
|------|-----------|-------------|--------------|-------------|---------------|
| SD | 9.05 | 11.13 | 0.39 | 0.30 | 0.97 |
| ESD | 9.49 | 11.28 | 0.19 | 0.17 | - |
| NP | 9.12 | 11.13 | 0.16 | 0.08 | 0.80 |
| SLD-Strong | 9.87 | 9.92 | 0.13 | 0.09 | 0.72 |
| **Geom-Erasing** | **8.34** | **7.31** | **0.09** | **0.05** | **0.63** |

**Data Removal 设置**（消除微调数据中注入的隐式概念）：

| 数据集 | 指标 | SD | ESD | FMN | NP | **Geom-Erasing** |
|--------|------|-----|-----|-----|-----|------------------|
| ICD-QR | ICR(%)↓ | 74.59 | 17.64 | 80.42 | 59.64 | **5.38** |
| ICD-QR | FID↓ | 65.82 | 90.97 | 71.76 | 69.31 | **41.41** |
| ICD-Watermark | ICR(%)↓ | 30.40 | 28.98 | 30.76 | 27.71 | **5.02** |
| ICD-Text | ICR(%)↓ | 71.84 | 38.08 | 74.75 | 65.63 | **13.48** |

### 消融实验

**不同组件的消融**（ICD-Watermark, Data Removal）：

| 概念名称 | 几何信息 | 损失重加权 | FID↓ | ICR(%)↓ | F*R↓ | 说明 |
|---------|---------|-----------|------|---------|------|------|
| ✗ | ✗ | ✗ | 7.59 | 30.40 | 230.74 | 基线（原始SD微调） |
| ✓ | ✗ | ✗ | 7.06 | 17.04 | 120.30 | 仅概念名不够 |
| ✓ | ✓ | ✗ | 6.81 | 7.36 | 50.12 | 几何信息是关键 |
| ✓ | ✓ | ✓ | **6.42** | **7.23** | **46.42** | 完整方法 |
| 0%水印训练(oracle) | - | - | 6.93 | 7.13 | 49.41 | 理论最优 |

### 关键发现

- **几何信息是隐式概念消除的关键**：仅添加概念名称将 ICR 从 30.40% 降至 17.04%，加入几何信息后进一步降至 7.36%
- **消除隐式概念同时提升生成质量**：FID 从 7.59 降至 6.42，甚至优于"0%水印训练"的理想情况（6.93），说明几何信息帮助了更好的概念学习
- **对检测器精度不敏感**：IoU 容忍度约 0.4，即粗略的位置信息即可有效消除
- 现有方法（FMN、NP、SLD）依赖模型自身的概念识别能力，对隐式概念效果极差

## 亮点与洞察

- **问题定义新颖**：首次系统定义了"隐式概念"问题，并通过实验证明现有方法失败的本质原因
- **方法设计巧妙**：借助外部检测器弥补模型自身的盲区，将空间位置信息编码为文本 token 是一种优雅的跨模态信息注入方式
- **构建了 ICD 数据集**：包含 QR 码、水印、文本三种隐式概念的标准基准，填补了评估空白
- **实际价值大**：水印和不安全内容是扩散模型部署的核心合规问题，Geom-Erasing 提供了有效的后处理方案

## 局限与展望

- 依赖外部检测器的可用性，对于全新的隐式概念类型需要额外训练检测器
- 位置 token 作为负提示加入几何信息虽有效但也略微增加 FID（见 Table 7），原因待进一步分析
- 仅在 Stable Diffusion v1.5 上验证，对更大模型（如 SDXL、DALL-E 3）的泛化性未知
- bin 大小和数量的选择需要调参，虽然消融实验表明方法对此不太敏感
- 损失重加权中隐式概念区域的固定低权重可能导致该区域的生成质量下降

## 相关工作与启发

- **ESD (CVPR 2023)**：通过让模型远离自身生成的图像来消除概念，但会误伤正常内容
- **Negative Prompt / SLD**：利用增强的无分类器引导来规避特定概念，但依赖模型自身的概念理解
- **FMN**：使用文本反演增强模型对概念的识别，再调整交叉注意力分数，但对隐式概念无效
- **启发**：几何信息编码为离散 token 的思路可推广到其他需要空间控制的生成任务

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次定义隐式概念问题，揭示现有方法的本质缺陷，提出的几何驱动方案独到
- 实验充分度: ⭐⭐⭐⭐ Model/Data Removal 双设置验证，自建三个ICD数据集，消融全面，但缺少更多模型的泛化验证
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，预实验有说服力，方法阐述系统
- 价值: ⭐⭐⭐⭐ 解决了扩散模型部署中的重要合规问题，ICD 数据集对社区有长期价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] OMG: Occlusion-friendly Personalized Multi-concept Generation in Diffusion Models](omg_occlusion-friendly_personalized_multi-concept_generation_in_diffusion_models.md)
- [\[ECCV 2024\] L-DiffER: Single Image Reflection Removal with Language-Based Diffusion Model](l-differ_single_image_reflection_removal_with_language-based_diffusion_model.md)
- [\[ECCV 2024\] Implicit Style-Content Separation using B-LoRA](implicit_style-content_separation_using_b-lora.md)
- [\[AAAI 2026\] Mass Concept Erasure in Diffusion Models with Concept Hierarchy](../../AAAI2026/image_generation/mass_concept_erasure_in_diffusion_models_with_concept_hierarchy.md)
- [\[ECCV 2024\] SAIR: Learning Semantic-aware Implicit Representation](sair_learning_semantic-aware_implicit_representation.md)

</div>

<!-- RELATED:END -->
