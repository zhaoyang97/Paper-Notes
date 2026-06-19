---
title: >-
  [论文解读] Paint by Inpaint: Learning to Add Image Objects by Removing Them First
description: >-
  [CVPR 2025][语义分割][图像编辑] 提出"Paint by Inpaint"框架，利用"添加对象是移除对象的逆过程"这一洞察，通过自动化 inpainting 管线构建包含约 100 万高质量图像对的 PIPE 数据集，训练的扩散模型在对象添加和通用编辑任务上达到 SOTA。 基于文本指令的无 mask 图像对象…
tags:
  - "CVPR 2025"
  - "语义分割"
  - "图像编辑"
  - "对象添加"
  - "扩散模型"
  - "图像修复"
  - "数据集构建"
---

# Paint by Inpaint: Learning to Add Image Objects by Removing Them First

**会议**: CVPR 2025  
**arXiv**: [2404.18212](https://arxiv.org/abs/2404.18212)  
**代码**: [项目页面](https://rotsteinnoam.github.io/Paint-by-Inpaint/)  
**领域**: 图像分割  
**关键词**: 图像编辑, 对象添加, 扩散模型, 图像修复, 数据集构建

## 一句话总结

提出"Paint by Inpaint"框架，利用"添加对象是移除对象的逆过程"这一洞察，通过自动化 inpainting 管线构建包含约 100 万高质量图像对的 PIPE 数据集，训练的扩散模型在对象添加和通用编辑任务上达到 SOTA。

## 研究背景与动机

基于文本指令的无 mask 图像对象添加是一个极具挑战的任务，需要理解全局上下文（位置、尺度、风格）。现有方法的主要瓶颈在于**训练数据的质量**：

1. **InstructPix2Pix (IP2P)** 的数据问题：使用 GPT-3 + Prompt-to-Prompt 合成数据集，但源图和目标图都是合成的，且常存在不一致性。虽然使用 Directional CLIP 过滤但效果有限
2. **MagicBrush**：人工使用 DALL-E2 创建的半合成数据集，质量较好但规模受限于人工标注成本
3. **根本矛盾**：在无 mask 设置下，几乎不可能获得一对仅在编辑区域不同的自然图像

本文的核心 insight 非常优雅：**添加对象 (Paint) 本质上是移除对象 (Inpaint) 的逆过程**。利用大量分割数据集中已有的图像和 mask，可以通过 inpainting 模型移除对象生成"源图"，将原始图像作为"目标图"，从而构建大规模、高质量、mask-free 的对象添加数据集。

关键优势在于：(i) 目标图是**真实自然图像**而非合成图，(ii) 源-目标一致性**天然保证**——改动仅限于被移除对象的区域。

## 方法详解

### 整体框架

分为两阶段框架：
1. **数据集构建阶段 (PIPE)**：移除对象 → 过滤 → 生成指令 → 形成训练三元组
2. **模型训练阶段**：基于 SD 1.5 架构训练去噪扩散编辑模型

### 关键设计

1. **图像对生成管线 (Source-Target Pairs)**:
    - 功能：从分割数据集自动构建高质量的对象添加图像对
    - 核心思路：
        - **数据源**：统一 COCO + Open Images + LVIS = 889,230 张图像，1400+ 类别
        - **预移除过滤**：排除太大/太小/边缘处 mask，用 CLIP 语义相似度过滤异常对象视图（模糊、遮挡），对 mask 做形态学膨胀确保完全覆盖
        - **对象移除**：使用 SD inpainting 模型，正面提示"a photo of a background"，负面提示"an object, a <class>"，10步去噪生成 3 个候选
        - **后移除验证**：CLIP 共识检测（3 个候选嵌入标准差低=一致移除）+ 多模态 CLIP 过滤（区域与原类别相似度低=确实移除）+ α-blending 一致性增强 + 重要性过滤
    - 设计动机：inpainting 模型不是专为移除训练的，可能留下残留或生成新对象，因此需要多层过滤确保质量

2. **对象添加指令生成**:
    - 功能：为每对图像生成自然语言编辑指令
    - 核心思路：三种策略——(i) 类名指令"add a <class>"；(ii) VLM-LLM 指令：CogVLM 描述对象 → Mistral-7B 通过 5-shot ICL 转为指令；(iii) 参考指令：利用 RefCOCO 系列的人工标注描述
    - 设计动机：多种策略结合产生 1,879,919 条指令，涵盖简洁和详细两种编辑场景

3. **扩散编辑模型训练**:
    - 功能：学习根据指令向图像添加对象
    - 核心思路：基于 SD 1.5 架构，同时以文本指令 c_T 和源图像 c_I 为条件。训练时以 5% 概率分别丢弃 c_T、c_I 或两者，支持推理时 classifier-free guidance
    - 设计动机：双条件 CFG 允许在推理时平衡编辑忠实度和原图一致性

### 损失函数 / 训练策略

- 标准扩散去噪损失
- Classifier-free guidance 中分别对文本和图像条件进行概率 dropout（各 5%）
- 可通过调整图像/文本 guidance scale 控制一致性-编辑权衡

## 实验关键数据

### 主实验

MagicBrush 对象添加子集（144 edits）：

| 方法 | L1↓ | CLIP-I↑ | DINO↑ | CMMD↓ |
|------|-----|---------|-------|-------|
| VQGAN-CLIP | .211 | .670 | .507 | .862 |
| SDEdit | .168 | .765 | .572 | .539 |
| IP2P | .100 | .860 | .766 | .363 |
| Hive | .095 | .846 | .782 | .353 |
| **Ours** | **.072** | **.900** | **.852** | **.301** |

PIPE 测试集（750 images）：

| 方法 | L1↓ | CLIP-I↑ | DINO↑ | CMMD↓ |
|------|-----|---------|-------|-------|
| IP2P | .098 | .861 | .753 | .142 |
| Hive | .088 | .849 | .754 | .232 |
| **Ours** | **.057** | **.945** | **.903** | **.060** |

### 消融实验

人类评估（100 张图，57 名评估者，1833 条评价）：

| 指标 | IP2P | Ours |
|------|------|------|
| 编辑忠实度偏好 (Overall%) | 26.4% | **73.6%** |
| 输出质量偏好 (Overall%) | 28.5% | **71.5%** |
| 编辑忠实度 (Per-image wins) | 28 | **72** |
| 输出质量 (Per-image wins) | 31 | **69** |

### 关键发现

1. **数据集规模和质量是决定性因素**：PIPE 的约 100 万真实目标图像全面超越 IP2P 的 31 万合成对
2. **一致性优势显著**：L1 指标大幅领先，说明编辑区域外的一致性天然保证
3. **泛化到通用编辑**：将 PIPE 与 IP2P 数据集合并训练的模型在通用编辑任务上也超越仅用 IP2P 训练的模型
4. **人类评估压倒性优势**：73.6% 的全局偏好率确认了生成质量的显著提升
5. **Fine-tuning 进一步提升**：在 MagicBrush 上 fine-tune 后 L1 从 .072 降至 .067

## 亮点与洞察

1. **"逆向思维"的核心洞察**：识别到"添加 = 移除的逆"这一对称性是全文最精彩的部分，将困难问题转化为简单问题
2. **工业级数据管线**：多层过滤机制（CLIP 共识、多模态 CLIP、一致性增强、重要性过滤）确保了大规模自动化流程中的数据质量
3. **VLM-LLM 两阶段指令生成**：将对象描述和指令生成分开，避免 VLM 幻觉
4. **一致性的天然保证**：通过 α-blending 和 mask 约束，源-目标一致性是构造式保证的

## 局限与展望

1. **受限于 inpainting 模型质量**：如果 inpainting 没有完全移除对象或产生异常，即使有过滤也可能有漏网之鱼
2. **对象添加位置不可控**：模型从数据中隐式学习位置，用户无法精确指定添加位置
3. **类别覆盖受限于分割数据集**：虽然有 1400+ 类别，但长尾类别覆盖仍不足
4. **基于 SD 1.5 架构**：在更先进的基础模型上可能获得更好效果
5. 未来可探索多对象添加、精确位置控制、与更大 VLM 结合等

## 相关工作与启发

- **vs InstructPix2Pix**: IP2P 的数据全是合成的且有一致性问题；PIPE 的目标图是真实图像，一致性天然保证
- **vs MagicBrush**: MagicBrush 是人工标注的高质量数据但规模受限（约 10K）；PIPE 自动化产生约 100 万对
- **vs Inst-Inpaint**: Inst-Inpaint 也利用分割+inpainting 但用于对象移除；PIPE 反向使用实现对象添加
- **vs SmartBrush**: SmartBrush 需要用户提供 mask；PIPE 的模型不需要任何 mask 输入

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ "Paint by Inpaint"的核心洞察极其优雅，将困难问题通过对称性转化为简单问题
- 实验充分度: ⭐⭐⭐⭐ 三个基准测试 + 人类评估 + 通用编辑扩展 + 多指标评估
- 写作质量: ⭐⭐⭐⭐ 动机清晰，管线描述详细，图示丰富
- 价值: ⭐⭐⭐⭐⭐ PIPE 数据集本身具有极高价值，框架思路对数据构建领域有广泛启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] QMamba: On First Exploration of Vision Mamba for Image Quality Assessment](../../ICML2025/segmentation/qmamba_on_first_exploration_of_vision_mamba_for_image_quality_assessment.md)
- [\[NeurIPS 2025\] FineRS: Fine-grained Reasoning and Segmentation of Small Objects with Reinforcement Learning](../../NeurIPS2025/segmentation/finers_fine-grained_reasoning_and_segmentation_of_small_objects_with_reinforceme.md)
- [\[CVPR 2025\] The Power of Context: How Multimodality Improves Image Super-Resolution](the_power_of_context_how_multimodality_improves_image_super-resolution.md)
- [\[CVPR 2025\] OverLoCK: An Overview-first-Look-Closely-next ConvNet with Context-Mixing Dynamic Kernels](overlock_an_overview-first-look-closely-next_convnet_with_context-mixing_dynamic.md)
- [\[ICCV 2025\] LEGION: Learning to Ground and Explain for Synthetic Image Detection](../../ICCV2025/segmentation/legion_learning_to_ground_and_explain_for_synthetic_image_detection.md)

</div>

<!-- RELATED:END -->
