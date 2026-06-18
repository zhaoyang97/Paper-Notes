---
title: >-
  [论文解读] The Role of Visual Modality in Multimodal Mathematical Reasoning: Challenges and Insights
description: >-
  [ACL 2025][多模态VLM][多模态数学推理] 系统性揭示了现有多模态数学推理模型对视觉信息的利用极其有限——打乱或移除训练图像对模型性能影响甚微——并提出 HC-M3D 基准来真正测试视觉依赖性，发现主流模型无法识别图像中的细微差异。 多模态数学推理是近年来 LVLM 研究的热点，但一个关键问题被忽视：模型是否真正…
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "多模态数学推理"
  - "视觉依赖性"
  - "基准评估"
  - "图像编码器"
  - "数据集构建"
---

# The Role of Visual Modality in Multimodal Mathematical Reasoning: Challenges and Insights

**会议**: ACL 2025  
**arXiv**: [2503.04167](https://arxiv.org/abs/2503.04167)  
**代码**: [GitHub](https://github.com/Yufang-Liu/visual_modality_role)  
**领域**: 多模态VLM  
**关键词**: 多模态数学推理, 视觉依赖性, 基准评估, 图像编码器, 数据集构建

## 一句话总结

系统性揭示了现有多模态数学推理模型对视觉信息的利用极其有限——打乱或移除训练图像对模型性能影响甚微——并提出 HC-M3D 基准来真正测试视觉依赖性，发现主流模型无法识别图像中的细微差异。

## 研究背景与动机

多模态数学推理是近年来 LVLM 研究的热点，但一个关键问题被忽视：**模型是否真正利用了图像信息进行数学推理？**

现有方法（G-LLaVA、MathLLaVA、MAVIS、MultiMath）大多聚焦于提升训练数据的多样性和质量，并在各种基准上报告性能提升。然而，作者发现了一个令人震惊的现象：

1. 在数学 SFT 阶段，**打乱图文对应关系后**，模型性能仅下降 0-4 个百分点，甚至部分数据集上性能反而提升
2. **完全移除图像后**，性能下降同样微弱
3. 与通用 VQA 任务形成鲜明对比——后者打乱图像导致性能骤降 30-40%

这表明现有多模态数学模型本质上主要依赖文本而非图像进行推理，视觉模态的作用被严重高估。

## 方法详解

### 整体框架

本文的工作分为三个递进层次：
1. **视觉扰动实验**：在统一架构下重现主流方法，通过打乱/移除图像验证视觉依赖性
2. **现有基准问题分析**：揭示文本信息过于丰富和选项泄露答案两大问题
3. **HC-M3D 基准构建与评估**：构建真正需要视觉依赖的数据集并测试主流模型

### 关键设计

1. **视觉模态扰动实验**: 在 LLaVA 三阶段（预训练-通用SFT-数学SFT）统一框架下，仅在数学 SFT 阶段改变视觉信息：① 正确图文对（baseline）、② 打乱图文对应（图像分布不变但对应关系错乱）、③ 完全移除图像。使用统一基础模型（DeepSeek-Math-RL-7B + CLIP-ViT-L-14-336）确保公平比较。四种方法（G-LLaVA、MathLLaVA、MAVIS、MultiMath）在五个数学基准上的实验结果均显示图像扰动影响极小。

2. **HC-M3D 基准数据集**: 人工构建的 1,851 样本多模态数学基准，遵循三个原则：① 数据正确性（问题可基于图文解答且答案正确）、② 视觉依赖性（答案必须依赖图像）、③ 图像-答案高相关性（对 429 个问题提供了修改后的相似图像，仅改变图像即改变正确答案）。数据来源包括 GeoQA（48.4%）、MathVista（14.7%）和精优网（37.0%）。

3. **多图像编码器实验**: 针对"组合多种图像编码器能否提升数学推理"这一流行做法进行验证。测试了 CLIP-B、CLIP-L、SigLIP、DINOv2 等编码器的多种组合（拼接隐层特征和投票两种融合方式），发现虽然通用 VQA 性能有所提升，但数学推理性能反而可能下降。

### 损失函数 / 训练策略

- 统一使用 LLaVA 三阶段训练流程：Stage1 预训练连接模块、Stage2 在通用数据上 SFT、Stage3 在数学数据上 SFT
- 基础语言模型：DeepSeek-Math-RL-7B
- 图像编码器：CLIP-ViT-L-14-336
- 选项扰动实验：打乱多选题选项顺序，观察预测是否一致

## 实验关键数据

### 主实验

视觉扰动对数学推理影响（5 数据集平均准确率）：

| 方法 | 正确图像 | 打乱图像 | 无图像 |
|------|---------|---------|--------|
| G-LLaVA | 35.2 | 34.6 (-0.6) | 35.4 (+0.2) |
| MathLLaVA | 36.1 | 32.5 (-3.6) | 32.7 (-3.4) |
| MAVIS | 37.1 | 34.7 (-2.4) | 34.2 (-2.9) |
| MultiMath | 40.1 | 37.2 (-2.9) | 36.4 (-3.7) |

对比通用 VQA 任务（LLaVA-1.5）：

| 设置 | VQAv2 | MMBench |
|------|-------|---------|
| 正确图像 | 79.2 | 66.8 |
| 打乱图像 | 46.2 (-33.0) | 26.6 (-40.2) |
| 无图像 | 60.8 (-18.4) | 54.3 (-12.5) |

### HC-M3D 基准评估

| 模型 | 参数量 | ALL↑ | DI↑ | BC↑ | AG↓ |
|------|--------|------|-----|-----|-----|
| G-LLaVA | 7B | 45.4 | 41.5 | 15.2 | 52.2 |
| MultiMath | 7B | 49.2 | 44.8 | 16.6 | 56.9 |
| InternVL2 | 8B | 41.9 | 38.3 | 16.6 | 34.0 |
| Qwen2-VL | 72B | 51.8 | 48.3 | 20.3 | 51.5 |
| GPT-4o | — | 49.0 | 45.8 | 19.1 | 42.0 |

### 关键发现

- **BC（两张相似图都答对）指标极低**：即使 GPT-4o 也只有 19.1%，说明模型无法识别图像间的细微差异
- **AG（一致性）偏高**：说明模型在图像改变后仍倾向给出原始答案，未真正感知视觉变化
- 纯文本 LLM（如 QWen-2.5-Math-7B）在数学基准上的平均分 33.3 已很接近多模态模型 G-LLaVA 的 35.2
- 选项打乱实验中，MultiMath 在 GeoQA 上的 BC/CR 仅为 9.5%，暴露了选项泄露问题
- 组合多编码器对通用 VQA 有效但对数学推理无效甚至有害

## 亮点与洞察

- 实验设计巧妙：通过打乱/移除图像这一简单操作，有力地质疑了"多模态=用了图像"的隐含假设
- HC-M3D 的"修改图、不改题"方法论非常精妙——直接用控制变量法验证视觉理解
- 与通用 VQA 的对比实验形成强烈反差，说明问题是数学领域特有的
- 隐含了一个重要洞察：当前多模态数学推理的评估基准本身存在系统性缺陷

## 局限与展望

- HC-M3D 数据集规模相对较小（1,851 样本），且聚焦平面几何
- 论文主要揭示问题但未提出有效解决方案——如何真正增强视觉依赖仍是开放问题
- 提出了几个方向但未验证：① 描述两图差异的预训练数据、② 更精细的图像编码器、③ 增强视觉依赖的损失函数
- 未分析为何 CLIP 编码器难以捕捉数学图像中的细微差异

## 相关工作与启发

- MathVerse 也强调了文本和视觉信息的平衡，但 HC-M3D 通过修改图像的方式更直接
- 与 CLIP 在细粒度视觉理解上的局限性（Liu et al. 2024b; Tong et al. 2024a）形成呼应
- 启发：多模态基准需要设计"反事实"样本（仅改变一个模态、观察模型反应）来真正测量模态贡献

## 评分

- 新颖性: ⭐⭐⭐⭐ — 视角独特，揭示了社区忽视的重要问题
- 实验: ⭐⭐⭐⭐ — 扰动实验设计精巧，数据集构建严谨
- 写作: ⭐⭐⭐⭐ — 层层递进，论证有力
- 实用性: ⭐⭐⭐⭐ — HC-M3D 基准将推动社区反思和改进多模态数学推理

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] MathCoder-VL: Bridging Vision and Code for Enhanced Multimodal Mathematical Reasoning](mathcoder-vl_bridging_vision_and_code_for_enhanced_multimodal_mathematical_reaso.md)
- [\[ACL 2025\] We-Math: Does Your Large Multimodal Model Achieve Human-like Mathematical Reasoning?](wemath_knowledge_reasoning.md)
- [\[ACL 2026\] A Survey of Multimodal Mathematical Reasoning: From Perception, Alignment to Reasoning](../../ACL2026/multimodal_vlm/a_survey_of_multimodal_mathematical_reasoning_from_perception_alignment_to_reaso.md)
- [\[ACL 2025\] MIRe: Enhancing Multimodal Queries Representation via Fusion-Free Modality Interaction](mire_enhancing_multimodal_queries_representation_via_fusion-free_modality_intera.md)
- [\[CVPR 2026\] Role-SynthCLIP: A Role-Play Driven Diverse Synthetic Data Approach](../../CVPR2026/multimodal_vlm/role-synthclip_a_role-play_driven_diverse_synthetic_data_approach.md)

</div>

<!-- RELATED:END -->
