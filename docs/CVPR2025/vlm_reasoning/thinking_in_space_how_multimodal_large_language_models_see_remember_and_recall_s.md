---
title: >-
  [论文解读] Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces
description: >-
  [CVPR 2025][多模态VLM][视觉空间智能] 本文提出 VSI-Bench，一个基于视频的视觉空间智能基准（5000+ QA对），系统评估了 MLLM 的空间推理能力，发现空间推理是主要瓶颈，传统语言推理技术（CoT等）无法提升性能，但显式生成认知地图可改善空间距离推理。 - 领域现状： MLLM 在语言和视觉理解…
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "视觉空间智能"
  - "认知地图"
  - "视频理解"
  - "空间推理"
  - "基准测试"
---

# Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces

**会议**: CVPR 2025  
**arXiv**: [2412.14171](https://arxiv.org/abs/2412.14171)  
**代码**: [https://github.com/vision-x-nyu/thinking-in-space](https://github.com/vision-x-nyu/thinking-in-space)  
**领域**: 多模态VLM  
**关键词**: 视觉空间智能, 认知地图, 视频理解, 空间推理, 基准测试

## 一句话总结
本文提出 VSI-Bench，一个基于视频的视觉空间智能基准（5000+ QA对），系统评估了 MLLM 的空间推理能力，发现空间推理是主要瓶颈，传统语言推理技术（CoT等）无法提升性能，但显式生成认知地图可改善空间距离推理。

## 研究背景与动机
- **领域现状**: MLLM 在语言和视觉理解方面取得显著进展，但视觉空间智能（感知空间、记忆布局、回忆空间信息）尚未被充分探索
- **现有痛点**: 现有视频理解基准主要关注内容级别理解（识别、感知），缺少对3D空间推理能力的评估；且现有空间相关工作多基于2D图像或纯语言
- **核心矛盾**: 人类能从视频序列中自然地构建空间心理模型，但 MLLM 是否也能从视频中"空间思考"尚不清楚
- **本文解决什么**: 构建系统性的视觉空间智能评估体系，深入分析 MLLM 的空间推理强项和瓶颈
- **切入角度**: 从认知心理学的双重编码理论出发，分别从语言（自我解释）和视觉（认知地图）两个维度探测模型的空间思维
- **核心idea**: 空间推理是 MLLM 的主要瓶颈，显式构建认知地图可以增强距离推理能力

## 方法详解

### 整体框架
VSI-Bench 是一个视频级空间智能基准，包含 288 个真实室内场景视频和 5000+ QA 对。数据来源于 ScanNet、ScanNet++、ARKitScenes 三个 3D 重建数据集。基准包含 8 种任务、3 大类型：配置型（物体计数、相对距离、相对方向、路线规划）、测量估计型（物体大小、房间大小、绝对距离）和时空型（出现顺序）。评估体系包括多选题准确率和数值预测的 Mean Relative Accuracy (MRA)。

### 关键设计
1. **VSI-Bench 基准构建**:
    - 功能：系统化评估 MLLM 的视觉空间智能
    - 核心思路：将多个 3D 数据集统一为标准化格式，结合模板自动生成和人工标注生成 QA 对，通过人工审核迭代保证质量
    - 设计动机：视频数据比静态图像更接近人类观察空间的方式，3D 数据集提供精确物体级标注作为 ground truth

2. **Mean Relative Accuracy (MRA) 评估指标**:
    - 功能：评估数值预测任务（距离/尺寸估计）的准确度
    - 核心思路：在多个置信阈值 $\mathcal{C}=\{0.5, 0.55, \ldots, 0.95\}$ 上计算相对准确率并平均：$\mathcal{MRA}=\frac{1}{10}\sum_{\theta\in\mathcal{C}}\mathbb{1}(\frac{|\hat{y}-y|}{y}<1-\theta)$
    - 设计动机：传统精确匹配无法衡量数值预测的接近程度，MRA 在多粒度上综合评估预测质量

3. **认知地图探测与增强**:
    - 功能：可视化探测 MLLM 的空间内部表征，并利用其增强空间推理
    - 核心思路：提示 MLLM 在 $10 \times 10$ 网格上预测物体中心位置生成认知地图，发现模型具有强局部空间感知但全局感知较弱；将认知地图生成作为问答的前置步骤可提升相对距离准确率
    - 设计动机：受认知心理学中人类通过心理意象进行空间推理的启发，探索 MLLM 是否也能通过类似机制改善空间推理

### 损失函数 / 训练策略
本文是评估工作，不涉及训练新模型。所有模型评估在零样本设置下进行，使用贪婪解码确保可复现性。评估涵盖3个闭源模型（GPT-4o、Gemini-1.5 Flash/Pro）和12个开源模型（InternVL2、VILA、LongVA、LLaVA系列等），参数规模从0.5B到72B。

## 实验关键数据

### 主实验

| 模型 | 平均分 | 物体计数 | 绝对距离 | 相对距离 | 相对方向 | 路线规划 | 出现顺序 |
|------|--------|---------|---------|---------|---------|---------|---------|
| 人类水平 | 79.2 | 94.3 | 47.0 | 94.7 | 95.8 | 95.8 | 100.0 |
| Gemini-1.5 Pro | 45.4 | 56.2 | 43.6 | 51.3 | 46.3 | 36.0 | 34.6 |
| LLaVA-Video-72B | 40.9 | 48.9 | 35.3 | 42.4 | 36.7 | 35.0 | 48.6 |
| GPT-4o | 34.0 | 46.2 | 38.2 | 37.0 | 41.3 | 31.5 | 28.5 |

### 消融实验 — 推理技巧对比

| 方法 | 平均变化 | 说明 |
|------|---------|------|
| Zero-Shot CoT | -4% | 鼓励思考反而降低性能 |
| Self-Consistency | -1.1% | 多次采样投票也无帮助 |
| Tree-of-Thoughts | -4% | 计划+推理模式同样失败 |
| 认知地图增强 | +10% (相对距离) | 显式生成地图显著提升距离推理 |

### 关键发现
- 人类在配置和时空任务（94%-100%）远超 MLLM，但在测量估计任务差距较小，说明 MLLM 在定量估计上有相对优势
- 约 71% 的错误源自空间推理（关系推理 + 自我中心-异心中心变换），视觉感知和语言智能并非瓶颈
- MLLM 的认知地图具有强局部感知（相邻物体准确率 64%）但全局感知严重退化
- 使用 GT 认知地图可将相对距离准确率从 46% 提升至 66%-78%
- 开源72B模型（LLaVA-Video/OneVision）已接近闭源模型（仅差4-5%），但大多数小模型低于随机基线
- 在 VideoMME 上 CoT 可带来 1.6% 提升，但在 VSI-Bench 上反而下降，说明空间任务本质不同于一般视频理解

## 亮点与洞察
- 首次系统性地从认知心理学角度建模和评估 MLLM 的视觉空间智能，提供了完整的能力分类法（视觉感知、语言智能、时序处理、空间推理四大维度）
- 发现传统 CoT 方法在空间推理上完全失效这一反直觉结论，说明空间智能需要不同于语言推理的增强策略
- 用自我解释分析法定量揭示了 71% 错误来自空间推理而非视觉感知或语言理解，为后续研究明确了改进方向
- 认知地图作为"空间思维"的具象化工具，为 MLLM 的空间推理增强提供了新方向——相邻物体定位准确率达64%证明模型具有涌现的局部空间意识
- MRA 指标设计巧妙，解决了数值预测评估中精确匹配过于严格的问题
- 使用 GT 认知地图时相对距离任务从 46% 跃升至 78%，说明准确的空间世界模型是解题的关键一环
（住宅、办公室、工厂），室外和大规模环境的空间推理未涉及
- 认知地图目前仅是 $10\times10$ 的粗粒度网格，更精细的空间表征可能带来更大提升（$20\times20$ 网格在MLLM预测地图上反而降低性能，但GT地图下提升至78%）
- 未探索训练时引入空间推理数据或自监督空间目标的效果，这是有前景的未来方向
- 视频帧采样策略对空间理解的影响未被系统研究
- 仅测试了英文提示，不同语言描述空间关系的差异未被考虑 的粗粒度网格，更精细的空间表征可能带来更大提升
- 未探索训练时引入空间推理数据或自监督空间目标的效果
- 视频帧采样策略对空间理解的影响未被系统研究

## 相关工作与启发
- **vs SpatialVLM**: SpatialVLM 通过图像微调赋予空间能力，本文发现仅微调可能不够，需要更根本的空间表征增强（如认知地图）
- **vs Video-MME**: Video-MME 关注内容级视频理解（识别、叙事），本文拓展到3D空间level的理解，是互补关系
- **vs EgoSchema**: EgoSchema 评估自我中心视频理解能力，本文进一步强调自我中心到异心中心的变换能力（allocentric transformation）
- **vs OpenEQA**: OpenEQA也用自我中心视频评估空间理解，但本文提供更系统的能力分类和更深入的错误分析
- **vs SpatialBot**: SpatialBot关注2D图像中的空间感知，本文拓展至视频级的空间记忆与推理

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个系统性视频空间智能基准，认知地图探测思路新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 15个模型全面评估，深入的错误分析和消融实验
- 写作质量: ⭐⭐⭐⭐⭐ 叙事结构清晰，从评估到分析到改进层层递进
- 价值: ⭐⭐⭐⭐⭐ 揭示了空间推理是MLLM的核心瓶颈，对具身智能和导航任务有重要指导意义

---

> 本笔记基于论文全文阅读生成，覆盖了 Method、Experiments 和 Analysis 全部内容。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Thinking in Dynamics: How Multimodal Large Language Models Perceive, Track, and Reason Dynamics in Physical 4D World](thinking_in_dynamics_how_multimodal_large_language_models_perceive_track_and_rea.md)
- [\[CVPR 2025\] Towards Understanding How Knowledge Evolves in Large Vision-Language Models](towards_understanding_how_knowledge_evolves_in_large_vision-language_models.md)
- [\[ICML 2026\] Thinking in Structures: Evaluating Spatial Intelligence in Constraint-Governed Spaces](../../ICML2026/multimodal_vlm/thinking_in_structures_evaluating_spatial_intelligence_in_constraint-governed_sp.md)
- [\[CVPR 2025\] Task Preference Optimization: Improving Multimodal Large Language Models with Vision Task Alignment](task_preference_optimization_improving_multimodal_large_language_models_with_vis.md)
- [\[CVPR 2025\] On the Out-of-Distribution Generalization of Multimodal Large Language Models](on_the_out-of-distribution_generalization_of_large_multimodal_models.md)

</div>

<!-- RELATED:END -->
