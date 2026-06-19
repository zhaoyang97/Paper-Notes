---
title: >-
  [论文解读] PetFace: A Large-Scale Dataset and Benchmark for Animal Identification
description: >-
  [ECCV 2024][LLM评测][动物识别] 构建了包含13个动物科、319个品种、257,484个个体（超100万张图像）的大规模动物面部识别数据集PetFace，并建立了已见个体重识别和未见个体验证两套基准测试，为动物非侵入式自动识别提供基础设施。 动物个体识别在行为监测、栖息地调查、走失动物寻找和健康检查等场景中至…
tags:
  - "ECCV 2024"
  - "LLM评测"
  - "动物识别"
  - "人脸识别"
  - "大规模数据集"
  - "重识别"
  - "benchmark"
---

# PetFace: A Large-Scale Dataset and Benchmark for Animal Identification

**会议**: ECCV 2024  
**arXiv**: [2407.13555](https://arxiv.org/abs/2407.13555)  
**代码**: 有 ([https://dahlian00.github.io/PetFacePage/](https://dahlian00.github.io/PetFacePage/))  
**领域**: LLM评测  
**关键词**: 动物识别, 人脸识别, 大规模数据集, 重识别, benchmark

## 一句话总结

构建了包含13个动物科、319个品种、257,484个个体（超100万张图像）的大规模动物面部识别数据集PetFace，并建立了已见个体重识别和未见个体验证两套基准测试，为动物非侵入式自动识别提供基础设施。

## 研究背景与动机

动物个体识别在行为监测、栖息地调查、走失动物寻找和健康检查等场景中至关重要。传统方法（耳标、纹身、趾剪）具有侵入性，会导致动物应激和疼痛，应尽量减少使用。数字ID等新工具虽然减少了侵害，但需要逐一为动物安装设备，成本高且仍有应激。

人类面部识别已高度成熟，得益于大规模数据集和基准（如MS-Celeb、VGGFace2等）。然而动物面部识别的发展受限于**数据集的严重匮乏**：

| 数据集 | 物种 | 个体数 | 图像数 |
|--------|------|--------|--------|
| CTai | 黑猩猩 | 78 | 5,078 |
| DogFaceNet | 狗 | 1,393 | 8,363 |
| MacaqueFaces | 猴 | 34 | 6,280 |
| **PetFace (本文)** | **13科** | **257,484** | **1,012,934** |

PetFace的个体数量是此前最大动物面部数据集（DogFaceNet）的110多倍，且跨越13个动物科和319个品种，填补了动物面部识别领域的数据空白。

## 方法详解

### 整体框架

PetFace不是一个模型方法论文，而是一个**数据集+基准论文**。其核心贡献包括：

1. **数据集构建**：从网络中高效收集大规模高质量动物面部图像
2. **两套评估协议**：已见个体重识别(Re-ID)和未见个体验证(Verification)
3. **基准实验**：在多种损失函数和预训练模型上建立基线

### 关键设计

**1. 数据采集策略**

从两类网络来源获取图像：
- **宠物商店网站**：提供高质量多角度图像，含详细个体信息（颜色、性别、品种）
- **动物领养网站**：提供多样背景下的图像，由宠物主人上传

每个地区仅选一个领养网站以避免重复。黑猩猩数据额外来自合作研究机构。初始收集1,443,737张图像/325,420个个体。

**2. 面部检测与对齐**

使用AnyFace模型检测面部关键点。因不同物种面部结构差异大，为每个物种定义独立的参考点和对齐方式。先选一张正面参考图，计算所有图像与参考对齐后的平均关键点位置作为对齐目标。

**3. 数据过滤**

两阶段过滤流程：
- **自动阶段**：移除检测到多个面部的图像
- **人工阶段**：由作者（约100人时）逐一检查并移除非动物图像、对齐不佳的图像。最终保留70%的初始图像。

**4. 细粒度标注**

| 标注类型 | 覆盖率 | 说明 |
|----------|--------|------|
| 性别 | 94% (240,861个体) | 从网站提取 |
| 品种 | 8个物种 (319种) | Cat, Dog, Guinea pig等 |
| 颜色/花纹 | 11个物种 | 两层层级标注 |

### 损失函数 / 训练策略

基准实验使用ResNet-50骨干网络，比较四种损失函数：

1. **Softmax**：基础分类损失
2. **Center Loss**：最小化类内变异，使同一个体的特征更紧凑
3. **Triplet Loss**：使正样本对距离小于负样本对
4. **ArcFace Loss**：在角度空间中添加边际惩罚，增强特征区分度

## 实验关键数据

### 主实验（表格）

**重识别结果（Top-1 准确率%）—— 按物种分**

| 方法 | Cat | Dog | Chimp | Chinchilla | Guinea | Hamster | Hedgehog | 平均 |
|------|-----|-----|-------|------------|--------|---------|----------|------|
| Softmax | 30.46 | 59.14 | 41.70 | 58.13 | 60.07 | 38.27 | 27.81 | 41.88 |
| Center | 0.00 | 0.00 | 5.38 | 29.76 | 31.77 | 9.46 | 13.76 | 9.81 |
| ArcFace | 54.29 | 77.86 | 43.27 | 67.34 | 67.90 | 47.37 | 30.90 | 51.23 |
| Joint ArcFace | **70.30** | 68.75 | 34.30 | **69.86** | 68.66 | **54.33** | **44.38** | **53.80** |

**验证结果（AUC%）**

| 方法 | Cat | Dog | Chimp | Chinchilla | Guinea pig | 平均 |
|------|-----|-----|-------|------------|------------|------|
| Softmax | 97.97 | 98.98 | 85.22 | 84.44 | 95.30 | 90.38 |
| Triplet | 96.94 | 97.97 | 77.10 | 76.12 | 83.37 | 83.48 |
| ArcFace | 97.71 | **99.45** | 83.76 | **87.70** | **96.03** | **91.30** |

### 消融实验（表格）

与其他数据集训练模型的对比：

| 预训练数据 | 架构 | Cat验证AUC | Dog验证AUC | 平均AUC |
|------------|------|-----------|-----------|---------|
| ImageNet | ResNet-50 | 73.71 | 73.04 | - |
| CLIP | ResNet-50 | 74.98 | 87.22 | - |
| MegaDescriptor | SwinT-B | 88.52 | 97.44 | - |
| **PetFace (ArcFace)** | ResNet-50 | **97.71** | **99.45** | **91.30** |

### 关键发现

1. **ArcFace是最适合动物面部识别的损失函数**：在重识别和验证任务上均一致领先
2. **Center Loss大幅失败**：在Cat和Dog等个体数极多的类别上完全无法学习（0%准确率），说明纯类内紧凑约束不够
3. **联合训练的双面性**：跨物种联合训练（Joint）在Cat上从54.29%提升到70.30%，但在Chimp上从43.27%下降到34.30%，说明不平衡数据的联合训练仍需改进
4. **PetFace训练的模型显著优于其他数据集**：即使使用简单的ResNet-50，也优于在MegaDescriptor（33个数据集联合）上训练的SwinTransformer
5. **跨物种泛化有潜力**：在未见动物科上也展示了一定的泛化能力

## 亮点与洞察

- **数据获取策略创新**：巧妙利用宠物商店和领养网站作为数据来源，每个页面天然对应一个个体ID，避免了昂贵的实地拍摄
- **规模优势明显**：从1,393个体到257,484个体的飞跃，使得未见个体验证终于有了可靠的评估基础
- **细粒度标注的价值**：品种、颜色、性别等标注可用于构建更有挑战性的细粒度评估（如同品种不同个体区分）
- **揭示了动物ID的独特挑战**：与人脸不同，动物面部结构跨物种差异巨大，统一模型难度高

## 局限与展望

1. **数据来源偏向宠物**：主要是家养动物，野生动物（如斑马、鲸鱼等）未覆盖
2. **图像质量不均**：网络来源的图像质量、光照、背景差异大
3. **个体间图像数差异大**：部分个体仅有2-3张图像，限制了训练效果
4. **联合训练策略待优化**：当前简单的跨物种联合训练在类别不平衡时效果不稳定
5. **缺少3D信息**：仅使用2D面部图像，未利用3D面部形状信息

## 相关工作与启发

- **与WildlifeDatasets的区别**：WildlifeDatasets聚合了33个已有小数据集，但每个数据集仍独立且个体数少；PetFace是统一收集的大规模新数据集
- **人脸识别技术的直接迁移**：ArcFace等人脸识别技术在动物面部上同样有效，证实了方法的领域通用性
- **启发**：动物ID系统若要大规模部署，必须解决跨物种统一建模的挑战——可能需要物种感知的分层识别架构

## 评分

| 维度 | 分数 (1-5) | 评价 |
|------|-----------|------|
| 新颖性 | 4 | 数据集规模和覆盖范围的跃迁式提升 |
| 技术深度 | 3 | 模型方面主要是已有方法的应用，创新在数据集构建 |
| 实验充分性 | 4 | 多种损失函数、多种预训练对比、跨物种评估全面 |
| 写作质量 | 4 | 结构清晰，数据集构建过程描述详细 |
| 实用价值 | 4.5 | 对动物识别社区有重大基础设施价值 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Sync from the Sea: Retrieving Alignable Videos from Large-Scale Datasets](sync_from_the_sea_retrieving_alignable_videos_from_large-scale_datasets.md)
- [\[ACL 2025\] HellaSwag-Pro: A Large-Scale Bilingual Benchmark for Evaluating the Robustness of LLMs in Commonsense Reasoning](../../ACL2025/llm_evaluation/hellaswag-pro_a_large-scale_bilingual_benchmark_for_evaluating_the_robustness_of.md)
- [\[ICML 2026\] Authority, Truth, and Citation Bias: A Large-Scale Multi-Domain Benchmark for Studying Epistemic Susceptibility in Large Language Models](../../ICML2026/llm_evaluation/authority_truth_and_citation_bias_a_large-scale_multi-domain_benchmark_for_study.md)
- [\[ICLR 2026\] AnesSuite: A Comprehensive Benchmark and Dataset Suite for Anesthesiology Reasoning](../../ICLR2026/llm_evaluation/anessuite_a_comprehensive_benchmark_and_dataset_suite_for_anesthesiology_reasoni.md)
- [\[ACL 2025\] TiC-LM: A Web-Scale Benchmark for Time-Continual LLM Pretraining](../../ACL2025/llm_evaluation/tic-lm_a_web-scale_benchmark_for_time-continual_llm_pretraining.md)

</div>

<!-- RELATED:END -->
