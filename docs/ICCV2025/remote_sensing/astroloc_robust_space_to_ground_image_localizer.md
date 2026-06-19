---
title: >-
  [论文解读] AstroLoc: Robust Space to Ground Image Localizer
description: >-
  [ICCV 2025][遥感][宇航员照片定位] 提出AstroLoc，首个利用30万张人工标注宇航员照片进行训练的太空对地定位模型，通过查询-卫星配对损失和无监督挖掘技术学习鲁棒的地球表面特征表征，在recall@1上平均提升35%，recall@100持续超过99%，已在实际中完成50万+照片的定位。
tags:
  - "ICCV 2025"
  - "遥感"
  - "宇航员照片定位"
  - "跨域图像检索"
  - "对比学习"
  - "无监督挖掘"
  - "太空对地观测"
---

# AstroLoc: Robust Space to Ground Image Localizer

**会议**: ICCV 2025  
**arXiv**: [2502.07003](https://arxiv.org/abs/2502.07003)  
**代码**: 无（数据集可在 [https://eol.jsc.nasa.gov/](https://eol.jsc.nasa.gov/) 获取）  
**领域**: 遥感 / 图像检索 / 地理定位  
**关键词**: 宇航员照片定位, 跨域图像检索, 对比学习, 无监督挖掘, 太空对地观测

## 一句话总结
提出AstroLoc，首个利用30万张人工标注宇航员照片进行训练的太空对地定位模型，通过查询-卫星配对损失和无监督挖掘技术学习鲁棒的地球表面特征表征，在recall@1上平均提升35%，recall@100持续超过99%，已在实际中完成50万+照片的定位。

## 研究背景与动机
国际空间站（ISS）的宇航员每天用手持相机拍摄大量地球照片，自2000年以来已积累超过500万张。这些照片具有独特价值：分辨率最高可达2米/像素、可拍摄倾斜视角、覆盖不同光照条件、是最高分辨率的开源地球观测数据。在气候科学、大气研究、城市规划和灾害响应中有重要应用。

然而，与卫星影像不同，宇航员照片没有自动地理定位。宇航员可以将相机指向视野范围内（约2000万平方公里）的任何方向，而一张照片覆盖的面积可能只有100平方公里，这相当于在0.0005%的可视区域中大海捞针。NASA将手动定位描述为"极其重要但极其耗时的工作"——30万张照片花费了数十万人工小时。

现有APL方法（如EarthLoc）仅使用卫星图像训练模型，从未利用已标注的30万张宇航员照片。这是关键局限：目标是定位宇航员照片，但模型从未见过这类图像。AstroLoc的核心创新在于：首次将这些宇航员照片引入训练流程，通过两种巧妙的训练技术（配对损失+无监督挖掘）充分利用跨域数据。

## 方法详解

### 整体框架
AstroLoc的训练流程包含两个并行分支：
1. **上分支**：将配对的宇航员-卫星图像输入**配对损失**，直接学习跨域对应关系
2. **下分支**：对卫星图像做聚类，按照宇航员照片在各聚类中的分布进行加权采样（**无监督挖掘**），构建训练batch送入Multi-Similarity损失

推理时：给定一张宇航员照片作为查询，在全球卫星图像数据库中通过最近邻搜索找到最相似的卫星瓦片，从而估计地理位置。

### 数据准备——30万弱标注照片的精确定位

这是论文的重要预处理贡献。30万张照片仅有弱标注（一个大致中心点坐标），无法直接用于需要精确footprint（四角坐标）的训练。解决方案：

1. 对每张查询，在5个zoom层级×4个旋转×4个覆盖瓦片 = 80个候选卫星图中搜索
2. 使用SuperPoint + LightGlue + EarthMatch进行特征匹配，估计精确footprint
3. 成功标注22.1万张（剩余因标注错误、云遮挡或地平线照片失败）
4. 每张查询与IoU > 0.2的所有卫星瓦片配对，生成86.5万查询-数据库训练对

### 关键设计

1. **查询-卫星配对损失（Query-Satellite Pairwise Loss）**:

    - 功能：直接学习宇航员照片和卫星图像的跨域特征对应关系
    - 核心思路：构建batch $\mathcal{P} = \{(q_i, d_i)\}_{i=1}^B$，每对图像有足够IoU，批内无地理重叠。包含吸引项和排斥项：
        - 吸引项：$\mathcal{L}_{pos} = \frac{1}{\alpha_1 B}\sum_{i=1}^B \log[1 + e^{-\alpha_1 \times \mathcal{S}(q_i, d_i)}]$
        - 排斥项覆盖查询-查询、查询-数据库、数据库-查询、数据库-数据库四种组合：$\varphi(x, y, \mathcal{Z}) = \log(1 + \sum_j e^{x \times \mathcal{S}(y, \mathcal{Z}_j)})$
    - 总损失：$\mathcal{L}_{pairs} = \mathcal{L}_{pos} + \mathcal{L}_{neg}$
    - 设计动机：跨域对比学习是解决宇航员照片-卫星图像域差距的直接方式

2. **无监督挖掘（Unsupervised Mining, MUM）**:

    - 功能：利用全球550万卫星图像进行训练，同时让训练分布偏向宇航员更关注的区域
    - 核心思路分三步演进：
        - **方案1（朴素采样）**：随机采样四元组，缺乏困难负样本，模型不够鲁棒
        - **方案2（数据库聚类）**：k-means聚出K个视觉特征相似的簇（森林、沙漠等），每个batch从同一簇采样以获得困难负样本。问题：沙漠/海洋等无信息簇浪费训练资源
        - **方案3（完整方案）**：将查询特征分配到K个卫星簇中，按查询数量$b_k$加权采样：$Pr(k) = \frac{b_k}{\sum_{i=1}^K b_i}$。宇航员多拍的区域（火山、冰川、湖泊）被更频繁采样，沙漠等区域较少采样
    - 损失：对采样的四元组应用Multi-Similarity Loss
    - 关键特性：(1) 首个用一个分布（查询）引导另一个分布（数据库）采样的挖掘方法；(2) 不需要查询标签，潜在可用全部500万未标注照片
    - 设计动机：卫星图像全球均匀分布，但宇航员照片分布不均（偏向显著区域），需要对齐两个分布

3. **模型架构**:

    - 骨干：DINOv2-base + SALAD描述子 + 线性降维层（8448→2048维）
    - 比AnyLoc轻量10倍以上（DINOv2-base vs DINOv2-giant）
    - 总损失：$\mathcal{L} = \lambda_1 \mathcal{L}_{pairs} + \lambda_2 \mathcal{L}_{MUM}$

### 训练策略
- 超参数：$t_{iou}=0.2, \alpha_1=\alpha_2=1, \beta_1=\beta_2=50, \lambda_1=\lambda_2=1, K=50$
- 批量大小48，学习率5e-5，Adam优化器
- 训练30k迭代，每5000迭代重新计算聚类特征
- 评估时对每张图做4个90°旋转增强
- Texas数据集用作验证集

## 实验关键数据

### 主实验（原始测试集 Recall@N）

| 方法 | Texas R@1 | Alps R@1 | California R@1 | Gobi R@1 | Amazon R@1 | Toshka R@1 |
|------|-----------|----------|---------------|----------|------------|------------|
| AnyLoc | 44.1 | 40.7 | 48.7 | 28.7 | 38.6 | 63.7 |
| EarthLoc | 55.9 | 58.4 | 58.0 | 51.1 | 47.2 | 72.2 |
| EarthLoc++ | 80.0 | 80.6 | 82.9 | 67.6 | 73.6 | 90.1 |
| **AstroLoc** | **96.1** | **98.1** | **97.4** | **94.6** | **93.0** | **99.0** |

扩展测试集（更具挑战性，包含所有查询）R@100均超过96%。

### 消融实验

| 配对损失 | 方案1 | 方案2 | 方案3(MUM) | Texas-L R@1 | Alps-L R@1 |
|---------|------|------|-----------|-------------|------------|
| ✓ | | | | 83.6 | 87.2 |
| | | | ✓ | 82.2 | 86.9 |
| ✓ | | | ✓ | **91.1** | **94.6** |
| | ✓ | | | 67.6 | 76.5 |
| | | ✓ | | 72.4 | 79.4 |

### 关键发现
- AstroLoc在原始测试集上R@100全部超过99%，已饱和现有基准
- 在更具挑战性的扩展测试集（L版本）上R@100仍超过96%
- 配对损失和MUM损失具有正交性——组合使用远优于各自单独使用
- 无监督挖掘（方案3）明显优于朴素采样和纯聚类方案
- 零样本迁移到"太空迷失"问题：R@1达52.7%，超出其他方法45%
- 零样本迁移到历史航天飞机照片（40年前胶片照片）：R@1达82.0%
- 全球搜索（88万张数据库）：R@100达96.8%
- 已在实际中定位50万+照片，预计数月内清空ISS照片待定位积压

## 亮点与洞察
- 数据工程的典范：将30万弱标注照片通过自动化流水线精确标注，创造了宝贵的训练资源
- 无监督挖掘的思想优雅：用一个分布引导另一个分布的采样，完全不需要查询标签
- 实际落地价值极高：不是学术花瓶，已在NASA实际使用中定位数十万照片
- 模型泛化能力惊人：对40年前的胶片照片、微纳卫星照片等从未见过的域都有出色表现
- 轻量化设计：比AnyLoc小10倍但性能大幅领先

## 局限与展望
- 夜间照片和严重云覆盖的照片仍是难题
- 极端倾斜角度的"地球边缘"照片无法处理（footprint无效）
- 当前仅做粗检索，缺少学习型精细重排序
- 扩展到更高分辨率zoom级别可能需要更大数据库和更高效的检索
- 宇宙射线导致的时间戳比特翻转问题需额外处理

## 相关工作与启发
- **vs EarthLoc**: 相同架构下（EarthLoc++），AstroLoc仍领先16-27% R@1，证明训练数据和损失设计的关键作用
- **vs AnyLoc**: AnyLoc特征维度49152 (需235GB存储)，AstroLoc仅2048维(9GB)，速度快20倍且性能大幅领先
- **vs UAV定位**: 虽然问题相关，但宇航员照片面临更极端的域差距（倾斜角度、大范围视野、ISS硬件遮挡等）

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次利用宇航员照片训练APL模型，无监督挖掘技术独创
- 实验充分度: ⭐⭐⭐⭐⭐ 6个原始+6个扩展测试集、3个跨域任务、详尽消融、全球搜索实验
- 写作质量: ⭐⭐⭐⭐⭐ 动机阐述引人入胜，问题重要性论证有力，图表清晰
- 价值: ⭐⭐⭐⭐⭐ 已实际部署、解决NASA真实需求的系统，学术价值和实用价值兼具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] ConGeo: Robust Cross-View Geo-Localization Across Ground View Variations](../../ECCV2024/remote_sensing/congeo_robust_cross-view_geo-localization_across_ground_view_variations.md)
- [\[ECCV 2024\] Weakly-Supervised Camera Localization by Ground-to-Satellite Image Registration](../../ECCV2024/remote_sensing/weakly-supervised_camera_localization_by_ground-to-satellite_image_registration.md)
- [\[ICCV 2025\] WildSAT: Learning Satellite Image Representations from Wildlife Observations](wildsat_learning_satellite_image_representations_from_wildlife_observations.md)
- [\[CVPR 2026\] Robust Remote Sensing Image–Text Retrieval with Noisy Correspondence](../../CVPR2026/remote_sensing/robust_remote_sensing_image-text_retrieval_with_noisy_correspondence.md)
- [\[CVPR 2025\] Learning Occlusion-Robust Vision Transformers for Real-Time UAV Tracking](../../CVPR2025/remote_sensing/learning_occlusion-robust_vision_transformers_for_real-time_uav_tracking.md)

</div>

<!-- RELATED:END -->
