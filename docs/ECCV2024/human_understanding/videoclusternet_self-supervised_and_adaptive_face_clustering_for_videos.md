---
title: >-
  [论文解读] VideoClusterNet: Self-Supervised and Adaptive Face Clustering for Videos
description: >-
  [ECCV 2024][人体理解][人脸聚类] VideoClusterNet 提出了一种全自监督视频人脸聚类方法：通过自蒸馏机制自适应微调通用人脸识别模型，并设计了一种基于学习损失度量的无参数聚类算法，在电影/电视剧场景中达到 SOTA。 领域现状： 视频人脸聚类旨在将视频中检测到的人脸按身份分组，是视频场景描述、视频问答…
tags:
  - "ECCV 2024"
  - "人体理解"
  - "人脸聚类"
  - "自监督学习"
  - "自蒸馏"
  - "视频理解"
  - "无参数聚类"
---

# VideoClusterNet: Self-Supervised and Adaptive Face Clustering for Videos

**会议**: ECCV 2024  
**arXiv**: [2407.12214](https://arxiv.org/abs/2407.12214)  
**代码**: 无  
**领域**: 人体理解 / 视频人脸聚类  
**关键词**: 人脸聚类, 自监督学习, 自蒸馏, 视频理解, 无参数聚类

## 一句话总结

VideoClusterNet 提出了一种全自监督视频人脸聚类方法：通过自蒸馏机制自适应微调通用人脸识别模型，并设计了一种基于学习损失度量的无参数聚类算法，在电影/电视剧场景中达到 SOTA。

## 研究背景与动机

**领域现状**: 视频人脸聚类旨在将视频中检测到的人脸按身份分组，是视频场景描述、视频问答、视频理解等任务的重要基础。随着影视制作内容的增长，电影/电视剧领域的人脸聚类需求日益增长。

**现有痛点**:
   - 通用预训练人脸识别（Face ID）模型在电影场景中效果有限：高动态范围、独特电影风格、极端姿态/光照/表情变化
   - **负样本挖掘的困难**：大多数方法依赖对比学习的正负样本对，负样本来自共现 track 的时间约束，但错误的负样本会导致次优解
   - **聚类超参数依赖**：Bottom-up 方法需要预定义距离阈值，Top-down 方法需要预定义聚类数量——两者都是非直觉的人工参数
   - 缺乏高质量的电影人脸聚类基准数据集

**核心矛盾**: 电影场景中同一角色的人脸变化巨大（姿态、光照、装扮），但通用模型不适配特定视频风格，且标注成本高昂使得监督微调不可行。

**本文目标** (1) 无需人工标注地自适应微调 Face ID 模型到特定视频；(2) 设计无需任何用户输入参数的聚类算法。

**切入角度**: 将自蒸馏（DINO 风格）的自监督学习应用于视频人脸 track 的正样本对学习，完全跳过负样本挖掘；用训练损失函数本身作为聚类的距离度量。

**核心 idea**: 仅用正样本对的自蒸馏微调 + 将训练损失直接作为聚类距离度量 = 完全自动化的视频人脸聚类。

## 方法详解

### 整体框架

VideoClusterNet 分为三个阶段：
- **Stage 1 — Track 预处理**: 场景切割 → 人脸检测 → 质量过滤 → 运动跟踪生成 face tracks
- **Stage 2 — 自监督模型微调**: 通过 teacher-student 自蒸馏在 track 内和 track 间的正样本对上微调 Face ID 模型
- **Stage 3 — 无参数聚类**: 使用微调模型的嵌入和训练损失作为距离度量，自适应计算每个 track 的匹配阈值进行层次聚类

### 关键设计

1. **自监督模型微调 (Self-Supervised Model Finetuning)**:

   **功能**: 将通用 Face ID 模型自适应到特定视频中观察到的人脸变化。

   **核心思路**: 采用 teacher-student 自蒸馏机制。给预训练 Face ID 模型附加随机初始化的 MLP head，复制为 teacher 和 student 分支。从同一/匹配 track 采样正样本对，通过交叉熵风格的 similarity loss 训练：

    $L_{ssl} = -1 \times softmax\left(\frac{embed_t - c}{temp}\right) \times \log(softmax(embed_s))$

   其中 $embed_t$、$embed_s$ 分别为 teacher/student 输出，$c$ 为 teacher 嵌入的滚动均值。梯度仅通过 student 分支反传，teacher 权重通过 student 的指数移动平均更新。

   **设计动机**: 
    - **仅用正样本对**：完全跳过负样本挖掘，避免错误负样本（如同一角色在不同场景被误认为不同人）导致的次优解
    - **分阶段训练**：先冻结基模型只训练 head（head 随机初始化），再联合微调——确保训练稳定性
    - **数据增强**：水平翻转、旋转、色温变化——减少对视频自然变化的依赖

2. **粗粒度 Track 匹配 (Coarse Track Matching)**:

   **功能**: 跨 shot 边界匹配可能属于同一身份的不同 tracks，为微调提供更多样本对。

   **核心思路**: 对每个 track 的所有人脸嵌入拟合多元高斯分布 $N_{t_j}(\mu_{t_j}, \Sigma_{t_j})$。自适应阈值通过取该 track 内嵌入 PDF 值最低 25% 的均值确定。若另一 track 的均值嵌入的 PDF 值 ≥ 阈值，则认为两个 track 匹配。

   **设计动机**: 单个 track 局限于一个 shot，光照/外观变化有限；通过跨 shot 匹配，模型能接触到角色在整个视频中的全部变化范围。微调和粗匹配交替进行，逐步提升匹配质量。

3. **无参数 Track 聚类 (Parameter-Free Track Clustering)**:

   **功能**: 将所有 track 按身份聚类，无需用户指定任何参数。

   **核心思路**: 
    - **距离度量**: 直接使用 SSL 训练损失 $L_{ssl}$ 作为 track 相似度度量（非对称，取两方向均值）
    - **自适应阈值**: 每个 track 内所有采样人脸两两计算 $L_{ssl}$，取均值作为该 track 的匹配阈值
    - **层次聚合**: 初始每个 track 为一个 cluster → 两两比较 → 若匹配值低于任一方的阈值则合并 → 传递性合并 → 迭代直到无新合并

   **设计动机**: 
    - 训练损失恰好在微调过程中优化了嵌入空间，用它作为距离度量比 Euclidean/Cosine 更适配
    - 每个 track 有独立阈值，适应模型对不同身份的识别信心差异

### Track 质量估计

采用 SER-FIQ 方法通过 dropout 评估模型对人脸的确定性。低质量 track 被标记为 Unknown 并排除在聚类外。使用 MAD（中位数绝对偏差）检测异常值。

### 损失函数 / 训练策略

- 训练策略：冻结 base model → 训练 MLP head → 联合微调 + 粗匹配交替
- 所有超参数（epoch、batch size、learning rate）跨数据集不变
- 图像增强：水平翻转、旋转、色温变化

## 实验关键数据

### 主实验 — BBT 和 BVS 电视剧基准

| 方法 | BBT S01 Combined WCP(%) | BVS S05 Combined WCP(%) |
|------|------------------------|------------------------|
| SCTL | 66.48（仅 E1） | - |
| TSiam | 96.4（仅 E1） | 92.46（仅 E2） |
| SSiam | 96.2（仅 E1） | 90.87（仅 E2） |
| MLR | 83.71 | 66.37 |
| BCL | 89.63 | 83.62 |
| CCL | 98.2（仅 E1） | 92.1（仅 E2） |
| VCTRSF | 94.20 | - |
| **Ours** | **98.70** | **96.10** |

### 消融实验 — MovieFaceCluster: The Hidden Soldier

| 消融维度 | 配置 | 聚类准确率(%) | 聚类比率(Pred/GT) |
|---------|------|-------------|-----------------|
| 模型微调 | 未微调 + HAC | 86.10 | - |
| 模型微调 | **微调 + HAC** | **91.52** | - |
| 聚类算法 | HAC + Cosine | 91.52 | 1.43 (30/21) |
| 聚类算法 | Ours + Cosine | 93.70 | 2.0 (42/21) |
| 聚类算法 | Ours + Euclidean | 96.50 | 3.5 (74/21) |
| 聚类算法 | **Ours + Loss Func.** | **98.50** | **1.04 (22/21)** |
| 基模型 | FaRL-P16 (baseline/ours) | 78.7 → 90.2 | - |
| 基模型 | VGGFace2-R50 (baseline/ours) | 84.2 → 95.7 | - |
| 基模型 | ArcFace-R100 (baseline/ours) | 86.1 → 98.5 | - |
| 基模型 | AdaFace-R100 (baseline/ours) | 86.9 → 98.4 | - |

### 关键发现

- **自监督微调带来 ~6% 的聚类准确率提升**（86.1→91.5），且对所有测试的 Face ID 模型（CNN/Transformer）均有 5-12% 提升
- **Loss 函数作为距离度量远优于预定义度量**: Loss Func. (98.5%, 比率 1.04) vs Cosine (91.5%, 比率 1.43) vs Euclidean (96.5%, 比率 3.5)——Loss 不仅准确率最高，预测的聚类数也最接近真实值
- **MovieFaceCluster 数据集**数据更具挑战：平均质量分 0.706（低于 BBT 0.714 和 BVS 0.712），平均 23.2 个角色（远超 BBT 6.33 和 BVS 14.5）
- 在 BBT 上单集最高 99.70%（S1E1），Combined 98.70%
- 在 BVS 上单集最高 99.10%（S5E2），Combined 96.10%
- 9 部电影的 MovieFaceCluster 上全面超越所有方法

## 亮点与洞察

1. **"只用正样本"的极简思路**: 完全跳过负样本挖掘，一举解决了视频人脸聚类中负样本构建困难（共现 track 不一定是不同人）的问题
2. **训练损失 = 聚类度量**的优雅闭环: 嵌入空间针对该度量优化，所以用它做聚类天然最优——不需要对嵌入空间做任何假设
3. **自适应阈值消除超参数**: 每个 track 有自己的匹配阈值，自动适应模型对不同身份的信心差异
4. **通用性强**: 对 CNN（ArcFace-R100）和 Transformer（FaRL-P16）架构均显著提升

## 局限与展望

1. **预训练偏见传播**: 若通用 Face ID 模型对两个不同人有错误的高相似度，自蒸馏可能强化这种错误
2. **计算效率**: 每个视频需要独立微调模型，不适合实时应用——未来可探索 test-time adaptation 或 few-shot 方法
3. **缺少手部/服装等上下文利用**: 仅依赖人脸特征，未利用服装、体型等上下文信息
4. **MovieFaceCluster 数据集规模有限**: 仅 9 部电影，未涵盖更大规模的场景
5. **聚类迭代收敛速度**: 层次聚合的迭代次数未分析，大规模 track 集的效率可能成为瓶颈

## 相关工作与启发

- **自蒸馏 (DINO/iBOT 风格)**: 本文将图像级自蒸馏巧妙适配到视频人脸 track 的正样本对学习，展示了 SSL 方法在特定领域微调中的潜力
- **TSiam/SSiam**: 需要复杂的负样本挖掘策略（共现 track 约束、伪相关反馈），本文证明仅靠正样本就能超越
- **BCL**: 需要输入已知聚类数量，本文完全自动化
- **VCTRSF**: 视频中心 Transformer 方法，利用时间建模但同样需要人工参数

## 评分

- **新颖性**: ⭐⭐⭐⭐ "正样本 only + 训练损失作距离度量"的闭环设计简洁而新颖
- **实验充分度**: ⭐⭐⭐⭐ BBT/BVS/MovieFaceCluster 三大数据集 + 微调/聚类/基模型三方面消融
- **写作质量**: ⭐⭐⭐⭐ 方法动机清晰，算法伪代码完整
- **实用价值**: ⭐⭐⭐ 无开源代码、需逐视频微调限制了实际应用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Pose-Aware Self-Supervised Learning with Viewpoint Trajectory Regularization](pose-aware_self-supervised_learning_with_viewpoint_trajectory_regularization.md)
- [\[ICCV 2025\] Bi-Level Optimization for Self-Supervised AI-Generated Face Detection](../../ICCV2025/human_understanding/bi-level_optimization_for_self-supervised_ai-generated_face_detection.md)
- [\[ECCV 2024\] AdaDistill: Adaptive Knowledge Distillation for Deep Face Recognition](adadistill_adaptive_knowledge_distillation_for_deep_face_rec.md)
- [\[CVPR 2025\] FSFM: A Generalizable Face Security Foundation Model via Self-Supervised Facial Representation Learning](../../CVPR2025/human_understanding/fsfm_a_generalizable_face_security_foundation_model_via_self-supervised_facial_r.md)
- [\[ECCV 2024\] Avatar Fingerprinting for Authorized Use of Synthetic Talking-Head Videos](avatar_fingerprinting_for_authorized_use_of_synthetic_talking-head_videos.md)

</div>

<!-- RELATED:END -->
