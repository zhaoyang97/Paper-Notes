---
title: >-
  [论文解读] Bi-Level Optimization for Self-Supervised AI-Generated Face Detection
description: >-
  [ICCV 2025][人体理解][AI生成人脸检测] 提出BLADES方法，通过双层优化（bi-level optimization）将自监督预训练与AI生成人脸检测目标显式对齐：内层优化视觉编码器学习EXIF分类/排序和人脸篡改检测等前置任务，外层优化各任务权重以提升代理检测任务性能，实现不依赖合成人脸的跨生成器泛化检测。
tags:
  - "ICCV 2025"
  - "人体理解"
  - "AI生成人脸检测"
  - "双层优化"
  - "自监督学习"
  - "EXIF元数据"
  - "异常检测"
---

# Bi-Level Optimization for Self-Supervised AI-Generated Face Detection

**会议**: ICCV 2025  
**arXiv**: [2507.22824](https://arxiv.org/abs/2507.22824)  
**代码**: [github.com/MZMMSEC/AIGFD_BLO](https://github.com/MZMMSEC/AIGFD_BLO)  
**领域**: 人脸理解 / AI生成检测  
**关键词**: AI生成人脸检测, 双层优化, 自监督学习, EXIF元数据, 异常检测

## 一句话总结

提出BLADES方法，通过双层优化（bi-level optimization）将自监督预训练与AI生成人脸检测目标显式对齐：内层优化视觉编码器学习EXIF分类/排序和人脸篡改检测等前置任务，外层优化各任务权重以提升代理检测任务性能，实现不依赖合成人脸的跨生成器泛化检测。

## 研究背景与动机

现代生成模型（GAN、扩散模型）合成的人脸已达到以假乱真的程度，迫切需要鲁棒的检测方法。现有方法的核心问题：
- **模型依赖检测器**（监督学习方式）容易过拟合训练时见过的特定生成器的伪影，无法泛化到新兴生成技术
- **模型无关检测器**（利用生理不一致性如瞳孔形状、角膜高光等手工特征）可能遗漏先进生成器引入的细微统计差异
- **自监督特征方法**虽然跨生成器性能更好，但其前置任务并非为检测目标显式设计，仍然是次优的

BLADES的关键洞察是：传统自监督学习流程不会显式地将前置损失与下游任务对齐，导致表征不够专业化。双层优化提供了一种有原则的、数学上有依据的方式来端到端地引导自监督预训练朝向下游目标。

## 方法详解

### 整体框架

系统分为两个阶段：(1) 双层优化驱动的自监督预训练；(2) 冻结编码器后的检测推理。预训练阶段采用联合图像-文本嵌入架构（CLIP风格，ResNet-50视觉编码器 + GPT-2文本编码器），在纯真实人脸照片上训练。

### 关键设计

1. **双层优化问题建模**:

    - 内层（Inner Loop）：在训练集 $\mathcal{B}_{tr}$ 上，用加权组合的前置任务损失优化编码器参数 $\boldsymbol{\theta}$
    $\boldsymbol{\theta}^{\star} = \arg\min_{\boldsymbol{\theta}} \sum_{\boldsymbol{x} \in \mathcal{B}_{tr}} \sum_{i=1}^{K} \lambda_i \ell_i(\boldsymbol{x}; \boldsymbol{\theta})$
    - 外层（Outer Loop）：在验证集 $\mathcal{B}_{val}$ 上，优化任务权重 $\boldsymbol{\lambda}$ 以最小化代理检测任务损失
    $\min_{\boldsymbol{\lambda}} \sum_{\boldsymbol{x} \in \mathcal{B}_{val}} \ell_1(\boldsymbol{x}; \boldsymbol{\theta}^{\star})$
    - 交替更新 $\boldsymbol{\theta}$ 和 $\boldsymbol{\lambda}$，自动优先选择对下游检测最有益的前置任务

2. **四类前置任务设计**:

    - **粗粒度人脸篡改检测（代理主任务 $\ell_1$）**：通过局部翻转和全局仿射变换生成篡改人脸，用fidelity损失训练编码器区分"manipulated"/"photographic"
    - **分类型EXIF标签分类（$\ell_i$）**：预测白平衡模式、闪光灯等分类EXIF标签，用focal fidelity损失处理长尾分布
    - **有序型EXIF标签排序（$\ell_j$）**：将ISO、光圈等数值EXIF标签离散化为low/medium/high三级，通过Thurstone模型进行成对排序
    - **细粒度人脸篡改检测（$\ell_m$）**：识别被篡改的具体人脸区域（眼、嘴、鼻），用sigmoid输出区域级篡改概率

3. **检测推理策略**:

    - **单类异常检测（BLADES-OC）**：对真实人脸特征拟合10组分GMM，测试时低于第5百分位似然阈值则判为AI生成
    - **二分类检测（BLADES-BC）**：训练轻量两层感知机（768→1536→2），用低似然真实人脸作为伪离群样本增强AI生成人脸样本

### 损失函数 / 训练策略

- 所有任务统一使用基于fidelity的损失函数 $\ell = 1 - \sum \sqrt{p \cdot \hat{p}}$（Bhattacharyya系数）
- 分类EXIF任务引入focal版本处理类别不平衡，$\gamma=2$
- 内层学习率 $\alpha=10^{-5}$（AdamW, cosine退火），外层学习率 $\beta=3 \times 10^{-4}$（Adam）
- 预训练20 epochs，batch=48，输入尺寸 $224 \times 224$
- 训练数据：FDF数据集中130K张有EXIF元数据的人脸照片

## 实验关键数据

### 主实验（跨生成器检测准确率 %）

| 方法 | StyleGAN2 | VQGAN | LDM | DDIM | SDv2.1 | Midjourney | SDXL | 平均 |
|------|-----------|-------|-----|------|--------|------------|------|------|
| CNND | 50.61 | 99.89 | 53.07 | 56.55 | 50.51 | 51.66 | 54.49 | 58.41 |
| FatFormer | 98.91 | 98.30 | 97.82 | 95.63 | 68.88 | 88.20 | 88.08 | 89.68 |
| Zou25 | 76.88 | 74.59 | 93.83 | 93.63 | 78.62 | 91.29 | 91.71 | 86.63 |
| BLADES-OC | 76.75 | 76.78 | 93.63 | 96.05 | 80.70 | 92.79 | 94.48 | 88.01 |
| **BLADES-BC** | **94.22** | **97.24** | **96.95** | **94.33** | **74.83** | **95.19** | **93.84** | **91.86** |

### 消融实验（特征可分性 AUC %）

| 方法 | StyleGAN2 | LDM | SDv2.1 | FreeDoM | SDXL | 平均 |
|------|-----------|-----|--------|---------|------|------|
| CLIP | 33.99 | 55.49 | 90.15 | 85.39 | 93.66 | 76.13 |
| FaRL | 34.35 | 47.26 | 95.24 | 79.91 | 94.61 | 75.12 |
| EAL | 69.71 | 85.81 | 74.21 | 97.92 | 93.04 | 84.12 |
| Zou25 | 85.69 | 98.66 | 88.29 | 99.79 | 97.23 | 93.43 |
| **BLADES-OC** | **87.89** | **98.24** | **91.72** | **99.92** | **98.36** | **95.05** |

### 敏感性分析

| 方法 | TNR↑ | FPR↓ | TPR↑ | FNR↓ | F-score↑ |
|------|------|------|------|------|----------|
| LGrad | 99.98 | 0.02 | 44.97 | 55.03 | 0.60 |
| FatFormer | 97.61 | 2.38 | 81.74 | 18.26 | 0.87 |
| **BLADES-BC** | **94.64** | **5.36** | **88.97** | **11.03** | **0.91** |

### 关键发现

- BLADES-BC以91.86%平均准确率大幅超越所有竞争方法
- BLADES-OC仅用真实人脸训练就超过大多数监督方法，验证双层优化对齐策略的有效性
- 先前方法普遍高特异性（TNR>97%）但低敏感性（TPR<82%），BLADES-BC实现了两者平衡
- 基于扩散模型的检测方法（如DIRE、AEROBLADE）在同一生成家族内泛化能力也差，说明依赖模型特定线索本质上脆弱

## 亮点与洞察

- 双层优化思想的应用非常巧妙：外层优化通过代理任务（篡改检测）间接优化真正目标（AI生成检测），不需要任何合成人脸训练数据
- EXIF元数据作为自监督信号的利用方式新颖——分类型标签做分类、有序型标签做排序，充分挖掘了元数据的结构
- 联合嵌入架构利用文本模板将各种前置任务统一到同一框架中

## 局限与展望

- 依赖EXIF元数据的人脸照片进行预训练，社交媒体上大量图片的EXIF信息被剥离
- 单类检测设定下对GAN生成的图像检测能力相对较弱（StyleGAN2仅76.75%）
- 二分类设定仍需少量合成数据用于微调分类头

## 相关工作与启发

- 双层优化用于任务权重学习的思路可推广到其他多任务自监督场景
- EXIF元数据信号或可用于更广泛的图像溯源和完整性验证任务
- Fidelity损失（基于Bhattacharyya系数）作为分类损失的替代方案值得关注

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 双层优化对齐自监督与检测目标的idea非常新颖，EXIF任务设计全面
- 实验充分度: ⭐⭐⭐⭐⭐ 9种生成器、跨数据集、特征可分性、敏感性分析，非常全面
- 写作质量: ⭐⭐⭐⭐ 问题建模清晰，公式推导严谨
- 价值: ⭐⭐⭐⭐ 解决AI生成检测泛化难题，在深度伪造治理方面有重要应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] VideoClusterNet: Self-Supervised and Adaptive Face Clustering for Videos](../../ECCV2024/human_understanding/videoclusternet_self-supervised_and_adaptive_face_clustering_for_videos.md)
- [\[ICCV 2025\] LVFace: Progressive Cluster Optimization for Large Vision Models in Face Recognition](lvface_progressive_cluster_optimization_for_large_vision_models_in_face_recognit.md)
- [\[CVPR 2025\] FSFM: A Generalizable Face Security Foundation Model via Self-Supervised Facial Representation Learning](../../CVPR2025/human_understanding/fsfm_a_generalizable_face_security_foundation_model_via_self-supervised_facial_r.md)
- [\[ECCV 2024\] Pose-Aware Self-Supervised Learning with Viewpoint Trajectory Regularization](../../ECCV2024/human_understanding/pose-aware_self-supervised_learning_with_viewpoint_trajectory_regularization.md)
- [\[ICCV 2025\] Weakly Supervised Visible-Infrared Person Re-Identification via Heterogeneous Expert Collaborative Consistency Learning](weakly_supervised_visible-infrared_person_re-identification_via_heterogeneous_ex.md)

</div>

<!-- RELATED:END -->
