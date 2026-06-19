---
title: >-
  [论文解读] Surf2CT: Cascaded 3D Flow Matching Models for Torso 3D CT Synthesis from Skin Surface
description: >-
  [NeurIPS 2025][医学图像][Flow Matching] 提出 Surf2CT，一种级联式 3D Flow Matching 框架，首次实现仅从外部体表扫描和人口学数据（年龄、性别、身高、体重）合成完整的高分辨率 3D CT 体积，无需任何内部成像输入。 CT 扫描能提供详尽的体内解剖信息，但成本高、辐射暴露大…
tags:
  - "NeurIPS 2025"
  - "医学图像"
  - "Flow Matching"
  - "CT 合成"
  - "体表扫描"
  - "3D 生成"
  - "非侵入式成像"
---

# Surf2CT: Cascaded 3D Flow Matching Models for Torso 3D CT Synthesis from Skin Surface

**会议**: NeurIPS 2025  
**arXiv**: [2505.22511](https://arxiv.org/abs/2505.22511)  
**代码**: 暂无  
**领域**: 医学图像 / CT 合成  
**关键词**: Flow Matching, CT 合成, 体表扫描, 3D 生成, 非侵入式成像

## 一句话总结

提出 Surf2CT，一种级联式 3D Flow Matching 框架，首次实现仅从外部体表扫描和人口学数据（年龄、性别、身高、体重）合成完整的高分辨率 3D CT 体积，无需任何内部成像输入。

## 研究背景与动机

CT 扫描能提供详尽的体内解剖信息，但成本高、辐射暴露大，不适合常规筛查。一个理想的替代方案是从非侵入式的外部体形数据推断内部解剖结构——这将使家庭健康监测、预防医学和个性化临床评估成为可能。然而，外部体形并不能唯一确定内部解剖：相似的体表测量可对应不同的内脏器官大小和组成。

现有方法的局限：
- **数字人体模型**（如 XCAT）：依赖有限的模板库，仅支持粗粒度的人口学缩放，无法生成体素级 CT
- **稀疏 X 光重建**（如 X2CT-GAN、DIFR3CT）：仍需至少部分内部成像输入
- **分割引导 CT 合成**（如 Seg2Med、MAISI）：需要预先存在的内部分割图，通常来自已有 CT/MRI

Surf2CT 的核心创新在于完全消除对内部成像的依赖，仅用体表 3D 扫描（甚至是消费级深度相机可采集的数据）和基本人口学信息即可生成解剖学上合理的 CT 体积。

## 方法详解

### 整体框架

Surf2CT 采用三阶段级联生成流程，每个阶段使用独立的条件 3D Flow Matching 模型：

1. **Stage 1: 体表补全** — 从不完整体表扫描恢复完整的 SDF（签名距离函数）
2. **Stage 2: 粗分辨率 CT 合成** — 从完整 SDF + 人口学数据生成 8mm 分辨率 CT
3. **Stage 3: CT 超分辨率** — 逐块细化为 2mm 高分辨率 CT

所有阶段的骨干网络基于 3D 改编的 EDM2 U-Net 架构。

### 关键设计

1. **SDF 体表补全（Stage 1）**：将体形表示为签名距离函数（SDF），定义 $f(\mathbf{x}) = \pm\min_{\mathbf{y} \in \partial\Omega} \|\mathbf{x} - \mathbf{y}\|_2$，在 $56 \times 56 \times 88$ 网格（8mm 等距）上离散化。训练条件 Flow Matching 模型 $G_1$，学习从部分 SDF（如仅保留正面视角）恢复完整 SDF 的速度场：

$$\frac{d\mathbf{x}(t)}{dt} = v_\theta^{(1)}(t, \mathbf{x}(t), f_{\text{partial}}, \mathbf{z}_{\text{demo}})$$

人口学属性（年龄、性别、身高、体重）作为同空间维度的恒定通道与部分 SDF 拼接输入。训练目标为标准 Flow Matching 损失 $\mathcal{L}_1 = \mathbb{E}[\|v_\theta^{(1)}(t, \mathbf{x}_t) - (f_{\text{full}}^{\text{gt}} - \boldsymbol{\eta})\|^2]$。

2. **粗分辨率 CT 合成（Stage 2）**：在相同 $56 \times 56 \times 88$ 网格上（对应 $448 \times 448 \times 704$ mm 视场），学习从完整 SDF 到低分辨率 CT 的映射。条件包括完整 SDF 和人口学数据。模型隐式学习外部形态特征与内部器官位置、尺寸、组织密度之间的关联——如高 BMI 个体通常有更多脂肪组织。

3. **逐块超分辨率（Stage 3）**：目标分辨率为 $224 \times 224 \times 352$（2mm 等距），直接生成全分辨率体积计算量过大。采用逐块策略：随机采样 $56 \times 56 \times 88$ 高分辨率块，条件为上采样后的粗 CT 对应区域 + 正弦位置编码 + 人口学信息。推理时逐块生成并融合。Stage 3 模型仅 1.89M 参数（vs Stage 1/2 的 80.68M），通过减少通道数和深度控制计算。

### 损失函数 / 训练策略

- 三个阶段均使用标准 Flow Matching $L_2$ 回归损失 + 条件最优传输概率路径
- 训练数据：MGH 2633 例 + AutoPET 565 例 = 3198 例躯干 CT（约 113 万张轴向切片）
- 预处理：TotalSegmentator 提取躯干区域，2mm 等距重采样，HU 值裁剪到 [-500, 500]
- Stage 1/2 各训练约 3500 万步（4×A100，72h/stage），Stage 3 训练 1500 万步
- AdamW 优化器，初始 lr=$10^{-4}$，线性衰减；梯度裁剪 + EMA
- 采样：200 步积分，Dormand-Prince 求解器

## 实验关键数据

### 主实验：体成分评估

| 指标 | 原始 CT | Surf2CT | 差异(%) | R² |
|------|---------|---------|---------|-----|
| 男性肌肉量 (mL) | 9388±1860 | 8483±1424 | -9.6% | 0.81 |
| 男性皮下脂肪 | 8288±4189 | 9357±3992 | +12.9% | 0.86 |
| 男性内脏脂肪 | 5030±2472 | 4901±2406 | -2.5% | 0.74 |
| 女性肌肉量 | 5304±1152 | 4817±993 | -9.2% | 0.87 |
| 女性皮下脂肪 | 9210±5368 | 9866±4999 | +7.1% | **0.96** |
| 女性内脏脂肪 | 2324±1514 | 2213±1476 | -4.8% | 0.79 |

### 器官体积评估

| 器官 | 原始 CT (mL) | Surf2CT (mL) | 差异(%) | R² |
|------|------------|-------------|---------|-----|
| 男性心脏 | 689.7±133.2 | 720.0±84.6 | +4.4% | 0.12 |
| 男性肝脏 | 1695.4±380.8 | 1761.1±268.7 | +3.9% | 0.25 |
| 男性肾脏 | 337.4±70.1 | 329.1±56.9 | -2.5% | 0.16 |
| 女性心脏 | 543.9±82.9 | 557.0±92.0 | +2.4% | 0.11 |
| 女性肝脏 | 1524.9±346.7 | 1495.7±239.0 | -1.9% | 0.33 |
| 女性肾脏 | 293.1±55.3 | 260.5±58.9 | **-11.1%** | 0.04 |

### 体表补全评估

| 指标 | 部分 SDF | 补全后 |
|------|---------|--------|
| Chamfer Distance (mm) | 521.78±228.09 | **2.71±1.80** |
| IoU | 0.87±0.09 | **0.98±0.02** |
| NMAE | 0.14±0.07 | **0.02±0.01** |

### 消融/其他评估

| 评估项 | 结果 | 说明 |
|--------|------|------|
| 肺部定位偏差 | -2.5 mm | Bland-Altman 分析，一致性范围 [-62.6, +57.5] mm |
| 肺定位 R² | 0.36 | 中等相关性 |
| 性别差异建模 | 正确反映 | 男女器官大小、肌肉/脂肪分布差异与真实 CT 一致 |

### 关键发现

- 体表补全极为有效：Chamfer Distance 从 521.8mm 降至 2.7mm
- 体成分指标强相关（R² 达 0.67-0.96），临床可用
- 器官体积平均误差在 ±5% 以内（多数），但个体预测准确度较低（R² 普遍 < 0.35）
- 女性受试者结果整体优于男性（可能因训练数据中女性比例更低导致更大不确定性）
- 模型能正确学习人口学与解剖的关联（如 BMI 与脂肪分布）

## 亮点与洞察

- **范式创新**：首次实现从纯外部数据（体表+人口学）到内部 3D CT 的生成，开辟了非侵入式解剖成像的新方向
- **级联设计的实用性**：三阶段从粗到精的级联策略优雅地解决了分辨率和计算量的矛盾，每个阶段有明确的物理意义
- **临床场景想象空间大**：家庭 3D 扫描 → 虚拟 CT → 健康监测，以及手术规划的无辐射数字孪生
- **潜在异常检测**：合成 CT 与实际 CT 的偏差可能反映隐藏的病理异常

## 局限与展望

- 训练数据以男性癌症患者为主，性别和病种偏差严重
- 个体器官体积预测的 R² 较低（特别是肺和肾），说明外部体形对内部器官的约束力有限
- 未处理消费级扫描器的噪声和伪影
- 体表到 CT 的映射本质上是不适定的（一对多），生成的 CT 可能不反映真实个体的病理状态
- 未评估在实际医疗决策中的可靠性和公平性

## 相关工作与启发

- **XCAT**: 传统数字人体模型，模板化方法
- **EDM2**: 扩散模型架构，本文改编为 3D Flow Matching
- **TotalSegmentator**: 自动器官分割，用于评估和预处理
- **BOSS**: 统计形体模型，联合学习皮肤/骨骼/器官几何
- 启发：Flow Matching 在 3D 医学图像生成中的潜力巨大，级联策略可推广到其他大尺度 3D 生成任务

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首创性任务定义，从纯外部数据合成内部 CT
- **实验充分度**: ⭐⭐⭐⭐ 多维度评估全面，但训练数据偏差明显
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，三阶段描述条理分明
- **价值**: ⭐⭐⭐⭐ 开辟新范式，但距临床应用仍有距离

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Riemannian Flow Matching for Brain Connectivity Matrices via Pullback Geometry](riemannian_flow_matching_for_brain_connectivity_matrices_via_pullback_geometry.md)
- [\[NeurIPS 2025\] 3D-RAD: A Comprehensive 3D Radiology Med-VQA Dataset with Multi-Temporal Analysis and Diverse Diagnostic Tasks](3drad_a_comprehensive_3d_radiology_medvqa_dataset_with_multi.md)
- [\[ICML 2026\] Foundation VAEs for 3D CT Reconstruction, Augmentation, and Generation](../../ICML2026/medical_imaging/foundation_vaes_for_3d_ct_reconstruction_augmentation_and_generation.md)
- [\[CVPR 2026\] MicroFM: Physics-guided Flow Matching for Isotropic Microscopy Reconstruction](../../CVPR2026/medical_imaging/microfm_physics-guided_flow_matching_for_isotropic_microscopy_reconstruction.md)
- [\[NeurIPS 2025\] Semantic and Visual Crop-Guided Diffusion Models for Heterogeneous Tissue Synthesis in Histopathology](semantic_and_visual_crop-guided_diffusion_models_for_heterogeneous_tissue_synthe.md)

</div>

<!-- RELATED:END -->
