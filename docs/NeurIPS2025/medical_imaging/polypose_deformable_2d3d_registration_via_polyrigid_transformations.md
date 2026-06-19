---
title: >-
  [论文解读] PolyPose: Deformable 2D/3D Registration via Polyrigid Transformations
description: >-
  [NeurIPS 2025][医学图像][2D/3D配准] 提出PolyPose，一种基于多刚体变换（polyrigid）的可变形2D/3D配准方法，利用"骨骼是刚体"这一解剖学先验，将复杂3D形变场参数化为多个刚体变换在切空间 $\mathfrak{se}(3)$ 中的加权组合，无需正则化和超参数调优即可从少至两张X光片实现精确的3D体积配准。
tags:
  - "NeurIPS 2025"
  - "医学图像"
  - "2D/3D配准"
  - "多刚体变换"
  - "可微分X射线渲染"
  - "术中导航"
  - "稀疏视角"
---

# PolyPose: Deformable 2D/3D Registration via Polyrigid Transformations

**会议**: NeurIPS 2025  
**arXiv**: [2505.19256](https://arxiv.org/abs/2505.19256)  
**代码**: [项目主页](https://polypose.csail.mit.edu)  
**领域**: 医学图像 / 医学配准  
**关键词**: 2D/3D配准, 多刚体变换, 可微分X射线渲染, 术中导航, 稀疏视角

## 一句话总结

提出PolyPose，一种基于多刚体变换（polyrigid）的可变形2D/3D配准方法，利用"骨骼是刚体"这一解剖学先验，将复杂3D形变场参数化为多个刚体变换在切空间 $\mathfrak{se}(3)$ 中的加权组合，无需正则化和超参数调优即可从少至两张X光片实现精确的3D体积配准。

## 研究背景与动机

术中从2D X光图像估计患者3D姿态是图像引导手术和放射治疗中的关键任务。获取的X光数量与辐射暴露直接相关，因此临床实际中可用的X光片极少（稀疏视角），且扫描器几何限制了角度范围（有限角度）。虽然患者通常有术前CT，但患者在两次采集之间会移动，导致术前CT与术中X光不对齐。

现有方法的不足：
- **刚性配准**：仅估计全局SE(3)变换，无法处理关节运动和软组织变形
- **密集形变场**：采用逐体素位移场，在极度欠定的2D→3D场景下需要大量针对每个患者和手术的正则化超参数调优，且容易产生解剖学上不合理的变形
- **基于学习的方法**：需要每位患者多套纵向CT数据进行训练，这在多数临床场景中不可行

PolyPose的核心洞察：人体骨骼是刚体，不会弯曲。利用这一通用解剖学先验可以将优化参数从体素量级 $\mathcal{O}(M)$（$M \approx 10^7$）降低到刚体数量级 $\mathcal{O}(K)$（$K=3$即可），从根本上缓解欠定问题。

## 方法详解

### 整体框架

PolyPose分为两步：(1) 通过对锚定结构的刚性配准估计每幅X光的相机矩阵；(2) 利用估计的相机矩阵联合优化所有刚体的姿态，通过可微分渲染构建多刚体变形场。

给定3D CT体积 $\mathbf{V}$ 和一组2D X光图像 $\mathbf{I} = \{\mathbf{I}_n\}_{n=1}^N$，建模关系为：$\mathbf{I}_n = \mathcal{P}(\mathbf{\Pi}_n) \circ \mathbf{V} \circ \mathbf{\Phi}$，其中 $\mathcal{P}$ 是X光投影算子，$\mathbf{\Phi}$ 是3D形变场。

### 关键设计

1. **多刚体形变场参数化**：设 $\{\mathbf{S}_1, \ldots, \mathbf{S}_K\}$ 为体积中关节刚体的二值掩码，每个结构关联一个SE(3)变换 $\mathbf{T}_k$。任意点 $\mathbf{x}$ 处的形变通过切空间中的加权组合计算：

$$\mathbf{\Phi}[\mathbf{T}_1, \ldots, \mathbf{T}_K](\mathbf{x}) = \overline{\mathbf{T}}(\mathbf{x})\tilde{\mathbf{x}}, \quad \overline{\mathbf{T}}(\mathbf{x}) \triangleq \exp\left(\frac{\sum_{k=1}^K w_k(\mathbf{x}) \log \mathbf{T}_k}{\sum_{k=1}^K w_k(\mathbf{x})}\right)$$

在切空间 $\mathfrak{se}(3)$ 中线性组合log变换再映射回SE(3)，保证了生成的变形场天然具有光滑性、可逆性和坐标系不变性。这是与简单平均位移的本质区别。

2. **无超参数的权重函数**：先前方法使用 $w_k(\mathbf{x}) = \frac{1}{1 + \epsilon d_k^2(\mathbf{x})}$（包含需调优的 $\epsilon$），且不同刚体的最优 $\epsilon$ 差异巨大（如左股骨 $\epsilon = 10^0$ vs. 右股骨 $\epsilon = 10^{-3}$）。PolyPose提出受引力启发的权重函数：

$$w_k(\mathbf{x}) = \frac{m_k}{1 + d_k^2(\mathbf{x})}$$

其中 $m_k$ 是结构 $\mathbf{S}_k$ 相对于所有结构的归一化质量（由体积估计）。完全消除了超参数，同时通过质量加权处理不同大小的结构。

3. **可微分X光渲染**：基于Beer-Lambert定律，X光像素强度为沿射线的线性衰减系数积分：

$$\mathbf{I}_n(\mathbf{p}) = \|\mathbf{P} - \mathbf{S}\| \int_0^1 \mathbf{V}(\mathbf{S} + \lambda(\mathbf{P} - \mathbf{S})) d\lambda$$

通过插值求积离散化实现可微分渲染。向量化前向模型：预计算权重矩阵 $\mathbf{W} \in \mathbb{R}^{M \times K}$，所有变换通过 $\hat{\mathbf{\Phi}}(\mathbf{X}) = \exp(\mathbf{W}\hat{\bm{\mathfrak{T}}})\tilde{\mathbf{X}}$ 的批量矩阵乘法高效计算。

### 损失函数 / 训练策略

联合优化目标为最大化渲染X光与真实X光之间的图像相似度：

$$(\hat{\mathbf{T}}_1, \ldots, \hat{\mathbf{T}}_K) = \arg\max_{\mathbf{T}_1, \ldots, \mathbf{T}_K} \frac{1}{N} \sum_{n=1}^N \mathcal{L}(\mathbf{I}_n, \mathcal{P}(\hat{\mathbf{\Pi}}_n) \circ \mathbf{V} \circ \mathbf{\Phi})$$

使用多尺度patch-wise归一化互相关损失，计算原始图像和Sobel滤波图像的相似度。Adam优化器，旋转分量步长 $\beta_{\text{rot}} = 10^{-2}$，平移分量步长 $\beta_{\text{xyz}} = 10^0$。全程无需任何正则化项。

## 实验关键数据

### 主实验（DeepFluoro有限角度配准，仅2张X光，约30°间距）

| 方法 | 骨盆 Dice ↑ | 左股骨 Dice ↑ | 右股骨 Dice ↑ | %Folds ↓ |
|------|-----------|-------------|-------------|---------|
| **PolyPose** | **0.99** | **0.98** | **0.98** | **0.00%** |
| Dense $\mathbb{R}^3$ | 0.98 | 0.97 | 0.96 | 0.44% |
| xvr (刚性) | 0.99 | 0.96 | 0.94 | 0.00% |
| FireANTs | 0.99 | 0.96 | 0.93 | 0.00% |
| anatomix | 0.95 | 0.93 | 0.92 | 3.01% |
| multiGradICON | 0.83 | 0.86 | 0.77 | 0.00% |

### 消融实验（形变参数化和权重函数对比）

| 配置 | 左股骨 Dice | 右股骨 Dice | %Folds | 说明 |
|------|-----------|-----------|--------|------|
| PolyPose (Eq.6) | 0.98 | 0.98 | 0.00% | 无超参数权重函数(最优) |
| Eq.5, $\epsilon=10^0$ | 0.93 | 0.96 | 0.03% | 右股骨好但左股骨差 |
| Eq.5, $\epsilon=10^{-3}$ | 0.95 | 0.95 | 0.00% | 左股骨好但右股骨差 |
| Dense SE(3) | 0.90 | 0.88 | 44.08% | 密集场严重拓扑缺陷 |

### 关键发现

- 仅从2张约30°间距的X光片，PolyPose即可恢复最准确的3D形变场，零拓扑缺陷
- 在Head&Neck数据集上，PolyPose不仅在刚性结构上最优，还能外推到未直接优化的软组织器官（甲状腺、脊髓、大脑等）
- 密集形变模型虽然在训练视角上图像相似度接近完美（NCC≈0.99），但无法泛化到未见视角
- PolyPose对分割标签腐蚀具有强鲁棒性，即使在3mm侵蚀（40-60%体积减小）下仍优于基线

## 亮点与洞察

- **极其简洁的归纳偏置**：整个方法的核心思想"骨骼不弯曲"非常直觉，却产生了极强的效果。将 $10^7$ 量级的优化参数降至个位数，是"less is more"的完美范例
- **无超参数设计**：受引力启发的权重函数完全消除了超参数搜索，实现了开箱即用的跨手术/解剖区域泛化
- **理论保证**：多刚体变形场天然是微分同胚的，这不是通过正则化强制的，而是参数化本身的性质

## 局限与展望

- 对远离骨骼的软组织极端变形（如腹部）的建模能力有待验证
- 微分同胚约束无法表示某些变形（如张嘴），可通过引入运动链缓解
- 需要CT中刚体结构的分割掩码，虽然对标签质量鲁棒，但增加了流程前置步骤

## 相关工作与启发

- 多刚体变换框架源自Arsigny等人的开创性工作，PolyPose将其成功推广到极度欠定的2D/3D配准设置
- 可微分渲染的使用连接了计算机视觉（NeRF等）和医学影像配准两个领域
- 对骨科手术导航和放射治疗定位等临床应用有直接价值

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 多刚体参数化+无超参数权重函数的组合极具创新性
- 实验充分度: ⭐⭐⭐⭐ 两个不同临床场景（头颈放疗+骨科手术），定性定量均充分，鲁棒性分析全面
- 写作质量: ⭐⭐⭐⭐⭐ 数学推导严谨清晰，可视化出色
- 价值: ⭐⭐⭐⭐⭐ 解决了临床实际痛点，方法简洁通用，有很强的临床转化价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Dynamic Stream Network for Combinatorial Explosion Problem in Deformable Medical Image Registration](../../CVPR2026/medical_imaging/dynamic_stream_network_for_combinatorial_explosion_problem_in_deformable_medical.md)
- [\[NeurIPS 2025\] Surf2CT: Cascaded 3D Flow Matching Models for Torso 3D CT Synthesis from Skin Surface](surf2ct_cascaded_3d_flow_matching_models_for_torso_3d_ct_synthesis_from_skin_sur.md)
- [\[ICML 2025\] Raptor: Scalable Train-Free Embeddings for 3D Medical Volumes Leveraging Pretrained 2D Foundation Models](../../ICML2025/medical_imaging/raptor_scalable_train-free_embeddings_for_3d_medical_volumes_leveraging_pretrain.md)
- [\[CVPR 2026\] Revisiting 2D Foundation Models for Scalable 3D Medical Image Classification](../../CVPR2026/medical_imaging/revisiting_2d_foundation_models_for_scalable_3d_medical_image_classification.md)
- [\[NeurIPS 2025\] 3D-RAD: A Comprehensive 3D Radiology Med-VQA Dataset with Multi-Temporal Analysis and Diverse Diagnostic Tasks](3drad_a_comprehensive_3d_radiology_medvqa_dataset_with_multi.md)

</div>

<!-- RELATED:END -->
