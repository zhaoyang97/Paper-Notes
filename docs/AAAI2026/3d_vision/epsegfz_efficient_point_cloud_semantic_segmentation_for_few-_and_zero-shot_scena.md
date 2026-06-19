---
title: >-
  [论文解读] EPSegFZ: Efficient Point Cloud Semantic Segmentation for Few- and Zero-Shot Scenarios
description: >-
  [AAAI 2026][3D视觉][点云语义分割] 提出 EPSegFZ，一个无需预训练的3D点云少样本/零样本语义分割框架，通过 ProERA 提取高频特征、LGPE 融合文本信息更新原型、DRPE 建立精确的查询-原型对应关系，在 S3DIS 和 ScanNet 上分别超越 SOTA 5.68% 和 3.82%。
tags:
  - "AAAI 2026"
  - "3D视觉"
  - "点云语义分割"
  - "少样本学习"
  - "零样本学习"
  - "语言引导"
  - "注意力机制"
---

# EPSegFZ: Efficient Point Cloud Semantic Segmentation for Few- and Zero-Shot Scenarios

**会议**: AAAI 2026  
**arXiv**: [2511.11700](https://arxiv.org/abs/2511.11700)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 点云语义分割, 少样本学习, 零样本学习, 语言引导, 注意力机制

## 一句话总结

提出 EPSegFZ，一个无需预训练的3D点云少样本/零样本语义分割框架，通过 ProERA 提取高频特征、LGPE 融合文本信息更新原型、DRPE 建立精确的查询-原型对应关系，在 S3DIS 和 ScanNet 上分别超越 SOTA 5.68% 和 3.82%。

## 研究背景与动机

3D点云少样本语义分割（FS-SemSeg）面临三个核心挑战：

**过度依赖预训练**：现有方法（AttMPTI、COSeg等）严重依赖全监督预训练backbone，带来域差异偏差，且3D数据集小，容易过拟合。预训练本身也消耗大量资源

**高频信息丢失**：Seg-PN 等无预训练方法为保持鲁棒性丢弃了高频信息，但高频特征包含关键的边缘细节，对精确分割必不可少

**支持集信息利用不充分**：现有方法仅依赖点云标签，忽略了文本标注等互补信息，限制了性能和零样本能力

作者的核心思路是：同时利用高频视觉特征和低频文本特征，在不预训练的前提下实现高效的少样本/零样本分割。

## 方法详解

### 整体框架

EPSegFZ 包含三个核心模块，构成一个不需要预训练的端到端框架：

1. **ProERA（Prototype-Enhanced Registers Attention）**：增强特征提取，捕获高频信息
2. **LGPE（Language-Guided Prototype Embedding）**：利用文本信息更新原型，支持零样本推理
3. **DRPE（Dual Relative Positional Encoding）-based cross-attention**：建立精确的查询-原型对应关系

工作流程：DGCNN（从头训练）提取点云特征 → MPS 采样多原型 → ProERA 细化特征（附加寄存器和原型token） → LGPE 更新原型（加入文本嵌入） → DRPE交叉注意力建立对应 → 点积计算预测结果。

### 关键设计

#### ProERA 模块

核心创新在于通过减去低频分量来逐步聚焦高频信息：

- 在输入token序列中附加可学习的 register token $\mathbf{r}_t \in \mathbb{R}^{n_r \times D}$ 和prototype token
- 自注意力本质是低通滤波器，因此在注意力输出后减去输入特征的均值（即低频分量），得到高频主导的特征：

$$\tilde{\mathbf{X}}_j^i = \text{Res}(\text{SA}([\hat{\mathbf{X}}_j^{i-1}; \mathbf{r}_j; \hat{\mathbf{p}}^{i-1}])) - \frac{1}{n_j}\sum_{n_j}\hat{\mathbf{X}}_j^{i-1}$$

- Register token 学会关注不同区域：一个关注背景/无物体区域，另一个关注包含多个目标的区域，隐式缓解前景-背景不平衡

#### LGPE 模块

解决原型质量问题，特别是在训练早期backbone随机初始化导致原型无区分性的问题：

- 使用预训练 CLIP 文本编码器获取类别文本嵌入 $\mathbf{T}^c$，通过投影网络映射到统一空间
- 原型更新融合四个来源：前一层原型token $\tilde{\mathbf{p}}^i$、原始原型 $\mathbf{p}_{raw}$、动态原型 $\mathbf{p}_{dyn}^i$、文本原型 $\mathbf{p}_{text}$

$$\mathbf{p}^i = \lambda_1 \tilde{\mathbf{p}}^i + \lambda_2 \mathbf{p}_{raw} + \lambda_3 \mathbf{p}_{dyn}^i + \lambda_4 \mathbf{p}_{text}$$

- **动态权重调度**：文本权重 $\lambda_4(t) = \lambda_4^* e^{-0.5t}$（指数衰减），视觉权重 $\lambda_i(t) = \lambda_i^*(1 - e^{-0.5t})$（逐渐增长），实现从文本驱动到视觉-文本平衡的平滑过渡
- 零样本能力：仅使用文本嵌入即可构建原型，无需支持集点云

#### DRPE 模块

首次将查询-原型在隐空间中的相关性作为位置编码信号引入交叉注意力：

- 计算查询点与原型之间的**欧氏距离** $d_E^{i,j,c}$，通过正弦位置编码函数得到 $\mathbf{R}_E^i$
- 计算查询向量与原型向量之间的**余弦相似度** $d_C^{i,j,c}$，同样编码得到 $\mathbf{R}_C^i$
- 双重编码相加：$\mathbf{R}^i = \mathbf{R}_C^i + \mathbf{R}_E^i$
- 优势：不引入额外训练参数，高效捕获查询-原型相关性作为先验知识

### 损失函数 / 训练策略

三个损失函数联合优化：

1. **分割损失** $\mathcal{L}_{seg} = \text{CE}(\mathbf{Y}_q, \hat{\mathbf{Y}}_q)$：主监督信号
2. **前景一致性损失** $\mathcal{L}_{con} = \text{InfoNCE}(\mathbf{x}_q, \mathbf{x}_s)$：鼓励同类前景特征在嵌入空间中聚集，弥补无预训练backbone的表征不足
3. **前景感知对齐损失** $\mathcal{L}_{align}$：最小化文本-视觉相似度与文本标签之间的交叉熵，增强文本-视觉联合空间

$$\mathcal{L} = \mathcal{L}_{seg} + \lambda_{con}\mathcal{L}_{con} + \lambda_{align}\mathcal{L}_{align}$$

训练策略：episodic learning，30000次迭代，backbone较大学习率但快速衰减。

## 实验关键数据

### 主实验

**S3DIS 数据集（2-way 1-shot）**：

| 方法 | S⁰ | S¹ | 均值 | Δ |
|------|-----|-----|------|-----|
| AttMPTI | 53.77 | 55.94 | 54.86 | -18.56 |
| PAPFZS3D | 59.45 | 66.08 | 62.76 | -10.66 |
| Seg-PN | 64.84 | 67.98 | 66.41 | -7.01 |
| SDSimPoint | 68.73 | 70.61 | 69.67 | -3.75 |
| **EPSegFZ** | **73.08** | **73.75** | **73.42** | **-** |

**ScanNet 数据集（2-way 1-shot）**：

| 方法 | 均值 | Δ |
|------|------|-----|
| Seg-PN | 63.74 | -5.10 |
| SDSimPoint | 65.19 | -3.65 |
| **EPSegFZ** | **68.84** | **-** |

效率分析：EPSegFZ 仅 2.02M 参数、2.11 GFLOPs、0.36s 推理时间，与轻量级 Seg-PN（241K/1.95/0.32）相当，远优于 COSeg（7.69M/9.71/1.35）。

### 消融实验

| 配置 | mIoU | Δ |
|------|------|-----|
| 无任何模块 | 31.55 | -41.53 |
| 仅 ProERA | 64.84 | -8.24 |
| ProERA + LGPE | 70.48 | -2.60 |
| ProERA + DRPE | 70.17 | -2.91 |
| 完整模型 | **73.08** | - |

原型消融中，动态原型 $\mathbf{p}_{dyn}$ 贡献最大。去除 $\mathcal{L}_{con}$ 或 $\mathcal{L}_{align}$ 各降低约4-5%。DRPE 优于可学习位置编码和正弦编码。

### 关键发现

- 零样本评估（S3DIS，CLIP，2-way 1-shot）：EPSegFZ 达 63.84%，超过 PAPFZS3D 的 61.09%
- t-SNE 可视化显示 EPSegFZ 的同类特征分布更紧凑，类间分离更清晰
- Register 数量 N+1（N为类别数）效果最佳；3个decoder block是性能-效率最佳权衡

## 亮点与洞察

1. **巧妙的高频提取**：自注意力天然是低通滤波器，输出减去均值即可提取高频，简单而有效
2. **动态权重调度**：文本-视觉权重随训练阶段自适应，解决了无预训练冷启动问题
3. **DRPE零参数开销**：利用正弦编码将查询-原型关系注入注意力，不增加可训练参数
4. **统一的少样本/零样本框架**：LGPE 使得仅文本即可构建原型，自然支持零样本

## 局限与展望

- 零样本性能与大规模预训练模型（如 SegPoint）相比仍有差距，但训练资源和数据量不可比
- 文本嵌入来自冻结的CLIP，未探索更细粒度的文本描述
- 仅在室内场景数据集（S3DIS、ScanNet）验证，缺乏户外大规模场景实验
- 动态权重调度使用固定的指数衰减/增长函数，可考虑自适应学习

## 相关工作与启发

- Seg-PN（无预训练方法）启发了无预训练设计，但其丢弃高频信息是本文改进切入点
- ViT 中 Register token 的研究（Darcet et al.）启发了 ProERA 中寄存器的使用
- CLIP 的文本-视觉对齐启发了 LGPE 模块
- 对后续研究的启发：可将此框架扩展到其他3D任务（检测、实例分割），或探索更强的语言模型替代CLIP

## 评分

- 新颖性: ⭐⭐⭐⭐ — 三个模块设计各有创新，ProERA的高频提取思路新颖
- 实验充分度: ⭐⭐⭐⭐ — 多基准多设置，消融全面
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，频率分析可视化有说服力
- 价值: ⭐⭐⭐⭐ — 无需预训练，参数少，推理快

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] CLIPoint3D: Language-Grounded Few-Shot Unsupervised 3D Point Cloud Domain Adaptation](../../CVPR2026/3d_vision/clipoint3d_language-grounded_few-shot_unsupervised_3d_point_cloud_domain_adaptat.md)
- [\[ICCV 2025\] BUFFER-X: Towards Zero-Shot Point Cloud Registration in Diverse Scenes](../../ICCV2025/3d_vision/bufferx_towards_zeroshot_point_cloud_registration_in_diverse.md)
- [\[CVPR 2026\] Image-to-Point Cloud Feature Back-Projection for Multimodal Training of 3D Semantic Segmentation](../../CVPR2026/3d_vision/image-to-point_cloud_feature_back-projection_for_multimodal_training_of_3d_seman.md)
- [\[CVPR 2026\] PointGS: Semantic-Consistent Unsupervised 3D Point Cloud Segmentation with 3D Gaussian Splatting](../../CVPR2026/3d_vision/pointgs_semantic-consistent_unsupervised_3d_point_cloud_segmentation_with_3d_gau.md)
- [\[CVPR 2026\] Lite Any Stereo: Efficient Zero-Shot Stereo Matching](../../CVPR2026/3d_vision/lite_any_stereo_efficient_zero-shot_stereo_matching.md)

</div>

<!-- RELATED:END -->
