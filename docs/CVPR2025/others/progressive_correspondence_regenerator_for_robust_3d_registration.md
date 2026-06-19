---
title: >-
  [论文解读] Regor: Progressive Correspondence Regenerator for Robust 3D Registration
description: >-
  [CVPR 2025][点云配准] Regor提出了一种渐进式对应关系再生策略，不同于传统的"自上而下"外点剔除方法，通过"自下而上"地在局部球体内迭代生成更多高质量对应关系，生成的正确匹配数量是现有方法的10倍，即使在弱特征条件下也能实现鲁棒配准。 点云配准是3D计算机视觉的基础任务，需要估计两组点云之间的刚性变换…
tags:
  - "CVPR 2025"
  - "点云配准"
  - "对应关系再生"
  - "外点剔除"
  - "三点一致性"
  - "渐进迭代"
---

# Regor: Progressive Correspondence Regenerator for Robust 3D Registration

**会议**: CVPR 2025  
**arXiv**: [2502.02163](https://arxiv.org/abs/2502.02163)  
**代码**: 即将发布  
**领域**: 其他（3D配准）  
**关键词**: 点云配准, 对应关系再生, 外点剔除, 三点一致性, 渐进迭代

## 一句话总结

Regor提出了一种渐进式对应关系再生策略，不同于传统的"自上而下"外点剔除方法，通过"自下而上"地在局部球体内迭代生成更多高质量对应关系，生成的正确匹配数量是现有方法的10倍，即使在弱特征条件下也能实现鲁棒配准。

## 研究背景与动机

点云配准是3D计算机视觉的基础任务，需要估计两组点云之间的刚性变换。在低重叠、高噪声和强自相似场景中，特征匹配往往产生大量外点。

现有外点剔除方法的根本问题：
- **几何方法**（RANSAC及变体、SC2-PCR）：通过几何一致性筛选对应关系，但在极端外点比例下难以正确识别内点
- **学习方法**（PointDSC、VBReg）：将外点剔除视为分类问题，但本质上仍依赖初始对应关系
- **共同缺陷**：所有方法都是"做减法"——从初始对应关系中剔除外点。当初始内点极少时，即使完美识别出所有内点，也不足以支撑稳健的位姿估计

关键洞察：与其"剔除错误匹配"，不如"生成更多正确匹配"——从做减法变为做加法。

## 方法详解

### 整体框架

Regor采用渐进迭代框架，每次迭代包含三个模块：(1) 先验引导的局部分组+广义互匹配，在局部区域生成新对应关系；(2) 中心感知三点一致性实现局部对应关系校正；(3) 全局对应关系精练从整体视角优化。通过多次迭代，逐步增加对应关系数量和质量。

### 关键设计

**设计一：先验引导的局部分组 + 广义互匹配（GMM）**

- **功能**：在局部小空间内高效生成新的对应关系，避免全局匹配的巨大搜索空间
- **核心思路**：以上一轮对应关系$\mathcal{G}^{t-1}$的位置先验为引导，在半径$r^t$内进行近邻搜索构建局部区域对$(\mathbf{P}_i^t, \mathbf{Q}_i^t)$。在局部区域内提出广义互匹配：计算双向NN和MNN匹配矩阵，通过Hadamard积获得互匹配矩阵，再用逻辑OR操作放松严格互约束：$\mathbf{M}^* = (\mathbf{M}_1^{P \to Q} \odot \mathbf{M}_2^{Q \to P}) \otimes (\mathbf{M}_1^{Q \to P} \odot \mathbf{M}_2^{P \to Q})$
- **设计动机**：局部区域特征相似度高，标准NN匹配会产生大量误匹配，但严格的互NN匹配可能找不到任何对应。GMM通过放松互约束平衡了精度和召回率。随着迭代进行，局部球体半径$r^t$逐步缩小以实现精确收敛

**设计二：中心感知三点一致性（CTC） — 局部对应关系校正**

- **功能**：利用种子点的先验信息，在局部区域内识别内点并校正错误对应
- **核心思路**：利用种子对应$g_i^{t-1}$（中心点）的高精度先验，计算CTC：$s_\Delta(g_j, g_k) = (s_\sigma(g_j, g_i^{t-1}) \cdot s_\sigma(g_i^{t-1}, g_k)) \| s_{\sigma/2}(g_j, g_k)$，其中$s_\sigma(g_i, g_j) = \mathbb{1}(|\|\mathbf{p}_i - \mathbf{p}_j\|_2 - \|\mathbf{q}_i - \mathbf{q}_j\|_2| \leq \sigma)$。高分数对应用于估计局部变换$\mathbf{T}_i^t$，再通过最近邻搜索纠正错误匹配
- **设计动机**：传统二点一致性无法区分相似结构的误匹配。引入中心点（准确先验）构成三点约束，大幅提升辨别力。额外的严格点对约束$s_{\sigma/2}$防止种子点错误时传播误差

**设计三：渐进迭代 + 全局精练 — 密度和精度的同步提升**

- **功能**：通过多轮迭代逐步增加内点数量并提升精度
- **核心思路**：每轮迭代先做局部再生+校正，再通过哈希表合并局部对应为全局对应$\mathcal{G}^t = \bigcup_{i=1}^n \mathcal{G}_i^t$。全局阶段使用二阶一致性识别高质量对应并纠正异常匹配。随迭代进行局部球体半径递减
- **设计动机**：单次匹配受限于搜索空间和特征质量。渐进策略使每轮都能利用前一轮的改进结果，形成正反馈循环。局部操作大幅降低问题规模和计算时间

### 损失函数

Regor为非学习方法，无需训练。最终位姿估计通过SVD直接从精练后的对应关系求解最小二乘。

## 实验关键数据

### 主实验：3DMatch数据集配准

| 方法 | 注册召回率(%) | 正确对应数量 |
|------|-------------|------------|
| RANSAC | 基线 | 基线 |
| SC2-PCR | 高 | N |
| PointDSC | 高 | N |
| MAC | 高 | N |
| **Regor** | **SOTA** | **~10×N** |

（注：Regor生成的正确对应数量是outlier removal方法的10倍）

### 消融实验：各模块贡献

| 配置 | 效果 |
|------|------|
| 无GMM（使用标准互匹配） | 局部匹配数量大幅减少 |
| 无CTC（使用标准二阶一致性） | 局部精度下降 |
| 无全局精练 | 跨区域一致性不足 |
| 减少迭代次数 | 对应数量和精度均降低 |

### 关键发现

- Regor在3DMatch和KITTI数据集上均达到SOTA，特别是在低重叠场景下优势显著
- 生成的正确对应关系数量是外点剔除方法的**10倍**，这对鲁棒位姿估计至关重要
- 即使使用弱特征描述子（非SOTA的特征提取器），Regor仍能实现鲁棒配准
- 渐进迭代中对应关系的数量和精度同步提升

## 亮点与洞察

1. **范式转变**：从"做减法"（外点剔除）到"做加法"（对应再生），根本性地解决了初始内点稀疏的问题
2. **利用先验的渐进策略**：每轮迭代的输出成为下一轮的先验，形成自我改进的良性循环
3. **实用价值突出**：不依赖特定特征提取器，可插入任何配准管线作为后处理模块

## 局限与展望

- 迭代过程引入额外计算时间，虽然局部操作控制了复杂度
- 核心参数（球体半径递减策略、阈值$\sigma$等）需要调节
- 未来可探索学习引导的迭代策略以进一步提升效率和鲁棒性

## 相关工作与启发

- **SC2-PCR**：二阶空间一致性的外点剔除，本文在此基础上提出中心感知三点一致性
- **GeoTransformer**：粗到细的Transformer匹配框架
- **TEASER**：基于TLS代价的鲁棒配准，使用旋转不变测量
- 启发：当问题的"做减法"方案难以奏效时，反思是否可以转为"做加法"——这种范式转换的思维方式具有广泛启发性

## 评分

⭐⭐⭐⭐ — 从外点剔除到对应再生的范式转换简洁而深刻，10倍正确匹配的结果令人印象深刻。方法设计清晰，各模块的动机合理。对3D配准领域提供了新的视角。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Towards In-the-Wild 3D Plane Reconstruction from a Single Image](towards_in-the-wild_3d_plane_reconstruction_from_a_single_image.md)
- [\[CVPR 2025\] MagicArticulate: Make Your 3D Models Articulation-Ready](magicarticulate_make_your_3d_models_articulation-ready.md)
- [\[CVPR 2026\] Progressive Neural Architecture Generation](../../CVPR2026/others/progressive_neural_architecture_generation.md)
- [\[CVPR 2025\] Tuning the Frequencies: Robust Training for Sinusoidal Neural Networks](tuning_the_frequencies_robust_training_for_sinusoidal_neural_networks.md)
- [\[CVPR 2026\] PAUL: Uncertainty-Guided Partition and Augmentation for Robust Cross-View Geo-Localization under Noisy Correspondence](../../CVPR2026/others/paul_uncertainty-guided_partition_and_augmentation_for_robust_cross-view_geo-loc.md)

</div>

<!-- RELATED:END -->
