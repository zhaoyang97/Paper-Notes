---
title: >-
  [论文解读] G2SF: Geometry-Guided Score Fusion for Multimodal Industrial Anomaly Detection
description: >-
  [ICCV 2025][3D视觉][多模态异常检测] 提出 G2SF 框架，将基于 memory bank 的异常分数重新解释为局部特征空间中的各向同性欧氏距离，进而通过 Local Scale Prediction Network (LSPN) 学习方向感知的缩放因子，将其渐进演化为各向异性的统一融合度量，实现多模态工业异常检测 SOTA。
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "多模态异常检测"
  - "度量学习"
  - "各向异性距离"
  - "工业检测"
  - "点云-RGB融合"
---

# G2SF: Geometry-Guided Score Fusion for Multimodal Industrial Anomaly Detection

**会议**: ICCV 2025  
**arXiv**: [2503.10091](https://arxiv.org/abs/2503.10091)  
**代码**: [GitHub](https://github.com/ctaoaa/G2SF)  
**领域**: 3D视觉  
**关键词**: 多模态异常检测, 度量学习, 各向异性距离, 工业检测, 点云-RGB融合

## 一句话总结

提出 G2SF 框架，将基于 memory bank 的异常分数重新解释为局部特征空间中的各向同性欧氏距离，进而通过 Local Scale Prediction Network (LSPN) 学习方向感知的缩放因子，将其渐进演化为各向异性的统一融合度量，实现多模态工业异常检测 SOTA。

## 研究背景与动机

工业质量检测是现代制造业的关键环节。由于缺陷产品在真实场景中极为稀有，无监督异常检测（仅使用正常样本训练）成为主流方案。然而：

**单模态信息不完整**：3D 点云提供丰富的几何细节但缺乏纹理/颜色信息；2D RGB 图像包含丰富外观信息但缺乏几何线索。某些异常只在特定模态下可检测（如 foam 的切割缺陷在 RGB 中几乎不可见，cable gland 的螺纹异常在点云中区分度不足）。

**单模态异常分数区分能力不足**：基于 memory bank 的方法（如 PatchCore）用欧氏距离到最近原型的距离作为异常分数。但关键问题在于该距离是**各向同性**的——它忽略了局部特征空间中的方向性分布模式。正常区域也可能展现出较高的欧氏距离，导致融合后的区分效果不佳。

**现有融合策略粗糙**：分数相加（BTF）、取最大值（Shape-Guided）、或 OCSVM（M3DM）都无法有效处理单模态异常分数的不可靠性。特征自适应方法（LSFA）在降维过程中可能丢失检测关键细节。

**核心几何洞察**：memory bank 方法的异常分数本质上是局部特征空间中以原型为圆心的等距球面（各向同性）。但正常样本的分布通常具有方向性——沿某些方向延展更多。如果能学习一个各向异性的距离度量，就可以抑制正常方向上的分数、放大异常方向上的分数，从而大幅提升区分能力。

## 方法详解

### 整体框架

G2SF 采用冻结的 DINO（图像）和 Point-MAE（点云）提取特征，构建各模态的 memory bank，通过几何编码将特征分解为（原型、方向、距离）三元组，LSPN 预测方向感知缩放因子，与原始距离组合形成统一的各向异性度量。

### 关键设计

1. **几何特征编码 (Geometric Feature Encoding)**：将特征 $\mathbf{f}^m_i$ 相对于其最近 memory 原型 $\mathbf{m}^m_{i,j}$ 分解为三元组：
    $(\mathbf{m}^m_{i,j}, \mathbf{d}^m_{i,j}, s^m_{i,j}) = \mathcal{E}_{\mathbf{m}^m_{i,j}}(\mathbf{f}^m_i)$
   其中 $s^m_{i,j} = \|\mathbf{f}^m_i - \mathbf{m}^m_{i,j}\|$ 是欧氏距离（即传统异常分数），$\mathbf{d}^m_{i,j} = (\mathbf{f}^m_i - \mathbf{m}^m_{i,j}) / s^m_{i,j}$ 是单位方向向量。这个编码完整保留了原始特征的全部信息，避免了特征自适应中的信息丢失。

2. **Local Scale Prediction Network (LSPN)**：一个简单的 MLP 网络预测方向感知的缩放因子：
    $[w^P_{i,j}, w^R_{i,j}] = \Phi(\mathbf{m}^P_{i,j}, \mathbf{m}^R_{i,j}, \mathbf{d}^P_{i,j}, \mathbf{d}^R_{i,j})$
   LSPN 包含两个并行分支分别处理原型和方向输入，最终通过 $\exp(\tanh(\cdot))$ 激活函数输出缩放因子，确保 $w^m_{i,j} \in [e^{-1}, e^1]$，对称分布在 1 周围。对正常样本期望 $w < 1$（抑制），异常样本期望 $w > 1$（放大）。

3. **各向异性局部距离度量**：融合度量定义为：
    $l_{i,j} = \sum_{m \in \{P,R\}} w^m_{i,j} s^m_{i,j} \sigma^m$
   其中 $\sigma^m$ 是可训练的全局模态权重。当 LSPN 初始化为 $w \approx 1$ 时，该度量退化为联合特征空间的欧氏度量，然后在训练中逐渐演化为各向异性度量，避免了从零学习度量的过拟合风险。

### 损失函数 / 训练策略

损失由 **判别损失** 和 **几何保持损失** 两部分组成：

- **分离损失 $\mathcal{L}_{sep}$**：最小化正常样本的 $l_{i,0}$，最大化异常样本的 $l_{i,0}^{-1}$
- **边际损失 $\mathcal{L}_{mar}$**：减少正常/异常分布的重叠区域
- **一致性损失 $\mathcal{L}_{cns}$**：近邻原型的度量不应剧烈变化（近邻一致性），远处原型的度量应保持足够大（远端分离性）
- **缩放损失 $\mathcal{L}_{sc}$**：非对称地约束正常样本 $w \leq 1$、异常样本 $w \to e^1$
- **跨模态对齐损失 $\mathcal{L}_{cma}$**：通过随机置换索引构造负样本，强制 LSPN 利用跨模态对应关系

训练使用合成异常注入（基于 CutPaste 策略）。推理时取 $k+1$ 个近邻局部空间的度量最小值作为最终异常分数 $s_i = \min\{l_{i,j} | j=0,...,k\}$。

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文 | 之前SOTA (DAUP) | 提升 |
|--------|------|------|-----------------|------|
| MVTec-3D AD | I-AUROC | 0.971 | 0.970 | +0.001 |
| MVTec-3D AD | AUPRO@30% | - | 0.969(DAUP) | 竞争 |
| MVTec-3D AD (Mean) | I-AUROC | 0.971 | 0.970 | SOTA |

各类别 I-AUROC 中，Cable Gland 从 0.889(DAUP) 提升至 0.923，Foam 达到 0.991（SOTA），展示了跨模态融合的优势。

### 消融实验

| 配置 | I-AUROC | 说明 |
|------|---------|------|
| 各向同性基线 (Euclidean) | ~0.945 | PatchCore 基线 |
| + LSPN 方向感知缩放 | ~0.960 | 各向异性带来显著提升 |
| + 一致性损失 | ~0.965 | 近邻一致性约束 |
| + 跨模态对齐 | ~0.970 | 利用跨模态对应 |
| 完整 G2SF | 0.971 | 全部组件 |

几何推理的消融（score aggregation）比较了 mean、max、min 策略，证明 min 操作符最优（对应到正常数据流形的距离）。

### 关键发现

- 在 AUPRO@1%（更严格的指标）上优势尤为显著，说明方法在精确定位异常区域方面表现突出
- G2SF 在极少训练 epoch 内就达到了竞争性能，得益于从欧氏度量逐渐演化的设计
- Cable Gland 和 Foam 两个困难类别的提升最为明显，验证了跨模态互补的有效性
- $\exp(\tanh(\cdot))$ 激活函数的对称性确保了缩放因子不会过度偏离欧氏基线

## 亮点与洞察

- 将 memory bank 异常分数重新解释为几何距离是一个非常优雅的视角转换
- 从各向同性到各向异性的渐进演化策略是避免高维度量学习过拟合的巧妙方案
- 完全保留原始特征信息（通过几何编码），不做特征压缩/自适应，是一个重要的设计选择
- LSPN 结构简单但方向/位置双分支设计计算效率高

## 局限与展望

- LSPN 作为 MLP 可能无法捕捉复杂的非线性方向依赖
- 合成异常的质量和多样性可能限制 LSPN 的泛化能力
- 未探索超过两种模态（如 RGB + 点云 + 热成像）的扩展
- memory bank 的大小对推理速度有直接影响，大规模部署需要考虑效率

## 相关工作与启发

- PatchCore 的 memory bank 框架提供了坚实的基础，G2SF 在其上构建了更强的度量
- 局部 Mahalanobis 距离的思想被简化为方向缩放因子，降低了计算复杂度
- 合成异常注入策略（CutPaste）使无监督问题转化为半监督问题，值得其他任务借鉴

## 评分

- 新颖性: ⭐⭐⭐⭐ 几何度量演化的视角新颖，LSPN 设计精巧
- 实验充分度: ⭐⭐⭐⭐ 两个数据集、详尽消融、各类别分析
- 写作质量: ⭐⭐⭐⭐⭐ 从几何直觉出发，推导严谨清晰
- 价值: ⭐⭐⭐⭐ 对多模态融合异常检测具有实际工业应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] SiM3D: Single-Instance Multiview Multimodal and Multisetup 3D Anomaly Detection Benchmark](sim3d_single-instance_multiview_multimodal_and_multisetup_3d_anomaly_detection_b.md)
- [\[CVPR 2026\] Geometry-Aligned and Anomaly-Aware Reconstruction for 3D Anomaly Detection](../../CVPR2026/3d_vision/geometry-aligned_and_anomaly-aware_reconstruction_for_3d_anomaly_detection.md)
- [\[ICCV 2025\] Stable Score Distillation](stable_score_distillation.md)
- [\[ICCV 2025\] GeoSplatting: Towards Geometry Guided Gaussian Splatting for Physically-based Inverse Rendering](geosplatting_towards_geometry_guided_gaussian_splatting_for_physically-based_inv.md)
- [\[CVPR 2026\] Hierarchical Point-Patch Fusion with Adaptive Patch Codebook for 3D Shape Anomaly Detection](../../CVPR2026/3d_vision/hierarchical_point-patch_fusion_with_adaptive_patch_codebook_for_3d_shape_anomal.md)

</div>

<!-- RELATED:END -->
