---
title: >-
  [论文解读] Learning to Normalize on the SPD Manifold under Bures-Wasserstein Geometry
description: >-
  [CVPR 2025][自监督学习][SPD流形] 本文提出 GBWBN，首个基于广义 Bures-Wasserstein 几何的 SPD 流形批归一化方法，引入可学习的度量参数和矩阵幂非线性变形来有效处理病态协方差矩阵，在骨骼动作识别和脑电分类上取得 SOTA。 领域现状：协方差矩阵（SPD 矩阵）在脑机接口、动作识别、无…
tags:
  - "CVPR 2025"
  - "自监督学习"
  - "SPD流形"
  - "Bures-Wasserstein度量"
  - "黎曼批归一化"
  - "病态矩阵"
  - "可学习几何"
---

# Learning to Normalize on the SPD Manifold under Bures-Wasserstein Geometry

**会议**: CVPR 2025  
**arXiv**: [2504.00660](https://arxiv.org/abs/2504.00660)  
**代码**: [https://github.com/jjscc/GBWBN](https://github.com/jjscc/GBWBN)  
**领域**: 流形学习 / SPD网络  
**关键词**: SPD流形、Bures-Wasserstein度量、黎曼批归一化、病态矩阵、可学习几何

## 一句话总结
本文提出 GBWBN，首个基于广义 Bures-Wasserstein 几何的 SPD 流形批归一化方法，引入可学习的度量参数和矩阵幂非线性变形来有效处理病态协方差矩阵，在骨骼动作识别和脑电分类上取得 SOTA。

## 研究背景与动机

**领域现状**：协方差矩阵（SPD 矩阵）在脑机接口、动作识别、无人机识别等领域广泛应用。SPD 矩阵不在欧氏空间中而在黎曼流形上，需要专用的黎曼神经网络（如 SPDNet）。黎曼批归一化（RBN）已被证明能提升 SPD 网络性能。

**现有痛点**：现有 RBN 使用 AIM（仿射不变度量）或 LEM（对数欧氏度量），但对病态 SPD 矩阵（条件数极大，如 $\kappa > 10^5$）表现差。Han et al. 证明 AIM 对 SPD 矩阵有二次依赖，对病态矩阵学习效率低。实际数据中病态矩阵非常普遍：HDM05 和 NTU RGB+D 数据集中 100% 的样本条件数超过 $10^5$。

**核心矛盾**：广泛使用的正则化（$X \leftarrow X + \lambda I$）只能保证正定性但无法有效缓解病态性。现有 RBN 的底层度量（AIM、LEM、LCM）对病态矩阵不友好。

**本文目标**：设计基于对病态矩阵更友好的 BW 度量的 RBN，并引入可学习几何参数。

**切入角度**：BWM 对 SPD 矩阵有线性依赖（vs AIM 的二次依赖），天然更适合病态场景。广义 BWM (GBWM) 通过一个 SPD 参数对 BWM 进行参数化，允许更灵活的几何表示。

**核心 idea**：基于 GBWM 构建 RBN，使度量参数可学习以自适应数据几何，再引入矩阵幂变形增强表示能力。

## 方法详解

### 整体框架
将 SPD 网络中的批归一化层替换为 GBWBN。对每个 batch 的 SPD 特征计算 BW 几何下的黎曼均值和方差，进行标准化、再缩放，并通过可学习 SPD 参数动态调整底层几何结构。

### 关键设计

1. **基于 BW 几何的批归一化**:

    - 功能：在 SPD 流形上进行归一化，处理病态矩阵
    - 核心思路：使用 BWM 的黎曼算子（测地线、对数映射、指数映射、Fréchet 均值）替代 AIM 下的对应算子来计算 batch 均值和方差。BWM 下 Lyapunov 算子的求解涉及特征分解后的逐元素运算，比 AIM 下的矩阵求逆更稳定。
    - 设计动机：BWM 的线性依赖使其对病态矩阵的小特征值不会产生数值爆炸。

2. **可学习广义 BW 度量 (GBWM)**:

    - 功能：让归一化空间的几何结构适应数据分布
    - 核心思路：GBWM 引入 SPD 参数 $M$ 来参数化 BWM，$M$ 设为可学习参数与网络一起训练。GBWM 局部等价于 AIM，因此兼具 BWM 的病态鲁棒性和 AIM 的几何灵活性。
    - 设计动机：固定度量可能不适应不同层/任务的数据分布变化，可学习度量增加自适应能力。

3. **矩阵幂非线性变形**:

    - 功能：进一步增强 GBWM 的表示能力
    - 核心思路：对 GBWM 应用矩阵幂变形 $d_p(X,Y) = d(X^p, Y^p)^{1/p}$，引入额外的非线性来改变流形的几何形状。
    - 设计动机：参考 LieBN 中矩阵幂变形对 AIM 性能的提升，将同样思路应用于 GBWM。

### 损失函数 / 训练策略
标准分类交叉熵损失。GBWBN 作为即插即用模块嵌入 SPDNet 等骨干网络。

## 实验关键数据

### 主实验
三个数据集上的分类精度：

| 数据集 | 任务 | GBWBN | 次优RBN | 提升 |
|--------|------|-------|---------|------|
| HDM05 | 动作识别 | SOTA | LieBN | 显著 |
| NTU RGB+D | 动作识别 | SOTA | LieBN | 显著 |
| MAMEM-SSVEP-II | 脑电分类 | SOTA | SPDBN | 显著 |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| BWM（固定几何） | 优于 AIM | BWM 对病态更鲁棒 |
| + 可学习 GBWM | 进一步提升 | 自适应几何有效 |
| + 矩阵幂变形 | 最佳 | 非线性增强有帮助 |
| 无 RBN | 显著下降 | RBN 对 SPD 网络仍然关键 |

### 关键发现
- 病态程度越高的数据集上 GBWBN 优势越明显
- 可学习度量参数确实在训练中发生了有意义的变化，不是固定不动的
- GBWBN 是即插即用的，可以替换任何现有 SPD 网络中的 RBN 层

## 亮点与洞察
- **几何驱动的归一化设计**：而非简单地将欧氏概念搬到流形上，而是根据数据特性（病态性）选择合适的底层几何，体现了"几何应该服务于数据"的理念。
- **可学习度量的思路**：让网络自己学习最适合的流形几何，类似于学习特征空间的"距离度量"。

## 局限与展望
- GBWM 下的计算比 AIM 更复杂（需要 Lyapunov 方程求解）
- 仅验证了分类任务，生成或对比学习等场景未探索
- 可学习度量的训练稳定性需进一步研究

## 相关工作与启发
- **vs SPDBN/LieBN**: 使用 AIM/LEM 度量，对病态矩阵不友好
- **vs ManifoldNorm**: 使用一阶和二阶统计量但在 AIM/LEM 下，本文在 BWM 下

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个基于 BW 几何的 RBN，可学习度量有新意
- 实验充分度: ⭐⭐⭐ 三个数据集但都偏小，更大规模验证需要
- 写作质量: ⭐⭐⭐⭐ 数学推导严谨，背景介绍充分
- 价值: ⭐⭐⭐⭐ 对 SPD 网络社区的重要改进

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] SegMASt3R: Geometry Grounded Segment Matching](../../NeurIPS2025/self_supervised/segmast3r_geometry_grounded_segment_matching.md)
- [\[ICML 2025\] Generalization Analysis for Supervised Contrastive Representation Learning under Non-IID Settings](../../ICML2025/self_supervised/generalization_analysis_for_supervised_contrastive_representation_learning_under.md)
- [\[ICCV 2025\] WIR3D: Visually-Informed and Geometry-Aware 3D Shape Abstraction](../../ICCV2025/self_supervised/wir3d_visually-informed_and_geometry-aware_3d_shape_abstraction.md)
- [\[CVPR 2026\] Teaching DINOv3 About Partial 3D Geometry: A Self-Supervised Geometry-Aware Approach](../../CVPR2026/self_supervised/teaching_dinov3_about_partial_3d_geometry_a_self-supervised_geometry-aware_appro.md)
- [\[CVPR 2026\] Reframing Long-Tailed Learning via Loss Landscape Geometry](../../CVPR2026/self_supervised/reframing_long-tailed_learning_via_loss_landscape_geometry.md)

</div>

<!-- RELATED:END -->
