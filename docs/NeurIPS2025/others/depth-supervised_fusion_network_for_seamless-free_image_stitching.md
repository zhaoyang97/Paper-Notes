---
title: >-
  [论文解读] Depth-Supervised Fusion Network for Seamless-Free Image Stitching
description: >-
  [NeurIPS 2025][图像拼接] DSFN 提出深度一致性约束的无缝图像拼接方法：通过深度感知的两阶段变换估计解决大视差对齐，软缝合区域扩散实现自然融合，结合重参数化策略提升效率，在 UDIS-D 和 IVSD 数据集上全面超越 SOTA。 领域现状：图像拼接将多视角图像合成宽视场全景图，广泛应用于全景摄影、遥感、医…
tags:
  - "NeurIPS 2025"
  - "图像拼接"
  - "深度监督"
  - "大视差对齐"
  - "软缝合融合"
  - "重参数化"
---

# Depth-Supervised Fusion Network for Seamless-Free Image Stitching

**会议**: NeurIPS 2025  
**arXiv**: [2510.21396](https://arxiv.org/abs/2510.21396)  
**代码**: [GitHub](https://github.com/DLUT-YRH/DSFN)  
**领域**: others (计算机视觉 / 图像拼接)  
**关键词**: 图像拼接, 深度监督, 大视差对齐, 软缝合融合, 重参数化

## 一句话总结
DSFN 提出深度一致性约束的无缝图像拼接方法：通过深度感知的两阶段变换估计解决大视差对齐，软缝合区域扩散实现自然融合，结合重参数化策略提升效率，在 UDIS-D 和 IVSD 数据集上全面超越 SOTA。

## 研究背景与动机

**领域现状**：图像拼接将多视角图像合成宽视场全景图，广泛应用于全景摄影、遥感、医学成像和 VR。

**传统方法局限**：
   - 基于特征的方法（SIFT+RANSAC+单应矩阵）假设场景为平面模型，多深度层场景下产生鬼影和错位
   - Mesh 变形（APAP 等）提升局部对齐但在深度不连续处引入伪影
   - 接缝优化计算量大

**深度学习方法痛点**：依赖合成训练数据导致跨域泛化弱；大视差场景下结构一致性仍然困难

**核心idea**：利用单目深度估计（Depth Anything）提供的深度信息作为几何先验，监督多视角对齐学习；用基于图的soft-seam扩散取代硬接缝切割

## 方法详解

### 整体框架

输入目标图像 $I_t$ 和参考图像 $I_r$ → ResNet50 特征编码 → 深度感知变换估计 → 对齐 → 软缝合融合 → 输出宽视场拼接结果 $I_s$

### 深度感知变换估计

**两阶段递进策略**：

**阶段一：粗对齐（1/16尺度）**
- 特征相关聚合（FCA）计算跨视角对应：$C_{i,j} = FCA(F_r^{1/16}, F_t^{1/16})$
- 回归四边形顶点偏移 $\Delta p \in \mathbb{R}^{4 \times 2}$
- DLT 求解粗糙单应矩阵：$H_C = \arg\min_H \sum_{k=1}^4 \|p_k' - H \cdot p_k\|_2^2$

**阶段二：精细对齐（1/8尺度）**
- 将目标特征用 $H_C$ 变换到参考空间
- 网格级偏移估计，RBF 插值生成连续形变场：

$$\Delta(x,y) = \sum_{m=1}^M w_m \phi(\|(x,y) - (x_m, y_m)\|)$$

其中 $\phi(r) = -e^{-(\epsilon r)^2}$ 为高斯基函数。最终密集变形场：

$$\mathcal{W}(p) = H_C \cdot p + \Delta(p)$$

**深度监督**：通过 Depth Anything 获取深度图 $I_{dr}, I_{dt}$，在重叠区域归一化后加入对齐损失：

$$\mathcal{L}_{depth} = f_{alignment}(I_{dr}, I_{dt}, \lambda', \gamma', \eta')$$

**总变换损失**：

$$\mathcal{L}^t = \mathcal{L}_{alignment} + \mu \mathcal{L}_{edge} + \zeta \mathcal{L}_{angle} + \xi \mathcal{L}_{depth}$$

其中 $\mathcal{L}_{edge}$ 限制网格拉伸，$\mathcal{L}_{angle}$ 约束非重叠区域邻边平行。

### 软缝合融合

**核心思想**：放松传统硬接缝定义，将重叠区域中任何需要融合的区域都视为潜在缝合区。

1. **SSE 模块**：基于 UNet 架构（普通卷积替换为空洞卷积，dilation rate 1-5），输入对齐图像的 mask，输出软缝合 mask $M_s$
2. **自适应权重**：$M_s$ 与原始 mask 经 sigmoid 生成像素级自适应融合权重 $M_{sr}, M_{st}$

**融合损失**：

$$\mathcal{L}^f = \rho \mathcal{L}_{terminal} + \tau \mathcal{L}_{cost} + \iota \mathcal{L}_{smooth} + \sigma \mathcal{L}_{reg}$$

- $\mathcal{L}_{cost}$：基于像素差平方的代价图，在 mask 变化处惩罚高代价区域
- $\mathcal{L}_{smooth}$：相邻像素平滑性约束
- $\mathcal{L}_{reg}$：深度一致性正则——对齐后的深度图在拼接区域的局部一致性

### 重参数化回归（RBA）

在 shift 回归中引入 RepBlock（1×1 + 3×3 卷积并行），训练时评估各分支贡献：

$$c_1 = \frac{\frac{1}{C_{out}}\sum \mathbf{w_1}}{\frac{1}{C_{out}}\sum \mathbf{w_1} + \frac{1}{C_{out}}\sum \mathbf{w_3}}$$

若 $c_1 < \hat{c}$（阈值），则将 1×1 分支耦合到 3×3：

$$\mathbf{W}_3^{new} = \mathbf{w_3} \cdot \mathbf{W_3} + \mathbf{w_1} \cdot pad(\mathbf{W_1})$$

实验选定 $\hat{c}=0.25$ 为最优阈值。

## 实验关键数据

### UDIS-D 数据集定量对比

| 方法 | PSNR↑ | SSIM↑ | SIQE↑ | LPIPS↓ |
|------|-------|-------|-------|--------|
| APAP | 23.792 | 0.794 | 41.707 | 0.472 |
| ELA | 24.012 | 0.808 | 41.781 | 0.470 |
| UDIS | 21.171 | 0.648 | 42.186 | 0.475 |
| UDIS++ | 25.426 | 0.837 | 43.184 | 0.469 |
| SRS | 24.828 | 0.811 | 41.857 | 0.473 |
| **DSFN (Ours)** | **25.467** | **0.839** | **43.732** | **0.462** |

### IVSD 数据集泛化验证

| 方法 | PSNR↑ | SSIM↑ | SIQE↑ | LPIPS↓ |
|------|-------|-------|-------|--------|
| UDIS++ | 26.649 | 0.819 | 46.383 | 0.439 |
| SRS | 24.234 | 0.796 | 35.641 | 0.445 |
| **DSFN (Ours)** | **26.778** | **0.820** | **46.568** | **0.436** |

跨数据集性能一致领先。

### 运行效率（512×512 图像）

| 方法 | 时间 (ms) |
|------|----------|
| APAP | 6683 |
| ELA | 8348 |
| UDIS | 194 |
| UDIS++ | 80 |
| SRS | 83 |
| **DSFN** | **67** |

DSFN 是最快的方法——尽管引入了深度估计和推理过程。

### 消融实验

| 配置 | PSNR | SSIM | SIQE | LPIPS |
|------|------|------|------|-------|
| w/o $\mathcal{L}_{smooth}$ | 25.431 | 0.833 | 43.156 | 0.466 |
| w/o $\mathcal{L}_{cost}$ | 25.438 | 0.836 | 43.186 | 0.463 |
| w/o $\mathcal{L}_{depth}$ | 25.434 | 0.838 | 43.703 | 0.463 |
| w/o $\mathcal{L}_{mesh}$ | 25.473 | 0.840 | 43.701 | 0.463 |
| **Full** | **25.470** | **0.839** | **43.732** | **0.462** |

去掉 mesh 约束虽然指标略升（放松了变形限制），但视觉上产生明显畸变。

### 用户研究
50 名参与者（30 名 CV 背景），在 1-5 分制评分中，DSFN 一致获得最高评分。

## 亮点与洞察
- **深度信息作为对齐先验**是处理大视差拼接的自然且有效的思路——利用现成的 Depth Anything 模型零成本获取
- **软缝合**替代硬接缝是关键创新：通过扩散 mask 实现像素级自适应融合，比 graph-cut 更平滑且可端到端训练
- **RBA 重参数化**在训练时保持多分支多样性、推理时合并为单分支——效率与性能兼顾
- 运行速度最快（67ms）说明整体架构设计紧凑高效

## 局限与展望
- 深度监督依赖 Depth Anything 的质量——若单目深度估计在特定场景不可靠则可能传播误差
- 仅在 UDIS-D 和 IVSD 两个数据集上验证，缺少大规模真实全景数据集（如 Google Street View）
- 未处理动态物体（移动行人/车辆）的遮挡问题
- 软缝合模块的空洞卷积参数是手动设定的，未做自动搜索
- 两阶段训练（变换+融合分开训练），端到端联合训练可能进一步提升

## 评分
- 新颖性: ⭐⭐⭐⭐ 深度监督+软缝合融合的组合新颖，RBA 策略有工程创新
- 实验充分度: ⭐⭐⭐⭐ 定量+定性+消融+用户研究+效率对比，较完整
- 写作质量: ⭐⭐⭐ 公式和损失函数定义清晰，但部分符号重载（如 $\sigma$ 既是激活函数又是损失权重）
- 价值: ⭐⭐⭐⭐ 对大视差图像拼接实际应用有直接价值，且速度最快

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Customized Fusion: A Closed-Loop Dynamic Network for Adaptive Multi-Task-Aware Infrared-Visible Image Fusion](../../CVPR2026/others/customized_fusion_a_closed-loop_dynamic_network_for_adaptive_multi-task-aware_in.md)
- [\[NeurIPS 2025\] Depth-Bounds for Neural Networks via the Braid Arrangement](depth-bounds_for_neural_networks_via_the_braid_arrangement.md)
- [\[ICCV 2025\] Revisiting Image Fusion for Multi-Illuminant White-Balance Correction](../../ICCV2025/others/revisiting_image_fusion_for_multi-illuminant_white-balance_correction.md)
- [\[NeurIPS 2025\] Semi-Supervised Regression with Heteroscedastic Pseudo-Labels](semi-supervised_regression_with_heteroscedastic_pseudo-labels.md)
- [\[ECCV 2024\] Momentum Auxiliary Network for Supervised Local Learning](../../ECCV2024/others/momentum_auxiliary_network_for_supervised_local_learning.md)

</div>

<!-- RELATED:END -->
