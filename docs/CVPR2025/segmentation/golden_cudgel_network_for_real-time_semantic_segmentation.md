---
title: >-
  [论文解读] Golden Cudgel Network for Real-Time Semantic Segmentation
description: >-
  [CVPR 2025][语义分割][图像分割] 提出 GCNet，核心是 Golden Cudgel Block (GCBlock)，训练时自膨胀（多卷积多路径）提升学习能力，推理时自收缩（重参数化为单个 3×3 卷积）加速推理，无需外部教师模型即成为"自蒸馏"方案，在 Cityscapes 上以 77.3% mIoU / 193.3 FPS 超越现有实时分割模型。
tags:
  - "CVPR 2025"
  - "语义分割"
  - "图像分割"
  - "reparameterization"
  - "GCBlock"
  - "dual-branch"
  - "Cityscapes"
---

# Golden Cudgel Network for Real-Time Semantic Segmentation

**会议**: CVPR 2025  
**arXiv**: [2503.03325](https://arxiv.org/abs/2503.03325)  
**代码**: [GitHub](https://github.com/gyyang23/GCNet)  
**领域**: 图像分割  
**关键词**: real-time semantic segmentation, reparameterization, GCBlock, dual-branch, Cityscapes

## 一句话总结

提出 GCNet，核心是 Golden Cudgel Block (GCBlock)，训练时自膨胀（多卷积多路径）提升学习能力，推理时自收缩（重参数化为单个 3×3 卷积）加速推理，无需外部教师模型即成为"自蒸馏"方案，在 Cityscapes 上以 77.3% mIoU / 193.3 FPS 超越现有实时分割模型。

## 研究背景与动机

**领域现状**: 实时语义分割模型追求精度与速度的平衡。单分支模型（ERFNet, STDC, SCTNet）通过轻量设计或知识蒸馏提升表现；多分支模型（BiSeNet, DDRNet, PIDNet）通过语义+细节双分支架构增强空间细节捕捉。

**现有痛点**:
1. **多路径块降低推理速度**: 残差连接增加显存访问频率，Conv-Former Block 等类 Transformer 结构进一步影响效率
2. **依赖外部教师模型**: SCTNet 等方法需要高性能 Transformer 分割模型（如 SegFormer）进行知识蒸馏，增加训练成本和复杂度

**核心矛盾**: 多路径结构有利于训练（防梯度消失/爆炸），但不利于推理速度；单路径结构推理快但学习能力弱。

**本文目标**: 同时利用多路径块的训练优势和单路径块的推理优势，且不依赖外部教师模型。

**切入角度**: 结构重参数化——训练时展开为多卷积多路径，推理时无损合并为单个卷积。

**核心 idea**: 像金箍棒一样，训练时可大（多路径多卷积），推理时可小（单个 3×3 卷积），自身即为"教师"与"学生"的统一。

## 方法详解

### 整体框架

GCNet 采用双分支架构：
1. **Stem**: 两个 stride=2 的 3×3 卷积快速下采样
2. **Stage 2-3**: 共享的 GCBlock 堆叠
3. **Stage 4-6**: 分为语义分支（深层语义）和细节分支（空间细节），通过双向特征融合（卷积调整通道+双线性插值）交互
4. **PPM**: 语义分支末端接 Deep Aggregation Pyramid Pooling Module
5. **分割头**: 3×3 卷积融合 + 1×1 卷积对齐类别数

三个版本：GCNet-S（C=32）、GCNet-M（C=64）、GCNet-L（C=64，更深）

### 关键设计

#### 1. Golden Cudgel Block (GCBlock) — 核心创新

**训练结构**（多卷积多路径）：
- **Path₃ₓ₃_₁ₓ₁** (×N): 一个 3×3 卷积 + 一个 1×1 卷积，N 条并行路径
- **Path₁ₓ₁_₁ₓ₁**: 两个 1×1 卷积的串联
- **Path_residual**: 残差连接（stride=1 时使用，通过 BN 的恒等卷积实现）

**推理结构**（单个 3×3 卷积）：
- Conv-BN 融合: $W' = \frac{\gamma}{\sqrt{\sigma+\varepsilon}}W$
- 垂直重参数化: 3×3 后接 1×1 等价合并为新的 3×3（利用 im2col 的矩阵乘法等价性）
- 水平重参数化: 多条并行路径的 3×3 卷积直接对权重和偏置求和

关键发现：去掉 bottleneck 中的第一个 1×1 卷积（因其训练后参数值太小影响无损融合）；Path₁ₓ₁_₁ₓ₁ 堆叠 2 层效果最佳。

#### 2. 自蒸馏机制

训练时膨胀的 GCNet 相当于"教师模型"，推理时收缩的版本相当于"学生模型"。重参数化保证无损转换，无需传统的两阶段蒸馏流程。

#### 3. 轻量特征融合

双分支间的特征融合仅用 3×3 卷积（通道压缩/扩展）+ 双线性插值（上/下采样），**不使用任何注意力模块**，避免额外推理开销。

### 损失函数

$$L = L_{sh} + \alpha L_{ash}$$

- $L_{sh}$: 主分割头的 OHEM Cross Entropy 损失
- $L_{ash}$: 辅助分割头的 OHEM Cross Entropy 损失（训练时用于深监督，推理时移除）
- $\alpha = 0.4$

## 实验关键数据

### 主实验表（Cityscapes validation set, A100）

| 模型 | mIoU (%) | FPS | Params | ImageNet |
|------|----------|-----|--------|----------|
| DDRNet-23-Slim | 76.3 | 166.4 | 5.7M | ✗ |
| PIDNet-S | 76.4 | 128.7 | 7.7M | ✗ |
| SCTNet-B-Seg100 | 79.0 | 117.0 | 17.4M | ✗ |
| **GCNet-S** | **77.3** | **193.3** | 9.2M | ✗ |
| **GCNet-M** | **79.0** | **105.0** | 34.2M | ✗ |
| **GCNet-L** | **79.6** | **88.0** | 45.2M | ✗ |

- GCNet-S 以 193.3 FPS 实现 77.3% mIoU，同速度段无竞争对手
- GCNet-M 与 SCTNet-B-Seg100 精度持平（79.0%）但**无需 ImageNet 预训练和教师模型**
- GCNet-L 达到 79.6% mIoU，在不用预训练的实时模型中最高

### 消融表

**Path₁ₓ₁_₁ₓ₁ 中 1×1 卷积数量（GCNet-S）**：

| 数量 | 显存 | 训练时间 | mIoU |
|------|------|----------|------|
| 0 | 20.58 GiB | 4.0h | 76.1 |
| 1 | 21.87 GiB | 4.5h | 76.6 |
| **2** | **24.61 GiB** | **5.0h** | **76.7** |
| 3 | 27.31 GiB | 5.4h | 76.4 |

2 层最优，3 层过拟合反降。

**Path₃ₓ₃_₁ₓ₁ 数量 N**: 增加 N 可加速收敛（N=3 在 20k 步时相较 N=1 提升显著），但最终精度差异不大，N 越大训练越稳定。

### 关键发现

1. 重参数化 **无精度损失**：训练→推理结构转换是数学等价的
2. 不用 ImageNet 预训练即可达到与使用预训练的模型竞争的精度
3. 辅助分割头（深监督）在 stage 4 后效果最佳（α=0.4）
4. 在 CamVid 和 Pascal VOC 2012 上也展现一致优势

## 亮点与洞察

1. **"金箍棒"隐喻精妙**: 训练膨胀/推理收缩的设计理念直观易懂，命名巧妙
2. **彻底去除教师依赖**: 自身训练时的大模型即为教师，推理时的小模型为学生，无需额外训练流程
3. **重参数化数学推导完整**: 从 Conv-BN 融合到垂直/水平合并的每一步都有严格公式
4. **纯从头训练**: 不使用 ImageNet 预训练，降低了方法的前置条件要求
5. **速度优势明显**: GCNet-S 在全分辨率 1024×2048 上达到 193.3 FPS，远超同精度段模型

## 局限与展望

1. GCNet-L 的参数量（45.2M）在实时模型中偏大，边缘设备部署受限
2. 双分支融合未使用注意力机制，虽提速但可能损失细粒度特征对齐
3. 仅在城市场景数据集验证，未评估 ADE20K 等更大规模/更多类别的数据集
4. 重参数化技术（RepVGG 系列）并非首创，创新主要在将其系统化应用于分割任务
5. 训练时计算和显存开销随路径数 N 增加而增长，需合理选择

## 相关工作与启发

1. **RepVGG** (Ding et al., 2021): 重参数化的先驱工作，GCBlock 的核心思想来源
2. **DDRNet** (Hong et al., 2021): 双分支架构和 DAPPM 的来源
3. **PIDNet** (Xu et al., 2023): 三分支架构，GCNet 证明双分支+重参数化即可胜过
4. **SCTNet** (Xu et al., 2024): 使用教师模型的单分支方案，GCNet 免教师达到同等精度

**启发**: 重参数化不仅适用于分类（RepVGG），在密集预测任务中同样有效。训练时"可大可小"的思路可推广到其他计算敏感的视觉任务（如目标检测、深度估计）。

## 评分

⭐⭐⭐⭐ (4/5)

- **创新性**: ⭐⭐⭐ — 核心思想（重参数化）非全新，但在语义分割中的系统化应用和"金箍棒"设计有新意
- **实验充分度**: ⭐⭐⭐⭐ — 三个数据集、详细消融、公平 GPU 统一测速
- **论文写作**: ⭐⭐⭐⭐ — 数学推导详尽，结构清晰
- **实用价值**: ⭐⭐⭐⭐⭐ — 无需预训练/教师模型、开源代码、速度优异，工程落地友好

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] PicoSAM3: Real-Time In-Sensor Region-of-Interest Segmentation](picosam3_real-time_in-sensor_region-of-interest_segmentation.md)
- [\[CVPR 2026\] The Golden Subspace: Where Efficiency Meets Generalization in Continual Test-Time Adaptation](../../CVPR2026/segmentation/the_golden_subspace_where_efficiency_meets_generalization_in_continual_test-time.md)
- [\[CVPR 2025\] Condensing Action Segmentation Datasets via Generative Network Inversion](condensing_action_segmentation_datasets_via_generative_network_inversion.md)
- [\[CVPR 2025\] Universal Domain Adaptation for Semantic Segmentation](universal_domain_adaptation_for_semantic_segmentation.md)
- [\[NeurIPS 2025\] Attention (as Discrete-Time Markov) Chains](../../NeurIPS2025/segmentation/attention_as_discrete-time_markov_chains.md)

</div>

<!-- RELATED:END -->
