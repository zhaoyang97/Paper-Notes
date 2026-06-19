---
title: >-
  [论文解读] MobileMamba: Lightweight Multi-Receptive Visual Mamba Network
description: >-
  [CVPR 2025][模型压缩][lightweight network] 提出 MobileMamba 轻量级视觉网络，通过三阶段粗粒度架构设计和 MRFFI 细粒度模块（融合 Mamba 全局建模、多核卷积多尺度感知和 Identity 冗余消除），在分类和下游高分辨率任务上实现速度与精度的最优平衡。
tags:
  - "CVPR 2025"
  - "模型压缩"
  - "lightweight network"
  - "Mamba"
  - "SSM"
  - "multi-receptive field"
  - "wavelet transform"
  - "efficient inference"
---

# MobileMamba: Lightweight Multi-Receptive Visual Mamba Network

**会议**: CVPR 2025  
**arXiv**: [2411.15941](https://arxiv.org/abs/2411.15941)  
**代码**: [GitHub](https://github.com/lewandofskee/MobileMamba)  
**领域**: 模型压缩  
**关键词**: lightweight network, Mamba, SSM, multi-receptive field, wavelet transform, efficient inference

## 一句话总结

提出 MobileMamba 轻量级视觉网络，通过三阶段粗粒度架构设计和 MRFFI 细粒度模块（融合 Mamba 全局建模、多核卷积多尺度感知和 Identity 冗余消除），在分类和下游高分辨率任务上实现速度与精度的最优平衡。

## 研究背景与动机

**领域现状**: 移动端轻量模型主要分两类：
- **CNN-based**（MobileNet, GhostNet）：局部感受野（ERF），缺乏长程依赖，高分辨率下需增大计算量
- **ViT-based**（EfficientViT, SHViT）：全局 ERF，但二次方计算复杂度在高分辨率场景下代价高

**核心矛盾**: 近期的 Mamba（状态空间模型）以线性复杂度实现全局建模，但现有轻量 Mamba 网络（EfficientVMamba、LocalVim）**虽然 FLOPs 低但实际吞吐量很差**。FLOPs 不等于推理速度——扫描方式的内存访问模式、网络拓扑等因素严重影响实际吞吐量。

**本文切入点**: 从粗粒度（宏观架构）和细粒度（模块设计）两个层面系统优化 Mamba-based 轻量网络的效率和性能。

## 方法详解

### 整体框架

MobileMamba 采用**三阶段**网络结构（vs 主流四阶段）：

- **16×16 PatchEmbed**: 首次下采样直接降到 H/16×W/16（四阶段是 H/4×W/4）
- **三个阶段**: 逐步下采样到 H/64×W/64
- **优势**: 特征图更小→计算量更低→推理速度更快，同等吞吐量下精度更高（+0.4% Top-1）

每个阶段由多个 MobileMamba Block 组成，Block 内包含：局部信息感知 → **MRFFI** → FFN。

### 关键设计

#### 1. MRFFI 多感受野特征交互模块

将输入特征沿通道维度分为三部分：

**Part 1 — WTE-Mamba（长程小波增强 Mamba）**:
- Mamba 模块提取全局特征（双向扫描）
- Haar 小波变换提取高频边缘细节（LL/LH/HL/HH 子带）
- 小波域卷积的 ERF 更大且计算复杂度更低
- 两路特征相加融合：$x_G^O = x_m^O + x_w^O$

**Part 2 — MK-DeConv（多核深度可分离卷积）**:
- 将通道分为 $n$ 组，每组使用不同大小的卷积核 $(2j+1)$
- $j=1,2,...,n$ 对应 3×3, 5×5, 7×7 等
- 多尺度局部感受野，增强相邻信息提取

**Part 3 — Eliminate Redundant Identity**:
- 剩余通道直接跳连（Identity mapping）
- 减少高维空间的特征冗余
- 降低计算复杂度，提升处理速度

最终拼接三部分：$x^O = \text{Concat}(x_G^O, x_L^O, x^I[\text{identity channels}])$

#### 2. 训练策略

- **知识蒸馏**: 使用大模型作为 teacher，软蒸馏（DeiT 方式）
- **延长训练 Epochs**: 小模型 300 epochs 未收敛，延长至 1000 epochs

#### 3. 测试策略

- **Normalization Layer Fusion**: 将 BN 层融入卷积/线性层，减少推理时的层数和计算

### 损失函数

标准交叉熵分类损失 + 知识蒸馏的 KL 散度损失。

## 实验关键数据

### 主实验表（ImageNet-1K 分类）

| 模型 | FLOPs | Throughput | Params | Top-1 |
|------|-------|-----------|--------|-------|
| EfficientVMamba-T | 800M | — | 6.0M | 76.5 |
| LocalVim-T | 1500M | — | 8.0M | 76.2 |
| MobileMamba-S6 | 652M | — | 15.0M | 78.0 |
| **MobileMamba-S6†** | **652M** | — | 15.0M | **80.7** |
| SHViT-S3 | 601M | — | 14.2M | 77.4 |
| EfficientViT-M5 | 522M | — | 12.4M | 77.1 |
| **MobileMamba-B4†** | ~4G | — | — | **83.6** |

MobileMamba-S6†（带训练策略）在 652M FLOPs 下达到 80.7%，超越所有同量级 CNN/ViT/Mamba 方法。

### 下游任务性能

| 任务 | 对比方法 | MobileMamba 提升 |
|------|---------|----------------|
| Mask RCNN 检测 | vs EMO | mAP^b +1.3↑, mAP^m +1.0↑, 吞吐 +56%↑ |
| RetinaNet 检测 | vs EfficientVMamba | mAP^b +2.1↑, 吞吐 ×4.3↑ |
| PSPNet 语义分割 | vs MobileNetv2 | mIoU +7.2↑, 仅 8.5% FLOPs |
| PSPNet 语义分割 | vs MobileViTv2 | mIoU +0.4↑, 仅 11.2% FLOPs |

### 消融表（替换 Mamba 为其他 RNN 范式）

| 方法 | FLOPs | Throughput | Top-1 |
|------|-------|-----------|-------|
| TTT | 625M | 9569 | 77.0 |
| xLSTM | 695M | 6868 | 77.3 |
| RWKV6 | 658M | 10331 | 77.8 |
| **Mamba** | **652M** | **11000** | **78.0** |

### 关键发现

1. **三阶段优于四阶段**: 同等吞吐量下精度更高（+0.4%），同等精度下吞吐量更快
2. **MRFFI 三路分割有效**: 全局 Mamba + 多尺度局部卷积 + Identity，各有不可替代的作用
3. **小波变换增益显著**: WTE-Mamba 中的小波分支增强了高频边缘信息提取
4. **训练策略加成大**: 知识蒸馏 + 1000 epochs 可将 78.0% 提升至 80.7%（+2.7%）
5. **Mamba 优于其他 RNN 范式**: 对比 TTT、xLSTM、RWKV6，Mamba 在吞吐量和精度上均最优
6. **高分辨率下游任务优势明显**: 线性复杂度在检测/分割任务上的效率优势远超 CNN 和 ViT

## 亮点与洞察

1. **系统性设计**: 从宏观架构（三阶段）到微观模块（MRFFI）再到训练/测试策略，三个层面协同优化
2. **实际吞吐量驱动**: 不仅追求低 FLOPs，更关注实际推理速度（比 LocalVim 快 21×）
3. **多感受野融合思路新颖**: Mamba 全局 + 小波高频 + 多核局部 + Identity 冗余消除的通道分割策略
4. **下游任务验证充分**: 在 5 个不同的检测/分割框架上全面验证，不仅是分类

## 局限性

1. **参数量偏大**: MobileMamba-S6 有 15M 参数，相比一些极轻量模型（1-6M params）偏大
2. **Mamba 算子成熟度**: Mamba 的 CUDA 算子优化不如 CNN/ViT 成熟，部分设备支持受限
3. **三阶段的下游适配**: 三阶段的特征图尺度（H/16, H/32, H/64）与主流四阶段检测/分割头不完全兼容
4. **小波变换增加实现复杂度**: Haar 小波变换虽然理论简洁，但增加了工程实现和调试难度
5. **知识蒸馏依赖**: MobileMamba†的最佳结果依赖 teacher 模型，限制了独立部署

## 相关工作与启发

- **EfficientViT / SHViT**: ViT-based 轻量化的重要参考，三阶段结构设计受 EfficientViT 启发
- **EfficientVMamba / LocalVim**: Mamba-based 轻量化先驱，本文在此基础上大幅提升速度和精度
- **GhostNet**: Identity/Cheap Operation 减少冗余计算的思想在 MRFFI 中得到继承
- **启发**: 「通道维度分而治之」（部分全局、部分局部、部分跳连）是一种高效且通用的轻量模块设计范式

## 评分 ⭐

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐⭐ |
| 工程实用性 | ⭐⭐⭐⭐⭐ |
| 总体推荐 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] JamMa: Ultra-lightweight Local Feature Matching with Joint Mamba](jamma_ultra-lightweight_local_feature_matching_with_joint_mamba.md)
- [\[CVPR 2025\] Mamba-Adaptor: State Space Model Adaptor for Visual Recognition](mamba-adaptor_state_space_model_adaptor_for_visual_recognition.md)
- [\[CVPR 2025\] Binarized Mamba-Transformer for Lightweight Quad Bayer HybridEVS Demosaicing](binarized_mamba-transformer_for_lightweight_quad_bayer_hybridevs_demosaicing.md)
- [\[CVPR 2025\] EfficientViM: Efficient Vision Mamba with Hidden State Mixer based State Space Duality](efficientvim_efficient_vision_mamba_with_hidden_state_mixer_based_state_space_du.md)
- [\[CVPR 2025\] Embracing Collaboration Over Competition: Condensing Multiple Prompts for Visual In-Context Learning](embracing_collaboration_over_competition_condensing_multiple_prompts_for_visual_.md)

</div>

<!-- RELATED:END -->
