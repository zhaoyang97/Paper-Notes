---
title: >-
  [论文解读] BHViT: Binarized Hybrid Vision Transformer
description: >-
  [CVPR 2025][模型压缩][二值化神经网络] 针对 ViT 二值化性能严重下降的问题，提出专为二值化设计的混合 ViT 架构 BHViT，包含多尺度分组空洞卷积 token mixer、量化分解注意力矩阵二值化、shift 增强的 MLP 和正则化损失，在 ImageNet-1K 上达到 1-bit 二值化模型的 SOTA 性能。
tags:
  - "CVPR 2025"
  - "模型压缩"
  - "二值化神经网络"
  - "Transformer"
  - "混合架构"
  - "量化分解"
  - "权重振荡"
---

# BHViT: Binarized Hybrid Vision Transformer

**会议**: CVPR 2025  
**arXiv**: [2503.02394](https://arxiv.org/abs/2503.02394)  
**代码**: [GitHub](https://github.com/IMRL/BHViT)  
**领域**: 模型压缩 / 二值化  
**关键词**: 二值化神经网络, Vision Transformer, 混合架构, 量化分解, 权重振荡

## 一句话总结

针对 ViT 二值化性能严重下降的问题，提出专为二值化设计的混合 ViT 架构 BHViT，包含多尺度分组空洞卷积 token mixer、量化分解注意力矩阵二值化、shift 增强的 MLP 和正则化损失，在 ImageNet-1K 上达到 1-bit 二值化模型的 SOTA 性能。

## 研究背景与动机

ViT 模型规模大、计算复杂度高，难以在资源受限的边缘设备上部署。模型二值化（将权重和激活限制为 +1/-1）是最极端的量化方案，可用 XNOR 和 popcount 操作替代矩阵乘法，大幅降低计算和存储开销。

然而，直接将已有的 CNN 二值化技术（如 ReActNet 中的 RSign、RPReLU）应用到 ViT 上会导致严重性能下降。如图 1 所示，ReActNet 在 CNN 架构上表现尚可，但迁移到 ViT 架构后准确率大幅下跌。

**两个核心原因**：
1. 注意力模块中多个 clip 函数和 sign 算子导致梯度消失——反向传播被严重破坏
2. 二值化注意力矩阵无法准确表示不同 token 间的相似度差异——信噪比急剧下降

## 方法详解

### 整体框架

BHViT 采用四阶段特征金字塔结构，每阶段通道翻倍、空间减半：
- **Stage 1-2**：使用二值多尺度分组空洞卷积（MSGDC）作为 token mixer——避免早期大量 token 带来的注意力退化
- **Stage 3-4**：使用二值多尺度多头注意力（MSMHA）——在 token 数量减少后发挥全局建模优势
- 每个 block 内搭配 shift 增强的二值 MLP

### 关键设计

#### 1. 多尺度分组空洞卷积（MSGDC）

**功能**：替代前两阶段的自注意力，实现局部多尺度特征融合  
**核心思路**：使用三组不同膨胀率（dil=1,3,5）的 3×3 分组二值卷积，每组后加 RPReLU 激活和残差连接，最终求和并 BN  
**设计动机**：
- Observation 1 表明过多 token 对二值 ViT 有害——前两阶段 token 数巨大（如 56×56），自注意力计算复杂度高且二值化后注意力矩阵趋近均匀分布
- 分组卷积大幅减少参数量和计算量，多尺度膨胀率覆盖不同感受野

#### 2. 多尺度多头注意力（MSMHA）

**功能**：在后两阶段执行高效全局注意力  
**核心思路**：将输入分为窗口级特征和全局下采样特征，拼接后生成 Q/K/V，实现局部+全局混合注意力  
**具体流程**：
- 对输入做 7×7 平均池化得到高尺度特征 → 同时将输入划分为 7×7 窗口
- 拼接窗口特征和重复后的高尺度特征作为隐状态 H
- H 经三个二值线性层生成 Q/K/V，计算注意力
- 为 Q/K/V 各添加残差连接（Observation 2），缓解梯度消失

#### 3. 量化分解（Quantization Decomposition, QD）

**功能**：解决二值注意力矩阵无法区分 token 重要性的问题  
**核心思路**：引入全局缩放常数 $s=2^n-1$（n=2，即 s=3），将注意力矩阵分解为 s 个二值矩阵

$$\hat{A}_{tt}^\sigma = \varphi(\text{round}(s \cdot A_{tt}) \geq \sigma - 0.5), \quad \sigma = (1, 2, \ldots, s)$$

每个 token 的重要性由它在多少个二值矩阵中被"激活"来表示（0 次、1 次、2 次或 3 次），实现准 2-bit 的注意力权重区分。最终输出为所有二值矩阵与 V 的乘积之和。

**设计动机**：原始二值注意力只有 0/1 两态，softmax 后的连续权重信息几乎完全丢失；QD 通过多阈值分解恢复了部分排序信息。

#### 4. Shift 增强的二值 MLP

**功能**：增强二值 MLP 的表示能力  
**核心思路**：在 MLP 的两个二值线性层之外，添加两组 shift 操作（水平/垂直/混合移位），通过无参数的空间位移引入邻域信息  
**设计动机**：二值 MLP 的信息损失严重，shift 操作不引入额外计算（仅数据搬运），但能有效融合邻近 token 的特征

### 损失函数 / 训练策略

$$L = (1-\lambda-\beta)L_{cls} + \lambda L_{dis} + \beta L_{re}$$

- $L_{cls}$：交叉熵分类损失
- $L_{dis}$：用 DeiT-Small 蒸馏的知识蒸馏损失（$\lambda=0.8$ 时最优）
- $L_{re}$：正则化损失 $\frac{1}{n}\sum|\ |w_i|-1\ |$，强制隐权重远离 0 靠近 ±1

**关键发现（Observation 3）**：Adam 优化器的二阶动量在训练后期放大了二值网络的权重振荡。当权重在 0 附近反复跳变时，一阶动量的正负梯度相互抵消（分子趋零），而二阶动量持续累积（分母增大），导致有效梯度 $g_t' \to 0$，大量参数停止更新。正则化损失在最后 10% 的 epoch 激活（$\beta=0.1$），将摇摆的权重推向 ±1。

## 实验关键数据

### 主实验（ImageNet-1K 分类）

| 模型 | W-A (bit) | OPs (G) | Top-1 (%) |
|------|-----------|---------|-----------|
| ReActNet (CNN) | 1-1 | 4.69 | 65.5 |
| BiViT | 1-1 | - | 58.6 |
| Bi-ViT (AAAI'24) | 1-1 | 9.87 | 63.8 |
| Si-BiViT | 1-1 | 9.87 | 63.8 |
| **BHViT-Small** | **1-1** | **3.5** | **68.4** |
| **BHViT-Small†** (全精度下采样) | **1-1** | - | **70.1** |

BHViT-Small 以更少的 OPs 实现了 68.4% 的准确率，比之前最好的二值 ViT 高出近 5 个百分点。

### 消融实验（CIFAR-10）

| Shift | MSGDC | MSMHA | QD | RL | FDL | Top-1 (%) |
|-------|-------|-------|----|----|-----|-----------|
| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **95.0** |
| ✓ | ✓ | ✓ | ✓ | ✓ | - | 92.1 |
| ✓ | ✓ | ✓ | ✓ | - | - | 90.7 |
| ✓ | ✓ | ✓ | - | - | - | 88.9 |
| ✓ | ✓ | - | - | - | - | 86.7 |
| ✓ | - | - | - | - | - | 85.6 |
| - | - | - | - | - | - | 83.2 |

每个模块贡献约 1.1-2.9 个百分点。全精度下采样层（FDL）贡献最大（+2.9%），其次是正则化损失（+1.4%）和量化分解（+1.8%）。

| 架构类型 | Token Mixer | Top-1 (%) |
|---------|-------------|-----------|
| 混合 (BHViT) | Hybrid | **70.1** |
| 纯 ViT | MSMHA only | 68.8 |
| 纯 CNN | MSGDC only | 67.2 |

混合架构比纯 ViT 和纯 CNN 都更适合二值化。

### 关键发现

1. 过多 token 对二值注意力有害——信息熵分析表明 token 越多，softmax 后注意力越趋近均匀分布
2. 逐层残差连接对二值 ViT 至关重要——不仅增强表示能力，更重要的是缓解梯度消失
3. Adam 优化器在二值网络训练后期反而成为阻碍——需要额外正则化应对权重振荡
4. 在分割任务（ADE20K）上也显示了泛化性：mIoU 从 ReActNet 的 9.22 提升到 14.87

## 亮点与洞察

- **三个 Observation 驱动设计**：不是凭空设计架构，而是通过分析（信息熵、梯度传播、优化器行为）发现问题后针对性设计
- **量化分解是核心创新**：用极低额外代价（仅逻辑运算）恢复了注意力矩阵的排序信息
- **正则化损失解决 Adam-二值兼容性问题**：揭示了一个此前被忽视的训练陷阱

## 局限与展望

- 边缘设备的实际加速仍受限于缺乏针对 ViT 特殊模块（窗口注意力、shift 操作）的优化部署工具
- 延迟测试显示 BHViT 二值版 157ms vs 全精度 612ms（ARM），但理论加速倍率未完全实现
- 分割任务上 mIoU 仅 14.87%，与全精度差距仍然巨大
- 未探索更大规模模型（如 Base/Large 配置）

## 相关工作与启发

- **ReActNet** 提供了 RSign/RPReLU 等基础二值化组件
- **BiReaL-Net** 启发了逐层残差连接设计
- **Swin Transformer** 的窗口注意力机制被适配到二值化场景
- **MetaFormer** 的发现（架构比注意力更重要）支持了混合设计的合理性

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 量化分解和 Adam 兼容性分析均有独到见解，混合架构设计有理有据
- **实验充分度**: ⭐⭐⭐⭐ — ImageNet/CIFAR-10/ADE20K 多任务验证 + 详尽消融 + 部署延迟测试
- **写作质量**: ⭐⭐⭐⭐ — 三个 Observation 结构清晰，理论推导严谨
- **价值**: ⭐⭐⭐⭐ — 推进了二值化 ViT 的实际可用性，对边缘部署有重要意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Binarized Mamba-Transformer for Lightweight Quad Bayer HybridEVS Demosaicing](binarized_mamba-transformer_for_lightweight_quad_bayer_hybridevs_demosaicing.md)
- [\[ICCV 2025\] EA-ViT: Efficient Adaptation for Elastic Vision Transformer](../../ICCV2025/model_compression/ea-vit_efficient_adaptation_for_elastic_vision_transformer.md)
- [\[ICML 2026\] QHyer: Q-conditioned Hybrid Attention-mamba Transformer for Offline Goal-conditioned RL](../../ICML2026/model_compression/qhyer_q-conditioned_hybrid_attention-mamba_transformer_for_offline_goal-conditio.md)
- [\[ICCV 2025\] Efficient Adaptation of Pre-Trained Vision Transformer Underpinned by Approximation Theory](../../ICCV2025/model_compression/efficient_adaptation_of_pre-trained_vision_transformer_underpinned_by_approximat.md)
- [\[CVPR 2025\] TADFormer: Task-Adaptive Dynamic Transformer for Efficient Multi-Task Learning](tadformer_task-adaptive_dynamic_transformer_for_efficient_multi-task_learning.md)

</div>

<!-- RELATED:END -->
