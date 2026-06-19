---
title: >-
  [论文解读] Beyond the Mean: Fisher-Orthogonal Projection for Natural Gradient Descent in Large Batch Training
description: >-
  [AAAI2026][优化/理论][Natural Gradient Descent] 提出 Fisher-Orthogonal Projection (FOP)，通过在 Fisher 度量下对子批次梯度差做正交投影来补充方差信息，使二阶优化器 KFAC 在超大 batch 训练中保持有效，实现最高 ×7.5 的加速。
tags:
  - "AAAI2026"
  - "优化/理论"
  - "Natural Gradient Descent"
  - "Fisher Information Matrix"
  - "KFAC"
  - "Large Batch Training"
  - "Second-Order Optimization"
---

# Beyond the Mean: Fisher-Orthogonal Projection for Natural Gradient Descent in Large Batch Training

**会议**: AAAI2026  
**arXiv**: [2508.13898](https://arxiv.org/abs/2508.13898)  
**代码**: [yishunlu-222/fop](https://github.com/yishunlu-222/fop)  
**领域**: 优化  
**关键词**: Natural Gradient Descent, Fisher Information Matrix, KFAC, Large Batch Training, Second-Order Optimization  

## 一句话总结
提出 Fisher-Orthogonal Projection (FOP)，通过在 Fisher 度量下对子批次梯度差做正交投影来补充方差信息，使二阶优化器 KFAC 在超大 batch 训练中保持有效，实现最高 ×7.5 的加速。

## 背景与动机
- 现代 GPU 拥有巨大显存（如 AMD MI300X 的 192GB），支持数万级 batch size 的训练，但大多数优化器在超大 batch 下表现不佳
- **一阶方法的困境**：batch 增大后梯度噪声减小，SGD/Adam/AdamW 失去了帮助逃离尖锐极小值的随机噪声，导致泛化变差
- **二阶方法的困境**：KFAC 等自然梯度法在大 batch 下 Fisher 矩阵严重病态，要求极高的 damping 才能稳定，但高 damping 又抹掉了曲率信息，退化为普通梯度下降
- 已有改进（SENG、SP-NGD 等）引入额外超参数、依赖过时统计量或任务特定调参，缺乏通用性

## 核心问题
在超大 batch 训练中，如何在不引入额外超参数的前提下恢复自然梯度法的曲率优势，同时保持小 batch 时的泛化精度？

## 方法详解

### 核心思想
简单取平均的梯度会丢失子批次之间有价值的方向信息。FOP 的核心是：将 mini-batch 拆为两个子批次，计算各自梯度 $g_1, g_2$，既保留平均梯度 $g_{\text{avg}}$ 作为主要下降方向，又提取梯度差 $g_{\text{diff}}$ 中与均值在 Fisher 度量下正交的分量，作为补充更新方向。

### 具体步骤

**1. 双子批次梯度计算**：将 mini-batch 分为两半，分别计算梯度：

$$g_{\text{avg}} = \frac{1}{2}(g_1 + g_2), \quad g_{\text{diff}} = g_1 - g_2$$

**2. Fisher-正交投影**：在 Fisher 内积下，将 $g_{\text{diff}}$ 对 $g_{\text{avg}}$ 做正交化，去除冗余信息：

$$s_{\text{proj}} = \frac{g_{\text{diff}}^\top F g_{\text{avg}}}{g_{\text{avg}}^\top F g_{\text{avg}} + \epsilon}, \quad g_{\text{diff}}^{\perp} = g_{\text{diff}} - s_{\text{proj}} \cdot g_{\text{avg}}$$

保证 $\langle g_{\text{avg}}, g_{\text{diff}}^{\perp} \rangle_F = 0$。

**3. 自适应混合系数 $\beta^*$**：对总损失做二阶 Taylor 展开，求解使代理目标最小的混合比例（逐层计算）：

$$\beta^* = \frac{g_{\text{avg}}^\top F^{-1} g_{\text{diff}}^{\perp}}{(g_{\text{diff}}^{\perp})^\top F^{-1} g_{\text{diff}}^{\perp}}$$

当正交分量方向错误时 $\beta^* \to 0$，FOP 自动退化为标准 KFAC，保证安全性。

**4. 逐层自适应步长 $\eta_\ell^*$**：为每一层单独计算最优步长，根据曲率和梯度对齐程度自动调整：

$$\eta_\ell^* = \frac{g_{\ell,\text{tot}}^\top F_\ell^{-1} g_{\ell,\text{comb}}}{g_{\ell,\text{comb}}^\top F_\ell^{-1} g_{\ell,\text{comb}}}$$

**5. 最终更新**：$d_\ell = \eta_0 \eta_\ell^* F_\ell^{-1} g_{\ell,\text{comb}}$

### KL 范数分析
- FOP 更新的 KL 范数分解为：基础项（$\mathcal{O}(1/\lambda^2)$）+ 交叉项与正交项（$\mathcal{O}(1/\lambda)$）
- 训练早期 $\beta < 0$ 时，交叉项为负，与基础项部分抵消，形成安全余量，允许降低 damping $\lambda$
- 这解释了 FOP 在大 batch 下不需要高 damping 仍能保持稳定的理论机制

### 分布式 FOP
- 将 GPU 分为 primary/secondary 两组，各组分别做 AllReduce 得到 $g_1, g_2$
- 每层指定一个 GPU 为 "曲率专家"，负责更新和求逆该层的 Fisher 矩阵
- 专家 GPU 计算 FOP 更新后广播给所有进程
- 通信开销与标准数据并行相当，仅多一次 AllReduce

## 实验关键数据

### CIFAR-10 + ResNet-18（5 seeds）

| Batch Size | SGD | AdamW | KFAC | FOP |
|---|---|---|---|---|
| 2048 | 58ep/743s | 61ep/768s | 37ep/589s | **29ep/475s** |
| 4096 | 73ep/458s | 73ep/454s | 34ep/271s (×1.69) | **22ep/182s (×2.52)** |
| 8192 | — | — | 71ep/296s (×1.54) | **35ep/158s (×2.91)** |
| 16384 | — | — | 99ep/186s (×2.46) | **58ep/121s (×3.78)** |
| 32768 | — | — | — | **60ep/91s (×5.05)** |
| 50000 | — | — | — | **82ep/84s (×5.43)** |

达到 91% 准确率的时间；"—"表示未达阈值。FOP 是唯一在 BS≥32768 仍能收敛的方法。

### ImageNet-100 + T2T-ViT（3 seeds）
- 达到 80.6% Top-1：FOP 在 BS=4096 仅需 49ep/27.8min，相比 AdamW(BS=512) 加速 **×10.48**
- KFAC 在同配置下加速 ×6.45，FOP 持续领先

### ImageNet-1K + ResNet-50（3 seeds）
- 达到 75.9% Top-1：FOP 在 BS=8192 需 40ep/335min，加速 **×7.50**（相对 SGD BS=1024）
- SGD、Shampoo、LAMB、KFAC 在 BS>1024 均未能达阈值

### CIFAR-LT + ResNet-32（长尾分布，5 seeds）
- CIFAR-10-LT (IF=100)：FOP 错误率 26.65%，比 baseline 低 1.4%，比 KFAC 低 1.94%
- CIFAR-100-LT (IF=100)：FOP 错误率 58.97%，比 baseline 低 **3.3%**
- KFAC 在 IF=100 反而比 baseline 更差，FOP 全面优于所有方法

## 亮点
1. **理论优雅**：Fisher 正交投影的设计有几何直觉，KL 范数分析提供了理论解释
2. **自适应安全**：$\beta^*$ 自动退化机制确保在正交分量无益时回退为 KFAC，不会造成损害
3. **极端可扩展**：唯一在 batch=50000（全量 CIFAR-10）仍收敛的优化器
4. **无额外超参**：无需任务特定调参，只需按线性规则缩放学习率
5. **即插即用**：开源 pip 安装，一行代码集成到现有训练流程

## 局限与展望
- 仅在视觉任务（CNN、ViT）上验证，缺乏大规模语言模型的实验
- 二阶 Taylor 展开中 Hessian≈Fisher 的近似在训练早期和高度非线性模型上可能不准确
- 分布式实现需要 GPU 数量为偶数以划分 primary/secondary 组
- 相比一阶方法仍有额外的 Fisher 矩阵计算和求逆开销
- 仅与 KFAC 框架结合，未探索与其他二阶方法（如 Shampoo 的 Kronecker 结构）结合的可能

## 与相关工作的对比

| 方法 | 阶数 | 大 batch 能力 | 额外超参 | ImageNet-1K 75.9% |
|---|---|---|---|---|
| SGD/AdamW | 一阶 | 差（>4096 失效）| 无 | 71ep/2511min (BS=1024) |
| LAMB | 一阶 | 一般 | layer-wise LR | 67ep/2493min (BS=1024) |
| KFAC | 二阶 | 一般（>1024 失效）| damping | 35ep/1337min (BS=1024) |
| SENG | 二阶 | 中等 | 低秩参数 | 41ep (BS=4096) |
| SP-NGD | 二阶 | 中等 | 多个启发式 | 74.8-75.3% |
| **FOP** | **二阶** | **强** | **无** | **40ep/335min (BS=8192)** |

## 启发与关联
- 思路可推广：利用子批次方差信息做正交校正的范式，可能适用于其他优化器（如 Adam 的 Fisher 变体）
- 长尾分布上的优势暗示 FOP 的曲率感知更新对类不平衡场景有天然适配性，值得在医学影像等长尾任务中探索
- 分布式设计中的 primary/secondary 分组策略有趣，可能对联邦学习中的梯度聚合有启发

## 评分
- 新颖性: ⭐⭐⭐⭐ — Fisher 正交投影思路新颖，自适应 β 和逐层步长设计完整
- 实验充分度: ⭐⭐⭐⭐ — 覆盖 CNN/ViT、多数据集、长尾场景，多 seed 重复，但缺 NLP 实验
- 写作质量: ⭐⭐⭐⭐ — 数学推导清晰，实验组织良好
- 价值: ⭐⭐⭐⭐ — 对大 batch 训练有实际意义，即插即用设计降低使用门槛

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Learning to Recall with Transformers Beyond Orthogonal Embeddings](../../ICLR2026/optimization/learning_to_recall_with_transformers_beyond_orthogonal_embeddings.md)
- [\[ICLR 2026\] Πnet: Optimizing Hard-Constrained Neural Networks with Orthogonal Projection Layers](../../ICLR2026/optimization/pinet_optimizing_hard-constrained_neural_networks_with_orthogonal_projection_lay.md)
- [\[NeurIPS 2025\] Natural Gradient Descent for Improving Variational Inference Based Classification of Radio Galaxies](../../NeurIPS2025/optimization/natural_gradient_descent_for_improving_variational_inference_based_classificatio.md)
- [\[ICML 2026\] Flatland: The Adventures of Gradient Descent with Large Step Sizes](../../ICML2026/optimization/flatland_the_adventures_of_gradient_descent_with_large_step_sizes.md)
- [\[CVPR 2025\] SCOPE: Semantic Coreset with Orthogonal Projection Embeddings for Federated Learning](../../CVPR2025/optimization/scope_semantic_coreset_with_orthogonal_projection_embeddings_for_federated_learn.md)

</div>

<!-- RELATED:END -->
