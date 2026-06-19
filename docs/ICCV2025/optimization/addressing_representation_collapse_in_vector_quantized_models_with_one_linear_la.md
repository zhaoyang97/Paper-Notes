---
title: >-
  [论文解读] Addressing Representation Collapse in Vector Quantized Models with One Linear Layer
description: >-
  [ICCV 2025][优化/理论][向量量化] 提出SimVQ方法，通过一个可学习的线性变换层对码本向量进行重参数化（$\bm{C}\bm{W}$），将码本的不相交优化转化为联合空间优化，从根本上解决VQ模型中的表示崩塌问题，实现接近100%的码本利用率。 向量量化（VQ）是离散化连续表示的基础技术…
tags:
  - "ICCV 2025"
  - "优化/理论"
  - "向量量化"
  - "表示崩塌"
  - "码本利用率"
  - "线性变换"
  - "多模态"
---

# Addressing Representation Collapse in Vector Quantized Models with One Linear Layer

**会议**: ICCV 2025  
**arXiv**: [2411.02038](https://arxiv.org/abs/2411.02038)  
**代码**: [https://github.com/youngsheen/SimVQ](https://github.com/youngsheen/SimVQ)  
**领域**: 优化  
**关键词**: 向量量化, 表示崩塌, 码本利用率, 线性变换, 多模态

## 一句话总结

提出SimVQ方法，通过一个可学习的线性变换层对码本向量进行重参数化（$\bm{C}\bm{W}$），将码本的不相交优化转化为联合空间优化，从根本上解决VQ模型中的表示崩塌问题，实现接近100%的码本利用率。

## 研究背景与动机

向量量化（VQ）是离散化连续表示的基础技术，广泛用于图像生成（VQGAN）和音频合成（EnCodec）。然而VQ模型存在严重的"表示崩塌"问题：随着码本规模增大，大部分码向量变成"死码"，从未被选中和更新，导致码本利用率极低（如VQGAN在65k码本上仅1.4%利用率）。

这一问题直接制约了VQ模型作为多模态tokenizer与大语言模型（LLM）结合的潜力——例如Chameleon模型的码本大小被限制在8k，远低于LLM词汇表的128k。

现有解决方案的局限：
- 复杂优化策略（stochastic quantization、codebook reset）：工程复杂
- 降低潜空间维度（FSQ、LFQ、VQGAN-FC）：虽提升利用率但牺牲模型容量（维度从128降至8）
- VQGAN-LC-CLIP：依赖外部预训练模型，泛化受限且性能存在天花板

## 方法详解

### 整体框架

SimVQ的核心修改极其简洁：在标准VQ的码本$\bm{C} \in \mathbb{R}^{K \times d}$后接一个可学习的线性层$\bm{W} \in \mathbb{R}^{d \times d}$，实际使用的码本变为$\bm{CW}$。训练时冻结$\bm{C}$（高斯随机初始化），只优化$\bm{W}$。

### 关键设计

1. **不相交优化的理论分析（Root Cause Analysis）**: 通过分析VQ的commitment loss梯度 $\bm{C}^{(t+1)} = \bm{C}^{(t)} - \eta\mathbb{E}[\delta_k^T\delta_k\bm{C}^{(t)}] + \eta\mathbb{E}[\delta_k^T z_e]$，揭示了崩塌的根本原因：$\delta_k^T\delta_k$是Kronecker delta矩阵，仅第$k$行第$k$列为1，因此只有被选中的码向量通过梯度下降更新。理想情况下$\mathbb{E}[\delta_k^T\delta_k]$应收敛到单位矩阵，但VQ的最近邻选择机制导致只有一小部分码被反复选中和更新，其余码"冻结"——形成类似"蚕茧效应"的恶性循环。

2. **线性重参数化与非对称优化（Asymmetric Optimization）**: 将码本重参数化为$\bm{CW}$后，commitment loss变为 $\mathcal{L}_{commit} = \|z_e - \delta_k\bm{CW}\|_2^2$。$\bm{W}$的更新方程为 $\bm{W}^{(t+1)} = (\bm{I} - \eta\mathbb{E}[\bm{C}^T\delta_k^T\delta_k\bm{C}])\bm{W}^{(t)} + \eta\mathbb{E}[\bm{C}^T\delta_k^T z_e]$。由于$\bm{C}$从高斯分布采样且被冻结，$\mathbb{E}[\bm{q}_k^T\bm{q}_k] = \bm{I}$，因此$\bm{W}$的所有元素被均匀更新。优化$\bm{W}$时，整个码本$\bm{CW}$通过空间的旋转和拉伸被联合更新。关键洞察：虽然两个线性矩阵相乘等价于单个线性层，但VQ中的非对称优化动态使这种看似冗余的分解变得至关重要——必须冻结$\bm{C}$、只优化$\bm{W}$，否则$\bm{C}$会主导优化导致崩塌重现。

3. **效率优势**: 标准VQ的码本优化内存复杂度为$O(Kd)$（K为码本大小），SimVQ仅需$O(d^2)$，因为$\bm{C}$被冻结。当$K \gg d$时（如$K=65536, d=128$），内存大幅节省，且独立于词汇表大小。

### 损失函数 / 训练策略

训练目标与标准VQ-VAE相同：$\mathcal{L} = \text{MSE}(x, \hat{x}) + \beta\|z_e - \text{sg}(q_k\bm{W})\|_2^2 + \|\text{sg}(z_e) - q_k\bm{W}\|_2^2$。码本$\bm{C}$用高斯分布初始化并冻结，只有编码器$f_\theta$、解码器$g_\phi$和线性层$\bm{W}_\psi$参与梯度优化。

## 实验关键数据

### 主实验

| 方法 | 潜空间维度 | 码本大小 | 利用率↑ | rFID↓ | LPIPS↓ | PSNR↑ | SSIM↑ |
|------|----------|---------|--------|-------|--------|-------|-------|
| VQGAN | 128 | 65,536 | 1.4% | 3.74 | 0.17 | 22.20 | 70.6 |
| VQGAN-FC | 8 | 65,536 | 100% | 2.63 | 0.13 | 23.79 | 77.5 |
| LFQ | 16 | 65,536 | 100% | 2.88 | 0.13 | 23.60 | 77.2 |
| VQGAN-LC-CLIP | 768 | 65,536 | 100% | 2.40 | 0.13 | 23.98 | 77.3 |
| **SimVQ** | **128** | **65,536** | **100%** | **2.24** | **0.12** | **24.15** | **78.4** |
| **SimVQ** | **128** | **262,144** | **100%** | **1.99** | **0.11** | **24.68** | **80.3** |

音频模态（LibriTTS，0.9kbps带宽）：SimVQ的UTMOS达4.00/3.51，优于WavTokenizer的3.74/3.43，同时保持100%码本利用率。

### 消融实验

| 配置 | 利用率 | rFID | 说明 |
|------|-------|------|------|
| SimVQ码本1k | 100% | 3.67 | 小码本也不崩塌 |
| SimVQ码本8k | 100% | 2.98 | 码本增大持续提升 |
| SimVQ码本65k | 100% | 2.24 | SOTA |
| SimVQ码本262k | 100% | 1.99 | 继续扩展仍有收益 |
| VQGAN-LC 100k | 99.9% | 2.62 | |
| VQGAN-LC 200k | 99.8% | 2.66 | 性能饱和反转 |
| $\bm{C}$高斯初始化+冻结 | 100% | 2.24 | 默认设置 |
| $\bm{C}$均匀初始化+冻结 | 100% | 2.31 | 初始化不敏感 |
| $\bm{C}$高斯初始化+可训练 | 100% | 2.31 | 性能轻微下降 |

### 关键发现

- 码本规模从1k扩展到262k时，SimVQ始终保持100%利用率且性能持续提升（rFID从3.67降至1.99），而VQGAN-LC在超过100k后性能饱和
- 同时训练$\bm{C}$和$\bm{W}$会导致$\bm{C}$主导优化，$\bm{W}$几乎不变，崩塌重现——这通过二维toy实验可视化地展示
- 线性基矩阵$\bm{W}$的秩在训练中自适应降低，大码本对应更低的秩收敛值，说明大码本缓解了潜空间的维度压力
- SimVQ在图像和音频两种模态上效果一致，证明了方法的通用性

## 亮点与洞察

- 解决方案极其简洁（一个线性层），但背后的理论分析深入且令人信服
- "两个线性矩阵相乘虽等价于单线性层，但非对称优化动态改变了一切"——这一洞察出人意料
- 二维toy实验的可视化非常直观，清楚地展示了不同优化模式下码本的行为差异
- 完全不依赖外部预训练模型，无域限制，真正的即插即用

## 局限与展望

- VQ重建质量的提升不一定直接转化为下游生成模型的性能提升（作者在Discussion中诚实地讨论了这一点）
- 理论分析基于码本从高斯分布采样的假设，实际中的分布可能有偏差
- 线性变换可能无法充分捕获非线性的潜空间结构，可考虑轻量级非线性拓展

## 相关工作与启发

- 与VQ-VAE/VQGAN系列的经典问题（表示崩塌）对话，方法定位精准
- 码本扩展到262k的能力直接对接LLM tokenizer的需求，为统一多模态tokenizer提供了可能
- "冻结+重参数化"的思路可推广到其他涉及离散选择的优化问题中

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 极简方法解决根本问题，理论洞察深刻
- 实验充分度: ⭐⭐⭐⭐ 图像+音频双模态，大规模码本测试，完整消融
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，toy实验可视化精彩
- 价值: ⭐⭐⭐⭐⭐ 即插即用，对VQ生态系统有重大意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Layer-wise Quantization for Quantized Optimistic Dual Averaging](../../ICML2025/optimization/layer-wise_quantization_for_quantized_optimistic_dual_averaging.md)
- [\[ICLR 2026\] Constraint Matters: Multi-Modal Representation for Reducing Mixed-Integer Linear programming](../../ICLR2026/optimization/constraint_matters_multi-modal_representation_for_reducing_mixed-integer_linear_.md)
- [\[ICLR 2026\] Gradient-Sign Masking for Task Vector Transport Across Pre-Trained Models](../../ICLR2026/optimization/gradient-sign_masking_for_task_vector_transport_across_pre-trained_models.md)
- [\[AAAI 2026\] On the Learning Dynamics of Two-Layer Linear Networks with Label Noise SGD](../../AAAI2026/optimization/on_the_learning_dynamics_of_two-layer_linear_networks_with_label_noise_sgd.md)
- [\[ICML 2025\] Random Feature Representation Boosting](../../ICML2025/optimization/random_feature_representation_boosting.md)

</div>

<!-- RELATED:END -->
