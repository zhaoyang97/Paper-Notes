---
title: >-
  [论文解读] Alias-Free ViT: Fractional Shift Invariance via Linear Attention
description: >-
  [NeurIPS 2025][平移不变性] 提出Alias-Free Vision Transformer（AFT），结合抗混叠信号处理技术和shift-equivariant线性交叉协方差注意力，首次使ViT在分数像素（亚像素）平移下保持接近完美的一致性（~99%），同时在ImageNet分类准确率上几乎无损。
tags:
  - "NeurIPS 2025"
  - "平移不变性"
  - "抗混叠"
  - "线性注意力"
  - "亚像素偏移"
  - "shift equivariance"
---

# Alias-Free ViT: Fractional Shift Invariance via Linear Attention

**会议**: NeurIPS 2025  
**arXiv**: [2510.22673](https://arxiv.org/abs/2510.22673)  
**代码**: [https://github.com/hmichaeli/alias_free_vit](https://github.com/hmichaeli/alias_free_vit)  
**领域**: 视觉Transformer / 鲁棒性  
**关键词**: 平移不变性, 抗混叠, 线性注意力, 亚像素偏移, shift equivariance

## 一句话总结

提出Alias-Free Vision Transformer（AFT），结合抗混叠信号处理技术和shift-equivariant线性交叉协方差注意力，首次使ViT在分数像素（亚像素）平移下保持接近完美的一致性（~99%），同时在ImageNet分类准确率上几乎无损。

## 研究背景与动机

ViT已成为视觉任务的主流架构，但缺乏CNN的平移不变性归纳偏置，对微小图像平移高度敏感。对于CNN，已有研究表明混叠（aliasing）是破坏平移不变性的根本原因——下采样和非线性层中的混叠导致信号失真。已有的抗混叠方法如APS（Adaptive Polyphase Sampling）虽然能保证整数像素循环平移的一致性，但对亚像素（fractional）平移和实际场景中的平移（如相机移动导致的crop-shift）无效。

问题核心在于：标准的softmax自注意力不是shift-equivariant的。平移信号后，softmax中的非线性归一化会改变token之间的相对权重分配，导致输出不再等变。作者的关键洞察是：如果移除softmax，采用线性注意力或交叉协方差注意力（XCA），则注意力操作变为shift-equivariant的。结合抗混叠的下采样和非线性层处理，就能构建一个对亚像素平移也鲁棒的ViT。

## 方法详解

### 整体框架

以XCiT架构为基础，系统替换每个非shift-equivariant的组件：抗混叠patch embedding → Alias-Free Transformer Block（抗混叠LayerNorm + XCA + 抗混叠LPI + 抗混叠MLP）→ Alias-Free Class Attention。整体架构保持与XCiT相同的参数量。

### 关键设计

1. **Shift-Equivariant Attention（SEA）理论**:
    - 功能：证明一类线性注意力操作天然具有shift-equivariance
    - 核心思路：定义 $\text{SEA}(X) = Q f(K^\top V)$，证明三步：(i) Q/K/V是shift-equivariant的（每列是输入信号通道的线性组合），(ii) $K^\top V$ 是shift-invariant的（Parseval定理，频域中相移相消），(iii) 乘积 $Q \cdot f(K^\top V)$ 是shift-equivariant的
    - 设计动机：标准softmax注意力因按行归一化而破坏等变性；XCA中的 $K^\top Q$ 也是shift-invariant的，因此对其施加softmax不影响等变性

2. **抗混叠Patch Embedding**:
    - 功能：消除tokenization过程中的混叠
    - 核心思路：将原始的stride-p卷积替换为stride-1卷积 + 抗混叠下采样（FFT域频率截断）。采用渐进式多层卷积（而非单次大stride），每层只做小步长下采样配合更高的频率截止，保留高频特征
    - 设计动机：单次stride-p会需要cutoff 1/p的低通滤波，严重衰减高频信息

3. **抗混叠非线性层和LayerNorm**:
    - 功能：消除GELU等非线性函数和LayerNorm引入的混叠
    - 核心思路：GELU前先上采样，GELU后低通滤波再下采样回来；LayerNorm改为全局变体——均值per-token但标准差per-layer计算，避免逐token不同的缩放破坏等变性
    - 设计动机：点式非线性在频域会产生新的频率成分（谐波），导致混叠

### 损失函数 / 训练策略

- 训练策略与XCiT完全相同（400 epochs, ImageNet），不需要额外的平移增强
- 去除位置编码（实验发现在有卷积层的架构中位置编码不必要，去除后准确率反而略升）
- 用Class Attention替代全局平均池化获取分类表示（性能更好）

## 实验关键数据

### 主实验

| 模型 | Top-1 Acc | 整数shift一致性 | 半像素shift一致性 | 对抗整数grid | 对抗半像素grid |
|------|----------|---------------|-----------------|------------|-------------|
| XCiT-Nano (Baseline) | 70.4 | 83.7 | 82.0 | 50.9 | 52.9 |
| XCiT-Nano-APS | 68.4 | **100.0** | 87.5 | 68.4 | 62.9 |
| XCiT-Nano-AF (ours) | **70.5** | 99.2 | **98.7** | **69.9** | **69.5** |
| XCiT-Small (Baseline) | 82.0 | 91.4 | 89.8 | 70.9 | 71.3 |
| XCiT-Small-APS | 81.3 | **100.0** | 94.0 | 81.3 | 78.2 |
| XCiT-Small-AF (ours) | **81.8** | 99.5 | **99.4** | **81.3** | **81.1** |

### 消融实验

| 修改 | Accuracy | 变化 |
|------|----------|------|
| Baseline XCiT-Nano | 70.4 | - |
| + 循环卷积 | 70.4 | +0.0% |
| + 全局平均池化替代Class Attn | 69.1 | -1.8% |
| + AF-LayerNorm | 69.6 | -1.1% |
| - 位置编码 | 70.7 | **+0.4%** |
| 完整AF (AF Class Attention) | 70.6 | +0.3% |

### 关键发现

- 准确率几乎无损甚至微升，但平移一致性从~83%提升到~99%
- APS方法在整数平移上完美，但亚像素和实际平移上明显不如AFT
- 对抗平移攻击下，AFT的准确率衰减极小（Nano版仅2%相对衰减 vs baseline 25%）
- 在crop-shift和bilinear fractional shift等真实场景平移下，AFT一致优于所有对比模型（含CvT、Swin、vanilla ViT）
- 去除位置编码不仅不降反升——说明在有卷积层的hybrid架构中显式位置编码可能是多余的

## 亮点与洞察

- **线性注意力天然shift-equivariant**这个观察非常优雅——Parseval定理证明 $K^\top V$ shift-invariant，因此整个线性注意力 $Q f(K^\top V)$ 自动是shift-equivariant的。这给了选择attention变体一个全新的理论依据
- 将信号处理的抗混叠理论与Transformer架构系统结合，方法论上很完整
- 位置编码在hybrid架构中可能多余的发现值得注意
- 对于视频生成、神经算子等需要平移一致性的领域有直接应用价值

## 局限与展望

- 运行时间开销显著：训练时间从69小时增至487小时（7×），主要源于FFT域的上下采样操作GPU优化不足
- 未使用多项式激活函数替代GELU（可达理论完美shift-invariance，但实验发现准确率严重下降）
- 仅在图像分类上验证，检测/分割/生成等下游任务未测试
- 线性注意力的表达能力是否在大规模数据和复杂任务上受限仍不确定

## 相关工作与启发

- **vs APS**: APS保证整数循环平移一致性（100%），但对亚像素平移和实际平移提升有限；AFT牺牲极小的整数一致性（99.5% vs 100%）换取大幅亚像素鲁棒性
- **vs Qian et al.**: 仅在注意力后加低通滤波是不完整方案，不解决注意力本身的非等变性
- **vs AFC (Alias-Free Convnet)**: AFT扩展了AFC的思路到Transformer，核心新贡献是SEA理论
- **启发**：shift-invariance应在架构设计阶段就作为显式约束，而非事后修补

## 评分

- 新颖性: ⭐⭐⭐⭐ 信号处理+Transformer的系统性结合，SEA理论证明优雅
- 实验充分度: ⭐⭐⭐⭐ 多种shift类型（循环/crop/bilinear）+多模型对比+消融
- 写作质量: ⭐⭐⭐⭐⭐ 理论证明简洁清晰，架构图直观
- 价值: ⭐⭐⭐⭐ 对需要平移鲁棒性的应用有直接价值，但运行时开销限制了通用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Test-Time Training with KV Binding Is Secretly Linear Attention](../../ICML2026/others/test-time_training_with_kv_binding_is_secretly_linear_attention.md)
- [\[NeurIPS 2025\] Normalization in Attention Dynamics](normalization_in_attention_dynamics.md)
- [\[NeurIPS 2025\] Modeling Neural Activity with Conditionally Linear Dynamical Systems](modeling_neural_activity_with_conditionally_linear_dynamical_systems.md)
- [\[NeurIPS 2025\] Position: There Is No Free Bayesian Uncertainty Quantification](position_there_is_no_free_bayesian_uncertainty_quantification.md)
- [\[ICML 2026\] Functional Equivalence in Attention: A Comprehensive Study with Applications to Linear Mode Connectivity](../../ICML2026/others/functional_equivalence_in_attention_a_comprehensive_study_with_applications_to_l.md)

</div>

<!-- RELATED:END -->
