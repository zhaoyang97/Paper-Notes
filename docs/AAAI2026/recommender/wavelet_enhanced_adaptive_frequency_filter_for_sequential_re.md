---
title: >-
  [论文解读] Wavelet Enhanced Adaptive Frequency Filter for Sequential Recommendation
description: >-
  [AAAI2026][推荐系统][sequential recommendation] 提出WEARec模型，通过动态频域滤波（DFF）根据用户上下文自适应调整频域滤波器捕获个性化全局偏好，并用小波特征增强（WFE）弥补全局DFT模糊短期波动的缺陷，在四个数据集上超越全部9个基线，长序列场景最高提升11.4%且训练速度快39-45%。
tags:
  - "AAAI2026"
  - "推荐系统"
  - "sequential recommendation"
  - "frequency-domain filtering"
  - "wavelet transform"
  - "dynamic filter"
  - "personalized recommendation"
---

# Wavelet Enhanced Adaptive Frequency Filter for Sequential Recommendation

**会议**: AAAI2026  
**arXiv**: [2511.07028](https://arxiv.org/abs/2511.07028)  
**代码**: [GitHub](https://github.com/xhy963319431/WEARec)  
**领域**: 序列推荐 / 频域信号处理  
**关键词**: sequential recommendation, frequency-domain filtering, wavelet transform, dynamic filter, personalized recommendation

## 一句话总结

提出WEARec模型，通过动态频域滤波（DFF）根据用户上下文自适应调整频域滤波器捕获个性化全局偏好，并用小波特征增强（WFE）弥补全局DFT模糊短期波动的缺陷，在四个数据集上超越全部9个基线，长序列场景最高提升11.4%且训练速度快39-45%。

## 研究背景与动机

**领域现状**：序列推荐通过分析用户历史交互捕获动态偏好。频域方法（FMLPRec、BSARec、SLIME4Rec等）利用傅里叶变换将用户行为序列分解为频率成分，有效捕获时域中难以识别的周期模式，逐渐成为替代self-attention的高效范式。

**痛点一——静态滤波忽略个性化**：现有频域方法使用固定模式的静态滤波器统一处理所有用户序列。但实验发现不同用户被完全不同的频率成分驱动——有些用户行为由低频长期偏好主导，另一些则由高频短期兴趣驱动（论文通过统计各频率成分独立驱动的用户数量证实了这一点）。

**痛点二——全局DFT模糊短期波动**：DFT是全局频率分析工具，擅长捕获长程依赖但会模糊非平稳信号和短期兴趣变化。FMLPRec本质上是低通滤波器，SLIME4Rec虽然尝试分层学习不同频段但仍偏向低频。

**核心idea**：用MLP根据用户序列上下文动态生成个性化滤波器参数（解决痛点一），同时引入小波变换增强被DFT模糊的高频局部特征（解决痛点二），两者融合实现全局+局部的全频段覆盖。

## 方法详解

### 整体框架

类Transformer编码器架构：Embedding层（item embedding + positional embedding + LayerNorm + Dropout）→ $L=2$ 层WEARec Block（每层包含DFF模块 + WFE模块 + Feature Integration + FFN + 残差连接）→ Prediction层（softmax输出候选item概率）。DFF和WFE共同替代传统的self-attention模块。

### 关键设计

1. **动态频域滤波模块（DFF）**:
    - 功能：根据每个用户的行为序列上下文，自适应生成个性化的频域滤波器参数，实现用户级别的频率选择
    - 核心思路：先将嵌入沿维度拆分为 $k$ 个子空间（多头投影），对每个子空间做1D FFT得到频域表示 $\mathbb{F}_i^l \in \mathbb{C}^{M \times d/k}$。同时在时域计算用户上下文均值 $\mathbf{c}^l = \frac{1}{N}\sum_{i=1}^{N}\mathbb{H}_i^l$，通过两个MLP分别生成缩放因子 $\Delta\mathbf{s}^l$ 和偏置 $\Delta\mathbf{b}^l$。用它们调制基础滤波器：$\hat{\mathbb{W}}^l = \mathbb{W}^l \odot (1 + \Delta\mathbf{s}^l)$，$\hat{\mathbf{b}}^l = \mathbf{b}^l + \Delta\mathbf{b}^l$。最后频域线性变换 $\tilde{\mathbb{F}}_i^l = \mathbb{F}_i^l \odot \hat{\mathbb{W}}^l + \hat{\mathbf{b}}^l$ 后IFFT回时域
    - 设计动机：缩放因子 $\Delta\mathbf{s}$ 控制滤波器的整体频率响应形状，偏置 $\Delta\mathbf{b}$ 针对特定频段做微调，两者结合实现"基于用户上下文的条件滤波"。频谱可视化证实WEARec能覆盖全频段，而FMLPRec和SLIME4Rec偏向低频

2. **小波特征增强模块（WFE）**:
    - 功能：通过离散小波变换（DWT）分解序列的高频和低频成分，用可学习矩阵自适应增强高频细节信号
    - 核心思路：对每个子空间沿item维度做Haar小波分解 $\mathbb{A}_i^l, \mathbb{D}_i^l = \mathcal{W}(\mathbb{B}_i^l)$，其中 $\mathbb{A}$ 为低频近似系数、$\mathbb{D}$ 为高频细节系数。低频不修改（保留序列主要趋势），高频用可学习矩阵增强 $\tilde{\mathbb{D}}_i^l = \mathbb{D}_i^l \odot \mathbb{T}^l$，最后IDWT重构 $\mathbb{Y}_i^l = \mathcal{W}^{-1}(\mathbb{A}_i^l, \tilde{\mathbb{D}}_i^l)$
    - 设计动机：小波变换具有时频局部化特性（DFT不具备），能精准定位短期非平稳事件。选择Haar小波因其结构最简单、计算高效且支持完美重构。只增强高频而不修改低频，避免破坏序列的主要趋势信息

3. **特征融合与预测**:
    - 功能：将DFF提取的全局频域特征与WFE增强的局部时频特征按权重融合，输出最终推荐
    - 核心思路：加权融合 $\hat{\mathbb{H}}^l = \alpha \odot \mathbb{X}^l + (1-\alpha) \odot \mathbb{Y}^l$（$\alpha \approx 0.3$ 最优），后接残差连接、LayerNorm、FFN（GELU激活）。预测层 $\hat{\mathbf{y}} = \text{softmax}(\mathbf{h}^L (\mathbb{M})^\top)$
    - 设计动机：DFF捕获全局个性化频域分布，WFE增强局部非平稳细节，两者互补。$\alpha < 0.5$ 表明局部高频增强应占较大权重，因为频域方法的主要短板恰恰在局部特征

### 损失函数 / 训练策略

标准交叉熵损失 $\mathcal{L}_{Rec} = -\sum_{i=1}^{|\mathcal{V}|} y_i \log(\hat{y}_i)$。Adam优化器，学习率 $\in \{0.0005, 0.001\}$，embedding维度64，最大序列长度 $N=50$（标准）/ $N=200$（长序列），batch size 256。小波分解层数1，$k \in \{1,2,4,8\}$。不使用对比学习和self-attention，显著降低计算开销。

## 实验关键数据

### 主实验

| 数据集 | 指标 | WEARec | BSARec | SLIME4Rec | DuoRec | 提升(vs最优基线) |
|--------|------|--------|--------|-----------|--------|--------------|
| Beauty | HR@10 | **0.1041** | 0.1008 | 0.1006 | 0.0965 | +3.27% |
| Sports | HR@10 | **0.0631** | 0.0612 | 0.0611 | 0.0569 | +3.10% |
| LastFM | HR@10 | **0.0899** | 0.0807 | 0.0633 | 0.0624 | +11.40% |
| ML-1M | HR@10 | **0.2952** | 0.2757 | 0.2891 | 0.2704 | +2.10% |
| ML-1M | NDCG@10 | **0.1696** | 0.1568 | 0.1673 | 0.1530 | +1.37% |

### 消融实验

| 配置 | Beauty HR@20 | Sports HR@20 | LastFM HR@20 | ML-1M HR@20 | 说明 |
|------|-------------|-------------|-------------|-------------|------|
| WEARec（完整） | **0.1391** | **0.0895** | **0.1202** | **0.4031** | 最优 |
| w/o WFE | 下降 | 下降 | 下降 | 下降 | 去掉小波增强，损失局部信息 |
| w/o DFF | 下降最大 | 下降最大 | 下降最大 | 下降最大 | 去掉动态滤波，DFF是核心 |
| w/o 多头投影 | 下降 | 下降 | 下降 | 下降 | 去掉子空间划分 |

长序列效率（N=200）：WEARec 66.46s/epoch vs BSARec 109.26s vs SLIME4Rec 120.43s，快39-45%。

### 关键发现

- DFF是核心模块，去除后性能下降最大——个性化频域滤波比静态滤波重要
- LastFM数据集提升最大（+11.4%），因其平均序列长度48.2远长于Beauty/Sports（~8），长序列场景WEARec优势更显著
- $\alpha \approx 0.3$ 最优，即70%的权重给WFE局部增强、30%给DFF全局滤波
- 频谱可视化显示WEARec覆盖全频段，而FMLPRec和SLIME4Rec均偏向低频
- WEARec参数量（426K/ML-1M）高于FMLPRec（324K）但低于BSARec和SLIME4Rec的训练时间

## 亮点与洞察

- 动态滤波器的设计优雅简洁——用用户序列均值作为上下文，通过两个小MLP生成缩放/偏置来调制基础滤波器，参数开销极低但实现了真正的个性化频域处理。
- 傅里叶（全局频率分析）+小波（时频局部化）的互补组合在信号处理中有理论基础，本文是将这一经典思路成功引入推荐系统的首次尝试。
- 完全不使用self-attention和对比学习，靠纯频域/时频分析实现SOTA，且训练速度反而更快，说明频域方法在序列推荐中有巨大潜力。

## 局限与展望

- 仅使用Haar小波（最简单的小波基），复杂小波基（Daubechies、Symlet等）可能在捕获更精细时频特征上更优
- $\alpha$ 为全局固定超参数，可改为样本自适应或层自适应的可学习权重
- 未考虑item的内容特征（文本、图像等side information），仅基于交互序列建模
- Beauty/Sports上提升幅度较小（3%），这些短序列数据集的个性化和局部增强优势不如长序列明显

## 相关工作与启发

- **vs FMLPRec**：FMLPRec本质是低通滤波器，WEARec的DFF覆盖全频段；FMLPRec是静态滤波，WEARec是动态个性化滤波
- **vs BSARec**：BSARec用频域分量作为self-attention的归纳偏置，WEARec完全替代self-attention，训练速度更快
- **vs SLIME4Rec**：SLIME4Rec分层处理不同频段但仍偏低频，WEARec的WFE模块显式增强高频信号

## 评分

- 新颖性: ⭐⭐⭐⭐ 动态频域滤波+小波增强的组合设计新颖，信号处理思路清晰
- 实验充分度: ⭐⭐⭐⭐ 四数据集、长序列分析、频谱可视化、超参敏感性分析齐全
- 写作质量: ⭐⭐⭐⭐ 信号处理基础部分特别清晰，方法推导完整
- 价值: ⭐⭐⭐⭐ 为频域推荐提供了高效且有效的新范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] FreqRec: Exploiting Inter-Session Information with Frequency-enhanced Dual-Path Networks for Sequential Recommendation](exploiting_inter-session_information_with_frequency-enhanced_dual-path_networks_.md)
- [\[NeurIPS 2025\] TV-Rec: Time-Variant Convolutional Filter for Sequential Recommendation](../../NeurIPS2025/recommender/tv-rec_time-variant_convolutional_filter_for_sequential_recommendation.md)
- [\[AAAI 2026\] HyMoERec: Hybrid Mixture-of-Experts for Sequential Recommendation](hymoerec_hybrid_mixture-of-experts_for_sequential_recommendation.md)
- [\[AAAI 2026\] From IDs to Semantics: A Generative Framework for Cross-Domain Recommendation with Adaptive Semantic Tokenization](from_ids_to_semantics_a_generative_framework_for_cross-domain_recommendation_wit.md)
- [\[ICLR 2026\] CollectiveKV: Decoupling and Sharing Collaborative Information in Sequential Recommendation](../../ICLR2026/recommender/collectivekv_decoupling_and_sharing_collaborative_information_in_sequential_reco.md)

</div>

<!-- RELATED:END -->
