---
title: >-
  [论文解读] Semantically Guided Representation Learning For Action Anticipation
description: >-
  [ECCV2024][时间序列][Action Anticipation] 提出 S-GEAR 框架，通过学习视觉动作原型并利用语言模型的语义关联来引导原型之间的几何关系，使模型理解动作间的语义互联性，从而提升动作预测性能，在 Epic-Kitchens 55/100、EGTEA Gaze+、50 Salads 四个基准上取得 SOTA 或极具竞争力的结果。
tags:
  - "ECCV2024"
  - "时间序列"
  - "Action Anticipation"
  - "Prototype Learning"
  - "语义引导"
  - "视觉-语言原型"
  - "几何关联迁移"
---

# Semantically Guided Representation Learning For Action Anticipation

**会议**: ECCV2024  
**arXiv**: [2407.02309](https://arxiv.org/abs/2407.02309)  
**代码**: [github.com/ADiko1997/S-GEAR](https://github.com/ADiko1997/S-GEAR)  
**领域**: 自监督  
**关键词**: Action Anticipation, Prototype Learning, 语义引导, 视觉-语言原型, 几何关联迁移

## 一句话总结
提出 S-GEAR 框架，通过学习视觉动作原型并利用语言模型的语义关联来引导原型之间的几何关系，使模型理解动作间的语义互联性，从而提升动作预测性能，在 Epic-Kitchens 55/100、EGTEA Gaze+、50 Salads 四个基准上取得 SOTA 或极具竞争力的结果。

## 研究背景与动机

**领域现状**：动作预测（Action Anticipation）旨在从已观察的部分事件序列中预测未来活动，是自动驾驶和可穿戴助手等应用的关键能力。现有方法主要通过 LSTM 或因果 Transformer 等序列模型来处理时序信息。

**现有痛点**：
   - 传统方法侧重于提取更好的视觉和时序信息，但无法显式建模动作之间超越即时视频上下文的语义连接关系
   - 认知科学研究表明，语义互联性是人类预测未来行为的基础——我们通过将动作与物体、意图、可能结果相关联来形成可靠预测
   - 仅从视频中建模动作间语义关系面临两大困难：(a) 需要处理极长序列才能捕获足够的共现上下文；(b) 动作在视频中的分布严重不均衡

**核心思路**：利用语言模型天然编码了概念间语义关系这一特性，将语言空间中动作标签之间的几何关联（而非特征本身）迁移到视觉原型空间，使视觉原型在保留视觉线索的同时获得语义感知能力。

## 方法详解

### 整体框架
S-GEAR 包含四个核心组件：(1) ViT 视觉编码器提取帧级特征；(2) Temporal Context Aggregator (TCA) 模块融合时序上下文；(3) Prototype Attention (PA) 模块将特征与可学习原型交互；(4) Causal Transformer 解码器预测未来表示。此外还包含一套基于视觉/语言双原型的语义引导策略。

### 视觉编码器
- 输入视频段 $V_o = \{f_0, \ldots, f_{T-1}\}$，使用 ViT-B/16 将每帧分割为 $P$ 个 patch token
- 添加可学习位置编码和 CLS token，经 Transformer blocks 处理得到帧级特征 $I_t = \phi(S_t)$

### Temporal Context Aggregator (TCA)
- 灵感来自因果 Transformer，但考虑帧内所有 patch token（而非仅 CLS token）
- 在帧序列上施加因果掩码注意力，使当前帧能接收过去帧的全部细粒度上下文
- 输出因果增强中间特征 $\bar{I} \in \mathbb{R}^{T \times (P+1) \times d}$

### Prototype Attention (PA)
- 与 TCA 并行运行，将帧的 CLS token $I_t^0$ 作为 query，视觉原型作为 key 和 value
- 通过注意力机制聚合与当前帧最相关的原型信息，产生语义增强特征 $\tilde{I} \in \mathbb{R}^{T \times d}$
- 最终通过可学习权重 $\lambda$ 加权融合：$\hat{I} = \lambda \bar{I}^0 + (1-\lambda)\tilde{I}$

### 因果 Transformer 解码器
- 自回归解码器 $\Omega$ 处理融合特征 $\hat{I}$，基于掩码自注意力生成未来特征序列 $\zeta$
- 对于 $t=T-1$，$z_t$ 即为预测的未来动作特征

### 语义引导策略（核心创新）

**双原型定义**：
- **语言原型** $\rho_\ell \in \mathbb{R}^{K \times d}$：用 Sentence Transformer 编码动作标签（动词+名词），固定不动，作为语义关系的参考
- **视觉原型** $\rho_v \in \mathbb{R}^{K \times d}$：可学习参数，从预训练动作识别模型的典型样本编码初始化

**公共通信空间（非直接对齐）**：
- 关键洞察：不直接对齐视觉和语言特征（会丢失视觉线索），而是对齐它们各自原型空间中的**相对位置关系**
- 视觉相对表示：$r_k^{z_t} = \cos(z_t, \rho_v[k])$，得到动作特征与所有视觉原型的余弦相似度向量
- 语言相对表示：$r_k^{\text{enc}(y)_t} = \cos(\text{enc}(y)_t, \rho_\ell[k])$，得到标签编码与所有语言原型的相似度向量
- 语义损失：$\mathcal{L}_{Sem} = |r^{z_t} - r^{\text{enc}(y)_t}|$，促使视觉空间中动作间的几何关系模仿语言空间

**Lasso 正则化**：$\mathcal{L}_{reg} = ||z_t - \rho_v[k]||_2^2$，将动作表示拉向其对应类别的视觉原型

**分类头（Cosine Attention）**：
- 计算预测特征 $z_{T-1}$ 与所有视觉原型的余弦相似度
- 经 softmax 转化为权重后加权聚合原型：$\bar{z}_{T-1} = \text{softmax}(r^{z_{T-1}}) \cdot \rho_v$
- 通过可学习 sigmoid 门控融合：$\hat{z}_{T-1} = \sigma(\alpha) z_{T-1} + (1-\sigma(\alpha)) \bar{z}_{T-1}$
- 最终线性层 + softmax 输出类别概率

### 总损失函数
$$\mathcal{L}_{tot} = \lambda_1 \mathcal{L}_{Sem} + \lambda_2 \mathcal{L}_{Cls} + \lambda_3 \mathcal{L}_{Past} + \lambda_4 \mathcal{L}_{Feat}$$

- $\mathcal{L}_{Cls}$：未来动作分类交叉熵损失
- $\mathcal{L}_{Past}$：利用因果解码器对已观察帧的过去动作分类损失
- $\mathcal{L}_{Feat}$：预测的未来帧特征与实际下一帧特征的距离损失

## 实验关键数据

### 数据集

| 数据集 | 类型 | 规模 | 动作类数 |
|--------|------|------|----------|
| Epic-Kitchens 55 | 第一人称厨房 | 432 视频 / ~40K 段 | 2,747 |
| Epic-Kitchens 100 | 第一人称厨房 | 700 视频 / ~90K 段 | 4,053 |
| EGTEA Gaze+ | 第一人称厨房 | 28 小时 | 106 |
| 50 Salads | 第三人称沙拉制备 | 50 视频 | 17 |

### 核心结果
- **EK55 多模态**（RGB+Obj+Flow）：Top-1 Acc 22.7（+3.5），Top-5 Acc 43.2（+2.0）
- **EK100 单模态**（RGB，ViT-B）：Action Top-5 Recall 18.3（+0.7 vs RAFTformer-16），且无需时空预训练
- **EK100 S-GEAR-2B 融合**：Action Top-5 Recall 19.6（+0.5 vs RAFTformer-2B）
- **EGTEA Gaze+**：仅用 RGB 即达 Top-1 Acc 45.7（+2.7），Top-5 Acc 71.9（+0.4 vs HRO 使用三模态）
- **50 Salads**：在 8 种设置中 5 种超越 SOTA，Top-1 Acc 提升最高达 3.5 个绝对点

### 消融实验（EK100 Action Top-5 Recall）

| 配置 | Action |
|------|--------|
| Baseline（ViT+CT） | 15.2 |
| + Semantic 原型学习 | 17.8（+2.6） |
| + TCA | 16.7 |
| + PA + Sem | 18.0 |
| S-GEAR（TCA+PA+Sem） | 18.3 |
| TCA+PA 但使用语言原型 $\rho_\ell$（无语义迁移） | 17.4 |

- 语义原型学习贡献最大（+2.6）
- 直接使用语言原型不如学习视觉原型（17.4 vs 18.3），验证了间接几何关系迁移的优越性
- 仅使用 10% 原型子集即可达到 17.8（vs 100% 的 18.3），计算量可减 90%

## 亮点

1. **新颖的语义迁移范式**：不直接对齐视觉和语言特征，而是对齐二者在各自原型空间中的几何关联。这避免了跨模态对齐中丢失模态特有信息的问题
2. **认知科学启发**：从人类的语义互联性认知出发，将这种能力编码到视觉模型中
3. **跨场景泛化**：在第一人称/第三人称、短期/长期预测、不同规模数据集上均有效
4. **仅 RGB 即超越多模态**：在 EGTEA Gaze+ 上使用单 RGB 模态即超过使用三模态的 HRO
5. **计算效率**：可使用少量原型子集近似完整原型，大幅降低计算量

## 局限与展望

1. **缺乏内建多模态机制**：当前多模态结果依赖后期融合（late fusion），没有在架构层面集成多模态信息
2. **语义关系未考虑时序**：当前建模的是动作共现关系，未显式考虑动作的先后顺序。考虑顺序可缩小未来预测的不确定性
3. **依赖预训练语言模型的质量**：实验表明不同 Sentence Transformer 的语义建模质量直接影响结果（STSB 优于 BERT）
4. **帧尺寸较大**：ViT-B 使用 384×384 输入，比同类方法的 224×224 更耗计算资源

## 与相关工作的对比

| 方法 | 核心差异 |
|------|----------|
| AVT | 同为 ViT+CT 架构，但无语义引导，S-GEAR 在 EK55 ViT-B 上 Top-1 +3.3 |
| DCR | 课程学习方法，在 EK55 TSN 特征上 S-GEAR Top-1 +2.0 |
| MeMViT/RAFTformer | 使用更强的 MViTv2 编码器和 Kinetics 预训练，S-GEAR 用更简单的 ViT-B+IN21K 即可匹敌 |
| HRO | 存储长期动作原型，但需三模态；S-GEAR 仅 RGB 即超越 |
| CLIP 类方法 | 直接对齐视觉-语言空间；S-GEAR 只迁移几何关系，不做空间对齐 |

## 启发与关联

1. **几何关系迁移 vs 特征对齐**是一个有价值的一般性范式——当两个空间的特征语义差异很大时，迁移拓扑结构比对齐特征更灵活
2. 原型学习在动作预测中的成功可推广到其他时序预测任务（如轨迹预测、事件预测）
3. 可探索将动作序列的马尔可夫链/图结构融入语义引导，弥补当前不考虑顺序的局限

## 评分
- 新颖性: ⭐⭐⭐⭐ — 几何关联迁移而非直接特征对齐的思路新颖
- 实验充分度: ⭐⭐⭐⭐ — 四个数据集、单/多模态、多骨干、详细消融
- 写作质量: ⭐⭐⭐⭐ — 动机清晰、图表规范
- 价值: ⭐⭐⭐⭐ — 打开动作语义互联性研究的新方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] iTimER: Reconstruction Error-Guided Irregularly Sampled Time Series Representation Learning](../../AAAI2026/time_series/beyond_observations_reconstruction_error-guided_irregularly_sampled_time_series_.md)
- [\[ICLR 2026\] GTM: A General Time-series Model for Enhanced Representation Learning](../../ICLR2026/time_series/gtm_a_general_time-series_model_for_enhanced_representation_learning_of_time-series.md)
- [\[ICLR 2026\] Towards Generalizable PDE Dynamics Forecasting via Physics-Guided Invariant Learning](../../ICLR2026/time_series/towards_generalizable_pde_dynamics_forecasting_via_physics-guided_invariant_lear.md)
- [\[NeurIPS 2025\] Universal Spectral Tokenization via Self-Supervised Panchromatic Representation Learning](../../NeurIPS2025/time_series/universal_spectral_tokenization_via_self-supervised_panchromatic_representation_.md)
- [\[AAAI 2026\] C3RL: Rethinking the Combination of Channel-independence and Channel-mixing from Representation Learning](../../AAAI2026/time_series/c3rl_rethinking_the_combination_of_channel-independence_and_channel-mixing_from_.md)

</div>

<!-- RELATED:END -->
