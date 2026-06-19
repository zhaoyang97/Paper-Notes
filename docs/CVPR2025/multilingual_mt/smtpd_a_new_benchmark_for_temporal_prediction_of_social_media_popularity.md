---
title: >-
  [论文解读] SMTPD: A New Benchmark for Temporal Prediction of Social Media Popularity
description: >-
  [CVPR 2025][多语言/翻译][社交媒体流行度预测] 构建首个时间对齐的社交媒体流行度时序预测基准SMTPD（282K YouTube样本，30天连续观测），并提出基于多模态特征提取+LSTM时序回归的baseline框架，发现早期流行度（EP）是准确预测后续流行度的关键。 社交媒体流行度预测对内容优化、数字营销、在…
tags:
  - "CVPR 2025"
  - "多语言/翻译"
  - "社交媒体流行度预测"
  - "时序预测"
  - "多模态特征"
  - "多语言"
  - "基准数据集"
---

# SMTPD: A New Benchmark for Temporal Prediction of Social Media Popularity

**会议**: CVPR 2025  
**arXiv**: [2503.04446](https://arxiv.org/abs/2503.04446)  
**代码**: [https://github.com/zhuwei321/SMTPD](https://github.com/zhuwei321/SMTPD)  
**领域**: 多语言翻译  
**关键词**: 社交媒体流行度预测、时序预测、多模态特征、多语言、基准数据集

## 一句话总结

构建首个时间对齐的社交媒体流行度时序预测基准SMTPD（282K YouTube样本，30天连续观测），并提出基于多模态特征提取+LSTM时序回归的baseline框架，发现早期流行度（EP）是准确预测后续流行度的关键。

## 研究背景与动机

社交媒体流行度预测对内容优化、数字营销、在线广告有重要应用价值。现有数据集（如SMPD、TPIC17）存在三大问题：
1. **缺乏时间对齐**：不同帖子的发布时间不同，预测时间点不一致，而流行度分布随时间变化（呈衰减模式），不对齐的时间使预测难以准确
2. **多模态数据不足**：部分数据集只有单一模态
3. **语言多样性有限**：大多仅覆盖英语，无法适应国际化社交媒体平台

此外，现有方法大多做单点预测（预测某一时刻的popularity），忽视了流行度的时序动态。SMTPD通过从帖子发布起跟踪30天每日流行度，实现时间对齐的时序预测。

## 方法详解

### 整体框架

框架分为两层：上层为多模态特征提取（视觉、文本、数值、类别四个模态独立提取特征后MLP对齐+拼接）；下层为LSTM时序回归（融合特征编码为初始状态和每步输入，30个LSTM cell依次输出30天的流行度预测值）。

### 关键设计

1. **多模态特征提取（Multi-Modal Feature Extraction）**:
    - 功能：从帖子的多模态内容中提取丰富特征
    - 核心思路：
        - **视觉**：用ImageNet预训练的ResNet-101提取封面图2048维语义特征 $f_v = \text{ResNet}(\mathcal{S}(I))$，缺失封面用全零替代
        - **文本**：用BERT-Multilingual分别编码category/title/tags/description/userID得到5个768维向量，拼接后用 $5\times5$ 卷积融合为768维文本特征 $f_t$
        - **数值**：对follower数取log变换处理长尾分布，所有数值指标Z-score标准化后拼接（followers/posts/duration/title长度/tag数/description长度/EP）
        - **类别**：label和语言分别经embedding+MLP后逐元素相乘 $f_c = \text{MLP}_1(\text{E}_1(T^{cat})) \odot \text{MLP}_2(\text{E}_2(\text{langid}(T^{tit})))$
    - 设计动机：多语言环境下单语言模型失效，BERT-Multilingual解决跨语言问题；类别特征用乘法融合label和language信息以捕获地域-类别交互

2. **LSTM时序回归（Temporal Popularity Regression）**:
    - 功能：利用相邻日流行度的高相关性进行序列预测
    - 核心思路：融合特征 $F$ 通过同一个MLP编码为LSTM初始隐状态和细胞状态 $h_0 = c_0 = \text{MLP}^{hc}(F)$，每步输入由独立的temporal encoding MLP生成 $x_s = \text{MLP}_s^x(F)$。每步输出拼接 $h_s, c_s$ 后经MLP+ReLU（确保非负）得到预测值 $pre_s = \max(0, \text{MLP}_s^{out}(\text{Concat}(h_s, c_s)))$
    - 设计动机：相邻日流行度PC和SRC均很高，LSTM天然适合建模这种时序依赖；30个独立的output MLP允许每步有不同的映射

3. **Composite Gradient Loss (CGL，复合梯度损失)**:
    - 功能：综合考虑预测值、趋势和峰值位置的多目标损失
    - 核心思路：由5项组成——SmoothL1Loss（预测值误差）、一阶导差异（趋势斜率）、二阶导差异（趋势曲率）、峰值位置的onehot编码L1差异、Laplacian余项正则化。权重比1:1:1:1e-6，一阶/二阶权重用cosine annealing动态调整
    - 设计动机：仅用MSE/L1损失只优化每天绝对值，无法保证趋势形状正确。加入导数项约束趋势单调性和曲率，峰值项确保热度高峰时间准确

### 损失函数 / 训练策略

$\mathcal{L} = \text{SL}(\hat{P}, P) + \lambda_1 \text{SL}(\hat{P}^{(1)}, P^{(1)}) + \lambda_2 \text{SL}(\hat{P}^{(2)}, P^{(2)}) + \alpha \sum|\delta\text{argmax}(\hat{P}) - \delta\text{argmax}(P)| + \epsilon \cdot \text{LR}$

Adam优化器，L2正则1e-3，batch size 64，lr=1e-3 + ReduceLROnPlateau，5折交叉验证。

## 实验关键数据

### 主实验（SMTPD上多方法对比，含/不含Early Popularity）

| 方法 | Day7 MAE/SRC | Day14 MAE/SRC | Day30 MAE/SRC |
|------|-------------|--------------|--------------|
| Ding et al. (无EP) | 1.715/0.849 | 1.669/0.846 | 1.592/0.843 |
| Ding et al. (+EP) | 0.715/0.964 | 0.742/0.959 | 0.749/0.931 |
| Lai et al. (无EP) | 1.573/0.875 | 1.524/0.872 | 1.495/0.864 |
| Lai et al. (+EP) | 0.725/0.957 | 0.753/0.962 | 0.760/0.957 |
| **Ours (无EP)** | 1.673/0.852 | 1.628/0.850 | 1.563/0.848 |
| **Ours (+EP)** | **0.713/0.964** | **0.735/0.959** | **0.732/0.959** |

### 消融实验

| 配置 | AMAE↓ | ASRC↑ | 说明 |
|------|-------|-------|------|
| BERT-Base + LSTM | 0.782 | 0.958 | 单语言模型性能下降 |
| BERT-Multilingual + MLP | 0.786 | 0.958 | MLP缺乏时序建模能力 |
| **BERT-Multilingual + LSTM** | **0.717** | **0.959** | 多语言+时序建模最优 |

| EP配置 | AMAE↓ | Day30 MAE↓ | Day30 SRC↑ |
|--------|-------|-----------|-----------|
| w/o EP (预测1-30天) | 1.562 | 1.530 | 0.850 |
| w/o EP (预测2-30天) | 1.630 | 1.551 | 0.849 |
| w/o EP + Lai预测EP | 1.628 | 1.555 | 0.848 |
| **w/ EP (ours)** | **0.717** | **0.732** | **0.959** |

### 关键发现

- **早期流行度（EP）是决定性因素**：加入第1天真实流行度后，所有方法MAE降低约55-60%，SRC提升10+个百分点
- **SMTPD上SRC普遍高于SMPD**：因为时间对齐消除了不同发布时间带来的分布偏移
- **多语言建模必要性**：BERT-Base换成BERT-Multilingual后MAE降低8.3%
- **MLP vs LSTM**：MLP回归（类似Ding的方法）不如LSTM，说明时序依赖建模至关重要
- 流行度趋势高度可预测：30天内相邻日PC/SRC都非常高

## 亮点与洞察

- **数据集构建思路新颖**：从帖子发布开始连续30天每日观测，实现时间对齐，比"回顾式"采集更贴合实际应用
- **EP的核心作用被系统性验证**：EP与后续流行度的强相关性被多角度分析和实验验证
- **多语言覆盖**：282K样本覆盖90+种语言，揭示了不同语言的流行度偏差

## 局限与展望

- 未利用视频多帧信息（仅用封面图），丢失了动态视觉内容
- 深层多模态交互未深入探索（简单MLP拼接）
- EP作为输入需要第1天真实数据，限制了纯内容预测场景的应用
- 预训练特征提取器（ResNet-101/BERT）较陈旧，未尝试更强模型（如ViT/LLM）
- 仅基于YouTube平台，泛化到其他平台（TikTok/Instagram）未验证

## 相关工作与启发

- Szabo等人早期发现初始流行度对后续流行度有显著影响，本文用大规模数据再次验证
- SMPD/TPIC17等现有数据集的非对齐问题被首次明确指出并解决
- Khosla的popularity score变换 $p = \log_2(v/d + 1)$ 有效将长尾view count转化为合理分布
- 该benchmark可推动流行度预测从单点预测向时序预测的范式转变

## 评分
- 新颖性: ⭐⭐⭐ 主要贡献是benchmark和分析，方法创新有限
- 实验充分度: ⭐⭐⭐⭐ 多方法对比+消融+EP分析全面，但缺少与更多SOTA的比较
- 写作质量: ⭐⭐⭐ 结构清晰但部分公式冗余，baseline方法描述过于详细
- 价值: ⭐⭐⭐⭐ 数据集和时序预测范式对社交媒体研究社区有持续价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] PluRule: A Benchmark for Moderating Pluralistic Communities on Social Media](../../ACL2026/multilingual_mt/plurule_a_benchmark_for_moderating_pluralistic_communities_on_social_media.md)
- [\[ACL 2026\] Cross-Cultural Transfer of Emoji Semantics and Sentiment in Financial Social Media](../../ACL2026/multilingual_mt/cross-cultural_transfer_of_emoji_semantics_and_sentiment_in_financial_social_med.md)
- [\[NeurIPS 2025\] Zero-Shot Performance Prediction for Probabilistic Scaling Laws](../../NeurIPS2025/multilingual_mt/zero-shot_performance_prediction_for_probabilistic_scaling_laws.md)
- [\[ACL 2025\] EXECUTE: A Multilingual Benchmark for LLM Token Understanding](../../ACL2025/multilingual_mt/execute_a_multilingual_benchmark_for_llm_token_understanding.md)
- [\[ACL 2025\] CruxEval-X: A Benchmark for Multilingual Code Reasoning, Understanding and Execution](../../ACL2025/multilingual_mt/cruxeval-x_a_benchmark_for_multilingual_code_reasoning_understanding_and_executi.md)

</div>

<!-- RELATED:END -->
