---
title: >-
  [论文解读] Guided Perturbation Sensitivity (GPS): Detecting Adversarial Text via Embedding Stability and Word Importance
description: >-
  [AAAI 2026][文本检测] 提出 Guided Perturbation Sensitivity (GPS) 框架，通过对重要词进行遮蔽并测量嵌入表示的稳定性变化来检测对抗文本样本，在3个数据集、3种攻击、2个模型上实现85%+检测准确率，且无需重训练即可跨数据集/攻击/模型泛化。 对抗文本攻击是 Transform…
tags:
  - "AAAI 2026"
  - "文本检测"
  - "embedding stability"
  - "word importance"
  - "BiLSTM"
  - "perturbation sensitivity"
---

# Guided Perturbation Sensitivity (GPS): Detecting Adversarial Text via Embedding Stability and Word Importance

**会议**: AAAI 2026  
**arXiv**: [2508.11667](https://arxiv.org/abs/2508.11667)  
**代码**: [GitHub](https://github.com/ReDASers/Guided-Perturbation-Sensitivity)  
**领域**: 其他  
**关键词**: adversarial text detection, embedding stability, word importance, BiLSTM, perturbation sensitivity

## 一句话总结

提出 Guided Perturbation Sensitivity (GPS) 框架，通过对重要词进行遮蔽并测量嵌入表示的稳定性变化来检测对抗文本样本，在3个数据集、3种攻击、2个模型上实现85%+检测准确率，且无需重训练即可跨数据集/攻击/模型泛化。

## 研究背景与动机

对抗文本攻击是 Transformer 模型面临的持续威胁——一个单词的替换就可以让SOTA模型把正面影评分类为负面。与视觉领域的连续像素扰动不同，文本攻击在离散的词汇空间中操作，需要在保持语义的同时欺骗模型，这让检测变得更加困难。

现有防御方法面临的核心矛盾在于：要么依赖特定攻击模式的先验知识（缺乏通用性），要么需要昂贵的模型重训练（实用性差）。基于输出层信号的方法容易过拟合特定攻击，基于梯度的检测器忽略了对抗操纵的序列结构信息。

本文的核心洞察来自一个理论基础：对抗样本位于决策边界附近的高曲率区域，微小扰动会导致剧烈的分类变化。作者假设这种不稳定性不仅存在于决策边界，还延伸到表示空间本身——通过策略性地遮蔽重要词，对抗样本应该展现出与自然重要词不成比例的高敏感度。这一洞察催生了 GPS 方法：用嵌入不稳定性的模式来"指纹"对抗样本。

## 方法详解

### 整体框架

GPS 的流程分为四个阶段：(1) 计算参考嵌入；(2) 用重要性启发式方法排序词；(3) 通过序贯遮蔽测量嵌入敏感度；(4) 将敏感度-重要性特征张量送入 BiLSTM 检测器进行分类。整个过程不需要修改目标模型。

### 关键设计

1. **参考嵌入计算**:

    - 功能：为输入文本计算一个全局的句子嵌入表示
    - 核心思路：将最后一层所有非特殊 token 的隐状态取均值，$\mathbf{e}(\mathcal{T}) = \frac{1}{|\Omega|}\sum_{i \in \Omega} \mathbf{h}_i^{(L)}$，其中 $\Omega$ 为非特殊 token 集合
    - 设计动机：取均值嵌入比 [CLS] token 更能捕捉全局语义变化，为后续衡量遮蔽引发的漂移提供基准线

2. **重要词识别（四种启发式方法）**:

    - 功能：对输入文本中的每个词赋予重要性分数并排序
    - 核心思路——梯度归因（最优）：对交叉熵损失相对于输入嵌入反向传播梯度，每个词的重要性为其子 token 梯度 $\ell_2$ 范数之和，$\alpha_k^{\text{sal}} = \sum_{j \in \mathcal{S}_k} \|\nabla_{\mathbf{e}_j} \ell(\mathcal{T})\|_2$
    - 还评估了 Attention Rollout（聚合多层注意力权重）、Grad-SAM（梯度与注意力的逐元素乘积）、随机选择
    - 设计动机：梯度方法直接反映哪些词对模型预测最关键，而被对抗修改的词通常是那些梯度信号最强的词，因此梯度方法能更精准地定位被篡改词

3. **序贯敏感度分析（核心模块）**:

    - 功能：逐一遮蔽 top-K 重要词，测量每次遮蔽造成的嵌入变化
    - 核心思路：对每个被选中的词 $w_k$，用 [MASK] 替换后重新计算嵌入 $\tilde{\mathbf{e}}_k$，敏感度定义为余弦距离 $s_k = 1 - \frac{\mathbf{e}(\mathcal{T}) \cdot \tilde{\mathbf{e}}_k}{\|\mathbf{e}(\mathcal{T})\|_2 \|\tilde{\mathbf{e}}_k\|_2}$
    - 关键发现：对抗样本中被篡改的词展现出约2倍于良性样本中自然重要词的敏感度，说明嵌入不稳定性是对抗样本的内在属性
    - 设计动机：逐词遮蔽避免了同时遮蔽多个词带来的交互效应，能精准量化每个词对整体表示的影响

4. **GPS 特征张量与 BiLSTM 检测器**:

    - 功能：将敏感度序列与重要性序列堆叠成 $N \times 2$ 的特征矩阵 $\mathbf{Z} = [\mathbf{s} \| \boldsymbol{\alpha}]$，保留原始词序位置信息
    - BiLSTM 参数量仅 257,154，用 AdamW 优化器（$lr = 5 \times 10^{-4}$），batch size 32，早停策略（patience=5 epoch）
    - 设计动机：BiLSTM 能捕捉敏感度模式中的序列依赖关系（如对抗修改常聚集在特定位置），同时参数量低、计算开销可控

### 损失函数 / 训练策略

使用标准的二分类交叉熵损失，5000 条平衡训练集（20%作为验证集），1000 条测试集。训练数据中的对抗样本仅包含成功欺骗模型的样本（排除不成功的扰动），确保训练质量。

## 实验关键数据

### 主实验

实验矩阵覆盖 3 数据集 × 3 攻击 × 2 模型 × 4 重要性方法 = 72 种配置，并与 TextShield 和 Sharpness-based 两个 SOTA 基线对比。

| 数据集 | 模型 | 攻击 | GPS(Grad) | TextShield | Sharp | 提升(vs最佳基线) |
|--------|------|------|-----------|------------|-------|---------|
| AG News | RoBERTa | TextFooler | 0.887 | 0.893 | 0.874 | -0.6% |
| AG News | RoBERTa | DeepWordBug | 0.895 | 0.883 | 0.860 | +1.2% |
| IMDB | RoBERTa | TextFooler | 0.919 | 0.870 | 0.888 | +3.1% |
| IMDB | DeBERTa | DeepWordBug | **0.968** | 0.775 | 0.775 | **+19.3%** |
| Yelp | DeBERTa | TextFooler | 0.917 | 0.917 | 0.911 | +0.0% |
| Yelp | DeBERTa | DeepWordBug | 0.931 | 0.902 | 0.893 | +2.9% |

GPS(Grad) 在 18 种配置中大多数达到或超过基线。在 IMDB+DeBERTa+DeepWordBug 上性能优势最显著（+19.3%），TextShield 和 Sharp 在该配置下严重退化。

### 消融实验

**重要性方法消融**（敏感度分析，Table 1）:

| 重要性方法 | 良性均值 | 对抗均值 | 比值 |
|-----------|---------|---------|------|
| Gradient | 0.014 | 0.028 | 1.932 |
| Attention Rollout | 0.014 | 0.028 | 1.912 |
| Grad-SAM | 0.014 | 0.027 | 1.836 |
| Random | 0.013 | 0.026 | 1.880 |

所有方法下对抗样本的敏感度都约为良性样本的2倍，说明嵌入不稳定性是对抗样本的固有特性。

**K值消融**:

| K值 | 相对性能(vs K=50) | 计算时间趋势 |
|-----|-----------------|-------------|
| K=5 | 98%+ | 最小 |
| K=10 | 99%+ | 线性增长 |
| K=20 | ~100% | 线性增长 |
| K=50 | 100% | 最大 |

K=5 即可达到峰值性能的 98%，性能随 K 增大变化极小（<0.015 F1），而计算时间线性增长。最优平衡点为 K∈[5,10]。

### 关键发现

- 对抗样本展现出约 2× 的嵌入敏感度，88.9% 的实验配置中对抗样本比良性样本更不稳定
- 梯度归因在词级攻击的扰动识别上显著优于注意力方法，NDCG 排序质量与检测性能强相关（$\rho > 0.65$）
- 字符级攻击（DeepWordBug）的检测机制不同，扰动识别质量与检测性能无相关性
- GPS 在跨数据集、跨攻击、跨模型三种迁移场景中均展现稳健泛化能力

## 亮点与洞察

- 从理论到实践的优雅桥接：将决策边界不稳定性的理论洞察延伸到嵌入空间，并设计出简单有效的检测方法
- "嵌入敏感度指纹"的思想具有高度普适性，不依赖特定攻击类型或模型架构
- K=5 即可达到 98% 的检测性能，意味着每条文本只需 5 次前向传播即可完成检测，实际部署成本很低
- 揭示了词级攻击与字符级攻击的检测机制本质不同，为未来的统一检测框架设计提供了方向

## 局限与展望

- 需要白盒模型访问（梯度计算），在纯黑盒场景下需要替代方案（如代理模型的显著性）
- BiLSTM 检测器需要标注训练数据，在全新攻击类型上可能需要少量标签
- 对字符级攻击的检测依赖不同的机制，当前框架未特别优化这一场景
- 未探索基于输入特征自适应选择 K 值的策略
- 可考虑融合梯度和注意力启发式方法来同时应对词级和字符级攻击

## 相关工作与启发

- TextShield (Shen et al., 2023) 使用四个 LSTM 的集成处理梯度特征，GPS 用单个 BiLSTM 处理敏感度-重要性对就能达到相当甚至更好的效果，体现了特征设计比模型复杂度更重要
- Sharp (Zheng et al., 2023) 基于损失景观曲率检测，但对数据集和模型偏移敏感；GPS 的嵌入敏感度更稳定
- 对"注意力能否解释预测"这一议题的实证贡献：在对抗检测任务中，注意力确实不如梯度可靠

## 评分

- 新颖性: ⭐⭐⭐⭐ — 核心idea（嵌入遮蔽敏感度）直觉清晰且有理论依据，但技术组件（BiLSTM分类器等）相对常规
- 实验充分度: ⭐⭐⭐⭐⭐ — 18种配置 + 4种启发式方法 + 3维迁移实验 + K值消融 + NDCG排序分析，非常全面
- 写作质量: ⭐⭐⭐⭐ — 逻辑清晰，图表丰富，分析深入
- 价值: ⭐⭐⭐⭐ — 提供了实用的攻击无关检测框架，计算效率优势明显，但需白盒访问限制了部分场景的应用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Map&Make: Schema Guided Text to Table Generation](../../ACL2025/others/mapmake_schema_guided_text_to_table_generation.md)
- [\[AAAI 2026\] Boosting Adversarial Transferability via Ensemble Non-Attention](boosting_adversarial_transferability_via_ensemble_non-attention.md)
- [\[ECCV 2024\] Wavelength-Embedding-guided Filter-Array Transformer for Spectral Demosaicing](../../ECCV2024/others/wavelength-embedding-guided_filter-array_transformer_for_spectral_demosaicing.md)
- [\[ICML 2025\] On the Importance of Gaussianizing Representations](../../ICML2025/others/on_the_importance_of_gaussianizing_representations.md)
- [\[AAAI 2026\] ASAG: Toward the Frontiers of Reliable Diffusion Sampling via Adversarial Sinkhorn Attention Guidance](toward_the_frontiers_of_reliable_diffusion_sampling_via_adversarial_sinkhorn_att.md)

</div>

<!-- RELATED:END -->
