---
title: >-
  [论文解读] Reliably Detecting Model Failures in Deployment Without Labels
description: >-
  [NeurIPS 2025][模型监控 / 分布偏移检测 / 可信AI][部署后退化监控] 提出D3M(Disagreement-Driven Deterioration Monitoring)，一种基于变分贝叶斯后验采样的三阶段模型监控算法，在无标签、无训练数据的部署场景下可靠检测模型性能退化，同时对非退化性偏移保持低误报率。
tags:
  - "NeurIPS 2025"
  - "模型监控 / 分布偏移检测 / 可信AI"
  - "部署后退化监控"
  - "模型分歧"
  - "变分贝叶斯"
  - "无标签检测"
  - "临床AI"
---

# Reliably Detecting Model Failures in Deployment Without Labels

**会议**: NeurIPS 2025  
**arXiv**: [2506.05047](https://arxiv.org/abs/2506.05047)  
**代码**: [GitHub](https://github.com/teivng/d3m)  
**领域**: 模型监控 / 分布偏移检测 / 可信AI  
**关键词**: 部署后退化监控, 模型分歧, 变分贝叶斯, 无标签检测, 临床AI

## 一句话总结

提出D3M(Disagreement-Driven Deterioration Monitoring)，一种基于变分贝叶斯后验采样的三阶段模型监控算法，在无标签、无训练数据的部署场景下可靠检测模型性能退化，同时对非退化性偏移保持低误报率。

## 研究背景与动机

ML模型部署后面临分布偏移问题，但**并非所有偏移都导致模型性能退化**。核心挑战在于设计一个监控机制，在无标签的情况下区分"退化性偏移"（需要重新训练）和"良性偏移"（模型仍然表现良好）。

现有方法的desiderata和不足：(1) **无标签操作**——分布偏移检测方法(MMD-D, H-divergence等)可以在无标签下工作，但对非退化偏移有高误报率；(2) **无需训练数据**——模型分歧框架(Detectron等)需要持续访问训练数据来计算分歧统计量，这在隐私法规和边缘部署中不可行；(3) **鲁棒性**——应对退化偏移有高检出率，对非退化偏移有低误报率。D3M是文献中**唯一同时满足这三个desiderata**的方法(Table 1)。

核心idea：用变分贝叶斯最后一层(VBLL)替代完整贝叶斯神经网络或模型微调，通过从后验分布采样来近似最大分歧(maximum disagreement)，避免训练数据依赖。

## 方法详解

### 整体框架

D3M分三阶段：(1) **Train**——训练特征提取器+VBLL来建模logits的后验预测分布(PPD)；(2) **Calibrate**——在ID验证集上通过bootstrap采样建立最大分歧率的参考分布 $\Phi$；(3) **Deploy**——在部署数据上计算最大分歧率 $\tilde{\phi}$，分位数检验判断退化。

### 关键设计

1. **VBLL后验预测分布 (Section 2.2, Step 1)**: 由特征提取器 $\operatorname{FE}_\theta: \mathcal{X} \to \mathbb{R}^d$ 和VBLL $\operatorname{VBLL}_\theta: \mathbb{R}^d \to \mathcal{P}(\mathbb{R}^C)$ 组成。输出每个样本logits的高斯后验 $q_\theta(z|x) = \mathcal{N}(z|\mu_\theta(x), \operatorname{diag}(\sigma^2_\theta(x)))$。相比完整贝叶斯网络，VBLL仅对最后一层做变分推断，计算高效但采样多样性较低。训练目标为ELBO：$\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{z \sim q_\theta}[\log \operatorname{softmax}(z)_y] - \operatorname{KL}[q_\theta(z|x) \| p(z)]$。

2. **校准阶段——最大分歧率 (Step 2)**: 对 $T$ 轮bootstrap，每轮采样 $m$ 个ID样本，从每个样本的后验 $q_\theta(\cdot|x_i)$ 中抽 $K$ 个logit样本 $z_i^{(k)}$，经温度缩放softmax后从Categorical分布采样类标签 $\hat{y}_i^{(k)}$。计算每个采样与基础模型均值预测 $\bar{y}_i$ 的分歧率 $\operatorname{DisRate}(k) = \frac{1}{m}\sum_i \mathbb{1}\{\hat{y}_i^{(k)} \neq \bar{y}_i\}$，取最大值 $\phi_t = \max_k \operatorname{DisRate}(k)$。$T$ 轮后得到参考分布 $\Phi = \{\phi_t\}_{t \in [T]}$。

3. **部署监控 (Step 3)**: 收集部署数据 $\mathcal{D}_{\text{te}}^m$，用相同流程计算 $\tilde{\phi}$。如果 $\tilde{\phi} \geq \operatorname{Quantile}_{1-\alpha}(\Phi)$ 则报警。在分布未变时，$\tilde{\phi}$ 超过阈值的概率恰为 $\alpha$，直接控制了FPR。

4. **多样性增强策略**: 由于VBLL只变分化最后一层，采样多样性不足。采用两个技巧：(a) 温度缩放 $\operatorname{softmax}(z^{(k)}/\tau)$ 增加softmax输出的多样性；(b) 从Categorical分布采样(而非argmax)来获得标签，进一步增加分歧信号。

### 损失函数 / 训练策略

训练阶段使用ELBO最大化。基础模型的均值预测不在校准或部署过程中修改，保持原有的泛化保证。温度 $\tau$、bootstrap大小 $m$、后验采样数 $K$ 为可调超参数，训练和部署必须保持一致。

## 实验关键数据

### 主实验（退化偏移——TPR越高越好）

| 数据集 | 查询大小 | D3M | Detectron | MMD-D | H-Div | BBSD |
|--------|---------|-----|-----------|-------|-------|------|
| UCI Heart | 10 | .38±.19 | .24±.04 | .09±.03 | .15±.04 | .13±.03 |
| UCI Heart | 50 | .69±.33 | .82±.04 | .27±.04 | .37±.05 | .46±.05 |
| CIFAR-10.1 | 50 | .74±.12 | .83±.04 | .05±.02 | .04±.02 | .12±.03 |
| Camelyon17 | 10 | .89±.20 | .97±.02 | .42±.05 | .03±.02 | .16±.04 |
| Camelyon17 | 50 | .99±.02 | .96±.02 | .69±.05 | .23±.04 | .87±.03 |

### 消融实验（GEMINI临床数据）

| 场景 | 指标 | D3M | 其他基线 | 说明 |
|------|------|-----|---------|------|
| 时间偏移(非退化) | FPR@α=0.05 | <0.05 | 多数>0.05 | D3M抵抗良性偏移 |
| 年龄偏移(退化), 混合比0.75 | TPR | ~0.90 | ~0.90 | 与最强基线持平 |
| 年龄偏移(退化), 混合比0.25 | TPR | ~0.45 | ~0.40 | 低混合比仍有竞争力 |

### 关键发现

- **D3M最大优势是在非退化偏移下的低FPR**：GEMINI时间偏移实验中，性能未退化但分布确实改变(含COVID期间)，D3M保持低FPR而多数基线误报。
- **小查询大小下方差较大**：UCI数据集query=10时，D3M的TPR方差(±0.19)明显大于Detectron(±0.04)，这是采样策略带来的固有噪声。
- **D3M无需训练数据且免维调**：Detectron需要持续访问训练数据并做梯度微调，D3M只需前向传播。
- **较大查询大小(100/200)下D3M匹配最强基线**：在更大样本下性能和置信区间均改善。

## 亮点与洞察

- 首次提出满足"无标签+无训练数据+理论保证"三重要求的监控机制
- VBLL的使用非常聪明——避免了完整贝叶斯网络的计算开销，又得到了足够的后验不确定性
- 三阶段设计使得训练、校准、部署完全解耦，适合现实部署流程
- 在GEMINI这样的真实临床数据上验证，展示了选择性报警(该报报、不该报不报)的能力

## 局限与展望

- 小查询大小下的高方差是主要弱点——VBLL采样多样性不足导致
- 温度τ的选择需要sweep——过大导致过拟合参考分布造成高FPR
- 理论保证基于Idealized D3M的oracle版本，D3M是实际的近似，可能过度采样到ℋp之外
- 仅支持分类任务，回归任务的退化检测需要不同的分歧定义
- 预训练特征提取器的质量直接影响检测性能(Camelyon17用ImageNet预训练ResNet效果好)

## 相关工作与启发

- 位于模型分歧框架(Ginsberg 2023, Rosenfeld 2024)的延长线上，关键区别是无需训练数据
- 利用了VBLL(Harrison et al. 2024)做高效不确定性估计，但以非标准方式使用——不是为了OOD检测，而是为了找最大分歧
- 与测试时适应(TTA, Wang 2020)互补：D3M检测退化，TTA处理退化

## 方法核心流程

1. **Train**: FE + VBLL端到端训练，最大化ELBO
2. **Calibrate**: T轮bootstrap采样ID数据，计算最大分歧率集合$\Phi$
3. **Deploy**: 收集部署数据，计算$\tilde{\phi}$，与$\text{Quantile}_{1-\alpha}(\Phi)$比较
4. **关键公式**: $\operatorname{DisRate}(k) = \frac{1}{m}\sum_{i=1}^m \mathbb{1}\{\hat{y}_i^{(k)} \neq \bar{y}_i\}$, $\phi_t = \max_k \operatorname{DisRate}(k)$

## 评分

- 新颖性: ⭐⭐⭐⭐ VBLL+分歧的组合新颖，三个desiderata的形式化清晰
- 实验充分度: ⭐⭐⭐⭐⭐ 标准benchmark+真实临床GEMINI数据，退化/非退化场景全覆盖
- 写作质量: ⭐⭐⭐⭐ 动机和方法描述清楚，但理论部分主要在附录
- 价值: ⭐⭐⭐⭐⭐ 直接解决真实部署痛点，GEMINI实验特别有说服力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Optimism Without Regularization: Constant Regret in Zero-Sum Games](optimism_without_regularization_constant_regret_in_zero-sum_games.md)
- [\[ICML 2026\] When Sample Selection Bias Precipitates Model Collapse](../../ICML2026/learning_theory/when_sample_selection_bias_precipitates_model_collapse.md)
- [\[ICLR 2026\] Function Spaces Without Kernels: Learning Compact Hilbert Space Representations](../../ICLR2026/learning_theory/function_spaces_without_kernels_learning_compact_hilbert_space_representations.md)
- [\[ICLR 2026\] An Improved Model-free Decision-estimation Coefficient with Applications in Adversarial MDPs](../../ICLR2026/learning_theory/an_improved_model-free_decision-estimation_coefficient_with_applications_in_adve.md)
- [\[ICML 2026\] Optimal Design for Multinomial Logit Model with Applications to Best Assortment Identification](../../ICML2026/learning_theory/optimal_design_for_multinomial_logit_model_with_applications_to_best_assortment_.md)

</div>

<!-- RELATED:END -->
