---
title: >-
  [论文解读] Elastic ViTs from Pretrained Models without Retraining
description: >-
  [NeurIPS 2025][模型压缩][Transformer] SnapViT 提出一种后训练结构化剪枝方法：结合自监督梯度的局部 Hessian 和进化算法估计的全局跨模块相关性，无需重训练或标签即可在一次运行中生成连续稀疏度的弹性 ViT 子网络，在 A100 上仅需不到 5 分钟。 领域现状：视觉基础模型性能强大但…
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "Transformer"
  - "弹性推理"
  - "Hessian近似"
  - "进化算法"
  - "自监督重要性评分"
---

# Elastic ViTs from Pretrained Models without Retraining

**会议**: NeurIPS 2025  
**arXiv**: [2510.17700](https://arxiv.org/abs/2510.17700)  
**代码**: [elastic.ashita.nl](https://elastic.ashita.nl/)  
**领域**: 模型压缩 / 结构化剪枝  
**关键词**: Vision Transformer剪枝, 弹性推理, Hessian近似, 进化算法, 自监督重要性评分

## 一句话总结
SnapViT 提出一种后训练结构化剪枝方法：结合自监督梯度的局部 Hessian 和进化算法估计的全局跨模块相关性，无需重训练或标签即可在一次运行中生成连续稀疏度的弹性 ViT 子网络，在 A100 上仅需不到 5 分钟。

## 研究背景与动机

**领域现状**：视觉基础模型性能强大但仅提供几种固定尺寸（如 DINOv3 的 21M→6.7B），迫使用户选择"能塞进约束的最大模型"，常导致次优部署。

**现有痛点**：(a) 知识蒸馏需要预定义目标架构 + 通常不公开的预训练数据；(b) 弹性推理方法（Matryoshka/Matformer）需要在预训练时设计嵌套结构，无法应用于现有模型；(c) 现有剪枝方法针对特定计算约束和任务，通常需要重训练，且每次只能优化单一稀疏度。

**核心矛盾**：Hessian 的对角线近似（或 block-diagonal/K-FAC）仅捕获局部层内依赖，忽略了层间的跨模块相关性——但全 Hessian 有 $N^2$ 个元素，不可计算。

**本文目标** 从任意预训练 ViT 中提取横跨连续稀疏度的子网络族，无需重训练、无需标签、极快速。

**切入角度**：将 Hessian 分解为局部项（自监督梯度对角近似）和全局项（进化算法学习的跨模块相关性），两者结合得到统一的可剪枝性评分。

**核心 idea**：自监督梯度提供局部敏感度 + xNES 进化算法学习全局跨模块相关性 = 单次运行产生所有稀疏度的弹性模型。

## 方法详解

### 整体框架
输入：任意预训练 ViT → Step 1: 用 DINO 自监督损失在少量无标签数据上计算参数级梯度平方（局部 Hessian 对角近似）→ Step 2: xNES 进化算法优化全局结构缩放因子（全局 Hessian 近似）→ Step 3: 两者相乘得到统一可剪枝性评分 → 全局排序后按任意目标稀疏度剪枝。

### 关键设计

1. **局部 Hessian 近似（自监督梯度）**:

    - 功能：估计每个参数的局部敏感度
    - 核心思路：$H^{(l)} \approx \frac{1}{N_D}\sum_{i=1}^{N_D} \|\nabla_\theta \mathcal{L}_i\|^2$，仅保留 Hessian 对角项。使用 DINO 自监督目标 $\mathcal{L}^{\text{SSL}} = \sum_k \sum_m \mathcal{L}_{\text{CE}}(z_k^g, z_m^l)$（全局/局部 crop 的交叉视角一致性损失），无需分类头
    - 设计动机：自监督损失使方法适用于任何模型（有/无分类头），且对下游任务泛化性好

2. **全局 Hessian 估计（xNES 进化算法）**:

    - 功能：捕获跨模块（注意力头/FFN 块之间）的相关性
    - 核心思路：参数化搜索分布 $\mathcal{N}(\mu, \Sigma)$，其中 $\Sigma = BB^T$（$B = e^A$）。每次采样一个结构级缩放向量 $c$，与局部评分相乘后剪枝，用无标签适应度评估（原始/剪枝模型嵌入的 PCA 后余弦相似度）。xNES 自然梯度更新使 $\Sigma^{-1}$ 逼近跨模块 Hessian：$H^{(g)} \approx \alpha \Sigma^{-1}$
    - 设计动机：直接计算结构级 Hessian 仍不可行，但 xNES 通过黑盒优化隐式建模——沿敏感方向收缩方差、沿平坦方向扩展方差
    - 适应度函数：$F = \frac{1}{|\mathcal{S}|} \sum_{s \in \mathcal{S}} \text{sim}(\text{PCA}(z), \text{PCA}(z_{p_s}))$，在多个稀疏度 s 上评估

3. **弹性单次剪枝**:

    - 功能：一次计算产生所有稀疏度的子网络
    - 核心思路：统一评分 $P = \text{diag}(\frac{1}{N_D}\sum_i \|\nabla_\theta \mathcal{L}^{\text{SSL}}\|^2) \odot Mc$，其中 $M \in \{0,1\}^{N \times B}$ 是成员矩阵将模块因子扩展到参数级。全局排序后，任意稀疏度 $S$ 的子网络为 $\Theta_S = \{\theta_i | \text{rank}(P_i) < |\Theta|(1-S)\}$
    - 设计动机：一次进化优化覆盖所有稀疏度，而每个基线方法需要为每个目标稀疏度独立运行

### 损失函数 / 训练策略
无需训练/微调模型权重。xNES 优化 50-500 迭代（大规模预训练模型需更多迭代），每次在少量无标签图片上评估。总耗时 <5 分钟（A100 GPU）。

## 实验关键数据

### 主实验（DINO ViT-B/16，k-NN + Linear，7 数据集平均）

| 稀疏度 | 方法 | Avg k-NN | Avg Linear | 说明 |
|--------|------|----------|-----------|------|
| 0% | Unpruned | 69.1 | 72.0 | 原始模型 |
| 40% | SnapViT (Ours) | ~65 | ~68 | 精度下降<5%，加速1.58x |
| 40% | SNIP Magnitude | ~56 | ~57 | 大幅落后 |
| 40% | LAMP | ~45 | ~42 | 严重退化 |
| 50% | SnapViT (Ours) | 63.5 | — | 建模 FFN+heads 交互的效果 |
| 50% | Only FFN (12 interactions) | 60.1 | — | 仅建模 FFN |
| 50% | None (0 interactions) | 56.6 | — | 无全局相关性 |

### 消融实验

| 消融维度 | 配置 | 关键结果 |
|---------|------|---------|
| 全局交互 | 0 interactions | 56.6% k-NN (50% sparsity) |
| 全局交互 | FFN only (12) | 60.1% (+3.5) |
| 全局交互 | FFN + heads (156) | **63.5%** (+6.9) |
| 进化迭代 | 50 iter | 42.2% Linear (50% sparsity) |
| 进化迭代 | 500 iter | **44.0%** (+1.8) |
| 优化稀疏度数 | 1 个 | 与 6 个相近但偏低 |
| 优化稀疏度数 | 6 个 | 更稳健 |
| 损失函数 | SSL vs CE | SSL 仅微弱劣于 CE |

### ImageNet-1k 全微调（DeIT ViT-B/16，50% sparsity，300 epochs）

| 方法 | Avg k-NN | Avg Linear | ImageNet-1k |
|------|----------|-----------|-------------|
| Unpruned | 75.8 | 78.5 | 81.8 |
| NViT | 73.7 | 72.0 | **83.3** |
| SnapViT (Ours) | **75.4** | **75.9** | 82.6 |

SnapViT 在 ImageNet 上接近 NViT，但在 7 数据集泛化上远超（k-NN +1.7%, Linear +3.9%）。

### 关键发现
- **跨模块相关性至关重要**：从 0→156 interactions，k-NN 提升 6.9%——对角 Hessian 严重不足
- 大规模预训练模型（DINOv3/SigLIPv2）更难剪枝——大数据训练将表征均匀分布在参数上
- 剪枝自然先削减 FFN（尤其深层 8-12 块），注意力头更鲁棒
- 自监督梯度与有监督梯度性能差异极小，使方法真正无需标签
- 单次权重修正（SparseGPT 风格）可大幅恢复极端稀疏度下的性能

## 亮点与洞察
- **进化算法近似跨模块 Hessian**：绕过了显式计算 $N^2$ Hessian 元素的不可行性，用黑盒适应度评估隐式学习结构间相关性——这是真正的关键贡献
- **弹性一次运行**：所有基线每个目标稀疏度需要独立运行一次，而 SnapViT 一次 xNES 优化覆盖所有稀疏度——从实用角度极具吸引力
- **自监督使方法通用化**：DINO 目标使评分不依赖分类头，DINOv3/SigLIPv2 等 foundation model 首次可以被有效剪枝

## 局限与展望
- 大规模预训练模型（SigLIPv2/DINOv3）在 30% 以上稀疏度性能急剧下降——需要更好的后处理（权重修正已部分解决）
- xNES 的迭代次数需要人工调节（小模型 50，大规模预训练需 500），缺乏自动停止准则
- 仅针对 ViT 架构验证，CNN/混合架构的适用性未知
- 剪枝粒度为 FFN row-column 和整个注意力头——更细粒度（如通道级）可能获得更好的效率-精度权衡
- 适应度评估依赖 PCA 降维后的余弦相似度——可能丢失高维结构信息

## 相关工作与启发
- **vs LAMP**: 基于权重幅度的剪枝，在自监督模型上严重退化（50% sparsity 下差 SnapViT 约 21%）。缺乏全局依赖建模是根本原因
- **vs LLM Surgeon**: 5-shot 逐步剪枝 + 权重修正，需要分类头。SnapViT 单次无标签就能匹配或超越
- **vs Matformer**: 需要在预训练时设计嵌套结构。SnapViT 可应用于任意已有预训练模型
- **vs NViT**: 需要 300 epoch 完整微调才能超越，SnapViT 无训练就能接近，微调后泛化更好

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ xNES 近似全局 Hessian + 自监督评分 + 单次弹性剪枝的组合高度原创
- 实验充分度: ⭐⭐⭐⭐⭐ 8 数据集 + 5 个模型家族 + k-NN/Linear/Seg 三种评估 + 详尽消融
- 写作质量: ⭐⭐⭐⭐ 方法推导层次分明，但部分符号较密集
- 价值: ⭐⭐⭐⭐⭐ 对 ViT 部署有直接实用价值，<5分钟生成弹性模型，行业可直接采用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Olica: Efficient Structured Pruning of Large Language Models without Retraining](../../ICML2025/model_compression/olica_efficient_structured_pruning_of_large_language_models_without_retraining.md)
- [\[NeurIPS 2025\] Understanding Differential Transformer Unchains Pretrained Self-Attentions](understanding_differential_transformer_unchains_pretrained_self-attentions.md)
- [\[NeurIPS 2025\] Deterministic Continuous Replacement: Fast and Stable Module Replacement in Pretrained Transformers](deterministic_continuous_replacement_fast_and_stable_module_replacement_in_pretr.md)
- [\[NeurIPS 2025\] AutoJudge: Judge Decoding Without Manual Annotation](autojudge_judge_decoding_without_manual_annotation.md)
- [\[CVPR 2026\] ThinkingViT: Matryoshka Thinking Vision Transformer for Elastic Inference](../../CVPR2026/model_compression/thinkingvit_matryoshka_thinking_vision_transformer_for_elastic_inference.md)

</div>

<!-- RELATED:END -->
