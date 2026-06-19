---
title: >-
  [论文解读] CoIDO: Efficient Data Selection for Visual Instruction Tuning via Coupled Importance-Diversity Optimization
description: >-
  [NeurIPS 2025][多模态VLM][数据选择] 提出 CoIDO，一个双目标优化数据选择框架，通过联合优化数据重要性和多样性，仅用 20% 随机数据训练轻量评分器，即可从 LLaVA-665K 中选出 20% 子集达到全量微调 98.2% 的性能，同时计算开销为所有方法最低。 多模态大语言模型（MLLM）的指令微调…
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "数据选择"
  - "指令微调"
  - "重要性-多样性优化"
  - "轻量评分器"
  - "MLLM"
---

# CoIDO: Efficient Data Selection for Visual Instruction Tuning via Coupled Importance-Diversity Optimization

**会议**: NeurIPS 2025  
**arXiv**: [2510.17847](https://arxiv.org/abs/2510.17847)  
**代码**: [GitHub](https://github.com/SuDIS-ZJU/CoIDO)  
**领域**: 多模态VLM  
**关键词**: 数据选择, 指令微调, 重要性-多样性优化, 轻量评分器, MLLM

## 一句话总结

提出 CoIDO，一个双目标优化数据选择框架，通过联合优化数据重要性和多样性，仅用 20% 随机数据训练轻量评分器，即可从 LLaVA-665K 中选出 20% 子集达到全量微调 98.2% 的性能，同时计算开销为所有方法最低。

## 研究背景与动机

多模态大语言模型（MLLM）的指令微调依赖大规模数据集（如 LLaVA-665K），但训练一轮 LLaVA-1.5-7B 就需要 8×A100 40GB GPU 超过 20 小时。数据选择方法旨在找到高质量子集来降低成本，但现有方法存在两个关键缺陷：

**重要性与多样性解耦**: 训练阶段评估数据重要性，选择阶段通过独立算法优化多样性，两个目标无法联合优化，容易导致次优权衡

**需要遍历全量数据**: 需要目标 MLLM 处理整个数据集来计算梯度或特征，计算成本与全量微调相当，违背了数据选择的初衷

CoIDO 的核心思想是：用少量数据训练一个轻量评分器，让它学会数据分布，然后推断全数据集的选择分数。

## 方法详解

### 整体框架

给定大规模视觉指令数据集 $\mathcal{D} = \{z_j\}_{j=1}^N$，目标是选择子集 $\mathcal{D}_h \subset \mathcal{D}$，使得 $|\mathcal{D}_h| = \gamma N$（如 $\gamma = 0.2$），同时保持至少 95% 的全量性能。

CoIDO 的流程：

1. **特征提取**: 提取文本特征（LLM Score）、图像特征（ImageReward Score）、图文对齐特征（CLIP Score）
2. **谱聚类**: 对多模态特征进行谱聚类，得到 $M$ 个类簇
3. **评分器训练**: 仅用 $p\%$（如 20%）随机数据训练 CoIDO Scorer（4 层 MLP），与目标 MLLM 联合训练
4. **数据选择**: 对全量数据推断 CoIDO 分数，按任务分层选择 top-$\gamma$ 子集

### 关键设计

**重要性损失 $\mathcal{L}_I$**: 评分器为每个样本输出 CoIDO 分数 $w_{ik}$，经 Softmax 归一化后加权目标 MLLM 的交叉熵损失：

$$\mathcal{L}_I = \sum_{i=1}^{m} \sum_{k=1}^{n_i} w_{ik} \cdot \text{CE}(y_{ik}, \hat{y}_{ik})$$

根据反向传播原理，高交叉熵（难学样本）对应低 $w_{ik}$，即低分 = 高重要性。

**多样性损失 $\mathcal{L}_D$**: 最小化各簇平均权重的方差，防止某些簇主导选择：

$$\mathcal{L}_D = \text{Var}(\{\bar{w}_1, \bar{w}_2, \ldots, \bar{w}_m\}), \quad \bar{w}_i = \frac{1}{n_i} \sum_{k=1}^{n_i} w_{ik}$$

注意 $\mathcal{L}_D$ 仅约束簇间均值，保留簇内排序。

### 损失函数

采用同方差不确定性（homoscedastic uncertainty）框架动态平衡两个目标：

$$\mathcal{L}_{\text{total}} = \frac{1}{\sigma_I^2} \mathcal{L}_I + \frac{1}{2\sigma_D^2} \mathcal{L}_D + \log \sigma_I + \log \sigma_D$$

其中 $\sigma_I, \sigma_D$ 是可学习参数，分别调节重要性和多样性目标的不确定性。推理时丢弃这两个参数，直接用训练好的评分器打分。

重要性目标的推导基于温度缩放的 Boltzmann 分布：

$$p(y_{ik} | x_{ik}, \theta, \sigma_I, w_{ik}) = \text{Softmax}\left(\frac{w_{ik}}{\sigma_I^2} f_\theta(x_{ik})\right)$$

通过二阶 Taylor 展开近似 log-sum-exp 项，利用 LLM 输出分布高度集中（有效候选 token $T \approx 5\text{-}10$）的特性忽略一阶误差项。

## 实验关键数据

### 主实验

**LLaVA-665K 上选择 20% 数据微调 LLaVA-1.5-7B-LoRA**:

| 方法 | VQAv2 | GQA | SQA-I | POPE | MME | MMBench(en) | Rel.(%) | 训练数据量 | FLOPs |
|------|-------|-----|-------|------|-----|-------------|---------|-----------|-------|
| Full Data | 79.1 | 63.0 | 68.4 | 86.4 | 1476.9 | 66.1 | 100 | — | 10.2E |
| Random | 75.9 | 59.3 | 68.6 | 85.9 | 1461.0 | 60.3 | 95.1 | — | — |
| ICONS | 77.0 | 60.4 | 70.4 | 86.1 | 1447.7 | 64.6 | 97.1 | 100+5+2.2% | 12.6E |
| COINCIDE | 76.5 | 59.8 | 69.2 | 86.1 | 1495.6 | 63.1 | 97.4 | 100% | 4.9E |
| **CoIDO** | **77.2** | **60.4** | 69.4 | 85.4 | 1450.2 | **63.8** | **98.2** | **20%** | **4.2E** |

### 消融实验

**优化策略对比**:

| 损失函数 | VQAv2 | GQA | MMBench(en) | Rel.(%) |
|---------|-------|-----|-------------|---------|
| 仅 $\mathcal{L}_I$ | 77.9 | 48.9 | 51.1 | 89.0 |
| $\mathcal{L}_I + \mathcal{L}_D$ | 74.5 | 55.8 | 57.0 | 92.0 |
| $\lambda \mathcal{L}_I + (1-\lambda) \mathcal{L}_D$ | 76.1 | 59.4 | 60.5 | 95.9 |
| 同方差不确定性（Ours） | **77.2** | **60.4** | **63.8** | **98.2** |

**训练数据比例敏感性**: $p < 10\%$ 时性能一般，$p > 20\%$ 后稳定，20% 即可捕获数据分布。

### 关键发现

1. CoIDO 以 20% 训练成本和 4.2 ExaFLOPs 达到 98.2% 相对性能，效率和精度双优
2. 仅优化重要性（$\mathcal{L}_I$）时 Rel. 仅 89%，加入多样性后跃升至 98.2%，证明联合优化的必要性
3. MLP 评分器因输入特征已足够丰富（CLIP Score、ImageReward 等），性能优于更复杂的 Transformer 评分器

## 亮点与洞察

- **极致效率**: 仅需处理 20% 数据即可完成选择器训练，相比 ICONS 需要 100%+5%+2.2% 的数据量大幅降低
- **理论优雅**: 基于 MLE 框架和同方差不确定性自然推导出损失平衡机制，避免了超参数手动调整
- **可迁移性**: 评分器学习数据分布后可直接应用于新的同域数据，无需重训练
- **无需专用选择算法**: 训练后直接按分数排序选择，完全避免了复杂的选择后处理

## 局限性

- 实验仅在 LLaVA-1.5-7B 上验证，未覆盖更大规模或更新架构的 MLLM
- 谱聚类的簇数 $M$ 需要预设，可能影响多样性建模质量
- 仅评估了视觉指令微调场景，对预训练数据选择的适用性未验证
- 20% 的选择比例下 POPE 性能（85.4 vs 86.4）有轻微下降，幻觉检测可能对数据更敏感

## 相关工作与启发

- **TIVE / ICONS**: 基于梯度的数据选择方法，需要全量数据遍历，CoIDO 通过轻量评分器避免了这一开销
- **COINCIDE**: 基于聚类的方法，也使用全量数据，CoIDO 效率更高
- **Kendall et al. (CVPR 2018)**: 同方差不确定性多任务学习框架，CoIDO 将其创新性地应用于数据选择中的重要性-多样性平衡
- 启发：数据选择问题可以被建模为多目标优化问题，不确定性加权是一种自然且有效的平衡手段

## 评分

- ⭐ 新颖性: 4/5 — 联合优化框架和轻量评分器设计新颖，理论推导扎实
- ⭐ 实验充分度: 4/5 — 消融详尽，多种评分器和优化策略对比，但模型规模有限
- ⭐ 写作质量: 4/5 — 公式推导清晰，框架图直观
- ⭐ 价值: 4/5 — 为大规模 MLLM 指令微调提供了高效实用的数据选择工具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Visual Instruction Bottleneck Tuning](visual_instruction_bottleneck_tuning.md)
- [\[NeurIPS 2025\] Learning to Instruct for Visual Instruction Tuning](learning_to_instruct_for_visual_instruction_tuning.md)
- [\[ICCV 2025\] From Holistic to Localized: Local Enhanced Adapters for Efficient Visual Instruction Fine-Tuning](../../ICCV2025/multimodal_vlm/from_holistic_to_localized_local_enhanced_adapters_for_efficient_visual_instruct.md)
- [\[ICML 2025\] Parrot: Multilingual Visual Instruction Tuning](../../ICML2025/multimodal_vlm/parrot_multilingual_visual_instruction_tuning.md)
- [\[ICCV 2025\] Mastering Collaborative Multi-modal Data Selection: A Focus on Informativeness, Uniqueness, and Representativeness](../../ICCV2025/multimodal_vlm/mastering_collaborative_multi-modal_data_selection_a_focus_on_informativeness_un.md)

</div>

<!-- RELATED:END -->
