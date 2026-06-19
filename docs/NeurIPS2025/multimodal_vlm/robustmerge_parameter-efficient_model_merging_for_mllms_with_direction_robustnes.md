---
title: >-
  [论文解读] RobustMerge: Parameter-Efficient Model Merging for MLLMs with Direction Robustness
description: >-
  [NeurIPS 2025 Spotlight][多模态VLM][模型合并] 本文从低秩分解的角度揭示了参数高效模块合并中"方向鲁棒性"是关键因素（而非全参数合并中的符号冲突），提出RobustMerge通过互补参数自适应缩放和跨任务归一化维持奇异值方向稳定性，在多模态生成任务上平均提升3.4%（已见任务）和4.5%（未见任务）。
tags:
  - "NeurIPS 2025 Spotlight"
  - "多模态VLM"
  - "模型合并"
  - "LoRA"
  - "参数高效微调"
  - "方向鲁棒性"
  - "多任务学习"
---

# RobustMerge: Parameter-Efficient Model Merging for MLLMs with Direction Robustness

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2502.17159](https://arxiv.org/abs/2502.17159)  
**代码**: [GitHub](https://github.com/AuroraZengfh/RobustMerge)  
**领域**: 多模态VLM  
**关键词**: 模型合并, LoRA, 参数高效微调, 方向鲁棒性, 多任务学习

## 一句话总结
本文从低秩分解的角度揭示了参数高效模块合并中"方向鲁棒性"是关键因素（而非全参数合并中的符号冲突），提出RobustMerge通过互补参数自适应缩放和跨任务归一化维持奇异值方向稳定性，在多模态生成任务上平均提升3.4%（已见任务）和4.5%（未见任务）。

## 研究背景与动机
当前大模型部署中，用户经常需要将多个针对特定任务微调的专家模型合并为一个通用模型，以获得多任务能力且避免数据泄露。随着模型规模增大（如MLLM），参数高效微调（PEFT，特别是LoRA）已成为训练专家模型的标准做法。

然而，现有模型合并方法（如Task Arithmetic、TIES-merging、DARE等）主要针对**全参数微调（FFT）**设计，核心策略是解决参数间的**符号冲突**。当这些方法直接应用于LoRA模块合并时，性能严重下降——甚至不如零样本基线。

**为什么FFT合并方法对PEFT失效？** 本文作者通过深入分析发现了关键差异：

**分布不同**：FFT参数分布窄且集中（均值附近），符号冲突确实是主要问题；而LoRA参数分布明显更宽，B矩阵近似高斯、A矩阵近似均匀分布

**奇异值差异悬殊**：LoRA模块中存在显著的头部-尾部奇异值差距，大奇异值对应的方向在合并时天然鲁棒，而小奇异值的方向极易受到干扰

**核心矛盾**：对PEFT而言，问题不是符号冲突而是**方向不稳定性**——合并时小奇异值对应的任务特异性知识方向被改变，导致性能崩溃

此外，现有高性能方法还存在可扩展性问题：AdaMerging需要验证数据，EMR-Merging需要额外存储，LoraHub需要测试时优化——这些都无法泛化到未见任务。RobustMerge的目标是一个**免训练、无需额外数据或存储、可泛化到未见任务**的参数高效合并算法。

## 方法详解

### 整体框架
RobustMerge分为两步：(1) 剪枝+互补参数缩放，(2) 跨任务归一化。将处理后的LoRA模块通过特定公式合并，得到最终的多任务模型。

### 关键设计

1. **基于幅度的参数剪枝**:

    - 将小幅度参数（而非随机或基于符号的选择）置零：$\widetilde{A} = \mathcal{M}_A(k) \odot A$，$\widetilde{B} = \mathcal{M}_B(k) \odot B$
    - 其中 $k$ 为剪枝率，$\mathcal{M}(\cdot)$ 按幅度排序保留top-k非零参数
    - 设计动机：大幅度参数更能保持低秩空间中的方向稳定，与LoRA宽分布特性匹配；对比实验证明基于符号的剪枝（TIES方式）在PEFT合并中反而最差

2. **互补参数自适应缩放（核心创新）**:

    - 利用LoRA的A、B矩阵间的**不对称关系**（B更关键，A近似正交）直接在原始低秩矩阵上调整奇异值，避免显式SVD分解
    - 缩放矩阵 $S$ 为对角矩阵，$S^i = \frac{\sum_j |A_{[i,j]}|}{\sum_j |\mathcal{M}_A{[i,j]} \odot A_{[i,j]}|}$
    - 关键效果：该系数对小奇异值施加更大的缩放倍数（因为小奇异值对应行在剪枝后损失比例更大），对大奇异值影响小，从而**自适应缩小头尾奇异值差距**
    - 设计动机：缩小奇异值差距可减轻合并时的方向不稳定性——大值方向天然鲁棒，真正需要保护的是小值方向对应的任务知识

3. **跨任务归一化**:

    - 对缩放系数在所有任务间归一化：$\widetilde{S}_n^i = S_n^i / \sum_{n=1}^N S_n^i$
    - 最终合并：$\Delta\widetilde{W} = \lambda \sum_{n=1}^N (\widetilde{B}_n \cdot \widetilde{S}_n) \cdot \sum_{n=1}^N \widetilde{A}_n$
    - 设计动机：不同任务的数据量不平衡会导致部分任务过拟合，归一化在任务间平衡系数，不仅稳定已见任务性能，还显著增强对未见任务的泛化

### 损失函数 / 训练策略
RobustMerge是**免训练**方法，所有操作都是后处理：
- 输入：N个针对不同任务微调的LoRA模块 $\{A_n, B_n\}_{n=1}^N$
- 超参数：剪枝率 $k$、缩放系数 $\lambda$（默认设为2）
- 无需任何验证数据、额外存储或训练
- 可在单张NVIDIA A6000上完成所有合并实验

## 实验关键数据

### 主实验（MM-MergeBench，8已见+4未见任务，LLaVA基础模型）

| 方法 | 已见任务均值 | 未见任务均值 | 说明 |
|------|---------|---------|------|
| Zero-Shot | 43.37 | 25.22 | 无合并基线 |
| Multi-Task | 63.62 | 36.06 | 多任务联合训练上限 |
| Task Arithmetic | 53.93 | 33.31 | 简单加法合并 |
| DARE | 53.84 | 33.15 | 随机丢弃+重缩放 |
| TIES-merging | 53.09 | 33.14 | 符号选举策略 |
| PCB-merging | 53.70 | 33.53 | 竞争平衡 |
| **RobustMerge** | **57.33 (+3.4%)** | **37.99 (+4.5%)** | 超越多任务学习 |

### 消融实验

| 组件 | 已见任务均值 | 增量 | 说明 |
|------|---------|------|------|
| 基线（无适应） | 53.93 | - | Task Arithmetic |
| + 剪枝&缩放 | 56.14 | +2.21 | 方向鲁棒性基础贡献 |
| + 剪枝&缩放 + 归一化 | **57.33** | **+3.40** | 跨任务平衡进一步提升 |

### 关键发现
- 现有FFT合并方法在PEFT场景下甚至不如简单的Task Arithmetic，说明符号冲突策略不适用于LoRA
- RobustMerge在未见任务上的提升（4.5%）甚至超过已见任务（3.4%），跨任务归一化是关键
- 在通用基准（POPE/MME/MMBench）上，RobustMerge优于所有对比方法且保持基础能力
- 视觉任务实验（CLIP-ViT-B-32合并8个数据集）：RobustMerge比零样本提升7.9%，比最佳先前方法提升4.4%
- 方向鲁棒性分析：RobustMerge显著提高了小奇异值对应向量的方向相似度，同时增强小奇异值的数值比率

## 亮点与洞察
- **理论洞见深刻**：从SVD方向鲁棒性的角度理解PEFT合并，首次揭示了"方向不稳定性"而非"符号冲突"是参数高效合并的核心问题
- **方法优雅高效**：无需显式SVD分解，利用A/B矩阵的不对称性直接在原始矩阵上操作，计算开销极小
- **未见任务泛化强**：这是第一个在无需额外数据/存储的条件下，在未见任务上也能稳定超越多任务学习的合并方法

## 局限与展望
- 未在更多PEFT结构（如Adapter、Prompt Tuning）上验证
- 理论分析限于简化的两模型合并场景，多模型复杂交互的形式化有待深入
- 未设计直接在分解矩阵上操作的专用算法（出于效率考虑），这可能是进一步提升的方向

## 相关工作与启发
- **vs TIES-merging/DARE**: 这些方法基于符号冲突假设，但PEFT参数分布更宽，符号信息对方向的影响不如幅度重要
- **vs AdaMerging/EMR-Merging**: 这些方法需要验证数据或额外存储，RobustMerge完全免训练且更泛化
- **vs LoraHub**: LoraHub需要测试时自适应优化系数，RobustMerge直接确定性地合并

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 方向鲁棒性视角新颖，从根本上解释了FFT方法对PEFT失效的原因，互补参数缩放设计巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 多模态+视觉双轨验证、12个任务基准、详尽消融（剪枝率、rank、组件、缩放策略）
- 写作质量: ⭐⭐⭐⭐ 分析逻辑清晰，可视化丰富，但符号较多需要仔细阅读
- 价值: ⭐⭐⭐⭐⭐ 为PEFT时代的模型合并提供了理论基础和实用方案，HuggingFace上海量LoRA模型可直接受益

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Parameter-Efficient Adaptation for MLLMs via Implicit Modality Decomposition](../../CVPR2026/multimodal_vlm/parameter-efficient_adaptation_for_mllms_via_implicit_modality_decomposition.md)
- [\[ACL 2025\] Transferring Textual Preferences to Vision-Language Understanding through Model Merging](../../ACL2025/multimodal_vlm/transferring_textual_preferences_to_vision-language_understanding_through_model_.md)
- [\[ACL 2025\] A Parameter-Efficient and Fine-Grained Prompt Learning for Vision-Language Models](../../ACL2025/multimodal_vlm/a_parameter-efficient_and_fine-grained_prompt_learning_for_vision-language_model.md)
- [\[CVPR 2026\] FairLLaVA: Fairness-Aware Parameter-Efficient Fine-Tuning for Large Vision-Language Models](../../CVPR2026/multimodal_vlm/fairllava_fairness-aware_parameter-efficient_fine-tuning_for_large_vision-langua.md)
- [\[NeurIPS 2025\] AdaLRS: Loss-Guided Adaptive Learning Rate Search for Efficient Foundation Model Pretraining](adalrs_lossguided_adaptive_learning_rate_search_for_efficien.md)

</div>

<!-- RELATED:END -->
