---
title: >-
  [论文解读] KLAS: Using Similarity to Stitch Neural Networks for Improved Accuracy-Efficiency Tradeoffs
description: >-
  [ICLR 2026][模型压缩][model stitching] KLAS 用 **KL 散度** 度量预训练模型中间表示的相似性，自动从 $O(k^2n^2)$ 种缝合配置里挑出最优的"锚点+块对"，在与基线相同的微调成本下把缝合网络的精度-效率曲线整体抬高（ImageNet-1K 同算力下 +1.21% top-1，或同精度下省 1.33× FLOPs）。
tags:
  - "ICLR 2026"
  - "模型压缩"
  - "model stitching"
  - "KL divergence"
  - "accuracy-efficiency tradeoff"
  - "many-to-many NAS"
  - "linear probe"
---

# KLAS: Using Similarity to Stitch Neural Networks for Improved Accuracy-Efficiency Tradeoffs

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=iTaQmRWa7Y](https://openreview.net/forum?id=iTaQmRWa7Y)  
**代码**: 待确认  
**领域**: 模型压缩 / 模型缝合 / 神经架构搜索  
**关键词**: model stitching, KL divergence, accuracy-efficiency tradeoff, many-to-many NAS, linear probe  

## 一句话总结
KLAS 用 **KL 散度** 度量预训练模型中间表示的相似性，自动从 $O(k^2n^2)$ 种缝合配置里挑出最优的"锚点+块对"，在与基线相同的微调成本下把缝合网络的精度-效率曲线整体抬高（ImageNet-1K 同算力下 +1.21% top-1，或同精度下省 1.33× FLOPs）。

## 研究背景与动机

**领域现状**：面对从云服务器到边缘设备的多样化部署目标，需要在给定算力预算下灵活选模型。传统 NAS 成本高昂；one-shot/zero-shot NAS 虽降本，但局限在单一设计空间、无法复用现成的预训练模型库。近年的"多对多 NAS + 模型缝合（model stitching）"——以 SN-Net 为代表——把不同预训练"锚点（anchor）"模型的块拼起来，用一层轻量线性 stitching layer 把一个网络的中间激活喂给另一个网络，从而以极低成本插值出一整条精度-效率谱上的可部署模型。

**现有痛点**：缝合的关键在于"选配置"——选哪些锚点、缝哪两个块。即便只有两个深度为 $n$ 的锚点，配置空间也有 $O(n^2)$ 种，穷举不可行。SN-Net 只能靠**朴素启发式**：锚点用"最近缝合（nearest stitching，只连复杂度相邻的模型）"，块用"paired/unpaired"假设。这些启发式忽略了块之间真实的兼容性，导致：(1) 精度-效率折中次优；(2) 跨模型族不通用。论文实验直接打脸——在 Swin 上跳过中间的 S、直接"远缝合" Ti-B，反而比 SN-Net 的 Ti-S-B 在同 FLOPs 下高 0.9%。

**核心矛盾**：要选出好缝合，必须**显式刻画两个被缝模型的相似性**，但现有相似度指标都不靠谱。论文实证发现（Tab.1）：CKA、MSE、CE、DM 乃至 SN-Net 启发式，对 Swin Ti-B 最优配置的恢复重叠率最高只有 61.1%，CKA 甚至只有 5.5%。原因在于它们只抓住了"表征相似"或"功能相似"中的**一半**。

**本文目标**：设计一个**自动、可泛化、零额外微调成本**的缝合配置选择框架，同时满足表征对齐（让 stitching layer 易学）和功能对齐（保住下游精度）。

**核心 idea**：**用 KL 散度作为统一相似度量**。KL 散度植根于信息论，既度量中间激活分布的统计距离（表征相似），又度量类判别信息的保留程度（功能相似），天然"一石二鸟"——这正是 MSE/CE/CKA 各自欠缺的那一半。

## 方法详解

### 整体框架
KLAS（KL divergence based Anchor Stitching）把缝合拆成"先选锚点、再选块对"两步，全程只靠在每个块后挂一个**线性探针（linear probe）**估出的 softmax 分布算 KL 散度来打分，完全不需要真的实例化或训练 stitching layer，因此评估候选配置零微调成本。打分挑出候选集后，才进入和 SN-Net 完全相同的缝合超网微调阶段。

```mermaid
flowchart LR
    A[预训练锚点池<br/>Swin/DeiT/ResNet...] --> B[ProbeNet:每块挂1×1线性探针<br/>一次前后向联合训练全部探针]
    B --> C[锚点选择<br/>末块 softmax 间 KL 散度最小→最兼容]
    C --> D[块对选择<br/>stitch score Γ=Ω/Σ + 分桶+阈值τ]
    D --> E[候选缝合配置集 S]
    E --> F[缝合超网微调 50 epoch<br/>同 SN-Net 设置]
    F --> G[精度-效率 Pareto 前沿]
```

### 关键设计

**1. 用 KL 散度统一表征相似与功能相似：缝合配置的"真"打分器。** 给定源网络 $f$ 的前 $i$ 块 $f_{\le i}$ 和目标网络 $g$ 的后段 $g_{>j}$，缝合网络是 $g_{>j}\circ T\circ f_{\le i}$，其中 $T$ 是把源激活 $A^f_i$ 映到目标激活 $A^g_j$ 的 stitching layer。KLAS 的洞察是：**缝合能否成功，取决于源、目标激活分布是否接近**——分布越近，$T$ 只需做轻微仿射变换（表征对齐），同时源块输出已保留足够的类判别信息让目标维持原决策边界（功能对齐）。KL 散度同时刻画这两点：低 $D_{KL}$ 意味着既好学又不掉精度。这是 KLAS 相比"只优化表征"（CKA）或"只优化任务损失"（TLM）的本质优势。

**2. ProbeNet：一次前后向联合训练所有线性探针，把探测成本压到近乎零。** 要算任意块对的 KL 散度，需要把每个中间块的激活投影到输出类空间。KLAS 在每个块后挂一个 $1\times1$ 卷积探针（Swin-B 需 24 个、DeiT-S 需 12 个），但逐个独立训练会很贵。论文提出 **ProbeNet** 统一架构：在单次前向-反向中每次只激活一个探针，从而把"训 24 个探针"的成本从 $24\times0.25$ GPU-day 压到 **0.25 GPU-day**。探针收敛极快（第 4 个 epoch 即可，Fig.3），是一次性可忽略开销。训练后用验证集上的平均 KL 散度作为块对相似度分数：

$$\Theta(P^f_i, P^g_j)=\frac{\sum_{x\in D_v} D_{KL}\!\left(P^f_i(x)\,\|\,P^g_j(x)\right)}{|D_v|}$$

其中 $P^f_i(x)$ 是块 $i$ 后探针输出的 softmax 分布。

**3. 锚点选择 = 末块 KL 散度最小。** 缝合第一步是挑哪两个预训练模型当锚点。KLAS 直接比较各锚点**最后一块** softmax 分布间的 KL 散度：值越低说明两模型决策边界、置信度分布越像，越适合缝合。这一步连探针都不用训（末块本来就输出 softmax）。实测（Tab.3）证明它能跨族给出正确答案——Swin 族里 Ti-B 的末块 KL 最小（$2.38\times10^{-4}$），对应最优曲线；DeiT 族里却是 Ti-S/S-B 最小。这恰好解释了"最近缝合启发式并非普适"：KLAS 用一个统一准则就自适应地在不同模型族给出不同的最优锚点对。

**4. 块对选择 = stitch score $\Gamma=\Omega/\Sigma$，分桶+阈值保覆盖。** 选好锚点后，KLAS 给每个块对 $(i,j)$ 算一个 stitch score：

$$\Gamma(i,j)=\frac{\overbrace{\Theta(P^f_i, P^g_j)}^{\Omega:\text{跨锚点激活距离}}}{\underbrace{\Theta(P^g_j, P^g_{j+1})}_{\Sigma:\text{锚内块容量}}}$$

分子 $\Omega$ 衡量假想 stitching layer 要把源块 $i$ 激活搬到目标块 $j$ 需多大变换——越小越好缝；分母 $\Sigma$ 衡量目标锚内块 $j\to j+1$ 的表征变化幅度，$\Sigma$ 大代表块 $j+1$ 学习容量高、能"吸收"源带来的分布失配。$\Gamma$ 越小越优。为产出一条密度可控的 Pareto 前沿，KLAS 把 $f$、$g$ 之间的算力空间切成若干 FLOPs 桶 $B$，每个桶里既取 $\Gamma$ 低于"桶内最小值 $\times(1+\tau)$"的所有配置，又强制至少保留桶内 $\arg\min\Gamma$ 一个配置，保证覆盖：

$$S=\bigcup_{b\in B} R^*_b,\quad R^*_b=\left\{(i,j)\,|\,\Gamma(i,j)\le\tau\right\}\cup\left\{\arg\min_{(i,j)\in b}\Gamma(i,j)\right\}$$

阈值 $\tau$ 平衡"前沿稀疏度 vs 噪声配置"，默认 5%。

## 实验关键数据

设置：ImageNet-1K / CIFAR-100，锚点含 DeiT(Ti/S/B)、Swin(Ti/S/B)、LeViT、ResNet，及 ResNet-Swin 跨架构；缝合后统一微调 50 epoch（与 SN-Net 相同）。Swin 族端到端开销在 8×A40 上仅 16 小时。

### 主实验表格

| 任务 / 族 | 指标 | SN-Net | KLAS | 改进 |
|---|---|---|---|---|
| ImageNet-1K Swin | top-1 同算力 | — | — | **+1.21%**（或同精度省 **1.33× FLOPs**） |
| Swin AUC | ∆AUC | 0.8345 | 0.8950 | **+0.06** |
| DeiT | ∆AUC | — | — | +0.012 |
| LeViT | ∆AUC | — | — | +0.006 |
| ResNet | ∆AUC | — | — | +0.014 |
| ResNet-Swin（跨架构） | ∆AUC | — | — | +0.002 |
| Swin CIFAR-100 | ∆AUC | — | — | +0.014 |
| ADE20K 分割（Mask2Former, mIoU） | 同算力桶 | 32.6 | **33.5** | 最高 **+0.9% mIoU** |
| LLM（LLaMA 1B→3B, TruthfulQA） | ROUGE-1/2 | ESTA 0.631/0.353 | **0.645/0.379**（更少参数） | +0.017 / +0.033 |

相似度指标对比（Swin 族，AUC，Tab.2）：KLAS **0.8950** > SN-Net 0.8345 > CKA 0.8124 > CE 0.8023 > DM 0.7642 > MSE 0.7564。配置恢复重叠率（Tab.1）：KL 散度 **88.9%** vs SN-Net 61.1% vs CKA 仅 5.5%。

### 消融实验表格

| 阈值 $\tau$ | Avg Top-1(%) | AUC | | #桶 | Avg Top-1(%) | AUC |
|---|---|---|---|---|---|---|
| 1% | 83.72 | 0.8934 | | 10 | 83.65 | 0.8902 |
| 3% | 83.74 | 0.8942 | | 15 | 83.75 | 0.8947 |
| **5%** | **83.76** | **0.8950** | | **20** | **83.76** | **0.8950** |
| 10% | 83.69 | 0.8931 | | | | |

### 关键发现
- $\tau$ 太大引入噪声配置、太小导致前沿稀疏，5% 最优；桶粒度 10–20 之间整体不敏感，KLAS 鲁棒。
- KLAS 既能**自动恢复** SN-Net 启发式认为有效的配置（Swin 重叠 28/38、DeiT 40/66），又能**发现新配置**（如 Swin-Ti→Swin-B 这种被最近缝合启发式漏掉的远缝合）。
- 即便**固定锚点为 SN-Net 选的 Ti-S-B**，仅靠 KL 块选择，KLAS 的曲线仍优于 SN-Net（Fig.6）——说明增益不只来自锚点选择。

## 亮点与洞察
- **把"缝哪里"从手工启发式变成可计算的相似度问题**，并指出关键是同时满足表征+功能两种相似——这个二元视角解释了为何 CKA/MSE 单打独斗都失败，很有说服力。
- **零微调成本评估候选**：用线性探针的 softmax 分布算 KL，不必真的实例化 stitching layer，这是 KLAS 在"探索海量配置"时仍便宜的根本原因。
- **泛化性强**：从 ViT 到 CNN、到 ViT-CNN 跨架构、到分类/分割/LLM 三类任务都正增益，证明 KL 散度作为缝合相似度量是通用的。
- $\Gamma=\Omega/\Sigma$ 的设计直觉漂亮：分子管"好不好缝"，分母管"目标块能不能扛住失配"，把缝合难度和目标块容量同时编码进一个标量。

## 局限与展望
- 增益绝对值偏小：多数族 ∆AUC 在 +0.006~+0.014，跨架构 ResNet-Swin 仅 +0.002，部分场景提升接近噪声范围。
- 缝合方向被固定为"源复杂度低 < 目标复杂度高"，未探索反向或多源融合。
- LLM 实验仅 LLaMA 1B→3B + TruthfulQA 单数据集，规模与广度有限。
- 论文自身指出：生成式负载有 prefill/decode 双阶段延迟约束（TTFT、TBT）和 KV-cache 兼容问题，当前相似度量未考虑，是明确的未来方向。

## 相关工作与启发
- **模型缝合谱系**：从 Lenc & Vedaldi(2015)/Bansal(2021) 把缝合当表征分析工具，到 SN-Net 把它用于 NAS（训练成本降 22×），再到 ESTA/StitchLLM 把同套启发式搬到 LLM——KLAS 是第一个把"选配置"本身做成有原则的相似度问题的工作。
- **相似度量负面结论的回应**：Csiszárik(2021) 和 Balogh & Jelasity(2025) 都发现 CKA 等常见相似度与缝合精度无相关，主张 task loss matching。KLAS 反其道证明：换成 KL 散度（同时编码功能信息），相似度量确实能预测缝合质量。
- **启发**：探针+分布散度这套"零成本评估拼接兼容性"的思路，可迁移到模型合并（model merging）、早退/级联、模块化复用等任何需要判断"两段网络能否对接"的场景。

## 评分
- **新颖性**: ⭐⭐⭐⭐ 把缝合配置选择重构为 KL 散度相似度问题，二元相似（表征+功能）视角清晰，正面回应了"相似度量无用"的既有负面结论。
- **实验充分度**: ⭐⭐⭐⭐ 覆盖 ViT/CNN/跨架构 + 分类/分割/LLM + 多数据集，消融完整；但部分族增益偏小、LLM 验证略单薄。
- **写作质量**: ⭐⭐⭐⭐ 动机层层递进（启发式失败→指标失败→KL 一石二鸟），公式与图表配合到位。
- **价值**: ⭐⭐⭐⭐ 给"复用预训练模型库做灵活部署"提供了通用、低成本、可泛化的配置选择器，工程落地友好。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Adaptive Width Neural Networks](adaptive_width_neural_networks.md)
- [\[ICML 2026\] Partial Fusion of Neural Networks: Efficient Tradeoffs Between Ensembles and Weight Aggregation](../../ICML2026/model_compression/partial_fusion_of_neural_networks_efficient_tradeoffs_between_ensembles_and_weig.md)
- [\[ICLR 2026\] A Recovery Guarantee for Sparse Neural Networks](a_recovery_guarantee_for_sparse_neural_networks.md)
- [\[ICLR 2026\] Fine-tuning Quantized Neural Networks with Zeroth-order Optimization](fine-tuning_quantized_neural_networks_with_zeroth-order_optimization.md)
- [\[ICLR 2026\] BEP: A Binary Error Propagation Algorithm for Binary Neural Networks Training](bep_a_binary_error_propagation_algorithm_for_binary_neural_networks_training.md)

</div>

<!-- RELATED:END -->
