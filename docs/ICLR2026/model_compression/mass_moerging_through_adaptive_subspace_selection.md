---
title: >-
  [论文解读] MASS: MoErging through Adaptive Subspace Selection
description: >-
  [ICLR 2026][模型压缩][model merging] MASS 把每个任务更新的低秩奇异子空间存进一个共享模型，推理时用一个无需数据、无需训练的"投影残差"路由器，在不知道任务身份的情况下自动选出最匹配输入的任务子空间和分类头，把模型合并的精度推到单独微调模型的约 98%。 - 领域现状：模型合并（model m…
tags:
  - "ICLR 2026"
  - "模型压缩"
  - "model merging"
  - "MoErging"
  - "task vectors"
  - "SVD"
  - "training-free routing"
---

# MASS: MoErging through Adaptive Subspace Selection

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=CRBt6DNaBE](https://openreview.net/forum?id=CRBt6DNaBE)  
**代码**: [https://github.com/crisostomi/mass](https://github.com/crisostomi/mass)  
**领域**: 模型融合 / 多任务模型合并  
**关键词**: model merging, MoErging, task vectors, SVD, training-free routing  

## 一句话总结
MASS 把每个任务更新的低秩奇异子空间存进一个共享模型，推理时用一个无需数据、无需训练的"投影残差"路由器，在不知道任务身份的情况下自动选出最匹配输入的任务子空间和分类头，把模型合并的精度推到单独微调模型的约 98%。

## 研究背景与动机
- **领域现状**：模型合并（model merging）想把多个共享同一预训练 backbone、在不同任务上微调出来的模型，免训练地揉成一组参数。从 Task Arithmetic（把"微调权重−预训练权重"的任务向量直接相加）到 Task Singular Vectors（TSV，发现任务更新矩阵 $\Delta_i$ 是低秩的，只留每层 top-k 奇异向量就能恢复大部分精度），合并质量在稳步上升。
- **现有痛点**：① 纯合并方法（Weight Averaging / TA / TSV-M / Iso-C）始终追不上"每个任务单独微调"的精度上界；② 现有 MoErging 方法（SMILE、WeMoE、TwinMerging）虽然加了路由器，却普遍**默认推理时已知任务身份、已知正确分类头**——这是一个不现实的 oracle 假设；③ 想训练真正的路由器又往往需要原始任务数据，而合并场景里这些数据常常拿不到（模型是从 HuggingFace 下载来的）。
- **核心矛盾**：要么假设知道任务（不现实），要么就回退到压缩（如 TSV-C 用 2× 存储拿到 99.5% 归一化精度，几乎"解决"了已知任务的设定）。真正难且有意义的设定是：**任务未知，既要自动选对编码器子空间，也要自动选对分类头**。
- **本文目标**：造一个单一通才模型，在没有外部监督、不知道任务的前提下处理所有微调过的任务。
- **核心 idea**：**[免数据、免训练的权重空间路由]** TSV-M 已经把各任务的 top 奇异方向编进了一个共享模型的正交子空间里，那么"判断输入属于哪个任务"就退化成"测量输入激活被哪个子空间重建得最好"——一次正交投影残差就够了，既不需要标签也不需要再训练。

## 方法详解

### 整体框架
MASS 分两段：**固定合并**（一次性预处理）用 TSV-M 把各任务的低秩奇异子空间聚合成一个有任务判别力的共享编码器 $\theta_{MT}$；**自适应合并**（推理时）对每个输入跑两趟前向——第一趟用 $\theta_{MT}$ 取中间层激活并路由选出相关任务子空间，第二趟把选中的子空间临时合进模型并跨所选分类头取最高 logit。整体只比单个预训练模型多 ~2× 存储、~2× 前向，且与任务数无关。

```mermaid
flowchart LR
    A[输入 x] --> B[第一趟前向<br/>θMT 取中间激活 zℓ]
    B --> C[投影路由<br/>对每个任务算残差 rᵢ]
    C --> D[softmax(-r)+阈值η+TopK<br/>选出任务子集 Ω]
    D --> E[自适应合并 Δada<br/>= Σ UΣVᵀ over Ω]
    E --> F[θMASS = θpre + αΔada]
    F --> G[第二趟前向<br/>跨 Ω 内各分类头取最高 logit]
    G --> H[预测类别 c*]
```

### 关键设计

**1. 投影残差路由：用重建误差当任务似然。** 给定输入在第 $\ell$ 层的中间激活 $z_\ell$，MASS 不去训练任何路由网络，而是直接把 $z_\ell$ 正交投影到每个任务右奇异向量张成的子空间 $\mathrm{span}(V_i^{(\ell)})$ 上，算欧氏残差 $r_i = \|z_\ell - V_i^{(\ell)}(V_i^{(\ell)})^\top z_\ell\|_2$。残差越小说明这个子空间越能解释输入。把 $-r$ 过一个 softmax 得到任务系数，再用阈值 $\eta$ 筛掉系数过低的任务、超出时只留 top-k。这套机制完全数据无关、训练无关，直接可用于合并场景。作者进一步证明（Prop. 3.1）：若残差服从各向同性高斯 $\varepsilon_i\sim\mathcal{N}(0,\sigma^2 I)$ 且任务先验均匀，则**选残差最小的任务等价于 MAP 估计**，和概率 PCA 的最大似然解一脉相承——在没有训练数据拟合复杂分布时，这种"最不带偏见"的各向同性先验恰好最适合 MASS。

**2. 冗余方向剔除：防止相似任务互相盖住对方。** 投影路由有个陷阱：若两个任务（如 MNIST 和 EMNIST）训练数据高度相似、右奇异方向大量重合，它们子空间的并集会在某片特征区域显得"更宽更强"，导致路由器对本属于第三个相似任务（KMNIST）的样本也判成 MNIST/EMNIST——因为 $\|z_\ell-\mathrm{Proj}_{V_{MN}\cup V_{EMN}}(z_\ell)\|_2 < \|z_\ell-\mathrm{Proj}_{V_{KM}}(z_\ell)\|_2$。MASS 在固定合并阶段做去冗余：先选一个任务矩阵做种子，逐个考察其余任务，把 $\Delta_i$ 拉平成 $\delta_i=\mathrm{vec}(\Delta_i)$，只有当它与所有已接受任务的余弦相似度 $\max_m \mathrm{sim}(\delta_i,\delta_{a_m}) \le \varepsilon$（如 $\varepsilon=0.3$）时才合入。这样高度相似的子空间不会盖过那些罕见任务，保证"没有任务能压倒其他任务"。

**3. 联合选编码器子空间与分类头：彻底丢掉 oracle。** 路由器选出任务子集 $\Omega$ 后，MASS 用 TSV-M 把这些子空间合成 $\theta_{MASS}=\theta_{pre}+\alpha\Delta_{ada}$，跑第二趟前向得到共享表征 $z_{L-1}$。和"假设 oracle 给出正确分类头"的常规合并不同，MASS 对 $\Omega$ 里每个任务的分类头 $h_i$ 都算一遍 $z_i=h_i(z_{L-1})$，然后在所有头的所有类别里取最高 logit：$(i^\star,c^\star)=\arg\max_{(i,c)\in\Omega\times\{1,\dots,C_i\}} z_i[c]$。即让最"自信"的头胜出，从而在每个输入上同时定下编码器子空间、分类头和标签空间——这正是它能处理"任务未知"设定的关键。

**4. 路由层的选择。** 残差在哪一层算很关键。实验发现 ViT-B-32 和 ViT-B-16 都在第 9 层（且 MLP 层略优于自注意力层）整体最佳，但**最优层强烈依赖任务**：STL10 在早层（3/4/5）路由更准，SUN397 反而在晚层（9/10/11）更好，单层精度跨任务方差可达 40%。这提示自适应选层是值得深挖的方向；MASS 当前用固定的最佳层（第 9 层）作为折中。

## 实验关键数据

### 主实验：视觉任务合并（CLIP，归一化精度，括号内为相对单微调上界的百分比）

| Method | ViT-B-32 (8/14/20) | ViT-B-16 (8/14/20) | ViT-L-14 (8/14/20) |
|---|---|---|---|
| Finetuned 上界 | 90.3(100)/89.0(100)/89.5(100) | 92.4/91.3/91.9 | 94.2/93.4/94.0 |
| Task Arithmetic | 68.8(75.7)/64.6/64.0 | 73.0/70.6/69.0 | 84.4/80.4/76.9 |
| TSV-M（MASS 基座） | 83.2(91.8)/78.6/75.6 | 85.5/81.4/78.8 | 91.2/88.8/87.5 |
| Iso-CTS | 82.0/80.6/77.0 | 88.7/84.1/80.7 | 92.8/91.1/89.2 |
| SMILE-2 | 84.4/76.4/74.1 | 89.0/82.7/80.4 | 92.0/87.1/85.5 |
| **MASS** | **87.0(96.5)/82.9(93.2)/81.1(90.9)** | **90.6/87.8/81.1** | **92.9/90.9/90.8** |

MASS 在 9 个基准里 8 个取得 SOTA，对最优 baseline 提升最高约 6%；相比所依赖的 TSV-M，路由带来约 5% 的提升。

### 跨模态验证：Flan-T5-Base 在 GLUE 8 任务（归一化平均）

| Method | Avg. |
|---|---|
| Task Arithmetic | 91.3 |
| SMILE-2 | 99.0 |
| **MASS** | **99.4** |

MASS 在 9 个 GLUE 子任务中 5 个拿到最高绝对精度，仅在 NLI 类任务（MNLI/QNLI）略逊——作者归因于 NLI 需要更宽泛的语义推理、任务向量更弥散、秩更高。证明方法**与模态无关**。

### 消融与对比
**路由器对比（ViT-L-14，归一化精度）**

| Router | 8 / 14 / 20 tasks | 备注 |
|---|---|---|
| nn（最近邻） | 94.0 / 92.1 / 92.0 | 需存每任务验证集 embedding |
| mlp（训练 MLP） | 98.9 / 99.5 / 98.3 | 需要带标签验证集，违背合并前提 |
| proj-PRE（从预训练 backbone 投影） | 99.1 / 97.7 / 91.9 | 任务多时崩 |
| **proj-TSV-M（本文）** | 98.6 / 97.3 / **96.6** | 免数据免训练，扩展性最好 |

**关键发现**：
- 从 TSV-M 模型投影路由（proj-TSV-M）在任务多（20）时显著优于从预训练 backbone 投影（proj-PRE），ViT-B-32 上差距达约 10%——印证核心洞察：TSV-M 已把各任务方向编成正交子空间，路由只需"找回"对应子空间。
- **存储**：MASS 恒为 2× 参数（与任务数无关），而其他 MoErging baseline 在 ~2.5×–14× 之间。
- **批量推理**：若一批样本同域，每批只需路由一次，9 个设定里 8 个达到 ≥97% 归一化精度，几乎追平单独微调模型。
- **任务扩展（2→33 任务）**：MASS 全程保持高精度，且精度非任务数的单调函数——扩展性更取决于任务集的构成而非数量，与 TA 的陡峭退化形成鲜明对比。

## 亮点与洞察
- **把"路由"重新表述成"子空间投影残差"**：因为 TSV-M 把任务方向编成正交子空间，路由器无需学习、无需数据，只是一次线性代数运算，优雅且零成本。
- **MAP 视角**：残差最小化 = 各向同性高斯下的 MAP 估计，给免训练路由器一个干净的概率论解释。
- **直面 oracle 假设**：第一个在"任务未知 + 分类头未知"的更现实设定下做 MoErging，且仍超过那些假设已知任务的 fixed-merging baseline。
- **恒定 2× 存储**：不随任务数膨胀，相比动辄十几倍存储的 MoE 路由方法是实打实的工程优势。

## 局限与展望
- **路由层固定**：最优层强烈依赖任务（单层跨任务方差达 40%），当前用全局最佳层（第 9 层）折中，自适应选层尚未解决。
- **两趟前向**：推理需要两次 forward，比纯压缩方法（如 TSV-C 零额外推理开销）成本更高。
- **去冗余阈值 $\varepsilon$ 手工设定**：相似任务剔除依赖人工阈值（如 0.3），缺乏自适应机制。
- **NLI 类任务偏弱**：高秩、语义弥散的任务向量上投影路由优势减弱，提示对"非局部特征位移"型任务需要更强的子空间刻画。
- 仍局限于共享同一预训练 backbone 的同构模型合并。

## 相关工作与启发
- **Task Arithmetic / Ties / Consensus TA**：任务向量加法这一系族奠定了免训练合并的基础，MASS 继承其"$\theta_{pre}+\alpha\sum\tau_i$"骨架但加上输入条件化的门控。
- **TSV-M / Iso-C（Gargiulo et al. 2025）**：发现任务更新的低秩与子空间正交结构，是 MASS 固定合并阶段和路由可行性的直接前提。
- **MoErging：SMILE / WeMoE / TwinMerging**：把路由引入合并，但都默认已知分类头；MASS 的核心区分点就是丢掉这个 oracle。
- **概率 PCA（Tipping & Bishop 1999）**：为残差最小化路由提供了最大似然的理论镜像。
- **启发**：当多个专家被嵌进同一组正交子空间后，"选专家"可以彻底退化为"投影找回子空间"——这一思路可能迁移到 LoRA 库的动态组合、跨模态专家路由等更广的 modular learning 场景。

## 评分
- **新颖性**: ⭐⭐⭐⭐ 把路由重述为子空间投影残差并给出 MAP 解释，直面"任务/头未知"的现实设定，角度新且自洽。
- **实验充分度**: ⭐⭐⭐⭐ 3 种 ViT × {8,14,20} 任务 + GLUE 跨模态 + 2→33 任务扩展 + 路由器/路由层/批量推理多维消融，覆盖扎实。
- **写作质量**: ⭐⭐⭐⭐ 动机层层递进，MNIST/EMNIST/KMNIST 的冗余方向例子直观，理论与图示配合清晰。
- **价值**: ⭐⭐⭐⭐ 恒定 2× 存储下逼近单微调上界（~98%），且免数据免训练，对模型合并的落地很有实用意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Achieving low-bit Muon through subspace preservation and grid quantization](achieving_low-bit_muon_through_subspace_preservation_and_grid_quantization.md)
- [\[CVPR 2026\] Bridging Domains through Subspace-Aware Model Merging](../../CVPR2026/model_compression/bridging_domains_through_subspace-aware_model_merging.md)
- [\[ICML 2026\] MIC: Maximizing Informational Capacity in Adaptive Representations via Isotropic Subspace Alignment](../../ICML2026/model_compression/mic_maximizing_informational_capacity_in_adaptive_representations_via_isotropic_.md)
- [\[ICLR 2026\] Efficient Orthogonal Fine-Tuning with Principal Subspace Adaptation](efficient_orthogonal_fine-tuning_with_principal_subspace_adaptation.md)
- [\[ICLR 2026\] AdaRank: Adaptive Rank Pruning for Enhanced Model Merging](adarank_adaptive_rank_pruning_for_enhanced_model_merging.md)

</div>

<!-- RELATED:END -->
