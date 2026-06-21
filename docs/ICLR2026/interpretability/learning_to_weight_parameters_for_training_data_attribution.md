---
title: >-
  [论文解读] Learning to Weight Parameters for Training Data Attribution
description: >-
  [ICLR 2026][可解释性][data attribution] 本文指出梯度归因里"不同参数组的归因质量差异巨大"被现有方法忽视，提出一个统一框架用**自监督**直接从数据里学一组参数组权重 $w$，无需标注就把 TracIn / TRAK / EK-FAC 等方法的归因精度系统性拉高，并能解耦 subject/style/background 等语义维度。
tags:
  - "ICLR 2026"
  - "可解释性"
  - "data attribution"
  - "influence functions"
  - "gradient-based attribution"
  - "parameter heterogeneity"
  - "自监督学习"
  - "扩散模型"
---

# Learning to Weight Parameters for Training Data Attribution

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=EhUkQp9Yah](https://openreview.net/forum?id=EhUkQp9Yah)  
**代码**: 待确认  
**领域**: 可解释性 / 训练数据归因（Data Attribution）  
**关键词**: data attribution, influence functions, gradient-based attribution, parameter heterogeneity, self-supervised, diffusion models  

## 一句话总结
本文指出梯度归因里"不同参数组的归因质量差异巨大"被现有方法忽视，提出一个统一框架用**自监督**直接从数据里学一组参数组权重 $w$，无需标注就把 TracIn / TRAK / EK-FAC 等方法的归因精度系统性拉高，并能解耦 subject/style/background 等语义维度。

## 研究背景与动机

**领域现状**：训练数据归因（data attribution）要回答"模型这个输出，是哪些训练样本最该负责"，对版权、隐私、数据治理至关重要。可扩展的主流做法是梯度类方法：TracIn 直接算 query 与训练样本的梯度内积；Influence Functions / TRAK 进一步引入 Hessian（或其低秩/Kronecker 近似 + 随机投影）来对梯度做二阶预处理。

**现有痛点**：这些方法对参数的处理要么**一视同仁**（TracIn 把所有参数的梯度等权拼接），要么靠 Hessian 近似做**隐式加权**。但 Hessian 在大模型上不可解、模型也很少真正收敛到最优、再叠加随机投影的信息损失——隐式加权来自一个"不精确且带噪"的信号，并不能可靠反映各参数组的真实重要性。

**核心矛盾**：作者用 LDS（Linear Datamodeling Score）实测发现，归因信号在网络里**高度不均匀**：UNet 里 up_blocks.1/2 的平均 LDS（5.58）远高于其它块，self-attention 的输出投影层（attn.to_out）显著强于 cross-attention 的 k/q/v 投影；更进一步，不同参数组还专门负责不同语义——style 归因集中在浅层，background 归因集中在特定 attention 组件。也就是说，**归因质量随参数位置和功能系统性变化，而这恰恰是现有等权/隐式加权方案没有利用的信息**。

**本文目标**：不再依赖噪声 Hessian 近似，而是**显式、直接地从数据中学习每个参数组的重要性权重**，让任意梯度归因方法都能即插即用地变强，同时让权重本身可解释。

**核心 idea**：**[显式参数加权 + 自监督 bootstrap]** 把归因写成"对角权重矩阵 $\mathrm{Diag}(w)$ 调制的加权相似度"这一统一形式，再用现有方法的 top-k 排名当伪正样本，自监督地优化 $w$ 去最大化归因信噪比。

## 方法详解

### 整体框架
方法分两步。第一步是**统一加权形式**：把模型参数切成 $M$ 个不相交组（按层/张量），给每组一个非负标量权重 $w_j$，让归因得分变成被 $\mathrm{Diag}(w)$ 调制的加权相似度，这一形式同时涵盖 TracIn（核为单位阵）和 TRAK 类核方法。第二步是**自监督学权重**：没有归因 ground-truth，就用 base 方法当前排名的 top-k 当伪正样本，构造一个"信号/噪声"比例的损失去优化 $w$，迭代中 top-k 集合随权重动态刷新，从弱信号里 bootstrap 出越来越好的权重。

```mermaid
flowchart TD
    A[训练集 D + query 集 Q] --> B[base 归因方法<br/>TracIn/TRAK/EK-FAC...]
    B --> C[按参数组拆梯度特征<br/>g_j(x), 预计算各组归因贡献]
    C --> D[加权得分<br/>τ̃ = g(query)ᵀ·Diag(w)·K·g(xₙ)]
    D --> E[取 top-k 当伪正样本<br/>I_top-k(w)]
    E --> F[自监督损失 L_SSL<br/>正样本均分 / ℓ2 范数]
    F -->|softmax 参数化保证 w≥0| G[更新 w]
    G -->|每步重算 top-k| D
    G --> H[学到的参数组权重 w*<br/>可整体/可按语义 subject·style·background]
```

### 关键设计

**1. 统一的参数加权归因形式：把"该信任哪些参数组"显式写进得分**。作者把任意梯度归因抽象成一个可加权的双线性形式。设 query 与训练样本 $x_n$ 的拼接梯度特征为 $g(\cdot)=[g_1,\dots,g_M]$，引入非负权重 $w$ 后，加权得分为

$$\tilde\tau(x_{query}, x_n; w) = g(x_{query})^\top \cdot \mathrm{Diag}(w) \cdot K \cdot g(x_n),$$

其中 $\mathrm{Diag}(w)$ 把 $w_j$ 沿组 $j$ 的所有维度复制展开，$K$ 是相似度度量。当 $K=I$ 退化成 TracIn 式加权内积；当 $K=(\Phi^\top\Phi+\lambda I)^{-1}$ 就是 TRAK 这类核方法。关键的工程取舍是：权重**只乘在 query 一侧**，训练侧的 $K\,g(x_n)$ 当作固定项预计算一次——因为若两侧对称加权，每次更新 $w$ 都要重算所有训练样本的核项，代价不可承受；而在 $K=I$ 时单侧/双侧加权本就等价。直观理解，$w$ 编码的是"读取 query 梯度信号时，每个参数组有多可信"。

**2. 自监督权重学习：用 base 方法的 top-k 当伪标签去最大化信噪比**。既然拿不到真值归因，就假设"base 方法当前打分最高的 top-k 训练样本"是弱可信的伪正样本。对 query $x_{query}$，设全部加权得分向量为 $\tilde\tau(x_{query},D;w)$、其 top-k 索引集为 $I_{top\text{-}k}(w)$，损失定义为伪正样本的平均得分除以整体得分幅度：

$$L_{SSL}(w) = -\frac{1}{\lVert \tilde\tau(x_{query},D;w)\rVert_2}\Big(\frac{1}{k}\sum_{i\in I_{top\text{-}k}(w)} \tilde\tau(x_{query},x_i;w)\Big).$$

分子相当于信号强度估计、分母 $\ell_2$ 范数相当于总噪声估计，作者在附录证明最小化它等价于最大化归因得分的 SNR。优化时 $I_{top\text{-}k}(w)$ 每步随更新后的 $w$ 重新评估，使排名信号自举式变好；再对 query 分布 $Q$ 取期望并加 $\lambda\lVert w\rVert^2$ 正则，最终 $w^*=\arg\min_{w\ge0} \mathbb{E}_{x_{query}\sim Q}[L_{SSL}]+\lambda\lVert w\rVert^2$，非负性用 softmax 参数化保证。

**3. 极致高效：预计算组级贡献，整个学习"一分钟内收敛"**。效率来自两点：一是待学权重极少（每个参数组一个标量，如每层一个）；二是注意到得分对各参数组**线性可分**，于是把每组的归因贡献**预计算并缓存**，优化时只对组级标量得分施加权重，完全避免每步用加权梯度特征重算归因。这让权重学习通常一分钟内收敛。

**4. 细粒度语义归因：换 query 集就能解耦 subject/style/background**。同一套机制可学多组语义专用权重 $w_{style}, w_{subject}, w_{background}$。诀窍是构造**只强调目标属性、其余属性留空**的 query 集：例如学 style 权重时，prompt 只指定风格、不指定主体和背景，于是风格相关训练样本排名升高，优化自然把"持续贡献风格语义"的参数组权重抬高。学出的语义权重比通用权重更聚焦，呈现明显不同的分布模式。

## 实验关键数据

### 主实验表格

图像分类（ImageNet，LDS %，越高越好）：

| 方法 | ResNet-18 w/o w | ResNet-18 **w** | ViT-B/16 w/o w | ViT-B/16 **w** |
|------|------|------|------|------|
| TracIn | 11.39 | **23.92** | 9.67 | **17.63** |
| TRAK | 16.86 | **23.30** | 14.77 | **16.74** |

语言建模（WikiText-103, GPT-2-small，LDS %）：

| 方法 | w/o w | **w** |
|------|------|------|
| TracIn | 6.31 | **9.23** |
| TRAK | 12.69 | **14.63** |
| LoGRA | 11.42 | **12.86** |
| EK-FAC | 15.14 | **18.33** |

扩散模型（LDS %，四数据集，节选）：

| 方法 | ArtBench-2 w/o→w | Naruto w/o→w | SB-Pokemon w/o→w | CIFAR-2 w/o→w |
|------|------|------|------|------|
| TracIn | 17.63→**22.02** | 10.54→**13.59** | 9.34→**11.79** | 1.39→**8.48** |
| TRAK | 18.39→**22.15** | 14.61→**17.02** | 10.68→**12.24** | 8.51→**10.59** |
| D-TRAK | 22.72→**25.15** | 16.75→**17.85** | 33.88→**35.05** | 10.17→**12.18** |
| DAS | 30.47→**31.58** | 18.72→**20.44** | 33.55→**36.12** | 12.66→**13.79** |

### 消融实验表格

下游任务验证（误标检测 AUC / tail-patch，越高越好）：

| 任务 | 方法 | w/o w | **w** |
|------|------|------|------|
| 误标检测 AUC (ResNet-18) | TracIn | 54.40 | **61.46** |
| 误标检测 AUC (ViT-B/16) | TracIn | 71.27 | **83.58** |
| 误标检测 AUC (ViT-B/16) | TRAK | 80.08 | **83.48** |
| Tail-patch (WikiText) | TracIn | 4.66 | **5.60** (Δ+0.94) |
| Tail-patch (WikiText) | EK-FAC | 5.54 | **6.09** (Δ+0.55) |

细粒度归因（SB-Pokemon, D-TRAK, Recall@10 %）：学语义专用权重后，对应语义维度的召回明显优于无权重基线，验证了 subject/style/background 的可解耦性。

### 关键发现
- **普适增益**：在分类/语言/扩散三大任务、6 种 base 方法（TracIn/TRAK/EK-FAC/JourneyTRAK/D-TRAK/DAS）上加权后 LDS 一致提升，CIFAR-2 上 TracIn 从 1.39 暴涨到 8.48。
- **归因异质性是模型的内在稳定属性**：附录的余弦相似度分析显示，per-group LDS 与学到的权重在不同数据集、不同归因方法间高度一致，说明这种异质性不是特定设置的 artifact。
- **k 不敏感、训练极快**：超参 $k$ 在很宽范围内稳定，权重学习通常一分钟内收敛。

## 亮点与洞察
- **把"隐式加权"显式化**：以往 Hessian 预处理本质上也是在加权，但它来自不可靠近似；本文把这一步剥离出来直接从数据学，绕开了噪声近似链条。
- **统一框架的优雅**：一个 $\mathrm{Diag}(w)\cdot K$ 形式同时覆盖内积法和核方法，让所有梯度归因方法即插即用受益。
- **单侧加权 + 预计算**是让方法真正可扩展的关键工程洞察，把权重学习从"每步重算全量核项"降到分钟级。
- **可解释性是顺带的红利**：学出的权重直接给出"哪些层负责风格、哪些负责主体"的语义地图，归因方法第一次能做语义解耦。

## 局限与展望
- 自监督依赖 base 方法的初始排名当伪标签，若 base 方法在某任务上极差，bootstrap 的弱信号可能不足以纠偏。
- 权重粒度是"参数组级"（每层一个标量），未探索组内更细粒度（单参数/方向级）加权是否带来额外收益。
- 语义解耦实验主要在合成 SB-Pokemon 等可控数据集上，真实开放域生成中"语义"边界更模糊，专用 query 集的构造可能更难。
- 训练侧核项固定为预计算值，意味着权重无法反作用于训练侧表示，理论上是次优近似（但换来可扩展性）。

## 相关工作与启发
- **梯度归因谱系**：TracIn（梯度内积）→ Influence Functions / TRAK（Hessian/核预处理）→ 扩散专用 JourneyTRAK / D-TRAK / DAS、LLM 专用 LoGRA / TrackStar。本文与它们正交，是一层可叠加的"参数加权"增强。
- **参数重要性异质性**：在剪枝、机器遗忘、知识编辑、量化（知识定位 / 混合精度）里早被利用，但本文首次把它显式引入数据归因。
- **启发**：把"模型里不同部件承担不同功能"这一普遍观察，转化成一个轻量、自监督、可解释的加权层，是一个可推广到其它依赖梯度相似度任务（如检索、影响估计、数据筛选）的范式。

## 评分
- 新颖性: ⭐⭐⭐⭐ — 首次显式建模数据归因中的参数异质性，统一框架 + 自监督 bootstrap 思路清晰且与已有方法正交。
- 实验充分度: ⭐⭐⭐⭐ — 覆盖分类/语言/扩散三域、6 种 base 方法、多数据集，并补充误标检测/tail-patch 下游任务与语义解耦，证据扎实。
- 写作质量: ⭐⭐⭐⭐ — 动机（异质性实测）→ 统一形式 → 自监督目标的逻辑链顺畅，图表充分。
- 价值: ⭐⭐⭐⭐ — 即插即用、分钟级训练、一致涨点，且带来语义可解释性，对版权/数据治理等落地场景实用。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Influence Dynamics and Stagewise Data Attribution](influence_dynamics_and_stagewise_data_attribution.md)
- [\[ICLR 2026\] Learning to Interpret Weight Differences in Language Models](learning_to_interpret_weight_differences_in_language_models.md)
- [\[ICML 2026\] ExPLAIND: Unifying Model, Data, and Training Attribution to Study Model Behavior](../../ICML2026/interpretability/explaind_unifying_model_data_and_training_attribution_to_study_model_behavior.md)
- [\[ICLR 2026\] Learning is Forgetting: LLM Training As Lossy Compression](learning_is_forgetting_llm_training_as_lossy_compression.md)
- [\[ICLR 2026\] LORE: Jointly Learning the Intrinsic Dimensionality and Relative Similarity Structure from Ordinal Data](lore_jointly_learning_the_intrinsic_dimensionality_and_relative_similarity_struc.md)

</div>

<!-- RELATED:END -->
