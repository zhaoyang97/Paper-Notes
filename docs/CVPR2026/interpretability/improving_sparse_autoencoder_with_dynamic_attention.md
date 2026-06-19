---
title: >-
  [论文解读] Improving Sparse Autoencoder with Dynamic Attention
description: >-
  [CVPR 2026][可解释性][稀疏自编码器] 这篇论文把稀疏自编码器（SAE）重写成一个共享概念向量的交叉注意力结构，并用 sparsemax 取代 softmax，让每个样本**按自身复杂度自动决定激活几个概念**，从而摆脱 TopK 里"K 该设多少"的老问题，在图像和文本上都拿到更低重构误差和更清晰的概念。
tags:
  - "CVPR 2026"
  - "可解释性"
  - "稀疏自编码器"
  - "sparsemax"
  - "交叉注意力"
  - "动态稀疏"
  - "概念解耦"
---

# Improving Sparse Autoencoder with Dynamic Attention

**会议**: CVPR 2026  
**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Improving_Sparse_Autoencoder_with_Dynamic_Attention_CVPR_2026_paper.html)  
**代码**: https://github.com/qyj-bkjx/Sparsemax-SAE  
**领域**: 可解释性 / 机制可解释性 / 稀疏自编码器  
**关键词**: 稀疏自编码器, sparsemax, 交叉注意力, 动态稀疏, 概念解耦

## 一句话总结
这篇论文把稀疏自编码器（SAE）重写成一个共享概念向量的交叉注意力结构，并用 sparsemax 取代 softmax，让每个样本**按自身复杂度自动决定激活几个概念**，从而摆脱 TopK 里"K 该设多少"的老问题，在图像和文本上都拿到更低重构误差和更清晰的概念。

## 研究背景与动机

**领域现状**：大模型里的神经元是"多义"的（一个神经元同时响应多个无关概念，即 superposition）。稀疏自编码器把激活解耦成一组稀疏、单义、可解释的概念，是当前机制可解释性的主流工具。

**现有痛点**：SAE 的核心难题是"每个特征该用几个概念"——给太多概念损害可解释性，给太少又损害重构，两边都会让概念学得不好。现有激活函数各有硬伤：ReLU 系（含 GatedReLU/JumpReLU）要配 L1/L0 正则，且 L1 会导致 feature shrinkage（激活整体被拉向 0），平衡系数还得手调；TopK / BatchTopK 直接保留最大的 K 个概念、其余清零，省了正则但把 K 当超参，**K 设错就会在复杂样本上漏概念、在简单样本上塞死概念（dead concepts）**。

**核心矛盾**：稀疏度本该是**数据依赖**的（复杂图像需要更多概念、简单图像更少），但 ReLU 的正则和 TopK 的固定 K 都是"一刀切"的全局设定，无法逐样本自适应。

**本文目标**：设计一种 SAE，让稀疏度**按每个样本的内容复杂度自动确定**，且只用重构损失训练、不依赖额外正则或 K 调参。

**切入角度**：作者注意到 sparsemax 这个激活——它把输入投影到概率单纯形上、能给低分项**精确赋 0**，且处处可微、有闭式阈值解。这正好和 SAE"稀疏激活"的诉求同源：阈值 $\tau$ 可以由样本自身算出来，相当于逐样本的动态 K。

**核心 idea**：用交叉注意力框架重写 SAE（特征当 query、字典概念当 key/value，编码解码共享同一组概念向量），并把注意力里的 softmax 换成 sparsemax，让稀疏度动态自适应。

## 方法详解

### 整体框架
传统 SAE 是单层 MLP 编码-解码：$z=\sigma(W_{enc}(x-b_{enc})),\ \hat{x}=W_{dec}z+b_{dec}$，把多义特征 $x\in\mathbb{R}^d$ 解成 $M\gg d$ 个概念 $C=\{c_1,\dots,c_M\}$ 的稀疏组合，$W_{dec}$ 的列就是概念，$\sigma$ 决定稀疏模式。本文在两点上改造它：(1) 用交叉注意力把编码器和解码器**用同一组概念向量连起来**，而不是两个独立 MLP；(2) 把注意力里的 softmax 换成 sparsemax，让每个样本动态决定激活几个概念。整个模型只用重构损失训练，无需稀疏正则、无需调 K。这是一个单次前向的机制改造（非多阶段 pipeline），下面直接用公式讲清两个设计。

### 关键设计

**1. Transformer 化 SAE：用共享概念向量连接编解码**

针对传统 SAE 把 $W_{enc}$、$W_{dec}$ 当两个独立投影、导致编码权重和解码概念脱节的问题，作者把 SAE 重写成交叉注意力：把待学字典当一组概念向量，经投影同时充当 key 和 value；把每个潜在特征当 query，做交叉注意力得到重构特征：

$$Q=x^\top W_Q,\quad K=C^\top W_K,\quad V=C^\top W_V,\quad \hat{x}=\sigma\!\left(\frac{QK^\top}{\sqrt{d}}\right)V$$

注意力权重的计算天然就是 SAE 的"编码"阶段——它度量 query 与各概念的相关性分数（$z$ 越高表示特征与概念在嵌入空间越近）；用这些权重加权 value（同一组概念）就是"解码"。关键在于 **key 和 value 来自同一个概念集 $C$**：编码时算相关性用的概念、解码时加权重构用的概念是同一批，于是权重和概念在加权（解码）阶段强协同，比 MLP 式 SAE 把 $W_{enc}/W_{dec}$ 当两套独立参数更连贯，重构能力更强、概念质量更高。

**2. Sparsemax 注意力：逐样本动态决定激活概念数**

针对 TopK 把 K 写死、softmax 又输出稠密分布的问题，作者用 sparsemax 替换注意力里的 softmax。设 $z=QK^\top\in\mathbb{R}^M$ 是 query 与 $M$ 个概念的相似度，sparsemax 把 $z$ 投影到概率单纯形上、取欧氏距离最近的点：

$$\text{sparsemax}(z)=\arg\min_{p\in\Delta^{M-1}}\|p-z\|^2$$

它的闭式解是软阈值 $\text{sparsemax}(z)_m=\max(z_m-\tau,0)$，阈值 $\tau$ 由"被选中项之和为 1"这一约束解出：把 $z$ 降序排成 $z_{(1)}\ge\cdots\ge z_{(M)}$，取 $k=\max\{r: z_{(r)}+\frac{1-\sum_{i=1}^r z_{(i)}}{r}>0\}$，则 $\tau=\frac{\sum_{i=1}^k z_{(i)}-1}{k}$。和 TopK 设死阈值不同，这里的 $\tau$ 是按输入内容复杂度**动态算**出来的：query 特征若包含多个概念，$z$ 里会有很多接近的值、支撑集 $S$（即被激活的概念集）就大；若是纯概念，$S$ 很小。论文示例里 sparsemax 会给复杂图像分配 6 个概念、给简单图像只分 2 个。sparsemax 处处可微、有良定义雅可比，能直接梯度优化（它其实是 $\alpha$-entmax 在 $\alpha=2$ 时的特例），因此可以看作"样本级的 BatchTopK"——把 K 从 batch 级细化到样本级，更灵活也更准。

### 一个例子：复杂图 vs 简单图
对一张内容丰富的复杂图像，query 与字典里多个概念都高相关，$z$ 中出现一批量级相近的大值，sparsemax 解出的阈值 $\tau$ 较低、支撑集 $S$ 较大（如激活 6 个概念）；对一张内容单一的简单图像，只有少数概念高相关，$z$ 里大值很少，$\tau$ 相对更"切"、$S$ 很小（如只激活 2 个概念）。同一个模型、同一组参数，激活数随图自适应——这是 TopK 固定 K 做不到的。

### 损失函数 / 训练策略
只用重构损失训练，不加任何稀疏正则、不调 K。视觉侧用 CLIP ViT-B/16，取倒数第二层注意力残差流输出，按 PatchSAE 设概念数 $M=49152$（ViT 隐维的 64 倍），ImageNet 上训练、batch 32、共喂 2,621,440 个 patch。文本侧用 GPT-2 Small，取第 8 层残差流，OpenWebText 上训练、序列长 128、batch 128、共喂 $10^9$ token，字典 $M\in\{3072,6144,12288,24576\}$。统一 Adam，lr=$3\times10^{-4}$，$\beta_1=0.9,\beta_2=0.99$；对比基线按各自论文取 K=32（TopK 系）、稀疏权重 1e-3（ReLU 系）。

## 实验关键数据

### 主实验
文本重构用 NMSE（归一化均方误差，越低越好）和 CE degradation（把 GPT-2 中间特征换成 SAE 重构后输出的交叉熵退化，越接近 0 越好）衡量。下表为 OpenWeb 上不同字典规模 $M$ 的 NMSE：

| Method | M=3072 | M=6144 | M=12288 | M=24576 |
|--------|--------|--------|---------|---------|
| ReLU | 0.064 | 0.064 | 0.064 | 0.059 |
| JumpReLU | 0.051 | 0.050 | 0.050 | 0.051 |
| Gated | 0.078 | 0.092 | 0.129 | 0.489 |
| TopK | 0.014 | 0.059 | 0.010 | 0.055 |
| BatchTopK | 0.014 | 0.061 | 0.060 | 0.060 |
| **Sparsemax SAE (Ours)** | **0.005** | **0.038** | **0.004** | **0.039** |

跨所有字典规模，Sparsemax SAE 的 NMSE 显著低于所有基线（在 WikiText-103 上同样如此），CE degradation 也更小——说明动态稀疏注意力既能把多义特征解成可解释概念，又能用更低信息损失重构输入。零样本图像分类（用 top-n 概念替换 ViT 中间嵌入做 11 数据集分类）上，Sparsemax SAE 在所有 top-n 设置（n=1/5/10/50）的平均表现最佳，尤其在极小 n（1/5/10）时明显领先次优。

### 消融实验
ImageNet 上拆开"transformer 架构"和"sparsemax 激活"两个贡献（top-n 概念分类准确率）：

| 配置 | on 1 | on 5 | on 10 | on 50 |
|------|------|------|-------|-------|
| ReLU SAE | 3.12 | 15.83 | 22.17 | 34.87 |
| Transformer + ReLU | 3.86 | 16.85 | 24.08 | 36.33 |
| MLP + Sparsemax | 7.91 | 29.87 | 39.73 | 55.32 |
| **Sparsemax SAE (Ours)** | **10.93** | **33.47** | **42.13** | **59.95** |

### 关键发现
- **两个设计都正向、且 sparsemax 是主力**：从 ReLU SAE 单独加 transformer（→Transformer+ReLU）只小涨，单独换 sparsemax（→MLP+Sparsemax）大涨（on 1 从 3.12 升到 7.91），二者叠加（完整模型）最好（on 1 达 10.93），说明动态稀疏激活贡献最大、共享概念的交叉注意力架构进一步加成。
- **能反哺现有 SAE 选 K**：Sparsemax SAE 算出的逐样本稀疏度可作为 TopK 系 SAE 的调参向导（在 Food101 上，Sparsemax 的 on-1 准确率 26.11 远超固定 K=24/32 的 TopK/BatchTopK 的 0.99~8.64），即"动态 K"可以指导"固定 K"该设多少。
- **概念更干净**：可视化显示，相比 BatchTopK，Sparsemax SAE 学到的概念掩码图和 top-5 参考图更清晰、更可解释；在 EuroSAT、DTD 这类与预训练自然图差异大的数据上，SAE 概念甚至超过原始 CLIP，说明学到的概念有泛化性。

## 亮点与洞察
- **把"稀疏度选择"从超参变成模型的内生计算**：sparsemax 的阈值 $\tau$ 有闭式解、随样本复杂度自动浮动，等于把 BatchTopK 的"batch 级 K"细化到"样本级 K"，这是从"调参"到"自适应"的范式转变。
- **共享概念向量连接编解码**很巧：传统 SAE 的编码权重和解码概念是两套参数、容易脱节；用交叉注意力让 key/value 同源于一组概念，编码相关性和解码重构天然协同——这个"权重即相似度、字典即 key/value"的视角值得借鉴。
- **可迁移点**：sparsemax 替 softmax 这一招，凡是"需要稀疏、可解释、且稀疏度该随输入变"的注意力/路由场景（如 MoE 专家选择、检索 top-k）都可借用，且无需额外正则、处处可微好优化。

## 局限与展望
- sparsemax 是 $\alpha$-entmax 在 $\alpha=2$ 的特例，$\alpha$ 写死，未探索可学习 $\alpha$ 或其它 entmax 变体是否更优。⚠️
- 阈值 $\tau$ 需对相似度分数排序求解，字典 $M$ 很大（如 49152）时排序开销和效率论文未深入分析。
- 评估集中在 CLIP ViT 和 GPT-2 Small 两类中等规模模型，能否扩到更大 LLM、扩散模型、多模态 LLM 上仍待验证。
- "动态决定概念数"虽灵活，但缺少对激活数稳定性 / 训练收敛性的系统分析，极端样本下是否会激活过多概念未讨论。

## 相关工作与启发
- **vs TopK / BatchTopK SAE**：它们靠固定 K（样本级/batch 级）选概念，K 设错就漏概念或塞死概念；Sparsemax SAE 把 K 细化到逐样本动态阈值，无需调 K，重构和概念质量都更好，还能反过来指导它们选 K。
- **vs ReLU / GatedReLU / JumpReLU SAE**：这些靠 L1/L0 正则促稀疏、L1 还引 feature shrinkage、平衡系数难调；本文只用重构损失、稀疏由 sparsemax 内生，无正则无平衡系数。
- **vs PatchSAE 等视觉 SAE**：PatchSAE 等只是把 ReLU/TopK SAE 搬到视觉域、没改架构；本文提出全新的交叉注意力 SAE 架构 + sparsemax 激活，且图像文本通用。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 交叉注意力 SAE + sparsemax 动态稀疏，从根上换掉了 SAE 的稀疏选择机制
- 实验充分度: ⭐⭐⭐⭐ 图像/文本双域、多字典规模、含架构与激活的拆分消融，但缺大模型与效率分析
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰、sparsemax 推导完整、可视化有说服力
- 价值: ⭐⭐⭐⭐ 解掉 SAE 选 K 痛点且能反哺现有方法，对机制可解释性社区实用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] SparseRM: A Lightweight Preference Modeling with Sparse Autoencoder](../../AAAI2026/interpretability/sparserm_a_lightweight_preference_modeling_with_sparse_autoencoder.md)
- [\[ICML 2026\] Rational Sparse Autoencoder](../../ICML2026/interpretability/rational_sparse_autoencoder.md)
- [\[ICML 2026\] CorrSteer: Generation-Time LLM Steering via Correlated Sparse Autoencoder Features](../../ICML2026/interpretability/corrsteer_generation-time_llm_steering_via_correlated_sparse_autoencoder_feature.md)
- [\[AAAI 2026\] Data Whitening Improves Sparse Autoencoder Learning](../../AAAI2026/interpretability/data_whitening_improves_sparse_autoencoder_learning.md)
- [\[ICLR 2026\] SALVE: Sparse Autoencoder-Latent Vector Editing for Mechanistic Control of Neural Networks](../../ICLR2026/interpretability/salve_sparse_autoencoder-latent_vector_editing_for_mechanistic_control_of_neural.md)

</div>

<!-- RELATED:END -->
