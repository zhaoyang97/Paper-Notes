---
title: >-
  [论文解读] QUEST: A Robust Attention Formulation Using Query-Modulated Spherical Attention
description: >-
  [ICLR 2026][注意力机制] QUEST 把标准缩放点积注意力中的 key 向量归一化到超球面、同时保留 query 的范数自由度（即 $A=\mathrm{softmax}(Q\bar{K}^\top)$），用一个不到一行的改动同时消除了注意力 logit 爆炸导致的训练不稳定、并让模型学到更分散、更鲁棒的注意力，在 ImageNet 分类、分割、对抗攻击等多个任务上稳定地优于标准注意力与 QKNorm。
tags:
  - "ICLR 2026"
  - "注意力机制"
  - "训练稳定性"
  - "超球面归一化"
  - "鲁棒性"
  - "Transformer"
---

# QUEST: A Robust Attention Formulation Using Query-Modulated Spherical Attention

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=HkztQWZfl2](https://openreview.net/forum?id=HkztQWZfl2)  
**代码**: 待确认  
**领域**: Transformer 架构 / 注意力机制 / 训练稳定性 / 鲁棒性  
**关键词**: 注意力机制, 训练稳定性, 超球面归一化, 鲁棒性, Vision Transformer

## 一句话总结
QUEST 把标准缩放点积注意力中的 key 向量归一化到超球面、同时保留 query 的范数自由度（即 $A=\mathrm{softmax}(Q\bar{K}^\top)$），用一个不到一行的改动同时消除了注意力 logit 爆炸导致的训练不稳定、并让模型学到更分散、更鲁棒的注意力，在 ImageNet 分类、分割、对抗攻击等多个任务上稳定地优于标准注意力与 QKNorm。

## 研究背景与动机
**领域现状**：Transformer 的核心是缩放点积注意力（SDPA）$A=\mathrm{softmax}(C\,QK^\top)$，其中 $C=1/\sqrt{D_H}$ 是固定缩放因子。这套公式被 ViT、GPT、PointTransformer、Conformer 等几乎所有变体沿用，但大模型训练时经常出现不稳定，业界靠初始化技巧、学习率调度、各种 normalization 和优化策略来缓解，却始终没把"为什么会不稳定"讲清楚。

**现有痛点**：作者把 SDPA 拆开看——令 $q=\|q\|\bar q$（范数乘单位向量），单个 token 的注意力可写成 $A_i=\mathrm{softmax}\big(C\|q_i\|\|k_j\|(\bar q_i\cdot \bar k_j)\big)_j$。这暴露出两个被忽视的角色：**query 范数 $\|q_i\|$ 缩放该 token 的所有 logit，控制其注意力分布的"锐度"**（范数大 → 注意力集中在少数 token，范数小 → 更平滑）；而 **key 范数 $\|k_j\|$ 会放大某个 token $j$ 在所有 query 眼里的得分**，让它"全局抢注意力"。当 query/key 范数在训练中任意增长，logit 就会爆炸，模型卡在次优解。

**核心矛盾**：现有的稳定化方案 QKNorm（把 query 和 key 都做 $\ell_2$ 归一化，再用可学习参数 $C_q,C_k$ 逐维或逐头重新缩放）确实能稳住训练，但它用**同一个缩放因子作用到所有 token**，强行让每个 token 的注意力锐度一致，牺牲了表达力——在小模型上甚至比标准注意力还差。于是"训练稳定"与"注意力表达力"之间形成了 trade-off。

**本文目标**：找到一个既能消除范数爆炸、又不牺牲 per-token 锐度控制的注意力公式，而且要能作为 drop-in 替换塞进任意 Transformer。

**切入角度**：既然 query 范数负责"有用的"锐度控制、key 范数负责"有害的"全局抢注意力，那为什么要把两者一起归一化？只归一化其中一边，就能在打破 query↔key 梯度交叉耦合的同时，保住一边的范数自由度。

**核心 idea**：只归一化 key、让 query 保持自由——key 被约束到超球面（排序完全由 query-key 的余弦相似度决定），query 范数继续让每个 token 独立调节自己的注意力锐度。

## 方法详解

### 整体框架
QUEST 不改变 Transformer 的任何其它结构，只替换注意力那一步的计算公式。标准多头自注意力中，每个头把输入序列 $X\in\mathbb{R}^{N\times D}$ 投影出 $Q=XW_Q^\top$、$K=XW_K^\top$、$V=XW_V^\top$，再算 $Z=\mathrm{softmax}(C\,QK^\top)V$。QUEST 把这里改成：

$$A=\mathrm{softmax}\big(Q\bar{K}^\top\big),\qquad \bar K = \text{对 }K\text{ 逐行做 }\ell_2\text{ 归一化}$$

注意两点：**只归一化 key，query 完全不动**；并且**不再用任何缩放因子**（即 $C=1$）。这样一来，注意力 logit $= \|q_i\|\,(\bar q_i\cdot \bar k_j)$ 的排序完全由超球面上 query 与 key 的余弦对齐决定，而每个 query 的范数 $\|q_i\|$ 独立控制它自己那一行 softmax 的锐度。这是一个高度可解释、且只需几行代码的修改，能直接套进 ViT、语言模型、图 Transformer、时间序列、点云等各种架构。

为了把设计动机讲透，论文还对照了一组邻近变体：标准注意力（key/query 都不归一化）、QNorm（只归一化 query）、QKNorm-HS（query/key 都归一化、每个头一个可学习标量 $C\in\mathbb{R}^H$）、QKNorm-DS（每个特征维一组可学习 $C_q,C_k\in\mathbb{R}^{D_H}$）。QUEST 处在"只归一化 key 且不加缩放"这个此前文献从未尝试过的位置上。

### 关键设计

**1. 球面化 key、自由化 query：用单边归一化同时拿下稳定性与表达力**

标准注意力之所以不稳，是因为 key 范数能任意增长、让某个 token 在所有 query 眼里都得分极高，把注意力"全局抢走"，进而推高 logit 引发爆炸。QUEST 把 key 逐行 $\ell_2$ 归一化到超球面（$\|\bar k_j\|=1$），从机制上断掉了"靠涨 key 范数抢注意力"这条路——任何 token 都不可能仅凭范数大就垄断注意力。但与 QKNorm 不同的是，QUEST **不动 query**：保留的 query 范数 $\|q_i\|$ 让每个 token 仍能独立选择"看得专注还是看得分散"，因此不会像 QKNorm 那样把所有 token 的锐度锁死、损失表达力。一句话，QUEST 砍掉了 key 范数这个"有害自由度"，保住了 query 范数这个"有用自由度"，这正是它能在稳定与表达之间两头都占的原因。

**2. 为什么是归一化 key 而不是 query：切断 query↔key 梯度的交叉耦合**

标准注意力训练崩坏的一个加速器，是 query 与 key 参数更新之间的"交叉作用"——logit 里 $\|q_i\|\|k_j\|$ 相乘，使得 query 范数的梯度依赖 key 范数、反之亦然，两者互相助推一起膨胀。归一化其中任意一边都能打破这种耦合，所以论文也认真评估了对称的另一半 QNorm（只归一化 query）。QNorm 确实减轻了交叉耦合、比标准注意力略好，但它把 key 留作自由——key 范数照样能增长、照样能全局抢注意力，因此并没有堵住真正的失败机制。QUEST 选择归一化 key，恰好命中"全局抢注意力"这个病根，同时把锐度控制权留给 query。值得一提的是，论文实验观察到：虽然只归一化了 key，query 的范数和最大 logit 也随之被稳住了——单边约束反而把整条链路都拉稳了。

**3. Elliptical QUEST：与椭圆注意力正交叠加，进一步提升鲁棒性**

QUEST 用的是超球面上的余弦相似度，而 Elliptical Attention（Nielsen et al., 2024）把标准注意力的各向同性高斯核扩展成基于 Mahalanobis 度量的超椭球。两者作用在不同层面、彼此正交，可以叠加成 Elliptical-QUEST：用椭圆度量替代纯余弦相似度来度量 query 与 key 的对齐。实验显示，Elliptical 本身鲁棒性强但会牺牲干净数据上的分类精度（71.53% vs QUEST 72.50%），而 Elliptical-QUEST 既继承了更强的对抗鲁棒性、又把干净精度拉回来，说明 QUEST 这条"key 球面化"的思路能和其它注意力改进兼容增益。

### 一个例子：玩具检索任务暴露"虚假注意力"
论文构造了一个简单检索任务来直观展示问题。输入是一串向量 $X=[x_1,\dots,x_N]$，每个向量含一个实值部分 $x^k_i$ 和一个 one-hot 的"答案"部分 $x^v_i$。除了随机位置 $L$ 的答案 token 外，所有非答案 token 的实值部分都从 $\mathcal{N}(0,I)$ 采样；答案 token 是"分布外"的（$x^k_L\sim\mathcal{N}(0,\Sigma)$，$\Sigma\neq I$），这是**永远成立的鲁棒信号**，正确模型应当学会靠它定位答案。

但作者又注入了一个**只对约一半样本成立的偏置信号**：以 $p=0.5$ 把样本标为"偏置样本"，其答案 token 改从 $\mathcal{N}(b,0.1I)$ 采样，偏置向量 $b$ 在所有偏置样本间共享。于是模型有了一条捷径——只要让 key 权重矩阵在 $b$ 方向上的放大倍数变大、把偏置答案 token 的 key 范数顶上去，就能在偏置出现时全局把注意力集中到答案位置，从而"偷懒"解决一半样本。标准注意力和 QNorm 正是栽在这里：训练中偏置答案 token 的 key 范数越长越大（论文 Figure 4 实测），模型只学会查 $b$、学不到真正的鲁棒解；QKNorm 因为完全丢掉了 key 范数里携带的"答案位置反常分布"信息，干脆退化成随机猜。QUEST 因为禁止任何 token 靠范数全局抢注意力，把整体成功率从标准注意力的 25%、QNorm 的 49%、QKNorm 的约 0% 提到 **58%**，而且在更宽的学习率/权重衰减范围内都管用。

## 实验关键数据

### 主实验
ImageNet-1K 上用 DeiT 训练 ViT-Tiny 300 epoch，对比各种 QK 归一化方案（多次训练均值）：

| 注意力 | IN-val Top-1 | IN-ReaL Top-1 | IN-C MCE ↓ | IN-A Top-1 |
|--------|--------------|---------------|------------|------------|
| Standard | 72.6 | 80.4 | 55.7 | 8.2 |
| **QUEST** | **73.4** | **81.2** | **55.0** | 8.5 |
| QNorm | 72.7 | 80.6 | 55.3 | 8.2 |
| QKNorm-HS | 72.5 | 80.5 | 56.4 | 7.9 |
| QKNorm-DS | 71.6 | 79.6 | 57.4 | 7.2 |
| QKNorm | 71.9 | 79.0 | 58.1 | 7.0 |

QUEST 在干净精度、ReaL、以及损坏鲁棒性（IN-C 的 MCE 越低越好）上全面领先；QKNorm 系列虽然能稳定大模型，但在小模型上因为锐度被锁死，干净精度反而比标准注意力还低。

更大模型上（DeiT / DeiT-3 训练）：

| 模型 | 注意力 | IN-val | IN-C MCE ↓ | 备注 |
|------|--------|--------|------------|------|
| ViT-S/16 (200ep) | Standard | 79.6 | 44.8 | — |
| ViT-S/16 (200ep) | **QUEST** | **80.2** | **43.2** | — |
| ViT-B/16 (100ep) | Standard | — | — | 训练崩溃 |
| ViT-B/16 (100ep) | QKNorm-DS | 79.0 | 44.4 | baseline |
| ViT-B/16 (100ep) | **QUEST** | **79.7** | **42.9** | 稳定 |
| ViT-L/16 (100ep) | Standard | — | — | 训练崩溃 |
| ViT-L/16 (100ep) | QKNorm-DS | 72.5 | 54.4 | baseline |
| ViT-L/16 (100ep) | **QUEST** | **74.9** | **50.3** | +2.4 |

标准注意力在 ViT-B/L 上直接发散崩溃，QUEST 则稳定收敛，且优于 QKNorm-DS；DeiT-3 长训下 ViT-B 也从 82.7 提到 83.2。

### 消融实验
对抗鲁棒性（ViT-Ti/16，DeiT 训练，Top-1）：

| 配置 | Clean | FGSM | PGD | Auto |
|------|-------|------|-----|------|
| Standard | 72.50 | 54.23 | 43.65 | 26.57 |
| **QUEST** | **73.33** | **56.90** | 45.26 | 27.29 |
| Elliptical | 71.53 | 55.96 | 46.30 | 27.35 |
| **Elliptical-QUEST** | 72.48 | 56.39 | **47.25** | **28.54** |

QUEST 在所有攻击下都比标准注意力更鲁棒且干净精度更高；Elliptical-QUEST 把两者优势叠起来，对抗鲁棒性最强、同时把 Elliptical 损失的干净精度补回来。ADE20K 分割上 ViT-Ti 干净 mIoU 37.34→38.87、损坏数据 32.19→33.55，同样既涨精度又涨鲁棒。

### 关键发现
- **只归一化 key 就能把整条链路拉稳**：虽然只约束了 key，但 query 范数和最大注意力 logit 也随之被稳住，这是标准注意力发散而 QUEST 稳定的直接原因。
- **稳定 ≠ 牺牲表达力**：QKNorm 用全局统一缩放换稳定，小模型上掉点；QUEST 保留 query 范数的 per-token 锐度，稳定与精度兼得。
- **鲁棒性提升集中在 IN-C / IN-A 以及对抗攻击**：QUEST 让注意力更均匀地铺在相关物体区域，而非死盯几个最显著的实例，因此输入被扰动/加噪时不容易误判（论文用 AG-CAM 的可解释性图佐证）。

## 亮点与洞察
- **"拆开注意力看范数角色"这一步最关键**：把 query 范数（锐度控制，有用）和 key 范数（全局抢注意力，有害）分离开，整个方法的合理性立刻清晰——单边归一化不是拍脑袋，而是精准切除有害自由度。
- **几乎零成本的 drop-in 改动**：去掉缩放因子、只对 key 做 $\ell_2$ 归一化，几行代码，跨视觉/语言/图/时间序列/点云通用，工程落地门槛极低。
- **正交可叠加**：与 Elliptical Attention 组合即得 Elliptical-QUEST，说明"key 球面化"是一个能和其它注意力改进相加的独立维度，迁移性强。
- **玩具实验设计巧妙**：用"鲁棒信号 vs 只对一半样本成立的偏置捷径"把训练不稳定/虚假相关的失败机制可视化成 key 范数的增长曲线，比单纯报指标更有说服力。

## 局限与展望
- 作者明确把 QUEST 限定在 **softmax 注意力**，未覆盖线性/无 softmax 注意力；而线性注意力同样存在熵坍缩问题，扩展留作future work。
- 实验以**视觉为主**，语言建模、图、时间序列、点云只作为"通用性"佐证，缺少大规模 LLM 预训练上的系统验证。
- 去掉缩放因子 $C=1$ 在更深/更大规模（论文最多到 2B ViT）之外是否依然稳定，仍有待更大尺度验证。
- 自己的观察：QUEST 与 Elliptical-QUEST 的鲁棒性提升在不同攻击间并不一致（如 PGD 上 Elliptical 系列更强、Auto 上 QUEST 干净精度更高），具体选型需按任务权衡，不能一概而论。

## 相关工作与启发
- **vs 标准注意力 (SDPA)**：标准注意力靠固定 $1/\sqrt{D_H}$ 缩放，仍会因 query/key 范数任意增长而 logit 爆炸；QUEST 通过球面化 key 从机制上堵死这条路，且无需任何缩放因子。
- **vs QKNorm (HS/DS)**：QKNorm 把 query/key 都归一化再加可学习缩放，能稳定大 ViT 但用全局统一缩放锁死了 per-token 锐度，小模型掉点；QUEST 只归一化 key、保留 query 范数，稳定的同时不损表达力。
- **vs QNorm（只归一化 query）**：QNorm 是 QUEST 的对称对照，能减轻交叉耦合但放任 key 范数增长、堵不住全局抢注意力，因此鲁棒性与成功率都不如 QUEST。
- **vs Elliptical Attention**：Elliptical 用 Mahalanobis 椭球度量提升鲁棒但牺牲干净精度；QUEST 与之正交，组合成 Elliptical-QUEST 可同时拿到更强鲁棒性与更高干净精度。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把"分离 query/key 范数角色 → 只归一化 key"的洞察讲得干净，填补了文献里"单边归一化"的空白
- 实验充分度: ⭐⭐⭐⭐ 覆盖分类/分割/对抗/多域、含玩具机理分析与多次重复，唯缺大规模 LLM 验证
- 写作质量: ⭐⭐⭐⭐⭐ 从范数分解一路推到方法，动机与机制讲得透彻、可解释性强
- 价值: ⭐⭐⭐⭐ 零成本 drop-in、跨域通用、可与其它注意力改进叠加，工程与研究都好用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Hilbert-Guided Sparse Local Attention](hilbert-guided_sparse_local_attention.md)
- [\[ICLR 2026\] Neural Dynamics Self-Attention for Spiking Transformers](neural_dynamics_self-attention_for_spiking_transformers.md)
- [\[CVPR 2026\] Graph Attention Prototypical Network for Robust Few-Shot Classification](../../CVPR2026/others/graph_attention_prototypical_network_for_robust_few-shot_classification.md)
- [\[ACL 2025\] Unique Hard Attention: A Tale of Two Sides](../../ACL2025/others/unique_hard_attention_a_tale_of_two_sides.md)
- [\[NeurIPS 2025\] Normalization in Attention Dynamics](../../NeurIPS2025/others/normalization_in_attention_dynamics.md)

</div>

<!-- RELATED:END -->
