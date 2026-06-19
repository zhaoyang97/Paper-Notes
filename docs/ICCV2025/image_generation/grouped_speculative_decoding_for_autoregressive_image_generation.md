---
title: >-
  [论文解读] Grouped Speculative Decoding for Autoregressive Image Generation
description: >-
  [ICCV 2025][图像生成][自回归图像生成] 提出 Grouped Speculative Decoding (GSD)，一种免训练的自回归图像生成加速方法，通过在语义有效的 token 簇级别（而非单一最可能 token）进行推测验证，平均实现 3.7× 加速且不损失图像质量。 自回归（AR）图像模型（如 Lumi…
tags:
  - "ICCV 2025"
  - "图像生成"
  - "自回归图像生成"
  - "推测解码"
  - "推理加速"
  - "免训练"
  - "视觉token冗余"
---

# Grouped Speculative Decoding for Autoregressive Image Generation

**会议**: ICCV 2025  
**arXiv**: [2508.07747](https://arxiv.org/abs/2508.07747)  
**代码**: [GitHub](https://github.com/junhyukso/GSD)  
**领域**: 图像生成  
**关键词**: 自回归图像生成, 推测解码, 推理加速, 免训练, 视觉token冗余

## 一句话总结

提出 Grouped Speculative Decoding (GSD)，一种免训练的自回归图像生成加速方法，通过在语义有效的 token 簇级别（而非单一最可能 token）进行推测验证，平均实现 3.7× 加速且不损失图像质量。

## 研究背景与动机

自回归（AR）图像模型（如 Lumina-mGPT、LlamaGen）展示了出色的生成能力，但面临核心挑战：**逐 token 的顺序生成**。生成高分辨率图像需要产生数千个 token，且图像必须完全渲染后才能呈现有意义的输出。

**推测解码（SD）的困境**：SD 在 LLM 加速中效果显著（接受率约 70%），但在图像 AR 中接受率仅约 40%，加速效果有限。现有图像 SD 方法要么加速有限（SJD 仅 2.1×），要么需要额外训练（LANTERN 仅 1.6×）。

**核心发现——图像 token 的本质差异**：

**冗余性**：视觉 token 来自连续潜空间的向量量化，保留了低频冗余，许多 token 在细微高频细节上不同但视觉效果相似

**多样性**：不受语法约束，多个有效视觉模式均可作为下一步选择。50-95% 的 token 的 top-1 概率低于 5%

**关键实验证据**（Fig. 4）：随机将 50% 的 token 替换为 Top-100 候选，整体图像质量几乎不受影响（CLIP_score=32.089），证实了多 token 有效性。

## 方法详解

### 问题分析：为何 SD 在图像 AR 中失败

通过全变差（Total Variation）分析：

$$\alpha_{p,q} = 1 - TV(p,q) = 1 - \frac{1}{2}\sum_x |p(x) - q(x)|$$

即使 $p(x)$ 和 $q(x)$ 都认为多个 token 可行（top-1 概率低），两个分布之间微小差异的累积导致 TV 很高。关键机制：词表大小约 20,000，这种累积效应被放大——如同 Fig. 6 的玩具示例所示，均匀分布的 TV 反而比集中分布更大。

### GSD 核心思想

将词表 $\mathcal{X}$ 划分为不相交的簇：$\mathcal{C} = \{C_1, C_2, \ldots, C_K\}$

定义分组概率质量：

$$p'(C_i) = \sum_{x \in C_i} p(x), \quad q'(C_i) = \sum_{x \in C_i} q(x)$$

验证准则变为在簇级别进行：

$$\min\left(1, \frac{p'(C(x))}{q'(C(x))}\right) \geq r, \quad r \sim \mathcal{U}[0,1]$$

### 理论保证（Theorem 1）

**GSD 的接受率严格不低于标准 SD**：

$$\alpha_{GSD} \geq \alpha_{SD}$$

证明基于三角不等式：对任意簇 $C_i$，

$$\left|\sum_{x \in C_i}(p(x)-q(x))\right| \leq \sum_{x \in C_i}|p(x)-q(x)|$$

对所有簇求和得 $TV(p', q') \leq TV(p, q)$，从而 $\alpha_{GSD} = 1 - TV(p', q') \geq 1 - TV(p, q) = \alpha_{SD}$。

### 上下文感知的动态聚类

**静态聚类失败的原因**：
1. **码本嵌入空间的均匀性**：t-SNE 可视化（Fig. 3）显示嵌入近乎均匀分布，难以形成有意义的簇
2. **token 上下文的影响**：相同 token 在不同解码进度下被解码为不同的 RGB 值（Fig. 8）

**动态 GSD 策略**：
- 每步 $t$ 按 $p(x)$ 概率值排序 token
- 将被验证 token 附近 $G$ 个概率最近的 token 分为一簇
- 过滤条件：排除嵌入距离超过 $d$ 或概率差异超过 $\delta$ 的 token

关键洞见：模型预测概率本身已包含了上下文信息，比静态嵌入距离更好地反映 token 相似性。

### 算法流程（Algorithm 3, VERIFY_GSD）

1. 对每个待验证 token $\hat{X}_k$：
    - 按 $p_k$ 排序，找到 $\hat{X}_k$ 在排序中的位置
    - 取前后各 $G/2$ 个 token 组成簇 $C_{idxs}$
    - 过滤嵌入距离和概率差异超限的 token
    - 计算簇概率 $p'_C = \sum p_k[C_{idxs}]$, $q'_C = \sum q_k[C_{idxs}]$
    - 以 $\min(1, p'_C/q'_C)$ 为接受率进行验证
2. 拒绝时从 $[p_k - q_k]_+$ 重采样

## 实验关键数据

### 主实验：Parti-Prompt 和 MS-COCO（Table 1）

**Parti-Prompt（1600 提示）**：

| 配置 | 延迟 | NFE↓ | 加速比↑ | CLIP Score↑ |
|------|------|------|---------|-----------|
| Vanilla AR | 112.29s | 2392 | 1.00× | 32.091 |
| SJD | 52.34s | 1035 | 2.15× | 32.090 |
| Amplify (k=3) | 31.31s | 586 | 3.59× | 31.774 |
| **Ours (G=50)** | **24.13s** | **637** | **4.65×** | **32.075** |

**MS-COCO（5000 提示）**：

| 配置 | 延迟 | FID↓ | CLIP Score↑ |
|------|------|------|-----------|
| Vanilla AR | 122.45s | 30.79 | 31.308 |
| SJD | 55.26s | 30.78 | 31.308 |
| Amplify (k=4) | 32.29s | 40.05 | 30.99 |
| **Ours (G=25)** | **32.52s** | **33.55** | **31.24** |

关键发现：
- G=3 时 CLIP Score（32.125）反而高于 Vanilla（32.091），聚类平滑了预测噪声！
- 朴素有损方法（Amplify、Addition）虽有加速但严重降低质量（FID 40.05 vs 30.79）
- GSD 在 3.7× 加速下仅有极小质量损失

### 消融：不同聚类策略（Table 2，Parti-Prompt）

| 聚类方法 | NFE | CLIP Score |
|----------|-----|-----------|
| Baseline (SJD) | 1035 | 32.090 |
| (A) 嵌入距离 | 913.80 | 32.081 |
| (B) Draft q(x) | 685.31 | 31.791 |
| (C) Expert p(x) | 636.24 | 32.056 |
| **(D) Ours（动态）** | **636.75** | **32.075** |

静态嵌入距离聚类（A）几乎无效；用 draft q(x) 聚类（B）导致质量显著下降；用 expert p(x) 并加过滤（D）方案最优。

### Pareto 前沿分析（Fig. 10）

GSD 在 NFE vs CLIP Score 的 Pareto 前沿上严格优于朴素有损方法，甚至通过平滑效应在部分 NFE 值下优于无损 SJD。

## 亮点与洞察

1. **揭示图像 SD 失败的根本原因**：通过 Total Variation 分析建立了"低 top-1 概率 + 大词表 = 高 TV"的因果链，这是此前工作未充分认识到的
2. **理论保证的优雅性**：GSD 的加速保证仅依赖三角不等式，与聚类策略无关——无论如何聚类都不会变慢
3. **G=3 时的"免费午餐"**：小 group size 时不仅加速还提升质量，因为簇概率平滑消除了预测噪声。这是一个非常实用的发现
4. **动态聚类的必要性**：实验充分证明了静态嵌入距离聚类的无效性（均匀分布的码本空间），以及利用 p(x) 作为上下文感知相似度度量的有效性

## 局限性

1. 仅在 Lumina-mGPT (7B) 上验证，未测试更多 AR 图像模型
2. 超参数 $G, d, \delta$ 需要针对不同模型和分辨率调整
3. 未与连续推测解码、并行化 AR 等正交加速方法进行组合实验
4. 聚类操作引入的额外计算开销在非常大 G 值时可能变得显著

## 相关工作与启发

- 与 LANTERN（Jang et al., 2024）相比，GSD 无需训练且通过同时增大分子和分母避免了偏差和爆炸问题
- GSD 原理上与 tree-based SD、multi-draft SD 等方法正交，可进一步组合
- 图像 token 冗余性的发现可启发 VQ 码本设计——是否可以在训练时就减少冗余码字？
- 动态聚类思路可推广到其他高熵 token 预测场景（如蛋白质序列生成）

## 评分 ⭐⭐⭐⭐

创新性 ★★★★☆：核心思想简洁优雅，理论分析扎实
实验 ★★★★☆：定量和定性评估充分，Pareto 前沿分析有说服力
写作 ★★★★★：问题动机展开清晰，从观察到理论到方法的逻辑链完整
实用性 ★★★★★：免训练、即插即用、代码已公开，应用门槛低

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Annealed Relaxation of Speculative Decoding for Faster Autoregressive Image Generation](../../AAAI2026/image_generation/annealed_relaxation_of_speculative_decoding_for_faster_autor.md)
- [\[CVPR 2026\] SJD-PAC: Accelerating Speculative Jacobi Decoding via Proactive Drafting and Adaptive Continuation](../../CVPR2026/image_generation/sjd-pac_accelerating_speculative_jacobi_decoding_via_proactive_drafting_and_adap.md)
- [\[ICML 2026\] Speculative Coupled Decoding for Training-Free Lossless Acceleration of Autoregressive Visual Generation](../../ICML2026/image_generation/speculative_coupled_decoding_for_training-free_lossless_acceleration_of_autoregr.md)
- [\[CVPR 2026\] Multi-Scale Local Speculative Decoding for Image Generation](../../CVPR2026/image_generation/multi-scale_local_speculative_decoding_for_image_generation.md)
- [\[ICCV 2025\] DC-AR: Efficient Masked Autoregressive Image Generation with Deep Compression Hybrid Tokenizer](dc-ar_efficient_masked_autoregressive_image_generation_with_deep_compression_hyb.md)

</div>

<!-- RELATED:END -->
