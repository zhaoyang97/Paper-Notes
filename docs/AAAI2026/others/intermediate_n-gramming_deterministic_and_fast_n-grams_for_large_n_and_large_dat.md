---
title: >-
  [论文解读] Intermediate N-Gramming: Deterministic and Fast N-Grams For Large N and Large Datasets
description: >-
  [AAAI 2026][n-gram计算] 提出 Intergrams 多遍扫描算法，利用较短 n-gram 作为前缀递推过滤候选更长 n-gram，充分利用处理器缓存层次结构实现缓存友好的内存访问模式，在 TB 级数据集上比此前最快的 hash-gramming 方法加速 6-33 倍，同时几乎精确恢复所有 top-k n-gram。
tags:
  - "AAAI 2026"
  - "n-gram计算"
  - "硬件感知算法"
  - "缓存友好"
  - "Zipf分布"
  - "多遍扫描"
---

# Intermediate N-Gramming: Deterministic and Fast N-Grams For Large N and Large Datasets

**会议**: AAAI 2026  
**arXiv**: [2511.14955](https://arxiv.org/abs/2511.14955)  
**代码**: [GitHub](https://github.com/rcurtin/Intergrams)  
**领域**: 高性能计算 / 数据挖掘  
**关键词**: n-gram计算, 硬件感知算法, 缓存友好, Zipf分布, 多遍扫描

## 一句话总结

提出 Intergrams 多遍扫描算法，利用较短 n-gram 作为前缀递推过滤候选更长 n-gram，充分利用处理器缓存层次结构实现缓存友好的内存访问模式，在 TB 级数据集上比此前最快的 hash-gramming 方法加速 6-33 倍，同时几乎精确恢复所有 top-k n-gram。

## 研究背景与动机

n-gram 特征在多个领域有广泛应用：恶意软件检测中使用字节序列特征训练分类器；自然语言处理中字符级 n-gram 在某些任务上可媲美 embedding 模型；基因组分析中 k-mer 被用于物种分类和系统发育分析。实际应用中的常见流水线是：先计算数据集中出现频率最高的 top-k 个 n-gram，再将每个样本转为稀疏特征向量用于训练。

然而，n-gram 特征的数量随 n 呈指数增长（对字节数据，8-gram 有 $256^8 \approx 1.8 \times 10^{19}$ 种可能），这使得在大规模数据集上计算 top-k n-gram 成为整条流水线中耗时最长的瓶颈。现有方法中，scikit-learn 的 CountVectorizer 需维护完整字典，对非平凡的 n 和 k 无法扩展。流式算法（如 Space-Saving）需要远多于 k 的计数器且不适合"每序列只计一次"的场景。当前最快的 hash-gramming 方法虽远快于朴素方法，但吞吐量仅 10-20 MB/s，离磁盘带宽（~600 MB/s）相差甚远。

**核心矛盾**在于：现有算法忽视了硬件内存层次结构。hash-gramming 对一个 $2^{31}$ 大小的桶数组做随机访问，这远超处理器缓存容量，导致大量缓存未命中、TLB 未命中和页面错误，严重拖慢吞吐量。作者的核心洞察是：**如果将中间数据结构控制在缓存可容纳的范围内，并保证对大数组的顺序访问，就能让 n-gram 计算达到磁盘 I/O 的速度上限。**

## 方法详解

### 整体框架

Intergrams 采用"递推多遍扫描"策略：先用一种极快的专用算法计算 top-$zk$ 个 3-gram（$z$ 是过采样因子），然后利用这些 3-gram 作为前缀过滤候选 4-gram，计算 top-$zk$ 个 4-gram，再用 4-gram 前缀过滤候选 5-gram……依此类推直到目标长度 $n$。每一轮只需遍历一遍全部数据，且由于前缀过滤大幅缩减了候选集，中间数据结构可以始终保持在缓存友好的大小。

### 关键设计

1. **快速 3-gram 算法（基础步骤）**:

    - 功能：利用 3-gram 可枚举的特性（$256^3 = 16M$ 种），直接用位向量统计
    - 核心思路：对每个序列 $s_i$ 维护一个 2 MB 的位向量 $C_i$（$16M$ 位），标记出现过的 3-gram，然后顺序刷新到全局计数数组 $C$（$16M \times 4$ 字节 = 64 MB）。位向量随机写但仅 2 MB，轻松放入 L2 缓存；全局数组 64 MB 但顺序访问，可以利用硬件预取达到高带宽
    - 设计动机：位向量大小 2 MB 适配 L2 缓存，全局数组 64 MB 适配 L3 缓存，且顺序访问模式避免了缓存未命中。额外使用 AVX2 SIMD 指令加速刷新操作（5 条 SIMD 指令可将一个位向量的 8 个元素增量到 $C$ 中），在强力硬件上可达到磁盘峰值带宽约 5 GB/s

2. **递推前缀过滤（Intergrams 核心）**:

    - 功能：从 3-gram 递推到任意长度 n-gram，每步只保留 $zk$ 个候选前缀
    - 核心思路：在第 $j$ 步（$j \geq 4$），仅统计那些 $(j-1)$-gram 前缀在上一步 top-$zk$ 集合 $P^{(j-1)}$ 中的 $j$-gram。候选数量从 $256^j$ 骤降至 $256 \times zk$，位向量大小为 $32zk$ 字节，通常远小于缓存容量
    - 设计动机：利用了真实数据 n-gram 频率服从 Zipf 分布的特性——前缀频率与后续 n-gram 频率高度相关，因此用少量高频前缀就能覆盖绝大多数高频长 n-gram

3. **Trie 数据结构优化**:

    - 功能：查询一个 n-gram 的前缀是否属于候选集 $P$
    - 核心思路：用紧凑的 trie 树存储候选前缀，每个节点仅 2 字节（值 + 子节点数），少于 4 个子节点用列表（12字节），多于 4 个子节点使用 256 元素查找表（1 KB）。按频率排列子节点的内存布局，使高频前缀的 trie 路径在内存中连续
    - 设计动机：trie 查找是算法最内层循环，必须尽量小以装入缓存，且高频路径的空间局部性可减少缓存未命中。此优化使整体运行时间减少约 10-15%

4. **并行和 I/O 优化**:

    - 功能：充分利用多核并行和磁盘 I/O
    - 核心思路：使用线程对（thread pair）策略——一个线程专门做磁盘 I/O 读取，另一个线程处理已缓冲的数据。由于 Intergrams 的瓶颈在磁盘而非计算，专门的 I/O 线程可确保最大磁盘吞吐
    - 设计动机：hash-gramming 因计算瓶颈无法饱和磁盘带宽，每个序列只能分配一个线程同时做 I/O 和计算

### 理论分析

假设 n-gram 频率服从参数为 $a$ 的 Zipf 分布（$p_i \propto 1/i^a$），作者证明：
- **Lemma 1**: 若 top-$k'$ 个 n-gram 占总出现次数比例为 $\beta$，则以它们为前缀的 $(n+1)$-gram 至少占总 $(n+1)$-gram 比例 $\beta' = \beta - m/(N-m)$
- **Theorem 1**: Intergrams 召回的 top-$k$ $(n+1)$-gram 比例至少为 $1 - \frac{(|\mathbf{D}_{n+1}|^{1-a} - a)(1-\beta') - a}{(k+1)^{1-a} - 1}$
- **Theorem 2**: 考虑采样噪声后，以概率 $1-\delta$ 成立，只需将 $\beta'$ 替换为 $\beta'' = \beta' - \Delta(\delta)$，其中 $\Delta(\delta) = 4\sqrt{k^2 \ln(2|\mathbf{D}_n|/\delta) / (2N)}$

当 $a > 1$ 时（真实数据通常满足），召回率渐近趋于 $1 - O((k/|D_{n+1}|)^{a-1})$，随 $k$ 增大迅速趋近 1。

## 实验关键数据

### 数据集

| 数据集 | 类型 | 大小 | 序列数 |
|--------|------|------|--------|
| EMBER | 字节（可执行程序） | 1009 GB | 800k |
| C4 | 文本 | 751 GB | 6.22M |
| 1000gp | 基因组 | 1.4 TB | 1.58M |

### 主实验（运行时间与加速比，n=6, 不同 k）

| 算法 | EMBER (k=100k) | C4 (k=10k) | 1000gp (k=10k) |
|------|----------------|------------|----------------|
| hg-vanilla | 9078.5s (1x) | 39787.1s (1x) | 10042.0s (1x) |
| hg-fast（所有优化） | 7162.6s (1.27x) | 30811.2s (1.29x) | 7061.5s (1.42x) |
| Intergrams, z=1 | 1413.7s (6.42x), Jaccard=0.71 | 1183.9s (33.6x), Jaccard=0.91 | 1215.7s (8.26x), Jaccard=1.0 |
| Intergrams, z=1.5 | 1458.8s (6.22x), Jaccard=0.91 | 1373.7s (29.0x), Jaccard=1.0 | 1152.9s (8.71x), Jaccard=1.0 |
| Intergrams, z=2 | 1771.4s (5.13x), Jaccard=0.97 | 1501.9s (26.5x), Jaccard=1.0 | 1144.6s (8.77x), Jaccard=1.0 |

### 消融实验（各步骤耗时分解，EMBER, k=10k, z=1.5）

| 步骤 | 运行时间 | 吞吐量 |
|------|---------|--------|
| 3-gram 扫描 | 228.2s | 4.42 GB/s |
| top-zk 3-gram / trie 构建 | 3.25s | — |
| 4-gram 扫描 | 205.95s | 4.90 GB/s |
| 5-gram 扫描 | 189.37s | 5.33 GB/s |
| 6-gram 扫描 | 190.57s | 5.29 GB/s |

### 关键发现
- Intergrams 在所有三个不同领域的 TB 级数据集上一致性地大幅超越 hash-gramming（最高 33 倍加速）
- 即使过采样因子 $z=1$，在 C4 和 1000gp 上也能精确恢复全部 top-k n-gram（Jaccard=1.0）
- 各步骤吞吐量接近甚至达到磁盘峰值带宽（~5 GB/s），验证了"计算瓶颈在磁盘 I/O"的设计目标
- 后续 pass 比初始 pass 更快，因为 trie 过滤淘汰了更多候选，减少了实际更新次数
- 对 hash-gramming 施加的各种优化（大页、预取、cuckoo hash、trie）加速有限（最多 1.42x），因为根本问题是随机内存访问模式

## 亮点与洞察
- 极好的"系统+算法"协同设计范例：不是在算法层面做微调，而是从硬件内存层次出发重新设计计算流程，实现数量级加速
- 理论分析与实证完美吻合：Zipf 分布假设下的理论召回率保证在实验中得到验证
- 3-gram 到任意 n-gram 的递推思路非常优雅——将指数级搜索空间通过多个线性级步骤递推解决
- 实用性极强：算法确定性、可并行、代码已开源，可直接用于恶意软件检测、NLP 和基因组分析场景
- 过采样因子 $z$ 提供了精度-速度的简洁权衡旋钮

## 局限与展望
- 当前实现针对字节序列优化（alphabet=256），对更大字母表（如 Unicode）需要调整 trie 结构和 3-gram 基础步骤
- 理论分析假设 Zipf 分布参数在不同 n 之间恒定，实际中更长 n-gram 的 Zipf 参数通常更大（作者承认这是悲观假设）
- 对于 n 很小的情况（如 n=3），Intergrams 无法提供相对于直接枚举的优势
- 未讨论分布式（多机）场景，当前仅限单机多核

## 相关工作与启发
- hash-gramming (Raff 2018) 是此前最快方法，但忽视硬件限制；本文的核心贡献就是证明"算法设计必须感知硬件"
- 缓存无关算法 (Frigo 1999) 的思想在本文中得到了很好的实践
- 本文的递推过滤思路可能启发其他"频繁模式挖掘"场景中的类似优化

## 评分
- 新颖性: ⭐⭐⭐⭐ （思路核心是硬件感知+递推过滤，简洁但不算全新范式）
- 实验充分度: ⭐⭐⭐⭐⭐ （三个领域的 TB 级数据集，详细的消融和步骤分解）
- 写作质量: ⭐⭐⭐⭐⭐ （叙述清晰，硬件分析深入浅出，理论和实验紧密结合）
- 价值: ⭐⭐⭐⭐ （实用价值高，但应用场景相对特定）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Torus Graphs for Large-Scale Neural Phase Analysis](../../ICML2026/others/torus_graphs_for_large_scale_neural_phase_analysis.md)
- [\[CVPR 2026\] Large-scale Robust Enhanced Ensemble Clustering via Outlier Decoupling](../../CVPR2026/others/large-scale_robust_enhanced_ensemble_clustering_via_outlier_decoupling.md)
- [\[CVPR 2026\] Bi-directional Autoregressive Diffusion for Large Complex Motion Interpolation](../../CVPR2026/others/bi-directional_autoregressive_diffusion_for_large_complex_motion_interpolation.md)
- [\[ACL 2025\] Code-Switching and Syntax: A Large-Scale Experiment](../../ACL2025/others/code-switching_and_syntax_a_large-scale_experiment.md)
- [\[ICML 2026\] HASTE: Hardware-Aware Dynamic Sparse Training for Large Output Spaces](../../ICML2026/others/haste_hardware-aware_dynamic_sparse_training_for_large_output_spaces.md)

</div>

<!-- RELATED:END -->
