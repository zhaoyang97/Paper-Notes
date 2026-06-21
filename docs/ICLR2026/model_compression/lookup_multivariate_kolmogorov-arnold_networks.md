---
title: >-
  [论文解读] Lookup multivariate Kolmogorov-Arnold Networks
description: >-
  [ICLR 2026][模型压缩][Kolmogorov-Arnold Networks] 把 KAN 的可训练函数从一维换成二维、并用 B-样条查找表实现 O(1) 求值，得到一个能直接替换线性层的 lmKAN 模块——在同等精度下把推理 FLOPs 砍掉 1.6–78×，并配套 CUDA kernel 实现 H100 上 1 个数量级的实测加速。
tags:
  - "ICLR 2026"
  - "模型压缩"
  - "Kolmogorov-Arnold Networks"
  - "样条查找表"
  - "高效推理"
  - "线性层替代"
  - "CUDA kernel"
---

# Lookup multivariate Kolmogorov-Arnold Networks

**会议**: ICLR 2026  
**代码**: [https://github.com/schwallergroup/lmkan](https://github.com/schwallergroup/lmkan)  
**领域**: 模型压缩 / 高效推理  
**关键词**: Kolmogorov-Arnold Networks, 样条查找表, 高效推理, 线性层替代, CUDA kernel  

## 一句话总结
把 KAN 的可训练函数从一维换成二维、并用 B-样条查找表实现 O(1) 求值，得到一个能直接替换线性层的 lmKAN 模块——在同等精度下把推理 FLOPs 砍掉 1.6–78×，并配套 CUDA kernel 实现 H100 上 1 个数量级的实测加速。

## 研究背景与动机
**领域现状**：现代深度模型（MLP、Transformer、CNN、GNN）的参数量与计算量绝大部分集中在高维线性映射上：宽度为 $N$ 的线性层，参数与算力都按 $O(N^2)$ 增长，而其它层只占 $O(N)$。部署大模型的瓶颈因此主要是这些线性层的推理开销。

**现有痛点**：Kolmogorov-Arnold Networks（KAN）用一组可训练的一维函数来构造高维映射，原则上很适合"参数多但求值便宜"的样条查找表——因为分段函数求值是 $O(1)$，与参数量无关。但实践中 KAN 几乎从不给每个函数超过几十个参数：一维函数堆很多参数等于去拟合极高频段，会带来训练不稳定与泛化问题；而样条查找表的 O(1) 思路虽常被提及，却几乎没人真正高效地在 GPU 上落地，主流工作反而走向 Chebyshev、Fourier、Gaussian RBF 这类**稠密**基函数（如 FastKAN），放弃了紧支撑带来的 O(1) 优势。

**核心矛盾**：想要"每个内函数承载大量参数 + 推理几乎不涨成本"，一维函数会把表达力浪费在高频上、难训练，而高效 O(1) 实现又缺乏能跑赢 MLP 的工程落地。

**本文目标**：造一个**通用、即插即用**替换线性层的模块，在 FLOPs 和实测墙钟时间上都对 MLP 形成帕累托占优，且在没有已知闭式 KART 表示、也不比一般任务更光滑的真实任务上验证。

**核心 idea**：**【多元函数 + 查找表】** 把 KAN 的内函数从一维升到二维（更一般地 $d$ 维），二维函数能"消化"远多于一维的参数而不溢出到高频；同时用二阶 B-样条 + 无界 sigma 网格实现严格 O(1) 求值，再写专用 CUDA kernel 把纸面效率兑现成真实加速。

## 方法详解

### 整体框架
一个 $d$ 维 lmKAN 层把输入按 $d$ 个一组切块，每块喂给一个可训练的低维函数，再把同一输出位置上的若干函数求和得到输出：
$$y_q = \sum_{p=0}^{N_{in}/d-1} f_{qp}(x_{dp}, x_{dp+1}, \dots, x_{dp+d-1})$$
其中 $f_{qp}$ 是可训练的 $d$ 维函数。和 KAN 一样，这些层之间不需要额外激活，可任意堆叠，直接替换 MLP 里"线性层+激活"的组合。论文实现并优化了 $d=1$ 和 $d=2$ 两种 CUDA kernel，二维是主力。每个二维函数用二阶 B-样条在 $(G+1)^2$ 个系数上参数化，但任意点只有 4 个 B-样条非零，因此一次求值只要 4 次乘加。

```mermaid
graph LR
    X["输入 x (按2个一组切块)"] --> BN["BatchNorm<br/>无仿射参数"]
    BN --> G["sigma 无界网格<br/>i = ⌊σ(x)·G⌋ O(1)定位"]
    G --> F["2D 函数 f(x1,x2)<br/>=Σ p·B样条<br/>仅4项非零"]
    F --> SUM["按输出位置求和 y_q"]
    SUM --> Y["输出 y (可继续堆叠)"]
```

### 关键设计

**1. 多元内函数：用维度而非频率来吸纳参数。** 这是全文最核心的取舍。一维函数想要更多参数，只能加密网格 $G$，但这等价于让函数去表征更高的频率带——拟合 KART 那种"狂野"的内函数时确实需要，但在真实任务里会牺牲训练稳定性和泛化。二维函数则能"容纳"多得多的参数而不把表达力外溢到极高频：一个每维 10 个网格点的四维函数，参数量和一个有约 $10^4$ 网格区间的一维函数相当。式 $(2)$ 把一维 KAN 推广为 $d$ 维分块映射，论文用 2D 作为甜点——既显著比 1D 更准更易训，求值成本又恰好和 1D 持平（见下）。若需要，多元函数还能退化成一维之和，从而整个 lmKAN 退回标准 KAN，KART 定理依然适用。

**2. 无界 sigma 网格：让 O(1) 定位在训练动态下依然成立。** 训练中神经元激活会任意漂移，定义在有界区间的网格会失效。作者设计了覆盖整条实轴的 sigma 网格：取任意 sigmoid 形函数 $\sigma(x)$，用 $G-1$ 个等分位水平线与 $\sigma(x)$ 的交点作为网格点，于是网格在原点附近最细、向两侧逐渐变粗。关键是给定 $x$，所在区间下标可直接 $i = \lfloor \sigma(x)\,G \rfloor$ 算出，保持 O(1)。每层 lmKAN 前再接一个无仿射参数的 BatchNorm，把激活控制在合理范围以均衡各区间的占用率。

**3. 紧支撑二阶 B-样条：把"参数多"和"求值快"在数学上解耦。** 内函数用建在 sigma 网格上的二阶 B-样条作基。每个 B-样条只在中心点两侧两个区间非零，$G$ 个区间用 $G+1$ 个基函数（$G-1$ 个内点 B-样条 + 两端各一个无穷区间上的线性段）。二维基取张量积 $B_{i_1 i_2}(x_1,x_2)=B_{i_1}(x_1)B_{i_2}(x_2)$，函数写成 $f(x_1,x_2)=\sum_{i_1,i_2} p_{i_1 i_2} B_{i_1 i_2}(x_1,x_2)$。由于任意点只有 4 个二维 B-样条非零，求值恒为 4 项，与 $G$ 无关——这正是"参数量可以是几十上百倍、推理却几乎不变"的根源。二阶 B-样条的光滑度（$C^0$ 但非 $C^1$）恰好对齐 ReLU，作者论证更高阶带来的额外光滑性未必值得其计算开销。

**4. 复用中间量把 FLOPs 压到线性层的 2×，再用 CUDA kernel 兑现。** 同一列的多个低维函数共享完全相同的自变量，网格下标与 B-样条值只需对每对输入算一次即可复用，于是每个二维函数实际只花 4 次乘加。一层有 $\lceil N_{in}/2\rceil N_{out}$ 个二维函数，主导 $O(N^2)$ 项的乘加总数为 $4\lceil N_{in}/2\rceil N_{out}=2N_{in}N_{out}$，**恰好是同形状线性层的 2×**；而那个 $O(N)$ 项不是额外开销，它替代了 lmKAN 不需要的逐元素 bias 与（可能昂贵的 tanh 等）激活。工程上作者用 GEMM 式共享内存分块写 CUDA kernel：H100 上 16×16 tile 比稠密线性层慢约 8×（访存不如稠密 GEMM 连贯），但因每层参数量是基线的约 220×，**按同参数量算反而快约 27.5×**；换 8×8 tile 可把网格上限提到 $G=40$，每参数效率达约 88.5×。

## 实验关键数据

### 主实验表格

| 任务 | Backbone | 同精度 FLOPs 降低 | H100 实测加速 |
|------|----------|------------------|---------------|
| 通用高维函数逼近（蒸馏随机 teacher MLP，R³²→R¹） | 2 隐层全连接 | 最高 **6.0×** | 1.8× |
| 随机扰动甲烷构型（12 维表格回归，DFT 能量） | 2 隐层全连接 | 最高 **78.0×** | **12.9×** |
| CIFAR-10 图像分类 | lmKAN-CNN | **1.6–2.1×** | — |
| ImageNet（81×81）Top-5 | lmKAN-CNN | **1.7×** | — |

跨所有任务，lmKAN 都在 "推理 FLOPs–精度" 上帕累托占优；MLP 基线均用足/半训练预算两条线证明已紧致收敛。

### 消融实验表格

| 对比项 | 现象 / 结论 |
|--------|------------|
| 网格分辨率 $G$ 扫描（隐维固定 256） | 精度随 $G$ 呈 **U 形**，并非越细越好；2D 在更大参数量处才饱和、且精度显著更高 |
| 2D lmKAN（最优 $G$）vs 更大 MLP | 与 **大约 16× 更大**（隐维 4×）的 MLP 精度相当 |
| 推理成本 vs $G$ | FLOPs 与 H100 墙钟时间**与 $G$ 无关**（验证 O(1) 设计） |
| 2D lmKAN vs 1D lmKAN vs FastKAN（CIFAR-10，隐维 256） | 网格过细时 1D lmKAN/FastKAN 退化到**比 MLP 还差**；2D lmKAN 退化轻微、更易训且精度明显更高 |

### 关键发现
- **维度比频率更会"装"参数**：2D 内函数能在富参数化下稳定训练，而 1D（含 FastKAN）在细网格下崩坏——这是 lmKAN 优势的根本来源。
- **纸面效率能落地**：得益于 CUDA kernel，甲烷任务实现超过一个数量级（12.9×）的真实 H100 加速，而非只有 FLOPs 好看。
- **任务越"表格化"收益越大**：甲烷这类低维稠密回归收益最高（78×），CNN 因卷积结构收益较小（~1.7×，且卷积尚未写专用 kernel）。
- **同精度可用更小模型**：在通用函数逼近上，最优网格的 2D lmKAN 可匹敌一个隐维 4×、整体约 16× 大的 MLP，直观说明"把参数塞进低维函数"比"加宽线性层"更划算。

## 亮点与洞察
- **重新激活了一个被忽视的方向**：当主流 KAN 工作纷纷转向稠密基函数（牺牲 O(1)），本文反向坚持紧支撑 B-样条 + 查找表，并真正把 GPU 工程做出来，证明这条路能跑赢 MLP。
- **"2D 是免费午餐"的洞察很漂亮**：二阶 B-样条下 1D 和 2D 的求值都恰好是线性层的 2×，于是从 1D 升到 2D 不增推理成本却大幅提升表达力与可训练性。
- **无界 sigma 网格**解决了"训练中激活漂移导致有界网格失效"这一实际工程痛点，且仍保持 O(1) 定位。
- 严谨的基线处理（半预算/全预算双线证明收敛）让"更省"的主张可信。

## 局限与展望
- **访存不友好**：lmKAN 的查找表访存模式不如稠密 GEMM 连贯，实测加速（~8–9.5× 慢于同形状线性层）明显逊于 2× 的 FLOPs 理论值，需要持续的 kernel 优化。
- **共享内存上限约束网格**：H100 上 $G\le 20$（16×16 tile）或 $\le 40$（8×8 tile），限制了单层可达的参数密度。
- **卷积尚未专用化**：lmKAN-CNN 训练时把卷积展平成全连接，**还没有为卷积写专门 CUDA kernel**，CNN 上的实测加速因此未展示。
- **维度增长代价**：$d$ 维二阶 B-样条求值为 $2^d/d$ 倍线性层成本，$d\ge 3$ 后迅速变贵，故只实现到 2D。
- 大规模/复杂 backbone（如真正的大模型、Transformer 主干）上的效果仍待验证，论文主动选择了"设置多样性"而非"超大规模"。

## 相关工作与启发
- **KAN 谱系**：基于 Liu et al. (2024) 的现代 KAN，但把一维内函数升级为多元函数；并把常被提及却少落地的"查找表 O(1)"思路真正工程化。
- **与 FastKAN 的对照**：FastKAN 用稠密 Gaussian RBF 替换稀疏 B-样条纯为优化便利，恰好放弃了 O(1)；本文实验直接说明这条稠密路线在富参数化下不如 2D 紧支撑方案。
- **对效率研究的启发**：在"参数量 vs 推理成本"普遍正比的范式下，查找表给出了一个解耦二者的具体可行点，提示线性层未必是高维映射的唯一高效形态。
- **跨架构通用性**：把同一模块装进 MLP、CNN，提示 GNN/Transformer 的线性映射也可能受益，是一个有潜力的通用替换件。

## 评分
- **新颖性**: ⭐⭐⭐⭐⭐ —— "多元内函数 + 紧支撑查找表 + 自研 CUDA kernel" 的组合是一个清晰且被忽视的方向，扭转了主流 KAN 走向稠密基的趋势。
- **实验充分度**: ⭐⭐⭐⭐ —— 覆盖函数逼近/表格回归/CNN 三类任务与 FLOPs/墙钟双指标，基线收敛严谨；但缺大规模与 Transformer 主干，卷积加速未实测。
- **写作质量**: ⭐⭐⭐⭐ —— 动机、取舍（频率 vs 维度）与成本分析讲得清晰可信，图表组织合理。
- **价值**: ⭐⭐⭐⭐ —— 即插即用、配套开源 CUDA kernel，对推理成本敏感的部署场景有直接实用价值，受访存与规模验证所限暂未到顶。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Color Matching Using Hypernetwork-Based Kolmogorov-Arnold Networks (cmKAN)](../../ICCV2025/model_compression/color_matching_using_hypernetwork-based_kolmogorov-arnold_networks.md)
- [\[ICLR 2026\] Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval](enhancing_multivariate_time_series_forecasting_with_global_temporal_retrieval.md)
- [\[ICLR 2026\] Bridging Kolmogorov Complexity and Deep Learning: Asymptotically Optimal Description Length Objectives for Transformers](bridging_kolmogorov_complexity_and_deep_learning_asymptotically_optimal_descript.md)
- [\[ICLR 2026\] Adaptive Width Neural Networks](adaptive_width_neural_networks.md)
- [\[ICLR 2026\] A Recovery Guarantee for Sparse Neural Networks](a_recovery_guarantee_for_sparse_neural_networks.md)

</div>

<!-- RELATED:END -->
