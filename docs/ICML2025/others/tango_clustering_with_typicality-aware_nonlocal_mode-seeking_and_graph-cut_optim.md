---
title: >-
  [论文解读] TANGO: Clustering with Typicality-Aware Nonlocal Mode-Seeking and Graph-Cut Optimization
description: >-
  [ICML2025][密度聚类] 提出"典型性(typicality)"概念，从全局视角量化数据点作为模式(聚类中心)的置信度，结合改进的路径相似度与图割优化，实现无需人工阈值设定的自动模式检测与聚类。 密度聚类是处理复杂数据分布的重要方法。Quick Shift 和 DPC（Density Peaks Clustering…
tags:
  - "ICML2025"
  - "密度聚类"
  - "模式搜索"
  - "典型性"
  - "图割优化"
  - "谱聚类"
---

# TANGO: Clustering with Typicality-Aware Nonlocal Mode-Seeking and Graph-Cut Optimization

**会议**: ICML2025  
**arXiv**: [2408.10084](https://arxiv.org/abs/2408.10084)  
**代码**: [GitHub](https://github.com/SWJTU-ML/TANGO_code)  
**领域**: 聚类  
**关键词**: 密度聚类, 模式搜索, 典型性, 图割优化, 谱聚类

## 一句话总结

提出"典型性(typicality)"概念，从全局视角量化数据点作为模式(聚类中心)的置信度，结合改进的路径相似度与图割优化，实现无需人工阈值设定的自动模式检测与聚类。

## 研究背景与动机

密度聚类是处理复杂数据分布的重要方法。Quick Shift 和 DPC（Density Peaks Clustering）是两类代表性的基于模式搜索的方法：

- **Quick Shift**: 将每个点链接到其密度更高的最近邻，通过距离阈值断开依赖以获得模式和簇。但对阈值参数高度敏感，且易将离群点误判为模式。
- **DPC**: 通过密度和距离的 decision graph 来选择聚类中心，但仍依赖人工设定阈值或手动操作。

**核心问题**: 现有方法仅依赖局部信息（密度、邻近点距离）来判断一个点是否为模式，但一个点能否作为模式实际上受**全局数据结构**的影响。例如，同一个局部密度峰值点 P，当周围有大量从属点时应被识别为模式；当仅有少量点时则应合并到邻近簇中。现有方法无法自动处理这种情况，需要逐数据集调参或人工干预。

## 方法详解

### 1. 典型性（Typicality）定义

TANGO 的核心创新是引入**典型性** $T_i$ 来量化点 $x_i$ 作为模式的置信度。设计遵循两个公理：

- **(P1)** 高密度点应有更高的典型性
- **(P2)** 通过密度上升依赖关系最终汇聚到该点的所有点及其依赖强度，也应贡献于典型性

递归定义为：

$$T_i = \rho_i + \sum_j B_{ji} T_j$$

其中 $\rho_i$ 是点 $x_i$ 的密度，$B_{ji}$ 是依赖矩阵中 $x_j$ 到 $x_i$ 的依赖权重。向量形式为 $T = B^\top T + \rho$，闭式解 $T = (I - B^\top)^{-1}\rho$。

> 与 PageRank 的联系：可以将数据点视为网页，密度上升依赖视为超链接，则典型性类似于一种特殊的 PageRank 中心性。

### 2. 基于层级依赖的相似度与密度

**相似度（基于共享近邻 SNN）**：

$$A(x_i, x_j) = \sum_{p \in \text{SNN}_k(x_i, x_j)} \exp\left(-\left(\frac{d(p,x_i)+d(p,x_j)}{2d_{\max}}\right)^2\right)$$

仅在 $x_i \in N_k(x_j)$ 或 $x_j \in N_k(x_i)$ 时非零。该定义考虑了每个共享近邻的不同贡献，增强了鲁棒性。

**密度（归一化）**：$\rho(x_i) = \frac{\sum_{p \in L(x_i)} A(x_i, p)}{\max_{x} \sum_{p \in L(x)} A(x, p)}$

### 3. Leader 关系与依赖矩阵

每个点的 **leader** 是与其相似度最高的密度更高的邻居。为消除稠密/稀疏区域的偏差，用 leader 在相似度排序邻居列表中的**排名(rank)** 而非直接相似度来定义依赖强度：

$$B_{ij} = \exp\left(-\left(\frac{\text{rank}(x_i)}{\max(\text{rank})}\right)^2\right), \quad \text{if } x_j = \text{leader}(x_i)$$

**定理 1**: 对于上述定义的 $B$，存在唯一的典型性向量 $T$ 满足 $T = B^\top T + \rho$。

**定理 2**: 按密度升序排列后，$T$ 可在 $O(n)$ 时间内通过前向替代法计算。

### 4. 基于典型性的模式检测

对于点 $x_i$ 及其 leader $x_j$：

- 若 $T(x_i) < T(x_j)$：保留依赖（$x_j$ 可以代表 $x_i$）
- 若 $T(x_i) \geq T(x_j)$：断开依赖，$x_i$ 成为模式

直觉：典型性更高的点不应被典型性更低的点所代表。该判据**完全自动**，无需人工阈值。

### 5. 路径相似度与图割聚合

模式检测后形成多个树状子簇，通过**改进的路径相似度**度量子簇间关系：

$$\text{PBSim}(r_i, r_j) = \max_{P \in \mathcal{P}_{ij}} \left\{ \min_{1 \leq h < |P|} C(p_h, p_{h+1}) \right\}$$

其中边权 $C(p_h, p_{h+1}) = A(p_h, p_{h+1}) \cdot \rho(p_h) \cdot \rho(p_{h+1})$（跨子簇），同子簇内设为 1。使用 Kruskal 算法变体高效计算，复杂度 $O(nk\log(nk))$。

最终对子簇的路径相似度矩阵施加**谱聚类（Normalized Cut）**得到最终聚类结果。

### 6. 整体复杂度

总时间复杂度为 $O(nk^2d)$（$k, q \ll n$），瓶颈在于 KD-Tree 加速的相似度矩阵计算，可并行化。

## 实验关键数据

### 合成数据可视化

在 4 个合成数据集上，TANGO 通过典型性自动检测模式并正确聚类，而 Quick Shift 和 DPC 因仅考虑局部结构表现较差。

### 真实数据集对比（16 个数据集，10 个对比方法）

| 数据集 | 规模 n | 维度 d | 簇数 | 方法说明 |
|--------|--------|--------|------|----------|
| wdbc | 569 | 30 | 2 | 医学诊断 |
| semeion | 1593 | 256 | 10 | 手写数字 |
| waveform | 5000 | 21 | 3 | 波形分类 |
| isolet1234 | 6238 | 617 | 26 | 语音字母 |
| MNIST (AE) | 10000 | 64 | 10 | 手写数字 |
| Umist (AE) | 575 | 64 | 20 | 人脸 |

对比方法包括 LDP-SC、DCDP-ASC、LDP-MST、NDP-Kmeans、DEMOS、CPF、QKSPP、DPC-DBFN、USPEC、KNN-SC。评估指标为 ARI、NMI、ACC。

**统计检验**: Friedman 检验 p < 0.05，后续 Nemenyi 检验中 TANGO 与所有对比方法的平均排名差均超过 CD = 2.30 的临界阈值，证明优势具有统计显著性。

**图像分割实验**: 在 Berkeley 分割基准上对 154,401 像素进行聚类（5 维特征：2 空间 + 3 RGB），验证了效果与计算效率。

## 亮点与洞察

1. **概念创新性强**: "典型性"概念将局部密度依赖提升到全局视角，通过递归加权汇聚下游节点的密度信息，优雅地解决了模式检测中的全局 vs. 局部问题
2. **完全自动化**: 消除了 Quick Shift / DPC 系列方法中普遍存在的模式检测阈值，仅需一个参数 $k$（近邻数）
3. **理论保证丰富**: 证明了典型性的唯一性（定理 1）、$O(n)$ 计算效率（定理 2）、路径相似度的高效计算（定理 3），以及典型性的幂律尾行为
4. **与 PageRank 的联系**: 将密度聚类中的局部依赖关系映射到图上的随机游走/PageRank 框架，提供了新的理论视角
5. **框架设计合理**: 局部→全局的两阶段策略——先用典型性分割子簇，再用路径相似度 + 谱聚类聚合——兼顾了局部结构保持和全局最优分割

## 局限与展望

1. **参数 $k$ 的自动选择**: 近邻数 $k$ 仍需搜索（2~100），论文未给出自动确定方法，作者在未来工作中也提到这一点
2. **需要预设簇数**: 最终谱聚类阶段需要指定簇的数量，限制了完全无监督的适用性
3. **高维数据的 KD-Tree 退化**: 相似度计算依赖 KD-Tree，高维空间中效率会退化
4. **图像数据需预处理**: MNIST 和 Umist 需先用 AutoEncoder 降维到 64 维，方法不直接适用于原始高维图像
5. **路径相似度的密度加权**: 低密度区域的跨簇连接可能被过度惩罚，对桥接区域宽但稀疏的簇可能不利

## 相关工作与启发

- **Quick Shift → DPC → TANGO** 的演进脉络清晰：从纯局部到引入全局信息
- 典型性的递归定义 + PageRank 视角可推广到其他需要局部→全局信息传播的场景（如图神经网络中的节点重要性）
- 路径相似度 + 谱聚类的子簇聚合策略可作为通用的后处理模块，应用于其他产生过分割结果的方法

## 评分
- 新颖性: ⭐⭐⭐⭐ (典型性概念及其理论分析有较强原创性)
- 实验充分度: ⭐⭐⭐⭐ (16 个数据集 + 10 个对比方法 + 统计检验 + 图像分割)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，理论与直觉解释兼顾)
- 价值: ⭐⭐⭐⭐ (为密度聚类中模式检测的自动化提供了有效方案)** 点自身密度越高，作为 mode 的置信度越大
- **(P2)** 通过依赖关系汇聚到该点的所有子孙点越多、依赖强度越大，置信度越大

递归定义：

$$T_i = \rho_i + \sum_j B_{ji} T_j$$

其中 $B_{ji} \neq 0$ 当且仅当 $x_j$ 依赖于 $x_i$。向量形式为 $T = B^\top T + \rho$，即 $T = (I - B^\top)^{-1} \rho$。

### 相似度与依赖矩阵

**共享近邻相似度 (Definition 1)**：对两点 $x_i, x_j$，基于其 k 近邻交集加权求和：

$$A(x_i, x_j) = \sum_{p \in \text{SNN}_k(x_i, x_j)} \exp\!\Big(-\Big(\frac{d(p, x_i) + d(p, x_j)}{2d_{\max}}\Big)^2\Big)$$

**密度 (Definition 2)**：对 $x_i$ 取最相似的 k 个邻居的相似度之和再归一化：$\rho(x_i) = \frac{\sum_{p \in L(x_i)} A(x_i, p)}{\max_x \sum_{p \in L(x)} A(x, p)}$

**Leader 关系 (Definition 3)**：每个点依赖于密度更高且相似度最大的邻居。

**依赖权重 (Eq. 6)**：用 leader 在邻居排序中的位置 rank 来定义，避免稠密区偏差：

$$B_{ij} = \exp\!\Big(-\Big(\frac{\text{rank}(x_i)}{\max(\text{rank})}\Big)^2\Big), \quad \text{if } x_j = \text{leader}(x_i)$$

### 典型性感知 Mode 搜索 (Algorithm 2)

1. 初始化 $T(x_i) \leftarrow \rho(x_i)$
2. 按密度升序遍历，向 leader 传播：$T(x_j) \leftarrow T(x_j) + T(x_i) \cdot B_{ij}$
3. 判断 mode：若 $T(x_i) \geq T(\text{leader}(x_i))$，则 $x_i$ 为 mode（即置信度高的点不应被低置信度的 leader 代表）

**关键定理**：
- **唯一性 (Theorem 1)**：typicality 向量 $T$ 存在且唯一
- **效率 (Theorem 2)**：按密度排序后，$T$ 可在 $O(n)$ 时间内前代消元求解

### 子簇聚合：改进路径相似度 + 谱聚类

**路径相似度 (Definition 5)**：两个子簇间的相似度为连接其 mode 的所有路径中"最大最小连通度"：

$$\text{PBSim}(r_i, r_j) = \max_{P \in \mathcal{P}_{ij}} \min_{1 \leq h < |P|} C(p_h, p_{h+1})$$

其中同一子簇内的边 $C = 1$，跨子簇的边 $C(p_h, p_{h+1}) = A(p_h, p_{h+1}) \cdot \rho(p_h) \cdot \rho(p_{h+1})$。用 Kruskal 变体 $O(nk \log(nk))$ 求解。

最终对子簇 mode 构成的图做 **Normalized Cut 谱聚类**。

### 总体时间复杂度

$O(nk^2d)$，由相似度矩阵计算主导（可并行化），其中 $k, q \ll n$。

## 实验关键数据

### 数据集概况

| 数据集 | 样本数 $n$ | 维度 $d$ | 簇数 |
|---|---|---|---|
| wdbc | 569 | 30 | 2 |
| segmentation | 2100 | 19 | 7 |
| semeion | 1593 | 256 | 10 |
| waveform | 5000 | 21 | 3 |
| isolet1234 | 6238 | 617 | 26 |
| MNIST (AE) | 10000 | 64 | 10 |
| Umist (AE) | 575 | 64 | 20 |

### 主要结果

- 与 **10 种 SOTA 方法**（LDP-SC, DCDP-ASC, LDP-MST, NDP-Kmeans, DEMOS, CPF, QKSPP, DPC-DBFN, USPEC, KNN-SC）在 **16 个数据集**上对比
- 评价指标：ARI、NMI、ACC
- Friedman 检验 $p < 0.05$；Nemenyi 事后检验中 TANGO 与所有对比方法的平均秩差均超过临界差 CD = 2.30，**统计显著优于所有对比方法**
- 合成数据可视化表明 TANGO 在复杂流形结构（如环形、螺旋、不均匀密度）上远优于 Quick Shift 和 DPC
- 图像分割任务（Berkeley 数据集，154,401 像素）也验证了有效性和计算效率

### 唯一超参数

仅需设定近邻数 $k$（搜索范围 2∼100，步长 1），无需手动选 mode 或设定阈值。

## 亮点与洞察

1. **Typicality 概念精巧**：将局部定义的密度依赖从全局角度重新诠释，与 PageRank 有深层联系——可视为以密度为权重、在依赖图上做信息传播
2. **理论完备**：证明了 typicality 的唯一性、$O(n)$ 可计算性、尾部幂律分布等性质
3. **消除人工干预**：不再需要手动设置阈值或查看决策图来选 mode，自动化程度显著提升
4. **两阶段解耦**：先过分割再图割聚合，局部精细性与全局最优性兼顾（类似超像素 + 图割的图像分割范式）
5. **只有一个超参数 $k$**，相比 DPC 系列方法更易调

## 局限与展望

1. **需要预设簇数**：谱聚类阶段仍需指定目标簇数，未来可结合自动确定簇数的方法（如特征值间隙）
2. **$k$ 的选取**：虽然只有一个超参数，但最优 $k$ 仍需搜索，论文未提供自适应选择策略
3. **高维数据**：SNN 相似度和 KD-Tree 在极高维数据上可能退化（维数灾难）
4. **大规模扩展**：$O(nk^2d)$ 对于百万级数据仍有挑战，虽然相似度计算可并行化但谱聚类部分 $O(q^3)$ 在子簇数很多时也需关注
5. **依赖单一 leader**：每个点只依赖一个 leader（最相似的更高密度邻居），在密度平台或对称结构中可能不够鲁棒

## 相关工作与启发

- **Quick Shift** → 本文的直接改进对象，TANGO 保留其依赖结构但用 typicality 替代阈值
- **DPC (Rodriguez & Laio, 2014)** → 决策图的半自动化 vs TANGO 的全自动化
- **LDP-SC / DCDP-ASC** → 同属"先过分割再聚合"范式，但 TANGO 在 mode 检测和路径相似度上有创新
- **PageRank** → typicality 的理论根基，启发：图上的信息传播 + 密度估计可产生有力的全局指标

## 评分

- 新颖性: ⭐⭐⭐⭐ (typicality 概念新颖，将 mode-seeking 从局部提升到全局视角)
- 实验充分度: ⭐⭐⭐⭐ (16 数据集 + 10 对比方法 + 统计检验 + 消融 + 图像分割)
- 写作质量: ⭐⭐⭐⭐ (定义—定理—算法结构清晰，图示直观)
- 价值: ⭐⭐⭐⭐ (解决了密度聚类中 mode 检测依赖人工干预的关键痛点)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] FSNet: Feasibility-Seeking Neural Network for Constrained Optimization with Guarantees](../../NeurIPS2025/others/fsnet_feasibility-seeking_neural_network_for_constrained_optimization_with_guara.md)
- [\[ICML 2025\] Exploiting Similarity for Computation and Communication-Efficient Decentralized Optimization](exploiting_similarity_for_computation_and_communication-efficient_decentralized_.md)
- [\[ICML 2025\] Efficient Optimization with Orthogonality Constraint: a Randomized Riemannian Submanifold Method](efficient_optimization_with_orthogonality_constraint_a_randomized_riemannian_sub.md)
- [\[NeurIPS 2025\] Generalized Linear Mode Connectivity for Transformers](../../NeurIPS2025/others/generalized_linear_mode_connectivity_for_transformers.md)
- [\[ICML 2025\] Symmetry-Aware GFlowNets](symmetry-aware_gflownets.md)

</div>

<!-- RELATED:END -->
