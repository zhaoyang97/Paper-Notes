---
title: >-
  [论文解读] Retraining-Free Merging of Sparse MoE via Hierarchical Clustering
description: >-
  [ICML 2025][LLM效率][Sparse Mixture-of-Experts] 提出 HC-SMoE，一种基于专家输出层次聚类的无需重训练专家合并框架，通过输出相似度度量和层次聚类实现 SMoE 模型的高效压缩，在 Qwen 和 Mixtral 上分别实现 25%-50% 的专家参数缩减并保持优越性能。
tags:
  - "ICML 2025"
  - "LLM效率"
  - "Sparse Mixture-of-Experts"
  - "专家合并"
  - "层次聚类"
  - "模型压缩"
  - "无需重训练"
---

# Retraining-Free Merging of Sparse MoE via Hierarchical Clustering

**会议**: ICML 2025  
**arXiv**: [2410.08589](https://arxiv.org/abs/2410.08589)  
**代码**: [GitHub](https://github.com/wazenmai/HC-SMoE)  
**领域**: LLM效率  
**关键词**: Sparse Mixture-of-Experts, 专家合并, 层次聚类, 模型压缩, 无需重训练

## 一句话总结

提出 HC-SMoE，一种基于专家输出层次聚类的无需重训练专家合并框架，通过输出相似度度量和层次聚类实现 SMoE 模型的高效压缩，在 Qwen 和 Mixtral 上分别实现 25%-50% 的专家参数缩减并保持优越性能。

## 研究背景与动机

Sparse Mixture-of-Experts (SMoE) 模型通过稀疏激活机制实现了高效的参数利用——每个输入 token 仅激活部分专家，从而在不增加推理成本的情况下扩展模型容量。然而 SMoE 模型的**总参数量**依然庞大（所有专家参数都需要加载到内存中），这在资源受限环境下构成严重部署瓶颈。

已有研究揭示了 SMoE 中**专家冗余**问题的严重性：

**Liu et al. (2023)** 发现专家之间存在高度的表征相似性

**Lu et al. (2024)** 提供了进一步的实证支持
3. 这些冗余意味着存在巨大的优化空间

现有的专家缩减方法存在明显局限：

- **专家剪枝**（TSEP, O-prune, S-prune）：直接删除专家会造成已学表征的不可逆损失
- **M-SMoE 专家合并**：依赖 router logits 进行分组，对任务数据敏感，且使用激活频率决定每层保留数量在任务无关设置下效果差
- **ZipIt 模型合并**：特征相关性计算开销大，不适合大规模专家合并

**核心洞察**：与其丢弃专家参数（剪枝），不如将功能相似的专家合并，保留更多知识；使用专家输出而非 router logits 作为相似度度量可以更好地捕捉专家的功能等价性。

## 方法详解

### 整体框架

HC-SMoE 是一个两阶段框架：**聚类（Grouping）** → **合并（Merging）**，无需重训练，任务无关，可扩展。

整体流程：
1. 在校准数据集 $\mathcal{D}_{cal}$（如 C4）上收集每个专家的平均输出向量
2. 基于输出向量的余弦相似度，对每层专家进行层次聚类
3. 将同一簇内的专家按频率加权合并为一个新专家
4. Router 网络保持不变，原先路由到同一簇内任意专家的输入统一指向合并后的新专家

框架采用**静态分组策略**：每层合并后保留固定数量 $r$ 个专家，与 O-prune 对齐，便于公平比较。

### 关键设计

#### 1. 基于专家输出的相似度度量

这是 HC-SMoE 最核心的设计创新。对于专家 $E_j$，其代表向量定义为：

$$o_j := \mathbb{E}_{x \sim \mathcal{D}_{cal}}[E_j(x)] = \frac{1}{T}\sum_{x \in \mathcal{D}_{cal}}^{T} E_j(x)$$

其中 $T$ 为校准数据集中的 token 总数。

**为什么不用 router logits？**
- Router logits $R(x)$ 反映的是输入相关的分配偏好，而非专家的内在功能
- Router logits 具有任务特异性偏差，阻碍泛化能力
- 参数空间比较（如拼接扁平化权重）在高维空间中失效

**为什么用专家输出？**
- 输出相似度与功能等价性高度相关（Li et al., 2016; Stoica et al., 2024）
- 同时捕获上下文输入信息和专家学到的变换
- 由 Appendix 中 L2 误差实验验证其有效性

#### 2. 层次聚类算法

HC-SMoE 采用**凝聚式层次聚类**（agglomerative hierarchical clustering），核心步骤：

1. **初始化**：每个专家作为独立簇
2. **迭代合并**：每步找到最相似的两个簇进行合并
3. **终止条件**：当簇数减少到目标数 $r$ 时停止

簇间距离采用**平均连接**（average linkage），即两簇间所有专家对的平均相似度。

**相比 M-SMoE 的 one-pass 分组**，层次聚类的优势：
- **迭代比较**：每步合并后重新评估所有簇间距离，保证全局最优
- **簇间多样性**：保持更好的 inter-cluster 多样性和 intra-cluster 相似性
- **理论保证**：层次聚类提供可证明的聚类质量保证
- **对初始化不敏感**：不像 K-means 等方法依赖初始化，结果确定性更强

#### 3. 频率加权合并策略

合并同一簇内的专家时，HC-SMoE 使用**基于激活频率的加权平均**：

对于簇 $C_i = \{E_0^i, E_1^i, \ldots, E_{|C_i|}^i\}$，合并后的新专家权重为：

$$W_{\text{merged}}^i = \sum_{j=0}^{|C_i|} \frac{f_j}{\sum_k f_k} \cdot W_j^i$$

其中 $f_j$ 为专家 $E_j$ 在校准数据上的激活频率。

**设计直觉**：高频专家贡献更大，其参数在合并时应获得更大权重。注意这里频率仅用于合并权重计算，而不用于聚类分组——这是与 M-SMoE 的关键区别。

### 损失函数 / 训练策略

HC-SMoE 是**完全无需训练**的方法。不涉及损失函数优化或梯度更新。整个流程仅需：

1. **一次前向传播**通过校准数据集收集专家输出
2. **层次聚类**计算（时间复杂度 $O(n^2 \log n)$，其中 $n$ 为专家数）
3. **参数加权平均**完成合并

这使得 HC-SMoE 在实际部署中非常高效——相比需要微调的方法（如 TSEP、M-SMoE），HC-SMoE 无 GPU 训练开销。

## 实验关键数据

### 主实验

在 8 个零样本语言任务（LM-Harness benchmark）上评估。

**Qwen1.5-MoE-A2.7B-Chat 模型:**

| 压缩率 | 方法 | 平均准确率 | 与最强基线比 |
|:---:|:---:|:---:|:---:|
| 0% | 原模型 | ~56% | - |
| 25% | S-prune | 较低 | - |
| 25% | HC-SMoE | 最优 | **+6.95%** |
| 37.5% | 最强基线 | 显著下降 | - |
| 37.5% | HC-SMoE | 最优 | **+2.14%** |
| 50% | 各基线 | 大幅下降 | - |
| 50% | HC-SMoE | 最优 | 显著领先 |

**Mixtral 8×7B 模型:**

| 方法 | 类型 | 任务无关 | 无需重训 | 性能表现 |
|:---:|:---:|:---:|:---:|:---:|
| O-prune | 剪枝 | ✓ | ✓ | 基线 |
| S-prune | 剪枝 | ✓ | ✓ | 基线 |
| F-prune | 剪枝 | ✓ | ✓ | 基线 |
| M-SMoE | 合并 | ✗ | ✗ | 低于基线 |
| HC-SMoE | 合并 | ✓ | ✓ | **全面最优** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|:---:|:---:|:---:|
| Router logits 聚类 | 较低准确率 | M-SMoE 方式，任务敏感 |
| 参数空间聚类 | 较低准确率 | 高维空间距离度量失效 |
| **专家输出聚类** | **最高准确率** | HC-SMoE，捕获功能等价性 |
| K-means 聚类 | 不稳定 | 对初始化敏感 |
| One-pass 分组 | 较低质量 | M-SMoE 方式 |
| **层次聚类** | **最优且稳定** | HC-SMoE，确定性结果 |
| 均匀权重合并 | 可接受 | 简单平均 |
| **频率加权合并** | **最优** | HC-SMoE 方式 |

### 关键发现

1. **输出相似度优于 router logits**：专家输出直接反映功能等价性，而 router logits 是输入依赖的分配偏好，聚类效果差异显著
2. **层次聚类优于 one-pass 和 K-means**：迭代式合并保证全局最优聚类质量，且结果确定性强
3. **聚类质量是合并效果的关键**：好的聚类 + 简单合并策略即可取得优异结果，差的聚类即使用复杂合并策略也难以弥补
4. **合并优于剪枝**：在同等压缩率下，合并保留了更多知识，性能显著优于直接删除专家
5. **跨数据集泛化性强**：在不同校准数据集上聚类结果稳定，验证了任务无关性
6. **M-SMoE 在任务无关设置下失效**：依赖频率的分组和保留策略在零样本设置下表现差

## 亮点与洞察

1. **思路简洁有效**：将专家合并问题分解为"度量→聚类→合并"三步，每一步都给出了清晰的设计理由和对比实验
2. **输出 vs Router 的洞察**：指出 router logits 本质上是输入依赖的路由偏好，不等于专家功能相似度，这一观察对后续 MoE 研究有广泛启发
3. **理论与实践兼顾**：提供了层次聚类质量的理论分析，同时也给出了充分的实验验证
4. **工程友好**：完全无需重训练，仅需一次前向传播 + 层次聚类，实际部署中计算成本极低
5. **方法通用性**：在 Qwen（60 个专家）和 Mixtral（8 个专家）两种不同规模的 MoE 架构上均有效，展示了跨模型的泛化能力

## 局限与展望

1. **仅验证了语言任务**：所有实验基于零样本语言理解任务，缺乏在生成任务（摘要、翻译等）上的验证
2. **静态分组策略**：每层保留相同数量的专家，未考虑不同层冗余程度不同——动态分配可能获得更好的压缩-性能平衡
3. **校准数据集敏感性**：虽然论文声称任务无关，但校准数据集的选择仍可能影响专家输出的代表性
4. **合并策略较简单**：仅使用频率加权平均，未探索更复杂的合并方式（如 TIES-Merging、DARE 等）
5. **缺乏与量化等方法的组合**：未探索 HC-SMoE 与权重量化、知识蒸馏等互补压缩技术的结合
6. **层次聚类的扩展性**：当专家数显著增加时（如未来可能的百/千专家模型），$O(n^2)$ 的聚类复杂度可能成为瓶颈

## 相关工作与启发

- **M-SMoE (Li et al., 2024)**：最直接的前序工作，提出了专家合并框架但依赖 router logits 和频率信息，HC-SMoE 在度量、聚类、合并三方面均做了改进
- **ZipIt (Stoica et al., 2024)**：模型合并领域的工作，利用特征相关性合并不同任务模型，启发了专家合并可视为多模型合并问题
- **O-prune / S-prune (Lu et al., 2024; He et al., 2024)**：剪枝方向的工作，验证了专家冗余的存在，但直接删除造成知识损失
- **对 MoE 压缩的启发**：输出代表向量 + 层次聚类的范式可能推广到更多 MoE 变体（如 Soft-MoE、Expert Choice）

## 评分

| 维度 | 分数 | 说明 |
|:---:|:---:|:---|
| 新颖性 | 7/10 | 层次聚类+输出度量组合新颖，但单个组件并非全新 |
| 技术深度 | 7/10 | 理论分析扎实，但方法本身较为直观 |
| 实验充分性 | 8/10 | 覆盖多模型、多压缩率、多消融，但缺生成任务验证 |
| 实用价值 | 9/10 | 无需重训练+任务无关+代码开源，工程实用性极强 |
| 写作质量 | 8/10 | 结构清晰，动机论述充分 |
| **总分** | **7.8/10** | 扎实的工程改进型工作，实用价值突出 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Long-Context Modeling with Dynamic Hierarchical Sparse Attention for On-Device LLMs](../../NeurIPS2025/llm_efficiency/long-context_modeling_with_dynamic_hierarchical_sparse_attention_for_on-device_l.md)
- [\[NeurIPS 2025\] Hardware-aligned Hierarchical Sparse Attention for Efficient Long-term Memory Access](../../NeurIPS2025/llm_efficiency/hardware-aligned_hierarchical_sparse_attention_for_efficient_long-term_memory_ac.md)
- [\[ICLR 2026\] Understanding and Improving Length Generalization in Hierarchical Sparse Attention Models](../../ICLR2026/llm_efficiency/understanding_and_improving_length_generalization_in_hierarchical_sparse_attenti.md)
- [\[NeurIPS 2025\] MoESD: Unveil Speculative Decoding's Potential for Accelerating Sparse MoE](../../NeurIPS2025/llm_efficiency/moesd_unveil_speculative_decodings_potential_for_accelerating_sparse_moe.md)
- [\[ACL 2025\] Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](../../ACL2025/llm_efficiency/native_sparse_attention.md)

</div>

<!-- RELATED:END -->
