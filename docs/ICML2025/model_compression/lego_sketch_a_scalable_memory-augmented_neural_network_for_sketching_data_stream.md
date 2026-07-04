---
title: >-
  [论文解读] Lego Sketch: A Scalable Memory-augmented Neural Network for Sketching Data Streams
description: >-
  [ICML2025][模型压缩][data stream sketch] 提出 Lego Sketch，一种基于模块化"记忆积木"的可扩展记忆增强神经网络（MANN），通过 normalized multi-hash embedding、可扩展内存和自引导加权损失，解决了现有 neural sketch 在跨数据域和不同空间预算下需要重新训练的可扩展性难题，并首次给出了 neural sketch 的误差上界。
tags:
  - "ICML2025"
  - "模型压缩"
  - "data stream sketch"
  - "frequency estimation"
  - "memory-augmented neural network"
  - "scalability"
  - "meta-learning"
---

# Lego Sketch: A Scalable Memory-augmented Neural Network for Sketching Data Streams

**会议**: ICML2025  
**arXiv**: [2505.19561](https://arxiv.org/abs/2505.19561)  
**代码**: [FFY0/LegoSketch_ICML](https://github.com/FFY0/LegoSketch_ICML)  
**领域**: 模型压缩  
**关键词**: data stream sketch, frequency estimation, memory-augmented neural network, scalability, meta-learning

## 一句话总结

提出 Lego Sketch，一种基于模块化"记忆积木"的可扩展记忆增强神经网络（MANN），通过 normalized multi-hash embedding、可扩展内存和自引导加权损失，解决了现有 neural sketch 在跨数据域和不同空间预算下需要重新训练的可扩展性难题，并首次给出了 neural sketch 的误差上界。

## 研究背景与动机

### 问题定义

数据流频率估计（stream item frequency estimation）：给定一个无限长的数据流 $\mathcal{X} = (x_1, \dots, x_N)$，包含 $n$ 个不同元素，目标是在亚线性空间内准确估计任意元素 $e_i$ 的出现频率 $f_i$。Sketch 是解决该问题的经典概率数据结构。

### 已有方法的不足

**手工 sketch**（CM-Sketch、C-Sketch 等）：依赖预定义的 2D 数组 + 哈希函数 + 固定策略，空间-精度权衡有上限

**Neural sketch**（Meta-Sketch 系列）：用 MANN 替代手工核心结构，在紧凑空间预算下精度更高，但存在两大可扩展性瓶颈：
   - **域可扩展性差**：嵌入模块使用 MLP/CNN 提取域特定特征，切换数据域（如从网页点击流到文本流）需重训练
   - **预算可扩展性差**：固定大小的单一内存块 $M \in \mathbb{R}^{d_1 \times d_2}$，变更空间预算需重训练，且训练成本随预算增大而增长
3. 如 Figure 2 所示，Meta-Sketch 在高预算区间相对手工 sketch 的优势反而递减

### 核心动机

像搭乐高积木一样，设计一种模块化的 MANN 架构，使 neural sketch 能在**不重新训练**的前提下自由适配不同数据域和空间预算。

## 方法详解

### 整体架构

Lego Sketch 由五个模块组成，协同完成 Store 和 Query 两种操作：

| 模块 | 符号 | 功能 |
|------|------|------|
| Scalable Embedding | $\mathcal{E}$ | 域无关的归一化多哈希嵌入 |
| Hash Addressing | $\mathcal{A}$ | 哈希产生稀疏地址向量 |
| Scalable Memory | $\mathcal{M}$ | 管理 $K$ 个记忆积木，哈希分流 |
| Memory Scanning | $\mathcal{S}$ | 从压缩记忆中重建流特征 |
| Ensemble Decoding | $\mathcal{D}$ | 融合所有信息输出频率估计 |

每次 Store/Query 操作均为单次前向传播，计算复杂度 $O(1)$，与流长度无关。

### 模块 1：Normalized Multi-Hash Embedding（$\mathcal{E}$）

传统 neural sketch 使用 MLP 编码器提取域特定特征，导致跨域时需要重训练。Lego Sketch 的嵌入模块**不提取任何域特定特征**，转而生成满足特定偏度范围的嵌入向量，实现域无关性。

具体做法受 NLP 领域 Hash Embeddings 启发：
- 维护一个可学习向量 $V$ 和 $d_1$ 个独立哈希函数 $\{\mathcal{H}_1, \dots, \mathcal{H}_{d_1}\}$
- 每个哈希函数将 $x_i$ 映射到 $V$ 的一个索引，取出对应值，得到 $v_i \in \mathbb{R}^{d_1}$
- 对 $v_i$ 做 **$L_1$ 归一化**

$L_1$ 归一化的关键作用：在加法式内存存储中保持累积稳定性，直接提升估计精度。论文 Section 4.1 给出了域无关可扩展性的理论证明。

### 模块 2：Hash Addressing（$\mathcal{A}$）

用另一组 $d_1$ 个哈希函数 $\{\mathcal{H}'_1, \dots, \mathcal{H}'_{d_1}\}$ 生成稀疏地址向量：

$$a_i = \mathcal{A}(x_i) = \text{SparseVector}(\mathcal{H}'_1(x_i), \dots, \mathcal{H}'_{d_1}(x_i)) \in \mathbb{R}^{d_1 \times d_2}$$

哈希映射位置赋值 1，其余为 0。相比 Meta-Sketch 的学习式寻址，这种纯哈希方式更简洁且支持可扩展。

### 模块 3：Scalable Memory（$\mathcal{M}$）

核心创新——用 $K$ 个**记忆积木** $M_1, \dots, M_K$ 替代单一固定内存块：

- **Store**：哈希函数 $\mathcal{H}$ 将 $x_i$ 分配到 $M_{\mathcal{H}(x_i)}$，执行加法写入：
  $$M_{\mathcal{H}(x_i)} = M_{\mathcal{H}(x_i)} + v_i \circ a_i$$
  其中 $\circ$ 为逐元素乘法
- **Query**：同理定位对应积木，提取信息：
  $$m_i = M_{\mathcal{H}(x_i)}^T a_i$$

原始流 $\mathcal{X}$ 被均匀划分为 $K$ 个子流，每个子流独立压缩到一个积木中。扩展空间预算只需增加 $K$，**完全不需要重训练**。实验中可轻松扩展到 140MB 处理亿级数据流。

### 模块 4：Memory Scanning（$\mathcal{S}$）

创新性模块，基于 **DeepSets** 架构从压缩记忆中自主重建流的统计特征（如元素数量 $n$、分布偏度 $\alpha$ 等）。

输出三个特征：$s_{\mathcal{H}(x_i)}, s^{(n)}_{\mathcal{H}(x_i)}, s^{(\alpha)}_{\mathcal{H}(x_i)}$，供解码模块使用。这使得模型能够在查询时感知当前子流的分布特性，显著提升估计精度。

### 模块 5：Ensemble Decoding（$\mathcal{D}$）

融合记忆读取 $m_i$、嵌入向量 $v_i$ 和扫描特征 $s$，输出频率估计 $\hat{f}_i$。

### 自引导加权损失（Self-Guided Weighting Loss）

传统 neural sketch 使用均匀加权的元学习损失 $\mathcal{L}_o$，但不同 meta-task 的难度差异很大。本文提出的自引导损失 $\mathcal{L}'$ 根据模型自身在每个 meta-task 上的表现**动态调整权重**，有效解决高预算下优势递减的问题。

### 理论保证

1. **域无关可扩展性**：证明 normalized multi-hash embedding 的偏度在可控范围内，无需针对特定域重训练
2. **预算无关可扩展性**：增加记忆积木 $K$ 等价于线性提升空间预算，不影响已训练参数
3. **误差上界**：首次为 neural sketch 提供估计误差的理论上界，填补领域空白

## 实验关键数据

### 数据集

| 数据集 | 类型 | 规模 | 说明 |
|--------|------|------|------|
| AOL | 搜索日志 | 真实 | 网页搜索查询流 |
| CAIDA | 网络流量 | 真实 | IP 数据包流 |
| Zipf (synthetic) | 合成 | 可配置 | 不同偏度参数的 Zipf 分布 |
| 大规模流 | 多域 | 1亿级 | 验证可扩展性 |

### 主要对比方法

- **手工 sketch**：CM-Sketch、C-Sketch、CU-Sketch、A-Sketch
- **Neural sketch**：Meta-Sketch 及其变体
- **指标**：AAE（平均绝对误差）、ARE（平均相对误差）

### 核心结果

1. **空间-精度权衡**（Figure 2, AOL 数据集）：在所有空间预算下 Lego Sketch 均取得最低误差，且随预算增大优势持续增长（不像 Meta-Sketch 在高预算区间优势递减）
2. **跨域泛化**：同一个训练好的模型直接部署到不同数据域（AOL→CAIDA 等），**无需重训练**即保持竞争力甚至超越域内训练的 Meta-Sketch
3. **大规模可扩展性**：通过增加记忆积木 $K$ 将总内存扩大到 140MB，处理 1 亿级数据流，精度持续提升
4. **消融实验**（Section 5.5）验证了各模块的贡献：
    - $L_1$ 归一化 → 精度显著提升
    - Memory Scanning → 对高预算场景帮助最大
    - Self-guided loss → 解决高预算优势递减

### 作为核心结构的衍生能力

Section 3.3 展示 Lego Sketch 可作为核心结构替换 CM-Sketch 等，与 filter 等外部增强组件组合，构建更强的衍生 sketch 系统。

## 亮点与洞察

1. **乐高积木隐喻恰当**：模块化记忆设计让可扩展性从"需要重训练"变为"加积木即可"，工程和理论上都优雅
2. **域无关嵌入是关键创新**：放弃提取域特定特征，转而用哈希+归一化生成域无关嵌入，思路反直觉但有效
3. **首个 neural sketch 误差上界**：将理论分析引入 neural sketch 领域，提升了方法的可信度
4. **Memory Scanning 的自省能力**：让模型在查询时"感知"当前内存中存储的流的统计特性，是一个巧妙的信息利用方式
5. **代码开源**，可复现性好

## 局限与展望

1. **训练仍需元学习**：虽然部署时跨域/跨预算无需重训练，但初始训练阶段仍需自监督元学习，训练成本不低
2. **哈希函数质量依赖**：多处使用独立哈希函数，实际应用中哈希函数的独立性和均匀性可能影响性能
3. **仅关注频率估计**：数据流还有其他查询任务（如分位数、heavy hitter 检测等），方法是否推广未验证
4. **缓存内容截断**：论文中 Memory Scanning 和 Self-guided Loss 的完整公式细节未在缓存中完全提供，仅有框架描述
5. **与 counter-based 方法不可直接比较**：论文也承认 sketch 和 counter-based summarization（如 SpaceSaving）适用场景不同，但在只支持插入的流模型下是否仍有优势值得讨论
6. **实际工程部署成本**：MANN 前向推理相比手工 sketch 的简单哈希+数组操作仍有额外开销，延迟敏感场景可能受限

## 相关工作与启发

- **CM-Sketch / C-Sketch**：经典手工 sketch 核心结构，Lego Sketch 实质是其 neural 升级版
- **Meta-Sketch**（Cao et al., 2023, 2024）：直接前身，用固定 MANN 做 neural sketch，Lego Sketch 解决了其可扩展性问题
- **Hash Embeddings**（Svenstrup et al., 2017）：NLP 嵌入技术，启发了 normalized multi-hash embedding 的设计
- **DeepSets**（Zaheer et al., 2017）：置换不变架构，用于 Memory Scanning 模块
- **MANN / 元学习**（Santoro et al., 2016; Graves et al., 2016）：记忆增强神经网络范式，Lego Sketch 的基础架构

**启发**：模块化设计+域无关表征的组合思路，可能启发其他需要跨域泛化的流数据处理任务（如异常检测、变化点检测等）。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 模块化记忆积木 + 域无关嵌入 + 首个 neural sketch 理论分析，多个新颖点结合
- 实验充分度: ⭐⭐⭐⭐ — 多数据集、消融、大规模实验齐全，跨域泛化验证有说服力
- 写作质量: ⭐⭐⭐⭐ — 乐高隐喻贯穿全文，结构清晰，理论与实验结合好
- 价值: ⭐⭐⭐⭐ — 解决 neural sketch 实际部署的核心痛点，有明确的工程价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] LightMem: Lightweight and Efficient Memory-Augmented Generation](../../ICLR2026/model_compression/lightmem_lightweight_and_efficient_memory-augmented_generation.md)
- [\[ICML 2025\] Predictive Data Selection: The Data That Predicts Is the Data That Teaches](predictive_data_selection_the_data_that_predicts_is_the_data_that_teaches.md)
- [\[NeurIPS 2025\] GraSS: Scalable Data Attribution with Gradient Sparsification and Sparse Projection](../../NeurIPS2025/model_compression/grass_scalable_data_attribution_with_gradient_sparsification_and_sparse_projecti.md)
- [\[CVPR 2025\] Sketch Down the FLOPs: Towards Efficient Networks for Human Sketch](../../CVPR2025/model_compression/sketch_down_the_flops_towards_efficient_networks_for_human_sketch.md)
- [\[NeurIPS 2025\] On the Creation of Narrow AI: Hierarchy and Nonlocality of Neural Network Skills](../../NeurIPS2025/model_compression/on_the_creation_of_narrow_ai_hierarchy_and_nonlocality_of_neural_network_skills.md)

</div>

<!-- RELATED:END -->
