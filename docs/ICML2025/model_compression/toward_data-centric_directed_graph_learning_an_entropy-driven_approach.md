---
title: >-
  [论文解读] Toward Data-centric Directed Graph Learning: An Entropy-driven Approach
description: >-
  [ICML 2025][模型压缩][有向图学习] 提出 EDEN（Entropy-driven Digraph Knowledge Distillation），从数据中心视角构建层级知识树（HKT），通过有向拓扑结构度量和节点互信息量化，揭示有向图中拓扑与节点属性的潜在关联，作为即插即用模块可为任意 DiGNN 带来平均 2-5% 的性能提升，在 14 个数据集和 4 个下游任务上取得 SOTA。
tags:
  - "ICML 2025"
  - "模型压缩"
  - "有向图学习"
  - "数据中心ML"
  - "知识蒸馏"
  - "信息熵"
  - "层级编码"
---

# Toward Data-centric Directed Graph Learning: An Entropy-driven Approach

**会议**: ICML 2025  
**arXiv**: [2505.00983](https://arxiv.org/abs/2505.00983)  
**代码**: 待确认  
**领域**: 模型压缩  
**关键词**: 有向图学习, 数据中心ML, 知识蒸馏, 信息熵, 层级编码

## 一句话总结

提出 EDEN（Entropy-driven Digraph Knowledge Distillation），从数据中心视角构建层级知识树（HKT），通过有向拓扑结构度量和节点互信息量化，揭示有向图中拓扑与节点属性的潜在关联，作为即插即用模块可为任意 DiGNN 带来平均 2-5% 的性能提升，在 14 个数据集和 4 个下游任务上取得 SOTA。

## 研究背景与动机

**领域现状**：有向图神经网络（DiGNN）近年来受到关注，包括基于谱域的方法（MagNet、HoloNet）和空间域的方法（Dir-GNN、NSTE），它们通过考虑边的方向性来建模复杂拓扑。然而，现有工作主要集中在模型架构创新（model-centric），忽视了数据层面的深度挖掘。

**现有痛点**：现有 DiGNN 虽然利用了有向边，但未能全面挖掘有向图数据中隐藏的丰富知识——拓扑和节点属性之间存在大量潜在关联（如未被观测到的结构模式），这些数据级别的局限导致模型级别的预测性能次优。此外，无向 GNN 在同质/异质纠缠的图上表现挣扎，根本原因在于忽略了有价值的有向拓扑信息。

**核心矛盾**：数据质量决定了模型性能的上限（data-level limitation → model-level sub-optimal），但现有方法都在模型端做文章，忽视了从数据端提升的可能性。有向图中的拓扑复杂性和节点属性多样性带来了更丰富但也更难解码的数据知识。

**本文目标** 如何从数据中心视角揭示有向图中隐藏的结构知识，并将其作为监督信号注入模型训练过程。

**切入角度**：基于层级编码理论——真实世界有向图受自然噪声影响，其信息熵 $\mathcal{H}$ 由拓扑和属性决定，通过最小化 $\mathcal{H}$ 可以揭示真实结构 $\mathcal{T}$，而数据知识 $\mathcal{K}$ 隐藏在 $\mathcal{T}$ 中。

**核心 idea**：构建层级知识树（HKT）从拓扑和属性双视角对有向图进行层级编码，通过数据级在线知识蒸馏将挖掘的知识注入模型训练。

## 方法详解

### 整体框架

EDEN 的 pipeline 分为三步：(1) **知识发现**——先用有向拓扑结构度量构建粗粒度 HKT，再通过节点互信息神经估计细化 HKT；(2) **知识蒸馏**——将 HKT 中父节点（教师）的知识个性化传递给子节点（学生），实现数据级在线 KD；(3) **叶节点预测**——通过树上随机游走聚合多层级表征，生成叶节点预测用于下游任务。

### 关键设计

1. **拓扑感知的有向结构度量与粗粒度 HKT 构建**:

    - 功能：基于有向图的信息熵量化拓扑不确定性，构建初始层级知识树
    - 核心思路：首先定义一维结构度量 $\mathcal{H}^1(\mathcal{G}) = -\sum_{v \in \mathcal{V}} (\frac{\tilde{d}_v^{in}}{m} \log \frac{\tilde{d}_v^{in}}{m} + \frac{\tilde{d}_v^{out}}{m} \log \frac{\tilde{d}_v^{out}}{m})$，扩展到 $h$ 维分区树 $\mathcal{H}^h(\mathcal{G}) = \min_{\mathcal{T}: Height(\mathcal{T})=h}\{\mathcal{H}^{\mathcal{T}}_{in} + \mathcal{H}^{\mathcal{T}}_{out}\}$，通过贪心算法找最优 HKT 最小化不确定性。创新点：引入反向概率解决有向图非强连通时随机游走路径急剧衰减的问题，并为汇节点添加自环
    - 设计动机：与前人仅用正向随机游走不同，实验发现有向图中正向游走仅 5 步后完整路径比例急剧下降，引入反向概率能捕获更远邻域信息

2. **基于互信息的 HKT 细化**:

    - 功能：利用节点属性信息对粗粒度 HKT 进行细粒度修正
    - 核心思路：对每个分区 $\mathcal{X}_p$，定义分区内和分区间的互信息神经估计。基于 f-散度的 DV 表示和 GAN-like 散度（定理 3.1-3.3），训练神经网络 $F_w$ 估计节点与其广义邻域的互信息，判断节点归属是否正确。准则函数 $C(\Omega) = \hat{I}_{GAN}^{(\Omega)}$ 用于发现每个分区中最有信息量的节点子集
    - 设计动机：仅靠拓扑度量得到的分区过于粗糙，节点属性（特征和标签）在有向图中呈现更复杂的知识模式，需要互信息驱动的细化来确保同分区内高相似性、不同分区间高区分度

3. **节点自适应知识蒸馏**:

    - 功能：将 HKT 中父节点的知识个性化传递给子节点
    - 核心思路：父节点知识通过权重聚合生成 $\mathbf{X}_p = \mathcal{S}_{\Omega_p} \mathbf{X}_{\Omega_p}$，再通过不确定性量化 $\mathcal{U}_p = \sigma(\mathcal{Q}_{parent}(-\sum_i \mathbf{X}_{p,i} \log \mathbf{X}_{p,i}))$ 过滤不明确知识，最终 KD 损失 $\mathcal{L}_{kd} = \|\mathbf{X}_p / \mathcal{U}_p - \mathcal{Q}_{child}(\mathbf{X}_{v_1,v_2})\|_F$。引入 $\mathcal{S}_{v_2}$ 多样性知识防止过拟合
    - 设计动机：不是所有父节点知识都清晰可表达，且每个子节点有独特的图上下文和知识需求，因此需要个性化传递而非统一蒸馏

### 损失函数 / 训练策略

总训练目标 $\mathcal{L} = \mathcal{L}_{\text{cross-entropy}}(\hat{\mathbf{Y}}, \mathbf{Y}) + \alpha \mathcal{L}_{kd}$，其中 $\alpha$ 控制 KD 强度。叶节点预测使用树上随机游走获取多层级表征序列 $\mathcal{P}_{rw}^k$，节点级任务优先采样父/子节点（多层级表征），链接级任务优先采样兄弟节点（同级表征）。轻量化设计：使用蒙特卡洛方法近似最优 HKT、增量训练和原型表示减少计算开销、无参数特征传播替代可学习 GNN。

## 实验关键数据

### 主实验（Node-C 节点分类准确率 %）

| 模型 | CoraML | CiteSeer | WikiCS | Tolokers | Empire | Rating | Arxiv |
|------|--------|----------|--------|----------|--------|--------|-------|
| GCNII | 80.8 | 62.5 | 78.1 | 78.5 | 76.3 | 42.3 | 65.4 |
| HoloNet | 82.5 | 64.1 | 79.2 | 79.4 | 78.7 | 44.5 | 67.5 |
| **EDEN** | **84.6** | **65.8** | **81.4** | **81.3** | **81.1** | **46.3** | **69.7** |

### 即插即用增强效果（Node-C 准确率 %）

| DiGNN + EDEN | CoraML | CiteSeer | WikiCS | Arxiv | 平均提升 |
|-------------|--------|----------|--------|-------|---------|
| Dir-GNN → +EDEN | 82.6→85.9 | 64.5→67.2 | 79.1→82.8 | 66.9→70.5 | **⇑4.68%** |
| HoloNet → +EDEN | 82.5→86.0 | 64.1→67.5 | 79.2→82.6 | 67.5→70.8 | **⇑4.46%** |
| DIMPA → +EDEN | 82.4→85.4 | 64.0→66.9 | 78.8→82.2 | 67.1→69.9 | **⇑4.32%** |
| OptBG → +EDEN | 81.5→82.8 | 62.4→64.6 | 77.9→79.4 | 66.4→67.9 | ↑2.75% |

### 链接级任务（AUC/AP/ACC %）

| 数据集 | 任务 | 最佳基线 | EDEN | 提升 |
|--------|------|---------|------|------|
| Slashdot | Existence (AUC) | 90.6 (NSTE) | **91.8** | +1.2 |
| Slashdot | Direction (AUC) | 92.2 (NSTE/MagNet) | **93.3** | +1.1 |
| Slashdot | Link-C (ACC) | 85.4 (NSTE) | **87.1** | +1.7 |
| WikiTalk | Existence (AUC) | 94.7 (Dir-GNN) | **95.4** | +0.7 |
| WikiTalk | Direction (AUC) | 90.9 (Dir-GNN) | **91.5** | +0.6 |

### 消融实验

| 配置 | Tolokers Node-C | Slashdot Existence | Slashdot Direction |
|------|-----------------|--------------------|--------------------|
| EDEN (Full) | 81.33 | 91.82 | 93.29 |
| w/o Diverse Knowledge | 80.90 (-0.43) | 91.40 (-0.42) | 92.96 (-0.33) |
| w/o Personalized Transfer | 80.84 (-0.49) | 91.49 (-0.33) | 93.01 (-0.28) |
| w/o Tree-based Random Walk | 80.67 (-0.66) | 91.16 (-0.66) | 92.77 (-0.52) |
| w/o KD Loss | 80.01 (-1.32) | 90.84 (-0.98) | 92.25 (-1.04) |

### 关键发现
- KD 损失是最关键组件，去除后平均掉点约 1%，说明数据级知识蒸馏确实是性能提升的核心驱动力
- 树上随机游走是第二重要模块，去除后 Slashdot Existence AUC 掉 0.66，说明多层级表征聚合对预测有显著帮助
- 作为即插即用模块时，EDEN 在 DiGNN（Dir-GNN +4.68%）上比 GNN（OptBG +2.75%）带来更大提升，验证了有向图数据知识更丰富的假设
- EDEN 作为独立方法相比 Node-C 最佳基线平均提升 2.78%，链接级任务平均提升 2.24%

## 亮点与洞察

- **数据中心的图学习范式**：EDEN 首次提出层级数据级知识蒸馏（区别于模型级离线 KD），将"数据质量限制模型上限"的洞察转化为可操作的框架。这种视角可迁移到任何结构化数据学习场景
- **有向拓扑度量的巧妙扩展**：引入反向概率解决非强连通有向图随机游走衰减问题，以及 h 维分区树将一维 Shannon 熵推广到层级结构度量，理论自然且实用
- **模型无关的即插即用设计**：EDEN 可以直接附加到任意 DiGNN 上作为在线 KD 模块，无需修改基础模型架构，实际部署灵活度高

## 局限与展望

- 可扩展性仍是瓶颈，尽管提出了轻量化方案（蒙特卡洛、增量训练、无参数传播），大规模图上的实际效率尚未充分验证
- HKT 的层数 $h$ 是一个超参数，需要对不同图结构调优，缺少自适应选择机制
- 互信息神经估计需要额外的神经网络训练（MLP 等），增加了整体训练复杂度
- 仅在同质和异质无向图上做了扩展实验，在动态图、超图等更复杂图类型上的适用性未探索

## 相关工作与启发

- **vs 传统 DiGNN (Dir-GNN, HoloNet)**: 这些方法从模型端利用有向边，EDEN 从数据端补充——两者正交互补，EDEN+Dir-GNN 的组合效果最好
- **vs 图结构学习 (CoGSL)**: CoGSL 通过互信息优化图视图生成和融合，EDEN 借鉴类似思路但聚焦于层级结构发现而非视图增强
- **vs 图对比学习 (DGI, GMI)**: DGI/GMI 最大化节点与图/邻域的互信息做自监督，EDEN 的互信息用于知识树修正和知识蒸馏，目标不同但信息论工具共享

## 评分

- 新颖性: ⭐⭐⭐⭐ 数据中心+层级编码理论在图学习中是新颖视角，但 HKT 构建在本质上是层级聚类的信息论重述
- 实验充分度: ⭐⭐⭐⭐⭐ 14 个数据集、4 个下游任务、即插即用验证、消融实验、效率分析，非常全面
- 写作质量: ⭐⭐⭐⭐ 理论推导严谨详尽，但公式密集可能影响可读性
- 价值: ⭐⭐⭐⭐ 数据中心视角和即插即用特性使其有实际应用价值，但核心贡献的不可替代性需进一步验证

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] CodeGEMM: A Codebook-Centric Approach to Efficient GEMM in Quantized LLMs](../../NeurIPS2025/model_compression/codegemm_a_codebook-centric_approach_to_efficient_gemm_in_quantized_llms.md)
- [\[ICML 2025\] Beyond Communication Overhead: A Multilevel Monte Carlo Approach for Mitigating Compression Bias in Distributed Learning](beyond_communication_overhead_a_multilevel_monte_carlo_approach_for_mitigating_c.md)
- [\[ACL 2025\] Prompt Candidates, then Distill: A Teacher-Student Framework for LLM-driven Data Annotation](../../ACL2025/model_compression/prompt_distill_teacher_student.md)
- [\[ACL 2026\] Alignment Tuning for Large Language Models: A Data-Centric Lens on Alignment Data Pipelines](../../ACL2026/model_compression/alignment_tuning_for_large_language_models_a_data-centric_lens_on_alignment_data.md)
- [\[ACL 2025\] Magnet: Multi-turn Tool-use Data Synthesis and Distillation via Graph Translation](../../ACL2025/model_compression/magnet_multi-turn_tool-use_data_synthesis_and_distillation_via_graph_translation.md)

</div>

<!-- RELATED:END -->
