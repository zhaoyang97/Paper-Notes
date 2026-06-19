---
title: >-
  [论文解读] InfoCom: Kilobyte-Scale Communication-Efficient Collaborative Perception with Information-Aware Feature Compression
description: >-
  [AAAI 2026][模型压缩][协同感知] 提出InfoCom框架，基于扩展的信息瓶颈原理将协同感知的通信量从MB级压缩至KB级（相比Where2comm降低440倍），同时保持近无损的感知性能，核心包含信息感知编码、稀疏掩码生成和多尺度解码三个模块。 协同感知通过多智能体信息共享来弥补单智能体感知的局限性（遮挡、远距离…
tags:
  - "AAAI 2026"
  - "模型压缩"
  - "协同感知"
  - "通信效率"
  - "信息瓶颈"
  - "特征压缩"
  - "3D目标检测"
---

# InfoCom: Kilobyte-Scale Communication-Efficient Collaborative Perception with Information-Aware Feature Compression

**会议**: AAAI 2026  
**arXiv**: [2512.10305](https://arxiv.org/abs/2512.10305)  
**代码**: [GitHub](https://weiquanmin.github.io/infocom)  
**领域**: 模型压缩  
**关键词**: 协同感知, 通信效率, 信息瓶颈, 特征压缩, 3D目标检测

## 一句话总结

提出InfoCom框架，基于扩展的信息瓶颈原理将协同感知的通信量从MB级压缩至KB级（相比Where2comm降低440倍），同时保持近无损的感知性能，核心包含信息感知编码、稀疏掩码生成和多尺度解码三个模块。

## 研究背景与动机

协同感知通过多智能体信息共享来弥补单智能体感知的局限性（遮挡、远距离等），是自动驾驶安全性的关键保障。然而，现有协同感知方法面临一个根本性的通信-性能权衡：

现有的通信高效方法分为两类：(1) **特征选择方法**（如Where2comm），选择性传输关键特征，但由于特征维度高，通信量仍在MB级；(2) **特征压缩方法**（如ERMVP），将特征映射到低维空间，通信量有所降低但仍远超实际网络约束。核心矛盾在于：这些方法均假设MB级的通信带宽可用，但5G车联网场景的平均速率仅3.5 MB/s，且可能波动至0.4 MB/s以下，使得MB级通信在实际中不可靠。

更根本的问题是，现有方法缺乏通信-性能权衡的**理论分析**，仅通过启发式设计优化。本文从信息论的视角出发，提出了一种**信息纯化范式**（information purification paradigm）：不再在特征空间中操作，而是直接利用信息瓶颈(IB)原理提取最小充分的任务关键信息。但标准IB直接应用受限于数据处理不等式 $I(Z;Y) \leq I(X;Y)$，极端压缩与高精度感知之间存在固有张力。InfoCom通过扩展IB的马尔可夫链并引入稀疏掩码作为辅助信息来突破这一限制。

## 方法详解

### 整体框架

InfoCom是一个即插即用的通信高效协同感知框架，仅需替换现有协同系统的通信层并在训练损失中加入IB正则化项。流程为：各智能体用本地编码器提取中间BEV特征 $Z$ → 信息感知编码压缩为极低维消息 $E$ → 稀疏掩码生成识别空间线索 $M$ → 传输 $\{E, M\}$（KB级）→ 接收端多尺度解码恢复可操作的BEV特征 → 融合网络聚合多视角信息 → 输出3D检测结果。

### 关键设计

1. **信息感知编码（Information-Aware Encoding, IAE）**:

    - 功能：将高维中间特征 $Z \in \mathbb{R}^{N \times C \times H \times W}$ 压缩为极低维的信息感知特征 $E \in \mathbb{R}^{N \times D}$，其中 $D \ll C \times H \times W$
    - 核心思路：扩展标准IB的马尔可夫链从 $Y \to X \to Z$ 到 $Y \to X \to Z \to (E, M)$，将空间线索解耦为辅助信息 $M$，使 $E$ 专注于保留感知关键信息。优化目标为 $E, M = \arg\min_{E,M} -I(E,M;Y) + \beta I(E,M;Z)$。IAEncoder输出高斯参数 $(\mu_i, \sigma_i)$，通过重参数化技巧采样 $E_i = \mu_i + \sigma_i \odot \epsilon_i$
    - 设计动机：标准IB仅在sufficiency和minimality之间权衡，无法同时实现极端压缩和高精度。通过引入超低维特征空间和辅助掩码，将IB的内在矛盾分解为两个子问题

2. **稀疏掩码生成（Sparse Mask Generation, SMG）**:

    - 功能：以极小的通信开销提供关键的空间先验信息，弥补极端压缩下的信息损失
    - 核心思路：通过多尺度卷积（$3 \times 3, 5 \times 5, 7 \times 7$）和投影层生成空间重要性掩码 $M_i \in \mathbb{R}^{H \times W}$，然后通过两步联合压缩后处理：(1) 过滤——仅保留top-$k$关键位置，$k = \lfloor \alpha \cdot HW \rfloor$，$\alpha = 0.1$；(2) 量化——统一量化到 $b=4$ bit精度。梯度通过直通估计器（STE）传播
    - 设计动机：数据处理不等式表明 $D/(C \times H \times W)$ 的压缩比越高，任务相关信息损失的风险越大。但极端压缩 $E$ 释放了带宽空间用于传输辅助空间线索。实验证实仅10%的空间位置即提供了主要收益，高精度表示也无必要

3. **多尺度解码（Multi-Scale Decoding, MSD）**:

    - 功能：在接收端从极度压缩的消息 $\{E, M^q\}$ 渐进式重建可操作的BEV特征
    - 核心思路：三步流程——(1) 特征初始化：$E$ 经全连接和转置卷积扩展为低分辨率特征图 $F_{init}^0 \in \mathbb{R}^{C^0 \times H^0 \times W^0}$；(2) 掩码引导调制：下采样后的掩码与初始特征逐元素相乘 $F^0 = F_{init}^0 \odot M^0$，引导重建关注任务关键区域；(3) 多尺度重建：级联解码块逐步上采样，每步分辨率翻倍、通道减半，经 $K$ 次迭代达到目标分辨率
    - 设计动机：不同于简单的特征重建，MSD聚焦于恢复感知信息（而非完整特征），掩码引导使重建资源集中在关键区域

### 损失函数 / 训练策略

总损失函数为 $\mathcal{L} = \mathcal{L}_{detect} + \beta \text{KL}(p(E|Z) \| r(E))$，其中检测损失 $\mathcal{L}_{detect}$ 隐式最大化 $I(E,M;Y)$，KL散度项在高斯先验 $r(E) = \mathcal{N}(0, I)$ 下有闭式解，控制 $I(E;Z)$。两项联合优化实现噪声抑制（由Lemma 1理论保证：$I(E,M;Y_N) \leq I(E,M;Z) - I(E,M;Y)$）。

## 实验关键数据

### 主实验

| 数据集 | 方法 | 通信量 | AP@50 | AP@70 |
|--------|------|--------|-------|-------|
| OPV2V | Standard Colla. | 34.375 MB | 0.9653 | 0.9229 |
| OPV2V | Where2comm | 3.439 MB | 0.9463 | 0.8820 |
| OPV2V | ERMVP | 0.741 MB | 0.9557 | 0.9127 |
| OPV2V | **InfoCom** | **7.875 KB** | **0.9650** | **0.9202** |
| V2XSet | Standard Colla. | 34.375 MB | 0.9212 | 0.8426 |
| V2XSet | ERMVP | / | OOM | OOM |
| V2XSet | **InfoCom** | **7.875 KB** | **0.9273** | **0.8488** |
| DAIR-V2X | Standard Colla. | 24.609 MB | 0.7843 | 0.6353 |
| DAIR-V2X | ERMVP | 0.531 MB | 0.7791 | 0.6324 |
| DAIR-V2X | **InfoCom** | **5.922 KB** | **0.7789** | **0.6385** |

### 消融实验

| 配置 | 变体 | Mean AP | 说明 |
|------|------|---------|------|
| InfoCom完整 | Full | 0.9518 | 基线 |
| IAE | Simple Encoder | 0.9320 | 简化编码器，降2% |
| SMG | Simple Generator | 0.9379 | 简化掩码生成，降1.4% |
| 量化 | w/o STE | 0.8845 | 无直通估计器，降6.7%（严重） |
| MSD | w/o Mask | 0.8839 | 无掩码引导，降6.8%（严重） |
| MSD | w/o Multi-Scale Rec. | 0.9439 | 单尺度重建，降0.8% |

### 关键发现

- InfoCom将通信量从MB压缩到KB（7.875 KB vs Where2comm的3.439 MB，440倍降低），同时AP@50仅下降0.13个百分点
- 在V2XSet上ERMVP出现OOM，InfoCom反而超越了标准协作的性能（AP@50: 0.9273 vs 0.9212）
- 集成到弱backbone（AttFuse、MKD-Cooper）中时，InfoCom甚至提升了原模型性能1.27%，信息纯化机制有效抑制了噪声
- 掩码引导和STE是最关键的组件（去除后性能下降约6.8%），证明空间先验对极端压缩下的信息恢复不可或缺
- 保留率 $\alpha > 0.1$ 时性能增益不足0.2%，量化到4bit精度后AP波动不足0.18%，印证了空间线索的固有稀疏性

## 亮点与洞察

- 从信息论角度重新定义了协同感知的通信效率问题，建立了通信-性能权衡的理论基础（不只是工程优化）
- 信息纯化范式是概念上的突破：不再试图压缩或选择特征（在特征空间操作），而是直接提取最小充分信息（在信息空间操作）
- 扩展IB马尔可夫链 $Y \to X \to Z \to (E, M)$ 的设计巧妙地将压缩与空间先验解耦，使两者可独立优化
- 即插即用架构设计实用价值高：仅替换通信层即可与现有协同感知模型兼容，降低了部署门槛
- Lemma 1的噪声抑制理论 $I(E,M;Y_N) \leq I(E,M;Z) - I(E,M;Y)$ 为"信息压缩反而能提升性能"提供了理论解释

## 局限与展望

- 计算开销约为Where2comm的2倍（从Fig 4(a)可见），虽然传输时间大幅减少，但端到端延迟的改进程度取决于具体网络条件
- IAEncoder使用简单的残差块设计（面向资源受限的智能体），更强的编码器可能进一步提升性能
- 仅在3D目标检测任务上验证，未扩展到占据预测、运动预测等其他协同感知任务
- 理论分析基于高斯先验假设，实际特征分布可能偏离此假设
- 极端压缩（KB级）对延迟敏感型场景有利，但在带宽充裕时是否不如MB级方法尚不明确
- 量化的STE梯度估计可能影响训练稳定性，论文中未讨论训练收敛性

## 相关工作与启发

- Where2comm (Hu et al.) 通过空间重要性加权选择关键信息，是特征选择方法的代表；ERMVP通过空间过滤和聚类实现SOTA通信效率
- 信息瓶颈理论 (Tishby et al.) 提供了表示学习的数学框架，本文首次将其系统性地应用于协同感知的通信优化
- CoAlign (Lu et al.) 是本文默认的协同感知基座模型，采用多尺度特征
- 启发：在通信受限场景中，"传输信息"比"传输特征"更高效——这一思路可推广到联邦学习、分布式推理等更广泛的场景

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （信息纯化范式有概念突破，KB级通信效率是量级飞跃）
- 实验充分度: ⭐⭐⭐⭐⭐ （三个数据集、多个backbone、详尽的消融和可视化分析）
- 写作质量: ⭐⭐⭐⭐ （理论与实验结合紧密，但符号较多，阅读门槛稍高）
- 价值: ⭐⭐⭐⭐⭐ （解决了实际部署中的通信瓶颈，理论+实践双重贡献）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Sharp Eyes and Memory for VideoLLMs: Information-Aware Visual Token Pruning for Efficient and Reliable VideoLLM Reasoning](sharp_eyes_and_memory_for_videollms_information-aware_visual_token_pruning_for_e.md)
- [\[AAAI 2026\] Distillation Dynamics: Towards Understanding Feature-Based Distillation in Vision Transformers](distillation_dynamics_towards_understanding_feature-based_di.md)
- [\[ICML 2026\] Learned Subspace Compression for Communication-Efficient Pipeline Parallelism](../../ICML2026/model_compression/learned_subspace_compression_for_communication-efficient_pipeline_parallelism.md)
- [\[ACL 2026\] Efficient Learned Data Compression via Dual-Stream Feature Decoupling](../../ACL2026/model_compression/efficient_learned_data_compression_via_dual-stream_feature_decoupling.md)
- [\[AAAI 2026\] Distilling Cross-Modal Knowledge via Feature Disentanglement](distilling_cross-modal_knowledge_via_feature_disentanglement.md)

</div>

<!-- RELATED:END -->
