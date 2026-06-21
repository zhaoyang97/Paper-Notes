---
title: >-
  [论文解读] PlanetAlign: A Comprehensive Python Library for Benchmarking Network Alignment
description: >-
  [ICLR 2026][网络对齐] 提出 PlanetAlign，一个集成 18 个跨 6 个领域的数据集、14 种覆盖三大类别（一致性、嵌入、最优传输）方法和标准化评估流程的 PyTorch 网络对齐基准库，通过大规模系统实验揭示了 OT 类方法（PARROT/JOENA）在有效性上的全面领先以及各类方法在可扩展性和鲁棒性上的差异化表现。
tags:
  - "ICLR 2026"
  - "网络对齐"
  - "基准库"
  - "图匹配"
  - "最优传输"
  - "评估框架"
---

# PlanetAlign: A Comprehensive Python Library for Benchmarking Network Alignment

**会议**: ICLR 2026  
**arXiv**: [2505.21366](https://arxiv.org/abs/2505.21366)  
**代码**: [GitHub](https://github.com/yq-leo/PlanetAlign)  
**领域**: LLM评测  
**关键词**: 网络对齐, 基准库, 图匹配, 最优传输, 评估框架

## 一句话总结

提出 PlanetAlign，一个集成 18 个跨 6 个领域的数据集、14 种覆盖三大类别（一致性、嵌入、最优传输）方法和标准化评估流程的 PyTorch 网络对齐基准库，通过大规模系统实验揭示了 OT 类方法（PARROT/JOENA）在有效性上的全面领先以及各类方法在可扩展性和鲁棒性上的差异化表现。

## 研究背景与动机

**领域现状**：网络对齐（Network Alignment, NA）旨在发现不同网络中节点的对应关系，是跨社交网络用户匹配、蛋白质同源性发现、知识图谱融合、反欺诈检测等多种下游任务的关键基础。研究领域已发展出三大类方法：一致性方法（如 IsoRank、FINAL）、嵌入方法（如 REGAL、BRIGHT）和最优传输方法（如 PARROT、JOENA），但方法之间的系统性比较一直是缺失的。

**现有痛点**：已有的五个 NA 基准/库（SGAPBSA、CAPABN、ASNets、NAB、OpenEA）均存在明显的局限性：(1) 数据集局限于单一领域——SGAPBSA 和 CAPABN 仅有生物网络，ASNets 仅有社交网络，OpenEA 仅有知识图谱；(2) 方法覆盖不全——没有任何现有库包含最新且表现最优的 OT 类方法；(3) 评估维度单一——大多只评估有效性，忽略可扩展性和鲁棒性，且数据集划分方式不统一导致结果不可复现。

**核心矛盾**：NA 领域的方法日益丰富，但评估基础设施严重滞后。研究者在各自不同的数据集、划分方式和指标下报告结果，导致方法之间的真实性能差异无法公平比较，阻碍了研究进展。

**本文目标** 构建一个全面、统一、易用的 NA 基准库，覆盖多领域数据集、多类别方法和多维度评估。

**切入角度**：借鉴 CV/NLP 领域成功的基准库设计经验（如 MMDetection、HuggingFace），通过统一的 API 设计、标准化的数据划分和可复现的评估流程，解决 NA 领域的评估碎片化问题。

**核心 idea**：通过构建覆盖 6 个领域、3 大方法类别的统一基准库 PlanetAlign，首次实现了 NA 方法在有效性/可扩展性/鲁棒性/监督敏感性四个维度上的系统性公平比较。

## 方法详解

### 整体框架

网络对齐（Network Alignment, NA）领域方法越来越多，但大家各用各的数据集、各自的划分和指标报结果，谁强谁弱无从公平比较。PlanetAlign 要解决的就是这个评估碎片化问题：把数据、方法、评估三件事统一进一个基于 PyTorch 的库，用户几行代码就能跑完「加载数据 → 训练算法 → 多维度评估」全流程。

整体架构对应后面三个关键设计、分三层串起来：(1) **数据层**——18 个数据集覆盖社交、出版物、生物、知识图谱、基础设施、通信 6 个领域（设计 1）；(2) **算法层**——14 种 NA 方法统一封装在继承自 `BaseModel` 的类里，共用 `.train()` / `.test()` 接口（设计 2）；(3) **评估层**——标准化的 Hits@K / MRR 指标，加上时间/内存开销追踪和鲁棒性测试工具（设计 3）。三层之间数据流单向流动：从数据层取出两个待对齐网络与锚节点对，喂进算法层算出对齐矩阵 $\mathbf{S}$，再交给评估层在有效性/可扩展性/鲁棒性/监督敏感性四个维度上打分。

### 关键设计

**1. 全面的数据集收集与合成：把跨领域泛化能力变得可测**

之前的库最多覆盖 1-2 个领域，方法在不同网络结构下的真实表现无从对比。PlanetAlign 收集了 18 个数据集，其中 11 个来自真实世界、7 个合成而来，横跨社交（Foursquare-Twitter、Douban 等 4 个）、出版物（ACM-DBLP、Cora、ArXiv 等 3 个）、生物（SacchCere、PPI、GGI 等 3 个）、知识图谱（DBP15K 的 ZH-EN/JA-EN/FR-EN 3 个变体）、基础设施（Italy、Airport、PeMS08 等 3 个）和通信（Phone-Email、Arenas 等 2 个）6 个领域。合成数据集用一套经典的网络扰动策略生成：对原始网络插入 10% 噪声边、再删除 15% 已有边，得到两个带已知排列（ground-truth 对齐）的网络。这样一来，方法在结构特征差异很大的网络上的泛化能力第一次能被横向衡量。

**2. 三大类 14 种方法的统一实现：把 OT 方法首次纳入同一框架**

NA 领域方法日益丰富，却没有任何现有库收齐三大类、尤其漏掉了最新且最强的 OT 方法。PlanetAlign 把一致性方法（IsoRank、FINAL）、嵌入方法（IONE、REGAL、CrossMNA、NetTrans、WAlign、BRIGHT、NeXtAlign、WLAlign）和 OT 方法（PARROT、SLOTAlign、HOT、JOENA）共 14 种全部重写为统一的 PyTorch 实现，让它们继承自同一个 `BaseModel` 基类、共用 `.train()` / `.test()` 接口。库里内置了随机游走重启（RWR）嵌入、锚节点嵌入等常用工具函数，新方法只需极少代码就能接进来。与各方法的官方实现相比，重写版在保持相近有效性的同时最高拿到 3 倍加速——既补齐了现有库最大的空白（OT 方法），也保证了对比是在同一套优化过的代码上进行的。

**3. 多维度标准化评估工具：从只看 Hits@1 扩展到四个维度**

以往的库评估维度单一，通常只报 Hits@1，无法刻画一个方法在实际部署中是否够快、够稳。PlanetAlign 把评估展开成四个维度。有效性用 Hits@K 和 MRR，并对双向对齐（$\mathcal{G}_1 \to \mathcal{G}_2$ 与其反向）都计算后取平均，避免方向偏置；可扩展性靠内置 Logger 自动追踪运行时间和峰值内存；鲁棒性通过工具函数往图里注入不同类型、不同程度的边/属性/监督噪声，观察方法的退化幅度；监督敏感性则通过改变训练比例来看方法对标注量的依赖。整套流程统一随机种子和数据划分，保证结果可复现、可公平比较。

## 实验关键数据

### 主实验：有效性与效率

在 6 个领域数据集上评估 14 种方法（训练比例 20%），报告平均 Hits@1、Hits@10、MRR（%）：

| 方法 | 类别 | 社交 H@1 | 出版物 H@1 | 生物 H@1 | 知识图谱 H@1 | 基础设施 H@1 | 通信 H@1 |
|------|------|---------|-----------|---------|------------|------------|---------|
| JOENA | OT | **18.7** | **73.2** | **63.7** | **66.3** | **62.9** | **66.3** |
| PARROT | OT | 12.6 | 66.6 | 61.6 | 66.0 | 51.8 | 63.3 |
| NetTrans | 嵌入 | 7.2 | 40.7 | 34.2 | 28.8 | 29.3 | 45.2 |
| BRIGHT | 嵌入 | 5.1 | 40.4 | 30.5 | 30.4 | 29.9 | 50.9 |
| NeXtAlign | 嵌入 | 7.1 | 43.2 | 25.9 | 27.5 | 28.0 | 29.6 |
| FINAL | 一致性 | 4.9 | 22.3 | 22.9 | 13.9 | 15.1 | 21.7 |
| IsoRank | 一致性 | 4.2 | 18.9 | 21.6 | 11.5 | 14.2 | 22.1 |
| REGAL | 嵌入 | 0.3 | 1.8 | 1.0 | 0.8 | 2.8 | 45.3 |

OT 类方法（JOENA、PARROT）在所有 6 个领域上均取得最优 Hits@1，领先优势在知识图谱和基础设施网络上尤为显著。

### 消融实验：效率与可扩展性

| 方法 | 类别 | 社交时间(s) | 社交内存(GB) | 出版物时间(s) | 出版物内存(GB) |
|------|------|-----------|-------------|-------------|---------------|
| WAlign | 嵌入 | **0.61** | 2.65 | 9.41 | 9.88 |
| REGAL | 嵌入 | 9.38 | **1.16** | 16.14 | **3.18** |
| FINAL | 一致性 | 5.91 | 5.39 | 6.75 | 10.06 |
| PARROT | OT | — | — | — | — |
| JOENA | OT | — | — | — | — |
| IONE | 嵌入 | $6.34\times10^3$ | 1.94 | $1.43\times10^4$ | 4.16 |

效率上，WAlign 和 REGAL 速度最快、内存最小；IONE 的训练时间比其他方法高出数个数量级（社交网络上需 6000+ 秒），是可扩展性最差的方法。

### 关键发现

- **OT 方法全面领先**：JOENA 在全部 6 个领域的 Hits@1 上均排名第一，PARROT 稳居第二，验证了 OT 框架在 NA 任务上的优越性
- **嵌入方法差异巨大**：同为嵌入方法，REGAL 在社交网络上 Hits@1 仅 0.3%（几乎失效），而在通信网络上达到 45.3%，方法-数据的适配性极为重要
- **一致性方法不占优但稳定**：IsoRank 和 FINAL 在所有领域上表现一致性最好（方差小），虽然绝对性能不高但很少出现灾难性失败
- **效率-效果 trade-off 显著**：WAlign 速度最快但有效性中等，IONE 有效性中上但训练时间不可接受，OT 方法在效果和效率之间取得了较好平衡
- **PlanetAlign 实现质量高**：与官方实现对比，PlanetAlign 的实现在保持相似有效性的同时达到最高 3 倍加速

## 亮点与洞察

- **首次纳入 OT 方法的统一基准**：这是 NA 领域第一个包含最优传输方法的基准库，实验结果确认了 OT 方法的全面领先，这一发现对后续研究方向有重要引导价值
- **跨领域评估揭示方法偏见**：同一方法在不同领域上的表现可以相差 50 倍以上（如 REGAL），单领域基准容易给出误导性结论。这启示我们在开发新 NA 方法时必须在多领域上验证
- **API 设计思路可迁移**：PlanetAlign 的 `BaseData` + `BaseModel` + `Logger` 三层抽象设计是构建领域基准库的优秀范例，可以直接迁移到其他图学习任务（如链接预测、社区检测）的基准库建设中

## 局限与展望

- **方法覆盖仍有遗漏**：未包含基于 GNN 的端到端 NA 方法（如 DGMC）和基于 LLM 的新兴方法，随着领域发展需要持续更新
- **数据集规模有限**：最大数据集的节点数约数万级，缺乏百万级节点的大规模数据集来充分测试可扩展性
- **合成数据集方法单一**：所有合成数据集都使用相同的 10% 插入 + 15% 删除策略，可能无法覆盖真实场景中更复杂的网络差异模式
- **缺少无监督方法的公平评估**：部分方法是无监督的、部分是半监督的，当前评估框架虽然支持不同训练比例但未专门针对纯无监督场景优化

## 相关工作与启发

- **vs NAB** (Trung et al., 2020)：NAB 覆盖了有效性/可扩展性/鲁棒性三个维度但只有社交网络数据集且没有 OT 方法。PlanetAlign 在数据集领域覆盖和方法覆盖上全面超越
- **vs OpenEA** (Sun et al., 2020)：OpenEA 专注于知识图谱上的嵌入方法，是 KG 对齐的优秀基准。PlanetAlign 更广泛但在 KG 领域的深度可能不及 OpenEA
- **vs PyG/DGL 生态**：PlanetAlign 可以视为图学习生态系统中专门面向 NA 任务的垂直基准库，与通用图学习框架形成互补

## 评分

- 新颖性: ⭐⭐⭐ 作为基准库论文，核心贡献在于工程整合而非算法创新，但首次纳入 OT 方法有一定新意
- 实验充分度: ⭐⭐⭐⭐⭐ 14 种方法 × 18 个数据集 × 4 个评估维度，5 次重复取均值和标准差，实验规模堪称全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰、表格丰富、API 示例直观，但部分内容略显冗长
- 价值: ⭐⭐⭐⭐ 对 NA 研究社区有直接推动作用，系统性实验结论（OT 方法领先）具有重要参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] PopAlign: Diversifying Contrasting Patterns for a More Comprehensive Alignment](../../ACL2025/others/popalign_diversifying_contrasting_patterns_for_a_more_comprehensive_alignment.md)
- [\[ICLR 2026\] Fractional-Order Spiking Neural Network](fractional-order_spiking_neural_network.md)
- [\[ACL 2025\] GA-S3: Comprehensive Social Network Simulation with Group Agents](../../ACL2025/others/ga-s3_comprehensive_social_network_simulation_with_group_agents.md)
- [\[ICLR 2026\] Scaling Direct Feedback Learning with Jacobian Alignment Guarantees](scaling_direct_feedback_learning_with_jacobian_alignment_guarantees.md)
- [\[ICLR 2026\] Out of the Shadows: Exploring a Latent Space for Neural Network Verification](out_of_the_shadows_exploring_a_latent_space_for_neural_network_verification.md)

</div>

<!-- RELATED:END -->
