---
title: >-
  [论文解读] BeeRNA: Tertiary Structure-Based RNA Inverse Folding Using Artificial Bee Colony
description: >-
  [AAAI2026][计算生物][RNA inverse folding] 提出 BeeRNA，将人工蜂群（ABC）优化算法应用于 RNA 三级结构逆折叠问题，通过碱基对距离预筛选 + RMSD 两阶段适应度评估，在短/中长度 RNA（<100 nt）上超越深度学习方法 gRNAde 和 RiboDiffusion。
tags:
  - "AAAI2026"
  - "计算生物"
  - "RNA inverse folding"
  - "Artificial Bee Colony"
  - "tertiary structure"
  - "bio-inspired optimization"
  - "RhoFold"
---

# BeeRNA: Tertiary Structure-Based RNA Inverse Folding Using Artificial Bee Colony

**会议**: AAAI2026  
**arXiv**: [2511.21781](https://arxiv.org/abs/2511.21781)  
**代码**: 待公开  
**领域**: 计算生物  
**关键词**: RNA inverse folding, Artificial Bee Colony, tertiary structure, bio-inspired optimization, RhoFold  

## 一句话总结

提出 BeeRNA，将人工蜂群（ABC）优化算法应用于 RNA 三级结构逆折叠问题，通过碱基对距离预筛选 + RMSD 两阶段适应度评估，在短/中长度 RNA（<100 nt）上超越深度学习方法 gRNAde 和 RiboDiffusion。

## 背景与动机

- RNA 逆折叠问题（inverse folding）旨在设计能够折叠成特定目标结构的核苷酸序列，在合成生物学、aptamer 治疗、核糖开关等领域有重要应用
- 现有方法大多聚焦于**二级结构**逆折叠（如 ViennaRNA、NUPACK），三级结构逆折叠仍然是计算生物学中未充分解决的难题
- 深度学习方法（gRNAde、RiboDiffusion、RISoTTo）虽然推理速度快，但依赖大规模训练数据，且在**短 RNA（<50 nt）**上表现不佳——这恰恰是 miRNA、aptamer、核酶等功能性 RNA 所在的长度区间
- ABC 算法已在蛋白质逆折叠中展示了遍历复杂能量景观的能力，但尚未被应用于 RNA 三级结构逆折叠

## 核心问题

给定目标 RNA 三级结构 $T_{\text{3D}}$（PDB 文件中的 3D 原子坐标），寻找核苷酸序列 $S^* = \arg\min_S \text{RMSD}(F(S), T_{\text{3D}})$，其中 $F(S)$ 是 RhoFold 对序列 $S$ 的结构预测。附加约束包括热力学稳定性（最小自由能）和 GC 含量在 40%–60% 之间。

## 方法详解

### 整体框架

BeeRNA 将 ABC 蜂群优化与 RhoFold 结构预测结合，采用两阶段适应度评估策略：

1. **第一阶段（快速筛选）**：用 ViennaRNA 计算候选序列的二级结构，与目标二级结构比较碱基对距离（BPD）。BPD > 0 的序列直接标记为不合格（fitness = ∞）
2. **第二阶段（精确评估）**：仅对 BPD = 0 的序列调用 RhoFold 预测三级结构，计算与目标的 RMSD 作为适应度值

### ABC 算法三阶段

**初始化**：生成 40 条 RNA 序列（种群大小 N=40），从目标二级结构提取碱基对约束，配对位置分配互补碱基（G-C 或 C-G），未配对位置随机分配核苷酸，保持 GC 含量 40%–60%

**雇佣蜂阶段**：每条序列通过自适应变异率生成邻域解。变异率公式为：

$$\text{mutation\_rate} = \max\left(0.1,\ 0.095 \cdot e^{-\frac{\text{best\_RMSD}}{5n}}\right)$$

变异操作包括：随机位置核苷酸替换、相邻位置（3 个位置内）交换变异（20% 概率）、GC 含量超 50% 时的 {A,U}/{G,C} 互换。若邻域解 RMSD 更低则替换原序列，否则试验计数器递增

**旁观蜂阶段**：基于 softmax 选择概率 $p_i = e^{-r_i/\tau} / \sum_j e^{-r_j/\tau}$ 按概率选择序列进一步探索，温度参数 $\tau = 5.0 \cdot (1 + t/T)$ 随迭代增大，实现早期探索、后期利用

**侦察蜂阶段**：连续 5 次未改进的序列被随机重新初始化，防止陷入局部最优

### 评估指标

- **RMSD**：主要指标，通过 US-align 进行最优叠合后计算骨架磷原子（P）、糖碳原子（C4'）和碱基氮原子（N1/N9）的偏差
- **GDT-TS**：辅助指标，衡量预测结构中在 1/2/4/8 Å 距离阈值内的残基比例

## 实验关键数据

### RNASolo 数据集（短 RNA，3–30 nt）

| 指标 | BeeRNA | gRNAde |
|------|--------|--------|
| RMSD (Å) | **2.50** | 9.33 |
| GDT-TS (%) | **26.91** | 18.97 |

### RFAM 数据集（25–200 nt ncRNA）

| 指标 | BeeRNA | gRNAde |
|------|--------|--------|
| RMSD (Å) | **14.98** | 16.24 |
| GDT-TS (%) | **11.56** | 9.77 |

### 14 个基准 RNA 结构

| 指标 | BeeRNA | gRNAde | RiboDiffusion |
|------|--------|--------|---------------|
| RMSD (Å) | 12.02 | 14.63 | **10.31** |
| GDT-TS (%) | 15.92 | 10.16 | **22.69** |

- BeeRNA 在 14 个结构中的 **10 个**取得最低 RMSD，尤其在短 RNA 上优势显著（如 1F27: 2.21 Å vs gRNAde 14.94 Å）
- RiboDiffusion 在长 RNA（>100 nt）上表现更好，但其训练数据可能与测试集重叠
- 运行效率：<50 nt 约 3 分钟，50–100 nt 约 7–10 分钟（64 核 CPU）

### 关键发现

论文展示了一个有力的案例：对 RNA 2OUE（61 nt），仅做单核苷酸突变（序列恢复率 98.4%），RMSD 就飙升至 19.34 Å，说明**高序列相似度不能保证结构正确性**，凸显了基于结构的评估方式比序列恢复率更合理。

## 亮点

- **无需训练**：不依赖大规模数据集预训练，即插即用，对新出现的 RNA 家族同样适用
- **两阶段筛选设计巧妙**：先用轻量级 BPD 过滤大量不合格序列，避免昂贵的 RhoFold 调用，显著降低计算成本
- **自适应变异机制**：结合模拟退火思想的自适应变异率，兼顾探索与利用
- **结构导向评估**：论文有力论证了以 RMSD/GDT-TS 代替序列恢复率作为评估指标的合理性
- **生物约束整合**：GC 含量、Watson-Crick 配对等约束直接嵌入优化流程

## 局限与展望

- **长 RNA 可扩展性差**：>100 nt 时搜索空间指数增长，RMSD 明显升高（如 2R8S 159 nt 达 26 Å）
- **依赖 RhoFold 预测精度**：RhoFold 本身的误差会传递到 BeeRNA 的优化结果
- **BPD=0 前提严格**：当目标结构含 wobble 或非典型碱基对时，BPD 无法归零，需用固定 20 Å 罚分替代，影响收敛
- **串行 CPU 推理**：每轮迭代需多次调用 RhoFold，GPU 加速可大幅提速
- **种群/迭代参数固定**：40×40 的设置可能对不同长度的 RNA 不够灵活

## 与相关工作的对比

| 方法 | 类型 | 训练需求 | 短 RNA 优势 | 长 RNA 优势 | 结构评估 |
|------|------|----------|------------|------------|----------|
| ViennaRNA | 确定性 | 无 | 仅二级结构 | 仅二级结构 | 无 |
| gRNAde | 深度学习 GNN | 大规模预训练 | 弱 | 较强 | RMSD |
| RiboDiffusion | 扩散模型 | 大规模预训练 | 弱（<50 nt 困难） | 强 | RMSD |
| RISoTTo | 几何 Transformer | 大规模预训练 | 未详测 | 强 | 序列恢复率 |
| **BeeRNA** | 仿生元启发 | **无** | **强** | 弱 | RMSD + GDT-TS |

BeeRNA 填补了"无需训练 + 三级结构逆折叠"这一空白，与深度学习方法形成互补：短 RNA 用 BeeRNA，长 RNA 用 gRNAde/RiboDiffusion。

## 启发与关联

- ABC 算法成功从蛋白质逆折叠迁移到 RNA 逆折叠，暗示其他仿生算法（蚁群、粒子群）也可尝试
- 两阶段筛选（廉价预筛 + 昂贵精评）是一种通用的搜索加速范式，可迁移到其他结构设计问题
- 未来可探索将 BeeRNA 作为深度学习方法的后处理优化器，或用深度学习方法初始化 BeeRNA 种群
- 随着 AlphaFold3 等更精确的 RNA 结构预测工具出现，BeeRNA 可无缝替换 RhoFold 获得更好结果

## 评分

- 新颖性: ⭐⭐⭐（ABC→RNA 三级结构的首次应用，但方法本身较传统）
- 实验充分度: ⭐⭐⭐⭐（三个数据集、多指标对比，但缺少消融实验）
- 写作质量: ⭐⭐⭐⭐（清晰完整，动机论证有力）
- 价值: ⭐⭐⭐（短 RNA 领域有实用价值，但长 RNA 局限明显）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] A Joint Diffusion Model with Pre-Trained Priors for RNA Sequence-Structure Co-Design](../../ICLR2026/computational_biology/a_joint_diffusion_model_with_pre-trained_priors_for_rna_sequence-structure_co-de.md)
- [\[CVPR 2026\] TRIDENT: A Trimodal Cascade Generative Framework for Drug and RNA-Conditioned Cellular Morphology Synthesis](../../CVPR2026/computational_biology/trident_a_trimodal_cascade_generative_framework_for_drug_and_rna-conditioned_cel.md)
- [\[CVPR 2026\] Bulk RNA-seq Guided Multi-modal Detection of Anomalous Regions in Human Cancer via Spatial Transcriptomics](../../CVPR2026/computational_biology/bulk_rna-seq_guided_multi-modal_detection_of_anomalous_regions_in_human_cancer_v.md)
- [\[AAAI 2026\] S2Drug: Bridging Protein Sequence and 3D Structure in Contrastive Representation Learning for Virtual Screening](s2drug_bridging_protein_sequence_and_3d_structure_in_contrastive_representation_.md)
- [\[ICML 2026\] Protein Autoregressive Modeling via Multiscale Structure Generation](../../ICML2026/computational_biology/protein_autoregressive_modeling_via_multiscale_structure_generation.md)

</div>

<!-- RELATED:END -->
