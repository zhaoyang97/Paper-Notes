---
title: >-
  [论文解读] All-atom Diffusion Transformers: Unified Generative Modelling of Molecules and Materials
description: >-
  [ICML2025][图像生成][扩散模型] 提出 All-atom Diffusion Transformer (ADiT)，通过 VAE 将分子和晶体映射到统一潜空间、再用 Diffusion Transformer 在潜空间生成的两阶段框架，首次实现**单一模型**同时生成周期性材料（晶体）和非周期性分子系统，在 MP20、QM9、GEOM-DRUGS 上达到 SOTA，且比等变扩散模型快一个数量级。
tags:
  - "ICML2025"
  - "图像生成"
  - "扩散模型"
  - "Transformer"
  - "molecular generation"
  - "crystal generation"
  - "unified model"
  - "scaling law"
---

# All-atom Diffusion Transformers: Unified Generative Modelling of Molecules and Materials

**会议**: ICML2025  
**arXiv**: [2503.03965](https://arxiv.org/abs/2503.03965)  
**作者**: Chaitanya K. Joshi, Xiang Fu, Yi-Lun Liao, Vahe Gharakhanyan, Benjamin Kurt Miller, Anuroop Sriram, Zachary W. Ulissi (Meta FAIR, Cambridge, MIT)
**代码**: [facebookresearch/all-atom-diffusion-transformer](https://github.com/facebookresearch/all-atom-diffusion-transformer)  
**领域**: 图像生成  
**关键词**: latent diffusion, diffusion transformer, molecular generation, crystal generation, unified model, scaling law

## 一句话总结

提出 All-atom Diffusion Transformer (ADiT)，通过 VAE 将分子和晶体映射到统一潜空间、再用 Diffusion Transformer 在潜空间生成的两阶段框架，首次实现**单一模型**同时生成周期性材料（晶体）和非周期性分子系统，在 MP20、QM9、GEOM-DRUGS 上达到 SOTA，且比等变扩散模型快一个数量级。

## 研究背景与动机

3D 原子系统的生成建模是分子设计和材料发现的核心问题。当前扩散模型在不同类型原子系统上高度特化：

- **小分子生成**（如 EDM）：对原子类型（离散）和 3D 坐标（连续）分别建立扩散过程，中间状态不真实
- **生物大分子**（如 FrameDiff）：将原子组视为刚体，引入额外的旋转流形
- **晶体/材料**（如 DiffCSP, FlowMM）：需处理周期性，在原子类型、分数坐标、晶格参数的联合流形上扩散

这些方法底层物理相同，但模型设计完全不同，无法共享知识。**核心问题**：能否构建一个统一的扩散模型，同时生成周期性材料和非周期性分子？

现有方法的关键瓶颈：

**等变网络效率低**：EGNN、GVP 等等变网络作为去噪器，计算开销大，难以扩展

**多流形扩散复杂**：类别、坐标、晶格角度等异构数据需要在乘积流形上分别定义扩散过程

**领域隔离**：分子模型和材料模型各自训练，无法实现跨领域迁移学习

## 方法详解

### 整体框架

ADiT 采用 **Latent Diffusion** 两阶段架构：

1. **Stage 1 — VAE 自编码器**：学习将分子和晶体的 all-atom 表示映射到共享潜空间
2. **Stage 2 — Diffusion Transformer**：在潜空间中训练 DiT 生成新潜向量，再由 VAE 解码为有效的分子或晶体

核心思想是用自编码器吸收异构数据类型（类别 + 连续）的复杂性，使潜空间中的生成过程统一且可扩展。

### 关键设计 1: 统一的 All-atom 表示

所有原子系统（分子和晶体）被统一表示为 3D 空间中的原子集合：

- **原子类型** $\mathbf{A} = \{a_i\}_{i=1}^N \in \mathbb{Z}^{1 \times N}$：离散类别属性
- **3D 坐标** $\mathbf{X} = \{x_i\}_{i=1}^N \in \mathbb{R}^{3 \times N}$：连续空间属性
- **晶格矩阵** $\mathbf{L} \in \mathbb{R}^{3 \times 3}$（仅晶体）：定义周期性单元格

对于分子，$\mathbf{L}$ 设为零矩阵或特殊标记，从而在同一表示框架下处理周期性和非周期性系统。

### 关键设计 2: VAE 自编码器

VAE 编码器和解码器均使用**标准 Transformer**（非等变网络），核心设计包括：

- **编码器**：将 all-atom 表示（原子类型 one-hot + 坐标 + 晶格信息）编码为一组固定维度的潜向量 $\mathbf{Z} \in \mathbb{R}^{M \times d}$
- **解码器**：从潜向量 $\mathbf{Z}$ 重建原子类型（分类损失）、3D 坐标（回归损失）和晶格参数
- **数据增强**：对分子施加随机旋转和平移，使模型隐式学习 SE(3) 不变性，避免显式等变约束带来的计算开销
- **训练目标**：重建损失 + KL 散度正则化，联合训练分子和晶体数据

放弃等变网络是 ADiT 的关键决策——通过数据增强代替架构约束，大幅简化模型并提升训练/推理速度。

### 关键设计 3: Diffusion Transformer (DiT)

在 VAE 潜空间中训练标准的 Diffusion Transformer：

- **前向过程**：对潜向量 $\mathbf{Z}$ 添加高斯噪声
- **去噪网络**：标准 Transformer + AdaLN（自适应层归一化）条件化时间步
- **Classifier-free guidance (CFG)**：使用系统类型标签（分子 vs 晶体）作为条件，训练时随机丢弃标签实现无分类器引导
- **推理**：抽样潜向量 -> VAE 解码 -> 后处理得到有效分子/晶体

DiT 在潜空间操作，避免了直接在原子类型和坐标的乘积流形上扩散的复杂性。

### 训练策略

- **联合训练**：将 QM9 分子数据和 MP20 材料数据混合训练同一模型
- **两阶段训练**：先训练 VAE 至收敛，再冻结 VAE 训练 DiT
- **数据增强**：随机 SE(3) 变换替代等变架构约束
- **模型缩放**：从 30M 参数扩展至 500M 参数，观察到可预测的 scaling law

## 实验关键数据

### 数据集

| 数据集 | 类型 | 样本数 | 原子数范围 | 特点 |
|---|---|---|---|---|
| QM9 | 小分子 | ~130K | ≤9 重原子 | 标准分子生成基准 |
| MP20 | 晶体材料 | ~45K | ≤20 原子/单元格 | Materials Project 子集 |
| GEOM-DRUGS | 药物分子 | ~300K 构象 | 数百原子 | 大分子 3D 构象生成 |

### Table 1: MP20 晶体生成结果

| 方法 | 类型 | Match Rate (%) | RMSD | S.U.N. Rate (%) |
|---|---|---|---|---|
| CDVAE | 等变扩散 | 45.4 | 0.356 | 3.5 |
| DiffCSP | 等变扩散 | 51.1 | 0.252 | 4.0 |
| FlowMM | 等变流匹配 | 65.3 | 0.195 | 4.8 |
| **ADiT (MP20-only)** | 潜扩散 | 62.8 | 0.201 | 5.2 |
| **ADiT (Joint)** | 潜扩散 | **67.1** | **0.188** | **6.0** |

- ADiT 联合训练的 S.U.N.（Stable, Unique, Novel）率达 **6.0%**，比最佳基线 FlowMM 提升 **25%**
- 联合训练优于单数据集训练，证明分子-材料迁移学习的有效性

### Table 2: QM9 分子生成结果

| 方法 | 类型 | Atom Stability (%) | Mol Stability (%) | Validity (%) | Uniqueness (%) |
|---|---|---|---|---|---|
| EDM | 等变扩散 | 98.7 | 82.0 | 91.9 | 90.7 |
| GeoLDM | 潜扩散 | 98.9 | 89.4 | 93.8 | 92.7 |
| EQGAT-diff | 等变扩散 | 98.2 | 71.6 | 86.7 | 96.1 |
| **ADiT (QM9-only)** | 潜扩散 | 99.0 | 90.1 | 95.2 | 91.8 |
| **ADiT (Joint)** | 潜扩散 | **99.2** | **91.3** | **96.1** | 92.4 |

- ADiT 在原子稳定性和分子稳定性上均超越专门化模型
- 联合训练再次优于单一训练，Validity 从 95.2% 提升至 **96.1%**

### 效率对比

| 方法 | 10K 样本推理时间 | 加速比 |
|---|---|---|
| 等变扩散基线 | ~2.5 小时 (V100) | 1x |
| **ADiT** | **< 20 分钟 (V100)** | **~7-8x** |

ADiT 推理速度比等变扩散模型快约 **7-8x**，这得益于标准 Transformer 避免了等变计算的开销。

### Scaling Law

ADiT 从 ~30M 扩展至 ~500M 参数，在数据规模不变的情况下：
- 生成质量（各项指标）随模型参数量可预测地提升
- 表明继续扩展仍可获得收益，指向分子/材料生成的基础模型方向

## 亮点

- **首个统一模型**：第一个能同时生成周期性晶体和非周期性分子的扩散模型，打破了领域隔离
- **迁移学习有效**：联合训练在分子和材料两个领域均优于单独训练，证明跨系统知识迁移是可行的
- **从等变到数据增强的范式转换**：用标准 Transformer + 数据增强替代等变网络，大幅提升效率且不损失性能，这一设计哲学对科学机器学习社区有重要启示
- **可扩展性**：DiT 架构天然支持参数扩展，观察到清晰的 scaling law，为构建化学生成基础模型奠定基础
- **实用效率**：单 V100 上 20 分钟生成 10K 样本，比等变基线快一个数量级，具备实际应用价值

## 局限性

- **VAE 信息瓶颈**：两阶段方法中 VAE 的重建质量决定了生成上限，潜空间压缩可能丢失细粒度结构信息
- **数据增强 vs 等变性**：数据增强隐式学习的对称性不如显式等变约束严格，在数据稀缺场景可能不够
- **仅验证小规模系统**：QM9 (≤9 重原子) 和 MP20 (≤20 原子) 规模有限，对蛋白质等大规模系统的泛化性未验证
- **DFT 评估成本高**：S.U.N. 指标需要 DFT 计算验证稳定性，大规模评估成本高昂
- **条件生成能力未探索**：仅展示无条件生成，面向实际应用的属性条件生成（如指定带隙、溶解度）尚未验证

## 相关工作

- **分子扩散模型**：EDM (Hoogeboom et al., 2022) 在原子类型+坐标联合流形上扩散；GeoLDM (Xu et al., 2023) 引入分子潜扩散；EQGAT-diff (Le et al., 2024) 使用等变图注意力。ADiT 用统一潜空间避免了复杂的多流形扩散
- **晶体生成**：CDVAE (Xie et al., 2022) 开创了晶体 VAE+扩散；DiffCSP (Jiao et al., 2023) 和 FlowMM (Miller et al., 2024) 改进了晶体流形上的生成。ADiT 首次实现分子-晶体统一生成
- **Diffusion Transformer**：DiT (Peebles & Xie, 2023) 在图像生成中用 Transformer 替代 U-Net；Latent Diffusion (Rombach et al., 2022) 引入两阶段范式。ADiT 将这些思想成功迁移到 3D 原子系统
- **科学机器学习中的 Transformer**：近期工作（如 Liao & Smidt, 2023; Duval et al., 2023）开始探索标准 Transformer 在原子系统建模中的潜力，ADiT 在生成任务上验证了这一方向的可行性

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次统一分子和晶体生成，潜扩散思路虽借鉴图像领域但迁移到原子系统有方法论贡献
- 实验充分度: ⭐⭐⭐⭐ — 三个数据集 + DFT 验证 + scaling law 分析 + 效率对比，较为全面
- 写作质量: ⭐⭐⭐⭐⭐ — 动机清晰、方法描述严谨、图表质量高，Meta FAIR 出品
- 价值: ⭐⭐⭐⭐ — 为化学生成基础模型指明方向，统一框架+扩展性分析具有长期价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Position: All Current Generative Fidelity and Diversity Metrics are Flawed](position_all_current_generative_fidelity_and_diversity_metrics_are_flawed.md)
- [\[NeurIPS 2025\] Scaling Diffusion Transformers Efficiently via μP](../../NeurIPS2025/image_generation/scaling_diffusion_transformers_efficiently_via_μp.md)
- [\[CVPR 2025\] GenDeg: Diffusion-based Degradation Synthesis for Generalizable All-In-One Image Restoration](../../CVPR2025/image_generation/gendeg_diffusion-based_degradation_synthesis_for_generalizable_all-in-one_image_.md)
- [\[CVPR 2025\] TinyFusion: Diffusion Transformers Learned Shallow](../../CVPR2025/image_generation/tinyfusion_diffusion_transformers_learned_shallow.md)
- [\[ICML 2025\] Normalizing Flows are Capable Generative Models](normalizing_flows_are_capable_generative_models.md)

</div>

<!-- RELATED:END -->
