---
title: >-
  [论文解读] Structure-Aware Fusion with Progressive Injection for Multimodal Molecular Representation Learning
description: >-
  [NeurIPS 2025][多模态VLM][分子表示学习] 提出 MuMo 框架，通过结构化融合管线（SFP）将 2D 拓扑和 3D 几何统一为稳定的结构先验，并通过渐进注入（PI）机制非对称地将该先验整合到序列流中，在 29 个分子性质预测任务中平均超过最佳基线 2.7%，在 22 个任务上排名第一。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "分子表示学习"
  - "多模态融合"
  - "状态空间模型"
  - "3D构象"
  - "渐进注入"
---

# Structure-Aware Fusion with Progressive Injection for Multimodal Molecular Representation Learning

**会议**: NeurIPS 2025  
**arXiv**: [2510.23640](https://arxiv.org/abs/2510.23640)  
**代码**: [有](https://github.com/selmiss/MuMo)  
**领域**: 多模态学习 / 分子表示  
**关键词**: 分子表示学习, 多模态融合, 状态空间模型, 3D构象, 渐进注入

## 一句话总结

提出 MuMo 框架，通过结构化融合管线（SFP）将 2D 拓扑和 3D 几何统一为稳定的结构先验，并通过渐进注入（PI）机制非对称地将该先验整合到序列流中，在 29 个分子性质预测任务中平均超过最佳基线 2.7%，在 22 个任务上排名第一。

## 研究背景与动机

分子表示学习是药物发现和材料科学的基础任务。分子数据自然具有多种表示形态：
- **1D**: SMILES 序列（字符串表示）
- **2D**: 分子图（原子-键拓扑）
- **3D**: 空间构象（原子 3D 坐标）

多模态分子模型旨在融合这些信息以获得更丰富的表示。然而，现有方法面临两个核心问题：

1. **3D 构象不可靠**: 分子的 3D 构象通常由 RDKit 等工具生成（非实验测定），存在噪声和不确定性。直接依赖 3D 构象进行融合会引入不稳定性。

2. **模态坍塌（Modality Collapse）**: 朴素的融合策略（如简单拼接、平均）容易导致一种模态主导，另一种模态的信息被忽略。这在分子领域尤为严重，因为不同模态的信息密度和可靠性差异很大。

## 方法详解

### 整体框架

MuMo 的设计分为三个核心模块：

1. **序列编码器**: 使用状态空间模型（SSM，如 Mamba）处理 SMILES 序列，捕获长程依赖
2. **结构化融合管线（SFP）**: 融合 2D 拓扑和 3D 几何为统一的结构先验
3. **渐进注入（PI）**: 将结构先验逐层注入序列编码器

### 关键设计

**结构化融合管线（Structured Fusion Pipeline, SFP）**: 

SFP 的目标是将 2D 和 3D 信息融合为一个稳定的结构先验，而不是直接使用不可靠的 3D 构象：

- **2D 编码**: 使用 GNN（图神经网络）提取分子图特征，包括原子特征和键特征
- **3D 编码**: 使用几何感知的网络（如 SchNet/GemNet 风格）提取 3D 空间特征
- **融合**: 通过注意力机制将 2D 拓扑信息作为"锚点"，3D 几何信息作为"增强"，生成稳定的结构先验 $\mathbf{S}$

关键思想：2D 拓扑是确定性的（由分子式决定），因此将其作为融合的基础可以降低 3D 噪声的影响。

**渐进注入（Progressive Injection, PI）**: 

为避免模态坍塌，PI 采用非对称融合策略：

- **主流（Main Stream）**: 序列模型（Mamba）处理 SMILES，保持其独立的表示能力
- **注入方式**: 在序列模型的每一层，通过交叉注意力或门控机制将结构先验 $\mathbf{S}$ 注入
- **渐进性**: 浅层注入少量结构信息，深层注入更多，使模型逐步整合多模态信息
- **非对称性**: 结构先验增强序列表示，但序列信息不回传到结构编码器，避免相互干扰

$$\mathbf{h}_l^{\text{out}} = \text{SSM}_l(\mathbf{h}_l^{\text{in}}) + \lambda_l \cdot \text{CrossAttn}(\mathbf{h}_l^{\text{in}}, \mathbf{S})$$

其中 $\lambda_l$ 随层数增加而增大，实现渐进注入。

### 损失函数 / 训练策略

根据下游任务选择损失函数：
- **分类任务**: 交叉熵损失
- **回归任务**: MSE / MAE 损失
- 端到端训练，无需预训练阶段

## 实验关键数据

### 主实验

在 Therapeutics Data Commons (TDC) 和 MoleculeNet 的 29 个基准任务上评估。

**TDC 任务（ADMET 性质预测）**:

| 方法 | Caco2 ↑ | HIA ↑ | BBB ↑ | LD50 ↑ | CYP2D6 ↑ | 平均排名 |
|------|---------|-------|-------|--------|----------|---------|
| Uni-Mol | 0.672 | 0.823 | 0.891 | 0.615 | 0.852 | 3.2 |
| 3D-MoLM | 0.681 | 0.831 | 0.885 | 0.623 | 0.845 | 3.8 |
| MoleculeSTM | 0.665 | 0.818 | 0.878 | 0.605 | 0.839 | 4.5 |
| GEM | 0.658 | 0.812 | 0.872 | 0.598 | 0.832 | 5.1 |
| **MuMo** | **0.695** | **0.845** | **0.903** | **0.782** | **0.868** | **1.4** |

MuMo 在 LD50 任务上取得了 **27% 的显著提升**（0.615 → 0.782），并在 22/29 个任务中排名第一。

**MoleculeNet 任务（分类/回归）**:

| 方法 | BBBP (AUC) | BACE (AUC) | Tox21 (AUC) | ESOL (RMSE↓) | FreeSolv (RMSE↓) |
|------|-----------|-----------|------------|-------------|-----------------|
| GROVER | 0.940 | 0.826 | 0.743 | 0.831 | 2.176 |
| MolCLR | 0.932 | 0.819 | 0.738 | 0.845 | 2.238 |
| Uni-Mol | 0.945 | 0.835 | 0.751 | 0.788 | 1.923 |
| **MuMo** | **0.958** | **0.852** | **0.769** | **0.712** | **1.685** |

### 消融实验

**组件消融（在 TDC 基准上的平均性能）**:

| 配置 | 平均 AUC/R² | vs. Full |
|------|------------|---------|
| MuMo Full | **0.812** | — |
| 去除 PI (直接拼接) | 0.785 | -2.7% |
| 去除 SFP (仅 3D) | 0.778 | -3.4% |
| 去除 SFP (仅 2D) | 0.791 | -2.1% |
| 去除渐进性 (均匀注入) | 0.798 | -1.4% |
| 使用 Transformer 替代 SSM | 0.803 | -0.9% |

- SFP 和 PI 都是关键组件，去除任一都导致显著性能下降
- 仅使用 3D 信息（不稳定的构象）比仅使用 2D 差，验证了 3D 构象不可靠的问题
- 渐进注入优于均匀注入，说明浅层需要保持序列模型的独立性

### 关键发现

1. **3D 构象噪声的影响**: 直接使用 3D 构象不如 2D+3D 融合，SFP 有效缓解了这一问题
2. **模态坍塌的解决**: PI 的非对称设计避免了序列模态被结构模态淹没
3. **SSM 骨干的优势**: 状态空间模型在长 SMILES 序列上优于 Transformer
4. **LD50 的显著提升**: 27% 的改进表明 MuMo 在毒性预测等高价值任务上有特殊优势

## 亮点与洞察

- **问题意识精准**: 准确识别了多模态分子学习中的两个核心痛点（3D 不可靠 + 模态坍塌）
- **设计思路清晰**: SFP 解决第一个问题，PI 解决第二个问题，各自有明确的目标
- **Mamba 骨干的合理选择**: SMILES 序列可以很长，SSM 的线性复杂度相比 Transformer 更合适
- **全面的实验**: 29 个基准任务，覆盖 ADMET 和 MoleculeNet 两大标准集

## 局限与展望

1. **构象生成方法的影响**: 不同的 3D 构象生成工具（RDKit vs. ETKDG vs. 力场优化）可能影响结果，但论文未充分分析
2. **大分子适用性**: SMILES 对于蛋白质等大分子表示能力有限
3. **预训练**: 未利用大规模无标注分子数据进行预训练，可能限制了泛化能力
4. **可解释性**: 融合后的表示缺乏化学层面的可解释性
5. **多构象采样**: 仅使用单一构象，而分子在实际中存在构象集合

## 相关工作与启发

- **Uni-Mol**: He et al. (2023) — 基于 3D 的统一分子表示学习
- **3D-MoLM**: Li et al. (2024) — 3D 分子语言模型
- **MoleculeSTM**: Liu et al. (2023) — SMILES + 文本的多模态分子模型
- **Mamba**: Gu & Dao (2024) — 选择性状态空间模型
- **GEM**: Fang et al. (2022) — 几何增强的分子表示

## 评分

- **创新性**: 4/5 — SFP + PI 的组合设计针对性强
- **技术质量**: 4/5 — 29 个基准任务的全面验证
- **表达质量**: 4/5 — 论文结构清晰，动机阐述充分
- **实用性**: 4/5 — 开源代码，直接可用于药物发现
- **综合评分**: 4/5

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] STRUCTURE: With Limited Data for Multimodal Alignment, Let the Structure Guide You](with_limited_data_for_multimodal_alignment_let_the_structure_guide_you.md)
- [\[NeurIPS 2025\] On the Value of Cross-Modal Misalignment in Multimodal Representation Learning](on_the_value_of_cross-modal_misalignment_in_multimodal_representation_learning.md)
- [\[NeurIPS 2025\] Aligning by Misaligning: Boundary-aware Curriculum Learning for Multimodal Alignment](aligning_by_misaligning_boundaryaware_curriculum_learning_fo.md)
- [\[NeurIPS 2025\] Multimodal Negative Learning](multimodal_negative_learning.md)
- [\[ACL 2025\] MIRe: Enhancing Multimodal Queries Representation via Fusion-Free Modality Interaction](../../ACL2025/multimodal_vlm/mire_enhancing_multimodal_queries_representation_via_fusion-free_modality_intera.md)

</div>

<!-- RELATED:END -->
