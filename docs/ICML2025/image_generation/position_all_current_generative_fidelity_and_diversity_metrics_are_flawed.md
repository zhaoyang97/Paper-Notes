---
title: >-
  [论文解读] Position: All Current Generative Fidelity and Diversity Metrics are Flawed
description: >-
  [ICML2025][图像生成][生成模型评估] Position paper：系统性地证明了所有现有生成模型 fidelity 和 diversity 指标（包括 Improved Precision/Recall、Density/Coverage、α-precision/β-recall 等六对指标）在精心设计的 sanity check 中均存在大量失败，呼吁社区投入更多精力研发更可靠的评估指标。
tags:
  - "ICML2025"
  - "图像生成"
  - "生成模型评估"
  - "fidelity/diversity 指标"
  - "precision/recall"
  - "合成数据质量"
  - "sanity check"
---

# Position: All Current Generative Fidelity and Diversity Metrics are Flawed

**会议**: ICML2025  
**arXiv**: [2505.22450](https://arxiv.org/abs/2505.22450)  
**代码**: [vanderschaarlab/position-fidelity-diversity-metrics-flawed](https://github.com/vanderschaarlab/position-fidelity-diversity-metrics-flawed)  
**领域**: 图像生成  
**关键词**: 生成模型评估, fidelity/diversity 指标, precision/recall, 合成数据质量, sanity check

## 一句话总结

Position paper：系统性地证明了所有现有生成模型 fidelity 和 diversity 指标（包括 Improved Precision/Recall、Density/Coverage、α-precision/β-recall 等六对指标）在精心设计的 sanity check 中均存在大量失败，呼吁社区投入更多精力研发更可靠的评估指标。

## 研究背景与动机

生成模型（GAN、扩散模型、LLM 生成表格数据等）的快速发展依赖于可靠的评估指标。传统指标如 FID 只能给出整体质量分数，无法区分生成质量的不同维度。为此，社区提出了 precision/recall 类指标，将评估拆分为两个维度：

- **Fidelity（保真度）**：生成样本是否逼真（synthetic 样本是否落在真实分布中）
- **Diversity（多样性）**：生成分布是否覆盖了真实分布的全部模式

目前主流的 fidelity/diversity 指标对包括：

| 论文 | Fidelity 指标 | Diversity 指标 |
|------|-------------|---------------|
| Kynkäänniemi et al., 2019 | Improved Precision (I-Prec) | Improved Recall (I-Rec) |
| Naeem et al., 2020 | Density | Coverage |
| Alaa et al., 2022 | Integrated α-precision (IAP) | Integrated β-recall (IBR) |
| Cheema & Urner, 2023 | Precision Cover (C-Prec) | Recall Cover (C-Rec) |
| Khayatkhoei & Abdalmageed, 2023 | Symmetric Precision (symPrec) | Symmetric Recall (symRec) |
| Park & Kim, 2023 | Probabilistic Precision (P-Prec) | Probabilistic Recall (P-Rec) |

已有部分工作发现了这些指标的个别失败案例（如缺乏 outlier 鲁棒性、上下界不清等），但每项工作只关注少数问题并修补，缺乏全面系统的评估。**核心研究问题**：当现有指标被汇集在一起，用一套统一的标准全面检验时，是否还有指标能通过所有测试？

## 核心思想

本文提出三大贡献：

1. **六项 Desiderata（理想标准）**：定义合成数据评估指标应满足的六条准则
2. **14 项 Sanity Check（健全性测试）**：将文献中报告的失败案例提炼为自动化的简单测试
3. **系统评估**：对 6 对指标（12 个指标）在所有 sanity check 上进行评估

核心立场（Position）：**所有现有的 fidelity 和 diversity 指标都有缺陷**，很多指标甚至无法可靠地度量它们本应度量的最基本属性。

## 方法细节

### 六项 Desiderata

| 编号 | 名称 | 要求 |
|------|------|------|
| D1 | Purpose（目标性） | 度量有直接实用价值的量（D1a）、给出可解释的分布差异信息（D1b）、或作为可靠代理指标（D1c） |
| D2 | Hyperparameters（超参少） | 超参数量尽量少，且影响清晰可控 |
| D3 | Data（数据需求低） | 所需真实数据量＜实际可用数据量，阈值设为 1000 |
| D4 | Bounds（上下界明确） | 有明确的上下界，可做绝对评价而非仅相对比较 |
| D5 | Invariance（不变性） | 对不影响数据质量的变换（缩放、分类变量排列等）保持不变 |
| D6 | Computation（计算效率） | 可在合理时间内计算完成 |

### Embedding 策略

所有指标均先将数据嵌入到更适合度量几何关系的空间：

- **图像数据**：使用预训练神经网络（如 InceptionV3）
- **表格数据**：对分类变量做 one-hot 编码，对数值变量标准化为零均值单位方差。这种简单嵌入满足 D5（缩放不变性、类别排列不变性）且无需额外超参

### 14 项 Sanity Check 设计

每项测试使用人工构造的真实/合成分布，聚焦单一潜在问题，设定明确的通过/失败准则：

**高斯类测试（5 项）**：

- **Gaussian Mean Difference**：两个高斯分布仅均值不同，测试指标能否检测到分布偏移
- **Gaussian Mean Diff + Outlier**：加入异常点，测试 outlier 鲁棒性
- **Gaussian Std Deviation Difference**：仅标准差不同，测试对分布宽度差异的敏感度
- **One Disjoint Dim + Many Identical Dim**：仅一个维度有差异，其余维度相同，测试高维下的检测能力
- **Scaling One Dimension**：对一个维度做缩放变换，测试 D5 不变性

**混合高斯类测试（3 项）**：

- **Mode Collapse**：两个 mode 的真实分布 vs 一个宽 mode 的合成分布
- **Mode Dropping + Invention**：合成分布逐渐增加 mode 数量，先覆盖真实 mode 再发明新 mode
- **Sequential / Simultaneous Mode Dropping**：10 个 mode 中逐一丢弃或同时降低权重

**超立方体/超球面测试（3 项）**：

- **Hypercube, Varying Sample Size**：固定分布，变化样本量，测试 D3
- **Hypercube, Varying Syn. Size**：固定真实样本数，变化合成样本数，测试 D2
- **Hypersphere Surface**：不同半径的超球面上均匀分布，测试高维环境下的正确性

**几何/表格测试（3 项）**：

- **Sphere vs. Torus**：球面 vs 环面的不相交分布
- **Discrete Num. vs. Continuous Num.**：高斯分布 vs 取整后的离散分布（表格数据常见场景）
- **Gaussian Mean Diff + Pareto**：加入重尾 Pareto 分布的额外维度（表格数据常见）

### 通过/失败准则

每项测试关联一个或多个 desiderata：
- **D1b**：指标行为整体正确（趋势方向对）
- **D4**：在极端情况下指标接近理论上下界（0 或 1）
- **D3**：样本量 > 1000 后指标稳定收敛
- **D5**：对缩放变换保持不变

对 diversity 指标引入 **High/Low 区分**：当合成分布完全覆盖但远宽于真实分布时，diversity 是高还是低取决于"覆盖"的定义，允许指标一致性地选择任一解读。

## 实验设置与主要结果

### Fidelity 指标结果（Table 3 精选）

| Sanity Check | I-Prec | Density | IAP | C-Prec | symPrec | P-Prec |
|---|---|---|---|---|---|---|
| Gaussian Mean Diff (D1b) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| + Outlier (D1b) | ✗ | ✓ | ✓ | ✓ | ✗ | ✓ |
| Gaussian Std Diff (D1b) | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ |
| Hypercube Vary Size (D1b) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Hypersphere Surface (D1b) | ✗ | ✗ | ✓ | ✗ | ✓ | ✗ |
| Mode Drop+Invention (D1b) | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ |
| 1 Disjoint + Many Ident (D1b) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Discrete vs Continuous (D1b) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Scaling One Dim (D5) | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Hypercube Vary Size (D3) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

**关键发现**：

- **没有任何一个 fidelity 指标通过所有测试**
- **D3（数据需求）全军覆没**：所有指标在样本量变化时都不稳定
- **D1b 高维测试全失败**：所有指标在"一个维度有差异+多个维度相同"的设置下均失败
- **Discrete vs Continuous 全失败**：没有指标能区分离散与连续数值分布
- Density 和 P-Prec 相对较好（D1b 通过较多），但仍有大量失败
- I-Prec 缺乏 outlier 鲁棒性和缩放不变性

### Diversity 指标结果

Diversity 指标同样普遍失败：
- 所有指标在 Hypercube Varying Sample Size（D3）上失败
- 区分离散 vs 连续分布的能力普遍不足
- Coverage 在多项 D4（上下界）测试中表现较好，但 D1b 也有显著失败

### 核心结论与实践建议

1. **所有指标都有缺陷**——不存在可以放心使用的"黄金指标"
2. **对从业者的建议**：使用这些指标时必须了解其局限，不应将某个指标的高分解读为生成质量无条件好
3. **对研究者的呼吁**：社区应投入更多精力开发新指标而非新模型，新指标必须通过广泛的 sanity check 验证

## 亮点与洞察

1. **方法论价值极高**：将散落在不同论文中的失败案例统一为可复现的程式化测试套件，形成标准化 benchmark
2. **Desiderata 框架全面**：六条准则抓住了指标设计的核心需求，可作为未来指标设计的参考标准
3. **High/Low diversity 区分**：对 diversity 指标的模糊地带给出了合理的处理方式，避免不公平的判定
4. **覆盖表格数据**：补充了表格数据特有的测试场景（重尾分布、离散 vs 连续），填补了以往图像中心评估的盲区
5. **开源代码**：全部 sanity check 代码公开，便于后续研究者复现和扩展

## 局限与展望

1. **只评估不修复**：作为 position paper 仅暴露问题，未提出具体的替代指标
2. **sanity check 使用人工分布**：所有测试都基于合成的简单分布（高斯、超立方体等），与真实数据（自然图像、复杂表格）的行为可能存在差异
3. **图像领域的 embedding 未深入分析**：对 InceptionV3 等预训练嵌入本身带来的偏差未做系统评估
4. **未覆盖曲线值指标**：排除了 Sajjadi et al. 2018 等返回曲线而非单值的指标
5. **部分指标因计算代价被排除**：如 Kim et al. 2023 的拓扑指标，但这类指标可能有独特优势
6. **缺少对最新扩散模型评估场景的讨论**：现代大规模扩散模型的评估需求可能与传统 GAN 评估有所不同

## 相关工作与启发

- **Borji (2019, 2022); Xu et al. (2018)**：早期 GAN 评估指标综述，提出了部分重叠的 desiderata，但要求单一指标同时衡量多个方面
- **Theis et al. (2016)**：发现经典指标可产生矛盾评估
- **Theis (2024)**：理论探讨 fidelity（"realism"）指标应具备的性质
- **Sajjadi et al. (2018) → Kynkäänniemi et al. (2019)**：precision/recall 指标的开创与改进
- 本文对未来指标设计的启发：需要从设计之初就建立系统化的 sanity check 验证流程，而非事后发现问题再打补丁

## 评分
- 新颖性: ⭐⭐⭐（Position paper 不提新方法，但系统性评估框架有创新）
- 实验充分度: ⭐⭐⭐⭐⭐（14 项 sanity check × 12 个指标，覆盖全面）
- 写作质量: ⭐⭐⭐⭐（结构清晰，desiderata-check-result 三段式逻辑严密）
- 价值: ⭐⭐⭐⭐（对社区有警醒作用，sanity check 套件可成为标准验证工具）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] All-atom Diffusion Transformers: Unified Generative Modelling of Molecules and Materials](all-atom_diffusion_transformers_unified_generative_modelling_of_molecules_and_ma.md)
- [\[NeurIPS 2025\] Continuous Uniqueness and Novelty Metrics for Generative Modeling of Inorganic Crystals](../../NeurIPS2025/image_generation/continuous_uniqueness_and_novelty_metrics_for_generative_modeling_of_inorganic_c.md)
- [\[ICML 2026\] Balancing Fidelity and Diversity in Diffusion Models via Symmetric Attention Decomposition: Hopfield Perspective](../../ICML2026/image_generation/balancing_fidelity_and_diversity_in_diffusion_models_via_symmetric_attention_dec.md)
- [\[NeurIPS 2025\] Pairwise Optimal Transports for Training All-to-All Flow-Based Condition Transfer Model](../../NeurIPS2025/image_generation/pairwise_optimal_transports_for_training_all-to-all_flow-based_condition_transfe.md)
- [\[NeurIPS 2025\] Evaluating the Evaluators: Metrics for Compositional Text-to-Image Generation](../../NeurIPS2025/image_generation/evaluating_the_evaluators_metrics_for_compositional_text-to-image_generation.md)

</div>

<!-- RELATED:END -->
