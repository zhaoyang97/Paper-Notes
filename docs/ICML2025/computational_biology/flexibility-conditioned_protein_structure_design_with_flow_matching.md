---
title: >-
  [论文解读] Flexibility-conditioned Protein Structure Design with Flow Matching
description: >-
  [ICML 2025][计算生物][蛋白质结构生成] 提出 BackFlip（从骨架预测残基级柔性）和 FliPS（以柔性 profile 为条件的 SE(3)-等变 flow matching 模型），首次实现根据目标柔性分布生成具有期望动态特性的蛋白质骨架结构，并通过 300 ns 分子动力学模拟验证。
tags:
  - "ICML 2025"
  - "计算生物"
  - "蛋白质结构生成"
  - "柔性条件化"
  - "Flow Matching"
  - "SE(3)-等变"
  - "分子动力学"
---

# Flexibility-conditioned Protein Structure Design with Flow Matching

**会议**: ICML 2025  
**arXiv**: [2508.18211](https://arxiv.org/abs/2508.18211)  
**代码**: [graeter-group/flips](https://github.com/graeter-group/flips)  
**领域**: 医学图像 / 蛋白质设计  
**关键词**: 蛋白质结构生成, 柔性条件化, Flow Matching, SE(3)-等变, 分子动力学

## 一句话总结

提出 BackFlip（从骨架预测残基级柔性）和 FliPS（以柔性 profile 为条件的 SE(3)-等变 flow matching 模型），首次实现根据目标柔性分布生成具有期望动态特性的蛋白质骨架结构，并通过 300 ns 分子动力学模拟验证。

## 研究背景与动机

当前深度学习蛋白质设计方法（如 RFdiffusion、FrameFlow）能生成高质量的蛋白质骨架，但生成的结构通常具有高热稳定性和结构刚性。然而，蛋白质的功能（催化、分子识别、别构调控等）高度依赖于其动态行为和局部柔性。现有方法仅能以静态属性（motif、对称性、结合靶标）作为条件，无法控制生成结构的柔性分布。

**核心问题**：如何让生成模型"理解"并控制蛋白质的动态特性？

**动机来源**：
- 酶在催化循环中需要 loop 开合、domain 运动等动态行为
- 蛋白质 binder 需要局部结构柔性才能与 DNA/RNA/配体结合
- 现有生成模型倾向于产生过度刚性的结构，限制了功能空间的探索
- 柔性度量（B-factor、pLDDT）存在各自局限性，需要更可靠的指标

## 方法详解

### 整体框架

本文提出一个两阶段框架：

1. **BackFlip**（Backbone Flexibility Predictor）：从蛋白质骨架结构预测每个残基的柔性（local RMSF），不依赖序列信息
2. **FliPS**（Flexibility-conditioned Protein Structure generation）：以目标柔性 profile 为条件的 SE(3)-等变条件 flow matching 模型

整体 pipeline：
1. 给定目标柔性 profile ξ
2. 用 FliPS 条件生成多个候选骨架
3. 用 BackFlip 预测每个候选骨架的柔性 profile
4. 通过 BackFlip 筛选（BFS）选出最匹配的候选
5. （可选）用分子动力学模拟验证

### 关键设计

#### 1. Local RMSF —— 新的柔性度量

传统 RMSF 基于全局叠合，存在非局部性导致的歧义和伪影。本文提出 **Local RMSF**，以局部邻域（窗口大小 S=12）进行对齐后计算残基波动：

$$\xi_i = \sqrt{\frac{1}{|\mathcal{C}|}\sum_{x \in \mathcal{C}} \left|(T_{\text{Align}}^{(S)} \circ x)_i - x_i^{(\text{ref})}\right|^2}$$

其中 $T_{\text{Align}}^{(S)}$ 是在残基 i 的序列邻域窗口上的局部刚体对齐变换。为消除参考构象的选择偏差，随机选 $N_{\text{ref}}=10$ 个参考构象取中位数。

#### 2. BackFlip 架构

- 基于 GAFL 架构中的 Clifford Frame Attention（CFA），SE(3)-等变 Transformer
- 骨架表示为 SE(3)^N 中的刚体帧序列
- 使用 $n_{\text{cfa}}=4$ 层 CFA 提取节点特征
- 节点特征通过 MLP + scaled sigmoid 映射到 $[0, \xi_{\max}]$（$\xi_{\max}=5$ Å）
- 仅 **0.68M 参数**（对比 GAFL 的 16.7M），节点/边嵌入维度分别为 64/32
- **关键特点**：纯骨架输入，不需要序列、进化信息或预训练语言模型

$$\xi_i = \xi_{\max} \cdot \sigma(\text{MLP}(h_i))$$

#### 3. FliPS 条件生成模型

FliPS 在 GAFL（无条件 flow matching 模型）基础上引入三个关键修改：

**a) 柔性嵌入（Flexibility Embedding）**：
- 将每个残基的柔性值离散化为 8 个 bin（最大值 $\xi_{\max}=3$ Å）
- 作为额外的节点输入特征传入模型

**b) 柔性辅助损失（Flexibility Auxiliary Loss）**：
- 利用 BackFlip 的可微性，对预测的骨架结构 $\hat{T}_1$ 计算柔性预测 $\xi_{\text{BF}}(\hat{T}_1)$
- 在原始 FrameDiff 辅助损失基础上加入柔性 MSE 惩罚项

$$l_{\text{aux}} = l_{\text{aux, FD}} + \frac{\lambda_{\text{flex}}}{N} \|\xi_1 - \xi_{\text{BF}}(\hat{T}_1)\|^2$$

其中 $\lambda_{\text{flex}} = 100$。这个设计之所以可行，是因为 BackFlip 对骨架结构可微且不依赖序列。

**c) 柔性遮蔽（Flexibility Masking）**：
- 训练时随机 mask 部分或全部柔性 profile，避免记忆化并保留无条件生成能力
- 类似 classifier-free guidance 的 drop-out 策略

#### 4. BackFlip Guidance（BG）—— 无训练替代方案

受 classifier-guidance 启发，在推理时将 BackFlip 梯度加入无条件模型的向量场：

$$\hat{v}_{\text{cond}}(T_t, t, \xi) = \hat{v}(T_t, t) - \eta \nabla_{T_t} \|\xi - \xi_{\text{BF}}(T_t)\|^2$$

其中 η 为引导强度超参数。消融实验表明 BG 性能不如直接训练的条件模型 FliPS。

#### 5. BackFlip 筛选（BFS）

生成后使用 BackFlip 筛选最佳候选：

$$s(\xi, \xi_{\text{ref}}) = w_{\text{corr}} \cdot r(\xi, \xi_{\text{ref}}) - w_{\text{mae}} \cdot \text{MAE}(\xi, \xi_{\text{ref}})$$

权重 $w_{\text{corr}}=1$，$w_{\text{mae}}=2$，综合考虑柔性 profile 的形状（Pearson 相关）和幅度（MAE）匹配。

### 损失函数 / 训练策略

**BackFlip 训练**：
- 数据集：ATLAS（1294 个蛋白质，300 ns MD 模拟）
- 损失：per-residue local RMSF 的 MSE
- 划分：1035 训练 / 130 验证 / 129 测试

**FliPS 训练**：
- 数据集：PDB 数据集（22977 个蛋白质，60-512 残基，用 BackFlip 标注柔性）
- 损失：flow matching 向量场回归 + 辅助损失（含柔性项）
- 训练量：8 × NVIDIA A100，共 21 GPU days
- $\lambda_{\text{flex}} = 100$，柔性嵌入 8 bins

## 实验关键数据

### 主实验

**BackFlip 柔性预测性能**

| 方法 | ATLAS 测试集 Global r ↑ | ATLAS MAE [Å] ↓ | De novo r ↑ | De novo MAE [Å] ↓ | 推理时间 [s] ↓ |
|------|------------------------|------------------|-------------|-------------------|---------------|
| MD（Ground Truth） | 0.84 | 0.14 | 0.80 | 0.10 | ~10,000 |
| B-factor | 0.16 | - | - | - | - |
| Negative pLDDT | 0.54 | - | 0.48 | - | 118 |
| **BackFlip** | **0.80** | **0.17** | **0.73** | **0.11** | **0.6** |

BackFlip 在 ATLAS 测试集上达到 0.80 的 Pearson 相关系数，接近 MD 模拟本身的噪声上界（0.84）；推理速度比 pLDDT 快约 200 倍，比 MD 模拟快约 17000 倍。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| FliPS（完整模型） | 最佳柔性匹配 | 条件 flow matching + 柔性辅助损失 + BFS |
| BackFlip Guidance (BG) + FrameFlow | 较差 | 无训练引导方式性能不如直接条件化 |
| 无 BackFlip 筛选 (BFS) | 柔性匹配下降 | 筛选对最终质量至关重要 |
| 无柔性辅助损失 | 条件遵循减弱 | 辅助损失显著提升柔性条件遵循度 |
| SCOPe 天然蛋白筛选 | 有限匹配 | 天然蛋白库难以覆盖任意柔性 profile |
| FoldFlow2 + BFS | 低于 FliPS | 无条件模型 + 筛选不如条件模型 |
| RFdiffusion + BFS | 低于 FliPS | 同上 |

### 关键发现

1. **BackFlip 显著优于现有柔性代理指标**：B-factor 相关仅 0.16，pLDDT 为 0.54，BackFlip 达 0.80
2. **FliPS 生成的骨架经 300 ns MD 验证**：确认生成结构确实展现目标柔性行为
3. **条件化 > 引导 > 筛选**：直接训练条件模型优于推理时引导，两者均优于从无条件样本中筛选
4. **结构多样性保持**：FliPS 在满足柔性条件的同时生成结构多样的蛋白质骨架
5. **Loop/turn 被正确识别为最柔性区域**，α-helix 和 β-sheet 核心区域最刚性
6. **泛化到 de novo 蛋白质**：BackFlip 在非天然蛋白上也保持 0.73 的高相关

## 亮点与洞察

1. **首创性**：第一个以结构柔性为条件的蛋白质骨架生成模型，填补了"静态条件"到"动态条件"的空白
2. **Local RMSF 的提出**：解决传统 RMSF 全局对齐的非局部性问题，是一个简洁但有价值的方法学贡献
3. **两阶段协同设计的优雅性**：BackFlip 既是独立的柔性预测工具，又通过可微性嵌入 FliPS 的训练损失，还用于推理时筛选——一个模型发挥三重作用
4. **纯骨架输入的设计选择**：BackFlip 不依赖序列，使其天然适配 de novo 设计场景（先生成骨架再设计序列）
5. **极低的参数量**：BackFlip 仅 0.68M 参数却达到接近 MD 的预测精度，展现了几何先验的强大归纳偏置
6. **实际速度提升**：相比 MD 模拟（~10000 秒），BackFlip 仅需 0.6 秒，使大规模筛选成为可能

## 局限与展望

1. **柔性标注依赖 BackFlip 而非真实 MD**：FliPS 训练数据的柔性标签来自 BackFlip 预测而非 MD 模拟，引入了预测器误差的传播
2. **蛋白质长度限制**：训练数据主要覆盖 60-512 残基，更大蛋白质的泛化性未验证
3. **仅考虑骨架层面**：未涉及侧链柔性和序列-柔性的协同设计
4. **Local RMSF 的窗口大小 S=12 为固定超参**：不同尺度的柔性特征可能需要自适应窗口
5. **MD 验证成本高**：最终验证仍需 300 ns MD 模拟，限制了大规模应用的闭环迭代速度
6. **缺少功能验证**：生成的柔性蛋白质是否真正具有催化或结合功能尚未在实验中验证
7. **BG 引导方案效果有限**：说明 classifier-guidance 范式在蛋白质结构生成中可能不如直接条件化有效

## 相关工作与启发

- **GAFL / FrameFlow**（Wagner et al., 2024; Yim et al., 2023）：FliPS 的基础架构，利用几何代数和 SE(3)-等变 flow matching
- **RFdiffusion**（Watson et al., 2023）：当前主流的蛋白质骨架生成模型，但仅支持静态条件
- **FlexPert-3D**（Kouba et al., 2024）：依赖预训练蛋白质语言模型的柔性预测器，BackFlip 在不使用 pLM 的情况下达到同等或更优性能
- **AlphaFold2**（Jumper et al., 2021）：IPA 机制和 pLDDT 柔性代理的来源
- **Classifier Guidance**（Dhariwal & Nichol, 2021）：BackFlip Guidance 的灵感来源
- **启发**：这种"先训练预测器 → 再用预测器的可微性驱动生成"的范式可推广到其他动态属性的条件化（如别构运动、结合自由能等）

## 评分

| 维度 | 分数 (1-10) | 说明 |
|------|-----------|------|
| 创新性 | 9 | 首个柔性条件化蛋白质生成模型，Local RMSF 指标也是新贡献 |
| 技术深度 | 8 | SE(3)-等变 flow matching + 可微预测器嵌入损失，数学表达严谨 |
| 实验完整性 | 8 | 覆盖天然蛋白/de novo 蛋白、多种 baseline 和消融、MD 验证 |
| 实用价值 | 7 | 蛋白质设计的真实需求，但缺少生物实验功能验证 |
| 写作质量 | 8 | 结构清晰、图表精美，动机阐述充分 |
| **综合** | **8** | 开创性工作，方法优雅且经 MD 模拟充分验证 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Improving Flow Matching by Aligning Flow Divergence](improving_flow_matching_by_aligning_flow_divergence.md)
- [\[ICLR 2026\] EvoFlows: Evolutionary Edit-Based Flow-Matching for Protein Engineering](../../ICLR2026/computational_biology/evoflows_evolutionary_edit-based_flow-matching_for_protein_engineering.md)
- [\[ICML 2025\] Efficient Molecular Conformer Generation with SO(3)-Averaged Flow Matching and Reflow](efficient_molecular_conformer_generation_with_so3-averaged_flow_matching_and_ref.md)
- [\[ICML 2025\] Scalable Generation of Spatial Transcriptomics from Histology Images via Whole-Slide Flow Matching](scalable_generation_of_spatial_transcriptomics_from_histology_images_via_whole-s.md)
- [\[ICML 2025\] Piloting Structure-Based Drug Design via Modality-Specific Optimal Schedule](piloting_structure-based_drug_design_via_modality-specific_optimal_schedule.md)

</div>

<!-- RELATED:END -->
