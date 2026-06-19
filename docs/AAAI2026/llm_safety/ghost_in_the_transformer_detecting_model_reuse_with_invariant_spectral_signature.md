---
title: >-
  [论文解读] Ghost in the Transformer: Detecting Model Reuse with Invariant Spectral Signatures
description: >-
  [LLM安全] 提出 GhostSpec，一种无需数据、不修改模型行为的白盒方法，通过对注意力权重矩阵的不变乘积做 SVD 提取光谱指纹，可在微调、剪枝、合并、扩展甚至对抗性变换下稳健地验证 LLM 血统。 - LLM 训练成本极高，许多开发者基于开源模型微调后发布，大多遵守开源协议；但也有人将微调衍生模型伪称为"从零训练"…
tags:
  - "LLM安全"
---

# Ghost in the Transformer: Detecting Model Reuse with Invariant Spectral Signatures

- **会议**: AAAI 2026
- **arXiv**: [2511.06390](https://arxiv.org/abs/2511.06390)
- **代码**: [DX0369/GhostSpec](https://github.com/DX0369/GhostSpec)
- **领域**: AI Safety / 模型知识产权保护
- **关键词**: LLM 溯源, 模型指纹, 奇异值分解, 注意力矩阵, 光谱不变量

## 一句话总结

提出 GhostSpec，一种无需数据、不修改模型行为的白盒方法，通过对注意力权重矩阵的不变乘积做 SVD 提取光谱指纹，可在微调、剪枝、合并、扩展甚至对抗性变换下稳健地验证 LLM 血统。

## 背景与动机

- LLM 训练成本极高，许多开发者基于开源模型微调后发布，大多遵守开源协议；但也有人将微调衍生模型伪称为"从零训练"（如 Llama3-V 事件），构成知识产权侵权。
- **黑盒方法**（行为指纹、水印）对解码随机性敏感、易被对抗性改写破坏，且水印需要模型创建者配合。
- **白盒方法**中：表示空间方法依赖输入数据、计算开销大；直接权重比较在微调/剪枝后脆弱。
- 已有研究表明：大奇异值编码预训练知识，在微调过程中保持稳定；微调主要影响小奇异值方向。这启发作者利用**大奇异值谱**作为模型的稳健指纹。
- 额外挑战：攻击者可对权重做置换（permutation）或缩放（scaling）变换，不改变功能但大幅改变权重分布，直接比较单个 Q/K/V/O 矩阵的奇异值会失效。

## 方法详解

### 1. 不变光谱指纹构造

核心洞察：单独的 $W_q, W_k, W_v, W_o$ 矩阵的奇异值可被置换/缩放攻击改变，但特定**矩阵乘积**的奇异值谱对功能保持变换具有不变性。

对每层 $i$，定义两个不变矩阵：

$$M_{qk}^{(i)} = W_q^{(i)} (W_k^{(i)})^T, \quad M_{vo}^{(i)} = W_v^{(i)} W_o^{(i)}$$

- $M_{qk}$ 对应注意力分数计算中的 Q-K 交互。
- $M_{vo}$ 对应 value-output 投影。

**不变性证明**：对 V-O 做可逆变换 $\tilde{W}_v = CW_v, \tilde{W}_o = W_oC^{-1}$（功能不变），乘积 $\tilde{W}_v \tilde{W}_o = CW_vW_oC^{-1}$，与原矩阵相似，奇异值不变。Q-K 变换同理。

每层的指纹定义为：

$$\mathcal{S}_M^{(i)} = (\mathbf{s}_{qk,M}^{(i)}, \mathbf{s}_{vo,M}^{(i)})$$

其中 $\mathbf{s}_{p,M}^{(i)} = \text{SVD}(M_{p,M}^{(i)})$。完整模型指纹 $\mathcal{F}_M$ 为所有层指纹的序列。

### 2. 双重相似度度量

#### GhostSpec-mse（细粒度逐层比较）

对两个模型 A（N 层）、B（M 层），构建聚合距离矩阵 $D_{\text{avg}} \in \mathbb{R}^{N \times M}$：

$$(\text{D}_{\text{avg}})_{ij} = \frac{1}{2} \sum_{p \in \{qk, vo\}} \frac{1}{r_{p,ij}} \|\hat{\mathbf{s}}_{p,A}^{(i)} - \hat{\mathbf{s}}_{p,B}^{(j)}\|_2^2$$

- $r_{p,ij}$ 为两个奇异值向量有效秩的最小值，用于截断。
- $\hat{\mathbf{s}}$ 经截断后做 min-max 归一化到 $[0,1]$。

使用 **POSA 算法**（Penalty-based Optimal Spectral Alignment）在距离矩阵中找最优对齐路径，处理模型深度不同的情况。最终用反 Sigmoid 将路径平均 MSE 转为相似度分数：

$$\text{Sim}_{\text{MSE}}(A,B) = 1 - \frac{1}{1+e^{-k(d_{\text{path}} - \tau)}}$$

#### GhostSpec-corr（轻量级趋势相关性）

- 每层每个不变分量取 top-K 归一化奇异值的均值，形成趋势序列 $\boldsymbol{\mu}_{qk}, \boldsymbol{\mu}_{vo}$。
- 用动态序列对齐算法匹配不同长度的序列。
- 拼接对齐后的序列，计算**距离相关系数**（distance correlation）作为相似度。
- 同源模型趋势高度相关，无关模型趋势发散。

### 3. POSA 对齐算法

- 解决模型层数不一致（如结构化剪枝、层扩展）的对齐问题。
- 输入距离矩阵 $D \in \mathbb{R}^{N \times M}$（$N \leq M$）和间隙惩罚 $\rho$。
- 用动态规划找最小代价的单调对齐路径，跳过的层产生惩罚 $\rho$。
- 输出最优路径上的平均 MSE 距离。

## 实验

### 实验设置

- 以 Llama-2-7b 和 Mistral-7B 为基础模型，构建 55 对模型数据集。
- 变换类型：微调、非结构化剪枝（30%/50%/70%）、结构化剪枝、模型合并、MoE 扩展、置换/缩放对抗攻击。
- 基线：QueRE（黑盒）、Logits（黑盒）、REEF（白盒需数据）、PCS（白盒无数据）。

### 表1：Llama-2-7b 为基础模型的综合对比

| 方法 | 数据依赖 | Vicuna微调 | Llemma微调 | 缩放攻击 | 置换攻击 | 剪枝50% | 剪枝70% |
|------|---------|-----------|-----------|---------|---------|--------|--------|
| QueRE | 需数据 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| REEF | 需数据 | 0.999 | 0.998 | 1.000 | 1.000 | 0.997 | 0.995 |
| PCS | 无数据 | 0.999 | 0.505 | 0.597 | 0.386 | 0.906 | 0.783 |
| GhostSpec-mse | 无数据 | 0.976 | 0.953 | 0.976 | 0.976 | 0.973 | 0.965 |
| GhostSpec-corr | 无数据 | 0.999 | 0.760 | 1.000 | 1.000 | 0.897 | 0.705 |

### 表2：结构化剪枝、合并与无关模型

| 方法 | Sheared-1.3B | Sheared-2.7B | SLERP合并 | MoE扩展 | Qwen2.5(无关↓) | OPT(无关↓) |
|------|-------------|-------------|----------|---------|---------------|-----------|
| PCS | 0.000 | 0.000 | 0.999 | 0.020 | 0.000 | 0.000 |
| GhostSpec-mse | 0.889 | 0.905 | 0.976 | 0.976 | 0.000 | 0.503 |
| GhostSpec-corr | 0.940 | 0.941 | 1.000 | 1.000 | 0.294 | 0.342 |

### 最大 F1 分数

- **GhostSpec-mse: F1 = 0.9867**（最优）
- **GhostSpec-corr: F1 = 0.9730**
- REEF: 0.9474, QueRE: 0.8649, Logits: 0.8378, PCS: 0.7945

## 关键发现

1. **光谱稳定性**：大奇异值编码预训练知识，微调仅影响小奇异值方向，因此光谱指纹在微调后保持稳定。
2. **不变矩阵乘积**：$W_qW_k^T$ 和 $W_vW_o$ 的奇异值对置换/缩放攻击严格不变，从理论上保证了指纹的鲁棒性。
3. **对抗性规避困难**：用联合损失（任务损失 + 光谱散度）尝试擦除指纹，实验显示难以在不损害性能的情况下显著改变光谱特征。
4. **案例研究**：GhostSpec 发现 Pangu-Pro-MoE 与 Qwen2.5-14B 系列高度相似，为血统争议提供了定量证据。
5. **MLP 模块也有效**：MLP 的 up_proj/down_proj 奇异值同样可以区分衍生模型与无关模型，但计算开销更大且对 MoE 扩展不够鲁棒。

## 亮点

- **完全无数据**：不需要任何输入样本，纯静态权重分析，计算开销极低。
- **理论保证**：从功能保持变换的数学定义出发，严格证明了指纹的不变性。
- **POSA 算法**：优雅地解决了不同深度模型的层对齐问题，使方法能处理结构化剪枝和模型扩展场景。
- **实用性强**：在开源权重生态中可直接使用，有开源代码，适合实际的知识产权审查。
- **双指标互补**：MSE 捕捉细粒度差异，corr 捕捉宏观趋势，两者结合覆盖更广泛的场景。

## 局限性

- **仅适用于白盒场景**：需要完整的模型权重访问，无法用于闭源 API-only 模型。
- **OPT 误判**：GhostSpec-mse 在部分无关模型上给出 0.503 的中间分数（如 OPT-6.7b），存在误判风险。
- **阈值依赖**：Sigmoid 转换中的阈值 $\tau$ 和斜率 $k$ 需要经验调整，不同数据集可能需要重新标定。
- **MoE 架构**：虽然 POSA 算法可以处理深度不同，但对于注意力层被大幅修改的架构变体（如 expert 级别重构），鲁棒性未充分验证。
- **实验规模有限**：55 对模型的数据集偏小，更大规模的评估（如数百模型对）会更有说服力。
- **未探讨跨模态场景**：方法针对纯文本 LLM 设计，对多模态模型的适用性未知。

## 相关工作

- **行为指纹**：LLMMap、模型输出风格分析，被动但对解码随机性敏感。
- **水印方法**：InstructMark、KGW 水印，需模型创建者配合嵌入。
- **表示空间方法**：REEF（CKA 相似度）、梯度统计，有效但依赖数据。
- **权重直接分析**：HuRef/PCS（不变子矩阵）、内在维度分析。
- **随机矩阵理论**：Staats et al. 发现大奇异值的 Marchenko-Pastur 偏差编码了模型身份。

## 评分

⭐⭐⭐⭐ (4/5)

问题重要且实用，方法有数学保证且实现简洁。理论与实验结合紧密，不变性证明清晰。主要扣分在实验规模偏小和部分场景（OPT）的判别力不足，以及白盒限制在实际版权纠纷中可能受限。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Multi-Faceted Attack: Exposing Cross-Model Vulnerabilities in Defense-Equipped Vision-Language Models](multi-faceted_attack_exposing_cross-model_vulnerabilities_in_defense-equipped_vi.md)
- [\[ICLR 2026\] Unlearning Evaluation through Subset Statistical Independence](../../ICLR2026/llm_safety/unlearning_evaluation_through_subset_statistical_independence.md)
- [\[ICML 2025\] Sorbet: A Neuromorphic Hardware-Compatible Transformer-Based Spiking Language Model](../../ICML2025/llm_safety/sorbet_a_neuromorphic_hardware-compatible_transformer-based_spiking_language_mod.md)
- [\[ACL 2026\] Gap-K%: Measuring Top-1 Prediction Gap for Detecting Pretraining Data](../../ACL2026/llm_safety/gap-k_measuring_top-1_prediction_gap_for_detecting_pretraining_data.md)
- [\[ACL 2026\] Detecting RAG Extraction Attack via Dual-Path Runtime Integrity Game](../../ACL2026/llm_safety/detecting_rag_extraction_attack_via_dual-path_runtime_integrity_game.md)

</div>

<!-- RELATED:END -->
