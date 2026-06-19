---
title: >-
  [论文解读] Omni-Mol: Multitask Molecular Model for Any-to-Any Modalities
description: >-
  [NeurIPS2025][计算生物][分子大语言模型] 提出 Omni-Mol，一个基于多模态 LLM 的统一分子理解与生成框架，通过构建 142 万样本的指令微调数据集、Gradient Adaptive LoRA (GAL) 和 Mixture-of-GAL-Experts (MoGE) 架构，首次在单一模型中统一学习 16 个分子任务（Mol2Mol/Mol2Text/Mol2Num/Text2Mol），以仅 2.2B 参数在 13 个任务上达到 SOTA。
tags:
  - "NeurIPS2025"
  - "计算生物"
  - "分子大语言模型"
  - "多任务学习"
  - "混合专家"
  - "自适应LoRA"
  - "统一指令微调"
---

# Omni-Mol: Multitask Molecular Model for Any-to-Any Modalities

**会议**: NeurIPS2025  
**arXiv**: [2502.01074](https://arxiv.org/abs/2502.01074)  
**作者**: Chengxin Hu, Hao Li, Yihe Yuan, Zezheng Song, Chenyang Zhao, Haixin Wang (NUS, UMD, UCLA)
**代码**: [Omni-Mol-Code](https://github.com/)  
**领域**: 计算生物  
**关键词**: 分子大语言模型, 多任务学习, 混合专家, 自适应LoRA, 统一指令微调

## 一句话总结

提出 Omni-Mol，一个基于多模态 LLM 的统一分子理解与生成框架，通过构建 142 万样本的指令微调数据集、Gradient Adaptive LoRA (GAL) 和 Mixture-of-GAL-Experts (MoGE) 架构，首次在单一模型中统一学习 16 个分子任务（Mol2Mol/Mol2Text/Mol2Num/Text2Mol），以仅 2.2B 参数在 13 个任务上达到 SOTA。

## 研究背景与动机

构建通用分子 AI（AI Chemist）是药物发现和化学研究的核心目标，但现有分子多模态 LLM 距"one-model-fits-all"仍有三大差距：

**数据规模不足且覆盖不全**：现有分子指令数据集规模小，任务类型受限。如 PRESTO 支持 Mol2Num 和 Mol2Mol 但不支持 Mol2Text 和 Text2Mol；InstructMol 不支持 Text2Mol

**多任务联合学习困难**：不同分子子领域的任务存在显著分布差异和任务竞争，LLM 难以稳定地同时学好所有任务

**内在维度不匹配**：不同任务和模态在语言空间中需要不同的内在维度（intrinsic dimension），标准 LoRA 使用固定 rank 无法兼顾冗余和不足

核心目标：构建一个真正支持任意模态组合（任意输入→任意输出）的通用分子模型，同时解决多任务学习中的维度自适应和任务冲突问题。

## 方法详解

### 任务分类与数据集构建

将小分子任务按输入输出模态创新性地分为四大类：

- **Mol2Mol**（68.9万样本）：前向反应预测、逆合成、试剂预测、溶剂预测、催化剂预测、分子编辑
- **Mol2Num**（41.2万样本）：HOMO-LUMO 量子力学性质预测、分子量、TPSA、LogP、产率预测
- **Mol2Text**（24.8万样本）：实验过程描述、描述 Q&A、分子标题生成（Molcap）
- **Text2Mol**（7.3万样本）：IUPAC→SELFIES 转换、文本引导分子生成

总计 **142 万样本**，为迄今最大的分子指令微调数据集。分子表示采用 SELFIES 而非 SMILES，因为 SELFIES 可保证解码后的分子有效性。

### 整体架构

Omni-Mol 由三部分组成：
1. **LLM 骨干**：LLaMA 3.2-1B
2. **图编码器** $f_{\mathcal{G}}$：MoleculeSTM，编码分子 2D/3D 图结构
3. **投影器** $f_p$：单层线性层，将图表示对齐到 LLM 隐藏空间

建模为自回归生成：$P(\mathbf{Y}|\mathbf{X}_I, \mathbf{X}_S, \mathbf{H}_G) = \prod_i P_\theta(\mathbf{Y}_i | \mathbf{X}_I, \mathbf{X}_S, \mathbf{H}_G, \mathbf{Y}_{<i})$

### Gradient Adaptive LoRA (GAL)

**动机**：实验发现不同任务的最优 LoRA rank 不同（如前向预测最优 rank=128，Molcap 最优 rank=32），固定 rank 的标准 LoRA 无法适应多任务的内在维度差异。

**设计**：引入可学习的动态缩放因子替代 LoRA 的固定缩放：
$$\gamma_\theta = \alpha / r^p + \beta$$
其中 $\theta = \{\alpha, p, \beta\}$ 为可学习参数。$p$ 指数建模 rank 效应，$\beta$ 提供直接调整。训练过程中动态调整梯度幅度，使 adapter 自适应数据的内在维度。

### Mixture-of-GAL-Experts (MoGE)

**动机**：模型需同时处理图特征、文本、SELFIES 等多模态，且 SELFIES 虽以文本形式输入但语义上与自然语言差异大。

**设计**：
- 在 LLM 后 3/4 层的 FFN 替换为 MoGE 层
- 包含 $\mathcal{N}$ 个**路由专家**（学习专业知识）+ 1 个**共享专家**（学习跨任务通用知识）
- 所有专家从预训练 FFN 权重初始化，路由器 Kaiming 均匀初始化
- 实际配置：5 个专家中 2 个路由 + 1 个共享
- MHA 层包裹 GAL adapter，前 1/4 层 FFN 也包裹 GAL

### 两阶段训练

- **Stage 1（多模态对齐）**：在 PubChem 上学习通过图特征描述分子，仅训练投影器
- **Stage 2（统一指令微调）**：冻结预训练参数，训练 GAL adapters、专家路由器和投影器；总损失 = 语言模型损失 + $\lambda$ × 负载均衡损失

## 实验关键数据

### Table 3: Mol2Mol 核心任务（与专家模型和通用模型对比）

| 任务 | 模型 | 参数量 | 类型 | Exact Match | Morgan ↑ | Lev ↓ |
|------|------|--------|------|-------------|----------|-------|
| **前向反应预测** | InstructMol | 6.7B | 专家 | 0.54 | 0.74 | 10.85 |
| | PRESTO | 3.2B | 通用 | 0.69 | 0.84 | 6.53 |
| | **Omni-Mol** | **2.2B** | **通用** | **0.73** | **0.87** | **5.55** |
| **逆合成** | InstructMol | 6.7B | 专家 | 0.41 | 0.71 | 13.97 |
| | PRESTO | 3.2B | 通用 | 0.53 | 0.79 | 10.30 |
| | **Omni-Mol** | **2.2B** | **通用** | **0.57** | **0.83** | **8.97** |
| **试剂预测** | PRESTO | 3.2B | 通用 | 0.21 | 0.48 | 16.31 |
| | **Omni-Mol** | **2.2B** | **通用** | **0.23** | **0.52** | **14.59** |
| **溶剂预测** | PRESTO | 6.7B | 通用 | 0.42 | 0.51 | 2.76 |
| | **Omni-Mol** | **2.2B** | **通用** | **0.52** | **0.64** | **2.71** |

Omni-Mol 以仅 33% 参数量超越几乎所有专家基线，在前向预测、逆合成、试剂预测分别较 PRESTO 提升约 5%、7%、9%。

### Table 3 续: Mol2Num 与 Mol2Text 任务

| 任务 | 模型 | MAE / 指标 | 结果 |
|------|------|-----------|------|
| **HOMO-LUMO** | InstructMol | Avg MAE | 0.0050 |
| | **Omni-Mol** | **Avg MAE** | **0.0044** (↓12%) |
| **分子量/LogP/TPSA** | 3D-MoLM | MAE | 14.79 / 0.66 / 9.71 |
| | **Omni-Mol** | **MAE** | **11.07 / 0.49 / 5.89** (↓25-39%) |
| **Molcap** | HIGHT | BLEU-4 | 0.397 |
| | **Omni-Mol** | **BLEU-4** | **0.440** (↑11%) |
| **Description Q&A** | 3D-MoLM | BLEU-4 | 0.26 |
| | **Omni-Mol** | **BLEU-4** | **0.44** (↑69%) |

### 缩放性实验

- **数据缩放**：从 20% → 100% 数据量，性能持续提升，未见饱和，表明更多数据可进一步增强
- **参数缩放**：LLaMA 1B → 3B → 8B，所有任务性能随模型增大而提升
- 数据缩放的收益比参数缩放更显著，说明数据扩展仍有很大空间

### 消融实验

- **联合训练 vs 分离训练**：联合训练在 Omni-Mol 数据集上始终优于单任务分离训练
- **去除 GAL**：使用标准 LoRA 替代 GAL 后性能一致下降
- **去除 MoE**：仅用 GAL 不加 MoGE 扩展，在试剂预测、Molcap、产率回归等多任务上性能下降，产率回归下降最为显著
- **表征收敛分析**：随任务数从 1→8 增加，Omni-Mol 学到的表征互相似度持续增加（收敛到通用表示），而 InstructMol 的表征则越来越不相似（发散）

## 亮点

- **最全面的分子通用模型**：首个同时支持 Mol2Mol / Mol2Text / Mol2Num / Text2Mol 四种模态的统一框架，覆盖 16 个任务，142 万样本
- **GAL 自适应机制**：通过可学习缩放因子解决多任务内在维度不匹配问题，简洁有效，直接解决了固定 rank LoRA 在多任务场景中的根本限制
- **MoGE 架构**：共享专家+路由专家的组合既保持通用知识又实现任务特化，融合了 MoE 和自适应 LoRA 的双重优势
- **强缩放性证据**：数据和参数双维度缩放均展现清晰趋势，且表征收敛分析为"通用分子表示"假说提供了实证支持
- **参数效率极高**：2.2B 参数超越 6.7B 专家模型，甚至优于 685B DeepSeekV3 的 few-shot 表现

## 局限性

- **计算资源受限**：未能在更大规模上探索模型性能上限，8B 模型的缩放趋势表明更大模型仍有提升空间
- **仅限小分子**：当前数据集和任务聚焦小分子，未覆盖蛋白质及蛋白质-小分子相互作用等重要生物学场景
- **MoGE 配置较固定**：专家数量（5个）和 MoGE 层起始位置（1/4L）均为超参数，不同任务规模下最优配置可能不同
- **SELFIES 依赖**：虽然 SELFIES 保证有效性，但社区主流仍以 SMILES 为主，可能限制与现有工具链的兼容性
- **缺少下游应用验证**：未在实际药物发现 pipeline（如分子对接、ADMET 预测）中验证端到端价值

## 相关工作

- **分子基础模型**：Mol-Instruction（首个分子指令微调）→ InstructMol（引入 2D 图+多 LoRA）→ HIGHT（多层 2D 图特征）→ 3D-MoLM（3D 分子表示）→ PRESTO（领域预训练+多任务）→ Omni-Mol 实现了最全面的"one-model-fits-all"
- **统一生成建模**：受 GPT/Flamingo/LLaVA 统一多模态理解与生成启发，将范式引入分子领域
- **LoRA 及变体**：在标准 LoRA 基础上引入可学习动态缩放，结合 MoE 实现自适应多任务微调
- **表征收敛假说**：受 Platonic Representation Hypothesis 启发，验证了多任务训练中分子表征向通用空间收敛的趋势

## 评分

- 新颖性: ⭐⭐⭐⭐ — GAL 和 MoGE 的结合新颖，任务分类体系清晰，但核心组件（LoRA + MoE）均为已有技术的组合
- 实验充分度: ⭐⭐⭐⭐⭐ — 16 任务全面评测，消融实验、缩放实验、表征收敛分析均完整，基线对比包含专家和通用模型
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，动机推导严密，但符号和公式较多，部分表格信息密度极高
- 价值: ⭐⭐⭐⭐ — 为构建通用 AI 化学家提供了扎实的基线框架和大规模数据集，数据和模型均开源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Mol-LLaMA: Towards General Understanding of Molecules in Large Molecular Language Models](mol-llama_towards_general_understanding_of_molecules_in_large_molecular_language.md)
- [\[NeurIPS 2025\] Consistent Sampling and Simulation: Molecular Dynamics with Energy-Based Diffusion Models](consistent_sampling_and_simulation_molecular_dynamics_with_energy-based_diffusio.md)
- [\[NeurIPS 2025\] EDBench: Large-Scale Electron Density Data for Molecular Modeling](edbench_large-scale_electron_density_data_for_molecular_modeling.md)
- [\[NeurIPS 2025\] Iterative Foundation Model Fine-Tuning on Multiple Rewards](iterative_foundation_model_fine-tuning_on_multiple_rewards.md)
- [\[NeurIPS 2025\] JAMUN: Bridging Smoothed Molecular Dynamics and Score-Based Learning for Conformational Ensembles](jamun_bridging_smoothed_molecular_dynamics_and_score-based_learning_for_conforma.md)

</div>

<!-- RELATED:END -->
