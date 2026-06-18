---
title: >-
  [论文解读] HAWAII: Hierarchical Visual Knowledge Transfer for Efficient VLM
description: >-
  [NeurIPS 2025][多模态VLM][知识蒸馏] 提出 Hawaii 框架，通过混合 LoRA 适配器（MoLA）和分层知识蒸馏（HKD），将多个视觉专家的知识蒸馏到单个视觉编码器中，在不增加推理成本的前提下显著提升 VLM 的视觉理解能力。 VLM 的性能很大程度上取决于视觉编码器的能力。近期研究表明…
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "知识蒸馏"
  - "视觉编码器"
  - "LoRA"
  - "MoE"
  - "多教师蒸馏"
---

# HAWAII: Hierarchical Visual Knowledge Transfer for Efficient VLM

**会议**: NeurIPS 2025  
**arXiv**: [2506.19072](https://arxiv.org/abs/2506.19072)  
**代码**: [有](https://github.com/yimuwangcs/wise-hawaii)  
**领域**: 多模态VLM  
**关键词**: 知识蒸馏, 视觉编码器, LoRA, MoE, 多教师蒸馏

## 一句话总结

提出 Hawaii 框架，通过混合 LoRA 适配器（MoLA）和分层知识蒸馏（HKD），将多个视觉专家的知识蒸馏到单个视觉编码器中，在不增加推理成本的前提下显著提升 VLM 的视觉理解能力。

## 研究背景与动机

VLM 的性能很大程度上取决于视觉编码器的能力。近期研究表明，融合多个视觉专家（如 SAM、ConvNeXt、EVA 等）可以大幅提升性能，但存在严重的效率问题：

**推理时成本高**：多专家方案需要在训练和推理时都计算所有专家的视觉 token，计算和延迟开销大

**多教师知识冲突**：不同教师的训练数据、架构和目标各异，直接蒸馏会产生噪声和冗余知识

**现有蒸馏方法不足**：MoVE-KD 使用固定的 LoRA 适配器集处理所有教师，无法有效区分不同教师的知识

核心问题：**如何在保持单编码器推理效率的同时，有效吸收多个视觉专家的互补知识？**

## 方法详解

### 整体框架

Hawaii 遵循标准 VLM 架构（视觉编码器 → 投影器 → LLM），关键创新在视觉编码器部分。它由两个核心模块组成：
1. **MoLA（Mixture of LoRA Adapters）**：管理教师特异性和通用知识的适配器
2. **HKD（Hierarchical Knowledge Distillation）**：在细粒度和粗粒度两个层次进行知识蒸馏

### 关键设计

#### 1. 混合 LoRA 适配器（MoLA）

MoLA 应用于学生编码器（CLIP）的每个前馈层，包含两组适配器：

**教师特异性 LoRA 适配器** $\{a_i^T\}_{i=1}^{N_t}$：
- 每个适配器只对齐一个教师，避免不同教师知识间的冲突
- 由稀疏路由器 $f_r^T(\cdot)$ 根据隐层输入动态选择

**通用知识 LoRA 适配器** $\{a_i^G\}_{i=1}^{N_g}$：
- 学习多教师的集体共识
- 由独立稀疏路由器 $f_r^G(\cdot)$ 选择

前馈层输出：$F^*(h) = F(h) + a_i^T(h) + a_j^G(h)$，其中 $i = \text{argmax}(f_r^T(h))$，$j = \text{argmax}(f_r^G(h))$。

每个适配器为 LoRA 块（rank=32），路由器为 2 层 MLP + GELU，每次只激活 top-1 适配器（稀疏设计）。

#### 2. 粗粒度知识蒸馏（CGKD）

目标：蒸馏多教师的集体共识。

- 将各教师的视觉特征通过 pixel unshuffle 统一到学生的 token 长度
- 通道拼接后通过 2 层 MLP 生成汇总特征：$I_{cg}^T = f_{cg}(\text{Concat}(I_1^T, I_2^T, ..., I_{N_t}^T))$
- 用 MSE 损失对齐学生输出与汇总特征：$\mathcal{L}_{cg} = \text{MSE}(I^S, I_{cg}^T)$

通用 LoRA 适配器在此阶段发挥作用，学习全局对齐。

#### 3. 细粒度知识蒸馏（FGKD）

目标：精确学习每个教师的独特知识。

**教师特异性适配**：激活第 $i$ 个教师特异性 LoRA 时，学生输出 $I_i^S$ 只需与第 $i$ 个教师的特征 $I_i^T$ 对齐。

**Token 重要性评分**：并非所有 token 同等重要。通过相似度评分选择最有信息量的 token：
$$s_i = \text{mean}\left(\text{softmax}\left(\frac{\text{Concat}(\hat{I}_i^T, \hat{T})(\hat{I}_i^T)^\top}{\sqrt{D}}\right)\right)$$

这里同时考虑了教师视觉 token 和输入文本指令 $T$ 的相关性，优先学习与任务更相关的 token。

细粒度蒸馏损失：$\mathcal{L}_{fg} = \frac{1}{N_t} \sum_{i=1}^{N_t} s_i \cdot \text{MSE}(I_i^S, \hat{I}_i^T)$

### 损失函数 / 训练策略

**总体训练目标**：
$$\mathcal{L} = \mathcal{L}_{gen} + \lambda_1(\mathcal{L}_{fg} + \mathcal{L}_{cg}) + \lambda_2 \mathcal{L}_{mb}$$

- $\mathcal{L}_{gen}$：文本生成损失（自回归）
- $\lambda_1 = 0.5$：蒸馏损失权重
- $\lambda_2 = 0.05$：MoE 平衡损失权重
- $\mathcal{L}_{mb}$：MoE 负载均衡损失

**两阶段训练**（沿用 LLaVA-1.5 范式）：
1. **预训练**：558K 图文对，仅训练投影器、LoRA 适配器和路由器
2. **指令微调**：665K 指令数据，全模型训练

**教师配置**：
- 基础版 Hawaii：CLIP + ConvNeXt + EVA-02（3 个教师）
- Hawaii†：额外加 Pix2Struct（4 个教师）
- Hawaii‡：CLIP + ConvNeXt + EVA-02 + SAM

硬件：8 × NVIDIA A6000 (48GB)

## 实验关键数据

### 主实验

| 方法 | VQA-T | VizWiz | GQA | SQA | POPE | MME | MMB | MMMU | SeedB |
|------|-------|--------|-----|-----|------|-----|-----|------|-------|
| LLaVA-1.5 (Baseline) | 58.2 | 50.0 | 62.0 | 66.8 | 85.9 | 1510.7 | 64.3 | 34.7 | 66.1 |
| MoVE-KD | 58.3 | 52.3 | 63.2 | 69.4 | 86.9 | 1524.5 | 66.3 | - | - |
| **Hawaii** | **58.7** | **53.9** | **62.8** | **70.5** | **87.3** | **1540.2** | **66.9** | **36.6** | **67.5** |
| Δ vs Baseline | +0.5 | +3.9 | +0.8 | +3.7 | +1.4 | +29.5 | +2.6 | +1.9 | +1.4 |

Hawaii 在所有基准上均优于 LLaVA-1.5 和 MoVE-KD。VizWiz 提升最为显著（+3.9%），SQA 提升 +3.7%。

### 消融实验

| 配置 | Avg. |
|------|------|
| LLaVA-1.5 (Baseline) | 61.9 |
| + FGKD (无 token scoring) | 63.2 |
| + token scoring | 63.5 |
| + CGKD (完整 Hawaii) | **63.7** |

| 通用适配器数量 | MME | POPE | SeedB |
|---------------|-----|------|-------|
| 1 | 1516.2 | 84.5 | 67.4 |
| **3** | **1540.2** | **87.3** | **67.5** |
| 5 | 1530.2 | 85.2 | 66.9 |

### 关键发现

1. **逐步添加每个组件都带来提升**：FGKD → token scoring → CGKD，证明分层设计有效
2. **教师特异性 LoRA 优于共享 LoRA**：对比 MoVE-KD（共享适配器），Hawaii 的独立适配器策略更优
3. **通用适配器数量 3 个最优**：过多（5 个）反而下降，过少（1 个）表示能力不足
4. **13B 模型同样有效**：Hawaii-13B 在对应基准上也优于 LLaVA-1.5-13B 和 MoVE-KD-13B
5. **不同教师组合有差异**：添加 SAM（Hawaii‡）比添加 Pix2Struct（Hawaii†）在某些任务上更优

## 亮点与洞察

1. **推理零开销**：蒸馏完成后只用一个视觉编码器（带 LoRA），推理成本与基线相同
2. **MoLA 的 MoE 设计精巧**：教师特异性 + 通用知识的双路由机制，既避免冲突又学到共识
3. **Token 重要性评分考虑多信号**：同时考虑教师视觉特征和文本指令，比纯视觉选择更合理
4. **分层蒸馏思路通用**：FGKD 精确学习，CGKD 全局对齐，两者互补

## 局限与展望

1. 教师模型在训练时仍需前向传播（仅推理时免开销），训练阶段计算量较大
2. 仅以 LLaVA-1.5 为基线，未与更强的 VLM（如 InternVL2、Qwen-VL2）对比
3. 教师选择相对固定（CLIP/ConvNeXt/EVA/SAM），缺少自动选择机制
4. LoRA rank=32 和适配器数量为经验设定，缺少系统性搜索
5. 可探索动态路由权重而非 top-1 硬选择

## 相关工作与启发

- **与 MoVE-KD 的核心区别**：MoVE-KD 用固定 LoRA 适配器处理所有教师，Hawaii 用教师特异性适配器消除冲突
- **与 Eagle/Cambrian 的区别**：这些方法在推理时仍需多专家，Hawaii 仅在训练时用多专家
- **启发**：MoLA 的设计思路可推广到其他多源知识融合场景（如多模态 fusion、多任务学习）

## 评分

- 新颖性: ⭐⭐⭐⭐ (MoLA模块和分层蒸馏的组合设计有创意)
- 实验充分度: ⭐⭐⭐⭐ (10个基准+详细消融，但基线是较旧的LLaVA-1.5)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，图示直观)
- 价值: ⭐⭐⭐⭐ (零推理开销+一致性提升，实用性强)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Decoupled and Reusable Adaptation for Efficient Cross-Modal Transfer](../../CVPR2026/multimodal_vlm/decoupled_and_reusable_adaptation_for_efficient_cross-modal_transfer.md)
- [\[CVPR 2025\] MoVE-KD: Knowledge Distillation for VLMs with Mixture of Visual Encoders](../../CVPR2025/multimodal_vlm/move-kd_knowledge_distillation_for_vlms_with_mixture_of_visual_encoders.md)
- [\[ICCV 2025\] SparseVILA: Decoupling Visual Sparsity for Efficient VLM Inference](../../ICCV2025/multimodal_vlm/sparsevila_decoupling_visual_sparsity_for_efficient_vlm_inference.md)
- [\[NeurIPS 2025\] SpatialTraceGen: High-Fidelity Traces for Efficient VLM Spatial Reasoning Distillation](spatialtracegen_high-fidelity_traces_for_efficient_vlm_spatial_reasoning_distill.md)
- [\[AAAI 2026\] Harnessing Textual Semantic Priors for Knowledge Transfer and Refinement in CLIP-Driven Continual Learning](../../AAAI2026/multimodal_vlm/harnessing_textual_semantic_priors_for_knowledge_transfer_and_refinement_in_clip.md)

</div>

<!-- RELATED:END -->
