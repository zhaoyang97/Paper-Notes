---
title: >-
  [论文解读] TOSC: Task-Oriented Shape Completion for Open-World Dexterous Grasp Generation from Partial Point Clouds
description: >-
  [AAAI 2026][3D视觉][任务导向形状补全] 提出任务导向形状补全（TOSC）这一新任务，仅补全与操控任务相关的接触区域（而非整个物体），利用预训练基础模型生成候选形状、3D 判别自编码器筛选最优形状、FlowGrasp 流匹配模型生成灵巧抓取，在抓取位移和 Chamfer 距离上分别提升 16.17% 和 55.26%。
tags:
  - "AAAI 2026"
  - "3D视觉"
  - "任务导向形状补全"
  - "灵巧抓取"
  - "点云补全"
  - "流匹配"
  - "基础模型"
---

# TOSC: Task-Oriented Shape Completion for Open-World Dexterous Grasp Generation from Partial Point Clouds

**会议**: AAAI 2026  
**arXiv**: [2601.05499](https://arxiv.org/abs/2601.05499)  
**代码**: [github.com/SyKszzzzz/TOSC](https://github.com/SyKszzzzz/TOSC)  
**领域**: 3D视觉  
**关键词**: 任务导向形状补全, 灵巧抓取, 点云补全, 流匹配, 基础模型

## 一句话总结

提出任务导向形状补全（TOSC）这一新任务，仅补全与操控任务相关的接触区域（而非整个物体），利用预训练基础模型生成候选形状、3D 判别自编码器筛选最优形状、FlowGrasp 流匹配模型生成灵巧抓取，在抓取位移和 Chamfer 距离上分别提升 16.17% 和 55.26%。

## 研究背景与动机

### 问题定义

任务导向的灵巧抓取要求生成既稳定又适合特定下游操控任务的抓取姿态。例如，"用剪刀剪东西"需要手指放在剪刀的握把上，而"递剪刀给别人"则需要握住刀片部分。

### 已有方法的不足

**完整几何依赖**：现有方法（DexTOG、DexGraspVLA 等）假设输入是完整的物体点云，但真实世界中由于遮挡、杂乱和传感器噪声，输入往往是严重残缺的部分点云

**通用形状补全的局限性**：先补全再抓取的解耦方案存在根本问题——通用补全方法对整个物体进行形状恢复，但由于遮挡导致的歧义性，既可能补全错误，也可能把不相关区域的错误传播到接触区域

**补全与任务脱耦**：通用补全不知道下游任务是什么，因此无法针对任务相关的接触区域优先保证精度

### 核心动机

**关键洞察**：用于抓取的形状补全应当被下游操控任务**显式引导**。我们不需要补全整个物体的完美形状，只需要保证与抓取任务相关的**接触区域**的几何精度即可，可以容忍不相关区域的瑕疵。

例如，要"用杯子喝水"，关键是补全杯子把手的几何形状，杯底的形状是否完美并不重要。这就是**任务导向形状补全（TOSC）**的核心思想。

## 方法详解

### 整体框架

方法分三个阶段：
1. **TOSC 候选生成**：利用预训练基础模型（ControlNet + 3D 生成模型）生成多个任务导向的形状补全候选
2. **TOSC 选择与恢复**：通过 3D 判别自编码器（DAE）评估候选的可信度并恢复最优形状
3. **FlowGrasp**：条件流匹配模型生成任务导向的灵巧抓取

### 关键设计

#### 1. **TOSC 候选生成**：利用基础模型的零样本能力

**功能**：从部分点云出发，利用 ControlNet 和 3D 生成模型生成多个可能的补全形状。

**核心思路**：
1. 将输入点云 $P_{in}$ 渲染为深度图 $I_{depth}$（使用 HPR 算法选择最优视角）
2. 将深度图作为 ControlNet 的条件，使用物体类别作为提示，合成多张 RGB 图像
3. 使用不同的控制强度 $\lambda$ 生成多个 RGB 变体——大 $\lambda$ 严格遵循深度图但可能保留残缺伪影，小 $\lambda$ 则更自由但可能偏离输入
4. 使用 3D 形状生成网络（Hunyuan3D-DiT-v2-mini-Fast）从每张 RGB 图像生成 3D 网格
5. 对生成的 3D 形状和输入点云，使用 SAM 分割 + GPT-4o 检测任务相关区域
6. 通过 ICP + 任务感知对齐将生成形状与输入点云融合

对齐优化：
$$\underset{k,tr}{\text{argmin}}[\text{CD}(P_{in}, tr(kP_{gen})) + w_{task} \cdot \text{CD}(P_{in}^{task}, tr(kP_{gen}^{task}))]$$

**设计动机**：利用基础模型（ControlNet、Hunyuan3D、GPT-4o、SAM）的零样本能力，不需要任何任务特定的训练数据即可为开放世界物体生成合理的形状补全候选。多种控制强度可覆盖不同的补全可能性。

#### 2. **3D 判别自编码器（DAE）**：评估可信度并恢复全局几何

**功能**：从多个候选中选出最可信的形状，并从全局角度修复其几何。

**核心思路**：

**训练数据生成**：从 6 个数据集（72,524 个物体）构建正样本（可信形状），通过删除任务相关段、添加噪声、扰动局部 patch 生成负样本（不可信形状）。

**网络结构**：
- 编码器 $\varepsilon$：$N_{encoder}$ 个 Transformer 块，将点云 token 化（FPS + KNN 分 patch → PointNet 编码）
- 训练时随机 mask 部分 token（对正样本不 mask 任务相关段）
- 输出隐变量 $l_{can}$，估计分布 $\mathcal{N}(\mu, \sigma)$

**可信度评估**：通过 KL 散度判别输入形状是否可信：
- 正样本分布优化到 $\mathcal{N}(0,1)$
- 负样本分布优化到 $\mathcal{N}(1,1)$
- 推理时的可信度得分：

$$s_{can} = \text{Sigmoid}[-\mathcal{D}_{KL}(d_{can}\|\mathcal{N}(0,1)) + \mathcal{D}_{KL}(d_{can}\|\mathcal{N}(1,1))]$$

**设计动机**：基础模型生成的形状可能存在幻觉（ControlNet 的 RGB 图像不准确）或几何错误（3D 生成模型的重建不精确），需要一个判别机制来筛选最可信的候选。同时，通过 masked autoencoder 的重建能力，可以从全局角度修复局部缺陷。

#### 3. **FlowGrasp**：约束感知的条件流匹配模型

**功能**：从恢复后的 3D 形状生成满足几何和语义约束的灵巧抓取。

**核心思路**：在标准流匹配框架中，对预测速度进行单步梯度修正以隐式强制约束：

$$u_t^*(x_t) = u_t(x_t) - \alpha(t) \nabla\left(\sum_i w_{con}^i g_i(x_t)\right)$$

其中 $g_i$ 编码几何或语义约束，$\alpha(t)$ 是时间衰减因子。

训练时直接回归修正后的速度目标：

$$\mathcal{L}_{CFM}(\theta) = \mathbb{E}_{x_0,x_1,t,l_{con}} \|v_t^\theta(x_t|l_{con}) - u_t^*(x_t)\|^2$$

条件向量 $l_{con}$ 由 PointNet++ 特征（3D 形状）和 CLIP 嵌入（任务语言描述）拼接而成。

**设计动机**：传统方法通过加权惩罚损失或推理时梯度调整来强制约束，前者破坏概率模型的似然解释，后者引入推理开销。FlowGrasp 将约束直接融入训练目标，既不需要额外损失项，也不增加推理开销。

### 损失函数 / 训练策略

3D DAE 总损失：$L = L_{pos}^{KL} + L_{neg}^{KL} + L_{recon} + L_{mask}$

- $L_{pos}^{KL}$：正样本分布 → $\mathcal{N}(0,1)$
- $L_{neg}^{KL}$：负样本分布 → $\mathcal{N}(1,1)$
- $L_{recon}$：Chamfer 距离重建损失
- $L_{mask}$：mask 前后特征一致性（MSE）

FlowGrasp：标准 CFM 损失 + 单步梯度修正。

训练：DAE 300 epochs，FlowGrasp 350 epochs，单张 RTX 4090。

## 实验关键数据

### 主实验

**任务导向灵巧抓取（OakInk-PartialPC 数据集）**：

| 方法 | 穿透体积↓ | 穿透深度↓ | 抓取位移Mean↓ | 接触比↑ | P-FID↓ | LLM评分↑ |
|------|----------|----------|-------------|--------|--------|---------|
| GraspCVAE | 16.84 | 0.141 | 3.92 | 94.74% | 39.03 | 55.0 |
| SceneDiffuser | 6.52 | 0.090 | 3.81 | 95.62% | 29.38 | 61.7 |
| DexGYSGrasp | 7.16 | 0.096 | 3.76 | 97.20% | 25.98 | 68.3 |
| **Ours** | 6.87 | 0.090 | **3.11** | **98.30%** | **21.60** | **88.3** |

**任务导向形状补全**：

| 方法 | CD-ℓ₂×10⁻⁴↓ | F-Score@1↑ | DCD↓ |
|------|-------------|-----------|------|
| PointAttn | 4.58 | 0.512 | 0.698 |
| SVDFormer | 3.71 | 0.643 | 0.603 |
| SymmCompletion | 3.94 | 0.618 | 0.611 |
| **Ours** | **1.66** | **0.860** | **0.488** |

### 消融实验

| 配置 | 抓取位移Mean↓ | P-FID↓ | LLM评分↑ | SC↑ | PP↑ | IS↑ |
|------|-------------|--------|---------|-----|-----|-----|
| w/o TCG (候选生成) | 3.50 | 22.34 | 71.7 | 2.63 | 0.81 | 1.72 |
| w/o TSR (选择恢复) | 3.35 | 22.96 | 75.0 | 2.36 | 2.63 | 2.72 |
| w/o TOSC (通用补全替代) | 3.51 | 23.83 | 66.7 | 2.18 | 1.36 | 2.18 |
| w/o token masking | 3.21 | 22.53 | 78.3 | 3.09 | 3.18 | 3.36 |
| w/o gradient guidance | 3.43 | 24.01 | 83.3 | 3.72 | 2.18 | 3.63 |
| **Full method** | **3.11** | **21.60** | **88.3** | **4.38** | **3.84** | **3.80** |

### 关键发现

1. **TOSC 远优于通用补全**：CD 从 3.71 降至 1.66（↓55.26%），证实任务导向补全的核心价值
2. **抓取稳定性显著提升**：Grasp Displacement 从 3.76 降至 3.11（↓16.17%）
3. **零样本泛化能力强**：在 9 个未见类别和新语言指令上仍保持最优性能
4. **LLM 评分大幅领先**：88.3 vs 68.3（次优），证明生成的抓取在语义一致性上远超基线
5. **每个组件都不可或缺**：TCG 提供零样本能力，TSR 修复幻觉，梯度引导确保约束满足

## 亮点与洞察

1. **问题定义的创新**：TOSC 将"补全整个物体"重新定义为"补全任务相关的接触区域"，这是一个很有洞察力的范式转变
2. **基础模型链式组合**：ControlNet → Hunyuan3D → SAM → GPT-4o 的管线巧妙利用了多个基础模型的互补能力
3. **判别式 + 生成式结合**：3D DAE 同时具备判别（可信度评分）和生成（形状恢复）能力
4. **FlowGrasp 的约束集成方式**：将约束融入训练目标而非推理过程，避免了推理时耗时的梯度优化
5. **人类感知评分**：引入用户研究（SC、PP、IS 三个维度），全面评估抓取质量

## 局限与展望

1. **基础模型依赖性强**：ControlNet 和 Hunyuan3D 的质量直接影响候选生成，基础模型的换代可能显著改变性能
2. **管线较长**：从点云到最终抓取需经过渲染 → RGB 生成 → 3D 重建 → 分割 → 对齐 → DAE → FlowGrasp 多个步骤，推理延迟可能较高
3. **GPT-4o 的使用**：用于检测任务相关区域，增加了 API 调用成本和延迟
4. **DAE 训练数据的局限性**：负样本通过人为破坏生成，分布可能与实际基础模型错误不一致
5. **未在物理机器人上验证**：所有实验在模拟环境中进行

## 相关工作与启发

- 与 DexGraspVLA 的区别：后者使用大规模监督数据训练视觉-语言-动作模型，而 TOSC 利用基础模型零样本能力，数据需求更低
- Point-MAE 的 masked autoencoder 思路被扩展加入了判别能力
- ControlNet 的多控制强度变体策略为处理歧义提供了一种实用方案

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — TOSC 任务定义新颖且有说服力，FlowGrasp 的约束集成方式巧妙
- **实验充分度**: ⭐⭐⭐⭐ — 评估指标全面（物理 + 语义 + 人类），但缺少真实机器人实验
- **写作质量**: ⭐⭐⭐⭐ — 方法描述清晰，图示丰富
- **价值**: ⭐⭐⭐⭐⭐ — 为部分观测下的灵巧抓取提供了范式级的解决思路

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Rethinking Multimodal Point Cloud Completion: A Completion-by-Correction Perspective](rethinking_multimodal_point_cloud_completion_a_completion-by-correction_perspect.md)
- [\[AAAI 2026\] DAPointMamba: Domain Adaptive Point Mamba for Point Cloud Completion](dapointmamba_domain_adaptive_point_mamba_for_point_cloud_completion.md)
- [\[AAAI 2026\] Open-World 3D Scene Graph Generation for Retrieval-Augmented Reasoning](open-world_3d_scene_graph_generation_for_retrieval-augmented_reasoning.md)
- [\[AAAI 2026\] Simba: Towards High-Fidelity and Geometrically-Consistent Point Cloud Completion via Transformation Diffusion](simba_towards_high-fidelity_and_geometrically-consistent_point_cloud_completion_.md)
- [\[ICML 2026\] DynaTok: Token-Based 4D Reconstruction from Partial Point Clouds](../../ICML2026/3d_vision/dynatok_token-based_4d_reconstruction_from_partial_point_clouds.md)

</div>

<!-- RELATED:END -->
