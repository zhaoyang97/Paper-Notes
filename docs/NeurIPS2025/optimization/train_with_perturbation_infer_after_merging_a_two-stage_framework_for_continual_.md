---
title: >-
  [论文解读] Train with Perturbation, Infer after Merging: A Two-Stage Framework for Continual Learning
description: >-
  [NeurIPS 2025][优化/理论][持续学习] 提出Perturb-and-Merge (P&M)框架，将模型合并机制引入持续学习范式：训练时沿任务向量方向添加随机扰动以平滑损失面，推理时通过闭式最优系数对历史模型和当前任务模型做凸组合合并，结合LoRA实现内存高效的SOTA持续学习性能。 持续学习（CL）旨在让模型…
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "持续学习"
  - "模型合并"
  - "任务向量"
  - "参数扰动"
  - "LoRA"
---

# Train with Perturbation, Infer after Merging: A Two-Stage Framework for Continual Learning

**会议**: NeurIPS 2025  
**arXiv**: [2505.22389](https://arxiv.org/abs/2505.22389)  
**代码**: [github.com/qhmiao/P-M-for-Continual-Learning](https://github.com/qhmiao/P-M-for-Continual-Learning)  
**领域**: 模型压缩  
**关键词**: 持续学习, 模型合并, 任务向量, 参数扰动, LoRA

## 一句话总结
提出Perturb-and-Merge (P&M)框架，将模型合并机制引入持续学习范式：训练时沿任务向量方向添加随机扰动以平滑损失面，推理时通过闭式最优系数对历史模型和当前任务模型做凸组合合并，结合LoRA实现内存高效的SOTA持续学习性能。

## 研究背景与动机

持续学习（CL）旨在让模型从任务序列中不断吸收新知识同时保留旧知识。现有方法（正则化、记忆回放、架构扩展等）虽取得进展，但几乎所有方法在完成任务 $t$ 的训练后，直接用 $\theta_t^*$ 作为所有任务1到 $t$ 的推理参数——而这个参数主要针对任务 $t$ 进行了优化，对历史任务的保持缺乏显式保障，容易导致灾难性遗忘。

与此同时，模型合并（Model Merging）领域展示了一种有趣的能力：从同一预训练模型出发、在不同任务上独立训练的多个模型，通过参数插值或更精细的方法合并为一个统一模型，可以在多任务上都保持良好性能。

这两个领域虽然流程不同，但共享同一个核心目标——学习一个在多任务上表现良好的单一模型。CL的优势在于所有任务沿共享优化轨迹训练，参数更可能处于联合最优附近；模型合并的优势在于提供了稳定的后训练集成机制。而现有CL方法完全没有利用模型合并的这一优势。

本文的核心idea是统一二者：**训练阶段**沿用CL的序列训练获得 $\theta_t^*$，**推理阶段**不直接使用 $\theta_t^*$，而是将其与历史推理参数 $\hat{\theta}_{t-1}$ 做最优凸组合 $\hat{\theta}_t = \hat{\theta}_{t-1} + \alpha_t \Delta\theta_t^*$。进一步发现，合并导致的性能退化可以通过训练时沿任务向量方向的参数扰动来缓解，且该扰动作为正则项的无偏随机近似，不增加任何额外前向/反向传播开销。

## 方法详解

### 整体框架
P&M分为两个阶段：(1) **Train with Perturbation**：在每个任务训练时，以概率随机在参数上添加沿任务向量方向的扰动，作为Hessian二次型正则项的随机近似；(2) **Infer after Merging**：训练完成后，计算Fisher信息矩阵，求解闭式最优合并系数 $\alpha_t^*$，将 $\hat{\theta}_{t-1}$ 和 $\theta_t^*$ 做凸组合得到推理参数。结合LoRA降低内存开销。

### 关键设计

1. **Infer after Merging（推理时合并）**: 

    - 功能：每完成一个任务的训练后，不直接用训练后的参数做推理，而是与历史推理参数做加权合并
    - 核心思路：凸组合 $\hat{\theta}_t = (1 - \alpha_t)\hat{\theta}_{t-1} + \alpha_t\theta_t^*$，等价于缩放任务向量 $\hat{\theta}_t = \hat{\theta}_{t-1} + \alpha_t \Delta\theta_t^*$。为求最优 $\alpha_t$，分析合并模型在所有任务上的总损失增加量，通过二阶Taylor展开得到：
    $\alpha_t^* = -\frac{\sum_{i=1}^{t}(\hat{\theta}_{t-1} - \theta_i^*)^\top \mathbf{H}_i(\theta_i^*) \Delta\theta_t^*}{\sum_{i=1}^{t}\Delta\theta_t^{*\top} \mathbf{H}_i(\theta_i^*) \Delta\theta_t^*}$
   Hessian矩阵用对角经验Fisher信息矩阵近似。
    - 设计动机：缩放任务向量不会破坏历史参数 $\hat{\theta}_{t-1}$，只调整新任务的贡献程度来减少遗忘。闭式解避免了超参数搜索。

2. **Train with Perturbation（训练时扰动）**: 

    - 功能：通过训练时的参数扰动减少合并带来的性能退化
    - 核心思路：合并退化的上界包含 $\Delta\theta_t^{*\top} \mathbf{H}_t \Delta\theta_t^*$ 项，可作为训练正则项。利用二阶对称有限差分近似：
    $\Delta\theta_t^\top \mathbf{H}_t \Delta\theta_t \approx \frac{1}{\epsilon^2}(\mathcal{L}_t(\theta_t + \epsilon\Delta\theta_t) + \mathcal{L}_t(\theta_t - \epsilon\Delta\theta_t) - 2\mathcal{L}_t(\theta_t))$
   但直接计算需要3倍前向传播。因此提出随机近似：每个训练step以概率 $p_0, p_+, p_-$ 分别采样原始参数、正扰动参数或负扰动参数之一来计算损失，期望匹配完整正则化损失：$\mathbb{E}[\tilde{\mathcal{L}}_t(\theta)] = \mathcal{L}_t(\theta)$
    - 设计动机：扰动方向沿任务向量，相当于让模型在训练时就预先适应合并操作可能带来的参数偏移，扩大损失面的平坦区域，减少参数冲突。关键优势是零额外计算开销——每步只需一次前向传播。

3. **LoRA-P&M（结合LoRA的高效实现）**: 

    - 功能：通过LoRA低秩分解降低每个任务的参数存储和Fisher矩阵计算成本
    - 线性层更新为 $\mathbf{W}_t = \mathbf{W}_{t-1} + \mathbf{A}_t\mathbf{B}_t$，仅更新rank=10的LoRA模块（应用于key和value投影）
    - 扰动操作变为：$\theta_t + \epsilon \cdot \text{LoRA}_t$（即在LoRA参数上缩放 $1+\epsilon$）
    - 设计动机：需要存储所有历史任务的 $\theta_i^*$ 和Fisher矩阵来计算 $\alpha_t^*$，LoRA使得存储成本从全参数降到低秩参数

### 损失函数 / 训练策略
基础损失为交叉熵。扰动参数 $\epsilon = 0.5$，采样概率 $p_0 = p_+ = p_- = 1/3$。使用AdamW优化器，LoRA学习率1e-3，分类头学习率1e-2，batch size 256。每任务训练10 epochs（DomainNet为5 epochs）。使用ViT-B/16（ImageNet-21K预训练+ImageNet-1K微调）作为骨干网络。

## 实验关键数据

### 主实验

**ImageNet-R（10 tasks）CL方法对比**

| 方法 | Acc↑ | AAA↑ |
|------|------|------|
| Full Fine-Tuning | 60.57 | 72.31 |
| L2P | 71.26 | 76.13 |
| CODA-Prompt | 74.05 | 78.14 |
| InfLoRA | 74.75 | 80.67 |
| SD-LoRA | 77.34 | 82.04 |
| LoRA（基线） | 65.72 | 76.14 |
| **LoRA-P&M** | **79.95** | **85.29** |

**六个benchmark上的模型合并方法对比（Acc）**

| 方法 | INR-10 | INR-20 | INA-10 | DN*-5 | C100-10 | CUB-10 |
|------|--------|--------|--------|-------|---------|--------|
| LoRA | 65.72 | 56.35 | 44.41 | 71.81 | 72.58 | 64.82 |
| Model Averaging | 76.90 | 74.64 | 54.54 | 81.84 | 87.52 | 74.87 |
| DARE | 75.09 | 66.03 | 55.87 | 80.58 | 87.28 | 76.57 |
| CoMA | 79.34 | 75.60 | 53.24 | 83.98 | 86.95 | 74.65 |
| **P&M** | **79.95** | **76.37** | **56.57** | **84.71** | **88.45** | **78.29** |

### 消融实验

**P&M各组件消融（六个benchmark上的Acc）**

| 配置 | INR-10 | INR-20 | INA-10 | DN*-5 | C100-10 | CUB-10 |
|------|--------|--------|--------|-------|---------|--------|
| LoRA | 65.72 | 56.35 | 44.41 | 71.81 | 72.58 | 64.82 |
| LoRA-M（仅合并） | 78.35 | 74.26 | 56.16 | 81.28 | 86.57 | 74.98 |
| LoRA-M+高斯噪声 | 78.48 | 74.13 | 49.51 | 83.00 | 85.83 | 74.09 |
| **LoRA-P&M（完整）** | **79.95** | **76.37** | **56.57** | **84.71** | **88.45** | **78.29** |

**全局 vs 逐模块合并系数（ImageNet-R）**

| 策略 | 5 tasks | 10 tasks | 20 tasks |
|------|---------|----------|----------|
| 逐模块 $\alpha$ | 80.53 | 76.82 | 74.68 |
| 全局 $\alpha$（本文） | 80.88 | 78.48 | 74.13 |

### 关键发现
- 仅添加"Infer after Merging"就将LoRA从65.72提升到78.35（INR-10），证明了推理时合并的巨大价值
- 任务向量方向的扰动比随机高斯噪声效果好得多（CUB-10: 78.29 vs 74.09），证明扰动方向的选择至关重要
- P&M主要靠减少遗忘提升性能，对可塑性的影响微乎其微——类似于"只收益不付代价"的效果
- 全局合并系数与逐模块策略性能相当，且学习到的逐模块系数值彼此高度相似，支持了使用单一全局系数的决定
- 损失面可视化显示：凸组合的路径一致位于低损失区域，且扰动训练扩大了低损失盆地的平坦度和宽度

## 亮点与洞察
- 方法设计中"理论推导→实际近似→高效实现"的链条非常完整：从分析合并退化→推导Hessian正则项→有限差分近似→随机扰动实现，每一步都有清晰的理论和实践支撑。最终实现不增加任何额外计算的正则化效果，是罕见的"免费午餐"。
- 将CL和模型合并两个独立研究方向统一的视角具有启发性：CL提供共享优化轨迹使参数更适合合并，模型合并提供显式的知识保留机制，二者互补而非替代。

## 局限与展望
- Fisher信息矩阵使用对角近似，无法完全捕捉损失面的真实曲率结构，更精确但高效的曲率近似值得探索
- 需要存储每个历史任务的LoRA参数和Fisher矩阵，任务数量很大时仍有内存压力
- 仅在分类任务上验证，生成、检测等其他任务类型有待探索
- 凸组合的假设限制了合并的灵活性，非凸组合（如task arithmetic）可能在某些场景下更优

## 相关工作与启发
- **vs SD-LoRA**: SD-LoRA是之前的SOTA CL方法，P&M在INR-10上超过2.61%，在DomainNet上超过1.89%，且P&M的方法更简洁（无需特殊LoRA结构设计）
- **vs CoMA/CoFIMA**: 这两种模型合并方法在CL设定下被P&M全面超越，说明CL的共享优化轨迹确实有助于合并质量
- **vs EWC**: EWC也使用Fisher信息矩阵，但用于正则化训练损失。P&M更巧妙地将其用于求解合并系数的闭式解

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 将模型合并引入CL推理阶段的想法新颖且自然，扰动训练作为Hessian正则的随机近似理论优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 5个数据集、不同任务数、与CL和模型合并两类方法全面对比、丰富的消融和分析
- 写作质量: ⭐⭐⭐⭐⭐ 从理论到方法到实验的逻辑链严谨清晰，Loss landscape可视化的四个观察点非常有说服力
- 价值: ⭐⭐⭐⭐⭐ 方法简单有效、理论完备、计算开销极低，为CL领域提供了新范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Online Two-Stage Submodular Maximization](online_two-stage_submodular_maximization.md)
- [\[NeurIPS 2025\] MergeBench: A Benchmark for Merging Domain-Specialized LLMs](mergebench_a_benchmark_for_merging_domain-specialized_llms.md)
- [\[NeurIPS 2025\] Gradient Descent as Loss Landscape Navigation: a Normative Framework for Deriving Learning Rules](gradient_descent_as_loss_landscape_navigation_a_normative_framework_for_deriving.md)
- [\[CVPR 2026\] Defending Unauthorized Model Merging via Dual-Stage Weight Protection](../../CVPR2026/optimization/defending_unauthorized_model_merging_via_dual-stage_weight_protection.md)
- [\[ICCV 2025\] Federated Continual Instruction Tuning](../../ICCV2025/optimization/federated_continual_instruction_tuning.md)

</div>

<!-- RELATED:END -->
