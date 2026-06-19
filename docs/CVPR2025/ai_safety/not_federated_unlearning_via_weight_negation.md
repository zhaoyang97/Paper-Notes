---
title: >-
  [论文解读] NoT: Federated Unlearning via Weight Negation
description: >-
  [CVPR 2025][AI安全][联邦遗忘] 提出 NoT 算法，通过对全局模型特定层的权重乘以 -1（取反）来破坏层间协同适应从而实现遗忘，再用保留数据微调恢复性能，无需额外存储或访问目标数据，在 CIFAR-10/100、Caltech-101 上以最低通信/计算开销显著优于七种基线方法。 领域现状：联邦学习（FL）通…
tags:
  - "CVPR 2025"
  - "AI安全"
  - "联邦遗忘"
  - "权重取反"
  - "层间协同适应"
  - "隐私保护"
  - "模型扰动"
---

# NoT: Federated Unlearning via Weight Negation

**会议**: CVPR 2025  
**arXiv**: [2503.05657](https://arxiv.org/abs/2503.05657)  
**代码**: 无  
**领域**: AI Safety / 联邦遗忘  
**关键词**: 联邦遗忘, 权重取反, 层间协同适应, 隐私保护, 模型扰动

## 一句话总结
提出 NoT 算法，通过对全局模型特定层的权重乘以 -1（取反）来破坏层间协同适应从而实现遗忘，再用保留数据微调恢复性能，无需额外存储或访问目标数据，在 CIFAR-10/100、Caltech-101 上以最低通信/计算开销显著优于七种基线方法。

## 研究背景与动机

**领域现状**：联邦学习（FL）通过分布式训练保护数据隐私，但面临日益增长的数据删除需求——GDPR 等法规赋予用户"被遗忘权"，要求模型能删除特定参与者的数据贡献。联邦遗忘（FU）正是为解决这一需求而生。

**现有痛点**：现有 FU 方法存在明显缺陷。第一类方法（如 FedEraser、FUKD）需存储历史模型更新，带来额外存储开销和隐私泄露风险；第二类方法（如 PGD、MoDE、FCU）通过梯度修改实现遗忘，但需要目标客户端参与且计算开销高。更关键的是，许多方法依赖对目标数据的直接访问，而目标数据可能已不可用。

**核心矛盾**：有效遗忘需要对模型参数进行足够大的扰动以"忘记"目标数据，但过大的扰动需要大量微调才能恢复性能——即遗忘彻底性与恢复效率之间的 trade-off。

**本文目标**：设计一种既不需要额外存储、也不需要访问目标数据的遗忘算法，同时保证遗忘有效性和快速恢复能力。

**切入角度**：作者从 inter-layer co-adaptation（层间协同适应）的角度出发，观察到神经网络的性能高度依赖层与层之间的参数协调关系。如果打破这种协调，模型将丧失已学知识。

**核心 idea**：用权重取反（乘以 -1）来最大化破坏层间协同适应，同时理论证明取反后的模型保持"层级最优性"（layer-wise optimality），使得后续微调可以快速恢复性能。

## 方法详解

### 整体框架
NoT 遵循"扰动 + 微调"两步范式。当某客户端发起遗忘请求后：(1) 服务器对全局模型指定层执行权重取反 $\theta'_\ell = -\theta^*_\ell$；(2) 对取反后的模型使用保留数据进行联邦微调，恢复关键知识。整个过程不需要存储历史更新，也不需要访问目标数据。

### 关键设计

1. **权重取反扰动（Weight Negation）**:

    - 功能：通过将指定层参数乘以 -1 来破坏模型的层间协同适应
    - 核心思路：对选定层集合 $\mathscr{L}_{\text{neg}}$ 中的每一层 $\ell$，执行 $\theta'_\ell = -\theta^*_\ell$，其余层保持不变。取反后经过 ReLU 激活的层输出发生根本性改变——原本正的预激活值变负被裁剪为 0，原本负的变正被保留，等价于"翻转"了特征选择模式
    - 设计动机：Theorem 2 证明在温和假设下，权重取反在所有等范数扰动中对激活值的改变最大（即 $\mathbb{E}\|\sigma(Y) - \sigma(-Y)\|^2$ 对任意其他扰动都是上界），因此是最强的扰动方式

2. **理论遗忘框架（Unlearning via Loss Gap）**:

    - 功能：为"扰动 + 微调"范式提供理论基础，量化遗忘速度
    - 核心思路：定义 loss gap $\delta(\theta) = |\mathcal{L}_{D_r}(\theta) - \mathcal{L}_{D_u}(\theta)|$ 衡量遗忘程度。Theorem 1 给出了达到目标遗忘所需的最少微调时间下界，说明初始损失越大、Hessian 谱越大，遗忘越快。自然遗忘（不扰动直接微调）因初始损失接近最优且梯度小而极其缓慢
    - 设计动机：需要强扰动来加速遗忘过程，而不是坐等自然遗忘

3. **弹性扰动理论（Resilient Perturbation）**:

    - 功能：保证取反后的模型可以快速恢复
    - 核心思路：弹性扰动需满足两个条件——(C2a) Jacobian 控制：Theorem 3 证明对 $\ell > \mathscr{L}_{\text{neg}}$ 的层，梯度 Jacobian 的 Wasserstein 距离受 $TV(Y_-; -Y_-)$ 控制，在宽网络极限下趋近于 0；(C2b) 层级最优性保持（LWOP）：Theorem 4 证明若 $\mathscr{L}_{\text{neg}}$ 是计算图偏序集中的反链且不含最大元素，取反保持 LWOP
    - 设计动机：随机重置参数也是强扰动，但缺乏弹性，恢复极慢；权重取反因保留了参数的绝对值信息和 Jacobian 谱分布，恢复速度快得多

### 损失函数 / 训练策略
取反后的微调使用标准 FedAvg 算法在保留数据 $D_r$ 上进行。对于客户端级遗忘，目标客户端不参与微调；对于类级或实例级遗忘，目标客户端使用其保留数据参与微调。默认只取反第一层权重——这足以改变底层特征表示，通过微调传播导致深层参数大幅更新。

## 实验关键数据

### 主实验
在 CIFAR-10、CIFAR-100、Caltech-101 上与 7 种基线对比，IID 设置、10 个客户端、1 个客户端请求遗忘：

| 数据集 / 模型 | 方法 | Retain Acc (Δ↓) | Forget Acc (Δ↓) | Test Acc (Δ↓) | MIA (Δ↓) | Avg Gap ↓ |
|---|---|---|---|---|---|---|
| CIFAR-10 / CNN | Retrain (oracle) | 91.66 (0.00) | 83.05 (0.00) | 82.32 (0.00) | 50.23 (0.00) | 0.00 |
| | FT | 92.48 (0.82) | 85.56 (2.51) | 82.36 (0.04) | 50.90 (0.67) | 1.01 |
| | MoDE | 92.56 (0.90) | 85.25 (2.20) | 82.31 (0.01) | 50.70 (0.47) | 0.90 |
| | **NoT** | **91.69 (0.03)** | **83.86 (0.81)** | **82.65 (0.33)** | **50.23 (0.00)** | **0.29** |
| CIFAR-100 / CNN | Retrain | 72.32 (0.00) | 53.31 (0.00) | 54.28 (0.00) | 49.70 (0.00) | 0.00 |
| | FCU | 73.40 (1.08) | 56.68 (3.37) | 55.37 (1.09) | 50.03 (0.33) | 1.47 |
| | **NoT** | **72.25 (0.07)** | **55.22 (1.91)** | **55.23 (0.95)** | **49.63 (0.07)** | **0.75** |
| Caltech-101 / ViT | Retrain | 99.73 (0.00) | 48.29 (0.00) | 48.02 (0.00) | 49.67 (0.00) | 0.00 |
| | FT | 99.96 (0.23) | 94.23 (45.94) | 48.75 (0.73) | 73.80 (24.13) | 17.76 |
| | **NoT** | **99.70 (0.03)** | **50.81 (2.52)** | **47.83 (0.19)** | **50.07 (0.40)** | **0.79** |

### Non-IID 实验
Caltech-101 / ViT，Dirichlet β=0.1：

| 方法 | Retain Δ↓ | Forget Δ↓ | Test Δ↓ | MIA Δ↓ | Avg Gap ↓ |
|---|---|---|---|---|---|
| FT | 1.38 | 36.24 | 2.19 | 15.67 | 13.87 |
| PGD | 1.30 | 32.14 | 1.69 | 13.53 | 12.16 |
| **NoT** | **0.71** | **1.40** | **0.37** | **0.97** | **0.86** |

### 消融实验

| 扰动方式 | Avg Gap ↓ | 说明 |
|---|---|---|
| Weight Negation (NoT) | 0.29 | 第一层取反，最优 |
| Gaussian Noise | 1.55 | 高斯噪声扰动，恢复慢 |
| Random Reinit | 3.23 | 随机重初始化，恢复最慢 |
| Negation (all layers) | 0.84 | 所有层取反，梯度消失影响恢复 |
| Negation (last layer) | 1.12 | 只取反最后层，遗忘不充分 |

### 关键发现
- **取反第一层效果最佳**：第一层控制底层特征提取，取反后通过微调级联影响深层参数更新，实现有效遗忘
- **通信/计算开销最低**：NoT 的通信量约为 Retrain 的 50-60%，计算量约为 50-55%
- **在 ViT 上优势尤为突出**：FT 几乎完全无法遗忘（Forget Acc 94.23% vs Retrain 48.29%），而 NoT 达到 50.81%
- **支持三种遗忘粒度**：客户端级、类级、实例级遗忘均有效，无需修改算法

## 亮点与洞察
- **极简设计，深厚理论**：权重乘以 -1 看似简单粗暴，但背后有完整的理论体系——Theorem 2 证明它是最强扰动，Theorem 3-4 证明它保持弹性。这种"简单方法 + 严谨证明"的风格值得学习
- **零额外需求**：不需要存储历史更新，不需要访问数据，不需要目标客户端参与。对比 FedEraser 需要存储所有轮次更新、PGD 需要目标客户端执行梯度反转，NoT 的实际部署门槛极低
- **层间协同适应的利用**：把通常被视为黑箱的层间依赖关系转化为可操控的遗忘工具，这个视角可以迁移到模型安全、后门清除等场景

## 局限与展望
- 理论 Theorem 4 不覆盖 ReLU 激活函数的 LWOP 证明（ReLU 既非 odd 也非 even），只有猜想层面的"近似 LWOP"
- 在 ResNet-18/CIFAR-100 上 Avg Gap (4.73) 高于 FCU (2.51)，复杂模型+细粒度类别场景下取反恢复可能较慢
- 只考虑了 IID 和简单 non-IID 分布，对极端异构数据分布的表现未知
- 未讨论对模型后门攻击的系统性防御效果

## 相关工作与启发
- **vs FedEraser**：需存储历史模型状态，存储开销大且有隐私风险；NoT 无需存储
- **vs PGD**：对目标数据做梯度反转再约束更新幅度，需目标客户端参与；NoT 只需服务端操作
- **vs MoDE**：使用随机初始化模型做退化，概念类似但缺乏理论保证；NoT 有严格理论支撑

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 权重取反的idea极其简洁但有效，理论分析全面
- 实验充分度: ⭐⭐⭐⭐ 三数据集三架构多设置，但缺少更大规模实验
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，实验组织有条理
- 价值: ⭐⭐⭐⭐ 实用价值高，部署简单，但适用范围受限于联邦场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Towards Source-Free Machine Unlearning](towards_source-free_machine_unlearning.md)
- [\[ICLR 2026\] Dataless Weight Disentanglement in Task Arithmetic via Kronecker-Factored Approximate Curvature](../../ICLR2026/ai_safety/dataless_weight_disentanglement_in_task_arithmetic_via_kronecker-factored_approx.md)
- [\[CVPR 2026\] Computation and Communication Efficient Federated Unlearning via On-server Gradient Conflict Mitigation and Expression](../../CVPR2026/ai_safety/computation_and_communication_efficient_federated_unlearning_via_on-server_gradi.md)
- [\[NeurIPS 2025\] Rewind-to-Delete: Certified Machine Unlearning for Nonconvex Functions](../../NeurIPS2025/ai_safety/rewind-to-delete_certified_machine_unlearning_for_nonconvex_functions.md)
- [\[NeurIPS 2025\] Not All Deepfakes Are Created Equal: Triaging Audio Forgeries for Robust Deepfake Singer Identification](../../NeurIPS2025/ai_safety/not_all_deepfakes_are_created_equal_triaging_audio_forgeries_for_robust_deepfake.md)

</div>

<!-- RELATED:END -->
