---
title: >-
  [论文解读] MUNBa: Machine Unlearning via Nash Bargaining
description: >-
  [ICCV 2025][LLM安全][机器遗忘] 将机器遗忘（Machine Unlearning）建模为双玩家合作博弈问题，利用 Nash 讨价还价理论推导闭式解来同时解决遗忘目标与保留目标之间的梯度冲突和梯度支配问题，在分类和生成任务上实现遗忘与保留的最优平衡。 机器遗忘（Machine Unlearning…
tags:
  - "ICCV 2025"
  - "LLM安全"
  - "机器遗忘"
  - "Nash博弈"
  - "梯度冲突"
  - "Pareto最优"
  - "扩散模型"
  - "CLIP"
  - "多目标优化"
---

# MUNBa: Machine Unlearning via Nash Bargaining

**会议**: ICCV 2025  
**arXiv**: [2411.15537](https://arxiv.org/abs/2411.15537)  
**代码**: 未公开  
**领域**: 机器遗忘 / 图像生成安全  
**关键词**: 机器遗忘, Nash博弈, 梯度冲突, Pareto最优, 扩散模型, CLIP, 多目标优化

## 一句话总结

将机器遗忘（Machine Unlearning）建模为双玩家合作博弈问题，利用 Nash 讨价还价理论推导闭式解来同时解决遗忘目标与保留目标之间的梯度冲突和梯度支配问题，在分类和生成任务上实现遗忘与保留的最优平衡。

## 研究背景与动机

机器遗忘（Machine Unlearning, MU）旨在从已训练的模型中选择性地擦除特定数据或概念的影响，同时保持模型在其余数据上的性能。这一需求来源于多方面驱动力：

**隐私法规要求**：GDPR（欧盟通用数据保护条例）和 CCPA（加州消费者隐私法）赋予用户"被遗忘权"，企业有义务在收到删除请求后擦除用户相关数据的影响。

**安全与版权保护**：文本-图像扩散模型（如 Stable Diffusion）可能在训练数据中包含 NSFW 内容或侵犯版权的素材，需要将这些不良概念从模型中移除。

**从头重训不现实**：理想的遗忘方式是在排除待忘数据后从头重训模型，但对于当今大规模模型（如扩散模型、CLIP），重训的计算成本极高，因此需要近似遗忘算法。

**现有方法的核心问题——梯度冲突与梯度支配**：

当前 MU 方法通常将遗忘和保留形式化为两个子目标的加权和。其中保留损失在剩余数据上微调保持性能，遗忘损失在待忘数据上最大化损失使模型"忘记"。然而作者通过实证分析发现：

- **梯度冲突**：遗忘梯度与保留梯度的余弦相似度频繁为负值，说明两个目标的优化方向经常相互矛盾。
- **梯度支配**：两个梯度的范数差异悬殊，导致联合更新方向被某一个目标主导，另一个目标被忽视。

这两个问题在多目标优化（MOO）文献中已被充分研究，但在 MU 领域却长期被忽视。

## 方法详解

### 核心思想：将 MU 建模为合作博弈

MUNBa 的核心创新在于将 MU 问题重新定义为一个**双玩家合作讨价还价博弈**：

- **遗忘玩家**（Forgetting Player）：提出梯度 g_f，希望最终更新方向有利于遗忘。
- **保留玩家**（Preservation Player）：提出梯度 g_r，希望最终更新方向有利于保留模型性能。

两个玩家通过"协商"找到一个互利的联合更新方向，使得双方的整体收益最大化。

### 效用函数定义

每个玩家的效用函数定义为其梯度与最终联合更新方向的内积。保留玩家的效用 u_r 等于 g_r 与联合方向的内积，遗忘玩家的效用 u_f 等于 g_f 与联合方向的内积。效用函数衡量的是最终更新方向与各自目标方向的对齐程度。如果联合更新方向偏离某个玩家的梯度方向，该玩家的收益就会下降。

### Nash 讨价还价目标

受 Nash 讨价还价理论启发，优化目标被重写为最大化 log(u_r) + log(u_f)，约束联合更新向量在一个以原点为中心、半径为 epsilon 的球内。对数形式确保边际效用递减（效用越高时增益越少），从而自然地平衡两个目标的贡献。

### 闭式解推导

这是本文最关键的技术贡献。通过一系列定理推导，最终得到联合更新方向为 alpha_r 乘以 g_r 加上 alpha_f 乘以 g_f，其中系数 alpha 有**闭式解**：

- alpha_r = (1 / ||g_r||) * sqrt((1 - cos(phi)) / (sin^2(phi) + xi))
- alpha_f = (1 / ||g_f||) * sqrt((1 - cos(phi)) / (sin^2(phi) + xi))

其中 phi 是 g_r 与 g_f 之间的夹角，xi 是防止除零的小常数。

**闭式解的关键性质**：

1. **系数与梯度范数成反比**（正比于 1/||g||）：自动抑制梯度范数较大的目标，解决梯度支配问题。
2. **冲突自适应**：当梯度冲突严重时（cos(phi) 接近 -1），系数增大以增强各目标的权重；当梯度对齐时（cos(phi) 接近 1），系数减小。
3. **精确高效**：相比之前工作需要近似求解 alpha，本文利用 MU 恰好只有两个目标的特点推导出精确闭式解，计算开销极小。

### 退化情况处理

当两个梯度线性相关时（g_r = zeta * g_f），Gram 矩阵行列式为零。处理方式为：
- 若 zeta < 0（方向相反），向范数较小的梯度添加噪声打破线性相关。
- 若 zeta >= 0（方向一致），直接取 alpha = [0.5, 0.5]。

### 算法流程

1. 采样遗忘数据和保留数据的 mini-batch。
2. 分别计算遗忘梯度 g_f 和保留梯度 g_r。
3. 构建 Gram 矩阵 K = G^T * G，用闭式解计算系数 alpha。
4. 用联合梯度 (alpha_r * g_r + alpha_f * g_f) 更新模型参数。

### 理论保证

- **Pareto 改进**（Theorem 2.9）：在 Lipschitz 平滑条件下，适当设置学习率可使两个玩家的损失单调递减。
- **收敛性**（Theorem 2.10）：联合损失收敛到 Pareto 驻点，即任何偏离最终状态的变动都会导致至少一个目标恶化。
- **下界保证**（Lemma 2.8）：每个玩家的系数 alpha_i 被下界限制在 1/(sqrt(2)*M)，确保两个目标都不会被完全忽视。

## 实验关键数据

### 1. 分类任务（ResNet）

在 Celeb-HQ-307 上遗忘 10% 身份、CIFAR-10 上遗忘 10% 数据：

| 方法 | Celeb-HQ Avg. Gap (越低越好) | CIFAR-10 Avg. Gap (越低越好) |
|------|---------------------|---------------------|
| SalUn | 0.60 | 1.24 |
| SHs | 0.39 | 1.62 |
| **MUNBa** | **0.10** | **0.97** |

MUNBa 在两个数据集上的 Avg. Gap 均最小，与从头重训模型最接近。在 Celeb-HQ-307 上实现 0% 的遗忘数据准确率和 87.24% 的测试准确率。

### 2. CLIP 遗忘（Oxford Pets 类别遗忘）

| 方法 | 遗忘1类 Acc_Df (越低越好) | Acc_Dt (越高越好) | ImageNet Acc (越高越好) |
|------|------------------|----------|----------------|
| SHs | 0.00% | 91.41% | 37.97% |
| SalUn | 4.69% | 82.93% | 59.94% |
| **MUNBa** | 2.50% | **94.99%** | 59.36% |

MUNBa 在保持 CLIP 泛化能力（ImageNet 精度 59.36% vs 原始 60.09%）的同时实现有效遗忘，而 SHs 虽然遗忘彻底但 ImageNet 精度暴跌至 37.97%。

### 3. 扩散模型类别遗忘（Imagenette）

MUNBa 平均 FID=1.20、UA=99.94%，在 10 个类别的整体表现上优于 ESD（FID=1.49, UA=99.40%）和 SalUn（FID=1.22, UA=99.82%）。

### 4. NSFW 概念擦除（Stable Diffusion v1.4）

| 方法 | FID (越低越好) | CLIP Score (越高越好) | ASR (越低越好) |
|------|-------|-------------|-------|
| ESD | 15.76 | 30.33 | 73.24% |
| SA | 25.58 | 31.03 | 48.59% |
| SalUn | 25.06 | 28.91 | 11.27% |
| **MUNBa** | **15.92** | 30.43 | **3.52%** |

MUNBa 在对抗 UnlearnDiffAtk 攻击时表现尤为突出：攻击成功率仅 3.52%（SalUn 为 11.27%），同时保持 FID=15.92 的高生成质量（SalUn 的 FID 高达 25.06）。

## 亮点与洞察

1. **问题定义精准**：首次系统地分析并实证展示了 MU 中梯度冲突和梯度支配问题，通过余弦相似度直方图与梯度范数比等可视化手段使问题一目了然。
2. **闭式解的优雅**：利用 MU 恰好只有两个目标（遗忘与保留）的结构特点，从 Nash 讨价还价理论推导出精确闭式解，避免了一般多目标优化方法中的近似求解。系数的形式直觉清晰：与梯度范数成反比，自动平衡两个目标的贡献。
3. **通用框架**：MUNBa 不是针对特定模型或特定遗忘场景设计的，可以无缝应用于 ResNet 分类、CLIP 视觉语言模型、Stable Diffusion 扩散模型等不同架构。
4. **对抗鲁棒性出色**：在 UnlearnDiffAtk 攻击下 ASR 仅 3.52%，远低于其他方法，说明 Nash 博弈解不仅在正常场景下有效，在对抗场景下也更稳健。这是一个有意义的发现——理论上最优的平衡点在对抗攻击下也更难被突破。
5. **CLIP 遗忘后仍可迁移**：将遗忘后的 CLIP 文本编码器接入 SD，生成图像的质量保持，且不会生成遗忘类别的图像，展示了实际应用价值。

## 局限与展望

1. **计算速度**：作者承认 MUNBa 比部分基线方法更慢，因为每步都需要分别计算两个梯度然后求解系数。虽然闭式解本身计算量极小，但双倍的梯度计算增加了总体开销。
2. **遗忘的不完美性**：在某些场景下 MUNBa 仍可能失败（论文附录 8.4 提到），且所有 MU 方法（包括 MUNBa）对剩余概念/类别仍会产生一定影响。
3. **缺乏无数据场景的讨论**：当前方法假设可获得保留数据 D_r，但在实际部署中训练数据可能已不可用。
4. **潜在滥用风险**：遗忘技术可能被恶意用于擦除关键信息、偏倚决策过程或掩盖重要数据。
5. **扩展到多目标**：当前框架仅处理两个玩家（遗忘+保留），如果需要同时遗忘多个概念或引入更多约束（如公平性），则需要扩展到多玩家博弈。

## 相关工作与启发

- **SalUn (Fan et al., ICLR 2024)**：通过梯度显著性图识别与遗忘数据相关的重要参数进行选择性遗忘，是分类和生成任务上的强基线。
- **Scissorhands (Wu & Harandi, ECCV 2024)**：通过网络连接敏感度进行数据影响擦除，同一作者组之前的工作。
- **ESD (Gandikota et al., ICCV 2023)**：基于能量的方法，针对无分类器引导机制进行概念擦除。
- **Nash-MTL (Navon et al., ICML 2022)**：将多任务学习建模为讨价还价博弈，MUNBa 的直接灵感来源，但 MUNBa 利用 MU 只有两个目标的特点推导出了闭式解。
- **CAGrad (Liu et al., NeurIPS 2021)**：冲突感知梯度下降，解决梯度冲突但不处理梯度支配问题。

**启发**：Nash 博弈框架为多目标优化中的平衡问题提供了一种有理论保证的通用解法，可以推广到其他存在目标冲突的场景，如持续学习中遗忘与新知识获取的平衡、联邦学习中不同客户端目标的协调等。

## 评分
- 新颖性: 4/5 - 将博弈论引入 MU 的梯度冲突/支配问题，闭式解推导优雅
- 实验充分度: 5/5 - 覆盖分类(ResNet)、VLM(CLIP)、生成(SD)三大类模型，含对抗攻击鲁棒性评估
- 写作质量: 4/5 - 问题动机清晰，理论推导完整，可视化直观
- 价值: 4/5 - 通用性强，理论与实验均扎实，对 MU 领域有实际推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] NegMerge: Sign-Consensual Weight Merging for Machine Unlearning](../../ICML2025/llm_safety/negmerge_sign-consensual_weight_merging_for_machine_unlearning.md)
- [\[NeurIPS 2025\] SIMU: Selective Influence Machine Unlearning](../../NeurIPS2025/llm_safety/simu_selective_influence_machine_unlearning.md)
- [\[CVPR 2025\] LoTUS: Large-Scale Machine Unlearning with a Taste of Uncertainty](../../CVPR2025/llm_safety/lotus_large-scale_machine_unlearning_with_a_taste_of_uncertainty.md)
- [\[ICLR 2026\] OFMU: Optimization-Driven Framework for Machine Unlearning](../../ICLR2026/llm_safety/ofmu_optimization-driven_framework_for_machine_unlearning.md)
- [\[CVPR 2026\] Machine Unlearning via Adaptive Gradient Reweighting and Multi-stage Objective Optimization](../../CVPR2026/llm_safety/machine_unlearning_via_adaptive_gradient_reweighting_and_multi-stage_objective_o.md)

</div>

<!-- RELATED:END -->
