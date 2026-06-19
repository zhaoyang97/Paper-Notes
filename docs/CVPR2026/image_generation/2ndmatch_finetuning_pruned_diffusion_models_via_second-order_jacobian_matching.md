---
title: >-
  [论文解读] 2ndMatch: Finetuning Pruned Diffusion Models via Second-Order Jacobian Matching
description: >-
  [CVPR 2026][图像生成][扩散模型] 提出2ndMatch微调框架，通过对齐剪枝模型与原始模型的二阶Jacobian矩阵 $J^\top J$（灵感来自有限时间Lyapunov指数），匹配两者对输入扰动的时间敏感性，从而显著缩小剪枝扩散模型与原始模型的生成质量差距。 领域现状：扩散模型在图像生成中效果出色…
tags:
  - "CVPR 2026"
  - "图像生成"
  - "扩散模型"
  - "模型剪枝"
  - "Jacobian匹配"
  - "有限时间Lyapunov指数"
  - "知识蒸馏"
---

# 2ndMatch: Finetuning Pruned Diffusion Models via Second-Order Jacobian Matching

**会议**: CVPR 2026  
**arXiv**: [2506.05398](https://arxiv.org/abs/2506.05398)  
**代码**: 无  
**领域**: 扩散模型 / 模型压缩  
**关键词**: 扩散模型, 模型剪枝, Jacobian匹配, 有限时间Lyapunov指数, 知识蒸馏

## 一句话总结
提出2ndMatch微调框架，通过对齐剪枝模型与原始模型的二阶Jacobian矩阵 $J^\top J$（灵感来自有限时间Lyapunov指数），匹配两者对输入扰动的时间敏感性，从而显著缩小剪枝扩散模型与原始模型的生成质量差距。

## 研究背景与动机

**领域现状**：扩散模型在图像生成中效果出色，但推理时需要数百次去噪步骤，计算开销巨大。模型剪枝是减少每步计算量的有效策略。

**现有痛点**：剪枝后的微调通常复用原始去噪分数匹配（DSM）目标，对容量减小的剪枝模型来说不足够。现有知识蒸馏对齐输出或中间特征，但忽视了模型的**敏感性**——即分数函数对输入扰动的响应。一阶Jacobian匹配对扩散模型基本等价于KD（因输入本身就有噪声扰动），且无法捕捉跨时间步的扰动传播。

**核心矛盾**：剪枝模型容量减小→对扰动的敏感性与原始模型偏离→去噪轨迹漂移→生成质量下降。需要一种方法约束剪枝模型保持与原始模型相同的时间动力学行为。

**切入角度**：将扩散模型视为离散时间动力系统，从有限时间Lyapunov指数（FTLE）理论出发，FTLE量化了微小扰动在有限时间内的放大/收缩率。

**核心idea**：对齐剪枝模型和原始模型的 $J^\top J$（二阶Jacobian度量），通过随机投影 $v^\top J^\top J v$ 高效估计方向性膨胀率，实现可扩展的二阶Jacobian匹配。

## 方法详解

### 整体框架
2ndMatch 想解决的问题是：扩散模型被剪枝后容量变小，单靠标准去噪目标微调补不回生成质量。作者把整个微调过程看成在三个层面上同时向原始（dense）模型看齐——预测对不对、输出像不像、以及对输入扰动的"反应"是否一致。前两者是常规手段，真正的新东西是第三个：让剪枝模型在每个时间步上对扰动的放大行为和原始模型保持一致。三者合成一个混合目标共同训练：

$$\mathcal{L}_{total} = \lambda_{NP}\mathcal{L}_{NP} + \lambda_{KD}\mathcal{L}_{KD} + \lambda_{Jac}\mathcal{L}_{2nd\text{-}Jac}$$

### 关键设计

**1. 噪声预测：给剪枝模型保底的基础监督**

这是标准 DDPM 目标，让模型预测前向过程加进去的那一份噪声 $\epsilon$，写作 $\mathcal{L}_{NP} = \mathbb{E}_{\tilde{x},t,\epsilon}[\|s(\tilde{x},t;\theta) - \epsilon\|_2^2]$。它是任何扩散模型训练都绕不开的监督信号，但问题恰恰在于：对一个被砍掉近一半参数的剪枝模型，只靠这个目标拟合噪声并不够，收敛慢、终点也偏。所以它在这里只是"地基"，真正补质量要靠后面两项。

**2. 知识蒸馏：用原始模型的输出当更平滑的老师**

直接对齐剪枝模型和原始模型在同一输入上的分数输出：$\mathcal{L}_{KD} = \mathbb{E}_{\tilde{x},t}[\|s(\tilde{x},t;\theta) - s_\mathcal{D}(\tilde{x},t;\theta_\mathcal{D})\|_2^2]$。相比预测原始噪声 $\epsilon$（本身带很大随机性），原始模型给出的分数是一个更平滑、信息更密的监督目标，因此能加速收敛、把学生拉到离老师更近的位置。但它只管住了"输出值"这一层，管不了模型对扰动的动态响应——这正是第三项要补的缺口。

**3. 二阶Jacobian匹配：对齐两者对扰动的时间敏感性（核心创新）**

前两项只盯着"输出像不像"，却忽略了一件对扩散尤其要命的事：去噪是一个多步迭代的动力系统，某一步对输入的微小扰动会沿着后续时间步被放大或收缩，剪枝模型一旦在这个放大率上和原始模型跑偏，轨迹就会越走越歪，最终生成质量塌掉。作者从有限时间Lyapunov指数（FTLE）出发刻画这件事——FTLE 量化扰动在有限时间内的膨胀率，而一步的局部膨胀由二阶Jacobian度量 $J^\top J$ 决定：$\|v_1\| \approx \sqrt{v_0^\top J^\top J v_0}$。

直接构造完整 Jacobian 在高维上不可行，作者用随机投影绕开：采一个随机方向 $v\sim\mathcal{N}(0,I)$、归一化为 $\hat{v}=v/\|v\|$，只比较剪枝模型与原始模型在这个方向上的方向性膨胀率，并用 Jacobian-向量积（JVP）算 $J\hat{v}$，全程不显式形成 Jacobian：

$$\mathcal{L}_{2nd\text{-}Jac} = \mathbb{E}_{\tilde{x},t,v}\left[(\|J\hat{v}\|_2^2 - \|J_\mathcal{D}\hat{v}\|_2^2)^2\right]$$

之所以非要做"二阶"而不是更直觉的一阶 Jacobian 匹配，是因为后者在扩散里其实是冗余的。作者对带噪输入做 Taylor 展开得到 $\|s(x') - s_\mathcal{D}(x')\|_2^2 = \|s(x) - s_\mathcal{D}(x)\|_2^2 + \sigma^2\|J - J_\mathcal{D}\|_F^2 + \mathcal{O}(\sigma^4)$：由于扩散输入本身就带噪声扰动 $\sigma$，单纯的输出对齐（即 KD 项）里已经隐式包含了一阶 Jacobian 匹配那一项，再显式加一阶约束只会徒增计算、带不来新信息（实验里也确实让 FID 不降反升）。二阶项捕捉的是扰动跨时间步的传播行为，正好对应动力系统的稳定性，这是输出对齐和一阶匹配都够不到的层面。

### 损失函数 / 训练策略
三项加权求和即为总目标，整套方法对架构和剪枝方式都不挑：U-Net 与 Transformer 两类扩散骨干都适用，也能直接叠在 Diff-Pruning、BK-SDM 等不同剪枝方法之上。工程上靠 PyTorch 的 JVP 功能高效计算 $J\hat{v}$，避免显式构造 Jacobian 带来的内存与算力爆炸。

## 实验关键数据

### 主实验（LSUN + ImageNet 256×256，U-Net模型）

| 数据集 | 方法 | 参数量 | MACs | FID↓ | rFID↓ |
|--------|------|--------|------|------|-------|
| LSUN-Church | DDPM（原始） | 113.7M | 248.7G | 10.58 | - |
| | Diff-Pruning | 63.2M | 138.8G | 13.90 | 4.09 |
| | **2ndM (Ours)** | 63.2M | 138.8G | **11.25** | **2.08** |
| LSUN-Bedroom | DDPM（原始） | 113.7M | 248.7G | 6.62 | - |
| | Diff-Pruning | 63.2M | 138.8G | 17.90 | 7.62 |
| | **2ndM (Ours)** | 63.2M | 138.8G | **9.68** | **2.16** |
| ImageNet | LDM-4（原始） | 400.9M | 99.8G | 3.60 | - |
| | Diff-Pruning | 175.8M | 43.2G | 10.23 | 9.28 |
| | **2ndM (Ours)** | 175.8M | 43.2G | **5.68** | **4.11** |

Stable Diffusion (COCO 512×512)：Base+2ndM FID从15.76降至13.84，Small+2ndM从16.98降至16.17。

### 消融实验（CIFAR-10）

| 配置 | FID↓ | FTLE |
|------|------|------|
| NP only | 5.29 | 0.413 |
| NP + KD | 5.05 | 0.418 |
| NP + KD + 1st JM | 5.14 | - |
| NP + KD + 2ndM (Ours) | **4.58** | - |
| Dense（原始） | 4.19 | - |

### 关键发现
- **一阶Jacobian匹配无效**：加入一阶JM后FID反而从5.05升至5.14，验证了理论分析
- **二阶匹配至关重要**：加入2ndM将FID从5.05大幅降至4.58，且FTLE更接近原始模型，证明时间敏感性对齐的有效性
- LSUN-Bedroom上FID改进46%（17.90→9.68），ImageNet上rFID改进55%
- Transformer模型上同样有效：U-ViT在CIFAR-10上FID从4.63降至4.05

## 亮点与洞察
- **动力系统视角的创新**：将扩散模型的微调问题重新表述为动力系统稳定性问题，用FTLE理论指导损失函数设计，这个视角对理解扩散模型的训练和生成过程有深刻启发
- **Taylor展开的优雅证明**：严格证明了一阶Jacobian匹配在扩散模型中的冗余性，为模型压缩中的损失设计提供理论指导
- **随机投影的实用性**：通过随机方向估计 $v^\top J^\top J v$ 绕过了高维Jacobian计算的瓶颈，使方法可扩展到大规模模型（Stable Diffusion 1.04B参数）

## 局限与展望
- 当前使用步级（step-wise）匹配近似多步Jacobian传播，对长程时间依赖的捕捉能力有限
- 随机投影的效率与估计精度之间的trade-off未充分探讨
- 仅在图像生成上验证，视频/3D等更复杂的扩散模型应用有待探索
- 可将FTLE思想扩展到蒸馏（非剪枝）场景、或用于指导采样调度器设计

## 相关工作与启发
- **vs Diff-Pruning**: Diff-Pruning仅用DSM微调剪枝模型，2ndM在此基础上加入敏感性对齐，同参数量下FID显著改善
- **vs DeepCache**: DeepCache通过缓存中间特征加速但不减参数，和剪枝方法互补
- **vs BK-SDM**: BK-SDM为Stable Diffusion设计的剪枝方法，2ndM可直接叠加使用提升效果

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ FTLE理论引入模型压缩领域，二阶Jacobian匹配的formulation优雅且有理论支撑
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖U-Net和Transformer架构、5个数据集、多种剪枝方法、充分消融
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，motivation清晰，实验设计系统
- 价值: ⭐⭐⭐⭐ 通用微调框架，但仅限模型剪枝场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Efficient Fine-Tuning and Concept Suppression for Pruned Diffusion Models](../../CVPR2025/image_generation/efficient_fine-tuning_and_concept_suppression_for_pruned_diffusion_models.md)
- [\[ICML 2026\] Esoteric Language Models: A Family of Any-Order Diffusion LLMs](../../ICML2026/image_generation/esoteric_language_models_a_family_of_any-order_diffusion_llms.md)
- [\[CVPR 2026\] When Local Rules Create Global Order: Self-Organized Representation Learning for Latent Diffusion Models](when_local_rules_create_global_order_self-organized_representation_learning_for_.md)
- [\[CVPR 2026\] LeapAlign: Post-Training Flow Matching Models at Any Generation Step by Building Two-Step Trajectories](leapalign_post_training_flow_matching_models_at_any_generation_step.md)
- [\[ICLR 2026\] HOG-Diff: Higher-Order Guided Diffusion for Graph Generation](../../ICLR2026/image_generation/hog-diff_higher-order_guided_diffusion_for_graph_generation.md)

</div>

<!-- RELATED:END -->
