---
title: >-
  [论文解读] Active Generation for Image Classification
description: >-
  [ECCV 2024][主动学习] 本文提出ActGen，将主动学习思想融入扩散模型的图像生成过程，通过识别模型误分类的验证样本作为引导图像、结合注意力引导和基于梯度的生成控制，仅用10%的合成图像即可在ImageNet上实现+2.26%的准确率提升，超过了使用94%合成数据的先前方法。 深度生成模型（特别是扩散模型）在提升…
tags:
  - "ECCV 2024"
  - "主动学习"
  - "扩散模型"
  - "数据增强"
  - "图像生成"
  - "困难样本挖掘"
---

# Active Generation for Image Classification

**会议**: ECCV 2024  
**arXiv**: [2403.06517](https://arxiv.org/abs/2403.06517)  
**代码**: [https://github.com/hunto/ActGen](https://github.com/hunto/ActGen)  
**领域**: 数据增强 / 图像分类  
**关键词**: 主动学习, 扩散模型, 数据增强, 图像生成, 困难样本挖掘

## 一句话总结
本文提出ActGen，将主动学习思想融入扩散模型的图像生成过程，通过识别模型误分类的验证样本作为引导图像、结合注意力引导和基于梯度的生成控制，仅用10%的合成图像即可在ImageNet上实现+2.26%的准确率提升，超过了使用94%合成数据的先前方法。

## 研究背景与动机
深度生成模型（特别是扩散模型）在提升图像分类精度方面具有显著潜力，但现有方法存在严重的效率问题：需要生成与原始数据集比例极大的合成图像，却只能获得微小的精度提升。例如Azizi等人在ImageNet上使用120万张合成图像（接近原始数据集大小），仅获得了1.78%的精度提升。这种巨大的计算开销和微小的收益之间的矛盾严重限制了合成数据方法的实用性。核心问题在于：现有方法**不加选择地随机生成图像**，导致大量冗余的合成样本对模型训练几乎没有帮助。本文的切入角度是借鉴主动学习的思想——只生成模型真正需要的样本，即那些模型当前分类错误的困难样本的变体。核心idea：**生成少量、精准的困难样本比生成大量随机样本更有效**。

## 方法详解

### 整体框架
ActGen采用训练感知的在线生成策略：在每个训练epoch之后，用验证集评估模型找出误分类样本，以这些样本作为引导图像通过扩散模型生成类似的困难样本，然后将生成的图像加入训练集进入下一轮训练。生成过程结合了两种引导机制：注意力图像引导（保留前景物体、多样化背景）和基于梯度的生成引导（通过对比损失和分类损失控制生成难度和多样性）。

### 关键设计
1. **主动困难样本生成策略**:
    - 功能：精准识别并增强模型最需要的训练样本
    - 核心思路：从训练集中划分出验证集，每个epoch训练完成后在验证集上评估模型，将误分类样本作为困难样本的原型。这些样本通常具有不完整物体、异常姿态或类内罕见模式等特征
    - 设计动机：借鉴主动学习和课程学习的核心发现——在困难样本批次上训练比随机样本批次收敛更快，因此生成这些困难样本的变体能以最少的样本数获得最大的性能提升

2. **注意力图像引导 (Attentive Image Guidance)**:
    - 功能：在扩散去噪过程中引导生成与误分类样本相似的新图像
    - 核心思路：在DDPM的每个去噪步骤中，将生成的latent与引导图像的latent进行插值，得到最终的latent特征 $\tilde{x}_{t-1} = x_{t-1} + m_t \odot \gamma_t(x_{t-1}^{(g)} - x_{t-1})$。利用cross-attention层中类别prompt的注意力图作为前景mask $m_t$，仅对前景区域进行引导，保持背景的自由生成
    - 设计动机：直接对整张图进行像素级引导会导致背景场景缺乏多样性，使用注意力mask进行选择性引导可以保留前景物体特征的同时生成多样化的背景

3. **基于梯度的生成引导 (Gradient-based Guidance)**:
    - 功能：通过损失函数控制生成图像的多样性和分类难度
    - 核心思路：设计两个损失函数并通过梯度更新text embedding $c_{t-1} = c_t - \nu \frac{\nabla_{c_t}\mathcal{L}}{||\nabla_{c_t}\mathcal{L}||_2}$。对比损失 $\mathcal{L}_{contra}$ 用memory bank存储已生成图像的latent，惩罚当前生成与已生成图像过于相似的情况；对抗损失 $\mathcal{L}_{adv} = -\text{CE}(\Omega(o_t), y)$ 最大化分类损失，使生成图像对当前模型更具挑战性
    - 设计动机：仅靠图像引导容易产生冗余图像，对比损失保证多样性；对抗损失确保生成的是真正有助于模型学习的困难样本

### 损失函数 / 训练策略
生成引导的总损失为 $\mathcal{L} = \mathcal{L}_{contra} + \lambda \mathcal{L}_{adv}$，其中 $\lambda$ 为平衡因子。对比损失使用欧氏距离并设置margin $\rho=200$。梯度更新使用归一化梯度并应用于text embedding。训练策略上，只在前半部分epoch进行生成（后半部分学习率小，新生成的样本效果有限）。每个GPU在每个epoch后生成64张图像。验证集大小为10K。

## 实验关键数据

### 主实验
ImageNet分类结果：

| 模型 | 真实数据 | 生成数据 | 生成/真实比例 | 准确率 | 提升 |
|------|---------|---------|-------------|--------|------|
| ResNet-50 (Real only) | 1.28M | 0 | 0% | 76.39% | - |
| Azizi et al. (Imagen) | 1.28M | 1.2M | 94% | 78.17% | +1.78% |
| **ActGen (Ours)** | **1.28M** | **0.13M** | **10%** | **78.65%** | **+2.26%** |
| ViT-S/16 (Real only) | 1.28M | 0 | 0% | 79.89% | - |
| Azizi et al. (Imagen) | 1.28M | 1.2M | 94% | 81.00% | +1.11% |
| **ActGen (Ours)** | **1.28M** | **0.08M** | **6%** | **81.12%** | **+1.23%** |

### 消融实验

| 配置 | ImageNet准确率 | 说明 |
|------|---------------|------|
| Baseline (Real only) | 76.39% | 无合成数据 |
| Random SD生成 | 76.64% | +0.25%，随机生成效果有限 |
| + Image Guidance (IG) | 77.93% | +1.54%，图像引导贡献最大 |
| + Attentive IG (AIG) | 78.15% | +0.22%，前景mask提升多样性 |
| + 对比损失 $\mathcal{L}_{contra}$ | 78.36% | +0.21%，减少冗余 |
| + 对抗损失 $\mathcal{L}_{adv}$ | 78.65% | +0.29%，增加分类难度 |

### 关键发现
- **效率提升显著**：ActGen仅用先前方法约10%的合成图像，就超越了其性能（ImageNet上+0.48%优于Azizi et al.，同时节省约100万张合成图像）
- 验证集大小在5K以上时性能趋于稳定，说明不需要很大的验证集来识别困难样本
- 在few-shot场景（EuroSAT）中，ActGen用2K生成图像就能匹配甚至超越使用8K图像的Real guidance方法
- 附加的生成计算成本可控：与传统训练相比，额外增加约58%的GPU时间（15.2 vs 9.6 GPU days），远优于生成10倍数据的方案（~40 GPU days）

## 亮点与洞察
- 将主动学习的思想引入生成式数据增强，是一个简洁而强大的方法论创新——不生成更多数据，而是生成更对的数据
- 注意力mask的使用非常巧妙：利用cross-attention天然编码的前景位置信息，无需额外的分割模型
- 对抗样本生成不是向图像加噪声，而是通过text embedding的梯度更新改变生成语义，产生更自然的困难样本（如模糊、遮挡、风格变化）
- 训练感知的在线生成避免了两阶段方法的域差距问题

## 局限与展望
- 方法计算开销仍然不可忽略，每次epoch后都需运行扩散模型生成，在大规模训练中仍是瓶颈
- 当前仅在分类任务上验证，检测和分割等下游任务是否同样受益还需探索
- 对比损失使用全局latent距离，未考虑语义级别的相似性度量
- 验证集从训练集中划分，本身减少了可用的训练数据，对小数据集可能产生负面影响
- 生成图像质量依赖于底层扩散模型对特定类别的生成能力，对生成能力弱的类别效果可能不佳

## 相关工作与启发
- 与Azizi et al.的最大区别：从"生成足够多"转变为"生成足够好"，大幅降低计算成本
- 与Real guidance类似使用真实图像引导扩散过程，但引入了训练感知的主动选择机制
- 启发：数据增强的关键不在于数量而在于质量和针对性，困难样本挖掘思想在生成式增强中同样重要
- 未来方向：可以结合更先进的扩散模型（如SDXL）进一步提升生成质量，或将该框架推广到其他视觉任务

## 评分
- 新颖性: ⭐⭐⭐⭐ 主动学习与扩散生成的结合思路新颖，注意力引导和梯度控制设计巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖ImageNet、CIFAR、EuroSAT三个数据集，多种模型架构，消融详尽
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，方法动机阐述充分，图示设计直观
- 价值: ⭐⭐⭐⭐ 大幅降低了生成式数据增强的计算门槛，具有实际应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] ET: The Exceptional Trajectories - Text-to-Camera-Trajectory Generation with Character Awareness](et_the_exceptional_trajectories_text-to-camera-trajectory_generation_with_charac.md)
- [\[CVPR 2026\] Prototype-based Causal Intervention for Multi-Label Image Classification](../../CVPR2026/others/prototype-based_causal_intervention_for_multi-label_image_classification.md)
- [\[ICCV 2025\] NAPPure: Adversarial Purification for Robust Image Classification under Non-Additive Perturbations](../../ICCV2025/others/nappure_adversarial_purification_for_robust_image_classification_under_non-addit.md)
- [\[ECCV 2024\] COIN-Matting: Confounder Intervention for Image Matting](coin-matting_confounder_intervention_for_image_matting.md)
- [\[ECCV 2024\] FisherRF: Active View Selection and Mapping with Radiance Fields Using Fisher Information](fisherrf_active_view_selection_and_mapping_with_radiance_fields_using_fisher_inf.md)

</div>

<!-- RELATED:END -->
