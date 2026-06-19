---
title: >-
  [论文解读] CORAL: Disentangling Latent Representations in Long-Tailed Diffusion
description: >-
  [NeurIPS 2025][图像生成][扩散模型] 深入诊断长尾数据下扩散模型尾类生成退化的根因为 U-Net 瓶颈层的"表示纠缠"（representation entanglement），提出 CORAL 通过在瓶颈层施加监督对比损失来解耦类别表示，在 CIFAR10/100-LT、CelebA-5、ImageNet-LT 上全面超越 DDPM/CBDM/T2H 等基线。
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "扩散模型"
  - "long-tailed distribution"
  - "对比学习"
  - "latent space"
  - "representation entanglement"
---

# CORAL: Disentangling Latent Representations in Long-Tailed Diffusion

**会议**: NeurIPS 2025  
**arXiv**: [2506.15933](https://arxiv.org/abs/2506.15933)  
**代码**: [https://github.com/SankarLab/coral-lt-diffusion](https://github.com/SankarLab/coral-lt-diffusion)  
**领域**: 图像生成 / 长尾学习  
**关键词**: diffusion model, long-tailed distribution, contrastive learning, latent space, representation entanglement

## 一句话总结

深入诊断长尾数据下扩散模型尾类生成退化的根因为 U-Net 瓶颈层的"表示纠缠"（representation entanglement），提出 CORAL 通过在瓶颈层施加监督对比损失来解耦类别表示，在 CIFAR10/100-LT、CelebA-5、ImageNet-LT 上全面超越 DDPM/CBDM/T2H 等基线。

## 研究背景与动机

**扩散模型在长尾数据下的失败模式**。扩散模型在类别均衡数据上表现出色，但现实世界数据通常呈长尾分布——少数头部类占据大量样本，大量尾部类样本稀缺。在这种设置下，扩散模型对尾类的生成质量严重退化，表现为多样性低、出现"特征借用"（feature borrowing）。

**已有方法的局限**。现有的改进主要在数据层面（如 CBDM 的平衡采样正则）或输出空间（如 DiffROP 的输出分布对比损失）操作，但忽略了问题的根源所在——去噪网络内部的潜表示结构。T2H 使用贝叶斯门控在头尾类之间迁移知识，但仍未直接干预潜空间的类别结构。

**本文的关键发现**。作者通过 t-SNE 等可视化手段发现，长尾训练下 U-Net 瓶颈层的尾类潜表示与头类发生严重重叠（representation entanglement）。这不单是"数据少"的问题，而是类别间相对不平衡导致头类主导了参数更新，使尾类丧失了结构化的潜表示。CORAL 的核心 idea 是在潜空间中直接施加监督对比正则，在生成过程的源头解决类别纠缠问题。

## 方法详解

### 整体框架

CORAL 在标准 DDPM 训练流程上做了两处修改：(1) 在 U-Net 瓶颈层输出后添加一个轻量级的投影头（projection head），将瓶颈特征映射到对比学习的嵌入空间；(2) 在训练目标中加入时间依赖的监督对比损失项。训练完成后投影头被丢弃，推理时零额外计算开销，完全兼容标准 DDPM 采样流程。

### 关键设计

1. **投影头（Projection Head）**:

    - 功能：将 U-Net 瓶颈层特征映射到对比学习的嵌入空间
    - 核心思路：在瓶颈输出后添加一个线性层 + 归一化层的轻量级 MLP，对投影后的嵌入施加对比约束，而非直接约束瓶颈特征本身
    - 设计动机：遵循对比学习的最佳实践——投影头将对比目标与主要生成特征解耦，防止对比损失直接压缩瓶颈特征的类内多样性；瓶颈层是语义信息最集中的位置，也是表示纠缠发生的关键节点

2. **时间依赖的对比损失加权**:

    - 功能：在不同噪声水平下动态调节对比损失的权重
    - 核心思路：加权函数 $\lambda(t) = w \cdot \exp(\frac{1-t/T}{\tau_r})$，在低噪声时步（$t\approx 0$，语义结构可恢复）给予更高对比权重，高噪声时步（$t\approx T$，噪声主导）减弱对比约束
    - 设计动机：扩散过程中不同时步的语义信息量不同，早期步骤中语义结构更清晰，此时施加对比约束更有效；后期噪声过大强行对比反而会干扰训练

3. **潜空间 vs 环境空间的正则化选择**:

    - 功能：在生成过程的源头而非输出端干预类别结构
    - 核心思路：对比损失直接作用于 U-Net 内部的瓶颈表示，而非像 DiffROP 那样在生成的图像上做后验约束
    - 设计动机：环境空间的正则化是在"已经生成"的输出上约束，此时纠缠已经发生；潜空间干预是在类别重叠发生的位置直接分离，从根源解决问题。消融实验也证实潜空间方法一致优于环境空间方法

### 损失函数 / 训练策略

总体训练目标 $\mathcal{L}_{\text{CORAL}} = \mathcal{L}_{\text{diff}} + \lambda(t) \cdot \mathcal{L}_{\text{con}}$，其中 $\mathcal{L}_{\text{diff}}$ 为标准噪声预测损失，$\mathcal{L}_{\text{con}}$ 使用 SupCon 损失（在每个 mini-batch 内拉近同类投影嵌入、推远异类嵌入），温度参数 $\tau_{\text{SC}}$ 控制分布锐度。训练时与 CFG 兼容，以概率 $p_{\text{uncond}}$ 随机丢弃类别标签进行无条件训练，但对比损失始终使用真实标签。推理时投影头被丢弃，标准 CFG 采样。

## 实验关键数据

### 主实验

| 数据集 | 指标 | CORAL(本文) | DDPM | CBDM | T2H |
|--------|------|------|------|------|------|
| CIFAR10-LT (ρ=0.01) | FID↓ | **5.32** | 6.17 | 5.62 | 7.01 |
| CIFAR10-LT (ρ=0.01) | IS↑ | **9.69** | 9.43 | 9.28 | 9.63 |
| CIFAR10-LT (ρ=0.01) | Recall↑ | **0.59** | 0.52 | 0.57 | 0.54 |
| CIFAR10-LT (ρ=0.001) | FID↓ | **11.03** | 13.05 | 12.74 | 12.80 |
| CIFAR100-LT (ρ=0.01) | FID↓ | **5.37** | 7.70 | 6.02 | 6.78 |
| CIFAR100-LT (ρ=0.01) | IS↑ | **13.53** | 13.20 | 12.92 | 12.97 |
| CelebA-5 | FID↓ | **8.12** | 10.28 | 8.74 | 9.50 |
| ImageNet-LT (1000类) | FID↓ | **16.11** | 17.08 | 22.66 | 18.59 |
| ImageNet-LT (1000类) | IS↑ | **24.17** | 21.03 | 17.13 | 19.15 |
| ImageNet-LT (1000类) | Recall↑ | **0.48** | 0.39 | 0.42 | 0.44 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 潜空间正则(CORAL) vs 环境空间正则 | CORAL一致更优 | 验证在潜空间干预优于输出端约束 |
| 不同 SupCon 温度 τ_SC | [0.1, 0.5] 范围较好 | 详见附录消融图 |
| 不同衰减温度 τ_r | [0.5, 1.0] 范围最佳 | 控制时间依赖权重的衰减速率 |
| 不同 CFG 权重 ω | 与标准 DDPM 趋势一致 | CORAL 不改变最优 CFG 配置 |

### 关键发现
- CORAL 在所有数据集、所有指标上一致超越全部基线，尤其在多样性/覆盖度指标（Recall、F8）上提升最大
- ImageNet-LT（1000 类）上 CORAL 优势最显著——环境空间方法（如 CBDM FID=22.66）在大类别数下性能退化，而潜空间干预保持稳定
- Per-class FID 分析显示 CORAL 在尾类上的提升最为突出，头类性能不受损
- t-SNE 可视化清晰展示 CORAL 训练后瓶颈层的类别分离效果

## 亮点与洞察
- 根因诊断深入：不只观察"尾类生成差"的现象，而是定位到 U-Net 瓶颈层的表示纠缠机制，并通过平衡与不平衡数据集的对比证明这是类别不平衡（而非数据量少）导致的
- 方法巧妙简洁：投影头 + 对比损失的设计训练时可插拔、推理时零开销，与现有扩散训练框架完全兼容
- 时间依赖加权从物理直觉出发：低噪声步骤语义清晰适合施加对比约束，高噪声步骤减弱——符合扩散过程的信息论特性

## 局限与展望
- 需要训练阶段介入，不能做 test-time 修复或后处理
- 实验限于 32×32 和 64×64 分辨率，高分辨率（如 256×256）和 Stable Diffusion 等大模型的可扩展性未验证
- 对比损失的超参数在不同数据集间可能需要调优
- 仅验证了类别条件生成，文本条件（text-to-image）场景下的长尾问题是重要的未来方向

## 相关工作与启发
- 与 CBDM 的区别：CBDM 在数据采样层面做平衡，CORAL 在特征空间层面做结构化分离
- 与 DiffROP 的区别：DiffROP 在输出分布上做 KL 对比，CORAL 在内部瓶颈层做 SupCon——干预位置更靠近问题根源
- 潜在扩展：可与 LoRA 微调结合用于大规模 T2I 模型的领域适配中的稀有概念保护

## 评分
- 新颖性: ⭐⭐⭐⭐ 表示纠缠的根因诊断和潜空间干预的思路新颖
- 实验充分度: ⭐⭐⭐⭐ 4个数据集全面对比+消融+可视化+per-class分析
- 写作质量: ⭐⭐⭐⭐ 问题诊断→方法设计→实验验证的逻辑链条清晰
- 价值: ⭐⭐⭐⭐ 长尾生成是实际场景常态，方法简洁实用且可扩展

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Bitrate-Controlled Diffusion for Disentangling Motion and Content in Video](../../ICCV2025/image_generation/bitrate-controlled_diffusion_for_disentangling_motion_and_content_in_video.md)
- [\[NeurIPS 2025\] Diffusion Model as a Noise-Aware Latent Reward Model for Step-Level Preference Optimization](diffusion_model_as_a_noiseaware_latent_reward_model_for_step.md)
- [\[NeurIPS 2025\] Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge](towards_general_modality_translation_with_contrastive_and_predictive_latent_diff.md)
- [\[NeurIPS 2025\] A Gradient Flow Approach to Solving Inverse Problems with Latent Diffusion Models](a_gradient_flow_approach_to_solving_inverse_problems_with_latent_diffusion_model.md)
- [\[ICCV 2025\] 3DSR: Bridging Diffusion Models and 3D Representations for 3D Consistent Super-Resolution](../../ICCV2025/image_generation/bridging_diffusion_models_and_3d_representations_a_3d_consis.md)

</div>

<!-- RELATED:END -->
