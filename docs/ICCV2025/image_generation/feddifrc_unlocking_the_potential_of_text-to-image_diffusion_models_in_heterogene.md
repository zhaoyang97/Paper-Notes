---
title: >-
  [论文解读] FedDifRC: Unlocking the Potential of Text-to-Image Diffusion Models in Heterogeneous Federated Learning
description: >-
  [ICCV 2025][图像生成][联邦学习] 首次将预训练文本到图像扩散模型（Stable Diffusion）的内部表示引入联邦学习，提出 FedDifRC 框架，通过文本驱动的扩散对比学习（TDCL）和噪声驱动的扩散一致性正则化（NDCR）两个互补模块，有效缓解数据异质性问题，在多种 non-iid 场景下显著提升全局模型性能。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "联邦学习"
  - "数据异质性"
  - "扩散模型表示"
  - "对比学习"
  - "一致性正则化"
---

# FedDifRC: Unlocking the Potential of Text-to-Image Diffusion Models in Heterogeneous Federated Learning

**会议**: ICCV 2025  
**arXiv**: [2507.06482](https://arxiv.org/abs/2507.06482)  
**代码**: [https://github.com/hwang52/FedDifRC](https://github.com/hwang52/FedDifRC)  
**领域**: 图像生成  
**关键词**: 联邦学习, 数据异质性, 扩散模型表示, 对比学习, 一致性正则化

## 一句话总结

首次将预训练文本到图像扩散模型（Stable Diffusion）的内部表示引入联邦学习，提出 FedDifRC 框架，通过文本驱动的扩散对比学习（TDCL）和噪声驱动的扩散一致性正则化（NDCR）两个互补模块，有效缓解数据异质性问题，在多种 non-iid 场景下显著提升全局模型性能。

## 研究背景与动机

联邦学习（FL）面临的核心挑战之一是数据异质性（non-iid）：各客户端的本地数据分布差异导致局部模型优化方向不一致，使全局模型收敛缓慢且不稳定。

**现有解决方案的局限**：
- **客户端优化方法**（FedProx、SCAFFOLD 等）：通过限制本地更新与全局模型的偏差来减少梯度不一致性，但无法从根本上解决局部模型过拟合本地分布的问题
- **服务器聚合方法**（FedNova 等）：改进全局聚合策略，但参数偏差仍在逐步扩大
- **基于合成数据的方法**：用扩散模型生成合成数据增强训练，但合成数据仍可能使本地模型过拟合局部域分布，异质性问题并未根本解决

**核心洞察**：预训练的 Stable Diffusion 模型蕴含丰富的视觉语义表示能力。作者通过 t-SNE 可视化（Fig. 2）发现，即使未在特定数据集上训练，SD 模型的 UNet 解码器在适当的时间步和层级上就能自然地将不同类别的样本聚类。这启发了两个关键观察：

**扩散模型的广泛通用知识可增强 FL 的局部语义多样性**（→ TDCL）

**扩散模型中关于语义对象的平滑对应关系是 FL 的自然引导信号**（→ NDCR）

## 方法详解

### 整体框架

FedDifRC 在标准 FL 流程（FedAvg）基础上，在每个客户端的本地训练中添加两个基于预训练 SD 模型的正则化模块。SD 模型参数在整个训练过程中**完全冻结**，仅用于提取表示。总体损失函数为：

$$\mathcal{L} = \mathcal{L}_{TDCL} + \mathcal{L}_{NDCR} + \mathcal{L}_{CE}$$

### 关键设计

1. **条件扩散表示（Conditional Diffusion Representations）**：

    - 功能：利用 SD 模型的条件生成反馈，为每个样本构建丰富的类别相关语义表示
    - 核心思路：将样本 $x_i$ 的特征编码 $\mathbf{c}_i = h_k(x_i)$ 作为条件注入 SD 模型，配合文本提示 $\mathcal{P}_{y_i}$ = "a photo of a [类别名]"，从 UNet 解码器第 2-4 层提取特征图，经 PCA 降维后拼接为融合特征 $\widetilde{\mathcal{F}}_i$
    - 设计动机：K-Means 聚类（Fig. 3）和 PCA 可视化（Fig. 4）表明，融合多层特征能同时捕获高层语义和低层纹理信息，比单层特征更全面

2. **文本驱动扩散对比学习（TDCL）**：

    - 功能：构建基于扩散表示的跨模态对比学习，增强局部模型的类别判别能力
    - 核心思路：对每个样本的嵌入 $z_i$，以匹配文本提示生成的条件扩散表示 $\widetilde{\mathcal{F}}_i$ 为正样本对，以不匹配文本提示生成的表示 $\widetilde{\mathcal{F}}_{\mathcal{N}_i}$ 为负样本对，使用改进的 InfoNCE 损失：$\mathcal{L}_{TDCL} = \log(1 + \frac{\sum_j \exp(s(z_i, \widetilde{\mathcal{F}}_j)/\tau)}{\exp(s(z_i, \widetilde{\mathcal{F}}_i)/\tau)})$
    - 相似度计算使用归一化余弦相似度，归一化因子 $\mathcal{U}$ 为所有样本嵌入与当前扩散表示的平均距离
    - 设计动机：正负样本对来自同一输入但不同文本条件的扩散模型反馈，提供了丰富的域间变化信息，帮助局部模型学习更通用的类别区分知识

3. **噪声驱动扩散一致性正则化（NDCR）**：

    - 功能：利用去噪扩散表示作为稳定的收敛目标，约束局部模型的特征空间
    - 核心思路：对输入 $x_i$ 添加 $t$ 步噪声后输入 SD 模型进行去噪，提取 UNet 解码器特征融合为 $\widetilde{\mathcal{H}}_i$，用 L2 损失对齐：$\mathcal{L}_{NDCR} = \sum_{q=1}^{d}(z_{i(q)} - \widetilde{\mathcal{H}}_{i(q)})^2$
    - 设计动机：TDCL 的条件扩散表示依赖于动态生成的条件，每轮变化，无法提供稳定的收敛信号。去噪扩散表示来自 SD 模型的固定去噪过程，作为"虚拟教师"提供一致的特征级对齐目标

### 损失函数 / 训练策略

- 总损失：$\mathcal{L} = \mathcal{L}_{TDCL} + \mathcal{L}_{NDCR} + \mathcal{L}_{CE}$，其中 $\mathcal{L}_{CE}$ 是标准交叉熵
- SD 模型完全冻结不参与训练，仅用作表示提取器
- 可扩展为自监督方案：TDCL 使用 "a photo of a similar object" 作为正样本提示，从 Tiny-ImageNet 类别名中随机选取用于负样本提示；NDCR 使用 "a photo of a visual object" 生成去噪表示
- 作者提供了非凸目标下的收敛性理论分析，给出了通信轮数 $R$ 和学习率 $\eta$ 的收敛条件

## 实验关键数据

### 主实验（CIFAR-10，不同 non-iid 设置，准确率%）

| 方法 | NID1_0.05 | NID1_0.2 | NID1_0.5 | NID2 | AVG |
|------|-----------|----------|----------|------|-----|
| FedAvg | 78.27 | 84.65 | 86.11 | 72.60 | 80.41 |
| FedProx | 78.42 | 84.59 | — | 72.81 | — |
| MOON | 80.79 | 86.10 | — | 73.35 | — |
| FedNH | 80.25 | — | — | — | — |
| **FedDifRC** | **83.14** | **88.27** | **89.31** | **76.45** | **84.29** |

### 消融实验（CIFAR-10，NID1_0.2 和不同层级特征）

| 配置 | NID1_0.05 | NID1_0.2 | NID1_0.5 | NID2 | 说明 |
|------|-----------|----------|----------|------|------|
| Baseline (FedAvg) | 78.27 | 84.65 | 86.11 | 72.60 | 无扩散模型辅助 |
| + TDCL only | 81.39 | 86.03 | 88.16 | 75.67 | 对比学习有效 |
| + NDCR only | 80.35 | 86.40 | 87.54 | 75.33 | 一致性正则有效 |
| + TDCL + NDCR | **83.14** | **88.27** | **89.31** | **76.45** | 两模块互补 |
| 层 L=2 only | — | 87.28 | — | 75.73 | 高层语义 |
| 层 L=3 only | — | 87.81 | — | 75.61 | 低层纹理 |
| 融合 L={2,3,4} | — | **88.27** | — | **76.45** | 融合效果最优(+0.46) |

### 关键发现
- TDCL 和 NDCR 是互补的：单独使用各带来 ~2% 提升，组合使用提升 ~4%
- 融合多层特征（L={2,3,4}）持续优于任何单层特征，但改善幅度有限（+0.28~0.72%）
- 去噪时间步 $t=300$ 是最优选择（Fig. 6 左），过大（t=999）导致表示模糊不可分
- 可扩展到长尾分布、域偏移等多种异质性场景，均有显著效果
- 自监督方案（无标签数据）也能有效工作

## 亮点与洞察
- 首次系统探索将预训练扩散模型的内部表示用于增强联邦学习，开辟了 FL 的新方向
- 通过 t-SNE 和 K-Means 的详细分析（Fig. 2-4），为"扩散模型是有效的表示学习器"提供了直观验证
- 从理论角度证明了 SD 模型的去噪过程等价于学习数据主成分空间的线性自编码器（Eq. 6），为利用扩散表示提供了理论基础
- TDCL 和 NDCR 分别解决了对比学习中正/负样本构建和收敛稳定性两个不同层面的问题，设计思路清晰

## 局限与展望
- 需要在每个客户端部署预训练 SD 模型进行推理，增加了客户端的计算和存储负担
- PCA 降维的主成分数量（256/128）需要预定义，可能不是所有数据集的最优选择
- 文本提示模板固定为 "a photo of a [class]"，对细粒度类别可能不够表达力
- 目前仅在图像分类任务上验证，未在目标检测、语义分割等下游任务上测试
- 未探索更新型的扩散模型（如 DiT 架构）是否能提供更好的表示

## 相关工作与启发
- DIFT 和 Diffusion Hyperfeatures 已证明扩散模型的中间特征是优秀的视觉表示，本文首次将此应用于 FL
- 对比学习在 FL 中的应用（MOON、FedCR）已有探索，FedDifRC 的创新在于使用扩散模型作为对比学习的"锚点"
- 这种"利用大型生成模型的表示来辅助判别式任务训练"的思路可以推广到其他分布式学习场景

## 评分
- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Unlocking the Potential of Diffusion Priors in Blind Face Restoration](unlocking_the_potential_of_diffusion_priors_in_blind_face_restoration.md)
- [\[ICLR 2026\] Bridging Generalization Gap of Heterogeneous Federated Clients Using Generative Models](../../ICLR2026/image_generation/bridging_generalization_gap_of_heterogeneous_federated_clients_using_generative_.md)
- [\[ICCV 2025\] PLA: Prompt Learning Attack against Text-to-Image Generative Models](pla_prompt_learning_attack_against_text-to-image_generative_models.md)
- [\[ICCV 2025\] Golden Noise for Diffusion Models: A Learning Framework](golden_noise_for_diffusion_models_a_learning_framework.md)
- [\[ICCV 2025\] REPA-E: Unlocking VAE for End-to-End Tuning with Latent Diffusion Transformers](repa-e_unlocking_vae_for_end-to-end_tuning_of_latent_diffusion_transformers.md)

</div>

<!-- RELATED:END -->
