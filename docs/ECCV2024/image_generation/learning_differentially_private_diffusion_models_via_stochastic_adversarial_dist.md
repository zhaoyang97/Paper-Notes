---
title: >-
  [论文解读] Learning Differentially Private Diffusion Models via Stochastic Adversarial Distillation
description: >-
  [ECCV 2024][图像生成] 提出 DP-SAD 框架，通过随机对抗蒸馏训练差分隐私扩散模型：利用扩散模型的时间步稀释 DP 噪声影响，引入判别器加速收敛，并结合梯度链式法则与 DP 后处理特性减少随机性引入，在不需要预训练的条件下实现了 SOTA 的隐私保护图像生成质量。 隐私敏感领域（医疗、金融）中数据共享受限…
tags:
  - "ECCV 2024"
  - "图像生成"
---

# Learning Differentially Private Diffusion Models via Stochastic Adversarial Distillation

**会议**: ECCV 2024  
**arXiv**: [2408.14738](https://arxiv.org/abs/2408.14738)  
**领域**: 图像生成

## 一句话总结

提出 DP-SAD 框架，通过随机对抗蒸馏训练差分隐私扩散模型：利用扩散模型的时间步稀释 DP 噪声影响，引入判别器加速收敛，并结合梯度链式法则与 DP 后处理特性减少随机性引入，在不需要预训练的条件下实现了 SOTA 的隐私保护图像生成质量。

## 研究背景与动机

隐私敏感领域（医疗、金融）中数据共享受限，差分隐私（DP）生成模型是一种解决方案——训练生成模型后发布模型而非数据。现有方法面临三大挑战：

**GAN 训练困难**：隐私约束使本就不稳定的 GAN 训练雪上加霜

**高维噪声放大**：数据/模型维度增大时，维持相同隐私级别需要更多 DP 噪声

**直接加噪损害严重**：DPSGD 对所有梯度加噪引入过多随机性

已有 DP 扩散模型（DP-DM、DP-LDM）需要在大数据集上预训练，且直接用 DPSGD 训练扩散模型导致隐私消耗过大。

## 方法详解

### 整体框架

DP-SAD 包含三个组件：
1. **教师模型 $\epsilon_\psi$**：使用隐私数据直接训练（无保护），仅用于指导学生训练，不发布
2. **学生模型 $\epsilon_\theta$**：通过蒸馏从教师学习，在梯度上施加裁剪和加噪实现 DP
3. **判别器 $\epsilon_\phi$**：区分教师和学生输出，形成对抗训练加速收敛

### 关键设计

**时间步稀释 DP 噪声**：核心创新在于利用扩散模型的 $T$ 个时间步来稀释 DP 噪声的影响。DP 噪声 $\mathcal{N}(0, \sigma^2 C^2 \mathbf{I})$ 被 $B \cdot T$ 分摊：

$$\bar{g} \approx \frac{1}{B}\sum_{i=1}^{B} CLIP\left(\frac{\partial \mathcal{L}^{i,r}}{\partial \theta}, C\right) + \frac{\mathcal{N}(0, \sigma^2 C^2 \mathbf{I})}{B \cdot T}$$

增大 $T$ 不增加隐私消耗但能减少噪声对梯度的影响。

**随机时间步采样**：用随机选择的单一时间步 $r$ 的梯度替代 $T$ 步梯度的平均，大幅降低计算开销（无需对每个样本跑完整 $T$ 步扩散链），通过前向过程直接获得 $x_r$ 而非从噪声逆向推断。

**链式法则+后处理特性**：利用梯度链式法则 $\frac{\partial \mathcal{L}}{\partial \theta} = \frac{\partial \mathcal{L}}{\partial x_{\theta}} \cdot \frac{\partial x_\theta}{\partial \theta}$，仅对 $\frac{\partial \mathcal{L}}{\partial x_\theta}$ 进行裁剪和加噪，$\frac{\partial x_\theta}{\partial \theta}$ 部分无需加噪（后处理特性），减少随机性引入。

**对抗判别器**：将教师和学生输出拼接作为判别器输入，教师标签 $[1,0]$、学生标签 $[0,1]$，驱动学生输出尽可能逼近教师。

### 损失函数

$$\mathcal{L} = \mathcal{L}_{dis} + \lambda \mathcal{L}_{adv}$$

- $\mathcal{L}_{dis}$：MSE 蒸馏损失 = MSE(教师输出, 学生输出) + MSE(真实数据, 学生输出)
- $\mathcal{L}_{adv}$：对抗损失 = $\log(1 - \epsilon_\phi(\mathcal{C}(x_{\psi}, x_{\theta})))$
- $\lambda = 1$（最优权衡系数，通过消融确定）

## 实验关键数据

### 主实验

CelebA 64×64 感知质量对比（$\varepsilon=10$，无需预训练的方法用 $\varepsilon=10^4$）：

| 方法 | 是否需要预训练 | IS ↑ | FID ↓ |
|---|---|---|---|
| DP-GAN | 否 | 1.00 | 403.94 |
| PATE-GAN | 否 | 1.00 | 397.62 |
| GS-WGAN | 否 | 1.00 | 384.78 |
| DP-MERF | 否 | 1.36 | 327.24 |
| DPGEN | 否 | 1.48 | 55.91 |
| DP-LDM | 是 | N/A | 14.30 |
| **DP-SAD** | **否** | **2.37** | **11.26** |

分类准确率对比（$\varepsilon=1$ / $\varepsilon=10$）：

| 方法 | MNIST | FMNIST | CelebA-H | CelebA-G |
|---|---|---|---|---|
| DP-GAN | 0.404/0.801 | 0.105/0.610 | 0.533/0.521 | 0.345/0.392 |
| GS-WGAN | 0.143/0.808 | 0.166/0.658 | 0.590/0.614 | 0.420/0.523 |
| DPGEN | 0.905/0.936 | 0.828/0.878 | 0.700/0.884 | 0.661/0.815 |
| DP-DM (预训练) | 0.952/0.981 | 0.794/0.862 | N/A | N/A |
| **DP-SAD** | **0.962/0.976** | **0.844/0.896** | **0.915/0.928** | **0.826/0.841** |

### 消融实验

权衡系数 $\lambda$ 影响（CelebA，$\varepsilon=10$）：

| $\lambda$ | 0.0 | 0.2 | 0.5 | **1.0** | 2.0 | 4.0 |
|---|---|---|---|---|---|---|
| FID ↓ | 14.63 | 13.41 | 12.38 | **11.68** | 12.11 | 13.55 |
| IS ↑ | 2.12 | 2.18 | 2.26 | **2.37** | 2.35 | 2.31 |

时间步 $T$ 的影响：$T$ 从小到大增加时，IS 持续上升、FID 持续下降，验证了时间步稀释 DP 噪声的有效性。实验中选择 $T=500$ 作为效率与质量的平衡点。

模型条件化消融：student conditioning + discriminator conditioning 组合效果最优。

### 关键发现

- **不需要预训练**仍超越需要预训练的 DP-LDM（FID: 11.26 vs 14.30），证明方法本身的优势
- 在复杂任务（CelebA，$\varepsilon=1$）上，超出无预训练方法至少 **16 个百分点**
- 增大 $T$ 可作为"免费"提升隐私-效用平衡的手段——不增加隐私消耗但改善生成质量
- 方法允许**小 batch size + 大 $T$** 替代大 batch size，使资源受限场景下的训练成为可能

## 亮点与洞察

1. **扩散模型时间步的新用途**：首次发现并利用扩散时间步来稀释 DP 噪声影响，建立了扩散模型结构与隐私保护之间的新联系
2. **严格隐私证明**：通过 Rényi DP 和高斯机制给出完整的隐私界分析
3. **实用性强**：不需要大规模预训练数据集，支持资源受限场景
4. **梯度链式法则+后处理**：巧妙利用 DP 的后处理特性减少对模型参数梯度的噪声注入

## 局限性

- 教师模型使用私有数据无保护训练——虽然不发布但存在潜在风险
- 仅在相对低分辨率（32×32、64×64）上验证，高分辨率效果未知
- 对于无标签数据需要额外的聚类步骤（MoCo + k-means）
- 理论隐私界可能与实际隐私保护水平存在差距

## 评分

- **创新性**: ⭐⭐⭐⭐ — 时间步稀释 DP 噪声的思路新颖，对抗蒸馏框架设计合理
- **实用性**: ⭐⭐⭐⭐ — 无需预训练、支持资源受限、SOTA 性能
- **实验充分度**: ⭐⭐⭐⭐ — 11 个基线、3 个数据集、充分的消融分析
- **论文质量**: ⭐⭐⭐⭐ — 方法推导严谨，隐私分析完整，但写作可以更精炼

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Mutual Learning for Acoustic Matching and Dereverberation via Visual Scene-driven Diffusion](mutual_learning_for_acoustic_matching_and_dereverberation_via_visual_scene-drive.md)
- [\[ECCV 2024\] Realistic Human Motion Generation with Cross-Diffusion Models](realistic_human_motion_generation_with_cross-diffusion_models.md)
- [\[ECCV 2024\] M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models](m2d2m_multi-motion_generation_from_text_with_discrete_diffusion_models.md)
- [\[ECCV 2024\] Learning Trimodal Relation for Audio-Visual Question Answering with Missing Modality](learning_trimodal_relation_for_audio-visual_question_answering_with_missing_moda.md)
- [\[ECCV 2024\] Learning Semantic Latent Directions for Accurate and Controllable Human Motion Prediction](learning_semantic_latent_directions_for_accurate_and_controllable_human_motion_p.md)

</div>

<!-- RELATED:END -->
