---
title: >-
  [论文解读] Expressive Score-Based Priors for Distribution Matching with Geometry-Preserving Regularization
description: >-
  [ICML2025][图像生成][Distribution Matching] 提出基于 score function 的表达性先验分布（SAUB），通过 Score Function Substitution (SFS) 技巧绕过先验密度估计，结合 Gromov-Wasserstein 几何保持约束实现稳定高效的分布匹配，在公平分类、域适应和域翻译任务上取得优越表现。
tags:
  - "ICML2025"
  - "图像生成"
  - "Distribution Matching"
  - "Score-based Prior"
  - "VAE"
  - "Gromov-Wasserstein"
  - "几何保持正则化"
  - "CLIP语义空间"
---

# Expressive Score-Based Priors for Distribution Matching with Geometry-Preserving Regularization

**会议**: ICML2025  
**arXiv**: [2506.14607](https://arxiv.org/abs/2506.14607)  
**代码**: [inouye-lab/SAUB](https://github.com/inouye-lab/SAUB)  
**领域**: 生成模型 / 分布匹配  
**关键词**: Distribution Matching, Score-based Prior, VAE, Gromov-Wasserstein, 几何保持正则化, CLIP语义空间

## 一句话总结

提出基于 score function 的表达性先验分布（SAUB），通过 Score Function Substitution (SFS) 技巧绕过先验密度估计，结合 Gromov-Wasserstein 几何保持约束实现稳定高效的分布匹配，在公平分类、域适应和域翻译任务上取得优越表现。

## 研究背景与动机

分布匹配（Distribution Matching, DM）是一种通用的域不变表示学习技术，广泛应用于公平分类、域适应和域翻译等任务。现有方法存在以下问题：

- **非参数方法**（如 MMD、Sinkhorn）：可扩展性差，高维场景效率低
- **对抗方法**（如 GAN）：训练不稳定，易模式崩塌，超参敏感
- **似然方法**（如 VAE）：通常采用固定高斯先验，表达力不足，导致编码器变换被扭曲
- **归一化流先验**：要求隐空间与输入同维度，限制了灵活性
- **LSGM**（潜在 score 生成模型）：需反向传播通过扩散模型 U-Net 计算 Jacobian，低噪声水平下梯度不稳定

核心动机：对于基于梯度的 DM 训练，似然目标实际上**不需要先验密度本身**，只需要其 score function（对数概率梯度）。这一洞察为用 score-based 模型替代显式密度估计打开了大门。

## 方法详解

### 1. VAUB 目标函数

在 VAUB（Variational Alignment Upper Bound）框架下，分布匹配目标可分解为三项：

$$\mathcal{L}_{\text{VAUB}} = \sum_d \left\{ \underbrace{\mathbb{E}_{q_\theta}[-\log p_\varphi(x|z,d)]}_{\text{重建项}} - \underbrace{\mathbb{E}_{q_\theta}[-\log q_\theta(z|x,d)]}_{\text{熵项}} + \underbrace{\mathbb{E}_{q_\theta}[-\log Q_\psi(z)]}_{\text{交叉熵项}} \right\}$$

其中 $q_\theta(z|x,d)$ 为编码器，$p_\varphi(x|z,d)$ 为解码器，$Q_\psi(z)$ 为跨域共享先验。

### 2. Score Function Substitution (SFS) 技巧

**核心创新**：交叉熵项的编码器梯度可通过 score function 等价替换：

$$\nabla_\theta \mathbb{E}_{z_\theta \sim q_\theta(z|x)}[-\log Q_\psi(z_\theta)] = \nabla_\theta \mathbb{E}_{z_\theta}\left[-\left(\nabla_{\bar{z}} \log Q_\psi(\bar{z})\big|_{\bar{z}=z_\theta}\right)^\top z_\theta\right]$$

关键点：score function 评估值在计算编码器梯度时作为**常数**处理（从计算图 detach），因此：

- 无需计算先验密度 $Q_\psi(z)$
- 无需反向传播通过 score 网络（避免 LSGM 的不稳定性）
- 只需一次前向评估 score function

### 3. SAUB 目标（Score-based Prior Alignment Upper Bound）

$$\mathcal{L}_{\text{SAUB}} = \sum_d \mathbb{E}_{z \sim q_\theta(z|x,d)}\left[-\log p_\varphi(x|z,d) + \log q_\theta(z|x,d) - \left(\nabla_{\bar{z}} \log Q_\psi(\bar{z})\big|_{\bar{z}=z}\right)^\top z\right]$$

满足 $\nabla_{\theta,\varphi} \mathcal{L}_{\text{VAUB}} = \nabla_{\theta,\varphi} \mathcal{L}_{\text{SAUB}}$，即对编码器和解码器参数梯度完全一致。

### 4. 交替优化算法

整体训练形式化为双层优化问题：

- **上层**：固定 score 先验，优化编码器 $\theta$ 和解码器 $\varphi$（SAUB 目标）
- **下层**：固定编码器/解码器，用去噪 score matching 更新 score 模型 $\psi$

$$\mathcal{L}_{\text{DSM}} = \mathbb{E}_{q_\theta}\left[\|S_\psi(\tilde{z}, \sigma_i) - \nabla_{\tilde{z}} \log q_{\sigma_i}(\tilde{z}|z)\|_2^2\right]$$

当 score 先验与编码器边际后验完全匹配时，变分界变紧。

### 5. Gromov-Wasserstein 几何保持正则化

总损失加入 GW 约束：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{DM}} + \lambda_{\text{GW}} \cdot \mathbb{E}\left[\|d_X(x,x') - d_Z(z,z')\|_2^2\right]$$

度量空间选择：

- **GW-EP**（欧氏保持）：输入/隐空间均用 L2 距离，适合低维数据
- **GW-SP**（语义保持）：用预训练 CLIP 嵌入计算输入空间距离，适合高维图像数据

## 实验关键数据

### 域适应（MNIST ↔ USPS）

| 模型 | MNIST→USPS | USPS→MNIST |
|------|-----------|-----------|
| ADDA | 89.4% | 90.1% |
| DANN | 77.1% | 73.0% |
| VAUB | 40.7% | 45.3% |
| **Ours w/o GW** | 88.1% | 85.5% |
| **Ours w/ GW-EP** | 91.4% | 92.7% |
| **Ours w/ GW-SP** | **96.1%** | **97.4%** |

GW-SP 变体在双向域适应上全面领先，较 ADDA 提升 +6.7%/+7.3%。

### 域翻译（CelebA 发色转换，Black↔Blonde Hair）

| 任务/模型 | Top-1 检索 | SSIM ↑ | LPIPS ↓ |
|----------|-----------|--------|---------|
| No GW | 5.0% | 0.393 | 0.431 |
| GW-EP | 4.0% | 0.428 | 0.371 |
| **GW-SP** | **9.0%** | **0.542** | **0.285** |

GW-SP 在语义空间中的检索准确率几乎翻倍，LPIPS 大幅降低，证明语义保持效果显著优于欧氏空间。

### 稳定性对比（SFS vs LSGM）

- LSGM 在 $\sigma_{\min}=0.001$ 时出现灾难性不稳定（NLL 发散、重建损失飙升）
- SFS 在各噪声水平下均保持稳定，$\sigma_{\min}=0.01$ 时 NLL 优于 LSGM 的最佳结果
- SFS 无需反向传播 score 网络，显存效率更高

### 公平分类（UCI Adult 数据集）

在相同 Demographic Parity gap 下，SAUB+GW-EP 的准确率优于 FCRL、CVIB、VAUB 和 LAFTR-DP 等基线，实现近零 DP gap 的同时保持高准确率。

## 亮点与洞察

1. **SFS 技巧极为简洁**：核心思想仅一个公式——将交叉熵梯度替换为 score function 的 detach 内积，却彻底规避了显式密度估计和 Jacobian 计算
2. **与 LSGM 的关键区别**：LSGM 需通过 U-Net 反向传播（估计 Hessian），低噪声下不稳定；SFS 只需前向评估 score，梯度始终稳定
3. **GW-SP 是实用创新**：将 CLIP 语义嵌入引入 GW 距离度量空间，解决了高维图像中欧氏距离语义不足的老问题
4. **框架通用性强**：同一框架适配公平分类（表格数据）、域适应（手写数字）、域翻译（人脸图像）三类完全不同的任务
5. **合成实验说服力强**：20 个样本即可实现良好隐空间分离，Gaussian/MoG 先验需要 100+ 样本

## 局限与展望

1. **图像生成质量有限**：域翻译实验用的是简单 VAE+扩散架构，图像质量离 SOTA 有差距，作者也承认这是 proof-of-concept
2. **交替优化理论保证不足**：采用简单交替优化近似双层优化，缺乏收敛性分析
3. **GW-SP 依赖预训练 CLIP**：语义保持效果受限于 CLIP 模型质量，且不适用于非视觉数据
4. **实验规模偏小**：域适应只做了 MNIST↔USPS，未在 VisDA、DomainNet 等大规模 benchmark 验证
5. **score 模型训练开销**：虽然比 LSGM 高效，但交替训练 score 网络仍增加了整体训练时间
6. **仅验证二域场景**：多域（>2）分布匹配的扩展性和性能未被讨论

## 评分

- 新颖性: ⭐⭐⭐⭐ — SFS 技巧巧妙且实用，score-based prior + GW-SP 组合新颖
- 实验充分度: ⭐⭐⭐ — 覆盖多任务但规模偏小，缺少大规模 benchmark
- 写作质量: ⭐⭐⭐⭐ — 数学推导清晰，motivation 层层递进
- 价值: ⭐⭐⭐⭐ — 为分布匹配提供了稳定高效的新范式，score prior 思路可推广

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Elucidating Flow Matching ODE Dynamics via Data Geometry and Denoisers](elucidating_flow_matching_ode_dynamics_with_respect_to_data_geometries_and_denoi.md)
- [\[ICCV 2025\] Unsupervised Imaging Inverse Problems with Diffusion Distribution Matching](../../ICCV2025/image_generation/unsupervised_imaging_inverse_problems_with_diffusion_distribution_matching.md)
- [\[ICCV 2025\] Balanced Image Stylization with Style Matching Score](../../ICCV2025/image_generation/balanced_image_stylization_with_style_matching_score.md)
- [\[ICCV 2025\] Learning Few-Step Diffusion Models by Trajectory Distribution Matching](../../ICCV2025/image_generation/learning_few-step_diffusion_models_by_trajectory_distribution_matching.md)
- [\[CVPR 2025\] SGMatch: Semantic-Guided Non-Rigid Shape Matching with Flow Regularization](../../CVPR2025/image_generation/sgmatch_semantic-guided_non-rigid_shape_matching_with_flow_regularization.md)

</div>

<!-- RELATED:END -->
