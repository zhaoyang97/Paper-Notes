---
title: >-
  [论文解读] Measurement Score-based Diffusion Model (MSM)
description: >-
  [ICLR 2026][图像生成][扩散模型] 不去硬学"干净图像的 score"，而是直接在测量域：里学被子采样、带噪声的"局部测量 score"，再通过随机掩码聚合还原出完整测量——让扩散模型完全用退化观测就能训练，既能无条件生成又能解线性逆问题。 领域现状：score-based 扩散模型靠学 score functi…
tags:
  - "ICLR 2026"
  - "图像生成"
  - "扩散模型"
  - "自监督"
  - "测量域"
  - "无干净数据训练"
  - "MRI 重建"
  - "后验采样"
---

# Measurement Score-based Diffusion Model (MSM)

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=pFByPVh6bd](https://openreview.net/forum?id=pFByPVh6bd)  
**代码**: [https://github.com/wustl-cig/MSM](https://github.com/wustl-cig/MSM)  
**领域**: 图像生成 / 扩散模型 / 逆问题  
**关键词**: 扩散模型, 自监督, 测量域, 无干净数据训练, MRI 重建, 后验采样  

## 一句话总结
不去硬学"干净图像的 score"，而是直接在**测量域**里学被子采样、带噪声的"局部测量 score"，再通过随机掩码聚合还原出完整测量——让扩散模型完全用退化观测就能训练，既能无条件生成又能解线性逆问题。

## 研究背景与动机
**领域现状**：score-based 扩散模型靠学 score function（log 密度梯度）从高维分布采样，在自然图像、医学图像生成上都达到 SOTA，还能改造成条件采样去解逆问题。但训练通常需要大量**干净 ground-truth 图像**。

**现有痛点**：很多场景拿不到干净数据——高分辨率图受硬件限制，MRI 全采样扫描时间长、病人难受。已有工作（Ambient diffusion / SURE-score / GSURE diffusion）都想从退化数据里**直接逼近干净图像的 score**。

**核心矛盾**：从退化数据恢复"全图 score"本身是个**不必要地困难**的目标。测量天然落在某个结构化子空间里，硬要还原全图 score 既别扭又难。更糟的是，一张被破坏的图像并不被测量唯一确定——无穷多图像能映射到同一退化图像，监督信号有歧义。GSURE diffusion 还有两个硬伤：只在单线圈 MRI 上验证过，扩到多线圈需要对完整测量算子做 SVD 计算上不可行；而且要求最小扩散噪声 $\sigma_0$ 必须匹配测量噪声 $\rho$，当 $\rho$ 偏大时采样质量严重退化。

**本文目标**：彻底绕开"还原全图 score"，让扩散模型**只用退化测量**就能训练，并同时支持无条件生成与逆问题求解。

**核心 idea（测量域学习）**：把图像域里成功的 patch-based 学习思想搬到**测量域**——不学全图 score，而学"限制在被观测区域上的局部测量 score"。关键优势在于：每个子采样测量由采集算子**唯一确定**（不像被破坏的图像有歧义），模型学的是物理上有明确定义的去噪输入。

## 方法详解

### 整体框架
MSM 分三块：训练时只在子采样测量上学**局部测量 score**（Tweedie 公式从去噪器导出）；无条件采样时，把多张随机掩码下的局部 score 在期望意义下**聚合**成"MSM score"，逐步重建完整测量 $z$，再映射回图像；逆问题求解时，在采样循环里插入一个数据保真梯度项，把无条件采样改成后验采样。

```mermaid
flowchart TB
    subgraph 训练
    A[退化测量 s=Sz<br/>子采样+可选噪声] --> B[加扩散噪声 s_t=s+σ_t n]
    B --> C[去噪器 D_θ 预测 ŝ_θ]
    C --> D[Tweedie 公式得<br/>局部测量 score S_θ]
    end
    subgraph 采样
    E[完整测量迭代 z_t] --> F[随机抽 w 个掩码 S^i<br/>得局部测量 s_t^i]
    F --> G[各自局部去噪 ŝ_θ^i]
    G --> H[加权聚合 W·ΣS^iᵀŝ_θ^i<br/>→ MMSE 估计 ẑ_θ]
    H --> I[反向扩散一步 z_{t-1}]
    I -.可选插入数据保真.-> H
    end
```

### 关键设计

**1. 局部测量 score：把 score 学习限制到可观测子空间。** 设完整测量 $z\in\mathbb{R}^n$ 由图像唯一决定（自然图像 $z=x$，MRI 取 $z=FCx$，$F$ 为傅里叶变换、$C$ 为线圈灵敏度，更一般地 $z=Tx$ 且 $T$ 可逆）。掩码 $S\in\{0,1\}^{m\times n}$（$m<n$）从分布 $p(S)$ 抽取，得子采样测量 $s=Sz$。对 $s$ 加扩散噪声 $s_t=s+\sigma_t n$，去噪器输出 $\hat s_\theta(s_t;\sigma_t)=D_\theta(s_t;\sigma_t)$，用 MSE 损失训练。训练好后由 Tweedie 公式得到局部 score $S_\theta(s_t;\sigma_t,S)=\frac{1}{\sigma_t^2}(\hat s_\theta(s_t;\sigma_t)-s_t)$，它**显式条件于**生成 $s_t$ 的掩码 $S$。整个过程不碰任何完整测量 $z$ 或干净图像 $x$，这正是自监督的来源。

**2. MSM score：用随机掩码的期望聚合还原全测量 score。** 真正想要的是完整测量上的 score $\nabla\log p_{\sigma_t}(z_t)$，但只学到了局部 score。MSM 把它定义为对所有掩码的期望：$\nabla\log q_{\sigma_t}(z_t):=W\,\mathbb{E}_{S\sim p(S)}\big[S^\top\nabla\log p_{\sigma_t}(s_t\mid S)\big]_{s_t=Sz_t}$，其中转置 $S^\top$ 把局部 score 映回完整测量空间，权重向量 $W=\big[\max(\mathbb{E}_S[\mathrm{diag}(S^\top S)],1)\big]^{-1}$ 按"每个坐标被覆盖的期望次数的倒数"补偿不同掩码间的重叠贡献，逐元素取 max 避免未覆盖区域除零。这个聚合可以解释为**product-of-experts（复合似然）**模型——每张掩码是一个 expert，MSM score 是它们的乘积模型的 score。

**3. 随机采样近似：用 $w$ 个掩码做无偏估计，理论保证收敛。** 期望算不动，于是每步随机抽 $w$ 个掩码 $S^{(i)}$ 做无偏估计 $\nabla\log\hat q_{\sigma_t}(z_t):=W\big[\frac{1}{w}\sum_i S^{(i)\top}\nabla\log p_{\sigma_t}(s_t^{(i)}\mid S^{(i)})\big]$。采样时（Algorithm 1）对每个掩码：局部去噪 → 抽噪声估计 $s_t^{(i)}\sim p(s_t^{(i)}\mid\hat s_\theta^{(i)})$ → **回插更新** $z_t\leftarrow S^{(i)\top}s_t^{(i)}+(I-S^{(i)\top}S^{(i)})z_t$，使后续掩码总在已吸收前面信息的迭代上工作，$w$ 个随机循环互补地精修不同区域。最后聚合成 MMSE 估计 $\hat z_\theta=W\sum_i S^{(i)\top}\hat s_\theta^{(i)}+\mathbf{1}_{C=0}\cdot\hat z_\theta$（未覆盖坐标保留旧值），当作干净预测做标准反向扩散。理论上 $D_{KL}(q\|\hat q)\le\frac{v^2}{w}C$，随机迭代数 $w$ 越大越逼近理想分布。

**4. 后验采样：插一个数据保真梯度把无条件采样变成解逆问题。** 对线性逆问题 $y=Hz+e$（$A=HT$，$H$ 为下采样/模糊/inpainting/随机投影），把后验 score 拆成先验 + 似然：$\nabla\log p_{\sigma_t}(z_t\mid y)\approx\nabla\log\hat q_{\sigma_t}(z_t)+\gamma_t\nabla\|y-H\hat z_\theta\|_2^2$。实现上只需在 Algorithm 1 的聚合估计 $\hat z_\theta$ 上插一步 $\hat z_\theta\leftarrow\hat z_\theta-\gamma_t\nabla_{\hat z_\theta}\|y-H\hat z_\theta\|_2^2$，无需重训练就把预训练的 MSM 先验用于 inpainting、超分、CS-MRI。注意 $H$ 可以和训练时的随机子采样算子 $S$ 不同。

**5. 噪声 + 子采样训练：按扩散/测量噪声大小分两种情形。** 观测带噪 $s=Sz+\nu,\nu\sim\mathcal N(0,\rho I)$ 时，逐步比较扩散噪声 $\sigma_t$ 与测量噪声 $\rho$：当 $\sigma_t>\rho$（实践中大多数步），补残差噪声 $s_t\leftarrow s+\sqrt{\sigma_t^2-\rho^2}\,n$，损失为"用更少噪声参考去噪更噪输入"项加 SURE 损失 $L_{\text{SURE}}$（Stein 无偏风险估计，让模型学会去测量噪声）；当 $\sigma_t\le\rho$，先用 $\rho$ 条件去噪得伪干净参考 $\hat s_\theta(s;\rho)$，再加扩散噪声并在非子采样区域约束一致性。由于 Case 1 更常被采到，伪干净参考会随训练自然变好，保证两种情形下训练稳定。

## 实验关键数据

设置：统一用 Dhariwal & Nichol 扩散架构，单卡 A100 从零训练 1M 步。数据为 69k FFHQ 人脸（128×128 RGB）与 2k fastMRI T2 切片（256×256 复数多线圈），逆问题各 100 张测试图。

### 主实验：无条件生成 FID

| 数据/退化 | 方法 | 人脸 FID↓ | MRI FID↓ |
|---|---|---|---|
| 无退化（上界） | Oracle diffusion（干净训练） | 10.21 | 28.41 |
| 仅子采样 $\rho=0$ | **MSM** | **29.14** | **64.37** |
| 仅子采样 $\rho=0$ | Ambient diffusion | 55.90 | 70.07 |
| 子采样+噪声 $\rho=0.1$ | **MSM** | **37.14** | **82.17** |
| 子采样+噪声 $\rho=0.1$ | GSURE diffusion | 89.71 | （多线圈不可行） |

MSM 在所有"无干净数据"设定下 FID 都大幅低于 Ambient / GSURE；MRI 多线圈场景 GSURE 直接因 SVD 不可行而缺席。

### 逆问题：自然图像 + CS-MRI

| 任务 | 指标 | Input | A-DPS | SSDU | **MSM** |
|---|---|---|---|---|---|
| Inpainting | PSNR↑ | 18.26 | 20.14 | — | **24.71** |
| Inpainting | LPIPS↓ | 0.304 | 0.305 | — | **0.076** |
| SR ×4 | PSNR↑ | 23.21 | 22.61 | — | **28.11** |
| SR ×4 | LPIPS↓ | 0.459 | 0.277 | — | **0.117** |
| CS-MRI ×4 | PSNR↑ | 22.75 | 27.28 | 29.65 | **30.71** |
| CS-MRI ×4 | LPIPS↓ | 0.306 | 0.173 | 0.160 | **0.145** |
| CS-MRI ×6 | PSNR↑ | 21.94 | 26.29 | 28.02 | **28.86** |
| CS-MRI ×6 | LPIPS↓ | 0.342 | 0.201 | 0.186 | **0.168** |

MSM 在 inpainting/超分上全面超过 A-DPS（A-DPS 在 PSNR/SSIM 上甚至不如输入图，因 Ambient 先验在 box 掩码这类非稀疏图样上难补细节）；CS-MRI 上同时超过扩散基线 A-DPS 与重建专用自监督 SSDU。

### 关键发现
- **效率**：A-DPS 需 1000 步，MSM 仅 200 步（逆问题 $w=3$，生成 $w=1$）就更好。
- **$w$ 的作用**：随机循环数 $w$ 越大采样质量越高（与 KL 界一致），但生成时 $w=1$ 已够。
- **泛化掩码**：逆问题里的 $H$（box inpainting、bicubic 超分）与训练子采样掩码不同，仍能直接用预训练 MSM 先验，无需重训。

## 亮点与洞察
- **换问题比换技巧更聪明**：别人都在"如何从退化数据逼近全图 score"上加损失修补，MSM 直接换成"学局部测量 score 再聚合"，把一个病态目标变成定义良好的去噪问题。
- **测量域消歧**：子采样测量被采集算子唯一确定，而被破坏的图像不唯一——在测量域训练天然去掉了监督歧义，这是个被以往工作忽视的关键观察。
- **统一框架**：同一个 MSM score 既支持无条件生成又支持后验采样，逆问题只是"插一个数据保真梯度"，工程上极简。
- **product-of-experts 视角**：把随机掩码聚合解释为复合似然模型的 score，给"为什么聚合局部 score 能逼近全测量 score"提供了优雅解释，并配 KL 收敛界。

## 局限与展望
- **采样成本**：$w$ 个随机循环 × 多步反向扩散，$w$ 大时算力上升，存在质量-时间权衡（虽然 200 步已比 A-DPS 1000 步省）。
- **要求 $z=Tx$ 可逆变换**：依赖完整测量与图像间存在可逆映射（MRI 是 $FC$），对更一般的非线性/欠定采集算子如何推广未明确。
- **噪声训练分情形稍 ad-hoc**：Case 1/Case 2 按 $\sigma_t$ vs $\rho$ 切换并依赖 SURE，伪干净参考质量在训练早期可能不稳。
- **验证规模有限**：实验集中在 FFHQ 人脸与 fastMRI，未在更大/更多样数据集与更多采集模态上验证。

## 相关工作与启发
- **无干净数据扩散训练**：Ambient diffusion（仅子采样）、SURE-score / Daras 2024b（仅带噪）、GSURE diffusion（带噪+子采样）——MSM 与它们的本质区别是不学全图 score。
- **自监督重建**：SSDU / Robust SSDU 用测量子集互相监督训练端到端网络；MSM 把同样的自监督思想引入扩散先验。
- **扩散逆问题求解**：DPS / A-DPS（用退化训练的扩散先验做后验采样）；MSM 的后验采样可看作把 DPS 的数据保真项接到测量域聚合估计上。
- **启发**：patch-based 学习提升可扩展性的思路，在测量域同样成立——"把学习对象限制在物理上唯一定义的子空间"是个可迁移的设计原则，可启发其它带退化观测的生成/重建任务。

## 评分
- **新颖性**: ⭐⭐⭐⭐⭐ 把 score 学习从图像域搬到测量域、用随机掩码期望聚合还原全测量 score，是对"无干净数据扩散训练"问题设定的重新定义，视角新颖且有理论支撑。
- **实验充分度**: ⭐⭐⭐⭐ 覆盖生成（FID）与逆问题（inpainting/超分/CS-MRI），自然图像与多线圈 MRI 双域，baseline 选得切题；但数据集规模与模态多样性偏有限。
- **写作质量**: ⭐⭐⭐⭐ 动机层层递进，从"为何不学全图 score"到测量域消歧讲得清楚，算法与公式完整；噪声训练两情形稍密集。
- **价值**: ⭐⭐⭐⭐⭐ 直击医学影像等拿不到干净数据的真实痛点，框架统一、工程改动小、MRI 上超过专用自监督方法，落地价值高。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Sample-Efficient Evidence Estimation of Score-Based Priors for Model Selection](sample-efficient_evidence_estimation_of_score_based_priors_for_model_selection.md)
- [\[ICLR 2026\] Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency](large_scale_diffusion_distillation_via_score-regularized_continuous-time_consist.md)
- [\[CVPR 2025\] Traversing Distortion-Perception Tradeoff Using a Single Score-Based Generative Model](../../CVPR2025/image_generation/traversing_distortion-perception_tradeoff_using_a_single_score-based_generative_.md)
- [\[ICML 2026\] Alignment-Guided Score Matching for Text-to-Image Alignment in Diffusion Models](../../ICML2026/image_generation/alignment-guided_score_matching_for_text-to-image_alignment_in_diffusion_models.md)
- [\[ICLR 2026\] Latent Diffusion Model without Variational Autoencoder](latent_diffusion_model_without_variational_autoencoder.md)

</div>

<!-- RELATED:END -->
