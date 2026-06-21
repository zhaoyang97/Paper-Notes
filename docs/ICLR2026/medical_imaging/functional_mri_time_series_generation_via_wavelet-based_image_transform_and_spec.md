---
title: >-
  [论文解读] Functional MRI Time Series Generation via Wavelet-Based Image Transform and Spectral Flow Matching for Brain Disorder Identification
description: >-
  [ICLR 2026][医学图像][fMRI 生成] DSFM 把 fMRI 的 BOLD 时间序列先经小波变换（DWT）转成多尺度时-频图像、再经分块 DCT 压到低频稀疏域，然后在 DCT 频域里用"热扩散式"流匹配做类别条件生成，反变换回时域 BOLD 信号做数据增强，从而提升下游脑疾病的功能连接（FC）分类性能。
tags:
  - "ICLR 2026"
  - "医学图像"
  - "fMRI 生成"
  - "BOLD 信号"
  - "离散小波变换 (DWT)"
  - "离散余弦变换 (DCT)"
  - "流匹配 (Flow Matching)"
  - "脑网络分类"
---

# Functional MRI Time Series Generation via Wavelet-Based Image Transform and Spectral Flow Matching for Brain Disorder Identification

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=Dgphd9qizu](https://openreview.net/forum?id=Dgphd9qizu)  
**代码**: [https://github.com/htew0001/DSFM](https://github.com/htew0001/DSFM)  
**领域**: 医学影像 / fMRI 生成 / 脑疾病分类  
**关键词**: fMRI 生成, BOLD 信号, 离散小波变换 (DWT), 离散余弦变换 (DCT), 流匹配 (Flow Matching), 脑网络分类  

## 一句话总结
DSFM 把 fMRI 的 BOLD 时间序列先经小波变换（DWT）转成多尺度时-频图像、再经分块 DCT 压到低频稀疏域，然后在 DCT 频域里用"热扩散式"流匹配做类别条件生成，反变换回时域 BOLD 信号做数据增强，从而提升下游脑疾病的功能连接（FC）分类性能。

## 研究背景与动机
- **领域现状**：fMRI 通过 BOLD 信号无创地观测大脑动态，是诊断抑郁症、自闭症等神经精神疾病的关键模态；但 fMRI 采集昂贵、样本少且类别不均衡，制约了数据驱动脑分析模型的泛化能力，因此生成模型被用来做数据增强。
- **现有痛点**：① 主流方法在**功能连接（FC）矩阵空间**直接生成（如 DCGAN、BrainFC-CGAN），但 FC 把信号依赖压成一张静态相关矩阵，丢掉了脑网络的瞬态状态与跨频交互；② 转向**时域时序生成**的方法（Diffusion-TS、FM-TS）又难以解耦生理性波动（心跳、呼吸、运动伪影），还原不出多尺度振荡；③ 把时序当**图像生成**的 T2I-Diff 用固定分辨率的 STFT，丢失细粒度瞬态、衰减频率调幅，导致图像→信号重建有伪影、增益有限。
- **核心矛盾**：要忠实复现 BOLD 信号的非平稳性、时空动态与生理变化，单一的 FC / 原始时序 / 固定分辨率时频表示都不够，需要一个能**同时刻画全局多尺度趋势与局部能量压缩**的表示。
- **本文目标**：构造一种兼顾全局与局部的双谱表示，并在该谱域里设计高效、与频率层级对齐的生成过程，最终提升下游脑疾病分类。
- **核心 idea**：**【双谱级联】** 用 DWT 抓全局多尺度瞬态、再用分块 DCT 抓局部低频能量压缩；**【频域流匹配】** 把扩散模型"先去高频后去低频"的频域自回归性质，落到 DCT 域的热扩散过程，用 ODE 流匹配做粗到细、与频率层级对齐的高效生成。

## 方法详解

### 整体框架
DSFM（Dual-Spectral Flow Matching）是一条 6 步流水线：从 ROI 提取 BOLD 时序 → DWT 多分辨率分解成时-尺度 scalogram 图像 → 分块 2D DCT 做局部谱编码 → 在 DCT 域用 U-ViT 学习速度场并 ODE 采样生成类别条件样本 → 经 IDCT+IDWT 反变换回时域 BOLD 信号 → 用合成信号做数据增强、构建 FC 矩阵、训练分类器并评估。关键在于把"时序生成"重塑成"双谱图像生成"，让结构化的频率先验贯穿生成全程。

```mermaid
flowchart LR
    A[ROI BOLD 时序<br/>D×T] --> B[DWT 多分辨率分解<br/>时-尺度 scalogram]
    B --> C[分块 2D DCT<br/>局部低频能量压缩]
    C --> D[Zig-zag 展平<br/>低→高频排序]
    D --> E[DCT 域谱流匹配<br/>U-ViT 速度场 + ODE 采样]
    E --> F[IDCT + IDWT<br/>反变换回时域 BOLD]
    F --> G[数据增强 → FC 构建 → 分类]
```

### 关键设计

**1. 双谱图像变换（DWT → 分块 DCT）：把时序铺成兼顾全局与局部的频率图。** 给定 $x_s \in \mathbb{R}^{D\times T}$（$D$ 个 ROI、$T$ 个时间点），先用离散小波变换 $W(k,j)=\sum_n x(n)\,\psi_{j,k}[n]$（采用 Haar 基、5 级分解）把每条 BOLD 信号拆成多尺度子带，再把各子带上采样到原始时长并沿尺度轴堆叠，得到张量 $W(i,j,k)\in\mathbb{R}^{D\times T_\psi\times C}$，同时捕捉低频趋势与高频瞬态，并做逐分量归一化拉大高低系数对比，使时序变成"保谱"的多通道图像。随后把每个尺度子带切成不重叠的 $B\times B$ 块，对每块做 2D type-II DCT $D^{(k)}(u,v)=\alpha(u)\alpha(v)\sum_{x,y}W^{(k)}(x,y)\cos\!\frac{(2x+1)u\pi}{2B}\cos\!\frac{(2y+1)v\pi}{2B}$，利用 DCT 在 ROI-时间空间的低频能量压缩保留主结构、滤掉高频噪声。重建时按"块内 IDCT 拼回子带 → IDWT 反变换回时域"逐级还原，保证全局与局部谱特性都不丢。

**2. DCT 域谱流匹配（把扩散热扩散过程搬进频域并配 ODE）：高效、粗到细、与频率层级对齐的生成。** 出发点是扩散模型在频域近似自回归——前向过程先消高频、再渐次消低频，这一性质在 DCT 域同样成立，且 DCT 兼具实值正交、低频能量压缩、与分块架构兼容三大优势。作者以热扩散 SPDE $\mathrm{d}x_t(c)=\eta(t)\Delta_c x_t(c)\,\mathrm{d}t+G(t)\,\mathrm{d}W(t)$ 替代各向同性扩散，并利用拉普拉斯算子可被 DCT 基对角化 $\Delta_c=V\Lambda V^T$，把过程变换到 DCT 域得到逐模态解耦的 $\mathrm{d}z_t=-\eta(t)\Lambda z_t\,\mathrm{d}t+G(t)\,\mathrm{d}W(t)$；再经 zig-zag 展平把二维 DCT 系数排成低→高频的一维序列。逆时概率流 ODE 因此可逐模态分解为 $\frac{\mathrm{d}z_t[k]}{\mathrm{d}t}=-\eta(t)\lambda_k z_t[k]-\frac12 g(t,k)^2\nabla_{z_t[k]}\log p(z_t)$。两条命题把该 mode-wise 概率流 ODE 与流匹配的条件速度场 $v(z_t|z_0;t,k)=\dot\mu(t,k)z_0[k]+\dot\sigma(t,k)\epsilon$ 建立等价，训练即最小化条件谱流匹配损失 $\mathcal{L}_{\text{CSFM}}(\theta)=\mathbb{E}\,\lVert v_\theta(z_t;t,k)-v(z_t|z_0;t,k)\rVert^2$。该损失在 $\mu(t)=1-t,\sigma(t)=t$ 时退化为标准流匹配，因此 DSFM 把流匹配推广成 DCT 域的热扩散过程；ODE 采样相比扩散 SDE 的上千步可压到 20/50/100 步，速度大幅提升。

**3. 类别条件生成 + U-ViT 速度场：让合成信号服务于脑疾病判别。** 速度场 $v_\theta$ 由 U-ViT 参数化，并通过 classifier-free guidance 注入类别标签 $c$（健康对照 / 患者），训练时以概率 $p_\varnothing$ 把 $c$ 替换成空标记 $\varnothing$ 联合训练条件与无条件模型，采样时用引导权重组合两者输出，再用自适应 ODE 求解器积分得到 DCT 样本。均值调度取方差保持（VP）余弦调度、$\tau(t)=\sigma_{\max}\sin^2(\tfrac{\pi}{2}t)$ 且 $\sigma_{\max}=20$。生成样本经反变换得到合成 BOLD 信号，再用 Ledoit-Wolf 收缩协方差估计构建稀疏（保留最强 40% 连接）FC 矩阵，喂给下游分类器，把生成质量直接转化为分类增益。

## 实验关键数据

### 主实验
- **数据集**：NetSim（仿真，50 通道）、MDD（REST-meta-MDD，250 HC / 227 MDD，AAL 116 ROI、232 时间点）、ABIDE（488 ASD / 537 NC，Schaefer 100 ROI、200 时间点）。
- **无条件生成（NetSim，越低越好）**：DSFM 取得 cFID **0.105±.006**（对比 Diffusion-TS 0.193、T2I-Diff 1.384、TimeGAN 14.449），整体生成保真度领先。

**下游分类（MDD，AAL 图谱，与不增强 / 各类生成基线对比）**

| 方法 | Context-FID ↓ | Accuracy ↑ | F1 ↑ | ROC ↑ |
|---|---|---|---|---|
| 不增强（Real） | — | 58.90 | 58.39 | 59.00 |
| 2D-DCGAN (FC) | — | 62.88 | 62.48 | 62.67 |
| TimeGAN | 4.98 | 66.78 | 66.48 | 67.26 |
| Diffusion-TS | 2.06 | 67.29 | 67.21 | 64.57 |
| T2I-Diff | 7.45 | 66.87 | 66.83 | 67.26 |
| **DSFM (Ours)** | **1.51** | **70.84** | **70.77** | **71.49** |

**下游分类（ABIDE，Schaefer 图谱）**

| 方法 | Context-FID ↓ | Accuracy ↑ | F1 ↑ | ROC ↑ |
|---|---|---|---|---|
| 不增强（Real） | — | 64.67 | 64.12 | 67.28 |
| Diffusion-TS | 0.51 | 66.60 | 66.58 | 68.85 |
| T2I-Diff | 0.82 | 69.69 | 69.65 | 71.88 |
| **DSFM (Ours)** | **0.07** | **71.54** | **70.98** | **73.78** |

### 消融实验（MDD 频段子带分析，相对全频段的掉点幅度）

| 设置 | 使用子带 | Accuracy ↑ | ROC 掉点 |
|---|---|---|---|
| Full-band | LH1–LH5 + LL | 70.84 | — |
| Low-pass | LH3–5 + LL | 66.89 | -7.97% |
| Mid-pass | LH1,2,5 + LL | 63.30 | -15.50% |
| High-pass | LH1–4 | 65.40 | -11.0% |
| Band-pass 3 | LH2–5 | 66.88 | -5.20% |

### 关键发现
- **双谱表示带来一致增益**：DSFM 在 FID/cFID/相关性等生成指标与三个数据集的分类指标上均超越 FC 类、时序类、时频图像类基线；ABIDE 上 Context-FID 降到 0.07，分类准确率较不增强提升约 7 个百分点。
- **频段都重要、中频最关键**：去掉任意子带都掉点，Mid-pass（只留部分中频+低频）掉得最狠（ROC -15.5%），说明全频段联合建模对脑疾病判别不可或缺，单纯低频不足以刻画瞬态。
- **采样高效**：ODE 流匹配 20–100 步即可生成高质量样本，避免扩散 SDE 的上千步开销。

## 亮点与洞察
- **把"时序生成"重塑成"双谱图像生成"**：DWT 抓全局多尺度、分块 DCT 抓局部低频压缩，两级级联给生成过程施加了结构化频率先验，比单一 STFT 时频图更细腻。
- **理论桥接漂亮**：用拉普拉斯算子被 DCT 基对角化的事实，把热扩散 SPDE 化成 DCT 域逐模态解耦的 ODE，并以两条命题严格把概率流 ODE 与流匹配条件速度场对应起来，使"频域自回归"性质有了可训练的流匹配实现。
- **生成质量直接兑现为临床增益**：不停留在 FID，而是用合成信号构建 FC 跑下游脑疾病分类，三数据集一致提升，验证了"生成是为了更好诊断"。

## 局限与展望
- **依赖固定小波基与分块设置**：Haar 基、5 级分解、$B\times B$ 分块为人工设定，对不同采集协议/图谱的鲁棒性与自适应性有待验证。
- **数据规模仍有限**：MDD/ABIDE 各几百例，类别二分类，跨站点异质性与多疾病/多类别推广尚未充分检验。
- **生理可解释性待深挖**：虽强调保留生理动态，但未系统验证生成信号在心跳/呼吸/运动伪影等具体生理成分上的可分离与真实性。
- **计算与谱设计耦合复杂**：双谱级联 + DCT 域流匹配 + U-ViT 引入较多超参（$\sigma_{\max}$、调度、引导权重等），落地调参成本不低。

## 相关工作与启发
- **FC 空间生成**：DCGAN、BrainFC-CGAN 保持连接组结构/受试者身份，但静态相关矩阵丢失瞬态网络状态。
- **时域时序生成**：Diffusion-TS（DDPM）、FM-TS（流匹配加速）把焦点从 FC 移到原始时序，但难解耦生理波动。
- **时序当图像**：T2I-Diff、ImagenTime 把时序变图像生成，本文沿此路线但用 DWT+DCT 取代固定分辨率 STFT。
- **频域扩散/流匹配**：借鉴扩散在频域近似自回归（先去高频）的观察与热扩散过程（Rissanen et al.），并用 U-ViT、classifier-free guidance、OT-CFM 等组件实现；启发是**"频率层级 = 生成顺序"**这一先验可被显式编码进生成动力学。

## 评分
- **新颖性**: ⭐⭐⭐⭐ — DWT+DCT 双谱级联 + DCT 域热扩散流匹配的组合在 fMRI 生成中是首创，理论桥接（拉普拉斯对角化↔逐模态 ODE↔流匹配）扎实。
- **实验充分度**: ⭐⭐⭐⭐ — 覆盖仿真+两个真实脑疾病数据集，含生成质量、下游分类、频段消融、NFE 步数对比，但数据规模偏小、跨站点泛化未充分展开。
- **写作质量**: ⭐⭐⭐⭐ — 动机层层递进，公式与命题清晰，图 1 流水线完整；部分推导细节放入补充材料。
- **价值**: ⭐⭐⭐⭐ — 直击 fMRI 样本稀缺痛点，把生成增益落到临床分类提升，对神经影像数据增强有实际意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Learning Patient-Specific Disease Dynamics with Latent Flow Matching for Longitudinal Imaging Generation](learning_patient-specific_disease_dynamics_with_latent_flow_matching_for_longitu.md)
- [\[CVPR 2026\] Continual Learning for fMRI-Based Brain Disorder Diagnosis via Functional Connectivity Matrices Generative Replay](../../CVPR2026/medical_imaging/forge_continual_learning_for_fmri_based_brain_disorder_diagnosis.md)
- [\[ICLR 2026\] CRONOS: Continuous time reconstruction for 4D medical longitudinal series](cronos_continuous_time_reconstruction_for_4d_medical_longitudinal_series.md)
- [\[NeurIPS 2025\] Self-Supervised Learning via Flow-Guided Neural Operator on Time-Series Data](../../NeurIPS2025/medical_imaging/self-supervised_learning_via_flow-guided_neural_operator_on_time-series_data.md)
- [\[AAAI 2026\] Ambiguity-aware Truncated Flow Matching for Ambiguous Medical Image Segmentation](../../AAAI2026/medical_imaging/ambiguity-aware_truncated_flow_matching_for_ambiguous_medica.md)

</div>

<!-- RELATED:END -->
