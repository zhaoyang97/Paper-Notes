---
title: >-
  [论文解读] Flow Along the $K$-Amplitude for Generative Modeling
description: >-
  [ICLR 2026][图像生成][K-Flow] 本文提出 K-Flow，把流匹配的"时间"重新解释为组织频率/尺度的**标度参数 $k$**，让生成沿着 K-amplitude（频带/系数）空间从低频到高频逐级展开，从而获得天然的尺度可控生成能力（类条件可省、频段可编辑、免训练复原），并在图像生成上取得有竞争力的 FID。
tags:
  - "ICLR 2026"
  - "图像生成"
  - "K-Flow"
  - "Flow Matching"
  - "频域生成"
  - "小波变换"
  - "傅里叶变换"
  - "PCA"
  - "可控生成"
---

# Flow Along the $K$-Amplitude for Generative Modeling

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=O224NIizhz](https://openreview.net/forum?id=O224NIizhz)  
**代码**: 待确认  
**领域**: 图像生成 / Flow Matching / 频域生成  
**关键词**: K-Flow, Flow Matching, 频域生成, 小波变换, 傅里叶变换, PCA, 可控生成  

## 一句话总结
本文提出 K-Flow，把流匹配的"时间"重新解释为组织频率/尺度的**标度参数 $k$**，让生成沿着 K-amplitude（频带/系数）空间从低频到高频逐级展开，从而获得天然的尺度可控生成能力（类条件可省、频段可编辑、免训练复原），并在图像生成上取得有竞争力的 FID。

## 研究背景与动机
- **领域现状**：流匹配（FM）已成为最前沿的生成范式，它在像素/隐空间里学习一个时间相关的速度场，把高斯噪声连续地搬运到数据分布。自然数据有内在的频率结构——能量主要集中在低频带，且经验上 DDPM 倾向于"先恢复低频、后补高频"。
- **现有痛点**：常规 FM 的去噪路径并不严格遵循频率渐进顺序，其频率演化既不可量化也不可控（论文图 S2）。模型内部虽然隐式地学到了"低分辨率特征/高分辨率特征"，但**各分辨率之间的边界是模糊的**——你无法说清楚推理过程的哪一步对应哪个频段，因此也无法对频率做精细干预（频段编辑、复原、按尺度施加条件）。
- **核心矛盾**：数据天然是分层的（多尺度/多频段），但生成过程却把这一层级结构压扁在一个不可解释的时间轴上，导致"想控某个频段"无从下手。
- **本文目标**：建立一个能在频域做细粒度控制、同时生成质量不输常规 FM 的范式，把"频率/尺度的渐进展开"显式编码进生成路径。
- **核心 idea**：**将流匹配的时间 $t$ 替换为标度参数 $k$**——$k$ 是组织投影系数（频带）的统一标尺，amplitude 是这些系数的范数。生成不再沿时间走，而是沿 K-amplitude 从 $k=0$（纯噪声）单调走到 $k_{max}$（完整频谱），自然得到一条"低频先生、高频后补"的可控路径。

## 方法详解

### 整体框架
K-Flow 由两大组件构成：**K-amplitude 分解**（一族可逆线性变换 $\mathcal{F}$，把数据从空间域投影到由标度参数 $k$ 组织的频带空间，本文实例化为 Fourier / Wavelet / PCA 三种）与**沿 $k$ 的流匹配**（构造一个随 $k$ 单调展开各频带、其余用噪声填充的随机插值，再用条件流匹配学习其速度场）。推理时从全噪声出发，令 $k$ 由小到大行进，逐步把低频到高频的系数从噪声中"长"出来，最后用 $\mathcal{F}^{-1}$ 逆变换回像素。

```mermaid
flowchart LR
    A[数据 φ] -->|K-amplitude 变换 F| B[频带系数 φ_k 按 k 分组]
    B --> C[随机插值: k 以下用真系数,<br/>k 以上用噪声 ε 填充]
    C --> D[条件流匹配学习<br/>局部速度场 dΨ_k/dk]
    D -->|推理 k:0→k_max| E[逐频带从噪声展开]
    E -->|逆变换 F⁻¹| F[生成图像]
```

### 关键设计

**1. K-amplitude 分解：用一个标量 $k$ 统一组织所有多尺度变换。** 任何完备基 $\{e_j\}$ 都可以按某个标度参数 $k$ 把基切成若干子集 $\{e_k\}$，于是信号写成 $\phi = \sum_k \phi_k$，其中 $\phi_k$ 是落在第 $k$ 个频带的分量、其范数即 "K-amplitude"。以三维傅里叶为例，作者把高维频率向量 $(k_x,k_y,k_z)$ 压成一维标量 $k=\sqrt{k_x^2+k_y^2+k_z^2}$（傅里叶空间里"扩张球"的半径），同一个 $k$ 上的所有频率分量被归为一组。这个抽象的妙处在于：Fourier、Wavelet、PCA 三种本质不同的变换都能套进同一个 $\phi=\sum_k\phi_k$ 框架——只要 $\mathcal{F}$ 线性可逆即可，方法对具体变换是无关的（agnostic）。

**2. K-amplitude 随机插值：把"频带逐级显现"写成可微的连续流。** 离散情形下 $k$ 只取格点上的整数值，作者通过**噪声填充**构造离散流 $\varphi_k = \mathcal{F}^{-1}\big(\mathbb{I}_{k'\le k}\cdot\mathcal{F}\{\phi\} + (1-\mathbb{I}_{k'\le k})\cdot\epsilon\big)$：标度 $k$ 以内用数据真实系数、以外用噪声 $\epsilon$ 顶替，满足 $\lim_{k\to k_{max}}\varphi_k=\phi$、$\varphi_0$ 为可处理的先验。为了能用流匹配（需要对 $k$ 求导），再用一个**碰撞函数** $\mu(t)$（$t=k-\lfloor k\rfloor$，满足 $\mu(0)=1,\mu(1)=0,\mu'(0)=-\mu'(1)$）在相邻整数频带之间做 $\mu(t)\cdot\mathcal{F}\{\phi\}+(1-\mu(t))\cdot\epsilon$ 的线性过渡，其反对称导数保证 $\Psi_k$ 对 $k$ 处处可微。

**3. 局部化速度场：把每一步的优化限制在低维子流形。** 不直接建模 $\Psi_k$，而是学其条件梯度场 $\frac{d\Psi_k}{dk}$。由插值式求导可得条件速度场 $\frac{d\Psi_k}{dk}(\phi,\epsilon)=\mathcal{F}^{-1}\big(\mathbb{I}_{k'\in[\lfloor k\rfloor,\lfloor k\rfloor+1)}\cdot\mu'(t)(\epsilon-\mathcal{F}\{\phi\})\big)$，训练目标为条件流匹配 $\mathcal{L}_{\text{K-Flow}}=\mathbb{E}\int_0^K\|\frac{d\Psi_k}{dk}-v_k(\Psi_k,\theta)\|^2$。关键观察是这个速度场**天然只在 $\sqrt{k_x^2+k_y^2+k_z^2}\in[\lfloor k\rfloor,\lfloor k\rfloor+1)$ 这条窄频带上非零**——即每一步重建只涉及当前 $k$ 附近的一小撮系数。相比像素空间里整张图一起动，K-Flow 把每步的优化约束在一个低维子流形上，降低了优化空间的维度。实践中还可把区间从 $[\lfloor k\rfloor,\lfloor k\rfloor+1)$ 放宽到 $[k_m,k_n)$，把频谱切成两段或三段以平衡效率。

**4. 三种实例化变换：从无数据先验到数据自适应。** Fourier 处理标度局部化的全局频率；Wavelet 借多分辨率分析（用尺度函数 $\omega$ 与小波 $\psi$），既频段局部又空间局部，本文用离散小波（如 db6）；PCA 则是**数据相关**的分解——前两者与数据无关，PCA 把主成分按能量排序当作"频带"，能捕获数据集特有的低维结构。三者共享线性可逆这一前提，因此可无缝接入同一套 K-Flow 流程。

## 实验关键数据

### 主实验表格
CelebA-HQ 256×256 无条件生成（与 LFM 共用同一 VAE 隐空间，K-Flow 用 MoE 版 DiT-L/2 骨干）：

| 模型 | FID↓ | Recall↑ |
|------|------|---------|
| K-Flow, Wave-DiT L/2（本文 db6） | **4.99** | 0.46 |
| K-Flow, Fourier-DiT L/2（本文） | 5.11 | 0.47 |
| K-Flow, PCA-DiT L/2（本文） | 5.19 | 0.48 |
| LFM, DiT L/2 | 5.28 | 0.48 |
| LDM | 5.11 | 0.49 |
| WaveDiff | 5.94 | 0.37 |
| FM | 7.34 | - |

ImageNet 256×256 类条件生成：

| 模型 | FID↓ | Recall↑ |
|------|------|---------|
| K-Flow, Fourier-DiT L/2 + cfg=1.5 | **2.73** | 0.45 |
| K-Flow, PCA-DiT L/2, cfg=1.5 | 4.19 | 0.43 |
| LFM, DiT L/2 + cfg=1.5 | 2.85 | 0.42 |
| LDM-8-G | 7.76 | 0.35 |
| VAR-d16 (cfg=2.0) | 3.30 | 0.51 |
| FlowAR-L (cfg=2.4, 更大模型/不同 VAE) | 1.90 | 0.57 |

### 消融实验表格
围绕"标度可控性"做了三组消融，核心由 CDR（Conditional Discrimination Ratio，越接近 1 越说明省条件后性能不掉）量化：

| 实验设置 | 现象 / 指标 |
|----------|------------|
| 类条件 drop（最后 70% 标度步不给类别） | K-Flow CDR ≈ 1.49（接近 1，几乎不退化）；LFM CDR = 3.25（明显退化、画面模糊） |
| 保高频改低频（固定高标度噪声） | 同组图像五官细节一致、背景/性别/年龄/发型变化——频带↔语义对齐 |
| 图像复原（超分/去模糊） | CelebA 上 PSNR/SSIM 达 SOTA（免训练，附录 Table S6） |

### 关键发现
- **语义被编码进低频**：类别这类高层语义集中在低 K-amplitude 频带，所以推理后期省略类条件几乎不影响质量——意味着可在合成后段省去条件输入以提效。
- **频带天然对应语义属性**：高频锁五官细节、低频管背景/整体外观，从而实现免微调（finetuning-free）的无监督可控编辑；同样协议用在普通 LFM 上则看不到这种频率-语义对应。
- **多样性更优**：低标度期路径维度更高，使 K-Flow 在 Recall 上普遍优于标准 LFM。

## 亮点与洞察
- **概念统一漂亮**：用一个标量标度参数 $k$ 把 Fourier / Wavelet / PCA 三种异质变换统一进同一套流匹配，且方法对变换选择 agnostic，扩展性强。
- **"时间即尺度"的重解释**：把 FM 的时间轴换成有物理含义的频率渐进轴，让一直隐式存在的"低频先生高频后补"变成显式、可量化、可干预的路径。
- **可控性是免费的副产品**：尺度解耦让类条件可省、频段可编辑、复原免训练，这些能力都不需额外模块，源自路径设计本身。
- **局部化降复杂度**：速度场天然窄带化，每步只在低维子流形更新，给出了一个与像素空间 FM 不同的优化视角。

## 局限与展望
- **仅在图像上验证**：未覆盖多模态/密集 caption 引导的大规模生成，class-dropping 实验若扩到文本对齐能更好展示可控性。
- **能量视角未深挖**：作者列出 K-Flow 的六条性质（含 amplitude 对应能量），但与能量模型（EBM）结合的潜力只是点到为止。
- **依赖预训练 VAE 隐空间**：当前 K-分解作用在现成 VAE latent 上；若换更适合频域分解的输入表示（如 RGB→YCbCr + 稀疏 DCT）需重训自编码器。
- **统一理解与生成是开放方向**：低频编码全局语义，作者认为可把理解任务的预训练表示接入生成，迈向统一视觉建模，但尚未实现。

## 相关工作与启发
- **Flow Matching / Rectified Flow / Stochastic Interpolants**：K-Flow 是随机插值框架的一个新实例，把插值的"进度"沿频带组织。
- **频域/多尺度生成（WaveDiff、多分辨率扩散）**：以往多把小波/频率当作架构 trick 或正则；本文把频率渐进直接做成生成路径的主轴。
- **多尺度自回归（VAR、FlowAR）**：同样信奉 coarse-to-fine，但走自回归；K-Flow 用连续流给出 coarse-to-fine 的另一条路，且天生支持频段级编辑。
- **启发**：把"时间"替换成任何有结构的单调标尺（频率、分辨率、能量、甚至语义层级），都可能解锁新的可控生成路径——这是一个可迁移到其他模态的设计模式。

## 评分
- **新颖性**: ⭐⭐⭐⭐⭐ "时间即标度参数"的重解释加上 Fourier/Wavelet/PCA 三合一的统一框架，视角清新且自洽。
- **实验充分度**: ⭐⭐⭐⭐ 无条件/类条件生成 + 三组可控性消融 + 复原任务，覆盖面好；但仅限图像、缺更大规模与多模态验证。
- **写作质量**: ⭐⭐⭐⭐ 概念层层递进、公式与图清晰；六条性质等细节挪到附录，正文略需对照附录阅读。
- **价值**: ⭐⭐⭐⭐ 提供了一个原生支持频段级可控生成与免训练编辑/复原的通用范式，对可控生成与统一视觉建模有启发。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] LapFlow: Laplacian Multi-scale Flow Matching for Generative Modeling](lapflow_laplacian_multi-scale_flow_matching_for_generative_modeling.md)
- [\[ICLR 2026\] Mirror Flow Matching with Heavy-Tailed Priors for Generative Modeling on Convex Domains](mirror_flow_matching_with_heavy-tailed_priors_for_generative_modeling_on_convex_.md)
- [\[ICLR 2026\] GenCP: Towards Generative Modeling Paradigm of Coupled Physics](gencp_towards_generative_modeling_paradigm_of_coupled_physics.md)
- [\[CVPR 2026\] RecTok: Reconstruction Distillation along Rectified Flow](../../CVPR2026/image_generation/rectok_reconstruction_distillation_along_rectified_flow.md)
- [\[ICLR 2026\] Partition Generative Modeling: Masked Modeling Without Masks](partition_generative_modeling_masked_modeling_without_masks.md)

</div>

<!-- RELATED:END -->
