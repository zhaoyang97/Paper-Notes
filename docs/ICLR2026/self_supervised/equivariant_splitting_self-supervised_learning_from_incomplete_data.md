---
title: >-
  [论文解读] Equivariant Splitting: Self-supervised learning from incomplete data
description: >-
  [ICLR 2026][自监督学习][逆问题] 把"等变成像（EI）"的不变性先验和"测量拆分（splitting）"的高效无偏特性结合起来，提出 Equivariant Splitting (ES) 损失，让网络仅凭**单个高度欠采样前向算子**下的不完整测量也能训出逼近 MMSE 的重建器，且无需 EI 那样多次前向评估。
tags:
  - "ICLR 2026"
  - "自监督学习"
  - "逆问题"
  - "等变网络"
  - "测量拆分"
  - "MMSE 估计"
---

# Equivariant Splitting: Self-supervised learning from incomplete data

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=upMIVpe467](https://openreview.net/forum?id=upMIVpe467)  
**代码**: [vsechaud/Equivariant-Splitting](https://github.com/vsechaud/Equivariant-Splitting)  
**领域**: 自监督学习 / 逆问题重建  
**关键词**: 自监督学习, 逆问题, 等变网络, 测量拆分, MMSE 估计

## 一句话总结
把"等变成像（EI）"的不变性先验和"测量拆分（splitting）"的高效无偏特性结合起来，提出 Equivariant Splitting (ES) 损失，让网络仅凭**单个高度欠采样前向算子**下的不完整测量也能训出逼近 MMSE 的重建器，且无需 EI 那样多次前向评估。

## 研究背景与动机
**领域现状**：逆问题 $y = Ax + \varepsilon$（MRI、CT、压缩感知、inpainting）通常前向矩阵 $A$ 秩亏，监督训练需要难以获取的 ground-truth，于是自监督重建成为刚需。

**现有痛点**：
- **测量拆分（Splitting）** 把测量拆成输入/目标两部分预测彼此，是监督损失的无偏估计——但只在数据集里有**多样化算子**（如每张图 mask 不同的加速 MRI）时成立，单算子场景失效。
- **等变成像（EI）** 能处理单算子，但需要每步 2–3 次网络前向（昂贵），且其等变损失只在重建近乎完美时才有效、对极病态问题不可靠，更不保证收敛到 MMSE 估计。

**核心矛盾**：单个、固定、高度不完整前向算子下，splitting 没有算子多样性可用，EI 又慢且不最优。

**本文目标**：在单算子欠采样设定（如固定 mask 的加速 MRI、稀疏视角 CT）下，既要 splitting 的高效无偏，又要 EI 的不变性先验来突破零空间。

**核心 idea**：在不变性假设 $p(T_g x)=p(x)$ 下，同一组测量 $y=Ax$ 可重新解释为"虚拟真值 $x_g=T_g^{-1}x$ + 虚拟算子 $A_g=AT_g$"——这就**人为造出了一个隐式多算子结构** $\{AT_g\}_{g\in G}$，从而能把 splitting 用上去。

## 方法详解

### 整体框架
ES 的关键观察：$y = Ax + \varepsilon = AT_g\,T_g^{-1}x + \varepsilon = A_g x_g + \varepsilon$。利用群变换 $\{T_g\}$ 把单算子伪装成多算子，再对虚拟算子做测量拆分。再进一步，证明只要重建网络**架构等变**，就不必显式采样变换、不必多次前向，ES 损失直接退化为普通 splitting 损失，实现"零开销的隐式数据增强"。

```mermaid
graph LR
    A[单测量 y, 单算子 A] --> B[不变性假设 p(Tg·x)=p(x)]
    B --> C["虚拟多算子 {A·Tg}<br/>y=A·Tg·(Tg⁻¹x)"]
    C --> D[对虚拟算子做测量拆分 splitting]
    D --> E["等变重建器<br/>f(y,A·Tg)=Tg⁻¹·f(y,A)"]
    E --> F["Thm3: ES 损失 ≡ 普通 splitting 损失<br/>无需多次前向"]
    F --> G[全局最优 = MMSE 估计]
```

### 关键设计
**1. Equivariant Splitting 损失：把不变性变成隐式多算子**
ES 损失定义为对群变换求期望的拆分损失：
$$\mathcal{L}_{\mathrm{ES}}(y,A,f) = \mathbb{E}_g\big\{\mathcal{L}_{\mathrm{SPLIT}}(y, AT_g, f)\big\} = \mathbb{E}_g\Big\{\mathbb{E}_{y_1,A_1|y,AT_g}\big[\|AT_g f(y_1,A_1)-y\|^2\big]\Big\}$$
其中 $A_1$ 是 $AT_g$ 的随机拆分。**Theorem 1** 给出最优性保证：在无噪、$p(x)$ 不变的前提下，只要矩阵 $Q_{A_1}=\mathbb{E}_{g|A_1}[(AT_g)^\top AT_g]$ 满秩，ES 的全局最小化器就是监督 MMSE 估计 $f^*(y_1,A_1)=\mathbb{E}_{x|y_1,A_1}\{x\}$。这点正是 EI 和此前 splitting 都不保证的。

**2. 等变重建器的新定义：突破零空间的必要条件**
作者首次给出**面向重建函数**（而非 image-to-image）的等变定义：
$$f(y, AT_g) = T_g^{-1} f(y, A),\quad \forall y, g, A.$$
**Theorem 2** 证明一大类经典重建器都满足它：去伪影网络 $f=\phi(A^\top y)$、展开网络（unrolled，含梯度步 + 等变去噪 $\phi$）、Reynolds 平均、MAP、MMSE 估计。同时 **Corollary 1** 指出关键约束——前向算子本身**不能**对该组变换等变（$\exists g, AT_g\neq T_g A$），否则 $Q_{A_1}$ 不满秩、学不到零空间信息。这也指导了每个任务选哪组变换：压缩感知/inpainting 用平移等变，MRI/CT 用旋转翻转等变。

**3. 计算协同：等变架构让 ES 退化为零开销 splitting**
**Theorem 3**：若 $f$ 是等变重建器，则 $\mathcal{L}_{\mathrm{ES}}(y,A,f)=\mathcal{L}_{\mathrm{SPLIT}}(y,A,f)$。也就是说，**不必显式采样、施加 $T_g$ 变换**，只要把等变约束写进架构（用 Chaman & Dokmanić 的平移等变 UNet，或对旋转/翻转做 Reynolds 平均），就自动获得"对所有 $T_g$ 做数据增强"的效果，却没有任何变换或额外前向开销——这正是相对 EI（2–3 次前向）的效率优势来源。

**4. 含噪推广：用 R2R 替换一致性项**
ES 损失可拆成测量一致性项 + 预测项。当测量含高斯噪声时，把一致性项替换为 Recorrupted2Recorrupted (R2R) 损失：
$$\mathcal{L}_{\text{G-ES}} = \mathbb{E}\Big\{\big\|A_1 f(y_1+\alpha\omega, A_1)-(y_1-\tfrac{\omega}{\alpha})\big\|^2 + \|A_2 f(y_1+\alpha\omega,A_1)-y_2\|^2\Big\},\ \omega\sim\mathcal{N}(0,\sigma^2 I)$$
R2R 提供一致性项的无偏估计，故 Theorem 1 仍成立、仍收敛到 MMSE；非高斯噪声可换其非高斯推广。推理时对多次随机拆分（10 个）+ 合成噪声做平均。

## 实验关键数据
覆盖 4 类逆问题：压缩感知（MNIST）、inpainting（DIV2K，约 30% 像素）、加速 MRI（FastMRI，×8）、稀疏视角 CT（LIDC-IDRI，50 views）。统一架构、统一训练流程公平比较。

### 主实验

| 任务 | 指标 | ES (本文) | EI (SOTA 自监督) | 监督上界 | Δ vs EI |
|--------|------|------|------|------|---|
| Inpainting (DIV2K) | PSNR | **27.45** | 25.89 | 28.46 | **+1.56 dB** |
| Inpainting | SSIM | **0.8737** | 0.8332 | 0.8982 | **+0.040** |
| MRI ×8, 40dB | PSNR | **28.54** | 27.88 | 28.74 | **+0.66 dB** |
| MRI ×8 (真实测量) | PSNR | **28.30** | 27.88 | 28.81 | **+0.42 dB** |
| CT 50 views, 50dB | PSNR | **32.62** | 28.61 | 33.99 | **+4.01 dB** |
| CT 50 views | SSIM | **0.8570** | 0.7400 | 0.8819 | **+0.117** |

ES 在所有任务上都优于 EI/SURE/MC 等自监督基线，且逼近监督上界；CT 这种高度病态问题上对 EI 的领先最显著（+4 dB）。压缩感知曲线显示 EI 的差距随压缩率升高而扩大，ES 始终贴近监督线。

### 消融实验（等变架构的影响，Splitting 损失）

| 任务 | 等变架构 | PSNR | EQUIV |
|------|----------|------|-------|
| Inpainting | ✓ | **27.45** | **27.46** |
| Inpainting | ✗ | 27.20 | 26.52 |
| MRI ×8 | ✓ | **28.54** | **31.53** |
| MRI ×8 | ✗ | 28.18 | 27.28 |

### 关键发现
- 等变架构 + splitting 损失存在**协同效应**：等变版本一致地更优、等变度（EQUIV）也更高，验证了 Theorem 3 的理论预测。
- 非等变架构仍表现出"出乎意料"的高等变度——即 learned equivariance 现象，可能解释了 splitting 方法即便用非等变架构也表现不错。
- SURE / IDFT / FBP 等无法恢复零空间信息的基线在 MRI/CT 上明显失败，印证理论。

## 亮点与洞察
- **概念重构很优雅**：用"$y=A_g x_g$"把不变性假设翻译成隐式多算子，让原本互斥的 splitting（需多算子）和 EI（单算子）在单算子设定里融为一体。
- **首个面向重建函数的等变定义** $f(y,AT_g)=T_g^{-1}f(y,A)$，区别于 EI 强制的 $f(AT_g x,A)=T_g f(Ax,A)$（后者仅在完美重建时等价），并证明展开网络/去伪影网络/MAP/MMSE 都满足。
- **理论 + 效率双赢**：既给出 MMSE 最优性保证（EI 没有），又靠架构等变省掉 EI 的 2–3 次前向开销。

## 局限与展望
- 最优性依赖 $Q_{A_1}$ / $\bar{Q}_A$ 满秩，且闭式不可得时只能用非加权平均近似（式 12），实际最优性由谱的"非可忽略特征值数"决定，病态时仍打折扣。
- 不变性假设对自然/遥感/显微图像较自然，但医学图像只是"近似不变"，方法依赖这种近似成立。
- 仅报告失真指标（PSNR/SSIM），未评估感知质量；作者明示感知与失真有 trade-off 故未纳入。
- 变换组需按问题手工选择（Corollary 1 指导），自动选择仍是开放问题。

## 相关工作与启发
- **vs 测量拆分（Millard & Chiew 2023; Daras 2023）**：原方法要求数据集算子多样，本文首次把它扩展到单算子欠采样（固定 mask MRI、稀疏 CT）。
- **vs 等变成像 EI（Chen 2021/2022）**：共享不变性假设，但 ES 用架构等变替代损失等变，更快、更稳、有 MMSE 保证。
- **vs 等变网络（Cohen & Welling；Chaman & Dokmanić）**：把"提升测试期泛化"的等变架构重新用作"从不完整数据学习"的工具，拓展了等变网络的用途。
- **启发**：不变性是从不完整测量中学习的强先验；把"数据增强"内化进架构而非损失，能在自监督逆问题里同时拿到效率和最优性。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用"虚拟多算子"统一 splitting 与 EI，并首次给出重建函数等变定义，概念清晰且填补单算子欠采样空白
- 实验充分度: ⭐⭐⭐⭐ 覆盖 4 类逆问题 + 真实 MRI + 消融，统一公平比较；但缺感知指标与更大规模自然图像
- 写作质量: ⭐⭐⭐⭐⭐ 理论（3 个定理 + 推论）与动机层层递进，背景铺垫到位，可读性强
- 价值: ⭐⭐⭐⭐ 对医学影像/天文/显微等无 GT 场景实用，理论保证 + 效率优势使其有望成为自监督逆问题新基线

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Understanding the Robustness of Distributed Self-Supervised Learning Frameworks Against Non-IID Data](understanding_the_robustness_of_distributed_self-supervised_learning_frameworks_.md)
- [\[ICLR 2026\] On the Alignment Between Supervised and Self-Supervised Contrastive Learning](on_the_alignment_between_supervised_and_self-supervised_contrastive_learning.md)
- [\[ICLR 2026\] Incomplete Multi-View Multi-Label Classification via Shared Codebook and Fused-Teacher Self-Distillation](incomplete_multi-view_multi-label_classification_via_shared_codebook_and_fused-t.md)
- [\[ICLR 2026\] Understanding the Learning Phases in Self-Supervised Learning via Critical Periods](understanding_the_learning_phases_in_self-supervised_learning_via_critical_perio.md)
- [\[ICLR 2026\] Soft Equivariance Regularization for Invariant Self-Supervised Learning](soft_equivariance_regularization_for_invariant_self-supervised_learning.md)

</div>

<!-- RELATED:END -->
