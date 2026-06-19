---
title: >-
  [论文解读] Hankel Singular Value Regularization for Highly Compressible State Space Models
description: >-
  [NeurIPS 2025][模型压缩][SSM压缩] 通过在训练中正则化 SSM 层的 **Hankel 奇异值核范数**促使其快速衰减，使训练后模型可用平衡截断压缩至原始阶数的 **10%** 而保持精度，并利用旋转矩阵块对角参数化将 Gramian 计算从 $\mathcal{O}(n^3)$ 降至 $\mathcal{O}(n^2)$。
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "SSM压缩"
  - "Hankel奇异值"
  - "平衡截断"
  - "正则化"
  - "Long Range Arena"
---

# Hankel Singular Value Regularization for Highly Compressible State Space Models

**会议**: NeurIPS 2025  
**arXiv**: [2510.22951](https://arxiv.org/abs/2510.22951)  
**代码**: [GitHub](https://github.com/Algopaul/hankelreg)  
**领域**: 模型压缩 / 状态空间模型  
**关键词**: SSM压缩, Hankel奇异值, 平衡截断, 正则化, Long Range Arena

## 一句话总结

通过在训练中正则化 SSM 层的 **Hankel 奇异值核范数**促使其快速衰减，使训练后模型可用平衡截断压缩至原始阶数的 **10%** 而保持精度，并利用旋转矩阵块对角参数化将 Gramian 计算从 $\mathcal{O}(n^3)$ 降至 $\mathcal{O}(n^2)$。

## 研究背景与动机

SSM（如 S4/S5/Mamba）在长序列任务上表现优异，但推理成本与状态维度 $n$ 正相关。压缩需降低 $n$，关键在于 Hankel 奇异值的衰减速度。

**核心系统论视角**：线性时不变系统的序列到序列映射由 Hankel 算子描述，其奇异值 $\sigma_1, \ldots, \sigma_n$ 决定了可压缩性。将阶 $n$ 截断到 $r$ 的映射误差满足：

$$\|\hat{y}_k - y_k\|_{\ell_2} \leq 2\|u\|_{\ell_2} \sum_{i=r+1}^n \sigma_i$$

**问题**：标准训练的 SSM 的 Hankel 奇异值衰减缓慢，直接截断导致精度骤降。先前工作（Ezoe et al.）需要截断后重新训练。

## 方法详解

### 旋转矩阵块对角参数化

将 $\mathbf{A}$ 参数化为 $2 \times 2$ 旋转块的对角：

$$\mathbf{A}_i(\rho_i, \alpha_i) = \rho_i \begin{bmatrix} \cos(\alpha_i) & \sin(\alpha_i) \\ -\sin(\alpha_i) & \cos(\alpha_i) \end{bmatrix}$$

- **稳定性保证**：$\rho_i$ 通过 $\tanh$ 约束在 $[0, 1)$
- **普适性**：Proposition 1 证明该参数化在 SSM 空间中稠密
- **参数量**：线性于 $n$，与 S5 对角参数化相同

### Gramian 高效计算

Hankel 奇异值 $\sigma_i = \sqrt{\lambda_i(\mathbf{P}\mathbf{Q})}$，需解 Lyapunov 方程：

$$\mathbf{A}\mathbf{P}\mathbf{A}^\top - \mathbf{P} + \mathbf{B}\mathbf{B}^\top = \mathbf{0}$$

标准 Bartels-Stewart 算法为 $\mathcal{O}(n^3)$。利用块对角结构，原方程分解为 $q$ 个 $2 \times 2$ Lyapunov 方程和 $q(q-1)/2$ 个 $2 \times 2$ Sylvester 方程（$q = n/2$），每个独立可解且可并行化：

$$\mathbf{A}_i \mathbf{P}_{ij} \mathbf{A}_j^\top - \mathbf{P}_{ij} + \mathbf{B}_i \mathbf{B}_j^\top = \mathbf{0}$$

总复杂度 $\mathcal{O}(q^2) = \mathcal{O}(n^2)$，且独立于序列长度 $n_s$。

### 可微正则器

虽然单个 Hankel 奇异值关于系统矩阵不可微，但 Proposition 2 证明其**和**是光滑的。定义核范数正则器：

$$\mathcal{R}_*(\boldsymbol{\sigma}^{(1)}, \ldots, \boldsymbol{\sigma}^{(L)}) = \sum_{\ell=1}^L \sum_{i=1}^n \sigma_i^{(\ell)}$$

加入训练损失，鼓励奇异值集中于前几个从而快速衰减。

### 关联扫描加速

利用旋转矩阵乘法的性质 $\mathbf{A}(\mathbf{x}, \boldsymbol{\beta}) \mathbf{A}(\mathbf{y}, \boldsymbol{\gamma}) = \mathbf{A}(\mathbf{x} \odot \mathbf{y}, \boldsymbol{\beta} + \boldsymbol{\gamma})$，定义新的关联二元运算，避免显式块矩阵乘积，实现与对角 S5 相同的 $\mathcal{O}(\log(n_s) \cdot n)$ 并行扫描。

### 训练后压缩

1. 解最终 Gramian $\mathbf{P}$, $\mathbf{Q}$
2. 平衡截断（balanced truncation）得到 $r$ 阶缩减系统
3. 层自适应秩分配：给定总预算 $r_t$，用二分法分配各层 $r^{(\ell)}$ 使各层保留相同比例能量
4. 对角化缩减系统以恢复高效扫描

## 实验关键数据

### Long Range Arena 不同截断比的测试精度

| 方法 | sCIFAR 60% | 70% | 80% | 90% | sMNIST 60% | 70% | 80% | 90% |
|------|-----------|-----|-----|-----|-----------|-----|-----|-----|
| LAST | 62.93 | 36.66 | 17.35 | 11.19 | 95.11 | 89.17 | 62.37 | 27.67 |
| global | 28.91 | 13.62 | 11.12 | 10.47 | 91.67 | 83.32 | 52.52 | 21.94 |
| no reg. | 71.28 | 41.98 | 21.14 | 9.84 | 91.32 | 13.35 | 11.05 | 10.55 |
| **HSVR** | **81.84** | **81.75** | **81.37** | **51.08** | **99.45** | **99.22** | **98.90** | **86.95** |

| 方法 | IMDB 60% | 70% | 80% | 90% | PATH-X 60% | 70% | 80% |
|------|---------|-----|-----|-----|-----------|-----|-----|
| LAST | 88.48 | 85.05 | 80.26 | 57.08 | 50.33 | 49.16 | 49.53 |
| no reg. | 71.45 | 71.04 | 51.32 | 50.00 | 56.09 | 50.39 | 50.16 |
| **HSVR** | **87.26** | **87.16** | **86.97** | **86.40** | **87.74** | **82.82** | **54.02** |

### 关键发现

- **sMNIST 80% 截断**：HSVR 保持 98.90%，无正则化暴跌至 11.05%（近随机）
- **IMDB 90% 截断**：HSVR 仅损失 ~1%（86.40 vs 87.26），无正则化降至 50%
- **PATH-X 60% 截断**：HSVR 达 87.74%，所有基线均降至随机水平（~50%）
- 层自适应秩分配比均匀分配更优：不同层的"重要状态数"差异显著

## 亮点与洞察

- **系统论与深度学习的优雅桥接**：将神经网络 SSM 层的可压缩性归化为经典 Hankel 奇异值分析
- **不可微变可微**：利用奇异值曲线交叉处的光滑性证明核范数可微——精巧的分析技巧
- **训练即压缩准备**：一次正则化训练，后处理仅需标准平衡截断，无需重训
- **$\mathcal{O}(n^2)$ Gramian 算法**：实用的效率提升使正则化在训练中可行

## 局限与展望

1. **不适用于预训练模型**：必须从头带正则化训练，无法对已训练 SSM 事后压缩
2. **仅限线性时不变系统**：Mamba 的时变/输入依赖 SSM 不适用
3. 仅考虑单向扫描，双向 SSM（sCIFAR/IMDB 标准配置）的结果可能不同
4. 核范数正则强度需手动调节，不同任务敏感度差异大
5. 压缩后需对角化恢复高效扫描，若特征值近简并则数值不稳定

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 将经典系统理论引入 SSM 压缩，视角独特
- **技术深度**: ⭐⭐⭐⭐⭐ — 参数化设计、Gramian 算法、可微性证明、关联扫描均有贡献
- **实验充分度**: ⭐⭐⭐⭐ — LRA 5 个 benchmark 全面，但缺少语言建模等实际 NLP 任务
- **实用性**: ⭐⭐⭐ — 需从头训练的限制降低了实际价值
- **总体**: ⭐⭐⭐⭐

## 与相关工作的对比

## 启发与关联

## 评分

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] State-offset Tuning: State-based Parameter-Efficient Fine-Tuning for State Space Models](../../ACL2025/model_compression/state_offset_tuning_ssm_peft.md)
- [\[ICML 2026\] Quantifying the Uncertainty of Foundation Models with Singular Value Ensembles](../../ICML2026/model_compression/quantifying_the_uncertainty_of_foundation_models_with_singular_value_ensembles.md)
- [\[ICML 2025\] Parameter-Efficient Fine-Tuning of State Space Models](../../ICML2025/model_compression/parameter-efficient_fine-tuning_of_state_space_models.md)
- [\[NeurIPS 2025\] Smooth Regularization for Efficient Video Recognition](smooth_regularization_for_efficient_video_recognition.md)
- [\[CVPR 2026\] AdaSVD: Singular Value Decomposition with Adaptive Mechanisms for Large Multimodal Models](../../CVPR2026/model_compression/adasvd_singular_value_decomposition_with_adaptive_mechanisms_for_large_multimoda.md)

</div>

<!-- RELATED:END -->
