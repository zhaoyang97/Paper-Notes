---
title: >-
  [论文解读] Beyond Masked and Unmasked: Discrete Diffusion Models via Partial Masking
description: >-
  [NeurIPS 2025][图像生成][离散扩散模型] Prime（Partial masking scheme）通过将每个token用base-b子token序列表示并在子token级别独立掩码，为掩码扩散模型引入中间状态，实现细粒度去噪过程，在OpenWebText上以15.36困惑度首次让MDM在不使用自回归公式的情况下超越ARM（17.54）。
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "离散扩散模型"
  - "掩码扩散"
  - "部分掩码"
  - "子token"
  - "文本生成"
---

# Beyond Masked and Unmasked: Discrete Diffusion Models via Partial Masking

**会议**: NeurIPS 2025  
**arXiv**: [2505.18495](https://arxiv.org/abs/2505.18495)  
**代码**: 无  
**领域**: 图像生成 / 离散扩散  
**关键词**: 离散扩散模型, 掩码扩散, 部分掩码, 子token, 文本生成

## 一句话总结
Prime（Partial masking scheme）通过将每个token用base-b子token序列表示并在子token级别独立掩码，为掩码扩散模型引入中间状态，实现细粒度去噪过程，在OpenWebText上以15.36困惑度首次让MDM在不使用自回归公式的情况下超越ARM（17.54）。

## 研究背景与动机

**领域现状** 掩码扩散模型（MDM）是离散数据生成的有力模型，通过逐步揭示掩码token来生成样本。每个token只有两种状态：掩码或未掩码。

**现有痛点** 二值表示导致严重的计算浪费——在逆扩散过程中，大量步骤序列不发生任何变化（idle steps），模型在重复处理完全相同的输入。实验表明37%的步骤是无效的。

**核心矛盾** MDM的二值状态限制了模型利用率：要么完全掩码（无信息），要么完全揭示（最终确定），缺少中间过渡状态来实现渐进的信息释放。

**本文目标** 重新定义扩散过程，将idle步转化为有信息量的更新，提升模型在生成过程中的利用率。

**切入角度** 用base-b编码将每个token拆分为子token序列，在子token级别独立掩码，自然产生中间状态。

**核心 idea** 通过子token的部分掩码从"二值掩码/未掩码"扩展为"多级中间状态"，使四选一预测可分解为多步二选一决策。

## 方法详解

### 整体框架
MDM-Prime包含三步：(1) 用可逆函数 $f$ 将每个token $x_0^i \in \mathcal{X}$ 映射为长度 $\ell$ 的子token序列 $\mathbf{y}_0^i \in \mathcal{Y}^\ell$（base-$b$编码，$b = \lceil \sqrt[\ell]{C} \rceil$）；(2) 在子token级别独立执行掩码扩散前向过程；(3) 逆扩散过程中逐步揭示子token，实现从完全掩码到中间状态再到完全揭示的细粒度转换。

### 关键设计

1. **部分掩码方案（Prime）**:
    - 功能：为离散扩散引入中间状态
    - 核心思路：将token $x_0^i$ 编码为子token序列 $\mathbf{y}_0^i = f(x_0^i)$，子token独立掩码产生中间状态。例如4类token用2-bit编码，中间状态为"m0"或"1m"，提供部分信息。中间状态数为 $(b+1)^\ell - (C+1)$，始终为正
    - 设计动机：中间状态使模型能基于部分已知的token信息做更精确的预测，减少idle步。理论证明 $\ell$ 增大时idle步单调递减

2. **联合概率参数化**:
    - 功能：建模子token间的依赖并防止生成无效样本
    - 核心思路：直接参数化联合分布 $p_\theta(\mathbf{y}_0^i|\mathbf{y}_t)$，只对有效的base-$b$编码（$\mathbf{y}_0^i \in f(\mathcal{X})$）分配概率权重，将 $|\mathcal{V}(\mathbf{y}_t^i)|$ 外的logit显式置零。同时满足carry-over约束：已揭示的子token保持不变
    - 设计动机：独立参数化 $\prod_j p_\theta(y_0^{i,j}|\mathbf{y}_t)$ 不仅引入错误独立性假设（导致采样分布退化），还可能生成无效的子token组合（如GPT-2 50257词表映射时）

3. **子token嵌入编码器**:
    - 功能：高效处理子token输入
    - 核心思路：为每个子token创建独立的 $D/\ell$ 维嵌入查表，拼接 $\ell$ 个嵌入得到 $D$ 维token嵌入。查表大小仅需 $(b+1) \times D/\ell$，远小于完整的 $|\tilde{\mathcal{Y}}^\ell|$ 维查表
    - 设计动机：子token空间 $\tilde{\mathcal{Y}}^\ell$ 可能远大于原token空间，直接建查表不可行；拼接策略保持与标准MDM架构兼容

### 损失函数 / 训练策略
变分上界损失：$\mathcal{L}_{vb}(\mathbf{y}_0;\theta) = \int_0^1 \frac{\alpha'_t}{1-\alpha_t} \mathbb{E}_{q(\mathbf{y}_t|\mathbf{y}_0)}[\sum_i \log p_\theta(\mathbf{y}_0^i|\mathbf{y}_t)] dt$，即加权交叉熵损失，理论保证为负对数似然的上界。

## 实验关键数据

### 主实验——文本生成（OpenWebText困惑度PPL）

| 方法 | PPL ↓ | Idle步比例 |
|------|-------|-----------|
| ARM（自回归）* | 17.54 | - |
| MDLM | ≤22.98 | 36.77% |
| EDLM-coAR* | ≤17.58 | - |
| MDLM-Prime (ℓ=2) | ≤17.90 | 13.52% |
| MDLM-Prime (ℓ=4) | ≤15.62 | 1.83% |
| **MDLM-Prime (ℓ=6)** | **≤15.36** | **0.25%** |

### 主实验——图像生成

| 方法 | CIFAR-10 FID ↓ | ImageNet-32 FID ↓ |
|------|---------------|-------------------|
| 连续扩散SOTA | ~2.5-3.5 | ~6-8 |
| MDM-Prime | **3.26** | **6.98** |

### 消融实验

| 配置 | OWT PPL | 说明 |
|------|---------|------|
| 独立参数化 | 退化 | 子token独立假设导致分布扭曲 |
| 联合参数化 | 15.36 | 捕捉子token依赖 |
| 无carry-over | 更高 | carry-over对零样本泛化很重要 |
| ℓ=2→8 | 17.90→15.48 | ℓ≥4时性能收敛 |

### 关键发现
- 首次让MDM在不依赖自回归公式的情况下超越ARM（15.36 vs 17.54）
- idle步比例与PPL高度相关——从36.77%（MDLM）降至0.25%（Prime ℓ=6）时PPL从22.98降到15.36
- 在图像生成上与连续扩散方法相当（CIFAR-10 FID 3.26）
- ℓ≥4时性能收敛，推荐选择ℓ=4或6

## 亮点与洞察
- "二值→多级中间状态"的核心idea直觉简单但效果惊人——仅修改嵌入层即可将MDLM提升7个PPL点
- idle步分析为理解MDM性能瓶颈提供了新视角
- 联合参数化+carry-over的设计既保证了理论正确性又实现了高效实现

## 局限与展望
- 子token编码增加了序列长度（$L \times \ell$），增加Transformer的计算量
- 当前仅在130M参数模型上验证，更大规模LLM上的表现待确认
- base-b编码是手工设计的，可能存在更优的token分解策略

## 相关工作与启发
- **vs MDLM**: Prime是MDLM的直接增强，仅修改嵌入层，架构完全兼容
- **vs SEDD**: SEDD用吸收状态+得分匹配，Prime用部分掩码+变分上界，两种互补视角
- **vs BD3-LM**: BD3混合自回归公式使MDM更强，但Prime证明无需AR也可超越ARM

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 部分掩码idea简洁有力，首次让MDM超越ARM是里程碑式结果
- 实验充分度: ⭐⭐⭐⭐ 文本+图像跨模态验证，七个零样本基准，消融充分
- 写作质量: ⭐⭐⭐⭐ 理论推导严谨，图示清晰
- 价值: ⭐⭐⭐⭐⭐ 对离散扩散模型领域有重要推动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Learnable Sampler Distillation for Discrete Diffusion Models](learnable_sampler_distillation_for_discrete_diffusion_models.md)
- [\[NeurIPS 2025\] Non-Markovian Discrete Diffusion with Causal Language Models](non-markovian_discrete_diffusion_with_causal_language_models.md)
- [\[NeurIPS 2025\] Information-Theoretic Discrete Diffusion](information-theoretic_discrete_diffusion.md)
- [\[NeurIPS 2025\] Fast Solvers for Discrete Diffusion Models: Theory and Applications of High-Order Algorithms](fast_solvers_for_discrete_diffusion_models_theory_and_applications_of_high-order.md)
- [\[NeurIPS 2025\] A Diffusion Model for Regular Time Series Generation from Irregular Data with Completion and Masking](a_diffusion_model_for_regular_time_series_generation_from_irregular_data_with_co.md)

</div>

<!-- RELATED:END -->
