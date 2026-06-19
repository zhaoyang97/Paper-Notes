---
title: >-
  [论文解读] RefiDiff: Progressive Refinement Diffusion for Efficient Missing Data Imputation
description: >-
  [AAAI 2026][图像恢复][missing data imputation] 提出 RefiDiff 四阶段框架（预处理→warm-up→扩散→polish），首次将 predictive 和 generative 缺失值填补范式渐进统一，结合 Mamba-based denoising 在 9 个数据集上取得 SOTA，速度比 DIFFPUTER 快 4 倍。
tags:
  - "AAAI 2026"
  - "图像恢复"
  - "missing data imputation"
  - "扩散模型"
  - "Mamba"
  - "tabular data"
  - "MNAR"
---

# RefiDiff: Progressive Refinement Diffusion for Efficient Missing Data Imputation

**会议**: AAAI 2026  
**arXiv**: [2505.14451](https://arxiv.org/abs/2505.14451)  
**代码**: [GitHub](https://github.com/Atik-Ahamed/RefiDiff)  
**领域**: Data Imputation / Tabular Data  
**关键词**: missing data imputation, diffusion model, Mamba, tabular data, MNAR

## 一句话总结
提出 RefiDiff 四阶段框架（预处理→warm-up→扩散→polish），首次将 predictive 和 generative 缺失值填补范式渐进统一，结合 Mamba-based denoising 在 9 个数据集上取得 SOTA，速度比 DIFFPUTER 快 4 倍。

## 研究背景与动机

### 领域现状

**领域现状**：缺失值在高维混合类型（数值+分类）数据集中普遍存在，涉及 MCAR（完全随机缺失）、MAR（条件随机缺失）和 MNAR（非随机缺失）三种机制。现有填补方法分为两大范式：Predictive 方法（如 XGBoost 回归）高效确定性但缺乏不确定性建模，偏向 local view；Generative 方法（如 diffusion models）能建模全局分布但计算密集。

### 现有痛点

**现有痛点**：(1) 两种范式各有优劣但很少被有效统一——MICE 类方法需要迭代收敛慢，DIFFPUTER 需要 EM 迭代效率低；(2) 现有 diffusion 方法（TabDDPM、DIFFPUTER）使用 Transformer 作为 denoiser，在高维表格数据上计算开销大；(3) MNAR 场景最具挑战性，因为缺失机制本身与缺失值相关，大多数方法在此场景下性能显著下降。

### 核心矛盾

**核心矛盾**：Predictive 方法擅长局部精确填补但无法建模全局分布不确定性，Generative 方法擅长全局分布建模但引入过多噪声和计算开销。如何在不牺牲效率的前提下同时获得两者的优势？

### 解决思路

**本文目标**：设计一个渐进式框架，让 predictive 方法提供良好初始化，generative 方法在此基础上进一步优化全局分布。**切入角度**：将填补过程分为 warm-up（predictive）+ diffusion（generative）+ polishing（predictive）三阶段，每个阶段渐进地提升填补质量。**核心idea**：用 XGBoost 做单遍 warm-up 提供初始填补，Mamba-based diffusion 在此基础上做全局分布修正，最后再用 regression polishing 去除残余噪声。

## 方法详解

### 整体框架
RefiDiff 包含四阶段流水线：Pre-processing → Warm-up Refinement → Diffusion Imputation → Post-processing & Polishing。输入为含缺失值的混合类型表格数据 $\mathbf{Z} \in \mathbb{R}^{N \times D}$ 和缺失 mask $\mathbf{M}$，输出为完整填补后的数据。

### 关键设计

1. **Pre-processing + Warm-up Refinement**:

    - 功能：将原始数据标准化并提供一次性初始填补
    - 核心思路：分类特征 binary encoding，数值特征用 observed 数据的 $\mu, \sigma$ 标准化，缺失位置填零。随后对每个特征列 $f_j$ 训练轻量 XGBoost 模型 $\theta_1^{(j)}$，单遍扫描填补所有缺失值。三个关键性质：Non-overwriting（已观测值不被修改）、Well-defined mapping（每列有确定的预测器）、One-pass completion（不需要迭代）
    - 设计动机：与 MICE 的多轮迭代不同，单遍 warm-up 极大降低了计算代价，同时为 diffusion 提供比随机噪声更好的起点

2. **Mamba-based Diffusion Module**:

    - 功能：全局分布建模和条件采样
    - 核心思路：采用 VE SDE 连续时间扩散，训练 Mamba-based denoising network $\theta_2$，使用 diamond 结构（2 个 up-sampling + 2 个 down-sampling 残差块），每块含 Mamba 层 + FC + PE + LayerNorm + Dropout。损失函数为 EDM loss：$\mathcal{L}_{SM} = \mathbb{E}_{X_0,\varepsilon,t}[\|\theta_2(X_t,t,M) - \nabla_{X_t}\log p(X_t|X_0)\|_2^2]$。推理时进行 $N$ 次 reverse diffusion 取平均，已观测位置始终 clamp 不变
    - 设计动机：Mamba 替代 Transformer 实现线性复杂度，在保持长程依赖捕获能力的同时实现 4× 提速。diamond 结构通过多尺度特征融合提升表达力

3. **Post-refinement (Polishing)**:

    - 功能：去除 diffusion 输出中的残余噪声
    - 核心思路：对 diffusion 输出再做一遍 column-wise regression 修正，利用 predictive 方法的局部精确性弥补 generative 方法可能引入的噪声
    - 设计动机：diffusion 采样的随机性可能在某些列引入小幅噪声，polishing 阶段用确定性预测修正这些偏差

### 理论保证
论文提供了 KL 散度上界：$\text{KL} \leq C_1 T \varepsilon_\theta^2 + C_2 \delta t + C_3 / N$，证明了条件采样结果随 denoiser 精度、时间步密度和采样次数的提升而趋近真实分布。

## 实验关键数据

### 主实验

在 9 个 real-world 数据集上评测，包括 MCAR、MAR、MNAR 三种缺失机制。

| 方法 | MNAR MAE/RMSE | MCAR MAE/RMSE | MAR MAE/RMSE | Avg Rank |
|------|:---:|:---:|:---:|:---:|
| DIFFPUTER | 37.27/86.86 | 31.72/63.49 | 39.15/90.95 | 2.67 |
| ReMasker | 39.66/80.23 | 35.84/65.19 | 38.39/78.82 | 3.00 |
| **RefiDiff** | **34.49/78.83** | **31.41/63.16** | **34.52/78.22** | **1.17** |

比 DIFFPUTER **快 4 倍**，参数量更少。

### 消融实验

| 配置 | OOS RMSE (MAR) | OOS RMSE (MNAR) | 说明 |
|------|:---:|:---:|------|
| Full RefiDiff | 73.82 | 70.12 | 完整模型 |
| w/o Diffusion | 91.80 | 81.07 | 去 diffusion 掉 24.3%/15.6% |
| w/o Warm-up | 82.45 | 76.89 | 去 warm-up 掉 11.7%/9.6% |
| w/o Polishing | 78.93 | 73.41 | 去 polish 掉 6.9%/4.7% |

### 关键发现
- Diffusion 模块贡献最大：去除后 MAR RMSE 从 73.82 升至 91.80（+24.3%），证明全局分布建模不可或缺
- Warm-up 提供了有效初始化：去除后性能掉 11.7%，说明 predictive 初始化对 diffusion 质量有显著影响
- 在 MNAR 场景下提升最明显：RMSE 比 DIFFPUTER 低 9.3%（78.83 vs 86.86），因为 warm-up 为 MNAR 的缺失值提供了更准确的初始估计
- 分类精度同样排名第一（平均 rank 1.17）

## 亮点与洞察
- **范式统一设计**：首次将 predictive + generative 通过"warm-up→diffusion→polish"渐进式 refinement 无缝融合，思路优雅且实用
- **Mamba 替代 Transformer**：在非序列数据（表格）上验证了 Mamba 的长程依赖捕获优势，4× 提速的同时性能更优
- **Plug-and-play**：无需针对不同数据集调整架构或超参，通用性强
- "Warm-up + Generation + Polishing" 三阶段渐进式设计可推广到其他生成式填补/修复任务

## 局限与展望
- 分类特征采用 binary encoding 再做连续扩散，可能损失离散语义保真度
- 仅在 UCI 等中等规模数据集验证，百万级超大规模场景待探索  
- Warm-up 仅做单遍扫描，对极端高缺失率（>70%）可能不够充分
- 未探索 streaming/在线数据的增量填补场景

## 相关工作与启发
- **vs DIFFPUTER**: 去除迭代 EM，warm-up 单遍 + 单次 diffusion 即可达更好效果；Mamba 替代 TabDDPM 的 Transformer，4× 提速
- **vs ReMasker**: ReMasker 用 masked autoencoding 建模，RefiDiff 用扩散 + 渐进 refinement，OOS 泛化更好
- **vs MICE**: MICE 需多轮迭代直到收敛，RefiDiff 的 predictive 阶段是 one-pass（效率高一个数量级）
- Mamba 在非图像非文本任务上的成功应用值得关注，可考虑扩展到音频、时序信号等其他长序列模态

## 评分
- 新颖性: ⭐⭐⭐⭐ 渐进式统一两种范式的思路优雅，Mamba 在表格数据上的应用新颖
- 实验充分度: ⭐⭐⭐⭐ 9 数据集、3 种缺失机制、完整消融
- 写作质量: ⭐⭐⭐⭐ 理论与实验结合好，四阶段流程清晰
- 价值: ⭐⭐⭐⭐ 对缺失值填补社区有实用价值，MNAR 贡献突出

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Encoder-Decoder Diffusion Language Models for Efficient Training and Inference](../../NeurIPS2025/image_restoration/encoder-decoder_diffusion_language_models_for_efficient_training_and_inference.md)
- [\[ICLR 2026\] Horizon Imagination: Efficient On-Policy Rollout in Diffusion World Models](../../ICLR2026/image_restoration/horizon_imagination_efficient_on-policy_rollout_in_diffusion_world_models.md)
- [\[CVPR 2025\] Efficient Diffusion as Low Light Enhancer (ReDDiT)](../../CVPR2025/image_restoration/efficient_diffusion_as_low_light_enhancer.md)
- [\[CVPR 2025\] Progressive Focused Transformer for Single Image Super-Resolution](../../CVPR2025/image_restoration/progressive_focused_transformer_for_single_image_super-resolution.md)
- [\[ICML 2026\] DyLLM: Efficient Diffusion LLM Inference via Saliency-based Token Selection and Partial Attention](../../ICML2026/image_restoration/dyllm_efficient_diffusion_llm_inference_via_saliency-based_token_selection_and_p.md)

</div>

<!-- RELATED:END -->
