---
title: >-
  [论文解读] h-Edit: Effective and Flexible Diffusion-Based Editing via Doob's h-Transform
description: >-
  [CVPR 2025][图像生成][图像编辑] h-Edit 基于 Doob's h-transform 将扩散图像编辑形式化为反向时间桥建模问题，通过将编辑更新解耦为"重建项"和"编辑项"，首次实现了免训练的文本引导+奖励模型联合编辑，在 PIE-Bench 上全面超越现有 SOTA 方法。 领域现状：基于扩散模型的图像编…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "图像编辑"
  - "Doob h变换"
  - "扩散桥"
  - "免训练编辑"
  - "奖励模型编辑"
---

# h-Edit: Effective and Flexible Diffusion-Based Editing via Doob's h-Transform

**会议**: CVPR 2025  
**arXiv**: [2503.02187](https://arxiv.org/abs/2503.02187)  
**代码**: [https://github.com/nktoan/h-edit](https://github.com/nktoan/h-edit)  
**领域**: 扩散模型 / 图像编辑  
**关键词**: 图像编辑, Doob h变换, 扩散桥, 免训练编辑, 奖励模型编辑

## 一句话总结
h-Edit 基于 Doob's h-transform 将扩散图像编辑形式化为反向时间桥建模问题，通过将编辑更新解耦为"重建项"和"编辑项"，首次实现了免训练的文本引导+奖励模型联合编辑，在 PIE-Bench 上全面超越现有 SOTA 方法。

## 研究背景与动机

**领域现状**：基于扩散模型的图像编辑方法（DDIM inversion + P2P、Edit Friendly、PnP Inversion 等）已取得显著进展。这些方法的核心思路是通过反转过程将原图映射到噪声空间，再用目标条件重新采样得到编辑结果。

**现有痛点**：(1) 大多数方法基于启发式或直觉设计，缺乏清晰的理论基础，导致难以推广到复杂场景；(2) "编辑有效性"和"内容保真度"之间存在难以调和的 trade-off——增强编辑力度往往以牺牲未编辑区域为代价；(3) 几乎所有免训练方法都局限于文本引导编辑，无法结合外部奖励模型（如风格迁移、人脸识别）进行组合式编辑。

**核心矛盾**：现有方法把重建和编辑混在一起优化，没有明确的理论框架来解耦这两者，导致无法灵活组合不同类型的编辑目标。

**本文目标**：建立一个有理论保证的扩散编辑框架，能将编辑更新解耦为独立的重建项和编辑项，使得不同编辑目标可以自由组合。

**切入角度**：概率论中的 Doob's h-transform 可以将一个马尔可夫过程修改为收敛到指定分布的桥过程。将扩散的反向过程视为基础过程，编辑目标编码为 h 函数，即可自然地得到一个收敛到"既真实又具有目标属性"的分布的桥过程。

**核心 idea**：将图像编辑形式化为对扩散反向过程的 h-transform，使编辑采样更新分解为重建项 $x_{t-1}^{base}$ + 编辑梯度 $\nabla \log h(x_{t-1}, t-1)$，其中 h 函数可以是文本条件、奖励模型或它们的组合。

## 方法详解

### 整体框架
给定原图 $x_0^{orig}$ 和编辑条件，首先通过前向过程（DDIM inversion 或随机 inversion）将原图映射到 $x_T^{orig}$。令 $x_T^{edit} = x_T^{orig}$，沿着 h-transform 修改的反向桥过程从 T 到 0 采样，得到编辑图 $x_0^{edit}$。桥过程的转移核为 $p^h(x_{t-1}|x_t) \propto p(x_{t-1}|x_t) \cdot h(x_{t-1}, t-1)$，其中 $p$ 为原始扩散的反向过程，$h$ 编码编辑目标。

### 关键设计

1. **Doob's h-transform 编辑框架**:

    - 功能：提供图像编辑的统一理论基础
    - 核心思路：定义 h 函数为满足 $h(x_0, 0) = p_{\mathcal{Y}}(x_0)$（目标属性概率）的正值函数，通过 $h(x_t, t) = \mathbb{E}_{p(x_0|x_t)}[h(x_0, 0)]$ 递推到任意时间步。由 Proposition 1 保证，构造的桥过程在 $t=0$ 时收敛到 $p^h(x_0) \propto p(x_0) \cdot p_{\mathcal Y}(x_0)$——既真实又具有目标属性。由于 $p^h(x_{t-1}|x_t)$ 一般非高斯，用 Langevin Monte Carlo 近似采样，自然得到 $x_{t-1} = x_{t-1}^{base} + \gamma \nabla \log h(x_{t-1}, t-1)$ 的分解形式
    - 设计动机：现有方法大多是对 DDIM 采样过程的临时修补，缺乏理论保证。h-transform 框架给出了为什么可以把编辑分解成重建+编辑的严格数学解释

2. **显式与隐式 h-Edit 更新**:

    - 功能：提供两种灵活的编辑实现方式
    - 核心思路：**显式更新**（Eq.15）直接在 $x_t$ 上计算编辑梯度 $\nabla \log h(x_t, t)$，适合梯度容易计算的情况。**隐式更新**（Eq.18）在 $x_{t-1}^{base}$ 上计算 $\nabla \log h(x_{t-1}^{base}, t-1)$，可视为以 $x_{t-1}^{base}$ 为初始值对 $\log h$ 做优化，支持多步梯度上升（Eq.21）以增强编辑效果。对 Stable Diffusion 的文本引导编辑，编辑项简化为 $f(x_t, t) = w_{edit}\epsilon_\theta(x_t, t, c_{edit}) - \hat{w}_{orig}\epsilon_\theta(x_t, t, c_{orig}) + (\hat{w}_{orig} - w_{edit})\epsilon_\theta(x_t, t, \emptyset)$
    - 设计动机：显式更新计算快但编辑力度受限；隐式更新通过多步优化可以处理更困难的编辑场景，且 $x_{t-1}^{base}$ 作为初始值天然提供了保真度锚点

3. **Product of h-Experts 组合编辑**:

    - 功能：实现多种编辑目标的自由组合
    - 核心思路：由于 $\log h$ 可以解释为负能量函数，多个 h 函数可以简单相乘组合 $h = h_1 \cdot h_2 \cdots h_m$，在梯度层面就是各分量的梯度之和 $\nabla \log h = \sum_i \nabla \log h_i$。这允许将文本引导编辑（$h_1$ 来自 classifier-free guidance）、风格迁移（$h_2$ 来自 Gram 矩阵匹配的奖励）、人脸身份保持（$h_3$ 来自 ArcFace）等不同目标无缝组合。另外还设计了重建专用的 h 函数 $h_{rec} = \exp(-\lambda \|x_{t-1} - x_{t-1}^{base}\|^2)$，同时实现无优化和基于优化的重建
    - 设计动机：现有方法很难同时做文本编辑+风格迁移+身份保持等复合任务，而 h-Edit 通过能量函数的可加性优雅地解决了这个问题

### 损失函数 / 训练策略
完全免训练（training-free）。核心超参数为引导权重 $w_{edit}$, $\hat{w}_{orig}$（控制编辑强度与保真度的权衡）和隐式更新的优化步数 $K$。确定性 inversion 版本（h-Edit-D）和随机 inversion 版本（h-Edit-R）使用不同的默认参数。

## 实验关键数据

### 主实验：PIE-Bench 文本引导编辑

| 方法 | Inversion | LPIPS↓ | DINO↓ | Local CLIP↑ | Whole CLIP↑ |
|------|-----------|--------|-------|-------------|-------------|
| h-Edit-D + P2P | 确定性 | **0.253** | 0.147 | **8.54** | **27.87** |
| PnP Inv + P2P | 确定性 | 0.250 | 0.095 | 8.48 | 27.22 |
| NT + P2P | 确定性 | 0.248 | 0.130 | 8.41 | 27.03 |
| NMG + P2P | 确定性 | 0.249 | 0.087 | 8.47 | 27.05 |
| h-Edit-R + P2P | 随机 | **0.256** | **0.159** | **8.50** | **26.97** |
| EF + P2P | 随机 | 0.255 | 0.126 | 8.40 | 26.30 |
| LEDITS++ | 随机 | 0.254 | 0.113 | 8.11 | 23.36 |

### 人脸置换实验（CelebA-HQ）

| 方法 | ID↑ | Expr.↓ | Pose↓ | LPIPS↓ | FID↓ |
|------|-----|--------|-------|--------|------|
| h-Edit-R (1step) | **0.80** | **2.76** | **3.78** | **0.04** | 17.68 |
| h-Edit-R (3steps) | **0.84** | 3.10 | 4.29 | 0.05 | 19.12 |
| DiffFace | 0.61 | 3.04 | 4.35 | 0.10 | **11.89** |
| FaceShifter | 0.70 | 2.39 | 2.81 | 0.08 | 10.16 |
| EF | 0.74 | 3.10 | 4.12 | 0.06 | 20.78 |

### 关键发现
- h-Edit-D + P2P 在确定性 inversion 方法中全面最优，Local CLIP 提升 0.06，LPIPS 有竞争力，说明理论框架确实提升了编辑有效性
- PnP Inv 和 NMG 在困难编辑场景中经常"假装没编辑"（重建原图），导致看似好的保真度数值但实际编辑失败。h-Edit 不存在此问题
- 隐式更新的多步优化（3步 vs 1步）在人脸置换任务中将 ID 相似度从 0.80 提升到 0.84，但以轻微保真度下降为代价
- 组合编辑（文本+风格）中 h-Edit-R + P2P 明显优于 EF + P2P，EF 在组合任务中容易产生伪影或篡改未编辑内容

## 亮点与洞察
- **理论框架的优雅性**：将扩散编辑形式化为 Doob's h-transform 桥过程是非常漂亮的理论贡献。重建项和编辑项的分解不再是启发式的，而有严格的概率论依据。这为后续方法提供了统一的理论语言
- **Product of h-Experts 是灵活的组合机制**：在能量函数视角下，不同编辑目标简单相加即可，这比现有方法中复杂的多阶段pipeline简洁得多。特别是首次实现文本+奖励模型的免训练联合编辑
- **隐式更新的优化视角**：将每一步的编辑写成以 $x_{t-1}^{base}$ 为起点的梯度上升优化问题，步数可调——这给了实践者一个直观的"编辑力度旋钮"

## 局限与展望
- 依赖 Stable Diffusion v1.4，未在 SDXL 或 SD3 等更新架构上验证
- 文本引导编辑仍需配合 P2P 的注意力映射来保持结构，框架本身不能完全取代 P2P
- 隐式更新的多步优化会线性增加推理时间
- h 函数的梯度计算对于某些奖励模型可能不可微或不稳定
- 目前仅处理图像编辑，扩展到视频编辑需要解决时序一致性问题

## 相关工作与启发
- **vs PnP Inversion**: PnP 将 inversion 残差直接注入编辑更新，相当于 h-Edit 框架中只有重建项而无编辑项的特例，编辑力度弱
- **vs Edit Friendly (EF)**: EF 用随机 inversion + 残差注入实现编辑，但缺乏理论基础；h-Edit-R 可视为 EF 的理论增强版，增加了显式的编辑项
- **vs FreeDoM/Universal Guidance**: 这些方法也利用外部奖励的梯度引导扩散采样，但没有统一到 h-transform 框架中，也不支持与文本编辑的无缝组合

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Doob's h-transform 引入扩散编辑是全新的理论贡献，框架优雅且实用
- 实验充分度: ⭐⭐⭐⭐ 文本编辑、人脸置换、组合编辑三个任务覆盖较好，但缺少用户研究
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰完整，框架图直观
- 价值: ⭐⭐⭐⭐⭐ 统一了扩散编辑的理论基础，Product of h-Experts 对组合编辑有重要应用前景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Learning to Sample Effective and Diverse Prompts for Text-to-Image Generation](learning_to_sample_effective_and_diverse_prompts_for_text-to-image_generation.md)
- [\[ICCV 2025\] EDiT: Efficient Diffusion Transformers with Linear Compressed Attention](../../ICCV2025/image_generation/edit_efficient_diffusion_transformers_with_linear_compressed_attention.md)
- [\[CVPR 2026\] CARE-Edit: Condition-Aware Routing of Experts for Contextual Image Editing](../../CVPR2026/image_generation/care-edit_condition-aware_routing_of_experts_for_contextual_image_editing.md)
- [\[CVPR 2026\] HP-Edit: A Human-Preference Post-Training Framework for Image Editing](../../CVPR2026/image_generation/hp-edit_a_human-preference_post-training_framework_for_image_editing.md)
- [\[ICML 2026\] AesFormer: Transform Everyday Photos into Beautiful Memories](../../ICML2026/image_generation/aesformer_transform_everyday_photos_into_beautiful_memories.md)

</div>

<!-- RELATED:END -->
