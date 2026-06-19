---
title: >-
  [论文解读] Towards Precise Scaling Laws for Video Diffusion Transformers
description: >-
  [CVPR 2025][视频生成][缩放法则] 本文首次系统验证了视频扩散 Transformer（Video DiT）中缩放法则的存在，并发现视频模型比语言模型对学习率和 batch size 更敏感，提出了同时预测最优超参数、最优模型大小和验证损失的精确缩放法则公式，在相同计算预算下可减少 40.1% 的推理成本或 39.9% 的模型大小。
tags:
  - "CVPR 2025"
  - "视频生成"
  - "缩放法则"
  - "视频扩散"
  - "超参数优化"
  - "DiT"
  - "计算预算分配"
---

# Towards Precise Scaling Laws for Video Diffusion Transformers

**会议**: CVPR 2025  
**arXiv**: [2411.17470](https://arxiv.org/abs/2411.17470)  
**代码**: 无  
**领域**: 扩散模型/视频生成  
**关键词**: 缩放法则, 视频扩散, 超参数优化, DiT, 计算预算分配

## 一句话总结

本文首次系统验证了视频扩散 Transformer（Video DiT）中缩放法则的存在，并发现视频模型比语言模型对学习率和 batch size 更敏感，提出了同时预测最优超参数、最优模型大小和验证损失的精确缩放法则公式，在相同计算预算下可减少 40.1% 的推理成本或 39.9% 的模型大小。

## 研究背景与动机

**领域现状**：视频扩散 Transformer（如 Sora/Movie Gen）通过扩大模型规模显著提升了视频生成质量，目前最大模型已达 300 亿参数。大语言模型的缩放法则（OpenAI Scaling Law、Chinchilla）已被广泛用于预测最优模型大小和预算分配，但在视觉生成模型尤其是视频模型中，缩放法则几乎未被探索。

**现有痛点**：(1) 现有 LLM 缩放法则直接应用于 Video DiT 时预测不准确，因为视频模型对 batch size 和学习率更敏感；(2) OpenAI 缩放法则假设较小 batch size 更高效但忽略了超参数对拟合精度的影响；(3) Chinchilla 法则通过损失函数与计算预算的关系推导最优模型大小，但其损失拟合本身不够精确；(4) DeepSeek 法则引入了最优超参数但忽略了模型大小的影响。

**核心矛盾**：Video DiT 的超参数（学习率、batch size）敏感性导致使用固定非最优超参数拟合的缩放法则系统性偏差——会推荐过大的模型大小并获得更高的验证损失。

**本文目标**：(1) 确认 Video DiT 中缩放法则的存在；(2) 建立包含模型大小和数据量的最优超参数预测公式；(3) 推导任意模型大小和计算预算下的精确性能预测。

**切入角度**：从 mini-batch SGD 的收敛性理论出发，推导出最优 batch size 和学习率与模型大小 $N$ 和训练 token 数 $T$ 的幂律关系，用小模型实验拟合参数后外推到大模型。

**核心 idea**：提出 $B_{\text{opt}} = \alpha_B T^{\beta_B} N^{\gamma_B}$ 和 $\eta_{\text{opt}} = \alpha_\eta T^{\beta_\eta} N^{\gamma_\eta}$ 两个幂律公式，将模型大小显式纳入超参数预测，在最优超参数下拟合更精确的模型-性能-预算缩放法则。

## 方法详解

### 整体框架

三层递进的缩放法则体系：(1) **超参数缩放**——建立最优 batch size $B_{\text{opt}}(N, T)$ 和学习率 $\eta_{\text{opt}}(N, T)$ 的幂律公式；(2) **性能缩放**——在最优超参数下拟合验证损失 $L$ 与模型大小 $N$ 和训练 token 数 $T$ 的关系；(3) **预算分配**——推导给定计算预算 $C$ 时的最优模型大小 $N_{\text{opt}}(C)$。实验在 0.02B-0.26B 的小模型上拟合，外推验证到 1.07B。

### 关键设计

1. **最优超参数的幂律预测（Scaling Laws for Hyperparameters）**:

    - 功能：给定模型大小和训练数据量，精确预测最优 batch size 和学习率
    - 核心思路：理论推导：从 Lipschitz 平滑假设和 mini-batch SGD 收敛分析出发，步长损失变化量 $\Delta L_k \approx -\eta \|G(\theta_k)\|^2 + \frac{1}{2}\eta^2(G^T H G + \frac{\text{tr}(H\Sigma)}{B})$。最优学习率为 $\eta_{\text{opt}}(B) = \frac{\|G\|^2}{G^T H G + \text{tr}(H\Sigma)/B}$。随着模型增大 Lipschitz 常数 $L$ 增加，需要更小学习率和更大 batch size。实验拟合得 $B_{\text{opt}} = 2.18 \times 10^4 \cdot T^{0.81} \cdot N^{0.19}$ 和 $\eta_{\text{opt}} = 0.0002 \cdot T^{-0.045} \cdot N^{-0.162}$。在 1.07B 模型上验证预测值确实对应最小验证损失。
    - 设计动机：以往缩放法则研究要么忽略超参数（OpenAI）要么不考虑模型大小（DeepSeek）。实验证明在 Video DiT 中固定超参数会导致损失曲线上出现系统性偏高的点（Fig.1 灰点 vs 红点），使拟合不精确。将模型大小显式建模为独立变量是关键。

2. **基于最优超参数的性能缩放（Performance Scaling with Optimal Hyperparameters）**:

    - 功能：精确预测任意模型大小和训练 token 数下的可达验证损失
    - 核心思路：在最优超参数下拟合 $L(N, T)$ 的参数化形式。与 Chinchilla 使用 IsoFLOP 曲线不同，直接在实验点上拟合损失曲面。外推到更大模型时预测更准确——1.07B/10B tokens 的预测误差仅约 0.5%。发现在固定计算预算下，当模型大小在最优值附近调整时验证损失变化很小（平坦区域），这为实际应用中使用较小模型（减少推理成本）提供了理论支撑。
    - 设计动机：使用非最优超参数拟合的损失曲线会系统性偏高，导致最优模型大小被高估。在最优超参数下拟合消除了这种偏差，外推精度更高。

3. **最优模型大小与计算预算的关系**:

    - 功能：给定计算预算直接预测最优模型参数量
    - 核心思路：拟合得到 $N_{\text{opt}} = 1.5787 \cdot C^{0.4146}$，指数 0.4146 介于 OpenAI（0.73）和 Chinchilla（0.50）之间，说明 Video DiT 需要比 LLM 更均衡的模型-数据分配。以 Movie Gen 的 6144 H100 GPU 计算预算为例，本文方法推荐 18.05B 模型（vs 固定超参数方法的 30.05B），减少 39.9% 参数但性能相当。
    - 设计动机：实际部署中推理成本与模型大小直接相关。如果能用更小的模型达到相似性能，就能大幅降低部署成本。

### 损失函数 / 训练策略

标准 DDPM 去噪目标。使用固定学习率（非余弦退火）简化问题。实验在 0.017B/0.057B/0.13B/0.26B 四个模型上进行，每个模型训练 2B-12B tokens 的多种配置。计算 $C_{\text{token}} = \frac{3}{4}N(7 + n_{\text{ctx}}/d)$。

## 实验关键数据

### 主实验

| 模型大小 | 方法 | 推荐模型 | 验证损失 |
|---------|------|---------|---------|
| Movie Gen 预算 | 固定超参数 | 30.05B | 基准 |
| Movie Gen 预算 | **最优超参数** | **18.05B (-39.9%)** | **相当** |
| 1e10 TFlops | 固定超参数 | 基准 | 基准 |
| 1e10 TFlops | **最优超参数** | **-40.1% 推理成本** | **相当** |

### 消融实验（超参数预测验证 - 1.07B 模型）

| 配置 | 4B tokens 预测 | 4B tokens 实际 | 10B tokens 预测 | 10B tokens 实际 |
|-----|-------------|-------------|-------------|-------------|
| 最优 BS | 预测值 | ✓ 对应最小损失 | 预测值 | ✓ 对应最小损失 |
| 最优 LR | 预测值 | ✓ 对应最小损失 | 预测值 | ✓ 对应最小损失 |

### 关键发现

- **Video DiT 比 LLM 对超参数更敏感**：使用固定非最优超参数拟合的缩放法则预测的最优模型大小偏大 39.9%，这在 LLM 中不明显但在 Video DiT 中影响巨大。
- **模型大小影响最优超参数**：更大模型需要更大 batch size（$\gamma_B = 0.19 > 0$）和更小学习率（$\gamma_\eta = -0.16 < 0$），这与 SGD 收敛理论一致（大模型 Lipschitz 常数更大）。
- **最优区域附近损失平坦**：在固定计算预算下偏离最优模型大小 ±30% 导致的损失增加 <1%，使得在推理效率和性能间灵活权衡成为可能。
- 缩放法则从 0.26B 成功外推到 1.07B（4× 放大），验证了外推的可靠性。
- 附录中验证了相同公式在图像 DiT（1帧视频）上也成立。

## 亮点与洞察

- **将超参数和模型大小联合建模为缩放法则的一部分**是方法论上的关键贡献：以往研究要么忽略超参数（假设fixed good enough）要么假设超参数仅与数据量有关。本文从理论和实验双重验证了模型大小是超参数选择的独立变量。
- **实用价值极高**：训练 Video DiT 代价巨大（Movie Gen 用 6144 块 H100），用小模型实验拟合缩放法则后可在动手训练前就确定最优配置，节省大量计算资源。
- 平坦最优区域的发现为实际部署提供了灵活性——不必使用理论最优模型大小，可以根据硬件约束选择略小的模型。

## 局限与展望

- 仅在验证损失上建立缩放法则，未与下游视频质量指标（如 FVD、人类偏好）直接关联。
- 使用固定学习率简化问题，现实训练中常用的余弦退火可能改变最优配置。
- 外推仅验证到 1.07B（约 4× 放大），在 10B+ 规模上的准确性未知。
- 数据质量、数据多样性等因素对缩放法则的影响未纳入分析。

## 相关工作与启发

- **vs Chinchilla (Hoffmann et al.)**: Chinchilla 对 LLM 建立了模型-数据等比分配法则（指数 0.50），Video DiT 的指数为 0.4146，说明视频模型需要分配更多比例给数据。但 Chinchilla 使用 IsoFLOP 方法，本文发现其损失拟合不够精确。
- **vs DeepSeek Scaling Law**: DeepSeek 首次建模了最优超参数，但未将模型大小作为独立变量。本文扩展了这一思路并提供了理论支撑。
- **vs Movie Gen**: Movie Gen 使用 30B 参数，本文缩放法则建议在同等计算预算下仅需 18B 即可达到相同性能，减少 40% 推理成本。

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次为 Video DiT 建立包含超参数的精确缩放法则
- 实验充分度: ⭐⭐⭐⭐ 理论推导+小模型拟合+大模型验证的完整流程
- 写作质量: ⭐⭐⭐⭐⭐ 理论分析严谨，公式推导清晰
- 价值: ⭐⭐⭐⭐⭐ 对视频生成模型的训练具有直接的实际指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] DiTFlow: Video Motion Transfer with Diffusion Transformers](video_motion_transfer_with_diffusion_transformers.md)
- [\[CVPR 2025\] Mimir: Improving Video Diffusion Models for Precise Text Understanding](mimir_improving_video_diffusion_models_for_precise_text_understanding.md)
- [\[CVPR 2025\] VideoDirector: Precise Video Editing via Text-to-Video Models](videodirector_precise_video_editing_via_text-to-video_models.md)
- [\[CVPR 2025\] MotionPro: A Precise Motion Controller for Image-to-Video Generation](motionpro_a_precise_motion_controller_for_image-to-video_generation.md)
- [\[CVPR 2026\] ReHyAt: Recurrent Hybrid Attention for Video Diffusion Transformers](../../CVPR2026/video_generation/rehyat_recurrent_hybrid_attention_for_video_diffusion_transformers.md)

</div>

<!-- RELATED:END -->
