---
title: >-
  [论文解读] MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space
description: >-
  [图像生成] 提出 MotionStreamer，将连续因果潜空间与扩散头结合到自回归框架中，实现文本条件下的流式人体动作生成，支持在线多轮生成和动态运动组合。 流式动作生成（streaming motion generation）要求模型在接收到增量文本输入的同时，逐步生成连贯的人体动作序列，这对实时游戏、动画和机器人等应…
tags:
  - "图像生成"
---

# MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space

> **会议**: ICCV 2025
> **arXiv**: [2503.15451](https://arxiv.org/abs/2503.15451)
> **代码**: [项目页面](https://zju3dv.github.io/MotionStreamer/)
> **领域**: 动作生成·扩散模型·自回归
> **关键词**: streaming motion generation, causal latent space, diffusion head, autoregressive, text-to-motion

## 一句话总结

提出 MotionStreamer，将连续因果潜空间与扩散头结合到自回归框架中，实现文本条件下的流式人体动作生成，支持在线多轮生成和动态运动组合。

## 研究背景与动机

流式动作生成（streaming motion generation）要求模型在接收到增量文本输入的同时，逐步生成连贯的人体动作序列，这对实时游戏、动画和机器人等应用至关重要。现有方法面临以下核心挑战：

**扩散模型的固定长度限制**：传统扩散动作生成模型（如 MDM、MLD）需要预定义运动长度，无法动态响应在线文本输入，缺乏增量生成能力。

**GPT 方法的延迟与误差累积**：基于离散 VQ 的自回归方法（如 T2M-GPT、MotionGPT）使用非因果分词器，导致无法在线解码部分 token，延迟较高；同时离散化带来的量化误差在长序列自回归中逐步累积。

**固定窗口方法的局限**：DART 等实时方法依赖固定窗口局部运动原语（motion primitives），无法建模可变长度的历史上下文。

MotionStreamer 的核心思路是：**将扩散头集成到自回归框架中预测连续运动潜变量，并引入因果运动压缩器实现在线解码**，从而兼顾流式生成、在线响应和长期一致性。

## 方法详解

### 整体框架

MotionStreamer 由三个核心组件构成（如 Fig. 2 所示）：

- **预训练文本编码器**：使用 T5-XXL 提取文本特征 $T_i \in \mathbb{R}^{1 \times d_t}$
- **因果时序自编码器（Causal TAE）**：将原始运动序列编码为连续因果潜变量序列
- **扩散自回归模型**：基于 Transformer + 扩散头，在因果潜空间中自回归预测下一步运动潜变量

### 关键设计 1：因果时序自编码器（Causal TAE）

Causal TAE 采用 1D 因果卷积构建编码器 $\mathcal{E}$ 和解码器 $\mathcal{D}$，通过特殊的时序填充方案保证因果性：对于卷积核大小 $k_t$、步幅 $s_t$、膨胀率 $d_t$ 的卷积层，在序列开头填充 $(k_t - 1) \times d_t + (1 - s_t)$ 帧。

给定运动序列 $X = \{x_1, \ldots, x_N\}$（$x_t \in \mathbb{R}^{272}$），Causal TAE 输出连续潜变量 $Z = \{z_1, \ldots, z_{N/l}\}$（$z_i \in \mathbb{R}^{d_c}$），其中 $l=4$ 为时序下采样率，$d_c=16$ 为潜维度。

运动表示采用 272 维 SMPL 6D 旋转向量：$x = \{\dot{r}^x, \dot{r}^z, \dot{r}^a, j^p, j^v, j^r\}$，直接用于驱动 SMPL 角色，无需后处理。

### 关键设计 2：扩散自回归生成器

每个训练样本表示为 $S_i = (T_i, C_i, Z_i)$，其中 $C_i$ 是历史运动潜变量，$Z_i$ 是当前运动潜变量。沿时间轴拼接后送入带因果 mask 的 Transformer，得到中间潜变量 $\{c_i^1, \ldots, c_i^n\}$，再由扩散头（小型 MLP）预测运动潜变量。

训练损失为标准扩散损失：

$$\mathcal{L} = \mathbb{E}_{\epsilon, t}\left[\|\epsilon - \epsilon_\theta(Z_t | t, C_i, T_i)\|^2\right]$$

### 关键设计 3：Two-Forward 训练策略

为缓解自回归的 exposure bias，提出两次前向策略：第一次用 GT 潜变量前向生成，第二次将部分 GT 替换为第一次预测结果进行混合前向，仅在第二次反传梯度。替换比例由余弦调度器控制。

### 连续停止条件

编码一个"不可能姿势"（全零向量）为参考结束潜变量，当生成的潜变量与参考结束潜变量距离低于阈值时停止生成，实现自动确定生成长度。

### 损失函数

Causal TAE 训练使用 $\sigma$-VAE 损失加根关节损失：

$$\mathcal{L} = \mathcal{L}_{recon} + D_{KL}(q(z|x) \| p(z)) + \lambda \mathcal{L}_{root}$$

## 实验

### 主实验：文本到动作生成（Tab. 1）

| 方法 | FID ↓ | R@3 ↑ | MM-Dist ↓ | Div → |
|------|-------|-------|-----------|-------|
| Real motion | 0.002 | 0.914 | 15.151 | 27.492 |
| MDM | 23.454 | 0.764 | 17.423 | 26.325 |
| T2M-GPT | 12.475 | 0.838 | 16.812 | 27.275 |
| MoMask | 12.232 | 0.846 | 16.138 | 27.127 |
| **Ours** | **11.790** | **0.859** | **16.081** | 27.284 |

MotionStreamer 在 FID、R@3 和 MM-Dist 上均优于所有基线方法。

### 长序列生成实验（Tab. 2, BABEL 数据集）

| 方法 | 子序列 FID ↓ | 过渡 FID ↓ | PJ → | AUJ ↓ |
|------|------------|-----------|------|-------|
| DoubleTake | 23.937 | 51.232 | 0.48 | 1.83 |
| FlowMDM | 18.736 | 34.721 | 0.06 | 0.51 |
| VQ-LLaMA | 24.342 | 36.293 | 0.08 | 1.20 |
| **Ours** | **15.743** | **32.888** | **0.04** | 0.90 |

在长期生成任务中，MotionStreamer 的子序列 FID 和过渡 FID 均显著优于基线。

### 消融实验（Tab. 3）

| 压缩器 | 重建 FID ↓ | MPJPE ↓ | 生成 FID ↓ |
|--------|----------|---------|----------|
| VQ-VAE | 5.173 | 63.9 mm | 13.226 |
| AE | 0.001 | 1.7 mm | 43.828 |
| VAE（非因果） | 2.092 | 26.2 mm | 19.902 |
| **Causal TAE** | 0.661 | 22.9 mm | **11.790** |

关键发现：AE 重建质量最好但生成性能最差（潜空间缺乏正则化）；Causal TAE 在重建和生成之间取得最佳平衡。

### 关键发现

- 连续潜空间避免了 VQ 离散化的信息损失，有效降低误差累积
- 因果结构的潜空间天然适配自回归生成的因果 mask
- First-frame Latency 实验表明 Causal TAE 的首帧延迟最低且不随序列长度增长

## 亮点与洞察

1. **连续+因果的潜空间设计**是关键创新，同时解决了 VQ 误差累积和在线解码两个问题
2. **Two-Forward 策略**有效缓解了自回归训练中的 exposure bias，且保持了并行训练效率
3. **连续停止条件**比二分类停止器更优雅，避免了类别不平衡问题
4. 支持多轮生成、长序列生成和动态运动组合等丰富应用

## 局限性

- 依赖 SMPL 骨架表示，难以直接推广到非人形角色
- 历史上下文长度受限于 Transformer 序列长度
- 长序列生成中仍存在一定的语义漂移现象

## 相关工作

- 扩散动作生成：MDM、MLD、MotionDiffuse
- 自回归动作生成：T2M-GPT、MotionGPT、MoMask
- 实时控制：CAMDM、AMDM、DART、CLoSD

## 评分

- **新颖性**: ★★★★☆ — 因果潜空间+扩散头的组合设计新颖
- **技术深度**: ★★★★☆ — Two-Forward 策略和连续停止条件设计精巧
- **实验质量**: ★★★★☆ — 在多个基准上全面验证，消融详尽
- **写作质量**: ★★★★☆ — 结构清晰，图表表达力强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Less is More: Improving Motion Diffusion Models with Sparse Keyframes](less_is_more_improving_motion_diffusion_models_with_sparse_keyframes.md)
- [\[ICCV 2025\] EmotiCrafter: Text-to-Emotional-Image Generation based on Valence-Arousal Model](emoticrafter_text-to-emotional-image_generation_based_on_valence-arousal_model.md)
- [\[ICCV 2025\] MamTiff-CAD: Multi-Scale Latent Diffusion with Mamba+ for Complex Parametric Sequence](mamtiff-cad_multi-scale_latent_diffusion_with_mamba_for_complex_parametric_seque.md)
- [\[ICCV 2025\] Timestep-Aware Diffusion Model for Extreme Image Rescaling](timestep-aware_diffusion_model_for_extreme_image_rescaling.md)
- [\[NeurIPS 2025\] InfinityStar: Unified Spacetime AutoRegressive Modeling for Visual Generation](../../NeurIPS2025/image_generation/infinitystar_unified_spacetime_autoregressive_modeling_for_v.md)

</div>

<!-- RELATED:END -->
