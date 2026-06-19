---
title: >-
  [论文解读] FastVAR: Linear Visual Autoregressive Modeling via Cached Token Pruning
description: >-
  [ICCV 2025][模型压缩][Visual Autoregressive] FastVAR 提出一种无需训练的后处理加速方法，通过观察 VAR 模型中大尺度步骤主要建模高频纹理且对剪枝鲁棒的特性，利用频域引导的关键 token 选择（PTS）仅保留高频 token 参与前向，并用缓存的早期尺度 token 恢复被剪枝的位置（CTR），在 FlashAttention 基础上实现额外 2.7× 加速且性能损失 <1%，并首次实现单张 3090 GPU 上 1.5 秒生成 2K 图像。
tags:
  - "ICCV 2025"
  - "模型压缩"
  - "Visual Autoregressive"
  - "next-scale prediction"
  - "剪枝"
  - "high-resolution generation"
  - "training-free acceleration"
  - "cached token restoration"
---

# FastVAR: Linear Visual Autoregressive Modeling via Cached Token Pruning

**会议**: ICCV 2025  
**arXiv**: [2503.23367](https://arxiv.org/abs/2503.23367)  
**代码**: [https://github.com/csguoh/FastVAR](https://github.com/csguoh/FastVAR)  
**领域**: 模型加速 / 视觉自回归 / Token 剪枝  
**关键词**: Visual Autoregressive, next-scale prediction, token pruning, high-resolution generation, training-free acceleration, cached token restoration  

## 一句话总结
FastVAR 提出一种无需训练的后处理加速方法，通过观察 VAR 模型中大尺度步骤主要建模高频纹理且对剪枝鲁棒的特性，利用频域引导的关键 token 选择（PTS）仅保留高频 token 参与前向，并用缓存的早期尺度 token 恢复被剪枝的位置（CTR），在 FlashAttention 基础上实现额外 2.7× 加速且性能损失 <1%，并首次实现单张 3090 GPU 上 1.5 秒生成 2K 图像。

## 研究背景与动机

视觉自回归（VAR）模型将传统的 next-token 预测范式转变为 next-scale 预测，大幅减少了生成步数。然而 VAR 面临严重的分辨率可扩展性问题：

**计算复杂度随分辨率剧增**：不同于 next-token 每步只处理一个 token，VAR 每步需处理整个 token map。Token 数量以 O(n²) 增长（n 为图像分辨率），注意力层达到 O(n⁴) 复杂度。

**大尺度步骤是延迟瓶颈**：即使启用 FlashAttention，推理延迟仍呈超线性增长。最后两个尺度步骤占总运行时间的 60%。

**无法扩展到高分辨率**：现有 VAR 模型在生成 2K 分辨率图像时因显存不足而 OOM，阻碍了实际应用。

**现有加速方法不适用**：扩散模型的加速方法（DeepCache、ToMeSD）无法直接用于 VAR；AR 的并行解码策略（speculative decoding）也不适用于 next-scale 范式。

## 方法详解

### 三个关键观察

**观察一：大尺度步骤是瓶颈但对剪枝鲁棒**
- 运行时分析：最后两个尺度步骤占 60% 总推理时间。
- 剪枝敏感性实验：大尺度步骤在相同剪枝比例下性能下降远小于小尺度步骤。
- 结论：应集中在大尺度步骤做 token 剪枝。

**观察二：大尺度步骤主要建模高频内容**
- 可视化中间预测 r̃_k：小尺度步骤生成主体轮廓（"结构构建阶段"），大尺度步骤添加纹理细节（"纹理填充阶段"）。
- 频谱分析：低频分量在大尺度步骤已基本收敛，高频分量仍有显著变化。
- 结论：可以剪枝冗余的低频 token，只保留高频 token。

**观察三：不同尺度的 token 存在跨尺度关联**
- 注意力图分析：当前尺度的 token 不仅关注同尺度邻居，还与前一尺度对应位置的 token 有强对角线稀疏相关性。
- 结论：可以用前一尺度的 token map 近似被剪枝位置的输出，补偿信息损失。

### 核心设计

**关键 Token 选择 (Pivotal Token Selection, PTS)**

针对频域 token 筛选的挑战（FFT 在频域工作，难以定位空间域中特定 token 的频率特性），PTS 提出近似方案：

1. 估计低频分量：对输入 x_k 做全局平均池化得到直流分量 x̄_k = global_avg_pool(x_k)。
2. 计算高频分量：高频 = x_k - x̄_k。
3. 关键性评分：s_k = ||x_k - x̄_k||₂（L2 范数）。
4. Token 选择：保留评分最高的 Top-K 个 token 作为关键 token。

PTS 的额外好处：减少输入 token 同时也减小了 KV-Cache 大小，优化了 GPU 显存和后续跨尺度注意力。

**缓存 Token 恢复 (Cached Token Restoration, CTR)**

为恢复被剪枝后的 2D 图像结构：

1. 在结构构建阶段的最后一步（第 K-N 步）缓存每层的输出 token map。
2. 对缓存 token map 做插值上采样到当前尺度大小：y_k^cache = interpolate(y_{K-N}, (h_k, w_k))。
3. 将插值后的缓存值按索引集 I 填入被剪枝的位置，恢复完整 token map。

这种设计利用了跨尺度的强对角线注意力相关性——前一尺度对应位置的 token 是被剪枝位置输出的良好近似。

**渐进式剪枝率调度**

更大的尺度步骤对剪枝更鲁棒，因此分配更大的剪枝比例：
- Infinity 模型：{40%, 50%, 100%, 100%}（最后两步完全跳过，直接插值）
- HART 模型：{50%, 75%}

### 实现特点
- **无需训练**：即插即用地应用于预训练 VAR 模型，与 backbone 无关。
- **兼容 FlashAttention**：在 FlashAttention 加速基础上叠加使用，进一步 2.7× 加速。
- **零样本高分辨率扩展**：通过缓存恢复机制，可以零样本生成训练分辨率之外的 2K 图像。

## 实验关键数据

### GenEval 基准 (1024×1024)

| 方法 | 类型 | 延迟 | 加速比 | GenEval Overall |
|------|------|-----:|-------:|----------------:|
| SDXL | Diffusion | 4.3s | - | 0.55 |
| SD3-medium | Diffusion | 4.4s | - | 0.62 |
| LlamaGen | AR | 37.7s | - | 0.32 |
| Show-o | AR | 50.3s | - | 0.68 |
| HART | VAR | 0.95s | 1.0× | 0.51 |
| **HART + FastVAR** | VAR | **0.63s** | **1.5×** | **0.51** |
| Infinity | VAR | 2.61s | 1.0× | 0.73 |
| **Infinity + FastVAR** | VAR | **0.95s** | **2.7×** | **0.72** |

- HART + FastVAR：1.5× 加速，GenEval 分数不变
- Infinity + FastVAR：2.7× 加速，GenEval 仅降 0.01
- 相比 LlamaGen，Infinity+FastVAR 实现 39.7× 加速 + 125% 性能提升

### MJHQ30K 基准 (FID)

| 方法 | 加速比 | landscape FID | people FID |
|------|-------:|--------------:|-----------:|
| HART | 1.0× | 25.43 | 30.61 |
| HART + FastVAR | 1.5× | 22.52 | 28.19 |
| Infinity | 1.0× | 24.68 | 30.27 |
| Infinity + FastVAR | 2.7× | 24.68 | 30.55 |

HART + FastVAR 在 people 类 FID 反而降低了 2.42（质量提升）。

### vs. Token Merging (ToMe)

| 方法 | 加速比 | FID↓ | GenEval↑ |
|------|-------:|-----:|---------:|
| ToMe (1.19×) | 1.19× | 29.07 | 0.48 |
| ToMe (1.36×) | 1.36× | 35.22 | 0.46 |
| **FastVAR (1.51×)** | **1.51×** | **28.19** | **0.51** |
| **FastVAR (1.70×)** | **1.70×** | **28.97** | **0.50** |

FastVAR 在更高加速比下保持更优 FID 和 GenEval，证明缓存恢复策略优于 token 合并策略。

### 2K 分辨率零样本生成
- 单张 NVIDIA 3090 GPU（24GB）
- 15GB 显存占用
- 1.5 秒生成一张 2K 图像
- 原始 baseline 因 OOM 无法生成

### 显存优化
- FlashAttention baseline：18.9GB
- FastVAR：14.7GB（节省 22.2%）

## 亮点与洞察

- **问题分析极其深入**：从延迟分析、频谱分析、注意力图分析三个角度系统性地揭示了 VAR 的计算特性，每个 observation 都直接指导方法设计。
- **简洁而优雅的方法**：PTS 只需一个全局平均池化 + L2 范数排序，CTR 只需一次插值 + 索引填充，实现复杂度极低。
- **"100% 剪枝率"的大胆设计**：最后两个尺度步骤可以完全跳过，直接插值，说明 VAR 大尺度步骤的冗余远超预期。
- **训练无关 + FlashAttention 兼容**：完全正交于模型训练和已有加速技术，可叠加使用。
- **首次实现消费级 GPU 上 2K 生成**：从"无法运行"到"15GB / 1.5s"，具有重要的工程实用价值。
- **频域视角的 token 重要性度量**：用直流分量近似低频的思路简单但有效，避免了 FFT 的额外开销。

## 局限与展望

- 100% 剪枝率（完全跳过最后两步）的适用性依赖具体 backbone，并非对所有 VAR 模型通用。
- PTS 的全局平均池化作为低频估计是粗糙近似，可能对复杂纹理场景不够精确。
- 渐进式剪枝率调度目前手动设定，未自适应，不同内容/分辨率可能需要调整。
- 主要在 class-conditional 和 text-to-image 上验证，未涉及 video VAR 或其他模态。
- 2K 零样本生成虽然可行但质量可能不如专门训练的高分辨率模型，缺乏与训练过高分辨率模型的直接对比。
- 缓存步骤选择 K-N 的理由不够充分，可能存在更优的缓存策略。

## 相关工作与启发

- **vs. CoDe**：CoDe 用模型集成（大模型处理小尺度 + 小模型处理大尺度），依赖不同大小模型的可用性。FastVAR 无需额外模型，更通用。
- **vs. ToMe/ToMeSD**：Token Merging 将多个 token 合并为一个，但在 VAR 中难以压缩整个 token map 到有限 token 数，性能下降快。FastVAR 用缓存恢复替代 token 合并，利用了 VAR 独特的跨尺度结构。
- **vs. DeepCache**：DeepCache 复用 U-Net 低分辨率层特征，FastVAR 用类似思路缓存早期尺度输出。核心区别在于 VAR 的多尺度自回归结构提供了天然的缓存层级。
- **对 VAR 生态的推动**：解锁了 VAR 在消费级硬件上的高分辨率应用，有望推动 VAR 模型在实际产品中的部署。

## 评分
- 新颖性: ⭐⭐⭐⭐ 三个核心观察驱动的方法设计逻辑清晰，缓存恢复策略充分利用 VAR 独特结构
- 实验充分度: ⭐⭐⭐⭐ GenEval + MJHQ30K 两个基准，两个 VAR backbone，详细消融（尺度敏感性、剪枝率、vs. ToMe）
- 写作质量: ⭐⭐⭐⭐⭐ 论文结构极佳，observe-then-design 的叙事方式层层递进，图示丰富直观
- 价值: ⭐⭐⭐⭐⭐ 训练无关的即插即用加速方案，首次实现消费级 GPU 2K 生成，对 VAR 社区有重大实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Bridging Continuous and Discrete Tokens for Autoregressive Visual Generation](bridging_continuous_and_discrete_tokens_for_autoregressive_visual_generation.md)
- [\[CVPR 2026\] Progressive Supernet Training for Efficient Visual Autoregressive Modeling](../../CVPR2026/model_compression/progressive_supernet_training_for_efficient_visual_autoregressive_modeling.md)
- [\[NeurIPS 2025\] Linear Attention for Efficient Bidirectional Sequence Modeling](../../NeurIPS2025/model_compression/linear_attention_for_efficient_bidirectional_sequence_modeling.md)
- [\[ICCV 2025\] Representation Shift: Unifying Token Compression with FlashAttention](representation_shift_unifying_token_compression_with_flashattention.md)
- [\[ICLR 2026\] AgilePruner: An Empirical Study of Attention and Diversity for Adaptive Visual Token Pruning in LVLMs](../../ICLR2026/model_compression/agilepruner_an_empirical_study_of_attention_and_diversity_for_adaptive_visual_to.md)

</div>

<!-- RELATED:END -->
