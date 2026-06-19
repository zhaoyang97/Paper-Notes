---
title: >-
  [论文解读] VQToken: Neural Discrete Token Representation Learning for Extreme Token Reduction in Video Large Language Models
description: >-
  [NeurIPS 2025][模型压缩][token reduction] VQToken 提出了首个基于向量量化的视频 token 极限压缩框架，通过自适应离散化将连续 ViT embedding 聚类为紧凑码本，并用 token hash 函数保留时空位置信息，在 NextQA-MC 上仅用原始 0.07% 的 token（约 13 个）实现了仅 0.66% 的精度损失。
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "token reduction"
  - "量化"
  - "video LLM"
  - "discrete representation"
  - "token information density"
---

# VQToken: Neural Discrete Token Representation Learning for Extreme Token Reduction in Video Large Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2503.16980](https://arxiv.org/abs/2503.16980)  
**代码**: [https://github.com/](https://github.com/) (有，含 Homepage、GitHub、HuggingFace)  
**领域**: 模型压缩 / 视频LLM效率  
**关键词**: token reduction, vector quantization, video LLM, discrete representation, token information density

## 一句话总结

VQToken 提出了首个基于向量量化的视频 token 极限压缩框架，通过自适应离散化将连续 ViT embedding 聚类为紧凑码本，并用 token hash 函数保留时空位置信息，在 NextQA-MC 上仅用原始 0.07% 的 token（约 13 个）实现了仅 0.66% 的精度损失。

## 研究背景与动机

视频大语言模型（vLLM）面临严峻的计算效率挑战：视频输入需要将每帧像素 tokenize 后拼接成超长序列，Transformer 的注意力机制对序列长度 $n$ 呈 $O(n^2 DL)$ 复杂度。关键瓶颈在于 token 序列长度 $n$ 的影响远超模型参数和层数。

现有 token 压缩方法存在三大问题：

**Token Pruning** 直接移除 token 会丢失关键信息，破坏位置编码

**Token Merging**（如 ToMe、VidToMe）采用固定压缩比，灵活性差，且序列仍然很长
3. 压缩后的 token 仍高度连续相似，信息密度低，难以进一步压缩

根本原因在于：(1) 固定数量/比例的压缩策略要么压缩不够要么损失过大；(2) 缺乏自适应的上下文感知机制选择最有信息量的 token；(3) 没有利用向量量化将 token 聚类为离散类别以提升信息密度。

本文的核心思路：**用向量量化（VQ）将连续 ViT embedding 聚类为极少量离散 token，配合 token hash 函数保留时空位置关系，实现 99.9%+ 的 token 压缩率且几乎不损失性能。**

## 方法详解

### 整体框架

VQToken 框架流程：
1. 输入视频经 ViT tokenize 为连续视觉 token 序列
2. 自适应离散化过程（Adaptive Discrete Process）将 token 聚类并量化为紧凑码本
3. Token hash 函数记录每个 token 的原始时空位置并映射到最近的码本条目
4. VQ-Attention 模块整合码本与位置索引，生成保留位置信息的压缩 token 序列
5. 压缩 token + tokenized query 送入 LLM 进行零样本推理

### 关键设计

1. **自适应离散化过程（Adaptive Discrete Process）**：

    - 对连续 ViT token embedding 使用余弦相似度进行向量量化聚类
    - 固定长度压缩：使用经典 K-Means
    - 自适应长度压缩：使用自适应 K-Means 变体（根据视频内容复杂度动态决定聚类数 K）
    - 输出：K 个聚类中心作为离散码本条目 + 每个 token 的聚类分配

2. **简洁 Token 码本（Concise Token Codebook）**：

    - 每个码本条目 $b_k$ 为对应聚类中所有 token embedding 的质心：$b_k = \frac{1}{|s_k|} \sum_{i \in s_k} t_i$
    - 码本 $B \in \mathbb{R}^{K \times D}$ 捕获代表性视觉模式，以最小冗余存储

3. **Token Hash 函数映射**：

    - 构建 3D 索引图 $M \in \{1,...,K\}^{T \times H \times W}$（T=帧数，H/W=ViT 网格尺寸）
    - $M_{f,h,w} = c_i$，记录每个网格位置对应的聚类索引
    - 保留了时空位置编码，类似轻量级的运动跟踪替代方案（替代光流等昂贵方法）

4. **VQ-Attention 模块**：

    - 将索引图展平后通过 MLP 映射：$\widetilde{M} = \text{MLP}(\text{Flatten}(M)) \in \mathbb{R}^{K \times D}$
    - 多头注意力融合码本和位置信息：$B' = \text{MultiHeadAttn}(Q=BW_Q, K=BW_K, V=\widetilde{M}W_V)$
    - 输出 $B' \in \mathbb{R}^{K \times D}$ 为最终压缩 token，携带运动上下文

### 损失函数 / 训练策略

- 基于 LLaVA-OneVision 0.5B（Qwen2 backbone）
- 训练数据：LLaVA-Video-178K 数据集，1.3M 指令跟随样本
- 4 块 A100 GPU，85K 迭代，AdamW + cosine decay
- VQ-Attention 学习率 1×10⁻⁵，ViT backbone 学习率 2×10⁻⁶
- Zero2 优化，batch size 8，梯度累积 2 步

### Token 信息密度指标（TokDense）

- 定义：$\text{TokDense} = \frac{\text{Accuracy}}{\text{Token Count}}$
- 衡量每个保留 token 对任务性能的贡献
- 本文还定义了模块复杂度（token 压缩模块本身的开销）和 LLM 复杂度（压缩后下游推理的开销）

## 实验关键数据

### 主实验：与 vLLM 基线对比

| 模型 | 参数量 | Zero-Shot | 准确率(%) | Token 数量% |
|------|--------|-----------|-----------|-------------|
| LLaVA-OneVision | 0.5B | ✓ | 57.2 | 100% |
| LLaVA-OV-SI | 0.5B | ✓ | 53.6 | 27% |
| VQToken (Ours) | 0.5B | ✓ | **57.5** | **0.14%** |
| Mistral | 7B | ✓ | 51.1 | 100% |
| LLoVi | 7B | ✓ | 54.3 | 100% |
| MVU | 7B | ✓ | 55.2 | 100% |

### 极限 Token 压缩任务

**固定长度子任务**：

| 方法 | Token=12 | Token=32 | Token=64 |
|------|----------|----------|----------|
| Token Pruning | 29.12 | 34.50 | 31.31 |
| ToMe | 35.72 | 38.50 | 40.10 |
| VidToMe | 39.64 | 45.10 | 46.20 |
| VQToken (Ours) | **57.03** | **57.46** | **57.10** |

**自适应长度子任务**：

| 方法 | 平均 Token 数 | 准确率 | TokDense |
|------|--------------|--------|----------|
| Interpolating | 3136 | 57.20 | 0.018 |
| DyCoke | 1662.12 | 57.70 | 0.035 |
| Ours-Fixed(m=32) | 32 | 57.46 | 1.796 |
| Ours-Dynamic | **13.08** | **57.72** | **4.413** |

### 消融实验

| VLM | Codebook | Hash Fn. | VQ-Attn | 准确率 | Tokens | TokDense |
|-----|----------|----------|---------|--------|--------|----------|
| ✓ | — | — | — | 57.2 | 23328 | 0.002 |
| ✓ | ✓ | — | — | 35.2 | 32 | 1.100 |
| ✓ | ✓ | ✓ | ✓ | **57.5** | **32** | **1.797** |
| ✓ | rand | ✓ | ✓ | 37.7 | 32 | 1.178 |
| ✓ | ✓ | rand | ✓ | 46.9 | 32 | 1.466 |

### 关键发现

- VQToken 仅用 0.14%（32个）的 token 即可匹配甚至略超 100% token 的基线性能
- 自适应模式下平均仅需 13.08 个 token，TokDense 达到 4.413，是 Interpolation 方法的 245×
- 0.5B 参数的 VQToken 在零样本设定下超越了多个 7B 参数的 vLLM
- 所有三个组件（码本、hash 函数、VQ-Attention）缺一不可：仅有码本但无位置信息会丢失 22% 准确率，VQ-Attention 是恢复性能的关键
- 在 MVBench 的 20 个子任务上表现均衡，尤其在动作识别和物体交互任务上突出

## 亮点与洞察

- **极限压缩的可行性**：首次证明视频 token 可以压缩到原始的 0.07%（动态模式）且几乎不损失性能，彻底改变了对视频 token 冗余度的认知
- **离散化 vs 连续化**：与现有的 token pruning/merging（仍在连续空间操作）不同，VQToken 将视频表示转化为离散码本，信息密度大幅提升
- **Token Hash 函数设计精巧**：用简洁的 3D 索引图替代昂贵的运动跟踪（如光流），既保留了时空关系又几乎不增加计算开销
- **TokDense 评价指标**：提出了"每 token 贡献的准确率"这一新指标，为极限压缩场景下的方法比较提供了更合理的衡量标准

## 局限与展望

- 仅在 0.5B 参数的 LLaVA-OneVision 上验证，更大规模 LLM 的效果未知
- K-Means 聚类需要对全序列 token 一次性计算，难以流式处理
- 码本大小 K 的选择策略（自适应 K-Means）的鲁棒性和泛化性有待更多数据集验证
- 在需要精细空间定位的任务（如视频 grounding）上的表现未评估
- ActNet-QA 和 LongVideoBench 上相比完整 token 基线有约 4-6% 的精度下降，说明极限压缩在某些长视频理解任务上仍有信息损失

## 相关工作与启发

- 与 ToMe/VidToMe（token merging）形成直接对比：VQToken 通过离散化实现了数量级的更高压缩率
- 与 DyCoke（动态 KV cache 压缩）互补：DyCoke 在推理端压缩 KV cache，VQToken 在输入端压缩 token
- VQ 在视觉生成（VQ-VAE、VQ-GAN）中已被广泛验证，本文首次将其引入 vLLM 的 token 压缩领域
- 启发：离散化 + 位置索引的方案可能适用于其他超长序列任务（如长文档理解、音频处理等）

## 评分

- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Vision-centric Token Compression in Large Language Model](vision-centric_token_compression_in_large_language_model.md)
- [\[NeurIPS 2025\] Recurrent Attention-based Token Selection for Efficient Streaming Video-LLMs](recurrent_attention-based_token_selection_for_efficient_streaming_video-llms.md)
- [\[ICCV 2025\] Representation Shift: Unifying Token Compression with FlashAttention](../../ICCV2025/model_compression/representation_shift_unifying_token_compression_with_flashattention.md)
- [\[CVPR 2025\] Faster Parameter-Efficient Tuning with Token Redundancy Reduction (FPET)](../../CVPR2025/model_compression/faster_parameter-efficient_tuning_with_token_redundancy_reduction.md)
- [\[NeurIPS 2025\] Gated Integration of Low-Rank Adaptation for Continual Learning of Large Language Models](gated_integration_of_low-rank_adaptation_for_continual_learning_of_large_languag.md)

</div>

<!-- RELATED:END -->
