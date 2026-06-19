---
title: >-
  [论文解读] Bridging Continuous and Discrete Tokens for Autoregressive Visual Generation
description: >-
  [ICCV 2025][模型压缩][自回归生成] 提出TokenBridge，通过对预训练连续VAE特征进行后训练维度级量化，将连续token转化为离散token，在保持连续token高保真表示能力的同时，使用标准交叉熵损失进行简洁的自回归建模，在ImageNet 256×256上达到与连续方法可比的生成质量。
tags:
  - "ICCV 2025"
  - "模型压缩"
  - "自回归生成"
  - "视觉tokenizer"
  - "后训练量化"
  - "离散-连续桥接"
  - "维度自回归"
---

# Bridging Continuous and Discrete Tokens for Autoregressive Visual Generation

**会议**: ICCV 2025  
**arXiv**: [2503.16430](https://arxiv.org/abs/2503.16430)  
**代码**: [https://yuqingwang1029.github.io/TokenBridge](https://yuqingwang1029.github.io/TokenBridge)  
**领域**: 视觉生成 / 模型压缩  
**关键词**: 自回归生成, 视觉tokenizer, 后训练量化, 离散-连续桥接, 维度自回归

## 一句话总结

提出TokenBridge，通过对预训练连续VAE特征进行后训练维度级量化，将连续token转化为离散token，在保持连续token高保真表示能力的同时，使用标准交叉熵损失进行简洁的自回归建模，在ImageNet 256×256上达到与连续方法可比的生成质量。

## 研究背景与动机

自回归视觉生成模型面临一个核心困境：**离散token vs 连续token**。离散token（如VQ、LFQ）可以直接用交叉熵损失建模，但量化过程引入梯度近似导致训练不稳定，且有限的码本大小限制了表示能力。连续token（如VAE latent）虽能更好地保留视觉细节，但需要复杂的分布建模技术（如扩散头、GMM），增加了生成流程的复杂度。

作者观察到：能否**既享受连续token的高质量表示，又保持离散token建模的简洁性**？关键思路是将量化过程从tokenizer训练中解耦——先训好连续VAE，再通过后训练量化得到离散token。

## 方法详解

### 整体框架

TokenBridge包含两个核心组件：（1）后训练维度级量化策略，将预训练VAE的连续特征转为离散token；（2）高效的维度自回归预测机制，处理指数级大的token空间。

### 关键设计

1. **后训练维度级量化（Post-Training Dimension-wise Quantization）**:

    - 不在tokenizer训练中引入量化，而是在VAE完全训练好之后，对其连续特征 $\mathbf{X} \in \mathbb{R}^{H \times W \times C}$ 进行逐通道独立量化
    - 利用VAE特征的两个关键性质：（a）KL约束使值域有界；（b）近似高斯分布，可进行非均匀量化
    - 量化流程：先将每个维度归一化到 $[-r, r]$（$r=3$），然后基于标准正态CDF将分布等概率划分为 $B$ 个区间，每个区间用条件期望值作为重建值 $\gamma_i = \mathbb{E}[\xi | b_i \leq \xi < b_{i+1}]$
    - 反量化时将离散索引映射回连续值，直接送入预训练VAE解码器
    - 设计动机：避免了传统VQ的码本崩溃和梯度近似问题，且无需训练额外参数

2. **维度自回归预测头（Dimension-wise Autoregressive Head）**:

    - 维度级量化产生指数级大的token空间（$B^C$ 种组合），直接用softmax分类不可行
    - 并行独立预测各维度会忽略维度间的关键依赖关系（实验验证FID从1.94劣化到15.7）
    - 解决方案：在每个空间位置引入轻量级自回归头，将联合分布分解为：$p(\mathbf{q}) = \prod_{c=1}^{C} p(q^c | \mathbf{q}^{<c}, \mathbf{z})$
    - 每次只预测 $B$ 个类别的分类问题，计算可行
    - 设计动机：将大词汇空间的建模转化为一系列小分类问题，同时保留关键的通道间依赖

3. **基于频率的维度排序**:

    - 通过FFT分析各维度的频谱特性，按低频能量占比排序
    - 优先生成承载更多低频（结构）信息的维度，再生成高频（细节）维度
    - 设计动机：结构先行、细节后补，提升生成图像的整体一致性

### 损失函数 / 训练策略

- 训练使用标准交叉熵损失，无需复杂的分布建模
- 推理时先空间自回归生成各位置的上下文特征 $\mathbf{z}$，再由维度自回归头逐通道预测离散索引，每生成完一个空间位置的完整token后立即反量化为连续特征，作为下一个位置的输入条件
- 采用温度采样和classifier-free guidance

## 实验关键数据

### 主实验

| 方法 | Token类型 | 损失 | 参数量 | FID↓ | IS↑ | Recall↑ |
|------|----------|------|--------|------|-----|---------|
| LlamaGen | 训练量化离散 | CE | 3.1B | 2.18 | 263.3 | 0.58 |
| VAR | 训练量化离散 | CE | 2.0B | 1.73 | 350.2 | 0.60 |
| MAR-L | 连续 | Diff. | 479M | 1.78 | 296.0 | 0.60 |
| MAR-H | 连续 | Diff. | 943M | 1.55 | 303.7 | 0.62 |
| **Ours-L** | 后训练量化离散 | CE | 486M | **1.76** | 294.8 | 0.63 |
| **Ours-H** | 后训练量化离散 | CE | 910M | **1.55** | **313.3** | **0.65** |

### 消融实验

| 配置 | gFID↓ | IS↑ | 说明 |
|------|-------|-----|------|
| 并行预测 | 15.7 | 158.5 | 忽略维度间依赖，质量极差 |
| 自回归预测 | 1.94 | 306.1 | 捕获依赖关系，FID提升8× |
| B=16 量化 | 2.03 | 295.0 | 粗粒度量化 |
| B=64 量化 | 1.94 | 306.1 | 细粒度量化最优 |
| 默认维度顺序 | 1.94 | 306.1 | 基准 |
| 频率排序 | 1.89 | 307.3 | 轻微提升 |
| AR头 3M参数 | 2.88 | 277.3 | 最小配置仍可用 |
| AR头 94M参数 | 1.94 | 306.1 | 增大容量持续提升 |

### 关键发现

- 后训练量化在 $B=64$ 时重建质量（rFID=1.11）完全匹配连续VAE基线
- 维度自回归预测相比并行预测带来约8倍FID提升，证明通道间依赖关系至关重要
- 离散token天然支持置信度引导生成，可生成前景清晰、背景简洁的图像，这是连续方法不具备的优势

## 亮点与洞察

- **范式逆转**：传统方法在tokenizer训练时量化，本文反其道行之——先训好连续tokenizer再后训练量化，完美解耦了两个目标
- **指数空间的优雅处理**：$64^{16}$ 的巨大token空间通过维度自回归分解为16个64分类问题
- 证明了**标准交叉熵可以达到扩散头/GMM的生成质量**，大大简化了自回归视觉生成的流程

## 局限与展望

- 维度自回归增加了每个spatial token的推理步数（C步），可能影响生成速度
- 当前仅在ImageNet 256×256验证，未扩展到更高分辨率和文本条件生成
- 频率排序的提升有限，维度排序策略可能还有更优解

## 相关工作与启发

- 与MAR（连续token+扩散头）的直接对比最有说服力：相同参数量下FID持平，但本文方法训练更简单
- 后训练量化的思路类似于LLM中的PTQ技术（如GPTQ），将这一思想迁移到视觉tokenizer是有趣的跨领域创新
- 为统一多模态框架（视觉+语言token共用交叉熵建模）提供了可行路径

## 评分

- 新颖性: ⭐⭐⭐⭐ 后训练量化+维度自回归的组合思路新颖，范式逆转的设计很巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 消融实验全面，涵盖量化粒度、预测策略、AR头规模等关键变量
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，图示直观
- 价值: ⭐⭐⭐⭐ 为自回归视觉生成提供了一条简洁高效的新路线

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] PTQ4ARVG: Post-Training Quantization for AutoRegressive Visual Generation Models](../../ICLR2026/model_compression/ptq4arvg_post-training_quantization_for_autoregressive_visual_generation_models.md)
- [\[ICCV 2025\] FastVAR: Linear Visual Autoregressive Modeling via Cached Token Pruning](fastvar_linear_visual_autoregressive_modeling_via_cached_token_pruning.md)
- [\[NeurIPS 2025\] When Worse is Better: Navigating the Compression-Generation Trade-off in Visual Tokenization](../../NeurIPS2025/model_compression/when_worse_is_better_navigating_the_compression-generation_tradeoff_in_visual_to.md)
- [\[CVPR 2026\] Planning in 8 Tokens: A Compact Discrete Tokenizer for Latent World Model](../../CVPR2026/model_compression/planning_in_8_tokens_a_compact_discrete_tokenizer_for_latent_world_model.md)
- [\[ICCV 2025\] B-VLLM: A Vision Large Language Model with Balanced Spatio-Temporal Tokens](b-vllm_a_vision_large_language_model_with_balanced_spatio-temporal_tokens.md)

</div>

<!-- RELATED:END -->
