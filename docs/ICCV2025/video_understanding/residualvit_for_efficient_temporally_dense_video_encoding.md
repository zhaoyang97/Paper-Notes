---
title: >-
  [论文解读] ResidualViT for Efficient Temporally Dense Video Encoding
description: >-
  [ICCV 2025][视频理解][视频编码效率] 本文提出 ResidualViT，通过类比视频压缩中的 I帧/P帧 策略，交替使用完整 ViT 和轻量残差 ViT 编码视频帧，在保持接近原始 CLIP 精度的同时，实现最高 60% 的计算成本降低和 2.5 倍推理加速。 许多视频理解任务（如自然语言时序视频定位 NLTV…
tags:
  - "ICCV 2025"
  - "视频理解"
  - "视频编码效率"
  - "ViT"
  - "时序冗余"
  - "知识蒸馏"
  - "时间密集特征"
---

# ResidualViT for Efficient Temporally Dense Video Encoding

**会议**: ICCV 2025  
**arXiv**: [2509.13255](https://arxiv.org/abs/2509.13255)  
**代码**: 无  
**领域**: 视频理解  
**关键词**: 视频编码效率, ViT, 时序冗余, 知识蒸馏, 时间密集特征

## 一句话总结
本文提出 ResidualViT，通过类比视频压缩中的 I帧/P帧 策略，交替使用完整 ViT 和轻量残差 ViT 编码视频帧，在保持接近原始 CLIP 精度的同时，实现最高 60% 的计算成本降低和 2.5 倍推理加速。

## 研究背景与动机
许多视频理解任务（如自然语言时序视频定位 NLTVG、时序活动定位 TAL、音频描述生成 AD）需要"时间密集"的帧级推理，通常要求在 1-5 FPS 的高时间分辨率下采样帧。然而，当帧率从 0.1-0.5 FPS 提升到 1-5 FPS 时，计算资源需求增加 2-50 倍，这对大规模部署构成了巨大挑战。

现有的减少计算成本的方法主要依赖将大模型蒸馏到更小的模型中，但这种方式往往导致识别精度下降。更关键的是，这些方法将视频帧独立对待，**没有利用视频中固有的时间冗余性**——相邻帧通常在视觉上高度相似。

本文的核心洞察是：视频压缩技术长期利用这种时间冗余性（I帧存储完整信息，P帧只存储差异），同样的策略可以应用于视觉特征编码。通过在少量帧上运行完整 ViT（I-features），在相邻帧上运行高效近似编码器（P-features），可以在几乎不损失精度的情况下大幅降低计算成本。

## 方法详解

### 整体框架
ResidualViT 采用交叉编码策略：对视频帧序列中每 $N+1$ 帧使用完整的 CLIP ViT 编码器 $\mathcal{E}_\mathcal{V}$ 计算 I-feature，后续 $N$ 帧使用轻量 ResidualViT 编码器 $\mathcal{E}_\mathcal{S}$ 计算 P-feature。P-feature 的计算利用了前一个 I-feature 提供的时间上下文信息。

### 关键设计
1. **Token 缩减模块 $\mathcal{R}$**:

    - 功能：在 ResidualViT 编码 P-feature 时，大幅减少输入 token 的数量
    - 核心思路：采用 PatchDropout 策略，以概率 $p$ 丢弃部分 patch token，保留最具信息量的 token。探索了随机、均匀、中心和基于运动的丢弃策略
    - 设计动机：ViT 的计算复杂度随 token 数量二次增长，减少 token 数可显著降低编码成本。消融实验表明 token dropping 在效率-精度权衡上优于 token merging 和分辨率降低

2. **残差 Tokenizer 模块 $\mathcal{A}$**:

    - 功能：将 I-feature $f_t$ 转换为残差 token，注入到 P-feature 的计算中
    - 核心思路：通过一个可学习的线性映射 $\mathcal{A}: \mathbb{R}^b \rightarrow \mathbb{R}^d$ 将 I-feature 变换为与 ViT 输入空间兼容的 token，然后与 [CLS] token 和稀疏帧 token 拼接后送入 ViT
    - 设计动机：Token 缩减不可避免地丢弃了部分视觉信息，而相邻帧的时间连续性意味着前一帧的特征包含了大量可复用的语义信息。残差 token 仅增加约 0.1 GFLOPs（占帧编码成本的 0.1%），代价极小

3. **交叉编码策略**:

    - 功能：确定 I-feature 和 P-feature 的交替频率
    - 核心思路：平均编码成本为 $C = C_{\mathcal{E}_\mathcal{V}} \frac{1+(1-p)N}{1+N}$，当 $N>0$ 且 $p>0$ 时，$C$ 严格小于 $C_{\mathcal{E}_\mathcal{V}}$
    - 设计动机：$N=2$ 是最佳权衡点——成本节省达 56%，而精度几乎无损；$N$ 过大时，I-feature 与 P-feature 的时间距离过远，视觉相关性减弱

### 损失函数 / 训练策略
采用视觉-语言特征蒸馏训练：教师网络为原始 CLIP ViT 编码器 $\mathcal{E}_\mathcal{V}$，学生网络为 ResidualViT $\mathcal{E}_\mathcal{S}$。损失函数为双向软目标交叉熵：

$$\mathcal{J}_{L \rightarrow V} = -\sum_{i=1}^{B}\sum_{k=1}^{N}\sum_{j=1}^{B} \sigma_j(g^\top f_{i,t+k}^{(\mathcal{V})}) \log(\sigma_j(g^\top f_{i,t+k}^{(\mathcal{S})}))$$

最终损失为 $\min_\mathcal{A}(\mathcal{J}_{L \rightarrow V} + \mathcal{J}_{V \rightarrow L})$。关键特点：
- 仅训练残差 tokenizer $\mathcal{A}$（单层线性变换），ViT 权重冻结
- 在 WebVid-2.5M 上训练 5 个 epoch，使用 4 块 V100 GPU
- 不仅鼓励视觉特征接近，还保持 CLIP 的视觉-语言联合空间对齐

## 实验关键数据

### 主实验

| 数据集 | 指标 | ResidualViT (L/14) | CLIP (L/14) | 成本节省 |
|--------|------|---------------------|-------------|---------|
| Charades-STA | R@1, IoU=0.5 | 41.5 | 42.9 | 56% |
| Charades-STA | R@1, IoU=0.7 | 23.8 | 24.1 | 56% |
| ActivityNet-Captions | R@1, IoU=0.5 | 28.3 | 29.1 | 56% |
| MAD (长视频) | R@1, IoU=0.5 | 4.3 | 5.0 | 56% |
| MAD (长视频) | R@1, IoU=0.3 | 7.3 | 8.6 | 56% |

### 消融实验

| 配置 | R@1 IoU=0.5 | R@1 IoU=0.7 | 平均成本 (GFLOPs) | 说明 |
|------|------------|------------|-------------------|------|
| CLIP baseline | 42.9 | 24.1 | 233.4 | 上界 |
| 仅 Token Reduction | 28.5 | 14.5 | 35.7 (-85%) | 精度大幅下降 |
| + Interleave (N=2) | 38.9 | 22.8 | 102.0 (-56%) | 恢复大部分精度 |
| + 残差 Tokenizer (蒸馏) | 41.5 | 23.8 | 102.6 (-56%) | 接近原始精度 |

### 关键发现
- 单独使用 Token Reduction 会导致 34-40% 的相对精度下降，但结合交叉编码后精度损失缩小到仅 5-9%
- 残差 tokenizer 在计算成本几乎不增加的情况下，进一步将精度提升约 2.6 个百分点（IoU=0.5）
- $N=2$ 是最佳交叉因子：56% 的成本节省与几乎无损的精度
- 方法在零样本和有监督设置下均表现出色，且适用于短视频和长视频
- ResidualViT 结合零样本 grounding 算法，在 MAD 长视频数据集上将 SOTA 提升至 3.1 R@1（IoU=0.5），同时计算量减半

## 亮点与洞察
- **简洁有效的类比**：将视频压缩的 I帧/P帧 概念引入特征编码，直觉清晰且效果显著
- **极低训练成本**：只需训练一个线性层（残差 tokenizer），不需要大规模训练数据，整个训练在 4 块 V100 上 5 个 epoch 即可完成
- **保持视觉-语言对齐**：蒸馏目标不仅逼近视觉特征，还保持了 CLIP 的多模态空间，这是方法能在零样本设置下工作的关键
- **广泛的任务泛化性**：在 NLTVG、AD、TAL、AR 四个任务和五个数据集上均有效，覆盖短视频和长视频
- **实际加速**：不仅 GFLOPs 减少，实际推理墙钟时间也提升 2.5 倍，工程价值明确

## 局限与展望
- 当 $N$ 较大（>3）时精度下降明显，对于变化剧烈的视频场景（如快速切换、大幅度运动），时间冗余假设可能不成立
- 目前仅在 CLIP 上验证，是否能推广到其他视觉基础模型（如 DINOv2、SigLIP、InternVL）有待探索
- 基于运动的 token dropping 策略依赖光流估计，可能引入额外计算开销，在实际部署中需要权衡
- 长视频中场景切换时，I-feature 对后续 P-feature 的参考价值可能急剧下降，需要自适应检测机制
- 训练仅使用 WebVid-2.5M 数据集，在更大规模或不同领域（如医学、遥感）的泛化性需要进一步验证
- 未探索不同 token reduction 策略之间的动态组合或自适应选择

## 相关工作与启发
- 与视频压缩的联系提示我们：**视频的时间冗余性是降低计算成本的宝贵资源**，这一洞察可以推广到更多视频理解场景
- Token reduction 技术（PatchDropout、ToMe）正在成为高效 ViT 的重要工具，值得在更多架构中探索
- 轻量级蒸馏策略（只训练几个参数）展示了大模型适配的高效范式，对资源受限场景有参考价值
- 交叉编码的 I/P 设计思想可以推广到视频 embedding 的存储优化——仅存 I-feature，P-feature 在需要时在线计算
- 视觉-语言联合蒸馏（而非仅视觉蒸馏）是保持多模态能力的关键策略，可应用于其他 VLM 压缩场景

## 评分
- 新颖性: ⭐⭐⭐⭐ 视频压缩到特征编码的类比巧妙，但token reduction和蒸馏都是已有技术
- 实验充分度: ⭐⭐⭐⭐⭐ 四个任务五个数据集，零样本和有监督设置均覆盖，消融全面
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，动机-方法-实验逻辑链完整，图表精美
- 价值: ⭐⭐⭐⭐ 对时间密集视频任务有实际工程价值，但通用性还需验证

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Sparse-Dense Side-Tuner for Efficient Video Temporal Grounding](sparse-dense_side-tuner_for_efficient_video_temporal_grounding.md)
- [\[ICCV 2025\] AllTracker: Efficient Dense Point Tracking at High Resolution](alltracker_efficient_dense_point_tracking_at_high_resolution.md)
- [\[ICCV 2025\] Factorized Learning for Temporally Grounded Video-Language Models](factorized_learning_for_temporally_grounded_video-language_models.md)
- [\[CVPR 2025\] EDCFlow: Exploring Temporally Dense Difference Maps for Event-based Optical Flow Estimation](../../CVPR2025/video_understanding/edcflow_exploring_temporally_dense_difference_maps_for_event-based_optical_flow_.md)
- [\[ICCV 2025\] TOGA: Temporally Grounded Open-Ended Video QA with Weak Supervision](toga_temporally_grounded_open-ended_video_qa_with_weak_supervision.md)

</div>

<!-- RELATED:END -->
