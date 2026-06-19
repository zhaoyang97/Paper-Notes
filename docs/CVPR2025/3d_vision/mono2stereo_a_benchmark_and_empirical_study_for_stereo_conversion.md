---
title: >-
  [论文解读] Mono2Stereo: A Benchmark and Empirical Study for Stereo Conversion
description: >-
  [CVPR 2025][3D视觉][stereo conversion] 构建首个大规模立体转换基准 Mono2Stereo（240 万对），提出立体质量指标 SIoU（与人类判断相关性 0.84 Spearman）和双条件扩散模型 + Edge Consistency 损失，同时解决单阶段方法立体效果弱和两阶段方法图像质量差的矛盾。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "stereo conversion"
  - "扩散模型"
  - "SIoU metric"
  - "Edge Consistency loss"
  - "benchmark"
---

# Mono2Stereo: A Benchmark and Empirical Study for Stereo Conversion

**会议**: CVPR 2025  
**arXiv**: [2503.22262](https://arxiv.org/abs/2503.22262)  
**代码**: [mono2stereo-bench.github.io](https://mono2stereo-bench.github.io)  
**领域**: 图像生成  
**关键词**: stereo conversion, dual-condition diffusion, SIoU metric, Edge Consistency loss, benchmark

## 一句话总结

构建首个大规模立体转换基准 Mono2Stereo（240 万对），提出立体质量指标 SIoU（与人类判断相关性 0.84 Spearman）和双条件扩散模型 + Edge Consistency 损失，同时解决单阶段方法立体效果弱和两阶段方法图像质量差的矛盾。

## 研究背景与动机

**领域现状**: 随着 AR/VR 设备普及，2D 转 3D 立体内容的需求激增。现有方法分为两类：
- **两阶段方法**（如 StereoDiffusion）: 先估计视差图扭曲左图，再用修复模型补全遮挡区域——立体效果好但图像质量差
- **单阶段方法**: 直接从左图生成右图——图像质量好但容易退化为恒等映射（输出≈输入），立体效果弱

**核心问题**: 两个关键问题未被解决：
1. 缺乏大规模公开训练数据集和标准化基准，无法系统比较方法
2. 传统指标（RMSE、SSIM）衡量全局像素差异，但左右视角差异集中在物体边缘的极小区域，传统指标无法反映立体感知质量

**关键观察**:
- 恒等映射模型（输出=输入）在 RMSE、PSNR、SSIM 上甚至优于商业立体转换模型 OWL3D，但完全没有立体效果——证明传统指标对立体质量评估失效
- 左右视图差异主要集中在物体边缘的高频细节

## 方法详解

### 整体框架

三大贡献互补：
1. **Mono2Stereo 数据集**: 240 万立体对 + 5 类测试集
2. **SIoU 评估指标**: 从视差一致性和边缘结构双重评估立体质量
3. **双条件扩散基线模型**: 同时输入几何条件（左图）和视角条件（扭曲图），配合 Edge Consistency 损失

### 关键设计 1：Mono2Stereo 数据集构建

- **数据来源**: 从互联网收集 200 部立体电影/视频（2000 万帧），人工审核去除低质和敏感内容
- **格式**: Side-by-side 3D 格式，保留最大信息
- **处理**: 每 8 帧采样 1 帧减少冗余，统一缩放至 960×540，最终 240 万对
- **测试集划分**: 5 个维度各 500 对——Indoor（视差范围）、Outdoor、Simple（场景复杂度）、Complex、Animation（色彩分布）
- **OOD 测试**: 额外使用 Inria 3DMovie 数据集（2727 对）评估域外泛化

### 关键设计 2：SIoU 评估指标

$$\text{SIoU}(\mathbf{L}, \mathbf{R}, \mathbf{G}) = \alpha \cdot \frac{\mathbf{E}_g \cap \mathbf{E}_r}{\mathbf{E}_g \cup \mathbf{E}_r} + (1-\alpha) \cdot \frac{|\mathbf{G}-\mathbf{L}| \cap |\mathbf{R}-\mathbf{L}|}{|\mathbf{G}-\mathbf{L}| \cup |\mathbf{R}-\mathbf{L}|}$$

- **第一项（权重 $\alpha=0.75$）**: 边缘结构 IoU——生成图 $\mathbf{G}$ 与右图 $\mathbf{R}$ 的 Canny 边缘图的 IoU，衡量边缘位移是否正确
- **第二项（权重 $1-\alpha=0.25$）**: 视差一致性 IoU——生成图与左图的差异图 与 右图与左图的差异图 的 IoU，衡量视差区域是否匹配
- **人类评估验证**: Spearman 相关系数 0.84，Kendall 0.73；而 RMSE 仅 0.26，SSIM 仅 0.21

### 关键设计 3：双条件扩散模型 + Edge Consistency 损失

**双条件输入**:
- **几何条件 $\mathbf{Z}_l$**: 左图的 VAE 编码，提供完整结构和纹理信息（单阶段优势）
- **视角条件 $\mathbf{Z}_w$**: 基于视差扭曲后左图的 VAE 编码，提供显式视角线索（两阶段优势）
- 两者与噪声样本 $\mathbf{Z}_r^t$ 拼接输入 U-Net，通过复制第一层卷积层适配额外通道

**Edge Consistency 损失**:

$$\ell = \mathbb{E}_{t,\mathbf{Z}_r,\epsilon}\|\epsilon - \hat{\epsilon}\|^2 + \alpha \cdot \mathbb{E}_{t,\mathbf{Z}_r,\epsilon}\|S(\epsilon) - S(\hat{\epsilon})\|^2$$

- $S$: Sobel 边缘检测算子
- **直觉**: 左右视图的主要差异在物体边缘，模型容易退化为恒等映射；对速度预测的边缘部分施加额外约束，迫使模型关注边缘位移
- **无需解码**: 利用速度表示与图像的空间相关性，在 latent 空间直接约束边缘，避免训练时通过解码器的前传开销

### 损失函数

总损失 = 标准扩散去噪损失 + $\alpha$ × 边缘一致性损失（$\alpha=1$）

## 实验关键数据

### 主实验表

Mono2Stereo 测试集上的对比（Table 3）:

| 方法 | SIoU↑ | RMSE↓ | PSNR↑ | SSIM↑ |
|---|---|---|---|---|
| Identity Mapping | 0.164 | 5.07 | 34.76 | 0.788 |
| 3D Photography | 0.210 | 8.89 | 29.30 | 0.285 |
| StereoDiffusion | 0.238 | 7.42 | 30.96 | 0.621 |
| OWL3D（商业） | 0.278 | 5.81 | 33.25 | 0.732 |
| Geometric Cond. | 0.259 | 5.40 | 33.48 | **0.829** |
| Viewpoint Cond. | 0.281 | 6.12 | 32.53 | 0.762 |
| **Dual Condition** | **0.284** | **5.34** | **34.09** | 0.795 |

- 双条件模型在 SIoU 和 PSNR 上全面最优
- Inria 3DMovie 域外测试: 双条件 SIoU 0.340 vs OWL3D 0.286 vs Geometric 0.314

### 消融表

**不同条件组合（Table 6）**:
- 几何 + 深度图/边缘图：未见提升甚至下降，因为深度图/边缘图与自然图像分布差异大
- 几何 + 视角（双条件）: SIoU 0.262 > 几何 0.253 > 视角 0.260

**EC 损失效果（Table 7）**:
- 几何条件: SIoU 0.242 → 0.253 (+4.5%)
- 视角条件: SIoU 0.257 → 0.260
- 双条件: SIoU 0.259 → 0.262

### 关键发现

1. **恒等映射陷阱**: 传统指标下恒等映射"效果最佳"，揭示了评估方法论的根本缺陷
2. 几何条件擅长图像质量（SSIM 0.829），视角条件擅长立体效果（SIoU 更高），双条件兼得
3. 深度图、边缘图等额外条件因与自然图像分布差异大，反而帮倒忙
4. EC 损失在 latent 空间操作边缘约束，避免训练时解码器开销但有效缓解退化
5. SIoU 与人类判断高度一致（0.84），可作为立体转换领域的标准评估指标

## 亮点与洞察

1. **评估指标贡献可能比模型更持久**: SIoU 填补了立体质量评估的空白，对整个领域有基础设施级别的价值
2. **恒等映射悖论的揭示**: 一个"什么都不做"的模型在传统指标上表现更好，这是对领域评估方法论的有力批判
3. **双条件设计直觉明确**: 不是简单堆叠，而是明确赋予两个条件不同的职能（结构 vs 视角），对症下药
4. **EC 损失设计巧妙**: 利用 latent 空间的空间相关性在训练时避免解码，兼顾效果和效率

## 局限性

1. 仅使用图像扩散模型（SD V2），未探索视频扩散模型（如 StereoCrafter），时序一致性未涉及
2. 数据集来源为互联网电影，场景多样性仍有提升空间（如缺少医学、遥感等专业场景）
3. 训练分辨率 640×480 偏低，4K/8K 高分辨率场景的泛化能力待验证
4. 视差估计依赖 DepthAnything，其准确度直接影响视角条件质量

## 相关工作与启发

- **StereoDiffusion**: 无训练的立体转换方法，依赖参数调整、复杂场景伪影多，本文通过大规模训练数据解决
- **Marigold**: 利用预训练扩散模型做深度估计的范式，本文借鉴其 latent space 编解码策略
- **StereoCrafter**: 同期使用视频扩散模型的方法，与本文互补（视频 vs 图像）
- **启发**: 在"左右差异极微小"的任务中，模型退化为恒等映射是一个普遍问题（也出现在暗光增强等任务），EC 损失的"边缘约束"思路可推广到其他差异微小但关键的生成任务

## 评分

⭐⭐⭐⭐ — 三位一体贡献（数据集 + 指标 + 模型）扎实完整，SIoU 指标有望成为领域标准，双条件设计和 EC 损失解决了真实矛盾。扣 1 星因为仅覆盖图像不含视频，且训练分辨率偏低。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] From Image to Video: An Empirical Study of Diffusion Representations](../../ICCV2025/3d_vision/from_image_to_video_an_empirical_study_of_diffusion_representations.md)
- [\[CVPR 2025\] DEFOM-Stereo: Depth Foundation Model Based Stereo Matching](defom-stereo_depth_foundation_model_based_stereo_matching.md)
- [\[CVPR 2025\] FoundationStereo: Zero-Shot Stereo Matching](foundationstereo_zero-shot_stereo_matching.md)
- [\[CVPR 2025\] Consistency-aware Self-Training for Iterative-based Stereo Matching](consistency-aware_self-training_for_iterative-based_stereo_matching.md)
- [\[CVPR 2025\] MVSAnywhere: Zero-Shot Multi-View Stereo](mvsanywhere_zero-shot_multi-view_stereo.md)

</div>

<!-- RELATED:END -->
