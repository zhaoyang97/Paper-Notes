---
title: >-
  [论文解读] FerretNet: Efficient Synthetic Image Detection via Local Pixel Dependencies
description: >-
  [NeurIPS 2025][图像生成][合成图像检测] 基于马尔可夫随机场（MRF）理论，提出局部像素依赖（LPD）特征表示，通过中值滤波重建暴露生成图像的纹理不一致性，配合仅 1.1M 参数的轻量卷积网络 FerretNet，在仅用 4 类 ProGAN 数据训练的情况下，实现跨 22 个生成模型 97.1% 的平均检测准确率。
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "合成图像检测"
  - "局部像素依赖"
  - "Markov随机场"
  - "轻量网络"
  - "跨模型泛化"
---

# FerretNet: Efficient Synthetic Image Detection via Local Pixel Dependencies

**会议**: NeurIPS 2025  
**arXiv**: [2509.20890](https://arxiv.org/abs/2509.20890)  
**代码**: [https://github.com/xigua7105/FerretNet](https://github.com/xigua7105/FerretNet)  
**领域**: 图像生成  
**关键词**: 合成图像检测, 局部像素依赖, Markov随机场, 轻量网络, 跨模型泛化

## 一句话总结

基于马尔可夫随机场（MRF）理论，提出局部像素依赖（LPD）特征表示，通过中值滤波重建暴露生成图像的纹理不一致性，配合仅 1.1M 参数的轻量卷积网络 FerretNet，在仅用 4 类 ProGAN 数据训练的情况下，实现跨 22 个生成模型 97.1% 的平均检测准确率。

## 研究背景与动机

随着 VAE、GAN、LDM 等生成模型能力的提升，区分合成图像与真实图像变得越来越困难。现有检测方法面临两大挑战：

(1) **泛化能力不足**: 许多方法依赖特定模型的伪影特征。频域方法（如 F3Net、FrePGAN）在已知模型上表现好但难以泛化到未见架构；DIRE 使用扩散重建检测但在 GAN 内容上效果差。

(2) **计算效率低**: 基于大型预训练模型的方法（如 Ojha 用冻结 CLIP、FatFormer 适配 CLIP）参数量大、推理速度慢，难以在资源受限场景部署。

FerretNet 的核心洞察是：尽管不同生成模型架构各异，它们共享两个统一的伪影来源——潜变量分布偏移和解码过程平滑效应。这些伪影表现为局部像素依赖关系的破坏，可通过 MRF 框架统一建模。

## 方法详解

### 整体框架

FerretNet 流程分两步：(1) LPD 特征提取——用中值滤波重建图像，计算原始图像与重建图像的差值图；(2) 轻量分类——FerretNet 网络处理 LPD 特征图并输出真/假判断。两步合计仅 1.1M 参数。

### 关键设计

1. **基于 MRF 的局部像素依赖（LPD）特征**: 根据 MRF 假设，自然图像中每个像素的分布仅依赖其局部邻域。对于 $n \times n$ 窗口，先将中心像素置零（zero-masking，防止生成像素污染中值计算），然后计算邻域中值作为重建值。LPD 特征图即原图与中值重建图的差值：$\text{LPD} = I - I'$。自然图像因局部统计一致性强，LPD 接近零；而生成图像在纹理边缘和颜色过渡区域的 LPD 呈现显著异常模式。这一设计的物理直觉是：真实图像的像素关联源于光学物理过程（光照、材质相互作用），生成模型难以完美复现这种底层统计关系。

2. **Ferret Block 双路径架构**: 网络核心是 4 个级联的 Ferret Block，每个 block 包含双路径并行结构：主路径使用 $3\times3$ 空洞分组卷积（dilation=2），扩大感受野至有效 $5\times5$；辅路径使用标准 $3\times3$ 分组卷积，捕捉细粒度局部模式。两路输出通过 $1\times1$ 卷积融合。分组卷积+深度可分离设计大幅降低参数量，残差连接保证梯度稳定传播。整体设计哲学：在浅层网络中模拟深层网络的行为。

3. **Zero-masking 中值滤波策略**: 传统中值滤波在窗口大小为偶数时存在歧义，且中心像素参与计算会引入自相关。zero-masking 将中心像素置零后计算中值，确保 LPD 纯粹反映邻域预测能力而非自身信息。这个小设计对鲁棒性至关重要。

### 损失函数 / 训练策略

- 使用 BCEWithLogitsLoss（二值交叉熵）
- Adam 优化器，lr=$2 \times 10^{-4}$，betas=(0.937, 0.999)，weight decay=$5 \times 10^{-4}$
- 从零训练 100 epochs，batch size 32
- 训练数据仅 4 类 ProGAN 生成图像（car, cat, chair, horse），配等量 LSUN 真实图像
- 数据增强：随机裁剪至 224×224，随机水平翻转
- 测试时中心裁剪至 256×256

## 实验关键数据

### 主实验

ForenSynths 测试集（8 个 GAN 模型），仅在 4 类 ProGAN 上训练：

| 方法 | 参数量 | ProGAN | StyleGAN | StyleGAN2 | BigGAN | Mean ACC/AP |
|------|--------|--------|----------|-----------|--------|-------------|
| Wang et al. | - | 91.4/99.4 | 63.8/91.4 | 76.4/97.5 | 52.9/73.3 | 67.1/86.9 |
| Ojha (CLIP-based) | ~150M | 99.7/100 | 89.0/98.7 | 83.9/98.4 | 90.5/99.1 | 89.1/98.3 |
| FatFormer (CLIP-adapted) | ~150M | 99.9/100 | 97.2/99.8 | 98.8/99.9 | 99.5/100 | 98.4/99.7 |
| NPR | - | 99.8/100 | 96.3/99.8 | 97.3/100 | 87.5/94.5 | 92.5/96.1 |
| **FerretNet (1.1M)** | **1.1M** | 99.9/100 | 98.0/100 | 98.5/100 | 92.6/98.5 | **95.9/99.3** |

Diffusion-6-cls（6 个扩散模型变体）：

| 方法 | Mean ACC/AP |
|------|-------------|
| FatFormer | 95.0/98.8 |
| SAFE | 94.5/99.1 |
| **FerretNet** | **96.9/99.6** |

Synthetic-Pop（6 个最新高保真模型，包含 SDXL-Turbo、SD-3.5-Medium）：

| 方法 | Openjourney | RealVisXL | SD-3.5-Medium | SDXL-Turbo | Mean |
|------|------------|-----------|---------------|------------|------|
| FreqNet | 56.3/63.6 | 59.4/66.6 | 78.5/86.8 | 77.5/86.0 | 65.0/71.4 |
| NPR | 78.8/83.5 | 78.1/82.0 | 80.4/84.1 | 78.2/82.9 | 77.9/81.9 |
| FatFormer | 97.3/99.7 | 99.3/100 | 99.2/100 | 98.5/100 | 98.8/99.9 |
| **FerretNet** | 96.7/99.5 | 98.9/100 | 98.0/99.9 | 97.9/100 | **97.1/99.6** |

### 消融实验

吞吐量对比（RTX 4090，batch=128，Synthetic-Aesthetic 测试集）：

| 方法 | 参数量 | 吞吐量 (img/s) | Mean ACC |
|------|--------|----------------|----------|
| Ojha (CLIP) | ~150M | 较低 | 82.5 |
| FatFormer | ~150M | 较低 | 93.1 |
| **FerretNet** | **1.1M** | **高** | **91.5** |

FerretNet 参数量仅为 CLIP-based 方法的约 1/136，在多数基准上性能接近或超过。

### 关键发现

- LPD 特征具有优秀的跨模型泛化性——仅在 4 类 ProGAN 上训练，即可检测 VAE、GAN、LDM 等 22 种架构
- 在最新高保真扩散模型（SD 3.5、SDXL-Turbo、RealVisXL）上仍保持 >97% 准确率
- FerretNet 的 1.1M 参数相比 FatFormer 等百万级参数的 CLIP-based 方法，推理效率显著更高
- LPD 可视化图直观展示了自然图像和合成图像的差异：真实图纹理均匀一致，合成图在细节区域有清晰的结构性残差

## 亮点与洞察

- **统一理论视角**: 从 MRF 理论出发，揭示所有生成模型的共性弱点——无法完美复现局部像素依赖关系，这是一个优雅且可解释的检测原理
- **极端轻量化**: 1.1M 参数实现与百倍参数量的 CLIP-based 方法可比的性能，适合边缘部署
- 提出 Synthetic-Pop 新基准（6 个最新生成模型，6 万张图像），填补了对最新高保真模型评估的空白
- 中值滤波+差值这一"老技术"在新任务上焕发生机——提示我们不应忽视传统信号处理方法在深度学习时代的价值

## 局限与展望

- 在 BigGAN 类别上性能相对较弱（92.6% vs FatFormer 99.5%），可能因 BigGAN 的潜空间结构更规则
- CO-SPY 使用了不同训练数据导致比较不完全公平
- 仅验证了图像级检测，未扩展到视频或局部篡改检测
- LPD 的窗口大小 $n$ 是固定超参，自适应窗口可能进一步提升性能
- 对高质量后处理（如 JPEG 压缩、社交媒体分享后的降质）的鲁棒性未充分评估

## 相关工作与启发

- **NPR（Neighbor Pixel Relations）**: 思路相近（邻域像素关系），但 NPR 关注上采样模式，FerretNet 基于 MRF 的中值偏差更具理论基础
- **DIRE**: 用扩散模型重建差检测合成图像，但计算代价极高且对 GAN 效果差；FerretNet 的 LPD 提取几乎无计算开销
- **频域方法（F3Net、BiHPF、FreqNet）**: 捕捉特定频率特征但跨模型泛化有限
- 启发：检测问题的关键不在于更大的模型，而在于更好的特征表示——LPD 证明了对物理规律的建模比暴力拟合更有价值

## 评分

⭐⭐⭐⭐ — 理论清晰（MRF→LPD），极致轻量（1.1M），跨 22 个模型泛化出色。在实际部署价值上非常突出。BigGAN 上的性能差距和后处理鲁棒性有待验证是主要不足。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] PixPerfect: Seamless Latent Diffusion Local Editing with Discriminative Pixel-Space Refinement](pixperfect_seamless_latent_diffusion_local_editing_with_discriminative_pixel-spa.md)
- [\[NeurIPS 2025\] Epistemic Uncertainty for Generated Image Detection](epistemic_uncertainty_for_generated_image_detection.md)
- [\[AAAI 2026\] Beyond Semantic Features: Pixel-Level Mapping for Generalized AI-Generated Image Detection](../../AAAI2026/image_generation/beyond_semantic_features_pixel-level_mapping_for_generalized_ai-generated_image_.md)
- [\[ICCV 2025\] DeepShield: Fortifying Deepfake Video Detection with Local and Global Forgery Analysis](../../ICCV2025/image_generation/deepshield_fortifying_deepfake_video_detection_with_local_and_global_forgery_ana.md)
- [\[CVPR 2026\] Layer Consistency Matters: Elegant Latent Transition Discrepancy for Generalizable Synthetic Image Detection](../../CVPR2026/image_generation/layer_consistency_matters_elegant_latent_transition_discrepancy_for_generalizabl.md)

</div>

<!-- RELATED:END -->
