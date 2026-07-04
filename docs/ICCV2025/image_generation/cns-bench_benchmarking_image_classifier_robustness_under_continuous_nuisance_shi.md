---
title: >-
  [论文解读] CNS-Bench: Benchmarking Image Classifier Robustness Under Continuous Nuisance Shifts
description: >-
  [ICCV2025][图像生成][OOD鲁棒性] 提出 CNS-Bench，首个利用 LoRA 适配器对扩散模型施加连续：且逼真：的干扰偏移（nuisance shift）来系统评估图像分类器 OOD 鲁棒性的基准，覆盖 14 种偏移类型、5 个严重度级别和 40+ 分类器。 在真实世界部署视觉模型时…
tags:
  - "ICCV2025"
  - "图像生成"
  - "OOD鲁棒性"
  - "连续干扰偏移"
  - "LoRA适配器"
  - "扩散模型"
  - "图像分类器基准测试"
---

# CNS-Bench: Benchmarking Image Classifier Robustness Under Continuous Nuisance Shifts

**会议**: ICCV2025  
**arXiv**: [2507.17651](https://arxiv.org/abs/2507.17651)  
**代码**: [odunkel/CNS-Bench](https://github.com/odunkel/CNS-Bench)  
**领域**: 图像生成  
**关键词**: OOD鲁棒性, 连续干扰偏移, LoRA适配器, 扩散模型, 图像分类器基准测试

## 一句话总结

提出 CNS-Bench，首个利用 LoRA 适配器对扩散模型施加**连续**且**逼真**的干扰偏移（nuisance shift）来系统评估图像分类器 OOD 鲁棒性的基准，覆盖 14 种偏移类型、5 个严重度级别和 40+ 分类器。

## 研究背景与动机

在真实世界部署视觉模型时，评估其在分布外（OOD）场景下的表现至关重要。现有评估鲁棒性的方法主要有四类：

**人工采集数据**（如 OOD-CV）：费时费力，难以控制干扰因素，且不同干扰之间耦合严重

**合成腐蚀**（如 ImageNet-C）：支持连续严重度，但仅限于简单的像素级扰动，无法反映真实世界分布偏移

**渲染管线**（如 3D 资产渲染）：需要大量 3D 模型，难以扩展到大规模类别

**基于扩散模型的生成**（如 Dataset Interfaces）：能生成逼真图像，但仅支持**二值**偏移（有/无），无法捕捉连续变化

**关键 gap**：真实世界中的干扰偏移（如降雪、雾气、风格变化）本质上是**连续**的。例如自动驾驶场景中，雪从微量到完全覆盖是渐变过程；不同模型可能在不同严重度下失效。现有基准无法同时满足"逼真"、"连续"、"可扩展"三个特性。CNS-Bench 填补了这一空白。

## 方法详解

### 3.1 复制 ImageNet 分布（IN*）

直接用 Stable Diffusion 生成的图像分布 $p(X_{SD}|c)$ 与 ImageNet 分布 $p(X_{IN}|c)$ 存在显著差异，会大幅降低分类精度。为此，作者采用 **Textual Inversion** 为每个 ImageNet 类别学习专用的文本嵌入，使生成图像更贴近 ImageNet 分布。优化目标为最小化扩散模型的噪声预测误差：

$$\|\epsilon - \epsilon_\psi(\cdot, f_\psi(c))\|^2$$

将学到的分布称为 IN*：$p(X|c) = p(X_{IN^*}|c)$。实验验证 IN* 相比标准 SD 生成，FID 从 33.8 降至 27.1，ResNet-50 分类精度从 0.68 提升至 0.74。

### 3.2 基于 LoRA 的连续干扰偏移

核心思想是利用 **LoRA 适配器**学习特定干扰偏移的"方向"，通过调节缩放因子 $s$ 实现连续控制。具体地：

- 对每个 ImageNet 类别和偏移类型，训练独立的 LoRA 适配器
- LoRA 参数修改原始模型权重：$\theta^* = \theta + s \cdot \theta_{LoRA}$
- 训练目标基于 Concept Sliders 框架，使适配器捕捉从 "`<class>`" 到 "`<class> in <shift>`" 的语义方向
- 训练损失采用 MSE 目标结合 Tweedie 公式：

$$\text{MSE}(\epsilon_{\theta^*}(X, c, t); \epsilon_\theta(X, c, t) + \epsilon_\theta(X, c^+, t))$$

**关键设计**：LoRA 适配器仅在扩散过程的后 75% 噪声步激活（即前 25% 停用），以保留图像的语义结构，仅改变外观。这避免了二值文本提示方法会大幅改变图像空间结构的问题。

共考虑 **14 种偏移**，包括：

- **风格类**（8 种）：卡通、毛绒玩具、铅笔素描、绘画、雕塑、涂鸦、电子游戏、纹身
- **天气类**（6 种）：大雪、大雨、浓雾、雾霾、灰尘、沙尘暴

### 3.3 失效点（Failure Point）概念

连续偏移还引入了**失效点**分析——模型首次分类错误的最小偏移尺度：

$$s^* = \min\{S \in \mathbb{R} \mid f(X(S)) \neq c\}$$

通过统计所有样本的失效点分布，可以更精细地理解不同模型在面对不同偏移时的退化模式：有些模型逐渐退化（如天气偏移），有些则在特定尺度突然崩溃（如卡通风格在 $s=1.5$ 处）。

### 3.4 OOC 过滤机制

生成的图像可能偏离目标类别（out-of-class, OOC），需要过滤。作者提出的过滤策略包含**四个过滤器**的集成，采用"4 选 2"投票机制：

1. **CLIP 文本对齐**：计算图像与 "`A picture of a <class>`" 的余弦相似度
2. **CLIP 文本对齐（含偏移）**：计算图像与 "`A picture of a <class> in <shift>`" 的余弦相似度
3. **CLIP 图像相似度**：偏移图像与原始图像的 CLIP 特征余弦相似度
4. **DINOv2 CLS token 相似度**：偏移图像与原始图像的 DINOv2 特征余弦相似度

$$\mathcal{A}_{text} = \cos(\text{CLIP}_{img}(I_k), \text{CLIP}_{text}(p))$$
$$\mathcal{A}_{feat} = \cos(\mathcal{F}_0, \mathcal{F}_k)$$

当 4 个过滤器中有 ≥2 个触发时，该图像被过滤掉。每个过滤器的阈值设定为移除 >90% 的 OOC 样本。关键约束：所有过滤器均未在 ImageNet 数据上训练，避免引入评估偏差。

## 实验关键数据

### 分布差距与过滤效果

| 指标 | SD | IN* |
|------|-----|------|
| FID (↓) | 33.8 | **27.1** |
| ResNet-50 精度 | 0.68 | **0.74** |

| 过滤方法 | TPR | FPR (↓) | 过滤精度 (↑) |
|----------|------|---------|-------------|
| CLIP-only | 0.90 | 0.36 | 0.65 |
| **Ours** | **0.88** | **0.12** | **0.88** |

### 大规模鲁棒性评估（40+ 分类器）

基准数据集包含 **192,168 张图像**，覆盖 100 个 ImageNet 类别、14 种偏移、6 个尺度（0, 0.5, 1, 1.5, 2, 2.5）。

**架构对比**（相近参数量、相同训练数据，rCE 越低越鲁棒）：

| 模型 | rCE (↓) |
|------|---------|
| ViT | 0.926 |
| RN152 | 0.790 |
| ConvNeXt | 0.686 |
| DeiT3 | 0.610 |
| **VMamba** | **0.574** |

**模型规模**（DeiT3 系列）：

| 模型 | rCE (↓) |
|------|---------|
| DeiT3-S | 0.747 |
| DeiT3-M | 0.758 |
| DeiT3-B | 0.610 |
| DeiT3-L | **0.574** |
| DeiT3-H | 0.582 |

**预训练范式**（均使用 ViT-B/16）：

| 预训练 | rCE (↓) |
|--------|---------|
| SUP-IN1k | 0.926 |
| SUP-IN21k-1k | 0.722 |
| MAE-IN1k | 0.732 |
| MoCov3-IN1k | 0.669 |
| **DINOv1-IN1k** | **0.636** |

### 与 OOD-CV 真实数据对比

在 OOD-CV 的 10 个类别和天气偏移上训练 ResNet-50 后评估：CNS-Bench 生成的图像准确率始终高于 OOD-CV 的真实图像，说明 OOD-CV 数据中混杂了其他干扰因素（图像质量、裁剪、遮挡等），而 CNS-Bench 能更好地解耦单一干扰。

### 合成数据微调增益

用 CNS-Bench 数据微调 ResNet-50 后，ImageNet-R 精度从 27.34% 提升至 **37.57%**（+10.23%），而 ImageNet 验证集精度仅轻微下降（80.15% → 78.11%）。

## 亮点与洞察

1. **模型排名会随偏移类型和严重度变化**：例如 ViT 在低尺度绘画风格偏移下优于其他模型，但在高尺度下退化更严重。这是二值偏移基准无法捕捉的现象
2. **VMamba（视觉状态空间模型）最鲁棒**：在参数量相近的条件下，VMamba 的 rCE 优于 Transformer 和 CNN
3. **自监督预训练优于更多监督数据**：DINOv1 仅用 IN1k 数据的自监督预训练就超过了 IN21k 的监督预训练，说明学习到的表征质量比数据量更重要
4. **扩散分类器反而不鲁棒**：DiT 分类器在 snow 和 cartoon 偏移下的平均精度下降（0.106）远大于判别式模型（ViT: 0.07, MAE: 0.05）
5. **失效点分析揭示退化模式差异**：天气偏移导致的失效逐渐累积，而风格偏移（如卡通）的失效集中在特定尺度，可能与 ImageNet 类别（如"comic book"）的混淆有关
6. **用户研究验证**：最终数据集中仅 1% 为 OOC 样本，误差范围 ±0.5%

## 局限与展望

1. **CLIP 训练数据偏差**：生成偏移时无法完全消除 CLIP 和 Stable Diffusion 训练数据中固有的偏差，失效不能总是完全归因于目标干扰概念
2. **合成 vs. 真实的分布偏移**：生成图像本身与真实图像之间存在域差距，可能引入额外偏差
3. **类别覆盖有限**：目前仅评估 100 个 ImageNet 类别（共 1000 类），虽然消融实验表明精度下降趋势一致
4. **LoRA 滑块的一致性**：增加滑块权重后偏移仅在 73% 的情况下单调增加，存在不一致情况
5. **计算成本高**：训练 1400 个 LoRA 适配器约需 2000 GPU 小时，生成图像约 350 GPU 小时
6. **可扩展到更多任务**：当前仅评估分类任务，未来可扩展至分割、检测、域自适应等

## 相关工作与启发

- **Concept Sliders**（Gandikota et al., ECCV 2024）：本文 LoRA 滑块训练的基础框架
- **Dataset Interfaces**（Vendrow et al., 2023）：用扩散模型生成基准图像的先驱，但仅支持二值偏移
- **ImageNet-C**（Hendrycks & Dietterich, ICLR 2018）：经典合成腐蚀基准，本文填补了其"连续真实偏移"的空白
- **OOD-CV**（Zhao et al., ECCV 2022）：真实世界 OOD 数据集，本文与之直接对比验证了生成偏移的真实性
- **DINOv2**（Oquab et al., 2023）：用于 OOC 过滤的纯视觉自监督特征
- **启发**：连续偏移评估思路可推广到视频理解（时序连续变化）、3D 视觉（视角连续变化）等领域

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Omni IIE Bench: Benchmarking the Practical Capabilities of Image Editing Models](../../CVPR2026/image_generation/omni_iie_bench_benchmarking_the_practical_capabilities_of_image_editing_models.md)
- [\[CVPR 2026\] MICON-Bench: Benchmarking and Enhancing Multi-Image Context Image Generation in Unified Multimodal Models](../../CVPR2026/image_generation/micon-bench_benchmarking_and_enhancing_multi-image_context_image_generation_in_u.md)
- [\[NeurIPS 2025\] MultiHuman-Testbench: Benchmarking Image Generation for Multiple Humans](../../NeurIPS2025/image_generation/multihuman-testbench_benchmarking_image_generation_for_multiple_humans.md)
- [\[NeurIPS 2025\] T2SMark: Balancing Robustness and Diversity in Noise-as-Watermark for Diffusion Models](../../NeurIPS2025/image_generation/t2smark_balancing_robustness_and_diversity_in_noise-as-watermark_for_diffusion_m.md)
- [\[ICCV 2025\] Semantic Watermarking Reinvented: Enhancing Robustness and Generation Quality with Fourier Integrity](semantic_watermarking_reinvented_enhancing_robustness_and_generation_quality_wit.md)

</div>

<!-- RELATED:END -->
