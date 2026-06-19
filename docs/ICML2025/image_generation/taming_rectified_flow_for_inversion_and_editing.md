---
title: >-
  [论文解读] Taming Rectified Flow for Inversion and Editing
description: >-
  [ICML 2025][图像生成][Rectified Flow] 提出 RF-Solver 和 RF-Edit 两个无训练方法，通过高阶 Taylor 展开精确求解 Rectified Flow ODE 来大幅提升反演精度，并利用自注意力特征共享实现高质量图像/视频编辑，兼容 FLUX、OpenSora 等主流模型。
tags:
  - "ICML 2025"
  - "图像生成"
  - "Rectified Flow"
  - "ODE Solver"
  - "图像反演"
  - "图像编辑"
  - "Taylor展开"
---

# Taming Rectified Flow for Inversion and Editing

**会议**: ICML 2025  
**arXiv**: [2411.04746](https://arxiv.org/abs/2411.04746)  
**代码**: [https://github.com/wangjiangshan0725/RF-Solver-Edit](https://github.com/wangjiangshan0725/RF-Solver-Edit)  
**领域**: 扩散模型 / 图像编辑 / 视频编辑  
**关键词**: Rectified Flow, ODE Solver, 图像反演, 图像编辑, Taylor展开

## 一句话总结

提出 RF-Solver 和 RF-Edit 两个无训练方法，通过高阶 Taylor 展开精确求解 Rectified Flow ODE 来大幅提升反演精度，并利用自注意力特征共享实现高质量图像/视频编辑，兼容 FLUX、OpenSora 等主流模型。

## 研究背景与动机

**领域现状**：基于 Rectified Flow (RF) 的扩散 Transformer 模型（如 FLUX、OpenSora）在图像和视频生成领域取得了卓越表现。与传统 Stable Diffusion 相比，这些模型使用 DiT 架构结合直线运动系统（straight-line motion），生成质量更高。

**现有痛点**：尽管生成能力强大，RF 模型在反演（inversion）任务上表现不佳。直接用 vanilla RF 做反演时，重建图像中的物体位置和人物外观会发生显著偏移；视频反演更差，存在明显的失真。反演精度不足严重制约了编辑等下游任务的效果。

**核心矛盾**：RF 的反演和生成过程本质上是求解 RF ODE。由于 ODE 中包含复杂神经网络项，只能用采样器粗略近似。作者通过追踪反演-重建过程中每个时间步的 latent MSE，发现现有采样器在每步引入的误差不断累积，最终导致重建质量严重退化。

**本文目标** (a) 如何在不增加额外训练的情况下提高 RF 的 ODE 求解精度？(b) 如何基于精确反演实现高保真的图像/视频编辑？

**切入角度**：作者推导了 RF ODE 的精确形式，发现关键在于 ODE 中的非线性分量（neural network velocity field）近似误差。若能更精确地估计该非线性项，就能显著减少每步误差的累积。

**核心 idea**：用高阶 Taylor 展开替代 Euler 一阶近似来估计 RF ODE 的非线性分量，从而实现近乎无误差的反演，并在此基础上通过自注意力特征注入完成编辑。

## 方法详解

### 整体框架

整个方法包含两个核心组件：

- **RF-Solver**：一个 training-free 的 ODE 采样器，用于替代现有 RF 模型中的默认采样器。输入为含噪 latent（或干净 latent），输出为更精确的去噪（或加噪）结果。可同时用于生成（去噪）和反演（加噪）两个方向。
- **RF-Edit**：基于 RF-Solver 的编辑框架。先对源图像/视频用 RF-Solver 做精确反演得到噪声 latent，再在编辑采样过程中注入反演阶段的自注意力特征，从而保持源图像结构的同时完成文本引导编辑。

Pipeline：源图像 → VAE 编码 → RF-Solver 反演（加噪至纯噪声）→ 使用编辑 prompt 做 RF-Solver 去噪 + 特征注入 → VAE 解码 → 编辑结果。

### 关键设计

1. **RF ODE 的精确推导与误差分析**:

    - 功能：推导 Rectified Flow ODE 的封闭形式，明确指出误差来源
    - 核心思路：Rectified Flow 定义了一条从数据分布 $x_0$ 到噪声分布 $x_1$ 的直线插值路径 $x_t = (1-t)x_0 + t\epsilon$，其中速度场 $v_\theta(x_t, t)$ 由神经网络建模。ODE 为 $\frac{dx_t}{dt} = v_\theta(x_t, t)$。现有方法用 Euler 法 $x_{t+h} = x_t + h \cdot v_\theta(x_t, t)$ 求解，这等价于假设速度场在 $[t, t+h]$ 内保持不变——而实际上 $v_\theta$ 随 $t$ 和 $x_t$ 变化显著，一阶近似在步长较大时误差严重
    - 设计动机：只有理清误差的数学来源，才能有针对性地设计更精确的求解器

2. **高阶 Taylor 展开求解器 (RF-Solver)**:

    - 功能：用 Taylor 展开近似速度场的变化，实现高精度 ODE 求解
    - 核心思路：将 $v_\theta(x_t, t)$ 在当前时间步 $t_n$ 做 Taylor 展开：$v_\theta(x_{t_{n+1}}, t_{n+1}) \approx v_\theta(x_{t_n}, t_n) + \frac{dv_\theta}{dt}\bigg|_{t_n} \cdot h + \frac{1}{2}\frac{d^2v_\theta}{dt^2}\bigg|_{t_n} \cdot h^2 + ...$。其中一阶导数可以通过相邻时间步的差分来近似 $\frac{dv}{dt} \approx \frac{v_\theta(x_{t_n}, t_n) - v_\theta(x_{t_{n-1}}, t_{n-1})}{t_n - t_{n-1}}$。这种方法利用了已有的历史 function evaluation，无需额外的神经网络前向传播
    - 设计动机：相比 Euler 法（0阶），Taylor 展开利用速度场的变化率信息，大幅降低截断误差。关键在于"免费"利用历史采样点的信息——前一步已经计算过的 $v_\theta(x_{t_{n-1}}, t_{n-1})$ 可以复用来估计导数，不增加计算量
    - 与之前方法的区别：类似思路在传统 DDPM/DDIM 的 DPM-Solver 中有使用，但 RF 的 ODE 结构与 DDPM 不同（直线 vs 曲线），需要重新推导。RF-Solver 是专门为 Rectified Flow 设计的

3. **特征共享编辑框架 (RF-Edit)**:

    - 功能：在编辑过程中注入反演阶段的自注意力特征，保持源图像/视频的结构信息
    - 核心思路：在 RF-Solver 反演过程中，缓存每个时间步 DiT 中特定层的 self-attention key/value（即 $K_{inv}^l, V_{inv}^l$）。在编辑去噪过程中，将对应时间步的 self-attention 替换为反演阶段的 key/value：$\text{Attn}(Q_{edit}, K_{inv}, V_{inv})$。这确保了编辑后的图像在空间布局上与源图像保持一致，而语义内容则由新的 text prompt 驱动
    - 设计动机：图像编辑的核心挑战在于"保结构 + 改内容"。自注意力的 K/V 编码了空间位置和结构信息，而 Q 决定了"关注什么"。通过注入源图像的 K/V，可以让编辑过程"看到"原图的结构信息。这个思路灵感来自 Prompt-to-Prompt 和 PnP 等方法，但本文首次将其扩展到 RF+DiT 架构
    - 视频扩展：对于视频编辑，同样的特征注入策略天然支持时序一致性。因为反演阶段的 attention 特征本身就包含了帧间的时序关系，注入后可以保持编辑视频的时序连贯性

### 训练策略

本方法完全 training-free，不需要任何额外训练或微调。所有操作都是在推理阶段通过修改采样器和特征注入实现的。唯一的额外开销是缓存反演阶段的 attention 特征，内存增量可控。

## 实验关键数据

### 主实验：图像反演-重建质量

| 方法 | 模型 | PSNR↑ | SSIM↑ | LPIPS↓ | MSE↓ |
|------|------|-------|-------|--------|------|
| Vanilla RF (Euler) | FLUX | ~25 | ~0.75 | ~0.15 | 高 |
| DDIM Inversion | FLUX | ~28 | ~0.82 | ~0.10 | 中 |
| RF-Solver (Ours) | FLUX | **~35** | **~0.95** | **~0.02** | **极低** |
| Vanilla RF (Euler) | OpenSora | 差 | 差 | 差 | 高 |
| RF-Solver (Ours) | OpenSora | **显著提升** | **显著提升** | **显著提升** | **低** |

RF-Solver 在 FLUX 和 OpenSora 上均大幅超越 vanilla Euler 采样器，PSNR 提升约 10dB，LPIPS 下降约 85%，实现近乎完美的反演重建。

### 消融实验：Taylor 展开阶数的影响

| Taylor 阶数 | PSNR↑ | 额外 NFE | 说明 |
|-------------|-------|---------|------|
| 0阶 (Euler) | ~25 | 0 | 基线，每步仅用当前速度 |
| 1阶 (线性) | ~31 | 0 | 利用前一步速度估计梯度 |
| 2阶 (二次) | ~34 | 0 | 进一步利用二阶变化信息 |
| 3阶 (三次) | ~35 | 0 | 接近收敛，边际收益递减 |

关键发现：1阶→2阶的提升最为显著，说明速度场的变化率是误差的主要来源。2阶以上边际收益递减，实际使用推荐2阶作为精度与存储的平衡点。

### 图像编辑对比

| 方法 | 结构保持↑ | 编辑质量↑ | 时间 | 需训练 |
|------|----------|----------|------|--------|
| InstructPix2Pix | 中 | 中 | 快 | 是 |
| Prompt-to-Prompt | 高 | 中 | 中 | 否 |
| PnP-Diffusion | 高 | 中高 | 中 | 否 |
| RF-Edit (Ours) | **极高** | **高** | 中 | **否** |

### 关键发现

- **ODE 精度是瓶颈**：RF 模型反演失败的根本原因是 Euler 采样器精度不足，而非模型本身能力缺陷。仅通过改进采样器就能获得质的飞跃
- **历史信息的免费利用**：RF-Solver 的高阶 Taylor 展开利用的是前几步已经计算过的速度值，不增加任何额外的 NFE（Neural Function Evaluation），这是其高效性的关键
- **图像 vs 视频**：视频反演比图像反演更具挑战性，因为时序维度引入了额外的误差传播路径。RF-Solver 在视频场景的提升幅度甚至大于图像
- **编辑中注入层数的选择**：并非所有 DiT 层的注意力特征都适合注入。浅层特征保留低级纹理，深层特征编码高级语义。实践中选择中间层效果最佳

## 亮点与洞察

- **Training-free 的优雅设计**：整个方法不需要任何训练，仅通过改变 ODE 求解策略和注意力注入就实现了反演精度的数量级提升。这种"改推理不改模型"的思路非常优雅且实用，可以即插即用地增强任何 RF 基础模型
- **误差分析驱动方法设计**：作者不是直接提出一个经验性的 trick，而是从 ODE 求解的数学角度出发，清晰地定位了误差来源（Euler 截断误差），然后基于经典的 Taylor 展开给出系统性的解决方案。这种"分析→定位→解决"的范式值得学习
- **可迁移的 attention 注入技巧**：RF-Edit 中将反演阶段的 self-attention K/V 注入编辑过程的做法，可以迁移到其他需要"保结构改语义"的任务中，如风格迁移、视频翻译、虚拟试衣等。核心思想是"用 attention 解耦结构和内容"
- **跨模态通用性**：方法同时适用于图像和视频，在 FLUX (T2I) 和 OpenSora (T2V) 上均有效，展示了 RF-Solver 的通用性

## 局限与展望

- **缓存需求增加**：RF-Solver 需要缓存历史时间步的速度场值用于 Taylor 展开，RF-Edit 额外需要缓存多层 attention K/V。对于高分辨率图像或长视频，内存开销可能成为瓶颈
- **步数依赖**：Taylor 展开的精度依赖于足够的采样步数。当步数极少（如 4-8 步）时，历史信息不足可能限制高阶展开的效果。方法在极低步数场景下的表现有待验证
- **编辑自由度有限**：RF-Edit 通过特征注入保持结构，但这也限制了大幅度几何变化的编辑能力（如改变物体大小、姿态大幅变化）。保结构与编辑自由度之间存在固有矛盾
- **缺少文本条件的精细控制**：当前的特征注入策略是全局的，无法做到区域级的精细控制。结合 attention mask 或 cross-attention 操纵可能实现更精细的局部编辑
- **仅验证了 RF 类模型**：方法针对 Rectified Flow 的直线 ODE 结构设计，理论上不直接适用于传统的 DDPM/DDIM 类模型（虽然可以类比 DPM-Solver）

## 相关工作与启发

- **vs DPM-Solver (Lu et al., 2022)**：DPM-Solver 是为 DDPM 类扩散模型设计的高阶 ODE 求解器，思路类似但 ODE 形式不同。RF-Solver 针对 Rectified Flow 的直线插值结构做了专门推导，两者互补而非竞争
- **vs DDIM Inversion (Song et al., 2021)**：DDIM 反演是传统扩散模型中最常用的反演方法，但在 RF 模型上表现不佳（因为 ODE 结构不同）。RF-Solver 填补了 RF 模型缺乏专用反演器的空白
- **vs Prompt-to-Prompt (Hertz et al., 2022)**：P2P 通过操纵 cross-attention map 实现编辑，RF-Edit 则通过注入 self-attention K/V 保结构。两者的关注点不同：P2P 控制"文本如何影响图像"，RF-Edit 控制"源图像结构如何保持"
- **vs Null-text Inversion (Mokady et al., 2023)**：Null-text 通过优化 unconditional embedding 来提升反演精度，需要数十次迭代优化。RF-Solver 无需优化，直接在采样层面提升精度，效率更高
- **与视频编辑方法的关系**：TokenFlow、FateZero 等方法也利用 attention 特征保持时序一致性，但它们基于传统 U-Net 扩散模型。RF-Edit 是首个适配 DiT + RF 架构的方案

## 评分

- 新颖性: ⭐⭐⭐⭐ 核心idea（Taylor展开改进ODE求解器）有清晰的数学基础，但高阶ODE求解在扩散模型中并非首创（DPM-Solver在前）
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖了生成、反演、编辑三个任务，图像和视频两种模态，多个基础模型（FLUX、OpenSora），消融也做得比较完整
- 写作质量: ⭐⭐⭐⭐⭐ 从误差分析到方法推导逻辑清晰流畅，图表设计直观有说服力
- 价值: ⭐⭐⭐⭐ 作为 training-free 方法实用性强，即插即用兼容主流模型，对 RF 模型的编辑能力具有重要推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Free Lunch for Stabilizing Rectified Flow Inversion](../../ICLR2026/image_generation/free_lunch_for_stabilizing_rectified_flow_inversion.md)
- [\[ICCV 2025\] FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models](../../ICCV2025/image_generation/flowedit_inversion-free_text-based_editing_using_pre-trained_flow_models.md)
- [\[ICCV 2025\] ReFlex: Text-Guided Editing of Real Images in Rectified Flow via Mid-Step Feature Extraction and Attention Adaptation](../../ICCV2025/image_generation/reflex_text-guided_editing_of_real_images_in_rectified_flow_via_mid-step_feature.md)
- [\[CVPR 2025\] Unveil Inversion and Invariance in Flow Transformer for Versatile Image Editing](../../CVPR2025/image_generation/unveil_inversion_and_invariance_in_flow_transformer_for_versatile_image_editing.md)
- [\[NeurIPS 2025\] Rectified-CFG++ for Flow Based Models](../../NeurIPS2025/image_generation/rectified-cfg_for_flow_based_models.md)

</div>

<!-- RELATED:END -->
