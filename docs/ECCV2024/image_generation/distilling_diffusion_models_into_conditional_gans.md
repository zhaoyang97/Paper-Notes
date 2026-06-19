---
title: >-
  [论文解读] Distilling Diffusion Models into Conditional GANs
description: >-
  [ECCV 2024][图像生成][扩散蒸馏] 提出 Diffusion2GAN 框架，将多步扩散模型蒸馏为单步条件GAN，核心创新是 E-LatentLPIPS 潜空间感知损失和基于预训练扩散模型的多尺度条件判别器，在零样本 COCO 基准上超越 DMD、SDXL-Turbo 和 SDXL-Lightning。
tags:
  - "ECCV 2024"
  - "图像生成"
  - "扩散蒸馏"
  - "GAN"
  - "感知损失"
  - "潜空间"
  - "单步生成"
---

# Distilling Diffusion Models into Conditional GANs

**会议**: ECCV 2024  
**arXiv**: [2405.05967](https://arxiv.org/abs/2405.05967)  
**代码**: 有（项目页面提供）  
**领域**: 扩散模型 / 图像生成 / 知识蒸馏  
**关键词**: 扩散蒸馏, 条件GAN, 感知损失, 潜空间, 单步生成

## 一句话总结

提出 Diffusion2GAN 框架，将多步扩散模型蒸馏为单步条件GAN，核心创新是 E-LatentLPIPS 潜空间感知损失和基于预训练扩散模型的多尺度条件判别器，在零样本 COCO 基准上超越 DMD、SDXL-Turbo 和 SDXL-Lightning。

## 研究背景与动机

扩散模型（如 Stable Diffusion、DALL·E 2）在图像生成质量上取得了前所未有的成果，但其采样过程需要数十甚至上百次去噪步骤，导致推理延迟高（通常超过10秒），严重阻碍了实时交互和下游 3D/视频应用的扩展。

**为什么不直接训练单步模型？** 直接训练 GAN 做文本到图像生成面临两个同时需要解决的难题：(1) 在噪声和自然图像之间建立对应关系；(2) 有效优化生成器完成从噪声到图像的映射。这种"无配对"学习比有配对学习更加病态。

**核心洞察**：将上述两个任务解耦——先用预训练扩散模型的 ODE 轨迹建立噪声-图像对应关系，再用条件 GAN 在配对图像翻译框架下学习映射。这让我们同时利用扩散模型寻找高质量对应关系和 GAN 实现快速映射的优势。

**另一个关键发现**：现有蒸馏方法的回归损失设计不够充分。作者发现，只要精心设计回归损失（使用感知损失而非 L2），直接蒸馏就能达到与近期蒸馏方法（如一致性蒸馏）相当的结果，且计算成本更低。然而，标准 LPIPS 需要将潜空间解码到像素空间，与潜空间扩散模型的高效性理念矛盾，这促使了 E-LatentLPIPS 的提出。

## 方法详解

### 整体框架

Diffusion2GAN 分为两个阶段：
1. **数据集构建**：用预训练扩散模型（如 SD 1.5）的 DDIM 采样器（50步）生成大量噪声-潜码对 $\{(\mathbf{z}, \mathbf{c}, \mathbf{x})\}$
2. **配对蒸馏训练**：将噪声到图像的映射视为配对图像翻译问题，用 E-LatentLPIPS 回归损失 + 条件 GAN 对抗损失训练单步生成器

生成器 $G$ 与教师扩散模型共享架构（UNet），用教师权重初始化。最终损失：

$$\mathcal{L}_G = \mathcal{L}_{\text{E-LatentLPIPS}} + \lambda_{\text{GAN}} \mathcal{L}_{\text{GAN}}$$

### 关键设计

#### 1. E-LatentLPIPS（核心贡献）

**功能**：在潜空间直接计算感知距离，替代传统 LPIPS 需要解码到像素空间的低效操作。

**核心思路**：
- 首先训练 LatentLPIPS：在 SD 潜空间上训练 VGG 网络做 ImageNet 分类，去掉最大池化层（因潜空间已8×下采样），修改输入为4通道，然后用 BAPPS 数据集线性校准中间特征
- 直接使用 LatentLPIPS 会产生波浪状伪影（损失landscape有盲点），因此引入**集成增强策略**：每次迭代对生成和目标潜码施加随机可微增强（几何变换 + cutout）

$$d_{\text{E-LatentLPIPS}}(\mathbf{x}_0, \mathbf{x}_1) = \mathbb{E}_{\mathcal{T}}\left[\ell\left(F(\mathcal{T}(\mathbf{x}_0)), F(\mathcal{T}(\mathbf{x}_1))\right)\right]$$

**设计动机**：
- 传统 LPIPS 需解码到像素空间，每次迭代额外耗时 117ms 和 15GB 显存
- E-LatentLPIPS 仅需 12.1ms 和 0.6GB，感知损失计算加速 **9.7倍**
- 集成增强是关键——单纯的 LatentLPIPS 无法收敛（单图重建实验验证），但加入增强后可精确重建目标图像

#### 2. 条件扩散判别器

**功能**：以噪声 $\mathbf{z}$、文本 $\mathbf{c}$ 和图像 $\mathbf{x}$ 为条件的 GAN 判别器，用于提升生成质量。

**核心思路**：复用预训练扩散模型 UNet 权重初始化判别器，进行以下改进：
- **噪声条件**：在输入端添加零初始化卷积层处理 $\mathbf{z}$，与判别器输入相加
- **文本条件**：直接利用 UNet 内置的交叉注意力层
- **多尺度输入输出**：修改 UNet 编码器接收各下采样层的缩放输入，解码器各尺度附加三个读出层做独立真/假预测
- **单样本 R1 正则化**：每批仅对一个样本计算 R1 正则化，配合间隔 16 的惰性正则化，减少显存消耗
- **Mix-and-match 增强**：训练判别器时，随机替换部分生成潜码为不相关潜码，同时保持其他条件不变，增强文本对齐和噪声条件化能力

**设计动机**：预训练扩散模型的 UNet 包含丰富的图像先验，比从头训练 GigaGAN 判别器更有效。多尺度设计确保所有 UNet 层（从浅层 skip connection 到深层 bottleneck）都参与预测，增强低频结构一致性。

### 损失函数 / 训练策略

- **回归损失**：$\mathcal{L}_{\text{E-LatentLPIPS}}$ — 带集成增强的潜空间感知损失
- **对抗损失**：$\mathcal{L}_{\text{GAN}} = -\mathbb{E}_{\mathbf{c},\mathbf{z}}[\log(D(\mathbf{c},\mathbf{z},G(\mathbf{z},\mathbf{c})))]$ — 非饱和 GAN 损失
- **判别器正则化**：单样本 R1 正则化（间隔16）
- **训练配置**：SD-CFG-3 数据集 300 万对，SD-CFG-8 数据集 1200 万对；64 块 A100-80GB GPU，batch size 1024
- **重要说明**：整个训练过程完全在潜空间进行，**从不**需要解码到像素空间

## 实验关键数据

### 主实验

**COCO2014 零样本基准（蒸馏 SD 1.5）**：

| 方法 | 类型 | FID-30k↓ | 推理时间(s) |
|------|------|----------|------------|
| Stable Diffusion 1.5 (Teacher) | 扩散 | 8.74 | 2.59 |
| GigaGAN | GAN | 9.09 | 0.13 |
| **Diffusion2GAN** | **蒸馏** | **9.29** | **0.09** |
| DMD | 蒸馏 | 11.49 | 0.09 |
| UFOGen | 蒸馏 | 12.78 | 0.09 |
| InstaFlow-0.9B | 蒸馏 | 13.10 | 0.09 |

**COCO2017 基准（蒸馏 SDXL）**：

| 方法 | FID-5k↓ | CLIP-5k↑ | DreamDiv↑ |
|------|---------|----------|-----------|
| SDXL-Base-1.0 (Teacher, 50步) | 25.56 | 0.346 | 0.338 |
| **SDXL-Diffusion2GAN (1步)** | **25.49** | **0.347** | **0.268** |
| SDXL-Turbo (1步) | 28.10 | 0.342 | 0.232 |
| SDXL-Lightning (1步) | 30.14 | 0.324 | 0.315 |

### 消融实验

**回归损失选择（SD-CFG-3, 20k iter, batch=256）**：

| 损失函数 | 空间 | FID↓ | CLIP↑ |
|---------|------|------|-------|
| MSE | 潜空间 | 110.55 | 0.222 |
| Pseudo Huber | 潜空间 | 87.60 | 0.230 |
| LPIPS | 像素空间 | 25.94 | 0.288 |
| LatentLPIPS | 潜空间 | 67.17 | 0.244 |
| **E-LatentLPIPS** | **潜空间** | **22.95** | **0.299** |

**判别器组件消融**：

| 配置 | FID-30k↓ | CLIP-30k↑ |
|------|----------|-----------|
| E-LatentLPIPS (仅回归) | 14.72 | 0.292 |
| + Diffusion D | 12.04 | 0.300 |
| + z条件 | 11.97 | 0.302 |
| + 单样本R1 | 10.60 | 0.303 |
| + 多尺度训练 | 9.58 | 0.308 |
| + Mix-and-match增强 | 9.45 | 0.310 |

### 关键发现

1. **E-LatentLPIPS 是关键**：直接在潜空间计算感知损失 + 集成增强，FID 从 67.17 降至 22.95，同时计算效率提升 9.7 倍
2. **单步生成器可匹敌教师**：SDXL-Diffusion2GAN 的 FID (25.49) 与 50步教师 SDXL (25.56) 几乎相同
3. **ODE 轨迹忠实度**：Diffusion2GAN 比 SDXL-Turbo 和 SDXL-Lightning 更好地保持了教师模型的噪声-图像映射（DreamSim-5k 最低）
4. 人类偏好评估显示 Diffusion2GAN 在真实感和文本对齐方面优于 InstaFlow，与 SDXL-Turbo/Lightning 可比或更优

## 亮点与洞察

- **将蒸馏重新定义为配对翻译**：这个视角使得成熟的 pix2pix 工具箱可以直接应用于扩散蒸馏
- **潜空间感知损失的可行性**：证明了压缩到潜空间虽然损失了部分低级分类信息，但保留了感知相关细节
- **集成增强的必要性**：在像素空间是锦上添花，在潜空间是生死攸关——没有增强的 LatentLPIPS 根本无法收敛
- **预训练权重的多重利用**：教师扩散模型的权重同时用于初始化生成器和判别器

## 局限与展望

- 虽然单步推理极快（0.09秒/图），但**训练成本高昂**：需要预生成数百万 ODE 对 + 64块 A100 训练
- 多步教师模型在人类偏好评估中仍然被偏好，说明单步蒸馏的上限还有提升空间
- DreamDiv 指标需要与 CLIP-score 联合报告，否则可能被低文本对齐导致的虚假多样性误导
- 未探索更新的扩散架构（如 DiT）的蒸馏

## 相关工作与启发

- 与 CycleGAN 的"无配对 vs 有配对"洞察一致，本文证明有配对翻译范式在蒸馏中同样远优于无配对范式
- 受 E-LPIPS 启发的集成增强策略，在潜空间发挥了远超预期的作用
- DreamSim-5k 和 DreamDiv 两个新指标值得未来工作采用，分别衡量 ODE 轨迹忠实度和生成多样性

## 评分

- **新颖性**: ⭐⭐⭐⭐ — E-LatentLPIPS 是一个简洁有效的创新，将蒸馏视为配对翻译的视角新颖
- **实验充分度**: ⭐⭐⭐⭐⭐ — SD 1.5 和 SDXL 双重验证，详尽的消融，人类偏好评估，新指标设计
- **写作质量**: ⭐⭐⭐⭐ — 逻辑清晰，每个设计决策都有实验支持
- **价值**: ⭐⭐⭐⭐ — 实用价值高，单步高质量文本到图像生成对实际应用意义重大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Scalable GANs with Transformers](../../ICML2026/image_generation/scalable_gans_with_transformers.md)
- [\[ECCV 2024\] MacDiff: Unified Skeleton Modeling with Masked Conditional Diffusion](macdiff_unified_skeleton_modeling_with_masked_conditional_diffusion.md)
- [\[ICLR 2026\] Condition Matters in Full-head 3D GANs](../../ICLR2026/image_generation/condition_matters_in_full-head_3d_gans.md)
- [\[ECCV 2024\] Realistic Human Motion Generation with Cross-Diffusion Models](realistic_human_motion_generation_with_cross-diffusion_models.md)
- [\[ECCV 2024\] M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models](m2d2m_multi-motion_generation_from_text_with_discrete_diffusion_models.md)

</div>

<!-- RELATED:END -->
