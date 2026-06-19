---
title: >-
  [论文解读] Diff-ICMH: Harmonizing Machine and Human Vision in Image Compression with Generative Prior
description: >-
  [图像生成] 提出 Diff-ICMH，一种基于扩散模型的生成式图像压缩框架，通过语义一致性损失（SC loss）保持语义完整性，通过标签引导模块（TGM）激活生成先验，以单一编解码器和码流同时服务 10+ 种智能任务和人类视觉感知，无需任何任务特定适配。 图像压缩面临两种优化目标的割裂： 1. 面向人类的压缩：优化像素保真…
tags:
  - "图像生成"
---

# Diff-ICMH: Harmonizing Machine and Human Vision in Image Compression with Generative Prior

## 基本信息
- **arXiv**: 2511.22549
- **会议**: NeurIPS 2025
- **作者**: Ruoyu Feng, Yunpeng Qi, Jinming Liu, Yixin Gao, Xin Li, Xin Jin, Zhibo Chen
- **机构**: USTC, Eastern Institute of Technology (Ningbo)
- **代码**: https://github.com/RuoyuFeng/Diff-ICMH

## 一句话总结
提出 Diff-ICMH，一种基于扩散模型的生成式图像压缩框架，通过语义一致性损失（SC loss）保持语义完整性，通过标签引导模块（TGM）激活生成先验，以单一编解码器和码流同时服务 10+ 种智能任务和人类视觉感知，无需任何任务特定适配。

## 背景与动机
图像压缩面临两种优化目标的割裂：

1. **面向人类的压缩**：优化像素保真度（PSNR）或感知质量（LPIPS/FID），但对机器视觉任务支持差
2. **面向机器的压缩（ICM）**：
    - 传统编解码器方法：通过量化参数调优/比特分配适配，受限于非可微的保真度导向设计
    - 任务驱动端到端方法：特定任务性能好但跨任务泛化差
    - 特征压缩方法：直接压缩中间特征，与特定模型耦合，无法支持人类查看

**关键洞察**：语义完整性和感知真实性是机器智能和人类感知共同需要的——两者并非对立。

## 核心问题
如何设计一个通用图像编解码器，以单一码流同时高效服务多种下游智能任务和人类视觉感知？

## 方法详解

### 1. 设计哲学
传统保真度导向压缩的两大信息损失：
- **语义失真**：核心语义信息丢失→直接损害任务分析
- **感知失配**：纹理/细节偏离自然分布→域偏移导致特征提取误差累积

实验验证（图3）：在 ResNet50 深层（layer4），保真度导向编解码器（VTM, ELIC）的特征偏差远大于生成式编解码器（MS-ILLM），说明**真实纹理能有效缓解深层误差累积**。

### 2. 整体框架
- **编码侧**：输入图像 $\mathbf{x}$ 压缩为潜在特征 $\hat{\mathbf{z}}$，目标空间为 Stable Diffusion 的 VAE 潜在空间（$8\times$ 空间下采样）
- **标签提取**：Recognize Anything 提取词级别语义标签 $\mathbf{c}$
- **码流**：压缩潜变量 + 标签 ID（无损编码，约 100 bits/图）
- **解码侧**：$\hat{\mathbf{z}}$ 作为条件输入 ControlNet 式控制模块，联合冻结的 Stable Diffusion 执行生成式重建

### 3. 语义一致性损失（SC Loss）
利用预训练扩散模型的特征提取能力作为语义空间：
$$\mathcal{L}_\text{sem} = -\mathbb{E}_{\mathbf{z}, \hat{\mathbf{z}}} \left[ \frac{1}{N} \sum_{n=1}^N \text{sim}(f(\mathbf{z})_n, f(\hat{\mathbf{z}})_n) \right]$$

其中 $f(\cdot)$ 为冻结扩散模型的前向传播，$\text{sim}$ 为余弦相似度：
$$\text{sim}(\mathbf{z}, \hat{\mathbf{z}}) = \frac{\mathbf{z}^T \hat{\mathbf{z}}}{|\mathbf{z}|_2 |\hat{\mathbf{z}}|_2}$$

关键设计选择：
- 在 U-Net **中间块**（middle block）应用效果最佳——深层特征更好地捕获抽象语义
- 使用**无噪声输入**（$t=0$）——扩散模型的语义特征提取在干净信号上最优

### 4. 标签引导模块（TGM）
- 预训练标签提取器 $\mathcal{E}_t$（Recognize Anything）生成图像级标签
- 标签映射为预定义字典中的数值索引，无损编码
- 解码端转回文本字符串，作为条件输入扩散模型和控制模块
- 推理时使用 **Classifier-Free Guidance**（CFG scale = 5.0）增强语义清晰度
- 开销极低：约 **100 bits/图**

### 5. 完整损失函数
$$\mathcal{L}_\text{final} = \lambda_\text{rate} \mathcal{L}_\text{rate} + \lambda_\text{dist} \mathcal{L}_\text{dist} + \lambda_\text{diff} \mathcal{L}_\text{diff} + \lambda_\text{sem} \mathcal{L}_\text{sem}$$

各项：
- $\mathcal{L}_\text{rate}$：码率损失（量化潜变量 + 超先验的估计熵）
- $\mathcal{L}_\text{dist} = \|\mathcal{E}_\text{VAE}(\mathbf{x}) - \mathcal{D}_c(\hat{\mathbf{y}})\|_2^2$：潜在空间重建损失
- $\mathcal{L}_\text{diff} = \mathbb{E}[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, \hat{\mathbf{z}}, \mathbf{c}, t)\|_2^2]$：扩散噪声预测损失
- 权重：$\lambda_\text{dist} = \lambda_\text{diff} = 1, \lambda_\text{sem} = 2, \lambda_\text{rate} \in \{2, 4, 8, 16, 32\}$

### 6. 训练策略
- 数据集：LSDIR，随机裁切 $512\times512$
- 基础模型：Stable Diffusion 2.1（冻结）
- 两阶段训练：200K 步高比特率 → 200K 步多比特率微调
- DDIM 采样 50 步用于推理

## 实验关键数据

### 智能任务性能（10+ 任务，单一编解码器/码流）
- **COCO 目标检测/实例分割/全景分割**：全面优于 VTM、ELIC 和多数感知导向方法
- **Flickr30K 跨模态检索**：在超低比特率（0.01-0.05 bpp）优势显著
- **ADE20K 开放词汇分割**：在 0.02-0.1 bpp 大幅领先
- 所有结果无需任务特定微调

### 感知质量（Kodak / Tecnick / CLIC2020）

| 指标类型 | Diff-ICMH vs. 竞品 |
|---------|-------------------|
| PSNR/MS-SSIM（保真度） | 低于 VTM/ELIC（生成式方法的固有特点），与 DiffEIC 持平 |
| LPIPS ↓ | 优于所有方法 |
| **FID ↓** | **SOTA** |
| **DISTS ↓** | **SOTA** |

- 在超低比特率下感知优势最显著

### 消融实验（COCO 目标检测 mAP）
- SC loss + TGM 组合：约 0.025 bpp 处比 baseline 提升 ~4 mAP
- SC loss 最佳配置：$\lambda_\text{sem}=2.0$，中间块位置，无噪声输入

## 亮点
1. **一码多用**：同一编解码器 + 同一码流支持 10+ 种下游任务 + 人类视觉，无需适配
2. **语义 + 感知双保障**：SC loss 保语义完整性，生成式框架保感知真实性
3. **TGM 极低开销**：仅 ~100 bits 的标签信息即可显著激活生成先验
4. **潜在空间压缩设计**：解码到 VAE 潜在空间而非像素空间，天然过滤语义无关冗余
5. **超低比特率优势**：在 0.01-0.05 bpp 极端条件下优势最显著

## 局限性
1. **解码速度**：扩散去噪需 50 步迭代前向传播，解码耗时远高于传统编解码器
2. **保真度损失**：PSNR/MS-SSIM 低于保真度导向方法，不适合需要像素精确还原的场景
3. **VAE 瓶颈**：$8\times$ 下采样的潜在空间可能丢失精细空间细节（如姿态估计）
4. **训练成本**：需要 Stable Diffusion 的 GPU 资源进行两阶段训练

## 与相关工作的对比
- **vs. VTM/ELIC（保真度导向）**：Diff-ICMH 在 PSNR 上低但在智能任务和感知质量上大幅领先
- **vs. DiffEIC（扩散压缩）**：在智能任务上全面超越，感知质量持平或更优
- **vs. TransTIC/Adapter-ICMH（任务适配型）**：Diff-ICMH 无需任何适配即达到更好性能
- **vs. 特征压缩方法**：Diff-ICMH 在图像域工作，同时支持人类查看

## 启发与关联
- **语义保持是通用压缩的关键**：SC loss 的成功说明语义信息是机器任务和人类理解的共同基础
- **生成式压缩的新定位**：不只是追求感知质量，更是通向通用智能编解码的路径
- **CFG 在压缩中的应用**：Classifier-Free Guidance 从生成扩展到了压缩条件增强

## 评分
- 新颖性：⭐⭐⭐⭐⭐ — 首次从语义+感知统一视角设计面向机器和人类的通用编解码器
- 技术深度：⭐⭐⭐⭐⭐ — SC loss 设计有理有据，消融实验验证了每个设计选择
- 实验完整度：⭐⭐⭐⭐⭐ — 10 种任务 + 3 个感知数据集 + 完整消融
- 写作质量：⭐⭐⭐⭐☆ — 框架清晰，但引用列表过长影响可读性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] DEXTER: Diffusion-Guided EXplanations with TExtual Reasoning for Vision Models](dexter_diffusion-guided_explanations_with_textual_reasoning_for_vision_models.md)
- [\[ICCV 2025\] Rethinking the Embodied Gap in Vision-and-Language Navigation: A Holistic Study of Physical and Visual Disparities](../../ICCV2025/image_generation/rethinking_the_embodied_gap_in_vision-and-language_navigation_a_holistic_study_o.md)
- [\[NeurIPS 2025\] Denoising Weak Lensing Mass Maps with Diffusion Model and Generative Adversarial Network](denoising_weak_lensing_mass_maps_with_diffusion_model_and_generative_adversarial.md)
- [\[NeurIPS 2025\] Detecting Generated Images by Fitting Natural Image Distributions](detecting_generated_images_by_fitting_natural_image_distributions.md)
- [\[ICCV 2025\] REGEN: Learning Compact Video Embedding with (Re-)Generative Decoder](../../ICCV2025/image_generation/regen_learning_compact_video_embedding_with_re-generative_decoder.md)

</div>

<!-- RELATED:END -->
