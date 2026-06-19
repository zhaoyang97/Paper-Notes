---
title: >-
  [论文解读] Toward Tiny and High-quality Facial Makeup with Data Amplify Learning
description: >-
  [ECCV 2024][图像生成][人脸妆容迁移] 提出 Data Amplify Learning (DAL) 学习范式，用 Diffusion-based Data Amplifier 从仅 5 张标注图像"放大"生成大量配对训练数据，训练出仅 80K 参数的 TinyBeauty 模型，在 iPhone 13 上以 460fps 实现 SOTA 妆容迁移效果。
tags:
  - "ECCV 2024"
  - "图像生成"
  - "人脸妆容迁移"
  - "数据放大学习"
  - "扩散模型"
  - "轻量化模型"
  - "移动端部署"
---

# Toward Tiny and High-quality Facial Makeup with Data Amplify Learning

**会议**: ECCV 2024  
**arXiv**: [2403.15033](https://arxiv.org/abs/2403.15033)  
**代码**: [有](https://github.com/TinyBeauty)  
**领域**: 人体理解  
**关键词**: 人脸妆容迁移, 数据放大学习, 扩散模型, 轻量化模型, 移动端部署

## 一句话总结

提出 Data Amplify Learning (DAL) 学习范式，用 Diffusion-based Data Amplifier 从仅 5 张标注图像"放大"生成大量配对训练数据，训练出仅 80K 参数的 TinyBeauty 模型，在 iPhone 13 上以 460fps 实现 SOTA 妆容迁移效果。

## 研究背景与动机

当前主流妆容迁移方法存在根本性缺陷：

**模型过大**：现有方法（如 EleGANt 10M、SCGAN 35M）参数远超移动端要求（<100K），难以在手机上实时运行

**依赖复杂 pipeline**：需要人脸解析、关键点检测等辅助模块，增加延迟和部署难度

**非配对学习不稳定**：标注配对妆容数据代价昂贵，主流方法采用非配对学习范式，依赖对抗训练和不精确的监督信号（颜色直方图匹配、推土机距离等），鲁棒性差

根本原因是**学习框架本身的缺陷**：非配对数据存在严重的面部对齐问题，迫使模型引入面部空间 prompt 和 warping 操作，进一步增加复杂度。而不精确的监督方法依赖理论近似，模型越小越敏感。

核心 insight：如果能获得高质量配对数据，就可以用简单的 L1 损失替代复杂的非配对学习，大幅降低优化难度和模型复杂度。

## 方法详解

### 整体框架

Data Amplify Learning (DAL) 包含两个核心组件：

1. **Diffusion-based Data Amplifier (DDA)**：基于预训练扩散模型，从 5 张种子图像"放大"生成 4000 张高质量配对数据
2. **TinyBeauty Model**：一个仅 14 层卷积的超轻量 U-Net，在放大数据上用像素级 L1 损失训练

### 关键设计

#### 1. Residual Diffusion Model (RDM)

解决扩散模型生成人像时面部细节（皱纹、纹理）被平滑化的核心难题。使用并行双分支推理：

- **有条件分支**：输入内容条件 $\mathbf{c}_{con}$ 和风格条件 $\mathbf{c}_{sty}$，生成平滑妆容图像
- **无条件分支**：不使用任何条件，生成平滑非妆容图像

定义两个关键残差：
- **细节残差** $R_d = x - \mathcal{F}_{fine}(x)$：原始图像与无条件重建之差，包含皱纹等面部细节
- **妆容残差** $R_m = \mathcal{F}_{fine}(x, c_{sty}+c_{con}) - \mathcal{F}_{fine}(x)$：有条件与无条件输出之差，包含纯粹的妆容变化

最终合成：

$$y_{detail} = \mathcal{F}_{fine}(x) + \lambda_m R_m + \lambda_d R_d$$

$\lambda_m=1, \lambda_d=0.8$，通过控制系数调节妆容强度和细节清晰度。$\lambda_d$ 设为 1.0 会因妆容相关细节的叠加导致过饱和效果。

#### 2. Fine-Grained Makeup Module (FGMM)

包含三个子模块：

**Style Preservation Block (SPB)**：
- 文本描述不足以精确描述妆容细微差异（如唇膏的具体色调和质感），改用视觉示例作为风格参考
- 将妆容应用到正面人脸图像上，用面部 mask 隔离妆容区域，创建纯妆容参考图
- 预训练图像编码器 + 可训练 MLP 将风格图像投影为 style tokens

**Identity Preservation Block (IPB)**：
- 学习妆容风格时可能无意中改变面部特征，需要将风格与身份解耦
- 初始考虑用 ArcFace 作为独立面部编码器，但编码空间不兼容
- 最终统一 SPB 和 IPB 的编码空间，共享 MLP 进行特征融合
- 全局条件向量分解为独立的内容条件 $\mathbf{c}_{con}$ 和风格条件 $\mathbf{c}_{sty}$

**Mask Guidance**：
- 将特征空间划分为三个区域：$M_{face}$、$M_{lips}$、$M_{eyes}$
- 训练时仅在 $M_{changed} = M_{face} + M_{lips} + M_{eyes}$ 区域计算损失
- 推理时限制 latent 空间修改仅作用于 mask 区域：$L'_y = L_y \odot M_{changed} + L_x \odot (1-M_{changed})$
- 支持多妆容风格组合——对不同面部 mask 应用不同妆容条件

#### 3. TinyBeauty 模型架构

受益于 DDA 生成的配对数据，模型可以极度精简：

- **纯卷积 U-Net**：仅 4 层卷积 + 4 个残差块，共 14 个卷积操作
- **参数量仅 81KB**（约 80K 参数）
- **输出残差而非完整图像**：$y' = M(x) + x$，只生成妆容残差，消除背景和头发区域的噪声伪影
- **分辨率无关**：残差可应用于任意分辨率图像而不损失纹理
- **无需面部预处理**：不依赖人脸解析、关键点检测等模块

### 损失函数 / 训练策略

**DDA 训练**：
- 基于 SD v1.5 + LoRA 微调，学习率 1e-4，训练 500 epochs
- 使用 OpenCLIP ViT-H/14 作为图像编码器，style token 和 identity token 长度均为 32
- 用 FaRL 生成 64×64 面部 mask 引导 latent 空间训练
- 五种妆容风格在单一模型中并发训练，V100 约 50 分钟
- Mask 区域损失：$L^M_{simple} = \mathbb{E}[\|(\epsilon - \epsilon_\theta) * M_{changed}\|^2]$

**TinyBeauty 训练**：
- 在 4000 张 DDA 生成图像上训练 50 epochs，学习率 2e-4，Adam 优化器
- **重建损失**：全局 L1 损失 $\mathcal{L}_{rec} = \|y - y'\|_1$
- **眼线损失**：用 Sobel 边缘算子提取眼线轮廓，$\mathcal{L}_s = \|\mathcal{S}(y) - \mathcal{S}(y')\|^2_2 * M_{eyes}$
- 加上感知损失和对抗损失
- V100 约 12 小时训练完成

## 实验关键数据

### 主实验

**FFHQ 和 MT 数据集上的定量对比（Style 1）：**

| 方法 | FFHQ PSNR↑ | FFHQ FID↓ | FFHQ LPIPS↓ | MT PSNR↑ | MT FID↓ | MT LPIPS↓ |
|------|-----------|----------|------------|---------|--------|----------|
| BeautyGAN | 26.50 | 45.25 | 0.0564 | 27.49 | 25.05 | 0.0434 |
| PSGAN | 25.65 | 36.22 | 0.0594 | 28.05 | 18.72 | 0.0301 |
| SCGAN | 27.55 | 36.98 | 0.0485 | 27.22 | 30.85 | 0.0467 |
| EleGANt | 30.18 | 25.47 | 0.0396 | 32.77 | 12.55 | 0.0191 |
| EleGANt* (DAL) | 35.45 | 10.78 | 0.0148 | 34.65 | 11.57 | 0.0164 |
| DDA | 35.96 | 10.28 | 0.0195 | 34.79 | 10.37 | 0.0231 |
| **TinyBeauty** | **35.39** | **8.03** | **0.0146** | **34.26** | **9.33** | **0.0181** |

TinyBeauty 在 FFHQ 上 PSNR 比 EleGANt 高 +5.21dB（17.3% 提升），FID 从 25.47 降至 8.03（68.5% 下降）。MT 数据集虽未参与训练，仍超越所有方法。

**模型效率对比（iPhone 13）：**

| 方法 | 参数量(M)↓ | FLOPs(G)↓ | 推理时间(ms)↓ |
|------|----------|----------|-------------|
| BeautyGAN | 8.04 | 24.70 | 27.89 |
| PSGAN | 8.41 | 91.28 | N/A |
| SCGAN | 35.33 | 288.51 | 195.61 |
| EleGANt | 10.27 | 127.94 | N/A |
| BeautyREC | 0.99 | 12.58 | 206.46 |
| **TinyBeauty** | **0.08** | **0.69** | **2.18** |

TinyBeauty 仅 80K 参数，推理速度 2.18ms，比最快竞争方法快 **13倍**，比面部解析预处理模块快 6倍。

### 消融实验

**DDA 各模块消融**（可视化）：
- 去掉 Mask Guidance + IPB → 面部身份丢失
- 去掉 SPB → 妆容风格不一致
- 去掉 RDM → 面部纹理被平滑化
- 完整模型结合所有模块 → 高质量妆容图像

**用户研究排名（100名评估者）：**

| 方法 | Rank-1 | Rank-2 | Rank-3 |
|------|--------|--------|--------|
| BeautyGAN | 0.18% | 0.55% | 0.98% |
| EleGANt | 10.46% | 84.07% | 2.27% |
| BeautyREC | 0.28% | 1.95% | 55.06% |
| **TinyBeauty** | **86.56%** | 10.81% | 2.55% |

TinyBeauty 获得 86.56% 的 Rank-1 票数，遥遥领先。

### 关键发现

- DAL 范式可泛化：用 DAL 重新训练 EleGANt，PSNR 从 30.18 提升至 35.45，证明数据质量比模型架构更重要
- 眼线损失对学习高频细节至关重要：没有它网络无法捕捉眼线轮廓
- 仅需 5 张种子图像即可训练出 SOTA 妆容模型
- 残差输出设计显著减少背景和头发区域的伪影

## 亮点与洞察

1. **范式转换**：从非配对学习 → 数据放大学习，彻底改变妆容迁移的训练范式
2. **极致压缩**：80K 参数 + 2.18ms 推理，实现了真正可部署于移动端的妆容模型
3. **RDM 的细节保真思路**：用双分支差值分离面部细节和妆容变化，具有广泛的扩散模型应用潜力
4. **少样本数据放大**：5 张图像生成 4000 张训练数据的思路可迁移到其他数据匮乏的领域

## 局限与展望

- DDA 生成数据质量依赖预训练扩散模型能力，极端妆容风格可能不够真实
- 当前仅验证 5 种妆容风格，可扩展到更多样化和复杂的妆容
- Mask Guidance 依赖 FaRL 生成面部 mask，极端姿态下可能不准确
- 模型在大姿态/大表情场景下仍有改进空间
- 未探索视频妆容迁移的时序一致性

## 相关工作与启发

- **EleGANt (ECCV 2022)**：通过生成伪 GT 简化优化为 L1 损失，思路相近但仍受限于非配对数据質量和大模型
- **IP-Adapter**：用图像 prompt 控制扩散模型生成，DDA 的 SPB 受其启发但针对妆容场景做了定制改进
- **BeautyREC**：抛弃 CycleGAN 结构实现较轻量化，但仍需面部预处理，0.99M 参数仍远大于 TinyBeauty
- **LoRA**：DDA 使用 LoRA 微调 SD，实现了 50 分钟内完成 5 种风格的并发训练

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 数据放大学习范式创新性极高，RDM 设计巧妙
- **有效性**: ⭐⭐⭐⭐⭐ — PSNR 提升 17.3%，模型压缩 100 倍以上，用户研究碾压式胜出
- **工程价值**: ⭐⭐⭐⭐⭐ — 2.18ms iPhone 推理，代码开源，极具实用价值
- **推荐度**: ⭐⭐⭐⭐⭐ — 方法创新 + 效果惊艳 + 工程落地，强烈推荐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Diffusion-Based Makeup Transfer with Facial Region-Aware Makeup Features](../../CVPR2026/image_generation/diffusion-based_makeup_transfer_with_facial_region-aware_makeup_features.md)
- [\[ECCV 2024\] A High-Quality Robust Diffusion Framework for Corrupted Dataset](a_highquality_robust_diffusion_framework_for_corrupted_datas.md)
- [\[ECCV 2024\] EMDM: Efficient Motion Diffusion Model for Fast and High-Quality Motion Generation](emdm_efficient_motion_diffusion_model_for_fast_and_high-quality_motion_generatio.md)
- [\[ECCV 2024\] DreamDiffusion: High-Quality EEG-to-Image Generation with Temporal Masked Signal Modeling and CLIP Alignment](dreamdiffusion_high-quality_eeg-to-image_generation_with_temporal_masked_signal_.md)
- [\[ECCV 2024\] UDiffText: A Unified Framework for High-quality Text Synthesis in Arbitrary Images via Character-aware Diffusion Models](udifftext_a_unified_framework_for_high-quality_text_synthesis_in_arbitrary_image.md)

</div>

<!-- RELATED:END -->
