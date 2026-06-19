---
title: >-
  [论文解读] Audio Super-Resolution with Latent Bridge Models
description: >-
  [NeurIPS 2025][图像恢复][音频超分辨率] 提出 AudioLBM，将音频波形压缩到连续隐空间，用桥模型实现从低分辨率到高分辨率的 latent-to-latent 生成过程，配合频率感知训练扩展数据利用和级联设计突破 48kHz 上限，在语音/音效/音乐上全面超越 AudioSR 等方法，并首次实现 any-to-192kHz 音频超分。
tags:
  - "NeurIPS 2025"
  - "图像恢复"
  - "音频超分辨率"
  - "隐空间桥模型"
  - "频率感知训练"
  - "级联超分"
  - "any-to-192kHz"
---

# Audio Super-Resolution with Latent Bridge Models

**会议**: NeurIPS 2025  
**arXiv**: [2509.17609](https://arxiv.org/abs/2509.17609)  
**代码**: 有（Demo: [https://AudioLBM.github.io/](https://AudioLBM.github.io/)）  
**领域**: 音频超分辨率 / 生成模型  
**关键词**: 音频超分辨率, 隐空间桥模型, 频率感知训练, 级联超分, any-to-192kHz

## 一句话总结

提出 AudioLBM，将音频波形压缩到连续隐空间，用桥模型实现从低分辨率到高分辨率的 latent-to-latent 生成过程，配合频率感知训练扩展数据利用和级联设计突破 48kHz 上限，在语音/音效/音乐上全面超越 AudioSR 等方法，并首次实现 any-to-192kHz 音频超分。

## 研究背景与动机

**领域现状**：音频超分辨率（SR）旨在将低分辨率波形上采样到高分辨率。现有方法涵盖映射式、GAN、扩散模型和桥模型。AudioSR 是目前最有代表性的跨域 any-to-48kHz 方法，基于梅尔频谱图隐空间的扩散模型。A2SB 在 STFT 域用 Schrödinger Bridge 做音乐带宽扩展。

**现有痛点**：现有方法的生成质量受限于生成先验与超分任务的不匹配：（1）AudioSR 从高斯噪声生成高分辨率内容（noise-to-latent），忽略了 LR 波形中丰富的先验信息；（2）A2SB 在 STFT 域将高频区域视为缺失并用高斯噪声填充，先验同样缺乏信息。此外，所有方法均无法超越 48kHz 上限，而专业音频制作需要 96kHz 甚至 192kHz。

**核心矛盾**：LR 波形本身已是 HR 目标的高信息量先验，但现有框架未能有效利用。生成过程应该是 LR→HR 的条件转换，而非从噪声出发。高分辨率训练数据稀缺也限制了方法的扩展性。

**本文目标**（1）设计能充分利用 LR 先验信息的生成框架；（2）解决高采样率训练数据稀缺问题；（3）突破 48kHz 超分上限至 96kHz 和 192kHz。

**切入角度**：将音频波形直接压缩到连续隐空间（保留 LR 先验信息），用桥模型连接 LR 和 HR 隐表示。引入频率感知训练实现 any-to-any 上采样，以及级联设计和先验增强突破采样率上限。

**核心 idea**：用波形隐空间中的桥模型替代噪声→数据的扩散生成，配合频率感知 + 级联设计实现高质量音频超分。

## 方法详解

### 整体框架

输入：LR 波形 $\bm{x}^{LR}$ → 波形 VAE 编码 → LR 隐向量 $\bm{z}^{LR}$ → 桥模型反向采样 → HR 隐向量 $\bm{z}^{HR}$ → VAE 解码 → HR 波形 $\bm{x}^{HR}$。训练时以 LR-HR 隐向量对为桥模型的两端边界分布，学习从 LR 到 HR 的生成路径。

### 关键设计

1. **波形隐空间桥模型（AudioLBM）**:

    - 功能：在连续隐空间中建立从 LR 到 HR 的生成路径，充分利用 LR 先验
    - 核心思路：训练卷积 VAE 将波形压缩为 $\bm{z} \in \mathbb{R}^{c \times l}$。以 $\bm{z}^{LR}$ 为先验（$t=T$, Dirac 分布）、$\bm{z}^{HR}$ 为目标（$t=0$），建立桥过程。训练噪声预测网络 $\epsilon_\theta(\bm{z}_t, t, \bm{z}_T)$，损失为 $\|\epsilon_\theta - (\bm{z}_t - \alpha_t \bm{z}_0)/(\alpha_t \sigma_t)\|_2^2$。推理时从 $\bm{z}^{LR}$ 出发执行一阶 SDE 反向采样（50步）
    - 设计动机：与 AudioSR 的 noise-to-latent 不同，桥模型的 latent-to-latent 路径天然继承 LR 波形的频谱结构和能量分布；与在 STFT 域操作相比，直接压缩波形避免了频率带不对齐

2. **频率感知训练（Frequency-aware LBMs）**:

    - 功能：克服高采样率训练数据稀缺，实现 any-to-any 超分
    - 核心思路：训练时动态采样 LR/HR 采样率对：先滤波得到 HR 版本（采样率 $SR_{HR}$ 低于原始但保留核心频段），再随机采样 $SR_{LR} \sim \mathcal{U}(0, SR_{HR})$ 生成 LR 版本。将先验频率 $f_{prior}$ 和目标频率 $f_{target}$ 作为正弦嵌入 token 前置到 DiT 输入。同时用常数缩放因子 $s$ 重缩放隐向量稳定训练
    - 设计动机：固定采样率训练浪费了大量非 48kHz 数据。频率感知使模型显式学习不同频带映射，推理时指定目标频率即可。实验证实训练数据多样性远比仅用 48kHz 数据重要

3. **级联 LBMs + 先验增强**:

    - 功能：突破单模型容量限制，实现 48→96→192kHz 渐进超分
    - 核心思路：分阶段训练多个 AudioLBM。为缓解级联误差，提出两种先验增强：（i）波形域退化——随机移除 HR 先验在 Nyquist 边界附近的部分高频细节；（ii）隐空间模糊——沿时间轴施加动态高斯平滑，比率 $b_r \sim \mathcal{U}(0, b_r^{max})$。训练目标变为从退化/模糊后的先验生成 UHR 目标
    - 设计动机：与扩散模型的噪声增强不同，桥模型边界是 Dirac 分布，模糊/退化更加自然。让模型训练时就接触降质先验，推理时对前级输出瑕疵更鲁棒

### 损失函数 / 训练策略

基础损失为噪声预测 MSE。频率感知版本增加频率条件输入。级联版本引入模糊先验和退化条件 $b_r$。训练数据约 5000 小时（语音+音效+音乐），有效 batch 128，1M 迭代。推理用 50 步一阶 SDE 采样。

## 实验关键数据

### 主实验

| 设定 | 指标 | AudioSR | 本文 (zero-shot) | 提升 |
|------|------|---------|---------------|------|
| VCTK 8→48kHz | LSD↓ | 0.940 | 0.753 | 19.9% |
| VCTK 8→48kHz | SSIM↑ | 0.809 | 0.893 | +0.084 |
| VCTK 8→48kHz | SigMOS↑ | 2.846 | 3.023 | +0.177 |
| 48Audio 8→48kHz | LSD↓ | 1.468 | 1.066 | 27.4% |
| ESC-50 16→44.1kHz | LSD↓ | 1.292 | 0.999 | 22.7% |
| SDS 16→44.1kHz | LSD↓ | 1.352 | 1.160 | 14.2% |

### 消融实验

| 配置 | ESC-50 LSD↓ | SDS LSD↓ | 说明 |
|------|-------------|----------|------|
| w/o Filter | 1.366 | 1.461 | 不过滤低采样率数据 |
| w/o Input-A | 1.052 | 1.187 | 无输入频率感知 |
| w/o Target-A | 1.022 | 1.166 | 仅输入频率感知 |
| Full (Ours) | 0.994 | 1.124 | 双向频率感知 |
| only 48kHz | 1.127 | 1.198 | 仅用48kHz训练数据 |

### 关键发现

- **频率感知训练逐步贡献明确**：数据过滤、输入频率感知、输出频率感知三组件依次改善，总共约 20% LSD 提升
- **级联系统显著优于直接训练**：16→96kHz 时，级联模型比直接 any-to-96kHz 模型降低 LSD(0-48) 达 0.415，ViSQOL 提升 0.32——让每阶段专注特定频段更有效
- **噪声预测优于数据预测**：在隐空间中，噪声预测目标优于桥模型文献中常用的数据预测目标
- 仅在 VCTK 训练的模型可进一步超越 zero-shot 版本，SigMOS 达 3.095 超过 GAN 方法 AP-BWE 的 3.082
- 192kHz 超分首次实现：LSD 从直接训练的 1.913 降至级联 1.365

## 亮点与洞察

- **LR→HR 先验利用的范式突破**：桥模型的 latent-to-latent 路径与超分本质完全对齐——LR 波形不是噪声而是 HR 的有信息降质版本。可迁移到图像/视频超分等条件生成任务
- **频率感知 + any-to-any 训练**非常巧妙：将固定条件变为可学习条件，用所有可用数据训练，同时模型获得了更好的频率理解能力。这是克服数据稀缺的通用范式
- **先验增强策略与桥模型特性的匹配**：扩散模型用加噪做级联增强，桥模型边界是 Dirac 分布用模糊更自然。"退化模拟前一阶段瑕疵"的思路对任何级联生成系统适用
- 首次突破 48kHz 上限至 192kHz，开辟专业音频制作新可能

## 局限与展望

- Zero-shot 模型在语音场景下有时将低频噪声误认为音效纹理，可通过域自适应改善
- 192kHz 阶段数据极度稀缺，必须依赖微调和数据增强，训练不够充分
- 50 步 SDE 采样速度较慢，可探索一致性蒸馏或流匹配加速
- VAE 压缩损失是系统上界——VAE 重建质量限制了最终性能上限
- 未在真实退化场景（混响、压缩伪影）下系统评估

## 相关工作与启发

- **vs AudioSR**: 在梅尔频谱图隐空间做 noise-to-latent 扩散，LR 仅作为条件。本文在波形隐空间做 latent-to-latent 桥过程，先验利用更自然有效
- **vs Bridge-SR**: 在波形域直接做桥模型（WaveNet 架构），泛化弱。本文提升到隐空间 + DiT 骨干
- **vs A2SB**: 在 STFT 域做桥模型但高频区域用噪声填充。本文避免了"挖空再填"

## 评分

- 新颖性: ⭐⭐⭐⭐ 桥模型+波形隐空间+频率感知三位一体的系统设计
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖三域、多采样率、完整消融和级联验证
- 写作质量: ⭐⭐⭐⭐ 图示清晰，方法描述完整
- 价值: ⭐⭐⭐⭐ 音频超分新 SOTA，打开了 >48kHz 方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Prior Does Matter: Visual Navigation via Denoising Diffusion Bridge Models](../../CVPR2025/image_restoration/prior_does_matter_visual_navigation_via_denoising_diffusion_bridge_models.md)
- [\[NeurIPS 2025\] Latent Harmony: Synergistic Unified UHD Image Restoration via Latent Space Regularization and Controllable Refinement](latent_harmony_synergistic_unified_uhd_image_restoration_via_latent_space_regula.md)
- [\[NeurIPS 2025\] Adaptive Discretization for Consistency Models](adaptive_discretization_for_consistency_models.md)
- [\[NeurIPS 2025\] Implicit Augmentation from Distributional Symmetry in Turbulence Super-Resolution](implicit_augmentation_from_distributional_symmetry_in_turbulence_super-resolutio.md)
- [\[ICML 2026\] PODiff: Latent Diffusion in Proper Orthogonal Decomposition Space for Scientific Super-Resolution](../../ICML2026/image_restoration/podiff_latent_diffusion_in_proper_orthogonal_decomposition_space_for_scientific_.md)

</div>

<!-- RELATED:END -->
