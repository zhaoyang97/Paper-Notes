---
title: >-
  [论文解读] Semantic Watermarking Reinvented: Enhancing Robustness and Generation Quality with Fourier Integrity
description: >-
  [ICCV 2025][图像生成][语义水印] 针对潜扩散模型（LDM）的语义水印方法因丢弃虚部而导致频率完整性缺失的问题，提出厄密对称傅里叶水印（SFW）和中心感知嵌入策略，在维持频域完整性的同时增强检测鲁棒性和生成质量。 随着 Stable Diffusion 等大规模语言-图像模型的开源，AI 生成内容的版权追踪和来源…
tags:
  - "ICCV 2025"
  - "图像生成"
  - "语义水印"
  - "潜空间傅里叶水印"
  - "厄密对称性"
  - "中心感知嵌入"
  - "潜扩散模型"
---

# Semantic Watermarking Reinvented: Enhancing Robustness and Generation Quality with Fourier Integrity

**会议**: ICCV 2025  
**arXiv**: [2509.07647](https://arxiv.org/abs/2509.07647)  
**代码**: [https://github.com/thomas11809/SFWMark](https://github.com/thomas11809/SFWMark)  
**领域**: 扩散模型/数字水印  
**关键词**: 语义水印, 潜空间傅里叶水印, 厄密对称性, 中心感知嵌入, 潜扩散模型

## 一句话总结

针对潜扩散模型（LDM）的语义水印方法因丢弃虚部而导致频率完整性缺失的问题，提出厄密对称傅里叶水印（SFW）和中心感知嵌入策略，在维持频域完整性的同时增强检测鲁棒性和生成质量。

## 研究背景与动机

随着 Stable Diffusion 等大规模语言-图像模型的开源，AI 生成内容的版权追踪和来源验证变得迫切。在生成过程中嵌入不可见水印是主要解决方案之一。

**语义水印方法现状与问题**：

**Tree-Ring / RingID 等方法**：在潜向量的傅里叶域中嵌入几何图案（环形水印），采用 merged-in-generation 方案，对再生攻击具有天然鲁棒性。

**频率完整性缺失**：现有方法在傅里叶域修改后做逆 FFT 时直接丢弃虚部，导致：
   - **实部信息失真**：原始水印图案被扭曲
   - **虚部完全丢失**：频域中关键区域变空
   - **检测精度下降**：检测只能利用不完整的频率信息
   - **生成质量降低**：空间域信号偏离实高斯分布

**裁剪攻击脆弱**：在全空间矩阵上应用 FFT 嵌入水印，裁剪后水印信息大量丢失。

**核心洞察**：如果在频域修改时保持**厄密对称性**（Hermitian Symmetry），逆 FFT 自然得到实信号，无需丢弃虚部，从而保留完整的频率信息。

## 方法详解

### 整体框架

在潜扩散模型的 merged-in-generation 流程中：潜噪声 → FFT → 嵌入水印到关键区域 → IFFT → 文本引导生成水印图像。检测时通过 DDIM 反演获取潜查询，分析频域关键区域。

本文在此流程中引入两个改进：(1) 厄密对称傅里叶水印（SFW）；(2) 中心感知嵌入策略。

### 关键设计

1. **厄密对称傅里叶水印（SFW）**：

    - **核心原理**：实信号的 DFT 满足厄密对称条件 $F[M-k, N-l] = \overline{F[k,l]}$，即关于 DC 中心共轭对称。
    - **设计约束**：频域的自由区域为半区域（另一半由对称性决定）；DC 中心和 Nyquist 频率点的虚部必须为零。
    - **效果**：IFFT 后的空间域信号为纯实数，无需丢弃虚部；水印的实部和虚部信息都被完整保留，检测可利用全部频率信息。
    - **高斯性保持**：实高斯噪声经 FFT 变换为复高斯噪声 $f[m,n] \sim \mathcal{N}(0, \sigma^2) \Rightarrow F[k,l] \sim \mathcal{CN}(0, MN\sigma^2)$。保持厄密对称使空间域信号更贴近实高斯分布，扩散模型初始化更稳定。

2. **中心感知嵌入策略**：

    - 不在全空间矩阵（64×64）上做 FFT，而仅对中心区域（44×44）应用 FFT 后嵌入水印。
    - **设计动机**：裁剪攻击通常去除边缘区域，中心区域的信息保留率最高。
    - 显著提升了对各种比例裁剪攻击的鲁棒性。

3. **HSTR（改进 Tree-Ring）**：对 Tree-Ring 的水印图案施加厄密对称约束并结合中心感知嵌入。

4. **HSQR（QR码水印）**：

    - 将 QR 码一分为二，分别嵌入频域自由半区域的实部和虚部。
    - 嵌入公式：$\text{HSQR}(\tilde{x}, c) = \begin{cases} +|F(\tilde{x},c)|, & \text{if QR}(x)=1 \\ -|F(\tilde{x},c)|, & \text{if QR}(x)=0 \end{cases}$
    - 嵌入区域偏移 DC 轴一个像素以避免数值不稳定。

### 损失函数 / 训练策略

本方法为无需训练的嵌入方案（merged-in-generation），不涉及额外的损失函数训练。水印嵌入在生成流程中完成，不增加额外处理时间。

## 实验关键数据

### 主实验 — 验证任务（TPR@1%FPR, MS-COCO）

| 方法 | 无攻击 | 亮度 | JPEG | 模糊 | 噪声 | BM3D | VAE-B | Diff | 中心裁剪 | 随机裁剪 | 平均 |
|------|--------|------|------|------|------|------|-------|------|---------|---------|------|
| Tree-Ring | 0.957 | 0.463 | 0.548 | 0.934 | 0.412 | 0.815 | 0.509 | 0.543 | 0.509 | 0.734 | 0.655 |
| Zodiac | 0.998 | 0.843 | 0.973 | 0.998 | 0.880 | 0.997 | 0.944 | 0.972 | 0.989 | 0.995 | 0.962 |
| **HSTR (ours)** | **1.000** | 0.899 | **0.994** | **1.000** | 0.806 | **0.999** | **0.973** | **0.997** | **1.000** | **1.000** | **0.971** |
| RingID | 1.000 | 0.988 | 1.000 | 1.000 | 0.987 | 1.000 | 0.992 | 1.000 | 1.000 | 1.000 | 0.997 |
| **HSQR (ours)** | **1.000** | 0.991 | **1.000** | **1.000** | 0.983 | **1.000** | **0.992** | **1.000** | **1.000** | **1.000** | **0.997** |

HSTR 相比 Tree-Ring 平均提升 31.6 个百分点；HSQR 与 RingID 精度相当但生成质量更优。

### 生成质量对比

| 方法 | FID↓ | CLIP Score↑ | 说明 |
|------|------|------------|------|
| 无水印 | 基准 | 基准 | - |
| Tree-Ring | +轻微 | 轻微下降 | 频率失真影响质量 |
| RingID | +明显 | 下降 | 高能量图案产生可见环形伪影 |
| **HSTR** | +极小 | 几乎不变 | 频率完整性保护质量 |
| **HSQR** | +极小 | 几乎不变 | 同上 |

### 消融实验

| 配置 | 验证性能 | 生成质量 | 说明 |
|------|---------|---------|------|
| 无 SFW | 基线(Tree-Ring) | 频率失真 | 虚部丢失导致检测退化 |
| + SFW（厄密对称） | 大幅提升 | 显著改善 | 频率完整性恢复 |
| + 中心感知嵌入 | 裁剪鲁棒性大增 | 无损 | 中心 44×44 区域 |
| 信息容量分析 | QR码容量 vs 精度权衡 | - | 更大 QR 码 → 更低匹配率 |

### 关键发现

- **频率完整性是核心**：仅通过保持厄密对称（无需任何训练或额外计算），即可大幅提升检测精度和生成质量。
- RingID 的高能量图案导致可见伪影（论文 Fig.4 右侧可见环形纹理），而 HSTR/HSQR 无此问题。
- 中心感知嵌入在中心裁剪和随机裁剪场景中均显著提升（Tree-Ring 裁剪场景从 0.509/0.734 提升到 HSTR 的 1.000/1.000）。
- 扩散再生攻击（Diff）下，HSTR 和 HSQR 的 TPR 接近 1.0，验证了语义水印固有的再生鲁棒性。

## 亮点与洞察

- **问题定义精准**：准确识别了现有语义水印方法中"丢弃虚部"这一被忽视但关键的问题。
- **解决方案优雅**：利用已知的傅里叶数学性质（厄密对称）无需训练即可解决问题，"修复 bug" 级别的改进带来巨大收益。
- **无额外计算开销**：所有改进都在嵌入阶段完成，不增加推理时间。
- QR 码水印方案兼具高容量（支持 identification）和强鲁棒性。

## 局限与展望

- 当前中心感知嵌入使用固定的 44×44 区域，可探索自适应区域选择。
- HSQR 的 QR 码容量受频域面积限制，更大信息量需要更精细的编码方案。
- 虽然方法无需训练，但依赖 DDIM 反演质量——反演误差会影响检测。
- 尚未探索在其他生成架构（如 DiT、FLUX）上的适用性。

## 相关工作与启发

- 对 Tree-Ring、RingID 等开创性工作的直接改进；Zodiac 的频域优化方案因迭代成本高而另辟蹊径。
- 启发：在信号处理基础理论上的"微小修正"（厄密对称性）有时比复杂的学习方法更有效。

## 评分

- **新颖性**: ⭐⭐⭐ — 核心是对已知数学性质的正确应用，创新性偏增量
- **技术深度**: ⭐⭐⭐⭐ — 频域分析透彻，高斯性保持论证完整
- **实验充分度**: ⭐⭐⭐⭐⭐ — 12 种攻击、4 个数据集、多基线对比
- **实用价值**: ⭐⭐⭐⭐⭐ — 无训练、无开销、即插即用，极强实用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Enhancing Reward Models for High-quality Image Generation: Beyond Text-Image Alignment](enhancing_reward_models_for_high-quality_image_generation_beyond_text-image_alig.md)
- [\[CVPR 2025\] Noise Diffusion for Enhancing Semantic Faithfulness in Text-to-Image Synthesis](../../CVPR2025/image_generation/noise_diffusion_for_enhancing_semantic_faithfulness_in_text-to-image_synthesis.md)
- [\[ICCV 2025\] Your Text Encoder Can Be An Object-Level Watermarking Controller](your_text_encoder_can_be_an_object-level_watermarking_controller.md)
- [\[NeurIPS 2025\] UltraHR-100K: Enhancing UHR Image Synthesis with A Large-Scale High-Quality Dataset](../../NeurIPS2025/image_generation/ultrahr-100k_enhancing_uhr_image_synthesis_with_a_large-scale_high-quality_datas.md)
- [\[ICCV 2025\] Invisible Watermarks, Visible Gains: Steering Machine Unlearning with Bi-Level Watermarking Design](invisible_watermarks_visible_gains_steering_machine_unlearning_with_bi-level_wat.md)

</div>

<!-- RELATED:END -->
