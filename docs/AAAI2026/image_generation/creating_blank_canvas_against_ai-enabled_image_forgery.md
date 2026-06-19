---
title: >-
  [论文解读] Creating Blank Canvas Against AI-Enabled Image Forgery
description: >-
  [AAAI 2026][图像生成][图像篡改检测] 提出"空白画布"机制，通过对抗扰动使 SAM 对受保护图像"视而不见"，当图像被篡改后篡改区域会被 SAM 自动识别，实现无需篡改训练数据的主动式篡改定位。 AIGC 图像编辑技术（如 SD Inpaint、ControlNet、SDXL）使得高度逼真的图像篡改变得极为简单…
tags:
  - "AAAI 2026"
  - "图像生成"
  - "图像篡改检测"
  - "对抗扰动"
  - "SAM"
  - "频率感知优化"
  - "主动保护"
---

# Creating Blank Canvas Against AI-Enabled Image Forgery

**会议**: AAAI 2026  
**arXiv**: [2511.22237](https://arxiv.org/abs/2511.22237)  
**代码**: [GitHub](https://github.com/qsong2001/blank_canvas)  
**领域**: 图像生成  
**关键词**: 图像篡改检测, 对抗扰动, SAM, 频率感知优化, 主动保护

## 一句话总结

提出"空白画布"机制，通过对抗扰动使 SAM 对受保护图像"视而不见"，当图像被篡改后篡改区域会被 SAM 自动识别，实现无需篡改训练数据的主动式篡改定位。

## 研究背景与动机

AIGC 图像编辑技术（如 SD Inpaint、ControlNet、SDXL）使得高度逼真的图像篡改变得极为简单，对公众信任和社会稳定构成严重威胁。现有篡改定位方法主要采用**被动事后分析**策略，依赖于训练阶段学到的特定伪造模式，面临以下问题：

**泛化能力不足**：仅对训练中见过的篡改类型有效，遇到新型 AIGC 编辑时效果骤降

**数据依赖严重**：需要大量标注的篡改样本用于训练

**计算资源门槛高**：训练篡改检测基础模型需要巨大算力

作者提出一个关键洞察：**在空白画布上的涂改远比在复杂图像上的修改更容易被发现**。因此，与其事后检测篡改痕迹，不如事先将图像在视觉基础模型的视角下转化为"空白画布"——任何篡改都会变得显而易见。

## 方法详解

### 整体框架

整体流程包含两个阶段：

1. **空白画布创建阶段**：对原始图像添加不可见对抗扰动，使 SAM 无法对图像进行任何分割（"Segment Nothing"）
2. **篡改定位阶段**：当受保护图像被篡改后，篡改区域破坏了对抗扰动，SAM 可重新感知并分割出这些区域

图像拥有者在发布前对图像执行保护操作，扰动对人眼不可见但能完全欺骗 SAM。使用 SAM ViT-H 作为基础模型，采用单点提示（坐标 (0,0)）进行推理。

### 关键设计

#### 1. 空白画布创建——基础对抗攻击

核心目标是找到扰动 $\delta$，使 SAM 对受保护图像的输出置信度收敛于常数 $C$：

$$\Phi' = \text{SAM}(x_{\text{clear}} + \delta, \mathcal{P}), \quad \Phi'[i,j] \approx C, \; \forall i,j$$

采用 MSE 损失优化扰动：

$$\mathcal{L}_{\text{attack}} = \text{MSE}(\Phi, C)$$

其中 $C=15$（与 SAM 背景区域的典型置信度值一致）。使用 PGD 框架优化，最大扰动幅度 $16/255$，步长 $2/255$。

#### 2. 频率感知优化（核心创新）

实验发现朴素对抗攻击无法完全欺骗 SAM——SAM 在高频边缘和纹理区域仍保留感知能力。因此提出频率感知优化策略，包含三个协同组件：

**（a）小波域高频分解（$\mathcal{L}_{\text{hfc}}$）**：使用 Daubechies-8 小波基进行离散小波变换，提取高频分量并以 Canny 边缘掩码约束扰动：

$$\mathcal{L}_{\text{hfc}} = \sum_{k=1}^{K} \|\mathcal{W}_k(\tilde{x}) \odot M_{\text{edge}} - \mathcal{W}_k(x) \odot M_{\text{edge}}\|_F^2$$

**（b）结构保持约束（$\mathcal{L}_{\text{lfc}}$）**：利用自适应 SSIM 保护低频分量，维持图像视觉自然度：

$$\mathcal{L}_{\text{lfc}} = \text{SSIM}(\phi_m, \tilde{\phi}_m)$$

**（c）自适应频谱优化**：在频率域中设计频谱投影掩码 $\mathcal{M}$，将扰动能量集中在高频带：

$$\mathcal{M}(u,v) = \begin{cases} 1, & \sqrt{u^2+v^2} \geq f_{\text{cutoff}} \\ 0, & \text{otherwise} \end{cases}$$

最终优化目标融合所有损失项，并采用带动量的自适应步长梯度更新方法。

#### 3. 篡改定位

对受保护的"空白画布"图像，篡改操作 $\Delta x$ 会破坏局部对抗扰动，使 SAM 重新感知这些区域：

$$\mathcal{M}_{\text{tamper}} = \mathbb{I}(\|\text{SAM}(\tilde{x} + \Delta x)\|_2 > \tau_{\text{detect}})$$

检测阈值 $\tau_{\text{detect}}$ 通过 Otsu 方法自适应确定。

### 损失函数 / 训练策略

总体优化目标为：

$$\delta^* = \arg\max_{\|\delta\|_\infty \leq \epsilon} \mathcal{L}_{\text{attack}} + \lambda \mathcal{L}_{\text{lfc}} - \beta \mathcal{L}_{\text{hfc}}$$

- **免训练方法**：整个过程不需要训练任何网络，仅优化图像级扰动
- 使用 PGD 框架，带频谱投影的动量梯度更新
- 学习率采用指数预热策略：$\alpha_t = \alpha_0(1 - e^{-5t/T})$

## 实验关键数据

### 主实验

**表1：经典篡改定位基准对比**

| 方法 | CASIA1+ IoU/F1 | Columbia IoU/F1 | NIST IoU/F1 |
|------|---------------|-----------------|-------------|
| MVSS-Net | 0.40/0.48 | 0.48/0.61 | 0.24/0.29 |
| FakeShield | 0.56/0.62 | 0.68/0.76 | 0.34/0.39 |
| EditGuard | 0.60/0.67 | 0.70/0.78 | 0.35/0.40 |
| **Ours** | **0.62/0.67** | **0.74/0.81** | **0.31/0.45** |

**表2：AIGC 编辑方法对比（F1/IoU）**

| 方法 | SD Inpaint | ControlNet | SDXL | RePaint |
|------|-----------|------------|------|---------|
| MVSS-Net† | 0.694/0.575 | 0.678/0.558 | 0.482/0.359 | 0.185/0.111 |
| EditGuard | 0.966/0.936 | 0.968/0.940 | 0.965/0.936 | 0.967/0.938 |
| **Ours** | **0.972/0.958** | **0.973/0.938** | **0.970/0.958** | **0.961/0.957** |

在 AIGC 编辑场景下，本方法 F1 > 95%、IoU ≈ 95%，全面超越被动方法且与 EditGuard 持平或略优。

### 消融实验

| 配置 | F1 | IoU |
|------|----|----|
| 无保护 (a) | 0.352 | 0.378 |
| 仅 $\mathcal{L}_{\text{mse}}$ + $\mathcal{L}_{\text{stealth}}$ (b) | 0.934 | 0.928 |
| $\mathcal{L}_{\text{mse}}$ + 自适应优化 (c) | 0.931 | 0.921 |
| **完整方法** | **0.964** | **0.955** |

### 关键发现

1. 朴素对抗攻击（仅 MSE loss）在高频区域仍无法完全欺骗 SAM，会产生假阳性
2. 频率感知优化是关键——移除后性能退化到无保护水平
3. 方法无需任何篡改训练数据，是真正的零样本篡改定位
4. 在 AIGC 编辑场景下显著优于所有被动方法（被动方法 F1 < 0.7）

## 亮点与洞察

1. **范式创新**：从被动检测转向主动保护，将"检测篡改"问题转化为"在空白画布上找痕迹"——概念简洁而有效
2. **免训练设计**：仅优化图像级扰动，不需要训练任何模型，利用现成 SAM 即可
3. **频域理解深刻**：准确诊断了 SAM 在高频区域的鲁棒性问题，并提出针对性的频率感知优化
4. **实用性强**：图像拥有者可在发布前一键保护，任何人都可用标准 SAM 验证篡改

## 局限与展望

1. **白盒假设**：当前方法需要访问 SAM 的完整模型权重，对其他视觉基础模型的迁移性未验证
2. **扰动可见性**：$16/255$ 的扰动上界在某些场景下可能影响图像质量
3. **社交平台压缩鲁棒性**：JPEG 压缩、缩放等社交媒体常见操作可能破坏对抗扰动
4. **对抗性篡改**：若篡改者知道保护机制，可能设计针对性的"去保护"攻击
5. 可探索将方法扩展到 SAM2 或其他视觉基础模型

## 相关工作与启发

- **EditGuard**：同为主动保护方法，但依赖隐写术嵌入信息，可解释性不足
- **SAM-Attack / Dark-SAM**：研究 SAM 对抗鲁棒性，但目标是攻击而非保护
- 本文将对抗攻击创造性地应用于图像保护，思路值得借鉴到其他领域
- 启发：可否用类似思路保护视频、3D 内容免受 AI 篡改？

## 评分

- 新颖性: ⭐⭐⭐⭐⭐（空白画布概念非常新颖）
- 技术深度: ⭐⭐⭐⭐（频率感知优化设计精巧）
- 实验完整性: ⭐⭐⭐⭐（经典+AIGC 双场景验证）
- 实用价值: ⭐⭐⭐⭐（免训练，直接可用）
- 总体评分: 8.5/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] BlurGuard: A Simple Approach for Robustifying Image Protection Against AI-Powered Edit](../../NeurIPS2025/image_generation/blurguard_a_simple_approach_for_robustifying_image_protection_against_ai-powered.md)
- [\[ICML 2026\] Order within Chaos: Capturing Intrinsic Energy Anomalies for AI-Manipulated Image Forgery Localization](../../ICML2026/image_generation/order_within_chaos_capturing_intrinsic_energy_anomalies_for_ai-manipulated_image.md)
- [\[AAAI 2026\] MacPrompt: Maraconic-guided Jailbreak against Text-to-Image Models](macprompt_maraconic-guided_jailbreak_against_text-to-image_models.md)
- [\[AAAI 2026\] Aggregating Diverse Cue Experts for AI-Generated Image Detection](aggregating_diverse_cue_experts_for_ai-generated_image_detec.md)
- [\[CVPR 2026\] Rel-Zero: Harnessing Patch-Pair Invariance for Robust Zero-Watermarking Against AI Editing](../../CVPR2026/image_generation/rel-zero_harnessing_patch-pair_invariance_for_robust_zero-watermarking_against_a.md)

</div>

<!-- RELATED:END -->
