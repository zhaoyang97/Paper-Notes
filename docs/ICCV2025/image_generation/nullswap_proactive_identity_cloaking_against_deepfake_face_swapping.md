---
title: >-
  [论文解读] NullSwap: Proactive Identity Cloaking Against Deepfake Face Swapping
description: >-
  [ICCV2025][图像生成][Deepfake防御] 提出 NullSwap，通过在源图像中嵌入身份引导的不可见扰动来伪装面部身份信息，使 Deepfake 换脸模型无法提取正确身份，从而在纯黑盒场景下主动防御换脸攻击。 问题背景 Deepfake 换脸技术日益成熟，被动检测方法因生成质量提升而面临瓶颈…
tags:
  - "ICCV2025"
  - "图像生成"
  - "Deepfake防御"
  - "人脸换脸"
  - "主动扰动"
  - "身份伪装"
  - "对抗扰动"
  - "黑盒防御"
---

# NullSwap: Proactive Identity Cloaking Against Deepfake Face Swapping

**会议**: ICCV2025  
**arXiv**: [2503.18678](https://arxiv.org/abs/2503.18678)  
**代码**: 未开源  
**领域**: 图像生成  
**关键词**: Deepfake防御, 人脸换脸, 主动扰动, 身份伪装, 对抗扰动, 黑盒防御

## 一句话总结

提出 NullSwap，通过在源图像中嵌入身份引导的不可见扰动来伪装面部身份信息，使 Deepfake 换脸模型无法提取正确身份，从而在纯黑盒场景下主动防御换脸攻击。

## 研究背景与动机

### 问题背景
Deepfake 换脸技术日益成熟，被动检测方法因生成质量提升而面临瓶颈。主动防御通过在良性图像中预先插入不可见信号来破坏 Deepfake 操作，是更有前景的方向。

### 现有方法的不足

**视觉退化**：现有方法通过直接逐元素相加方式插入扰动，导致可见的视觉伪影（光照异常、模糊等）

**换脸防御薄弱**：已有方法主要针对面部属性编辑和面部重演，对换脸攻击的防御能力有限

**依赖生成模型**：大多数方法在训练时需要白盒或灰盒设置，引入实际的 Deepfake 生成模型或替代模型，计算开销大

### 核心洞察
本文分析了 Deepfake 换脸的本质：换脸攻击中，真正的受害者是**源图像**（身份提供者），而非目标图像。当名人的面孔被换到不当内容中时，需要保护的是被冒用身份的人。因此，防御重点应从保护目标图像转移到保护源身份信息。

## 方法详解

### 整体框架
NullSwap 框架包含四个核心模块：

#### 1. Identity Extraction（身份提取模块）
- 输入源图像 $I_s$，提取面部身份特征
- 使用 ConvBlock（CNN + BatchNorm + ReLU）+ MaxPooling
- 后接 $L=4$ 个 SEResBlock（ResNet bottleneck + SENet），以矩阵格式保留身份特征
- SENet 的 squeeze-and-excitation 机制高效分析通道间相关性

#### 2. Perturbation Block（扰动生成模块）
- 接收身份特征，生成身份引导的扰动
- ConvBlock 特征精炼后，$M=3$ 个 SEResBlock 进行层次化特征聚合
- 引入自适应随机噪声防止过拟合：

$$\text{RandNoise} = \beta \cdot (\alpha \cdot \text{RandNoise} + \eta)$$

其中 $\eta$ 是可学习噪声，$\alpha, \beta$ 是可学习的幅度调节参数

#### 3. Feature Block（特征提取模块）
- 对输入图像进行浅层特征提取
- 三个 ConvBlock 用于局部特征分析和维度调整
- $N=5$ 个 SEResBlock 增强特征适应性和上下文感知能力

#### 4. Cloaking Block（伪装重建模块）
- **特征级重建**：为扰动分配可学习权重 $\gamma$，与图像特征拼接后经 SEResBlock → DeConvBlock → ConvBlock → DeConvBlock 进行融合重建
- **图像级重建**：将特征级结果与原始输入拼接，经三个 ConvBlock 输出最终伪装图像 $I_s'$
- 双层重建确保视觉保真度的同时成功嵌入身份扰动

### Dynamic Loss Weighting（动态损失加权）

为确保对不同换脸算法（使用不同身份提取器）的泛化性，提出 DLW 机制自适应平衡多个人脸识别工具的身份损失：

$$\mathcal{L}_{id}(t_e, t_b) = \sum_{i=1}^{c} \hat{w_i}(t_e, t_b) \cdot \mathcal{L}_i(t_b)$$

权重由两个核心组件决定：
- **损失方差** $\sigma_i^2$：衡量近 $k=30$ 次迭代的损失稳定性，方差大则降权
- **相对进步** $\Delta_i$：评估损失改善速率，进步慢的损失获得更高权重

权重计算中 $\beta$ 随训练轮次线性增长（从 $\beta_{init}=0.5$ 到 $\beta^*=2$），逐步增强进步因子的影响。

### 总损失函数

$$\mathcal{L}_{total} = \lambda_{id} \mathcal{L}_{id} + \lambda_{MSE} \mathcal{L}_{MSE} + \lambda_{LPIPS} \mathcal{L}_{LPIPS} + \lambda_D \mathcal{L}_D$$

- $\mathcal{L}_{MSE}$：像素级重建质量（$\lambda_{MSE}=1.8$）
- $\mathcal{L}_{LPIPS}$：感知相似度（$\lambda_{LPIPS}=1.2$）
- $\mathcal{L}_D$：对抗损失，从头训练判别器（$\lambda_D=0.1$）
- $\mathcal{L}_{id}$：身份伪装损失，由 DLW 加权（$\lambda_{id}=0.08$）

## 实验关键数据

### 实验设置
- 数据集：CelebA-HQ（30K 图像，6217 身份）训练/测试 + LFW（5749 身份）跨数据集验证
- 人脸识别工具：ArcFace、FaceNet（训练用），VGGFace、SFace（测试用，验证泛化性）
- 换脸模型：SimSwap、InfoSwap、UniFace、E4S、DiffSwap（均仅在测试阶段出现）
- 硬件：8× Tesla A100 GPU，batch size 256，60 epochs

### 视觉质量（扰动图像 vs 原图）

| 指标 | Initiative | Anti-Forgery | CMUA | DF-RAP | **NullSwap** |
|------|-----------|-------------|------|--------|-------------|
| PSNR↑ | 39.38 | 38.07 | 38.64 | 38.85 | **41.31** |
| SSIM↑ | 0.9544 | 0.9530 | 0.9504 | 0.9349 | **0.9864** |
| LPIPS↓ | 0.0200 | 0.0282 | 0.0333 | 0.0511 | **0.0049** |

NullSwap 是唯一 PSNR > 40、SSIM > 0.98、LPIPS < 0.005 的方法。

### 身份伪装（CelebA-HQ，Top-1 准确率↓）

| 识别工具 | Clean | Initiative | Anti-Forgery | CMUA | DF-RAP | **NullSwap** |
|---------|-------|-----------|-------------|------|--------|-------------|
| ArcFace | 0.976 | 0.968 | 0.975 | 0.976 | 0.974 | **0.628** |
| FaceNet | 0.918 | 0.925 | 0.862 | 0.865 | 0.920 | **0.590** |
| VGGFace | 0.853 | 0.856 | 0.858 | 0.864 | 0.847 | **0.529** |
| SFace | 0.791 | 0.720 | 0.732 | 0.720 | 0.680 | **0.513** |
| 平均 | 0.885 | 0.867 | 0.857 | 0.856 | 0.855 | **0.565** |

其他方法平均准确率仍高于 0.85，NullSwap 降至 0.565，大幅领先。

### 换脸身份相似度（CelebA-HQ，ArcFace/VGGFace 余弦相似度↓）

| 换脸模型 | Initiative | CMUA | DF-RAP | **NullSwap** |
|---------|-----------|------|--------|-------------|
| SimSwap | 0.928/0.897 | 0.921/0.930 | 0.468/0.431 | **0.217/0.240** |
| InfoSwap | 0.941/0.919 | 0.947/0.888 | 0.913/0.849 | **0.375/0.359** |
| UniFace | 0.987/0.960 | 0.983/0.965 | 0.947/0.891 | **0.369/0.329** |
| E4S | 0.925/0.893 | 0.920/0.900 | 0.880/0.855 | **0.398/0.368** |
| DiffSwap | 0.660/0.657 | 0.658/0.648 | 0.636/0.631 | **0.310/0.352** |
| 平均 | 0.888/0.865 | 0.886/0.866 | 0.769/0.731 | **0.334/0.330** |

NullSwap 将换脸结果的身份相似度从 ~0.9 降至 ~0.33，所有换脸模型均被有效防御。

### DLW 消融实验

| 策略 | ArcFace↓ | FaceNet↓ | VGGFace↓ | SFace↓ | 平均↓ |
|------|---------|---------|---------|-------|------|
| 仅 ArcFace | 0.653 | 0.758 | 0.680 | 0.576 | 0.667 |
| 仅 FaceNet | 0.843 | 0.546 | 0.504 | 0.495 | 0.597 |
| 平均 | 0.846 | 0.613 | 0.600 | 0.525 | 0.646 |
| **DLW** | **0.628** | **0.590** | **0.529** | **0.513** | **0.565** |

DLW 在所有识别工具上表现均衡，平均最优。

## 亮点与洞察

1. **视角转换**：首次指出换脸防御应保护源身份而非目标图像，这一洞察改变了问题定义
2. **纯黑盒**：训练过程完全不涉及任何生成模型，仅依赖多个人脸识别工具，大幅降低计算成本
3. **特征级嵌入**：摒弃直接逐元素加法，通过浅层特征提取 + 双层重建将扰动自然融入图像，视觉质量全面超越现有方法
4. **DLW 自适应**：通过损失方差和相对进步双因子动态调权，避免简单平均导致的优化失衡
5. **跨模型泛化**：训练时仅用 ArcFace + FaceNet，测试时对未见过的 VGGFace、SFace 以及 5 种换脸模型均有效

## 局限与展望

1. **身份提取器多样性**：虽然 DLW 提升了泛化性，但训练仅用 2 个识别工具，更多样的训练信号可能进一步提升鲁棒性
2. **高分辨率适应**：实验在 256×256 分辨率进行，实际应用中高分辨率图像的扰动效果有待验证
3. **社交网络压缩**：虽然 DF-RAP 已探索 OSN 压缩，NullSwap 未明确讨论扰动在 JPEG 压缩/社交网络传输后的鲁棒性
4. **自适应攻击**：若攻击者知晓 NullSwap 的存在，可能设计针对性的净化方法去除扰动
5. **视频场景**：当前针对静态图像，视频换脸中帧间一致性的扰动策略值得探索
6. **伦理问题**：技术本身可能被反向利用（如阻止合法的身份验证），需要考虑部署场景

## 相关工作与启发

- **SimSwap (ACM MM 2020)**：基于 AdaIN 的身份注入范式，是换脸的里程碑工作
- **Initiative (AAAI 2021)**：灰盒主动防御，构建替代模型模仿 Deepfake 操作生成毒化扰动
- **Anti-Forgery (ACM MM 2022)**：在 Lab 颜色空间嵌入扰动以提升视觉兼容性
- **CMUA (AAAI 2022)**：跨模型通用对抗水印，迭代攻击多个 Deepfake 模型
- **DF-RAP (TIFS 2024)**：通过压缩近似 GAN 增强 OSN 场景下的扰动持久性
- **ArcFace / FaceNet / SENet**：身份特征提取与通道注意力的核心工具

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] VIGFace: Virtual Identity Generation for Privacy-Free Face Recognition Dataset](vigface_virtual_identity_generation_for_privacy-free_face_recognition_dataset.md)
- [\[CVPR 2026\] Attribute-Preserving Pseudo-Labeling for Diffusion-Based Face Swapping](../../CVPR2026/image_generation/attribute-preserving_pseudo-labeling_for_diffusion-based_face_swapping.md)
- [\[ICML 2025\] Synthetic Face Datasets Generation via Latent Space Exploration from Brownian Identity Diffusion](../../ICML2025/image_generation/synthetic_face_datasets_generation_via_latent_space_exploration_from_brownian_id.md)
- [\[CVPR 2026\] High-Fidelity Diffusion Face Swapping with ID-Constrained Facial Conditioning](../../CVPR2026/image_generation/high-fidelity_diffusion_face_swapping_with_id-constrained_facial_conditioning.md)
- [\[CVPR 2026\] Preserving Source Video Realism: High-Fidelity Face Swapping for Cinematic Quality](../../CVPR2026/image_generation/preserving_source_video_realism_high-fidelity_face_swapping_for_cinematic_qualit.md)

</div>

<!-- RELATED:END -->
