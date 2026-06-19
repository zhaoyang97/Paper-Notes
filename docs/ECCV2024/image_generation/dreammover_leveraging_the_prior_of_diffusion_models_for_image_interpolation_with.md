---
title: >-
  [论文解读] DreamMover: Leveraging the Prior of Diffusion Models for Image Interpolation with Large Motion
description: >-
  [ECCV 2024][图像生成][图像插值] 提出 DreamMover，基于预训练文本到图像扩散模型实现大运动图像对之间的插值，通过扩散感知光流估计、两级潜空间融合和自注意力拼接替换三个核心组件，生成语义一致的中间帧。 从两张存在大运动的图像生成中间过渡帧是一个具有重要实际价值但极具挑战性的任务。现有方法存在明显局限：…
tags:
  - "ECCV 2024"
  - "图像生成"
  - "图像插值"
  - "扩散模型先验"
  - "光流估计"
  - "语义一致性"
  - "大运动"
---

# DreamMover: Leveraging the Prior of Diffusion Models for Image Interpolation with Large Motion

**会议**: ECCV 2024  
**arXiv**: [2409.09605](https://arxiv.org/abs/2409.09605)  
**代码**: [项目页面](https://dreamm0ver.github.io)  
**领域**: 图像生成  
**关键词**: 图像插值, 扩散模型先验, 光流估计, 语义一致性, 大运动

## 一句话总结

提出 DreamMover，基于预训练文本到图像扩散模型实现大运动图像对之间的插值，通过扩散感知光流估计、两级潜空间融合和自注意力拼接替换三个核心组件，生成语义一致的中间帧。

## 研究背景与动机

从两张存在大运动的图像生成中间过渡帧是一个具有重要实际价值但极具挑战性的任务。现有方法存在明显局限：

- **视频帧插值方法**（如 Film、LDMVFI）：主要设计用于提高视频帧率，处理的是相邻帧间的小运动差异。在大运动场景下缺乏语义认知能力，容易产生伪影和物体断裂
- **图像变形方法**（如 DiffInterp、DiffMorpher）：主要关注拓扑相似物体间的过渡（如不同表情的人脸），对同一物体在大运动场景下的语义一致性建模能力有限

核心挑战在于：当两张输入图像的运动差异很大时，中间帧的语义信息可能在两张输入图像中都不存在（如动物张嘴过程中的半开状态）。作者提出利用预训练扩散模型的丰富隐式语义信息来补充缺失的中间语义表示，确保生成结果与输入保持一致。

## 方法详解

### 整体框架

DreamMover 基于 Stable Diffusion 1.5，流程分为三步：
1. **扩散感知光流估计**：从 U-Net 特征图建立两张图像间的像素对应关系
2. **两级潜空间融合**：将高层语义和低层细节分别处理，避免加权平均导致的高频信息丢失
3. **参考引导一致性增强**：通过自注意力拼接替换 + LoRA 微调确保输出与输入图像的语义一致性

给定输入图像对 $\mathcal{I}^0$ 和 $\mathcal{I}^1$，目标是生成中间图像 $\mathcal{I}^\delta$（$\delta \in (0,1)$）构成语义一致的视频。

### 关键设计

**1. 扩散感知光流估计**

利用扩散模型 U-Net 在加噪过程中提取的特征图隐式建立语义对应关系，无需额外的光流预测模块：

- 将两张图像编码到潜空间后，通过 DDIM 反转送入 U-Net 加噪
- 从 U-Net 上采样块的第 14 步提取第二个上采样块的特征图 $f^0$, $f^1$
- 通过遍历一个特征图的所有点，在另一个特征图中找到余弦相似度最高的对应位置，得到双向光流：

$$F^{0 \to 1}(x,y) = \arg\max_{i,j} \langle f^0(x,y), f^1(i,j) \rangle$$

- 通过线性缩放得到任意中间时刻的光流：$F^{0 \to \delta} = \delta \cdot F^{0 \to 1}$

作者通过 PCA 可视化验证了扩散模型特征图的空间布局与原始图像高度一致，为光流估计提供了可靠基础。

**2. 两级潜空间融合**

直接在潜空间进行加权平均融合会导致严重的高频信息丢失（模糊），原因是 softmax splatting 和时间插值都引入了平均操作。

核心观察：DDIM 去噪过程的两个分量具有不同的频率特性：
- $z_{t \to 0}$（一步预测的干净潜码）：主要包含高层上下文信息，缺少高频细节
- $\epsilon_\theta(z_t, t)$（预测噪声）：包含更多低层纹理的高频成分

基于此提出两级融合策略：
- **高层信息**：在 $z_{T \to 0}$ 空间使用 softmax splatting + 时间加权插值

$$z^\delta_{T \to 0} = (1-\delta) \cdot \vec{\sigma}(z^0_{T \to 0}, F^{0 \to \delta}) + \delta \cdot \vec{\sigma}(z^1_{T \to 0}, F^{1 \to \delta})$$

- **低层信息**：在 $\epsilon_\theta$ 空间使用 Winner-Takes-All (WTA) 操作，取权重最高的值，避免平均操作导致的高频丢失

$$\epsilon^\delta = WTA(\epsilon_\theta(z^0_T), \epsilon_\theta(z^1_T))$$

最后将两部分合成：$z^\delta_T = \sqrt{\alpha_T} \cdot z^\delta_{T \to 0} + \sqrt{1-\alpha_T} \cdot \epsilon^\delta$

频谱分析证实：两级融合保留了显著更多的高频能量。

**3. 自注意力拼接与替换**

在去噪过程中，将输入图像对的自注意力特征注入中间图像的去噪过程：

- 将两张输入图像的噪声潜码分别送入 U-Net，提取各层的 Key 和 Value 矩阵
- 对中间图像的自注意力模块，保留其 Query，用输入图像的 Key/Value 拼接替换：

$$Q = Q^\delta, \quad K = (K^0 \oplus K^1), \quad V = (V^0 \oplus V^1)$$

这样中间潜码可以从两张输入图像中查询相关的局部结构和纹理，增强一致性。

此外还使用单个 LoRA（秩 16，80 步微调，约 40 秒完成）适配输入图像对，进一步提高语义一致性。注意与 DiffMorpher 不同，DreamMover 只需一个 LoRA 同时适配两张图像。

### 损失函数 / 训练策略

- DDIM 使用 50 步，在第 30 步加噪处执行潜码优化
- 生成 32 张中间插值图像
- LoRA 使用 AdamW 优化器，学习率 $5 \times 10^{-4}$
- 不使用 classifier-free guidance（CFG 会累积数值误差导致过饱和）
- 整个流程在单块 NVIDIA RTX 3090 上运行

## 实验关键数据

### 主实验

在自建 InterpBench 基准（100 对大运动图像）上的定量比较：

| 方法 | FID ↓ | LPIPS ↓ | WE ↓ | WE_mid ↓ |
|------|-------|---------|------|----------|
| DiffInterp | 185.78 | 0.5375 | 0.5112 | 0.9573 |
| DiffMorpher | 68.23 | 0.3061 | 0.2673 | 0.7784 |
| Film | 54.28 | 0.2313 | **0.1244** | 0.4176 |
| LDMVFI | 48.35 | 0.2347 | 0.1453 | 0.4373 |
| **DreamMover** | **43.18** | **0.2227** | 0.2069 | **0.3687** |

### 消融实验

用户偏好研究（两两比较，偏好本方法的百分比）：

| 对比方法 | 更偏好 DreamMover 的比例 |
|----------|------------------------|
| vs DiffInterp | 显著优势 |
| vs DiffMorpher | 显著优势 |
| vs Film | 明显优势 |
| vs LDMVFI | 优势 |

关键消融：
- 两级融合 vs 直接融合：两级融合显著减少模糊，保留更多高频细节
- 自注意力替换：是保持外观一致性的关键
- LoRA 微调：进一步增强语义身份一致性

### 关键发现

1. 扩散模型的 U-Net 特征天然适合做语义对应和光流估计，无需额外训练
2. 直觉上的加权平均融合会摧毁高频信息——将信号分解为高层/低层分别处理是关键创新
3. Film 和 LDMVFI 在时间一致性指标上更好，但生成的中间内容不保证语义正确，影响实际视频质量
4. 本方法推理无需训练，仅需 LoRA 微调（~40秒）和优化

## 亮点与洞察

1. **利用扩散模型先验做光流**：无需光流网络，直接从 U-Net 特征建立对应关系，这是一个被低估的能力
2. **两级融合的频率分析洞察**：清晰地揭示了潜空间中 $z_{t \to 0}$ 和 $\epsilon_\theta$ 的频率特性差异，为融合策略提供了理论支撑
3. **InterpBench 基准**：首个专门针对大运动图像插值语义一致性评估的基准数据集
4. **轻量级方法**：基于预训练模型，无需从头训练，单 GPU 可用

## 局限与展望

1. 光流估计基于余弦相似度最大值匹配，在严重遮挡或对称场景下可能失效
2. 时间一致性指标 WE 不如 Film/LDMVFI，说明逐帧生成的平滑度仍有改进空间
3. 生成 32 帧需要多次 DDIM 反转和去噪，计算成本较高
4. 依赖 DDIM 反转的精度，反转误差会累积到最终结果
5. 仅基于 SD 1.5，未验证在更强的扩散模型上的效果

## 相关工作与启发

- **Diffusion Features for Correspondence**：与 DIFT 等工作共享"扩散特征可做对应"的洞察，但首次用于图像插值的光流估计
- **视频帧插值**：Film 处理近似重复帧效果好，但缺乏语义认知；DreamMover 填补了大运动场景的空白
- **启发**：两级融合策略可推广到其他需要保留细节的潜空间操作（如编辑、修复）

## 评分

- 新颖性：⭐⭐⭐⭐ — 三个组件各有创新，两级融合尤为巧妙
- 技术深度：⭐⭐⭐⭐ — 频率分析为设计选择提供了扎实理论基础
- 实验充分度：⭐⭐⭐⭐ — 自建基准 + 多方法对比 + 用户研究
- 实用价值：⭐⭐⭐⭐ — 短视频制作场景有直接应用价值
- 总体推荐：⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] EDEN: Enhanced Diffusion for High-quality Large-motion Video Frame Interpolation](../../CVPR2025/image_generation/eden_enhanced_diffusion_for_high-quality_large-motion_video_frame_interpolation.md)
- [\[ECCV 2024\] Realistic Human Motion Generation with Cross-Diffusion Models](realistic_human_motion_generation_with_cross-diffusion_models.md)
- [\[ECCV 2024\] COIN: Control-Inpainting Diffusion Prior for Human and Camera Motion Estimation](coin_control-inpainting_diffusion_prior_for_human_and_camera_motion_estimation.md)
- [\[ECCV 2024\] M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models](m2d2m_multi-motion_generation_from_text_with_discrete_diffusion_models.md)
- [\[ECCV 2024\] Diffusion-based Image-to-Image Translation by Noise Correction via Prompt Interpolation](diffusion-based_image-to-image_translation_by_noise_correction_via_prompt_interp.md)

</div>

<!-- RELATED:END -->
