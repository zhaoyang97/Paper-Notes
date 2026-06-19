---
title: >-
  [论文解读] Image Referenced Sketch Colorization Based on Animation Creation Workflow
description: >-
  [CVPR 2025][图像生成][草图上色] 本文模仿真实动画制作流程，提出一种基于扩散模型的图像参考草图上色框架，通过分割交叉注意力（Split Cross-Attention）配合可切换LoRA机制分别处理前景和背景的上色，消除了空间纠缠伪影（spatial entanglement），在4.8M图像上训练后在定性、定量和用户研究中均优于现有方法。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "草图上色"
  - "扩散模型"
  - "分割交叉注意力"
  - "LoRA"
  - "动画制作流程"
---

# Image Referenced Sketch Colorization Based on Animation Creation Workflow

**会议**: CVPR 2025  
**arXiv**: [2502.19937](https://arxiv.org/abs/2502.19937)  
**代码**: [https://github.com/tellurion-kanata/colorizeDiffusion](https://github.com/tellurion-kanata/colorizeDiffusion)  
**领域**: 扩散模型 / 图像生成  
**关键词**: 草图上色, 扩散模型, 分割交叉注意力, LoRA, 动画制作流程

## 一句话总结
本文模仿真实动画制作流程，提出一种基于扩散模型的图像参考草图上色框架，通过分割交叉注意力（Split Cross-Attention）配合可切换LoRA机制分别处理前景和背景的上色，消除了空间纠缠伪影（spatial entanglement），在4.8M图像上训练后在定性、定量和用户研究中均优于现有方法。

## 研究背景与动机

1. **领域现状**：草图上色是动画和数字插画产业中最耗费人力的环节。深度学习方法包括文本引导、用户引导和图像参考三大类。图像参考方法可以无缝集成到现有流程中。

2. **现有痛点**：文本引导方法无法提供精确的颜色和风格参考；用户引导方法仍需手动操作；图像参考方法（如IP-Adapter、ColorizeDiffusion）会产生空间纠缠问题——参考图像和草图的空间不匹配导致多余的手臂、错误的发型等伪影。

3. **核心矛盾**：图像参考编码器提取的局部嵌入（local embeddings）包含过多的空间位置信息，与草图的空间结构产生冲突，导致前景和背景区域互相干扰。

4. **本文目标**：在保持参考颜色忠实度的前提下，消除空间纠缠伪影，实现前景角色和背景的独立上色。

5. **切入角度**：观察动画工作室的实际流程——角色设计师设计参考图→原画师画草稿→上色人员先上前景角色→再上背景→合成最终帧。作者决定用算法模拟这一分离式流程。

6. **核心 idea**：通过分割交叉注意力将前景和背景的LoRA参数分离训练，使扩散模型能在单次前向传播中独立处理两个区域的上色，彻底解耦空间信息干扰。

## 方法详解

### 整体框架
输入：草图$X_s$、参考彩色图像$X_r$、前景mask $X_m$。输出：上色结果$Y$。流程：(1) 使用预训练ViT（OpenCLIP-H）提取参考图像的局部嵌入作为颜色参考；(2) 多层草图编码器注入空间引导信息到U-Net的latent层；(3) 分割交叉注意力层使用前景和背景各自的LoRA权重分别处理对应区域；(4) 推理时可切换不同LoRA模式适配不同场景。

### 关键设计

1. **分割交叉注意力（Split Cross-Attention）**:

    - 功能：在单次前向传播中分别用不同的参数处理前景和背景区域的上色
    - 核心思路：每个交叉注意力层包含两组LoRA权重$W_f^t$（前景）和$W_b^t$（背景），对应各自的Q/K/V投影。利用空间mask$m_s$将像素分为前景（$m_s > ts_s$）和背景两类，前景区域用前景LoRA修改后的权重$\hat{W}_f^t = W^t + W_f^t$计算注意力，背景区域用背景LoRA修改后的权重$\hat{W}_b^t = W^t + W_b^t$计算。两组LoRA独立训练，互不干扰。前景LoRA rank固定为16，背景LoRA rank为$0.5 \times \min(D_q, D_{kv})$。
    - 设计动机：动画图像中前景角色和背景在色彩分布、色块大小、色调、纹理上差异显著。将两者分开处理避免了参考图像中角色的空间位置信息泄露到草图的背景区域（即空间纠缠）。

2. **恢复Transformer（Recovery Transformer）**:

    - 功能：处理背景嵌入以弥合前景和背景参考信息之间的差异
    - 核心思路：背景参考图像经ViT编码后，再通过一个可训练的Transformer $\varphi$ 恢复详细信息，得到$e_b = \varphi(\phi(r_b))$。这一步在前景和背景嵌入拼接作为K/V输入之前进行。
    - 设计动机：直接将前景和背景嵌入分别注入交叉注意力会导致结构保留下降和合成质量退化。恢复Transformer弥补了分割操作带来的信息损失。

3. **可切换LoRA推理模式**:

    - 功能：在不改变模型权重的情况下，通过切换激活的LoRA模块来适配不同场景
    - 核心思路：设计三种推理模式——Vanilla模式（不激活LoRA，适合大多数场景）；Bg2Fig模式（仅激活前景LoRA，适合参考图有复杂背景时）；Fig2Fig模式（激活两组LoRA和恢复Transformer，适合角色到角色的上色，最能消除空间纠缠）。
    - 设计动机：真实应用中草图和参考图的组合非常多样，单一模式无法兼顾所有场景。可切换设计为用户提供了灵活选择。

### 损失函数 / 训练策略
- 预训练阶段：使用WaifuDiffusion初始化VAE和U-Net，动态参考丢弃率从80%降到50%以避免分布偏移，训练6个epoch
- 微调阶段：冻结VAE、U-Net和草图编码器，仅训练恢复Transformer和可切换LoRA，训练3个epoch
- 损失为标准的扩散去噪损失$\mathcal{L}(\theta) = \mathbb{E}[\|\epsilon - \epsilon_\theta(z_t, t, s, \phi(r))\|_2^2]$
- 使用4×H100训练，DeepSpeed ZeRO2加速，AdamW优化器，lr=0.0001

## 实验关键数据

### 主实验
在4.8M训练数据、52K验证数据上评估：

| 方法 | FID↓ | PSNR↑ | MS-SSIM↑ | CLIP相似度↑ |
|------|------|-------|----------|------------|
| IP-Adapter | 38.92 | 28.68 | 0.5478 | 0.8672 |
| InstantStyle | 40.81 | 28.11 | 0.4459 | 0.8042 |
| T2I-Adapter | 41.16 | 28.13 | 0.3243 | 0.7180 |
| AnimeDiffusion | 61.60 | 27.85 | 0.3185 | 0.7319 |
| ColorizeDiffusion | 9.53 | 28.74 | 0.5913 | 0.8775 |
| Yan et al. (GAN) | 27.01 | **29.25** | 0.5253 | 0.7634 |
| **Ours** | **6.83** | 28.91 | **0.6002** | **0.8829** |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| Baseline (无分割CA) | 严重空间纠缠 | 多余手臂、错误衣服 |
| + Split Cross-Attention | 减轻伪影但色彩饱和度下降 | 空间分离有效但信息不足 |
| + Split CA + LoRA | 色彩和细节改善 | 但仍有部分伪影 |
| + Split CA + LoRA + Recovery Trans (Full) | **无伪影，高质量** | 完整pipeline最优 |

### 关键发现
- 分割交叉注意力是消除空间纠缠的核心，恢复Transformer是保证质量的关键补充
- 用户研究中40名参与者在25组图像中更偏好本方法（p<0.01显著性水平）
- Fig2Fig模式在消除空间纠缠方面最强，但Bg2Fig和Vanilla模式在背景生成方面更好
- GAN方法（Yan et al.）PSNR最高是因为其生成能力有限，结果接近平均值，反而在感知失真权衡中占优

## 亮点与洞察
- **工作流启发的设计**：从动画工作室的实际流程中提取出"前景背景分离上色"的核心原则并转化为技术方案，非常自然且有效。这种从领域知识中提炼设计原则的思路值得借鉴。
- **分割交叉注意力的巧妙之处**：不修改预训练权重，仅通过LoRA残差项实现前景/背景分离，保留了预训练模型的全部能力。
- **可切换推理模式**：无需重新训练就能适配不同场景，提高实用性。这种模块化的推理时切换策略可以推广到其他条件生成任务。

## 局限与展望
- 强依赖mask提取质量，mask错误会直接导致上色失败
- 仅在动画风格数据上训练和验证，对真实照片的泛化能力未知
- 目前是单帧处理，作者提出未来将扩展到视频上色
- 可以探索无需mask的自动前景/背景分离方案
- 参考图与草图的语义匹配度对结果质量有影响，极端不匹配时可能效果不佳

## 相关工作与启发
- **vs ColorizeDiffusion**: 同样使用局部嵌入，但ColorizeDiffusion没有分割前景/背景，仍受空间纠缠困扰。本文通过分割CA+LoRA直接解决了这个问题。
- **vs IP-Adapter/T2I-Adapter**: 这些通用适配器方法在草图上色场景中表现很差（FID 38-41 vs 本文6.83），因为它们没有考虑参考图与目标的空间不匹配问题。
- **vs GAN方法**: GAN在PSNR上占优但FID和感知质量差很多，说明扩散模型在复杂纹理和颜色生成方面有质的优势。

## 评分
- 新颖性: ⭐⭐⭐⭐ 分割交叉注意力+可切换LoRA的组合设计新颖，工作流启发的思路自然
- 实验充分度: ⭐⭐⭐⭐⭐ 定性定量对比+消融+用户研究，非常全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图示丰富，动机论述流畅
- 价值: ⭐⭐⭐⭐ 对动画产业的草图上色自动化有直接应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] MangaNinja: Line Art Colorization with Precise Reference Following](manganinja_line_art_colorization_with_precise_reference_following.md)
- [\[CVPR 2025\] AniDoc: Animation Creation Made Easier](anidoc_animation_creation_made_easier.md)
- [\[CVPR 2026\] Towards High-resolution and Disentangled Reference-based Sketch Colorization](../../CVPR2026/image_generation/towards_high-resolution_and_disentangled_reference-based_sketch_colorization.md)
- [\[CVPR 2025\] Free-viewpoint Human Animation with Pose-correlated Reference Selection](free-viewpoint_human_animation_with_pose-correlated_reference_selection.md)
- [\[CVPR 2025\] Consistent and Controllable Image Animation with Motion Diffusion Models](consistent_and_controllable_image_animation_with_motion_diffusion_models.md)

</div>

<!-- RELATED:END -->
