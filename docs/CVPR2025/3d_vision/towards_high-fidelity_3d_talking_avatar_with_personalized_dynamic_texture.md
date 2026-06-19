---
title: >-
  [论文解读] Towards High-fidelity 3D Talking Avatar with Personalized Dynamic Texture
description: >-
  [CVPR 2025][3D视觉][语音驱动面部动画] 提出TexTalk4D数据集（100分钟扫描级8K动态纹理）和TexTalker框架，首次实现从语音同时生成面部运动和对应的动态纹理（皱纹变化），并通过基于风格锚点(style pivot)的注入策略实现解耦的运动/纹理风格控制。 语音驱动3D面部动画已有大量研究…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "语音驱动面部动画"
  - "动态纹理"
  - "3D说话头像"
  - "扩散模型"
  - "风格解耦控制"
---

# Towards High-fidelity 3D Talking Avatar with Personalized Dynamic Texture

**会议**: CVPR 2025  
**arXiv**: [2503.00495](https://arxiv.org/abs/2503.00495)  
**代码**: [项目主页](https://xuanchenli.github.io/TexTalk/)  
**领域**: 3D视觉  
**关键词**: 语音驱动面部动画, 动态纹理, 3D说话头像, 扩散模型, 风格解耦控制

## 一句话总结

提出TexTalk4D数据集（100分钟扫描级8K动态纹理）和TexTalker框架，首次实现从语音同时生成面部运动和对应的动态纹理（皱纹变化），并通过基于风格锚点(style pivot)的注入策略实现解耦的运动/纹理风格控制。

## 研究背景与动机

语音驱动3D面部动画已有大量研究，但大多只关注几何运动（网格/顶点位移），忽视了动态纹理的重要性。说话时面部的皱纹压缩与形成反映了肌肉力量，缺少动态纹理会显著降低渲染真实感甚至产生恐怖谷效应。

面临两大挑战：(1) **缺乏高质量动态纹理数据集** — 现有4D数据集要么从单目视频估计（精度低、时间不一致），要么通过捕捉系统获取但缺乏纹理（如VOCASET、MEAD-3D）。唯一包含动态纹理的Multiface仅13个主体，分辨率1024且多样性不足。(2) **几何与纹理的联合生成未被研究** — 运动是3D顶点偏移而纹理是2D图像，表示空间差异大，难以学习跨域关联。

## 方法详解

### 整体框架

TexTalker分三阶段：(1) 将面部运动和纹理变化统一为UV空间的motion map和wrinkle map，通过量化自编码器学习紧凑的面部动画原语(primitives)；(2) 训练Transformer-based潜在扩散模型(LDM)在音频引导下联合生成运动和皱纹的潜在编码；(3) 使用style pivot实现解耦的运动和皱纹风格控制。

### 关键设计1：面部动画原语学习（统一表示）

- **功能**: 将异质的3D几何运动和2D纹理变化统一到相似的潜在空间
- **核心思路**: 将顶点偏移$\mathbf{m}_t$映射到UV空间得到motion map $\mathbf{f}_t$，将纹理变化表示为与中性表情的比值得到wrinkle map $\mathbf{w}_t$。分别训练两个VQGAN编码器$\mathcal{E}_f, \mathcal{E}_w$，将它们压缩到$16 \times 16 \times 16$的离散潜在空间（码本大小1024）
- **设计动机**: 直接在UV map上生成成本过高，且运动和纹理表示差异大难以学习关联。通过统一到UV空间+量化编码，既压缩了维度又使两种模态在同一表示框架下，便于后续联合建模

### 关键设计2：运动-皱纹联合潜在扩散模型

- **功能**: 在音频引导下联合生成时间一致的面部运动和纹理变化序列
- **核心思路**: 将运动和皱纹的潜在编码拼接为$\mathbf{X}^0 = [\mathbf{z}_f, \mathbf{z}_w]$，使用8层Transformer decoder作为去噪网络，以HuBERT提取的音频特征为条件。采用滑动窗口策略（$T_w=90, T_p=10$）学习长期依赖，使用对齐掩码确保运动-皱纹特征仅关联同帧音频
- **设计动机**: 几何和纹理本质上强相关（皱纹反映肌肉运动），联合生成能利用互补信息促进彼此质量。实验证明联合学习在两个指标上都优于分离学习

### 关键设计3：基于风格锚点的解耦风格注入

- **功能**: 实现说话风格和皱纹风格的独立控制
- **核心思路**: 利用学到的码本空间的聚类性质——同一主体的潜在编码自然聚类，聚类中心（style pivot $\mathbf{p} = \frac{1}{T}\sum_{t=1}^T \mathbf{z}_t$）捕获风格特征。LDM改为预测与锚点的偏移$\Delta\mathbf{z} = \mathbf{z} - \mathbf{p}$（与音素相关但与风格无关），推理时通过添加不同主体的$\mathbf{p}_{f,i}$和$\mathbf{p}_{w,j}$实现任意组合
- **设计动机**: one-hot风格嵌入无法捕获复杂风格且泛化差；基于示例的方法需要额外网络。style pivot直接来自学到的空间，简洁且具表达力

### 损失函数

- 动画原语: $\mathcal{L}_{\text{latent}} = \mathcal{L}_{\text{rec}} + \eta_{\text{per}}\mathcal{L}_{\text{per}} + \eta_{\text{adv}}\mathcal{L}_{\text{adv}} + \eta_{\text{code}}\mathcal{L}_{\text{code}}$
- LDM训练: $\mathcal{L}_{\mathcal{F}} = \|\hat{\mathbf{X}}^0 - \mathbf{X}^0\|^2$（简单MSE损失）

## 实验关键数据

### 主实验：面部运动质量对比（TexTalk4D-Test-A）

| 方法 | LVE↓ ($10^{-2}$mm) | MVE↓ ($10^{-2}$mm) | FDD↓ ($10^{-3}$mm) |
|------|-----|-----|-----|
| FaceFormer | 1.80 | 2.94 | 1.68 |
| CodeTalker | 1.83 | 2.80 | 1.38 |
| FaceDiffuser | 1.53 | 2.38 | 1.64 |
| **TexTalker** | **1.49** | **2.34** | **1.20** |

### 纹理质量对比

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | Realism↑ | Consistency↑ |
|------|-------|-------|--------|----------|-------------|
| 静态纹理 | 39.79 | 0.967 | 0.0146 | 3.10 | 2.65 |
| Li et al. | 42.34 | 0.981 | 0.0187 | 3.91 | 3.78 |
| **Ours** | **44.13** | **0.985** | **0.0101** | **4.13** | **3.97** |

### 消融实验

| 变体 | LVE↓ | MVE↓ | PSNR↑ | SSIM↑ |
|------|------|------|-------|-------|
| w/o Wrinkle (仅运动) | 1.73 | 2.76 | - | - |
| w/o Motion (仅皱纹) | - | - | 43.87 | 0.985 |
| Joint Codebook | 1.71 | 2.68 | 43.45 | 0.981 |
| w/o Pivot (one-hot) | 1.70 | 2.60 | 43.61 | 0.984 |
| **Full** | **1.49** | **2.34** | **44.13** | **0.985** |

### 关键发现

- 联合学习比分离学习在运动和纹理质量上都更优，验证了跨模态互补信息的价值
- 分离码本优于联合码本，说明运动和纹理虽相关但最优表示空间不同
- style pivot注入显著优于one-hot风格嵌入（LVE: 1.49 vs 1.70）

## 亮点与洞察

1. **首个音频驱动带动态纹理的3D说话头生成方法**: 填补了领域空白，揭示动态纹理对渲染真实感的关键作用
2. **TexTalk4D数据集**: 100主体、8K分辨率、扫描级精度的标杆数据集
3. **Style pivot的优雅性**: 直接从学到的空间提取风格表示，无需额外网络，且实现解耦控制

## 局限与展望

- 当前动态纹理在512分辨率下生成，需要超分辨率网络上采样到8K，细节可能有损
- 数据集仅包含亚洲青年面孔，多样性有限
- 未探索表情驱动的纹理生成（仅语音驱动）

## 相关工作与启发

- UV空间的统一表示思路可推广到其他需要联合生成不同模态的任务
- 基于潜在空间聚类性质提取风格的思路简洁有效，可用于其他风格化生成任务

## 评分

⭐⭐⭐⭐ — 提出了新任务和高质量数据集，方法设计优雅（特别是style pivot），实验全面。数据集的开源将推动领域发展。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Ctrl-D: Controllable Dynamic 3D Scene Editing with Personalized 2D Diffusion](ctrl-d_controllable_dynamic_3d_scene_editing_with_personalized_2d_diffusion.md)
- [\[CVPR 2025\] MARVEL-40M+: Multi-Level Visual Elaboration for High-Fidelity Text-to-3D Content Creation](marvel-40m_multi-level_visual_elaboration_for_high-fidelity_text-to-3d_content_c.md)
- [\[CVPR 2025\] HRAvatar: High-Quality and Relightable Gaussian Head Avatar](hravatar_high-quality_and_relightable_gaussian_head_avatar.md)
- [\[ICCV 2025\] HiNeuS: High-fidelity Neural Surface Mitigating Low-texture and Reflective Ambiguity](../../ICCV2025/3d_vision/hineus_high-fidelity_neural_surface_mitigating_low-texture_and_reflective_ambigu.md)
- [\[CVPR 2025\] PERSE: Personalized 3D Generative Avatars from A Single Portrait](perse_personalized_3d_generative_avatars_from_a_single_portrait.md)

</div>

<!-- RELATED:END -->
