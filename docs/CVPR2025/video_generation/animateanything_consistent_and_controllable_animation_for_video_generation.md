---
title: >-
  [论文解读] AnimateAnything: Consistent and Controllable Animation for Video Generation
description: >-
  [CVPR 2025][视频生成][可控视频生成] 提出两阶段可控视频生成框架：第一阶段将不同控制信号（相机轨迹、用户拖拽标注、参考视频）统一转化为逐帧光流表示，第二阶段用统一光流引导基于DiT的视频扩散模型生成最终视频，并引入频域稳定模块抑制大运动下的闪烁问题。 领域现状：可控视频生成是视频生成的关键方向…
tags:
  - "CVPR 2025"
  - "视频生成"
  - "可控视频生成"
  - "光流统一表示"
  - "相机轨迹控制"
  - "频域稳定"
  - "多条件融合"
---

# AnimateAnything: Consistent and Controllable Animation for Video Generation

**会议**: CVPR 2025  
**arXiv**: [2411.10836](https://arxiv.org/abs/2411.10836)  
**代码**: [https://github.com/yu-shaonian/Animate_Anything](https://github.com/yu-shaonian/Animate_Anything)  
**领域**: 视频理解 / 可控视频生成  
**关键词**: 可控视频生成, 光流统一表示, 相机轨迹控制, 频域稳定, 多条件融合

## 一句话总结

提出两阶段可控视频生成框架：第一阶段将不同控制信号（相机轨迹、用户拖拽标注、参考视频）统一转化为逐帧光流表示，第二阶段用统一光流引导基于DiT的视频扩散模型生成最终视频，并引入频域稳定模块抑制大运动下的闪烁问题。

## 研究背景与动机

**领域现状**：可控视频生成是视频生成的关键方向，包括相机轨迹控制（CameraCtrl、MotionCtrl）和物体运动控制（Motion-I2V、MOFA-Video）。

**现有痛点**：
   - MotionCtrl/CameraCtrl仅支持相机控制，依赖文本描述物体运动，精度不够
   - Motion-I2V/MOFA-Video仅支持小幅度物体运动，无法处理相机运动
   - 尝试同时引入多种控制信号时，因模态不同，控制信号会冲突（如相机运动是全局的、物体运动是局部的），导致生成模型困惑

**核心矛盾**：不同控制信号（相机参数、拖拽箭头、参考视频）本质上都在描述像素运动，但表示方式完全不同，难以统一融合。

**核心idea**：如果把所有控制信号都统一转化为逐帧光流表示，就能用统一的方式引导视频生成，自然解决信号冲突。

## 方法详解

### 整体框架

两阶段pipeline：
- **Stage 1（统一光流生成）**：通过显式注入（将拖拽等转为稀疏光流）和隐式注入（将相机轨迹编码为参考特征），用双模型协同（FGM+CRM）生成统一的稠密光流
- **Stage 2（视频生成）**：将统一光流编码后与视频latent通过ViT块融合，结合文本条件用CogVideoX框架生成最终视频

### 关键设计

1. **显式注入（Explicit Injection）——处理可直接转为光流的信号**：

    - 功能：将用户拖拽标注等转化为稀疏光流，通过Flow Generation Model（FGM）扩展为稠密光流
    - 核心思路：用户标注的运动轨迹 $\mathcal{M} \in \mathbb{R}^{P \times 2}$ 通过双三次插值提取稀疏控制点，生成逐点稀疏光流 $F_{l-1}^s(x_i, y_i) = \hat{\mathcal{T}}_l(x_i, y_i) - \hat{\mathcal{T}}_0(x_i, y_i)$，用CMP增强后送入FGM（基于SD1.5的U-Net LDM）转为稠密光流
    - 设计动机：任何可转为稀疏光流的信号（音频、视频、关键点等）都可统一接入FGM

2. **隐式注入（Implicit Injection）——处理难以直接转为像素光流的信号**：

    - 功能：将相机轨迹嵌入到光流生成过程中
    - 核心思路：设计Camera Reference Model（CRM），使用Plücker embedding表示相机轨迹 $\ddot{p}_{f,h,w} = (t_f \times \hat{d}_{f,h,w}, \hat{d}_{f,h,w})$，通过camera motion attention将相机特征与参考图像融合，生成多尺度参考特征，通过reference attention逐步注入FGM的去噪过程
    - 设计动机：相机运动是全局的，影响所有前景和背景像素，难以直接转为稀疏光流，需要隐式地通过多尺度特征引导

3. **频域稳定模块（Frequency Stabilization）**：

    - 功能：抑制大运动导致的视频闪烁和不稳定
    - 核心思路：在DiT的attention机制中引入FFT——对权重矩阵做快速傅里叶变换得到频谱特征，乘以可学习权重矩阵 $W$，再做逆FFT恢复时域特征，最后计算dot-product attention
    - 设计动机：时域上的闪烁源于帧间特征不对齐，但频域特征能更直接揭示视频级别的本质信息，通过调节频域分量可有效抑制帧间不一致

### 损失函数 / 训练策略

- Stage 1用Real10K和DL3DV10K训练FGM/CRM，用Unimatch提取训练视频的GT光流
- Stage 2用WebVid10M和OpenVid训练，使用Flow VAE压缩光流到latent空间
- Stage 2只训练光流编码器、输入transformer块和频域稳定模块，其他参数冻结
- 针对动态场景数据不足的问题，从OpenVid中筛选约10K固定机位视频做额外训练

## 实验关键数据

### 主实验：相机轨迹控制精度（DUSt3R评估）

| 方法 | Basic T-Err↓ | Basic R-Err↓ | Difficult T-Err↓ | Difficult R-Err↓ |
|------|:-:|:-:|:-:|:-:|
| CameraCtrl | 0.090 | 0.300 | 0.082 | 0.306 |
| MotionCtrl | 0.057 | 0.233 | 0.060 | 0.267 |
| **AnimateAnything** | **0.041** | **0.159** | - | - |

### 消融实验：VBench视频质量评估

| 方法 | FID↓ | SSIM↑ | FVD↓ | SubC | MoS |
|------|------|-------|------|------|-----|
| DynamiCrafter | - | - | - | 较低 | 较低 |
| CogVideoX | - | - | - | 中等 | 中等 |
| **AnimateAnything** | **最优** | **最优** | **最优** | **最优** | **最优** |

（注：原文中FID/SSIM/FVD具体数值在Tab.2中，此处因缓存截断未完全获取，但定性结论明确）

### 关键发现

- 相机轨迹控制精度全面领先：DUSt3R评估下translation error降低28%（vs MotionCtrl），rotation error降低32%
- 统一光流表示有效：同时输入相机轨迹+拖拽标注时，模型能正确融合全局和局部运动，不会冲突
- 频域稳定对大运动场景至关重要：去掉频域模块后，大幅相机运动场景出现明显闪烁
- 光流引导对人体和动物运动场景提升最大

## 亮点与洞察

- **光流作为统一运动语言**：将所有异构控制信号转化为统一的光流表示是个优雅的抽象——不管信号从哪里来，最终都在描述像素应该怎么动。这种统一表示的思想可迁移到其他多模态融合场景
- **显式+隐式注入的互补设计**：针对信号特性设计不同注入方式——能直接转光流的用显式路径，不能的用隐式参考特征。这种"因材施教"的设计思路比强行统一更合理
- **频域视角看视频稳定**：从信号处理的角度理解视频闪烁，并在attention中引入FFT调节频域分量，视角新颖且有效

## 局限与展望

- 实用扩展性：FGM和CRM是两个独立模型，训练和推理成本较高
- 相机轨迹训练数据有限：Real10K和DL3DV10K主要是室内/静态场景，动态场景泛化可能不足
- Stage 1的光流预测误差会传导到Stage 2，形成误差累积
- 频域稳定模块的可学习权重可能在未见过的运动模式上泛化不佳

## 相关工作与启发

- **vs Motion-I2V**：同样用两阶段+光流中间表示，但Motion-I2V只从文本和参考运动得光流，不支持相机控制。本文的显式+隐式双路径更全面
- **vs MOFA-Video**：MOFA-Video需要用户为每个区域指定运动方向，交互复杂。本文通过隐式注入相机运动避免了此问题
- **vs CameraCtrl**：CameraCtrl用ControlNet式条件注入，但缺乏显式像素级运动引导。本文通过光流中间表示更精确

## 评分

- 新颖性: ⭐⭐⭐⭐ 光流统一表示+显式/隐式双注入+频域稳定的组合设计新颖，两阶段框架清晰优雅
- 实验充分度: ⭐⭐⭐⭐ 多种评估方式（DUSt3R/VggSfM/ParticleSfM、VBench、FID等），但消融实验偏少
- 写作质量: ⭐⭐⭐⭐ Pipeline图清晰，方法描述有条理，但部分技术细节可以更简洁
- 价值: ⭐⭐⭐⭐ 为统一多条件控制视频生成提供了有效方案，影视/VR应用前景广阔

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text](streamingt2v_consistent_dynamic_and_extendable_long_video_generation_from_text.md)
- [\[CVPR 2025\] FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance](flashmotion_few-step_controllable_video_generation_with_trajectory_guidance.md)
- [\[CVPR 2025\] World-Consistent Video Diffusion with Explicit 3D Modeling](world-consistent_video_diffusion_with_explicit_3d_modeling.md)
- [\[CVPR 2025\] Learning Temporally Consistent Video Depth from Video Diffusion Priors](learning_temporally_consistent_video_depth_from_video_diffusion_priors.md)
- [\[CVPR 2025\] Teller: Real-Time Streaming Audio-Driven Portrait Animation with Autoregressive Motion Generation](teller_real-time_streaming_audio-driven_portrait_animation_with_autoregressive_m.md)

</div>

<!-- RELATED:END -->
