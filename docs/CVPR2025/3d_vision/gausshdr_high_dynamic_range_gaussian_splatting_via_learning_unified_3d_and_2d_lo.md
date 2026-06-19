---
title: >-
  [论文解读] GaussHDR: High Dynamic Range Gaussian Splatting via Learning Unified 3D and 2D Local Tone Mapping
description: >-
  [CVPR 2025][3D视觉][HDR新视角合成] 提出 GaussHDR，通过统一 3D 和 2D 局部色调映射来改进 HDR 高斯溅射，设计残差局部色调映射器和不确定性自适应调制机制，同时提升 HDR 重建稳定性和 LDR 拟合质量，在合成和真实场景上大幅超越现有方法。 1. 领域现状：HDR 新视角合成 (NVS)…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "HDR新视角合成"
  - "高斯溅射"
  - "色调映射"
  - "不确定性学习"
  - "局部色调映射"
---

# GaussHDR: High Dynamic Range Gaussian Splatting via Learning Unified 3D and 2D Local Tone Mapping

**会议**: CVPR 2025  
**arXiv**: [2503.10143](https://arxiv.org/abs/2503.10143)  
**代码**: [https://liujf1226.github.io/GaussHDR](https://liujf1226.github.io/GaussHDR) (项目页)  
**领域**: 3D视觉  
**关键词**: HDR新视角合成, 高斯溅射, 色调映射, 不确定性学习, 局部色调映射

## 一句话总结
提出 GaussHDR，通过统一 3D 和 2D 局部色调映射来改进 HDR 高斯溅射，设计残差局部色调映射器和不确定性自适应调制机制，同时提升 HDR 重建稳定性和 LDR 拟合质量，在合成和真实场景上大幅超越现有方法。

## 研究背景与动机

1. **领域现状**：HDR 新视角合成 (NVS) 利用不同曝光度的多视角 LDR 图像重建 HDR 场景。主流方法将颜色表示从 LDR 扩展到 HDR，再用色调映射器 (tone mapper) 建模相机响应函数 (CRF) 将 HDR 辐照度映射到 LDR。

2. **现有痛点**：存在两种训练范式的困境——(a) **3D 色调映射**（先逐 Gaussian 映射再渲染）：LDR 拟合好但 HDR 重建不稳定，因为 HDR 渲染与 LDR 监督分离，可能导致沿光线的 HDR 和 LDR 分布不一致；(b) **2D 色调映射**（先渲染 HDR 图再映射）：HDR 重建稳定但 LDR 拟合变差，因为 HDR 值域 $[0, +\infty)$ 的射线累积不如 LDR $[0,1]$ 的累积鲁棒。

3. **核心矛盾**：3D 和 2D 色调映射各有优缺点，需要互补。同时，现有方法使用全局色调映射器，对整个场景施加相同映射特性，忽略了不同空间位置的细粒度差异。

4. **本文目标** (i) 如何结合 3D 和 2D 色调映射的优势？(ii) 如何实现局部自适应的色调映射？(iii) 如何自适应平衡不同场景中二者的权重？

5. **切入角度**：为每个 Gaussian 引入上下文特征作为色调映射器的额外输入，实现局部化；通过不确定性学习自适应平衡 3D 和 2D 结果。

6. **核心 idea**：用共享的残差局部色调映射器同时进行 3D 和 2D 局部色调映射，将双路 LDR 结果在 loss 层面通过不确定性自适应融合。

## 方法详解

### 整体框架
基于 3DGS，每个 Gaussian 除了 HDR 辐照度外还额外存储上下文特征 $f \in \mathbb{R}^d$。渲染时同时输出 HDR 图像 $E$、上下文特征图 $F$。通过共享的残差局部色调映射器分别进行 3D 色调映射（逐 Gaussian）和 2D 色调映射（逐像素），得到双路 LDR 结果 $I_{3d}^*$ 和 $I_{2d}^*$。通过不确定性预测器自适应地在 loss 层面融合双路结果。

### 关键设计

1. **残差局部色调映射器（Residual Local Tone Mapper）**:

    - 功能：在全局色调映射基础上增加局部自适应调整
    - 核心思路：将局部色调映射分解为全局映射+残差项：$c^*=g(\ln(et))+\Delta g([\ln(et), f])$，其中 $g$ 是全局色调映射 MLP，$\Delta g$ 是残差 MLP，$f$ 是上下文特征。同一个色调映射器 $g^*$ 同时用于 3D（输入 per-Gaussian 特征 $f_i$）和 2D（输入渲染后的像素特征 $F$）局部映射。训练前 6K 步只用全局映射，之后启用残差项联合优化。
    - 设计动机：直接学局部映射困难且参数量大；残差设计让全局 MLP 提供基础映射，残差 MLP 只需捕捉局部变化，降低学习难度。上下文特征通过渲染自然实现从 3D 到 2D 的一致传递。

2. **上下文特征 & 3D/2D 统一（Context Feature for Unified Local TM）**:

    - 功能：为色调映射提供空间位置感知的局部特征
    - 核心思路：每个 Gaussian 存储一个上下文特征 $f_i \in \mathbb{R}^d$（$d=4$）。3D 色调映射时：$c_i^*=g^*(\ln(e_i t), f_i)$，然后渲染得 $I_{3d}^*$。上下文特征也可像颜色一样通过 alpha-blending 渲染到像素：$F=\mathcal{R}_P(\{f_i\})$，每个像素获得对应特征后做 2D 色调映射：$I_{2d}^*=g^*(\ln(Et), F)$。这种设计受语言嵌入场景表示（如 LangSplat）的启发——渲染后的像素特征和 3D Gaussian 特征在同一语义空间。
    - 设计动机：全局映射器假设所有位置共享相同映射特性，这在有复杂光照的场景中不成立。通过上下文特征引入局部特性，同时借助 Gaussian 渲染的连续性保证 3D 和 2D 特征空间一致。

3. **不确定性自适应融合（Uncertainty-based Joint Learning）**:

    - 功能：自适应平衡 3D 和 2D 色调映射结果的贡献
    - 核心思路：训练一个不确定性 MLP $\rho$，分别预测 3D 和 2D 结果的不确定性图 $U_{3d}$ 和 $U_{2d}$。融合损失为 $\mathcal{L}_{gs}=(U_{2d}^2 \mathcal{L}_{3d}+U_{3d}^2 \mathcal{L}_{2d})/(U_{3d}^2+U_{2d}^2)$——不确定性高的一方权重低。不确定性通过基于 DSSIM 的损失单独优化（梯度停止），与主模型训练解耦。推理时也用不确定性融合双路 LDR：$I_{merge}=(U_{2d}^2 I_{3d}^*+U_{3d}^2 I_{2d}^*)/(U_{3d}^2+U_{2d}^2)$。
    - 设计动机：不同场景中 3D 和 2D 色调映射的最优平衡不同。固定权重无法适应所有场景。不确定性学习让模型自动识别每种映射在每个像素处的可靠程度，实现像素级自适应。

### 损失函数 / 训练策略
总损失 $\mathcal{L}=\mathcal{L}_{gs}+\mathcal{L}_{unc}+\lambda_e \mathcal{L}_e$。$\mathcal{L}_{gs}$ 是不确定性加权的重建损失（DSSIM + L1）；$\mathcal{L}_{unc}$ 是不确定性预测损失；$\mathcal{L}_e$ 是合成场景的单位曝光约束 $\|g(0)-0.73\|_2^2$。$\lambda_d=0.2, \lambda_u=0.5, \lambda_e=0.5$。所有 MLP 仅 1 层隐藏层 64 通道。30K 迭代，前 6K 仅全局映射。单卡 RTX 3090。

## 实验关键数据

### 主实验

| 数据集 | 设置 | 指标 | GaussHDR(3DGS) | HDR-GS(SOTA) | GaussHDR(Scaffold) | 提升 |
|--------|------|------|----------------|-------------|---------------------|------|
| HDR-NeRF Real | LDR-OE PSNR↑ | dB | 35.78 | 35.47 | **36.77** | +1.3 |
| HDR-NeRF Real | LDR-NE PSNR↑ | dB | **33.33** | 31.66 | **33.92** | +2.3 |
| HDR-Plenoxels Real | LDR-OE PSNR↑ | dB | 31.51 | 31.04 | **32.85** | +1.8 |
| HDR-NeRF Synth | LDR-OE PSNR↑ | dB | 42.29 | 41.13 | **43.78** | +2.7 |
| HDR-NeRF Synth | HDR PSNR↑ | dB | 37.62 | 26.98* | **39.02** | +12 |

*注：HDR-GS 在公平比较设置（无 HDR GT 监督）下 HDR 重建严重失败。

### 消融实验

| 配置 | LDR-OE PSNR | LDR-NE PSNR | HDR PSNR | 说明 |
|------|-------------|-------------|----------|------|
| 3D Global TM | 33.94 | 31.88 | 26.11 | 3D全局映射（HDR差） |
| 3D Local TM | 34.41 | 32.56 | 26.78 | 局部化提升LDR |
| 2D Global TM | 32.45 | - | - | 2D全局（LDR差） |
| 3D+2D Global TM | 34.42 | 32.54 | 28.22 | 联合全局 |
| 3D+2D Local TM (无不确定性) | 35.51 | 33.48 | 34.12 | 联合局部大幅提升HDR |
| Full (含不确定性) | **36.77** | **33.92** | **35.47** | 不确定性额外提升 |

### 关键发现
- **3D+2D 联合训练是 HDR 重建的关键**：从 3D-only 的 26.11 PSNR 跃升到联合的 34.12，HDR 提升 8dB
- **局部色调映射 vs 全局**：在 LDR-OE 上从 33.94→34.41（3D）和 34.42→35.51（联合），局部化对所有设置都有正向贡献
- **不确定性融合的必要性**：从 35.51→36.77 (LDR-OE)，额外带来 1.3dB 提升
- **残差设计关键**：HDR 可视化对比显示，有残差设计的局部映射能保留更多 HDR 细节，无残差的版本色调映射结果退化
- 方法对底层表示兼容性好：3DGS 和 Scaffold-GS 作为基础均有效，Scaffold-GS 更优

## 亮点与洞察
- **3D/2D 色调映射互补的洞察**：精确诊断了两种范式各自的失败模式——3D 映射中 HDR-LDR 分布沿光线不一致导致 HDR 陷入局部最优，2D 映射中无穷 HDR 值域的射线累积不如有界 LDR 鲁棒。这一分析本身就很有价值
- **上下文特征复用于 3D 和 2D**：一个特征同时服务两种映射方式，通过渲染自然实现跨域一致性。这种"3D attribute → 2D map"的设计思路可推广到其他需要 3D-2D 一致性的任务
- **不确定性驱动的像素级融合**：比手动调节权重更优雅，且推理时也可用于融合双路结果，实际提升了最终渲染质量
- **残差设计的简洁性**：全局映射打底+残差微调的思路简单有效，适用于任何需要局部化的全局模型

## 局限与展望
- 上下文特征维度仅为 4，可能限制了局部映射的表达能力，更高维特征是否更好值得探索
- 两阶段训练（前 6K 全局→后续局部）的切换点是手动设定的，不同场景可能需要不同策略
- 未考虑动态场景的 HDR 重建
- 不确定性 MLP 与色调映射 MLP 共享输入但梯度分离，是否有更好的联合训练方式？
- 实验主要在小规模室内场景上验证，大规模户外场景的表现未知

## 相关工作与启发
- **vs HDR-GS**: HDR-GS 使用 3D 色调映射，HDR 重建不稳定（公平设置下仅 26.98 PSNR）；GaussHDR 通过 3D+2D 联合局部映射解决了这个核心问题
- **vs HDR-Plenoxels**: 使用 2D 色调映射，LDR 质量受限；GaussHDR 的 3D 分支弥补了这一不足
- **vs HDR-NeRF**: NeRF 基础的方法速度慢，且同样是 3D 全局映射；GaussHDR 在效率和质量上全面超越
- 上下文特征+渲染的框架思想与 LangSplat、LERF 等语义场表示有异曲同工之妙，可考虑结合语义特征做语义感知的 HDR 重建

## 评分
- 新颖性: ⭐⭐⭐⭐ 3D+2D 统一局部色调映射的思路新颖，不确定性融合设计优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 合成+真实多个数据集、完整消融、两种基础表示验证，非常充分
- 写作质量: ⭐⭐⭐⭐ 问题分析清晰透彻，方法描述有序
- 价值: ⭐⭐⭐⭐ HDR NVS 的重要进步，特别是解决了 HDR 重建不稳定这个关键问题

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction](instanthdr_single-forward_gaussian_splatting_for_high_dynamic_range_3d_reconstru.md)
- [\[CVPR 2025\] Event Fields: Capturing Light Fields at High Speed, Resolution, and Dynamic Range](event_fields_capturing_light_fields_at_high_speed_resolution_and_dynamic_range.md)
- [\[CVPR 2025\] Rethinking End-to-End 2D to 3D Scene Segmentation in Gaussian Splatting](rethinking_end-to-end_2d_to_3d_scene_segmentation_in_gaussian_splatting.md)
- [\[CVPR 2025\] UVGS: Reimagining Unstructured 3D Gaussian Splatting using UV Mapping](uvgs_reimagining_unstructured_3d_gaussian_splatting_using_uv_mapping.md)
- [\[ICML 2025\] High Dynamic Range Novel View Synthesis with Single Exposure](../../ICML2025/3d_vision/high_dynamic_range_novel_view_synthesis_with_single_exposure.md)

</div>

<!-- RELATED:END -->
