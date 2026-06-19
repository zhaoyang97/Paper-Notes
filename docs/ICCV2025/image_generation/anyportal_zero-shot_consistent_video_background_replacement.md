---
title: >-
  [论文解读] AnyPortal: Zero-Shot Consistent Video Background Replacement
description: >-
  [ICCV 2025][图像生成][视频背景替换] AnyPortal 提出了一个零样本、免训练的视频背景替换框架，通过协同利用 IC-Light 的重光照能力和视频扩散模型（CogVideoX）的时序先验，配合新提出的 Refinement Projection Algorithm (RPA) 实现像素级前景保持，在单张 24GB GPU 上即可高效运行。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "视频背景替换"
  - "前景重光照"
  - "零样本"
  - "扩散模型"
  - "时序一致性"
---

# AnyPortal: Zero-Shot Consistent Video Background Replacement

**会议**: ICCV 2025  
**arXiv**: [2509.07472](https://arxiv.org/abs/2509.07472)  
**代码**: 待发布  
**领域**: 扩散模型 / 视频编辑  
**关键词**: 视频背景替换, 前景重光照, 零样本, 扩散模型, 时序一致性

## 一句话总结
AnyPortal 提出了一个零样本、免训练的视频背景替换框架，通过协同利用 IC-Light 的重光照能力和视频扩散模型（CogVideoX）的时序先验，配合新提出的 Refinement Projection Algorithm (RPA) 实现像素级前景保持，在单张 24GB GPU 上即可高效运行。

## 研究背景与动机

**领域现状**：视频背景替换（"虚拟传送"）在影视行业依赖绿幕和复杂后期流程，成本高、门槛高。AIGC 快速发展使得图像级别的背景替换（如 IC-Light）已经效果出色，但视频级别仍然困难。

**现有痛点**：
   - IC-Light 仅支持图像，逐帧处理会导致严重的帧间不一致
   - 现有视频扩散模型（CogVideoX、OpenSora）可控性有限，只能提供粗粒度控制（边缘、姿态），缺乏像素级精度
   - 将视频模型适配到背景替换需要大量配对视频数据训练，但这类数据极度稀缺

**核心矛盾**：IC-Light 有良好的光照先验但缺乏视频时序建模；视频扩散模型有时序先验但无法精确保留前景细节。简单组合两者会面临前景一致性问题——已有的 DDIM inversion 和 latent manipulation 方案在视频模型的高度压缩 3D 潜空间中表现不佳。

**本文目标** 在不需要任何训练的前提下，实现视频背景替换 + 前景自然重光照 + 帧间时序一致 + 前景像素级保持。

**切入角度**：预训练的大扩散模型已经蕴含了丰富的先验知识，关键在于如何在零样本设定下协同利用它们。

**核心 idea**：通过三阶段流水线（背景生成→光照协调→一致性增强）和新提出的 RPA 算法，在不做任何训练的情况下实现高质量视频背景替换。

## 方法详解

### 整体框架
AnyPortal 是一个三阶段流水线：输入为前景视频 $\mathbf{I}$ 和描述目标背景的文本提示 $p$（或背景图），输出为前景保持、光照协调、时序一致的替换结果视频 $\mathbf{I}'$。所有模型均冻结，不做任何训练或推理时优化。

### 关键设计

1. **Stage 1: 运动感知背景生成 (Background Generation)**

    - 功能：生成与输入视频相机运动一致的纯背景视频 $\mathbf{I}_b$
    - 核心思路：先用 IC-Light $\delta_p$ 处理第一帧得到 $I_1'$，然后利用 Diffusion-As-Shader (DAS) 框架以 $I_1'$ 为首帧、以原视频的 3D 点运动为引导生成初步视频 $\bar{\mathbf{I}}_b$，最后用 ProPainter 去除前景物体得到纯背景 $\mathbf{I}_b$
    - 设计动机：需要背景的相机运动与输入视频匹配，但 DAS 生成的前景可能与原视频不同，因此需要 inpainting 去除

2. **Stage 2: 两步光照协调 (Two-Step Light Harmonization)**

    - 功能：将前景与新背景组合并实现自然的光照融合
    - 核心思路：先用图像引导模型 $\delta_I(I_f, I_b)$ 得到基础融合结果，再用 SDEdit 思路将其加噪后用文本引导模型 $\delta_p$ 去噪 $T_0$ 步来增强光照效果。同时在两个 IC-Light 模型中引入跨帧注意力（cross-frame attention），让所有帧聚合第一帧的 key/value 以维持风格一致性
    - 设计动机：单独用 $\delta_I$ 光照效果不够强（缺少逆光等效果），单独用 $\delta_p$ 背景不一致且缺少图像引导。两步结合取长补短，$T_0$ 可调节光照强度

3. **Stage 3: 一致性增强 + Refinement Projection Algorithm (RPA)**

    - 功能：用视频扩散模型增强帧间时序一致性，同时用 RPA 保持前景像素级细节
    - 核心思路：
        - 用 SDEdit 对 $\mathbf{I}_L$ 加噪 $T_1$ 步后用视频模型 $\epsilon_\theta$ 去噪（附带 edge ControlNet 保持粗结构）
        - RPA 在每个去噪步骤中：① 将 $x_0^t$ 解码到像素域；② 分离高低频，用原视频的高频替换前景高频、保留去噪结果的低频（光照），背景区域用 inpaint 结果；③ 将修改后的 $\tilde{\mathbf{I}}_0^t$ 重新编码回潜空间
        - **关键创新**：为避免 VAE 编解码的重建误差和随机性累积导致背景模糊，RPA 计算确定性采样方向 $\hat{\epsilon} = (x_0^t - \mu) / \sigma$，使得在未修改区域 $\hat{x}_0^t$ 完全等于 $x_0^t$（零误差投影）
    - 设计动机：视频模型的 3D 潜空间高度压缩，传统的像素域操作后重编码会引入误差；RPA 的零误差投影特性确保只有前景细节被改变，背景区域完全不受影响

### 损失函数 / 训练策略
完全免训练——所有模型冻结，无需任何损失函数或优化。这是本文的核心优势之一。

## 实验关键数据

### 主实验
在 30 个样本的测试集上与零样本基线对比：

| 指标 | IC-Light | TokenFlow | DAS | **AnyPortal** |
|------|----------|-----------|-----|---------------|
| Fram-Acc ↑ | **0.983** | 0.541 | 0.937 | 0.973 |
| Tem-Con ↑ | 0.945 | 0.981 | 0.986 | **0.993** |
| ID-Psrv ↓ | 0.578 | 0.632 | 0.364 | **0.313** |
| Mtn-Psrv ↑ | 0.844 | 0.985 | 0.878 | **0.987** |
| User-Pmt | 1.11% | 1.11% | 29.72% | **68.06%** |
| User-Tem | 0.56% | 5.56% | 28.61% | **65.28%** |

### 消融实验

| 配置 | Fram-Acc ↑ | Tem-Con ↑ | ID-Psrv ↓ | Mtn-Psrv ↑ |
|------|-----------|-----------|-----------|-----------|
| Full model | 0.973 | **0.993** | **0.313** | **0.987** |
| w/o $\delta_p$ | 0.966 | 0.989 | 0.329 | 0.987 |
| w/o Cst-Enh | 0.970 | 0.961 | 0.353 | 0.973 |
| w/o RPA | 0.970 | 0.987 | 0.371 | 0.984 |

### 关键发现
- **一致性增强阶段贡献最大**：去掉后 Tem-Con 从 0.993 降到 0.961，视频扩散模型的时序先验至关重要
- **RPA 对前景保持至关重要**：去掉后 ID-Psrv 从 0.313 升到 0.371，且视觉上背景会变模糊
- 在所有用户偏好指标中 AnyPortal 均以 60%+ 的选择率大幅领先
- 单张 4090 GPU，每段视频（49帧 480×720）推理约 12 分钟

## 亮点与洞察
- **RPA 的零误差投影**是最精妙的设计：通过计算 $\hat{\epsilon} = (x_0^t - \mu)/\sigma$ 作为确定性采样方向，避免了 VAE 编解码误差累积。这个思路可以迁移到任何需要在 3D 潜空间中做像素级操作的场景
- **模块化设计**使框架可以随时替换为最新的预训练模型，天然兼容 AIGC 技术进步
- **两步 IC-Light 协调**巧妙利用 $\delta_I$ 提供空间一致性、$\delta_p$ 增强光照效果，跨帧注意力附加风格一致性

## 局限与展望
- 低质量/低分辨率输入会导致高频细节传递不佳（头发区域模糊）
- 前景-背景边界不清晰时主体周围出现模糊区域
- 快速运动场景下扩散模型仍会产生伪影
- 推理时间约 12 分钟/视频，离实时有差距
- 固定规格 480×720、49帧（受 CogVideoX 限制）

## 相关工作与启发
- **vs IC-Light (逐帧)**：帧间不一致严重且改变前景颜色，AnyPortal 通过视频模型+RPA 大幅改善
- **vs TokenFlow**：编辑能力有限、前景控制不足，AnyPortal 在所有指标全面领先
- **vs DAS**：无法维持前景运动动态，AnyPortal 的 RPA 提供像素级保障
- **vs RelightVid**：需微调 AnimateDiff，AnyPortal 完全免训练且可用更强的 CogVideoX

## 评分
- 新颖性: ⭐⭐⭐⭐ RPA 的零误差投影思路新颖，但整体框架是已有模块的组合
- 实验充分度: ⭐⭐⭐⭐ 定量+用户研究+完整消融，但测试集仅 30 个样本偏小
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，图示直观，三阶段层层递进
- 价值: ⭐⭐⭐⭐ 首个免训练视频背景替换框架，实用性强，模块化设计有前瞻性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Early Timestep Zero-Shot Candidate Selection for Instruction-Guided Image Editing](early_timestep_zero-shot_candidate_selection_for_instruction-guided_image_editin.md)
- [\[ICCV 2025\] BVINet: Unlocking Blind Video Inpainting with Zero Annotations](bvinet_unlocking_blind_video_inpainting_with_zero_annotations.md)
- [\[NeurIPS 2025\] Semantic Surgery: Zero-Shot Concept Erasure in Diffusion Models](../../NeurIPS2025/image_generation/semantic_surgery_zero-shot_concept_erasure_in_diffusion_models.md)
- [\[ICCV 2025\] Bridging the Skeleton-Text Modality Gap: Diffusion-Powered Modality Alignment for Zero-shot Skeleton-based Action Recognition](bridging_the_skeleton_text_modality_gap_diffusion_powered_modality_alignment_for.md)
- [\[CVPR 2025\] Pursuing Temporal-Consistent Video Virtual Try-On via Dynamic Pose Interaction](../../CVPR2025/image_generation/pursuing_temporal-consistent_video_virtual_try-on_via_dynamic_pose_interaction.md)

</div>

<!-- RELATED:END -->
