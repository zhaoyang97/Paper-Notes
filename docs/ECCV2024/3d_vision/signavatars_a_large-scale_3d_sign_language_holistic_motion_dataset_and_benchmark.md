---
title: >-
  [论文解读] SignAvatars: A Large-scale 3D Sign Language Holistic Motion Dataset and Benchmark
description: >-
  [ECCV 2024][3D视觉][sign language] 提出 SignAvatars，首个大规模多提示（HamNoSys/语言/单词）3D 手语全身运动数据集（70K 视频、8.34M 帧、153 名手语者），设计了带生物力学约束的自动 3D 标注流水线，并提出基于 VQ-VAE 的 SignVAE 模型作为 3D 手语生产（SLP）的首个 benchmark baseline。
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "sign language"
  - "3D holistic motion"
  - "dataset"
  - "motion generation"
  - "SMPL-X"
---

# SignAvatars: A Large-scale 3D Sign Language Holistic Motion Dataset and Benchmark

**会议**: ECCV 2024  
**arXiv**: [2310.20436](https://arxiv.org/abs/2310.20436)  
**代码**: [项目主页](https://signavatars.github.io/)  
**领域**: LLM评测  
**关键词**: sign language, 3D holistic motion, dataset, motion generation, SMPL-X

## 一句话总结

提出 SignAvatars，首个大规模多提示（HamNoSys/语言/单词）3D 手语全身运动数据集（70K 视频、8.34M 帧、153 名手语者），设计了带生物力学约束的自动 3D 标注流水线，并提出基于 VQ-VAE 的 SignVAE 模型作为 3D 手语生产（SLP）的首个 benchmark baseline。

## 研究背景与动机

**领域现状**: 全球有 4.66 亿聋人和听力障碍人群，使用超过 300 种手语进行交流。虽然 NLP 和计算机视觉的研究已非常成熟，但手语领域的 3D 数字化研究严重滞后。

**现有痛点**:
   - 现有手语数据集（Phoenix、How2Sign 等）仅限于 2D 视频或 2D 关键点标注，存在深度模糊问题（不同手势在 2D 下可能看起来相同）
   - 唯一的 3D 手语数据集 SGNify 仅有 50 个视频，且只支持孤立手势（每视频一个手势）、仅有 HamNoSys 一种标注
   - 3D 头像标注是完全由手语专家手工完成的劳动密集型工作，结果往往不自然

**核心矛盾**: 手语的 3D 数字化需要精确的全身（尤其是手部）mesh 标注，但现有自动方法难以处理手语场景中频繁的手部遮挡和双手交互，而手工标注成本极高且效果不自然。

**本文目标** (1) 构建首个大规模、多提示的 3D 全身手语运动数据集；(2) 设计能处理复杂手部交互的自动标注流水线；(3) 建立 3D 手语生产（SLP）的首个 benchmark。

**切入角度**: 从公开数据集和在线视频中大规模收集手语视频，设计结合时序平滑和生物力学约束的自动 SMPL-X 拟合流水线获取 3D 标注，并基于 VQ-VAE 架构提出首个可从多种提示生成 3D 全身手语动作的基线模型。

**核心 idea**: 通过自动标注流水线将海量 2D 手语视频提升为精确 3D 全身 mesh 标注，构建首个支持 HamNoSys/语言/单词多种提示的大规模 3D 手语数据集和生产 benchmark。

## 方法详解

### 整体框架

系统由三个核心部分组成：

1. **数据收集与整理**: 从多个公开数据源汇集 70K 视频（ASL 34K、GSL 8.3K、HamNoSys 5.8K、Word 21K）
2. **自动 3D 标注流水线**: 层次化初始化 → 多目标优化（时序平滑+生物力学约束）→ 输出 SMPL-X 参数
3. **3D SLP Benchmark**: 基于 VQ-VAE 的 SignVAE 模型，从文本/HamNoSys/单词提示生成 3D 全身运动

### SignAvatars 数据集

**数据规模与组成**:

| 子集 | 视频数 | 帧数 | 类型 | 手语者数 |
|------|--------|------|------|---------|
| Word | 21K | 1.39M | 单词级 | 119 |
| PJM | 2.6K | 0.21M | HamNoSys | 2 |
| DGS | 1.9K | 0.12M | HamNoSys | 8 |
| GRSL | 0.8K | 0.06M | HamNoSys | 2 |
| LSF | 0.4K | 0.03M | HamNoSys | 2 |
| ASL | 34K | 5.7M | 语言句子级 | 11 |
| GSL | 8.3K | 0.83M | 语言+gloss | 9 |
| **总计** | **70K** | **8.34M** | **多种** | **153** |

**表示形式**: 采用 SMPL-X 参数化模型，每帧运动状态 $M_t = (\theta_t^b, \theta_t^h, \theta_t^f, \phi, \tilde{\beta})$，其中：
- $\theta_t^b \in \mathbb{R}^{23 \times 6}$: 身体姿态（含全局朝向）
- $\theta_t^h \in \mathbb{R}^{30 \times 6}$: 手部姿态
- $\theta_t^f \in \mathbb{R}^{6}$: 下巴姿态
- $\phi$: 面部表情
- $\tilde{\beta}$: 优化后的一致体形参数

同时提供 MANO 手部子集：$M_t^h = (\theta_t^h, \tilde{\beta})$

### 关键设计

1. **自动标注流水线（多目标优化）**: 通过最小化包含多项正则化的目标函数进行全身 SMPL-X 拟合：

    $E(\theta, \beta, \phi) = \lambda_J L_J + \lambda_\theta L_\theta + \lambda_\alpha L_\alpha + \lambda_\beta L_\beta + \lambda_s L_{\text{smooth}} + \lambda_a L_{\text{angle}} + L_{\text{bio}}$

   各项含义：
    - $L_J$: 2D 重投影关节损失（SMPL-X 关节投影到图像后与 ViTPose + MediaPipe 预测的 2D 关键点对齐）
    - $L_\theta$: 姿态先验项（来自 SMPLify-X）
    - $L_\alpha$: 弯曲惩罚（仅约束肘和膝的极端弯曲）
    - $L_\beta$: 体形先验
    - $L_{\text{smooth}}$: 时序平滑正则化，约束身体和手部帧间变化连续
    - $L_{\text{angle}}$: 关节角度限制先验
    - $L_{\text{bio}}$: 生物力学约束

   **设计动机**: 标准的 2D→3D 重建方法（如 OSX）在手语场景下因频繁的手部遮挡和交互表现不佳，需要时序信息和生物力学约束来恢复合理手势。

2. **生物力学手部约束**: 手部姿态估计极具挑战性（快速运动、交互、遮挡），设计三重约束：

    - $L_{\text{bl}}$: 骨骼长度约束（每根手指骨长度需在合理范围内）
    - $L_{\text{palm}}$: 掌部区域优化（约束四根掌骨的曲率和角距）
    - $L_{\text{ja}}$: 关节角度先验（在屈伸-外展平面上约束关节角在凸包内）
   
   总生物力学损失：$L_{\text{bio}} = \lambda_{bl} L_{\text{bl}} + \lambda_{palm} L_{\text{palm}} + \lambda_{ja} L_{\text{ja}}$

   **核心思路**: 利用人手的解剖学约束消除不可能的手部姿态，即使在严重遮挡下也能保持生理合理性。

3. **层次化初始化**: 初始化是优化成功的关键。采用多源融合策略：

    - 3D 初始化：OSX + ACR + PARE 融合（提升遮挡/截断下的稳定性）
    - 2D 关键点：基于 ViTPose 的全身姿态估计 + MediaPipe 融合（置信度引导过滤）
    - 分五阶段优化，前三阶段优化体形参数获取均值体形，后两阶段冻结体形

4. **SignVAE（3D SLP 基线模型）**: 两阶段 VQ-VAE 架构：

    - **第一阶段 - 双码本训练**:
        - Motion VQ-VAE：将运动序列 $M_{1:T}$ 编码→量化为码本索引→解码重建，下采样率 $w=4$
        - Linguistic VQ-VAE (PLFG)：将文本/HamNoSys 等提示通过 CLIP 或自定义嵌入编码→量化为语言码本索引→解码重建
    - **第二阶段 - 自回归生成**: 融合语言特征嵌入和语言码本索引向量，用自回归模型根据语义码本索引预测运动码本索引序列
   
   **设计动机**: 通过在离散码本层面建立语义-运动的对应关系，比直接从高层 CLIP 特征回归效果更好。

### 损失函数 / 训练策略

**标注流水线损失**: 多目标加权优化（公式如上），五阶段渐进优化

**Motion VQ-VAE 训练**:
$$L_{m\text{-}vq} = L_{\text{recon}}(M_{1:T}, \hat{M}_{1:T}) + \|sg[F^m_{1:T}] - \hat{F^m_{1:T}}\|_2 + \beta \|F^m_{1:T} - sg[\hat{F^m_{1:T}}]\|_2$$

**SLP 自回归训练**: 交叉熵损失
$$L_{\text{SLP}} = \mathbb{E}_{X \sim p(X)}[-\log p(X|c)]$$

## 实验关键数据

### 主实验

**3D 全身重建精度 (EHF 数据集)**:

| 方法 | PA-MPVPE 全身 | PA-MPVPE 手部 | PA-MPJPE 身体 | PA-MPJPE 手部 |
|------|-------------|-------------|-------------|-------------|
| SMPLify-X | 65.3 | 75.4 | 62.6 | 12.9 |
| PyMAF-X | 50.2 | 10.2 | 52.8 | 10.3 |
| Motion-X w/GT 3Dkpt | 19.7 | - | 23.9 | - |
| **Ours (w/ bio)** | **12.9** | **4.7** | **15.6** | **5.8** |

**3D SLP HamNoSys 子集**:

| 方法 | DTW-MJE Top1↑ | DTW-MJE Top3↑ | DTW-MJE Top5↑ |
|------|--------------|--------------|--------------|
| Ham2Pose (2D only) | 0.092 | 0.197 | 0.354 |
| Ham2Pose-3d | 0.253 | 0.369 | 0.511 |
| **SignVAE (Ours)** | **0.516** | **0.694** | **0.786** |

### 消融实验

**PLFG 模块消融 (HamNoSys holistic)**:

| 方法 | R-Precision Top1↑ | R-Precision Top3↑ | MM-dist↓ |
|------|-------------------|-------------------|----------|
| Ham2Pose-3d | 0.291 | 0.386 | 3.875 |
| SignDiffuse (MDM改) | 0.285 | 0.415 | 3.866 |
| SignVAE (Base, 无PLFG) | 0.385 | 0.613 | 3.056 |
| **SignVAE (Ours, 有PLFG)** | **0.429** | **0.657** | **2.651** |

### 关键发现

1. **标注质量大幅超越 SoTA**: 在 EHF 数据集上，PA-MPJPE 身体误差 15.6mm，相比 Motion-X (w/ GT 3Dkpt) 的 23.9mm 改善 ~35%；手部 PA-MPVPE 仅 4.7mm。
2. **生物力学约束的关键作用**: 加入生物力学约束后，手部 PA-MPVPE 从 5.4 降至 4.7，MPVPE 手部从 12.5 降至 9.7，消除了大量不合理手势。
3. **离散码本交互优于端到端回归**: SignVAE 在 R-Precision Top1 上比 SignDiffuse (基于 MDM) 高出 50%（0.429 vs 0.285），说明离散表示层面的语义-运动对齐更有效。
4. **PLFG 模块显著提升**: 相比直接使用 CLIP 特征的 Base 版本，加入 PLFG 后 R-Precision Top1 从 0.385 提升至 0.429，MM-dist 从 3.056 降至 2.651。
5. **Word 级提示效果最好**: 在三种提示类型中，word 级的 FID 最低（0.756）、R-Precision 最高（0.475 Top1），spoken language 最具挑战性。

## 亮点与洞察

1. **填补领域空白**: 在 3D 手语领域，这是首个同时提供大规模数据集、自动标注方法和生产 benchmark 的完整工作，具有显著的社会价值。
2. **生物力学约束的巧妙应用**: 将手部解剖学知识（骨骼长度、掌部曲率、关节角度凸包）作为优化约束，是处理手部遮挡/交互的有效方案，可推广到其他手部重建任务。
3. **多源初始化融合策略**: OSX + ACR + PARE + ViTPose + MediaPipe 的多模型融合+置信度过滤，是提升鲁棒性的实用工程方案。
4. **VQ-VAE 双码本设计**: 通过在离散码本层面建立语义-运动对应关系（而非在连续特征空间），利用了 VQ-VAE 天然的离散结构来增强跨模态关联，这个思路在其他 text-to-motion 任务中也值得借鉴。
5. **多提示支持**: 统一框架同时支持 HamNoSys（语言学符号）、spoken language（自然语言）、word（单词）三种提示，覆盖了手语研究的不同需求场景。

## 局限与展望

1. **缺乏成熟的 3D 回译评估方法**: 目前没有通用的 3D 手语回译方法，评估指标（DTW-MJE、FID 等）可能无法完全反映生成质量。
2. **Spoken language 生成仍有较大差距**: 语言级 SLP 的 FID 为 4.359，远高于 word 级的 0.756，说明从自然语言到手语动作的映射仍非常困难。
3. **下肢未纳入评估**: 手语主要涉及上半身，当前评估忽略了下肢运动，但在实际应用中站姿和重心也影响自然度。
4. **自动标注仍有误差**: 虽然大幅优于现有方法，但在极端遮挡和快速运动下仍可能产生不准确标注。
5. **未来方向**: 结合 3D SLT（手语翻译）和 SLP 构建多模态手语框架，以及开发 AR/VR 中的大规模手语运动模型。

## 相关工作与启发

- **SGNify**: 唯一前序 3D 手语数据集（仅 50 视频），SignAvatars 在规模上实现 1400x 的提升
- **Ham2Pose**: HamNoSys→2D 姿态的代表方法，被改造为 3D 版本作为 baseline
- **MDM (Motion Diffuse Model)**: 通用运动扩散模型，被改造为 SignDiffuse baseline
- **SMPL-X / MANO**: 参数化人体/手部模型，是 3D 手语表示的基础
- **启发**: 该数据集为聋哑人数字化通信提供了重要基础设施，VQ-VAE 的双码本跨模态对齐思路值得在更多多模态生成任务中探索

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首个大规模 3D 手语全身运动数据集，填补重要空白
- **实验充分度**: ⭐⭐⭐⭐ 标注质量、SLP 性能、消融实验均有覆盖，但受限于评估指标的成熟度
- **写作质量**: ⭐⭐⭐⭐ 问题动机清晰，方法描述详实，但数学符号繁多，部分段落较密
- **价值**: ⭐⭐⭐⭐⭐ 兼具技术价值和社会价值，为聋哑人社区的数字化通信提供关键基础设施

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Text-Driven 3D Hand Motion Generation from Sign Language Data](../../CVPR2026/3d_vision/text-driven_3d_hand_motion_generation_from_sign_language_data.md)
- [\[ECCV 2024\] SegPoint: Segment Any Point Cloud via Large Language Model](segpoint_segment_any_point_cloud_via_large_language_model.md)
- [\[ECCV 2024\] Omni6D: Large-Vocabulary 3D Object Dataset for Category-Level 6D Object Pose Estimation](omni6d_large-vocabulary_3d_object_dataset_for_category-level_6d_object_pose_esti.md)
- [\[ECCV 2024\] Power Variable Projection for Initialization-Free Large-Scale Bundle Adjustment](power_variable_projection_for_initialization-free_large-scale_bundle_adjustment.md)
- [\[ECCV 2024\] TRAM: Global Trajectory and Motion of 3D Humans from in-the-wild Videos](tram_global_trajectory_and_motion_of_3d_humans_from_in-the-wild_videos.md)

</div>

<!-- RELATED:END -->
