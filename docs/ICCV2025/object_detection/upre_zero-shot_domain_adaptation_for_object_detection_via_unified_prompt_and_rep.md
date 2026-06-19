---
title: >-
  [论文解读] UPRE: Zero-Shot Domain Adaptation for Object Detection via Unified Prompt and Representation Enhancement
description: >-
  [ICCV 2025][目标检测][零样本域自适应] 提出 UPRE 框架，通过联合优化多视角域提示（MDP）和统一表示增强（URE）来同时缓解零样本域自适应目标检测中的检测偏差和域偏差，在恶劣天气、跨城市、虚拟到现实三类场景的九个数据集上取得 SOTA 性能。 零样本域自适应（ZSDA）旨在将模型从源域迁移到目标域…
tags:
  - "ICCV 2025"
  - "目标检测"
  - "零样本域自适应"
  - "提示学习"
  - "视觉-语言模型"
  - "域偏移"
---

# UPRE: Zero-Shot Domain Adaptation for Object Detection via Unified Prompt and Representation Enhancement

**会议**: ICCV 2025  
**arXiv**: [2507.00721](https://arxiv.org/abs/2507.00721)  
**代码**: [GitHub](https://github.com/AMAP-ML/UPRE)  
**领域**: 目标检测  
**关键词**: 零样本域自适应, 目标检测, Prompt学习, 视觉-语言模型, 域偏移

## 一句话总结

提出 UPRE 框架，通过联合优化多视角域提示（MDP）和统一表示增强（URE）来同时缓解零样本域自适应目标检测中的检测偏差和域偏差，在恶劣天气、跨城市、虚拟到现实三类场景的九个数据集上取得 SOTA 性能。

## 研究背景与动机

零样本域自适应（ZSDA）旨在将模型从源域迁移到目标域，且不访问任何目标域图像。这在目标检测中尤为困难，因为需要同时处理分布偏移和精确定位。现有方法面临两个核心偏差：

**域偏差（Domain Bias）**：源域和目标域之间的分布偏移引入与任务无关的噪声，损害模型性能。例如白天晴天训练的检测器难以泛化到夜间多雨场景，因为光照、反射等视觉风格差异巨大。

**检测偏差（Detection Bias）**：CLIP 等 VLM 强调全局图像表示，但忽略了目标检测所需的实例级细节。现有方法使用手工构造的 prompt（如"a photo of a [class]"），无法充分捕获前景和背景物体的上下文属性。

关键矛盾在于：
- 解决域偏差的方法（如 PODA 的 prompt 驱动语义增强）使用手工 prompt 生成伪目标域特征，但忽略甚至加剧了检测偏差
- 解决检测偏差的方法（如 DetPro 学习 prompt 表示）仅在源域训练，导致 prompt 虽适合源域检测但放大了域偏差

本文首次提出同时解决这两种偏差的统一框架。

## 方法详解

### 整体框架

UPRE 基于 Faster R-CNN + CLIP ResNet-101 构建，包含三个核心组件和两个多层次策略：

1. **多视角域提示（MDP）**：融合语言域先验和可学习 prompt，提供检测特定的跨域知识
2. **统一表示增强（URE）**：通过特征变换生成域风格多样化的伪目标域表示
3. **多层次增强策略**：相对域距离（RDD）在图像级对齐多模态表示，正负分离（PNS）在实例级捕获检测知识

### 关键设计

1. **Multi-view Domain Prompt（MDP）**:

    - 功能：学习三种 prompt 表示——图像 prompt、正 prompt、负 prompt，分别提供不同层次的域适应知识
    - 核心思路：
        - **图像 prompt** $\mathcal{R}_i^d = [u_1, u_2, \ldots, u_L, k_d]$：可学习向量 + 域描述词嵌入（如"a photo taken on a [domain]"），提供图像级域风格先验
        - **正 prompt** $\mathcal{R}_p^t(c) = [v_1, \ldots, v_L, (k_t, k_c)]$：可学习向量 + "a [domain] photo of a [class]"，捕获前景目标在目标域中的风格变化
        - **负 prompt** $\mathcal{R}_n^t(\mathcal{C}_{bg}) = [w_1, \ldots, w_L, (k_t, k_{\mathcal{C}_{bg}})]$：类似结构但用于背景类"unknown class"，捕获背景上下文
    - 设计动机：保留手工 prompt 的静态语言先验，让可学习部分专注于捕获检测特定的跨域知识；三种 prompt 分别覆盖图像级、前景和背景的不同视角

2. **Unified Representation Enhancement（URE）**:

    - 功能：将源域特征增强为伪目标域特征，增加域风格多样性
    - 核心思路：将源域特征图 $F_s$ 分割为 $M \times N$ 个块，对每个块应用可学习的均值增强 $\mathcal{E}_\mu$ 和偏差增强 $\mathcal{E}_\sigma$：$F_{s \to t} = \{\mathcal{E}_\sigma^j \cdot F_s^j + \mathcal{E}_\mu^j\}_{j=1}^{M \times N}$
    - 设计动机：现实世界中同一图像不同区域的风格常常不同（如雨夜图像中，近处区域因照明呈"多雨"风格，远处因光线不足呈"夜晚"风格）。PODA 等方法用全局 AdaIN 转换风格是不够的，逐块增强能捕获细粒度的局部风格变化

3. **Relative Domain Distance（RDD）**:

    - 功能：在图像级约束多模态表示的对齐，稳定增强和约束训练之间的矛盾
    - 核心思路：定义三个互补的损失函数：
        - **对齐损失** $\mathcal{L}_a = \mathbb{E}[1 - f(\mathbf{e}_i^{s \to t}, \mathbf{t}_i^t)]$：拉近增强后的图像表示与目标域文本表示
        - **约束损失** $\mathcal{L}_s = \mathbb{E}[\|\mathbf{e}_i^s - \mathbf{e}_i^{s \to t}\|_1]$：防止过度增强破坏语义信息
        - **相对距离损失** $\mathcal{L}_r = \mathbb{E}[\|(\mathbf{e}_i^s - \mathbf{e}_i^{s \to t}) - (\mathbf{t}_i^s - \mathbf{t}_i^t)\|_1]$：确保视觉域偏移与语言域偏移一致
    - 设计动机：$\mathcal{L}_a$ 和 $\mathcal{L}_s$ 目标相互矛盾（一个要求距离远，一个要求距离近），$\mathcal{L}_r$ 在重叠区域引导搜索，稳定训练

4. **Positive-Negative Separation（PNS）**:

    - 功能：在实例级分别对正负 proposal 应用不同的分类目标
    - 核心思路：正 proposal 用交叉熵损失 $\mathcal{L}_c$ 仅在前景类之间分类；负 proposal 用 $\mathcal{L}_{bg}$ 并设置均匀目标分布 $y_{bg}$，防止背景类过度自信。正负 proposal 分别使用正 prompt 和负 prompt 的表示
    - 设计动机：不同于 DetPro 仅用共享 prompt 和类别关键字，PNS 让正负 proposal 学习不同的上下文信息——正 proposal 学习前景物体的域风格变化，负 proposal 学习背景上下文

### 损失函数 / 训练策略

训练分两阶段：
1. **Prompt 和增强学习阶段**（5k iterations）：同时训练 MDP 和 URE，两者相互增强——MDP 精细化驱动 URE 改进，URE 生成的伪特征反过来帮助 MDP 学习检测知识
2. **检测器微调阶段**（100k iterations）：冻结 MDP、URE 和文本编码器，微调 CLIP 骨干和 RCNN 检测器

推理时禁用 URE 变换，保持特征处理一致性。

## 实验关键数据

### 主实验

| 场景 | 数据集 | UPRE (mAP) | 之前 SOTA | 提升 |
|------|--------|-----------|----------|------|
| 恶劣天气 | Daytime Foggy | **40.0** | 39.6 (UFR) | +0.4 |
| 恶劣天气 | Night Clear | **41.5** | 41.0 (DAI-Net) | +0.5 |
| 恶劣天气 | Night Rainy | **19.8** | 19.2 (PDD/UFR) | +0.6 |
| 恶劣天气 | Dusk Rainy | **34.5** | 33.9 (OA-DG) | +0.6 |
| 跨城市 | Cityscapes→BDD100K | **28.7** | 27.2 (OA-DG) | +1.5 |
| 跨城市 | Cityscapes→KITTI | **74.3** | 73.6 (PODA) | +0.7 |
| 虚拟到真实 | Sim10K→Cityscapes | **47.9** | 47.0 (OA-DG) | +0.9 |
| 虚拟到真实 | Sim10K→BDD100K | **37.8** | 36.1 (PODA) | +1.7 |
| 虚拟到真实 | Sim10K→KITTI | **61.9** | 60.7 (CLIP-GAP) | +1.2 |

对比 Faster R-CNN 基线，UPRE 在恶劣天气条件下平均提升 7.8% mAP。

### 消融实验

| 配置 | Daytime Foggy | Night Clear | Night Rainy | Dusk Rainy | 说明 |
|------|--------------|-------------|-------------|------------|------|
| 无可学习 prompt + 关键词 | 37.2 | 37.5 | 17.0 | 31.7 | 基线 |
| 无可学习 prompt + 完整描述 | 38.2 | 39.3 | 17.9 | 32.2 | 完整描述有帮助 |
| 可学习 prompt + 关键词 | 38.7 | 40.1 | 18.6 | 33.0 | 可学习提升明显 |
| 可学习(共享) + 完整描述 | 38.0 | 39.7 | 17.4 | 32.2 | 共享参数反而有害！ |
| 可学习 prompt + 完整描述 | **40.0** | **41.5** | **19.8** | **34.5** | 完整方案最优 |

RDD 损失组合消融（MAD = 均值绝对偏差，衡量训练稳定性）：

| $\mathcal{L}_a$ | $\mathcal{L}_s$ | $\mathcal{L}_r$ | 平均 mAP | MAD | 说明 |
|:---:|:---:|:---:|:---:|:---:|------|
| ✓ | - | - | 28.95 | 0.8 | 仅对齐不稳定 |
| ✓ | ✓ | - | 31.75 | 1.7 | 矛盾目标导致最大波动 |
| ✓ | ✓ | ✓ | **32.50** | **0.4** | RDD 稳定训练 + 最优性能 |

### 关键发现

- **Prompt 设计至关重要**：保留完整的手工 prompt 结构（而非仅关键词），让可学习部分专注于域知识，平均提升 3.0% mAP
- **共享参数的可学习 prompt 有害**：一个共享 prompt 同时描述图像/正/负三种视角会导致信息冲突
- $\mathcal{L}_r$ 是训练稳定性的关键——MAD 从 1.7 降至 0.4
- t-SNE 可视化显示 UPRE 成功分离了不同天气条件的域嵌入（如 Night Clear vs Night Rainy），而原始 CLIP 无法区分
- 在 Daytime Foggy 场景中，bus(+6.8%)、motor(+9.9%)、rider(+9.1%) 等类别提升最大，表明细粒度上下文捕获能力强

## 亮点与洞察

- **统一框架同时解决两种偏差**：首次明确指出域偏差和检测偏差的相互对抗关系，并用统一训练流程同时缓解
- **逐块特征增强**优于全局 AdaIN（PODA），更贴合现实世界中局部风格变化的特点
- **三种 prompt 的分工设计**（图像级/正/负）比共享 prompt 更有效，体现了检测任务中不同组件需要不同域知识的洞察
- RDD 的"相对距离"思想比单纯拉近/推远更优雅——要求视觉域偏移与语言域偏移保持一致

## 局限与展望

- 域描述 $k_d$ 仍需手工定义（如"rainy night"），如何自动发现目标域的语言描述是开放问题
- URE 的增强粒度（$M \times N$ 的块划分）是固定的，自适应的粒度选择可能更优
- 实验主要集中在自动驾驶场景，其他域（如医学影像、遥感）的泛化性未验证
- 两阶段训练增加了流程复杂性，端到端的统一训练可能更高效
- 相比 DAI-Net 在 Night Clear 上的优势不大（41.5 vs 41.0），说明在某些域偏移较小的场景中改进空间有限

## 相关工作与启发

- PODA（ICCV 2023）通过 AdaIN 进行全局风格转换是本文的重要对比对象，URE 的逐块增强是对其的直接改进
- CLIP-GAP（CVPR 2023）首先利用 VLM 进行 prompt 驱动的语义增强，但忽略了检测偏差
- DetPro 的正负 proposal 分离思想在本文中被扩展到跨域场景
- 本文的统一训练思路（prompt 和增强相互驱动）可推广到其他视觉-语言任务的域自适应

## 评分

- 新颖性: ⭐⭐⭐⭐ 统一解决域偏差+检测偏差的框架设计有新意，但各组件相对增量
- 实验充分度: ⭐⭐⭐⭐⭐ 九个数据集、三类场景全面覆盖，消融实验深入
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，但模块较多导致方法部分略紧凑
- 价值: ⭐⭐⭐⭐ 对自动驾驶等需要跨域泛化的实际应用有较强指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] OpenKD: Opening Prompt Diversity for Zero- and Few-shot Keypoint Detection](../../ECCV2024/object_detection/openkd_opening_prompt_diversity_for_zero-_and_few-shot_keypoint_detection.md)
- [\[CVPR 2026\] GS-CLIP: Zero-shot 3D Anomaly Detection by Geometry-Aware Prompt and Synergistic View Representation Learning](../../CVPR2026/object_detection/gs-clip_zero-shot_3d_anomaly_detection_by_geometry-aware_prompt_and_synergistic_.md)
- [\[ICCV 2025\] Visual Modality Prompt for Adapting Vision-Language Object Detectors](visual_modality_prompt_for_adapting_vision-language_object_detectors.md)
- [\[ICLR 2026\] OwlEye: Zero-Shot Learner for Cross-Domain Graph Data Anomaly Detection](../../ICLR2026/object_detection/owleye_zero-shot_learner_for_cross-domain_graph_data_anomaly_detection.md)
- [\[ICCV 2025\] Revisiting Adversarial Patch Defenses on Object Detectors: Unified Evaluation, Large-Scale Dataset, and New Insights](revisiting_adversarial_patch_defenses_on_object_detectors_unified_evaluation_lar.md)

</div>

<!-- RELATED:END -->
