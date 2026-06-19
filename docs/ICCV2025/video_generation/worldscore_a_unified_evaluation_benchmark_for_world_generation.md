---
title: >-
  [论文解读] WorldScore: A Unified Evaluation Benchmark for World Generation
description: >-
  [ICCV 2025][视频生成][世界生成] 提出 WorldScore —— 首个统一的世界生成评估基准，将世界生成分解为一系列"下一场景生成"任务，支持对 3D、4D、I2V 和 T2V 模型的统一评测，并涵盖 3000 个测试样本和 10 项指标。 近年来视频生成（Sora、CogVideoX 等）和 3D/4D 场…
tags:
  - "ICCV 2025"
  - "视频生成"
  - "世界生成"
  - "统一评估基准"
  - "3D/4D场景生成"
  - "多维度评价指标"
---

# WorldScore: A Unified Evaluation Benchmark for World Generation

**会议**: ICCV 2025  
**arXiv**: [2504.00983](https://arxiv.org/abs/2504.00983)  
**代码**: [GitHub](https://haoyi-duan.github.io/WorldScore/)  
**领域**: 视频生成  
**关键词**: 世界生成, 统一评估基准, 3D/4D场景生成, 视频生成, 多维度评价指标

## 一句话总结

提出 WorldScore —— 首个统一的世界生成评估基准，将世界生成分解为一系列"下一场景生成"任务，支持对 3D、4D、I2V 和 T2V 模型的统一评测，并涵盖 3000 个测试样本和 10 项指标。

## 研究背景与动机

近年来视频生成（Sora、CogVideoX 等）和 3D/4D 场景生成（LucidDreamer、WonderWorld 等）取得了飞速进展，催生了"世界生成"这一概念 —— 创建由多个多样场景无缝衔接的大规模世界。然而现有基准存在三大缺陷：

**仅评估单场景质量**：VBench、EvalCrafter 等指标只评价单个视频片段的画质，不涉及场景间的过渡和布局控制。

**不兼容 3D/4D 方法**：现有 benchmark 缺少相机轨迹和参考图像输入，无法对需要相机位姿或种子图像的 3D/4D 方法进行评估。

**缺乏多场景、长序列评测**：没有 benchmark 要求模型进行多步场景生成和长序列世界生成任务。

WorldScore 的出发点是：**世界生成不等于视频生成**，需要在可控性、质量和动态三个维度上建立统一的评测协议，打通不同方法族的比较。

## 方法详解

### 整体框架

WorldScore 将世界生成任务分解为一系列**下一场景生成步骤**（next-scene generation），每一步由三元组 $(\mathcal{C}, \mathcal{N}, \mathcal{L})$ 描述：

- $\mathcal{C} = \{\mathbf{I}, \mathcal{P}\}$：当前场景（图像 + 文本描述）
- $\mathcal{N}$：下一场景文本提示
- $\mathcal{L} = \{\mathcal{T}, \mathcal{Y}\}$：布局（相机轨迹 + 相机运动文本描述）

生成过程表述为：$\mathbf{V} = g_{\text{world}}(w_{\text{proc}}(\mathcal{C}, \mathcal{N}, \mathcal{L}))$

基准进一步将评估划分为**静态世界**（评估可控性 + 质量）和**动态世界**（评估动态行为）。静态任务要求模型生成新场景序列，动态任务要求模型在同一场景内生成运动。

### 关键设计

1. **统一世界规范 (World Specification)**：每个测试样本同时提供图像条件和文本提示（适配 I2V 和 T2V）、相机矩阵和文本化相机运动描述（适配 3D/4D），使所有模型都能被评估。所有方法统一输出为视频格式，消除了方法间的不可比性。

2. **数据集策划**：总计 3000 个高质量测试样本。静态世界 2000 个样本分为 5 类室内 + 5 类室外场景（含小世界 1 场景 + 大世界 3 场景），动态世界 1000 个样本包含 5 种运动类型（刚体运动、流体运动等）。每个样本均有对应的风格化版本（7 种风格候选），同时支持真实感与风格化评测。

3. **WorldScore 指标体系**：共 10 项子指标，按三个维度组织：

    - **可控性 (Controllability)**：相机误差 $e_{\text{camera}} = \sqrt{e_\theta \cdot e_t}$（旋转+平移误差）、物体可控性（开放集目标检测成功率）、内容对齐度（CLIPScore）
    - **质量 (Quality)**：3D 一致性（DROID-SLAM 重投影误差）、光度一致性（光流 AEPE 检测纹理闪烁）、风格一致性（Gram 矩阵 F 范数差异）、主观质量（CLIP-IQA+ 与 CLIP Aesthetic 集成）
    - **动态 (Dynamics)**：运动精度（指定区域 vs 非指定区域的光流对比）、运动幅度（连续帧光流大小）、运动平滑度（帧插值 ground truth 对比）

### 损失函数 / 训练策略

WorldScore 本身是评测基准而非训练方法，因此不涉及损失函数。评分流程为：先对各子指标进行线性归一化映射至 [0, 100]，再取算术平均得到 WorldScore-Static（可控性 + 质量指标均值）和 WorldScore-Dynamic（进一步加入动态指标）。对不支持动态生成的 3D 方法，动态指标设为 0。

## 实验关键数据

### 主实验

| 模型 | WorldScore-Static | WorldScore-Dynamic | 相机可控性 | 3D一致性 | 运动精度 |
|------|-----|-----|-----|-----|-----|
| WonderWorld (3D) | 72.69 | 50.88 | 92.98 | 86.87 | 0.00 |
| LucidDreamer (3D) | 70.40 | 49.28 | 88.93 | 90.37 | 0.00 |
| CogVideoX-I2V | 62.15 | 59.12 | 38.27 | 86.21 | 69.56 |
| Gen-3 | 60.71 | 57.58 | 29.47 | 68.31 | 54.53 |
| Hailuo | 57.55 | 56.36 | 22.39 | 67.18 | 63.46 |
| CogVideoX-T2V | 54.18 | 48.79 | 40.22 | 68.81 | 25.00 |
| 4D-fy | 27.98 | 32.10 | 69.92 | 35.47 | 22.22 |

### 消融实验

| 维度分析 | 关键发现 |
|---------|---------|
| 室内 vs 室外 | 视频模型在室外场景显著弱于 3D 模型，室内差距较小 |
| 小世界 vs 大世界 | 视频模型在长序列（大世界 = 4 场景）任务上表现急剧下降 |
| T2V vs I2V | T2V 可控性更高、运动幅度更大；I2V 质量更高但倾向"粘附"输入视角 |
| 运动幅度 vs 平滑度 | 较大运动往往以牺牲平滑度为代价，揭示当前模型的运动质量瓶颈 |

### 关键发现

- 3D 模型在静态世界生成上大幅领先（WonderWorld 72.69 vs CogVideoX-I2V 62.15），因其天然具备高相机可控性和 3D 一致性。
- 视频模型的主要短板是**相机可控性**，最优视频模型 CogVideoX-T2V 仅 40.22，远低于 3D 方法。
- 开源最强视频模型 CogVideoX-I2V 的综合得分已超越闭源商业模型 Gen-3 和 Hailuo。
- 大运动幅度不等于高运动精度，模型可能产生非预期的相机运动或无关运动。

## 亮点与洞察

- **统一评测范式**的设计非常精巧：通过同时提供文本 + 图像 + 相机矩阵 + 相机文本描述，一套测试样本适配所有方法族。
- **将世界生成分解为 next-scene generation**是核心贡献之一，使评测具有可扩展性（可调节场景序列长度）。
- 光度一致性指标的设计值得借鉴 —— CLIP/DINO 特征无法捕捉细粒度纹理变化，光流 AEPE 更有效。
- 3000 个精心策划的测试样本涵盖室内外、真实感/风格化、静态/动态等多个维度，数据质量高。

## 局限与展望

- 评测仍以自动指标为主，部分子指标（如主观质量、内容对齐）与人类判断存在偏差。
- 目前仅包含一个 4D 模型（4D-fy），该领域代表性不足。
- 相机可控性指标依赖现有 SLAM 方法的精度，对低质量生成视频可能引入噪声。
- 动态评估中固定相机的假设可能过于简化，真实世界生成常伴随相机运动与场景动态的耦合。

## 相关工作与启发

- VBench / EvalCrafter / WorldModelBench 等仅评估单场景质量，WorldScore 填补了多场景世界生成评估空白。
- 3D 场景生成（SceneScape, WonderJourney）的高可控性启示视频模型需注入相机条件（如 CamI2V）。
- 最终 benchmark 结果揭示了 3D 方法与视频方法的互补性，未来融合方向（3D + video prior）值得探索。

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首个统一世界生成基准，设计精巧且概念新颖
- **实验充分度**: ⭐⭐⭐⭐⭐ 20 个模型、3000 测试样本、10 项指标，非常全面
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，图表丰富，但部分指标细节需翻阅补充材料
- **价值**: ⭐⭐⭐⭐⭐ 为世界生成社区提供了急需的标准化评测基础设施

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] VGA-Bench: A Unified Benchmark and Multi-Model Framework for Video Aesthetics and Generation Quality Evaluation](../../CVPR2026/video_generation/vga-bench_a_unified_benchmark_and_multi-model_framework_for_video_aesthetics_and.md)
- [\[ICCV 2025\] VMBench: A Benchmark for Perception-Aligned Video Motion Generation](vmbench_a_benchmark_for_perception-aligned_video_motion_generation.md)
- [\[ICCV 2025\] ETVA: Evaluation of Text-to-Video Alignment via Fine-Grained Question Generation and Answering](etva_evaluation_of_text-to-video_alignment_via_fine-grained_question_generation_.md)
- [\[ICML 2026\] T2AV-Compass: Towards Unified Evaluation for Text-to-Audio-Video Generation](../../ICML2026/video_generation/t2av-compass_towards_unified_evaluation_for_text-to-audio-video_generation.md)
- [\[CVPR 2026\] SLVMEval: Synthetic Meta Evaluation Benchmark for Text-to-Long Video Generation](../../CVPR2026/video_generation/slvmeval_synthetic_meta_evaluation_benchmark_for_text-to-long_video_generation.md)

</div>

<!-- RELATED:END -->
