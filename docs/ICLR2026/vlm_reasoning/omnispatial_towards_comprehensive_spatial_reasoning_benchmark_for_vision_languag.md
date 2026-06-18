---
title: >-
  [论文解读] OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models
description: >-
  [ICLR 2026][多模态VLM][空间推理] 基于认知心理学构建OmniSpatial——首个全面空间推理基准，系统覆盖动态推理、复杂空间逻辑、空间交互和透视转换4大维度50个子类别共8.4K人工标注QA对，让o3最强推理模型仅达56.33%而人类达92.63%→揭示复杂空间推理仍是VLM的核心瓶颈。
tags:
  - "ICLR 2026"
  - "多模态VLM"
  - "空间推理"
  - "VLM基准"
  - "认知心理学"
  - "动态推理"
  - "透视转换"
---

# OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models

**会议**: ICLR 2026  
**arXiv**: [2506.03135](https://arxiv.org/abs/2506.03135)  
**代码**: [项目页面](https://github.com/omnispatial)  
**领域**: 多模态VLM/基准测试  
**关键词**: 空间推理, VLM基准, 认知心理学, 动态推理, 透视转换

## 一句话总结

基于认知心理学构建OmniSpatial——首个全面空间推理基准，系统覆盖动态推理、复杂空间逻辑、空间交互和透视转换4大维度50个子类别共8.4K人工标注QA对，让o3最强推理模型仅达56.33%而人类达92.63%→揭示复杂空间推理仍是VLM的核心瓶颈。

## 研究背景与动机

**领域现状**：空间推理是VLM的核心能力。现有基准（SpatialBot-Bench、EmbSpatial等）集中于基础空间关系——辨别左右、估计远近、物体计数。最新推理模型（o3、Gemini-2.5-Pro）已在这些基准上达到>90%准确率→基础空间理解接近饱和。

**现有痛点**：
- 基础空间关系（左/右/前/后/计数）≠复杂空间推理（旋转/变形/路径规划/视角变换）→现有基准低估了真实能力差距
- 现有基准多采用模板自动标注→数据多样性和挑战性不足，问题表述刻板（如"A在B的左边吗？"）
- 缺乏认知心理学理论支撑的系统化分类→各基准任务设计零散、覆盖面有限

**核心矛盾**：VLM在现有基准上的"高分"掩盖了其在真实场景复杂空间推理上的根本性不足——人类在紧急情况下理解AED位置不仅需要辨别"在门右边"，还需要读懂示意图、关联地图与实景、规划路线。

**本文目标**：构建一个"不可饱和"的综合性空间推理基准，覆盖从基础到高阶的完整空间认知能力谱。

**切入角度**：从认知心理学的空间认知理论出发（Chabris 2006; Meneghetti 2022），将复杂空间推理分为4个互补维度→以此为框架设计50个子类别→确保理论完备性。

**核心 idea**：用认知心理学的空间认知理论重新定义"空间推理"评测的完整边界。

## 方法详解

### 整体框架

OmniSpatial 把视觉-空间推理形式化为映射 $f:(\mathbf{I}_{1:T}, q) \longrightarrow a$：给定 RGB 观测流 $\mathbf{I}_{1:T}$ 和任务查询 $q$，模型输出落在可验证答案/动作空间中的 $a$；标注刻意排除可纯靠语言常识答出的题，确保分数提升能归因于视觉推理本身。整个基准围绕一套认知心理学驱动的分类体系搭建——把空间认知拆成 4 个互补维度、共 50 个子类别，并配上多源采集 + 全人工标注的数据管线，最后辅以两条推理增强策略来探测瓶颈到底在哪。

### 关键设计

**1. 四大认知维度的分类体系：把"空间推理"重新定义到完备**

现有基准只覆盖左右、远近、计数这类基础关系，最新模型已刷到 >90%、失去区分度。OmniSpatial 不靠随意堆难题来对抗饱和，而是从空间认知的几项独立官能（可视化、心理旋转、透视转换、空间更新）出发，划出四个互补维度、对应不同认知能力：动态推理（11 子类）从视觉证据推断运动与时间变化，如轨迹预测、物理仿真、交通情境分析；复杂空间逻辑（15 子类）做关系/变换/几何结构的高阶推理，如 3D 结构推理、心理折叠展开、空间兼容性判断；空间交互（12 子类）在环境约束下做任务导向推理，如路径规划、障碍避免、上下文动作选择；透视转换（12 子类）考察采纳他者视角的能力，如心理旋转、镜像理解、多智能体视角协调。四个维度合计 50 个子类别，既因为锚在认知理论上而有完备性，又覆盖了从机器人操作到自动驾驶的完整应用谱——任何维度被刷高都还有其余维度兜底，使整个基准"不可饱和"。

**2. 多源采集与会话式人工标注：用难度和多样性堵住模板化漏洞**

模板自动标注（"A 在 B 左边吗？"）表述刻板、多样性差，是现有基准被刷穿的另一原因。OmniSpatial 改从四类来源采集图像以拉开分布：网络图片用 `-ai`、`-generated` 等搜索词排除合成内容，覆盖多国、多场景、多天气；公共空间认知测试题侧重纯空间推理；至少 3 个国家的驾照考试题（含从美国驾考视频抽帧再标）引入真实交通场景；以及 MME（带深度信息）和 HOI4D（人-物交互帧序列）等已有数据集。所有问题都改用会话式自然表述（如"如果你正进教室，学生在你哪一侧？"），并由 6 名标注者交叉验证，标注一致性 Krippendorff's $\alpha = 0.84$，最终切出 1.5K 纯人工标注的测试集与 6.9K 训练集（合计约 8.4K QA 对）。难度与表述的多样性让题目无法被模板套路猜中，分数提升只能来自真实的视觉空间推理。

**3. PointGraph 与 SpatialCoT 增强：把几何先验显式喂给模型，借此定位瓶颈**

VLM 缺乏内在 3D 表示，遮挡和视角歧义常导致空间推理崩塌；作者于是设计两条增强策略，既想拉高分、更想用它们当探针看瓶颈在哪。PointGraph 用开放词汇 grounding 模型 Florence-2 定位场景中的多个物体、提取中心点与边界框，组装成 JSON 格式的场景图作为显式几何线索；SpatialCoT 则受人类心理意象能力启发，用 InstantMesh 为每张输入图生成 6 个新视角拼成多视角图，连同问题一起送入模型做链式思考，提供强几何先验以消解遮挡与视角相关的歧义。结果是两条策略只在部分维度带来有限提升——这种"补了先验也救不回来"的现象反过来印证：瓶颈是 VLM 基础空间认知能力的缺失，而不是缺少几何标注。

## 实验关键数据

### 主实验：OmniSpatial-test上代表性模型表现（%）

| 模型 | 平均 | 操作 | 运动分析 | 交通 | 定位 | 地理 | 策略 | 模式识别 | 几何推理 | 自我中心 | 他者中心 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Random | 24.98 | - | 24.86 | 26.30 | 25.88 | 23.43 | 27.27 | 21.44 | 24.77 | 22.55 | 24.84 |
| GPT-4o | 47.81 | 65.54 | 57.23 | 56.47 | 52.38 | 54.09 | 26.29 | 25.48 | 75.98 | 39.49 | 39.76 |
| o3 | **56.33** | 71.89 | 66.18 | 61.18 | 68.57 | 65.45 | **40.21** | 29.68 | 77.06 | **48.40** | **48.19** |
| Gemini-2.5-Pro | 55.19 | 67.57 | **71.39** | 62.35 | **75.24** | **64.55** | 43.30 | 34.84 | 74.51 | 38.03 | 37.35 |
| InternVL3-78B | 49.33 | 63.78 | 63.12 | 56.24 | 59.24 | 51.45 | 27.63 | 30.19 | 74.51 | 38.46 | 35.90 |
| SoFar-3B | 45.14 | 56.49 | 51.16 | 54.12 | 53.14 | 52.73 | 31.75 | 22.88 | **71.60** | 36.56 | 41.69 |
| **人类** | **92.63** | **94.62** | **96.07** | **91.38** | **95.11** | **92.15** | **89.02** | **85.90** | **98.53** | **94.30** | **90.26** |

### 现有基准 vs OmniSpatial 的饱和度对比

| 模型 | SpatialBot-Bench | EmbSpatial | OmniSpatial |
|------|:---:|:---:|:---:|
| o3 | >90% | >90% | 56.33% |
| Gemini-2.5-Pro | >90% | >90% | 55.19% |
| 人类 | ~95% | ~95% | 92.63% |

### 关键发现

- 最强推理模型o3（56.33%）vs 人类（92.63%）→差距达36个百分点→复杂空间推理远未解决
- **策略（Strategy, ~40%）和模式识别（Pattern Recognition, ~30%）**是最难维度→即使o3也仅半数正确
- 透视转换（自我中心/他者中心，~48%/~48%）难度显著→VLM缺乏内在3D表示和心理旋转能力
- 专用空间模型（SpatialBot、RoboPoint）在OmniSpatial上并无优势（35-40%）→"专用"训练集太简单
- PointGraph和SpatialCoT可改善部分维度但提升有限→基础空间认知能力缺失是根源

## 亮点与洞察

- **"饱和的警示"**：清晰展示现有基准已被最新模型"解决"→社区需要更难的评估标杆。OmniSpatial将评测从"Pattern Matching"提升到"Cognitive Reasoning"
- **认知心理学的理论锚点**：不是随意添加难题，而是从空间认知理论出发→系统性和完备性有理论保证
- **50个子类别的诊断价值**：不同子任务难度差异极大（几何推理~75% vs 模式识别~30%）→为模型改进提供精确方向
- **人类92.63%的性能上界**：即使人类也并非100%→有些题目（如模式识别85.90%）对人类也有挑战性→说明题目设计有深度

## 局限与展望

- 主要基于静态图片/少量视频帧→动态空间推理可进一步扩展到连续视频
- 所有3D推理任务仍是在2D图片上进行→真正的3D交互环境（VR/模拟器）未涉及
- 人工标注质量高但扩展成本大→需要探索半自动标注方案以持续扩充数据
- PointGraph和SpatialCoT作为增强策略效果有限→更根本的改进可能需要在模型架构层面引入3D空间先验

## 相关工作与启发

- **vs SpatialBot-Bench/EmbSpatial**：仅6-8类基础空间关系+模板标注→OmniSpatial 50类+人工标注→维度和难度全面升级
- **vs VSI-Bench (Yang et al., 2024)**：8类室内场景+模板标注288样本→OmniSpatial覆盖室内外多国场景6.5K图
- **vs RoboSpatial (Song et al., 2024)**：模板自动标注百万级→规模大但多样性和难度受限
- **启发**：可否将OmniSpatial与具身AI结合→在模拟器中让模型执行空间推理后的动作→从"回答问题"到"执行任务"？

## 评分

⭐⭐⭐⭐⭐ (5/5)

综合评价：首个基于认知心理学理论的全面空间推理基准，50类×8.4K题的精心人工标注，o3仅56%/人类93%的巨大差距证明了基准的区分度和价值——为VLM空间认知能力评估设定了新标杆。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] SpatiaLab: Can Vision-Language Models Perform Spatial Reasoning in the Wild?](spatialab_can_vision-language_models_perform_spatial_reasoning_in_the_wild.md)
- [\[ICLR 2026\] Spatial-DISE: A Unified Benchmark for Evaluating Spatial Reasoning in Vision-Language Models](spatial-dise_a_unified_benchmark_for_evaluating_spatial_reasoning_in_vision-lang.md)
- [\[CVPR 2026\] SpatiaLQA: A Benchmark for Evaluating Spatial Logical Reasoning in Vision-Language Models](../../CVPR2026/multimodal_vlm/spatialqa_a_benchmark_for_evaluating_spatial_logical_reasoning_in_vision-languag.md)
- [\[ICLR 2026\] Spatial CAPTCHA: Generatively Benchmarking Spatial Reasoning for Human-Machine Differentiation](spatial_captcha_generatively_benchmarking_spatial_reasoning_for_human-machine_di.md)
- [\[CVPR 2025\] ESPIRE: A Diagnostic Benchmark for Embodied Spatial Reasoning of Vision-Language Models](../../CVPR2025/multimodal_vlm/espire_a_diagnostic_benchmark_for_embodied_spatial_reasoning_of_vision-language_.md)

</div>

<!-- RELATED:END -->
