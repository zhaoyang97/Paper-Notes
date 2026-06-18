---
title: >-
  [论文解读] Seeing Across Views: Benchmarking Spatial Reasoning of Vision-Language Models in Robotic Scenes
description: >-
  [ICLR2026][多模态VLM][multi-view spatial reasoning] 提出 MV-RoboBench，首个整合多视角空间推理与机器人操作执行评测的 benchmark，包含 1.7K 人工标注 QA，揭示当前最强 VLM（GPT-5 仅 56.4%）与人类（91.0%）之间存在巨大差距。
tags:
  - "ICLR2026"
  - "多模态VLM"
  - "multi-view spatial reasoning"
  - "robotic manipulation"
  - "VLM benchmark"
  - "embodied AI"
  - "MV-RoboBench"
---

# Seeing Across Views: Benchmarking Spatial Reasoning of Vision-Language Models in Robotic Scenes

**会议**: ICLR2026  
**arXiv**: [2510.19400](https://arxiv.org/abs/2510.19400)  
**代码**: [GitHub](https://github.com/) (项目页面已发布)  
**领域**: 多模态VLM  
**关键词**: multi-view spatial reasoning, robotic manipulation, VLM benchmark, embodied AI, MV-RoboBench  

## 一句话总结
提出 MV-RoboBench，首个整合多视角空间推理与机器人操作执行评测的 benchmark，包含 1.7K 人工标注 QA，揭示当前最强 VLM（GPT-5 仅 56.4%）与人类（91.0%）之间存在巨大差距。

## 研究背景与动机
- 视觉语言模型（VLM）是具身 AI 和视觉-语言-动作（VLA）模型的核心基础，在机器人感知、推理和决策中扮演关键角色
- 大多数 VLM 评测聚焦于单视角设定，但多相机配置在机器人平台上已日益普及，可提供互补视角以缓解遮挡和深度模糊
- 现有空间推理基准（EmbSpatial-Bench、Visual Spatial、RoboSpatial 等）主要关注单视角关系推理，缺乏多视角+机器人操作的结合
- 少数多视角基准（All-Angles Bench、Ego3D-Bench）仅关注照片对齐或导航感知，未涉及操作导向的具身推理
- **核心 gap**: 缺少一个系统评测 VLM 在多视角机器人操作场景中空间推理能力的基准

## 方法详解

### 整体框架
MV-RoboBench 是一个专为多视角机器人操作场景设计的空间推理评测基准。它基于 AgiWorld 和 BridgeV2 两个真实机器人数据集构建，覆盖单臂与双臂操作，经一条三阶段人工流水线从约 980 个操作 episode 中提炼出 1,708 道五选一 QA。题目沿"空间理解"与"机器人执行"两条主线、共八个子任务展开，把"看懂多视角场景"与"判断操作是否合理"统一在同一套协议下评测，从而能横向比较 40+ 个 VLM（开源 / 闭源 / 推理增强）的真实水平。在此之上基准再叠两层分析：一是在输入端注入三类思维链（chain-of-thought, CoT）式增强，探测模型差到底是"没看到线索"还是"看到了也不会推理"；二是用内、外两条相关性轴，检验"空间看得准是否就操作得对""单视角强是否能迁移到多视角"这两个常被默认的迁移假设。

### 关键设计

**1. 两大类八任务评测体系：把多视角具身推理拆成可定位的能力维度**
单纯报告一个总准确率无法说明模型究竟卡在哪一步，因此基准把能力切成两大类共八个子任务。空间理解类聚焦"跨视角把场景拼成一致的 3D 心理表征"，包含 Cross-View Matching（在不同相机视角下识别同一物体）、Distance Judgement（判断物体间相对距离）、Viewpoint Identification（推理相机视角的变换关系）与 3D Spatial Consistency（在多视角间维护物体一致的相对位置）。机器人执行类则考察"基于这种空间理解做操作决策"，包含 Action Planning（选出合理的多步操作序列）、Step Execution（验证下一步单步动作是否正确）、Trajectory Selection（评估候选运动路径的可行性）与 Affordance Recognition（判断对物体的某种交互是否可行）。这种细粒度拆分让后续能精确定位失败来源——例如可以直接发现几乎所有非推理模型都栽在 3D Spatial Consistency 上，而非笼统地说"模型不行"。

**2. 高质量人工构建流程：让干扰项合理但答案唯一**
基准走一条三阶段流水线来保证题目质量。数据收集阶段先用规则过滤候选场景，再让 GPT-4.1 做一轮辅助分诊（仅用于候选分流，不参与生成任何 QA 内容），最后由人工验证；QA 生成阶段用任务特定模板搭配训练有素的标注员逐题构建五选一题，刻意让四个干扰项看上去合理但与正确答案可区分；最后再经过一轮 human-in-the-loop 审查，迭代修订并平衡各选项的答案分布。把 LLM 严格限制在分诊而非内容生成，既避免了模型自评偏差，也让平衡后的答案分布堵住了靠选项先验蒙题的捷径。

**3. CoT 增强探索：检验外挂线索能否补上空间短板**
为了回答"模型差是因为没看到线索、还是看到了也不会推理"，基准在输入端注入三类 CoT 式增强而不改动模型本身：文本 CoT 用 GPT-4.1 生成场景描述作为补充文本，视觉 CoT 用 VGGT 做新视角合成提供额外视觉证据，结构 CoT 用 MoGe-2 估计深度先验注入几何约束。三者分别从语言、视觉、几何三个方向补料，配合后续消融就能区分出究竟是哪一类信息缺口在拖累多视角推理，也能看出不同容量的模型对外部线索的吸收能力差异。

**4. 双轴相关性分析：检验两个被默认的假设是否真成立**
基准还设计了两条分析轴来拷问两个常被想当然的迁移假设。内部轴度量同一批模型在多视角场景下空间推理得分与机器人执行得分之间的相关性，用来检验"空间看得准是否就操作得对"；外部轴则把模型在单视角空间基准 OmniSpatial 上的表现与其在 MV-RoboBench 上的多视角具身推理表现对照，用来检验"单视角强是否能可靠迁移到多视角"。这两条轴让基准不止给出一张排行榜，而能直接给出关于能力可迁移性的可证伪结论。

## 实验

### 主实验表：多模型多类别评测

| 模型 | 平均准确率 | 空间理解 | 机器人执行 |
|------|-----------|---------|-----------|
| Random Choice | 19.71% | ~19% | ~20% |
| GPT-4.1 | 30.90% | 26.8% avg | 32.8% avg |
| GPT-5 (最强) | **56.41%** | 52.7% avg | 60.4% avg |
| Gemini-2.5-pro | 49.52% | 45.8% avg | 53.2% avg |
| o4-mini | 46.47% | 40.4% avg | 52.5% avg |
| Qwen2.5-vl-72B (开源最强) | 24.29% | 21.9% avg | 26.7% avg |
| InternVL3-78B | 23.25% | 20.9% avg | 25.6% avg |
| 人类 | **91.04%** | 93.7% avg | 88.2% avg |

### CoT 增强消融

| 增强方式 | Qwen2.5-vl-7B | Gemma-3-12B | GPT-4.1 |
|---------|---------------|-------------|---------|
| 无增强 (baseline) | 20.84% | 20.49% | 29.87% |
| + CoT prompting | 20.49 (-0.35) | **24.19 (+3.70)** | 29.84 (-0.03) |
| + 文本描述 | 20.90 (+0.06) | 18.43 (-2.06) | **31.66 (+1.79)** |
| + 新视角合成 | 20.02 (-0.82) | 18.31 (-2.18) | 28.02 (-1.85) |
| + 深度先验 | 21.14 (+0.30) | 20.41 (-0.08) | **33.12 (+3.25)** |

### 关键发现
1. **3D 空间一致性最具挑战性**：大多数非推理模型在此子任务上接近甚至低于随机（~19%），推理增强模型可提升至 49-82%
2. **空间与机器人推理正相关**：但仅在模型具备充足跨视角融合能力时成立
3. **单视角表现不可靠迁移**：OmniSpatial 上表现优秀的模型在 MV-RoboBench 上仍可能接近随机
4. **CoT 增强效果混合**：合成新视角倾向于降低性能，深度先验仅对高容量模型有效
5. **推理优化架构显著优于感知模型**：GPT-5 vs GPT-4.1 提升约 25 个百分点

## 亮点
- 首个将多视角空间推理与机器人操作执行系统整合的评测基准，填补重要空白
- 1,708 道人工精标 QA 质量高，覆盖八个子任务维度，评测粒度细
- 发现了两个重要结论：空间-机器人推理正相关 + 单视角不可靠迁移，对后续研究有指导意义
- 系统性地探索了 CoT 增强方式在多视角场景中的效果，发现简单堆叠几何线索不够

## 局限性
- 基准规模相对小（1.7K QA），可能不足以覆盖所有操作场景的多样性
- 所有任务均为五选一 MCQ 格式，未涉及开放式空间推理
- 仅使用两个数据源（AgiWorld + BridgeV2），场景多样性有限
- CoT 增强探索较初步，未深入结合几何编码器等更深层方法
- 未包含动态/视频场景的多视角推理

## 相关工作
- 单视角空间基准：EmbSpatial-Bench、Visual Spatial、RoboSpatial、SpatialVLM、VSI-Bench、OmniSpatial
- 多视角基准：All-Angles Bench、Ego3D-Bench、ERQA、MMSI-Bench
- 机器人评测：ShareRobot
- 3D 理解方法：SpatialRGPT、3D-LLM、SpatialBot、VLM-3R
- VLA 模型：π0、CogAct、OpenVLA

## 总体评价

| 维度 | 评分 |
|------|------|
| 创新性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 影响力 | ⭐⭐⭐⭐ |
| **综合** | **⭐⭐⭐⭐** |

> 评测类工作，核心贡献在于识别了多视角+机器人操作这一关键 gap 并构建了高质量基准。实验覆盖 30+ 个模型非常全面，内外部相关性分析有洞见。但技术方法上偏数据构建，无模型创新。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] 3ViewSense: Spatial and Mental Perspective Reasoning from Orthographic Views in Vision-Language Models](../../ICML2026/multimodal_vlm/3viewsense_spatial_and_mental_perspective_reasoning_from_orthographic_views_in_v.md)
- [\[ICLR 2026\] Spatial-DISE: A Unified Benchmark for Evaluating Spatial Reasoning in Vision-Language Models](spatial-dise_a_unified_benchmark_for_evaluating_spatial_reasoning_in_vision-lang.md)
- [\[ICLR 2026\] FRIEDA: Benchmarking Multi-Step Cartographic Reasoning in Vision-Language Models](frieda_benchmarking_multi-step_cartographic_reasoning_in_vision-language_models.md)
- [\[ICLR 2026\] Spatial CAPTCHA: Generatively Benchmarking Spatial Reasoning for Human-Machine Differentiation](spatial_captcha_generatively_benchmarking_spatial_reasoning_for_human-machine_di.md)
- [\[CVPR 2026\] Think with 3D: Geometric Imagination Grounded Spatial Reasoning from Limited Views](../../CVPR2026/multimodal_vlm/think_with_3d_geometric_imagination_grounded_spatial_reasoning_from_limited_view.md)

</div>

<!-- RELATED:END -->
