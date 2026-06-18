---
title: >-
  [论文解读] MMPerspective: Do MLLMs Understand Perspective? A Comprehensive Benchmark for Perspective Perception, Reasoning, and Robustness
description: >-
  [NeurIPS 2025][多模态VLM][透视理解] 首个系统评估多模态大语言模型 (MLLMs) 透视理解能力的基准，包含10个任务、3个维度、2711张图像和5083个问答对，揭示了43个SOTA模型在透视推理和鲁棒性方面的显著不足。 领域现状： 透视理解是人类视觉认知的基础，从文艺复兴绘画到现代摄像机标定…
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "透视理解"
  - "多模态大语言模型"
  - "benchmark"
  - "空间推理"
  - "几何感知"
---

# MMPerspective: Do MLLMs Understand Perspective? A Comprehensive Benchmark for Perspective Perception, Reasoning, and Robustness

**会议**: NeurIPS 2025  
**arXiv**: [2505.20426](https://arxiv.org/abs/2505.20426)  
**代码**: [GitHub](https://yunlong10.github.io/MMPerspective/)  
**领域**: Multimodal VLM  
**关键词**: 透视理解, 多模态大语言模型, benchmark, 空间推理, 几何感知

## 一句话总结

首个系统评估多模态大语言模型 (MLLMs) 透视理解能力的基准，包含10个任务、3个维度、2711张图像和5083个问答对，揭示了43个SOTA模型在透视推理和鲁棒性方面的显著不足。

## 研究背景与动机

**领域现状**: 透视理解是人类视觉认知的基础，从文艺复兴绘画到现代摄像机标定，透视投影广泛用于三维空间在二维平面上的表征。当前MLLMs在视觉问答、图像描述等高层任务上表现出色。

**现有痛点**: 现有基准（如MMBench、MVBench）很少评估模型的几何推理能力，尤其是透视理解——包括灭点识别、平行线汇聚推理、空间关系判断等基础视觉能力。

**核心矛盾**: MLLMs展现了类人的视觉感知能力，但其是否内化了透视几何先验知识完全未知。专用的透视方法依赖精确数学模型或特殊数据集，难以推广到通用任务。

**本文目标**: MLLMs是否真正理解透视？能否定位灭点、理解平行线汇聚、推理三维空间关系、并在视角变换下保持一致？

**切入角度**: 构建从低层感知到高层推理再到鲁棒性的层次化、系统性评估框架。

**核心 idea**: 设计覆盖感知-推理-鲁棒性三个维度的10类透视理解任务，系统揭示MLLMs的空间几何短板。

## 方法详解

### 整体框架

MMPerspective 基准由三个互补的层次化维度构成：**透视感知 (P'Percep)**、**透视推理 (P'Reason)** 和 **透视鲁棒性 (P'Robust)**，共包含10个任务。评估难度从低层视觉识别逐步递增到高层空间推断和变换一致性验证。

### 关键设计

1. **透视感知 (P'Percep)**: 评估模型检测和解释图像中显式透视线索的能力。

    - **灭点感知 (VPP)**: 判断灭点位置或其是否存在于给定区域
    - **关键线感知 (CLP)**: 从候选线中识别地平线
    - **镜头畸变感知 (LDP)**: 区分图像中无曲线畸变的区域
    - **视角感知 (VAP)**: 利用视觉线索推断视线方向（向上/向下/水平）

2. **透视推理 (P'Reason)**: 测试模型整合多个空间线索进行几何推理的能力。

    - **透视类型推理 (PTR)**: 分类图像的透视结构（一点/两点/三点/非线性透视）
    - **线关系推理 (LRR)**: 判断3D空间中两条线是平行、垂直还是相交
    - **透视变换检测 (PTS)**: 检测配对图像间的透视类型变化
    - **灭点计数 (VPC)**: 估算场景中可识别灭点的数量
    - **视野外推理 (OVR)**: 推断灭点不在图像中时所在的象限

3. **透视鲁棒性 (P'Robust)**: 评估模型在保持透视不变的图像变换下的一致性。通过裁剪、翻转、遮挡等操作增强原始图像，测试模型是否给出一致的正确答案。采用两种指标：

    - **Binary P'Robust Score**: 要求所有变换版本全部正确，$\text{Binary-Robust}_{\mathcal{M}} = \frac{1}{|\mathcal{S}|}\sum \mathbb{1}[\bigwedge_{I \in V_s} \mathcal{M}(I,q) = a^*]$
    - **Graded P'Robust Score**: 计算每组变换中正确回答的比例平均值

### 数据构建流程

- **数据来源**: 网络采集的建筑/室内场景、真实拍摄的鱼眼/线性透视图像对、开源RPVP数据集、Blender合成（含精确灭点坐标真值）
- **标注**: 混合流水线——PTS手工标注、LDP随机组合记录、PTR/LRR/VAP/CLP/VPC网络图像+手工标注、VPP结合网络与Blender图像
- **质量控制**: 多阶段审核，至少两名标注者独立标注主观任务，排除模糊样本

## 实验关键数据

### 主实验

| 模型 | VPP | CLP | VAP | LDP | PTR | LRR | OVR | PTS | VPC | Overall | Graded Robust |
|------|-----|-----|-----|-----|-----|-----|-----|-----|-----|---------|---------------|
| InternVL2.5-8B | 38.5 | 17.9 | 53.1 | 75.4 | 40.8 | 48.3 | 34.7 | 24.9 | 67.5 | 44.6 | 38.7 |
| Qwen2.5-VL-7B | 35.3 | 29.3 | 70.4 | 73.7 | 42.4 | 44.4 | 32.1 | 28.6 | 44.7 | 44.5 | 33.2 |
| InternVL2.5-26B | 41.7 | 35.0 | 55.6 | 81.8 | 65.5 | 46.4 | 43.5 | 34.3 | 46.5 | 50.0 | 52.9 |
| Eagle-X4-8B | 39.1 | 17.1 | 46.9 | 47.7 | 65.3 | 37.1 | 18.2 | 32.9 | 68.4 | 41.4 | 60.7 |
| InternVL2.5-2B | 47.4 | 22.8 | 13.0 | 65.3 | 62.2 | 31.8 | 16.6 | 30.0 | 50.0 | 37.7 | 59.1 |

### 消融实验

| 分析维度 | 关键发现 |
|---------|---------|
| 模型规模 | 更大模型在推理任务上通常更好，但鲁棒性与规模无明确正相关 |
| 感知 vs 推理 | 模型在表层感知任务上表现尚可，但推理和鲁棒性任务上明显退化 |
| 开源 vs 闭源 | GPT-4o等闭源模型总体领先，但仍远非完美 |
| CoT Prompting | Chain-of-thought提示在部分任务上有帮助 |

### 关键发现

- 所有43个SOTA模型在透视推理和鲁棒性上均表现不佳，即使是GPT-4o也存在显著局限
- 模型在表层感知任务（如LDP、VPC）上表现相对较好，但在组合推理（如OVR、PTS）和鲁棒性一致性上大幅退化
- 简单的几何保持变换（翻转、裁剪）就能严重干扰模型预测，说明模型缺乏真正的几何理解
- 模型架构和规模与透视能力之间存在有趣的非单调关系

## 亮点与洞察

- **首创性**: 第一个专门针对透视理解设计的MLLM基准，填补了几何感知评估的空白
- **层次化设计**: 感知→推理→鲁棒性的三维评估框架非常系统，难度递进合理
- **大规模评估**: 43个模型的全面评测提供了丰富的分析维度
- **数据多样性**: 结合真实拍摄、网络采集、合成渲染等多种数据源，保证评估的全面性
- **Blender合成创新**: 使用Claude 3.7 Sonnet + Blender-MCP自动生成带精确灭点标注的合成数据

## 局限与展望

- 主要聚焦于多选题格式，可以扩展到开放式问答或生成式评估
- 合成数据与真实场景的领域差距需要进一步研究
- 鲁棒性只考虑了几何保持变换，可以引入更多类型的扰动
- 未涉及视频中的透视理解或动态场景
- 可以结合透视理解训练数据来改进模型能力，而非仅评估

## 相关工作与启发

- 与3D空间理解基准（SpatialBench、ScanQA）互补，专注于2D图像中的透视几何
- 启发了将经典计算机视觉的几何先验引入MLLM训练的研究方向
- 说明"涌现能力"并不意味着系统性的空间认知，基础几何理解仍需专门设计

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个透视理解专用基准，问题定义和任务设计极具创新性
- 实验充分度: ⭐⭐⭐⭐⭐ 43个模型的大规模评测，多维度分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，从几何基础知识到任务定义逻辑通顺
- 价值: ⭐⭐⭐⭐ 为提升MLLM的空间几何能力提供了重要评测工具和方向指引

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Struct2D: A Perception-Guided Framework for Spatial Reasoning in MLLMs](struct2d_a_perception-guided_framework_for_spatial_reasoning_in_mllms.md)
- [\[ICLR 2026\] SpinBench: Perspective and Rotation as a Lens on Spatial Reasoning in VLMs](../../ICLR2026/multimodal_vlm/spinbench_perspective_and_rotation_as_a_lens_on_spatial_reasoning_in_vlms.md)
- [\[NeurIPS 2025\] Rethinking Multimodal Learning from the Perspective of Mitigating Classification Ability Disproportion](rethinking_multimodal_learning_from_the_perspective_of_mitig.md)
- [\[CVPR 2026\] Is the Modality Gap a Bug or a Feature? A Robustness Perspective](../../CVPR2026/multimodal_vlm/is_the_modality_gap_a_bug_or_a_feature_a_robustness_perspective.md)
- [\[ICCV 2025\] Perspective-Aware Reasoning in Vision-Language Models via Mental Imagery Simulation](../../ICCV2025/multimodal_vlm/perspective-aware_reasoning_in_vision-language_models_via_mental_imagery_simulat.md)

</div>

<!-- RELATED:END -->
