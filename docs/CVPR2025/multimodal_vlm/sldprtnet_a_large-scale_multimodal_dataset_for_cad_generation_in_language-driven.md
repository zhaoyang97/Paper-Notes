---
title: >-
  [论文解读] SldprtNet: A Large-Scale Multimodal Dataset for CAD Generation in Language-Driven 3D Design
description: >-
  [CVPR 2025][多模态VLM][CAD数据集] 本文构建了一个包含24万+工业零件的大规模多模态CAD数据集 SldprtNet，每个样本对齐了3D模型、多视角图像、参数化建模脚本和自然语言描述四种模态，并开发了支持13种CAD操作的编码器/解码器工具实现无损双向转换，实验证明多模态输入显著优于纯文本输入。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "CAD数据集"
  - "多模态"
  - "Text-to-CAD"
  - "参数化建模"
  - "SolidWorks"
---

# SldprtNet: A Large-Scale Multimodal Dataset for CAD Generation in Language-Driven 3D Design

**会议**: CVPR 2025  
**arXiv**: [2603.13098](https://arxiv.org/abs/2603.13098)  
**代码**: 未公开  
**领域**: 其他  
**关键词**: CAD数据集, 多模态, Text-to-CAD, 参数化建模, SolidWorks

## 一句话总结

本文构建了一个包含24万+工业零件的大规模多模态CAD数据集 SldprtNet，每个样本对齐了3D模型、多视角图像、参数化建模脚本和自然语言描述四种模态，并开发了支持13种CAD操作的编码器/解码器工具实现无损双向转换，实验证明多模态输入显著优于纯文本输入。

## 研究背景与动机

**领域现状**：CAD（计算机辅助设计）在机械设计和制造中至关重要。现有的3D模型数据集如 ShapeNet、ModelNet 主要以 mesh 或点云形式存储，只保留了最终表面形状而丢失了建模历史和参数化信息。少数参数化数据集如 ABC、Fusion 360 Gallery 虽然保留了几何精度，但缺少语义级别的文本标注。

**现有痛点**：Text-to-CAD 建模面临三个核心问题：(1) 数据量小——CAD 数据集需要专业人员手动创建，远小于图像/文本数据集；(2) 模态缺失——现有数据集通常只覆盖单一模态（几何、序列或文本），无法支持跨模态学习；(3) 操作类型受限——DeepCAD 和 Text2CAD 只支持草图+拉伸两种操作，覆盖的零件类型非常有限。

**核心矛盾**：现代多模态模型（如 CLIP、Flamingo、BLIP-2）已经证明跨模态对齐学习对泛化和迁移至关重要，但 CAD 领域缺少一个真正对齐多种模态的大规模数据集来支撑这些方法。

**本文目标** 构建一个支持多模态、双向转换、语义标注、可编辑且人类可读的大规模 CAD 数据集。

**切入角度**：作者利用 SolidWorks API 开发编码器/解码器工具，将原生 .sldprt 文件转换为结构化文本，同时渲染多视角图像并用多模态 LLM 生成自然语言描述，实现四种模态的完全对齐。

**核心 idea**：用SolidWorks API构建一套闭环工具链，从24万+工业零件中提取对齐的3D模型、图像、参数化脚本和自然语言描述，形成支持 Text-to-CAD 的多模态数据集。

## 方法详解

### 整体框架

SldprtNet 的构建 pipeline 包含四个阶段：(1) 从 GrabCAD、McMaster-Carr、FreeCAD 三大平台收集约68万 .sldprt 文件；(2) 过滤保留包含至少一种13类特征的24万+高质量模型；(3) 通过自动化工具提取四种模态数据；(4) 使用多模态 LLM 生成并人工校验自然语言描述。

### 关键设计

1. **编码器（CAD → Text）**:

    - 功能：将 .sldprt 文件无损转换为结构化参数文本
    - 核心思路：自动遍历 Feature Tree，按建模历史顺序提取特征类型、名称和父子关系，然后对每种特征调用对应模块提取详细参数（尺寸、约束、草图实体等），生成人类和机器都可读的文本格式
    - 设计动机：支持13种 CAD 操作（包括拉伸、倒角、圆角、线性阵列、镜像阵列等），远超 DeepCAD 的2种操作，大幅扩展了可覆盖的零件复杂度和多样性

2. **解码器（Text → CAD）**:

    - 功能：从参数化文本重建完整的3D零件模型
    - 核心思路：先创建空白 .sldprt 文档，解析 Feature Tree，按特征顺序和层级关系依次调用 SolidWorks API 创建对应特征，确保几何和拓扑一致性
    - 设计动机：与编码器构成闭环系统，支持"模型→文本→模型"的往返转换，可用于结构验证、数据增强和合成数据生成

3. **多模态对齐生成**:

    - 功能：为每个3D模型生成对齐的多视角合成图像和自然语言描述
    - 核心思路：渲染6个正交视图（前/后/左/右/上/下）+ 1个等轴视图，合并为单张图像以减少 token 数量；将合成图像 + 参数化文本输入 Qwen2.5-VL-7B 生成描述，再经人工验证对齐
    - 设计动机：利用12块 A100 GPU 共368 GPU-小时完成24万+样本的描述生成，七视图合成图既保证了几何完整性又优化了推理效率

### 损失函数 / 训练策略

基线实验使用标准的语言模型微调策略，在5万样本子集上分别微调 Qwen2.5-7B（纯文本）和 Qwen2.5-7B-VL（图像+文本），使用 Exact Match Score、BLEU Score、Command-Level F1 等指标评估。

## 实验关键数据

### 主实验

| 指标 | Qwen2.5-7B (纯文本) | Qwen2.5-7B-VL (图+文) | 提升 |
|------|---------------------|----------------------|------|
| Exact Match Score | 0.0058 | 0.0099 | +70.7% |
| BLEU Score | 97.18 | 97.93 | +0.77% |
| Command-Level F1 | 0.3247 | 0.3670 | +13.0% |
| Partial Match Rate | 0.5554 | 0.6162 | +10.9% |
| Tolerance Accuracy | 0.5016 | 0.4630 | -7.7% |

### 数据集统计

| 复杂度等级 | 特征数量 | 样本数 | 比例 |
|-----------|---------|--------|------|
| Level 1 (Simple) | 1-5 | 93,188 | ~38.4% |
| Level 2 (Moderate) | 6-10 | 78,926 | ~32.5% |
| Level 3 (Advanced) | 11-100 | 69,259 | ~28.5% |
| Level 4 (Expert) | 100+ | 1,234 | ~0.5% |

### 关键发现
- 多模态输入（图像+文本）在 Exact Match、F1、Partial Match 三个结构对齐指标上全面优于纯文本输入，验证了视觉信息对 CAD 语义理解的重要性
- 纯文本模型在 Tolerance Accuracy 上略优，可能因为过拟合于数值参数而非结构语义
- 2D Sketch 是使用频率最高的特征类型，Chamfer 和 Fillet 也很常见，反映数据集的工业导向

## 亮点与洞察
- **闭环编解码设计**非常巧妙——支持模型→文本→模型的无损往返，可用于自动验证生成结果的正确性，也方便数据增强和规模扩展
- **七视图合成为单图**的策略既减少了输入 token 长度又保留了完整几何信息，适合多模态模型推理
- 支持13种 CAD 操作的参数化表示是对 DeepCAD（仅2种）的重大升级，使数据集能覆盖真实工业零件的复杂度

## 局限与展望
- 论文发表在 ICRA 而非典型的 CAD/Vision 顶会，数据集的实际 CAD 生成效果的评估还比较初步（仅做了简单 baseline 对比）
- 编码器/解码器依赖 SolidWorks 商业软件的 API，难以复现和扩展到开源 CAD 平台
- Baseline 实验使用的数据子集仅5万条（占总量约20%），未展示全量训练效果
- 自然语言描述虽经人工验证，但24万+的量级下人工校验的覆盖率和质量存疑

## 相关工作与启发
- **vs DeepCAD**: DeepCAD 开创了将 CAD 建模视为序列生成问题的范式，但仅支持草图+拉伸2种操作，且缺少自然语言输入。SldprtNet 在操作种类（13种）和模态丰富度上全面升级
- **vs Text2CAD**: Text2CAD 虽然加入了文本描述，但其描述是从建模序列合成的而非从视觉信息生成，容易与实际几何产生语义偏差。SldprtNet 利用多模态 LLM 从图像+参数同时生成描述，对齐质量更高
- **vs ABC/Fusion 360**: ABC 规模大（100万+B-Rep）但缺少序列和文本标注；Fusion 360 有建模历史但规模小且无文本标注。SldprtNet 在模态完整性上是最全面的

## 评分
- 新颖性: ⭐⭐⭐ 数据集构建工作为主，方法创新有限
- 实验充分度: ⭐⭐⭐ 仅有简单 baseline 对比，缺少更多下游任务验证
- 写作质量: ⭐⭐⭐⭐ 结构清晰，数据处理流程描述详细
- 价值: ⭐⭐⭐⭐ 填补了多模态 CAD 数据集的空白，对 Text-to-CAD 研究有重要支撑价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] SketchAgent: Language-Driven Sequential Sketch Generation](sketchagent_language-driven_sequential_sketch_generation.md)
- [\[CVPR 2026\] CADFS: A Big CAD Program Dataset and Framework for Computer-Aided Design with Large Language Models](../../CVPR2026/multimodal_vlm/cadfs_a_big_cad_program_dataset_and_framework_for_computer-aided_design_with_lar.md)
- [\[CVPR 2025\] CoMM: A Coherent Interleaved Image-Text Dataset for Multimodal Understanding and Generation](comm_a_coherent_interleaved_image-text_dataset_for_multimodal_understanding_and_.md)
- [\[CVPR 2025\] Active Data Curation Effectively Distills Large-Scale Multimodal Models](active_data_curation_effectively_distills_large-scale_multimodal_models.md)
- [\[CVPR 2025\] Scalable Video-to-Dataset Generation for Cross-Platform Mobile Agents](scalable_video-to-dataset_generation_for_cross-platform_mobile_agents.md)

</div>

<!-- RELATED:END -->
