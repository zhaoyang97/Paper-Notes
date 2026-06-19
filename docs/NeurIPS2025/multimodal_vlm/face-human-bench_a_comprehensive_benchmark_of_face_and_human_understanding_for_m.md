---
title: >-
  [论文解读] Face-Human-Bench: A Comprehensive Benchmark of Face and Human Understanding for Multi-modal Assistants
description: >-
  [NeurIPS 2025][多模态VLM][多模态大模型] 提出 Face-Human-Bench，首个系统评估多模态大模型人脸与人体理解能力的基准，包含三级能力分类体系（2个L1 × 10个L2 × 18个L3），开发集与测试集各 1800 题，支持中英双语，评测 25 个主流 MLLM 并揭示其与专家模型的显著差距。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "多模态大模型"
  - "基准评测"
  - "人脸理解"
  - "人体理解"
  - "MLLM"
  - "能力分级"
---

# Face-Human-Bench: A Comprehensive Benchmark of Face and Human Understanding for Multi-modal Assistants

**会议**: NeurIPS 2025  
**arXiv**: [2501.01243](https://arxiv.org/abs/2501.01243)  
**代码**: [项目主页](https://face-human-bench.github.io/)  
**领域**: 多模态基准 / 人脸人体理解  
**关键词**: 多模态大模型, 基准评测, 人脸理解, 人体理解, MLLM, 能力分级

## 一句话总结
提出 Face-Human-Bench，首个系统评估多模态大模型人脸与人体理解能力的基准，包含三级能力分类体系（2个L1 × 10个L2 × 18个L3），开发集与测试集各 1800 题，支持中英双语，评测 25 个主流 MLLM 并揭示其与专家模型的显著差距。

## 研究背景与动机

**领域现状**：人脸和人体是社交互动的核心元素，广泛出现在日常照片和视频中。多模态助手若能深度理解人脸人体信息，将大幅提升响应质量和应用范围。

**现有痛点**：(a) 现有多模态基准（MME、SEED-Bench、MMBench 等）仅涵盖少量人脸人体能力（如名人识别、动作识别），许多重要能力未被评估；(b) 人脸人体领域虽有大量专用数据集（CelebA、LFW 等），但未系统整合为评估 MLLM 的统一基准。

**核心矛盾**：MLLM 在通用视觉任务上表现出色，但在 deepfake 检测、人群计数、跨姿态人脸识别等专业任务上与专家模型差距如何？缺乏系统化度量。

**本文目标**：构建首个专门评估 MLLM 人脸和人体理解的全面基准。

## 方法详解

### 三级能力分类体系

**Level-1 (L1)**: 两个维度
- 目标维度：人脸理解 vs 人体理解  
- 认知过程：感知 (Perception) vs 推理 (Reasoning)

**Level-2 (L2)**: 10 项能力
- 人脸：面部属性识别、年龄估计、面部表情识别、人脸攻击检测、人脸识别
- 人体：人体属性识别、动作识别、空间关系理解、社会关系理解、行人重识别

**Level-3 (L3)**: 18 项细粒度能力
- 表情 → 基础 / 复合表情
- 攻击检测 → Deepfake 检测 / 活体检测
- 人脸识别 → 基础 / 跨姿态 / 跨年龄 / 相似面孔 / 遮挡
- 空间关系 → 相对位置 / 计数
- 社会关系 → 社交关系识别 / 身份推理

### 数据生成流水线

1. **源数据**：从 16 个公开数据集收集图像和标注
2. **图像处理 $p_{image}$**：裁剪、拼接、框标注或保持原图
3. **文本处理 $p_{text}$**：将标签转化为 1 个正确选项 + $n-1$ 个干扰选项（借助 ChatGPT 生成干扰选项）
4. **质量控制**：人工审核确保选项无歧义且有唯一正确答案
5. **题目格式**：多项选择题 $(V_i, Q_i, O_i, A_i)$，2-4 个选项

### 评估设计
- 加权准确率，L2 各能力等权
- 选项打乱防止模型偏好特定字母
- 约束指令确保模型输出选项字母
- 正则提取 + ChatGPT 后备提取答案

### 新指标：RPSS
相对位置敏感度得分 (Relative Position Sensitivity Score)，度量目标在图中不同位置时模型性能的波动程度。

## 实验关键数据

### 25 个 MLLM 总体表现

| 模型 | 参数量 | 人脸 | 人体 | 感知 | 推理 | 总分 |
|------|--------|------|------|------|------|------|
| GPT-4o | - | 72.5 | 73.6 | ~68 | ~70 | ~70 |
| InternVL-Chat-v1.2-Plus | - | 67.0 | 70.8 | ~66 | ~68 | ~67 |
| LLaVA-NeXT-34B | - | 71.0 | 72.0 | ~65 | ~66 | ~66 |
| Gemini-1.5-Pro | - | 60.0 | 85.6 | ~60 | ~62 | ~61 |
| MiniGPT-4-7B | - | 25.0 | 29.0 | ~24 | ~34 | ~28 |
| Random | - | 25-50 | 25-50 | 29.2 | 37.5 | 32.5 |

### 关键发现

**Q1 性能分析**:
- 最佳开源模型 InternVL-Chat-v1.2-Plus 在零样本下超越最佳闭源 GPT-4o
- 面部属性识别：InternLM-XComposer2-VL-7B 达 92.0（最高）
- 人脸识别整体：Gemini-1.5-Pro 以 85.6 大幅领先
- 位置敏感度 RPSS：InternLM-XComposer2-VL-7B 最稳定
- CoT 提示显著提升 GPT-4o 但对开源模型无效

**Q2 专家模型 vs MLLM**:

| 任务 | 最佳 MLLM | 专家模型 | 差距 |
|------|-----------|----------|------|
| Deepfake 检测 | ~64% | >90% | 显著 |
| 人群计数 | ~35% | >80% | 显著 |
| 人脸识别(困难场景) | ~55% | >95% | 显著 |

### 能力相关性
- L2 和 L3 级别存在显著正相关能力组
- 人脸属性与表情识别正相关
- 空间关系与社会关系正相关

## 亮点与洞察
- **首个系统性基准**：三级分类覆盖 18 项细粒度能力，填补了 MLLM 在人脸人体评估的空白
- **中英双语支持**：便于跨语言评估
- **实用洞察丰富**：(a) 开源可超闭源；(b) 位置敏感度是被忽视的重要维度；(c) CoT 对闭源有效对开源无效
- **明确了人-机能力鸿沟**：deepfake 检测、人群计数、困难人脸识别三大任务中 MLLM 远不如专家模型，指明了改进方向

## 局限与展望
- 仅评估静态图片，未覆盖视频中的人脸人体理解
- 多选题格式限制了开放式理解能力的评估
- 数据来源为公开数据集，可能存在数据泄露（模型训练数据包含部分测试集）
- L3 各能力样本量不均，部分能力统计显著性有限
- 未评估最新模型（如 GPT-4.1、Qwen2.5-VL、InternVL3）
- 社会关系理解的标注主观性较强

## 相关工作对比
- **vs MMBench/SEED-Bench**: 通用基准仅含少量人脸人体题目；Face-Human-Bench 专注且全面
- **vs CelebA/LFW**: 专家数据集评估专家模型；本工作将其转化为评估 MLLM 的格式
- **vs FaceCaption-15M**: 人脸描述数据集；不同于评估基准

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个系统化的人脸人体理解 MLLM 基准
- 实验充分度: ⭐⭐⭐⭐⭐ 25 个模型，18 项能力，多维分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，分类体系合理
- 价值: ⭐⭐⭐⭐ 为 MLLM 在人脸人体方向的发展提供了清晰路线图

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Rethinking Vision-Language Model in Face Forensics: Multi-Modal Interpretable Forged Face Detector](../../CVPR2025/multimodal_vlm/rethinking_vision-language_model_in_face_forensics_multi-modal_interpretable_for.md)
- [\[NeurIPS 2025\] DanmakuTPPBench: A Multi-modal Benchmark for Temporal Point Process Modeling and Understanding](danmakutppbench_a_multimodal_benchmark_for_temporal_point_pr.md)
- [\[NeurIPS 2025\] CAPability: A Comprehensive Visual Caption Benchmark for Evaluating Both Correctness and Thoroughness](capability_a_comprehensive_visual_caption_benchmark_for_eval.md)
- [\[ICLR 2026\] Can Vision-Language Models Answer Face to Face Questions in the Real-World?](../../ICLR2026/multimodal_vlm/can_vision-language_models_answer_face_to_face_questions_in_the_real-world.md)
- [\[NeurIPS 2025\] Scene-Aware Urban Design: A Human-AI Recommendation Framework Using Co-Occurrence Embeddings and Vision-Language Models](scene-aware_urban_design_a_human-ai_recommendation_framework_using_co-occurrence.md)

</div>

<!-- RELATED:END -->
