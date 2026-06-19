---
title: >-
  [论文解读] MME-VideoOCR: Evaluating OCR-Based Capabilities of Multimodal LLMs in Video Scenarios
description: >-
  [NeurIPS 2025][多模态VLM][OCR] 提出 MME-VideoOCR，一个包含 25 个任务、44 个场景、1464 个视频和 2000 个人工标注 QA 对的视频 OCR 综合评估基准，涵盖文本识别、理解和推理三个层次。评估 18 个 SOTA MLLM 揭示最强模型（Gemini-2.5 Pro）仅达 73.7%，跨帧理解任务低至 25% 以下。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "OCR"
  - "benchmark"
  - "cross-frame understanding"
  - "language prior bias"
  - "多模态"
---

# MME-VideoOCR: Evaluating OCR-Based Capabilities of Multimodal LLMs in Video Scenarios

**会议**: NeurIPS 2025  
**arXiv**: [2505.21333](https://arxiv.org/abs/2505.21333)  
**代码**: [https://mme-videoocr.github.io/](https://mme-videoocr.github.io/)  
**领域**: 多模态VLM / 视频理解 / OCR评估  
**关键词**: video OCR, benchmark, cross-frame understanding, language prior bias, multimodal LLM evaluation

## 一句话总结
提出 MME-VideoOCR，一个包含 25 个任务、44 个场景、1464 个视频和 2000 个人工标注 QA 对的视频 OCR 综合评估基准，涵盖文本识别、理解和推理三个层次。评估 18 个 SOTA MLLM 揭示最强模型（Gemini-2.5 Pro）仅达 73.7%，跨帧理解任务低至 25% 以下。

## 研究背景与动机

**领域现状**：MLLM 在静态图像 OCR 上已取得不错效果，但视频 OCR 面临运动模糊、时序变化、视觉特效等独特挑战，性能显著下降。

**现有 benchmark 局限**：
   - OCR Benchmark：仅 25 个视频、1 种任务类型，缺乏多样性
   - FG Bench：1028 个视频但使用自动+人工混合标注，仅 6 种任务
   - 两者都偏重文字感知，忽视基于文字的理解和推理

**视频 OCR 的三大挑战**：
   - (1) 文字以多种形式出现（前景、背景、弹幕、水印等），需建立时空视觉-文本关联
   - (2) 关键文字信息分布在多帧中，需跨帧聚合和时序理解
   - (3) 任务复杂度上升时需对识别的文字进行推理

## 方法详解

### 任务体系（10 大类 25 个子任务）

1. **文本识别**：指定位置识别、指定属性识别
2. **视觉文本问答**：以文本为中心的 QA、翻译
3. **文本定位**：空间定位、时间定位
4. **属性识别**：颜色识别、命名实体识别、计数
5. **变化检测与追踪**：变化检测、文字追踪
6. **特殊文本解析**：表格/图表/文档/数学公式/手写体解析
7. **跨帧文本理解**：滚动文字理解、轨迹识别、乱序拼合
8. **基于文本的推理**：综合散布线索、识别隐含关系、解决歧义
9. **基于文本的视频理解**：字幕视频理解、多跳大海捞针
10. **鲁棒性测试**：AIGC 视频、长视频、对抗视频

### 数据构建

**视频来源**（三种途径）：
- 从已有数据集（BOVText、RoadTextVQA 等）重构：GPT-4o 评估视觉动态+文字语义质量，通过筛选
- 人工从公开平台（YouTube、B站、快手）收集
- AI 生成（Wan 文生视频模型）：2000 短语→场景描述→视频生成→筛选

**标注流程**：
- 人工标注（非模型标注）：每视频 3-4 个 QA 对 → 专家二轮筛选保留 1-2 个高质量对
- 专家验证：审查歧义问题、不准确答案、难度不足的问题
- 选项均匀分布 + 去偏测试

**去偏测试**：无视觉输入情况下模型仅凭文字先验的准确率应接近随机水平（Containment Match 0%、选择题 25.1%），验证排除知识泄露和文字先验偏差。

### 评估方式
- **包含匹配**（Containment Match）：文本识别、手写识别任务
- **GPT 辅助评分**：翻译等多答案任务
- **选择题**：其他理解和推理任务

## 实验关键数据

### 主要结果（18 个模型）

| 模型 | 规模 | TR | VTQA | TG | AR | CDT | STP | CFTU | TBR | TBVU | RVT | 总分 |
|------|------|-----|------|-----|-----|-----|-----|------|-----|------|-----|------|
| Gemini-2.5 Pro | - | 83.0 | 91.6 | 64.5 | 74.0 | 70.0 | 84.4 | 48.7 | 74.0 | 56.5 | 72.0 | **73.7** |
| GPT-4o | - | 83.3 | 81.6 | 60.5 | 74.7 | 51.5 | 68.0 | 30.7 | 60.7 | 59.0 | 75.3 | 66.4 |
| Qwen2.5-VL | 72B | 80.7 | 80.0 | 65.0 | 74.0 | 56.5 | 79.6 | 26.7 | 74.7 | 57.0 | 78.7 | 69.0 |
| InternVL3 | 78B | 70.0 | 77.6 | 67.5 | 76.0 | 65.5 | 71.6 | 24.7 | 77.3 | 57.0 | 75.3 | 67.2 |
| InternVL3 | 8B | 61.3 | 72.0 | 60.0 | 69.3 | 56.5 | 62.4 | 23.3 | 57.3 | 55.0 | 71.3 | 59.8 |
| LLaVA-OneVision | 7B | 42.0 | 50.0 | 49.0 | 54.0 | 41.0 | 46.4 | 20.0 | 45.3 | 52.0 | 60.0 | 46.0 |

### 细粒度任务分析（Top-5 模型）

| 任务 | Gemini-2.5 Pro | Qwen2.5-VL 72B | InternVL3 78B | GPT-4o |
|------|---------------|----------------|---------------|--------|
| 轨迹识别 | **0.0%** | 0.0% | 0.0% | 0.0% |
| 乱序拼合 | 76.0% | 16.0% | 4.0% | 30.0% |
| 多跳大海捞针 | 27.0% | 18.0% | 18.0% | 25.0% |
| 字幕视频理解 | 86.0% | 96.0% | 96.0% | 93.0% |
| 翻译 | 84.0% | 66.0% | 68.0% | 70.0% |

### 关键发现

1. **跨帧理解是最大短板**：Cross-Frame Text Understanding 大多数模型 <25%，所有 Top-5 模型在**轨迹识别**上均为 **0%**
2. **分辨率和帧数至关重要**：提高分辨率和帧数持续提升性能，但帧数从 32→64 时部分模型反而下降（注意力分散）
3. **Token 压缩不适合 OCR**：VideoChat-Flash、Slow-fast MLLM 等压缩方法在 OCR 任务上表现劣势
4. **语言先验偏差严重**：模型倾向将拼写错误"修正"为语义合理的词（如 "throuh" → "through"），而非忠实识别视觉内容
5. **单帧 vs 跨帧差距悬殊**：字幕理解（单帧信息）90%+ vs 多跳大海捞针（跨帧聚合）<30%，说明模型依赖少量帧而非真正整合时序信息
6. **显著的 scaling 效应**：Qwen2.5-VL 7B→72B 提升 10%+，InternVL3 8B→78B 提升 7%+

## 亮点
- ⭐⭐⭐⭐ **任务设计全面**：25 个子任务覆盖感知→理解→推理完整链路，含跨帧理解和鲁棒性等创新维度
- ⭐⭐⭐⭐ **去偏设计严谨**：去偏测试+选项均衡+多轮专家审核，排除文字先验和知识泄露
- ⭐⭐⭐⭐ **发现有价值**：轨迹识别 0%、语言先验偏差、token 压缩缺陷等发现直接指导模型优化方向
- ⭐⭐⭐ **纯人工标注**：区别于混合标注的 benchmark，质量更可控

## 局限与展望
1. 总量 2000 QA 对，部分子类别样本数有限（如轨迹识别仅约 50 个），可能导致分数波动
2. 主要覆盖中英双语，未包含更多语种
3. 难度分层（易/中/难）设计中，前沿模型在易中难度上表现较好，需持续补充高难度样本
4. 未评估模型的视频 OCR 微调后效果——该 benchmark 是否适合作为训练目标尚不清楚
5. 对抗视频仅采用全黑帧插入策略，对抗形式较单一

## 总评
⭐⭐⭐⭐ 视频 OCR 领域急需的综合评估基准。任务设计维度丰富、标注质量高、去偏处理严谨。揭示的跨帧理解瓶颈和语言先验偏差等问题对 MLLM 优化具有直接指导意义。benchmark 的区分度和挑战性都很强。

## 与相关工作的对比

## 启发与关联

## 评分

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Hidden in Plain Sight: Evaluation of the Deception Detection Capabilities of LLMs in Multimodal Settings](../../ACL2025/multimodal_vlm/hidden_in_plain_sight_evaluation_of_the_deception_detection_capabilities_of_llms.md)
- [\[NeurIPS 2025\] Towards Evaluating Proactive Risk Awareness of Multimodal Language Models](towards_evaluating_proactive_risk_awareness_of_multimodal_language_models.md)
- [\[ACL 2025\] VF-Eval: Evaluating Multimodal LLMs for Generating Feedback on AIGC Videos](../../ACL2025/multimodal_vlm/vf_eval_aigc_video_feedback.md)
- [\[CVPR 2025\] Multimodal OCR: Parse Anything from Documents](../../CVPR2025/multimodal_vlm/multimodal_ocr_parse_anything_from_documents.md)
- [\[NeurIPS 2025\] WearVQA: A Visual Question Answering Benchmark for Wearables in Egocentric Authentic Real-world scenarios](wearvqa_a_visual_question_answering_benchmark_for_wearables_in_egocentric_authen.md)

</div>

<!-- RELATED:END -->
