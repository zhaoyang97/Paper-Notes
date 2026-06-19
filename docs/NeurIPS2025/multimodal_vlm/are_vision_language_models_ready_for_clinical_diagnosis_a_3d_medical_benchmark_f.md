---
title: >-
  [论文解读] Are Vision Language Models Ready for Clinical Diagnosis? A 3D Medical Benchmark for Tumor-centric Visual Question Answering
description: >-
  [NeurIPS 2025 (Datasets & Benchmarks Track)][多模态VLM][VQA] 本文提出 DeepTumorVQA，一个针对腹部CT肿瘤的3D诊断级视觉问答基准，包含9,262个CT体积（370万切片）和395K专家级问题，系统评估了4个先进VLM的临床诊断能力，发现当前模型在测量任务上尚可但在病灶识别和推理上远未达到临床要求。
tags:
  - "NeurIPS 2025 (Datasets & Benchmarks Track)"
  - "多模态VLM"
  - "VQA"
  - "3D医学影像"
  - "CT肿瘤"
  - "视觉语言模型"
  - "benchmark"
---

# Are Vision Language Models Ready for Clinical Diagnosis? A 3D Medical Benchmark for Tumor-centric Visual Question Answering

**会议**: NeurIPS 2025 (Datasets & Benchmarks Track)  
**arXiv**: [2505.18915](https://arxiv.org/abs/2505.18915)  
**代码**: [GitHub](https://github.com/Schuture/DeepTumorVQA)  
**领域**: 多模态VLM  
**关键词**: VQA, 3D医学影像, CT肿瘤, 视觉语言模型, benchmark

## 一句话总结

本文提出 DeepTumorVQA，一个针对腹部CT肿瘤的3D诊断级视觉问答基准，包含9,262个CT体积（370万切片）和395K专家级问题，系统评估了4个先进VLM的临床诊断能力，发现当前模型在测量任务上尚可但在病灶识别和推理上远未达到临床要求。

## 研究背景与动机

视觉语言模型（VLMs）在2D视觉任务上取得了显著进展，但其在3D临床诊断中的就绪程度仍不明确。3D医学诊断对模型提出三重严苛要求：

**识别精度**：需要在复杂3D解剖结构中精确定位微小病灶

**推理能力**：需要跨切片整合信息进行空间推理

**领域知识**：需要具备临床医学知识以支撑诊断

现有医学VQA基准主要聚焦2D图像（如X光、皮肤镜），缺乏针对3D CT扫描的系统评测。此外，现有基准的问题设计往往过于简单，无法反映真实临床工作流程中的诊断复杂性。

## 方法详解

### 整体框架

DeepTumorVQA 基准的构建流程：
1. **数据收集**：从17个公开数据集中汇集9,262个腹部CT体积，涵盖多种肿瘤类型和解剖器官
2. **问题生成**：设计四类诊断级问题，共395K个问答对
3. **质量控制**：由医学专家验证问题的临床合理性和答案准确性
4. **标准化评测**：建立统一的评价协议和指标体系

### 关键设计

**四类问题体系**：

| 类别 | 描述 | 示例 | 难度 |
|------|------|------|------|
| Recognition（识别） | 检测病灶存在、计数、分类 | "该CT中有几个肝脏肿瘤？" | 高 |
| Measurement（测量） | 测量病灶/器官的大小、HU值等 | "最大肿瘤的长径是多少mm？" | 中 |
| Visual Reasoning（视觉推理） | 需要空间推理的视觉问题 | "肿瘤是否侵犯了相邻血管？" | 高 |
| Medical Reasoning（医学推理） | 需要临床知识的推理问题 | "基于影像特征，最可能的病理类型是？" | 极高 |

**数据集规模与多样性**：
- 17个公开CT数据集，覆盖肝脏、胰腺、肾脏等多个腹部器官
- 包含多种肿瘤类型：肝细胞癌、胰腺癌、肾细胞癌、囊肿等
- 标注信息包括：体素间距、图像尺寸、患者性别/年龄、扫描仪类型、对比剂使用情况

**评测VLM列表**：
- **RadFM**：大规模多模态预训练的3D医学VLM
- **M3D**：面向3D医学图像理解的多模态模型
- **Merlin**：融合视觉和文本的3D医学模型
- **CT-CHAT**：专为CT图像对话设计的VLM

### 损失函数 / 训练策略

本文为基准测试论文，不涉及新模型训练。评测采用零样本和少样本设定，使用精确匹配（EM）和F1分数作为主要评价指标。对于选择题格式，使用选项准确率；对于开放题，使用GPT辅助评分。

## 实验关键数据

### 主实验

**四类任务上的整体性能比较**：

| 模型 | Recognition | Measurement | Visual Reasoning | Medical Reasoning | 总体 |
|------|------------|-------------|-----------------|-------------------|------|
| RadFM | **32.4** | **45.8** | **28.6** | **24.1** | **32.7** |
| M3D | 21.5 | 38.2 | 18.9 | 16.3 | 23.7 |
| Merlin | 19.8 | 41.3 | 15.2 | 14.7 | 22.8 |
| CT-CHAT | 24.1 | 36.7 | 20.4 | 18.5 | 24.9 |

**各子任务的细粒度分析**：

| 子任务 | RadFM | M3D | Merlin | CT-CHAT |
|--------|-------|-----|--------|---------|
| 病灶检测 | 28.3 | 15.7 | 12.4 | 18.9 |
| 病灶计数 | 18.9 | 10.2 | 8.6 | 12.1 |
| 器官HU测量 | 52.1 | 44.6 | 48.9 | 42.3 |
| 肿瘤大小测量 | 39.5 | 31.8 | 33.7 | 31.1 |
| 空间关系推理 | 25.4 | 16.3 | 13.8 | 18.7 |
| 病理类型分类 | 22.8 | 14.8 | 13.1 | 17.2 |

### 消融实验

- **图像预处理的影响**：不同的窗宽窗位设置（如肝窗 vs 肺窗 vs 腹部窗）显著影响模型在不同器官上的识别性能，最佳预处理因任务而异
- **视觉模块设计**：3D卷积编码器 vs 2D切片逐帧编码在肿瘤检测任务上有15-20%的性能差距
- **输入切片数量**：从16切片增加到64切片在测量任务上提升约8%，但在推理任务上提升有限

### 关键发现

1. **RadFM一枝独秀**：大规模多模态预训练（包括3D医学影像）是性能差异的主因，RadFM在所有类别上均领先
2. **测量 vs 识别的鸿沟**：所有模型在测量任务上相对较好（利用像素级信息），但在识别和推理上表现不佳
3. **小肿瘤检测是瓶颈**：直径小于2cm的肿瘤检测准确率普遍低于15%
4. **视觉模块关键**：图像预处理和3D感知能力是区分模型性能的核心因素

## 亮点与洞察

- **临床导向的评测设计**：四类问题体系模拟了真实放射科医生的工作流程，从识别到推理层层递进
- **规模空前**：9,262个CT体积、395K问答对是目前最大的3D医学VQA基准
- **暴露核心短板**：清晰揭示了当前VLM在3D医学诊断中的具体失败模式，为未来研究指明方向
- **可复现性强**：基于17个公开数据集构建，数据集已通过HuggingFace公开发布
- **实用启示**：表明单纯扩大2D预训练数据不足以解决3D医学理解问题，需要专门的3D架构和训练策略

## 局限与展望

1. **仅覆盖腹部CT**：未包含胸部、头颈部等其他解剖区域
2. **评测VLM数量有限**：仅测试了4个模型，未包含GPT-4V等通用VLM
3. **问题生成的自动化程度**：部分问题可能存在模板化倾向
4. **缺少人类基线**：未提供放射科医生在同一问题集上的表现作为参照
5. **未考虑多模态融合**：未结合临床文本（如影像报告）进行联合评测

## 相关工作与启发

- **PathVQA、SLAKE**：2D医学VQA基准的先驱，但限于2D
- **CT-RATE、RadBench**：3D医学基准的早期尝试，规模较小
- **Abdomen Atlas**：本文数据基础之一，提供了大规模腹部CT标注
- 本文强调了3D医学VLM研发中"数据规模 × 3D架构 × 临床知识"三角关系的重要性

## 评分

- 数据集价值: ⭐⭐⭐⭐⭐
- 评测设计: ⭐⭐⭐⭐⭐
- 分析深度: ⭐⭐⭐⭐
- 创新性: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] WearVQA: A Visual Question Answering Benchmark for Wearables in Egocentric Authentic Real-world scenarios](wearvqa_a_visual_question_answering_benchmark_for_wearables_in_egocentric_authen.md)
- [\[ACL 2025\] WikiMixQA: A Multimodal Benchmark for Question Answering over Tables and Charts](../../ACL2025/multimodal_vlm/wikimixqa_a_multimodal_benchmark_for_question_answering_over_tables_and_charts.md)
- [\[ACL 2025\] MTVQA: Benchmarking Multilingual Text-Centric Visual Question Answering](../../ACL2025/multimodal_vlm/mtvqa_benchmarking_multilingual_text-centric_visual_question_answering.md)
- [\[NeurIPS 2025\] Better Tokens for Better 3D: Advancing Vision-Language Modeling in 3D Medical Imaging](better_tokens_for_better_3d_advancing_vision-language_modeling_in_3d_medical_ima.md)
- [\[NeurIPS 2025\] FOCUS: Internal MLLM Representations for Efficient Fine-Grained Visual Question Answering](focus_internal_mllm_representations_for_efficient_fine-grained_visual_question_a.md)

</div>

<!-- RELATED:END -->
