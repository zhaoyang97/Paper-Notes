---
title: >-
  [论文解读] RAP: Retrieval-Augmented Personalization for Multimodal Large Language Models
description: >-
  [CVPR 2025][多模态VLM][个性化多模态大模型] 提出 RAP（Retrieval-Augmented Personalization）框架，通过"记忆-检索-生成"三步实现 MLLM 的个性化：用外部数据库存储用户概念，用多模态检索器动态检索相关概念信息，再注入 MLLM 生成个性化响应，每个概念仅需1张图+描述即可，且支持实时更新。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "个性化多模态大模型"
  - "检索增强生成"
  - "用户概念记忆"
  - "个性化对话"
  - "数据构建"
---

# RAP: Retrieval-Augmented Personalization for Multimodal Large Language Models

**会议**: CVPR 2025  
**arXiv**: [2410.13360](https://arxiv.org/abs/2410.13360)  
**代码**: [https://hoar012.github.io/RAP-Project/](https://hoar012.github.io/RAP-Project/)  
**领域**: 多模态VLM  
**关键词**: 个性化多模态大模型, 检索增强生成, 用户概念记忆, 个性化对话, 数据构建

## 一句话总结

提出 RAP（Retrieval-Augmented Personalization）框架，通过"记忆-检索-生成"三步实现 MLLM 的个性化：用外部数据库存储用户概念，用多模态检索器动态检索相关概念信息，再注入 MLLM 生成个性化响应，每个概念仅需1张图+描述即可，且支持实时更新。

## 研究背景与动机

现有 MLLM 虽然在通用视觉理解上表现优异，但不具备用户特定知识（如宠物名字、朋友身份等），无法作为个性化助手。已有方法如 MyVLM 和 Yo'LLaVA 通过学习外部分类头或特殊 token 来记忆概念，但存在三大问题：(1) 需要大量标注图像（5张正样本+150~200张负样本）；(2) 每增加新概念都需要重新训练模型；(3) 无法实时编辑概念。RAP 的核心动机是借鉴 RAG 的思路，将概念知识从模型参数中解耦到外部数据库，实现"训练一次，适配无限用户"。

## 方法详解

### 整体框架

RAP 由三个阶段组成：**(a) Remember**：构建键值数据库存储每个概念的图像和描述；**(b) Retrieve**：用户发起对话时，用开放世界检测器检测图像中的 ROI，再用多模态检索器从数据库中检索相关概念；**(c) Generate**：将检索到的概念信息（图像+文本描述）与原始输入一起送入 MLLM 生成个性化响应。

### 关键设计

1. **外部概念数据库（Remember）**:
    - 功能：以键值对形式存储用户的个人概念，每个概念包含头像 $\mathbf{I}_j$、名称和简短描述 $\mathbf{T}_j$
    - 核心思路：概念的键 $k_j$ 是其图像经预训练图像编码器 $\mathcal{E}(\cdot)$ 提取的视觉特征。数据库支持实时增删概念，无需重新训练
    - 设计动机：将用户知识从模型参数解耦到外部存储，每个概念仅需1张图+描述，大幅降低个性化门槛

2. **多模态检索器（Retrieve）**:
    - 功能：在用户发起对话时，自动识别图像中可能的概念并从数据库中检索匹配项
    - 核心思路：使用 YOLO-World 作为通用检测器 $\mathcal{R}(\cdot)$ 检测 ROI，对每个 ROI 提取 CLIP 视觉特征 $v_i = \mathcal{E}(\mathbf{X}_u^i)$，计算与数据库所有键的欧氏距离 $Dist(v_i, k_j) = \|v_i - k_j\|$，选取 Top-K 最近邻。同时支持基于概念名称的文本检索
    - 设计动机：用通用检测器替代针对每个概念学习专用分类头，实现对无限新概念的泛化，无需重新训练

3. **个性化训练数据集构建流水线**:
    - 功能：为 MLLM 的个性化生成能力提供大规模训练数据
    - 核心思路：包含三类数据：(a) 视觉定位数据（RefCOCO + ILSVRC-VID + TAO + CustomConcept101），训练模型识别概念在图中的位置；(b) 指令跟随数据（图像描述、问答），用 Gemini-1.5 生成标注；(c) 负样本数据（在输入中加噪声概念但保持答案不变），训练模型的噪声过滤能力。此外用扩散模型做数据增强生成新视角
    - 设计动机：现有缺乏面向个性化的大规模训练数据，且模型需要学会"利用相关信息+忽略无关信息"

### 损失函数 / 训练策略

- 使用标准自回归语言建模损失 $\prod_{i=1}^{L} p_\theta(\mathbf{X}_{a,i} | \mathbf{X}_v, \mathbf{X}_q, \mathbf{M}_1, \cdots \mathbf{M}_K, \mathbf{X}_{a,<i})$
- 训练时冻结检测器和检索器参数，仅训练 MLLM（使用 LoRA 减少可训练参数）
- 基于 LLaVA-1.5-13B 和 Phi3-V-3.8B 训练，8xA100，batch size 64，学习率 1e-4，1个 epoch
- 保留部分 LLaVA-Instruct-665k 数据以保持通用知识

## 实验关键数据

### 主实验

| 任务 | 指标 | RAP-LLaVA | MyVLM | Yo'LLaVA | LLaVA-LoRA |
|------|------|-----------|-------|----------|------------|
| 个性化图像描述 | F1-score | **94.97** | 85.50 | - | 87.82 |
| 视觉问答 | Weighted Acc | **0.936** | - | 0.906 | 0.741 |
| 视觉识别 | Weighted Acc | **0.980** | 0.919 | 0.924 | 0.825 |
| 文本问答 | Acc | **0.938** | - | 0.883 | 0.583 |

| 对比方法 | 所需正样本数 | 所需负样本数 | 支持实时编辑 | 支持纯文本QA |
|---------|-----------|-----------|-----------|------------|
| RAP(Ours) | **1** | **0** | **✓** | **✓** |
| MyVLM | n | 150 | ✗ | ✗ |
| Yo'LLaVA | n | 200 | ✗ | ✓ |
| Fine-tuning | n | 0 | ✗ | ✓ |

### 消融实验

| 配置 | Recall | Precision | F1-score | 说明 |
|------|--------|-----------|----------|------|
| RAP-LLaVA (完整) | 93.51 | 96.47 | 94.97 | - |
| 跳过检索（给完美信息） | 96.16 | 100.0 | 98.04 | 检索瓶颈约3%性能 |
| 去掉文本信息 | 94.91 | 88.66 | 91.68 | 文本描述帮助精确匹配 |
| 去掉数据增强 | 89.25 | 98.01 | 93.42 | 增强提升recall |
| 去掉负样本训练 | 95.74 | 58.21 | 72.40 | 负样本对precision至关重要 |

### 关键发现

- 负样本训练是精度的关键：去掉后 Precision 从96.47%暴跌至58.21%，模型会将不相关概念也错误输出
- RAP 可与 GPT-4V 竞争：RAP-LLaVA 用1张图的表现（0.936）接近 GPT-4V 用5张图的表现（0.937）
- 数据库中概念数量增加时，RAP 的性能衰减最慢（得益于检索而非记忆的架构）
- 纯文本问答中，RAP-LLaVA（0.938）远超 LLaVA-LoRA（0.583），因为RAP可基于文本名称检索相关信息

## 亮点与洞察

- **最低数据需求**：每个概念仅需1张图+描述，无需负样本和重新训练，是目前数据效率最高的个性化方案
- **实时概念编辑**：通过修改外部数据库即可增删概念，无需模型更新，适合真实动态场景
- **RAG+个性化的首次结合**：将RAG思路引入MLLM个性化，是检索增强与用户个性化交叉的开创性工作
- **数据构建流水线**：系统性设计了包含定位、指令跟随、负样本三类数据的构建流程，可复用

## 局限与展望

- 检索器的准确度是系统瓶颈：在完美检索条件下 F1 可达98.04%，说明检索仍有~3%的提升空间
- 受限于LLM上下文长度，RAP-LLaVA仅能检索2个概念（RAP-Phi3-V为3个），限制了多概念场景
- 需要用户手动提供每个概念的描述信息，自动化描述生成可进一步降低用户负担
- 未在视频理解等时序场景中验证

## 相关工作与启发

- 与 MyVLM（外部分类头）和 Yo'LLaVA（学习special token）的核心区别：RAP将概念知识外化到数据库，避免了持续学习的计算开销
- RAG 在 NLP 中已被充分验证（DPR, Self-RAG），但在 MLLM 个性化领域是首次系统性应用
- 负样本训练策略对检索增强系统的鲁棒性至关重要，可推广到其他 RAG 场景

## 评分

- 新颖性: ⭐⭐⭐⭐ RAG+个性化是新组合，但各组件（检索、数据库、LoRA微调）本身是成熟技术
- 实验充分度: ⭐⭐⭐⭐ 覆盖了描述、问答、识别三大任务，消融充分，但评估数据集规模偏小
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机阐述到位，数据构建流水线描述详细
- 价值: ⭐⭐⭐⭐ 极低的数据需求和实时编辑能力使其在实际应用中非常有价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Windsock is Dancing: Adaptive Multimodal Retrieval-Augmented Generation](../../NeurIPS2025/multimodal_vlm/windsock_is_dancing_adaptive_multimodal_retrieval-augmented_generation.md)
- [\[CVPR 2025\] CoLLM: A Large Language Model for Composed Image Retrieval](collm_a_large_language_model_for_composed_image_retrieval.md)
- [\[ICLR 2026\] RAVENEA: A Benchmark for Multimodal Retrieval-Augmented Visual Culture Understanding](../../ICLR2026/multimodal_vlm/ravenea_a_benchmark_for_multimodal_retrieval-augmented_visual_culture_understand.md)
- [\[CVPR 2025\] On the Out-of-Distribution Generalization of Multimodal Large Language Models](on_the_out-of-distribution_generalization_of_large_multimodal_models.md)
- [\[CVPR 2025\] Cross-modal Information Flow in Multimodal Large Language Models](cross-modal_information_flow_in_multimodal_large_language_models.md)

</div>

<!-- RELATED:END -->
