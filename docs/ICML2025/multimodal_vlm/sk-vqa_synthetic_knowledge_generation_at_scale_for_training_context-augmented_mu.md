---
title: >-
  [论文解读] SK-VQA: Synthetic Knowledge Generation at Scale for Training Context-Augmented Multimodal LLMs
description: >-
  [ICML2025 Oral Spotlight][多模态VLM][知识型VQA] 利用 GPT-4 全自动生成包含 200 万+ QA 对的大规模合成 KB-VQA 数据集 SK-VQA，训练 MLLM 适配上下文增强生成，在跨域泛化性能上显著优于已有数据集。 核心问题： 现有 MLLM 并非为"上下文增强生成"（cont…
tags:
  - "ICML2025 Oral Spotlight"
  - "多模态VLM"
  - "知识型VQA"
  - "合成数据"
  - "多模态RAG"
  - "上下文增强生成"
  - "MLLM微调"
---

# SK-VQA: Synthetic Knowledge Generation at Scale for Training Context-Augmented Multimodal LLMs

**会议**: ICML2025 Oral Spotlight  
**arXiv**: [2406.19593](https://arxiv.org/abs/2406.19593)  
**代码**: [GitHub](https://github.com/UKPLab/SK-VQA) / [HuggingFace](https://huggingface.co/datasets/skvqa/SK-VQA)  
**领域**: 多模态VLM  
**关键词**: 知识型VQA, 合成数据, 多模态RAG, 上下文增强生成, MLLM微调

## 一句话总结
利用 GPT-4 全自动生成包含 200 万+ QA 对的大规模合成 KB-VQA 数据集 SK-VQA，训练 MLLM 适配上下文增强生成，在跨域泛化性能上显著优于已有数据集。

## 研究背景与动机

**核心问题**: 现有 MLLM 并非为"上下文增强生成"（context-augmented generation）设计，无法直接用于多模态 RAG 系统。要让 MLLM 在 RAG 场景下有效工作，需要大量包含「图像 + 问题 + 上下文文档」的训练数据，但这种天然配对的数据在互联网上极为稀缺。

**已有数据集的局限**:

- **ViQuAE**: 仅 3.7k QA 对，规模太小
- **InfoSeek**: 130 万 QA，但不到 1% 唯一，依赖模板构建，多样性极差
- **Enc-VQA**: 100 万 QA，仅 17% 唯一，图片仅来自 iNaturalist 和 Google Landmarks
- 上述数据集均受限于图片必须能链接到 Wikipedia 页面，覆盖领域窄，且模板生成导致语言风格单一

**动机**: 利用强大的基础模型（GPT-4）进行全自动合成数据生成，突破图片来源和问题多样性的瓶颈，构建能有效训练 MLLM 适配上下文增强生成的大规模数据集。

## 方法详解

### 3.1 数据生成管线

给定一张输入图片，用单条 prompt 驱动 GPT-4 同时生成：
1. **上下文文档**: 与图像相关的 Wikipedia 风格文章（不直接引用图像）
2. **多组 QA 对**: 需要联合图像和上下文文档进行推理才能作答

关键设计——**单步生成**: 上下文文档和 QA 对在一次推理中同时生成。这使得上下文的生成被「需要产出需要图像+上下文联合推理的 QA」这一任务所约束，确保上下文与 QA 的高度匹配。每张图平均生成 7.1 个 QA 对（GPT-4 上下文），对比 Wikipedia 上下文仅 5.7 个。

**图像来源**（三种，确保领域多样性）:

| 图像来源 | 上下文来源 | QA 对数 |
|---------|-----------|--------|
| LAION-400M | GPT-4 | 908,116 |
| Wikipedia (WIT) | GPT-4 | 702,332 |
| Wikipedia (WIT) | Wikipedia | 181,554 |
| COCO-Counterfactuals | GPT-4 | 214,487 |
| **合计** | | **2,006,489** |

### 3.2 Image Reference (IR) 过滤

GPT-4 有时在生成的上下文中直接引用输入图像（如 "In the image, …"）。这类上下文更像扩展 caption 而非知识文档，在真实 RAG 场景中不现实。通过检测上下文中是否出现 picture/photo/image/painting 等词进行过滤，得到 $\text{SK-VQA}_{\text{IR}}$（153 万 QA）。

### 3.3 Context Answer Presence (CAP) 过滤

进一步要求至少一个答案候选显式出现在上下文文档中，同时不直接引用图像，得到 $\text{SK-VQA}_{\text{IR+CAP}}$（98.5 万 QA）。该过滤提升数据质量——人类在该子集上准确率达 87%（vs 全集 77%）。

### 数据多样性分析

| 指标 | InfoSeek | Enc-VQA | SK-VQA |
|-----|----------|---------|--------|
| 总 QA 数 | 1,356K | 1,036K | 2,006K |
| 唯一问题数 | 1,498 | 175K | **1,928K** |
| 唯一问题比例 | <1% | ~17% | **96%+** |
| 词汇量 | 725 | 40,787 | **138,372** |
| 平均问题长度 | 8.9 | 11.6 | **12.7** |

SK-VQA 唯一问题数是 Enc-VQA 的 **11 倍**，充分体现了用强模型生成相比模板的优势。

## 实验关键数据

### 零样本评估（6 个 SOTA MLLM）

| 模型 | InfoSeek | Enc-VQA | ViQuAE | SK-VQA |
|------|----------|---------|--------|--------|
| PaliGemma-3B | 25.66 | 32.89 | 47.72 | 25.51 |
| LLaVA-v1.5-7B | 42.82 | 53.69 | 78.41 | 40.99 |
| LLaVA-v1.6-7B | 41.94 | 57.92 | 72.00 | 46.68 |
| Idefics2-8B | 44.33 | 67.92 | 82.43 | 38.08 |
| LLaVA-v1.6-34B | 38.81 | 77.73 | 79.17 | 50.02 |

SK-VQA 对所有模型都极具挑战性，与 InfoSeek 相当，远低于 Enc-VQA/ViQuAE 的得分，且更大模型不一定更好——说明规模不足以解决该数据集的推理难度。

### 微调泛化实验（核心结论）

在 LLaVA-7B 和 PaliGemma-3B 上，分别用 InfoSeek / Enc-VQA / SK-VQA 微调（各 200K 样本），测跨域性能：

- **InfoSeek 微调**: 在 SK-VQA 上有提升，但 Enc-VQA、ViQuAE 无改善
- **Enc-VQA 微调**: 所有跨域指标均未超过基线
- **SK-VQA 微调**: 在 InfoSeek 和 Enc-VQA 上均取得显著零样本提升，在 ViQuAE 上也优于其他两个数据集的微调模型

PaliGemma-3B 上 SK-VQA 微调在全部 9 个跨域评测中均有显著提升，且是唯一不造成性能退化的训练集。

### 数据来源消融

| 图像+上下文 | InfoSeek | Enc-VQA | ViQuAE | 平均 |
|------------|----------|---------|--------|------|
| LAION + GPT-4 | 44.32 | 65.44 | 79.22 | 62.99 |
| Wiki + GPT-4 | 47.00 | 53.98 | 78.58 | 59.85 |
| Wiki + Wiki | 47.75 | 66.67 | 77.95 | 64.12 |
| COCO-CFs + GPT-4 | **48.00** | **65.42** | **79.23** | **64.22** |

最佳组合是 COCO-CFs（合成图像）+ GPT-4 上下文，甚至超过了 Wiki 真实图像+真实上下文，说明合成数据可以比真实数据更有效。

### RAG 实验

在 PaliGemma-3B 上用 CLIP Score Fusion 检索 top-10 段落模拟真实 RAG 环境，SK-VQA 微调模型在域内和域外均表现最强，全部 9 个跨域场景均超过基线和其他数据集的微调模型。

### 人工评估

- QA 质量: 人类准确率 77%（全集）、87%（IR+CAP 子集），标准差仅 0.02-0.03
- 事实性: 86% 可验证为事实，仅 4% 非事实
- GPT-4o 自动评估: 上下文事实性 4.6/5，问题相关性 4.9/5，可回答性 99.6%，答案正确性 90.7%

## 亮点与洞察

1. **单步生成策略**的精巧之处在于让上下文生成被 QA 任务需求所约束，避免了上下文与 QA 脱节的问题
2. **合成图像（COCO-CFs）+ 合成上下文** 竟然超过真实数据的微调效果，这挑战了"真实数据一定更好"的直觉
3. 不同图像来源贡献不同的泛化能力（LAION 利于 Enc-VQA/ViQuAE，Wiki 利于 InfoSeek），混合多来源是关键
4. 尝试用 LLaVA-34B 替代 GPT-4 生成数据，但 76% 的问题无效（多数仅需上下文即可回答），说明开源模型在该任务上仍有显著差距
5. 数据集覆盖艺术、时尚、体育、音乐等多元领域，远超现有 KB-VQA 数据集的实体知识范畴

## 局限与展望

1. **依赖 GPT-4 生成**: 数据集构建成本高，且无法避免 GPT-4 自身的偏差和幻觉（虽然人工验证 86% 事实性，但仍有 4% 非事实内容）
2. **开源替代不成熟**: LLaVA-34B 替代 GPT-4 的尝试失败（76% 问题无效），限制了社区复现和扩展
3. **过滤后数据量减半**: IR+CAP 过滤后从 200 万降至 98.5 万，高质量子集的规模折损较大
4. **仅聚焦文本上下文**: 未探索多模态上下文（如图表、视频片段）的增强
5. **评测指标有限**: InfoSeek 和 ViQuAE 使用精确匹配，可能低估模型实际能力；仅在有限模型上微调
6. 9% 的 QA 对仅需上下文即可回答（不需要图像），这部分数据对多模态推理训练价值有限

## 相关工作与启发

- **OK-VQA → InfoSeek**: 从"需要外部知识"到"信息检索式 QA"的演进，但均受限于模板构建
- **REVEAL / Wiki-LLaVA / Re-ViLM**: 多模态 RAG 系统的代表，聚焦检索器端，而本文聚焦生成器适配
- **Shumailov et al. (2023)**: 用模型生成数据训练模型可能导致 model collapse，但本文通过混合真实和合成数据缓解
- **UniIR / UniMur**: 多模态检索的统一方法，可与本文的生成器训练互补

## 评分
- 新颖性: ⭐⭐⭐⭐ (全自动合成管线 + 单步生成设计新颖，但核心思路仍是"用强模型造数据")
- 实验充分度: ⭐⭐⭐⭐⭐ (零样本/微调/消融/RAG/人工评估/自动评估，覆盖非常全面)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，实验编排合理)
- 价值: ⭐⭐⭐⭐ (数据集公开可用，对多模态RAG社区有实际推动，但GPT-4依赖限制可扩展性)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] MAGIC-VQA: Multimodal and Grounded Inference with Commonsense Knowledge for Visual Question Answering](../../ACL2025/multimodal_vlm/magic-vqa_multimodal_and_grounded_inference_with_commonsense_knowledge_for_visua.md)
- [\[ACL 2025\] Scaling Text-Rich Image Understanding via Code-Guided Synthetic Multimodal Data Generation](../../ACL2025/multimodal_vlm/code_guided_text_rich_image.md)
- [\[NeurIPS 2025\] Windsock is Dancing: Adaptive Multimodal Retrieval-Augmented Generation](../../NeurIPS2025/multimodal_vlm/windsock_is_dancing_adaptive_multimodal_retrieval-augmented_generation.md)
- [\[ACL 2025\] Insight Over Sight: Exploring the Vision-Knowledge Conflicts in Multimodal LLMs](../../ACL2025/multimodal_vlm/conflictvis_vision_knowledge_conflict.md)
- [\[NeurIPS 2025\] Benchmarking Retrieval-Augmented Multimodal Generation for Document Question Answering](../../NeurIPS2025/multimodal_vlm/benchmarking_retrievalaugmented_multimodal_generation_for_do.md)

</div>

<!-- RELATED:END -->
