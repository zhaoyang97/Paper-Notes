---
title: >-
  [论文解读] Understanding Museum Exhibits using Vision-Language Reasoning
description: >-
  [ICCV 2025][多模态VLM][博物馆视觉问答] 构建了一个包含 6500 万张图片和 2 亿个问答对的大规模博物馆展品数据集 Museum-65，并通过在该数据集上微调 BLIP 和 LLaVA 证明：领域特定的大规模数据集显著优于零样本 SOTA VLM，微调后的 LLaVA 在展品标题和产地识别上分别达到 57% 和 70% 的准确率（vs. GPT-4o 的 22% 和 33%）。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "博物馆视觉问答"
  - "大规模数据集"
  - "领域微调"
  - "文化遗产"
  - "视觉语言模型"
---

# Understanding Museum Exhibits using Vision-Language Reasoning

**会议**: ICCV 2025  
**arXiv**: [2412.01370](https://arxiv.org/abs/2412.01370)  
**代码**: [github.com/insait-institute/Museum-65](https://github.com/insait-institute/Museum-65)  
**领域**: 多模态VLM  
**关键词**: 博物馆视觉问答, 大规模数据集, 领域微调, 文化遗产, 视觉语言模型

## 一句话总结

构建了一个包含 6500 万张图片和 2 亿个问答对的大规模博物馆展品数据集 Museum-65，并通过在该数据集上微调 BLIP 和 LLaVA 证明：领域特定的大规模数据集显著优于零样本 SOTA VLM，微调后的 LLaVA 在展品标题和产地识别上分别达到 57% 和 70% 的准确率（vs. GPT-4o 的 22% 和 33%）。

## 研究背景与动机

### 问题定义

博物馆作为文化遗产和历史文物的宝库，其展品理解需要将视觉特征与历史知识相结合。本文的目标是构建能够理解博物馆展品的 AI 模型，使其在视觉问答（VQA）任务中能够准确回答关于展品标题、创作者、时期、技法、文化背景等多维度问题。

### 已有方法的不足

**通用 VLM 在专业领域表现差**：CLIP、Gemini、LLaVA 等模型虽然在通用视觉理解上表现优异，但在博物馆等需要跨学科知识的领域中表现不足，尤其在结构化属性预测（如年代、产地、材质）方面远低于需求

**现有数据集规模不足且领域受限**：已有文化遗产数据集（如 AQUA 的 2.1 万图 / MUZE 的 21 万图）大多仅关注艺术类展品，且规模远远不够训练专业模型

**缺乏真实场景的评估基准**：现有工作没有针对博物馆真实使用场景（如多角度拍摄、多语言查询、需要常识推理的问题）设计系统性评估

### 核心动机

**关键洞察**：博物馆是人类知识的高质量来源——展品信息由领域专家标注，具有极高的准确性和深度。将这些结构化知识系统地转化为大规模数据集，可以训练出在该领域远超通用模型的专业模型。同时，真实世界中游客与展品的交互方式（好奇提问、多角度观察、多语言询问）为模型评估提供了自然的任务定义。

## 方法详解

### 整体框架

本文的方法论包含三个核心部分：
1. **数据集构建**：从全球博物馆聚合器和独立博物馆采集、清洗、结构化 6500 万张展品图片和 2 亿个问答对
2. **模型微调**：在 Museum-65 上分别微调 BLIP（编码器-解码器架构）和 LLaVA（指令微调的 LLM + 视觉编码器）
3. **多任务评估**：设计 5 个反映真实博物馆场景的 VQA 任务进行系统性评估

### 关键设计

#### 1. **Museum-65 数据集构建**

- **功能**：构建迄今最大的博物馆展品多模态数据集
- **核心思路**：
    - 数据来源：3 大国际聚合器（DPLA、Europeana、Smithsonian）+ 12 个独立博物馆，覆盖欧洲、北美、亚洲、非洲、大洋洲
    - 数据规模：5000 万英文对象 + 1500 万多语种对象（37 种语言）
    - 属性到问题的转化：将展品的属性-值对（如 material: bronze）转化为自然语言问题（"What is the material used in the object?"），63 个手工制作的问题模板
    - 质量保证：10 名专家 3 个月清洗标注，2 名专家交叉检验
- **设计动机**：博物馆聚合器提供了经专家策展的高质量数据，通过规模化采集和结构化处理，可以生成足够训练大规模 VLM 的数据。多样化的问题模板模拟了游客的自然提问方式

#### 2. **双模型微调策略**

- **功能**：分别微调 BLIP 和 LLaVA，比较不同架构在领域 VQA 中的表现
- **核心思路**：
    - BLIP：编码器-解码器架构（BERT-base，110M 参数），擅长图文对齐但指令遵循能力弱
    - LLaVA：基于 Llama-7B 的指令微调 LLM，具有更强的推理和指令理解能力
    - 训练配置：BLIP 使用 1M/10M/20M 数据分别训练 5 epochs；LLaVA 使用 1M 训练 5 epochs 和 20M 训练 1 epoch
    - 每个 epoch 对每张图片随机选择一个问答对
- **设计动机**：选择两种代表性架构——BLIP 代表传统视觉语言对齐模型，LLaVA 代表新一代基于 LLM 的多模态模型，通过对比揭示不同架构在领域特定任务中的优劣

#### 3. **五项真实场景 VQA 任务**

- **功能**：设计全面覆盖博物馆真实应用场景的评估体系
- **核心思路**：
    - **Task 1 通用 VQA**：在全部问题上评估，考察模型的综合能力
    - **Task 2 分类 VQA**：按属性类别（标题、创作者、材质等）分组评估，揭示模型在不同知识类型上的优劣
    - **Task 3 多角度**：使用同一展品不同拍摄角度的图片，测试模型对视角变化的鲁棒性
    - **Task 4 视觉不可回答问题**：需结合常识推理的问题（如"画家的导师是谁"），测试深层知识整合能力
    - **Task 5 多语言**：法语、德语等非英语问题，评估跨语言泛化能力
- **设计动机**：真实博物馆场景中，游客的问题远不止简单的属性查询——他们可能从不同角度拍照、用母语提问、问出需要背景知识才能回答的问题

### 损失函数 / 训练策略

- BLIP：使用标准的 VQA 微调方案，batch size 512
- LLaVA：使用标准的指令微调方案，batch size 512
- 硬件：64× NVIDIA H100 GPU
- 评估指标：精度（完全匹配 / 部分匹配）、召回率、BLEU1/2、METEOR、WMD 准确率

## 实验关键数据

### 主实验

**零样本 SOTA vs. 微调模型**：

| 模型 | 标题准确率 | 产地准确率 |
|------|----------|----------|
| GPT-4o (零样本) | 22.03 | 33.33 |
| Claude-3-7-sonnet (零样本) | 21.89 | 40.43 |
| Gemini-1.5B-flash (零样本) | 27.08 | 32.98 |
| LLaVA 无微调 | 10.13 | 23.42 |
| **LLaVA-ours (20M, 1ep)** | **57.00** | **70.00** |
| BLIP 无微调 | 3.00 | 5.00 |
| **BLIP-ours (20M, 5ep)** | **52.00** | **61.00** |

**语义评估**：

| 模型 | METEOR | WMD 准确率 |
|------|--------|----------|
| BLIP 无微调 | 3.24 | 35.54 |
| BLIP-ours (20M, 5ep) | 37.45 | 74.02 |
| LLaVA 无微调 | 2.96 | 54.50 |
| **LLaVA-ours (20M, 1ep)** | **58.85** | **87.02** |

### 消融实验

**各大洲展品的部分精度**：

| 模型 | 欧洲 | 北美 | 南美 | 亚洲 | 非洲 | 大洋洲 |
|------|------|------|------|------|------|--------|
| LLaVA-ours | 85.2 | 79.6 | 86.6 | 67.4 | 86.7 | 99.2 |
| LLaVA 无微调 | 8.6 | 43.57 | 20.3 | 23.4 | 20.79 | 52.4 |
| BLIP-ours | 79.1 | 73.1 | 76.4 | 65.5 | 76.4 | 49.7 |
| BLIP 无微调 | 4.3 | 15.2 | 19.7 | 9.3 | 19.7 | 6.6 |

**微调模型 vs. 人类专家（分类 VQA）**：微调模型在所有类别上均超越 10 名博物馆专家

### 关键发现

1. **领域微调效果巨大**：微调后的 LLaVA 在标题识别上从 10.13% 跃升至 57%（+46.87 pp），证明领域数据的极端重要性
2. **LLaVA 全面优于 BLIP**：在所有任务上，LLaVA 凭借更大的语言模型（7B vs. 110M）和更强的推理能力占优，尤其在需要常识推理的任务上差距更大
3. **视角鲁棒性强**：多角度测试中，微调模型仅有微小精度下降（58.09→56.14），展示了对视角变化的强鲁棒性
4. **多语言能力有限**：仅英语微调导致 LLaVA 的多语言能力有所退化（法语部分精度从 41.81 降至 10.37），提示需要多语言微调
5. **训练数据不平衡不影响全局收益**：尽管训练数据以欧美为主，但微调对所有大洲的展品识别均有显著提升

## 亮点与洞察

1. **数据集规模和质量的突破**：6500 万图 + 2 亿 QA 对，是现有最大文化遗产多模态数据集（VISCOUNTH 的 130 倍）
2. **实验设计贴近真实场景**：5 项任务完整覆盖了博物馆 AI 助手的实际需求——多角度识别、跨语言服务、深度知识问答
3. **"领域数据 > 模型大小"的论证**：7B 微调模型超越零样本 GPT-4o，有力证明了领域微调在专业场景中的价值
4. **微调模型超越人类专家**：这是一个重要的里程碑，证明 AI 在结构化知识密集型任务中已可超越领域专家
5. **对数据偏差的系统性分析**：包括地域偏差、语言偏差、拍摄角度偏差的全面分析和缓解措施

## 局限与展望

1. **多语言微调缺失**：目前仅英语微调导致多语言能力退化，需加入 1500 万多语种样本进行微调
2. **BLIP 容量限制**：BLIP 的 512 token 限制和较小的模型容量导致在复杂推理任务上表现不佳，微调后甚至在某些任务上遗忘先验知识
3. **任务 4 和 5 评估规模较小**：视觉不可回答问题（~500 对）和多语言评估（~500 张图）由于人工标注成本限制较小
4. **未探索更新的模型架构**：仅微调了 BLIP 和 LLaVA-7B，未尝试更大或更新的模型（如 Gemini、Qwen2-VL 等）
5. **仅限于问答任务**：未探索图像描述生成、检索、或交互式对话等更丰富的应用场景

## 相关工作与启发

- 与 MUZE 的区别：MUZE 使用 CLIP 的多模态表示和独立注意力头处理各属性，计算代价高且不适合直接 QA；Museum-65 则直接微调端到端 VLM
- 与 VISCOUNTH 的区别：VISCOUNTH 仅覆盖绘画和雕塑（500K 图），Museum-65 涵盖艺术、历史和自然科学
- 启发：该工作验证了一个重要范式——在专业领域，构建大规模高质量数据集 + 微调通用 VLM 是目前最有效的方案

## 评分

- **新颖性**: ⭐⭐⭐ — 方法层面相对简单（标准微调），核心贡献在数据集和评估体系
- **实验充分度**: ⭐⭐⭐⭐ — 5 项任务全面覆盖真实场景，含人类专家对比，但部分任务评估规模偏小
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，数据集构建过程详尽
- **价值**: ⭐⭐⭐⭐ — Museum-65 作为领域数据集具有长期价值，为文化遗产 AI 研究提供了重要基础设施

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Perspective-Aware Reasoning in Vision-Language Models via Mental Imagery Simulation](perspective-aware_reasoning_in_vision-language_models_via_mental_imagery_simulat.md)
- [\[ACL 2025\] Benchmarking and Improving Large Vision-Language Models for Fundamental Visual Graph Understanding and Reasoning](../../ACL2025/multimodal_vlm/benchmarking_and_improving_large_vision-language_models_for_fundamental_visual_g.md)
- [\[ICML 2025\] Understanding and Mitigating Miscalibration in Prompt Tuning for Vision-Language Models](../../ICML2025/multimodal_vlm/understanding_and_mitigating_miscalibration_in_prompt_tuning_for_vision-language.md)
- [\[ACL 2026\] ChemVLR: Prioritizing Reasoning in Perception for Chemical Vision-Language Understanding](../../ACL2026/multimodal_vlm/chemvlr_prioritizing_reasoning_in_perception_for_chemical_vision-language_unders.md)
- [\[ICCV 2025\] CAPTURe: Evaluating Spatial Reasoning in Vision Language Models via Occluded Object Counting](capture_evaluating_spatial_reasoning_in_vision_language_models_via_occluded_obje.md)

</div>

<!-- RELATED:END -->
