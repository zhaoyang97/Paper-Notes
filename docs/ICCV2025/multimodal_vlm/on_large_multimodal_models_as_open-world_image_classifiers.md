---
title: >-
  [论文解读] On Large Multimodal Models as Open-World Image Classifiers
description: >-
  [ICCV 2025][多模态VLM][大型多模态模型] 系统性地评估了 13 个大型多模态模型（LMM）在开放世界图像分类任务上的表现，提出包含 4 种互补指标的评估协议，揭示了 LMM 在粒度判断和细粒度区分上的系统性错误模式。 传统图像分类任务需要预定义的类别集合（闭合世界），而 LMM 天然支持开放式输出——通过回答…
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "大型多模态模型"
  - "开放世界分类"
  - "评估协议"
  - "LMM"
  - "图像分类"
---

# On Large Multimodal Models as Open-World Image Classifiers

**会议**: ICCV 2025  
**arXiv**: [2503.21851](https://arxiv.org/abs/2503.21851)  
**代码**: [GitHub](https://github.com/altndrr/lmms-owc)  
**领域**: 多模态VLM  
**关键词**: 大型多模态模型, 开放世界分类, 评估协议, LMM, 图像分类

## 一句话总结

系统性地评估了 13 个大型多模态模型（LMM）在开放世界图像分类任务上的表现，提出包含 4 种互补指标的评估协议，揭示了 LMM 在粒度判断和细粒度区分上的系统性错误模式。

## 研究背景与动机

传统图像分类任务需要预定义的类别集合（闭合世界），而 LMM 天然支持开放式输出——通过回答"图片中的物体是什么？"直接生成类别名称，无需固定候选列表。这种能力极具实用价值，但现有研究存在明显不足：

**问题一**：大多数 LMM 分类评估仍局限在闭合世界设定下（给模型提供候选类别列表），无法体现 LMM 的真实开放世界能力。Zhang et al. (2024) 虽尝试了开放世界评估，但仅限 4 个数据集和单一指标（文本包含）。

**问题二**：现有评估指标过于简单。文本包含指标（text inclusion）只检查预测中是否包含正确标签文字，无法处理以下场景：
- 语义等价但文字不同（sofa vs. couch）
- 粒度不同但合理（dog vs. pug）
- 字面正确但语义有误（can vs. trash can 中 can 的误匹配）

**本文动机**：提供首个大规模、多维度的 LMM 开放世界分类评估基准，通过多指标分析 LMM 的错误模式，为后续研究提供方向指引。

## 方法详解

### 整体框架

本文不引入新模型，而是构建一套完整的开放世界分类评估框架：
1. 形式化开放世界分类任务定义
2. 提出 4 种互补评估指标
3. 在 10 个数据集、13 个模型上进行大规模实验
4. 通过指标组合分析错误类型及应对策略

### 关键设计

1. **任务形式化（Open-World Classification）**:

    - 功能：定义 LMM 作为函数 $f_{\text{LMM}}: \mathcal{X} \times \mathcal{T} \rightarrow \mathcal{T}$，输入图像和查询文本，输出文本预测
    - 核心思路：闭合世界在查询中提供候选集 $\mathcal{C}$，开放世界不限制输出空间，模型从所有可能语义概念 $\mathcal{Y}$ 中自由预测，其中 $|\mathcal{C}| \ll |\mathcal{Y}|$
    - 设计动机：LMM 的生成能力不应被预定义类别列表所限制，开放世界设定更符合真实应用场景

2. **四种评估指标**:

    - 功能：从不同角度衡量预测与真值的对齐程度
    - 核心思路：
        - **文本包含（TI）**：检查真值标签是否为预测文本的子串，$\text{TI}(y, \hat{y}) = \mathbf{1}[y \subseteq \hat{y}]$。简单但过于严格，语义等价但文字不同会被判错
        - **Llama 包含（LI）**：使用 Llama 3.2 3B 作为判官，判断预测是否与真值语义一致。LLM-as-judge 范式的分类任务特化版本
        - **语义相似度（SS）**：$\text{SS} = \langle g_{\text{emb}}(\hat{y}), g_{\text{emb}}(y) \rangle$，使用 Sentence-BERT 计算预测与真值的向量余弦相似度，提供 0-1 连续分数
        - **概念相似度（CS）**：$\text{CS} = \max_{p \in \text{split}(\hat{y})} \langle g_{\text{emb}}(p), g_{\text{emb}}(y) \rangle$，先用 spaCy 对预测分句，取最相似片段与真值的余弦相似度，解决冗长预测被整体稀释的问题
    - 设计动机：单一指标无法全面评估，不同指标的不一致恰好可以揭示错误模式

3. **错误分析框架**:

    - 功能：利用指标间的差异定位模型错误类型
    - 核心思路：
        - CS 高而 LI 低 → 粒度错误（correct but too generic，如预测"动物"而非"拉布拉多"）
        - LI 高而 CS 低 → 标注歧义或多标签问题
        - 两者均低 → 细粒度区分失败（wrong but specific，如混淆两种相似花卉）
    - 设计动机：不同于简单报告准确率，组合指标提供可操作的改进方向

### 损失函数 / 训练策略

本文为评估研究，不涉及模型训练。实验中统一使用标准 prompt "What type of object is in this image?" 进行零样本推理。

## 实验关键数据

### 主实验

**LMM 开放世界分类性能**（按数据集粒度分组均值）：

| 模型 | Prototypical TI | Prototypical LI | Fine-grained TI | Very Fine TI | 说明 |
|------|----------------|-----------------|-----------------|-------------|------|
| Qwen2VL 7B | 46.4 | 78.7 | 34.6 | 0.8 | LMM 最佳 |
| InternVL2 8B | 40.6 | 74.4 | 22.3 | 2.3 | 次优 |
| LLaVA-1.5 7B | 34.6 | 63.1 | 8.4 | 0.0 | 表现较差 |
| CaSED (OW baseline) | 24.5 | 46.3 | 27.4 | 0.7 | 对比式 OW 方法 |
| CLIP (闭合世界) | 76.4 | - | 85.0 | - | 闭合世界上界 |
| SigLIP (闭合世界) | 81.8 | - | 92.6 | - | 闭合世界上界 |

### 消融实验

**Prompt 设计对粒度的影响**：

| Prompt 策略 | CS 提升 | 说明 |
|-------------|---------|------|
| 默认 prompt | - | "What type of object?" |
| 细粒度 prompt | +5-10% CS | 添加"be specific"等引导 |
| CoT 推理 | +2-5% | 添加 Chain-of-Thought 步骤 |

**推理策略对细粒度区分的影响**：

| 策略 | Fine-grained 改善 | 说明 |
|------|-------------------|------|
| 直接预测 | baseline | 默认模式 |
| 结构化推理 | +3-8% LI | 减少相似类别间的混淆 |

### 关键发现

1. **LMM 在开放世界中优于对比式基线**：与 CaSED、CLIP 检索等无需类别列表的方法相比，生成式 LMM 表现更好，但距离给定候选列表的闭合世界模型仍有显著差距
2. **粒度越细，性能越差**：从 prototypical（~46% TI）到 very fine-grained（~1% TI），性能断崖式下降
3. **粒度错误是主要问题**：模型倾向于给出过于笼统的答案（如"bird"而非"scarlet tanager"），通过 prompt 引导可部分缓解
4. **模型规模有帮助但非决定性**：InternVL2 2B→8B 有稳定提升，但 LLaVA-OV 0.5B→7B 反而在部分指标上退步
5. **标注歧义广泛存在**：使用 tagging 模型验证发现很多"错误"预测实际上是因为图像中包含多个合理标签

## 亮点与洞察

- **评估而非方法**创新，但贡献同等重要——首次将 LMM 开放世界分类从"简单跑个分"提升到严谨的评估科学
- 4 种指标的**互补性**设计精巧：TI 捕捉严格匹配，LI 捕捉语义正确性，SS/CS 提供连续评分，指标间的差异能诊断具体问题
- 粒度-性能的断崖关系揭示 LMM 视觉感知的根本瓶颈——不是"不认识"而是"说不出准确名字"
- 对标注歧义的分析诚实且有价值，提醒社区不要过度解读开放世界评估中的"错误"

## 局限与展望

- 缺少最新的 LMM（如 GPT-4o、Gemini 2），结论在更强模型上是否成立需要验证
- 评估 prompt 统一使用"What type of object"，对非物体数据集（DTD 纹理、UCF101 动作）不够自然
- Llama inclusion 指标依赖 Llama 3.2 3B 的判断能力，本身也可能引入偏差
- 未探索 few-shot 或 retrieval-augmented 策略能否系统性提升细粒度识别

## 相关工作与启发

- 与 CaSED (CVPR 2023) 的对比揭示了生成式和对比式模型在开放世界识别上的不同优劣
- 粒度问题与 prompt engineering 的关联提示了针对性 prompt 模板库的实用价值
- 评估框架可直接用于其他视觉-语言理解任务的开放世界评估
- Zhang et al. (2024) 的 OW 分类研究是最直接的前驱工作，本文在规模和深度上全面超越

## 评分

- 新颖性: ⭐⭐⭐ 评估研究非方法创新，但任务定义和指标设计有原创性
- 实验充分度: ⭐⭐⭐⭐⭐ 13 模型 × 10 数据集 × 4 指标，覆盖面极广
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，分析层层递进，图表丰富
- 价值: ⭐⭐⭐⭐ 为 LMM 社区提供了急需的评估基础设施和有洞察力的错误分析

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] OpenHOI: Open-World Hand-Object Interaction Synthesis with Multimodal Large Language Models](../../NeurIPS2025/multimodal_vlm/openhoi_open-world_hand-object_interaction_synthesis_with_multimodal_large_langu.md)
- [\[ICCV 2025\] ProbRes: Probabilistic Jump Diffusion for Open-World Egocentric Activity Recognition](probres_probabilistic_jump_diffusion_for_open-world_egocentric_activity_recognit.md)
- [\[ICCV 2025\] GRAB: A Challenging GRaph Analysis Benchmark for Large Multimodal Models](grab_a_challenging_graph_analysis_benchmark_for_large_multimodal_models.md)
- [\[ICCV 2025\] AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models](aigi_holmes_towards_explainable_and_generalizable_ai_generated_image_detection_via_mllm.md)
- [\[ACL 2025\] CrafText Benchmark: Advancing Instruction Following in Complex Multimodal Open-Ended World](../../ACL2025/multimodal_vlm/craftext_benchmark_advancing_instruction_following_in_complex_multimodal_open-en.md)

</div>

<!-- RELATED:END -->
