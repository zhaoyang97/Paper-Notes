---
title: >-
  [论文解读] MERIT: Multilingual Semantic Retrieval with Interleaved Multi-Condition Query
description: >-
  [NeurIPS 2025][多语言/翻译][interleaved retrieval] 提出首个多语言交错多条件语义检索数据集 MERIT（320K queries, 135K products, 5种语言, 7大品类），揭示现有检索模型仅关注全局语义而忽略条件细节的瓶颈，并设计 Coral 微调框架通过嵌入重建+对比学习将检索性能提升 45.9%。
tags:
  - "NeurIPS 2025"
  - "多语言/翻译"
  - "interleaved retrieval"
  - "multilingual"
  - "multi-condition query"
  - "对比学习"
  - "embedding reconstruction"
---

# MERIT: Multilingual Semantic Retrieval with Interleaved Multi-Condition Query

**会议**: NeurIPS 2025  
**arXiv**: [2506.03144](https://arxiv.org/abs/2506.03144)  
**代码**: [https://github.com/weichow23/merit](https://github.com/weichow23/merit)  
**领域**: 多语言翻译  
**关键词**: interleaved retrieval, multilingual, multi-condition query, contrastive learning, embedding reconstruction

## 一句话总结
提出首个多语言交错多条件语义检索数据集 MERIT（320K queries, 135K products, 5种语言, 7大品类），揭示现有检索模型仅关注全局语义而忽略条件细节的瓶颈，并设计 Coral 微调框架通过嵌入重建+对比学习将检索性能提升 45.9%。

## 研究背景与动机

**领域现状**：语义检索在产品搜索、RAG 等场景中至关重要，但现有数据集局限于单语言、单图像、单检索条件，远未覆盖真实场景的复杂性。

**现有痛点**：大量已有工作（Fashion-IQ、CIRR、Magiclens 等）在图像被替换为对应 caption 后性能不受影响，说明这些数据集没有真正利用图像的表达能力（Vision Unnecessarity）。

**核心矛盾**：真实产品检索经常涉及交错的多条件查询（如特定花纹+特定材质），其中许多属性必须通过图像表达，现有数据集无法评估这类能力。

**本文要回答的两个问题**：(1) 如何全面衡量现有模型在交错多条件检索任务上的能力？(2) 限制性能的关键因素是什么、如何改善？

## 方法详解

### MERIT 数据集构建

- **规模**：135K 产品，320K 检索对（310K 训练 + 10K 测试），涵盖 5 种语言（英/马来/印尼/越南/泰）和 7 大产品品类
- **标注流程**（4步）：
  1. **高质量产品选择**：从东南亚 6 国内部数据集中筛选热门产品，GPT-4o 生成标题，美学评分过滤
  2. **开放式属性标注**：116 个唯一属性、2594 个属性值，采用开放标注+统计分析的方式确定属性体系
  3. **查询对组合**：三种采样策略融合——均匀采样、属性均匀采样、高相似度产品优先采样
  4. **多轮过滤**：自动规则过滤 + 人工专家审核，总计投入 10,000 人工小时
- **关键特性**：首个支持多图像交错输入的语义检索数据集；查询包含 ≥2 个条件，大多为双条件（319,600）

### Coral 微调框架

核心思想：将预训练 MLLM 适配为多模态检索模型时，在对比学习之外加入嵌入重建，保留细粒度条件信息。

1. **对比学习损失** $\mathcal{L}_{cl}$：
   $$\mathcal{L}_{cl} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(q_i \cdot k_{i+}/\tau)}{\sum_{j=1}^{N}\exp(q_i \cdot k_j/\tau)}$$
   标准 InfoNCE Loss，拉近 query 与正样本、推远负样本。

2. **视觉重建损失** $\mathcal{L}_{mse}$：
    - 对多模态嵌入 $E=[e_{img};e_{txt}]$ 中的视觉部分进行随机掩码（比率 $\delta=0.5$），使用随机初始化的 BERT 解码层 $\mathcal{F}_{\theta}^{v}$ 重建
   $$\mathcal{L}_{mse} = -\frac{1}{N}\sum_{i=1}^{N}\|\hat{E} - E\|_2^2, \quad \hat{E} = \mathcal{F}_{\theta}^{v}[\mathcal{MASK}_v(E); h_{eos}]$$
   - 设计动机：仅靠 [EOS] token 的对比学习会过度压缩全局语义，掩码重建迫使模型在 [EOS] 中保留细粒度视觉信息

3. **掩码语言建模损失** $\mathcal{L}_{mlm}$：
    - 对文本部分掩码后重建，解码器 $\mathcal{F}_{\theta}^{l}$ 与 MLLM 的 LM head 共享参数
   $$\mathcal{L}_{mlm} = -\frac{1}{N}\sum_{i=1}^{N}\log P(\hat{x}_i \mid X)$$

4. **总损失**：
   $$\mathcal{L} = \mathcal{L}_{cl} + \lambda_1 \mathcal{L}_{reg} + \lambda_2 \mathcal{L}_{rec}$$
   其中 $\mathcal{L}_{reg}$ 和 $\mathcal{L}_{rec}$ 分别用条件的 [EOS] 和目标自身的 [EOS] 作为 attention query 来重建检索目标。

## 实验关键数据

### 现有模型在 MERIT 上的表现（零样本 + Embedding 模型）

| 方法 | 规模 | 输入类型 | R@1 | R@5 | R@10 | MRR |
|------|------|----------|-----|-----|------|-----|
| Qwen2.5-VL (Zero-Shot) | 3B | Seq | 0.09 | 0.39 | 0.56 | 0.21 |
| LamRA-Qwen2.5VL | 7B | Cat | **12.05** | 39.13 | 48.03 | 23.80 |
| GME-Qwen2VL | 2B | Cat | 8.47 | **47.13** | **56.18** | **25.02** |
| BGE-VL | 7B | Cat | 11.55 | 38.01 | 46.26 | 23.00 |

### Coral 消融实验（Qwen2.5-VL）

| 方法 | LoRA | 类型 | R@1 | R@5 | R@10 | MRR |
|------|------|------|-----|-----|------|-----|
| CL baseline | ✓ | Seq | 48.52 | 73.11 | 77.93 | 59.48 |
| CL baseline | ✗ | Seq | 47.76 | 73.97 | 80.47 | 59.06 |
| +Coral (Full) | ✗ | Seq | **69.68** | **89.26** | **93.08** | **78.33** |
| +Coral | ✗ | Cat | 60.94 | 85.60 | 90.40 | 71.70 |

- Coral 相比纯对比学习 R@1 提升 **45.9%**（47.76 → 69.68）
- 序列输入（Seq）始终优于图像拼接（Cat）
- 全参微调优于 LoRA
- 在 8 个外部检索 benchmark 上也取得一致提升，VisDial 上提升 181%

### 关键发现
- 图像拼接输入 R@5 比交错输入高 **119.7%**，但训练后交错输入性能提升 14.3%
- 图像被替换为 caption 后性能下降 **73.9%**，证实图像不可或缺
- 错误分析：属性错误和视觉理解错误占比最高

## 亮点
- ⭐⭐⭐⭐ **首个交错多条件多语言语义检索数据集**：填补了重要空白，10K 人工小时标注保证质量
- ⭐⭐⭐⭐ **问题诊断精准**：清晰揭示现有方法"只看全局、忽略条件细节"的瓶颈
- ⭐⭐⭐⭐ **Coral 设计优雅**：掩码重建作为对比学习的互补手段，思路简洁有效
- ⭐⭐⭐ **实验全面**：9 个 SOTA 基线 + 8 个外部 benchmark + OOD 分析 + 错误归因

## 局限与展望
1. 数据集仅覆盖电商产品检索场景，向其他领域（学术搜索、新闻检索）的迁移性待验证
2. 重建解码器增加训练开销，推理时虽可丢弃但训练效率需关注
3. 语言覆盖偏向东南亚，未包含中文、日文等东亚语言
4. 属性标注依赖内部数据，可复现性受限

## 总评
⭐⭐⭐⭐ 扎实的 benchmark + 方法论文，数据集构建规范、问题定位清晰、方法设计巧妙。MERIT 有望成为多模态检索领域的重要评估标准，Coral 的"重建+对比"范式具有推广价值。

## 与相关工作的对比

## 启发与关联

## 评分

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] KELPS: A Framework for Verified Multi-Language Autoformalization via Semantic-Syntactic Alignment](../../ICML2025/multilingual_mt/kelps_a_framework_for_verified_multi-language_autoformalization_via_semantic-syn.md)
- [\[ACL 2025\] M3FinMeeting: A Multilingual, Multi-Sector, and Multi-Task Financial Meeting Understanding Evaluation Dataset](../../ACL2025/multilingual_mt/m3finmeeting_a_multilingual_multi-sector_and_multi-task_financial_meeting_unders.md)
- [\[ACL 2025\] Semantic Aware Linear Transfer by Recycling Pre-trained Language Models for Cross-Lingual Transfer](../../ACL2025/multilingual_mt/semantic_aware_linear_transfer_by_recycling_pre-trained_language_models_for_cros.md)
- [\[ACL 2025\] Multi-perspective Alignment for Increasing Naturalness in Neural Machine Translation](../../ACL2025/multilingual_mt/multi-perspective_alignment_for_increasing_naturalness_in_neural_machine_transla.md)
- [\[ACL 2026\] SERM: Self-Evolving Relevance Model with Agent-Driven Learning from Massive Query Streams](../../ACL2026/multilingual_mt/serm_self-evolving_relevance_model_with_agent-driven_learning_from_massive_query.md)

</div>

<!-- RELATED:END -->
