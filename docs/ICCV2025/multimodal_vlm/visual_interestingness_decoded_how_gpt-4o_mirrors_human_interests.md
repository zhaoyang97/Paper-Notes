---
title: >-
  [论文解读] Visual Interestingness Decoded: How GPT-4o Mirrors Human Interests
description: >-
  [ICCV2025][多模态VLM][Visual Interestingness] 系统性研究了 GPT-4o 等大型多模态模型对"图像有趣性"这一主观视觉概念的理解程度，发现 GPT-4o 与人类评判有中等正相关（配对图像一致率 73.8%），并提出利用 GPT-4o 自动标注图像对训练 learning-to-rank 模型来预测图像有趣性，超越了所有现有方法。
tags:
  - "ICCV2025"
  - "多模态VLM"
  - "Visual Interestingness"
  - "GPT-4o"
  - "多模态"
  - "Human-AI Alignment"
  - "Learning to Rank"
---

# Visual Interestingness Decoded: How GPT-4o Mirrors Human Interests

**会议**: ICCV2025  
**arXiv**: [2510.13316](https://arxiv.org/abs/2510.13316)  
**代码**: [https://github.com/fiabdu/Visual-Interestingness-Decoded](https://github.com/fiabdu/Visual-Interestingness-Decoded)  
**领域**: 多模态VLM / 视觉理解  
**关键词**: Visual Interestingness, GPT-4o, Large Multimodal Models, Human-AI Alignment, Learning to Rank

## 一句话总结
系统性研究了 GPT-4o 等大型多模态模型对"图像有趣性"这一主观视觉概念的理解程度，发现 GPT-4o 与人类评判有中等正相关（配对图像一致率 73.8%），并提出利用 GPT-4o 自动标注图像对训练 learning-to-rank 模型来预测图像有趣性，超越了所有现有方法。

## 研究背景与动机

**领域现状**：视觉有趣性（visual interestingness）是一个高度主观的概念——什么图像能吸引人的注意力？这个问题自 Berlyne 1949 年提出以来一直是计算机视觉和心理学的交叉领域。现有研究主要依赖人工标注（成本高、规模受限）或社交平台隐式信号（如 Flickr 收藏数，但有平台偏差）。

**现有痛点**：
   - 直接标注（AMT crowd-sourcing）成本高昂且难以规模化
   - 社交媒体指标（阅读量、收藏量）反映的是社交互动而非纯视觉有趣性
   - 缺乏可扩展的标注方法来获取大规模有趣性标签
   - LMM 在客观视觉任务（分类、VQA）上表现优异，但其对主观概念的理解能力尚未被系统探索

**核心矛盾**：有趣性是主观的（因人而异），但也存在某些图像"普遍有趣"的现象。能否利用 LMM 编码的大规模人类知识来自动捕捉这种"共识有趣性"？

**本文目标**：(1) LMM（特别是 GPT-4o）在多大程度上理解视觉有趣性？(2) LMM 的标注与人类判断的一致性和差异性在哪里？(3) 能否用 LMM 的知识训练一个轻量级的有趣性预测模型？

**切入角度**：用配对比较（pairwise comparison）替代绝对评分来评估有趣性，因为人对"哪个更有趣"的判断比"是否有趣"更可靠、更有区分度。

**核心 idea**：用 GPT-4o 的配对有趣性标注替代人工标注，通过 learning-to-rank 框架蒸馏出轻量级有趣性预测模型。

## 方法详解

### 整体框架
研究分为四个递进实验：(1) **单图有趣性评估**：人类和 LMM 各自判断 1000 张图像是否有趣；(2) **配对有趣性评估**：人类和 GPT-4o 对 2500 个图像对判断哪个更有趣；(3) **Learning-to-Rank 模型训练**：用标注数据训练 Siamese 网络预测有趣性排序；(4) **深度分析**：通过聚类解释人类和 GPT-4o 的一致与分歧。

### 关键设计

1. **单图有趣性评估实验**:

    - 功能：建立基线——人类和 LMM 对绝对有趣性的判断
    - 核心思路：从 Flickr-User 数据集均匀采样 1000 张图像，AMT 上 258 名工人各给每张图 5 次"是否有趣"的二元判断，同时 GPT-4o/Llama 3.2/DeepSeek-VL2 各做 5 次判断。一致性 $|\mathcal{C}_x|$ 定义为 4/5 以上工人一致的比例
    - 关键发现：人类一致率 91.9%，GPT-4o 一致率 93.9%，但 **几乎所有图像都被判为"有趣"**（人类 99.9%，GPT-4o 95.3%）。这说明绝对有趣性评估几乎无区分度——人和模型被问到时都倾向于"找有趣的地方"

2. **配对有趣性评估实验**:

    - 功能：通过相对比较获得有区分度的有趣性标注
    - 核心思路：构建 2500 个图像对，553 名 AMT 工人做配对判断。GPT-4o 同样做配对判断。发现并处理了 GPT-4o 的系统性偏差——36% 的图像对中 GPT-4o 总是偏好第二张图（位置偏差）。通过正反序双跑筛除不一致的对，保留 1599 个可靠标注对
    - 关键发现：人类在配对比较中一致率降至 56.3%（说明有趣性确实主观）。GPT-4o 与人类的整体一致率为 66.2%，在人类高度共识的图像对上升至 73.8%
    - 设计动机：配对比较比绝对评分更有效地捕捉有趣性差异

3. **Learning-to-Rank 蒸馏模型**:

    - 功能：用标注数据训练一个轻量级的有趣性预测模型
    - 核心思路：Siamese 网络架构，共享权重。输入图像对 $(I_0, I_1)$ → 分别用 CLIP 提取特征 → 共享线性层（单神经元）→ 打分差异通过 sigmoid → 二元交叉熵损失训练。得分函数 $S(I_0, I_1) = \sigma(\mathbf{w}^\top \text{CLIP}(I_0) - \mathbf{w}^\top \text{CLIP}(I_1))$。训练后对单张图像也能输出有趣性分数 $S(I) = \sigma(\mathbf{w}^\top \text{CLIP}(I))$
    - 设计动机：模型极其轻量（单层线性），训练数据来自 GPT-4o 自动标注（可大规模扩展）。CLIP 特征已编码了丰富的视觉语义，线性层只需学习"有趣性"这个维度

4. **人机差异分析**:

    - 功能：通过文本 embedding 聚类理解人类和 GPT-4o 对有趣性的不同理解方式
    - 核心思路：对标注者的"原因"文本用 OpenAI embedding 模型编码后做层次聚类，发现共识和分歧模式
    - 关键发现：人类和 GPT-4o 在"可爱/情感共鸣"、"独特性"上高度一致，但 GPT-4o 还额外偏好"色彩鲜艳"和"动态场景"（人类未必认为这些有趣）

### 损失函数 / 训练策略
二元交叉熵损失，数据 50/50 划分训练/测试，训练 25 epochs，无过拟合。50 次不同 split 的平均结果作为最终报告。

## 实验关键数据

### 主实验（配对有趣性预测准确率）

| 方法 | 与人类一致率 $A^{(H,x)}$ | LtR 模型 Acc (人类GT) | Spearman $r_S$ (人类) |
|------|------------------------|-------------------|--------------------|
| GPT-4o 直接标注 | **73.8%** | 73.4% | 0.59 |
| CuPL (zero-shot) | 60.3% | 61.5% | 0.34 |
| CI (Commonly Interesting) | 69.6% | 69.6% | 0.54 |
| Aesthetic (VILA) | 68.3% | 69.0% | 0.50 |
| Memorability | 35.5% | 34.7% | -0.42 |
| #Comments (社交指标) | 68.0% | 66.6% | 0.46 |
| #Favorites (社交指标) | 66.4% | 66.3% | 0.47 |
| Human baseline | - | **77.5%** | - |

### 消融实验（不同标注源训练的 LtR 模型对比）

| 训练标注源 | 测试 Acc (人类GT) | 测试 Acc (GPT-4o GT) | 说明 |
|-----------|-----------------|--------------------|----|
| Human | **77.5%** | 72.0% | 最优人类预测 |
| GPT-4o | 73.4% | **84.8%** | 最优 GPT-4o 预测 |
| Aesthetic | 69.0% | 73.6% | 审美≠有趣 |
| CI | 69.6% | 69.1% | 关联但不同 |
| Memorability | 34.7% | 38.3% | 记忆性与有趣性负相关 |

### 关键发现
- **绝对评估无意义**：单图"是否有趣"的问题几乎无区分度——人和 LMM 都倾向找到有趣之处，99%+ 判为有趣
- **GPT-4o 有位置偏差**：36% 的配对中总是选第二张图，这是一个值得注意的系统性错误
- **记忆性与有趣性负相关**：Memorability 模型的 Spearman 相关为 -0.42，说明容易记住的图像不一定有趣
- **GPT-4o 不受人口统计影响**：测试了不同性别、年龄、地区的 persona prompt，116 个图像对的结果完全相同
- **GPT-4o 额外偏好鲜艳色彩和动态场景**：人类不一定认为这些有趣，暗示 LMM 的"有趣性"概念部分来自其训练数据中的视觉显著性

## 亮点与洞察
- **提出了一个有价值的研究范式**：用 LMM 的知识蒸馏来替代人工标注，这个思路不局限于有趣性，可以推广到任何主观视觉属性（审美、情绪、风格等）
- **配对比较的 insight**：绝对评分（"是否有趣"）几乎无效，配对比较（"哪个更有趣"）才是获取主观评价的正确方式。这与心理物理学中 2AFC（two-alternative forced choice）的经典方法论一致
- **GPT-4o 位置偏差的发现**：36% 的配对存在系统性第二图偏好，这对所有使用 LMM 做配对标注的研究都是重要的 caveat
- **模型极简但有效**：最终的 learning-to-rank 模型就是 CLIP 特征上的单层线性变换，极其轻量却能超越所有专用方法

## 局限与展望
- **数据集规模有限**：仅 1000 张图像 / 2500 图像对，限制了结论的普遍性
- **仅限日常图像**：聚焦于 Flickr 日常照片，未涉及艺术作品、科学图像、医学图像等特殊领域
- **GPT-4o 位置偏差处理过于粗暴**：简单丢弃 36% 不一致的图像对，更好的做法可能是对位置偏差建模并校正
- **Llama 3.2 和 DeepSeek 因不支持多图输入而被排除**：只比较了 GPT-4o，缺乏对更多 LMM 的系统比较
- **learning-to-rank 模型架构过简**：仅用 CLIP + 线性层，更复杂的架构可能进一步提升性能

## 相关工作与启发
- **vs CI (Commonly Interesting)**：CI 通过 Flickr 用户收藏数量定义"共识有趣性"，本文用 GPT-4o 配对标注。GPT-4o 方法与人类一致率更高（73.8% vs 69.6%），且不受平台偏差影响
- **vs VILA (Aesthetic)**：审美与有趣性高度相关但不等价——审美高的图像通常有趣，但有趣的图像不一定审美高（如杂乱但有故事性的场景）。实验中审美模型一致率 68.3%，低于 GPT-4o 的 73.8%
- **vs Memorability**：记忆性与有趣性呈负相关（r = -0.42），这个反直觉发现与先前研究一致，说明二者是不同的视觉属性维度

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统性研究 LMM 对主观视觉概念的理解，研究范式有创新
- 实验充分度: ⭐⭐⭐ 多角度分析深入，但数据规模较小且仅限 GPT-4o
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，层层递进，key insight 提炼准确
- 价值: ⭐⭐⭐⭐ 开辟了 LMM 主观视觉理解的研究方向，蒸馏范式有广泛应用前景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] A Frustratingly Simple Yet Highly Effective Attack Baseline: Over 90% Success Rate Against the Strong Black-box Models of GPT-4.5/4o/o1](../../NeurIPS2025/multimodal_vlm/a_frustratingly_simple_yet_highly_effective_attack_baseline.md)
- [\[ACL 2025\] Evaluating Visual and Cultural Interpretation: The K-Viscuit Benchmark with Human-VLM Collaboration](../../ACL2025/multimodal_vlm/evaluating_visual_and_cultural_interpretation_the_k-viscuit_benchmark_with_human.md)
- [\[ACL 2025\] OmniAlign-V: Towards Enhanced Alignment of MLLMs with Human Preference](../../ACL2025/multimodal_vlm/omnialign-v_towards_enhanced_alignment_of_mllms_with_human_preference.md)
- [\[ICCV 2025\] Unified Multimodal Understanding via Byte-Pair Visual Encoding](unified_multimodal_understanding_via_byte-pair_visual_encoding.md)
- [\[ICCV 2025\] HRScene: How Far Are VLMs from Effective High-Resolution Image Understanding?](hrscene_how_far_are_vlms_from_effective_high-resolution_image_understanding.md)

</div>

<!-- RELATED:END -->
