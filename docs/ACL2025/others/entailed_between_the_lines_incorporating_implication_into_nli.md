---
title: >-
  [论文解读] Entailed Between the Lines: Incorporating Implication into NLI
description: >-
  [ACL 2025][隐含蕴涵] 形式化定义"隐含蕴涵"（implied entailment）任务，将传统NLI的三分类扩展为四分类（隐式蕴涵/显式蕴涵/中立/矛盾），构建包含10K前提和40K假设的INLI数据集，实验表明微调后的模型能有效识别隐含蕴涵并跨领域泛化。 语言中的隐含信息：人类沟通高度依赖隐含表达——情感、社…
tags:
  - "ACL 2025"
  - "隐含蕴涵"
  - "NLI"
  - "语用推理"
  - "显式vs隐式蕴涵"
  - "四分类NLI"
  - "INLI数据集"
---

# Entailed Between the Lines: Incorporating Implication into NLI

**会议**: ACL 2025  
**arXiv**: [2501.07719](https://arxiv.org/abs/2501.07719)  
**代码**: [https://github.com/google-deepmind/inli](https://github.com/google-deepmind/inli)  
**作者**: Shreya Havaldar, Hamidreza Alvari, John Palowitch, Mohammad Javad Hosseini, Senaka Buthpitiya, Alex Fabrikant
**机构**: University of Pennsylvania, Google DeepMind
**领域**: 其他  
**关键词**: 隐含蕴涵, NLI, 语用推理, 显式vs隐式蕴涵, 四分类NLI, INLI数据集

## 一句话总结

形式化定义"隐含蕴涵"（implied entailment）任务，将传统NLI的三分类扩展为四分类（隐式蕴涵/显式蕴涵/中立/矛盾），构建包含10K前提和40K假设的INLI数据集，实验表明微调后的模型能有效识别隐含蕴涵并跨领域泛化。

## 研究背景与动机

**语言中的隐含信息**：人类沟通高度依赖隐含表达——情感、社交信号、讽刺等经常是隐式传递而非显式陈述。例如"读完ARR审稿意见后，Kim不得不去大吃芝士蛋糕"，除了字面意思，读者还能推断出(c)Kim觉得读审稿很不愉快、(d)Kim确实去吃了芝士蛋糕、(e)Kim吃了超常量的芝士蛋糕。

**NLI数据集的缺陷**：现有NLI基准（SNLI、MNLI、ANLI、WANLI）中隐含蕴涵极少——SNLI仅9.33%、MNLI仅3.68%、WANLI仅5.48%。唯一例外是对抗性的ANLI（15.66%），这侧面说明模型最难处理的就是隐式推理。

**模型能力不足**：在现有NLI数据集上训练的模型对隐含蕴涵的推理准确率仅约50%（随机猜测水平），而显式蕴涵可达90%以上。

**核心动机**：需要一个专门关注隐含蕴涵的NLI数据集，帮助模型学会"读懂言外之意"，并区分显式和隐式蕴涵。

## 方法详解

### 2.1 形式化隐含蕴涵

在传统三分类NLI（蕴涵/中立/矛盾）基础上，将"蕴涵"细分为两类：

- **显式蕴涵 (Explicit Entailment)**：直接从文本的词汇语义（同义词、复述）和句法（代词共指、连接等）推导而来
- **隐式蕴涵 (Implied Entailment)**：需要额外认知步骤，如逻辑推理、世界知识、会话语用或修辞性语言理解

**四分类标签**：隐式蕴涵 / 显式蕴涵 / 中立 / 矛盾

### 2.2 INLI数据集构建

数据集构建包含两个核心阶段：

**Stage 1：隐含语框架增强 (Implicature Augmentation)**

从四个现有数据集中提取隐含语框架（Implicature Frames）：

| 数据集 | 隐含语框架 | 样本数 |
|--------|-----------|--------|
| Ludwig | 问题 → 间接回答 → 隐含含义 | 1,956 |
| Circa | 对话语境 → 问题 → 间接回答 → 隐含含义 | 18,044 |
| NormBank | 行为 → 情境语境 → 隐含社会规范 | 10,000 |
| SocialChem | 社交情境 → 隐含经验法则 | 10,000 |

对于**对话隐含语**（Ludwig、Circa）：使用模板+随机人名模拟对话场景，提示Gemini-Pro从间接回答的隐含含义生成隐式蕴涵假设。

对于**情境隐含语**（NormBank、SocialChem）：从行为/社交情境生成前提，再提示Gemini-Pro基于社会规范生成隐式蕴涵。

**Stage 2：替代假设生成 (Alternative Hypothesis Generation)**

为每个前提-隐式蕴涵对生成三个额外假设（显式蕴涵、中立、矛盾）：
- 以隐式蕴涵为起点，替换必要的词或短语来转化为其他类别假设
- 确保四类假设在语义上相近，增加分类难度
- 最后对所有生成假设进行复述(paraphrase)，最小化数据中的生成痕迹

**最终规模**：约10K前提 × 4种假设 = **40K 前提-假设对**

### 2.3 数据质量验证

- **假设独立性测试**：仅用假设（不看前提）训练的模型准确率与其他NLI基准相当，说明无显著标注偏差
- **人工标注验证**：6位作者对200个样本进行标注
    - Fleiss' κ = 0.711（与ANLI的0.679-0.721和WANLI的0.60相当）
    - 多数一致率 = 0.935（至少2/3标注者与INLI标签一致）

## 实验

### 主实验：LLM在INLI上的基准测试

| 模型 | 总体准确率 | 隐含蕴涵准确率 |
|------|-----------|--------------|
| T5-Small (微调) | 0.813 | 0.731 |
| T5-Base (微调) | 0.871 | 0.817 |
| T5-Large (微调) | 0.913 | 0.870 |
| T5-XXL (微调) | **0.924** | **0.885** |
| GPT-4o (8-shot) | 0.749 | 0.608 |
| GPT-4 (8-shot) | 0.753 | 0.645 |
| Claude-3-Sonnet (8-shot) | 0.686 | 0.738 |
| Gemini-Pro (8-shot) | 0.770 | 0.628 |

**关键发现**：
1. 所有模型在隐含蕴涵上的准确率都低于总体——即便T5-XXL也仅0.885（人类上限约0.94）
2. 大型LLM的few-shot表现反而不如微调的小模型——GPT-4o隐含蕴涵准确率仅0.608
3. 即便Gemini-Pro是数据集构建所用模型，其表现同样不佳（0.628），说明生成≠理解

### 与现有NLI基准的兼容性

| 训练数据 | 标准NLI准确率 | 3-way INLI准确率 |
|---------|-------------|----------------|
| SNLI | 0.934 | 0.921 |
| MNLI | 0.916 | 0.914 |
| ANLI | 0.725 | **0.734** |
| WANLI | 0.825 | 0.822 |
| 3-way INLI | 0.778 | 0.909 |

在INLI上微调后，模型在传统NLI基准上的性能基本保持不变，ANLI上甚至略有提升（0.725→0.734），说明隐含理解能力有助于应对ANLI中的困难样本。

### 泛化实验

| 实验类型 | 训练集 | 测试集 | 准确率 |
|---------|-------|-------|--------|
| 域内泛化 | NormBank | SocialChem | 0.795 |
| 域内泛化 | SocialChem | NormBank | 0.850 |
| 跨域泛化 | 对话类 | 情境类 | 0.695 |
| 跨域泛化 | 情境类 | 对话类 | 0.796 |
| 跨数据集 | 其他3个 | SocialChem | 0.804 |
| 跨数据集 | 其他3个 | NormBank | 0.851 |

**重要发现**：从未见过NormBank但在其他三个数据集上微调的模型，在NormBank上的准确率(0.851)超过了GPT-4和Claude-3的few-shot表现——说明INLI训练能帮助模型习得可迁移的隐含推理能力。

### 现有NLI基准中隐含蕴涵占比

| 数据集 | 隐含蕴涵占比 |
|--------|------------|
| SNLI | 9.33% |
| MNLI | 3.68% |
| ANLI | 15.66% |
| WANLI | 5.48% |

验证方法：在INLI上训练T5-XXL区分显式/隐式蕴涵（97.3%准确率），再应用到其他基准。人工验证显示92.0%的模型输出与标注者一致（Cohen's κ = 0.768）。

## 亮点与洞察

1. **形式化隐含蕴涵**：首次在NLI框架中将蕴涵细分为显式和隐式，填补了自然语言推理在语用理解方面的空白
2. **巧妙的数据构建策略**：不是从头众包标注，而是利用现有隐含语数据集（Ludwig、Circa、NormBank、SocialChem）通过LLM增强转换为NLI格式，成本更低、质量更高、可复现性更强
3. **生成≠理解**：Gemini-Pro用于构建数据集，但自身在INLI上的隐含蕴涵准确率仅0.628，说明能生成隐含语不代表能理解隐含语
4. **小模型微调胜过大模型提示**：T5-XXL微调（0.885）远超GPT-4o 8-shot（0.608），凸显专门训练的重要性
5. **与现有NLI能力兼容**：在INLI上微调不会损害模型在传统NLI任务上的表现

## 局限性

1. 数据集聚焦于情境类和对话类两个领域，可能在正式文本（医疗、法律等）上泛化受限
2. 隐含语的理解具有主观性，不同文化背景的人可能对同一前提有不同理解
3. 数据集由LLM（Gemini-Pro）生成，可能存在生成偏差和多样性不足问题
4. 未进行全量人工验证，部分样本可能存在错误

## 相关工作

- **结构化隐含语**：间接问答 (Ludwig, Circa)、标量隐含语 (Jeretic et al.)、成对实体选择 (Hosseini et al.) ——受限于固定输入结构
- **隐含语框架**：NormBank (社会规范)、SocialChem (社交准则)、文化规范 (Rai et al.) ——提供隐含语素材但非NLI格式
- **隐含语理解**：通过CoT、解释生成、人类对比等方式测量LLM隐含语理解能力，结论不一
- **常识NLI**：HellaSwag、PIQA等关注物理/时间常识，但未区分显式与隐式信息

## 评分 ⭐⭐⭐⭐

- **创新性**：⭐⭐⭐⭐⭐ — 形式化隐含蕴涵任务，扩展NLI分类体系，角度新颖且有理论深度
- **实用性**：⭐⭐⭐⭐ — 为提升LLM语用理解能力提供直接可用的训练资源
- **实验充分性**：⭐⭐⭐⭐⭐ — 基准测试、兼容性验证、多维度泛化实验：体系完整
- **写作质量**：⭐⭐⭐⭐⭐ — 结构清晰，动机充分，例证丰富

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] The Time Scale of Redundancy between Prosody and Linguistic Context](the_time_scale_of_redundancy_between_prosody_and_linguistic_context.md)
- [\[CVPR 2025\] Bounds on Agreement between Subjective and Objective Measurements](../../CVPR2025/others/bounds_on_agreement_between_subjective_and_objective_measurements.md)
- [\[CVPR 2026\] Region-Wise Correspondence Prediction between Manga Line Art Images](../../CVPR2026/others/region-wise_correspondence_prediction_between_manga_line_art_images.md)
- [\[AAAI 2026\] Human Cognitive Biases in Explanation-based Interaction: The Case of Within and Between Session Order Effect](../../AAAI2026/others/human_cognitive_biases_in_explanation-based_interaction_the_case_of_within_and_b.md)
- [\[CVPR 2026\] What Is the Optimal Ranking Score Between Precision and Recall? We Can Always Find It and It Is Rarely F₁](../../CVPR2026/others/what_is_the_optimal_ranking_score_between_precision_and_recall_we_can_always_fin.md)

</div>

<!-- RELATED:END -->
