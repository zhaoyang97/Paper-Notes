---
title: >-
  [论文解读] When Semantics Mislead Vision: Mitigating Large Multimodal Models Hallucinations
description: >-
  [NeurIPS 2025][幻觉检测][语义幻觉] 发现大多模态模型（LMMs）在场景文字识别中存在"语义幻觉"问题（将无语义文本误识为语义合理的词），分析发现注意力集中于文本区域的Transformer层更不易幻觉，据此提出训练无关的ZoomText+Grounded Layer Correction框架，在TextHalu-Bench上提升约4-5%，在ST-VQA上提升约4%。
tags:
  - "NeurIPS 2025"
  - "幻觉检测"
  - "语义幻觉"
  - "场景文字识别"
  - "大多模态模型"
  - "注意力校正"
  - "训练无关"
---

# When Semantics Mislead Vision: Mitigating Large Multimodal Models Hallucinations

**会议**: NeurIPS 2025  
**arXiv**: [2506.05551](https://arxiv.org/abs/2506.05551)  
**代码**: [GitHub](https://github.com/shuyansy/MLLM-Semantic-Hallucination)  
**领域**: 幻觉检测  
**关键词**: 语义幻觉, 场景文字识别, 大多模态模型, 注意力校正, 训练无关

## 一句话总结

发现大多模态模型（LMMs）在场景文字识别中存在"语义幻觉"问题（将无语义文本误识为语义合理的词），分析发现注意力集中于文本区域的Transformer层更不易幻觉，据此提出训练无关的ZoomText+Grounded Layer Correction框架，在TextHalu-Bench上提升约4-5%，在ST-VQA上提升约4%。

## 研究背景与动机

LMMs在视觉感知和推理上表现出色，但在处理视觉模糊或非语义场景文字时容易产生"语义幻觉"——生成语义合理但视觉上不正确的答案。例如将"MMOTEL"（无语义的编辑字符）识别为"MOTEL"，将"PULLa"识别为"PULL"。**核心矛盾**：模型在大规模语义连贯文本上预训练，产生了强烈的语义先验，导致在OCR任务中依赖语义猜测而非真正的视觉定位。现有幻觉缓解工作主要关注物体幻觉和事实幻觉，OCR特有的语义幻觉几乎未被研究。本文的核心idea是利用LLM内部不同层的注意力差异——关注文本区域更多的层更不容易幻觉——来指导解码过程。

## 方法详解

### 整体框架

提出训练无关的语义幻觉缓解框架，包含两个模块：

- **输入**：图像 + 文本问题
- **输出**：校正后的场景文字识别/理解结果
- **Pipeline**：①ZoomText定位场景文本区域（无需外部检测器）→ ②Grounded Layer Correction选择最优层的隐状态融合到解码过程

### 关键设计

1. **语义幻觉的原因分析**:

    - **幻觉倾向评分**：对每个Transformer层 $\ell$，比较幻觉token $y_{hal}$ 和真实token $y_{gt}$ 的输出概率：$S_{hal}^{\ell} = P_{hal}^{\ell} / (P_{hal}^{\ell} + P_{gt}^{\ell})$
    - **文本区域注意力分数**：定义 $A_{\ell} = \frac{\sum_{i \in \mathcal{I}} \sum_{j \in \mathcal{T}} \alpha_{i,j}^{\ell}}{\sum_{i \in \mathcal{I}} \sum_{j \in \mathcal{I}} \alpha_{i,j}^{\ell}}$，衡量第 $\ell$ 层对文本区域的注意力比例
    - **关键发现**：Spearman相关分析显示，幻觉倾向与文本区域注意力呈强负相关——注意力越集中于文本区域的层越不容易幻觉

2. **ZoomText（粗到细文本区域定位）**:

    - **Glimpse步骤**：提取LLM最后一层的query-to-image交叉注意力，跨所有头和query token平均得到全局图像注意力图 $A_{text} = \frac{1}{HQ}\sum_{h=1}^{H}\sum_{q=1}^{Q} A_{q2v}^{(h,q)}$，选取top-K个token作为粗略文本区域候选
    - **Refocus步骤**：计算首尾Transformer层自注意力的归一化偏移分数 $A_{text}^{normalized} = (A_{v2v}^{(L)} - A_{v2v}^{(1)}) / (A_{v2v}^{(1)} + \epsilon)$，过滤掉注意力模式跨层稳定的非语义token（全局上下文"寄存器"），保留真正的文本区域

3. **Grounded Layer Correction (GLC)**:

    - 选择文本区域注意力最强的层：$\ell^{\star} = \arg\max_{\ell} A_{\ell}$
    - 提出三种校正策略：
        - **Replacement**：直接用 $\ell^{\star}$ 层隐状态替换最终层
        - **Selective Replacement**：仅对文本区域token替换
        - **Fusion（默认）**：加权融合 $\hat{H}_i = (1-w) \cdot H_i^{(L)} + w \cdot H_i^{(\ell^{\star})}$，$w=0.1$
    - Fusion策略在缓解幻觉和保持语义能力之间取得最佳平衡

### 损失函数 / 训练策略

完全训练无关，测试时自适应插件。ZoomText的 $K=128$（top-K image tokens），Fusion权重 $w=0.1$。无额外模块或可训练参数。可直接集成到Mini-Monkey、Qwen2.5-VL、LLaVA-NeXT等现有LMMs中。

## 实验关键数据

### 主实验

| 模型 | TextHalu-Bench | ST-VQA | TextVQA | GOT | SEED-Bench |
|------|---------------|--------|---------|-----|------------|
| GPT-4o | 45.3 | - | 71.0 | - | 70.2 |
| Mini-Monkey (baseline) | 46.5 | 66.7 | 74.1 | 88.8 | 83.3 |
| Mini-Monkey + Ours | **50.6 (+4.1)** | **70.6 (+3.9)** | **75.0 (+0.9)** | **89.2 (+0.4)** | **84.5 (+1.2)** |
| Qwen2.5-VL (baseline) | 48.3 | 67.3 | 79.1 | 85.2 | 66.7 |
| Qwen2.5-VL + Ours | **53.8 (+5.5)** | **67.6 (+0.3)** | **80.3 (+1.2)** | **86.0 (+0.8)** | **70.2 (+3.5)** |
| LLaVA-NeXT (baseline) | 27.9 | 65.1 | 65.3 | 41.9 | 50.0 |
| LLaVA-NeXT + Ours | 28.5 (+0.6) | 65.2 (+0.1) | 65.5 (+0.2) | 42.0 (+0.1) | 51.2 (+1.2) |

### 消融实验

| 配置 | TextHalu-Bench | ST-VQA | 说明 |
|------|---------------|--------|------|
| Baseline (Mini-Monkey) | 46.5 | 66.7 | -- |
| Adversarial Training | 47.5 (+1.0) | 66.8 (+0.1) | 训练式方法效果有限 |
| Chain-of-Thought | 46.8 (+0.3) | 68.2 (+1.5) | CoT对通用任务有帮助但不治本 |
| Ours (Fusion) | **50.6 (+4.1)** | **70.6 (+3.9)** | 最优方案 |
| 用外部文本检测器替代ZoomText | 50.4 (+3.9) | 70.8 (+4.1) | ZoomText接近外部检测器 |
| w/o Glimpse | 50.2 (+3.7) | 70.2 (+3.5) | Glimpse有贡献 |
| w/o Refocus | 49.8 (+3.3) | 69.5 (+2.8) | Refocus过滤噪声重要 |
| Replacement策略 | 下降 | 下降 | 直接替换破坏语义 |
| Selective Replacement | 中等提升 | 通用任务下降 | 过度覆写影响对齐 |
| Fusion (w=0.1) | **最优** | **最优** | 温和融合最佳 |

### 关键发现

1. **语义幻觉是LMMs的根本性缺陷**：即便GPT-4o在TextHalu-Bench上也仅45.3分，远低于人类96.8分
2. **不同Transformer层的幻觉倾向差异显著**：中间层往往比最后几层更能正确预测真实token
3. **注意力集中度与幻觉呈强负相关**：注意力更多分配给文本区域的层，幻觉概率更低
4. **方法对OCR能力强的模型增益更大**：Mini-Monkey和Qwen2.5-VL提升明显，但OCR能力弱的LLaVA-NeXT提升有限
5. **ZoomText无需外部检测器即可达到可比性能**："glimpse-refocus"策略有效利用了模型自身注意力

## 亮点与洞察

- 首次系统定义并研究LMMs中的"语义幻觉"问题，概念清晰且与实际应用高度相关
- 从注意力机制出发分析幻觉成因的方法论值得借鉴——直接量化层级attention与幻觉概率的关系
- ZoomText的"glimpse-refocus"是一种优雅的无需外部模块的文本区域定位方案
- 融合策略简单有效（仅一个权重$w=0.1$），计算开销极低

## 局限与展望

- 融合权重 $w$ 和 top-K 值需要手动设定，不同模型可能需要调整
- 对OCR能力本身较弱的模型（如LLaVA-NeXT）效果有限，说明方法依赖底层视觉编码能力
- ZoomText假设文本出现在语义有意义的背景上（如招牌、海报），对纯文本图像可能不适用
- TextHalu-Bench仅1740样本，覆盖场景有限

## 相关工作与启发

- 与VCD（Visual Contrastive Decoding）等幻觉缓解方法互补——VCD关注物体幻觉，本文关注文字幻觉
- 层级注意力分析可推广到其他需要精确视觉定位的任务（如细粒度识别）
- Grounded Layer Correction的隐状态融合思路可应用于其他"模型自身信息引导解码"的场景
- TextHalu-Bench的评估思路（对比语义vs非语义文本）对评估OCR模型有启发

## 评分

- 新颖性: ⭐⭐⭐⭐ 语义幻觉的定义和分析方法新颖，但解决方案（隐状态融合）相对常规
- 实验充分度: ⭐⭐⭐⭐ 多个基准、多种消融、对比多种幻觉缓解方法
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，可视化丰富，论证逻辑流畅
- 价值: ⭐⭐⭐⭐ 揭示了LMM在OCR任务中的重要缺陷，训练无关方案实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Seeing is Believing? Mitigating OCR Hallucinations in Multimodal Large Language Models](seeing_is_believing_mitigating_ocr_hallucinations_in_multimodal_large_language_m.md)
- [\[NeurIPS 2025\] Causal-LLaVA: Causal Disentanglement for Mitigating Hallucination in Multimodal Large Language Models](causalllava_causal_disentanglement_for_mitigating_hallucinat.md)
- [\[ICML 2026\] Mitigating Hallucinations in Large Vision-Language Models via Causal Route Gating](../../ICML2026/hallucination/mitigating_hallucinations_in_large_vision-language_models_via_causal_route_gatin.md)
- [\[ACL 2026\] Mitigating Hallucinations in Large Vision-Language Models without Performance Degradation](../../ACL2026/hallucination/mitigating_hallucinations_in_large_vision-language_models_without_performance_de.md)
- [\[NeurIPS 2025\] Mitigating Hallucination Through Theory-Consistent Symmetric Multimodal Preference Optimization](mitigating_hallucination_through_theory-consistent_symmetric_multimodal_preferen.md)

</div>

<!-- RELATED:END -->
