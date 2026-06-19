---
title: >-
  [论文解读] Synergy of Sight and Semantics: Visual Intention Understanding with CLIP
description: >-
  [ECCV 2024][多标签意图理解] 提出了 IntCLIP 框架，通过双分支编码策略将 CLIP 中的"视觉感知"（Sight）知识迁移到"语义中心"（Semantic）的多标签意图理解任务中，结合层次化类别整合和视觉辅助聚合，在标准 MIU benchmark 和图像情感识别任务上显著超越 SOTA。
tags:
  - "ECCV 2024"
  - "多标签意图理解"
  - "CLIP"
  - "双分支架构"
  - "层次类别整合"
  - "视觉语义融合"
---

# Synergy of Sight and Semantics: Visual Intention Understanding with CLIP

**会议**: ECCV 2024  
**PDF**: [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01721.pdf) / [作者版](https://marswhu.github.io/publications/files/ECCV24_IntCLIP.pdf)
**代码**: [https://github.com/yan9qu/IntCLIP](https://github.com/yan9qu/IntCLIP)  
**领域**: 其他  
**关键词**: 多标签意图理解, CLIP, 双分支架构, 层次类别整合, 视觉语义融合

## 一句话总结

提出了 IntCLIP 框架，通过双分支编码策略将 CLIP 中的"视觉感知"（Sight）知识迁移到"语义中心"（Semantic）的多标签意图理解任务中，结合层次化类别整合和视觉辅助聚合，在标准 MIU benchmark 和图像情感识别任务上显著超越 SOTA。

## 研究背景与动机

**领域现状**：多标签意图理解（Multi-label Intention Understanding, MIU）是一个新兴且极具挑战性的任务——给定一张图片，需要预测拍摄者的多个潜在意图（如"展示食物""记录风景""表达情感"等）。这些意图是高度主观和抽象的语义概念，不像传统目标检测那样有明确的视觉特征。目前 MIU 领域最大的 benchmark 是 Intentonomy 数据集。

**现有痛点**：MIU 面临两个核心困难。第一，标注数据极其稀缺——意图的模糊性使得标注过程极为耗时（需要多位标注者讨论），现有数据集规模远小于主流视觉数据集。第二，意图是"跨模态"概念——它既依赖视觉内容（图片中"拍了什么"），又依赖语义推理（"为什么拍这个"）。现有方法主要是 CNN + 分类头的架构，无法有效融合视觉感知和语义推理能力。

**核心矛盾**：MIU 需要模型理解"为什么"而非"是什么"——这需要从视觉内容到抽象意图的跨层次推理。但现有的少量标注数据不足以让模型学会这种复杂的跨层次映射。CLIP 等大规模预训练视觉语言模型拥有丰富的视觉-语义知识，但它的知识主要是"客观视觉"导向的（描述图片中"有什么"），直接应用到主观的意图理解任务效果不佳。

**本文目标** 1）如何有效利用 CLIP 的预训练知识来弥补 MIU 标注数据的不足；2）如何在 CLIP 的"客观视觉"表示和 MIU 的"主观语义"需求之间架起桥梁；3）如何处理 MIU 中层次化的标签结构（粗粒度和细粒度意图的嵌套关系）。

**切入角度**：作者敏锐地区分了两种类型的视觉表示——"Sight"（纯视觉感知，关注图像中有什么物体、什么场景）和"Semantic"（主观语义，关注图像背后的意图和情感）。CLIP 擅长前者，MIU 需要后者。作者提出不要试图让一个分支同时处理两种任务，而是设计双分支架构分别处理，再智能融合。

**核心 idea**：用 CLIP 的冻结分支保留客观视觉知识作为"锚点"，用可训练分支学习意图语义，再通过注意力聚合将视觉线索注入语义特征，实现视觉与语义的协同增效。

## 方法详解

### 整体框架

IntCLIP 采用双分支图像编码和文本对齐的架构。输入图像同时通过两个视觉编码器分支处理：Sight 分支（完全冻结 CLIP 参数）和 Semantic 分支（深层可训练）。Sight 分支保留 CLIP 原始的客观视觉表示能力，Semantic 分支通过微调适应主观语义理解任务。两个分支的特征通过 Sight-assisted Aggregation 模块融合，最终与文本特征对齐完成多标签分类。文本端，层次化类别整合（HCI）模块将多层次的意图标签转化为 CLIP 可理解的自然语言描述。

### 关键设计

1. **Sight-Semantic 双分支图像编码**:

    - 功能：同时保持 CLIP 的客观视觉能力和学习 MIU 所需的主观语义能力
    - 核心思路：两个分支共享 CLIP ViT 编码器的前几层（浅层层提取通用低级特征），在深层分叉。Sight 分支完全冻结所有参数，相当于一个只读的视觉知识库，输出的特征图 $F_{\text{sight}} \in \mathbb{R}^{N \times D}$ 包含 CLIP 训练期间学到的丰富的物体、场景和属性知识。Semantic 分支对深层的 Transformer 块解锁梯度，通过在 MIU 标注数据上微调，逐渐偏向于捕捉主观意图相关的模式。这种部分可训练的设计既避免了灾难性遗忘（Sight 分支不动），又允许模型适应目标任务（Semantic 分支可调）
    - 设计动机：如果全部冻结 CLIP，模型无法学习到 MIU 特有的语义模式；如果全部微调，少量的 MIU 数据会导致 CLIP 预训练知识的遗忘。双分支巧妙地解决了这个两难

2. **层次化类别整合（Hierarchical Class Integration, HCI）**:

    - 功能：将 MIU 数据集中多层次的意图标签转化为 CLIP 文本编码器可以有效处理的自然语言描述
    - 核心思路：MIU 的标签体系通常具有层次结构——例如"展示/分享"→"展示食物"→"展示自制甜点"。HCI 将每个细粒度标签与其上层粗粒度标签组合，生成层次化的文本描述，如 "An image showing food, specifically homemade dessert"。然后用 CLIP 文本编码器将这些描述编码为文本特征 $T \in \mathbb{R}^{C \times D}$，与图像特征进行对齐。通过引入层次上下文，文本特征能更精确地定义每个意图类别在 CLIP 嵌入空间中的位置
    - 设计动机：直接将简短的标签词（如"homemade dessert"）输入 CLIP 文本编码器，产生的特征过于简单，无法区分相似但层次不同的意图。HCI 通过补充层次上下文来丰富文本特征的语义信息，使得 CLIP 强大的句子理解能力得到充分利用

3. **视觉辅助聚合（Sight-assisted Aggregation, SAA）**:

    - 功能：将 Sight 分支的客观视觉信息注入 Semantic 分支的语义特征中
    - 核心思路：以 Semantic 分支的特征图 $F_{\text{sem}}$ 为查询（Q），Sight 分支的特征图 $F_{\text{sight}}$ 为键（K）和值（V），通过交叉注意力机制让语义特征有选择性地汲取视觉线索。$F_{\text{fused}} = \text{Softmax}(F_{\text{sem}} W_Q (F_{\text{sight}} W_K)^\top / \sqrt{d}) \cdot F_{\text{sight}} W_V + F_{\text{sem}}$。这种设计允许语义分支"查看"由 Sight 分支提供的客观视觉信息——例如当需要判断"展示食物"这一意图时，语义分支可以关注 Sight 分支检测到的食物区域的视觉特征
    - 设计动机：Semantic 分支在微调过程中可能逐渐偏离视觉底层信息而过度抽象化。SAA 确保最终的特征表示始终有视觉"锚点"支撑语义推理，既不会过于视觉化也不会过于抽象化

### 损失函数 / 训练策略

使用非对称损失（Asymmetric Loss, ASL）处理多标签分类中的正负样本不平衡问题。对于正标签使用标准交叉熵，对于负标签使用 hard threshold 截断梯度来抑制易分负样本的影响。总损失 $L = L_{\text{ASL}}(F_{\text{fused}}, T) + \lambda L_{\text{sight}}(F_{\text{sight}}, T)$，其中辅助的 Sight 损失帮助稳定训练。

## 实验关键数据

### 主实验

| 数据集/任务 | 指标 | IntCLIP | 之前SOTA | 提升 |
|------------|------|---------|----------|------|
| Intentonomy (MIU) | mAP | 36.2 | 31.8 | +4.4 |
| Intentonomy (MIU) | CF1 | 35.6 | 31.3 | +4.3 |
| Intentonomy (MIU) | OF1 | 52.8 | 49.1 | +3.7 |
| ArtPhoto (Emotion) | mAP | 73.5 | 69.2 | +4.3 |
| FI (Emotion) | Acc | 68.1 | 65.3 | +2.8 |

### 消融实验

| 配置 | Intentonomy mAP | 说明 |
|------|-----------------|------|
| Full IntCLIP | 36.2 | 完整模型 |
| Single branch (frozen) | 28.5 | 只用冻结 CLIP，不微调 |
| Single branch (finetuned) | 32.1 | 全部微调 CLIP，丢失预训练知识 |
| Dual branch w/o SAA | 33.8 | 有双分支但无视觉辅助聚合 |
| w/o HCI | 34.1 | 不使用层次化类别整合 |
| w/o ASL (用 BCE) | 34.7 | 不使用非对称损失 |

### 关键发现

- 双分支 vs 单分支是最关键的设计选择——冻结单分支只有 28.5 mAP，全微调单分支有 32.1，双分支提升到 33.8+，说明"保留+适应"的策略远优于任何极端方案
- SAA 贡献了 2.4 mAP 的提升（33.8→36.2），证明视觉信息对语义推理确实有辅助作用
- HCI 贡献了 2.1 mAP（34.1→36.2），说明层次化标签信息帮助 CLIP 更准确地定义了意图类别
- IntCLIP 在图像情感识别（另一种主观理解任务）上也有显著提升，说明框架的通用性好

## 亮点与洞察

- **"Sight vs Semantic"的二分法**非常直觉且实用——它把 CLIP 迁移到主观理解任务的问题分解为两个可管理的子问题。这种分析视角可以迁移到任何需要将预训练 VLM 适配到抽象语义任务的场景（如情感分析、美学评估、讽刺检测等）
- **双分支的冻结/微调策略**是 parameter-efficient fine-tuning 的一种变体，但比 LoRA 等方法更有结构化的解释——它不是简单地减少可训练参数，而是明确分离了"保留什么"和"学习什么"
- HCI 的想法可以直接应用到具有层次化标签的其他多标签分类任务中——在 CLIP 的嵌入空间中，层次上下文能有效增加类别间的区分度

## 局限与展望

- SAA 是单向的（从 Sight 到 Semantic），后续论文（TPAMI 2025版）已改进为双向对称聚合，说明此处有改进空间
- HCI 的模板设计较为手工，依赖于 MIU 数据集特定的标签层次结构。对于其他数据集需要重新设计模板
- 仅用 CLIP ViT-B/16 作为骨干网络，未探索更大的 ViT-L 或最新的 SigLIP/EVA-CLIP，更强的基础模型可能带来进一步提升
- 多标签意图理解中，不同意图之间的共现关系和互斥关系没有被显式建模。可以考虑引入标签图（Label Graph）来捕捉意图间的结构化关系
- 未在单标签意图预测场景下验证，可以扩展为同时支持单标签和多标签

## 相关工作与启发

- **vs DualCoOp**: DualCoOp 也使用 CLIP 做多标签分类，但通过 prompt tuning 来自适应，没有区分视觉和语义两种知识类型。IntCLIP 的双分支设计更符合 MIU 的特性
- **vs FameVIL**: FameVIL 是之前 MIU 领域的 SOTA，基于多模态 late fusion 架构。IntCLIP 利用 CLIP 的强预训练知识大幅超越了它
- **vs CoOp/CoCoOp**: 这些 prompt learning 方法为 CLIP 学习最优 prompt，但不改变视觉编码器。IntCLIP 的部分微调策略在视觉侧也做了自适应，更适合需要深度理解图像内容的任务

## 评分

- 新颖性: ⭐⭐⭐⭐ Sight-Semantic 双分支的设计思路清晰且有启发性
- 实验充分度: ⭐⭐⭐⭐ MIU 和情感识别两个任务验证，消融实验详细
- 写作质量: ⭐⭐⭐⭐ "Sight vs Semantic"的叙事贯穿全文，逻辑流畅
- 价值: ⭐⭐⭐⭐ 为主观视觉理解任务利用 VLM 提供了通用范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Domain Reduction Strategy for Non-Line-of-Sight Imaging](domain_reduction_strategy_for_non-line-of-sight_imaging.md)
- [\[ICCV 2025\] Processing and Acquisition Traces in Visual Encoders: What Does CLIP Know About Your Camera?](../../ICCV2025/others/processing_and_acquisition_traces_in_visual_encoders_what_does_clip_know_about_y.md)
- [\[ECCV 2024\] SpatialFormer: Towards Generalizable Vision Transformers with Explicit Spatial Understanding](spatialformer_towards_generalizable_vision_transformers_with_explicit_spatial_un.md)
- [\[ECCV 2024\] Dropout Mixture Low-Rank Adaptation for Visual Parameters-Efficient Fine-Tuning](dropout_mixture_low-rank_adaptation_for_visual_parameters-efficient_fine-tuning.md)
- [\[AAAI 2026\] Data Complexity of Querying Description Logic Knowledge Bases under Cost-Based Semantics](../../AAAI2026/others/data_complexity_of_querying_description_logic_knowledge_bases_under_cost-based_s.md)

</div>

<!-- RELATED:END -->
