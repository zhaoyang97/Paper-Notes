---
title: >-
  [论文解读] PropVG: End-to-End Proposal-Driven Visual Grounding with Multi-Granularity Discrimination
description: >-
  [ICCV 2025][社会计算][visual grounding] 提出PropVG，首个无需预训练检测器的端到端proposal-based视觉定位框架，将视觉定位分解为前景proposal生成+基于对比学习的指代评分两阶段，并引入多粒度目标判别模块（MTD）融合物体级和语义级信息判断目标是否存在，在10个数据集上刷新SOTA且推理速度比传统proposal方法快4倍。
tags:
  - "ICCV 2025"
  - "社会计算"
  - "visual grounding"
  - "proposal-based"
  - "对比学习"
  - "referring expression"
  - "target existence discrimination"
---

# PropVG: End-to-End Proposal-Driven Visual Grounding with Multi-Granularity Discrimination

**会议**: ICCV 2025  
**arXiv**: [2509.04833](https://arxiv.org/abs/2509.04833)  
**代码**: [GitHub](https://github.com/Dmmm1997/PropVG)  
**领域**: 社会计算  
**关键词**: visual grounding, proposal-based, contrastive learning, referring expression, target existence discrimination

## 一句话总结
提出PropVG，首个无需预训练检测器的端到端proposal-based视觉定位框架，将视觉定位分解为前景proposal生成+基于对比学习的指代评分两阶段，并引入多粒度目标判别模块（MTD）融合物体级和语义级信息判断目标是否存在，在10个数据集上刷新SOTA且推理速度比传统proposal方法快4倍。

## 研究背景与动机

**视觉定位任务演进**：
   - **经典VG**：REC（给框）和RES（给mask），一句话对应一个目标
   - **广义VG（GVG）**：扩展到零个或多个目标，需要判断目标是否存在

**两类主流框架的矛盾**：
   - **传统两阶段proposal方法**（MAttNet等）：依赖预训练检测器生成候选框再匹配表达式。优势是能感知全局前景物体，劣势是性能差、速度慢（320ms/帧）
   - **直接指代方法**（TransVG, SimVG等）：端到端直接预测目标位置。优势是简洁高效，劣势是只关注被指代目标，忽略其他前景物体，削弱了对"参照性"的理解和可解释性

**两个待解决的关键问题**：
   - **(1)** 如何在保持proposal的全局感知优势的同时，消除对预训练检测器的依赖，实现端到端训练？
   - **(2)** 在广义场景中，如何准确判断目标是否存在？现有方法仅用全局或单粒度预测，忽略了多粒度信息的互补

**核心insight**：视觉定位可以自然分解为"哪些是前景物体"（检测问题）+"哪个是被指代的"（匹配问题），两步端到端联合训练可以同时获得全局感知能力和指代理解能力。

## 方法详解

### 整体框架
输入图像 $\mathcal{I}$ 和文本表达式 $\mathcal{T}$ → BEiT-3多模态编码器 → SimFPN多尺度特征 → 双分支：(1) UNet解码器+SegHead做全局分割 $M_{seg}$；(2) 多尺度可变形解码器+DetHead生成前景proposal → CRS模块计算指代分数 → MTD模块判断目标是否存在。

### 关键设计1：端到端Proposal-based框架

**前景Proposal阶段**：
- 初始化 $N$ 个可学习查询 $Q_{init}$
- 通过多尺度可变形解码器与SimFPN多尺度特征交互，生成proposal查询 $Q_{prop} \in \mathbb{R}^{N \times C}$
- DetHead输出前景bbox $P_{bbox} \in \mathbb{R}^{N \times 4}$ 和置信度 $P_{score} \in \mathbb{R}^{N \times 2}$
- 用Hungarian匹配将查询分配给**所有前景物体**（不仅是被指代的目标），提供全局监督

**指代评分阶段**：
- Query Proj.将 $Q_{prop}$ 映射到 $Q_{prop}'$ 用于指代分类
- CRS模块计算每个query的指代分数

与传统两阶段方法的关键区别：不依赖外部预训练检测器，proposal和指代在统一框架中端到端训练。

### 关键设计2：Contrastive-based Refer Scoring（CRS）

CRS模块结合句子级和词级对比学习来评估每个proposal与文本表达式的相关性。

**句子级对比学习**：计算query特征 $Q_p$ 与全局文本特征 $f_s$ 的相似度矩阵 $S_{sent} \in \mathbb{R}^{N \times 1}$。全局文本特征通过valid mask pooling获得：

$$f_s^i = \max[f_t^i \times (\sim m)]$$

**词级对比学习**：计算query特征与词级文本特征 $f_t$ 的相似度矩阵 $S_{word} \in \mathbb{R}^{N \times N_t}$。

**自适应加权融合**：用可学习权重 $w_s$（由MLP+sigmoid从 $f_s$ 得到）动态平衡两个层级的贡献：

$$S_{ref} = w_s \cdot S_{sent} + (1 - w_s) \cdot \text{MaxPool}(S_{word})$$

相似度计算使用带可学习温度参数 $T$（初始化为0.07）的余弦相似度：

$$\text{Sim}(f_1, f_2) = \frac{f_1 \cdot f_2}{\|f_1\| \|f_2\|} / T$$

### 关键设计3：Multi-granularity Target Discrimination（MTD）

MTD融合物体级（检测分支的指代分数）和语义级（分割分支的mask预测）信息来判断目标是否存在。

**Score Prior Cross Attention (SPCA)**：在标准注意力基础上注入先验分数信息：

$$O = \text{Softmax}(QK^T + \text{MLP}(S))V$$

其中 $S$ 包括 $S_{ref}$（物体级先验）和 $M_{seg}$（语义级先验）。

**最终目标存在分数**：融合三个粒度的信息：

$$S_{exist} = \text{Max}(S_{ref}) \times \text{TAS}(M_{seg}) \times \varepsilon_{exist}$$

其中TAS（TopK Average Score）计算分割预测中Top-K像素的平均分数，缓解高置信异常值的影响。

### 损失函数

$$\mathcal{L}_{total} = \mathcal{L}_{seg} + \lambda_{det} \cdot \mathcal{L}_{det} + \lambda_{exist} \cdot \mathcal{L}_{exist} + \lambda_{ref} \cdot \mathcal{L}_{ref}$$

默认 $\lambda_{det}=0.1$, $\lambda_{exist}=0.2$, $\lambda_{ref}=1.0$。

## 实验

### 评估范围
10个数据集：RefCOCO/+/g（REC/RES）、gRefCOCO（GREC/GRES）、R-RefCOCO/+/g、Ref-ZOM

### 主实验结果（REC - RefCOCO）

| 方法 | Backbone | val | testA | testB |
|------|----------|-----|-------|-------|
| MAttNet*（传统proposal） | ResNet-101 | 76.65 | 81.14 | 69.99 |
| SimVG-DB | BEiT3-ViT-B | 91.47 | 93.65 | 87.94 |
| OneRef | BEiT3-ViT-B | 91.89 | 94.31 | 88.58 |
| **PropVG** | BEiT3-ViT-B | **92.70** | **95.07** | **89.58** |

PropVG在RefCOCO上全面超越同backbone的直接指代方法，比MAttNet快4倍且性能提升+14%。

### 广义视觉定位（GRES - gRefCOCO）

| 方法 | val gIoU | testA gIoU | testB gIoU |
|------|----------|------------|------------|
| HDC (Swin-B) | 68.28 | 72.52 | 63.85 |
| **PropVG** | **73.29** | **74.43** | **65.87** |

### GREC检测任务

| 方法 | val F1 | val N-acc |
|------|--------|----------|
| SimVG | 62.1 | 54.7 |
| **PropVG** | **72.2** | **72.8** |

PropVG在GREC上F1超SimVG +10.1%，"无目标"判别准确率超+18.1%。

### 消融实验

| 组件 | F1score | N-acc. | gIoU |
|------|---------|--------|------|
| Basic Setting | 63.41 | 64.11 | 65.98 |
| + SimFPN | 65.14 | 68.02 | 66.86 |
| + UNet Decoder | 65.87 | 69.19 | 68.16 |
| + Multi-scale Deformable Decoder | 67.44 | 69.47 | 69.10 |
| + Channel Splitting | 67.98 | 70.44 | 69.59 |
| + Query Proj. (Baseline) | 68.81 | 70.39 | 69.85 |
| + CRS | 70.61 | 74.78 | 71.30 |
| + MTD | 72.20 | 72.83 | 73.29 |
| - 前景监督 | 66.83 | 61.06 | 66.37 |

### 关键发现
1. CRS带来 +1.8 F1score 和 +4.4 N-acc.，词级与句级对比学习的自适应权重融合至关重要
2. 移除前景监督导致性能下降约2%，证明proposal阶段的全局物体感知为指代评分提供了有价值的先验
3. PropVG在推理速度上仅需76ms/帧，远优于传统proposal方法（MAttNet 320ms, PolyFormer 150ms, GroundingDINO 120ms）
4. 在R-RefCOCO/+/g上rIoU提升9.5~11.2%，多目标和目标缺失场景收益最大

## 亮点与洞察
1. **复兴proposal-based范式**：通过端到端设计消除了传统两阶段方法的速度和性能瓶颈，证明proposal思路本身并不过时，问题在于实现方式
2. **将VG重新定义为"检测+二分类"**：proposal阶段提供候选，指代阶段做二分决策，降低了任务复杂度
3. **多粒度判别设计精巧**：MTD将检测分数、分割预测、可学习query三路信息融合，乘法结构确保任何一路置信度低都会拉低最终分数

## 局限性
1. 需要前景物体的标注进行proposal阶段的训练，标注成本增加
2. 在无目标场景中依赖MTD的阈值设定（0.7），不同数据集可能需要调整
3. 对比MLLM方法（GSVA-13B）在部分指标上仍有差距（但参数量0.2B vs 13B+）
4. 未在开放词汇或零样本设定下评估

## 相关工作
- **REC/RES**：TransVG, LAVT, SeqTR, SimVG, OneRef
- **GVG**：ReLA, GREC, RefSegformer, HDC
- **Proposal-based**：MAttNet, NMTree, Ref-NMS
- **MLLM方法**：LISA, GSVA, GLaMM

## 评分
- 新颖性：4/5（端到端proposal-based + 多粒度目标判别，框架设计有新意）
- 技术深度：4/5（CRS的双层对比学习、MTD的SPCA机制设计合理）
- 实验充分度：5/5（10个数据集、丰富的消融、速度对比、与MLLM的参数量对比）
- 写作质量：4/5（结构清晰，图示全面）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Decide less, communicate more: On the construct validity of end-to-end fact-checking in medicine](../../ACL2026/social_computing/decide_less_communicate_more_on_the_construct_validity_of_end-to-end_fact-checki.md)
- [\[ICCV 2025\] Learning Visual Proxy for Compositional Zero-Shot Learning](learning_visual_proxy_for_compositional_zero-shot_learning.md)
- [\[ACL 2025\] taz2024full: Analysing German Newspapers for Gender Bias and Discrimination across Decades](../../ACL2025/social_computing/taz2024full_analysing_german_newspapers_for_gender_bias_and_discrimination_acros.md)
- [\[ICCV 2025\] Gradient Extrapolation for Debiased Representation Learning](gradient_extrapolation_for_debiased_representation_learning.md)
- [\[ICCV 2025\] No More Sibling Rivalry: Debiasing Human-Object Interaction Detection](no_more_sibling_rivalry_debiasing_human-object_interaction_detection.md)

</div>

<!-- RELATED:END -->
