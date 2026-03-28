# PARTONOMY: Large Multimodal Models with Part-Level Visual Understanding

**会议**: NeurIPS 2025  
**arXiv**: [2505.20759](https://arxiv.org/abs/2505.20759)  
**代码**: [GitHub](https://github.com/AnselBlume/partonomy)  
**领域**: 分割 / 多模态  
**关键词**: 部件级理解, LMM, 解释性部件分割, span标记, mask反馈, benchmark

## 一句话总结
提出 Partonomy 部件级分割 benchmark（862 部件标签/534 物体标签）和 Plum 模型（用 span 标记替代 [SEG] token + mask 反馈循环），发现 SOTA 分割 LMM 在部件理解上仅 5.9% gIoU，Plum 通过避免分布偏移和利用历史预测显著提升。

## 研究背景与动机
1. **领域现状**：分割型 LMM（如 LISA、GLaMM、PixelLM）可以从文本指令生成分割 mask，在 referring expression 和推理分割上表现不错。
2. **现有痛点**：这些 LMM 尽管在 PACO、Pascal-Part 等部件数据上训练过，却几乎无法完成部件级分割（LISA-13B 仅 5.9% gIoU）。两个架构缺陷：(a) 使用预训练未见过的特殊 [SEG] token 导致分布偏移；(b) 顺序生成多个 mask 时丢弃之前的预测，无法利用历史信息。
3. **核心矛盾**：部件理解需要细粒度的组合推理（一个物体由哪些部件组成？两个物体共享哪些部件？），但现有 LMM 和评估集都缺乏覆盖这种推理的能力和设计。
4. **本文要解决什么**：(1) 构建全面的部件级 LMM 评估基准；(2) 设计解决上述两个架构缺陷的新模型。
5. **切入角度**：分割型 LMM 的 [SEG] token 是预训练后添加的，必然引起分布偏移；如果用已有的文本 span 来指示分割区域，就能保留预训练表示。
6. **核心idea一句话**：用 BIO span 标记替代 [SEG] token 避免分布偏移，加 mask 反馈循环让后续预测能利用历史 mask 信息。

## 方法详解

### 整体框架
Plum = LLaVA（视觉-语言模型）+ 双向 Span Extractor（自动标记哪些文本 token 对应需分割的部件）+ SAM mask decoder（带 FiLM mask 反馈）。输入图像+部件问题 → LMM 生成文本回答 → Span Extractor 标记部件名 span → 投影为 mask query → 增强的 SAM decoder 生成分割 mask（利用之前的 mask 作为反馈）。

### 关键设计

1. **Span Extractor（替代 [SEG] token）**:
   - 做什么：在 LMM 输出的 token embedding 上做 BIO 标注，自动识别哪些文本 span 是需要分割的部件名
   - 核心思路：先经 MLP，再通过双向 Transformer encoder 获取全局上下文（因为 LLM 的因果 mask 阻止看后续 token），最后投影为 B/I/O 标签
   - 设计动机：[SEG] token 在预训练中从未见过，其 embedding 是随机初始化的，微调会扰动原有 token 分布。Span tagging 完全基于已有 vocabulary，不引入新 token
   - 效果：保留了 LMM 的文本推理能力，zero-shot 分割和 VQA/幻觉基准上都优于 LISA/GLaMM

2. **KL 约束的 Query 投影**:
   - 做什么：将标记为 B/I 的 token embedding 投影为 mask query 时，用 KL 散度约束不偏离预训练的 frozen teacher embedding
   - 核心思路：$\mathcal{L}_{\text{KL}} = \frac{1}{N_+}\sum_{s} \frac{\|h^L_{i_s:j_s} - t^L_{i_s:j_s}\|_2^2}{2\sigma^2}$
   - 设计动机：防止分割微调过程中 hidden state 偏离原始语言表示空间，保护文本推理能力

3. **Mask 反馈循环**:
   - 做什么：将之前预测的 mask 通过 FiLM 层注入 SAM 的 mask encoder，使后续 mask 预测能看到历史预测
   - 核心思路：对每个历史 mask 用修改后的 mask encoder 编码为文本增强的特征图，然后通过 patch-wise attention pooling 将所有历史 mask 聚合为单一特征图，传递给 mask decoder
   - 设计动机：部件间有空间关系（如翅膀左右对称），看到之前分割了左翅膀有助于更准确地分割右翅膀

### 损失函数 / 训练策略
$\mathcal{L} = \mathcal{L}_{\text{LM}} + \lambda_1 \mathcal{L}_{\text{span}} + \lambda_2 \mathcal{L}_{\text{KL}} + \lambda_3 \mathcal{L}_{\text{seg}} + \lambda_4 \mathcal{L}_{\text{BCE}}$。分割损失用 Focal-Tversky loss（偏重 recall $\alpha=0.7$，precision $\beta=0.3$）。两阶段微调：先在混合分割数据上训练，再在 Partonomy 训练集上微调。

## 实验关键数据

### 主实验（Partonomy-Core 分割 gIoU, %）

| 方法 | 额外分割数据 | Part Identification (macro) | Part Intersection (macro) | Part Difference (macro) |
|------|------------|---------------------------|--------------------------|------------------------|
| LISA-13B | ✗ | 7.0 | 7.5 | 7.1 |
| PixelLM-13B | ✓ | 8.4 | 4.8 | 4.8 |
| GLaMM | ✓ | 5.9 | 6.2 | 6.0 |
| **Plum-13B (zero-shot)** | ✗ | **27.4** | **29.9** | **24.8** |
| LISA-13B (ft) | ✗ | 35.4 | 38.4 | 31.6 |
| GLaMM (ft) | ✓ | 38.8 | 42.1 | 34.8 |
| **Plum-13B (ft)** | ✗ | **41.6** | **45.9** | **39.4** |
| Grounded SAM 2 (gt) | – | 16.8 | 23.6 | 17.1 |

### 消融实验
| 配置 | 关键发现 | 说明 |
|------|---------|------|
| Span Extractor vs [SEG] token | Span 在 VQA/幻觉基准上保留推理能力 | 证明 [SEG] token 引起分布偏移 |
| 有 vs 无 Mask 反馈 | 反馈改善 Part-Whole 推理分割 | 历史 mask 提供空间先验 |
| 有 vs 无 KL 约束 | KL 保护文本推理质量 | 防止 hidden state 漂移 |
| Zero-shot Plum vs 训练后的 LISA | Plum zero-shot (27.4) > LISA zero-shot (7.0) | 4× 提升 |

### 关键发现
- SOTA 分割 LMM 在部件级理解上全面崩溃，证明部件理解是当前 LMM 的真空地带
- [SEG] token 的分布偏移是性能差的主要原因之一——Plum 的 span 标记方案使 zero-shot 性能提升 4×
- Partonomy-Core 的 862 部件标签是现有最大规模的部件标注（4× PACO）
- 即使是 GPT-4o，在需要精细部件比较的问题上准确率也有限

## 亮点与洞察
- **揭示 LMM 部件理解瓶颈**：5.9% gIoU 这个数字极其震撼，说明"分割型 LMM 能理解图像"的说法至少在部件级别不成立。这对 LMM 评估方法提出了新挑战。
- **Span 标记的通用性**：用文本本身的 span 而非特殊 token 来指示分割，是一种更优雅的文本-视觉对齐方式，避免了预训练/微调阶段的 vocabulary 不匹配。可以推广到任何需要将文本与视觉区域关联的任务。
- **部件间推理任务（Part Comparison/Reasoning）**：Partonomy 不只是识别部件，还要求比较部件（两个物体的共有/差异部件）和推理部件-整体关系。这种组合推理能力是走向细粒度视觉理解的关键。

## 局限性 / 可改进方向
- Plum 微调后在 Partonomy-Core 上仍只有约 41% macro-gIoU，距实用还有很大差距
- Partonomy-Core 仅 1K 图片，部件标注可能存在不完整性（有些可见部件未被标注）
- 模型设计依赖 SAM 的 mask decoder，对超细粒度部件（如电路板上的元件）可能分辨率不足
- 未探索与 SAM 2 等更新分割模型的结合

## 相关工作与启发
- **vs LISA (Lai et al., 2024)**：LISA 用 [SEG] token 触发分割，本文证明这引起严重分布偏移；Plum 的 span 标记方案提供了更好的替代
- **vs GLaMM (Rasheed et al., 2024)**：GLaMM 训练数据更多但在部件上表现差，说明数据量不能补偿架构缺陷
- **vs SegLLM (基于 HIPIE)**：SegLLM 也有类似 mask 反馈机制，在部件上表现最好（32.4 macro-gIoU），验证了 mask 反馈的重要性

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个部件级 LMM benchmark + span 标记替代 [SEG] token 的新范式
- 实验充分度: ⭐⭐⭐⭐ 多基线对比、消融完整、text+segmentation 双评估
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，论据充分
- 价值: ⭐⭐⭐⭐⭐ 揭示 LMM 的重要盲区，benchmark + 方法论双贡献
