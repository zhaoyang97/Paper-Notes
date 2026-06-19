---
title: >-
  [论文解读] FALCON: Fine-grained Activation Manipulation by Contrastive Orthogonal Unalignment for Large Language Model
description: >-
  [NeurIPS 2025][LLM安全][machine unlearning] 提出 FALCON——基于表示引导的 LLM 遗忘框架，利用互信息进行参数选择、对比机制实现精细知识分离、梯度正交投影解决遗忘-保留冲突，在有害知识/版权/实体遗忘任务上全面超越现有方法。 1. LLM 安全风险：大模型可能编码有害、偏见或敏…
tags:
  - "NeurIPS 2025"
  - "LLM安全"
  - "machine unlearning"
  - "对比学习"
  - "梯度正交投影"
  - "互信息"
  - "知识解纠缠"
---

# FALCON: Fine-grained Activation Manipulation by Contrastive Orthogonal Unalignment for Large Language Model

**会议**: NeurIPS 2025  
**arXiv**: [2502.01472](https://arxiv.org/abs/2502.01472)  
**代码**: [FALCON](https://github.com/CharlesJW222/FALCON)  
**领域**: AI 安全 / 机器遗忘 / LLM 对齐  
**关键词**: machine unlearning, 对比学习, 梯度正交投影, 互信息, 知识解纠缠

## 一句话总结
提出 FALCON——基于表示引导的 LLM 遗忘框架，利用互信息进行参数选择、对比机制实现精细知识分离、梯度正交投影解决遗忘-保留冲突，在有害知识/版权/实体遗忘任务上全面超越现有方法。

## 背景与动机

1. **LLM 安全风险**：大模型可能编码有害、偏见或敏感信息，导致伦理违规和合规问题。
2. **现有方法不足**：
    - 护栏方法计算昂贵且不能抵抗对抗攻击
    - 全量重训不现实
    - 现有遗忘方法依赖粗粒度损失组合，难以精确分离知识并平衡遗忘效果与模型效用
3. **三大核心挑战**：(I1) 缺乏高效可解释的参数选择指导；(I2) 粗粒度操纵导致表示随机分散、梯度动态不可控；(I3) 遗忘后的知识可被越狱攻击恢复。

## 核心问题
如何在 LLM 中精确地遗忘特定领域知识，同时保持模型通用能力不受损，并抵抗知识恢复攻击？

## 方法详解

### 整体框架（三步走）
**Step 1: 互信息引导的参数选择** → **Step 2: 对比正交反对齐** → **Step 3: 模型更新**

### Step 1: 基于互信息的层选择

对遗忘集 $\mathcal{F}$ 和保留集 $\mathcal{R}$ 在每层 $l$ 的激活，计算互信息：

$$I(\mathcal{F};\mathcal{R}) = H(\mathcal{F}) + H(\mathcal{R}) - H(\mathcal{F},\mathcal{R})$$

选择互信息最低的层进行干预（知识纠缠最少）：

$$l^* = \arg\min_l I(\mathcal{F}^{(l)};\mathcal{R}^{(l)})$$

对多领域遗忘，定义聚合互信息：

$$I^{(l)} = \sum_{i=1}^m I(\mathcal{F}_i^{(l)};\mathcal{R}^{(l)}) + \eta\sum_{i=1}^m\sum_{j=i+1}^m I(\mathcal{F}_i^{(l)};\mathcal{F}_j^{(l)})$$

使用 KDE + PCA 降维来近似高维连续激活的熵估计。

### Step 2.1: 对比表示遗忘

构造 **Principal Offset Vectors (POVs)** 引导模型激活远离待遗忘知识的主方向。对冻结模型的激活矩阵做 SVD 获得主方向 $v_1,...,v_K$，POVs 定义为：

$$\mathcal{H}^+ = \frac{f(r \cdot (I - w\sum_{i=1}^K v_iv_i^\top), \epsilon)}{|f(r \cdot (I - w\sum_{i=1}^K v_iv_i^\top), \epsilon)|}$$

遗忘损失使用 InfoNCE：

$$\mathcal{L}_\mathcal{F} = -\frac{1}{|B|}\sum_{b=1}^{|B|}\log\frac{\exp(S_b^+/\tau)}{\exp(S_b^+/\tau)+\sum \exp(S_b^-/\tau)}$$

保留损失使用余弦相似度对齐更新模型与冻结模型在保留集上的激活：

$$\mathcal{L}_\mathcal{R} = 1 - \frac{1}{|B|}\sum \text{cos}(\mathcal{H}_b^u, \mathcal{H}_b^f)$$

### Step 2.2: 梯度正交投影

当遗忘梯度与保留梯度冲突时（$\cos(\nabla\mathcal{L}_\mathcal{F}, \nabla\mathcal{L}_\mathcal{R}) < 0$），将遗忘梯度投影到保留梯度的正交补空间：

$$\nabla\mathcal{L}_\mathcal{F}^{\text{proj}} = \nabla\mathcal{L}_\mathcal{F} - \frac{\nabla\mathcal{L}_\mathcal{F} \cdot \nabla\mathcal{L}_\mathcal{R}}{\|\nabla\mathcal{L}_\mathcal{R}\|^2}\nabla\mathcal{L}_\mathcal{R}$$

最终更新方向：$\nabla\mathcal{L}_{FALCON} = \alpha\nabla\mathcal{L}_\mathcal{F}^{\text{proj}} + \beta\nabla\mathcal{L}_\mathcal{R}$

## 实验关键数据

### 有害知识遗忘 (WMDP Benchmark)

| 方法 | WMDP-Bio ↓ | WMDP-Cyber ↓ | MMLU ↑ | PPL ↓ |
|------|-----------|--------------|--------|-------|
| Zephyr-7B 基线 | 63.7 | 43.8 | 58.1 | 1.5 |
| + RMU | 34.5 | 28.9 | 57.4 | 1.5 |
| + SCRUB | 38.7 | 35.4 | 50.0 | 16.5 |
| **+ FALCON** | **26.7** | **25.3** | **57.4** | **1.5** |
| Yi-6B-Chat 基线 | 65.4 | 42.6 | 61.8 | 1.5 |
| + RMU | 50.8 | 33.5 | 59.6 | 1.6 |
| **+ FALCON** | **27.7** | **25.3** | **60.3** | **1.5** |

### 版权内容遗忘 (MUSE Benchmark)

| 方法 | forget_know ↓ | forget_verb ↓ | retain_know ↑ |
|------|--------------|--------------|---------------|
| GradAscent | 0.00 | 0.00 | 0.00 |
| NPO | 0.56 | 0.35 | 0.51 |
| RMU | 0.48 | 0.05 | 0.51 |
| **FALCON** | **0.02** | **0.03** | **0.54** |

## 亮点
- **设计精巧**：互信息→对比学习→梯度正交投影三步环环相扣，理论动机清晰
- **MI 的双重作用**：既提供可解释的参数选择，又大幅缩减参数搜索空间
- **跨任务泛化**：在有害知识、版权、实体遗忘三大基准上均为 SOTA
- **抗恢复能力**：对越狱攻击展现出强鲁棒性

## 局限性
- MI 估计依赖 KDE + PCA 降维，PCA 保留 95% 方差的阈值选择对不同模型可能不通用
- 单层干预假设可能在知识高度分布式的模型上受限
- 多领域遗忘中 $\eta$ 的调节缺乏理论指导
- 实验主要在 7B 规模模型上验证，更大模型的可扩展性未知

## 评分
- 新颖性: ⭐⭐⭐⭐ — 对比+正交投影组合原创性强，MI 引导参数选择有新意
- 实验充分度: ⭐⭐⭐⭐⭐ — 三大基准、三种模型、多项消融与对比
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，公式推导完整
- 综合价值: ⭐⭐⭐⭐ — LLM 遗忘领域的扎实进步

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Improving Model Factuality with Fine-grained Critique-based Evaluator](../../ACL2025/llm_safety/improving_model_factuality_with_fine-grained_critique-based_evaluator.md)
- [\[NeurIPS 2025\] Geo-Sign: Hyperbolic Contrastive Regularisation for Geometrically Aware Sign Language Translation](geo-sign_hyperbolic_contrastive_regularisation_for_geometrically_aware_sign_lang.md)
- [\[NeurIPS 2025\] AgentStealth: Reinforcing Large Language Model for Anonymizing User-generated Text](agentstealth_reinforcing_large_language_model_for_anonymizing_user-generated_tex.md)
- [\[NeurIPS 2025\] PULSE: Practical Evaluation Scenarios for Large Multimodal Model Unlearning](pulse_practical_evaluation_scenarios_for_large_multimodal_model_unlearning.md)
- [\[NeurIPS 2025\] When AI Democratizes Exploitation: LLM-Assisted Strategic Manipulation of Fair Division Algorithms](when_ai_democratizes_exploitation_llm-assisted_strategic_manipulation_of_fair_di.md)

</div>

<!-- RELATED:END -->
