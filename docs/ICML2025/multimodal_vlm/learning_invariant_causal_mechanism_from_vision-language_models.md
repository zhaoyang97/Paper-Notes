---
title: >-
  [论文解读] Learning Invariant Causal Mechanism from Vision-Language Models
description: >-
  [ICML2025][多模态VLM][CLIP] 通过因果分析证明 CLIP 嵌入是真实不变/可变因子的线性变换，提出 CLIP-ICM 框架利用干预数据估计线性投影矩阵，将预测限定在不变子空间中以实现跨环境一致预测。 领域现状 领域现状：CLIP 在零样本任务上表现优异，但 fine-tune 到 OOD 场景时性能不稳定…
tags:
  - "ICML2025"
  - "多模态VLM"
  - "CLIP"
  - "OOD generalization"
  - "causal inference"
  - "invariant representation"
  - "linear projection"
---

# Learning Invariant Causal Mechanism from Vision-Language Models

**会议**: ICML2025  
**arXiv**: [2405.15289](https://arxiv.org/abs/2405.15289)  
**代码**: [GitHub](https://github.com/ZeenSong/CLIP-ICM)  
**领域**: 多模态VLM  
**关键词**: CLIP, OOD generalization, causal inference, invariant representation, linear projection

## 一句话总结
通过因果分析证明 CLIP 嵌入是真实不变/可变因子的线性变换，提出 CLIP-ICM 框架利用干预数据估计线性投影矩阵，将预测限定在不变子空间中以实现跨环境一致预测。

## 研究背景与动机

### 领域现状

**领域现状**：CLIP 在零样本任务上表现优异，但 fine-tune 到 OOD 场景时性能不稳定

### 解决思路

**解决思路**：在 Terra Incognita 数据集上，leave-one-out fine-tune 后目标域准确率仅 47.8%（直接 fine-tune 78.9%），差距高达 31.1%

### 现有痛点

**现有痛点**：Fine-tune 后零样本新类能力也显著下降（63.6% → 24.6%）

### 核心矛盾

**核心矛盾**：因果分析**：SCM 中图像由不变因子 $Z_{inv}$（如翅膀形状）和可变因子 $Z_{var}$（如羽毛颜色）生成，环境变化只影响 $Z_{var}$

### 补充说明

**补充说明**：基于 $Z_{inv}$ 的预测机制在不同环境间保持不变（Proposition 5.1），而依赖 $Z_{var}$ 的预测则不一致

## 方法详解

### 理论基础
1. **可识别性分析**（Proposition 5.3）：在 Condition 5.2 下，CLIP 图像编码器输出是真实潜变量的**线性变换**：$f_I(\mathbf{x}) = A\mathbf{z}$，其中 $A$ 可逆
2. **投影矩阵存在性**（Proposition 5.5）：利用干预数据（固定 $z_{inv}$，变化 $z_{var}$），可估计 $A_{inv}$ 满足 $A_{inv}(f_I(\mathbf{x}_1^{do}) - f_I(\mathbf{x}_2^{do})) = 0$
3. **OOD 风险保证**（Theorem 5.6）：当 $I(Z_{inv};Z) > c$ 时，不变预测器的 OOD 风险严格小于常规预测器

### CLIP-ICM 三阶段
1. **收集干预数据**：
    - 图像方式：数据增强（颜色扰动、灰度、高斯模糊）保持 $Z_{inv}$ 不变
    - 文本方式：用图像描述模型生成文本，再用 LLM 修改变体因子
2. **估计 $A_{inv}$**：学习投影矩阵使干预对的嵌入差在不变子空间中为零
3. **不变预测**：在不变子空间中计算图像与文本嵌入的余弦相似度做分类

### 不变预测器
$$P_{inv}(c|\mathbf{x}) = \frac{\exp(S(A_{inv}f_I(\mathbf{x}), A_{inv}f_T(\mathbf{t}_c)))}{\sum_{c'}\exp(S(A_{inv}f_I(\mathbf{x}), A_{inv}f_T(\mathbf{t}_{c'})))}$$

## 实验关键数据


### 主实验

| 方法 | PACS | VLCS | OfficeHome | TerraInc | DomainNet | Avg |
|------|------|------|------------|----------|-----------|-----|
| Zero-shot | 96.1 | 82.4 | 71.5 | 34.2 | 56.8 | 68.2 |
| Linear-Probe | 96.4 | 78.7 | 81.9 | 60.2 | 55.0 | 74.4 |
| CLIP-Adapter | 96.4 | 84.3 | 82.2 | 57.5 | 59.9 | 76.1 |
| CLIP-ICM | **最优** | **最优** | **最优** | **最优** | **最优** | **最优** |

- 在 DomainBed 基准上全面超越 CoOp、CoCoOp、CLIP-Adapter、DPL 等方法
- 在 ImageNet 变体上也展现出优势
- 不需要重训 CLIP backbone，计算成本低

## 亮点与洞察
- **理论驱动**：从因果可识别性出发推导出线性投影的存在性，理论链条完整
- **简洁高效**：仅需学一个线性矩阵，不重训backbone，实用性强
- **两种干预数据收集方式**：图像增强+文本编辑，灵活适配不同场景
- **OOD 风险理论保证**：不仅是经验有效，还有严格的理论下界

## 局限与展望
- 线性变换假设（Proposition 5.3）对所有 CLIP 模型是否成立需进一步验证
- 干预数据的质量直接影响 $A_{inv}$ 的估计精度
- 图像增强作为干预可能无法完全保持 $Z_{inv}$ 不变
- 仅考虑了分类任务，检索、生成等下游任务未验证
- 不变子空间的维度需要预先设定，最优维度选择缺乏理论指导
- 文本编码器的不变性投影与图像编码器需共享同一个 $A_{inv}$，但两者的表示特性可能不同
- 当环境变化不仅体现在 $Z_{var}$ 分布偏移，还涉及新概念出现时（concept shift），方法是否仍然有效
- $I(Z_{inv};Z) > c$ 的条件在实际数据中难以验证
- 对于大型数据集（如 ImageNet 规模），收集足够质量的干预数据可能成本较高
- 与 fine-tuning 方法的结合（如 LoRA + CLIP-ICM）是一个有前景的方向
- 方法假设 CLIP 已经具备较好的可识别性（Condition 5.2），对于较小或领域特定的 VLM 可能不成立

### 补充实验细节
- DomainBed 上采用标准 leave-one-out 评估协议
- ImageNet 变体包括 ImageNet-V2、ImageNet-R、ImageNet-Sketch、ImageNet-A
- 投影矩阵估计使用梯度下降优化，收敛通常在几百次迭代内
- 文本干预使用 GPT-3.5 生成，图像干预使用标准数据增强组合
- 在 Terra Incognita 上 domain shift + open class 联合评估是本文独特贡献

## 相关工作与启发
- **CoOp/CoCoOp**（Zhou et al., 2022）：可学习prompt，但缺乏OOD理论保证
- **IRM**（Arjovsky et al., 2020）：不变学习，但未利用VLM特性
- **因果表示学习**（Schölkopf et al., 2021）：本文将其与CLIP结合是新颖贡献
- 启发：VLM的跨模态对齐天然提供了因果可识别性条件

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ (因果视角分析CLIP的OOD问题+线性投影理论)
- 实验充分度: ⭐⭐⭐⭐ (多个benchmark+消融)
- 写作质量: ⭐⭐⭐⭐ (因果分析推导严谨)
- 价值: ⭐⭐⭐⭐⭐ (为VLM的OOD泛化提供了理论和实践方案)

### 核心理论补充
- Condition 5.2 要求存在 D+1 个文本描述对，保证矩阵 A 可逆
- Theorem 5.6 的条件 $I(Z_{inv};Z) > c$ 确保不变因子包含足够信息
- 因果机制的一致性通过 do-calculus 证明：$P^*(y|do(\mathbf{z}_{inv})) = P(y|do(\mathbf{z}_{inv}))$
- 投影矩阵估计基于对比学习目标，最小化干预对在不变子空间的差异
- 图像干预与文本干预可单独或组合使用，实验表明组合效果最佳

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] BiomedCCPL: Causal Conditional Prompt Learning for Biomedical Vision-Language Models](../../CVPR2026/multimodal_vlm/biomedccpl_causal_conditional_prompt_learning_for_biomedical_vision-language_mod.md)
- [\[ICCV 2025\] Causal Disentanglement and Cross-Modal Alignment for Enhanced Few-Shot Learning](../../ICCV2025/multimodal_vlm/causal_disentanglement_and_cross-modal_alignment_for_enhanced_few-shot_learning.md)
- [\[ICML 2025\] Kernel-based Unsupervised Embedding Alignment for Enhanced Visual Representation in Vision-language Models](kernel-based_unsupervised_embedding_alignment_for_enhanced_visual_representation.md)
- [\[ICCV 2025\] Dynamic Multimodal Prototype Learning in Vision-Language Models](../../ICCV2025/multimodal_vlm/dynamic_multimodal_prototype_learning_in_vision-language_models.md)
- [\[CVPR 2025\] NLPrompt: Noise-Label Prompt Learning for Vision-Language Models](../../CVPR2025/multimodal_vlm/nlprompt_noise-label_prompt_learning_for_vision-language_models.md)

</div>

<!-- RELATED:END -->
