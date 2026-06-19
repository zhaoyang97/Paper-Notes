---
title: >-
  [论文解读] HSCR: Hierarchical Self-Contrastive Rewarding for Aligning Medical Vision Language Models
description: >-
  [ACL2025][多模态VLM][Medical VLM] 提出层级自对比奖励方法 HSCR，通过视觉 token dropout 暴露模型内在的模态失对齐（misalignment），自动生成高质量偏好数据，并结合显式/隐式多层级偏好优化，仅用2000条训练样本即显著提升医学VLM的零样本性能和可信度。
tags:
  - "ACL2025"
  - "多模态VLM"
  - "Medical VLM"
  - "Preference Optimization"
  - "Self-Contrastive Rewarding"
  - "Modality Alignment"
  - "Hallucination"
---

# HSCR: Hierarchical Self-Contrastive Rewarding for Aligning Medical Vision Language Models

**会议**: ACL2025  
**arXiv**: [2506.00805](https://arxiv.org/abs/2506.00805)  
**代码**: [GitHub](https://github.com/jiangsongtao/HSCR)  
**领域**: Multimodal VLM / 医学视觉语言模型对齐  
**关键词**: Medical VLM, Preference Optimization, Self-Contrastive Rewarding, Modality Alignment, Hallucination  

## 一句话总结

提出层级自对比奖励方法 HSCR，通过视觉 token dropout 暴露模型内在的模态失对齐（misalignment），自动生成高质量偏好数据，并结合显式/隐式多层级偏好优化，仅用2000条训练样本即显著提升医学VLM的零样本性能和可信度。

## 研究背景与动机

### 问题定义
医学视觉语言模型（Med-VLM）通过将视觉编码器集成到LLM中来处理医学图像，但有限的配对多模态医学训练数据导致了严重的**模态失对齐**问题——模型可能对图像内容产生幻觉，偏好基于文本的先验而忽略实际视觉信息。这在高风险的医学场景中严重损害了模型的可信度。

### 现有方法的两大挑战

**挑战一：偏好数据采样概率低**
- 人工标注或GPT-4o生成的偏好数据与Med-VLM的解码行为存在分布偏差
- 这些外部偏好数据在Med-VLM优化时的采样概率很低
- 导致奖励信号微弱，对齐效果不佳

**挑战二：相邻层级对比的有效性不足**
- 传统的二元偏好优化（正确 vs 错误）中，偏好gap过大
- 弱训练的Med-VLM很容易饱和其区分预选/非预选输出的能力
- 粗粒度的对比无法学习到细微的偏好差异

## 方法详解

### 整体框架

HSCR 包含三个步骤：
1. **Token级自对比奖励数据生成**（Section 3.1）
2. **相似度感知偏好重排序**（Section 3.2）
3. **层级多层级偏好优化（MLPO）**（Section 3.3）

### 关键设计1：Token级自对比奖励数据生成

**核心思路**：利用Med-VLM自身的内在失对齐来生成非偏好响应，无需外部工具或标注。

**Step 1 - 视觉 Token Dropout**：
- 对原始视觉token $i$ 应用 70% 的dropout率得到 $i'$
- 分别计算完整和dropout后的token logits：$\text{logit}_\theta(y|i,x)$ 和 $\text{logit}_\theta(y|i',x)$

**Step 2 - 识别模态耦合token**：
通过对比两组logits的差异定位最容易被视觉信息影响的token：

$$P_{\text{diff}} = \text{Softmax}[(1+\beta)\cdot\text{logit}_\theta(y|i,x) - \beta\cdot\text{logit}_\theta(y|i',x)]$$

其中 $\beta=0.9$ 控制对比强度。选取logit差异最大的top-$n$个token（$n=10$），这些token与视觉模态强耦合，容易在失对齐时诱发幻觉。

**Step 3 - 生成非偏好响应**：
对识别出的敏感token，基于 $P_{\text{diff}}$ 按升序解码替换为弱视觉相关的token（即幻觉输出）。通过替换不同数量的敏感token，生成具有不同错误程度的非偏好响应集合 $\{y_{l1}, y_{l2}, ..., y_{lk}\}$。

### 关键设计2：相似度感知偏好重排序

为确保偏好排序准确反映语义差异：
- 计算每个非偏好响应 $y_{lk}$ 与偏好响应 $y_w$ 的语义相似度 $\text{sim}(y_{lk}, y_w)$
- 按相似度降序重排
- 选择相似度差异至少为0.1的 $j$ 个响应（$j=3$）用于优化

### 关键设计3：层级多层级偏好优化（MLPO）

**显式偏好学习**——区分正确与错误响应：

$$L_E = -\sum_{j=1}^{k} \mathbb{E}_{(x,y_w,y_{lj})\sim D}\left[\log\sigma\left(\gamma\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{sft}}(y_w|x)} - \gamma\log\frac{\pi_\theta(y_{lj}|x)}{\pi_{\text{sft}}(y_{lj}|x)}\right)\right]$$

这与标准DPO类似，但将偏好响应与**所有**非偏好响应进行对比（而非仅一对）。

**隐式偏好学习**——区分不同程度的错误响应：

$$L_I = -\sum_{j=1}^{k}\sum_{m=j+1}^{k} \mathbb{E}_{(x,y_{lj},y_{lm})\sim D}\left[\log\sigma\left(\gamma\log\frac{\pi_\theta(y_{lj}|x)}{\pi_{\text{sft}}(y_{lj}|x)} - \gamma\log\frac{\pi_\theta(y_{lm}|x)}{\pi_{\text{sft}}(y_{lm}|x)}\right)\right]$$

这鼓励模型学习非偏好响应之间的相对质量排序（更接近正确的应高于更远离正确的）。

**总损失**：$L_{\text{HSCR}} = L_E + L_I$

### 损失函数设计的巧妙之处
- 显式偏好提供粗粒度的对齐方向
- 隐式偏好捕捉细粒度的偏好梯度
- 两者互补，使模型同时学会"什么是正确的"和"错误有多严重"

## 实验

### 实验设置
- **视觉编码器**：CLIP-ViT-L/14@336px
- **LLM**：Mistral-7B
- **训练数据**：仅2000条训练样本
- **超参数**：$j=3, \beta=0.9, n=10, \gamma=0.1$，LoRA rank=16，2 epochs
- **评估**：Rad-VQA, SLAKE, PathVQA（开放/封闭）；captioning & instruction-following

### Med-VQA 主结果

| 方法 | RAD-VQA Open/Closed | SLAKE Open/Closed | PathVQA Open/Closed |
|------|--------------------|--------------------|---------------------|
| GPT-4o | 51.6/63.97 | 59.06/71.63 | 24.14/75.97 |
| LLaVA-Med1.5 | 32.31/56.62 | 42.45/56.49 | 10.01/59.75 |
| ST-LLaVA | 33.81/59.16 | 40.13/55.53 | 10.38/52.05 |
| LiPO | 31.85/57.37 | 43.18/58.13 | 9.37/60.17 |
| **HSCR** | **35.92/60.13** | **45.32/63.46** | **12.36/64.17** |

HSCR 在零样本设置下取得 SOTA，在 SLAKE 封闭题上提升 6.97%，在 RAD-VQA 封闭题上接近 GPT-4 水平。

### Captioning & Instruction-Following 结果

| 方法 | Conversation | Description | Overall |
|------|-------------|-------------|---------|
| LLaVA-Med1.5 SFT(60K-IM) | 58.6 | 42.5 | 54.4 |
| **HSCR (2K)** | **59.4(+0.8)** | **52.9(+10.4)** | **57.7(+3.3)** |

关键发现：2000条样本的HSCR比从10K扩展到60K的SFT带来更大的性能提升（描述任务+10.4% vs +4.4%）。

### 消融实验

**1. 显式 vs 隐式偏好**

| 显式 | 隐式 | SLAKE Closed |
|------|------|-------------|
| ✗ | ✗ | 56.49 |
| ✓ | ✗ | 57.78(+1.29) |
| ✗ | ✓ | 60.32(+3.83) |
| ✓ | ✓ | **63.46(+6.97)** |

隐式偏好单独使用时效果优于显式偏好，但二者结合效果最佳。

**2. 偏好数据构建方式**

| 方法 | SLAKE Closed |
|------|-------------|
| LLaVA-Med1.5 基线 | 56.49 |
| GPT-4o 生成偏好 | 57.96(+1.47) |
| **HSCR 自对比偏好** | **63.46(+6.97)** |

自对比偏好远优于GPT-4o外部偏好，验证了利用模型内在失对齐的有效性。

**3. 掩码策略对比**

| 策略 | SLAKE Closed | PathVQA Closed |
|------|-------------|----------------|
| Pixel-Level Mask | 57.49 | 60.79 |
| Patch-Level Mask | 58.44 | 61.83 |
| Latent Space Mask | 60.32 | 62.77 |
| **Visual Token Dropout** | **63.46** | **64.17** |

越接近LLM骨干输入的扰动越有效，视觉token dropout直接消除视觉信息传入LLM，最能有效触发内在失对齐。

**4. 掩码比例**：70%为最优，低于50%的掩码不足以有效扰动视觉信息。

### 通用多模态任务泛化性
将HSCR应用于通用VLM（LLaVA-v1.5）后，在AMBER基准上超越DPO基线，证明方法不局限于医学领域。

## 亮点与洞察

1. **巧妙利用"缺陷"**：不是试图修复模型的失对齐，而是利用这种失对齐来生成高质量的偏好数据——让模型"暴露自己的问题"来训练自己
2. **极高的数据效率**：仅2000条样本即可达到SOTA，比标准SFT扩展数据量更有效
3. **隐式偏好学习的价值**：揭示了非偏好响应之间的相对质量包含丰富的信号，传统二元DPO忽略了这些信息
4. **视觉token dropout的设计动机**：受MAE和ViT启发，通过直接在LLM输入层面消除视觉信息来暴露模态耦合程度
5. **表示学习分析**：通过t-SNE可视化展示HSCR使正确响应嵌入与图像嵌入更紧密对齐，直观验证了模态对齐效果

## 局限性

1. 医学数据质量和多样性仍然有限，影响模型在稀少病例上的泛化
2. 评估主要在受控实验环境中进行，缺乏临床工作流整合和真实世界验证
3. 70%的dropout率较高，是否会在某些场景下丢失关键视觉信息值得探讨
4. 隐式偏好的计算复杂度为 $O(k^2)$——随非偏好响应数量增加成二次增长
5. 仅在 Mistral-7B 上验证，更大规模模型或不同架构的效果未知

## 相关工作

- **VLM偏好优化**：RLHF-V（人工反馈）, POVID（扩散噪声生成拒绝响应）, RLAIF-V（多VLM聚合）
- **医学VLM对齐**：ST-LLaVA（自训练+GPT-4o评分）, MMedPO（多智能体构建偏好数据）
- **对比解码**：VCD（Contrastive Decoding减少幻觉）
- **DPO及变体**：Rafailov et al., LiPO（listwise偏好优化）

## 评分 ⭐⭐⭐⭐

方法设计巧妙且有深刻的动机分析，数据效率极高（2K样本），多层级偏好优化的设计有创新性。消融实验充分验证了各组件的贡献。不足在于仅限7B模型验证，且缺少临床环境下的真实应用评估。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Symmetrical Visual Contrastive Optimization: Aligning Vision-Language Models with Minimal Contrastive Images](symmetrical_visual_contrastive_optimization_aligning_visionlanguage.md)
- [\[ICML 2025\] MMedPO: Aligning Medical Vision-Language Models with Clinical-Aware Multimodal Preference Optimization](../../ICML2025/multimodal_vlm/mmedpo_aligning_medical_vision-language_models_with_clinical-aware_multimodal_pr.md)
- [\[ACL 2025\] Improving Medical Large Vision-Language Models with Abnormal-Aware Feedback](improving_medical_large_vision-language_models_with_abnormal-aware_feedback.md)
- [\[NeurIPS 2025\] BioCLIP 2: Emergent Properties from Scaling Hierarchical Contrastive Learning](../../NeurIPS2025/multimodal_vlm/bioclip_2_emergent_properties_from_scaling_hierarchical_contrastive_learning.md)
- [\[ACL 2025\] SPHERE: Unveiling Spatial Blind Spots in Vision-Language Models Through Hierarchical Evaluation](sphere_unveiling_spatial_blind_spots_in.md)

</div>

<!-- RELATED:END -->
