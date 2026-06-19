---
title: >-
  [论文解读] BiPrompt: Bilateral Prompt Optimization for Visual and Textual Debiasing in Vision-Language Models
description: >-
  [AAAI 2026][多模态VLM][视觉-语言模型去偏] 提出 BiPrompt，一种双边 prompt 优化框架，在测试时同时缓解 CLIP 等 VLM 中视觉侧（结构化注意力擦除）和文本侧（平衡 prompt 归一化）的虚假偏差，无需重训练即可提升 OOD 鲁棒性。 CLIP 等视觉-语言基础模型虽然具备出色的零样本…
tags:
  - "AAAI 2026"
  - "多模态VLM"
  - "视觉-语言模型去偏"
  - "测试时适应"
  - "因果推理"
  - "提示学习"
  - "虚假相关性"
---

# BiPrompt: Bilateral Prompt Optimization for Visual and Textual Debiasing in Vision-Language Models

**会议**: AAAI 2026  
**arXiv**: [2601.02147](https://arxiv.org/abs/2601.02147)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 视觉-语言模型去偏, 测试时适应, 因果推理, prompt优化, 虚假相关性

## 一句话总结

提出 BiPrompt，一种双边 prompt 优化框架，在测试时同时缓解 CLIP 等 VLM 中视觉侧（结构化注意力擦除）和文本侧（平衡 prompt 归一化）的虚假偏差，无需重训练即可提升 OOD 鲁棒性。

## 研究背景与动机

CLIP 等视觉-语言基础模型虽然具备出色的零样本泛化能力，但在 OOD 场景中严重依赖虚假相关性（如背景纹理、共现物体），导致可靠性下降。例如模型可能因沙滩背景将蜘蛛误分类为螃蟹。

现有去偏方法的局限：
- **区域感知方法**（如 Alpha-CLIP）：需要架构修改或代价高昂的微调，不适合即插即用的测试时适应
- **测试时 prompt 调优**（如 TPT）：假设虚假特征产生低置信度预测，但实际中强虚假特征（如水面背景）可能导致高置信度错误预测
- **SEraser**：首个关注虚假特征擦除的方法，但存在两个核心问题：(1) 随机视觉擦除可能误删因果特征；(2) 仅处理视觉偏差，忽略文本 prompt 中的语言偏差

BiPrompt 的核心动机：**同时从视觉和文本两个模态进行去偏**，实现更稳健的因果推理。

## 方法详解

### 整体框架

BiPrompt 在 SEraser 基础上引入两个互补模块：
1. **平衡 Prompt 归一化（Balanced Prompt Normalization）**：缓解文本空间的各向异性偏差
2. **结构化虚假区域擦除（Structured Spurious-Region Erasure）**：用注意力引导替代随机擦除

两者共同最小化虚假特征与预测之间的条件互信息 $I(z_s; y | z_c) \approx 0$。

### 关键设计

#### 平衡 Prompt 归一化

标准 prompt 嵌入 $f_t(t_c)$ 在文本空间中呈现各向异性，偏向高频或主导类别。BiPrompt 学习归一化的文本嵌入：

$$\hat{f}_t(t_c) = \alpha f_t(t_c) + (1-\alpha) \bar{f}_t$$

其中 $\bar{f}_t = \frac{1}{C}\sum_{c=1}^C f_t(t_c)$ 是全局语义质心，$\alpha$ 是可学习门控参数。该自适应插值鼓励各向同性的文本嵌入分布，减少语言主导性。

**核心思想**：将每个类别的文本嵌入向全局均值方向拉近，使嵌入空间更均匀，避免某些类别因文本表示占主导而获得不公平优势。

#### 结构化虚假区域擦除

用 Grad-CAM 计算软注意力图 $m(x)$ 标识因果区域，构建互补的前景和背景视图：

$$x_{\text{fg}} = m(x) \odot x, \quad x_{\text{bg}} = (1 - m(x)) \odot x$$

设计双向约束损失：

$$\mathcal{L}_{\text{BSE}} = D_{\text{KL}}(p(y|x_{\text{fg}}) \| p(y|x)) - \beta \cos(p(y|x_{\text{bg}}), p(y|x))$$

- **第一项**：前景视图与原图预测一致性（KL 散度最小化）——确保因果特征足够
- **第二项**：背景视图与原图预测正交性（余弦相似度最小化）——抑制虚假特征

相比 SEraser 的随机擦除，结构化擦除的优势：
- 利用 Grad-CAM 精确定位因果/虚假区域，避免误删因果特征
- 同时强制前景一致性和背景正交性，双向约束更稳健

### 损失函数 / 训练策略

总体测试时目标函数：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda_1 \mathcal{L}_{\text{BSE}} + \lambda_2 \mathcal{L}_{\text{ent}}$$

其中 $\mathcal{L}_{\text{ent}}$ 是防止退化均匀预测的熵正则化项。

**测试时推理流程**（4步）：
1. 计算 Grad-CAM 注意力图，得到前景/背景视图
2. 提取视觉特征 $f_v(x)$, $f_v(x_{\text{fg}})$, $f_v(x_{\text{bg}})$ 和归一化文本嵌入 $\hat{f}_t(t_c)$
3. 执行少量梯度更新最小化 $\mathcal{L}_{\text{total}}$
4. 用相似度 $s(f_v(x), \hat{f}_t(t_c))$ 计算最终预测

仅更新门控 $\alpha$ 和归一化权重等少量轻量参数，优化高效且内存友好。

## 实验关键数据

### 主实验

**表1：真实世界 OOD 数据集零样本分类（Top-1 Accuracy %）**

| 数据集 | Vanilla | TPT | RoSHOT | α-CLIP | SEraser-Blocks | **BiPrompt** |
|---|---|---|---|---|---|---|
| Tiny-ImageNet | 23.2 | 29.6 | 49.2 | **76.0** | 42.8 | 44.1 (+20.9) |
| CUB-200 | 12.1 | 8.7 | 25.5 | **44.3** | 28.9 | 31.0 (+18.9) |
| ImageNet-A | 42.1 | 49.7 | 38.9 | 51.5 | 49.7 | **52.2** (+10.1) |
| 平均 | 25.8 | 29.3 | 37.9 | **57.3** | 40.5 | 42.4 (+16.6) |

**表2：模拟虚假偏差数据集，平均/最差组准确率（%）**

| 数据集 | 指标 | Vanilla | SEraser | **BiPrompt** |
|---|---|---|---|---|
| Waterbirds | AVG | 67.7 | 78.2 | **79.9** (+12.2) |
| | W.G. | 40.0 | 65.3 | **66.6** (+26.5) |
| CamelDeer | AVG | 83.2 | 95.7 | **97.2** (+14.0) |
| | W.G. | 66.4 | 91.6 | **92.8** (+26.4) |
| SpiderCrab | AVG | 66.0 | 95.3 | **97.4** (+31.4) |
| | W.G. | 42.0 | 94.7 | **95.4** (+53.4) |
| 三数据集平均 | AVG | 72.3 | 89.8 | **91.3** (+19.0) |
| | W.G. | 49.5 | 83.7 | **85.0** (+35.5) |

### 消融实验

**跨架构泛化（表3 — Waterbirds）**：
- CLIP-L14：BiPrompt AVG 88.4（+4.7），W.G. 60.1（+27.2）
- BLIP-2：BiPrompt W.G. 35.5（+7.3），表明对不同 VLM 架构均有效

### 关键发现

1. **最差组准确率提升显著**：在 SpiderCrab 上 W.G. 提升 53.4%，说明双边去偏对虚假相关性最严重的场景效果最好
2. **BiPrompt 在所有虚假偏差数据集上均超越 SEraser**，验证了同时处理视觉和文本偏差的必要性
3. **α-CLIP 在 Tiny-ImageNet 上表现最优**是因为其检查点在 ImageNet 上重新训练过，属于不公平比较
4. TPT 在虚假偏差场景中表现反而更差，说明高置信度≠正确预测的假设不成立

## 亮点与洞察

- **双边去偏**的理念简洁有效：文本空间的各向异性偏差是一个被长期忽视的问题，BiPrompt 首次在测试时同时处理两侧偏差
- **结构化擦除**用 Grad-CAM 替代随机擦除，从"不确定性擦除"变为"精准擦除"，思路清晰
- **极轻量的测试时适应**：仅需更新门控参数和归一化权重，无需重训练或额外领域监督
- 理论上等价于最小化虚假特征与预测的条件互信息，有信息论支撑

## 局限与展望

1. **性能上限受限于 Grad-CAM 质量**：如果注意力图不准确，结构化擦除可能失效
2. **与重训练方法（α-CLIP）差距仍明显**：在 Tiny-ImageNet 上差距超过 30%，说明测试时适应的上限有限
3. **未在更大规模 VLM（如 ViT-L/14、EVA-CLIP）上验证**
4. **缺少对 prompt 归一化中 α 的分析**：α 如何随类别数和数据分布变化值得探究
5. **可扩展到多模态大语言模型（如 LLaVA）的去偏场景**

## 相关工作与启发

- **SEraser**：BiPrompt 的直接前身，提出"在虚假特征上最大化熵"的核心思想
- **TPT（Test-time Prompt Tuning）**：通过最小化增强视图间的边际熵适应，但对虚假特征不鲁棒
- **RoSHOT**：通过偏差 prompt 的正交投影去偏，但效果不稳定
- **Alpha-CLIP**：引入额外 alpha 通道提供像素级前景/背景信息，效果好但需重训练
- 启发：测试时适应的"双边"思想可推广到其他多模态任务（如 VQA 去偏、跨域检索）

## 评分

| 维度 | 分数 (1-5) |
|---|---|
| 新颖性 | 3.5 |
| 技术深度 | 3.5 |
| 实验充分性 | 4.0 |
| 写作质量 | 4.0 |
| 实用价值 | 3.5 |
| **总评** | **3.7** |


- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] TTL: Test-time Textual Learning for OOD Detection with Pretrained Vision-Language Models](../../CVPR2026/multimodal_vlm/ttl_test-time_textual_learning_for_ood_detection_with_pretrained_vision-language.md)
- [\[AAAI 2026\] Branch, or Layer? Zeroth-Order Optimization for Continual Learning of Vision-Language Models](branch_or_layer_zeroth-order_optimization_for_continual_lear.md)
- [\[CVPR 2026\] BiomedCCPL: Causal Conditional Prompt Learning for Biomedical Vision-Language Models](../../CVPR2026/multimodal_vlm/biomedccpl_causal_conditional_prompt_learning_for_biomedical_vision-language_mod.md)
- [\[CVPR 2025\] Debiasing Multimodal Large Language Models via Noise-Aware Preference Optimization](../../CVPR2025/multimodal_vlm/debiasing_multimodal_large_language_models_via_noise-aware_preference_optimizati.md)
- [\[CVPR 2026\] Interpretable Debiasing of Vision-Language Models for Social Fairness](../../CVPR2026/multimodal_vlm/interpretable_debiasing_of_vision-language_models_for_social_fairness.md)

</div>

<!-- RELATED:END -->
