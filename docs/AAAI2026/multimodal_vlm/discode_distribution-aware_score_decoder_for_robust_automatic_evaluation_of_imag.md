---
title: >-
  [论文解读] DisCode: Distribution-Aware Score Decoder for Robust Automatic Evaluation of Image Captioning
description: >-
  [AAAI 2026][多模态VLM][图像描述评估] 提出 DISCODE，一种免微调的测试时自适应解码器，通过引入高斯先验分布最小化 ATT 损失，使 LVLM 生成的图像描述评估分数更鲁棒地对齐人类判断，并构建了覆盖六个视觉域的 MCEval 基准。 - 核心问题： 使用 LVLM 对图像描述进行自动评分时…
tags:
  - "AAAI 2026"
  - "多模态VLM"
  - "图像描述评估"
  - "大型视觉语言模型"
  - "测试时自适应"
  - "评分鲁棒性"
  - "分布先验"
---

# DisCode: Distribution-Aware Score Decoder for Robust Automatic Evaluation of Image Captioning

**会议**: AAAI 2026  
**arXiv**: [2512.14420](https://arxiv.org/abs/2512.14420)  
**代码**: 未公开  
**领域**: 多模态VLM  
**关键词**: 图像描述评估, 大型视觉语言模型, 测试时自适应, 评分鲁棒性, 分布先验

## 一句话总结

提出 DISCODE，一种免微调的测试时自适应解码器，通过引入高斯先验分布最小化 ATT 损失，使 LVLM 生成的图像描述评估分数更鲁棒地对齐人类判断，并构建了覆盖六个视觉域的 MCEval 基准。

## 背景与动机

- **核心问题**: 使用 LVLM 对图像描述进行自动评分时，输出 token 概率分布与人类评分分布存在偏差（**符号偏差**，如数字 "0" 概率被系统性高估），导致评分不鲁棒
- **现有方法局限**: FLEUR 和 G-VEval 等方法依赖 score smoothing 技术，直接将 token 概率分布等同于评分分布 $p = p_{\text{LVLM}}$，但 token 概率分布往往是非单峰的
- **关键洞察**: 由中心极限定理，人类评分自然趋向高斯分布（单峰），而 token 概率分布因符号偏差呈现多峰现象；这种差异在绘画、抽象草图等非真实图像域更加显著
- **数据集缺口**: 现有基准（Flickr8k、Composite、Pascal-50S）仅覆盖真实图像域，无法评估评估指标的跨域鲁棒性

## 方法详解

### 1. 评分流水线与 ATT 损失

DISCODE 的评分分三步：(1) LVLM 生成原始分数 $s_{\text{raw}} \in S = \{0,1,\cdots,9\}$，同时提取最后一层解码器特征 $\boldsymbol{h}_T \in \mathbb{R}^d$ 和 token 概率 $p_{\text{LVLM}}$；(2) 通过最小化 ATT 损失估计评分分布 $p$；(3) 取期望值 $s = \mathbb{E}_{x \sim p}[x]$ 作为最终评分。

ATT 损失定义为交叉熵项与散度项之和：

$$\mathcal{L}_{\text{ATT}}(\theta; \boldsymbol{h}_T) = \underbrace{H(\psi_\theta(\boldsymbol{h}_T), p_{\text{LVLM}})}_{\text{交叉熵}} + \underbrace{D_\alpha(\psi_\theta(\boldsymbol{h}_T) \| q)}_{\text{散度正则}}$$

其中 $q(x) \propto \exp(-(x - s_{\text{raw}})^2 / 2)$ 是以原始分数为中心的高斯先验。交叉熵项使估计分布贴近 LVLM 输出，散度项通过高斯先验对分布施加单峰约束，抑制符号偏差。

### 2. 加权 KL 散度与自适应权重 α

散度项采用加权 KL 散度：

$$D_\alpha(p \| q) = (1 - \alpha) H(p, q) - \alpha H(p, p)$$

权重 $\alpha$ 基于原始分数自适应确定：

$$\alpha = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(s_{\text{raw}} - \mu)^2}{2\sigma^2}\right)$$

其中 $\mu$ 是候选数字的均值，$\sigma^2 = 0.1$。设计动机：当 LVLM 预测极端分数（接近最高/最低）时，$\alpha$ 较小，先验权重增大，更强地抑制符号偏差；当预测中间分数时，$\alpha$ 较大，更多依赖 LVLM 自身概率。$\alpha = 0.5$ 时退化为标准 KL 散度。

### 3. 解析解

DISCODE 解码器 $\psi_\theta(\boldsymbol{h}) = \text{softmax}(W^\top \boldsymbol{h} + \boldsymbol{b})$，通过线性层 + softmax 实现。在 LVLM 预测头也为线性 $p_{\text{LVLM}} = \text{softmax}(V^\top \boldsymbol{h}_T + \boldsymbol{c})$ 的假设下，ATT 损失存在解析解：

$$\hat{W} = \frac{1}{\alpha} V, \quad \hat{\boldsymbol{b}} = \frac{1-\alpha}{\alpha} \log \boldsymbol{q} + \frac{1}{\alpha} \boldsymbol{c}$$

无需迭代优化，每个样本独立高效求解，实现真正的测试时自适应。

### 4. MCEval 基准构建

覆盖 6 个视觉域（real、painting、sketch、quickdraw、clipart、infograph），共 6000 张图像 × 3 个描述 = 18000 条图像-文本对。使用 GPT-4o-mini/GPT-4o/Gemini 2.0 Flash/Claude 3.5 Sonnet 生成候选描述，81 名众包标注者进行偏好标注（三人共识）。

## 实验结果

### 表1: MCEval 基准上的跨域性能（准确率 %）

| 指标 | 类型 | Real | Painting | Sketch | Quickdraw | Clipart | Infograph | Mean |
|------|------|------|----------|--------|-----------|---------|-----------|------|
| CIDEr | ref-based | 66.7 | 64.5 | 68.7 | 62.8 | 64.5 | 60.2 | 64.6 |
| Polos | ref-based | 81.3 | 75.0 | 77.6 | 76.8 | 74.5 | 69.0 | 75.7 |
| CLIP-S | ref-free | 79.2 | 78.0 | 78.3 | 75.4 | 73.9 | 66.7 | 75.3 |
| FLEUR | ref-free | 84.7 | 83.6 | 80.4 | 45.6 | 79.9 | 86.0 | 76.7 |
| G-VEval (GPT-4o) | ref-free | 86.0 | 80.2 | 81.2 | 76.9 | 80.6 | 81.0 | 81.0 |
| FLEUR† (72B) | ref-free | 86.9 | 84.3 | 83.1 | 76.3 | 82.0 | 82.3 | 82.5 |
| **DISCODE** | **ref-free** | **87.8** | **85.2** | **83.9** | **78.5** | **83.5** | **82.8** | **83.6** |

DISCODE 在所有域上均为最优，尤其在 Quickdraw（+2.2 vs FLEUR†）等抽象域提升明显；FLEUR 在 Quickdraw 仅 45.6%，暴露了 score smoothing 的脆弱性。

### 表2: 传统真实图像基准性能

| 指标 | Flickr8k-EX τ_c | Flickr8k-CF τ_b | Composite τ_c | Pascal-50S |
|------|-----------------|-----------------|---------------|------------|
| CLIP-S | 51.2 | 34.4 | 53.8 | 80.9 |
| G-VEval (GPT-4o) | 59.7 | 38.7 | 63.0 | 82.3 |
| FLEUR† (72B) | 55.7 | 40.1 | 65.7 | 83.8 |
| **DISCODE-LV** | **56.1** | **40.2** | **66.0** | **84.5** |
| FLEUR† (IN-78B) | 56.9 | 36.4 | 64.2 | 80.8 |
| **DISCODE-IN** | **58.1** | **40.1** | **64.9** | **83.5** |

DISCODE 在真实域同样一致领先 FLEUR，在 Flickr8k-CF 上超越 GPT-4o 驱动的 G-VEval。

### 表3: 消融实验

| 变体 | FEX τ_c | FCF τ_b | Com τ_c | Pascal | MCEval |
|------|---------|---------|---------|--------|--------|
| DISCODE 完整 | 56.1 | 40.2 | 66.0 | 84.5 | 83.6 |
| 去交叉熵项 | 54.6 | 39.5 | 63.1 | 83.8 | 81.8 |
| 去散度项 | 49.9 | 39.9 | 64.4 | 83.0 | 80.9 |
| 去自适应 α | 55.6 | 40.2 | 65.4 | 84.3 | 83.0 |

三个组件均有贡献，散度项对 Flickr8k-Expert 提升最大（+6.2 τ_c）。

## 关键发现

1. **符号偏差是核心瓶颈**: token 概率分布中数字 "0" 被系统性高估，使评分分布偏离单峰，尤其在抽象域
2. **解析解实现高效自适应**: DISCODE 的闭式解避免了迭代优化开销，每个样本独立求解
3. **模型无关性**: 在 10 个开源 LVLM 上均稳定提升，更大模型收益更高
4. **评分尺度影响**: 0.0-1.0 连续尺度略优于离散尺度，因小数点位置稳定了自回归解码
5. **加权 KL 散度最优**: 对比 Jensen-Shannon、Beta、Rényi 散度，加权 KLD 表现最佳且有解析解

## 亮点

- **零微调**: 无需训练数据，完全在测试时通过解析解自适应，即插即用
- **理论驱动**: 从中心极限定理出发，用高斯先验合理约束评分分布
- **跨域鲁棒**: MCEval 六域平均 83.6%，比 FLEUR 提升 1.1 个百分点，比 G-VEval (GPT-4o) 提升 2.6 个百分点
- **新基准贡献**: MCEval 是首个覆盖六个视觉域的图像描述评估基准，填补了跨域评估空白

## 局限性

1. **仅适用于开源模型**: 需要提取解码器内部特征 $\boldsymbol{h}_T$，无法用于 GPT-4o 等闭源 API
2. **评估任务受限**: 目前仅验证了图像描述评估，未扩展到 VQA、图像生成评估等其他任务
3. **高斯先验假设**: 假定人类评分为单峰分布，在极端争议场景（两极分化评价）可能不成立
4. **解析解依赖线性头**: 闭式解要求 LVLM 预测头为线性层 + softmax，非线性头需退回迭代优化

## 相关工作

- **参考依赖指标**: BLEU → CIDEr → BERTScore → Polos → DENEB，从 n-gram 匹配到嵌入空间再到微调模型
- **参考无关指标**: CLIP-Score → PAC-S → HiFi-Score，利用预训练视觉-语言模型的对齐能力
- **LVLM 评估器**: FLEUR（score smoothing + LLaVA）、G-VEval（CoT + GPT-4o），核心假设 $p = p_{\text{LVLM}}$，DISCODE 突破了这一限制
- **测试时自适应**: 借鉴 TTA 思想但首次应用于 LVLM 评估任务

## 评分

⭐⭐⭐⭐ — 方法简洁优雅（解析解）、实验全面（10个模型×5个基准）、新基准有价值；但适用场景限于图像描述评估且依赖开源模型特征。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Toward Robust Hyper-Detailed Image Captioning: A Multiagent Approach and Dual Evaluation Metrics for Factuality and Coverage](../../ICML2025/multimodal_vlm/toward_robust_hyper-detailed_image_captioning_a_multiagent_approach_and_dual_eva.md)
- [\[AAAI 2026\] Knowledge Completes the Vision: A Multimodal Entity-aware Retrieval-Augmented Generation Framework for News Image Captioning](knowledge_completes_the_vision_a_multimodal_entity-aware_retrieval-augmented_gen.md)
- [\[CVPR 2025\] Teaching Large Language Models to Regress Accurate Image Quality Scores Using Score Distribution](../../CVPR2025/multimodal_vlm/teaching_large_language_models_to_regress_accurate_image_quality_scores_using_sc.md)
- [\[AAAI 2026\] Aligning the True Semantics: Constrained Decoupling and Distribution Sampling for Cross-Modal Alignment](aligning_the_true_semantics_constrained_decoupling_and_distr.md)
- [\[AAAI 2026\] Difference Vector Equalization for Robust Fine-tuning of Vision-Language Models](difference_vector_equalization_for_robust_fine-tuning_of_vis.md)

</div>

<!-- RELATED:END -->
