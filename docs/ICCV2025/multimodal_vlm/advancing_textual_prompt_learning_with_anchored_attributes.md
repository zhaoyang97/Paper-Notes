---
title: >-
  [论文解读] Advancing Textual Prompt Learning with Anchored Attributes
description: >-
  [ICCV 2025][多模态VLM][提示学习] 本文提出 ATPrompt，通过在文本 prompt 中嵌入通用属性 token（如颜色、形状），将软 prompt 的学习空间从一维类别级别拓展到多维属性级别，作为即插即用的模块可无缝集成到现有文本 prompt 学习方法中，在 11 个数据集上一致性提升基线性能。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "提示学习"
  - "CLIP"
  - "属性锚定"
  - "零样本泛化"
  - "可微属性搜索"
---

# Advancing Textual Prompt Learning with Anchored Attributes

**会议**: ICCV 2025  
**arXiv**: [2412.09442](https://arxiv.org/abs/2412.09442)  
**代码**: [https://github.com/zhengli97/ATPrompt](https://github.com/zhengli97/ATPrompt)  
**领域**: 多模态VLM  
**关键词**: Prompt Learning, CLIP, 属性锚定, 零样本泛化, 可微属性搜索

## 一句话总结
本文提出 ATPrompt，通过在文本 prompt 中嵌入通用属性 token（如颜色、形状），将软 prompt 的学习空间从一维类别级别拓展到多维属性级别，作为即插即用的模块可无缝集成到现有文本 prompt 学习方法中，在 11 个数据集上一致性提升基线性能。

## 研究背景与动机
视觉-语言模型（如 CLIP）通过对比学习建立图像-文本对齐，prompt learning（如 CoOp）通过学习软 prompt token 来高效适配下游任务。然而，现有文本 prompt 学习方法的核心限制在于：训练时只能将图像与预定义的已知类别对齐，无法与未知类别建立准确的关联。

这导致了一个根本性问题：模型在已知类别（base class）上表现良好，但在未知类别（novel class）上泛化能力有限，本质上是过拟合于训练时见过的类别。

直觉来源：人类面对不熟悉的类别时，会通过属性（如颜色、形状、纹理）来理解它——"猎豹是一种头小、短黄毛、有黑斑的猫科动物"远比"这是猎豹"更具描述力。属性可以作为连接未知类别与已知知识的桥梁。

本文的关键洞察：**通用（跨类别）属性比类内属性更高效、鲁棒**。将通用属性作为学习组件嵌入 prompt 模板，而非作为学习目标，从而在不增加计算开销的情况下拓展软 prompt 的表达能力。

## 方法详解

### 整体框架
ATPrompt 将传统的"软 prompt + 类别 token"格式改造为"属性软 prompt + 属性 token + 类别软 prompt + 类别 token"的混合格式。提供浅层版本和深层版本以兼容不同深度的现有方法。另外设计了一套基于 LLM 的属性池构建 + 可微搜索的自动属性选择管线。

### 关键设计

1. **属性锚定文本 Prompt（浅层版本）**:

    - 功能：在文本编码器的输入层嵌入固定的属性 token 和对应的可学习软 token
    - 核心思路：以两个通用属性 A 和 B 为例，文本 prompt 变为：$P_T = [T_{a_1}]\ldots[T_{a_m}][\text{A}][T_{b_1}]\ldots[T_{b_m}][\text{B}][T_1]\ldots[T_M][\text{CLS}]$，其中属性 token 是固定的硬 token，属性相关软 token 和类别相关软 token 均为可学习参数
    - 设计动机：属性 token 的锚定作用引导软 token 学习不仅包含类别特定的表征，还包含属性相关的通用表征。当遇到未知类别时，这些属性相关 token 提供额外信息促进更好的图文对齐

2. **深层版本**:

    - 功能：在 Transformer 深层也引入软 token，但选择性地仅丢弃和重新添加类别相关软 token，保留属性相关硬 token 和软 token
    - 核心思路：第 $i$ 层的计算为 $[\text{F}_i, \_, \text{CLS}_i] = L_i([\text{F}_{i-1}, \text{T}_{i-1}, \text{CLS}_{i-1}])$，其中 $\text{F}$ 为属性特征，跨层保持不丢弃，$\text{T}$ 为类别软 token，按传统方式丢弃重添加
    - 设计动机：完全丢弃属性 token 会破坏属性表征的跨层连续性，导致引入的新低层 token 与已有高层 token 之间产生"间隙"

3. **可微属性搜索**:

    - 功能：自动为下游任务选择最合适的属性组合及数量
    - 核心思路：（a）利用 LLM（GPT-4o）先为每个已知类别生成描述，再基于描述总结出独立的属性基（如 color, shape, size, habitat, behavior），所有组合形成搜索空间（$N$ 个基产生 $2^N - 1$ 种组合）。（b）受 DARTS 启发，将离散选择松弛为 softmax 加权和：$f(x, v; \alpha, \theta) = \sum_{i \in \mathcal{V}} \frac{\exp(\alpha_i)}{\sum_{i'} \exp(\alpha_{i'})} f(x, v_i; \theta)$。通过交替优化属性权重 $\alpha$（最小化验证损失）和软 prompt 参数 $\theta$（最小化训练损失）来找到最佳组合
    - 设计动机：直接向 LLM 询问属性无法确定最优数量，且单纯按类别名查询可能引入语义偏差。可微搜索在 token 级别操作，比传统 NAS 方法高效得多，约 5 epoch（不到 5 分钟）即可收敛

### 损失函数 / 训练策略
训练使用标准交叉熵损失：$L_{train} = \sum_{x \in D} \text{CE}(f(x; v, \theta), c)$。属性搜索使用双层优化，验证集优化 $\alpha$，训练集优化 $\theta$。搜索只需执行一次，完成后将选定属性用于正式训练。默认使用 5 个属性基（31 种候选组合）。

## 实验关键数据

### 主实验：Base-to-Novel 泛化（11 个数据集平均 HM）

| 方法 | Base | Novel | HM |
|------|------|-------|-----|
| CoOp | 82.69 | 63.22 | 71.66 |
| CoOp + ATPrompt | 82.68 | 68.04 | **74.65 (+2.99)** |
| CoCoOp | 80.47 | 71.69 | 75.83 |
| CoCoOp + ATPrompt | 81.69 | 74.54 | **77.95 (+2.12)** |
| MaPLe | 82.28 | 75.14 | 78.55 |
| MaPLe + ATPrompt | 82.98 | 75.76 | **79.21 (+0.66)** |
| PromptKD | 86.96 | 80.73 | 83.73 |
| PromptKD + ATPrompt | 87.05 | 81.82 | **84.35 (+0.62)** |

### 消融实验

| 消融项 | Base | Novel | HM | 说明 |
|--------|------|-------|-----|------|
| CoOp + ATPrompt (color, shape) | 76.27 | 70.60 | 73.33 | 默认配置 |
| 类别 token 在前 | 76.12 | 70.50 | 73.20 | 性能略降 |
| 类别 token 在中 | 76.13 | 70.29 | 73.09 | 性能略降 |
| 深层：全部丢弃重添加 | 76.83 | 70.10 | 73.31 | 破坏连续性 |
| 深层：部分丢弃重添加 | 76.87 | 70.44 | 73.51 | 仍有损失 |
| 深层：保留所有属性 token | 76.94 | 70.72 | 73.70 | 最优 |

### 关键发现
- ATPrompt 在所有 5 个基线方法上均带来一致的平均 HM 提升（+0.62 到 +2.99）
- 提升主要来自 Novel 类准确率的显著增长，证实属性作为桥梁有效连接了已知和未知类别
- 跨数据集泛化实验中提升 0.45~1.38%，域泛化实验中提升 0.33~0.90%
- 属性顺序对性能影响不大（波动在合理范围内），说明方法是鲁棒的
- 搜索出的属性（如 ImageNet 选择 color+shape）与直觉一致且比人工选择更优

## 亮点与洞察
- **即插即用的通用改进**：仅修改文本 prompt 格式而不改变模型结构或训练流程，可无缝替换任意基于文本 prompt 的方法
- **通用属性 vs 类内属性**：通用属性不需要为新类别重新获取，一次搜索即可复用，比类内属性更实用
- **可微搜索**：DARTS 思想用在属性选择上是新颖且高效的

## 局限与展望
- 仅在文本分支上改进，未涉及视觉 prompt（如 VPT）或多模态联合 prompt
- 对于已有额外可学习模块的最新方法（如 PromptKD），改进幅度有所减小
- 属性空间仍受 LLM 生成质量限制，且每个任务需独立搜索

## 搜索结果示例

| 数据集 | 属性候选 | 搜索结果 |
|--------|---------|----------|
| ImageNet | color, size, shape, habitat, behavior | (color, shape) |
| Caltech101 | shape, color, material, function, size | (shape, size) |
| OxfordPets | loyalty, affection, energy, playfulness, intelligence | (playfulness, energy) |
| StanfordCars | design, engine, performance, luxury, color | (luxury) |
| Flowers102 | color, flower, habitat, growth, season | (color, habitat, growth) |

搜索过程约 5 epoch 收敛，在单 A800 GPU 上不到 5 分钟。搜索结果与领域直觉高度一致，且自动确定了最优属性数量。

## 相关工作与启发
- **vs ArGue**: ArGue 利用 LLM 挖掘类内属性构建多组文本进行正则化和集成预测，计算开销更大；ATPrompt 将通用属性作为学习组件直接嵌入 prompt，更轻量
- **vs VCD**: VCD 用 LLM 将类名分解为类内属性，面对新类需重新获取；ATPrompt 的通用属性一次搜索即可适用于所有类别
- **vs KgCoOp/PromptSRC**: 这些方法通过正则化来缓解过拟合，但未解决 prompt 格式本身的限制；ATPrompt 从根本上扩展了 prompt 的表达空间

## 评分
- 新颖性: ⭐⭐⭐⭐ 属性锚定 prompt 的概念新颖且直觉清晰，可微搜索的结合很巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 11 个数据集×5 个基线×3 种实验设置，消融极其详尽
- 写作质量: ⭐⭐⭐⭐⭐ 动机-方法-实验逻辑清晰，与现有方法的架构比较图直观
- 价值: ⭐⭐⭐⭐ 作为即插即用模块具有很高的实用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Adaptive Prompt Learning via Gaussian Outlier Synthesis for Out-of-distribution Detection](adaptive_prompt_learning_via_gaussian_outlier_synthesis_for_out_of_distribution_detection.md)
- [\[NeurIPS 2025\] Learning Skill-Attributes for Transferable Assessment in Video](../../NeurIPS2025/multimodal_vlm/learning_skill-attributes_for_transferable_assessment_in_video.md)
- [\[CVPR 2025\] Vision-Language Model IP Protection via Prompt-based Learning](../../CVPR2025/multimodal_vlm/vision-language_model_ip_protection_via_prompt-based_learning.md)
- [\[ICCV 2025\] CoA-VLA: Improving Vision-Language-Action Models via Visual-Textual Chain-of-Affordance](coavla_improving_visionlanguageaction_models_via_visualtext.md)
- [\[CVPR 2025\] NLPrompt: Noise-Label Prompt Learning for Vision-Language Models](../../CVPR2025/multimodal_vlm/nlprompt_noise-label_prompt_learning_for_vision-language_models.md)

</div>

<!-- RELATED:END -->
