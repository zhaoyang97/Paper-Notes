---
title: >-
  [论文解读] Beyond Words: Augmenting Discriminative Richness via Diffusions in Unsupervised Prompt Learning
description: >-
  [CVPR 2025][多模态VLM][VLM] 提出AiR（Augmenting discriminative Richness）方法，利用LoRA微调的Stable Diffusion生成合成图像构建辅助分类器，与文本分类器互补融合，将无监督prompt learning中的文本-图像匹配扩展为图像-图像匹配，显著提升细粒度/遥感等困难数据集上的分类准确率。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "VLM"
  - "提示学习"
  - "扩散模型"
  - "伪标签"
  - "auxiliary classifier"
---

# Beyond Words: Augmenting Discriminative Richness via Diffusions in Unsupervised Prompt Learning

**会议**: CVPR 2025  
**arXiv**: [2504.11930](https://arxiv.org/abs/2504.11930)  
**作者**: Hairui Ren, Fan Tang, He Zhao, et al.
**机构**: Jilin University, Chinese Academy of Sciences (CAS), CSIRO
**领域**: 多模态VLM  
**关键词**: VLM, unsupervised prompt learning, diffusion model, pseudo-label, auxiliary classifier

## 一句话总结
提出AiR（Augmenting discriminative Richness）方法，利用LoRA微调的Stable Diffusion生成合成图像构建辅助分类器，与文本分类器互补融合，将无监督prompt learning中的文本-图像匹配扩展为图像-图像匹配，显著提升细粒度/遥感等困难数据集上的分类准确率。

## 研究背景与动机
**领域现状**：CLIP等视觉语言模型通过文本-图像对齐实现了强大的零样本分类能力，但在特定下游任务上仍有明显性能差距。Prompt learning（如CoOp、CoCoOp）通过学习连续prompt来适配下游任务，但大多需要标注数据。无监督prompt learning（如UPL、CPL）利用伪标签避免标注，但伪标签的噪声会严重影响学习质量。

**现有痛点**：(1) 纯文本prompt的表达能力有限，难以捕捉视觉细节的细粒度语义差异（如不同花卉品种、遥感地物类型）；(2) 伪标签依赖CLIP的初始文本分类器，而该分类器在困难数据集上本身准确率就不高，形成"鸡蛋悖论"；(3) 现有方法仍然局限在文本-图像的匹配范式中，忽略了图像-图像相似性可能提供的判别信息。

**核心矛盾**：文本描述与视觉特征之间存在固有的模态鸿沟——同一类别的文本描述是唯一确定的，但视觉表现却高度多样化。仅依赖文本prompt无法充分表达类内多样性和类间差异性。

**本文解决什么？** 如何在无标注数据的情况下，利用生成模型的"先验知识"来增强分类器的判别能力，弥补纯文本prompt在视觉判别上的不足。

**切入角度**：将扩散模型作为"视觉知识库"，通过生成类别代表性样本来构建辅助的图像-图像分类器，与原始文本分类器加权融合。

**核心 idea**：用扩散模型生成的合成图像作为每个类别的"视觉原型"，将分类从文本-图像匹配扩展为文本+图像到图像的联合匹配。

## 方法详解

### 整体框架
AiR方法包含三个核心模块：(1) LoRA微调的Stable Diffusion生成器，用于生成高质量的类别代表性合成图像；(2) ACG（Auxiliary Classifier Generation）模块，从合成图像中选择代表性样本构建辅助分类器；(3) PLG（Pseudo-Label Generation）模块，融合文本分类器和辅助分类器生成更准确的伪标签。

### 关键设计

1. **LoRA微调的扩散模型**:

    - 功能：对Stable Diffusion进行领域自适应，使生成图像更贴合目标数据集的视觉分布
    - 核心思路：使用目标数据集的无标注图像，通过LoRA对Stable Diffusion的U-Net进行轻量微调。LoRA仅更新低秩分解矩阵，参数量极小（约0.1%的原始参数）
    - 设计动机：预训练的Stable Diffusion在通用场景图像上表现良好，但在遥感、医学等专业领域的生成质量不足。LoRA微调可以以极低成本适配目标域分布，生成更具判别性的合成样本
    - 效果：LoRA微调带来+3.4%～+8%的跨数据集准确率提升

2. **ACG模块——辅助分类器构建**:

    - 功能：为每个类别生成 $M$ 张合成图像，并从中选择最具代表性的样本作为类别原型
    - 核心思路：使用类别名称作为文本prompt（如"a photo of a residential area"）生成 $M$ 张图像。通过CLIP视觉编码器提取特征，计算每张合成图像与该类别所有合成图像均值特征的余弦相似度，选择相似度最高的 $K$ 张作为代表样本。辅助分类器的预测为：$\hat{p}_c = \frac{1}{K}\sum_{k=1}^{K} \text{sim}(f_{\text{img}}, f_{\text{syn},k}^c)$
    - 设计动机：直接使用所有合成图像会引入噪声（部分生成图像质量差或偏离类别语义），选择最接近类别中心的样本可以提升辅助分类器的可靠性。实验发现约120张/类是最优数量

3. **PLG模块——伪标签融合**:

    - 功能：融合文本分类器和辅助分类器的预测，生成更准确的伪标签
    - 核心思路：最终预测为加权融合 $p_c^* = p_c + \lambda \hat{p}_c$，其中 $p_c$ 是CLIP文本分类器的预测概率，$\hat{p}_c$ 是辅助分类器的预测概率，$\lambda$ 控制辅助分类器的权重
    - 设计动机：两个分类器提供互补的判别信息——文本分类器擅长捕捉语义级别的类别特征，辅助分类器擅长捕捉视觉纹理和结构级别的差异。加权融合可以取长补短

4. **训练损失**:

    - 总训练损失 $L = L_r + \beta L_s$
    - $L_r$ 是基于融合伪标签的交叉熵损失，用于学习连续prompt
    - $L_s$ 是辅助的自监督正则化损失，通过augmented view的一致性约束防止过拟合伪标签中的噪声
    - $\beta$ 控制正则化强度

## 实验关键数据

### 主实验：与SOTA方法的准确率对比

| 方法 | RESISC45 | Flowers102 | EuroSAT | DTD | 平均 |
|------|----------|------------|---------|-----|------|
| CLIP零样本 | 60.2% | 66.1% | 42.0% | 43.8% | 53.0% |
| UPL | 72.4% | 65.8% | 48.3% | 55.2% | 60.4% |
| CPL | 77.3% | 69.2% | 52.1% | 57.9% | 64.1% |
| **AiR (本文)** | **79.9%** | **71.4%** | **55.7%** | **60.1%** | **66.8%** |
| vs CPL提升 | +2.6% | +2.2% | +3.6% | +2.2% | +2.7% |

### 消融实验

| 组件配置 | RESISC45 | 说明 |
|---------|----------|------|
| Baseline (CPL) | 70.6% | 无辅助分类器 |
| + $\hat{p}_c$ (辅助分类器) | 72.3% | +1.7%，证明图像-图像匹配有效 |
| + $L_s$ (正则化损失) | 72.9% | +2.3%，正则化减少伪标签噪声 |
| + $\hat{p}_c$ + $L_s$ | 73.6% | +3.0%，两者互补 |
| + LoRA微调 | 76.5% → 79.9% | LoRA额外带来+3.4%～+6.3%提升 |
| 无LoRA vs 有LoRA | 76.5% vs 79.9% | LoRA微调是关键组件 |

### 合成样本数量的影响

| 每类合成样本数 | RESISC45 | Flowers102 | EuroSAT |
|--------------|----------|------------|---------|
| 20 | 77.1% | 68.9% | 52.8% |
| 60 | 78.4% | 70.1% | 54.2% |
| 120 | **79.9%** | **71.4%** | **55.7%** |
| 200 | 79.6% | 71.1% | 55.3% |
| 300 | 79.2% | 70.8% | 54.9% |

### 关键发现
- **辅助分类器的互补性**：通过辅助分类器 $\hat{p}_c$ 的加入，在RESISC45上提升1.7%，验证了图像-图像匹配相比文本-图像匹配能捕捉更丰富的视觉判别信息
- **LoRA微调至关重要**：LoRA微调在所有数据集上带来+3.4%～+8%的提升，说明领域自适应对合成图像质量的影响是决定性的
- **最优合成样本数约120张/类**：过少样本（<60）不足以覆盖类内多样性，过多样本（>200）引入噪声反而降低性能
- **在困难数据集上提升更大**：EuroSAT（+3.6%）和DTD（+2.2%）是文本描述难以精确区分的细粒度/遥感数据集，辅助分类器的优势更加明显
- **$L_s$ 正则化独立有效**：即使不加辅助分类器，仅加正则化损失也能提升2.3%，说明自监督一致性约束有效缓解了伪标签噪声

## 亮点与洞察
- **将生成模型作为"视觉知识库"**：不同于传统的"生成→增强训练数据"范式，AiR将合成图像直接作为分类器的一部分（类别原型），避免了将合成数据与真实数据混合训练导致的域差距问题
- **文本+视觉双通道分类**：融合文本-图像匹配和图像-图像匹配的思路，可以看作是在CLIP的特征空间中同时利用了语言和视觉两种"锚点"进行分类
- **LoRA微调的成本效益比极高**：仅微调约0.1%的参数就带来显著提升，且不需要标注数据（使用目标域无标注图像即可），在实际应用中非常实用
- **与伪标签方法正交**：AiR的辅助分类器理论上可与任何基于伪标签的无监督方法结合，具有良好的可扩展性

## 局限与展望
- 合成图像生成需要额外的计算开销（LoRA微调+图像生成），在资源受限场景下可能不实用
- 辅助分类器的有效性依赖于扩散模型对目标域的覆盖能力，在极端域外奇异类别上可能失效
- $\lambda$ 和 $\beta$ 等超参数需要在验证集上调优，而无监督设置下验证集的构建本身也是挑战
- 仅在CLIP的视觉编码器上验证，是否适用于其他VLM（如BLIP-2、SigLIP）尚待验证
- 未探索使用更先进的扩散模型（如SDXL、Flux）是否能进一步提升合成图像质量和最终分类性能

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Vision-Language Model IP Protection via Prompt-based Learning](vision-language_model_ip_protection_via_prompt-based_learning.md)
- [\[CVPR 2025\] VladVA: Discriminative Fine-tuning of LVLMs](vladva_discriminative_fine-tuning_of_lvlms.md)
- [\[CVPR 2025\] NLPrompt: Noise-Label Prompt Learning for Vision-Language Models](nlprompt_noise-label_prompt_learning_for_vision-language_models.md)
- [\[CVPR 2025\] Visual and Semantic Prompt Collaboration for Generalized Zero-Shot Learning](visual_and_semantic_prompt_collaboration_for_generalized_zero-shot_learning.md)
- [\[CVPR 2025\] Words or Vision: Do Vision-Language Models Have Blind Faith in Text?](words_or_vision_do_vision-language_models_have_blind_faith_in_text.md)

</div>

<!-- RELATED:END -->
