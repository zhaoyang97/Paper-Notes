---
title: >-
  [论文解读] ClearAIR: A Human-Visual-Perception-Inspired All-in-One Image Restoration
description: >-
  [AAAI 2026][多模态VLM][图像恢复] 受人类视觉感知（HVP）启发，提出一种从粗到细的统一图像复原框架 ClearAIR，通过 MLLM 质量评估 → 语义区域感知 → 退化类型识别 → 内部线索复用四阶段逐步恢复图像质量，在多种退化任务上取得 SOTA。 任务特定模型泛化性差：早期图像复原方法针对去噪、去雾、…
tags:
  - "AAAI 2026"
  - "多模态VLM"
  - "图像恢复"
  - "Human Visual Perception"
  - "MLLM-based IQA"
  - "Semantic Guidance"
  - "自监督学习"
---

# ClearAIR: A Human-Visual-Perception-Inspired All-in-One Image Restoration

**会议**: AAAI 2026  
**arXiv**: [2601.02763](https://arxiv.org/abs/2601.02763)  
**代码**: 未开源  
**领域**: 图像复原  
**关键词**: All-in-One Image Restoration, Human Visual Perception, MLLM-based IQA, Semantic Guidance, Self-supervised Learning  

## 一句话总结

受人类视觉感知（HVP）启发，提出一种从粗到细的统一图像复原框架 ClearAIR，通过 MLLM 质量评估 → 语义区域感知 → 退化类型识别 → 内部线索复用四阶段逐步恢复图像质量，在多种退化任务上取得 SOTA。

## 研究背景与动机

**任务特定模型泛化性差**：早期图像复原方法针对去噪、去雾、去雨等单一退化设计专用网络，无法跨任务泛化，部署成本高。

**通用模型仍需多实例**：NAFNet、Restormer 等通用复原模型虽可处理多种退化，但每种退化仍需独立模型，推理流程复杂。

**现有 AiOIR 忽略空间非均匀退化**：AirNet、PromptIR 等 All-in-One 方法对整幅图像施加统一处理策略，未考虑不同区域退化分布和严重程度的差异。

**纹理复杂度影响复原难度**：即使退化均匀分布，平坦区域与复杂纹理区域的复原难度也显著不同，统一策略导致过平滑或伪影。

**缺乏层次化感知机制**：人类视觉先整体后局部，而现有方法缺乏从全局结构到局部细节的渐进感知流程。

**细节恢复能力不足**：现有方法在精细纹理恢复上仍有欠缺，缺乏对图像内在结构信息的挖掘。

## 方法详解

### 整体框架

ClearAIR 模仿人类视觉感知的层次化处理流程，包含四个核心组件：

1. **MLLM-based IQA**（全局质量评估）→ 2. **SGU**（区域语义感知）→ 3. **Task Identifier**（退化类型识别）→ 4. **ICRM**（内部线索复用）

复原骨干采用 Restormer，四个层级的 Prompt Transformer Block 数分别为 [3, 5, 6, 8]，通道维度为 [48, 96, 192, 384]。

### 关键设计一：MLLM-based Overall Assessment

- 利用 DeQA（基于 MLLM 的图像质量评估模型）作为全局质量感知器
- 视觉编码器将退化图像编码为 visual tokens，经 vision abstractor 压缩后与文本 tokens 融合送入 MLLM
- 提取 "quality level" token 前一层的隐状态 $\mathcal{Q}$ 作为质量表征
- 通过 Quality Guidance Module (QGM) 以 affine transformation 形式注入复原骨干：$\mathbf{X}_{out} = \mathbf{X}_{in} \odot \text{Linear}(\mathbf{F}_q) + \text{Linear}(\mathbf{F}_q)$

### 关键设计二：Semantic Guidance Unit (SGU) + Task Identifier

**区域感知 (SGU)**：
- 利用预训练 SAM2 对退化图像生成 $N_m$ 个二值语义 mask
- 通过 Mask Average Pooling (MAP) 计算每个 mask 区域内特征的均值并广播回对应位置，得到语义先验 $\mathbf{F}_{sem}$
- 训练时引入 mask dropout 策略，随机移除部分 mask 并合并至背景，增强鲁棒性
- 语义特征通过 Semantic Cross-Attention (SCA) 与骨干交互

**退化识别 (Task Identifier)**：
- 使用 DA-CLIP 生成内容嵌入 $\mathbf{F}_c \in \mathbb{R}^{1 \times 512}$ 和退化嵌入 $\mathbf{F}_d \in \mathbb{R}^{1 \times 512}$
- 退化嵌入经 MLP + Softmax 与可学习 prompt 集 $\mathcal{P}$ 加权得到 degradation prompt $\mathbf{F}_p$
- Degradation-Aware Module (DAM) 利用 $\mathbf{F}_c$ 做 cross-attention 进行内容感知增强，同时由 $\mathbf{F}_p$ 生成退化 mask 对特征做空间调制

### 关键设计三：Internal Clue Reuse Mechanism (ICRM)

- 对复原结果 $\mathbf{I}_r$ 依次施加弱增强和强增强
- 计算弱增强与强增强输出之间的 L2 距离作为内部一致性损失：$\mathcal{L}_{inter} = \gamma \cdot \|\mathbf{I}_r^w - \mathbf{I}_r^s\|_2^2$
- 以自监督方式挖掘图像内在结构信息，增强细节恢复能力
- 该机制不需要额外标注，利用图像自身的内部统计信息

### 损失函数与训练

- 总损失：$\mathcal{L}_{total} = \mathcal{L}_1 + \alpha \cdot \mathcal{L}_{inter}$，其中 $\alpha = 0.25$，$\gamma = 0.05$
- 优化器：AdamW ($\beta_1=0.9$, $\beta_2=0.999$)，学习率 $2 \times 10^{-4}$，batch size 4
- 训练 300K 迭代，输入随机裁剪到 256×256，随机水平/垂直翻转
- 硬件：NVIDIA RTX 4090

## 实验关键数据

### Three Degradations（去噪 + 去雾 + 去雨）

| 方法 | 参数量 | SOTS (PSNR/SSIM) | Rain100L | BSD68 σ=15 | BSD68 σ=25 | BSD68 σ=50 | 平均 |
|---|---|---|---|---|---|---|---|
| PromptIR | 36M | 30.58/.974 | 36.37/.972 | 33.98/.933 | 31.31/.888 | 28.06/.799 | 32.06/.913 |
| AdaIR | 29M | 31.06/.980 | 38.64/.983 | 34.12/.934 | 31.45/.892 | 28.19/.802 | 32.69/.918 |
| VLU-Net | 35M | 30.71/.980 | 38.93/.984 | 34.13/.935 | 31.48/.892 | 28.23/.804 | 32.70/.919 |
| **ClearAIR** | **31M** | **31.08/.981** | 38.61/.984 | **34.18/.935** | 31.50/.891 | **28.31/.804** | **32.74/.919** |

### Five Degradations（+去模糊 + 低光增强）

| 方法 | SOTS | Rain100L | BSD68 σ=25 | GoPro | LOLv1 | 平均 |
|---|---|---|---|---|---|---|
| Perceive-IR | 28.19/.964 | 37.25/.977 | 31.44/.887 | 29.46/.886 | 22.81/.833 | 29.84/.909 |
| AdaIR | 30.53/.978 | 38.02/.981 | 31.35/.888 | 28.12/.858 | 23.00/.845 | 30.20/.910 |
| **ClearAIR** | 30.12/.978 | 38.20/.982 | **31.53/.888** | **29.67/.887** | 22.83/.846 | **30.45/.916** |

### All-Weather（雪 + 雨雾 + 雨滴）

| 方法 | Snow100K-S | Snow100K-L | Outdoor-Rain | RainDrop | 平均 |
|---|---|---|---|---|---|
| Histoformer | 37.41/.966 | 32.16/.926 | 32.08/.939 | 33.06/.944 | 33.68/.945 |
| **ClearAIR** | **37.79/.967** | **32.53/.932** | **32.45/.941** | 32.82/.942 | **33.90/.946** |

### Composited Degradations (CDD-11)

ClearAIR 达到 29.34 dB / 0.886 SSIM，比 OneRestore (28.72 dB) 提升 0.62 dB。

### 消融实验

- 感知顺序："How-Where-What"（本文）最优 38.21 dB，优于 "What-How-Where" (38.04) 和 "Where-What-How" (37.89)
- 各组件贡献：去除 IQA/SGU/TI/ICRM 中任意一个均导致性能下降，四者协同达到最佳 38.21/0.986

## 亮点

1. **HVP 启发的渐进式设计理念新颖**：将人类"先整体后局部"的视觉感知规律引入 AiOIR，四阶段流程逻辑清晰
2. **多模态大模型驱动质量评估**：首次将 MLLM-based IQA 引入图像复原作为全局先验，跨模态理解增强退化表征
3. **区域级自适应处理**：SGU + Task Identifier 实现空间非均匀退化的差异化处理，解决统一策略的根本缺陷
4. **ICRM 自监督细节恢复无需额外标注**：巧妙利用图像内在统计信息，以增强一致性约束提升纹理恢复
5. **四个 AiOIR 设定下全面 SOTA**：31M 参数量适中，性能全面超越 AdaIR、VLU-Net 等最新方法

## 局限性

1. **推理效率存疑**：引入 MLLM-IQA (DeQA)、SAM2、DA-CLIP 三个大型预训练模型，推理开销和延迟可能较高，论文未报告推理速度
2. **预训练模型依赖重**：框架强依赖 DeQA、SAM2、DA-CLIP 的质量，这些模型在严重退化下的表现未充分讨论
3. **ICRM 增益有限**：消融实验中去除 ICRM 仅损失 0.18 dB (38.03→38.21)，设计复杂度与收益不完全匹配
4. **低光增强任务表现一般**：五退化设定中 LOLv1 上 22.83 dB 不及 AdaIR 的 23.00 dB，说明全局质量引导对低光场景帮助有限
5. **缺少真实世界大规模评测**：主要在合成数据集上验证，真实退化场景的泛化能力有待进一步验证

## 相关工作

- **AirNet** (CVPR'22)：对比学习编码退化表征，ClearAIR 在其基础上增加了区域级感知
- **PromptIR** (NeurIPS'23)：视觉 prompt 引导多退化处理，ClearAIR 的退化 prompt 由 DA-CLIP 显式生成而非单纯可学习
- **Perceive-IR** (TIP'25)：联合感知退化类型和严重程度，ClearAIR 进一步引入 MLLM 全局评估和语义区域分割
- **AdaIR** (ICLR'25)：自适应图像复原，ClearAIR 在五退化平均 PSNR 上超出 0.25 dB
- **VLU-Net** (CVPR'25)：视觉语言统一复原，ClearAIR 在三退化上超出 0.04 dB 且参数更少
- **OneRestore** (ECCV'24)：复合退化复原，ClearAIR 在 CDD-11 上超出 0.62 dB

## 评分

- 新颖性: ⭐⭐⭐⭐ — HVP 启发的四阶段框架设计思路新颖，MLLM-IQA 引入复原任务有创新
- 实验充分度: ⭐⭐⭐⭐ — 四种 AiOIR 设定 + 消融实验 + 定性比较，覆盖面广
- 写作质量: ⭐⭐⭐⭐ — 层次清晰，HVP 类比贯穿全文，公式表达规范
- 价值: ⭐⭐⭐⭐ — 在 AiOIR 领域推动了区域级自适应感知的研究方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Oasis: One Image is All You Need for Multimodal Instruction Data Synthesis](../../ICCV2025/multimodal_vlm/oasis_one_image_is_all_you_need_for_multimodal_instruction_data_synthesis.md)
- [\[CVPR 2026\] UARE: A Unified Vision-Language Model for Image Quality Assessment, Restoration, and Enhancement](../../CVPR2026/multimodal_vlm/uare_a_unified_vision-language_model_for_image_quality_assessment_restoration_an.md)
- [\[CVPR 2026\] One Patch to Caption Them All: A Unified Zero-Shot Captioning Framework](../../CVPR2026/multimodal_vlm/one_patch_to_caption_them_all_a_unified_zero-shot_captioning_framework.md)
- [\[AAAI 2026\] VipAct: Visual-Perception Enhancement via Specialized VLM Agent Collaboration and Tool-use](vipact_visual-perception_enhancement_via_specialized_vlm_age.md)
- [\[AAAI 2026\] SatireDecoder: Visual Cascaded Decoupling for Enhancing Satirical Image Comprehension](satiredecoder_visual_cascaded_decoupling_for_enhancing_satirical_image_comprehen.md)

</div>

<!-- RELATED:END -->
