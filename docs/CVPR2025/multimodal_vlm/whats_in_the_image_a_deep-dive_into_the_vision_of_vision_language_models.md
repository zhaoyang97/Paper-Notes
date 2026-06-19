---
title: >-
  [论文解读] What's in the Image? A Deep-Dive into the Vision of Vision Language Models
description: >-
  [CVPR 2025][多模态VLM][VLM可解释性] 本文通过 Attention Knockout 实验系统分析了 VLM（InternVL2-76B 和 LLaVA-1.5-7B）的视觉信息处理机制，揭示了三个关键发现：(1) query text token 充当全局图像描述器压缩高层视觉信息，(2) 中间层（约 25%）主导跨模态信息传递而早晚层贡献极少，(3) 细粒度物体细节通过空间局部化的方式从 image token 中提取。基于这些发现提出了 Image Re-prompting 应用，用仅 5% 的 image token 即可保持 96% 的 VQA 性能。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "VLM可解释性"
  - "注意力机制"
  - "视觉信息压缩"
  - "中间层分析"
  - "提示学习"
---

# What's in the Image? A Deep-Dive into the Vision of Vision Language Models

**会议**: CVPR 2025  
**arXiv**: [2411.17491](https://arxiv.org/abs/2411.17491)  
**代码**: [https://vision-of-vlm.github.io/](https://vision-of-vlm.github.io/) (有项目页面)  
**领域**: 多模态VLM  
**关键词**: VLM可解释性, 注意力机制, 视觉信息压缩, 中间层分析, Image Re-prompting

## 一句话总结
本文通过 Attention Knockout 实验系统分析了 VLM（InternVL2-76B 和 LLaVA-1.5-7B）的视觉信息处理机制，揭示了三个关键发现：(1) query text token 充当全局图像描述器压缩高层视觉信息，(2) 中间层（约 25%）主导跨模态信息传递而早晚层贡献极少，(3) 细粒度物体细节通过空间局部化的方式从 image token 中提取。基于这些发现提出了 Image Re-prompting 应用，用仅 5% 的 image token 即可保持 96% 的 VQA 性能。

## 研究背景与动机
VLM 在图像描述、VQA 等任务上展现了惊人能力，但其内部视觉信息处理机制仍然是黑箱——我们不知道模型如何从图像 token 中提取信息、跨模态信息如何流动、不同层各司什么职。理解这些机制对于提升模型透明性、提高推理效率、指导未来 VLM 设计至关重要。现有可解释性工作要么聚焦 LLM（不涉及视觉）、要么聚焦小模型（<10B），很少有工作在 76B 参数规模的 SOTA VLM 上做深入分析。**核心问题**：VLM 在生成文本时，视觉信息是如何被编码、传递和利用的？

## 方法详解

### 整体框架
分析框架基于 Attention Knockout：在 VLM 推理时，通过修改注意力掩码来阻断特定 token 类型之间的信息流动，观察输出的变化。结合自定义的 LLM-as-a-judge 评估协议来量化这些变化。分析聚焦三种 token 类型：image token $\mathbf{T}_{img}$、query text token $\mathbf{T}_{txt}$（如"describe the image"）、generated token $\mathbf{T}_{gen}$。

### 关键设计
1. **Attention Knockout 实验框架**:
    - 功能：通过阻断信息流来揭示各 token 类型和各层的作用
    - 核心思路：定义 knockout 掩码 $\mathbf{M}_{ko}[p,q; P_{src}, P_{tgt}]$，当 $q \in P_{src}$ 且 $p \in P_{tgt}$ 时设为 $-\infty$。三种核心配置：(a) $\text{KO}_{img \to gen}$——阻断 image token 对 generated token 的直接影响，视觉信息只能通过 query token 间接传递；(b) $\text{KO}_{img \to txt}$——阻断 image token 对 query token 的影响；(c) $\text{KO}_{img \to txt+gen}$——完全阻断 image token 对其他 token 的影响。通过从不同层 $l$ 开始施加 knockout，可以分析各层的贡献
    - 设计动机：由于 VLM 使用因果注意力掩码，image token 的表征在第一步解码后即固定不变，因此阻断不同方向的信息流可以干净地隔离各信息通路的贡献

2. **LLM-as-a-Judge 评估协议**:
    - 功能：自动量化 knockout 对输出质量的影响
    - 核心思路：让 LLM 分别从原始和修改后的 VLM 描述中提取物体列表 $O_{orig}$ 和 $O_{ko}$，计算 TP（两者都提到的物体）、FN（原始有但修改后缺失的）、FP（修改后新出现的幻觉物体），得到 Precision/Recall/F1 分数。使用 chain-of-thought 和 3 个 in-context example 提升可靠性
    - 设计动机：比较两段自由文本的语义差异非常困难（措辞、风格差异大），COCO 等数据集的标注也不完整，因此利用 LLM 的语言理解能力做自动评估。用户研究表明 LLM 判断与人类的一致率高达 95%

3. **空间局部化分析**:
    - 功能：揭示 VLM 如何从 image token 中提取特定物体的细粒度信息
    - 核心思路：找出在 $\text{KO}_{img \to gen}$ 下丢失的物体（即需要直接从 image token 获取的细节），可视化 generated token 对 image token 的注意力热力图。用 SAM 生成物体的伪真实分割掩码，检测注意力峰值是否落在物体区域内（40 像素容差 ≈ 1 token 距离）。定义 Localization Accuracy 指标
    - 设计动机：$\text{KO}_{img \to gen}$ 实验表明高层信息压缩在 query token 中，但细节丢失。那么细节是通过什么机制获取的？答案是空间局部化——中间层的注意力精确指向目标物体

### 损失函数 / 训练策略
本文为分析工作，无训练。Image Re-prompting 应用部分：提取 "describe the image" 时中间层注意力值最高的 top-K% image token 作为 compressed context，后续问题直接使用此压缩上下文而非完整图像。

## 实验关键数据

### 主要发现的量化指标

| Knockout 配置 | F1 Score | 含义 |
|--------------|---------|------|
| $\text{KO}_{img \to gen}$（全层） | 0.40 | 仅靠 query token 中的压缩信息可识别部分物体 |
| $\text{KO}_{img \to txt}$（全层） | 0.00 | 阻断 query token 获取视觉信息后完全失效！ |
| $\text{KO}_{img \to gen}^{l \notin [20,40]}$（仅中间层访问） | 0.75-0.81 | 中间层 20-40 几乎保持完整性能 |
| KO after layer 40 | ≈0.80 | 40 层之后的注意力几乎无作用 |

### Image Re-prompting（MME benchmark）

| 方法 | ACC | ACC+ | Token 数 |
|------|-----|------|---------|
| Naive (InternVL2 完整) | 84.83 | 70.60 | 1695 |
| Describe-to-LLM | 73.21 | 56.14 | 172 |
| Query + K=5% | **81.46** | **64.52** | 201 |
| Query + K=2% | 77.16 | 55.94 | 151 |
| 仅 Query（无 image token） | 61.03 | 28.45 | 60 |

### 空间局部化精度

| 层范围 | Localization Accuracy |
|--------|---------------------|
| 层 0-10 | ~30% |
| 层 10-20 | ~45% |
| 层 20-30 | ~65% |
| 层 30-40 | **~73%** |
| 层 40-50 | ~60% |

### 关键发现
- **Query token 是全局图像描述器**：尽管 query token 不到总 token 的 5%，却在大多数层占据 60% 以上的注意力。完全依赖 query token（$\text{KO}_{img \to gen}$）仍能产生描述性响应
- **中间层主导视觉信息传递**：仅允许 20-40 层访问 image token（总共 80 层的 25%），F1 可达 0.75-0.81，与完整模型几乎无差异
- **细粒度信息通过空间局部化检索**：中间层的注意力峰值精确对应到目标物体在图像中的位置，定位精度在中间层达到 73%
- 仅 5% 的高注意力 image token 即可保持近乎完整的性能（F1 快速饱和）

## 亮点与洞察
- **双通路视觉处理机制**的发现极具洞察力：(1) 全局信息通过中间层压缩到 query token；(2) 细粒度信息通过中间层的空间局部化注意力从 image token 提取
- **LLM-as-a-judge 评估协议**是一个有用的通用工具，解决了自由文本比较的难题，与人类评估一致率 95%
- Image Re-prompting 作为分析的应用衍生品很有实用价值——可以显著减少多轮问答中的 token 消耗（12x 压缩）
- 是首个在 76B 参数规模 VLM 上进行可解释性分析的工作
- Query token 仅占总 token 的不到 5%，却承载了超过 60% 的注意力，这一发现暗示了 VLM 的内在压缩倾向
- 在 $\text{KO}_{img \to txt}$ 实验中 F1 为 0，证明 query token 对 VLM 来说不仅是指令载体，更是不可缺少的视觉信息中转站
- 注意力的长尾分布意味着仅少量 image token 获得显著注意力，为 token pruning 提供了有力的理论支持

## 局限与展望
- 分析仅覆盖 "describe the image" 这一单一 query，不同 query 类型（如推理、计数）的信息流模式可能不同
- 仅分析了注意力模块，忽略了 FFN 在信息存储/转换中的作用
- Image Re-prompting 需要先做一次完整的 "describe" 前向传播来提取 compressed context，不算真正的"免费"加速
- 定位精度分析使用 SAM 生成伪真实标注，本身存在噪声
- 中间层的定义（20-40）是针对 InternVL2-76B 的，不同架构/规模可能不同

## 相关工作与启发
- **vs Logit Lens 方法**: Logit Lens 揭示每层编码了什么信息，本文揭示信息如何在模态间流动——互补视角
- **vs LLM 可解释性工作（如 Geva et al.）**: 它们分析单模态 LLM 的 FFN/注意力，本文首次系统分析了 VLM 的跨模态信息流
- **vs token pruning 方法**: 现有 token 压缩通常基于重要性评分，本文从可解释性角度给出了理论支撑——大部分 image token 确实是冗余的
- **vs Basu et al. (2024)**: 研究信息存储位置，而本文更关注信息如何跨模态流动

## 补充说明
- 实验在 COCO 数据集的 80 张随机图像上进行注意力分析，231 个物体用于定位精度评估
- InternVL2-76B 共 80 层 Transformer，中间层定义为 20-40 层
- LLaVA-1.5-7B 上的结果在附录中呈现，趋势一致

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次在大规模 VLM 上系统揭示视觉信息的双通路处理机制，发现非常 surprising
- 实验充分度: ⭐⭐⭐⭐ 多种 knockout 配置、LLM 评估 + 人类验证、空间定位分析，但数据规模较小（80张图）
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑链条清晰，实验逐步深入，可视化优秀
- 价值: ⭐⭐⭐⭐⭐ 对理解 VLM 内部机制和指导高效 VLM 设计有重要启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Identifying and Mitigating Position Bias of Multi-image Vision-Language Models](identifying_and_mitigating_position_bias_of_multi-image_vision-language_models.md)
- [\[CVPR 2025\] Cropper: Vision-Language Model for Image Cropping through In-Context Learning](cropper_vision-language_model_for_image_cropping_through_in-context_learning.md)
- [\[CVPR 2025\] MMRL: Multi-Modal Representation Learning for Vision-Language Models](mmrl_multi-modal_representation_learning_for_vision-language_models.md)
- [\[CVPR 2025\] DPC: Dual-Prompt Collaboration for Tuning Vision-Language Models](dpc_dual-prompt_collaboration_for_tuning_vision-language_models.md)
- [\[CVPR 2025\] NLPrompt: Noise-Label Prompt Learning for Vision-Language Models](nlprompt_noise-label_prompt_learning_for_vision-language_models.md)

</div>

<!-- RELATED:END -->
