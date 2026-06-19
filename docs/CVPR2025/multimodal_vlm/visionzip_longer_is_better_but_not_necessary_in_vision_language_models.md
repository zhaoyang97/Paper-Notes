---
title: >-
  [论文解读] VisionZip: Longer is Better but Not Necessary in Vision Language Models
description: >-
  [CVPR 2025][多模态VLM][视觉Token压缩] VisionZip 发现视觉编码器（CLIP/SigLIP）生成的视觉Token存在严重冗余——仅少数Token聚集了绝大部分注意力和信息，基于此提出一种文本无关的Token选择与合并方法，在仅保留10%Token的情况下保持95%的模型性能，并实现8倍预填充加速。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "视觉Token压缩"
  - "注意力冗余"
  - "Token选择与合并"
  - "高效推理"
  - "多轮对话"
---

# VisionZip: Longer is Better but Not Necessary in Vision Language Models

**会议**: CVPR 2025  
**arXiv**: [2412.04467](https://arxiv.org/abs/2412.04467)  
**代码**: [https://github.com/dvlab-research/VisionZip](https://github.com/dvlab-research/VisionZip)  
**领域**: 多模态VLM  
**关键词**: 视觉Token压缩, 注意力冗余, Token选择与合并, 高效推理, 多轮对话

## 一句话总结

VisionZip 发现视觉编码器（CLIP/SigLIP）生成的视觉Token存在严重冗余——仅少数Token聚集了绝大部分注意力和信息，基于此提出一种文本无关的Token选择与合并方法，在仅保留10%Token的情况下保持95%的模型性能，并实现8倍预填充加速。

## 研究背景与动机

VLM性能提升很大程度上依赖增加视觉Token长度：LLaVA-1.5用576个视觉Token，LLaVA-NeXT的672×672图像产生2880个Token，而文本Token通常只有几十到上百个。视觉Token远多于文本Token，造成巨大的计算和内存开销，严重限制了边缘计算、自动驾驶等实际应用场景。核心矛盾是：图像信息天然比文本稀疏得多，但VLM的视觉Token数量却远超文本Token。

现有方法（如FastV、SparseVLM）依赖LLM层内的文本-视觉注意力来逐步剪枝视觉Token，但这些方法存在根本缺陷——视觉编码器已经将信息聚集到少量"代理Token"上，而这些代理Token的位置往往与图像主体不一致（可能在背景或边缘），导致基于文本相关性的选择方法选到的Token实际信息量不足。VisionZip的切入点是：直接在视觉编码器输出端、以文本无关的方式选择最具信息量的Token。

## 方法详解

### 整体框架

VisionZip 在视觉编码器和LLM之间插入一个轻量的Token压缩模块：首先根据视觉编码器内部的注意力分数选择 $K$ 个"主导Token"(Dominant Tokens)，然后将剩余Token通过相似度匹配合并为 $M$ 个"上下文Token"(Contextual Tokens)，最终仅将 $K+M$ 个压缩后的Token送入LLM。整个过程完全不需要训练（training-free），也可选择用少量数据微调投影器进一步提升效果。

### 关键设计

1. **主导Token选择 (Dominant Token Selection)**:
    - 功能：从视觉编码器的输出中筛选出聚集了最多信息的核心Token
    - 核心思路：利用视觉编码器倒数第2层的注意力分数来评估每个Token的重要性。对于有CLS Token的模型(如CLIP)，选择被CLS Token最关注的Top-K个Token；对于无CLS Token的模型(如SigLIP)，计算每个Token从所有其他Token接收的平均注意力，选择最高的K个
    - 设计动机：通过可视化分析发现，视觉编码器中层之后注意力急剧收敛到少数Token，到倒数第2层（VLM通常选取的层）时，注意力和信息高度集中于极少数"主导Token"。这些Token自然包含了图像的绝大部分信息

2. **上下文Token合并 (Contextual Token Merging)**:
    - 功能：从被丢弃的Token中挽回可能遗漏的小型但重要的细节信息
    - 核心思路：将非主导Token均匀分为"目标Token"和"待合并Token"两组，利用Key值计算相似度矩阵，将每个待合并Token分配给最相似的目标Token，通过平均合并生成上下文Token
    - 设计动机：仅保留主导Token可能遗漏图像中小但重要的细节（如小物体、文字），合并策略能以极低成本保留语义相似性信息，弥补信息损失

3. **高效投影器微调 (Efficient Tuning)**:
    - 功能：解决Token数量大幅减少后视觉-语言空间轻微失配的问题
    - 核心思路：仅用LLaVA-1.5训练数据的1/10微调多模态投影器(projector)，其他组件冻结。8块A800共30分钟完成，甚至可在3090上运行
    - 设计动机：VLM原本在全量视觉Token上训练，突然减少到1/10会导致视觉输入空间和LLM空间的轻微失配，微调投影器即可修复这种对齐偏移

### 损失函数 / 训练策略

VisionZip本身是training-free的，不涉及新的损失函数。可选的Efficient Tuning阶段使用标准的指令微调损失，仅更新投影器参数。

## 实验关键数据

### 主实验（LLaVA-1.5, 576 tokens基线）

| Token数 | 方法 | 11个Benchmark平均保持率 | vs FastV | vs SparseVLM |
|---------|------|----------------------|----------|------------|
| 192 (↓66.7%) | VisionZip | 98.5% | +10.3% | +2.1% |
| 192 (↓66.7%) | VisionZip‡ | **99.1%** | +10.9% | +2.7% |
| 128 (↓77.8%) | VisionZip | 97.6% | +14.1% | +4.2% |
| 64 (↓88.9%) | VisionZip | 94.0% | +18.4% | +8.2% |
| 64 (↓88.9%) | VisionZip‡ | **95.2%** | +19.6% | +9.4% |

### 主实验（LLaVA-NeXT, 2880 tokens基线）

| Token数 | 方法 | 7个Benchmark平均保持率 | 说明 |
|---------|------|---------------------|------|
| 640 (↓77.8%) | VisionZip | 97.6% | 不需额外训练 |
| 640 (↓77.8%) | VisionZip‡ | **98.9%** | 仅30分钟微调 |
| 320 (↓88.9%) | VisionZip‡ | **97.9%** | Token减少近90%仍保持高精度 |
| 160 (↓94.4%) | VisionZip‡ | **95.5%** | 仅用5%的Token |

### 效率分析

| 方法 | Token数 | 总推理时间 | 加速比 | 预填充延迟 | 预填充加速 |
|------|---------|----------|--------|----------|----------|
| 基线 | 2880 | 2293s | 1.0× | 218ms | 1.0× |
| FastV | 160 | 1792s | 1.3× | 119ms | 1.8× |
| SparseVLM | 160 | 1895s | 1.2× | 128ms | 1.7× |
| VisionZip | 160 | **756s** | **3.0×** | **27.8ms** | **7.8×** |

### 消融实验

| 配置 | TextVQA (64 tokens) | 说明 |
|------|-------------------|------|
| SparseVLM基线 | 51.1 | 从576个Token中用LLM注意力选64 |
| 先去掉Top50视觉注意力Token | 46.4 (−9.2%) | 证明SparseVLM依赖的Token其实信息丰富 |
| 仅给SparseVLM Top128 VisionZip Token | 52.5 (+2.7%) | 预筛选后反而更好 |

### 关键发现

- 在MMMU和MMVet等Benchmark上，减少Token数反而提升精度，说明冗余视觉Token可能充当噪声
- 视觉编码器的注意力集中现象源于Softmax函数的梯度特性：$\frac{\partial \text{softmax}(z_i)}{\partial z_i} = \text{softmax}(z_i) \cdot (1-\text{softmax}(z_i))$，高注意力区域梯度大导致"马太效应"
- 文本相关方法（FastV/SparseVLM）选到的Token与视觉编码器聚集信息的Token位置不匹配——后者倾向于出现在背景区域而非图像主体上

## 亮点与洞察

- **深入追问冗余根因**：不止提方法，而是从Softmax梯度特性和"Attention Sink"现象深入解释了为什么视觉编码器会产生Token冗余，这为视觉编码器设计提供了方向性启发
- **文本无关的优势**：与依赖LLM内部注意力的FastV/SparseVLM不同，VisionZip在视觉编码器端完成压缩，天然支持多轮对话场景——KV Cache中存储的Token不会因为对话主题切换而失效
- **13B反超7B**：VisionZip使LLaVA-NeXT 13B推理比原生7B更快且精度更高，是一个极有实际价值的发现
- **极简设计哲学**：不需要任何额外模块，仅用注意力选择+相似度合并就大幅超越需要LLM前向传播的方法

## 局限与展望

- 主导Token的选择仅依赖视觉编码器的最后几层注意力，可能错过在浅层才有信息量的细节
- 上下文Token合并策略(均匀分割+平均合并)较为粗糙，可探索更精细的合并方案
- 论文主要在LLaVA系列和Mini-Gemini上验证，对Qwen2-VL、InternVL等主流模型的适配验证不足
- Training-free模式下在极低Token数（如64）时精度仍有约5%损失，对于严格的应用场景可能需要更多微调

## 相关工作与启发

- **vs FastV**: FastV在LLM浅层用文本注意力逐层剪枝Token，但无法避免LLM浅层的完整计算开销，且选择的Token可能信息不足。VisionZip在进入LLM前就完成压缩，效率优势巨大（3.0× vs 1.3×）
- **vs SparseVLM**: SparseVLM同样依赖LLM层内的文本-视觉注意力，在视频任务（MSRVTT）上失败严重（仅54.7%保持率），而VisionZip达91.9%
- **vs NVILA (2412.04468)**: NVILA用STC空间压缩是在训练时就融入的设计，VisionZip则是可即插即用的推理加速方案，两者正交互补

## 评分

- 新颖性: ⭐⭐⭐⭐ 核心观察（视觉Token冗余）并非全新，但从Softmax梯度角度的分析和文本无关选择策略有独到见解
- 实验充分度: ⭐⭐⭐⭐⭐ 11个图像benchmark、4个视频benchmark、3种VLM架构、详细的效率分析和消融实验
- 写作质量: ⭐⭐⭐⭐ 结构清晰，可视化丰富（注意力分布、特征错位），但部分表格密度过高
- 价值: ⭐⭐⭐⭐ 作为即插即用的推理加速方案实用价值高，但需要更多主流模型的验证

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Vision-Language Models Do Not Understand Negation](vision-language_models_do_not_understand_negation.md)
- [\[ECCV 2024\] BLINK: Multimodal Large Language Models Can See but Not Perceive](../../ECCV2024/multimodal_vlm/blink_multimodal_large_language_models_can_see_but_not_perceive.md)
- [\[CVPR 2025\] VLsI: Verbalized Layers-to-Interactions from Large to Small Vision Language Models](vlsi_verbalized_layers-to-interactions_from_large_to_small_vision_language_model.md)
- [\[NeurIPS 2025\] Better Tokens for Better 3D: Advancing Vision-Language Modeling in 3D Medical Imaging](../../NeurIPS2025/multimodal_vlm/better_tokens_for_better_3d_advancing_vision-language_modeling_in_3d_medical_ima.md)
- [\[CVPR 2025\] NVILA: Efficient Frontier Visual Language Models](nvila_efficient_frontier_visual_language_models.md)

</div>

<!-- RELATED:END -->
