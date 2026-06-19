---
title: >-
  [论文解读] ReVisionLLM: Recursive Vision-Language Model for Temporal Grounding in Hour-Long Videos
description: >-
  [CVPR 2025][多模态VLM][长视频时序定位] 提出 ReVisionLLM，首个能在小时级长视频中进行时序定位的视觉语言模型，模仿人类搜索策略递归处理视频——先粗粒度锁定相关片段，再逐级细化至精确时间边界，在 MAD 数据集上超越 SOTA +2.6% R1@0.1。 长视频时序定位（temporal groun…
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "长视频时序定位"
  - "递归视觉语言模型"
  - "层次化适配器"
  - "渐进训练"
  - "视频理解"
---

# ReVisionLLM: Recursive Vision-Language Model for Temporal Grounding in Hour-Long Videos

**会议**: CVPR 2025  
**arXiv**: [2411.14901](https://arxiv.org/abs/2411.14901)  
**代码**: [https://github.com/Tanveer81/ReVisionLLM](https://github.com/Tanveer81/ReVisionLLM)  
**领域**: 多模态VLM  
**关键词**: 长视频时序定位, 递归视觉语言模型, 层次化适配器, 渐进训练, 视频理解

## 一句话总结

提出 ReVisionLLM，首个能在小时级长视频中进行时序定位的视觉语言模型，模仿人类搜索策略递归处理视频——先粗粒度锁定相关片段，再逐级细化至精确时间边界，在 MAD 数据集上超越 SOTA +2.6% R1@0.1。

## 研究背景与动机

长视频时序定位（temporal grounding）是指根据文本查询在长视频中定位事件的起止时间。现有方法面临三大挑战：

1. **VLM 帧数受限**：现有 VLM（如 VTimeLLM）通过均匀采样固定帧数处理视频。对小时级视频（如 2 小时电影），100 帧均匀采样意味着每帧间隔 72 秒，导致时序细节完全丢失——实验中 VTimeLLM 在 MAD 上所有指标为 **0**
2. **训练资源瓶颈**：直接在小时级视频上训练 VLM 需要巨大的显存和算力
3. **置信度校准问题**：VLM 的视觉预测经常过度自信（高置信度误检），在长视频中这一问题被放大——需要从大量无关片段中区分出真正的目标事件
4. **非 LLM 方法的局限**：CONE、SOONet、RGNet 等非 LLM 方法使用多网络+复杂后处理，灵活性差，无法处理自由文本查询

## 方法详解

### 整体框架

ReVisionLLM 包含三个核心组件：(1) 多模态编码器（CLIP ViT-L/14 提取帧级 CLS 特征）；(2) 层次化适配器（Hierarchical Adapter，生成密集/稀疏两种时间特征）；(3) LLM（Vicuna-7B，作为时序定位解码器递归预测事件边界）。模型递归处理视频：顶层用稀疏特征扫描全视频锁定候选片段，逐层下沉到底层用密集特征精确定位事件边界。

### 关键设计

1. **层次化适配器（Hierarchical Adapter）**:
    - 功能：将视频帧特征转化为多粒度的时间表示——底层密集特征用于精确定位，上层稀疏特征用于高效扫描
    - 核心思路：先将帧特征 $\mathcal{F}$ 切分为滑动窗口片段 $C = [C^i]_{i=1,...,|C|}$，每个片段 $C^i \in \mathbb{R}^{L_w \times D}$。**密集特征** $\mathcal{D}^i = h_d(C^i)$ 通过线性投影保留原始时间分辨率。**稀疏特征**的生成分两步：(a) 交叉注意力用片段特征作 query、文本特征作 key，得到文本对齐特征 $\tilde{C}^i = \text{Cross-Attention}(C^i, Q)$；(b) 自注意力将 $\tilde{C}^i$ 压缩为单个向量 $\mathcal{S}^i = A_0$，其中 $A = \text{Self-Attention}([\mathcal{S}^i; \tilde{C}^i])$
    - 设计动机：对小时级视频（万级帧），如果全部以密集特征输入 LLM，token 数将远超上下文长度。稀疏特征将几分钟的片段压缩为一个 768 维向量，大幅减少输入长度。交叉注意力让稀疏特征与文本查询对齐，使得上层扫描更有方向性

2. **递归视频定位（Recursive Video Grounding）**:
    - 功能：模仿人类搜索策略，从全局到局部逐级缩小搜索范围
    - 核心思路：构建 $L$ 层层次化视频输入 $[I^{(\ell)}]_{\ell=1,...,L}$。最底层 $I^{(1)}$ 使用密集特征 $\mathcal{D}$，上层 $[I^{(\ell)}]_{\ell=2,...,L}$ 使用稀疏特征 $\mathcal{S}$。LLM 在每层 $l$ 接收输入 $P^{(l)} = [I^{(l)}, w_1,...,w_M]$，预测该层的时间边界 $\tau^{(l)}$ 输出格式为 "From $s$ to $e$." 或 "Not Present."。下层处理时将上层预测的边界 $\tau^{(<l)}$ 作为先验，逐步精化。训练目标：$p(T^{(l)}|P^{(l)}) = \prod_{k=1}^{K} p(T_k^{(\ell)} | T_{<k}^{(\ell)}, P^{(l)})$
    - 设计动机：直接在全视频上定位细粒度事件（如 2 小时电影中 4 秒片段）相当于大海捞针。递归 "zoom-in" 策略让每层只需在有限范围内做决策，难度大幅降低。同时每层输入 token 数可控，解决了长视频的显存问题

3. **渐进训练策略（Progressive Training）**:
    - 功能：分阶段训练，解决长视频训练的数据和效率问题
    - 核心思路：分两阶段——**阶段一（短片段训练）**：(a) 先用密集特征训练 LLM（LoRA），学习精确边界预测（"From $s$ to $e$."）；关键创新是引入**对比片段**——不含目标事件的视频片段，模型需输出 "Not Present."，以校准置信度。(b) 冻结 LLM，用简化目标（"Yes/No"）训练层次化适配器生成稀疏特征。**阶段二（长视频训练）**：利用阶段一学到的稀疏特征在小时级视频上训练 LoRA，定位包含事件的片段
    - 设计动机：(a) 对比片段解决 VLM 的过度自信问题——传统 VLM 只见过含事件的片段，无法识别事件缺失的情况，导致对所有片段都高置信度预测；(b) 先短后长的策略让模型先学会事件识别基础能力，再迁移到长视频场景

### 损失函数 / 训练策略

- 标准自回归语言模型训练损失（next-token prediction）
- 推理时用 **校准置信度排序**：计算 LLM 预测每个词的概率分布熵 $H_k^{(i)} = -\sum_w p(w|T_{<k}, \mathcal{D}^{(i)}) \log p(w|T_{<k}, \mathcal{D}^{(i)})$，置信度分数为平均熵的倒数 $R^i = \frac{1}{\frac{1}{K}\sum_{k=1}^{K} H_k^i}$，低熵（高置信度）的预测排在前面
- 优化器：AdamW + cosine decay；LoRA 参数 $r=64, \alpha=128$
- 阶段一训 1 epoch（lr=1e-3），阶段二训 5 epoch（MAD）或 1 epoch（VidChapters-7M），lr=1e-4

## 实验关键数据

### 主实验

MAD 数据集（小时级电影，平均 110 分钟，事件平均 4.1 秒）：

| 方法 | R1@0.1 | R5@0.1 | R1@0.3 | R5@0.3 | Avg. |
|------|--------|--------|--------|--------|------|
| RGNet | 12.4 | 25.1 | 9.5 | 18.7 | 13.7 |
| SnAG | 10.3 | 24.4 | 8.5 | 20.6 | 13.8 |
| VTimeLLM+CONE | 1.4 | 3.1 | 1.3 | 2.5 | 1.7 |
| **ReVisionLLM** | **15.0** | **25.1** | **11.0** | **18.8** | **14.4** |
| **ReVisionLLM-I** | **17.3** | **31.4** | **12.7** | **23.5** | **17.5** |

VidChapters-7M 数据集（YouTube 视频，最长 12 小时）：

| 方法 | R1@0.3 | R1@0.5 | R1@0.7 | R1@0.9 | Avg. |
|------|--------|--------|--------|--------|------|
| M-DETR | 37.4 | 27.3 | 17.6 | 6.4 | 22.1 |
| **ReVisionLLM** | **33.8** | **27.4** | **21.8** | **15.2** | **24.6** |

### 消融实验

累积消融（MAD 数据集）：

| 模块 | R1@0.1 | R5@0.1 | 说明 |
|------|--------|--------|------|
| Baseline (VTimeLLM) | 0.0 | 0.0 | 均匀采样 100 帧，完全失败 |
| +CONE 排序 | 1.4 | 2.4 | 短片段+CLIP排序 |
| +对比片段 | 4.8 | 6.7 | 学会识别事件缺失 |
| +校准置信度 | 8.4 | 12.7 | LLM 内部置信度替代 CLIP |
| +递归处理 | 15.0 | 25.1 | 最大提升来源 |

层次数消融：

| 层次数 | R1@0.1 | R5@0.1 | R1@0.3 | R5@0.3 |
|--------|--------|--------|--------|--------|
| 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 1 | 8.4 | 12.7 | 6.6 | 8.9 |
| 2 | 11.9 | 17.5 | 8.7 | 13.2 |
| 3 | 15.0 | 25.1 | 11.0 | 18.8 |

### 关键发现

- **递归处理是最关键的模块**：贡献了从 8.4% 到 15.0% 的提升（R1@0.1），几乎将性能翻倍
- **对比片段有效校准置信度**：引入后 R1@0.1 从 1.4% 提升到 4.8%，首次让 VLM 在长视频任务上有意义的表现
- **ReVisionLLM 仅处理 57% 的帧就超越了处理 100% 帧的基线**（VTimeLLM+CONE），证明递归策略的效率优势
- **视频长度鲁棒性强**：扩展到 10 小时视频后性能仅略微下降，非递归方法则完全失败
- 在文本到视频检索任务（MSRVTT）上也取得竞争性结果，证明模型学到了通用的视频-文本对应能力

## 亮点与洞察

- **人类搜索策略的仿生设计**：递归 zoom-in 完美对应了认知科学中关于人类视觉搜索的发现——先粗略扫描建立目标表征，再逐步聚焦到精确位置
- **对比片段的引入解决了VLM一个根本性的缺陷**：过度自信问题在短视频任务中不太明显，但在长视频中被急剧放大。对比训练是一个简单但关键的设计
- **稀疏特征压缩极为高效**：将几分钟视频压缩为单个向量，使得小时级视频不超过 LLM 的上下文窗口
- **用 LLM 熵做置信度排序**比 CLIP 相似度排序更有效，因为经过对比训练后 LLM 的置信度已被校准

## 局限与展望

- 稀疏特征压缩不可避免地丢失了细节信息，可能遗漏细微事件
- 层次数固定为 3，自适应层次数可能更优
- 当前只用 CLS token 全局特征，忽略了帧内空间信息
- LLM 基座为 Vicuna-7B，升级到更强 LLM 可能进一步提升
- ReVisionLLM-I 模式虽更准但需处理全部帧，效率较低

## 相关工作与启发

- 与 CONE、SnAG 等非 LLM 方法互补：ReVisionLLM 的优势在于自然语言交互能力和置信度校准
- 递归处理思路可推广到其他长序列理解任务（如长文档检索、长音频事件检测）
- 稀疏-密集特征的层次化设计与图像处理中的 coarse-to-fine 策略相呼应
- 对比片段训练策略可启发其他 VLM 解决过度自信问题

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个解决小时级视频时序定位的 VLM，递归架构设计优雅
- 实验充分度: ⭐⭐⭐⭐⭐ MAD+VidChapters-7M两个数据集，累积消融+变体对比+长度鲁棒性+泛化实验
- 写作质量: ⭐⭐⭐⭐ 从问题到方法到实验逻辑通顺
- 价值: ⭐⭐⭐⭐⭐ 开辟了VLM在小时级视频时序定位的新方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Video-XL: Extra-Long Vision Language Model for Hour-Scale Video Understanding](video-xl_extra-long_vision_language_model_for_hour-scale_video_understanding.md)
- [\[CVPR 2025\] VideoGLaMM: A Large Multimodal Model for Pixel-Level Visual Grounding in Videos](videoglamm_a_large_multimodal_model_for_pixel-level_visual_grounding_in_videos.md)
- [\[CVPR 2025\] VidComposition: Can MLLMs Analyze Compositions in Compiled Videos?](vidcomposition_can_mllms_analyze_compositions_in_compiled_videos.md)
- [\[CVPR 2025\] Your Large Vision-Language Model Only Needs a Few Attention Heads for Visual Grounding](your_large_vision-language_model_only_needs_a_few_attention_heads_for_visual_gro.md)
- [\[CVPR 2025\] MIMO: A Medical Vision Language Model with Visual Referring Multimodal Input and Pixel Grounding Multimodal Output](mimo_a_medical_vision_language_model_with_visual_referring_multimodal_input_and_.md)

</div>

<!-- RELATED:END -->
