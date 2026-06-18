---
title: >-
  [论文解读] LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models
description: >-
  [ICCV 2025][多模态VLM][视觉token压缩] 利用视觉编码器中CLS token与空间token之间注意力分数的稀疏性，自适应地剪枝和合并视觉token，在仅保留5.5%视觉token的情况下维持LMM的可比性能。 大型多模态模型（LMM）通过连接视觉编码器（如CLIP-ViT）和大型语言模型来实现视觉推理…
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "视觉token压缩"
  - "大型多模态模型"
  - "token剪枝与合并"
  - "注意力稀疏性"
  - "高效推理"
---

# LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models

**会议**: ICCV 2025  
**arXiv**: [2403.15388](https://arxiv.org/abs/2403.15388)  
**代码**: [https://github.com/yuzhangshang/LLaVA-PruMerge](https://github.com/yuzhangshang/LLaVA-PruMerge)  
**领域**: 多模态VLM  
**关键词**: 视觉token压缩, 大型多模态模型, token剪枝与合并, 注意力稀疏性, 高效推理

## 一句话总结

利用视觉编码器中CLS token与空间token之间注意力分数的稀疏性，自适应地剪枝和合并视觉token，在仅保留5.5%视觉token的情况下维持LMM的可比性能。

## 研究背景与动机

大型多模态模型（LMM）通过连接视觉编码器（如CLIP-ViT）和大型语言模型来实现视觉推理。然而，这类模型面临严重的效率问题：LLM骨干网络的计算复杂度随输入token数量呈**二次增长**。以LLaVA-1.5为例，单张图像需要576个视觉token作为前缀输入LLM；Video-LLaVA甚至需要2048个视觉token。随着高分辨率图像和视频理解需求的增长，视觉token数量还在不断膨胀。

现有提升LMM效率的方法主要有两条路线：一是用更小的LLM骨干（如Phi-2替代LLaMA-7B），但这会牺牲推理能力；二是量化压缩，但仍无法解决输入序列过长的根本问题。**关键矛盾在于**：LMM需要大量视觉token来全面表达视觉内容，但过多的token又带来难以承受的计算开销。

作者的核心观察是：视觉编码器中存在**显著的空间冗余**。具体而言，在ViT最后一层中，CLS token与绝大多数空间token之间的注意力分数近乎为零（呈高度稀疏分布），这表明只有少数视觉token携带了关键视觉信息。由此，作者提出了一个自然的切入角度：能否利用这种稀疏性来自适应地选择重要的视觉token，从而大幅减少LLM的输入序列长度？

## 方法详解

### 整体框架

PruMerge是一个即插即用的视觉token压缩模块，插入在视觉编码器和LLM之间。整个流程分三步：(1) 利用异常检测算法从CLS-空间注意力分数中筛选重要token；(2) 基于Key向量相似度对剩余token进行K近邻聚类；(3) 通过加权平均将被裁减的token信息融合回保留的anchor token中。

### 关键设计

#### 1. 自适应重要Token选择（AITS）——基于异常检测

- **功能**：根据每张图像的内容复杂度，自适应决定保留多少个视觉token。
- **核心思路**：利用ViT倒数第二层中 $\mathbf{a}_{\text{cls}} = \text{softmax}\left(\frac{\mathbf{q}_{\text{cls}} \cdot \mathbf{K}^T}{\sqrt{d_k}}\right)$ 计算出的CLS-空间注意力分数。这些分数呈高度稀疏分布——绝大多数token的注意力值接近零，只有少数token有较高的值。采用四分位距（IQR）异常检测方法：上界 fence = Q3 + 1.5 × IQR，超过上界的token被视为"异常值"（即重要token）并保留。
- **设计动机**：简单图像（如蓝天中的广告牌）只需少量token即可充分表达，而包含密集文字的复杂图像则需要更多token。IQR方法无需手动设定阈值，能根据图像复杂度自动调整保留数量。实验显示，不同benchmark的平均token数差异明显（如ScienceQA仅需16个，TextVQA需40个），验证了自适应性的必要性。

#### 2. Token补充（TS）——基于相似Key聚类

- **功能**：将被剪枝的token信息融合到保留的anchor token中，避免信息损失。
- **核心思路**：利用ViT最后一层的Key向量计算token间相似度 $\text{Sim}(\mathbf{y}_i, \mathbf{y}_j) = \mathbf{k}_i \cdot \mathbf{k}_j^T$，对每个unpruned token找K近邻（被裁减的token作为候选），然后以CLS注意力值 $\mathbf{a}[i]$ 为权重做加权平均，更新anchor token的表示。
- **设计动机**：当图像中有大面积物体时（如全景中的建筑），过度剪枝可能丢失重要的空间信息。通过合并而非简单丢弃，被裁减token的信息得以保留。Key向量作为相似度度量的选择源于——Key已经在自注意力过程中聚合了位置和语义信息。时间复杂度为 $O(n)$，优于CrossGet的 $O(n^2)$。

#### 3. PruMerge+——空间均匀采样增强

- **功能**：在PruMerge的基础上，额外从"不重要"区域中按空间均匀分布采样补充token。
- **核心思路**：以异常token数量为参考比例，在非异常区域中等间距采样，确保被忽略区域也有代表性token。最终token数约为原始的25%（约144个），远少于原始576个，但显著降低了性能损失。
- **设计动机**：纯异常检测可能遗漏虽注意力值不高但对空间布局理解有用的区域。空间均匀采样弥补了这一缺陷，在token压缩率和性能之间取得了更好的平衡。

### 损失函数 / 训练策略

- PruMerge可以在**无训练**模式下直接使用（training-free），也可以通过**LoRA微调**进一步适配。
- 微调时使用LLaVA-1.5原始的指令微调数据，仅训练1个epoch。
- 微调让LLM适应压缩后的视觉token结构，在大多数benchmark上带来进一步提升。

## 实验关键数据

### 主实验

| 方法 | LLM | VQAv2 | SQA-I | TextVQA | POPE | MME | MMB |
|------|-----|-------|-------|---------|------|-----|-----|
| LLaVA-1.5 | Vicuna-7B | 78.5 | 66.8 | 58.2 | 85.9 | 1510.7 | 64.3 |
| + PruMerge (5.5% tokens) | Vicuna-7B | 72.0 | **68.5** | 56.0 | 76.3 | 1350.3 | 60.9 |
| + PruMerge+ (25% tokens) | Vicuna-7B | 76.8 | **68.3** | 57.1 | 84.0 | 1462.4 | 64.9 |
| LLaVA-1.5 | Vicuna-13B | 80.0 | 71.6 | 61.3 | 85.9 | 1531.3 | 67.7 |
| + PruMerge+ (25% tokens) | Vicuna-13B | 77.8 | 71.0 | 58.6 | 84.4 | 1485.5 | 65.7 |

PruMerge+在ScienceQA上甚至超过原始LLaVA-1.5，说明去除冗余token实际上有助于模型聚焦关键信息。

### 消融实验

| 配置 | TextVQA | MME | POPE | 说明 |
|------|---------|-----|------|------|
| PruMerge (AITS only) | 54.8 | 1221.6 | 75.7 | 仅剪枝，无合并 |
| PruMerge (AITS + TS) | 56.0 | 1350.3 | 76.3 | 加入token合并，性能显著恢复 |
| Sequential Sampling (40 tokens) | 42.7 | 703.6 | 11.7 | 随机序列采样，性能崩溃 |
| Spatial 5×8 (40 tokens) | 46.9 | 1180.2 | 69.8 | 均匀空间采样 |
| PruMerge (40 tokens) | **54.0** | **1250.1** | **76.2** | 自适应选择明显优于固定策略 |

效率分析：PruMerge将LLM Prefill FLOPs从9.3TB降至0.91TB（降低**10倍**），Prefill时间从88.6ms降至15.3ms，激活内存从4.60GB降至0.28GB。

### 关键发现

- 与单模态token压缩方法（ToMe、EViT、ATS）在LMM上的对比中，PruMerge+以25%压缩率取得76.8 VQAv2分数，大幅超越ToMe(66.0)、ATS(66.7)、EViT(65.5)。
- Video-LLaVA上**无需训练**即可使用PruMerge，且反而提升了性能，说明视频LLM中存在更严重的token冗余。
- 注意力稀疏性是**跨模型普遍存在**的现象，不局限于特定的ViT架构。

## 亮点与洞察

- **稀疏性即信号**：CLS注意力的稀疏分布不是缺陷，而是天然的token重要性指标。将统计异常检测用于视觉token选择是一个简洁而有效的创新。
- **即插即用**：PruMerge不需要修改视觉编码器或LLM内部结构，仅在二者之间插入轻量模块。
- **区分LMM与ViT的token压缩**：作者明确指出LMM的效率瓶颈在LLM而非ViT，因此token压缩的目标应是减少LLM的输入长度，而非ViT的内部计算——这与单模态方法的设计理念根本不同。

## 局限与展望

- VQAv2和POPE等依赖全局空间理解的任务上仍有可观的性能下降（PruMerge在POPE上从85.9降到76.3）。
- 当前的IQR阈值是固定策略，未针对不同任务或数据集动态调整。
- 仅在LLaVA和Video-LLaVA上验证，缺少在更先进的LMM（如Qwen-VL、InternVL）上的实验。
- token选择是在最后一层做的一次性决策，未考虑多层次信息的逐步筛选。

## 相关工作与启发

- 与ToMe（Token Merging）的思路相承，但ToMe专注于ViT内部的逐层token合并，时间复杂度更高且不适合LMM场景。
- 对高分辨率LMM（如LLaVA-Next、Monkey等动辄数千token的模型）有很强的实用价值。
- 启发方向：能否将token选择与LLM内部的注意力机制联合优化？例如让LLM在推理过程中动态请求更多/更少的视觉token。

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] AVG-LLaVA: An Efficient Large Multimodal Model with Adaptive Visual Granularity](../../ACL2025/multimodal_vlm/avg-llava_an_efficient_large_multimodal_model_with_adaptive_visual_granularity.md)
- [\[ICCV 2025\] LLaVA-KD: A Framework of Distilling Multimodal Large Language Models](llava-kd_a_framework_of_distilling_multimodal_large_language_models.md)
- [\[ICCV 2025\] On Large Multimodal Models as Open-World Image Classifiers](on_large_multimodal_models_as_open-world_image_classifiers.md)
- [\[ICCV 2025\] METEOR: Multi-Encoder Collaborative Token Pruning for Efficient Vision Language Models](meteor_multi-encoder_collaborative_token_pruning_for_efficient_vision_language_m.md)
- [\[ICCV 2025\] GRAB: A Challenging GRaph Analysis Benchmark for Large Multimodal Models](grab_a_challenging_graph_analysis_benchmark_for_large_multimodal_models.md)

</div>

<!-- RELATED:END -->
