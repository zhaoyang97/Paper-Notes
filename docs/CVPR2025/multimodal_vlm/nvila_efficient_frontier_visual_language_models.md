---
title: >-
  [论文解读] NVILA: Efficient Frontier Visual Language Models
description: >-
  [CVPR 2025][多模态VLM][视觉语言模型] NVILA 提出"先放大再压缩"(Scale-then-Compress)的范式，通过提升空间和时间分辨率后再压缩视觉Token，在保持甚至超越SOTA精度的同时，将训练成本降低1.9-5.1倍、推理预填充延迟降低1.6-2.2倍、解码延迟降低1.2-2.8倍。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "视觉语言模型"
  - "高效训练"
  - "视觉Token压缩"
  - "多分辨率处理"
  - "模型部署"
---

# NVILA: Efficient Frontier Visual Language Models

**会议**: CVPR 2025  
**arXiv**: [2412.04468](https://arxiv.org/abs/2412.04468)  
**代码**: [https://github.com/NVlabs/NVILA](https://github.com/NVlabs/NVILA)  
**领域**: 多模态VLM  
**关键词**: 视觉语言模型, 高效训练, 视觉Token压缩, 多分辨率处理, 模型部署

## 一句话总结

NVILA 提出"先放大再压缩"(Scale-then-Compress)的范式，通过提升空间和时间分辨率后再压缩视觉Token，在保持甚至超越SOTA精度的同时，将训练成本降低1.9-5.1倍、推理预填充延迟降低1.6-2.2倍、解码延迟降低1.2-2.8倍。

## 研究背景与动机

视觉语言模型(VLM)近年来在精度上取得显著进展，但其效率问题长期被忽视。VLM在多个维度上成本高昂：(1) 训练一个7B VLM需要400 GPU-days；(2) 全量微调7B VLM需要超过64GB显存；(3) 边缘部署（笔记本、机器人）资源极度受限。现有方法要么牺牲精度换效率，要么暴力堆分辨率导致计算爆炸（分辨率翻倍，自注意力计算量翻四倍）。核心矛盾在于：高分辨率/长视频带来的信息增益与Token数量爆炸之间的不可调和。NVILA的切入角度是"先获取最大信息量，再高效压缩"，通过全生命周期的效率优化（训练→微调→部署）系统性解决这一问题。核心idea：高信息密度的压缩Token优于低分辨率的原始Token。

## 方法详解

### 整体框架

NVILA 基于 VILA 构建，是一个自回归VLM，由三部分组成：视觉编码器（SigLIP）提取视觉特征，投影器（两层MLP）对齐跨模态嵌入，Token处理器（Qwen2 LLM）接收视觉和语言Token并输出语言Token。整体pipeline采用五阶段训练：投影器初始化→视觉编码器预训练→Token处理器预训练→图像指令微调→视频指令微调。

### 关键设计

1. **Dynamic-S2 空间放大**:
    - 功能：自适应处理不同宽高比的高分辨率图像
    - 核心思路：基于S2多尺度Tiling策略，在最大尺度上不再强制resize为正方形，而是自适应调整图像尺寸至最接近原始宽高比且可被 $448^2$ 整除的大小。各尺度Feature Map插值到最大尺度大小后在通道维度拼接
    - 设计动机：原始S2总是将图像resize为正方形，对窄长或宽扁图像造成严重变形。Dynamic-S2灵感来自InternVL的动态分辨率策略，在文本密集型Benchmark上带来高达30%的精度提升

2. **空间Token压缩 (STC + VEP)**:
    - 功能：将视觉Token数量压缩2.4倍，同时保持精度
    - 核心思路：使用 $3\times3$ Spatial-to-Channel (STC) reshape将Token数从 $16\times16=256$ 压缩到 $11\times11=121$。由于激进压缩使投影器训练困难，额外引入视觉编码器预训练阶段(VEP)联合调整视觉编码器和投影器
    - 设计动机：简单的 $2\times2$ STC无损压缩已被VILA验证，但更大压缩比（如 $3\times3$）会导致DocQA下降约10%。VEP阶段成功恢复了大部分精度损失。实验还表明TokenLearner、Perceiver Resampler等可学习压缩方法在同等压缩比下并不优于简单STC

3. **时间Token压缩 (Temporal Averaging)**:
    - 功能：处理长视频（最多256帧）的同时控制Token数量
    - 核心思路：将视频帧分组后在组内进行时间维度平均池化，利用视频固有的时间连续性消除冗余。例如32帧压缩4倍后Token数与8帧基线相同，但精度高出5%以上
    - 设计动机：连续帧通常包含相似信息，时间池化能在保留重要时空信息的同时有效降低冗余。进一步扩展到256帧配合8倍压缩，在Video-MME上实现7B模型SOTA

### 损失函数 / 训练策略

- **DeltaLoss数据集剪枝**：用大模型和小模型的输出概率比 $\log \frac{p_{\text{large}}(x)}{p_{\text{small}}(x)}$ 对训练数据打分。ratio接近0的样本过于简单（两个模型都对/都错），ratio为负的样本有干扰性（小模型对大模型错），ratio为正的样本最有学习价值（小模型错大模型对）。剪枝50%数据几乎无精度损失，训练提速2倍
- **FP8混合精度训练**：利用H100原生FP8支持，将权重和激活量化到FP8，允许batch size从4扩大到16，训练吞吐提升2倍。搭配gradient checkpointing时仍有1.2倍加速
- **高效微调策略**：ViT部分的学习率应比LLM部分小5-50倍；ViT仅调LayerNorm即可达到LoRA级效果，且训练时间减少25%；搭配QLoRA可在24GB显存内微调
- **量化部署**：视觉编码器W8A8量化几乎无损，LLM骨干W4A16量化配合优化的FP16累加GEMM核带来1.7倍核心加速

## 实验关键数据

### 主实验（图像）

| Benchmark | 指标 | NVILA-8B | Qwen2-VL-8B | LLaVA-OV-8B | GPT-4o |
|-----------|------|----------|-------------|-------------|--------|
| AI2D | test | **92.3** | 83.0 | 81.4 | 94.2 |
| DocVQA | test | **93.7** | 94.5 | 87.5 | 92.8 |
| ChartQA | test | **86.1** | 83.0 | 80.0 | 85.7 |
| TextVQA | val | 80.1 | **84.3** | 78.3 | 78.7 |
| MMMU | val | 49.9 | **54.1** | 48.8 | 69.1 |
| MathVista | testmini | **65.4** | 58.2 | 63.2 | 63.8 |

### 主实验（视频）

| Benchmark | NVILA-8B (256帧) | Qwen2-VL-8B | LLaVA-OV-8B | GPT-4o |
|-----------|-----------------|-------------|-------------|--------|
| Video-MME (w/o sub) | **64.2** | 63.3 | 58.2 | 71.9 |
| Video-MME (w/ sub) | **70.0** | 69.0 | 61.5 | 77.2 |
| MLVU m-avg | **70.1** | 65.5 | 64.7 | 64.6 |
| MVBench | **68.1** | 67.0 | 56.7 | - |
| ActivityNet-QA | **60.9** | - | 56.6 | - |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 数据剪枝 50% (DeltaLoss) | IM-10: 75.5 vs 基线75.6 | 精度几乎无损，训练提速2× |
| 数据剪枝 50% (Random) | IM-10: 74.0 | 远差于DeltaLoss |
| FP8 无GC | 吞吐 390.1 vs BF16 199.2 | 2.0× 加速 |
| 空间压缩 3×3 STC + VEP | IM-10: 70.8 vs 无VEP 67.1 | VEP恢复了3.7个点 |
| 时间压缩 32帧4× | Video-MME: 60.1 vs 不压缩61.0 | 仅0.9%精度损失，Token数减4× |

### 关键发现

- "先放大再压缩"策略使NVILA在相同Token预算下拥有比直接低分辨率更高的精度
- 简单的STC空间压缩+VEP预训练阶段优于TokenLearner和Perceiver Resampler等复杂可学习方法
- DeltaLoss数据剪枝尤其在DocQA等任务上远优于随机和聚类剪枝
- NVILA-8B在视频理解所有Benchmark上都超越同尺寸开源模型，甚至接近GPT-4o mini

## 亮点与洞察

- **全生命周期效率优化**：从训练数据剪枝、FP8训练、高效微调到量化部署，形成一套完整的效率优化方案论
- **简单胜过复杂**：STC reshape比TokenLearner、Perceiver Resampler更有效，验证了"好的信息压缩不需要可学习参数"的直觉
- **VEP阶段是关键创新**：解决了激进压缩导致投影器训练困难的问题，是"补偿性预训练"的成功实践
- **DeltaLoss数据剪枝的哲学**：过于简单或过于困难的样本都不利于学习，只有"对大模型有信息量但对小模型有挑战"的样本最有价值

## 局限与展望

- MMMU等知识推理Benchmark上仍落后于Qwen2-VL，说明"压缩"可能丢失了部分高级语义信息
- DeltaLoss需要用大小两个模型分别推理所有数据打分，计算开销本身不低
- 256帧处理仍需视觉编码器逐帧推理（只是不是主要瓶颈），未来可探索更高效的视频编码方式
- 论文聚焦token压缩策略是"简单设计"，可能在极端分辨率场景下不够灵活

## 相关工作与启发

- **vs LLaVA-OneVision**: NVILA训练成本仅为其1/5，精度更高，充分说明"效率vs精度"不是trade-off而是可以同时优化的
- **vs Qwen2-VL**: NVILA在同等精度下推理快1.6-2.8倍，得益于token压缩使视觉输入更紧凑
- **vs InternVL2**: NVILA借鉴了其动态分辨率思想，但进一步引入压缩使Token更高效

## 评分

- 新颖性: ⭐⭐⭐⭐ "先放大再压缩"范式清晰优雅但各组件并非全新，VEP阶段是有价值的工程创新
- 实验充分度: ⭐⭐⭐⭐⭐ 涵盖10+图像benchmark、6+视频benchmark、效率对比、消融实验、下游应用（机器人/医学）非常全面
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，"scale-then-compress"主线贯穿始终，图表精美
- 价值: ⭐⭐⭐⭐⭐ 全栈效率优化方案对VLM社区有极高实用价值，代码和模型开源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] FastVLM: Efficient Vision Encoding for Vision Language Models](fastvlm_efficient_vision_encoding_for_vision_language_models.md)
- [\[CVPR 2025\] Skip Tuning: Pre-trained Vision-Language Models are Effective and Efficient Adapters Themselves](skip_tuning_pre-trained_vision-language_models_are_effective_and_efficient_adapt.md)
- [\[ACL 2025\] A Parameter-Efficient and Fine-Grained Prompt Learning for Vision-Language Models](../../ACL2025/multimodal_vlm/a_parameter-efficient_and_fine-grained_prompt_learning_for_vision-language_model.md)
- [\[NeurIPS 2025\] Efficient Multi-modal Large Language Models via Progressive Consistency Distillation](../../NeurIPS2025/multimodal_vlm/efficient_multi-modal_large_language_models_via_progressive_consistency_distilla.md)
- [\[ACL 2025\] Activating Distributed Visual Region within LLMs for Efficient and Effective Vision-Language Training and Inference](../../ACL2025/multimodal_vlm/activating_distributed_visual_region_within_llms_for_efficient_and_effective_vis.md)

</div>

<!-- RELATED:END -->
