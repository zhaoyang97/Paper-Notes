---
title: >-
  [论文解读] FastVLM: Efficient Vision Encoding for Vision Language Models
description: >-
  [CVPR 2025][多模态VLM][视觉编码器] 提出混合卷积-Transformer视觉编码器 FastViTHD，通过 5 阶段架构实现 32× 空间下采样，在同等精度下比 ViT-L/14 生成 16× 更少的视觉 token 且编码速度提升 3.7×，TTFT 降低高达 85×。 领域现状：当前主流 VLM（如…
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "视觉编码器"
  - "Transformer"
  - "高效VLM"
  - "token压缩"
  - "TTFT优化"
---

# FastVLM: Efficient Vision Encoding for Vision Language Models

**会议**: CVPR 2025  
**arXiv**: [2412.13303](https://arxiv.org/abs/2412.13303)  
**代码**: [https://github.com/apple/ml-fastvlm](https://github.com/apple/ml-fastvlm)  
**领域**: 多模态VLM  
**关键词**: 视觉编码器、混合卷积-Transformer、高效VLM、token压缩、TTFT优化

## 一句话总结
提出混合卷积-Transformer视觉编码器 FastViTHD，通过 5 阶段架构实现 32× 空间下采样，在同等精度下比 ViT-L/14 生成 16× 更少的视觉 token 且编码速度提升 3.7×，TTFT 降低高达 85×。

## 研究背景与动机

**领域现状**：当前主流 VLM（如 LLaVA）使用 ViT 作为视觉编码器，通过提升输入分辨率来增强文本密集型任务的能力。ViT 对图像进行 14× 或 16× 下采样产生 patch token，高分辨率输入意味着大量视觉 token。

**现有痛点**：ViT 的 token 数量与分辨率平方成正比增长——336px 分辨率就产生 576 个 token，1024px 则高达数千个。这导致两个瓶颈：(1) 编码器延迟高，(2) LLM prefilling 大量视觉 token 造成 TTFT 过长。现有对策（token pruning、动态分辨率 AnyRes）都是事后补救，引入额外开销或破坏语义连续性。

**核心矛盾**：高分辨率对 VLM 性能至关重要（尤其 OCR、文档理解），但 ViT 架构天然无法高效处理高分辨率输入——分辨率提升带来的性能收益被延迟增长抵消。

**本文目标** 设计一个从架构层面就高效的视觉编码器，在高分辨率下产生极少的 token，从根本上解决分辨率-延迟-精度三角矛盾。

**切入角度**：混合卷积-Transformer 架构天然具有层级下采样能力，每个阶段都可以降低空间分辨率。通过增加一个额外的第 5 阶段实现 32× 下采样（而非常规的 16×），让自注意力在极低分辨率的特征图上运算。

**核心 idea**：用 5 阶段混合卷积-Transformer（FastViTHD）替换 ViT 作为 VLM 视觉编码器，通过架构级 token 减少实现比 token pruning 更优的精度-效率权衡。

## 方法详解

### 整体框架
标准 LLaVA 架构：FastViTHD 视觉编码器 → 视觉-语言投影层 → LLM 解码器（Vicuna/Qwen2）。输入图像经 FastViTHD 编码为少量视觉 token（如 1024px 仅 256 个 token），经投影后与文本 token 拼接送入 LLM。

### 关键设计

1. **FastViTHD 5 阶段架构**:

    - 功能：从图像中提取视觉特征，同时将空间分辨率降到极低
    - 核心思路：5 个阶段依次处理，前三阶段用轻量的 RepMixer 卷积块做局部特征提取和空间下采样，第 4-5 阶段用多头自注意力做全局特征建模。阶段深度为 [2, 12, 24, 4, 2]，embedding 维度逐阶段翻倍 [96, 192, 384, 768, 1536]。关键创新在第 5 阶段：额外增加一层下采样，使自注意力操作在 32× 下采样后的特征图上进行（常规混合架构如 ViTamin 是 16×），直接将 token 数降低 4×
    - 设计动机：4 阶段设计在高分辨率下仍然需要在较大特征图上做自注意力，延迟甚至超过 ConvNeXt-L。第 5 阶段的额外下采样以极小的参数增量（总共 125M，仍比 ViT-L/14 的 304M 小 2.4×）换来了根本性的 token 数量和延迟优势

2. **多尺度特征聚合**:

    - 功能：补充高层特征丢失的底层局部细节信息
    - 核心思路：将前几个卷积阶段的特征通过 2D 深度可分离卷积（DWConv）池化到与最终特征图相同的空间尺寸，然后与第 5 阶段输出拼接。DWConv 比 AvgPool 略好，因为它保贝更多局部结构信息
    - 设计动机：32× 下采样丢失了大量细粒度信息（如文字边缘、小目标），多尺度聚合弥补了这一损失

3. **静态分辨率优于 AnyRes 动态分辨率**:

    - 功能：确定最佳的输入分辨率处理策略
    - 核心思路：直接将输入图像 resize 到目标分辨率（如 512/768/1024）送入编码器，而不是像 LLaVA-NeXT 那样将高分辨率图像切片（tile）后分别编码再拼接。实验证明在绝大多数分辨率下，静态缩放的精度-延迟权衡更优。AnyRes 仅在极高分辨率（1536×1536+）且使用少量 tile（2×2）时略有优势
    - 设计动机：切片破坏了跨 tile 的语义连续性，且引入大量额外 token；FastViTHD 天然支持任意分辨率输入，无需切片

### 损失函数 / 训练策略
采用 3 阶段训练：Stage 1 冻结编码器和 LLM 只训练投影层（558K 对齐数据）→ Stage 1.5 解冻编码器和 LLM 在 15M 密集描述数据上做分辨率适配 → Stage 2 在 1.1M-23.1M 视觉指令微调数据上全参数微调 → 可选 Stage 3 用 MammothVL 10.6M CoT 推理数据进一步提升。FastViTHD 的 CLIP 预训练使用 DataCompDR-1B 数据集。

## 实验关键数据

### 主实验

| 方法 | 编码器 | 分辨率 | #Token | TTFT(ms) | GQA | TextVQA | DocVQA | Avg |
|------|--------|--------|--------|----------|-----|---------|--------|-----|
| LLaVA-1.5 | ViT-L/14 | 336 | 576 | 127ms+ | 62.0 | 58.2 | 28.1 | 60.1 |
| FastVLM | FastViT | 768 | 576 | 34.5ms | 62.7 | 62.3 | 34.4 | 62.6 |
| FastVLM | FastViTHD | 1024 | 256 | 236ms | 63.1 | 64.4 | 35.6 | 63.9 |
| LLaVA-OV (0.5B) | SigLIP-SO400M | 1152 | 7290 | 14124ms | - | - | 70.0 | - |
| FastVLM (0.5B) | FastViTHD | 1024 | 256 | 166ms | 63.1 | 62.9 | 70.4 | - |
| FastVLM (7B) | FastViTHD+Qwen2 | 1024 | 256 | 641ms | 65.2 | 73.4 | 82.7 | - |

### 消融实验

| 配置 | GQA | TextVQA | DocVQA | Avg-5 | 说明 |
|------|-----|---------|--------|-------|------|
| 无多尺度特征 | 62.7 | 62.3 | 34.4 | 62.6 | baseline |
| +多尺度 (AvgPool) | 63.0 | 62.2 | 35.1 | 62.7 | 轻微提升 |
| +多尺度 (DWConv) | 63.0 | 62.5 | 34.7 | 62.9 | DWConv 更优 |
| ViT+MQT pruning 16 token | 57.6 | - | - | - | token pruning 效果差 |
| FastViTHD 256px 16 token | 60.6 | 53.1 | - | - | 架构减 token 远优于 pruning |

### 关键发现
- **架构级 token 减少 >> token pruning**：FastViTHD@256 仅用 16 个 token 就比 ViT-L/14 + MQT pruning (16 token) 在 GQA 上高 3 个点，说明从编码器架构出发比事后裁剪更有效
- **第 5 阶段是关键创新**：4 阶段设计（16× 下采样）在高分辨率下延迟甚至超过 ConvNeXt-L，25 阶段（32× 下采样）才实现了帕累托最优
- **分辨率提升在小 LLM 上收益递减**：高分辨率配 0.5B LLM 不如中等分辨率配 7B LLM，因为小 LLM 无法有效利用大量 token 且 TTFT 被编码器延迟主导
- **数据规模持续提升性能**：指令微调数据从 1.1M 扩展到 23.1M 持续带来增益，说明 FastViTHD 不是性能瓶颈

## 亮点与洞察
- **"从架构解决 token 冗余"的思路比 token pruning 更优雅**：pruning 方法在 ViT 上用复杂策略选择保留哪些 token，而 FastViTHD 通过层级下采样天然产生少量高质量 token，避免了信息损失。这个思路可以迁移到视频 VLM 中解决时间维度 token 爆炸问题
- **静态分辨率优于 AnyRes 的发现具有实践意义**：表明目前流行的 AnyRes/tile 策略并非最优选择，简单的 resize 在有高效编码器时反而更好。这挑战了"高分辨率必须切片处理"的主流范式
- **85× TTFT 加速**（vs LLaVA-OneVision）在移动端 VLM 部署上非常有价值

## 局限与展望
- FastViTHD 的 CLIP 预训练仍然使用传统的对比学习范式，没有探索更新的预训练方法（如 SigLIP、EVA-CLIP）
- 32× 下采样不可避免地丢失极细粒度信息，在需要像素级理解的任务（如 grounding、细粒度 OCR）上可能存在天花板
- 未与 InternVL、Qwen-VL 等使用动态分辨率的最新 SOTA 在完全相同设置下对比
- 5 阶段设计的超参（每阶段深度、维度）是否最优？搜索空间很大但论文只报告了一组配置

## 相关工作与启发
- **vs LLaVA-NeXT AnyRes**：AnyRes 通过切片处理高分辨率，但引入大量 token 和高延迟。FastVLM 证明用高效编码器 + 静态分辨率可以达到更好的精度-延迟权衡
- **vs VisionZip / DynamicLLaVA (token pruning)**：这些方法在 ViT 输出后做 token 选择/合并，但信息在 ViT 编码阶段就已经均匀分布在大量 token 中了。FastViTHD 从编码器设计出发，在生成 token 时就控制数量
- **vs ConvNeXt-XXL**：ConvNeXt 也是纯卷积的高效编码器，但 FastViTHD@1024 在匹配精度的同时参数量小 6.8 倍，速度快 1.7 倍，得益于混合架构结合了卷积的局部效率与自注意力的全局建模

## 评分
- 新颖性: ⭐⭐⭐⭐ 5 阶段混合架构设计思路简洁有效，但整体框架仍是 LLaVA 的直接替换
- 实验充分度: ⭐⭐⭐⭐⭐ 极其全面的对比实验，包括 vs ViT/ConvNeXt/token pruning、不同 LLM 规模、静态 vs 动态分辨率、数据规模消融
- 写作质量: ⭐⭐⭐⭐ 实验组织清晰，帕累托分析图很有说服力，但方法部分偏简短
- 价值: ⭐⭐⭐⭐⭐ 对移动端和实时 VLM 部署有直接工程价值，Apple 出品有落地可信度

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Skip Tuning: Pre-trained Vision-Language Models are Effective and Efficient Adapters Themselves](skip_tuning_pre-trained_vision-language_models_are_effective_and_efficient_adapt.md)
- [\[CVPR 2025\] NVILA: Efficient Frontier Visual Language Models](nvila_efficient_frontier_visual_language_models.md)
- [\[ECCV 2024\] BRAVE: Broadening the Visual Encoding of Vision-Language Models](../../ECCV2024/multimodal_vlm/brave_broadening_the_visual_encoding_of_vision-language_models.md)
- [\[CVPR 2025\] What's in the Image? A Deep-Dive into the Vision of Vision Language Models](whats_in_the_image_a_deep-dive_into_the_vision_of_vision_language_models.md)
- [\[CVPR 2025\] Words or Vision: Do Vision-Language Models Have Blind Faith in Text?](words_or_vision_do_vision-language_models_have_blind_faith_in_text.md)

</div>

<!-- RELATED:END -->
