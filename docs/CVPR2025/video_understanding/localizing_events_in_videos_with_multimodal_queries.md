---
title: >-
  [论文解读] Localizing Events in Videos with Multimodal Queries
description: >-
  [CVPR 2025][视频理解][多模态查询] 提出 ICQ 基准和 ICQ-Highlight 数据集，首次系统研究用多模态查询（图像+文本）替代纯文本查询进行视频事件定位，并设计 3 种查询适配方法和 SUIT 代理微调策略。 视频事件定位（包括 moment retrieval、highlight detection…
tags:
  - "CVPR 2025"
  - "视频理解"
  - "多模态查询"
  - "视频事件定位"
  - "基准测试"
  - "查询适配"
  - "视频时空定位"
---

# Localizing Events in Videos with Multimodal Queries

**会议**: CVPR 2025  
**arXiv**: [2406.10079](https://arxiv.org/abs/2406.10079)  
**代码**: [https://icq-benchmark.github.io/](https://icq-benchmark.github.io/)  
**领域**: 视频理解  
**关键词**: 多模态查询, 视频事件定位, 基准测试, 查询适配, 视频时空定位

## 一句话总结

提出 ICQ 基准和 ICQ-Highlight 数据集，首次系统研究用多模态查询（图像+文本）替代纯文本查询进行视频事件定位，并设计 3 种查询适配方法和 SUIT 代理微调策略。

## 研究背景与动机

视频事件定位（包括 moment retrieval、highlight detection、temporal grounding）长期以来依赖纯文本查询（NLQ），但实际应用中存在显著局限：

1. **文本查询的歧义性**：用户倾向写简短查询如"游泳"，但可能指自由泳、蝶泳等多种模式，NLQ 无法精确描述
2. **非语言概念难以表达**：不熟悉的物体、抽象的美学概念（几何风格等）难以用文字准确描述
3. **语言障碍**：对不识字用户或跨语言场景，图像查询更直观
4. **现有方法无法直接处理多模态查询**：所有 NLQ-based 模型的输入编码器仅接受文本

因此，**多模态查询（MQ）= 参考图像 + 修正文本**是一种更灵活通用的范式，但面临两个挑战：视觉查询可能引入无关细节，且参考图像与目标视频之间存在分布偏移。

## 方法详解

### 整体框架

ICQ 包含三大贡献：

- **ICQ-Highlight 数据集**：基于 QVHighlights 验证集构建，为每个原始文本查询创建多模态查询（4 种参考图像风格 × 修正文本），包含人工标注
- **3 种多模态查询适配方法 (MQA)**：将 MQ 转换为已有 NLQ 模型可用的输入
- **SUIT 代理微调策略**：用伪 MQ 微调 MLLM 提升适配质量

### 关键设计

1. **多模态查询定义与数据构建**：
    - 参考图像 $v_{ref}$：通过 DALL-E-2 和 Stable Diffusion 生成 4 种风格——涂鸦(scribble)、卡通(cartoon)、电影(cinematic)、写实(realistic)
    - 修正文本 $t_{ref}$：分 5 类——物体、动作、关系、属性、环境，提供补充或纠正信息
    - 人工标注：每个查询由不同标注者标注和审核，确保一致性
    - 任务定义：给定 $q_m = (v_{ref}, t_{ref})$，预测视频中所有相关片段 $[\tau_{start}, \tau_{end}]$

2. **三种 MQA 适配方法**：
    - **MQ-Cap (Language-Space)**：用 MLLM(LLaVA) 为参考图像生成描述 → LLM(GPT-3.5) 整合修正文本 → 生成 NLQ 输入。两步分离，更可控
    - **MQ-Sum (Language-Space)**：用 MLLM 一步将参考图像和修正文本合并为文本摘要。更简洁但不够可控，对 prompt 敏感
    - **VQ-Enc (Embedding-Space)**：直接用 CLIP 视觉编码器编码参考图像为查询嵌入 $e_q$，利用 CLIP 双流编码器的共享嵌入空间。不使用修正文本

3. **SUIT 代理微调策略**：解决 MQ 训练数据不足的问题：
    - **伪 MQ 生成**：从 Flickr30K + COCO 的图文对出发，用 GPT-3.5 将 caption 拆分为"篡改 caption" + "修正文本"，原图 + 修正文本 = 伪 MQ
    - **代理微调**：在伪 MQ → 篡改 caption 的任务上微调 LLaVA (LoRA, rank 32, alpha 64)
    - **迁移**：微调后的 MLLM 直接用于 ICQ-Highlight 评估
    - 89,420 条训练数据，LR $2 \times 10^{-4}$

### 损失函数 / 训练策略

- SUIT 使用 next-token prediction loss + LoRA PEFT
- 适配后直接使用各 backbone 预训练检查点，无需修改 backbone
- 评估 12 个 backbone：9 个专用模型 (Moment-DETR, QD-DETR 等) + 3 个 LLM-based (SeViLA, TimeChat, VTimeLLM)

## 实验关键数据

### 主实验

| 方法 | 模型 | R1@0.5 (realistic) | R1@0.7 (realistic) | 说明 |
|------|------|---------------------|---------------------|------|
| VQ-Enc | CG-DETR | 24.74 | 14.23 | 仅参考图像 |
| MQ-Cap | TR-DETR | **56.94** | **41.99** | 训练免方法最优 |
| MQ-Cap | CG-DETR | 56.72 | 41.79 | 第二 |
| MQ-Sum | TR-DETR | 52.87 | 36.77 | 不如 MQ-Cap |
| MQ-Sum+SUIT | TR-DETR | **57.39** | **42.64** | 整体最优 |
| MQ-Sum+SUIT | CG-DETR | 55.47 | 40.17 | |
| MQ-Cap | SeViLA | 26.83 | 16.83 | LLM模型表现差 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 无修正文本 vs 有 | 下降 2.8%-14% | 修正文本帮助精确定位 |
| scribble vs realistic | 差距 <3% | 即使极简涂鸦也有效 |
| 合成图 vs 检索真图 | 性能接近 | 生成伪影不影响结论 |
| MQ-Cap vs MQ-Sum | MQ-Cap +3.6% avg | Captioning 更稳定 |
| MQ-Sum vs MQ-Sum+SUIT | SUIT +4.3%-9.7% | 微调大幅提升且更稳定 |
| t-SNE 可视化 | SUIT输出分布更接近NLQ | 解释了 SUIT 为何有效 |

### 关键发现

- **MQ 可有效定位视频事件**：各适配方法在不同风格下表现一致，证明 MQ 的可行性
- **MQ-Cap > MQ-Sum > VQ-Enc**：分步 caption + 修正 比一步摘要更可控；纯视觉编码最差
- **SUIT 是最佳策略**：非marginal 提升(4.3%-9.7%)，且性能更稳定（标准差更小）
- **涂鸦图像也有效**：scribble 风格性能仅略低于 realistic/cinematic，展现了极简视觉查询的潜力
- **专用模型 >> LLM-based 模型**：SeViLA/TimeChat/VTimeLLM 在所有适配方法下都远弱于 TR-DETR/CG-DETR/UVCOM
- **不同 backbone 的排名在各适配方法间一致**：说明 backbone 能力是决定性因素
- MQ 与 NLQ 的性能差距仍然显著，多模态查询语义在跨模态转换中存在损耗

## 亮点与洞察

- **开创性地定义了"多模态查询视频事件定位"任务**，填补了 NLQ-only 的研究空白
- **4 种参考图像风格的设计**（涂鸦→写实渐变）覆盖了从最简到最丰富的真实场景
- **SUIT 的伪 MQ 生成管道**巧妙利用已有图文数据，避免了昂贵的 MQ 标注
- t-SNE 可视化直观展示了 SUIT 如何缓解分布偏移
- 大规模系统基准（12 模型 × 4 适配方法 × 4 风格）为后续研究提供了完整参考

## 局限与展望

- ICQ-Highlight 基于 QVHighlights 验证集，规模有限，且参考图像为合成而非真实用户输入
- 修正文本的类型分布可能不均匀，某些类型（如"关系"）样本较少
- 所有适配方法均为 pipeline 式，端到端的 MQ 模型设计尚未探索
- LLM-based 模型表现差，可能是因为它们在 NLQ benchmark 上本身就不佳，而非 MQ 的限制
- 未考虑用户实时交互场景（如边看边搜索）

## 相关工作与启发

- 与 composed image retrieval (CIR) 任务类似，但 CIR 是实例级匹配而 ICQ 是时序密集处理，复杂度更高
- SUIT 的代理微调思路类似知识蒸馏，可推广到其他缺少训练数据的适配场景
- MQ-Cap 的分步策略（先 caption 再修改）在其他 VLM 适配任务中也具有参考价值
- scribble 有效的发现启发了零样本/少样本视频搜索的可能性

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 全新任务定义 + 系统基准 + 创新适配策略
- 实验充分度: ⭐⭐⭐⭐⭐ 12 模型 × 4 方法 × 4 风格，消融详尽
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表丰富，略显冗长
- 价值: ⭐⭐⭐⭐⭐ 开创新方向，受众广泛，数据集有长期价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] BlinkTrack: Feature Tracking over 80 FPS via Events and Images](../../ICCV2025/video_understanding/blinktrack_feature_tracking_over_80_fps_via_events_and_images.md)
- [\[CVPR 2025\] VideoGEM: Training-Free Action Grounding in Videos](videogem_training-free_action_grounding_in_videos.md)
- [\[CVPR 2025\] DPU: Dynamic Prototype Updating for Multimodal Out-of-Distribution Detection](dpu_dynamic_prototype_updating_for_multimodal_out-of-distribution_detection.md)
- [\[CVPR 2025\] DivPrune: Diversity-Based Visual Token Pruning for Large Multimodal Models](divprune_diversity-based_visual_token_pruning_for_large_multimodal_models.md)
- [\[CVPR 2025\] ReWind: Understanding Long Videos with Instructed Learnable Memory](rewind_understanding_long_videos_with_instructed_learnable_memory.md)

</div>

<!-- RELATED:END -->
