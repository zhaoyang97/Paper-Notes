---
title: >-
  [论文解读] CTRL-O: Language-Controllable Object-Centric Visual Representation Learning
description: >-
  [CVPR 2025][图像生成][目标中心表示] CTRL-O 将语言可控性引入目标中心表示学习，通过语言嵌入初始化 slot query、解码器语言条件化和控制对比损失，在无 mask 监督下实现语言-物体绑定，COCO 上 FG-ARI 47.5（比 Dinosaur +7.0），同时支持零样本参考表达分割、实例级图像生成和 VQA。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "目标中心表示"
  - "注意力机制"
  - "语言控制"
  - "对比损失"
  - "物体发现"
---

# CTRL-O: Language-Controllable Object-Centric Visual Representation Learning

**会议**: CVPR 2025  
**arXiv**: [2503.21747](https://arxiv.org/abs/2503.21747)  
**代码**: [https://ctrl-o-paper.github.io](https://ctrl-o-paper.github.io)  
**领域**: 图像生成  
**关键词**: 目标中心表示、Slot Attention、语言控制、对比损失、物体发现

## 一句话总结

CTRL-O 将语言可控性引入目标中心表示学习，通过语言嵌入初始化 slot query、解码器语言条件化和控制对比损失，在无 mask 监督下实现语言-物体绑定，COCO 上 FG-ARI 47.5（比 Dinosaur +7.0），同时支持零样本参考表达分割、实例级图像生成和 VQA。

## 研究背景与动机

1. **领域现状**：目标中心表示学习（如 Slot Attention、Dinosaur）将场景分解为独立的物体表示（slots），但 slots 不受控——无法指定哪个 slot 对应哪个物体。
2. **现有痛点**：(1) Slots 的分配是随机的，用户无法通过语言指定感兴趣的物体；(2) 在复杂真实场景中物体发现精度有限；(3) 学到的表示难以直接用于下游任务。
3. **核心矛盾**：目标中心表示需要"发现"物体（无监督），但"控制"物体绑定需要语义理解——如何在不需要 mask 标注的情况下引入语言控制？
4. **本文目标**：用语言描述控制 slot 到物体的绑定，无需 mask 监督。
5. **切入角度**：用预训练 LLM 嵌入初始化 slot query，使 slots 天然倾向于绑定到对应语义的物体上。
6. **核心 idea**：查询初始化（LLM 嵌入 + 位置信息）+ 解码器语言条件化 + 控制对比损失。

## 方法详解

### 整体框架

输入图像 → 冻结 DINOv2 提取特征 → 语言描述通过 LLaMA-3-8B (LLM2Vec) 编码 + 质心坐标 → 初始化 slot queries → Slot Attention 迭代分配 → 解码器用 slot+控制 query 条件化重建 → 控制对比损失约束 slot-语言对齐。

### 关键设计

1. **语言查询初始化**

    - 功能：让 slot 从一开始就倾向于绑定到语言描述的物体
    - 核心思路：将 LLaMA-3-8B (LLM2Vec) 的语言嵌入与质心坐标拼接作为 slot 的初始 query。动态类-prompt 映射：k-means 将 C 个类聚为 M 个 prompt（每 epoch 更新）
    - 设计动机：随机初始化的 slot 绑定到哪个物体是不可控的，语言初始化提供了语义"锚点"

2. **控制对比损失**

    - 功能：强制 slot 表示对齐到对应的语言嵌入
    - 核心思路：$\mathcal{L}_{CC}^l = -\sum_i \log\frac{\exp(z_i^{emb} \cdot l_i / \tau)}{\sum_t \exp(z_i^{emb} \cdot l_t / \tau)}$，其中 $z_i = \sum_k a_{ik} h_k$ 是按 slot attention 权重聚合的 DINO 特征，$\tau=0.1$
    - 设计动机：仅靠初始化不能保证 slot 在注意力迭代后仍保持正确绑定

3. **解码器语言条件化**

    - 功能：在重建阶段注入语言信息以增强物体-语言关联
    - 核心思路：将 slot 与控制 query 拼接后送入 MLP 解码器
    - 设计动机：解码器接收语言条件后可以学到更有语义意义的重建

### 损失函数 / 训练策略

重建损失 + 控制对比损失。冻结 DINOv2 ViT 骨干。COCO+VG 300K 步，batch 128。

## 实验关键数据

### 主实验

| 方法 | FG-ARI↑ | mBO↑ | Binding Hits↑ |
|------|---------|------|---------------|
| Dinosaur | 40.5 | 27.7 | - |
| **CTRL-O** | **47.5** | 27.2 | **61.3%** |

| 任务 | CTRL-O | 最佳基线 |
|------|--------|---------|
| RefCOCO mIoU (零样本) | 28.2 | Shatter&Gather 21.8 |
| 图像生成 FID (COCO) | 25.20 | Stable LSD 26.20 |
| VQAv2 准确率 | 60.25% | CLIP 58.64% |

### 消融实验

| 配置 | Binding Hits | 说明 |
|------|-------------|------|
| w/o 语言初始化 | ~40% | 丢失语义锚点 |
| w/o 对比损失 | ~48% | 对齐不持久 |
| w/ GT masks (上界) | 71.2% | 监督的天花板 |
| **Full CTRL-O** | **61.3%** | 无监督接近上界 |

### 关键发现

- FG-ARI +7.0 提升主要来自语言引导使 slot 更准确地分割边界
- 零样本参考表达分割 28.2 mIoU 比非语言方法高 30%+
- 即使有 GT mask 监督，Binding Hits 也仅 71.2%，说明物体绑定本身是困难问题

## 亮点与洞察

- **可控性是目标中心表示的关键缺失环节**：CTRL-O 填补了这个空白
- **无 mask 监督下 61.3% 绑定命中率**：不需要分割标注就能实现合理的语言-物体对齐
- **统一框架支持多任务**：物体发现、参考分割、图像生成和 VQA

## 局限与展望

- 同类多实例需要额外的位置消歧（质心坐标）
- MLP 解码器的 mBO（27.2）略低于 Dinosaur（27.7）
- VQA 60.25% 仍远低于大语言模型方案（>80%）
- 扩散生成存在物体变形/重复的失败案例

## 相关工作与启发

- **vs Dinosaur**: 无语言控制，FG-ARI 40.5。CTRL-O 通过语言锚点提升到 47.5
- **vs CLIP**: 全图级别对比学习，不做物体级分解。CTRL-O 在物体级做语言对齐

## 评分

- 新颖性: ⭐⭐⭐⭐ 将语言控制引入目标中心学习是自然但重要的扩展
- 实验充分度: ⭐⭐⭐⭐ 物体发现+分割+生成+VQA多角度验证
- 写作质量: ⭐⭐⭐⭐ 清晰
- 价值: ⭐⭐⭐⭐ 统一多任务的可控物体表示有长期研究价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] GLASS: Guided Latent Slot Diffusion for Object-Centric Learning](glass_guided_latent_slot_diffusion_for_object-centric_learning.md)
- [\[ICCV 2025\] GenFlowRL: Shaping Rewards with Generative Object-Centric Flow in Visual Reinforcement Learning](../../ICCV2025/image_generation/genflowrl_shaping_rewards_with_generative_object-centric_flow_in_visual_reinforc.md)
- [\[CVPR 2025\] Learning Flow Fields in Attention for Controllable Person Image Generation](learning_flow_fields_in_attention_for_controllable_person_image_generation.md)
- [\[CVPR 2025\] ORIDa: Object-Centric Real-World Image Composition Dataset](orida_object-centric_real-world_image_composition_dataset.md)
- [\[CVPR 2025\] Visual Lexicon: Rich Image Features in Language Space](visual_lexicon_rich_image_features_in_language_space.md)

</div>

<!-- RELATED:END -->
