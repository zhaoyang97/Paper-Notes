---
title: >-
  [论文解读] CoMemo: LVLMs Need Image Context with Image Memory
description: >-
  [ICML2025][多模态VLM][LVLM] 提出CoMemo双路径架构——Context路径将图像token拼入文本做自回归、Memory路径用交叉注意力做图像持久记忆，结合RoPE-DHR位置编码保持2D空间感知和缓解远程衰减，通过三阶段训练策略平衡双路径，在同等设置下全面超越LVLM-S和LVLM-X。
tags:
  - "ICML2025"
  - "多模态VLM"
  - "LVLM"
  - "Dual-Path Architecture"
  - "注意力机制"
  - "RoPE"
  - "Dynamic High Resolution"
  - "Lost in the Middle"
---

# CoMemo: LVLMs Need Image Context with Image Memory

**会议**: ICML2025  
**arXiv**: [2506.06279](https://arxiv.org/abs/2506.06279)  
**代码**: [项目页](https://lalbj.github.io/projects/CoMemo/)  
**领域**: 多模态VLM  
**关键词**: LVLM, Dual-Path Architecture, Cross-Attention, RoPE, Dynamic High Resolution, Lost in the Middle

## 一句话总结
提出CoMemo双路径架构——Context路径将图像token拼入文本做自回归、Memory路径用交叉注意力做图像持久记忆，结合RoPE-DHR位置编码保持2D空间感知和缓解远程衰减，通过三阶段训练策略平衡双路径，在同等设置下全面超越LVLM-S和LVLM-X。

## 研究背景与动机

**领域现状**：大视觉语言模型(LVLM)有两种主流架构——LVLM-S (LLaVA式，视觉token拼入文本序列做自回归) 和 LVLM-X (Flamingo式，通过交叉注意力注入视觉信息)。LVLM-S兼容性好但性能受限于LLM的注意力机制缺陷；LVLM-X灵活但在相同设置下不如LVLM-S。

**现有痛点**：LVLM-S继承了LLM的两个致命弱点：(1) **"Lost in the middle"**——因果自注意力导致注意力集中在序列首尾，中间位置的视觉内容被系统性忽略，上下文越长越严重；(2) **动态高分辨率(DHR)的位置编码灾难**——1D递增的RoPE使得DHR下图像patch token占据大量位置ID，引发远程衰减且破坏图像2D空间结构。

**核心矛盾**：LVLM-S的自回归处理天然会"遗忘"中间视觉信息（这是因果注意力训练范式的固有属性），但交叉注意力路径（LVLM-X）单独使用又性能不佳。简单合并两者（如NVLM-H）也不work——因为模型会过度依赖某一路径。

**切入角度**：作者通过梯度和注意力权重可视化发现，训练中大部分梯度反传到邻近token和attention sink，中间token被系统性忽视。这启发了"额外加一条不受上下文位置影响的视觉处理路径"的思路。

**核心 idea**：在保留LLaVA式自回归处理（Context路径）的同时，增加一条交叉注意力路径（Memory路径）作为图像的"不会遗忘"的持久记忆，通过三阶段训练防止路径失衡。

## 方法详解

### 整体框架
CoMemo基于InternVL2架构，使用InternLM-1.8B做LLM backbone、InternViT-300M做图像编码。输入图像经encoder和projector后产生visual tokens，同时送入两条路径：Context路径将visual tokens与text tokens拼接做标准自回归处理；Memory路径将visual tokens缓存为KV，在mixin层中通过交叉注意力让解码token查询视觉信息。两条路径共享相同的encoder和projector。

### 关键设计

1. **双路径架构：Context + Memory**:

    - 功能：确保视觉信息在长上下文和长生成中不被遗忘
    - 核心思路：**Context路径**是主处理流，图像token和文本token拼接输入LLM做标准自回归，提供图像上下文；**Memory路径**是辅助流，mixin层按1:4比例与标准Transformer层交错插入。每个mixin层包含：(a) 门控交叉注意力——用可学习的$\text{attn\_gate}$调制：$h_s \leftarrow h_s + \tanh(\text{attn\_gate}) \odot \text{cross\_attn}(q=h_s, kv=h_i)$；(b) 门控前馈网络——$h_s \leftarrow h_s + \tanh(\text{ffw\_gate}) \odot \text{ffw}(h_s)$。交叉注意力基于文本查询检索视觉信息，绕过因果自注意力的"中间遗忘"问题
    - 设计动机：交叉注意力天然避免了因果注意力的位置偏好问题——它按内容相关性而非位置来检索视觉信息。解码时只需一步计算，无需维护视觉KV缓存

2. **RoPE-DHR位置编码**:

    - 功能：在动态高分辨率(DHR)场景下保持图像2D空间关系、缓解远程衰减
    - 核心思路：标准InternVL2每张图使用256个token，激活DHR后可扩展到1792个token（7倍增加），这导致严重的位置编码稀疏化和远程衰减。RoPE-DHR的策略是：先对缩略图(thumbnail)使用标准顺序位置编码生成基准位置ID；然后将高分辨率图块(tile)的每个patch按其2D坐标映射到对应的缩略图patch索引：$i_{thumb} = (\lfloor x_{tile} \times \frac{W_{tile}}{W_{orig}} \rfloor + wb_{tile}, \lfloor y_{tile} \times \frac{H_{orig}}{H_{thumb}} \rfloor + hb_{tile})$。这样高分辨率patch继承了缩略图的紧凑位置空间
    - 设计动机：实现两个目标——(1) 长度压缩：从全局视角防止DHR的稀疏位置编码；(2) 几何保持：tile patch从缩略图锚点继承位置上下文，保留2D空间关系

3. **三阶段训练策略**:

    - 功能：平衡Context和Memory双路径，防止模型过度依赖某一通道
    - 核心思路：**阶段1**——训练projector和mixin层参数（含gate），学习表示对齐和双路径平衡；**阶段2**——冻结gate参数继续训练，防止在预训练后期过度依赖交叉注意力路径；**阶段3**——全参数微调，转向指令跟随。实验发现预训练步数对平衡至关重要：步数不足导致对齐不够，步数过多导致Memory路径主导（因为全参数更新的Memory模块学得比部分更新的projector快）
    - 设计动机：直接合并双路径（如NVLM-H的做法）会导致模型一边倒地依赖某一路径。gate值监控显示，用合适步数训练后冻结gate再继续预训练是达到平衡的关键

### 损失函数 / 训练策略
预训练阶段使用标准next-token prediction损失，仅更新projector和Memory架构参数。微调阶段全参数训练，使用InternVL-2相同的训练数据。DHR信息分配到两条路径（DHR-B策略），实验表明单侧分配会导致严重路径偏好。

## 实验关键数据

### 三种架构公平对比（2B模型，相同数据）

| 任务类别 | LVLM-X | LVLM-S (InternVL2) | CoMemo | 相对提升 |
|---------|--------|-------------------|--------|---------|
| Caption (COCO/Flickr/NoCaps) | 84.9/68.9/62.9 | 79.1/65.4/60.0 | **98.6/78.5/78.8** | +17.2% |
| Long-Gen (LLaVABench/MMDU) | 50.8/31.5 | 62.9/28.7 | **66.9/38.7** | +7.0% |
| Long-Context (MMT/MMNIAH) | -/- | 50.2/27.0 | **51.3/34.2** | +5.6% |
| Multi-Image (BLINK/Mantis) | 50.3/63.1 | 38.2/48.3 | **43.5/50.6** | 显著 |
| Math (MathVista/MathVision) | 44.2/15.8 | 48.0/16.4 | **50.0/17.0** | — |

### DHR分配策略消融

| DHR分配 | Caption | VQA | OCR | 路径偏好 |
|---------|---------|-----|-----|---------|
| DHR-S (仅Context路径) | 中 | 低 | 高 | 偏向Context |
| DHR-X (仅Memory路径) | 低 | 中 | 低 | 偏向Memory |
| **DHR-B (双路径)** | **高** | **高** | **高** | **平衡** |

### 关键发现
- 双路径的关键不是简单合并而是三阶段训练——NVLM-H式的直接合并效果不佳，但通过适当步数的预训练+gate冻结+全参微调，CoMemo实现了路径平衡
- RoPE-DHR在OCR和高分辨率任务上改善最显著，因为这些任务最依赖2D空间感知
- 在同等数据/模型/训练设置下，CoMemo在7类基准的所有大类上一致优于LVLM-S和LVLM-X，证明了架构设计本身的优越性
- Memory路径在解码时只需一步交叉注意力计算，无需维护视觉KV缓存，避免了长序列下KV缓存膨胀问题

## 亮点与洞察
- "Context + Memory"的双路径命名直观且有认知科学启发——Context提供即时视觉上下文，Memory提供持久视觉记忆，两者在人类认知中也是互补的
- 三阶段训练策略是使双路径真正Work的关键创新。通过gate值监控路径偏好是一个实用的调试手段，此方法论可迁移到任何多路径架构的训练平衡问题
- RoPE-DHR通过缩略图索引映射巧妙地将"长度压缩"和"几何保持"统一在一个简洁框架里
- Section 2的分析驱动式设计方法值得学习：先通过梯度/注意力分析诊断问题根因，再针对性地设计架构解决方案
- 在解码阶段Memory路径只需与缓存的视觉表示做一次交叉注意力，避免了随序列长度线性增长的KV缓存问题
- 门控机制(tanh gate)的设计使得Memory路径的影响可以在训练中被自动调节——从接近零到正常贡献，提供了平滑的控制

## 局限性
- 仅在2B模型上验证，大规模（7B/70B）效果未知——双路径的平衡可能随规模变化
- 交叉注意力mixin层增加了参数量和计算开销，具体量级未详细报告
- 三阶段训练增加了训练流程的复杂度，预训练步数的选择需要仔细调优
- 对视频/极长视觉序列的适配未探索——Memory路径是否能扩展为外部可写记忆？
- RoPE-DHR的缩略图映射可能在动态tile数较多时产生位置冲突，具体影响未充分讨论
- 只使用了InternVL-2的训练数据，在更大规模数据上的表现有待验证

## 相关工作与启发
- **vs NVLM-H (Dai et al. 2024)**: NVLM-H DHR只分配给交叉注意力路径导致偏好失衡，CoMemo通过DHR-B双路径分配+三阶段训练彻底解决了这个问题
- **vs mPLUG-Owl3**: 也使用交叉注意力mixin层，但缺乏双路径平衡的训练策略，且无RoPE-DHR的位置编码改进
- 启发：可将Memory路径扩展为RAG式的可写外部记忆，在多轮对话中持续更新视觉记忆
- 启发：三阶段训练中gate冻结的时机控制可推广为通用的多模块训练稳定化工具

## 评分
- 新颖性: ⭐⭐⭐⭐ 双路径+三阶段训练+RoPE-DHR三个创新点
- 实验充分度: ⭐⭐⭐⭐⭐ 7类基准全面评测，公平控制变量的三架构对比极为充分
- 写作质量: ⭐⭐⭐⭐ 设计思考(Section 2)的分析驱动式写作值得学习
- 价值: ⭐⭐⭐⭐ 对LVLM架构设计有明确指导意义
- 总体: ⭐⭐⭐⭐ 双路径视觉处理的开创性工作，三阶段训练策略具有重要方法论价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Oasis: One Image is All You Need for Multimodal Instruction Data Synthesis](../../ICCV2025/multimodal_vlm/oasis_one_image_is_all_you_need_for_multimodal_instruction_data_synthesis.md)
- [\[CVPR 2025\] Cropper: Vision-Language Model for Image Cropping through In-Context Learning](../../CVPR2025/multimodal_vlm/cropper_vision-language_model_for_image_cropping_through_in-context_learning.md)
- [\[ICML 2025\] Towards Rationale-Answer Alignment of LVLMs via Self-Rationale Calibration](towards_rationale-answer_alignment_of_lvlms_via_self-rationale_calibration.md)
- [\[CVPR 2026\] Adapting In-context Generation for Enhanced Composed Image Retrieval](../../CVPR2026/multimodal_vlm/adapting_in-context_generation_for_enhanced_composed_image_retrieval.md)
- [\[CVPR 2025\] VladVA: Discriminative Fine-tuning of LVLMs](../../CVPR2025/multimodal_vlm/vladva_discriminative_fine-tuning_of_lvlms.md)

</div>

<!-- RELATED:END -->
