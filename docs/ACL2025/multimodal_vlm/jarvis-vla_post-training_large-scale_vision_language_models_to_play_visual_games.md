---
title: >-
  [论文解读] JARVIS-VLA: Post-Training Large-Scale Vision Language Models to Play Visual Games
description: >-
  [ACL 2025][多模态VLM][视觉语言动作模型] 提出ActVLP训练范式，在动作模仿学习之前增加视觉语言后训练阶段（世界知识、视觉对齐、空间定位），构建首个能在Minecraft中执行1000+原子任务的VLA模型JARVIS-VLA，相比最佳基线提升40%。 问题背景 视觉语言动作（VLA）模型是将预训练VLM用…
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "视觉语言动作模型"
  - "后训练"
  - "Minecraft"
  - "模仿学习"
  - "决策"
  - "空间定位"
---

# JARVIS-VLA: Post-Training Large-Scale Vision Language Models to Play Visual Games

**会议**: ACL 2025  
**arXiv**: [2503.16365](https://arxiv.org/abs/2503.16365)  
**作者**: Muyao Li, Zihao Wang, Kaichen He (Peking University), Xiaojian Ma (BIGAI), Yitao Liang  
**代码**: [开源](https://craftjarvis.github.io/JarvisVLA)  
**领域**: 多模态VLM  
**关键词**: 视觉语言动作模型, 后训练, Minecraft, 模仿学习, 决策, 空间定位  

## 一句话总结

提出ActVLP训练范式，在动作模仿学习之前增加视觉语言后训练阶段（世界知识、视觉对齐、空间定位），构建首个能在Minecraft中执行1000+原子任务的VLA模型JARVIS-VLA，相比最佳基线提升40%。

## 研究背景与动机

### 问题背景
视觉语言动作（VLA）模型是将预训练VLM用于决策的新兴方向，通过在大规模轨迹数据上进行模仿学习来生成动作。VPT等先驱工作在Minecraft中展示了从大规模YouTube视频预训练后用IL微调的路线，成功完成ObtainDiamond等挑战。然而，现有VLA方法的核心问题在于**仅聚焦于动作后训练**，忽视了对基础模型本身的增强。

### 已有工作的不足
- **纯模仿学习的局限**：从动作标注轨迹中学习世界知识本身就很困难，且大规模动作标注数据集稀缺
- **OpenVLA等方法**：直接在预训练VLM上做动作微调，未利用环境相关的视觉-语言数据增强基础能力
- **层级式智能体**（Voyager、JARVIS-1）：依赖VLM的零/少样本推理做规划，仍需额外的底层策略来执行动作
- **泛化困难**：轨迹数据中观测与行为的复杂耦合使得预训练范式难以跨任务和环境迁移

### 核心动机
在动作学习之前先通过**非轨迹的视觉语言任务**增强VLM的环境理解、视觉识别和空间定位能力，使其成为更好的决策基础模型。

## 方法详解

### 模型架构
JARVIS-VLA采用类LLaVA架构，包含三个核心组件：
- **视觉编码器**：ViT将原始图像（644×364分辨率）转换为patch序列
- **图像投影模块**：两层MLP将图像patch嵌入对齐到词嵌入空间
- **语言模型**：自回归Transformer作为核心推理和决策引擎

与OpenVLA不同，JARVIS-VLA采用**非马尔可夫架构**，在提示中保留历史观测图像序列以保持时序上下文。动作解码方面，离散动作合并为统一类别，连续动作（如鼠标移动）通过$\mu$-law编码离散化为21个bin，总计分配51个特殊token（22个鼠标控制 + 29个键盘输入），复用分词器中最低频token而非修改原始架构。

### ActVLP三阶段训练流程

**Stage I: 语言模型后训练**

冻结视觉相关组件（ViT和视觉适配器），仅用约277K条Minecraft世界知识文本数据对语言Transformer做SFT，增强模型对决策环境的文本理解能力。

**Stage II: 视觉编码器与语言模型联合后训练**

完全解冻VLM，使用图像描述、视觉问答（VQA）和空间定位数据集进行微调。其中视觉语言对齐数据包含35K关键帧，空间定位数据超过404K条。两个阶段均使用next-token预测的SFT损失：

$$\mathcal{L}_{\text{SFT}} = -\sum_{i=1} \log \mathcal{P}_{\theta}(x_i \mid x_v, x_{\text{ins}}, x_{1:i-1})$$

其中$x_v$为视觉token，$x_{\text{ins}}$为指令，$x$为答案。

**Stage III: 动作后训练**

冻结视觉模块，修改语言分词器加入动作token，在轨迹数据上通过模仿学习微调语言Transformer。模型学习将文本指令和视觉观测映射为动作块（action chunking）：

$$\mathcal{L}_{\text{IL}} = -\sum_{t=1} \log \pi_{\theta}(a_{t:t+\tau} \mid o_t, x_{\text{ins}})$$

其中$\pi_{\theta}$为参数化策略，$a_{t:t+\tau}$为连续$\tau$步的专家动作。动作分块技术提升了动作的时序一致性和训练效率。

### 数据集构成
- **世界知识**：277K条Minecraft相关文本（Stage I）
- **视觉语言对齐**：35K关键帧 + 高级VLM生成的描述和QA对（Stage II）
- **空间定位**：404K+条目标定位数据（Stage II）
- **轨迹数据**：7.4M帧Minecraft游戏数据，包含人类操作、YouTube视频、已有智能体rollout和合成的GUI操作数据（Stage III）

## 实验关键数据

### 实验1：MCU Benchmark主结果

| 模型 | 参数量 | Mine Blocks | Kill Entities | Craft Items | Smelt Items |
|------|--------|-------------|---------------|-------------|-------------|
| VPT-BC | 248M | 0.33 | 0.44 | 0.41 | 0.05 |
| VPT-RL | 248M | 0.25 | 0.28 | 0.55 | 0.20 |
| STEVE-1 | 248M | 0.54 | 0.38 | 0.57 | 0.33 |
| GROOT | 248M | 0.67 | 0.52 | 0.40 | 0.30 |
| MineDreamer | 7B | 0.55 | 0.39 | 0.42 | 0.30 |
| Qwen2-VL (raw) | 7B | 0.79 | 0.84 | 0.60 | 0.07 |
| Qwen2-VL (IL) | 7B | 0.75 | 0.86 | 0.65 | 0.29 |
| **JARVIS-VLA-Qwen2** | **7B** | **0.88** | **0.95** | **0.77** | **0.70** |

JARVIS-VLA在所有四类任务上均达到最优。特别是Craft和Smelt任务（需精确GUI操作），成功率分别达到0.77和0.70，是基线模型的两倍以上。即使是未经后训练的原始Qwen2-VL微调版本（raw）也超过了多个248M参数的专用基线。

### 实验2：训练范式消融

| 模型 | Craft Diamond Sword | Craft Ladder | Cook Beef | Smelt Iron Ingot |
|------|---------------------|-------------|-----------|-----------------|
| Qwen2-VL (raw) | 0.53 | 0.40 | 0.03 | 0.10 |
| Qwen2-VL (one-stage) | 0.10 | 0.40 | 0.07 | 0.13 |
| ActVLP-Qwen2-VL | **0.83** | **0.63** | **0.77** | **0.70** |

将视觉语言后训练与动作学习分离（而非合并为单阶段训练）带来显著提升。one-stage方法甚至劣于raw基线，说明混合训练数据会产生负迁移。ActVLP相比one-stage在Smelt Iron Ingot上提升57个百分点。

## 关键发现

- **空间定位贡献最大**：消融实验显示，三类非轨迹数据中，空间定位对下游决策任务提升最为显著，因为精确的目标定位是动作执行的关键前提
- **缩放定律存在**：增大非轨迹视觉语言后训练数据量，下游任务成功率与后训练评估loss呈线性正相关；增大下游轨迹数据量也能提升成功率，但需loss降到0.30以下才出现非零成功率
- **高分辨率至关重要**：644×364分辨率（远高于VPT的128×128）是Craft/Smelt GUI任务大幅领先的关键因素之一
- **仅用21%轨迹数据即超越IL基线**：JARVIS-VLA-Qwen2-VL仅用Qwen2-VL(IL)五分之一的轨迹数据，性能却高出15%以上
- **不同VLM骨干均有效**：在LLaVA-Next和Qwen2-VL两种骨干上均验证了ActVLP范式的有效性

## 亮点

- **范式创新**：首次系统提出在VLA动作训练前增加视觉语言后训练的三阶段范式，而非简单的"预训练VLM → 动作微调"
- **非轨迹数据的价值**：揭示了非轨迹视觉语言任务（VQA、描述、空间定位）对决策能力提升的显著贡献，且存在类似LLM的缩放效应
- **强实验对比**：分离vs混合训练的消融直接证明了阶段式后训练的必要性，one-stage反而性能下降
- **实用架构设计**：复用最低频token表示动作、非马尔可夫多帧观测、动作分块等工程决策合理且无需修改原始VLM架构
- **全面开源**：代码、模型、数据集均开源，促进后续研究

## 局限与展望

- **推理速度受限**：7B参数量的VLA模型推理吞吐量远不及人类玩家的实时操作速率（40Hz+），论文建议未来集成MoE改善
- **仍不及顶尖人类**：尽管达到SOTA，成功率仍低于90%以上的高水平人类玩家
- **仅限Minecraft**：ActVLP范式虽可推广，但本文实验仅在Minecraft中验证，对机器人操作等真实场景的迁移性未知
- **数据构建成本高**：空间定位和视觉语言对齐数据集需要利用SAM2等高级工具和多个VLM来生成，构建流程较重
- **无强化学习阶段**：仅使用模仿学习，缺乏RL对次优轨迹的纠正能力
- **非轨迹数据的$t_{\mathrm{mix}}^2$因子**：空间定位数据的构造依赖于SAM2和特定场景标注工具，场景迁移时需重建

## 与相关工作的对比

- **VPT (Baker et al., 2022)**：预训练+IL路线的先驱，248M参数，依赖大规模YouTube视频，本文在其基础上用VLM替代并增加视觉语言后训练
- **OpenVLA (Kim et al., 2024)**：直接对预训练VLM做动作微调的VLA方法，忽略了基础能力增强；本文通过实验证明了后训练的必要性
- **RT-2 (Brohan et al., 2023)**：提出web数据co-training提升VLA泛化性，本文的Stage I/II可视为对这一思路更结构化的实现
- **STEVE-1 (Lifshitz et al., 2024)**：结合VPT和MineCLIP的文本条件策略，JARVIS-VLA在所有任务上均大幅超越
- **MineDreamer (Zhou et al., 2024)**：利用VLM预测未来帧指导STEVE-1策略，属于层级架构，性能逊于端到端VLA
- **OmniJARVIS (Wang et al., 2024d)**：使用行为tokenizer建模轨迹，仍需额外策略做动作落地；JARVIS-VLA直接端到端生成动作
- **GROOT (Cai et al., 2024c)**：视频提示任务指定，248M参数，在Mine/Kill上有竞争力但Craft/Smelt较弱

## 评分

- 新颖性: ⭐⭐⭐⭐ — ActVLP范式是对VLA训练流程的有意义创新，但三阶段训练本身不算突破性
- 实验充分度: ⭐⭐⭐⭐ — 消融全面（训练范式、数据类型、缩放定律、骨干选择），但仅限Minecraft一个环境
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，图表信息量大，训练流程讲解透彻
- 价值: ⭐⭐⭐⭐ — 揭示了非轨迹后训练对VLA的重要性，对后续VLA研究有启发；开源贡献突出

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Post-pre-training for Modality Alignment in Vision-Language Foundation Models](../../CVPR2025/multimodal_vlm/post-pre-training_for_modality_alignment_in_vision-language_foundation_models.md)
- [\[ACL 2025\] Activating Distributed Visual Region within LLMs for Efficient and Effective Vision-Language Training and Inference](activating_distributed_visual_region_within_llms_for_efficient_and_effective_vis.md)
- [\[ACL 2025\] Speaking Beyond Language: A Large-Scale Multimodal Dataset for Learning Nonverbal Cues from Video-Grounded Dialogues](speaking_beyond_language.md)
- [\[ICCV 2025\] CoA-VLA: Improving Vision-Language-Action Models via Visual-Textual Chain-of-Affordance](../../ICCV2025/multimodal_vlm/coavla_improving_visionlanguageaction_models_via_visualtext.md)
- [\[ACL 2025\] Don't Miss the Forest for the Trees: Attentional Vision Calibration for Large Vision Language Models](dont_miss_the_forest_for_the_trees_attentional_vision_calibration_for_large_visi.md)

</div>

<!-- RELATED:END -->
