---
title: >-
  [论文解读] TAViS: Text-bridged Audio-Visual Segmentation with Foundation Models
description: >-
  [ICCV 2025][语义分割][音频-视觉分割] 提出 TAViS，一种文本桥接的音频-视觉分割框架，通过耦合 ImageBind 的跨模态对齐能力与 SAM2 的精确分割能力，引入文本桥接的混合提示机制和对齐监督策略，在单源、多源、语义及零样本分割场景上均取得 SOTA 性能。 音频-视觉分割（AVS）旨在根据音频信号…
tags:
  - "ICCV 2025"
  - "语义分割"
  - "音频-视觉分割"
  - "基础模型"
  - "SAM2"
  - "ImageBind"
  - "文本桥接"
  - "跨模态对齐"
  - "零样本分割"
---

# TAViS: Text-bridged Audio-Visual Segmentation with Foundation Models

**会议**: ICCV 2025  
**arXiv**: [2506.11436](https://arxiv.org/abs/2506.11436)  
**代码**: 未公开  
**领域**: 图像分割  
**关键词**: 音频-视觉分割, 基础模型, SAM2, ImageBind, 文本桥接, 跨模态对齐, 零样本分割

## 一句话总结

提出 TAViS，一种文本桥接的音频-视觉分割框架，通过耦合 ImageBind 的跨模态对齐能力与 SAM2 的精确分割能力，引入文本桥接的混合提示机制和对齐监督策略，在单源、多源、语义及零样本分割场景上均取得 SOTA 性能。

## 研究背景与动机

音频-视觉分割（AVS）旨在根据音频信号生成场景中发声物体的像素级分割图，核心挑战是有效对齐音频和视觉两种模态。

现有方法的局限性可归类为三种：

**融合型方法**（AVSBench、CATR、AVSegFormer）：用交叉注意力建立音视觉关系，但受限于小规模训练数据，无法充分捕获模态间复杂关系

**单模态基础模型方法**（AV-SAM、SAMA-AVS、BAVS）：利用 SAM/Semantic-SAM 等视觉基础模型或 BEATS 等音频基础模型，但仅依赖单模态先验知识，无法解决跨模态对齐问题

**拼装式基础模型组合**：虽然有少数工作组合了 SAM 和 ImageBind，但两个模型独立运行无交互，是"离线"组合

**核心挑战**：
- **特征空间不匹配**：SAM2 和 ImageBind 处于不同的特征空间，直接迁移知识困难
- **监督信号不足**：仅用分割损失监督隐式暗示了对齐需求，但未显式引导模型学习有意义的音视觉关联
- **模态内噪声**：音频含非语义信息（音色、语调），图像含多样背景和外观信息，两者都有显著类内多样性，直接音视觉对齐效果差

**核心洞察**：文本作为"桥梁"可以简洁地表达高层原型信息，提取并对齐音视觉模态间的共享语义概念。

## 方法详解

### 整体框架

TAViS 集成两个基础模型：
- **SAM2**：图像编码器 $F_E$ + 记忆注意力模块 $F_M$ + 掩码解码器 $F_D$，负责精确分割
- **ImageBind**：图像编码器 $E_I$ + 文本编码器 $E_T$ + 音频编码器 $E_A$，负责跨模态对齐

冻结 ImageBind 全部参数和 SAM2 图像编码器，微调 SAM2 的记忆注意力和掩码解码器。

### 关键设计一：ImageBind 引导的查询分解（IBQD）

之前的 SAM-based AVS 方法将整个音频特征作为单一提示，但多声源的混合信息与 SAM2 要求的目标级查询冲突。

IBQD 将音频特征分解为目标级查询，同时保留 ImageBind 的对齐特征空间：

引入可学习查询 $t_W \in \mathbb{R}^{N \times C}$，通过多头交叉注意力从音频 trunk 特征 $f_a$ 中分解：

$$t_W = \text{Softmax}((W_q t_W)(W_k f_a^T))(W_v f_a) + t_W$$

生成目标级偏置并加到音频 cls token $t_a$ 上：

$$t'_a = t_a + \text{Linear}(t_W)$$

再通过一次交叉注意力更新 $t'_a$。关键点：分解查询是 cls token 加偏置的形式，保留了 $t_a$ 的对齐特征空间。

### 关键设计二：文本桥接的混合提示（Text-bridged Hybrid Prompting）

**稀疏提示**：音频-文本双提示
- **伪文本提示** $p^t$：将分解后的音频查询 $t'_a$ 和全局 cls token $t_a$ 通过 MLP 拼接后送入 ImageBind 文本编码器 $E_T$，生成高层类别原型信息
- **音频提示** $p^a$：通过另一个 MLP 拼接 $t_W$ 和 $t_a$，保留音频特有的细节特征
- 最终稀疏提示：$p = \text{MLP}([p^a; p^t])$

伪文本生成的监督：

$$\mathcal{L}_{a2pt} = \text{MSE}(E_T(\text{MLP}(t'_a)), E_T(t^t))$$

**稠密提示**：将图像通过 ImageBind 图像编码器获取 cls token $t_v$，重复后加到 SAM2 图像嵌入的每个像素位置上，提供对齐的视觉上下文。

### 关键设计三：文本桥接对齐监督（Text-bridged Alignment Supervision）

用文本作为中间桥梁，分别建立音频-文本和图像-文本的对齐关系：

**音频-文本损失** $\mathcal{L}_{a2t}$：
将音频查询 $t'_a$ 通过 ImageBind 投影层映射到共享嵌入空间，与所有类别的文本嵌入计算相似度，经 Hungarian 匹配后用交叉熵损失监督。

**图像-文本损失** $\mathcal{L}_{i2t}$：
用预测的分割图高亮原图中的前景区域（背景高斯模糊），通过 ImageBind 图像编码器提取视觉 token，与文本嵌入计算相似度后同样用交叉熵损失监督。

整体训练损失：

$$\mathcal{L} = \mathcal{L}_{a2pt} + \mathcal{L}_{a2t} + \mathcal{L}_{i2t} + \sum_{i=0}^{N}\mathcal{L}_{sep} + \mathcal{L}_{binary}$$

### 为什么不直接对齐音频和视觉？

实验验证了直接加入 $\mathcal{L}_{a2i}$ 反而降低性能，因为音频和视觉 token 来自查询和分割掩码，缺乏准确的 ground truth 监督，对齐这些含噪 token 引入了不确定性。文本把两个模态的噪声过滤为简洁的原型信息。

## 实验

### 主实验

在 AVSBench 三个子集上与 14 种方法对比：

| 方法 | Backbone | Size | S4 $\mathcal{M_J}$ | S4 $\mathcal{M_F}$ | MS3 $\mathcal{M_J}$ | MS3 $\mathcal{M_F}$ | AVSS $\mathcal{M_J^I}$ |
|------|----------|------|-----|-----|------|------|------|
| COMBO | PVT-v2 | 224 | 84.7 | **0.919** | 59.2 | 0.712 | 42.1 |
| SAMA-AVS | ViT-H | 1024 | 83.2 | 0.901 | 66.9 | 0.754 | - |
| **TAViS** | ViT-L | 224 | 84.8 | 0.912 | **68.2** | **0.759** | **44.2** |
| **TAViS** | ViT-L | 1024 | **87.0** | **0.926** | **71.2** | **0.796** | - |

TAViS@224 已超越使用 1024 分辨率的 SAMA-AVS（MS3: 68.2 vs 66.9），同时模型 MACs 更低（255G vs 598G）。

### 消融实验

核心组件消融：

| 设置 | S4 $\mathcal{M_J}$ | MS3 $\mathcal{M_J}$ |
|------|-----|------|
| w/o IBQD | 79.6 | 58.5 |
| w/o TbAS | 83.6 | 64.9 |
| w/o TbHP | 83.9 | 65.1 |
| TAViS | **84.8** | **68.2** |

对齐监督消融：

| 设置 | S4 $\mathcal{M_J}$ | MS3 $\mathcal{M_J}$ |
|------|-----|------|
| w/o $\mathcal{L}_{a2t}$ | 83.8 | 65.2 |
| w/o $\mathcal{L}_{i2t}$ | 84.0 | 64.4 |
| $\mathcal{L}_{a2i}$（直接对齐） | 83.1 | 66.5 |
| $\mathcal{L}_{a2t} + \mathcal{L}_{i2t}$（文本桥接） | **84.8** | **68.2** |

关键发现：
1. IBQD 移除后 MS3 下降 9.7，证明音频查询分解对多声源场景至关重要
2. $\mathcal{L}_{a2t}$ 和 $\mathcal{L}_{i2t}$ 都是必要的；仅用 $\mathcal{L}_{a2i}$ 时 S4 下降 1.7
3. 在 $\mathcal{L}_{a2t} + \mathcal{L}_{i2t}$ 基础上再加 $\mathcal{L}_{a2i}$ 也会降低性能（68.2→66.5），证实了文本桥接优于直接对齐
4. t-SNE 可视化显示：文本桥接使类内嵌入更紧凑，去除后类内分散显著增加

### 零样本泛化

| 方法 | Zero-shot $\mathcal{M_J^I}$ | 可训练参数 |
|------|---------------|-----------|
| OV-AVSS | 22.20 | 183.6M |
| **TAViS** | **28.21** | **54.9M** |

TAViS 以更少参数（54.9M vs 183.6M）在零样本上超越 OV-AVSS（+6.01 mIoU），验证了文本桥接对泛化的重要性。

## 亮点与洞察

1. **文本桥接的深刻洞察**：不直接对齐两个含噪模态，而是让文本充当去噪中介提取共享语义，理论和实验上都有力支撑
2. **IBQD 的设计巧妙**：分解查询保持为 cls token + 偏置的形式，保留了 ImageBind 的对齐空间不被破坏
3. **统一架构**：单一框架支持二值、语义、零样本三种分割模式
4. **效率优势**：224 分辨率即超越 1024 分辨率的竞品，MACs 降低 57%

## 局限性

1. 受计算资源限制，1024 分辨率下未在 AVSS（10帧/视频）上测试
2. 零样本设置下的类别预测依赖音频-文本和图像-文本相似度的简单加法组合，缺乏自适应权重
3. ImageBind 的预训练知识边界限制了零样本泛化的上限
4. 对 AVSBench 数据集规模的依赖仍存在（S4 仅 4932 视频）

## 相关工作

- **AVS 方法**：AVSBench、CATR、AVSegFormer 基于融合的方法；DiffusionAVS 基于生成的方法；AV-SAM、SAMA-AVS 基于 SAM的方法
- **多模态联合表示**：ImageBind、LanguageBind、OmniBind 多模态对齐框架
- **音频-文本对齐**：WavCaps 音频-文本对表示学习；AudioCLIP、WAV2CLIP 扩展 CLIP 到音频

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 文本桥接的设计哲学具有原创性，从理论到实验完整闭环
- **技术质量**: ⭐⭐⭐⭐⭐ — 消融极为充分，每个设计选择都有对比验证
- **实用性**: ⭐⭐⭐⭐ — 统一框架+零样本能力，但需同时加载两个基础模型
- **写作质量**: ⭐⭐⭐⭐⭐ — 逻辑链清晰，动机→挑战→方案→验证一气呵成

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Implicit Counterfactual Learning for Audio-Visual Segmentation](implicit_counterfactual_learning_for_audio-visual_segmentation.md)
- [\[ICCV 2025\] Towards Omnimodal Expressions and Reasoning in Referring Audio-Visual Segmentation](towards_omnimodal_expressions_and_reasoning_in_referring_audio-visual_segmentati.md)
- [\[ICCV 2025\] Can Generative Geospatial Diffusion Models Excel as Discriminative Geospatial Foundation Models?](can_generative_geospatial_diffusion_models_excel_as_discriminative_geospatial_fo.md)
- [\[ICCV 2025\] How Do Optical Flow and Textual Prompts Collaborate to Assist in Audio-Visual Semantic Segmentation?](how_do_optical_flow_and_textual_prompts_collaborate_to_assist_in_audio-visual_se.md)
- [\[CVPR 2025\] Robust Audio-Visual Segmentation via Audio-Guided Visual Convergent Alignment](../../CVPR2025/segmentation/robust_audio-visual_segmentation_via_audio-guided_visual_convergent_alignment.md)

</div>

<!-- RELATED:END -->
