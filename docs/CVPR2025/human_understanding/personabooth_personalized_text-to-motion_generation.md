---
title: >-
  [论文解读] PersonaBooth: Personalized Text-to-Motion Generation
description: >-
  [CVPR 2025][人体理解][动作个性化] 定义 Motion Personalization 新任务，提出 PersonaBooth 多模态微调方法和 PerMo 大规模动作个性数据集，通过 persona token、对比学习和上下文感知融合，从几个基础动作中捕捉个人独特运动风格并生成文本驱动的个性化动作。
tags:
  - "CVPR 2025"
  - "人体理解"
  - "动作个性化"
  - "运动扩散模型"
  - "多模态微调"
  - "对比学习"
  - "上下文感知融合"
---

# PersonaBooth: Personalized Text-to-Motion Generation

**会议**: CVPR 2025  
**arXiv**: [2503.07390](https://arxiv.org/abs/2503.07390)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: 动作个性化, 运动扩散模型, 多模态微调, 对比学习, 上下文感知融合

## 一句话总结

定义 Motion Personalization 新任务，提出 PersonaBooth 多模态微调方法和 PerMo 大规模动作个性数据集，通过 persona token、对比学习和上下文感知融合，从几个基础动作中捕捉个人独特运动风格并生成文本驱动的个性化动作。

## 研究背景与动机

动作风格迁移（MST）仅使用单个源动作来迁移风格，限制了生成动作的多样性。现有方法面临以下不足：

1. **缺乏个性化概念**：现有 MST 方法关注抽象风格迁移，忽视了反映个体独特表达的 persona（如同一个「调皮」风格下不同人的表现方式不同）
2. **预训练数据中无 persona 信息**：HumanML3D 仅 15K 样本，几乎没有 persona 相关数据，与 persona 专注的微调数据存在显著分布差距
3. **内容干扰**：从不同内容（如跳跃 vs 行走）的动作中提取一致的 persona 非常困难
4. **已有方法只做视觉适配**：现有 MST 扩散方法不对文本通道做适配，限制了新 persona 信息的融入

本文定义 Motion Personalization 新任务：给定几个包含 persona 的基础动作，生成文本驱动的个性化运动。

## 方法详解

### 整体框架

PersonaBooth 对预训练的 MDM（Motion Diffusion Model）进行多模态微调。Persona Extractor 从输入动作提取视觉 persona 特征 $V^*$ 和 persona token $P^*$。$V^*$ 送入扩散模型的自适应层，$P^*$ 融入 Personalized Text Encoder 生成个性化文本特征 $T^*$，共同条件化扩散生成。

### 关键设计

**1. Persona Extractor 与 Persona Cohesion Loss**

- **功能**：从输入动作中提取内容无关的 persona 特征
- **核心思路**：基于预训练的 TMR motion-clip 模型提取通用动作特征，附加 transformer $\mathcal{E}_P$ 提取 persona 特征 $V^*$。$P^* = \text{MLP}(V^*[0])$ 作为文本空间的 persona token。通过监督对比学习 $L_{pc}$ 拉近同 persona 不同内容的特征，推远不同 persona 的特征
- **设计动机**：同一个人的 persona 在不同动作中表现方式不同（如优雅的芭蕾舞者走路时表现在脚，挥手时表现在手），对比学习帮助提取跨内容的一致 persona 特征

**2. 文本与视觉双通道自适应**

- **功能**：将 persona 信息同时融入文本和视觉两个模态
- **核心思路**：文本适配——将 "$P^*$ person is dancing" 中的 $[P^*]$ 替换为 persona token，用零初始化门控自适应合并原始和个性化文本嵌入：$T^* = \mathcal{X}_{clip}(T_{in}) + s_t \cdot \tanh(\gamma_t) \cdot \mathcal{X}_{clip}(\tilde{T}_{in}, P^*)$。视觉适配——在扩散模型 transformer 每层插入自适应 self-attention 层注入 $V^*$
- **设计动机**：仅做视觉适配会限制 persona 融入能力（HumanML3D 文本描述中没有个性修饰词），双通道确保 persona 信息的完整传递

**3. Context-Aware Fusion (CAF)**

- **功能**：多输入动作推理时，根据与文本 prompt 的相关性加权融合不同输入的 persona 特征
- **核心思路**：用 motion-clip 计算每个输入动作与 prompt 的余弦相似度 $S_i$，选取 Top-k 最相关动作，用 softmax 加权融合：$V^* = \sum_i w_i V_i^*$, $P^* = \sum_i w_i P_i^*$
- **设计动机**：简单取平均会导致动作混合（如一只手弯曲另一只手下垂的不自然姿势），CAF 根据上下文选择最相关的 persona 线索

### 损失函数

$$L = L_D + \lambda L_{pc}$$

其中 $L_D$ 为扩散重构损失加几何损失，$L_{pc}$ 为 persona 凝聚力对比损失，$\lambda = 10^{-2}$。训练使用 Classifier-Free Guidance，persona 条件以 10% 概率随机丢弃。

## 实验关键数据

### 消融实验：组件有效性（PerMo 数据集）

| 方法 | FID↓ | R-Prec Top-1↑ | PRA avg.↑ | Diversity↑ |
|------|------|-------------|-----------|-----------|
| Baseline (仅视觉适配) | 7.45 | 0.06 | 17.99 | 7.48 |
| + $P^*$ (文本适配) | 5.06 | 0.05 | 18.26 | 8.01 |
| + $L_{pc}$ | **3.18** | **0.15** | 18.05 | 7.74 |
| MI + Mean fusion | 3.52 | 0.19 | **19.24** | 7.88 |
| MI + CAF | **2.95** | **0.19** | 18.13 | **8.12** |

### PerMo 数据集对比

| 数据集 | 演员数 | 风格数 | 内容数 | 片段数 | Mesh | 文本 |
|--------|--------|--------|--------|--------|------|------|
| Xia | - | 8 | 6 | 572 | ✗ | ✗ |
| 100Style | 1 | 100 | 8 | 810 | ✗ | ✗ |
| **PerMo** | **5** | **34** | **10** | **6,610** | **✓** | **✓** |

### 关键发现

- 添加 persona token 文本适配使 FID 从 7.45 降至 5.06（-32%）
- $L_{pc}$ 对比学习进一步将 FID 降至 3.18，R-Precision 大幅提升（0.06→0.15），证实 persona 与内容解耦的重要性
- CAF 比简单平均降低 FID（3.52→2.95）同时提高 Diversity（7.88→8.12）

## 亮点与洞察

1. **定义新任务有远见**：Motion Personalization 比 Style Transfer 更贴近真实需求，数字人/元宇宙场景有直接应用
2. **多模态微调策略完善**：文本和视觉双通道适配加零初始化门控，避免灾难性遗忘
3. **PerMo 数据集贡献大**：首个多演员、多风格、带 mesh 和文本的大规模 persona 动作数据集

## 局限与展望

- 目前仅支持 SMPL-H 格式，未涵盖面部表情和手指细节
- 5 个演员的多样性可能不足以覆盖所有人群
- Persona 定义较模糊，不同风格之间可能存在重叠
- 未来可扩展到视频驱动的动作个性化

## 相关工作与启发

- **MDM**：基础运动扩散模型，PersonaBooth 在其上微调
- **MoMo**：零样本动作风格迁移，但源-目标内容不同时失败
- **SMooDi**：扩散微调方法，但重新训练慢且有遗忘问题
- **InstantBooth**：图像域个性化的简单平均策略不适用于动作

## 评分

⭐⭐⭐⭐ — 任务定义新颖，数据集贡献实质性，方法设计完善。多模态微调策略和对比学习的组合解决了核心挑战，消融实验充分验证了每个组件的贡献。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Few-Shot Personalized Scanpath Prediction](few-shot_personalized_scanpath_prediction.md)
- [\[CVPR 2025\] SimMotionEdit: Text-Based Human Motion Editing with Motion Similarity Prediction](simmotionedit_text-based_human_motion_editing_with_motion_similarity_prediction.md)
- [\[CVPR 2025\] FRESA: Feedforward Reconstruction of Personalized Skinned Avatars from Few Images](fresa_feedforward_reconstruction_of_personalized_skinned_avatars_from_few_images.md)
- [\[CVPR 2026\] MoLingo: Motion-Language Alignment for Text-to-Human Motion Generation](../../CVPR2026/human_understanding/molingo_motion-language_alignment_for_text-to-motion_generation.md)
- [\[CVPR 2025\] Exploring Timeline Control for Facial Motion Generation](exploring_timeline_control_for_facial_motion_generation.md)

</div>

<!-- RELATED:END -->
