---
title: >-
  [论文解读] StyleMaster: Stylize Your Video with Artistic Generation and Translation
description: >-
  [CVPR 2025][图像生成][视频风格化] StyleMaster通过基于prompt-patch相似度的局部纹理选择和基于模型幻觉生成的对比学习全局风格提取，结合运动适配器和灰度Tile ControlNet，实现了兼具风格忠实度和内容保持的高质量视频风格化生成与迁移。 视频风格化任务需要生成或迁移视频到给定参考图像…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "视频风格化"
  - "风格迁移"
  - "扩散模型"
  - "对比学习"
  - "纹理特征"
---

# StyleMaster: Stylize Your Video with Artistic Generation and Translation

**会议**: CVPR 2025  
**arXiv**: [2412.07744](https://arxiv.org/abs/2412.07744)  
**代码**: [项目页面](https://zixuan-ye.github.io/stylemaster)  
**领域**: 图像生成 / 视频风格化  
**关键词**: 视频风格化, 风格迁移, 扩散模型, 对比学习, 纹理特征

## 一句话总结

StyleMaster通过基于prompt-patch相似度的局部纹理选择和基于模型幻觉生成的对比学习全局风格提取，结合运动适配器和灰度Tile ControlNet，实现了兼具风格忠实度和内容保持的高质量视频风格化生成与迁移。

## 研究背景与动机

视频风格化任务需要生成或迁移视频到给定参考图像的风格，但现有方法面临三个核心问题：(1) **局部纹理丢失**：现有方法侧重全局风格但忽略笔触等局部纹理细节；(2) **内容泄漏**：直接使用CLIP全局嵌入或所有patch特征会导致参考图像的内容混入生成结果；(3) **风格-内容解耦不足**：全局表示难以同时避免内容泄漏和保持风格精度。

此外，现有风格数据集（如Style30K）无法保证组内风格一致性，影响对比学习效果。例如同组中一张属于真实世界域而另一张属于动画域。收集和分组过程也耗费大量人力。

StyleMaster的核心洞察是：风格提取阶段至关重要，需要同时考虑全局和局部信息，并利用模型幻觉(model illusion)自动生成绝对风格一致的配对数据。

## 方法详解

### 整体框架

StyleMaster基于DiT视频生成模型，包含四个核心组件：(1) 基于模型幻觉的对比数据集构建与全局风格投影；(2) 基于prompt-patch相似度的局部纹理选择；(3) 运动适配器增强时序质量和风格程度；(4) 灰度Tile ControlNet实现视频风格迁移的内容控制。

### 关键设计1: 模型幻觉对比数据集与全局风格提取

- **功能**: 生成绝对风格一致的配对数据，训练将CLIP图像嵌入投影为纯风格表示的模块
- **核心思路**: 利用VisualAnagrams的模型幻觉性质——在T2I模型采样过程中，复制并变换（旋转/翻转）含噪图像形成并行过程，用不同prompt引导预测，生成像素重排后改变内容但保持风格的配对图像。用两个列表（物体+风格描述）随机组合生成无限配对数据。训练MLP投影层 $F_{\text{global}} = \text{MLP}(F_i)$ 将CLIP图像嵌入转化为全局风格描述，使用triplet loss训练
- **设计动机**: Style30K等手工收集的数据集无法保证风格一致性，而模型幻觉生成的配对仅是像素重排，可确保组内绝对风格一致。不微调整个CLIP以保持泛化能力

### 关键设计2: 基于Prompt-Patch相似度的局部纹理选择

- **功能**: 从CLIP patch特征中选择携带纹理信息但不含内容的patches，避免内容泄漏
- **核心思路**: 计算CLIP patch特征 $F_p$ 与文本特征 $F_{\text{text}}$ 的相似度，选择相似度最低的$k=15$个patch作为纹理特征（因为与文本内容最不相关的patch更可能只携带纹理信息）。通过Q-Former结构聚合选定patch特征，再与全局风格拼接后通过dual cross-attention注入模型：$F_{\text{out}} = \text{TCA}(F_{\text{in}}, F_{\text{text}}) + \text{SCA}(F_{\text{in}}, F_{\text{style}})$
- **设计动机**: 直接使用所有256个patch特征会导致严重内容泄漏（UMT Score从2.329降至0.771），随机丢弃可以缓解但不如基于相似度的选择有效

### 关键设计3: 运动适配器与灰度Tile ControlNet

- **功能**: 运动适配器增强视频时序质量和风格程度；灰度Tile ControlNet提供精确的内容控制
- **核心思路**: 在temporal attention的$W_Q, W_K, W_V$上训练LoRA $\widetilde{W} = W + \alpha \cdot A_t^{W,\text{down}} \cdot A_t^{W,\text{up}}$，训练时$\alpha=1$生成静止视频。推理时设$\alpha=-0.3$，负值不仅增加动态范围还将生成结果远离真实域，增强风格化程度。对于视频风格迁移，使用灰度tile图像作为ControlNet条件，去除RGB颜色信息以免干扰风格注入
- **设计动机**: 图像训练会导致视频时序闪烁和动态不足；RGB tile会引入颜色干扰

### 损失函数

- 全局投影训练：triplet loss $\mathcal{L} = \sum_{n=1}^{N}[\|f(F_{i,n}^{\text{anc}}) - f(F_{i,n}^{\text{pos}})\| - \|f(F_{i,n}^{\text{anc}}) - f(F_{i,n}^{\text{neg}})\| + \alpha]$
- 风格模块训练：标准扩散去噪损失
- 灰度Tile ControlNet训练：标准扩散去噪损失

## 实验关键数据

### 主实验1: 图像风格迁移

| 方法 | CSD-Score↑ | ArtFID↓ | FID↓ | LPIPS↓ |
|------|-----------|---------|------|--------|
| StyleID (CVPR'24) | 0.40 | 38.57 | 23.91 | 0.55 |
| InstantStyle | 0.32 | 42.48 | 24.59 | 0.67 |
| CSGO | 0.35 | 41.42 | 25.71 | 0.56 |
| **StyleMaster** | **0.45** | **36.89** | **22.11** | 0.61 |

### 主实验2: 视频风格化生成

| 方法 | CLIP-Text↑ | UMT-Score↑ | CSD-Score↑ | MotionSmooth↑ |
|------|-----------|-----------|-----------|--------------|
| VideoComposer | 0.057 | -2.268 | 0.680 | 0.975 |
| StyleCrafter | 0.294 | 1.994 | 0.448 | 0.973 |
| **StyleMaster** | **0.305** | **2.329** | 0.463 | **0.994** |

### 消融实验: 风格提取模块设计

| 配置 | 全局(GP) | 纹理(选择) | UMT Score | CSD Score |
|------|---------|-----------|-----------|-----------|
| B1: CLIP嵌入 | ✗ | - | 0.892 | 0.561 |
| B2: 全局投影 | ✓ | - | 2.337 | 0.443 |
| B3: 所有patch | - | 无选择 | 0.771 | 0.534 |
| B5: 相似度选择 | - | ✓ | 2.331 | 0.452 |
| **B6: 全局+局部** | **✓** | **✓** | **2.329** | **0.463** |

### 关键发现

- ArtFID（综合考虑风格和内容的指标）上显著优于所有竞争者
- 全局投影使UMT Score从0.892提升到2.337，有效防止内容泄漏
- 运动适配器$\alpha=-0.3$在视觉质量、动态程度和风格相似度间取得最佳平衡
- VideoComposer的CSD最高(0.680)但因为直接复制参考图像内容，UMT Score为负

## 亮点与洞察

1. **模型幻觉生成配对数据**：巧妙利用T2I模型的幻觉性质，几乎零成本生成无限量的绝对风格一致配对数据，这一数据生成策略可广泛应用于风格相关研究
2. **以退为进的运动适配器**：训练适配器做"静止"，推理时取负值做"运动+"，同时隐式增强风格化程度
3. **选择性纹理保持**：基于prompt-patch相似度的简洁选择策略，有效平衡纹理保持与内容泄漏

## 局限与展望

- 当前方法仅处理图像的静态风格，未涉及动态风格（如粒子效果、运动特征）
- 风格提取仍依赖参考图像，未来可探索从参考视频中提取和迁移动态风格
- 灰度Tile ControlNet会丢失部分颜色信息，在某些场景下可能影响内容保持

## 相关工作与启发

- **IP-Adapter**: 图像适配器方案但无法解耦风格和内容
- **StyleTokenizer**: 在Style30K上用对比学习微调CLIP，但数据集风格一致性差
- **CSGO**: 使用B-LoRA生成triplet数据集，但只能提取全局表示
- **StillMoving**: 提出运动适配器概念，本文在此基础上发现负向比率的风格增强效果
- 启发：生成模型的"缺陷"（幻觉性质）可以被转化为有价值的特性

## 评分

⭐⭐⭐⭐ — 多个创新点（模型幻觉数据生成、prompt-patch选择、负向运动适配器）有机结合，每个设计都有充分的消融验证。在图像和视频风格化任务上全面超越现有方法。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Divot: Diffusion Powers Video Tokenizer for Comprehension and Generation](divot_diffusion_powers_video_tokenizer_for_comprehension_and_generation.md)
- [\[CVPR 2025\] Boost Your Human Image Generation Model via Direct Preference Optimization](boost_your_human_image_generation_model_via_direct_preference_optimization.md)
- [\[ICCV 2025\] Video Color Grading via Look-Up Table Generation](../../ICCV2025/image_generation/video_color_grading_via_look-up_table_generation.md)
- [\[CVPR 2025\] Trust Your Critic: Robust Reward Modeling and Reinforcement Learning for Faithful Image Editing and Generation](trust_your_critic_robust_reward_modeling_and_reinforcement_learning_for_faithful.md)
- [\[CVPR 2025\] SVFR: A Unified Framework for Generalized Video Face Restoration](svfr_a_unified_framework_for_generalized_video_face_restoration.md)

</div>

<!-- RELATED:END -->
