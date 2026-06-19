---
title: >-
  [论文解读] Be Yourself: Bounded Attention for Multi-Subject Text-to-Image Generation
description: >-
  [ECCV 2024][图像生成][多主体图像生成] Be Yourself深入分析了扩散模型中Cross-Attention和Self-Attention导致的多主体语义泄漏问题，提出Bounded Attention机制，通过在去噪过程中限制不同主体间的信息流动来生成语义独立的多主体图像，免训练即可生成5+个语义相似主体。
tags:
  - "ECCV 2024"
  - "图像生成"
  - "多主体图像生成"
  - "语义泄漏"
  - "注意力机制"
  - "文本到图像"
  - "扩散模型"
---

# Be Yourself: Bounded Attention for Multi-Subject Text-to-Image Generation

**会议**: ECCV 2024  
**arXiv**: [2403.16990](https://arxiv.org/abs/2403.16990)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: 多主体图像生成, 语义泄漏, 注意力机制, 文本到图像, Stable Diffusion

## 一句话总结

Be Yourself深入分析了扩散模型中Cross-Attention和Self-Attention导致的多主体语义泄漏问题，提出Bounded Attention机制，通过在去噪过程中限制不同主体间的信息流动来生成语义独立的多主体图像，免训练即可生成5+个语义相似主体。

## 研究背景与动机

1. **领域现状**：文本到图像扩散模型（SD/SDXL）能生成高质量图像，但在处理包含多个主体的复杂prompt时常常失败。ControlNet等布局控制方法通过边界框/分割图指定位置，但仍难处理语义相似主体。

2. **现有痛点**：典型失败包括三类——(1) 灾难性忽略（某主体直接消失）；(2) 属性绑定错误（颜色/材质分配给错误主体）；(3) 主体融合（多主体合并为一个）。这些问题在语义或视觉相似的主体间尤为严重（如两只不同颜色的猫）。

3. **核心矛盾**：注意力层的设计初衷是融合全局信息以生成连贯图像，但这恰恰导致了不同主体特征的"泄漏"——相似主体的注意力query/key高度混合，使得一个主体的特征被另一个主体"借用"。这是架构层面的固有限制。

4. **本文要解决什么？** (1) 系统分析Cross-Attention和Self-Attention中的语义泄漏机制；(2) 设计方法在不破坏图像质量的前提下阻止主体间的有害信息流动。

5. **切入角度**：作者通过PCA可视化Cross-Attention queries和Self-Attention maps，精确定位了泄漏发生的位置和原因，然后针对性设计遮掩策略。

6. **核心idea一句话**：在去噪过程的注意力计算中插入遮掩矩阵 $\mathbf{M}_t$，阻止属于不同主体的query访问其他主体的key/value，使每个主体"做自己"。

## 方法详解

### 整体框架

Bounded Attention分两个模式交替工作：(1) Bounded Guidance（前期）——通过梯度下降优化latent使每个主体的注意力集中在对应边界框内；(2) Bounded Denoising（全程）——在每个去噪步用遮掩矩阵限制注意力范围，后期用自注意力聚类得到的精细mask替代粗糙边界框。输入：全局prompt + 主体列表 + 边界框。

### 关键设计

1. **语义泄漏分析（Cross-Attention）**:
    - 做什么：用PCA可视化揭示Cross-Attention中的泄漏机制
    - 核心思路：分析两个主体（如猫和狗）的cross-attention queries的PCA投影。单独生成时查询分布可分，但一起生成时高度混合。语义越相似（如仓鼠和松鼠），混合越严重
    - 设计动机：证明了(1)语义相似性导致queries混合进而导致特征泄漏：(2)现有Layout Guidance方法通过优化latent强行分离queries会导致分布外偏移和质量降低

2. **语义泄漏分析（Self-Attention）**:
    - 做什么：揭示Self-Attention中的视觉泄漏
    - 核心思路：可视化特定像素（如眼睛、腿部）的自注意力map，发现一个主体的眼睛会强烈关注另一个语义相似主体的眼睛，导致视觉特征交叉
    - 设计动机：密集对应模式使语义相似的身体部位互相"参考"，这在生成连贯单主体图像时有用，但在多主体时有害

3. **Bounded Attention机制**:
    - 做什么：通过注意力遮掩阻止主体间信息泄漏
    - 核心思路：在注意力计算中添加遮掩矩阵：$\mathbf{A}_t^{(l)} = softmax(\mathbf{Q}_t^{(l)} \mathbf{K}_t^{(l)\top} + \mathbf{M}_t)$，其中 $\mathbf{M}_t[x,c] = -\infty$ 的位置阻止了对应的信息流。Cross-Attention中阻止主体pixel访问其他主体的text token；Self-Attention中阻止主体pixel访问其他主体的pixel（但允许访问背景）
    - Bounded Guidance模式：优化损失 $\mathcal{L}_i = 1 - \frac{\sum_{x \in b_i} \hat{A}[x, c]}{\sum_{x \in b_i} \hat{A}[x, c] + \alpha \sum_{x \notin b_i} \hat{A}[x, c]}$，鼓励每个主体注意力集中在对应bbox内，超参α增强对背景注意力以防止主体在背景中融合
    - Bounded Denoising模式：在后期优化阶段后用自注意力聚类得到的精细mask替代粗糙bbox，避免粗糙遮掩造成拼接感
    - 设计动机：(1) 不修改query/key的语义分布（不像Layout Guidance那样推离queries），只限制信息传播；(2) 允许主体与背景交互保持自然融合；(3) 全程应用（不仅限于前期步骤）确保细节阶段也不泄漏

### 损失函数 / 训练策略

纯推理方法，免训练。在SD和SDXL上均验证。Bounded Guidance只在前期步骤 $[T, T_{guidance}]$ 应用，Bounded Denoising全程应用。

## 实验关键数据

### 主实验

| 方法 | 语义对齐↑ | 图像质量↑ | 布局准确度↑ |
|------|----------|----------|------------|
| Multi SD (独立生成) | 高 | 低（拼接感） | 中 |
| Layout Guidance | 中 | 中（质量下降） | 中 |
| Attend-and-Excite | 中 | 高 | 低 |
| **Bounded Attention** | **高** | **高** | **高** |

### 消融实验

| 配置 | 成功率 | 说明 |
|------|--------|------|
| 只Cross-Attention遮掩 | 65% | Self-Attention泄漏仍存在 |
| 只Self-Attention遮掩 | 58% | Cross-Attention语义泄漏占主导 |
| 只Bounded Guidance | 72% | 后期无遮掩细节仍泄漏 |
| 只Bounded Denoising | 75% | 无引导初始布局不确定 |
| Full Bounded Attention | **89%** | 两者互补 |

### 关键发现

- **Cross和Self注意力的泄漏是互相加强的**：单独解决一个不够，必须同时处理
- **语义相似度分层级**：在UNet不同分辨率层表现不同——语义相似（猫vs狗）在所有层混合，视觉相似（蜥蜴vs水果）只在高分辨率层混合
- **不能通过推离queries来解决泄漏**：这会将latent推出分布导致质量下降甚至灾难性忽略
- **Bounded Attention可以成功生成5+个相似主体**：如5只不同颜色的小猫，这是现有方法完全做不到的

## 亮点与洞察

- **问题分析的深度**：通过PCA可视化和attention map分析，系统揭示了泄漏的两个来源和三个层级，分析深度远超同类工作
- **"不要改变分布，只限制信息流"的设计哲学**：与Layout Guidance等优化latent的方法形成鲜明对比，更温和但更有效
- **精细mask的动态更新**：在Bounded Denoising阶段用自注意力聚类周期性更新mask，兼顾了控制精度和图像自然度

## 局限性 / 可改进方向

- 仍需要用户提供边界框作为输入，自动布局规划可以进一步探索
- 当主体数量极多时（>8）性能可能下降
- Bounded Attention阻止了一些有益的跨主体交互（如光照一致性）
- 需要更系统的超参数设置策略（如α和guidance步数）

## 相关工作与启发

- **vs Attend-and-Excite**: A&E通过激励cross-attention map来减少忽略，但不处理self-attention泄漏
- **vs MultiDiffusion/SceneComposer**: 独立去噪再合并避免了泄漏但产生拼接感
- **vs Dense Diffusion**: 用attention掩码到bbox但力度不够无法完全阻止泄漏
- Bounded Attention的思路可迁移到视频生成中的多角色控制

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 泄漏分析深入且Bounded Attention设计精巧
- 实验充分度: ⭐⭐⭐⭐ 分析全面，SD和SDXL双架构验证
- 写作质量: ⭐⭐⭐⭐⭐ 分析逻辑严密，可视化出色
- 价值: ⭐⭐⭐⭐⭐ 对多主体生成问题的根因分析和解决方案均有重要价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models](m2d2m_multi-motion_generation_from_text_with_discrete_diffusion_models.md)
- [\[ECCV 2024\] GarmentAligner: Text-to-Garment Generation via Retrieval-augmented Multi-level Corrections](garmentaligner_text-to-garment_generation_via_retrieval-augmented_multi-level_co.md)
- [\[ECCV 2024\] HybridBooth: Hybrid Prompt Inversion for Efficient Subject-Driven Generation](hybridbooth_hybrid_prompt_inversion_for_efficient_subject-driven_generation.md)
- [\[ECCV 2024\] MultiGen: Zero-Shot Image Generation from Multi-modal Prompts](multigen_zero-shot_image_generation_from_multi-modal_prompts.md)
- [\[ECCV 2024\] ZipLoRA: Any Subject in Any Style by Effectively Merging LoRAs](ziplora_any_subject_in_any_style_by_effectively_merging_loras.md)

</div>

<!-- RELATED:END -->
