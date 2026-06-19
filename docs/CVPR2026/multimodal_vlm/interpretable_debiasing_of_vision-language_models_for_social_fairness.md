---
title: >-
  [论文解读] Interpretable Debiasing of Vision-Language Models for Social Fairness
description: >-
  [CVPR 2026][多模态VLM][VLM去偏] 提出 DeBiasLens，通过在 VLM 编码器上训练稀疏自编码器（SAE）来定位编码社会属性的"社会神经元"，然后在推理时选择性去激活这些神经元以缓解偏见，在 CLIP 上降低 Max Skew 9-16%，在 InternVL2 上降低性别偏差比例 40-50%，同时保持通用性能。
tags:
  - "CVPR 2026"
  - "多模态VLM"
  - "VLM去偏"
  - "社会公平"
  - "稀疏自编码器"
  - "可解释性"
  - "神经元调控"
---

# Interpretable Debiasing of Vision-Language Models for Social Fairness

**会议**: CVPR 2026  
**arXiv**: [2602.24014](https://arxiv.org/abs/2602.24014)  
**代码**: 待确认  
**领域**: 多模态VLM  
**关键词**: VLM去偏, 社会公平, 稀疏自编码器, 可解释性, 神经元调控

## 一句话总结
提出 DeBiasLens，通过在 VLM 编码器上训练稀疏自编码器（SAE）来定位编码社会属性的"社会神经元"，然后在推理时选择性去激活这些神经元以缓解偏见，在 CLIP 上降低 Max Skew 9-16%，在 InternVL2 上降低性别偏差比例 40-50%，同时保持通用性能。

## 研究背景与动机
**领域现状**：VLM/LVLM 从大规模数据中继承和放大社会偏见——如 CLIP 对"CEO"检索偏向男性，InternVL 在模糊上下文中偏向特定性别。现有去偏方法包括微调、prompt tuning、剪枝等。

**现有痛点**：现有去偏方法只处理表面偏见症状而未触及内部表示中偏见的传播机制。剪枝虽然试图找到关键参数，但因神经元的多语义性（一个神经元同时编码偏见和有用知识），去偏往往以牺牲通用性能为代价。

**核心矛盾**：模型权重中的偏见和有用知识纠缠在一起，直接修改权重必然导致性能退化。

**本文目标** 如何在可解释的框架下精确定位和调控偏见相关的单一语义特征，而不波及有用知识。

**切入角度**：利用 SAE 将纠缠的特征空间分解为稀疏的、语义单一的神经元（满足 monosemanticity），使得偏见相关的"社会神经元"可以被独立定位和操控。

**核心idea**：SAE 把多语义特征解耦为单语义 → 筛选出编码特定社会属性的神经元 → 推理时去激活这些神经元消除偏见。

## 方法详解

### 整体框架
DeBiasLens 要解决的核心难题是：偏见和有用知识在模型权重里纠缠在一起，直接改权重必然伤性能。它的破局思路是先把纠缠的特征空间"拆干净"，再做精准手术。整条流水线分三步：先在 VLM 编码器最后一层挂一个稀疏自编码器（SAE），把稠密、多语义的编码器输出解耦成高维、单语义的稀疏表示；再在这个稀疏空间里筛出专门编码某个社会属性的"社会神经元"；最后在推理时把这些神经元单独关掉，用重建特征替换原特征送进下游，偏见信号被剥离而其余知识不动。三步分别对应下面三个关键设计，全程冻结原模型、只训练并操作 SAE。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["冻结的 VLM 图像/文本编码器<br/>最后一层稠密多语义特征 v"] --> B["SAE 解耦<br/>过完备稀疏编码（扩展因子8）<br/>解出单语义稀疏码 z"]
    B --> C["社会神经元探测<br/>高一致性(τ=0.9)+群组差集<br/>锁定属性专属神经元 𝒵B"]
    C --> D["社会神经元调控推理<br/>把 𝒵B 内激活置 γ→解码重建 v̂"]
    D -->|加权混合 v′=αv̂+(1−α)v, α=0.6| E["去偏特征 v′<br/>送入下游 T2I 检索 / VQA"]
```

### 关键设计

**1. SAE 解耦：把多语义特征拆成单语义神经元**

去偏失败的根因在于编码器神经元的多语义性——一个神经元同时编码性别和有用知识，剪掉它就连带损伤通用能力。SAE 用一个过完备的稀疏编码 $\phi(\mathbf{v}) = \sigma(\mathbf{W}_{enc}^\top(\mathbf{v} - \mathbf{b}_1))$ 把编码器输出 $\mathbf{v}$ 投到维度高得多（扩展因子设为 8）的稀疏空间，逼每个激活神经元只负责一个概念，从而满足单语义性（monosemanticity）。训练采用 Matryoshka SAE 的多尺度重建损失，关键是**全程不需要社会属性标签**——只在人脸/描述数据集上重建即可让 SAE 自发捕捉社会属性方向。一个直接的验证信号是：挂上 SAE 后，社会属性重叠的图像对与随机图像对的余弦相似度差异显著拉大，说明社会属性确实被某些专属神经元单独编码了。

**2. 社会神经元探测：用群组差集锁定属性专属神经元**

解耦之后还要回答"哪几个神经元才是负责性别/年龄/种族的"。论文对每个社会群组 $g$ 统计神经元的有效性——非零激活比例达到阈值 $\tau=0.9$ 才算"在该群组中稳定激活"，得到有效神经元集合 $\mathcal{E}_g$。光靠高激活还不够，因为通用神经元在所有群组都活跃；真正编码该群组的应当只在它身上活跃，于是取群组差集

$$\mathcal{N}_g = \mathcal{E}_g \setminus \bigcup_{h \neq g} \mathcal{E}_h$$

把对其他群组也有效的神经元剔除，再从中选均值激活最高的 top 神经元。"高一致性（$\geq\tau$）+ 群组特异性（差集）"两个条件叠加，筛出的就是只编码该群组社会属性的单语义神经元。

**3. 社会神经元调控推理：关闭神经元而不动权重**

拿到社会神经元后，推理时把它们在 SAE 稀疏码里的激活强制设为 $\gamma$（通常取 0），得到去偏后的稀疏码 $\mathbf{z}'$，再经 SAE 解码器重建出去偏特征 $\hat{\mathbf{v}} = \psi(\mathbf{z}')$。为避免重建误差伤及下游，最终送出的是重建特征与原始特征的加权混合 $\mathbf{v}' = \alpha\hat{\mathbf{v}} + (1-\alpha)\mathbf{v}$，$\alpha=0.6$。整个过程只动 SAE 的稀疏表示、不碰原模型权重，所以偏见被剥离的同时通用能力几乎无损——这正是它相比剪枝/微调的关键优势。

## 实验关键数据

### 主实验（CLIP ViT-B/16 性别偏见，Max Skew↓）

| 方法 | 可解释? | Adj | Occup | Act | Ster |
|------|--------|------|-------|------|------|
| CLIP 基线 | - | 21.9 | 33.5 | 19.8 | 32.5 |
| Bend-VLM | ✗ | 10.8 | 10.2 | 9.8 | 9.1 |
| SANER | ✗ | 8.9 | 14.5 | 7.7 | - |
| **DeBiasLens (T)** | **✓** | **7.1** | **16.2** | **14.2** | **8.1** |
| **DeBiasLens (I)** | **✓** | 14.2 | 21.5 | 20.0 | 18.3 |

### LVLM 去偏效果

| 配置 | 性别偏差率降低 | 通用性能下降 |
|------|-------------|------------|
| DeBiasLens-InternVL2 (α=0.6) | 40-50% | 仅 4-10 分 |
| 剪枝方法 | 类似 | 更大下降 |
| Prompt Engineering | 有限 | 最小 |

### 关键发现
- DeBiasLens(T) 在形容词和刻板印象类 prompt 上效果最优，无需属性标签训练
- 去激活仅 top-1 社会神经元即可达到与去激活所有有效神经元相当的效果，证实神经元间不互相干扰
- 性别神经元具有高特异性——去激活性别神经元不影响年龄偏见；但年龄神经元存在交叉效应（40%的年龄神经元有性别倾斜）
- 图像编码器对高分辨率 VLM (ViT-L/14@336) 更有效，文本编码器对普通分辨率更有效

## 亮点与洞察
- **可解释性驱动的去偏**是全新范式：不是"黑箱式"减轻偏见输出，而是精确定位和操控偏见产生的内部机制
- SAE 的单语义性质使其成为去偏的理想工具：每个神经元编码单一概念，去激活不产生连锁反应
- 框架同时适用于编码器型(CLIP)和编码器-解码器型(InternVL2)VLM，通用性好
- 仅需修改中间表示，不改模型权重，部署简单

## 局限与展望
- 社会神经元探测阶段仍需社会属性标签来划分群组（虽然SAE训练不需要）
- 目前主要验证了性别偏见，年龄和种族的消融较少
- SAE 的扩展因子和阈值 τ 需要调参
- 跨文化/跨语言的偏见缓解效果未验证

## 相关工作与启发
- **vs Bend-VLM**: Bend-VLM 直接去偏嵌入，黑箱操作；DeBiasLens 通过可解释的神经元操控，透明可审计
- **vs SANER**: SANER 在文本编码器上训练残差层擦除属性信息；DeBiasLens 通过 SAE 解耦后选择性去激活，更精准
- **vs MMNeuron**: MMNeuron 在预训练权重中找属性特定神经元，但权重神经元是多语义的；SAE 神经元是单语义的，去偏更精准

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次用 SAE 做可解释的 VLM 去偏，切入点独特
- 实验充分度: ⭐⭐⭐⭐ 多 VLM/LVLM + 多评估维度 + 神经元特异性验证
- 写作质量: ⭐⭐⭐⭐ 方法论述清晰，"社会神经元"概念形象
- 价值: ⭐⭐⭐⭐⭐ 为 AI 公平性提供了可解释、可审计的新工具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] A Closed-Form Solution for Debiasing Vision-Language Models with Utility Guarantees Across Modalities and Tasks](a_closedform_solution_for_debiasing_visionlanguage.md)
- [\[CVPR 2026\] FairLLaVA: Fairness-Aware Parameter-Efficient Fine-Tuning for Large Vision-Language Models](fairllava_fairness-aware_parameter-efficient_fine-tuning_for_large_vision-langua.md)
- [\[CVPR 2026\] Bias Is a Subspace, Not a Coordinate: A Geometric Rethinking of Post-hoc Debiasing in Vision-Language Models](bias_is_a_subspace_not_a_coordinate_a_geometric_rethinking_of_post-hoc_debiasing.md)
- [\[AAAI 2026\] BiPrompt: Bilateral Prompt Optimization for Visual and Textual Debiasing in Vision-Language Models](../../AAAI2026/multimodal_vlm/biprompt_bilateral_prompt_optimization_for_visual_and_textual_debiasing_in_visio.md)
- [\[CVPR 2026\] Spot The Ball: A Benchmark for Visual Social Inference](spot_the_ball_a_benchmark_for_visual_social_inference.md)

</div>

<!-- RELATED:END -->
