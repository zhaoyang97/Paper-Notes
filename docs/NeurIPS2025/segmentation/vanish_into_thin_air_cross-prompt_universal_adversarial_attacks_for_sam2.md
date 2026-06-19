---
title: >-
  [论文解读] Vanish into Thin Air: Cross-prompt Universal Adversarial Attacks for SAM2
description: >-
  [NeurIPS 2025][语义分割][对抗攻击] 提出UAP-SAM2——首个针对SAM2的跨提示通用对抗攻击方法，通过双重语义偏移框架（帧内语义混淆+帧间语义不一致）生成一个通用扰动，使SAM2在不同视频、帧和提示下的分割目标"消失无踪"。 SAM2是SAM的升级版，通过记忆机制将SAM从图像扩展到视频分割：用户仅需在…
tags:
  - "NeurIPS 2025"
  - "语义分割"
  - "对抗攻击"
  - "SAM2"
  - "通用对抗扰动"
  - "视频分割"
  - "语义偏移"
---

# Vanish into Thin Air: Cross-prompt Universal Adversarial Attacks for SAM2

**会议**: NeurIPS 2025  
**arXiv**: [2510.24195](https://arxiv.org/abs/2510.24195)  
**代码**: [GitHub](https://github.com/CGCL-codes/UAP-SAM2)  
**领域**: 图像分割  
**关键词**: 对抗攻击, SAM2, 通用对抗扰动, 视频分割, 语义偏移

## 一句话总结

提出UAP-SAM2——首个针对SAM2的跨提示通用对抗攻击方法，通过双重语义偏移框架（帧内语义混淆+帧间语义不一致）生成一个通用扰动，使SAM2在不同视频、帧和提示下的分割目标"消失无踪"。

## 研究背景与动机

SAM2是SAM的升级版，通过记忆机制将SAM从图像扩展到视频分割：用户仅需在第一帧提供提示，SAM2即可在后续帧中持续跟踪和分割目标。尽管对SAM的对抗鲁棒性已有较多研究，但SAM2的鲁棒性仍未被探索。

作者首先实验发现，**现有SAM攻击方法无法直接迁移到SAM2**。例如DarkSAM能使SAM性能下降98.25%，但对SAM2仅造成22.26%的下降。这一巨大差距源于SAM和SAM2之间两个关键的架构差异：

**提示的方向性引导（Directional Guidance from the Prompt）**：SAM为每帧提供独立提示，而SAM2仅在第一帧提供提示并持久存储，后续帧复用。即使第一帧被成功攻击，扰动也难以传递到后续帧。实验表明，即使将扰动预算提高到32/255，仅攻击第一帧仍无法显著影响SAM2的后续帧分割。

**帧间语义纠缠（Semantic Entanglement across Consecutive Frames）**：SAM2维护一个记忆库，缓存过去k帧的语义特征，通过记忆注意力模块融合历史信息引导当前帧分割。攻击单帧不足以破坏分割效果，因为记忆库中的干净特征会"修复"被攻击帧的影响。然而，如果能同时干扰记忆库中的历史特征，则可以显著损害当前帧的分割性能——这就是"雪崩效应"的原理。

## 方法详解

### 整体框架

UAP-SAM2从两个维度构建攻击：(1) 帧内语义扭曲——混淆当前帧中前景和背景的语义；(2) 帧间语义不一致——破坏连续帧之间的语义连续性。总优化目标为三个攻击损失之和：$\mathcal{J}_{\text{total}} = \mathcal{J}_{\text{sa}} + \mathcal{J}_{\text{fa}} + \mathcal{J}_{\text{ma}}$。

同时设计目标扫描策略（Target-scanning Strategy）实现跨提示迁移：将每帧均匀划分为 $m$ 个区域，每个区域随机分配一个提示，减少优化过程中对特定提示的依赖。攻击目标是图像编码器的输出特征而非依赖提示的mask，进一步提升跨提示泛化性。

### 关键设计

1. **语义混淆攻击（Semantic Confusion Attack, $\mathcal{J}_{\text{sa}}$）**: 利用二值前景mask $m_+$ 和背景mask $m_-$ 分离目标与背景。优化目标是让模型将前景错判为背景，同时强化背景区域的分类置信度。使用BCE损失增强对决策边界附近像素（logits接近0）的攻击力度：$\mathcal{J}_{\text{sa}} = \frac{1}{N}\sum_{i=1}^{N}[\text{BCE}(f_\theta(\tilde{x}_i, \mathcal{P}) \cdot m_+, y_-) + \text{BCE}((1 - f_\theta(\tilde{x}_i, \mathcal{P})) \cdot m_-, y_-)]$，其中 $y_-$ 是目标区域填充阈值-1、其余为0的mask。

2. **特征偏移攻击（Feature Shift Attack, $\mathcal{J}_{\text{fa}}$）**: 最大化对抗帧与干净帧在图像编码器特征空间中的距离。初始版本使用余弦相似度最小化。增强版本采用对比学习框架：将对抗帧和干净帧的增强原型视为负例对，其他视频的帧作为正例对，通过InfoNCE损失拉远对抗特征：$\mathcal{J}_{\text{fa}} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(\text{cos}(\mathcal{E}_{\text{img}}(\tilde{x}_i), e_i)/\tau)}{\sum_{k=1}^{N}\mathbf{1}_{k \neq i}\exp(\text{cos}(\mathcal{E}_{\text{img}}(\tilde{x}_i), \mathcal{E}_{\text{img}}(x_k))/\tau)}$，其中 $e_i = \frac{1}{\rho}\sum_{j=1}^{\rho}\mathcal{E}_{\text{img}}(\mathcal{T}(x_i))$ 是 $\rho$ 次随机增强的特征原型。

3. **记忆错位攻击（Memory Misalignment Attack, $\mathcal{J}_{\text{ma}}$）**: 从第二帧开始，最大化连续对抗帧之间的特征差异，诱发雪崩效应——当前帧与前一帧和第一帧的相似度逐步下降：$\mathcal{J}_{\text{ma}} = -\frac{1}{N}\sum_{i=1}^{N}\text{cos}(\mathcal{E}_{\text{img}}(\tilde{x}_{i+1}), \mathcal{E}_{\text{img}}(\tilde{x}_i))$。这种累积的语义不一致使记忆库中存储的特征与当前帧越来越不匹配。

### 损失函数 / 训练策略

UAP的扰动上界 $\epsilon = 10/255$（通用版）和 $8/255$（样本版），batch size 1，训练10个epoch。区域划分数 $m=256$，负样本数30，每个视频使用15帧。使用固定随机种子30保证可复现。实验在两张NVIDIA A100-SXM4 GPU上进行。

## 实验关键数据

### 主实验

**与现有方法对比（UAP，mIoU%，越低攻击越好）**:

| 方法 | YouTube视频(点) | DAVIS视频(点) | MOSE视频(点) | YouTube图像(点) | DAVIS图像(点) | MOSE图像(点) |
|------|----------------|-------------|-------------|----------------|-------------|-------------|
| UAPGD | 42.59 | 53.60 | 50.80 | 54.42 | 50.11 | 61.76 |
| AttackSAM | 64.35 | 62.31 | 63.05 | 64.18 | 55.53 | 63.92 |
| DarkSAM | 67.51 | 57.00 | 51.96 | 64.38 | 52.99 | 64.38 |
| **UAP-SAM2** | **37.03** | **42.47** | **33.67** | **27.54** | **48.45** | **50.13** |

UAP-SAM2在视频分割上的平均mIoU为37.72%，比最好的基线UAPGD低10.28%（攻击更强）。

### 消融实验

| 组件配置 | DAVIS mIoU(SAM2-T) | DAVIS mIoU(SAM2-S) | 说明 |
|---------|-------------------|-------------------|------|
| A (语义混淆) | ~52 | ~55 | 仅帧内攻击 |
| A+B (语义混淆+特征偏移) | ~45 | ~48 | 特征级别增强 |
| A+C (语义混淆+记忆错位) | ~43 | ~47 | 帧间攻击有效 |
| A+B+C (完整UAP-SAM2) | **~38** | **~42** | 三个组件互补 |

区域数m=256为最优，帧数15即可逼近全帧效果。

### 关键发现

- SAM2在无攻击时平均mIoU>76%，UAP-SAM2将其降至37.72%（平均降幅>38%）
- 视频分割比图像分割更容易被攻击，验证了帧间语义不一致攻击的有效性
- 跨数据集和跨模型迁移性强：在SAM2-T上生成的UAP迁移到SAM2-S和SAM2.1-T仍有效
- 模型剪枝和数据预处理（如模糊、遮挡）防御效果有限：剪枝40%时干净样本严重退化但对抗样本几乎不受影响
- 即使在极小扰动（$\epsilon = 4/255$）下，UAP-SAM2仍能造成>33%的mIoU下降

## 亮点与洞察

- **"雪崩效应"的发现和利用**是本文最大的贡献：通过逐帧累积语义差异，使整个视频的分割崩溃
- 目标扫描策略设计巧妙：不直接攻击依赖提示的mask，而是攻击图像编码器的通用特征
- 对比学习思想的引入（特征偏移攻击）使对抗样本的特征偏离本征语义空间
- 首次揭示了视频分割基础模型的安全漏洞，对安全关键应用有重要警示

## 局限与展望

- 仅针对SAM2系列模型，能否推广到其他视频分割模型（如XMem、Cutie）未知
- 当前UAP需要渐进优化，实时攻击场景下的效率可能不足
- 防御方面仅测试了简单的剪枝和预处理，未尝试对抗训练等更强防御
- 攻击假设可以获取完整模型（白盒），黑盒场景下的迁移性虽有验证但效果有所下降

## 相关工作与启发

- 与DarkSAM（SAM的SOTA攻击）相比，UAP-SAM2专门针对SAM2的记忆机制设计了帧间攻击
- 雪崩效应的发现暗示：视频模型中记忆机制可能是一把双刃剑，提升性能的同时也引入了新的攻击面
- 对安全关键领域（自动驾驶、医学影像）的视频分割系统设计具有重要警示意义

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个针对SAM2的UAP攻击，雪崩效应的发现很有洞察力
- 实验充分度: ⭐⭐⭐⭐⭐ 6个数据集、3个模型、72种设置，消融和防御实验全面
- 写作质量: ⭐⭐⭐⭐ 观察-设计-验证的叙事结构清晰
- 价值: ⭐⭐⭐⭐ 揭示了视频分割基础模型的安全隐患，对社区有警示价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] V²-SAM: Marrying SAM2 with Multi-Prompt Experts for Cross-View Object Correspondence](../../CVPR2026/segmentation/v2-sam_marrying_sam2_with_multi-prompt_experts_for_cross-view_object_corresponde.md)
- [\[NeurIPS 2025\] SANSA: Unleashing the Hidden Semantics in SAM2 for Few-Shot Segmentation](sansa_unleashing_the_hidden_semantics_in_sam2_for_few-shot_segmentation.md)
- [\[NeurIPS 2025\] Instant Video Models: Universal Adapters for Stabilizing Image-Based Networks](instant_video_models_universal_adapters_for_stabilizing_image-based_networks.md)
- [\[CVPR 2026\] Attack for Defense: Adversarial Agents for Point Prompt Optimization Empowering Segment Anything Model](../../CVPR2026/segmentation/attack_for_defense_adversarial_agents_for_point_prompt_optimization_empowering_s.md)
- [\[NeurIPS 2025\] HAODiff: Human-Aware One-Step Diffusion via Dual-Prompt Guidance](haodiff_human-aware_one-step_diffusion_via_dual-prompt_guidance.md)

</div>

<!-- RELATED:END -->
