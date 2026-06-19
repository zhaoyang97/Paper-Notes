---
title: >-
  [论文解读] RelaCtrl: Relevance-Guided Efficient Control for Diffusion Transformers
description: >-
  [AAAI 2026][图像生成][可控生成] 提出 RelaCtrl 框架，通过 ControlNet 相关性评分分析 DiT 各层对控制信息的敏感度差异，据此指导控制块的放置位置和建模强度，并设计二维混洗混合器（TDSM）替代自注意力和 FFN，以仅 15% 的参数量和计算复杂度实现优于 PixArt-δ 的可控生成效果。
tags:
  - "AAAI 2026"
  - "图像生成"
  - "可控生成"
  - "Transformer"
  - "ControlNet"
  - "参数高效"
  - "通道-Token 混洗"
---

# RelaCtrl: Relevance-Guided Efficient Control for Diffusion Transformers

**会议**: AAAI 2026  
**arXiv**: [2502.14377](https://arxiv.org/abs/2502.14377)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: 可控生成, 扩散 Transformer, ControlNet, 参数高效, 通道-Token 混洗

## 一句话总结

提出 RelaCtrl 框架，通过 ControlNet 相关性评分分析 DiT 各层对控制信息的敏感度差异，据此指导控制块的放置位置和建模强度，并设计二维混洗混合器（TDSM）替代自注意力和 FFN，以仅 15% 的参数量和计算复杂度实现优于 PixArt-δ 的可控生成效果。

## 研究背景与动机

### 领域现状

Diffusion Transformer（DiT）凭借强大的可扩展性在文本到图像/视频生成中取得了显著进展（PixArt-α、Flux、SD3、Sora 等）。可控生成是 DiT 的重要应用方向，当前主要通过附加控制分支（如 ControlNet）来实现对边缘、深度、分割等条件的遵循。

### 现有痛点

**问题一：参数和计算开销过大**
- PixArt-δ 直接复制前 13 个 Transformer 块，导致参数量和计算量增加 **50%**
- OminiControl 通过拼接控制 token 使 token 数翻倍，计算复杂度增加约 **70%**

**问题二：资源分配不均**
- 不同 DiT 层对控制信息的相关性差异被忽略
- 浅层到中层对控制信号更敏感，深层相关性较弱
- 对所有层使用统一的控制块配置导致深层存在大量冗余参数和计算

### 核心矛盾

如何在大幅减少控制分支参数量和计算量的同时，保持甚至提升可控生成的质量和精度？

### 切入角度

首先通过**系统性实验**量化每一层对控制效果的重要性（ControlNet Relevance Score），然后根据相关性**差异化分配**控制块的位置、参数规模和建模能力。

## 方法详解

### 整体框架

RelaCtrl 包含三个核心设计：
1. **ControlNet 相关性先验**：量化各层控制信息的重要性
2. **相关性引导的控制块放置**：在高相关性位置放置控制块
3. **相关性引导的轻量控制块（RGLC）**：用 TDSM 替代原始 Transformer 块

### 关键设计

#### 1. **ControlNet 相关性评分（CRS）**

**核心思路**：训练一个包含所有 27 层控制块的完整 ControlNet，在推理时逐一跳过每个控制块，用 FID（生成质量）和 HDD（控制精度）评估跳过该层的影响程度。

评分公式：

$$CRS_i = \frac{1}{2}\left(\frac{F_i - F_{min}}{F_{max} - F_{min}} + \frac{H_i - H_{min}}{H_{max} - H_{min}}\right)$$

其中 $F_i$ 和 $H_i$ 分别是跳过第 $i$ 层控制块后的 FID 和 HDD 排名。

**关键发现**：
- 相关性呈现**先升后降**的趋势
- 最关键的层集中在**浅层到中层**（如第 5、6、7 块）
- 去除最后几层控制块仅导致**微弱性能下降**
- 这与 LLM 中层重要性的分布规律（逐渐递减或 U 型）不同

**设计动机**：这意味着 PixArt-δ 直接复制前 13 层的策略并非最优——它可能遗漏了中间的关键层，同时保留了不必要的深层控制块。

#### 2. **相关性引导的控制块放置与建模**

根据 CRS 排名选择 top-11 个位置放置控制块（vs PixArt-δ 的 13 个连续前层），在保持可比性能的同时减少约 15% 的控制块数量。

进一步策略（Prior 2）：根据相关性调整每个位置的建模强度——高相关性位置减少通道分组数（扩大注意力特征维度），增强建模能力；低相关性位置增加分组数以减少计算。

#### 3. **二维混洗混合器（Two-Dimensional Shuffle Mixer, TDSM）**

**核心思路**：从 MetaFormer 的视角出发，Transformer 的两个核心组件是 token 混合器（自注意力）和通道混合器（FFN）。TDSM 将两者**统一为一个操作**。

**具体步骤**：
1. **随机通道选择**：将输入 $c_{in} \in \mathbb{R}^{H \times W \times D}$ 沿通道维度随机分为 $n$ 组 $c_{rs}^i \in \mathbb{R}^{H \times W \times d_i}$
2. **随机 3D 混洗**：在每组内打乱 token 的三维空间位置
3. **局部自注意力**：在固定大小 $s \times s \times d$ 的局部窗口内计算注意力
4. **逆恢复**：对 token 和通道维度执行逆操作恢复原始排列

**理论保证**：

$$d(t_j) \geq \frac{\sqrt{2}}{4}(H + Wd_i)$$

即 TDSM 中分组注意力的平均交互距离下界为 $\Omega(\frac{\sqrt{2}}{4}(H+Wd_i))$，保证了非局部交互的建模能力。

**设计动机**：
- 标准自注意力的 $O(N^2)$ 复杂度对控制分支来说过于昂贵
- FFN 层高度冗余（已有研究证明）
- 通过随机混洗打破局部分组的限制，在低计算开销下实现非局部建模

#### RGLC 块的完整流程

$$c_{cond} = ZC(TDSM(c_{in}) + c_{in})$$

其中 $c_{in}$ = 控制条件输入 $c$ + 零卷积($x$)（$x$ 来自对应的冻结主干块），$ZC$ 为零卷积。

### 训练策略

- 冻结 PixArt-α 主干网络
- 从头训练控制分支（RGLC 块 + 零卷积）
- 使用与 PixArt-δ 完全相同的训练设置以公平比较

## 实验关键数据

### 主实验

COCO 验证集上的定量比较：

| 模型/方法 | 条件 | HDD↓ | FID↓ | C-Ae↑ | C-SC↑ |
|----------|------|------|------|-------|-------|
| PixArt-δ | Canny | 96.26 | 21.38 | 5.508 | 0.279 |
| **RelaCtrl** | Canny | **94.04** | **20.34** | **5.584** | **0.282** |
| PixArt-δ | HED | 98.91 | 29.22 | 5.243 | 0.275 |
| **RelaCtrl** | HED | **96.11** | **27.73** | **5.451** | **0.276** |
| PixArt-δ | Depth | 99.69 | 35.21 | 5.723 | 0.283 |
| **RelaCtrl** | Depth | **99.11** | **33.93** | **5.887** | **0.285** |
| PixArt-δ | Seg. | 0.379(mIoU) | 35.50 | 5.668 | 0.282 |
| **RelaCtrl** | Seg. | **0.405** | **33.76** | **5.702** | **0.287** |

RelaCtrl 在所有 4 种条件控制任务上**全面超越** PixArt-δ，同时参数量仅为其 15.3%。

### 消融实验

控制块数量的影响（基于相关性排名）：

| 配置 | HDD↓ | FID↓ | 参数比例 |
|------|------|------|---------|
| ControlNet-top13（基线） | 96.26 | 21.38 | 100% |
| Relevance-top13 | 94.57 | 20.31 | 100% |
| Relevance-top12 | 95.88 | 20.79 | 92.5% |
| **Relevance-top11** | 95.57 | 21.28 | 84.6% |
| Relevance-top10 | 96.36 | 22.24 | 76.9% |

RGLC 和 Prior 2 的影响：

| 配置 | HDD↓ | FID↓ | 参数比例 |
|------|------|------|---------|
| **RelaCtrl（完整）** | **94.04** | **20.34** | **15.3%** |
| w/o RGLC（用原始拷贝块） | 95.57 | 21.28 | 84.6% |
| w/o Prior 2（均匀 TDSM） | 97.30 | 22.47 | 17.1% |
| Baseline (PixArt-δ) | 96.26 | 21.38 | 100% |

### 效率分析

| 方法 | 参数量(M) | 计算量(GFLOPs) | 推理时间(s) |
|------|----------|---------------|------------|
| PixArt-α（基线） | 611.15 | 542.56 | 3.81 |
| +ControlNet | +294.34 (+48.16%) | +270.57 (+49.87%) | +0.51 |
| **+RelaCtrl** | **+45.15 (+7.38%)** | **+46.71 (+8.61%)** | **+0.24** |

### 关键发现

- **相关性引导 > 顺序复制**：即使使用相同 13 个控制块，按相关性排名放置（Relevance-top13）比顺序前 13 层效果更好（FID 20.31 vs 21.38）
- **11 块 ≈ 13 块**：在相关性引导下，11 个控制块即可达到 13 个的效果
- **RGLC 块比原始拷贝块更好**：用 TDSM 替换自注意力+FFN 在参数减少 85% 的情况下反而提升了效果
- **Prior 2 很重要**：去除相关性引导的 TDSM 通道调整后性能显著下降
- **四种条件全面有效**：Canny、HED、Depth、Segmentation 均有提升

## 亮点与洞察

1. **分析性研究驱动设计**：不是凭直觉而是通过系统实验（逐层删除）量化每层的控制贡献，这种"先分析再设计"的方法论值得借鉴
2. **相关性分布的反直觉发现**：DiT 中控制信息的相关性呈"先升后降"而非单调趋势，与 LLM 的规律不同，提示不同任务下层重要性分布可能完全不同  
3. **TDSM 的理论保证**：不仅设计了高效替代模块，还从理论上证明了其非局部建模能力的下界
4. **极致效率**：7.38% 额外参数 + 8.61% 额外计算 = 超越 48%+ 参数的 PixArt-δ，效率比约 **6.5 倍**

## 局限与展望

- CRS 需要先训练一个完整的 ControlNet（27 个控制块），这一前期分析的计算开销未被讨论
- 目前的相关性分析基于 PixArt-α，其结论是否适用于 Flux、SD3 等其他 DiT 架构有待验证
- TDSM 中的随机混洗可能引入噪声，长期来看对训练稳定性的影响未充分讨论
- 仅在 512 分辨率上验证，高分辨率（如 1024、2048）下的效率优势可能更大但未探索
- 视频生成（如 CogVideoX）中的控制也面临类似效率问题，但本文未涉及

## 相关工作与启发

- **与 ControlNet-XS 的关系**：ControlNet-XS 从反馈控制系统角度改进交互带宽，RelaCtrl 从层重要性分析角度优化资源分配，两者互补
- **MetaFormer 的影响**：将 Transformer 解构为 token mixer + channel mixer 的视角，为 TDSM 的设计提供了理论指导
- **启发**：相关性分析可以推广到其他需要附加模块的场景（如 LoRA 的层级分配、适配器位置选择等）
- 对于 DiT 的 ControlNet 设计，未来可能的方向是**自适应相关性估计**（不需要预训练完整 ControlNet）

## 评分

- 新颖性: ⭐⭐⭐⭐ （相关性分析+TDSM 设计有新意，但整体思路是"分析+剪枝"）
- 实验充分度: ⭐⭐⭐⭐⭐ （4 种条件任务、多个基线、详尽消融、效率分析）
- 写作质量: ⭐⭐⭐⭐⭐ （结构清晰、定理证明严谨、可视化丰富）
- 价值: ⭐⭐⭐⭐⭐ （解决了 DiT 可控生成效率的关键问题，实用性强）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Laytrol: Preserving Pretrained Knowledge in Layout Control for Multimodal Diffusion Transformers](laytrol_preserving_pretrained_knowledge_in_layout_control_fo.md)
- [\[CVPR 2026\] DDiT: Dynamic Patch Scheduling for Efficient Diffusion Transformers](../../CVPR2026/image_generation/ddit_dynamic_patch_scheduling_for_efficient_diffusion_transformers.md)
- [\[AAAI 2026\] DogFit: Domain-guided Fine-tuning for Efficient Transfer Learning of Diffusion Models](dogfit_domain-guided_fine-tuning_for_efficient_transfer_learning_of_diffusion_mo.md)
- [\[CVPR 2026\] Layer-wise Instance Binding for Regional and Occlusion Control in Text-to-Image Diffusion Transformers](../../CVPR2026/image_generation/layer-wise_instance_binding_for_regional_and_occlusion_control_in_text-to-image_.md)
- [\[ICML 2026\] Spectral Guidance for Flexible and Efficient Control of Diffusion Models](../../ICML2026/image_generation/spectral_guidance_for_flexible_and_efficient_control_of_diffusion_models.md)

</div>

<!-- RELATED:END -->
