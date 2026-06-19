---
title: >-
  [论文解读] Empowering Semantic-Sensitive Underwater Image Enhancement with VLM
description: >-
  [AAAI2026 Oral][多模态VLM][Underwater Image Enhancement] 利用 VLM 生成空间语义引导图，通过 cross-attention 注入和语义对齐损失的双重引导机制，赋予水下图像增强网络语义感知能力，使增强结果同时有利于人类感知和下游检测/分割任务。 水下图像增强（UIE）是海…
tags:
  - "AAAI2026 Oral"
  - "多模态VLM"
  - "Underwater Image Enhancement"
  - "视觉语言"
  - "Semantic Guidance"
  - "注意力机制"
  - "Downstream Tasks"
---

# Empowering Semantic-Sensitive Underwater Image Enhancement with VLM

**会议**: AAAI2026 Oral  
**arXiv**: [2603.12773](https://arxiv.org/abs/2603.12773)  
**代码**: 待确认  
**领域**: 多模态VLM  
**关键词**: Underwater Image Enhancement, Vision-Language Model, Semantic Guidance, Cross-Attention, Downstream Tasks  

## 一句话总结
利用 VLM 生成空间语义引导图，通过 cross-attention 注入和语义对齐损失的双重引导机制，赋予水下图像增强网络语义感知能力，使增强结果同时有利于人类感知和下游检测/分割任务。

## 背景与动机
水下图像增强（UIE）是海洋探索、水下机器人等领域的关键前处理步骤。当前基于深度学习的 UIE 方法虽然能产生视觉上令人满意的图像，但存在一个根本矛盾：**感知质量的提升并不能一致地转化为下游任务（检测、分割）的性能提升**。这是因为现有方法本质上是"语义盲"的——它们追求全局均匀增强，无法区分语义焦点区域（如海洋生物、水下物体）和非焦点区域（如水体背景），导致增强过程中引入分布偏移或隐性伪影，破坏了对下游模型至关重要的语义线索。

此前的解决方案存在明显不足：
- **基于语义分割图的方法**：依赖像素级标注数据，水下场景中此类标注极度稀缺
- **基于全局文本提示的方法**（如用 CLIP 做风格引导）：虽利用了 VLM 能力，但仍是"一刀切"策略，不关注图像内具体语义内容

## 核心问题
如何让 UIE 模型在增强过程中感知并聚焦关键语义区域，使增强结果既在视觉上更优，又能保留对下游机器视觉任务有利的语义特征？

## 方法详解

### 整体框架
该方法是一个**即插即用的语义敏感学习策略**（标记为 -SS），可适配多种编码器-解码器结构的 UIE 基线模型。核心流程分三步：

### 1. 语义引导图生成
- 输入退化水下图像 $I_d$，通过 **LLaVA**（VLM）生成图像中关键物体的文本描述 $T$
- 利用预训练的 **BLIP** 模型的视觉编码器 $\Phi_v$ 和文本编码器 $\Phi_t$ 分别提取 patch 特征 $F_v \in \mathbb{R}^{N \times C}$ 和全局文本特征 $f_t \in \mathbb{R}^C$
- 计算每个 patch 与文本的余弦相似度 $s_i = \hat{\mathbf{v}}_i^\top \hat{\mathbf{t}}$
- 通过**语义锐化函数**处理相似度分布：$s_i' = (\max(0, \mathcal{N}(s_i) - \delta))^\gamma$，其中 $\delta$ 为阈值过滤低响应噪声，$\gamma > 1$ 为幂律系数非线性放大分数差距
- 将一维分数序列上采样到原图尺寸，得到语义引导图 $M_{sem} \in \mathbb{R}^{H \times W}$

选择 BLIP 而非 CLIP 或 ViT 的原因：消融实验表明 BLIP 基于融合的对齐策略生成的热力图最干净、边界最锐利、空间精度最高，而 CLIP 容易在背景区域产生虚假激活，ViT 的类注意力图过于粗糙。

### 2. Cross-Attention 注入机制（结构引导）
在 UIE 网络**解码器**的各阶段 $l$：
- 解码器特征 $d_l$ 作为 Query $Q_l$
- 编码器跳跃连接特征 $e_l$ 经 $M_{sem}$（下采样至对应分辨率 $\tilde{M}^{(l)}$）逐元素加权后，投影为 Key $K_l$ 和 Value $V_l$
- 标准注意力计算：$d_l' = \text{softmax}(\frac{Q_l K_l^\top}{\sqrt{d_k}}) V_l$
- 该设计使解码器优先从语义"高亮"的编码器特征中提取信息

### 3. 显式语义对齐损失（特征监督）
对解码器第 $l$ 阶段的特征图 $\mathbf{F}^{(l)}$，设计两项损失：
- **背景抑制项**：$\|\mathbf{F}^{(l)} \odot (1 - \tilde{M}^{(l)})\|_F^2$ — 惩罚非关键区域的不必要强激活
- **前景增强项**：$-\eta \langle \mathbf{F}^{(l)}, \tilde{M}^{(l)} \rangle$ — 奖励与语义引导图一致的强响应

### 总损失函数
$$\mathcal{L}_{total} = \mathcal{L}_{recon} + \lambda_{align} \sum_{l \in L} \mathcal{L}_{align}^{(l)}$$
其中 $\mathcal{L}_{recon}$ 由 L1 损失和基于 VGG-19 的感知损失组成，$\lambda_{align}$ 经验设为 0.1。

## 实验关键数据

### UIE 感知质量（UIEB 数据集，全参考）

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| PFormer | 23.53 | 0.877 | 0.113 |
| PFormer-SS | **24.97**(+1.44) | **0.933**(+0.056) | **0.087**(-0.026) |
| FDCE | 23.66 | 0.909 | 0.111 |
| FDCE-SS | **24.63**(+0.97) | **0.927**(+0.018) | **0.093**(-0.018) |
| UIR-SS | **24.62**(+1.73) | 0.901 | 0.113 |

五个基线模型添加 -SS 后在 PSNR 和 SSIM 上全部提升。

### 下游任务——目标检测（Trash-ICRA19）

| 方法 | mAP↑ |
|------|------|
| SMDR | 95.76 |
| SMDR-SS | **96.98**(+1.22) |
| FDCE | 95.72 |
| FDCE-SS | **97.01**(+1.29) |

### 下游任务——语义分割（SUIM）

| 方法 | mIoU↑ |
|------|-------|
| PFormer | 69.34 |
| PFormer-SS | **74.75**(+5.41) |
| SMDR | 68.18 |
| SMDR-SS | **73.51**(+5.33) |
| PUIE | 66.20 |
| PUIE-SS | **70.80**(+4.60) |

语义分割的提升尤为显著，mIoU 普遍提升 2-5 个百分点，某些类别（如 RO）提升高达 15 个百分点。

### 消融实验关键结论
- **引导图注入位置**：仅解码器 > 编码器+解码器 > 仅编码器，验证了在重建阶段注入语义引导最有效
- **VLM 选择**：BLIP > CLIP > ViT，BLIP 的融合对齐策略生成的引导图质量最高

## 亮点
1. **即插即用设计**：策略可无缝集成到任意编码器-解码器结构的 UIE 模型，五个不同基线均获得一致提升
2. **双重引导机制**：cross-attention 提供隐式结构引导 + 语义对齐损失提供显式特征监督，互为补充
3. **弥合感知-认知鸿沟**：首次系统性地用 VLM 驱动的语义理解来解决水下增强"好看不好用"的问题
4. **语义锐化函数设计**：通过阈值+幂律变换组合，将平滑的相似度分布转为高对比度引导图

## 局限与展望
1. **推理开销**：需要 LLaVA 生成文本描述 + BLIP 计算对齐，推理时可能显著增加延迟，论文未讨论计算成本
2. **VLM 依赖**：引导图质量完全取决于 VLM 对退化图像的理解能力，严重退化场景下 VLM 可能描述不准确
3. **训练数据规模有限**：UIEB 仅 790 张训练图，更大规模数据集的表现未验证
4. **无参考指标提升有限**：在 U45 和 Challenge60 上的 UIQM/UCIQE 提升幅度不如全参考指标显著
5. **下游任务仍有波动**：个别类别（如 FV、RI）偶尔出现轻微下降，语义引导并非完全稳定

## 与相关工作的对比
- **vs 传统语义分割引导**（Wu et al. 2023）：本文用 VLM 替代像素级标注，避免了水下场景标注稀缺的问题
- **vs CLIP 全局文本引导**（Liu et al. 2024）：本文从全局风格引导升级为空间级逐 patch 的语义引导，实现内容感知的细粒度处理
- **vs 联合训练方案**（Yu et al. 2023）：本文的即插即用策略不需要为特定下游任务定制模型，通用性更强

## 启发与关联
- 该策略的核心思想——**用 VLM 生成空间语义先验引导低层视觉任务**——具有广泛迁移潜力，可应用于去雾、去雨、低光增强等任务
- 语义锐化函数的设计对 VLM 引导图后处理有参考价值
- 双重引导（结构+监督）的注入范式可推广到其他需要先验引导的图像恢复任务

## 评分
- 新颖性: 7/10 — VLM 驱动的空间语义引导思路新颖，但 cross-attention 注入机制本身较常规
- 实验充分度: 8/10 — 五个基线 + 三个 UIE 数据集 + 两个下游任务 + 消融实验，覆盖全面
- 写作质量: 8/10 — 动机阐述清晰，实验组织系统
- 价值: 7/10 — 即插即用的实用性强，对水下视觉社区有价值，但计算成本分析缺失影响实际部署评估

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] VipAct: Visual-Perception Enhancement via Specialized VLM Agent Collaboration and Tool-use](vipact_visual-perception_enhancement_via_specialized_vlm_age.md)
- [\[AAAI 2026\] Zero-Reference Joint Low-Light Enhancement and Deblurring via Visual Autoregressive Modeling with VLM-Derived Modulation](zero-reference_joint_low-light_enhancement_and_deblurring_via_visual_autoregress.md)
- [\[CVPR 2026\] UARE: A Unified Vision-Language Model for Image Quality Assessment, Restoration, and Enhancement](../../CVPR2026/multimodal_vlm/uare_a_unified_vision-language_model_for_image_quality_assessment_restoration_an.md)
- [\[CVPR 2025\] Completion as Enhancement: A Degradation-Aware Selective Image Guided Network](../../CVPR2025/multimodal_vlm/completion_as_enhancement_a_degradation-aware_selective_image_guided_network_for.md)
- [\[ICML 2026\] Benchmarking and Enhancing VLM for Compressed Image Understanding](../../ICML2026/multimodal_vlm/benchmarking_and_enhancing_vlm_for_compressed_image_understanding.md)

</div>

<!-- RELATED:END -->
