---
title: >-
  [论文解读] Enhancing Vision Transformers for Object Detection via Context-Aware Token Selection and Packing
description: >-
  [ICLR 2026][目标检测][稀疏注意力] 提出 Select and Pack Attention（SPA）：用一个轻量门控层在多尺度物体标签监督下**动态**挑出每张图里真正有信息的 token，再把数量参差不齐的 token **打包**进定长容器恢复批并行，从而在目标检测上同时拿到 +0.5~2.7 AP 的精度提升和 10.9%~24.9% 的计算量下降。
tags:
  - "ICLR 2026"
  - "目标检测"
  - "稀疏注意力"
  - "Token 选择"
  - "Transformer"
  - "多尺度监督"
  - "Token Packing"
---

# Enhancing Vision Transformers for Object Detection via Context-Aware Token Selection and Packing

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=Q1LVcZ1PWc](https://openreview.net/forum?id=Q1LVcZ1PWc)  
**代码**: 待确认  
**领域**: 高效 Transformer / 稀疏注意力 / 目标检测  
**关键词**: 稀疏注意力, Token 选择, 目标检测, Vision Transformer, 多尺度监督, Token Packing  

## 一句话总结
提出 Select and Pack Attention（SPA）：用一个轻量门控层在多尺度物体标签监督下**动态**挑出每张图里真正有信息的 token，再把数量参差不齐的 token **打包**进定长容器恢复批并行，从而在目标检测上同时拿到 +0.5~2.7 AP 的精度提升和 10.9%~24.9% 的计算量下降。

## 研究背景与动机
**领域现状**：ViT 凭借全局自注意力在检测/分割上超过 CNN，但注意力计算量随 token 数平方增长，而图像里背景 token 往往占大多数——尤其是小目标检测这种稀疏场景，绝大部分像素无信息却仍被算进注意力。为此学界提出了一批稀疏注意力方法来只对"重要 token"做注意力。

**现有痛点**：本文把已有方法的失败归为两类。**效率上**，GPU 批训练要求一个 batch 内 token 数对齐，但每张图的有效 token 数不一样——SparseViT 用 padding 把所有图补到 batch 内最大长度，反而塞回大量背景 token；DynamicViT/EViT 只在推理时丢固定数量 token，训练阶段仍对全部 token 算注意力外加一个 mask 预测模块，训练开销甚至超过原始 ViT；DAT 只是缩小 query 的感受野，token 一个没少。**性能上**，这些方法主要在分类这种容忍信息丢失的简单任务上有效，一旦迁到检测/实例分割这类需要丰富语义的任务，DynamicSwin 因为 token 选得不准会丢掉整个物体，性能明显掉。

**核心矛盾**：稀疏注意力想省算力就得丢 token，但"丢哪些 token"既缺乏对下游检测目标的**上下文感知**（多用启发式/固定比例），又因为每张图保留的 token 数不一致而**破坏了批并行**——省了 FLOPs 却省不出实际吞吐。

**本文目标**：设计一种既能按输入内容自适应地、准确地挑 token，又能在挑完之后仍保持 GPU 批并行的稀疏注意力，使其在复杂检测任务上做到精度、效率双赢。

**核心 idea**：**(1) 动态选择 + 显式监督**——用线性门控层给每个 token 打分，并用物体标签（bbox/mask）派生的**多尺度选择标签**直接监督这个分数，让"挑 token"不再是隐式学习的副产物；**(2) Token Packing**——把各图选出的、数量不等的正 token 装进定长容器拼成新 batch，用注意力 mask 隔开不同图，从而恢复批并行、把计算量真正降下来。

## 方法详解

### 整体框架
SPT（Select and Pack Transformer）是一个四阶段层级骨干网，逐级下采样产出 4 种尺度的特征 $r_1,\dots,r_4$。前两阶段用标准 Swin block，后两阶段换成 SPA block——这一分工借鉴 DAT 的观察：浅层特征区分度不够，过早做 token 选择会误删导致严重信息丢失，所以从第三阶段起才开始选 token。每个 SPA block 既输出给下一层，又把 score map 往下传，配合该尺度自己的选择标签构成多尺度监督。由于 SPA 比 Swin 更省，作者在第三阶段额外多塞了 4 个 block 来换精度，整体计算量仍更低。

```mermaid
flowchart LR
    X[输入图像] --> S1[Stage1<br/>Swin Block]
    S1 --> S2[Stage2<br/>Swin Block]
    S2 -->|门控 s0| S3[Stage3<br/>SPA Block ×N+4]
    S3 -->|score s1 上传| S4[Stage4<br/>SPA Block]
    S2 -. 多尺度选择标签 .-> S3
    S3 -. 多尺度选择标签 .-> S4
    S4 --> OUT[多尺度特征 → 检测头]
```

### 关键设计

**1. 多尺度监督的 Token 选择：把"挑 token"从隐式副产物变成被显式教学的任务。** 作者发现若只靠最终任务损失隐式引导，门控层会给几乎所有 token 都打高分，token 选不出来、效率上不去。于是引入由物体标签派生的选择标签直接监督——检测任务把所有 bbox 叠成一张聚合二值 mask（物体区域为 1）。但单尺度标签又会把选择卡得太死、丢信息，所以做两件事：调低 Gumbel-Softmax 阈值，并融合多尺度标签——每个 SPA block 不仅用与当前特征匹配的尺度，还把上一阶段经 max-pooling 对齐后的上采样分数取**逐元素最大值**纳入，从而多保留一些有信息的 token。具体地，门控 $f_{\theta_g}$ 对展平输入 $r\in\mathbb{R}^{B\times N\times C}$ 打分，先与上采样分数取大、再经 sigmoid 门控、最后用 Gumbel-Softmax 分离出正 token $r_p$：

$$s = \mathrm{Max}(f_{\theta_g}(r),\, s_{up}),\quad r_g = \mathrm{Sigmoid}(s)\odot r,\quad r_p = \text{Gumbel-Softmax}(s)\odot r_g$$

选择损失用二值交叉熵并对所有 SPA block 求和：$\mathcal{L}_{select} = -\sum_{block}\big(y\log s + (1-y)\log(1-s)\big)$，总损失 $\mathcal{L}_{SPT}=\mathcal{L}_{task}+\alpha\mathcal{L}_{select}$。

**2. Token Packing：用定长容器把"数量参差不齐的正 token"重新塞回批并行。** 动态选择后每张图选出的 token 数都不一样，若像 SparseViT 那样 padding 到 batch 内最大长度就会塞回大量背景 token。作者借鉴序列打包思路，预置一系列定长 $L$ 的容器，把所有选出的正 token 依次填进去，只有最后一个容器在凑不满时才 padding。打包后得到 $p\in\mathbb{R}^{B'\times L\times C}$，新 batch 数 $B'$ 远小于原始的 $B$，token 总量 $B'\times L \ll B\times N$（稀疏数据上尤其明显）。容器内做注意力时用 mask 保证每个 token 只与**同一张原图**的 token 互相注意，避免跨图污染；$L$ 取 Swin 窗口大小的平方 $M^2$。复杂度对比直观说明收益：

$$\Omega(\text{MSA}) = B(4NC^2+2N^2C),\quad \Omega(\text{W-MSA}) = B(4NC^2+2M^2NC)$$
$$\Omega(\text{SPA}) = B(NC+NC^2) + B'(3LC^2+2L^2C)$$

W-MSA 把对 $N$ 的依赖从平方降到线性，而 SPA 不仅对 $N$ 线性，新批量 $B'\ll B$ 让计算量进一步下降。

**3. 与 Swin 移窗结合防止跨容器信息丢失：靠特征图移位让"被分进不同容器的 token 对"轮换。** 打包会把原本空间相邻的 token 拆进不同容器，容器间互不注意可能丢失跨容器关联。作者把 SPA 嵌入 Swin block，复用其移窗（shifted window）机制——每隔两个 transformer block 对特征图做一次移位，使得装进容器的 token 配对随之变化，多个 block 累积下来注意力计算就能覆盖到全部 token，弥补单次打包的局部性。

## 实验关键数据

### 主实验：COCO2017 目标检测（Cascade Mask RCNN，FLOPs 为训练阶段）

| 方法 | 注意力 | AP | AP50 | AP75 | overall FLOPs(G) | FPS |
|------|--------|----|----|----|----|----|
| Swin-T (dense) | Dense | 46 | 68.1 | 50.3 | 267 | 50 |
| DAT-T | Sparse | 44.4 | 67.6 | 48.5 | 272 | 46 |
| DynamicSwin-T | Sparse | 44.3 | 65.9 | 48.5 | 272 | 46 |
| **SPT-T (ours)** | Sparse | **47.1 (+2.7)** | 68.9 | 51.6 | **261 (-4.0%)** | 54 |
| Swin-S (dense) | Dense | 48.5 | 70.2 | 53.5 | 359 | 32 |
| **SPT-S (ours)** | Sparse | **49.3 (+2.1)** | 71 | 55.2 | **342 (-5.8%)** | 33 |
| Swin-B (dense) | Dense | 51.9 | 70.5 | 56.4 | 982 | 11 |
| **SPT-B (ours)** | Sparse | **53.2 (+2.7)** | 71.3 | 58.9 | **944 (-3.9%)** | 12 |

SPT 三种规模全面超过包括 dense Swin 在内的所有 baseline，相对最强稀疏 baseline 有 +2.1~2.7 AP；backbone 计算量降 10.9%~11.4%。BDD100K 上 +0.6~0.7 AP，backbone 计算降最高 24.9%；BDD-S（早期/小目标检测）上 SPT-T/-S 相对提升 19.1%/9.6%，backbone 算力降 20.8%~22.4%。

### 消融实验：选择设计与 SPA block 起始阶段（PASCAL VOC 多标签分类 / BDD100K）

| 设置 | Mean Select Ratio(%) | mAP |
|------|------|------|
| 均匀 top-50 选择（仿 SparseViT） | 50 | 44.42 |
| SPA 动态选择（无 $\mathcal{L}_{select}$） | 59.77 | 44.49 |
| SPA + $\mathcal{L}_{select}$ | **29.60** | **44.60** |

| SPA 起始阶段（BDD100K） | AP | AP50 | AP75 |
|------|----|----|----|
| 仅第 4 阶段 | 21.9 | 32.7 | 24.2 |
| 第 3-4 阶段 | **22.6** | **33.1** | **24.6** |
| 第 2-4 阶段 | 20.5 | 31.3 | 22.3 |
| 第 1-4 阶段 | 18.3 | 29.4 | 20.6 |

### 关键发现
- **动态 + 显式监督双重必要**：动态选择优于固定比例；加上 $\mathcal{L}_{select}$ 后选择比例从 ~60% 压到 29.6%，mAP 反而更高——监督让选择又准又省。
- **从第三阶段起选 token 最优**：太早（浅层特征区分度不足）会误删整物体，AP 从 22.6 一路掉到 18.3，印证 DAT 的观察。
- **越稀疏收益越大**：BDD-S 小目标场景算力下降幅度（20.8%）大于完整 BDD100K（16.8%），符合"只算含物体 token"的预期。
- 通用性：实例分割（COCO，SPT-S 39.6→40.9 AP）、多标签分类（VOC，44.12→44.60 mAP）同样获益。

## 亮点与洞察
- **把"恢复批并行"作为一等公民**：很多稀疏方法只报 FLOPs 下降却跑不出真吞吐，根因就是 token 数不齐导致 padding。Token Packing 直击这个工程痛点，让稀疏注意力的理论收益真正落到 FPS 上（SPT-T FPS 54 vs Swin-T 50）。
- **用现成物体标签监督 token 选择**：检测/分割本来就有 bbox/mask，作者顺手把它们聚合成选择标签，几乎零额外标注成本就把"挑 token"从难训的隐式任务变成可监督任务，解决了门控塌缩（给所有 token 打高分）的痛点。
- **多尺度取大保信息**：单尺度标签太激进，跨尺度逐元素取最大值是个简单但有效的"召回兜底"，在压低选择比例的同时不丢小物体。

## 局限与展望
- **整体效率受早期阶段瓶颈限制**：SPA 只用在后两阶段，而下采样使后期 token 数仅为早期的 1/16，即便 SPA 丢掉 78% 的 token，整体算力也只降约 16.8%——真正的计算大头仍在没被稀疏化的前两阶段，整网加速天花板有限。
- **依赖物体级标签**：多尺度选择监督来自 bbox/mask，迁移到无密集标注的任务（如自监督预训练、纯分类无定位信息）时如何构造选择标签尚不清楚。
- **Packing 的跨容器关联靠移窗间接补偿**：定长容器内"仅同图互注意"加上每两个 block 才移一次窗，长程跨容器依赖的保留程度仍是近似，论文未量化这部分潜在损失。
- 实验集中在检测/分割/多标签分类，未覆盖语义分割、视频等更长序列或更密集场景，普适性有待进一步验证。

## 相关工作与启发
- **稀疏注意力谱系**：相对 SparseViT（按 l2 norm 选、padding 到 max）、DynamicViT/EViT（推理期丢固定数量、训练仍全算）、DAT（缩感受野不减 token），本文的差异点在于**显式多尺度监督 + 打包恢复并行**这两点组合。
- **借来的两个外部 idea**：门控选择借鉴 MoE / 异构联邦学习的"门控选计算路径"；定长容器打包借鉴序列 packing（Dehghani et al. 2024）。这提示一个通用思路：把 NLP/MoE 里成熟的"路由 + 打包"机制迁到视觉稀疏注意力，能同时解效率与并行问题。
- **对后续工作的启发**：能否把稀疏化推到更早阶段（或设计浅层也可靠的选择标签）以突破 16.8% 的整体加速天花板，是个自然的延伸方向；此外"用任务自带标签监督内部稀疏决策"的范式或可推广到 token 合并、KV-cache 压缩等其他效率场景。

## 评分
- **新颖性**: ⭐⭐⭐⭐ — 多尺度物体标签监督 token 选择 + Token Packing 恢复批并行的组合是新的，尤其把"工程上的批并行"作为核心设计点而非事后优化，切中了稀疏注意力落地的真实痛点；单个组件（门控、Gumbel-Softmax、移窗、序列打包）多为借鉴。
- **实验充分度**: ⭐⭐⭐⭐ — 覆盖 COCO/BDD100K/BDD-S/VOC 四数据集、检测/实例分割/多标签分类三任务、T/S/B 三规模，且同时报告 AP、FLOPs、FPS、参数量与选择比例，消融把"动态 vs 固定""监督 vs 无监督""起始阶段"都拆清楚；不足是缺与更多近期 SOTA 稀疏方法的横向对比、缺方差/多次实验报告。
- **写作质量**: ⭐⭐⭐⭐ — 痛点（效率/性能两类失败）梳理清晰，复杂度公式与图示到位，方法动机（为何门控会塌缩、为何从第三阶段起）解释充分；公式与表格略密。
- **价值**: ⭐⭐⭐⭐ — 对小目标/稀疏数据检测这类实际场景给出了精度效率双赢的可即插即用骨干，且 FPS 真实提升，工程实用性强；天花板受限于只稀疏化后两阶段，潜力尚未完全释放。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Long-Context Generalization with Sparse Attention](long-context_generalization_with_sparse_attention.md)
- [\[AAAI 2026\] LampQ: Towards Accurate Layer-wise Mixed Precision Quantization for Vision Transformers](../../AAAI2026/object_detection/lampq_towards_accurate_layer-wise_mixed_precision_quantization_for_vision_transf.md)
- [\[AAAI 2026\] Temporal Object-Aware Vision Transformer for Few-Shot Video Object Detection](../../AAAI2026/object_detection/temporal_object-aware_vision_transformer_for_few-shot_video_object_detection.md)
- [\[CVPR 2026\] Tri-Modal Fusion Transformers for UAV-based Object Detection](../../CVPR2026/object_detection/tri-modal_fusion_transformers_for_uav-based_object_detection.md)
- [\[ICLR 2026\] DiffuDETR: Rethinking Detection Transformers with Denoising Diffusion Process](diffudetr_rethinking_detection_transformers_with_denoising_diffusion_process.md)

</div>

<!-- RELATED:END -->
