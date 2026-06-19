---
title: >-
  [论文解读] Activation-Free Backbones for Image Recognition: Polynomial Alternatives within MetaFormer-Style Vision Models
description: >-
  [ICML2026][语义分割][激活函数替代] 本文用 Hadamard 乘积构造 PolyMLP、PolyConv 和 PolyAttn，替代 MLP、卷积和注意力中的点激活/softmax，在 MetaFormer 风格骨干中无需常规激活函数也能在 ImageNet、鲁棒性和 ADE20K 分割上达到或超过激活式模型。
tags:
  - "ICML2026"
  - "语义分割"
  - "激活函数替代"
  - "多项式网络"
  - "Hadamard乘积"
  - "MetaFormer"
  - "PolyNeXt"
---

# Activation-Free Backbones for Image Recognition: Polynomial Alternatives within MetaFormer-Style Vision Models

**会议**: ICML2026  
**arXiv**: [2605.20839](https://arxiv.org/abs/2605.20839)  
**代码**: https://github.com/jjwang8/PolyNeXt  
**领域**: 视觉骨干 / 图像识别 / 语义分割迁移  
**关键词**: 激活函数替代、多项式网络、Hadamard乘积、MetaFormer、PolyNeXt  

## 一句话总结
本文用 Hadamard 乘积构造 PolyMLP、PolyConv 和 PolyAttn，替代 MLP、卷积和注意力中的点激活/softmax，在 MetaFormer 风格骨干中无需常规激活函数也能在 ImageNet、鲁棒性和 ADE20K 分割上达到或超过激活式模型。

## 研究背景与动机
**领域现状**：现代视觉骨干几乎默认依赖 ReLU、GELU、SiLU 等点激活函数，以及 self-attention 中的 softmax 指数归一化。ConvFormer、CAFormer、ConvNeXt、ViT 等架构都把这些非线性作为高性能视觉表示的基础组件。

**现有痛点**：激活函数并不是唯一的非线性来源。已有 polynomial networks 说明乘法交互也能表达复杂函数，但很多方法需要从头设计专门架构，难以复用已有 MetaFormer/attention/conv 改进；同时深层多项式网络容易因为乘法放大而训练不稳定。

**核心矛盾**：如果直接去掉激活函数，模型可能缺少非线性或训练崩溃；如果保留复杂自定义多项式结构，又很难成为通用视觉模块。论文要证明的是：只替换标准模块里的非线性算子，同时保持接口不变，是否足够训练出竞争力 backbone。

**本文目标**：作者希望设计一套 activation-free 的 channel mixing、spatial convolution mixing 和 attention mixing 模块，使其可以插入 MetaFormer 风格架构，兼顾 ImageNet 分类、OOD 鲁棒性、ADE20K 语义分割和面向 FHE 的多项式推理潜力。

**切入角度**：Hadamard product 本身会产生输入的二阶多项式，层层堆叠后多项式次数随深度指数增长。只要控制残差幅度和梯度流，深度而窄的多项式网络可以获得足够表达力，而不需要点激活函数。

**核心 idea**：用“平行线性/卷积分支的逐元素乘法 + 稳定化残差设计”替代标准激活函数，让视觉 backbone 的非线性来自可组合的多项式交互。

## 方法详解
本文的核心是把常见视觉骨干里的三个非线性来源逐一改造成多项式模块。MLP 中的 GELU 被两个线性投影的 Hadamard product 替代；separable convolution 中的激活被粗/细两条卷积分支的乘法融合替代；attention 中的 softmax 指数核被多项式 kernel 替代。然后作者把这些模块组装成 PolyNeXt，并加入 Sigmoid-Scale、多输入 skip、depth-over-width 等稳定化策略。

### 整体框架
PolyNeXt 采用四阶段层级视觉 backbone，整体仍遵循 MetaFormer 模板：每个 cell 接收前两个 cell 的输出，先经过空间 mixer，再经过 PolyMLP。CPolyNeXt 在所有阶段使用 PolyConv；APolyNeXt 在前两阶段用 PolyConv 处理高分辨率局部信息，在后两阶段用 PolyAttn 处理低分辨率全局信息。Stem 是 stride 4 的 $7\times7$ 卷积，阶段之间用 stride 2 卷积下采样。

一个 cell 内可以包含多个 stack，每个 stack 是“空间混合器 + PolyMLP”。作者强调 depth-over-width（深而窄）：与其把单层做宽，不如堆更多较窄的多项式层，因为多项式次数随层数增长更快。为了避免乘法链路导致数值爆炸，每个残差分支都用可学习的 sigmoid 标量（Sigmoid-Scale）限制输出幅度。下图给出一个 PolyNeXt cell 的完整数据流，对应下面四个关键设计：

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    A["输入图像 → Stem(7×7 stride-4 卷积)"] --> B["4 阶段层级骨干<br/>阶段间 stride-2 卷积下采样"]
    B --> C["多输入 skip：取前两个 cell 输出<br/>per-channel 缩放相加 → LayerNorm"]
    C --> STACK
    subgraph STACK["PolyNeXt stack（每个 cell 深堆 X 个，depth-over-width）"]
        direction TB
        D["空间混合器"] -->|前两阶段·高分辨率| E["PolyConv<br/>粗(空洞)/细分支翻转通道后逐元素相乘"]
        D -->|后两阶段·低分辨率| F["PolyAttn<br/>多项式核替代 softmax + ℓ1 归一化"]
        E --> G["PolyMLP<br/>两线性投影逐元素相乘"]
        F --> G
    end
    STACK -->|每个残差分支 y=x+σ(λ)·f(x)| H["Sigmoid-Scale 限幅"]
    H --> I["分类头 / ADE20K UperNet 分割"]
```

### 关键设计
**1. PolyMLP：用 Hadamard 乘积替代 channel mixing 的激活**
标准 FFN 在两个线性投影之间插一个 GELU 提供非线性，GLU 类变体虽加了乘法交互但仍在一支保留激活。PolyMLP 彻底去掉激活：$\text{PolyMLP}(x)=W_o((W_a x)*(W_b x))$——把输入投影到中间维度的两支 $W_a x$、$W_b x$ 逐元素相乘（乘后接 LayerNorm），再投影回原维度。这一乘法恰好得到输入的二阶多项式，层层堆叠后多项式次数随深度快速增长，因此不靠点激活也能积累足够非线性。它为什么有效还藏着一个反直觉点：反向传播时 $W_a$ 的梯度被 $W_b x$ 缩放、反之亦然，两支通过对方的输出互相学习（mutual gradient coupling）；一旦给某支加上 GELU，其负区间的近零导数会切断这种耦合——这正解释了消融里“加回激活反而掉点”的现象。

**2. PolyConv：异质感受野双分支相乘替代可分离卷积的激活**
MetaFormer 的 ConvFormer 用可分离卷积（depthwise 空间滤波 + pointwise 混合），中间也夹一个激活。要把这个激活换成 Hadamard 乘积，光靠两个同构分支相乘交互不够丰富。PolyConv 先用 pointwise 卷积得到 hidden feature，再分出两条感受野不同的 depthwise 分支：粗分支用空洞卷积（$5\times5$ kernel、dilation 2，覆盖 $9\times9$）抓大范围上下文，细分支用标准 $3\times3$ 抓局部细节；融合前对其中一支做 channel-flip（翻转通道顺序）进一步解耦，两支逐元素相乘后再用 $3\times3$ 卷积整合、pointwise 投影输出。这样设计的关键在于：异质感受野相乘能显式产生跨尺度交互项，比 MONet/DTTN 那种两支同构的做法更有表达力——消融中去掉粗分支、或换回标准可分离卷积都明显掉点。

**3. PolyAttn：多项式核替代 softmax 的指数**
self-attention 里的 softmax 依赖指数函数，它既是一处必须保留的非线性，也阻碍完全多项式（FHE 友好）推理。PolyAttn 把未归一化权重写成 $A=(s\cdot QK^\top+1)^p$（$p=4$，$s=\sigma(\lambda)$ 是每个 head 可学习的 scale），再用 $\ell_1$ 归一化替代 softmax 归一化；同时仿照 PolyConv，在 $Q,K,V$ 上加 depthwise 卷积注入局部空间上下文，并共享 $Q/K$ 投影省参数。它保留了 query-key 相似度加权的注意力语义却避开指数，且只改动 kernel、接口不变，因此仍可兼容 window/sparse attention。消融揭示了一个细节：把核单换成 softmax 只差 0.1 点，但整体换成标准注意力掉 1.3 点——说明共享 $Q/K$ 投影与 depthwise 卷积比核本身贡献更大。

**4. 深层多项式网络的稳定化配方**
与 ReLU 不同，Hadamard 乘积会把两个大值相乘成更大的值，这种放大沿深度累积，深网若直接训练就会发散。作者用三件套撑起接近 200 层的训练：① Sigmoid-Scale 把每个残差分支写成 $y=x+\sigma(\lambda)f(x)$，用 sigmoid 限幅的可学习标量约束残差幅度，并按深度初始化更小的贡献（消融表明起决定作用的是初始化几何，换成标准初始化的 LayerScale 会直接训练崩溃掉 12.8 点）；② multi-input skip（仿 NASNet）让每个 cell 同时接收前一个和前两个 cell 的输出，经可学习 per-channel 缩放相加后 LayerNorm，改善梯度流；③ depth-over-width：相近参数量下堆更多较窄的 stack 而非加宽单层，因为多项式次数随深度指数增长（3 stack/cell 比 1 stack 高 1.5 点）。三者合起来才让多项式模块能真正吃到“深度”的表达力红利。

### 损失函数 / 训练策略
模型按 ImageNet-1K 监督分类训练，训练 recipe 基于 MetaFormer/MONet 但使用更小 batch size 和更强 regularization。语义分割迁移使用 UperNet，在 ADE20K 上训练 160K iterations，采用 ConvNeXt recipe，并对 Sigmoid-Scale、多输入 skip 和 normalization 参数设特殊 weight decay 分组。论文还训练 LayerNorm 替换为 polynomial-compatible BatchNorm 的 fully polynomial 变体，以探索 FHE 友好推理。

## 实验关键数据

### 主实验
ImageNet-1K 主结果说明，PolyNeXt 在不同规模上都能接近或超过激活式 MetaFormer，也明显强于 prior polynomial networks。

| 模型 | Params | FLOPs | Top-1 | 说明 |
|------|--------|-------|-------|------|
| DTTN-T | 7.1M | 2.4G | 77.9 | prior polynomial tiny |
| MONet-T | 10M | 2.8G | 77.0 | prior polynomial tiny |
| CPolyNeXt-T | 6.4M | 1.2G | 80.2 | 更少参数/FLOPs 下高 2-3 点 |
| ConvFormer-S18 | 27M | 3.9G | 83.0 | 激活式 MetaFormer conv baseline |
| CPolyNeXt-S | 26M | 4.8G | 83.9 | 高 0.9 点 |
| DTTN-B | 36M | 12.3G | 82.4 | prior polynomial base |
| CPolyNeXt-B | 40M | 8.5G | 84.7 | 比 DTTN-B 高 2.3 点且 FLOPs 更低 |
| CAFormer-S18 | 26M | 4.1G | 83.6 | 激活式 hybrid baseline |
| APolyNeXt-S | 26M | 5.3G | 84.3 | 高 0.7 点 |
| CAFormer-M36 | 56M | 13.2G | 85.2 | 大模型 hybrid baseline |
| APolyNeXt-L | 57M | 13.3G | 85.2 | 持平 |

鲁棒性和下游分割结果也支持 polynomial backbone 的泛化。

| 任务 | 模型 | Clean / 主指标 | OOD / 下游指标 | 结论 |
|------|------|----------------|----------------|------|
| ImageNet-C/A/R/Sketch | CAFormer-S18 | 83.6 clean, IN-C 47.4, IN-A 33.5 | IN-R 48.7, IN-Sk 36.6 | 强 hybrid baseline |
| ImageNet-C/A/R/Sketch | APolyNeXt-S | 84.3 clean, IN-C 45.0, IN-A 39.6 | IN-R 49.7, IN-Sk 37.5 | clean 与鲁棒性同步提升，mCE 更低 |
| ADE20K UperNet | ConvFormer-S18 | 54M, 925G | 48.6 mIoU | MetaFormer conv baseline |
| ADE20K UperNet | CAFormer-S18 | 54M, 1024G | 48.9 mIoU | MetaFormer hybrid baseline |
| ADE20K UperNet | CPolyNeXt-S | 54M, 941G | 50.6 mIoU | 比 ConvFormer-S18 高 2.0 |
| ADE20K UperNet | APolyNeXt-S | 55M, 1121G | 49.9 mIoU | 比 CAFormer-S18 高 1.0 |

### 消融实验
消融直接检验“激活函数是否必要”和“稳定化是否关键”。

| 配置 | Δ Acc | 说明 |
|------|-------|------|
| CPolyNeXt-T baseline | 80.2 | 完整多项式卷积模型 |
| PolyMLP → MLP+GELU | -0.1 到 -0.4 | 加回 MLP 激活没有帮助 |
| PolyConv → SepConv+GELU | -0.9 | 标准 separable conv 更差 |
| 在一支乘法分支加 GELU | -0.4 | 破坏部分互梯度耦合 |
| 在乘积后加 GELU | -1.0 | 单个 gate 同时阻断两支梯度 |
| Hadamard → Addition | -22.3 | 乘法交互是核心非线性来源 |
| APolyNeXt-T baseline | 80.9 | 完整多项式注意力模型 |
| PolyAttn → Std Attn | -1.3 | 标准注意力替代整体结构明显更差 |
| polynomial kernel → softmax | -0.1 | kernel 本身不是唯一贡献，Q/K 共享和局部卷积也重要 |

| 稳定化/架构消融 | Δ Acc | 说明 |
|-------------------|-------|------|
| Sigmoid-Scale → free scalar | -0.5 | 初始化几何最关键，sigmoid 还有次级优化收益 |
| Sigmoid-Scale → LayerScale init=1e-6 | -0.8 | 传统 LayerScale 不够适配 |
| Sigmoid-Scale → LayerScale init=1.0 | -12.8 | 训练几乎崩溃 |
| 移除 multi-input skip | -0.6 | 跨 cell 梯度流有贡献 |
| 移除 cell 前 norm | -0.4 | 归一化位置重要 |
| 更宽 2 stacks/cell | -0.7 | 深度优于宽度 |
| 更宽 1 stack/cell | -1.5 | 多项式次数不足 |

### 关键发现
- 激活函数在这个设计里不是越多越好。加回 GELU 往往降低性能，说明乘法分支之间的 mutual gradient coupling 是有效非线性来源。
- Hadamard product 不可替代。换成 addition 掉 22.3 点，基本证明模型不是靠外壳结构，而是靠乘法交互表达。
- 稳定化是成败关键。没有合理残差尺度，深层多项式网络会因乘法放大而不稳定；Sigmoid-Scale 和多输入 skip 让接近 200 层训练成为可能。
- 分割迁移收益比分类边际更明显。CPolyNeXt-S 在 ADE20K 上比 ConvFormer-S18 高 2.0 mIoU，说明多项式 backbone 学到的表示不仅服务分类。

## 亮点与洞察
- 论文最有价值的点不是“再造一个新 backbone”，而是把 activation-free 设计做成标准 MLP/Conv/Attention 的接口级替换。这让它能继承 MetaFormer 生态，而不是孤立架构。
- 对“为什么激活会伤害”的解释很有启发：乘法的两支投影在反向传播中互相调制，GELU 的负区间会切断这种耦合；这和我们通常“激活增加表达力”的直觉相反。
- FHE 视角让工作不只是性能论文。完全多项式 BN 版本仍能达到 CPolyNeXt-S BN 82.7%，超过 ConvNeXt-T，这说明隐私计算友好网络不一定只能牺牲大量精度。
- Depth-over-width 的结论可迁移到其他 multiplicative architectures。乘法网络的能力来自可组合次数，而不是单层宽度，设计空间和常规 ReLU 网络不同。

## 局限与展望
- 训练 recipe 不完全通用。作者承认需要更小 batch、更强 regularization、渐进 dropout 和谨慎初始化，直接套标准训练配置可能不稳定。
- 深而窄的设计有吞吐开销。即使 FLOPs 接近，实际速度可能慢于更浅更宽的 MetaFormer。
- Hadamard product 对学习率敏感，乘法放大让超参调节更脆弱。
- Fully polynomial 版本仍只是朝 FHE 迈进，真正端到端加密推理还要解决归一化、注意力归一、硬件和数值范围等问题。
- 论文主要在 ImageNet/ADE20K 验证，迁移到检测、实例分割、多模态视觉编码器或视频 backbone 还需要进一步实验。

## 相关工作与启发
- **vs MONet / DTTN**: 这些 prior polynomial networks 更依赖定制架构，PolyNeXt 只替换标准模块中的非线性，性能更高且更容易迁移到 MetaFormer。
- **vs ConvFormer / CAFormer**: 二者依赖 separable conv、gated MLP 和 softmax attention；本文保留整体模板但把激活换成多项式交互，在同规模上匹配或超过它们。
- **vs StarNet / GLU**: StarNet 和 GLU 也用逐元素乘法，但仍保留激活；本文强调完全去掉点激活后，乘法本身足以提供非线性。
- **vs linear attention / efficient attention**: PolyAttn 不是单纯追求线性复杂度，而是替换 softmax 的指数核为多项式核；它可以与 window/sparse attention 等结构进一步结合。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ 激活替代思路简洁但很系统，尤其是 PolyConv/PolyAttn 与稳定化组合有辨识度。
- 实验充分度: ⭐⭐⭐⭐☆ ImageNet、鲁棒性、ADE20K、FHE 变体和消融都较完整；检测/视频等任务还可补强。
- 写作质量: ⭐⭐⭐⭐☆ 结构清楚、表格扎实，对 activation hurt 的分析有洞察；部分附录配置较多。
- 价值: ⭐⭐⭐⭐☆ 对视觉 backbone、隐私计算友好网络和 multiplicative architecture 设计都有启发。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] InstructSAM: A Training-Free Framework for Instruction-Oriented Remote Sensing Object Recognition](../../NeurIPS2025/segmentation/instructsam_a_training-free_framework_for_instruction-oriented_remote_sensing_ob.md)
- [\[CVPR 2025\] Style-Editor: Text-driven Object-Centric Style Editing](../../CVPR2025/segmentation/style-editor_text-driven_object-centric_style_editing.md)
- [\[CVPR 2026\] PIX-TAB: Efficient PIXel-Precise TABle Structure Recognition Approach with Speculative Decoding and Region-Based Image Segmentation](../../CVPR2026/segmentation/pix-tab_efficient_pixel-precise_table_structure_recognition_approach_with_specul.md)
- [\[CVPR 2026\] GKD: Generalizable Knowledge Distillation from Vision Foundation Models for Semantic Segmentation](../../CVPR2026/segmentation/gkd_generalizable_knowledge_distillation_vfm.md)
- [\[CVPR 2026\] The Missing Point in Vision Transformers for Universal Image Segmentation](../../CVPR2026/segmentation/the_missing_point_in_vision_transformers_for_universal_image_segmentation.md)

</div>

<!-- RELATED:END -->
