# DenseNets Reloaded: Paradigm Shift Beyond ResNets and ViTs

**会议**: ECCV 2024  
**arXiv**: [2403.19588](https://arxiv.org/abs/2403.19588)  
**代码**: [https://github.com/naver-ai/rdnet](https://github.com/naver-ai/rdnet)  
**领域**: 网络架构设计  
**关键词**: DenseNet, 密集连接, 特征拼接, ConvNeXt, 现代化改造  

## 一句话总结
重新审视 DenseNet 的密集拼接连接（concatenation shortcut），通过系统性现代化改造（加宽减深、现代化 block、扩大中间维度、更多 transition 层等），提出 RDNet（Revitalized DenseNet），在 ImageNet-1K 上超越 Swin Transformer、ConvNeXt、DeiT-III，证明了拼接连接作为一种被低估的范式具有强大潜力。

## 背景与动机
- ResNet 的加法残差连接（additive shortcut）成为现代视觉架构的标配（Swin、ConvNeXt、ViT 等），而 DenseNet 的拼接连接（concatenation）逐渐被遗忘
- DenseNet 早期超越了 ResNet，但因内存问题（拼接导致特征维度膨胀）和宽度缩放困难而失去发展势头
- 作者的核心猜想：**拼接连接是一种更有效的增加矩阵秩的方式**。对于 $f(XW)$，拼接 $[X, f(XW)]$ 的秩为 $\text{rank}(X) + \text{rank}(f(XW))$，而加法 $X + f(XW)$ 的秩不超过 $\text{rank}(X)$
- Pilot study：在 Tiny-ImageNet 上采样 15k+ 网络对比，拼接连接平均准确率 **56.8±3.9** vs 加法连接 **55.9±4.1**

## 核心问题
DenseNet 的密集拼接连接是否真的不如 ResNet 式的加法连接？能否通过现代化设计使 DenseNet 重回 SOTA？

## 方法详解

### 整体框架
从 DenseNet-201 出发，逐步进行 7 项现代化改造，最终得到 RDNet 系列模型。每个 stage 由若干 mixing block 组成，每个 mixing block 包含 3 个 feature mixer + 1 个 transition layer。保留密集拼接连接作为核心原则。

### 关键设计

1. **加宽减深（Wider & Shallower）**：将 growth rate (GR) 从 32 大幅增加到 120，深度从 (6,12,48,32) 减少到 (3,3,12,3)。训练速度提升约 35%，内存减少 18%。准确率仅下降 0.2pp（79.7→79.5）

2. **现代化 block（Improved Feature Mixers）**：采用 ConvNeXt 风格的 block 设计——Layer Norm 替代 Batch Norm、post-activation、7×7 深度可分离卷积、更少的归一化/激活。准确率提升 0.9pp（79.5→80.4）

3. **扩大中间维度（Larger Intermediate Dimensions）**：将 expansion ratio (ER) 从绑定于 GR 改为绑定于输入维度（解耦 ER 和 GR），同时将 GR 从 120 减半到 60。训练速度提升 21%，准确率再提 0.4pp

4. **更多 transition 层**：不仅在 stage 之间放 transition 层，还在 stage 内部每 3 个 block 后加一个 stride=1 的 transition 层做维度缩减（不下采样）。大幅降低计算量，允许进一步增大 GR。配合不同 stage 使用不同 GR（如 64,104,128,224）

5. **Patchification stem**：使用 patch_size=4, stride=4 的 patchification（类似 ConvNeXt），加速计算且不损精度

6. **改进的 transition 层**：去掉 average pooling，用调整 kernel size 和 stride 的卷积替代，并用 LN 替换 BN。+0.2pp

7. **Channel re-scaling**：融合 channel layer-scale 和 squeeze-excitation 的通道重缩放。+0.2pp

### RDNet 模型族配置

| 模型 | GR | Blocks (B) |
|------|-----|-----------|
| RDNet-T | (64,104,128,224) | (3,3,12,3) |
| RDNet-S | (64,128,128,240) | (3,3,21,6) |
| RDNet-B | (96,128,168,336) | (3,3,21,6) |
| RDNet-L | (128,192,256,360) | (3,3,24,6) |

## 实验关键数据

### ImageNet-1K 分类

| 模型 | 参数量 (M) | FLOPs (G) | Top-1 Acc (%) |
|------|-----------|----------|--------------|
| ConvNeXt-T | 29 | 4.5 | 82.1 |
| Swin-T | 28 | 4.5 | 81.3 |
| DeiT-III-S | 22 | 4.6 | 81.4 |
| **RDNet-T** | ~25 | ~5.0 | **82.8** |
| ConvNeXt-S | 50 | 8.7 | 83.1 |
| Swin-S | 50 | 8.7 | 83.0 |
| **RDNet-S** | ~40 | ~8.7 | **83.7** |
| ConvNeXt-B | 89 | 15.4 | 83.8 |
| Swin-B | 88 | 15.4 | 83.5 |
| **RDNet-B** | ~87 | ~15.4 | **84.4** |
| **RDNet-L** | ~186 | ~34.7 | **84.8** |

### 渐进式改造效果（从 DenseNet-201 到 RDNet）

| 步骤 | Top-1 (%) | 训练延迟 (ms) | 训练内存 (GB) |
|------|-----------|-------------|------------|
| (a) DenseNet-201 baseline | 79.7 | 131 | 3.9 |
| (b) +Wider & shallower | 79.5 | 85 (-35%) | 3.2 (-18%) |
| (c) +Modernized blocks | 80.4 | - | - |
| (d) +Larger intermediate dims | 80.8 | - | - |
| (e) +More transition layers | 81.2 | - | - |
| (f) +Patchification | ~81.2 | - | - |
| (g) +Refined transition | ~81.4 | - | - |
| (h) +Channel re-scaling | ~81.6 | - | - |

### 下游任务

- **ADE20K 语义分割**：RDNet 作为 backbone 在 UperNet 框架下优于 ConvNeXt 和 Swin
- **COCO 目标检测/实例分割**：RDNet 作为 backbone 在 Mask R-CNN 框架下竞争力强

### 消融实验要点
- **宽度 vs 深度**：优先增加宽度（GR）比增加深度更高效
- **Expansion ratio**：ER 绑定输入维度（而非 GR）是最优选择；ER=4 效果最好
- **Transition 间隔**：频繁使用 transition 层（每 3 block）通常优于仅在 stage 之间使用
- **统一 vs 可变 GR**：不同 stage 使用不同 GR 优于统一 GR
- **Transition ratio**：0.5（原始 DenseNet 设置）仍是最优，高 ratio 会损害精度

## 亮点
- **挑战主流范式**的勇气：在加法残差连接已成"共识"的时代，证明拼接连接同样甚至更强
- 用矩阵秩理论给出拼接优于加法的理论支撑，且有 15k 网络的 pilot study 经验验证
- 渐进式改造路线图清晰，每一步改动都有消融验证，可复用性强
- 最终 RDNet 在 ImageNet-1K 上近 SOTA，并在多个下游任务上表现优异

## 局限性
- 拼接连接的内存开销仍高于加法连接，尽管通过 transition 层缓解但未根本解决
- 频繁 transition 层的设计增加了模型复杂度和超参数
- 未在大规模预训练（如 ImageNet-22K）和更多下游任务上充分验证
- 在移动端等计算受限场景下性能不确定

## 与相关工作的对比
- **vs ConvNeXt**：都是"现代化改造"经典架构，但 ConvNeXt 改造 ResNet（加法连接），本文改造 DenseNet（拼接连接），RDNet 在同等 FLOPs 下约高 0.5-0.7pp
- **vs VOVNet**：VOVNet 也用了拼接但走向简化（one-shot aggregation），本文保留完整密集连接并通过 transition 层管理
- **vs Swin Transformer**：纯 CNN 架构超越了 Swin，说明局部注意力并非唯一有效的信息聚合方式

## 启发与关联
- **feature concatenation 作为增秩手段**：这个理论视角很有启发性，可以推广到其他需要保持特征多样性的场景
- DenseNet 风格的密集连接在语义分割等需要多尺度信息的下游任务上可能有更显著优势
- 频繁 transition 层的设计思路可借鉴到其他需要控制特征维度的架构中

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 挑战了"加法连接是最优"的主流认知，理论与实验俱到
- 实验充分度: ⭐⭐⭐⭐ 覆盖分类/分割/检测，但大规模预训练实验不足
- 写作质量: ⭐⭐⭐⭐ 渐进式改造叙述清晰，pilot study 有说服力
- 对我的价值: ⭐⭐⭐⭐ 提供了一种被忽视的架构范式，对网络设计有启发
