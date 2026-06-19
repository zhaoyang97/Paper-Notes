---
title: >-
  [论文解读] Dense Policy: Bidirectional Autoregressive Learning of Actions
description: >-
  [ICCV 2025][图像生成][自回归策略] 提出 Dense Policy，一种基于双向自回归扩展的机器人操作策略，通过对数时间的粗到细层次化动作生成，在仿真和真实世界任务中超越 Diffusion Policy 和 ACT 等主流生成式策略。 模仿学习中的动作生成范式分为两类： - 整体生成式策略：（ACT…
tags:
  - "ICCV 2025"
  - "图像生成"
  - "自回归策略"
  - "双向扩展"
  - "粗到细生成"
  - "扩散策略"
  - "模仿学习"
---

# Dense Policy: Bidirectional Autoregressive Learning of Actions

**会议**: ICCV 2025  
**arXiv**: [2503.13217](https://arxiv.org/abs/2503.13217)  
**代码**: [https://selen-suyue.github.io/DspNet/](https://selen-suyue.github.io/DspNet/)  
**领域**: 图像生成  
**关键词**: 自回归策略, 双向扩展, 粗到细生成, 扩散策略, 模仿学习

## 一句话总结

提出 Dense Policy，一种基于双向自回归扩展的机器人操作策略，通过对数时间的粗到细层次化动作生成，在仿真和真实世界任务中超越 Diffusion Policy 和 ACT 等主流生成式策略。

## 研究背景与动机

模仿学习中的动作生成范式分为两类：
- **整体生成式策略**（ACT, Diffusion Policy）：建模动作序列的联合分布，一次性生成所有动作。效果好但推理开销大
- **自回归策略**（ICRT, ARP）：逐Token或逐Chunk递增生成。虽然在语言和视觉中效果好，但在动作预测中表现次优

自回归策略在动作生成中的困境：
1. Next-token预测难以捕获长程时序依赖
2. Next-chunk预测的注意力范围有限
3. 现有多尺度方法（CARP）依赖离散化+码本构建，精度不足

核心观察：人类操作时并非逐步推理动作轨迹，而是**先规划几个关键帧，然后逐步细化**。这类似于视觉中的"感受野"概念——粗到细的感知过程。

## 方法详解

### 整体框架

Dense Policy 采用 encoder-only 架构，从初始的零向量开始，通过递归的"Dense Process"逐级双向扩展动作序列：
$$P(A|O) = \prod_{i=1}^{n} P(A^i | A^{i-1}, A^{i-2}, ..., A^0, O)$$

每级序列长度翻倍，经过 $\log_2 T$ 次递归后达到目标 horizon $T$。

### 关键设计

1. **Bidirectional Expansion（双向扩展）**: 核心是 Dense Process 机制。给定上一级的稀疏关键帧动作 $A^n$（含 $2^n$ 个动作点），通过线性上采样扩展到 $2^{n+1}$ 个点：

    - 已有位置保持原值
    - 新位置取两邻居的线性插值：$\tilde{a}_{t+j}^n = \frac{1}{2}(a_{t+j-T/2^{n+1}} + a_{t+j+T/2^{n+1}})$
    - 边界位置复制最近的原始点
   
   然后通过 4 层 BERT Encoder 进行交叉注意力得到下一级 $A^{n+1} = \text{Enc}(A_{up}^n, O)$。

   与 next-token 和 next-chunk 的关键区别：双向扩展能同时捕获前后方向的时序依赖，生成更连贯的动作序列。推理复杂度为 $O(\log T)$ 而非 $O(T)$。

2. **Encoder-Only 架构**: 使用共享的 BERT Encoder 处理所有层级，利用交叉注意力将视觉观测特征融入动作表示。好处：

    - 参数量小（对比Diffusion Head和CVAE Head）
    - 推理速度快（接近ACT，比Diffusion Policy快约10倍）
    - 训练稳定（无需VAE的变分推理或扩散模型的多步去噪）

3. **灵活的视觉输入**: 

    - 2D: ResNet18 + GroupNorm（RGB 图像）
    - 3D: Sparse Convolutional Network（点云）
    - 可无缝替换其他视觉backbone（如 RISE 的稀疏卷积网络）
    - 训练时随机mask部分本体感知信息避免位置记忆偏差

### 损失函数 / 训练策略

- 使用 L2 损失监督各级动作预测与ground truth的偏差
- 初始动作向量 $A^0 = \mathbf{0}$，提供无偏起点
- 训练本体感知时随机mask部分末端执行器姿态，提升泛化性
- 与对比方法使用相同训练迭代数和专家演示数

## 实验关键数据

### 主实验 (表格)

**仿真环境 (11 tasks, 3 benchmarks)：**

| Method | Adroit-Door | Adroit-Pen | DexArt-Laptop | DexArt-Toilet | MetaWorld-BinPick | MetaWorld-BoxClose | MetaWorld-Hammer | MetaWorld-PegInsert | MetaWorld-Disassemble | MetaWorld-ShelfPlace | MetaWorld-Reach | **Avg** |
|--------|------------|-----------|---------------|---------------|-------------------|-------------------|-----------------|--------------------|-----------------------|---------------------|----------------|---------|
| DP3 | 62±4 | 43±6 | 81±2 | 71±3 | 34±30 | 42±3 | 76±4 | 69±7 | 69±4 | 17±10 | 24±1 | 53±7 |
| **3D Dense** | **72±3** | **61±0** | **85±4** | **74±3** | **47±10** | **69±8** | **100±0** | **82±4** | **98±1** | **77±4** | **31±3** | **72±4** |
| DP | 37±2 | 13±2 | 31±4 | 26±8 | 15±4 | 30±5 | 15±6 | 34±7 | 43±7 | 11±3 | 18±2 | 25±5 |
| **2D Dense** | **59±8** | **65±1** | 28±7 | **36±8** | **25±2** | **51±3** | **86±4** | **60±7** | **71±6** | **59±6** | **27±4** | **52±5** |

3D Dense Policy 平均成功率比 DP3 高 **19%**，2D Dense Policy 比 Diffusion Policy 高 **27%**。

**真实世界 (4 tasks)：**

| Method | Put Bread | Open Drawer | Pour Balls (Complete) | Flower Arr. (Succ) | Flower (Avg Flowers) |
|--------|----------|------------|---------------------|-------------------|---------------------|
| ACT | 35% | 10% | 20% | - | - |
| Diffusion Policy | 40% | 20% | 20% | - | - |
| RISE | 75% | 40% | 25% | 50% | 0.6/3.0 |
| **2D Dense** | **55%** | **20%** | **25%** | - | - |
| **3D Dense** | **85%** | **45%** | **60%** | **70%** | **1.0/3.0** |

### 消融实验 (表格)

**不同自回归范式对比（学习效率与最终性能）：**

| Paradigm | Door | Bin Picking | Shelf Place | Box Close |
|----------|------|------------|-------------|-----------|
| Next-Token | 较低 | 较低 | 较低 | 较低 |
| Next-Chunk | 中等 | 中等 | 中等 | 中等 |
| **Bidirectional (Dense)** | **最高** | **最高** | **最高** | **最高** |

双向预测在所有四个有挑战性的任务上展现了更高的学习效率和更高的成功率天花板。

**推理时间与参数量对比：**

| Policy | 参数量 (Action Head) | 推理时间 |
|--------|---------------------|---------|
| ACT | 较大 | ~同 Dense |
| DP (Diffusion) | Dense + 9.19M | ~10× Dense |
| **Dense Policy** | **最小** | **最快** |

### 关键发现

- **双向依赖是关键**: 动作序列中不同时间步之间存在双向依赖，next-token 只能捕获单向关系
- **长程任务优势更大**: 在 Flower Arrangement（长程多物体）中 Dense Policy 相对优势最显著（成功率 70% vs 50%）
- **动作精度提升**: 在 Peg Insert Side（接触密集型）和 Pen Rotation（高自由度灵巧操作）中表现突出
- **训练更稳定**: 无需 VAE 的变分推理或扩散模型的多步去噪，避免了 ACT 的训练不稳定问题
- **对数时间推理**: $O(\log T)$ 复杂度，比 next-token 的 $O(T)$ 显著更快

## 亮点与洞察

- **粗到细的动作生成范式**: 类比人类先规划关键帧再细化的操作方式，提供了全新的动作序列建模视角
- **双向自回归首次用于动作空间**: 打破了自回归 = 单向的固有认知，证明双向扩展在连续动作空间中的有效性
- **encoder-only 架构的高效性**: 用共享Encoder处理多层级动作表示，显著减少参数
- **对数复杂度的推理**: 在不牺牲性能的前提下实现极快推理

## 局限与展望

- 未探索将 Dense Policy 扩展为通用 VLA（Vision-Language-Action）模型
- 在大规模基础模型上的扩展性和稳定性未验证
- 2D Dense Policy 在需要复杂空间推理的任务中仍受限于2D表示
- 倾倒球任务中偶尔出现夹爪过紧问题，自适应纠错能力待提升
- horizon T 需为2的幂次，限制了灵活性

## 相关工作与启发

- BERT 的双向上下文学习 → Dense Policy 的双向动作扩展
- MAR / SAR 的非光栅序自回归图像生成 → 动作空间的双向生成
- 与 Diffusion Policy、ACT 形成互补而非替代关系，指明了自回归策略的潜力
- 可启发将类似的层次化扩展策略用于轨迹规划、运动生成等领域

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 双向自回归动作生成是全新范式，粗到细层次化设计优雅
- **实验充分度**: ⭐⭐⭐⭐⭐ 3仿真基准11任务 + 4真实任务，2D/3D全覆盖，消融充分
- **写作质量**: ⭐⭐⭐⭐ 方法动机清晰，与现有工作区分到位
- **价值**: ⭐⭐⭐⭐⭐ 为自回归策略在机器人操作中的应用开辟了新方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Inpaint4Drag: Repurposing Inpainting Models for Drag-Based Image Editing via Bidirectional Warping](inpaint4drag_repurposing_inpainting_models_for_drag-based_image_editing_via_bidi.md)
- [\[CVPR 2026\] VA-π: Variational Policy Alignment for Pixel-Aware Autoregressive Generation](../../CVPR2026/image_generation/va-p_variational_policy_alignment_for_pixel-aware_autoregressive_generation.md)
- [\[ECCV 2024\] AdaGen: Learning Adaptive Policy for Image Synthesis](../../ECCV2024/image_generation/adagen_learning_adaptive_policy_for_image_synthesis.md)
- [\[ICCV 2025\] Holistic Tokenizer for Autoregressive Image Generation](holistic_tokenizer_for_autoregressive_image_generation.md)
- [\[NeurIPS 2025\] OverLayBench: A Benchmark for Layout-to-Image Generation with Dense Overlaps](../../NeurIPS2025/image_generation/overlaybench_a_benchmark_for_layout-to-image_generation_with_dense_overlaps.md)

</div>

<!-- RELATED:END -->
