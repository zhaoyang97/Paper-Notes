---
title: >-
  [论文解读] VGGT-DP: Generalizable Robot Control via Vision Foundation Models
description: >-
  [AAAI 2026][3D视觉][视觉运动策略] 提出 VGGT-DP，一个受生物视觉系统启发的视觉运动策略框架，将预训练的 3D 感知基础模型 VGGT 作为视觉编码器并与扩散策略（Diffusion Policy）结合，通过帧级 Token 复用机制、随机 Token 裁剪和本体感知引导视觉学习三个关键设计，在 MetaWorld 高精度操作任务上显著超越 DP 和 DP3 基线。
tags:
  - "AAAI 2026"
  - "3D视觉"
  - "视觉运动策略"
  - "扩散策略"
  - "VGGT"
  - "本体感知引导"
  - "机器人操作"
---

# VGGT-DP: Generalizable Robot Control via Vision Foundation Models

**会议**: AAAI 2026  
**arXiv**: [2509.18778](https://arxiv.org/abs/2509.18778)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 视觉运动策略, 扩散策略, VGGT, 本体感知引导, 机器人操作

## 一句话总结

提出 VGGT-DP，一个受生物视觉系统启发的视觉运动策略框架，将预训练的 3D 感知基础模型 VGGT 作为视觉编码器并与扩散策略（Diffusion Policy）结合，通过帧级 Token 复用机制、随机 Token 裁剪和本体感知引导视觉学习三个关键设计，在 MetaWorld 高精度操作任务上显著超越 DP 和 DP3 基线。

## 研究背景与动机

### 从生物视觉到机器人感知

视觉运动策略（visuomotor policy）是机器人操作的核心。当前主流的研究范式有两类：

**Vision-Action (VA)** 范式：小型视觉编码器 + 大型策略头，如 Diffusion Policy。

**Vision-Language-Action (VLA)** 范式：利用大规模视觉-语言模型提供语言先验，增强泛化。

然而，作者提出了一个深刻的生物学洞察：**许多非语言生物体也展现出超群的操作能力**。昆虫、果蝇甚至单细胞生物在没有任何语言/符号推理的情况下，也能出色地感知、导航和操作环境。生物学研究表明，动物大量的神经资源专门用于**视觉处理**。

因此，核心问题不在于语言先验，而在于**视觉表示的容量和质量**。当前机器人系统中使用的视觉编码器往往过于简单，难以捕捉复杂的空间和几何关系。

### 为什么选择 VGGT

VGGT（Visual Geometry Grounded Transformer）是在大规模 3D 重建任务上预训练的视觉基础模型，能统一预测相机位姿、稠密深度图、3D 点云和视觉特征。与 CLIP/DINOv2 等关注语义的模型不同，VGGT 提供了**几何感知的空间表示**，更适合需要精确空间推理的操作任务。

## 方法详解

### 整体框架

VGGT-DP 由三个核心组件构成：
1. **VGGT 编码器 + Token 裁剪**：提取几何感知的视觉特征
2. **帧级 Token 复用机制（FTR）**：减少推理延迟
3. **本体感知引导的扩散策略**：融合视觉和本体感知信号进行动作预测

### 关键设计

#### 1. **VGGT 作为特征投影器**

不使用 VGGT 的低层视觉输出（深度图、点云），而是利用其**聚合器（aggregator）输出的 token**，这些 token 是紧凑的、语义丰富的 3D 场景表示。

给定 $B \cdot T$ 帧、$V$ 个视角的输入图像，VGGT 输出视觉 token：
$$\mathcal{T}_{vggt} = \text{VGGT}_{agg}(\mathcal{I}) \in \mathbb{R}^{B \cdot T \times V \times (N_p+1) \times D}$$

接着通过 Transformer Encoder 进一步处理，再经过平均池化和 MLP 投影得到条件嵌入 $\mathcal{C} \in \mathbb{R}^{B \cdot T \times d_c}$，作为扩散策略的条件输入。

设计动机：VGGT 聚合器已经融合了多视角的空间和外观线索，产生了具有全局上下文的几何感知 token，比原始图像特征更强。

#### 2. **帧级 Token 复用机制（Frame-Wise Token Reuse, FTR）**

现有方法在每个推理步骤中对所有观测帧重新计算视觉嵌入，即使帧在时间窗口间是重叠的。对于大模型如 VGGT，计算开销巨大。

FTR 的核心思路：**对重叠帧复用已计算的 token**。仅对最新帧计算新的 VGGT 特征，旧帧的 token 缓存在 CPU 上：
$$\mathcal{T}_{vggt}^{(t)} = \text{Concat}(\mathcal{T}_{cache}^{(t-1)}, \text{VGGT}_{agg}(\mathcal{I}_t))$$

设计动机：在机器人控制的滑动窗口中，大部分观测帧在相邻时间步之间是共享的。FTR 将推理开销从 $O(T)$ 降为 $O(1)$。

#### 3. **随机 Token 裁剪（Random Token Pruning）**

在将 VGGT token 输入 Transformer Encoder 之前，随机丢弃一定比例 $r_{prune}$ 的 patch token。

设计动机：引入 token 级随机性防止过拟合、减少计算量加速推理，同时鼓励模型学习对部分观测丢失不变的表示。

#### 4. **本体感知引导的视觉学习**

设计一个辅助解码器 $D$，从视觉特征中预测机器人的本体感知状态（关节角度 + 末端执行器位置）：
$$\hat{p}_t = D(f_t), \quad \mathcal{L}_{proprio} = \mathbb{E}_t[\|p_t - \hat{p}_t\|^2]$$

设计动机：迫使视觉编码器学习与操作相关的空间信息特征，改善闭环反馈控制质量。

### 训练策略

- 架构：U-Net-1D 扩散模型，使用 FiLM 条件注入
- 调度器：DDIM，100 训练时间步，10 推理去噪步
- 预测窗口：16 步，观测窗口：2 步
- 优化器：AdamW，lr=$1 \times 10^{-4}$，权重衰减 $1 \times 10^{-6}$
- 训练 3000 epochs，batch size 128，EMA 衰减 0.9999

## 实验关键数据

### 主实验

**MetaWorld 10 个挑选任务的成功率（%）**：

| 任务 | DP | DP3 | **VGGT-DP** | 类型 |
|------|-----|-----|-------------|------|
| Disassemble | 43±7 | **69±4** | 55±2.5 | 简单 |
| Peg Unplug Side | 74±3 | **75±5** | 63±6 | 简单 |
| Pick out of Hole | 0±0 | 14±9 | **55±6** | 复杂空间 |
| Shelf Place | 11±3 | **17±10** | 10±0 | 放置 |
| Reach | 18±2 | 24±1 | **42±8** | 复杂空间 |
| Soccer | 14±4 | 18±3 | **30±7** | 复杂空间 |
| Sweep Into | 10±4 | 15±5 | **44±4** | 复杂空间 |
| Hand Insert | 10±4 | 15±5 | **19±4** | 复杂空间 |
| Pick Place | 0±0 | **12±4** | 0±0 | 放置 |
| Stick Pull | 11±2 | 27±8 | **48±5** | 复杂空间 |
| **平均** | 19.1 | 28.6 | **36.6** | — |

VGGT-DP 平均成功率 36.6%，比 DP 高 17.5 个百分点，比 DP3 高 8.0 个百分点。

### 消融实验

**视角扰动鲁棒性测试（Stick Pull 任务）**：

| 扰动角度 δ | 成功率 | 说明 |
|------------|--------|------|
| 0° | 39% | 正常视角 |
| 5° | 5% | 轻微扰动即断崖式下降 |
| 10° | 0% | 完全失败 |
| 15° | 0% | 完全失败 |

**FTR 机制效果**：FTR 在大 batch 和长时序窗口下显著降低推理延迟，对部署大型视觉模型到实时系统有重要价值。

### 关键发现

1. **VGGT-DP 在复杂空间推理任务上表现突出**：Pick out of Hole（0→55%）、Sweep Into（10→44%）、Stick Pull（11→48%）
2. **在简单任务上无优势**：小型编码器即可胜任
3. **放置任务失败**：目标物体小/细长/部分遮挡时无法精确定位
4. **视角鲁棒性极差**：仅 5° 扰动即导致成功率断崖式下降（39%→5%），严重过拟合训练时的相机位姿

## 亮点与洞察

1. **生物学启发的深刻洞察**：绕过语言先验、回归视觉感知本质的思路很有说服力
2. **FTR 机制简单而有效**：利用时序冗余进行 token 复用，对部署大型视觉模型到实时机器人系统有参考价值
3. **本体感知引导的视觉学习**：用机器人内部状态作为辅助监督信号来引导视觉特征学习，在 embodied AI 中有前景

## 局限与展望

1. **视角鲁棒性严重不足**：最大的短板。5° 扰动即失败，需要引入等变编码器或视角域随机化
2. **仅在仿真环境（MetaWorld）中验证**：缺少真实世界机器人实验
3. **VGGT 计算开销大**：大参数量限制实时控制的部署
4. **放置任务失败**：对小物体和精细操作支持不足

## 相关工作与启发

- **Diffusion Policy (DP)**：本文的策略基础，VGGT-DP 主要改进视觉编码器
- **DP3**：利用点云的 3D 信息，本文用 VGGT 替代点云
- **VLA 模型**：语言驱动的控制范式，本文提供了无语言替代方案的论证

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将 3D 重建预训练模型引入机器人控制有启发性
- **实验充分度**: ⭐⭐⭐ — 仅 MetaWorld 仿真环境，缺少真实世界实验
- **写作质量**: ⭐⭐⭐⭐ — 生物学动机论述有说服力
- **实用价值**: ⭐⭐⭐ — 视角鲁棒性差和高计算开销限制实际应用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Parameter-Free Fine-tuning via Redundancy Elimination for Vision Foundation Models](parameter-free_fine-tuning_via_redundancy_elimination_for_vision_foundation_mode.md)
- [\[ECCV 2024\] Sapiens: Foundation for Human Vision Models](../../ECCV2024/3d_vision/sapiens_foundation_for_human_vision_models.md)
- [\[AAAI 2026\] Adapt-As-You-Walk Through the Clouds: Training-Free Online Test-Time Adaptation of 3D Vision-Language Foundation Models](adapt-as-you-walk_through_the_clouds_training-free_online_te.md)
- [\[CVPR 2026\] AVA-Bench: Atomic Visual Ability Benchmark for Vision Foundation Models](../../CVPR2026/3d_vision/ava-bench_atomic_visual_ability_benchmark_for_vision_foundation_models.md)
- [\[ICLR 2026\] GIQ: Benchmarking 3D Geometric Reasoning of Vision Foundation Models with Simulated and Real Polyhedra](../../ICLR2026/3d_vision/giq_benchmarking_3d_geometric_reasoning_of_vision_foundation_models_with_simulat.md)

</div>

<!-- RELATED:END -->
