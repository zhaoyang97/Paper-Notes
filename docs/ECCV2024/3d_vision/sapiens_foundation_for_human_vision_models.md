---
title: >-
  [论文解读] Sapiens: Foundation for Human Vision Models
description: >-
  [ECCV 2024][3D视觉][人体视觉基础模型] Sapiens 提出了一个以人为中心的视觉基础模型家族（0.3B-2B参数），通过在3亿张人体图像上进行 MAE 自监督预训练，原生支持1K高分辨率推理，在2D姿态估计、身体部位分割、深度估计和表面法线预测四个人体视觉任务上全面超越现有SOTA。
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "人体视觉基础模型"
  - "自监督预训练"
  - "Transformer"
  - "高分辨率推理"
  - "人体感知"
---

# Sapiens: Foundation for Human Vision Models

**会议**: ECCV 2024  
**arXiv**: [2408.12569](https://arxiv.org/abs/2408.12569)  
**代码**: [https://github.com/facebookresearch/sapiens](https://github.com/facebookresearch/sapiens)  
**领域**: 3D视觉  
**关键词**: 人体视觉基础模型, 自监督预训练, Vision Transformer, 高分辨率推理, 人体感知

## 一句话总结
Sapiens 提出了一个以人为中心的视觉基础模型家族（0.3B-2B参数），通过在3亿张人体图像上进行 MAE 自监督预训练，原生支持1K高分辨率推理，在2D姿态估计、身体部位分割、深度估计和表面法线预测四个人体视觉任务上全面超越现有SOTA。

## 研究背景与动机
- **领域现状**: 生成逼真人体（2D/3D）的方法近年取得巨大进展，但这些方法依赖于鲁棒的人体感知资产（2D关键点、身体分割、深度、法线），而这些资产的精确估计仍是活跃研究领域。
- **现有痛点**: (1) 各任务的方法高度特化、系统复杂，不利于推广；(2) 野外（in-the-wild）场景的标注数据难以大规模获取；(3) 现有视觉基础模型主要针对通用图像，未针对人体领域优化。
- **核心矛盾**: 通用视觉基础模型在人体任务中未必最优，而人体专用模型缺乏规模化预训练。
- **本文切入角度**: 领域特化的大规模预训练——收集3亿人体图像进行 MAE 预训练，再用高质量（甚至合成）标注微调，实现泛化性、广泛适用性和高保真度三位一体。
- **核心idea**: 相同计算预算下，在人体领域数据集上的自监督预训练显著优于通用数据集预训练。

## 方法详解

### 整体框架
Sapiens 采用 pretrain-then-finetune 范式：
1. **预训练阶段**: 在 Humans-300M 数据集上用 MAE（Masked Autoencoder）方法对 ViT 进行自监督预训练，原生输入分辨率为 1024×1024，patch size 为16
2. **微调阶段**: 对四个下游任务分别微调，采用统一的 encoder-decoder 架构——encoder 用预训练权重初始化，decoder 随机初始化，端到端微调

### 关键设计
1. **Humans-300M 数据集**: 从约10亿张野外图像中筛选得到3亿张人体图像。筛选标准包括：去除水印/文字/艺术画、使用人体检测器过滤（检测分数>0.9，bbox>300像素）。超2.48亿张包含多人场景。这是区别于通用预训练的核心——领域特化数据策略。

2. **高分辨率 MAE 预训练**: 不同于现有 ViT 在 224×224 上预训练，Sapiens 在 1024×1024 上预训练，FLOPs 约为最大现有 ViT 的4倍。每个 patch token 仅占图像面积的 0.02%（vs 标准 ViT 的 0.4%），提供更细粒度的 token 间推理能力。即使 masking ratio 高达 95%，模型仍能合理重建人体解剖结构。

3. **模型缩放策略**: 提供4个尺度（0.3B/0.6B/1B/2B），优先按宽度而非深度缩放。最大模型 2B 参数，FLOPs 约 8.7T。所有模型在 1.2 万亿 token 上预训练。

4. **高质量标注**: 

    - 姿态估计：引入 308 个全身关键点（含 243 个面部关键点），在室内多视角捕捉系统中标注 100 万 4K 图像
    - 身体分割：28 类词表（含上下唇、牙齿、舌头等细粒度类别），标注 10 万张 4K 图像
    - 深度/法线：使用 600 个高分辨率真人扫描体的合成数据

5. **差异化学习率**: 底层 encoder 使用更低学习率（层级学习率衰减 0.85），保持预训练泛化能力。

### 损失函数 / 训练策略
- **姿态估计**: MSE loss（热力图回归）
- **身体分割**: Weighted Cross-Entropy loss
- **深度估计**: Scale-invariant log depth loss（公式见论文 Eq.1-3）
- **法线估计**: L1 loss + cosine similarity loss（1 - n·n̂）
- 训练使用 AdamW 优化器，线性 warmup + cosine annealing/linear decay
- 2B 模型在 1024 张 A100 GPU 上预训练 18 天

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文(2B) | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| Humans-5K (Pose) | Whole-body AP | 61.1 | DWPose-l: 53.1 | **+7.6** |
| Humans-2K (Seg) | mIoU | 81.2 | DeepLabV3+: 64.1 | **+17.1** |
| Hi4D (Depth) | RMSE | 0.114 | DepthAnything-L: 0.147 | **-22.4%** |
| THuman2.0 (Normal) | Mean Angular Error | 大幅降低 | PIFuHD: 30.51° | **-53.5%** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Sapiens-0.3B vs ViTPose+-L | +5.6 AP (Pose) | 相同参数量，领域预训练大幅优胜 |
| Sapiens-0.6B vs ViTPose+-H | +7.9 AP (Pose) | 持续扩大优势 |
| 0.3B → 0.6B → 1B → 2B | AP 单调递增 | 模型规模持续带来收益 |
| Sapiens-0.3B vs Mask2Former (Seg) | +12.6 mIoU | 高分辨率+人体预训练 vs 通用分割 |
| 只用合成数据微调 (Depth) | 0.008 RMSE (Face) | 仅凭合成数据即超越真实数据训练的 DepthAnything |

### 关键发现
- 领域特化预训练在相同计算预算下对人体任务的提升远大于通用预训练
- 高质量or合成标注 + 领域预训练 = 出色的野外泛化能力
- 模型性能随参数量增长呈正相关，未出现饱和
- 仅用室内多视角标注微调，也能泛化到各种野外场景

## 亮点与洞察
- **数据策略的力量**: 相比 DINOv2（通用 142M 图像）和 AIM（通用 2B 图像），Sapiens 用「更少但更聚焦」的人体数据取得更好的人体任务表现，验证了"领域数据>数据量"的假说
- **简洁统一的架构**: 四个任务共享相同的 encoder-decoder 框架，仅更换 decoder 输出头，说明强大的预训练表征足以支撑多任务
- **高分辨率原生支持**: 1K 分辨率预训练是直觉但少有人做的选择（FLOPs代价巨大），但对人体细节感知至关重要
- **合成数据的惊喜**: 深度/法线任务仅用合成数据微调，就实现了顶级in-the-wild性能

## 局限与展望
- 预训练数据集 Humans-300M 为私有数据，不公开，可复现性受限
- 2B 模型的推理 FLOPs 高达 8.7T，实际部署受限
- 只做了4个人体任务，未探索体型估计 (body shape)、手势识别等其他人体任务
- 未与 DINOv2 等在人体任务上做直接的预训练数据消融对比实验
- Top-down 范式依赖人体检测器，multi-person 场景可能受检测器质量影响

## 相关工作与启发
- 与 DINOv2/AIM/MAWS 等通用预训练对比，验证了领域特化预训练的价值
- 与 ViTPose+/DWPose 等专用方法对比，证明了「简单架构 + 强预训练」可以超越复杂专用设计
- 启发：其他领域（如自动驾驶、医学影像）是否也能通过领域特化大规模预训练获得类似收益？

## 评分
- 新颖性: ⭐⭐⭐⭐ 核心idea（领域特化预训练）并不新颖，但在人体视觉的系统性验证很充分
- 实验充分度: ⭐⭐⭐⭐⭐ 四个任务、多个数据集、完整的缩放实验，极为扎实
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰、表述精简，表格图示信息密度高
- 价值: ⭐⭐⭐⭐⭐ 提供了实用的人体视觉基础模型家族，对社区有重要推动价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] VGGT-DP: Generalizable Robot Control via Vision Foundation Models](../../AAAI2026/3d_vision/vggt-dp_generalizable_robot_control_via_vision_foundation_models.md)
- [\[CVPR 2026\] AVA-Bench: Atomic Visual Ability Benchmark for Vision Foundation Models](../../CVPR2026/3d_vision/ava-bench_atomic_visual_ability_benchmark_for_vision_foundation_models.md)
- [\[AAAI 2026\] Parameter-Free Fine-tuning via Redundancy Elimination for Vision Foundation Models](../../AAAI2026/3d_vision/parameter-free_fine-tuning_via_redundancy_elimination_for_vision_foundation_mode.md)
- [\[ECCV 2024\] When Do We Not Need Larger Vision Models?](when_do_we_not_need_larger_vision_models.md)
- [\[ECCV 2024\] Transferable 3D Adversarial Shape Completion using Diffusion Models](transferable_3d_adversarial_shape_completion_using_diffusion_models.md)

</div>

<!-- RELATED:END -->
