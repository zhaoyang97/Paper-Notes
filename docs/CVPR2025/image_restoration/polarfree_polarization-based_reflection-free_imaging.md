---
title: >-
  [论文解读] PolarFree: Polarization-based Reflection-Free Imaging
description: >-
  [CVPR 2025][图像恢复][反射去除] 构建 6500 对的大规模 RGB-偏振图像数据集 PolaRGB，并提出 PolarFree 两阶段网络——先用条件扩散模型生成无反射先验，再用去反射骨干网络分离透射层，在偏振引导的反射去除任务上超越先前方法约 2dB PSNR。 1. 领域现状：反射去除是计算机视觉中的经典…
tags:
  - "CVPR 2025"
  - "图像恢复"
  - "反射去除"
  - "偏振成像"
  - "扩散模型"
  - "大规模数据集"
  - "频域损失"
---

# PolarFree: Polarization-based Reflection-Free Imaging

**会议**: CVPR 2025  
**arXiv**: [2503.18055](https://arxiv.org/abs/2503.18055)  
**代码**: [https://github.com/mdyao/PolarFree](https://github.com/mdyao/PolarFree)  
**领域**: 图像复原  
**关键词**: 反射去除, 偏振成像, 扩散模型, 大规模数据集, 频域损失

## 一句话总结
构建 6500 对的大规模 RGB-偏振图像数据集 PolaRGB，并提出 PolarFree 两阶段网络——先用条件扩散模型生成无反射先验，再用去反射骨干网络分离透射层，在偏振引导的反射去除任务上超越先前方法约 2dB PSNR。

## 研究背景与动机
1. **领域现状**：反射去除是计算机视觉中的经典难题，已有方法主要依赖 RGB 图像的亮度/梯度等强度线索，但反射去除是高度病态的逆问题（从一个观测恢复两个未知层）。偏振成像提供了物理层面区分反射光和透射光的天然线索。
2. **现有痛点**：(a) 现有偏振反射去除数据集规模小（<1000 对）、缺乏 RGB 信息、或为合成数据；(b) 从偏振数据中提取无反射信息具有挑战性，因拍摄角度、场景和光照变化多端。
3. **核心矛盾**：偏振提供了强大的物理线索但缺乏高质量大规模数据集来训练模型；同时偏振数据中的无反射信息提取需要强大的生成模型来处理复杂场景变化。
4. **本文目标**：(a) 构建首个大规模真实 RGB+偏振反射去除数据集；(b) 设计利用扩散模型充分挖掘偏振线索的反射去除网络。
5. **切入角度**：偏振光的物理特性——反射光和透射光具有不同的偏振度（DoLP）和偏振角（AoLP），在布儒斯特角反射光完全偏振。
6. **核心 idea**：用扩散模型生成"无反射先验"作为中间引导信号，再用去反射网络精确分离。

## 方法详解

### 整体框架
PolarFree 包含两步推理流程：(1) **先验生成步**：条件扩散模型 $\mathcal{F}_{diff}$ 接收偏振图像+RGB 图像作为条件 $M_{cond} = \{M_{polar}, M_{aolp}, M_{dolp}, M_{rgb}\}$，从随机噪声逐步去噪生成无反射先验 $\hat{z}_0$；(2) **反射去除步**：去反射骨干网络 $\mathcal{F}_{remove}$ 综合先验 $\hat{z}_0$ 和输入条件，输出干净的透射层 $\hat{T}_{rgb}$。

### 关键设计

1. **PolaRGB 大规模数据集**

    - 功能：提供首个大规模真实 RGB+偏振反射去除训练数据
    - 核心思路：使用分光偏振相机（division-of-focal-plane）通过视频采集工作流高效获取数据——先拍无反射的透射图 $T_{raw}$，再放置半反射玻璃板连续旋转采集混合图 $M_{raw}$。在 RAW 域做仿射变换实现像素级配准，再进行偏振分离得到 4 个角度（0°/45°/90°/135°）的偏振图像和非偏振 RGB。数据集含 6500 对，是前作 Lei et al. 的 8 倍。
    - 设计动机：真实偏振数据远比合成数据更能反映实际场景复杂性，大规模数据是数据驱动方法成功的基础

2. **条件扩散模型生成无反射先验**

    - 功能：从偏振和 RGB 输入中提取"无反射"的中间表示作为去反射的引导信号
    - 核心思路：分两阶段训练：第一阶段训练编码器 $\mathcal{E}$ 从干净透射图的偏振数据中提取先验 $z_0 = \mathcal{E}(M_{cond}, T_{rgb})$，同时训练去反射骨干网络 $\mathcal{F}_{remove}$；第二阶段用 $z_0$ 作为扩散模型的目标监督，训练条件扩散模型 $\mathcal{F}_{diff}$ 从混合图像（含反射）的偏振数据生成先验 $\hat{z}_0$。推理时只需扩散模型和去反射网络。
    - 设计动机：扩散模型缺乏直接的"无反射先验"训练目标，两阶段训练巧妙地用编码器桥接了这个gap

3. **频域相位损失 (Phase Loss)**

    - 功能：缓解半反射表面引起的色差问题
    - 核心思路：利用 FFT 的相位信息主要编码形状和纹理、对颜色不敏感的特性，定义 $\mathcal{L}_{phase} = \|\angle(FFT(\hat{T})) - \angle(FFT(T_{rgb}))\|_1$。相比空间域的 L1/VGG loss，相位损失不受采集过程中半反射表面引入的色偏干扰。
    - 设计动机：训练数据中透射层和混合图像可能存在色差（物理采集引入），空间域损失会误导模型学习颜色调整而非反射去除

### 损失函数 / 训练策略
- 基础损失：$\mathcal{L}_1$（像素级）+ $\mathcal{L}_{VGG}$（感知损失）+ $\mathcal{L}_{TV}$（总变差）
- 相位损失：$\mathcal{L}_{phase} = \|\angle(FFT(\hat{T})) - \angle(FFT(T_{rgb}))\|_1$
- 扩散损失：标准 DDPM 噪声预测损失 $\mathcal{L}_{diff}$
- 两阶段训练：先训练先验编码器+去反射网络，再训练扩散模型+微调去反射网络

## 实验关键数据

### 主实验

| 方法 | PSNR↑ | SSIM↑ | 说明 |
|------|-------|-------|------|
| RRW (CVPR'24) | ~27 | ~0.85 | 无偏振、纯 RGB |
| IBCLN | ~28 | ~0.87 | 无偏振 |
| Lei et al. | ~29 | ~0.88 | 小数据集偏振 |
| **PolarFree** | **~31** | **~0.91** | 超越前方法 ~2dB |

### 消融实验

| 配置 | PSNR↑ | 说明 |
|------|-------|------|
| Full PolarFree | ~31 | 完整模型 |
| w/o 扩散先验 | ~29 | 直接从偏振做去反射，损失 ~2dB |
| w/o Phase loss | ~30.2 | 相位损失贡献约 0.8dB |
| w/o 偏振输入 | ~28 | 仅用 RGB，退化为传统方法 |

### 关键发现
- 偏振信息相比纯 RGB 方法带来约 3dB 的 PSNR 提升，验证了偏振线索的关键价值
- 扩散模型生成的无反射先验比直接回归更有效，因为扩散模型能恢复被反射遮挡的细节
- Phase loss 在色差严重的场景（如高反射环境）中贡献尤为显著
- 在博物馆/画廊等实际高反射场景的定性测试中，PolarFree 保留细节能力远超对比方法

## 亮点与洞察
- **数据集贡献**可能比方法本身更具长期价值——PolaRGB 是首个大规模真实 RGB+偏振反射去除数据集，将推动整个偏振反射去除领域
- **相位损失**是一个可迁移的技巧——任何存在色差干扰的恢复任务都可以考虑在频域相位空间添加约束
- **两阶段训练策略**巧妙解决了扩散模型缺乏目标的问题，这个"先训编码器提供目标、再训生成器"的范式可推广到其他中间表示学习任务

## 局限与展望
- 偏振相机硬件仍不够普及，限制了方法的实际部署
- 扩散模型推理速度较慢，不适合实时应用
- 数据集主要覆盖玻璃反射场景，水面反射等其他类型覆盖不足
- 未来可探索用更快的生成模型（如 consistency model）替代 DDPM

## 相关工作与启发
- **vs RRW**: 纯 RGB 反射去除，缺乏物理线索；本文利用偏振物理大幅提升
- **vs Lei et al.**: 同为偏振反射去除但数据集仅 807 张且无 RGB，本文 6500 张+RGB 显著扩大规模和适用性
- **vs ReflectNet**: 使用合成偏振数据训练，泛化性不如真实数据

## 评分
- 新颖性: ⭐⭐⭐⭐ 扩散模型+偏振的结合和两阶段训练策略新颖，但各组件有前人基础
- 实验充分度: ⭐⭐⭐⭐ 数据集构建详实，消融充分，有真实场景测试
- 写作质量: ⭐⭐⭐⭐ 偏振物理背景介绍清晰，整体结构完整
- 价值: ⭐⭐⭐⭐⭐ 数据集+方法双重贡献，对偏振反射去除领域有里程碑意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Polarization State Tracing for Reflection Removal and Color-Consistent Reconstruction](../../CVPR2026/image_restoration/polarization_state_tracing_for_reflection_removal_and_color-consistent_reconstru.md)
- [\[CVPR 2025\] Reversible Decoupling Network for Single Image Reflection Removal](reversible_decoupling_network_for_single_image_reflection_removal.md)
- [\[CVPR 2025\] A Physics-Informed Blur Learning Framework for Imaging Systems](a_physics-informed_blur_learning_framework_for_imaging_systems.md)
- [\[NeurIPS 2025\] RGB-to-Polarization Estimation: A New Task and Benchmark Study](../../NeurIPS2025/image_restoration/rgb-to-polarization_estimation_a_new_task_and_benchmark_study.md)
- [\[CVPR 2025\] Proximal Algorithm Unrolling: Flexible and Efficient Reconstruction Networks for Single-Pixel Imaging](proximal_algorithm_unrolling_flexible_and_efficient_reconstruction_networks_for_.md)

</div>

<!-- RELATED:END -->
