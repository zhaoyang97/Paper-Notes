---
title: >-
  [论文解读] TokenUnify: Scaling Up Autoregressive Pretraining for Neuron Segmentation
description: >-
  [ICCV 2025][3D视觉][自回归预训练] 提出 TokenUnify，通过统一随机 token 预测、下一 token 预测和下一全部 token 预测三种互补学习目标，在大规模电子显微镜数据上实现层次化预测编码，将自回归误差累积从 O(K) 降至 O(√K)，下游神经元分割提升 44%。
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "自回归预训练"
  - "神经元分割"
  - "电子显微镜"
  - "Mamba架构"
  - "层次预测编码"
---

# TokenUnify: Scaling Up Autoregressive Pretraining for Neuron Segmentation

**会议**: ICCV 2025  
**arXiv**: [2405.16847](https://arxiv.org/abs/2405.16847)  
**代码**: [https://github.com/ydchen0806/TokenUnify](https://github.com/ydchen0806/TokenUnify)  
**领域**: 3D Vision / Neuron Segmentation  
**关键词**: 自回归预训练, 神经元分割, 电子显微镜, Mamba架构, 层次预测编码

## 一句话总结

提出 TokenUnify，通过统一随机 token 预测、下一 token 预测和下一全部 token 预测三种互补学习目标，在大规模电子显微镜数据上实现层次化预测编码，将自回归误差累积从 O(K) 降至 O(√K)，下游神经元分割提升 44%。

## 研究背景与动机

**领域现状**：从电子显微镜（EM）体积图像中进行神经元分割是理解脑回路的关键步骤。EM 数据具有三大独特特性：(1) 高噪声（电子束交互），(2) 各向异性体素（z 轴分辨率粗），(3) 超长空间依赖（跨越数千 patch）。

**现有痛点**：
- 对比学习（DINO v2）和掩码重建（MAE）虽然表征能力强，但缺乏有利的 scaling law。MAE 的估计误差为 $O(\sqrt{s \log p / n})$，随模型容量增加收益递减。
- 自回归方法（AIM, LVM）试图弥合差距，但标准自回归的误差**线性累积** $O(K)$，对长序列（EM 数据 K 可达数千）非常不利。
- 传统视觉模型无法有效处理 EM 数据中的长程空间连续性。

**核心矛盾**：视觉数据结构比文本更复杂——单一预训练目标无法同时捕捉局部空间模式、序列依赖和全局结构。自回归在文本上成功的 scaling law 在视觉领域未能复现。

**本文切入角度**：从**信息论**出发，证明三种预测任务捕捉了视觉数据结构的互补方面。利用 Mamba 的线性复杂度序列建模能力处理长序列 EM 数据，并构建了 12 亿标注体素的大规模 EM 数据集。

## 方法详解

### 整体框架

两阶段流程：
1. **预训练阶段**：在 1TB+ 无标注 EM 数据上，使用三种互补预测任务训练通用视觉表征 $f_{\theta_1}(\cdot)$
2. **微调阶段**：在标注数据上微调分割模型 $g_{\theta_2}(\cdot)$，初始化自预训练权重

输入 3D EM 体积被分割为 $D' \times H' \times W'$ 的小 patch，tokenize 为长度 K 的序列，由 Mamba 高效处理。

### 关键设计

1. **随机 Token 预测（微观层面）**：类似 MAE，随机掩码比例 ρ 的 token，从未掩码上下文预测被掩码 token：
    $\mathcal{L}_{random} = -\mathbb{E}_{\mathcal{M} \sim \mathcal{D}_\rho} \left[\sum_{i \in \mathcal{M}} \log p_\theta(x_i | x_{\mathcal{M}^c})\right]$
   
   作用：学习位置不变的局部特征检测器，对噪声鲁棒，捕捉细胞膜和细胞器的重复模式。

2. **下一 Token 预测（中观层面）**：沿预定路径 π 进行自回归建模：
    $\mathcal{L}_{next} = -\mathbb{E}_\pi \left[\sum_{i=1}^K \log p_\theta(x_{\pi(i)} | x_{\pi(<i)})\right]$
   
   作用：捕捉神经元形态中的过渡模式——膜连续性、树突/轴突方向一致性等中尺度结构。

3. **下一全部 Token 预测（宏观层面）**：预测给定前文的**所有**后续 token：
    $\mathcal{L}_{next\text{-}all} = -\mathbb{E}_\pi \left[\sum_{i=1}^K \sum_{j=i}^K \log p_\theta(x_{\pi(j)} | x_{\pi(<i)})\right]$
   
   作用：捕捉分支模式、细胞类型特异形态和区域组织等长程关联。**关键理论贡献**——预测误差在多个位置分散而非累积，类似中心极限定理，将误差从 O(K) 降至 O(√K)。使用 Perceiver Resampler 通过交叉注意力聚合全序列信息，保持计算效率。

4. **多分辨率优化协议**：课程学习式权重调度——先易后难：

    - t < T₁ (30%)：随机预测主导 (权重 0.73)
    - T₁ ≤ t < T₂ (70%)：下一 token 预测主导
    - t ≥ T₂：下一全部预测主导
   
   通过 softmax 温度衰减平滑过渡，始终保持辅助任务贡献（~0.18 和 ~0.09），维持多任务协同。

5. **EMmamba 分割网络**：基于 SegMamba 改进的编-解码器，使用各向异性下采样层（z轴不下采样），适配 EM 数据的各向异性分辨率。

### 损失函数 / 训练策略

统一预训练目标：$\mathcal{L}_{TokenUnify} = \alpha(t) \cdot \mathcal{L}_{random} + \beta(t) \cdot \mathcal{L}_{next} + \gamma(t) \cdot \mathcal{L}_{next\text{-}all}$

分割微调使用仿射图预测 + MSE 损失，后处理采用 seeded watershed + 区域合并算法。

## 实验关键数据

### 主实验：MEC 数据集（Waterz 后处理）

| 预训练方法 | VOI_M↓ | VOI_S↓ | VOI↓ | ARAND↓ |
|-----------|--------|--------|------|--------|
| Random (无预训练) | 0.4915 | 1.2924 | 1.7839 | 0.2052 |
| MAE | 0.2325 | 1.0923 | 1.3248 | 0.0978 |
| BYOL | 0.2584 | 0.9453 | 1.2037 | 0.0891 |
| dbMIM | 0.2342 | 0.8796 | 1.1138 | 0.0742 |
| **TokenUnify** | **0.1953** | **0.7998** | **0.9951** | **0.0509** |

TokenUnify 相比随机初始化提升 **44%**（VOI: 1.78→1.00），比 MAE 提升 **25%**（1.32→1.00）。

### 消融实验

| 预训练策略 | VOI↓ | ARAND↓ |
|-----------|------|--------|
| Random (仅掩码预测) | 1.2680 | 0.0862 |
| Next (仅自回归) | 4.0418 | 0.4416 |
| Random + Next | 1.1300 | 0.0692 |
| Random + Next-all | 1.1907 | 0.1203 |
| **Random + Next + Next-all** | **0.9951** | **0.0509** |

关键观察：
- 纯自回归（Next only）效果极差(VOI=4.04)——单纯自回归不适合视觉任务，需要全局空间理解
- 三策略完整组合最优，验证了互补性假设
- Random 提供空间一致性初始化（1.27），是最佳单一策略

| 微调模块 | VOI↓ | ARAND↓ |
|---------|------|--------|
| 仅 Mamba blocks | 1.1362 | 0.0782 |
| 仅 Encoder | 1.5556 | 0.1370 |
| 仅 Decoder | 1.5295 | 0.1212 |
| Mamba + Encoder | 1.1065 | 0.0629 |
| **全部微调** | **0.9951** | **0.0509** |

Mamba blocks 是最关键组件（序列建模能力的核心），资源受限时优先微调。

### 关键发现

- **Scaling Law**：从 100M 到 1B 参数，TokenUnify 持续优于其他方法，展现出语言模型般的 scaling 特性。Mamba 比 Transformer 以更少参数实现相当性能，验证了线性复杂度架构在长序列视觉数据上的效率优势。
- **AC3/4 小数据集**（仅 1/10 MEC 标注量）：TokenUnify + Mamba 性能接近有监督 SOTA 方法 PEA，比 MAE 提升 11%，证明在标注稀缺场景下的有效性。
- **跨域初步验证**：在 Kodak 自然图像上预训练，TokenUnify 重建质量 PSNR 比纯自回归高 2-4 dB，说明框架不局限于 EM 领域。

## 亮点与洞察

- **信息论视角的统一**：三个预测任务分别捕捉 $I(x_i; x_{\mathcal{M}^c})$、$I(x_i; x_{<i})$、$I(\{x_i,...,x_K\}; x_{<i})$，合在一起最大化总信息提取。
- **O(K) → O(√K) 的误差累积降低**：next-all 预测通过在多个位置分散误差实现，类似中心极限定理的√n scaling。这是对纯自回归方法的重要理论改进。
- **首个十亿参数级 Mamba 视觉网络**：证明 Mamba 在长序列视觉建模中的 scaling 可行性。
- **12 亿标注体素的 MEC 数据集**：6 个脑功能区域、两位专家 6 个月标注，是同类最大标注 EM 数据集之一。

## 局限与展望

- Next-all 预测使用 Perceiver Resampler 做近似，是否可设计更直接的全局预测机制值得探索。
- 课程学习的阶段划分（30%/70%）和权重比例靠经验设定，可考虑自适应调度。
- Mamba 在小模型（28M参数）上提升显著，但原始 EMmamba 不加预训练表现不如传统 CNN（如 Superhuman 1.5M 参数），说明 Mamba 架构本身对标注高效利用仍有差距。

## 相关工作与启发

- AIM [El-Nouby 2024] 和 LVM [Bai 2023] 在自然图像上探索自回归预训练，但缺乏 TokenUnify 的多任务互补设计和理论分析。
- MAGE [Li 2023] 结合掩码和生成目标但局限于 2D，TokenUnify 扩展到 3D 长序列。
- 对比 EM 专用方法 dbMIM 和 MS-Con-EM，TokenUnify 在同一 Mamba backbone 上有显著优势，证明预训练目标设计的重要性。

## 评分

- 新颖性：⭐⭐⭐⭐⭐ — 三种预测任务的信息论统一 + 误差累积分析
- 技术深度：⭐⭐⭐⭐ — 理论分析（虽部分在附录）+ 多分辨率优化协议
- 实验充分度：⭐⭐⭐⭐⭐ — 大规模数据集构建、多方法对比、scaling分析、充分消融
- 实用性：⭐⭐⭐⭐ — 对连接组学和生物图像分析有直接应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] MegaSynth: Scaling Up 3D Scene Reconstruction with Synthesized Data](../../CVPR2025/3d_vision/megasynth_scaling_up_3d_scene_reconstruction_with_synthesized_data.md)
- [\[ICCV 2025\] EgoM2P: Egocentric Multimodal Multitask Pretraining](egom2p_egocentric_multimodal_multitask_pretraining.md)
- [\[ICCV 2025\] PanSt3R: Multi-view Consistent Panoptic Segmentation](panst3r_multi-view_consistent_panoptic_segmentation.md)
- [\[ICCV 2025\] Trace3D: Consistent Segmentation Lifting via Gaussian Instance Tracing](trace3d_consistent_segmentation_lifting_via_gaussian_instance_tracing.md)
- [\[NeurIPS 2025\] Enhancing Multilingual LLM Pretraining with Model-Based Data Selection](../../NeurIPS2025/3d_vision/enhancing_multilingual_llm_pretraining_with_model-based_data_selection.md)

</div>

<!-- RELATED:END -->
