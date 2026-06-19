---
title: >-
  [论文解读] Temporal Action Detection Model Compression by Progressive Block Drop
description: >-
  [CVPR 2025][自动驾驶][时序动作检测] 提出渐进式块丢弃(Progressive Block Drop)方法从深度维度压缩时序动作检测(TAD)模型，通过逐步移除冗余块并使用参数高效的跨深度对齐策略恢复性能，实现 25% 计算量减少的同时性能不降反升。 时序动作检测(TAD)旨在未剪辑视频中识别和定位动作实例的起…
tags:
  - "CVPR 2025"
  - "自动驾驶"
  - "时序动作检测"
  - "模型压缩"
  - "渐进式块丢弃"
  - "深度剪枝"
  - "LoRA"
---

# Temporal Action Detection Model Compression by Progressive Block Drop

**会议**: CVPR 2025  
**arXiv**: [2503.16916](https://arxiv.org/abs/2503.16916)  
**代码**: 无  
**领域**: 自动驾驶/视频理解  
**关键词**: 时序动作检测, 模型压缩, 渐进式块丢弃, 深度剪枝, LoRA

## 一句话总结

提出渐进式块丢弃(Progressive Block Drop)方法从深度维度压缩时序动作检测(TAD)模型，通过逐步移除冗余块并使用参数高效的跨深度对齐策略恢复性能，实现 25% 计算量减少的同时性能不降反升。

## 研究背景与动机

时序动作检测(TAD)旨在未剪辑视频中识别和定位动作实例的起止时间，是视频问答和视频描述等应用的基础。随着更大的特征提取器和数据集的使用，TAD 模型的计算需求显著增长，限制了在自动驾驶和机器人等资源受限场景中的部署。

- **计算瓶颈在特征提取器**：特征提取阶段占总计算量的 **95%**，因为需要滑动窗口逐段处理整个视频
- **通道剪枝不利于 GPU 并行**：传统的通道剪枝(width reduction)使权重矩阵变小，但小矩阵乘法在 GPU 上并行效率低
- **深而窄 vs 浅而宽**：研究表明在相同参数量下，浅而宽的网络比深而窄的网络推理更快
- **块级冗余存在**：实验发现某些块的输入输出特征差异很小(MSE 接近 0)，且移除单个块对性能影响很小
- **直接丢弃多块性能损失大**：同时移除多个块导致显著的性能下降，需要渐进式策略

## 方法详解

### 整体框架

Progressive Block Drop 采用多步迭代压缩策略。每次迭代包含两个阶段：(1) 块选择评估器自动选择影响最小的块并移除；(2) 参数高效的跨深度对齐策略通过 LoRA 微调恢复模型性能。迭代持续进行直到压缩模型性能不再可恢复到未压缩模型水平。

### 关键设计1: Block Selection Evaluator — 自动选择可丢弃的冗余块

**功能**: 评估每个块的重要性，自动选择对性能影响最小的块进行移除。

**核心思路**: 在第 $t$ 步迭代中，遍历当前模型 $M_{t-1}$ 的所有 $K$ 个块，分别构建丢弃每个块后的子网络 $\mathcal{S}_t = \{M_{t,k}^{sub} = M_{t-1} \setminus b_k\}$。使用评估函数 $f_E$ 在训练集上衡量每个子网络与未压缩模型 $M_0$ 的性能差异，选择性能差异最小的子网络（即丢弃对性能影响最小的块）。

**设计动机**: 基于 Curriculum Learning 的启发，逐步从简单到困难地移除冗余，而非一次性丢弃多块。实验验证基于 mAP 指标的选择效果最优。

### 关键设计2: 参数高效的跨深度对齐 — LoRA + 特征/预测双层蒸馏

**功能**: 在块被移除后，通过低成本训练恢复压缩模型的性能。

**核心思路**: 在每个剩余块的注意力层中插入 LoRA 参数 $\theta_{\text{LoRA}}$，仅训练这些新增参数。提出跨深度对齐损失，当第 $i$ 个块被丢弃时，对齐压缩模型和未压缩模型中剩余块的输出特征：

$$\mathcal{L}_f = \frac{1}{I-1} \sum_{m \neq i} (f_{b_m}^{M_0} - f_{b_m}^{\hat{M}_t})^2$$

同时对齐预测层：分类用 KL 散度 $\mathcal{L}_{pc}$，定位用 GIoU 损失 $\mathcal{L}_{pr}$。总损失为：

$$\mathcal{L}_{total} = \mathcal{L}_{pc} + \mathcal{L}_{pr} + \mathcal{L}_f + \mathcal{L}_{cls} + \mathcal{L}_{reg}$$

**设计动机**: (1) 仅训练 LoRA 参数大幅减少 GPU 内存消耗（如 VideoMAE-L 的全参数微调需 13.1GB/batch）；(2) 训练后 LoRA 可合并到原始参数中，不增加推理开销；(3) 实验证明纯 GT 监督无法恢复性能，特征级对齐是关键。

### 关键设计3: 深度减少的硬件友好性 — 浅而宽优于深而窄

**功能**: 与传统通道剪枝方法正交，可组合使用实现更大压缩率。

**核心思路**: 减少网络深度保持层宽度不变，得到的浅而宽架构在 GPU 上推理速度更快。相同 MACs 下，深度减少比宽度减少实现更高的实际加速比。

**设计动机**: GPU 推理中大矩阵运算的并行效率显著高于多次小矩阵运算。通道剪枝虽然减少了 FLOPs，但小矩阵降低了 GPU 利用率。

### 损失函数

$$\mathcal{L}_{total} = \mathcal{L}_{pc} + \mathcal{L}_{pr} + \mathcal{L}_f + \mathcal{L}_{cls} + \mathcal{L}_{reg}$$

包含分类对齐(KL散度)、定位对齐(GIoU)、跨深度特征对齐(MSE)以及标准分类和回归损失。

## 实验关键数据

### 主实验: VideoMAE-S (12 blocks) + ActionFormer

| 丢弃块数 | MACs (G) | 推理时间 | THUMOS14 mAP@0.5 | ActivityNet mAP |
|---------|----------|---------|-----------------|----------------|
| 0 (基线) | 286.3 | 104.9ms | 70.43 | 37.75 |
| 1 | 263.5 (92%) | 98.4ms | 71.06 (+0.63) | 37.94 (+0.19) |
| 2 | 240.8 (84%) | 89.8ms | 71.37 (+0.94) | 37.81 (+0.06) |
| 3 | 218.0 (76%) | 81.2ms | 70.47 (+0.04) | 37.77 (+0.02) |
| 4 | 195.2 (68%) | 73.6ms | 69.65 (-0.78) | 37.72 (-0.03) |

### 更深网络: VideoMAE-L (24 blocks)

| 丢弃块数 | MACs | THUMOS14 mAP@0.5 |
|---------|------|-----------------|
| 0 (基线) | 3886.9G | 76.01 |
| 3 | 3402.7G (87.5%) | **78.29 (+2.28)** |
| 6 | 2918.6G (75.1%) | 77.25 (+1.24) |

### 推理速度对比 (相同 MACs 下)

| 方法 | MACs (G) | 推理时间 | 加速比 |
|------|----------|---------|-------|
| 未压缩 | 286.3 | 104.9ms | 1.00× |
| 通道剪枝 | 218.2 | 96.6ms | 1.09× |
| **块丢弃 (Ours)** | 218.0 | 81.2ms | **1.29×** |

### 与通道剪枝组合

在 70.2% mAP 精度下，块丢弃+通道剪枝减少 **30%** MACs，通道剪枝单独仅减少 **10%**。

### 关键发现

- 丢弃 3 块（25% MACs 减少）在 THUMOS14 上性能 **不降反升**
- 更深的模型（24层）冗余更多，25% 压缩下性能提升 **+1.24%**（vs 12层的 +0.04%）
- 渐进式丢弃比一次性丢弃效果好（70.47% vs 68.91%），验证了渐进策略的必要性
- 方法可跨架构（AdaTAD、ActionFormer）、跨数据集（FineAction）和跨任务（自然语言定位）泛化

## 亮点与洞察

1. **从深度维度压缩 TAD 模型是新视角**：之前的工作主要关注通道剪枝，本文首次系统探索了块级深度压缩
2. **压缩后性能反而提升**：这一反直觉的结果表明 TAD 模型确实存在显著的深度冗余
3. **与通道剪枝正交**：两种方法可以组合使用获得更大的压缩率，具有很好的实用性
4. **硬件友好性分析有说服力**：同等 MACs 下 1.29× vs 1.09× 的加速比差异直观地展示了浅宽架构的优势

## 局限与展望

- 当前仅在 VideoMAE Transformer 架构上验证，CNN 主干的适用性未探索
- 块选择需要在训练集上评估所有候选子网络，选择开销随块数线性增长
- LoRA 的秩选择对性能恢复的影响未深入讨论
- 可探索自动确定最优压缩率的策略，而非依赖人工设定终止条件
- 结合知识蒸馏和量化等其他压缩技术可能进一步提升压缩效果

## 相关工作与启发

- **ActionFormer / AdaTAD**: 主流的 TAD 检测头架构
- **VideoMAE**: 基于掩码自编码的视频特征提取器，本文的压缩目标
- **LoRA**: 参数高效微调方法，这里巧妙用于压缩后的性能恢复
- **Curriculum Learning**: 渐进式学习的方法论启发了逐步丢弃策略

## 评分

⭐⭐⭐⭐ — 问题定义清晰（95% 计算在特征提取器），方法简洁有效（渐进丢弃 + LoRA 恢复），实验非常全面（多架构、多数据集、多任务、与剪枝组合）。压缩后性能提升的结果很有说服力，硬件友好性分析是加分项。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] EV-3DOD: Pushing the Temporal Boundaries of 3D Object Detection with Event Cameras](ev-3dod_pushing_the_temporal_boundaries_of_3d_object_detection_with_event_camera.md)
- [\[CVPR 2025\] RENO: Real-Time Neural Compression for 3D LiDAR Point Clouds](reno_real-time_neural_compression_for_3d_lidar_point_clouds.md)
- [\[CVPR 2026\] Counterfactual VLA: Self-Reflective Vision-Language-Action Model with Adaptive Reasoning](../../CVPR2026/autonomous_driving/counterfactual_vla_self-reflective_vision-language-action_model_with_adaptive_re.md)
- [\[ICCV 2025\] SparseLaneSTP: Leveraging Spatio-Temporal Priors with Sparse Transformers for 3D Lane Detection](../../ICCV2025/autonomous_driving/sparselanestp_leveraging_spatio-temporal_priors_with_sparse_transformers_for_3d_.md)
- [\[CVPR 2026\] NoRD: A Data-Efficient Vision-Language-Action Model that Drives without Reasoning](../../CVPR2026/autonomous_driving/nord_a_data-efficient_vision-language-action_model_that_drives_without_reasoning.md)

</div>

<!-- RELATED:END -->
