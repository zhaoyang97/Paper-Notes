---
title: >-
  [论文解读] EgoAdapt: Adaptive Multisensory Distillation and Policy Learning for Efficient Egocentric Perception
description: >-
  [ICCV 2025][视频理解][自我中心感知] 提出 EgoAdapt 框架，将跨模态蒸馏与策略学习联合训练，自适应选择最优模态组合，在自我中心感知任务中实现最高 89% GMACs 缩减的同时保持与 SOTA 持平甚至更优的性能。 现代 AR/VR 系统依赖多传感器数据流（RGB 视频、多通道音频、行为数据）进行自我中…
tags:
  - "ICCV 2025"
  - "视频理解"
  - "自我中心感知"
  - "多模态蒸馏"
  - "策略学习"
  - "高效推理"
  - "多感官融合"
---

# EgoAdapt: Adaptive Multisensory Distillation and Policy Learning for Efficient Egocentric Perception

**会议**: ICCV 2025  
**arXiv**: [2506.21080](https://arxiv.org/abs/2506.21080)  
**代码**: 无  
**领域**: 视频理解  
**关键词**: 自我中心感知, 多模态蒸馏, 策略学习, 高效推理, 多感官融合

## 一句话总结

提出 EgoAdapt 框架，将跨模态蒸馏与策略学习联合训练，自适应选择最优模态组合，在自我中心感知任务中实现最高 89% GMACs 缩减的同时保持与 SOTA 持平甚至更优的性能。

## 研究背景与动机

现代 AR/VR 系统依赖多传感器数据流（RGB 视频、多通道音频、行为数据）进行自我中心感知，但 SOTA 模型计算开销巨大，难以在资源受限设备上实时运行。关键观察是：**并非所有模态都需要同时处理**。例如在活跃说话人定位任务中，当多人可见时视觉线索足够，当说话人在视野外时则多通道音频更有效。

现有方法要么只做模型蒸馏（静态，无法适应变化的任务需求），要么只做自适应模态选择（仍依赖昂贵模型），缺乏**统一框架**同时兼顾两者。EgoAdapt 的核心洞察是：联合训练蒸馏和策略优化可以产生**协同效应**——蒸馏模型的高效性被策略优化的自适应性互补，整体既轻量又灵活。

## 方法详解

### 整体框架

EgoAdapt 包含两个核心模块：**跨模态特征蒸馏（CFD）** 模块 Φ 和**任务感知多感官策略学习（TeMPLe）** 模块 Π，通过三阶段联合训练策略实现端到端优化。框架适用于三个自我中心任务：动作识别（AR）、活跃说话人定位（ASL）和行为预测（BA）。

### 关键设计

1. **跨模态特征蒸馏（CFD）**：使用轻量学生模型 Φ 逼近重型教师模型 Ω 的性能。从视觉帧 I、音频 A 和行为数据 B 中分别提取特征 $z_I, z_A, z_B$，通过融合网络 ξ 得到融合特征 $z_\phi$，使 $\Phi(I, A, B) \approx \Omega(V)$。训练损失包含三项：

    - L1 特征匹配损失：$\mathcal{L}_1 = \sum \| z_{\Omega_i} - z_{\phi_i} \|_1$
    - KL 散度知识蒸馏损失：$\mathcal{L}_{KD} = \sum D_{KL}(\sigma(\Omega(V)/\tau), \sigma(\Phi(I,A,B)/\tau))$
    - 交叉熵预测损失：$\mathcal{L}_{GT} = \sum \mathcal{L}_{CE}(c_i, \sigma(\Phi(I,A,B)))$
    - 综合损失：$\mathcal{L}_\Phi = \alpha \mathcal{L}_{KD} + (1-\alpha)\mathcal{L}_{GT} + \beta \mathcal{L}_1$

2. **任务感知多感官策略学习（TeMPLe）**：设计策略网络包含轻量模态特征提取器和 LSTM 模块，通过 **Gumbel-Softmax 采样**实现离散策略的可微训练。在每个时间步 t，LSTM 接收联合特征和历史隐藏状态，输出二值策略决策 $u_{t,k}$，决定是否启用模态 k。

    - 对 ASL 和 BA 任务：学习模态切换策略（选择哪些模态、哪些音频通道）
    - 对 AR 任务：采用**音频预览**策略——先分析音频检测兴趣区域，再从中选择最具信息量的单帧进行识别

3. **音频引导帧选择（AR 特有）**：通过多头注意力与递归 CNN 的"握手"机制提取时间感知音频特征，使用 LSTM 检测潜在事件区域，每个区域仅选一帧进行动作识别，实现极致的帧效率。握手公式为 $z_{l+1}^{RCNN} = z_l^{RCNN} + \rho z_l^{MH}$，其中 ρ 为可学习参数。

### 损失函数 / 训练策略

三阶段训练：
- **Stage 1**：禁用策略模块，仅训练蒸馏模块 Φ
- **Stage 2**：冻结 Φ，使用效率惩罚损失训练策略模块 Π
- **Stage 3**：联合微调 Φ 和 Π，总目标 $\mathcal{L}_\Theta = \eta_1 \mathcal{L}_\Pi + \eta_2 \mathcal{L}_\Phi$

策略训练中的效率惩罚：$\mathcal{C}_k = (|U_k|_0 / C)^2$，通过 λ_k 和 γ 控制精度与效率的权衡。

## 实验关键数据

### 主实验（表格）

| 方法 | Verb↑ | Noun↑ | Action↑ | GMACs↓ |
|------|-------|-------|---------|--------|
| TIM AV (教师) | 77.19 | 67.22 | 57.57 | 26.62 |
| Ego-only | 73.33 | 59.48 | 52.59 | 507.39 |
| AdaMML | 64.95 | 55.27 | 41.73 | 277.76 |
| EgoAdapt w/o TeMPLe | 68.34 | 59.02 | 50.88 | 5.79 |
| **EgoAdapt** | **76.65** | **66.83** | **56.74** | **7.14** |

ASL 任务（EasyCom）：EgoAdapt 达到 89.74% mAP，仅 0.070 GMACs 和 0.39M 参数，比 MAVSLC+E 的 6.852 GMACs 减少 ~99%。

### 消融实验（表格）

| λ_K 设置 | γ | AR Acc↑ | AR GMACs↓ | ASL mAP↑ | ASL GMACs↓ |
|----------|---|---------|-----------|----------|------------|
| [0,0,0] | 0 | 56.99 | 13.68 | 89.77 | 0.391 |
| [1,1,1] | 1 | 56.27 | 9.23 | 89.48 | 0.102 |
| [1,0.05,0.03] | 1 | 56.83 | 7.67 | 89.76 | 0.092 |
| [1,0.05,0.03] | 10 | 56.74 | 7.14 | 89.75 | 0.070 |

训练阶段消融：CFD + random policy 仅 67.41 mAP，Stage 1+2 为 83.64，联合训练到 Stage 3 达 89.74，证明联合训练的重要性。

### 关键发现

- EgoAdapt 在 AR 上以 7.14 GMACs 接近教师模型 TIM（26.62 GMACs）的 57.57% Action 准确率
- 在 ASL 上实现 82.02% 参数缩减，性能与教师 MUST 持平
- 行为预测上 MAE 降低 18.08%，能耗仅 0.003J（比 GLC 的 0.972J 大幅下降）
- 4-bit 量化后 ASL 仍达 78.92 mAP，功耗仅 9.94mW

## 亮点与洞察

- 首次将蒸馏和策略学习**统一到一个可微框架**中，通过 Gumbel-Softmax 解决离散策略优化问题
- 音频预览的单帧策略非常巧妙——用最便宜的模态指导最贵模态的使用
- 策略模块的动作空间可按任务灵活调整，具有良好的通用性
- 在 GTX2080Ti 上实现 >180 FPS，28% GPU 利用率

## 局限与展望

- 目前仅验证了三个自我中心任务，更复杂的手-物交互和长期活动理解待探索
- 策略空间可扩展到空间分辨率选择、网络量化等维度
- 音频预览策略对无声场景可能失效

## 相关工作与启发

- 相比 AdaMML 只做自适应多模态学习，EgoAdapt 额外引入蒸馏降低基础模型成本
- 相比 EgoDistill 只做蒸馏，EgoAdapt 增加了动态模态选择能力
- 联合训练思路可推广到其他多模态效率优化场景（如自动驾驶多传感器融合）

## 评分

- 新颖性: ⭐⭐⭐⭐ （联合蒸馏+策略的统一框架有创新性）
- 实验充分度: ⭐⭐⭐⭐⭐ （三任务三数据集+全面消融+量化+定性分析）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，公式完整）
- 价值: ⭐⭐⭐⭐ （对 AR/VR 边缘部署有实际意义）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Fine-grained Spatiotemporal Grounding on Egocentric Videos](fine-grained_spatiotemporal_grounding_on_egocentric_videos.md)
- [\[ICCV 2025\] MobileViCLIP: An Efficient Video-Text Model for Mobile Devices](mobileviclip_an_efficient_video-text_model_for_mobile_devices.md)
- [\[ICCV 2025\] General Compression Framework for Efficient Transformer Object Tracking](general_compression_framework_for_efficient_transformer_object_tracking.md)
- [\[ICCV 2025\] VideoMiner: Iteratively Grounding Key Frames of Hour-Long Videos via Tree-based Group Relative Policy Optimization](videominer_iteratively_grounding_key_frames_of_hour-long_videos_via_tree-based_g.md)
- [\[CVPR 2026\] VideoChat-M1: Collaborative Policy Planning for Video Understanding via Multi-Agent Reinforcement Learning](../../CVPR2026/video_understanding/videochatm1_collaborative_policy_planning_for_vide.md)

</div>

<!-- RELATED:END -->
