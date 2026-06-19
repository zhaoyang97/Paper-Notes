---
title: >-
  [论文解读] RIFLEx: A Free Lunch for Length Extrapolation in Video Diffusion Transformers
description: >-
  [ICML 2025][视频生成][Transformer] 通过系统分析RoPE位置编码中各频率分量的角色，发现存在一个"固有频率"主导外推时的时间重复行为，提出仅降低该频率使其在外推后保持单周期的最小化方案RIFLEx，在CogVideoX-5B和HunyuanVideo上实现无训练2×高质量视频外推。
tags:
  - "ICML 2025"
  - "视频生成"
  - "Transformer"
  - "RoPE"
  - "频率分析"
  - "长度外推"
  - "training-free"
---

# RIFLEx: A Free Lunch for Length Extrapolation in Video Diffusion Transformers

**会议**: ICML 2025  
**arXiv**: [2502.15894](https://arxiv.org/abs/2502.15894)  
**代码**: [riflex-video.github.io](https://riflex-video.github.io/)  
**领域**: 视频生成 / 长度外推  
**关键词**: 视频扩散Transformer, RoPE, 频率分析, 长度外推, training-free

## 一句话总结
通过系统分析RoPE位置编码中各频率分量的角色，发现存在一个"固有频率"主导外推时的时间重复行为，提出仅降低该频率使其在外推后保持单周期的最小化方案RIFLEx，在CogVideoX-5B和HunyuanVideo上实现无训练2×高质量视频外推。

## 研究背景与动机
**领域现状**：视频扩散Transformer（如CogVideoX、HunyuanVideo）能生成高质量分钟级视频，但受限于训练长度上限，无法直接生成更长视频。

**现有痛点**：(1) 文本/图像领域的长度外推方法（PE、PI、NTK、YaRN）直接应用于视频时失败——出现时间重复（视频循环播放）或运动减速（帧被拉伸）两种典型失败模式；(2) 视频外推的目标与文本/图像本质不同——需要生成时间上连贯演进的新内容，而非扩展上下文窗口或增加分辨率细节。

**核心矛盾**：如何在不改变模型权重的前提下，抑制时间重复同时保持运动一致性？

**切入角度**：逐一隔离RoPE中的各频率分量，通过置零+微调实验分析其对视频生成的影响。

**核心 idea**：降低"固有频率"使其在外推长度内保持不超过一个完整周期，即可消除重复。

## 方法详解

### 整体框架
RIFLEx分三步：(1) 分析RoPE各频率分量 $\theta_j$ 的周期 $N_j = 2\pi/\theta_j$；(2) 识别固有频率分量 $k = \arg\min_j |N_j - N|$，其中 $N$ 是首次观测到的重复帧位置；(3) 仅修改该分量 $\theta_k' = 2\pi/(Ls)$，使其周期覆盖整个外推长度。其他频率完全不变。

### 关键设计
1. **频率分量角色分析**:
    - 功能：通过隔离实验揭示RoPE中不同频率分量对视频生成的影响
    - 核心思路：将RoPE中除某一频率 $\theta_j$ 外的所有分量置零，微调模型后观察生成行为。发现高频分量（$r_j = L\theta_j/(2\pi) > 1$）捕捉短期依赖和快速运动，导致时间重复；低频分量（$r_j < 1$）编码长期依赖但导致运动减速
    - 设计动机：已有方法（PE/PI/NTK/YaRN）对所有频率统一操作，缺乏对各频率角色的理解

2. **固有频率识别**:
    - 功能：找到主导视频外推重复行为的关键频率分量
    - 核心思路：定义固有频率为周期 $N_j$ 最接近首次重复帧 $N$ 的分量：$k = \arg\min_j |N_j - N|$。实验发现对于同一模型，不同视频的固有频率保持一致（CogVideoX-5B为 $k=2$，HunyuanVideo为 $k=4$）
    - 设计动机：并非所有频率都导致重复——只需修改这一个关键频率即可

3. **最小化频率修改**:
    - 功能：仅修改固有频率使其在外推后不超过一个周期
    - 核心思路：将 $\theta_k$ 降低为 $\theta_k' = 2\pi/(Ls)$，其中 $s$ 是外推倍数。这确保 $N_k' = Ls \geq Ls$，即外推后仍在单周期内。消融实验证实修改更高频分量会破坏快速运动，修改更低频分量则几乎无效
    - 设计动机：最小化修改 = 最小化训练-推理mismatch，使2×外推无需任何微调即可工作

### 损失函数 / 训练策略
2×外推完全无需训练（training-free）。若需3×外推或进一步提升质量，仅需用2万条原始长度视频微调（仅为预训练计算量的1/50000），使用标准扩散训练损失。

## 实验关键数据

### 主实验（CogVideoX-5B 2×外推，training-free）

| 方法 | NoRepeat Score↑ | Dynamic Degree↑ | Imaging Quality↑ | Overall Consistency↑ | User-Overall排名↓ |
|------|----------------|----------------|-------------------|---------------------|-------------------|
| PE | 46.6 | 58.6 | 55.0 | 22.9 | 2.4 |
| NTK | 43.4 | 58.3 | 55.3 | 22.9 | 2.1 |
| PI | 59.0 | **5.0** | 44.3 | 19.2 | 3.8 |
| YaRN | 59.4 | **5.6** | 44.6 | 19.3 | 3.7 |
| TASR | 10.8 | 26.9 | 50.5 | 21.5 | 3.6 |
| **RIFLEx** | 54.2 | 59.4 | **56.9** | **23.5** | **1.1** |

### 消融实验 / 扩展

| 配置 | 说明 |
|------|------|
| 2× training-free | NoRepeat 54.2, Dynamic 59.4——无训练即可高质量外推 |
| 2× fine-tuned | NoRepeat 61.3, Dynamic 54.7——微调进一步提升 |
| HunyuanVideo 2× | 用户综合排名1.6，NTK为1.6但Dynamic更低 |
| 空间外推 480p→960p | 同样适用于图像空间分辨率外推 |
| 时空联合外推 | 同时2×时间+2×空间均有效 |

### 关键发现
- PI/YaRN完全消除了重复，但Dynamic Degree仅5-6（几乎静止画面）——运动减速是致命问题
- PE/NTK的NoRepeat Score仅43-47，严重时间重复
- RIFLEx在用户研究中以61.6-70.2%的比例优于原始训练长度视频
- 固有频率在同一模型的不同视频间保持一致

## 亮点与洞察
- "一个频率决定一切"的insight极其优雅——复杂的外推问题归结为修改一个标量值
- 真正的"free lunch"——2×外推零额外训练成本，零参数修改，仅改一个RoPE频率
- 为现有方法（PE/PI/NTK/YaRN/TASR）的失败提供了统一的原理性解释
- 时间外推 + 空间外推 + 联合外推的统一框架，展示了该insight的通用性

## 局限与展望
- 固有频率的识别需要预先生成视频观察重复帧位置，缺少自动化方法
- 3×外推仍需微调，更大倍数外推（4×+）的效果未知
- 仅在CogVideoX-5B和HunyuanVideo两个模型上验证
- 对于固有频率在不同视频间不一致的罕见情况，需要特殊处理

## 相关工作与启发
- **vs NTK-Aware RoPE**: 调整所有频率的base $b$，但对视频仍导致时间重复（NoRepeat 43.4）。RIFLEx证明只需改一个频率
- **vs PI**: 均匀压缩所有频率 $\theta/s$，消除重复但运动完全停滞（Dynamic仅5.0）。RIFLEx保留高频运动信息
- **vs YaRN**: 分组调整策略在视频上同样导致运动减速（Dynamic 5.6）
- **vs TASR**: CogVideoX/HunyuanVideo使用3D RoPE，TASR的时间步自适应策略在此基础上效果有限
- 启示：视频外推问题中，频率分量的角色分析为未来设计更好位置编码提供方向

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ insight新颖优雅，"固有频率"概念有力，解决方案极简
- 实验充分度: ⭐⭐⭐⭐ 两个SOTA模型、多种外推倍数、时空联合、用户研究
- 写作质量: ⭐⭐⭐⭐⭐ 问题分析透彻，从失败模式→频率分析→固有频率→解决方案，逻辑链完美
- 价值: ⭐⭐⭐⭐⭐ 对视频生成社区有范式性贡献，training-free解决方案极具实用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] AsymRnR: Video Diffusion Transformers Acceleration with Asymmetric Reduction and Restoration](asymrnr_video_diffusion_transformers_acceleration_with_asymmetric_reduction_and_.md)
- [\[CVPR 2026\] Free-Lunch Long Video Generation via Layer-Adaptive O.O.D Correction](../../CVPR2026/video_generation/free-lunch_long_video_generation_via_layer-adaptive_ood_correction.md)
- [\[CVPR 2025\] DiTFlow: Video Motion Transfer with Diffusion Transformers](../../CVPR2025/video_generation/video_motion_transfer_with_diffusion_transformers.md)
- [\[ICCV 2025\] MagicMirror: ID-Preserved Video Generation in Video Diffusion Transformers](../../ICCV2025/video_generation/magicmirror_id-preserved_video_generation_in_video_diffusion_transformers.md)
- [\[ICCV 2025\] Decouple and Track: Benchmarking and Improving Video Diffusion Transformers for Motion Transfer](../../ICCV2025/video_generation/decouple_and_track_benchmarking_and_improving_video_diffusion_transformers_for_m.md)

</div>

<!-- RELATED:END -->
