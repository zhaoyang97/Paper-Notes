---
title: >-
  [论文解读] AdaDrive: Self-Adaptive Slow-Fast System for Language-Grounded Autonomous Driving
description: >-
  [ICCV 2025][自动驾驶][大语言模型] AdaDrive提出了首个自适应慢-快架构的LLM增强自动驾驶框架，通过两个自适应连接器动态决定"何时激活LLM"（Connector-W）和"LLM贡献多少"（Connector-H），在语言引导驾驶基准上实现了SOTA性能（驾驶分数80.9%），同时将推理延迟降低至189ms、显存降至6.79GB。
tags:
  - "ICCV 2025"
  - "自动驾驶"
  - "大语言模型"
  - "自适应慢快系统"
  - "语言引导驾驶"
  - "高效推理"
---

# AdaDrive: Self-Adaptive Slow-Fast System for Language-Grounded Autonomous Driving

**会议**: ICCV 2025  
**arXiv**: [2511.06253](https://arxiv.org/abs/2511.06253)  
**代码**: [https://github.com/ReaFly/AdaDrive](https://github.com/ReaFly/AdaDrive)  
**领域**: 自动驾驶  
**关键词**: 大语言模型, 自动驾驶, 自适应慢快系统, 语言引导驾驶, 高效推理

## 一句话总结
AdaDrive提出了首个自适应慢-快架构的LLM增强自动驾驶框架，通过两个自适应连接器动态决定"何时激活LLM"（Connector-W）和"LLM贡献多少"（Connector-H），在语言引导驾驶基准上实现了SOTA性能（驾驶分数80.9%），同时将推理延迟降低至189ms、显存降至6.79GB。

## 研究背景与动机
- **领域现状**：LLM在自动驾驶中可以提供高级推理和决策能力，但如何高效集成仍是开放问题
- **第一代方法**（LMDrive、AD-H）：同步架构，LLM每步都参与——推理精确但延迟大、显存高，无法实时部署
- **第二代方法**（AsyncDriver、DriveVLM）：异步架构，固定频率激活LLM——减少了开销但无法适应动态驾驶场景。紧急情况下可能不激活LLM，简单场景下又浪费计算资源
- **核心矛盾**：高频激活LLM保证性能但延迟不可接受；低频固定激活遗漏关键场景且不灵活
- **本文关键洞察**：（1）LLM不应固定频率激活，而应基于场景复杂度自适应决定；（2）LLM的贡献不应是全有或全无的二值决策——连续加权融合（如0.7权重）优于全权重（1.0），这从实验中得到验证（Table 4: ID#3 vs ID#4）

## 方法详解

### 整体框架
AdaDrive采用并行慢-快双路径架构：快路径（轻量planner）高频处理每帧数据进行轨迹预测；慢路径（LLM）低频激活作为认知单元提供决策辅助。两条路径通过Connector-W和Connector-H自适应连接，另有LS-Qformer处理时序特征和流式记忆缓冲区管理历史上下文。

### 关键设计

1. **Connector-W：自适应LLM激活**:

    - 功能：动态判断当前帧是否需要激活LLM
    - 核心思路：用MLP从当前驾驶上下文特征 $f_T'$ 预测置信度分数 $\theta_T$，通过Gumbel-Softmax转换为二值决策 $\pi_T \in \{0, 1\}$
    - 自适应激活损失（核心创新）：
    $\mathcal{L}_{ada} = \pi_T \cdot (\mathcal{L}_T^{LLM} + \gamma) + (1-\pi_T) \cdot \mathcal{L}_T$
      其中 $\gamma = \max(d - (L_T - L_T^{LLM}), 0)$
    - 训练机制：每步做两次前向传播——一次有LLM辅助（$W_T^{LLM}$），一次无LLM（$W_T$），对比两者的轨迹损失。当LLM帮助显著时（$\mathcal{L}_T^{LLM} << \mathcal{L}_T$）自动学会激活LLM
    - 惩罚项 $\gamma$ 的作用：通过预设边距 $d=0.3$ 控制激活频率，确保LLM只在贡献足够显著时才被激活
    - 设计动机：无需人工标注"何时需要LLM"的ground truth，通过比较学习自动发现最优激活时机

2. **Connector-H：动态LLM贡献缩放**:

    - 功能：在LLM被激活时，控制LLM特征对轨迹预测的贡献程度
    - 核心思路：使用Connector-W预测的置信度 $\theta_T$ 作为连续权重系数，而非简单的全权重融合
    - 融合公式：$W_T^{Fuse} = \mathcal{P}(f_T' + \theta_T \cdot f_T'')$
    - 统一推理公式：
    $W_T = \begin{cases} \mathcal{P}(f_T'), & \text{LLM未激活} \\ \mathcal{P}(f_T' + \theta_T \cdot f_T''), & \text{LLM已激活} \end{cases}$
    - 设计动机：实验证明连续加权（如$\theta_T=0.7$）效果优于二值全权重（$\theta_T=1.0$），自适应缩放使融合更精细

3. **Long-Short Q-former (LS-Qformer)**:

    - 功能：增强视觉特征的时序建模，同时兼顾当前帧精度和长程上下文保持
    - 核心思路：将可学习token分为两组——记忆token $\mathbf{Q}^m$ 在帧间传播聚合长程信息，局部token $\mathbf{Q}^l$ 关注当前帧
    - 公式：$f_T' = [\mathbf{Q}^l; \mathbf{Q}_T^m] = \text{Q-former}(\mathbf{Q}^l, \mathbf{Q}_{T-1}^m, f_T, \mathbf{I}_T)$
    - 超参数：20个局部token + 20个记忆token
    - 设计动机：标准Q-former逐帧独立处理忽略了时序依赖；LS-Qformer通过分组机制同时提取当前帧关键特征和建模时序演化

4. **传播式记忆融合 (PMF)**:

    - 功能：管理流式数据的固定大小记忆缓冲区，防止显存无限增长
    - 核心思路：当缓冲区满时，将即将淘汰帧的特征融合到相邻帧中：$\hat{f}_{T-k+1}' = (f_{T-k}' + f_{T-k+1}')/2$
    - 对比FIFO：FIFO直接丢弃最旧帧的信息；PMF通过融合保留了历史上下文
    - 缓冲区容量：$k=10$

### 损失函数 / 训练策略
- 使用AdamW优化器，余弦学习率调度，初始学习率 $1 \times 10^{-5}$
- 训练15个epoch，自适应激活损失中的边距 $d=0.3$
- 视觉编码器来自LMDrive预训练并冻结，LLM使用TinyLLaMA (1.1B)
- Planner为4层Transformer，仅3M参数
- 训练阶段包含warmup使LLM和非LLM的轨迹损失先收敛到稳定值

## 实验关键数据

### 主实验

| 方法 | LLM参数量 | DS ↑ | RC ↑ | IS ↑ | 显存 ↓ | 推理时间 ↓ |
|------|-----------|------|------|------|--------|-----------|
| LMDrive (LLaMA2-7B) | 7B | 32.8 | 40.1 | 0.81 | 26.91G | 526ms |
| LMDrive (TinyLLaMA) | 1.1B | 25.2 | 38.6 | 0.71 | 16.29G | 445ms |
| AD-H (Mipha-3B) | 3.35B | 41.1 | 48.5 | 0.86 | - | - |
| **AdaDrive** | **1.1B+3M** | **42.9** | **53.4** | **0.82** | **6.79G** | **189ms** |

### 消融实验

| ID | Connector-W | Connector-H | LS-Qformer | DS ↑ | RC ↑ | IS ↑ |
|----|-------------|-------------|------------|------|------|------|
| 1 | ✗ | ✗ | ✗ | 67.4 | 75.3 | 0.86 |
| 2 | ✗ | ✗ | ✓ | 71.9 | 82.6 | 0.84 |
| 3 | ✓ | ✗ | ✓ | 77.9 | 84.8 | 0.89 |
| 4 | ✓ | ✓ | ✓ | **80.9** | **87.6** | **0.90** |

### 关键发现
- 自适应激活的平均频率仅为0.28（短距离）和0.33（全程），但性能接近全激活（频率=1.0），GFLOPs降低62%
- 困难路线（密集城市街道、夜间、山路）的激活频率更高，验证了自适应机制的合理性
- 时序分布分析显示LLM主要在转弯和十字路口等关键时刻被激活，巡航阶段保持沉默
- LS-Qformer相比标准Q-former：DS从75.8提升到80.9（+5.1）
- PMF优于FIFO硬替换策略，较小的记忆缓冲区（$k=10$）反而效果最优

## 亮点与洞察
- **自适应激活的损失设计精妙**：不需要ground truth标注何时该激活LLM，通过训练时的比较学习自动发现最优时机。这个设计具有通用性，可推广到其他需要按需激活昂贵模块的系统
- **连续融合优于二值融合**：Connector-H的实验结果（ID#3 vs ID#4）揭示了一个有价值的洞察——LLM的输出不应该被全量使用，自适应权重更优
- **极致的效率**：使用1.1B小模型+3M planner，在显存和速度上远优于7B模型方案（6.79G vs 26.91G，189ms vs 526ms），同时性能更强
- **LLM激活模式的可解释性**：激活主要集中在转弯和交叉路口，符合直觉

## 局限与展望
- Connector-W的训练需要每步做两次前向传播（有/无LLM），增加训练成本
- 惩罚项的边距 $d=0.3$ 是预设超参数，不同场景可能需要调整
- PMF的简单平均融合可能不是最优选择，注意力加权融合可能更好
- 长距离任务上IS分数（0.82）低于AD-H（0.86），安全性方面仍有提升空间
- 仅在CARLA仿真中验证，真实世界部署效果未知

## 相关工作与启发
- **vs LMDrive**：同步每步调用LLM，延迟526ms不可接受；AdaDrive自适应调用仅189ms
- **vs AsyncDriver/DriveVLM**：固定频率激活，无法适应动态场景；AdaDrive按需激活
- **vs AD-H**：AD-H使用额外中级语言命令训练分层多智能体系统，参数总量更大（3.35B）；AdaDrive更轻量但性能更强
- **vs Flash-VStream**：视频理解关注高层语义对话，自动驾驶关注低层高频轨迹预测，两者目标不同

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 自适应激活损失和连续融合的设计非常创新，双连接器架构优雅
- 实验充分度: ⭐⭐⭐⭐ 消融完整，激活模式分析有说服力，但仅限CARLA仿真
- 写作质量: ⭐⭐⭐⭐ 动机清晰，图表信息丰富
- 价值: ⭐⭐⭐⭐⭐ 为LLM在自动驾驶中的高效部署提供了实用范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] AD-GS: Object-Aware B-Spline Gaussian Splatting for Self-Supervised Autonomous Driving](ad-gs_object-aware_b-spline_gaussian_splatting_for_self-supervised_autonomous_dr.md)
- [\[ICCV 2025\] Hermes: A Unified Self-Driving World Model for Simultaneous 3D Scene Understanding and Generation](hermes_a_unified_self-driving_world_model_for_simultaneous_3d_scene_understandin.md)
- [\[CVPR 2026\] MeanFuser: Fast One-Step Multi-Modal Trajectory Generation and Adaptive Reconstruction via MeanFlow for End-to-End Autonomous Driving](../../CVPR2026/autonomous_driving/meanfuser_fast_one-step_multi-modal_trajectory_generation_and_adaptive_reconstru.md)
- [\[CVPR 2026\] Counterfactual VLA: Self-Reflective Vision-Language-Action Model with Adaptive Reasoning](../../CVPR2026/autonomous_driving/counterfactual_vla_self-reflective_vision-language-action_model_with_adaptive_re.md)
- [\[CVPR 2026\] KnowVal: A Knowledge-Augmented and Value-Guided Autonomous Driving System](../../CVPR2026/autonomous_driving/knowval_a_knowledge-augmented_and_value-guided_autonomous_driving_system.md)

</div>

<!-- RELATED:END -->
