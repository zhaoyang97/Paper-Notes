---
title: >-
  [论文解读] Certified Human Trajectory Prediction
description: >-
  [CVPR 2025][自动驾驶][认证鲁棒性] 首次将随机平滑（Randomized Smoothing）认证技术引入人类轨迹预测任务，通过mean/median聚合函数和扩散去噪器为轨迹预测模型提供保证性鲁棒性——即无论输入噪声如何扰动（在半径R内），输出始终保持在认证边界内。 领域现状：人类轨迹预测是自动驾驶的关键任务…
tags:
  - "CVPR 2025"
  - "自动驾驶"
  - "认证鲁棒性"
  - "轨迹预测"
  - "随机平滑"
  - "扩散去噪器"
  - "对抗攻击防御"
---

# Certified Human Trajectory Prediction

**会议**: CVPR 2025  
**arXiv**: [2403.13778](https://arxiv.org/abs/2403.13778)  
**代码**: [https://s-attack.github.io/](https://s-attack.github.io/)  
**领域**: 自动驾驶 / 轨迹预测  
**关键词**: 认证鲁棒性, 轨迹预测, 随机平滑, 扩散去噪器, 对抗攻击防御

## 一句话总结

首次将随机平滑（Randomized Smoothing）认证技术引入人类轨迹预测任务，通过mean/median聚合函数和扩散去噪器为轨迹预测模型提供保证性鲁棒性——即无论输入噪声如何扰动（在半径R内），输出始终保持在认证边界内。

## 研究背景与动机

**领域现状**：人类轨迹预测是自动驾驶的关键任务，现有数据驱动方法（Social-LSTM、EqMotion等）在干净输入上表现优秀，但已被证明对对抗攻击和感知噪声极其脆弱。

**现有痛点**：之前的鲁棒性防御方法（数据增强、对抗训练等）都是启发式的，没有保证性——面对足够强的攻击者终究会失效。轨迹预测的输出是无界的连续值（不像分类有有限类别），这使得直接套用图像分类中的认证方法存在根本困难。

**核心矛盾**：轨迹预测的三大特殊挑战：(1) 输出无界——连续坐标没有天然的上下界；(2) 多模态性——多条合理预测轨迹如何认证；(3) 性能下降——平滑操作本身会降低预测精度。

**核心idea**：将随机平滑泛化到多输出回归任务，提出自适应截断解决无界问题，引入扩散去噪器缓解精度下降，设计新认证指标评估模型在噪声下的真实可靠性。

## 方法详解

### 整体框架

输入观测轨迹 $X$ → 添加 $n$ 个高斯扰动 $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$ → 去噪器 $h$ 预处理 → 预测器 $g$ 得到 $n$ 个预测 → 聚合函数 $\mathcal{A}$（mean或median）得到最终平滑预测 $Y$ 及认证边界。

### 关键设计

1. **Mean vs Median平滑**：

    - Mean平滑：需要输出有界 $f: \mathbb{R}^d \to [l, u]$，认证边界通过正态CDF $\Phi$ 计算；要求知道输出上下界
    - Median平滑：不需要输出有界，通过分位数函数计算边界；更鲁棒因为不受异常值影响
    - 核心发现：**median聚合显著优于mean**——因为轨迹预测器对噪声输入可能产生极端异常预测，mean容易被拉偏，median更稳健

2. **自适应截断（Adaptive Clamping）**：

    - 功能：为无界的轨迹预测器补上输出上下界
    - 核心思路：在训练集上遍历预测器，统计每个坐标维度的最大/最小值作为 $l_j, u_j$，然后用 $\min(u_j, \max(l_j, \cdot))$ 截断
    - 仅mean平滑需要，median平滑不需要此步骤

3. **扩散去噪器**：

    - 功能：在平滑操作前预处理噪声输入，缓解精度下降
    - 核心思路：训练一个无条件扩散模型学习轨迹数据分布，推理时用多步去噪恢复干净轨迹 $h(X + \epsilon) \approx X$
    - 效果：显著收紧认证边界（FBD从0.96降至0.65，在相同FDE=1.3下），同时保持预测精度
    - 对比Wiener滤波器、移动平均、多项式拟合，扩散去噪器残余噪声最小

4. **认证指标设计**：

    - **ABD/FBD**（Average/Final Bound half-Diameter）：衡量认证边界大小
    - **Certified-ADE/FDE**：认证边界内最坏情况的位移误差，反映噪声下的真实性能
    - **Certified-Collision Rate**：邻近agent是否落入预测的认证边界内
    - 核心发现：**最准确的模型不一定最鲁棒**——EqMotion的FDE最低(1.12)但Certified-FDE并非最低，D-Pool的Certified-FDE(2.0)反而最好

### 多模态/多agent扩展

- 多模态：对k=20个模式分别认证，选Certified-FDE最小的模式。多模态FBD(0.64)远小于单模态(0.99)
- 多agent：对所有agent的轨迹同时加噪，相互依赖放大认证边界(FBD 1.21 vs 单agent 0.99)

## 实验关键数据

### 主实验：平滑vs非平滑预测器对比

| 模型 | FDE | Certified-FDE | Col(%) | Certified-Col(%) |
|------|-----|---------------|--------|-------------------|
| Social-Force | 1.25 | N/A | 7.4 | N/A |
| Smoothed Social-Force | 1.26 | 2.27 | 8.0 | 46 |
| D-Pool | 1.14 | N/A | 9.4 | N/A |
| Smoothed D-Pool | 1.23 | **2.00** | 9.0 | 49 |
| EqMotion | **1.12** | N/A | 10.1 | N/A |
| Smoothed EqMotion | 1.14 | 2.07 | 10.6 | 57 |

### 消融实验：去噪器效果

| 配置 | FDE=1.2时FBD | FDE=1.3时FBD | FDE=1.4时FBD |
|------|:-:|:-:|:-:|
| 无去噪器 | 1.20 | 0.96 | 0.80 |
| **有扩散去噪器** | **0.78** | **0.65** | **0.57** |

### 关键发现

- **精度代价极小**：平滑操作仅增加1-6%的FDE（EqMotion: 1.12→1.14），但获得了保证性鲁棒性
- **最准≠最鲁棒**：EqMotion精度最高但D-Pool认证鲁棒性最强，揭示仅看FDE会遗漏安全隐患
- **碰撞率差距巨大**：Col vs Certified-Col差距高达40%+（10.1% vs 57%），说明噪声下碰撞风险被严重低估
- **计算开销可接受**：n=100次Monte Carlo仅比单次推理慢42%（0.07s→0.1s），可并行化
- **下游任务受益**：在机器人导航任务中，带去噪器的鲁棒模型碰撞率从21%降至15.1%

## 亮点与洞察

- **首次轨迹预测认证**：将随机平滑从分类扩展到多输出时序回归，解决了无界输出和多模态两大技术挑战。这为安全关键系统的AI可靠性提供了新范式
- **"最准≠最安全"的警示**：这个发现对自动驾驶社区有重要意义——仅追求FDE/ADE可能引入安全隐患，应同时评估鲁棒性指标
- **扩散模型做轨迹去噪**：将扩散模型用于时序数据去噪（而非生成），应用方向新颖且效果显著
- **σ的精度-鲁棒性权衡**：通过调节σ可灵活控制精度和认证边界的平衡，为实际部署提供了可调节的安全裕度

## 局限与展望

- 随机平滑需要n次前向计算，实时性受限（虽然可并行）
- 自适应截断的上下界依赖训练集分布，新场景可能需要重新校准
- 目前仅在Trajnet++上验证，未在更大规模的真实驾驶数据集(nuScenes full)上测试
- 多agent认证边界显著增大，密集场景下实用性待验证

## 相关工作与启发

- **vs Trajpac**：Trajpac用PAC策略做概率性验证，但依赖噪声分布、需要30000+样本且无保证性。本文仅需100样本、噪声分布无关、有保证性
- **vs 对抗训练**：对抗训练只能提供经验鲁棒性，面对更强攻击者会失效。随机平滑提供数学保证
- **vs Conformal Prediction**：CP保证真值覆盖率，本文保证输出区域——两者互补

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将认证鲁棒性引入轨迹预测，解决了多个非平凡的技术挑战
- 实验充分度: ⭐⭐⭐⭐ 多模型、多模态、多agent、下游任务验证全面，但缺乏大规模数据集
- 写作质量: ⭐⭐⭐⭐⭐ 问题形式化清晰，理论推导严谨，新指标设计合理
- 价值: ⭐⭐⭐⭐⭐ 为安全关键的自动驾驶系统提供了可靠性保障新工具，具有重要实用和学术价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Physical Plausibility-aware Trajectory Prediction via Locomotion Embodiment](physical_plausibility-aware_trajectory_prediction_via_locomotion_embodiment.md)
- [\[CVPR 2025\] Tra-MoE: Learning Trajectory Prediction Model from Multiple Domains for Adaptive Policy Conditioning](tra-moe_learning_trajectory_prediction_model_from_multiple_domains_for_adaptive_.md)
- [\[ECCV 2024\] Adaptive Human Trajectory Prediction via Latent Corridors](../../ECCV2024/autonomous_driving/adaptive_human_trajectory_prediction_via_latent_corridors.md)
- [\[CVPR 2025\] SocialMOIF: Multi-Order Intention Fusion for Pedestrian Trajectory Prediction](socialmoif_multi-order_intention_fusion_for_pedestrian_trajectory_prediction.md)
- [\[ECCV 2024\] Progressive Pretext Task Learning for Human Trajectory Prediction](../../ECCV2024/autonomous_driving/progressive_pretext_task_learning_for_human_trajectory_prediction.md)

</div>

<!-- RELATED:END -->
