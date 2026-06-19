---
title: >-
  [论文解读] SOLVE: Synergy of Language-Vision and End-to-End Networks for Autonomous Driving
description: >-
  [CVPR 2025][自动驾驶][VLM协同] 提出 SOLVE，通过共享 SQ-Former 视觉编码器实现 VLM 和端到端驾驶模型的特征级协同，用 Trajectory Chain-of-Thought（T-CoT）将 VLM 的长程轨迹作为 E2E 模型的初始化先验，在 nuScenes 上达到 0.28m 平均 L2 误差 SOTA。
tags:
  - "CVPR 2025"
  - "自动驾驶"
  - "VLM协同"
  - "端到端驾驶"
  - "轨迹思维链"
  - "时序解耦"
  - "共享编码器"
---

# SOLVE: Synergy of Language-Vision and End-to-End Networks for Autonomous Driving

**会议**: CVPR 2025  
**arXiv**: [2505.16805](https://arxiv.org/abs/2505.16805)  
**代码**: 无  
**领域**: 自动驾驶 / 端到端驾驶  
**关键词**: VLM协同, 端到端驾驶, 轨迹思维链, 时序解耦, 共享编码器

## 一句话总结

提出 SOLVE，通过共享 SQ-Former 视觉编码器实现 VLM 和端到端驾驶模型的特征级协同，用 Trajectory Chain-of-Thought（T-CoT）将 VLM 的长程轨迹作为 E2E 模型的初始化先验，在 nuScenes 上达到 0.28m 平均 L2 误差 SOTA。

## 研究背景与动机

**领域现状**：端到端自动驾驶（E2E）直接从传感器输出规划轨迹，但缺乏语义理解能力。VLM 有强大的场景理解和推理能力，但推理延迟高且无法直接输出精确控制信号。

**现有痛点**：VLM 和 E2E 模型各有优势——VLM 擅长理解"应该怎么做"，E2E 擅长执行"具体怎么做"。但两者通常独立设计，信息传递仅通过语言中间表示（如"减速"），丢失了大量空间信息。

**核心矛盾**：VLM 推理慢但理解深，E2E 响应快但理解浅。如何让两者在特征层面深度协同，而非仅在语言层面串联？

**切入角度**：共享视觉编码器让两个模型"看到同样的东西"，VLM 的长程轨迹异步存入记忆供 E2E 实时查询，避免 VLM 成为延迟瓶颈。

**核心 idea**：共享 SQ-Former + T-CoT 轨迹先验 + 异步时序解耦 = VLM 与 E2E 的深度协同。

## 方法详解

### 关键设计

1. **共享 Sequential Q-Former (SQ-Former)**:

    - 功能：为 VLM 和 E2E 模型提供统一的视觉编码
    - 核心思路：384 个可学习查询依次与图像特征、检测 token、车道 token 做交叉注意力，按 Q→Img→Det→Lane 顺序编码。编码结果同时输入 VLM 分支和 E2E 分支
    - 设计动机：共享编码器让 VLM 的语义理解能力通过梯度反向传播增强 E2E 的视觉特征（VLM 分支对 E2E 提升 1.5cm L2，反之对 VLM 提升 0.6cm）

2. **Trajectory Chain-of-Thought (T-CoT)**:

    - 功能：将轨迹规划从离散文本推理转为连续空间推理
    - 核心思路：维护 36 条候选轨迹 bank（通过 K-means 从训练数据中聚类），VLM 先选择最佳粗轨迹，再用轨迹 token 精化。相比直接回归坐标，这种从粗到细的方式让 VLM 的推理更可靠
    - 设计动机：6 条参考轨迹时最优（4 条太少混淆，8 条太多干扰）

3. **异步时序解耦**:

    - 功能：解决 VLM 推理延迟不适合实时控制的问题
    - 核心思路：VLM 低频生成长程轨迹（如 3 秒），存入全局记忆。E2E 高频运行，每步从记忆中取 VLM 轨迹作为初始化先验
    - 设计动机：VLM 不需要每帧推理——它的价值在于长程理解，E2E 负责实时精化

### 损失函数 / 训练策略

三阶段训练：QA 训练（交叉熵）→ 轨迹适配器（MSE）→ VLM+E2E 联合训练（L2 轨迹损失）。

## 实验关键数据

### 主实验

nuScenes 开环规划 L2 误差↓：

| 方法 | 1s | 2s | 3s | 平均 | 碰撞率 |
|------|-----|-----|-----|------|--------|
| OmniDrive | 0.17 | 0.30 | 0.55 | 0.33 | 0.25% |
| **SOLVE-VLM** | **0.15** | **0.23** | **0.47** | **0.28** | **0.20%** |

### 消融实验

| 配置 | 平均 L2 | 说明 |
|------|---------|------|
| 无共享 SQ-Former | 0.30 | 共享贡献 0.02m |
| 无 T-CoT | 0.295 | T-CoT 贡献 0.015m |
| 完整 SOLVE | **0.28** | — |

### 关键发现
- 共享编码器是双向互利的——VLM 的语义增强了 E2E 特征，E2E 的空间精度改善了 VLM 的轨迹预测
- T-CoT 的参考轨迹数量 6 条最优——从候选中选择比直接回归更稳定
- 异步解耦让实时部署成为可能

## 亮点与洞察
- **特征级协同 > 语言级串联**——共享编码器让两个模型在特征空间中深度融合，比用文字传递信息更高效
- **异步设计的实用性**——VLM 不需要实时运行，长程理解存入记忆供 E2E 按需取用

## 局限与展望
- 仅 nuScenes 开环评估——无闭环验证
- VLM 推理仍有延迟成本
- 需要 OmniDrive-nuScenes QA 数据集

## 评分
- 新颖性: ⭐⭐⭐⭐ VLM-E2E 特征级协同设计有启发性
- 实验充分度: ⭐⭐⭐⭐ 详细消融，但仅开环
- 写作质量: ⭐⭐⭐⭐ 清晰
- 价值: ⭐⭐⭐⭐ 为 VLM+E2E 融合提供了实用框架

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving](diffusiondrive_truncated_diffusion_model_for_end-to-end_autonomous_driving.md)
- [\[CVPR 2026\] DriveMoE: Mixture-of-Experts for Vision-Language-Action Model in End-to-End Autonomous Driving](../../CVPR2026/autonomous_driving/drivemoe_mixture-of-experts_for_vision-language-action_model_in_end-to-end_auton.md)
- [\[CVPR 2025\] RC-AutoCalib: An End-to-End Radar-Camera Automatic Calibration Network](rc-autocalib_an_end-to-end_radar-camera_automatic_calibration_network.md)
- [\[CVPR 2026\] E3AD: An Emotion-Aware Vision-Language-Action Model for Human-Centric End-to-End Autonomous Driving](../../CVPR2026/autonomous_driving/e3ad_an_emotion-aware_vision-language-action_model_for_human-centric_end-to-end_.md)
- [\[ICCV 2025\] World4Drive: End-to-End Autonomous Driving via Intention-aware Physical Latent World Model](../../ICCV2025/autonomous_driving/world4drive_end-to-end_autonomous_driving_via_intention-aware_physical_latent_wo.md)

</div>

<!-- RELATED:END -->
