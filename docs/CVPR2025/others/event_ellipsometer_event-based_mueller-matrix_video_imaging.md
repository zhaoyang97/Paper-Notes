---
title: >-
  [论文解读] Event Ellipsometer: Event-based Mueller-Matrix Video Imaging
description: >-
  [CVPR 2025][事件相机] 首个实现 30fps 视频级穆勒矩阵成像的系统——用事件相机捕捉快速旋转 QWP 产生的光强调制，将事件时间差映射到穆勒矩阵比值，通过 SVD 估计+时空传播重建物理有效的穆勒矩阵视频。 领域现状 领域现状：穆勒矩阵（Mueller Matrix）完整描述了材料对偏振光的变换特性…
tags:
  - "CVPR 2025"
  - "事件相机"
  - "穆勒矩阵"
  - "偏振成像"
  - "高速成像"
  - "HDR"
---

# Event Ellipsometer: Event-based Mueller-Matrix Video Imaging

**会议**: CVPR 2025  
**arXiv**: [2411.17313](https://arxiv.org/abs/2411.17313)  
**代码**: 无  
**领域**: LLM评测  
**关键词**: 事件相机、穆勒矩阵、偏振成像、高速成像、HDR

## 一句话总结
首个实现 30fps 视频级穆勒矩阵成像的系统——用事件相机捕捉快速旋转 QWP 产生的光强调制，将事件时间差映射到穆勒矩阵比值，通过 SVD 估计+时空传播重建物理有效的穆勒矩阵视频。

## 研究背景与动机

### 领域现状

**领域现状**：穆勒矩阵（Mueller Matrix）完整描述了材料对偏振光的变换特性，是光弹性分析、透明材料检测、生物组织成像的核心工具。传统穆勒矩阵测量需要多次采集不同偏振态的图像，耗时长，无法处理动态场景。

**现有痛点**：(1) 传统帧相机方案需要多帧合成，无法实时。(2) 高速相机方案受限于低动态范围和同步时序。(3) 现有偏振相机只能测量斯托克斯参数（4 参数），无法测完整穆勒矩阵（16 参数）。

**核心矛盾**：完整穆勒矩阵需要不同偏振配置下的多次测量，但动态场景中连续测量之间场景已经变化。

**本文目标** 利用事件相机的异步高速特性实现单次扫描的动态穆勒矩阵视频成像。

**切入角度**：事件相机记录的是亮度变化的时间戳（而非绝对强度）。快速旋转 QWP 产生周期性亮度变化，事件的时间差可以直接映射到穆勒矩阵元素的比值——不需要绝对强度，天然具有 HDR 能力。

**核心 idea**：将事件相机的时间差信号（而非强度）映射到穆勒矩阵比值，通过旋转 QWP 的调制实现 30fps 动态穆勒矩阵成像。

## 方法详解

### 整体框架
硬件：事件相机(EVK4) + LED 光源，各配一个快速旋转 QWP + 固定线偏振片。相机 QWP 转速 5× 光源 QWP → 事件在像素级记录亮度变化时间戳 → 椭偏图像形成模型将 ∂log(I)/∂t 关联到穆勒矩阵 → SVD 逐像素估计 + Cloude 滤波物理约束 + 时空传播精炼。

### 关键设计

1. **椭偏-事件图像形成模型**:

    - 功能：将事件时间差映射到穆勒矩阵
    - 核心思路：旋转 QWP 产生的光强 I(t) 是穆勒矩阵元素和旋转角度的函数。事件记录 ∂log(I)/∂t，可以表达为系统矩阵 $B_{tk}$ 与归一化穆勒矩阵的线性组合。多次事件提供超定方程组
    - 设计动机：用时间差而非绝对强度使系统对环境光变化和传感器增益不敏感——天然 HDR

2. **SVD 估计 + Cloude 滤波**:

    - 功能：从噪声事件数据恢复物理有效的穆勒矩阵
    - 核心思路：加权最小二乘通过 SVD 求解穆勒矩阵的初始估计。Cloude 滤波将估计投影到物理可实现的穆勒矩阵空间（满足 Stokes transformation 约束）。迭代重加权（5 轮）抑制异常值
    - 设计动机：原始 SVD 估计 MSE 0.11，加 Cloude 和重加权后降到 0.04

3. **时空传播精炼**:

    - 功能：填补事件稀疏区域并抑制噪声
    - 核心思路：PatchMatch 风格的传播——用周围像素和前一帧的穆勒矩阵"提议"初始值，随机扰动搜索更优解（10 轮迭代）
    - 设计动机：事件相机在无亮度变化区域不产生事件，传播机制用时空邻域填补这些"空洞"

### 损失函数 / 训练策略
无学习方法——纯物理模型+优化。QWP 旋转周期 ~33ms → 30fps 帧率。500×500 分辨率处理 30 帧需 ~130 秒。

## 实验关键数据

### 主实验

| 评估 | MSE |
|------|-----|
| SVD only | 0.11 |
| + Cloude 滤波 | 0.07 |
| + 重加权 | 0.05 |
| **+ 时空传播** | **0.04** |
| 已知样品验证 | 0.045 |

### 应用演示

| 应用 | 效果 |
|------|------|
| 光弹性（明胶盘受力） | 实时显示应力分布变化 |
| 透明胶带检测 | 区分有/无胶带区域 |
| 动态人脸/头发 | 捕捉毛发偏振特征 |
| HDR 穆勒矩阵 | 高亮和暗处同时准确 |

### 关键发现
- **首次实现视频级穆勒矩阵成像**：30fps 处理动态场景
- **天然 HDR**：基于时间差而非强度，对光照变化免疫
- **精度与传统方法可比**：已知偏振器件验证 MSE 0.045

## 亮点与洞察
- **"事件时间差→穆勒矩阵比值"的映射**是核心物理洞察——巧妙利用了事件相机记录变化而非绝对值的特性
- **硬件设计简洁**：仅在标准事件相机前加旋转 QWP，无需特殊传感器
- **开辟事件相机×偏振成像的新交叉领域**

## 局限与展望
- 处理速度（130s/30帧@500×500）尚未实时——GPU 加速可能改善
- 旋转 QWP 的机械精度限制了测量精度
- 仅演示了定性应用，定量材料表征需要更多标定

## 相关工作与启发
- **vs DoFP 偏振相机**：只能测 4 个 Stokes 参数。事件椭偏仪测完整 16 参数穆勒矩阵
- **vs 传统穆勒矩阵仪**：需要多次静态采集。事件方案实现单次扫描成像

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首创事件相机穆勒矩阵成像，物理洞察深刻
- 实验充分度: ⭐⭐⭐⭐ 仿真+真实验证、已知样品标定、多应用演示
- 写作质量: ⭐⭐⭐⭐ 物理模型推导清楚
- 价值: ⭐⭐⭐⭐ 对计算成像和材料检测开辟新方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Full-DoF Egomotion Estimation for Event Cameras Using Geometric Solvers](full-dof_egomotion_estimation_for_event_cameras_using_geometric_solvers.md)
- [\[CVPR 2025\] EBS-EKF: Accurate and High Frequency Event-based Star Tracking](ebs-ekf_accurate_and_high_frequency_event-based_star_tracking.md)
- [\[NeurIPS 2025\] Deep Continuous-Time State-Space Models for Marked Event Sequences](../../NeurIPS2025/others/deep_continuous-time_state-space_models_for_marked_event_sequences.md)
- [\[ACL 2025\] SEOE: A Scalable and Reliable Semantic Evaluation Framework for Open Domain Event Detection](../../ACL2025/others/seoe_semantic_eval.md)
- [\[AAAI 2026\] I2E: Real-Time Image-to-Event Conversion for High-Performance Spiking Neural Networks](../../AAAI2026/others/i2e_real-time_image-to-event_conversion_for_high-performance_spiking_neural_netw.md)

</div>

<!-- RELATED:END -->
