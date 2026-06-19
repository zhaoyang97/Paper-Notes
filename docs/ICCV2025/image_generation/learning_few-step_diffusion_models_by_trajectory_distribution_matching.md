---
title: >-
  [论文解读] Learning Few-Step Diffusion Models by Trajectory Distribution Matching
description: >-
  [ICCV 2025][图像生成][扩散蒸馏] 提出 Trajectory Distribution Matching（TDM），一种统一轨迹蒸馏和分布匹配的新范式，在分布层面对齐学生与教师的 ODE 轨迹，实现高效的少步扩散模型蒸馏，仅需 2 A800 小时即可将 PixArt-α 蒸馏为超越教师的 4 步生成器。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "扩散蒸馏"
  - "少步生成"
  - "轨迹分布匹配"
  - "Score Distillation"
  - "文生图加速"
---

# Learning Few-Step Diffusion Models by Trajectory Distribution Matching

**会议**: ICCV 2025  
**arXiv**: [2503.06674](https://arxiv.org/abs/2503.06674)  
**代码**: [项目页面](https://tdm-t2x.github.io/)  
**领域**: 扩散模型/图像生成  
**关键词**: 扩散蒸馏, 少步生成, 轨迹分布匹配, Score Distillation, 文生图加速

## 一句话总结

提出 Trajectory Distribution Matching（TDM），一种统一轨迹蒸馏和分布匹配的新范式，在分布层面对齐学生与教师的 ODE 轨迹，实现高效的少步扩散模型蒸馏，仅需 2 A800 小时即可将 PixArt-α 蒸馏为超越教师的 4 步生成器。

## 研究背景与动机

加速扩散模型采样是 AIGC 高效部署的关键。现有蒸馏方法主要分为两大类，各有明显局限：

**分布匹配方法**（如 DMD、SiD）：通过 score distillation 实现分布级匹配，在一步生成上表现强劲，但主要为一步生成优化，**缺乏多步采样的灵活性**——增加采样步数时无法有效利用额外信息

**轨迹蒸馏方法**（如 Progressive Distillation、Consistency Models）：在实例级模拟教师 ODE 轨迹，支持多步采样，但**匹配整个实例级轨迹对模型容量要求极高**，且求解教师 ODE 的数值误差会传播到学生

核心矛盾：分布匹配忽视中间轨迹信息，轨迹蒸馏经受实例级匹配困难。

TDM 的关键洞察：**在分布层面匹配轨迹**，而非在实例层面。这非平凡地统一了两种范式的优势——既利用轨迹信息实现细粒度知识传递，又通过分布级匹配降低学习难度。

## 方法详解

### 整体框架

TDM 将 K 步学生模型参数化为离散 ODE 采样器，每步生成轨迹上的中间样本 $x_{t_i}$。然后通过分布级 score distillation 对齐学生轨迹上每个时间步的边际分布 $p_{\theta, t_i}$ 与教师的对应分布 $p_{\phi, t_i}$。整个过程无需数据（data-free），也无需求解教师 ODE。

### 关键设计

1. **轨迹分布匹配目标（TDM Objective）**:

    - 功能：在学生轨迹的每个时间步上，最小化与教师分布的反向 KL 散度
    - 核心思路：目标函数为：
    $L(\theta) = \sum_{i=0}^{K-1} \sum_{\tau=t_i}^{t_{i+1}} \lambda_\tau \text{KL}(p_{\theta, \tau|t_i}(x_\tau) \| p_{\phi, \tau}(x_\tau))$
      其中 $p_{\theta, \tau|t_i}$ 是学生轨迹样本的扩散分布。梯度计算为：
    $\nabla_\theta L(\theta) \approx \sum_{i,\tau} \lambda_\tau [s_\psi(x_\tau, \tau) - s_\phi(x_\tau, \tau)] \frac{\partial x_{t_i}}{\partial \theta}$
      需要一个 fake score $s_\psi$ 来近似学生分布的 score。
    - 设计动机：(1) 只需从学生采样，无需教师 ODE 采样（data-free 且高效）；(2) 避免了教师 ODE 求解的数值误差；(3) 分布级匹配比实例级匹配对模型容量要求更低。关键设计是确保不同时间步的扩散区间**不重叠**，使同一个 fake score 模型可以自然地区分不同分布。

2. **采样步感知目标（Sampling-Steps-Aware Objective）**:

    - 功能：使单一模型支持灵活的确定性多步采样
    - 核心思路：将目标扩展为对采样步数 K 的期望：
    $\mathbb{E}_K \sum_{i=0}^{K-1} \sum_{\tau=t_i^K}^{t_{i+1}^K} \lambda_\tau \text{KL}(p_{\theta, \tau|t_i^K}(x_\tau|K) \| p_{\phi,\tau}(x_\tau))$
      其中 K 作为条件注入学生和 fake score 模型。这得到的模型称为 TDM-unify。
    - 设计动机：现有确定性采样蒸馏方法的学习目标绑定到固定步数，训练 K 步模型后无法灵活切换到 M<K 步。共享 fake score 会引入理论偏差（论文中有严格推导），因此需要步感知的 fake score。

3. **Pseudo-Huber 代理训练目标**:

    - 功能：用 Pseudo-Huber 度量替代 L2 损失来稳定训练
    - 核心思路：学习目标变为：
    $L(\theta) = \sum_{i,\tau} \sqrt{\|x_{t_i} - \text{sg}(\tilde{x}_{t_i})\|_2^2 + c^2} - c$
      其中 $c = 0.00054\sqrt{d}$。这对梯度进行了归一化，产生更稳定的训练。
    - 设计动机：观察到 TDM 与 Consistency Models 在学习方式上有共通性（都最小化生成样本与修正样本的距离），因此借鉴 iCT 中 Pseudo-Huber 度量的成功经验。

### 损失函数 / 训练策略

- **Fake Score 训练**：使用带重要性采样的去噪 score matching，在学生轨迹样本附近高效学习 score
- **Better Teacher Better Student**：对于 SD-v1.5，建议先在高质量数据集上微调教师再蒸馏
- 反向传播时仅考虑一步 ODE 以节省 GPU 显存
- 训练效率极高：SDXL 仅需 2 A800 天，PixArt-α 仅需 500 迭代 / 2 A800 小时

## 实验关键数据

### 主实验

文生图生成质量对比（SDXL backbone, 4步）：

| 方法 | HPS↑ | AeS↑ | CLIP↑ | 训练成本 |
|------|------|------|-------|---------|
| SDXL-Lightning | 32.71 | 6.23 | 34.62 | — |
| Hyper-SD | 34.14 | 6.18 | 34.27 | — |
| DMD2 | 31.46 | 5.88 | 35.51 | 160 A100天 |
| LCM | 29.41 | 5.84 | 34.84 | 32 A100天 |
| **TDM (Ours)** | **34.88** | **6.28** | **36.08** | **2 A800天** |

PixArt-α backbone（4步, 1024分辨率）：

| 方法 | HPS↑ | AeS↑ | CLIP↑ | Data-Free? |
|------|------|------|-------|-----------|
| PixArt-α 教师 (25步) | 32.21 | 6.23 | 34.11 | — |
| LCM (4步) | 30.55 | 6.17 | 33.49 | ✗ |
| **TDM (4步)** | **33.21** | **6.42** | **33.66** | **✓** |

SD-v1.5 backbone（TDM-unify, 1步 & 4步）：

| 方法 | 步数 | HPS Avg↑ | AeS↑ | CLIP↑ |
|------|------|---------|------|-------|
| Hyper-SD | 1 | 28.01 | 5.64 | 30.87 |
| TDM-unify-SFT | 1 | **28.90** | **6.02** | **32.12** |
| DMD2 | 4 | 29.49 | 5.91 | 31.53 |
| TDM-unify-SFT | 4 | **31.31** | **6.08** | **32.77** |

### 消融实验

| 配置 | HPS Avg↑ | AeS↑ | 说明 |
|------|---------|------|------|
| TDM w/o trajectory (K=1) | 28.54 | 5.97 | 退化为纯分布匹配 |
| TDM w/ trajectory (K=4) | 30.83 | 6.07 | 轨迹信息带来显著提升 |
| TDM + Pseudo-Huber | 31.31 | 6.08 | Huber度量进一步改善 |
| 共享 fake score (不感知步数) | 下降 | 下降 | 验证步感知目标的必要性 |

LoRA 适配到未见过的定制模型（Realistic, SD-v1.5, 4步）：

| 方法 | HPS Avg↑ | FID↓ |
|------|---------|------|
| LCM | 27.72 | 26.89 |
| Hyper-SD | 30.36 | 37.83 |
| **TDM** | **31.22** | **20.23** |

### 关键发现

- TDM 在 SDXL 上以 HPS +2.17 超越 SDXL-Lightning，+3.42 超越 DMD2，**同时训练成本不到 DMD2 的 1/80**
- 4 步 TDM 蒸馏的 PixArt-α **超越了 25 步教师模型**在 HPS 和 AeS 上的表现
- TDM-unify 实现了**单一模型支持 1 步和 4 步**的灵活采样
- 方法可扩展到视频扩散蒸馏：将 CogVideoX-2B 蒸馏为 4 步生成器，VBench 总分从 80.91 提升到 81.65

## 亮点与洞察

- **统一范式**：首次非平凡地统一了轨迹蒸馏和分布匹配，理论上证明了两者的连接
- **极致训练效率**：PixArt-α 仅需 500 迭代（教师训练成本的 0.01%），SDXL 仅需 2 A800 天
- **全面 data-free**：完全不需要真实图像数据，降低了数据获取和版权问题
- **确定性+灵活采样**：TDM-unify 是首个同时支持确定性采样和灵活步数调整的蒸馏方法
- **与 Consistency Models 的理论连接**：揭示了 TDM 和 CM 在学习目标上的深层相似性

## 局限与展望

- Fake score 的学习质量直接影响蒸馏效果，需要仔细调节训练策略
- 性能上限受限于教师模型质量（data-free 的双刃剑），"Better Teacher, Better Student" 策略可缓解但增加了前期投入
- 反向传播时仅考虑一步 ODE 是显存限制下的妥协，多步反传可能带来更好效果
- 步感知 fake score 增加了训练复杂度

## 相关工作与启发

- 与 DMD2 密切相关但本质不同：DMD2 每步都预测干净样本忽视中间轨迹，TDM 显式模拟确定性轨迹
- Consistency Models 的 Pseudo-Huber 度量成功经验被巧妙迁移到 score distillation 框架
- 启示：轨迹信息是扩散模型的独特资产，蒸馏时不应丢弃而应在分布层面利用

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 统一两种蒸馏范式的理论贡献突出，理论推导严谨
- 实验充分度: ⭐⭐⭐⭐⭐ 多 backbone（SD-v1.5/SDXL/PixArt-α/CogVideoX）、多步数、多指标全面评测
- 写作质量: ⭐⭐⭐⭐⭐ 理论清晰，从轨迹蒸馏到分布匹配的统一逻辑连贯流畅
- 价值: ⭐⭐⭐⭐⭐ 效果 SOTA + 极致训练效率 + 理论新范式，实用性和影响力兼备

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Unsupervised Imaging Inverse Problems with Diffusion Distribution Matching](unsupervised_imaging_inverse_problems_with_diffusion_distribution_matching.md)
- [\[CVPR 2026\] Refining Few-Step Text-to-Multiview Diffusion via Reinforcement Learning](../../CVPR2026/image_generation/refining_few-step_text-to-multiview_diffusion_via_reinforcement_learning.md)
- [\[ICCV 2025\] Golden Noise for Diffusion Models: A Learning Framework](golden_noise_for_diffusion_models_a_learning_framework.md)
- [\[ICCV 2025\] Joint Diffusion Models in Continual Learning](joint_diffusion_models_in_continual_learning.md)
- [\[ICCV 2025\] DiffuMatch: Category-Agnostic Spectral Diffusion Priors for Robust Non-rigid Shape Matching](diffumatch_category-agnostic_spectral_diffusion_priors_for_robust_non-rigid_shap.md)

</div>

<!-- RELATED:END -->
