---
title: >-
  [论文解读] DPoser-X: Diffusion Model as Robust 3D Whole-Body Human Pose Prior
description: >-
  [ICCV 2025][图像生成][扩散模型] 提出 DPoser-X，基于无条件扩散模型的 3D 全身人体姿态先验，将各种姿态相关任务统一为逆问题，通过变分扩散采样的截断时间步调度进行测试时优化，并引入混合训练策略有效结合全身和部位数据集，在身体、手、脸和全身建模的 8 个基准上取得最高 61% 的提升。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "扩散模型"
  - "人体姿态先验"
  - "全身建模"
  - "逆问题"
  - "变分扩散采样"
---

# DPoser-X: Diffusion Model as Robust 3D Whole-Body Human Pose Prior

**会议**: ICCV 2025  
**arXiv**: [2508.00599](https://arxiv.org/abs/2508.00599)  
**代码**: [https://dposer.github.io/](https://dposer.github.io/)  
**领域**: 图像生成  
**关键词**: 扩散模型, 人体姿态先验, 全身建模, 逆问题, 变分扩散采样

## 一句话总结

提出 DPoser-X，基于无条件扩散模型的 3D 全身人体姿态先验，将各种姿态相关任务统一为逆问题，通过变分扩散采样的截断时间步调度进行测试时优化，并引入混合训练策略有效结合全身和部位数据集，在身体、手、脸和全身建模的 8 个基准上取得最高 61% 的提升。

## 研究背景与动机

人体姿态先验建模是 3D 人体建模的基础课题，其目标是从大规模数据中学习合理的姿态分布，作为下游任务（如人体网格恢复、运动捕捉、姿态补全）的正则化。

**现有方法的不足**：
- **GMM**（如 SMPLify）：无界性导致可能生成不合理姿态
- **VAE**（如 VPoser）：高斯先验限制了潜在空间的表达力，生成多样性不足
- **NDF**（如 Pose-NDF、NRDF）：难以泛化到高维人体姿态流形的完整空间

**核心矛盾**：上述方法主要关注身体姿态，忽略了包含手部和面部表情的全身建模。全身姿态数据稀缺（现有数据集主要覆盖特定动作如抓取、手语），且身体各部分之间存在复杂的相互依赖关系（如站立时双手姿态通常对称）。

**切入角度**：利用扩散模型在复杂分布学习上的优势，训练无条件扩散模型学习姿态分布，将下游任务统一为逆问题框架，通过变分扩散采样在测试时求解。关键创新包括：面向姿态数据的截断时间步调度和结合全身/部位数据集的混合训练策略。

## 方法详解

### 整体框架

DPoser-X 由三个层次组成：(1) 部位级 DPoser（body/hand/face 各一个无条件扩散模型）；(2) 融合模块将三个部位模型的最后层特征通过全连接网络融合，捕获部位间关联；(3) 混合训练策略结合全身和部位数据集。

### 关键设计

1. **DPoser 正则化**：核心思想是用扩散模型的单步去噪作为姿态先验的正则化项。对当前优化变量 $\mathbf{x}_0$（即 SMPL 姿态参数 $\theta$），加噪到时间步 $t$ 得到 $\mathbf{x}_t$，然后用训练好的噪声预测器 $\epsilon_\phi$ 进行单步去噪得到 $\hat{\mathbf{x}}_0(t)$，正则化损失为：

    $L_{\text{DPoser}} = w_t \|\mathbf{x}_0 - \text{sg}[\hat{\mathbf{x}}_0(t)]\|_2^2$

   其中 $\hat{\mathbf{x}}_0(t) = \frac{\mathbf{x}_t - \sigma_t \epsilon_\phi(\mathbf{x}_t; t)}{\alpha_t}$，$\text{sg}$ 表示停止梯度。该损失的梯度方向与变分扩散采样（Eq. 4）中的正则化项一致（$\propto \epsilon_\phi(\mathbf{x}_t; t) - \epsilon$），但形式更直观且自然等价于 Score Distillation Sampling。

   **设计动机**：停止梯度确保不需要通过训练好的扩散网络反向传播，只需一次前向传播，计算开销极小（相比无先验基线仅增加 10%）。

2. **截断时间步调度**：传统图像扩散优化使用均匀时间步调度 $[1.0, 0.0]$，但论文发现姿态数据与图像不同——**关键姿态信息集中在小 $t$ 阶段**（$t \leq 0.3$）。实验验证（Fig. 3）：用 DDIM 采样器在有限步数下，将步数集中在后期（小 $t$）比均匀分配产生更好的姿态。

   截断调度公式：$t = t_{\max} - \frac{(t_{\max} - t_{\min}) \times \text{iter}}{N-1}$。典型区间：人体网格恢复 $[0.12, 0.08]$，运动去噪 $[0.2, 0.05]$，姿态补全 $[0.15, 0.05]$。

   **直觉解释**：小 $t$ 时加噪和去噪路径短，$\hat{\mathbf{x}}_0(t)$ 接近 $\mathbf{x}_0$，DPoser 引导弱但精准；大 $t$ 时引导强但可能导致去噪后的姿态与原始关联性降低。根据任务噪声水平选择合适范围是关键。

3. **混合训练策略（DPoser-X-mixed）**：解决全身姿态数据稀缺问题。

    - 将部位数据（body-only/hand-only/face-only）视为不完整的全身数据，仅对可用部分计算损失
    - 对全身数据以 20% 概率随机遮盖某些部分，强制模型预测被遮盖部分（防止全身与部位数据分布偏差过大）
    - 数据混合比例：约 65% 全身 + 14% 身体 + 12% 单手 + 4% 双手 + 5% 面部
   
   该策略使模型既能学习部位间关联（如全身数据中的双手协调），又能通过部位数据增强泛化能力。

### 损失函数 / 训练策略

DPoser 使用 sub-VP SDE 参数化训练无条件扩散模型，噪声预测目标加权为 $w(t) = \sigma_t^2$。身体模型基于 AMASS 数据集（约 5500 万姿态），使用轴角表示（零均值、单位方差归一化），约 8.28M 参数的全连接网络，Adam 优化器训练 80 万轮。

## 实验关键数据

### 主实验

**人体网格恢复（EHF 数据集，PA-MPJPE mm）**：

| 初始化 | 无先验 | GMM | VPoser | Pose-NDF | NRDF | GAN-S | **DPoser** |
|--------|-------|-----|--------|----------|------|-------|-----------|
| 从头 | 108.57 | 58.32 | 58.08 | 57.87 | 57.38 | 57.26 | **56.05** |
| CLIFF | 56.62 | 51.02 | 49.39 | 49.50 | 49.27 | 49.58 | **49.05** |

**全身姿态补全（遮盖一只手，min/mean MPVPE mm）**：

| 方法 | ARCTIC | BEAT2 |
|------|--------|-------|
| VPoser-X | 37.34/43.24 | 27.49/35.46 |
| **DPoser-X** | **21.81/30.99** | **15.92/25.89** |

**全身网格恢复（ARCTIC 数据集，PA-MPVPE mm）**：

| 方法 | All | Hands | Face | Body |
|------|-----|-------|------|------|
| VPoser-X | 66.74 | 17.44 | 10.99 | 79.88 |
| **DPoser-X** | **60.98** | **15.60** | **9.75** | **73.00** |

### 消融实验

**时间步调度策略对比**：

| 调度策略 | 全身网格恢复 (All/Hands) | 运动去噪 (MPVPE/MPJPE) |
|---------|--------------------------|----------------------|
| Random | 62.28 / 16.63 | 43.33 / 23.87 |
| Fixed | 61.69 / 15.71 | 45.69 / 22.54 |
| Uniform | 62.13 / 17.32 | 39.72 / 20.80 |
| **Truncated** | **60.98 / 15.60** | **38.21 / 19.87** |

截断调度在所有任务上均优于现有策略。Uniform 在网格恢复（低噪声）上表现差，Fixed 在运动去噪（渐变噪声）上表现差。

**混合训练策略对比**：

| 模型 | ARCTIC 补全 (min MPVPE) | Fit3D 恢复 (All PA-MPVPE) |
|------|------------------------|--------------------------|
| DPoser-X-base | 25.49 | 72.79 |
| DPoser-X-fused | 21.51 | 72.06 |
| **DPoser-X-mixed** | 21.81 | **70.91** |

Mixed 策略在零样本泛化（Fit3D 运动场景）上显著优于 fused，补全精度与 fused 相当。

### 关键发现

- **手部逆运动学**：DPoser-hand 在 ReInterHand 稀疏设置下比次优方法降低 50%+ MPJPE（3.21 vs 8.25 mm）
- **运动去噪**：DPoser 超越专门的运动先验 HuMoR（19.87 vs 22.69 MPJPE），尽管 DPoser 不是为时序任务设计的
- **面部重建**：在 NOW 基准上与 MICA 初始化结合时达到 8.76mm 均值误差（SOTA）
- **计算开销极低**：DPoser 正则化仅增加约 10% 优化时间

## 亮点与洞察

- **统一逆问题框架**：将姿态补全、逆运动学、人体网格恢复等多种任务统一为逆问题，DPoser 作为通用正则化项即插即用
- **截断时间步调度**是面向姿态数据的重要发现——姿态信息集中在低噪声区间，与图像的"先生成结构后填细节"恰好相反
- 混合训练策略设计优雅——将部位数据视为缺失值问题，20% 随机遮盖全身数据作为数据增强
- 停止梯度设计确保了与任何下游优化器的兼容性，不增加记忆负担

## 局限与展望

- 基于 SMPL-X 的旋转角度表示，表达能力受限于骨骼模型的关节自由度
- 截断时间步区间 $[t_{\max}, t_{\min}]$ 需要针对每个任务手动选择
- 混合训练的各数据源权重平衡目前靠经验确定
- 未探索与条件扩散模型（如以图像为条件）的结合
- 全身生成质量（Table 6）因训练数据有限仍有提升空间

## 相关工作与启发

- **VPoser**（VAE 先验）是最广泛使用的基线，DPoser 在表达力上全面超越
- **BUDDI**（人体交互先验）使用了类似的 SDS 优化思路，但 DPoser 更通用且引入了截断调度
- **Score Distillation Sampling** 在 3D 生成中广泛使用，DPoser 将其推广到姿态域并给出等价的直观形式
- 混合训练策略可推广到其他数据稀缺场景（如面部 + 身体 + 手部的多源学习）

## 评分

- 新颖性：⭐⭐⭐⭐ — 扩散模型作为姿态先验 + 截断调度是新颖贡献
- 理论深度：⭐⭐⭐⭐ — 变分扩散采样的推导严谨，DPoser 损失与 SDS 的等价性证明教科书级
- 实验充分度：⭐⭐⭐⭐⭐ — 8 个基准、身体/手/脸/全身全覆盖、充分消融
- 实用性：⭐⭐⭐⭐⭐ — 即插即用正则化项，计算开销极低

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] InterMimic: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions](../../CVPR2025/image_generation/intermimic_towards_universal_whole-body_control_for_physics-based_human-object_i.md)
- [\[ICCV 2025\] Learning Deblurring Texture Prior from Unpaired Data with Diffusion Model](learning_deblurring_texture_prior_from_unpaired_data_with_diffusion_model.md)
- [\[CVPR 2025\] Visual Persona: Foundation Model for Full-Body Human Customization](../../CVPR2025/image_generation/visual_persona_foundation_model_for_full-body_human_customization.md)
- [\[ICCV 2025\] DIIP: Diffusion Image Prior](diffusion_image_prior.md)
- [\[CVPR 2026\] PoseD-Flow: Versatile and Guided Flow Matching Model of Human Pose](../../CVPR2026/image_generation/posed-flow_versatile_and_guided_flow_matching_model_of_human_pose.md)

</div>

<!-- RELATED:END -->
