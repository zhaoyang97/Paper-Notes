---
title: >-
  [论文解读] Domain Reduction Strategy for Non-Line-of-Sight Imaging
description: >-
  [ECCV 2024][非视线成像] 提出一种面向非视线成像（NLOS）的优化方法，通过将瞬态信号建模为逐点光传播函数的叠加，并设计由粗到细的域缩减策略剪除空白区域，在通用NLOS场景下实现约20倍加速且同时重建反射率和表面法线。 非视线成像（NLOS）旨在通过中继墙的间接反射来重建隐藏场景，在自动驾驶、医疗和救援中有重要应…
tags:
  - "ECCV 2024"
  - "非视线成像"
  - "NLOS"
  - "域缩减"
  - "优化重建"
  - "表面法线"
---

# Domain Reduction Strategy for Non-Line-of-Sight Imaging

**会议**: ECCV 2024  
**arXiv**: [2308.10269](https://arxiv.org/abs/2308.10269)  
**代码**: [GitHub](https://github.com/hyunbo9/domain-reduction-strategy)  
**领域**: 人体理解 / 计算成像  
**关键词**: 非视线成像, NLOS, 域缩减, 优化重建, 表面法线

## 一句话总结

提出一种面向非视线成像（NLOS）的优化方法，通过将瞬态信号建模为逐点光传播函数的叠加，并设计由粗到细的域缩减策略剪除空白区域，在通用NLOS场景下实现约20倍加速且同时重建反射率和表面法线。

## 研究背景与动机

非视线成像（NLOS）旨在通过中继墙的间接反射来重建隐藏场景，在自动驾驶、医疗和救援中有重要应用。现有方法分为两大类：

**FFT逆解方法**（LCT、FK、Phasor）：速度快但需要严格假设（平面中继墙、共焦扫描、密集光栅扫描），且多数只能重建反射率，无法获得表面几何

**优化方法**（NeTF等）：通用性好，不受扫描系统限制，但计算代价巨大（每次迭代 $O(N^5)$ 或 $O(N^6)$），大量计算浪费在空白区域

关键观察：NLOS场景中目标物体的可见表面极其稀疏，**仅占隐藏空间不到5%**。这意味着绝大部分计算都花在了空白区域，是巨大的浪费。

## 方法详解

### 整体框架

方法的核心思路是：
1. 将瞬态测量建模为来自隐藏空间各点的光传播函数的线性叠加
2. 在优化过程中周期性地剪除空白区域（域缩减），只保留活跃区域
3. 采用由粗到细策略逐步提升分辨率

输入为时间分辨传感器测量的瞬态信号 $\tau(t, \mathbf{l}, \mathbf{s})$，输出为隐藏场景的反射率 $\rho$ 和表面法线 $\vec{n}$。

### 关键设计

#### 1. 逐点光传播模型

将瞬态信号分解为每个隐藏点 $\mathbf{p}$ 的独立贡献：

$$\tau = \int_{\mathbf{p} \in \Omega} \rho(\mathbf{p}) \cdot g_{\mathbf{p}} \, d\mathbf{p}$$

每点的光传播函数 $g_{\mathbf{p}}$ 包含三个因子：
- **余弦项 $\Phi_{\mathbf{p}}$**：朗伯余弦定律的视角依赖项，利用表面法线建模
- **距离衰减项 $\Upsilon_{\mathbf{p}}$**：$1/d_l^2 \cdot 1/d_s^2$，激光到点和点到传感器的距离平方反比
- **时间约束 $\delta$**：光程等于测量时间×光速

关键优势：此模型不对中继墙几何和扫描系统做任何假设，天然支持非平面墙和非共焦配置。

#### 2. 基于随机采样的重建方案

- 将隐藏空间划分为网格，每个顶点分配4维变量（反射率 $\rho$ + 法线 $\vec{n}$）
- 每步从网格随机采样一组点（每个cell一个点），通过三线性插值获得 $\rho$ 和 $\vec{n}$
- 计算各点光传播函数并线性叠加得到预测瞬态信号
- 通过最小化 L2 距离优化变量

使用无矩阵的 CUDA kernel 实现，所有采样点的光传播计算可GPU并行。

#### 3. 域缩减策略 (Domain Reduction Strategy)

核心创新，分三个层次：

**基础域缩减**：周期性（每50步）检查反射率，将 $\rho < \epsilon$（最大值的5%）的区域从采样域中剔除，只在活跃区域采样

**软域缩减**：为防止误删非空区域，先对反射率体素应用高斯核低通滤波，再阈值裁剪——相当于将域扩展到周围区域，给误删区域二次机会

**由粗到细策略**：
- 初始使用粗网格（计算快）
- 域缩减到足够小后，将活跃区域的网格细分为更高分辨率
- 新网格变量通过三线性插值初始化
- 最终可在单GPU上重建高分辨率（128×128）输出

#### 4. 噪声正则化

- L1正则化鼓励反射率稀疏
- 联合优化可学习噪声参数 $d(t, \mathbf{l}, \mathbf{s}) = b + \lambda\sigma(z)$，覆盖环境光和暗计数的影响
- $b$ 控制最小噪声级别，$\lambda$ 控制最大值

### 损失函数 / 训练策略

优化目标为预测瞬态与真实瞬态的L2距离 + L1反射率正则化。使用Adam优化器，学习率1，共1000步。域缩减每50步执行一次。

## 实验关键数据

### 主实验（稀疏扫描 32×32 共焦设置）

| 方法 | 深度MAE(5%)↓ | 深度RMSE(5%)↓ | 法线MAE(5%)↓ | 法线RMSE(5%)↓ |
|------|-------------|--------------|-------------|--------------|
| LCT | 0.1977 | 0.2875 | - | - |
| FK | 0.0719 | 0.1871 | - | - |
| Phasor(BP) | 0.1348 | 0.2049 | - | - |
| NeTF | 0.0679 | 0.1748 | - | - |
| DLCT | 0.3189 | 0.4220 | 0.3796 | 0.4856 |
| **Ours** | **0.0477** | **0.1523** | **0.1147** | **0.2394** |

定性结果：在合成（ZNLOS Bunny/Serapis）和真实（Stanford Statue/Dragon）数据上，本方法重建结果最清晰锐利，细节最丰富。DLCT在32×32稀疏扫描下出现明显伪影，FK丢失细节，NeTF结果模糊变形。

### 消融实验

| 域缩减 | 粗到细 | 100步时间 | 500步时间 | 1000步时间 | 剩余域比例 |
|--------|--------|-----------|-----------|------------|-----------|
| ✗ | ✗ | 109s | 539s | 1087s | 100% |
| ✓ | ✗ | 67s | 103s | 134s | 4% |
| ✓ | ✓ | 4s | 27s | **54s** | 3% |

GPU显存对比（N=128分辨率）：LCT 7101MB, FK 6813MB, 本方法(无DR) 4065MB, **本方法(有DR) 1659MB**

### 关键发现

- 域缩减+粗到细策略实现约**20倍加速**（1087s → 54s）
- 仅32×32稀疏扫描即可达到高分辨率重建效果，无需密集光栅扫描
- 增加到64×64扫描分辨率后质量无明显提升，说明32×32已足够
- 噪声正则化（L1+可学习噪声）对真实数据的鲁棒性至关重要
- 可在**1分钟内**完成128×128输出的重建
- 域缩减不牺牲重建质量——有无域缩减的重建结果视觉上无差异

## 亮点与洞察

1. **逐点分解的灵活性**：将瞬态信号分解为逐点贡献后，域缩减变得trivial——只需从活跃区域采样即可
2. **稀疏性的深刻利用**：NLOS场景中<5%的区域有物体，这个"诅咒"反而成为算法加速的关键优势
3. **兼顾通用性与效率**：既不需要FFT方法的严格假设，又通过域缩减弥补了优化方法的效率劣势
4. **同时重建反射率和法线**：通过精确建模余弦项，无需额外代价即可获得表面几何

## 局限与展望

- 假设隐藏场景遵循朗伯反射模型，忽略互反射和自遮挡
- 域缩减阈值（最大反射率的5%）为手动设定，可探索自适应阈值
- 在非共焦设置下阈值需调低（3%），参数敏感性可进一步研究
- 可结合神经场表示（如NeRF）进一步提升细节重建能力

## 相关工作与启发

- **LCT/DLCT/FK**：FFT逆解方法，快但受限于扫描假设，DLCT是唯一能重建法线的FFT方法
- **NeTF**：将NeRF引入NLOS，通用但效率低且结果模糊
- **Phasor Field**：波动光学视角的NLOS方法，需平面墙和共焦扫描
- 本文方法思路类似于3D重建中的coarse-to-fine策略和八叉树加速

## 评分

- 新颖性: ⭐⭐⭐⭐ — 逐点分解+域缩减的组合在NLOS中是新的
- 技术深度: ⭐⭐⭐⭐⭐ — CUDA实现、光传播模型、噪声正则化都很扎实
- 实验充分度: ⭐⭐⭐⭐ — 合成+真实数据、多种扫描配置、详细消融
- 写作质量: ⭐⭐⭐⭐ — 方法描述清晰，算法伪代码易理解

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Synergy of Sight and Semantics: Visual Intention Understanding with CLIP](synergy_of_sight_and_semantics_visual_intention_understanding_with_clip.md)
- [\[ECCV 2024\] Non-parametric Sensor Noise Modeling and Synthesis](non-parametric_sensor_noise_modeling_and_synthesis.md)
- [\[ACL 2025\] Battling against Tough Resister: Strategy Planning with Adversarial Game for Non-collaborative Dialogues](../../ACL2025/others/battling_against_tough_resister_strategy_planning_with_adversarial_game_for_non-.md)
- [\[ACL 2025\] Verbosity-Aware Rationale Reduction: Effective Reduction of Redundant Rationale](../../ACL2025/others/verbosity-aware_rationale_reduction_effective_reduction_of_redundant_rationale_v.md)
- [\[NeurIPS 2025\] Frequency-Aware Token Reduction for Efficient Vision Transformer](../../NeurIPS2025/others/frequency-aware_token_reduction_for_efficient_vision_transformer.md)

</div>

<!-- RELATED:END -->
