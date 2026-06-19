---
title: >-
  [论文解读] Learnable Sampler Distillation for Discrete Diffusion Models
description: >-
  [NeurIPS 2025][图像生成][离散扩散模型] 提出LSD和LSD+方法，通过蒸馏将高保真教师采样器的中间分数轨迹知识迁移给少步数学生采样器，以可学习的采样系数和非均匀时间调度实现离散扩散模型的高效高质量采样。 离散扩散模型（DDMs）在文本、分子等离散数据生成上展现了强大能力，但采样效率低下是其实用化的主要瓶颈—…
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "离散扩散模型"
  - "采样加速"
  - "蒸馏"
  - "可学习系数"
  - "时间调度"
---

# Learnable Sampler Distillation for Discrete Diffusion Models

**会议**: NeurIPS 2025  
**arXiv**: [2509.19962](https://arxiv.org/abs/2509.19962)  
**代码**: [GitHub](https://github.com/feiyangfu/LSD)  
**领域**: 扩散模型 / 离散扩散加速  
**关键词**: 离散扩散模型, 采样加速, 蒸馏, 可学习系数, 时间调度

## 一句话总结

提出LSD和LSD+方法，通过蒸馏将高保真教师采样器的中间分数轨迹知识迁移给少步数学生采样器，以可学习的采样系数和非均匀时间调度实现离散扩散模型的高效高质量采样。

## 研究背景与动机

离散扩散模型（DDMs）在文本、分子等离散数据生成上展现了强大能力，但采样效率低下是其实用化的主要瓶颈——通常需要1024+步函数评估（NFEs）。

直接减少采样步数会严重放大两类误差：

**复合解码误差（Compounding Decoding Error）**：DDMs为计算效率采用因式化参数化，独立预测每个token的去噪状态，忽略了token间的固有依赖。步数减少使这种近似质量下降。

**离散化误差（Discretization Error）**：Euler或τ-leaping等数值方法在大步长下无法准确逼近逆向动态。

这两类误差沿采样轨迹累积，在低NFEs时严重降低生成质量。

现有加速方法面临的关键挑战：
- 精确模拟方法（如Gillespie）计算代价高且不可并行
- τ-leaping等近似方法仅一阶精确，需要小步长保证质量
- JYS方法仅优化"何时采样"（时间步选择），但在每个时间步仍使用标准的大步长近似
- 连续扩散的蒸馏方法（如S4S）依赖最终样本比较，无法直接应用于DDM，因为离散采样的不可微性阻断梯度流

## 方法详解

### 整体框架

LSD采用师生蒸馏框架：教师采样器用 $N$ 步（小步长、高保真），学生采样器仅用 $M \ll N$ 步。核心创新在于对齐**中间分数轨迹**而非最终输出，绕过离散采样的不可微问题。LSD+进一步学习非均匀时间调度。

### 关键设计

1. **可学习采样系数 (Learnable Coefficients)**

   对标准Euler采样器的更新规则引入可学习的时间相关系数 $\Phi(t_k)$：
    $p(x^i_{t_{k+1}}|x^i_{t_k}) = \delta_{x^i_{t_k}}(x^i_{t_{k+1}}) + \Delta t \cdot Q_{t_k}(x^i_{t_k}, x^i_{t_{k+1}}) \cdot (\Phi(t_k) s_\theta(x_{t_k}, t_k))_{i, x^i_{t_{k+1}}}$
   
   系数 $\Phi(t_k)$ 自适应调节每步混凝土分数的影响强度，补偿大步长带来的累积误差。通过最小化学生与教师在各时间步的分数差异来优化：
    $\mathcal{L}_k(\Phi(t_k)) = \mathbb{E}_{x_{t_0}\sim\pi} \left[ d(s^*_k, \Phi(t_k) s_k) \right]$
   
   动机：直接最小化最终输出距离 $d(x_\epsilon, x^*_\epsilon)$ 因离散采样不可微而不可行，而中间分数轨迹对齐提供了可微的优化路径。

2. **可学习非均匀时间调度 (LSD+)**

   在LSD基础上，额外学习非均匀步长 $\{\kappa_k\}_{k=1}^M$（初始化为 $\Delta t$），生成学习时间步 $\tau_k = T - \sum_{\ell=1}^k \kappa_\ell$。通过对齐学生和教师在逆过程中的**有效转移项**来更新步长：
    $\tilde{\mathcal{L}}_k(\kappa_k) = \mathbb{E}_{x_{t_0}\sim\pi} \left[ d\left(\kappa_k s_\theta(x_{\tau_k}, \tau_k), \frac{T-\epsilon}{N} s_\theta(x^*_{t_k}, t_k)\right) \right]$
   
   动机：逆向扩散动态在不同时间段变化显著，自适应分配步长能更好地捕跨这种变化，进一步减少累积误差。

3. **松弛目标 (Relaxed Objective)**

   允许学生采样器从扰动起始点 $\tilde{x}_{t_0}$（在原始 $x_{t_0}$ 的Hamming距离 $\zeta$ 内）开始匹配教师轨迹：
    $d_H(x_{t_0}, \tilde{x}_{t_0}) \leq \zeta$
   
   $\zeta$ 设为序列长度的约5%。松弛后的目标：
    $\mathcal{L}_{\text{relaxed},k}(\Phi(t_k)) = \mathbb{E}_{x_{t_0}, \tilde{x}_{t_0}} \left[ d(s_\theta(x^*_{t_k}, t_k), \Phi(t_k) s_\theta(\tilde{x}_{t_k}, t_k)) \right]$
   
   动机：容量受限的学生采样器难以严格匹配教师输出，松弛使优化更可行。推理时仍使用原始未扰动输入。

### 损失函数 / 训练策略

训练过程高效（RTX4090上约5分钟），学到的参数在推理时不增加额外计算开销：
- LSD：初始化 $\Phi(t_k)=1$，用SGD逐步优化各时间步的系数
- LSD+：同时优化系数和步长 $\kappa_k$
- 距离度量 $d$ 使用标准L2范数

## 实验关键数据

### 主实验

文本生成（生成困惑度↓，SEDD-small backbone，1024 tokens × 1024 samples）：

| 采样器 | NFE=8 | NFE=16 | NFE=32 | NFE=64 |
|--------|-------|--------|--------|--------|
| Euler | 423.1 | 215.5 | 72.8 | 56.2 |
| Tweedie | 404.9 | 177.5 | 64.3 | 50.2 |
| JYS-Euler | 308.1 | 125.3 | 55.8 | 32.9 |
| LSD+-Euler | **128.4** | **51.8** | **36.8** | **20.7** |
| LSD+-Tweedie | 137.9 | 61.0 | 38.2 | 20.5 |

RADD backbone下LSD+的提升更显著：NFE=8时89.8 vs Euler的671.0，降低约87%。

### 消融实验

| 配置 | NFE=32 (困惑度) | 说明 |
|------|----------------|------|
| LSD+ w/o 松弛目标 | 34.98 | 严格匹配更难优化 |
| LSD+ w/ 松弛目标 | **33.23** | 松弛提升可行性 |

Hamming距离阈值消融（SEDD-small，Euler，32步）：

| 阈值 | 0% | 1% | 5% (选择) | 10% | 20% |
|------|----|----|-----------|-----|-----|
| 困惑度↓ | 35.98 | 32.15 | **31.24** | 39.97 | 51.52 |

5%为最优平衡点，过大或过小都不利。

### 关键发现

- LSD和LSD+在所有三种backbone（SEDD-small、SEDD-medium、RADD）和所有NFEs下均显著超越基线
- LSD+通常优于LSD，证实非均匀时间调度的价值
- 在8步（极低NFEs）时改善最为剧烈：LSD+-Euler比Euler降低困惑度约70%
- 图像生成和合成任务也验证了方法的通用性：CIFAR-10的FID和countdown任务的错误率均显著改善

## 亮点与洞察

- 核心insight（对齐中间分数轨迹而非最终输出）巧妙绕过了离散采样不可微的根本障碍
- 可学习系数赋予采样器自适应补偿误差的能力，是一种极为轻量但高效的增强方式
- 松弛目标的设计利用了Hamming距离作为离散空间的自然度量，概念优雅
- 训练成本极低（5分钟），且推理零开销，具有很高的实用性

## 局限与展望

- 目前学习的系数和时间调度是全局的（与具体输入无关），输入条件化的自适应可能进一步提升
- 松弛目标的Hamming距离阈值需要manual tuning
- 未提供师生采样器输出分布差异的理论保证
- 在更大规模的DDM模型上的表现有待验证

## 相关工作与启发

- 与连续扩散的LD3和S4S的思路关联，但解决了DDM特有的不可微挑战
- JYS仅优化"何时采样"，LSD同时优化"如何采样"（系数）和"何时采样"（时间调度）
- 对离散扩散模型加速领域，本文指出了中间轨迹对齐这个可能比最终样本匹配更可行的方向

## 评分

- **新颖性**: ⭐⭐⭐⭐ 中间分数轨迹对齐+可学习系数是DDM采样加速的全新范式
- **实验充分度**: ⭐⭐⭐⭐⭐ 三种backbone、三种任务（文本/图像/合成）、充分消融
- **写作质量**: ⭐⭐⭐⭐ 技术呈现清晰，与相关工作的区别阐述充分
- **价值**: ⭐⭐⭐⭐⭐ 训练成本极低、推理零开销、大幅提升低步数生成质量，实用性极强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Non-Markovian Discrete Diffusion with Causal Language Models](non-markovian_discrete_diffusion_with_causal_language_models.md)
- [\[NeurIPS 2025\] Beyond Masked and Unmasked: Discrete Diffusion Models via Partial Masking](beyond_masked_and_unmasked_discrete_diffusion_models_via_par.md)
- [\[ICML 2025\] Distillation of Discrete Diffusion through Dimensional Correlations (Di4C)](../../ICML2025/image_generation/distillation_of_discrete_diffusion_through_dimensional_correlations.md)
- [\[NeurIPS 2025\] Information-Theoretic Discrete Diffusion](information-theoretic_discrete_diffusion.md)
- [\[NeurIPS 2025\] Fast Solvers for Discrete Diffusion Models: Theory and Applications of High-Order Algorithms](fast_solvers_for_discrete_diffusion_models_theory_and_applications_of_high-order.md)

</div>

<!-- RELATED:END -->
