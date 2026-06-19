---
title: >-
  [论文解读] Quantum Doubly Stochastic Transformers
description: >-
  [NeurIPS 2025 Spotlight][物理/科学计算][量子变分电路] 提出QDSFormer（量子双随机Transformer），用变分量子电路QontOT替代softmax生成双随机注意力矩阵，理论和实验证明量子电路生成的DSM更多样、更好保持信息，在多个小规模视觉识别任务上一致超越标准ViT和Sinkformer。
tags:
  - "NeurIPS 2025 Spotlight"
  - "物理/科学计算"
  - "量子变分电路"
  - "双随机矩阵"
  - "注意力机制"
  - "ViT"
  - "Birkhoff多面体"
---

# Quantum Doubly Stochastic Transformers

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2504.16275](https://arxiv.org/abs/2504.16275)  
**代码**: 无  
**领域**: 量子计算 / Transformer  
**关键词**: 量子变分电路, 双随机矩阵, 注意力机制, ViT, Birkhoff多面体

## 一句话总结

提出QDSFormer（量子双随机Transformer），用变分量子电路QontOT替代softmax生成双随机注意力矩阵，理论和实验证明量子电路生成的DSM更多样、更好保持信息，在多个小规模视觉识别任务上一致超越标准ViT和Sinkformer。

## 研究背景与动机

- Transformer中softmax使注意力矩阵为右随机（行和=1），这导致多种训练不稳定性：
    - 熵坍缩：注意力过于尖锐导致梯度消失
    - 秩坍缩和token均匀化问题
    - Eureka时刻：组合问题中的突变学习
- Sinkformer发现注意力自然趋向双随机矩阵（行和列和均=1），强制双随机性可提升性能
- Sinkhorn算法的局限：
    - 迭代近似，实际中难以收敛到真正的DSM（$k=21$时Frobenius距离仍为0.23）
    - 非参数化，无法学习应该返回哪个DSM
    - 需要输入非负（通过指数化实现），损失了表达能力
    - 反向传播梯度可能病态
- 关键突破：QontOT量子电路证明可天然产生DSM（$\mathbf{U} \odot \bar{\mathbf{U}} \in \Omega_n$），且无已知经典参数化方法能做到相同的事

## 方法详解

### 整体框架

QDSFormer在ViT中用QontOT量子电路替代softmax，将注意力矩阵 $\mathbf{QK}^\top$ 输入量子电路得到双随机注意力矩阵。电路利用酉矩阵的Hadamard积天然产生DSM这一性质，通过参数化量子门实现灵活的DSM生成。

### 关键设计

1. **QontOT量子电路产生DSM**:
    - 功能：将未归一化的注意力矩阵映射为双随机矩阵
    - 核心思路：对任意酉矩阵 $\mathbf{U}$，$\mathbf{U} \odot \bar{\mathbf{U}} \in \Omega_n$。QontOT通过参数 $\theta$ 和数据 $\mathbf{M}$ 的乘积注入控制电路角度，产生参数化的DSM
    - 设计动机：量子电路天然保证DSM性质，且是参数化的（不同于非参数的Sinkhorn），可学习最优DSM

2. **表达能力分析**:
    - 功能：系统对比QontOT、Sinkhorn和QR分解在DSM多样性上的差异
    - 核心思路：在离散化超立方体上穷举输入，统计产生的唯一DSM数量
    - 关键结果：QontOT（8层）对每个输入产生唯一的DSM（近似单射），而Sinkhorn和QR有大量碰撞

3. **QR分解双随机算子（量子启发）**:
    - 功能：作为经典的量子启发替代方案
    - 核心思路：对 $\mathbf{M}$ 做QR分解得酉矩阵 $\mathbf{U}$，再计算 $\mathbf{U} \odot \bar{\mathbf{U}}$
    - 局限：$O(n^3)$ 复杂度，碰撞率高，但在某些单层ViT设置中表现不错

### 损失函数 / 训练策略

- 三种电路训练策略：
    - **Static**：使用从量子硬件实验获得的固定参数，无需训练
    - **Mixed**：每个epoch交替200步梯度无关优化
    - **Differentiable**：端到端联合训练（最慢，受Barren Plateaus影响）
- 静态策略表现最好或与优化版持平（可能因Barren Plateaus）
- 使用8×8注意力矩阵、16层电路、4个辅助量子比特（总16量子比特）

## 实验关键数据

### 主实验（表格）

2层ViT在FashionMNIST和MNIST上的验证准确率：

| 方法 | FashionMNIST | MNIST |
|------|-------------|-------|
| Softmax | 88.9 ± 0.1 | 98.1 ± 0.3 |
| Softmax_σ² | 84.6 ± 2.1 | 93.0 ± 4.6 |
| QR | 89.3 ± 0.1 | 98.3 ± 0.1 |
| Sinkhorn | 89.1 ± 0.7 | 98.2 ± 0.3 |
| **QontOT** | **90.0 ± 0.2** | **98.4 ± 0.1** |

MedMNIST（7个数据集）：QontOT在5/7个数据集上最优。

### 消融实验

- **电路层数**：4-8层开始超越ViT，更多层带来对数级提升，>16层后收益递减
- **静态 vs 优化**：静态配置表现等于甚至优于端到端优化，可能因Barren Plateaus
- **Eureka实验**：QDSFormer提前出现Eureka时刻（组合推理突变），训练更稳定
- **Sinkhorn迭代次数**：$k=3$时距Birkhoff多面体Frobenius距离0.84，$k=21$仍为0.23；QontOT < 5e-6

### 关键发现

- QontOT在43M个输入矩阵上产生了最多唯一DSM，行为接近单射（近乎无信息损失）
- QontOT天然具有更高的注意力熵，可缓解ViT训练中的熵坍缩问题
- Sinkhorn会将行常量矩阵（如 $\mathbf{e}_2\mathbf{1}^\top$ 和 $\mathbf{e}_4\mathbf{1}^\top$）映射到同一DSM，丢失信息
- 电路规模对数缩放 $O(\log_2(T))$，理论上对大序列友好

## 亮点与洞察

- 首次将量子计算的"DSM归纳偏置"用于Transformer，开辟了经典ML无法参数化达到的设计空间
- 表达能力分析严谨全面，穷举法+信息保持+熵分析三个维度
- 量子启发的QR分解方法作为经典替代方案，本身也有独立价值
- 静态电路即优于经典方法的发现简化了部署——无需量子-经典混合训练

## 局限与展望

- 实验限于小规模数据集（MNIST级别）和小模型（1-4层ViT），大规模验证缺失
- 量子电路仿真速度是瓶颈，当前无法在真实量子硬件上高效运行注意力计算
- 注意力矩阵大小需为2的幂（量子比特限制），需要padding
- 仅验证了encoder自注意力，decoder和交叉注意力的适用性未探索

## 相关工作与启发

- Sinkformer和ESPFormer/LOTFormer是经典双随机Transformer的代表
- QontOT的"$\mathbf{U} \odot \bar{\mathbf{U}}$天然为DSM"是独特的量子归纳偏置，无已知经典等价物
- 随着量子硬件发展，QDSFormer可能在更大规模上展现优势
- 启发了在神经网络中引入物理约束的新范式

## 评分

- ⭐⭐⭐⭐ — 理论新颖，量子-ML交叉有开创性，但受限于小规模验证和量子硬件成熟度

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Adaptive Stochastic Coefficients for Accelerating Diffusion Sampling](adaptive_stochastic_coefficients_for_accelerating_diffusion_sampling.md)
- [\[NeurIPS 2025\] Vision Transformers for Cosmological Fields: Application to Weak Lensing Mass Maps](vision_transformers_for_cosmological_fields_application_to_weak_lensing_mass_map.md)
- [\[NeurIPS 2025\] AstroCo: Self-Supervised Conformer-Style Transformers for Light-Curve Embeddings](astroco_self-supervised_conformer-style_transformers_for_light-curve_embeddings.md)
- [\[ICML 2026\] REX: A Family of Reversible Exponential Stochastic Runge-Kutta Solvers](../../ICML2026/physics/rex_a_family_of_reversible_exponential_stochastic_runge-kutta_solvers.md)
- [\[ICML 2026\] Distribution Transformers: Fast Approximate Bayesian Inference With On-The-Fly Prior Adaptation](../../ICML2026/physics/distribution_transformers_fast_approximate_bayesian_inference_with_on-the-fly_pr.md)

</div>

<!-- RELATED:END -->
